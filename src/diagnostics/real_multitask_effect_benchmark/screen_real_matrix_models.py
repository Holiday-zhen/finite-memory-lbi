"""Screen matrix-structured objectives built directly from genuine datasets."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes, load_digits
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "four_virtual_examples"
GEOMETRY = HERE.parent / "geometry_favorable_benchmark"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(GEOMETRY))

import run_virtual_examples as exp
from screen_candidates import onecut_geometry


FAMILIES = [
    "Diabetes robust matrix recovery",
    "Binary Digits pixel feasibility",
    "Digits matrix completion",
]
CALIBRATION_SEEDS = list(range(10))
PATTERNS = ["random", "column-block", "heterogeneous"]
OBSERVATIONS = [0.38, 0.50]
TAU_RATIOS = [0.10, 0.20, 0.35, 0.50, 0.80]
GAIN_THRESHOLD = 1.0e-3


def make_dataset_mask(rng, pattern, nr, nc, observation):
    if pattern == "random":
        return rng.random((nr, nc)) < observation
    if pattern == "column-block":
        mask = rng.random((nr, nc)) < observation
        width = min(5, max(2, nc // 3))
        start = int(rng.integers(0, nc - width + 1))
        mask[:, start : start + width] = rng.random((nr, width)) < 0.06
        return mask
    row_profile = np.geomspace(0.12, 0.85, nr)
    col_profile = np.geomspace(0.85, 0.12, nc)
    probability = np.sqrt(row_profile[:, None] * col_profile[None, :])
    probability = np.clip(probability * observation / np.mean(probability), 0.03, 0.92)
    return rng.random((nr, nc)) < probability


def build_problem(family, seed, pattern, observation, tau_ratio):
    family_index = FAMILIES.index(family)
    pattern_index = PATTERNS.index(pattern)
    rng = np.random.default_rng(710000 + 10000 * family_index + 1000 * pattern_index + seed)

    if family == "Diabetes robust matrix recovery":
        data = load_diabetes()
        chosen = rng.choice(len(data.data), 120, replace=False)
        features = StandardScaler().fit_transform(data.data[chosen])
        response = data.target[chosen]
        response = ((response - np.mean(response)) / np.std(response))[:, None]
        truth = np.column_stack([features, response])
    else:
        data = load_digits()
        if family == "Binary Digits pixel feasibility":
            eligible = np.flatnonzero((data.target == 0) | (data.target == 1))
        else:
            eligible = np.arange(len(data.data))
        chosen = rng.choice(eligible, 36, replace=False)
        truth = data.data[chosen] / 16.0

    nr, nc = truth.shape
    mask = make_dataset_mask(rng, pattern, nr, nc, observation)

    if family == "Diabetes robust matrix recovery":
        delta = 0.25

        def grad(x):
            residual = x.reshape(nr, nc) - truth
            score = residual / np.sqrt(1.0 + (residual / delta) ** 2)
            return np.where(mask, score, 0.0).reshape(-1)

        lipschitz = 1.0
    elif family == "Binary Digits pixel feasibility":
        half_width = 0.035
        lower = np.maximum(truth - half_width, 0.0)
        upper = np.minimum(truth + half_width, 1.0)

        def grad(x):
            matrix = x.reshape(nr, nc)
            above = np.maximum(matrix - upper, 0.0)
            below = np.maximum(lower - matrix, 0.0)
            return np.where(mask, above - below, 0.0).reshape(-1)

        lipschitz = 1.0
    else:
        def grad(x):
            residual = x.reshape(nr, nc) - truth
            return np.where(mask, residual, 0.0).reshape(-1)

        lipschitz = 1.0

    initial = grad(np.zeros(nr * nc)).reshape(nr, nc)
    tau = tau_ratio * float(np.linalg.norm(initial, 2))
    mirror = lambda s: exp.svt_flat(s, tau, (nr, nc))
    omega = lambda x: float(
        tau * np.linalg.norm(x.reshape(nr, nc), ord="nuc") + 0.5 * (x @ x)
    )
    omega_star = lambda s: float(0.5 * np.linalg.norm(mirror(s)) ** 2)
    problem = exp.Problem(
        family, nr * nc, grad, lipschitz, tau,
        mirror, omega, omega_star, 160,
    )
    problem.shape = (nr, nc)
    problem.truth = truth
    problem.mask = mask
    return problem


def summarize(family, setting, arrays):
    combined = np.vstack(arrays)
    gains = combined[:, 4]
    positive = gains > 0
    shares = [float(np.mean(array[:, 4] > GAIN_THRESHOLD)) for array in arrays]
    return {
        "family": family,
        "setting": setting,
        "observations": len(gains),
        "max_violation": float(np.max(combined[:, 3])),
        "max_gain": float(np.max(gains)),
        "median_positive_gain": float(np.median(gains[positive])) if np.any(positive) else 0.0,
        "share_gain_gt_1e-3": float(np.mean(gains > GAIN_THRESHOLD)),
        "median_seed_share": float(np.median(shares)),
        "seeds_passing_10pct": int(sum(share > 0.10 for share in shares)),
        "passes": bool(np.median(shares) > 0.10),
    }


def main():
    rows = []
    for family in FAMILIES:
        for pattern in PATTERNS:
            for observation in OBSERVATIONS:
                for tau_ratio in TAU_RATIOS:
                    arrays = [
                        onecut_geometry(build_problem(family, seed, pattern, observation, tau_ratio))
                        for seed in CALIBRATION_SEEDS
                    ]
                    setting = f"pattern={pattern},obs={observation:g},tau={tau_ratio:g}"
                    row = summarize(family, setting, arrays)
                    rows.append(row)
                    print(f"{family:34s} {setting:43s} median={row['median_seed_share']:.3f}", flush=True)
    with (HERE / "real_matrix_model_screen.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = []
    for family in FAMILIES:
        candidates = [row for row in rows if row["family"] == family]
        best = max(candidates, key=lambda row: (row["median_seed_share"], row["median_positive_gain"]))
        selected.append(best)
        print("BEST", best, flush=True)
    with (HERE / "selected_real_matrix_models.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)


if __name__ == "__main__":
    main()
