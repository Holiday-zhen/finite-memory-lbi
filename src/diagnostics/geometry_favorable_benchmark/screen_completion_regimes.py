"""Calibrate four predeclared matrix-completion missingness regimes."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

import screen_candidates as screen


exp = screen.exp
HERE = Path(__file__).resolve().parent
CALIBRATION_SEEDS = list(range(10))
PATTERNS = ["random", "column-block", "cross-block", "heterogeneous"]
OBSERVATIONS = [0.25, 0.38, 0.50]
TAU_RATIOS = [0.02, 0.05, 0.10, 0.20]


def make_mask(rng: np.random.Generator, pattern: str, nr: int, nc: int, observation: float) -> np.ndarray:
    if pattern == "random":
        return rng.random((nr, nc)) < observation
    if pattern == "column-block":
        mask = rng.random((nr, nc)) < observation
        start = int(rng.integers(4, nc - 7))
        mask[:, start : start + 5] = rng.random((nr, 5)) < 0.06
        return mask
    if pattern == "cross-block":
        mask = rng.random((nr, nc)) < observation
        row_start = int(rng.integers(3, nr - 6))
        col_start = int(rng.integers(3, nc - 6))
        mask[row_start : row_start + 4, :] = rng.random((4, nc)) < 0.08
        mask[:, col_start : col_start + 4] = rng.random((nr, 4)) < 0.08
        return mask
    row_profile = np.geomspace(0.12, 0.85, nr)
    col_profile = np.geomspace(0.85, 0.12, nc)
    probability = np.sqrt(row_profile[:, None] * col_profile[None, :])
    probability = np.clip(probability * observation / np.mean(probability), 0.03, 0.92)
    return rng.random((nr, nc)) < probability


def build_problem(pattern: str, seed: int, observation: float, tau_ratio: float) -> exp.Problem:
    pattern_index = PATTERNS.index(pattern)
    rng = np.random.default_rng(120000 + 1000 * pattern_index + seed)
    nr, nc, rank = 20, 20, 3
    left = rng.normal(size=(nr, rank))
    right = rng.normal(size=(nc, rank))
    truth = left @ right.T / np.sqrt(rank)
    mask = make_mask(rng, pattern, nr, nc, observation)
    observed = np.where(mask, truth, 0.0)

    def grad(x):
        matrix = x.reshape(nr, nc)
        return np.where(mask, matrix - observed, 0.0).reshape(-1)

    initial_gradient = grad(np.zeros(nr * nc)).reshape(nr, nc)
    tau = tau_ratio * float(np.linalg.norm(initial_gradient, 2))
    mirror = lambda s: exp.svt_flat(s, tau, (nr, nc))
    omega = lambda x: float(
        tau * np.linalg.norm(x.reshape(nr, nc), ord="nuc") + 0.5 * (x @ x)
    )
    omega_star = lambda s: float(0.5 * np.linalg.norm(mirror(s)) ** 2)
    problem = exp.Problem(
        f"{pattern} matrix completion",
        nr * nc,
        grad,
        1.0,
        tau,
        mirror,
        omega,
        omega_star,
        120,
    )
    problem.pattern = pattern
    problem.shape = (nr, nc)
    problem.truth = truth
    problem.mask = mask
    return problem


def main() -> None:
    rows = []
    details = []
    for pattern in PATTERNS:
        for observation in OBSERVATIONS:
            for tau_ratio in TAU_RATIOS:
                setting = f"observation={observation:g},tau_ratio={tau_ratio:g}"
                values_by_seed = []
                print(f"{pattern}: {setting}", flush=True)
                for seed in CALIBRATION_SEEDS:
                    values = screen.onecut_geometry(build_problem(pattern, seed, observation, tau_ratio))
                    values_by_seed.append(values)
                    details.extend([[pattern, setting, seed, *row] for row in values])
                rows.append(screen.summarise(pattern, setting, values_by_seed))

    with (HERE / "regime_screen_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (HERE / "regime_screen_detail.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "pattern", "setting", "seed", "iteration", "relative_gradient",
            "one_step", "max_old_cut_violation", "gain_lower_minus_one", "normal_novelty",
        ])
        writer.writerows(details)

    selected = {}
    for pattern in PATTERNS:
        candidates = [row for row in rows if row["family"] == pattern and row["passes"]]
        if not candidates:
            candidates = [row for row in rows if row["family"] == pattern]
        selected[pattern] = max(
            candidates,
            key=lambda row: (row["share_gain_gt_1e-3"], row["median_positive_gain"]),
        )
    with (HERE / "selected_regime_configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
    for pattern, row in selected.items():
        print(
            f"SELECTED {pattern}: {row['setting']} "
            f"share={row['share_gain_gt_1e-3']:.3f} passes={row['passes']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
