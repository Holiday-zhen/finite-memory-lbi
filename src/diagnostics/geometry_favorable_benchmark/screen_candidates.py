"""Screen reproducible geometry-favourable benchmark candidates.

Only the one-cut trajectory is used during calibration.  A candidate passes
when old cuts exclude the one-cut candidate often enough before numerical
convergence.  No five-method result is consulted while selecting a setting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "four_virtual_examples"
TARGETED = HERE.parent / "targeted_memory_examples"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(TARGETED))

import run_real_examples as real
import run_virtual_examples as exp
import matrix_completion_comparison as matrix_exp


CALIBRATION_SEEDS = [0, 1, 2]
MIRROR_CONDITIONS = [2.0, 5.0, 10.0, 30.0, 100.0]
MIRROR_RANKS = [4, 8, 16]
ADMISSION_GAIN = 1e-3
ADMISSION_SHARE = 0.10
NUMERICAL_STEP_FLOOR = 1e-14


BASE_BUILDERS = {
    "grouped pseudo-Huber": real.make_diabetes_huber,
    "boundary logistic": real.make_cancer_logistic,
    "alternating feasibility": real.make_digits_feasibility,
}

MATRIX_LOSS_FAMILIES = [
    "low-rank pseudo-Huber sensing",
    "low-rank logistic sensing",
    "low-rank convex feasibility",
]

ENTRYWISE_MATRIX_FAMILIES = [
    "robust pseudo-Huber completion",
    "binary logistic completion",
    "inequality matrix feasibility",
    "heteroscedastic weighted completion",
]


def coupled_quadratic_problem(base: exp.Problem, condition: float, rank: int, seed: int) -> exp.Problem:
    """Replace the separable mirror by a fixed low-rank coupled metric.

    M = I + (condition - 1) U U^T has lambda_min(M)=1, so grad(omega*)
    remains 1-Lipschitz and the existing dual projected-step safeguard remains
    valid.  The same metric is used by all five methods.
    """

    rng = np.random.default_rng(91000 + 101 * seed + base.dim)
    effective_rank = min(rank, base.dim)
    raw = rng.normal(size=(base.dim, effective_rank))
    u, _ = np.linalg.qr(raw, mode="reduced")
    shrink = 1.0 - 1.0 / condition

    def mirror(s):
        return s - shrink * u @ (u.T @ s)

    def omega(x):
        return float(0.5 * (x @ x + (condition - 1.0) * np.linalg.norm(u.T @ x) ** 2))

    def omega_star(s):
        return float(0.5 * s @ mirror(s))

    return exp.Problem(
        base.name,
        base.dim,
        base.grad,
        base.lipschitz,
        0.0,
        mirror,
        omega,
        omega_star,
        base.max_iter,
    )


def build_coupled_candidate(family: str, seed: int, condition: float, rank: int) -> exp.Problem:
    base = BASE_BUILDERS[family](seed)
    problem = coupled_quadratic_problem(base, condition, rank, seed)
    problem.name = family
    return problem


def group_sparse_problem(base: exp.Problem, group_size: int, tau_ratio: float, seed: int) -> exp.Problem:
    """Use a disjoint group-lasso plus quadratic Bregman function."""

    rng = np.random.default_rng(93000 + 103 * seed + base.dim)
    permutation = rng.permutation(base.dim)
    groups = [permutation[start : start + group_size] for start in range(0, base.dim, group_size)]
    initial_gradient = base.grad(np.zeros(base.dim))
    group_scale = max(float(np.linalg.norm(initial_gradient[group])) for group in groups)
    tau = tau_ratio * group_scale

    def mirror(s):
        x = np.zeros_like(s)
        for group in groups:
            norm = float(np.linalg.norm(s[group]))
            if norm > tau:
                x[group] = (1.0 - tau / norm) * s[group]
        return x

    def omega(x):
        penalty = sum(float(np.linalg.norm(x[group])) for group in groups)
        return float(tau * penalty + 0.5 * (x @ x))

    def omega_star(s):
        mapped = mirror(s)
        return float(0.5 * (mapped @ mapped))

    return exp.Problem(
        base.name,
        base.dim,
        base.grad,
        base.lipschitz,
        tau,
        mirror,
        omega,
        omega_star,
        base.max_iter,
    )


def build_group_candidate(family: str, seed: int, group_size: int, tau_ratio: float) -> exp.Problem:
    base = BASE_BUILDERS[family](seed)
    problem = group_sparse_problem(base, group_size, tau_ratio, seed)
    problem.name = family
    return problem


def build_matrix_candidate(seed: int, observation: float, tau_ratio: float) -> exp.Problem:
    old_observation = matrix_exp.OBSERVATION
    old_tau_ratio = matrix_exp.TAU_RATIO
    try:
        matrix_exp.OBSERVATION = observation
        matrix_exp.TAU_RATIO = tau_ratio
        problem = matrix_exp.make_matrix_completion(seed)
    finally:
        matrix_exp.OBSERVATION = old_observation
        matrix_exp.TAU_RATIO = old_tau_ratio
    problem.name = "structured matrix completion"
    return problem


def build_matrix_loss_candidate(family: str, seed: int, tau_ratio: float) -> exp.Problem:
    """Construct three smooth convex losses with a shared nuclear-norm mirror."""

    family_offset = MATRIX_LOSS_FAMILIES.index(family)
    rng = np.random.default_rng(95000 + 1000 * family_offset + seed)
    nr, nc, rank = 14, 14, 3
    dimension = nr * nc
    measurements = 150
    left = rng.normal(size=(nr, rank))
    right = rng.normal(size=(nc, rank))
    truth = (left @ right.T / np.sqrt(rank)).reshape(-1)
    operator = rng.normal(size=(measurements, dimension)) / np.sqrt(dimension)

    if family == "low-rank pseudo-Huber sensing":
        target = operator @ truth + 0.02 * rng.normal(size=measurements)
        outliers = rng.choice(measurements, 18, replace=False)
        target[outliers] += 1.5 * rng.normal(size=len(outliers))
        delta = 0.12

        def grad(x):
            residual = operator @ x - target
            score = residual / np.sqrt(1.0 + (residual / delta) ** 2)
            return operator.T @ score / measurements

        lipschitz = float(np.linalg.norm(operator, 2) ** 2 / measurements)
    elif family == "low-rank logistic sensing":
        latent = operator @ truth
        probability = 1.0 / (1.0 + np.exp(-np.clip(latent, -30, 30)))
        labels = np.where(rng.random(measurements) < probability, 1.0, -1.0)
        flip = rng.choice(measurements, 12, replace=False)
        labels[flip] *= -1.0

        def grad(x):
            score = operator @ x
            weight = -labels / (1.0 + np.exp(np.clip(labels * score, -50, 50)))
            return operator.T @ weight / measurements

        lipschitz = float(0.25 * np.linalg.norm(operator, 2) ** 2 / measurements)
    else:
        signed_projection = operator @ truth
        signs = np.where(signed_projection >= 0.0, 1.0, -1.0)
        constraint_operator = -signs[:, None] * operator
        constraint_beta = -0.80 * np.abs(signed_projection)

        def grad(x):
            violation = np.maximum(constraint_operator @ x - constraint_beta, 0.0)
            return constraint_operator.T @ violation / measurements

        lipschitz = float(np.linalg.norm(constraint_operator, 2) ** 2 / measurements)

    initial_matrix_gradient = grad(np.zeros(dimension)).reshape(nr, nc)
    tau = tau_ratio * float(np.linalg.norm(initial_matrix_gradient, 2))
    mirror = lambda s: exp.svt_flat(s, tau, (nr, nc))
    omega = lambda x: float(
        tau * np.linalg.norm(x.reshape(nr, nc), ord="nuc") + 0.5 * (x @ x)
    )
    omega_star = lambda s: float(0.5 * np.linalg.norm(mirror(s)) ** 2)
    problem = exp.Problem(
        family,
        dimension,
        grad,
        lipschitz,
        tau,
        mirror,
        omega,
        omega_star,
        150,
    )
    problem.shape = (nr, nc)
    problem.truth = truth.reshape(nr, nc)
    return problem


def build_entrywise_matrix_candidate(family: str, seed: int, observation: float, tau_ratio: float) -> exp.Problem:
    """Matrix problems with incomplete coordinate observations and distinct losses."""

    family_offset = ENTRYWISE_MATRIX_FAMILIES.index(family)
    rng = np.random.default_rng(98000 + 1000 * family_offset + seed)
    nr, nc, rank = 18, 18, 3
    left = rng.normal(size=(nr, rank))
    right = rng.normal(size=(nc, rank))
    truth = left @ right.T / np.sqrt(rank)
    mask = rng.random((nr, nc)) < observation
    block_start = int(rng.integers(4, 11))
    mask[:, block_start : block_start + 4] = False
    mask[:, block_start : block_start + 4] |= rng.random((nr, 4)) < 0.08

    if family == "robust pseudo-Huber completion":
        observed = truth + 0.01 * rng.normal(size=(nr, nc))
        observed_indices = np.argwhere(mask)
        outlier_count = max(1, len(observed_indices) // 10)
        for row, col in observed_indices[rng.choice(len(observed_indices), outlier_count, replace=False)]:
            observed[row, col] += 1.5 * rng.normal()
        delta = 0.12

        def grad(x):
            residual = x.reshape(nr, nc) - observed
            score = residual / np.sqrt(1.0 + (residual / delta) ** 2)
            return np.where(mask, score, 0.0).reshape(-1)

        lipschitz = 1.0
    elif family == "binary logistic completion":
        probability = 1.0 / (1.0 + np.exp(-np.clip(truth, -30, 30)))
        labels = np.where(rng.random((nr, nc)) < probability, 1.0, -1.0)

        def grad(x):
            score = x.reshape(nr, nc)
            weight = -labels / (1.0 + np.exp(np.clip(labels * score, -50, 50)))
            return np.where(mask, weight, 0.0).reshape(-1)

        lipschitz = 0.25
    elif family == "inequality matrix feasibility":
        signs = np.where(truth >= 0.0, 1.0, -1.0)
        margin = 0.80 * np.abs(truth)

        def grad(x):
            matrix = x.reshape(nr, nc)
            violation = np.maximum(margin - signs * matrix, 0.0)
            return np.where(mask, -signs * violation, 0.0).reshape(-1)

        lipschitz = 1.0
    else:
        observed = truth + 0.01 * rng.normal(size=(nr, nc))
        row_weight = np.geomspace(0.15, 1.0, nr)[:, None]
        column_weight = np.geomspace(1.0, 0.20, nc)[None, :]
        weights = np.sqrt(row_weight * column_weight)

        def grad(x):
            residual = x.reshape(nr, nc) - observed
            return np.where(mask, weights * residual, 0.0).reshape(-1)

        lipschitz = float(np.max(weights))

    initial_matrix_gradient = grad(np.zeros(nr * nc)).reshape(nr, nc)
    tau = tau_ratio * float(np.linalg.norm(initial_matrix_gradient, 2))
    mirror = lambda s: exp.svt_flat(s, tau, (nr, nc))
    omega = lambda x: float(
        tau * np.linalg.norm(x.reshape(nr, nc), ord="nuc") + 0.5 * (x @ x)
    )
    omega_star = lambda s: float(0.5 * np.linalg.norm(mirror(s)) ** 2)
    problem = exp.Problem(
        family,
        nr * nc,
        grad,
        lipschitz,
        tau,
        mirror,
        omega,
        omega_star,
        100,
    )
    problem.shape = (nr, nc)
    problem.truth = truth
    problem.mask = mask
    return problem


def onecut_geometry(problem: exp.Problem, pool_size: int = 12) -> np.ndarray:
    x = np.zeros(problem.dim)
    x_star = np.zeros(problem.dim)
    pool = []
    rows = []
    initial_norm = max(float(np.linalg.norm(problem.grad(x))), 1e-30)

    for iteration in range(problem.max_iter):
        raw = problem.grad(x)
        raw_norm = float(np.linalg.norm(raw))
        if raw_norm / initial_norm <= 1e-10:
            break
        raw_beta = float(raw @ x - (raw @ raw) / problem.lipschitz)
        scale = max(raw_norm, 1e-30)
        a = raw / scale
        beta = raw_beta / scale
        current = {"id": iteration, "a": a.copy(), "beta": beta}
        pool.append(current)
        pool = exp.prune_pool(pool, pool_size, x)

        alpha = exp.one_cut(problem, x_star, a, beta)
        candidate = problem.mirror(x_star - alpha[0] * a)
        one_step = problem.omega(candidate) - problem.omega(x) - x_star @ (candidate - x)
        old = [cut for cut in pool if cut["id"] != iteration]
        violations = np.asarray([exp.positive_violation(cut, candidate) for cut in old])
        if len(violations):
            index = int(np.argmax(violations))
            violation = float(violations[index])
            old_normal = old[index]["a"] / max(float(np.linalg.norm(old[index]["a"])), 1e-30)
            novelty = 1.0 - abs(float(old_normal @ a))
        else:
            violation = 0.0
            novelty = 0.0
        gain = 0.5 * violation**2 / one_step if one_step > NUMERICAL_STEP_FLOOR else 0.0
        rows.append([
            iteration + 1,
            raw_norm / initial_norm,
            one_step,
            violation,
            gain,
            novelty,
        ])
        x_star = x_star - alpha[0] * a
        x = candidate
        pool = exp.prune_pool(pool, pool_size, x)
    return np.asarray(rows, dtype=float)


def summarise(family: str, setting: str, seed_values: list[np.ndarray]) -> dict:
    values = np.vstack([value for value in seed_values if len(value)])
    gains = values[:, 4]
    violations = values[:, 3]
    positive = gains > 0
    return {
        "family": family,
        "setting": setting,
        "observations": len(values),
        "max_violation": float(np.max(violations)),
        "max_gain": float(np.max(gains)),
        "median_positive_gain": float(np.median(gains[positive])) if np.any(positive) else 0.0,
        "share_gain_gt_1e-4": float(np.mean(gains > 1e-4)),
        "share_gain_gt_1e-3": float(np.mean(gains > 1e-3)),
        "share_gain_gt_1e-2": float(np.mean(gains > 1e-2)),
        "median_novelty_positive": float(np.median(values[positive, 5])) if np.any(positive) else 0.0,
        "passes": bool(np.mean(gains > ADMISSION_GAIN) >= ADMISSION_SHARE),
    }


def choose_setting(rows: list[dict], family: str) -> dict:
    candidates = [row for row in rows if row["family"] == family and row["passes"]]
    if not candidates:
        candidates = [row for row in rows if row["family"] == family]
    # Prefer broad reproducible gain, then moderate rather than extreme settings.
    return max(candidates, key=lambda row: (row["share_gain_gt_1e-3"], row["median_positive_gain"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use a reduced calibration grid")
    args = parser.parse_args()
    conditions = [5.0, 30.0] if args.quick else MIRROR_CONDITIONS
    ranks = [8] if args.quick else MIRROR_RANKS

    rows: list[dict] = []
    detail_rows = []
    for family in BASE_BUILDERS:
        for condition in conditions:
            for rank in ranks:
                setting = f"mirror_condition={condition:g},mirror_rank={rank}"
                seed_values = []
                print(f"{family}: {setting}", flush=True)
                for seed in CALIBRATION_SEEDS:
                    values = onecut_geometry(build_coupled_candidate(family, seed, condition, rank))
                    seed_values.append(values)
                    detail_rows.extend([[family, setting, seed, *row] for row in values])
                rows.append(summarise(family, setting, seed_values))

        group_sizes = [8, 32] if args.quick else [4, 8, 16, 32, 64]
        tau_ratios = [0.05, 0.30] if args.quick else [0.01, 0.03, 0.10, 0.30, 1.00]
        for group_size in group_sizes:
            for tau_ratio in tau_ratios:
                setting = f"group_size={group_size},group_tau_ratio={tau_ratio:g}"
                seed_values = []
                print(f"{family}: {setting}", flush=True)
                for seed in CALIBRATION_SEEDS:
                    values = onecut_geometry(build_group_candidate(family, seed, group_size, tau_ratio))
                    seed_values.append(values)
                    detail_rows.extend([[family, setting, seed, *row] for row in values])
                rows.append(summarise(family, setting, seed_values))

    matrix_tau_ratios = [0.03, 0.20] if args.quick else [0.01, 0.03, 0.10, 0.20, 0.50]
    for family in MATRIX_LOSS_FAMILIES:
        for tau_ratio in matrix_tau_ratios:
            setting = f"nuclear_tau_ratio={tau_ratio:g}"
            seed_values = []
            print(f"{family}: {setting}", flush=True)
            for seed in CALIBRATION_SEEDS:
                values = onecut_geometry(build_matrix_loss_candidate(family, seed, tau_ratio))
                seed_values.append(values)
                detail_rows.extend([[family, setting, seed, *row] for row in values])
            rows.append(summarise(family, setting, seed_values))

    entry_observations = [0.25, 0.38] if args.quick else [0.20, 0.25, 0.30, 0.38]
    entry_tau_ratios = [0.02, 0.10] if args.quick else [0.01, 0.02, 0.05, 0.10, 0.20]
    for family in ENTRYWISE_MATRIX_FAMILIES:
        for observation in entry_observations:
            for tau_ratio in entry_tau_ratios:
                setting = f"observation={observation:g},nuclear_tau_ratio={tau_ratio:g}"
                seed_values = []
                print(f"{family}: {setting}", flush=True)
                for seed in CALIBRATION_SEEDS:
                    values = onecut_geometry(build_entrywise_matrix_candidate(family, seed, observation, tau_ratio))
                    seed_values.append(values)
                    detail_rows.extend([[family, setting, seed, *row] for row in values])
                rows.append(summarise(family, setting, seed_values))

    matrix_grid = [(0.25, 0.02), (0.25, 0.05), (0.38, 0.02), (0.38, 0.05)]
    if not args.quick:
        matrix_grid += [(0.20, 0.02), (0.30, 0.03)]
    family = "structured matrix completion"
    for observation, tau_ratio in matrix_grid:
        setting = f"observation={observation:g},tau_ratio={tau_ratio:g}"
        seed_values = []
        print(f"{family}: {setting}", flush=True)
        for seed in CALIBRATION_SEEDS:
            values = onecut_geometry(build_matrix_candidate(seed, observation, tau_ratio))
            seed_values.append(values)
            detail_rows.extend([[family, setting, seed, *row] for row in values])
        rows.append(summarise(family, setting, seed_values))

    fields = list(rows[0])
    with (HERE / "screen_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (HERE / "screen_detail.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "family", "setting", "seed", "iteration", "relative_gradient",
            "one_step", "max_old_cut_violation", "gain_lower_minus_one", "normal_novelty",
        ])
        writer.writerows(detail_rows)

    selection_families = [
        *BASE_BUILDERS,
        *MATRIX_LOSS_FAMILIES,
        *ENTRYWISE_MATRIX_FAMILIES,
        "structured matrix completion",
    ]
    selected = {family: choose_setting(rows, family) for family in selection_families}
    with (HERE / "selected_configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)
    for family, row in selected.items():
        print(
            f"SELECTED {family}: {row['setting']} "
            f"share={row['share_gain_gt_1e-3']:.3f} passes={row['passes']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
