"""Tune capped projected-BB memory projections with algorithm-only timing.

The outer horizon and test problem stay fixed at 100 iterations on the
4000-by-12000 sparse least-squares instances.  Calibration uses disjoint
seeds 30--34; the selected configurations are then evaluated on seeds 0--19.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = (
    ROOT / "src" / "diagnostics"
    / "revision_expanded_20260904" / "engine"
)
sys.path.insert(0, str(ENGINE))
import experiments_adaptive_revision as ad
import experiments_bounded as base


OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
OUTER_ITERATIONS = 100
GRAD2_STOP_RATIO = 1e-20
ONE_CUT_TOL = 1e-13
CALIBRATION_SEEDS = range(30, 35)
TEST_SEEDS = range(20)
QP_GRID = ((2, 4), (3, 6), (4, 8))
INNER_GRID = (1, 2, 3, 5)
TRIGGER_GRID = (0.02, 0.05, 0.10)
METHODS = ("recent", "violated", "angle", "adaptive")


@dataclass(frozen=True)
class Config:
    method: str
    q: int
    p: int
    inner_max: int
    trigger: float = 0.05
    original_solver: bool = False

    @property
    def label(self) -> str:
        return f"{self.method}__q{self.q}p{self.p}__i{self.inner_max}__c{self.trigger:g}"


def capped_projected_bb(
    x_star, G, beta, tau, warm, max_inner, tol=1e-9, step0=None,
):
    """Warm-started projected gradient with safeguarded BB steps."""
    # Diagonal dual preconditioning: normalize every cut.  If
    # Gs=G/scale and alpha_s=scale*alpha, then Gs@alpha_s=G@alpha.
    scale = np.maximum(np.linalg.norm(G, axis=0), 1e-15)
    Gs = G / scale
    beta_s = beta / scale
    alpha = np.maximum(np.asarray(warm, dtype=float) * scale, 0.0)
    lipschitz = max(float(np.linalg.norm(Gs, 2) ** 2), 1e-16)
    safe_step = 1.0 / lipschitz
    step = (
        float(np.clip(step0, 1e-8 * safe_step, 1e6 * safe_step))
        if step0 is not None and np.isfinite(step0) else safe_step
    )
    phi, grad = ad.dual_phi_grad(alpha, x_star, Gs, beta_s, tau)
    evaluations = 1
    pgn = ad.projected_grad_norm(alpha, grad)
    iterations = 0
    for iteration in range(1, max_inner + 1):
        if pgn <= tol:
            break
        old_alpha, old_grad, old_phi = alpha.copy(), grad.copy(), phi
        local = step
        for _ in range(8):
            trial = np.maximum(old_alpha - local * old_grad, 0.0)
            phi_trial, grad_trial = ad.dual_phi_grad(
                trial, x_star, Gs, beta_s, tau
            )
            evaluations += 1
            displacement = trial - old_alpha
            if phi_trial <= old_phi + 1e-4 * float(old_grad @ displacement):
                break
            local *= 0.5
        alpha, phi, grad = trial, phi_trial, grad_trial
        iterations = iteration
        pgn = ad.projected_grad_norm(alpha, grad)
        s = alpha - old_alpha
        y = grad - old_grad
        sy = float(s @ y)
        step = (
            float(np.clip(float(s @ s) / sy, 1e-8 * safe_step, 1e6 * safe_step))
            if sy > 1e-24 else safe_step
        )
    return alpha / scale, {
        "nit": iterations,
        "nfev": evaluations,
        "pgn": pgn,
        "success": pgn <= tol,
        "next_step": step,
    }


def run_method(A, b, tau, L, config: Config | None, return_history=False):
    """Run the configured outer horizon; reported time excludes diagnostics."""
    n = A.shape[1]
    x = np.zeros(n)
    x_star = np.zeros(n)
    pool = []
    warm_by_cut = {}
    previous_current_alpha = 0.0
    bb_step = None
    atb = np.asarray(A.T @ b).reshape(-1)
    norm_atb = max(float(np.linalg.norm(atb)), 1e-16)
    initial_grad2 = max(float(atb @ atb), 1e-300)
    algorithm_time = 0.0
    normal_history = []
    objective_history = []
    time_history = []
    inner_iterations = []
    inner_evaluations = []
    inner_pgn = []
    memory_sizes = []

    for k in range(OUTER_ITERATIONS):
        tic = time.perf_counter()
        residual = np.asarray(A @ x - b).reshape(-1)
        a = np.asarray(A.T @ residual).reshape(-1)
        grad2 = float(a @ a)
        if grad2 <= GRAD2_STOP_RATIO * initial_grad2:
            algorithm_time += time.perf_counter() - tic
            break
        beta = float(a @ x - grad2 / L)
        current = {"id": k, "a": a.copy(), "beta": beta}

        if config is None:
            selected = [current]
            G = a.reshape(-1, 1)
            betas = np.asarray([beta])
            alpha, info = base.solve_one_cut_exact(
                x_star, a, beta, tau, L, tol=ONE_CUT_TOL, max_iter=100
            )
            pool = [current]
            warm_by_cut = {}
        else:
            pool.append(current)
            pool = ad._prune_pool(pool, config.p, x)
            method = ad.Method(
                config.method, "angle" if config.method == "adaptive" else config.method,
                qmax=config.q, pmax=config.p, solver="bb",
                adaptive=config.method == "adaptive", c_trig=config.trigger,
            )
            if config.method == "adaptive":
                one_alpha, _ = base.solve_one_cut_exact(
                    x_star, a, beta, tau, L, tol=ONE_CUT_TOL, max_iter=100
                )
                x_one = ad.soft_threshold(x_star - a * one_alpha[0], tau)
                one_step = ad.bregman(x_one, x, x_star, tau)
                old = [cut for cut in pool if cut["id"] != k]
                required = max(
                    old, key=lambda cut: ad._positive_violation(cut, x_one),
                    default=None,
                )
                delta = (
                    ad._positive_violation(required, x_one)
                    if required is not None else 0.0
                )
                gain = 0.5 * delta * delta / one_step if one_step > 1e-18 else 0.0
                selected = (
                    ad._select_angle(current, old, x_one, config.q, required=required)
                    if gain >= config.trigger and delta > 0.0 else [current]
                )
            else:
                selected = ad._select_fixed(method, pool, current, x)

            ids = [cut["id"] for cut in selected]
            G = np.column_stack([cut["a"] for cut in selected])
            betas = np.asarray([cut["beta"] for cut in selected])
            # Reuse multipliers by cut identity.  The newly generated cut has
            # no matching identity, so initialize it from the preceding
            # iteration's current-cut multiplier instead of zero.
            warm = np.asarray([
                warm_by_cut.get(
                    i,
                    0.0 if config.original_solver else (
                        previous_current_alpha if i == k else 0.0
                    ),
                )
                for i in ids
            ])
            if len(selected) == 1:
                alpha, one_info = base.solve_one_cut_exact(
                    x_star, a, beta, tau, L, tol=ONE_CUT_TOL, max_iter=100
                )
                info = {
                    "nit": int(one_info["nit"]), "nfev": int(one_info["nit"]),
                    "pgn": float(one_info["pgn"]), "success": True,
                }
            else:
                if config.original_solver:
                    alpha, info = ad._dual_solver(
                        x_star, G, betas, tau, solver="bb", warm=warm,
                        tol=1e-9, max_iter=config.inner_max,
                    )
                    info["nfev"] = info["nit"]
                else:
                    alpha, info = capped_projected_bb(
                        x_star, G, betas, tau, warm, config.inner_max,
                        step0=bb_step,
                    )
                    bb_step = info["next_step"]
            current_position = ids.index(k)
            previous_current_alpha = float(alpha[current_position])
            warm_by_cut = {
                i: float(value) for i, value in zip(ids, alpha) if value > 1e-14
            }
            pool = ad._prune_pool(pool, config.p, x)
            live = {cut["id"] for cut in pool}
            warm_by_cut = {i: value for i, value in warm_by_cut.items() if i in live}

        x_star = x_star - G @ alpha
        x = ad.soft_threshold(x_star, tau)
        algorithm_time += time.perf_counter() - tic

        # Diagnostics deliberately outside the algorithm timer.
        normal = float(
            np.linalg.norm(np.asarray(A.T @ (A @ x - b)).reshape(-1)) / norm_atb
        )
        normal_history.append(normal)
        objective_history.append(float(tau * np.sum(np.abs(x)) + 0.5 * (x @ x)))
        time_history.append(algorithm_time)
        inner_iterations.append(float(info["nit"]))
        inner_evaluations.append(float(info.get("nfev", info["nit"])))
        inner_pgn.append(float(info["pgn"]))
        memory_sizes.append(float(len(selected)))

    result = {
        "final_residual": normal_history[-1],
        "algorithm_time": algorithm_time,
        "mean_inner_iterations": float(np.mean(inner_iterations)),
        "mean_inner_evaluations": float(np.mean(inner_evaluations)),
        "max_pgn": float(np.max(inner_pgn)),
        "final_pgn": float(inner_pgn[-1]),
        "tail20_pgn_median": float(np.median(inner_pgn[-20:])),
        "inner_success_rate": float(np.mean(np.asarray(inner_pgn) <= 1e-9)),
        "mean_q": float(np.mean(memory_sizes)),
    }
    if return_history:
        result.update({
            "normal_history": np.asarray(normal_history),
            "objective_history": np.asarray(objective_history),
            "time_history": np.asarray(time_history),
            "inner_iterations_history": np.asarray(inner_iterations),
            "inner_evaluations_history": np.asarray(inner_evaluations),
            "inner_pgn_history": np.asarray(inner_pgn),
            "memory_size_history": np.asarray(memory_sizes),
        })
    return result


def make_problem(seed):
    A, b, tau = base.make_sparse_problem(10100 + 12000 + seed, 4000, 12000)
    return A, b, tau, ad.spectral_lipschitz(A)


def config_grid():
    for method in METHODS:
        triggers = TRIGGER_GRID if method == "adaptive" else (0.05,)
        for q, p in QP_GRID:
            for inner_max in INNER_GRID:
                for trigger in triggers:
                    yield Config(method, q, p, inner_max, trigger)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows, phase):
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[row["label"]].append(row)
    result = []
    for label, rows in grouped.items():
        first = rows[0]
        result.append({
            "phase": phase,
            "label": label,
            "method": first["method"],
            "q": first["q"],
            "p": first["p"],
            "inner_max": first["inner_max"],
            "trigger": first["trigger"],
            "instances": len(rows),
            "residual_median": float(np.median([r["final_residual"] for r in rows])),
            "time_median_s": float(np.median([r["algorithm_time"] for r in rows])),
            "mean_inner_iterations": float(np.mean([r["mean_inner_iterations"] for r in rows])),
            "mean_inner_evaluations": float(np.mean([r["mean_inner_evaluations"] for r in rows])),
            "inner_success_rate": float(np.mean([r["inner_success_rate"] for r in rows])),
            "max_pgn": float(np.max([r["max_pgn"] for r in rows])),
            "final_pgn_median": float(np.median([r["final_pgn"] for r in rows])),
            "tail20_pgn_median": float(np.median([r["tail20_pgn_median"] for r in rows])),
            "mean_q": float(np.mean([r["mean_q"] for r in rows])),
        })
    return result


def select_configs(summary):
    selected = []
    for method in METHODS:
        candidates = [row for row in summary if row["method"] == method]
        best_residual = min(row["residual_median"] for row in candidates)
        near_best = [
            row for row in candidates
            if row["residual_median"] <= 1.10 * best_residual
        ]
        selected.append(min(near_best, key=lambda row: row["time_median_s"]))
    return selected


def evaluate(seeds, configs, phase):
    rows = []
    for seed in seeds:
        A, b, tau, L = make_problem(seed)
        if phase == "test":
            baseline = run_method(A, b, tau, L, None)
            rows.append({
                "phase": phase, "seed": seed, "label": "one-cut",
                "method": "one-cut", "q": 1, "p": 1, "inner_max": 0,
                "trigger": 0.0, **baseline,
            })
        for index, config in enumerate(configs, start=1):
            print(
                f"{phase} seed={seed} {index}/{len(configs)} {config.label}",
                flush=True,
            )
            outcome = run_method(A, b, tau, L, config)
            rows.append({
                "phase": phase, "seed": seed, "label": config.label,
                "method": config.method, "q": config.q, "p": config.p,
                "inner_max": config.inner_max, "trigger": config.trigger,
                **outcome,
            })
    return rows


def main():
    with threadpool_limits(1):
        grid = list(config_grid())
        calibration_raw = evaluate(CALIBRATION_SEEDS, grid, "calibration")
        calibration_summary = summarize(calibration_raw, "calibration")
        selected_rows = select_configs(calibration_summary)
        selected = [
            Config(
                row["method"], int(row["q"]), int(row["p"]),
                int(row["inner_max"]), float(row["trigger"]),
            )
            for row in selected_rows
        ]
        test_raw = evaluate(TEST_SEEDS, selected, "test")
        test_summary = summarize(test_raw, "test")

    write_rows(OUT / "calibration_raw.csv", calibration_raw)
    write_rows(OUT / "calibration_summary.csv", calibration_summary)
    write_rows(OUT / "selected_configs.csv", selected_rows)
    write_rows(OUT / "test_raw.csv", test_raw)
    write_rows(OUT / "test_summary.csv", test_summary)
    print("SELECTED")
    for row in selected_rows:
        print(row)
    print("TEST")
    for row in test_summary:
        print(row)


if __name__ == "__main__":
    main()
