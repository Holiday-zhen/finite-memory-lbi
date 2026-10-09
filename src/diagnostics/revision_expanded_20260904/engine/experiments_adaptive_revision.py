"""Adaptive finite-memory experiments for the JSC revision.

This file is deliberately separate from the submitted fixed-memory baseline.
It implements

* an exact scalar one-cut step;
* bounded recent, violated, and angle-ranked memory rules;
* projected-gradient, accelerated projected-gradient, projected-BB, and
  L-BFGS-B solvers for the small nonnegative dual;
* an adaptive rule triggered by the computable lower gain
  ``theta_lower - 1 >= c_trig``;
* calibration/test seed separation and fixed-work/fixed-accuracy reporting.

All algorithm timings include cut selection, candidate-pool maintenance, the
one-cut preview used by the adaptive rule, and the inner dual solve.  Data
generation, reference solves, plotting, and diagnostics are excluded.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-adaptive-revision"),
)

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

import experiments_bounded as base
from experiments_theory_aligned import (
    bregman,
    dual_phi_grad,
    projected_grad_norm,
    soft_threshold,
    solve_bilevel_reference,
    solve_dual,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "fig_adaptive_revision"
RESULTS = ROOT / "experiments" / "results_adaptive_revision"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "one-cut": "#0072B2",
    "recent": "#E69F00",
    "violated": "#009E73",
    "angle": "#CC79A7",
    "adaptive": "#D55E00",
}


@dataclass(frozen=True)
class Method:
    name: str
    rule: str
    qmax: int = 5
    pmax: int = 25
    solver: str = "lbfgsb"
    adaptive: bool = False
    c_trig: float = 0.01


def _asvec(value):
    return np.asarray(value, dtype=float).reshape(-1)


def spectral_lipschitz(A):
    """Deterministic estimate of ||A||_2^2 used by every reported run."""
    if sparse.issparse(A):
        v0 = np.ones(min(A.shape), dtype=float)
        sigma = float(svds(
            A, k=1, which="LM", return_singular_vectors=False,
            v0=v0, tol=1e-12,
        )[0])
        return sigma * sigma
    return float(np.linalg.norm(A, 2) ** 2)


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _dual_solver(
    x_star, G, beta, tau, *, solver="lbfgsb", warm=None,
    tol=1e-9, max_iter=600,
):
    """Solve the nonnegative memory dual with a selected first-order method."""
    q = len(beta)
    if warm is None or len(warm) != q:
        alpha = np.zeros(q)
    else:
        alpha = np.maximum(np.asarray(warm, dtype=float), 0.0)
    if solver == "lbfgsb":
        return solve_dual(
            x_star, G, beta, tau, warm=alpha,
            dual_tol=tol, max_iter=max_iter,
        )

    # grad Phi is ||G||_2^2-Lipschitz for the sparse elastic-net geometry.
    lipschitz = max(float(np.linalg.norm(G, 2) ** 2), 1e-16)
    safe_step = 1.0 / lipschitz
    phi, grad = dual_phi_grad(alpha, x_star, G, beta, tau)

    if solver == "pg":
        for it in range(max_iter):
            pgn = projected_grad_norm(alpha, grad)
            if pgn <= tol:
                break
            alpha = np.maximum(alpha - safe_step * grad, 0.0)
            phi, grad = dual_phi_grad(alpha, x_star, G, beta, tau)
        nit = it

    elif solver == "fista":
        y = alpha.copy()
        t = 1.0
        nit = 0
        for it in range(max_iter):
            phi_y, grad_y = dual_phi_grad(y, x_star, G, beta, tau)
            trial = np.maximum(y - safe_step * grad_y, 0.0)
            phi_trial, grad_trial = dual_phi_grad(
                trial, x_star, G, beta, tau
            )
            # Monotone restart avoids oscillations near active-set changes.
            if phi_trial > phi + 1e-14 * max(1.0, abs(phi)):
                y = alpha.copy()
                t = 1.0
                _, grad_y = dual_phi_grad(y, x_star, G, beta, tau)
                trial = np.maximum(y - safe_step * grad_y, 0.0)
                phi_trial, grad_trial = dual_phi_grad(
                    trial, x_star, G, beta, tau
                )
            alpha_old = alpha
            alpha, phi, grad = trial, phi_trial, grad_trial
            nit = it + 1
            if projected_grad_norm(alpha, grad) <= tol:
                break
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = alpha + ((t - 1.0) / t_new) * (alpha - alpha_old)
            t = t_new

    elif solver == "bb":
        step = safe_step
        nit = 0
        for it in range(max_iter):
            pgn = projected_grad_norm(alpha, grad)
            if pgn <= tol:
                break
            old_alpha, old_grad, old_phi = alpha.copy(), grad.copy(), phi
            local = step
            # Projected Armijo line search; all operations are q-dimensional
            # except the inexpensive G@alpha evaluation in Phi.
            for _ in range(25):
                trial = np.maximum(old_alpha - local * old_grad, 0.0)
                phi_trial, grad_trial = dual_phi_grad(
                    trial, x_star, G, beta, tau
                )
                displacement = trial - old_alpha
                if phi_trial <= old_phi + 1e-4 * float(old_grad @ displacement):
                    break
                local *= 0.5
            alpha, phi, grad = trial, phi_trial, grad_trial
            s = alpha - old_alpha
            y = grad - old_grad
            sy = float(s @ y)
            if sy > 1e-20:
                step = float(np.clip(float(s @ s) / sy, 1e-6 * safe_step, 1e3 * safe_step))
            else:
                step = safe_step
            nit = it + 1
    else:
        raise ValueError(f"unknown dual solver {solver}")

    _, grad = dual_phi_grad(alpha, x_star, G, beta, tau)
    pgn = projected_grad_norm(alpha, grad)
    return alpha, {
        "nit": int(nit), "pgn": pgn,
        "success": bool(pgn <= 20.0 * tol), "solver": solver,
    }


def _positive_violation(cut, point):
    norm_a = max(float(np.linalg.norm(cut["a"])), 1e-300)
    return max(float(cut["a"] @ point - cut["beta"]), 0.0) / norm_a


def _select_angle(current, candidates, point, qmax, required=None):
    """Greedily rank useful cuts, optionally retaining a certified cut first."""
    useful = [cut for cut in candidates if _positive_violation(cut, point) > 0.0]
    selected = [current]
    if required is not None and qmax > 1:
        match = next((cut for cut in useful if cut["id"] == required["id"]), None)
        if match is not None:
            selected.append(match)
            useful.remove(match)
    while useful and len(selected) < qmax:
        best = None
        best_score = -np.inf
        for cut in useful:
            a = cut["a"]
            na = max(float(np.linalg.norm(a)), 1e-300)
            max_cos = max(
                abs(float(a @ kept["a"])) /
                max(na * float(np.linalg.norm(kept["a"])), 1e-300)
                for kept in selected
            )
            score = _positive_violation(cut, point) * max(1.0 - max_cos, 1e-12)
            if score > best_score:
                best, best_score = cut, score
        selected.append(best)
        useful.remove(best)
    return selected


def _select_fixed(method, pool, current, x):
    old = [cut for cut in pool if cut["id"] != current["id"]]
    if method.qmax <= 1 or not old:
        return [current]
    if method.rule == "recent":
        recent = sorted(old, key=lambda cut: cut["id"], reverse=True)
        return list(reversed(recent[: method.qmax - 1])) + [current]
    useful = [cut for cut in old if _positive_violation(cut, x) > 0.0]
    if method.rule == "violated":
        useful.sort(key=lambda cut: _positive_violation(cut, x), reverse=True)
        return useful[: method.qmax - 1] + [current]
    if method.rule == "angle":
        return _select_angle(current, useful, x, method.qmax)
    raise ValueError(method.rule)


def _prune_pool(pool, pmax, x):
    if len(pool) <= pmax:
        return pool
    recent_count = max(1, pmax // 2)
    recent = sorted(pool, key=lambda cut: cut["id"], reverse=True)[:recent_count]
    ids = {cut["id"] for cut in recent}
    remaining = [cut for cut in pool if cut["id"] not in ids]
    informative = sorted(
        remaining, key=lambda cut: _positive_violation(cut, x), reverse=True
    )[: pmax - len(recent)]
    return sorted(recent + informative, key=lambda cut: cut["id"])


def run_method(A, b, tau, method, max_iter, *, L=None, xbar=None,
               track_theta=False, dual_tol=1e-9):
    """Run one method with a genuinely bounded candidate pool."""
    n = A.shape[1]
    L = spectral_lipschitz(A) if L is None else float(L)
    x = np.zeros(n)
    x_star = np.zeros(n)
    pool = []
    warm_by_cut = {}
    atb = _asvec(A.T @ b)
    norm_atb = max(float(np.linalg.norm(atb)), 1e-16)
    initial_grad2 = max(float(atb @ atb), 1e-300)
    initial_universal = max(initial_grad2 / (2.0 * L * L), 1e-300)
    energy0 = bregman(xbar, x, x_star, tau) if xbar is not None else np.nan
    keys = [
        "iter", "normal", "energy_rel", "runtime", "dual_nit", "pgn",
        "memory_size", "pool_size", "predicted_gain", "theta", "theta_lb",
        "triggered", "inner_success",
    ]
    hist = {key: [] for key in keys}
    if xbar is not None:
        hist['primal_trace'] = []
        hist['dual_trace'] = []
    started = time.perf_counter()

    for k in range(max_iter):
        residual = _asvec(A @ x - b)
        a = _asvec(A.T @ residual)
        grad2 = float(a @ a)
        if grad2 <= 1e-20 * initial_grad2:
            break
        beta = float(a @ x - grad2 / L)
        current = {"id": k, "a": a.copy(), "beta": beta}
        pool.append(current)
        # The pool capacity includes the current cut.  Prune before screening
        # so the number of candidates never exceeds pmax at selection time.
        pool = _prune_pool(pool, method.pmax, x)

        # The adaptive rule first computes the exact one-cut point.  This work
        # is part of its reported runtime and is reused if memory is not enabled.
        one_alpha = one_info = x_one = None
        one_step = np.nan
        predicted_gain = 0.0
        theta_lb = 1.0
        triggered = False
        if method.adaptive or track_theta:
            one_alpha, one_info = base.solve_one_cut_exact(
                x_star, a, beta, tau, L, tol=1e-13, max_iter=100
            )
            x_one = soft_threshold(x_star - a * one_alpha[0], tau)
            one_step = bregman(x_one, x, x_star, tau)

        if method.adaptive:
            old = [cut for cut in pool if cut["id"] != k]
            j_star = max(
                old, key=lambda cut: _positive_violation(cut, x_one),
                default=None,
            )
            delta = _positive_violation(j_star, x_one) if j_star is not None else 0.0
            if one_step > 1e-18:
                predicted_gain = 0.5 * delta * delta / one_step
            theta_lb = 1.0 + predicted_gain
            triggered = bool(
                method.qmax > 1 and predicted_gain >= method.c_trig and delta > 0.0
            )
            selected = (
                _select_angle(
                    current, old, x_one, method.qmax, required=j_star,
                )
                if triggered else [current]
            )
            # Numerical ties can leave no useful historical cut.
            triggered = len(selected) > 1
        else:
            selected = _select_fixed(method, pool, current, x)

        ids = [cut["id"] for cut in selected]
        G = np.column_stack([cut["a"] for cut in selected])
        betas = np.asarray([cut["beta"] for cut in selected])
        warm = np.asarray([warm_by_cut.get(i, 0.0) for i in ids])

        if len(selected) == 1:
            if one_alpha is None:
                alpha, info = base.solve_one_cut_exact(
                    x_star, a, beta, tau, L, tol=1e-13, max_iter=100
                )
            else:
                alpha, info = one_alpha, one_info
        else:
            alpha, info = _dual_solver(
                x_star, G, betas, tau, solver=method.solver,
                warm=warm, tol=dual_tol,
            )

        x_old, x_star_old = x, x_star
        x_star = x_star_old - G @ alpha
        x = soft_threshold(x_star, tau)
        step = bregman(x, x_old, x_star_old, tau)
        theta = np.nan
        if (
            (method.adaptive or track_theta)
            and one_step > 1e-12 * initial_universal
        ):
            theta = step / one_step
            if not method.adaptive:
                deltas = [_positive_violation(cut, x_one) for cut in selected]
                theta_lb = 1.0 + 0.5 * max(deltas, default=0.0) ** 2 / one_step

        warm_by_cut = {
            i: float(v) for i, v in zip(ids, alpha) if v > 1e-14
        }
        normal = float(np.linalg.norm(_asvec(A.T @ (A @ x - b))) / norm_atb)
        energy_rel = (
            bregman(xbar, x, x_star, tau) / max(energy0, 1e-300)
            if xbar is not None else np.nan
        )
        hist["iter"].append(k + 1)
        hist["normal"].append(normal)
        hist["energy_rel"].append(energy_rel)
        hist["runtime"].append(time.perf_counter() - started)
        hist["dual_nit"].append(info["nit"])
        hist["pgn"].append(info["pgn"])
        hist["memory_size"].append(len(selected))
        hist["predicted_gain"].append(predicted_gain)
        hist["theta"].append(theta)
        hist["theta_lb"].append(theta_lb)
        hist["triggered"].append(float(triggered))
        hist["inner_success"].append(float(info.get("success", True)))
        if xbar is not None:
            hist['primal_trace'].append(x.copy())
            hist['dual_trace'].append(x_star.copy())

        pool = _prune_pool(pool, method.pmax, x)
        live = {cut["id"] for cut in pool}
        warm_by_cut = {i: v for i, v in warm_by_cut.items() if i in live}
        hist["pool_size"].append(len(pool))

    return {key: np.asarray(value, dtype=float) for key, value in hist.items()}


def _stack(runs, key):
    width = max(len(run[key]) for run in runs)
    out = np.full((len(runs), width), np.nan)
    for row, run in enumerate(runs):
        out[row, :len(run[key])] = run[key]
    return out


def _persistent_time(run, threshold, window=5, key="normal"):
    values = run[key]
    for i in range(max(0, len(values) - window + 1)):
        if np.all(values[i:i + window] <= threshold):
            return float(run["runtime"][i])
    return np.nan


def _summary(runs, threshold=5e-4):
    final = np.asarray([run["normal"][-1] for run in runs])
    runtime = np.asarray([run["runtime"][-1] for run in runs])
    hits = np.asarray([_persistent_time(run, threshold) for run in runs])
    finite = hits[np.isfinite(hits)]
    return {
        "final_mean": float(np.mean(final)),
        "final_std": float(np.std(final, ddof=1)) if len(final) > 1 else 0.0,
        "time_mean": float(np.mean(runtime)),
        "time_std": float(np.std(runtime, ddof=1)) if len(runtime) > 1 else 0.0,
        "target_mean": float(np.mean(finite)) if len(finite) else np.nan,
        "target_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "success": int(len(finite)),
        "avg_q": float(np.mean([np.mean(run["memory_size"]) for run in runs])),
        "q1_share": float(np.mean([np.mean(run["memory_size"] == 1) for run in runs])),
        "inner_success": float(np.mean([np.mean(run["inner_success"]) for run in runs])),
        "avg_inner": float(np.mean([np.mean(run["dual_nit"]) for run in runs])),
    }


def _timed_run(A, b, tau, method, max_iter, *, repeats=3, **kwargs):
    """Return one deterministic trajectory with a median cumulative timer.

    The numerical path is repeated on the same instance.  We verify that the
    repeated paths agree, then replace the first path's cumulative runtime by
    the pointwise median.  This keeps the fixed-work data paired while reducing
    operating-system noise in the wall-clock comparison.
    """
    repeated = [
        run_method(A, b, tau, method, max_iter, **kwargs)
        for _ in range(repeats)
    ]
    reference = repeated[0]
    for candidate in repeated[1:]:
        if not np.allclose(
            reference["normal"], candidate["normal"], rtol=1e-10, atol=1e-13,
        ):
            raise RuntimeError("timing repeats produced different trajectories")
    timed = {key: value.copy() for key, value in reference.items()}
    timed["runtime"] = np.median(
        np.vstack([run["runtime"] for run in repeated]), axis=0,
    )
    return timed


def _time_curve(runs, key="normal", points=250):
    start = max(float(run["runtime"][0]) for run in runs)
    stop = min(float(run["runtime"][-1]) for run in runs)
    grid = np.linspace(start, stop, points)
    values = np.vstack([np.interp(grid, run["runtime"], run[key]) for run in runs])
    return grid, np.nanmean(values, axis=0)


def _log_pareto_choice(rows, residual_col=1, time_col=2):
    """Return the row nearest the normalized log residual--time ideal."""
    residuals = np.asarray([row[residual_col] for row in rows], dtype=float)
    runtimes = np.asarray([row[time_col] for row in rows], dtype=float)
    log_r = np.log(np.maximum(residuals, 1e-300))
    log_t = np.log(np.maximum(runtimes, 1e-300))
    rspan = max(float(np.ptp(log_r)), 1e-16)
    tspan = max(float(np.ptp(log_t)), 1e-16)
    score = ((log_r - np.min(log_r)) / rspan) ** 2
    score += ((log_t - np.min(log_t)) / tspan) ** 2
    return rows[int(np.argmin(score))]


def experiment_1_calibration():
    """Calibrate the solver, bounded memory sizes, and adaptive trigger."""
    solver_names = ["lbfgsb", "pg", "fista", "bb"]
    solver_runs = {solver: [] for solver in solver_names}
    # Rotate the execution order across seeds and exclude one warm-up path.
    # This avoids systematically favoring the solver that happens to run first.
    for seed in range(10):
        shift = seed % len(solver_names)
        order = solver_names[shift:] + solver_names[:shift]
        for solver in order:
            method = Method(f"angle-{solver}", "angle", solver=solver)
            A, b, tau = base.make_sparse_problem(7100 + seed, 300, 900)
            L = spectral_lipschitz(A)
            print(f"calibration solver={solver} seed={seed}", flush=True)
            run_method(A, b, tau, method, 70, L=L)
            solver_runs[solver].append(
                _timed_run(A, b, tau, method, 70, L=L, repeats=3)
            )

    solver_rows = []
    solver_time_std = []
    for solver in solver_names:
        runs = solver_runs[solver]
        s = _summary(runs, threshold=1e-3)
        solver_rows.append([
            solver, s["final_mean"], s["time_mean"], s["target_mean"],
            s["success"], np.mean([np.mean(r["dual_nit"]) for r in runs]),
            s["inner_success"],
        ])
        solver_time_std.append(s["time_std"])

    # All candidates must satisfy the same inner KKT diagnostic.  Among those
    # that reach the outer target on every calibration seed, select by
    # time-to-target; this avoids rewarding oversolving at fixed outer work.
    feasible = [
        row for row in solver_rows
        if row[4] == 10 and row[6] >= 0.99 and np.isfinite(row[3])
    ]
    chosen_solver = min(feasible or solver_rows, key=lambda row: row[3])[0]

    # Select q at a fixed, deliberately generous pool capacity.  Then select
    # the smallest useful pool scale for the chosen q.  These seeds are used
    # only for calibration and never appear in the test experiments.
    q_rows = []
    q_time_std = []
    for qmax in [1, 3, 5, 10]:
        method = Method(
            f"angle q={qmax}", "angle", qmax=qmax, pmax=25,
            solver=chosen_solver,
        )
        runs = []
        for seed in range(10):
            A, b, tau = base.make_sparse_problem(7150 + seed, 300, 900)
            L = spectral_lipschitz(A)
            print(f"calibration q={qmax} seed={seed}", flush=True)
            runs.append(_timed_run(A, b, tau, method, 80, L=L, repeats=3))
        s = _summary(runs, threshold=8e-4)
        q_rows.append([qmax, s["final_mean"], s["time_mean"], s["avg_q"]])
        q_time_std.append(s["time_std"])
    best_q_residual = min(row[1] for row in q_rows)
    q_near_best = [row for row in q_rows if row[1] <= 1.05 * best_q_residual]
    chosen_q = int(min(q_near_best, key=lambda row: row[0])[0])

    # The pool must contain candidates that are not already in the working
    # set; p=q leaves no screening buffer and is not a meaningful memory-pool
    # configuration.  We therefore calibrate p >= 2q.
    p_grid = sorted(set([
        max(2 * chosen_q, 10), max(2 * chosen_q, 25),
        max(2 * chosen_q, 50),
    ]))
    p_rows = []
    p_time_std = []
    for pmax in p_grid:
        method = Method(
            f"angle p={pmax}", "angle", qmax=chosen_q, pmax=pmax,
            solver=chosen_solver,
        )
        runs = []
        for seed in range(10):
            A, b, tau = base.make_sparse_problem(7150 + seed, 300, 900)
            L = spectral_lipschitz(A)
            print(f"calibration p={pmax} seed={seed}", flush=True)
            runs.append(_timed_run(A, b, tau, method, 80, L=L, repeats=3))
        s = _summary(runs, threshold=8e-4)
        p_rows.append([pmax, s["final_mean"], s["time_mean"], s["avg_q"]])
        p_time_std.append(s["time_std"])
    best_p_residual = min(row[1] for row in p_rows)
    p_near_best = [row for row in p_rows if row[1] <= 1.005 * best_p_residual]
    chosen_p = int(min(p_near_best, key=lambda row: row[0])[0])

    c_grid = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
    c_rows = []
    c_runs = {}
    for c in c_grid:
        method = Method(
            f"adaptive c={c:g}", "angle", solver=chosen_solver,
            qmax=chosen_q, pmax=chosen_p, adaptive=True, c_trig=c,
        )
        runs = []
        for seed in range(10):
            A, b, tau = base.make_sparse_problem(7150 + seed, 300, 900)
            L = spectral_lipschitz(A)
            print(f"calibration c={c:g} seed={seed}", flush=True)
            runs.append(_timed_run(A, b, tau, method, 90, L=L, repeats=3))
        c_runs[c] = runs
        s = _summary(runs, threshold=8e-4)
        c_rows.append([
            c, s["final_mean"], s["time_mean"], s["target_mean"],
            s["success"], s["avg_q"], s["q1_share"],
        ])

    fully_successful = [row for row in c_rows if row[4] == 10 and np.isfinite(row[3])]
    if fully_successful:
        # A coarse target alone would always select the degenerate q_k=1 rule.
        # Select the knee of the calibration Pareto curve instead: normalize
        # log(final residual) and log(mean working-set size) to [0,1] and choose
        # the point
        # closest to the unattainable lower-left ideal.  Equal log weights make
        # the rule scale-free and avoid an arbitrary residual unit.
        residuals = np.asarray([row[1] for row in fully_successful], dtype=float)
        # Mean working-set size is a deterministic proxy for trigger cost and
        # avoids selecting c from millisecond-level wall-clock fluctuations.
        work = np.asarray([row[5] for row in fully_successful], dtype=float)
        log_r = np.log(np.maximum(residuals, 1e-300))
        log_t = np.log(np.maximum(work, 1e-300))
        scores = ((log_r - np.min(log_r)) / max(float(np.ptp(log_r)), 1e-16)) ** 2
        scores += ((log_t - np.min(log_t)) / max(float(np.ptp(log_t)), 1e-16)) ** 2
        near_knee = [
            row for row, score in zip(fully_successful, scores)
            if score <= 1.02 ** 2 * float(np.min(scores))
        ]
        # Within a two-percent neighborhood, prefer the smaller trigger, which
        # gives the more accurate member of the nearly identical knee set.
        chosen_c = min(near_knee, key=lambda row: row[0])[0]
    else:
        chosen_c = min(c_rows, key=lambda row: (row[1], row[2]))[0]

    _write_csv(
        RESULTS / "exp1_solver_calibration.csv",
        ["solver", "final_residual", "runtime", "time_to_1e-3",
         "successes", "avg_inner_iterations", "inner_success_rate"],
        solver_rows,
    )
    _write_csv(
        RESULTS / "exp1_q_calibration.csv",
        ["qmax", "final_residual", "runtime", "average_q"], q_rows,
    )
    _write_csv(
        RESULTS / "exp1_p_calibration.csv",
        ["pmax", "final_residual", "runtime", "average_q"], p_rows,
    )
    _write_csv(
        RESULTS / "exp1_trigger_calibration.csv",
        ["c_trig", "final_residual", "runtime", "time_to_8e-4",
         "successes", "average_q", "q1_share"], c_rows,
    )
    selection = {
        "solver": chosen_solver, "c_trig": chosen_c,
        "qmax": chosen_q, "pmax": chosen_p,
    }
    with open(RESULTS / "selected_configuration.json", "w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)

    # Four panels follow the sequential calibration: solver, q, p, then c.
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.25))
    axes[0, 0].bar(
        [row[0] for row in solver_rows], [row[2] for row in solver_rows],
        yerr=solver_time_std, capsize=2.5, color="#7AA6C2",
    )
    axes[0, 0].set_ylabel("mean runtime (s)")
    axes[0, 0].set_title("inner solver")

    axes[0, 1].errorbar(
        [row[2] for row in q_rows], [row[1] for row in q_rows],
        xerr=q_time_std, fmt="o-", capsize=2.5, color=COLORS["angle"],
    )
    q_offsets = {1: (3, 3), 3: (3, 3), 5: (-22, -12), 10: (3, 5)}
    for row in q_rows:
        axes[0, 1].annotate(
            f"q={int(row[0])}", (row[2], row[1]),
            xytext=q_offsets[int(row[0])], textcoords="offset points",
            fontsize=6.5,
        )
    chosen_q_row = next(row for row in q_rows if int(row[0]) == chosen_q)
    axes[0, 1].scatter([chosen_q_row[2]], [chosen_q_row[1]], s=58,
                       facecolors="none", edgecolors="#222222", linewidths=1.0, zorder=5)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("runtime (s)")
    axes[0, 1].set_ylabel("final residual")
    axes[0, 1].set_title("working-set cap $q$")

    p_relative = [row[1] / best_p_residual for row in p_rows]
    axes[1, 0].errorbar(
        [row[2] for row in p_rows], p_relative, xerr=p_time_std,
        fmt="s--", capsize=2.5, color=COLORS["violated"],
    )
    p_offsets = {
        int(row[0]): offset for row, offset in zip(
            p_rows, [(-22, -14), (-8, 10), (5, -11)]
        )
    }
    for row, rel in zip(p_rows, p_relative):
        axes[1, 0].annotate(
            f"p={int(row[0])}", (row[2], rel),
            xytext=p_offsets[int(row[0])],
            textcoords="offset points", fontsize=6.5,
        )
    chosen_p_row = next(row for row in p_rows if int(row[0]) == chosen_p)
    axes[1, 0].scatter([chosen_p_row[2]], [chosen_p_row[1] / best_p_residual], s=58,
                       facecolors="none", edgecolors="#222222", linewidths=1.0, zorder=5)
    axes[1, 0].axhline(1.005, color="#777777", linestyle=":", linewidth=0.8)
    axes[1, 0].set_xlabel("runtime (s)")
    axes[1, 0].set_ylabel("residual / best residual")
    axes[1, 0].set_title(f"pool cap $p$ at $q={chosen_q}$")

    axes[1, 1].plot([row[5] for row in c_rows], [row[1] for row in c_rows],
                    "o-", color=COLORS["adaptive"])
    chosen_c_row = next(row for row in c_rows if row[0] == chosen_c)
    axes[1, 1].scatter([chosen_c_row[5]], [chosen_c_row[1]], s=58, facecolors="none", edgecolors="#222222", linewidths=1.0, zorder=5)
    axes[1, 1].annotate(
        rf"$c_{{\rm trig}}={chosen_c:g}$", (chosen_c_row[5], chosen_c_row[1]),
        xytext=(5, 5), textcoords="offset points", fontsize=6.5,
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel(r"mean working-set size $q_k$")
    axes[1, 1].set_ylabel("final residual")
    axes[1, 1].set_title(r"trigger $c_{\rm trig}$")
    for ax in axes.flat:
        ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "exp1_calibration.pdf", bbox_inches="tight")
    plt.close(fig)
    return selection


def _comparison_methods(selection):
    solver = selection["solver"]
    c = float(selection["c_trig"])
    qmax = int(selection["qmax"])
    pmax = int(selection["pmax"])
    return [
        Method("one-cut", "recent", qmax=1, pmax=1, solver=solver),
        Method("recent", "recent", qmax=qmax, pmax=pmax, solver=solver),
        Method("violated", "violated", qmax=qmax, pmax=pmax, solver=solver),
        Method("angle", "angle", qmax=qmax, pmax=pmax, solver=solver),
        Method(
            "adaptive", "angle", qmax=qmax, pmax=pmax, solver=solver,
            adaptive=True, c_trig=c,
        ),
    ]


def experiment_2_main(selection, seeds=range(5)):
    methods = _comparison_methods(selection)
    groups = {method.name: [] for method in methods}
    for seed in seeds:
        A, b, tau = base.make_sparse_problem(8100 + seed, 500, 1500)
        L = spectral_lipschitz(A)
        for method in methods:
            print(f"main seed={seed} method={method.name}", flush=True)
            groups[method.name].append(
                _timed_run(A, b, tau, method, 100, L=L, repeats=3)
            )

    rows = []
    for method in methods:
        s = _summary(groups[method.name])
        strict_hits = np.asarray([
            _persistent_time(run, 1e-4) for run in groups[method.name]
        ])
        strict_finite = strict_hits[np.isfinite(strict_hits)]
        rows.append([
            method.name, s["final_mean"], s["final_std"], s["time_mean"],
            s["time_std"], s["target_mean"], s["success"], s["avg_q"],
            s["q1_share"],
            s["avg_inner"],
            (float(np.mean(strict_finite)) if len(strict_finite) else np.nan),
            int(len(strict_finite)),
        ])
    _write_csv(
        RESULTS / "exp2_main_comparison.csv",
        ["method", "final_mean", "final_std", "runtime_mean", "runtime_std",
         "time_to_5e-4", "successes", "average_q", "q1_share",
         "avg_inner_iterations", "time_to_1e-4", "strict_successes"], rows,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.25))
    styles = {"one-cut": "-", "recent": "--", "violated": "-.", "angle": ":", "adaptive": "-"}
    for method in methods:
        runs = groups[method.name]
        values = _stack(runs, "normal")
        mean = np.nanmean(values, axis=0)
        x = np.arange(1, len(mean) + 1)
        axes[0].semilogy(x, mean, styles[method.name], color=COLORS[method.name], label=method.name, linewidth=1.7)
        axes[0].fill_between(
            x, np.nanpercentile(values, 25, axis=0),
            np.nanpercentile(values, 75, axis=0),
            color=COLORS[method.name], alpha=0.08, linewidth=0,
        )
        grid, curve = _time_curve(runs)
        axes[1].semilogy(grid, curve, styles[method.name], color=COLORS[method.name], label=method.name, linewidth=1.7)
    axes[0].set_xlabel("iteration")
    axes[1].set_xlabel("elapsed algorithm time (s)")
    for ax in axes:
        ax.set_ylabel("normalized normal residual")
        ax.grid(True, which="major", color="#D3D3D3", linewidth=0.65)
    axes[0].legend(fontsize=7.2)
    fig.tight_layout()
    fig.savefig(OUT / "exp2_main_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    return groups


def _tail_factor(run, start=25, stop=55):
    e = np.asarray(run["energy_rel"])
    stop = min(stop, len(e))
    ratios = e[start:stop] / np.maximum(e[start - 1:stop - 1], 1e-300)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    return float(np.median(ratios)) if len(ratios) else np.nan


def experiment_3_energy_conditioning(selection):
    methods = _comparison_methods(selection)
    groups = {"sparse": {m.name: [] for m in methods}, "k10": {m.name: [] for m in methods}, "k1000": {m.name: [] for m in methods}}
    for seed in range(8):
        A, b, _, tau = base.make_gaussian_problem(9100 + seed, 60, 180, sparsity=12)
        xbar, _ = solve_bilevel_reference(A, b, tau)
        L = spectral_lipschitz(A)
        for method in methods:
            print(f"energy sparse seed={seed} method={method.name}", flush=True)
            groups["sparse"][method.name].append(run_method(
                A, b, tau, method, 75, L=L, xbar=xbar,
                track_theta=(method.name in {"violated", "angle"}),
                dual_tol=(1e-12 if method.name in {"violated", "angle"} else 1e-9),
            ))
    for label, condition in [("k10", 10.0), ("k1000", 1000.0)]:
        for seed in range(5):
            A, b, _, tau = base.make_spectral_problem(9200 + seed, 55, 150, rank=45, condition=condition, sparsity=10)
            xbar, _ = solve_bilevel_reference(A, b, tau)
            L = spectral_lipschitz(A)
            for method in methods:
                print(f"energy {label} seed={seed} method={method.name}", flush=True)
                groups[label][method.name].append(run_method(A, b, tau, method, 65, L=L, xbar=xbar))

    rows = []
    for label in groups:
        for method in methods:
            runs = groups[label][method.name]
            final = np.asarray([run["energy_rel"][-1] for run in runs])
            tails = np.asarray([_tail_factor(run) for run in runs])
            if method.name == "one-cut":
                log_gain = np.zeros_like(final)
            else:
                one_final = np.asarray([
                    run["energy_rel"][-1]
                    for run in groups[label]["one-cut"]
                ])
                log_gain = np.log10(np.maximum(one_final, 1e-300))
                log_gain -= np.log10(np.maximum(final, 1e-300))
            rows.append([
                label, method.name, np.mean(final), np.std(final, ddof=1),
                np.nanmean(tails), np.nanstd(tails, ddof=1),
                np.nanmedian(log_gain), np.nanpercentile(log_gain, 25),
                np.nanpercentile(log_gain, 75), len(final),
            ])
    _write_csv(
        RESULTS / "exp3_energy_conditioning.csv",
        ["problem", "method", "final_energy_mean", "final_energy_std",
         "tail_factor_mean", "tail_factor_std", "paired_log10_gain_median",
         "paired_log10_gain_q25", "paired_log10_gain_q75", "seeds"], rows,
    )
    # A single late-stage semilog plot shows both geometric decay and the
    # genuine separation between methods without introducing a second gain
    # transformation.  Curves are paired-seed medians with IQR bands.
    fig, ax = plt.subplots(figsize=(6.55, 3.45))
    one = _stack(groups["sparse"]["one-cut"], "energy_rel")
    x = np.arange(1, one.shape[1] + 1)
    abs_styles = {"one-cut": "-", "recent": "--", "violated": "-.", "angle": ":", "adaptive": "-"}
    visible_values = []
    for method in methods:
        energy = _stack(groups["sparse"][method.name], "energy_rel")
        median = np.nanmedian(energy, axis=0)
        q25 = np.nanpercentile(energy, 25, axis=0)
        q75 = np.nanpercentile(energy, 75, axis=0)
        ax.semilogy(
            x, median, abs_styles[method.name], color=COLORS[method.name],
            linewidth=(2.0 if method.name == "one-cut" else 1.55),
            label=method.name,
            marker={"one-cut": None, "recent": "o", "violated": "s", "angle": "^", "adaptive": "D"}[method.name],
            markevery=7, markersize=3.0,
        )
        ax.fill_between(
            x, np.maximum(q25, 1e-300), np.maximum(q75, 1e-300),
            color=COLORS[method.name], alpha=0.055, linewidth=0,
        )
        visible_values.extend(median[39:].tolist())
    positive = np.asarray([v for v in visible_values if np.isfinite(v) and v > 0])
    ax.set_xlim(40, 75)
    if positive.size:
        ax.set_ylim(0.65 * np.min(positive), 1.6 * np.max(positive))
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"relative Bregman energy $\mathcal{E}_k/\mathcal{E}_0$")
    ax.set_title("late-stage Bregman-energy decay", fontsize=9)
    ax.legend(fontsize=7.0, ncol=2)
    ax.grid(True, which="major", color="#D3D3D3", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "exp3_energy.pdf", bbox_inches="tight")
    plt.close(fig)
    return groups


def experiment_4_theta(energy_groups):
    """Validate the gain bound with the fixed violated rule."""
    fig, ax = plt.subplots(figsize=(6.15, 3.25))
    csv_rows = []
    audit_rows = []
    for name in ["violated"]:
        runs = energy_groups["sparse"][name]
        stop = 55
        theta = _stack(runs, "theta")[:, :stop]
        lower = _stack(runs, "theta_lb")[:, :stop]
        observed = np.nanmean(theta, axis=0)
        certified = np.nanmean(lower, axis=0)
        margin = theta - lower
        finite = np.isfinite(margin)
        min_margin = float(np.min(margin[finite])) if np.any(finite) else np.nan
        violations = int(np.sum(margin[finite] < -1e-9)) if np.any(finite) else 0
        audit_rows.append([name, int(np.sum(finite)), min_margin, violations])
        iterations = np.arange(1, observed.size + 1)
        active = np.flatnonzero(
            np.maximum(np.abs(observed - 1.0), np.abs(certified - 1.0)) > 1e-8
        )
        if active.size:
            left = max(int(active[0]) - 3, 0)
            right = min(int(active[-1]) + 4, observed.size)
        else:
            left, right = 0, observed.size
        sl = slice(left, right)
        ax.plot(
            iterations[sl], observed[sl], "o-", color=COLORS[name],
            linewidth=1.7, markersize=3.2, label=r"observed $\theta_k$",
        )
        ax.plot(
            iterations[sl], certified[sl], "s--", color="#D55E00",
            linewidth=1.45, markersize=2.8,
            label=r"lower bound $\theta_k^{\rm lb}$",
        )
        ax.axhline(1.0, color="#333333", linewidth=0.9, label="one-cut baseline")
        ax.set_xlabel("iteration")
        ax.set_title(f"fixed {name}", fontsize=9)
        ax.grid(True, color="#D3D3D3", linewidth=0.6)
        ax.legend(fontsize=6.5, loc="upper left")
        for k, obs, lb in zip(iterations, observed, certified):
            csv_rows.append([name, k, obs, lb])
    ax.set_ylabel("memory-gain factor")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(0.9995, ymin), max(1.001, ymax))
    fig.tight_layout()
    fig.savefig(OUT / "exp4_theta_fixed.pdf", bbox_inches="tight")
    plt.close(fig)
    _write_csv(
        RESULTS / "exp4_theta_fixed.csv",
        ["method", "iteration", "observed_theta", "lower_theta"],
        csv_rows,
    )
    _write_csv(
        RESULTS / "exp4_theta_audit.csv",
        ["method", "finite_seed_iterations", "minimum_theta_minus_bound",
         "violations_below_minus_1e-9"], audit_rows,
    )


def experiment_5_certificate():
    """Run the certificate study and report its numerical cost explicitly."""
    old_out, old_results = base.OUT, base.RESULTS
    base.OUT, base.RESULTS = OUT, RESULTS
    try:
        data = base.run_experiment_5(seeds=range(5))
    finally:
        base.OUT, base.RESULTS = old_out, old_results
    rows = []
    certified_runs = data["certified"]
    for name, runs in data.items():
        final = np.asarray([run["energy_rel"][-1] for run in runs])
        runtime = np.asarray([run["runtime"][-1] for run in runs])
        inner = np.asarray([np.mean(run["dual_nit"]) for run in runs])
        hits = np.asarray([
            _persistent_time(run, 1e-5, key="energy_rel") for run in runs
        ])
        finite_hits = hits[np.isfinite(hits)]
        violations = []
        for index, run in enumerate(runs):
            eta = certified_runs[index]["eta"][:len(run["eps_hat"])]
            violations.append(np.sum(
                run["eps_hat"][:len(eta)] > eta * (1.0 + 1e-8)
            ))
        rows.append([
            name, np.mean(final), np.std(final, ddof=1),
            np.mean(runtime), np.std(runtime, ddof=1), np.mean(inner),
            (np.mean(finite_hits) if len(finite_hits) else np.nan),
            len(finite_hits), np.mean(violations),
        ])
    _write_csv(
        RESULTS / "exp5_inexact_cost.csv",
        ["inner_rule", "final_energy_mean", "final_energy_std",
         "runtime_mean", "runtime_std", "avg_dual_iterations",
         "time_to_1e-5", "successes", "budget_violations_mean"], rows,
    )
    return data


def experiment_6_large_scale(selection):
    methods = _comparison_methods(selection)
    sizes = [(1000, 3000), (2000, 6000), (4000, 12000)]
    rows = []
    raw = {}
    for m, n in sizes:
        for seed in range(5):
            A, b, tau = base.make_sparse_problem(10100 + n + seed, m, n)
            L = spectral_lipschitz(A)
            for method in methods:
                print(f"large m={m} n={n} seed={seed} method={method.name}", flush=True)
                run = _timed_run(
                    A, b, tau, method, 100, L=L, repeats=3,
                )
                raw[(m, n, seed, method.name)] = run
        for method in methods:
            runs = [raw[(m, n, seed, method.name)] for seed in range(5)]
            s = _summary(runs)
            final = np.asarray([run["normal"][-1] for run in runs])
            runtime = np.asarray([run["runtime"][-1] for run in runs])
            rows.append([
                m, n, method.name, np.median(final),
                np.percentile(final, 25), np.percentile(final, 75),
                np.median(runtime), np.percentile(runtime, 25),
                np.percentile(runtime, 75), s["target_mean"],
                s["success"], s["avg_q"], s["q1_share"], s["avg_inner"],
            ])
    _write_csv(
        RESULTS / "exp6_large_scale.csv",
        ["m", "n", "method", "final_residual_median", "final_residual_q25",
         "final_residual_q75", "runtime_median", "runtime_q25", "runtime_q75",
         "time_to_5e-4", "successes",
         "average_q", "q1_share", "avg_inner_iterations"], rows,
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.25))
    markers = {"one-cut": "o", "recent": "s", "violated": "^", "angle": "D", "adaptive": "P"}
    styles = {"one-cut": "-", "recent": "--", "violated": "-.", "angle": ":", "adaptive": "-"}
    for method in methods:
        subset = [row for row in rows if row[2] == method.name]
        ns = [row[1] for row in subset]
        plot_kw = dict(
            color=COLORS[method.name], label=method.name,
            marker=markers[method.name], linestyle=styles[method.name],
            linewidth=1.6, markersize=4.2,
        )
        time_median = np.asarray([row[6] / 100 for row in subset])
        time_q25 = np.asarray([row[7] / 100 for row in subset])
        time_q75 = np.asarray([row[8] / 100 for row in subset])
        residual_median = np.asarray([row[3] for row in subset])
        residual_q25 = np.asarray([row[4] for row in subset])
        residual_q75 = np.asarray([row[5] for row in subset])
        axes[0].plot(ns, time_median, **plot_kw)
        axes[0].fill_between(
            ns, np.maximum(time_q25, 1e-12), time_q75,
            color=COLORS[method.name],
            alpha=0.08, linewidth=0,
        )
        axes[1].plot(ns, residual_median, **plot_kw)
        axes[1].fill_between(
            ns, np.maximum(residual_q25, 1e-12), residual_q75,
            color=COLORS[method.name],
            alpha=0.08, linewidth=0,
        )
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[1].axhline(5e-4, color="#555555", linestyle="--", linewidth=0.9, label=r"target $5\times10^{-4}$")
    axes[0].set_ylabel("time per iteration (s)")
    axes[1].set_ylabel("residual at iteration 100")
    for ax in axes:
        ax.set_xlabel("dimension n")
        ax.grid(True, which="major", color="#D3D3D3", linewidth=0.65)
    axes[0].legend(fontsize=6.6, ncol=2)
    axes[1].legend(fontsize=6.4, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "exp6_large_scale.pdf", bbox_inches="tight")
    plt.close(fig)
    return raw


def experiment_7_parameter_sensitivity(selection):
    """Small large-scale sensitivity study for the supplementary material.

    Each configuration changes one calibrated quantity whenever possible.  The
    q=10 case uses p=20 because the candidate pool is required to contain at
    least twice the maximum working-set size.
    """
    solver = selection["solver"]
    configurations = [
        ("q=3", 3, 10, 0.05),
        ("baseline", 5, 10, 0.05),
        ("q=10", 10, 20, 0.05),
        ("p=25", 5, 25, 0.05),
        ("p=50", 5, 50, 0.05),
        ("c=0.02", 5, 10, 0.02),
        ("c=0.10", 5, 10, 0.10),
    ]
    raw = {label: [] for label, _, _, _ in configurations}
    for seed in range(5):
        A, b, tau = base.make_sparse_problem(10100 + 12000 + seed, 4000, 12000)
        L = spectral_lipschitz(A)
        for label, qmax, pmax, c_trig in configurations:
            method = Method(
                label, "angle", qmax=qmax, pmax=pmax, solver=solver,
                adaptive=True, c_trig=c_trig,
            )
            print(f"sensitivity seed={seed} config={label}", flush=True)
            raw[label].append(
                _timed_run(A, b, tau, method, 100, L=L, repeats=3)
            )

    rows = []
    for label, qmax, pmax, c_trig in configurations:
        runs = raw[label]
        final = np.asarray([run["normal"][-1] for run in runs])
        runtime = np.asarray([run["runtime"][-1] for run in runs])
        summary = _summary(runs)
        rows.append([
            label, qmax, pmax, c_trig,
            np.median(final), np.percentile(final, 25),
            np.percentile(final, 75), np.median(runtime),
            np.percentile(runtime, 25), np.percentile(runtime, 75),
            summary["success"], summary["avg_q"], summary["q1_share"],
            summary["avg_inner"],
        ])
    _write_csv(
        RESULTS / "exp7_parameter_sensitivity.csv",
        ["configuration", "qmax", "pmax", "c_trig",
         "final_residual_median", "final_residual_q25",
         "final_residual_q75", "runtime_median", "runtime_q25",
         "runtime_q75", "successes", "average_q", "q1_share",
         "avg_inner_iterations"],
        rows,
    )

    labels = [row[0] for row in rows]
    xpos = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    residual = np.asarray([row[4] for row in rows])
    residual_low = residual - np.asarray([row[5] for row in rows])
    residual_high = np.asarray([row[6] for row in rows]) - residual
    runtime = np.asarray([row[7] for row in rows])
    runtime_low = runtime - np.asarray([row[8] for row in rows])
    runtime_high = np.asarray([row[9] for row in rows]) - runtime
    colors = ["#0072B2" if label == "baseline" else "#7A7A7A" for label in labels]
    axes[0].errorbar(
        xpos, residual, yerr=np.vstack([residual_low, residual_high]),
        fmt="none", ecolor="#555555", elinewidth=1.0, capsize=3,
    )
    axes[0].scatter(xpos, residual, c=colors, s=30, zorder=3)
    axes[0].set_yscale("log")
    axes[0].axhline(5e-4, color="#D55E00", linestyle="--", linewidth=1.0,
                    label=r"target $5\times10^{-4}$")
    axes[0].set_ylabel("residual at iteration 100")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].errorbar(
        xpos, runtime, yerr=np.vstack([runtime_low, runtime_high]),
        fmt="none", ecolor="#555555", elinewidth=1.0, capsize=3,
    )
    axes[1].scatter(xpos, runtime, c=colors, s=30, zorder=3)
    axes[1].set_ylabel("total algorithm time (s)")
    for ax in axes:
        ax.set_xticks(xpos, labels, rotation=28, ha="right")
        ax.grid(True, axis="y", color="#D3D3D3", linewidth=0.65)
        ax.set_xlabel("adaptive configuration")
    fig.tight_layout()
    fig.savefig(OUT / "exp7_parameter_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    return raw


def _save_npz(groups):
    data = {}
    stack = [("adaptive_revision", groups)]
    while stack:
        prefix, value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                clean = str(key).replace(" ", "_").replace("=", "")
                stack.append((f"{prefix}__{clean}", child))
        elif isinstance(value, list):
            for seed, run in enumerate(value):
                for metric, array in run.items():
                    data[f"{prefix}__seed{seed}__{metric}"] = array
    np.savez_compressed(RESULTS / "adaptive_revision_raw.npz", **data)


def main():
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5, "legend.fontsize": 7.2, "figure.dpi": 150})
    selection = experiment_1_calibration()
    exp2 = experiment_2_main(selection)
    exp3 = experiment_3_energy_conditioning(selection)
    experiment_4_theta(exp3)
    exp5 = experiment_5_certificate()
    exp6 = experiment_6_large_scale(selection)
    exp7 = experiment_7_parameter_sensitivity(selection)
    _save_npz({"exp2": exp2, "exp3": exp3, "exp5": exp5,
               "exp6": exp6, "exp7": exp7})
    print(f"adaptive revision complete: {selection}", flush=True)


if __name__ == "__main__":
    main()
