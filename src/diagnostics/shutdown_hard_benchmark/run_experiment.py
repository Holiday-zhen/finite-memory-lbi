"""Long-run shutdown benchmark for revised Experiments 5.2 and 5.7.

This is an isolated diagnostic entry point.  It does not overwrite any of the
submission-package data.  It supports two predeclared permanent-q=1 rules: an
online plateau detector and a fixed switch after the 100 iterations already
used by the submitted Experiment 5.2.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPERIMENTS = REPO / "experiments"
VIRTUAL = REPO / "diagnostics" / "four_virtual_examples"
GEOMETRY = REPO / "diagnostics" / "geometry_favorable_benchmark"
REAL = REPO / "diagnostics" / "real_multitask_effect_benchmark"
for folder in [EXPERIMENTS, VIRTUAL, GEOMETRY, REAL]:
    sys.path.insert(0, str(folder))

import experiments_bounded as sparse_base
from experiments_theory_aligned import (
    bregman,
    projected_grad_norm,
    soft_threshold,
    solve_dual,
)
import run_virtual_examples as virtual
import screen_completion_regimes as completion
import screen_real_matrix_models as real_matrix


OUT = HERE / "results"
TARGET = 1.0e-6
SHUTDOWN = {
    "policy": "plateau",
    "min_iter": 100,
    "window": 50,
    "relative_improvement": 0.01,
    "consecutive_checks": 10,
}
FIXED_100_SHUTDOWN = {
    "policy": "fixed_iteration",
    "iteration": 100,
}
ACTIVE_SHUTDOWN = SHUTDOWN


@dataclass(frozen=True)
class Method:
    name: str
    rule: str
    q: int
    p: int
    adaptive: bool = False
    trigger: float = 0.0


SPARSE_METHODS = [
    Method("one-cut", "recent", 1, 1),
    Method("recent-shutdown", "recent", 5, 10),
    Method("violated-shutdown", "violated", 5, 10),
    Method("angle-shutdown", "angle", 5, 10),
    Method("adaptive-shutdown", "angle", 5, 10, True, 0.05),
]
HARD_METHODS = [
    Method("one-cut", "recent", 1, 1),
    Method("recent-shutdown", "recent", 3, 6),
    Method("violated-shutdown", "violated", 3, 6),
    Method("angle-shutdown", "angle", 3, 6),
    Method("adaptive-shutdown", "angle", 3, 6, True, 1.0e-3),
]


def positive_violation(cut, point):
    norm = max(float(np.linalg.norm(cut["a"])), 1.0e-300)
    return max(float(cut["a"] @ point - cut["beta"]), 0.0) / norm


def prune_pool(pool, capacity, point):
    if len(pool) <= capacity:
        return pool
    recent_count = max(1, capacity // 2)
    recent = sorted(pool, key=lambda cut: cut["id"], reverse=True)[:recent_count]
    used = {cut["id"] for cut in recent}
    remaining = [cut for cut in pool if cut["id"] not in used]
    informative = sorted(
        remaining, key=lambda cut: positive_violation(cut, point), reverse=True
    )[: capacity - len(recent)]
    return sorted(recent + informative, key=lambda cut: cut["id"])


def angle_select(current, candidates, point, q, required=None):
    useful = [cut for cut in candidates if positive_violation(cut, point) > 0.0]
    selected = [current]
    if required is not None and q > 1:
        match = next((cut for cut in useful if cut["id"] == required["id"]), None)
        if match is not None:
            selected.append(match)
            useful.remove(match)
    while useful and len(selected) < q:
        best = None
        best_score = -np.inf
        for cut in useful:
            normal = cut["a"]
            normal_norm = max(float(np.linalg.norm(normal)), 1.0e-300)
            max_cos = max(
                abs(float(normal @ kept["a"]))
                / max(normal_norm * float(np.linalg.norm(kept["a"])), 1.0e-300)
                for kept in selected
            )
            score = positive_violation(cut, point) * max(1.0 - max_cos, 1.0e-12)
            if score > best_score:
                best, best_score = cut, score
        selected.append(best)
        useful.remove(best)
    return selected


def fixed_select(method, pool, current, point):
    old = [cut for cut in pool if cut["id"] != current["id"]]
    if method.q <= 1 or not old:
        return [current]
    if method.rule == "recent":
        recent = sorted(old, key=lambda cut: cut["id"], reverse=True)
        return list(reversed(recent[: method.q - 1])) + [current]
    useful = [cut for cut in old if positive_violation(cut, point) > 0.0]
    if method.rule == "violated":
        useful.sort(key=lambda cut: positive_violation(cut, point), reverse=True)
        return useful[: method.q - 1] + [current]
    return angle_select(current, useful, point, method.q)


def shutdown_update(best_history, streak, enabled):
    """Return (new_streak, switch_now) using only past observed residuals."""
    if not enabled:
        return 0, False
    if ACTIVE_SHUTDOWN["policy"] == "fixed_iteration":
        return streak, len(best_history) >= ACTIVE_SHUTDOWN["iteration"]
    if len(best_history) < ACTIVE_SHUTDOWN["min_iter"]:
        return 0, False
    window = ACTIVE_SHUTDOWN["window"]
    if len(best_history) <= window:
        return 0, False
    old = best_history[-window - 1]
    new = best_history[-1]
    improvement = (old - new) / max(old, 1.0e-300)
    streak = streak + 1 if improvement < ACTIVE_SHUTDOWN["relative_improvement"] else 0
    return streak, streak >= ACTIVE_SHUTDOWN["consecutive_checks"]


def solve_sparse_memory(x_star, selected, tau, warm_by_cut, tol=1.0e-9):
    ids = [cut["id"] for cut in selected]
    G = np.column_stack([cut["a"] for cut in selected])
    betas = np.asarray([cut["beta"] for cut in selected])
    warm = np.asarray([warm_by_cut.get(i, 0.0) for i in ids])
    alpha, info = solve_dual(
        x_star, G, betas, tau, warm=warm, dual_tol=tol, max_iter=600
    )
    return alpha, info, ids, G


def run_sparse(A, b, tau, method, max_iter=2000, L=None):
    n = A.shape[1]
    L = sparse_base.spectral_lipschitz(A) if L is None else float(L)
    x = np.zeros(n)
    x_star = np.zeros(n)
    pool = []
    warm_by_cut = {}
    atb = np.asarray(A.T @ b).reshape(-1)
    norm_atb = max(float(np.linalg.norm(atb)), 1.0e-16)
    initial_grad2 = max(float(atb @ atb), 1.0e-300)
    initial_universal = max(initial_grad2 / (2.0 * L * L), 1.0e-300)
    keys = ["residual", "runtime", "q", "pgn", "theta_lb", "switched"]
    hist = {key: [] for key in keys}
    best_history = []
    streak = 0
    shutdown = False
    shutdown_iteration = 0
    started = time.perf_counter()

    for k in range(max_iter):
        streak, switch_now = shutdown_update(best_history, streak, method.q > 1 and not shutdown)
        if switch_now:
            shutdown = True
            shutdown_iteration = k + 1
            pool = []
            warm_by_cut = {}

        residual = np.asarray(A @ x - b).reshape(-1)
        raw_a = np.asarray(A.T @ residual).reshape(-1)
        grad2 = float(raw_a @ raw_a)
        if grad2 <= 1.0e-20 * initial_grad2:
            break
        beta = float(raw_a @ x - grad2 / L)
        current = {"id": k, "a": raw_a.copy(), "beta": beta}
        pool.append(current)
        pool = prune_pool(pool, 1 if shutdown else method.p, x)

        one_alpha, one_info = sparse_base.solve_one_cut_exact(
            x_star, raw_a, beta, tau, L, tol=1.0e-13, max_iter=100
        )
        x_one = soft_threshold(x_star - raw_a * one_alpha[0], tau)
        one_step = bregman(x_one, x, x_star, tau)
        theta_lb = 1.0

        if shutdown or method.q == 1:
            selected = [current]
        elif method.adaptive:
            old = [cut for cut in pool if cut["id"] != k]
            required = max(old, key=lambda cut: positive_violation(cut, x_one), default=None)
            delta = positive_violation(required, x_one) if required is not None else 0.0
            gain = (
                0.5 * delta * delta / one_step
                if one_step > 1.0e-12 * initial_universal else 0.0
            )
            theta_lb = 1.0 + gain
            selected = (
                angle_select(current, old, x_one, method.q, required)
                if gain >= method.trigger and delta > 0.0 else [current]
            )
        else:
            selected = fixed_select(method, pool, current, x)

        if len(selected) == 1:
            alpha, info = one_alpha, one_info
            ids = [current["id"]]
            G = raw_a[:, None]
        else:
            alpha, info, ids, G = solve_sparse_memory(
                x_star, selected, tau, warm_by_cut
            )
        x_star = x_star - G @ alpha
        x = soft_threshold(x_star, tau)
        warm_by_cut = {i: float(v) for i, v in zip(ids, alpha) if v > 1.0e-14}
        normal = float(np.linalg.norm(np.asarray(A.T @ (A @ x - b)).reshape(-1)) / norm_atb)
        best_history.append(min(best_history[-1], normal) if best_history else normal)
        hist["residual"].append(normal)
        hist["runtime"].append(time.perf_counter() - started)
        hist["q"].append(len(selected))
        hist["pgn"].append(float(info.get("pgn", 0.0)))
        hist["theta_lb"].append(theta_lb)
        hist["switched"].append(float(shutdown))

        pool = prune_pool(pool, 1 if shutdown else method.p, x)
        live = {cut["id"] for cut in pool}
        warm_by_cut = {i: v for i, v in warm_by_cut.items() if i in live}

    result = {key: np.asarray(value, dtype=float) for key, value in hist.items()}
    result["shutdown_iteration"] = np.asarray([shutdown_iteration], dtype=float)
    return result


def run_hard(problem, method):
    x = np.zeros(problem.dim)
    x_star = np.zeros(problem.dim)
    pool = []
    warm_by_cut = {}
    g0 = max(float(np.linalg.norm(problem.grad(x))), 1.0e-30)
    keys = ["residual", "runtime", "q", "pgn", "theta_lb", "switched"]
    hist = {key: [] for key in keys}
    best_history = []
    streak = 0
    shutdown = False
    shutdown_iteration = 0
    initial_one_step = None
    started = time.perf_counter()

    for k in range(problem.max_iter):
        streak, switch_now = shutdown_update(best_history, streak, method.q > 1 and not shutdown)
        if switch_now:
            shutdown = True
            shutdown_iteration = k + 1
            pool = []
            warm_by_cut = {}

        raw_a = problem.grad(x)
        residual_before = float(np.linalg.norm(raw_a) / g0)
        if residual_before <= 1.0e-10:
            break
        raw_beta = float(raw_a @ x - (raw_a @ raw_a) / problem.lipschitz)
        scale = max(float(np.linalg.norm(raw_a)), 1.0e-30)
        a = raw_a / scale
        beta = raw_beta / scale
        current = {"id": k, "a": a.copy(), "beta": beta}
        pool.append(current)
        pool = prune_pool(pool, 1 if shutdown else method.p, x)

        one_alpha = virtual.one_cut(problem, x_star, a, beta)
        x_one = problem.mirror(x_star - one_alpha[0] * a)
        one_step = problem.omega(x_one) - problem.omega(x) - x_star @ (x_one - x)
        if initial_one_step is None and one_step > 0.0:
            initial_one_step = one_step
        denom_floor = max(1.0e-14, 1.0e-12 * (initial_one_step or 0.0))
        theta_lb = 1.0

        if shutdown or method.q == 1:
            selected = [current]
        elif method.adaptive:
            old = [cut for cut in pool if cut["id"] != k]
            required = max(old, key=lambda cut: positive_violation(cut, x_one), default=None)
            delta = positive_violation(required, x_one) if required is not None else 0.0
            gain = 0.5 * delta * delta / one_step if one_step > denom_floor else 0.0
            theta_lb = 1.0 + gain
            selected = (
                angle_select(current, old, x_one, method.q, required)
                if gain >= method.trigger and delta > 0.0 else [current]
            )
        else:
            selected = fixed_select(method, pool, current, x)

        ids = [cut["id"] for cut in selected]
        G = np.column_stack([cut["a"] for cut in selected])
        betas = np.asarray([cut["beta"] for cut in selected])
        if len(selected) == 1:
            alpha = one_alpha
            pgn = 0.0
        else:
            warm = np.asarray([warm_by_cut.get(i, 0.0) for i in ids])
            alpha, _, pgn = virtual.solve_dual(problem, x_star, G, betas, warm=warm)
        x_star = x_star - G @ alpha
        x = problem.mirror(x_star)
        warm_by_cut = {i: float(v) for i, v in zip(ids, alpha) if v > 1.0e-14}
        normal = float(np.linalg.norm(problem.grad(x)) / g0)
        best_history.append(min(best_history[-1], normal) if best_history else normal)
        hist["residual"].append(normal)
        hist["runtime"].append(time.perf_counter() - started)
        hist["q"].append(len(selected))
        hist["pgn"].append(float(pgn))
        hist["theta_lb"].append(theta_lb)
        hist["switched"].append(float(shutdown))

        pool = prune_pool(pool, 1 if shutdown else method.p, x)
        live = {cut["id"] for cut in pool}
        warm_by_cut = {i: v for i, v in warm_by_cut.items() if i in live}

    result = {key: np.asarray(value, dtype=float) for key, value in hist.items()}
    result["shutdown_iteration"] = np.asarray([shutdown_iteration], dtype=float)
    return result


def first_hit(run, target=TARGET):
    hits = np.flatnonzero(run["residual"] <= target)
    return int(hits[0] + 1) if len(hits) else np.nan


def save_run(problem_name, seed, method, run):
    safe_problem = problem_name.lower().replace(" ", "_").replace("-", "_")
    np.savez_compressed(OUT / f"{safe_problem}__seed_{seed}__{method.name}.npz", **run)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_suite(suite, smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    all_runs = {}
    rows = []

    if suite in {"sparse", "all"}:
        seeds = [0] if smoke else list(range(5))
        max_iter = 140 if smoke else 2000
        for seed_index, seed in enumerate(seeds):
            A, b, tau = sparse_base.make_sparse_problem(8100 + seed, 500, 1500)
            L = sparse_base.spectral_lipschitz(A)
            order = SPARSE_METHODS[seed_index % len(SPARSE_METHODS):] + SPARSE_METHODS[:seed_index % len(SPARSE_METHODS)]
            for method in order:
                print(f"sparse seed={seed} method={method.name}", flush=True)
                run = run_sparse(A, b, tau, method, max_iter=max_iter, L=L)
                all_runs.setdefault(("Sparse regression 5.2", method.name), []).append(run)
                save_run("Sparse regression 5.2", seed, method, run)
                rows.append(run_row("Sparse regression 5.2", seed, method, run))

    if suite in {"hard", "all"}:
        seeds = [20] if smoke else list(range(20, 30))
        problem_builders = {
            "Column-block matrix completion": lambda seed: completion.build_problem(
                "column-block", seed, 0.50, 0.20
            ),
            "Binary Digits pixel-interval feasibility": lambda seed: real_matrix.build_problem(
                "Binary Digits pixel feasibility", seed, "heterogeneous", 0.50, 0.80
            ),
        }
        for problem_index, (problem_name, builder) in enumerate(problem_builders.items()):
            for seed_index, seed in enumerate(seeds):
                problem = builder(seed)
                if smoke:
                    problem.max_iter = min(problem.max_iter, 35)
                shift = (problem_index + seed_index) % len(HARD_METHODS)
                order = HARD_METHODS[shift:] + HARD_METHODS[:shift]
                for method in order:
                    print(f"{problem_name} seed={seed} method={method.name}", flush=True)
                    run = run_hard(problem, method)
                    all_runs.setdefault((problem_name, method.name), []).append(run)
                    save_run(problem_name, seed, method, run)
                    rows.append(run_row(problem_name, seed, method, run))

    fields = list(rows[0]) if rows else []
    if rows:
        write_csv(OUT / "per_run.csv", fields, rows)
        summaries = summarize(all_runs)
        write_csv(OUT / "summary.csv", list(summaries[0]), summaries)
        plot_curves(all_runs)
        write_report(summaries)
    return rows


def run_row(problem, seed, method, run):
    hit = first_hit(run)
    return {
        "problem": problem,
        "seed": seed,
        "method": method.name,
        "iterations": len(run["residual"]),
        "final_residual": float(run["residual"][-1]),
        "iterations_to_1e-6": hit,
        "runtime_sec": float(run["runtime"][-1]),
        "mean_q": float(np.mean(run["q"])),
        "multicut_share": float(np.mean(run["q"] > 1)),
        "shutdown_iteration": int(run["shutdown_iteration"][0]),
        "max_pgn": float(np.max(run["pgn"])),
    }


def summarize(all_runs):
    rows = []
    for (problem, method), runs in sorted(all_runs.items()):
        hits = np.asarray([first_hit(run) for run in runs], dtype=float)
        finite = hits[np.isfinite(hits)]
        switches = np.asarray([run["shutdown_iteration"][0] for run in runs], dtype=float)
        positive_switches = switches[switches > 0]
        rows.append({
            "problem": problem,
            "method": method,
            "runs": len(runs),
            "successes_at_1e-6": len(finite),
            "median_iterations_to_1e-6": float(np.median(finite)) if len(finite) else np.nan,
            "median_final_residual": float(np.median([run["residual"][-1] for run in runs])),
            "median_runtime_sec": float(np.median([run["runtime"][-1] for run in runs])),
            "median_mean_q": float(np.median([np.mean(run["q"]) for run in runs])),
            "median_multicut_share": float(np.median([np.mean(run["q"] > 1) for run in runs])),
            "shutdown_successes": len(positive_switches),
            "median_shutdown_iteration": float(np.median(positive_switches)) if len(positive_switches) else np.nan,
            "max_pgn": float(max(np.max(run["pgn"]) for run in runs)),
        })
    return rows


def plot_curves(all_runs):
    problems = sorted({problem for problem, _ in all_runs})
    colors = {
        "one-cut": "#0072B2",
        "recent-shutdown": "#E69F00",
        "violated-shutdown": "#009E73",
        "angle-shutdown": "#CC79A7",
        "adaptive-shutdown": "#D55E00",
    }
    for problem in problems:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for method in colors:
            runs = all_runs.get((problem, method), [])
            if not runs:
                continue
            width = max(len(run["residual"]) for run in runs)
            grid = np.arange(1, width + 1)
            matrix = np.vstack([
                np.interp(grid, np.arange(1, len(run["residual"]) + 1), run["residual"])
                for run in runs
            ])
            median = np.median(matrix, axis=0)
            low, high = np.quantile(matrix, [0.25, 0.75], axis=0)
            ax.plot(grid, median, label=method, color=colors[method])
            ax.fill_between(grid, low, high, color=colors[method], alpha=0.12)
        ax.axhline(TARGET, color="black", linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_xlabel("outer iteration")
        ax.set_ylabel("relative residual")
        ax.set_title(problem)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        stem = problem.lower().replace(" ", "_").replace("-", "_")
        fig.savefig(OUT / f"{stem}_curves.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"{stem}_curves.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_report(summaries):
    lines = [
        "# Shutdown and hard-problem benchmark",
        "",
        "Common shutdown rule: " + json.dumps(ACTIVE_SHUTDOWN, ensure_ascii=False),
        "",
        "The target used for iteration-efficiency summaries is 1e-6.",
        "",
        "| Problem | Method | Success | Median iter. | Median final | Median runtime (s) | Median shutdown |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['problem']} | {row['method']} | {row['successes_at_1e-6']}/{row['runs']} "
            f"| {row['median_iterations_to_1e-6']:.4g} | {row['median_final_residual']:.3e} "
            f"| {row['median_runtime_sec']:.3g} | {row['median_shutdown_iteration']:.4g} |"
        )
    (OUT / "result_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global ACTIVE_SHUTDOWN, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["sparse", "hard", "all"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--policy", choices=["plateau", "fixed100"], default="plateau")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.policy == "fixed100":
        ACTIVE_SHUTDOWN = FIXED_100_SHUTDOWN
        OUT = HERE / "results_fixed100"
    if args.output_dir is not None:
        OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    config = {
        "suite": args.suite,
        "smoke": args.smoke,
        "target": TARGET,
        "shutdown": ACTIVE_SHUTDOWN,
        "sparse_seeds": list(range(5)),
        "hard_test_seeds": list(range(20, 30)),
    }
    (OUT / "configuration.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run_suite(args.suite, smoke=args.smoke)
    print(f"outputs: {OUT}", flush=True)


if __name__ == "__main__":
    main()
