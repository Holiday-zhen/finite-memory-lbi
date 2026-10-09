"""Long-horizon diagnostic for the two matrix problems in Experiment 5.6."""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results" / "final_exp56"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

DIAGNOSTICS = ROOT / "src" / "diagnostics"
sys.path.insert(0, str(ROOT / "experiments" / "pq_followup"))
sys.path.insert(0, str(DIAGNOSTICS / "shutdown_hard_benchmark"))
sys.path.insert(0, str(DIAGNOSTICS / "precision_followup_20260904"))
import run_pq_followup as pq
import run_experiment as hard


HORIZON = 2000
CHECKPOINTS = (100, 200, 500, 1000, 2000)
SEEDS = range(100, 120)
METHODS = (
    hard.Method("one-cut", "recent", 1, 1),
    hard.Method("recent", "recent", 3, 6),
    hard.Method("violated", "violated", 3, 6),
    hard.Method("angle", "angle", 3, 6),
    hard.Method("adaptive", "angle", 3, 6, True, 1e-3),
)
BUILDERS = {
    "column_block": lambda seed: hard.completion.build_problem(
        "column-block", seed, 0.50, 0.20
    ),
    "binary_digits": lambda seed: hard.real_matrix.build_problem(
        "Binary Digits pixel feasibility", seed, "heterogeneous", 0.50, 0.80
    ),
}


def projected_residual(alpha, gradient):
    return float(np.linalg.norm(alpha - np.maximum(alpha - gradient, 0.0)))


def capped_bb(problem, x_star, G, beta, warm, step0=None, max_inner=5):
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

    def value_grad(value):
        dual_point = x_star - Gs @ value
        primal = problem.mirror(dual_point)
        objective = problem.omega_star(dual_point) + float(beta_s @ value)
        gradient = beta_s - Gs.T @ primal
        return float(objective), np.asarray(gradient)

    phi, grad = value_grad(alpha)
    pgn = projected_residual(alpha, grad)
    iterations = 0
    for iteration in range(1, max_inner + 1):
        old_alpha, old_grad, old_phi = alpha.copy(), grad.copy(), phi
        local = step
        for _ in range(8):
            trial = np.maximum(old_alpha - local * old_grad, 0.0)
            phi_trial, grad_trial = value_grad(trial)
            displacement = trial - old_alpha
            if phi_trial <= old_phi + 1e-4 * float(old_grad @ displacement):
                break
            local *= 0.5
        alpha, phi, grad = trial, phi_trial, grad_trial
        iterations = iteration
        pgn = projected_residual(alpha, grad)
        displacement = alpha - old_alpha
        gradient_change = grad - old_grad
        curvature = float(displacement @ gradient_change)
        step = (
            float(np.clip(
                float(displacement @ displacement) / curvature,
                1e-8 * safe_step, 1e6 * safe_step,
            )) if curvature > 1e-24 else safe_step
        )
    return alpha / scale, iterations, pgn, step


def run(problem, method):
    x = np.zeros(problem.dim)
    x_star = np.zeros(problem.dim)
    g0 = max(float(np.linalg.norm(problem.grad(x))), 1e-300)
    pool = []
    warm_by_cut = {}
    previous_current_alpha = 0.0
    bb_step = None
    initial_one_step = None
    residuals, objectives, pgns, times = [], [], [], []
    algorithm_time = 0.0

    for k in range(HORIZON):
        tic = time.perf_counter()
        raw_a = np.asarray(problem.grad(x))
        scale = max(float(np.linalg.norm(raw_a)), 1e-300)
        if scale <= 1e-30 * g0:
            break
        raw_beta = float(raw_a @ x - (raw_a @ raw_a) / problem.lipschitz)
        a, beta = raw_a / scale, raw_beta / scale
        current = {"id": k, "a": a.copy(), "beta": beta}
        pool.append(current)
        pool = hard.prune_pool(pool, method.p, x)

        one_alpha = pq.strict_one_cut(problem, x_star, a, beta)
        x_one = problem.mirror(x_star - one_alpha[0] * a)
        one_step = problem.omega(x_one) - problem.omega(x) - float(
            x_star @ (x_one - x)
        )
        if initial_one_step is None and one_step > 0.0:
            initial_one_step = one_step

        if method.q == 1:
            selected = [current]
        elif method.adaptive:
            old = [cut for cut in pool if cut["id"] != k]
            required = max(
                old, key=lambda cut: hard.positive_violation(cut, x_one),
                default=None,
            )
            delta = hard.positive_violation(required, x_one) if required else 0.0
            floor = max(1e-18, 1e-14 * (initial_one_step or 0.0))
            gain = 0.5 * delta * delta / one_step if one_step > floor else 0.0
            selected = (
                hard.angle_select(current, old, x_one, method.q, required)
                if gain >= method.trigger and delta > 0.0 else [current]
            )
        else:
            selected = hard.fixed_select(method, pool, current, x)

        ids = [cut["id"] for cut in selected]
        G = np.column_stack([cut["a"] for cut in selected])
        betas = np.asarray([cut["beta"] for cut in selected])
        if len(selected) == 1:
            alpha, nit, pgn = one_alpha, 0, 0.0
        else:
            warm = np.asarray([
                warm_by_cut.get(
                    cut_id,
                    previous_current_alpha if cut_id == k else 0.0,
                ) for cut_id in ids
            ])
            alpha, nit, pgn, bb_step = capped_bb(
                problem, x_star, G, betas, warm, step0=bb_step
            )
        current_position = ids.index(k)
        previous_current_alpha = float(alpha[current_position])
        x_star = x_star - G @ alpha
        x = problem.mirror(x_star)
        warm_by_cut = {
            cut_id: float(value) for cut_id, value in zip(ids, alpha)
            if value > 1e-14
        }
        pool = hard.prune_pool(pool, method.p, x)
        live = {cut["id"] for cut in pool}
        warm_by_cut = {
            cut_id: value for cut_id, value in warm_by_cut.items()
            if cut_id in live
        }
        algorithm_time += time.perf_counter() - tic

        residuals.append(float(np.linalg.norm(problem.grad(x)) / g0))
        objectives.append(float(problem.omega(x)))
        pgns.append(float(pgn))
        times.append(algorithm_time)

    def padded(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return np.zeros(HORIZON)
        return np.pad(values, (0, HORIZON - len(values)), mode="edge")

    return {
        "residual": padded(residuals),
        "objective": padded(objectives),
        "pgn": padded(pgns),
        "runtime": padded(times),
    }


def main():
    runs = {}
    rows = []
    with threadpool_limits(1):
        for problem_name, builder in BUILDERS.items():
            for seed in SEEDS:
                problem = builder(seed)
                for method in METHODS:
                    result = run(problem, method)
                    runs[(problem_name, seed, method.name)] = result
                    np.savez_compressed(
                        OUT / f"{problem_name}__{seed}__{method.name}.npz", **result
                    )
                    row = {
                        "problem": problem_name,
                        "seed": seed,
                        "method": method.name,
                        "best_residual": float(np.min(result["residual"])),
                        "best_iteration": int(np.argmin(result["residual"]) + 1),
                        "final_residual": float(result["residual"][-1]),
                        "final_objective": float(result["objective"][-1]),
                        "final_core_time_s": float(result["runtime"][-1]),
                        "tail100_pgn_median": float(np.median(result["pgn"][-100:])),
                    }
                    for checkpoint in CHECKPOINTS:
                        row[f"residual_{checkpoint}"] = float(
                            result["residual"][checkpoint - 1]
                        )
                    rows.append(row)
                    print(
                        f"{problem_name} seed={seed} {method.name:8s} "
                        f"r200={row['residual_200']:.3e} "
                        f"r2000={row['residual_2000']:.3e} "
                        f"best={row['best_residual']:.3e}@{row['best_iteration']}",
                        flush=True,
                    )

    with (OUT / "raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for problem_name in BUILDERS:
        for method in METHODS:
            group = [
                row for row in rows
                if row["problem"] == problem_name and row["method"] == method.name
            ]
            final_residuals = [row["final_residual"] for row in group]
            final_objectives = [row["final_objective"] for row in group]
            final_times = [row["final_core_time_s"] for row in group]
            summary = {
                "problem": problem_name,
                "method": method.name,
                "n": len(group),
                "median_best_residual": float(np.median([
                    row["best_residual"] for row in group
                ])),
                "median_best_iteration": float(np.median([
                    row["best_iteration"] for row in group
                ])),
                "median_final_residual": float(np.median([
                    row["final_residual"] for row in group
                ])),
                "median_final_core_time_s": float(np.median([
                    row["final_core_time_s"] for row in group
                ])),
                "median_tail100_pgn": float(np.median([
                    row["tail100_pgn_median"] for row in group
                ])),
                "final_residual_q1": float(np.quantile(final_residuals, 0.25)),
                "final_residual_q3": float(np.quantile(final_residuals, 0.75)),
                "objective_q1": float(np.quantile(final_objectives, 0.25)),
                "objective_median": float(np.median(final_objectives)),
                "objective_q3": float(np.quantile(final_objectives, 0.75)),
                "core_time_q1_s": float(np.quantile(final_times, 0.25)),
                "core_time_q3_s": float(np.quantile(final_times, 0.75)),
            }
            for checkpoint in CHECKPOINTS:
                values = [row[f"residual_{checkpoint}"] for row in group]
                summary[f"median_residual_{checkpoint}"] = float(np.median(values))
            summaries.append(summary)

    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    colors = {name: pq.COLORS[name] for name in pq.METHOD_NAMES}
    styles = {name: pq.STYLES[name] for name in pq.METHOD_NAMES}
    figure_specs = (
        ("residual_time.png", "runtime", "residual", "Core time (s)", "Relative residual", True),
        ("residual_iteration.png", "iteration", "residual", "Outer iteration", "Relative residual", True),
        ("objective_iteration.png", "iteration", "objective", "Outer iteration", "Objective value", False),
    )
    for filename, xmetric, ymetric, xlabel, ylabel, show_iqr in figure_specs:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.45), constrained_layout=True)
        for ax, problem_name in zip(axes, BUILDERS):
            for method in METHODS:
                group = [runs[(problem_name, seed, method.name)] for seed in SEEDS]
                values = np.vstack([run[ymetric] for run in group])
                median = np.median(values, axis=0)
                q1 = np.quantile(values, 0.25, axis=0)
                q3 = np.quantile(values, 0.75, axis=0)
                x = (
                    np.median(np.vstack([run["runtime"] for run in group]), axis=0)
                    if xmetric == "runtime" else np.arange(1, HORIZON + 1)
                )
                y = np.maximum(median, 1e-20) if ymetric == "residual" else median
                ax.plot(x, y, color=colors[method.name], linestyle=styles[method.name],
                        linewidth=1.4, label=method.name)
                if show_iqr:
                    ax.fill_between(x, np.maximum(q1, 1e-20), np.maximum(q3, 1e-20),
                                    color=colors[method.name], alpha=0.09, linewidth=0)
            if ymetric == "residual":
                ax.set_yscale("log")
            ax.set_title(
                "Column-block completion" if problem_name == "column_block"
                else "Binary Digits feasibility"
            )
            ax.set_xlabel(xlabel)
            ax.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
        fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    latex = [
        r"\begin{tabular}{llccc}", r"\hline",
        r"Problem & Method & Final residual & Objective & Core time (s)\\", r"\hline",
    ]
    for row in summaries:
        latex.append(
            f"{row['problem'].replace('_', ' ')} & {row['method']} & "
            f"${row['median_final_residual']:.2e}$ & "
            f"${row['objective_median']:.4f}$ & {row['median_final_core_time_s']:.3f}\\\\"
        )
    latex.extend([r"\hline", r"\end{tabular}"])
    (OUT / "main_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
