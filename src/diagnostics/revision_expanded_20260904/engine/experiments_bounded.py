"""Submission experiments for finite-memory cut-and-project LBI.

The six experiments follow the narrative used in the final manuscript:
  1. convergence and angle-rule cost effectiveness;
  2. memory-size tradeoff and the choice q=5;
  3. linear Bregman-energy behavior;
  4. observed theta_k versus its retained-cut lower bound;
  5. computable inexact-projection certificate;
  6. sparse large-scale feasibility.

Every reported curve or table aggregates several deterministic random seeds.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import svds

from experiments_theory_aligned import (
    bregman,
    computable_epsilon,
    dual_phi_grad,
    make_gaussian_problem,
    make_spectral_problem,
    soft_threshold,
    solve_bilevel_reference,
    solve_dual,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "fig_submission"
RESULTS = ROOT / "experiments" / "results_submission"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "one-cut": "#0072B2",
    "recent q=5": "#E69F00",
    "violated q=5": "#009E73",
    "angle q=5": "#CC79A7",
    "fixed PG": "#D55E00",
    "certified": "#009E73",
    "high accuracy": "#0072B2",
}


@dataclass(frozen=True)
class Method:
    name: str
    rule: str
    q: int
    pool_size: int | None = None


MAIN_METHODS = [
    Method("one-cut", "recent", 1),
    Method("recent q=5", "recent", 5),
    Method("violated q=5", "violated", 5),
]
ANGLE_METHOD = Method("angle q=5", "angle", 5)
COMPARISON_METHODS = MAIN_METHODS + [ANGLE_METHOD]


def normalized_violation(cut, x):
    norm_a = max(float(np.linalg.norm(cut["a"])), 1e-300)
    return float((cut["a"] @ x - cut["beta"]) / norm_a)


def choose_memory(rule, q, pool, current_id, x, angle_threshold=0.98):
    """Choose at most q cuts from a bounded candidate pool."""
    current = next(cut for cut in pool if cut["id"] == current_id)
    if q <= 1 or len(pool) == 1:
        return [current]
    old = [cut for cut in pool if cut["id"] != current_id]
    if rule == "recent":
        selected = sorted(old, key=lambda cut: cut["id"], reverse=True)[: q - 1]
        return list(reversed(selected)) + [current]

    order = sorted(old, key=lambda cut: normalized_violation(cut, x), reverse=True)
    if rule == "violated":
        return order[: q - 1] + [current]
    if rule == "angle":
        selected = [current]
        for cut in order:
            if len(selected) >= q:
                break
            ai = cut["a"]
            ni = float(np.linalg.norm(ai))
            if ni == 0:
                continue
            if all(
                abs(float(ai @ kept["a"]) /
                    max(ni * float(np.linalg.norm(kept["a"])), 1e-300))
                <= angle_threshold
                for kept in selected
            ):
                selected.append(cut)
        if len(selected) < q:
            selected_ids = {cut["id"] for cut in selected}
            selected.extend(
                cut for cut in order
                if cut["id"] not in selected_ids
            )
        return selected[:q]
    raise ValueError(f"unknown memory rule: {rule}")


def prune_pool(pool, pool_size, x):
    """Keep a bounded mixture of recent and currently violated cuts."""
    if len(pool) <= pool_size:
        return pool
    recent_count = max(1, pool_size // 2)
    recent = sorted(pool, key=lambda cut: cut["id"], reverse=True)[:recent_count]
    keep_ids = {cut["id"] for cut in recent}
    remaining = [cut for cut in pool if cut["id"] not in keep_ids]
    informative = sorted(
        remaining, key=lambda cut: normalized_violation(cut, x), reverse=True
    )[: pool_size - len(recent)]
    kept = recent + informative
    return sorted(kept, key=lambda cut: cut["id"])


def solve_one_cut_exact(x_star, a, beta, tau, L, tol=1e-13, max_iter=100):
    """Solve the scalar exact one-cut dual by bracketed root finding."""
    upper = 1.0 / float(L)

    def derivative(t):
        return float(beta - a @ soft_threshold(x_star - t * a, tau))

    left_value = derivative(0.0)
    if left_value >= -tol:
        alpha = 0.0
        nit = 0
    else:
        right_value = derivative(upper)
        expansions = 0
        while right_value < 0.0 and expansions < 60:
            upper *= 2.0
            right_value = derivative(upper)
            expansions += 1
        if right_value < 0.0:
            raise RuntimeError("failed to bracket the scalar one-cut dual root")
        calls = [0]

        def counted(t):
            calls[0] += 1
            return derivative(t)

        alpha = float(brentq(
            counted, 0.0, upper, xtol=tol, rtol=4 * np.finfo(float).eps,
            maxiter=max_iter,
        ))
        nit = expansions + calls[0]
    grad = derivative(alpha)
    return np.asarray([alpha]), {
        "nit": nit,
        "pgn": abs(alpha - max(alpha - grad, 0.0)),
        "success": True,
        "solver": "scalar-brent",
    }


def spectral_lipschitz(A):
    if sparse.issparse(A):
        v0 = np.random.default_rng(0).standard_normal(min(A.shape))
        sigma = float(
            svds(A, k=1, which="LM", v0=v0, return_singular_vectors=False)[0]
        )
        return sigma * sigma
    return float(np.linalg.norm(A, 2) ** 2)


def as_vector(value):
    return np.asarray(value, dtype=float).reshape(-1)


def run_algorithm(
    A,
    b,
    tau,
    method,
    max_iter,
    *,
    xbar=None,
    L=None,
    solve_mode="accurate",
    fixed_tol=1e-3,
    eta0=None,
    track_theta=False,
    memory_start=0,
    pool_size=None,
):
    n = A.shape[1]
    L = spectral_lipschitz(A) if L is None else float(L)
    x = np.zeros(n)
    x_star = np.zeros(n)
    if pool_size is None:
        pool_size = method.pool_size if method.pool_size is not None else max(method.q, 5 * method.q)
    pool_size = max(int(pool_size), method.q)
    pool = []
    warm_by_cut = {}
    atb = as_vector(A.T @ b)
    norm_atb = max(float(np.linalg.norm(atb)), 1e-16)
    initial_grad2 = max(float(atb @ atb), 1e-300)
    energy0 = bregman(xbar, x, x_star, tau) if xbar is not None else np.nan
    initial_universal = max(initial_grad2 / (2 * L * L), 1e-16)
    if eta0 is None:
        eta0 = 1e-3 * initial_universal

    keys = [
        "iter", "normal", "energy_rel", "theta", "theta_lb", "eps_hat",
        "eta", "pgn", "dual_nit", "runtime", "memory_size",
        "candidate_pool_size",
    ]
    hist = {key: [] for key in keys}
    if xbar is not None:
        hist['primal_trace'] = []
        hist['dual_trace'] = []
    start = time.perf_counter()
    diagnostic_time = 0.0

    for k in range(max_iter):
        residual = as_vector(A @ x - b)
        a = as_vector(A.T @ residual)
        grad2 = float(a @ a)
        if grad2 <= 1e-20 * initial_grad2:
            break
        beta = float(a @ x - grad2 / L)
        current_cut = {"id": k, "a": a.copy(), "beta": beta}
        pool.append(current_cut)
        # ``memory_start`` is used only by the separate diagnostic study.  Its
        # default value preserves every reported submission experiment.
        selected_cuts = (
            [current_cut]
            if k < memory_start
            else choose_memory(method.rule, method.q, pool, k, x)
        )
        selected_ids = [cut["id"] for cut in selected_cuts]
        G = np.column_stack([cut["a"] for cut in selected_cuts])
        beta_vec = np.asarray([cut["beta"] for cut in selected_cuts])
        warm = np.asarray([warm_by_cut.get(i, 0.0) for i in selected_ids])

        one_step = np.nan
        x_one = None
        if track_theta:
            t_diag = time.perf_counter()
            one_alpha, _ = solve_one_cut_exact(
                x_star, a, beta, tau, L, tol=1e-13, max_iter=100,
            )
            x_one = soft_threshold(x_star - a * one_alpha[0], tau)
            diagnostic_time += time.perf_counter() - t_diag

        universal = grad2 / (2 * L * L)
        eta_k = eta0 / ((k + 1) ** 1.5) if solve_mode == "certified" else np.nan
        if solve_mode == "accurate":
            tolerances = [1e-12]
        elif solve_mode == "fixed":
            tolerances = [fixed_tol]
        elif solve_mode == "certified":
            tolerances = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13]
        else:
            raise ValueError(solve_mode)

        total_nit = 0
        alpha = None
        info = None
        for inner_tol in tolerances:
            alpha, info = solve_dual(
                x_star, G, beta_vec, tau, warm=warm,
                dual_tol=inner_tol, max_iter=500,
            )
            total_nit += info['nit']
            x_trial_star = x_star - G @ alpha
            x_trial = soft_threshold(x_trial_star, tau)
            step_trial = bregman(x_trial, x, x_star, tau)
            _, grad_trial = dual_phi_grad(alpha, x_star, G, beta_vec, tau)
            eps_trial = computable_epsilon(alpha, grad_trial, universal, step_trial)
            warm = alpha
            if solve_mode != 'certified' or eps_trial <= eta_k:
                break
        if solve_mode == 'certified' and eps_trial > eta_k:
            raise FloatingPointError(f'Certificate budget unmet at k={k}: {eps_trial} > {eta_k}')

        warm_by_cut = {
            i: float(v) for i, v in zip(selected_ids, alpha) if v > 1e-14
        }
        x_old, x_star_old = x, x_star
        x_star = x_star_old - G @ alpha
        x = soft_threshold(x_star, tau)
        step = bregman(x, x_old, x_star_old, tau)
        _, grad_phi = dual_phi_grad(alpha, x_star_old, G, beta_vec, tau)
        eps_hat = computable_epsilon(alpha, grad_phi, universal, step)

        theta = np.nan
        theta_lb = np.nan
        if track_theta:
            one_step = bregman(x_one, x_old, x_star_old, tau)
            # The ratio is not reported once the one-cut displacement reaches
            # numerical resolution; both numerator and denominator then lose
            # relative accuracy although the iterates themselves remain valid.
            if one_step > 1e-12 * initial_universal:
                theta = step / one_step
                deltas = []
                for cut in selected_cuts:
                    norm_aj = float(np.linalg.norm(cut["a"]))
                    if norm_aj > 0:
                        violation = max(
                            float(cut["a"] @ x_one - cut["beta"]), 0.0
                        )
                        deltas.append(violation / norm_aj)
                delta = max(deltas, default=0.0)
                theta_lb = 1.0 + 0.5 * delta * delta / one_step

        normal = float(np.linalg.norm(as_vector(A.T @ (A @ x - b))) / norm_atb)
        energy_rel = (
            bregman(xbar, x, x_star, tau) / max(energy0, 1e-300)
            if xbar is not None else np.nan
        )

        hist["iter"].append(k + 1)
        hist["normal"].append(normal)
        hist["energy_rel"].append(energy_rel)
        hist["theta"].append(theta)
        hist["theta_lb"].append(theta_lb)
        hist["eps_hat"].append(eps_hat)
        hist["eta"].append(eta_k)
        hist["pgn"].append(info["pgn"])
        hist["dual_nit"].append(total_nit)
        hist["runtime"].append(time.perf_counter() - start - diagnostic_time)
        hist["memory_size"].append(len(selected_cuts))
        if xbar is not None:
            hist['primal_trace'].append(x.copy())
            hist['dual_trace'].append(x_star.copy())
        pool = prune_pool(pool, pool_size, x)
        live_ids = {cut["id"] for cut in pool}
        warm_by_cut = {
            i: value for i, value in warm_by_cut.items() if i in live_ids
        }
        hist["candidate_pool_size"].append(len(pool))

    return {key: np.asarray(value, dtype=float) for key, value in hist.items()}


def stack_metric(runs, key):
    width = max(len(run[key]) for run in runs)
    values = np.full((len(runs), width), np.nan)
    for row, run in enumerate(runs):
        values[row, : len(run[key])] = run[key]
    return values


def mean_std(runs, key):
    values = stack_metric(runs, key)
    return np.nanmean(values, axis=0), np.nanstd(values, axis=0, ddof=1)


def plot_mean_band(
    ax, runs, key, label, color, *, log=True, linewidth=1.8,
    linestyle="-", marker=None, markevery=None,
):
    values = stack_metric(runs, key)
    mean = np.nanmean(values, axis=0)
    x = np.arange(1, len(mean) + 1)
    if log:
        mean = np.maximum(mean, 1e-18)
        lower = np.maximum(np.nanpercentile(values, 25, axis=0), 1e-18)
        upper = np.maximum(np.nanpercentile(values, 75, axis=0), 1e-18)
        ax.semilogy(
            x, mean, label=label, color=color, linewidth=linewidth,
            linestyle=linestyle, marker=marker, markevery=markevery,
        )
    else:
        std = np.nanstd(values, axis=0, ddof=1)
        lower, upper = mean - std, mean + std
        ax.plot(
            x, mean, label=label, color=color, linewidth=linewidth,
            linestyle=linestyle, marker=marker, markevery=markevery,
        )
    ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)


def plot_log10_band(
    ax, runs, key, label, color, *, linewidth=1.8,
    linestyle="-", marker=None, markevery=None,
):
    """Plot log10 of a positive mean curve with transformed IQR bands."""
    values = stack_metric(runs, key)
    mean = np.maximum(np.nanmean(values, axis=0), 1e-18)
    lower = np.maximum(np.nanpercentile(values, 25, axis=0), 1e-18)
    upper = np.maximum(np.nanpercentile(values, 75, axis=0), 1e-18)
    x = np.arange(1, len(mean) + 1)
    ax.plot(
        x, np.log10(mean), label=label, color=color, linewidth=linewidth,
        linestyle=linestyle, marker=marker, markevery=markevery,
        markersize=4.0,
    )
    ax.fill_between(
        x, np.log10(lower), np.log10(upper), color=color,
        alpha=0.10, linewidth=0,
    )


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def final_values(runs, key):
    return np.asarray([run[key][-1] for run in runs], dtype=float)


def tail_factor(run, start=30, end=55):
    """Median energy ratio on the pre-resolution linear tail."""
    energy = np.asarray(run["energy_rel"], dtype=float)
    if len(energy) < 2:
        return np.nan
    stop = min(end, len(energy) - 1)
    begin = min(start, max(stop - 1, 0))
    ratios = energy[begin:stop] / np.maximum(energy[begin - 1:stop - 1], 1e-300)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    return float(np.median(ratios)) if len(ratios) else np.nan


def total_runtime(run):
    return float(run["runtime"][-1])


def run_experiment_1(seeds=range(5)):
    methods = MAIN_METHODS + [ANGLE_METHOD]
    all_runs = {problem: {m.name: [] for m in methods} for problem in ["consistent", "inconsistent"]}
    for problem in all_runs:
        for seed in seeds:
            A, b, _, tau = make_gaussian_problem(
                1000 + 10 * seed + (problem == "inconsistent"),
                80, 240, inconsistent=(problem == "inconsistent"), sparsity=16,
            )
            L = spectral_lipschitz(A)
            for method in methods:
                print(f"exp1 {problem} seed={seed}: {method.name}", flush=True)
                all_runs[problem][method.name].append(
                    run_algorithm(A, b, tau, method, 220, L=L)
                )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, problem in zip(axes, ["consistent", "inconsistent"]):
        for method in methods:
            plot_mean_band(ax, all_runs[problem][method.name], "normal", method.name, COLORS[method.name])
        ax.set_title(f"{problem} least squares")
        ax.set_xlabel("iteration")
        ax.set_ylabel("normalized normal residual")
        ax.grid(True, which="major", color="#D0D0D0", linewidth=0.7)
        ax.grid(False, which="minor")
        ax.tick_params(axis="y", which="minor", length=2.0)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "exp1_main_convergence.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for problem in ["consistent", "inconsistent"]:
        for method in methods:
            runs = all_runs[problem][method.name]
            finals = final_values(runs, "normal")
            times = np.asarray([total_runtime(r) for r in runs])
            dual = np.asarray([np.mean(r["dual_nit"]) for r in runs])
            rows.append([
                problem, method.name,
                f"{np.mean(finals):.2e}", f"{np.std(finals, ddof=1):.1e}",
                f"{np.mean(times):.3f}", f"{np.std(times, ddof=1):.3f}",
                f"{np.mean(dual):.2f}",
            ])
    write_csv(
        RESULTS / "exp1_angle_cost.csv",
        ["problem", "method", "final_normal_mean", "final_normal_std", "time_mean", "time_std", "avg_dual_iterations"],
        rows,
    )
    with open(OUT / "exp1_angle_cost_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\small\n")
        f.write("\\caption{Five-seed convergence and cost comparison.}\n\\label{tab:main-convergence}\n")
        f.write("\\begin{tabular}{llrrr}\n\\hline\n")
        f.write("Problem & Method & Final normal residual & Time (s) & Avg. dual it.\\\\\n\\hline\n")
        for row in rows:
            f.write(f"{row[0]} & {row[1]} & {row[2]}$\\pm${row[3]} & {row[4]}$\\pm${row[5]} & {row[6]}\\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    return all_runs


def run_experiment_2(seeds=range(5)):
    methods = [Method("recent q=1", "recent", 1)]
    for rule in ["recent", "violated", "angle"]:
        for q in [2, 5, 10, 20]:
            methods.append(Method(f"{rule} q={q}", rule, q))
    runs = {m.name: [] for m in methods}
    for seed in seeds:
        A, b, _, tau = make_gaussian_problem(2000 + seed, 80, 240, inconsistent=False, sparsity=16)
        L = spectral_lipschitz(A)
        for method in methods:
            print(f"exp2 seed={seed}: {method.name}", flush=True)
            runs[method.name].append(run_algorithm(A, b, tau, method, 180, L=L))

    rows = []
    for method in methods:
        finals = final_values(runs[method.name], "normal")
        times = np.asarray([total_runtime(r) for r in runs[method.name]])
        dual = np.asarray([np.mean(r["dual_nit"]) for r in runs[method.name]])
        rows.append([
            method.rule, method.q,
            f"{np.mean(finals):.2e}", f"{np.std(finals, ddof=1):.1e}",
            f"{np.mean(times):.3f}", f"{np.std(times, ddof=1):.3f}",
            f"{np.mean(dual):.2f}",
        ])
    write_csv(
        RESULTS / "exp2_q_tradeoff.csv",
        ["rule", "q", "final_normal_mean", "final_normal_std", "time_mean", "time_std", "avg_dual_iterations"],
        rows,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    plot_styles = [
        ("recent", "#E69F00", "o", "-"),
        ("violated", "#009E73", "s", "--"),
        ("angle", "#CC79A7", "^", "-."),
    ]
    for rule, color, marker, linestyle in plot_styles:
        subset = [row for row in rows if row[0] == rule]
        if rule != "recent":
            baseline = next(row for row in rows if row[1] == 1)
            subset = [baseline] + subset
        qvals = [int(row[1]) for row in subset]
        axes[0].semilogy(
            qvals, [float(row[2]) for row in subset], marker=marker,
            linestyle=linestyle, color=color, label=rule,
        )
        axes[1].plot(
            qvals, [float(row[4]) for row in subset], marker=marker,
            linestyle=linestyle, color=color, label=rule,
        )
    axes[0].set_ylabel("final normal residual")
    axes[1].set_ylabel("runtime (s)")
    for ax in axes:
        ax.set_xlabel("memory size q")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp2_q_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(OUT / "exp2_q_tradeoff_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\small\n")
        f.write("\\caption{Five-seed memory-size tradeoff on consistent problems.}\n\\label{tab:q-tradeoff}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\hline\n")
        f.write("Rule & $q$ & Final normal residual & Time (s) & Avg. dual it.\\\\\n\\hline\n")
        for row in rows:
            f.write(f"{row[0]} & {row[1]} & {row[2]}$\\pm${row[3]} & {row[4]}$\\pm${row[5]} & {row[6]}\\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    return runs


def run_experiment_3(seeds=range(10)):
    methods = COMPARISON_METHODS
    results = {problem: {m.name: [] for m in methods} for problem in ["consistent", "inconsistent"]}
    for problem in results:
        for seed in seeds:
            A, b, _, tau = make_gaussian_problem(
                3000 + 10 * seed + (problem == "inconsistent"),
                60, 180, inconsistent=(problem == "inconsistent"), sparsity=12,
            )
            xbar, _ = solve_bilevel_reference(A, b, tau)
            L = spectral_lipschitz(A)
            for method in methods:
                print(f"exp3 {problem} seed={seed}: {method.name}", flush=True)
                results[problem][method.name].append(
                    run_algorithm(
                        A, b, tau, method, 80, xbar=xbar, L=L,
                        track_theta=(problem == "consistent" and method.name == "violated q=5"),
                    )
                )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    styles = {
        "one-cut": {"linestyle": "-", "marker": None},
        "recent q=5": {"linestyle": "--", "marker": "o"},
        "violated q=5": {"linestyle": "-.", "marker": "s"},
        "angle q=5": {"linestyle": ":", "marker": "^"},
    }
    for ax, problem in zip(axes, ["consistent", "inconsistent"]):
        for method in methods:
            plot_log10_band(
                ax, results[problem][method.name], "energy_rel", method.name,
                COLORS[method.name], linewidth=1.9, markevery=10,
                **styles[method.name],
            )
        ax.set_title(f"{problem} least squares")
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\log_{10}(\mathcal{E}_k/\mathcal{E}_0)$")
        ax.set_ylim(-9.4, -0.4)
        ax.set_yticks(np.arange(-9, 0, 1))
        ax.grid(True, which="major", color="#D0D0D0", linewidth=0.7)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp3_linear_convergence.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for problem in ["consistent", "inconsistent"]:
        for method in methods:
            runs = results[problem][method.name]
            finals = final_values(runs, "energy_rel")
            qvals = np.asarray([tail_factor(r) for r in runs])
            rows.append([
                problem, method.name,
                f"{np.mean(finals):.2e}", f"{np.std(finals, ddof=1):.1e}",
                f"{np.mean(qvals):.4f}", f"{np.std(qvals, ddof=1):.4f}",
            ])
    write_csv(
        RESULTS / "exp3_linear_convergence.csv",
        ["problem", "method", "final_energy_mean", "final_energy_std", "tail_factor_30_55_mean", "tail_factor_30_55_std"],
        rows,
    )
    with open(OUT / "exp3_linear_convergence_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\small\n")
        f.write("\\caption{Ten-seed Bregman-energy behavior; mean $\\pm$ standard deviation.  Tail factors use iterations 30--55, before the numerical energy floor.}\n\\label{tab:linear-convergence}\n")
        f.write("\\begin{tabular}{llrr}\n\\hline\n")
        f.write("Problem & Method & Final energy ratio & Tail factor (30--55)\\\\\n\\hline\n")
        for row in rows:
            f.write(f"{row[0]} & {row[1]} & {row[2]}$\\pm${row[3]} & {row[4]}$\\pm${row[5]}\\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    return results


def run_experiment_4(linear_results):
    runs = linear_results["consistent"]["violated q=5"]
    # Keep the original 55-iteration theta diagnostic even though Experiment 3
    # now continues to 80 iterations to separate its energy curves.
    theta_values = stack_metric(runs, "theta")[:, :55]
    bound_values = stack_metric(runs, "theta_lb")[:, :55]
    valid = np.any(np.isfinite(theta_values), axis=0) & np.any(np.isfinite(bound_values), axis=0)
    theta_values, bound_values = theta_values[:, valid], bound_values[:, valid]
    theta_mean = np.nanmean(theta_values, axis=0)
    theta_std = np.nanstd(theta_values, axis=0, ddof=1)
    bound_mean = np.nanmean(bound_values, axis=0)
    bound_std = np.nanstd(bound_values, axis=0, ddof=1)
    x = np.arange(1, len(theta_mean) + 1)
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    ax.plot(x, theta_mean, color="#009E73", linewidth=1.9, label=r"observed $\theta_k$")
    ax.fill_between(x, theta_mean - theta_std, theta_mean + theta_std, color="#009E73", alpha=0.15, linewidth=0)
    ax.plot(x, bound_mean, "--", color="#D55E00", linewidth=1.7, label="retained-cut lower bound")
    ax.fill_between(x, bound_mean - bound_std, bound_mean + bound_std, color="#D55E00", alpha=0.10, linewidth=0)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("memory gain")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp4_theta_bound.pdf", bbox_inches="tight")
    plt.close(fig)


def load_cached_theta_runs(path=RESULTS / "submission_experiment_data.npz", seeds=range(10)):
    """Load Experiment 3 theta diagnostics for figure-only reproduction."""
    archive = np.load(path)
    runs = []
    for seed in seeds:
        prefix = f"exp3__consistent__violated_q5__seed{seed}"
        runs.append({
            "theta": archive[f"{prefix}__theta"],
            "theta_lb": archive[f"{prefix}__theta_lb"],
        })
    return {"consistent": {"violated q=5": runs}}


def run_experiment_5(seeds=range(5)):
    results = {name: [] for name in ["fixed PG", "certified", "high accuracy"]}
    method = MAIN_METHODS[2]
    for seed in seeds:
        A, b, _, tau = make_gaussian_problem(5000 + seed, 60, 180, inconsistent=False, sparsity=12)
        xbar, _ = solve_bilevel_reference(A, b, tau)
        L = spectral_lipschitz(A)
        configs = [
            ("fixed PG", "fixed"),
            ("certified", "certified"),
            ("high accuracy", "accurate"),
        ]
        for name, mode in configs:
            print(f"exp5 seed={seed}: {name}", flush=True)
            results[name].append(
                run_algorithm(A, b, tau, method, 80, xbar=xbar, L=L, solve_mode=mode, fixed_tol=1e-3)
            )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for name in results:
        plot_mean_band(axes[0], results[name], "energy_rel", name, COLORS[name])
        plot_mean_band(axes[1], results[name], "eps_hat", name, COLORS[name])
    eta_mean, _ = mean_std(results["certified"], "eta")
    axes[1].semilogy(np.arange(1, len(eta_mean) + 1), eta_mean, "k--", linewidth=1.2, label=r"budget $\eta_k$")
    axes[0].set_ylabel(r"$\mathcal{E}_k/\mathcal{E}_0$")
    axes[1].set_ylabel(r"$\widehat\varepsilon_k$")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(True, which="major", color="#D0D0D0", linewidth=0.7)
        ax.grid(False, which="minor")
        ax.tick_params(axis="y", which="minor", length=2.0)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp5_inexact_certificate.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = []
    eta_runs = results["certified"]
    for name, runs in results.items():
        finals = final_values(runs, "energy_rel")
        max_eps = np.asarray([np.max(r["eps_hat"]) for r in runs])
        violations = []
        for idx, run in enumerate(runs):
            eta = eta_runs[idx]["eta"][: len(run["eps_hat"])]
            violations.append(np.sum(run["eps_hat"][: len(eta)] > eta * (1 + 1e-8)))
        dual = np.asarray([np.mean(r["dual_nit"]) for r in runs])
        rows.append([
            name, f"{np.mean(finals):.2e}", f"{np.std(finals, ddof=1):.1e}",
            f"{np.mean(max_eps):.2e}", f"{np.mean(violations):.1f}", f"{np.mean(dual):.2f}",
        ])
    write_csv(
        RESULTS / "exp5_inexact_certificate.csv",
        ["inner_rule", "final_energy_mean", "final_energy_std", "max_epsilon_mean", "budget_violations_mean", "avg_dual_iterations"],
        rows,
    )
    with open(OUT / "exp5_inexact_certificate_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\small\n")
        f.write("\\caption{Five-seed comparison of inner stopping rules.}\n\\label{tab:inexact-certificate}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrrr}\n\\hline\n")
        f.write("Inner rule & Final energy ratio & Max. certificate & Budget violations & Avg. dual it.\\\\\n\\hline\n")
        for row in rows:
            f.write(f"{row[0]} & {row[1]}$\\pm${row[2]} & {row[3]} & {row[4]} & {row[5]}\\\\\n")
        f.write("\\hline\n\\end{tabular}%\n}\n\\end{table}\n")
    return results


def make_sparse_problem(seed, m, n, row_nnz=30):
    rng = np.random.default_rng(seed)
    nnz = m * row_nnz
    rows = rng.integers(0, m, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    data = rng.standard_normal(nnz)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsc()
    norms = np.sqrt(np.asarray(A.power(2).sum(axis=0)).reshape(-1))
    norms[norms == 0] = 1.0
    A = (A @ sparse.diags(1.0 / norms)).tocsr()
    x_true = np.zeros(n)
    support = rng.choice(n, size=max(20, n // 100), replace=False)
    x_true[support] = rng.normal(size=len(support))
    b = as_vector(A @ x_true)
    tau = 0.05 * max(float(np.max(np.abs(as_vector(A.T @ b)))), 1e-8)
    return A, b, tau


def run_experiment_6(seeds=range(3)):
    sizes = [(1000, 3000), (2000, 6000), (4000, 12000)]
    methods = COMPARISON_METHODS
    records = []
    raw_runs = {}
    for m, n in sizes:
        for seed in seeds:
            A, b, tau = make_sparse_problem(6000 + n + seed, m, n)
            L = spectral_lipschitz(A)
            for method in methods:
                print(f"exp6 m={m} n={n} seed={seed}: {method.name}", flush=True)
                run = run_algorithm(A, b, tau, method, 100, L=L)
                raw_runs[(m, n, seed, method.name)] = run
                records.append({
                    "m": m, "n": n, "seed": seed, "method": method.name,
                    "time_per_iter": total_runtime(run) / len(run["iter"]),
                    "final_normal": run["normal"][-1],
                    "total_time": total_runtime(run),
                })

    rows = []
    for m, n in sizes:
        for method in methods:
            subset = [r for r in records if r["n"] == n and r["method"] == method.name]
            tpi = np.asarray([r["time_per_iter"] for r in subset])
            final = np.asarray([r["final_normal"] for r in subset])
            total = np.asarray([r["total_time"] for r in subset])
            rows.append([
                m, n, method.name,
                f"{np.mean(tpi):.4f}", f"{np.std(tpi, ddof=1):.4f}",
                f"{np.mean(final):.2e}", f"{np.std(final, ddof=1):.1e}",
                f"{np.mean(total):.2f}",
            ])
    write_csv(
        RESULTS / "exp6_large_scale.csv",
        ["m", "n", "method", "time_per_iteration_mean", "time_per_iteration_std", "final_normal_mean", "final_normal_std", "total_time_mean"],
        rows,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    styles = {
        "one-cut": ("o", "-"),
        "recent q=5": ("o", "-"),
        "violated q=5": ("s", "--"),
        "angle q=5": ("^", ":"),
    }
    for method in methods:
        subset = [row for row in rows if row[2] == method.name]
        ns = [int(row[1]) for row in subset]
        marker, linestyle = styles[method.name]
        axes[0].plot(ns, [float(row[3]) for row in subset], marker=marker, linestyle=linestyle, color=COLORS[method.name], label=method.name)
        axes[1].semilogy(ns, [float(row[5]) for row in subset], marker=marker, linestyle=linestyle, color=COLORS[method.name], label=method.name)
    axes[0].set_ylabel("time per iteration (s)")
    axes[1].set_ylabel("normal residual after 100 iterations")
    for ax in axes:
        ax.set_xlabel("dimension n")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp6_large_scale.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(OUT / "exp6_large_scale_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n\\small\n")
        f.write("\\caption{Three-seed sparse large-scale comparison after 100 iterations.}\n\\label{tab:large-scale}\n")
        f.write("\\begin{tabular}{rrlrrr}\n\\hline\n")
        f.write("$m$ & $n$ & Method & Time/iter. & Final normal residual & Total time\\\\\n\\hline\n")
        for row in rows:
            f.write(f"{row[0]} & {row[1]} & {row[2]} & {row[3]}$\\pm${row[4]} & {row[5]}$\\pm${row[6]} & {row[7]}\\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    return raw_runs


def save_raw_npz(groups, *, merge=False):
    data = {}
    output_path = RESULTS / "submission_experiment_data.npz"
    if merge and output_path.exists():
        with np.load(output_path) as archive:
            replaced = tuple(f"{group}__" for group in groups)
            data.update({key: archive[key] for key in archive.files if not key.startswith(replaced)})
    for group, value in groups.items():
        if isinstance(value, dict):
            stack = [(str(group), value)]
            while stack:
                prefix, node = stack.pop()
                for key, child in node.items():
                    clean = str(key).replace(" ", "_").replace("=", "")
                    if isinstance(child, dict):
                        stack.append((f"{prefix}__{clean}", child))
                    elif isinstance(child, list):
                        for idx, run in enumerate(child):
                            for metric, array in run.items():
                                data[f"{prefix}__{clean}__seed{idx}__{metric}"] = array
                    elif isinstance(child, np.ndarray):
                        data[f"{prefix}__{clean}"] = child
        
    np.savez_compressed(output_path, **data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure5-only", action="store_true",
        help="redraw paper Figure 5 from the saved ten-seed raw data",
    )
    parser.add_argument(
        "--experiment3-only", action="store_true",
        help="rerun Experiment 3 and regenerate paper Figures 4 and 5",
    )
    parser.add_argument(
        "--angle-update-only", action="store_true",
        help="rerun only Experiments 2, 3, and 6 after adding angle filtering",
    )
    args = parser.parse_args()
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 130,
    })
    if args.figure5_only:
        run_experiment_4(load_cached_theta_runs())
        print("paper Figure 5 regenerated from cached raw data", flush=True)
        return
    if args.experiment3_only:
        exp3 = run_experiment_3()
        run_experiment_4(exp3)
        save_raw_npz({"exp3": exp3}, merge=True)
        print("Experiment 3 and paper Figures 4--5 regenerated", flush=True)
        return
    if args.angle_update_only:
        exp2 = run_experiment_2()
        exp3 = run_experiment_3()
        exp6 = run_experiment_6()
        save_raw_npz({"exp2": exp2, "exp3": exp3, "exp6": exp6}, merge=True)
        print("Experiments 2, 3, and 6 regenerated with angle filtering", flush=True)
        return
    exp1 = run_experiment_1()
    exp2 = run_experiment_2()
    exp3 = run_experiment_3()
    run_experiment_4(exp3)
    exp5 = run_experiment_5()
    exp6 = run_experiment_6()
    save_raw_npz({"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp5": exp5, "exp6": exp6})
    print("submission experiments complete", flush=True)


if __name__ == "__main__":
    main()
