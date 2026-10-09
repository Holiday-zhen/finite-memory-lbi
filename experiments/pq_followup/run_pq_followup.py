"""Parameter-size follow-up for manuscript Experiments 5.5 and 5.6.

This driver preserves the numerical methods used by the archived experiment
code and writes all follow-up outputs to a separate results directory.
"""

from __future__ import annotations

import argparse
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
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl"))

METHOD_NAMES = ("one-cut", "recent", "violated", "angle", "adaptive")
COLORS = {
    "one-cut": "#0072B2",
    "recent": "#E69F00",
    "violated": "#009E73",
    "angle": "#CC79A7",
    "adaptive": "#D55E00",
}
STYLES = {
    "one-cut": "-",
    "recent": "--",
    "violated": "-.",
    "angle": ":",
    "adaptive": "-",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quartiles(values):
    values = np.asarray(values, dtype=float)
    return tuple(float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75))


def make_ad_methods(ad, q, p, trigger=0.05, solver="lbfgsb"):
    return [
        ad.Method("one-cut", "recent", qmax=1, pmax=1, solver=solver),
        ad.Method("recent", "recent", qmax=q, pmax=p, solver=solver),
        ad.Method("violated", "violated", qmax=q, pmax=p, solver=solver),
        ad.Method("angle", "angle", qmax=q, pmax=p, solver=solver),
        ad.Method(
            "adaptive", "angle", qmax=q, pmax=p, solver=solver,
            adaptive=True, c_trig=trigger,
        ),
    ]


def run_scale() -> None:
    engine = (
        ROOT / "src" / "diagnostics"
        / "revision_expanded_20260904" / "engine"
    )
    sys.path.insert(0, str(engine))
    import experiments_adaptive_revision as ad
    import experiments_bounded as base

    configs = (("q2p4", 2, 4), ("q3p6", 3, 6))
    runs: dict[tuple[str, int, str], dict] = {}
    rows = []
    with threadpool_limits(1):
        for seed in range(20):
            A, b, tau = base.make_sparse_problem(10100 + 12000 + seed, 4000, 12000)
            L = ad.spectral_lipschitz(A)
            for ci, (label, q, p) in enumerate(configs):
                methods = make_ad_methods(ad, q, p)
                shift = (seed + ci) % len(methods)
                for method in methods[shift:] + methods[:shift]:
                    print(f"SCALE seed={seed:02d} {label} {method.name}", flush=True)
                    run = ad._timed_run(A, b, tau, method, 100, L=L, repeats=3)
                    runs[(label, seed, method.name)] = run
                    np.savez_compressed(
                        OUT / f"scale__{label}__{seed}__{method.name}.npz", **run
                    )

    for label, q, p in configs:
        for name in METHOD_NAMES:
            group = [runs[(label, seed, name)] for seed in range(20)]
            rq1, rmed, rq3 = quartiles([r["normal"][-1] for r in group])
            tq1, tmed, tq3 = quartiles([r["runtime"][-1] for r in group])
            rows.append({
                "configuration": label, "q": 1 if name == "one-cut" else q,
                "p": 1 if name == "one-cut" else p, "method": name,
                "residual_q1": rq1, "residual_median": rmed,
                "residual_q3": rq3, "runtime_q1_s": tq1,
                "runtime_median_s": tmed, "runtime_q3_s": tq3,
                "inner_success_rate": float(np.mean([
                    np.mean(r["inner_success"]) for r in group
                ])),
            })
    write_csv(OUT / "scale_pq_summary.csv", rows)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
    for row_index, (label, q, p) in enumerate(configs):
        for name in METHOD_NAMES:
            group = [runs[(label, seed, name)] for seed in range(20)]
            values = np.vstack([r["normal"] for r in group])
            med = np.median(values, axis=0)
            q1 = np.quantile(values, 0.25, axis=0)
            q3 = np.quantile(values, 0.75, axis=0)
            k = np.arange(1, len(med) + 1)
            axes[row_index, 0].semilogy(
                k, med, STYLES[name], color=COLORS[name], label=name,
            )
            axes[row_index, 0].fill_between(k, q1, q3, color=COLORS[name], alpha=0.08)
            stop = min(float(r["runtime"][-1]) for r in group)
            start = max(float(r["runtime"][0]) for r in group)
            grid = np.linspace(start, stop, 250)
            time_values = np.vstack([
                np.interp(grid, r["runtime"], r["normal"]) for r in group
            ])
            axes[row_index, 1].semilogy(
                grid, np.median(time_values, axis=0), STYLES[name],
                color=COLORS[name], label=name,
            )
        axes[row_index, 0].set_title(f"(q,p)=({q},{p}): iteration")
        axes[row_index, 1].set_title(f"(q,p)=({q},{p}): time")
        axes[row_index, 0].set_xlabel("outer iteration")
        axes[row_index, 1].set_xlabel("time (s)")
        for ax in axes[row_index]:
            ax.set_ylabel("normalized gradient residual")
            ax.grid(True, color="#dddddd", linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.91,
                        wspace=0.28, hspace=0.38)
    fig.savefig(OUT / "scale_pq_residual_time.png", dpi=240)
    plt.close(fig)
    print("SCALE COMPLETED", flush=True)


def run_bas1lp() -> None:
    module_dir = ROOT / "experiments" / "suitesparse_pilot"
    sys.path.insert(0, str(module_dir))
    import run_pilot as bp

    A, eligible = bp.load_and_normalize("bas1lp")
    L = bp.alg.spectral_lipschitz(A)
    b, tau = bp.make_rhs(A, eligible, seed=20260909)
    configs = (("one-cut", 1, 1), ("recent q=2", 2, 4),
               ("recent q=3", 3, 6), ("recent q=5", 5, 10))
    colors = ("#0072B2", "#D55E00", "#009E73", "#E69F00")
    stored = {}
    rows = []
    with threadpool_limits(1):
        for label, q, p in configs:
            method = "one-cut" if q == 1 else "recent"
            bp.Q = q
            bp.run_method(A, b, tau, L, method, max_iter=4, record=False)
            repeats = []
            for repeat in range(3):
                print(f"BAS1LP {label} repeat={repeat + 1}", flush=True)
                repeats.append(bp.run_method(A, b, tau, L, method, max_iter=5000))
            reference = repeats[0]
            width = min(len(r["iter"]) for r in repeats)
            hist = {key: value[:width].copy() for key, value in reference.items()}
            hist["runtime"] = np.median(
                np.vstack([r["runtime"][:width] for r in repeats]), axis=0
            )
            stored[label] = hist
            np.savez_compressed(OUT / f"bas1lp__q{q}p{p}.npz", **hist)
            rows.append({
                "label": label, "q": q, "p": p,
                "iterations": len(hist["iter"]),
                "final_outer_residual": float(hist["normal"][-1]),
                "runtime_median_s": float(hist["runtime"][-1]),
                "max_inner_residual": float(np.max(hist["pgn"])),
                "median_inner_residual": float(np.median(hist["pgn"])),
            })
    write_csv(OUT / "bas1lp_pq_summary.csv", rows)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for (label, q, p), color in zip(configs, colors):
        h = stored[label]
        axes[0].semilogy(h["runtime"], h["normal"], color=color, label=label)
        axes[1].semilogy(h["iter"], h["normal"], color=color, label=label)
        axes[2].semilogy(
            h["iter"], np.maximum(h["pgn"], 1e-18), color=color, label=label,
        )
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("outer residual")
    axes[0].set_title("residual vs time")
    axes[1].set_xlabel("outer iteration")
    axes[1].set_ylabel("outer residual")
    axes[1].set_title("outer residual")
    axes[2].set_xlabel("outer iteration")
    axes[2].set_ylabel("projected-gradient residual")
    axes[2].set_title("inner residual")
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.78, wspace=0.30)
    fig.savefig(OUT / "bas1lp_pq_residual_time_inner.png", dpi=240)
    plt.close(fig)
    print("BAS1LP COMPLETED", flush=True)


def strict_one_cut(problem, s, a, beta):
    def derivative(t):
        return float(beta - a @ problem.mirror(s - t * a))
    if derivative(0.0) >= 0.0:
        return np.zeros(1)
    upper = 1.0
    while derivative(upper) < 0.0 and upper < 1e14:
        upper *= 2.0
    return np.array([brentq(
        derivative, 0.0, upper, xtol=1e-18,
        rtol=4 * np.finfo(float).eps, maxiter=200,
    )])


def run_matrix_once(problem, method, hard, robust, max_iter=200):
    x = np.zeros(problem.dim)
    s = np.zeros(problem.dim)
    g0 = max(float(np.linalg.norm(problem.grad(x))), 1e-300)
    pool = []
    warm_by_cut = {}
    history = {key: [] for key in (
        "iter", "residual", "runtime", "pgn", "inner_nit", "memory_size",
    )}
    started = time.perf_counter()
    initial_one_step = None
    for k in range(max_iter):
        raw_a = problem.grad(x)
        scale = float(np.linalg.norm(raw_a))
        if scale == 0.0:
            for key in history:
                if key == "iter":
                    history[key].append(k + 1)
                elif key == "runtime":
                    history[key].append(time.perf_counter() - started)
                elif key == "residual":
                    history[key].append(0.0)
                else:
                    history[key].append(0.0 if key != "memory_size" else 1.0)
            continue
        raw_beta = float(raw_a @ x - (raw_a @ raw_a) / problem.lipschitz)
        a, beta = raw_a / scale, raw_beta / scale
        current = {"id": k, "a": a.copy(), "beta": beta}
        pool.append(current)
        pool = hard.prune_pool(pool, method.p, x)
        one_alpha = strict_one_cut(problem, s, a, beta)
        x_one = problem.mirror(s - one_alpha[0] * a)
        one_step = problem.omega(x_one) - problem.omega(x) - float(s @ (x_one - x))
        if initial_one_step is None and one_step > 0.0:
            initial_one_step = one_step
        if method.q == 1:
            selected = [current]
        elif method.adaptive:
            old = [cut for cut in pool if cut["id"] != k]
            required = max(
                old, key=lambda cut: hard.positive_violation(cut, x_one), default=None
            )
            delta = hard.positive_violation(required, x_one) if required else 0.0
            floor = max(1e-14, 1e-12 * (initial_one_step or 0.0))
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
            warm = np.asarray([warm_by_cut.get(i, 0.0) for i in ids])
            alpha, nit, pgn = robust.solve_dual(
                problem, s, G, betas, warm=warm, tol=1e-13
            )
        s = s - G @ alpha
        x = problem.mirror(s)
        warm_by_cut = {i: float(v) for i, v in zip(ids, alpha) if v > 1e-14}
        pool = hard.prune_pool(pool, method.p, x)
        live = {cut["id"] for cut in pool}
        warm_by_cut = {i: v for i, v in warm_by_cut.items() if i in live}
        history["iter"].append(k + 1)
        history["residual"].append(float(np.linalg.norm(problem.grad(x)) / g0))
        history["runtime"].append(time.perf_counter() - started)
        history["pgn"].append(float(pgn))
        history["inner_nit"].append(float(nit))
        history["memory_size"].append(float(len(selected)))
    return {key: np.asarray(value, dtype=float) for key, value in history.items()}


def run_matrix() -> None:
    diagnostics = ROOT / "src" / "diagnostics"
    sys.path.insert(0, str(diagnostics / "shutdown_hard_benchmark"))
    sys.path.insert(0, str(diagnostics / "precision_followup_20260904"))
    import run_experiment as hard
    import robust_inner as robust

    builders = {
        "column_block": lambda seed: hard.completion.build_problem(
            "column-block", seed, 0.50, 0.20
        ),
        "binary_digits": lambda seed: hard.real_matrix.build_problem(
            "Binary Digits pixel feasibility", seed, "heterogeneous", 0.50, 0.80
        ),
    }
    configs = (("q3p6", 3, 6), ("q2p4", 2, 4))
    runs: dict[tuple[str, str, int, str], dict] = {}
    rows = []
    with threadpool_limits(1):
        for problem_index, (problem_name, builder) in enumerate(builders.items()):
            for seed_index, seed in enumerate(range(100, 120)):
                problem = builder(seed)
                for ci, (label, q, p) in enumerate(configs):
                    methods = [
                        hard.Method("one-cut", "recent", 1, 1),
                        hard.Method("recent", "recent", q, p),
                        hard.Method("violated", "violated", q, p),
                        hard.Method("angle", "angle", q, p),
                        hard.Method("adaptive", "angle", q, p, True, 1e-3),
                    ]
                    shift = (problem_index + seed_index + ci) % len(methods)
                    for method in methods[shift:] + methods[:shift]:
                        print(
                            f"MATRIX {problem_name} seed={seed} {label} {method.name}",
                            flush=True,
                        )
                        run = run_matrix_once(problem, method, hard, robust)
                        runs[(label, problem_name, seed, method.name)] = run
                        np.savez_compressed(
                            OUT / f"matrix__{label}__{problem_name}__{seed}__{method.name}.npz",
                            **run,
                        )

    for label, q, p in configs:
        for problem_name in builders:
            for name in METHOD_NAMES:
                group = [
                    runs[(label, problem_name, seed, name)] for seed in range(100, 120)
                ]
                rq1, rmed, rq3 = quartiles([r["residual"][-1] for r in group])
                tq1, tmed, tq3 = quartiles([r["runtime"][-1] for r in group])
                rows.append({
                    "configuration": label, "problem": problem_name,
                    "q": 1 if name == "one-cut" else q,
                    "p": 1 if name == "one-cut" else p, "method": name,
                    "residual_q1": rq1, "residual_median": rmed,
                    "residual_q3": rq3, "runtime_q1_s": tq1,
                    "runtime_median_s": tmed, "runtime_q3_s": tq3,
                    "max_inner_residual": float(max(np.max(r["pgn"]) for r in group)),
                })
    write_csv(OUT / "matrix_pq_summary.csv", rows)

    for label, q, p in configs:
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8))
        for col, problem_name in enumerate(builders):
            for name in METHOD_NAMES:
                group = [
                    runs[(label, problem_name, seed, name)] for seed in range(100, 120)
                ]
                values = np.vstack([r["residual"] for r in group])
                med = np.median(values, axis=0)
                q1 = np.quantile(values, 0.25, axis=0)
                q3 = np.quantile(values, 0.75, axis=0)
                k = np.arange(1, len(med) + 1)
                axes[0, col].semilogy(k, np.maximum(med, 1e-19),
                                     STYLES[name], color=COLORS[name], label=name)
                axes[0, col].fill_between(
                    k, np.maximum(q1, 1e-19), np.maximum(q3, 1e-19),
                    color=COLORS[name], alpha=0.08,
                )
                stop = min(float(r["runtime"][-1]) for r in group)
                start = max(float(r["runtime"][0]) for r in group)
                grid = np.linspace(start, stop, 250)
                vals = np.vstack([
                    np.interp(grid, r["runtime"], r["residual"]) for r in group
                ])
                axes[1, col].semilogy(
                    grid, np.maximum(np.median(vals, axis=0), 1e-19),
                    STYLES[name], color=COLORS[name], label=name,
                )
            title = "Column-block completion" if col == 0 else "Binary Digits"
            axes[0, col].set_title(title)
            axes[0, col].set_xlabel("outer iteration")
            axes[1, col].set_xlabel("time (s)")
            for ax in axes[:, col]:
                ax.set_ylabel("relative gradient residual")
                ax.grid(True, color="#dddddd", linewidth=0.6)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
        fig.suptitle(f"(q,p)=({q},{p})", y=0.945, fontsize=10)
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.86,
                            wspace=0.27, hspace=0.34)
        fig.savefig(OUT / f"matrix_{label}_residual_time.png", dpi=240)
        plt.close(fig)
    print("MATRIX COMPLETED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=("scale", "bas1lp", "matrix"))
    args = parser.parse_args()
    if args.study == "scale":
        run_scale()
    elif args.study == "bas1lp":
        run_bas1lp()
    else:
        run_matrix()


if __name__ == "__main__":
    main()
