"""Final long-horizon Experiment 5.5 with at most five multi-cut BB steps."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

import run_tuning as tune


HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "final_exp55"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

HORIZON = 2000
SEEDS = range(20)
SIZES = ((1000, 3000), (2000, 6000), (4000, 12000))
METHODS = ("one-cut", "recent", "violated", "angle", "adaptive")
CHECKPOINTS = (100, 200, 500, 1000, 2000)
COLORS = {
    "one-cut": "#0072B2", "recent": "#E69F00", "violated": "#009E73",
    "angle": "#CC79A7", "adaptive": "#D55E00",
}
STYLES = {
    "one-cut": "-", "recent": "--", "violated": "-.",
    "angle": ":", "adaptive": (0, (4, 1, 1, 1)),
}


def config(method):
    if method == "one-cut":
        return None
    return tune.Config(
        method=method, q=4, p=8, inner_max=5,
        trigger=0.02 if method == "adaptive" else 0.05,
    )


def pad(array):
    array = np.asarray(array, dtype=float)
    return np.pad(array, (0, HORIZON - len(array)), mode="edge")


def quartiles(values):
    return tuple(float(np.quantile(values, q)) for q in (0.25, 0.5, 0.75))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    tune.OUTER_ITERATIONS = HORIZON
    tune.GRAD2_STOP_RATIO = 1e-32
    tune.ONE_CUT_TOL = 1e-24
    runs = {}
    raw_rows = []

    with threadpool_limits(1):
        for m, n in SIZES:
            for seed in SEEDS:
                A, b, tau = tune.base.make_sparse_problem(10100 + n + seed, m, n)
                L = tune.ad.spectral_lipschitz(A)
                for method in METHODS:
                    path = OUT / f"m{m}_n{n}__seed{seed:02d}__{method}.npz"
                    result = tune.run_method(
                        A, b, tau, L, config(method), return_history=True
                    )
                    saved = {
                        "residual": pad(result["normal_history"]),
                        "objective": pad(result["objective_history"]),
                        "runtime": pad(result["time_history"]),
                        "pgn": pad(result["inner_pgn_history"]),
                    }
                    np.savez_compressed(path, **saved)
                    runs[(m, n, seed, method)] = saved
                    row = {
                        "m": m, "n": n, "seed": seed, "method": method,
                        "final_residual": saved["residual"][-1],
                        "best_residual": np.min(saved["residual"]),
                        "best_iteration": np.argmin(saved["residual"]) + 1,
                        "final_objective": saved["objective"][-1],
                        "core_time_s": saved["runtime"][-1],
                        "tail100_pgn_median": np.median(saved["pgn"][-100:]),
                    }
                    for checkpoint in CHECKPOINTS:
                        row[f"residual_{checkpoint}"] = saved["residual"][checkpoint - 1]
                    raw_rows.append(row)
                    print(
                        f"m={m} n={n} seed={seed:02d} {method:8s} "
                        f"r200={row['residual_200']:.3e} "
                        f"r2000={row['final_residual']:.3e} "
                        f"t={row['core_time_s']:.2f}s", flush=True,
                    )

    write_csv(OUT / "raw.csv", raw_rows)
    summary_rows = []
    for m, n in SIZES:
        for method in METHODS:
            group = [r for r in raw_rows if r["n"] == n and r["method"] == method]
            rq1, rmed, rq3 = quartiles([r["final_residual"] for r in group])
            oq1, omed, oq3 = quartiles([r["final_objective"] for r in group])
            tq1, tmed, tq3 = quartiles([r["core_time_s"] for r in group])
            summary = {
                "m": m, "n": n, "method": method, "instances": len(group),
                "final_residual_q1": rq1, "final_residual_median": rmed,
                "final_residual_q3": rq3, "objective_q1": oq1,
                "objective_median": omed, "objective_q3": oq3,
                "core_time_q1_s": tq1, "core_time_median_s": tmed,
                "core_time_q3_s": tq3,
                "best_residual_median": float(np.median([r["best_residual"] for r in group])),
                "best_iteration_median": float(np.median([r["best_iteration"] for r in group])),
                "tail100_pgn_median": float(np.median([r["tail100_pgn_median"] for r in group])),
            }
            for checkpoint in CHECKPOINTS:
                summary[f"residual_{checkpoint}_median"] = float(np.median([
                    r[f"residual_{checkpoint}"] for r in group
                ]))
            summary_rows.append(summary)
    write_csv(OUT / "summary.csv", summary_rows)

    figure_specs = (
        ("residual_time.png", "time", "residual", "Core time (s)", "Relative residual", True),
        ("residual_iteration.png", "iteration", "residual", "Outer iteration", "Relative residual", True),
        ("objective_iteration.png", "iteration", "objective", "Outer iteration", "Objective value", False),
    )
    for filename, xkind, metric, xlabel, ylabel, use_iqr in figure_specs:
        fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.25), constrained_layout=True)
        for ax, (m, n) in zip(axes, SIZES):
            for method in METHODS:
                group = [runs[(m, n, seed, method)] for seed in SEEDS]
                values = np.vstack([r[metric] for r in group])
                median = np.median(values, axis=0)
                q1 = np.quantile(values, 0.25, axis=0)
                q3 = np.quantile(values, 0.75, axis=0)
                x = (
                    np.median(np.vstack([r["runtime"] for r in group]), axis=0)
                    if xkind == "time" else np.arange(1, HORIZON + 1)
                )
                ax.plot(x, np.maximum(median, 1e-20) if metric == "residual" else median,
                        color=COLORS[method], linestyle=STYLES[method],
                        linewidth=1.35, label=method)
                if use_iqr:
                    ax.fill_between(x, np.maximum(q1, 1e-20), np.maximum(q3, 1e-20),
                                    color=COLORS[method], alpha=0.08, linewidth=0)
            if metric == "residual":
                ax.set_yscale("log")
            ax.set_title(rf"$m={m},\ n={n}$")
            ax.set_xlabel(xlabel)
            ax.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
        fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    latex = [
        r"\begin{tabular}{rrlccc}", r"\hline",
        r"$m$ & $n$ & Method & Final residual & Objective & Core time (s)\\", r"\hline",
    ]
    for row in summary_rows:
        latex.append(
            f"{row['m']} & {row['n']} & {row['method']} & "
            f"${row['final_residual_median']:.2e}$ & "
            f"${row['objective_median']:.4f}$ & {row['core_time_median_s']:.3f}\\\\"
        )
    latex.extend([r"\hline", r"\end{tabular}"])
    (OUT / "main_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
