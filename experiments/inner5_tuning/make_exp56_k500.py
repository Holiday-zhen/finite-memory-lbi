"""Create the 500-iteration Experiment 5.6 figures from verified long traces."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "results" / "final_exp56"
OUT = HERE / "results" / "final_exp56_k500"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

K = 500
PROBLEMS = ("column_block", "binary_digits")
METHODS = ("one-cut", "recent", "violated", "angle", "adaptive")
COLORS = {
    "one-cut": "#0072B2", "recent": "#E69F00", "violated": "#009E73",
    "angle": "#CC79A7", "adaptive": "#D55E00",
}
STYLES = {
    "one-cut": "-", "recent": "--", "violated": "-.",
    "angle": ":", "adaptive": (0, (4, 1, 1, 1)),
}


def load_group(problem, method):
    paths = sorted(SOURCE.glob(f"{problem}__*__{method}.npz"))
    if len(paths) != 20:
        raise RuntimeError(f"Expected 20 runs for {problem}/{method}, found {len(paths)}")
    return [
        {key: np.asarray(value)[:K] for key, value in np.load(path).items()}
        for path in paths
    ]


def q(values):
    return tuple(float(np.quantile(values, level)) for level in (0.25, 0.5, 0.75))


def main():
    groups = {
        (problem, method): load_group(problem, method)
        for problem in PROBLEMS for method in METHODS
    }
    rows = []
    for problem in PROBLEMS:
        for method in METHODS:
            group = groups[(problem, method)]
            rq1, rmed, rq3 = q([run["residual"][-1] for run in group])
            tq1, tmed, tq3 = q([run["runtime"][-1] for run in group])
            rows.append({
                "problem": problem, "method": method, "instances": len(group),
                "iteration": K, "residual_q1": rq1, "residual_median": rmed,
                "residual_q3": rq3, "core_time_q1_s": tq1,
                "core_time_median_s": tmed, "core_time_q3_s": tq3,
            })
    with (OUT / "summary_k500.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    specs = (
        ("residual_time_k500.png", True, "Core time (s)"),
        ("residual_iteration_k500.png", False, "Outer iteration"),
    )
    for filename, versus_time, xlabel in specs:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.45), constrained_layout=True)
        for ax, problem in zip(axes, PROBLEMS):
            for method in METHODS:
                group = groups[(problem, method)]
                residual = np.vstack([run["residual"] for run in group])
                median = np.median(residual, axis=0)
                q1 = np.quantile(residual, 0.25, axis=0)
                q3 = np.quantile(residual, 0.75, axis=0)
                x = (
                    np.median(np.vstack([run["runtime"] for run in group]), axis=0)
                    if versus_time else np.arange(1, K + 1)
                )
                ax.plot(x, np.maximum(median, 1e-20), color=COLORS[method],
                        linestyle=STYLES[method], linewidth=1.45, label=method)
                ax.fill_between(x, np.maximum(q1, 1e-20), np.maximum(q3, 1e-20),
                                color=COLORS[method], alpha=0.09, linewidth=0)
            ax.set_yscale("log")
            ax.set_title(
                "Column-block completion" if problem == "column_block"
                else "Binary Digits feasibility"
            )
            ax.set_xlabel(xlabel)
            ax.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel("Relative residual")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
        fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    latex = [
        r"\begin{tabular}{llcc}", r"\hline",
        r"Problem & Method & Residual at 500 & Core time (s)\\", r"\hline",
    ]
    for row in rows:
        latex.append(
            f"{row['problem'].replace('_', ' ')} & {row['method']} & "
            f"${row['residual_median']:.2e}$ & {row['core_time_median_s']:.3f}\\\\"
        )
    latex.extend([r"\hline", r"\end{tabular}"])
    (OUT / "table_k500.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
