"""Focused matrix-completion comparison for finite-memory cut rules.

This diagnostic keeps the problem size modest so that q=2 and q=3 can be
compared across fresh random instances.  It reports iteration count, wall time,
observed/full reconstruction error, residual, average active cuts, and the
fraction of iterations using more than one cut.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "four_virtual_examples"
sys.path.insert(0, str(BASE))
import run_virtual_examples as exp


SEEDS = [10, 11, 12, 13, 14]
OBSERVATION = 0.38
BLOCKED = True
TAU_RATIO = 0.02


def make_matrix_completion(seed: int) -> exp.Problem:
    rng = np.random.default_rng(40000 + seed)
    nr, nc, rank = 18, 18, 3
    left = rng.normal(size=(nr, rank))
    right = rng.normal(size=(nc, rank))
    truth = left @ right.T / np.sqrt(rank)
    mask = rng.random((nr, nc)) < OBSERVATION
    if BLOCKED:
        start = int(rng.integers(4, 11))
        mask[:, start : start + 4] = False
        # Keep the instance identifiable while retaining a structured gap.
        mask[:, start : start + 4] |= rng.random((nr, 4)) < 0.08
    observed = np.where(mask, truth, 0.0)

    def grad(x):
        matrix = x.reshape(nr, nc)
        return np.where(mask, matrix - observed, 0.0).reshape(-1)

    g0 = grad(np.zeros(nr * nc)).reshape(nr, nc)
    tau = TAU_RATIO * np.linalg.norm(g0, 2)
    mirror = lambda s: exp.svt_flat(s, tau, (nr, nc))
    omega = lambda x: float(
        tau * np.linalg.norm(x.reshape(nr, nc), ord="nuc") + 0.5 * (x @ x)
    )
    omega_star = lambda s: float(0.5 * np.linalg.norm(mirror(s)) ** 2)
    problem = exp.Problem(
        "matrix completion", nr * nc, grad, 1.0, tau,
        mirror, omega, omega_star, 100,
    )
    # Metadata used only by this diagnostic for reconstruction-error reporting.
    problem.truth = truth
    problem.mask = mask
    problem.observed = observed
    problem.shape = (nr, nc)
    return problem


def methods_for_q(q: int) -> list[exp.Method]:
    return [
        exp.Method("one-cut", "recent", q=1, p=1),
        exp.Method("recent", "recent", q=q, p=2 * q),
        exp.Method("violated", "violated", q=q, p=2 * q),
        exp.Method("angle", "angle", q=q, p=2 * q),
        exp.Method("adaptive", "angle", adaptive=True, q=q, p=2 * q, trigger=0.01),
    ]


def run_one(problem: exp.Problem, method: exp.Method, seed: int, q_label: int) -> dict:
    started = time.perf_counter()
    result = exp.run(problem, method)
    elapsed = time.perf_counter() - started
    final = result["x_final"].reshape(problem.shape)
    truth = problem.truth
    observed_rmse = float(np.linalg.norm((final - truth)[problem.mask]) / np.sqrt(np.sum(problem.mask)))
    full_rmse = float(np.linalg.norm(final - truth) / np.sqrt(truth.size))
    residual = float(result["residual"][-1]) if len(result["residual"]) else np.nan
    mean_q = float(np.mean(result["q"])) if len(result["q"]) else np.nan
    multi_share = float(np.mean(result["q"] > 1)) if len(result["q"]) else 0.0
    return {
        "seed": seed,
        "q_max": q_label,
        "method": method.name,
        "iterations": int(len(result["residual"])),
        "final_residual": residual,
        "runtime_sec": elapsed,
        "observed_rmse": observed_rmse,
        "full_rmse": full_rmse,
        "mean_q": mean_q,
        "multi_cut_share": multi_share,
        "mean_theta": float(np.nanmean(result["theta"])),
        "max_projected_gradient": float(np.max(result["pgn"])) if len(result["pgn"]) else 0.0,
    }


def main() -> None:
    rows: list[dict] = []
    for q in (2, 3):
        for seed in SEEDS:
            print(f"q={q}, seed={seed}", flush=True)
            problem = make_matrix_completion(seed)
            for method in methods_for_q(q):
                row = run_one(problem, method, seed, q)
                rows.append(row)
                print(
                    f"  {method.name}: it={row['iterations']} "
                    f"time={row['runtime_sec']:.3f}s full_rmse={row['full_rmse']:.3e}",
                    flush=True,
                )

    fields = list(rows[0])
    out_csv = HERE / "matrix_completion_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Median summary is easier to compare across the independent seeds.
    summary_csv = HERE / "matrix_completion_comparison_summary.csv"
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["q_max"], row["method"]), []).append(row)
    summary_fields = ["q_max", "method", "median_iterations", "median_runtime_sec", "median_final_residual", "median_observed_rmse", "median_full_rmse", "median_mean_q", "median_multi_cut_share"]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for (q, method), items in groups.items():
            writer.writerow({
                "q_max": q,
                "method": method,
                "median_iterations": np.median([x["iterations"] for x in items]),
                "median_runtime_sec": np.median([x["runtime_sec"] for x in items]),
                "median_final_residual": np.median([x["final_residual"] for x in items]),
                "median_observed_rmse": np.median([x["observed_rmse"] for x in items]),
                "median_full_rmse": np.median([x["full_rmse"] for x in items]),
                "median_mean_q": np.median([x["mean_q"] for x in items]),
                "median_multi_cut_share": np.median([x["multi_cut_share"] for x in items]),
            })

    # Compact visual summary: quality/runtime and actual memory usage.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    labels = ["one-cut", "recent", "violated", "angle", "adaptive"]
    colors = [exp.COLORS[x] for x in labels]
    for q_idx, q in enumerate((2, 3)):
        offset = -0.18 if q == 2 else 0.18
        for metric_idx, metric in enumerate(("median_runtime_sec", "median_full_rmse")):
            values = [next(r for r in csv.DictReader(summary_csv.open(encoding="utf-8")) if int(r["q_max"]) == q and r["method"] == method)[metric] for method in labels]
            values = np.asarray(values, dtype=float)
            x = np.arange(len(labels)) + offset
            axes[metric_idx].bar(x, values, width=0.32, color=colors, alpha=0.85 if q == 2 else 0.55, label=f"q={q}")
    axes[0].set_title("Median runtime")
    axes[0].set_ylabel("seconds")
    axes[1].set_title("Median full reconstruction RMSE")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("RMSE (log scale)")
    for axis in axes[:2]:
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(axis="y", color="#D4D4D4", linewidth=0.6)
    # Mean active q, shown separately to make the cost/benefit trade-off explicit.
    for q, alpha in ((2, 0.85), (3, 0.55)):
        values = [next(r for r in csv.DictReader(summary_csv.open(encoding="utf-8")) if int(r["q_max"]) == q and r["method"] == method)["median_mean_q"] for method in labels]
        axes[2].plot(labels, np.asarray(values, dtype=float), "o-", alpha=alpha, linewidth=2, label=f"q={q}")
    axes[2].set_title("Median mean active cuts")
    axes[2].set_ylabel("mean q_k")
    axes[2].grid(axis="y", color="#D4D4D4", linewidth=0.6)
    axes[2].legend()
    fig.suptitle(f"Matrix completion: obs={OBSERVATION}, blocked={BLOCKED}, tau ratio={TAU_RATIO}")
    fig.tight_layout()
    fig.savefig(HERE / "matrix_completion_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(HERE / "matrix_completion_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
