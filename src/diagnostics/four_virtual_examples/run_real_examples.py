"""Exploratory tests on real datasets bundled with scikit-learn."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import run_virtual_examples as exp


OUT = Path(__file__).resolve().parent / "real_data"
OUT.mkdir(parents=True, exist_ok=True)


def vector_problem(name, A, b, loss, seed, max_iter=150):
    del seed
    m, n = A.shape
    if loss == "huber":
        delta = 0.30

        def grad(x):
            r = A @ x - b
            return A.T @ (r / np.sqrt(1.0 + (r / delta) ** 2)) / m

        L = np.linalg.norm(A, 2) ** 2 / m
    elif loss == "logistic":
        lam = 0.02

        def grad(x):
            score = A @ x
            weight = -b / (1.0 + np.exp(np.clip(b * score, -50, 50))) + lam * score
            return A.T @ weight / m

        L = (0.25 + lam) * np.linalg.norm(A, 2) ** 2 / m
    else:
        raise ValueError(loss)
    tau = 0.02 * np.linalg.norm(grad(np.zeros(n)), np.inf)
    mirror, omega, omega_star = exp.vector_geometry(tau)
    return exp.Problem(name, n, grad, float(L), float(tau), mirror, omega, omega_star, max_iter)


def make_diabetes_huber(seed):
    data = load_diabetes()
    rng = np.random.default_rng(5000 + seed)
    indices = rng.choice(len(data.data), 55, replace=False)
    X = data.data[indices]
    y = data.target[indices]
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    y = (y - y.mean()) / y.std()
    return vector_problem("diabetes pseudo-Huber", X, y, "huber", seed)


def make_cancer_logistic(seed):
    data = load_breast_cancer()
    rng = np.random.default_rng(6000 + seed)
    indices = rng.choice(len(data.data), 140, replace=False)
    X = StandardScaler().fit_transform(data.data[indices])
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    X = StandardScaler().fit_transform(X)
    labels = 2.0 * data.target[indices] - 1.0
    return vector_problem("breast-cancer logistic", X, labels, "logistic", seed)


def make_digits_feasibility(seed):
    data = load_digits()
    keep = np.flatnonzero((data.target == 0) | (data.target == 1))
    rng = np.random.default_rng(7000 + seed)
    indices = rng.choice(keep, 160, replace=False)
    X = StandardScaler().fit_transform(data.data[indices])
    X = np.column_stack([X, np.ones(len(X))])
    labels = np.where(data.target[indices] == 1, 1.0, -1.0)
    # A deterministic least-squares separator verifies feasibility; it is not
    # used by the tested method.  This avoids introducing a second iterative
    # solver (and its convergence warnings) into data construction.
    separator, *_ = np.linalg.lstsq(X, labels, rcond=None)
    signed = labels * (X @ separator)
    if np.min(signed) <= 0:
        raise RuntimeError("selected digit constraints are not separable")
    rows = -labels[:, None] * X
    norms = np.maximum(np.linalg.norm(rows, axis=1), 1e-30)
    A = rows / norms[:, None]
    beta = -np.ones(len(rows)) / norms
    m, n = A.shape

    def grad(x):
        return A.T @ np.maximum(A @ x - beta, 0.0) / m

    L = np.linalg.norm(A, 2) ** 2 / m
    tau = 0.02 * np.linalg.norm(grad(np.zeros(n)), np.inf)
    mirror, omega, omega_star = exp.vector_geometry(tau)
    return exp.Problem("digits feasibility", n, grad, float(L), float(tau), mirror, omega, omega_star, 150)


def make_digits_matrix(seed):
    data = load_digits()
    rng = np.random.default_rng(8000 + seed)
    indices = rng.choice(len(data.data), 32, replace=False)
    truth = data.data[indices] / 16.0
    nr, nc = truth.shape
    mask = rng.random((nr, nc)) < 0.40
    observed = np.where(mask, truth, 0.0)

    def grad(x):
        matrix = x.reshape(nr, nc)
        return np.where(mask, matrix - observed, 0.0).reshape(-1)

    g0 = grad(np.zeros(nr * nc)).reshape(nr, nc)
    tau = 0.02 * np.linalg.norm(g0, 2)
    mirror = lambda s: exp.svt_flat(s, tau, (nr, nc))
    omega = lambda x: float(
        tau * np.linalg.norm(x.reshape(nr, nc), ord="nuc") + 0.5 * (x @ x)
    )
    omega_star = lambda s: float(0.5 * np.linalg.norm(mirror(s)) ** 2)
    return exp.Problem("digits matrix completion", nr * nc, grad, 1.0, tau, mirror, omega, omega_star, 40)


BUILDERS = [make_diabetes_huber, make_cancer_logistic, make_digits_feasibility, make_digits_matrix]


def geometry(problem):
    x = np.zeros(problem.dim)
    x_star = np.zeros(problem.dim)
    pool = []
    gains = []
    for k in range(problem.max_iter):
        raw = problem.grad(x)
        norm = np.linalg.norm(raw)
        if norm <= 1e-14:
            break
        beta_raw = raw @ x - (raw @ raw) / problem.lipschitz
        a, beta = raw / norm, beta_raw / norm
        current = {"id": k, "a": a, "beta": beta}
        pool.append(current)
        pool = exp.prune_pool(pool, 10, x)
        alpha = exp.one_cut(problem, x_star, a, beta)
        y = problem.mirror(x_star - alpha[0] * a)
        step = problem.omega(y) - problem.omega(x) - x_star @ (y - x)
        old = [cut for cut in pool if cut["id"] != k]
        violation = max([exp.positive_violation(cut, y) for cut in old], default=0.0)
        gains.append(0.5 * violation**2 / max(step, 1e-30))
        x_star = x_star - alpha[0] * a
        x = y
        pool = exp.prune_pool(pool, 10, x)
    return np.asarray(gains)


def main():
    geometry_rows = []
    groups = {}
    result_rows = []
    for builder in BUILDERS:
        label = builder(0).name
        groups[label] = {method.name: [] for method in exp.METHODS}
        all_gains = []
        for seed in range(2):
            problem = builder(seed)
            gains = geometry(problem)
            all_gains.extend(gains)
            order = exp.METHODS[seed % 5:] + exp.METHODS[:seed % 5]
            temporary = {}
            print(label, seed, [m.name for m in order], flush=True)
            for method in order:
                print(" ", method.name, flush=True)
                temporary[method.name] = exp.run(problem, method)
            for method in exp.METHODS:
                run = temporary[method.name]
                groups[label][method.name].append(run)
                result_rows.append([
                    label, seed, method.name, len(run["residual"]),
                    run["residual"][-1], run["runtime"][-1], np.mean(run["q"]),
                ])
        all_gains = np.asarray(all_gains)
        geometry_rows.append([
            label, len(all_gains), np.max(all_gains),
            np.mean(all_gains > 1e-8), np.mean(all_gains > 1e-4),
            np.mean(all_gains > 1e-3), np.mean(all_gains > 1e-2),
        ])

    with open(OUT / "geometry_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["problem", "observations", "max_gain", "share_gt_1e-8", "share_gt_1e-4", "share_gt_1e-3", "share_gt_1e-2"])
        writer.writerows(geometry_rows)
    with open(OUT / "per_seed_results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["problem", "seed", "method", "iterations", "final_residual", "runtime", "mean_q"])
        writer.writerows(result_rows)

    summary = []
    for label, methods in groups.items():
        for name, runs in methods.items():
            summary.append([
                label, name,
                np.median([r["residual"][-1] for r in runs]),
                np.median([r["runtime"][-1] for r in runs]),
                np.median([len(r["residual"]) for r in runs]),
                np.median([np.mean(r["q"]) for r in runs]),
            ])
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["problem", "method", "median_final_residual", "median_runtime", "median_iterations", "median_mean_q"])
        writer.writerows(summary)

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.0))
    styles = {"one-cut": "-", "recent": "--", "violated": "-.", "angle": ":", "adaptive": "-"}
    for col, (label, methods) in enumerate(groups.items()):
        for name, runs in methods.items():
            width = min(len(r["residual"]) for r in runs)
            values = np.vstack([r["residual"][:width] for r in runs])
            axes[0, col].semilogy(np.arange(1, width + 1), np.median(values, axis=0), styles[name], color=exp.COLORS[name], linewidth=1.5, label=name)
            start = max(r["runtime"][0] for r in runs)
            stop = min(r["runtime"][-1] for r in runs)
            grid = np.linspace(start, stop, 250)
            timed = np.vstack([np.interp(grid, r["runtime"], r["residual"]) for r in runs])
            axes[1, col].semilogy(grid, np.median(timed, axis=0), styles[name], color=exp.COLORS[name], linewidth=1.5)
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("iteration")
        axes[1, col].set_xlabel("time (s)")
        for row in range(2):
            axes[row, col].grid(True, which="major", color="#D4D4D4", linewidth=0.6)
        if col == 0:
            axes[0, col].set_ylabel("relative gradient residual")
            axes[1, col].set_ylabel("relative gradient residual")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "real_data_examples.pdf", bbox_inches="tight")
    fig.savefig(OUT / "real_data_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
