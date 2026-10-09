"""Exploratory finite-memory experiments on four virtual convex models.

This script is intentionally separate from the submission experiments.  It
compares one-cut, recent, violated, angle, and adaptive finite-memory rules on
pseudo-Huber regression, logistic selection, convex feasibility, and matrix
completion.  The multicut dual is solved by L-BFGS-B followed by a projected
gradient safeguard when the KKT residual is not yet small.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
ROOT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "one-cut": "#0072B2",
    "recent": "#E69F00",
    "violated": "#009E73",
    "angle": "#CC79A7",
    "adaptive": "#D55E00",
}


def soft_threshold(s, tau):
    return np.sign(s) * np.maximum(np.abs(s) - tau, 0.0)


def svt_flat(s, tau, shape):
    matrix = s.reshape(shape)
    u, sigma, vt = np.linalg.svd(matrix, full_matrices=False)
    sigma = np.maximum(sigma - tau, 0.0)
    return ((u * sigma) @ vt).reshape(-1)


@dataclass
class Problem:
    name: str
    dim: int
    grad: object
    lipschitz: float
    tau: float
    mirror: object
    omega: object
    omega_star: object
    max_iter: int


@dataclass(frozen=True)
class Method:
    name: str
    rule: str
    adaptive: bool = False
    q: int = 5
    p: int = 10
    trigger: float = 0.01


METHODS = [
    Method("one-cut", "recent", q=1, p=1),
    Method("recent", "recent"),
    Method("violated", "violated"),
    Method("angle", "angle"),
    Method("adaptive", "angle", adaptive=True),
]


def positive_violation(cut, x):
    norm = max(float(np.linalg.norm(cut["a"])), 1e-30)
    return max(float(cut["a"] @ x - cut["beta"]), 0.0) / norm


def prune_pool(pool, p, x):
    if len(pool) <= p:
        return pool
    recent_count = max(1, p // 2)
    recent = sorted(pool, key=lambda c: c["id"], reverse=True)[:recent_count]
    used = {c["id"] for c in recent}
    remaining = [c for c in pool if c["id"] not in used]
    useful = sorted(
        remaining, key=lambda c: positive_violation(c, x), reverse=True
    )[: p - len(recent)]
    return sorted(recent + useful, key=lambda c: c["id"])


def angle_select(current, candidates, x, q, required=None):
    selected = [current]
    if required is not None and required["id"] != current["id"]:
        selected.append(required)
    used = {c["id"] for c in selected}
    while len(selected) < q:
        best = None
        best_score = -1.0
        selected_normals = [
            c["a"] / max(np.linalg.norm(c["a"]), 1e-30) for c in selected
        ]
        for cut in candidates:
            if cut["id"] in used:
                continue
            normal = cut["a"] / max(np.linalg.norm(cut["a"]), 1e-30)
            novelty = 1.0 - max(abs(float(normal @ n)) for n in selected_normals)
            score = positive_violation(cut, x) * max(novelty, 0.0)
            if score > best_score:
                best, best_score = cut, score
        if best is None:
            break
        selected.append(best)
        used.add(best["id"])
    return selected


def fixed_select(method, pool, current, x):
    old = [c for c in pool if c["id"] != current["id"]]
    if method.rule == "recent":
        return [current] + sorted(old, key=lambda c: c["id"], reverse=True)[: method.q - 1]
    if method.rule == "violated":
        return [current] + sorted(old, key=lambda c: positive_violation(c, x), reverse=True)[: method.q - 1]
    return angle_select(current, old, x, method.q)


def projected_residual(alpha, grad):
    return float(np.linalg.norm(alpha - np.maximum(alpha - grad, 0.0)))


def one_cut(problem, x_star, a, beta):
    def derivative(alpha):
        return float(beta - a @ problem.mirror(x_star - alpha * a))

    if derivative(0.0) >= 0.0:
        return np.zeros(1)
    upper = 1.0
    while derivative(upper) < 0.0 and upper < 1e14:
        upper *= 2.0
    return np.array([brentq(derivative, 0.0, upper, xtol=1e-13, rtol=1e-13)])


def solve_dual(problem, x_star, G, beta, warm=None, tol=1e-9):
    q = len(beta)
    alpha0 = np.zeros(q) if warm is None or len(warm) != q else np.maximum(warm, 0.0)

    def value_grad(alpha):
        s = x_star - G @ alpha
        x = problem.mirror(s)
        value = float(problem.omega_star(s) + beta @ alpha)
        grad = beta - G.T @ x
        return value, grad

    result = minimize(
        value_grad,
        alpha0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * q,
        options={"maxiter": 300, "ftol": 1e-15, "gtol": tol, "maxls": 50},
    )
    alpha = np.maximum(result.x, 0.0)
    value, grad = value_grad(alpha)
    nit = int(result.nit)

    # KKT safeguard: L-BFGS-B may stop on function change while the projected
    # gradient is still too large.  Continue with monotone projected steps.
    step = 1.0 / max(float(np.linalg.norm(G, 2) ** 2), 1e-14)
    for extra in range(500):
        if projected_residual(alpha, grad) <= tol:
            break
        trial_step = step
        for _ in range(25):
            trial = np.maximum(alpha - trial_step * grad, 0.0)
            trial_value, trial_grad = value_grad(trial)
            if trial_value <= value - 1e-4 * float(np.linalg.norm(trial - alpha) ** 2) / max(trial_step, 1e-30):
                break
            trial_step *= 0.5
        delta = trial - alpha
        grad_delta = trial_grad - grad
        denom = float(delta @ grad_delta)
        if denom > 1e-20:
            step = float(np.clip((delta @ delta) / denom, 1e-12, 1e12))
        else:
            step = trial_step
        alpha, value, grad = trial, trial_value, trial_grad
    return alpha, nit + extra + 1, projected_residual(alpha, grad)


def run(problem, method):
    x = np.zeros(problem.dim)
    x_star = np.zeros(problem.dim)
    pool = []
    warm = {}
    g0 = max(float(np.linalg.norm(problem.grad(x))), 1e-30)
    history = {k: [] for k in ["residual", "runtime", "q", "theta", "theta_lb", "pgn"]}
    started = time.perf_counter()

    for k in range(problem.max_iter):
        raw_a = problem.grad(x)
        residual = float(np.linalg.norm(raw_a) / g0)
        if residual <= 1e-10:
            break
        raw_beta = float(raw_a @ x - (raw_a @ raw_a) / problem.lipschitz)
        # Positive rescaling leaves the halfspace unchanged.  Unit normals
        # prevent the absolute dual tolerance from becoming meaningless when
        # the outer gradient is small.
        cut_scale = max(float(np.linalg.norm(raw_a)), 1e-30)
        a = raw_a / cut_scale
        beta = raw_beta / cut_scale
        current = {"id": k, "a": a.copy(), "beta": beta}
        pool.append(current)
        pool = prune_pool(pool, method.p, x)

        alpha_one = one_cut(problem, x_star, a, beta)
        x_one = problem.mirror(x_star - alpha_one[0] * a)
        one_step = problem.omega(x_one) - problem.omega(x) - x_star @ (x_one - x)
        theta_lb = 1.0

        if method.adaptive:
            old = [c for c in pool if c["id"] != k]
            required = max(old, key=lambda c: positive_violation(c, x_one), default=None)
            delta = positive_violation(required, x_one) if required is not None else 0.0
            # Near convergence, ``one_step`` is obtained from a subtraction
            # of nearly equal Bregman-energy terms.  Reusing a cancellation-
            # dominated denominator here can create an enormous spurious
            # predicted gain and trigger an expensive multicut solve after the
            # useful outer progress is already below numerical resolution.
            gain = 0.5 * delta * delta / one_step if one_step > 1e-14 else 0.0
            theta_lb = 1.0 + gain
            selected = angle_select(current, old, x_one, method.q, required) if gain >= method.trigger and delta > 0 else [current]
        else:
            selected = fixed_select(method, pool, current, x)

        G = np.column_stack([c["a"] for c in selected])
        betas = np.array([c["beta"] for c in selected])
        ids = [c["id"] for c in selected]
        if len(selected) == 1:
            alpha = alpha_one
            pgn = 0.0
        else:
            alpha, _, pgn = solve_dual(
                problem, x_star, G, betas,
                np.array([warm.get(i, 0.0) for i in ids]),
            )

        x_old, x_star_old = x, x_star
        x_star = x_star_old - G @ alpha
        x = problem.mirror(x_star)
        step_value = problem.omega(x) - problem.omega(x_old) - x_star_old @ (x - x_old)
        # Very small Bregman displacements are dominated by cancellation and
        # should not be used to form a gain ratio.
        theta = step_value / one_step if one_step > 1e-14 else np.nan
        warm = {i: float(v) for i, v in zip(ids, alpha) if v > 1e-14}
        pool = prune_pool(pool, method.p, x)
        live = {c["id"] for c in pool}
        warm = {i: v for i, v in warm.items() if i in live}

        history["residual"].append(float(np.linalg.norm(problem.grad(x)) / g0))
        history["runtime"].append(time.perf_counter() - started)
        history["q"].append(len(selected))
        history["theta"].append(theta)
        history["theta_lb"].append(theta_lb)
        history["pgn"].append(pgn)

    result = {key: np.asarray(value) for key, value in history.items()}
    result["x_final"] = x.copy()
    return result


def vector_geometry(tau):
    mirror = lambda s: soft_threshold(s, tau)
    omega = lambda x: float(tau * np.linalg.norm(x, 1) + 0.5 * (x @ x))
    omega_star = lambda s: float(0.5 * np.linalg.norm(soft_threshold(s, tau)) ** 2)
    return mirror, omega, omega_star


def make_pseudo_huber(seed):
    rng = np.random.default_rng(1000 + seed)
    m, n = 120, 300
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    x_true = np.zeros(n)
    support = rng.choice(n, 18, replace=False)
    x_true[support] = rng.normal(size=len(support))
    b = A @ x_true + 0.02 * rng.normal(size=m)
    outliers = rng.choice(m, 12, replace=False)
    b[outliers] += 1.5 * rng.normal(size=len(outliers))
    delta = 0.15

    def grad(x):
        r = A @ x - b
        psi = r / np.sqrt(1.0 + (r / delta) ** 2)
        return A.T @ psi / m

    L = float(np.linalg.norm(A, 2) ** 2 / m)
    tau = 0.02 * np.linalg.norm(grad(np.zeros(n)), np.inf)
    mirror, omega, omega_star = vector_geometry(tau)
    return Problem("pseudo-Huber", n, grad, L, tau, mirror, omega, omega_star, 300)


def make_logistic(seed):
    rng = np.random.default_rng(2000 + seed)
    m, n = 180, 400
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    x_true = np.zeros(n)
    support = rng.choice(n, 20, replace=False)
    x_true[support] = 2.0 * rng.normal(size=len(support))
    scores = A @ x_true
    prob = 1.0 / (1.0 + np.exp(-scores))
    labels = np.where(rng.random(m) < prob, 1.0, -1.0)
    flip = rng.choice(m, 18, replace=False)
    labels[flip] *= -1.0
    lam = 0.05

    def grad(x):
        y = A @ x
        weight = -labels / (1.0 + np.exp(np.clip(labels * y, -50, 50))) + lam * y
        return A.T @ weight / m

    L = float((0.25 + lam) * np.linalg.norm(A, 2) ** 2 / m)
    tau = 0.02 * np.linalg.norm(grad(np.zeros(n)), np.inf)
    mirror, omega, omega_star = vector_geometry(tau)
    return Problem("logistic", n, grad, L, tau, mirror, omega, omega_star, 300)


def make_feasibility(seed):
    rng = np.random.default_rng(3000 + seed)
    m, n = 180, 120
    x_feas = np.zeros(n)
    support = rng.choice(n, 15, replace=False)
    x_feas[support] = 1.5 + 0.3 * rng.random(len(support))
    direction = x_feas / np.linalg.norm(x_feas)
    A = np.empty((m, n))
    for i in range(m):
        noise = rng.normal(size=n)
        noise -= direction * (direction @ noise)
        noise /= np.linalg.norm(noise)
        A[i] = -0.72 * direction + np.sqrt(1.0 - 0.72**2) * noise
    b = A @ x_feas + 0.08 * np.linalg.norm(x_feas)

    def grad(x):
        violation = np.maximum(A @ x - b, 0.0)
        return A.T @ violation / m

    L = float(np.linalg.norm(A, 2) ** 2 / m)
    tau = 0.02 * np.linalg.norm(grad(np.zeros(n)), np.inf)
    mirror, omega, omega_star = vector_geometry(tau)
    return Problem("convex feasibility", n, grad, L, tau, mirror, omega, omega_star, 300)


def make_matrix_completion(seed):
    rng = np.random.default_rng(4000 + seed)
    nr, nc, rank = 22, 22, 3
    left = rng.normal(size=(nr, rank))
    right = rng.normal(size=(nc, rank))
    truth = left @ right.T / np.sqrt(rank)
    mask = rng.random((nr, nc)) < 0.38
    observed = np.where(mask, truth, 0.0)

    def grad(x):
        matrix = x.reshape(nr, nc)
        return np.where(mask, matrix - observed, 0.0).reshape(-1)

    g0 = grad(np.zeros(nr * nc)).reshape(nr, nc)
    tau = 0.02 * np.linalg.norm(g0, 2)
    mirror = lambda s: svt_flat(s, tau, (nr, nc))
    omega = lambda x: float(
        tau * np.linalg.norm(x.reshape(nr, nc), ord="nuc") + 0.5 * (x @ x)
    )
    omega_star = lambda s: float(0.5 * np.linalg.norm(mirror(s)) ** 2)
    return Problem(
        "matrix completion", nr * nc, grad, 1.0, tau,
        mirror, omega, omega_star, 180,
    )


BUILDERS = [make_pseudo_huber, make_logistic, make_feasibility, make_matrix_completion]


def main():
    all_runs = {}
    rows = []
    for builder in BUILDERS:
        label = builder(0).name
        all_runs[label] = {method.name: [] for method in METHODS}
        for seed in range(3):
            problem = builder(seed)
            order = METHODS[seed % len(METHODS):] + METHODS[:seed % len(METHODS)]
            print(f"{problem.name}, seed={seed}, order={[m.name for m in order]}", flush=True)
            temporary = {}
            for method in order:
                print(f"  {method.name}", flush=True)
                temporary[method.name] = run(problem, method)
            for method in METHODS:
                result = temporary[method.name]
                all_runs[label][method.name].append(result)
                rows.append([
                    label, seed, method.name, len(result["residual"]),
                    result["residual"][-1], result["runtime"][-1],
                    np.mean(result["q"]), np.nanmean(result["theta"]),
                    np.max(result["pgn"]),
                ])

    with open(ROOT / "per_seed_results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "problem", "seed", "method", "iterations", "final_residual",
            "runtime", "mean_q", "mean_theta", "max_projected_gradient",
        ])
        writer.writerows(rows)

    summary = []
    for problem, groups in all_runs.items():
        one = np.median([r["residual"][-1] for r in groups["one-cut"]])
        for method, runs in groups.items():
            final = np.array([r["residual"][-1] for r in runs])
            runtime = np.array([r["runtime"][-1] for r in runs])
            summary.append([
                problem, method, np.median(final), np.median(runtime),
                np.median([np.mean(r["q"]) for r in runs]),
                np.median([np.nanmean(r["theta"]) for r in runs]),
                np.median(final) / max(one, 1e-300),
            ])
    with open(ROOT / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "problem", "method", "median_final_residual", "median_runtime",
            "median_mean_q", "median_mean_theta", "residual_ratio_to_onecut",
        ])
        writer.writerows(summary)

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.0))
    styles = {"one-cut": "-", "recent": "--", "violated": "-.", "angle": ":", "adaptive": "-"}
    for col, (problem, groups) in enumerate(all_runs.items()):
        for method, runs in groups.items():
            width = min(len(r["residual"]) for r in runs)
            values = np.vstack([r["residual"][:width] for r in runs])
            x = np.arange(1, width + 1)
            axes[0, col].semilogy(
                x, np.median(values, axis=0), styles[method],
                color=COLORS[method], linewidth=1.5, label=method,
            )
            stop = min(r["runtime"][-1] for r in runs)
            start = max(r["runtime"][0] for r in runs)
            grid = np.linspace(start, stop, 250)
            timed = np.vstack([
                np.interp(grid, r["runtime"], r["residual"]) for r in runs
            ])
            axes[1, col].semilogy(
                grid, np.median(timed, axis=0), styles[method],
                color=COLORS[method], linewidth=1.5,
            )
        axes[0, col].set_title(problem)
        axes[0, col].set_xlabel("iteration")
        axes[1, col].set_xlabel("time (s)")
        for row in range(2):
            axes[row, col].grid(True, which="major", color="#D4D4D4", linewidth=0.6)
        if col == 0:
            axes[0, col].set_ylabel("relative gradient residual")
            axes[1, col].set_ylabel("relative gradient residual")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "four_virtual_examples.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "four_virtual_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
