"""Theory-aligned experiments for finite-memory cut-and-project LBI.

Each output corresponds to a specific statement in the manuscript:
  1. global linear convergence for exact sparse least squares;
  2. rho = theta * rho_one and the retained-cut lower bound for theta;
  3. the computable inexact Fejer certificate;
  4. robustness over random seeds and moderate scale growth.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "fig_memory"
RESULTS = ROOT / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def soft_threshold(z: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)


def omega(x: np.ndarray, tau: float) -> float:
    return tau * float(np.linalg.norm(x, 1)) + 0.5 * float(x @ x)


def omega_star(s: np.ndarray, tau: float) -> float:
    y = soft_threshold(s, tau)
    return 0.5 * float(y @ y)


def bregman(target: np.ndarray, x: np.ndarray, x_star: np.ndarray, tau: float) -> float:
    value = omega(target, tau) - omega(x, tau) - float(x_star @ (target - x))
    return max(value, 0.0)


def dual_phi_grad(alpha, x_star, G, beta, tau):
    s = x_star - G @ alpha
    x = soft_threshold(s, tau)
    phi = omega_star(s, tau) + float(beta @ alpha)
    grad = beta - G.T @ x
    return phi, grad


def projected_grad_norm(alpha, grad):
    return float(np.linalg.norm(alpha - np.maximum(alpha - grad, 0.0)))


def solve_dual(x_star, G, beta, tau, warm=None, dual_tol=1e-11, max_iter=250):
    q = len(beta)
    if warm is None or len(warm) != q:
        warm = np.zeros(q)
    warm = np.maximum(np.asarray(warm, dtype=float), 0.0)

    def fun(alpha):
        return dual_phi_grad(alpha, x_star, G, beta, tau)

    options = {"maxiter": max_iter, "ftol": 1e-15, "gtol": dual_tol, "maxls": 50}
    bounds = [(0.0, None)] * q
    res = minimize(fun, warm, method="L-BFGS-B", jac=True, bounds=bounds, options=options)
    alpha = np.maximum(np.asarray(res.x), 0.0)
    _, grad = dual_phi_grad(alpha, x_star, G, beta, tau)
    return alpha, {
        "nit": int(getattr(res, "nit", 0)),
        "pgn": projected_grad_norm(alpha, grad),
        "success": bool(res.success or projected_grad_norm(alpha, grad) <= 20 * dual_tol),
    }


def compact_normal_constraint(A: np.ndarray, b: np.ndarray, rank_tol=1e-11):
    """Return R,c with {x:A^T A x=A^T b}={x:R x=c}."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = int(np.sum(s > rank_tol * max(s[0], 1.0)))
    U = U[:, :r]
    s = s[:r]
    R = Vt[:r, :]
    c = (U.T @ b) / s
    return R, c, U, s


def solve_bilevel_reference(A: np.ndarray, b: np.ndarray, tau: float):
    R, c, _, _ = compact_normal_constraint(A, b)

    def fun(y):
        s = R.T @ y
        x = soft_threshold(s, tau)
        return omega_star(s, tau) - float(c @ y), R @ x - c

    res = minimize(
        fun,
        np.zeros(R.shape[0]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-13, "maxls": 80},
    )
    refine = least_squares(
        lambda y: R @ soft_threshold(R.T @ y, tau) - c,
        np.asarray(res.x),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=3000,
    )
    y = np.asarray(refine.x)
    xbar = soft_threshold(R.T @ y, tau)
    constraint_residual = float(np.linalg.norm(R @ xbar - c))
    if constraint_residual > 1e-6:
        raise RuntimeError(f"reference solve insufficiently accurate: {constraint_residual:.3e}")
    return xbar, constraint_residual


def orthonormal_columns(rng, rows, cols):
    Q, _ = np.linalg.qr(rng.normal(size=(rows, cols)), mode="reduced")
    return Q[:, :cols]


def make_spectral_problem(seed, m, n, rank, condition=20.0, inconsistent=False, sparsity=None):
    rng = np.random.default_rng(seed)
    U = orthonormal_columns(rng, m, rank)
    V = orthonormal_columns(rng, n, rank)
    s = np.geomspace(1.0, 1.0 / condition, rank)
    A = (U * s) @ V.T

    if sparsity is None:
        sparsity = max(8, n // 16)
    x_true = np.zeros(n)
    support = rng.choice(n, size=min(sparsity, n), replace=False)
    x_true[support] = rng.normal(size=len(support))
    b_clean = A @ x_true
    b = b_clean.copy()
    if inconsistent:
        if m <= rank:
            raise ValueError("inconsistent construction requires m > rank")
        e = rng.normal(size=m)
        e -= U @ (U.T @ e)
        e /= max(np.linalg.norm(e), 1e-16)
        b += 0.08 * max(np.linalg.norm(b_clean), 1e-12) * e
    tau = 0.05 * max(float(np.max(np.abs(A.T @ b))), 1e-8)
    return A, b, x_true, tau


def make_gaussian_problem(seed, m, n, inconsistent=False, extra_rows=35, sparsity=None):
    """Gaussian sparse-recovery model; optionally add a true inconsistent component."""
    rng = np.random.default_rng(seed)
    A0 = rng.normal(size=(m, n)) / np.sqrt(m)
    A0 /= np.maximum(np.linalg.norm(A0, axis=0), 1e-12)
    if sparsity is None:
        sparsity = max(8, n // 15)
    x_true = np.zeros(n)
    support = rng.choice(n, size=min(sparsity, n), replace=False)
    x_true[support] = rng.normal(size=len(support))

    if not inconsistent:
        A = A0
        b = A @ x_true
    else:
        U = orthonormal_columns(rng, m + extra_rows, m)
        A = U @ A0
        b_clean = A @ x_true
        e = rng.normal(size=m + extra_rows)
        e -= U @ (U.T @ e)
        e /= max(np.linalg.norm(e), 1e-16)
        b = b_clean + 0.08 * np.linalg.norm(b_clean) * e
    tau = 0.05 * max(float(np.max(np.abs(A.T @ b))), 1e-8)
    return A, b, x_true, tau


@dataclass(frozen=True)
class Method:
    name: str
    rule: str
    q: int
    color: str


METHODS = [
    Method("one-cut", "recent", 1, "#0072B2"),
    Method("recent q=5", "recent", 5, "#E69F00"),
    Method("violated q=5", "violated", 5, "#009E73"),
]


def choose_memory(rule, q, k, cuts_a, cuts_beta, x):
    if q <= 1 or k == 0:
        return [k]
    old = list(range(k))
    if rule == "recent":
        return list(range(max(0, k - q + 1), k + 1))
    violations = np.array([cuts_a[i] @ x - cuts_beta[i] for i in old])
    order = [old[i] for i in np.argsort(-violations)]
    if rule == "violated":
        return order[: q - 1] + [k]
    raise ValueError(rule)


def computable_epsilon(alpha, grad_phi, universal, step):
    complementarity = max(float(alpha @ grad_phi), 0.0)
    shortfall = max(float(universal - step), 0.0)
    return complementarity + shortfall


def run_method(
    A,
    b,
    tau,
    xbar,
    method,
    max_iter=350,
    solve_mode="accurate",
    fixed_tol=1e-4,
    eta0=None,
):
    n = A.shape[1]
    L = float(np.linalg.norm(A, 2) ** 2)
    x = np.zeros(n)
    x_star = np.zeros(n)
    cuts_a, cuts_beta = [], []
    warm_by_cut = {}
    norm_atb = max(float(np.linalg.norm(A.T @ b)), 1e-16)
    initial_grad2 = max(float((A.T @ b) @ (A.T @ b)), 1e-300)
    energy0 = bregman(xbar, x, x_star, tau)
    initial_universal = max(float(np.linalg.norm(A.T @ b) ** 2 / (2 * L * L)), 1e-16)
    if eta0 is None:
        eta0 = 1e-3 * initial_universal

    keys = [
        "iter", "energy", "energy_rel", "normal", "theta", "theta_lb", "rho",
        "rho_one", "factor_error", "range_ratio", "contraction", "eps_hat", "eta",
        "pgn", "dual_nit", "runtime", "strict_gain", "bound_gap",
    ]
    hist = {key: [] for key in keys}
    start = time.perf_counter()

    for k in range(max_iter):
        residual = A @ x - b
        a = A.T @ residual
        grad2 = float(a @ a)
        if grad2 <= 1e-16 * initial_grad2:
            break
        beta = float(a @ x - grad2 / L)
        cuts_a.append(a.copy())
        cuts_beta.append(beta)
        Ik = choose_memory(method.rule, method.q, k, cuts_a, cuts_beta, x)
        G = np.column_stack([cuts_a[i] for i in Ik])
        beta_vec = np.array([cuts_beta[i] for i in Ik])
        warm = np.array([warm_by_cut.get(i, 0.0) for i in Ik])

        # Same-state exact one-cut baseline for theta_k.
        one_alpha, _ = solve_dual(
            x_star, a.reshape(-1, 1), np.array([beta]), tau, dual_tol=1e-13, max_iter=500
        )
        x_one_star = x_star - a * one_alpha[0]
        x_one = soft_threshold(x_one_star, tau)

        universal = grad2 / (2 * L * L)
        eta_k = eta0 / ((k + 1) ** 1.5) if solve_mode == "certified" else np.nan
        if solve_mode == "accurate":
            tolerances = [1e-13]
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
                x_star, G, beta_vec, tau, warm=warm, dual_tol=inner_tol, max_iter=500
            )
            total_nit += info["nit"]
            x_trial_star = x_star - G @ alpha
            x_trial = soft_threshold(x_trial_star, tau)
            step_trial = bregman(x_trial, x, x_star, tau)
            _, grad_phi_trial = dual_phi_grad(alpha, x_star, G, beta_vec, tau)
            eps_trial = computable_epsilon(alpha, grad_phi_trial, universal, step_trial)
            warm = alpha
            if solve_mode != "certified" or eps_trial <= eta_k:
                break

        warm_by_cut = {i: float(v) for i, v in zip(Ik, alpha) if v > 1e-14}
        x_old, x_star_old = x, x_star
        energy_old = bregman(xbar, x_old, x_star_old, tau)
        x_star = x_star_old - G @ alpha
        x = soft_threshold(x_star, tau)
        energy_new = bregman(xbar, x, x_star, tau)
        step = bregman(x, x_old, x_star_old, tau)
        one_step = bregman(x_one, x_old, x_star_old, tau)
        theta = step / max(one_step, 1e-300)
        rho_one = one_step / max(universal, 1e-300)
        rho = step / max(universal, 1e-300)

        deltas = []
        for j in Ik:
            norm_aj = float(np.linalg.norm(cuts_a[j]))
            if norm_aj > 0:
                deltas.append(max(float(cuts_a[j] @ x_one - cuts_beta[j]), 0.0) / norm_aj)
        delta = max(deltas, default=0.0)
        theta_lb = 1.0 + 0.5 * delta * delta / max(one_step, 1e-300)
        _, grad_phi = dual_phi_grad(alpha, x_star_old, G, beta_vec, tau)
        eps_hat = computable_epsilon(alpha, grad_phi, universal, step)
        normal = float(np.linalg.norm(A.T @ (A @ x - b)) / norm_atb)

        hist["iter"].append(k + 1)
        hist["energy"].append(energy_new)
        hist["energy_rel"].append(energy_new / max(energy0, 1e-300))
        hist["normal"].append(normal)
        hist["theta"].append(theta)
        hist["theta_lb"].append(theta_lb)
        hist["rho"].append(rho)
        hist["rho_one"].append(rho_one)
        hist["factor_error"].append(abs(rho - theta * rho_one) / max(1.0, abs(rho)))
        hist["range_ratio"].append(energy_old / max(grad2, 1e-300))
        hist["contraction"].append(energy_new / max(energy_old, 1e-300))
        hist["eps_hat"].append(eps_hat)
        hist["eta"].append(eta_k)
        hist["pgn"].append(info["pgn"])
        hist["dual_nit"].append(total_nit)
        hist["runtime"].append(time.perf_counter() - start)
        hist["strict_gain"].append(float(theta > 1.0 + 1e-4))
        hist["bound_gap"].append(theta - theta_lb)

        if energy_new <= 1e-12 * max(energy0, 1.0) or normal <= 1e-8:
            break

    return {key: np.asarray(value, dtype=float) for key, value in hist.items()}


def empirical_q(hist, tail=25):
    ratios = hist["contraction"]
    ratios = ratios[np.isfinite(ratios) & (ratios >= 0) & (ratios < 1.2)]
    if len(ratios) == 0:
        return np.nan
    return float(np.median(ratios[-min(tail, len(ratios)) :]))


def iterations_to(hist, threshold=1e-6):
    idx = np.flatnonzero(hist["energy_rel"] <= threshold)
    return int(idx[0] + 1) if len(idx) else int(len(hist["iter"]) + 1)


def save_npz(path, groups):
    data = {}
    for group, methods in groups.items():
        for method, hist in methods.items():
            clean = method.replace(" ", "_").replace("=", "")
            for metric, values in hist.items():
                data[f"{group}__{clean}__{metric}"] = values
    np.savez(path, **data)


def run_exact_experiments():
    problems = {
        "consistent": make_gaussian_problem(12, 60, 180, inconsistent=False),
        "inconsistent": make_gaussian_problem(15, 60, 180, inconsistent=True),
    }
    results = {}
    refs = {}
    for problem, (A, b, _, tau) in problems.items():
        xbar, ref_res = solve_bilevel_reference(A, b, tau)
        refs[problem] = ref_res
        results[problem] = {}
        for method in METHODS:
            print(f"exact {problem}: {method.name}")
            results[problem][method.name] = run_method(
                A, b, tau, xbar, method, max_iter=55, solve_mode="accurate"
            )
    return results, refs


def plot_exact(results):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, problem in zip(axes, ["consistent", "inconsistent"]):
        for method in METHODS:
            h = results[problem][method.name]
            ax.semilogy(h["iter"], np.maximum(h["energy_rel"], 1e-16),
                        label=method.name, color=method.color, linewidth=1.8)
        ax.set_title(f"{problem} least squares")
        ax.set_xlabel("iteration")
        ax.set_ylabel(r"$\mathcal{E}_k/\mathcal{E}_0$")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exact_linear_convergence.pdf", bbox_inches="tight")
    plt.close(fig)

    h = results["consistent"]["violated q=5"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    axes[0, 0].plot(h["iter"], h["theta"], label=r"observed $\theta_k$", color="#009E73")
    axes[0, 0].plot(h["iter"], h["theta_lb"], "--", label="violation lower bound", color="#D55E00")
    axes[0, 0].axhline(1.0, color="black", linewidth=0.8)
    axes[0, 0].set_ylabel("memory gain")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(h["iter"], np.maximum(h["factor_error"], 1e-18), color="#0072B2")
    axes[0, 1].set_ylabel(r"relative error in $\rho_k=\theta_k\rho_k^{\rm one}$")

    axes[1, 0].semilogy(h["iter"], np.maximum(h["range_ratio"], 1e-18), color="#CC79A7")
    axes[1, 0].set_ylabel(r"$\mathcal{E}_k/\|\nabla f(x_k)\|^2$")

    axes[1, 1].semilogy(h["iter"], np.maximum(h["eps_hat"], 1e-18), color="#E69F00")
    axes[1, 1].set_ylabel(r"computable $\widehat\varepsilon_k$")
    for ax in axes.ravel():
        ax.set_xlabel("iteration")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "theta_theory_certificates.pdf", bbox_inches="tight")
    plt.close(fig)


def write_exact_table(results, refs):
    rows = []
    for problem in ["consistent", "inconsistent"]:
        for method in METHODS:
            h = results[problem][method.name]
            rows.append((
                problem, method.name, len(h["iter"]), h["energy_rel"][-1],
                empirical_q(h), np.mean(h["strict_gain"]), np.max(h["factor_error"]),
                np.min(h["bound_gap"]), np.max(h["eps_hat"]), refs[problem],
            ))
    with open(OUT / "theory_alignment_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\caption{High-accuracy verification of the theory-aligned quantities.}\n")
        f.write("\\label{tab:theory-alignment}\n\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{llrrrrr}\n\\hline\n")
        f.write("Problem & Method & Final $\\mathcal E_k/\\mathcal E_0$ & Emp. $q$ & Frac. $\\theta_k>1+10^{-4}$ & Min. bound gap & Max. $\\widehat\\varepsilon_k$\\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(f"{row[0]} & {row[1]} & {row[3]:.2e} & {row[4]:.4f} & {row[5]:.2f} & {row[7]:.1e} & {row[8]:.1e}\\\\\n")
        f.write("\\hline\n\\end{tabular}%\n}\n\\end{table}\n")

    with open(RESULTS / "theory_alignment.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "problem", "method", "final_energy_ratio", "empirical_q",
            "strict_gain_fraction", "min_bound_gap", "max_epsilon_hat",
        ])
        for row in rows:
            writer.writerow([
                row[0], row[1], f"{row[3]:.2e}", f"{row[4]:.4f}",
                f"{row[5]:.2f}", f"{row[7]:.1e}", f"{row[8]:.1e}",
            ])


def run_inexact_experiment():
    A, b, _, tau = make_gaussian_problem(33, 60, 180, inconsistent=False)
    xbar, _ = solve_bilevel_reference(A, b, tau)
    method = METHODS[2]
    fixed = run_method(A, b, tau, xbar, method, max_iter=80, solve_mode="fixed", fixed_tol=1e-3)
    certified = run_method(A, b, tau, xbar, method, max_iter=80, solve_mode="certified")
    accurate = run_method(A, b, tau, xbar, method, max_iter=80, solve_mode="accurate")
    return {"fixed PG": fixed, "certified": certified, "high accuracy": accurate}


def plot_inexact(results):
    colors = {"fixed PG": "#D55E00", "certified": "#009E73", "high accuracy": "#0072B2"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for name, h in results.items():
        axes[0].semilogy(h["iter"], np.maximum(h["energy_rel"], 1e-16), label=name,
                        color=colors[name], linewidth=1.8)
        axes[1].semilogy(h["iter"], np.maximum(h["eps_hat"], 1e-18), label=name,
                        color=colors[name], linewidth=1.5)
    certified = results["certified"]
    axes[1].semilogy(certified["iter"], certified["eta"], "k--", linewidth=1.2,
                    label=r"summable budget $\eta_k$")
    axes[0].set_ylabel(r"$\mathcal{E}_k/\mathcal{E}_0$")
    axes[1].set_ylabel(r"$\widehat\varepsilon_k$")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "inexact_certificate.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(OUT / "inexact_certificate_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\caption{Fixed projected-gradient stopping versus the computable Fejer certificate.}\n")
        f.write("\\label{tab:inexact-certificate}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\hline\n")
        f.write("Inner rule & Final $\\mathcal E_k/\\mathcal E_0$ & Max. $\\widehat\\varepsilon_k$ & Budget violations & Avg. dual it.\\\\\n\\hline\n")
        for name, h in results.items():
            if name == "certified":
                violations = int(np.sum(h["eps_hat"] > h["eta"] * (1 + 1e-8)))
            else:
                eta = results["certified"]["eta"][: len(h["eps_hat"])]
                violations = int(np.sum(h["eps_hat"][: len(eta)] > eta * (1 + 1e-8)))
            f.write(f"{name} & {h['energy_rel'][-1]:.2e} & {np.max(h['eps_hat']):.2e} & {violations} & {np.mean(h['dual_nit']):.2f}\\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    with open(RESULTS / "inexact_certificate.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "inner_rule", "final_energy_ratio", "max_epsilon_hat",
            "budget_violations", "average_dual_iterations",
        ])
        for name, h in results.items():
            if name == "certified":
                violations = int(np.sum(h["eps_hat"] > h["eta"] * (1 + 1e-8)))
            else:
                eta = results["certified"]["eta"][: len(h["eps_hat"])]
                violations = int(np.sum(h["eps_hat"][: len(eta)] > eta * (1 + 1e-8)))
            writer.writerow([
                name, f"{h['energy_rel'][-1]:.2e}", f"{np.max(h['eps_hat']):.2e}",
                violations, f"{np.mean(h['dual_nit']):.2f}",
            ])


def run_multiseed(seeds=range(10)):
    records = {method.name: [] for method in METHODS}
    for seed in seeds:
        A, b, _, tau = make_gaussian_problem(100 + seed, 50, 150, inconsistent=False)
        xbar, _ = solve_bilevel_reference(A, b, tau)
        for method in METHODS:
            print(f"seed {seed}: {method.name}")
            h = run_method(A, b, tau, xbar, method, max_iter=80, solve_mode="accurate")
            records[method.name].append({
                "iterations": iterations_to(h, 1e-7),
                "final_energy": h["energy_rel"][-1],
                "theta_fraction": float(np.mean(h["strict_gain"])),
                "time": h["runtime"][-1],
            })
    return records


def plot_multiseed(records):
    names = [m.name for m in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    axes[0].boxplot([[r["iterations"] for r in records[n]] for n in names], tick_labels=names)
    axes[0].set_ylabel(r"iterations to $\mathcal{E}_k/\mathcal{E}_0\leq10^{-7}$")
    axes[1].boxplot([[r["time"] for r in records[n]] for n in names], tick_labels=names)
    axes[1].set_ylabel("runtime (s)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "multiseed_robustness.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(OUT / "multiseed_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\caption{Ten-seed robustness study; entries are mean $\\pm$ standard deviation.}\n")
        f.write("\\label{tab:multiseed}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lrrrr}\n\\hline\n")
        f.write("Method & Iterations to target & Final energy ratio & Frac. strict gain & Runtime (s)\\\\\n\\hline\n")
        for name in names:
            rec = records[name]
            vals = [[r[key] for r in rec] for key in ["iterations", "final_energy", "theta_fraction", "time"]]
            f.write(f"{name} & {np.mean(vals[0]):.1f}$\\pm${np.std(vals[0], ddof=1):.1f} & {np.mean(vals[1]):.1e}$\\pm${np.std(vals[1], ddof=1):.1e} & {np.mean(vals[2]):.2f}$\\pm${np.std(vals[2], ddof=1):.2f} & {np.mean(vals[3]):.2f}$\\pm${np.std(vals[3], ddof=1):.2f}\\\\\n")
        f.write("\\hline\n\\end{tabular}%\n}\n\\end{table}\n")

    with open(RESULTS / "multiseed_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "iterations_mean", "iterations_std", "final_energy_mean",
            "final_energy_std", "strict_gain_mean", "strict_gain_std",
            "runtime_mean", "runtime_std",
        ])
        for name in names:
            rec = records[name]
            vals = [[r[key] for r in rec] for key in ["iterations", "final_energy", "theta_fraction", "time"]]
            writer.writerow([
                name, f"{np.mean(vals[0]):.1f}", f"{np.std(vals[0], ddof=1):.1f}",
                f"{np.mean(vals[1]):.1e}", f"{np.std(vals[1], ddof=1):.1e}",
                f"{np.mean(vals[2]):.2f}", f"{np.std(vals[2], ddof=1):.2f}",
                f"{np.mean(vals[3]):.2f}", f"{np.std(vals[3], ddof=1):.2f}",
            ])


def run_scaling():
    sizes = [(150, 50), (300, 100), (600, 200)]
    methods = [METHODS[0], METHODS[2]]
    records = []
    for n, m in sizes:
        for rep in range(3):
            A, b, _, tau = make_gaussian_problem(300 + 10 * n + rep, m, n, inconsistent=False)
            xbar, _ = solve_bilevel_reference(A, b, tau)
            for method in methods:
                print(f"scale n={n} rep={rep}: {method.name}")
                h = run_method(A, b, tau, xbar, method, max_iter=60, solve_mode="accurate")
                records.append({
                    "n": n, "m": m, "method": method.name,
                    "time_per_iter": h["runtime"][-1] / len(h["iter"]),
                    "energy_rel": h["energy_rel"][-1],
                    "theta_fraction": float(np.mean(h["strict_gain"])),
                })
    return records


def plot_scaling(records):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    for method in [METHODS[0], METHODS[2]]:
        ns = sorted({r["n"] for r in records})
        time_mean = [np.mean([r["time_per_iter"] for r in records if r["n"] == n and r["method"] == method.name]) for n in ns]
        energy_mean = [np.mean([r["energy_rel"] for r in records if r["n"] == n and r["method"] == method.name]) for n in ns]
        axes[0].plot(ns, time_mean, "o-", label=method.name, color=method.color)
        axes[1].semilogy(ns, energy_mean, "o-", label=method.name, color=method.color)
    axes[0].set_ylabel("time per iteration (s)")
    axes[1].set_ylabel(r"energy ratio after 60 iterations")
    for ax in axes:
        ax.set_xlabel("dimension n")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "scaling_results.pdf", bbox_inches="tight")
    plt.close(fig)

    with open(OUT / "scaling_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\caption{Moderate-scale study over three random instances per dimension.}\n")
        f.write("\\label{tab:scaling}\n")
        f.write("\\begin{tabular}{rrlrrr}\n\\hline\n")
        f.write("$m$ & $n$ & Method & Time/iter. & Energy ratio after 60 iter. & Frac. strict gain\\\\\n\\hline\n")
        for n, m in sorted({(r["n"], r["m"]) for r in records}):
            for method in [METHODS[0], METHODS[2]]:
                subset = [r for r in records if r["n"] == n and r["method"] == method.name]
                f.write(f"{m} & {n} & {method.name} & {np.mean([r['time_per_iter'] for r in subset]):.4f} & {np.mean([r['energy_rel'] for r in subset]):.2e} & {np.mean([r['theta_fraction'] for r in subset]):.2f}\\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    with open(RESULTS / "scaling_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "m", "n", "method", "time_per_iteration",
            "energy_ratio_after_60_iterations", "strict_gain_fraction",
        ])
        for n, m in sorted({(r["n"], r["m"]) for r in records}):
            for method in [METHODS[0], METHODS[2]]:
                subset = [r for r in records if r["n"] == n and r["method"] == method.name]
                writer.writerow([
                    m, n, method.name,
                    f"{np.mean([r['time_per_iter'] for r in subset]):.4f}",
                    f"{np.mean([r['energy_rel'] for r in subset]):.2e}",
                    f"{np.mean([r['theta_fraction'] for r in subset]):.2f}",
                ])


def main():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 130,
    })
    exact, refs = run_exact_experiments()
    plot_exact(exact)
    write_exact_table(exact, refs)

    inexact = run_inexact_experiment()
    plot_inexact(inexact)

    multiseed = run_multiseed()
    plot_multiseed(multiseed)

    scaling = run_scaling()
    plot_scaling(scaling)

    save_npz(RESULTS / "theory_aligned_experiment_data.npz", exact)
    print("theory-aligned experiments complete")


if __name__ == "__main__":
    main()
