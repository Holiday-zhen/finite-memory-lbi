from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


OUT = Path(__file__).resolve().parents[2] / "generated_figures" / "png1.png"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # Current point, its one-cut projection, and the projection onto two cuts.
    xk = np.array([2.4, 2.0])
    yk = np.array([0.5, 2.0])
    zk = np.array([0.5, 0.7])

    # C = {x_1 <= 0.5, x_1 + x_2 <= 1.2} in the plotting window.
    poly = Polygon(
        [(-1.1, -0.6), (0.5, -0.6), (0.5, 0.7), (-1.1, 2.3)],
        closed=True,
        facecolor="#DCEAF7",
        edgecolor="none",
        alpha=0.85,
        hatch="///",
    )
    ax.add_patch(poly)

    yy = np.linspace(-0.6, 2.8, 100)
    ax.plot(np.full_like(yy, 0.5), yy, color="#0077BB", linewidth=2.0, label=r"current boundary $\partial H_k$")
    xx = np.linspace(-1.1, 2.8, 100)
    ax.plot(xx, 1.2 - xx, color="#EE7733", linewidth=2.0, label=r"retained boundary $\partial H_j$")

    ax.annotate("", xy=yk, xytext=xk, arrowprops=dict(arrowstyle="->", color="#0077BB", linewidth=2.0))
    ax.annotate("", xy=zk, xytext=xk, arrowprops=dict(arrowstyle="->", color="#CC3311", linewidth=2.2))
    ax.annotate("", xy=zk, xytext=yk, arrowprops=dict(arrowstyle="<->", color="#555555", linestyle="--", linewidth=1.2))

    ax.scatter(*xk, s=55, color="black", zorder=5)
    ax.scatter(*yk, s=55, color="#0077BB", zorder=5)
    ax.scatter(*zk, s=55, color="#CC3311", zorder=5)
    ax.text(xk[0] + 0.08, xk[1] + 0.08, r"$x_k$", fontsize=11)
    ax.text(yk[0] + 0.08, yk[1] + 0.08, r"$y_k=P_{H_k}(x_k)$", fontsize=10, color="#0077BB")
    ax.text(zk[0] + 0.08, zk[1] - 0.18, r"$z_k=P_{H_k\cap H_j}(x_k)$", fontsize=10, color="#CC3311")
    ax.text(0.62, 1.30, r"$\delta_{j,k}>0$", fontsize=10, color="#555555")
    ax.text(-0.85, -0.35, r"$C_k=H_k\cap H_j$", fontsize=10)
    ax.text(1.25, -0.35, r"$\theta_k=\Delta_k^{\rm mem}/\Delta_k^{\rm one}>1$", fontsize=10)

    ax.set_xlim(-1.1, 2.9)
    ax.set_ylim(-0.6, 2.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
