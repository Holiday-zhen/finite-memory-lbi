"""Build compact tables and a report from the completed final experiments."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "final_exp55_exp56"
OUT.mkdir(parents=True, exist_ok=True)


def read(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sci(value):
    return f"{float(value):.2e}"


def fixed(value):
    return f"{float(value):.4f}"


def interval(median, q1, q3, formatter):
    return f"{formatter(median)} [{formatter(q1)}, {formatter(q3)}]"


def main():
    exp55 = read(RESULTS / "final_exp55" / "summary.csv")
    exp56 = read(RESULTS / "final_exp56" / "summary.csv")
    lines = [
        "# Final Experiment 5.5 and 5.6 results",
        "",
        "All multi-cut projections use warm-started projected-gradient BB with at most five updates per outer iteration. The one-cut baseline uses a high-accuracy scalar solve. Every row summarizes 20 matched instances over 2000 outer iterations. Algorithm time excludes residual, objective, plotting, and aggregation costs.",
        "",
        "## Experiment 5.5",
        "",
        "| Size | Method | Residual at 200 | Final residual, median [IQR] | Objective, median [IQR] | Core time (s), median [IQR] | Tail inner PG residual |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in exp55:
        lines.append(
            f"| ({row['m']}, {row['n']}) | {row['method']} | "
            f"{sci(row['residual_200_median'])} | "
            f"{interval(row['final_residual_median'], row['final_residual_q1'], row['final_residual_q3'], sci)} | "
            f"{interval(row['objective_median'], row['objective_q1'], row['objective_q3'], fixed)} | "
            f"{interval(row['core_time_median_s'], row['core_time_q1_s'], row['core_time_q3_s'], lambda x: f'{float(x):.3f}')} | "
            f"{sci(row['tail100_pgn_median'])} |"
        )
    lines.extend([
        "",
        "## Experiment 5.6",
        "",
        "| Problem | Method | Residual at 200 | Final residual, median [IQR] | Objective, median [IQR] | Core time (s), median [IQR] | Tail inner PG residual |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in exp56:
        lines.append(
            f"| {row['problem'].replace('_', ' ')} | {row['method']} | "
            f"{sci(row['median_residual_200'])} | "
            f"{interval(row['median_final_residual'], row['final_residual_q1'], row['final_residual_q3'], sci)} | "
            f"{interval(row['objective_median'], row['objective_q1'], row['objective_q3'], fixed)} | "
            f"{interval(row['median_final_core_time_s'], row['core_time_q1_s'], row['core_time_q3_s'], lambda x: f'{float(x):.3f}')} | "
            f"{sci(row['median_tail100_pgn'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- In Experiment 5.5, memory reduces the residual strongly at 200 iterations. At 2000 iterations, violated has the lowest median residual, whereas recent and angle can be limited by the five-step inner budget. Objectives agree to the displayed scale.",
        "- In Column-block completion, all methods ultimately reach the same double-precision residual band and objective scale. Memory mainly improves early iteration counts on the difficult instances.",
        "- In Binary Digits, recent, violated, and angle reach a computed zero residual on all 20 instances, while one-cut stops near 1e-17. Their objectives are not identical: violated and angle stay closer to one-cut than recent, whereas recent and adaptive have visibly higher objective values.",
        "- A computed zero means machine precision in this implementation, not a mathematical proof of exact zero.",
    ])
    (OUT / "RESULTS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
