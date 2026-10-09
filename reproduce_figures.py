"""Reproduce the manuscript's numerical figures by experiment group."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CORE = (
    ROOT
    / "src"
    / "diagnostics"
    / "revision_expanded_20260904"
)
INNER = ROOT / "experiments" / "inner5_tuning"
FIGURE1 = (
    ROOT / "src" / "experiments" / "make_theta_geometry.py"
)


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONHASHSEED", "0")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(name, "1")
    return env


def run(script: Path, *arguments: str) -> None:
    command = [sys.executable, str(script), *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=script.parent, env=environment(), check=True)


def smoke() -> None:
    run(FIGURE1)
    commands = (
        [sys.executable, "-c", "import run_tuning",],
        [sys.executable, "-c", "import run_exp56_long_horizon",],
    )
    for command in commands:
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=INNER, env=environment(), check=True)
    print("Smoke test passed.")


def core() -> None:
    run(CORE / "calibrate.py")
    run(CORE / "run_studies.py", "energy", "--verify-inner")
    run(CORE / "run_studies.py", "certificate")
    run(CORE / "run_studies.py", "scale")
    run(CORE / "analyze.py", "--core-only")
    run(CORE / "make_tables.py", "--core-only")


def scale() -> None:
    run(INNER / "run_exp55_final.py")


def matrix() -> None:
    run(INNER / "run_exp56_long_horizon.py")
    run(INNER / "make_exp56_k500.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run import checks and Figure 1 only.")
    parser.add_argument(
        "--group",
        choices=("figure1", "core", "scale", "matrix", "all"),
        default="figure1",
    )
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.group in ("figure1", "all"):
        run(FIGURE1)
    if args.group in ("core", "all"):
        core()
    if args.group in ("scale", "all"):
        scale()
    if args.group in ("matrix", "all"):
        matrix()


if __name__ == "__main__":
    main()
