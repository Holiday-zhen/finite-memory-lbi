# Finite-memory cut-and-project LBI: reproducibility code

This repository contains the Python code used for the numerical experiments in
**Finite-memory cut-and-project linearized Bregman iterations**.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the figures

The root driver keeps the experiment groups separate because the complete
multi-seed runs are computationally intensive.

```bash
# Verify imports and generate the inexpensive geometric illustration
python reproduce_figures.py --smoke

# Figure 1
python reproduce_figures.py --group figure1

# Sections 5.1--5.4: calibration, energy, gain, and certificates
python reproduce_figures.py --group core

# Section 5.5: sparse large-scale problems
python reproduce_figures.py --group scale

# Section 5.6: two matrix problems
python reproduce_figures.py --group matrix

# Run every group
python reproduce_figures.py --group all
```

Generated results are written inside the corresponding experiment directories.
The long-horizon groups may require substantial CPU time. Thread counts and
random seeds are fixed by the experiment drivers.

## Repository layout

- `SOURCE_FILES.md`: short description of the retained source modules.
- `src/diagnostics/revision_expanded_20260904/`:
  Sections 5.1--5.4 and numerical-accuracy checks.
- `experiments/inner5_tuning/`: final Section 5.5 and Section 5.6 drivers.
- `experiments/pq_followup/` and the small diagnostic helper modules: problem
  builders and multi-cut routines used by Section 5.6.
- `reference_figures/`: the nine PNG figures used in the manuscript, named in
  manuscript order (`png1.png` through `png9.png`).
- `reference_results/`: compact CSV/JSON summaries from the reported runs.

The Binary Digits experiment uses `sklearn.datasets.load_digits`; no manual
dataset download is required.

The repository was syntax-checked, and the `--smoke` command was run
successfully on the packaged directory before delivery.

## Notes

- Core timings exclude plotting and post-processing.
- The one-cut baseline uses the scalar one-cut solve; the multi-cut variants use
  the projected-BB inner solver specified in the experiment drivers.
- The helper module under `shutdown_hard_benchmark/` supplies shared problem
  classes and builders. The reported Section 5.6 experiment does not activate a
  shutdown rule.
