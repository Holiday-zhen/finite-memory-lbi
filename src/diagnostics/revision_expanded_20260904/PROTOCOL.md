# Expanded experimental revision

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run / revision
- Origin Date: 2026-09-04
- Verification Status: UNVERIFIED (protocol before runs)
- Version Label: expanded_v1

User scope: remove old 5.2 completely; correct energy accuracy, tolerance and fitting issues; explain three inner stopping rules; update two-problem long-run study; increase instances; deliver experiment-section LaTeX/PDF on Desktop.

Sample sizes frozen before new results: calibration 10 independent generated instances per sweep (7100--7109 solver; 7150--7159 memory/trigger); energy/gain 20 paired instances (9100--9119); certificate 20 (5000--5019); scale 20 per size (10100+n+s, s=0..19). Hard problems: old seeds20--29 plus new30--39, all20 shown; old/new groups also retained separately. Digits resamples a finite shared image dataset, so describe independent randomized problem constructions, not 20 independent patient/population datasets.

Descriptive mean/median/IQR only; no bootstrap or confidence intervals. No outcome-based seed selection. Preserve previous outputs and source package. Calibration uses original selection rules with expanded samples and 3-repeat median timing; freeze selected configuration before tests. Numerical accuracy overrides for energy will be explicit.

Energy: two reference acceptance tolerances 1e-10 and 1e-12, save actual feasibility/KKT/gap/status, untruncated stable and direct-subtraction energy, compare sensitivity on identical algorithm trajectories. Common requested multicut KKT tolerance1e-12 for every selection rule. Fit log relative energy against k over the same prespecified displayed window40--75, exp(slope), not adjacent ratios. Do not fit numerical-floor values; failed quality checks must be reported and corrected before manuscript claims.

Certificate: clarify that all three use L-BFGS-B, PG refers to its projected-gradient residual diagnostic, not a different outer method. Fixed tolerance1e-3, certificate budget eta0/(k+1)^1.5, high accuracy1e-12. Explicitly fail certification if tightening sequence exhausts without meeting budget. Use stable Bregman arithmetic and accurate references here too.

Hard problems: same frozen q3/p6/trigger.001 and strict float64 implementation from extended_horizon_20260904. 10000 iterations; exact-zero only early exit. Show full trajectories only, concise two-panel titles; terminal mean/median/IQR and per-instance data in tables. New instances not screened for advantage. Matrix precision-floor limitations remain explicit.

All metadata/code/raw trajectories and fit diagnostics are archived. Desktop output created in a new folder, not overwriting old manuscripts. Old section IDs map after deletion: 5.1->5.1, 5.3->5.2, 5.4->5.3, 5.5->5.4, 5.6->5.5, 5.7->5.6.

## Numerical acceptance amendment (before final reporting)

The first energy rerun used the common requested tolerance but some projected BB solves exhausted 600 iterations above that tolerance (maximum 6.20e-11). Preserve this pilot in data/energy. The final data/energy_verified applies the same deterministic projected-gradient polishing to every multicut solve whose measured residual exceeds 1e-12, starting from its BB output, with step 1/||G||^2 and at most 20000 polishing steps. Failure raises an exception; it is not silently accepted. All 20 instances are rerun with this common rule; no instance is removed. Final plotting and fitting use energy_verified only.

Calibration saves aggregate CSVs and deterministic seeds/driver, not individual calibration trajectories. All reported test trajectories are saved individually. The hard-problem endpoint audit re-evaluates the saved floating-point states at 50 decimal digits; it is not an arbitrary-precision algorithm rerun.
