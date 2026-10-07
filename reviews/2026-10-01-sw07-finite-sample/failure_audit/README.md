# Post-run unresolved-draw audit

All original artifacts and decisions remain unchanged. The four scenarios contain **22 unresolved draws** with **88 saved optimizer-start records**.

The recorded causes are separated into selected-solution stationarity failures, disagreement among the four starts, and unsuccessful starts. These are computational diagnostics, not additional model rejections. In particular, SciPy success alone does not override the weighted-residual stationarity check.

Here `nonstationary_solution` means failure of the optimizer's interior first-order condition: the weighted-residual projection onto the local Jacobian column space exceeds its numerical tolerance. It does **not** mean nonstationary DSGE dynamics; all four generating DGPs passed dynamic stationarity checks.

failure_counts.csv reports the original counts by scenario; unresolved_draws.csv and start_diagnostics.csv retain each failure and termination message. Cause flags can overlap in general.

Representative replay: **7/7** saved unresolved/usable decisions reproduced. The replay uses authenticated NPZ targets and their full covariance, the original four starts, fixed calibration, bounds and solver. It does not resimulate data or substitute re-estimated targets. audit.json records each replay, per-start stationarity residual/tolerance, numerical differences, all input hashes and matches against the checkpoint's nine mathematical source hashes and numerical-library versions.

No replay result replaces an original failure or changes a Monte Carlo denominator or tail bound. Tiny optimizer-path differences can arise from numerical state or evaluation history; any replay disagreement remains visible in audit.json.

Reproduce from the repository root with `.venv/bin/python reviews/2026-10-01-sw07-finite-sample/audit_failures.py`. Use `--summary-only` to summarize existing records without optimizer calls.
