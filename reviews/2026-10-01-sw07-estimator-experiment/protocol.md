# Controlled SW07 estimator experiment — 2026-10-01

Development protocol written before running this experiment's Monte Carlo
draws. It follows the completed finite-sample diagnosis; it is not an externally
registered analysis plan. The earlier study and its results remain unchanged.

## Fixed design

- One primary DGP: the complete rounded published SW07 calibration, including
  true `crr=0.81`, `em=0.24`. This isolates estimator behavior where the previous
  nominal 5% chi-square rule rejected approximately 62–64% of generated samples.
  It does not resolve calibration uncertainty or validate a composite null.
- Cross population versus exact finite-sample expected moments with per-draw
  HAC8 versus fixed exact Gaussian sampling covariance. Four variants:
  `population_hac`, `finite_hac`, `population_oracle`, `finite_oracle`.
- The oracle covariance is evaluated once at the known generating parameters
  and held fixed during every fit. It is a control, not an estimated feasible
  covariance or a proposed general-purpose production estimator. Exact expected
  moments vary with candidate `crr/em` throughout optimization.
- Preserve 156 quarters, stationary Gaussian initial states, the exact original
  sample-centering and 152-quarter common window, all15 moments, fit9/reserve6,
  four original optimizer starts, bounds, tolerance1e-9 and500 evaluations.
  Use exact calculations only, without interpolation or simulation inside fits.
- Each simulated sample and empirical HAC matrix is shared by all four variants.
  Preserve per-start diagnostics and every unresolved outcome. Boundary fits
  with otherwise usable optimization remain in criterion distributions.
- Calibration sample:399 independent draws. Validation sample:999 independent
  draws. Master seed20261002; SeedSequence([seed,phase_index,replication]), with
  phase indices0/1. These streams are disjoint from the preceding experiment.
  Smoke tests use different seeds and are excluded from scientific results.

## Comparisons selected before results

For every variant report the naive chi-square(7) 5% rejection frequency,
numerical failure/boundary frequencies, conditional parameter bias/RMSE, and
criterion distribution. Report paired differences on common usable samples
with their denominators; these selected comparisons are descriptive.

Use the calibration sample to determine a simulation-reference upper-tail
cutoff at rank ceil(0.95*(B+1)) with strict rejection above the cutoff. Missing
criteria are bounded by zero and positive infinity, producing a cutoff interval.
No failed fit is replaced or treated as a known nonrejection. In validation,
lower rejection counts require a usable criterion above the upper cutoff;
upper counts include usable criteria above the lower cutoff plus all unresolved
fits. Report 95% Clopper–Pearson Monte Carlo envelopes. These validation
intervals are conditional on the realized calibration pool; they are not
parameter intervals or a measurement of all cutoff-estimation uncertainty.

Inspect independent validation results only after defining all four variants
and cutoff rules. No estimator is selected using validation outcomes and then
described as independently validated. Agreement with a nominal5% rule at one
known DGP is diagnostic evidence only. No automatic replacement of the existing
empirical estimator, no parameter confidence intervals, and no claim of a
general inference repair follows from this experiment.

Observed-data fits for each variant are descriptive applications. Covariance
weights anchored to the published calibration and its fitted moment map do
not independently validate that calibration. Criteria across different weights
cannot be compared as if they had a common absolute scale.

## Reproducibility and stopping

Save every sample seed/hash, all15 target moments and HAC covariances, all
four optimizer-start records per fit, source/environment fingerprints and
atomic checkpoints. Resume only identical designs. Complete all preselected
draws regardless of outcomes. Any later sensitivity is separately labeled.
Independent tests compare the optimized exact-expectation implementation with
the existing dense Gaussian oracle. An independent results audit recomputes
counts, cutoff bounds and parameter recovery from saved artifacts.
