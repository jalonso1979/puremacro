# Empirical study and published-result replication protocol

Design selected on 2026-10-01 before inspecting the new estimation results.
This is a development record, not an externally registered analysis plan.

## Study A: observed macro moments to structural parameters

- Model: the existing first-order Smets–Wouters implementation and its
  explicit observation mapping. This is a new minimum-distance application,
  not a reproduction of the paper's Bayesian posterior.
- Data: the bundled revised-FRED snapshot, 1966Q1–2004Q4, with exact file
  hash, source definitions, build date and transformations retained.
- Parameters: policy smoothing `crr` and monetary innovation standard
  deviation `em`; other structural coefficients and innovation variances
  fixed at an explicitly recorded calibration.
- Observables: per-capita real GDP growth, GDP-deflator inflation and the
  federal funds rate, all in the model's quarterly percentage units.
- Fitted targets: six unique contemporaneous covariances and three own
  lag-one covariances. Reserved targets: own lag-two and lag-four covariances.
  All products use a common window with four initial observations reserved
  for lags, and full-sample means.
- Sampling covariance: full joint Bartlett HAC of the centered product
  scores. No diagonal approximation, ridge or sample-size double scaling.
- Model moments: stationary state covariance from the discrete Lyapunov
  equation, propagated through the observation matrix. Exclude the numerical
  Kalman measurement ridge. Check stationarity and equation residuals.
- Optimize from multiple initial values, report profile criteria, and report
  declared bandwidth sensitivities. Keep the chosen moments even if model
  fit is poor. Failures or binding bounds must remain visible.
- Correct-specification J test only under numerical regularity. Rejection
  at 5% is reported as evidence against this conditional specification;
  ordinary correctly-specified confidence intervals are withheld in that
  presentation. Nonrejection does not establish correct specification, and
  any displayed local uncertainty remains conditional on the assumptions.
- Reserved moments use the same observations and are descriptive checks,
  not independent validation samples. Fixed calibration uncertainty,
  structural breaks and identification of shocks from external instruments
  are outside this application.

Structural shocks here are latent and restricted by the model. The study
does not relabel a narrative/HFI instrument as an observed structural shock.
A future external-instrument IRF application needs joint IV inference and
normalization appropriate to that design.

### Post-result diagnostic addition

The first run under the declared bundled calibration reached both parameter
bounds with a large criterion. Independent moving-average summation agreed
with the Lyapunov moments; the calibration includes highly persistent shocks.
An explicitly exploratory sensitivity using the published SW07 posterior-mode
coefficients was therefore requested. The original baseline is retained, and
the comparison must not select whichever calibration fits better as a new
baseline. Published coefficients are also estimated quantities, so treating
them as fixed still omits calibration uncertainty. This addition occurred
after observing the initial boundary outcome.

## Study B: Romer–Romer (2010), Figure 4 baseline

- Authenticate the original authors' `DataSet.zip`, workbook and RATS
  `EXOGERNR.RAT` specification. Preserve original vintage and calendar.
- Reproduce the baseline distributed-lag regression with an intercept and
  contemporaneous through twelve quarterly lags of the exogenous tax series.
  The dependent variable is quarterly `100 * delta log(real GDP)` and the
  estimation sample is 1950Q1–2007Q4.
- Construct the tax measure exactly as in the authors' code:
  `100 * (DEFIC + LONGR) / NOMGDP`.
- Reproduce the original conventional OLS covariance and cumulative response
  transformation. Do not substitute HAC covariance or local projections and
  call that the original published specification.
- Run an independent software implementation without importing puremacro.
  Preserve complete coefficients, covariance and cumulative responses with
  software versions and source hashes. Reference regeneration is explicit.
- Compare the published horizon-ten response and t statistic (-3.08 and
  -3.53, rounded in the paper) separately from tight full-precision numerical
  comparisons. Verify the trough horizon. Negative controls must fail when
  outputs, signs, horizons, source files or uncertainty scaling are wrong.
- Scope: one baseline result, not every figure, robustness specification,
  narrative classification or causal-identification assumption in the paper.

The earlier modified Romer–Romer local-projection example has a different
sample, vintage and estimator. It is retained as a modified application and
must not serve as this replication's reference.

### Observed discrepancy retained

The original-data independent OLS run produces the published response and
trough, but t=-3.5249960750 misses the paper's -3.53 rounding interval by
0.000003925. The original RATS databanks corroborate this result. The selected
t statistic stays in the benchmark and retains its original 0.005 absolute
tolerance, so the published case fails. Numerical software parity is a
separate passing comparison. No selected metric was dropped and no threshold
was enlarged after observing this result.

## Delivery checks

Both studies must run offline from packaged inputs and export human-readable
reports, machine-readable provenance and figures. Tests must include independent
numerical references and meaningful failure paths. English and Spanish guides,
public API metadata, package-data rules and installed-wheel execution are checked.
Existing unrelated working-tree changes and API snapshot differences remain
separate from these additions. No release publication is authorized by this task.
