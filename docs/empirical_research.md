> 🇬🇧 English · 🇪🇸 [Español](es/empirical_research.md)

# Real-data structural estimation and published replication

These two studies extend the [research workflows](research_workflows.md) with
observed macro data and an original-paper specification. They answer different
questions: how a conditional structural model fits observed moments, and
whether puremacro reproduces a specified published result.

## Smets–Wouters: observed moments to structural parameters

```bash
python -m puremacro.examples.empirical_sw07_matching --output research_output/empirical_sw07
```

This is a new minimum-distance application of the existing first-order
Smets–Wouters model, not a replication of the paper's Bayesian posterior.
The data are the bundled **revised FRED vintage**, covering 1966Q1–2004Q4.
They follow the definitions in the [original data appendix](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf),
but are not the authors' historical data vintage. The manifest preserves the
input checksum, build date, source series and transformations. Loading verifies
the reviewed SHA256 (after normalizing line endings), all seven source columns
and the complete 156-quarter sample; a changed vintage requires deliberate review.

The application estimates two parameters: monetary-policy smoothing `crr` and
the standard deviation of the monetary innovation `em`. All other coefficients
and innovation standard deviations are fixed at the explicitly recorded
**Pfeifer model calibration**, which is not the published SW07 posterior mode.
Uncertainty in that calibration is not estimated. Structural shocks
are latent and restricted by the model; no external instrument is treated as
an observed structural shock.

The three observables are:

| Series | Transformation | Unit |
|---|---|---|
| Real GDP growth | `100 * dlog(GDPC1 / CNP16OV)` | Percent per quarter |
| GDP-deflator inflation | `100 * dlog(GDPDEF)` | Percent per quarter |
| Federal funds rate | Quarterly average of `FEDFUNDS / 4` | Quarterly percentage rate |

```python
from puremacro.structural import fit_empirical_sw07

study = fit_empirical_sw07(profile_points=5)
print(study.metadata["status"])
print(study.fit.summary())
print(study.fit.moment_fit(held_out=True))
```

The fitted moments are the six unique contemporaneous covariances of these
series and their three own lag-one covariances. Own lag-two and lag-four
covariances are reserved as descriptive checks. All products use the same
152-quarter window after reserving four initial quarters for lags, with
full-sample estimated means. The full 15-by-15 joint Bartlett HAC covariance
retains cross-moment uncertainty;
it is the covariance of the estimated moments, not an unscaled long-run
covariance. This construction assumes stationary, weakly dependent data with
finite fourth moments and an appropriate bandwidth. Demeaning has zero
population first-order derivative for stationary central covariances; this is
an asymptotic argument, not an exact finite-sample correction.

For state transition `T`, shock loading `R`, innovation covariance `Q` and
observation matrix `Z`, theoretical covariances come from

\[
P=TPT'+RQR',\qquad \Gamma_h=ZT^hPZ'.
\]

The numerical Kalman measurement ridge is excluded from these economic
moments. Model stationarity and the Lyapunov residual are checked. Multiple
optimizer starts and profile criteria help expose numerical and identification
problems; they do not prove global or strong statistical identification.

### How to read an unsuccessful fit

Optimization success and empirical adequacy are separate results. A binding
parameter bound, rank deficiency or failed stationarity check must remain
visible. The conditional overidentification test can reject the joint model,
calibration and selected moments. A rejection is a valid outcome of the study;
the application must not change targets or tolerances to obtain acceptance.

Ordinary correctly-specified parameter uncertainty is withheld by default.
`--allow-conditional-inference` requests it explicitly, but rejection,
boundary or other failed diagnostics still withhold it. Nonrejection does not
establish correct specification, and any local standard errors available remain
conditional on smoothness, identification, covariance consistency and the fixed
calibration. Profile criteria are descriptive, not weak-identification-robust
confidence sets. Reserved moments use the same sample, so they are not independent
validation data. Structural breaks and calibration uncertainty need additional
research designs.

The default run retains the baseline and reports the following outcomes:

| Fixed calibration | `crr` | `em` | Minimum-distance criterion | Interpretation |
|---|---:|---:|---:|---|
| Declared Pfeifer baseline | 0.98 | 0.01 | 396736.10 | Both bounds bind; `nonregular_fit`, with no regular J reference or parameter SEs |
| Exploratory published SW07 mode | 0.83826 | 0.25409 | 25.25541 | Interior fit; asymptotic J reference gives `p ≈ 0.000684` (7 degrees of freedom), but subsequent simulation finds severe finite-sample distortion |

The exploratory sensitivity was added **after** observing the baseline boundary
outcome. It fixes the remaining parameters at the full posterior-mode columns
of SW07 Tables 1a/b and reestimates the same two parameters, without changing the
baseline or targets. The much smaller criterion shows how much this exercise
depends on the fixed calibration; it does not establish empirical adequacy.
Those published coefficients were estimated on an overlapping historical sample
and are not independent validation. Their uncertainty is not propagated, and no
normal parameter inference is reported for this sensitivity.

The baseline transition has spectral radius 0.9977. With only 156 quarters,
this persistence makes finite-sample demeaning and HAC approximations a material
limitation even though stationary population moments exist. Bandwidths 4, 8 and
12 and descriptive historical sample splits are exported to expose sensitivity;
they do not correct near-unit-root inference or possible regime changes.
`calibration_sensitivity.csv` compares the two fits; standardized discrepancies
in the moment tables divide by target SEs and are descriptive, not fitted-residual
t tests. The default run reports no normal parameter confidence intervals.

The subsequent [finite-sample diagnosis](sw07_finite_sample.md) checks the data
mapping, exact demeaning bias, HAC estimation and the same bounded fitting
procedure under stationary Gaussian simulations. Its conditional simulation
fractions do not replace these historical results with validated bootstrap
p-values or restore ordinary parameter intervals. With 399 draws from the fixed
published calibration, 41.60–43.86% exceed the observed criterion, and the nominal
5% chi-square rule rejects 62.16–64.41% of model-generated samples. These ranges
retain unresolved fits. The small asymptotic p-value above is therefore not
reliable evidence of model rejection at this sample size. The declared
baseline remains an extreme mismatch, dominated by its risk-premium shock scale.

## Romer–Romer (2010): original baseline tax response

```bash
python -m puremacro.examples.romer_romer_2010_replication --output research_output/rr2010
```

This study checks one baseline result in
[Romer and Romer (2010), *The Macroeconomic Effects of Tax Changes*](https://www.aeaweb.org/articles?id=10.1257/aer.100.3.763).
The original workbook and `EXOGERNR.RAT` specification come from the
[authors' data archive](https://eml.berkeley.edu/~dromer/papers/DataSet.zip).
Their hashes and the derived data/reference hashes are retained.

```python
from puremacro.replication import estimate_rr2010_baseline

replication = estimate_rr2010_baseline()
print(replication.to_frame().loc[10])
# Preserve the complete response covariance when using these empirical targets.
targets = replication.to_moment_targets()
print(targets.labels)
```

For 1950Q1–2007Q4, the baseline regression explains
`100 * delta log(real GDP)` with a constant and contemporaneous through twelve
lags of `100 * (DEFIC + LONGR) / NOMGDP`. Cumulative sums of the thirteen tax
coefficients give the response of the GDP level to a tax increase equal to
one percent of GDP. Positive tax shocks mean tax increases.

The original specification uses conventional OLS covariance with the residual
degrees-of-freedom adjustment. Cumulative-response covariance applies the same
linear cumulative-sum transformation to the full coefficient covariance.
HAC and local projections can be useful extensions, but they change this
published specification.

Two different comparisons are reported:

1. **Numerical parity:** all coefficients, covariance entries and cumulative
   responses against a frozen independent-software run that does not import
   puremacro. Tolerances allow numerical roundoff.
2. **Published evidence:** the horizon-ten trough, GDP response `-3.08` and
   t statistic `-3.53`, reported on printed page 781. Only published rounding
   justifies the looser tolerance here; it is not used for software parity.

The original data give a horizon-ten response of **-3.0808127631** and
t statistic **-3.5249960750**. Software parity passes, and the response and trough
horizon reproduce the printed results. The t statistic misses the -3.53
rounding interval by **0.000003925**. An independent read of the authors' RATS
databanks corroborates this value. Its source is unresolved; neither intermediate
rounding nor a wider tolerance is used to force agreement. The published case
and application therefore report **FAIL**, with a nonzero command exit status.
This is a verified discrepancy, not a claim that all published targets agree.

The scope is this original-data baseline. It does not reproduce every
robustness exercise or independently establish the narrative exclusion
assumption. The earlier `regression.romer_romer_tax_multiplier_ols` gallery
case is a modified horizon-eight LP with a different sample and revised data;
its coarse headline comparison is explicitly labeled as such.

## Reproducibility

The applications run offline with packaged numerical inputs and reference
evidence. Exported reports include data/model definitions, sample and units,
estimates, uncertainty diagnostics and figures. Source conversion and independent
reference regeneration are development steps, distinct from the offline
runtime. Reference software is not a runtime dependency.

Use the [structural bridge guide](structural_bridge.md) for covariance conventions
and the [benchmark guide](research_benchmarks.md) for provenance, strict comparisons
and negative controls.
