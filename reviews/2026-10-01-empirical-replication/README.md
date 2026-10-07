# Observed-data structural study and published-result replication

These additions implement the two selected research directions. The
[development protocol](protocol.md) records the original choices and the
subsequent exploratory calibration sensitivity. This is not an externally
registered analysis plan.

## Observed macro moments to a structural model

The [SW07 report](structural/report.md) uses the authenticated revised-FRED
snapshot, 156 quarters from 1966Q1 through 2004Q4. Nine empirical covariance
moments fit policy smoothing and monetary innovation volatility; six unused
same-sample moments are descriptive checks. The complete 15-by-15 Bartlett
covariance is retained. Stationary model moments use a Lyapunov solve,
independently checked by a long moving-average sum.

The declared Pfeifer calibration produces a boundary solution (`crr=0.98`,
`em=0.01`) and criterion 396736.10. No regular J p-value or parameter confidence
intervals are reported. The exploratory published SW07 calibration gives
`crr=0.83826`, `em=0.25409`, criterion 25.25541 and conditional J p-value 0.000684
with seven degrees of freedom. It rejects the selected moment restrictions.
Both results remain visible; the second calibration was added after the first
outcome and does not replace it.

This is a conditional minimum-distance application, not a reproduction of the
paper's posterior or original data vintage. High persistence, possible regime
changes, finite-sample demeaning/HAC error and unpropagated fixed-calibration
uncertainty limit its interpretation. Reserved moments are not independent
validation data. No ordinary parameter confidence intervals are reported.

Reproduce:

```bash
python -m puremacro.examples.empirical_sw07_matching --output output/sw07
```

## Original-data published replication

The [Romer–Romer report](rr2010/report.md) uses the authors' original archive and no-controls
`EXOGERNR.RAT` specification: quarterly GDP growth on an intercept and tax
changes at lags zero through twelve, 1950Q1–2007Q4. Conventional OLS covariance
and its complete cumulative transformation reproduce the published definition.
The reference exporter uses separate statsmodels software without importing
puremacro. An offline runtime comparison is not a new RATS execution.

Full coefficients, covariance matrices, responses, standard errors, t
statistics and sample counts have a tight software-parity comparison. The
separate published comparison checks horizon ten, response -3.08 and t statistic
-3.53, with only the paper's two-decimal rounding tolerance. It reproduces one
baseline result, not every robustness exercise or a new validation of narrative
shock exogeneity. The earlier modified LP gallery entry has a different sample
and estimator and has been relabeled accordingly.

Independent statsmodels parity passes: the largest coefficient difference is
8.88e-16 and the largest cumulative-covariance difference is 6.66e-16. The
response is **-3.0808127631**, with its trough at quarter ten, reproducing those
printed results. The t statistic is **-3.5249960750**, which misses the printed
-3.53 rounding interval by **0.000003925**. An independent read of the authors'
original RATS databanks corroborates this discrepancy. Its cause is unresolved.
The selected statistic is retained and its tolerance unchanged.

The [eight-case benchmark dossier](benchmarks/benchmark_report.md) therefore
has **seven passes and one published-comparison failure**. The tax application
and full benchmark CLI intentionally exit with code 1. Tests verify preservation
of this scientific failure, so passing software tests must not be interpreted
as agreement with every published target.

Reproduce:

```bash
python -m puremacro.examples.romer_romer_2010_replication --output output/rr2010
python tools/run_research_benchmarks.py --output output/benchmarks
```

The [English guide](../../docs/empirical_research.md) and
[Spanish guide](../../docs/es/empirical_research.md) document both APIs and
their assumptions. Numerical source resources ship in the wheel; reference
software and workbook readers are development dependencies only.

## Validation record

- Structural, bridge, existing research-application and SW07 observation
  regressions: 127 passed, 1 deselected; see [log](structural-tests.log).
- Replication, application and independent-benchmark checks: 71 passed; see
  [log](replication-tests.log).
- Bilingual documentation and optional-dependency/import checks: 11 passed;
  see [log](docs-import-tests.log).
- Executable Python examples in ten English/Spanish research guides: 10 passed;
  see [log](docs-execution-tests.log). Total selected tests: 219 passed.
- Strict MkDocs build passed. A fresh wheel installed outside the repository
  runs both applications and all eight benchmarks with network access blocked
  and no statsmodels import. It reproduces seven benchmark passes, the retained
  published t-statistic failure, withheld SW07 inference and the RR CLI's
  intentional exit code 1. All 970 copied package/build-input files match the
  final checkout. See [validation summary](wheel-validation/summary.json),
  [installed execution](wheel-validation/installed_smoke.json) and
  [strict docs log](wheel-validation/mkdocs-strict.log).
- The global API snapshot has nine pre-existing unrelated differences. Only
  this task's new module exports, result fields and shared re-exports are
  deliberately incorporated into the snapshot. No release is published.
  See the [API comparison](api-snapshot-audit.json).

The [independent source audit](rr2010_source_audit.md) includes a separate
XLSX-XML reader and RATS binary reader, with SVD/QR comparisons against the
frozen external reference. No original RATS executable was run. The reference
exporter and source archive hashes make the evidence traceable without claiming
to explain the printed t-statistic discrepancy.
