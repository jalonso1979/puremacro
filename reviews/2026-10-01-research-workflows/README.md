# Research workflows: implementation and verification

2026-10-01. Unreleased work on top of the existing 4.3.0 working tree.
The tree already contained extensive unrelated changes; no release or commit
was created. [Source hashes](source_hashes.json) identify these new components.

## Delivered

1. **Independent benchmarks.** Six offline cases cover correlated GLS,
   expenditure minimization, distributional incidence, an analytical growth
   solution, authenticated live Dynare reference output, and official ENIGH
   totals. Cases record observed/reference values, provenance, units,
   tolerances and deliberately failing negative controls.
   [Benchmark report](benchmarks/benchmark_report.md) ·
   [Machine-readable evidence](benchmarks/benchmark_report.json).
2. **Empirical-to-structural bridge.** Labeled empirical moments retain their
   full estimator covariance. Joint LP HAC covariance feeds bounded structural
   minimum distance, with sandwich uncertainty, local identification and
   stationarity checks, conditional J tests, and descriptive held-out moments.
   The complete NK application estimates `(sigma, kappa, rho)` as
   `(1.469875, 0.152528, 0.594417)` from independently simulated data with truths
   `(1.5, 0.15, 0.6)`.
   [Application report](structural/report.md) ·
   [IRF figure](structural/irf_fit.png) · [Manifest](structural/manifest.json).
3. **Distributional trade.** Group-specific purchaser prices, factor-income
   exposure and explicit transfer allocations produce exact conditional EV/CV,
   population-weighted totals and price/income/transfer attribution. An adapter
   accepts audited trade GE results with supported consumption baskets. The
   observed-data application bundles authenticated INEGI ENIGH 2024 baskets
   for ten deciles representing 38,830,230 households and runs six scenarios.
   [Application report](distributional/report.md) ·
   [Incidence figure](distributional/distributional_incidence.png) ·
   [Manifest](distributional/manifest.json).

The ENIGH scenarios assume import exposure, tariff pass-through, wage responses
and an externally financed transfer pool. They are conditional illustrations,
not estimates of a historical tariff effect. The NK exercise is synthetic
parameter recovery. None of the six new benchmarks is presented as a published
empirical model replication. Held-out NK horizons use the same sample and do
not constitute independent validation data.

## Validation

| Check | Result |
|---|---|
| Structural, LP, distributional, ENIGH, benchmark, Pyodide and bilingual tests | 145 passed |
| Complete NK and ENIGH application tests | 15 passed |
| Executable examples in eight new English/Spanish guides | 8 passed |
| Related household/VFI/data/LP/validation/replication/report/import/release-check regressions | 253 passed, 2 skipped |
| Existing test-quality checks, including coverage-based execution checks | 22 passed |
| Independent benchmark cases | 6 passed, including all negative controls |
| Isolated installed wheel, with network explicitly blocked | Passed all six cases, data loading, LP-to-structural fit and incidence |
| Documentation build with `mkdocs build --strict` | Passed |
| Shared edited-file whitespace check | Passed |

These test runs total **443 passed and 2 skipped**. The two skipped household
parity comparisons require the external IO research volume. An initial
test-quality attempt lacked the optional `coverage` tool; rerunning with an
existing cached copy passed, without installing or changing the environment.

The isolated-wheel run verifies that imports came from the wheel rather than
the checkout. ENIGH data, metadata and authenticated Dynare resources were
included. [Wheel smoke evidence](verification/installed_smoke.json),
[installed benchmark evidence](verification/installed-wheel-benchmarks.json),
[core source match](verification/source_match.json), and
[wheel provenance](verification/wheel.json) are retained. The local wheel was
not published. Test/build output is in [verification](verification/).

The repository-wide public API snapshot test still fails on **nine pre-existing
entries** outside this work: DSGE exports and result fields in Bartik, DiD,
dynamic panels, nowcasting and regime VAR. The new workflow API entries match
their deliberately updated snapshot. Exact inherited differences are recorded
in [api_snapshot_drift.json](api_snapshot_drift.json); unrelated snapshot entries
were preserved. The entire repository test suite was not run.

## Reproduce

From the repository with its numerical dependencies installed:

```bash
python tools/run_research_benchmarks.py --output research_output/benchmarks
python -m puremacro.examples.structural_irf_matching --output research_output/structural
python -m puremacro.examples.distributional_trade_enigh --output research_output/distributional
```

The two applications also run as installed-package modules. Benchmark users
without a checkout can call `puremacro.validation.run_research_benchmarks()`
and save its report with `.write(directory)`.

The [workflow guide](../../docs/research_workflows.md) links complete English
and Spanish API documentation. The ENIGH conversion tool authenticates the
official source workbook before rebuilding the packaged data; only that
conversion step needs `openpyxl`. Runtime applications and benchmarks are
offline.
