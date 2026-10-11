# Reproducible panels to forecast evidence

Research and provenance milestone included in puremacro 4.8.0. The checks below
record its initial development validation; release-specific checks are recorded
separately in `../2026-10-10-release-4.8.0/`. No journal submission was performed.

## Deliverables

- Portable npz schema v2 preserves supported `DataFrame.attrs` without pickle.
  Sources, units, failures and splice decisions survive `.pmz` round trips.
  V1 files remain readable and verifiable; v2 requires an updated reader.
- Three-country offline GDP-growth forecasting study: 195 observed levels,
  225 forecasts, seven tables, figure, cartridge and checksummed manifest.
- English and Spanish workflow guides and reconciled support/status descriptions.
- Existing paper kept within its word limit and explicitly tied to its measured
  4.6.0 evidence snapshot. Counts and external adoption were not invented.

Start with [the study report](study/report.md) and [the fixed design](DESIGN.md).
The [English guide](../../docs/cross_country_forecasting.md) and
[Spanish guide](../../docs/es/cross_country_forecasting.md) describe interpretation
and the portable data format.

## Observed result

Equal-country RMSE (annual log-growth percentage points) is 3.039823 for AR(1),
3.191707 for historical mean, and 3.327973 for zero growth. The lowest RMSE
within each country is historical mean for USA, zero growth for Mexico, and
AR(1) for Brazil. These are descriptive losses for the fixed 2000–2024 sample;
they are not significance or general-superiority claims. Data are revised
latest-vintage observations, not the information available at each origin.

## Verification recorded

| Check | Evidence |
|---|---|
| Independent data conversion | All 195 levels equal direct extraction of authenticated WDI JSON after scaling |
| Independent numerical implementation | All 225 forecasts match separate SciPy SVD least squares; maximum difference 1.85e-13 pp; all seven tables pass; [validation.json](validation.json) |
| Installed development wheel | Separate working directory, HTTP connections blocked, cartridge verifies, seven CSVs byte-identical to source execution; [wheel-validation.json](wheel-validation.json) |
| Actual WebAssembly runtime | Pyodide 0.28.3 / Python 3.13.2: provenance tests, full study, PNG and cartridge succeed; runtime versions, test count and numerical difference are recorded in [pyodide-result.json](pyodide-result.json) |
| Forecast/data tests | 20 passed: independent OLS, future-data perturbation, prefix invariance, invalid samples, deterministic tables, source-revision rejection and exported evidence; [research-tests.log](research-tests.log) |
| Storage/provider/Pyodide-contract tests | 346 passed, 3 network tests deselected; includes actual offline WDI/OECD fixture metadata and legacy v1 data |
| Additional integration/data/paper tests | 172 passed, 39 deselected; [integration-tests.log](integration-tests.log). This batch overlaps the storage/provider batch; counts must not be summed as unique tests. |
| Release checks excluding baseline suite | Static Pyodide import contract, public API snapshot, version agreement and Python 3.11 syntax all pass; [release-gates.log](release-gates.log) |
| Documentation | Strict MkDocs build passes; [docs-build.log](docs-build.log) |

The initial development checks did not rerun the full repository test baseline,
live-provider suite, all notebooks or all platform/interpreter combinations;
see the separate release evidence for subsequent validation. The numerical oracle is a
separate implementation inside development; outside-user reproduction remains
an adoption milestone. The Pyodide check runs under Node-hosted WebAssembly,
not a deployed browser UI. It uses the distribution's cached dependencies
without PyPI fallback; the named-zone metadata test points Python's zoneinfo
at the timezone database bundled with pytz.

## Reproduce from this checkout

```bash
python -m pip install -e .
python -m puremacro.examples.cross_country_forecasting --output /tmp/puremacro-forecast-study
PYTHONPATH=. python tools/build_cross_country_forecasting_data.py --output-dir /tmp/puremacro-gdp-rebuild
python reviews/2026-10-10-cross-country-forecasting/independent_validation.py
```

The independent validator reads the stored development evidence under
`study/`; it checks every artifact and source hash, so regenerate that directory
after any numerical source change:

```bash
python -m puremacro.examples.cross_country_forecasting --output reviews/2026-10-10-cross-country-forecasting/study
python reviews/2026-10-10-cross-country-forecasting/independent_validation.py
```

To test installation, build a wheel from this source with `python -m build
--wheel`, install it in a separate environment, change to a directory outside
the repository and run the same application. `validate_wheel.py` automates the
offline comparison for a wheel installed with `pip install --target DIRECTORY
--no-deps PATH.whl` into an environment with the base dependencies. It accepts
`--installed-dir DIRECTORY --wheel PATH.whl`.

For an outside-user pilot, record the Python/platform versions, installation
obstacles, time to the first report, whether all seven tables match, and whether
the user can explain the latest-vintage limitation. No outside users have been
contacted or counted as adopters by this work.
