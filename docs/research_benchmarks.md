> 🇬🇧 English · 🇪🇸 [Español](es/research_benchmarks.md)

# Independent research benchmarks

Run an offline set of independent comparisons and save their evidence:

```python
from puremacro.validation import run_research_benchmarks

report = run_research_benchmarks()
print(report.passed)
paths = report.write("research-benchmarks")
print(paths["json"])
```

This works from an installed wheel. It needs neither a repository checkout nor
MATLAB, Dynare, a reference econometrics package, or network access. In a checkout,
`python tools/run_research_benchmarks.py --output /tmp/benchmarks` writes the same
dossier and exits nonzero if a case fails. Add `--case growth_analytical` to select
one case; repeat the option to select several.

The current suite reports **4/5 passing cases and an overall FAIL**. The original
RR2010 data reproduce the GDP response and trough horizon, but their t statistic
lies just outside the published rounding interval. This known discrepancy is
retained rather than loosening tolerance; the default command exits nonzero.

## What the evidence establishes

| Case | Evidence type | Independent reference | Scope |
|---|---|---|---|
| `linear_minimum_distance` | Analytical | Closed-form GLS coefficients, covariance and objective | Correlated empirical moment covariance, optimal weighting and parameter uncertainty |
| `growth_analytical` | Analytical | Exact log-utility, full-depreciation growth policy | Steady state and derivatives with respect to lagged capital and technology innovations |
| `dynare_rbc_order2` | External software | Previously exported live Dynare 7.0 decision rules and simulation moments | One RBC model, second-order tensors and a common-innovation simulation |
| `rr2010_baseline_software` | External software | Frozen independent OLS run on the authors' original data | Full coefficient and cumulative-response covariance, estimates, SEs, t statistics and sample sizes |
| `rr2010_published_peak` | Published empirical | Literal Romer–Romer (2010) page 781 values | Figure 4 baseline trough at quarter ten: response -3.08, t statistic -3.53 |

The first two cases are synthetic numerical exercises. The Dynare model also
uses simulated innovations. The last two comparisons use the original
Romer–Romer empirical data. Software parity and published rounded
evidence are separate cases with separate tolerances.

These cases complement the broader [validation gallery](VALIDATION.md) and
[structural validation status](STRUCTURAL_VALIDATION_STATUS.md). A pass concerns
only the model, metrics and tolerance recorded in its dossier. It does not
establish global numerical accuracy, finite-sample coverage, external validity,
or correct policy assumptions.

## Reading the dossier

`benchmark_report.json` contains every observed and reference value, shape, unit,
maximum absolute error, maximum tolerance excess and metric status. It also
records package and dependency versions, platform, UTC run time, source links,
data provenance, transformations, sampling conventions, tolerances and
limitations. `benchmark_report.md` presents a readable comparison; long arrays
are abbreviated only in Markdown.

Comparisons use

```text
abs(observed - reference) <= atol + rtol * abs(reference)
```

Keys must agree exactly. Empty inputs, missing metrics, shape mismatches,
nonfinite values, failed computations and incomplete provenance fail the case.
The comparison never broadcasts arrays or treats a failed case as a successful
skip. Empty suites and duplicate case identifiers raise `ValueError`.

Every successful case includes a negative control: the comparator must reject
a deliberately perturbed reference value submitted as an observed value. This
checks that the comparison can fail; it is not a replacement for the independent
reference itself. The test suite additionally perturbs actual outputs from every
built-in case and checks that each case fails.

### Empirical-to-structural covariance

The linear case supplies four correlated moment estimates and fits two
parameters through [`fit_structural`](structural_bridge.md). Its oracle directly
evaluates GLS normal equations. The input covariance is the covariance of the
**estimated moment vector**, so the oracle does not multiply or divide it by a
sample size. Both the fitted coefficients and their full covariance are tested.

### External-software provenance

The Dynare case authenticates the `.mod`, innovation CSV and reference NPZ against
the shipped manifest and pins the reference export checksum and run identifier.
The original execution used Dynare 7.0 with MATLAB R2026a on 2026-09-20. It used
2,500 common innovations and a 500-period burn-in. Sample covariance uses
`ddof=1`; lag-one covariance separately centers its endpoints and divides by
`N-2`. Variable, state and shock labels are aligned before comparison. The
offline run does not execute Dynare or regenerate the reference. See the
package's `puremacro.dsge/_references/DYNARE_FIXTURES.md` for regeneration details.

### Original-data published replication

The [Romer–Romer baseline](empirical_research.md) retains the authors' original
data vintage and `EXOGERNR.RAT` specification: quarterly GDP growth on an
intercept and contemporaneous through twelve lags of exogenous tax changes,
1950Q1–2007Q4. It uses conventional residual-df OLS covariance and cumulates
both coefficients and their full covariance. An independent software port of
that specification supplies the frozen reference; the offline benchmark does
not execute RATS. All coefficients, covariance entries, cumulative responses,
SEs, t statistics and sample sizes use `rtol=1e-10, atol=1e-11`.

The separate published case (`rr2010_published_peak`, Gate-5) checks horizon ten,
-3.08 log percent and the t statistic -3.53 from printed page 781 of Romer & Romer (2010 AER).
Its `rtol=0, atol=0.005` accommodates two-decimal publication rounding; the integer trough
horizon must agree. This case reproduces one published baseline, without establishing narrative
exogeneity or reproducing every robustness specification.

#### Gate-5 RR2010 Error Bound and Publication Rounding Discrepancy
The frozen original-data result computed directly from the authors' replication vintage is
`-3.0808127631` with `t = -3.5249960750`. Comparing the computed t-statistic against the
published two-decimal print `-3.53` yields an absolute difference:
```text
|-3.5249960750 - (-3.53)| = 0.0050039250
```
Under the standard two-decimal rounding tolerance `atol = 0.005`, the discrepancy excess is:
```text
0.0050039250 - 0.0050000000 = 3.925e-6 (3.925 × 10⁻⁶)
```
The original RATS databanks corroborate the workbook result. Software parity passes cleanly;
this published comparison is preserved as a documented and verified publication-rounding
discrepancy rather than artificially loosening the tolerance. Unit tests in
`tests/test_research_benchmarks.py` verify both the exact numerical values and the
strict bound `0 < max_tolerance_excess < 4e-6`.

## Adding a research benchmark

`ResearchBenchmark` takes separate `compute` and `reference` callables, explicit
units per metric, source citations, data provenance, sampling and transformation
descriptions, limitations and tolerances. For example:

```python
from scipy.optimize import brentq
from puremacro.validation import ResearchBenchmark, run_research_benchmarks

# Synthetic Solow steady state: saving equals depreciation.
alpha, saving, depreciation = 0.36, 0.20, 0.10

def my_model_output():
    capital = brentq(lambda k: saving * k**alpha - depreciation * k, 0.01, 100)
    return {"capital": capital}

def my_independent_reference():
    return {"capital": (saving / depreciation)**(1 / (1 - alpha))}

case = ResearchBenchmark(
    id="my_model", title="Solow steady state versus its analytical solution",
    evidence_kind="analytical",
    compute=my_model_output, reference=my_independent_reference,
    sources=("Analytical resource constraint: saving*k**alpha = depreciation*k",),
    data={"origin": "synthetic calibration", "parameters": {
        "alpha": alpha, "saving": saving, "depreciation": depreciation}},
    sampling="No sample; deterministic calibration",
    transformations="State and control variables in levels",
    units={"capital": "goods per capita"},
    limitations="Valid only for the documented calibration and steady state",
    rtol=1e-7, atol=1e-9,
)
report = run_research_benchmarks([case])
```

Both functions must return a nonempty mapping with the same metric keys, such
as `{"capital": 0.190117}`. A reference generated by the same function under test
is a regression snapshot rather than independent evidence. Record the exact
published table, specification, sample and data transformations before claiming
an empirical replication.

## Sources

- [Hansen, *Econometrics*](https://users.ssc.wisc.edu/~bhansen/econometrics/): minimum-distance and GLS framework. The particular linear benchmark is constructed here.
- [QuantEcon, Cass-Koopmans Model](https://python.quantecon.org/cass_koopmans_1.html): growth-model setting. Our benchmark specializes to log utility and full depreciation and states its exact policy in the dossier.
- [Dynare stochastic solution and simulation](https://www.dynare.org/manual/the-model-file.html#stochastic-solution-and-simulation): perturbation and simulation conventions; the frozen evidence records the actual version used.
