> 🇬🇧 English · 🇪🇸 [Español](es/workbench_and_scorecards.md)

# Empirical Workbench, Scorecards, and Solver Diagnostics

This guide documents six composable PureMacro APIs for data validation, specification sensitivity, impulse-response comparison, DSGE validation scorecards, solver diagnostic reporting, and composite trade tariff analysis. Every example below runs deterministically in pure NumPy/SciPy/pandas with zero network access and full Pyodide-core compatibility.

| Stage | API | Returned contract | Primary purpose |
| --- | --- | --- | --- |
| 1. Panel preflight | `puremacro.data.preflight_panel` | `PreflightReport` | Audit panel structure, missingness, duplicates, gaps, stationarity, and seasonality before estimation |
| 2. LP sensitivity | `puremacro.lp.sensitivity.lp_sensitivity_grid` | `IRFComparisonResult` | Sweep hyperparameter grids across local-projection estimators and align resulting IRFs |
| 3. IRF comparison | `puremacro.reports.compare_irfs` | `IRFComparisonResult` | Compare point paths, bands, signs, and baseline deviations across heterogeneous IRF results |
| 4. DSGE scorecards | `puremacro.dsge.prior_posterior_comparison`, `puremacro.dsge.score_irf_against_targets` | `PriorPosteriorComparisonResult`, `IRFScorecardResult` | Quantify prior-to-posterior learning, MCMC health, and model IRF fit vs empirical targets |
| 5. Solver diagnostics | `puremacro.solvers.diagnostics` | `SolverDiagnosticRecord` | Separate numerical convergence, equation accuracy, and regime/support validity |
| 6. Trade scenarios | `CompositeTariffScenario`, `combine_scenarios`, `summarize_scenario_batch` | `CompositeTariffScenario`, `ScenarioComparisonResult` | Compose bilateral tariff policies with explicit conflict rules and summarize multi-country welfare |
| 7. Robust workbench & CLI | `run_empirical_structural_workbench`, `EmpiricalStructuralDossier`, `puremacro.cli` | `EmpiricalStructuralDossier`, HTML/LaTeX/Markdown/JSON bundle | Unify preflight, LP, scorecards, robust moment sets, SMM, bilingual bundle export, and CLI workflows |

---

## 1. Panel Preflight Checks (`preflight_panel`)

`preflight_panel` inspects a panel `DataFrame` prior to running local projections or panel regressions and summarizes structural issues as typed `PreflightFinding` records inside a `PreflightReport`.

```text
preflight_panel(
    df: pd.DataFrame,
    *,
    unit_col: str | None = None,
    unit: str | None = None,
    time_col: str | None = None,
    time: str | None = None,
    variables: Sequence[str] | None = None,
    freq: str | None = None,
    min_obs: int = 12,
    max_missing_share: float = 0.25,
    alpha: float = 0.05,
    run_stationarity: bool = True,
    run_seasonality: bool = True,
    digits: int = 3,
) -> PreflightReport
```

`PreflightFinding` stores `(check, severity, variable, unit, message, recommendation, metric)` where `severity` is `"info"`, `"warning"`, or `"error"`. `PreflightReport` exposes:

- **Status and metadata:** `.passed` (`n_errors == 0`), `.status` (`"passed" | "warn" | "failed"`), `.panel_shape`, `.n_units`, `.n_periods`, `.unit_col`, `.time_col`, `.inferred_freq`, `.date_range`, `.balanced`, `.duplicate_count`, `.n_errors`, `.n_warnings`, `.findings`, `.recommendations`, and `.data_sha256`.
- **Tabular and text exports:** `.to_frame(layout="variables" | "findings" | "summary")` (alias `.to_dataframe(...)`), `.summary()`, `.to_markdown(...)`, `.to_latex(...)`, `.to_typst(...)`, `.to_json(indent=2)`, and `.plot(metric="missing_pct" | "n_valid" | "adf_median_p" | "kpss_median_p", ...)`.

```python
import numpy as np
import pandas as pd
from puremacro.data import preflight_panel

rng = np.random.default_rng(7)
dates = pd.date_range("2018-01-01", periods=24, freq="QS")
rows = [
    {"country": iso, "date": d, "gdp": float(y), "cpi": float(pi)}
    for iso in ("USA", "MEX", "CAN")
    for d, y, pi in zip(
        dates,
        np.cumsum(rng.normal(0.0, 0.5, len(dates))),
        2.0 + rng.normal(0.0, 0.3, len(dates)),
    )
]
panel = pd.DataFrame(rows)
panel.loc[(panel["country"] == "CAN") & (panel["date"] >= "2023-07-01"), "cpi"] = np.nan

report = preflight_panel(
    panel,
    unit_col="country",
    time_col="date",
    variables=["gdp", "cpi"],
    min_obs=12,
    max_missing_share=0.25,
)
print("status:", report.status, "| passed:", report.passed, "| freq:", report.inferred_freq)
print(report.to_frame("variables")[["variable", "n_obs", "n_valid", "missing_pct", "adf_median_p"]])
```

### Scope and limits

- Supports `MultiIndex` panels (`unit`, `time`), `DatetimeIndex`, `PeriodIndex`, or explicit/auto-detected `unit_col` and `time_col` columns (`unit` and `time` aliases are also accepted).
- Stationarity (`run_stationarity=True`, ADF and KPSS) and seasonality (`run_seasonality=True`) are advisory unit-by-unit checks evaluated at significance level `alpha`; short panels below `min_obs` or units exceeding `max_missing_share` generate explicit findings rather than silently dropping data or transforming variables.
- Passing preflight means the dataset satisfies the configured structural thresholds (`n_errors == 0`); it does not guarantee causal identification or correct lag specification.

---

## 2. Local-Projection Sensitivity Grids (`lp_sensitivity_grid`)

`lp_sensitivity_grid` runs the Cartesian product of a hyperparameter grid over local-projection estimators and compares every resulting impulse response in a unified `IRFComparisonResult`.

```text
lp_sensitivity_grid(
    data: pd.DataFrame,
    y: str,
    x: str,
    *,
    grid: Mapping[str, Sequence[Any]],
    estimator: Callable[..., Any] = lp_hac,
    baseline_spec: Mapping[str, Any] | None = None,
    base_kwargs: Mapping[str, Any] | None = None,
    on_error: Literal["raise", "skip"] = "raise",
    horizon_align: Literal["exact", "intersection"] = "exact",
    band_compat: Literal["warn", "raise", "ignore"] = "warn",
) -> IRFComparisonResult
```

Key behaviors:

- **Cross-estimator parameter normalization:** Automatically harmonizes `lags <-> n_lags`, `horizon <-> horizons`, `ci <-> alpha`, and maps `x` to `x`, `treat`, or `x_endog` depending on the estimator signature. Including `"estimator"` as a key inside `grid` sweeps across different estimators (for example `lp_hac` and `lp_iv`) alongside lag or horizon choices.
- **Provenance and error handling:** Each grid point's resolved keyword dictionary is stored in `metrics["resolved_params"]`. With `on_error="skip"`, failing specifications are recorded in `result.errors` instead of aborting the sweep.

```python
import numpy as np
import pandas as pd
from puremacro.lp.sensitivity import lp_sensitivity_grid

rng = np.random.default_rng(7)
T = 80
shock = rng.normal(size=T)
y = np.zeros(T)
for t in range(1, T):
    y[t] = 0.6 * y[t - 1] - 0.5 * shock[t] + 0.2 * rng.normal()
lp_df = pd.DataFrame({"y": y, "shock": shock})

grid_res = lp_sensitivity_grid(
    lp_df,
    "y",
    "shock",
    grid={"lags": [1, 2], "horizon": [4, 6]},
    baseline_spec={"lags": 2, "horizon": 4},
    horizon_align="intersection",
)
print(grid_res.to_frame("metrics")[["spec", "baseline", "peak_response", "rmse_vs_base", "resolved_params"]])
```

### Scope and limits

- At least two successful grid specifications are required; otherwise a `ValueError` is raised.
- By default (`horizon_align="exact"`), differing horizon grids raise an error; pass `horizon_align="intersection"` whenever `horizon` or `horizons` varies across the grid.
- Grid metrics compare specification stability across discrete choices; confidence intervals are pointwise or simultaneous from each individual run and are not adjusted for multiple specification testing across the grid.

---

## 3. Cross-Model IRF Comparison (`compare_irfs`)

`compare_irfs` aligns and summarizes two or more already-estimated impulse responses from local projections, structural VARs, Bayesian IRFs, or plain `DataFrame` objects.

```text
compare_irfs(
    results: Mapping[str, Any],
    *,
    baseline: str | None = None,
    horizon_align: Literal["exact", "intersection"] = "exact",
    target: str | int | Mapping[str, str | int] | None = None,
    shock: str | int | Mapping[str, str | int] | None = None,
    regime: str | Mapping[str, str] | None = None,
    quantile: float | Mapping[str, float] | None = None,
    ci: float | None = None,
    expand_groups: bool = False,
    cumulative: bool = False,
    allow_mixed_cumulative: bool = False,
    band_compat: Literal["warn", "raise", "ignore"] = "warn",
    digits: int = 3,
) -> IRFComparisonResult
```

`IRFComparisonResult` exposes `.table`, `.metrics`, `.baseline`, `.horizons`, `.warnings`, `.errors`, along with `.to_frame(layout="wide" | "coef_se" | "metrics")` (alias `.to_dataframe(...)`), `.to_manuscript_dataframe(...)`, `.summary()`, `.to_markdown(...)`, `.to_latex(...)`, `.to_typst(...)`, and `.plot(show_bands=True, ...)`.

```python
import numpy as np
import pandas as pd
from puremacro.lp import lp_hac
from puremacro.reports import compare_irfs

rng = np.random.default_rng(7)
T = 80
shock = rng.normal(size=T)
y = np.zeros(T)
for t in range(1, T):
    y[t] = 0.6 * y[t - 1] - 0.5 * shock[t] + 0.2 * rng.normal()
df = pd.DataFrame({"y": y, "shock": shock})

lp_2 = lp_hac(df, y="y", x="shock", lags=2, horizon=4, ci=0.90)
lp_4 = lp_hac(df, y="y", x="shock", lags=4, horizon=4, ci=0.90)

cmp_res = compare_irfs(
    {"LP (2 lags)": lp_2, "LP (4 lags)": lp_4},
    baseline="LP (2 lags)",
)
print(cmp_res.summary())
print(cmp_res.to_frame("wide").head())
```

### Scope and limits

- Selector arguments (`target`, `shock`, `regime`, `quantile`) accept either a single scalar or a per-model mapping to slice multi-dimensional VAR/regime/quantile results down to comparable 1-D series.
- Mixing cumulative and non-cumulative IRFs raises by default (`allow_mixed_cumulative=False`), and mismatched band widths or band kinds trigger warnings (`band_compat="warn"`) or errors (`band_compat="raise"`).
- `compare_irfs` reports descriptive distances (`max_abs_diff_vs_base`, `rmse_vs_base`, `sign_agreement`), not formal Hausman or encompassing hypothesis tests across estimators.

---

## 4. DSGE Validation Scorecards (`prior_posterior_comparison`, `score_irf_against_targets`)

The DSGE scorecard module provides two diagnostics: prior-to-posterior updating (`prior_posterior_comparison`) and impulse-response fit against empirical moments (`score_irf_against_targets`).

```text
prior_posterior_comparison(
    posterior: Any,
    priors: Mapping[str, Any] | Any | None = None,
    *,
    ci_level: float = 0.90,
    weak_id_threshold: float = 0.80,
    rhat_threshold: float = 1.05,
    ess_threshold: float = 100.0,
    divergence_threshold: float = 0.01,
    boundary_tol: float = 0.02,
    params: Sequence[str] | None = None,
    digits: int = 3,
) -> PriorPosteriorComparisonResult

score_irf_against_targets(
    model_irf: Any,
    targets: Any,
    *,
    shock: str | None = None,
    variables: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    weight_mode: Literal["auto", "covariance", "diagonal"] = "auto",
    ci_level: float = 0.90,
    cond_warn: float = 1e6,
    hank_unit_mode: Literal["warn", "raw", "scale_to_target"] = "warn",
    digits: int = 3,
) -> IRFScorecardResult
```

- **`PriorPosteriorComparisonResult`** checks marginal std contraction (`post_std / prior_std >= weak_id_threshold`), bound proximity (`boundary_tol`), split-`rhat`, bulk/tail ESS, and divergence fraction. It provides `.to_frame(layout="full" | "compact")`, `.summary()`, `.to_markdown(...)`, `.to_latex(...)`, `.to_typst(...)`, and `.plot(...)`.
- **`IRFScorecardResult`** computes `rmse`, `mae`, `wse`, quadratic form `chi2_stat` / `chi2_df` / `chi2_pvalue`, `sign_concordance`, `band_coverage`, `determinacy_rate`, `weight_mode_used`, and `covariance_condition_number`. When `targets` is a `MomentTargets` object with a finite covariance matrix, `weight_mode="auto"` computes the generalized Wald statistic; when given marginal `LPResult` mappings without cross-horizon covariance, it falls back to `"diagonal"` with `chi2_pvalue=NaN` and records an explicit warning.

```python
from types import SimpleNamespace
import numpy as np
import pandas as pd
from puremacro.dsge import BetaPrior, GammaPrior, prior_posterior_comparison, score_irf_against_targets
from puremacro.lp import lp_hac
from puremacro.structural import lp_moment_targets

rng = np.random.default_rng(7)
draws = np.stack(
    [
        np.column_stack([rng.normal(1.35, 0.08, 120), rng.beta(12, 88, 120)]),
        np.column_stack([rng.normal(1.36, 0.08, 120), rng.beta(12, 88, 120)]),
    ],
    axis=0,
)
posterior = SimpleNamespace(
    draws=draws,
    param_names=["sigma", "kappa"],
    priors={"sigma": GammaPrior(1.0, 0.25), "kappa": BetaPrior(0.10, 0.04)},
)
pp = prior_posterior_comparison(posterior)
print(pp.to_frame("compact")[["param", "prior_std", "post_std", "std_ratio", "rhat", "flags"]])

T = 80
shock = rng.normal(size=T)
y = np.zeros(T)
for t in range(1, T):
    y[t] = 0.6 * y[t - 1] - 0.5 * shock[t] + 0.2 * rng.normal()
df = pd.DataFrame({"y": y, "shock": shock})

targets = lp_moment_targets(
    df,
    responses=["y"],
    shock="shock",
    horizons=range(5),
    response_units={"y": "pct"},
    shock_unit="sd",
    frequency="Q",
    lags=2,
)
emp = lp_hac(df, y="y", x="shock", lags=2, horizon=4, ci=0.90)
model_irf = pd.DataFrame({"y": emp["beta"].to_numpy() * 0.85}, index=emp["h"].to_numpy())
scorecard = score_irf_against_targets(model_irf, targets)
print(scorecard.summary())
```

### Scope and limits

- Marginal standard-deviation contraction (`std_ratio >= weak_id_threshold`) detects univariate lack of updating relative to the prior; it is not a full Fisher-information or global multi-parameter identification test.
- `chi2_stat` treats the target covariance matrix as fixed weights and ignores posterior parameter estimation uncertainty unless posterior IRF bands are evaluated alongside a valid target covariance matrix.
- For HANK transitions whose shock normalization differs from aggregate empirical impulse conventions (such as -25 bp defaults), `hank_unit_mode="warn"` raises a unit-mismatch warning without altering paths; set `hank_unit_mode="scale_to_target"` only when linear scaling is theoretically valid.

---

## 5. Solver Convergence, Accuracy, and Regime Diagnostics (`puremacro.solvers.diagnostics`)

PureMacro solver wrappers separate three orthogonal health axes into structured dataclasses so downstream pipelines never treat "optimizer stopped" as synonymous with "economically accurate solution".

```text
ConvergenceStatus(converged, iterations=0, stopping_reason="", solver_method="", tolerance=0.0)
AccuracyStatus(is_accurate, max_residual=0.0, residual_norm=0.0, residual_type="", tolerance=0.0, residuals={})
SupportStatus(is_supported, regime="interior", warnings=(), boundary_metrics={})
VALID_OVERALL_STATUSES = ("PASS", "CONVERGED_INACCURATE", "UNSUPPORTED_REGIME", "FAILED")
determine_overall_status(convergence, accuracy, support) -> str
make_solver_diagnostic_record(...) -> SolverDiagnosticRecord
```

Precedence enforced by `determine_overall_status`:
1. `not convergence.converged` -> `"FAILED"`
2. `not accuracy.is_accurate` -> `"CONVERGED_INACCURATE"`
3. `not support.is_supported` -> `"UNSUPPORTED_REGIME"`
4. Otherwise -> `"PASS"`

```python
from puremacro.solvers.diagnostics import (
    AccuracyStatus,
    ConvergenceStatus,
    SupportStatus,
    determine_overall_status,
    make_solver_diagnostic_record,
)

pass_rec = make_solver_diagnostic_record(
    converged=True,
    iterations=9,
    stopping_reason="gtol_met",
    solver_method="equilibrated_newton",
    tolerance=1e-8,
    is_accurate=True,
    max_residual=4.2e-9,
    residual_norm=6.1e-9,
    residual_type="excess_demand",
    accuracy_tolerance=1e-6,
)
stall_status = determine_overall_status(
    ConvergenceStatus(converged=True, iterations=25, stopping_reason="xtol_stall", solver_method="newton", tolerance=1e-8),
    AccuracyStatus(is_accurate=False, max_residual=3.1e-3, residual_type="euler_eq", tolerance=1e-6),
    SupportStatus(is_supported=True),
)
print("pass record:", pass_rec.overall_status, pass_rec.is_pass)
print("stalled solve status:", stall_status)
```

### Scope and limits

- These helpers implement a typed classification and audit contract; they do not re-solve equations or compute residuals from model source objects.
- Because `make_solver_diagnostic_record` defaults `is_accurate=True` and `is_supported=True` when omitted, callers must pass the actual equation residual checks and domain-support checks from their solver backend rather than accepting defaults blindly.

---

## 6. Composite Trade Tariffs and Batch Summaries (`CompositeTariffScenario`, `summarize_scenario_batch`)

`CompositeTariffScenario` and `combine_scenarios` merge multiple bilateral/sectoral `TariffScenario` packages under explicit cell-level conflict policies, then `summarize_scenario_batch` summarizes multi-country welfare, GDP, CPI, and minimax/expected-payoff comparisons across scenarios.

```text
CompositeTariffScenario(
    scenarios: tuple[TariffScenario | CompositeTariffScenario, ...],
    name: str = "composite",
    on_conflict: Literal["last_wins", "first_wins", "error", "max", "sum"] = "last_wins",
    description: str = "",
    metadata: dict[str, Any] = ...,
)
combine_scenarios(scenarios, *, name="composite", on_conflict="last_wins", description="", metadata=None) -> CompositeTariffScenario

summarize_scenario_batch(
    batch: ScenarioBatchResult,
    calib: TradeCalibrationResult,
    *,
    countries: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    include_baseline: bool = True,
    consumption_categories: Sequence[int] = (0,),
    welfare_tol: float | Literal["auto"] = "auto",
    digits: int = 3,
) -> ScenarioComparisonResult
```

Key methods:
- On `CompositeTariffScenario`: `.importers`, `.to_rate_tensor(country_codes, sector_codes)`, `.to_frame(country_codes, sector_codes, *, nonzero_only=True)`, and `.conflicts(country_codes, sector_codes)`. Registered with `TariffScenario`, so `run_scenario` and `run_scenario_batch` solve it directly.
- On `ScenarioComparisonResult`: `.to_frame(layout="full" | "compact" | "summary")`, `.to_payoff_matrix(metric="ev_pct_consumption", *, include_baseline=True)`, `.compare_robust(...)`, `.summary()`, `.to_markdown(...)`, `.to_latex(...)`, `.to_typst(...)`, and `.plot(...)`.

```python
from puremacro.trade import (
    TariffScenario,
    calibrate_trade_model,
    combine_scenarios,
    run_scenario_batch,
    summarize_scenario_batch,
)
from puremacro.trade.data import synthetic_icio_table

icio = synthetic_icio_table(3, 2, nfd=3, seed=42)
calib = calibrate_trade_model(icio)
c0, c1, c2 = calib.country_codes[:3]

broad = TariffScenario(importer=c0, tariffs={(c1, "*"): 0.10}, name="broad_10pct")
targeted = TariffScenario(importer=c0, tariffs={(c1, "S01"): 0.25, (c2, "S02"): 0.15}, name="targeted")
combined = combine_scenarios([broad, targeted], name="combined_max", on_conflict="max")

print("Overlapping cells resolved by `max`:")
print(combined.conflicts(calib.country_codes, calib.sector_codes)[["exporter", "importer", "sector", "resolved_rate", "rule"]])

batch = run_scenario_batch(calib, [broad, targeted, combined])
summary = summarize_scenario_batch(batch, calib)
print(summary.to_frame("compact").head(6))
print(summary.to_payoff_matrix(metric="ev_pct_consumption"))
```

### Scope and limits

- Conflict rules (`"last_wins"`, `"first_wins"`, `"error"`, `"max"`, `"sum"`) resolve overlapping `(importer, exporter, sector)` tariff rates linearly before solving equilibrium; `"sum"` does not cap rates automatically beyond static ad valorem admissibility.
- `summarize_scenario_batch` inherits the static Dekle-Eaton-Kortum structure of `calib` (constant sectoral expenditure shares and trade deficits held fixed under the chosen numeraire); small numerical welfare swings within `welfare_tol` reflect solver tolerances rather than economically distinct rankings.

---

## 7. Identification-Robust Workbench, SMM Integration & Bilingual Bundle Export

`run_empirical_structural_workbench` and `EmpiricalStructuralDossier` unify panel preflight, LP sensitivity, IRF comparisons, validation scorecards, solver diagnostics, identification-robust moment testing / 1D quadratic confidence set inversion (`puremacro.structural.robust`), optional simulated method of moments estimation (`puremacro.structural.simulation`), and self-contained bilingual bundle exports.

```text
run_empirical_structural_workbench(
    data: pd.DataFrame,
    y: str,
    x: str,
    *,
    ...,
    moment_test_targets: MomentTargets | None = None,
    moment_test: MomentTestResult | None = None,
    model_moments: Any | None = None,
    simulation: SimulatedMoments | None = None,
    moment_test_covariance: Any | None = None,
    moment_test_alpha: float = 0.05,
    moment_test_calibration: str = "asymptotic",
    moment_confidence_set: MomentConfidenceSet | None = None,
    moment_curve: QuadraticMomentCurve | None = None,
    moment_domain: tuple[float, float] = (-inf, inf),
    moment_grid: MomentTestGridResult | None = None,
    moment_grid_fn: Callable[[float], MomentTestResult] | None = None,
    moment_grid_values: Sequence[float] | None = None,
    smm_fit: SMMFitResult | None = None,
    smm_simulator: Callable[..., Any] | None = None,
    smm_theta0: Any | None = None,
    smm_plan: SimulationPlan | None = None,
    smm_definition_id: str | None = None,
    smm_kwargs: Mapping[str, Any] | None = None,
    lang: Literal["en", "es"] = "en",
) -> EmpiricalStructuralDossier
```

New `EmpiricalStructuralDossier` attributes and methods:
- **Attached structural contracts**: `.moment_test` (`MomentTestResult`), `.moment_confidence_set` (`MomentConfidenceSet`), `.moment_grid` (`MomentTestGridResult`), and `.smm_fit` (`SMMFitResult`).
- **Additional `.to_frame(section=...)` sections**: `"moment_test"`, `"confidence_set"`, `"moment_grid"`, `"moment_grid_cells"`, `"smm_fit"`, and `"smm_moments"`.
- **Self-contained HTML & portable bundles**:
  - `dossier.to_html(lang="en" | "es", digits=3) -> str` renders a standalone UTF-8 HTML5 document with escaped dynamic fields and zero host filesystem paths.
  - `dossier.to_bundle(output_dir, *, languages=("en", "es"), digits=3, overwrite=True) -> dict[str, Any]` writes `dossier.json`, `dossier_{en,es}.{html,md,tex}`, and `manifest.json` containing SHA-256 checksums using relative filenames only (never leaking host directory paths).

### Unified CLI Workflows (`puremacro.cli`)

The CLI keeps stdlib-only top-level imports, exits with status `2` on argument or JSON schema errors, and uses encoding-safe stdout printing across Windows/POSIX terminals:

```bash
# Export an end-to-end bilingual dossier + SHA-256 bundle from LP-grid or workbench
python -m puremacro.cli workbench data.csv --y y --x shock --bundle-dir ./out --robust-spec spec.json --format html
python -m puremacro.cli lp-grid data.csv --y y --x shock --lags 1..3 --workbench --bundle-dir ./out --lang es

# Compare financed sequence-space HANK fiscal closures (balanced_budget vs debt_rule)
python -m puremacro.cli hank-fiscal --mode compare --horizon 40 --phi-b 0.08 --format markdown
python -m puremacro.cli hank-fiscal --mode single --closure debt_rule --tax-type lump_sum --horizon 40 --format json

# Evaluate identification-robust point nulls, quadratic curve sets, and moment grids from JSON
python -m puremacro.cli robust-set spec.json --alpha 0.05 --calibration asymptotic --format markdown
```

### Scope and limits

- Top-level imports in `puremacro.workbench` remain restricted to Python stdlib, NumPy, and pandas (`puremacro.structural.robust` and `puremacro.structural.simulation` load lazily only when robust or SMM parameters are requested).
- Exact 1D confidence sets via `invert_quadratic_moments` preserve disconnected intervals, boundary truncation flags (`truncated`, `boundary_contact`), and `unresolved_*` statuses (never substituting pseudoinverses or ridge regularizers for singular covariances).
- Sampled grids (`moment_grid` / `evaluate_moment_grid`) retain `continuous_complete=False`, `accepted_runs()`, and `unresolved_cells()` explicitly so untested grid interiors are never mislabeled as continuous confidence intervals.
