> 🇬🇧 [English](../workbench_and_scorecards.md) · 🇪🇸 Español

# Banco de Trabajo Empírico, Scorecards y Diagnósticos del Solver

Esta guía documenta seis APIs composables de PureMacro para validación de datos, sensibilidad de especificaciones, comparación de funciones de impulso-respuesta (IRF), scorecards de validación DSGE, reportes diagnósticos del solver y análisis compuesto de aranceles. Todos los ejemplos se ejecutan de forma determinista con NumPy/SciPy/pandas puros, sin acceso a red y con compatibilidad total con el núcleo Pyodide.

| Etapa | API | Contrato devuelto | Propósito principal |
| --- | --- | --- | --- |
| 1. Preflight de panel | `puremacro.data.preflight_panel` | `PreflightReport` | Auditar estructura del panel, valores faltantes, duplicados, brechas, estacionariedad y estacionalidad antes de estimar |
| 2. Sensibilidad LP | `puremacro.lp.sensitivity.lp_sensitivity_grid` | `IRFComparisonResult` | Evaluar mallas de hiperparámetros entre estimadores de proyecciones locales y alinear sus IRF |
| 3. Comparación de IRF | `puremacro.reports.compare_irfs` | `IRFComparisonResult` | Comparar trayectorias puntuales, bandas, signos y desviaciones respecto a una especificación base |
| 4. Scorecards DSGE | `puremacro.dsge.prior_posterior_comparison`, `puremacro.dsge.score_irf_against_targets` | `PriorPosteriorComparisonResult`, `IRFScorecardResult` | Cuantificar el aprendizaje previa-posterior, la salud MCMC y el ajuste de IRF del modelo frente a objetivos empíricos |
| 5. Diagnósticos del solver | `puremacro.solvers.diagnostics` | `SolverDiagnosticRecord` | Separar convergencia numérica, exactitud de las ecuaciones y validez de régimen o soporte |
| 6. Escenarios comerciales | `CompositeTariffScenario`, `combine_scenarios`, `summarize_scenario_batch` | `CompositeTariffScenario`, `ScenarioComparisonResult` | Combinar paquetes arancelarios bilaterales con reglas explícitas de conflicto y resumir bienestar multipaís |
| 7. Banco robusto y CLI | `run_empirical_structural_workbench`, `EmpiricalStructuralDossier`, `puremacro.cli` | `EmpiricalStructuralDossier`, paquete HTML/LaTeX/Markdown/JSON | Unificar preflight, LP, scorecards, conjuntos robustos de momentos, SMM, exportación bilingüe y flujos CLI |

---

## 1. Comprobaciones previas de panel (`preflight_panel`)

`preflight_panel` inspecciona un `DataFrame` de panel antes de ejecutar proyecciones locales o regresiones de panel y resume los hallazgos estructurales como registros tipados `PreflightFinding` dentro de un `PreflightReport`.

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

`PreflightFinding` almacena `(check, severity, variable, unit, message, recommendation, metric)`, donde `severity` es `"info"`, `"warning"` o `"error"`. `PreflightReport` expone:

- **Estado y metadatos:** `.passed` (`n_errors == 0`), `.status` (`"passed" | "warn" | "failed"`), `.panel_shape`, `.n_units`, `.n_periods`, `.unit_col`, `.time_col`, `.inferred_freq`, `.date_range`, `.balanced`, `.duplicate_count`, `.n_errors`, `.n_warnings`, `.findings`, `.recommendations` y `.data_sha256`.
- **Exportación tabular y de texto:** `.to_frame(layout="variables" | "findings" | "summary")` (alias `.to_dataframe(...)`), `.summary()`, `.to_markdown(...)`, `.to_latex(...)`, `.to_typst(...)`, `.to_json(indent=2)` y `.plot(metric="missing_pct" | "n_valid" | "adf_median_p" | "kpss_median_p", ...)`.

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

### Alcance y límites

- Admite paneles con `MultiIndex` (`unit`, `time`), `DatetimeIndex`, `PeriodIndex` o columnas explícitas/autodetectadas `unit_col` y `time_col` (también acepta los alias `unit` y `time`).
- Las pruebas de estacionariedad (`run_stationarity=True`, ADF y KPSS) y de estacionalidad (`run_seasonality=True`) son diagnósticos informativos evaluados unidad por unidad al nivel de significancia `alpha`; los paneles cortos por debajo de `min_obs` o las unidades que superan `max_missing_share` generan hallazgos explícitos en lugar de descartar datos o transformar variables en silencio.
- Superar el preflight indica que el conjunto cumple los umbrales estructurales configurados (`n_errors == 0`); no garantiza identificación causal ni correcta selección de rezagos.

---

## 2. Mallas de sensibilidad para proyecciones locales (`lp_sensitivity_grid`)

`lp_sensitivity_grid` evalúa el producto cartesiano de una malla de hiperparámetros sobre estimadores de proyecciones locales y compara cada respuesta resultante dentro de un `IRFComparisonResult` unificado.

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

Comportamientos clave:

- **Normalización entre estimadores:** Armoniza automáticamente `lags <-> n_lags`, `horizon <-> horizons`, `ci <-> alpha`, y mapea `x` hacia `x`, `treat` o `x_endog` según la firma del estimador. Incluir `"estimator"` dentro de `grid` permite barrer distintos estimadores (por ejemplo `lp_hac` y `lp_iv`) junto con rezagos u horizontes.
- **Trazabilidad y manejo de errores:** El diccionario resuelto de parámetros de cada celda se guarda en `metrics["resolved_params"]`. Con `on_error="skip"`, las especificaciones fallidas se registran en `result.errors` sin abortar el barrido.

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

### Alcance y límites

- Se requieren al menos dos especificaciones exitosas; de lo contrario se lanza `ValueError`.
- Por defecto (`horizon_align="exact"`), los horizontes distintos lanzan un error; pase `horizon_align="intersection"` siempre que `horizon` o `horizons` cambien en la malla.
- Las métricas de malla describen estabilidad entre elecciones discretas; los intervalos de confianza provienen de cada estimación individual y no se corrigen por comparaciones múltiples sobre la malla.

---

## 3. Comparación de IRF entre modelos (`compare_irfs`)

`compare_irfs` alinea y resume dos o más funciones de impulso-respuesta ya estimadas provenientes de proyecciones locales, VAR estructurales, IRF bayesianas o `DataFrame` simples.

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

`IRFComparisonResult` expone `.table`, `.metrics`, `.baseline`, `.horizons`, `.warnings` y `.errors`, además de `.to_frame(layout="wide" | "coef_se" | "metrics")` (alias `.to_dataframe(...)`), `.to_manuscript_dataframe(...)`, `.summary()`, `.to_markdown(...)`, `.to_latex(...)`, `.to_typst(...)` y `.plot(show_bands=True, ...)`.

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
    {"LP (2 rezagos)": lp_2, "LP (4 rezagos)": lp_4},
    baseline="LP (2 rezagos)",
)
print(cmp_res.summary())
print(cmp_res.to_frame("wide").head())
```

### Alcance y límites

- Los selectores (`target`, `shock`, `regime`, `quantile`) aceptan un escalar común o un mapeo por modelo para extraer series 1-D comparables desde resultados multidimensionales de VAR, regímenes o cuantiles.
- Mezclar respuestas acumuladas y en niveles falla por defecto (`allow_mixed_cumulative=False`), y las diferencias en el nivel o tipo de banda emiten advertencias (`band_compat="warn"`) o errores (`band_compat="raise"`).
- `compare_irfs` calcula distancias descriptivas (`max_abs_diff_vs_base`, `rmse_vs_base`, `sign_agreement`), no pruebas formales de Hausman o de abarcamiento entre estimadores.

---

## 4. Scorecards de validación DSGE (`prior_posterior_comparison`, `score_irf_against_targets`)

El módulo de scorecards DSGE provee dos diagnósticos: actualización previa-posterior (`prior_posterior_comparison`) y ajuste de las IRF del modelo frente a momentos empíricos (`score_irf_against_targets`).

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

- **`PriorPosteriorComparisonResult`** verifica contracción de desviación estándar marginal (`post_std / prior_std >= weak_id_threshold`), cercanía a cotas (`boundary_tol`), split-`rhat`, ESS de masa y colas, y tasa de divergencias. Provee `.to_frame(layout="full" | "compact")`, `.summary()`, `.to_markdown(...)`, `.to_latex(...)`, `.to_typst(...)` y `.plot(...)`.
- **`IRFScorecardResult`** calcula `rmse`, `mae`, `wse`, forma cuadrática `chi2_stat` / `chi2_df` / `chi2_pvalue`, `sign_concordance`, `band_coverage`, `determinacy_rate`, `weight_mode_used` y `covariance_condition_number`. Cuando `targets` es un objeto `MomentTargets` con covarianza finita, `weight_mode="auto"` calcula el estadístico de Wald generalizado; al recibir mapeos marginales `LPResult` sin covarianza entre horizontes, recurre a `"diagonal"` con `chi2_pvalue=NaN` y registra una advertencia explícita.

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

### Alcance y límites

- La contracción marginal (`std_ratio >= weak_id_threshold`) detecta ausencia univariada de actualización respecto a la previa; no equivale a una prueba global de identificación ni de matriz de información de Fisher multiparamétrica.
- `chi2_stat` trata la covarianza objetivo como ponderación fija y omite incertidumbre de estimación de parámetros salvo que se evalúen bandas posteriores junto con la covarianza de objetivos.
- En transiciones HANK cuya normalización del choque difiera de los objetivos empíricos agregados (como el valor por defecto de -25 pb), `hank_unit_mode="warn"` advierte la discrepancia de escala sin reescalar automáticamente; use `hank_unit_mode="scale_to_target"` solo cuando el escalado lineal sea válido teóricamente.

---

## 5. Convergencia, exactitud y diagnósticos de régimen del solver (`puremacro.solvers.diagnostics`)

Los envoltorios del solver separan tres ejes ortogonales de salud numérica en estructuras tipadas para que los flujos posteriores no confundan "el optimizador se detuvo" con "solución económicamente válida".

```text
ConvergenceStatus(converged, iterations=0, stopping_reason="", solver_method="", tolerance=0.0)
AccuracyStatus(is_accurate, max_residual=0.0, residual_norm=0.0, residual_type="", tolerance=0.0, residuals={})
SupportStatus(is_supported, regime="interior", warnings=(), boundary_metrics={})
VALID_OVERALL_STATUSES = ("PASS", "CONVERGED_INACCURATE", "UNSUPPORTED_REGIME", "FAILED")
determine_overall_status(convergence, accuracy, support) -> str
make_solver_diagnostic_record(...) -> SolverDiagnosticRecord
```

Precedencia aplicada por `determine_overall_status`:
1. `not convergence.converged` -> `"FAILED"`
2. `not accuracy.is_accurate` -> `"CONVERGED_INACCURATE"`
3. `not support.is_supported` -> `"UNSUPPORTED_REGIME"`
4. En otro caso -> `"PASS"`

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

### Alcance y límites

- Estos ayudantes implementan un contrato tipado de clasificación y auditoría; no vuelven a resolver ecuaciones ni evalúan residuos sobre los objetos del modelo.
- Dado que `make_solver_diagnostic_record` usa por omisión `is_accurate=True` e `is_supported=True`, el llamador debe aportar las comprobaciones reales de residuo y de dominio de soporte desde su solver en lugar de confiar en los valores predeterminados.

---

## 6. Aranceles compuestos y resúmenes por lotes (`CompositeTariffScenario`, `summarize_scenario_batch`)

`CompositeTariffScenario` y `combine_scenarios` fusionan múltiples paquetes bilaterales o sectoriales `TariffScenario` con políticas explícitas ante conflictos de celda, y `summarize_scenario_batch` resume bienestar multipaís, PIB, IPC y matrices de pagos entre escenarios.

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

Métodos clave:
- En `CompositeTariffScenario`: `.importers`, `.to_rate_tensor(country_codes, sector_codes)`, `.to_frame(country_codes, sector_codes, *, nonzero_only=True)` y `.conflicts(country_codes, sector_codes)`. Está registrado como `TariffScenario`, por lo que `run_scenario` y `run_scenario_batch` lo resuelven directamente.
- En `ScenarioComparisonResult`: `.to_frame(layout="full" | "compact" | "summary")`, `.to_payoff_matrix(metric="ev_pct_consumption", *, include_baseline=True)`, `.compare_robust(...)`, `.summary()`, `.to_markdown(...)`, `.to_latex(...)`, `.to_typst(...)` y `.plot(...)`.

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

print("Celdas superpuestas resueltas con `max`:")
print(combined.conflicts(calib.country_codes, calib.sector_codes)[["exporter", "importer", "sector", "resolved_rate", "rule"]])

batch = run_scenario_batch(calib, [broad, targeted, combined])
summary = summarize_scenario_batch(batch, calib)
print(summary.to_frame("compact").head(6))
print(summary.to_payoff_matrix(metric="ev_pct_consumption"))
```

### Alcance y límites

- Las reglas de conflicto (`"last_wins"`, `"first_wins"`, `"error"`, `"max"`, `"sum"`) resuelven tasas arancelarias superpuestas `(importer, exporter, sector)` linealmente antes del equilibrio; `"sum"` no acota automáticamente las tasas más allá de la admisibilidad ad valorem estática.
- `summarize_scenario_batch` hereda la estructura estática Dekle-Eaton-Kortum de `calib` (participaciones sectoriales de gasto constantes y déficits comerciales fijos bajo el numerario elegido); variaciones numéricas diminutas de bienestar dentro de `welfare_tol` reflejan tolerancias del solver más que diferencias económicas sustantivas.

---

## 7. Banco de trabajo robusto a la identificación, SMM y exportación bilingüe

`run_empirical_structural_workbench` y `EmpiricalStructuralDossier` unifican preflight de panel, sensibilidad LP, comparación IRF, scorecards de validación, diagnósticos del solver, pruebas de restricción de momentos e inversión exacta 1D robusta a la identificación (`puremacro.structural.robust`), estimación opcional por método simulado de momentos (`puremacro.structural.simulation`) y exportación autocontenida en paquetes bilingües.

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

Nuevos atributos y métodos en `EmpiricalStructuralDossier`:
- **Contratos estructurales adjuntos**: `.moment_test` (`MomentTestResult`), `.moment_confidence_set` (`MomentConfidenceSet`), `.moment_grid` (`MomentTestGridResult`) y `.smm_fit` (`SMMFitResult`).
- **Secciones adicionales en `.to_frame(section=...)`**: `"moment_test"`, `"confidence_set"`, `"moment_grid"`, `"moment_grid_cells"`, `"smm_fit"` y `"smm_moments"`.
- **HTML autocontenido y paquetes portables**:
  - `dossier.to_html(lang="en" | "es", digits=3) -> str` genera un documento HTML5 UTF-8 autocontenido con escape HTML de todos los campos dinámicos y cero rutas del sistema anfitrión.
  - `dossier.to_bundle(output_dir, *, languages=("en", "es"), digits=3, overwrite=True) -> dict[str, Any]` escribe `dossier.json`, `dossier_{en,es}.{html,md,tex}` y `manifest.json` con resúmenes SHA-256 usando únicamente nombres de archivo relativos (sin filtrar rutas del sistema de archivos anfitrión).

### Flujos CLI unificados (`puremacro.cli`)

La CLI conserva importaciones de biblioteca estándar en el nivel superior, devuelve código de salida `2` ante errores de argumentos o esquema JSON e imprime de forma segura en terminales Windows/POSIX:

```bash
# Exportar expediente bilingüe y paquete con manifiesto SHA-256 desde workbench o lp-grid
python -m puremacro.cli workbench data.csv --y y --x shock --bundle-dir ./out --robust-spec spec.json --format html
python -m puremacro.cli lp-grid data.csv --y y --x shock --lags 1..3 --workbench --bundle-dir ./out --lang es

# Comparar cierres fiscales HANK financiados en espacio de secuencias (balanced_budget vs debt_rule)
python -m puremacro.cli hank-fiscal --mode compare --horizon 40 --phi-b 0.08 --format markdown
python -m puremacro.cli hank-fiscal --mode single --closure debt_rule --tax-type lump_sum --horizon 40 --format json

# Evaluar hipótesis nulas de momentos, curvas cuadráticas y mallas desde un JSON
python -m puremacro.cli robust-set spec.json --alpha 0.05 --calibration asymptotic --format markdown
```

### Alcance y límites

- Las importaciones de nivel superior en `puremacro.workbench` siguen restringidas a biblioteca estándar, NumPy y pandas (`puremacro.structural.robust` y `puremacro.structural.simulation` se importan de forma perezosa solo cuando se solicitan).
- La inversión exacta 1D mediante `invert_quadratic_moments` preserva intervalos desconectados, indicadores de truncamiento (`truncated`, `boundary_contact`) y estados `unresolved_*` (sin sustituir matrices singulares mediante pseudoinversas o regularización ridge).
- Las mallas muestreadas (`moment_grid` / `evaluate_moment_grid`) conservan explícitamente `continuous_complete=False`, `accepted_runs()` y `unresolved_cells()` para no certificar falsamente interiores no evaluados como intervalos continuos de confianza.
