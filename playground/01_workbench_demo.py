# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PureMacro Empirical Workbench & Scorecards Demo
#
# End-to-end, zero-network, Pyodide-safe interactive tour connecting:
# 1. `preflight_panel` — structural & stationarity panel checks
# 2. `lp_sensitivity_grid` — local-projection hyperparameter sweep
# 3. `compare_irfs` — cross-specification IRF comparison
# 4. `CompositeTariffScenario` & `summarize_scenario_batch` — composite tariff policies & multi-country welfare
# 5. `ConvergenceStatus`, `AccuracyStatus`, `SupportStatus` & `make_solver_diagnostic_record` — solver health triage
#
# In Jupyter / JupyterLite, uncomment and run the `%pip install` line once if PureMacro is not already installed.

# %%
# %pip install puremacro

# %%
import numpy as np
import pandas as pd

from puremacro.data import preflight_panel
from puremacro.lp import lp_hac
from puremacro.lp.sensitivity import lp_sensitivity_grid
from puremacro.reports import compare_irfs
from puremacro.solvers.diagnostics import (
    AccuracyStatus,
    ConvergenceStatus,
    SupportStatus,
    determine_overall_status,
    make_solver_diagnostic_record,
)
from puremacro.trade import (
    TariffScenario,
    calibrate_trade_model,
    combine_scenarios,
    run_scenario_batch,
    summarize_scenario_batch,
)
from puremacro.trade.data import synthetic_icio_table

# %% [markdown]
# ## 1. Panel Preflight Checks (`preflight_panel`)
#
# Build a deterministic 3-country quarterly panel with a small tail missing block and run preflight diagnostics before estimating any regressions.

# %%
rng = np.random.default_rng(7)
dates = pd.date_range("2018-01-01", periods=24, freq="QS")

panel_rows = []
for iso in ("USA", "MEX", "CAN"):
    gdp = np.cumsum(rng.normal(0.0, 0.5, len(dates)))
    cpi = 2.0 + rng.normal(0.0, 0.3, len(dates))
    shock = rng.normal(0.0, 1.0, len(dates))
    for d, y_val, p_val, s_val in zip(dates, gdp, cpi, shock):
        panel_rows.append(
            {
                "country": iso,
                "date": d,
                "gdp": float(y_val),
                "cpi": float(p_val),
                "shock": float(s_val),
            }
        )

panel = pd.DataFrame(panel_rows)
panel.loc[(panel["country"] == "CAN") & (panel["date"] >= "2023-07-01"), "cpi"] = np.nan

preflight_report = preflight_panel(
    panel,
    unit_col="country",
    time_col="date",
    variables=["gdp", "cpi", "shock"],
    min_obs=12,
    max_missing_share=0.25,
)
print("=== 1. Panel Preflight ===")
print(f"status={preflight_report.status} passed={preflight_report.passed} freq={preflight_report.inferred_freq}")
print(preflight_report.to_frame("variables")[["variable", "n_obs", "n_valid", "missing_pct", "adf_median_p"]].to_string(index=False))

# %% [markdown]
# ## 2. Local-Projection Sensitivity Grid (`lp_sensitivity_grid`)
#
# Sweep lag and horizon choices over a single-country macro sample extracted from a stationary synthetic DGP, aligning mismatched horizons at their common intersection.

# %%
T = 80
shock_series = rng.normal(size=T)
y_series = np.zeros(T)
for t in range(1, T):
    y_series[t] = 0.6 * y_series[t - 1] - 0.5 * shock_series[t] + 0.2 * rng.normal()
lp_df = pd.DataFrame({"y": y_series, "shock": shock_series})

lp_grid_result = lp_sensitivity_grid(
    lp_df,
    "y",
    "shock",
    grid={"lags": [1, 2], "horizon": [4, 6]},
    baseline_spec={"lags": 2, "horizon": 4},
    horizon_align="intersection",
)
print("\n=== 2. LP Sensitivity Grid ===")
print(lp_grid_result.to_frame("metrics")[["spec", "baseline", "peak_h", "peak_response", "rmse_vs_base"]].to_string(index=False))

# %% [markdown]
# ## 3. Cross-Specification IRF Comparison (`compare_irfs`)
#
# Compare two explicit LP-HAC lag specifications directly against a designated baseline.

# %%
irf_2lags = lp_hac(lp_df, y="y", x="shock", lags=2, horizon=4, ci=0.90)
irf_4lags = lp_hac(lp_df, y="y", x="shock", lags=4, horizon=4, ci=0.90)

irf_comparison = compare_irfs(
    {"LP (lags=2)": irf_2lags, "LP (lags=4)": irf_4lags},
    baseline="LP (lags=2)",
)
print("\n=== 3. Cross-Specification IRF Comparison ===")
print(irf_comparison.summary())

# %% [markdown]
# ## 4. Composite Tariff Scenarios & Batch Summary (`CompositeTariffScenario`, `summarize_scenario_batch`)
#
# Calibrate a compact 3-country, 2-sector synthetic ICIO economy, combine overlapping tariff schedules with `on_conflict="max"`, solve the equilibrium batch, and inspect welfare payoffs.

# %%
icio = synthetic_icio_table(3, 2, nfd=3, seed=42)
calib = calibrate_trade_model(icio)
c0, c1, c2 = calib.country_codes[:3]
s0_code = calib.sector_codes[0]
s1_code = calib.sector_codes[1]

broad = TariffScenario(importer=c0, tariffs={(c1, "*"): 0.10}, name="broad_10pct")
targeted = TariffScenario(importer=c0, tariffs={(c1, s0_code): 0.25, (c2, s1_code): 0.15}, name="targeted")
composite_scenario = combine_scenarios([broad, targeted], name="combined_max", on_conflict="max")

conflicts_df = composite_scenario.conflicts(calib.country_codes, calib.sector_codes)
batch_tight = run_scenario_batch(calib, [broad, targeted, composite_scenario], tol=1e-7)
batch_summary = summarize_scenario_batch(batch_tight, calib)

print("\n=== 4. Composite Tariff Scenarios & Batch Welfare ===")
print(conflicts_df[["exporter", "importer", "sector", "resolved_rate", "rule"]].to_string(index=False))
print(batch_summary.to_payoff_matrix(metric="ev_pct_consumption").round(6).to_string())

# %% [markdown]
# ## 5. Solver Convergence, Accuracy, and Support Diagnostics
#
# Compare a loose solver run against the tight (`tol=1e-7`) run and construct structured `ConvergenceStatus`, `AccuracyStatus`, `SupportStatus`, and `SolverDiagnosticRecord` audits.

# %%
batch_loose = run_scenario_batch(calib, [composite_scenario], tol=2.5e-3)
loose_rec = batch_loose[composite_scenario.name].solver_diagnostics
tight_rec = batch_tight[composite_scenario.name].solver_diagnostics

manual_status = determine_overall_status(
    ConvergenceStatus(converged=True, iterations=12, stopping_reason="step_stall", solver_method="newton", tolerance=1e-8),
    AccuracyStatus(is_accurate=False, max_residual=3.2e-3, residual_type="excess_demand", tolerance=1e-6),
    SupportStatus(is_supported=True, regime="interior"),
)
constructed_pass = make_solver_diagnostic_record(
    converged=tight_rec.convergence.converged,
    iterations=tight_rec.convergence.iterations,
    stopping_reason=tight_rec.convergence.stopping_reason,
    solver_method=tight_rec.convergence.solver_method,
    tolerance=tight_rec.convergence.tolerance,
    is_accurate=tight_rec.accuracy.is_accurate,
    max_residual=tight_rec.accuracy.max_residual,
    residual_norm=tight_rec.accuracy.residual_norm,
    residual_type=tight_rec.accuracy.residual_type,
    accuracy_tolerance=tight_rec.accuracy.tolerance,
)
solver_records = {
    "loose_solve": loose_rec,
    "tight_solve": tight_rec,
    "constructed_tight": constructed_pass,
    "manual_stall_status": manual_status,
}

print("\n=== 5. Solver Diagnostics ===")
print(f"loose_solve: overall={loose_rec.overall_status} converged={loose_rec.convergence.converged} max_residual={loose_rec.accuracy.max_residual:.2e}")
print(f"tight_solve: overall={tight_rec.overall_status} converged={tight_rec.convergence.converged} max_residual={tight_rec.accuracy.max_residual:.2e}")
print(f"manual_stall_status={manual_status} | constructed_tight.is_pass={constructed_pass.is_pass}")
