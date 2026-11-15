# %% [markdown]
# # PureMacro Lab 02: Composite Tariffs, Fiscal HANK Closures, and CCE Mean-Group Scorecards
#
# Runs offline in your browser (Pyodide) or locally from the repository. Zero
# network requests and zero optional desktop dependencies are required.
#
# This notebook walks through three research-grade PureMacro v6.5 workflows:
# 1. **Composite tariff policy design**: `TariffScenario`,
#    `CompositeTariffScenario`, `combine_scenarios`, and
#    `summarize_scenario_batch`.
# 2. **Structural fiscal HANK closures & Walras audits**: `solve_hank_fiscal`
#    and `compare_fiscal_closures` across balanced-budget and debt-rule regimes
#    versus an unfinanced goods-only baseline.
# 3. **Factor-robust panel LP + workbench evaluation**: Pesaran Common
#    Correlated Effects Mean Group estimation (`cce_panel_lp(...,
#    estimator="mean_group")` / `cce_mg_panel_lp`) scored against the HANK
#    fiscal impulse responses via `run_empirical_structural_workbench`.

# %%
# %pip install puremacro

from __future__ import annotations

import numpy as np
import pandas as pd

from puremacro.lp.cce import cce_mg_panel_lp, cce_panel_lp
from puremacro.models.hank_fiscal import compare_fiscal_closures, solve_hank_fiscal
from puremacro.trade import (
    CompositeTariffScenario,
    TariffScenario,
    calibrate_trade_model,
    combine_scenarios,
    run_scenario_batch,
    summarize_scenario_batch,
)
from puremacro.trade.data import synthetic_icio_table
from puremacro.workbench import run_empirical_structural_workbench

# %% [markdown]
# ## 1. Composite Tariff Scenarios & Batch Payoff Matrix
#
# Construct a compact 3-country, 2-sector calibrated ICIO economy and compare
# single-country tariffs against composite multi-importer policy packages built
# with `CompositeTariffScenario` and `combine_scenarios`.

# %%
icio = synthetic_icio_table(3, 2, nfd=3, seed=42)
calib = calibrate_trade_model(icio)
c0, c1, c2 = calib.country_codes
s0, s1 = calib.sector_codes

c0_broad = TariffScenario(
    importer=c0,
    tariffs={(c1, "*"): 0.10, (c2, "*"): 0.10},
    name="c0_broad_10pct",
)
c0_carveout = TariffScenario(
    importer=c0,
    tariffs={(c1, s0): 0.00, (c2, s1): 0.20},
    name="c0_targeted_override",
)
c1_retaliation = TariffScenario(
    importer=c1,
    tariffs={(c0, "*"): 0.12},
    name="c1_retaliation",
)

# Composition rules resolve overlapping lines transparently:
# "last_wins" applies sectoral carve-outs on top of a broad tariff schedule.
c0_rebalanced = combine_scenarios(
    [c0_broad, c0_carveout],
    name="c0_carveout_package",
    on_conflict="last_wins",
)
trade_war_pkg = CompositeTariffScenario(
    scenarios=(c0_rebalanced, c1_retaliation),
    name="bilateral_package",
    on_conflict="max",
)

batch = run_scenario_batch(
    calib,
    [c0_broad, c0_rebalanced, trade_war_pkg],
    tol=1e-7,
)
comparison = summarize_scenario_batch(batch, calib)

print("EV welfare payoff matrix (% equivalent consumption):")
print(comparison.to_payoff_matrix(metric="ev_pct_consumption").round(4))
print("\nScenario batch summary:")
print(comparison.to_frame("summary").round(4))

# %% [markdown]
# ## 2. HANK Fiscal Closures & Walras Diagnostics
#
# Solve a temporary government spending shock (`G` shock) on a fast asset grid
# (`n_a=60`, `a_max=100.0`) under a stabilizing debt rule with proportional
# taxes, inspect its structured `solver_diagnostics` record, and compare all
# fiscal closures alongside the unfinanced goods-only diagnostic check.

# %%
fiscal_res = solve_hank_fiscal(
    closure="debt_rule",
    tax_type="proportional",
    horizon=40,
    n_a=60,
    a_max=100.0,
    tol=1e-7,
)

diag = fiscal_res.solver_diagnostics
print(
    f"HANK debt_rule (proportional): converged={fiscal_res.converged}, "
    f"impact_mult={fiscal_res.impact_multiplier:.4f}, "
    f"cum_mult={fiscal_res.cumulative_multiplier:.4f}"
)
print(
    f"Solver diagnostics: overall_status={diag.overall_status}, "
    f"max_residual={diag.accuracy.max_residual:.2e}, "
    f"regime={diag.support.regime}, top_grid_mass={diag.support.boundary_metrics['top_grid_mass']:.2e}"
)

closure_table = compare_fiscal_closures(
    horizon=40,
    n_a=60,
    a_max=100.0,
    tol=1e-7,
)
print("\nFiscal closure comparison (note GOODS_ONLY_UNFINANCED warning row):")
print(
    closure_table[
        ["closure", "tax_type", "impact_multiplier", "cumulative_multiplier", "goods_residual", "gov_budget_residual", "status"]
    ].round(6)
)

# %% [markdown]
# ## 3. CCE Mean Group Panel LP & Empirical-Structural Workbench
#
# Simulate a synthetic multi-country panel driven by an unobserved common
# global factor with country-specific factor loadings plus a fiscal shock `x`.
# Use Pesaran's `cce_panel_lp(..., estimator="mean_group")` to strip cross-section
# averages, then score the HANK fiscal IRF against the CCE Mean Group empirical
# path inside `run_empirical_structural_workbench`.

# %%
rng = np.random.default_rng(2026)
n_units = 10
n_periods = 44
n_horizons = 7

# Normalize HANK output response per unit of the spending shock (default shock=0.01)
true_irf = np.asarray(fiscal_res.dY[:n_horizons], dtype=float) / 0.01
factor_t = rng.normal(0.0, 0.8, size=n_periods)
factor_lag = np.roll(factor_t, 1)
factor_lag[0] = 0.0

rows: list[dict[str, float | int | str]] = []
for i in range(n_units):
    unit = f"C{i:02d}"
    loading_x = 0.6 + 0.15 * i
    loading_y = 0.9 + 0.20 * ((i % 4) - 1.5)
    shocks = rng.normal(0.0, 1.0, size=n_periods)
    x_series = shocks + loading_x * factor_t
    # Build y with dynamic shock response aligned to true_irf plus unit factor loadings
    y_series = loading_y * factor_t + 0.3 * loading_y * factor_lag
    for h_lag in range(n_horizons):
        shifted = np.roll(shocks, h_lag)
        shifted[:h_lag] = 0.0
        y_series = y_series + true_irf[h_lag] * shifted
    y_series = y_series + rng.normal(0.0, 0.20, size=n_periods)
    for t in range(n_periods):
        rows.append({"unit": unit, "period": t, "y": float(y_series[t]), "x": float(x_series[t])})

panel_df = pd.DataFrame(rows)

cce_mg = cce_panel_lp(
    panel_df,
    y="y",
    x="x",
    horizons=range(n_horizons),
    n_lags=2,
    estimator="mean_group",
    unit_col="unit",
    time_col="period",
)
print("CCE Mean Group Panel LP estimates:")
print(cce_mg[["h", "beta", "se", "lo", "hi", "n_entities_used"]].round(4))

model_irf_df = pd.DataFrame(
    {
        "h": np.arange(n_horizons, dtype=int),
        "point": true_irf,
    }
)
dossier = run_empirical_structural_workbench(
    panel_df,
    y="y",
    x="x",
    unit_col="unit",
    time_col="period",
    horizons=range(n_horizons),
    lags=2,
    estimator=cce_mg_panel_lp,
    run_sensitivity=False,
    model_irf=model_irf_df,
    model_label="HANK Fiscal (debt_rule)",
    solver_result=fiscal_res,
)

print("\nWorkbench status:", dossier.overall_status, "| passed:", dossier.passed)
if dossier.irf_comparison is not None:
    print("Workbench IRF comparison:")
    print(dossier.irf_comparison.to_frame().round(4))
if dossier.irf_scorecard is not None:
    print("\nWorkbench scorecard:")
    print(dossier.irf_scorecard.to_frame())
print(dossier.to_markdown())
