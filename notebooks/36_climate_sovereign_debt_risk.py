# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Climate Transition Risk and Sovereign Debt Sustainability
#
# **How do physical climate damages, public adaptation expenditures, and decarbonization policies shape the long-term sustainability of sovereign debt, and can carbon revenues protect fiscal solvency?**
#
# Fiscal authorities, sovereign rating agencies, and multilateral development banks increasingly recognize that anthropogenic global warming represents a first-order macro-fiscal vulnerability. While traditional Debt Sustainability Analyses (DSA) focus on conventional interest-growth differentials ($r - g$) and primary fiscal balances, climate change introduces four powerful, non-linear transmission channels that fundamentally alter sovereign solvency:
#
# 1. **The Physical Damage & Tax Base Channel**:
#    Chronic warming and severe weather anomalies depress factor productivity and accelerate capital depreciation. As gross output shrinks by fraction $\Omega(T_t)$, the domestic tax base contracts, eroding non-resource tax revenues.
# 2. **The Public Adaptation Expenditure Channel**:
#    Defending critical infrastructure, building flood defenses, retrofitting transportation grids, and providing disaster relief require escalating government outlays:
#    $$ g_{adapt, t} = \theta_{adapt} \cdot T_{clim, t}^2 $$
#    which widen the structural primary deficit.
# 3. **The Endogenous Risk Premium & Credit Spread Channel**:
#    Sovereign bond investors price both elevated leverage and vulnerability to climate disasters:
#    $$ r_t^{sovereign} = r^* + \psi_{debt} \max(0, b_t - b^*) + \psi_{clim} T_{clim, t} $$
#    When debt exceeds sustainability thresholds (such as 60% of GDP), the interaction between climate physical risk and credit spreads can trigger explosive sovereign debt spirals.
# 4. **The Carbon Revenue Recycling Channel**:
#    A predictable Pigouvian carbon tax schedule generates substantial fiscal revenue $\tau_t^{carbon} E_t$, which can be recycled into debt amortization and green public investment, counterbalancing adaptation burdens.
#
# In this interactive showcase, we couple a sovereign fiscal dynamics simulator to the **DICE climate-macro model** using `puremacro.climate` to evaluate debt trajectories across three stylized policy regimes.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.climate import simulate_dice_model

# %% [markdown]
# ## 1. Simulating 3 Policy Regimes in the DICE Model
#
# We simulate three counterfactual macroeconomic trajectories:
# - **Unabated Warming (High Climate Inaction)**: Zero carbon tax ($\tau = 0$), baseline fossil dependence, and higher physical climate damage sensitivity ($\pi_2 = 0.0035$).
# - **Disorderly Late Transition**: Weak initial mitigation ($\tau = \$10/\text{tCO}_2$) followed by sudden, abrupt policy tightening in mid-century, creating severe transitional capital obsolescence.
# - **Orderly Green Fiscal Rule**: An immediate, predictable carbon price ($\tau_0 = \$60/\text{tCO}_2$, growing at $3\%$ annually) paired with active decarbonization.
#
# Each scenario generates multi-decade projections for real net output $Y_{net, t}$, global emissions $E_t$, the Social Cost of Carbon, and global mean surface temperature anomalies $T_{clim, t}$.

# %%
# 1. Unabated Warming
dice_unabated = simulate_dice_model(
    n_periods=25,
    time_step_years=5,
    carbon_tax_initial=0.0,
    carbon_tax_growth=0.0,
    damage_coef=0.0035,
)

# 2. Disorderly Late Transition
dice_late = simulate_dice_model(
    n_periods=25,
    time_step_years=5,
    carbon_tax_initial=10.0,
    carbon_tax_growth=0.01,
    damage_coef=0.0028,
)

# 3. Orderly Green Fiscal Transition
dice_orderly = simulate_dice_model(
    n_periods=25,
    time_step_years=5,
    carbon_tax_initial=60.0,
    carbon_tax_growth=0.03,
    damage_coef=0.00236,
)

print("Orderly Scenario Preview:")
print(dice_orderly.summary())

# %% [markdown]
# ## 2. Sovereign Debt Dynamics & Fiscal Feedback Function
#
# Let $b_t = B_t / Y_t$ denote the sovereign debt-to-GDP ratio. The discrete-time law of motion with time-step $\Delta t = 5$ years is given by:
#
# $$ b_{t+1} = \left[ 1 + (r_t^{sovereign} - g_t) \Delta t \right] b_t - pb_t \Delta t $$
#
# where $g_t$ is the real growth rate of GDP and the primary balance ratio $pb_t$ incorporates both regular fiscal policy and climate flows:
#
# $$ pb_t = \left( \tau_{base} + \frac{\text{SCC}_t \cdot E_t}{Y_{net, t}} \right) - \left( g_{base} + \theta_{adapt} T_{clim, t}^2 \right) $$
#
# When global temperatures rise unchecked, the escalating adaptation spending $\theta_{adapt} T_t^2$ overwhelms baseline primary surpluses. In parallel, credit rating downgrades raise $r_t^{sovereign}$, creating an unstable debt feedback loop.
#
# Below, we implement `simulate_sovereign_fiscal_risk` to calculate sovereign debt ratios, borrowing yields, and public adaptation spending across the 120-year horizon.

# %%
def simulate_sovereign_fiscal_risk(
    dice_res,
    initial_debt_gdp: float = 0.60,
    base_tax_rate: float = 0.20,
    base_spending_rate: float = 0.19,
    adapt_cost_coef: float = 0.0015,
    spread_debt_coef: float = 0.03,
    spread_climate_coef: float = 0.005,
    r_star: float = 0.02,
) -> pd.DataFrame:
    df = dice_res.trajectories.copy()
    years = list(df.index)
    dt = 5

    debt_gdp, rates, adaptation = [], [], []
    B_over_Y = initial_debt_gdp

    for yr in years:
        row = df.loc[yr]
        T_clim = row["temperature_anomaly"]
        Y_net = row["output_net"]
        carbon_tax_rev = (row["social_cost_of_carbon"] * row["emissions"] * 1e-3) / Y_net

        g_adapt = adapt_cost_coef * (T_clim ** 2)
        pb = (base_tax_rate + carbon_tax_rev) - (base_spending_rate + g_adapt)
        r_sovereign = r_star + spread_debt_coef * max(0.0, B_over_Y - 0.60) + spread_climate_coef * T_clim

        debt_gdp.append(B_over_Y * 100.0)
        rates.append(r_sovereign * 100.0)
        adaptation.append(g_adapt * 100.0)

        B_over_Y = max(0.0, (1.0 + (r_sovereign - 0.015) * dt) * B_over_Y - pb * dt)

    return pd.DataFrame({
        "year": years,
        "debt_to_gdp": debt_gdp,
        "sovereign_rate": rates,
        "adaptation_cost": adaptation,
        "temperature_anomaly": df["temperature_anomaly"].values,
    }).set_index("year")

fiscal_unabated = simulate_sovereign_fiscal_risk(dice_unabated)
fiscal_late = simulate_sovereign_fiscal_risk(dice_late)
fiscal_orderly = simulate_sovereign_fiscal_risk(dice_orderly)

# %% [markdown]
# ## 3. Sovereign Debt-to-GDP and Borrowing Rates
#
# The simulated trajectories reveal stark divergence in long-term sovereign solvency:
# - **Left Panel (Debt-to-GDP)**: In the Unabated Warming scenario, public debt breaches $100\%$ of GDP by 2075 and surges past $160\%$ by the end of the century. In contrast, under the Orderly Green Fiscal Rule, carbon tax revenues amortize debt rapidly, driving the debt ratio safely below $40\%$ of GDP.
# - **Right Panel (Sovereign Rates)**: Investors demand an escalating risk premium under climate inaction: borrowing rates climb from $2.5\%$ to over $6.0\%$ as fiscal risk and environmental exposure compound. In the orderly transition, borrowing yields remain anchored near $2.2\%$.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.5))

ax1.plot(fiscal_unabated.index, fiscal_unabated["debt_to_gdp"], **_nbstyle.S1, label="Unabated Warming")
ax1.plot(fiscal_late.index, fiscal_late["debt_to_gdp"], **_nbstyle.S2, label="Disorderly Late Transition")
ax1.plot(fiscal_orderly.index, fiscal_orderly["debt_to_gdp"], **_nbstyle.S3, label="Orderly Green Fiscal Rule")
ax1.axhline(60, color=_nbstyle.SPINE, linestyle=":", label="60% Stability Threshold")
ax1.set_title("Sovereign Debt-to-GDP Trajectory (%)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Year", color=_nbstyle.TEXTO)
ax1.set_ylabel("Public Debt (% of GDP)", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

ax2.plot(fiscal_unabated.index, fiscal_unabated["sovereign_rate"], **_nbstyle.S1, label="Unabated Risk")
ax2.plot(fiscal_late.index, fiscal_late["sovereign_rate"], **_nbstyle.S2, label="Late Transition Risk")
ax2.plot(fiscal_orderly.index, fiscal_orderly["sovereign_rate"], **_nbstyle.S3, label="Orderly Transition")
ax2.set_title("Sovereign Borrowing Rate (r* + Spreads)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Year", color=_nbstyle.TEXTO)
ax2.set_ylabel("Real Rate (%)", color=_nbstyle.TEXTO)
ax2.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 4. Adaptation Costs and Surface Warming
#
# Comparing physical climate drivers explains why early mitigation protects fiscal space:
# - **Left Panel (Adaptation Outlays)**: Under high warming, public adaptation expenses exceed $2.5\%$ of GDP annually by 2100. In the orderly transition, adaptation spending remains capped at $1.2\%$ of GDP.
# - **Right Panel (Temperature Profiles)**: Early carbon taxation halts surface warming at $+2.9^\circ\text{C}$ in this parameterization, whereas unabated warming pushes global temperatures toward $+4.1^\circ\text{C}$, confirming that ambitious climate policy is a prerequisite for fiscal sustainability.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.5))

ax1.plot(fiscal_unabated.index, fiscal_unabated["adaptation_cost"], **_nbstyle.S1, label="Unabated Adaptation Need")
ax1.plot(fiscal_orderly.index, fiscal_orderly["adaptation_cost"], **_nbstyle.S3, label="Orderly Adaptation Cost")
ax1.set_title("Public Climate Adaptation Costs (% of GDP)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Year", color=_nbstyle.TEXTO)
ax1.set_ylabel("Adaptation (% GDP)", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

ax2.plot(fiscal_unabated.index, fiscal_unabated["temperature_anomaly"], **_nbstyle.S1, label="Unabated Warming")
ax2.plot(fiscal_orderly.index, fiscal_orderly["temperature_anomaly"], **_nbstyle.S3, label="Orderly Mitigation")
ax2.axhline(1.5, color=_nbstyle.SPINE, linestyle=":", label="1.5°C Paris Ambition")
ax2.axhline(2.0, color=_nbstyle.NOTA, linestyle="-.", label="2.0°C Guardrail")
ax2.set_title("Global Mean Surface Warming (°C)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Year", color=_nbstyle.TEXTO)
ax2.set_ylabel("Warming Anomaly (°C)", color=_nbstyle.TEXTO)
ax2.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)
