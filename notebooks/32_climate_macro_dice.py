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
# # Integrated Climate-Macroeconomics — The DICE Model and Carbon Pricing
#
# **How do economic growth, greenhouse gas emissions, and global climate feedback interact over century-long horizons?**
#
# The Dynamic Integrated Climate-Economy (DICE) model, pioneered by William Nordhaus (2018 Nobel Laureate) and extended by Golosov, Hassler, Krusell & Tsyvinski (2014, *Econometrica*), is the canonical benchmark for assessing:
# 1. **Atmospheric Carbon Stock Accumulation**: 3-reservoir carbon cycle dynamics.
# 2. **Surface Temperature Anomalies**: Radiative forcing and ocean heat uptake.
# 3. **Macroeconomic Climate Damages**: Nonlinear output loss $D(T_t) Y_t$.
# 4. **Optimal Carbon Pricing**: Calculating the Social Cost of Carbon (SCC).
#
# In this interactive showcase, we simulate the DICE model forward across 150 years using `puremacro.climate.simulate_dice_model`.

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
# ## 1. Simulating Baseline vs. Accelerated Carbon Tax Policies
#
# We compare:
# - **Baseline**: Moderate carbon tax ($40/tCO2, +2%/year).
# - **Accelerated Policy**: Ambitious climate policy ($80/tCO2, +3.5%/year).

# %%
res_base = simulate_dice_model(
    n_periods=30,
    time_step_years=5,
    start_year=2020,
    carbon_tax_initial=40.0,
    carbon_tax_growth=0.02,
)
print("=== Baseline Scenario ===")
print(res_base.summary())

res_policy = simulate_dice_model(
    n_periods=30,
    time_step_years=5,
    start_year=2020,
    carbon_tax_initial=80.0,
    carbon_tax_growth=0.035,
)
print("\n=== Accelerated Carbon Tax Scenario ===")
print(res_policy.summary())

# %% [markdown]
# ## 2. Surface Warming Anomalies & Paris Agreement Guardrails

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(res_base.trajectories.index, res_base.trajectories["temperature_anomaly"], color="#d62728", lw=2, label="Base Scenario ($40/t, +2%/yr)")
ax.plot(res_policy.trajectories.index, res_policy.trajectories["temperature_anomaly"], color="#2ca02c", lw=2, linestyle="--", label="Accelerated Scenario ($80/t, +3.5%/yr)")
ax.axhline(1.5, color="gray", linestyle=":", label="1.5°C Paris Ambition")
ax.axhline(2.0, color="gray", linestyle="-.", label="2.0°C Guardrail")
ax.set_title("Global Mean Surface Temperature Anomaly (°C above Pre-industrial)", fontsize=12, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Warming Anomaly (°C)")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Global CO2 Emissions and Decarbonization Paths

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(res_base.trajectories.index, res_base.trajectories["emissions"], color="#d62728", lw=2, label="Base Emissions")
ax.plot(res_policy.trajectories.index, res_policy.trajectories["emissions"], color="#2ca02c", lw=2, linestyle="--", label="Accelerated Emissions")
ax.set_title("Annual Global CO2 Emissions (GtCO2/year)", fontsize=12, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Emissions (GtCO2)")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Macroeconomic Climate Damages (% of Gross World Product)

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(res_base.trajectories.index, res_base.trajectories["damage_fraction"] * 100, color="#d62728", lw=2, label="Base Damage Share")
ax.plot(res_policy.trajectories.index, res_policy.trajectories["damage_fraction"] * 100, color="#2ca02c", lw=2, linestyle="--", label="Accelerated Damage Share")
ax.set_title("Macroeconomic Climate Damages (% of Global GDP)", fontsize=12, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Damages (% GDP)")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
