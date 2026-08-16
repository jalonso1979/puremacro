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
# # Riesgo de Transición Climática y Sostenibilidad de la Deuda Soberana
#
# **¿Cómo moldean los daños del cambio climático, los costos de adaptación y las políticas tributarias de descarbonización la sostenibilidad de la deuda pública a largo plazo?**
#
# Las autoridades fiscales y los bancos centrales reconocen cada vez más que el cambio climático constituye un riesgo macrofiscal de primer orden:
# 1. **Canal de Daños Directos**: El aumento de la temperatura reduce la productividad laboral y la eficiencia del capital, contrayendo la base tributaria.
# 2. **Canal del Gasto de Adaptación**: Mayor calentamiento exige mayor gasto público en infraestructura resiliente y atención a desastres.
# 3. **Canal de Prima de Riesgo Soberano**: Las tasas soberanas incorporan la vulnerabilidad climática mediante mayores diferenciales de crédito:
#    $$ r_t = r^* + \psi_{debt} \max(0, b_t - 0.60) + \psi_{clim} T_t $$
# 4. **Canal de Reciclaje de Ingresos**: Un precio al carbono predecible genera ingresos fiscales que amortizan deuda y financian infraestructura verde.
#
# En este tutorial interactivo simulamos la dinámica de deuda soberana acoplada al **modelo DICE** mediante `puremacro.climate`.

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
# ## 1. Simulación de 3 Regímenes de Política en el Modelo DICE

# %%
dice_unabated = simulate_dice_model(
    n_periods=25,
    time_step_years=5,
    carbon_tax_initial=0.0,
    carbon_tax_growth=0.0,
    damage_coef=0.0035,
)

dice_late = simulate_dice_model(
    n_periods=25,
    time_step_years=5,
    carbon_tax_initial=10.0,
    carbon_tax_growth=0.01,
    damage_coef=0.0028,
)

dice_orderly = simulate_dice_model(
    n_periods=25,
    time_step_years=5,
    carbon_tax_initial=60.0,
    carbon_tax_growth=0.03,
    damage_coef=0.00236,
)

print("Resumen Escenario Ordenado:")
print(dice_orderly.summary())

# %% [markdown]
# ## 2. Función de Dinámica Fiscal y Deuda Soberana

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
# ## 3. Trayectoria de Deuda/PIB y Tasas Soberanas

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(fiscal_unabated.index, fiscal_unabated["debt_to_gdp"], color="#d62728", lw=2, label="Sin Mitigación (Daños Elevados)")
ax1.plot(fiscal_late.index, fiscal_late["debt_to_gdp"], color="#ff7f0e", lw=2, linestyle="--", label="Transición Tardía Desordenada")
ax1.plot(fiscal_orderly.index, fiscal_orderly["debt_to_gdp"], color="#2ca02c", lw=2, label="Regla Fiscal Verde Ordenada")
ax1.axhline(60, color="gray", linestyle=":", label="Umbral de Estabilidad 60%")
ax1.set_title("Trayectoria de Deuda Soberana / PIB (%)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Año")
ax1.set_ylabel("Deuda Pública (% PIB)")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

ax2.plot(fiscal_unabated.index, fiscal_unabated["sovereign_rate"], color="#d62728", lw=2, label="Riesgo Sin Mitigación")
ax2.plot(fiscal_late.index, fiscal_late["sovereign_rate"], color="#ff7f0e", lw=2, linestyle="--", label="Riesgo Transición Tardía")
ax2.plot(fiscal_orderly.index, fiscal_orderly["sovereign_rate"], color="#2ca02c", lw=2, label="Transición Ordenada")
ax2.set_title("Tasa de Endeudamiento Soberano (r* + Spread)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Año")
ax2.set_ylabel("Tasa Real (%)")
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Costos de Adaptación y Calentamiento Global

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(fiscal_unabated.index, fiscal_unabated["adaptation_cost"], color="#d62728", lw=2, label="Adaptación Sin Mitigación")
ax1.plot(fiscal_orderly.index, fiscal_orderly["adaptation_cost"], color="#2ca02c", lw=2, label="Adaptación en Transición Ordenada")
ax1.set_title("Gasto Público de Adaptación (% PIB)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Año")
ax1.set_ylabel("Gasto (% PIB)")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

ax2.plot(fiscal_unabated.index, fiscal_unabated["temperature_anomaly"], color="#d62728", lw=2, label="Calentamiento Sin Mitigación")
ax2.plot(fiscal_orderly.index, fiscal_orderly["temperature_anomaly"], color="#2ca02c", lw=2, label="Mitigación Ordenada")
ax2.axhline(1.5, color="gray", linestyle=":", label="1.5°C Ambición París")
ax2.axhline(2.0, color="gray", linestyle="-.", label="2.0°C Guardarraíl")
ax2.set_title("Anomalía de Temperatura Superficial (°C)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Año")
ax2.set_ylabel("Anomalía (°C)")
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
