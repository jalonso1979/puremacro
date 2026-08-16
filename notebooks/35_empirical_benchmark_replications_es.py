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
# # Replicaciones Empíricas Canónicas — Galí (1999) y Mertens-Ravn (2013)
#
# **¿Cómo afectan los choques macroeconómicos fundamentales (tecnología e impuestos) a la producción, el empleo y la política monetaria en los datos empíricos?**
#
# En este tutorial interactivo, replicamos dos hitos de la identificación estructural mediante `puremacro.datasets`:
#
# 1. **Galí (1999, *AER*) Restricciones de Largo Plazo (Blanchard-Quah)**:
#    - Contrasta la predicción de los modelos de Ciclos Económicos Reales (RBC) de que los choques tecnológicos elevan el empleo.
#    - Identifica la tecnología como el único choque con efecto permanente en la productividad laboral ($Y/N$).
#    - Replica el célebre resultado: las horas trabajadas *caen* al impacto ante un choque tecnológico positivo, respaldando los modelos Nuevo Keynesianos con rigideces de precios.
#
# 2. **Mertens & Ravn (2013, *AER*) Instrumentos Externos (Proxy SVAR)**:
#    - Identifica choques impositivos no anticipados usando pasivos tributarios narrativos como instrumentos externos ($z_t$).
#    - Examina la contracción del PIB real y la función de reacción de la Reserva Federal.

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

from puremacro.datasets import load_gali1999, load_narrative_tax_shocks, load_macro_quarterly
from puremacro.var.identify import bq, proxy

# %% [markdown]
# ## 1. Replicación de Galí (1999, AER): Choques de Tecnología y Horas Trabajadas

# %%
df_gali = load_gali1999()
print("Vista previa de datos de Galí (1999):")
print(df_gali[["dlprod", "hours"]].head())

Z_gali = df_gali[["dlprod", "hours"]].to_numpy(dtype=float)
bq_res = bq(Z_gali, p=4, horizon=20)
print("\n" + bq_res.summary())

# %% [markdown]
# ### Respuestas al Impulso ante un Choque Tecnológico Positivo

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
h_gali = np.arange(len(bq_res.irf_point))

# Panel 1: Productividad Laboral (Nivel acumulado)
irf_prod = bq_res.irf_point[:, 0, 0]
irf_prod_lo = bq_res.irf_lower[:, 0, 0]
irf_prod_hi = bq_res.irf_upper[:, 0, 0]

ax1.plot(h_gali, irf_prod, color="#1f77b4", lw=2, label="Productividad Laboral (Nivel)")
ax1.fill_between(h_gali, irf_prod_lo, irf_prod_hi, color="#1f77b4", alpha=0.2)
ax1.axhline(0, color="black", lw=0.8, linestyle="--")
ax1.set_title("Respuesta de la Productividad Laboral", fontsize=11, fontweight="bold")
ax1.set_xlabel("Horizonte (Trimestres)")
ax1.set_ylabel("Puntos Porcentuales")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

# Panel 2: Horas Trabajadas (Contracción al impacto)
irf_hours = bq_res.irf_point[:, 1, 0]
irf_hours_lo = bq_res.irf_lower[:, 1, 0]
irf_hours_hi = bq_res.irf_upper[:, 1, 0]

ax2.plot(h_gali, irf_hours, color="#d62728", lw=2, label="Horas Trabajadas")
ax2.fill_between(h_gali, irf_hours_lo, irf_hours_hi, color="#d62728", alpha=0.2)
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_title("Respuesta de Horas Trabajadas (Contracción de Galí)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Horizonte (Trimestres)")
ax2.set_ylabel("Puntos Porcentuales")
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Replicación de Mertens & Ravn (2013, AER): Multiplicadores Impositivos Narrativos

# %%
df_macro_q = load_macro_quarterly()
df_tax = load_narrative_tax_shocks()

common_idx = [idx for idx in df_macro_q.index if idx in df_tax.index]
sub_macro = df_macro_q.loc[common_idx]
sub_tax = df_tax.loc[common_idx]

gdp_log = np.log(sub_macro["real_gdp"].to_numpy(dtype=float)) * 100.0
ffr = sub_macro["fed_funds"].to_numpy(dtype=float)
Z_tax = np.column_stack([gdp_log, ffr])
m_instrument = sub_tax["unanticipated"].to_numpy(dtype=float)

proxy_res = proxy(Z_tax, p=4, horizon=16, instrument_series=m_instrument, shock_target_idx=0)
print(proxy_res.summary())

# %% [markdown]
# ### Respuestas al Impulso ante un Alza Impositiva No Anticipada

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
h_tax = np.arange(len(proxy_res.irf_point))

irf_gdp = proxy_res.irf_point[:, 0, 0]
irf_gdp_lo = proxy_res.irf_lower[:, 0, 0]
irf_gdp_hi = proxy_res.irf_upper[:, 0, 0]

ax1.plot(h_tax, irf_gdp, color="#d62728", lw=2, label="PIB Real")
ax1.fill_between(h_tax, irf_gdp_lo, irf_gdp_hi, color="#d62728", alpha=0.2)
ax1.axhline(0, color="black", lw=0.8, linestyle="--")
ax1.set_title("Respuesta del PIB ante Alza Tributaria", fontsize=11, fontweight="bold")
ax1.set_xlabel("Horizonte (Trimestres)")
ax1.set_ylabel("Log PIB (%)")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

irf_ffr = proxy_res.irf_point[:, 1, 0]
irf_ffr_lo = proxy_res.irf_lower[:, 1, 0]
irf_ffr_hi = proxy_res.irf_upper[:, 1, 0]

ax2.plot(h_tax, irf_ffr, color="#2ca02c", lw=2, label="Tasa Fondos Federales")
ax2.fill_between(h_tax, irf_ffr_lo, irf_ffr_hi, color="#2ca02c", alpha=0.2)
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_title("Reacción de la Política Monetaria", fontsize=11, fontweight="bold")
ax2.set_xlabel("Horizonte (Trimestres)")
ax2.set_ylabel("Tasa de Interés (puntos %)")
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
