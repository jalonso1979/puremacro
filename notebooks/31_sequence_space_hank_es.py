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
# # Modelos HANK en Espacio de Secuencias — Heterogeneidad sin Explosión de Estados
#
# **¿Cómo se transmiten los choques de política monetaria en una economía donde los hogares enfrentan riesgo de ingreso no asegurable y restricciones de endeudamiento?**
#
# Los modelos tradicionales de Agente Representativo Nuevo Keynesiano (RANK) comprimen a los hogares en una única ecuación de Euler. Sin embargo, los microdatos empíricos revelan una profunda heterogeneidad en la Propensión Marginal a Consumir (PMC): los hogares con restricciones de liquidez consumen casi la totalidad de cualquier ingreso transitorio, mientras que los hogares ricos suavizan su consumo intertemporal.
#
# Adrien Auclert, Bence Bardóczy, Matthew Rognlie y Ludwig Straub (2021, *Econometrica*) introdujeron el método de **Jacobianos en Espacio de Secuencias (SSJ)**. Permite resolver equilibrios generales heterogéneos en tiempo $O(T^3)$ sin sufrir la maldición de la dimensionalidad.
#
# En este tutorial interactivo, resolvemos un **modelo HANK completo** mediante `puremacro.models.solve_hank_sequence_space`.

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

from puremacro.models import solve_hank_sequence_space

# %% [markdown]
# ## 1. Estado Estacionario y Método de Cuadrícula Endógena (EGM)

# %%
res = solve_hank_sequence_space(
    T=40,
    beta=0.985,
    gamma=1.0,
    r_ss=0.01,
    phi_pi=1.5,
    kappa=0.1,
    shock_magnitude=0.0025,
    shock_rho=0.7,
    n_a=60,
)
print(res.summary())

# %% [markdown]
# ## 2. Distribución de la Propensión Marginal a Consumir (PMC) por Decil

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
res.mpc_distribution.plot(kind="bar", ax=ax, color="#2ca02c", edgecolor="#333", alpha=0.85)
ax.set_title("Propensión Marginal a Consumir (PMC) por Decil de Riqueza", fontsize=12, fontweight="bold")
ax.set_ylabel("PMC Trimestral")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Jacobianos de Consumo en Espacio de Secuencias

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(res.jacobian_c_y[:15, :15], cmap="Blues", origin="upper")
ax1.set_title(r"Jacobiano del Ingreso $\mathcal{J}_{C, Y}$", fontsize=11, fontweight="bold")
ax1.set_xlabel("Período del Choque s")
ax1.set_ylabel("Período de Respuesta t")
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

im2 = ax2.imshow(res.jacobian_c_r[:15, :15], cmap="Reds_r", origin="upper")
ax2.set_title(r"Jacobiano de la Tasa de Interés $\mathcal{J}_{C, r}$", fontsize=11, fontweight="bold")
ax2.set_xlabel("Período del Choque s")
ax2.set_ylabel("Período de Respuesta t")
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Respuestas al Impulso en Equilibrio General (Alza de 25 pb)

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
h = np.arange(len(res.irf_output))
ax.plot(h, res.irf_output * 100, color="#d62728", lw=2, label="Producto dY (%)")
ax.plot(h, res.irf_consumption * 100, color="#ff7f0e", lw=2, linestyle="--", label="Consumo dC (%)")
ax.plot(h, res.irf_inflation * 100, color="#1f77b4", lw=2, linestyle=":", label="Inflación dpi (%)")
ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("Respuestas en Equilibrio General HANK ante Alza de 25 pb", fontsize=12, fontweight="bold")
ax.set_xlabel("Horizonte (Trimestres)")
ax.set_ylabel("Desviación Porcentual")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
