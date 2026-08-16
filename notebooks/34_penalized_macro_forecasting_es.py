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
# # Pronóstico Macroeconómico Penalizado de Alta Dimensión — Elastic Net y Lasso Adaptativo
#
# **¿Cómo pueden los econometristas extraer señales predictivas parsimoniosas de docenas o cientos de indicadores macroeconómicos sin sobreajuste?**
#
# Cuando el número de predictores $P$ se aproxima o supera el número de períodos $T$, las estimaciones MCO sufren de inflación de varianza y deterioro fuera de muestra.
#
# Los métodos de regularización resuelven esto:
# - **Elastic Net** (Zou & Hastie 2005): Combina penalizaciones $L_1$ (Lasso) y $L_2$ (Ridge).
# - **Lasso Adaptativo** (Zou 2006, *JASA*): Ponderaciones $w_j = 1/|\hat{\beta}_{j}|^\gamma$ para alcanzar la **propiedad oráculo**.
#
# En este tutorial interactivo, pronosticamos variables macroeconómicas mediante `puremacro.forecast.forecast_penalized`.

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

from puremacro.forecast import forecast_penalized

# %% [markdown]
# ## 1. Simulación de Panel de Alta Dimensión (P = 30 Predictores)

# %%
rng = np.random.default_rng(123)
T = 160
P = 30
dates = pd.date_range("2010-01-01", periods=T, freq="MS")

X = np.zeros((T, P))
for j in range(P):
    rho = rng.uniform(0.3, 0.8)
    for t in range(1, T):
        X[t, j] = rho * X[t-1, j] + rng.normal(scale=0.8)

y = np.zeros(T)
active_indices = [1, 5, 12, 22]
weights = [1.8, -1.4, 1.2, -0.9]
for t in range(1, T):
    signal = sum(w * X[t-1, idx] for w, idx in zip(weights, active_indices))
    y[t] = 2.0 + signal + rng.normal(scale=0.5)
y[0] = 2.0

df_X = pd.DataFrame(X, index=dates, columns=[f"Macro_Indicator_{j+1:02d}" for j in range(P)])
s_y = pd.Series(y, index=dates, name="Inflación")

# %% [markdown]
# ## 2. Estimación de Pronósticos con Elastic Net y Lasso Adaptativo

# %%
res_enet = forecast_penalized(df_X, s_y, horizon=1, alpha=0.5, adaptive=False)
print("=== Elastic Net ===")
print(res_enet.summary())

res_alasso = forecast_penalized(df_X, s_y, horizon=1, alpha=1.0, adaptive=True)
print("\n=== Lasso Adaptativo ===")
print(res_alasso.summary())

# %% [markdown]
# ## 3. Ajuste Muestral y Trayectorias de Regularización BIC

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

fitted_vals = res_alasso.intercept + df_X.iloc[:-1].to_numpy() @ res_alasso.coefficients.to_numpy()
ax1.plot(dates[1:], s_y.iloc[1:], color="black", lw=1.5, label="Inflación Observada")
ax1.plot(dates[1:], fitted_vals, color="#1f77b4", lw=2, linestyle="--", label=f"Ajuste Lasso Adaptativo (R²={res_alasso.in_sample_r2:.2f})")
ax1.set_title("Inflación Observada vs. Ajuste del Modelo", fontsize=11, fontweight="bold")
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Tasa de Inflación (%)")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

ax2.plot(np.log10(res_alasso.bic_path.index), res_alasso.bic_path.values, color="#d62728", lw=2, marker="o", markersize=3)
ax2.axvline(np.log10(res_alasso.optimal_lambda), color="black", linestyle="--", label=f"Óptimo λ* = {res_alasso.optimal_lambda:.4f}")
ax2.set_title("Trayectoria BIC a Través de las Penalizaciones Candidatas", fontsize=11, fontweight="bold")
ax2.set_xlabel("log10(λ)")
ax2.set_ylabel("Puntaje BIC")
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Comparación de Dispersión (Sparsity)

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
df_comp = pd.DataFrame({
    "Elastic Net (α=0.5)": res_enet.coefficients,
    "Lasso Adaptativo (α=1.0)": res_alasso.coefficients,
})
top_feats = df_comp.loc[(df_comp.abs() > 0.05).any(axis=1)]
top_feats.plot(kind="bar", ax=ax, edgecolor="#333", alpha=0.85)
ax.set_title("Comparación de Selección y Contracción de Coeficientes", fontsize=12, fontweight="bold")
ax.set_ylabel("Coeficiente Estimado")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
