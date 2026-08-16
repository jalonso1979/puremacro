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
# # High-Dimensional Penalized Macroeconomic Forecasting — Elastic Net & Adaptive Lasso
#
# **How can econometricians extract sparse, highly predictive signals from dozens or hundreds of macroeconomic indicators without overfitting?**
#
# In modern data-rich environments, the number of candidate predictors $P$ often approaches or exceeds the time sample $T$. Standard OLS estimates suffer from variance inflation and severe out-of-sample forecast deterioration.
#
# Regularisation methods solve this by adding sparsity and shrinkage penalties:
# - **Elastic Net** (Zou & Hastie 2005): Blends $L_1$ (Lasso) and $L_2$ (Ridge) penalties to handle correlated predictor clusters.
# - **Adaptive Lasso** (Zou 2006, *JASA*): Employs data-driven weights $w_j = 1/|\hat{\beta}_{j}|^\gamma$ to achieve the asymptotic **oracle property** (simultaneous variable selection consistency and asymptotic normality).
#
# In this interactive showcase, we forecast macroeconomic variables from large panels using `puremacro.forecast.forecast_penalized`.

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
# ## 1. Simulating a High-Dimensional Panel (P = 30 Predictors)

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

# Target variable driven by 4 key predictors
y = np.zeros(T)
active_indices = [1, 5, 12, 22]
weights = [1.8, -1.4, 1.2, -0.9]
for t in range(1, T):
    signal = sum(w * X[t-1, idx] for w, idx in zip(weights, active_indices))
    y[t] = 2.0 + signal + rng.normal(scale=0.5)
y[0] = 2.0

df_X = pd.DataFrame(X, index=dates, columns=[f"Macro_Indicator_{j+1:02d}" for j in range(P)])
s_y = pd.Series(y, index=dates, name="CPI Inflation")

# %% [markdown]
# ## 2. Estimating Elastic Net and Adaptive Lasso Forecasts

# %%
res_enet = forecast_penalized(df_X, s_y, horizon=1, alpha=0.5, adaptive=False)
print("=== Elastic Net ===")
print(res_enet.summary())

res_alasso = forecast_penalized(df_X, s_y, horizon=1, alpha=1.0, adaptive=True)
print("\n=== Adaptive Lasso ===")
print(res_alasso.summary())

# %% [markdown]
# ## 3. Actual vs. Fitted Values and Regularisation Paths

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

fitted_vals = res_alasso.intercept + df_X.iloc[:-1].to_numpy() @ res_alasso.coefficients.to_numpy()
ax1.plot(dates[1:], s_y.iloc[1:], color="black", lw=1.5, label="Actual Inflation")
ax1.plot(dates[1:], fitted_vals, color="#1f77b4", lw=2, linestyle="--", label=f"Adaptive Lasso Fit (R²={res_alasso.in_sample_r2:.2f})")
ax1.set_title("Actual vs. Penalized Model Fitted Path", fontsize=11, fontweight="bold")
ax1.set_xlabel("Date")
ax1.set_ylabel("Inflation Rate (%)")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

ax2.plot(np.log10(res_alasso.bic_path.index), res_alasso.bic_path.values, color="#d62728", lw=2, marker="o", markersize=3)
ax2.axvline(np.log10(res_alasso.optimal_lambda), color="black", linestyle="--", label=f"Optimal λ* = {res_alasso.optimal_lambda:.4f}")
ax2.set_title("BIC Regularisation Path Across Candidate Penalties", fontsize=11, fontweight="bold")
ax2.set_xlabel("log10(λ)")
ax2.set_ylabel("BIC Score")
ax2.legend()
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Sparsity Comparison: Elastic Net vs. Adaptive Lasso

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
df_comp = pd.DataFrame({
    "Elastic Net (α=0.5)": res_enet.coefficients,
    "Adaptive Lasso (α=1.0)": res_alasso.coefficients,
})
top_feats = df_comp.loc[(df_comp.abs() > 0.05).any(axis=1)]
top_feats.plot(kind="bar", ax=ax, edgecolor="#333", alpha=0.85)
ax.set_title("Coefficient Selection & Shrinkage Comparison", fontsize=12, fontweight="bold")
ax.set_ylabel("Estimated Coefficient")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
