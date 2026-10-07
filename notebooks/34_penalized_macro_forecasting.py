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
# **How can econometricians extract sparse, highly predictive signals from dozens or hundreds of macroeconomic indicators without overfitting or suffering from multicollinearity?**
#
# In modern empirical macroeconomics, forecasting key policy targets (such as core inflation, industrial production, or employment) routinely involves large panels of potential predictors: yield spreads, commodity prices, foreign exchange rates, real activity surveys, and credit aggregates. When the number of candidate variables $P$ is large relative to the effective time-series sample size $T$, standard Ordinary Least Squares (OLS) estimation fails catastrophically: parameter variances explode, in-sample fitting overfits idiosyncratic noise, and out-of-sample forecast accuracy collapses.
#
# Regularized estimation solves this high-dimensional dilemma by shrinking non-essential parameters toward zero through structured loss penalties:
# 1. **Ridge Regression** ($L_2$ penalty, Hoerl & Kennard 1970): Shrinks coefficients proportionally, stabilizing forecasts in the presence of strong collinearity, but retains all $P$ variables without sparsity.
# 2. **Lasso** ($L_1$ penalty, Tibshirani 1996): Imposes sharp corner solutions, driving irrelevant coefficients identically to zero to perform automated variable selection. However, when predictors are highly correlated, Lasso selects one variable arbitrarily and discards the rest.
# 3. **Elastic Net** (Zou & Hastie 2005): Concurrently combines $L_1$ sparsity and $L_2$ group shrinkage, selecting groups of correlated indicators together.
# 4. **Adaptive Lasso** (Hui Zou 2006, *JASA*): Employs predictor-specific weights $w_j = 1/|\hat{\beta}_{j, init}|^\gamma$. Under suitable design, initial-estimator and tuning conditions it has an **oracle property**: asymptotic variable selection and inference behave as if the active set were known. This is not a guarantee of exact selection or unbiased coefficients in a finite sample.
#
# We simulate $P=30$ serially persistent indicators with independent innovations over
# $T=160$ periods. Four indicators drive inflation. We fit Elastic Net and Adaptive Lasso
# using `puremacro.forecast.forecast_penalized` on the first 120 periods, then evaluate
# one-month-ahead predictions on the last 40 periods.

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
#
# Consider the predictive regression for target variable $y_{t+h}$ at horizon $h$:
#
# $$ y_{t+h} = \mu + \sum_{j=1}^P \beta_j X_{j, t} + \varepsilon_{t+h} $$
#
# The Elastic Net optimization objective minimizes the penalized sum of squared residuals:
#
# $$ \min_{\mu, \beta} \frac{1}{2T} \sum_{t=1}^T \left( y_{t+h} - \mu - X_t \beta \right)^2 + \lambda \left[ \alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2 \right] $$
#
# where $\alpha \in [0, 1]$ balances the $L_1$ Lasso penalty ($\alpha = 1$) against the $L_2$ Ridge penalty ($\alpha = 0$), and $\lambda > 0$ governs the overall regularization intensity.
#
# For the Adaptive Lasso ($\alpha = 1$), the penalty is weighted:
#
# $$ \min_{\mu, \beta} \frac{1}{2T} \sum_{t=1}^T \left( y_{t+h} - \mu - X_t \beta \right)^2 + \lambda \sum_{j=1}^P w_j |\beta_j| $$
#
# The implementation uses $w_j \propto (|\hat{\beta}_{j,\text{Ridge}}|+10^{-3})^{-1}$,
# normalized by their median. Predictors with small preliminary Ridge coefficients
# receive larger penalties.
#
# **Intuition.** Shrinkage trades some bias for lower estimation variance. Adaptive weights
# penalize weak preliminary signals more heavily. A good fit or sparse model alone does not
# establish forecasting skill: the held-out observations below test that separately.
#
# The four active Python column indices are `[1, 5, 12, 22]` (zero-based), corresponding
# to indicator labels 02, 06, 13 and 23, with coefficients `[1.8, -1.4, 1.2, -0.9]`.
# The remaining 26 indicators have zero coefficients in the population.

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
#
# `puremacro.forecast.forecast_penalized` solves the coordinate descent path over a geometric grid of penalty parameters $\lambda \in [\lambda_{\min}, \lambda_{\max}]$. For each candidate $\lambda$, the optimal model complexity is selected via the Bayesian Information Criterion (BIC):
#
# $$ \text{BIC}(\lambda) = T \log\left( \frac{\text{SSR}(\lambda)}{T} \right) + \text{df}(\lambda) \log(T) $$
#
# Here $T$ is the number of aligned training pairs and $\text{df}(\lambda)$ counts
# active coefficients plus the intercept. BIC selects a penalty within the supplied grid;
# sparsity alone does not guarantee consistent selection. Standardization, adaptive weights
# and BIC all use the training sample only.

# %%
TRAIN_END = 120
X_train, y_train = df_X.iloc[:TRAIN_END], s_y.iloc[:TRAIN_END]
assert X_train.index.equals(y_train.index)  # this API aligns inputs by position
res_enet = forecast_penalized(X_train, y_train, horizon=1, alpha=0.5, adaptive=False)
print("=== Elastic Net ===")
print(res_enet.summary())

res_alasso = forecast_penalized(X_train, y_train, horizon=1, alpha=1.0, adaptive=True)
print("\n=== Adaptive Lasso ===")
print(res_alasso.summary())

# %% [markdown]
# ## 3. Actual vs. Fitted Values and Regularisation Paths
#
# The plots below compare:
# - **Left panel**: Observed inflation and fitted values on the training sample. The
# printed $R^2$ measures in-sample fit; these are not held-out forecasts.
# - **Right panel**: Training-sample BIC over the penalty grid. The selected minimum
# balances residual fit and model size; the path need not be a smooth U shape.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.5))

fitted_vals = res_alasso.intercept + X_train.iloc[:-1].to_numpy() @ res_alasso.coefficients.to_numpy()
ax1.plot(dates[1:TRAIN_END], y_train.iloc[1:], **_nbstyle.S1, label="Actual Inflation")
ax1.plot(dates[1:TRAIN_END], fitted_vals, **_nbstyle.S2, label=f"Adaptive Lasso Fit (R²={res_alasso.in_sample_r2:.2f})")
ax1.set_title("Actual vs. Penalized Model Fitted Path", fontsize=11, fontweight="bold")
ax1.set_xlabel("Date", color=_nbstyle.TEXTO)
ax1.set_ylabel("Inflation Rate (%)", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

ax2.plot(np.log10(res_alasso.bic_path.index), res_alasso.bic_path.values, **_nbstyle.S1, marker="o", markersize=3)
ax2.axvline(np.log10(res_alasso.optimal_lambda), color=_nbstyle.SPINE, linestyle="--", label=f"Optimal λ* = {res_alasso.optimal_lambda:.4f}")
ax2.set_title("BIC Regularisation Path Across Candidate Penalties", fontsize=11, fontweight="bold")
ax2.set_xlabel(r"$\log_{10}(\lambda)$", color=_nbstyle.TEXTO)
ax2.set_ylabel("BIC Score", color=_nbstyle.TEXTO)
ax2.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 4. Sparsity Comparison: Elastic Net vs. Adaptive Lasso
#
# Compare fitted coefficients with the known simulation truth. The selection table counts
# recovered signals and selected noise predictors using each result's `selected_features`.
# This evaluates one finite sample, not the asymptotic oracle property. The plot shows
# coefficients larger than 0.05 in absolute value in either fit, together with all true signals;
# smaller selected coefficients still count in the table.

# %%
true_coefficients = pd.Series(0.0, index=df_X.columns)
true_coefficients.iloc[active_indices] = weights
true_features = set(true_coefficients[true_coefficients != 0].index)
selection_rows = []
for name, result in [("Elastic Net", res_enet), ("Adaptive Lasso", res_alasso)]:
    selected = set(result.selected_features)
    selection_rows.append({"Model": name, "Signals recovered": len(selected & true_features),
                           "Noise selected": len(selected - true_features)})
print(pd.DataFrame(selection_rows).to_string(index=False))

fig, ax = _nbstyle.figura(figsize=(9.0, 4.5))
df_comp = pd.DataFrame({
    "True coefficient": true_coefficients,
    "Elastic Net (α=0.5)": res_enet.coefficients,
    "Adaptive Lasso (α=1.0)": res_alasso.coefficients,
})
top_feats = df_comp.loc[(df_comp.abs() > 0.05).any(axis=1)]
top_feats.plot(kind="bar", ax=ax, color=_nbstyle.palette(3), edgecolor=_nbstyle.SPINE, alpha=0.85)
ax.set_title("Coefficient Selection & Shrinkage Comparison", fontsize=11, fontweight="bold")
ax.set_ylabel("Estimated Coefficient", color=_nbstyle.TEXTO)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 5. One-month-ahead forecasts on a chronological holdout
#
# Freeze the fitted coefficients at the training cutoff. For each held-out target $y_t$,
# use only $X_{t-1}$, which we assume is already observed. Later test outcomes never enter
# estimation, scaling or BIC selection. These are successive one-step forecasts with fixed
# coefficients, not a 40-step forecast made at the initial cutoff. The benchmark is the
# mean of the training outcomes used in estimation.

# %%
X_test_lagged = df_X.iloc[TRAIN_END - 1:-1]
actual_test = s_y.iloc[TRAIN_END:]
predictions = pd.DataFrame(index=actual_test.index)
for name, result in [("Elastic Net", res_enet), ("Adaptive Lasso", res_alasso)]:
    predictions[name] = result.intercept + X_test_lagged.to_numpy() @ result.coefficients.to_numpy()
    assert np.isclose(predictions[name].iloc[0], result.forecast)
predictions["Training mean"] = y_train.iloc[1:].mean()
assert len(predictions) == T - TRAIN_END and np.isfinite(predictions).all().all()
errors = predictions.sub(actual_test, axis=0)
metrics = pd.DataFrame({"RMSE": np.sqrt(errors.pow(2).mean()), "MAE": errors.abs().mean()})
print(metrics.round(3).to_string())
assert metrics.loc["Adaptive Lasso", "RMSE"] < metrics.loc["Training mean", "RMSE"]

fig, ax = _nbstyle.figura(figsize=(10.0, 4.0))
ax.plot(actual_test.index, actual_test, **_nbstyle.S1, label="Observed holdout")
ax.plot(predictions.index, predictions["Adaptive Lasso"], **_nbstyle.S2, label="Adaptive Lasso forecast")
ax.plot(predictions.index, predictions["Training mean"], **_nbstyle.S3, label="Training-mean benchmark")
ax.set_title("Forecast evaluation: the final 40 months were held out")
ax.set_xlabel("Target month")
ax.set_ylabel("Inflation rate (%)")
ax.legend()

# %% [markdown]
# **Read the output.** RMSE and MAE now describe errors on unseen target observations.
# Compare the two fitted models with the training-mean benchmark, not with training $R^2$.
# The strong planted signals let Adaptive Lasso beat that benchmark for this seed, while
# differences between the two penalized models remain sample-specific. Real-time data
# revisions and publication delays are absent from this simulation.
#
# ## Your turn — change the penalty mix
#
# Keep the chronological split fixed and change the Elastic Net mixing parameter.
# The assertion checks valid held-out predictions; it does not require every choice to win.

# %%
your_alpha = 0.8  # ← change this penalty mix in [0, 1]; 0 = Ridge, 1 = Lasso
your_result = forecast_penalized(X_train, y_train, horizon=1, alpha=your_alpha, adaptive=False)
your_predictions = your_result.intercept + X_test_lagged.to_numpy() @ your_result.coefficients.to_numpy()
your_rmse = float(np.sqrt(np.mean((your_predictions - actual_test.to_numpy()) ** 2)))
assert your_predictions.shape == actual_test.shape and np.isfinite(your_predictions).all()
print(f"alpha={your_alpha:.2f}: holdout RMSE={your_rmse:.3f}, selected={len(your_result.selected_features)}")

# %% [markdown]
# **Prompts.** (1) Try `your_alpha = 0` and `1`; compare selection counts and RMSE.
# (2) Enable adaptive weighting and compare false selections with the known true support.
# (3) Add an expanding-window evaluation that refits using only observations available at
# each forecast origin. If you tune choices using this holdout, reserve another later period
# for the final evaluation.
#
# **How comprehensive is this?** `forecast_penalized` also supports direct horizons and Ridge.
# Notebook 19 compares forecast losses using the model confidence set; Notebook 33 introduces
# ragged-edge nowcasting, where indicator publication timing becomes part of the model.
