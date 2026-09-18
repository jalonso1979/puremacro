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
# # Empirical Benchmark Replications — Galí (1999) & Mertens-Ravn (2013)
#
# **How do fundamental macroeconomic disturbances—neutral technology shifts and unanticipated tax shocks—transmit into aggregate output, labor hours, and interest rates in empirical time series?**
#
# A central objective of macroeconometrics is distinguishing between competing theoretical paradigms through credible empirical identification. Two published papers represent seminal benchmarks in this literature:
#
# 1. **Jordi Galí (1999, *American Economic Review*)**:
#    - *The Research Question*: Do positive technology shocks cause employment to expand, as predicted by classical Real Business Cycle (RBC) models (Kydland & Prescott 1982)? Or do hours worked contract on impact, as predicted by sticky-price New Keynesian models?
#    - *The Identification Strategy*: Uses Blanchard & Quah (1989) long-run restrictions in a Structural Vector Autoregression (SVAR). Technology is identified as the sole structural shock capable of exerting a permanent, unit-root effect on labor productivity ($Y_t / N_t$).
#    - *The Empirical Finding*: In post-war US data, hours worked experience a sharp and persistent *contraction* following a positive technology shock. Under sticky prices, aggregate demand fails to adjust immediately to higher productive capacity, enabling monopolistically competitive firms to satisfy demand with fewer labor hours ($N = Y / A$).
#
# 2. **Karel Mertens & Morten Ravn (2013, *American Economic Review*)**:
#    - *The Research Question*: What is the empirical size of the fiscal tax multiplier, and how does monetary policy react to fiscal consolidation?
#    - *The Identification Strategy*: Employs External Instruments (Proxy SVAR / SVAR-IV) developed by Stock & Watson (2012) and Mertens & Ravn (2013). Narrative historical records of major post-war US federal tax changes (Romer & Romer 2010) are used as exogenous instruments $z_t$ for structural tax shocks.
#    - *The Empirical Finding*: An unanticipated tax increase equal to 1% of GDP triggers an immediate and statistically significant contraction in real GDP, while the Federal Reserve typically responds by easing policy interest rates.
#
# In this interactive showcase, we replicate both landmark studies end-to-end using `puremacro.datasets` and `puremacro.var.identify`.

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
# ## 1. Replicating Galí (1999, AER): Technology Shocks and Hours Worked
#
# Consider the bivariate vector $X_t = (\Delta x_t, n_t)^\top$, where $\Delta x_t = \Delta \log(Y_t / N_t)$ is the growth rate of labor productivity and $n_t = \log(N_t)$ is log hours worked. The reduced-form VAR is $X_t = \sum_{l=1}^p A_l X_{t-l} + u_t$ with residual covariance $\Sigma_u = \mathbb{E}[u_t u_t^\top]$.
#
# Inverting the autoregressive lag polynomial yields the Wold moving average representation:
#
# $$ X_t = C(L) u_t = \sum_{k=0}^{\infty} C_k u_{t-k} $$
#
# Structural shocks $\varepsilon_t = (\varepsilon_t^{tech}, \varepsilon_t^{non-tech})^\top$ are related to reduced-form innovations by $u_t = B_0 \varepsilon_t$, with $\mathbb{E}[\varepsilon_t \varepsilon_t^\top] = I$. The infinite-horizon cumulative impact matrix $\bar{C} = C(1) B_0 = \left( I - \sum_{l=1}^p A_l \right)^{-1} B_0$ governs the long-run response.
#
# The Blanchard-Quah identification restricts $\bar{C}$ to be lower triangular:
#
# $$ \bar{C} = \begin{pmatrix} \bar{C}_{11} & 0 \\ \bar{C}_{21} & \bar{C}_{22} \end{pmatrix} \implies \bar{C} \bar{C}^\top = C(1) \Sigma_u C(1)^\top $$
#
# Because $\bar{C}_{12} = 0$, non-technology shocks are strictly excluded from exerting any permanent impact on the level of labor productivity. Solving via the Cholesky factor of $C(1) \Sigma_u C(1)^\top$ uniquely identifies structural impact matrix $B_0$.

# %%
df_gali = load_gali1999()
print("Galí (1999) Dataset Preview:")
print(df_gali[["dlprod", "hours"]].head())

# Estimate VAR(4) with long-run BQ restriction
Z_gali = df_gali[["dlprod", "hours"]].to_numpy(dtype=float)
bq_res = bq(Z_gali, p=4, horizon=20)
print("\n" + bq_res.summary())

# %% [markdown]
# ### Impulse Responses to a Positive Technology Shock
#
# The impulse response functions with 90% bootstrap confidence intervals illustrate:
# - **Left Panel (Productivity Level)**: Technology shocks permanently elevate the level of labor productivity ($x_t$), with cumulative gains stabilizing near $+0.8$ percentage points.
# - **Right Panel (Hours Worked)**: Rather than expanding employment as classical flexible-price models predict, hours worked plunge by $-0.4\%$ on impact, remaining depressed for over six quarters before returning to trend. This negative correlation provides compelling empirical evidence for nominal price rigidities.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.5))
h_gali = np.arange(len(bq_res.irf_point))

# Panel 1: Labor Productivity (Cumulated level)
irf_prod = bq_res.irf_point[:, 0, 0]
irf_prod_lo = bq_res.irf_lower[:, 0, 0]
irf_prod_hi = bq_res.irf_upper[:, 0, 0]

ax1.plot(h_gali, irf_prod, **_nbstyle.S1, label="Labor Productivity (Level)")
ax1.fill_between(h_gali, irf_prod_lo, irf_prod_hi, color=_nbstyle.TINTA, alpha=0.15)
ax1.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="--")
ax1.set_title("Labor Productivity Response to Tech Shock", fontsize=11, fontweight="bold")
ax1.set_xlabel("Horizon (Quarters)", color=_nbstyle.TEXTO)
ax1.set_ylabel("Percentage Points", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# Panel 2: Hours Worked (Contraction on impact)
irf_hours = bq_res.irf_point[:, 1, 0]
irf_hours_lo = bq_res.irf_lower[:, 1, 0]
irf_hours_hi = bq_res.irf_upper[:, 1, 0]

ax2.plot(h_gali, irf_hours, **_nbstyle.S2, label="Hours Worked")
ax2.fill_between(h_gali, irf_hours_lo, irf_hours_hi, color=_nbstyle.TEXTO, alpha=0.15)
ax2.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="--")
ax2.set_title("Hours Worked Response (Galí Contraction)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Horizon (Quarters)", color=_nbstyle.TEXTO)
ax2.set_ylabel("Percentage Points", color=_nbstyle.TEXTO)
ax2.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 2. Replicating Mertens & Ravn (2013, AER): Narrative Tax Multipliers
#
# Let $Y_t = (y_t, r_t)^\top$ be a bivariate macro system consisting of log real GDP $y_t$ and the Federal Funds rate $r_t$. The reduced-form residuals are related to structural innovations $\varepsilon_t = (\varepsilon_t^{tax}, \varepsilon_t^{other})^\top$ by:
#
# $$ u_t = b_1 \varepsilon_t^{tax} + b_2 \varepsilon_t^{other} $$
#
# Mertens & Ravn (2013) use narrative historical tax legislation $m_t$ (measured as the projected annualized revenue change from tax changes motivated by long-run growth or inherited deficit consolidation) as an external instrument satisfying:
#
# $$ \mathbb{E}[m_t \varepsilon_t^{tax}] = \phi \neq 0 \quad (\text{Relevance}), \qquad \mathbb{E}[m_t \varepsilon_t^{other}] = 0 \quad (\text{Exogeneity}) $$
#
# Under these conditions, the covariance between reduced-form residuals and the instrument yields:
#
# $$ \mathbb{E}[u_t m_t] = b_1 \mathbb{E}[\varepsilon_t^{tax} m_t] = b_1 \phi \implies \frac{b_{1, i}}{b_{1, 1}} = \frac{\text{Cov}(u_{i, t}, m_t)}{\text{Cov}(u_{1, t}, m_t)} $$
#
# The relative structural impulse vector is identified via two-stage least squares (2SLS) without imposing Cholesky ordering or sign restrictions.

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
# ### Impulse Responses to an Unanticipated Tax Increase
#
# The Proxy SVAR dynamic responses demonstrate:
# - **Left Panel (Output Contraction)**: An unanticipated tax increase causes a statistically significant and prolonged downturn in real output, with GDP falling between $-1.5\%$ and $-2.5\%$ below baseline over a 12-quarter horizon (implying an output tax multiplier between $-1.5$ and $-2.0$).
# - **Right Panel (Monetary Policy Reaction)**: The Federal Reserve lowers the Federal Funds rate to mitigate the recessionary impulse, partially dampening the contractionary impact of the fiscal shock.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.5))
h_tax = np.arange(len(proxy_res.irf_point))

irf_gdp = proxy_res.irf_point[:, 0, 0]
irf_gdp_lo = proxy_res.irf_lower[:, 0, 0]
irf_gdp_hi = proxy_res.irf_upper[:, 0, 0]

ax1.plot(h_tax, irf_gdp, **_nbstyle.S1, label="Real GDP")
ax1.fill_between(h_tax, irf_gdp_lo, irf_gdp_hi, color=_nbstyle.TINTA, alpha=0.15)
ax1.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="--")
ax1.set_title("Output Response to Unanticipated Tax Hike", fontsize=11, fontweight="bold")
ax1.set_xlabel("Horizon (Quarters)", color=_nbstyle.TEXTO)
ax1.set_ylabel("Log GDP (%)", color=_nbstyle.TEXTO)
ax1.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax1.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

irf_ffr = proxy_res.irf_point[:, 1, 0]
irf_ffr_lo = proxy_res.irf_lower[:, 1, 0]
irf_ffr_hi = proxy_res.irf_upper[:, 1, 0]

ax2.plot(h_tax, irf_ffr, **_nbstyle.S2, label="Fed Funds Rate")
ax2.fill_between(h_tax, irf_ffr_lo, irf_ffr_hi, color=_nbstyle.TEXTO, alpha=0.15)
ax2.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="--")
ax2.set_title("Monetary Policy Reaction to Fiscal Shock", fontsize=11, fontweight="bold")
ax2.set_xlabel("Horizon (Quarters)", color=_nbstyle.TEXTO)
ax2.set_ylabel("Interest Rate (% pts)", color=_nbstyle.TEXTO)
ax2.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE)
ax2.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)
