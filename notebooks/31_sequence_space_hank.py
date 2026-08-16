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
# # Sequence-Space HANK — Heterogeneous Agents without State Explosion
#
# **How do monetary policy shocks propagate through an economy where households face uninsurable idiosyncratic income risk and borrowing constraints?**
#
# Traditional Representative Agent New Keynesian (RANK) models compress the entire household sector into a single Euler equation. However, empirical microdata shows vast heterogeneity in Marginal Propensities to Consume (MPCs): hand-to-mouth households consume virtually all transitory income gains, while wealthy households smooth consumption.
#
# Adrien Auclert, Bence Bardóczy, Matthew Rognlie, and Ludwig Straub (2021, *Econometrica*) introduced the **Sequence-Space Jacobian (SSJ)** method. It calculates the high-dimensional linear response of heterogeneous-agent blocks in $O(T^3)$ time, completely avoiding the curse of dimensionality.
#
# In this interactive showcase, we solve a **HANK model in sequence space** using `puremacro.models.solve_hank_sequence_space`.

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
# ## 1. Solving the Steady State & Endogenous Grid Method (EGM)
#
# The household problem features idiosyncratic labor income $s_t \in \{s_L, s_H\}$ and borrowing constraint $a_{t+1} \ge 0$:
# $$ c_t + a_{t+1} = (1 + r) a_t + w s_t $$
# The Endogenous Grid Method solves the stationary Euler equation, yielding policy functions $c(a, s)$ and the invariant wealth distribution $D(a, s)$.

# %%
res = solve_hank_sequence_space(
    T=40,
    beta=0.985,
    gamma=1.0,
    r_ss=0.01,
    phi_pi=1.5,
    kappa=0.1,
    shock_magnitude=0.0025,  # 25 bps monetary hike
    shock_rho=0.7,
    n_a=60,
)
print(res.summary())

# %% [markdown]
# ## 2. Marginal Propensity to Consume (MPC) Distribution
#
# Wealth-poor households at the borrowing constraint have MPCs exceeding 60%, creating powerful aggregate demand feedback channels absent in RANK models.

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
res.mpc_distribution.plot(kind="bar", ax=ax, color="#2ca02c", edgecolor="#333", alpha=0.85)
ax.set_title("Marginal Propensity to Consume (MPC) by Wealth Decile", fontsize=12, fontweight="bold")
ax.set_ylabel("Quarterly MPC")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Sequence-Space Consumption Jacobians $\mathcal{J}_{C, r}$ and $\mathcal{J}_{C, Y}$
#
# The sequence Jacobian $\mathcal{J}_{C, Y}[t, s]$ measures the response of aggregate consumption at horizon $t$ to an anticipated income shock at horizon $s$.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(res.jacobian_c_y[:15, :15], cmap="Blues", origin="upper")
ax1.set_title(r"Income Jacobian $\mathcal{J}_{C, Y}$", fontsize=11, fontweight="bold")
ax1.set_xlabel("Shock Period s")
ax1.set_ylabel("Response Period t")
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

im2 = ax2.imshow(res.jacobian_c_r[:15, :15], cmap="Reds_r", origin="upper")
ax2.set_title(r"Interest Rate Jacobian $\mathcal{J}_{C, r}$", fontsize=11, fontweight="bold")
ax2.set_xlabel("Shock Period s")
ax2.set_ylabel("Response Period t")
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. General Equilibrium Impulse Responses to a 25 bps Rate Hike
#
# Using the sequence linear system $(I - \mathcal{J}_{C,Y} - \mathcal{J}_{C,r} \mathbf{M}_{r,Y}) d\mathbf{Y} = \mathcal{J}_{C,r} d\mathbf{\epsilon}$, the general equilibrium transition is solved instantly in pure NumPy.

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
h = np.arange(len(res.irf_output))
ax.plot(h, res.irf_output * 100, color="#d62728", lw=2, label="Output dY (%)")
ax.plot(h, res.irf_consumption * 100, color="#ff7f0e", lw=2, linestyle="--", label="Consumption dC (%)")
ax.plot(h, res.irf_inflation * 100, color="#1f77b4", lw=2, linestyle=":", label="Inflation dpi (%)")
ax.axhline(0, color="black", lw=0.8, linestyle="--")
ax.set_title("HANK General Equilibrium Impulse Responses to 25 bps Monetary Tightening", fontsize=12, fontweight="bold")
ax.set_xlabel("Horizon (Quarters)")
ax.set_ylabel("Percentage Deviation")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
