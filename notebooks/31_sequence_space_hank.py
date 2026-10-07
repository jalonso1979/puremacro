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
# In a Representative Agent New Keynesian (RANK) model, aggregate consumption follows a single Euler equation for an unconstrained saver. Monetary policy then works almost entirely through intertemporal substitution: a higher real rate makes households postpone spending.
#
# With uninsurable income risk and a borrowing limit, households differ in how much of an extra dollar they spend. Those with little wealth spend much of it at once; wealthy households spend little. In a Heterogeneous Agent New Keynesian (HANK) model this dispersion opens a second, indirect channel: tighter policy lowers output and labour income, and households with high marginal propensities to consume (MPCs) cut spending further. How large that channel is depends on who receives the lost income, and this notebook measures it.
#
# Solving such a model used to require tracking the whole wealth distribution as a state variable (Krusell & Smith 1998). Auclert, Bardóczy, Rognlie and Straub (2021, *Econometrica*) instead linearise in *sequence space*: the household block is summarised by Jacobians that map whole paths of income and interest rates into a path of consumption, and the Fake News Algorithm computes each Jacobian from one backward pass of the household problem. General equilibrium then reduces to one $T \times T$ linear system.
#
# We compute the stationary wealth distribution with the Endogenous Grid Method (EGM), extract the consumption Jacobians $\mathcal{J}_{C, Y}$ and $\mathcal{J}_{C, r}$, and solve for the general-equilibrium response to a monetary shock of $0.25$ percentage points on the quarterly policy rate (about one point at an annual rate) with `puremacro.models.solve_hank_sequence_space`. The model is a textbook calibration built into the library, not an estimated economy.

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
# ## 1. Stationary Equilibrium and the Endogenous Grid Method (EGM)
#
# A continuum of infinitely lived households differ in labour productivity $s_t \in \{0.5, 1.5\}$, which follows a two-state Markov chain that stays in its current state with probability $0.9$ each quarter (both are fixed in the library source). Households maximise
#
# $$ \mathbb{E}_0 \sum_{t=0}^{\infty} \beta^t \frac{c_t^{1-\gamma} - 1}{1-\gamma} $$
#
# subject to the budget constraint and a borrowing limit that rules out negative assets:
#
# $$ c_t + a_{t+1} = (1 + r_t) a_t + (1 - \tau_t) w_t s_t, \quad a_{t+1} \ge 0 $$
#
# Households hold government debt $B$; the interest bill $r_t B$ is paid by a proportional labour-income tax $\tau_t$. The first-order condition is the Euler inequality:
#
# $$ u'(c_t) \ge \beta (1 + r_{t+1}) \mathbb{E}_t \left[ u'(c_{t+1}) \right], \quad \text{with equality if } a_{t+1} > 0 $$
#
# To avoid non-linear root-finding, `puremacro` uses Carroll's (2006) **Endogenous Grid Method (EGM)**. On a grid of end-of-period assets $a_{t+1} \in [0, a_{\max}]$, consumption follows from inverting marginal utility:
#
# $$ c_t(a_{t+1}, s_t) = \left( \beta (1 + r_{ss}) \sum_{s'} \Pi(s_t, s') u'\left(c_{t+1}^*(a_{t+1}, s')\right) \right)^{-1/\gamma} $$
#
# Current assets then follow from the budget constraint, $a_t = \frac{c_t + a_{t+1} - (1-\tau) w_{ss} s_t}{1 + r_{ss}}$, and the policy is interpolated back onto the asset grid. The stationary distribution $\mathcal{D}^*(a, s)$ is the invariant vector of the lottery transition matrix of Young (2010), $\mathcal{T}^* \mathcal{D}^* = \mathcal{D}^*$.

# %%
res = solve_hank_sequence_space(
    T=40,
    beta=0.985,
    gamma=1.0,
    r_ss=0.01,               # quarterly real rate (about 4% a year)
    phi_pi=1.5,
    kappa=0.1,
    shock_magnitude=0.0025,  # +0.25 pp on the quarterly policy rate
    shock_rho=0.7,
    n_a=60,
)
print(res.summary())

# %% [markdown]
# ## 2. Marginal Propensity to Consume (MPC) Distribution
#
# The quarterly MPC out of an unexpected, one-off transfer $m$ is
#
# $$ \text{MPC}(a, s) \equiv \lim_{m \to 0} \frac{c(a + m, s) - c(a, s)}{m} = \frac{\partial c(a, s)}{\partial a} \cdot \frac{1}{1 + r_{ss}} $$
#
# A permanent-income consumer spends roughly the annuity value $r/(1+r) \approx 1\%$ of a transfer per quarter. Households at or near the borrowing limit cannot smooth, so they spend much more of it at once.
#
# The cell below prints the MPC by wealth decile and reads the gradient off those numbers. In this calibration the high MPCs are concentrated in the bottom decile only: the next decile already behaves much more like the rest of the distribution.

# %%
mpc = res.mpc_distribution
high = mpc[mpc > 0.5]
print(f"Deciles with a quarterly MPC above 0.5: {', '.join(high.index)} ({', '.join(f'{v:.4f}' for v in high)})")
print(f"Next decile: {mpc.index[len(high)]} ({mpc.iloc[len(high)]:.4f}); top decile: {mpc.iloc[-1]:.4f}; "
      f"aggregate MPC {res.steady_state_mpc:.4f}")

fig, ax = _nbstyle.figura(figsize=(8.5, 4.2))
mpc.plot(kind="bar", ax=ax, color=_nbstyle.TINTA, edgecolor=_nbstyle.SPINE, alpha=0.9)
ax.set_title("Marginal Propensity to Consume (MPC) by Wealth Decile", fontsize=11, fontweight="bold")
ax.set_ylabel("Quarterly MPC", color=_nbstyle.TEXTO)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## 3. Sequence-Space Consumption Jacobians $\mathcal{J}_{C, r}$ and $\mathcal{J}_{C, Y}$
#
# In sequence space, the path of aggregate consumption $\mathbf{C} = (C_0, C_1, \dots, C_{T-1})^\top$ is a function of the paths of aggregate income $\mathbf{Y}$ and of the real interest rate $\mathbf{r}$. Linearising around the stationary equilibrium gives
#
# $$ d\mathbf{C} = \mathcal{J}_{C, Y} \, d\mathbf{Y} + \mathcal{J}_{C, r} \, d\mathbf{r} $$
#
# where $\mathcal{J}_{C, Y}[t, s] = \partial C_t / \partial Y_s$ is the response of consumption at date $t$ to a rise in aggregate income at date $s$, announced at date 0. Income reaches households through the wage.
#
# The Fake News Algorithm computes these $T \times T$ matrices from one backward pass of the household problem per input:
# 1. $\mathcal{J}_{C, Y}$ (left panel) has a strong diagonal: households spend part of an income gain in the quarter it arrives. Above the diagonal ($t < s$), consumption rises a little in anticipation; below it ($t > s$), it stays higher while households spend down the savings they built.
# 2. $\mathcal{J}_{C, r}$ (right panel) captures intertemporal substitution and income effects: a higher expected real rate at date $s$ makes unconstrained households save more before $s$.

# %%
fig, (ax1, ax2) = _nbstyle.figura(1, 2, figsize=(11.0, 4.6))

im1 = ax1.imshow(res.jacobian_c_y[:15, :15], cmap=_nbstyle.CMAP_SEQ, origin="upper")
ax1.set_title(r"Income Jacobian $\mathcal{J}_{C, Y}$", fontsize=11, fontweight="bold")
ax1.set_xlabel("Shock Horizon s (quarters)", color=_nbstyle.TEXTO)
ax1.set_ylabel("Response Horizon t (quarters)", color=_nbstyle.TEXTO)
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.ax.tick_params(colors=_nbstyle.NOTA)

im2 = ax2.imshow(res.jacobian_c_r[:15, :15], cmap=_nbstyle.CMAP_SEQ_R, origin="upper")
ax2.set_title(r"Interest Rate Jacobian $\mathcal{J}_{C, r}$", fontsize=11, fontweight="bold")
ax2.set_xlabel("Shock Horizon s (quarters)", color=_nbstyle.TEXTO)
ax2.set_ylabel("Response Horizon t (quarters)", color=_nbstyle.TEXTO)
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.ax.tick_params(colors=_nbstyle.NOTA)

# %% [markdown]
# ## 4. General Equilibrium Impulse Responses to a Monetary Tightening
#
# The household block is closed with three standard New Keynesian equations in sequence space:
#
# 1. **New Keynesian Phillips Curve** with slope $\kappa$:
#    $$ \mathbf{\pi}_t = \beta \mathbb{E}_t \mathbf{\pi}_{t+1} + \kappa \left( \mathbf{Y}_t - Y_{ss} \right) $$
# 2. **Taylor Rule** with an exogenous policy disturbance $\mathbf{\epsilon}_t = 0.0025 \cdot 0.7^t$:
#    $$ \mathbf{i}_t = r_{ss} + \phi_{\pi} \mathbf{\pi}_t + \mathbf{\epsilon}_t $$
# 3. **Fisher Equation** for the ex-ante real rate: $d\mathbf{r}_t = d\mathbf{i}_t - \mathbb{E}_t d\mathbf{\pi}_{t+1}$.
# 4. **Goods Market Clearing**: $\mathbf{H}(\mathbf{Y}) \equiv \mathbf{C}(\mathbf{Y}, \mathbf{r}(\mathbf{Y})) - \mathbf{Y} = \mathbf{0}$.
#
# Differentiating market clearing gives one linear system in the output path:
#
# $$ \left( \mathbf{I} - \mathcal{J}_{C, Y} - \mathcal{J}_{C, r} \mathbf{M}_{r, Y} \right) d\mathbf{Y} = \mathcal{J}_{C, r} d\mathbf{\epsilon} $$
#
# where $\mathbf{M}_{r, Y} = \partial \mathbf{r} / \partial \mathbf{Y}$ collects the Phillips curve and the Taylor rule. Because $d\mathbf{C} = d\mathbf{Y}$, the consumption response splits exactly into a **direct** (interest-rate) channel $\mathcal{J}_{C, r} \, d\mathbf{r}$ and an **indirect** (income) channel $\mathcal{J}_{C, Y} \, d\mathbf{Y}$. The cell below computes both, so we can see how much of the recession the income channel accounts for in this calibration.

# %%
dY = res.irf_output
dr = res.irf_rate                          # ex-ante real rate path
direct = res.jacobian_c_r @ dr             # interest-rate channel, income held at steady state
indirect = res.jacobian_c_y @ dY           # income channel, rates held at steady state
share_indirect_0 = indirect[0] / dY[0]
share_indirect_2y = indirect[:8].sum() / dY[:8].sum()   # first two years; later sums depend on the horizon T

print(f"Impact (quarter 0), % of steady-state output:")
print(f"  output = consumption : {dY[0] * 100:+.4f}")
print(f"  direct (rate) channel: {direct[0] * 100:+.4f}")
print(f"  indirect (income)    : {indirect[0] * 100:+.4f} ({share_indirect_0:.1%} of the impact)")
print(f"Sum over the first 8 quarters: indirect channel {share_indirect_2y:.1%} of the output loss")
print(f"Impact real rate {dr[0] * 100:+.4f} pp per quarter; impact inflation {res.irf_inflation[0] * 100:+.4f} pp per quarter")
print(f"J_CY[0,0] = {res.jacobian_c_y[0, 0]:.4f} against an aggregate MPC of {res.steady_state_mpc:.4f}")

# Internal accounting checks (both hold by construction of the GE solve, so a failure means a bug)
assert np.allclose(res.irf_consumption, dY, atol=1e-14), "goods market: dC = dY"
assert np.allclose(direct + indirect, dY, atol=1e-12), "dC = J_CY dY + J_Cr dr"
assert dY[0] < 0 and dr[0] > 0 and res.irf_inflation[0] < 0, "a tightening raises the real rate and lowers output and inflation"

fig, ax = _nbstyle.figura(figsize=(8.5, 4.6))
h = np.arange(len(dY))
ax.plot(h, dY * 100, **_nbstyle.S1, label=r"Output = consumption $d\mathbf{Y}$ (% of $Y_{ss}$)")
ax.plot(h, direct * 100, **_nbstyle.S2, label=r"Direct channel $\mathcal{J}_{C,r}\,d\mathbf{r}$ (% of $Y_{ss}$)")
ax.plot(h, indirect * 100, **_nbstyle.S3, label=r"Indirect channel $\mathcal{J}_{C,Y}\,d\mathbf{Y}$ (% of $Y_{ss}$)")
ax.plot(h, res.irf_inflation * 100, **_nbstyle.S4, label=r"Inflation $d\mathbf{\pi}$ (pp per quarter)")
ax.plot(h, dr * 100, **_nbstyle.S5, label=r"Ex-ante real rate $d\mathbf{r}$ (pp per quarter)")
ax.axhline(0, color=_nbstyle.SPINE, lw=0.8, linestyle="--")
ax.set_title("HANK Responses to a +0.25 pp Quarterly Policy-Rate Shock", fontsize=11, fontweight="bold")
ax.set_xlabel("Horizon (Quarters)", color=_nbstyle.TEXTO)
ax.set_ylabel("Percent / percentage points", color=_nbstyle.TEXTO)
ax.legend(frameon=True, facecolor=_nbstyle.FONDO, edgecolor=_nbstyle.SPINE, fontsize=8, loc="lower right")
ax.grid(True, linestyle=":", color=_nbstyle.REJILLA, alpha=0.8)

# %% [markdown]
# ## Read the output
#
# **Read the output.**
#
# 1. **Who spends.** The aggregate quarterly MPC is $0.1022$, about ten times the permanent-income benchmark of $r/(1+r) \approx 1\%$. That average comes from one decile. Only Decile 1 has an MPC above one half ($0.6643$); Decile 2 is already down to $0.0880$, and the top decile spends $0.0216$. The households with high MPCs are those with no wealth at the borrowing limit.
# 2. **Who receives the income.** The first entry of the income Jacobian, $\mathcal{J}_{C,Y}[0,0] = 0.0679$, is well below the aggregate MPC of $0.1022$, although both measure spending out of a surprise one-quarter gain. Aggregate income arrives through the wage, in proportion to productivity, so most of it goes to high-productivity households, whose MPCs are low. Prompt 3 reproduces the gap.
# 3. **The general-equilibrium response.** The policy shock raises the ex-ante real rate by $0.1338$ percentage points per quarter on impact, less than the $0.25$-point shock, because inflation falls by $0.1489$ points and the Taylor rule leans against it. Output and consumption fall by $0.4335\%$ on impact. The direct interest-rate channel accounts for $-0.3684$ of that, and the income channel for $-0.0652$, or $15.0\%$ of the impact. Over the first 8 quarters the income channel accounts for $29.8\%$ of the output loss. In this calibration the income channel amplifies the recession but does not dominate it, because the households who would spend lost income are not the ones who earn most of it.
# 4. **What the checks show.** The two assertions (consumption equals output, and the two channels add up to it) are internal accounting identities of the linear solve. They show that the decomposition is complete, not that the calibration describes any economy. Sums over longer horizons are not reported, because at $T = 40$ they still depend on where the horizon is cut.

# %% [markdown]
# ## Your turn
#
# The household Jacobians are partial-equilibrium objects: they never see the Taylor rule. So you can change the monetary rule without re-solving the household problem. The cell rebuilds $\mathbf{M}_{r,Y}$ from the Phillips curve and the Taylor rule, $d\mathbf{r} = (\phi_\pi \mathbf{K} - \mathbf{K}_{+1})\, d\mathbf{Y} + \boldsymbol{\epsilon}$ with $\mathbf{K}[t,s] = \kappa \beta^{s-t}$ for $s \ge t$, and solves the linear system of section 4 with the Jacobians from section 1.
#
# **Predict first:** if the central bank responds more strongly to inflation ($\phi_\pi > 1.5$), is the impact recession deeper or shallower than in section 4? The first check compares the hand-built solve with the library's own re-solve (an internal check). The second is your prediction: it fails if the direction is wrong.

# %%
# Your turn: change the monetary rule and reuse the Jacobians
phi_turn = 2.5   # ← change this: Taylor-rule response to inflation, 1.1 to 3.0
assert 1.1 <= phi_turn <= 3.0

Jy, Jr = res.jacobian_c_y, res.jacobian_c_r
T = Jy.shape[0]
t_idx = np.arange(T)
beta_nk, kappa_nk = 0.985, 0.1
K_pi = np.where(t_idx[None, :] >= t_idx[:, None], kappa_nk * beta_nk ** (t_idx[None, :] - t_idx[:, None]), 0.0)
K_next = np.vstack([K_pi[1:], np.zeros((1, T))])     # maps dY into E_t pi_{t+1}
eps = 0.0025 * 0.7 ** t_idx

def ge_output(phi):
    """Output path from (I - J_CY - J_Cr M) dY = J_Cr eps with dr = M dY + eps."""
    M = phi * K_pi - K_next
    return np.linalg.solve(np.eye(T) - Jy - Jr @ M, Jr @ eps)

dY_15, dY_turn = ge_output(1.5), ge_output(phi_turn)
ref = solve_hank_sequence_space(T=T, beta=0.985, gamma=1.0, r_ss=0.01, phi_pi=phi_turn, kappa=0.1,
                                shock_magnitude=0.0025, shock_rho=0.7, n_a=60)
print(f"Impact output: phi = 1.5 -> {dY_15[0] * 100:+.4f}%, phi = {phi_turn} -> {dY_turn[0] * 100:+.4f}% "
      f"(library re-solve {ref.irf_output[0] * 100:+.4f}%)")
print(f"Cumulative output over {T} quarters: {dY_15.sum() * 100:+.3f} vs {dY_turn.sum() * 100:+.3f} (% of quarterly Y_ss)")

# Internal check: the hand-built GE system is the library's
assert np.max(np.abs(dY_15 - res.irf_output)) < 1e-12
assert np.max(np.abs(dY_turn - ref.irf_output)) < 1e-12
# Your prediction: a more hawkish rule gives a shallower impact recession, a more dovish one a deeper one
assert np.sign(dY_turn[0] - dY_15[0]) == np.sign(phi_turn - 1.5), "the direction of the change is wrong"

# %% [markdown]
# **Prompts.**
# 1. *Basic.* A rise in aggregate income in quarter 0 must eventually be spent. Before computing, use the intertemporal budget constraint to predict $x = \sum_t (1 + r_{ss})^{-t} \mathcal{J}_{C,Y}[t, 0]$. Compute it with `q = (1 + res.r_ss) ** -np.arange(T); x = q @ Jy[:, 0]`, then re-solve with `T=200` (everything else as in section 1). Where does the missing part go at $T = 40$? Self-check: `assert x < 0.8` at $T = 40$ and `abs(x - 1) < 0.01` at $T = 200$.
# 2. *Intermediate.* Explain the direction you found. Plot `ge_output(phi)` and the implied real rate `(phi * K_pi - K_next) @ ge_output(phi) + eps` for $\phi_\pi \in \{1.25, 1.5, 2.5\}$. Why does the same policy shock raise the real rate by less under a more hawkish rule?
# 3. *Stretch.* The aggregate MPC printed in section 1 and $\mathcal{J}_{C,Y}[0,0]$ both measure spending out of a surprise one-quarter income gain, yet they differ. Compute each household's MPC with `m = np.diff(res.policy_c, axis=0) / ((1 + res.r_ss) * np.diff(res.asset_grid))[:, None]`, repeat the last row, clip to $[0, 1]$, and average it with weights `res.distribution` (the plain MPC) and with weights `res.distribution * s` for $s = (0.5, 1.5)$ (each household's share of a wage increase). Self-check: `abs(mpc_plain - res.steady_state_mpc) < 1e-10` and `abs(mpc_earn - Jy[0, 0]) / Jy[0, 0] < 0.05`. What does this say about the size of the indirect channel?
#
# ## How comprehensive is this?
#
# - `puremacro.models.solve_hank_sequence_space` also returns `fake_news`, `simulate_transfer` (targeted fiscal transfers) and `solve_nonlinear` (a non-linear MIT transition started from this linear solution).
# - `puremacro.dsge.hank` bridges the household Jacobians into the DSGE toolkit, and notebook 44 connects them to a DSGE model; notebook 46 uses the two-asset version.
# - Notebook 52 solves the non-linear transition of a continuous-wealth Aiyagari economy after an MIT shock, the non-linear counterpart of the linear responses here.
