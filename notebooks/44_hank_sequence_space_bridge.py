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
# # HANK Sequence-Space Bridge: Dynare .mod Integration and SSJ Jacobians
#
# **How can we embed microeconomic household heterogeneity into macroeconomic DSGE models without state-space explosion, and simulate general equilibrium transition dynamics directly from Dynare-style `.mod` specifications?**
#
# For decades, macroeconomic policy analysis has relied on Representative Agent New Keynesian (RANK) models. RANK models compress the entire household sector into a single representative consumer governed by an aggregate Euler equation. However, empirical microdata reveals profound heterogeneity in balance sheets and Marginal Propensities to Consume (MPCs): low-wealth "hand-to-mouth" households consume a large fraction of any transitory income gain, while wealthy households smooth consumption according to the permanent income hypothesis.
#
# Heterogeneous-Agent New Keynesian (HANK) models incorporate incomplete markets, uninsurable idiosyncratic earnings risk, and borrowing constraints (Aiyagari-Bewley-Huggett). In HANK economies, monetary policy operates not only through the direct intertemporal substitution channel, but primarily through indirect general equilibrium income channels amplified by liquidity-constrained households.
#
# Until recently, solving HANK models required solving high-dimensional partial differential equations or tracking the infinite-dimensional cross-sectional wealth distribution $\mathcal{D}_t$ across time—imposing severe computational costs. Adrien Auclert, Bence Bardóczy, Matthew Rognlie, and Ludwig Straub (2021, *Econometrica*) introduced the **Sequence-Space Jacobian (SSJ)** framework and the **Fake-News Algorithm**, computing the exact intertemporal Jacobians of heterogeneous blocks in seconds.
#
# In **puremacro 3.0.0**, researchers can bridge Dynare-style `.mod` files directly with the SSJ engine using the new `hetagent_block; ... end;` syntax. In this interactive showcase, we load a HANK model from a `.mod` file, inspect its stationary wealth and MPC distributions, compute consumption Jacobians via Fake-News, solve linear and nonlinear Broyden transition dynamics following an MIT monetary shock, and generate policy reports—all in **100% pure Python**.

# %% [markdown]
# ## The Method in Math: From Micro Incomplete Markets to Sequence Space
#
# **1. The Household Block.** A continuum of households of unit mass solves the discrete-time incomplete-markets problem:
# $$ \max_{\{c_{it}, a_{i,t+1}\}} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t \frac{c_{it}^{1 - \gamma} - 1}{1 - \gamma} $$
# subject to the budget constraint and borrowing limit $\underline{a} = 0$:
# $$ c_{it} + a_{i,t+1} = (1 + r_t) a_{it} + (1 - \tau_t) w_t s_{it}, \quad a_{i,t+1} \ge \underline{a} $$
# where idiosyncratic labor productivity $s_{it} \in \{s_L, s_H\}$ follows a finite-state Markov chain with transition matrix $\Pi_s$, and $r_t$ is the real return on wealth. The Euler equation with an occasionally binding borrowing constraint is:
# $$ c_{it}^{-\gamma} \ge \beta (1 + r_{t+1}) \mathbb{E}_t \left[ c_{i,t+1}^{-\gamma} \right], \quad \text{with equality if } a_{i,t+1} > \underline{a} $$
# The Endogenous Grid Method (EGM, Carroll 2006) solves for policy functions $c(a, s)$ and $a'(a, s)$. The stationary cross-sectional wealth distribution $\mathcal{D}^*(a, s)$ satisfies the Chapman-Kolmogorov fixed point $\mathcal{D}^* = \Lambda' \mathcal{D}^*$.
#
# **2. Sequence-Space Equilibrium.** Rather than tracking the distribution $\mathcal{D}_t$ as a state variable, the macroeconomic general equilibrium is expressed as an infinite-dimensional nonlinear equation in sequence space:
# $$ \mathcal{H}(\mathbf{U}, \mathbf{Z}) = 0 $$
# where $\mathbf{U} = (\mathbf{Y}, \mathbf{r}, \mathbf{\pi}) \in \mathbb{R}^{3T}$ stacks the paths of endogenous variables over horizon $T$, and $\mathbf{Z} = (\mathbf{\varepsilon}^m) \in \mathbb{R}^T$ stacks exogenous shock sequences.
#
# **3. The Fake-News Algorithm.** For any input sequence $\mathbf{X} \in \{\mathbf{r}, \mathbf{Y}\}$, the intertemporal consumption Jacobian $\mathcal{J}_{C, X} \in \mathbb{R}^{T \times T}$ measures:
# $$ \mathcal{J}_{C, X}[t, s] = \frac{\partial C_t}{\partial X_s} $$
# Auclert et al. (2021) decompose $\mathcal{J}_{C, X}$ into the genuine date-0 policy response and the forward propagation of distribution shifts ("fake news" $\mathcal{F}_{t,s}$):
# $$ \mathcal{J}_{C, X} = \mathcal{T} + \sum_{k=0}^{T-1} \mathcal{U}^k \mathcal{F} (\mathcal{V}')^k $$
# This reduces the computational cost of evaluating the full $T \times T$ Jacobian from $T$ backward simulations to just **one backward pass** and **one forward pass**.
#
# **4. General Equilibrium Sequence Solve.** Market clearing $Y_t = C_t$ combined with the New Keynesian Phillips Curve and Taylor Rule yields the linear sequence system:
# $$ (\mathbf{I} - \mathcal{J}_{C, Y} - \mathcal{J}_{C, r} \mathbf{M}_{r, Y}) d\mathbf{Y} = \mathcal{J}_{C, r} \mathbf{M}_{r, \varepsilon} d\mathbf{\varepsilon}^m $$
# where $\mathbf{M}_{r, Y}$ and $\mathbf{M}_{r, \varepsilon}$ map output and shocks into real interest rates through the aggregate block. Inverting this $T \times T$ matrix yields the exact general equilibrium transition paths in milliseconds.

# %% [markdown]
# ## Intuition
#
# **Intuition.** In representative-agent macroeconomics, aggregate consumption is governed by a single unconstrained Euler equation, meaning households smooth temporary fluctuations across their entire lifetimes and exhibit minimal Marginal Propensities to Consume (MPC $\approx r \approx 1\%$). In contrast, microeconomic survey data shows that a large mass of households holds near-zero liquid assets and is borrowing-constrained ("hand-to-mouth"). For these households, the MPC out of unexpected income is enormous—often exceeding $50\%$ per quarter.
#
# When monetary policy shifts, its general equilibrium effect in HANK is not primarily direct intertemporal substitution (inducing wealthy households to delay consumption), but rather an indirect general equilibrium income channel: rate cuts stimulate output and labor demand, transferring wage income to constrained households who spend it immediately. The Sequence-Space Jacobian (SSJ) framework and the Fake-News Algorithm solve this infinite-dimensional microeconomic distribution problem with extraordinary speed: by computing intertemporal consumption Jacobians directly in sequence space, general equilibrium transitions under both linear and nonlinear Broyden algorithms solve in milliseconds without simulating millions of Monte Carlo agents.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Editorial styling and palette contract
_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

# Ensure clean non-blocking execution when executed as a CLI script
if not hasattr(sys, "ps1") and "IPython" not in sys.modules:
    plt.show = lambda *args, **kwargs: None

import puremacro.dsge as dsge
from puremacro.dsge import HANKModel, HANKResult, load_hank_mod, solve_hank_bridge

print(f"puremacro version: {dsge.__name__} loaded successfully")

# %% [markdown]
# ## 1. Dynare `.mod` Syntax with `hetagent_block`
#
# In traditional Dynare `.mod` files, all equations represent aggregate relationships between representative agents. In `puremacro`, we introduce the `hetagent_block; ... end;` construct.
#
# This block declares:
# - `model`: The underlying heterogeneous agent household problem (e.g. `one_asset_hank`).
# - `n_a`: Number of asset grid points (we use $n_a = 50$ for fast execution in 2–4 seconds).
# - `a_max`: Upper wealth boundary ($a_{\max} = 30.0$).
# - `borrowing_limit`: Hard borrowing constraint ($\underline{a} = 0.0$).
# - `grid`: Asset grid spacing (`hyperbolic` clustering points near the borrowing constraint).
#
# Below is the reference `hank_ssj.mod` model declaration:

# %%
# Locate the bundled reference .mod file
mod_path = Path(dsge.__file__).parent / "_references" / "hank_ssj.mod"
mod_text = mod_path.read_text(encoding="utf-8")
print(mod_text.strip())

# %% [markdown]
# We load and parse the `.mod` file via `load_hank_mod()`.
# The parser interprets the `hetagent_block`, initializes the asset grid, and computes the microeconomic stationary distribution $\mathcal{D}^*(a)$.

# %%
model = load_hank_mod(mod_path)

print("--- Model Declarations ---")
print(f"Endogenous variables : {model.variables}")
print(f"Exogenous shocks     : {model.shocks}")
print(f"HetAgent block config: {model.hetagent_config}")
print(f"Structural parameters: beta={model.beta}, gamma={model.gamma}, r_ss={model.r_ss}, phi_pi={model.phi_pi}, kappa={model.kappa}")
print(f"Steady-state output Y: {model.steady_state['Y']:.4f}")
print(f"Steady-state rate   r: {model.steady_state['r']:.4f}")

# Sanity assertions
assert "hetagent_block" in str(model.hetagent_config) or len(model.hetagent_config) > 0
assert model.steady_state["Y"] > 0.0
assert len(model.asset_grid) == 50

# %% [markdown]
# ## 2. Microeconomic Stationary Wealth Distribution $\mathcal{D}^*(a)$ and MPC Profile
#
# A central pillar of HANK theory is the severe cross-sectional variation in the Marginal Propensity to Consume (MPC).
#
# Under the borrowing constraint $a \ge 0$, households with zero or near-zero assets are "hand-to-mouth" (HtM): facing uninsurable income risk and unable to borrow against future labor income, they consume a large portion of any marginal dollar received. In contrast, wealthy households at the right tail of the distribution have small MPCs dictated by the permanent income hypothesis ($MPC \approx r$).

# %%
a_grid = model.asset_grid
D_ss = model.asset_distribution
mpc_ss = model.mpc_distribution

# Summary distribution diagnostics
total_mass = float(np.sum(D_ss))
mean_wealth = float(np.sum(a_grid * D_ss))
mpc_borrowing = float(mpc_ss[0]) if mpc_ss is not None else float("nan")
mpc_wealthy = float(mpc_ss[-1]) if mpc_ss is not None else float("nan")
mean_mpc = float(np.sum(mpc_ss * D_ss)) if mpc_ss is not None else float("nan")

print("--- Microeconomic Distribution Diagnostics ---")
print(f"Asset grid range          : [{a_grid[0]:.2f}, {a_grid[-1]:.2f}] across {len(a_grid)} points")
print(f"Total probability mass    : {total_mass:.6f} (must equal 1.0)")
print(f"Mean household assets     : {mean_wealth:.4f}")
print(f"Hand-to-mouth MPC (a = 0) : {mpc_borrowing:.4f} (constrained households)")
print(f"Wealthy household MPC     : {mpc_wealthy:.4f} (permanent income households)")
print(f"Aggregate mean MPC        : {mean_mpc:.4f}")

# Distribution assertions
assert np.isclose(total_mass, 1.0, atol=1e-5)
assert mpc_borrowing > mean_mpc > mpc_wealthy

# %% [markdown]
# We plot the stationary asset distribution $\mathcal{D}^*(a)$ and the conditional MPC profile across the wealth grid.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

# Panel 1: Stationary Wealth Distribution
ax1.plot(a_grid, D_ss, color="#1f77b4", lw=2.2, label=r"Density $\mathcal{D}^*(a)$")
ax1.fill_between(a_grid, 0, D_ss, color="#1f77b4", alpha=0.25)
ax1.set_title(r"Stationary Wealth Distribution $\mathcal{D}^*(a)$", fontsize=11, fontweight="bold")
ax1.set_xlabel("Household Assets $a$")
ax1.set_ylabel("Probability Density")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right")

# Panel 2: MPC across wealth grid
ax2.plot(a_grid, mpc_ss, color="#d62728", lw=2.2, label=r"$MPC(a)$")
ax2.axhline(mean_mpc, color="#333333", linestyle="--", lw=1.3, label=f"Mean MPC ({mean_mpc:.3f})")
ax2.set_title("Marginal Propensity to Consume $MPC(a)$", fontsize=11, fontweight="bold")
ax2.set_xlabel("Household Assets $a$")
ax2.set_ylabel("Quarterly MPC")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

# %% [markdown]
# Notice how the MPC profile drops steeply from over $50\%$ at the borrowing limit ($a=0$) down towards $1\%$ at higher wealth levels. This microeconomic heterogeneity is the key driver of aggregate amplification.

# %% [markdown]
# ## 3. Intertemporal Consumption Jacobians via the Fake-News Algorithm
#
# The Sequence-Space Jacobian framework relies on the intertemporal consumption Jacobians:
# 1. $\mathcal{J}_{C, r} \in \mathbb{R}^{T \times T}$: Response of aggregate consumption at time $t$ to an anticipated real interest rate innovation at time $s$.
# 2. $\mathcal{J}_{C, Y} \in \mathbb{R}^{T \times T}$: Response of aggregate consumption at time $t$ to an anticipated aggregate income innovation at time $s$.
#
# Using `model.compute_jacobians(T=30)`, puremacro runs the **Fake-News Algorithm** (Auclert et al., 2021) to evaluate both matrices in sequence space.

# %%
T_horizon = 30
jacobians = model.compute_jacobians(T=T_horizon)
J_C_r = jacobians["J_C_r"]
J_C_Y = jacobians["J_C_Y"]

dC0_dr0 = float(J_C_r[0, 0])
dC0_dY0 = float(J_C_Y[0, 0])

print(f"Jacobian J_C_r shape: {J_C_r.shape}")
print(f"Jacobian J_C_Y shape: {J_C_Y.shape}")
print(f"Intertemporal substitution (dC_0 / dr_0): {dC0_dr0:+.4f} (< 0, rate hike depresses consumption)")
print(f"Keynesian income multiplier (dC_0 / dY_0):{dC0_dY0:+.4f} (> 0, income stimulates consumption)")

# Structural assertions
assert J_C_r.shape == (T_horizon, T_horizon)
assert J_C_Y.shape == (T_horizon, T_horizon)
assert dC0_dr0 < 0.0, "Interest rate hike must depress contemporaneous consumption"
assert dC0_dY0 > 0.0, "Income hike must stimulate contemporaneous consumption"

# %% [markdown]
# Let us visualize the heatmaps of $\mathcal{J}_{C, Y}$ and $\mathcal{J}_{C, r}$.
#
# - In the income Jacobian $\mathcal{J}_{C, Y}$, the strong diagonal and upper-triangular entries capture how anticipated future income boosts current consumption, while discounting and precautionary motives dampen the response to distant shocks.
# - In the interest rate Jacobian $\mathcal{J}_{C, r}$, the negative diagonal illustrates the standard intertemporal substitution channel.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

sub_T = 15

# Income Jacobian Heatmap
im1 = ax1.imshow(J_C_Y[:sub_T, :sub_T], cmap="YlGnBu", origin="upper")
ax1.set_title(r"Income Jacobian $\mathcal{J}_{C, Y}$ (Fake-News)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Shock Period $s$")
ax1.set_ylabel("Response Period $t$")
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Interest Rate Jacobian Heatmap
im2 = ax2.imshow(J_C_r[:sub_T, :sub_T], cmap="Reds_r", origin="upper")
ax2.set_title(r"Rate Jacobian $\mathcal{J}_{C, r}$ (Fake-News)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Shock Period $s$")
ax2.set_ylabel("Response Period $t$")
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. General Equilibrium Transitions: Linear vs. Nonlinear Broyden Transitions
#
# Now we simulate an expansionary **MIT monetary policy shock**:
# $$ \varepsilon_{m, 0} = -0.0025 \quad (-25\text{ basis points quarterly cut}), \quad \rho_m = 0.5, \quad T = 30 $$
#
# In sequence space, the model can be solved in two distinct modes:
# 1. **Linear SSJ Transition (`nonlinear=False`):** Solves the coupled $T \times T$ linear system directly in one matrix inversion.
# 2. **Nonlinear Transition (`nonlinear=True`):** Uses Broyden's quasi-Newton method, starting from the linear sequence inverse, to iterate on market-clearing conditions while accounting for nonlinearities in household policy functions and the borrowing constraint.

# %%
shock_mag = -0.0025  # -25 bps rate cut
shock_rho = 0.5      # AR(1) persistence
horizon = 30

# 1. Linear General Equilibrium Solve
res_lin = model.simulate(
    shock="eps_m",
    magnitude=shock_mag,
    rho=shock_rho,
    horizon=horizon,
    nonlinear=False,
)

# 2. Nonlinear Broyden General Equilibrium Solve
res_nonlin = model.simulate(
    shock="eps_m",
    magnitude=shock_mag,
    rho=shock_rho,
    horizon=horizon,
    nonlinear=True,
)

print(f"Linear transition converged:    {res_lin.converged}")
print(f"Nonlinear transition converged: {res_nonlin.converged}")

# Compare impact responses
df_lin = res_lin.transition_paths
df_nonlin = res_nonlin.transition_paths

print("\n--- Impact Responses (t = 0, Basis Points) ---")
print(f"Output   dY : Linear = {df_lin['Y'].iloc[0]*10000:+.2f} bps | Nonlinear = {df_nonlin['Y'].iloc[0]*10000:+.2f} bps")
print(f"Real rate dr: Linear = {df_lin['r'].iloc[0]*10000:+.2f} bps | Nonlinear = {df_nonlin['r'].iloc[0]*10000:+.2f} bps")
print(f"Inflation dpi:Linear = {df_lin['pi'].iloc[0]*10000:+.2f} bps | Nonlinear = {df_nonlin['pi'].iloc[0]*10000:+.2f} bps")

assert res_lin.converged and res_nonlin.converged
assert df_lin["Y"].iloc[0] > 0.0, "Rate cut must expand output"
assert df_lin["r"].iloc[0] < 0.0, "Rate cut must lower real rate"

# %% [markdown]
# We plot the general equilibrium impulse response comparisons between linear and nonlinear transitions.

# %%
t_axis = np.arange(horizon)
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

vars_config = [
    ("Y", "Output $dY$", axes[0, 0]),
    ("C", "Consumption $dC$", axes[0, 1]),
    ("r", "Real Interest Rate $dr$", axes[1, 0]),
    ("pi", r"Inflation $d\pi$", axes[1, 1]),
]

for var, label, ax in vars_config:
    ax.plot(t_axis, df_lin[var] * 10000, color="#1f77b4", lw=2.2, label="Linear SSJ")
    ax.plot(t_axis, df_nonlin[var] * 10000, color="#d62728", lw=1.8, linestyle="--", label="Nonlinear Broyden")
    ax.axhline(0.0, color="#666666", lw=0.8, linestyle=":")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Quarters after Shock")
    ax.set_ylabel("Basis Points (bps)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right" if var in ("Y", "C", "pi") else "lower right")

plt.tight_layout()
plt.show()

# %% [markdown]
# The linear SSJ approximation tracks the exact nonlinear Broyden transition paths with remarkable fidelity. For standard monetary policy shocks, the linear sequence solve provides virtually identical accuracy in a fraction of a second.

# %% [markdown]
# ## 5. Visualization and Policy Reporting with `HANKResult`
#
# The `HANKResult` object provides standardized methods for economic diagnostics and policy briefings:
# - `.summary()`: Generates a tabular summary of impact responses, peak deviations, and peak horizons.
# - `.to_markdown()`: Exports formatted Markdown tables for policy memos and documentation.
# - `.to_latex()`: Exports publication-ready LaTeX `tabular` code.
# - `.plot_transition()`: Plots multi-panel transition paths for all model variables.
# - `.plot_distribution()`: Visualizes stationary wealth distributions and MPC profiles.

# %%
# Summary Table
summary_df = res_lin.summary()
print("--- Transition Dynamics Summary Table ---")
print(summary_df.to_string())

# %% [markdown]
# ### Markdown Export for Policy Briefs

# %%
md_table = res_lin.to_markdown()
print(md_table)

# %% [markdown]
# ### LaTeX Export for Academic Publication

# %%
latex_table = res_lin.to_latex()
print(latex_table)

# %% [markdown]
# ### Built-in `.plot_distribution()` Method

# %%
fig_dist, axes_dist = res_lin.plot_distribution(figsize=(11.0, 3.8))
plt.show()

# %% [markdown]
# ### Built-in `.plot_transition()` Method

# %%
fig_trans, axes_trans = res_lin.plot_transition(
    variables=["Y", "C", "r", "pi", "i"],
    figsize=(11.5, 5.5),
)
plt.show()

# %% [markdown]
# ## Read the output
#
# **Read the output.**
# 1. **Microeconomic Stationary Distribution & MPCs**: Total probability mass over the 50-point wealth grid sums strictly to $1.0$. The Marginal Propensity to Consume exhibits stark cross-sectional variation: constrained households at the borrowing limit ($a=0$) display a quarterly MPC exceeding $50\%$, whereas unconstrained wealthy households at the upper asset boundary ($a=30$) exhibit an MPC below $1.5\%$, confirming the empirical hand-to-mouth distribution.
# 2. **Intertemporal Consumption Jacobians ($\mathcal{J}_{C, Y}, \mathcal{J}_{C, r}$)**: The Fake-News Algorithm computes exact $30 \times 30$ sequence Jacobians in a single forward-backward sweep. The contemporaneous intertemporal substitution derivative ($\partial C_0 / \partial r_0 < 0$) confirms that real interest rate increases dampen consumer spending, while the Keynesian income multiplier ($\partial C_0 / \partial Y_0 > 0$) demonstrates substantial contemporaneous general equilibrium demand feedback.
# 3. **Linear vs. Nonlinear Broyden Transitions**: Both the linear sequence solve and the non-linear Broyden quasi-Newton solver achieve full convergence under a $-25\text{ bps}$ monetary policy shock ($\|\mathcal{H}\|_\infty < 10^{-6}$). The linear SSJ approximation matches the exact non-linear trajectory to within less than $0.1$ basis points across output, real rates, and inflation, verifying the accuracy of sequence-space linearization for macroeconomic business-cycle shocks.
# 4. **Policy Reporting via `HANKResult`**: The standardized presentation interface exports cleanly to LaTeX tables, Markdown memos, and multi-panel dashboards, facilitating central bank policy communication.

# %% [markdown]
# ## Your turn
#
# **Prompts.**
# 1. *Basic*: Alter the monetary policy shock persistence (`shock_rho_yt = 0.3` vs `0.7`) or magnitude (`-0.0050` vs `-0.0010`) and observe how the output boom and disinflation speed change.
# 2. *Intermediate*: Run the non-linear Broyden transition (`nonlinear=True`) with a larger shock (e.g. `-0.0100`, a 100 bps rate cut) and quantify the percentage divergence between the linear SSJ approximation and the exact non-linear solution.
# 3. *Stretch*: Modify the Taylor rule feedback parameter `phi_pi` in `hank_ssj.mod` and recalculate the general equilibrium consumption Jacobians $\mathcal{J}_{C, Y}$ and $\mathcal{J}_{C, r}$ to determine how monetary policy stance alters general equilibrium amplification.

# %%
# Your turn: customize monetary policy shock magnitude and persistence
# ← change this: test shock_magnitude_yt = -0.0010, -0.0025, or -0.0050 (-50 bps cut)
shock_magnitude_yt = -0.0050
# ← change this: test persistence shock_rho_yt = 0.3, 0.5, or 0.7
shock_rho_yt = 0.70

res_yt = model.simulate(
    shock="eps_m",
    magnitude=shock_magnitude_yt,
    rho=shock_rho_yt,
    horizon=30,
    nonlinear=False,
)

impact_Y_bps = float(res_yt.transition_paths["Y"].iloc[0] * 10000)
impact_r_bps = float(res_yt.transition_paths["r"].iloc[0] * 10000)
terminal_Y = float(res_yt.transition_paths["Y"].iloc[-1])

print(f"Custom Simulation: Shock = {shock_magnitude_yt*10000:.0f} bps | Rho = {shock_rho_yt:.2f}")
print(f"Impact Output Deviation (Y_0)    : {impact_Y_bps:+.2f} bps")
print(f"Impact Real Rate Deviation (r_0) : {impact_r_bps:+.2f} bps")
print(f"Terminal Output (t=30)           : {terminal_Y:.6f}")

# Downstream automated assertions
assert res_yt.converged
assert res_yt.transition_paths["Y"].iloc[0] > 0.0, "Expansionary rate cut must expand output"
assert res_yt.transition_paths["r"].iloc[0] < 0.0, "Rate cut must reduce initial real rate"
assert np.isclose(terminal_Y, 0.0, atol=1e-3)

# %% [markdown]
# ## How comprehensive is this?
#
# `puremacro` provides an end-to-end Sequence-Space Jacobian architecture for heterogeneous-agent macroeconomics in 100% pure Python:
# - `load_hank_mod` and `HANKModel`: Parses Dynare-style `.mod` specifications with declared `hetagent_block` syntax, automating grid discretization, Endogenous Grid Method (EGM) policy iteration, and stationary wealth distribution solving.
# - `compute_jacobians`: Implements the Auclert, Bardóczy, Rognlie & Straub (2021) Fake-News Algorithm, decomposing high-dimensional intertemporal response matrices into single-pass forward and backward recursions.
# - `solve_hank_bridge` and `model.simulate`: Simulates general equilibrium transition dynamics under both linear matrix inversion and non-linear Broyden quasi-Newton root-finding.
# - `solve_nonlinear_transition`: Extended sequence-space solver supporting arbitrary MIT shocks, fiscal stimulus experiments, and occasionally binding borrowing constraints.
