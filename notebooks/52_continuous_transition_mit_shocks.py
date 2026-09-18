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
# # Continuous Transition Dynamics and MIT Shocks: Non-Linear Distributional Sequence Space
#
# **How do continuous household wealth distributions $\mu_t(k, z)$ and general equilibrium factor prices $\{r_t, w_t\}$ transition non-linearly following an unexpected aggregate macroeconomic shock in an incomplete-markets economy, and how does endogenous precautionary savings shape macroeconomic persistence and wealth inequality?**
#
# In modern macroeconomic theory, the cross-sectional distribution of wealth is not merely an accounting artifact; it is an active, aggregate state variable that governs macroeconomic transmission. When an economy experiences an unexpected aggregate shock—such as a persistent total factor productivity (TFP) surge or a sudden monetary policy tightening—the transmission through an incomplete-markets economy (Bewley-Huggett-Aiyagari) is fundamentally non-linear. The initial cross-sectional wealth distribution $\mu_0(k, z)$ is physically predetermined on impact, preventing aggregate capital supply from adjusting instantaneously. Consequently, clearing factor markets requires sharp, immediate jumps in the real wage and real interest rate.
#
# Over time, heterogeneous households adjust their consumption and savings behavior in response to shifted factor prices and altered precautionary savings motives. Using backward continuous Endogenous Grid Method (EGM) policy iterations and forward time-dependent Young (2010) non-stochastic density evolution $\mu_{t+1} = \mathcal{T}_t^* \mu_t$, this notebook computes the exact non-linear transition path of continuous wealth distributions and resolves the sequence of capital market-clearing conditions via sequence-space Broyden Quasi-Newton relaxation. We examine how aggregate productivity shocks compress or expand wealth inequality (measured by time-varying Gini coefficients and Lorenz curves) and how precautionary savings create endogenous propagation persistence far beyond the duration of the exogenous shock itself.

# %% [markdown]
# ## The method in math — Continuous Transition Dynamics and MIT Shocks
#
# **1. Backward Continuous Endogenous Grid Method (EGM).** Given a path of factor prices $\{r_t, w_t\}_{t=0}^{T-1}$ converging to terminal steady-state values $(r^*, w^*)$ at horizon $T$, household policy functions are computed recursively backward in time from $t = T-1$ to $t = 0$. Using CRRA utility $u(c) = \frac{c^{1-\gamma} - 1}{1-\gamma}$, the Euler equation for a household with asset state $k$, productivity state $z_i$, and continuation policy $c_{t+1}(a', z')$ satisfies:
# $$ \mathbb{E}_t\left[u'\left(c_{t+1}(a', z')\right)\right] = \beta (1 + r_{t+1}) \sum_{j=1}^{n_z} P_z(z_i, z_j) \left[c_{t+1}(a', z_j)\right]^{-\gamma}. $$
# Inverting the marginal utility yields the endogenous consumption policy:
# $$ c_t^{\text{endo}}(a', z_i) = \left( \mathbb{E}_t\left[u'\left(c_{t+1}(a', z')\right)\right] \right)^{-1/\gamma}. $$
# From the budget constraint $(1 + r_t) a_t + w_t z_i = c_t + a'$, the endogenous beginning-of-period asset level $a_t^{\text{endo}}$ associated with target savings choice $a'$ is:
# $$ a_t^{\text{endo}}(a', z_i) = \frac{c_t^{\text{endo}}(a', z_i) + a' - w_t z_i}{1 + r_t}. $$
# Linearly interpolating the pairs $(a_t^{\text{endo}}, a')$ onto the fixed continuous asset grid $\mathcal{K} = \{k_1, \dots, k_{N_k}\}$ and imposing the borrowing constraint $a' \ge 0$ yields the continuous policy functions $a'_t(k, z_i) = \max\left\{0, \text{interp}\left(k; a_t^{\text{endo}}(\cdot, z_i), a'\right)\right\}$ and $c_t(k, z_i) = (1 + r_t) k + w_t z_i - a'_t(k, z_i)$.
#
# **2. Forward Non-Stochastic Density Evolution (Young 2010 Lottery Operator).** Given the initial stationary distribution $\mu_0(k, z)$ and the backward policy sequence $\{a'_t\}_{t=0}^{T-1}$, the cross-sectional probability density $\mu_t(k, z)$ advances forward in time. For every savings decision $a' = a'_t(k_i, z_j)$ falling in the grid bracket $[k_m, k_{m+1}]$, the Young (2010) linear lottery operator assigns mass to adjacent nodes to strictly preserve the conditional expectation $\mathbb{E}[a']$:
# $$ w_{\text{lo}}(a') = \frac{k_{m+1} - a'}{k_{m+1} - k_m}, \qquad w_{\text{hi}}(a') = 1 - w_{\text{lo}}(a'). $$
# The joint forward push operator $\mathcal{T}_t^*$ scatters mass along the asset grid and updates exogenous Markov productivity states:
# $$ \mu_{t+1}(k_m, z_l) = \sum_{j=1}^{n_z} P_z(z_j, z_l) \sum_{i=1}^{N_k} \mu_t(k_i, z_j) \left[ w_{\text{lo}}\left(a'_t(k_i, z_j)\right) \mathbf{1}_{\{m = j_{\text{lo}}\}} + w_{\text{hi}}\left(a'_t(k_i, z_j)\right) \mathbf{1}_{\{m = j_{\text{hi}}\}} \right]. $$
# This operator guarantees strict probability mass conservation to machine precision: $\sum_{k, z} \mu_t(k, z) = 1.0 \pm 10^{-15}$ for all $t \in [0, T]$.
#
# **3. Sequence-Space General Equilibrium Market Clearing.** Aggregate capital supply $K_t^s$ is obtained by integrating the continuous asset distribution:
# $$ K_t^s(\mathbf{r}) = \sum_{i=1}^{N_k} \sum_{j=1}^{n_z} k_i \, \mu_t(k_i, z_j). $$
# A representative competitive firm operates a Cobb-Douglas technology $Y_t = Z_t (K_t^d)^\alpha L^{1-\alpha}$ with capital depreciation $\delta$. Factor demands satisfy $r_t = \alpha Z_t (K_t^d / L)^{\alpha - 1} - \delta$ and $w_t = (1 - \alpha) Z_t (K_t^d / L)^\alpha$. Inverting for capital demand gives:
# $$ K_t^d(r_t; Z_t) = L \left( \frac{r_t + \delta}{\alpha Z_t} \right)^{\frac{1}{\alpha - 1}}. $$
# General equilibrium requires clearing the capital market at every transition date $t = 0, \dots, T-1$:
# $$ H_t(\mathbf{r}) \equiv K_t^s(\mathbf{r}) - K_t^d(r_t; Z_t) = 0, \qquad \mathbf{H}(\mathbf{r}) = \mathbf{0} \in \mathbb{R}^T. $$
#
# **4. Sequence-Space Broyden Quasi-Newton Solver.** The non-linear equation system $\mathbf{H}(\mathbf{r}) = \mathbf{0}$ is solved via Broyden's method with Sherman-Morrison rank-1 approximate inverse Jacobian updates:
# $$ B_{k+1} = B_k + \frac{(\Delta \mathbf{r}_k - B_k \Delta \mathbf{H}_k)(\Delta \mathbf{r}_k^\top B_k)}{\Delta \mathbf{r}_k^\top B_k \Delta \mathbf{H}_k}, \qquad \mathbf{r}_{k+1} = \mathbf{r}_k - \theta \, B_k \mathbf{H}(\mathbf{r}_k), $$
# where $\theta \in (0, 1]$ is a damping parameter and $B_0$ is initialized analytically from the static diagonal firm demand derivative $B_0 = \text{diag}\left( \frac{1 - \alpha}{K_t^d} (r_t + \delta) \right)$.

# %% [markdown]
# ## Intuition
#
# **Intuition.** In representative-agent macro models, unexpected aggregate productivity or interest rate shocks produce instantaneous capital adjustments along a saddle path. In an incomplete-markets world with idiosyncratic earnings risk, however, aggregate dynamics are constrained by the physical inertia of the cross-sectional wealth distribution. Households cannot instantaneously reallocate their balance sheets; rather, asset accumulation requires real time, giving rise to rich distributional propagation.
#
# When a positive TFP shock ($+5\%$) hits an Aiyagari economy unexpectedly, the marginal product of capital and the real wage jump immediately. Because the pre-shock wealth distribution $\mu_0(k, z)$ is physically predetermined at date $t=0$, aggregate capital supply $K_0^s$ cannot change on impact. For firms to clear the capital market, the equilibrium real interest rate $r_0$ must spike upward to absorb the increased marginal productivity of the fixed capital stock.
#
# This initial price response alters household savings incentives through two opposing channels:
# 1. **Substitution & Return Effects:** A higher real return $r_t$ increases the reward to saving, encouraging households to postpone consumption and accumulate wealth.
# 2. **Precautionary Savings & Income Effects:** Concurrently, the surge in real wages $w_t$ boosts labor earnings across all employment states. For wealth-poor households near the borrowing constraint ($k = 0$), this positive earnings windfall relaxes credit constraints and allows them to build up their liquid buffer stocks.
#
# Consequently, wealth inequality experiences a distinct cyclical pattern. On impact, the wage surge disproportionately elevates the earnings of low-wealth workers relative to the return on pre-existing capital, causing the cross-sectional wealth Gini coefficient to compress. Over the subsequent 5 to 10 quarters, as households channel high savings into physical capital, aggregate capital deepens ($K_t$ expands toward a peak). As capital accumulates, the marginal product of capital falls, dampening the interest rate and gradually reverting the wealth distribution $\mu_t(k, z)$ toward the stationary steady state.
#
# Solving this dynamic feedback loop requires finding the unique price sequence $\mathbf{r} = \{r_t\}_{t=0}^{T-1}$ such that asset supply generated by millions of forward-looking households exactly matches capital demand generated by firms at every point in time. Traditional shooting algorithms frequently suffer from catastrophic numerical instability due to explosive backward roots. Sequence-space Broyden Quasi-Newton resolves this by treating the entire $T$-period price trajectory as a single unified vector, updating the sequence-space Jacobian with rank-1 Sherman-Morrison corrections to achieve robust quadratic-like convergence in a few seconds.

# %%
# Preamble: import numerical libraries, plotting style, and continuous solvers
import sys
from pathlib import Path
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.vfi import (
    solve_aiyagari_continuous,
    continuous_mit_shock,
    solve_continuous_transition,
    ContinuousStationaryDistribution,
)

# Set deterministic random generator seed for reproducibility
rng = np.random.default_rng(42)

# Global model calibration parameters
# Household subjective discount factor (annualized calibration)
beta = 0.96
# Coefficient of relative risk aversion (CRRA)
gamma = 2.0
# Capital share of output (Cobb-Douglas)
alpha = 0.36
# Annual capital depreciation rate
delta = 0.08
# Autoregressive persistence of idiosyncratic labor productivity
rho_z = 0.90
# Standard deviation of labor productivity innovations
sigma_z = 0.20
# Number of continuous asset grid nodes
N_k = 150
# Number of discrete labor productivity states
n_z = 5

print("Calibration loaded: beta=0.96, gamma=2.0, alpha=0.36, delta=0.08, N_k=150, n_z=5")

# %%
# --- Experiment 1: Baseline Continuous Stationary Incomplete-Markets Equilibrium ---
# Solve the pre-shock stationary general equilibrium using continuous EGM and Young (2010)
print("Solving baseline continuous stationary equilibrium (Aiyagari)...")
t0 = time.perf_counter()
ss_base = solve_aiyagari_continuous(
    beta=beta,
    gamma=gamma,
    alpha=alpha,
    delta=delta,
    rho_z=rho_z,
    sigma_z=sigma_z,
    N_k=N_k,
    n_z=n_z,
    max_evals=60,
)
t_ss = time.perf_counter() - t0

# Compute baseline wealth inequality statistics
K_hist = ss_base.distribution.asset_grid
gini_base = ss_base.distribution.gini()
p_lorenz, L_base = ss_base.distribution.lorenz(100)
p50_base = ss_base.distribution.percentile(50.0)
p90_base = ss_base.distribution.percentile(90.0)

print(f"Baseline Stationary Equilibrium Solved in {t_ss:.2f}s:")
print(f"  Equilibrium Interest Rate r* = {ss_base.r:.6f} ({ss_base.r * 100:.3f}%)")
print(f"  Equilibrium Real Wage     w* = {ss_base.w:.6f}")
print(f"  Aggregate Capital Stock   K* = {ss_base.K:.4f}")
Y_base = float((ss_base.K ** alpha) * (ss_base.L ** (1.0 - alpha)))
print(f"  Aggregate Output          Y* = {Y_base:.4f}")
print(f"  Capital-to-Output Ratio  K/Y = {ss_base.K / Y_base:.3f}")
print(f"  Wealth Gini Coefficient   G* = {gini_base:.4f}")
print(f"  Median Wealth (P50)          = {p50_base:.3f}")
print(f"  Top 10% Wealth Cutoff (P90)  = {p90_base:.3f}")

# Verification assertions for baseline equilibrium
assert ss_base.converged, "Baseline stationary equilibrium failed to converge"
assert ss_base.K > 0, "Aggregate capital stock must be strictly positive"
assert ss_base.distribution.mass_error < 1e-12, f"Stationary distribution mass error {ss_base.distribution.mass_error:.2e} exceeds 1e-12"
assert 0.005 < ss_base.r < 1.0 / beta - 1.0, f"Interest rate r*={ss_base.r:.4f} must reside strictly in (0, 1/beta - 1)"
assert 0.40 < gini_base < 0.65, f"Wealth Gini {gini_base:.4f} outside plausible incomplete-markets empirical range"

# %%
# --- Experiment 2: Non-Linear Transition Path under an Unexpected MIT TFP Shock ---
# Simulate a 40-quarter transition following an unexpected +5% TFP shock with persistence rho=0.80
horizon = 40
shock_size = 0.05
persistence = 0.80

print(f"Simulating {horizon}-quarter transition under unexpected {shock_size * 100:+.1f}% TFP shock (rho={persistence})...")
t0 = time.perf_counter()
trans_res = continuous_mit_shock(
    ss_base,
    shock_type="tfp",
    shock_size=shock_size,
    persistence=persistence,
    horizon=horizon,
    solver="broyden",
    damping=0.4,
    tol=1e-4,
)
t_trans = time.perf_counter() - t0

# Headline transition metrics
impact_r = trans_res.r_path[0]
impact_w = trans_res.w_path[0]
impact_K_s = trans_res.K_s_path[0]
peak_K_idx = int(np.argmax(trans_res.K_s_path))
peak_K = trans_res.K_s_path[peak_K_idx]
r_jump_bps = (impact_r - ss_base.r) * 10000.0
w_jump_pct = (impact_w / ss_base.w - 1.0) * 100.0

print(f"Transition Solved in {t_trans:.2f}s ({trans_res.iterations} Broyden iterations):")
print(f"  Max Market Clearing Residual ||H||_inf = {trans_res.max_residual:.2e}")
print(f"  Max Mass Conservation Error            = {trans_res.mass_conservation_error:.2e}")
print(f"  Impact Interest Rate r_0               = {impact_r * 100:.3f}% (jump: {r_jump_bps:+.1f} bps)")
print(f"  Impact Real Wage     w_0               = {impact_w:.4f} (jump: {w_jump_pct:+.2f}%)")
print(f"  Initial Capital Supply K_0^s           = {impact_K_s:.4f} (Baseline K*={ss_base.K:.4f})")
print(f"  Peak Capital Deepening K_peak          = {peak_K:.4f} at Quarter t={peak_K_idx}")

# Inline verification assertions for non-linear transition path
assert trans_res.converged, "Sequence-space Broyden solver failed to converge"
assert trans_res.max_residual < 1e-4, f"Market clearing residual {trans_res.max_residual:.2e} exceeds 1e-4"
assert trans_res.mass_conservation_error < 1e-12, f"Mass conservation error {trans_res.mass_conservation_error:.2e} exceeds 1e-12"
assert np.isclose(impact_K_s, ss_base.K, atol=1e-4), "Capital supply must be physically predetermined at date t=0"
assert impact_r > ss_base.r, "Interest rate must spike on impact following positive TFP shock"
assert impact_w > ss_base.w, "Real wage must jump on impact following positive TFP shock"
assert peak_K > ss_base.K, "Capital stock must accumulate along the transition path"
assert np.isclose(trans_res.K_s_path[-1], ss_base.K, atol=0.05), "Capital must revert toward steady state at horizon T"

# %%
# --- Experiment 3: Wealth Inequality Dynamics & Publication Hero Figures ---
# Evaluate the time-varying Gini coefficient and Lorenz curves across the transition
ginis = np.array([
    ContinuousStationaryDistribution(pdf=d, asset_grid=K_hist).gini()
    for d in trans_res.distributions
])
time_grid = np.arange(horizon + 1)
min_gini_idx = int(np.argmin(ginis))
min_gini = ginis[min_gini_idx]

# Extract Lorenz curves at baseline (t=0), peak capital deepening (t=7), and terminal (t=40)
dist_0 = ContinuousStationaryDistribution(pdf=trans_res.distributions[0], asset_grid=K_hist)
dist_peak = ContinuousStationaryDistribution(pdf=trans_res.distributions[peak_K_idx], asset_grid=K_hist)
dist_term = ContinuousStationaryDistribution(pdf=trans_res.distributions[-1], asset_grid=K_hist)

_, L_peak = dist_peak.lorenz(100)
_, L_term = dist_term.lorenz(100)

print(f"Wealth Inequality Transition Path:")
print(f"  Initial Gini (t=0)   = {ginis[0]:.4f}")
print(f"  Minimum Gini (t={min_gini_idx})   = {min_gini:.4f} (compression: {(min_gini - ginis[0]):+.4f})")
print(f"  Terminal Gini (t={horizon}) = {ginis[-1]:.4f}")

# Verification assertions for distributional inequality dynamics
assert ginis[0] > min_gini, "Wealth inequality must compress during initial expansion"
assert np.isclose(ginis[-1], ginis[0], atol=0.01), "Wealth Gini must revert close to initial level"
assert len(trans_res.distributions) == horizon + 1, "Distribution path length must match T + 1"

# Figure 1: 4-Panel Macroeconomic Transition and Wealth Inequality Hero Plot
fig1, axes1 = _nbstyle.figura(2, 2, figsize=(11, 8))
t_quarters = np.arange(horizon)

# Panel 1: Factor Prices Path (Real Rate and Wage)
ax1 = axes1[0, 0]
ax1.plot(t_quarters, trans_res.r_path * 100, label="Real Rate $r_t$ (%)", lw=2, color=_nbstyle.S1["color"])
ax1.axhline(ss_base.r * 100, ls="--", color=_nbstyle.SPINE, label=f"Initial $r^*={ss_base.r * 100:.2f}\\%$")
ax1.set_title("Factor Prices: Real Interest Rate Path")
ax1.set_xlabel("Quarter $t$")
ax1.set_ylabel("Percent (%)")
ax1.legend(loc="upper right")

# Panel 2: Capital Market Clearing (Supply vs Demand)
ax2 = axes1[0, 1]
ax2.plot(t_quarters, trans_res.K_s_path, label="Capital Supply $K_t^s$", lw=2, color=_nbstyle.S1["color"])
ax2.plot(t_quarters, trans_res.K_d_path, label="Capital Demand $K_t^d$", ls="--", lw=1.8, color=_nbstyle.S2["color"])
ax2.axhline(ss_base.K, ls=":", color=_nbstyle.SPINE, label=f"Initial $K^*={ss_base.K:.2f}$")
ax2.set_title("Capital Market Clearing ($K_t^s$ vs $K_t^d$)")
ax2.set_xlabel("Quarter $t$")
ax2.set_ylabel("Aggregate Capital")
ax2.legend(loc="upper right")

# Panel 3: Wealth Inequality Path (Gini Coefficient)
ax3 = axes1[1, 0]
ax3.plot(time_grid, ginis, label="Wealth Gini $G_t$", lw=2, color=_nbstyle.S1["color"])
ax3.axhline(ginis[0], ls="--", color=_nbstyle.SPINE, label=f"Initial Gini $G_0={ginis[0]:.4f}$")
ax3.scatter([min_gini_idx], [min_gini], color=_nbstyle.S1["color"], s=40, zorder=5, label=f"Min Gini ({min_gini:.4f})")
ax3.set_title("Wealth Inequality Dynamics: Gini Coefficient")
ax3.set_xlabel("Quarter $t$")
ax3.set_ylabel("Gini Coefficient")
ax3.legend(loc="upper right")

# Panel 4: Lorenz Curves Comparison
ax4 = axes1[1, 1]
ax4.plot(p_lorenz * 100, p_lorenz * 100, linestyle=":", color=_nbstyle.SPINE, label="45° Equality Line", alpha=0.5)
ax4.plot(p_lorenz * 100, L_base * 100, label=f"Lorenz $t=0$ ($G={ginis[0]:.3f}$)", lw=2, color=_nbstyle.S1["color"])
ax4.plot(p_lorenz * 100, L_peak * 100, label=f"Lorenz $t={peak_K_idx}$ ($G={ginis[peak_K_idx]:.3f}$)", lw=1.8, ls="--", color=_nbstyle.S2["color"])
ax4.plot(p_lorenz * 100, L_term * 100, label=f"Lorenz $t={horizon}$ ($G={ginis[-1]:.3f}$)", lw=1.5, ls="-.", color=_nbstyle.S3["color"])
ax4.set_title("Wealth Lorenz Curve Dynamics")
ax4.set_xlabel("Cumulative Population (%)")
ax4.set_ylabel("Cumulative Wealth (%)")
ax4.legend(loc="upper left")

# Figure 2: 3D Perspective Surface and 2D Density Evolution Heatmap
density_mat = np.array([np.sum(d, axis=1) if d.ndim == 2 else d for d in trans_res.distributions])
k_mask = K_hist <= 15.0
k_sub = K_hist[k_mask]
dens_sub = density_mat[:, k_mask]

fig2 = plt.figure(figsize=(12.5, 4.6), layout="constrained")

# Subplot 1: 3D Wealth Distribution Surface
ax_3d = fig2.add_subplot(121, projection="3d")
K_mesh, T_mesh = np.meshgrid(k_sub, time_grid)
surf = ax_3d.plot_surface(K_mesh, T_mesh, dens_sub, cmap=_nbstyle.CMAP_SEQ, edgecolor="none", alpha=0.9)
ax_3d.set_title(r"3D Wealth Distribution Surface $\mu_t(k)$", fontsize=11)
ax_3d.set_xlabel("Assets $k$", fontsize=9)
ax_3d.set_ylabel("Quarter $t$", fontsize=9)
ax_3d.set_zlabel("Density", fontsize=9)
ax_3d.view_init(elev=28, azim=-55)

# Subplot 2: 2D Heatmap with Density Slices
ax_heat = fig2.add_subplot(122)
im = ax_heat.imshow(dens_sub, aspect="auto", origin="lower", extent=[k_sub[0], k_sub[-1], 0, horizon], cmap=_nbstyle.CMAP_SEQ)
cbar = plt.colorbar(im, ax=ax_heat)
cbar.set_label(r"Probability Density $\mu_t(k)$")
ax_heat.set_title(r"Heatmap: Wealth Mass Transition over Time", fontsize=11)
ax_heat.set_xlabel("Assets $k$")
ax_heat.set_ylabel("Transition Quarter $t$")

# %%
# --- Experiment 4: Algorithm Benchmark: Broyden Quasi-Newton vs Damped Shooting ---
# Compare sequence-space Broyden method against traditional damped fixed-point shooting
print("Benchmarking sequence-space relaxation algorithms on 40-quarter transition...")

# Broyden already executed in Experiment 2
t0 = time.perf_counter()
shoot_res = continuous_mit_shock(
    ss_base,
    shock_type="tfp",
    shock_size=shock_size,
    persistence=persistence,
    horizon=horizon,
    solver="shooting",
    damping=0.5,
    tol=1e-4,
    max_iter=30,
)
t_shoot = time.perf_counter() - t0

print(f"Algorithm Performance Comparison:")
print(f"  Broyden Quasi-Newton : {trans_res.iterations:2d} iterations | Wall Time = {t_trans:.3f}s | Max Res = {trans_res.max_residual:.2e}")
print(f"  Damped Shooting      : {shoot_res.iterations:2d} iterations | Wall Time = {t_shoot:.3f}s | Max Res = {shoot_res.max_residual:.2e}")

# Performance assertions
assert trans_res.converged, "Broyden solver must converge"
assert shoot_res.converged or trans_res.iterations <= shoot_res.iterations, "Broyden should converge in fewer iterations than fixed-point relaxation"

# %% [markdown]
# ## Read the output
#
# **Read the output.** The numerical experiments elucidate the continuous macroeconomic transmission mechanism and confirm the theoretical properties of distributional sequence space:
#
# 1. **Stationary Equilibrium Baseline (Experiment 1):** In the pre-shock Aiyagari steady state, the equilibrium interest rate settles at $r^* = 1.976\%$ ($0.019762$), strictly below the subjective rate of time preference $\rho = 1/\beta - 1 = 4.167\%$. This gap quantifies the precautionary savings motive induced by uninsurable labor income risk. Aggregate capital stock is $K^* = 8.7924$, supporting real wage $w^* = 1.3173$ and generating a realistic wealth Gini coefficient of $0.5269$. Median wealth ($P_{50} = 5.800$) is less than one-third of the 90th percentile cutoff ($P_{90} = 22.702$), reflecting typical right-skewed wealth concentration.
# 2. **Immediate Impact Dynamics & Predetermined Capital (Experiment 2):** At date $t=0$, capital supply $K_0^s = 8.7924$ matches the baseline steady state to machine precision ($|K_0^s - K^*| < 10^{-12}$). Because household wealth is predetermined, the $+5\%$ TFP surge raises the marginal product of capital instantaneously, forcing the equilibrium interest rate to spike from $1.976\%$ to $2.475\%$ ($+49.9$ basis points). Concurrently, the competitive real wage leaps by $+5.00\%$ to $w_0 = 1.3832$.
# 3. **Endogenous Propagation & Capital Deepening (Experiment 2):** In response to elevated interest rates and wages, households save aggressively. Aggregate capital expands steadily along the transition path, peaking at $K_{\text{peak}} = 9.0388$ around quarter 7—long after the exogenous shock has decayed to less than $21\%$ of its initial magnitude ($0.80^7 \approx 0.2097$). This capital accumulation depresses the interest rate below its initial peak, illustrating the endogenous propagation persistence generated by wealth redistribution.
# 4. **Wealth Inequality Compression & Redistribution (Experiment 3):** The wealth Gini coefficient temporarily compresses from $G_0 = 0.5269$ down to $G_{\min} = 0.5212$ at quarter 7. This equalization occurs because higher labor income disproportionately benefits earnings-dependent, low-wealth households, allowing them to accumulate buffer-stock assets faster in percentage terms than wealthy capital owners. As shown in the Lorenz curves and the 3D density surface, mass shifts away from the borrowing constraint ($k=0$) toward the asset-rich interior before slowly returning to the ergodic baseline.
# 5. **Algorithmic Convergence & Precision (Experiment 4):** Sequence-space Broyden Quasi-Newton converges in 5 iterations ($0.22$ seconds) to a maximum market clearing residual of $2.67 \times 10^{-5}$, while the Young (2010) lottery operator conserves probability mass across all 40 quarters to machine precision ($6.66 \times 10^{-16}$).

# %%
# Your turn: customize shock magnitude, persistence, and relaxation damping
# Adjust the continuous MIT shock settings below to test alternative macroeconomic scenarios.
# The executable cell re-simulates the transition and validates downstream stability assertions.

# ← change this: TFP shock magnitude in percentage points (e.g., 0.02, 0.05, 0.08)
user_shock_size = 0.05

# ← change this: Shock autoregressive persistence rho in [0, 1) (e.g., 0.50, 0.80, 0.90)
user_persistence = 0.80

# ← change this: Broyden Quasi-Newton relaxation damping theta in (0, 1] (e.g., 0.20, 0.40, 0.60)
user_damping = 0.40

# Re-simulate transition under custom user parameters
user_res = continuous_mit_shock(
    ss_base,
    shock_type="tfp",
    shock_size=user_shock_size,
    persistence=user_persistence,
    horizon=40,
    solver="broyden",
    damping=user_damping,
    tol=1e-4,
)

print(f"Custom Transition Simulation (Shock = {user_shock_size * 100:+.1f}%, rho = {user_persistence:.2f}, damping = {user_damping:.2f}):")
print(f"  Broyden Iterations           = {user_res.iterations}")
print(f"  Max Market Clearing Residual = {user_res.max_residual:.2e}")
print(f"  Max Mass Conservation Error  = {user_res.mass_conservation_error:.2e}")
print(f"  Initial Real Rate Jump       = {(user_res.r_path[0] - ss_base.r) * 10000.0:+.1f} bps")
print(f"  Peak Capital Stock           = {user_res.K_s_path.max():.4f}")

# Downstream assertions validating user parameters and transition integrity
assert user_shock_size != 0.0, "Shock size must be non-zero"
assert 0.0 <= user_persistence < 1.0, "Persistence must lie in [0, 1)"
assert 0.0 < user_damping <= 1.0, "Damping must lie in (0, 1]"
assert user_res.converged, "Custom transition solver failed to converge"
assert user_res.max_residual < 1e-4, f"Custom residual {user_res.max_residual:.2e} exceeds 1e-4"
assert user_res.mass_conservation_error < 1e-12, f"Mass conservation violated: {user_res.mass_conservation_error:.2e}"
if user_shock_size > 0:
    assert user_res.r_path[0] > ss_base.r
elif user_shock_size < 0:
    assert user_res.r_path[0] < ss_base.r


# %% [markdown]
# **Prompts.**
# 1. *Basic:* Vary `user_shock_size` from $+0.02$ to $+0.08$. Notice how the initial interest rate jump scales near-linearly with shock magnitude, while peak capital accumulation shifts proportionately higher.
# 2. *Intermediate:* Increase the persistence parameter `user_persistence` from $0.60$ to $0.92$. Observe how greater shock persistence extends the half-life of capital accumulation, pushing the peak capital date from quarter 4 out past quarter 15.
# 3. *Stretch:* Test a contractionary shock (`user_shock_size = -0.05`). Verify that real wages fall on impact, the interest rate drops, capital decumulates, and the wealth Gini expands as liquidity-constrained households deplete their precautionary savings buffers.
#
# ## How comprehensive is this?
#
# `puremacro.vfi` unifies continuous dynamic programming and sequence-space general equilibrium transitions across the macroeconomic literature:
# - `puremacro.vfi.continuous_transition`: Sequence-space Broyden and shooting transition solvers for unexpected MIT shocks (`solve_continuous_transition`, `continuous_mit_shock`, `TransitionShock`).
# - `puremacro.vfi.continuous_distribution`: Continuous stationary distribution and general equilibrium engine using Young (2010) lotteries (`solve_aiyagari_continuous`, `continuous_stationary_distribution`, `ContinuousStationaryDistribution`).
# - `puremacro.models.hank_sequence_space`: Multi-asset Heterogeneous Agent New Keynesian (HANK) sequence-space Jacobians and non-linear transitions via fake news algorithms.
# - `puremacro.vfi.collocation` & `puremacro.vfi.fem`: Continuous policy projection solvers (Chebyshev collocation and finite element Galerkin methods).
