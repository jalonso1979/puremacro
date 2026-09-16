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
# # Continuous-Time HJB and Adjoint KFE: Implicit Upwind M-Matrix Solver and Continuous Aiyagari General Equilibrium
#
# **How do quantitative macroeconomists solve continuous-time heterogeneous-agent models without the severe time-step constraints of explicit stepping, and how does the adjoint Kolmogorov Forward Equation yield the exact stationary wealth distribution and general equilibrium prices?**
#
# In modern quantitative macroeconomics, continuous-time formulation of heterogeneous-agent models (Achdou, Han, Lasry, Lions & Moll 2022) provides profound analytical clarity and computational tractability. In contrast to discrete-time models where agents make decisions over lumpy periods, continuous-time agents adjust their balance sheets continuously subject to uninsurable Poisson earnings shocks and borrowing constraints ($a \ge 0$). However, solving continuous-time Hamilton-Jacobi-Bellman (HJB) equations with explicit finite differences requires vanishingly small time increments — the HJB here is a first-order drift equation, so the Courant-Friedrichs-Lewy (CFL) condition binds at $\Delta t \sim \mathcal{O}(\Delta a)$ — requiring thousands of steps and often diverging at borrowing kinks.
#
# The canonical implicit upwind finite-difference scheme circumvents the CFL condition by discretizing the infinitesimal generator into a sparse, diagonally dominant $M$-matrix. Because an $M$-matrix has a strictly non-negative inverse, the value function updates monotonically and unconditionally, achieving machine-precision convergence in 10 to 20 iterations. Furthermore, the stationary cross-sectional wealth distribution $g(a, z)$ is solved directly as the null space of the adjoint transpose generator ($A^T g = 0$), preserving total probability mass to machine precision ($\sim 10^{-16}$) without simulation noise. This notebook demonstrates the full continuous-time heterogeneous-agent pipeline: implicit HJB value policy iteration, adjoint KFE wealth distribution solving, analytical cake-eating validation, and continuous Aiyagari general equilibrium factor price determination.

# %% [markdown]
# ## The method in math — Continuous-Time HJB, Adjoint KFE, and Aiyagari General Equilibrium
#
# **1. Continuous-Time Hamilton-Jacobi-Bellman (HJB) Equation.** An infinitely-lived household with discount rate $\rho > 0$, CRRA utility $u(c) = \frac{c^{1-\gamma} - 1}{1-\gamma}$, and uninsurable income state $z_i \in \{z_1, \dots, z_K\}$ governed by Poisson jump intensity $\lambda_{ij}$ solves:
# $$ \rho V_i(a) = \max_{c} \left\{ u(c) + V_i'(a) \big(r a + w z_i - c\big) \right\} + \sum_{j=1}^K \lambda_{ij} V_j(a). $$
# First-order optimality yields consumption policy $c(a, z_i) = \big(V_i'(a)\big)^{-1/\gamma}$ and savings drift $s(a, z_i) = r a + w z_i - c(a, z_i)$.
#
# **2. Implicit Upwind Finite-Difference Discretization.** On a non-uniform or uniform asset mesh $a_1 < a_2 < \dots < a_{N_a}$, the upwind derivative selects forward difference $\partial_a^+ V$ if drift $s^F > 0$, backward difference $\partial_a^- V$ if drift $s^B < 0$, and stationarity if $s^F \le 0 \le s^B$. With transition generator matrix $A^n$, the implicit time step with step size $\Delta t$ is:
# $$ \left( \left(\rho + \frac{1}{\Delta t}\right) I - A^n \right) V^{n+1} = u(c^n) + \frac{1}{\Delta t} V^n. $$
# The sparse matrix $\mathcal{M} = (\rho + 1/\Delta t)I - A^n$ is a strictly diagonally dominant $M$-matrix with positive diagonals and non-positive off-diagonals, guaranteeing unconditional stability and monotonic convergence.
#
# **3. Adjoint Kolmogorov Forward Equation (KFE).** The stationary cross-sectional joint density of wealth and productivity $g(a, z)$ satisfies the adjoint operator of the Markov generator:
# $$ A^T g = 0 \quad \text{subject to} \quad \sum_{i=1}^{N_a} \sum_{j=1}^K g(a_i, z_j) \Delta a_i = 1, \quad g(a_i, z_j) \ge 0. $$
#
# **4. Continuous Aiyagari General Equilibrium.** Competitive firms hire aggregate capital $K^d$ and labor $L = \sum_j z_j \pi_j$ with Cobb-Douglas technology $Y = K^\alpha L^{1-\alpha}$ and depreciation $\delta$. Factor prices satisfy $r(K) = \alpha K^{\alpha-1} L^{1-\alpha} - \delta$ and $w(K) = (1-\alpha) K^\alpha L^{-\alpha}$. General equilibrium requires clearing the capital asset market:
# $$ \mathcal{E}(r) \equiv K^s(r) - K^d(r) = \sum_{i=1}^{N_a} \sum_{j=1}^K a_i g(a_i, z_j; r) \Delta a_i - L \left( \frac{r + \delta}{\alpha} \right)^{\frac{1}{\alpha - 1}} = 0. $$

# %% [markdown]
# ## Intuition
#
# **Intuition.** In continuous time, households adjust their savings continuously rather than in discrete quarterly or annual jumps. When a household faces uninsurable labor income shocks and a hard borrowing limit ($a \ge 0$), the value function exhibits strong curvature near the borrowing constraint: as assets approach zero, the marginal value of wealth $V'(a)$ rises steeply to prevent the household from drifting into the forbidden negative-asset territory.
#
# In an explicit numerical scheme, the time step $\Delta t$ must be chosen small enough that no probability mass or value propagates across more than one spatial grid cell per iteration. Transport along the savings drift $s(a, z)$ is a first-order advection, so when the asset grid is refined to resolve the borrowing kink ($\Delta a \to 0$) the explicit stability limit requires $\Delta t \le \Delta a / \max_{a, z} |s(a, z)|$ — linear in $\Delta a$, not quadratic — forcing thousands of tiny iterations and often triggering numerical oscillations. The implicit upwind scheme eliminates this bottleneck entirely. By evaluating the future value function implicitly through the sparse generator matrix $A$, each iteration solves a linear system $(\rho I - A) V^{n+1} = u(c^n)$ using fast sparse matrix factorizations. The $M$-matrix property guarantees that the inverse operator is strictly positive, preserving monotonicity and enabling convergence in fewer than 15 iterations.
#
# Furthermore, the continuous-time framework establishes a duality between the household's HJB value problem and the cross-sectional wealth distribution. While the value function flows backward in time via the infinitesimal generator $A$, the wealth density $g(a, z)$ flows forward via the adjoint operator $A^T$. The stationary wealth distribution is thus computed directly as the eigenvector associated with the zero eigenvalue of $A^T$, conserving total probability mass to machine precision ($\sim 10^{-16}$) without stochastic Monte Carlo simulation noise. In general equilibrium, the interest rate $r^*$ balances the aggregate precautionary capital accumulated by households against the marginal productivity of capital demanded by competitive firms.

# %%
# Preamble: import numerical libraries, plotting style, and continuous solvers
import sys
from pathlib import Path
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
import _nbstyle
_nbstyle.apply_style()

from puremacro.vfi import (
    solve_hjb_achdou,
    solve_kfe_achdou,
    solve_aiyagari_continuous_hjb,
    HJBSolution,
    AiyagariContinuousHJBResult,
)

# Set deterministic random seed for reproducibility
rng = np.random.default_rng(42)

# Global model calibration parameters
# Subjective discount rate
rho_val = 0.05
# Coefficient of relative risk aversion
gamma_r = 2.0
# Real interest rate and wage rate for partial equilibrium
r_rate = 0.03
w_rate = 1.0
# Capital share and depreciation rate for general equilibrium
alpha = 0.33
delta = 0.05
# Asset grid parameters
Na = 50
a_min = 0.0
a_max = 30.0

print(f"Continuous-Time Calibration: rho = {rho_val:.3f}, gamma = {gamma_r:.2f}, r = {r_rate:.3f}, w = {w_rate:.2f}")
print(f"Asset Grid: [{a_min:.1f}, {a_max:.1f}] with Na = {Na} points")

# %%
# --- Experiment 1: Implicit Upwind HJB Solver & M-Matrix Convergence ---
# Solve the continuous-time consumption-saving problem with Poisson income risk
t0 = time.perf_counter()
sol = solve_hjb_achdou(
    r_rate=r_rate,
    w_rate=w_rate,
    rho_val=rho_val,
    gamma_r=gamma_r,
    Na=Na,
    a_min=a_min,
    a_max=a_max,
    tol=1e-8,
    max_iter=100,
)
t_hjb = time.perf_counter() - t0

print(f"HJB Implicit Solver Results:")
print(f"  Converged          : {sol.converged}")
print(f"  Iterations         : {sol.n_iter} (expected <= 20)")
print(f"  Wall Time          : {t_hjb:.4f} s")
print(f"  Max Absolute Drift : {np.max(np.abs(sol.s_drift)):.4f}")
print(f"  Borrowing Drift    : s(0, z_low) = {sol.s_drift[0, 0]:.6e}, s(0, z_high) = {sol.s_drift[0, 1]:.6e}")

# Verify structural mathematical properties of the HJB solution
assert sol.converged, "Implicit HJB solver must converge"
assert sol.n_iter <= 20, f"Implicit HJB required {sol.n_iter} iterations (expected <= 20)"
assert sol.V.shape == (Na, 2), "Value function shape must be (Na, 2)"
assert sol.c_policy.shape == (Na, 2), "Consumption policy shape must be (Na, 2)"
# Borrowing constraint enforcement: drift at a_min cannot be negative
assert np.all(sol.s_drift[0, :] >= -1e-12), "Savings drift at borrowing limit must be non-negative"
# Value function monotonicity in assets: V'(a) > 0
assert np.all(np.diff(sol.V[:, 0]) > 0), "Value function must be strictly increasing in assets for low state"
assert np.all(np.diff(sol.V[:, 1]) > 0), "Value function must be strictly increasing in assets for high state"
# Value function monotonicity in productivity: V(a, z_high) > V(a, z_low)
assert np.all(sol.V[:, 1] > sol.V[:, 0]), "Higher productivity must yield strictly higher value"

# %%
# --- Experiment 2: Adjoint Kolmogorov Forward Equation (KFE) & Wealth Distribution ---
# Evaluate the stationary wealth distribution g(a, z) and mass conservation
da = sol.a_grid[1] - sol.a_grid[0]
total_mass = float(np.sum(sol.g_dist * da))
mass_residual = abs(total_mass - 1.0)

print(f"Adjoint KFE Stationary Distribution Results:")
print(f"  Total Probability Mass : {total_mass:.16f}")
print(f"  Mass Residual Error    : {mass_residual:.2e} (expected <= 1e-12)")
print(f"  Non-negativity check   : np.all(g >= 0) is {np.all(sol.g_dist >= 0.0)}")

# Compute wealth inequality metrics: aggregate capital and wealth distribution percentiles
a_expanded = sol.a_grid[:, None]
capital_supply = float(np.sum(a_expanded * sol.g_dist * da))
marginal_g_a = np.sum(sol.g_dist, axis=1) * da
cumulative_g_a = np.cumsum(marginal_g_a)

# Wealth percentiles: 25th, 50th (median), 75th, 90th
p25 = float(sol.a_grid[np.searchsorted(cumulative_g_a, 0.25)])
p50 = float(sol.a_grid[np.searchsorted(cumulative_g_a, 0.50)])
p75 = float(sol.a_grid[np.searchsorted(cumulative_g_a, 0.75)])
p90 = float(sol.a_grid[np.searchsorted(cumulative_g_a, 0.90)])

# Continuous Gini coefficient of wealth
cum_wealth = np.cumsum(sol.a_grid * marginal_g_a) / capital_supply
gini_wealth = float(1.0 - np.sum((cum_wealth[:-1] + cum_wealth[1:]) * np.diff(cumulative_g_a)))

print(f"Aggregate Wealth & Inequality Statistics:")
print(f"  Aggregate Capital Supply (Ks) : {capital_supply:.4f}")
print(f"  Wealth Gini Coefficient       : {gini_wealth:.4f}")
print(f"  Wealth Percentiles            : P25 = {p25:.2f}, P50 (Median) = {p50:.2f}, P75 = {p75:.2f}, P90 = {p90:.2f}")

# Distributional assertions
assert mass_residual <= 1e-12, f"KFE mass conservation error {mass_residual:.2e} exceeds 1e-12"
assert sol.mass_residual <= 1e-12, "Result object mass_residual must satisfy tolerance"
assert np.all(sol.g_dist >= 0.0), "Probability density must be strictly non-negative everywhere"
assert capital_supply > 0.0, "Aggregate capital supply must be positive"
assert 0.0 < gini_wealth < 1.0, "Gini coefficient must lie in (0, 1)"

# %%
# --- Experiment 3: Closed-Form Cake-Eating Analytical Validation ---
# Verify numerical HJB against exact analytical closed-form solution:
# Under r = 0, w = 0, the unconstrained consumption policy is c(a) = (rho / gamma) * a
rho_cake = 0.05
gamma_cake = 2.0
mu_cake = rho_cake / gamma_cake
a_min_cake, a_max_cake = 1.0, 5.0

sol_cake = solve_hjb_achdou(
    r_rate=0.0,
    w_rate=0.0,
    rho_val=rho_cake,
    gamma_r=gamma_cake,
    a_min=a_min_cake,
    a_max=a_max_cake,
    Na=200,
    tol=1e-8,
)
c_analytical = mu_cake * sol_cake.a_grid
rel_err_cake = float(np.max(np.abs(sol_cake.c_policy[:, 0] - c_analytical) / c_analytical))

print(f"Analytical Cake-Eating Validation:")
print(f"  HJB Iterations       : {sol_cake.n_iter}")
print(f"  Max Relative Error   : {rel_err_cake:.4e} (expected < 0.02)")

# Benchmark assertion
assert sol_cake.converged, "Cake-eating HJB must converge"
assert rel_err_cake < 0.02, f"Cake-eating relative error {rel_err_cake:.4e} exceeds 0.02"

# %%
# --- Experiment 4: Continuous Aiyagari General Equilibrium ---
# Solve for market-clearing equilibrium interest rate r* where Ks(r*) = Kd(r*)
t0_ge = time.perf_counter()
ge_res = solve_aiyagari_continuous_hjb(
    alpha=alpha,
    delta=delta,
    rho_val=rho_val,
    gamma_r=gamma_r,
    Na=40,
    a_max=25.0,
    tol_ge=1e-4,
    max_iter_ge=30,
)
t_ge = time.perf_counter() - t0_ge

print(f"Continuous Aiyagari General Equilibrium Results:")
print(f"  Converged              : {ge_res.converged}")
print(f"  Equilibrium Rate (r*)  : {ge_res.r_star:.6f} ({ge_res.r_star * 100:.3f}%)")
print(f"  Equilibrium Wage (w*)  : {ge_res.w_star:.4f}")
print(f"  Aggregate Capital (K*) : {ge_res.K_star:.4f}")
print(f"  Aggregate Labor (L*)   : {ge_res.L_star:.4f}")
print(f"  Excess Capital Supply  : {ge_res.excess_capital:.2e}")
print(f"  GE Wall Time           : {t_ge:.4f} s")

# General equilibrium assertions
assert ge_res.converged, "Aiyagari general equilibrium bisection must converge"
assert abs(ge_res.excess_capital) < 1e-4, f"Excess capital {ge_res.excess_capital:.2e} exceeds 1e-4"
assert 0.005 < ge_res.r_star < rho_val, "Equilibrium interest rate must satisfy 0 < r* < rho"
assert ge_res.K_star > 0.0, "Equilibrium capital must be strictly positive"

# Compute capital supply and demand curves across a grid of interest rates for hero figure.
# Firm demand must be scaled by the SAME aggregate labor supply L* the general
# equilibrium solver uses (L* = sum_j z_j pi_j, the stationary mean of the income
# process), otherwise K^d is the capital-labor ratio K/L and the plotted curves
# cross away from the equilibrium rate r* marked on the panel.
L_star = ge_res.L_star
r_grid = np.linspace(0.010, 0.035, 6)
ks_curve = []
kd_curve = []
for r_val in r_grid:
    k_over_l = (alpha / (r_val + delta)) ** (1.0 / (1.0 - alpha))
    w_val = (1.0 - alpha) * k_over_l ** alpha
    s_temp = solve_hjb_achdou(r_rate=r_val, w_rate=w_val, Na=40, a_max=25.0, tol=1e-6)
    da_t = s_temp.a_grid[1] - s_temp.a_grid[0]
    ks_val = float(np.sum(s_temp.a_grid[:, None] * s_temp.g_dist * da_t))
    kd_val = float(L_star * k_over_l)
    ks_curve.append(ks_val)
    kd_curve.append(kd_val)

# The sampled curves must bracket the equilibrium rate the solver returned.
excess_curve = np.asarray(ks_curve) - np.asarray(kd_curve)
r_cross = float(np.interp(0.0, excess_curve, r_grid))
print(f"Capital Market Clearing Curves (L* = {L_star:.4f}):")
print(f"  Sampled Crossing Rate  : {r_cross:.6f} ({r_cross * 100:.3f}%)")
print(f"  Solver Equilibrium r*  : {ge_res.r_star:.6f} ({ge_res.r_star * 100:.3f}%)")
assert abs(r_cross - ge_res.r_star) < 5e-4, "Plotted Ks/Kd crossing must agree with the solver's r*"

# %%
# --- Hero Visualizations: Policy, Drift, Distribution, and Market Clearing ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Value Functions and Consumption Policies
ax1 = axes[0, 0]
ax1.plot(sol.a_grid, sol.c_policy[:, 0], color="black", linestyle="-", label=r"Consumption $c(a, z_{\mathrm{low}})$")
ax1.plot(sol.a_grid, sol.c_policy[:, 1], color="black", linestyle="--", label=r"Consumption $c(a, z_{\mathrm{high}})$")
ax1.set_title("Optimal Consumption Policies by Income State", fontsize=11)
ax1.set_xlabel("Assets $a$")
ax1.set_ylabel("Consumption $c$")
ax1.legend(frameon=False)

# Subplot 2: Savings Drift and Borrowing Constraint Kink
ax2 = axes[0, 1]
ax2.plot(sol.a_grid, sol.s_drift[:, 0], color="black", linestyle="-", label=r"Drift $s(a, z_{\mathrm{low}})$")
ax2.plot(sol.a_grid, sol.s_drift[:, 1], color="black", linestyle="--", label=r"Drift $s(a, z_{\mathrm{high}})$")
ax2.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
ax2.set_title(r"Savings Drift $s(a, z) = r a + w z - c(a, z)$", fontsize=11)
ax2.set_xlabel("Assets $a$")
ax2.set_ylabel("Drift $s(a, z)$")
ax2.legend(frameon=False)

# Subplot 3: Stationary Wealth Distribution from Adjoint KFE
ax3 = axes[1, 0]
ax3.plot(sol.a_grid, sol.g_dist[:, 0], color="black", linestyle="-", label=r"Density $g(a, z_{\mathrm{low}})$")
ax3.plot(sol.a_grid, sol.g_dist[:, 1], color="black", linestyle="--", label=r"Density $g(a, z_{\mathrm{high}})$")
ax3.set_title("Stationary Wealth Distribution (Adjoint KFE)", fontsize=11)
ax3.set_xlabel("Assets $a$")
ax3.set_ylabel("Density $g(a, z)$")
ax3.legend(frameon=False)

# Subplot 4: General Equilibrium Capital Market Clearing
ax4 = axes[1, 1]
ax4.plot(r_grid * 100, ks_curve, color="black", linestyle="-", marker="o", label=r"Capital Supply $K^s(r)$")
ax4.plot(r_grid * 100, kd_curve, color="black", linestyle="--", marker="s", label=r"Firm Capital Demand $K^d(r)$")
ax4.axvline(ge_res.r_star * 100, color="gray", linestyle=":", label=f"Equilibrium $r^* = {ge_res.r_star * 100:.2f}\\%$")
ax4.set_title("Aiyagari Asset Market Clearing Equilibrium", fontsize=11)
ax4.set_xlabel("Interest Rate $r$ (%)")
ax4.set_ylabel("Aggregate Capital $K$")
ax4.legend(frameon=False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Read the output
#
# **Read the output.** The computational results illustrate the mathematical and economic mechanisms of continuous-time heterogeneous-agent modeling:
#
# 1. **Implicit Solver Efficiency & Convergence (Experiment 1):** The canonical implicit upwind scheme converges to tolerance $10^{-8}$ in exactly 8 iterations, in a few milliseconds of wall time. The $M$-matrix structure completely bypasses the Courant-Friedrichs-Lewy (CFL) constraint. At the borrowing boundary $a = 0$, the savings drift strictly satisfies $s(0, z) \ge 0$ — the printed value is $s(0, z_{\mathrm{low}}) = 0$ exactly — verifying that households never violate the borrowing constraint.
# 2. **Exact Adjoint Density & Mass Conservation (Experiment 2):** The stationary wealth distribution $g(a, z)$ solved from the transpose generator $A^T g = 0$ integrates to $1.0000000000000000$ against the grid's quadrature weights, a mass residual of $0.00 \times 10^{0}$ — exact to the last bit of double precision, because the solver renormalizes the node mass after the sparse solve. The density peaks exactly at the borrowing limit $a = 0$ in the low-productivity state and is hump-shaped in the high-productivity state, peaking well inside the grid; the resulting aggregate wealth Gini coefficient is $0.3542$.
# 3. **Analytical Benchmark Precision (Experiment 3):** On the unconstrained cake-eating problem ($r=0, w=0$), the numerical policy matches the closed-form analytical consumption rule $c(a) = (\rho/\gamma)a$ with a maximum relative error of $0.0073$ ($0.73\%$), confirming high numerical accuracy on smooth domains.
# 4. **General Equilibrium Asset Market Clearing (Experiment 4):** The continuous Aiyagari general equilibrium root solve converges in a few hundredths of a second to market-clearing interest rate $r^* = 1.888\%$ ($0.018879$), real wage $w^* = 1.4495$ and aggregate capital $K^* = 6.2190$. Firm demand in the market-clearing panel is $K^d(r) = L^* \left( \alpha / (r + \delta) \right)^{1/(1-\alpha)}$ evaluated at the same aggregate labor supply the solver uses, $L^* = \sum_j z_j \pi_j = 0.6$; the notebook interpolates the sampled excess-demand curve and asserts that its zero sits within $5 \times 10^{-4}$ of $r^*$, so the plotted crossing and the dotted $r^*$ line agree. As predicted by macroeconomic theory, $r^* < \rho = 5.00\%$ because households accumulate precautionary buffer-stock savings against uninsurable idiosyncratic earnings risk, driving the equilibrium capital stock above the complete-markets level.

# %%
# Your turn: customize discount rate, risk aversion, and grid resolution
# Modify the structural parameters below to investigate how household patience
# and risk aversion reshape the stationary wealth distribution and policy functions.

# ← change this: household subjective discount rate rho (e.g. 0.04, 0.05, 0.06)
rho_custom = 0.05

# ← change this: coefficient of relative risk aversion gamma (e.g. 1.5, 2.0, 3.0)
gamma_custom = 2.0

# ← change this: number of wealth grid points Na (e.g. 30, 50, 80)
Na_custom = 50

# ← change this: maximum wealth bound a_max (e.g. 20.0, 30.0, 40.0)
a_max_custom = 30.0

# Solve implicit HJB under custom parameters
sol_custom = solve_hjb_achdou(
    r_rate=0.03,
    w_rate=1.0,
    rho_val=rho_custom,
    gamma_r=gamma_custom,
    Na=Na_custom,
    a_min=0.0,
    a_max=a_max_custom,
    tol=1e-8,
    max_iter=100,
)

da_custom = sol_custom.a_grid[1] - sol_custom.a_grid[0]
mass_custom = float(np.sum(sol_custom.g_dist * da_custom))
ks_custom = float(np.sum(sol_custom.a_grid[:, None] * sol_custom.g_dist * da_custom))

print(f"Custom Model Solution (rho = {rho_custom:.3f}, gamma = {gamma_custom:.1f}, Na = {Na_custom}):")
print(f"  Converged           : {sol_custom.converged}")
print(f"  Iterations          : {sol_custom.n_iter}")
print(f"  Mass Conservation   : {mass_custom:.16f} (residual = {abs(mass_custom - 1.0):.2e})")
print(f"  Aggregate Capital Ks: {ks_custom:.4f}")

# Downstream assertions verifying user parameter consistency and solver stability
assert 0.01 <= rho_custom <= 0.15, "Discount rate rho must be reasonable"
assert 1.0 <= gamma_custom <= 5.0, "Risk aversion gamma must be in [1.0, 5.0]"
assert Na_custom >= 20, "Grid size must be at least 20"
assert a_max_custom > 5.0, "Upper wealth bound must be greater than 5"
assert sol_custom.converged, "Custom HJB solve must converge"
assert sol_custom.n_iter <= 25, f"Custom HJB iterations {sol_custom.n_iter} exceeded 25"
assert sol_custom.mass_residual <= 1e-12, f"Custom mass residual {sol_custom.mass_residual:.2e} exceeded 1e-12"
assert abs(mass_custom - 1.0) <= 1e-12, "Total probability mass must equal 1.0"
assert ks_custom > 0.0, "Custom aggregate capital must be strictly positive"

# %% [markdown]
# **Prompts.**
# 1. *Basic:* Increase risk aversion `gamma_custom` from $2.0$ to $3.0$. Observe how the precautionary savings motive strengthens, depressing consumption at low wealth and increasing aggregate capital accumulation $K^s$.
# 2. *Intermediate:* Raise the discount rate `rho_custom` from $0.05$ to $0.07$. Verify that less patient households hold fewer assets, shifting the stationary wealth distribution leftward toward the borrowing constraint.
# 3. *Stretch:* Refine the grid `Na_custom` from $50$ to $100$. Verify that the implicit linear solve scales linearly in memory and time while preserving total probability mass to machine precision ($\le 10^{-14}$).
#
# ## How comprehensive is this?
#
# `puremacro.vfi` provides a comprehensive continuous dynamic programming and equilibrium suite:
# - `puremacro.vfi.hjb_achdou`: Canonical implicit upwind M-matrix HJB solver, adjoint continuous KFE wealth distribution (`solve_hjb_achdou`, `solve_kfe_achdou`), and continuous Aiyagari general equilibrium (`solve_aiyagari_continuous_hjb`).
# - `puremacro.vfi.collocation`: Continuous Chebyshev orthogonal collocation with Bellman value iteration and Euler equation projection (`CollocationProblem`, `solve_collocation`).
# - `puremacro.vfi.fem`: Piecewise-linear Finite Element Method Galerkin projection with Fischer-Burmeister complementarity for borrowing constraints (`FEMProblem`, `solve_fem`).
# - `puremacro.vfi.continuous_transition`: Non-linear distributional transition dynamics under MIT shocks via sequence-space Broyden relaxation (`solve_continuous_transition`).
# - `puremacro.models.hank_sequence_space`: Multi-asset Heterogeneous Agent New Keynesian (HANK) sequence-space Jacobians and Fake-News algorithm.
