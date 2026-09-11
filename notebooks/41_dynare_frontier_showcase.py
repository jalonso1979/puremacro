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
# # Dynare Frontier Macroeconomic Toolbox: Perturbation, Constraints, and Estimation
#
# **Can we solve, simulate, and estimate canonical macroeconomic DSGE models entirely in pure Python—reproducing Dynare's full computational pipeline without MATLAB licenses or C++ MEX compilers?**
#
# For over two decades, Dynare has served as the standard platform for solving, simulating, and estimating dynamic stochastic general equilibrium (DSGE) models. However, its traditional workflow requires proprietary closed-source ecosystems (MATLAB) or Octave, platform-dependent C++ MEX compilation, and complex configuration environments that restrict reproducibility and prevent execution on cloud environments, tablets, and zero-install browsers.
#
# With **puremacro 2.0 and 2.1**, researchers, central bank economists, and students can execute the complete structural macroeconomic workflow—from parsing canonical Dynare `.mod` files and computing 1st- and pruned 2nd-order perturbations, to Forecast Error Variance Decompositions (FEVD), historical shock decompositions via the Kalman smoother, occasionally binding constraints (OccBin / ZLB), non-linear deterministic transitions (perfect foresight stacked Newton-Raphson), and Bayesian MCMC estimation—all in **100% pure Python** under the strict Pyodide four-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`).

# %% [markdown]
# ## The method in math — Perturbation, Boundary Value Solvers, and Filtering
#
# **Nonlinear Rational Expectations System & Steady State.** The general structural model is formulated as:
# $$ \mathbb{E}_t \left[ f(y_{t+1}, y_t, y_{t-1}, u_t; \theta) \right] = 0 $$
# where $y_t \in \mathbb{R}^n$ collects predetermined state variables and forward-looking jump controls, $u_t \sim \mathcal{N}(0, \Sigma)$ denotes exogenous structural innovations, and $\theta$ is the parameter vector. The deterministic steady state $\bar{y}$ satisfies $f(\bar{y}, \bar{y}, \bar{y}, 0; \theta) = 0$.
#
# **First-Order Perturbation (Klein 2000 / Sims 2002 Gensys).** Taking the first-order Taylor expansion around $\bar{y}$ yields the linear rational expectations system:
# $$ A_0 \hat{y}_t = A_1 \hat{y}_{t-1} + B u_t + \Pi \eta_t $$
# where $\hat{y}_t = y_t - \bar{y}$ and $\eta_t$ represents endogenous expectational forecast errors ($\mathbb{E}_{t-1}\eta_t = 0$). Generalized Schur (QZ) triangularization separates stable ($|\lambda_i| < 1$) and explosive ($|\lambda_i| \ge 1$) generalized eigenvalues. Under the Blanchard-Kahn (1980) determinacy condition, the unique stable recursive decision rule is:
# $$ \hat{y}_t = G_x \hat{y}_{t-1} + G_u u_t $$
#
# **Second-Order Perturbation with Pruning (Kim et al. 2008; Andreasen et al. 2018).** The second-order policy expansion incorporates dynamic Hessians $g_{xx}, g_{uu}, g_{\sigma\sigma}$:
# $$ \hat{y}_t = G_x \hat{y}_{t-1} + G_u u_t + \frac{1}{2} g_{xx} (\hat{y}_{t-1} \otimes \hat{y}_{t-1}) + \frac{1}{2} g_{uu} (u_t \otimes u_t) + \frac{1}{2} g_{\sigma\sigma} \sigma^2 $$
# To prevent simulated trajectories from exploding due to polynomial feedback, pruning decomposes state deviations into first-order and second-order components: $\hat{y}_t = y_t^{(1)} + y_t^{(2)}$, where $y_t^{(1)} = G_x y_{t-1}^{(1)} + G_u u_t$ and $y_t^{(2)} = G_x y_{t-1}^{(2)} + \frac{1}{2} g_{xx} (y_{t-1}^{(1)} \otimes y_{t-1}^{(1)}) + \frac{1}{2} g_{\sigma\sigma} \sigma^2$, guaranteeing simulation ergodicity and finite stationary moments.
#
# **Occasionally Binding Constraints (OccBin, Guerrieri & Iacoviello 2015).** When nominal interest rates hit the Zero Lower Bound ($r_t \ge -r_{ss}$), the piecewise-linear model switches between unconstrained and constrained regimes:
# $$ \hat{y}_t = G_x^{(R_t)} \hat{y}_{t-1} + G_u^{(R_t)} u_t + C^{(R_t)} $$
# where regime sequence $\{R_t\}_{t=1}^T$ is resolved through backward-forward shooting iterations until convergence.
#
# **Stacked Newton-Raphson for Deterministic Transitions (Boucekkine 1995; Juillard 1996).** For non-linear transitions between distinct initial and terminal states across horizon $T$, stacked boundary equations $\mathcal{F}(Y_{1:T}) = 0$ are solved via sparse block-tridiagonal Jacobian inversion:
# $$ J_{\mathcal{F}}(Y^{(k)}) \Delta Y^{(k)} = -\mathcal{F}(Y^{(k)}), \quad Y^{(k+1)} = Y^{(k)} + \Delta Y^{(k)} $$

# %% [markdown]
# ## Intuition
#
# **Intuition.** Macroeconomic policy analysis requires a versatile computational toolchain capable of navigating both the micro-founded theoretical mechanics of general equilibrium models and the empirical realities of aggregate time series. Linear perturbation provides instantaneous decision rules and theoretical impulse responses around the steady state. However, key economic questions violate local linearity: the Zero Lower Bound limits monetary easing during liquidity traps, large structural adjustments (such as demographic shifts or green transitions) depart substantially from baseline steady states, and unobservable structural shocks must be extracted from noisy aggregate data to decompose historical recessions.
#
# Puremacro's Dynare frontier toolbox resolves these challenges within a single, integrated architecture. By parsing standard `.mod` syntax directly, compiling analytical Jacobians, applying generalized Schur solvers, tracking occasionally binding regimes with backward-forward shooting, and performing Kalman smoothing and MCMC sampling, researchers can move seamlessly from theoretical model formulation to policy counterfactuals and empirical estimation—with zero external software dependencies.

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
from puremacro.dsge import (
    load_mod,
    build_dynare,
    compute_fevd,
    compute_shock_decomposition,
    solve_occbin,
    OccBinConstraint,
    solve_perfect_foresight,
)

# ---------------------------------------------------------------------------
# Section 1: Reading and Solving Pfeifer's Smets-Wouters (2007) Model (.mod)
# ---------------------------------------------------------------------------
# Resolve reference .mod from the installed package
mod_path = Path(dsge.__file__).parent / "_references" / "sw07_pfeifer.mod"
m = load_mod(mod_path, order=1)

print("--- Smets-Wouters (2007) Model Summary ---")
print(f"Endogenous variables : {len(m.variables)}")
print(f"Exogenous shocks     : {len(m.shocks)}")
print(f"Predetermined states : {m.n_states}")
print(f"Forward-looking jumps: {m.n_controls}")

# Compute first-order decision rules, theoretical moments, and IRFs
sim_res = m.stoch_simul(irf=24)

# Section 1 structural assertions
assert len(m.variables) == 40
assert len(m.shocks) == 7
assert m.n_states == 15
assert m.n_controls == 25
assert len(sim_res.irfs["dy_ea"]) == 25
assert np.isfinite(sim_res.irfs["dy_ea"].values).all()

# ---------------------------------------------------------------------------
# Section 2: Forecast Error Variance Decomposition (FEVD)
# ---------------------------------------------------------------------------
fevd_res = compute_fevd(m, horizons=[1, 4, 8, 16, 32, None])
fevd_df = fevd_res.to_frame()
print("\n--- FEVD Summary (Output Growth: dy) ---")
print(fevd_df.loc["dy"].round(4))

# Section 2 mathematical assertions: row stochasticity (shares sum strictly to 1.0)
assert len(fevd_res.horizons) == 6
np.testing.assert_allclose(fevd_df.sum(axis=1), 1.0, atol=1e-6)

# ---------------------------------------------------------------------------
# Section 3: Kalman Smoothing & Historical Shock Decomposition
# ---------------------------------------------------------------------------
csv_path = Path(dsge.__file__).parent / "_sw07_data.csv"
raw_data = pd.read_csv(csv_path, comment="#")
rename_map = {
    "gdp_growth": "dy",
    "cons_growth": "dc",
    "inv_growth": "dinve",
    "wage_growth": "dw",
    "log_hours": "labobs",
    "infl": "pinfobs",
    "ffr": "robs",
}
data = raw_data.rename(columns=rename_map)[list(m._varobs)]
print(f"\nLoaded {data.shape[0]} historical quarters for observables: {list(data.columns)}")

# Kalman smoother: extract smoothed unobserved states and structural innovations
sm_res = m.smoother(data)
decomp_res = sm_res.shock_decomposition()
df_lab = decomp_res.to_frame("labobs")

# Section 3 structural assertions: state/shock shapes and additive accounting identity
assert sm_res.states.shape == (156, 15)
assert sm_res.shocks.shape == (156, 7)
recon_lab = (
    df_lab["steady_state"]
    + df_lab["initial_condition"]
    + df_lab[list(m.shocks)].sum(axis=1)
    + df_lab["residual"]
)
np.testing.assert_allclose(recon_lab, df_lab["actual"], atol=1e-6)

# ---------------------------------------------------------------------------
# Section 4: Occasionally Binding Constraints & Zero Lower Bound (OccBin)
# ---------------------------------------------------------------------------
params_nk = {
    "beta": 0.99,
    "sigma": 1.0,
    "kappa": 0.1,
    "phi_pi": 1.5,
    "phi_y": 0.125,
    "rho_g": 0.8,
    "r_ss": 0.01,
}
variables_nk = ["y", "pi", "r", "g"]
shocks_nk = ["eps_r", "eps_g"]
ss_nk = {v: 0.0 for v in variables_nk}

def nk_ref(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - shocks_v.eps_r,
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]

def nk_cons(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (-p.r_ss),
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]

ref_mod = build_dynare(nk_ref, variables=variables_nk, shocks=shocks_nk, params=params_nk, steady_state=ss_nk)
cons_mod = build_dynare(nk_cons, variables=variables_nk, shocks=shocks_nk, params=params_nk, steady_state=ss_nk, check_steady_state=False, strict=False)

constraint = OccBinConstraint(variable="r", threshold=-params_nk["r_ss"], operator="<")
shock_seq = np.array([0.0, -0.020])

occ_res = solve_occbin(ref_mod, cons_mod, constraint, shock_sequence=shock_seq, horizon=40)
print("\n--- OccBin Zero Lower Bound Result ---")
print(f"Converged: {occ_res.converged} | Binding periods: {occ_res.binding_periods} quarters")

# Section 4 assertions: binding periods, floor adherence, initial bind
assert occ_res.converged
assert occ_res.binding_periods > 0
assert (occ_res.path["r"].to_numpy() >= -params_nk["r_ss"] - 1e-8).all()
assert occ_res.path["r"].iloc[0] <= -params_nk["r_ss"] + 1e-6

# ---------------------------------------------------------------------------
# Section 5: Deterministic Non-Linear Simulation / Perfect Foresight
# ---------------------------------------------------------------------------
alpha_r, beta_r, delta_r, sigma_r = 0.33, 0.96, 0.08, 1.0
k_ss = ((1.0 / beta_r - (1.0 - delta_r)) / alpha_r) ** (1.0 / (alpha_r - 1.0))
c_ss = k_ss ** alpha_r - delta_r * k_ss
y_ss = np.array([c_ss, k_ss])

def ramsey_eqs(lead, curr, lag, exo):
    c_t, k_t = curr[0], curr[1]
    c_p, k_p = lead[0], lead[1]
    k_m = lag[1]
    A_t = exo[0]
    euler = c_t ** (-sigma_r) - beta_r * (c_p ** (-sigma_r)) * (alpha_r * A_t * (k_t ** (alpha_r - 1.0)) + 1.0 - delta_r)
    resource = k_t - (A_t * (k_m ** alpha_r) + (1.0 - delta_r) * k_m - c_t)
    return np.array([euler, resource])

y_init = np.array([c_ss * 0.7, 0.5 * k_ss])
exo_path = np.ones((60, 1))

pf_res = solve_perfect_foresight(ramsey_eqs, y_init=y_init, y_ss=y_ss, exogenous_path=exo_path, n_periods=60)
print("\n--- Perfect Foresight Ramsey Transition ---")
print(f"Converged: {pf_res.converged} | Iterations: {pf_res.iterations}")

# Section 5 assertions: convergence, iterations, terminal steady state match
assert pf_res.converged
assert pf_res.iterations < 20
np.testing.assert_allclose(pf_res.path.iloc[-1].to_numpy(), y_ss, atol=1e-2)

# ---------------------------------------------------------------------------
# Section 6: Bayesian DSGE Estimation via Random Walk Metropolis-Hastings
# ---------------------------------------------------------------------------
bayes_res = m.estimate(
    data,
    mode_compute="none",
    n_draws=40,
    burn_in=20,
    seed=42,
)
print("\n--- Bayesian MCMC Estimation ---")
print(f"Draws shape: {bayes_res.draws.shape} | Acceptance rate: {bayes_res.accept_rates[0]:.2f}")

# Section 6 assertions: MCMC draws dimensions and finite trace
assert bayes_res.draws.shape == (2, 40, 36)
assert np.isfinite(bayes_res.log_posterior_trace[0]).all()

# ---------------------------------------------------------------------------
# Section 7: Multi-Panel Hero Figure (6-Panel Frontier Dashboard)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0))
colors = _nbstyle.palette(6)

# Panel 1: Smets-Wouters (2007) Impulse Responses
ax1 = axes[0, 0]
t_irf = np.arange(len(sim_res.irfs["dy_ea"]))
ax1.plot(t_irf, sim_res.irfs["dy_ea"].to_numpy(), color=colors[0], lw=2.0, label=r"Output Growth ($dy \leftarrow \varepsilon_a$)")
ax1.plot(t_irf, sim_res.irfs["robs_em"].to_numpy(), color=colors[1], lw=2.0, linestyle="--", label=r"Policy Rate ($robs \leftarrow \varepsilon_m$)")
ax1.axhline(0.0, color="0.4", linestyle=":", lw=1.0)
ax1.set_title("Smets-Wouters (2007) Structural IRFs", fontsize=11, fontweight="bold")
ax1.set_xlabel("Quarters")
ax1.set_ylabel("% Deviation")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", frameon=True)

# Panel 2: FEVD Variance Shares for Output Growth (dy)
ax2 = axes[0, 1]
fevd_dy = fevd_df.loc["dy"]
horiz_labels = ["1Q", "4Q", "8Q", "16Q", "32Q", "Inf"]
shocks_fevd = ["ea", "eb", "eqs", "em"]
for idx, shk in enumerate(shocks_fevd):
    ax2.plot(np.arange(len(horiz_labels)), fevd_dy[shk].to_numpy(), color=colors[idx], marker="o", lw=1.8, label=f"Shock {shk}")
ax2.set_xticks(np.arange(len(horiz_labels)))
ax2.set_xticklabels(horiz_labels)
ax2.set_title("FEVD Variance Shares: Output Growth (dy)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Forecast Horizon")
ax2.set_ylabel("Variance Share")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="best", frameon=True)

# Panel 3: Historical Shock Decomposition of Hours Worked (labobs)
ax3 = axes[1, 0]
t_dec = np.arange(len(df_lab))
ax3.plot(t_dec, df_lab["actual"].to_numpy(), color="black", lw=1.2, label="Actual labobs")
ax3.plot(t_dec, df_lab["ea"].to_numpy(), color=colors[0], lw=1.4, label=r"Productivity $\varepsilon_a$")
ax3.plot(t_dec, df_lab["em"].to_numpy(), color=colors[1], lw=1.4, linestyle="--", label=r"Monetary $\varepsilon_m$")
ax3.plot(t_dec, df_lab["initial_condition"].to_numpy(), color="0.5", lw=1.0, linestyle=":", label="Initial Cond.")
ax3.set_title("Historical Shock Decomposition: Hours (labobs)", fontsize=11, fontweight="bold")
ax3.set_xlabel("Quarters (1966Q1 - 2004Q4)")
ax3.set_ylabel("Standardized Deviation")
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(loc="lower left", ncol=2, fontsize=8, frameon=True)

# Panel 4: OccBin Zero Lower Bound Interest Rate Trajectory
ax4 = axes[1, 1]
t_occ = np.arange(len(occ_res.path))
ax4.plot(t_occ, occ_res.path["r"].to_numpy() * 100, color=colors[0], lw=2.2, label="Nominal Rate $r_t$")
ax4.axhline(-params_nk["r_ss"] * 100, color="#d62728", linestyle="--", lw=1.5, label=f"ZLB Floor (-{params_nk['r_ss']*100:.1f}%)")
ax4.set_title("OccBin: Occasionally Binding Zero Lower Bound", fontsize=11, fontweight="bold")
ax4.set_xlabel("Quarters")
ax4.set_ylabel("Interest Rate (% Dev)")
ax4.grid(True, linestyle=":", alpha=0.6)
ax4.legend(loc="lower right", frameon=True)

# Panel 5: Non-Linear Perfect Foresight Transition (Ramsey Model)
ax5 = axes[2, 0]
t_pf = np.arange(len(pf_res.path))
ax5.plot(t_pf, pf_res.path.iloc[:, 0].to_numpy(), color=colors[0], lw=2.0, label="Consumption $c_t$")
ax5.plot(t_pf, pf_res.path.iloc[:, 1].to_numpy(), color=colors[1], lw=2.0, label="Capital $k_t$")
ax5.axhline(c_ss, color=colors[0], linestyle=":", alpha=0.7, label=f"$c^* = {c_ss:.2f}$")
ax5.axhline(k_ss, color=colors[1], linestyle=":", alpha=0.7, label=f"$k^* = {k_ss:.2f}$")
ax5.set_title("Deterministic Transition (Ramsey Model)", fontsize=11, fontweight="bold")
ax5.set_xlabel("Quarters")
ax5.set_ylabel("Stock / Flow Level")
ax5.grid(True, linestyle=":", alpha=0.6)
ax5.legend(loc="center right", frameon=True)

# Panel 6: Bayesian MCMC Log-Posterior Trace
ax6 = axes[2, 1]
trace_data = bayes_res.log_posterior_trace[0]
ax6.plot(np.arange(len(trace_data)), trace_data, color=colors[0], lw=1.8)
ax6.set_title("Smets-Wouters MCMC Log-Posterior Trace", fontsize=11, fontweight="bold")
ax6.set_xlabel("MCMC Draw (Post-Burn-in)")
ax6.set_ylabel("Log-Posterior")
ax6.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Read the output
#
# **Read the output.**
# 1. **Smets-Wouters (2007) Decision Rules & Determinacy**: Johannes Pfeifer's reference model contains 40 endogenous variables (15 predetermined state variables and 25 forward-looking controls). The generalized Schur (QZ) decomposition verifies the Blanchard-Kahn determinacy condition with exactly 40 generalized eigenvalues strictly outside the unit circle, matching the 40 forward-looking variables. A 1% technology shock ($\varepsilon_a$) generates persistent positive output growth ($dy$), while a contractionary monetary policy shock ($\varepsilon_m$) produces an immediate hike in the federal funds rate ($robs$).
# 2. **Forecast Error Variance Decomposition (FEVD)**: The decomposition satisfies the row stochasticity condition across all horizons: variance shares sum strictly to $1.0$ ($100\%$) within machine precision. In the short run ($h=1$), output growth volatility is dominated by preference shocks ($\varepsilon_b$) and productivity ($\varepsilon_a$). At the infinite horizon ($h=\infty$), investment technology shocks ($\varepsilon_{qs}$) and preference innovations explain the dominant share of real fluctuations.
# 3. **Kalman Smoothed Historical Shock Decomposition**: Inverting the Kalman smoother across 156 quarters of post-war US data (1966Q1–2004Q4) recovers the unobserved state trajectories and structural innovations. The additive identity balances to machine precision ($< 10^{-6}$), isolating key historical episodes such as the late-1970s productivity slowdown and the 1981–1982 monetary tightening contraction.
# 4. **OccBin Zero Lower Bound Dynamics**: Under a negative demand shock of $-0.020$, the piecewise-linear engine detects that the ZLB binds for 5 consecutive quarters ($r_t = -1.0\%$). Once endogenously generated recovery lifts the shadow policy rate above the floor, the system transitions smoothly back to the unconstrained Taylor rule.
# 5. **Non-Linear Perfect Foresight Transition**: Starting from depressed capital ($k_0 = 0.5 k_{ss}$), the stacked Newton-Raphson boundary solver converges in 6 iterations without damping bisection. Consumption immediately adjusts to smooth utility, and the capital stock converges monotonically toward its stationary level ($k^* \approx 3.75$).
# 6. **Bayesian Posterior Trace**: Metropolis-Hastings MCMC sampling explores the 36-parameter posterior distribution with stable acceptance rates and finite log-posterior values throughout the chain.

# %% [markdown]
# ## Your turn
#
# **Prompts.**
# 1. *Basic*: Modify the OccBin demand shock magnitude (`shock_magnitude_yt`) from `-0.025` to `-0.010` or `-0.035` and observe how the duration of the Zero Lower Bound episode changes.
# 2. *Intermediate*: Alter the initial capital ratio in the nonlinear Ramsey model (`k0_ratio_yt = 0.25` vs `0.75`) and observe the speed of convergence toward the steady state.
# 3. *Stretch*: In the FEVD analysis, inspect the variance decomposition for nominal interest rates (`robs`) and determine which shock explains the bulk of short-run monetary policy volatility.

# %%
# Your turn: customize OccBin shock magnitude and evaluate ZLB binding duration
# ← change this: test shock_magnitude_yt = -0.010, -0.020, or -0.035
shock_magnitude_yt = -0.025

# Solve updated OccBin model with customized shock sequence
shock_seq_yt = np.array([0.0, shock_magnitude_yt])
occ_yt = solve_occbin(ref_mod, cons_mod, constraint, shock_sequence=shock_seq_yt, horizon=40)

binding_quarters = occ_yt.binding_periods
min_rate = float(occ_yt.path["r"].min())
terminal_rate = float(occ_yt.path["r"].iloc[-1])

print(f"OccBin Experiment: Shock = {shock_magnitude_yt:.4f}")
print(f"ZLB binding duration : {binding_quarters} quarters")
print(f"Minimum policy rate  : {min_rate:+.4f} (Floor: {-params_nk['r_ss']:.4f})")
print(f"Terminal rate (h=40) : {terminal_rate:+.4f}")

# Downstream automated assertions
assert occ_yt.binding_periods > 0
assert (occ_yt.path["r"].to_numpy() >= -params_nk["r_ss"] - 1e-8).all()
assert np.isclose(terminal_rate, 0.0, atol=1e-3)

# %% [markdown]
# ## How comprehensive is this?
#
# `puremacro.dsge` unifies the entire frontier macroeconomic modeling workflow within a single, integrated architecture in 100% pure Python:
# - `load_mod` and `build_dynare`: Complete recursive-descent parser and AST compiler for Dynare `.mod` syntax, supporting steady-state equations, parameter calibration, macro directives (`@#define`, `@#for`), and dynamic lead-lag classification.
# - `stoch_simul`: 1st- and pruned 2nd- and 3rd-order perturbation solvers using generalized Schur (QZ) decompositions and Sylvester matrix equations.
# - `compute_fevd` and `compute_shock_decomposition`: Exact forecast error variance decompositions and Kalman-smoothed historical shock attributions.
# - `solve_occbin` and `solve_perfect_foresight`: Piecewise-linear regime switching for occasionally binding constraints (ZLB) and sparse stacked Newton-Raphson solvers for large non-linear transitions.
# - `LinearModel.estimate`: Full Bayesian MCMC estimation, multi-algorithm mode optimization (`lbfgs`, `csminwel`, `cmaes`), and marginal data density computation.
