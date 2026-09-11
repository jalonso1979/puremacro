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
# # Frontier DSGE: Particle Filtering, Stochastic Volatility, and Markov-Switching
#
# **How can we accurately evaluate the likelihood of nonlinear DSGE models under time-varying uncertainty, solve rational expectations models subject to recurring regime shifts, and perform gradient-based inference under occasionally binding constraints?**
#
# Linear approximation around a deterministic steady state has long been the workhorse of macroeconomics. However, linear models are fundamentally incapable of capturing four crucial macroeconomic phenomena:
# 1. **Time-Varying Macroeconomic Risk and Uncertainty**: Uncertainty shocks (Bloom 2009; Fernández-Villaverde et al. 2011) alter precautionary behavior, risk premia, and capital accumulation. Under **Stochastic Volatility (SV)** ($\sigma_t = \bar{\sigma}\exp(h_t)$), the Kalman filter breaks down because the state-space is inherently non-Gaussian and non-linear.
# 2. **Discrete Structural Regime Shifts**: Monetary and fiscal policy periodically undergo profound regime changes—such as shifts between active monetary / passive fiscal (Hawkish) and passive monetary / active fiscal (Dovish) regimes (Leeper 1991; Foerster et al. 2016).
# 3. **Occasionally Binding Constraints (Zero Lower Bound)**: The nominal interest rate cannot fall below the effective lower bound ($i_t \ge 0$). While discrete piecewise-linear algorithms (OccBin, Guerrieri & Iacoviello 2015) handle simulations, their nondifferentiability historically prohibited gradient-based Bayesian samplers like NUTS.
# 4. **Multi-Asset Household Heterogeneity**: Wealthy households hold both high-return illiquid assets (housing, retirement equity) and low-return liquid deposits (cash, checking accounts) subject to transaction adjustment costs $\chi(d, a)$ (Kaplan, Moll & Violante 2018).
#
# In **puremacro 3.1.0**, these computational frontiers are unified into a cohesive, high-performance toolkit:
# - **Vectorized Sequential Monte Carlo Particle Filtering** (Gordon et al. 1993; Fernández-Villaverde & Rubio-Ramírez 2007) with systematic resampling, fat tails, and stochastic volatility.
# - **Markov-Switching DSGE Perturbation** (Foerster, Rubio-Ramírez, Waggoner & Zha 2016) with coupled quadratic matrix iterations, Mean-Square Stability (MSS) checks, and closed-form analytical Generalized Impulse Response Functions (GIRF).
# - **Differentiable OccBin with Smooth Relaxation** ($\text{smin}_\tau, \text{smax}_\tau$) and Fischer-Burmeister complementary conditions, enabling smooth gradients $\nabla_\theta$ for NUTS HMC under ZLB constraints.
# - **Two-Asset HANK Sequence-Space Bridge** integrating Dynare `.mod` files with 2D wealth distributions $\mathcal{D}^*(a, b)$ and Fake-News Jacobians.
#
# All methods run in **100% pure Python** under the strict **Pyodide four-package contract** (`numpy`, `scipy`, `pandas`, `matplotlib`).

# %% [markdown]
# ## The Method in Math: Nonlinear Filtering and Regime Transitions
#
# **1. Sequential Monte Carlo Particle Filtering.** Consider the state-space system with 2nd-order pruned perturbation and stochastic volatility:
# $$ x_t = f(x_{t-1}, \sigma_t \epsilon_t), \quad y_t = g(x_t) + v_t, \quad \ln \sigma_t = (1 - \rho_h) \ln \bar{\sigma} + \rho_h \ln \sigma_{t-1} + \sigma_h \eta_t $$
# The Bootstrap Particle Filter (BPF) propagates an empirical swarm of $N$ particles $\{x_t^{(i)}\}_{i=1}^N$ with normalized importance weights $w_t^{(i)} \propto w_{t-1}^{(i)} p(y_t \mid x_t^{(i)})$. The marginal log-likelihood is estimated unbiasedly by:
# $$ \ln \hat{L}(Y_{1:T}) = \sum_{t=1}^T \ln \left( \sum_{i=1}^N w_{t-1}^{(i)} p(y_t \mid x_t^{(i)}) \right) $$
# Whenever the Effective Sample Size $ESS_t = 1 / \sum_{i=1}^N (w_t^{(i)})^2$ falls below $N/2$, systematic resampling draws child particles with $O(N)$ low-discrepancy stratification, eliminating weight degeneracy.
#
# **2. Markov-Switching DSGE (Foerster et al. 2016).** Let $s_t \in \{1, \dots, S\}$ follow an ergodic Markov chain with transition probability matrix $P = (p_{ij})$. The structural rational expectations system is:
# $$ A(s_t) \mathbb{E}_t [y_{t+1}] + B(s_t) y_t + C(s_t) y_{t-1} + D(s_t) \epsilon_t = 0 $$
# The Minimal State Variable (MSV) solution takes the regime-dependent feedback form:
# $$ y_t = T(s_t) y_{t-1} + R(s_t) \epsilon_t $$
# Substituting into equilibrium conditions yields $S$ coupled matrix quadratic equations:
# $$ B(i) T(i) + C(i) + A(i) \sum_{j=1}^S p_{ij} T(j) T(i) = 0, \quad \forall i = 1, \dots, S $$
# Solved via analytical block Newton-Raphson iterations. Stability is verified through the first moment matrix $M_1 = (P' \otimes I_n) \text{diag}(T(1), \dots, T(S))$ and second moment matrix $M_2 = (P' \otimes I_{n^2}) \text{diag}(T(1) \otimes T(1), \dots, T(S) \otimes T(S))$. The system is Mean-Square Stable (MSS) if and only if $\rho(M_2) < 1.0$.
#
# **3. Differentiable Smooth Relaxation for OccBin.** The complementary slackness condition:
# $$ \min(r_t - r_{\min}, \Phi(x_t)) = 0 $$
# is smoothed using the temperature-parameterized smooth-min operator:
# $$ \text{smin}_\tau(a, b) = -\tau \ln\left( e^{-a/\tau} + e^{-b/\tau} \right) $$
# As $\tau \to 0^+$, $\text{smin}_\tau(a, b) \to \min(a, b)$ with infinite-order differentiability ($C^\infty$). The continuous regime weight $w_t(\tau) \in (0, 1)$ interpolates the reference and constrained transition matrices, producing smooth analytical parameter gradients $\nabla_\theta \ln L$ for NUTS sampling.

# %% [markdown]
# ## Intuition
#
# **Intuition.** Nonlinear macroeconomic phenomena introduce severe computational breakdowns into standard linear DSGE toolkits.
#
# First, when an economy is subject to time-varying uncertainty—such as stochastic volatility in productivity or financial risk—agents demand precautionary wealth buffers and asset prices reflect time-varying risk premia. Because the conditional variance evolves stochastically, the state-space becomes fundamentally non-Gaussian, causing the Kalman filter to produce biased and misleading likelihood evaluations. Sequential Monte Carlo (particle filtering) overcomes this breakdown by propagating an empirical cloud of particles that tracks the complete, non-Gaussian posterior distribution across time.
#
# Second, economic policy is not immutable: central banks switch between aggressive inflation-fighting regimes and accommodative, growth-focused regimes. Standard rational expectations solvers cannot evaluate regimes where monetary policy violates the Taylor principle ($\phi_\pi < 1$). However, Foerster et al. (2016) show that if the public anticipates a sufficiently high probability of returning to an active regime in the future, the economy achieves global Mean-Square Stability despite temporary passages through locally indeterminate regimes.
#
# Third, occasionally binding constraints such as the Zero Lower Bound (ZLB) create non-differentiable kinks in policy functions. While discrete piecewise-linear algorithms (traditional OccBin) can simulate trajectories, their zero or infinite directional derivatives break modern gradient-based Bayesian samplers like NUTS (No-U-Turn Sampler). Differentiable OccBin solves this dilemma by smoothing the complementary slackness condition with a continuous temperature parameter $\tau$: as $\tau \to 0$, the smoothed path converges uniformly to the discrete boundary while preserving infinite-order differentiability ($C^\infty$) for Hamiltonian gradient evaluation.

# %%
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Editorial styling and palette contract
_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
try:
    import _nbstyle
    _nbstyle.apply_style()
except ImportError:
    pass

# Ensure clean non-blocking execution when executed as a CLI script
if not hasattr(sys, "ps1") and "IPython" not in sys.modules:
    plt.show = lambda *args, **kwargs: None

from puremacro.dsge.dynare import build_dynare, load_mod
from puremacro.dsge.occbin import (
    OccBinConstraint,
    solve_occbin,
    solve_differentiable_occbin,
    smin_tau,
    smax_tau,
    fischer_burmeister,
)
from puremacro.dsge.hank import load_hank_mod, solve_hank_bridge
from puremacro.dsge.markov_switching import solve_ms_dsge
from puremacro.dsge.particle_filter import (
    particle_filter,
    StochasticVolatilitySpec,
    systematic_resample,
)

print("puremacro Nonlinear DSGE module loaded successfully.")

# %% [markdown]
# ## 1. Differentiable OccBin for NUTS & Two-Asset HANK Bridge (Phase B)
#
# We first demonstrate the smooth relaxation of occasionally binding constraints. When the central bank is constrained by the Zero Lower Bound ($r_t \ge -r_{\text{ss}}$), traditional piecewise-linear algorithms introduce non-differentiable kinks that cause Hamiltonian Monte Carlo (NUTS) trajectories to diverge.
#
# By employing the smooth operator $\text{smin}_\tau(a, b)$, `puremacro` provides $C^\infty$ differentiable transition paths:

# %%
# Define canonical New Keynesian setup with ZLB constraint
params = {
    "beta": 0.99,
    "sigma": 1.0,
    "kappa": 0.1,
    "phi_pi": 1.5,
    "phi_y": 0.125,
    "rho_g": 0.8,
    "r_ss": 0.01,
}
variables = ["y", "pi", "r", "g"]
shocks = ["eps_r", "eps_g"]
steady_state = {v: 0.0 for v in variables}

# Unconstrained Reference Model
def nk_ref(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - p.phi_pi * curr.pi - p.phi_y * curr.y - shocks_v.eps_r,
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]

# Constrained Model at ZLB (r = -r_ss)
def nk_cons(lead, curr, lag, shocks_v, p):
    return [
        curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g,
        curr.pi - p.beta * lead.pi - p.kappa * curr.y,
        curr.r - (-p.r_ss),
        curr.g - p.rho_g * lag.g - shocks_v.eps_g,
    ]

m_ref = build_dynare(nk_ref, variables=variables, shocks=shocks, params=params, steady_state=steady_state)
m_cons = build_dynare(nk_cons, variables=variables, shocks=shocks, params=params, steady_state=steady_state, check_steady_state=False, strict=False)
zlb_constraint = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")

# Large deflationary demand shock pushing economy into ZLB for multiple quarters
T = 24
shocks_mat = np.zeros((T, 2))
shocks_mat[0, 1] = -0.045  # Negative demand innovation eps_g

# Solve discrete OccBin vs Differentiable OccBin across smoothing temperatures tau
res_discrete = solve_occbin(m_ref, m_cons, zlb_constraint, shocks_mat, horizon=T)
res_diff_02 = solve_differentiable_occbin(m_ref, m_cons, zlb_constraint, shocks_mat, tau=0.02, horizon=T)
res_diff_005 = solve_differentiable_occbin(m_ref, m_cons, zlb_constraint, shocks_mat, tau=0.005, horizon=T)

r_disc = res_discrete.path["r"].to_numpy()
r_smooth_02 = res_diff_02.path["r"].to_numpy()
r_smooth_005 = res_diff_005.path["r"].to_numpy()

err_02 = np.max(np.abs(r_disc - r_smooth_02))
err_005 = np.max(np.abs(r_disc - r_smooth_005))

print("--- Differentiable OccBin Convergence ---")
print(f"Discrete ZLB binding quarters : {np.sum(r_disc <= -params['r_ss'] + 1e-5)} quarters")
print(f"Max deviation at tau=0.020    : {err_02:.4e}")
print(f"Max deviation at tau=0.005    : {err_005:.4e}")
print(f"Convergence verification      : {err_005 < err_02} (Approaches discrete limit as tau -> 0)")

assert err_005 < err_02, "Smaller temperature parameter tau must tighten approximation"

# %% [markdown]
# We plot the smooth relaxation paths against the discrete OccBin kink:

# %%
plt.figure(figsize=(8.5, 4.2))
time_h = np.arange(T)
plt.plot(time_h, r_disc * 100, color="#1f77b4", lw=2.4, label="Discrete OccBin (Piecewise-Linear)")
plt.plot(time_h, r_smooth_02 * 100, color="#ff7f0e", lw=1.8, linestyle="--", label=r"Differentiable OccBin ($\tau=0.020$)")
plt.plot(time_h, r_smooth_005 * 100, color="#2ca02c", lw=1.8, linestyle=":", label=r"Differentiable OccBin ($\tau=0.005$)")
plt.axhline(-params["r_ss"] * 100, color="#d62728", linestyle="-.", lw=1.3, label=f"ZLB Floor (-{params['r_ss']*100:.1f}%)")
plt.title(r"Differentiable OccBin: Smooth ZLB Relaxation ($\tau \to 0$)", fontsize=11, fontweight="bold")
plt.xlabel("Quarters")
plt.ylabel("Policy Rate $r_t$ (% dev)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Two-Asset HANK Sequence-Space Bridge
#
# Moving from aggregate constraints to microeconomic heterogeneity, `puremacro` extends the sequence-space bridge to **Two-Asset HANK models** (Kaplan, Moll & Violante 2018). Households allocate wealth between liquid deposits $b$ and illiquid assets $a$ subject to portfolio transaction costs $\chi(d, a) = \chi_0 |d| + \frac{\chi_1}{2} \left(\frac{d}{a}\right)^2 a$.
#
# We load the reference `hank_two_asset.mod` model and solve its general equilibrium sequence-space transition:

# %%
import puremacro.dsge as _dsge
mod_two_asset = Path(_dsge.__file__).parent / "_references" / "hank_two_asset.mod"
res_two_asset = solve_hank_bridge(
    mod_two_asset,
    shock="eps_m",
    magnitude=-0.0025,
    horizon=16,
)

print("--- Two-Asset HANK General Equilibrium Bridge ---")
print(res_two_asset.summary())

# Extract aggregate consumption transition path
c_agg = res_two_asset.transition_paths["C"].to_numpy()
print(f"Impact aggregate consumption (C_0): {c_agg[0]:+.4f}")
print(f"Peak aggregate consumption       : {np.max(c_agg):+.4f}")
assert res_two_asset.converged, "Sequence-space HANK equilibrium must converge"
assert len(c_agg) == 16, "Transition horizon must match 16 quarters"

# %% [markdown]
# ## 2. Markov-Switching DSGE Perturbation (Foerster et al. 2016)
#
# We now explore discrete structural policy changes using the perturbation methods of Foerster, Rubio-Ramírez, Waggoner & Zha (2016).
#
# We model a 3-equation New Keynesian economy where the central bank alternates between two distinct regimes:
# 1. **Regime 1 (Hawkish / Active Monetary)**: $\phi_\pi = 1.80 > 1.0$, satisfying the Taylor principle and determinate in isolation.
# 2. **Regime 2 (Dovish / Passive Monetary)**: $\phi_\pi = 0.80 < 1.0$, violating the Taylor principle and indeterminate in isolation.
#
# The Markov transition matrix is:
# $$ P = \begin{bmatrix} 0.90 & 0.10 \\ 0.20 & 0.80 \end{bmatrix} $$
# with an expected duration of 10 quarters in the Hawkish regime and 5 quarters in the Dovish regime.

# %%
# Structural Calibration
beta = 0.99      # Discount factor
sigma = 1.00     # Elasticity of substitution
kappa = 0.10     # Slope of NK Phillips Curve
rho_i = 0.80     # Interest rate smoothing
phi_x = 0.10     # Output gap feedback

phi_pi_list = [1.80, 0.80]   # [Hawkish, Dovish]
regime_names = ["Hawkish", "Dovish"]
var_names = ["output_gap", "inflation", "interest_rate"]
shock_names = ["demand", "cost_push", "monetary_policy"]

P_trans = np.array([
    [0.90, 0.10],
    [0.20, 0.80],
])

# Structural matrices: A(s) E_t[y_{t+1}] + B(s) y_t + C(s) y_{t-1} + D(s) eps_t = 0
A_list = []
B_list = []
C_list = []
D_list = []

for s in range(2):
    phi_pi = phi_pi_list[s]
    As = np.array([
        [1.0, 1.0 / sigma, 0.0],
        [0.0, beta,        0.0],
        [0.0, 0.0,         0.0],
    ])
    Bs = np.array([
        [-1.0, 0.0,                              -1.0 / sigma],
        [kappa, -1.0,                            0.0],
        [(1.0 - rho_i) * phi_x, (1.0 - rho_i) * phi_pi, -1.0],
    ])
    Cs = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, rho_i],
    ])
    Ds = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    A_list.append(As)
    B_list.append(Bs)
    C_list.append(Cs)
    D_list.append(Ds)

# Solve coupled MS-DSGE equilibrium via block Newton-Raphson
ms_res = solve_ms_dsge(
    A=A_list,
    B=B_list,
    C=C_list,
    D=D_list,
    transition_matrix=P_trans,
    regime_names=regime_names,
    variable_names=var_names,
    shock_names=shock_names,
    method="newton",
    max_iter=100,
    tol=1e-12,
)

print(ms_res.summary())

# %% [markdown]
# ### Global Mean-Square Stability (MSS) Verification
#
# A central theoretical result in regime-switching macroeconomics is that an economy can achieve global stability even when passing through locally indeterminate regimes, provided the policymaker returns to the active regime with sufficient frequency.
#
# We verify the spectral radius of the first-moment matrix $\rho(M_1)$ and second-moment matrix $\rho(M_2)$:

# %%
print("--- Stability & Ergodic Properties ---")
print(f"Mean Stability rho(M1)          : {ms_res.spectral_radius_mean:.4f} (< 1.0 confirmed)")
print(f"Mean-Square Stability rho(M2)  : {ms_res.spectral_radius_mss:.4f} (< 1.0 confirmed)")
print(f"Mean-Square Stable (MSS)        : {ms_res.mean_square_stable}")
print(f"Ergodic Prob [Hawkish]          : {ms_res.ergodic_distribution['Hawkish']:.4f} (Theoretical 2/3 = 0.6667)")
print(f"Ergodic Prob [Dovish]           : {ms_res.ergodic_distribution['Dovish']:.4f} (Theoretical 1/3 = 0.3333)")

assert ms_res.mean_square_stable, "System must be Mean-Square Stable"
assert ms_res.spectral_radius_mean < 1.0
assert np.isclose(ms_res.ergodic_distribution["Hawkish"], 2.0 / 3.0, atol=1e-5)

# %% [markdown]
# ### Closed-Form Generalized Impulse Response Functions (GIRF)
#
# In linear models, impulse responses are independent of history and future regimes. In MS-DSGE, the transmission of a monetary shock depends crucially on the starting regime and the probability of transitioning across regimes over the forecast horizon.
#
# `puremacro` computes closed-form analytical GIRFs via the exact expectation operator $(1_S' \otimes I_n) M_1^h z_0$ with zero Monte Carlo simulation noise:

# %%
girf_df = ms_res.girf(shock=2, horizon=16, initial_regime=0)  # Contractionary monetary shock in Hawkish regime

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

time_axis = np.arange(len(girf_df))

axes[0].plot(time_axis, girf_df["interest_rate"], color="#d62728", lw=2.2, marker="o")
axes[0].set_title("Nominal Rate $i_t$ (Policy Hike)", fontweight="bold")
axes[0].set_xlabel("Quarters")
axes[0].grid(True, linestyle=":", alpha=0.6)

axes[1].plot(time_axis, girf_df["output_gap"], color="#1f77b4", lw=2.2, marker="s")
axes[1].set_title("Output Gap $y_t$ (Contraction)", fontweight="bold")
axes[1].set_xlabel("Quarters")
axes[1].grid(True, linestyle=":", alpha=0.6)

axes[2].plot(time_axis, girf_df["inflation"], color="#2ca02c", lw=2.2, marker="^")
axes[2].set_title(r"Inflation $\pi_t$ (Deceleration)", fontweight="bold")
axes[2].set_xlabel("Quarters")
axes[2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Vectorized Sequential Monte Carlo Particle Filtering
#
# We now turn to nonlinear likelihood evaluation. When models are approximated to higher orders (e.g. 2nd- or 3rd-order pruned perturbation) or feature **Stochastic Volatility**, standard Kalman filtering is invalid.
#
# We set up a nonlinear neoclassical growth model with capital accumulation, Euler equation curvature, and an autoregressive stochastic volatility process for total factor productivity:
# $$ \sigma_{z,t} = \bar{\sigma} \exp(h_t), \quad h_t = \rho_h h_{t-1} + \sigma_\eta \eta_t $$
#
# We run the vectorized Bootstrap Particle Filter (BPF) with $N=2,000$ particles and systematic resampling across $T=32$ quarters:

# %%
RBC_MOD = """
var c k z;
varexo eps;

parameters beta alpha delta rho sigma_pref sigma_eps;
beta       = 0.99;   // Subjective discount factor
alpha      = 0.33;   // Capital elasticity of output
delta      = 0.025;  // Capital depreciation rate
rho        = 0.95;   // Persistence of technology shock
sigma_pref = 1.00;   // Relative risk aversion coefficient
sigma_eps  = 0.01;   // Base technology shock innovation standard deviation

model;
  // Euler equation for capital accumulation:
  exp(-sigma_pref * c) - beta * exp(-sigma_pref * c(+1)) * (alpha * exp(z(+1)) * exp((alpha - 1.0) * k) + 1.0 - delta);

  // Resource constraint:
  exp(c) + exp(k) - exp(z) * exp(alpha * k(-1)) - (1.0 - delta) * exp(k(-1));

  // Technology AR(1) process:
  z - rho * z(-1) - sigma_eps * eps;
end;

initval;
  k = 3.8;
  c = 0.8;
  z = 0.0;
end;

steady;
"""

# Compile nonlinear model and solve 2nd-order pruned perturbation
model = load_mod(RBC_MOD)
sol2 = model.solve(order=2)

print("--- Nonlinear Model & Pruned Perturbation ---")
print(f"Model variables    : {list(model.variables)}")
print(f"Pruned solution    : {type(sol2).__name__}")
print(f"Steady state k*    : {sol2.steady_state['k']:.4f}")
print(f"Steady state c*    : {sol2.steady_state['c']:.4f}")

# Simulate synthetic observable path under true time-varying volatility
T = 32
sim = sol2.simulate(periods=T, seed=101).to_frame() + sol2.steady_state
varobs = ["c", "k"]

# Configure Stochastic Volatility specification
sv_spec = StochasticVolatilitySpec(
    rho=0.85,
    sigma_eta=0.25,
    base_scale=0.01,
    h0=0.0,
)

# Run Bootstrap Particle Filter with N=2,000 particles
n_particles = 2_000
res_sv = particle_filter(
    sol2,
    sim,
    varobs,
    n_particles=n_particles,
    method="bootstrap",
    resampling_method="systematic",
    stochastic_volatility=sv_spec,
    seed=42,
)

print(res_sv.summary())

# %% [markdown]
# ### Effective Sample Size (ESS) Trajectory and Degeneracy Diagnostics
#
# A critical diagnostic of particle filtering quality is the Effective Sample Size (ESS) across time:
# $$ ESS_t = \frac{1}{\sum_{i=1}^N (w_t^{(i)})^2} \in [1, N] $$
# Whenever $ESS_t < N/2$, systematic resampling replenishes particles with high likelihood, preventing sample impoverishment:

# %%
ess_path = res_sv.ess

plt.figure(figsize=(8.5, 3.8))
plt.plot(np.arange(len(ess_path)), ess_path, color="#1f77b4", lw=2.2, label=r"Effective Sample Size $ESS_t$")
plt.axhline(n_particles / 2, color="#d62728", linestyle="--", lw=1.5, label=f"Resampling Threshold ($N/2 = {n_particles//2:,}$)")
plt.title("Sequential Monte Carlo Diagnostics: Effective Sample Size Trajectory", fontsize=11, fontweight="bold")
plt.xlabel("Observation Index $t$")
plt.ylabel("Effective Particles")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

print(f"Mean Effective Particles : {res_sv.ess.mean():.1f} / {n_particles:,}")
print(f"Resampling Frequency     : {res_sv.resampling_frequency*100:.1f}% of periods")
print(f"Log-Likelihood under SV  : {res_sv.log_likelihood:.4f}")

assert res_sv.ess.mean() > 100.0, "Filter must maintain adequate particle diversity"
assert np.isfinite(res_sv.log_likelihood), "Log-likelihood must be finite"

# %% [markdown]
# ## 4. Publication-Ready Scientific Reporting
#
# We export our Markov-Switching DSGE summary and Particle Filter performance diagnostics to LaTeX and Markdown for inclusion in manuscripts:

# %%
print("--- LaTeX Table Export: Markov-Switching DSGE Regimes ---")
print(ms_res.to_latex())

print("--- Markdown Table Export: Particle Filter Performance ---")
print(res_sv.to_markdown())

# %% [markdown]
# ## Read the output
#
# **Read the output.**
# 1. **Differentiable OccBin Convergence ($\tau \to 0$)**: Under a massive deflationary demand shock ($\epsilon_g = -0.045$), the policy rate hits the $-1.0\%$ ZLB floor for 6 consecutive quarters. The smooth relaxation paths with temperature parameters $\tau = 0.020$ and $\tau = 0.005$ envelop the discrete piecewise-linear trajectory: the maximum absolute deviation shrinks from $6.89 \times 10^{-3}$ at $\tau=0.020$ down to $1.31 \times 10^{-3}$ at $\tau=0.005$, confirming monotonic uniform convergence toward the non-smooth boundary while retaining $C^\infty$ differentiability for gradient-based inference.
# 2. **Two-Asset HANK Sequence-Space Bridge**: Incorporating household portfolio allocation between liquid cash deposits ($b$) and illiquid capital equity ($a$) with quadratic adjustment costs, the general equilibrium transition under a $-25$ bps monetary policy shock converges rapidly. Aggregate consumption jumps expansionary ($C_0 > 0$), reflecting the high marginal propensity to consume among liquid-constrained households.
# 3. **Markov-Switching Global Mean-Square Stability**: Although the Dovish monetary regime fails the standard Taylor principle ($\phi_\pi = 0.80 < 1.0$, rendering it locally indeterminate in isolation), the coupled rational expectations system achieves global Mean-Square Stability ($\rho(M_2) = 0.6380 < 1.0$, $\rho(M_1) = 0.5729 < 1.0$). Because the ergodic probability of the Hawkish regime is $2/3$, the public's rational anticipation of eventual monetary tightening anchors expectations and stabilizes the aggregate economy.
# 4. **Closed-Form Generalized Impulse Responses (GIRF)**: Following an unanticipated monetary tightening shock in the Hawkish regime, the nominal policy rate hikes on impact, inducing an immediate contraction in the output gap and decelerating inflation. The analytical GIRF computes exact intertemporal expectations across all future regime paths in closed form, eliminating Monte Carlo simulation noise.
# 5. **Sequential Monte Carlo Particle Filter with Stochastic Volatility**: Propagating $N = 2,000$ particles across a nonlinear RBC model with an autoregressive log-volatility process, the Bootstrap Particle Filter maintains a healthy average Effective Sample Size ($ESS = 716.8 / 2,000$, with resampling triggered in $62.5\%$ of periods), preventing particle degeneracy and evaluating an exact nonlinear log-likelihood of $\ln \hat{L} = -15.15$.

# %% [markdown]
# ## Your turn
#
# **Prompts.**
# 1. *Basic*: Modify the smoothing temperature parameter in Differentiable OccBin (`tau_yt = 0.010` vs `0.002`) and observe how the approximation error scales with temperature.
# 2. *Intermediate*: Re-evaluate the closed-form MS-DSGE GIRF under an initial Dovish regime (`initial_regime_yt = 1` vs `0`) and contrast the resulting output contraction and interest rate hike against the Hawkish baseline.
# 3. *Stretch*: Increase the number of particles in the Sequential Monte Carlo filter (`n_particles_yt = 5_000`) and test whether the estimated log-likelihood converges and the resampling frequency stabilizes.

# %%
# Your turn: customize smoothing temperature or MS-DSGE initial regime
# ← change this: test OccBin smoothing temperature tau_yt = 0.010 (default), 0.015, or 0.002
tau_yt = 0.010
# ← change this: test MS-DSGE initial regime (0 = Hawkish, 1 = Dovish)
initial_regime_yt = 1

# 1. Re-solve Differentiable OccBin under custom smoothing temperature
res_diff_yt = solve_differentiable_occbin(
    m_ref, m_cons, zlb_constraint, shocks_mat, tau=tau_yt, horizon=len(shocks_mat)
)
r_smooth_yt = res_diff_yt.path["r"].to_numpy()
err_yt = float(np.max(np.abs(r_disc - r_smooth_yt)))

# 2. Re-solve MS-DSGE GIRF under custom starting regime
girf_yt = ms_res.girf(shock=2, horizon=16, initial_regime=initial_regime_yt)
regime_name_yt = regime_names[initial_regime_yt]

print(f"Differentiable OccBin (tau = {tau_yt:.4f}):")
print(f"  Max absolute deviation from discrete : {err_yt:.6e}")
print(f"  Monotonic error bound (err_005 < err_yt < err_02) : {err_005 < err_yt < err_02}")
print(f"MS-DSGE GIRF (Initial Regime = {regime_name_yt}):")
print(f"  Impact policy rate (i_0) : {girf_yt['interest_rate'].iloc[0]:+.4f} (Baseline Hawkish: {girf_df['interest_rate'].iloc[0]:+.4f})")
print(f"  Impact output gap  (y_0) : {girf_yt['output_gap'].iloc[0]:+.4f} (Baseline Hawkish: {girf_df['output_gap'].iloc[0]:+.4f})")
print(f"  Impact inflation  (pi_0) : {girf_yt['inflation'].iloc[0]:+.4f}")

# Downstream automated assertions
assert res_diff_yt.converged
assert err_yt < err_02, "Error at smaller tau must remain bounded by tau=0.020 error"
if initial_regime_yt == 1:
    assert girf_yt["output_gap"].iloc[0] < girf_df["output_gap"].iloc[0], "Dovish regime must exhibit deeper output contraction"
    assert girf_yt["interest_rate"].iloc[0] > girf_df["interest_rate"].iloc[0]
else:
    np.testing.assert_allclose(girf_yt["output_gap"].iloc[0], girf_df["output_gap"].iloc[0], atol=1e-6)
assert np.isclose(girf_yt["interest_rate"].iloc[-1], 0.0, atol=1e-3), "GIRF must mean-revert toward zero"

# %% [markdown]
# ## How comprehensive is this?
#
# `puremacro.dsge` unifies the advanced frontier of nonlinear and regime-switching macroeconomics in 100% pure Python:
# - `solve_differentiable_occbin`: Smooth relaxation of occasionally binding inequality constraints using parameterized smooth operators ($\text{smin}_\tau, \text{smax}_\tau$), providing $C^\infty$ differentiable trajectories for gradient-based Bayesian estimation via NUTS HMC.
# - `solve_ms_dsge`: Solves Markov-Switching DSGE models (Foerster et al. 2016) via coupled block Newton-Raphson iterations, checks first- and second-moment Mean-Square Stability ($\rho(M_1), \rho(M_2) < 1.0$), and evaluates closed-form analytical Generalized Impulse Responses (`girf`).
# - `particle_filter`: Vectorized Bootstrap Particle Filter with systematic stratified resampling, effective sample size ($ESS$) monitoring, and support for Stochastic Volatility (`StochasticVolatilitySpec`).
# - `solve_hank_bridge`: Bridges microeconomic household heterogeneity in liquid and illiquid assets (Kaplan, Moll & Violante 2018) with sequence-space general equilibrium transitions.
