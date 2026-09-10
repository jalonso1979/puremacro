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
# # What's New in puremacro 3.0: Milestone Tour
#
# `puremacro 3.0.0` represents a transformative generational leap for macroeconomic computing
# in Python. While the 2.x release cycle brought `puremacro` to **complete operational parity with Dynare**
# (encompassing full `.mod` parsing, 2nd- and 3rd-order pruned perturbation, extended path simulation,
# OccBin multi-regime filtering, and automated identification analysis), version 3.0.0 pivots from
# *emulating legacy toolchains* to *surpassing them*.
#
# Version 3.0.0 introduces three pioneering pillars directly into macroeconomic research workflows:
# 1. **Exact Analytic Likelihood Gradients ($\nabla_\theta \ln L$)** — Evaluates structural score vectors
#    via AST parameter differentiation, generalized Sylvester matrix equations solved via complex Schur
#    triangular back-substitution, and single-pass forward Kalman score recursions.
# 2. **Pure-Python Hamiltonian Monte Carlo & No-U-Turn Sampler (NUTS)** — Gradient-based posterior
#    sampling for high-dimensional macroeconomic models with Betancourt generalized U-turn stopping,
#    Hoffman-Gelman dual averaging, online Welford covariance adaptation with Stan shrinkage, and
#    complete MCMC diagnostics (split-$\hat{R}$, bulk/tail ESS, E-BFMI).
# 3. **Heterogeneous Agents (HANK) Sequence-Space Bridge in `.mod` Files** — Introduces the
#    `hetagent_block; ... end;` grammar into Dynare-style `.mod` specifications, seamlessly coupling
#    microeconomic household decisions (wealth distributions $\mathcal{D}^*(a)$, $MPC(a)$ schedules)
#    with aggregate DSGE equilibrium via Auclert, Bardóczy, Rognlie & Straub (2021) Fake-News Jacobians
#    ($\mathcal{J}_{C, r}, \mathcal{J}_{C, Y}$) and general equilibrium transition solvers.
#
# All three pillars run in **100% pure Python** under the strict **Pyodide four-package contract**
# (`numpy`, `scipy`, `pandas`, `matplotlib`), requiring zero C/Fortran compilers and enabling seamless
# execution in browsers, JupyterLite, iPads, and cloud notebooks.

# %%
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Headless rendering when executed from command line
if "ipykernel" not in sys.modules and not hasattr(sys, "ps1"):
    import matplotlib
    matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt

# Apply notebook publication styling if available
_cwd = Path.cwd()
sys.path.insert(0, str(_cwd if (_cwd / "_nbstyle.py").exists() else _cwd / "notebooks"))
try:
    import _nbstyle
    _nbstyle.apply_style()
except ImportError:
    pass

import puremacro
print(f"Loaded puremacro version: {puremacro.__version__}")

# %% [markdown]
# ---
# ## Section 1: Exact Analytic Likelihood Gradients ($\nabla_\theta \ln L$)
#
# ### 1.1 The Gradient Bottleneck in Macroeconomic Estimation
#
# In legacy toolchains (such as Dynare or MATLAB), estimating DSGE models via gradient-based optimizers
# (L-BFGS-B, SQP) or Hamiltonian Monte Carlo relies on two-sided numerical finite differences:
# $$
# \frac{\partial \ln L}{\partial \theta_j} \approx \frac{\ln L(\theta + h e_j) - \ln L(\theta - h e_j)}{2h}
# $$
#
# For $K$ estimated parameters, each gradient step requires $2K$ full Kalman filter sweeps over the sample history.
# For medium-scale models ($K \ge 35$), finite differences suffer from severe computational cost and numerical
# instability: choosing step size $h$ too large introduces truncation bias, while choosing $h$ too small causes
# catastrophic floating-point cancellation near unit-root boundaries.
#
# ### 1.2 The Generalized Sylvester Solver
#
# `puremacro 3.0` replaces finite differences with an **exact, single-pass forward analytic score engine**.
#
# Around steady state, first-order perturbation produces the quadratic matrix Riccati equation for the
# state transition matrix $G(\theta)$:
# $$
# \mathcal{F}(G; \theta) \equiv A_+(\theta) G(\theta)^2 + A_0(\theta) G(\theta) + A_-(\theta) = 0
# $$
#
# Differentiating with respect to parameter $\theta_j$ yields the **generalized Sylvester matrix equation**:
# $$
# \hat{A} \frac{\partial G}{\partial \theta_j} + B \frac{\partial G}{\partial \theta_j} C = D_j
# $$
# where:
# - $\hat{A} = A_0 + A_+ G \in \mathbb{R}^{N \times N}$
# - $B = A_+ \in \mathbb{R}^{N \times N}$
# - $C = G \in \mathbb{R}^{N \times N}$
# - $D_j = - \left( \frac{\partial A_+}{\partial \theta_j} G^2 + \frac{\partial A_0}{\partial \theta_j} G + \frac{\partial A_-}{\partial \theta_j} \right)$
#
# Because Blanchard-Kahn stability guarantees that all eigenvalues of $C = G$ lie strictly inside the open unit circle,
# `puremacro.dsge._gradients.solve_sylvester_generalized` computes the complex Schur decomposition $C = U_c T_c U_c^H$
# **once** and solves the triangular decoupled system column-by-column across all $K$ parameters simultaneously in
# $\mathcal{O}(K \cdot N^3)$ flops in $< 1$ ms.

# %%
from puremacro.dsge._gradients import solve_sylvester_generalized

# Demonstrate machine-precision generalized Sylvester solution: A_hat * X + B * X * C = D
N, M = 4, 3
rng = np.random.default_rng(2026)

A_hat = rng.standard_normal((N, N)) + 4.0 * np.eye(N)
B = rng.standard_normal((N, N)) * 0.2
# Stable transition matrix (eigenvalues strictly inside unit circle)
C = rng.standard_normal((M, M)) * 0.25
D = rng.standard_normal((N, M))

X_sol = solve_sylvester_generalized(A_hat, B, C, D)
residual = A_hat @ X_sol + B @ X_sol @ C - D
max_error = np.max(np.abs(residual))

print(f"Sylvester solution matrix X shape: {X_sol.shape}")
print(f"Maximum residual ||A_hat @ X + B @ X @ C - D||_inf: {max_error:.2e}")
assert max_error < 1e-12, "Sylvester solver did not achieve machine precision!"

# %% [markdown]
# ### 1.3 Single-Pass Forward Kalman Score Recursion
#
# With decision rule derivatives $\frac{\partial G}{\partial \theta_j}$ and $\frac{\partial N}{\partial \theta_j}$,
# `build_state_space_sensitivities` maps them to state-space sensitivities
# $(\frac{\partial T}{\partial \theta_j}, \frac{\partial Z}{\partial \theta_j}, \frac{\partial R}{\partial \theta_j}, \frac{\partial Q}{\partial \theta_j}, \frac{\partial H}{\partial \theta_j})$.
#
# The unconditional stationary covariance sensitivity $\frac{\partial P_0}{\partial \theta_j}$ is obtained from the
# differentiated discrete Lyapunov equation.
#
# The Gaussian log-likelihood and exact score $\nabla_\theta \ln L$ are evaluated in a single forward pass:
# $$
# \frac{\partial \ln L}{\partial \theta_j} = -\frac{1}{2} \sum_{t=1}^{T_{obs}} \left[ \operatorname{tr}\left( F_t^{-1} \frac{\partial F_t}{\partial \theta_j} \right) + 2 v_t^\top F_t^{-1} \frac{\partial v_t}{\partial \theta_j} - (F_t^{-1} v_t)^\top \frac{\partial F_t}{\partial \theta_j} (F_t^{-1} v_t) \right]
# $$
# propagating state and covariance sensitivities $(\frac{\partial a_{t|t-1}}{\partial \theta_j}, \frac{\partial P_{t|t-1}}{\partial \theta_j})$
# forward alongside the Kalman equations without storing any backpropagation graph.

# %%
from puremacro.dsge import load_mod
from puremacro.dsge.observation import make_state_space_from_varobs
from puremacro.dsge._gradients import (
    build_state_space_sensitivities,
    compute_decision_rule_derivatives,
    kalman_score,
    ScoreDiagnosticsResult,
)

MOD_CODE_GRAD = """
var y c a;
varexo e_a;
parameters beta sigma rho_a;
beta = 0.99;
sigma = 1.0;
rho_a = 0.8;

model;
  c = c(+1) - (1/sigma)*y;
  y = c + a;
  a = rho_a * a(-1) + e_a;
end;

steady_state_model;
  y = 0; c = 0; a = 0;
end;

shocks;
  var e_a; stderr 0.2;
end;

varobs y;
"""

# 1. Parse and compile DSGE model
model_grad = load_mod(MOD_CODE_GRAD)
varobs = ["y"]
param_names = ["sigma", "rho_a"]

# 2. Simulate observed sample data
sim_df = model_grad.simulate(periods=40, seed=42)
y_sim = sim_df[["y"]].to_numpy()

# 3. Construct state-space model and analytical parameter sensitivities
ssm = make_state_space_from_varobs(model_grad, varobs)
sensitivities = build_state_space_sensitivities(model_grad, varobs, param_names)

# 4. Evaluate exact log-likelihood and score vector in a single pass
t0 = time.perf_counter()
loglik, score = kalman_score(y_sim, ssm, sensitivities)
elapsed = time.perf_counter() - t0

# 5. Package into ScoreDiagnosticsResult
res_score = ScoreDiagnosticsResult(
    loglik=loglik,
    gradient=score,
    param_names=tuple(param_names),
    elapsed_sec=elapsed,
)

print(f"Exact Log-Likelihood : {res_score.loglik:.4f}")
print(f"Elapsed Time         : {res_score.elapsed_sec * 1000:.3f} ms")
print("\nAnalytical Score Diagnostics Table:")
print(res_score.to_markdown())

# Inspect analytical decision rule sensitivities: dghx / d_rho_a
dghx, dghu = compute_decision_rule_derivatives(model_grad, "rho_a")
print("\nAnalytical Transition Sensitivities (dghx / d_rho_a):")
print(pd.DataFrame(dghx, index=model_grad.variables, columns=model_grad.states))

# %% [markdown]
# ### 1.4 Visualizing the Exact Gradient vs Decision Rule Sensitivities
#
# We plot the analytical score vector and state transition sensitivities across the structural parameters.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Bar chart of analytical score vector
x_pos = np.arange(len(param_names))
bars = ax1.bar(x_pos, score, color=["#1f77b4", "#2ca02c"], edgecolor="black", width=0.5, alpha=0.85)
ax1.axhline(0, color="black", lw=0.8, linestyle="--")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([r"$\sigma$ (Risk Aversion)", r"$\rho_a$ (Tech Persistence)"], fontsize=10)
ax1.set_title(r"Exact Score Vector $\nabla_\theta \ln L$", fontsize=11, fontweight="bold")
ax1.set_ylabel("Log-Likelihood Score")
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f"{height:.3f}",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3 if height >= 0 else -12),
                 textcoords="offset points", ha="center", va="bottom", fontsize=9)

# Bar chart of decision rule sensitivity to persistence
vars_plot = model_grad.variables
dghx_vals = dghx[:, 0]
ax2.bar(vars_plot, dghx_vals, color="#d62728", edgecolor="black", width=0.5, alpha=0.85)
ax2.axhline(0, color="black", lw=0.8, linestyle="--")
ax2.set_title(r"State Sensitivity $\partial ghx / \partial \rho_a$", fontsize=11, fontweight="bold")
ax2.set_ylabel(r"$\partial y / \partial \rho_a$")
ax2.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Section 2: Pure-Python Hamiltonian Monte Carlo & NUTS
#
# ### 2.1 The Need for Gradient-Based MCMC in Macroeconomics
#
# Random-Walk Metropolis-Hastings (RWMH) explores high-dimensional posteriors via Brownian diffusion:
# the expected distance traversed scales as $\mathcal{O}(\sqrt{N})$. In medium- and large-scale DSGE models
# with curved parameter manifolds and strong posterior correlations, RWMH diffuses slowly, yields high
# autocorrelation, and produces low effective sample sizes ($ESS < 2\%$).
#
# Hamiltonian Monte Carlo (HMC) solves this by mapping posterior sampling into Hamiltonian mechanics.
# It introduces momentum variables $p \sim \mathcal{N}(0, M)$ conjugate to parameters $\theta$, defining
# the Hamiltonian energy:
# $$
# H(\theta, p) = V(\theta) + K(p) = - \ln p(\theta \mid Y) + \frac{1}{2} p^\top M^{-1} p
# $$
#
# The **No-U-Turn Sampler (NUTS)** (Hoffman & Gelman 2014; Betancourt 2017) builds recursive binary leapfrog
# trees to integrate Hamiltonian trajectories forward and backward in time, stopping automatically as soon
# as the trajectory begins to fold back on itself:
# $$
# (\theta^+ - \theta^-)^\top M^{-1} p^+ < 0 \quad \text{or} \quad (\theta^+ - \theta^-)^\top M^{-1} p^- < 0
# $$
#
# ### 2.2 Features of puremacro's Pure-Python NUTS Engine
#
# 1. **Symplectic Leapfrog Integrator**: Energy-conserving velocity Verlet integration with bounded $\mathcal{O}(\epsilon^2)$ shadow Hamiltonian error.
# 2. **Stan Staged Dual Averaging**: Step size $\epsilon$ adapts automatically to target acceptance probability $\delta^* = 0.80$.
# 3. **Welford Online Covariance Adaptation**: Estimates diagonal mass matrix $M^{-1}$ online with Stan prior shrinkage.
# 4. **Rigorous MCMC Diagnostics**: Rank-normalized split-$\hat{R}$, bulk ESS, tail ESS, and Energy Bayesian Fraction of Missing Information ($E\text{-}BFMI$).

# %%
MOD_CODE_NUTS = """
var y c a;
varexo e_a;
parameters beta sigma rho_a;
beta = 0.99;
sigma = 1.0;
rho_a = 0.8;

model;
  c = c(+1) - (1/sigma)*y;
  y = c + a;
  a = rho_a * a(-1) + e_a;
end;

steady_state_model;
  y = 0; c = 0; a = 0;
end;

varobs y;
estimated_params;
  sigma, gamma_pdf, 1.0, 0.25;
end;
"""

# 1. Load model with estimation specification
model_nuts = load_mod(MOD_CODE_NUTS)

# 2. Simulate observations
rng_nuts = np.random.default_rng(42)
df_data = pd.DataFrame({"y": rng_nuts.normal(0, 0.5, size=25)})

# 3. Estimate via pure-Python NUTS with exact analytic likelihood gradients
# Using compact draws for fast execution (< 3 seconds)
t0 = time.perf_counter()
res_nuts = model_nuts.estimate(
    data=df_data,
    method="nuts",
    n_draws=30,
    n_chains=2,
    burn_in=15,
    seed=42,
)
t_nuts = time.perf_counter() - t0

print(f"NUTS Estimation completed in {t_nuts:.2f} s")
print("\nPosterior Parameter Summary Table:")
print(res_nuts.summary())
print("\nNUTS Diagnostics:")
print(f"  Divergent Transitions: {res_nuts.diagnostics.get('n_divergences', 0)}")
print(f"  Adapted Step Sizes   : {np.round(res_nuts.diagnostics.get('step_sizes', []), 4)}")

# %% [markdown]
# ### 2.3 Posterior MCMC Diagnostics and Energy Distribution
#
# `NUTSResult` provides native diagnostic plotting methods:
# - `.plot_trace()` — Multiple chain trajectories and posterior exploration.
# - `.plot_posterior()` — Posterior density marginal with credible intervals.
# - `.energy_diagnostics()` — Energy transitions and $E\text{-}BFMI$ statistic.

# %%
# 1. Trace plot across chains
fig_trace, axes_trace = res_nuts.plot_trace()
plt.suptitle("NUTS Posterior Traces", fontsize=12, fontweight="bold", y=1.02)
plt.show()

# 2. Marginal posterior density
fig_post, axes_post = res_nuts.plot_posterior()
plt.suptitle("NUTS Marginal Posterior Distribution", fontsize=12, fontweight="bold", y=1.02)
plt.show()

# 3. Energy diagnostics (E-BFMI)
energy_stats, fig_energy, ax_energy = res_nuts.energy_diagnostics()
print(f"Energy Diagnostic Report:")
print(f"  Mean E-BFMI across chains : {energy_stats['mean_ebfmi']:.3f}")
print(f"  E-BFMI passed (>= 0.3)    : {energy_stats['passed']}")
plt.show()

# %% [markdown]
# ---
# ## Section 3: Heterogeneous-Agent (HANK) Sequence-Space Bridge in `.mod` Files
#
# ### 3.1 Overcoming the Representative-Agent Barrier
#
# Representative Agent New Keynesian (RANK) models collapse household behavior into a single Euler equation,
# ignoring income risk, wealth inequality, and liquidity-constrained hand-to-mouth households.
#
# Heterogeneous Agent New Keynesian (HANK) models replace the representative agent with a continuum of
# households facing idiosyncratic labor income risk and uninsurable borrowing constraints. However,
# state-space solutions to HANK models require carrying the infinite-dimensional wealth distribution
# $\mathcal{D}_t(a, s)$ as an aggregate state variable, leading to the curse of dimensionality.
#
# The **Sequence-Space Jacobian (SSJ)** method (**Auclert, Bardóczy, Rognlie & Straub 2021, *Econometrica*)**
# solves HANK models directly in the sequence space of shock sequences and macroeconomic paths over horizon $T$.
#
# ### 3.2 The `hetagent_block` Syntax in `.mod` Files
#
# `puremacro 3.0` brings the Sequence-Space Jacobian framework natively into Dynare-style `.mod` specifications
# via the new `hetagent_block; ... end;` construct:
#
# ```dynare
# hetagent_block;
#   model = one_asset_hank;
#   n_a = 40;
#   a_max = 25.0;
#   borrowing_limit = 0.0;
#   grid = hyperbolic;
# end;
# ```
#
# The model compiler automatically couples the microeconomic household block (solved via Endogenous Grid Method)
# with the aggregate DSGE market clearing conditions via the **Fake-News Algorithm**, evaluating the
# intertemporal consumption Jacobians $\mathcal{J}_{C, r} = \frac{\partial \mathbf{C}}{\partial \mathbf{r}}$ and
# $\mathcal{J}_{C, Y} = \frac{\partial \mathbf{C}}{\partial \mathbf{Y}}$ in $\mathcal{O}(T^2)$ time.

# %%
from puremacro.dsge.hank import load_hank_mod, solve_hank_bridge

SAMPLE_HANK_MOD = """
var Y C r pi i;
varexo eps_m;

parameters beta gamma r_ss phi_pi kappa;
beta = 0.985;
gamma = 1.0;
r_ss = 0.01;
phi_pi = 1.5;
kappa = 0.1;

hetagent_block;
  model = one_asset_hank;
  n_a = 40;
  a_max = 25.0;
  borrowing_limit = 0.0;
  grid = hyperbolic;
end;

model;
  Y = C;
  pi = beta * pi(+1) + kappa * Y;
  i = r_ss + phi_pi * pi + eps_m;
  r = i - pi(+1);
end;
"""

# 1. Load and compile HANK model from .mod code
model_hank = load_hank_mod(SAMPLE_HANK_MOD)

print("HANK Steady State Summary:")
for k, v in model_hank.steady_state.items():
    print(f"  {k:5s} = {v:8.4f}")

# 2. Compute Fake-News Sequence-Space Jacobians
T_horizon = 20
jacobians = model_hank.compute_jacobians(T=T_horizon)
J_C_r = jacobians["J_C_r"]
J_C_Y = jacobians["J_C_Y"]

print(f"\nJacobian J_C_r Shape: {J_C_r.shape}")
print(f"Contemporaneous Interest Elasticity (dC_0 / dr_0): {J_C_r[0, 0]:.4f}")
print(f"Contemporaneous Income MPC (dC_0 / dY_0)         : {J_C_Y[0, 0]:.4f}")

# %% [markdown]
# ### 3.3 General Equilibrium Transition Simulation
#
# We simulate the general equilibrium response of the HANK economy to an expansionary monetary policy shock
# ($\epsilon_m = -25$ bps, $\rho = 0.5$) over 15 quarters.
#
# The monetary easing lowers real interest rates, stimulating borrowing and investment, while
# redistributing income to high-MPC liquidity-constrained households, triggering a strong consumption multiplier.

# %%
# Simulate transition path
res_hank = model_hank.simulate(
    shock="eps_m",
    magnitude=-0.0025,
    rho=0.5,
    horizon=15,
    nonlinear=False,
)

print(f"HANK Simulation Status: Converged = {res_hank.converged}")
print("\nTransition Path Summary Table:")
print(res_hank.summary())

# Plot General Equilibrium IRF Transitions
fig_tr, axes_tr = res_hank.plot_transition()
plt.suptitle("HANK General Equilibrium Transition (-25 bps Rate Cut)", fontsize=12, fontweight="bold", y=1.02)
plt.show()

# Plot Stationary Wealth Distribution D*(a) and MPC Schedule
fig_dist, axes_dist = res_hank.plot_distribution()
plt.suptitle("HANK Microeconomic Distributions", fontsize=12, fontweight="bold", y=1.02)
plt.show()

# %% [markdown]
# ### 3.4 One-Liner Shortcut API: `solve_hank_bridge`
#
# For rapid modeling, `solve_hank_bridge` parses, solves the stationary equilibrium, computes the
# Fake-News Jacobians, and solves the coupled general equilibrium transition in a single line.

# %%
# Solve HANK model directly from string in 1 line
res_bridge = solve_hank_bridge(
    SAMPLE_HANK_MOD,
    shock="eps_m",
    magnitude=-0.0025,
    horizon=15,
)

print(f"solve_hank_bridge completed successfully: {res_bridge.converged}")
print("\nOutput (Y) and Consumption (C) first 5 periods:")
print(res_bridge.transition_paths[["Y", "C", "pi", "r"]].head())

# %% [markdown]
# ---
# ## Conclusion & Next Steps
#
# `puremacro 3.0.0` delivers three landmark capabilities for macroeconomic research:
# 1. **Exact Analytic Likelihood Gradients ($\nabla_\theta \ln L$)**: Eliminates noisy, slow finite differences via the generalized Sylvester solver and single-pass forward Kalman score recursion.
# 2. **Pure-Python HMC & NUTS**: High-efficiency posterior exploration with automatic step-size adaptation and audited MCMC diagnostics.
# 3. **Sequence-Space HANK in `.mod` Files**: Seamless integration of microeconomic wealth distributions and aggregate DSGE equilibrium via Fake-News Jacobians.
#
# All features adhere strictly to the **Pyodide four-package contract** (`numpy`, `scipy`, `pandas`, `matplotlib`) with **zero compiled dependencies**.
#
# - **Documentation**: [https://jalonso1979.github.io/puremacro/](https://jalonso1979.github.io/puremacro/)
# - **Source Code**: [https://github.com/jalonso1979/puremacro](https://github.com/jalonso1979/puremacro)
# - **Bilingual Guide**: [English (dsge_v3.md)](../docs/dsge_v3.md) · [Español (dsge_v3.md)](../docs/es/dsge_v3.md)
