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
# # Dynare Frontier Macroeconomic Toolbox — 2.0 & 2.1 Showcase
#
# **Can we run state-of-the-art macroeconomic DSGE modeling entirely in pure Python, without MATLAB, Dynare C++ compilation, or proprietary software?**
#
# Dynare is the standard platform for solving, simulating, and estimating dynamic stochastic general equilibrium (DSGE) models. However, its traditional workflow requires MATLAB or Octave licenses and C++ MEX compilation.
#
# With **puremacro 2.0 and 2.1**, researchers and students can parse standard Dynare `.mod` files directly, solve 1st and pruned 2nd-order perturbations, compute Forecast Error Variance Decompositions (FEVD), conduct Historical Shock Decompositions, solve occasionally binding constraints (OccBin / ZLB), execute deterministic non-linear perfect foresight simulations, and perform full Bayesian MCMC estimation—all in **100% pure Python** running on laptops, tablets, and zero-install browsers via WebAssembly.

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

from puremacro.dsge import (
    load_mod,
    build_dynare,
    compute_fevd,
    compute_shock_decomposition,
    solve_occbin,
    OccBinConstraint,
    solve_perfect_foresight,
    estimate_dsge_bayesian,
    BetaPrior,
    InvGammaPrior,
)

# %% [markdown]
# ## 1. Reading and Solving Pfeifer's Smets-Wouters (2007) Model (.mod)
#
# We load Johannes Pfeifer's canonical Smets & Wouters (2007, *AER*) model file directly (`sw07_pfeifer.mod`).
# `puremacro.dsge.load_mod` parses all declarations, model equations, steady-state relationships, and shocks blocks, automatically identifying the 15 predetermined state variables and 25 forward-looking controls.

# %%
import puremacro.dsge
# Resolve the reference .mod from the installed package so the notebook runs
# from any working directory (tools/build_notebooks.py uses notebooks/ as cwd).
mod_path = Path(puremacro.dsge.__file__).parent / "_references" / "sw07_pfeifer.mod"
m = load_mod(mod_path, order=1)

print(f"Endogenous variables : {len(m.variables)}")
print(f"Exogenous shocks     : {len(m.shocks)}")
print(f"Predetermined states : {m.n_states}")
print(f"Forward-looking jumps: {m.n_controls}")

# Execute stoch_simul to compute decision rules, theoretical moments, and IRFs
sim_res = m.stoch_simul(irf=24)
fig_irfs = sim_res.plot(variables=["labobs", "robs", "pinfobs", "dy"], shocks=["ea", "em"])
plt.show()

# %% [markdown]
# ## 2. Forecast Error Variance Decomposition (FEVD)
#
# The Forecast Error Variance Decomposition quantifies the percentage contribution of each structural innovation to the variance of forecast errors across horizons $h \in \{1, 4, 8, 16, \dots, \infty\}$.
#
# By the orthogonal VMA representation:
# $$ y_{t+h} - \mathbb{E}_t y_{t+h} = \sum_{k=0}^{h-1} \Psi_k u_{t+h-k} $$
# $$ \text{MSE}_i(h) = \sum_{j} \sum_{k=0}^{h-1} (\Psi_k[i, j])^2 \sigma_j^2 $$
# Puremacro guarantees within machine precision that variance shares sum strictly to 1.0 (100%) for all variables and horizons.

# %%
fevd_res = compute_fevd(m, horizons=[1, 4, 8, 16, 32, None])
print(fevd_res.summary())

# Plot variance shares
fig_fevd = fevd_res.plot(variables=["labobs", "robs", "pinfobs", "dy"])
plt.show()

# %% [markdown]
# ## 3. Historical Shock Decomposition
#
# Which historical shocks drove business cycle fluctuations?
# Using the Kalman smoother, puremacro reconstructs the trajectory of each observable variable into components explained by:
# 1. Steady state $\bar{y}$
# 2. Initial state decay $C A^t s_0$
# 3. Individual structural shock paths $\sum_j \text{Shock}_j(t)$

# %%
# Synthetic sample representing US business cycle fluctuations
np.random.seed(42)
T_hist = 40
data_hist = pd.DataFrame({
    "labobs": np.sin(np.linspace(0, 3 * np.pi, T_hist)) * 1.5 + np.random.randn(T_hist) * 0.2,
    "robs": np.cos(np.linspace(0, 2 * np.pi, T_hist)) * 0.8 + np.random.randn(T_hist) * 0.1,
    "pinfobs": np.sin(np.linspace(0, 2.5 * np.pi, T_hist)) * 0.5 + np.random.randn(T_hist) * 0.15,
    "dy": np.random.randn(T_hist) * 0.6,
})

decomp_res = compute_shock_decomposition(m, data_hist)
print(f"Decomposition variables: {decomp_res.variable_names}")

fig_decomp = decomp_res.plot(variable="labobs")
plt.show()

# %% [markdown]
# ## 4. Occasionally Binding Constraints & Zero Lower Bound (OccBin)
#
# When nominal interest rates hit the Zero Lower Bound ($r_t \ge -r_{ss}$), the linear approximation breaks down.
# Following Guerrieri & Iacoviello (2015, *JME*), `puremacro.dsge.solve_occbin` solves the piecewise-linear model via backward-forward regime iterations.
#
# Under normal times, monetary policy follows a standard Taylor rule:
# $$ r_t = \phi_\pi \pi_t + \phi_y y_t + \varepsilon_{r,t} $$
# When the constraint binds, the nominal rate is clamped to the floor:
# $$ r_t = -r_{ss} $$

# %%
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
cons_mod = build_dynare(nk_cons, variables=variables_nk, shocks=shocks_nk, params=params_nk, steady_state=ss_nk, check_steady_state=False)

constraint = OccBinConstraint(variable="r", threshold=-params_nk["r_ss"], operator="<")
shock_seq = np.array([0.0, -0.020])

occ_res = solve_occbin(ref_mod, cons_mod, constraint, shock_sequence=shock_seq, horizon=40)
print(occ_res.summary())

fig_occ = occ_res.plot()
plt.show()

# %% [markdown]
# ## 5. Deterministic Non-Linear Simulation / Perfect Foresight
#
# For large transitions far from the steady state (e.g. demographic shifts, green transition, sovereign debt restructurings), linear perturbation is inaccurate.
# `puremacro.dsge.solve_perfect_foresight` implements the stacked Newton-Raphson algorithm (Boucekkine 1995, Juillard 1996) with sparse block-tridiagonal Jacobian inversion.
#
# We simulate the non-linear Ramsey neoclassical growth model starting from depressed capital $k_0 = 0.5 \cdot k_{ss}$:
# $$ c_t^{-\sigma} = \beta c_{t+1}^{-\sigma} (\alpha A_t k_t^{\alpha - 1} + 1 - \delta) $$
# $$ k_t = A_t k_{t-1}^\alpha + (1-\delta) k_{t-1} - c_t $$

# %%
alpha, beta, delta, sigma = 0.33, 0.96, 0.08, 1.0
k_ss = ((1.0 / beta - (1.0 - delta)) / alpha) ** (1.0 / (alpha - 1.0))
c_ss = k_ss ** alpha - delta * k_ss
y_ss = np.array([c_ss, k_ss])

def ramsey_eqs(lead, curr, lag, exo):
    c_t, k_t = curr[0], curr[1]
    c_p, k_p = lead[0], lead[1]
    k_m = lag[1]
    A_t = exo[0]
    
    euler = c_t ** (-sigma) - beta * (c_p ** (-sigma)) * (alpha * A_t * (k_t ** (alpha - 1.0)) + 1.0 - delta)
    resource = k_t - (A_t * (k_m ** alpha) + (1.0 - delta) * k_m - c_t)
    return np.array([euler, resource])

y_init = np.array([c_ss * 0.7, 0.5 * k_ss])
exo_path = np.ones((60, 1))

pf_res = solve_perfect_foresight(ramsey_eqs, y_init=y_init, y_ss=y_ss, exogenous_path=exo_path, n_periods=60)
print(pf_res.summary())

fig_pf = pf_res.plot()
plt.show()

# %% [markdown]
# ## 6. Bayesian DSGE Estimation via Random Walk Metropolis-Hastings
#
# `puremacro.dsge.estimate_dsge_bayesian` implements the standard Dynare Bayesian estimation protocol:
# 1. Numerical mode finding (`scipy.optimize.minimize`).
# 2. Laplace approximation for proposal covariance $\Sigma = (-H)^{-1}$.
# 3. Random Walk Metropolis-Hastings MCMC sampling with adaptive scale tuning and Gelman-Rubin convergence diagnostics.

# %%
true_rho, true_sigma = 0.85, 0.02
np.random.seed(42)
y_obs = np.zeros(100)
for t in range(1, 100):
    y_obs[t] = true_rho * y_obs[t-1] + np.random.randn() * true_sigma

def log_lik_fn(theta):
    rho, sig = theta[0], theta[1]
    if not (0.01 < rho < 0.99) or sig <= 0.001:
        return -1e10
    resids = y_obs[1:] - rho * y_obs[:-1]
    return float(-0.5 * len(resids) * np.log(2 * np.pi * sig**2) - 0.5 * np.sum(resids**2) / (sig**2))

priors = {
    "rho": BetaPrior(mean=0.8, std=0.1),
    "sigma": InvGammaPrior(s=0.02, nu=4.0),
}

bayes_res = estimate_dsge_bayesian(
    log_lik_fn,
    priors=priors,
    initial_params=np.array([0.7, 0.04]),
    n_draws=400,
    n_burn=100,
    n_chains=2,
    seed=42,
)

print(bayes_res.summary())
fig_bayes = bayes_res.plot_priors_posteriors()
plt.show()

# %% [markdown]
# ## Conclusion
#
# With `puremacro`:
# 1. **Zero MATLAB / C++ dependency**: Solve and estimate canonical frontier DSGE models in pure Python.
# 2. **Dynare Parity**: Load `.mod` files, compute decision rules, theoretical moments, FEVD, historical shock decompositions, OccBin, and perfect foresight.
# 3. **Anywhere Execution**: Works natively on Apple Silicon, Linux, Windows, JupyterLite, and tablets via WebAssembly.
