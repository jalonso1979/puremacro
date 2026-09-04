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
# # What's New in puremacro 2.0: An Interactive Showcase
#
# `puremacro 2.0` unifies the empirical macro toolbox into a cohesive, production-grade
# computing engine running in **100% pure Python** (zero C/Fortran compilers) and adhering
# strictly to the **4-package Pyodide browser contract** (`numpy`, `scipy`, `pandas`, `matplotlib`).
#
# This interactive notebook walks through the primary milestone additions in 2.0:
# 1. **Unified `LPResult` Architecture** — 1-line `.plot()` and instant `.to_latex()` / `.to_typst()` exports.
# 2. **State-Dependent LP-IV** (Ramey & Zubairy 2018) — Spending multipliers with external instruments.
# 3. **Factor-Augmented VAR (FAVAR)** (Bernanke, Boivin & Eliasz 2005) — Latent factor extraction and panel IRFs.
# 4. **Modern Staggered Difference-in-Differences** — Heterogeneity-robust event studies.
# 5. **High-Speed Inference** — Multi-worker parallel bootstraps and vectorized Minnesota BVAR.
# 6. **Publication Reporting Pipeline** — Academic significance stars and manuscript table exports.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import puremacro

print(f"Loaded puremacro version: {puremacro.__version__}")

# %% [markdown]
# ## 1. Unified `LPResult` Architecture
#
# In 2.0, local projection estimators return an `LPResult` (subclass of `pd.DataFrame`).
# You get all pandas capabilities plus specialized econometric methods:
# - `.plot()` — renders publication-grade impulse response curves with shaded confidence bands in 1 line.
# - `.summary()` — clean, structured ASCII regression report.
# - `.to_latex()`, `.to_typst()`, `.to_markdown()` — instant table exports for Overleaf, Typst, and Quarto.

# %%
from puremacro.lp import lp_hac

# Synthetic macro series: response of real GDP to a monetary surprise
rng = np.random.default_rng(2026)
T = 180
shock = rng.standard_normal(T)
gdp_growth = np.cumsum(0.75 * shock + 0.35 * rng.standard_normal(T))
df_lp = pd.DataFrame({"gdp": gdp_growth, "shock": shock})

# Estimate LP up to horizon 16 with 4 lags of controls and 90% confidence interval
res_lp = lp_hac(df_lp, y="gdp", x="shock", horizon=16, lags=4, ci=0.90)

# Display tabular summary
print(res_lp.summary())

# Plot IRF with 1 line
fig, ax = plt.subplots(figsize=(7, 4))
res_lp.plot(ax=ax, title="Response of Real GDP to Monetary Shock (LP-HAC)")
plt.show()

# Export camera-ready LaTeX table
print("LaTeX Output (first 4 rows):")
print("\n".join(res_lp.to_latex().splitlines()[:6]))

# %% [markdown]
# ## 2. State-Dependent LP-IV (Ramey & Zubairy 2018)
#
# How large is the government spending multiplier during economic slack (high unemployment)
# versus normal times? Estimating this requires:
# 1. State-dependent interactions: $F(s_t) x_t$ and $(1 - F(s_t)) x_t$.
# 2. External instrumental variables: $F(s_t) z_t$ and $(1 - F(s_t)) z_t$ to resolve policy endogeneity.
# 3. Horizon-by-horizon 2SLS regressions with Newey-West HAC inference.

# %%
from puremacro.lp import lp_state_dep_iv

# Simulate Ramey-Zubairy style data
T = 200
unemployment = 6.5 + 1.5 * rng.standard_normal(T)       # State variable (slack)
military_news = rng.standard_normal(T)                  # Exogenous news instrument
gov_spending = 0.8 * military_news + 0.3 * rng.standard_normal(T) # Endogenous spending

# Outcome with state-varying transmission: stronger multiplier during high slack
is_slack = (unemployment > 6.5).astype(float)
y = np.cumsum(0.9 * (is_slack * gov_spending) + 0.4 * ((1.0 - is_slack) * gov_spending) + 0.2 * rng.standard_normal(T))
df_rz = pd.DataFrame({"gdp": y, "spending": gov_spending, "news": military_news, "unemp": unemployment})

# Estimate state-dependent multipliers with 6.5% unemployment threshold
res_rz = lp_state_dep_iv(
    df_rz,
    y="gdp",
    x="spending",
    z="news",
    state="unemp",
    threshold=6.5,
    transition="threshold",
    horizon=12,
    lags=2,
    ci=0.90,
)

print(res_rz[["h", "beta_H", "first_stage_f_H", "beta_L", "first_stage_f_L"]].head())

# %% [markdown]
# ## 3. Factor-Augmented VAR (FAVAR, Bernanke et al. 2005)
#
# Standard VARs are limited to 3-6 variables to avoid parameter explosion.
# `favar` summarizes high-dimensional informational panels (50+ macroeconomic indicators)
# into latent principal components, estimates a joint VAR on `[policy, factors]`, and
# projects structural impulse responses back onto all cross-sectional series with bootstrap bands.

# %%
from puremacro.var import favar

# Create high-dimensional informational panel (20 series, T=150)
N = 20
F_true = rng.standard_normal((T, 2))
policy_rate = np.zeros(T)
for t in range(1, T):
    policy_rate[t] = 0.6 * policy_rate[t-1] + 0.4 * F_true[t-1, 0] + 0.2 * rng.standard_normal()

loadings = rng.uniform(-1.0, 1.0, size=(N, 3))
Z = np.column_stack([policy_rate, F_true])
panel_X = Z @ loadings.T + 0.4 * rng.standard_normal((T, N))
panel_df = pd.DataFrame(panel_X, columns=[f"Macro_Var_{i+1}" for i in range(N)])

# Estimate FAVAR with 2 latent factors and 90% bootstrap bands
favar_res = favar(panel_df, policy_rate, n_factors=2, p=1, horizon=12, n_boot=50, seed=42)

print(favar_res.summary())

# Plot selected cross-sectional responses
favar_res.plot(variables=["Macro_Var_1", "Macro_Var_2"])
plt.show()

# %% [markdown]
# ## 4. Modern Staggered Difference-in-Differences
#
# Classical Two-Way Fixed Effects (TWFE) regressions break down when treatment adoption is
# staggered across units and treatment effects are dynamic or heterogeneous.
# `puremacro.did` implements heterogeneity-robust estimators:
# - `callaway_santanna` — Group-time $ATT(g, t)$ and dynamic event studies.
# - `synthetic_did` — Combines unit synthetic-control weights ($\omega$) and time weights ($\lambda$).

# %%
from puremacro.did import callaway_santanna

# Staggered adoption panel: 12 units over 8 periods
units = []
for u in range(12):
    treat_yr = 2012 if u < 4 else (2014 if u < 8 else np.nan) # NaN = never treated
    for yr in range(2008, 2016):
        d = 1.0 if not np.isnan(treat_yr) and yr >= treat_yr else 0.0
        y = 2.0 * d + 0.5 * (yr - 2008) + rng.standard_normal()
        units.append({"unit": f"U{u}", "year": yr, "treat_time": treat_yr, "outcome": y})

panel_did = pd.DataFrame(units)

# Fit Callaway-Sant'Anna
res_did = callaway_santanna(panel_did, unit="unit", time="year", outcome="outcome", treat_time="treat_time", ci=0.90)
print(res_did.summary())
print(res_did.att_event_study)

# %% [markdown]
# ## 5. Publication Reporting Pipeline & Significance Stars
#
# Format regression coefficients into publication-ready tables with academic significance stars:
# - `***` $p < 0.01$
# - `**` $p < 0.05$
# - `*` $p < 0.10$

# %%
from puremacro.reports import coef_table

betas = np.array([0.524, -1.892, 0.043])
ses = np.array([0.082, 0.410, 0.065])
varnames = ["Policy Rate", "Fiscal Spending", "Trade Openness"]

# LaTeX Table
print("LaTeX Format:")
print(coef_table(betas, ses, names=varnames, stars=True, fmt="latex"))

# Typst Format
print("\nTypst Format:")
print(coef_table(betas, ses, names=varnames, stars=True, fmt="typst"))

# %% [markdown]
# ## 6. Honest DiD Sensitivity Analysis (Rambachan & Roth 2023)
#
# Evaluate sensitivity of event-study estimates to violations of parallel trends.
# Computes identified sets, robust confidence intervals (Imbens & Manski 2004),
# and the breakdown value $M^*$ (the violation multiplier that overturns statistical significance).

# %%
from puremacro.did import honest_did_sensitivity

sens_res = honest_did_sensitivity(res_did, target_horizon=0, ci=0.90)
print(sens_res.summary())
print("\n" + sens_res.plot_ascii())

# %% [markdown]
# ## 7. 2nd-Order DSGE Perturbation with Pruning (Kim et al. 2008)
#
# Standard second-order DSGE perturbation produces explosive simulations due to
# spurious quadratic manifolds. The pruning method of Kim, Kim, Schaumburg & Sims (2008)
# decomposes the state space into first- and second-order components, ensuring
# strictly stationary trajectories, asymmetric impulse responses (GIRF), and analytical
# ergodic means.

# %%
from puremacro.dsge import canonical_growth_2nd_order

dsge_sol = canonical_growth_2nd_order()
print("First-order eigenvalues (all inside unit circle):", np.round(np.abs(dsge_sol.eigenvalues), 3))

# Simulate 100 periods with pruning
sim = dsge_sol.simulate(periods=100, seed=42)
print("\nSimulation Paths (first 5 rows):")
print(sim.to_frame().head())

# Generalized Impulse Response (GIRF) demonstrating non-linear asymmetry
girf_pos = dsge_sol.girf("eps", size=+2.0, horizon=8)
girf_neg = dsge_sol.girf("eps", size=-2.0, horizon=8)
print("\nAsymmetric GIRF of Consumption (positive vs negative shock):")
print(pd.DataFrame({
    "+2σ shock": girf_pos["c"],
    "-2σ shock": girf_neg["c"],
    "Sum (Asymmetry)": girf_pos["c"] + girf_neg["c"],
}))

# Risk-adjusted ergodic mean (stochastic steady state)
sss = dsge_sol.stochastic_steady_state(sigma=1.0)
print("\nStochastic Steady State (Precautionary Shift):")
print("Capital ergodic mean shift:", float(sss["states"]["k"]))

# %% [markdown]
# ## 8. Juno & Pyodide to Google Colab Compute Offloading
#
# Work seamlessly on iPad (Juno / Juno Connect / Safari) and offload heavy MCMC or
# large bootstrap simulations directly to Google Colab GPU/TPU with zero file transfers.

# %%
from puremacro.runtime import generate_colab_notebook, colab_auth_guide

# Print iPad authentication tips
print("iPad / Mobile Safari Authentication Guide:")
print(colab_auth_guide()[:300] + "...\n")

# Generate self-contained Colab notebook with embedded payload
nb_json = generate_colab_notebook(
    code="""
import puremacro
from puremacro.dsge import canonical_growth_2nd_order
sol = canonical_growth_2nd_order()
sim = sol.simulate(periods=10000, seed=42)
print("Colab Heavy Simulation Complete! Mean Capital:", sim.states['k'].mean())
""",
    title="puremacro_colab_heavy_task",
    save_path=None,
    mount_drive=False,
)
print("Generated self-contained Colab notebook JSON length:", len(str(nb_json)))

# %% [markdown]
# ## Conclusion
#
# `puremacro 2.0` combines the speed of modern vectorized computing with the accessibility
# of pure Python and zero-setup browser deployment.
#
# - **Documentation**: [https://jalonso1979.github.io/puremacro/](https://jalonso1979.github.io/puremacro/)
# - **Source Code**: [https://github.com/jalonso1979/puremacro](https://github.com/jalonso1979/puremacro)

