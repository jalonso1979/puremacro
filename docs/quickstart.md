# Quickstart Guide

Get up and running with `puremacro 2.0` in less than 2 minutes. All core estimators run on pure Python, NumPy, SciPy, Pandas, and Matplotlib — no C compilers, no Fortran runtimes, and 100% Pyodide/browser compatible.

---

## 1. Local Projections (LP) with 1-Line Visualization

Estimate impulse response functions directly via Jordà (2005) predictive regressions:

```python
import numpy as np
import pandas as pd
from puremacro.lp import lp_hac

# 1. Generate synthetic data: response of output to a policy shock
rng = np.random.default_rng(42)
T = 200
shock = rng.standard_normal(T)
gdp = np.cumsum(0.7 * shock + 0.3 * rng.standard_normal(T))
df = pd.DataFrame({"gdp": gdp, "shock": shock})

# 2. Fit Local Projection up to horizon 12 with 4 lags of controls
res = lp_hac(df, y="gdp", x="shock", horizon=12, lags=4, ci=0.90)

# 3. View summary & plot IRF with confidence bands
print(res.summary())
res.plot(title="Response of GDP to Structural Shock")

# 4. Export table directly to LaTeX or Typst for your paper
print(res.to_latex())
print(res.to_typst())
```

---

## 2. Structural VAR (SVAR) with Bootstrap Bands

Estimate a VAR and identify structural shocks via Cholesky, Sign Restrictions, or Proxy Instruments:

```python
from puremacro.var.identify import cholesky_svar

# (T, 3) macro system: [Output, Inflation, Interest Rate]
Y = rng.standard_normal((200, 3)).cumsum(axis=0)

# Estimate VAR(2) and compute impulse responses with 90% bootstrap bands
res_svar = cholesky_svar(Y, p=2, horizon=16, n_boot=500, ci=0.90, seed=42)

# Inspect summary & plot impulse response of variable 0 to shock 0
print(res_svar.summary())
res_svar.plot(target_idx=0, shock_idx=0, title="Output Response to Monetary Shock")

# Export to tidy DataFrame
df_irf = res_svar.to_frame(target_idx=0, shock_idx=0)
```

---

## 3. Modern Staggered Difference-in-Differences

Estimate heterogeneity-robust treatment effects under staggered adoption without negative weighting issues:

```python
from puremacro.did import callaway_santanna

# Callaway & Sant'Anna (2021) group-time ATTs
res_did = callaway_santanna(
    panel_df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    control_group="never_treated",
    n_boot=500,
    ci=0.95,
)

print(res_did.summary())
# Dynamic event-study profile relative to adoption timing
print(res_did.att_event_study.head())
print(res_did.to_markdown())
```

---

## 4. Sequence-Space HANK Model

Solve an incomplete-markets New Keynesian model with uninsurable idiosyncratic income risk using sequence Jacobians (Auclert et al. 2021):

```python
from puremacro.models import solve_hank_sequence_space

# Solve GE equilibrium transition path to a 25 bps rate hike
res_hank = solve_hank_sequence_space(T=40, beta=0.985, phi_pi=1.5, kappa=0.1)
print(res_hank.summary())

# Inspect output and inflation impulse responses
print("Peak output drop:", res_hank.irf_output.min())
print("Bottom-decile MPC:", res_hank.mpc_distribution["Decile 1"])
```

---

## 5. Mixed-Frequency GDP Nowcasting

Track quarterly GDP growth in real time from monthly indicators with ragged edges and missing releases:

```python
from puremacro.nowcast import nowcast_gdp

res_nowcast = nowcast_gdp(monthly_indicators_df, historical_gdp_series, n_factors=2)
print(res_nowcast.summary())

# View news contribution of the latest release
print(res_nowcast.news_decomposition)
```

---

## 6. Factor-Augmented VAR (FAVAR)

Extract latent factors from high-dimensional informational panels and project policy responses back to all individual series (Bernanke, Boivin & Eliasz 2005):

```python
from puremacro.var import favar

favar_res = favar(
    panel_macro_df,
    policy_rate_series,
    n_factors=3,
    p=2,
    horizon=20,
    ci=0.90,
)
print(favar_res.summary())

# Plot responses for specific macroeconomic variables
favar_res.plot(variables=["Industrial_Production", "CPI", "Employment"])
```

---

## 7. DSGE Higher-Order Approximation & Dynare Parity

Solve nonlinear DSGE models to 1st or 2nd order with Kim et al. (2008) pruning, cross-derivatives, and Dynare `oo_.dr` compatibility:

```python
from puremacro.dsge import build_dynare, load_mod

# 1. Load native Dynare .mod file with shocks and stoch_simul options
model = load_mod("rbc.mod")

# 2. Solve 2nd-order perturbation with pruning
sol = model.solve(order=2)

# 3. Inspect policy rules matching Dynare oo_.dr structure
print(sol.oo_dr["ghx"])   # first-order state derivatives
print(sol.oo_dr["ghxx"])  # second-order state derivatives
print(sol.oo_dr.summary())

# 4. Analytical theoretical moments & variance decomposition
mom = sol.theoretical_moments()
print(mom.summary())
print(mom.to_latex())
```

---

## 8. iPad / Juno / Pyodide to Google Colab Offloader

When working on a tablet or client-side Pyodide session that hits memory or runtime constraints, generate an executable Google Colab notebook with automatic Google Drive result syncing:

```python
from puremacro.runtime.colab import (
    generate_colab_notebook,
    show_colab_offload_dialog,
    load_colab_result,
)

# 1. Generate notebook with auth and Drive mounting
nb = generate_colab_notebook(
    task_code="""
import puremacro as pm
res = pm.dsge.estimate_sw07(n_draws=10000, n_chains=4)
pm.runtime.store.save_frame(res.summary(), "sw07_posterior.pmz")
""",
    mount_drive=True,
    export_result_file="sw07_posterior.pmz",
)

# 2. Launch in Colab with 1 click
show_colab_offload_dialog(nb, filename="sw07_offload.ipynb")

# 3. Retrieve output back in local session via pure .pmz cartridge
res = load_colab_result("sw07_posterior.pmz")
```
