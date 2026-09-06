> 🇬🇧 English · 🇪🇸 [Español](es/quickstart.md)

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
import numpy as np
import pandas as pd
from puremacro.did import callaway_santanna

# Staggered adoption: 60 counties observed 2000-2011, cohorts first treated in
# 2005 and 2008 plus a never-treated group (replace with your own panel)
rng = np.random.default_rng(0)
rows = []
for county in range(60):
    g = {0: 2005, 1: 2008, 2: np.nan}[county % 3]
    for year in range(2000, 2012):
        effect = 3.0 if (not np.isnan(g) and year >= g) else 0.0
        rows.append({"county_id": county, "year": year, "first_treated_year": g,
                     "employment": 100 + 0.5 * (year - 2000) + effect + rng.standard_normal()})
panel_df = pd.DataFrame(rows)

# Callaway & Sant'Anna (2021) group-time ATTs
res_did = callaway_santanna(
    panel_df,
    unit="county_id",
    time="year",
    outcome="employment",
    treat_time="first_treated_year",
    control="never_treated",
    n_boot=200,
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
import numpy as np
import pandas as pd
from puremacro.nowcast import nowcast_gdp

# Ten years of six monthly indicators driven by one common factor, and the
# quarterly GDP history; the last month of the current quarter is only
# partly published (a ragged edge)
rng = np.random.default_rng(1)
months = pd.date_range("2016-01-31", periods=120, freq="ME")
factor = np.cumsum(rng.standard_normal(120)) * 0.3
monthly_indicators_df = pd.DataFrame(
    np.outer(factor, rng.uniform(0.5, 1.5, 6)) + 0.5 * rng.standard_normal((120, 6)),
    index=months, columns=["ip", "retail", "orders", "hours", "pmi", "exports"],
)
monthly_indicators_df.iloc[-1, 3:] = np.nan            # not yet released
quarters = pd.period_range("2016Q1", periods=39, freq="Q")
historical_gdp_series = pd.Series(
    factor.reshape(-1, 3).mean(axis=1)[:39] + 0.2 * rng.standard_normal(39), index=quarters, name="gdp",
)

res_nowcast = nowcast_gdp(monthly_indicators_df, historical_gdp_series, n_factors=2)
print(res_nowcast.summary())
print(res_nowcast.to_frame().tail())
```

---

## 6. Factor-Augmented VAR (FAVAR)

Extract latent factors from high-dimensional informational panels and project policy responses back to all individual series (Bernanke, Boivin & Eliasz 2005):

```python
import numpy as np
import pandas as pd
from puremacro.var import favar

# A (T x N) informational panel and the policy rate, simulated for the example
rng = np.random.default_rng(2)
T = 240
f = np.cumsum(rng.standard_normal(T)) * 0.1
names = ["Industrial_Production", "CPI", "Employment"] + [f"x{i}" for i in range(9)]
panel_macro_df = pd.DataFrame(
    np.outer(f, rng.uniform(0.5, 1.5, len(names))) + 0.3 * rng.standard_normal((T, len(names))),
    columns=names,
)
policy_rate_series = pd.Series(0.5 * f + 0.2 * rng.standard_normal(T), name="policy_rate")

favar_res = favar(
    panel_macro_df,
    policy_rate_series,
    n_factors=3,
    p=2,
    horizon=20,
    ci=0.90,
    n_boot=50,
)
print(favar_res.summary())

# Plot responses for specific macroeconomic variables
favar_res.plot(variables=["Industrial_Production", "CPI", "Employment"])
```

---

## 7. DSGE Higher-Order Approximation & Dynare Parity

Solve nonlinear DSGE models to 1st or 2nd order with Kim et al. (2008) pruning, cross-derivatives, and Dynare `oo_.dr` compatibility:

```python
from puremacro.dsge import load_mod

# 1. A Dynare .mod file: a path, or (as here) the source text itself
rbc_mod = """
var c k a;
varexo eps;
parameters alpha beta delta gamma rho;

alpha = 0.30;
beta  = 0.99;
delta = 0.025;
gamma = 1.0;
rho   = 0.80;

model;
  c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha - 1.0) + 1.0 - delta);
  k = exp(a) * k(-1)^alpha - c + (1.0 - delta) * k(-1);
  a = rho * a(-1) + eps;
end;

initval;
  k = 38.0;
  a = 0.0;
  c = 2.0;
end;

shocks;
  var eps; stderr 0.01;
end;
"""
model = load_mod(rbc_mod)   # first-order LinearModel (load_mod(rbc_mod, order=2) goes straight to 2nd order)

# 2. Solve the 2nd-order pruned perturbation
sol = model.solve(order=2)

# 3. Access Dynare-style decision rules (oo_.dr)
print(sol.oo_dr["ghx"])   # first-order state transition
print(sol.oo_dr["ghxx"])  # second-order state curvature
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

# 1. Package the heavy task as a self-contained notebook (auth and Drive-mount cells
#    included). The task's `result` variable is exported as a .pmz cartridge.
nb = generate_colab_notebook(
    """
import puremacro as pm
result = pm.dsge.estimate_sw07(n_draws=10000, n_chains=4)
""",
    mount_drive=True,
    save_path="sw07_offload.ipynb",
    output_filename="sw07_posterior.pmz",
)

# 2. Show the upload instructions (HTML card in Juno / Jupyter, plain text in a terminal)
show_colab_offload_dialog("sw07_offload.ipynb")

# 3. When the cartridge comes back from Google Drive, load it into the local session:
#    posterior = load_colab_result("sw07_posterior.pmz")
```
