> 🇬🇧 English · 🇪🇸 [Español](es/quickstart.md)

# Quickstart Guide

Start with the local-projection example below, then choose a model or estimator.
The numerical core uses NumPy, SciPy, pandas and Matplotlib, with no compiled
extension of its own. A local workstation installation is the supported target:

```bash
python -m pip install puremacro
```

Browser execution under Pyodide is best effort. The [runtime guide](tablet.md)
explains file-format, network and compute limits; the
[notebook learning paths](notebooks.md) provide longer worked examples.

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
from puremacro.dsge import estimate_sw07
result = estimate_sw07(n_draws=10000, n_chains=4)
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

---

## 9. Forward Automatic Differentiation & Unified Model Protocol

Evaluate native Kalman sensitivity scores on explicit data and simulate state-space models via the `DifferentiableModel` protocol:

```python
import numpy as np
from puremacro.core.protocol import wrap_differentiable_model, DifferentiableModel
from puremacro.state_space import StateSpaceModel

# 1. Declare state-space system and wrap into universal protocol adapter
ssm = StateSpaceModel(
    T=np.array([[0.75]]),
    Z=np.array([[1.0]]),
    Q=np.array([[0.04]]),
    H=np.array([[0.0025]]),
)
model_diff = wrap_differentiable_model(ssm)

# 2. Evaluate the native likelihood and Kalman sensitivity score on explicit data
params = np.array([0.75, 0.20, 0.05])
data = model_diff.simulate(periods=100, seed=42)
loglik, score = model_diff.loglik_and_grad(params=params, data=data)
print(f"Log-likelihood: {loglik:.4f}")
print("Analytical score gradient:", score)

# 3. Simulate forward path and compute structural impulse responses
sim_df = model_diff.simulate(periods=30, seed=42)
irf_df = model_diff.impulse_response(shock="eps", horizon=12)
print(f"IRF length: {len(irf_df)}, Simulation length: {len(sim_df)}")
```

---

## 10. Adaptive Tempered Sequential Monte Carlo (SMC)

Estimate macroeconomic posterior distributions and the log marginal data density ($\ln p(Y)$) using gradient-guided MALA mutations:

```python
import numpy as np
from puremacro.inference.smc import smc_sample

# 1. Target log-likelihood function and analytical score
def loglik_and_grad(theta):
    diff = theta[0] - 0.45
    val = -0.5 * (diff / 0.04) ** 2
    grad = np.array([-diff / (0.04 ** 2)])
    return float(val), grad

# 2. Prior specification and adaptive tempered SMC sampling
priors = {"rho": {"dist": "uniform", "lb": -0.5, "ub": 0.95}}
res_smc = smc_sample(loglik_and_grad, priors, n_particles=200, mutation="mala", seed=42)

# 3. Log marginal data density and summary table
print(f"Log MDD: {res_smc.log_mdd:.4f} +/- {res_smc.mdd_se:.4f}")
print(res_smc.summary())
print(res_smc.to_latex())
```

---

## 11. Reduced Nonlinear HANK and Aggregate-Risk Demonstrations

Exercise a reduced DAG with prescribed MPCs and constraint fractions, and a separate aggregate-risk algebra example. These examples do not solve household policies or a distribution; see [the HANK scope guide](hank_ssj_and_aggregate_risk.md):

```python
import numpy as np
from puremacro.dsge import (
    solve_hank_nonlinear_transition,
    solve_hank_aggregate_risk_order2,
)

# 1. Solve nonlinear HANK transition path to an interest rate hike
T_hank = 30
shock_path = 0.02 * (0.65 ** np.arange(T_hank))
res_trans = solve_hank_nonlinear_transition(
    shocks={"eps_m": shock_path},
    horizon=T_hank,
    tol=1e-8,
    max_iter=30,
)
print("Transition converged:", res_trans.converged)
print(res_trans.summary())

# 2. Compute 2nd-order aggregate risk with precautionary shifts
res_risk = solve_hank_aggregate_risk_order2(
    sigma_agg=0.015,
    rho_agg=0.85,
    n_distribution_moments=4,
)
print("Precautionary wealth shift:", round(float(res_risk.precautionary_shift_wealth), 6))
print("Precautionary rate discount:", round(float(res_risk.precautionary_shift_rate), 6))
```

---

## 12. Quantitative Sovereign Default & Debt Restructuring

Compute endogenous default risk and bond price schedules in an Arellano (2008) framework and evaluate restructuring relief:

```python
import numpy as np
from puremacro.sovereign import (
    solve_sovereign_default,
    evaluate_debt_restructuring,
)

# 1. Solve sovereign default equilibrium with endogenous borrowing limit
res_sov = solve_sovereign_default(
    delta=1.0,
    kappa=0.0,
    r_star=0.01,
    beta=0.95,
    gamma=2.0,
    n_b=25,
    n_y=9,
    max_iter=150,
)
print("Sovereign solve converged:", res_sov.converged)
print(f"Mean spread: {res_sov.spread_mean * 10000:.1f} bps")

# 2. Evaluate a haircut at an explicit state; keep the short-term maturity
res_restr = evaluate_debt_restructuring(
    res_sov,
    haircut=0.20,
    maturity_extension=0.0,
    b_distress=0.20,
    y_distress=1.0,
    horizon=20,
    seed=42,
)
print(f"Spread reduction: {res_restr.spread_reduction_bps:.1f} bps")
print(f"Debt-to-GDP relief: {res_restr.debt_gdp_relief_pct:.2f}%")
```

---

## 13. Distributional Local Projections with Exact Adding-Up

Estimate dynamic cross-sectional share responses to macroeconomic shocks strictly satisfying the zero-sum adding-up constraint ($\sum \beta_b = 0$):

```python
import numpy as np
import pandas as pd
from puremacro.lp.distributional import (
    estimate_distributional_lp,
    estimate_quantile_lp,
)

# 1. Synthetic cross-sectional shares (5 quintiles summing to 1.0)
rng_dist = np.random.default_rng(42)
T_dist = 120
n_bins = 5
bin_cols = [f"Q{i+1}" for i in range(n_bins)]
raw_shares = rng_dist.dirichlet(np.ones(n_bins) * 2.0, size=T_dist)
df_dist = pd.DataFrame(raw_shares, columns=bin_cols)
df_dist["dist_shock"] = rng_dist.standard_normal(T_dist)
df_dist["gdp_growth"] = rng_dist.standard_normal(T_dist) * 0.5
df_dist["income"] = 10.0 + 2.0 * rng_dist.standard_normal(T_dist)

# 2. Fit Distributional Local Projection and verify exact adding-up
res_dlp = estimate_distributional_lp(
    df_dist,
    shock="dist_shock",
    bin_cols=bin_cols,
    horizons=range(0, 8),
    n_lags=2,
    controls=["gdp_growth"],
)
print("Adding-up maximum error:", res_dlp.adding_up_max_error)
print(res_dlp.summary())

# 3. Fit Quantile Local Projection (10th, 50th, 90th percentiles)
res_qlp = estimate_quantile_lp(
    df_dist,
    shock="dist_shock",
    y="income",
    quantiles=[0.10, 0.50, 0.90],
    horizons=range(0, 8),
    n_lags=1,
)
print("Quantile IRF matrix shape:", res_qlp.irf_matrix.shape)
```

---

## 14. Integrated Spatial Climate-Macro IAM & Zero-Server PolicyStudio

Analyze input-output bottlenecks, multi-region climate migration under carbon taxation, and export zero-server interactive visual dashboards:

```python
import numpy as np
from puremacro.trade import baqaee_farhi_decomposition
from puremacro.climate.spatial_iam import SpatialClimateIAM, solve_spatial_climate_iam
from puremacro.reports import result_to_interactive_html

# 1. Baqaee-Farhi 2nd-order production network bottleneck
Omega = np.array([[0.10, 0.05], [0.30, 0.10]], dtype=float)
b = np.array([0.40, 0.60], dtype=float)
res_net = baqaee_farhi_decomposition(
    Omega,
    b,
    elasticities=0.25,
    tfp_shocks=np.array([-0.05, 0.0]),
)
print("1st order effect:", round(float(res_net.first_order_dlog_y), 5))
print("2nd order curvature:", round(float(res_net.second_order_dlog_y), 5))

# 2. Multi-region Spatial Climate IAM with endogenous migration
iam_model = SpatialClimateIAM(
    initial_population=np.array([100.0, 150.0]),
    initial_trade_shares=np.array([[0.8, 0.2], [0.2, 0.8]]),
    initial_migration_shares=np.array([[0.95, 0.05], [0.05, 0.95]]),
    initial_temperatures=np.array([12.0, 26.0]),
    carbon_intensity=np.array([0.3, 0.7]),
    damage_coefficients=np.array([0.005, 0.02]),
    region_names=("North", "South"),
)
res_iam = solve_spatial_climate_iam(iam_model, horizon=10, policy_scenario="laissez_faire")
print("Population conservation error:", res_iam.population_conservation_error)

# 3. Zero-server self-contained HTML5 dashboard export
html_doc = result_to_interactive_html(res_net, title="Supply Chain Bottleneck Analysis")
print("HTML dashboard generated bytes:", len(html_doc))
```
