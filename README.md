> 🇬🇧 English · 🇪🇸 [Español](README.es.md)

# puremacro

A **Pyodide-compatible empirical macroeconomics toolbox**: the estimator
code runs on pure numpy + scipy + pandas + matplotlib, so the numerical
core stays importable under Pyodide (iPad / juno.sh, best-effort — see
"juno.sh / iPad" below). The supported target is a local install on a
regular workstation.

## 5-Minute Quickstart (2.0 Unified API)

`puremacro 2.0` standardizes the macro API around common parameter conventions (`lags`, `horizon`, `ci`), frozen-dataclass result objects, rich visualization (`.plot()`), and direct publication export (`.to_latex()`, `.to_typst()`, `.to_markdown()`):

### 1. Local Projections (LP) & Publication-Grade Export
```python
import numpy as np
import pandas as pd
from puremacro.lp import lp_hac

# Synthetic macro time series
rng = np.random.default_rng(42)
T = 200
shock = rng.standard_normal(T)
gdp = np.cumsum(0.7 * shock + 0.3 * rng.standard_normal(T))
df = pd.DataFrame({"gdp": gdp, "shock": shock})

# Unified API: horizon, lags, confidence level
res = lp_hac(df, y="gdp", x="shock", horizon=12, lags=4, ci=0.90)

# Instant visualization
res.plot(title="Output Response to Monetary Shock")

# Direct table export to LaTeX tabular or Typst table
print(res.to_latex())
print(res.to_typst())
```

### 2. DSGE Higher-Order Approximation & Dynare Parity
Solve nonlinear DSGE models up to second order with Kim, Kim, Schaumburg & Sims (2008) pruning, cross-derivatives ($g_{xu}, g_{uu}$), risk corrections ($g_{\sigma\sigma}$), and Dynare `oo_.dr` parity:
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

### 3. Bayesian DSGE Estimation via NUTS & Exact Analytic Gradients (puremacro 3.0)
Perform Hamiltonian Monte Carlo via the No-U-Turn Sampler (NUTS) using exact analytical Kalman likelihood score gradients ($\nabla_\theta \ln L$ via implicit state-space differentiation and generalized Sylvester solvers):
```python
# requires: numpy scipy
import numpy as np
import pandas as pd
from puremacro.dsge import load_mod, GammaPrior

# Load standard DSGE model and prepare estimation
mod_text = """
var y pi i; varexo eps_d eps_m;
parameters sigma kappa phi_pi rho_d;
sigma = 1.0; kappa = 0.1; phi_pi = 1.5; rho_d = 0.5;
model;
  y = y(+1) - (1/sigma) * (i - pi(+1)) + eps_d;
  pi = 0.99 * pi(+1) + kappa * y;
  i = phi_pi * pi + eps_m;
end;
"""
model = load_mod(mod_text)
# Simulate synthetic observables
sim = model.stoch_simul(periods=100, seed=42)
data = pd.DataFrame({"y": sim.paths["y"], "pi": sim.paths["pi"]})

# Bayesian NUTS estimation with exact analytic likelihood gradients
res = model.estimate(
    data,
    varobs=["y", "pi"],
    priors={"sigma": GammaPrior(1.0, 0.2), "kappa": GammaPrior(0.1, 0.05)},
    method="nuts",
    n_draws=200,
    burn_in=100,
    n_chains=2,
    seed=42,
)
print(res.summary())
```

### 4. Heterogeneous-Agent (HANK) Sequence-Space Bridge (puremacro 3.0)
Couple microeconomic heterogeneous-agent household blocks (`hetagent_block; ... end;`) with aggregate general-equilibrium DSGE conditions in Dynare `.mod` files using the Auclert et al. (2021) Sequence-Space Jacobian method:
```python
# requires: numpy scipy
from puremacro.dsge import solve_hank_bridge

hank_mod = """
var Y C r pi i; varexo eps_m;
parameters beta gamma r_ss phi_pi kappa;
beta = 0.985; gamma = 1.0; r_ss = 0.01; phi_pi = 1.5; kappa = 0.1;

hetagent_block;
  model = one_asset_hank;
  n_a = 50; a_max = 30.0; borrowing_limit = 0.0; grid = hyperbolic;
end;

model;
  Y = C;
  pi = beta * pi(+1) + kappa * Y;
  i = r_ss + phi_pi * pi + eps_m;
  r = i - pi(+1);
end;
"""
# One-line transition solve for an expansionary monetary policy shock (-25 bp)
res = solve_hank_bridge(hank_mod, shock="eps_m", magnitude=-0.0025, horizon=40)
print(res.summary())
```

### 5. Juno / iPad / Pyodide to Google Colab Offloading
When working on an iPad or client-side Pyodide session with compute or memory constraints, seamlessly offload heavy tasks (e.g. 10,000-draw MCMC or large bootstrap SVARs) to Google Colab:
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

### 6. Continuous Dynamic Programming & Projection Solvers (puremacro 3.3)
Solve continuous state-space dynamic programming models via orthogonal Chebyshev polynomial collocation or Finite Element Galerkin projection with Fischer-Burmeister NCP complementarity:
```python
from puremacro.vfi import CollocationProblem, solve_collocation

# Continuous neoclassical growth model: u(c)=ln(c), f(k)=k^alpha, delta=1.0
alpha, beta, delta = 0.36, 0.96, 1.0
k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
domain = (0.5 * k_ss, 1.5 * k_ss)

prob = CollocationProblem(
    domain=domain,
    orders=7,
    method="euler",
    params={"alpha": alpha, "delta": delta},
    beta=beta,
)
sol = solve_collocation(prob)
print(f"Collocation converged: {sol.converged}")
print(f"Policy at steady state: {sol.policy(k_ss):.4f}")
```

### 7. Deep Macro Physics-Informed Neural Networks (puremacro 3.3)
Solve ultra-high-dimensional continuous dynamic general equilibrium models (10+ continuous state variables) using Physics-Informed Neural Networks (PINNs) trained along simulated ergodic state trajectories in pure NumPy:
```python
from puremacro.vfi import DeepMacroModel, solve_deep_macro

# 10-country dynamic capital accumulation model (10 continuous state variables)
model = DeepMacroModel.multi_country_growth(
    n_countries=10, alpha=0.36, beta=0.96, delta=0.08, gamma=2.0,
)

# Solve in pure NumPy via ergodic trajectory sampling (Maliar et al. 2021)
sol = solve_deep_macro(
    model=model,
    hidden_dims=(32, 32),
    n_epochs=20,
    batch_size=64,
    trajectory_length=200,
    seed=42,
    verbose=False,
)
print(f"PINN converged: {sol.converged}")
print(f"Final training loss: {sol.loss_history[-1]:.4e}")
```

### 8. Quantitative Spatial Economics & Trade General Equilibrium (puremacro 3.3)
Evaluate the general equilibrium welfare, trade flow, and real wage impacts of tariffs, supply chain disruptions, and transport infrastructure investments:
```python
import numpy as np
from puremacro.spatial import AllenArkolakisModel

# 3-region continuous economic geography model (North, South, East)
coords = np.array([[45.0, -93.0], [30.0, -90.0], [40.7, -74.0]])
model = AllenArkolakisModel.from_coordinates(
    coords,
    region_names=["North", "South", "East"],
    theta=4.0,
    alpha=0.08,
    beta=-0.35,
    total_population=100.0,
)

# Simulate infrastructure investment reducing bilateral transport friction by 20%
res = model.simulate_infrastructure_shock(origin="North", destination="South", cost_reduction=0.20)
print(f"Spatial GE converged: {res.converged}")
print(f"Aggregate welfare change: {res.welfare_pct:+.4f}%")
```

### 9. Double / Debiased Machine Learning (puremacro 3.4)
Estimate a treatment effect with high-dimensional macro controls via Chernozhukov et al. (2018) partially linear regression: cross-fitted Robinson orthogonal scores, pure-NumPy penalized learners, and result objects that export to LaTeX, Typst and Markdown. The learners are **lasso and ridge only** — there is no elastic-net learner:
```python
import numpy as np
from puremacro.causal import dml_plr

# 500 observations, 50 controls; the true treatment effect is 1.5.
rng = np.random.default_rng(0)
n, p = 500, 50
X = rng.normal(size=(n, p))
beta = np.r_[1.0, 0.5, 0.25, np.zeros(p - 3)]
D = X @ beta + rng.normal(size=n)
Y = 1.5 * D + X @ beta + rng.normal(size=n)

res = dml_plr(Y, D, X, n_folds=5, learner="lasso")
print(f"theta = {res.theta:.4f}  se = {res.se:.4f}")
print(f"95% CI = [{res.ci_lower:.4f}, {res.ci_upper:.4f}]")
# theta = 1.4959  se = 0.0485
# 95% CI = [1.4009, 1.5909]
```
The class behind `dml_plr` is `DoubleMLPLR`. NaN / infinite inputs, more folds than
observations and arrays above two dimensions raise `ValueError`; a training fold with
`n_train <= p + 1` emits a `UserWarning` rather than returning a silently unidentified
estimate. A learner passed as an instance is deep-copied per fold, so the folds never
share fitted state, and `.to_latex()` compiles with `booktabs` alone.

### 10. Implicit Continuous-Time HJB, Adjoint KFE & Aiyagari GE (puremacro 3.4)
Solve the Achdou, Han, Lasry, Lions & Moll (2022) implicit upwind M-matrix system, then the adjoint Kolmogorov forward equation for the stationary density on the same grid, then the continuous-time Aiyagari general equilibrium:
```python
from puremacro.vfi import solve_hjb_achdou, solve_aiyagari_continuous_hjb

# Implicit upwind HJB with two-state Poisson income switching (default lambda = 0.1),
# plus the adjoint KFE stationary density g(a, z) on the same grid.
sol = solve_hjb_achdou(r_rate=0.03, w_rate=1.0, rho_val=0.05, Na=200, a_max=20.0)
print(f"HJB iterations: {sol.n_iter}   |mass - 1| = {sol.mass_residual:.1e}")
print(f"g_dist is a density of shape {sol.g_dist.shape} (mass per unit of a)")

# `converged` means |K^s - K^d| <= tol_ge, and tol_ge governs the GE loop.
ge = solve_aiyagari_continuous_hjb(Na=120, a_max=20.0, tol_ge=1e-4)
print(f"GE converged: {ge.converged}   r* = {ge.r_star:.6f}   K* = {ge.K_star:.4f}")
print(f"|K^s - K^d| = {abs(ge.excess_capital):.2e} <= tol_ge = 1e-4")
# HJB iterations: 9   |mass - 1| = 0.0e+00
# g_dist is a density of shape (200, 2) (mass per unit of a)
# GE converged: True   r* = 0.020571   K* = 5.9977
# |K^s - K^d| = 2.41e-07 <= tol_ge = 1e-4
```
`g_dist` is a **density**, not node mass: on a non-uniform `a_grid` the node mass is
divided by the cell width, so `sum_i g_i w_i = 1` with the trapezoid quadrature weights
`w`. The Poisson switching between the states of `e_grid` is new in 3.4: the 3.3.0
explicit solver carried the same two income states but no switching term, so they were
independent deterministic-income problems.

### 11. Montiel Olea-Pflueger Weak-IV Inference (puremacro 3.4)
`lp_iv` reports the effective F-statistic with its critical values, and inverts the Anderson-Rubin set exactly instead of scanning a fixed grid — the closed-form tail limit classifies it as a bounded interval, two rays, the whole line or empty, and the endpoints are Brent roots (a closed-form quadratic when there is a single instrument):
```python
import numpy as np
import pandas as pd
from puremacro.lp import lp_iv
from puremacro.lp.iv import mop_critical_values

# The closed form behind the MOP table: cv = chi2_{K, 1-alpha}(K / tau) / K.
cv10, cv20 = mop_critical_values(1)
print(f"K=1: 10% worst-case bias {cv10:.2f}, 20% {cv20:.2f}")

rng = np.random.default_rng(7)
T = 240
z1, z2 = rng.normal(size=T), rng.normal(size=T)
u = rng.normal(size=T)
x = 0.8 * z1 + 0.5 * z2 + u
y = np.cumsum(0.4 * x + u + rng.normal(size=T))
df = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2})

res = lp_iv(df, y="y", x="x", z=["z1", "z2"], horizon=3, weak_iv_robust=True)
print(res[["h", "beta", "se", "mop_f", "ar_lo", "ar_hi", "ar_set_type"]]
      .round(3).to_string(index=False))
# K=1: 10% worst-case bias 23.11, 20% 15.06
#  h  beta    se  mop_f  ar_lo  ar_hi ar_set_type
#  0 0.347 0.123 89.978  0.095  0.546     bounded
#  1 0.155 0.187 88.555 -0.257  0.495     bounded
#  2 0.186 0.244 84.872 -0.188  0.481     bounded
#  3 0.093 0.260 81.220 -0.481  0.548     bounded
```
The errors are heteroskedasticity- and autocorrelation-robust (Newey-West with the
horizon's bandwidth). **`lp_iv` has no cluster argument: neither the effective F it
reports nor the Anderson-Rubin set it inverts is cluster-robust.** Only
`inference.weak_iv.olea_pflueger_f(x, z, cluster=...)` takes clusters, and it returns
the effective F alone — there is no cluster-robust confidence set anywhere.

### 12. Latin American Real-Time Macro Data (puremacro 3.4)
Banco de México, INEGI, Banco Central do Brasil and Banco Central de Chile join the real-time providers. The catalog, the series identifiers and the vintage semantics all resolve offline, before any request is made:
```python
from puremacro.fetch.realtime import (
    available_providers, vintage_catalog, resolve_series, VINTAGE_SEMANTICS,
)

print(available_providers())

cat = vintage_catalog()
latam = cat[cat["provider"].isin(["banxico", "inegi", "bcb", "bcch"])]
print(latam.query("variable == 'cpi'")[["provider", "country", "series_id", "freq"]]
           .to_string(index=False))
print(resolve_series("banxico", "MEX", "cpi"))
print(VINTAGE_SEMANTICS["bcch"][:82])
# ['alfred', 'banxico', 'bcb', 'bcch', 'bundesbank', 'ecb_rtd', 'inegi', 'oecd_stes', 'ons', 'statcan']
# provider country            series_id freq
#  banxico     MEX                  SP1    M
#      bcb     BRA                  433    M
#     bcch     CHL F074.IPC.IND.Z.Z.C.M    M
#    inegi     MEX               628197    M
# SP1
# Banco Central de Chile Base de Datos Estadísticos (SIETE API). Upstream overwrites
```
**These four sources publish only the current edition of a series, so their vintage
date is the local *snapshot* date — the day this machine fetched the series and stored
it.** Revision history therefore accumulates only across repeated captures on different
days; it is not a publication archive like ALFRED or the ONS workbook. Catalog
entries that could not be confirmed against the live API are marked `VERIFY_ONLINE`.

### 13. GPU / Apple-Silicon Trade Equilibrium (puremacro 3.4)
The Torch and MLX trade solvers are imported lazily, so nothing heavy loads until you ask for a device, and their tariff defaults now match the NumPy solver:
```python
import sys
import puremacro.trade.gpu as gpu

print("torch in sys.modules:", "torch" in sys.modules)
print("matplotlib in sys.modules:", "matplotlib" in sys.modules)
print("has_torch:", gpu.has_torch(), " has_mlx:", gpu.has_mlx())
print("device:", gpu.detect_device())
print("selected:", gpu.select_compute_device())
# torch in sys.modules: False
# matplotlib in sys.modules: False
# has_torch: True  has_mlx: True
# device: <DeviceInfo: torch:mps (Apple Metal (arm), 36.0 GB, f32-only, UMA)>
# selected: ('torch', 'mps')
```
The output above is from a fresh interpreter on an Apple-Silicon machine. The last three
lines are hardware-specific: with neither framework installed, `has_torch()` and
`has_mlx()` are `False` and the GPU tests skip instead of failing.

---

## What's in it

**Core econometrics**

- **VAR** — reduced-form OLS, BVAR (Minnesota), VECM (Engle-Granger /
  Johansen), TVP-VAR, panel-VAR; IRF / FEVD / GFEVD; residual,
  block, moving-block, and wild bootstrap bands.
- **SVAR identification** (`var.identify.*`) — Cholesky, Blanchard-Quah,
  sign restrictions (Rubio-Ramirez-Waggoner-Zha), sign + zero
  restrictions (Arias-Rubio Ramirez-Waggoner), sign-robust bands
  (Giacomini-Kitagawa), proxy / external instruments, max-share / news,
  heteroskedasticity (Rigobon), non-Gaussian (Lanne-Meitz-Saikkonen).
  All public estimators return frozen-dataclass `…Result` objects.
- **Local projections** (`lp.*`) — single-country LP-HAC, LP-IV,
  lag-augmented LP (Plagborg-Møller-Wolf), panel LP with cluster /
  Driscoll-Kraay SE, state-dependent LP, smoothed LP (Barnichon-
  Brownlees B-splines), asymmetric LP (Tenreyro-Thwaites), LP-GARCH-
  state, LP-GARCH-in-mean, mean-group, CCE, quantile LP.
- **Inference** (`inference.*`) — central HAC OLS, Newey-West, Kiefer-
  Vogelsang fixed-b, Driscoll-Kraay; weak-IV diagnostics (Cragg-Donald,
  Kleibergen-Paap, Anderson-Rubin, Montiel Olea-Pflueger); Hansen-J /
  Stock-Yogo over-id; Pesaran CD, Swamy slope-homogeneity, Quandt-
  Andrews structural breaks, specification curves.
- **Other estimators** — Diebold-Yilmaz spillover index;
  Diebold-Mariano / Giacomini-White forecast comparison + density-
  forecast scoring (CRPS, log score); Bai-Perron breaks; unit-root
  tests (ADF, KPSS, PP, Zivot-Andrews); Klein QZ solver for linear DSGE
  and 2nd-order perturbation with Kim et al. (2008) pruning, cross-terms,
  and Dynare `oo_.dr` parity (`dsge.dynare`).

**Frontier methods (2.3)**

- **Narrative sign restrictions** (`var.identify.identify_narrative_sign`) — Antolín-Díaz & Rubio-Ramírez (2018) shock-sign and historical-contribution restrictions with importance weights; see `docs/narrative_sign_svar.md`.
- **Honest DiD** (`did.honest_did`) — Rambachan & Roth (2023) smoothness and relative-magnitude sensitivity sets with fixed-length confidence intervals and breakdown values; see `docs/honest_did.md`.
- **Smooth local projections** (`lp.smooth_lp`) — Barnichon & Brownlees (2019) penalised B-spline LPs with GCV/AIC/BIC/CV smoothing selection; see `docs/smooth_lp.md`.
- **Non-linear sequence-space HANK** (`models.hank_sequence_space.solve_nonlinear_transition`) — Broyden transitions for large MIT shocks on genuine household Jacobians; see `docs/hank_nonlinear.md`.
- **Gertler-Karadi (2011)** (`dsge.gertler_karadi`) — banking frictions DSGE under Klein and OccBin; see `docs/gertler_karadi.md`.
- **BVAR with stochastic volatility** (`var.bvar_sv`) — KSC mixture + FFBS Gibbs sampler, split-R̂, volatility-conditioned IRFs, hold-out log scores and fan charts; see `docs/bvar_sv.md`.

**Modern macro extensions**

- **Staggered DiD** (`did.*`) — Callaway-Sant'Anna, Sun-Abraham,
  Borusyak-Jaravel-Spiess, Synthetic-DiD; bootstrap SE throughout.
- **Dynamic-panel GMM** (`dynpanel.*`) — Arellano-Bond, Blundell-Bond
  two-step Windmeijer + Hansen-J + AR(1)/AR(2) + Roodman collapse.
- **High-frequency monetary surprises** (`hfi.*`) — Gertler-Karadi 2015,
  Nakamura-Steinsson 2018, Jarociński-Karadi 2020.
- **Volatility** (`volatility.*`) — `SigmaObject` (1:1 port of the MAV
  MATLAB class with extended decomposition API), BEKK, CCC, HAR-RV,
  range-based, ARCH-LM / Ljung-Box diagnostics.
- **Nowcasting** (`nowcast.*`) — Kalman-DFM (Doz-Giannone-Reichlin)
  with ragged-edge handling, Mariano-Murasawa MF-VAR, forecast
  combinations, probabilistic scoring rules.
- **Growth-at-risk** (`gar.*`) — quantile AR, ABG 2019 skew-t fit, NFCI-
  style FCI.
- **Cycles / cointegration / factors** — Hamilton 2018 trend-cycle
  filter (`cycles`), Phillips-Hansen FM-OLS / Stock-Watson DOLS /
  Phillips-Ouliaris (`cointegration_modern`), PCA factors + Bai-Ng IC
  (`factor`), MIDAS (`midas`), KORV (2000) system-GMM CES (`korv_gmm`),
  synthetic control + placebo inference (`synthetic_control`).
- **Spectral / wavelet** (`spectral`, `wavelet`) — Welch PSD / cross-
  spectrum / coherence (numpy.fft only); MODWT-Haar wavelet variance
  decomposition.
- **Realized volatility** (`realized_vol`) — realized variance, bipower
  variation, Corsi HAR-RV.
- **Heterogeneous-agent / VFI / Sequence-Space HANK** (`vfi.*`, `models.hank_sequence_space`) — value-function iteration with
  EGM, finite-horizon life-cycle, OLG, Krusell-Smith aggregate shocks,
  Hopenhayn firm entry/exit, Epstein-Zin, permanent types, transition
  paths, and method-of-moments estimation. In addition, full sequence-space
  HANK (Auclert et al. 2021) featuring the $\mathcal{O}(T^2)$ Fake News
  Algorithm (`fake_news_algorithm`, `FakeNewsResult`) and targeted fiscal
  transfer simulations across wealth deciles (`simulate_targeted_transfer`,
  `FiscalTransferResult`); numpy reference backend with optional
  numba / mlx / cupy acceleration. See `notebooks/` for a showcase suite.

**Continuous dynamic programming & projection engines (puremacro 3.3)**

- **Continuous projection solvers** (`vfi.collocation`, `vfi.fem`) — Chebyshev orthogonal polynomial collocation (Gauss and Lobatto extrema nodes) and Finite Element Galerkin projection with Fischer-Burmeister complementarity ($\Psi_{\text{FB}}^\epsilon(a, b) = 0$) for borrowing kinks ($k' \ge \bar{k}$); analytical Brock-Mirman neoclassical growth benchmark.
- **Shape-preserving splines & sparse grids** (`vfi.splines`, `vfi.smolyak`) — Cox-De Boor cubic B-splines, Schumaker (1983) shape-preserving quadratic splines ensuring strict monotonicity and concavity preservation ($\partial g / \partial k \ge 0$), and Smolyak (1963) sparse grid collocation scaling as $\mathcal{O}(N (\log N)^{d-1})$ for multidimensional models ($d \in [2, 6]$).
- **Discrete choice EGM** (`vfi.dcegm`) — Iskhakov, Jørgensen, Rust & Schjerning (2017) DC-EGM for models with discrete retirement/labor choices and continuous savings, featuring the fast Upper Envelope algorithm to prune sub-optimal branches, and extreme value taste shocks.
- **Continuous stationary distributions & GE** (`vfi.continuous_distribution`) — Young (2010) non-stochastic continuous density simulation preserving mass conservation to machine precision ($\sum \mu^* = 1.0 \pm 10^{-15}$), power iteration / sparse linear solvers, and Aiyagari continuous general equilibrium factor price root-finding ($K^s(r^*) = K^d(r^*)$).
- **Continuous transition dynamics under MIT shocks** (`vfi.continuous_transition`) — Non-linear transition paths coupling backward continuous EGM with forward time-dependent Young distribution operators ($\mu_{t+1} = T_t^* \mu_t$), solved via sequence-space Broyden Quasi-Newton.
- **Exact analytic IFT gradients** (`vfi.analytic_gradients`) — Machine-precision parameter sensitivities $\nabla_\theta c^*$ computed via the Implicit Function Theorem in a single linear solve, delivering $60\times+$ speedups over finite differences for structural estimation (GMM / SMM).
- **Deep Macro & PINNs** (`vfi.deep_macro`) — Physics-Informed Neural Networks in pure NumPy for ultra-high-dimensional dynamic models (10+ continuous states), bounded activations guaranteeing physical resource feasibility ($c > 0, k' > 0, c < W$), and Maliar et al. (2021) ergodic trajectory sampling.

**Quantitative spatial economics & trade general equilibrium (puremacro 3.3)**

- **Caliendo-Parro (2015) exact hat algebra** (`trade.caliendo_parro`, `spatial.caliendo_parro`) — Multi-country, multi-sector trade general equilibrium with input-output linkages, intermediate goods, and tariffs solved without estimating unobserved fundamentals (`CaliendoParroModel`).
- **Allen-Arkolakis (2014) spatial equilibrium** (`spatial.allen_arkolakis`, `trade.allen_arkolakis`) — Continuous geographic general equilibrium with bilateral iceberg trade costs, mobile labor, Marshallian agglomeration ($\alpha$), and amenity congestion ($\beta$) (`AllenArkolakisModel`).

**Macroeconomic Policy Simulators & Browser Laboratories (puremacro 4.1)**

- **Quantitative Trade Policy Simulator** (`models.trade_policy.TradePolicySimulator`) — Caliendo-Parro (2015) multi-country multi-sector Ricardian GE simulator with Eaton-Kortum gravity, input-output linkages, bilateral tariffs, terms-of-trade effects, and exact hat algebra counterfactuals with verified goods and factor market clearing ($\max_i |X_i - (Y_i + R_i + D_i)| < 10^{-6}$).
- **Monetary & Macroprudential Transmission Simulator** (`models.monetary_transmission.MonetaryTransmissionSimulator`) — Side-by-side comparative HANK vs. RANK transmission engine with Kaplan-Moll-Violante (2018) direct vs. indirect decomposition across 10 empirical MPC wealth deciles.
- **Interactive Browser Laboratories** (`curso/site/labs/`) — Static, zero-build client-side WebAssembly/Canvas laboratories: `comercio-aranceles.html` (tri-lateral trade war) and `politica-monetaria-hank.html` (HANK vs. RANK monetary transmission). See `docs/policy_simulators.md`.

**Causal ML, continuous-time HJB, multi-constraint OccBin & regional real time (puremacro 3.4)**

- **Double / Debiased Machine Learning** (`causal.dml`) — Chernozhukov et al. (2018) partially linear regression with Neyman-orthogonal Robinson scores, $K$-fold cross-fitting and pure-NumPy penalized learners (`LassoCoordinateDescent`, `RidgeGCV`); the entry points are `dml_plr` and `DoubleMLPLR`, and the learners are **lasso and ridge only**. `DMLResult` carries `.summary()`, `.plot()`, `.to_latex()`, `.to_typst()` and `.to_markdown()`.
- **Implicit continuous-time HJB & adjoint KFE** (`vfi.hjb_achdou`) — Achdou, Han, Lasry, Lions & Moll (2022) implicit upwind M-matrix scheme solved with `scipy.sparse.linalg.spsolve`, two-state Poisson income switching (default intensity $\lambda = 0.1$, new in 3.4), and the stationary density from $A^\top g = 0$. `solve_kfe_achdou` returns a **density** — node mass divided by the cell width — normalized with trapezoid quadrature weights, so it integrates to one on non-uniform grids too. `solve_aiyagari_continuous_hjb` closes the loop in general equilibrium, with `tol_ge` governing convergence and `converged` meaning $|K^s - K^d| \le$ `tol_ge`.
- **Multi-constraint OccBin** (`dsge.solve_multiconstraint_occbin`) — $M \ge 2$ simultaneous occasionally binding constraints (a zero lower bound and a collateral limit at once). Shipped in 2.8.0; 3.4 extends and hardens it: the regime splice carries **every** differing row, `result.regimes[t]` is a bitmask (bit $k$ set when constraint $k$ binds, so `3` means both bind), non-convergence returns `converged=False` *with* a `UserWarning` naming each reason, and a constraint that cannot be paired one-to-one with its model raises `ValueError`.
- **Montiel Olea-Pflueger weak-IV inference** (`lp.iv`, `inference.weak_iv`) — the effective $F$-statistic with critical values from the MOP closed form $\mathrm{cv} = \chi^2_{K,\,1-\alpha}(K/\tau)/K$ (23.11 and 15.06 for $K = 1$ at $\tau$ = 10% and 20%), and a multi-instrument Anderson-Rubin set inverted exactly — bounded interval, two rays, the whole line, or empty — rather than read off a grid. Errors are heteroskedasticity- and autocorrelation-robust; **`lp_iv` takes no clusters and no confidence set is cluster-robust** (`inference.weak_iv.olea_pflueger_f` accepts a `cluster` argument for the effective F alone). `la_lp_iv` is re-exported from `puremacro.lp`.
- **Latin American real-time connectors** (`fetch.realtime.{banxico,inegi,bcb,bcch}`) — Banco de México (SIE), INEGI (BIE), Banco Central do Brasil (SGS) and Banco Central de Chile (SIETE), with separate user and password resolution for BCCh (`PASSWORD_ENV_VARS`, `get_password`), a configurable schema-drift canary (`DRIFT_POLICIES`, `handle_drift`), `use_cache=False` that really bypasses the cache, and offline `.pmz` cartridges. **These four sources publish only the current edition, so their vintage date is the local snapshot date and revision history accumulates only across repeated captures on different days.** Catalog entries not yet confirmed against the live API are marked `VERIFY_ONLINE`.
- **GPU / Apple-Silicon trade solvers** (`trade.gpu`) — Torch and MLX paths for the Caliendo-Parro equilibrium, imported lazily so `import puremacro.trade` pulls in neither. Tariff defaults match the NumPy solver, the batched-Jacobian layout is validated rather than inferred from a magic batch size, and the tests skip (not fail) when torch / mlx or the private ICIO data are absent.

**Narrative econometrics** (`narrative.*`)

Fiscal- / labor- / uncertainty-narrative IV pipeline: canonical
`NarrativeEvent` / `NarrativeInstrument` schemas, deduplication,
keyword and manual scoring backends, panel construction, replication
loaders for canonical datasets (Romer-Romer, Mertens-Ravn). LLM
scoring backend (`narrative.scoring.llm`) and HTTP source modules
(`narrative.sources.*`) live as out-of-Pyodide side-channels.
Sources include:
- **Beige Book** — Fed Beige Book corpus from federalreserve.gov modern
  + FOMC historical pages, with per-canonical-section + per-district
  parsing (`puremacro.narrative.sources.iter_beige_book`,
  `puremacro.narrative.indices.bbui`).
- **US executive narrative** — Economic Report of the President
  (`iter_erp`), State of the Union (`iter_sotu`), and CBO reports
  (`iter_cbo`); three matching indices `erpui`, `sotuui`, `cboui`.
  CBO body fetches transparently fall back to the Wayback Machine
  when cbo.gov returns a DataDome challenge.
- **EU legislative narrative** — EUR-Lex binding acts (`iter_eurlex`)
  and EU Parliament plenary verbatim (`iter_ep_debates`); two trilingual
  EN/DE/FR indices `eurlex_ui` and `ep_ui`. EUR-Lex enumeration via
  the public Cellar SPARQL endpoint (Wayback-routed per-act fetch
  due to AWS-WAF on the live site); EP via Wayback CDX with coverage
  back to Term 7 (2009-07-14).
- **Bluesky archive** — central-bank governors + finance ministers via
  AT Protocol (`iter_bluesky_posts`, `bluesky_ui`). Hand-curated 29-
  handle seed list (`BLUESKY_KNOWN_HANDLES`); 12 resolved as of 2026-05-25.
  Multilingual via `languages=...` connector kwarg; the index defaults
  to monthly actor-level text aggregation (`aggregate_to="actor_month"`)
  to mitigate LUI's short-text degradation.
- **Cross-source disagreement** — `consensus_disagreement` computes
  the cross-sectional mean + std over any subset of narrative indices;
  `CROSS_SOURCE_GROUPS` documents thematic subsets.

Connectors hit by WAF / bot-protection (EUR-Lex, EU Parliament, CBO) fall back
to the Wayback Machine via the shared `puremacro.narrative.sources._wayback`
helper. Coverage is constrained by what Wayback has snapshotted.

**Data pipelines** (newly absorbed; see `ARCHITECTURE.md`)

- **Fetchers** (`fetch.*`) — FRED / ALFRED, SDMX-CSV
  (OECD, Eurostat, ECB, IMF SDMX-Central), EPU / GPR / WUI / JLN /
  Fernald, OECD-MEI / QNA / Energy / FX, ILOSTAT, Yahoo, WB pink
  sheet, plus per-state FRED loaders for the US subnational track.
- **Long national accounts** (`fetch.qna_long_panel`) — the OECD spine extended
  backwards per country by ratio-splicing archived national vintages onto it:
  **Spain to 1970Q1** (+100 quarters) and **Japan to 1955Q2** (+155), with
  provenance per series per quarter. The splice preserves the old vintage's
  growth rates and reports how stable each join's ratio is, because a ratio
  that drifts across the overlap means the two vintages disagree about growth
  and the spliced level depends on the anchor. Seven other candidate sources
  were measured to buy zero quarters and the reasons are kept in
  `LONG_PANEL_KNOWN_GAPS`. See `docs/long_panel.md`.
- **Real-time data** (`fetch.vintage_panel`, `fetch.realtime.*`) — published
  *editions* of a series across ten providers behind one call: the OECD STES
  revisions archive (42 economies, monthly editions from 1999), ALFRED, the
  Bundesbank Gerda database, the ONS real-time workbook (746 editions back to
  1961), Statistics Canada's vintage tables, the ECB/EABCN database, and the
  four Latin American connectors added in 3.4 (Banco de México, INEGI, Banco
  Central do Brasil, Banco Central de Chile). Comes
  with the revision toolkit — revision triangles, first/latest release,
  `r_t = y_f - y_p`, and the Mankiw-Shapiro news-vs-noise test
  (`vintages.mankiw_shapiro`). Each provider documents what its vintage date
  actually means, because they disagree: the first six carry a genuine
  publication or edition date, while **the four Latin American sources publish
  only the current edition, so their vintage date is the local snapshot date**
  and revision history accumulates only across repeated captures on different
  days. See `docs/real_time_data.md`.
- **Modular panel builders** (`build_panel`, `build_subnational_panel`, `build_climate_panel`, `build_financial_panel`) — single
  entry points that materialise quarterly / monthly cross-country,
  US-state, climate/energy, and macro-financial panels from the fetchers,
  with automated frequency rollups (M→Q, A→Q), regime tagging, SA
  (X-13 / STL fallback), and missing-data tracking.
- **Global macro data ecosystem** (`fetch.emissions`, `fetch.energy_transition`, `fetch.commodities`, `fetch.financial`) — keyless,
  cached collectors for World Bank WDI and OECD SDMX greenhouse gas emissions
  by sector; electricity generation by source and clean transition shares;
  expanded World Bank Pink Sheet / IMF commodity benchmark price suites
  (energy, metals, agriculture, fertilizers); and international financial
  stability metrics (sovereign yields, policy rates, BIS credit-to-GDP gaps,
  property prices, and financial conditions). See `docs/data_ecosystem.md`.
- **Instruments** (`instruments.*`) — instrument registry +
  composition + external loaders (FRED API key path); backbone of the
  LP-IV machinery.
- **Bartik / shift-share** (`bartik.*`) — shares, sensitivities,
  Rotemberg weights, county-level EPU exposure, and `shift_share_iv`
  (2SLS with Adão-Kolesár-Morales shock-level standard errors).
- **Spatial econometrics** (`spatial.*`) — spatial weights (contiguity,
  k-NN, distance decay, economic flows), Moran's I / Geary's C, Conley
  spatial HAC and the Hsiang space-time HAC behind
  `panel_lp(cov_type="conley")`, the cross-section models (`sar`, `sem`,
  `sdm`, `slx`) with the `lm_spatial_tests` specification battery and
  LeSage-Pace direct / indirect / total impacts (`spatial_effects`),
  spatial panels (`spatial_panel`) and spatial local projections
  (`spatial_lp`); plus spillover-robust DiD over distance rings
  (`did.spatial_did`) and the Pesaran Global VAR (`var.gvar`); see
  `docs/spatial.md` and `docs/gvar.md`.
- **Misc data utilities** — EU-KLEMS 2023 loader (`klems`), BIS NEER
  aggregator (`bis_neer`), G9 homogeneous-vintage splice
  (`long_panel`), Gollin labor share (`labor_share`), real-time
  vintages (`vintages`), seasonal adjustment (`sa`).
- **Labor flows** — 3-state E/U/N transitions from BLS CPS aggregates
  (`labor_flows`) and 4-state F/I/U/N transitions from ENOE microdata
  for Mexico (`labor_flows_enoe`).

**Running away from a workstation** (`runtime.*`, `runtime.colab`, `pocket.*`, `longrun.*`)

The package's headline promise is that the estimator core runs on an
iPad. These tools make the promise usable rather than merely true:
`runtime` reports what the machine can actually do (sockets? parquet?
threads?) and routes HTTP over the browser when there are no sockets;
`runtime.colab` provides seamless task offloading to Google Colab with
cloud auth and `.pmz` persistence when tasks exceed mobile memory;
`pocket` packs data into portable, self-verifying `.pmz` cartridges so a
panel built online opens offline; `longrun` runs bootstraps and chains in
resumable chunks that survive the OS suspending the app, with results
invariant to how the work was sliced. See "juno.sh / iPad" below.

**DSGE sketchpad, Dynare parity & frontier engines** (`dsge.build`, `dsge.dynare`, `dsge.cli`, `dsge.occbin`, `dsge.bayesian`, `dsge.perfect_foresight`)

Write equilibrium conditions as a Python function or parse native Dynare
`.mod` files (`load_mod`, `parse_mod`). Solves 1st- and 2nd-order approximations
with complex-step differentiation, Kim-Kim-Schaumburg-Sims (2008) pruning,
cross-derivatives ($g_{xu}, g_{uu}$), risk adjustments ($g_{\sigma\sigma}$),
Dynare `oo_.dr` decision rules, and analytical theoretical moments (`stoch_simul`).
Includes the dedicated `puremacro-dynare` CLI tool, Guerrieri-Iacoviello (2015)
OccBin piecewise-linear algorithm for occasionally binding constraints (ZLB),
Boucekkine-Juillard stacked Newton-Raphson relaxation for non-linear perfect
foresight, and full Bayesian MCMC estimation (Laplace Hessian covariance +
adaptive Random-Walk Metropolis-Hastings). No hand-derived matrices, no
Fortran/C++ compiler, 100% Pyodide-ready.

**Teaching artefacts**

`teaching.*` is a research / teaching side-channel that intentionally
wraps `statsmodels` / `linearmodels` / `arch` so notebooks can compare
puremacro's pure-numpy estimators against the canonical packages. **Not
covered by the Pyodide promise.**

## Installation

### From PyPI (users)

```bash
pip install puremacro
```

This pulls the five base dependencies (numpy, scipy, pandas, matplotlib,
requests): everything the estimators and the `fetch` layer need to import.
Every one of them ships with Pyodide, so the same install works in the browser.
Reading parquet or Excel files needs the optional file-format engines:

```bash
pip install "puremacro[io]"      # adds pyarrow (parquet) and openpyxl (.xlsx)
```

The ENOE lessons of the course (08, 24), `build_all`, the shock atlas and the
`fetch.labor*` caches need `[io]`; code that needs a missing engine fails
immediately and names the extra. The other extras are for the optional
features listed below.

### Local (development)

From the `puremacro/` package directory (the one containing this `README.md`
and `pyproject.toml`):

```bash
pip install -e .
```

To run the dev parity tests, install the optional dev deps too:

```bash
pip install -e '.[dev]'
```

To use the `narrative.sources` PDF body extractor:

```bash
pip install -e '.[narrative]'
```

Other optional extras: `[backend]` (numba + Apple-Silicon mlx), `[cuda]`
(NVIDIA cupy), `[io]` (pyarrow + openpyxl: parquet and `.xlsx`), `[data]`
(yfinance / fredapi / xlrd data fetchers, plus the `[io]` engines), `[llm]`
(Anthropic-backed narrative scoring), `[embeddings]` (sentence-transformers
narrative scoring), `[notebooks]` (jupytext notebook build).

For connectors that want opt-in on-disk caching + per-host throttling,
the variants `safe_get_bytes_cached` and `safe_get_text_cached` apply
a SHA-256-keyed cache at `~/.cache/puremacro/http/`. Set
`PUREMACRO_HTTP_NO_CACHE=1` to bypass.

### juno.sh / iPad (unsupported, best-effort)

Upload the `puremacro/` directory to your juno.sh workspace, then in a
notebook cell:

```python
# requires: a Jupyter kernel (IPython magic)
%pip install ./puremacro
```

Every base dependency ships with Pyodide, so that command resolves its
dependencies from the Pyodide distribution without reaching PyPI; so do
`%pip install puremacro` and `await micropip.install("puremacro")`. The parquet
paths (`cache`, `fetch.labor*`, `shock_atlas`, `build_panel`) also need
`pyarrow`: recent Pyodide distributions (314.x, the playground's) include it,
so `%pip install pyarrow` works there, and older ones (0.28) do not. `.xlsx`
readers need `openpyxl`, which is pure Python but installs only where the
kernel may reach PyPI. The browser is **not** a supported deployment target:
teaching material assumes a local install.

#### Finding out what the tablet can actually do

`puremacro.runtime` answers that at run time rather than leaving you to
discover it one traceback at a time:

```python
from puremacro import runtime
print(runtime.report())
#  host       : pyodide 3.12.7 (wasm32)
#  device     : tablet
#  network    : js-fetch (call runtime.enable_browser_network())
#  parquet    : unavailable -> use puremacro.runtime.store / pocket
#  threads    : no (1 cpu, unknown)
#  backends   : numpy
```

Detection is heuristic — no API tells you "this is Juno" — so every field
can be pinned with `PUREMACRO_HOST`, `PUREMACRO_DEVICE`,
`PUREMACRO_SOCKETS`, `PUREMACRO_PARQUET` or `PUREMACRO_THREADS`. The last
one decides whether the bootstrap engines build a thread pool at all: set
`PUREMACRO_THREADS=0` and every resampling engine runs the plain loop, which
is what a browser kernel needs. When threads are unavailable, `n_jobs` is
ignored rather than raising.

#### The three things that break, and what to do about them

**No sockets.** `requests` and `urllib` cannot open a connection under
Pyodide, so every `fetch.*` call fails even though the estimator core
imports perfectly. One call routes the whole existing fetch layer over
the browser's own networking:

```python
from puremacro import runtime
from puremacro.fetch import fetch_xrate_monthly

runtime.enable_browser_network()
fx = fetch_xrate_monthly(["MEX"])
```

Endpoints must send `Access-Control-Allow-Origin` — some public
statistical APIs do, many WAF-fronted government sites do not. A blocked
request says so and names CORS; `proxy=` routes through a CORS proxy you
control.

**No pyarrow.** Pack the data where the network and pyarrow are, open it
where they are not. A cartridge is one self-verifying file carrying its
own provenance:

```python
import numpy as np
import pandas as pd
from puremacro import pocket

# A frame to ship (in practice: the panel you just fetched on the workstation)
panel = pd.DataFrame(
    {"USA": [1.0, 1.2, 1.1], "DEU": [0.8, 0.9, 1.0]},
    index=pd.period_range("2025Q1", periods=3, freq="Q"),
)

# workstation
pocket.pack(panel, "g7.pmz", source="OECD QNA", vintage="2026-08-19")

# iPad, airplane mode
cart = pocket.load("g7.pmz")
panel = cart.frame()          # sha256-checked on read
cart.provenance.vintage       # '2026-08-19'
print(cart.summary())
```

Getting a *file* onto an iPad is often more friction than the analysis,
so a cartridge also travels as text: `pocket.to_base64("g7.pmz")` on
one machine, `pocket.from_base64(blob, "g7.pmz")` on the other.

**The app gets suspended.** iPadOS stops a backgrounded app, and a
four-minute bootstrap does not survive someone answering a message.
`puremacro.longrun` computes in chunks, persists after each, and resumes
in a later session:

```python
import numpy as np
from puremacro import longrun

rng = np.random.default_rng(0)
Y = rng.standard_normal((120, 2))          # your VAR data

def one_draw(i):
    """One bootstrap replication: returns whatever you want the bands of."""
    idx = np.random.default_rng(i).integers(0, len(Y), len(Y))
    return Y[idx].mean(axis=0)

job = longrun.bootstrap(one_draw, 2000, checkpoint="irf.ckpt")
job.run(seconds=30)     # 240/2000 · 12% · ~220s of compute left
job.run(seconds=30)     # ... and again after the app was suspended
bands = np.percentile(job.result(), [5, 95], axis=0)
```

Draw *i* always uses `default_rng([seed, i])`, so a job resumed across
five sessions gives bit-identical results to one that ran straight
through — which is what makes a resumed run publishable.

**Sizing the work to the device.** `runtime.fit(n_boot=2000)` returns
what this machine should actually attempt, and
`runtime.budgeted(estimator)` clamps the cost arguments of a call.
Both are opt-in: no estimator default changed, so a script that runs on
your laptop produces the same numbers it always did. Only cost knobs are
clamped — `horizon` changes what is being estimated, so it is left alone.

**Offloading heavy compute to Google Colab.** When tasks exceed tablet memory
or CPU limits (such as full Bayesian MCMC chains or 10,000-draw wild bootstraps),
`puremacro.runtime.colab` offloads the work seamlessly:

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
result = pm.dsge.estimate_sw07(n_draws=5000, n_chains=2)
""",
    mount_drive=True,
    save_path="offload.ipynb",
    output_filename="sw07_result.pmz",
)

# 2. Show the upload instructions (HTML card in Juno / Jupyter, plain text in a terminal)
show_colab_offload_dialog("offload.ipynb")

# 3. When the cartridge comes back from Google Drive, load it into the local session:
#    posterior = load_colab_result("sw07_result.pmz")
```

### Run the LLM features for free (local models)

The narrative LLM features (`score_llm`, `llm_prob_kernel`) run on a **local
model** — no API key, no paid API, $0. Everything else in puremacro is already
free; this closes the last paid gap.

Install an engine once (any one):

```bash
pip install "puremacro[local-llm]"     # MLX (Apple Silicon) + llama.cpp (any OS)
# or install Ollama (https://ollama.com) — no Python deps — then:  ollama pull qwen2.5:3b
```

Then swap in a local backend (same signatures as the paid backends):

```python
# requires: a local LLM engine (llama-cpp-python or mlx-lm) and a scored `records` corpus
from puremacro.narrative.scoring import score_llm, LocalBackend
events = score_llm(records, backend=LocalBackend("qwen2.5-3b-instruct", engine="auto"))

from puremacro.narrative.indices import llm_prob_kernel, LocalProvider
idx = llm_prob_kernel(records, provider=LocalProvider("qwen2.5-3b-instruct"),
                      category="economic uncertainty")
```

`engine="auto"` picks the best installed engine (Apple GPU via MLX → llama.cpp →
a running Ollama server; for LM Studio / vLLM / any OpenAI-compatible server,
pass `engine="openai"` with `base_url=`). Models: `qwen2.5-3b-instruct` (default),
`gemma2-2b` (Google), `llama3.2-3b` (Meta), `phi3.5` (Microsoft), or any raw
engine model id. See `puremacro/examples/narrative_local_llm.py` and the
`local_llm_uncertainty` notebook. (Local inference is desktop-only — it does not
run inside the browser playground.)

## Pyodide compatibility

The runtime promise is: only `numpy + scipy + pandas + matplotlib`
ever get imported by the *estimator* code that ships in the wheel.
`statsmodels`, `linearmodels`, `arch` and `pypdf` are dev-only /
extras-only or lazy-imported behind a guard.

One further package, `requests`, is declared as a base dependency in
`pyproject.toml` (five in all), because the `fetch` layer and the narrative
sources import it at module level, even though it never touches the
estimator path. It is pure Python and, like the other four, ships with
Pyodide. Every base dependency must: the JupyterLite playground installs
with PyPI fallback disabled, and `tests/test_pyodide_compat.py` enforces it.

The file-format engines are the optional `[io]` extra (also included in
`[data]` and `[dev]`):

- `pyarrow` — the parquet engine `pandas.read_parquet` needs (`cache`,
  `fetch.labor*`, `shock_atlas`, `build_panel`, and the ENOE parquet
  datasets of course lessons 08 and 24). Without it, `cache` falls back to
  the pyarrow-free `.pmz` store.
- `openpyxl` — the `.xlsx` engine `pandas.read_excel` needs. Eighteen
  shipped modules read Excel: the EPU, WUI, JLN, LMN, Fernald, GPR and
  World Bank Pink Sheet fetchers among them.

Both were base dependencies until 3.3.0. They moved because neither ships
with every Pyodide distribution, so either one broke the playground's
install. The risk that made openpyxl a base dependency in the first place
(`build_all` swallowing each failure into a `print` and returning a panel
quietly missing most of its uncertainty proxies) is now handled at the
source: `build_all`, the shock atlas and the climate panel check for the
engines before doing any work and fail with the install command.

See `ARCHITECTURE.md` → "Pyodide-compatibility contract" for the full
rationale.

The regression test is `tests/test_pyodide_compat.py` — it walks every
shippable submodule and asserts no forbidden module lands in
`sys.modules`. If you add a new optional dependency, follow the
existing lazy-import pattern (see `narrative.scoring.llm` or
`fetch._seasonal._x13_arima_analysis` for the canonical examples).

## Quickstart

**First 5 minutes — offline, no data files, no API key.** The quickest check
that your install works (a sign-restricted SVAR on a synthetic 3-variable DGP;
no network, no data, fixed seed):

```bash
python -m puremacro.examples.sign_restrictions_uhlig
```

Or, in Python, on a synthetic system you build in three lines:

```python
import numpy as np
import pandas as pd
import puremacro as pm

# A small synthetic 3-variable system (no data files, no API key).
rng = np.random.default_rng(0)
T = 200
Y = rng.standard_normal((T, 3)).cumsum(0)          # ndarray, shape (T, 3)

# Cholesky-identified SVAR with 90% residual-bootstrap bands.
from puremacro.var.identify import cholesky_svar
res = cholesky_svar(Y, p=2, horizon=20, n_boot=500, ci=0.90)
print(res.summary())
res.plot(target_idx=0, shock_idx=0)        # 1-line IRF plot with confidence bands
print(res.to_latex(target_idx=0, shock_idx=0))  # Camera-ready LaTeX table

# Single-country LP-HAC: response of y to a synthetic shock.
panel = pd.DataFrame({"y": Y[:, 0], "shock": rng.standard_normal(T)})
from puremacro.lp import lp_hac
irf = lp_hac(panel, y="y", x="shock", horizon=20, lags=2, ci=0.90)
print(irf.summary())
irf.plot(title="Response of y to Structural Shock")
print(irf.to_latex())                      # Camera-ready LaTeX table
```

Optional API keys are resolved centrally (none are needed for the synthetic
examples above):

```python
from puremacro import credentials
credentials.status()                  # see what's configured (no values leaked)
# credentials.require("fred")         # raises with a signup URL if the key is missing
```

## Rosetta Stone — Macroeconomist's Cheatsheet

If you are transitioning from Stata, MATLAB/Dynare, or statsmodels:

| Task / Estimator | Stata | MATLAB / Dynare | statsmodels / linearmodels | **`puremacro 2.0`** |
|---|---|---|---|---|
| **Cholesky SVAR** | `var y1 y2, lags(1/4)` + `irf create` | `varm` / VAR Toolbox | `VAR(Y).fit(4).irf(20)` | `var.identify.cholesky_svar(Y, p=4, horizon=20)` |
| **Blanchard–Quah SVAR** | `svar y1 y2, lreq(...)` | VAR Toolbox `bq_svar` | `SVAR(..., svar_type='B')` | `var.identify.bq_svar(Y, p=4, horizon=20)` |
| **Sign Restrictions** | User plugin | Rubio-Ramírez / VAR Toolbox | — | `var.identify.sign_restrictions(Y, restrictions={0: [+1, -1]}, p=4)` |
| **Proxy / External IV SVAR** | `svariv` | Mertens & Ravn SVAR-IV | — | `var.identify.proxy_svar(Y, p=4, instrument_series=z)` |
| **Local Projections (HAC)** | `jorda` / manual OLS | Jordà (2005) code | `OLS(y_h, X).fit(cov_type='HAC')` | `lp.lp_hac(df, y="y", x="shock", horizon=20, lags=4)` |
| **State-Dep LP-IV (Ramey-Zubairy)** | manual 2SLS interaction | — | — | `lp.lp_state_dep_iv(df, y="y", x="g", z="news", state="u")` |
| **Panel LP (Driscoll–Kraay)** | `xtscc` | Panel LP toolbox | `PanelOLS(..., cov_type='driscoll-kraay')` | `lp.panel_lp_dk(df, y="y", x="z", unit_col="id", time_col="t")` |
| **Dynamic Panel GMM** | `xtabond2 y L.y, gmm(y) two robust` | Arellano–Bond MATLAB | — | `dynpanel.ab_gmm(y, panel_id, time_id, two_step=True, windmeijer=True)` |
| **Staggered DiD** | `csdid y, ivar(id) time(t) gvar(g)` | — | — | `did.callaway_santanna(df, unit="id", time="t", outcome="y", treat_time="g")` |
| **Synthetic DiD** | `sdid y id t d` | synthdid R package | — | `did.synthetic_did(df, unit="id", time="t", outcome="y", treat_time="g")` |
| **Factor-Augmented VAR (FAVAR)**| — | BBE (2005) MATLAB | — | `var.favar(panel_df, policy_series, n_factors=3, horizon=20)` |
| **Value Function Iteration** | — | VFIToolkit `ValueFnIter_Case1` | — | `vfi.VFIProblem(a_grid, z_grid, P_z, return_fn, beta).solve()` |
| **Linear DSGE (QZ / BK)** | — | Dynare `stoch_simul` / Klein `solab` | — | `dsge.klein.klein_solve(A, B, C, n_pre=...)` |
| **DSGE from equations / .mod** | — | Dynare `.mod` file | — | `dsge.load_mod("rbc.mod")` / `dsge.build_dynare(eqs)` |
| **DSGE 2nd-Order Pruning** | — | Dynare `stoch_simul(order=2, pruning)` | — | `dsge.build_dynare(eqs, order=2)` / `m.solve_second_order()` |
| **`puremacro-dynare` CLI** | — | `dynare model.mod` command-line | — | `puremacro-dynare model.mod --order 2 --fevd --plot` |
| **OccBin (ZLB / piecewise)** | — | Dynare `occbin_solver` / Guerrieri & Iacoviello | — | `dsge.solve_occbin(m_normal, m_zlb, constraint, shocks)` |
| **Non-Linear Perfect Foresight** | — | Dynare `simul` (Boucekkine-Juillard) | — | `dsge.solve_perfect_foresight(m, shocks, T=100)` |
| **Bayesian DSGE (MCMC)** | — | Dynare `estimation(...)` (Metropolis-Hastings) | — | `dsge.estimate_dsge_bayesian(m, data, priors, n_draws=10000)` |
| **Sequence-Space Fake News** | — | SSJ (Auclert et al. 2021) Python/Julia | — | `models.fake_news_algorithm(T=40)` / `models.simulate_targeted_transfer(...)` |
| **GLS Unit Root (DF-GLS)** | `dfgls y, maxlag(4)` | ERS (1996) code | `adfuller` | `unit_root.dfgls_test(y, regression="ct")` |
| **Seasonal Adjustment** | `x13 y` | X-13 wrapper | `STL` / `x13` | `sa.stl_sa(y)` / `sa.x11_sa(y)` |
| **Continuous Collocation / FEM** | — | VFIToolkit / Miranda-Fackler | — | `vfi.CollocationProblem(domain, orders=7)` / `vfi.FEMProblem(...)` |
| **Deep Macro PINNs (10+ states)** | — | — | — | `vfi.DeepMacroModel.multi_country_growth(...)` / `vfi.solve_deep_macro(...)` |
| **Trade Exact Hat Algebra** | — | Costinot-Rodríguez-Clare code | — | `trade.CaliendoParroModel(...)` / `cp.solve_counterfactual(...)` |
| **Spatial Gravity GE** | — | Allen-Arkolakis replication | — | `spatial.AllenArkolakisModel.from_coordinates(...)` |

End-to-end replications of canonical papers and pedagogical showcases live under `notebooks/`
and `puremacro/examples/`:
- **Causal ML, continuous-time HJB & regional real time**: implicit HJB with the adjoint KFE and Aiyagari GE (`56`), multi-constraint OccBin coupled with DML (`57`), and Latin American real-time macro vintages (`58`).
- **Continuous DP, Deep Macro & Spatial GE**: Chebyshev vs FEM Galerkin (`51`), continuous transition dynamics under MIT shocks (`52`), exact analytic IFT parameter sensitivities & GMM estimation (`53`), 10-state Deep Macro PINNs (`54`), and quantitative spatial trade & infrastructure GE (`55`).
- **Applied Policy Showcases**: Central bank policy stance & fan charts (`47`), real-time DFM nowcasting & news decomposition (`48`), macroprudential GaR & systemic connectedness (`49`), and tri-method fiscal multipliers & sovereign DSA (`50`).
- **Frontier DSGE & HANK**: Exact analytic Kalman score recursion (`43`), HANK sequence-space bridge from `.mod` (`44`), optimal discretionary policy vs commitment & news shocks (`45`), and particle filtering with MS-DSGE (`46`).
- **Canonical Replications**: Smets-Wouters 2007 (`41`, `42`), Bloom 2009 (`bloom2009.py`), Mertens-Ravn narrative SVAR (`svariv_mertens_ravn.py`), Romer-Romer monetary narrative (`romer_romer_*.py`), and ~75 more.
All notebooks strictly adhere to the Pyodide contract and the 7-section pedagogical architecture in `notebooks/_TEMPLATE.md`.

## Documentation

- **`docs/quickstart.md`** — 2-minute quickstart covering core estimators and publication workflows.
- **`docs/vfi_continuous_projection.md`** — Continuous state-space dynamic programming: Chebyshev polynomial collocation and Finite Element Galerkin projection with Fischer-Burmeister borrowing constraints.
- **`docs/vfi_splines_and_sparse_grids.md`** — Shape-preserving cubic B-splines, Schumaker quadratic splines, and Smolyak high-dimensional sparse grids ($d \in [2, 6]$).
- **`docs/dcegm.md`** — Discrete Choice Endogenous Grid Method (DC-EGM) with fast Upper Envelope filtering for non-convex dynamic programming.
- **`docs/vfi_continuous_equilibrium.md`** — Young (2010) continuous stationary wealth distributions, mass conservation, and Aiyagari general equilibrium factor market clearing.
- **`docs/vfi_continuous_transition.md`** — Non-linear continuous transition dynamics under unexpected MIT shocks and sequence-space Broyden solvers.
- **`docs/vfi_analytic_gradients.md`** — Exact analytic gradients and Jacobians via the Implicit Function Theorem (IFT) for high-performance structural GMM/SMM estimation.
- **`docs/deep_macro.md`** — Deep Macro & Physics-Informed Neural Networks (PINNs) in pure NumPy for ultra-high-dimensional dynamic models (10+ continuous states) via ergodic sampling.
- **`docs/spatial_and_trade_ge.md`** — Quantitative spatial economics and international trade general equilibrium: Caliendo-Parro (2015) exact hat algebra and Allen-Arkolakis (2014) economic geography.
- **`docs/data_ecosystem.md`** — Global macro data ecosystem: emissions, energy transitions, commodity benchmark suites, international financial stability, and modular panel builders (`build_climate_panel`, `build_financial_panel`).
- **`docs/notebooks.md`** — Complete showcase catalog (notebooks 00–58), pedagogical 7-section architecture, and applied policy suites.
- **`docs/dsge_build.md`** — DSGE models from equations, native Dynare `.mod` loader, 2nd-order pruning, `puremacro-dynare` CLI, OccBin ZLB, non-linear relaxation, and Bayesian MCMC.
- **`docs/models.md`** — Structural models: Sequence-Space HANK, Fake News algorithm, targeted transfers, and DMP search-and-matching.
- **`docs/policy_simulators.md`** — Macroeconomic policy simulators & interactive browser labs: Caliendo-Parro (2015) quantitative trade general equilibrium, HANK vs. RANK monetary transmission, and client-side WebAssembly laboratories (Spanish twin under `docs/es/`).
- **`docs/narrative_sign_svar.md`**, **`docs/honest_did.md`**, **`docs/smooth_lp.md`**, **`docs/hank_nonlinear.md`**, **`docs/gertler_karadi.md`**, **`docs/bvar_sv.md`** — the six 2.3 feature guides (each with a Spanish twin under `docs/es/`).
- **`docs/spatial.md`** — spatial weights, Moran's I / Geary's C, Conley spatial HAC (cross-section and panel local projections), the cross-section models (`sar`, `sem`, `sdm`, `slx`), the `lm_spatial_tests` specification battery, LeSage-Pace impacts (`spatial_effects`), spatial panels (`spatial_panel`), spatial local projections (`spatial_lp`), spillover-robust DiD (`did.spatial_did`) and shift-share IV with Adão-Kolesár-Morales errors (Spanish twin under `docs/es/`).
- **`docs/gvar.md`** — Global VAR (Pesaran-Schuermann-Weiner): country VARX* blocks, trade-weighted star variables, the stacked solve and GIRFs (`var.gvar`; Spanish twin under `docs/es/`).
- **`docs/var.md`** — Reduced-form VAR, SVAR identification (Cholesky, signs, narrative, proxy/IV), FAVAR, and bootstrap bands.
- **`docs/lp.md`** — Local Projections guide (LP-HAC, LP-IV, State-Dependent LP-IV, Panel LP, `LPResult`).
- **`docs/did.md`** — Modern Difference-in-Differences (Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess, Synthetic DiD).
- **`docs/nowcast.md`** — GDP Nowcasting (Mixed-frequency dynamic factor models, ragged edges, news decomposition).
- **`docs/climate.md`** — Climate macroeconomics: Nordhaus DICE forward simulator and Social Cost of Carbon accounting.
- **`docs/forecast.md`** — Penalized macroeconomic forecasting: Coordinate-descent Elastic Net and Adaptive Lasso.
- **`docs/reporting.md`** — Publication reporting pipeline (LaTeX tabular, Typst tables, Markdown, significance stars).
- **`docs/tablet.md`** — Running on iPad, Juno, and WebAssembly, with Google Colab compute offloading.
- **`docs/benchmarks.md`** — Performance and computational benchmarks across econometric engines.
- **`docs/national_accounts.md`** — OECD Quarterly National Accounts extraction, deflators, and accounting identities.
- **`docs/real_time_data.md`** — Real-time data vintages, revision triangles, and Mankiw-Shapiro news-vs-noise testing.
- **`docs/long_panel.md`** — Historical long national accounts panel (ratio-spliced Spanish and Japanese series).
- **`docs/es/`** — Complete parallel documentation suite in native academic Spanish.
- **`ARCHITECTURE.md`** — module map, stability tiers, Pyodide contract, result-object standard.
- **`CHANGELOG.md`** — per-release diff, including internal-only refactors.
- **`docs/ADVISORY.md`** — correctness advisories: released versions that returned a wrong number.
- **Per-function docstrings** are the canonical reference; the module docstring of each subpackage explains its scope.

## Conventions

- **Public API per subpackage** is curated via `__init__.py::__all__`;
  the top-level `puremacro` package re-exports `__version__`, `build_climate_panel`, and `build_financial_panel`.
- **Frozen-dataclass result objects** for any estimator returning 3+
  fields or non-trivial diagnostics (see `ARCHITECTURE.md` § Result-
  object standard). DataFrames returning named columns are exempt.
- **Diagnostic errors over silent garbage** — singular `X'X`, non-PD
  Σ, BK violations, and ill-conditioned bootstrap draws raise or warn
  with a message naming the calling function and the likely cause.

## Bundled data and attribution

puremacro is MIT-licensed, but since 4.0.0 the wheel also carries the **OECD
Inter-Country Input-Output matrix** so `puremacro.trade` calibrates without
asking you for a file. That data is redistributed under the OECD terms of use,
which permit reuse with attribution.

**If you publish results computed from it, cite the OECD Inter-Country
Input-Output tables**, not puremacro. The OECD is not affiliated with this
project and does not endorse it. The MIT licence covers puremacro's code, not
this third-party data.

The full notice, with each file's provenance and SHA-256, ships as
`puremacro/trade/_datafiles/SOURCES.md`.

## Status

Production release, shipping **3.4.0**. `docs/1.0_path.md` § 5 lists which
subpackages are inside the release-gate promise and which are
research-experimental.

CI is live and runs on every push: the suite across three operating
systems and three Python versions (3.11, 3.12 and 3.13), the Pyodide
contract, mypy, the reference drift-guard, `mkdocs build --strict`, the
playground deploy, and a tag-triggered PyPI publish via trusted
publishing. See `.github/workflows/`. Run
`python tools/release_check.py` locally before tagging anyway — gates 5
and 6 are opt-in and CI does not run them.

When a released version has returned a wrong number, it is recorded in
**[`docs/ADVISORY.md`](docs/ADVISORY.md)**, with the condition under
which the error vanishes so you can rule your own run in or out.
