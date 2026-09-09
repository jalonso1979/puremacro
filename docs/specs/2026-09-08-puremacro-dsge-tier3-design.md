# `puremacro.dsge` Tier 3 — Parity & Surface Area

**Status:** Drafted 2026-09-08. Architectural specification for Tier 3 of the Dynare-parity roadmap (`docs/plans/2026-09-07-puremacro-dsge-dynare-parity-roadmap.md`).  
**Target release:** 2.9.0 (Phases A, B, C, D).  
**Baseline:** puremacro 2.8.0 (AST front end, order-3 perturbation with pruning, MCP semismooth Newton, 4-regime OccBin, Piecewise Kalman Filter, Sequential Monte Carlo, nonlinear Ramsey policy, BGP detrending).  
**Driving lens:** Complete Dynare's operational surface area (`stoch_simul` filtering options, extended path, conditional forecasting, shock decompositions, posterior IRF bands) and establish an automated CI parity test harness comparing puremacro against published Dynare models and `oo_.mat` golden files.

---

## 1. Motivation & Scope

`puremacro 2.8.0` completed the core numerical engines: high-order perturbation, constraints, simulation, and advanced Bayesian inference.

`puremacro 2.9.0` addresses the remaining user-facing operational surface area and automated parity verification:

1. **Filtering & Simulation Surface (`stoch_simul`)**:
   - Macroeconomists routinely compare empirical business cycle moments to Hodrick-Prescott (HP) filtered or bandpass-filtered model moments.
   - Dynare supports `hp_filter = lambda`, `one_sided_hp_filter`, `bandpass_filter = [low, high]`, `ar = n`, `contemporaneous_correlation`, and `simul_replic` (empirical simulated moments with Monte Carlo standard errors).
   - In puremacro, these must be integrated natively into `LinearModel.solve()` / `stoch_simul()` via closed-form spectral integration and empirical simulations.

2. **Extended Path (Fair & Taylor 1983)**:
   - Evaluates non-linear simulations under stochastic shocks without local polynomial expansions.
   - At each date $t$, given state $x_{t-1}$ and shock $u_t$, solves a forward-looking deterministic path over horizon $T_H$ assuming future shocks are zero, extracting $y_t$ and advancing recursively.
   - Exploits the fast sparse stacked Newton engine in `perfect_foresight.py`.

3. **Conditional Forecasting (Waggoner & Zha 1999)**:
   - In policy analysis (e.g. central bank forecast rounds), economists specify paths for certain endogenous variables (e.g., policy rate fixed at zero for 4 quarters, or inflation target path) and determine the required shock sequence and resulting trajectories for other variables.

4. **Shock Decompositions & Diagnostics**:
   - `shock_groups`: Grouping structural shocks into economic categories (e.g., Supply, Demand, Monetary, Fiscal) for aggregated historical shock decompositions.
   - `realtime_shock_decomposition` and `initial_condition_decomposition`: Isolates the decaying impact of initial conditions $x_0$ from cumulative structural shocks.
   - `bayesian_irf`: Computes posterior impulse response bands (median, 68%, 90%, 95%) from MCMC or SMC parameter draws.
   - `prior_predictive`: Samples from the prior distribution to inspect implied model dynamics before observing data.
   - `qz_criterium`: User-configurable eigenvalue cutoff on the unit circle for unit-root and cointegrated models.

5. **The Parity Dashboard**:
   - Automated CI testing harness (`puremacro/dsge/parity.py`) that loads published `.mod` models (including Johannes Pfeifer's `DSGE_mod` collection and Dynare test cases), executes them in puremacro, reads reference `oo_.mat` structures via `load_dynare.py`, and tabulates maximum absolute deviations on decision rules (`ghx`, `ghu`, `ghxx`), moments, and IRFs.

---

## 2. Architecture & Module Map

```
puremacro/dsge/
├── _moments.py            [ext]  HP filter & bandpass spectral integration for theoretical moments
├── load_dynare.py         [ext]  Extended to parse oo_.dr, oo_.mean, oo_.var, oo_.autocorr
├── conditional.py         [new]  Conditional forecasting (Waggoner & Zha 1999) with target paths
├── extended_path.py       [new]  Fair-Taylor (1983) stochastic simulation via stacked Newton
├── shock_groups.py        [new]  Shock group definitions & realtime / initial condition decomposition
├── parity.py              [new]  Automated Dynare parity test harness & scorecard generator
├── bayesian.py            [ext]  bayesian_irf posterior bands & prior_predictive simulation
├── dynare.py              [ext]  Grammar parsing for conditional_forecast, shock_groups, stoch_simul options
├── _results.py            [ext]  ConditionalForecastResult, ExtendedPathResult, ParityDashboardResult
└── cli.py                 [ext]  puremacro-dynare parity command
```

---

## 3. Detailed Specification by Phase

### Phase A: `stoch_simul` Filtering & Simulation Surface

#### A1. Spectral Filtering of Theoretical Moments
For a solved linear model with state transition $x_t = G x_{t-1} + N u_t$ and observation $y_t = F x_{t-1} + L u_t$:
The autocovariance sequence of the filtered series $y_t^{filt}$ is computed via spectral density integration:
$$S_y(\omega) = \frac{1}{2\pi} \left[ F (e^{i\omega} I - G)^{-1} N + L \right] \Sigma_u \left[ F (e^{-i\omega} I - G)^{-1} N + L \right]^H$$
$$S_{y, filt}(\omega) = |H_{filt}(\omega)|^2 S_y(\omega)$$
$$\Gamma_{y, filt}(k) = \int_{-\pi}^\pi S_{y, filt}(\omega) e^{i\omega k} d\omega$$

1. **Hodrick-Prescott Filter (`hp_filter = lambda`)**:
   $$|H_{hp}(\omega)|^2 = \frac{4\lambda (1 - \cos\omega)^2}{1 + 4\lambda (1 - \cos\omega)^2}$$
   Evaluated using Gauss-Legendre quadrature over $[0, \pi]$.
2. **Bandpass Filter (`bandpass_filter = [low, high]`)**:
   $$|H_{bp}(\omega)|^2 = \mathbf{1}_{\{2\pi / high \le \omega \le 2\pi / low\}}$$
   Evaluated by truncating the spectral integration limits to $[\omega_L, \omega_H]$.
3. **Autocorrelations & Correlations**:
   - `ar = n`: Computes autocorrelation matrices $R(k) = D^{-1/2} \Gamma(k) D^{-1/2}$ for $k = 1, \dots, n$ where $D = \text{diag}(\Gamma(0))$.
   - `contemporaneous_correlation`: Computes and formats the $N \times N$ correlation matrix at $k=0$.

#### A2. Empirical Simulation Moments (`simul_replic`)
When `simul_replic = M > 0`:
- Simulates $M$ independent stochastic paths of length `periods`.
- If `hp_filter` or `bandpass_filter` is specified, filters each simulated trajectory directly using `puremacro.data.hp_filter` or `bk_filter`.
- Computes sample mean, standard deviation, variance, skewness, kurtosis, and autocorrelation across replications with Monte Carlo standard errors:
  $$\text{SE}(\bar{\mu}) = \frac{s}{\sqrt{M}}$$

---

### Phase B: Extended Path (Fair & Taylor 1983)

#### B1. Stochastic Simulation via Deterministic Relaxation
Simulates non-linear rational expectations models under stochastic shocks without perturbation:
1. Initialize state $x_0 = y_{init}$ or steady state.
2. For each simulation period $t = 1, \dots, T$:
   a. Draw contemporaneous shock vector $u_t \sim \mathcal{N}(0, \Sigma_u)$.
   b. Set terminal horizon $T_H$ (e.g. 50–100 periods).
   c. Set future expected shocks $u_{t+1}, \dots, u_{t+T_H} = 0$.
   d. Solve the deterministic boundary problem:
      $$f(y_{t+1}, y_t, x_{t-1}, u_t) = 0$$
      $$f(y_{s+1}, y_s, y_{s-1}, 0) = 0, \quad s = t+1, \dots, t+T_H$$
      subject to terminal steady state $y_{t+T_H+1} = y_{ss}$, using the SuperLU sparse stacked Newton solver in `perfect_foresight.py`.
   e. Extract the realized endogenous state $y_t$.
   f. Set $x_t = y_t$ and advance to period $t+1$.
3. Returns `ExtendedPathResult` with trajectory data, convergence metrics, and `.summary()`, `.plot()`, `.to_markdown()`.

---

### Phase C: Advanced Forecasting & Shock Decompositions

#### C1. Conditional Forecasting (Waggoner & Zha 1999)
- **Problem**: Given a DSGE model, produce a forecast over $h = 1, \dots, H$ where a subset of variables $y_{1, t}$ are constrained to match target values $y_{1, t}^*$, driven by a designated subset of structural shocks $u_{1, t}$.
- **Linear State Space Formulation**:
  $$y_{t+h} = F G^{h-1} x_t + \sum_{k=1}^h M_{h-k} u_{t+k}$$
  Stack the linear restrictions into $R U = d$.
- **Shock Selection**:
  - Exact identification (number of target restrictions equals number of controlled shocks): $U = R^{-1} d$.
  - Minimum-entropy / least-squares shock allocation (more shocks than restrictions): $U = R^\top (R R^\top)^{-1} d$.
- Returns `ConditionalForecastResult` containing conditioned paths, unconditioned baseline paths, and structural shock trajectories.

#### C2. Shock Groups & Real-Time Decompositions
- **Grammar**:
  ```dynare
  shock_groups;
    supply = ea, eb;
    demand = eq, eg;
    monetary = em;
  end;
  ```
- **Historical & Real-Time Decomposition**:
  For observed or smoothed state trajectories:
  $$y_t = \sum_{g \in \text{groups}} y_t^{(g)} + y_t^{(init)} + y_t^{(drift)}$$
  where $y_t^{(g)}$ is the simulated response driven exclusively by historical shocks belonging to group $g$, and $y_t^{(init)}$ represents the decaying effect of $x_0$.

#### C3. Bayesian Posterior IRF Bands & Prior Predictive
- `bayesian_irf`: Evaluates model impulse responses across draws $\theta^{(s)}$ from an MCMC chain or SMC particle ensemble. Computes point-wise quantiles (5%, 16%, 50%, 84%, 95%) with publication-grade fan charts.
- `prior_predictive`: Draws $\theta \sim \pi(\theta)$, verifies Blanchard-Kahn determinacy, and computes prior distributions of moments and IRFs.
- `qz_criterium`: Exposes the root classification threshold in `gensys` / `_qz.py` (`1.0 + 1e-8` vs `1.0 - 1e-8`) to support cointegrated systems with unit roots.

---

### Phase D: The Dynare Parity Dashboard (Automated CI Harness)

#### D1. Parity Verification Engine (`puremacro/dsge/parity.py`)
Automated test harness comparing `puremacro` solutions against official Dynare outputs:
1. **Model Loader**: Scans directories of `.mod` files and paired `*_results.mat` files.
2. **Comparison Steps**:
   - Parse check: Does the `.mod` file parse cleanly into `ParsedModelDAG`?
   - Steady-state check: $|y_{ss, puremacro} - y_{ss, dynare}|_\infty \le 10^{-8}$.
   - First-order decision rules:
     $$\max \left( |ghx_{pm} - ghx_{dyn}|_\infty, |ghu_{pm} - ghu_{dyn}|_\infty \right) \le 10^{-10}$$
   - Second-order decision rules (if order $\ge 2$):
     $$\max \left( |ghxx_{pm} - ghxx_{dyn}|_\infty, |ghs2_{pm} - ghs2_{dyn}|_\infty \right) \le 10^{-8}$$
   - Moments & IRFs: $|Mom_{pm} - Mom_{dyn}|_\infty \le 10^{-8}$.
3. **Scorecard Presentation**:
   - `ParityDashboardResult`: Tabulates model name, order, variables, equations, status (PASS/FAIL), max absolute deviation, and runtime.
   - Outputs to Markdown, LaTeX, and terminal table.
   - CLI command: `puremacro-dynare parity [PATH]`.

---

## 4. Presentation Contract & Results Objects

| Class | Module | Methods |
|---|---|---|
| `ExtendedPathResult` | `puremacro.dsge.extended_path` | `.summary()`, `.plot()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |
| `ConditionalForecastResult` | `puremacro.dsge.conditional` | `.summary()`, `.plot()`, `.shock_paths()`, `.to_markdown()`, `.to_latex()` |
| `ShockDecompositionResult` | `puremacro.dsge.shock_groups` | `.summary()`, `.plot()`, `.to_frame()`, `.to_markdown()`, `.to_latex()` |
| `ParityDashboardResult` | `puremacro.dsge.parity` | `.summary()`, `.scorecard()`, `.to_markdown()`, `.to_latex()`, `.to_typst()` |
| `BayesianIRFResult` | `puremacro.dsge.bayesian` | `.summary()`, `.plot()`, `.bands()`, `.to_markdown()`, `.to_latex()` |

---

## 5. Acceptance Criteria & Verification Gates

### Phase A Verification (stoch_simul Options)
- [ ] `hp_filter=1600` theoretical variance matches Gauss-Legendre numerical integration benchmark to $10^{-8}$ relative error.
- [ ] `simul_replic=1000` empirical simulated moments match theoretical ergodic moments within $3\sigma$ Monte Carlo confidence bounds.
- [ ] Autocorrelation table (`ar=4`) and `contemporaneous_correlation` format cleanly with complete variable labels.

### Phase B Verification (Extended Path)
- [ ] Fair-Taylor extended path simulation executes on nonlinear RBC model across 100 periods with 0 solver divergences.
- [ ] Linear model extended path simulation identically matches first-order linear state space simulation to $10^{-10}$.

### Phase C Verification (Conditional Forecast & Shock Groups)
- [ ] Conditional forecast enforces fixed interest rate path for 4 quarters; inverted shock sequence exactly reproduces target path.
- [ ] Realtime shock decomposition balances: $\sum y^{(g)} + y^{(init)} = y^{obs}$ to machine precision ($10^{-12}$).
- [ ] `bayesian_irf` computes 68% and 95% posterior credible intervals from MCMC draws with publication-grade fan charts.

### Phase D Verification (Parity Dashboard)
- [ ] `puremacro.dsge.parity` runs against the canonical Pfeifer benchmark (`sw07_pfeifer.mod`) and achieves:
  - `ghx` max deviation $\le 10^{-12}$.
  - `ghu` max deviation $\le 10^{-12}$.
  - `ghxx` max deviation $\le 10^{-10}$.
- [ ] Parity dashboard produces a clean scorecard table reporting parsed, solved, and matched metrics.

### Release & Code Standards
- [ ] 100% pure Python adhering strictly to the Pyodide four-package contract (`numpy`, `scipy`, `pandas`, `matplotlib` only).
- [ ] All existing regression tests pass with zero failures.
- [ ] `python -m tools.release_check` passes all gates at version `2.9.0`.
- [ ] Bilingual documentation published in `docs/dsge_parity_surface.md` and `docs/es/dsge_parity_surface.md`.
