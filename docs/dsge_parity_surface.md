> 🇬🇧 English · 🇪🇸 [Español](es/dsge_parity_surface.md)

# DSGE Parity Surface, Advanced Simulation & Parity Dashboard

Puremacro 2.9.0 introduces **Tier 3: Parity & Surface Area**, completing the full Dynare operational surface area and automated parity verification under the zero-dependency **Pyodide four-package contract** (`numpy`, `scipy`, `pandas`, `matplotlib`).

This milestone delivers four core macroscopic capabilities:
1. **`stoch_simul` Filtering & Simulation Surface**: Spectral density integration via Gauss-Legendre quadrature for theoretical HP and bandpass filtering, one-sided recursive Kalman HP filter, full cross-variable autocorrelation matrices, contemporaneous correlations, and empirical simulation moments (`simul_replic`).
2. **Extended Path Non-Linear Stochastic Simulation (Fair & Taylor 1983)**: Global dynamic simulation without perturbation Taylor truncations, driven by a sparse SuperLU stacked Newton boundary-value solver with exact linear model invariance.
3. **Advanced Forecasting & Shock Decompositions**: Waggoner & Zha (1999) conditional forecasting via structural shock inversion, `.mod` `shock_groups;` grammar blocks with exact machine-precision adding-up decompositions, Bayesian impulse response function (IRF) fan charts, prior predictive analysis, and tunable `qz_criterium` for unit-root and cointegrated systems.
4. **Dynare Parity Dashboard & CLI**: Native reading of Dynare `oo_.dr` and theoretical moments (`.mat` files), automated deviation scorecards against official Dynare golden outputs, and the `puremacro-dynare parity` command-line benchmark tool.

---

## 1. Overview & Architecture

The puremacro DSGE engine bridges exact analytical solutions, higher-order perturbation with state pruning, and non-linear sequence-space transitions. Tier 3 finalizes the complete operational surface required by central banks, international institutions, and academic researchers to transition workloads directly from legacy Dynare environments into pure Python.

| Module | Core Capability | Key Method / Algorithm | Benchmark Reference |
| :--- | :--- | :--- | :--- |
| `puremacro.dsge._moments` | Spectral filtering & simulation moments | Gauss-Legendre quadrature & forward Kalman filter | Hodrick & Prescott (1997); Stock & Watson (1999) |
| `puremacro.dsge.extended_path` | Extended Path non-linear simulation | Stacked SuperLU Newton engine with zero future shock expectations | Fair & Taylor (1983); Adjemian & Juillard (2014) |
| `puremacro.dsge.conditional` | Conditional forecasting | Structural shock inversion ($U = R^{-1} d$) | Waggoner & Zha (1999) |
| `puremacro.dsge.shock_groups` | Grouped shock decompositions | Exact historical attribution with adding-up balance | Dynare 4.6+ / 5.x / 6.x specification |
| `puremacro.dsge.bayesian` | Bayesian IRFs & Prior Predictive | Posterior fan charts & prior simulation | Geweke (1999); Herbst & Schorfheide (2015) |
| `puremacro.dsge.parity` | Parity Dashboard & Verification | Automated golden comparisons & scorecard generation | Pfeifer (2014); Dynare Benchmark Suite |
| `puremacro.dsge.cli` | Command-line parity runner | `puremacro-dynare parity` CLI | puremacro CLI interface |

---

## 2. `stoch_simul` Filtering & Simulation Surface

### 2.1 Theoretical Spectral Density Filtering

In linear or linearized DSGE models, the state-space representation dates state deviations $x_t$ and observed/target variables $v_t$:

$$x_{t+1} = G x_t + N u_t, \quad v_t = M_x x_t + M_u u_t, \quad u_t \sim \mathcal{N}(0, \Sigma_u)$$

The continuous spectral density matrix of $v_t$ at frequency $\omega \in [-\pi, \pi]$ is:

$$S_v(\omega) = \frac{1}{2\pi} H(e^{-i\omega}) \Sigma_u H(e^{i\omega})^\top$$

where the transfer function is $H(e^{i\omega}) = M_x (e^{i\omega} I_{n_x} - G)^{-1} N + M_u$.

When a linear filter with squared frequency-response gain $|\Phi(\omega)|^2$ is applied, the theoretical filtered autocovariance at lag $k$ is evaluated by integrating over frequencies:

$$\Gamma_k^{filtered} = \int_{-\pi}^\pi |\Phi(\omega)|^2 S_v(\omega) e^{i\omega k} \, d\omega = 2 \int_0^\pi |\Phi(\omega)|^2 \operatorname{Re}\left[ S_v(\omega) e^{i\omega k} \right] \, d\omega$$

Puremacro integrates this spectral density using $n_{quad} = 256$ Gauss-Legendre quadrature nodes and weights mapped to the interval $[0, \pi]$.

#### Supported Spectral Filters

1. **Hodrick-Prescott Filter (`hp_filter = lambda`)**:
   $$|\Phi_{HP}(\omega)|^2 = \frac{4\lambda (1 - \cos\omega)^2}{1 + 4\lambda (1 - \cos\omega)^2}$$
   where $\lambda = 1600$ for quarterly data, $\lambda = 6.25$ for annual data, and $\lambda = 129600$ for monthly data.
2. **Frequency Bandpass Filter (`bandpass_filter = [low, high]`)**:
   Ideal bandpass filter retaining periodicities between periods $p_{high} = 2\pi/\omega_{low}$ and $p_{low} = 2\pi/\omega_{high}$ (e.g. $[6, 32]$ quarters for business cycles):
   $$|\Phi_{BP}(\omega)|^2 = \begin{cases} 1 & \text{if } \omega_{low} \le \omega \le \omega_{high} \\ 0 & \text{otherwise} \end{cases}$$

### 2.2 One-Sided Recursive Kalman HP Filter

While the standard two-sided HP filter relies on future data and cannot be applied in real-time causal contexts, the **one-sided HP filter** (`one_sided_hp_filter`) solves the recursive state-space model:

$$\Delta^2 \tau_t = \eta_t, \quad y_t = \tau_t + \varepsilon_t, \quad \frac{\sigma_\varepsilon^2}{\sigma_\eta^2} = \lambda$$

Writing in state space with state vector $s_t = [\tau_t, \tau_{t-1}]^\top$:

$$s_t = \begin{bmatrix} 2 & -1 \\ 1 & 0 \end{bmatrix} s_{t-1} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} \eta_t, \quad y_t = \begin{bmatrix} 1 & 0 \end{bmatrix} s_t + \varepsilon_t$$

Puremacro estimates the causal trend $\tau_{t|t}$ forward using the discrete Kalman filter with backwards linear extrapolation initialization matching Dynare.

### 2.3 Cross-Variable Autocorrelations & Contemporaneous Correlations

Given theoretical or filtered autocovariances $\Gamma_k$, the contemporaneous correlation matrix $R(0)$ and cross-variable autocorrelation matrices $R(k)$ for lags $k = 1, \dots, n$ are defined as:

$$D = \operatorname{diag}(\Gamma_0), \quad R(0) = D^{-1/2} \Gamma_0 D^{-1/2}, \quad R(k) = D^{-1/2} \Gamma_k D^{-1/2}$$

All diagonal entries of $R(0)$ are strictly 1.0, and values are constrained to the mathematical domain $[-1, 1]$.

### 2.4 Empirical Simulation Moments (`simul_replic = M`)

When `simul_replic = M` is specified in `stoch_simul`:
1. The engine simulates $M$ independent Monte Carlo sample paths of length `periods`.
2. Any requested trajectory filter (e.g., two-sided HP or one-sided HP) is evaluated on each replication path.
3. Sample means, standard deviations, variances, and cross-variable autocorrelations are computed across paths.
4. Monte Carlo standard errors $\sigma / \sqrt{M}$ are reported alongside point estimates.

```python
# requires: standalone snippet
from puremacro.dsge import load_mod

# Solve model and compute theoretical HP moments with autocorrelation matrix
m = load_mod("rbc.mod")
m.solve()

# Theoretical moments filtered with HP(1600) and order 4 autocorrelations
res = m.stoch_simul(hp_filter=1600.0, ar=4, contemporaneous_correlation=True)
print(res.summary())

# Empirical Monte Carlo simulation with M=500 replications
res_mc = m.stoch_simul(periods=200, simul_replic=500, hp_filter=1600.0)
print(res_mc.simulated_moments)
```

---

## 3. Extended Path Non-Linear Stochastic Simulation

### 3.1 The Fair & Taylor (1983) Algorithm

The **Extended Path** algorithm (`extended_path`) simulates non-linear dynamic macroeconomic models under rational expectations without Taylor series truncations or dynamic programming grid discretization.

At each simulation date $t = 0, \dots, T-1$:
1. A structural innovation vector $u_t$ is drawn or supplied.
2. Agents are assumed to expect all future innovations beyond period $t$ to be zero:
   $$\mathbb{E}_t[u_{t+s}] = 0 \quad \text{for } s \ge 1$$
3. The forward deterministic boundary problem is stacked over horizon $T_H$:
   $$f(y_{t+1|t}, y_{t|t}, y_{t-1}, u_t) = 0$$
   $$f(y_{t+s+1|t}, y_{t+s|t}, y_{t+s-1|t}, 0) = 0 \quad \text{for } s = 1, \dots, T_H - 1$$
   with terminal condition $y_{t+T_H|t} = y_{ss}$.
4. The system is solved simultaneously using the sparse SuperLU stacked Newton-Raphson engine from `puremacro.dsge.perfect_foresight`.
5. The contemporaneous realization $y_t = y_{t|t}$ is saved, the historical state is updated ($y_{t-1} \leftarrow y_t$), and the solver moves to date $t+1$ warm-starting from the previous path.

### 3.2 Linear Model Invariance

A fundamental correctness property of the Fair-Taylor algorithm is **linear model invariance**: when the model equations $f$ are linear in $(y_{t+1}, y_t, y_{t-1}, u_t)$, the extended path trajectory is mathematically identical to the first-order linear state space simulation:

$$\| y_t^{EP} - y_t^{Linear} \|_\infty \le 10^{-10}$$

Puremacro rigorously verifies this invariance property in its continuous test suite.

```python
# requires: standalone snippet
from puremacro.dsge.extended_path import extended_path

# Run non-linear stochastic simulation over 150 periods with horizon 80
ep_res = extended_path(
    model_or_equations=m,
    periods=150,
    horizon=80,
    seed=42,
    tol=1e-8,
)

# Access simulated paths and generate publication figures
df_sim = ep_res.to_frame()
fig = ep_res.plot(variables=["c", "k", "y", "r"])
```

---

## 4. Advanced Forecasting & Shock Decompositions

### 4.1 Conditional Forecasting (Waggoner & Zha 1999)

In economic policy analysis, central banks frequently construct forecasts conditioned on hypothetical trajectories of specific target variables (e.g., conditioning on a policy interest rate path remaining fixed for 4 quarters).

Following **Waggoner & Zha (1999, *Journal of Economic Dynamics and Control*)**, the conditional forecast solves for the structural innovation sequence $U = [u_1^\top, \dots, u_H^\top]^\top$ satisfying:

$$R \, U = d$$

where $R$ is the stacked impulse response mapping from structural shocks to target variables, and $d$ is the gap between unconditional baseline forecasts and the specified target trajectory.

#### Solution Methods:
- **Covariance-Weighted (`method="covariance_weighted"`)**: Minimizes the Mahalanobis distance $U^\top \Sigma_U^{-1} U$, weighting shocks by their structural probability:
  $$U = \Sigma_U R^\top \left( R \Sigma_U R^\top \right)^{-1} d$$
- **Minimum Energy (`method="minimum_energy"`)**: Minimizes the Euclidean 2-norm $\|U\|_2$:
  $$U = R^\top \left( R R^\top \right)^{-1} d$$

```python
# requires: standalone snippet
from puremacro.dsge.conditional import conditional_forecast

# Enforce interest rate 'r' to 0.015 for the first 4 quarters
cf_res = conditional_forecast(
    model=m,
    conditions={"r": [0.015, 0.015, 0.015, 0.015]},
    horizon=12,
    controlled_shocks=["eps_r"],
    method="covariance_weighted",
)

# Inspect implied structural shocks and fan chart
print(cf_res.shocks)
fig = cf_res.plot(variables=["y", "pi", "r"])
```

### 4.2 Grouped Historical Shock Decompositions (`shock_groups;`)

In `.mod` files, users can declare shock group categories:

```dynare
shock_groups;
  'Supply' = ea, eb;
  'Demand' = eq, ec;
  'Monetary Policy' = em;
end;
```

`puremacro.dsge.shock_groups` decomposes historical observed series into these economic categories plus initial condition contributions:

$$y_t^{obs} = \sum_{g \in \text{groups}} y_t^{(g)} + y_t^{(init)}$$

This decomposition preserves **exact adding-up balance** to machine precision ($\le 10^{-12}$).

```python
# requires: standalone snippet
from puremacro.dsge.shock_groups import shock_groups_decomposition

decomp = shock_groups_decomposition(
    model=m,
    data=observed_df,
    groups={
        "Supply": ["ea", "eb"],
        "Demand": ["eq", "ec"],
        "Monetary Policy": ["em"],
    }
)
fig = decomp.plot(variable="y")
```

### 4.3 Bayesian IRF Fan Charts & Prior Predictive Simulation

- **`bayesian_irf`**: Evaluates impulse responses across parameter draws from MCMC or SMC posterior chains, computing point-wise credible intervals (median, 68%, 90%, 95%) and generating fan charts (`BayesianIRFResult`).
- **`prior_predictive`**: Samples parameters from prior distributions and computes unconditional moments and IRF distributions before observing empirical data, diagnosing prior model plausibility (`PriorPredictiveResult`).
- **Tunable `qz_criterium`**: In models with unit roots or cointegrating vectors (such as growth models or monetary models without nominal anchors), the generalized Schur eigenvalue classification boundary can be tuned ($\alpha = 1.0 + \varepsilon$) to prevent spurious instability classifications.

---

## 5. Dynare Parity Dashboard & CLI

### 5.1 Verification Engine & Metrics

The parity harness (`puremacro.dsge.parity`) validates puremacro models directly against official Dynare outputs (`.mat` files containing `oo_.dr`, `oo_.mean`, `oo_.var`, `oo_.autocorr`).

The automated test engine checks maximum absolute deviations:

$$\Delta_{max}(A, B) = \max_{i,j} |A_{ij} - B_{ij}|$$

| Metric | Target Array | Tolerance Threshold |
| :--- | :--- | :--- |
| First-order states ($ghx$) | `oo_.dr.ghx` | $\le 10^{-6}$ (typically $\le 10^{-12}$) |
| First-order shocks ($ghu$) | `oo_.dr.ghu` | $\le 10^{-6}$ (typically $\le 10^{-12}$) |
| Second-order states ($ghxx$) | `oo_.dr.ghxx` | $\le 10^{-4}$ (typically $\le 10^{-10}$) |
| Second-order risk ($ghs2$) | `oo_.dr.ghs2` | $\le 10^{-4}$ (typically $\le 10^{-10}$) |
| Ergodic Means | `oo_.mean` | $\le 10^{-5}$ |
| Ergodic Variances | `oo_.var` | $\le 10^{-5}$ |
| Autocorrelations | `oo_.autocorr` | $\le 10^{-5}$ |

### 5.2 Python API

```python
# requires: standalone snippet
from puremacro.dsge.parity import verify_dynare_parity, run_parity_suite

# Single model verification
report = verify_dynare_parity(
    puremacro_model="sw07.mod",
    dynare_output="sw07_results.mat",
    order=2,
)
print(report.to_markdown())

# Suite verification across an entire directory
suite_report = run_parity_suite("tests/fixtures/dynare_benchmarks/")
assert suite_report.passed
```

### 5.3 Command-Line Interface (`puremacro-dynare parity`)

The package registers the standalone CLI command `puremacro-dynare parity`:

```bash
# Verify a single .mod against corresponding .mat results
puremacro-dynare parity models/sw07.mod --order 2

# Verify all models in a directory and export LaTeX scorecard
puremacro-dynare parity benchmarks/ --format latex --outdir reports/

# Strict CI mode with custom tolerance override
puremacro-dynare parity rbc.mod --tol 1e-8 --quiet
```

Output scorecard example:

```
================================================================================
Dynare Parity Scorecard: sw07.mod vs sw07_results.mat
================================================================================
Component        Max Dev        Tolerance     Status
--------------------------------------------------------------------------------
dr.ghx           4.12e-13       1.00e-06      PASSED
dr.ghu           2.88e-13       1.00e-06      PASSED
dr.ghxx          8.45e-11       1.00e-04      PASSED
dr.ghs2          3.20e-11       1.00e-04      PASSED
moments.var      5.14e-07       1.00e-05      PASSED
moments.corr     2.31e-07       1.00e-05      PASSED
--------------------------------------------------------------------------------
Overall Status: PASSED (6/6 checks clean)
================================================================================
```

---

## 6. Pyodide Purity & Zero-Dependency Guarantee

All Tier 3 components adhere strictly to the **Pyodide four-package contract**:
- **NumPy**: Linear algebra, vectorization, array manipulation, eigenvalues, and pseudo-random numbers.
- **SciPy**: Sparse linear systems (`scipy.sparse.linalg.splu`), special functions (`erf`, `normcdf`), and matrix factorizations (`scipy.linalg.ordqz`).
- **Pandas**: Labeled Series and DataFrames for variables, dates, and scorecards.
- **Matplotlib**: Presentation fan charts, shock decomposition bar graphs, and IRF visualizer.

No external solvers (SymPy, CasADi, JAX), parser generators (PLY, Lark, ANTLR), or compiled C/Fortran modules are required. The entire suite runs seamlessly in browser environments via WebAssembly (Pyodide), desktop environments, and high-performance server clusters.
