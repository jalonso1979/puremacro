> 🇬🇧 English · 🇪🇸 [Español](es/dsge_estimation.md)

# Estimating a Dynare `.mod` file

Until 2.6.0 puremacro could *solve* a `.mod` file and could not *estimate* one. Every piece was present — priors that read Dynare's `estimated_params` conventions, a Kalman filter and smoother with exact diffuse initialisation, an audited Metropolis driver — and one connector was missing: nothing turned a `varobs` declaration into a measurement equation. In practice that meant writing seventy lines of index arithmetic per model, which nobody does for their own model.

That connector now exists. A `.mod` file that declares `varobs` and `estimated_params` goes straight to a posterior.

```python
import numpy as np
import pandas as pd

from puremacro.dsge import load_mod

MOD = """
var y a;
varexo eps;
parameters rho;
rho = 0.7;
model;
  y = a;
  a = rho * a(-1) + eps;
end;
initval; a = 0; y = 0; end;
shocks; var eps; stderr 0.5; end;
varobs y;
estimated_params;
  rho, beta_pdf, 0.5, 0.2;
  stderr eps, inv_gamma_pdf, 0.5, 2;
end;
"""

model = load_mod(MOD)

# Data simulated at rho = 0.7, sigma = 0.5, started from the stationary
# distribution rather than from zero (a series that starts away from its own
# stationary distribution biases the estimate down, which is a property of the
# sample, not of the estimator).
rng = np.random.default_rng(1)
burn, T = 200, 400
eps = rng.standard_normal(burn + T) * 0.5
a = np.zeros(burn + T)
for t in range(1, burn + T):
    a[t] = 0.7 * a[t - 1] + eps[t]
data = pd.DataFrame({"y": a[burn:]})

res = model.estimate(data, n_draws=400, n_chains=1, burn_in=500, seed=0)
print(res.summary().round(3))
```

`priors` and `varobs` default to the file's own blocks, so nothing is declared twice. Pass them explicitly to override.

## What the file has to declare

`varobs` names the observed series, in any order; the data frame supplies a column per name. `estimated_params` names what is estimated and how.

```text
estimated_params;
  // NAME, INITVAL, LB, UB, PRIOR_SHAPE, P1, P2, P3, P4, JSCALE;
  alpha, 0.35, beta_pdf, 0.35, 0.02;
  rho, , 0, 1, beta_pdf, 0.5, 0.2;
  stderr eps_a, inv_gamma_pdf, 0.1, 2;
  corr eps_a, eps_b, normal_pdf, 0, 0.2;
end;
```

The grammar is positionally ambiguous — `NAME, 0.35, beta_pdf, …` and `NAME, beta_pdf, …` differ only by a leading field — and `corr a, b` carries a comma inside its target. Both are resolved by consuming any `stderr`/`corr` keyword first, then locating the first `*_pdf` field: what precedes it is read by length, what follows by position. Shape names are matched case-insensitively, because real files write `INV_GAMMA_PDF` while the manual writes `inv_gamma_pdf`.

| Dynare shape | puremacro | notes |
|---|---|---|
| `beta_pdf` | `BetaPrior` | `P3`/`P4` give the generalised beta on `[P3, P4]` |
| `gamma_pdf` | `GammaPrior` | `P3` is the lower bound |
| `normal_pdf` | `NormalPrior` | `P3`/`P4` truncate |
| `inv_gamma_pdf`, `inv_gamma1_pdf` | `InvGammaPrior` | a prior on a standard deviation |
| `inv_gamma2_pdf` | `InvGammaPrior(kind="type2")` | a prior on a variance |
| `uniform_pdf` | `UniformPrior` | `(P1, P2)` as mean/std, or `(P3, P4)` as bounds |
| `weibull_pdf` | `WeibullPrior` | |

In every family `P1` and `P2` are the mean and standard deviation of the **actual** variable, shift and scale included — Dynare's `beta_specification` / `gamma_specification` convention. Reversing that silently re-specifies a whole block.

Estimated objects are named the way Dynare displays them: `SE_<shock>`, `CORR_<s1>_<s2>`, `ME_<obsvar>`.

```python
ep = model._estimated_params
print([s.name for s in ep.specs])
print({s.name: s.kind for s in ep.specs})
```

That `kind` is not cosmetic. Only `"param"` enters the model's equations, so only `"param"` forces a re-solve; a shock standard deviation, a shock correlation and a measurement-error standard deviation are written straight into `Q` and `H`.

## The measurement equation

A solved model reports `v_t = ys + C x_t + D u_t`, so an observable loads the **contemporaneous** innovation, while a standard state space assumes measurement noise independent of the state shock. There is no exact way to write `D u_t` as measurement noise, so the innovation is carried in the state:

$$\alpha_t = \begin{bmatrix} x_t \\ u_t \end{bmatrix},\quad
T = \begin{bmatrix} A & B \\ 0 & 0\end{bmatrix},\quad
R = \begin{bmatrix} 0 \\ I\end{bmatrix},\quad
Z = \begin{bmatrix} C_O & D_O\end{bmatrix},\quad d = ys_O$$

Two things fall out of that choice. Growth-rate observables need no special handling — `dy = y - y(-1) + ctrend` is a declared variable, so it is a row of `(C, D)` like any other. And the smoothed structural shocks are literally the last rows of the smoothed state, so no disturbance smoother is involved.

```python
from puremacro.dsge import make_state_space_from_varobs

ssm = make_state_space_from_varobs(model, ["y"])
print(ssm.T.shape, ssm.R.shape, ssm.Z.shape, ssm.d)
```

Measurement error is **zero** by default, as in Dynare. Pass `measurement_error={"y": 0.01}` for a standard deviation, or `ridge=` for pure conditioning. If the observables cannot be generated by the shocks the model has, that is reported as the specification error it is before any sampling starts, not as a bad starting point.

## Smoothing and forecasting at calibrated parameters

`smoother` is Dynare's `calib_smoother`.

```python
sm = model.smoother(data)
print(sm.shocks.head(3).round(4))
print(f"log-likelihood {sm.loglik:.3f}")
print(sm.summary())
```

`sm.states`, `sm.shocks`, `sm.smoothed_obs` and `sm.filtered_states` are data frames indexed like the input. `sm.shock_decomposition()` gives the historical decomposition for the same model and data.

```python
fc = model.forecast(horizon=12, data=data, ci=0.90)
print(fc.mean.head(3).round(4))
```

The band reflects shock uncertainty only — the parameters are held fixed at the values the model was solved with.

One caveat on the smoother: **the first smoothed period is not reproducible to machine precision across platforms**. Carrying the innovation in the state makes the predicted covariance structurally rank-deficient (the augmented state has `n_states + n_shocks` dimensions driven by `n_shocks` innovations), and the RTS gain is built from `numpy.linalg.pinv` of it. On a small RBC that matrix has condition number ~1e15 with its smallest singular value sitting at pinv's default cutoff, so the rank decision differs between LAPACK builds. The effect is confined to `t = 0`: from `t = 1` onward the fit reproduces the data to machine precision everywhere, while the first period can differ by ~3e-06 on an observable of magnitude 2. Read `shocks.iloc[0]` with that in mind.

## Checking the mode

`estimate_dsge` used to do one bounded L-BFGS-B run, and a 2.5.0 audit caught it returning the starting values while reporting convergence. `mode_compute` now selects among `lbfgs` (the default), `simplex`, `csminwel`, `cmaes` and `none`.

The cheapest way to see that a reported mode is not one is to look at it along each parameter:

```python
from puremacro.dsge import mode_check
from puremacro.dsge.priors import log_prior

names = tuple(res.param_names)
priors = model._estimated_params.priors()

def neg_log_post(v):
    """The posterior, built from public pieces: re-solve, filter, add the prior."""
    theta = dict(zip(names, map(float, v)))
    m = load_mod(MOD, params={"rho": theta["rho"]})
    ll = m.smoother(data, shock_cov=np.array([[theta["SE_eps"] ** 2]])).loglik
    return -(ll + log_prior(theta, priors))

mc = mode_check(
    neg_log_post,
    np.array([res.mode[n] for n in names]), names,
    n_points=11, cov=res.mode_hessian_inv,
)
print(mc.summary())
```

A slice that improves away from the reported mode means the point is not a mode in that direction. `mc.summary()` names the offenders; passing is necessary, not sufficient, since these are one-parameter slices and say nothing about directions in between.

## Marginal likelihood and model comparison

```python
print(f"Laplace  log p(y) = {res.log_mdd('laplace'):.3f}")
```

Two estimators, and they are not equivalent. **Laplace** is exact when the posterior is Gaussian and only as good as the mode otherwise — check the mode first. **Geweke's modified harmonic mean** reweights the draws by a truncated normal and is evaluated at every truncation level, with the spread across levels returned rather than hidden: the estimator is supposed to be invariant to the truncation, so a swing past one log point means it has not converged, and it says so.

`model_comparison` scores every model by the *same* estimator and refuses when one cannot supply it. A Laplace value and a harmonic-mean value are not on the same footing, and mixing them silently is how a Bayes factor becomes fiction.

## Model Diagnostics & Residual Analysis

Before estimating a model or trusting policy simulations, puremacro provides three diagnostic entry points matching and extending Dynare's diagnostic toolkit: `check()`, `resid()`, and `model_diagnostics()`.

### Blanchard-Kahn Determinacy & Eigenvalues (`check()`)

`model.check()` evaluates the generalized eigenvalue spectrum of the companion pencil, classifies roots into stable, unit-root, and explosive categories, and verifies the Blanchard-Kahn order and rank conditions:

```python
table = model.check()
print(table.summary())
```

When determinacy fails (indeterminacy or explosiveness), `check()` goes beyond a scalar count: it computes the generalized eigenvectors associated with the offending roots and isolates the top variable loadings, naming which economic variables drive the failure:

```text
EIGENVALUES & BLANCHARD-KAHN DIAGNOSTICS (qz_criterium=1.000001)
========================================================================
Status       : 1 explosive roots for 2 forward-looking variables; missing root at |lambda| = 0.9421 loads chiefly on pi, y (indeterminacy).
Determinacy  : DETERMINACY FAILED
Forward vars : 2
Explosive    : 1
Stable       : 3
Unit roots   : 0
```

Visualizing the spectrum against the complex unit circle:

```python
fig = table.plot()  # plots stable, unit, and explosive roots with unit circle
```

### Steady-State Residuals (`resid()`)

`model.resid()` evaluates the dynamic equilibrium equations at steady state, $f(ss, ss, ss, 0)$, and returns a `pd.Series` of residuals sorted descending by absolute magnitude:

```python
resids = model.resid()
print(resids.head(5).round(6))
```

Equation labels and tags from the `.mod` file are preserved, allowing immediate isolation of equations with non-zero steady-state residuals (e.g. diagnosing calibration discrepancies).

### Structural Model Diagnostics (`model_diagnostics()`)

`model.model_diagnostics()` performs comprehensive static and dynamic structural checks:
- **Static Jacobian rank analysis**: evaluates $\text{rank}(J_{\text{static}})$ against the number of variables.
- **Collinear equations & variables**: uses Singular Value Decomposition (SVD) to pinpoint exact collinear equation subsets and unconstrained variables.
- **Numeric incidence matrix**: computes the union incidence across neighbourhood perturbations to detect unused variables and redundant equations.
- **Dynamic matrix pencil regularity**: verifies that $\det(B - \lambda A) \not\equiv 0$.
- **Stochastic singularity**: flags if the number of declared `varobs` exceeds the number of shocks plus measurement errors.

```python
diag = model.model_diagnostics()
print(diag.summary())

# Plot the boolean incidence matrix
fig = diag.plot()
```

## Robust Block-Triangular Steady-State Solving

Computing deterministic steady states in medium- and large-scale DSGE models frequently fails under naive multidimensional Newton solvers. Puremacro implements a block-triangular decomposition engine (`puremacro.dsge.steady`):

1. **Hopcroft-Karp Bipartite Matching**: finds a maximum cardinality matching between equations and variables in $O(|E|\sqrt{|V|})$ time.
2. **Dulmage-Mendelsohn Singularity Decomposition**: when a complete matching does not exist, decomposes the incidence bipartite graph into overdetermined, underdetermined, and well-determined subsets, raising an informative `StructuralSingularityError` that explicitly names the offending equation and variable subsets:
   ```python
   # requires: standalone snippet
   from puremacro.dsge.steady import StructuralSingularityError

   try:
       ss, info = steady(equations, variables, guess, params)
   except StructuralSingularityError as err:
       print("Overdetermined equations :", err.overdetermined_equations)
       print("Underdetermined variables:", err.underdetermined_variables)
   ```
3. **Tarjan Strongly Connected Components (SCC)**: decomposes the matching-directed dependency graph into topological sub-blocks solved in sequence. Singletons are solved with 1D scalar root finders (Brent / secant), while coupled sub-blocks use multidimensional solvers.

### Solver Menu & Homotopy Continuation

The `solve_algo` parameter selects among numerical algorithms:
- `"block"` (default): block-triangular recursive solver.
- `"hybr"`: MINPACK Powell hybrid method.
- `"lm"`: Levenberg-Marquardt least-squares solver.
- `"df-sane"`: derivative-free spectral method for large-scale systems.

When non-linear solvers fail from a distant initial guess, **adaptive homotopy continuation** traces a parameter path with automatic step bisection:

```python
# requires: standalone snippet
from puremacro.dsge.steady import steady

# Continuation from alpha=0.20 to alpha=0.36
ss, info = steady(
    equations,
    variables,
    guess,
    params={"alpha": 0.36, "beta": 0.99},
    solve_algo="block",
    homotopy={"alpha": (0.20, 0.36)},
    homotopy_steps=10,
)
```

## Parameter Identification Analysis

Following Iskrev (2010) and Ratto (2011), `model.identification()` assesses whether structural parameters can be uniquely recovered from the declared observables `varobs`:

```python
# requires: standalone snippet
ident = model.identification(varobs=["y", "pi", "i"], lags=2)
print(ident.summary())
```

The analysis evaluates two fundamental Jacobians:
1. **$J_1$ (State-space solution Jacobian)**: $\frac{\partial \text{vec}(T, R, Q, Z, H)}{\partial \theta}$.
2. **$J_2$ (Theoretical moment Jacobian)**: $\frac{\partial m(\theta)}{\partial \theta}$, where $m(\theta)$ stacks model autocovariances up to `lags`.

The returned `IdentificationResult` reports:
- **Rank deficiency**: whether $J_1$ and $J_2$ are full rank, and the dimension of their null spaces.
- **Null-space parameter combinations**: exact linear combinations spanning the null space (e.g. `0.7071 * theta1 - 0.7071 * theta2 = 0`), revealing redundant or proportional parameters.
- **Multi-way collinearity ($R^2$)**: $R^2$ from regressing each Jacobian column on all others; values near $1.0$ flag collinear parameters.
- **Identification strength**: Ratto's sensitivity and normalized strength measures.

```python
# requires: standalone snippet
# Visualize collinearity R^2 vs identification strength
fig = ident.plot()
```

### Pre-Flight Identification Check in Estimation

To prevent launching expensive MCMC chains on unidentifiable models, pass `check_identification=True` to `model.estimate()`:

```python
# requires: standalone snippet
# Issues an informative warning if parameters are rank-deficient
res = model.estimate(data, check_identification=True)

# Or pass check_identification="raise" to abort immediately on rank deficiency
# res = model.estimate(data, check_identification="raise")
```

## Optimal Policy Regimes & Optimal Simple Rules

Puremacro supports optimal monetary and macroprudential policy design across three standard regimes:

### Optimal Simple Rules (`osr()`)

`model.osr()` optimizes feedback coefficients in simple policy rules (such as Taylor rules) to minimize a quadratic target variance loss:
$$L(\gamma) = \sum_i w_i \text{Var}(y_i; \gamma)$$
subject to Blanchard-Kahn determinacy. A continuous penalty surface guarantees smooth gradient evaluation when parameter candidates venture into indeterminacy:

```python
# requires: standalone snippet
osr_res = model.osr(
    rule_params=["phi_pi", "phi_y"],
    target_vars=["pi", "y"],
    weights={"pi": 1.0, "y": 0.5},
)
print(osr_res.summary())

# Grouped bar chart comparing variances under baseline vs optimal rule
fig = osr_res.plot()
```

### Discretionary Policy (`discretionary_policy()`)

`discretionary_policy()` computes the Markov-perfect time-consistent discretionary equilibrium (Dennis 2007) by iterating on private-sector and central-bank feedback matrices:

```python
# requires: standalone snippet
from puremacro.dsge import discretionary_policy

disc_res = discretionary_policy(
    model,
    target_vars=["pi", "y"],
    weights={"pi": 1.0, "y": 0.25},
    instruments=["i"],
    beta=0.99,
)
print(disc_res.summary())
```

### Linear-Quadratic Commitment (`lq_commitment()`)

`lq_commitment()` solves optimal policy under commitment from the timeless perspective ($\lambda_{-1} = 0$). It forms the Lagrangian over the rational-expectations equilibrium conditions, augmenting the state vector with forward-looking Lagrange multipliers:

```python
# requires: standalone snippet
from puremacro.dsge import lq_commitment

commit_res = lq_commitment(
    model,
    target_vars=["pi", "y"],
    weights={"pi": 1.0, "y": 0.25},
    instruments=["i"],
    beta=0.99,
)
print(commit_res.summary())

# Access policy multipliers and plot impulse responses
augmented_model = commit_res.linear_model
print("Multipliers:", commit_res.multipliers)
fig = commit_res.plot(periods=16)
```

## Standard Presentation Contract

All DSGE result objects (`EigenvalueTable`, `ModelDiagnosticsResult`, `IdentificationResult`, `OSRResult`, `PolicyResult`) adhere to puremacro's unified presentation contract:

| Method | Return Type | Description |
|---|---|---|
| `.to_frame()` | `pandas.DataFrame` | Canonical tabular representation of results |
| `.summary()` | `str` | Clean, human-readable terminal output |
| `.plot()` | `matplotlib` Figure/Axes | Publication-ready visual diagnostic |
| `.to_markdown()` | `str` | GitHub-flavored Markdown table |
| `.to_latex()` | `str` | Publication-ready LaTeX `tabular` with escaped specials |
| `.to_typst()` | `str` | Typst `#table(...)` string for modern scientific typesetting |

## What this deliberately does not do

- **No macro processor.** `@#define`, `@#for`, `@#if`, `@#include` and `@{...}` raise `DynareFeatureError`. They used to be ignored, which meant a file loaded clean and a *different model* was solved. Expand them with `dynare model.mod savemacro` and pass the expanded file. Planned for 2.7.0.
- **No expression parser**, so `STEADY_STATE()`, `EXPECTATION()`, `normcdf`, and a model-local `#` variable defined over an endogenous variable all raise. Also 2.7.0.
- **First order only.** A second-order solution is refused rather than silently linearised; a particle filter for it is a later release.
- **The sampler adapts a scalar, not a covariance** — and its adaptation fires only every 100 iterations, so a `burn_in` below 100 never adapts at all and can leave the chain stuck with a 0% acceptance rate. Give it at least a few hundred.
- **`mode_compute` defaults to `"lbfgs"`**, not to the better `csminwel`, because changing it changes every posterior previously produced.
- **`observation_trends` is applied by detrending the data**, since the state space is time-invariant by construction.
- **Marginal likelihoods from before 2.5.0 are not comparable** to these or to each other: the Kalman recursion then started from a diffuse `P0`, worth about 114 log points on Smets–Wouters.

## Showcase notebooks

For complete, interactive tutorials with visualization and diagnostics:

- **`notebooks/42_dsge_bayesian_estimation_and_diagnostics.py`** (and Spanish edition `42_dsge_bayesian_estimation_and_diagnostics_es.py`): Flagship Bayesian estimation showcase reproducing the full Smets-Wouters (2007) workflow. Demonstrates multi-algorithm mode search comparing `"lbfgs"`, `"csminwel"`, and `"cmaes"`; visual mode diagnostics via `mode_check` curvature slices; Metropolis-Hastings MCMC sampling with Gelman-Rubin convergence metrics; Kalman smoother state/shock extraction with historical structural shock decomposition; out-of-sample and conditional forecasting with fan charts; and marginal data density computation (Laplace approximation vs Geweke modified harmonic mean) with Bayesian model comparison.
- **`notebooks/41_dynare_frontier_showcase.py`** (and Spanish edition `41_dynare_frontier_showcase_es.py`): Demonstrates Dynare frontier capabilities using native 2.6.0 `.smoother()` and `.estimate()` on `sw07_pfeifer.mod` with bundled quarterly US data (`_sw07_data.csv`), OccBin occasionally binding ZLB constraints, and perfect-foresight Ramsey transitions.

## Replication suite: `dsge_estimation` family

The `puremacro.replication` module provides automated verification of published empirical headline results from academic papers. The `dsge_estimation` family verifies:

- **`dsge_estimation.sw07_log_posterior_at_mode`**: Evaluates the exact Kalman log-posterior at the posterior mode on the 1966–2004 US dataset (`_sw07_data.csv`) under unconditional Lyapunov stationary covariance initialization (`_stationary_init`). Target: `-1673.72` (`Tol.TIGHT`).
- **`dsge_estimation.sw07_laplace_marginal_data_density`**: Evaluates the Laplace approximation to the marginal data density from the mode inverse Hessian. Target: `-1686.09` (`Tol.TIGHT`).
- **`dsge_estimation.sw07_harmonic_mean_mdd_consistency`**: Evaluates Geweke (1999) modified harmonic mean MDD across truncation parameters `[0.1, 0.3, 0.5, 0.7, 0.9]` and verifies consistency across truncation levels (spread $< 2.5$ log points).
- **`dsge_estimation.sw07_structural_parameters_mode`**: Verifies headline structural parameters from Smets & Wouters (2007, Table 1) at the posterior mode (`csadjcost`, `csigma`, `chabb`, `csigl`, `cprobp`, `cfc`, `crr`, `crdy`, `ctrend`).
