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
2. **Dulmage-Mendelsohn Singularity Decomposition**: decomposes the incidence bipartite graph into over-determined, under-determined and well-determined subsets. When a complete matching does not exist at the guess, `steady()` runs one full-system `hybr` solve and checks the matching again at the point it reaches. If the matching is complete there, the gap was local to the guess and the root is returned (`info["fallback_reason"] == "incomplete_matching_at_guess"`). Otherwise the model is structurally singular and its steady state is not locally unique:

   - If every over-determined equation reads `0 = 0` (the static form of a unit-root law of motion such as `z = z(-1) + e`), the point is returned with a `StructuralSingularityWarning` naming the free variables and `info["structurally_singular"] = True`, as Dynare does for unit-root models.
   - Any other structure (structurally duplicated equations, i.e. a subset of equations involving fewer variables than equations, or a variable that enters no equation) raises `StructuralSingularityError`. The error names the over- and under-determined subsets and carries the checked root as `hybr_point`. To accept the point instead, pass `allow_singular=True` to `steady()`, `build()`, `build_dynare()` or `load_mod()`, or wrap the call in `with allow_structural_singularity():` after `from puremacro.dsge.steady import allow_structural_singularity`. The context manager's setting applies to the current thread (or asyncio task) only.
   - `allow_singular=False` makes unit roots raise too.
   - An inconsistent system raises in every mode.

   The check is structural, like Dynare's `dmperm`: numerically collinear equations with a complete incidence (e.g. `x + y = 2` and `2x + 2y = 4`) are not detected. When the block pass leaves a residual above `tol`, a full-system `hybr` polish runs and is recorded in `info["full_system_polish"]`, as well as in an INFO log record. A model built by `build()`, `build_dynare()` or `load_mod()` keeps that `info` dict as `model.steady_state_info` (`None` when the steady state was supplied rather than solved for).

   ```python
   # requires: standalone snippet
   from puremacro.dsge.steady import StructuralSingularityError, steady

   try:
       ss, info = steady(equations, variables, guess, params)
   except StructuralSingularityError as err:
       print("Overdetermined equations :", err.overdetermined_equations)
       print("Underdetermined variables:", err.underdetermined_variables)
       print("Checked root             :", err.hybr_point)
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

### Where the Jacobians are evaluated

Local identification is a property of the Jacobian at one parameter vector $\theta_0$: the rank condition is $\operatorname{rank} J(\theta_0) = n_\theta$. Every column of $J_1$, $J_2$, $J_H$ and $J_S$ is taken at the same $\theta_0$. The model is re-solved once at $\theta_0$, and column $j$ perturbs $\theta_j$ alone, with every other parameter held at its $\theta_0$ value. $\theta_0$ is the model's calibration and declared shock covariance, overridden by `fixed_params`, and then by the value of each analysed parameter:

| `params` | Value of each analysed parameter |
|---|---|
| a mapping `{name: value}` (or `p_dict=`) | the value given |
| a list of names | its current value: the calibration for a structural parameter, $\sqrt{\Sigma_{u,ii}}$ for `SE_<shock>` (1.0 when no covariance is declared), the declared correlation for `CORR_<s1>_<s2>` (0.0 when none), the `measurement_error` entry for `ME_<obs>` (0.0 when none) |
| `EstimatedParams` / `EstimatedParamSpec`s | the spec's `start`: its `INITVAL` when declared, else its prior mean |
| omitted, on a `.mod` model with `estimated_params` | each declared parameter at its `start`, as in the row above |
| omitted, on any other model | every calibrated parameter at its calibration |

A list of names that the `estimated_params` block declares also takes each spec's `start`.

`fixed_params` (structural values held at $\theta_0$ but not analysed) and `measurement_error` (fixed measurement-error standard deviations by observable) are keywords of both the method `model.identification()` and the function `puremacro.dsge.identification(model, ...)`.

**How this differs from Dynare.** Dynare's `identification` command defaults to `parameter_set = prior_mean`. puremacro's default matches that only when the block declares no `INITVAL`. To pick the point explicitly, pass it:

```python
# requires: standalone snippet
ep = model._estimated_params
at_prior_mean = model.identification(params={s.name: s.prior.mean for s in ep.specs})
at_calibration = model.identification(params={n: model._params[n] for n in ["kappa", "rho_u"]})
```

`SE_` values are innovation standard deviations ($Q_{ii} = \sigma_i^2$). `CORR_` values are correlations ($Q_{ij} = \rho_{ij}\sigma_i\sigma_j$, using the standard deviations at $\theta_0$). `ME_` values are measurement-error standard deviations ($H_{ii} = \sigma^2$). This is the same mapping `estimate()` applies to a draw. A declared off-diagonal covariance that no `CORR_` parameter names stays fixed *as a covariance* when an `SE_` moves. Dynare instead keeps a declared correlation fixed as a correlation.

Structural columns are central differences with step $h = \max(10^{-5}, 10^{-4}|\theta_j|)$. The exception is when $\theta_j \pm h$ crosses a declared bound or a solve fails. The column is then a second-order one-sided difference taken on a side whose points stay inside the bounds, away from the active bound. A step never crosses a declared bound. Shock and measurement-error columns are analytic.

**State set of a `.mod` / `build_dynare` model.** `build_dynare` finds the predetermined variables numerically: a variable is a state when its lag has a non-zero coefficient at the calibration. A lag whose coefficient is calibrated to 0, such as SW07's `crhoms`, `crhopinf`, `crhow`, `cmap` and `cmaw`, is therefore not a state of the calibrated model. Every solve in one identification analysis uses the same state set: the model's states, plus the lags that are active at $\theta_0$, plus the lags that any analysed structural parameter switches on when it moves. When that set differs from the model's own, the model is re-solved with it, even at the calibration. The observables' moments and spectra are unchanged by the extra states; their derivatives are not. Models from `build()` keep the states you declared.

`prior_mc=N` repeats the rank analysis at `N` draws of $\theta_0$ from the priors, truncated to the declared bounds. Each draw is analysed at its own full point. Draws the model cannot be solved at are skipped and counted in `prior_mc_results["n_failed"]`; `n_draws` counts the draws that were analysed. If no draw can be solved, the rates are NaN and a `RuntimeWarning` is raised.

### Pre-Flight Identification Check in Estimation

To prevent launching expensive MCMC chains on unidentifiable models, pass `check_identification=True` to `model.estimate()`. The check runs at the point the estimation starts from: each estimated parameter at its `start` (its `INITVAL`, else its prior mean), `fixed_params` and `measurement_error` as passed, and the calibration for everything else. A `fixed_params` name that is not a model parameter makes `estimate()` raise `ValueError`, with or without the check.

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
subject to Blanchard-Kahn determinacy. At a candidate rule that is indeterminate, or at which the model cannot be solved, the loss is replaced by a continuous penalty, $10^8 + 10^4\lVert\gamma - \gamma_0\rVert^2$, so the gradient-free optimizers (Nelder–Mead, Powell) contract away from that region:

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

**Tolerances and accuracy.** `model.osr()` and the function `puremacro.dsge.osr(model, ...)` take `xatol` (default `1e-8`, an absolute tolerance on the coefficients, in their own units), `fatol` (default $10^{-12}$ times the initial loss, at least $10^{-12}$) and `options` (passed last to `scipy.optimize.minimize`, e.g. `{"xtol": ..., "ftol": ...}` for Powell). A search that compares loss values locates the coefficients, and the allocation they imply, only to a relative error of about $\sqrt{2\varepsilon/c}$, where $\varepsilon$ is the relative precision of the loss and $c$ its normalised curvature at the optimum. That is roughly $10^{-8}$ for a well-scaled problem and worse for a flat or badly scaled loss. Because `xatol` is absolute, scale it with the coefficients. `xatol=1e-4, fatol=1e-4` reproduces SciPy's own Nelder–Mead defaults, which `osr` used up to and including 4.3.0. Measured examples are in [DSGE Frontier, §1.6](dsge_phase_c.md).

`loss_opt` is recomputed by re-solving the model at the returned coefficients. If that re-solve fails, or the rule is indeterminate there, `loss_opt` is NaN with a `RuntimeWarning` and `optimal_model` is `None`; `loss_initial` is NaN, also with a warning, when the baseline moments cannot be evaluated. Neither is ever a placeholder or the optimizer's penalty value.

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

`lq_commitment()` solves optimal policy under commitment. It forms the Lagrangian over the rational-expectations equilibrium conditions and augments the state vector with the Lagrange multipliers. The law of motion it returns is the same for the Ramsey plan chosen at $t_0$ and for the timeless-perspective rule; the two differ only in the initial multiplier. The Ramsey plan sets $\lambda_{-1} = 0$ whatever the history, while the timeless perspective uses the multiplier implied by past policy. Impulse responses and `conditional_loss` start from the steady state, with $\lambda_{-1} = 0$, where the two coincide. `loss` averages over the stationary distribution, so it evaluates the timeless rule on average, and it is NaN with a `RuntimeWarning` when no stationary distribution exists. The `timeless` argument never changed the result and is deprecated. See [DSGE Frontier, §1.3](dsge_phase_c.md).

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

- **Macro directives are expanded, never ignored.** Since 2.7.0, `@#define`, `@#for`, `@#if`, `@#include` and `@{...}` are expanded by puremacro's own preprocessor before the file is parsed ([DSGE sketchpad, §4b](dsge_build.md)). In 2.6.0 they raised `DynareFeatureError`; before that they were ignored, which meant a file loaded clean and a *different model* was solved.
- **Parameters are re-read only where Dynare re-reads them.** The `.mod` reader parses `STEADY_STATE()`, `normcdf` and model-local `#` variables, over parameters or over endogenous variables. Locals are substituted symbolically, never frozen at the calibration, so an estimated parameter that enters the model only through them (SW07's `constebeta`, via `cbeta`, `cbetabar`, `cr`, `conster`, ...) moves the likelihood at every draw; up to and including 4.3.0 such locals were folded to numbers at load time and `constebeta`'s likelihood was exactly flat. Top-level assignments such as `cbeta = 1/(1+constebeta/100);` outside the model block are evaluated once when the file is read and do **not** follow the draws — the same as in Dynare — so write a parameter-derived quantity as a `#` local. `EXPECTATION(k)(...)` is parsed but its compiled residual raises `NameError`.
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

The `puremacro.replication` module checks published headline results against puremacro's own computations. The `dsge_estimation` family runs the hand-coded Smets & Wouters (2007) model (`puremacro.dsge.smets_wouters`) on the bundled 1966Q1–2004Q4 US data (`_sw07_data.csv`, rebuilt from FRED with the SW07 data-appendix definitions). Its fixture, `puremacro/replication/data/sw07_parity_seed0_200draws.npz`, ships as package data and holds an optimised posterior mode (L-BFGS-B from two starts, polished with Newton steps), the inverse Hessian there and 200 thinned random-walk Metropolis draws. It has four cases, and only the first compares with a published number:

- **`dsge_estimation.sw07_structural_parameters_mode`** (published target): the optimised mode against the posterior **Mode** column of SW07 Table 1a and Table 1b (ECB WP 722, PDF pp. 35–36), not the Mean column. It checks 13 parameters: `csadjcost`, `csigma`, `chabb`, `csigl`, `cprobp`, `cfc`, `crr`, `crdy`, `ctrend`, and the markup-process parameters `crhopinf`, `cmap`, `crhow`, `cmaw`. All 13 are within 6.2% of the published modes (for example $\varphi$ 5.694 against 5.48, $\sigma_c$ 1.405 against 1.39, $\mu_p$ 0.703 against 0.74); the case uses `Tol.COARSE` (25%) because the data are today's FRED vintage, not SW's 2006 files.
- **`dsge_estimation.sw07_log_posterior_at_mode`**, **`dsge_estimation.sw07_laplace_marginal_data_density`**, **`dsge_estimation.sw07_harmonic_mean_mdd_consistency`** (puremacro regression values, **not published numbers**): the log posterior at the mode, `-822.04`, recomputed live; the Laplace log marginal data density, `-902.83`, from that value and the stored inverse Hessian; and Geweke's (1999) modified harmonic mean on the 200 stored draws, `-908.03`, with a spread of `2.66` log points across the truncation levels 0.1–0.9. With only 200 draws that spread is above the one-log-point convergence threshold of `harmonic_mean_mdd`, so this case pins the estimator's output, not a converged estimate. The three cases use `Tol.TIGHT` (2% relative, about ±16 to ±18 log points at these values), so they catch large changes to the model, data, priors or estimators; the test suite pins the stored mode's log posterior to $10^{-6}$. SW07 report no log posterior at the mode, and their marginal likelihood (Table 2: −905.8) is computed over 1966–2004 with 1956:1–1965:4 as a training sample. Since 4.6.0 `sw07_laplace_mdd` reproduces that computation on the authors' data (`presample=40`, `lik_init="diffuse"`): −932.3, with −921.4 for the options of the authors' replication `.mod`; at the authors' own mode puremacro agrees with Dynare 8 to 0.65 log points (−840.81 against −841.46), and Dynare 8 itself gives −923.1 rather than −905.8 on the public files. The published figure is thus not reproducible from those files and is not a comparable target; see the [Replication Gallery](replication.md).

Up to and including 4.3.0 this page gave `-1673.72` and `-1686.09` as the log posterior and Laplace targets and described the mode case as a check of Table 1 at the posterior mode. Those were puremacro outputs, not SW07 results, and the mode case compared the best of 200 MCMC draws with Mean-column values. The full comparison table is in the [Replication Gallery](replication.md).
