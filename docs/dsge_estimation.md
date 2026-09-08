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

## What this deliberately does not do

- **No macro processor.** `@#define`, `@#for`, `@#if`, `@#include` and `@{...}` raise `DynareFeatureError`. They used to be ignored, which meant a file loaded clean and a *different model* was solved. Expand them with `dynare model.mod savemacro` and pass the expanded file. Planned for 2.7.0.
- **No expression parser**, so `STEADY_STATE()`, `EXPECTATION()`, `normcdf`, and a model-local `#` variable defined over an endogenous variable all raise. Also 2.7.0.
- **First order only.** A second-order solution is refused rather than silently linearised; a particle filter for it is a later release.
- **The sampler adapts a scalar, not a covariance** — and its adaptation fires only every 100 iterations, so a `burn_in` below 100 never adapts at all and can leave the chain stuck with a 0% acceptance rate. Give it at least a few hundred.
- **`mode_compute` defaults to `"lbfgs"`**, not to the better `csminwel`, because changing it changes every posterior previously produced.
- **`observation_trends` is applied by detrending the data**, since the state space is time-invariant by construction.
- **Marginal likelihoods from before 2.5.0 are not comparable** to these or to each other: the Kalman recursion then started from a diffuse `P0`, worth about 114 log points on Smets–Wouters.
