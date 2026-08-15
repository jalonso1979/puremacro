# puremacro 0.53.0 — generic Bayesian DSGE engine (R1a)

**Status:** draft 2026-05-23. Target release: **0.53.0**.

## Why

R1 from the 2026-05-23 research-directions brainstorm covers "fertility DSGE + generic Bayesian engine." That's too big for one release, so the user picked: engine first, fertility second. This spec is R1a — extract the generic Bayesian DSGE estimation pipeline from the SW07-specific implementation shipped in 0.50.0.

The shipping 0.50.0 surface (`puremacro.dsge.estimate_sw07` + `SW07PosteriorResult` + the priors / observation modules) is functional but hard-codes every step of the pipeline to Smets-Wouters. A user wanting to estimate any other DSGE model has to copy-paste and edit. R1b will port a fertility DSGE on top of the new engine, but only after the engine is independently testable and proven model-agnostic by a non-SW07 example.

This release extracts the pipeline into `estimate_dsge` (engine) + `puremacro.dsge.priors` (model-agnostic prior framework), refactors `estimate_sw07` into a thin wrapper, ships a toy AR(1) demo that proves the engine works on something other than SW07, and preserves backward compatibility via an alias `SW07PosteriorResult = DSGEPosteriorResult`.

## Scope

One release. No new external dependencies; no new MCMC samplers; no higher-order perturbation. Pure refactor + small generalisation + one demo.

**In scope:**

- New `puremacro/dsge/priors.py` — model-agnostic `log_prior`, `prior_means`, `prior_stds`, `param_bounds`, `param_names` helpers (extracted from `sw07_priors.py`).
- New `puremacro/dsge/estimate.py` — `estimate_dsge(data, *, observation_eq, priors, observed_vars, initial_params, fixed_params, model_name, n_draws, n_chains, burn_in, seed, proposal_scale) -> DSGEPosteriorResult`.
- Rename `SW07PosteriorResult` → `DSGEPosteriorResult` in `_results.py` with an alias for backward compat. Add optional `model_name: str = "unknown"` field.
- Refactor `puremacro/dsge/sw07_estimate.py::estimate_sw07` to a thin wrapper over `estimate_dsge`.
- Refactor `puremacro/dsge/sw07_priors.py` so `log_prior`, `prior_means`, `prior_stds`, `param_bounds`, `param_names` delegate to the generic helpers in `priors.py`.
- New example `puremacro/examples/dsge_ar1_demo.py` demonstrating `estimate_dsge` on a toy AR(1) state-space model.

**Out of scope:**

- Higher-order perturbation (2nd / 3rd order). The user's `fertility_housing_order2/3` Dynare files would need this — could become R1c.
- HMC / NUTS samplers. Could be R1d.
- Multi-model registry / discovery system.
- Posterior predictive checks, forecast evaluation, parallel tempering.
- **R1b — fertility DSGE port.** Queued as the next release (0.54.0) after this lands.

## Pre-conditions

- 0.52.0 shipped at tag `v0.52.0` (commit `9e2eb23`), pushed to `origin/feature/subnational-labor-uncertainty-us`.
- 6 release-gate gates green at 0.52.0 HEAD.
- Existing surface:
  - `puremacro.dsge.estimate_sw07(data=None, *, n_draws, n_chains, burn_in, seed) -> SW07PosteriorResult`.
  - `puremacro.dsge.SW07PosteriorResult` (10 fields).
  - `puremacro.dsge.sw07_priors.{PRIORS, log_prior, prior_means, prior_stds, param_bounds, param_names}`.
  - `puremacro.dsge.sw07_observation.{OBSERVED_VARS, make_state_space}`.
  - `puremacro.dsge.smets_wouters.{SW07_POSTERIOR_MODE, SW07_SHOCK_STDS, solve_sw07, STATE_NAMES, CONTROL_NAMES}`.
  - `puremacro.mcmc.random_walk_metropolis(log_posterior_fn, init, proposal_cov, n_draws, *, seed, accept_target, adapt_burnin) -> dict`.
  - `puremacro.numerics.numerical_hessian`.
  - `puremacro.state_space.{StateSpaceModel, kalman_filter}`.

## Architecture

```
puremacro/dsge/
  __init__.py          ← exports add: estimate_dsge, DSGEPosteriorResult, priors (module)
  priors.py            ← NEW (~150 LOC)
  estimate.py          ← NEW (~200 LOC)
  _results.py          ← rename SW07PosteriorResult → DSGEPosteriorResult; alias preserved
  sw07_priors.py       ← keep PRIORS dict; helpers become wrappers around priors.*
  sw07_estimate.py     ← estimate_sw07 becomes thin wrapper (~50 LOC after refactor)
  sw07_observation.py  ← unchanged
  smets_wouters.py     ← unchanged
  klein.py / gensys.py / load_dynare.py  ← unchanged
puremacro/examples/
  dsge_ar1_demo.py     ← NEW (~80 LOC) demo on a toy AR(1) state-space
tests/test_dsge/
  test_priors.py             ← NEW (~7 tests)
  test_estimate_dsge.py      ← NEW (~5 tests)
  test_sw07_wrapper.py       ← NEW or extended (~2-3 tests)
```

Each new file has one clear responsibility. Cross-file dependencies: `estimate.py → priors.py + numerics + state_space + mcmc`. `sw07_estimate.py → estimate.py + sw07_priors + sw07_observation`. `sw07_priors.py → priors.py`. Cycle-free.

## Component A — `puremacro/dsge/priors.py`

Model-agnostic prior framework. Extracted verbatim where possible from `sw07_priors.py`.

### Public API

```python
def log_prior(params: dict[str, float], priors: dict[str, dict]) -> float:
    """Sum log-pdfs across all params in priors.

    Returns ``-np.inf`` if any param value is outside its declared [lb, ub]
    support, or if a value is non-finite. Raises ``ValueError`` on an
    unknown ``dist`` name.
    """

def prior_means(priors: dict[str, dict]) -> dict[str, float]:
    """Return {param: mean} dict in the order priors was constructed."""

def prior_stds(priors: dict[str, dict]) -> dict[str, float]:
    """Return {param: std} dict."""

def param_bounds(priors: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (lb_array, ub_array) ordered by param insertion order."""

def param_names(priors: dict[str, dict]) -> tuple[str, ...]:
    """Return parameter names in insertion order."""
```

### Private dist-specific helpers

```python
def _logpdf_beta(x, mean, std) -> float       # mean/std parameterisation
def _logpdf_invgamma(x, s, nu) -> float       # Dynare PRIOR_P1=s, PRIOR_P2=nu
def _logpdf_normal(x, mean, std) -> float
def _logpdf_gamma(x, mean, std) -> float
```

These move from `sw07_priors.py` into `priors.py` unchanged (the math is model-agnostic). `sw07_priors._logpdf_*` private re-exports are dropped — internal callers update to `priors._logpdf_*`.

### Dispatch in `log_prior`

```python
_DIST_LOGPDF = {
    "beta":     _logpdf_beta,
    "invgamma": _logpdf_invgamma,
    "normal":   _logpdf_normal,
    "gamma":    _logpdf_gamma,
}

def log_prior(params, priors):
    total = 0.0
    for name, spec in priors.items():
        x = params[name]
        if not (spec["lb"] <= x <= spec["ub"]) or not math.isfinite(x):
            return -math.inf
        try:
            logpdf = _DIST_LOGPDF[spec["dist"]]
        except KeyError:
            raise ValueError(f"unknown distribution {spec['dist']!r} for param {name!r}")
        total += logpdf(x, spec["mean"], spec["std"])
    return total
```

## Component B — `puremacro/dsge/_results.py`

```python
@dataclass(frozen=True)
class DSGEPosteriorResult:
    """Result of estimate_dsge (and model-specific wrappers like estimate_sw07).

    [Existing fields preserved verbatim]
    draws : ndarray, shape (n_chains, n_draws, n_params)
    param_names : tuple of str
    log_posterior_trace : ndarray, shape (n_chains, n_draws)
    accept_rates : tuple of float
    mode : dict[str, float]
    mode_hessian_inv : ndarray, shape (n_params, n_params)
    n_burn_in : int
    data_n_obs : int
    seed : int
    model_name : str = "unknown"   # NEW in 0.53.0
    """
    draws: np.ndarray
    param_names: Tuple[str, ...]
    log_posterior_trace: np.ndarray
    accept_rates: Tuple[float, ...]
    mode: dict
    mode_hessian_inv: np.ndarray
    n_burn_in: int
    data_n_obs: int
    seed: int
    model_name: str = "unknown"

    def summary(self) -> pd.DataFrame:
        # unchanged
        ...

# Backward compatibility — pre-0.53.0 code imports SW07PosteriorResult.
SW07PosteriorResult = DSGEPosteriorResult
```

`SW07PosteriorResult` resolves to the same class object, so `isinstance(res, SW07PosteriorResult)` continues to work, and existing pickle / type hints carry through.

## Component C — `puremacro/dsge/estimate.py`

```python
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize as _scipy_minimize

from puremacro.dsge._results import DSGEPosteriorResult
from puremacro.dsge.priors import (
    log_prior, prior_means, prior_stds,
    param_bounds, param_names,
)
from puremacro.mcmc import random_walk_metropolis
from puremacro.numerics import numerical_hessian
from puremacro.state_space import StateSpaceModel, kalman_filter


def estimate_dsge(
    data: pd.DataFrame,
    *,
    observation_eq: Callable[[dict], StateSpaceModel],
    priors: dict[str, dict],
    observed_vars: Sequence[str],
    initial_params: dict[str, float],
    fixed_params: dict[str, float] | None = None,
    model_name: str = "unknown",
    n_draws: int = 20_000,
    n_chains: int = 4,
    burn_in: int = 2_000,
    seed: int = 0,
    proposal_scale: float | None = None,
) -> DSGEPosteriorResult:
    """Generic Bayesian DSGE estimator via random-walk Metropolis-Hastings.

    Pipeline: mode refinement (scipy.optimize.minimize on -log_posterior,
    L-BFGS-B with declared bounds) → numerical Hessian at mode → proposal
    cov c²·H⁻¹ (fallback diag(prior_stds²) if H not PD) → n_chains
    RW-MH chains via puremacro.mcmc.random_walk_metropolis →
    DSGEPosteriorResult.

    Default ``c = 2.38/sqrt(n_params)`` (Roberts-Gelman-Gilks 1997).
    User can override via ``proposal_scale``.
    """
```

Implementation outline (~200 LOC):

1. Validate `data.columns ⊇ observed_vars`; raise `ValueError` otherwise.
2. Combine `param_names(priors)` with `fixed_params` keys; verify `initial_params` covers all estimated names.
3. Build `log_posterior(theta_array) -> float` closure as described in Section 2 (rebuild dict, add fixed_params, log_prior + Kalman log-lik, propagate `-inf` on numerical failure).
4. Refine mode: `scipy.optimize.minimize(-log_posterior, init_array, bounds=list(zip(lb, ub)), method="L-BFGS-B")`. Warn if `res.success is False`; use `res.x` regardless.
5. Numerical Hessian at mode via `numerical_hessian(log_posterior, mode_array)`.
6. Proposal cov: `c² · np.linalg.inv(-H)` if `-H` PD; else `diag(prior_stds_array²)` with warning.
7. Run `n_chains` chains via `random_walk_metropolis(log_posterior, mode_array, proposal_cov, n_draws + burn_in, seed=seed + chain_idx, accept_target=0.234, adapt_burnin=burn_in)`.
8. Discard first `burn_in` draws per chain; stack into `(n_chains, n_draws, n_params)` array.
9. Return `DSGEPosteriorResult(...)`.

## Component D — `puremacro/dsge/sw07_estimate.py` (refactored)

After the refactor, the entire file is ~50 LOC:

```python
"""Thin wrapper: estimate Smets-Wouters (2007) via the generic Bayesian DSGE engine."""
from __future__ import annotations

import importlib.resources

import pandas as pd

from puremacro.dsge._results import DSGEPosteriorResult
from puremacro.dsge.estimate import estimate_dsge
from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.dsge.sw07_priors import PRIORS


_FIXED_PARAMS = {
    "ctou":     0.025,
    "clandaw":  1.5,
    "cg":       0.18,
    "curvp":    10.0,
    "curvw":    10.0,
}


def _load_bundled_data() -> pd.DataFrame:
    pkg = importlib.resources.files("puremacro.dsge")
    return pd.read_csv(pkg / "_sw07_data.csv", comment="#",
                       parse_dates=["date"], index_col="date")


def _validate_data(df: pd.DataFrame) -> None:
    missing = set(OBSERVED_VARS) - set(df.columns)
    if missing:
        raise ValueError(f"data missing columns: {sorted(missing)}")


def estimate_sw07(
    data: pd.DataFrame | None = None,
    *,
    n_draws: int = 20_000,
    n_chains: int = 4,
    burn_in: int = 2_000,
    seed: int = 0,
) -> DSGEPosteriorResult:
    """Estimate SW07 by RW-MH via the generic engine."""
    if data is None:
        data = _load_bundled_data()
    _validate_data(data)
    initial_params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    return estimate_dsge(
        data,
        observation_eq=make_state_space,
        priors=PRIORS,
        observed_vars=list(OBSERVED_VARS),
        initial_params=initial_params,
        fixed_params=_FIXED_PARAMS,
        model_name="SW07",
        n_draws=n_draws, n_chains=n_chains, burn_in=burn_in, seed=seed,
    )
```

All the SW07-specific logic that was inline in 0.52.0 — Hessian fallback, mode refinement, MH driver, accept-rate aggregation — now lives in `estimate.py`.

## Component E — `puremacro/dsge/sw07_priors.py` (refactored)

The `PRIORS` dict stays. The function bodies become 3-line wrappers:

```python
from puremacro.dsge import priors as _priors


def log_prior(params: dict[str, float]) -> float:
    return _priors.log_prior(params, PRIORS)


def prior_means() -> dict[str, float]:
    return _priors.prior_means(PRIORS)


def prior_stds() -> dict[str, float]:
    return _priors.prior_stds(PRIORS)


def param_bounds() -> tuple[np.ndarray, np.ndarray]:
    return _priors.param_bounds(PRIORS)


def param_names() -> tuple[str, ...]:
    return _priors.param_names(PRIORS)
```

The dist-specific `_logpdf_*` helpers are deleted (moved to `priors.py`).

## Component F — `puremacro/examples/dsge_ar1_demo.py`

A tiny demo proving the engine is model-agnostic. Estimates a 1-parameter AR(1) state-space model:

```
state:  x_t = ρ · x_{t-1} + σ · ε_t,    ε_t ~ N(0, 1)
obs:    y_t = x_t                       (no measurement noise)
```

Two parameters: `ρ` (beta prior, mean 0.5, std 0.2) and `σ` (invgamma prior, s=0.1, nu=2).

```python
"""Toy AR(1) state-space — demonstrates puremacro.dsge.estimate_dsge.

This is a 2-parameter model — the smallest non-trivial Bayesian DSGE
estimation. It proves the generic engine is genuinely model-agnostic
(it's not silently coupled to SW07-specific structure).
"""
import numpy as np
import pandas as pd

from puremacro.dsge import estimate_dsge
from puremacro.state_space import StateSpaceModel


PRIORS = {
    "rho":   {"dist": "beta",     "mean": 0.5, "std": 0.2, "lb": 0.001, "ub": 0.99},
    "sigma": {"dist": "invgamma", "mean": 0.1, "std": 2.0, "lb": 0.01,  "ub": 5.0},
}


def make_state_space(params: dict) -> StateSpaceModel:
    """y_t = x_t,  x_t = rho * x_{t-1} + sigma * eps_t."""
    rho = params["rho"]
    sigma = params["sigma"]
    return StateSpaceModel(
        T=np.array([[rho]]),
        Z=np.array([[1.0]]),
        R=np.array([[1.0]]),
        Q=np.array([[sigma ** 2]]),
        H=np.array([[1e-8]]),  # tiny measurement noise for numerical stability
        c=np.zeros(1),
        d=np.zeros(1),
    )


def _simulate(rho_true: float, sigma_true: float, T: int, seed: int):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T) * sigma_true
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho_true * x[t - 1] + eps[t]
    return pd.DataFrame({"y": x})


def main():
    rho_true, sigma_true = 0.7, 0.5
    data = _simulate(rho_true, sigma_true, T=500, seed=0)
    res = estimate_dsge(
        data,
        observation_eq=make_state_space,
        priors=PRIORS,
        observed_vars=["y"],
        initial_params={"rho": 0.5, "sigma": 0.4},
        model_name="AR1_demo",
        n_chains=1, n_draws=2000, burn_in=500, seed=0,
    )
    summary = res.summary()
    print(summary)
    print(f"\ntrue rho={rho_true:.3f}, posterior mean={summary.loc['rho', 'mean']:.3f}")
    print(f"true sigma={sigma_true:.3f}, posterior mean={summary.loc['sigma', 'mean']:.3f}")


if __name__ == "__main__":
    main()
```

## Data flow

```
estimate_dsge(data, observation_eq, priors, ...)
   │
   1. Validate data.columns ⊇ observed_vars
   2. Build log_posterior(theta) closure
       theta → params dict → log_prior(params, priors)
                         → observation_eq(params) → StateSpaceModel
                         → kalman_filter(ss, data) → log-likelihood
                         → log_prior + log_likelihood
   3. Mode refinement (scipy.optimize.minimize, L-BFGS-B, bounds)
   4. Numerical Hessian at mode
   5. Proposal cov c²·H⁻¹  (fallback diag(prior_stds²) if H not PD)
   6. n_chains × random_walk_metropolis(log_posterior, mode, Σ, ...)
   7. Drop burn_in; package DSGEPosteriorResult

estimate_sw07(data=None)
   │
   1. Load bundled CSV if data is None
   2. Validate against OBSERVED_VARS
   3. estimate_dsge(data, make_state_space, PRIORS, OBSERVED_VARS,
                    SW07_POSTERIOR_MODE | SW07_SHOCK_STDS, _FIXED_PARAMS,
                    model_name="SW07", ...)
```

## Error handling

| Failure mode | Component | Handling |
|---|---|---|
| `data` missing observed_var | estimate_dsge | `ValueError("data missing columns: [...]")` |
| `priors` has unknown `dist` | priors.log_prior | `ValueError("unknown distribution 'foo' for param 'x'")` |
| `initial_params` missing a name in priors | estimate_dsge | `KeyError("initial_params missing param 'foo'")` |
| `observation_eq` raises (LinAlgError, ValueError) inside log_posterior | log_posterior closure | catch → return `-inf` |
| `observation_eq` returns non-StateSpaceModel at validation | estimate_dsge | `TypeError` once, at startup |
| Kalman returns NaN log-likelihood | log_posterior closure | return `-inf` |
| Hessian at mode not PD | estimate_dsge | `warnings.warn`; fallback to `diag(prior_stds²)` |
| `scipy.optimize.minimize` non-success | estimate_dsge | `warnings.warn`; use best iterate |
| Persistent low accept rate (< 0.5%) on any chain | estimate_dsge | warn after MH completes |
| `proposal_scale` non-finite or non-positive | estimate_dsge | `ValueError` |

## Testing

Total: ~16-17 new unit tests across three test files (9 priors + 5 estimate + 2-3 sw07-wrapper).

### `tests/test_dsge/test_priors.py` (~9 tests)

- `test_logpdf_beta_matches_scipy_mean_std_parameterisation` — verify Beta logpdf.
- `test_logpdf_invgamma_dynare_convention` — verify the s/nu Dynare convention matches the implementation.
- `test_logpdf_normal_matches_scipy`.
- `test_logpdf_gamma_matches_scipy`.
- `test_log_prior_sums_across_params` — toy 3-param priors dict, hand-verified sum.
- `test_log_prior_returns_neg_inf_outside_bounds` — out-of-bound param → `-inf`.
- `test_log_prior_raises_on_unknown_dist` — `{"dist": "weibull"}` → `ValueError`.
- `test_param_bounds_returns_lb_ub_arrays_in_order` — order preserved.
- `test_param_names_preserves_dict_order` — Python 3.7+ guarantee.

### `tests/test_dsge/test_estimate_dsge.py` (~5 tests)

- `test_estimate_dsge_returns_dsgeposteriorresult` — basic API.
- `test_estimate_dsge_validates_missing_observed_var` — drop a column → `ValueError`.
- `test_estimate_dsge_toy_ar1_recovers_rho` — synthetic AR(1) with ρ=0.7, σ=0.5, T=500, seed=0 → posterior mean of ρ within ±0.1 of 0.7.
- `test_estimate_dsge_model_name_field_set` — `model_name="MyToy"` → result.model_name == "MyToy".
- `test_estimate_dsge_kalman_singular_returns_neg_inf` — observation_eq that returns singular Σ → log_posterior returns -inf, MCMC proceeds without crashing.

### `tests/test_dsge/test_sw07_wrapper.py` (~2-3 tests)

- `test_estimate_sw07_returns_dsge_posterior_result` — `isinstance(res, DSGEPosteriorResult)`.
- `test_sw07posteriorresult_is_alias_for_dsge` — `SW07PosteriorResult is DSGEPosteriorResult`.
- `test_sw07_parity_short_chain` — golden snapshot: estimate_sw07 with `seed=0, n_chains=1, n_draws=200, burn_in=50` produces draws array hashed to a frozen reference value (or compared to a stored ndarray fixture). Captures the pre-refactor behaviour BEFORE the refactor; passes after if numerical parity holds.

### Markers

- No `@pytest.mark.slow` on any new test (each <5s individually).
- The AR(1) recovery test runs n_draws=2000 + n_chains=1, expected ~3-5s. Acceptable.
- The SW07 parity test uses the smallest config that still exercises the full pipeline: `n_draws=200, n_chains=1, burn_in=50`. Expected ~30-60s. If this lands in the `slow` bucket, mark it `@pytest.mark.slow` — but try without first.

## Acceptance criteria for 0.53.0

1. `puremacro.dsge.estimate_dsge` exported from `puremacro.dsge.__init__`.
2. `puremacro.dsge.DSGEPosteriorResult` exported; has new optional `model_name: str = "unknown"` field.
3. `puremacro.dsge.SW07PosteriorResult` still importable (alias points to `DSGEPosteriorResult`).
4. `puremacro.dsge.priors` is a public submodule exposing `log_prior, prior_means, prior_stds, param_bounds, param_names`.
5. `estimate_sw07` returns numerically identical results to 0.52.0 on the bundled dataset with `seed=0, n_chains=1, n_draws=200, burn_in=50` (golden-snapshot parity test).
6. `puremacro/examples/dsge_ar1_demo.py` runs cleanly under the examples-gallery harness; recovers `ρ` within ±0.1 of the true value.
7. ~16-17 new unit tests green under CPython.
8. Public-API snapshot regenerated.
9. All 6 release-gate gates green at HEAD.
10. CHANGELOG 0.53.0 entry. **No breaking changes** — `SW07PosteriorResult` alias preserves backward compat.
11. Version bumped 0.52.0 → 0.53.0.

## Risks and mitigations

1. **Behavioural divergence from 0.50.0 SW07.** Refactoring `estimate_sw07` to call `estimate_dsge` could inadvertently change the posterior numerics. *Mitigation:* the golden-snapshot parity test (criterion 5) freezes a short-chain reference BEFORE the refactor; the refactor commit must reproduce the same draws byte-for-byte (or within `1e-12`). If the parity fails, investigate and fix before merge — don't relax the test.

2. **`observation_eq` callable contract.** Users may pass non-pure functions. *Mitigation:* engine docstring explicitly says the callable must be pure; no enforcement.

3. **Toy-AR(1) recovery test flakiness.** Statistical recovery test may flake on edge seeds. *Mitigation:* fixed seed, T=500, n_draws=2000, tolerance ±0.1. Implementer must verify the test passes on three different seeds before committing; if any single seed produces failure, widen tolerance or increase T.

4. **`param_bounds` shape vs. scipy.** scipy.optimize wants `[(lb_i, ub_i), ...]`; the helper returns `(lb_array, ub_array)`. *Mitigation:* engine adapter calls `list(zip(lb_array, ub_array))` internally.

5. **`priors.log_prior` performance.** Dict iteration per MH iteration adds small overhead vs. inlined SW07. *Mitigation:* benchmark on the SW07 short-chain. If overhead >10%, pre-extract `(name, dist_callable, mean, std, lb, ub)` tuples once at engine entry.

6. **`puremacro.dsge.priors` is a new module name.** No name collision today; flagged here for future awareness.

## Out of scope (deferred)

- Higher-order (2nd / 3rd) perturbation — needed for the user's `fertility_housing_order2/3` Dynare models. Could become R1c.
- HMC / NUTS samplers.
- Multi-model registry / discovery.
- Posterior predictive checks, forecast evaluation, parallel tempering.
- **R1b — fertility DSGE port.** Queued as 0.54.0.
- **R3 — paleoclimate VARX / long-run cliometrics.** Queued after R1b.
