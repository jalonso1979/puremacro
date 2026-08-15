# puremacro 0.53.0 Implementation Plan — generic Bayesian DSGE engine (R1a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract from `estimate_sw07` a model-agnostic Bayesian DSGE estimator (`estimate_dsge`) plus a generic prior framework (`puremacro.dsge.priors`); refactor `estimate_sw07` into a thin wrapper; ship a toy AR(1) demo proving the engine is genuinely model-agnostic.

**Architecture:** New `puremacro/dsge/priors.py` holds the model-agnostic prior helpers. New `puremacro/dsge/estimate.py` holds the generic estimator (mode refinement, Hessian, MH driver — currently inlined in `sw07_estimate.py`). `puremacro/dsge/sw07_*.py` files become thin wrappers around the generic engine. `SW07PosteriorResult` is renamed to `DSGEPosteriorResult` with a backward-compat alias.

**Tech Stack:** numpy, scipy, pandas. No new dependencies.

**Spec:** `docs/specs/2026-05-23-puremacro-053-generic-dsge-engine-design.md`

---

## File map

### New files
- `puremacro/dsge/priors.py` (~150 LOC).
- `puremacro/dsge/estimate.py` (~250 LOC after moving the helpers from `sw07_estimate.py`).
- `puremacro/examples/dsge_ar1_demo.py` (~80 LOC).
- `tests/test_dsge/test_priors.py` (~9 tests).
- `tests/test_dsge/test_estimate_dsge.py` (~5 tests).
- `tests/test_dsge/test_sw07_wrapper.py` (~3 tests including the parity test).
- `tests/fixtures/sw07_parity_seed0_200draws.npz` — frozen pre-refactor reference for the parity test.

### Modified files
- `puremacro/dsge/_results.py` — rename + alias + new `model_name` field.
- `puremacro/dsge/sw07_priors.py` — strip dist-specific logpdfs (moved to `priors.py`); 5 public helpers become 3-line delegators.
- `puremacro/dsge/sw07_estimate.py` — strip mode refinement / Hessian / MH driver / `_initial_vec` / `_nearest_pd` / `_find_finite_start` (moved to `estimate.py`). After refactor: ~60 LOC, all SW07-specific (data loading, validation, `_FIXED_PARAMS`, initial-params construction, call into `estimate_dsge`).
- `puremacro/dsge/__init__.py` — add `estimate_dsge`, `DSGEPosteriorResult`, `priors` (module) to exports.
- `puremacro/__init__.py` — bump `__version__` to `"0.53.0"`.
- `pyproject.toml` — bump `version`.
- `CHANGELOG.md` — add 0.53.0 section.
- `tests/test_import.py` — bump pinned version assertion.
- `tests/fixtures/public_api_snapshot.json` — regenerate.

### Verified API surfaces
- `puremacro.dsge.sw07_priors.PRIORS` — dict `{name: {dist, mean, std, lb, ub}}`, 36 entries.
- `puremacro.dsge.sw07_priors.{log_prior, prior_means, prior_stds, param_bounds, param_names}` — current bodies use module-global `PRIORS`; will become 3-line delegators in Task 3.
- `puremacro.dsge.sw07_priors._logpdf_{beta,gamma,normal,invgamma,for_spec}` — private helpers; move to `priors.py`.
- `puremacro.dsge.sw07_estimate.{_vec_to_dict, _make_neg_log_posterior, _initial_vec, _nearest_pd, _find_finite_start, _FIXED_PARAMS, estimate_sw07}` — current bodies; the `_*` privates move to `estimate.py` (model-agnostic) except `_FIXED_PARAMS` (SW07-specific, stays).
- `puremacro.dsge._results.SW07PosteriorResult` — 10 fields: `draws, param_names, log_posterior_trace, accept_rates, mode, mode_hessian_inv, n_burn_in, data_n_obs, seed`.
- `puremacro.dsge.sw07_observation.{OBSERVED_VARS, make_state_space}`.
- `puremacro.dsge.smets_wouters.{SW07_POSTERIOR_MODE, SW07_SHOCK_STDS}`.
- `puremacro.state_space.kalman_filter(y, ssm) -> dict` (positional y, ssm); returns dict with `"loglik"` key.
- `puremacro.numerics.numerical_hessian(fn, x, h=1e-4) -> ndarray`.
- `puremacro.mcmc.random_walk_metropolis(log_posterior_fn, init, proposal_cov, n_draws, *, seed, accept_target, adapt_burnin) -> dict` with keys `"chain"`, `"log_post"`, `"accept_rate"`.

---

## Task 1: Capture pre-refactor golden parity snapshot

**Files:**
- Create: `tests/fixtures/sw07_parity_seed0_200draws.npz`.
- Create: `tests/test_dsge/test_sw07_wrapper.py` (just the parity test for now; Tasks 4 + 7 add more).
- Modify: `puremacro/tests/test_dsge/__init__.py` if it doesn't already exist (empty file).

This task captures the current `estimate_sw07` behaviour BEFORE any refactor, so Task 6's refactor can be verified to preserve numerics.

- [ ] **Step 1: Create the capture script and run it**

Create a one-off script `/tmp/capture_sw07_parity.py`:
```python
"""One-off: capture pre-refactor estimate_sw07 output as a frozen fixture."""
import numpy as np
from puremacro.dsge.sw07_estimate import estimate_sw07

res = estimate_sw07(seed=0, n_chains=1, n_draws=200, burn_in=50)
np.savez(
    "tests/fixtures/sw07_parity_seed0_200draws.npz",
    draws=res.draws,
    log_posterior_trace=res.log_posterior_trace,
    accept_rates=np.array(res.accept_rates),
    mode_values=np.array([res.mode[n] for n in res.param_names]),
    mode_hessian_inv=res.mode_hessian_inv,
    param_names=np.array(res.param_names),
)
print(f"saved fixture: {res.draws.shape} draws, accept_rate={res.accept_rates[0]:.3f}")
```

Run:
```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
mkdir -p tests/fixtures tests/test_dsge
python /tmp/capture_sw07_parity.py
```

Expected: saves `tests/fixtures/sw07_parity_seed0_200draws.npz` (~60KB). May take 30–120 seconds (mode refinement + Hessian + 250 MH iterations).

- [ ] **Step 2: Create test_dsge/__init__.py if missing**

```bash
test -f tests/test_dsge/__init__.py || touch tests/test_dsge/__init__.py
```

- [ ] **Step 3: Write the parity test**

Create `tests/test_dsge/test_sw07_wrapper.py`:
```python
"""Tests for the SW07 thin-wrapper layer over the generic Bayesian DSGE engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


FIXTURE = Path(__file__).parent.parent / "fixtures" / "sw07_parity_seed0_200draws.npz"


@pytest.mark.slow
def test_sw07_parity_short_chain():
    """estimate_sw07(seed=0, n_chains=1, n_draws=200, burn_in=50) reproduces
    the pre-0.53.0 reference output byte-for-byte (within 1e-10 tolerance)."""
    from puremacro.dsge.sw07_estimate import estimate_sw07
    ref = np.load(FIXTURE)
    res = estimate_sw07(seed=0, n_chains=1, n_draws=200, burn_in=50)
    np.testing.assert_allclose(res.draws, ref["draws"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        res.log_posterior_trace, ref["log_posterior_trace"], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.array(res.accept_rates), ref["accept_rates"], rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.array([res.mode[n] for n in res.param_names]), ref["mode_values"],
        rtol=0, atol=1e-10,
    )
    np.testing.assert_allclose(
        res.mode_hessian_inv, ref["mode_hessian_inv"], rtol=0, atol=1e-10
    )
    assert tuple(res.param_names) == tuple(str(n) for n in ref["param_names"])
```

- [ ] **Step 4: Run the parity test, expect PASS**

```bash
pytest tests/test_dsge/test_sw07_wrapper.py::test_sw07_parity_short_chain -v -m slow
```

Expected: PASS. The fixture was just captured from the same function, so this is a tautology now — but it locks in the reference for Task 6.

- [ ] **Step 5: Commit fixture + test**

```bash
git add tests/fixtures/sw07_parity_seed0_200draws.npz tests/test_dsge/test_sw07_wrapper.py tests/test_dsge/__init__.py
git commit -m "test(dsge): capture pre-refactor estimate_sw07 parity snapshot"
```

---

## Task 2: Create puremacro/dsge/priors.py

**Files:**
- Create: `puremacro/dsge/priors.py`.
- Create: `tests/test_dsge/test_priors.py`.

- [ ] **Step 1: Write 9 failing tests**

Create `tests/test_dsge/test_priors.py`:
```python
"""Tests for the model-agnostic prior framework."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats


# Tiny toy priors dict for testing.
_TOY_PRIORS = {
    "rho":   {"dist": "beta",     "mean": 0.5, "std": 0.2, "lb": 0.001, "ub": 0.99},
    "mu":    {"dist": "normal",   "mean": 0.0, "std": 1.0, "lb": -5.0,  "ub": 5.0},
    "sigma": {"dist": "invgamma", "mean": 0.1, "std": 2.0, "lb": 0.01,  "ub": 5.0},
    "alpha": {"dist": "gamma",    "mean": 1.0, "std": 0.5, "lb": 0.001, "ub": 10.0},
}


def test_logpdf_beta_matches_scipy():
    from puremacro.dsge.priors import _logpdf_beta
    # Beta(a, b) where a, b derived from mean=0.5, std=0.2
    mean, std = 0.5, 0.2
    a = mean * (mean * (1 - mean) / std**2 - 1)
    b = a * (1 - mean) / mean
    x = 0.4
    expected = float(stats.beta.logpdf(x, a, b))
    assert _logpdf_beta(x, mean, std) == pytest.approx(expected, rel=1e-12)


def test_logpdf_invgamma_dynare_convention():
    from puremacro.dsge.priors import _logpdf_invgamma
    # Dynare convention: P1=s, P2=nu maps to scipy.invgamma(a=nu/2, scale=s**2*nu/2).
    s, nu = 0.1, 2.0
    x = 0.15
    expected = float(stats.invgamma.logpdf(x, a=nu/2, scale=s**2 * nu / 2))
    assert _logpdf_invgamma(x, s, nu) == pytest.approx(expected, rel=1e-12)


def test_logpdf_normal_matches_scipy():
    from puremacro.dsge.priors import _logpdf_normal
    expected = float(stats.norm.logpdf(0.3, loc=0.0, scale=1.0))
    assert _logpdf_normal(0.3, 0.0, 1.0) == pytest.approx(expected, rel=1e-12)


def test_logpdf_gamma_matches_scipy():
    from puremacro.dsge.priors import _logpdf_gamma
    mean, std = 1.0, 0.5
    k = (mean / std) ** 2
    theta = std**2 / mean
    expected = float(stats.gamma.logpdf(0.8, a=k, scale=theta))
    assert _logpdf_gamma(0.8, mean, std) == pytest.approx(expected, rel=1e-12)


def test_log_prior_sums_across_params():
    from puremacro.dsge.priors import log_prior, _logpdf_beta, _logpdf_normal, _logpdf_invgamma, _logpdf_gamma
    params = {"rho": 0.5, "mu": 0.0, "sigma": 0.1, "alpha": 1.0}
    expected = (
        _logpdf_beta(0.5, 0.5, 0.2)
        + _logpdf_normal(0.0, 0.0, 1.0)
        + _logpdf_invgamma(0.1, 0.1, 2.0)
        + _logpdf_gamma(1.0, 1.0, 0.5)
    )
    assert log_prior(params, _TOY_PRIORS) == pytest.approx(expected, rel=1e-12)


def test_log_prior_returns_neg_inf_outside_bounds():
    from puremacro.dsge.priors import log_prior
    params = {"rho": 1.5, "mu": 0.0, "sigma": 0.1, "alpha": 1.0}  # rho > ub
    assert log_prior(params, _TOY_PRIORS) == -math.inf


def test_log_prior_raises_on_unknown_dist():
    from puremacro.dsge.priors import log_prior
    weird = {"x": {"dist": "weibull", "mean": 1.0, "std": 1.0, "lb": 0.0, "ub": 10.0}}
    with pytest.raises(ValueError, match="unknown distribution"):
        log_prior({"x": 1.0}, weird)


def test_param_bounds_returns_list_of_tuples_in_order():
    from puremacro.dsge.priors import param_bounds
    bounds = param_bounds(_TOY_PRIORS)
    assert bounds == [(0.001, 0.99), (-5.0, 5.0), (0.01, 5.0), (0.001, 10.0)]


def test_param_names_preserves_dict_order():
    from puremacro.dsge.priors import param_names
    assert param_names(_TOY_PRIORS) == ("rho", "mu", "sigma", "alpha")
```

- [ ] **Step 2: Run tests, expect FAIL**

```bash
pytest tests/test_dsge/test_priors.py -v
```

Expected: `ImportError: cannot import 'priors' from 'puremacro.dsge'`

- [ ] **Step 3: Implement priors.py**

Create `puremacro/dsge/priors.py`:
```python
"""Model-agnostic prior framework for Bayesian DSGE estimation.

A `priors` dict has the shape:
    {param_name: {"dist": str, "mean": float, "std": float, "lb": float, "ub": float}}

Supported distributions: ``"beta"``, ``"gamma"``, ``"normal"``, ``"invgamma"``
(Dynare ``inv_gamma_pdf(P1=s, P2=nu)`` convention).

This module is the engine-side complement to model-specific prior dicts
like ``puremacro.dsge.sw07_priors.PRIORS``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _logpdf_beta(x: float, mean: float, std: float) -> float:
    """log-pdf of Beta(a, b) parameterised by mean/std (Pfeifer convention)."""
    a = mean * (mean * (1 - mean) / std**2 - 1)
    b = a * (1 - mean) / mean
    return float(stats.beta.logpdf(x, a, b))


def _logpdf_gamma(x: float, mean: float, std: float) -> float:
    """log-pdf of Gamma(k, theta) parameterised by mean/std."""
    if x <= 0.0:
        return -math.inf
    k = (mean / std) ** 2
    theta = std ** 2 / mean
    return float(stats.gamma.logpdf(x, a=k, scale=theta))


def _logpdf_normal(x: float, mean: float, std: float) -> float:
    """log-pdf of Normal(mean, std)."""
    return float(stats.norm.logpdf(x, loc=mean, scale=std))


def _logpdf_invgamma(x: float, s: float, nu: float) -> float:
    """log-pdf of Inverse-Gamma under Dynare's inv_gamma_pdf parameterisation.

    Dynare's inv_gamma_pdf(P1=s, P2=nu) corresponds to:
        IG(shape=nu/2, scale=s^2 * nu/2)
    """
    if x <= 0.0:
        return -math.inf
    a = nu / 2.0
    scale = s ** 2 * nu / 2.0
    return float(stats.invgamma.logpdf(x, a=a, scale=scale))


_DIST_LOGPDF = {
    "beta":     _logpdf_beta,
    "gamma":    _logpdf_gamma,
    "normal":   _logpdf_normal,
    "invgamma": _logpdf_invgamma,
}


def _logpdf_for_spec(spec: dict, x: float) -> float:
    """Dispatch to the correct log-pdf given a single param spec and a value."""
    if not (spec["lb"] <= x <= spec["ub"]):
        return -math.inf
    dist = spec["dist"]
    try:
        fn = _DIST_LOGPDF[dist]
    except KeyError as exc:
        raise ValueError(f"unknown distribution {dist!r} for prior spec") from exc
    return fn(x, spec["mean"], spec["std"])


def log_prior(params: dict, priors: dict) -> float:
    """Sum of log-prior densities across all parameters in ``priors``.

    Returns -inf if any parameter value is missing, non-finite, or outside
    its declared ``[lb, ub]`` support. Raises ``ValueError`` if any spec
    declares an unsupported ``dist``.
    """
    total = 0.0
    for name, spec in priors.items():
        if name not in params:
            return -math.inf
        x = params[name]
        if not math.isfinite(x):
            return -math.inf
        contrib = _logpdf_for_spec(spec, x)
        if contrib == -math.inf:
            return -math.inf
        total += contrib
    return total


def prior_means(priors: dict) -> dict[str, float]:
    """Return ``{name: mean}`` in priors-dict insertion order."""
    return {name: spec["mean"] for name, spec in priors.items()}


def prior_stds(priors: dict) -> dict[str, float]:
    """Return ``{name: std}`` in priors-dict insertion order."""
    return {name: spec["std"] for name, spec in priors.items()}


def param_bounds(priors: dict) -> list[tuple[float, float]]:
    """Return ``[(lb, ub), ...]`` in priors-dict insertion order.

    Matches the shape scipy.optimize.minimize expects for ``bounds``.
    """
    return [(spec["lb"], spec["ub"]) for spec in priors.values()]


def param_names(priors: dict) -> tuple[str, ...]:
    """Return parameter names in priors-dict insertion order."""
    return tuple(priors.keys())


__all__ = [
    "log_prior",
    "prior_means",
    "prior_stds",
    "param_bounds",
    "param_names",
]
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_dsge/test_priors.py -v
```

Expected: 9/9 PASS.

- [ ] **Step 5: Commit**

```bash
git add puremacro/dsge/priors.py tests/test_dsge/test_priors.py
git commit -m "feat(dsge): puremacro.dsge.priors — model-agnostic prior framework"
```

---

## Task 3: Refactor sw07_priors.py to delegate to puremacro.dsge.priors

**Files:**
- Modify: `puremacro/dsge/sw07_priors.py`.

The existing dist-specific helpers (`_logpdf_beta`, etc.) and dispatcher (`_logpdf_for_spec`) in `sw07_priors.py` now duplicate `puremacro.dsge.priors`. Strip them; rewire the public helpers (`log_prior`, `prior_means`, `prior_stds`, `param_bounds`, `param_names`) to call the generic versions with `PRIORS`.

- [ ] **Step 1: Apply the refactor**

In `puremacro/dsge/sw07_priors.py`:

(a) Delete the bodies of `_logpdf_beta`, `_logpdf_gamma`, `_logpdf_normal`, `_logpdf_invgamma`, `_logpdf_for_spec` (all the dist-specific code from after the `PRIORS` dict definition down to the `# Public API` header, plus the dispatcher).

(b) Replace the public API block with:
```python
# === Public API (delegators to puremacro.dsge.priors) =========================

from puremacro.dsge import priors as _priors


def log_prior(params: dict) -> float:
    """Sum of log-prior densities across the SW07 estimated parameters."""
    return _priors.log_prior(params, PRIORS)


def prior_means() -> dict[str, float]:
    """Return a dict of {param_name: prior_mean} for all SW07 estimated params."""
    return _priors.prior_means(PRIORS)


def prior_stds() -> dict[str, float]:
    """Return a dict of {param_name: prior_std} for all SW07 estimated params."""
    return _priors.prior_stds(PRIORS)


def param_bounds() -> list[tuple[float, float]]:
    """Return list of (lb, ub) tuples in PRIORS dict order."""
    return _priors.param_bounds(PRIORS)


def param_names() -> tuple[str, ...]:
    """Return parameter names in PRIORS dict order."""
    return _priors.param_names(PRIORS)
```

(c) Update `__all__` to drop the private helpers (which no longer exist in this module):
```python
__all__ = [
    "PRIORS",
    "log_prior",
    "prior_means",
    "prior_stds",
    "param_bounds",
    "param_names",
]
```

(d) Update the module imports — `math`, `numpy`, `scipy.stats` may no longer be used after the strip; remove if unused. Add `from puremacro.dsge import priors as _priors`.

- [ ] **Step 2: Run the existing SW07 priors tests + the new generic tests**

```bash
pytest tests/test_dsge/ -v -m "not slow"
```

Expected: ALL existing tests + the 9 new generic-priors tests PASS.

If any existing test fails (e.g., because it imported `puremacro.dsge.sw07_priors._logpdf_beta`), the fix is to update that test to import from `puremacro.dsge.priors` instead. Do not re-introduce the helpers in `sw07_priors.py`.

- [ ] **Step 3: Run the parity test, expect PASS**

```bash
pytest tests/test_dsge/test_sw07_wrapper.py::test_sw07_parity_short_chain -v -m slow
```

Expected: PASS (log_prior values are byte-identical because they go through the same `_logpdf_*` helpers, just moved).

- [ ] **Step 4: Commit**

```bash
git add puremacro/dsge/sw07_priors.py
git commit -m "refactor(dsge): sw07_priors delegates to puremacro.dsge.priors"
```

If Step 2 turned up any test file that needed adjustment, include it in this commit.

---

## Task 4: Rename SW07PosteriorResult → DSGEPosteriorResult + model_name field + alias

**Files:**
- Modify: `puremacro/dsge/_results.py`.
- Modify: `tests/test_dsge/test_sw07_wrapper.py` (append 2 tests).

- [ ] **Step 1: Append 2 tests**

Append to `tests/test_dsge/test_sw07_wrapper.py`:
```python
def test_sw07posteriorresult_is_alias_for_dsge():
    from puremacro.dsge._results import DSGEPosteriorResult, SW07PosteriorResult
    assert SW07PosteriorResult is DSGEPosteriorResult


def test_dsge_posterior_result_default_model_name_is_unknown():
    import numpy as np
    from puremacro.dsge._results import DSGEPosteriorResult
    res = DSGEPosteriorResult(
        draws=np.zeros((1, 5, 3)),
        param_names=("a", "b", "c"),
        log_posterior_trace=np.zeros((1, 5)),
        accept_rates=(0.25,),
        mode={"a": 0.0, "b": 0.0, "c": 0.0},
        mode_hessian_inv=np.eye(3),
        n_burn_in=0,
        data_n_obs=10,
        seed=0,
    )
    assert res.model_name == "unknown"
    res2 = DSGEPosteriorResult(
        draws=np.zeros((1, 5, 3)),
        param_names=("a", "b", "c"),
        log_posterior_trace=np.zeros((1, 5)),
        accept_rates=(0.25,),
        mode={"a": 0.0, "b": 0.0, "c": 0.0},
        mode_hessian_inv=np.eye(3),
        n_burn_in=0,
        data_n_obs=10,
        seed=0,
        model_name="MyModel",
    )
    assert res2.model_name == "MyModel"
```

- [ ] **Step 2: Run tests, expect FAIL**

```bash
pytest tests/test_dsge/test_sw07_wrapper.py -v -k "alias or default_model_name"
```

Expected: `ImportError` (`DSGEPosteriorResult` doesn't exist yet).

- [ ] **Step 3: Apply the rename + alias + new field**

Replace `puremacro/dsge/_results.py` contents:
```python
"""Frozen-dataclass result types for puremacro.dsge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DSGEPosteriorResult:
    """Result of puremacro.dsge.estimate_dsge (and model-specific wrappers
    like puremacro.dsge.estimate_sw07).

    Attributes
    ----------
    draws : ndarray, shape (n_chains, n_draws, n_params)
        Post-burn-in MCMC draws.
    param_names : tuple of str, length n_params
        Parameter names in column order matching `draws`.
    log_posterior_trace : ndarray, shape (n_chains, n_draws)
        Log-posterior at each retained draw.
    accept_rates : tuple of float, length n_chains
        Per-chain acceptance rate over the retained draws.
    mode : dict[str, float]
        Posterior mode (parameter name → value).
    mode_hessian_inv : ndarray, shape (n_params, n_params)
        Inverse Hessian at the mode when scipy.optimize converges + Hessian
        is PD; otherwise falls back to diag(prior_stds**2).
    n_burn_in : int
        Burn-in iterations dropped (also used as proposal-scale adaptation
        window).
    data_n_obs : int
        Number of observations in the input dataset.
    seed : int
        Master RNG seed.
    model_name : str, default 'unknown'
        Identifier for the underlying DSGE model. ``estimate_sw07`` sets
        this to ``"SW07"``; ``estimate_dsge`` callers can pass whatever
        string they want. New in 0.53.0.
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
        """Per-parameter mean, std, 5%/50%/95% quantiles across all chains."""
        flat = self.draws.reshape(-1, len(self.param_names))
        return pd.DataFrame({
            "mean":  flat.mean(axis=0),
            "std":   flat.std(axis=0),
            "q05":   np.quantile(flat, 0.05, axis=0),
            "q50":   np.quantile(flat, 0.50, axis=0),
            "q95":   np.quantile(flat, 0.95, axis=0),
            "mode":  np.array([self.mode[n] for n in self.param_names]),
        }, index=list(self.param_names))


# Backward-compatibility alias for code that imports SW07PosteriorResult.
# Resolves to the same class object; isinstance/pickle/type-hints continue
# to work. Drop in 1.0 if appropriate.
SW07PosteriorResult = DSGEPosteriorResult


__all__ = ["DSGEPosteriorResult", "SW07PosteriorResult"]
```

- [ ] **Step 4: Run all dsge tests, expect ALL PASS**

```bash
pytest tests/test_dsge/ -v -m "not slow"
```

Plus the parity test:
```bash
pytest tests/test_dsge/test_sw07_wrapper.py::test_sw07_parity_short_chain -v -m slow
```

Expected: everything passes. `SW07PosteriorResult` consumers continue to work via the alias.

- [ ] **Step 5: Commit**

```bash
git add puremacro/dsge/_results.py tests/test_dsge/test_sw07_wrapper.py
git commit -m "feat(dsge): rename SW07PosteriorResult to DSGEPosteriorResult (alias for BC)"
```

---

## Task 5: Create puremacro/dsge/estimate.py — generic estimator

**Files:**
- Create: `puremacro/dsge/estimate.py`.
- Create: `tests/test_dsge/test_estimate_dsge.py`.

This is the largest task. Move every generic helper out of `sw07_estimate.py` (which is done in Task 6) into a fresh module.

- [ ] **Step 1: Write 5 failing tests**

Create `tests/test_dsge/test_estimate_dsge.py`:
```python
"""Tests for puremacro.dsge.estimate_dsge — the generic Bayesian DSGE engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.state_space import StateSpaceModel


_AR1_PRIORS = {
    "rho":   {"dist": "beta",     "mean": 0.5, "std": 0.2, "lb": 0.001, "ub": 0.99},
    "sigma": {"dist": "invgamma", "mean": 0.1, "std": 2.0, "lb": 0.01,  "ub": 5.0},
}


def _ar1_state_space(params: dict) -> StateSpaceModel:
    """y_t = x_t,  x_t = rho * x_{t-1} + sigma * eps_t."""
    rho = params["rho"]
    sigma = params["sigma"]
    return StateSpaceModel(
        T=np.array([[rho]]),
        Z=np.array([[1.0]]),
        R=np.array([[1.0]]),
        Q=np.array([[sigma ** 2]]),
        H=np.array([[1e-8]]),
        c=np.zeros(1),
        d=np.zeros(1),
    )


def _simulate_ar1(rho_true: float, sigma_true: float, T: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T) * sigma_true
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho_true * x[t - 1] + eps[t]
    return pd.DataFrame({"y": x})


def test_estimate_dsge_returns_dsgeposteriorresult():
    from puremacro.dsge.estimate import estimate_dsge
    from puremacro.dsge._results import DSGEPosteriorResult
    data = _simulate_ar1(0.7, 0.5, T=200, seed=0)
    res = estimate_dsge(
        data, observation_eq=_ar1_state_space, priors=_AR1_PRIORS,
        observed_vars=["y"], initial_params={"rho": 0.5, "sigma": 0.4},
        n_chains=1, n_draws=300, burn_in=100, seed=0,
    )
    assert isinstance(res, DSGEPosteriorResult)
    assert res.draws.shape == (1, 300, 2)
    assert res.param_names == ("rho", "sigma")


def test_estimate_dsge_validates_missing_observed_var():
    from puremacro.dsge.estimate import estimate_dsge
    data = pd.DataFrame({"not_y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing columns"):
        estimate_dsge(
            data, observation_eq=_ar1_state_space, priors=_AR1_PRIORS,
            observed_vars=["y"], initial_params={"rho": 0.5, "sigma": 0.4},
            n_chains=1, n_draws=10, burn_in=5, seed=0,
        )


def test_estimate_dsge_toy_ar1_recovers_rho():
    from puremacro.dsge.estimate import estimate_dsge
    data = _simulate_ar1(rho_true=0.7, sigma_true=0.5, T=500, seed=0)
    res = estimate_dsge(
        data, observation_eq=_ar1_state_space, priors=_AR1_PRIORS,
        observed_vars=["y"], initial_params={"rho": 0.5, "sigma": 0.4},
        n_chains=1, n_draws=2000, burn_in=500, seed=0,
    )
    posterior_rho_mean = float(res.draws[0, :, 0].mean())
    assert abs(posterior_rho_mean - 0.7) < 0.1, (
        f"posterior mean rho={posterior_rho_mean:.3f} far from true 0.7"
    )


def test_estimate_dsge_model_name_field_set():
    from puremacro.dsge.estimate import estimate_dsge
    data = _simulate_ar1(0.5, 0.3, T=100, seed=1)
    res = estimate_dsge(
        data, observation_eq=_ar1_state_space, priors=_AR1_PRIORS,
        observed_vars=["y"], initial_params={"rho": 0.5, "sigma": 0.3},
        model_name="ToyAR1",
        n_chains=1, n_draws=200, burn_in=50, seed=0,
    )
    assert res.model_name == "ToyAR1"


def test_estimate_dsge_kalman_singular_returns_neg_inf_in_log_posterior():
    """A pathological observation_eq that raises LinAlgError should be
    caught and produce -inf log-posterior, not crash the MCMC."""
    from puremacro.dsge.estimate import _make_neg_log_posterior

    priors = {"x": {"dist": "normal", "mean": 0.0, "std": 1.0, "lb": -10, "ub": 10}}

    def bad_obs(params):
        raise np.linalg.LinAlgError("simulated singular")

    nlp = _make_neg_log_posterior(
        y=np.zeros((10, 1)),
        observation_eq=bad_obs,
        priors=priors,
        names=("x",),
        fixed_params={},
    )
    assert nlp(np.array([0.5])) == np.inf
```

- [ ] **Step 2: Run tests, expect FAIL**

```bash
pytest tests/test_dsge/test_estimate_dsge.py -v
```

Expected: `ImportError` on `puremacro.dsge.estimate`.

- [ ] **Step 3: Implement estimate.py**

Create `puremacro/dsge/estimate.py`:
```python
"""Generic Bayesian DSGE estimator via Random-Walk Metropolis-Hastings.

Model-agnostic. Mode refinement (scipy.optimize.minimize, L-BFGS-B) →
numerical Hessian → proposal cov c²·H⁻¹ (fallback diag(prior_stds²) if
H not PD) → multi-chain RW-MH via puremacro.mcmc.random_walk_metropolis.

This module hosts the helpers that were previously inlined in
``puremacro.dsge.sw07_estimate`` but contained nothing SW07-specific:
``_vec_to_dict``, ``_make_neg_log_posterior``, ``_initial_vec_from_dict``,
``_nearest_pd``, ``_find_finite_start``.
"""
from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize as _scipy_minimize

from puremacro.dsge._results import DSGEPosteriorResult
from puremacro.dsge.priors import (
    log_prior, prior_stds, param_bounds, param_names,
)
from puremacro.mcmc import random_walk_metropolis
from puremacro.numerics import numerical_hessian
from puremacro.state_space import StateSpaceModel, kalman_filter


def _vec_to_dict(
    vec: np.ndarray,
    names: Sequence[str],
    fixed_params: dict | None = None,
) -> dict:
    """Combine an estimated-param vector with fixed params into a single dict."""
    if len(vec) != len(names):
        raise ValueError(f"vec length {len(vec)} doesn't match {len(names)} names")
    out = dict(fixed_params or {})
    for nm, val in zip(names, vec):
        out[nm] = float(val)
    return out


def _make_neg_log_posterior(
    y: np.ndarray,
    observation_eq: Callable[[dict], StateSpaceModel],
    priors: dict,
    names: Sequence[str],
    fixed_params: dict | None,
):
    """Closure returning -log_posterior(vec). Returns +inf on numerical failure."""
    def neg_log_post(vec: np.ndarray) -> float:
        try:
            params = _vec_to_dict(vec, names, fixed_params)
        except ValueError:
            return np.inf
        lp = log_prior(params, priors)
        if not np.isfinite(lp):
            return np.inf
        try:
            ssm = observation_eq(params)
            out = kalman_filter(y, ssm)
            ll = out["loglik"]
        except (np.linalg.LinAlgError, ValueError, RuntimeError,
                ZeroDivisionError, FloatingPointError):
            return np.inf
        if not np.isfinite(ll):
            return np.inf
        return -(ll + lp)
    return neg_log_post


def _initial_vec_from_dict(
    initial_params: dict,
    priors: dict,
) -> np.ndarray:
    """Build the initial parameter vector from a dict, snapping out-of-bound
    values to lb + 1e-3 (well inside the support).
    """
    vec = []
    for name, spec in priors.items():
        val = initial_params.get(name)
        if val is None or not (spec["lb"] <= val <= spec["ub"]):
            val = spec["lb"] + 1e-3
        vec.append(val)
    return np.array(vec)


def _nearest_pd(A: np.ndarray) -> np.ndarray:
    """Nearest symmetric positive-definite matrix (Higham 2002, abridged)."""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    eps = np.finfo(float).eps
    I = np.eye(A.shape[0])
    k = 1
    while True:
        try:
            np.linalg.cholesky(A3)
            return A3
        except np.linalg.LinAlgError:
            mineig = np.min(np.linalg.eigvalsh(A3))
            A3 = A3 + I * (-mineig * k ** 2 + eps)
            k += 1
            if k > 100:
                raise RuntimeError("_nearest_pd did not converge")


def _find_finite_start(
    neg_log_post,
    rng: np.random.Generator,
    priors: dict,
    max_tries: int = 200,
) -> np.ndarray:
    """Random points inside the prior box until neg_log_post is finite."""
    bounds = param_bounds(priors)
    for _ in range(max_tries):
        vec = np.array([rng.uniform(lb, ub) for (lb, ub) in bounds])
        if np.isfinite(neg_log_post(vec)):
            return vec
    # Hard fallback: prior means clipped to interior.
    vec = np.array([
        np.clip(spec["mean"], spec["lb"] + 1e-6, spec["ub"] - 1e-6)
        for spec in priors.values()
    ])
    return vec


def estimate_dsge(
    data: pd.DataFrame,
    *,
    observation_eq: Callable[[dict], StateSpaceModel],
    priors: dict,
    observed_vars: Sequence[str],
    initial_params: dict,
    fixed_params: dict | None = None,
    model_name: str = "unknown",
    n_draws: int = 10_000,
    n_chains: int = 2,
    burn_in: int = 2_000,
    seed: int = 0,
) -> DSGEPosteriorResult:
    """Bayesian DSGE estimation via Random-Walk Metropolis-Hastings.

    Parameters
    ----------
    data : DataFrame with columns ``observed_vars``.
    observation_eq : pure callable params_dict → StateSpaceModel.
    priors : dict shaped ``{name: {dist, mean, std, lb, ub}}``.
    observed_vars : ordered list of column names from ``data`` to feed to
        the Kalman filter.
    initial_params : starting value dict; missing or out-of-bound entries
        are snapped to ``lb + 1e-3``.
    fixed_params : not-estimated params merged into every call to
        ``observation_eq``. Default ``None`` → empty dict.
    model_name : tag attached to the returned DSGEPosteriorResult.
    n_draws, n_chains, burn_in, seed : MCMC controls.

    Returns
    -------
    DSGEPosteriorResult
    """
    # 1. Validate data.
    missing = set(observed_vars) - set(data.columns)
    if missing:
        raise ValueError(f"data missing columns: {sorted(missing)}")
    if len(data) < 10:
        raise ValueError(f"data has only {len(data)} obs; need >= 10")
    if data[list(observed_vars)].isna().any().any():
        raise ValueError("data contains NaN in observed_vars")
    y = data[list(observed_vars)].to_numpy()

    # 2. Build neg_log_post + initial vec.
    names = param_names(priors)
    fixed = dict(fixed_params or {})
    neg_log_post = _make_neg_log_posterior(
        y, observation_eq, priors, names, fixed,
    )
    init_vec = _initial_vec_from_dict(initial_params, priors)

    # 3. Mode refinement.
    mode_vec = init_vec.copy()
    converged_mle = False
    try:
        opt = _scipy_minimize(
            neg_log_post, init_vec,
            method="L-BFGS-B",
            bounds=param_bounds(priors),
            options={"maxiter": 100, "maxfun": 500 * len(init_vec)},
        )
        if opt.success and np.isfinite(opt.fun):
            mode_vec = np.asarray(opt.x, dtype=float)
            converged_mle = True
        else:
            warnings.warn(
                "estimate_dsge: mode optimisation did not converge; using "
                "the snap-corrected initial_params as the mode.",
                UserWarning,
            )
    except Exception as e:
        warnings.warn(
            f"estimate_dsge: mode optimisation raised {type(e).__name__}; "
            f"using the snap-corrected initial_params as the mode.",
            UserWarning,
        )

    # 4. Hessian-based proposal cov (only when mode converged).
    use_hessian = False
    inv_H: np.ndarray
    if converged_mle:
        H = numerical_hessian(neg_log_post, mode_vec, h=1e-4)
        try:
            H_pd = _nearest_pd(H)
            inv_H = np.linalg.inv(H_pd)
            np.linalg.cholesky(inv_H)
            use_hessian = True
        except (np.linalg.LinAlgError, RuntimeError):
            warnings.warn(
                "estimate_dsge: Hessian non-PD even after _nearest_pd; "
                "falling back to diag(prior_stds**2).",
                UserWarning,
            )
    if not use_hessian:
        stds = np.array([prior_stds(priors)[n] for n in names])
        inv_H = np.diag(stds ** 2)

    # 5. Proposal scaling.
    n_params = len(names)
    c0 = 2.38 / np.sqrt(n_params) if use_hessian else 0.01
    proposal_cov = c0 ** 2 * inv_H

    # 6. Run chains.
    chains_arr = np.empty((n_chains, n_draws, n_params))
    log_post_arr = np.empty((n_chains, n_draws))
    accept_rates = []

    def log_post_fn(vec):
        return -neg_log_post(vec)

    for chain_idx in range(n_chains):
        rng = np.random.default_rng(seed + chain_idx)
        try:
            perturb = rng.multivariate_normal(np.zeros(n_params), 0.0025 * inv_H)
        except np.linalg.LinAlgError:
            perturb = 0.05 * rng.standard_normal(n_params)
        start = mode_vec + perturb
        if not np.isfinite(neg_log_post(start)):
            start = mode_vec.copy()
        if not np.isfinite(neg_log_post(start)):
            start = _find_finite_start(neg_log_post, rng, priors)

        out = random_walk_metropolis(
            log_post_fn, start, proposal_cov, n_draws=n_draws,
            seed=seed + chain_idx, accept_target=0.25, adapt_burnin=burn_in,
        )
        chains_arr[chain_idx] = out["chain"]
        log_post_arr[chain_idx] = out["log_post"]
        accept_rates.append(out["accept_rate"])

        if not (0.10 <= out["accept_rate"] <= 0.50):
            warnings.warn(
                f"estimate_dsge: chain {chain_idx} accept_rate="
                f"{out['accept_rate']:.3f} outside [0.10, 0.50]; "
                f"mixing may be poor.",
                UserWarning,
            )

    mode_dict = _vec_to_dict(mode_vec, names, fixed)

    return DSGEPosteriorResult(
        draws=chains_arr,
        param_names=names,
        log_posterior_trace=log_post_arr,
        accept_rates=tuple(accept_rates),
        mode=mode_dict,
        mode_hessian_inv=inv_H,
        n_burn_in=burn_in,
        data_n_obs=len(data),
        seed=seed,
        model_name=model_name,
    )


__all__ = ["estimate_dsge"]
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
pytest tests/test_dsge/test_estimate_dsge.py -v
```

Expected: 5/5 PASS.

The AR(1) recovery test (`test_estimate_dsge_toy_ar1_recovers_rho`) is statistical. If it fails on the default seed, try widening tolerance to 0.15 or boosting `n_draws` to 4000. Do NOT relax to "any number".

- [ ] **Step 5: Commit**

```bash
git add puremacro/dsge/estimate.py tests/test_dsge/test_estimate_dsge.py
git commit -m "feat(dsge): puremacro.dsge.estimate_dsge — generic Bayesian DSGE engine"
```

---

## Task 6: Refactor sw07_estimate.py to thin wrapper

**Files:**
- Modify: `puremacro/dsge/sw07_estimate.py`.

After Task 5, `sw07_estimate.py` has ~300 LOC, most of which is now duplicated in `estimate.py`. Strip everything that's not SW07-specific.

- [ ] **Step 1: Replace sw07_estimate.py with the thin wrapper**

Replace the entire contents of `puremacro/dsge/sw07_estimate.py` with:
```python
"""Thin wrapper: estimate Smets-Wouters (2007) via the generic
puremacro.dsge.estimate_dsge engine.

The model-specific bits — bundled-data loading, OBSERVED_VARS
validation, the fixed (non-estimated) calibrated parameters, and the
initial-params construction from SW07_POSTERIOR_MODE + SW07_SHOCK_STDS —
live here. Everything else (mode refinement, Hessian, proposal-cov
construction, MH chains) is in puremacro.dsge.estimate.
"""
from __future__ import annotations

import importlib.resources
from typing import Optional

import pandas as pd

from puremacro.dsge._results import DSGEPosteriorResult
from puremacro.dsge.estimate import estimate_dsge
from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.dsge.sw07_priors import PRIORS


# Calibrated (NOT estimated) SW07 parameters; merged into every observation_eq
# call. These appear in SW07_POSTERIOR_MODE but NOT in PRIORS.
_FIXED_PARAMS = {
    "ctou":     0.025,
    "clandaw":  1.5,
    "cg":       0.18,
    "curvp":    10.0,
    "curvw":    10.0,
}


def _load_bundled_data() -> pd.DataFrame:
    pkg = importlib.resources.files("puremacro.dsge")
    return pd.read_csv(
        pkg / "_sw07_data.csv",
        comment="#", parse_dates=["date"], index_col="date",
    )


def _validate_data(df: pd.DataFrame) -> None:
    missing = set(OBSERVED_VARS) - set(df.columns)
    if missing:
        raise ValueError(f"data missing columns: {sorted(missing)}")
    if len(df) < 50:
        raise ValueError(f"data has only {len(df)} obs; need >= 50")
    if df[list(OBSERVED_VARS)].isna().any().any():
        raise ValueError("data contains NaN values")


def estimate_sw07(
    data: Optional[pd.DataFrame] = None,
    *,
    n_draws: int = 10_000,
    n_chains: int = 2,
    burn_in: int = 2_000,
    seed: int = 0,
) -> DSGEPosteriorResult:
    """Bayesian estimation of Smets-Wouters (2007) via Random-Walk MH.

    Thin wrapper over :func:`puremacro.dsge.estimate.estimate_dsge`.

    Parameters
    ----------
    data : DataFrame with columns OBSERVED_VARS; if None, loads the
        bundled 1966Q1-2004Q4 US dataset (156 quarterly obs × 7 cols).
    n_draws, n_chains, burn_in, seed : MCMC controls.
    """
    df = _load_bundled_data() if data is None else data.copy()
    _validate_data(df)
    initial_params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    return estimate_dsge(
        df,
        observation_eq=make_state_space,
        priors=PRIORS,
        observed_vars=list(OBSERVED_VARS),
        initial_params=initial_params,
        fixed_params=_FIXED_PARAMS,
        model_name="SW07",
        n_draws=n_draws, n_chains=n_chains, burn_in=burn_in, seed=seed,
    )


__all__ = ["estimate_sw07"]
```

This deletes: `_vec_to_dict`, `_make_neg_log_posterior`, `_initial_vec`, `_nearest_pd`, `_find_finite_start`, the inlined mode-refinement / Hessian / MH driver block, and all of the now-unused imports (`numpy as np`, `warnings`, `scipy.optimize.minimize`, `random_walk_metropolis`, `numerical_hessian`, `kalman_filter`, `SW07PosteriorResult`).

- [ ] **Step 2: Run the parity test, expect PASS**

```bash
pytest tests/test_dsge/test_sw07_wrapper.py::test_sw07_parity_short_chain -v -m slow
```

Expected: PASS. The byte-for-byte snapshot from Task 1 must match the post-refactor output.

**If this fails**: investigate. The refactor introduced a numerical change. Common culprits:
- A helper was moved with a typo (compare `_make_neg_log_posterior` in `estimate.py` to the original line-by-line).
- Argument ordering of `kalman_filter(y, ssm)` flipped.
- Default seed propagation through `random_walk_metropolis` differs.

Do NOT relax the parity test tolerance. Find and fix the actual divergence.

- [ ] **Step 3: Run all dsge tests**

```bash
pytest tests/test_dsge/ -v
```

Plus the slow tests:
```bash
pytest tests/test_dsge/ -v -m slow
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add puremacro/dsge/sw07_estimate.py
git commit -m "refactor(dsge): sw07_estimate is now a thin wrapper over estimate_dsge"
```

---

## Task 7: Create AR(1) example demo

**Files:**
- Create: `puremacro/examples/dsge_ar1_demo.py`.

The examples gallery auto-discovers `puremacro/examples/*.py`, so no further registration is needed.

- [ ] **Step 1: Create the demo**

Create `puremacro/examples/dsge_ar1_demo.py`:
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
        H=np.array([[1e-8]]),
        c=np.zeros(1),
        d=np.zeros(1),
    )


def _simulate(rho_true: float, sigma_true: float, T: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T) * sigma_true
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = rho_true * x[t - 1] + eps[t]
    return pd.DataFrame({"y": x})


def main() -> None:
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

- [ ] **Step 2: Run the example**

```bash
python puremacro/examples/dsge_ar1_demo.py
```

Expected: prints a summary DataFrame followed by two "true / posterior mean" lines. The posterior mean for ρ should be within ±0.1 of 0.7; σ within ±0.1 of 0.5. Runtime ~3-5s.

- [ ] **Step 3: Commit**

```bash
git add puremacro/examples/dsge_ar1_demo.py
git commit -m "feat(examples): dsge_ar1_demo — proves estimate_dsge is model-agnostic"
```

---

## Task 8: Wire exports in puremacro/dsge/__init__.py

**Files:**
- Modify: `puremacro/dsge/__init__.py`.

- [ ] **Step 1: Update the package `__init__.py`**

Replace `puremacro/dsge/__init__.py` with:
```python
"""DSGE primitives for puremacro.

Includes:
- Klein (2000) QZ solver for linear rational-expectations models.
- Sims (2002) gensys solver (equivalent, model-agnostic input form).
- Bayesian estimation engine (random-walk Metropolis-Hastings) +
  model-agnostic priors framework.
- Smets-Wouters (2007) reference model + bundled US dataset.

For likelihood-based estimation, pair the state-space form returned by
``make_state_space`` (model-specific) with ``puremacro.dsge.estimate_dsge``.
"""
from .klein import BlanchardKahnError, KleinSolution, klein_solve
from ._results import DSGEPosteriorResult, SW07PosteriorResult
from .estimate import estimate_dsge
from .sw07_estimate import estimate_sw07
from . import priors

__all__ = [
    "klein_solve",
    "KleinSolution",
    "BlanchardKahnError",
    "DSGEPosteriorResult",
    "SW07PosteriorResult",
    "estimate_dsge",
    "estimate_sw07",
    "priors",
]
from . import smets_wouters  # re-export for back-compat with 0.50.0 callers
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from puremacro.dsge import estimate_dsge, DSGEPosteriorResult, SW07PosteriorResult, estimate_sw07, priors; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Run the full dsge test suite**

```bash
pytest tests/test_dsge/ -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add puremacro/dsge/__init__.py
git commit -m "feat(dsge): export estimate_dsge + DSGEPosteriorResult + priors module"
```

---

## Task 9: Version bump + CHANGELOG

**Files:**
- Modify: `puremacro/__init__.py`.
- Modify: `pyproject.toml`.
- Modify: `CHANGELOG.md`.
- Modify: `tests/test_import.py`.

- [ ] **Step 1: Bump `puremacro/__init__.py`**

Change `__version__ = "0.52.0"` to `__version__ = "0.53.0"`.

- [ ] **Step 2: Bump `pyproject.toml`**

In the `[project]` block, change `version = "0.52.0"` to `version = "0.53.0"`.

- [ ] **Step 3: Bump `tests/test_import.py`**

Change `assert puremacro.__version__ == "0.52.0"` to `assert puremacro.__version__ == "0.53.0"`.

- [ ] **Step 4: Add CHANGELOG entry**

Insert into `CHANGELOG.md` AFTER the `# Changelog` heading + preamble and BEFORE the existing `## 0.52.0 — 2026-05-23` entry:

```markdown
## 0.53.0 — 2026-05-23

Generic Bayesian DSGE engine. R1a from the 2026-05-23 research-directions
brainstorm: extracted from the SW07-specific Bayesian estimator shipped
in 0.50.0 a model-agnostic ``estimate_dsge`` function + a generic prior
framework at ``puremacro.dsge.priors``. ``estimate_sw07`` is now a thin
wrapper (~60 LOC) over the generic engine. R1b (fertility DSGE port)
follows in 0.54.0.

### Added
- `puremacro.dsge.estimate_dsge(data, *, observation_eq, priors,
  observed_vars, initial_params, fixed_params, model_name, n_draws,
  n_chains, burn_in, seed) -> DSGEPosteriorResult` — generic Bayesian
  DSGE estimator. Same pipeline as the 0.50.0 ``estimate_sw07``
  (L-BFGS-B mode refinement → numerical Hessian → c²·H⁻¹ proposal
  cov with diag(prior_stds²) fallback → multi-chain RW-MH) but
  accepts an arbitrary state-space-building callable + priors dict.
- `puremacro.dsge.priors` submodule with `log_prior(params, priors)`,
  `prior_means`, `prior_stds`, `param_bounds`, `param_names` — all
  model-agnostic (take a priors dict as the second argument). The
  dist-specific `_logpdf_{beta, gamma, normal, invgamma}` helpers
  move here from `sw07_priors`.
- `puremacro.dsge.DSGEPosteriorResult` (the existing dataclass, renamed
  from `SW07PosteriorResult`) gains an optional `model_name: str`
  field (default `"unknown"`; `estimate_sw07` sets `"SW07"`).
- `puremacro/examples/dsge_ar1_demo.py` — toy AR(1) state-space
  estimated via `estimate_dsge`. Proves the engine is model-agnostic.

### Changed
- `puremacro.dsge.sw07_estimate` is now a ~60-LOC thin wrapper over
  `estimate_dsge`. All mode-refinement / Hessian / MH-driver logic
  moved to `puremacro.dsge.estimate`. Numerical parity is verified by
  a frozen golden-snapshot test (``tests/test_dsge/test_sw07_wrapper.py
  ::test_sw07_parity_short_chain``).
- `puremacro.dsge.sw07_priors` public helpers are now 3-line delegators
  to the generic `puremacro.dsge.priors` API. The dist-specific
  `_logpdf_*` helpers were moved out (no longer importable from
  `sw07_priors`).

### Backward compatibility
- `puremacro.dsge.SW07PosteriorResult` remains importable (aliased to
  `DSGEPosteriorResult`). `isinstance`, pickle, and type-hint use
  cases continue to work.
- `estimate_sw07(...)` keeps the same call signature and returns the
  same dataclass (under the new name + alias).
- No public symbols removed.
```

- [ ] **Step 5: Smoke check the version bump**

```bash
python -c "import puremacro; assert puremacro.__version__ == '0.53.0'; print(puremacro.__version__)"
```

Expected output: `0.53.0`.

- [ ] **Step 6: Commit**

```bash
git add puremacro/__init__.py pyproject.toml CHANGELOG.md tests/test_import.py
git commit -m "chore(puremacro): bump 0.52.0 → 0.53.0 (generic Bayesian DSGE engine)"
```

---

## Task 10: Regenerate the public-API snapshot

**Files:**
- Modify: `tests/fixtures/public_api_snapshot.json`.

- [ ] **Step 1: Run the snapshot test to see the gaps**

```bash
pytest tests/ -k "public_api" -v 2>&1 | tail -30
```

Expected: FAIL — the snapshot is missing `puremacro.dsge.priors` and `puremacro.dsge.estimate`, plus the additions to `puremacro.dsge.__init__`.

- [ ] **Step 2: Regenerate**

Inspect the snapshot test (`tests/test_public_api.py` or similar) for a regeneration helper. If there is one, run it:

```bash
grep -rln "collect_current_api\|test_public_api_matches_snapshot" tests/ tools/ | head -3
```

If no helper exists, manually update `tests/fixtures/public_api_snapshot.json` to add:
- `puremacro.dsge` → add `"DSGEPosteriorResult"`, `"estimate_dsge"`, `"priors"` to its export list.
- `puremacro.dsge.estimate` → new module entry: `["estimate_dsge"]`.
- `puremacro.dsge.priors` → new module entry: `["log_prior", "param_bounds", "param_names", "prior_means", "prior_stds"]`.
- `puremacro.dsge.sw07_priors` → drop the `_logpdf_*` private helpers if the snapshot tracked them (Task 3 removed them).

Maintain alphabetical sort within each module's list.

- [ ] **Step 3: Re-run the snapshot test**

```bash
pytest tests/ -k "public_api" -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/public_api_snapshot.json
git commit -m "chore(tests): regenerate public_api_snapshot for 0.53.0 dsge additions"
```

---

## Task 11: Run the 6-gate release check

**Files:** none modified — verification only.

- [ ] **Step 1: Run gates 1-4**

```bash
python tools/release_check.py
```

Expected: `all 4 gates PASS`.

- [ ] **Step 2: Run gate 5 (examples gallery)**

```bash
python tools/release_check.py --examples
```

Expected: gate 5 PASS. The new `dsge_ar1_demo.py` should be picked up.

If gate 5 emits a "stale gallery" advisory, regenerate the gallery:
```bash
python tools/render_examples_gallery.py
```
Restore any flaky examples (e.g., `hfi_gertler_karadi`) from the previous PASS entry; see commit `e569dee` from the 0.51.0 release for the pattern.

- [ ] **Step 3: Run gate 6 (Pyodide smoke)**

```bash
python tools/release_check.py --pyodide
```

Expected: gate 6 PASS.

- [ ] **Step 4: Final integrated 6-gate check**

```bash
python tools/release_check.py --examples --pyodide
```

Expected: `all 6 gates PASS`.

If any gate fails, diagnose the underlying issue (no `--no-verify`, no hook skips).

---

## Self-review checklist (run AFTER all 11 tasks)

1. **Spec coverage:**
   - Component A (`priors.py`) — Task 2 ✓
   - Component B (`_results.py` rename + alias + model_name) — Task 4 ✓
   - Component C (`estimate.py`) — Task 5 ✓
   - Component D (`sw07_estimate.py` wrapper refactor) — Task 6 ✓
   - Component E (`sw07_priors.py` delegator refactor) — Task 3 ✓
   - Component F (AR(1) demo) — Task 7 ✓
   - Golden-snapshot parity test (criterion 5) — Tasks 1 + 6 ✓
   - Public exports + snapshot regen — Tasks 8 + 10 ✓
   - Version + CHANGELOG — Task 9 ✓
   - Release gates — Task 11 ✓
   - All 11 acceptance criteria map to a task.

2. **Placeholder scan:** None — every step contains runnable code or a concrete command.

3. **Type consistency:**
   - `priors` dict shape `{name: {dist, mean, std, lb, ub}}` is consistent across Tasks 2, 3, 5, 7.
   - `_logpdf_*` signatures (`x, mean, std`) consistent between Tasks 2 and 3 (Task 3 deletes them from `sw07_priors`).
   - `estimate_dsge` signature consistent between Task 5 (implementation), Task 6 (call site in `estimate_sw07`), Task 7 (call site in demo), and Task 5 tests.
   - `_make_neg_log_posterior(y, observation_eq, priors, names, fixed_params)` signature consistent between Task 5 implementation and Task 5 test (`test_estimate_dsge_kalman_singular_returns_neg_inf_in_log_posterior`).
   - `kalman_filter(y, ssm)` call order (positional y first, then ssm) used consistently in Task 5.
   - `DSGEPosteriorResult` field set: 10 existing + `model_name: str = "unknown"` — consistent in Task 4 dataclass def, Task 5 return, Task 6 wrapper return.
