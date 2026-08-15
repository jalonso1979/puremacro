# puremacro 0.50.0 — Bayesian SW07 Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `puremacro.dsge.estimate_sw07(data, n_draws, n_chains, burn_in, seed) -> SW07PosteriorResult` — a Bayesian Random-Walk Metropolis-Hastings estimator for the Smets-Wouters (2007) DSGE model, tying together the existing `state_space.kalman_filter`, `numerics.mle_fit`/`numerical_hessian`, and `mcmc.geweke_z`/`gelman_rubin` machinery via a new sampler (`mcmc.random_walk_metropolis`) and SW07-specific priors + observation equation. Tag as **0.50.0**, ticking nothing new in `docs/1.0_path.md` § 4 (this is research capability, not a 1.0 gate).

**Architecture:** Single driver function. Four building blocks: (1) declarative `PRIORS` dict + `log_prior(params)` matching SW07 Table 1A; (2) `make_state_space(params) -> StateSpaceModel` wraps `solve_sw07` + observation equation; (3) `random_walk_metropolis` sampler with optional scalar-c adaptation during burn-in; (4) `estimate_sw07` driver: load data → refine posterior mode (`mle_fit`) → compute Hessian → scale proposal → run chains → return result dataclass.

**Tech Stack:** Python ≥3.10, numpy/scipy/pandas, scipy.stats for prior log-densities, pytest. Pure-numpy + scipy throughout (Pyodide-safe in principle, though the slow replication test is opt-in and not added to Gate 6).

**Source spec:** `docs/specs/2026-05-23-puremacro-050-bayesian-dsge-design.md` (commit `75a5387`).

**Pre-execution state (HEAD `75a5387`, on `feature/subnational-labor-uncertainty-us`):**
- 0.49.0 shipped at tag `v0.49.0`. 6 gates green.
- `puremacro.dsge.smets_wouters` exports `solve_sw07`, `SWResult`, `SW07_POSTERIOR_MODE` (36 structural params), `SW07_SHOCK_STDS` (7 shock σ's).
- `puremacro.state_space` exports `StateSpaceModel(T, Z, Q, H, R=None, c=None, d=None)` and `kalman_filter(y, model, ...) -> {"loglik": float, ...}`.
- `puremacro.numerics` exports `mle_fit(neg_loglik, theta_init, *, bounds, method="L-BFGS-B") -> dict` and `numerical_hessian(f, x, h=1e-4) -> ndarray`.
- `puremacro.mcmc` exports diagnostics only — no sampler yet. This task adds `random_walk_metropolis`.
- `puremacro/dsge/_references/sw07_pfeifer.mod` is the canonical SW07 model reference (committed at 0.45.0).

---

## File structure

**Created:**
- `puremacro/puremacro/dsge/sw07_priors.py` — `PRIORS` dict + `log_prior`, `prior_means`, `prior_stds`, `param_bounds` helpers.
- `puremacro/puremacro/dsge/sw07_observation.py` — `make_state_space(params) -> StateSpaceModel`.
- `puremacro/puremacro/dsge/sw07_estimate.py` — `estimate_sw07` driver + module-level helpers.
- `puremacro/puremacro/dsge/_results.py` (new file in the `dsge/` package) — `SW07PosteriorResult` frozen dataclass.
- `puremacro/puremacro/dsge/_sw07_data.csv` — bundled US 7-variable dataset, 1966Q1–2004Q4 (155 obs × 7 columns + header comments).
- `puremacro/tools/build_sw07_data.py` — one-time script that fetches + transforms the SW07 series from FRED and writes the CSV. Lives in `tools/`, not shipped in the wheel.
- `puremacro/tests/test_dsge/test_sw07_priors.py` — unit tests for `log_prior`.
- `puremacro/tests/test_dsge/test_sw07_observation.py` — unit tests for `make_state_space`.
- `puremacro/tests/test_random_walk_metropolis.py` — unit tests for the sampler (in outer `tests/` since it lives in top-level `puremacro/mcmc.py`).
- `puremacro/tests/test_dsge/test_sw07_estimate_smoke.py` — fast smoke integration test.
- `puremacro/tests/test_dsge/test_sw07_estimate_replication.py` — slow `@pytest.mark.slow` replication test.

**Modified:**
- `puremacro/puremacro/mcmc.py` — add `random_walk_metropolis` function + extend `__all__`.
- `puremacro/puremacro/dsge/__init__.py` — export `estimate_sw07`, `SW07PosteriorResult`.
- `puremacro/pyproject.toml` — declare `slow` marker.
- `puremacro/CHANGELOG.md` — 0.50.0 entry.
- `puremacro/CONTRIBUTING.md` — document `pytest -m slow` opt-in.
- `puremacro/__init__.py`, `puremacro/pyproject.toml`, `puremacro/tests/test_import.py` — version bump.
- `puremacro/tests/fixtures/public_api_snapshot.json` — regenerated for new public symbols.

**Untouched:**
- `puremacro/puremacro/dsge/smets_wouters.py` — `solve_sw07` is consumed as-is.
- `puremacro/puremacro/dsge/klein.py` — Klein solver consumed as-is.
- `puremacro/puremacro/state_space.py` — `kalman_filter` + `StateSpaceModel` consumed as-is.
- `puremacro/puremacro/numerics.py` — `mle_fit` + `numerical_hessian` consumed as-is.
- The existing 8 `pyodide_smoke`-marked tests — no changes.

---

## Working-directory convention

All paths relative to the **repo root**:

`/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/`

Subproject at `puremacro/`. Package source at `puremacro/puremacro/`. Tests at `puremacro/tests/`. Docs at `puremacro/docs/`. Tools at `puremacro/tools/`.

---

## Task 0: Pre-flight + branch creation

- [ ] **Step 1: Verify clean state**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git status --short puremacro/ | grep -v "\.png$\|\.csv$\|^?? puremacro/build/" | head -10
```

Expected: empty (or only the pre-existing `puremacro/data/` untracked entry).

- [ ] **Step 2: Confirm HEAD on the post-0.49.0 spec commit**

```bash
git log --oneline -1
git branch --show-current
```

Expected: `75a5387 docs(0.50.0): spec — Bayesian estimation of SW07 (estimate_sw07)` on `feature/subnational-labor-uncertainty-us`.

- [ ] **Step 3: Confirm baseline gate is green (4-gate fast run)**

```bash
python puremacro/tools/release_check.py
```

Expected: 4 gates PASS at 0.49.0.

- [ ] **Step 4: Verify scipy.stats has the prior families**

```bash
python -c "
from scipy.stats import beta, gamma, norm, invgamma
print('all 4 prior families importable')
print('beta logpdf at 0.5, a=2, b=2:', beta.logpdf(0.5, 2, 2))
print('invgamma logpdf at 0.5, a=2:', invgamma.logpdf(0.5, 2))
"
```

Expected: prints the line "all 4 prior families importable" followed by two finite numbers. If any import fails, scipy version is too old.

- [ ] **Step 5: Create the release branch**

```bash
git checkout -b release/0.50.0
git branch --show-current
```

Expected: `release/0.50.0`.

- [ ] **Step 6: No commit** — Task 0 is verification only.

---

## Task 1: SW07 priors module + tests

This task encodes the SW07 Table 1A priors as a declarative dict + a `log_prior` function. Reference: `puremacro/puremacro/dsge/_references/sw07_pfeifer.mod` (the Pfeifer port committed at 0.45.0). Pfeifer's `.mod` file has a `priors` block enumerating each estimated parameter's distribution, mean, and std.

The implementer reads `sw07_pfeifer.mod` once at the start of this task and translates the priors block. The **canonical set of ~36 estimated parameters is whatever Pfeifer enumerates** — do not invent a different set.

- [ ] **Step 1: Inventory the estimated parameters**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
grep -nE "^(estimated_params|prior|csig|crho|cprob|cind|csadj|chabb|crpi|crr|cry|crdy|cmap|cmaw|constepinf|constebeta|ctrend|cgy|czcap|cfc)" puremacro/dsge/_references/sw07_pfeifer.mod | head -80
```

This produces the list of priors. Build a Python dict from each entry. Approximate set (the implementer verifies against the actual file):

```python
PRIORS = {
    # Structural
    "csigma":     {"dist": "normal",   "mean": 1.50,  "std": 0.375, "lb": 0.25, "ub": 3.0},
    "csadjcost":  {"dist": "normal",   "mean": 4.0,   "std": 1.5,   "lb": 0.0,  "ub": 15.0},
    "chabb":      {"dist": "beta",     "mean": 0.7,   "std": 0.1,   "lb": 0.0,  "ub": 1.0},
    "cprobw":     {"dist": "beta",     "mean": 0.5,   "std": 0.1,   "lb": 0.0,  "ub": 1.0},
    "csigl":      {"dist": "normal",   "mean": 2.0,   "std": 0.75,  "lb": 0.25, "ub": 10.0},
    "cprobp":     {"dist": "beta",     "mean": 0.5,   "std": 0.1,   "lb": 0.0,  "ub": 1.0},
    "cindw":      {"dist": "beta",     "mean": 0.5,   "std": 0.15,  "lb": 0.0,  "ub": 1.0},
    "cindp":      {"dist": "beta",     "mean": 0.5,   "std": 0.15,  "lb": 0.0,  "ub": 1.0},
    "czcap":      {"dist": "beta",     "mean": 0.5,   "std": 0.15,  "lb": 0.0,  "ub": 1.0},
    "cfc":        {"dist": "normal",   "mean": 1.25,  "std": 0.125, "lb": 1.0,  "ub": 3.0},
    # Taylor rule
    "crpi":       {"dist": "normal",   "mean": 1.5,   "std": 0.25,  "lb": 1.0,  "ub": 3.0},
    "crr":        {"dist": "beta",     "mean": 0.75,  "std": 0.10,  "lb": 0.0,  "ub": 1.0},
    "cry":        {"dist": "normal",   "mean": 0.125, "std": 0.05,  "lb": 0.0,  "ub": 0.5},
    "crdy":       {"dist": "normal",   "mean": 0.125, "std": 0.05,  "lb": 0.0,  "ub": 0.5},
    # Constants
    "constepinf": {"dist": "gamma",    "mean": 0.625, "std": 0.10,  "lb": 0.1,  "ub": 2.0},
    "constebeta": {"dist": "gamma",    "mean": 0.25,  "std": 0.10,  "lb": 0.01, "ub": 2.0},
    "constelab":  {"dist": "normal",   "mean": 0.0,   "std": 2.0,   "lb": -10., "ub": 10.0},
    "ctrend":     {"dist": "normal",   "mean": 0.4,   "std": 0.10,  "lb": 0.1,  "ub": 0.8},
    # Output-gap response to gov spending
    "cgy":        {"dist": "normal",   "mean": 0.5,   "std": 0.25,  "lb": 0.0,  "ub": 2.0},
    # Shock persistence (AR(1))
    "crhoa":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhob":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhog":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhoqs":     {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhoms":     {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhopinf":   {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhow":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    # Shock MA terms
    "cmap":       {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "cmaw":       {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    # Shock std devs
    "ea":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "eb":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "eg":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "eqs":        {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "em":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "epinf":      {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "ew":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
}
```

Verify the actual SW07 Table 1A means/stds against `_references/sw07_pfeifer.mod`. If Pfeifer's values differ from the sketch above, use Pfeifer's — the `.mod` file is the authoritative reference.

- [ ] **Step 2: Write the failing tests**

Create `puremacro/tests/test_dsge/test_sw07_priors.py`:

```python
"""Tests for puremacro.dsge.sw07_priors."""
import math

import numpy as np
import pytest


def test_priors_dict_has_expected_size():
    from puremacro.dsge.sw07_priors import PRIORS
    # 36 ± 1 estimated parameters per SW07 Table 1A (exact count depends on the
    # Pfeifer reference; allow a small range to absorb minor spec variants).
    assert 30 <= len(PRIORS) <= 40


def test_log_prior_finite_at_prior_means():
    """At the prior-mean parameter vector, log_prior must be finite."""
    from puremacro.dsge.sw07_priors import PRIORS, log_prior
    means = {name: spec["mean"] for name, spec in PRIORS.items()}
    val = log_prior(means)
    assert math.isfinite(val)


def test_log_prior_minus_inf_out_of_support_beta():
    """A beta parameter set outside (0, 1) returns -inf."""
    from puremacro.dsge.sw07_priors import PRIORS, log_prior
    # Find a beta-distributed parameter.
    beta_param = next(name for name, spec in PRIORS.items() if spec["dist"] == "beta")
    means = {name: spec["mean"] for name, spec in PRIORS.items()}
    means[beta_param] = 1.5  # outside (0, 1)
    val = log_prior(means)
    assert val == -math.inf


def test_log_prior_minus_inf_out_of_support_invgamma():
    """A negative invgamma parameter returns -inf."""
    from puremacro.dsge.sw07_priors import PRIORS, log_prior
    invgamma_param = next(name for name, spec in PRIORS.items() if spec["dist"] == "invgamma")
    means = {name: spec["mean"] for name, spec in PRIORS.items()}
    means[invgamma_param] = -1.0  # negative ⇒ outside support
    val = log_prior(means)
    assert val == -math.inf


def test_log_prior_density_matches_scipy_beta():
    """Spot-check one beta prior against scipy.stats.beta.logpdf."""
    from scipy.stats import beta as beta_dist
    from puremacro.dsge.sw07_priors import PRIORS, _logpdf_beta  # see implementation

    name = next(n for n, spec in PRIORS.items() if spec["dist"] == "beta")
    spec = PRIORS[name]
    x = 0.4
    # Beta param-to-shape conversion: a = mean*(mean*(1-mean)/std**2 - 1), b = a*(1-mean)/mean
    a = spec["mean"] * (spec["mean"] * (1 - spec["mean"]) / spec["std"]**2 - 1)
    b = a * (1 - spec["mean"]) / spec["mean"]
    expected = beta_dist.logpdf(x, a, b)
    actual = _logpdf_beta(x, spec["mean"], spec["std"])
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_log_prior_sums_components():
    """log_prior(params) = sum of individual logpdfs."""
    from puremacro.dsge.sw07_priors import PRIORS, log_prior, _logpdf_for_spec

    means = {name: spec["mean"] for name, spec in PRIORS.items()}
    total = log_prior(means)
    manual = sum(_logpdf_for_spec(spec, means[name]) for name, spec in PRIORS.items())
    assert math.isclose(total, manual, rel_tol=1e-12)


def test_prior_means_returns_dict():
    from puremacro.dsge.sw07_priors import PRIORS, prior_means
    out = prior_means()
    assert set(out.keys()) == set(PRIORS.keys())
    for k in out:
        assert out[k] == PRIORS[k]["mean"]


def test_param_bounds_returns_list_of_tuples():
    """param_bounds returns a list of (lb, ub) tuples ordered by PRIORS dict insertion order, for L-BFGS-B."""
    from puremacro.dsge.sw07_priors import PRIORS, param_bounds
    bounds = param_bounds()
    assert len(bounds) == len(PRIORS)
    for (lb, ub), (name, spec) in zip(bounds, PRIORS.items()):
        assert lb == spec["lb"]
        assert ub == spec["ub"]
```

- [ ] **Step 3: Run tests — should fail (module missing)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/test_dsge/test_sw07_priors.py -v 2>&1 | tail -5
```

Expected: collection error (`puremacro.dsge.sw07_priors` missing) or 8 FAILED.

- [ ] **Step 4: Write `puremacro/puremacro/dsge/sw07_priors.py`**

```python
"""Smets-Wouters (2007) priors for Bayesian estimation.

Mirrors the priors block in puremacro/dsge/_references/sw07_pfeifer.mod.
The PRIORS dict is the canonical source; log_prior(params) sums the
individual scipy.stats logpdfs across all entries, returning -inf if
any parameter is outside its declared [lb, ub] support.

For beta and gamma priors we use the "mean / std" parameterisation
(as in Pfeifer's .mod file) and convert to scipy's shape-rate
parameterisation inside _logpdf_*.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


# === Declarative priors (Pfeifer mirror) =====================================

PRIORS: dict[str, dict] = {
    # [see Step 1 — copy the dict you constructed there]
    # Structural
    "csigma":     {"dist": "normal",   "mean": 1.50,  "std": 0.375, "lb": 0.25, "ub": 3.0},
    "csadjcost":  {"dist": "normal",   "mean": 4.0,   "std": 1.5,   "lb": 0.0,  "ub": 15.0},
    "chabb":      {"dist": "beta",     "mean": 0.7,   "std": 0.1,   "lb": 0.0,  "ub": 1.0},
    "cprobw":     {"dist": "beta",     "mean": 0.5,   "std": 0.1,   "lb": 0.0,  "ub": 1.0},
    "csigl":      {"dist": "normal",   "mean": 2.0,   "std": 0.75,  "lb": 0.25, "ub": 10.0},
    "cprobp":     {"dist": "beta",     "mean": 0.5,   "std": 0.1,   "lb": 0.0,  "ub": 1.0},
    "cindw":      {"dist": "beta",     "mean": 0.5,   "std": 0.15,  "lb": 0.0,  "ub": 1.0},
    "cindp":      {"dist": "beta",     "mean": 0.5,   "std": 0.15,  "lb": 0.0,  "ub": 1.0},
    "czcap":      {"dist": "beta",     "mean": 0.5,   "std": 0.15,  "lb": 0.0,  "ub": 1.0},
    "cfc":        {"dist": "normal",   "mean": 1.25,  "std": 0.125, "lb": 1.0,  "ub": 3.0},
    "crpi":       {"dist": "normal",   "mean": 1.5,   "std": 0.25,  "lb": 1.0,  "ub": 3.0},
    "crr":        {"dist": "beta",     "mean": 0.75,  "std": 0.10,  "lb": 0.0,  "ub": 1.0},
    "cry":        {"dist": "normal",   "mean": 0.125, "std": 0.05,  "lb": 0.0,  "ub": 0.5},
    "crdy":       {"dist": "normal",   "mean": 0.125, "std": 0.05,  "lb": 0.0,  "ub": 0.5},
    "constepinf": {"dist": "gamma",    "mean": 0.625, "std": 0.10,  "lb": 0.1,  "ub": 2.0},
    "constebeta": {"dist": "gamma",    "mean": 0.25,  "std": 0.10,  "lb": 0.01, "ub": 2.0},
    "constelab":  {"dist": "normal",   "mean": 0.0,   "std": 2.0,   "lb": -10., "ub": 10.0},
    "ctrend":     {"dist": "normal",   "mean": 0.4,   "std": 0.10,  "lb": 0.1,  "ub": 0.8},
    "cgy":        {"dist": "normal",   "mean": 0.5,   "std": 0.25,  "lb": 0.0,  "ub": 2.0},
    "crhoa":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhob":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhog":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhoqs":     {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhoms":     {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhopinf":   {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "crhow":      {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "cmap":       {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "cmaw":       {"dist": "beta",     "mean": 0.5,   "std": 0.20,  "lb": 0.0,  "ub": 0.9999},
    "ea":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "eb":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "eg":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "eqs":        {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "em":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "epinf":      {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
    "ew":         {"dist": "invgamma", "mean": 0.1,   "std": 2.0,   "lb": 0.01, "ub": 5.0},
}


# === Per-distribution log-density helpers ====================================

def _logpdf_beta(x: float, mean: float, std: float) -> float:
    """log-pdf of Beta(a, b) parameterised by mean/std (Pfeifer convention)."""
    if not (0.0 < x < 1.0):
        return -math.inf
    var = std ** 2
    a = mean * (mean * (1 - mean) / var - 1)
    b = a * (1 - mean) / mean
    return float(stats.beta.logpdf(x, a, b))


def _logpdf_gamma(x: float, mean: float, std: float) -> float:
    """log-pdf of Gamma parameterised by mean/std."""
    if x <= 0.0:
        return -math.inf
    # Method of moments: shape k = (mean/std)**2; scale θ = std**2 / mean
    k = (mean / std) ** 2
    theta = std ** 2 / mean
    return float(stats.gamma.logpdf(x, a=k, scale=theta))


def _logpdf_normal(x: float, mean: float, std: float) -> float:
    return float(stats.norm.logpdf(x, loc=mean, scale=std))


def _logpdf_invgamma(x: float, mean: float, std: float) -> float:
    """log-pdf of Inverse-Gamma parameterised by mean/std (Pfeifer convention).

    For SW07's shock-σ priors, Pfeifer uses inverse-gamma 1 (Sims-Zha) with
    shape parameter ν and scale s: mean ≈ s / (ν - 1), std → ν.
    The simplest reproducible parameterisation that matches Dynare's convention:
        df = std (used as a degrees-of-freedom-like shape)
        s   = mean * df
    """
    if x <= 0.0:
        return -math.inf
    # Use scipy.stats.invgamma with shape a = df / 2, scale = (mean**2 * (df + 2)) / 2
    df = std
    a = df / 2
    scale = mean ** 2 * (df + 2) / 2
    return float(stats.invgamma.logpdf(x, a=a, scale=scale))


def _logpdf_for_spec(spec: dict, x: float) -> float:
    """Dispatch to the right log-pdf based on spec['dist']."""
    if not (spec["lb"] <= x <= spec["ub"]):
        return -math.inf
    if spec["dist"] == "beta":
        return _logpdf_beta(x, spec["mean"], spec["std"])
    if spec["dist"] == "gamma":
        return _logpdf_gamma(x, spec["mean"], spec["std"])
    if spec["dist"] == "normal":
        return _logpdf_normal(x, spec["mean"], spec["std"])
    if spec["dist"] == "invgamma":
        return _logpdf_invgamma(x, spec["mean"], spec["std"])
    raise ValueError(f"unknown prior dist: {spec['dist']!r}")


# === Public API ===============================================================

def log_prior(params: dict) -> float:
    """Sum of log-prior densities across the SW07 estimated parameters.

    Returns -inf if any parameter is outside its declared support OR if any
    estimated parameter is missing from `params`.
    """
    total = 0.0
    for name, spec in PRIORS.items():
        if name not in params:
            return -math.inf
        contrib = _logpdf_for_spec(spec, params[name])
        if contrib == -math.inf:
            return -math.inf
        total += contrib
    return total


def prior_means() -> dict[str, float]:
    """Map name -> prior mean. Useful as an MLE initial point."""
    return {name: spec["mean"] for name, spec in PRIORS.items()}


def prior_stds() -> dict[str, float]:
    """Map name -> prior std. Useful as a proposal-cov fallback."""
    return {name: spec["std"] for name, spec in PRIORS.items()}


def param_bounds() -> list[tuple[float, float]]:
    """List of (lb, ub) tuples in PRIORS-insertion order (for scipy minimize)."""
    return [(spec["lb"], spec["ub"]) for spec in PRIORS.values()]


def param_names() -> tuple[str, ...]:
    """Parameter names in PRIORS-insertion order (for vec-to-dict round-trips)."""
    return tuple(PRIORS.keys())


__all__ = [
    "PRIORS",
    "log_prior",
    "prior_means",
    "prior_stds",
    "param_bounds",
    "param_names",
]
```

- [ ] **Step 5: Run tests — should pass**

```bash
python -m pytest tests/test_dsge/test_sw07_priors.py -v
```

Expected: 8 PASSED (or close — verify against actual test names you wrote in Step 2).

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/dsge/sw07_priors.py puremacro/tests/test_dsge/test_sw07_priors.py
git commit -m "$(cat <<'EOF'
feat(0.50.0): puremacro/dsge/sw07_priors — declarative priors + log_prior

Mirrors the priors block in _references/sw07_pfeifer.mod. PRIORS dict
holds ~35 estimated parameters per SW07 Table 1A across beta / gamma /
normal / inv-gamma families. log_prior(params) returns sum of scipy
log-pdfs; -inf out of support. Helpers prior_means, prior_stds,
param_bounds, param_names.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Random-Walk Metropolis sampler in `mcmc.py`

Add a sampler. Existing diagnostics (`geweke_z`, `gelman_rubin`, `effective_sample_size`, etc.) unchanged.

- [ ] **Step 1: Write the failing sampler tests**

Create `puremacro/tests/test_random_walk_metropolis.py`:

```python
"""Tests for puremacro.mcmc.random_walk_metropolis."""
import numpy as np
import pytest
from scipy import stats


def _bivariate_normal_log_density(mean, cov):
    """Return a closure log_dens(x) -> log N(x | mean, cov)."""
    inv = np.linalg.inv(cov)
    log_det = np.log(np.linalg.det(cov))
    k = len(mean)
    norm_const = -0.5 * (k * np.log(2 * np.pi) + log_det)
    def log_dens(x):
        diff = x - mean
        return float(norm_const - 0.5 * diff @ inv @ diff)
    return log_dens


def test_metropolis_recovers_2d_normal():
    """Target: bivariate N([1, -2], diag([2, 0.5])). 20K draws after burn-in
    should give empirical mean within 0.1 and empirical cov within 15%."""
    from puremacro.mcmc import random_walk_metropolis

    true_mean = np.array([1.0, -2.0])
    true_cov = np.array([[2.0, 0.0], [0.0, 0.5]])
    log_dens = _bivariate_normal_log_density(true_mean, true_cov)

    init = np.zeros(2)
    proposal_cov = np.eye(2)  # will be adapted

    result = random_walk_metropolis(
        log_dens, init, proposal_cov, n_draws=20_000,
        seed=42, accept_target=0.30, adapt_burnin=2_000,
    )
    chain = result["chain"]
    assert chain.shape == (20_000, 2)
    emp_mean = chain.mean(axis=0)
    emp_cov = np.cov(chain.T)
    np.testing.assert_allclose(emp_mean, true_mean, atol=0.1)
    np.testing.assert_allclose(emp_cov, true_cov, rtol=0.15, atol=0.05)


def test_metropolis_accept_target_adaptation():
    """After adapt_burnin, accept_rate is within ±10pp of accept_target=0.25."""
    from puremacro.mcmc import random_walk_metropolis

    log_dens = _bivariate_normal_log_density(np.zeros(2), np.eye(2))
    init = np.zeros(2)
    # Start with a wildly wrong scale; adaptation should fix it.
    result = random_walk_metropolis(
        log_dens, init, np.eye(2) * 100, n_draws=5_000,
        seed=0, accept_target=0.25, adapt_burnin=2_000,
    )
    assert 0.15 <= result["accept_rate"] <= 0.40


def test_metropolis_handles_minus_inf():
    """log_dens returns -inf outside the unit ball; chain stays inside."""
    from puremacro.mcmc import random_walk_metropolis

    def log_dens(x):
        if np.dot(x, x) > 1.0:
            return -np.inf
        return 0.0  # uniform inside the unit ball

    init = np.array([0.1, 0.1])
    result = random_walk_metropolis(
        log_dens, init, np.eye(2) * 0.1, n_draws=2_000, seed=7,
    )
    norms_sq = (result["chain"] ** 2).sum(axis=1)
    assert (norms_sq <= 1.0 + 1e-9).all()


def test_metropolis_returns_documented_keys():
    """Result dict has chain, log_post, accept_rate, final_scale."""
    from puremacro.mcmc import random_walk_metropolis
    log_dens = lambda x: -0.5 * float(x @ x)
    result = random_walk_metropolis(
        log_dens, np.zeros(3), np.eye(3), n_draws=500, seed=1,
    )
    assert set(result.keys()) >= {"chain", "log_post", "accept_rate", "final_scale"}
    assert result["chain"].shape == (500, 3)
    assert result["log_post"].shape == (500,)
    assert 0.0 <= result["accept_rate"] <= 1.0
    assert result["final_scale"] > 0


def test_metropolis_seed_reproducibility():
    """Same seed → same chain."""
    from puremacro.mcmc import random_walk_metropolis
    log_dens = lambda x: -0.5 * float(x @ x)
    init = np.zeros(2)
    cov = np.eye(2)
    r1 = random_walk_metropolis(log_dens, init, cov, n_draws=200, seed=11)
    r2 = random_walk_metropolis(log_dens, init, cov, n_draws=200, seed=11)
    np.testing.assert_array_equal(r1["chain"], r2["chain"])
```

- [ ] **Step 2: Run tests — should fail**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/test_random_walk_metropolis.py -v
```

Expected: 5 FAILED (`random_walk_metropolis` not defined).

- [ ] **Step 3: Add `random_walk_metropolis` to `puremacro/puremacro/mcmc.py`**

Append to the existing `mcmc.py` (above `__all__`):

```python
def random_walk_metropolis(
    log_posterior_fn,
    init,
    proposal_cov,
    n_draws: int,
    *,
    seed: int = 0,
    accept_target: float = 0.25,
    adapt_burnin: int = 0,
) -> dict:
    """Random-Walk Metropolis-Hastings with optional scalar-c proposal adaptation.

    Parameters
    ----------
    log_posterior_fn : callable(np.ndarray) -> float
        Target log-density (may return -np.inf).
    init : np.ndarray, shape (n_params,)
        Initial parameter vector.
    proposal_cov : np.ndarray, shape (n_params, n_params)
        Base proposal covariance. Actual proposal at iteration t is
        N(0, c**2 * proposal_cov) where c is adapted during adapt_burnin
        (no covariance adaptation; only scalar c).
    n_draws : int
        Post-burn-in iterations to retain.
    seed : int
        RNG seed.
    accept_target : float, default 0.25
        Adaptation target acceptance rate (only used if adapt_burnin > 0).
    adapt_burnin : int, default 0
        Number of additional iterations BEFORE n_draws used for scalar-c
        adaptation. Every 100 iterations, scale c by 1.1 if recent
        accept-rate > accept_target * 1.2, divide by 1.1 if < accept_target * 0.8.
        c is frozen at the end of burn-in.

    Returns
    -------
    dict with keys:
        chain       : (n_draws, n_params) — post-burn-in samples
        log_post    : (n_draws,)          — log-density at each retained sample
        accept_rate : float               — over the n_draws iterations only
        final_scale : float               — scalar c after adaptation (1.0 if adapt_burnin=0)
    """
    init = np.asarray(init, dtype=float).ravel()
    n_params = len(init)
    cov = np.asarray(proposal_cov, dtype=float)
    if cov.shape != (n_params, n_params):
        raise ValueError(
            f"proposal_cov shape {cov.shape} doesn't match init length {n_params}"
        )
    # Cholesky for fast multivariate-normal sampling.
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Add jitter and retry.
        cov = cov + 1e-10 * np.eye(n_params)
        L = np.linalg.cholesky(cov)

    rng = np.random.default_rng(seed)

    # State.
    x = init.copy()
    lp = log_posterior_fn(x)
    c = 1.0  # scalar scale
    if not np.isfinite(lp):
        raise RuntimeError(
            "log_posterior_fn(init) is not finite; pick a better starting point"
        )

    # Adaptation phase.
    adapt_window = 100
    adapt_accepts = 0
    adapt_count = 0
    for it in range(adapt_burnin):
        # Propose x' = x + c * L @ z, z ~ N(0, I).
        z = rng.standard_normal(n_params)
        x_new = x + c * (L @ z)
        lp_new = log_posterior_fn(x_new)
        log_alpha = lp_new - lp
        if np.log(rng.uniform()) < log_alpha:
            x = x_new
            lp = lp_new
            adapt_accepts += 1
        adapt_count += 1
        # Adjust c every adapt_window iterations.
        if (it + 1) % adapt_window == 0:
            rate = adapt_accepts / adapt_count
            if rate > accept_target * 1.2:
                c *= 1.1
            elif rate < accept_target * 0.8:
                c /= 1.1
            adapt_accepts = 0
            adapt_count = 0

    # Retained chain.
    chain = np.empty((n_draws, n_params))
    log_post = np.empty(n_draws)
    accepts = 0
    for it in range(n_draws):
        z = rng.standard_normal(n_params)
        x_new = x + c * (L @ z)
        lp_new = log_posterior_fn(x_new)
        log_alpha = lp_new - lp
        if np.log(rng.uniform()) < log_alpha:
            x = x_new
            lp = lp_new
            accepts += 1
        chain[it] = x
        log_post[it] = lp

    return {
        "chain": chain,
        "log_post": log_post,
        "accept_rate": accepts / n_draws,
        "final_scale": c,
    }
```

Update `__all__` at the bottom of `mcmc.py`:

```python
__all__ = [
    "geweke_z",
    "gelman_rubin",
    "autocorrelations",
    "effective_sample_size",
    "trace_summary",
    "random_walk_metropolis",
]
```

- [ ] **Step 4: Run tests — should pass**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/test_random_walk_metropolis.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/mcmc.py puremacro/tests/test_random_walk_metropolis.py
git commit -m "$(cat <<'EOF'
feat(0.50.0): puremacro.mcmc.random_walk_metropolis — RW-MH sampler

Adds an RW-MH sampler alongside the existing diagnostics. Cholesky-
factorised proposal, optional scalar-c adaptation during adapt_burnin
(no covariance adaptation — keeps the Markov property simple), seeded
RNG for reproducibility. 5 unit tests cover unbiased recovery of a 2D
normal, acceptance-target adaptation, -inf handling, return-dict
shape, and seed reproducibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Bundled SW07 data + loader

Ship the canonical SW07 1966Q1–2004Q4 dataset as a CSV alongside the package. Plus a one-time fetcher script in `tools/` for reproducibility.

- [ ] **Step 1: Write the build script `puremacro/tools/build_sw07_data.py`**

This script is run once by the maintainer to produce the CSV. It uses `puremacro.fetch.fred` (which already exists) to pull the 7 SW07 series:

| Series | FRED ID | Transformation |
|---|---|---|
| Real GDP | GDPC1 | per-capita, 100·Δlog (quarterly growth %) |
| Real consumption | PCECC96 | per-capita, 100·Δlog |
| Real investment | GPDIC1 | per-capita, 100·Δlog |
| Hours worked | HOANBS | per-capita, log level, demeaned |
| Real wage | COMPNFB / GDPDEF | log of nominal wage / deflator, 100·Δlog |
| Inflation | GDPDEF | 100·Δlog |
| Federal funds rate | FEDFUNDS | quarterly average, level (not transformed) |

Per-capita uses civilian non-institutional population (CNP16OV or LNU00000000).

```python
"""tools/build_sw07_data.py — produce the bundled SW07 dataset.

Run once by the maintainer; the resulting CSV gets committed to
puremacro/dsge/_sw07_data.csv. End-users invoking estimate_sw07() load
that committed CSV, not this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.fetch.fred import fetch_fred  # existing fetcher


SERIES = {
    "gdp":         "GDPC1",
    "cons":        "PCECC96",
    "inv":         "GPDIC1",
    "hours":       "HOANBS",
    "wage_nom":    "COMPNFB",
    "deflator":    "GDPDEF",
    "ffr":         "FEDFUNDS",
    "pop":         "CNP16OV",
}

START = "1965-10-01"  # need one lag for log-diff at 1966Q1
END = "2004-12-31"


def main(out_path: Path) -> None:
    raw: dict[str, pd.Series] = {}
    for name, fred_id in SERIES.items():
        s = fetch_fred(fred_id, start=START, end=END)
        # Resample monthly → quarterly average.
        if s.index.freqstr and s.index.freqstr.startswith("M"):
            s = s.resample("Q").mean()
        raw[name] = s.dropna()

    df = pd.DataFrame(raw).dropna()
    # Per-capita transformations.
    for nm in ("gdp", "cons", "inv", "hours"):
        df[nm + "_pc"] = df[nm] / df["pop"]
    df["wage_real_pc"] = df["wage_nom"] / df["deflator"]

    out = pd.DataFrame(index=df.index)
    out["gdp_growth"]   = 100 * np.log(df["gdp_pc"] / df["gdp_pc"].shift(1))
    out["cons_growth"]  = 100 * np.log(df["cons_pc"] / df["cons_pc"].shift(1))
    out["inv_growth"]   = 100 * np.log(df["inv_pc"] / df["inv_pc"].shift(1))
    out["wage_growth"]  = 100 * np.log(df["wage_real_pc"] / df["wage_real_pc"].shift(1))
    out["log_hours"]    = np.log(df["hours_pc"])
    out["log_hours"]   -= out["log_hours"].mean()  # demean
    out["infl"]         = 100 * np.log(df["deflator"] / df["deflator"].shift(1))
    out["ffr"]          = df["ffr"] / 4.0  # SW07 uses quarterly rate

    out = out.dropna()  # drop 1965Q4 (no lag).
    out = out[(out.index >= "1966-01-01") & (out.index <= "2004-12-31")]

    # Header comment block (CSV `#` lines).
    header = [
        "# puremacro SW07 dataset — 1966Q1 to 2004Q4 (155 quarterly obs)",
        "# Source FRED series:",
    ]
    for nm, fred_id in SERIES.items():
        header.append(f"#   {nm}: {fred_id}")
    header.append("# Transformations: per-capita, 100*log-diff; hours demeaned; FFR quarterly")
    header.append(f"# Built: {pd.Timestamp.utcnow().strftime('%Y-%m-%d')}")
    header.append("# Reference: Smets & Wouters (2007), AER")
    header.append("date," + ",".join(out.columns))

    with out_path.open("w") as fh:
        fh.write("\n".join(header) + "\n")
        out.to_csv(fh, header=False)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "puremacro" / "dsge" / "_sw07_data.csv"
    )
    main(out)
    print(f"wrote {out}")
```

- [ ] **Step 2: Run the build script (controller-direct, requires FRED network access)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python tools/build_sw07_data.py
ls -la puremacro/dsge/_sw07_data.csv
```

Expected: file exists, ~15-30 KB. If FRED is unreachable, the script errors out — the maintainer must run this from a connected machine and commit the resulting CSV.

If FRED is genuinely unavailable on this machine, FALLBACK: hand-construct the CSV from the Smets-Wouters 2007 supplement Table 2 data (155 rows × 7 columns). The supplement is at the AEA website. Either path produces the same file.

- [ ] **Step 3: Write the loader test**

Create `puremacro/tests/test_dsge/test_sw07_data.py`:

```python
"""Test the bundled SW07 dataset loader."""
import importlib.resources
import pandas as pd
import pytest


def test_bundled_sw07_data_loads():
    """The CSV is accessible via importlib.resources and parses cleanly."""
    pkg = importlib.resources.files("puremacro.dsge")
    csv_path = pkg / "_sw07_data.csv"
    assert csv_path.is_file()
    df = pd.read_csv(csv_path, comment="#", parse_dates=["date"], index_col="date")
    assert df.shape[0] >= 150 and df.shape[0] <= 160
    assert set(df.columns) == {
        "gdp_growth", "cons_growth", "inv_growth",
        "wage_growth", "log_hours", "infl", "ffr",
    }
    assert df.notna().all().all()


def test_bundled_sw07_data_date_range():
    pkg = importlib.resources.files("puremacro.dsge")
    df = pd.read_csv(pkg / "_sw07_data.csv", comment="#", parse_dates=["date"], index_col="date")
    assert df.index.min() >= pd.Timestamp("1966-01-01")
    assert df.index.max() <= pd.Timestamp("2005-01-01")
```

- [ ] **Step 4: Run the loader test**

```bash
python -m pytest tests/test_dsge/test_sw07_data.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Verify the CSV ends up in the wheel build**

```bash
python -m build --wheel -o /tmp/sw07_check_wheel 2>&1 | tail -3
python -c "
import zipfile, glob
[f] = glob.glob('/tmp/sw07_check_wheel/puremacro-*.whl')
with zipfile.ZipFile(f) as z:
    names = [n for n in z.namelist() if '_sw07_data.csv' in n]
print('found:', names)
assert names, 'CSV not packaged into wheel'
"
```

Expected: prints `found: ['puremacro/dsge/_sw07_data.csv']`. If empty, `pyproject.toml`'s `[tool.setuptools.packages.find]` is excluding the CSV. Add `[tool.setuptools.package-data]` block:

```toml
[tool.setuptools.package-data]
"puremacro.dsge" = ["_sw07_data.csv"]
```

Add only if Step 5 found nothing.

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/tools/build_sw07_data.py puremacro/puremacro/dsge/_sw07_data.csv puremacro/tests/test_dsge/test_sw07_data.py
# If you added the package-data block, also:
# git add puremacro/pyproject.toml
git commit -m "$(cat <<'EOF'
feat(0.50.0): bundled SW07 dataset + tools/build_sw07_data.py

Ships puremacro/dsge/_sw07_data.csv (1966Q1-2004Q4, 7 series, ~15-30 KB)
for out-of-the-box SW07 Bayesian estimation. tools/build_sw07_data.py
documents the FRED sources + transformations; not part of the wheel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: SW07 observation equation + `make_state_space`

This task wraps `solve_sw07` + the SW07 observation equation into a `StateSpaceModel`. The observation equation maps the 44 model variables to the 7 observed series. **Port from `puremacro/dsge/_references/sw07_pfeifer.mod`** — the `varobs` and `observation_equations` blocks. Do not re-derive.

- [ ] **Step 1: Read the reference observation equation**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
grep -A 20 "varobs\|^// observation\|equations" puremacro/dsge/_references/sw07_pfeifer.mod | head -60
```

This shows the 7 measurement equations. Roughly (Pfeifer's exact form may differ slightly):

```
dy   = y - y(-1) + ctrend
dc   = c - c(-1) + ctrend
dinve = inve - inve(-1) + ctrend
dw   = w - w(-1) + ctrend
labobs = lab + constelab
pinfobs = pinf + constepinf
robs   = r + conster        # conster derived from constebeta
```

where `y, c, inve, w, lab, pinf, r` are model variables (from `CONTROL_NAMES` in `puremacro.dsge.smets_wouters`), and `dy, dc, dinve, dw, labobs, pinfobs, robs` are the seven observables.

- [ ] **Step 2: Write the failing tests**

Create `puremacro/tests/test_dsge/test_sw07_observation.py`:

```python
"""Tests for puremacro.dsge.sw07_observation."""
import numpy as np
import pytest


def test_make_state_space_returns_StateSpaceModel():
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    from puremacro.state_space import StateSpaceModel
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)
    assert isinstance(ssm, StateSpaceModel)


def test_make_state_space_shapes():
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)
    n_state = ssm.T.shape[0]
    n_obs = ssm.Z.shape[0]
    assert n_state == 44  # SW07 has 20 predetermined + 24 forward-looking
    assert n_obs == 7     # 7 observables


def test_make_state_space_q_psd():
    """Q (state-shock covariance) is PSD."""
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)
    eigs = np.linalg.eigvalsh(ssm.Q)
    assert (eigs >= -1e-10).all()


def test_make_state_space_measurement_intercept_d_finite():
    """d (measurement intercept) has finite, sensible entries."""
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)
    assert np.isfinite(ssm.d).all()
    assert ssm.d.shape == (7,)


def test_make_state_space_h_small_ridge():
    """H (measurement-error cov) is a small positive ridge (SW07 has no
    measurement error; we use 1e-8 to keep the Kalman filter well-conditioned)."""
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)
    assert ssm.H.shape == (7, 7)
    assert (np.diag(ssm.H) > 0).all()
    assert (np.diag(ssm.H) < 1e-4).all()  # tiny ridge


def test_make_state_space_log_likelihood_finite_at_mode():
    """The Kalman filter gives a finite log-likelihood at the posterior mode + bundled data."""
    import importlib.resources
    import pandas as pd
    from puremacro.dsge.sw07_observation import make_state_space
    from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
    from puremacro.state_space import kalman_filter

    params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    ssm = make_state_space(params)

    pkg = importlib.resources.files("puremacro.dsge")
    df = pd.read_csv(pkg / "_sw07_data.csv", comment="#", parse_dates=["date"], index_col="date")
    y = df[[
        "gdp_growth", "cons_growth", "inv_growth",
        "wage_growth", "log_hours", "infl", "ffr",
    ]].to_numpy()

    out = kalman_filter(y, ssm)
    assert np.isfinite(out["loglik"])
```

- [ ] **Step 3: Run tests — should fail**

```bash
python -m pytest tests/test_dsge/test_sw07_observation.py -v
```

Expected: 6 FAILED.

- [ ] **Step 4: Write `puremacro/puremacro/dsge/sw07_observation.py`**

```python
"""SW07 observation equation: maps model variables to observable series.

Ported from puremacro/dsge/_references/sw07_pfeifer.mod (varobs + observation
equations). The seven observables are:

    dy       = log-diff of real per-capita GDP             ~ trend growth + (y - y(-1))
    dc       = log-diff of real per-capita consumption     ~ trend growth + (c - c(-1))
    dinve    = log-diff of real per-capita investment      ~ trend growth + (inve - inve(-1))
    dw       = log-diff of real per-capita wage            ~ trend growth + (w - w(-1))
    labobs   = log-hours (demeaned)                        ~ lab + constelab
    pinfobs  = quarterly GDP-deflator inflation            ~ pinf + constepinf
    robs     = quarterly federal funds rate                ~ r + conster (steady-state nominal)

The model variables (y, c, inve, w, lab, pinf, r) live at known indices in the
44-row z_t vector (predetermined states first, then forward-looking controls).
"""
from __future__ import annotations

import numpy as np

from puremacro.dsge.smets_wouters import (
    solve_sw07,
    STATE_NAMES,
    CONTROL_NAMES,
    SW07_POSTERIOR_MODE,
    SW07_SHOCK_STDS,
)
from puremacro.state_space import StateSpaceModel


OBSERVED_VARS: tuple[str, ...] = (
    "gdp_growth", "cons_growth", "inv_growth",
    "wage_growth", "log_hours", "infl", "ffr",
)

# Model variables that appear in the observation equation.
# Indices into the 44-element z_t vector (states first, then controls).
def _idx(name: str) -> int:
    if name in STATE_NAMES:
        return STATE_NAMES.index(name)
    if name in CONTROL_NAMES:
        return len(STATE_NAMES) + CONTROL_NAMES.index(name)
    raise KeyError(f"variable {name!r} not in STATE_NAMES or CONTROL_NAMES")


def make_state_space(params: dict) -> StateSpaceModel:
    """Build the SW07 state-space form for Kalman filtering.

    Parameters
    ----------
    params : dict
        Combined structural + shock-std parameter values. Must contain every
        key in SW07_POSTERIOR_MODE plus the 7 shock-std names
        ("ea", "eb", "eg", "eqs", "em", "epinf", "ew").

    Returns
    -------
    StateSpaceModel
        With fields:
            T : (44, 44)  — state transition from solve_sw07(params).G
            Z : (7, 44)   — observation matrix; picks observable model vars
            R : (44, 7)   — shock loading on states from solve_sw07(params).Impact
            Q : (7, 7)    — shock innovation covariance, diag(σ_i²)
            H : (7, 7)    — measurement-error cov; tiny ridge (1e-8) on diagonal
            d : (7,)      — measurement intercept (constants + trend offsets)
            c : (44,)     — state intercept (zeros for SW07; deviations from SS)
    """
    sol = solve_sw07(params)
    T_mat = sol.G        # (44, 44)
    R_mat = sol.Impact   # (44, 7)

    # Z: picks the rows of z_t that enter each observation equation, plus the
    # difference operator for the four growth observables.
    # For a growth observation like dy = ctrend + (y_t - y_{t-1}), we'd
    # normally need an auxiliary state for y_{t-1}. SW07's solve_sw07
    # convention already includes lagged variables among STATE_NAMES (e.g.
    # "ylag"). The observation matrix picks the difference directly.
    Z = np.zeros((7, T_mat.shape[0]))
    # dy = y - y(-1) + ctrend
    Z[0, _idx("y")]    = 1.0
    Z[0, _idx("ylag")] = -1.0
    # dc = c - c(-1) + ctrend
    Z[1, _idx("c")]    = 1.0
    Z[1, _idx("clag")] = -1.0
    # dinve = inve - inve(-1) + ctrend
    Z[2, _idx("inve")]    = 1.0
    Z[2, _idx("invelag")] = -1.0
    # dw = w - w(-1) + ctrend
    Z[3, _idx("w")]    = 1.0
    Z[3, _idx("wlag")] = -1.0
    # labobs = lab + constelab
    Z[4, _idx("lab")] = 1.0
    # pinfobs = pinf + constepinf
    Z[5, _idx("pinf")] = 1.0
    # robs = r + conster
    Z[6, _idx("r")] = 1.0

    # Q: diagonal of shock variances.
    shock_names = ("ea", "eb", "eg", "eqs", "em", "epinf", "ew")
    sigmas = np.array([params[s] for s in shock_names])
    Q_mat = np.diag(sigmas ** 2)

    # H: small ridge for numerical stability.
    H_mat = 1e-8 * np.eye(7)

    # d: measurement intercept.
    ctrend     = params["ctrend"]
    constelab  = params["constelab"]
    constepinf = params["constepinf"]
    # conster (steady-state nominal interest rate, quarterly) derived from constebeta.
    # Pfeifer: conster = ((1+constepinf/100) / (cbetabar*cgamma^(-csigma))) ** 100 - 100
    # Simpler reproducible approximation: conster ≈ constepinf + constebeta + ctrend*csigma
    # (the exact derivation lives in solve_sw07's _compute_derived; if accessible, use that).
    derived = solve_sw07.__globals__["_compute_derived"](params)
    cbeta_bar = derived["cbetabar"]
    cgamma    = derived["cgamma"]
    csigma    = params["csigma"]
    conster   = ((1 + constepinf / 100) / (cbeta_bar * cgamma ** (-csigma))) ** 100 - 100

    d_vec = np.array([
        ctrend, ctrend, ctrend, ctrend,
        constelab, constepinf, conster,
    ])

    # c: state intercept — zeros in SW07 (the model is in deviations).
    c_vec = np.zeros(T_mat.shape[0])

    return StateSpaceModel(
        T=T_mat,
        Z=Z,
        R=R_mat,
        Q=Q_mat,
        H=H_mat,
        c=c_vec,
        d=d_vec,
    )


__all__ = ["OBSERVED_VARS", "make_state_space"]
```

Caveat for the implementer: the names `ylag`, `clag`, `invelag`, `wlag` in `_idx(...)` must actually exist in `STATE_NAMES` of `smets_wouters.py`. If the SW07 implementation uses different lag names (e.g. `yf_lag` or no explicit lag — relying on the matrix structure), the Z matrix construction must adapt. **Check `STATE_NAMES` first**:

```bash
python -c "from puremacro.dsge.smets_wouters import STATE_NAMES; print(STATE_NAMES)"
```

If `ylag` etc. aren't in the list, use whatever lag-state names appear. If lag states aren't explicit, the Z matrix needs to incorporate the AR(1) structure from `T`. Implementer iterates on this until `test_make_state_space_log_likelihood_finite_at_mode` passes.

- [ ] **Step 5: Run tests — iterate until they pass**

```bash
python -m pytest tests/test_dsge/test_sw07_observation.py -v
```

Expected: 6 PASSED. The hardest one is `test_make_state_space_log_likelihood_finite_at_mode` — it ensures the entire Kalman likelihood evaluates finite. If it fails (likely first try), debug the Z matrix construction by:

1. Printing `STATE_NAMES` + `CONTROL_NAMES` to verify variable layout.
2. Cross-checking the observation equation against `_references/sw07_pfeifer.mod`.
3. Possibly adjusting the H ridge upward (1e-6 instead of 1e-8) for numerical stability.

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/dsge/sw07_observation.py puremacro/tests/test_dsge/test_sw07_observation.py
git commit -m "$(cat <<'EOF'
feat(0.50.0): puremacro/dsge/sw07_observation — state-space construction

Ports the observation equation from _references/sw07_pfeifer.mod into
make_state_space(params) -> StateSpaceModel. 7 observables, 44 model
variables. Diagonal shock covariance Q from params["ea"], ["eb"], ...
Tiny measurement-error ridge for Kalman conditioning. 6 unit tests
including Kalman log-likelihood-finite-at-mode against the bundled
dataset.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: SW07PosteriorResult dataclass + `estimate_sw07` driver + smoke test

- [ ] **Step 1: Write `puremacro/puremacro/dsge/_results.py`**

```python
"""Frozen-dataclass result types for puremacro.dsge."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SW07PosteriorResult:
    """Result of puremacro.dsge.estimate_sw07.

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
        Posterior mode (parameter → value).
    mode_hessian_inv : ndarray, shape (n_params, n_params)
        Inverse Hessian at the mode (proposal-cov foundation).
    n_burn_in : int
        Burn-in iterations dropped.
    data_n_obs : int
        Number of observations in the input dataset.
    seed : int
        Master RNG seed.
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

    def summary(self) -> pd.DataFrame:
        """Per-parameter mean, std, 5%/50%/95% quantiles across all chains."""
        flat = self.draws.reshape(-1, len(self.param_names))
        out = pd.DataFrame({
            "mean":  flat.mean(axis=0),
            "std":   flat.std(axis=0),
            "q5":    np.quantile(flat, 0.05, axis=0),
            "q50":   np.quantile(flat, 0.50, axis=0),
            "q95":   np.quantile(flat, 0.95, axis=0),
            "mode":  [self.mode[n] for n in self.param_names],
        }, index=list(self.param_names))
        return out


__all__ = ["SW07PosteriorResult"]
```

- [ ] **Step 2: Write `puremacro/puremacro/dsge/sw07_estimate.py`**

```python
"""Bayesian estimation of SW07 via Random-Walk Metropolis-Hastings."""
from __future__ import annotations

import importlib.resources
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from puremacro.dsge._results import SW07PosteriorResult
from puremacro.dsge.smets_wouters import SW07_POSTERIOR_MODE, SW07_SHOCK_STDS
from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
from puremacro.dsge.sw07_priors import (
    PRIORS,
    log_prior,
    param_bounds,
    param_names,
    prior_stds,
)
from puremacro.mcmc import random_walk_metropolis
from puremacro.numerics import mle_fit, numerical_hessian
from puremacro.state_space import kalman_filter


_FIXED_PARAMS = {
    # Calibrated parameters from SW07_POSTERIOR_MODE that are NOT estimated.
    # Per Smets-Wouters 2007 Table 1A.
    "ctou":     0.025,
    "clandaw":  1.5,
    "cg":       0.18,
    "curvp":    10.0,
    "curvw":    10.0,
    # The remaining ~36 names appear in PRIORS and are estimated.
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


def _vec_to_dict(vec: np.ndarray) -> dict:
    """Convert a parameter vector (in PRIORS order) to a dict combined with FIXED."""
    names = param_names()
    if len(vec) != len(names):
        raise ValueError(f"vec length {len(vec)} doesn't match {len(names)} priors")
    params = dict(_FIXED_PARAMS)
    for nm, val in zip(names, vec):
        params[nm] = float(val)
    return params


def _make_neg_log_posterior(y: np.ndarray):
    """Closure returning -log posterior of a parameter vector."""
    def neg_log_post(vec: np.ndarray) -> float:
        try:
            params = _vec_to_dict(vec)
        except ValueError:
            return np.inf
        lp = log_prior(params)
        if not np.isfinite(lp):
            return np.inf
        try:
            ssm = make_state_space(params)
            out = kalman_filter(y, ssm)
            ll = out["loglik"]
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            return np.inf
        if not np.isfinite(ll):
            return np.inf
        return -(ll + lp)
    return neg_log_post


def _initial_vec() -> np.ndarray:
    """Build the initial parameter vector from SW07_POSTERIOR_MODE + SW07_SHOCK_STDS."""
    init_params = {**SW07_POSTERIOR_MODE, **SW07_SHOCK_STDS}
    return np.array([init_params[n] for n in param_names()])


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
            A3 += I * (-mineig * k ** 2 + eps)
            k += 1
            if k > 100:
                raise


def estimate_sw07(
    data: Optional[pd.DataFrame] = None,
    *,
    n_draws: int = 10_000,
    n_chains: int = 2,
    burn_in: int = 2_000,
    seed: int = 0,
) -> SW07PosteriorResult:
    """Bayesian estimation of Smets-Wouters (2007) via Random-Walk MH.

    See `docs/specs/2026-05-23-puremacro-050-bayesian-dsge-design.md`.
    """
    df = _load_bundled_data() if data is None else data.copy()
    _validate_data(df)
    y = df[list(OBSERVED_VARS)].to_numpy()

    neg_log_post = _make_neg_log_posterior(y)
    init_vec = _initial_vec()

    # Step 1: refine mode (or fall back to init).
    mode_vec = init_vec.copy()
    try:
        res = mle_fit(neg_log_post, init_vec, bounds=param_bounds(), method="L-BFGS-B")
        if res.get("success", False) and np.isfinite(res.get("fun", np.inf)):
            mode_vec = np.asarray(res["x"], dtype=float)
        else:
            warnings.warn(
                "mle_fit did not converge; using SW07_POSTERIOR_MODE as the mode.",
                UserWarning,
            )
    except Exception as e:
        warnings.warn(
            f"mle_fit raised {type(e).__name__}: using SW07_POSTERIOR_MODE as the mode.",
            UserWarning,
        )

    # Step 2: Hessian-based proposal covariance.
    H = numerical_hessian(neg_log_post, mode_vec, h=1e-5)
    try:
        H_pd = _nearest_pd(H)
        inv_H = np.linalg.inv(H_pd)
    except (np.linalg.LinAlgError, RuntimeError):
        warnings.warn(
            "Hessian non-PD even after _nearest_pd; falling back to diag(prior_stds**2).",
            UserWarning,
        )
        stds = np.array([prior_stds()[n] for n in param_names()])
        inv_H = np.diag(stds ** 2)

    n_params = len(param_names())
    c0 = 2.38 / np.sqrt(n_params)
    proposal_cov = c0 ** 2 * inv_H

    # Step 3: run chains.
    chains_arr = np.empty((n_chains, n_draws, n_params))
    log_post_arr = np.empty((n_chains, n_draws))
    accept_rates = []

    def log_post_fn(vec):
        return -neg_log_post(vec)

    for chain_idx in range(n_chains):
        rng = np.random.default_rng(seed + chain_idx)
        start = mode_vec + rng.multivariate_normal(
            np.zeros(n_params), 0.0025 * inv_H,
        )
        out = random_walk_metropolis(
            log_post_fn, start, proposal_cov, n_draws=n_draws,
            seed=seed + chain_idx, accept_target=0.25, adapt_burnin=burn_in,
        )
        chains_arr[chain_idx] = out["chain"]
        log_post_arr[chain_idx] = out["log_post"]
        accept_rates.append(out["accept_rate"])

        if not (0.10 <= out["accept_rate"] <= 0.50):
            warnings.warn(
                f"chain {chain_idx} accept_rate={out['accept_rate']:.3f} "
                f"outside [0.10, 0.50]; mixing may be poor.",
                UserWarning,
            )

    mode_dict = _vec_to_dict(mode_vec)

    return SW07PosteriorResult(
        draws=chains_arr,
        param_names=param_names(),
        log_posterior_trace=log_post_arr,
        accept_rates=tuple(accept_rates),
        mode=mode_dict,
        mode_hessian_inv=inv_H,
        n_burn_in=burn_in,
        data_n_obs=len(df),
        seed=seed,
    )


__all__ = ["estimate_sw07"]
```

- [ ] **Step 3: Add the new exports to `puremacro/puremacro/dsge/__init__.py`**

Find the existing `__all__` in `puremacro/dsge/__init__.py` (or the file's top-level imports). Add:

```python
from .sw07_estimate import estimate_sw07
from ._results import SW07PosteriorResult
```

Extend `__all__` to include `"estimate_sw07"` and `"SW07PosteriorResult"`.

- [ ] **Step 4: Write the smoke test**

Create `puremacro/tests/test_dsge/test_sw07_estimate_smoke.py`:

```python
"""Smoke test for puremacro.dsge.estimate_sw07 — runs with small n_draws."""
import time

import numpy as np
import pandas as pd
import pytest


def test_estimate_sw07_tiny_runs_clean():
    """n_draws=500, n_chains=1, burn_in=200 — runs without error in <120s."""
    from puremacro.dsge import estimate_sw07, SW07PosteriorResult
    from puremacro.dsge.sw07_priors import param_names

    t0 = time.time()
    result = estimate_sw07(n_draws=500, n_chains=1, burn_in=200, seed=0)
    elapsed = time.time() - t0

    assert isinstance(result, SW07PosteriorResult)
    n_params = len(param_names())
    assert result.draws.shape == (1, 500, n_params)
    assert result.log_posterior_trace.shape == (1, 500)
    assert len(result.accept_rates) == 1
    assert 0.05 <= result.accept_rates[0] <= 0.65
    assert np.isfinite(result.draws).all()
    assert elapsed < 600  # 10-minute soft cap


def test_estimate_sw07_summary_dataframe():
    """summary() returns a DataFrame with the expected columns + per-param rows."""
    from puremacro.dsge import estimate_sw07
    from puremacro.dsge.sw07_priors import param_names

    result = estimate_sw07(n_draws=300, n_chains=1, burn_in=100, seed=1)
    s = result.summary()
    assert list(s.columns) == ["mean", "std", "q5", "q50", "q95", "mode"]
    assert list(s.index) == list(param_names())
    assert np.isfinite(s.values).all()


def test_estimate_sw07_with_user_data_runs():
    """Synthetic user data (length 60) should run the pipeline without error.
    Values are nonsense but Kalman + MH shouldn't crash."""
    from puremacro.dsge import estimate_sw07
    from puremacro.dsge.sw07_observation import OBSERVED_VARS

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        0.1 * rng.standard_normal((60, 7)),
        columns=list(OBSERVED_VARS),
        index=pd.date_range("2000-01-01", periods=60, freq="Q"),
    )
    result = estimate_sw07(data=df, n_draws=200, n_chains=1, burn_in=50, seed=2)
    assert result.data_n_obs == 60
```

- [ ] **Step 5: Run the smoke test — controller-direct (60-600s realistic)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/test_dsge/test_sw07_estimate_smoke.py -v 2>&1 | tail -10
```

Expected: 3 PASSED. If the first test takes longer than ~10 min, raise the `elapsed < 600` cap or reduce `n_draws` to 300.

- [ ] **Step 6: Run the gate to confirm no regressions**

```bash
python tools/release_check.py
```

Expected: 4 gates PASS at 0.49.0 (still; we haven't bumped yet).

- [ ] **Step 7: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/puremacro/dsge/_results.py puremacro/puremacro/dsge/sw07_estimate.py puremacro/puremacro/dsge/__init__.py puremacro/tests/test_dsge/test_sw07_estimate_smoke.py
git commit -m "$(cat <<'EOF'
feat(0.50.0): estimate_sw07 driver + SW07PosteriorResult + smoke test

estimate_sw07(data, n_draws, n_chains, burn_in, seed) -> SW07PosteriorResult.
Internally: load bundled CSV (or user DataFrame), refine mode via
numerics.mle_fit, compute numerical_hessian, scale proposal_cov by
2.38/sqrt(n), run RW-MH chains via mcmc.random_walk_metropolis.
Fallbacks: SW07_POSTERIOR_MODE if mle_fit fails; diag(prior_stds**2)
if Hessian non-PD even after Higham _nearest_pd.

3 smoke tests: tiny (n_draws=500), summary DataFrame shape, user-data
pipeline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Slow replication test + `slow` marker

- [ ] **Step 1: Declare the `slow` marker in `pyproject.toml`**

Find the existing `[tool.pytest.ini_options].markers` block. Currently:

```toml
[tool.pytest.ini_options]
markers = [
    "network: tests requiring live network access (opt-in via `pytest -m network`)",
    "pyodide_smoke: tests safe to run under Pyodide; opt-in via `pytest -m pyodide_smoke`",
]
```

Replace with:

```toml
[tool.pytest.ini_options]
markers = [
    "network: tests requiring live network access (opt-in via `pytest -m network`)",
    "pyodide_smoke: tests safe to run under Pyodide; opt-in via `pytest -m pyodide_smoke`",
    "slow: long-running tests (minutes); opt-in via `pytest -m slow`",
]
```

- [ ] **Step 2: Write the slow replication test**

Create `puremacro/tests/test_dsge/test_sw07_estimate_replication.py`:

```python
"""Slow acceptance test: posterior means within ~25% of SW07 Table 1A."""
import numpy as np
import pytest


# Reference values from Smets-Wouters (2007) Table 1A posterior means.
# Eight parameters are picked as the most tightly identified.
SW07_TABLE1A_REFERENCE = {
    "ctrend":     0.43,
    "constebeta": 0.16,
    "chabb":      0.71,
    "cprobp":     0.65,
    "cprobw":     0.73,
    "cindp":      0.24,
    "cindw":      0.59,
    "csigma":     1.39,
}
TOL_RELATIVE = 0.25  # ±25% of reference


@pytest.mark.slow
def test_estimate_sw07_posterior_means_close_to_sw07_table1a():
    """10K draws × 2 chains. Posterior means for 8 anchor params within ±25%."""
    from puremacro.dsge import estimate_sw07

    result = estimate_sw07(n_draws=10_000, n_chains=2, burn_in=2_000, seed=0)
    summary = result.summary()

    failures = []
    for name, ref in SW07_TABLE1A_REFERENCE.items():
        if name not in summary.index:
            failures.append((name, ref, None, "not in posterior"))
            continue
        post_mean = summary.loc[name, "mean"]
        if not np.isfinite(post_mean):
            failures.append((name, ref, post_mean, "non-finite"))
            continue
        rel_err = abs(post_mean - ref) / max(abs(ref), 1e-3)
        if rel_err > TOL_RELATIVE:
            failures.append((name, ref, post_mean, f"rel_err={rel_err:.2%}"))

    assert not failures, (
        "Posterior means deviated from SW07 Table 1A:\n"
        + "\n".join(
            f"  {n}: ref={r}, got={got}, {msg}"
            for n, r, got, msg in failures
        )
    )
```

- [ ] **Step 3: Verify the marker is collected (without running)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/test_dsge/test_sw07_estimate_replication.py --collect-only -q
```

Expected: 1 test collected, marked `slow`.

- [ ] **Step 4: Verify default `pytest tests/` skips the slow test**

```bash
python -m pytest tests/test_dsge/test_sw07_estimate_replication.py -v
```

Expected: 1 deselected (the test has `@pytest.mark.slow` but pytest's default selection doesn't include it).

Actually, by default `pytest -m slow` is the opt-in. Without `-m slow`, pytest collects the test but does NOT deselect it (markers don't auto-skip; that requires either `-m "not slow"` or a conftest.py addopts entry). For this plan we accept the default behaviour: the test will run on `pytest tests/` and take ~20-40 minutes.

If you want it to skip by default, add to `pyproject.toml::[tool.pytest.ini_options]`:

```toml
addopts = ["-m", "not slow"]
```

Decide based on whether you want the slow test to run on a default `pytest`. **Recommendation: add `addopts = ["-m", "not slow"]` so the gate (Gate 1) stays fast**; the slow test runs only when explicitly requested.

- [ ] **Step 5: Add the addopts entry**

Update `puremacro/pyproject.toml` to add (if not already present):

```toml
[tool.pytest.ini_options]
addopts = ["-m", "not slow"]
markers = [
    # ... existing markers + the new slow one
]
```

- [ ] **Step 6: Verify the slow test is now skipped by default**

```bash
python -m pytest tests/test_dsge/test_sw07_estimate_replication.py -v 2>&1 | tail -5
```

Expected: `1 deselected`.

```bash
python -m pytest tests/test_dsge/test_sw07_estimate_replication.py -m slow -v
```

The replication test now actually runs. Wall-time 20-40 min. **The plan does NOT require this to pass before commit** — it's an opt-in acceptance test. If posterior means are off by more than 25%, that's a real research finding (the Bayesian estimation isn't replicating SW07) that needs separate diagnosis. **DO NOT block the release on this test failing — diagnose, document the deviation in the CHANGELOG, and ship 0.50.0 with a noted caveat.**

- [ ] **Step 7: Commit (regardless of whether the slow test passes)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/pyproject.toml puremacro/tests/test_dsge/test_sw07_estimate_replication.py
git commit -m "$(cat <<'EOF'
feat(0.50.0): slow replication test + slow marker + addopts

Adds @pytest.mark.slow to the pyproject markers list with
addopts = ["-m", "not slow"] so the default pytest run skips slow tests.
The new tests/test_dsge/test_sw07_estimate_replication.py checks that
8 posterior means from SW07 Table 1A are recovered within ±25%.
Opt-in via `pytest -m slow`; ~20-40 min wall.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Bump 0.49.0 → 0.50.0 + CHANGELOG + final gate

- [ ] **Step 1: Update CONTRIBUTING.md to document `pytest -m slow`**

In `puremacro/CONTRIBUTING.md`, in the existing "Before tagging a release" section (after the Gate 6 `--pyodide` subsection), append:

```markdown

### Opt-in: slow tests (`pytest -m slow`)

Some tests are long-running (minutes). They are skipped by default
via the `addopts = ["-m", "not slow"]` in `pyproject.toml`. Run them
explicitly before tag if the change touches Bayesian estimation or
DSGE code:

```bash
pytest -m slow tests/test_dsge/test_sw07_estimate_replication.py
```

The SW07 Bayesian replication test runs ~20-40 min.
```

(Use literal triple-backticks in the actual file.)

- [ ] **Step 2: Bump three version strings**

- `puremacro/pyproject.toml`: `version = "0.49.0"` → `"0.50.0"`.
- `puremacro/puremacro/__init__.py`: `__version__ = "0.49.0"` → `"0.50.0"`.
- `puremacro/tests/test_import.py`: `assert puremacro.__version__ == "0.49.0"` → `"0.50.0"`.

- [ ] **Step 3: Prepend CHANGELOG 0.50.0 entry**

Edit `puremacro/CHANGELOG.md`. After the `# Changelog` preamble and BEFORE the existing `## 0.49.0` heading, insert:

```markdown
## 0.50.0 — 2026-05-23

Bayesian estimation of Smets-Wouters (2007) (`estimate_sw07`).

Closes the loop between `solve_sw07` (which ships at posterior MODE)
and a true Bayesian estimator. P1 from the 2026-05-22 brainstorm; last
of the four picks (P4 → P5 → P3 → P1).

### Added
- `puremacro.dsge.estimate_sw07(data, n_draws, n_chains, burn_in, seed)
  -> SW07PosteriorResult` — single-function Bayesian DSGE estimator.
  Internally: refines posterior mode via `numerics.mle_fit`, computes
  numerical Hessian, scales the RW-MH proposal by `2.38/sqrt(n)`, runs
  `n_chains` sequential chains with optional scalar-c proposal-scale
  adaptation during burn-in.
- `puremacro.dsge.SW07PosteriorResult` (frozen dataclass) — draws,
  param_names, log-posterior trace, accept rates, mode, mode_hessian_inv,
  n_burn_in, data_n_obs, seed. `.summary()` returns a DataFrame with
  per-parameter mean/std/q5/q50/q95/mode.
- `puremacro.mcmc.random_walk_metropolis(log_posterior_fn, init,
  proposal_cov, n_draws, *, seed, accept_target, adapt_burnin)` — new
  sampler in mcmc.py (the existing diagnostics are unchanged).
- `puremacro.dsge.sw07_priors.PRIORS` + `log_prior`, `prior_means`,
  `prior_stds`, `param_bounds`, `param_names` — declarative priors
  per SW07 Table 1A, ported from `_references/sw07_pfeifer.mod`.
- `puremacro.dsge.sw07_observation.make_state_space(params)
  -> StateSpaceModel` — SW07 observation equation, 44 model variables
  to 7 observables (per-capita real GDP growth, consumption growth,
  investment growth, wage growth, log hours, GDP-deflator inflation,
  federal funds rate).
- `puremacro/dsge/_sw07_data.csv` — bundled 1966Q1-2004Q4 US dataset
  (155 quarterly obs × 7 columns). Built once via
  `tools/build_sw07_data.py`.

### Changed
- `pyproject.toml::[tool.pytest.ini_options]` declares the `slow`
  marker and adds `addopts = ["-m", "not slow"]` so default pytest
  runs skip slow tests. Slow tests run via `pytest -m slow`.
- `CONTRIBUTING.md` "Before tagging a release" section documents
  the slow-tests opt-in.

### Internal
- 5 new test files: `test_sw07_priors.py`, `test_random_walk_metropolis.py`,
  `test_sw07_data.py`, `test_sw07_observation.py`,
  `test_sw07_estimate_smoke.py` — plus the slow opt-in
  `test_sw07_estimate_replication.py`.
- The 8 `pyodide_smoke`-marked tests from 0.49.0 are unchanged.
- Live `estimate_sw07(n_draws=10_000, n_chains=2)` wall-time on the
  maintainer's machine: ~20-40 min (Kalman likelihood evaluations
  dominate).

### Out of scope (deferred to follow-on specs)
- Generic Bayesian DSGE engine (a `(solve_fn, prior_spec, obs_eq, data)
  -> Posterior` engine awaits a second DSGE model).
- HANK / TANK linearized solvers.
- Adaptive RW-MH with covariance adaptation, DEMC, NUTS, HMC.
- Fresh-data SW07 fetcher (the canonical vintage is bundled).
- Pyodide Gate 6 coverage of `estimate_sw07` — too slow for the
  6-second Gate 6 budget; existing 8-test smoke is unchanged.
- PyPI publishing (still queued from the 0.48.0 roadmap).

---
```

- [ ] **Step 4: Regenerate the public-API snapshot**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -c "
import sys
sys.path.insert(0, 'tests')
from test_public_api import collect_current_api
import json
data = collect_current_api()
with open('tests/fixtures/public_api_snapshot.json', 'w') as fh:
    json.dump(data, fh, indent=2, sort_keys=True)
print('snapshot regenerated')
"
```

Verify the new symbols appear:

```bash
grep "estimate_sw07\|SW07PosteriorResult\|random_walk_metropolis" tests/fixtures/public_api_snapshot.json | head -10
```

Expected: 3 entries (one for each new public symbol).

- [ ] **Step 5: Pre-commit full gate verification**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python tools/release_check.py --examples --pyodide
```

Expected: all 6 gates PASS at 0.50.0. Wall-time ~2-3 min (Gate 1 runs the full suite minus slow; Gate 6 ~6s).

- [ ] **Step 6: Commit the bump**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/pyproject.toml puremacro/puremacro/__init__.py puremacro/tests/test_import.py puremacro/CHANGELOG.md puremacro/CONTRIBUTING.md puremacro/tests/fixtures/public_api_snapshot.json
git commit -m "$(cat <<'EOF'
chore(puremacro): bump 0.49.0 → 0.50.0 (Bayesian SW07 estimation ships)

Three version strings synced. CHANGELOG 0.50.0 entry. Public-API
snapshot regenerated for estimate_sw07, SW07PosteriorResult,
random_walk_metropolis. CONTRIBUTING.md documents `pytest -m slow`
opt-in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Final gate run on the bump commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python tools/release_check.py --examples --pyodide
```

Expected: exit 0, all 6 gates PASS at 0.50.0.

- [ ] **Step 8: Hand off to controller for merge + tag**

Do NOT auto-merge or tag. Controller asks the user (same pattern as 0.46.0 / 0.47.0 / 0.48.0 / 0.49.0).

---

## Self-review notes

**Spec coverage** (from `docs/specs/2026-05-23-puremacro-050-bayesian-dsge-design.md`):

- `pyproject.toml` markers list adds `slow` — Task 6 ✓
- 8 priors (Table 1A subset) — Task 1 ✓
- `tools/pyodide/` etc. — N/A (this release doesn't extend Gate 6)
- `tools/pyodide_smoke.py` — N/A
- Gate 6 — N/A
- `random_walk_metropolis` — Task 2 ✓
- `sw07_priors.py` + `sw07_observation.py` — Tasks 1 + 4 ✓
- `sw07_estimate.py` + `SW07PosteriorResult` — Task 5 ✓
- `_sw07_data.csv` — Task 3 ✓
- Unit tests across the four `test_*` files — Tasks 1, 2, 4, 5 ✓
- Smoke test (n_draws=500) — Task 5 ✓
- Slow replication test — Task 6 ✓
- Public-API snapshot regen — Task 7 ✓
- Live 6-gate run at 0.50.0 — Task 7 ✓
- CONTRIBUTING `pytest -m slow` note — Task 7 ✓
- CHANGELOG 0.50.0 entry — Task 7 ✓
- Version bump — Task 7 ✓

**Placeholder scan:**

- Task 1 Step 1's PRIORS dict is a sketch — the implementer must verify against `_references/sw07_pfeifer.mod` and adjust means/stds. This is documented inline as "Verify the actual SW07 Table 1A means/stds against `_references/sw07_pfeifer.mod`" — explicit audit step, not a TBD.
- Task 4 Step 4 has a caveat block about lag-state names (`ylag`, `clag`, etc.) possibly not existing in `STATE_NAMES`. The implementer must check `STATE_NAMES` and adapt the Z matrix construction. Explicit "audit-then-decide" branch.
- No bare TBD / TODO / "implement later" patterns.

**Type consistency:**

- `log_prior(params: dict) -> float` consistent across Tasks 1, 5.
- `make_state_space(params: dict) -> StateSpaceModel` consistent across Tasks 4, 5.
- `random_walk_metropolis(log_posterior_fn, init, proposal_cov, n_draws, *, seed, accept_target, adapt_burnin) -> dict` with keys `{chain, log_post, accept_rate, final_scale}` consistent across Tasks 2, 5.
- `SW07PosteriorResult` fields: `draws, param_names, log_posterior_trace, accept_rates, mode, mode_hessian_inv, n_burn_in, data_n_obs, seed` — consistent in Tasks 5 (definition) and 6 (consumer via `summary()`).
- `param_names(), prior_means(), prior_stds(), param_bounds()` — names consistent between Task 1 (definition) and Task 5 (consumer).

**Risks pulled forward from the spec:**

- **R1 (mode init may not converge):** Task 5's `estimate_sw07` falls back to `SW07_POSTERIOR_MODE` directly with a warning. Documented.
- **R2 (proposal scaling):** Task 2's `adapt_burnin` scalar-c adaptation hits the target rate.
- **R3 (Kalman likelihood slow):** Task 5's smoke test caps elapsed at 600s; Task 6's slow replication is the 20-40 min target.
- **R4 (CSV vintage):** Task 3's CSV has header comments documenting FRED IDs + vintage date.
- **R5 (observation equation transcription):** Task 4 Step 4 directs the implementer to port from `_references/sw07_pfeifer.mod`; tests include an end-to-end Kalman-likelihood-finite-at-mode check.
- **R6 (Hessian non-PD):** Task 5's `_nearest_pd` fallback chain.
- **R7 (`slow` marker):** Task 6 declares the marker + addopts entry.
- **R8 (Pyodide gap):** Task 6's slow test is NOT `pyodide_smoke`-marked. Documented in the CHANGELOG.

**Out of scope (deferred to follow-on plans):**

- Generic Bayesian DSGE engine, HANK, alternate samplers — separate specs.
- PyPI publishing — still queued from 0.48.0.
- Fresh-data SW07 fetcher — small follow-up spec.
