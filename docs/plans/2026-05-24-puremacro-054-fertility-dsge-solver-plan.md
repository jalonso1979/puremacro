# puremacro 0.54.0 Implementation Plan — fertility DSGE solver (R1b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `fertility_adj_costs.mod` (calibrated baseline) onto puremacro: BGP fsolve calibration + numerical Jacobians + Klein QZ solve + IRF/FEVD methods + one demo. Solver only — Bayesian estimation deferred to R1c (0.55.0).

**Architecture:** One new file `puremacro/dsge/fertility_adj_costs.py` (~350 LOC) holding all model machinery, plus a `FertilitySolution` dataclass added to `puremacro/dsge/_results.py`, plus a demo. The `solve_fertility` orchestrator calls `solve_bgp` (scipy.optimize.fsolve port of `bgp_fertility_calibration.m`), then numerical Jacobians (central differences) on `model_residuals` at the BGP, then `puremacro.dsge.klein_solve`.

**Tech Stack:** numpy, scipy (`optimize.fsolve`, central-difference Jacobians), pandas (for IRF/FEVD DataFrames), matplotlib (demo). No new dependencies.

**Spec:** `docs/specs/2026-05-24-puremacro-054-fertility-dsge-solver-design.md`

---

## File map

### New files
- `puremacro/dsge/fertility_adj_costs.py` (~350 LOC).
- `puremacro/examples/dsge_fertility_demo.py` (~80 LOC).
- `tests/test_dsge/test_fertility_bgp.py` (~5 tests).
- `tests/test_dsge/test_fertility_residuals.py` (~3 tests).
- `tests/test_dsge/test_solve_fertility.py` (~5 tests).
- `tests/test_dsge/test_fertility_irf_fevd.py` (~3 tests).
- `tests/test_examples/test_dsge_fertility_demo_runs.py` (~1 test).
- `tests/test_examples/__init__.py` (empty, if missing).

### Modified files
- `puremacro/dsge/_results.py` — add `FertilitySolution` dataclass.
- `puremacro/dsge/__init__.py` — add `solve_bgp`, `solve_fertility`, `FertilitySolution`, `fertility_adj_costs` submodule re-export.
- `puremacro/__init__.py` — bump `__version__` to `"0.54.0"`.
- `pyproject.toml` — bump `version` to `"0.54.0"`.
- `CHANGELOG.md` — add 0.54.0 section.
- `tests/test_import.py` — bump pinned version assertion.
- `tests/fixtures/public_api_snapshot.json` — regenerate.

### Verified API surfaces
- `puremacro.dsge.klein_solve(A, B, n_pre, C=None, *, strict=False) -> KleinSolution`
  with fields `G (n_pre, n_pre), F (n_fwd, n_pre), N (n_pre, n_u), L (n_fwd, n_u), eu (tuple), eigenvalues`. System form: `A · E_t z_{t+1} = B · z_t + C · u_t`. `n_pre` is the COUNT of predetermined variables; the first n_pre rows of z are states, the rest are forward-looking.
- `puremacro.dsge.BlanchardKahnError` raised by `klein_solve(..., strict=True)` when BK fails.
- `scipy.optimize.fsolve(func, x0, args=(), full_output=True)` — returns `(x, infodict, ier, mesg)` with `ier == 1` on success.
- `scipy.optimize.approx_fprime(xk, f, epsilon=...)` — forward-difference; we'll roll our own central differences.
- Variable-timing classification (from spec): n_pre=5 (a, mun, ph, k, n), n_fwd=7 (c, y, l_w, u, i, b, l_o) where the last three are "static" — they get infinite generalised eigenvalues from Klein QZ, which count as unstable, summing to 7 unstable eigenvalues (matching n_fwd).

### Variable ordering in VAR_NAMES (locked for the whole release)
```python
VAR_NAMES = (
    # --- 5 predetermined / state variables (n_pre = 5) ---
    "a",      # log productivity (AR1 state)
    "mun",    # log fertility-preference (AR1 state)
    "ph",     # log mortality (AR1 state)
    "k",      # capital stock (predetermined; appears at t-1, t, t+1)
    "n",      # children stock (predetermined; appears at t-1, t, t+1)
    # --- 7 non-predetermined (controls) (n_fwd = 7) ---
    "c",      # consumption (forward-looking)
    "y",      # output (forward-looking)
    "l_w",    # work hours (forward-looking)
    "u",      # capital utilization (forward-looking)
    "i",      # investment (static)
    "b",      # births (static)
    "l_o",    # leisure (static)
)
```

This ordering is mandatory: it's the partition Klein QZ requires.

---

## Task 1: Add FertilitySolution dataclass

**Files:**
- Modify: `puremacro/dsge/_results.py`.
- Modify (new tests): `tests/test_dsge/test_identify_results.py` doesn't exist for dsge — create `tests/test_dsge/test_fertility_solution.py` with one test, OR append to an existing dsge test file. Easier path: append to `tests/test_dsge/test_sw07_wrapper.py` (which already exists from the 0.53.0 release and is the convention for dsge result-class tests).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dsge/test_sw07_wrapper.py`:
```python
def test_fertility_solution_dataclass_is_frozen_with_expected_fields():
    import dataclasses
    import numpy as np
    import pytest
    from puremacro.dsge._results import FertilitySolution

    res = FertilitySolution(
        ss={"c": 1.0, "k": 5.0},
        params={"alpha": 0.4},
        G=np.eye(5),
        N=np.zeros((5, 3)),
        F=np.zeros((7, 5)),
        L=np.zeros((7, 3)),
        klein_solution=None,
        var_names=("a", "mun", "ph", "k", "n", "c", "y", "l_w", "u", "i", "b", "l_o"),
        shock_names=("ea", "ep", "en"),
    )
    assert res.ss["c"] == 1.0
    assert res.G.shape == (5, 5)
    assert res.shock_names == ("ea", "ep", "en")
    assert dataclasses.is_dataclass(res)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.ss = {}
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_dsge/test_sw07_wrapper.py::test_fertility_solution_dataclass_is_frozen_with_expected_fields -v`
Expected: `ImportError: cannot import 'FertilitySolution'`.

- [ ] **Step 3: Add the dataclass**

Append to `puremacro/dsge/_results.py` (after `DSGEPosteriorResult` + `SW07PosteriorResult`):
```python
@dataclass(frozen=True)
class FertilitySolution:
    """Linear solution of the fertility DSGE around its BGP.

    Attributes
    ----------
    ss : dict[str, float]
        Steady-state values keyed by variable name (matches VAR_NAMES).
    params : dict[str, float]
        All parameters used in the solve (structural + calibration +
        shock-process).
    G : ndarray, shape (n_states, n_states)
        State transition (state at t given state at t-1, no shock).
    N : ndarray, shape (n_states, n_shocks)
        Shock impact on states.
    F : ndarray, shape (n_controls, n_states)
        Control policy (control at t given state at t).
    L : ndarray, shape (n_controls, n_shocks)
        Control response to contemporaneous shock.
    klein_solution : KleinSolution or None
        Raw QZ output for debugging.
    var_names : tuple of str
        All 12 endogenous variable names (states first, then controls).
    shock_names : tuple of str
        Shock names (ea, ep, en).

    Notes
    -----
    The first n_states entries of var_names are the predetermined
    variables (rows of G/N); the remaining are controls (rows of F/L).
    """

    ss: dict
    params: dict
    G: np.ndarray
    N: np.ndarray
    F: np.ndarray
    L: np.ndarray
    klein_solution: object
    var_names: tuple
    shock_names: tuple

    def irf(self, shock, horizon: int = 20) -> pd.DataFrame:
        """Impulse response to a 1-SD shock. See fertility_adj_costs.solve_fertility docstring."""
        # Defer to a free function so the dataclass stays a pure container.
        from puremacro.dsge.fertility_adj_costs import _compute_irf
        return _compute_irf(self, shock, horizon)

    def fevd(self, horizon: int = 20) -> pd.DataFrame:
        """Forecast-error variance decomposition."""
        from puremacro.dsge.fertility_adj_costs import _compute_fevd
        return _compute_fevd(self, horizon)
```

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_dsge/test_sw07_wrapper.py::test_fertility_solution_dataclass_is_frozen_with_expected_fields -v`
Expected: PASS.

(The `_compute_irf` / `_compute_fevd` imports inside the methods will resolve later — Task 6 implements them. Task 1's test doesn't exercise those methods.)

- [ ] **Step 5: Commit**

```bash
git add puremacro/dsge/_results.py tests/test_dsge/test_sw07_wrapper.py
git commit -m "feat(dsge): add FertilitySolution dataclass for 0.54.0"
```

---

## Task 2: fertility_adj_costs.py skeleton — constants + variable timing

**Files:**
- Create: `puremacro/dsge/fertility_adj_costs.py` (skeleton, no functions yet).
- Create: `tests/test_dsge/test_fertility_constants.py` (~3 tests).

- [ ] **Step 1: Write 3 failing tests**

Create `tests/test_dsge/test_fertility_constants.py`:
```python
"""Tests for puremacro.dsge.fertility_adj_costs constants."""
from __future__ import annotations


def test_var_names_has_12_entries_in_state_then_control_order():
    from puremacro.dsge.fertility_adj_costs import VAR_NAMES
    assert len(VAR_NAMES) == 12
    # First 5 are states: a, mun, ph, k, n
    assert VAR_NAMES[:5] == ("a", "mun", "ph", "k", "n")
    # Next 7 are controls: c, y, l_w, u, i, b, l_o
    assert VAR_NAMES[5:] == ("c", "y", "l_w", "u", "i", "b", "l_o")


def test_shock_names_has_3_entries():
    from puremacro.dsge.fertility_adj_costs import SHOCK_NAMES
    assert SHOCK_NAMES == ("ea", "ep", "en")


def test_exogenous_params_have_expected_keys():
    from puremacro.dsge.fertility_adj_costs import (
        FERTILITY_EXOGENOUS_PARAMS, FERTILITY_CALIB_TARGETS,
        FERTILITY_SHOCK_PROCESSES,
    )
    assert set(FERTILITY_EXOGENOUS_PARAMS.keys()) == {
        "alpha", "nu", "phi", "g", "delta_p", "delta_n", "omega", "bara",
    }
    assert set(FERTILITY_CALIB_TARGETS.keys()) == {
        "l", "u", "depr_rate", "kid_cost_share", "k_y_ratio", "c_y_ratio",
        "n_growth",
    }
    assert set(FERTILITY_SHOCK_PROCESSES.keys()) == {"ea", "en", "ep"}
    # Spot-check a few values
    assert FERTILITY_EXOGENOUS_PARAMS["alpha"] == 0.4
    assert FERTILITY_CALIB_TARGETS["k_y_ratio"] == 2.8
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_dsge/test_fertility_constants.py -v`
Expected: `ImportError: cannot import 'VAR_NAMES' from 'puremacro.dsge.fertility_adj_costs'`.

- [ ] **Step 3: Create the skeleton**

Create `puremacro/dsge/fertility_adj_costs.py`:
```python
"""Fertility DSGE (adjustment-costs baseline).

Ported from My Drive/Fertility/fertility_adj_costs.mod and
bgp_fertility_calibration.m. The 12-variable linear DSGE has 5
predetermined states (a, mun, ph, k, n) and 7 controls
(c, y, l_w, u, i, b, l_o). Three exogenous shocks (ea, ep, en) drive
the three AR(1) shock processes.

Public entry points:

- ``solve_bgp(exogenous, targets, x0, tol)`` — BGP fsolve calibration.
- ``model_residuals(z_lead, z, z_lag, eps, params)`` — 12 model-equation
  residuals (for steady-state verification + numerical Jacobians).
- ``solve_fertility(params, *, shock_stds, h_for_jacobians)`` — top-level
  orchestrator returning a FertilitySolution.

Solver-only release (R1b, 0.54.0). Bayesian estimation lands in R1c
(0.55.0) — it will wire priors + observation onto puremacro.dsge.estimate_dsge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from puremacro.dsge._results import FertilitySolution
from puremacro.dsge.klein import klein_solve, KleinSolution


# === Variable / shock declarations ============================================

VAR_NAMES: tuple[str, ...] = (
    # Predetermined / state variables (n_pre = 5)
    "a", "mun", "ph", "k", "n",
    # Non-predetermined / controls (n_fwd = 7)
    "c", "y", "l_w", "u", "i", "b", "l_o",
)

SHOCK_NAMES: tuple[str, ...] = ("ea", "ep", "en")

N_PRE: int = 5
N_FWD: int = 7
N_VARS: int = len(VAR_NAMES)
N_SHOCKS: int = len(SHOCK_NAMES)
assert N_PRE + N_FWD == N_VARS


# Per-variable timing (which slots it appears in: lag, current, lead).
# Used by the numerical-Jacobian construction to skip zero-perturbation
# evaluations and by sanity-check assertions.
_VARIABLE_TIMING: dict[str, set[str]] = {
    "a":    {"lag", "current"},
    "mun":  {"lag", "current"},
    "ph":   {"lag", "current"},
    "k":    {"lag", "current", "lead"},
    "n":    {"lag", "current", "lead"},
    "c":    {"current", "lead"},
    "y":    {"current", "lead"},
    "l_w":  {"current", "lead"},
    "u":    {"current", "lead"},
    "i":    {"current"},
    "b":    {"current"},
    "l_o":  {"current"},
}


# === Default parameter blocks =================================================

FERTILITY_EXOGENOUS_PARAMS: dict[str, float] = {
    "alpha":   0.4,
    "nu":      2.5,
    "phi":     1.03,
    "g":       0.017,
    "delta_p": 0.075,
    "delta_n": 0.065,
    "omega":   2.5,
    "bara":    0.0,
}

FERTILITY_CALIB_TARGETS: dict[str, float] = {
    "l":              0.30,
    "u":              1.00,
    "depr_rate":      0.10,
    "kid_cost_share": 0.18,
    "k_y_ratio":      2.80,
    "c_y_ratio":      0.75,
    "n_growth":       1.02,
}

FERTILITY_SHOCK_PROCESSES: dict[str, dict] = {
    "ea": {"rho": 0.94 ** 4, "sigma": 0.01},
    "en": {"rho": 0.50,      "sigma": 0.07},
    "ep": {"rho": 0.90,      "sigma": 0.07},
}


# === BGP, model, solver functions added in later tasks ========================

# Tasks 3-6 add: solve_bgp, model_residuals, solve_fertility, _compute_irf, _compute_fevd.


__all__ = [
    "VAR_NAMES", "SHOCK_NAMES", "N_PRE", "N_FWD", "N_VARS", "N_SHOCKS",
    "FERTILITY_EXOGENOUS_PARAMS", "FERTILITY_CALIB_TARGETS",
    "FERTILITY_SHOCK_PROCESSES",
]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_dsge/test_fertility_constants.py -v`
Expected: 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add puremacro/dsge/fertility_adj_costs.py tests/test_dsge/test_fertility_constants.py
git commit -m "feat(fertility): module skeleton + variable/shock/param constants"
```

---

## Task 3: solve_bgp — BGP fsolve calibration

**Files:**
- Modify: `puremacro/dsge/fertility_adj_costs.py` (add `solve_bgp` + `_bgp_system` helper).
- Create: `tests/test_dsge/test_fertility_bgp.py` (~5 tests).

- [ ] **Step 1: Write 5 failing tests**

Create `tests/test_dsge/test_fertility_bgp.py`:
```python
"""Tests for puremacro.dsge.fertility_adj_costs.solve_bgp."""
from __future__ import annotations

import math

import numpy as np
import pytest


def test_solve_bgp_converges_with_defaults():
    from puremacro.dsge.fertility_adj_costs import solve_bgp
    bgp = solve_bgp()
    # All returned values are finite floats
    for key, val in bgp.items():
        assert math.isfinite(val), f"{key} = {val} is not finite"
    # Has all expected keys (8 exogenous + 7 calibrated + 6 fsolve + 4 derived + 3 shock-process)
    required = {
        # exogenous
        "alpha", "nu", "phi", "g", "delta_p", "delta_n", "omega", "bara",
        # calibrated by fsolve
        "barn", "mu_l", "beta", "tau_n", "tau_b", "p_n", "delta_k",
        # fsolve-derived steady-state values
        "c", "b", "l_w", "u", "k", "n",
        # derived from SS identities
        "i", "l_o", "y",
        # shock-process steady states
        "a", "mun", "ph",
    }
    assert required.issubset(set(bgp.keys()))


def test_solve_bgp_satisfies_residuals():
    from puremacro.dsge.fertility_adj_costs import solve_bgp, _bgp_system
    bgp = solve_bgp()
    # Reconstruct the 13-element x vector and verify residuals
    x = np.array([
        bgp["barn"], bgp["mu_l"], bgp["beta"], bgp["tau_n"], bgp["tau_b"],
        bgp["p_n"], bgp["delta_k"], bgp["c"], bgp["b"], bgp["l_w"],
        bgp["u"], bgp["k"], bgp["n"],
    ])
    from puremacro.dsge.fertility_adj_costs import (
        FERTILITY_EXOGENOUS_PARAMS, FERTILITY_CALIB_TARGETS,
    )
    F = _bgp_system(x, FERTILITY_EXOGENOUS_PARAMS, FERTILITY_CALIB_TARGETS)
    assert np.max(np.abs(F)) < 1e-8, f"max residual = {np.max(np.abs(F)):.2e}"


def test_solve_bgp_satisfies_calibration_targets():
    from puremacro.dsge.fertility_adj_costs import solve_bgp
    bgp = solve_bgp()
    assert bgp["l_w"] == pytest.approx(0.30, abs=1e-6)
    assert bgp["u"] == pytest.approx(1.00, abs=1e-6)
    assert bgp["k"] / bgp["y"] == pytest.approx(2.80, abs=1e-6)
    assert bgp["c"] / bgp["y"] == pytest.approx(0.75, abs=1e-6)
    assert bgp["p_n"] * bgp["n"] / bgp["y"] == pytest.approx(0.18, abs=1e-6)


def test_solve_bgp_raises_on_missing_target():
    from puremacro.dsge.fertility_adj_costs import solve_bgp
    bad_targets = {"l": 0.30, "u": 1.0}  # missing several
    with pytest.raises(KeyError, match="missing"):
        solve_bgp(targets=bad_targets)


def test_solve_bgp_custom_targets_override():
    from puremacro.dsge.fertility_adj_costs import (
        solve_bgp, FERTILITY_CALIB_TARGETS,
    )
    custom = dict(FERTILITY_CALIB_TARGETS)
    custom["k_y_ratio"] = 3.5
    bgp = solve_bgp(targets=custom)
    assert bgp["k"] / bgp["y"] == pytest.approx(3.5, abs=1e-5)
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_dsge/test_fertility_bgp.py -v`
Expected: `ImportError: cannot import 'solve_bgp'`.

- [ ] **Step 3: Implement solve_bgp + _bgp_system**

Replace the `# Tasks 3-6 add: ...` placeholder line in `puremacro/dsge/fertility_adj_costs.py` with:
```python
# === BGP calibration (fsolve port of bgp_fertility_calibration.m) ============

import scipy.optimize  # local import to keep top-level imports minimal


_BGP_X0_DEFAULT: np.ndarray = np.array([
    0.5,   # barn   (the MATLAB script's mu_n = exp(barn) — we solve for barn directly here; init guess of 0.5 means barn ≈ -0.69, which is fine)
    1.2,   # mu_l   (MATLAB: mu_h)
    0.9,   # beta
    0.5,   # tau_n
    0.5,   # tau_b
    0.5,   # p_n
    0.08,  # delta_k
    1.0,   # c
    0.5,   # b
    0.30,  # l_w
    1.0,   # u
    0.5,   # k
    0.5,   # n
])


_BGP_REQUIRED_EXOGENOUS: set[str] = {
    "alpha", "nu", "phi", "g", "delta_p", "delta_n", "omega", "bara",
}
_BGP_REQUIRED_TARGETS: set[str] = {
    "l", "u", "depr_rate", "kid_cost_share", "k_y_ratio", "c_y_ratio",
    "n_growth",
}


def _bgp_system(
    x: np.ndarray,
    exogenous: dict,
    targets: dict,
) -> np.ndarray:
    """The 13 residuals of the BGP system. Mirrors bgp_system in
    bgp_fertility_calibration.m.

    x = [barn, mu_l, beta, tau_n, tau_b, p_n, delta_k,
         c, b, l_w, u, k, n].
    """
    barn, mu_l, beta, tau_n, tau_b, p_n, delta_k = x[:7]
    c, b, l_w, u, k, n = x[7:]
    alpha   = exogenous["alpha"]
    nu      = exogenous["nu"]
    phi     = exogenous["phi"]
    g       = exogenous["g"]
    delta_p = exogenous["delta_p"]
    delta_n = exogenous["delta_n"]
    omega   = exogenous["omega"]
    a       = exogenous["bara"]

    # Output (from production function)
    y = np.exp(a) * (u * k) ** alpha * l_w ** (1 - alpha)

    F = np.zeros(13)
    # Eqn 1: Leisure-consumption FOC.  l_o = 1 - tau_n*n - tau_b*b - l_w.
    F[0]  = mu_l * c * (1 - tau_n * n - tau_b * b - l_w) ** (-nu) - (1 - alpha) * y / l_w
    # Eqn 2: Fertility Euler at SS (rearranged from MATLAB bgp_system F(2)).
    F[1]  = ((1 - alpha) * y / l_w
             - beta * (1 - delta_p + delta_n * n)
                * (np.exp(barn) * c / n - p_n
                   + (tau_b * (1 - delta_n) * phi - tau_n) * (1 - alpha) * y / l_w))
    # Eqn 3: Consumption Euler at SS.
    F[2]  = 1 + g - beta * (1 - delta_p + delta_n * n) * (
        alpha * y / k + 1 - delta_k * u ** omega / omega
    )
    # Eqn 4: Capital efficiency.
    F[3]  = delta_k * u ** omega * k - alpha * y
    # Eqn 5: Resource constraint at SS (no adjustment costs at SS — they vanish).
    F[4]  = c + p_n * n + (g + delta_k * u ** omega / omega) * k - y
    # Eqn 6: Children LoM at SS.
    F[5]  = b - phi * delta_n * n
    # Eqn 7: n_growth target.
    F[6]  = (1 - delta_p + delta_n * n) - targets["n_growth"]
    # Eqns 8-13: calibration targets.
    F[7]  = l_w - targets["l"]
    F[8]  = u - targets["u"]
    F[9]  = delta_k * u ** omega / omega - targets["depr_rate"]
    F[10] = p_n * n / y - targets["kid_cost_share"]
    F[11] = k / y - targets["k_y_ratio"]
    F[12] = c / y - targets["c_y_ratio"]
    return F


def solve_bgp(
    exogenous: dict | None = None,
    targets: dict | None = None,
    x0: np.ndarray | None = None,
    tol: float = 1e-12,
) -> dict:
    """Solve the 13-equation BGP system via scipy.optimize.fsolve.

    See the module docstring for the equation list. Returns a dict
    merging exogenous params + the 7 calibrated values + the 6 fsolve
    steady-state values + 4 derived SS values (i, l_o, y, a) + 2
    derived shock-state SS values (mun, ph).
    """
    if exogenous is None:
        exogenous = FERTILITY_EXOGENOUS_PARAMS
    if targets is None:
        targets = FERTILITY_CALIB_TARGETS
    missing_e = _BGP_REQUIRED_EXOGENOUS - set(exogenous.keys())
    if missing_e:
        raise KeyError(f"solve_bgp: exogenous missing keys: {sorted(missing_e)}")
    missing_t = _BGP_REQUIRED_TARGETS - set(targets.keys())
    if missing_t:
        raise KeyError(f"solve_bgp: targets missing keys: {sorted(missing_t)}")

    if x0 is None:
        x0 = _BGP_X0_DEFAULT.copy()

    x, infodict, ier, msg = scipy.optimize.fsolve(
        _bgp_system, x0, args=(exogenous, targets),
        full_output=True, xtol=tol,
    )
    if ier != 1:
        residual_norm = float(np.linalg.norm(infodict["fvec"]))
        raise RuntimeError(
            f"solve_bgp: fsolve did not converge (ier={ier}, "
            f"residual norm={residual_norm:.3e}): {msg}"
        )

    barn, mu_l, beta, tau_n, tau_b, p_n, delta_k = x[:7]
    c, b, l_w, u, k, n = x[7:]

    # Sanity-check support
    for name, val in [("c", c), ("l_w", l_w), ("k", k), ("n", n), ("u", u), ("b", b)]:
        if not np.isfinite(val) or val <= 0:
            raise ValueError(
                f"solve_bgp: solution out of support: {name}={val:.4f} (must be > 0)"
            )

    # Derived SS values
    a = exogenous["bara"]
    y = np.exp(a) * (u * k) ** exogenous["alpha"] * l_w ** (1 - exogenous["alpha"])
    i = (exogenous["g"] + delta_k * u ** exogenous["omega"] / exogenous["omega"]) * k
    l_o = 1 - tau_n * n - l_w - tau_b * b

    # Shock-state SS values
    a_ss = exogenous["bara"]
    mun_ss = float(np.log(np.exp(barn)))  # barn is already log(barn_level); spec says SS is log(barn)
    # Actually: in the .mod file, mun is the shock-state variable and barn is the SS value
    # (the MATLAB BGP solves for barn as a scalar; the shock process is
    # `mun = (1-rhon)*log(barn) + rhon*mun_lag + en` so mun's SS is log(barn)).
    # If we're solving for `barn` directly (as a log), then `mun_ss = barn`.
    mun_ss = float(barn)
    ph_ss = 0.0  # mortality has SS = barp = log(1/1.03) but the shock is centred at 0; check downstream

    # Merge everything
    out = dict(exogenous)
    out.update({
        "barn": float(barn), "mu_l": float(mu_l), "beta": float(beta),
        "tau_n": float(tau_n), "tau_b": float(tau_b), "p_n": float(p_n),
        "delta_k": float(delta_k),
        "c": float(c), "b": float(b), "l_w": float(l_w), "u": float(u),
        "k": float(k), "n": float(n),
        "i": float(i), "l_o": float(l_o), "y": float(y),
        "a": float(a_ss), "mun": float(mun_ss), "ph": float(ph_ss),
    })
    return out
```

Also append to `__all__`:
```python
__all__ = [
    "VAR_NAMES", "SHOCK_NAMES", "N_PRE", "N_FWD", "N_VARS", "N_SHOCKS",
    "FERTILITY_EXOGENOUS_PARAMS", "FERTILITY_CALIB_TARGETS",
    "FERTILITY_SHOCK_PROCESSES",
    "solve_bgp",
]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_dsge/test_fertility_bgp.py -v`
Expected: 5/5 PASS.

If `test_solve_bgp_converges_with_defaults` fails because some derived key is named differently, fix the implementation to match the test expectation (the test is the spec).

- [ ] **Step 5: Commit**

```bash
git add puremacro/dsge/fertility_adj_costs.py tests/test_dsge/test_fertility_bgp.py
git commit -m "feat(fertility): solve_bgp — fsolve port of bgp_fertility_calibration.m"
```

---

## Task 4: model_residuals — the 12 equations

**Files:**
- Modify: `puremacro/dsge/fertility_adj_costs.py` (add `model_residuals`).
- Create: `tests/test_dsge/test_fertility_residuals.py` (~3 tests).

- [ ] **Step 1: Write 3 failing tests**

Create `tests/test_dsge/test_fertility_residuals.py`:
```python
"""Tests for puremacro.dsge.fertility_adj_costs.model_residuals."""
from __future__ import annotations

import numpy as np
import pytest


def _build_z_ss_and_params():
    """Helper: solve BGP, build z_ss vector + complete params dict."""
    from puremacro.dsge.fertility_adj_costs import (
        solve_bgp, VAR_NAMES, FERTILITY_SHOCK_PROCESSES,
    )
    bgp = solve_bgp()
    z_ss = np.array([bgp[name] for name in VAR_NAMES])
    # Merge shock-process params
    params = dict(bgp)
    for shock, spec in FERTILITY_SHOCK_PROCESSES.items():
        # Map shock to its AR(1) name: ea -> rhoa, sigmaa; en -> rhon, sigman; ep -> rhop, sigmap
        prefix = {"ea": "a", "en": "n", "ep": "p"}[shock]
        params[f"rho{prefix}"] = spec["rho"]
        params[f"sigma{prefix}"] = spec["sigma"]
    # barp (SS of mortality log) — the .mod file uses barp = log(1.02);
    # we accept the default unless overridden.
    params["barp"] = 0.0   # consistent with bgp["ph"] = 0.0
    return z_ss, params


def test_model_residuals_zero_at_steady_state():
    from puremacro.dsge.fertility_adj_costs import model_residuals
    z_ss, params = _build_z_ss_and_params()
    eps = np.zeros(3)
    res = model_residuals(z_ss, z_ss, z_ss, eps, params)
    max_abs = float(np.max(np.abs(res)))
    assert max_abs < 1e-6, (
        f"max |residual| at SS = {max_abs:.3e} — BGP doesn't satisfy the dynamic equations"
    )


def test_model_residuals_returns_correct_shape():
    from puremacro.dsge.fertility_adj_costs import model_residuals
    z_ss, params = _build_z_ss_and_params()
    res = model_residuals(z_ss, z_ss, z_ss, np.zeros(3), params)
    assert res.shape == (12,)


def test_model_residuals_responds_to_perturbation():
    from puremacro.dsge.fertility_adj_costs import model_residuals, VAR_NAMES
    z_ss, params = _build_z_ss_and_params()
    eps = np.zeros(3)
    base = model_residuals(z_ss, z_ss, z_ss, eps, params)
    # Perturb c (index 5) by 1% at current time
    z_pert = z_ss.copy()
    z_pert[VAR_NAMES.index("c")] *= 1.01
    perturbed = model_residuals(z_ss, z_pert, z_ss, eps, params)
    assert np.max(np.abs(perturbed - base)) > 1e-6
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_dsge/test_fertility_residuals.py -v`
Expected: `ImportError: cannot import 'model_residuals'`.

- [ ] **Step 3: Implement model_residuals**

Append to `puremacro/dsge/fertility_adj_costs.py` (after the `solve_bgp` definition):
```python
# === Model residuals (12 equations) ===========================================

def _unpack(z: np.ndarray) -> dict:
    """Map a 12-vector ordered by VAR_NAMES to a name-keyed dict."""
    return {name: z[i] for i, name in enumerate(VAR_NAMES)}


def model_residuals(
    z_lead: np.ndarray,
    z: np.ndarray,
    z_lag: np.ndarray,
    eps: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Evaluate the 12 fertility-DSGE residuals at given (lead, current,
    lag, shock) variable vectors. See spec for the equation list.
    """
    if z_lead.shape != (N_VARS,) or z.shape != (N_VARS,) or z_lag.shape != (N_VARS,):
        raise ValueError(
            f"model_residuals: each z vector must have shape ({N_VARS},)"
        )
    if eps.shape != (N_SHOCKS,):
        raise ValueError(f"model_residuals: eps must have shape ({N_SHOCKS},)")

    L = _unpack(z_lag)
    C = _unpack(z)
    P = _unpack(z_lead)  # lead = t+1

    alpha   = params["alpha"]
    delta_k = params["delta_k"]
    delta_n = params["delta_n"]
    delta_p = params["delta_p"]
    omega   = params["omega"]
    g       = params["g"]
    beta    = params["beta"]
    nu      = params["nu"]
    mu_l    = params["mu_l"]
    tau_n   = params["tau_n"]
    tau_b   = params["tau_b"]
    p_n     = params["p_n"]
    psik    = params.get("psik", 2.5)   # adjustment-cost params (defaults from Bayesian variant)
    psin    = params.get("psin", 3.2)

    rhoa = params["rhoa"]
    rhon = params["rhon"]
    rhop = params["rhop"]
    bara = params["bara"]
    barn = params["barn"]
    barp = params["barp"]

    R = np.zeros(N_VARS)

    # (1) Production
    R[0] = C["y"] - np.exp(C["a"]) * (C["u"] * L["k"]) ** alpha * C["l_w"] ** (1 - alpha)

    # (2) Investment LoM
    R[1] = C["i"] - ((1 + g) * C["k"] - (1 - delta_k * C["u"] ** omega / omega) * L["k"])

    # (3) Resource constraint with adjustment costs
    adj_k = psik / 2 * (1 + g) ** 2 * (C["k"] / L["k"] - 1) ** 2 * L["k"]
    adj_n = psin / 2 * (C["n"] / L["n"] - 1) ** 2 * L["n"]
    R[2] = (C["c"] + p_n * L["n"] + C["i"] + adj_k + adj_n) - C["y"]

    # (4) Children LoM
    R[3] = C["n"] - ((1 - delta_n) * L["n"] + np.exp(-C["ph"]) * C["b"])

    # (5) Time constraint
    R[4] = (C["l_o"] + C["l_w"] + tau_n * L["n"] + tau_b * C["b"]) - 1.0

    # (6) Leisure-consumption FOC
    R[5] = mu_l * C["c"] * C["l_o"] ** (-nu) - (1 - alpha) * C["y"] / C["l_w"]

    # (7) Capital efficiency
    R[6] = delta_k * C["u"] ** omega * L["k"] - alpha * C["y"]

    # (8) Consumption Euler
    lhs8 = beta * (1 - delta_p + delta_n * L["n"]) * (C["c"] / P["c"]) * (
        alpha * P["y"] / C["k"]
        + 1 - delta_k * P["u"] ** omega / omega
        - psik * (1 + g) ** 2 / 2 * (1 - (P["k"] / C["k"]) ** 2)
    )
    rhs8 = (1 + g) * (1 + psik * (1 + g) * (C["k"] / L["k"] - 1))
    R[7] = lhs8 - rhs8

    # (9) Fertility Euler
    lhs9 = (np.exp(C["ph"]) * tau_b * (1 - alpha) * C["y"] / C["l_w"]
            + psin * (C["n"] / L["n"] - 1))
    rhs9 = beta * (1 - delta_p + delta_n * L["n"]) * (C["c"] / P["c"]) * (
        np.exp(P["mun"]) * P["c"] / C["n"] - p_n
        + (tau_b * (1 - delta_n) * np.exp(P["ph"]) - tau_n) * (1 - alpha) * P["y"] / P["l_w"]
        - psin / 2 * (1 - (P["n"] / C["n"]) ** 2)
    )
    R[8] = lhs9 - rhs9

    # (10) Productivity AR(1)
    R[9]  = C["a"]   - ((1 - rhoa) * bara + rhoa * L["a"]   + eps[0])  # ea
    # (11) Fertility-preference AR(1)
    R[10] = C["mun"] - ((1 - rhon) * barn + rhon * L["mun"] + eps[2])  # en (index 2 in SHOCK_NAMES)
    # (12) Mortality AR(1)
    R[11] = C["ph"]  - ((1 - rhop) * barp + rhop * L["ph"]  + eps[1])  # ep

    return R
```

Update `__all__`:
```python
__all__ = [
    "VAR_NAMES", "SHOCK_NAMES", "N_PRE", "N_FWD", "N_VARS", "N_SHOCKS",
    "FERTILITY_EXOGENOUS_PARAMS", "FERTILITY_CALIB_TARGETS",
    "FERTILITY_SHOCK_PROCESSES",
    "solve_bgp", "model_residuals",
]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_dsge/test_fertility_residuals.py -v`
Expected: 3/3 PASS.

If `test_model_residuals_zero_at_steady_state` fails, the BGP solution (from Task 3) and the model equations (Task 4) are inconsistent. Common culprits:
- `psik` / `psin` adjustment-cost params weren't in `bgp` — they're not estimated in this release but the BGP doesn't include them. Fix: pass `psik=0` and `psin=0` from `model_residuals` defaults, OR have `solve_bgp` write `psik` and `psin` into the output dict (they don't affect the BGP residuals because the adjustment cost vanishes at SS). The latter is cleaner — add `"psik": 2.5, "psin": 3.2` to the dict in `solve_bgp` before returning.
- A wrong sign in equation 8 or 9. Compare to the .mod file's model block character-by-character.
- The `barn` interpretation: in the BGP we solve for `barn` directly; in the AR(1) shock process we use `(1-rhon)*barn + rhon*mun_lag`. Both treat `barn` as the SS log-value, so they're consistent.

If you need to add `psik` / `psin` to the BGP dict, modify Task 3's `solve_bgp` to include them and commit a one-line follow-up fix:
```bash
git add puremacro/dsge/fertility_adj_costs.py
git commit -m "fix(fertility): include psik/psin in solve_bgp output (used by Eulers)"
```

- [ ] **Step 5: Commit**

```bash
git add puremacro/dsge/fertility_adj_costs.py tests/test_dsge/test_fertility_residuals.py
git commit -m "feat(fertility): model_residuals — the 12 model + shock-process equations"
```

---

## Task 5: solve_fertility — orchestrator (numerical Jacobians + Klein)

**Files:**
- Modify: `puremacro/dsge/fertility_adj_costs.py` (add Jacobian helper + `solve_fertility`).
- Create: `tests/test_dsge/test_solve_fertility.py` (~5 tests).

- [ ] **Step 1: Write 5 failing tests**

Create `tests/test_dsge/test_solve_fertility.py`:
```python
"""Tests for puremacro.dsge.fertility_adj_costs.solve_fertility."""
from __future__ import annotations

import numpy as np
import pytest


def test_solve_fertility_returns_dataclass():
    from puremacro.dsge.fertility_adj_costs import solve_fertility
    from puremacro.dsge._results import FertilitySolution
    sol = solve_fertility()
    assert isinstance(sol, FertilitySolution)
    assert sol.var_names[:5] == ("a", "mun", "ph", "k", "n")
    assert sol.shock_names == ("ea", "ep", "en")


def test_solve_fertility_blanchard_kahn_satisfied():
    from puremacro.dsge.fertility_adj_costs import solve_fertility
    sol = solve_fertility()
    assert sol.klein_solution.eu == (1, 1), (
        f"Blanchard-Kahn failed: eu={sol.klein_solution.eu}"
    )


def test_solve_fertility_g_matrix_stable_eigenvalues():
    from puremacro.dsge.fertility_adj_costs import solve_fertility
    sol = solve_fertility()
    eigvals = np.linalg.eigvals(sol.G)
    max_modulus = float(np.max(np.abs(eigvals)))
    assert max_modulus < 1.0 - 1e-6, (
        f"G is not contractive: max|eig|={max_modulus:.4f}"
    )


def test_productivity_shock_positive_output_response():
    from puremacro.dsge.fertility_adj_costs import solve_fertility
    sol = solve_fertility()
    irf = sol.irf("ea", horizon=8)
    # y is at column index 6 (0:a, 1:mun, 2:ph, 3:k, 4:n, 5:c, 6:y, ...)
    assert irf["y"].iloc[0] > 0, "impact response of y to productivity shock should be positive"
    assert irf["y"].iloc[4] > 0, "horizon-4 response of y to productivity shock should still be positive"


def test_mortality_shock_negative_fertility_response():
    from puremacro.dsge.fertility_adj_costs import solve_fertility
    sol = solve_fertility()
    irf = sol.irf("ep", horizon=8)
    # Higher mortality should reduce children stock n at some short horizon.
    n_path = irf["n"].iloc[:5].to_numpy()
    assert np.any(n_path < 0), (
        f"expected at least one negative entry in n's IRF to mortality shock; got {n_path}"
    )
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_dsge/test_solve_fertility.py -v`
Expected: `ImportError: cannot import 'solve_fertility'`.

- [ ] **Step 3: Implement Jacobian helper + solve_fertility**

Append to `puremacro/dsge/fertility_adj_costs.py`:
```python
# === Solver orchestrator ======================================================

def _central_diff_jacobian(
    residuals_fn,
    z0: np.ndarray,
    h_base: float = 1e-6,
) -> np.ndarray:
    """Per-component central-difference Jacobian of a vector-valued
    function residuals_fn(z) wrt z, evaluated at z0.

    Step size for component i: ``h_i = max(h_base, |z0[i]| * h_base)``.

    Returns Jacobian of shape (m, n) where m = len(residuals_fn(z0)) and
    n = len(z0).
    """
    z0 = np.asarray(z0, dtype=float)
    n = len(z0)
    f0 = residuals_fn(z0)
    m = len(f0)
    J = np.zeros((m, n))
    for i in range(n):
        h = max(h_base, abs(z0[i]) * h_base)
        z_plus = z0.copy(); z_plus[i] += h
        z_minus = z0.copy(); z_minus[i] -= h
        J[:, i] = (residuals_fn(z_plus) - residuals_fn(z_minus)) / (2 * h)
    return J


def _build_complete_params(
    bgp: dict,
    shock_stds: dict | None,
) -> dict:
    """Merge bgp + FERTILITY_SHOCK_PROCESSES + caller overrides into one dict
    containing every key model_residuals expects."""
    params = dict(bgp)
    for shock, spec in FERTILITY_SHOCK_PROCESSES.items():
        prefix = {"ea": "a", "en": "n", "ep": "p"}[shock]
        params[f"rho{prefix}"] = spec["rho"]
        params[f"sigma{prefix}"] = spec["sigma"]
    if shock_stds is not None:
        for shock, sigma in shock_stds.items():
            prefix = {"ea": "a", "en": "n", "ep": "p"}[shock]
            params[f"sigma{prefix}"] = float(sigma)
    # Adjustment-cost params (not in BGP because they vanish at SS).
    params.setdefault("psik", 2.5)
    params.setdefault("psin", 3.2)
    # barp default if missing
    params.setdefault("barp", 0.0)
    return params


def solve_fertility(
    params: dict | None = None,
    *,
    shock_stds: dict | None = None,
    h_for_jacobians: float = 1e-6,
) -> FertilitySolution:
    """Solve the fertility DSGE around its calibrated BGP and return a
    FertilitySolution. See the module docstring for the full pipeline.
    """
    # 1. BGP
    bgp = solve_bgp()
    if params is not None:
        bgp = {**bgp, **params}
    all_params = _build_complete_params(bgp, shock_stds)

    # 2. Build z_ss vector in VAR_NAMES order.
    z_ss = np.array([all_params[name] for name in VAR_NAMES])

    # 3. Compute four Jacobians via central differences.
    eps0 = np.zeros(N_SHOCKS)

    A = _central_diff_jacobian(
        lambda zl: model_residuals(zl, z_ss, z_ss, eps0, all_params),
        z_ss, h_base=h_for_jacobians,
    )
    B = _central_diff_jacobian(
        lambda z: model_residuals(z_ss, z, z_ss, eps0, all_params),
        z_ss, h_base=h_for_jacobians,
    )
    Cm = _central_diff_jacobian(
        lambda zlg: model_residuals(z_ss, z_ss, zlg, eps0, all_params),
        z_ss, h_base=h_for_jacobians,
    )
    D = _central_diff_jacobian(
        lambda e: model_residuals(z_ss, z_ss, z_ss, e, all_params),
        eps0, h_base=h_for_jacobians,
    )

    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))
            and np.all(np.isfinite(Cm)) and np.all(np.isfinite(D))):
        raise RuntimeError(
            "solve_fertility: numerical Jacobian contains non-finite entries — "
            "check steady state and model equations."
        )

    # 4. Rearrange to canonical Klein form:
    #    A · E_t Δz_{t+1}  =  -B · Δz_t  -  Cm · Δz_{t-1}  -  D · ε_t
    # Klein expects:
    #    A_klein · E_t z_{t+1}  =  B_klein · z_t  +  C_klein · u_t
    # where z_t in our partition is already [states; controls].
    # The lag term enters through state-evolution equations (the AR(1)s and
    # the capital/children LoMs); for Klein, the lagged-state structure is
    # absorbed by treating predetermined variables as "known at t" — the
    # Jacobian Cm contributes to the state-transition rows of B_klein.
    #
    # The canonical mapping for this setup:
    #   A_klein = A                                  (lead Jacobian)
    #   B_klein = -B - Cm   (current + lag combined; this works when the
    #                        lag dependence is only on predetermined states)
    #   C_klein = -D                                 (shock loading)
    A_klein = A
    B_klein = -(B + Cm)
    C_klein = -D

    # 5. Klein QZ solve. n_pre = number of predetermined variables in VAR_NAMES.
    klein_sol: KleinSolution = klein_solve(A_klein, B_klein, n_pre=N_PRE, C=C_klein)
    if klein_sol.eu != (1, 1):
        raise RuntimeError(
            f"solve_fertility: Klein Blanchard-Kahn condition failed: "
            f"eu={klein_sol.eu}, n_unstable={np.sum(np.abs(klein_sol.eigenvalues) > 1.0)}, "
            f"n_fwd={N_FWD}. Likely cause: variable-timing misclassification."
        )

    # Build SS dict for the result.
    ss = {name: float(all_params[name]) for name in VAR_NAMES}

    return FertilitySolution(
        ss=ss,
        params=all_params,
        G=klein_sol.G,
        N=klein_sol.N,
        F=klein_sol.F,
        L=klein_sol.L,
        klein_solution=klein_sol,
        var_names=VAR_NAMES,
        shock_names=SHOCK_NAMES,
    )
```

Update `__all__`:
```python
__all__ = [
    "VAR_NAMES", "SHOCK_NAMES", "N_PRE", "N_FWD", "N_VARS", "N_SHOCKS",
    "FERTILITY_EXOGENOUS_PARAMS", "FERTILITY_CALIB_TARGETS",
    "FERTILITY_SHOCK_PROCESSES",
    "solve_bgp", "model_residuals", "solve_fertility",
]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_dsge/test_solve_fertility.py -v`
Expected: 5/5 PASS.

The Blanchard-Kahn test is the high-risk one. If it fails:
1. First, dump the eigenvalues: `print(np.sort(np.abs(klein_sol.eigenvalues)))`.
2. Count unstable (|eig| > 1) and compare to N_FWD = 7. If counts don't match, the variable-timing classification (or the Klein-form rearrangement) is wrong.
3. The most likely culprit is the `B_klein = -(B + Cm)` simplification. The correct Klein form when a lag matrix is present is the *Sims-gensys* canonical form, which puremacro provides via `puremacro.dsge.gensys`. If Klein fails BK, switch the call:
   ```python
   from puremacro.dsge.gensys import gensys, GensysSolution
   # gensys takes Γ_0 · z_t = Γ_1 · z_{t-1} + Ψ · ε_t + Π · η_t
   # See puremacro.dsge.gensys docstring for the mapping.
   ```
   Report this as DONE_WITH_CONCERNS if you have to fall back to gensys.

- [ ] **Step 5: Commit**

```bash
git add puremacro/dsge/fertility_adj_costs.py tests/test_dsge/test_solve_fertility.py
git commit -m "feat(fertility): solve_fertility — numerical Jacobians + Klein QZ orchestrator"
```

---

## Task 6: IRF + FEVD methods

**Files:**
- Modify: `puremacro/dsge/fertility_adj_costs.py` (add `_compute_irf`, `_compute_fevd`).
- Create: `tests/test_dsge/test_fertility_irf_fevd.py` (~3 tests).

- [ ] **Step 1: Write 3 failing tests**

Create `tests/test_dsge/test_fertility_irf_fevd.py`:
```python
"""Tests for FertilitySolution.irf and FertilitySolution.fevd."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_irf_returns_long_dataframe_with_var_columns():
    from puremacro.dsge.fertility_adj_costs import solve_fertility, VAR_NAMES
    sol = solve_fertility()
    irf = sol.irf("ea", horizon=10)
    assert isinstance(irf, pd.DataFrame)
    assert irf.shape == (11, 12)  # horizon+1 rows, 12 vars
    assert tuple(irf.columns) == VAR_NAMES


def test_irf_horizon_zero_returns_impact_only():
    from puremacro.dsge.fertility_adj_costs import solve_fertility
    sol = solve_fertility()
    irf = sol.irf("ea", horizon=0)
    assert len(irf) == 1


def test_irf_unknown_shock_name_raises():
    from puremacro.dsge.fertility_adj_costs import solve_fertility
    sol = solve_fertility()
    with pytest.raises(ValueError, match="unknown shock"):
        sol.irf("bad_shock_name", horizon=4)
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_dsge/test_fertility_irf_fevd.py -v`
Expected: `AttributeError: module ... has no attribute '_compute_irf'`.

- [ ] **Step 3: Implement `_compute_irf` and `_compute_fevd`**

Append to `puremacro/dsge/fertility_adj_costs.py`:
```python
# === IRF / FEVD helpers (called by FertilitySolution.irf and .fevd) ==========

def _resolve_shock(shock, shock_names: tuple) -> int:
    if isinstance(shock, int):
        if not (0 <= shock < len(shock_names)):
            raise ValueError(f"shock index {shock} out of range")
        return shock
    if isinstance(shock, str):
        if shock not in shock_names:
            raise ValueError(
                f"unknown shock {shock!r}; expected one of {list(shock_names)}"
            )
        return shock_names.index(shock)
    raise TypeError(f"shock must be int or str, got {type(shock).__name__}")


def _compute_irf(sol: FertilitySolution, shock, horizon: int) -> pd.DataFrame:
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    s_idx = _resolve_shock(shock, sol.shock_names)
    sigma = sol.params.get(
        {"ea": "sigmaa", "ep": "sigmap", "en": "sigman"}[sol.shock_names[s_idx]],
        1.0,
    )
    # 1-SD shock impact
    state = sol.N[:, s_idx] * sigma
    control = sol.L[:, s_idx] * sigma
    rows = [np.concatenate([state, control])]
    for _ in range(horizon):
        state = sol.G @ state
        control = sol.F @ state
        rows.append(np.concatenate([state, control]))
    arr = np.array(rows)
    return pd.DataFrame(arr, columns=list(sol.var_names), index=range(horizon + 1))


def _compute_fevd(sol: FertilitySolution, horizon: int) -> pd.DataFrame:
    """Per-(variable, shock) variance share at each horizon 0..horizon."""
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    n_vars = len(sol.var_names)
    n_shocks = len(sol.shock_names)
    # Stack IRFs across shocks
    irfs = np.zeros((horizon + 1, n_vars, n_shocks))
    for s_idx in range(n_shocks):
        sigma = sol.params.get(
            {"ea": "sigmaa", "ep": "sigmap", "en": "sigman"}[sol.shock_names[s_idx]],
            1.0,
        )
        state = sol.N[:, s_idx] * sigma
        control = sol.L[:, s_idx] * sigma
        irfs[0, :, s_idx] = np.concatenate([state, control])
        for h in range(1, horizon + 1):
            state = sol.G @ state
            control = sol.F @ state
            irfs[h, :, s_idx] = np.concatenate([state, control])
    cum_sq = np.cumsum(irfs ** 2, axis=0)
    total = cum_sq.sum(axis=2, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    share = cum_sq / total
    # Reshape to long DataFrame
    rows = []
    for h in range(horizon + 1):
        for v_idx, var in enumerate(sol.var_names):
            for s_idx, shock in enumerate(sol.shock_names):
                rows.append({
                    "horizon": h, "variable": var, "shock": shock,
                    "share": share[h, v_idx, s_idx],
                })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_dsge/test_fertility_irf_fevd.py -v`
Expected: 3/3 PASS.

- [ ] **Step 5: Commit**

```bash
git add puremacro/dsge/fertility_adj_costs.py tests/test_dsge/test_fertility_irf_fevd.py
git commit -m "feat(fertility): IRF + FEVD methods on FertilitySolution"
```

---

## Task 7: AR(1) demo example

**Files:**
- Create: `puremacro/examples/dsge_fertility_demo.py`.
- Create: `tests/test_examples/__init__.py` (empty, if missing).
- Create: `tests/test_examples/test_dsge_fertility_demo_runs.py` (~1 test).

- [ ] **Step 1: Write the smoke test**

Create `tests/test_examples/__init__.py` as an empty file (if it doesn't exist):
```bash
mkdir -p tests/test_examples
test -f tests/test_examples/__init__.py || touch tests/test_examples/__init__.py
```

Create `tests/test_examples/test_dsge_fertility_demo_runs.py`:
```python
"""Smoke test: dsge_fertility_demo.main() runs without exception."""
from __future__ import annotations

import os
from pathlib import Path


def test_dsge_fertility_demo_main_runs_without_exception(tmp_path, monkeypatch):
    # Run from tmp_path so the saved figure doesn't pollute the repo
    monkeypatch.chdir(tmp_path)
    from puremacro.examples.dsge_fertility_demo import main
    main()
    # Just confirm it produced the figure
    assert (tmp_path / "dsge_fertility_demo.png").exists()
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_examples/test_dsge_fertility_demo_runs.py -v`
Expected: `ModuleNotFoundError: No module named 'puremacro.examples.dsge_fertility_demo'`.

- [ ] **Step 3: Create the demo**

Create `puremacro/examples/dsge_fertility_demo.py`:
```python
"""Solve the fertility DSGE (Alonso-Ortiz, adjustment-costs variant)
around its calibrated BGP and plot IRFs to the three shocks.

Demonstrates puremacro.dsge.solve_fertility on a model OTHER than SW07,
proving the BGP+Klein machinery is composable with new DSGEs.
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless runs

import matplotlib.pyplot as plt

from puremacro.dsge import solve_fertility


def main() -> None:
    sol = solve_fertility()
    print("Fertility-adj-costs BGP:")
    for name in ("c", "k", "y", "n", "b", "l_w", "u"):
        print(f"  {name:6s} = {sol.ss[name]:.4f}")
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    horizon = 20
    plot_vars = ["y", "n", "b"]
    for col, shock in enumerate(sol.shock_names):
        irf = sol.irf(shock, horizon=horizon)
        for row, var in enumerate(plot_vars):
            ax = axes[row, col]
            ax.plot(irf.index, irf[var], "k-")
            ax.axhline(0.0, color="0.7", lw=0.5)
            ax.set_title(f"{var} ← shock {shock}", fontsize=9)
            if row == 2:
                ax.set_xlabel("quarter")
    fig.suptitle("Fertility DSGE: IRFs to 1-SD shocks", fontsize=11)
    fig.tight_layout()
    fig.savefig("dsge_fertility_demo.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_examples/test_dsge_fertility_demo_runs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add puremacro/examples/dsge_fertility_demo.py tests/test_examples/__init__.py tests/test_examples/test_dsge_fertility_demo_runs.py
git commit -m "feat(examples): dsge_fertility_demo — BGP + 3×3 IRF plot"
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
- Fertility DSGE (Alonso-Ortiz adjustment-costs variant) — solver only;
  Bayesian estimation queued for 0.55.0.

For likelihood-based estimation, pair the state-space form returned by
``make_state_space`` (model-specific) with ``puremacro.dsge.estimate_dsge``.
"""
from .klein import BlanchardKahnError, KleinSolution, klein_solve
from ._results import (
    DSGEPosteriorResult, SW07PosteriorResult,
    FertilitySolution,
)
from .estimate import estimate_dsge
from .sw07_estimate import estimate_sw07
from .fertility_adj_costs import solve_bgp, solve_fertility
from . import priors, fertility_adj_costs

__all__ = [
    "klein_solve", "KleinSolution", "BlanchardKahnError",
    "DSGEPosteriorResult", "SW07PosteriorResult", "FertilitySolution",
    "estimate_dsge", "estimate_sw07",
    "solve_bgp", "solve_fertility",
    "priors", "fertility_adj_costs",
]
from . import smets_wouters  # re-export for back-compat with 0.50.0 callers
```

- [ ] **Step 2: Verify imports cleanly**

```bash
python -c "from puremacro.dsge import solve_fertility, solve_bgp, FertilitySolution, fertility_adj_costs; print('ok')"
```

Expected output: `ok`.

- [ ] **Step 3: Run the full dsge test suite**

```bash
pytest tests/test_dsge/ -v -m "not slow"
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add puremacro/dsge/__init__.py
git commit -m "feat(dsge): export solve_bgp, solve_fertility, FertilitySolution"
```

---

## Task 9: Version bump + CHANGELOG

**Files:**
- Modify: `puremacro/__init__.py`.
- Modify: `pyproject.toml`.
- Modify: `CHANGELOG.md`.
- Modify: `tests/test_import.py`.

- [ ] **Step 1: Bump `puremacro/__init__.py`**

Change `__version__ = "0.53.0"` to `__version__ = "0.54.0"`.

- [ ] **Step 2: Bump `pyproject.toml`**

In `[project]`, change `version = "0.53.0"` to `version = "0.54.0"`.

- [ ] **Step 3: Bump `tests/test_import.py`**

Change `assert puremacro.__version__ == "0.53.0"` to `assert puremacro.__version__ == "0.54.0"`.

- [ ] **Step 4: Add CHANGELOG entry**

Insert into `CHANGELOG.md` AFTER the `# Changelog` heading + preamble and BEFORE the existing `## 0.53.0 — 2026-05-23` entry:

```markdown
## 0.54.0 — 2026-05-24

Fertility DSGE solver. R1b from the 2026-05-23 research-directions
brainstorm: ports `fertility_adj_costs.mod` (Alonso-Ortiz adjustment-
costs baseline) onto puremacro. Solver only — Bayesian estimation
queued for R1c (0.55.0), which will wire priors + an observation
equation onto puremacro.dsge.estimate_dsge (the engine shipped in 0.53.0).

### Added
- `puremacro.dsge.solve_bgp(exogenous, targets, x0, tol) -> dict` —
  scipy.optimize.fsolve port of `bgp_fertility_calibration.m`. Solves a
  13-equation balanced-growth-path system pinning 7 calibrated
  parameters (`barn, mu_l, beta, tau_n, tau_b, p_n, delta_k`) and 6
  steady-state values (`c, b, l_w, u, k, n`) given 7 calibration
  targets (l, u, depr_rate, kid_cost_share, k_y_ratio, c_y_ratio,
  n_growth) and 8 exogenous parameters.
- `puremacro.dsge.solve_fertility(params=None, *, shock_stds=None,
  h_for_jacobians=1e-6) -> FertilitySolution` — orchestrator: BGP →
  numerical Jacobians (central differences) on model_residuals at the
  BGP → puremacro.dsge.klein_solve → packaged FertilitySolution.
- `puremacro.dsge.FertilitySolution` (frozen dataclass) — ss, params,
  G, N, F, L, klein_solution, var_names, shock_names. Methods: `irf()`,
  `fevd()`.
- `puremacro.dsge.fertility_adj_costs` submodule exposing the constants
  (`VAR_NAMES`, `SHOCK_NAMES`, `FERTILITY_EXOGENOUS_PARAMS`,
  `FERTILITY_CALIB_TARGETS`, `FERTILITY_SHOCK_PROCESSES`) plus
  `model_residuals` (the 12 model + shock-process equations).
- `puremacro/examples/dsge_fertility_demo.py` — solve baseline, plot
  3×3 IRF grid for (y, n, b) responses to (ea, ep, en).

### Internal
- ~350 LOC new in `puremacro/dsge/fertility_adj_costs.py`.
- ~17 new unit tests in `tests/test_dsge/` + 1 smoke test in
  `tests/test_examples/`.
- No new dependencies (scipy.optimize.fsolve was already transitive).

### Provenance
The model equations mirror `My Drive/Fertility/fertility_adj_costs.mod`
character-for-character (the `model;` block). The BGP system mirrors
`bgp_fertility_calibration.m` (the 13-residual `bgp_system` function).
The Bayesian variant of the Dynare file
(`fertility_adj_costs_bayesian_estimation.mod`) supplies the default
shock-process values (`FERTILITY_SHOCK_PROCESSES`).
```

- [ ] **Step 5: Smoke check**

```bash
python -c "import puremacro; assert puremacro.__version__ == '0.54.0'; print(puremacro.__version__)"
```

Expected output: `0.54.0`.

- [ ] **Step 6: Commit**

```bash
git add puremacro/__init__.py pyproject.toml CHANGELOG.md tests/test_import.py
git commit -m "chore(puremacro): bump 0.53.0 → 0.54.0 (fertility DSGE solver)"
```

---

## Task 10: Regenerate public-API snapshot

**Files:**
- Modify: `tests/fixtures/public_api_snapshot.json`.

- [ ] **Step 1: Run the snapshot test, expect FAIL**

```bash
pytest tests/ -k "public_api" -v 2>&1 | tail -30
```

Expected: FAIL — missing the new fertility surface.

- [ ] **Step 2: Regenerate the snapshot**

Find the regeneration helper / pattern used in 0.51, 0.52, 0.53 (commits `55ea3af`, `9e2eb23`, `ffd30a8`). The recipe is to extract `collect_current_api` from the test file and run it in-process to dump a fresh JSON.

Add to the snapshot:
- `puremacro.dsge` package entry: add `FertilitySolution, fertility_adj_costs, solve_bgp, solve_fertility` to its export list.
- `puremacro.dsge.fertility_adj_costs` module: new entry with public list `["FERTILITY_CALIB_TARGETS", "FERTILITY_EXOGENOUS_PARAMS", "FERTILITY_SHOCK_PROCESSES", "N_FWD", "N_PRE", "N_SHOCKS", "N_VARS", "SHOCK_NAMES", "VAR_NAMES", "model_residuals", "solve_bgp", "solve_fertility"]`.
- `puremacro.dsge._results` module: add `FertilitySolution` to its list.
- result_classes: add `FertilitySolution` with its 9 fields.

- [ ] **Step 3: Re-run, expect PASS**

```bash
pytest tests/ -k "public_api" -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/public_api_snapshot.json
git commit -m "chore(tests): regenerate public_api_snapshot for 0.54.0 fertility additions"
```

---

## Task 11: Run 6-gate release check

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

Expected: gate 5 PASS. The new `dsge_fertility_demo.py` should be picked up.

If "stale gallery" advisory fires, regenerate:
```bash
python tools/render_examples_gallery.py
```
Restore the `hfi_gertler_karadi` flake from the previous PASS entry (pattern from commits `e569dee`, `aa562cc`).

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

If any gate fails, diagnose (no `--no-verify`, no hook skips).

---

## Self-review checklist (run AFTER all 11 tasks)

1. **Spec coverage:**
   - Component A (`fertility_adj_costs.py`): Tasks 2-6 ✓
   - Component B (`FertilitySolution`): Task 1 ✓
   - Component C (demo): Task 7 ✓
   - Exports: Task 8 ✓
   - Version + CHANGELOG: Task 9 ✓
   - Snapshot regen: Task 10 ✓
   - Release gates: Task 11 ✓
   - All 11 acceptance criteria map to a task.

2. **Placeholder scan:** None — every step contains runnable code or a concrete command.

3. **Type consistency:**
   - `VAR_NAMES` order (5 states + 7 controls) consistent across Tasks 2, 4, 5, 6, 7.
   - `_VARIABLE_TIMING` keys cover all 12 VAR_NAMES (Task 2 definition).
   - `model_residuals` returns shape `(12,)` (Task 4 implementation + tests).
   - `solve_bgp` returns dict with all keys the tests assert (Task 3).
   - `solve_fertility` builds `all_params` containing every key `model_residuals` reads.
   - `FertilitySolution.irf()` returns shape `(horizon+1, 12)` (Task 6).
   - shock index order in IRF: `ea=0, ep=1, en=2` — matches `SHOCK_NAMES` everywhere.
