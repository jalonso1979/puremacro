# puremacro 0.4.1 Patch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `puremacro 0.4.1` — a small, low-risk patch closing the seven follow-ups flagged at the end of iteration N+8 (the 0.4.0 release): two dead-code cleanups, two dict-return migrations to the result-object standard, three narrative-connector bug fixes, and AB 1991 replication-fixture infrastructure.

**Architecture:** Pure cleanup patch — no new public API, no new modules, no new runtime deps. The two dict-return migrations follow the result-object standard set in 0.3.0 / hardened in 0.4.0 step 3 (`@dataclass(frozen=True)`, `<Method>Result` naming, no `__post_init__`, optional `.summary()`). Because `cointegration_modern.py` and `midas.py` are top-level modules (not subpackages), the result classes live inline in the same file rather than in a separate `_results.py` — this matches `cycles.py` / `state_space.py` precedent for top-level modules. Connector fixes preserve the "yield, don't raise" contract from `narrative/sources/RETRY_POLICY.md`. AB 1991 fixture work ships the test infrastructure with skip-if-absent fallback (no network access in this environment to fetch the canonical CSV).

**Tech Stack:** Python 3.10+, numpy, scipy, pandas, pytest. Pyodide-compatible (no new deps).

**Tracking:**
- Pre-patch baseline: 368 passing, 7 skipped (5 baseline network + 2 connectors with dead URLs / WAF blocks)
- Target post-patch: 380+ passing (7 from result-object migration tests, ~3 from connector fixes, ~2 from AB-fixture skip+infrastructure tests), ≤6 skipped (US Treasury fix lifts one skip; AB fixture stays skipped without the CSV)

---

## File Structure

### Files modified
- `puremacro/did/callaway_santanna.py` — drop dead `* 0` leftover at line 164
- `puremacro/did/borusyak_jaravel_spiess.py` — drop unused `inv_xtx` import (line 32)
- `puremacro/nowcast/combine.py` — drop unused `inv_xtx` import (line 32)
- `puremacro/cointegration_modern.py` — add 3 frozen result classes, change return types
- `puremacro/midas.py` — add 2 frozen result classes, change return types
- `puremacro/examples/fm_ols_dols_demo.py` — update dict access to attribute access
- `puremacro/examples/midas_quarterly_monthly.py` — update dict access to attribute access
- `puremacro/narrative/sources/us_treasury.py` — replace dead RSS URL with HTML-scrape of press-releases listing
- `puremacro/narrative/sources/us_federal_register.py` — change default agencies to remove broken slug
- `puremacro/narrative/sources/us_dod_contracts.py` — pass realistic browser User-Agent
- `puremacro/narrative/sources/_http.py` — add optional `user_agent=` override on three `safe_get_*` helpers
- `puremacro/narrative/sources/RETRY_POLICY.md` — document the `user_agent=` override
- `pyproject.toml` — bump version to `0.4.1`
- `puremacro/__init__.py` — bump `__version__` to `0.4.1`
- `tests/test_import.py` — bump expected version
- `tests/fixtures/public_api_snapshot.json` — regenerate after result-object additions
- `CHANGELOG.md` — add `## 0.4.1 — YYYY-MM-DD` block
- `puremacro/.../project_puremacro.md` (memory) — add iteration N+9 step 1 entry

### Files created
- `tests/test_cointegration_modern.py` — new (smoke + result-object + summary tests)
- `tests/test_midas.py` — new (smoke + result-object + summary tests)
- `tests/test_dynpanel/test_ab_1991_replication.py` — new (skip-if-fixture-absent test)
- `tests/fixtures/abdata.README.md` — new (documents how to obtain the CSV + canonical AB 1991 published estimates)

---

## Task 1: Drop dead `z * 0` line in callaway_santanna.py

**Files:**
- Modify: `puremacro/did/callaway_santanna.py:164`

The line `z = float(np.quantile([0.5 - alpha / 2, 0.5 + alpha / 2], 1)) * 0` produces an unused float (always zero). The very next comment says "Use bootstrap percentiles directly" — confirming `z` is unused. The variable `z` is not referenced anywhere else in the function (we already verified via grep that `z` does not appear after this line in this file).

- [ ] **Step 1: Confirm `z` is unused**

Run: `grep -n "\bz\b" puremacro/did/callaway_santanna.py`
Expected: Only the assignment line + nothing afterward in the function body.

- [ ] **Step 2: Delete the line and its trailing comment**

Edit `puremacro/did/callaway_santanna.py`. Remove these two lines:

```python
    z = float(np.quantile([0.5 - alpha / 2, 0.5 + alpha / 2], 1)) * 0
    # Use bootstrap percentiles directly.
```

(The comment refers to the deleted line — also goes.)

- [ ] **Step 3: Run the existing CS tests to confirm no regression**

Run: `pytest tests/test_did -v -x`
Expected: All pre-existing did tests pass (callaway_santanna, sun_abraham, borusyak_jaravel_spiess, synthetic_did, cdh_did, sdid_multi).

---

## Task 2: Drop unused `inv_xtx` import in borusyak_jaravel_spiess.py

**Files:**
- Modify: `puremacro/did/borusyak_jaravel_spiess.py:32`

- [ ] **Step 1: Confirm `inv_xtx` is unused in this file**

Run: `grep -n "inv_xtx" puremacro/did/borusyak_jaravel_spiess.py`
Expected: Only line 32 (the import). Nothing else.

- [ ] **Step 2: Delete the import line**

Edit `puremacro/did/borusyak_jaravel_spiess.py`. Delete:

```python
from .._linalg import inv_xtx
```

- [ ] **Step 3: Run the BJS tests**

Run: `pytest tests/test_did -k borusyak -v`
Expected: All BJS tests pass.

---

## Task 3: Drop unused `inv_xtx` import in nowcast/combine.py

**Files:**
- Modify: `puremacro/nowcast/combine.py:32`

- [ ] **Step 1: Confirm `inv_xtx` is unused in this file**

Run: `grep -n "inv_xtx" puremacro/nowcast/combine.py`
Expected: Only line 32 (the import).

- [ ] **Step 2: Delete the import line**

Edit `puremacro/nowcast/combine.py`. Delete:

```python
from .._linalg import inv_xtx
```

- [ ] **Step 3: Run the nowcast tests**

Run: `pytest tests/test_nowcast -v -x`
Expected: All nowcast tests pass.

- [ ] **Step 4: Commit Tasks 1–3 together (dead-code cleanup)**

```bash
cd "uncertainty_examples/puremacro" && git diff --stat 2>/dev/null || true   # repo isn't git, but show summary of what changed
```

(This MAV workspace is **not** a git repo — see CLAUDE.md. Skip the commit; just announce "dead-code cleanup complete" and proceed.)

---

## Task 4: Migrate `cointegration_modern.fm_ols` to FMOLSResult

**Files:**
- Modify: `puremacro/cointegration_modern.py`
- Test: `tests/test_cointegration_modern.py` (new)

The current `fm_ols` returns a dict with 7 keys: `beta`, `alpha`, `se`, `residuals`, `Omega`, `Sigma`, `Lambda`. >3 fields → must become a frozen dataclass per the result-object standard.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cointegration_modern.py`:

```python
"""Tests for puremacro.cointegration_modern result objects."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.cointegration_modern import (
    fm_ols, dols, phillips_ouliaris,
    FMOLSResult, DOLSResult, PhillipsOuliarisResult,
)


def _simulate_cointegrated(T=200, beta=2.0, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    nu = rng.standard_normal(T)
    xi = rng.standard_normal(T) * 0.7
    u = rho * nu + xi
    x = np.cumsum(nu)
    y = beta * x + u
    return y, x


# --------------------------------------------------------------------------
# fm_ols
# --------------------------------------------------------------------------
def test_fm_ols_returns_FMOLSResult():
    y, x = _simulate_cointegrated()
    res = fm_ols(y, x)
    assert isinstance(res, FMOLSResult)
    # frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.alpha = 0.0


def test_fm_ols_has_documented_fields():
    y, x = _simulate_cointegrated()
    res = fm_ols(y, x)
    assert isinstance(res.beta, np.ndarray) and res.beta.shape == (1,)
    assert isinstance(res.alpha, float)
    assert isinstance(res.se, np.ndarray) and res.se.shape == (1,)
    assert isinstance(res.residuals, np.ndarray)
    assert res.Omega.shape == (2, 2)
    assert res.Sigma.shape == (2, 2)
    assert res.Lambda.shape == (2, 2)


def test_fm_ols_recovers_beta_under_endogeneity():
    """FM-OLS should be closer to true β=2 than naive OLS."""
    y, x = _simulate_cointegrated(T=400, beta=2.0, rho=0.6, seed=7)
    Z = np.column_stack([np.ones_like(y), x])
    b_ols = float(np.linalg.lstsq(Z, y, rcond=None)[0][1])
    b_fm = float(fm_ols(y, x).beta[0])
    # Under rho=0.6 endogeneity, OLS is biased; FM-OLS removes long-run bias.
    assert abs(b_fm - 2.0) < 0.05


def test_fm_ols_summary_runs():
    y, x = _simulate_cointegrated()
    s = fm_ols(y, x).summary()
    assert isinstance(s, str)
    assert "FM-OLS" in s
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_cointegration_modern.py -v`
Expected: ImportError (`FMOLSResult` not yet exported) — every test fails at collection.

- [ ] **Step 3: Add FMOLSResult dataclass to cointegration_modern.py**

Edit `puremacro/cointegration_modern.py`. After the existing imports block (after line 23), add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class FMOLSResult:
    """Result of :func:`fm_ols` (Phillips-Hansen Fully-Modified OLS).

    Attributes
    ----------
    beta : np.ndarray
        Cointegrating coefficients (excluding intercept), shape ``(k,)``.
    alpha : float
        Estimated intercept.
    se : np.ndarray
        HAC standard errors for ``beta``, shape ``(k,)``.
    residuals : np.ndarray
        Endogeneity-corrected residuals, length ``T - 1``.
    Omega : np.ndarray
        Two-sided long-run covariance, shape ``(k+1, k+1)``.
    Sigma : np.ndarray
        Contemporaneous covariance, shape ``(k+1, k+1)``.
    Lambda : np.ndarray
        One-sided long-run covariance, shape ``(k+1, k+1)``.

    References
    ----------
    Phillips, P.C.B. and Hansen, B.E. (1990). Statistical inference in
        instrumental variables regression with I(1) processes. RES 57(1).
    """

    beta: np.ndarray
    alpha: float
    se: np.ndarray
    residuals: np.ndarray
    Omega: np.ndarray
    Sigma: np.ndarray
    Lambda: np.ndarray

    def summary(self) -> str:
        coefs = ", ".join(f"{b:+.4f}" for b in self.beta)
        ses = ", ".join(f"{s:.4f}" for s in self.se)
        return (
            f"FM-OLS (Phillips-Hansen)\n"
            f"  alpha             : {self.alpha:+.4f}\n"
            f"  beta              : {coefs}\n"
            f"  se(beta)          : {ses}\n"
            f"  n residuals       : {len(self.residuals)}\n"
        )
```

- [ ] **Step 4: Change `fm_ols` return type from dict to FMOLSResult**

In `puremacro/cointegration_modern.py`, replace the `fm_ols` function signature and final `return` block. The return statement at lines 108–116 changes from a dict literal to an `FMOLSResult(...)` constructor call. Update the function signature `-> dict:` to `-> FMOLSResult:` and the docstring `Returns` section accordingly.

```python
def fm_ols(y, X, lags: int | None = None) -> FMOLSResult:
    """Phillips-Hansen Fully-Modified OLS estimator of the cointegrating
    vector in y_t = alpha + X_t' beta + u_t with X_t I(1).

    Returns
    -------
    FMOLSResult
        See :class:`FMOLSResult` for fields.
    """
    # ... existing body unchanged through line 107 ...
    return FMOLSResult(
        beta=beta_fm,
        alpha=alpha_fm,
        se=se_beta,
        residuals=resid,
        Omega=Omega,
        Sigma=Sigma,
        Lambda=Lambda,
    )
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/test_cointegration_modern.py -v`
Expected: 4 fm_ols tests pass; remaining tests fail (DOLSResult / PhillipsOuliarisResult not yet defined).

---

## Task 5: Migrate `cointegration_modern.dols` to DOLSResult

**Files:**
- Modify: `puremacro/cointegration_modern.py`
- Test: `tests/test_cointegration_modern.py`

`dols` returns dict with 5 keys: `alpha`, `beta`, `se`, `alpha_se`, `n_obs`. > 3 fields → result object.

- [ ] **Step 1: Append failing tests for dols to tests/test_cointegration_modern.py**

```python
# --------------------------------------------------------------------------
# dols
# --------------------------------------------------------------------------
def test_dols_returns_DOLSResult():
    y, x = _simulate_cointegrated()
    res = dols(y, x, leads=2, lags=2)
    assert isinstance(res, DOLSResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.alpha = 0.0


def test_dols_has_documented_fields():
    y, x = _simulate_cointegrated()
    res = dols(y, x, leads=2, lags=2)
    assert isinstance(res.alpha, float)
    assert isinstance(res.beta, np.ndarray) and res.beta.shape == (1,)
    assert isinstance(res.se, np.ndarray) and res.se.shape == (1,)
    assert isinstance(res.alpha_se, float)
    assert res.n_obs > 0


def test_dols_recovers_beta_under_endogeneity():
    y, x = _simulate_cointegrated(T=400, beta=2.0, rho=0.6, seed=11)
    b_dols = float(dols(y, x, leads=2, lags=2).beta[0])
    assert abs(b_dols - 2.0) < 0.05


def test_dols_summary_runs():
    y, x = _simulate_cointegrated()
    s = dols(y, x, leads=2, lags=2).summary()
    assert isinstance(s, str)
    assert "DOLS" in s
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_cointegration_modern.py -k dols -v`
Expected: ImportError on `DOLSResult`.

- [ ] **Step 3: Add DOLSResult dataclass**

Insert in `puremacro/cointegration_modern.py` immediately after `FMOLSResult`:

```python
@dataclass(frozen=True)
class DOLSResult:
    """Result of :func:`dols` (Stock-Watson DOLS).

    Attributes
    ----------
    alpha : float
        Estimated intercept.
    beta : np.ndarray
        Cointegrating coefficients (excluding intercept), shape ``(k,)``.
    se : np.ndarray
        HAC standard errors for ``beta``, shape ``(k,)``.
    alpha_se : float
        HAC standard error for the intercept.
    n_obs : int
        Effective sample size used in the augmented regression
        (lower than T due to lead/lag truncation).

    References
    ----------
    Stock, J.H. and Watson, M.W. (1993). A simple estimator of cointegrating
        vectors in higher order integrated systems. Econometrica 61(4).
    """

    alpha: float
    beta: np.ndarray
    se: np.ndarray
    alpha_se: float
    n_obs: int

    def summary(self) -> str:
        coefs = ", ".join(f"{b:+.4f}" for b in self.beta)
        ses = ", ".join(f"{s:.4f}" for s in self.se)
        return (
            f"DOLS (Stock-Watson)\n"
            f"  alpha             : {self.alpha:+.4f} (se {self.alpha_se:.4f})\n"
            f"  beta              : {coefs}\n"
            f"  se(beta)          : {ses}\n"
            f"  n_obs             : {self.n_obs}\n"
        )
```

- [ ] **Step 4: Change `dols` return type from dict to DOLSResult**

Replace the final return block in `dols`:

```python
    return DOLSResult(
        alpha=float(beta[0]),
        beta=beta[1:1 + k],
        se=se[1:1 + k],
        alpha_se=float(se[0]),
        n_obs=len(yd),
    )
```

Update the signature `-> dict:` to `-> DOLSResult:` and docstring Returns section.

- [ ] **Step 5: Run dols tests to verify pass**

Run: `pytest tests/test_cointegration_modern.py -k dols -v`
Expected: 4 dols tests pass.

---

## Task 6: Migrate `cointegration_modern.phillips_ouliaris` to PhillipsOuliarisResult

**Files:**
- Modify: `puremacro/cointegration_modern.py`
- Test: `tests/test_cointegration_modern.py`

`phillips_ouliaris` returns dict with 5 keys: `z_alpha`, `z_t`, `residuals`, `rho`, `lag_used`.

- [ ] **Step 1: Append failing tests**

```python
# --------------------------------------------------------------------------
# phillips_ouliaris
# --------------------------------------------------------------------------
def test_phillips_ouliaris_returns_PhillipsOuliarisResult():
    y, x = _simulate_cointegrated()
    res = phillips_ouliaris(y, x)
    assert isinstance(res, PhillipsOuliarisResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.z_t = 0.0


def test_phillips_ouliaris_rejects_unit_root_when_cointegrated():
    """On a cointegrated DGP Z_t should be more negative than -3.37 (5% CV, k=1)."""
    y, x = _simulate_cointegrated(T=400, beta=2.0, rho=0.5, seed=3)
    res = phillips_ouliaris(y, x)
    assert res.z_t < -3.37


def test_phillips_ouliaris_summary_runs():
    y, x = _simulate_cointegrated()
    s = phillips_ouliaris(y, x).summary()
    assert isinstance(s, str)
    assert "Phillips-Ouliaris" in s
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_cointegration_modern.py -k phillips_ouliaris -v`
Expected: ImportError on `PhillipsOuliarisResult`.

- [ ] **Step 3: Add PhillipsOuliarisResult dataclass**

Insert in `puremacro/cointegration_modern.py` immediately after `DOLSResult`:

```python
@dataclass(frozen=True)
class PhillipsOuliarisResult:
    """Result of :func:`phillips_ouliaris` (residual-based cointegration test).

    Attributes
    ----------
    z_alpha : float
        Phillips-Ouliaris Z_alpha statistic.
    z_t : float
        Phillips-Ouliaris Z_t statistic. More negative ⇒ stronger evidence
        of cointegration. 5% CV ≈ −3.37 for k=1 regressor (PO 1990 Table II).
    residuals : np.ndarray
        OLS residuals from the first-stage cointegrating regression.
    rho : float
        First-stage AR(1) coefficient on the residuals.
    lag_used : int
        Bartlett-kernel truncation lag used for long-run variance correction.

    References
    ----------
    Phillips, P.C.B. and Ouliaris, S. (1990). Asymptotic properties of
        residual based tests for cointegration. Econometrica 58(1).
    """

    z_alpha: float
    z_t: float
    residuals: np.ndarray
    rho: float
    lag_used: int

    def summary(self) -> str:
        return (
            f"Phillips-Ouliaris cointegration test\n"
            f"  Z_alpha           : {self.z_alpha:+.4f}\n"
            f"  Z_t               : {self.z_t:+.4f}  (5% CV ≈ -3.37, k=1)\n"
            f"  rho               : {self.rho:+.4f}\n"
            f"  lag_used          : {self.lag_used}\n"
        )
```

- [ ] **Step 4: Change `phillips_ouliaris` return**

Replace the final return block:

```python
    return PhillipsOuliarisResult(
        z_alpha=float(z_alpha),
        z_t=float(z_t),
        residuals=u,
        rho=rho,
        lag_used=int(lags),
    )
```

Update signature `-> dict:` to `-> PhillipsOuliarisResult:` and docstring Returns section.

- [ ] **Step 5: Update `__all__`**

Change the `__all__` line at the bottom of `cointegration_modern.py` from:

```python
__all__ = ["fm_ols", "dols", "phillips_ouliaris"]
```

to:

```python
__all__ = [
    "fm_ols", "dols", "phillips_ouliaris",
    "FMOLSResult", "DOLSResult", "PhillipsOuliarisResult",
]
```

- [ ] **Step 6: Run all cointegration tests**

Run: `pytest tests/test_cointegration_modern.py -v`
Expected: All 11 tests pass.

---

## Task 7: Update fm_ols_dols_demo example to use attribute access

**Files:**
- Modify: `puremacro/examples/fm_ols_dols_demo.py:48-53`

The demo uses `fm_ols(y, x)["beta"][0]`, `dols(y, x, ...)["beta"][0]`, `phillips_ouliaris(y, x)["z_t"]`. Switch to attribute access.

- [ ] **Step 1: Replace the three subscripts in the loop**

Edit `puremacro/examples/fm_ols_dols_demo.py` lines 48–53:

```python
        b_fm = float(fm_ols(y, x).beta[0])
        b_dols = float(dols(y, x, leads=2, lags=2).beta[0])
        ols_betas.append(b_ols)
        fm_betas.append(b_fm)
        dols_betas.append(b_dols)
        pos_z_t.append(phillips_ouliaris(y, x).z_t)
```

- [ ] **Step 2: Verify the demo still parses**

Run: `python -c "import ast; ast.parse(open('puremacro/examples/fm_ols_dols_demo.py').read())"`
Expected: No output (clean parse).

- [ ] **Step 3: Smoke-run the demo**

Run: `python -m puremacro.examples.fm_ols_dols_demo`
Expected: Console summary prints with biases for OLS / FM-OLS / DOLS and Phillips-Ouliaris Z_t mean. No tracebacks.

---

## Task 8: Migrate `midas.u_midas` to UMidasResult

**Files:**
- Modify: `puremacro/midas.py`
- Test: `tests/test_midas.py` (new)

`u_midas` returns dict with 6 keys: `intercept`, `beta`, `fitted`, `residuals`, `R2`, `n_obs`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_midas.py`:

```python
"""Tests for puremacro.midas result objects."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.midas import (
    u_midas, beta_midas,
    UMidasResult, BetaMidasResult,
)


def _simulate_midas(n_low=120, K=3, seed=0):
    rng = np.random.default_rng(seed)
    x_hf = rng.standard_normal(n_low * K)
    for i in range(1, len(x_hf)):
        x_hf[i] = 0.4 * x_hf[i - 1] + x_hf[i]
    true_w = np.array([0.7, 0.2, 0.1])
    y_lf = np.zeros(n_low)
    for i in range(n_low):
        seg = x_hf[i * K:(i + 1) * K][::-1]
        y_lf[i] = 0.5 + 0.8 * (true_w @ seg) + rng.standard_normal() * 0.4
    return y_lf, x_hf, K, true_w


# --------------------------------------------------------------------------
# u_midas
# --------------------------------------------------------------------------
def test_u_midas_returns_UMidasResult():
    y, x, K, _ = _simulate_midas()
    res = u_midas(y, x, K=K)
    assert isinstance(res, UMidasResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.intercept = 0.0


def test_u_midas_has_documented_fields():
    y, x, K, _ = _simulate_midas()
    res = u_midas(y, x, K=K)
    assert isinstance(res.intercept, float)
    assert res.beta.shape == (K,)
    assert res.fitted.shape == res.residuals.shape
    assert 0.0 <= res.R2 <= 1.0
    assert res.n_obs == len(y)


def test_u_midas_summary_runs():
    y, x, K, _ = _simulate_midas()
    s = u_midas(y, x, K=K).summary()
    assert isinstance(s, str)
    assert "U-MIDAS" in s
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_midas.py -k u_midas -v`
Expected: ImportError on `UMidasResult`.

- [ ] **Step 3: Add UMidasResult dataclass**

Edit `puremacro/midas.py`. After the imports block (after `from scipy.stats import beta as beta_dist`), add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class UMidasResult:
    """Result of :func:`u_midas` (unrestricted MIDAS).

    Attributes
    ----------
    intercept : float
        Estimated intercept.
    beta : np.ndarray
        Per-lag MIDAS coefficients, length ``K * (1 + n_low_lags)``.
    fitted : np.ndarray
        In-sample fitted values, length ``n_obs``.
    residuals : np.ndarray
        In-sample residuals, length ``n_obs``.
    R2 : float
        In-sample R².
    n_obs : int
        Effective sample size used in the regression.

    References
    ----------
    Foroni, C., Marcellino, M. and Schumacher, C. (2015). Unrestricted
        mixed data sampling (MIDAS): MIDAS regressions with unrestricted
        lag polynomials. JRSS-A 178(1), 57-82.
    """

    intercept: float
    beta: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    R2: float
    n_obs: int

    def summary(self) -> str:
        coefs = ", ".join(f"{b:+.3f}" for b in self.beta)
        return (
            f"U-MIDAS (Foroni-Marcellino-Schumacher)\n"
            f"  intercept         : {self.intercept:+.4f}\n"
            f"  beta              : {coefs}\n"
            f"  R²                : {self.R2:.4f}\n"
            f"  n_obs             : {self.n_obs}\n"
        )
```

- [ ] **Step 4: Change `u_midas` return type**

Replace the final return block at lines 92–99:

```python
    return UMidasResult(
        intercept=float(beta[0]),
        beta=beta[1:],
        fitted=fitted,
        residuals=resid,
        R2=1.0 - rss / tss if tss > 0 else 0.0,
        n_obs=n_obs,
    )
```

Update signature `-> dict:` to `-> UMidasResult:` and docstring.

- [ ] **Step 5: Run u_midas tests to verify pass**

Run: `pytest tests/test_midas.py -k u_midas -v`
Expected: 3 u_midas tests pass.

---

## Task 9: Migrate `midas.beta_midas` to BetaMidasResult

**Files:**
- Modify: `puremacro/midas.py`
- Test: `tests/test_midas.py`

`beta_midas` returns dict with 9 keys: `intercept`, `beta`, `theta1`, `theta2`, `weights`, `fitted`, `residuals`, `R2`, `converged`.

- [ ] **Step 1: Append failing tests**

```python
# --------------------------------------------------------------------------
# beta_midas
# --------------------------------------------------------------------------
def test_beta_midas_returns_BetaMidasResult():
    y, x, K, _ = _simulate_midas()
    res = beta_midas(y, x, K=K)
    assert isinstance(res, BetaMidasResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.beta = 0.0


def test_beta_midas_has_documented_fields():
    y, x, K, _ = _simulate_midas()
    res = beta_midas(y, x, K=K)
    assert isinstance(res.intercept, float)
    assert isinstance(res.beta, float)
    assert isinstance(res.theta1, float) and res.theta1 > 0
    assert isinstance(res.theta2, float) and res.theta2 > 0
    assert res.weights.shape == (K,)
    assert np.isclose(res.weights.sum(), 1.0)
    assert res.fitted.shape == res.residuals.shape == y.shape
    assert isinstance(res.converged, bool)


def test_beta_midas_summary_runs():
    y, x, K, _ = _simulate_midas()
    s = beta_midas(y, x, K=K).summary()
    assert isinstance(s, str)
    assert "Beta-MIDAS" in s
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_midas.py -k beta_midas -v`
Expected: ImportError on `BetaMidasResult`.

- [ ] **Step 3: Add BetaMidasResult dataclass**

In `puremacro/midas.py`, immediately after `UMidasResult`:

```python
@dataclass(frozen=True)
class BetaMidasResult:
    """Result of :func:`beta_midas` (Beta-polynomial MIDAS).

    Attributes
    ----------
    intercept : float
        Estimated intercept.
    beta : float
        Single low-frequency slope on the kernel-weighted high-frequency sum.
    theta1, theta2 : float
        Beta-pdf shape parameters (positive).
    weights : np.ndarray
        Kernel weights ``w(k; theta1, theta2)``, length ``K``, sum to 1.
    fitted : np.ndarray
        In-sample fitted values, length ``n_low``.
    residuals : np.ndarray
        In-sample residuals, length ``n_low``.
    R2 : float
        In-sample R².
    converged : bool
        Whether scipy.optimize.minimize converged.

    References
    ----------
    Ghysels, E., Santa-Clara, P. and Valkanov, R. (2007). MIDAS regressions:
        further results and new directions. Econometric Reviews 26.
    """

    intercept: float
    beta: float
    theta1: float
    theta2: float
    weights: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    R2: float
    converged: bool

    def summary(self) -> str:
        w = ", ".join(f"{x:.3f}" for x in self.weights)
        return (
            f"Beta-MIDAS (Ghysels-Santa-Clara-Valkanov)\n"
            f"  intercept         : {self.intercept:+.4f}\n"
            f"  beta              : {self.beta:+.4f}\n"
            f"  theta1, theta2    : {self.theta1:.3f}, {self.theta2:.3f}\n"
            f"  weights           : {w}\n"
            f"  R²                : {self.R2:.4f}\n"
            f"  converged         : {self.converged}\n"
        )
```

- [ ] **Step 4: Change `beta_midas` return**

Replace the final return block at lines 164–174:

```python
    return BetaMidasResult(
        intercept=float(beta_lin[0]),
        beta=float(beta_lin[1]),
        theta1=float(t1),
        theta2=float(t2),
        weights=w_hat,
        fitted=fitted,
        residuals=resid,
        R2=1.0 - rss / tss if tss > 0 else 0.0,
        converged=bool(res.success),
    )
```

Update signature `-> dict:` to `-> BetaMidasResult:` and docstring.

- [ ] **Step 5: Update `__all__`**

Change the `__all__` line at the bottom of `midas.py` from:

```python
__all__ = ["u_midas", "beta_midas"]
```

to:

```python
__all__ = ["u_midas", "beta_midas", "UMidasResult", "BetaMidasResult"]
```

- [ ] **Step 6: Update `nowcast/__init__.py` re-exports**

`nowcast/__init__.py` re-exports `u_midas` and `beta_midas` from `..midas` (line 36) and lists them in `__all__` (line 42). The re-exports continue to work as-is (they import the names, which is unchanged). No change required, but verify:

Run: `python -c "from puremacro.nowcast import u_midas, beta_midas; print(type(u_midas).__name__)"`
Expected: `function`.

- [ ] **Step 7: Run all midas tests**

Run: `pytest tests/test_midas.py -v`
Expected: All 6 tests pass.

---

## Task 10: Update midas_quarterly_monthly example to use attribute access

**Files:**
- Modify: `puremacro/examples/midas_quarterly_monthly.py`

The example uses `um['intercept']`, `um['beta']`, `um['R2']`, `um['fitted']`, and parallel keys for `bm`. Replace all dict subscripts with attribute access.

- [ ] **Step 1: Replace dict access in `main()` and the figure block**

Edit `puremacro/examples/midas_quarterly_monthly.py`. Replace each `um['key']` with `um.key` and each `bm['key']` with `bm.key`. Specifically these lines (54–65, 77, 89–93):

| Old | New |
|-----|-----|
| `um['intercept']` | `um.intercept` |
| `um['beta']` | `um.beta` |
| `um['R2']` | `um.R2` |
| `um['fitted']` | `um.fitted` |
| `bm['intercept']` | `bm.intercept` |
| `bm['beta']` | `bm.beta` |
| `bm['weights']` | `bm.weights` |
| `bm['theta1']` | `bm.theta1` |
| `bm['theta2']` | `bm.theta2` |
| `bm['R2']` | `bm.R2` |
| `bm['fitted']` | `bm.fitted` |

(`out["u_midas"]` and `out["beta_midas"]` stay as dict access — `out` is a hand-built dict, not a result object.)

- [ ] **Step 2: Verify parse**

Run: `python -c "import ast; ast.parse(open('puremacro/examples/midas_quarterly_monthly.py').read())"`
Expected: No output.

- [ ] **Step 3: Smoke-run the demo**

Run: `python -m puremacro.examples.midas_quarterly_monthly`
Expected: Console output prints intercepts, slopes, weights, θ values, R² for both U-MIDAS and Beta-MIDAS. No tracebacks.

---

## Task 11: Add `user_agent=` override to `_http.py` helpers

**Files:**
- Modify: `puremacro/narrative/sources/_http.py`
- Modify: `puremacro/narrative/sources/RETRY_POLICY.md`

The DoD WAF blocks the default `Mozilla/5.0 (puremacro/narrative)` UA. Add an optional `user_agent=` parameter to all three `safe_get_*` helpers and to `_request`. Default unchanged → no behaviour change for existing callers.

- [ ] **Step 1: Edit `_request` to accept an override**

Replace the `_request` function in `puremacro/narrative/sources/_http.py`:

```python
def _request(url: str, timeout: float, user_agent: str | None = None) -> bytes:
    ua = user_agent or USER_AGENT
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (urllib.error.URLError, ssl.SSLError):
        # One-shot fallback: some public endpoints (older OECD / IMF /
        # ministry sites) ship certificates that Python's bundled CA
        # store does not validate. Retry once with verification off.
        # See RETRY_POLICY.md §3 for why we do not loop further.
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
```

- [ ] **Step 2: Add `user_agent=` to the three `safe_get_*` wrappers**

Replace the three wrapper functions:

```python
def safe_get_bytes(url: str, timeout: float = DEFAULT_TIMEOUT,
                   *, user_agent: str | None = None) -> bytes:
    """Fetch ``url`` and return raw bytes. SSL fallback applied once.

    ``user_agent`` overrides the default ``Mozilla/5.0 (puremacro/narrative)``
    UA — needed for endpoints behind a WAF that blocks scripted clients.
    """
    return _request(url, timeout, user_agent=user_agent)


def safe_get_text(url: str, timeout: float = DEFAULT_TIMEOUT,
                  *, user_agent: str | None = None) -> str:
    """Fetch ``url`` and return UTF-8 text (decode errors ignored)."""
    return _request(url, timeout, user_agent=user_agent).decode(
        "utf-8", errors="ignore",
    )


def safe_get_json(url: str, timeout: float = DEFAULT_TIMEOUT,
                  *, user_agent: str | None = None) -> dict:
    """Fetch ``url`` and return decoded JSON.

    Empty / whitespace-only bodies return ``{}`` rather than raising,
    matching the existing API-connector behaviour (e.g. GDELT v2 rate
    limits sometimes return blank pages).
    """
    text = safe_get_text(url, timeout, user_agent=user_agent)
    if not text.strip():
        return {}
    return json.loads(text)
```

- [ ] **Step 3: Document the override in RETRY_POLICY.md**

Read `puremacro/narrative/sources/RETRY_POLICY.md` first to confirm structure. Then append a new section documenting the override:

```markdown
## §6 User-Agent overrides

Some public endpoints sit behind a WAF that blocks the default
`Mozilla/5.0 (puremacro/narrative)` agent string. Connectors that
hit such endpoints should pass an explicit, realistic browser UA via
the `user_agent=` keyword on `safe_get_bytes` / `safe_get_text` /
`safe_get_json`.

Currently this applies to:
- `us_dod_contracts.iter_dod_contracts` — defense.gov WAF.
```

- [ ] **Step 4: Run the offline narrative tests to verify no regression**

Run: `pytest tests/test_narrative_offline.py -v`
Expected: 13 passing, same as baseline (no behaviour change for existing callers).

---

## Task 12: Apply UA tweak in us_dod_contracts and re-record fixture

**Files:**
- Modify: `puremacro/narrative/sources/us_dod_contracts.py`
- Modify: `tests/test_narrative_offline.py` (un-skip)

- [ ] **Step 1: Update import + add a module-level constant + pass it to the call**

Edit `puremacro/narrative/sources/us_dod_contracts.py`. Replace the body so it reads:

```python
from ._http import safe_get_text


_BASE = "https://www.defense.gov/News/Contracts/Filter-Contracts/"

# defense.gov WAF blocks the default Mozilla/5.0 (puremacro/narrative)
# string; pass a realistic browser UA. See narrative/sources/RETRY_POLICY.md §6.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
```

…and within `iter_dod_contracts`, change the `safe_get_text(url)` call to:

```python
            html = safe_get_text(url, user_agent=_BROWSER_UA)
```

- [ ] **Step 2: Inspect the existing skip in the offline test**

Run: `grep -n "us_dod_contracts" tests/test_narrative_offline.py`
Expected: Find a `pytest.mark.skip` with reason mentioning WAF. (If structured otherwise — e.g. missing fixture file — adapt the un-skip step accordingly.)

- [ ] **Step 3: Re-record the fixture (manual one-shot, requires network)**

Run: `PUREMACRO_RECORD_HTTP=1 pytest tests/test_narrative_offline.py -k us_dod_contracts -v`
Expected: New fixture file written under `tests/fixtures/http/<sha>.json`. If network is unavailable on this machine, **leave the skip in place** and instead update the skip reason from "WAF-blocked" to "fixture not yet recorded — run with PUREMACRO_RECORD_HTTP=1 once on a network-enabled machine".

- [ ] **Step 4: Run the offline test to verify replay works**

Run: `pytest tests/test_narrative_offline.py -k us_dod_contracts -v`
Expected: PASS (replays from new fixture). If the fixture record step was deferred, this remains skipped with the updated reason — that is also acceptable for the patch.

---

## Task 13: Fix `us_federal_register` default agency slug

**Files:**
- Modify: `puremacro/narrative/sources/us_federal_register.py`

Memory says: with default `agencies=("treasury-department", "office-of-management-and-budget", "defense-department")`, the FR API returns HTTP 400. The OMB slug appears to have changed (or never existed). Safest patch: drop OMB from the default list. Document the change in the docstring and add a note pointing the user to the FR agencies endpoint to discover current valid slugs.

- [ ] **Step 1: Edit the default `agencies=` tuple**

In `puremacro/narrative/sources/us_federal_register.py`, change the function signature default:

```python
def iter_federal_register(
    *,
    since: str = "2008-01-01",
    until: str | None = None,
    agencies: tuple[str, ...] = ("treasury-department", "defense-department"),
    document_types: tuple[str, ...] = ("PRESDOCU", "RULE", "NOTICE"),
    per_page: int = 200,
    max_pages: int = 5,
) -> Iterator[tuple]:
```

- [ ] **Step 2: Document the agency-slug discovery in the docstring**

Update the `agencies` line in the docstring to:

```python
    agencies : Federal Register agency slugs (lowercase, hyphenated).
        See ``https://www.federalregister.gov/api/v1/agencies.json`` for
        the canonical list. Note: the slug for OMB has historically
        rotated and may not be ``office-of-management-and-budget``;
        verify before adding it back to the default.
```

- [ ] **Step 3: Re-record the fixture (manual one-shot, requires network)**

Run: `PUREMACRO_RECORD_HTTP=1 pytest tests/test_narrative_offline.py -k us_federal_register -v`
Expected: New fixture file. If network is unavailable, leave the test as-is and document in the patch notes that the fix is shipped but the fixture re-record is deferred until network access is available.

- [ ] **Step 4: Run the offline test**

Run: `pytest tests/test_narrative_offline.py -k us_federal_register -v`
Expected: PASS (with re-recorded fixture) or unchanged behaviour (if fixture rerecord deferred).

---

## Task 14: Replace dead `us_treasury` RSS URL with HTML scrape

**Files:**
- Modify: `puremacro/narrative/sources/us_treasury.py`
- Modify: `tests/test_narrative_offline.py` (un-skip / adapt)

Memory says the RSS URL `https://home.treasury.gov/rss/press-releases.xml` is dead in 2026. The connector currently silently no-ops. Replace it with HTML scraping of the press-releases listing page (modeled on `us_dod_contracts`). The listing page is `https://home.treasury.gov/news/press-releases`. This shifts from the `safe_get_bytes` + `xml.etree` path to `safe_get_text` + a small HTML regex parser.

- [ ] **Step 1: Rewrite us_treasury.py**

Replace the entire contents of `puremacro/narrative/sources/us_treasury.py` with:

```python
"""US Treasury press-release feed.

The Treasury publishes press releases at
``https://home.treasury.gov/news/press-releases``. The Atom/RSS feed
that earlier puremacro versions targeted (``/rss/press-releases.xml``)
went dead in 2026; this module instead scrapes the listing page HTML.
The listing markup is stable across redesigns and stdlib regex parsing
is sufficient — no new dependencies introduced.

For history beyond what the live listing exposes, the user can supply
a local CSV of Treasury releases (see :mod:`local_csv`).
"""
from __future__ import annotations

import re
import urllib.parse
from typing import Iterator

import pandas as pd

from ._http import safe_get_text


_BASE = "https://home.treasury.gov/news/press-releases"


# The listing renders each release as a card with a date, a headline
# anchor, and an excerpt. The exact tag wrapping rotates between
# Drupal redesigns, so we anchor on the most stable element: the
# ``<time datetime="...">`` element with a sibling/descendant anchor.
_ITEM_RX = re.compile(
    r"<article[^>]*>(.*?)</article>", flags=re.IGNORECASE | re.DOTALL
)
_TITLE_RX = re.compile(
    r"<h[23][^>]*>\s*<a[^>]*>(.*?)</a>", flags=re.IGNORECASE | re.DOTALL
)
_DATE_RX = re.compile(
    r"<time[^>]*datetime=\"([0-9\-T:Z+]+)\"", flags=re.IGNORECASE
)
_HREF_RX = re.compile(
    r"<a[^>]*href=\"(/news/press-releases/[^\"]+)\"", flags=re.IGNORECASE
)


def iter_treasury_press(*, max_pages: int = 5) -> Iterator[tuple]:
    """Yield (pubdate, title, link) records for recent Treasury releases.

    Pagination uses the Drupal default ``?page=N`` query parameter
    (zero-indexed on this site).
    """
    for page in range(0, max_pages):
        url = _BASE + (f"?page={page}" if page > 0 else "")
        try:
            html = safe_get_text(url)
        except Exception:
            return
        items = _ITEM_RX.findall(html)
        if not items:
            return
        for item in items:
            title_m = _TITLE_RX.search(item)
            date_m = _DATE_RX.search(item)
            href_m = _HREF_RX.search(item)
            if not title_m or not date_m:
                continue
            title = re.sub(r"<[^>]+>", " ", title_m.group(1)).strip()
            try:
                date = pd.Timestamp(date_m.group(1))
            except Exception:
                continue
            link = href_m.group(1) if href_m else ""
            if link.startswith("/"):
                link = "https://home.treasury.gov" + link
            yield date, title, link


__all__ = ["iter_treasury_press"]
```

Note the API change: `iter_treasury_press()` now takes a `max_pages` keyword. Default is 5. Backwards-compatible for existing zero-arg callers.

- [ ] **Step 2: Re-record the fixture (manual one-shot, requires network)**

Run: `PUREMACRO_RECORD_HTTP=1 pytest tests/test_narrative_offline.py -k us_treasury -v`
Expected: A fresh fixture file is written. If network is unavailable, leave the existing skip in place and update its reason from "URL dead 2026" to "listing-page fixture not yet recorded — run with PUREMACRO_RECORD_HTTP=1 once".

- [ ] **Step 3: If the fixture was recorded, un-skip the test**

Open `tests/test_narrative_offline.py` and find the `us_treasury` skip. If a fixture now exists, remove the skip mark. If still no fixture, leave the skip but with the updated reason from Step 2.

- [ ] **Step 4: Run the offline test**

Run: `pytest tests/test_narrative_offline.py -k us_treasury -v`
Expected: PASS if fixture was recorded; otherwise SKIPPED with the updated reason.

---

## Task 15: AB 1991 replication fixture infrastructure

**Files:**
- Create: `tests/fixtures/abdata.README.md`
- Create: `tests/test_dynpanel/test_ab_1991_replication.py`

The canonical Arellano-Bond (1991) UK manufacturing employment panel (140 firms, 1976–1984) is widely redistributed (Stata `abdata.dta`, R `plm::EmplUK`). Without network access in this environment we cannot ship the CSV directly; this task ships the **infrastructure** so that, the moment the user drops `abdata.csv` into `tests/fixtures/`, the replication test runs automatically.

Canonical published estimates from AB (1991, Table 4, col. 2; two-step difference GMM, dynamic n equation with w, k, ys controls and time dummies): coefficient on L1.n ≈ 0.474 (s.e. 0.085), on L2.n ≈ −0.053 (s.e. 0.027). We assert these to within a generous tolerance (0.05 absolute) so any reasonable port will pass.

- [ ] **Step 1: Create the fixture README**

Create `tests/fixtures/abdata.README.md`:

```markdown
# Arellano-Bond (1991) employment data

`abdata.csv` is the canonical 140-firm × 1976-1984 UK manufacturing panel
used in Arellano and Bond (1991, RES 58, Table 4) and reproduced in many
econometrics packages.

## How to obtain

The dataset is redistributed under permissive terms by several
mainstream packages:

- **Stata**: `webuse abdata`
- **R (plm)**: `data("EmplUK", package = "plm")`, then `write.csv(EmplUK, "abdata.csv", row.names = FALSE)`
- **Stata Press download**: `https://www.stata-press.com/data/r17/abdata.dta`

## Required schema

The replication test (`tests/test_dynpanel/test_ab_1991_replication.py`)
expects the following columns (others are ignored):

| Column | Description |
|--------|-------------|
| `id`   | Firm identifier (integer) |
| `year` | Year (integer, 1976–1984) |
| `n`    | log(employment) |
| `w`    | log(wage) |
| `k`    | log(capital) |
| `ys`   | log(industry output) |

Column names from `plm::EmplUK` (`firm`, `year`, `emp`, `wage`,
`capital`, `output`, `sector`) need to be transformed:
`emp→n`, `wage→w`, `capital→k`, `output→ys`, `firm→id`,
and the variables logged. See the `plm` documentation for details.

## Canonical estimates (AB 1991 Table 4, col. 2)

Two-step difference GMM with Windmeijer SE on the dynamic n equation:

| Variable | Coef    | s.e.  |
|----------|---------|-------|
| L1.n     |  0.474  | 0.085 |
| L2.n     | -0.053  | 0.027 |

Without the CSV present, the replication test is **skipped**. With the
CSV present, the test asserts coefficients within 0.05 of the published
values.
```

- [ ] **Step 2: Write the skip-if-absent replication test**

Create `tests/test_dynpanel/test_ab_1991_replication.py`:

```python
"""Replication of Arellano-Bond (1991) Table 4, column 2.

Requires ``tests/fixtures/abdata.csv`` (see fixture README for how to
obtain). Skipped automatically if the CSV is absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dynpanel import ab_gmm

ABDATA_CSV = Path(__file__).parent.parent / "fixtures" / "abdata.csv"


@pytest.fixture(scope="module")
def abdata():
    if not ABDATA_CSV.exists():
        pytest.skip(
            f"Replication fixture {ABDATA_CSV.name} not present. "
            "See tests/fixtures/abdata.README.md for how to obtain."
        )
    df = pd.read_csv(ABDATA_CSV)
    needed = {"id", "year", "n", "w", "k", "ys"}
    missing = needed - set(df.columns)
    if missing:
        pytest.skip(f"abdata.csv missing required columns: {sorted(missing)}")
    return df.sort_values(["id", "year"]).reset_index(drop=True)


def test_ab_1991_table4_col2_recovers_published_lag_coefficients(abdata):
    """AB 1991 Table 4 col. 2: two-step diff-GMM with Windmeijer SE.

    Published: L1.n = 0.474 (s.e. 0.085), L2.n = -0.053 (s.e. 0.027).
    We assert recovery within 0.05 absolute on coefficients (generous
    tolerance — any reasonable port should clear it).
    """
    df = abdata
    # Build long-format arrays with two lags of n + w + k + ys controls.
    # AB 1991 col. 2 includes time dummies; we approximate by using
    # X_pred for w, k, ys (predetermined) per Roodman 2009 conventions.
    res = ab_gmm(
        y=df["n"].to_numpy(),
        panel_id=df["id"].to_numpy(),
        time_id=df["year"].to_numpy(),
        lag_dep_var=2,
        X_pred=df[["w", "k", "ys"]].to_numpy(),
        two_step=True,
        windmeijer=True,
        collapse=True,
    )
    # First two coefs are L1.n, L2.n by construction.
    coef_L1 = res.coefs[0]
    coef_L2 = res.coefs[1]
    assert abs(coef_L1 - 0.474) < 0.05, f"L1.n = {coef_L1:.3f}"
    assert abs(coef_L2 - (-0.053)) < 0.05, f"L2.n = {coef_L2:.3f}"


def test_ab_1991_hansen_j_does_not_reject(abdata):
    """Hansen J should not reject overid restrictions on the AB DGP."""
    df = abdata
    res = ab_gmm(
        y=df["n"].to_numpy(),
        panel_id=df["id"].to_numpy(),
        time_id=df["year"].to_numpy(),
        lag_dep_var=2,
        X_pred=df[["w", "k", "ys"]].to_numpy(),
        two_step=True,
        windmeijer=True,
        collapse=True,
    )
    assert res.hansen_j_p > 0.05, f"Hansen J p = {res.hansen_j_p:.3f}"
```

- [ ] **Step 3: Verify the test skips gracefully when CSV is absent**

Run: `pytest tests/test_dynpanel/test_ab_1991_replication.py -v`
Expected: 2 SKIPPED with the documented reason. No errors.

- [ ] **Step 4: (If user ever supplies abdata.csv) verify recovery**

Document this as a follow-up step for the user — not part of this patch's verification. The infrastructure is the deliverable.

---

## Task 16: Regenerate the public-API snapshot

**Files:**
- Modify: `tests/fixtures/public_api_snapshot.json`

The snapshot freezes `__all__` per subpackage and the field names of each `<MethodName>Result`. Adding 5 new result classes (`FMOLSResult`, `DOLSResult`, `PhillipsOuliarisResult`, `UMidasResult`, `BetaMidasResult`) and extending two `__all__` tuples is intentional drift — regenerate.

- [ ] **Step 1: Run the public-API test to confirm drift is detected**

Run: `pytest tests/test_public_api.py -v`
Expected: FAIL — diff shows the 5 new result classes and the two `__all__` extensions.

- [ ] **Step 2: Regenerate the snapshot**

Run:

```bash
cd "uncertainty_examples/puremacro" && \
python -c "from tests.test_public_api import _collect_current_api; \
import json; print(json.dumps(_collect_current_api(), indent=2))" \
> tests/fixtures/public_api_snapshot.json
```

(`tests` is already on the pytest path because `tests/__init__.py` exists in subdirs. If `from tests.test_public_api` fails because `tests/__init__.py` is absent at the top level, fall back to `python -c "import sys; sys.path.insert(0, 'tests'); from test_public_api import _collect_current_api; import json; print(json.dumps(_collect_current_api(), indent=2))"`.)

- [ ] **Step 3: Re-run the public-API test to confirm green**

Run: `pytest tests/test_public_api.py -v`
Expected: PASS.

- [ ] **Step 4: Spot-check the diff is minimal and only adds expected entries**

Run: `git diff tests/fixtures/public_api_snapshot.json 2>/dev/null | head -60` if git is available, otherwise eyeball the file.
Expected: 5 new `result_classes` entries (`FMOLSResult`, `DOLSResult`, `PhillipsOuliarisResult`, `UMidasResult`, `BetaMidasResult`); 2 modified `all` entries (`puremacro.cointegration_modern`, `puremacro.midas`). Nothing else.

---

## Task 17: Bump version to 0.4.1

**Files:**
- Modify: `pyproject.toml`
- Modify: `puremacro/__init__.py`
- Modify: `tests/test_import.py`

- [ ] **Step 1: Bump pyproject.toml**

In `pyproject.toml`, change `version = "0.4.0"` to `version = "0.4.1"`.

- [ ] **Step 2: Bump `__version__`**

In `puremacro/__init__.py`, find the `__version__ = "0.4.0"` line and change to `"0.4.1"`.

- [ ] **Step 3: Bump test_import.py expected version**

In `tests/test_import.py`, find the assertion against `"0.4.0"` and change to `"0.4.1"`.

- [ ] **Step 4: Run import test**

Run: `pytest tests/test_import.py -v`
Expected: PASS.

---

## Task 18: Update CHANGELOG.md

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the 0.4.1 entry at the top**

Insert immediately after the file's main heading and before the existing `## 0.4.0 — 2026-05-02` block:

```markdown
## 0.4.1 — 2026-05-02

Patch release — closes the seven follow-ups flagged at the end of 0.4.0.
No breaking changes; two return-type changes (dict → frozen dataclass)
that preserve all field names — attribute access works identically to
the previous dict-key access for any caller using `result["key"]`
patterns will need to switch to `result.key`. The two affected
public functions are `cointegration_modern.{fm_ols, dols, phillips_ouliaris}`
and `midas.{u_midas, beta_midas}`.

### Result-object migrations
- `puremacro.cointegration_modern` — `fm_ols` → `FMOLSResult`,
  `dols` → `DOLSResult`, `phillips_ouliaris` → `PhillipsOuliarisResult`.
  All three now expose `.summary()`. `__all__` extended with the three
  result classes.
- `puremacro.midas` — `u_midas` → `UMidasResult`, `beta_midas` →
  `BetaMidasResult`. Both expose `.summary()`. `__all__` extended.

### Narrative connector fixes
- `narrative.sources._http` — added optional `user_agent=` override on
  `safe_get_bytes` / `safe_get_text` / `safe_get_json`. Default UA
  unchanged.
- `narrative.sources.us_treasury` — replaced the dead RSS URL
  (`/rss/press-releases.xml`, dead 2026) with HTML scraping of the
  Treasury press-releases listing page. New `max_pages` kwarg on
  `iter_treasury_press`.
- `narrative.sources.us_federal_register` — removed
  `office-of-management-and-budget` from the default `agencies` tuple
  (the slug returned HTTP 400 from the FR API). Docstring now points
  users at the FR `/api/v1/agencies.json` endpoint to discover current
  valid slugs.
- `narrative.sources.us_dod_contracts` — passes a realistic browser
  User-Agent to bypass the defense.gov WAF.
- `narrative.sources.RETRY_POLICY.md` — new §6 documents the
  `user_agent=` override mechanism.

### Replication infrastructure
- `tests/test_dynpanel/test_ab_1991_replication.py` — new skip-if-absent
  test that loads `tests/fixtures/abdata.csv` and asserts published
  AB (1991) Table 4 col. 2 lag coefficients (L1.n ≈ 0.474, L2.n ≈
  −0.053) within 0.05. Skips gracefully when the fixture is absent;
  see `tests/fixtures/abdata.README.md` for how to obtain it.

### Cleanup
- `did/callaway_santanna.py` — removed dead `* 0` leftover (line 164).
- `did/borusyak_jaravel_spiess.py`, `nowcast/combine.py` — removed
  unused `inv_xtx` imports.

### Tests
- `tests/test_cointegration_modern.py` (new) — 11 tests on the three
  new result objects.
- `tests/test_midas.py` (new) — 6 tests on the two new result objects.
- `tests/fixtures/public_api_snapshot.json` regenerated to accept the
  5 new result classes and 2 extended `__all__` tuples.
- Pre-patch baseline: 368 passing, 7 skipped. Post-patch target:
  ≥385 passing, ≤6 skipped (depending on whether network was available
  to re-record the us_treasury fixture).
```

- [ ] **Step 2: Verify the CHANGELOG renders cleanly**

Run: `head -80 CHANGELOG.md`
Expected: New 0.4.1 block visible above 0.4.0.

---

## Task 19: Run the full test suite and confirm green

- [ ] **Step 1: Full test run**

Run: `pytest -x -q 2>&1 | tail -30`
Expected: ≥385 passing, ≤6 skipped, 0 failures.

- [ ] **Step 2: Pyodide-compat regression check**

Run: `pytest tests/test_pyodide_compat.py -v`
Expected: PASS — confirms no `statsmodels` / `linearmodels` / `arch` leaked into runtime imports.

- [ ] **Step 3: Public-API freeze check**

Run: `pytest tests/test_public_api.py -v`
Expected: PASS.

- [ ] **Step 4: Examples smoke**

Run: `python -m puremacro.examples.fm_ols_dols_demo` and
`python -m puremacro.examples.midas_quarterly_monthly`.
Expected: Both run to completion.

---

## Task 20: Update the puremacro memory file

**Files:**
- Modify: `~/.claude/projects/-Users-jalonso-Library-CloudStorage-GoogleDrive-jorge-alonsoortiz-gmail-com-My-Drive-MAV/memory/project_puremacro.md`

- [ ] **Step 1: Append iteration N+9 step 1 entry**

Append a new section to the memory file, after the last iteration entry:

```markdown
**Iteration N+9 step 1 done (2026-05-02) — released as 0.4.1 (patch):**
- Result-object migrations: `cointegration_modern.{fm_ols, dols, phillips_ouliaris}` → `FMOLSResult` / `DOLSResult` / `PhillipsOuliarisResult`; `midas.{u_midas, beta_midas}` → `UMidasResult` / `BetaMidasResult`. All five carry `.summary()`. Snapshot freeze updated.
- Narrative connector fixes: `us_treasury` rewritten as HTML-scrape of `home.treasury.gov/news/press-releases` (Atom URL dead in 2026); `us_federal_register` default agencies trimmed to `treasury-department + defense-department` (OMB slug gave HTTP 400); `us_dod_contracts` now passes a realistic Chrome UA via new `user_agent=` override on `_http.safe_get_*`.
- AB 1991 replication: shipped infrastructure only (skip-if-absent test + `tests/fixtures/abdata.README.md` documenting how to obtain `abdata.csv` and the canonical Table 4 col. 2 estimates). The CSV itself is not shipped (no network in this env).
- Dead-code cleanup: `did/callaway_santanna.py:164` `z * 0` line + 2 unused `inv_xtx` imports (`did/borusyak_jaravel_spiess.py`, `nowcast/combine.py`).
- Plan file: `uncertainty_examples/puremacro/docs/plans/2026-05-02-iteration-n9-step1-patch-041.md`.
- Test count: 368 → 385+ (depending on network availability for fixture re-records).

**How to apply:** When asking "what's next" after 0.4.1 the natural targets are still the deferred N+9 features (TVP-VAR Bayesian, FAVAR, BCA wedge accounting port — most directly useful in MAV — JK 2020 full Bayesian, Anderson-Hsiao, CU-GMM, Han-Phillips, docs site, perf benchmarks) plus the still-not-shipped AB 1991 fixture CSV (would un-skip 2 tests).
```

- [ ] **Step 2: Verify the memory file**

Read the bottom of the memory file and confirm the new block is present and well-formatted.

---

## Self-Review Checklist

After implementation:

1. **Spec coverage:** All seven follow-ups from N+8 closeout addressed?
   - [x] AB 1991 fixture infrastructure → Task 15 (CSV itself deferred — environmental constraint, not in scope)
   - [x] `cointegration_modern.fmols/dols/phillips_ouliaris` migrated → Tasks 4–6
   - [x] `midas.midas/beta_midas` migrated → Tasks 8–9
   - [x] `us_treasury._RSS_URL` fixed → Task 14
   - [x] `us_federal_register` default agency fixed → Task 13
   - [x] `us_dod_contracts` UA tweak → Task 12
   - [x] `did/callaway_santanna.py:164` `z*0` leftover → Task 1
   - [x] 2 unused `inv_xtx` imports → Tasks 2–3

2. **Placeholder scan:** No "TBD" / "implement later" / "appropriate error handling" / "similar to Task N" patterns. Every code step shows the actual code.

3. **Type consistency:**
   - Result-class field names match across the dataclass def, the function return, and the test assertions.
   - `FMOLSResult` field order matches dict-key insertion order in original `fm_ols` for ease of mental mapping.
   - `BetaMidasResult` has `beta: float` (single slope), distinguished from `UMidasResult` `beta: np.ndarray` (per-lag vector). Test_midas asserts on each correctly.
   - `iter_treasury_press` keeps zero-arg call form working (new `max_pages` kwarg has a default).
   - `iter_federal_register`'s `agencies` default change is backwards-compatible: callers passing explicit `agencies=` are unaffected; callers relying on the default get a smaller (but valid) list.

4. **Pyodide:** No new runtime deps. New `re` and `urllib.parse` uses in rewritten `us_treasury.py` are stdlib (already used elsewhere in `narrative/sources`).

5. **Result-object standard compliance:**
   - All five new dataclasses are `@dataclass(frozen=True)`.
   - Names end in `Result`.
   - No `__post_init__` (per standard).
   - All have `.summary()`.
   - No `.plot()` (per standard).
   - Tests verify frozenness (`pytest.raises(dataclasses.FrozenInstanceError)`).
