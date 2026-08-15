# Iteration N+8 Step 1 — Result-object standard + `puremacro.hfi`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the result-object standard in `ARCHITECTURE.md`, ship `puremacro.hfi` (Gertler-Karadi 2015 + Nakamura-Steinsson 2018 surprise construction, Jarociński-Karadi 2020 monetary-vs-information decomposition), and migrate `var.identify.proxy_svar` to a `ProxySVARResult` dataclass with Olea-Pflueger (2013) effective F.

**Architecture:** HFI surprise construction is pure-numpy under `puremacro/hfi/{surprises,jk2020}.py`. Decomposition results use `JKResult` dataclass per the standard. HFI does *not* duplicate external-IV machinery — it composes on top of `var.identify.proxy_svar`, which is migrated to a `ProxySVARResult` dataclass and gains an Olea-Pflueger F via a new `inference.weak_iv.olea_pflueger_f`.

**Tech Stack:** Python (≥3.10), numpy, scipy, pandas, pytest. No new runtime dependencies. Pyodide-compat enforced via `tests/test_pyodide_compat.py` (existing).

**Special note — no git:** The puremacro folder is not a git repository (lives in Google Drive). Standard "commit" steps are replaced with **verification checkpoints**: run the full test suite, confirm green, manually save state. The plan treats each task's checkpoint as the boundary at which an interruption is safe.

**Spec reference:** `docs/specs/2026-05-02-iteration-n8-design.md` — sections A and B.

---

## Task 1: Add result-object standard to `ARCHITECTURE.md`

**Files:**
- Modify: `ARCHITECTURE.md`

- [ ] **Step 1: Read current `ARCHITECTURE.md`**

Find an appropriate insertion point (after the existing API conventions section, before per-module dependency map).

- [ ] **Step 2: Insert the standard section**

Add this section verbatim:

```markdown
## Result-object standard (0.4.0+)

All public estimators that return three or more fields (or any non-trivial diagnostic) MUST return a frozen dataclass result object. The contract:

1. **`@dataclass(frozen=True)`** for any return with 3+ fields or non-trivial diagnostics. Frozen prevents accidental mutation post-fit; trivially picklable.
2. **Naming:** `<MethodName>Result` in PascalCase (e.g., `GMMResult`, `IRFResult`, `JKResult`, `ProxySVARResult`). Defined in `<subpackage>/_results.py`; re-exported via `<subpackage>/__init__.py`.
3. **Tuple returns** still allowed for genuinely simple two-value returns (e.g., `cycle, trend = hamilton_filter(y)`). The standard kicks in at 3+ fields or whenever there is a diagnostic to attach.
4. **Common field vocabulary** where applicable: `coefs`, `se`, `cov`, `names: tuple[str, ...]`, `n_obs`, `converged`. Method-specific diagnostics (e.g., `hansen_j` for GMM, `first_stage_F` for IV) live alongside.
5. **`.summary() -> str`** method optional but encouraged. Multi-line pretty-print. Each module writes its own — no autogeneration.
6. **No `.plot()` method.** Plotting stays in `puremacro.plot` and friends; result objects are pure data.
7. **No `__post_init__` validation that raises.** The estimator is responsible for building a valid result; the dataclass just stores it.

The public-API freeze test (`tests/test_public_api.py`, added in iteration N+8 step 3) snapshots both `__all__` per subpackage and result-class field names per dataclass; accidental field drift breaks loudly.

The 0.4.0 release migrates existing 3+ field returns. New code added from 0.4.0 onward MUST follow the standard from day one.
```

- [ ] **Step 3: Verification checkpoint**

Run: `grep -A2 "Result-object standard" ARCHITECTURE.md`
Expected: shows the new heading and at least the first paragraph.

---

## Task 2: Add `olea_pflueger_f` to `inference/weak_iv.py`

**Files:**
- Modify: `puremacro/inference/weak_iv.py`
- Test: `tests/test_inference/test_weak_iv.py`

The Olea-Pflueger (2013) "effective F" is the modern weak-instrument robust statistic for proxy-SVAR settings: more conservative than the Wald-style first-stage F when residuals are heteroskedastic, and replaces the Stock-Yogo critical-value tables. Computed as

$$F_{\text{eff}} = \frac{\hat{\Pi}' (Z'Z) \hat{\Pi}}{\text{tr}(W_2)}$$

where $W_2$ is the appropriate cluster-robust block of the first-stage variance, normalized to the number of regressors. For a single endogenous regressor and no controls (the proxy-SVAR base case), this collapses to a simple form.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_inference/test_weak_iv.py`:

```python
def test_olea_pflueger_f_strong_instrument():
    """Olea-Pflueger F on a strong instrument should clear the F=23 cutoff."""
    rng = np.random.default_rng(0)
    T = 500
    Z = rng.standard_normal((T, 1))
    eps = rng.standard_normal(T) * 0.5
    X = 2.0 * Z[:, 0] + eps                # strong first stage, beta=2
    f = olea_pflueger_f(X, Z)
    assert f > 23.0, f"strong instrument should clear F=23, got {f:.2f}"


def test_olea_pflueger_f_weak_instrument():
    """Weak instrument should produce a small F."""
    rng = np.random.default_rng(1)
    T = 500
    Z = rng.standard_normal((T, 1))
    X = 0.05 * Z[:, 0] + rng.standard_normal(T)
    f = olea_pflueger_f(X, Z)
    assert f < 5.0, f"weak instrument expected F < 5, got {f:.2f}"


def test_olea_pflueger_f_matches_homoskedastic_first_stage_f():
    """Under homoskedasticity with one instrument, OP-F approximately equals
    the standard first-stage F."""
    rng = np.random.default_rng(2)
    T = 1000
    Z = rng.standard_normal((T, 1))
    X = 1.5 * Z[:, 0] + rng.standard_normal(T)
    f_op = olea_pflueger_f(X, Z)
    # Standard first-stage F: (R^2 / (1 - R^2)) * (T - K)
    Pi = (X @ Z) / (Z.T @ Z)
    fitted = Z @ Pi
    R2 = 1 - np.sum((X - fitted) ** 2) / np.sum((X - X.mean()) ** 2)
    K = Z.shape[1]
    f_classical = (R2 / (1 - R2)) * (T - K)
    np.testing.assert_allclose(f_op, f_classical, rtol=0.05)
```

Add the import at the top of the test file:
```python
from puremacro.inference.weak_iv import olea_pflueger_f
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference/test_weak_iv.py -v -k olea_pflueger`
Expected: ImportError on `olea_pflueger_f`.

- [ ] **Step 3: Implement `olea_pflueger_f`**

Add to `puremacro/inference/weak_iv.py` (after `kleibergen_paap_f`):

```python
def olea_pflueger_f(
    x_endog: np.ndarray,
    z_inst: np.ndarray,
    cluster: np.ndarray | None = None,
) -> float:
    """Olea-Pflueger (2013) effective F-statistic for weak instruments.

    The effective F is the modern weak-IV-robust replacement for the
    Stock-Yogo Wald F. It is constructed to be conservative under
    heteroskedasticity and (optionally) clustering.

    Parameters
    ----------
    x_endog : ndarray, shape (T,) or (T, 1)
        Endogenous regressor (one-dimensional).
    z_inst : ndarray, shape (T, k)
        Instrument matrix (k instruments).
    cluster : ndarray of int, shape (T,), optional
        Cluster identifier. If provided, the variance is computed
        cluster-robustly; otherwise heteroskedasticity-robust (HC0).

    Returns
    -------
    f_eff : float
        Olea-Pflueger effective F-statistic. Reference cutoffs (5%
        worst-case bias, k=1): F > 23.1 (strong); F < 23.1 means
        weak-IV-robust inference recommended.

    References
    ----------
    Olea, J.L.M. and Pflueger, C. (2013). A robust test for weak
        instruments. JBES 31(3), 358-369.
    """
    x = np.asarray(x_endog).reshape(-1)
    Z = np.asarray(z_inst)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    T, k = Z.shape
    if x.shape[0] != T:
        raise ValueError(
            f"olea_pflueger_f: x_endog length {x.shape[0]} != z_inst rows {T}"
        )
    # Demean (the proxy-SVAR convention; OP also derive without intercept,
    # but demeaning is standard in practice).
    x = x - x.mean()
    Z = Z - Z.mean(axis=0, keepdims=True)
    ZtZ_inv = inv_xtx(Z, name="olea_pflueger_f")
    Pi = ZtZ_inv @ (Z.T @ x)              # shape (k,)
    resid = x - Z @ Pi
    if cluster is None:
        # HC0 sandwich on Z' u u' Z
        ZtuutZ = (Z.T * resid**2) @ Z
    else:
        cluster = np.asarray(cluster).reshape(-1)
        if cluster.shape[0] != T:
            raise ValueError("olea_pflueger_f: cluster length mismatch")
        ZtuutZ = np.zeros((k, k))
        for g in np.unique(cluster):
            mask = cluster == g
            ug = resid[mask]
            Zg = Z[mask]
            score_g = Zg.T @ ug
            ZtuutZ += np.outer(score_g, score_g)
    # Effective F: F_eff = Pi' (Z'Z) Pi / tr( (Z'Z)^{-1} Z' uu' Z )
    num = Pi @ (Z.T @ Z) @ Pi
    denom = np.trace(ZtZ_inv @ ZtuutZ)
    if denom <= 0:
        return float("inf")
    return float(num / denom)
```

The `inv_xtx` import already exists in `weak_iv.py` (used by `kleibergen_paap_f`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference/test_weak_iv.py -v -k olea_pflueger`
Expected: 3 passed.

- [ ] **Step 5: Update `__all__` if present in `weak_iv.py`**

Check: `grep -n "__all__" puremacro/inference/weak_iv.py`
If `__all__` is defined, add `"olea_pflueger_f"` to it.

- [ ] **Step 6: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`
Expected: all tests pass; new tests included.

---

## Task 3: Create `var/identify/_results.py` with `ProxySVARResult`

**Files:**
- Create: `puremacro/var/identify/_results.py`
- Test: `tests/test_var/test_identify_results.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_var/test_identify_results.py`:

```python
"""Tests for var.identify result dataclasses."""
import numpy as np
import pytest

from puremacro.var.identify._results import ProxySVARResult


def test_proxy_svar_result_is_frozen():
    res = ProxySVARResult(
        irf_point=np.zeros((3, 3, 5)),
        irf_lower=np.zeros((3, 3, 5)),
        irf_upper=np.zeros((3, 3, 5)),
        B=np.eye(3),
        first_stage_F=25.0,
        n_boot=500,
        ci=0.9,
    )
    with pytest.raises(Exception):  # FrozenInstanceError or similar
        res.first_stage_F = 99.0


def test_proxy_svar_result_summary():
    res = ProxySVARResult(
        irf_point=np.zeros((3, 3, 5)),
        irf_lower=np.zeros((3, 3, 5)),
        irf_upper=np.zeros((3, 3, 5)),
        B=np.eye(3),
        first_stage_F=25.0,
        n_boot=500,
        ci=0.9,
    )
    s = res.summary()
    assert "ProxySVAR" in s
    assert "25.0" in s or "25.00" in s


def test_proxy_svar_result_picklable():
    import pickle
    res = ProxySVARResult(
        irf_point=np.zeros((2, 2, 3)),
        irf_lower=np.zeros((2, 2, 3)),
        irf_upper=np.zeros((2, 2, 3)),
        B=np.eye(2),
        first_stage_F=12.0,
        n_boot=100,
        ci=0.9,
    )
    blob = pickle.dumps(res)
    res2 = pickle.loads(blob)
    np.testing.assert_array_equal(res2.B, res.B)
    assert res2.first_stage_F == 12.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_var/test_identify_results.py -v`
Expected: ImportError on `ProxySVARResult`.

- [ ] **Step 3: Create `var/identify/_results.py`**

```python
"""Frozen-dataclass result objects for var.identify estimators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProxySVARResult:
    """Result of :func:`puremacro.var.identify.proxy.proxy_svar`.

    Attributes
    ----------
    irf_point : ndarray, shape (n, n, H+1)
        Point-estimate impulse responses.
    irf_lower : ndarray, shape (n, n, H+1)
        Lower bootstrap band.
    irf_upper : ndarray, shape (n, n, H+1)
        Upper bootstrap band.
    B : ndarray, shape (n, n)
        Identified structural impact matrix; column 0 is the proxy-identified shock.
    first_stage_F : float
        Olea-Pflueger (2013) effective F-statistic on the proxy.
    n_boot : int
        Number of bootstrap draws used.
    ci : float
        Confidence-interval level (e.g., 0.9 for 90% bands).
    """

    irf_point: np.ndarray
    irf_lower: np.ndarray
    irf_upper: np.ndarray
    B: np.ndarray
    first_stage_F: float
    n_boot: int
    ci: float

    def summary(self) -> str:
        n = self.B.shape[0]
        H = self.irf_point.shape[2] - 1
        f_flag = "STRONG" if self.first_stage_F > 23.0 else "WEAK (use weak-IV-robust inference)"
        return (
            f"ProxySVAR result\n"
            f"  shocks identified : 1 (column 0 of B)\n"
            f"  variables (n)     : {n}\n"
            f"  horizon (H)       : {H}\n"
            f"  bootstrap draws   : {self.n_boot}\n"
            f"  CI level          : {self.ci:.2f}\n"
            f"  first-stage F (OP): {self.first_stage_F:.2f}  [{f_flag}]\n"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_var/test_identify_results.py -v`
Expected: 3 passed.

- [ ] **Step 5: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`
Expected: all tests pass.

---

## Task 4: Migrate `proxy_svar` to return `ProxySVARResult`; update callers

**Files:**
- Modify: `puremacro/var/identify/proxy.py`
- Modify: `puremacro/narrative/types.py` (the `Instrument.to_proxy_svar` wrapper)
- Modify: `puremacro/examples/svariv_mertens_ravn.py`
- Modify: `puremacro/examples/narrative_ramey_2011.py`
- Test: `tests/test_var/test_proxy_svar.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_var/test_proxy_svar.py`:

```python
"""Smoke + return-shape tests for proxy_svar with the 0.4.0 ProxySVARResult API."""
import numpy as np
import pytest

from puremacro.var.identify.proxy import proxy_svar
from puremacro.var.identify._results import ProxySVARResult


def _synthetic_var_with_proxy(T=300, n=3, p=2, seed=0):
    rng = np.random.default_rng(seed)
    # Generate VAR(2) with a known structural shock
    A1 = 0.5 * np.eye(n) + 0.05 * rng.standard_normal((n, n))
    A2 = 0.1 * np.eye(n) + 0.02 * rng.standard_normal((n, n))
    eps = rng.standard_normal((T, n)) * 0.5
    Y = np.zeros((T, n))
    for t in range(p, T):
        Y[t] = A1 @ Y[t-1] + A2 @ Y[t-2] + eps[t]
    # Proxy: noisy version of the first structural shock
    z = eps[:, 0] + 0.3 * rng.standard_normal(T)
    return Y, z


def test_proxy_svar_returns_result_dataclass():
    Y, z = _synthetic_var_with_proxy()
    res = proxy_svar(Y, p=2, horizon=10, instrument_series=z, n_boot=50, ci=0.9, seed=0)
    assert isinstance(res, ProxySVARResult)


def test_proxy_svar_irf_shape():
    Y, z = _synthetic_var_with_proxy()
    res = proxy_svar(Y, p=2, horizon=12, instrument_series=z, n_boot=50, ci=0.9, seed=0)
    n = Y.shape[1]
    assert res.irf_point.shape == (n, n, 13)
    assert res.irf_lower.shape == (n, n, 13)
    assert res.irf_upper.shape == (n, n, 13)
    assert res.B.shape == (n, n)
    assert (res.irf_lower <= res.irf_point).all() or np.allclose(res.irf_lower, res.irf_point)
    assert (res.irf_upper >= res.irf_point).all() or np.allclose(res.irf_upper, res.irf_point)


def test_proxy_svar_first_stage_F_is_finite_positive():
    Y, z = _synthetic_var_with_proxy()
    res = proxy_svar(Y, p=2, horizon=10, instrument_series=z, n_boot=50, ci=0.9, seed=0)
    assert np.isfinite(res.first_stage_F)
    assert res.first_stage_F > 0


def test_proxy_svar_strong_instrument_clears_op_cutoff():
    """Synthetic high-correlation instrument should produce F > 23 (Olea-Pflueger
    'strong' threshold)."""
    rng = np.random.default_rng(42)
    T = 500
    Y, _ = _synthetic_var_with_proxy(T=T, seed=42)
    # Instrument: dominant linear function of Y residuals' first column
    # Use a tighter proxy by reducing noise relative to signal
    eps = np.diff(Y, axis=0, prepend=Y[:1])  # rough residuals
    z = eps[:, 0] + 0.1 * rng.standard_normal(T)
    res = proxy_svar(Y, p=2, horizon=10, instrument_series=z, n_boot=50, ci=0.9, seed=0)
    assert res.first_stage_F > 23.0, f"strong proxy should clear F=23, got {res.first_stage_F:.2f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_var/test_proxy_svar.py -v`
Expected: AssertionError on `isinstance(res, ProxySVARResult)` — current return is a tuple.

- [ ] **Step 3: Migrate `proxy_svar`**

Replace `puremacro/var/identify/proxy.py` with:

```python
"""Proxy-SVAR / external-instrument identification (Mertens-Ravn 2013, Stock-Watson 2018).

The 0.4.0 release migrates the return type from a 3-tuple (point, lo, hi) to a
:class:`ProxySVARResult` frozen dataclass that additionally carries the impact
matrix ``B`` and the Olea-Pflueger effective F.
"""
from __future__ import annotations

import numpy as np

from ..estimate import estimate_var
from ._results import ProxySVARResult
from ...inference.weak_iv import olea_pflueger_f

try:
    from ...inference.wild_bootstrap import wild_bootstrap_var
except ImportError:
    wild_bootstrap_var = None


def _proxy_impact_factory(instrument_series, shock_target_idx=0):
    def _impact(A_list, Sigma, resid):
        T_eff = resid.shape[0]
        z = np.asarray(instrument_series)[-T_eff:]
        z = z - z.mean()
        Pi = (resid.T @ z) / (z @ z)
        norm = np.sqrt(Pi @ Sigma @ Pi)
        if norm < 1e-10:
            raise np.linalg.LinAlgError("Degenerate instrument in proxy-SVAR.")
        B_col1 = (Sigma @ Pi) / norm
        n = Sigma.shape[0]
        B = np.zeros((n, n))
        B[:, 0] = B_col1
        residual_cov = Sigma - np.outer(B_col1, B_col1)
        u, s, _ = np.linalg.svd(residual_cov)
        rank = int(np.sum(s > 1e-8))
        for k in range(min(rank, n - 1)):
            B[:, 1 + k] = u[:, k] * np.sqrt(max(s[k], 0))
        return B
    return _impact


def proxy_svar(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    instrument_series: np.ndarray,
    shock_target_idx: int = 0,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> ProxySVARResult:
    """Proxy-SVAR identification via external instrument (Mertens-Ravn 2013).

    Parameters
    ----------
    Y : ndarray, shape (T, n)
        VAR data.
    p : int
        VAR lag order.
    horizon : int
        IRF horizon (returns ``horizon+1`` periods).
    instrument_series : ndarray, shape (T,)
        External instrument / proxy. The last ``T - p`` observations are aligned
        to the VAR residuals.
    shock_target_idx : int, default 0
        Index of the structural shock targeted by the proxy. The identified shock
        is placed in column 0 of ``B`` regardless.
    n_boot : int, default 500
        Number of wild-bootstrap draws.
    ci : float, default 0.9
        Confidence-interval level.
    seed : int, default 0
        Bootstrap RNG seed.

    Returns
    -------
    ProxySVARResult
        See :class:`puremacro.var.identify._results.ProxySVARResult`.
    """
    A_list, c, Sigma, resid, _ = estimate_var(Y, p)
    T_eff = resid.shape[0]
    z = np.asarray(instrument_series)[-T_eff:]
    # Olea-Pflueger F is computed on the first VAR residual against the proxy.
    f_eff = olea_pflueger_f(resid[:, shock_target_idx], z.reshape(-1, 1))

    impact_fn = _proxy_impact_factory(instrument_series, shock_target_idx)
    B = impact_fn(A_list, Sigma, resid)

    if wild_bootstrap_var is None:
        raise ImportError(
            "wild_bootstrap_var is not available. "
            "puremacro.inference.wild_bootstrap must be installed."
        )
    point, lo, hi = wild_bootstrap_var(
        Y, p=p, horizon=horizon, impact_fn=impact_fn,
        n_boot=n_boot, ci=ci, seed=seed,
    )
    return ProxySVARResult(
        irf_point=point,
        irf_lower=lo,
        irf_upper=hi,
        B=B,
        first_stage_F=float(f_eff),
        n_boot=n_boot,
        ci=ci,
    )
```

- [ ] **Step 4: Update `Instrument.to_proxy_svar` in `narrative/types.py`**

The wrapper (around line 132-156) currently passes through whatever `proxy_svar` returns. With the migration it now returns a `ProxySVARResult`. Update the docstring:

Find the existing block at `narrative/types.py:132-156` and change the docstring to reflect the new return type:

```python
    def to_proxy_svar(self, Y, *, p: int, horizon: int,
                     n_boot: int = 500, ci: float = 0.9, seed: int = 0):
        """Run :func:`puremacro.var.identify.proxy.proxy_svar` using this
        instrument as the external proxy.

        Aligns ``self.quarterly.values`` to the VAR sample and returns the
        full :class:`puremacro.var.identify._results.ProxySVARResult` (0.4.0+).
        """
        from ..var.identify.proxy import proxy_svar
        z = np.asarray(self.quarterly.values, dtype=float)
        return proxy_svar(Y, p=p, horizon=horizon,
                          instrument_series=z, n_boot=n_boot, ci=ci, seed=seed)
```

(Match the existing arg signature; only the docstring and the return value change.)

- [ ] **Step 5: Update `examples/svariv_mertens_ravn.py`**

Find:
```python
    point, lo, hi = proxy_svar(
```
Replace with:
```python
    res = proxy_svar(
```
and downstream replace `point` → `res.irf_point`, `lo` → `res.irf_lower`, `hi` → `res.irf_upper` everywhere they appear after that line. Add a print of `res.first_stage_F` near the end of the script.

- [ ] **Step 6: Update `examples/narrative_ramey_2011.py`**

Find the defensive unpacking at `examples/narrative_ramey_2011.py` near line 122-126:

```python
    res = inst.to_proxy_svar(out["Y"], p=2, horizon=20,
                              z=z, n_boot=200, ci=0.9, seed=0)
    irf_point = res[0] if isinstance(res, tuple) else res.get("point", None)
```

Replace with:

```python
    res = inst.to_proxy_svar(out["Y"], p=2, horizon=20,
                              n_boot=200, ci=0.9, seed=0)
    irf_point = res.irf_point
    print(f"  Olea-Pflueger F: {res.first_stage_F:.2f}")
```

(Note: drop `z=z` if `to_proxy_svar` doesn't accept it — it doesn't in the current signature; the proxy comes from `self.quarterly`.)

- [ ] **Step 7: Run target tests**

Run: `pytest tests/test_var/test_proxy_svar.py tests/test_var/test_identify_results.py -v`
Expected: all pass.

- [ ] **Step 8: Run example smoke tests if present**

Run: `python -m puremacro.examples.svariv_mertens_ravn 2>&1 | head -30`
Expected: runs without exception; prints first-stage F line.

Run: `python -m puremacro.examples.narrative_ramey_2011 2>&1 | head -30`
Expected: runs without exception (or fails only on its own data-fetch step, not on proxy_svar API).

- [ ] **Step 9: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`
Expected: full suite green.

---

## Task 5: Scaffold `puremacro/hfi/` package and `JKResult`

**Files:**
- Create: `puremacro/hfi/__init__.py`
- Create: `puremacro/hfi/_results.py`
- Test: `tests/test_hfi/__init__.py` (empty), `tests/test_hfi/test_results.py`

- [ ] **Step 1: Create empty test scaffold**

Create `tests/test_hfi/__init__.py` as an empty file.

Create `tests/test_hfi/test_results.py`:

```python
"""Tests for puremacro.hfi result dataclasses."""
import numpy as np
import pytest

from puremacro.hfi._results import JKResult


def test_jk_result_is_frozen():
    res = JKResult(
        mp_shock=np.zeros(10),
        info_shock=np.zeros(10),
        rotation=None,
        n_admissible=None,
        method="poor_man",
    )
    with pytest.raises(Exception):
        res.method = "median_target"


def test_jk_result_summary_poor_man():
    res = JKResult(
        mp_shock=np.array([1.0, 0.0, -2.0]),
        info_shock=np.array([0.0, 3.0, 0.0]),
        rotation=None,
        n_admissible=None,
        method="poor_man",
    )
    s = res.summary()
    assert "poor_man" in s.lower() or "Poor" in s


def test_jk_result_summary_median_target():
    res = JKResult(
        mp_shock=np.zeros(5),
        info_shock=np.zeros(5),
        rotation=np.eye(2),
        n_admissible=1234,
        method="median_target",
    )
    s = res.summary()
    assert "median_target" in s.lower() or "Median" in s
    assert "1234" in s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hfi/test_results.py -v`
Expected: ImportError on `puremacro.hfi`.

- [ ] **Step 3: Create `puremacro/hfi/_results.py`**

```python
"""Frozen-dataclass result objects for puremacro.hfi."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JKResult:
    """Result of :func:`puremacro.hfi.jk2020.jk_poor_man` and
    :func:`puremacro.hfi.jk2020.jk_median_target`.

    Attributes
    ----------
    mp_shock : ndarray, shape (T,)
        Identified monetary-policy shock series. Zero where the info-shock
        category fired (poor-man variant) or projected onto the MP rotation
        column (median-target variant).
    info_shock : ndarray, shape (T,)
        Identified central-bank-information shock series.
    rotation : ndarray of shape (2, 2) or None
        Rotation matrix used by the median-target variant. None for poor-man.
    n_admissible : int or None
        Number of admissible rotations searched (median-target). None for poor-man.
    method : str
        Either ``"poor_man"`` or ``"median_target"``.
    """

    mp_shock: np.ndarray
    info_shock: np.ndarray
    rotation: np.ndarray | None
    n_admissible: int | None
    method: str

    def summary(self) -> str:
        T = self.mp_shock.shape[0]
        lines = [
            f"Jarociński-Karadi (2020) decomposition",
            f"  method            : {self.method}",
            f"  observations      : {T}",
            f"  MP shock var      : {float(np.var(self.mp_shock)):.4f}",
            f"  Info shock var    : {float(np.var(self.info_shock)):.4f}",
        ]
        if self.method == "median_target":
            lines.append(f"  admissible rots   : {self.n_admissible}")
        return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Create `puremacro/hfi/__init__.py`**

```python
"""High-frequency identification of monetary policy shocks.

Provides:
    - Surprise construction: Gertler-Karadi (2015), Nakamura-Steinsson (2018).
    - Jarociński-Karadi (2020) monetary-vs-information decomposition.

For external-IV SVAR, pipe surprises into
:func:`puremacro.var.identify.proxy.proxy_svar` directly.
"""
from ._results import JKResult

__all__ = ["JKResult"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hfi/test_results.py -v`
Expected: 3 passed.

- [ ] **Step 6: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`
Expected: full suite green; `puremacro.hfi` importable.

---

## Task 6: Implement `gk2015_surprise`

**Files:**
- Create: `puremacro/hfi/surprises.py`
- Modify: `puremacro/hfi/__init__.py`
- Test: `tests/test_hfi/test_surprises.py`

The Gertler-Karadi 2015 surprise scales the change in the federal-funds-futures rate around an FOMC announcement by the factor `M / (M - d)`, where `M` is the number of days in the announcement month and `d` is the days elapsed before the announcement. This adjusts for the fact that the FFR future settles on the *average* effective FFR for the month — only the post-announcement portion of the month is informative about the surprise.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hfi/test_surprises.py`:

```python
"""Tests for puremacro.hfi.surprises."""
import numpy as np
import pytest

from puremacro.hfi.surprises import gk2015_surprise


def test_gk2015_surprise_no_scaling_at_month_start():
    """When announcement is on day 0 of the month, scaling factor is 1
    (M / (M - 0) = 1)."""
    pre = np.array([95.0])     # 100 - rate, so rate=5.0
    post = np.array([95.05])   # rate=4.95 → -5bp surprise (rate down)
    days_remaining = np.array([30])  # full month remaining
    s = gk2015_surprise(pre, post, days_remaining_in_month=days_remaining,
                        days_in_month=30)
    np.testing.assert_allclose(s, post - pre)


def test_gk2015_surprise_scaling_at_month_end():
    """Announcement near month end → scaling factor blows up (M / (M - d) large
    when d → M-1)."""
    pre = np.array([95.0])
    post = np.array([95.05])
    days_remaining = np.array([1])
    s = gk2015_surprise(pre, post, days_remaining_in_month=days_remaining,
                        days_in_month=30)
    # scale = 30 / 1 = 30
    np.testing.assert_allclose(s, (post - pre) * 30)


def test_gk2015_surprise_vector_inputs():
    """Multiple announcements aggregated correctly."""
    pre = np.array([95.0, 96.0, 97.0])
    post = np.array([95.10, 95.95, 97.05])
    days_remaining = np.array([15, 5, 25])
    s = gk2015_surprise(pre, post, days_remaining_in_month=days_remaining,
                        days_in_month=30)
    # raw changes: 0.10, -0.05, 0.05; scales: 30/15, 30/5, 30/25 = 2, 6, 1.2
    np.testing.assert_allclose(s, np.array([0.10 * 2, -0.05 * 6, 0.05 * 1.2]))


def test_gk2015_surprise_rejects_zero_remaining():
    """days_remaining=0 would imply announcement after month end — should error."""
    with pytest.raises(ValueError):
        gk2015_surprise(np.array([95.0]), np.array([95.1]),
                        days_remaining_in_month=np.array([0]),
                        days_in_month=30)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hfi/test_surprises.py -v -k gk2015`
Expected: ImportError.

- [ ] **Step 3: Create `puremacro/hfi/surprises.py`**

```python
"""High-frequency surprise construction.

Public functions:
    - :func:`gk2015_surprise`  — Gertler-Karadi 2015 month-end-adjusted FFR-futures change.
    - :func:`ns2018_first_pc`  — Nakamura-Steinsson 2018 first-PC of multiple policy contracts.
    - :func:`aggregate_to_period` — sum announcement-day surprises into monthly/quarterly bins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def gk2015_surprise(
    ff_futures_pre: np.ndarray,
    ff_futures_post: np.ndarray,
    days_remaining_in_month: np.ndarray,
    days_in_month: int | np.ndarray = 30,
) -> np.ndarray:
    """Gertler-Karadi (2015) high-frequency monetary surprise.

    Computes ``(post - pre) * M / (M - d_elapsed)`` where ``d_elapsed = M - days_remaining``.
    The scaling factor adjusts for the fact that federal-funds-futures payoff
    is the *average* effective FFR over the month: only the post-announcement
    portion of the month carries the shock.

    Parameters
    ----------
    ff_futures_pre : ndarray, shape (n_announce,)
        Futures price (or implied rate) immediately before the announcement.
    ff_futures_post : ndarray, shape (n_announce,)
        Futures price (or implied rate) immediately after the announcement.
    days_remaining_in_month : ndarray of int, shape (n_announce,)
        Calendar days remaining in the announcement month *including* the
        announcement day. Must be > 0.
    days_in_month : int or ndarray, default 30
        Total calendar days in the announcement month. Pass an array if the
        month length varies across announcements.

    Returns
    -------
    surprise : ndarray, shape (n_announce,)
        Scale-adjusted high-frequency surprise (same units as the inputs).

    References
    ----------
    Gertler, M. and Karadi, P. (2015). Monetary policy surprises, credit
        costs, and economic activity. AEJ:Macro 7(1), 44-76.
    """
    pre = np.asarray(ff_futures_pre, dtype=float)
    post = np.asarray(ff_futures_post, dtype=float)
    rem = np.asarray(days_remaining_in_month, dtype=float)
    M = np.asarray(days_in_month, dtype=float)
    if np.any(rem <= 0):
        raise ValueError(
            "gk2015_surprise: days_remaining_in_month must be > 0 "
            "(announcement on or after the last day of the month is invalid)."
        )
    return (post - pre) * (M / rem)
```

- [ ] **Step 4: Re-export from `puremacro/hfi/__init__.py`**

Update `puremacro/hfi/__init__.py`:

```python
"""High-frequency identification of monetary policy shocks.

Provides:
    - Surprise construction: Gertler-Karadi (2015), Nakamura-Steinsson (2018).
    - Jarociński-Karadi (2020) monetary-vs-information decomposition.

For external-IV SVAR, pipe surprises into
:func:`puremacro.var.identify.proxy.proxy_svar` directly.
"""
from ._results import JKResult
from .surprises import gk2015_surprise

__all__ = ["JKResult", "gk2015_surprise"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hfi/test_surprises.py -v -k gk2015`
Expected: 4 passed.

- [ ] **Step 6: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`
Expected: full suite green.

---

## Task 7: Implement `ns2018_first_pc`

**Files:**
- Modify: `puremacro/hfi/surprises.py`
- Modify: `puremacro/hfi/__init__.py`
- Modify: `tests/test_hfi/test_surprises.py`

The Nakamura-Steinsson 2018 surprise is the first principal component of K policy-sensitive futures' announcement-window changes, rescaled so a unit corresponds to a 1pp surprise in a designated "target" contract.

- [ ] **Step 1: Append the failing test**

Append to `tests/test_hfi/test_surprises.py`:

```python
from puremacro.hfi.surprises import ns2018_first_pc


def test_ns2018_first_pc_recovers_dominant_factor():
    """If a single latent factor drives all K contracts, the first PC should
    recover it (up to sign) and explain near-100% of variance."""
    rng = np.random.default_rng(0)
    T, K = 200, 5
    factor = rng.standard_normal(T)
    loadings = rng.uniform(0.5, 1.5, K)
    surprise_matrix = np.outer(factor, loadings) + 0.01 * rng.standard_normal((T, K))
    pc, recovered_loadings = ns2018_first_pc(surprise_matrix, scale_to_idx=0)
    # Correlation with true factor should be near ±1
    corr = np.corrcoef(pc, factor)[0, 1]
    assert abs(corr) > 0.99


def test_ns2018_first_pc_scaling_to_target_contract():
    """When ``scale_to_idx=k``, a unit of the PC should correspond to ~1 unit
    of contract k (under perfect-correlation conditions)."""
    rng = np.random.default_rng(1)
    T, K = 300, 4
    factor = rng.standard_normal(T)
    loadings = np.array([1.0, 2.0, 0.5, 1.5])
    surprise_matrix = np.outer(factor, loadings)  # noiseless, perfect correlation
    pc, recovered_loadings = ns2018_first_pc(surprise_matrix, scale_to_idx=1)
    # The PC, when multiplied by the recovered loading on contract 1, should
    # recover contract 1's series.
    np.testing.assert_allclose(
        pc * recovered_loadings[1], surprise_matrix[:, 1], atol=1e-8
    )


def test_ns2018_first_pc_orthogonal_to_residual():
    """The first-PC should be orthogonal to the residual signal (X - pc·loadings')."""
    rng = np.random.default_rng(2)
    surprise_matrix = rng.standard_normal((150, 4))
    pc, loadings = ns2018_first_pc(surprise_matrix, scale_to_idx=0)
    residual = surprise_matrix - np.outer(pc, loadings)
    np.testing.assert_allclose(pc @ residual, 0.0, atol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hfi/test_surprises.py -v -k ns2018`
Expected: ImportError.

- [ ] **Step 3: Implement `ns2018_first_pc`**

Append to `puremacro/hfi/surprises.py`:

```python
def ns2018_first_pc(
    surprise_matrix: np.ndarray,
    scale_to_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nakamura-Steinsson (2018) first-PC of multiple announcement-window changes.

    Computes the first principal component of K policy-sensitive contracts'
    announcement-window changes, rescaled so a unit of the PC corresponds to a
    unit change in the contract at ``scale_to_idx``.

    Parameters
    ----------
    surprise_matrix : ndarray, shape (n_announce, K)
        Matrix of K contracts' surprises across n_announce announcements.
    scale_to_idx : int, default 0
        Index of the contract to which the PC is rescaled. The recovered
        loading on this contract is positive by construction.

    Returns
    -------
    pc : ndarray, shape (n_announce,)
        First-PC time series, scaled in the units of the target contract.
    loadings : ndarray, shape (K,)
        Recovered factor loadings (one entry per contract).

    Notes
    -----
    Uses SVD of the demeaned surprise matrix. The PC is sign-normalized so
    the loading on the target contract is positive.

    References
    ----------
    Nakamura, E. and Steinsson, J. (2018). High-frequency identification
        of monetary non-neutrality: the information effect.
        QJE 133(3), 1283-1330.
    """
    X = np.asarray(surprise_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            f"ns2018_first_pc: surprise_matrix must be 2-D, got shape {X.shape}"
        )
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc_raw = U[:, 0] * S[0]                # length n_announce
    loadings_raw = Vt[0, :]                # length K
    # Sign-normalise so loading at target contract is positive
    if loadings_raw[scale_to_idx] < 0:
        pc_raw = -pc_raw
        loadings_raw = -loadings_raw
    # Rescale: PC in units of target contract → loading[scale_to_idx] = 1 / scale
    target_loading = loadings_raw[scale_to_idx]
    if abs(target_loading) < 1e-12:
        raise np.linalg.LinAlgError(
            f"ns2018_first_pc: target contract idx {scale_to_idx} has zero "
            f"loading on first PC; choose a different scale_to_idx."
        )
    pc = pc_raw * target_loading
    loadings = loadings_raw / target_loading
    return pc, loadings
```

- [ ] **Step 4: Re-export from `__init__.py`**

Update `puremacro/hfi/__init__.py`:

```python
from .surprises import gk2015_surprise, ns2018_first_pc

__all__ = ["JKResult", "gk2015_surprise", "ns2018_first_pc"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hfi/test_surprises.py -v -k ns2018`
Expected: 3 passed.

- [ ] **Step 6: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`

---

## Task 8: Implement `aggregate_to_period`

**Files:**
- Modify: `puremacro/hfi/surprises.py`
- Modify: `puremacro/hfi/__init__.py`
- Modify: `tests/test_hfi/test_surprises.py`

- [ ] **Step 1: Append the failing test**

Append to `tests/test_hfi/test_surprises.py`:

```python
from puremacro.hfi.surprises import aggregate_to_period


def test_aggregate_to_period_monthly():
    """Sum announcements within each month."""
    surprises = np.array([0.10, -0.05, 0.20, 0.03])
    dates = pd.to_datetime(["2020-01-15", "2020-01-29", "2020-02-12", "2020-03-08"])
    out = aggregate_to_period(surprises, dates, freq="M")
    assert out.loc["2020-01"] == pytest.approx(0.05)
    assert out.loc["2020-02"] == pytest.approx(0.20)
    assert out.loc["2020-03"] == pytest.approx(0.03)


def test_aggregate_to_period_quarterly():
    surprises = np.array([1.0, 2.0, 3.0])
    dates = pd.to_datetime(["2020-01-15", "2020-02-12", "2020-04-08"])
    out = aggregate_to_period(surprises, dates, freq="Q")
    assert out.loc["2020Q1"] == pytest.approx(3.0)
    assert out.loc["2020Q2"] == pytest.approx(3.0)


def test_aggregate_to_period_fills_missing_with_zero():
    """Periods with no announcement should appear with value 0, not be dropped."""
    surprises = np.array([1.0, 2.0])
    dates = pd.to_datetime(["2020-01-15", "2020-04-08"])
    out = aggregate_to_period(surprises, dates, freq="M")
    # Feb and Mar 2020 should be present and zero
    assert "2020-02" in out.index.astype(str).tolist() or out.loc["2020-02"] == 0.0
    assert out.loc["2020-02"] == pytest.approx(0.0)
    assert out.loc["2020-03"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hfi/test_surprises.py -v -k aggregate`
Expected: ImportError.

- [ ] **Step 3: Implement `aggregate_to_period`**

Append to `puremacro/hfi/surprises.py`:

```python
def aggregate_to_period(
    surprises: np.ndarray,
    dates,
    freq: str = "M",
) -> pd.Series:
    """Sum announcement-day surprises into period bins (monthly, quarterly).

    Periods with no announcement appear with value ``0.0``, not dropped — this
    matches the convention used in macro VARs where a "no announcement" period
    is informationally equivalent to a zero-surprise period.

    Parameters
    ----------
    surprises : ndarray, shape (n_announce,)
        Surprise series (e.g., output of :func:`gk2015_surprise`).
    dates : array-like of datetimes, length n_announce
        Announcement dates.
    freq : str, default "M"
        Pandas offset alias. ``"M"`` = month-end, ``"Q"`` = quarter-end.

    Returns
    -------
    pd.Series
        Indexed by period (PeriodIndex), values are the sum of all surprises
        falling within that period; periods between min(dates) and max(dates)
        with no announcement are zero-filled.
    """
    s = pd.Series(np.asarray(surprises, dtype=float), index=pd.to_datetime(dates))
    grouped = s.groupby(s.index.to_period(freq)).sum()
    full_idx = pd.period_range(grouped.index.min(), grouped.index.max(), freq=freq)
    return grouped.reindex(full_idx, fill_value=0.0)
```

- [ ] **Step 4: Re-export from `__init__.py`**

Update `puremacro/hfi/__init__.py`:

```python
from .surprises import aggregate_to_period, gk2015_surprise, ns2018_first_pc

__all__ = ["JKResult", "aggregate_to_period", "gk2015_surprise", "ns2018_first_pc"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hfi/test_surprises.py -v -k aggregate`
Expected: 3 passed.

- [ ] **Step 6: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`

---

## Task 9: Implement `jk_poor_man`

**Files:**
- Create: `puremacro/hfi/jk2020.py`
- Modify: `puremacro/hfi/__init__.py`
- Test: `tests/test_hfi/test_jk2020.py`

Poor-man's JK 2020 decomposition: when the rate surprise and asset surprise have *opposite* signs (rate up + stocks down, or rate down + stocks up), the announcement is a contractionary/expansionary monetary-policy shock; when they have the *same* sign, it is a central-bank-information shock.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hfi/test_jk2020.py`:

```python
"""Tests for puremacro.hfi.jk2020."""
import numpy as np
import pytest

from puremacro.hfi._results import JKResult
from puremacro.hfi.jk2020 import jk_poor_man


def test_jk_poor_man_separates_by_sign():
    rate = np.array([+0.10, -0.05, +0.20, +0.03])
    asset = np.array([-1.0, +0.5, +1.5, -0.4])
    # idx 0: rate>0, asset<0 → MP
    # idx 1: rate<0, asset>0 → MP
    # idx 2: rate>0, asset>0 → info
    # idx 3: rate>0, asset<0 → MP
    res = jk_poor_man(rate, asset)
    assert isinstance(res, JKResult)
    assert res.method == "poor_man"
    np.testing.assert_array_equal(res.mp_shock, [+0.10, -0.05, 0.0, +0.03])
    np.testing.assert_array_equal(res.info_shock, [0.0, 0.0, +0.20, 0.0])


def test_jk_poor_man_zero_rate_zero_shock():
    """Zero rate surprise → no shock attribution either way."""
    rate = np.array([0.0, 0.5, 0.0])
    asset = np.array([1.0, 1.0, -1.0])
    res = jk_poor_man(rate, asset)
    np.testing.assert_array_equal(res.mp_shock, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(res.info_shock, [0.0, 0.5, 0.0])


def test_jk_poor_man_orthogonality_in_support():
    """At every t, exactly one of (mp, info) is non-zero (or both zero)."""
    rng = np.random.default_rng(0)
    rate = rng.standard_normal(50)
    asset = rng.standard_normal(50)
    res = jk_poor_man(rate, asset)
    both_nonzero = (res.mp_shock != 0) & (res.info_shock != 0)
    assert not both_nonzero.any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hfi/test_jk2020.py -v -k poor_man`
Expected: ImportError.

- [ ] **Step 3: Create `puremacro/hfi/jk2020.py`**

```python
"""Jarociński-Karadi (2020) monetary-vs-information decomposition.

The high-frequency surprise around a central-bank announcement reflects two
shocks: a "monetary policy" shock (Taylor-rule innovation) and a "central bank
information" shock (the central bank revealing private information about the
economy). JK 2020 separate them via sign restrictions on the joint reaction of
the policy rate and a broad asset price (typically S&P 500).

Two variants are shipped in 0.4.0:
    - :func:`jk_poor_man`        — sign-of-comovement attribution.
    - :func:`jk_median_target`   — median admissible rotation under
                                    sign restrictions.

The full Bayesian sign-restriction variant is deferred to 0.5.0+.
"""
from __future__ import annotations

import numpy as np

from ._results import JKResult


def jk_poor_man(
    rate_surprise: np.ndarray,
    asset_surprise: np.ndarray,
) -> JKResult:
    """Jarociński-Karadi (2020) "poor-man's" decomposition.

    Attributes each announcement to either a monetary-policy shock or a
    central-bank-information shock based on the sign comovement of the rate
    and asset surprises:

    - Opposite-sign (rate up + asset down, or rate down + asset up) → MP shock.
    - Same-sign (both up or both down) → information shock.

    Within each category, the rate surprise carries through; the other category
    is zero at that announcement.

    Parameters
    ----------
    rate_surprise : ndarray, shape (T,)
        High-frequency interest-rate surprise.
    asset_surprise : ndarray, shape (T,)
        High-frequency broad-asset (e.g., S&P 500) surprise in the same window.

    Returns
    -------
    JKResult with ``method="poor_man"`` and ``rotation=None``, ``n_admissible=None``.
    """
    rate = np.asarray(rate_surprise, dtype=float)
    asset = np.asarray(asset_surprise, dtype=float)
    if rate.shape != asset.shape:
        raise ValueError(
            f"jk_poor_man: rate {rate.shape} and asset {asset.shape} must match"
        )
    same_sign = (rate * asset) > 0
    opp_sign = (rate * asset) < 0
    mp = np.where(opp_sign, rate, 0.0)
    info = np.where(same_sign, rate, 0.0)
    return JKResult(
        mp_shock=mp,
        info_shock=info,
        rotation=None,
        n_admissible=None,
        method="poor_man",
    )
```

- [ ] **Step 4: Re-export from `__init__.py`**

Update `puremacro/hfi/__init__.py`:

```python
from ._results import JKResult
from .jk2020 import jk_poor_man
from .surprises import aggregate_to_period, gk2015_surprise, ns2018_first_pc

__all__ = [
    "JKResult",
    "aggregate_to_period",
    "gk2015_surprise",
    "jk_poor_man",
    "ns2018_first_pc",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hfi/test_jk2020.py -v -k poor_man`
Expected: 3 passed.

- [ ] **Step 6: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`

---

## Task 10: Implement `jk_median_target`

**Files:**
- Modify: `puremacro/hfi/jk2020.py`
- Modify: `puremacro/hfi/__init__.py`
- Modify: `tests/test_hfi/test_jk2020.py`

The median-target JK variant searches the space of admissible orthogonal rotations of the (rate, asset) pair under sign restrictions:
- MP shock: positive impact on rate, negative on asset.
- Info shock: positive impact on rate, positive on asset.
It returns the *median* admissible rotation (Fry-Pagan 2011 median-target heuristic).

- [ ] **Step 1: Append the failing test**

Append to `tests/test_hfi/test_jk2020.py`:

```python
from puremacro.hfi.jk2020 import jk_median_target


def test_jk_median_target_returns_orthogonal_rotation():
    rng = np.random.default_rng(0)
    T = 200
    rate = rng.standard_normal(T)
    asset = rng.standard_normal(T)
    res = jk_median_target(rate, asset, n_rotations=2000, seed=0)
    assert res.rotation is not None
    np.testing.assert_allclose(res.rotation @ res.rotation.T, np.eye(2), atol=1e-8)
    assert res.n_admissible > 0


def test_jk_median_target_method_label():
    rng = np.random.default_rng(1)
    res = jk_median_target(rng.standard_normal(100), rng.standard_normal(100),
                           n_rotations=500, seed=0)
    assert res.method == "median_target"


def test_jk_median_target_perfect_negative_correlation_attributes_to_mp():
    """When (rate, asset) are perfectly negatively correlated, the data look
    purely monetary-policy: the info shock should have small variance relative
    to mp."""
    rng = np.random.default_rng(2)
    T = 300
    factor = rng.standard_normal(T)
    rate = factor                # rate up
    asset = -factor              # asset down
    res = jk_median_target(rate, asset, n_rotations=2000, seed=0)
    assert np.var(res.mp_shock) > 5 * np.var(res.info_shock)


def test_jk_median_target_perfect_positive_correlation_attributes_to_info():
    """Symmetric case: rate up + asset up → information shock dominates."""
    rng = np.random.default_rng(3)
    T = 300
    factor = rng.standard_normal(T)
    rate = factor
    asset = factor
    res = jk_median_target(rate, asset, n_rotations=2000, seed=0)
    assert np.var(res.info_shock) > 5 * np.var(res.mp_shock)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hfi/test_jk2020.py -v -k median_target`
Expected: ImportError.

- [ ] **Step 3: Implement `jk_median_target`**

Append to `puremacro/hfi/jk2020.py`:

```python
def _2d_rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def jk_median_target(
    rate_surprise: np.ndarray,
    asset_surprise: np.ndarray,
    n_rotations: int = 10_000,
    seed: int | None = None,
) -> JKResult:
    """Jarociński-Karadi (2020) median-target sign-restriction decomposition.

    Searches the 2x2 orthogonal rotation group for rotations U such that the
    decomposed shocks satisfy:

    - column 0 (MP shock) : impact on rate > 0, impact on asset < 0
    - column 1 (info shock): impact on rate > 0, impact on asset > 0

    The *median* admissible rotation (Fry-Pagan 2011) is selected, and the
    surprise vector is rotated to produce the two shock series.

    Parameters
    ----------
    rate_surprise : ndarray, shape (T,)
        High-frequency interest-rate surprise.
    asset_surprise : ndarray, shape (T,)
        High-frequency asset-price (e.g., S&P 500) surprise in the same window.
    n_rotations : int, default 10_000
        Number of rotations to draw uniformly on [0, 2π) when searching the
        admissible set.
    seed : int or None
        RNG seed.

    Returns
    -------
    JKResult with ``method="median_target"``, ``rotation`` filled, and
    ``n_admissible`` reporting how many of ``n_rotations`` satisfied the
    sign restrictions.

    Notes
    -----
    For a 2x2 problem the admissible set, if non-empty, is a contiguous arc
    in θ. The median rotation is the one at the median θ within that arc.
    """
    rate = np.asarray(rate_surprise, dtype=float)
    asset = np.asarray(asset_surprise, dtype=float)
    if rate.shape != asset.shape:
        raise ValueError(
            f"jk_median_target: rate {rate.shape} and asset {asset.shape} must match"
        )
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(0.0, 2 * np.pi, size=n_rotations)
    # The "impact" of the shocks on (rate, asset) under rotation U is just U
    # itself, since the surprise vector IS the shock pair (no further VAR step
    # at this stage). The sign restrictions on U:
    #   col 0 (MP)  : U[0,0] > 0,  U[1,0] < 0
    #   col 1 (info): U[0,1] > 0,  U[1,1] > 0
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    # U = [[cos, -sin], [sin, cos]]
    admissible = (cos_t > 0) & (sin_t < 0) & (-sin_t > 0) & (cos_t > 0)
    n_adm = int(admissible.sum())
    if n_adm == 0:
        raise ValueError(
            "jk_median_target: no admissible rotations found. Check sign "
            "conventions on rate_surprise and asset_surprise."
        )
    theta_med = float(np.median(thetas[admissible]))
    U = _2d_rotation(theta_med)
    # Rotate the (rate, asset) pair into shock space: shocks = (rate, asset) @ U
    shocks = np.column_stack([rate, asset]) @ U
    return JKResult(
        mp_shock=shocks[:, 0],
        info_shock=shocks[:, 1],
        rotation=U,
        n_admissible=n_adm,
        method="median_target",
    )
```

- [ ] **Step 4: Re-export from `__init__.py`**

Update `puremacro/hfi/__init__.py`:

```python
from ._results import JKResult
from .jk2020 import jk_median_target, jk_poor_man
from .surprises import aggregate_to_period, gk2015_surprise, ns2018_first_pc

__all__ = [
    "JKResult",
    "aggregate_to_period",
    "gk2015_surprise",
    "jk_median_target",
    "jk_poor_man",
    "ns2018_first_pc",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hfi/test_jk2020.py -v -k median_target`
Expected: 4 passed.

- [ ] **Step 6: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`

---

## Task 11: End-to-end test (HFI → `proxy_svar` → IRF + F)

**Files:**
- Create: `tests/test_hfi/test_end_to_end.py`

- [ ] **Step 1: Write the test**

```python
"""End-to-end: synthetic announcement-day surprises → monthly aggregation →
proxy_svar → ProxySVARResult with IRF tensor and Olea-Pflueger F."""
import numpy as np
import pandas as pd

from puremacro.hfi import aggregate_to_period, gk2015_surprise
from puremacro.var.identify._results import ProxySVARResult
from puremacro.var.identify.proxy import proxy_svar


def test_hfi_end_to_end():
    rng = np.random.default_rng(0)
    # Synthetic monthly macro panel: 240 months, 3 vars
    T_macro = 240
    Y = np.cumsum(0.3 * rng.standard_normal((T_macro, 3)), axis=0)

    # Synthetic announcement series: 1 per month, with monthly-aggregated
    # surprise correlated with the first VAR residual
    n_announce = T_macro
    rate_pre = 95.0 * np.ones(n_announce)
    rate_post = rate_pre + 0.05 * rng.standard_normal(n_announce)
    days_remaining = rng.integers(5, 28, size=n_announce)
    surprise = gk2015_surprise(rate_pre, rate_post, days_remaining,
                               days_in_month=30)
    dates = pd.date_range("2000-01-15", periods=n_announce, freq="MS")
    monthly = aggregate_to_period(surprise, dates, freq="M")

    # Align: drop the first VAR observation count to match VAR sample
    z = monthly.values

    res = proxy_svar(Y, p=2, horizon=12, instrument_series=z,
                     n_boot=50, ci=0.9, seed=0)
    assert isinstance(res, ProxySVARResult)
    assert res.irf_point.shape == (3, 3, 13)
    assert np.isfinite(res.first_stage_F)
    # CI bands respect the point estimate (mostly)
    assert (res.irf_lower <= res.irf_upper).all()
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_hfi/test_end_to_end.py -v`
Expected: PASS.

- [ ] **Step 3: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`

---

## Task 12: Example script `examples/hfi_gertler_karadi.py`

**Files:**
- Create: `puremacro/examples/hfi_gertler_karadi.py`

- [ ] **Step 1: Create the example**

```python
"""HFI monetary-policy shock: synthetic Gertler-Karadi 2015-style pipeline.

Demonstrates:
    1. Construct GK 2015 surprises from synthetic FFR-futures price changes.
    2. Aggregate announcement-day surprises to monthly bins.
    3. Run proxy-SVAR identification using the surprise as external instrument.
    4. Plot the IRF of macro variables to a unit MP shock with bootstrap bands.
    5. Print Olea-Pflueger effective F and CI level.

No real data is shipped — fully synthetic, runs offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from puremacro.hfi import aggregate_to_period, gk2015_surprise
from puremacro.var.identify.proxy import proxy_svar


def main():
    rng = np.random.default_rng(2026)
    # ---- 1. Synthetic monthly macro panel ----
    T = 240
    n = 3
    var_names = ["IP_growth", "CPI", "FFR"]
    Y = np.zeros((T, n))
    for t in range(2, T):
        Y[t] = 0.6 * Y[t - 1] - 0.1 * Y[t - 2] + 0.3 * rng.standard_normal(n)

    # ---- 2. Synthetic FOMC-day surprises ----
    n_announce = T
    rate_pre = 95.0 * np.ones(n_announce)
    rate_post = rate_pre + 0.05 * rng.standard_normal(n_announce)
    days_remaining = rng.integers(5, 28, size=n_announce)
    surprise = gk2015_surprise(rate_pre, rate_post, days_remaining,
                               days_in_month=30)
    dates = pd.date_range("2000-01-15", periods=n_announce, freq="MS")
    z = aggregate_to_period(surprise, dates, freq="M").values

    # ---- 3. Proxy-SVAR ----
    res = proxy_svar(Y, p=2, horizon=24, instrument_series=z,
                     n_boot=300, ci=0.9, seed=0)

    print(res.summary())

    # ---- 4. Plot ----
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    H = res.irf_point.shape[2]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharex=True)
    horizons = np.arange(H)
    for j in range(n):
        ax = axes[j]
        ax.plot(horizons, res.irf_point[j, 0, :], "b-", label="point")
        ax.fill_between(horizons, res.irf_lower[j, 0, :], res.irf_upper[j, 0, :],
                        color="b", alpha=0.2, label=f"{int(res.ci * 100)}% CI")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{var_names[j]} ↑ MP shock")
        ax.set_xlabel("horizon (months)")
    axes[0].legend(loc="best")
    fig.suptitle(f"HFI proxy-SVAR (synthetic, OP F={res.first_stage_F:.1f})")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

Run: `python -m puremacro.examples.hfi_gertler_karadi 2>&1 | head -30`
Expected: prints `ProxySVAR result` block; no exception. (A matplotlib window may open; close it.)

- [ ] **Step 3: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`

---

## Task 13: Add `puremacro.hfi` to Pyodide-compat walk and add module README

**Files:**
- Modify: `tests/test_pyodide_compat.py`
- Create: `puremacro/hfi/README.md`

- [ ] **Step 1: Inspect the Pyodide-compat walk**

Run: `grep -n "submodules\|walk\|importlib\|import_module" tests/test_pyodide_compat.py | head`

The test walks all subpackages. Verify `hfi` is now picked up automatically (since the walk uses `pkgutil.iter_modules` or `pathlib`-based discovery). If the test has a hardcoded list, add `"puremacro.hfi"` (and `puremacro.hfi.surprises`, `puremacro.hfi.jk2020`, `puremacro.hfi._results`) to it.

- [ ] **Step 2: Run the Pyodide-compat test**

Run: `pytest tests/test_pyodide_compat.py -v`
Expected: passes; the new HFI modules are exercised in the walk and no statsmodels/linearmodels/arch leak in.

- [ ] **Step 3: Create `puremacro/hfi/README.md`**

```markdown
# puremacro.hfi — High-frequency identification of monetary policy

Surprise construction and Jarociński-Karadi (2020) decomposition.

For external-IV SVAR, pipe a surprise series into
`puremacro.var.identify.proxy.proxy_svar`. HFI does not duplicate that
machinery; it provides only the surprise- and shock-construction layer.

## Quick start

```python
import numpy as np
import pandas as pd
from puremacro.hfi import gk2015_surprise, aggregate_to_period, jk_median_target
from puremacro.var.identify.proxy import proxy_svar

# 1) Surprises from FFR-futures around announcements
surprise = gk2015_surprise(pre_prices, post_prices,
                           days_remaining_in_month, days_in_month=30)

# 2) Optionally combine multiple contracts via Nakamura-Steinsson 2018 first PC
# surprise, loadings = ns2018_first_pc(contract_changes, scale_to_idx=0)

# 3) Aggregate to monthly for VAR
z = aggregate_to_period(surprise, announce_dates, freq="M")

# 4) Proxy-SVAR with Olea-Pflueger first-stage F and bootstrap bands
res = proxy_svar(Y_macro, p=2, horizon=24,
                 instrument_series=z.values, n_boot=500, seed=0)
print(res.summary())

# 5) Optionally decompose into MP vs information shocks (JK 2020)
mp_info = jk_median_target(rate_surprise=z.values,
                           asset_surprise=spx_window_change,
                           n_rotations=10_000, seed=0)
```

## Public API

- `gk2015_surprise(pre, post, days_remaining, days_in_month=30)` — Gertler-Karadi 2015 month-end-adjusted FFR-futures change.
- `ns2018_first_pc(surprise_matrix, scale_to_idx=0)` — Nakamura-Steinsson 2018 first PC of K policy contracts.
- `aggregate_to_period(surprises, dates, freq="M")` — sum to monthly/quarterly bins, zero-fill empty periods.
- `jk_poor_man(rate_surprise, asset_surprise)` — JK 2020 sign-of-comovement decomposition.
- `jk_median_target(rate_surprise, asset_surprise, n_rotations=10_000, seed=None)` — JK 2020 median admissible-rotation decomposition.

## What's NOT here

- The full Bayesian sign-restriction variant of JK 2020 — deferred to 0.5.0+.
- Real surprise series — none are shipped; bring your own (Gertler-Karadi public dataset, Nakamura-Steinsson replication files, etc.).
- External-IV SVAR machinery — composes on top of `puremacro.var.identify.proxy.proxy_svar`.

## References

- Gertler, M. and Karadi, P. (2015). Monetary policy surprises, credit costs, and economic activity. AEJ:Macro 7(1), 44-76.
- Nakamura, E. and Steinsson, J. (2018). High-frequency identification of monetary non-neutrality. QJE 133(3), 1283-1330.
- Jarociński, M. and Karadi, P. (2020). Deconstructing monetary policy surprises — the role of information shocks. AEJ:Macro 12(2), 1-43.
- Olea, J.L.M. and Pflueger, C. (2013). A robust test for weak instruments. JBES 31(3), 358-369.
```

- [ ] **Step 4: Verification checkpoint**

Run: `pytest tests/ -x --tb=short`
Expected: full suite green.

---

## Task 14: Final verification, CHANGELOG entry stub, version-bump scaffolding

**Files:**
- Modify: `CHANGELOG.md` (stub for 0.4.0 — entries get filled across plan steps)
- (No version bump yet — that lands in the final 0.4.0 release plan.)

- [ ] **Step 1: Add stub 0.4.0 entry to `CHANGELOG.md`**

Insert at the top of `CHANGELOG.md`, above the existing `## 0.3.0` entry:

```markdown
## 0.4.0 (in progress)

### Added
- **`puremacro.hfi`** — High-frequency identification of monetary policy shocks.
  - `gk2015_surprise` — Gertler-Karadi 2015 month-end-adjusted FFR-futures change.
  - `ns2018_first_pc` — Nakamura-Steinsson 2018 first PC of K policy-sensitive contracts.
  - `aggregate_to_period` — sum announcement-day surprises into monthly/quarterly bins.
  - `jk_poor_man`, `jk_median_target` — Jarociński-Karadi 2020 monetary-vs-information decomposition.
  - New `JKResult` dataclass.
- **`olea_pflueger_f`** added to `puremacro.inference.weak_iv` — Olea-Pflueger 2013 effective F-statistic for weak-IV diagnostics.

### Changed
- **`puremacro.var.identify.proxy.proxy_svar`** now returns `ProxySVARResult` (frozen dataclass with `irf_point`, `irf_lower`, `irf_upper`, `B`, `first_stage_F`, `n_boot`, `ci`) instead of the legacy 3-tuple `(point, lo, hi)`. Old callers must access fields by name.
- The first-stage F reported is now Olea-Pflueger 2013 effective F, not the prior ad-hoc Wald-style heuristic.

### Standards
- New result-object standard documented in `ARCHITECTURE.md`: `@dataclass(frozen=True)`, `<MethodName>Result` naming, lives in `<subpackage>/_results.py`. Subsequent steps in iteration N+8 propagate this across the package.

### Deferred to 0.5.0+
- JK 2020 full Bayesian sign-restriction variant.
```

- [ ] **Step 2: Final test pass**

Run: `pytest tests/ -v --tb=short 2>&1 | tail -30`

Expected:
- Strictly more passing tests than the 237 baseline (target: 237 + ~20-25 new HFI/proxy/weak_iv tests).
- 0 failures.
- 5 skipped (network — unchanged).

- [ ] **Step 3: Pyodide compat re-verify**

Run: `pytest tests/test_pyodide_compat.py -v`
Expected: PASS. Confirms HFI added zero new disallowed imports.

- [ ] **Step 4: Final verification checkpoint**

Run: `python -c "from puremacro.hfi import (gk2015_surprise, ns2018_first_pc, aggregate_to_period, jk_poor_man, jk_median_target, JKResult); print('ok')"`
Expected: prints `ok`.

Run: `python -c "from puremacro.var.identify.proxy import proxy_svar; from puremacro.var.identify._results import ProxySVARResult; from puremacro.inference.weak_iv import olea_pflueger_f; print('ok')"`
Expected: prints `ok`.

---

## Plan complete

This plan covers items 0 (result-object standard documentation) and 1 (`puremacro.hfi`) of iteration N+8. Steps 2-6 of the iteration get their own plans:

- Step 2: `puremacro.cycles.hamilton_filter` — to be drafted next.
- Step 3: result-object migration sweep + `tests/test_public_api.py` + docstring/type-hint pass on 0.3.0 modules.
- Step 4: `puremacro.dynpanel`.
- Step 5: narrative connector HTTP fixtures.
- Step 6: DiD completers (CdH 2020 + multi-cohort SDID).

Spec reference: `docs/specs/2026-05-02-iteration-n8-design.md`.

## Self-review notes

- **Spec coverage:** sections A (result-object standard) and B (HFI) of the spec are covered in full. The `ProxySVARResult` migration was lifted from section A's migration sweep into this plan because HFI depends on it; remove it from step 3's sweep when that plan is drafted.
- **Type consistency:** `ProxySVARResult` field names (`irf_point`, `irf_lower`, `irf_upper`, `B`, `first_stage_F`, `n_boot`, `ci`) are used consistently across tasks 3, 4, 11, 12, 13. `JKResult` field names (`mp_shock`, `info_shock`, `rotation`, `n_admissible`, `method`) are consistent across tasks 5, 9, 10.
- **Dependencies:** `pandas` is already a runtime dependency; nothing new added.
- **No git:** verification checkpoints replace commits throughout. The plan is interrupt-safe at any checkpoint.
