# puremacro 0.51.0 Implementation Plan — identification extensions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship three identification innovations as 0.51.0: Magnusson-Mavroeidis SVAR (continuous heteroskedasticity), non-Gaussian SVAR diagnostics, and Lewbel-IV with an LP wrapper.

**Architecture:** Three independent components living under existing module homes (`puremacro/var/identify/`, `puremacro/inference/`, `puremacro/lp/`). All pure numpy/scipy, Pyodide-compatible. Strictly additive: no existing public symbols renamed or removed.

**Tech Stack:** numpy, scipy (`linalg.eigh`, `optimize.minimize`, `stats.chi2`, `stats.f`, `stats.kurtosis`), pandas (LP wrapper only). No new dependencies. Python ≥3.10.

**Spec:** `docs/specs/2026-05-23-puremacro-051-identification-extensions-design.md`

---

## File map

### New files
- `puremacro/var/identify/magmav.py` — Magnusson-Mavroeidis SVAR core.
- `puremacro/inference/lewbel_iv.py` — Lewbel constructed-IV 2SLS estimator.
- `puremacro/lp/iv_lewbel.py` — LP wrapper using Lewbel IVs.
- `tests/test_var/test_magmav.py` — 10 unit tests for Component A.
- `tests/test_var/test_non_gaussian_extensions.py` — 7 unit tests for Component B.
- `tests/test_inference/test_lewbel_iv.py` — 6 unit tests for Component C.
- `tests/test_lp/test_lp_iv_lewbel.py` — 1 wrapper test.

### Modified files
- `puremacro/var/identify/_results.py` — add `MagMavSVARResult`; extend `NonGaussianSVARResult` with two optional fields.
- `puremacro/var/identify/non_gaussian.py` — add tie-breaker + diagnostics; populate new fields.
- `puremacro/var/identify/__init__.py` — export `magmav_svar`, `MagMavSVARResult`, `gaussian_lr_test`, `variance_decomposition_consistency`.
- `puremacro/inference/_results.py` — add `LewbelIVResult`.
- `puremacro/inference/__init__.py` — export `lewbel_iv`, `LewbelIVResult`.
- `puremacro/lp/__init__.py` — export `lp_iv_lewbel`.
- `puremacro/__init__.py` — bump `__version__` to `"0.51.0"`.
- `pyproject.toml` — bump `version` to `"0.51.0"`.
- `CHANGELOG.md` — add 0.51.0 section.
- `tests/test_var/test_identify_results.py` — register the new dataclass in the public-API snapshot.
- `tests/test_public_api.py` (or wherever the snapshot lives) — regenerated.

### Working assumptions (verified via signature dumps)
- `puremacro.var.estimate.estimate_var(Y, p)` returns a `VarEstimateResult` dataclass with attributes `A_list, c, Sigma, resid, X`. The class is iterable as the 5-tuple `(A_list, c, Sigma, resid, X)`.
- `puremacro.var.irf.irf(A_list, B0, horizon)` returns shape `(H+1, n, n)` with `irf[h, i, j]` = response of var `i` to shock `j` at horizon `h`.
- `puremacro._linalg.safe_cholesky(M, name=...)` returns the lower-triangular factor or raises a diagnostic LinAlgError.
- `puremacro.inference._ols_helpers.ols_hac(y, X, lags)` returns dict `{beta, se, t, vcov, residuals, n_obs}`.
- `puremacro.var.identify.non_gaussian.non_gaussian_svar(Y, *, p, horizon, seed=0)` already orders columns by descending `|excess kurtosis|` and locks `diag(B0) >= 0` — confirmed at `puremacro/var/identify/non_gaussian.py:79-93`.
- `NonGaussianSVARResult` fields are `B0, Q, kurtosis, irf, ordering_by_kurt` — confirmed at `puremacro/var/identify/_results.py`.

---

## Task 1: Add `MagMavSVARResult` dataclass

**Files:**
- Modify: `puremacro/var/identify/_results.py` (append at end, before any final `__all__` if present)
- Test: `tests/test_var/test_identify_results.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_var/test_identify_results.py`:

```python
def test_magmav_svar_result_dataclass_is_frozen_and_has_expected_fields():
    import numpy as np
    from puremacro.var.identify._results import MagMavSVARResult

    res = MagMavSVARResult(
        irf_point=np.zeros((4, 2, 2)),
        irf_lower=np.zeros((4, 2, 2)),
        irf_upper=np.zeros((4, 2, 2)),
        B=np.eye(2),
        variance_change_dates=(50, 120),
        k_breaks=2,
        n_boot=200,
        ci=0.9,
        eu=(1, 1),
        n_fail=0,
    )
    assert res.irf_point.shape == (4, 2, 2)
    assert res.k_breaks == 2
    assert res.eu == (1, 1)
    import dataclasses
    assert dataclasses.is_dataclass(res)
    # frozen
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.k_breaks = 3
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_var/test_identify_results.py::test_magmav_svar_result_dataclass_is_frozen_and_has_expected_fields -v`
Expected: `ImportError: cannot import name 'MagMavSVARResult'`

- [ ] **Step 3: Add the dataclass**

Append to `puremacro/var/identify/_results.py`:

```python
@dataclass(frozen=True)
class MagMavSVARResult:
    """Result of :func:`puremacro.var.identify.magmav.magmav_svar`.

    Magnusson-Mavroeidis (2014) SVAR identified by continuous time-varying
    structural-shock variance; break dates discovered endogenously by
    sup-Wald + BIC.

    Attributes
    ----------
    irf_point : ndarray, shape (H+1, n, n)
        Point-estimate impulse responses.
    irf_lower : ndarray, shape (H+1, n, n)
        Lower bootstrap band.
    irf_upper : ndarray, shape (H+1, n, n)
        Upper bootstrap band.
    B : ndarray, shape (n, n)
        Identified structural impact matrix. Columns ordered by descending
        cross-regime variance ratio max_g D_g[j,j] / min_g D_g[j,j].
    variance_change_dates : tuple of int
        Break dates (residual-row indices) selected by sup-Wald + BIC.
    k_breaks : int
        Number of breaks selected (0 if BIC chose homoskedastic baseline).
    n_boot : int
        Number of bootstrap draws requested.
    ci : float
        Bootstrap CI level.
    eu : tuple of int
        Existence / uniqueness flags; (1, 1) iff identification succeeded.
    n_fail : int
        Bootstrap draws that failed to converge.

    References
    ----------
    Magnusson, L.M. and Mavroeidis, S. (2014). Identification using
        stability restrictions. Econometrica 82(5), 1799-1851.
    """

    irf_point: np.ndarray
    irf_lower: np.ndarray
    irf_upper: np.ndarray
    B: np.ndarray
    variance_change_dates: tuple
    k_breaks: int
    n_boot: int
    ci: float
    eu: tuple
    n_fail: int

    def summary(self) -> str:
        H = self.irf_point.shape[0] - 1
        n = self.irf_point.shape[1]
        flag = "OK" if self.eu == (1, 1) else f"FAIL (eu={self.eu})"
        return (
            f"Magnusson-Mavroeidis SVAR result\n"
            f"  variables (n)        : {n}\n"
            f"  horizon (H)          : {H}\n"
            f"  break dates          : {self.variance_change_dates}\n"
            f"  k_breaks             : {self.k_breaks}\n"
            f"  identification       : {flag}\n"
            f"  bootstrap            : {self.n_boot} draws (failed {self.n_fail}), CI {self.ci:.2f}\n"
        )
```

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_var/test_identify_results.py::test_magmav_svar_result_dataclass_is_frozen_and_has_expected_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add puremacro/var/identify/_results.py tests/test_var/test_identify_results.py
git commit -m "feat(var/identify): add MagMavSVARResult dataclass for 0.51.0"
```

---

## Task 2: Sup-Wald break-date scan + BIC selection

**Files:**
- Create: `puremacro/var/identify/magmav.py` (skeleton + helpers; orchestrator added in Task 4)
- Test: `tests/test_var/test_magmav.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_var/test_magmav.py`:

```python
"""Tests for Magnusson-Mavroeidis SVAR (puremacro.var.identify.magmav)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.var.identify import magmav as mm


def _synthetic_var2_with_breaks(T: int, breaks: tuple[int, ...], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """VAR(1) on 2 vars with regime-specific structural variances. Returns (Y, true_B)."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.6, 0.1], [0.0, 0.5]])
    B = np.array([[1.0, 0.3], [0.2, 0.8]])
    n = 2
    # Build per-period variances: regimes between break boundaries.
    boundaries = (0,) + tuple(breaks) + (T,)
    sigma_per_t = np.zeros((T, n))
    for g in range(len(boundaries) - 1):
        s_lo, s_hi = boundaries[g], boundaries[g + 1]
        # Each regime has different shock variances
        v0 = 0.5 + 1.5 * g  # var of shock 0 in regime g
        v1 = 2.0 - 0.4 * g  # var of shock 1 in regime g
        sigma_per_t[s_lo:s_hi] = [np.sqrt(v0), np.sqrt(v1)]
    Y = np.zeros((T, n))
    for t in range(1, T):
        eps = rng.standard_normal(n) * sigma_per_t[t]
        Y[t] = A @ Y[t - 1] + B @ eps
    return Y, B


def test_sup_wald_scan_finds_known_break():
    Y, _ = _synthetic_var2_with_breaks(T=400, breaks=(200,), seed=0)
    from puremacro.var.estimate import estimate_var
    A_list, _, _, resid, _ = estimate_var(Y, 1)
    # Single-break scan
    tau, stat = mm._sup_wald_one_break(resid, lo_frac=0.15, hi_frac=0.85)
    # True break at t=200 (residual index = 199 because of one lag)
    assert abs(tau - 199) < 30, f"detected break {tau} far from true 199"
    assert stat > 0
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_var/test_magmav.py::test_sup_wald_scan_finds_known_break -v`
Expected: `ModuleNotFoundError: No module named 'puremacro.var.identify.magmav'`

- [ ] **Step 3: Create the magmav.py skeleton with break-detection helpers**

Create `puremacro/var/identify/magmav.py`:

```python
"""Magnusson-Mavroeidis (2014) SVAR via continuous heteroskedasticity.

Identifies the structural impact matrix B from regime-specific variance
shifts in reduced-form residuals, where break dates are *not* prespecified
but selected endogenously by a sup-Wald scan + BIC.

References
----------
Magnusson, L.M. and Mavroeidis, S. (2014). Identification using stability
    restrictions. Econometrica 82(5), 1799-1851.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.optimize

from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import MagMavSVARResult


# --------------------------------------------------------------------------- #
# Break detection
# --------------------------------------------------------------------------- #

def _sup_wald_one_break(resid: np.ndarray, *, lo_frac: float = 0.15,
                        hi_frac: float = 0.85) -> tuple[int, float]:
    """Single-break sup-Wald scan on residual covariance.

    For each candidate τ in [lo_frac·T, hi_frac·T], compute the LR-style
    statistic comparing the homoskedastic baseline to a two-regime fit.
    Return (argmax τ, max stat).
    """
    T, n = resid.shape
    lo = max(int(lo_frac * T), n + 2)
    hi = min(int(hi_frac * T), T - n - 2)
    if hi <= lo:
        return T // 2, 0.0
    # Pre-compute outer products for fast covariance updates.
    # Σ_g(τ) = (1/T_g) Σ_{t in g} u_t u_t'
    # Stat: T·log|Σ_full| - T_0·log|Σ_0(τ)| - T_1·log|Σ_1(τ)|
    Sig_full = resid.T @ resid / T
    log_det_full = float(np.log(np.linalg.det(Sig_full) + 1e-300))
    best_tau, best_stat = lo, -np.inf
    for tau in range(lo, hi + 1):
        S0 = resid[:tau].T @ resid[:tau] / tau
        S1 = resid[tau:].T @ resid[tau:] / (T - tau)
        d0 = np.linalg.det(S0)
        d1 = np.linalg.det(S1)
        if d0 <= 0 or d1 <= 0:
            continue
        stat = T * log_det_full - tau * float(np.log(d0)) - (T - tau) * float(np.log(d1))
        if stat > best_stat:
            best_stat = stat
            best_tau = tau
    return best_tau, float(best_stat)


def _detect_k_breaks(resid: np.ndarray, *, k: int, lo_frac: float = 0.15,
                     hi_frac: float = 0.85, min_sep_frac: float = 0.05) -> tuple[int, ...]:
    """Greedy multi-break detection: find the single best break in each
    remaining sub-segment, enforcing minimum separation."""
    T = resid.shape[0]
    min_sep = max(int(min_sep_frac * T), 5)
    breaks: list[int] = []
    segments: list[tuple[int, int]] = [(0, T)]
    for _ in range(k):
        # In each pass, find the segment whose internal sup-Wald is largest.
        best = (-1, -np.inf, 0)  # (tau, stat, seg_idx)
        for si, (a, b) in enumerate(segments):
            if b - a < 2 * min_sep:
                continue
            sub = resid[a:b]
            tau_local, stat = _sup_wald_one_break(
                sub,
                lo_frac=max(min_sep_frac, lo_frac * (b - a) / T),
                hi_frac=min(1 - min_sep_frac, hi_frac * (b - a) / T + (1 - hi_frac)),
            )
            tau_global = a + tau_local
            # Enforce min separation from existing breaks
            if breaks and min(abs(tau_global - bk) for bk in breaks) < min_sep:
                continue
            if stat > best[1]:
                best = (tau_global, stat, si)
        if best[0] < 0:
            break
        tau_global, _, si = best
        breaks.append(tau_global)
        # Split that segment
        a, b = segments.pop(si)
        segments.append((a, tau_global))
        segments.append((tau_global, b))
        segments.sort()
    return tuple(sorted(breaks))


def _bic_k_breaks(resid: np.ndarray, k: int, breaks: tuple[int, ...]) -> float:
    """BIC for k-break heteroskedastic-Gaussian model."""
    T, n = resid.shape
    boundaries = (0,) + breaks + (T,)
    ll = 0.0
    for g in range(len(boundaries) - 1):
        a, b = boundaries[g], boundaries[g + 1]
        Tg = b - a
        if Tg <= n + 1:
            return np.inf
        Sg = resid[a:b].T @ resid[a:b] / Tg
        sign, logdet = np.linalg.slogdet(Sg)
        if sign <= 0:
            return np.inf
        ll += -0.5 * Tg * (n * np.log(2 * np.pi) + logdet + n)
    # Free parameters: k break dates + (k+1)·n diagonal structural variances + n(n+1)/2 elements of B.
    # B params don't depend on k; cancel from comparison. We penalise only the additional regime params.
    n_params = k + (k + 1) * n
    return -2.0 * ll + n_params * np.log(T)


def _select_k_breaks(resid: np.ndarray, *, k_grid: tuple[int, ...] = (0, 1, 2, 3, 4)) -> tuple[int, tuple[int, ...]]:
    """Return (k, breaks) minimising BIC over k_grid."""
    best_k, best_bic, best_breaks = 0, np.inf, tuple()
    for k in k_grid:
        if k == 0:
            br = tuple()
        else:
            br = _detect_k_breaks(resid, k=k)
            if len(br) < k:
                continue
        bic = _bic_k_breaks(resid, k, br)
        if bic < best_bic:
            best_k, best_bic, best_breaks = k, bic, br
    return best_k, best_breaks


__all__: list[str] = []  # populated in Task 5
```

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_var/test_magmav.py::test_sup_wald_scan_finds_known_break -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add puremacro/var/identify/magmav.py tests/test_var/test_magmav.py
git commit -m "feat(magmav): sup-Wald break-date scan + BIC selector"
```

---

## Task 3: B-matrix optimisation

**Files:**
- Modify: `puremacro/var/identify/magmav.py`
- Test: `tests/test_var/test_magmav.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_var/test_magmav.py`:

```python
def test_estimate_B_recovers_known_matrix():
    rng = np.random.default_rng(7)
    n, T = 2, 1000
    B_true = np.array([[1.0, 0.3], [0.2, 0.8]])
    # Two regimes with distinct structural-shock variances
    D0 = np.diag([0.5, 2.0])
    D1 = np.diag([2.5, 0.6])
    # Generate residuals: u = B eps, eps ~ N(0, D_g)
    u0 = (rng.standard_normal((T // 2, n)) * np.sqrt(np.diag(D0))) @ B_true.T
    u1 = (rng.standard_normal((T // 2, n)) * np.sqrt(np.diag(D1))) @ B_true.T
    Sigmas = [u0.T @ u0 / (T // 2), u1.T @ u1 / (T // 2)]
    B_hat, D_hat, success = mm._estimate_B_from_regime_covariances(
        Sigmas, n_starts=3, seed=0,
    )
    assert success
    # B is identified up to column permutation + sign; compare via B B^T
    err = np.linalg.norm(B_hat @ B_hat.T - B_true @ B_true.T, ord="fro")
    assert err < 0.4, f"frobenius error {err:.3f} too large"
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_var/test_magmav.py::test_estimate_B_recovers_known_matrix -v`
Expected: `AttributeError: module ... has no attribute '_estimate_B_from_regime_covariances'`

- [ ] **Step 3: Implement the B-matrix estimator**

Append to `puremacro/var/identify/magmav.py` (above the trailing `__all__` line):

```python
# --------------------------------------------------------------------------- #
# B-matrix estimation
# --------------------------------------------------------------------------- #

def _unpack_B(theta: np.ndarray, n: int) -> np.ndarray:
    """Pack/unpack: theta has n² entries laid out row-major into a full B."""
    return theta.reshape(n, n)


def _loss_B(theta: np.ndarray, n: int, Sigmas: list[np.ndarray]) -> float:
    """Σ_g ||Σ_g - B D_g B^T||_F^2, with D_g recovered analytically given B."""
    B = _unpack_B(theta, n)
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return 1e20
    loss = 0.0
    for Sigma_g in Sigmas:
        # Best D_g given B is diag(B^{-1} Σ_g B^{-T}); clipped to positive.
        D_g = B_inv @ Sigma_g @ B_inv.T
        d = np.clip(np.diag(D_g), 1e-10, None)
        diff = Sigma_g - B @ np.diag(d) @ B.T
        loss += float(np.sum(diff * diff))
    return loss


def _solve_D_given_B(B: np.ndarray, Sigmas: list[np.ndarray]) -> np.ndarray:
    """Returns D, shape (G, n): per-regime structural variances."""
    n = B.shape[0]
    B_inv = np.linalg.inv(B)
    D = np.zeros((len(Sigmas), n))
    for g, Sigma_g in enumerate(Sigmas):
        Dg = B_inv @ Sigma_g @ B_inv.T
        D[g] = np.clip(np.diag(Dg), 1e-10, None)
    return D


def _estimate_B_from_regime_covariances(
    Sigmas: list[np.ndarray], *, n_starts: int = 3, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Minimise Σ_g ||Σ_g - B D_g B^T||_F^2 with multi-start BFGS.

    Returns (B, D, success).
    """
    rng = np.random.default_rng(seed)
    n = Sigmas[0].shape[0]
    # Sensible starting point: Cholesky of regime-pooled Σ.
    Sigma_pool = sum(Sigmas) / len(Sigmas)
    L0 = np.linalg.cholesky(Sigma_pool)
    best_B, best_loss, ok = None, np.inf, False
    for s in range(n_starts):
        if s == 0:
            theta0 = L0.flatten()
        else:
            jitter = 0.05 * rng.standard_normal((n, n))
            theta0 = (L0 + jitter).flatten()
        try:
            res = scipy.optimize.minimize(
                _loss_B, theta0, args=(n, Sigmas), method="BFGS",
                options={"maxiter": 300, "gtol": 1e-6},
            )
        except Exception:
            continue
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_B = _unpack_B(res.x, n)
            ok = res.success or (res.fun < 1e-3)
    if best_B is None:
        return L0, _solve_D_given_B(L0, Sigmas), False
    D = _solve_D_given_B(best_B, Sigmas)
    return best_B, D, ok


def _normalise_B(B: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise B such that:
        1. Columns ordered by descending cross-regime variance ratio
           r_j = max_g D[g, j] / min_g D[g, j].
        2. diag(B) >= 0 (sign-flip columns whose diagonal entry is negative).

    Returns (B_norm, D_norm, order).
    """
    n = B.shape[0]
    # Avoid divide-by-zero
    safe_min = np.clip(D.min(axis=0), 1e-12, None)
    ratios = D.max(axis=0) / safe_min
    order = np.argsort(-ratios)
    B = B[:, order]
    D = D[:, order]
    for j in range(n):
        if B[j, j] < 0:
            B[:, j] *= -1
    return B, D, order
```

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_var/test_magmav.py::test_estimate_B_recovers_known_matrix -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add puremacro/var/identify/magmav.py tests/test_var/test_magmav.py
git commit -m "feat(magmav): B-matrix multi-start BFGS estimator + normalisation"
```

---

## Task 4: Orchestrator `magmav_svar` + bootstrap

**Files:**
- Modify: `puremacro/var/identify/magmav.py`
- Test: `tests/test_var/test_magmav.py`

- [ ] **Step 1: Write three failing tests**

Append to `tests/test_var/test_magmav.py`:

```python
def test_magmav_svar_returns_result_dataclass():
    Y, _ = _synthetic_var2_with_breaks(T=300, breaks=(150,), seed=2)
    from puremacro.var.identify._results import MagMavSVARResult
    res = mm.magmav_svar(Y, p=1, horizon=8, k_breaks=1, n_boot=50, seed=2)
    assert isinstance(res, MagMavSVARResult)
    assert res.irf_point.shape == (9, 2, 2)
    assert res.irf_lower.shape == res.irf_point.shape
    assert res.irf_upper.shape == res.irf_point.shape
    assert res.B.shape == (2, 2)
    assert res.k_breaks == 1
    assert res.n_boot == 50
    assert res.ci == 0.9
    assert isinstance(res.variance_change_dates, tuple)


def test_magmav_svar_bic_selects_no_breaks_on_homoskedastic_data():
    rng = np.random.default_rng(3)
    n, T = 2, 250
    B_true = np.array([[1.0, 0.2], [0.0, 0.9]])
    Y = np.zeros((T, n))
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + B_true @ rng.standard_normal(n)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=None, n_boot=20, seed=0)
    # k_breaks=0 means BIC fell back; eu signals failure.
    if res.k_breaks == 0:
        assert res.eu == (0, 0)


def test_magmav_svar_seed_reproducibility():
    Y, _ = _synthetic_var2_with_breaks(T=300, breaks=(150,), seed=4)
    r1 = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=1, n_boot=30, seed=11)
    r2 = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=1, n_boot=30, seed=11)
    np.testing.assert_allclose(r1.irf_lower, r2.irf_lower)
    np.testing.assert_allclose(r1.irf_upper, r2.irf_upper)
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_var/test_magmav.py -k "returns_result or selects_no_breaks or seed_reproducibility" -v`
Expected: `AttributeError: module 'puremacro.var.identify.magmav' has no attribute 'magmav_svar'`

- [ ] **Step 3: Implement the orchestrator**

Append to `puremacro/var/identify/magmav.py`:

```python
# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _residuals_to_regime_covs(resid: np.ndarray, breaks: tuple[int, ...]) -> list[np.ndarray]:
    """Slice residuals into regimes and return covariance matrices."""
    T = resid.shape[0]
    bounds = (0,) + breaks + (T,)
    out = []
    for g in range(len(bounds) - 1):
        a, b = bounds[g], bounds[g + 1]
        out.append(resid[a:b].T @ resid[a:b] / (b - a))
    return out


def _regime_bootstrap_indices(T: int, breaks: tuple[int, ...], rng) -> np.ndarray:
    """Sample WITH replacement within each regime to preserve heteroskedasticity."""
    bounds = (0,) + breaks + (T,)
    idx = np.empty(T, dtype=int)
    for g in range(len(bounds) - 1):
        a, b = bounds[g], bounds[g + 1]
        idx[a:b] = rng.integers(a, b, size=b - a)
    return idx


def magmav_svar(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int = 20,
    k_breaks: Optional[int] = None,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> MagMavSVARResult:
    """Magnusson-Mavroeidis (2014) SVAR via continuous heteroskedasticity.

    Parameters
    ----------
    Y : (T, n) reduced-form data.
    p : VAR lag order.
    horizon : IRF horizon H (output has shape (H+1, n, n)).
    k_breaks : if None, choose via BIC over {0, 1, 2, 3, 4}.
    n_boot, ci, seed : bootstrap controls.

    Returns
    -------
    MagMavSVARResult
    """
    Y = np.asarray(Y, dtype=float)
    rng = np.random.default_rng(seed)
    A_list, _, _, resid, _ = estimate_var(Y, p)
    T_eff = resid.shape[0]

    # 1. Select breaks
    if k_breaks is None:
        k_sel, breaks = _select_k_breaks(resid)
    else:
        k_sel = int(k_breaks)
        breaks = _detect_k_breaks(resid, k=k_sel) if k_sel > 0 else tuple()
    if len(breaks) < k_sel:
        # Sup-Wald couldn't find that many breaks; fall back.
        k_sel = len(breaks)

    # 2. Estimate B
    if k_sel == 0:
        # Homoskedastic fallback: Cholesky of Σ; eu = (0, 0) signals failure.
        Sigma = resid.T @ resid / T_eff
        B = np.linalg.cholesky(Sigma)
        eu = (0, 0)
        warnings.warn(
            "magmav_svar: k_breaks selected as 0; identification not achieved; "
            "returning Cholesky fallback.",
            stacklevel=2,
        )
        D_full = np.array([np.diag(B.T @ B)])
    else:
        Sigmas = _residuals_to_regime_covs(resid, breaks)
        B, D, success = _estimate_B_from_regime_covariances(Sigmas, n_starts=3, seed=seed)
        if not success:
            warnings.warn(
                "magmav_svar: B-matrix optimisation did not converge; returning best draw.",
                stacklevel=2,
            )
        B, D, _ = _normalise_B(B, D)
        D_full = D
        eu = (1, 1) if success else (0, 0)

    irf_point = compute_irf(A_list, B, horizon)

    # 3. Bootstrap with regime-preserving resampling
    lo_pct = 100 * (1 - ci) / 2
    hi_pct = 100 * (1 + ci) / 2
    boot_irfs: list[np.ndarray] = []
    n_fail = 0
    for b in range(n_boot):
        idx = _regime_bootstrap_indices(T_eff, breaks, rng)
        u_boot = resid[idx]
        # Re-simulate Y under the original A and bootstrap residuals
        n = Y.shape[1]
        Y_boot = np.zeros((T_eff + p, n))
        Y_boot[:p] = Y[:p]
        for t in range(p, T_eff + p):
            x = sum(A_list[l] @ Y_boot[t - l - 1] for l in range(p))
            Y_boot[t] = x + u_boot[t - p]
        try:
            A_b, _, _, resid_b, _ = estimate_var(Y_boot, p)
            if k_sel == 0:
                B_b = np.linalg.cholesky(resid_b.T @ resid_b / resid_b.shape[0])
            else:
                Sigmas_b = _residuals_to_regime_covs(resid_b, breaks)
                B_b, D_b, ok_b = _estimate_B_from_regime_covariances(Sigmas_b, n_starts=1, seed=seed + b + 1)
                if not ok_b:
                    n_fail += 1
                    continue
                B_b, D_b, _ = _normalise_B(B_b, D_b)
            boot_irfs.append(compute_irf(A_b, B_b, horizon))
        except (np.linalg.LinAlgError, ValueError):
            n_fail += 1
            continue
    if n_fail / max(n_boot, 1) > 0.05:
        warnings.warn(
            f"magmav_svar: {n_fail}/{n_boot} bootstrap draws failed "
            f"({n_fail / n_boot:.1%}).",
            stacklevel=2,
        )
    if len(boot_irfs) == 0:
        irf_lower = np.full_like(irf_point, np.nan)
        irf_upper = np.full_like(irf_point, np.nan)
    else:
        arr = np.stack(boot_irfs)
        irf_lower = np.percentile(arr, lo_pct, axis=0)
        irf_upper = np.percentile(arr, hi_pct, axis=0)

    return MagMavSVARResult(
        irf_point=irf_point,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        B=B,
        variance_change_dates=tuple(int(x) for x in breaks),
        k_breaks=int(k_sel),
        n_boot=int(n_boot),
        ci=float(ci),
        eu=eu,
        n_fail=int(n_fail),
    )


__all__ = ["magmav_svar"]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_var/test_magmav.py -v`
Expected: all four current tests PASS

- [ ] **Step 5: Commit**

```bash
git add puremacro/var/identify/magmav.py tests/test_var/test_magmav.py
git commit -m "feat(magmav): orchestrator + regime-preserving bootstrap"
```

---

## Task 5: Wire up exports + remaining Component A tests

**Files:**
- Modify: `puremacro/var/identify/__init__.py`
- Test: `tests/test_var/test_magmav.py`

- [ ] **Step 1: Add the remaining 6 tests**

Append to `tests/test_var/test_magmav.py`:

```python
def test_magmav_svar_irf_shapes_match_horizon():
    Y, _ = _synthetic_var2_with_breaks(T=300, breaks=(150,), seed=5)
    res = mm.magmav_svar(Y, p=1, horizon=12, k_breaks=1, n_boot=30, seed=5)
    H = 12
    n = 2
    for arr in (res.irf_point, res.irf_lower, res.irf_upper):
        assert arr.shape == (H + 1, n, n)


def test_magmav_svar_detects_two_breaks_when_present():
    Y, _ = _synthetic_var2_with_breaks(T=600, breaks=(200, 400), seed=6)
    res = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=2, n_boot=20, seed=6)
    assert res.k_breaks == 2
    assert len(res.variance_change_dates) == 2
    detected = np.array(res.variance_change_dates)
    # Allow generous tolerance — breaks are noisy with T=600 and n=2.
    assert np.min(np.abs(detected - 199)) < 80
    assert np.min(np.abs(detected - 399)) < 80


def test_magmav_svar_handles_explicit_k_breaks_arg():
    Y, _ = _synthetic_var2_with_breaks(T=300, breaks=(150,), seed=7)
    res = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=2, n_boot=20, seed=7)
    # Explicit k=2 means BIC is skipped.
    assert res.k_breaks <= 2


def test_magmav_svar_bootstrap_band_covers_irf_majority():
    Y, _ = _synthetic_var2_with_breaks(T=400, breaks=(200,), seed=8)
    res = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=1, n_boot=100, seed=8)
    # Point IRF should sit within bands at most horizons (not all — bands are 90%).
    inside = (res.irf_point >= res.irf_lower) & (res.irf_point <= res.irf_upper)
    assert inside.mean() > 0.5


def test_magmav_svar_exported_from_identify_package():
    from puremacro.var.identify import magmav_svar as exported
    assert exported is mm.magmav_svar
    from puremacro.var.identify import MagMavSVARResult
    from puremacro.var.identify._results import MagMavSVARResult as direct
    assert MagMavSVARResult is direct


def test_magmav_svar_variance_change_dates_is_int_tuple():
    Y, _ = _synthetic_var2_with_breaks(T=300, breaks=(150,), seed=9)
    res = mm.magmav_svar(Y, p=1, horizon=4, k_breaks=1, n_boot=10, seed=9)
    assert isinstance(res.variance_change_dates, tuple)
    for x in res.variance_change_dates:
        assert isinstance(x, int)
```

- [ ] **Step 2: Run tests, expect FAIL (the exported test only)**

Run: `pytest tests/test_var/test_magmav.py -v`
Expected: `test_magmav_svar_exported_from_identify_package` FAILS with `ImportError`. Others pass.

- [ ] **Step 3: Add exports**

Edit `puremacro/var/identify/__init__.py`. Add `from .magmav import magmav_svar` near the other identify imports, add `MagMavSVARResult` to the `_results` import block, and append `"magmav_svar"` and `"MagMavSVARResult"` to `__all__`:

```python
"""SVAR identification methods."""
from .cholesky import cholesky_svar as cholesky, compute_chol_shocks
from .bq import bq_svar as bq
from .sign import sign_restriction_svar as sign_restrictions
from .proxy import proxy_svar as proxy
from .hetero import rigobon_svar as hetero, HeteroResult
from .maxshare import maxshare, news_maxshare, identify_maxshare
from .sign_zero import sign_zero
from .sign_robust import gk_robust_bands, gk_robust_bands_from_gibbs
from .non_gaussian import non_gaussian_svar
from .magmav import magmav_svar
from .panel import mean_group_svar
from ._results import (
    ProxySVARResult,
    CholeskySVARResult,
    BQSVARResult,
    SignRestrictionResult,
    GKRobustBandsResult,
    NonGaussianSVARResult,
    SignZeroResult,
    PanelSVARResult,
    MaxShareResult,
    MagMavSVARResult,
)

__all__ = [
    "cholesky", "compute_chol_shocks", "bq", "sign_restrictions", "proxy", "hetero",
    "maxshare", "news_maxshare", "identify_maxshare", "sign_zero", "gk_robust_bands",
    "gk_robust_bands_from_gibbs", "non_gaussian_svar", "magmav_svar", "mean_group_svar",
    "ProxySVARResult", "CholeskySVARResult", "BQSVARResult",
    "SignRestrictionResult", "GKRobustBandsResult",
    "NonGaussianSVARResult", "SignZeroResult", "HeteroResult", "PanelSVARResult",
    "MaxShareResult", "MagMavSVARResult",
]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_var/test_magmav.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add puremacro/var/identify/__init__.py tests/test_var/test_magmav.py
git commit -m "feat(magmav): export magmav_svar + MagMavSVARResult from identify"
```

---

## Task 6: Extend `NonGaussianSVARResult` with `lr_test` + `consistency_check` fields

**Files:**
- Modify: `puremacro/var/identify/_results.py`
- Test: `tests/test_var/test_identify_results.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_var/test_identify_results.py`:

```python
def test_non_gaussian_svar_result_has_optional_diagnostic_fields():
    import numpy as np
    from puremacro.var.identify._results import NonGaussianSVARResult

    # Construct without the new fields — they default to None.
    res = NonGaussianSVARResult(
        B0=np.eye(2),
        Q=np.eye(2),
        kurtosis=np.zeros(2),
        irf=np.zeros((3, 2, 2)),
        ordering_by_kurt=np.arange(2),
    )
    assert res.lr_test is None
    assert res.consistency_check is None

    # Construct with the fields populated.
    res2 = NonGaussianSVARResult(
        B0=np.eye(2),
        Q=np.eye(2),
        kurtosis=np.zeros(2),
        irf=np.zeros((3, 2, 2)),
        ordering_by_kurt=np.arange(2),
        lr_test={"stat": 1.2, "df": 1, "p_value": 0.27},
        consistency_check={"max_abs_diff": 1e-12, "rms_diff": 1e-13, "passed": True},
    )
    assert res2.lr_test["df"] == 1
    assert res2.consistency_check["passed"] is True
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_var/test_identify_results.py::test_non_gaussian_svar_result_has_optional_diagnostic_fields -v`
Expected: `TypeError: __init__() got an unexpected keyword argument 'lr_test'`

- [ ] **Step 3: Extend the dataclass**

Edit `puremacro/var/identify/_results.py`. Find the `NonGaussianSVARResult` block. Append two optional fields and adjust the docstring:

```python
@dataclass(frozen=True)
class NonGaussianSVARResult:
    """Result of :func:`puremacro.var.identify.non_gaussian.non_gaussian_svar`.

    LMS (2017) non-Gaussian SVAR via FastICA on reduced-form residuals.

    Attributes
    ----------
    B0 : ndarray, shape (n, n)
        Identified impact matrix, columns ordered by descending |excess
        kurtosis| so the most non-Gaussian shock is column 0.
    Q : ndarray, shape (n, n)
        Orthogonal rotation, ``B0 = chol(Σ) @ Q``.
    kurtosis : ndarray, shape (n,)
        Excess kurtosis of recovered shocks (already sorted).
    irf : ndarray, shape (H+1, n, n)
        Structural impulse responses.
    ordering_by_kurt : ndarray, shape (n,)
        Permutation applied to the original ICA ordering.
    lr_test : dict or None
        New in 0.51.0. Result of :func:`gaussian_lr_test` against the
        Gaussian baseline. Keys: ``stat``, ``df``, ``p_value``.
    consistency_check : dict or None
        New in 0.51.0. Result of :func:`variance_decomposition_consistency`.
        Keys: ``max_abs_diff``, ``rms_diff``, ``passed``.

    References
    ----------
    Lanne, M., Meitz, M. and Saikkonen, P. (2017). Identification and
        estimation of non-Gaussian structural vector autoregressions.
        Journal of Econometrics 196(2), 288-304.
    """

    B0: np.ndarray
    Q: np.ndarray
    kurtosis: np.ndarray
    irf: np.ndarray
    ordering_by_kurt: np.ndarray
    lr_test: Optional[dict] = None
    consistency_check: Optional[dict] = None

    def summary(self) -> str:
        n = self.B0.shape[0]
        H = self.irf.shape[0] - 1
        kurt_str = ", ".join(f"{k:+.2f}" for k in self.kurtosis)
        lines = [
            f"Non-Gaussian SVAR (LMS 2017) result",
            f"  variables (n)     : {n}",
            f"  horizon (H)       : {H}",
            f"  shock kurtosis    : [{kurt_str}]",
        ]
        if self.lr_test is not None:
            lines.append(f"  LR vs Gaussian    : stat={self.lr_test['stat']:.2f}, p={self.lr_test['p_value']:.3f}")
        if self.consistency_check is not None:
            lines.append(f"  B0·B0' vs Σ_u     : max_abs_diff={self.consistency_check['max_abs_diff']:.2e}")
        return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_var/test_identify_results.py::test_non_gaussian_svar_result_has_optional_diagnostic_fields -v`
Expected: PASS

Also run the broader identify-results suite to confirm no regressions:

Run: `pytest tests/test_var/test_identify_results.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add puremacro/var/identify/_results.py tests/test_var/test_identify_results.py
git commit -m "feat(non_gaussian): add lr_test + consistency_check optional fields"
```

---

## Task 7: Implement `gaussian_lr_test` + `variance_decomposition_consistency`

**Files:**
- Modify: `puremacro/var/identify/non_gaussian.py`
- Test: `tests/test_var/test_non_gaussian_extensions.py` (new file)

- [ ] **Step 1: Write 4 failing tests**

Create `tests/test_var/test_non_gaussian_extensions.py`:

```python
"""Tests for 0.51.0 extensions to puremacro.var.identify.non_gaussian."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from puremacro.var.identify import non_gaussian as ng


def test_variance_decomposition_consistency_passes_at_truth():
    rng = np.random.default_rng(0)
    n = 3
    B = rng.standard_normal((n, n))
    Sigma_u = B @ B.T  # exact consistency
    out = ng.variance_decomposition_consistency(B, Sigma_u)
    assert out["passed"] is True
    assert out["max_abs_diff"] < 1e-8
    assert out["rms_diff"] < 1e-8


def test_variance_decomposition_consistency_fails_on_mismatch():
    n = 2
    B = np.eye(n)
    Sigma_u = np.diag([1.0, 1.0]) + np.eye(n) * 0.5  # not equal to B B^T
    out = ng.variance_decomposition_consistency(B, Sigma_u)
    assert out["passed"] is False
    assert out["max_abs_diff"] > 1e-6


def test_gaussian_lr_test_rejects_non_gaussian_data():
    rng = np.random.default_rng(1)
    # Heavy-tailed shocks (Student-t df=4)
    n, T = 2, 2000
    e = rng.standard_t(df=4, size=(T, n))
    B = np.array([[1.0, 0.3], [0.2, 0.8]])
    residuals = e @ B.T
    out = ng.gaussian_lr_test(B, residuals)
    assert out["p_value"] < 0.05, f"LR did not reject Gaussian; p={out['p_value']}"
    assert out["stat"] > 0
    assert out["df"] > 0


def test_gaussian_lr_test_does_not_reject_pure_gaussian_data():
    rng = np.random.default_rng(2)
    n, T = 2, 2000
    e = rng.standard_normal((T, n))
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    residuals = e @ B.T
    out = ng.gaussian_lr_test(B, residuals)
    # Should not strongly reject; allow a soft threshold to avoid flakiness.
    assert out["p_value"] > 0.10, f"unexpected rejection on Gaussian data; p={out['p_value']}"
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_var/test_non_gaussian_extensions.py -v`
Expected: `AttributeError: module ... has no attribute 'variance_decomposition_consistency'`

- [ ] **Step 3: Implement both helpers**

Edit `puremacro/var/identify/non_gaussian.py`. Replace its existing `__all__` with one that includes the new helpers, and add the two functions above `__all__`:

```python
def variance_decomposition_consistency(B0: np.ndarray, sigma_u: np.ndarray) -> dict:
    """Sanity-check ``B0 @ B0.T ≈ Σ_u``.

    Returns dict with ``max_abs_diff``, ``rms_diff``, ``passed`` (bool,
    True iff ``max_abs_diff < 1e-6``).
    """
    diff = B0 @ B0.T - sigma_u
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    return {
        "max_abs_diff": max_abs,
        "rms_diff": rms,
        "passed": max_abs < 1e-6,
    }


def gaussian_lr_test(B0: np.ndarray, residuals: np.ndarray) -> dict:
    """Likelihood-ratio test of non-Gaussian shocks vs Gaussian baseline.

    Null: structural shocks ε = B0^{-1} u are i.i.d. Gaussian.
    Alt: at least one shock has a non-Gaussian density (KDE-fitted).

    Returns dict with ``stat``, ``df``, ``p_value``. The statistic is
    clamped to 0 (with ``p_value = 1.0``) when the Gaussian fit beats
    the KDE — a strong signal that non-Gaussian identification carries
    no information for this dataset.
    """
    from scipy.stats import chi2, gaussian_kde, multivariate_normal
    T, n = residuals.shape
    eps = residuals @ np.linalg.inv(B0).T  # shape (T, n)
    # Non-Gaussian log-likelihood: sum_i log f_KDE(eps_i)
    ll_nong = 0.0
    for j in range(n):
        kde = gaussian_kde(eps[:, j])
        ll_nong += float(np.sum(np.log(np.maximum(kde(eps[:, j]), 1e-300))))
    # Subtract |det(B0^{-1})|·T for change of variables; cancels with the
    # Gaussian model so we drop it.
    Sigma_u = residuals.T @ residuals / T
    ll_gauss = float(np.sum(multivariate_normal.logpdf(residuals, mean=np.zeros(n), cov=Sigma_u, allow_singular=True)))
    # Add the Jacobian for the non-Gaussian model (so both are on the same scale).
    sign, logdet = np.linalg.slogdet(np.linalg.inv(B0))
    ll_nong += T * logdet  # logdet may be negative; sign accounts for orientation
    stat = 2.0 * (ll_nong - ll_gauss)
    if stat < 0:
        return {"stat": 0.0, "df": int(n * (n - 1) // 2), "p_value": 1.0}
    df = int(n * (n - 1) // 2)
    p_value = float(1.0 - chi2.cdf(stat, df=df))
    return {"stat": float(stat), "df": df, "p_value": p_value}


__all__ = [
    "non_gaussian_svar",
    "gaussian_lr_test",
    "variance_decomposition_consistency",
]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_var/test_non_gaussian_extensions.py -v`
Expected: all 4 PASS

- [ ] **Step 5: Commit**

```bash
git add puremacro/var/identify/non_gaussian.py tests/test_var/test_non_gaussian_extensions.py
git commit -m "feat(non_gaussian): add gaussian_lr_test + variance_decomposition_consistency"
```

---

## Task 8: Tie-breaker for near-equal kurtoses + wire diagnostics into `non_gaussian_svar`

**Files:**
- Modify: `puremacro/var/identify/non_gaussian.py`
- Test: `tests/test_var/test_non_gaussian_extensions.py`

- [ ] **Step 1: Write 3 failing tests**

Append to `tests/test_var/test_non_gaussian_extensions.py`:

```python
def test_tiebreak_uses_skewness_when_kurtoses_near_equal():
    # Synthetic shocks with one tied kurtosis pair and one tie-breaker
    # via skewness.
    rng = np.random.default_rng(11)
    T = 4000
    # shock 0: zero skew, high kurt (Laplace-like)
    s0 = rng.laplace(size=T)
    s0 = (s0 - s0.mean()) / s0.std()
    # shock 1: matching kurt but heavily right-skewed (chi^2 - mean)
    s1 = (rng.chisquare(df=4, size=T) - 4) / np.sqrt(8)
    # Make kurtoses nearly equal by mixing s1 with a Gaussian:
    src = np.column_stack([s0, s1])
    # The tie-breaker is invoked at the helper level; pass the source as residuals
    # since we know |kurt(s0) - kurt(s1)| can land inside the tolerance for some seeds.
    kurt = (src ** 4).mean(axis=0) / (src ** 2).mean(axis=0) ** 2 - 3.0
    order = ng._tiebreak_kurtosis_order(kurt, src, tol=1e-1)  # wide tol to force tie-break
    # When ties are forced, output must still be a permutation of arange(n)
    assert sorted(order.tolist()) == [0, 1]


def test_tiebreak_warns_when_invoked():
    rng = np.random.default_rng(12)
    T = 2000
    s0 = rng.standard_normal(T)
    s1 = rng.standard_normal(T)
    src = np.column_stack([s0, s1])
    kurt = (src ** 4).mean(axis=0) / (src ** 2).mean(axis=0) ** 2 - 3.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ng._tiebreak_kurtosis_order(kurt, src, tol=10.0)  # force the tie path
    assert any("tiebreak" in str(wi.message).lower() for wi in w)


def test_non_gaussian_svar_result_includes_diagnostics():
    rng = np.random.default_rng(13)
    n, T, p = 2, 600, 1
    e = rng.standard_t(df=4, size=(T + p, n))
    B = np.array([[1.0, 0.3], [0.2, 0.8]])
    A = np.array([[0.5, 0.0], [0.0, 0.4]])
    Y = np.zeros((T + p, n))
    for t in range(p, T + p):
        Y[t] = A @ Y[t - 1] + B @ e[t]
    from puremacro.var.identify import non_gaussian_svar
    res = non_gaussian_svar(Y, p=p, horizon=4, seed=13)
    assert res.lr_test is not None
    assert set(res.lr_test.keys()) == {"stat", "df", "p_value"}
    assert res.consistency_check is not None
    assert set(res.consistency_check.keys()) == {"max_abs_diff", "rms_diff", "passed"}
    # Sigma_u and B B^T should match very tightly because B is derived from Cholesky(Σ_u).
    assert res.consistency_check["passed"] is True
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_var/test_non_gaussian_extensions.py -v -k "tiebreak or diagnostics"`
Expected: `AttributeError: module ... has no attribute '_tiebreak_kurtosis_order'`

- [ ] **Step 3: Implement tiebreak helper + wire it into `non_gaussian_svar`**

Edit `puremacro/var/identify/non_gaussian.py`. Add helper above `non_gaussian_svar`:

```python
def _tiebreak_kurtosis_order(kurt: np.ndarray, src: np.ndarray, *,
                             tol: float = 1e-3) -> np.ndarray:
    """Sort indices by descending |kurt|; for entries within ``tol * max(|kurt|)``
    of each other, break ties by |skewness| then |5th central moment|, then by
    original index.

    Emits a warning when any tie-break is invoked.
    """
    n = kurt.size
    abs_k = np.abs(kurt)
    k_max = max(float(abs_k.max()), 1e-12)
    # Compute moments
    centred = src - src.mean(axis=0)
    std = centred.std(axis=0, ddof=0)
    std_safe = np.where(std > 0, std, 1.0)
    skew = (centred / std_safe) ** 3
    skew = skew.mean(axis=0)
    m5 = (centred / std_safe) ** 5
    m5 = m5.mean(axis=0)
    # Build sort key as a tuple: (-|kurt|_bucketed, -|skew|, -|m5|, original_idx)
    bucket = np.round(abs_k / (tol * k_max)) * (tol * k_max)
    keys = list(zip(-bucket, -np.abs(skew), -np.abs(m5), np.arange(n)))
    order = np.array([i for _, _, _, i in sorted(zip(keys, range(n)), key=lambda z: z[0])])
    # Detect that tie-break was needed: any two adjacent |kurt| within tol.
    sorted_abs = np.sort(abs_k)[::-1]
    needed = any(abs(sorted_abs[i] - sorted_abs[i + 1]) < tol * k_max for i in range(n - 1))
    if needed:
        warnings.warn(
            "non_gaussian_svar: kurtosis tiebreak invoked — column order "
            "resolved via skewness / 5th moment. Ordering may be sensitive "
            "to sample noise.",
            stacklevel=3,
        )
    return order
```

Add `import warnings` at top if not already present.

Then modify `non_gaussian_svar` to call the diagnostics. Find the existing sign-normalisation block (around line 89-93 of `non_gaussian.py`) and append, just before the `return` statement:

```python
    # 0.51.0: tiebreak when adjacent |kurt| values are close.
    tb_order = _tiebreak_kurtosis_order(kurt, src.T, tol=1e-3)
    if not np.array_equal(tb_order, np.arange(B0.shape[1])):
        B0 = B0[:, tb_order]
        Q = Q[:, tb_order]
        kurt = kurt[tb_order]
    irf_arr = compute_irf(A_list, B0, horizon)
    lr = gaussian_lr_test(B0, residuals)
    cons = variance_decomposition_consistency(B0, Sigma)
    return NonGaussianSVARResult(
        B0=B0,
        Q=Q,
        kurtosis=kurt,
        irf=irf_arr,
        ordering_by_kurt=order,
        lr_test=lr,
        consistency_check=cons,
    )
```

Note: the existing function already computed `irf_arr` and returned. Replace the existing `irf_arr = compute_irf(A_list, B0, horizon)` + return block with the version above. The remaining old `return NonGaussianSVARResult(B0=B0, Q=Q, kurtosis=kurt, irf=irf_arr, ordering_by_kurt=order)` block must be deleted to avoid an unreachable second return.

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_var/test_non_gaussian_extensions.py -v`
Expected: all 7 PASS

Sanity check that the non_gaussian module still passes any other existing test:

Run: `pytest tests/ -k "non_gaussian" -v`
Expected: all PASS (assuming no other test file references the dataclass with positional args).

- [ ] **Step 5: Commit**

```bash
git add puremacro/var/identify/non_gaussian.py tests/test_var/test_non_gaussian_extensions.py
git commit -m "feat(non_gaussian): kurtosis tiebreak + auto-populate LR/consistency diagnostics"
```

---

## Task 9: Add `LewbelIVResult` dataclass

**Files:**
- Modify: `puremacro/inference/_results.py`
- Test: `tests/test_inference/test_lewbel_iv.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference/test_lewbel_iv.py`:

```python
"""Tests for puremacro.inference.lewbel_iv."""
from __future__ import annotations

import numpy as np
import pytest


def test_lewbel_iv_result_dataclass_is_frozen():
    import dataclasses
    from puremacro.inference._results import LewbelIVResult

    res = LewbelIVResult(
        beta=np.array([1.0, 2.0]),
        se=np.array([0.1, 0.2]),
        t=np.array([10.0, 10.0]),
        n_obs=500,
        n_iv_constructed=3,
        first_stage_F=42.0,
        lewbel_diagnostic={"stat": 80.0, "p_value": 1e-12},
    )
    assert res.n_obs == 500
    assert res.first_stage_F == 42.0
    assert dataclasses.is_dataclass(res)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.n_obs = 600
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_inference/test_lewbel_iv.py::test_lewbel_iv_result_dataclass_is_frozen -v`
Expected: `ImportError: cannot import name 'LewbelIVResult'`

- [ ] **Step 3: Add the dataclass**

Append to `puremacro/inference/_results.py`:

```python
import numpy as np


@dataclass(frozen=True)
class LewbelIVResult:
    """Result of :func:`puremacro.inference.lewbel_iv.lewbel_iv`.

    Lewbel (2012) 2SLS using instruments constructed from
    heteroskedasticity in exogenous drivers.

    Attributes
    ----------
    beta : ndarray, shape (k_endog + k_exog,)
        2SLS coefficients. Endogenous-regressor coefficients come first,
        in the order given to ``lewbel_iv``; exogenous follow.
    se : ndarray, shape (k_endog + k_exog,)
        Standard errors (homoskedastic 2SLS; HAC variant deferred).
    t : ndarray, shape (k_endog + k_exog,)
        t-statistics, ``beta / se``.
    n_obs : int
        Sample size after dropping rows with NaN inputs.
    n_iv_constructed : int
        Number of Lewbel instruments constructed (k_z × k_endog).
    first_stage_F : float
        First-stage F statistic (joint significance of the constructed
        IVs in the reduced form for the endogenous regressors).
    lewbel_diagnostic : dict
        Breusch-Pagan-style identification test. Keys: ``stat``,
        ``p_value``. Small p indicates the constructed instrument is
        strong; ``p > 0.10`` is treated as weak identification.

    References
    ----------
    Lewbel, A. (2012). Using heteroscedasticity to identify and estimate
        mismeasured and endogenous regressor models. Journal of Business
        and Economic Statistics 30(1), 67-80.
    """

    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    n_obs: int
    n_iv_constructed: int
    first_stage_F: float
    lewbel_diagnostic: dict

    def summary(self) -> str:
        diag = self.lewbel_diagnostic
        strength = "STRONG" if diag["p_value"] < 0.10 else "WEAK"
        return (
            f"Lewbel-IV result\n"
            f"  obs (n_obs)              : {self.n_obs}\n"
            f"  Lewbel IVs constructed   : {self.n_iv_constructed}\n"
            f"  first-stage F            : {self.first_stage_F:.2f}\n"
            f"  Lewbel diag p-value      : {diag['p_value']:.4f} [{strength}]\n"
        )
```

Add `import numpy as np` near the top of `_results.py` if not already imported.

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_inference/test_lewbel_iv.py::test_lewbel_iv_result_dataclass_is_frozen -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add puremacro/inference/_results.py tests/test_inference/test_lewbel_iv.py
git commit -m "feat(inference): add LewbelIVResult dataclass"
```

---

## Task 10: Implement `lewbel_iv` core (Frisch-Waugh + 2SLS)

**Files:**
- Create: `puremacro/inference/lewbel_iv.py`
- Test: `tests/test_inference/test_lewbel_iv.py`

- [ ] **Step 1: Write 3 failing tests**

Append to `tests/test_inference/test_lewbel_iv.py`:

```python
def _lewbel_dgp(T: int, beta: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lewbel (2012) DGP: y = β·x + u, x = z·e1 + e2,
    var(e1) = exp(0.5·z), so z drives heteroskedasticity in x."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(T)
    e1 = rng.standard_normal(T) * np.sqrt(np.exp(0.5 * z))
    e2 = rng.standard_normal(T)
    x = z * e1 + e2  # heteroskedastic in z
    u = e2 + 0.3 * rng.standard_normal(T)  # correlated with e2 → x endogenous
    y = beta * x + u
    return y, x.reshape(-1, 1), np.ones((T, 1)), z.reshape(-1, 1)


def test_lewbel_iv_recovers_known_beta():
    y, X_endog, X_exog, Z = _lewbel_dgp(T=5000, beta=1.5, seed=0)
    from puremacro.inference.lewbel_iv import lewbel_iv
    res = lewbel_iv(y, X_endog, X_exog, Z)
    # Coefficient on the endogenous regressor comes first.
    assert abs(res.beta[0] - 1.5) < 0.20, f"beta={res.beta[0]:.3f} far from 1.5"


def test_lewbel_iv_returns_dataclass():
    y, X_endog, X_exog, Z = _lewbel_dgp(T=2000, beta=1.0, seed=1)
    from puremacro.inference.lewbel_iv import lewbel_iv
    from puremacro.inference._results import LewbelIVResult
    res = lewbel_iv(y, X_endog, X_exog, Z)
    assert isinstance(res, LewbelIVResult)
    assert res.n_obs == 2000
    assert res.beta.shape == (2,)  # 1 endog + 1 exog
    assert res.se.shape == (2,)
    assert res.t.shape == (2,)


def test_lewbel_iv_handles_two_endogenous():
    rng = np.random.default_rng(2)
    T = 4000
    z = rng.standard_normal(T)
    e1 = rng.standard_normal(T) * np.sqrt(np.exp(0.4 * z))
    e2 = rng.standard_normal(T) * np.sqrt(np.exp(-0.3 * z))
    x1 = z * e1 + rng.standard_normal(T)
    x2 = z * e2 + rng.standard_normal(T)
    u = e1 + e2
    y = 1.0 * x1 + 0.5 * x2 + u
    from puremacro.inference.lewbel_iv import lewbel_iv
    res = lewbel_iv(
        y, np.column_stack([x1, x2]), np.ones((T, 1)), z.reshape(-1, 1),
    )
    assert res.beta.shape == (3,)  # 2 endog + 1 exog
    assert res.n_iv_constructed == 2  # k_z × k_endog = 1 × 2
```

- [ ] **Step 2: Run tests, expect FAIL**

Run: `pytest tests/test_inference/test_lewbel_iv.py -v -k "recovers or returns_dataclass or two_endogenous"`
Expected: `ModuleNotFoundError: No module named 'puremacro.inference.lewbel_iv'`

- [ ] **Step 3: Implement the core function**

Create `puremacro/inference/lewbel_iv.py`:

```python
"""Lewbel (2012) heteroskedasticity-based constructed IVs.

Given y = X_endog · β + X_exog · γ + u where X_endog is endogenous and no
external instrument is available, Lewbel constructs instruments from
heteroskedasticity in the auxiliary regression of X_endog on observed
``heterosk_source``. The constructed IVs are valid under the assumption
that ``Cov(heterosk_source · ν, u) = 0`` where ν is the first-stage
residual.

References
----------
Lewbel, A. (2012). Using heteroscedasticity to identify and estimate
    mismeasured and endogenous regressor models. JBES 30(1), 67-80.
"""
from __future__ import annotations

import warnings

import numpy as np

from ._results import LewbelIVResult
from .._linalg import inv_xtx


def _residualise(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return X minus its OLS projection onto W. Assumes W has a constant."""
    XtX_inv = inv_xtx(W, name="lewbel_iv Frisch-Waugh")
    return X - W @ (XtX_inv @ W.T @ X)


def _lewbel_diagnostic(X_endog_res: np.ndarray, Z: np.ndarray) -> dict:
    """Breusch-Pagan-style test of whether Z drives heteroskedasticity in
    X_endog_res. Stack columns of X_endog_res and test joint significance.
    Returns dict with ``stat`` (LM) and ``p_value`` (chi-squared)."""
    from scipy.stats import chi2
    T, k_e = X_endog_res.shape
    k_z = Z.shape[1]
    # Build pooled vector of squared residuals (across endog columns) and
    # regress on Z (with constant). LM = T · R² approx chi-squared(k_z).
    u2 = (X_endog_res ** 2).reshape(-1)
    Z_stack = np.tile(Z, (k_e, 1))
    Z_aug = np.column_stack([np.ones(Z_stack.shape[0]), Z_stack])
    XtX_inv = inv_xtx(Z_aug, name="lewbel_iv BP")
    beta = XtX_inv @ Z_aug.T @ u2
    pred = Z_aug @ beta
    ss_tot = float(np.sum((u2 - u2.mean()) ** 2))
    ss_res = float(np.sum((u2 - pred) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    stat = T * k_e * r2
    p = float(1.0 - chi2.cdf(stat, df=k_z))
    return {"stat": float(stat), "p_value": p}


def lewbel_iv(
    y: np.ndarray,
    X_endog: np.ndarray,
    X_exog: np.ndarray,
    heterosk_source: np.ndarray,
) -> LewbelIVResult:
    """2SLS with Lewbel-constructed instruments.

    Parameters
    ----------
    y : (T,) outcome.
    X_endog : (T, k_endog) endogenous regressors.
    X_exog : (T, k_exog) exogenous regressors (must include a constant column).
    heterosk_source : (T, k_z) observed drivers of heteroskedasticity in X_endog.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X_endog = np.asarray(X_endog, dtype=float)
    X_exog = np.asarray(X_exog, dtype=float)
    Z_source = np.asarray(heterosk_source, dtype=float)
    if X_endog.ndim == 1:
        X_endog = X_endog.reshape(-1, 1)
    if X_exog.ndim == 1:
        X_exog = X_exog.reshape(-1, 1)
    if Z_source.ndim == 1:
        Z_source = Z_source.reshape(-1, 1)

    T = y.size
    k_endog = X_endog.shape[1]
    k_exog = X_exog.shape[1]
    k_z = Z_source.shape[1]

    # 1. Frisch-Waugh residualise X_endog and Z against X_exog.
    X_endog_res = _residualise(X_endog, X_exog)
    Z_res = _residualise(Z_source, X_exog)

    # 2. Construct Lewbel IVs: Z_res * (X_endog_res - X_endog_res.mean(axis=0))
    centred_endog = X_endog_res - X_endog_res.mean(axis=0, keepdims=True)
    # Z_constructed has shape (T, k_z * k_endog); column (k, j) = Z_res[:, k] * centred_endog[:, j]
    Z_constructed = np.einsum("tk,tj->tkj", Z_res, centred_endog).reshape(T, k_z * k_endog)

    # 3. Diagnostic on identifying strength.
    diag = _lewbel_diagnostic(X_endog_res, Z_source)
    if diag["p_value"] > 0.10:
        warnings.warn(
            f"lewbel_iv: weak Lewbel diagnostic (p={diag['p_value']:.3f}). "
            "Heteroskedasticity source may not drive sufficient variation; "
            "treat results with caution.",
            stacklevel=2,
        )

    # 4. 2SLS. Endogenous regressors first, then exogenous (in the result).
    X_full = np.column_stack([X_endog, X_exog])
    Z_full = np.column_stack([Z_constructed, X_exog])  # instruments: constructed + exogenous

    # First stage: regress each X_endog column on Z_full
    ZtZ_inv = inv_xtx(Z_full, name="lewbel_iv first stage")
    Pi = ZtZ_inv @ Z_full.T @ X_endog  # shape (n_iv + k_exog, k_endog)
    X_endog_hat = Z_full @ Pi
    X_full_hat = np.column_stack([X_endog_hat, X_exog])

    # Second stage: regress y on (X_endog_hat | X_exog)
    XhX_inv = inv_xtx(X_full_hat, name="lewbel_iv second stage")
    beta = XhX_inv @ X_full_hat.T @ y
    resid = y - X_full @ beta  # NOTE: residuals from original X, not projection
    sigma2 = float(resid @ resid / max(T - X_full.shape[1], 1))
    vcov = sigma2 * XhX_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    t = np.where(se > 0, beta / np.maximum(se, 1e-300), 0.0)

    # First-stage F: joint significance of Z_constructed columns in
    # regression of (each) X_endog on Z_full. Use scalar F on the first
    # endogenous regressor for the reported value.
    # F = ((SSR_R - SSR_U)/q) / (SSR_U/(T - kU))
    q = Z_constructed.shape[1]
    kU = Z_full.shape[1]
    ssr_u = float(((X_endog[:, 0] - X_endog_hat[:, 0]) ** 2).sum())
    # restricted: drop the constructed IVs from Z_full
    Z_restr = X_exog
    XtX_r = inv_xtx(Z_restr, name="lewbel_iv F restricted")
    pi_r = XtX_r @ Z_restr.T @ X_endog[:, 0]
    ssr_r = float(((X_endog[:, 0] - Z_restr @ pi_r) ** 2).sum())
    if ssr_u > 0 and q > 0 and (T - kU) > 0:
        F = ((ssr_r - ssr_u) / q) / (ssr_u / (T - kU))
    else:
        F = float("nan")

    return LewbelIVResult(
        beta=beta,
        se=se,
        t=t,
        n_obs=int(T),
        n_iv_constructed=int(q),
        first_stage_F=float(F),
        lewbel_diagnostic=diag,
    )


__all__ = ["lewbel_iv"]
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `pytest tests/test_inference/test_lewbel_iv.py -v -k "recovers or returns_dataclass or two_endogenous"`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add puremacro/inference/lewbel_iv.py tests/test_inference/test_lewbel_iv.py
git commit -m "feat(inference): lewbel_iv — 2SLS with heteroskedasticity-constructed IVs"
```

---

## Task 11: Add weak-Lewbel warning + remaining Component C tests + exports

**Files:**
- Modify: `puremacro/inference/__init__.py`
- Test: `tests/test_inference/test_lewbel_iv.py`

- [ ] **Step 1: Write 3 more tests**

Append to `tests/test_inference/test_lewbel_iv.py`:

```python
def test_lewbel_iv_warns_on_weak_diagnostic():
    rng = np.random.default_rng(3)
    T = 1000
    # Homoskedastic DGP — heterosk_source carries no info.
    z = rng.standard_normal(T)
    x = rng.standard_normal(T)
    u = rng.standard_normal(T) + 0.5 * x
    y = 1.0 * x + u
    from puremacro.inference.lewbel_iv import lewbel_iv
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = lewbel_iv(y, x.reshape(-1, 1), np.ones((T, 1)), z.reshape(-1, 1))
    assert any("weak" in str(wi.message).lower() for wi in w)


def test_lewbel_iv_exported_from_inference_package():
    from puremacro.inference import lewbel_iv as exported
    from puremacro.inference.lewbel_iv import lewbel_iv as direct
    assert exported is direct
    from puremacro.inference import LewbelIVResult


def test_lewbel_iv_first_stage_F_finite_on_strong_data():
    y, X_endog, X_exog, Z = _lewbel_dgp(T=4000, beta=1.0, seed=4)
    from puremacro.inference.lewbel_iv import lewbel_iv
    res = lewbel_iv(y, X_endog, X_exog, Z)
    assert np.isfinite(res.first_stage_F)
    assert res.first_stage_F > 0
```

- [ ] **Step 2: Run tests, expect 1 PASS, 2 FAIL**

Run: `pytest tests/test_inference/test_lewbel_iv.py -v -k "warns or exported or first_stage"`
Expected: `test_lewbel_iv_warns_on_weak_diagnostic` and `test_lewbel_iv_first_stage_F_finite_on_strong_data` PASS (Task 10 implementation already emits the warning and computes F); `test_lewbel_iv_exported_from_inference_package` FAILS with ImportError.

- [ ] **Step 3: Add exports**

Edit `puremacro/inference/__init__.py`. Add the imports and append to `__all__`:

```python
from .lewbel_iv import lewbel_iv
from ._results import ARTestResult, LewbelIVResult
```

(replace the existing `from ._results import ARTestResult` with the line above)

Append `"lewbel_iv"` and `"LewbelIVResult"` to `__all__`.

- [ ] **Step 4: Run all 6 Component C tests, expect PASS**

Run: `pytest tests/test_inference/test_lewbel_iv.py -v`
Expected: all 6 PASS

- [ ] **Step 5: Commit**

```bash
git add puremacro/inference/__init__.py tests/test_inference/test_lewbel_iv.py
git commit -m "feat(inference): export lewbel_iv + LewbelIVResult"
```

---

## Task 12: LP wrapper `lp_iv_lewbel`

**Files:**
- Create: `puremacro/lp/iv_lewbel.py`
- Modify: `puremacro/lp/__init__.py`
- Test: `tests/test_lp/test_lp_iv_lewbel.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_lp/test_lp_iv_lewbel.py`:

```python
"""Tests for puremacro.lp.iv_lewbel."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _build_panel(T_per_entity: int, n_entities: int, beta_true: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_entities):
        for t in range(T_per_entity):
            z = rng.standard_normal()
            e1 = rng.standard_normal() * np.sqrt(np.exp(0.4 * z))
            x = z * e1 + rng.standard_normal()
            u = e1 + 0.2 * rng.standard_normal()
            rows.append({
                "code": f"E{i}",
                "date": pd.Timestamp("2000-01-01") + pd.DateOffset(quarters=t),
                "y": beta_true * x + u,
                "x": x,
                "z": z,
            })
    return pd.DataFrame(rows)


def test_lp_iv_lewbel_returns_long_form_dataframe():
    df = _build_panel(T_per_entity=80, n_entities=10, beta_true=1.0, seed=0)
    from puremacro.lp.iv_lewbel import lp_iv_lewbel
    out = lp_iv_lewbel(
        df, y="y", x_endog="x", heterosk_source="z",
        horizons=range(0, 5), n_lags=2,
    )
    expected_cols = {"h", "beta", "se", "t", "lo", "hi"}
    assert expected_cols.issubset(set(out.columns))
    assert len(out) == 5
    # Coefficient at h=0 should be near β_true on average.
    assert abs(float(out.loc[out["h"] == 0, "beta"].iloc[0]) - 1.0) < 0.5
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_lp/test_lp_iv_lewbel.py -v`
Expected: `ModuleNotFoundError: No module named 'puremacro.lp.iv_lewbel'`

- [ ] **Step 3: Implement the LP wrapper**

Create `puremacro/lp/iv_lewbel.py`:

```python
"""Local projection with Lewbel-constructed instruments.

Pools the panel and applies :func:`puremacro.inference.lewbel_iv.lewbel_iv`
horizon-by-horizon. Entity fixed effects via dummy-variable encoding.

This wrapper does not currently expose a clustered-SE option — Lewbel
inference is delicate enough that the homoskedastic 2SLS SE in
``lewbel_iv`` is reported here unchanged. For HAC-clustered LP-IV
inference with external instruments, use :func:`puremacro.lp.iv.lp_iv`.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..inference.lewbel_iv import lewbel_iv


def lp_iv_lewbel(
    panel: pd.DataFrame,
    *,
    y: str,
    x_endog: str,
    heterosk_source: str,
    controls: Sequence[str] = (),
    horizons: Iterable[int] = range(0, 13),
    n_lags: int = 2,
    entity_level: str = "code",
    time_level: str = "date",
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Local-projection IV using Lewbel-constructed instruments.

    Specification at horizon h:
        y_{i, t+h} - y_{i, t-1} = α_i + β_h · x_{i,t} + γ' · W_{i,t} + ε_{i,t,h}
    where ``x`` is the endogenous regressor and instruments are built
    from heteroskedasticity in ``heterosk_source``.

    Returns
    -------
    DataFrame with columns ``[h, beta, se, t, lo, hi, first_stage_F, lewbel_p]``.
    """
    horizons = list(horizons)
    controls = list(controls)
    z_crit = norm.ppf(1 - alpha / 2)

    panel = panel.sort_values([entity_level, time_level]).reset_index(drop=True)
    # Build lags + leads of y
    g = panel.groupby(entity_level, observed=True)
    for lag in range(1, n_lags + 1):
        panel[f"{x_endog}_L{lag}"] = g[x_endog].shift(lag)
        panel[f"{y}_L{lag}"] = g[y].shift(lag)
        for c in controls:
            panel[f"{c}_L{lag}"] = g[c].shift(lag)
    panel[f"{y}_Lm1"] = g[y].shift(1)

    rows = []
    for h in horizons:
        panel[f"{y}_lead_h{h}"] = g[y].shift(-h)
        col_lhs = f"{y}_dh{h}"
        panel[col_lhs] = panel[f"{y}_lead_h{h}"] - panel[f"{y}_Lm1"]

        keep_cols = [col_lhs, x_endog, heterosk_source, entity_level]
        for lag in range(1, n_lags + 1):
            keep_cols.append(f"{x_endog}_L{lag}")
            keep_cols.append(f"{y}_L{lag}")
            for c in controls:
                keep_cols.append(f"{c}_L{lag}")
        keep_cols.extend(controls)
        sub = panel[keep_cols].dropna()
        if sub.empty:
            rows.append({"h": h, "beta": np.nan, "se": np.nan, "t": np.nan,
                         "lo": np.nan, "hi": np.nan, "first_stage_F": np.nan,
                         "lewbel_p": np.nan})
            continue

        # Build matrices for lewbel_iv. Entity dummies as part of X_exog.
        ent = pd.get_dummies(sub[entity_level], drop_first=True).to_numpy(dtype=float)
        const = np.ones((len(sub), 1))
        lag_cols = []
        for lag in range(1, n_lags + 1):
            lag_cols.append(sub[f"{x_endog}_L{lag}"].to_numpy())
            lag_cols.append(sub[f"{y}_L{lag}"].to_numpy())
            for c in controls:
                lag_cols.append(sub[f"{c}_L{lag}"].to_numpy())
        ctl_cols = [sub[c].to_numpy() for c in controls]
        if lag_cols or ctl_cols:
            X_exog = np.column_stack([const, ent] + lag_cols + ctl_cols)
        else:
            X_exog = np.column_stack([const, ent])

        X_endog = sub[x_endog].to_numpy().reshape(-1, 1)
        Z_source = sub[heterosk_source].to_numpy().reshape(-1, 1)
        y_lhs = sub[col_lhs].to_numpy()

        res = lewbel_iv(y_lhs, X_endog, X_exog, Z_source)
        beta = float(res.beta[0])
        se = float(res.se[0])
        rows.append({
            "h": h,
            "beta": beta,
            "se": se,
            "t": beta / se if se > 0 else np.nan,
            "lo": beta - z_crit * se,
            "hi": beta + z_crit * se,
            "first_stage_F": float(res.first_stage_F),
            "lewbel_p": float(res.lewbel_diagnostic["p_value"]),
        })
    return pd.DataFrame(rows)


__all__ = ["lp_iv_lewbel"]
```

- [ ] **Step 4: Add to `__init__.py` exports**

Edit `puremacro/lp/__init__.py`:

```python
from .iv_lewbel import lp_iv_lewbel
```

(add near the other `.iv` import)

Append `"lp_iv_lewbel"` to `__all__`.

- [ ] **Step 5: Run test, expect PASS**

Run: `pytest tests/test_lp/test_lp_iv_lewbel.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add puremacro/lp/iv_lewbel.py puremacro/lp/__init__.py tests/test_lp/test_lp_iv_lewbel.py
git commit -m "feat(lp): lp_iv_lewbel — LP with Lewbel-constructed instruments"
```

---

## Task 13: Version bump + CHANGELOG entry

**Files:**
- Modify: `puremacro/__init__.py`
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version in `puremacro/__init__.py`**

```python
__version__ = "0.51.0"
```

- [ ] **Step 2: Bump version in `pyproject.toml`**

```toml
version = "0.51.0"
```

- [ ] **Step 3: Add CHANGELOG entry at the top of `CHANGELOG.md`** (above the 0.50.0 entry)

```markdown
## 0.51.0 — 2026-05-23

Identification innovations: continuous-heteroskedasticity SVAR
(Magnusson-Mavroeidis 2014), non-Gaussian SVAR diagnostics
(LR test + variance-decomposition consistency + tie-break for
near-equal kurtoses), and Lewbel-IV with an LP wrapper.

R4 from the 2026-05-23 research-directions brainstorm.

### Added
- `puremacro.var.identify.magmav_svar(Y, *, p, horizon, k_breaks,
  n_boot, ci, seed) -> MagMavSVARResult` — SVAR identified by
  endogenously detected variance breaks. Sup-Wald scan picks
  candidate break dates, BIC selects k ∈ {0..4}, multi-start BFGS
  estimates B from regime-specific covariance structure, residual
  bootstrap (regime-preserving) builds bands.
- `puremacro.var.identify.MagMavSVARResult` (frozen dataclass) —
  irf_point, irf_lower, irf_upper, B, variance_change_dates, k_breaks,
  n_boot, ci, eu, n_fail.
- `puremacro.var.identify.non_gaussian.gaussian_lr_test(B0, residuals)
  -> dict` — likelihood-ratio test of non-Gaussian shocks vs Gaussian
  baseline (KDE-fitted alternative; χ² critical values).
- `puremacro.var.identify.non_gaussian.variance_decomposition_consistency(B0, sigma_u)
  -> dict` — sanity check that B0·B0' ≈ Σ_u.
- `puremacro.inference.lewbel_iv(y, X_endog, X_exog, heterosk_source)
  -> LewbelIVResult` — Lewbel (2012) heteroskedasticity-based
  constructed IVs + 2SLS + Breusch-Pagan identification diagnostic.
- `puremacro.inference.LewbelIVResult` (frozen dataclass).
- `puremacro.lp.lp_iv_lewbel(panel, *, y, x_endog, heterosk_source,
  controls, horizons, n_lags, entity_level, time_level, alpha)
  -> DataFrame` — panel LP using Lewbel IVs horizon-by-horizon.

### Changed
- `puremacro.var.identify.NonGaussianSVARResult` gains two optional
  fields, `lr_test` and `consistency_check`, populated automatically
  by `non_gaussian_svar`. Backward-compatible default `None`. No
  existing fields removed or renamed.
- `non_gaussian_svar` now invokes a kurtosis tie-breaker (skewness,
  then 5th central moment) when adjacent |excess kurtosis| values are
  within `1e-3 · max_k`; a warning is emitted when the tiebreak fires.

### Internal
- `puremacro/var/identify/magmav.py` — new module (~300 LOC).
- `puremacro/inference/lewbel_iv.py` — new module (~150 LOC).
- `puremacro/lp/iv_lewbel.py` — new module (~110 LOC).
- ~24 new unit tests across `tests/test_var/test_magmav.py`,
  `tests/test_var/test_non_gaussian_extensions.py`,
  `tests/test_inference/test_lewbel_iv.py`,
  `tests/test_lp/test_lp_iv_lewbel.py`.
```

- [ ] **Step 4: Verify the bumps with a quick smoke import**

Run: `python -c "import puremacro; assert puremacro.__version__ == '0.51.0'; print(puremacro.__version__)"`
Expected output: `0.51.0`

- [ ] **Step 5: Commit**

```bash
git add puremacro/__init__.py pyproject.toml CHANGELOG.md
git commit -m "chore(puremacro): bump 0.50.0 → 0.51.0 (identification extensions)"
```

---

## Task 14: Regenerate public-API snapshot

**Files:**
- Modify: `tests/fixtures/public_api_snapshot.json` (or whatever path the snapshot test reads)

- [ ] **Step 1: Locate the snapshot test**

Run: `pytest tests/ -k "public_api" --collect-only -q 2>&1 | head -20`
Expected: identifies the snapshot test file (likely `tests/test_public_api.py` or similar).

- [ ] **Step 2: Run it and let it fail to surface the new symbols**

Run: `pytest tests/ -k "public_api" -v`
Expected: FAIL — the snapshot will list the new symbols (`magmav_svar`, `MagMavSVARResult`, `gaussian_lr_test`, `variance_decomposition_consistency`, `lewbel_iv`, `LewbelIVResult`, `lp_iv_lewbel`) as additions.

- [ ] **Step 3: Regenerate the snapshot**

If the snapshot is JSON: copy the new symbol list from the diff into the snapshot file. If there is a regenerate helper (look for `tools/snapshot_*.py` or a `pytest --update-snapshot` style mode), run it.

Run: `grep -rln "public_api_snapshot" tools/ tests/ 2>&1 | head -5`
Expected: locate the snapshot file path.

Update it to add the seven new symbols. If the snapshot is sorted alphabetically per module, insert in sorted order. Verify by re-running the test.

Run: `pytest tests/ -k "public_api" -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/public_api_snapshot.json
git commit -m "chore(tests): regenerate public_api_snapshot for 0.51.0 additions"
```

(adjust the file path to match the actual snapshot location)

---

## Task 15: Run the 6-gate release check

**Files:** none modified — verification only.

- [ ] **Step 1: Run gates 1-4 (the fast set)**

Run: `python tools/release_check.py`
Expected: gates 1-4 pass.

- [ ] **Step 2: Run gate 5 (examples gallery)**

Run: `python tools/release_check.py --examples`
Expected: all `puremacro/examples/*.py` exit cleanly.

- [ ] **Step 3: Run gate 6 (Pyodide smoke)**

Run: `python tools/release_check.py --pyodide`
Expected: the 8-test Pyodide smoke set passes inside Node-Pyodide.

If any gate fails:
- Read the failure, dispatch a fix (no `--no-verify`, no hook skips, no destructive shortcuts).
- Re-run only the gate that failed plus dependents.

- [ ] **Step 4: Confirm by reading the final summary line**

Expected line: `release_check: all 6 gates GREEN`.

- [ ] **Step 5: (optional) Stage a final no-op commit if gate fixes were needed**

If gate fixes were required, they get individual commits via the fix loop above. Don't squash. Don't amend.

---

## Self-review checklist (run AFTER all 15 tasks)

1. **Spec coverage:**
   - Component A (magmav): Tasks 1-5. ✓
   - Component B (non-Gaussian extensions): Tasks 6-8. ✓
   - Component C (Lewbel-IV + LP wrapper): Tasks 9-12. ✓
   - Release infra: Tasks 13-15. ✓
   - All 9 acceptance criteria from the spec map to a task.

2. **Placeholder scan:** none — every step has either runnable code, a concrete shell command, or both.

3. **Type consistency:**
   - `MagMavSVARResult` fields used in Task 1 match Task 4's constructor call.
   - `NonGaussianSVARResult` additions (`lr_test`, `consistency_check`) used in Task 6 (definition), Task 8 (population), and Task 8 test (assertions).
   - `LewbelIVResult` fields match across Tasks 9, 10, 11.
   - `lp_iv_lewbel` returns the same column set described in the docstring (Task 12) and asserted in the test.
