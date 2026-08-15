# puremacro Phase 2 Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the legacy `svar/*`, `lp/lp_*.py`, and (downstream) `inference/legacy/*` paths to thin DeprecationWarning shims of their canonical homes, shipping as 0.42.0 with full deletion targeted for 0.43.0.

**Architecture:** Each retired file becomes a re-export shim that (a) emits exactly one `DeprecationWarning` on import naming the canonical path and removal version, (b) translates return shapes where the canonical signature shifted (e.g. transpose `(H+1,n,n)` → `(n,n,H+1)`, unpack frozen dataclass → legacy tuple), (c) leaves notebook deep-imports working unchanged. Notebooks are deferred to the 0.43.0 paper-figure refresh per the "builders clobber executed outputs" memory pin.

**Tech Stack:** Python ≥3.10, numpy/scipy/pandas/matplotlib (Pyodide promise), pytest, frozen dataclasses, `warnings.warn(..., DeprecationWarning, stacklevel=2)`.

**Source spec:** `docs/specs/2026-05-17-puremacro-phase2-consolidation-design.md` (commit `39f8ff0`).

**Deviations from spec discovered during signature verification:**
1. The spec claimed "9 SVAR tests still import `puremacro.svar`" (sourced from ARCHITECTURE.md). Verification found tests are already on canonical paths. Tasks for "migrate 9 SVAR tests" are accordingly trimmed in the plan — only test files still touching legacy need work.
2. `experiment.py` is already on canonical — no migration needed.
3. `gar/qar.py` does not import legacy — no migration needed.
4. `svar/identify_maxshare.py` (285 LOC with its own `MaxShareResult`) is materially richer than `var/identify/maxshare.maxshare` (85 LOC, returns just `B0`) — it joins `panel_svar.py` as a Phase-2.5 candidate (not shimmed in 0.42.0).
5. `var/identify/proxy.py` keeps the `(n,n,H+1)` axis convention (it routes through `wild_bootstrap_var`), so the proxy shim does NOT transpose — unlike cholesky/bq/sign which do.
6. `lp/lp_*.py` files are full independent implementations, not aliases. Plan opens with a per-file parity check; files that fail parity get a Phase-2.5 banner rather than a shim.

---

## File structure

**New files:**
- `puremacro/var/_results.py` — add `VarEstimateResult` frozen dataclass.
- `tests/test_deprecation_warnings.py` — parametrized DeprecationWarning gate for every shim.
- `tests/test_shim_shape_preservation.py` — assert shim return shapes match the legacy contract on a fixed seed.

**Modified files (cut to canonical, no shim):**
- `puremacro/var/estimate.py` — wrap 5-tuple return in `VarEstimateResult`.
- `puremacro/teaching/bq_canonical.py` — switch from `svar.identify_bq.bq_svar` to `var.identify.bq.bq_svar`, attribute access instead of tuple unpack.

**Modified files (converted to shim):**
- `puremacro/svar/estimate_var.py`
- `puremacro/svar/identify_cholesky.py`
- `puremacro/svar/identify_bq.py`
- `puremacro/svar/identify_sign.py`
- `puremacro/svar/identify_proxy.py`
- `puremacro/svar/identify_heteroskedasticity.py`
- `puremacro/lp/lp_jorda.py` (conditional on parity)
- `puremacro/lp/lp_iv.py` (conditional)
- `puremacro/lp/lp_panel.py` (conditional)
- `puremacro/lp/lp_panel_dk.py` (conditional)
- `puremacro/lp/lp_state_dep.py` (conditional)
- `puremacro/lp/lp_smooth.py` (conditional)
- `puremacro/lp/lp_garch_state.py` (conditional)
- `puremacro/lp/lp_garch_in_mean.py` (conditional)
- `puremacro/lp/garch_utils.py` — promote to `_garch_utils.py` or keep as private helper; clarify in Task 7.

**Left alone at 0.42.0 (Phase-2.5 candidates, banner only):**
- `puremacro/svar/panel_svar.py` — no canonical `var/identify/panel.py` yet.
- `puremacro/svar/identify_maxshare.py` — canonical maxshare lacks the full pipeline.

**Documentation:**
- `ARCHITECTURE.md` — update "Known consolidation candidates" to point at this plan's outputs.
- `CHANGELOG.md` — 0.42.0 entry.
- `puremacro/__init__.py` — bump to 0.42.0.
- `pyproject.toml` — bump version to 0.42.0.
- `tests/test_import.py` — update expected version string.
- `tests/fixtures/public_api_snapshot.json` — regenerate for `VarEstimateResult`.

---

## Task 0: Baseline + parity audit

**Files:**
- Read-only: every file listed above.

- [ ] **Step 1: Confirm baseline pytest green and capture count**

Run:
```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

Expected last line: `XXXX passed in ...` (record the number; will use as baseline).

- [ ] **Step 2: Confirm pytest does not promote DeprecationWarning to error**

Run:
```bash
grep -E "filterwarnings|filter_warnings" pyproject.toml tests/conftest.py 2>&1 || echo "no filter config"
```

Expected: "no filter config" or zero hits.

If hits are found that promote `DeprecationWarning` to error, abort and add a `pyproject.toml` change to scope it back to `default::DeprecationWarning` before the rest of the plan.

- [ ] **Step 3: Per-file legacy-vs-canonical parity audit for lp/lp_*.py**

For each pair below, dump both signatures (`def` lines + return docstrings) and identify whether the legacy is reasonably treatable as a thin shim of the canonical:

```bash
for legacy in lp_jorda lp_iv lp_panel lp_panel_dk lp_state_dep lp_smooth lp_garch_state lp_garch_in_mean; do
  canonical="${legacy#lp_}"
  echo "=== $legacy vs $canonical ==="
  diff <(grep -nE "^def |^class |^__all__" "puremacro/lp/$legacy.py") \
       <(grep -nE "^def |^class |^__all__" "puremacro/lp/$canonical.py")
done
```

For each pair where the legacy and canonical share `__all__` / top-level def names (so a `from canonical import *` shim is shape-preserving), record "PARITY-OK" against that pair. Otherwise record "DEFER-2.5" and treat the file as a Phase-2.5 candidate later in the plan.

Record results in this format and keep the list — Task 5 references it:

```
parity_audit:
  lp_jorda:    PARITY-OK  | DEFER-2.5
  lp_iv:       PARITY-OK  | DEFER-2.5
  lp_panel:    PARITY-OK  | DEFER-2.5
  lp_panel_dk: PARITY-OK  | DEFER-2.5
  lp_state_dep: PARITY-OK | DEFER-2.5
  lp_smooth:    PARITY-OK | DEFER-2.5
  lp_garch_state:    PARITY-OK | DEFER-2.5
  lp_garch_in_mean:  PARITY-OK | DEFER-2.5
```

No commit yet — this is read-only.

- [ ] **Step 4: Create a working notes file to track the audit results**

```bash
cat > docs/plans/_phase2_audit_notes.md <<'EOF'
# Phase-2 audit notes — 2026-05-17

Baseline pytest count: <N> passed.

## Parity audit results
<paste step 3 output here>
EOF
git add docs/plans/_phase2_audit_notes.md
git commit -m "docs(phase2): record baseline + parity audit"
```

---

## Task 1: Add VarEstimateResult dataclass + wrap canonical estimate_var

**Files:**
- Create: `puremacro/var/_results.py`
- Modify: `puremacro/var/estimate.py`
- Modify: `puremacro/var/__init__.py`
- Test: `tests/test_var/test_estimate_result.py`

**Why:** The spec calls for retiring the only remaining tuple-return result-object violation in the canonical path. `var/estimate.estimate_var` currently returns a bare 5-tuple. Promoting it to `VarEstimateResult` is the precondition for the `svar/estimate_var.py` shim and a Phase-3-compliant API.

- [ ] **Step 1: Write the failing test**

Create `tests/test_var/test_estimate_result.py`:

```python
"""Tests for VarEstimateResult dataclass and var.estimate_var return shape."""
import numpy as np
import pytest


def _toy_var2(seed: int = 0):
    rng = np.random.default_rng(seed)
    T, n = 200, 2
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    return Y


def test_var_estimate_result_is_frozen_and_has_expected_fields():
    from puremacro.var._results import VarEstimateResult

    r = VarEstimateResult(
        A_list=[np.eye(2)],
        c=np.zeros(2),
        Sigma=np.eye(2),
        resid=np.zeros((10, 2)),
        X=np.zeros((10, 3)),
    )
    assert r.A_list[0].shape == (2, 2)
    with pytest.raises(Exception):
        r.c = np.ones(2)  # frozen


def test_estimate_var_returns_var_estimate_result():
    from puremacro.var.estimate import estimate_var
    from puremacro.var._results import VarEstimateResult

    Y = _toy_var2()
    r = estimate_var(Y, p=2)
    assert isinstance(r, VarEstimateResult)
    assert len(r.A_list) == 2
    assert r.Sigma.shape == (2, 2)


def test_var_estimate_result_supports_legacy_tuple_unpack():
    """Backwards-compatibility tail: existing code that does `A, c, S, e, X = estimate_var(Y, p)` must still work."""
    from puremacro.var.estimate import estimate_var

    Y = _toy_var2()
    A_list, c, Sigma, resid, X = estimate_var(Y, p=2)
    assert len(A_list) == 2
    assert c.shape == (2,)
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python -m pytest tests/test_var/test_estimate_result.py -v
```

Expected: All three tests FAIL — `ModuleNotFoundError: No module named 'puremacro.var._results'` or `AttributeError`.

- [ ] **Step 3: Create `puremacro/var/_results.py`**

```python
"""Frozen-dataclass result objects for puremacro.var.* estimators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VarEstimateResult:
    """Result of :func:`puremacro.var.estimate.estimate_var`.

    Attributes
    ----------
    A_list : list of ndarray, length p
        VAR coefficient matrices A_1, ..., A_p, each shape (n, n).
    c : ndarray, shape (n,)
        Constant term.
    Sigma : ndarray, shape (n, n)
        Reduced-form residual covariance.
    resid : ndarray, shape (T - p, n)
        Reduced-form residuals.
    X : ndarray, shape (T - p, 1 + n*p)
        Design matrix (constant + p lags).
    """

    A_list: list[np.ndarray]
    c: np.ndarray
    Sigma: np.ndarray
    resid: np.ndarray
    X: np.ndarray

    def __iter__(self):
        """Support legacy `A_list, c, Sigma, resid, X = estimate_var(...)` unpack."""
        yield self.A_list
        yield self.c
        yield self.Sigma
        yield self.resid
        yield self.X

    def summary(self) -> str:
        p = len(self.A_list)
        n = self.Sigma.shape[0]
        T_eff = self.resid.shape[0]
        return (
            f"VAR estimate\n"
            f"  variables (n)     : {n}\n"
            f"  lag order (p)     : {p}\n"
            f"  effective T       : {T_eff}\n"
            f"  Σ trace           : {float(np.trace(self.Sigma)):.4f}\n"
        )
```

- [ ] **Step 4: Modify `puremacro/var/estimate.py` to return `VarEstimateResult`**

Replace the existing `estimate_var` body. Open the file and find:

```python
def estimate_var(Y: np.ndarray, p: int):
    """OLS estimation with a constant. Returns (A_list, c, Sigma, resid, X_design)."""
    ...
    return A_list, c, Sigma, resid, X
```

Change to:

```python
def estimate_var(Y: np.ndarray, p: int):
    """OLS estimation with a constant.

    Returns
    -------
    VarEstimateResult
        Frozen dataclass with attributes ``A_list``, ``c``, ``Sigma``,
        ``resid``, ``X``. Iterable so existing 5-tuple unpacks continue to work.
    """
    from ._results import VarEstimateResult
    T, n = Y.shape
    X = np.column_stack([np.ones(T - p)] + [Y[p - l - 1 : T - l - 1] for l in range(p)])
    Yd = Y[p:]
    B = np.linalg.lstsq(X, Yd, rcond=None)[0]
    c = B[0]
    A_list = [B[1 + l * n : 1 + (l + 1) * n].T for l in range(p)]
    resid = Yd - X @ B
    Sigma = resid.T @ resid / (T - p - 1 - n * p)
    return VarEstimateResult(A_list=A_list, c=c, Sigma=Sigma, resid=resid, X=X)
```

Also update `select_lag_bic` and `lag_select` in the same file, where they do `_, _, Sigma, _, _ = estimate_var(Y, p)`. Both already work via `__iter__`, so no change needed — confirm by re-reading those functions and leaving them.

- [ ] **Step 5: Export `VarEstimateResult` from `puremacro/var/__init__.py`**

Read the current contents of `puremacro/var/__init__.py`, find its `__all__`, and add:

```python
from ._results import VarEstimateResult  # noqa: F401
```

…with `"VarEstimateResult"` appended to `__all__` if `__all__` is defined.

- [ ] **Step 6: Run the new test to confirm it passes**

```bash
python -m pytest tests/test_var/test_estimate_result.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Run the full var suite to confirm no regressions**

```bash
python -m pytest tests/test_var/ tests/test_cholesky_shocks.py tests/test_robustness.py -q
```

Expected: all pass. If any failure mentions tuple unpacking, the `__iter__` shim is misaligned — re-verify Step 3.

- [ ] **Step 8: Run the Pyodide compat sweep**

```bash
python -m pytest tests/test_pyodide_compat.py -v
```

Expected: 2 passed. (`_results.py` adds no forbidden imports.)

- [ ] **Step 9: Commit**

```bash
git add puremacro/var/_results.py puremacro/var/estimate.py puremacro/var/__init__.py tests/test_var/test_estimate_result.py
git commit -m "$(cat <<'EOF'
feat(var): VarEstimateResult dataclass — Phase 2 prep

Wraps estimate_var's 5-tuple in a frozen dataclass; __iter__ keeps
existing 5-tuple unpack callers working. Precondition for the
svar/estimate_var shim landing in this consolidation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Shim test infrastructure

**Files:**
- Create: `tests/test_deprecation_warnings.py`
- Create: `tests/test_shim_shape_preservation.py`

**Why:** Centralized parametrized test pattern catches missing warnings or mis-shaped shim returns across all 15 shims in one place. Failing fast and uniformly here beats per-file ad-hoc tests.

- [ ] **Step 1: Create `tests/test_deprecation_warnings.py`**

```python
"""Per-shim DeprecationWarning gate.

For each entry in SHIMS, asserts that importing the legacy path:
  1. emits exactly one DeprecationWarning,
  2. names the legacy path in the message,
  3. names the canonical replacement in the message.

The shim list starts empty and is populated by Tasks 3 and 5 as each
shim lands.
"""
from __future__ import annotations

import importlib
import sys
import warnings

import pytest


# Populated as shims land. Each tuple: (legacy_dotted_path, canonical_dotted_path).
SHIMS: list[tuple[str, str]] = []


@pytest.mark.parametrize("legacy_path,canonical_path", SHIMS)
def test_shim_emits_one_deprecation_warning(legacy_path, canonical_path):
    # Force a clean import so __warningregistry__ doesn't suppress the warning.
    for mod in list(sys.modules):
        if mod == legacy_path or mod.startswith(legacy_path + "."):
            del sys.modules[mod]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.import_module(legacy_path)
    dws = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert len(dws) == 1, (
        f"Expected exactly one DeprecationWarning from `import {legacy_path}`; "
        f"got {len(dws)}: {[str(w.message) for w in dws]}"
    )
    msg = str(dws[0].message)
    assert legacy_path in msg, f"warning text {msg!r} missing legacy path {legacy_path!r}"
    assert canonical_path in msg, f"warning text {msg!r} missing canonical path {canonical_path!r}"
```

- [ ] **Step 2: Create `tests/test_shim_shape_preservation.py` (skeleton)**

```python
"""Per-shim return-shape contract.

For shims whose canonical sibling has a different return shape than the
legacy contract (e.g. CholeskySVARResult dataclass vs legacy 3-tuple;
(H+1,n,n) axis order vs (n,n,H+1)), assert that the shim translates
correctly on a fixed-seed toy dataset.

Tests are added by Tasks 3a–3f as each non-trivial shim lands.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _toy_var2(seed: int = 0):
    rng = np.random.default_rng(seed)
    T, n = 100, 2
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    return Y


@pytest.fixture
def suppress_shim_warnings():
    """Capture and ignore the DeprecationWarning so subsequent assertions are clean."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


# Individual per-shim tests added by Tasks 3a–3f follow this fixture.
```

- [ ] **Step 3: Run the empty test files to verify they parse**

```bash
python -m pytest tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py -v
```

Expected: 0 passed (no parametrized cases yet); no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py
git commit -m "$(cat <<'EOF'
test(phase2): scaffold shim warning + shape-preservation gates

Empty parametrized test files. Tasks 3 and 5 populate them as each
shim lands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3a: svar/estimate_var.py → shim of var/estimate

**Files:**
- Modify: `puremacro/svar/estimate_var.py`
- Modify: `tests/test_deprecation_warnings.py`

**Goal:** Legacy callers can still do `A, c, S, e, X = svar.estimate_var.estimate_var(Y, p)`. The shim re-exports the canonical, which itself returns `VarEstimateResult` (iterable, so legacy unpack still works).

- [ ] **Step 1: Append the shim entry to `tests/test_deprecation_warnings.py::SHIMS`**

Find the line `SHIMS: list[tuple[str, str]] = []` and replace with:

```python
SHIMS: list[tuple[str, str]] = [
    ("puremacro.svar.estimate_var", "puremacro.var.estimate"),
]
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python -m pytest tests/test_deprecation_warnings.py -v
```

Expected: FAIL — `assert len(dws) == 1` fails because the current legacy module emits no warning.

- [ ] **Step 3: Rewrite `puremacro/svar/estimate_var.py` as a shim**

Replace the entire file content with:

```python
"""DEPRECATED — use puremacro.var.estimate.

Phase-2 shim (0.42.0). Removal target: 0.43.0 with the next paper-figure
refresh cycle.

Re-exports the canonical estimate_var. The canonical now returns a
VarEstimateResult frozen dataclass that is iterable, so legacy
`A_list, c, Sigma, resid, X = estimate_var(Y, p)` unpacks continue to work.
"""
import warnings as _w

from puremacro.var.estimate import estimate_var, select_lag_bic  # noqa: F401

_w.warn(
    "puremacro.svar.estimate_var is deprecated since 0.42.0; "
    "use puremacro.var.estimate. Removal target: 0.43.0.",
    DeprecationWarning,
    stacklevel=2,
)
```

- [ ] **Step 4: Run the deprecation-warning test**

```bash
python -m pytest tests/test_deprecation_warnings.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Verify the legacy unpack contract**

```bash
python -c "
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import numpy as np
from puremacro.svar.estimate_var import estimate_var
rng = np.random.default_rng(0); Y = rng.standard_normal((50, 2))
A, c, S, e, X = estimate_var(Y, p=1)
print('unpack ok:', len(A), c.shape, S.shape)
"
```

Expected output: `unpack ok: 1 (2,) (2, 2)` (or similar).

- [ ] **Step 6: Run downstream consumers (legacy svar identification files all import this)**

```bash
python -m pytest tests/test_var/ tests/test_cholesky_shocks.py tests/test_robustness.py -q -W "ignore::DeprecationWarning:puremacro.svar"
```

Expected: same green count as Task 1 / baseline. No regressions.

- [ ] **Step 7: Commit**

```bash
git add puremacro/svar/estimate_var.py tests/test_deprecation_warnings.py
git commit -m "$(cat <<'EOF'
refactor(svar): estimate_var → shim of puremacro.var.estimate

Phase-2 shim. Removal target: 0.43.0. Legacy 5-tuple unpack callers
keep working via VarEstimateResult.__iter__.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3b: svar/identify_cholesky.py → shim of var/identify/cholesky

**Files:**
- Modify: `puremacro/svar/identify_cholesky.py`
- Modify: `tests/test_deprecation_warnings.py`
- Modify: `tests/test_shim_shape_preservation.py`

**Key translation:** legacy returns `(point, lower, upper)` 3-tuple each shape `(n, n, H+1)`. Canonical returns `CholeskySVARResult` with `irf_point/lower/upper` shape `(H+1, n, n)`. Shim transposes via `.transpose(1, 2, 0)` and unpacks to 3-tuple.

- [ ] **Step 1: Add the shape-preservation test to `tests/test_shim_shape_preservation.py`**

Append:

```python
def test_cholesky_shim_returns_legacy_3tuple_with_legacy_axis(suppress_shim_warnings):
    """Legacy svar.identify_cholesky.cholesky_svar must return
    (point, lower, upper) each shape (n, n, H+1), matching the
    canonical (H+1, n, n) result transposed back."""
    from puremacro.svar.identify_cholesky import cholesky_svar as legacy
    from puremacro.var.identify.cholesky import cholesky_svar as canonical

    Y = _toy_var2(seed=0)
    p, H, B = 1, 4, 50

    leg = legacy(Y, p=p, horizon=H, n_boot=B, ci=0.9, seed=0)
    can = canonical(Y, p=p, horizon=H, n_boot=B, ci=0.9, seed=0)

    assert isinstance(leg, tuple) and len(leg) == 3
    leg_point, leg_lo, leg_hi = leg
    assert leg_point.shape == (2, 2, H + 1)

    # The shim transposes (H+1, n, n) → (n, n, H+1); compare elementwise.
    np.testing.assert_allclose(leg_point, can.irf_point.transpose(1, 2, 0))
    np.testing.assert_allclose(leg_lo, can.irf_lower.transpose(1, 2, 0))
    np.testing.assert_allclose(leg_hi, can.irf_upper.transpose(1, 2, 0))
```

- [ ] **Step 2: Append the shim entry to `SHIMS`**

In `tests/test_deprecation_warnings.py`, append to the `SHIMS` list:

```python
    ("puremacro.svar.identify_cholesky", "puremacro.var.identify.cholesky"),
```

- [ ] **Step 3: Run both new tests to confirm they fail**

```bash
python -m pytest tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py -v
```

Expected: the new cholesky test cases fail (legacy still returns its own raw tuple, no warning emitted).

- [ ] **Step 4: Rewrite `puremacro/svar/identify_cholesky.py` as a shim**

Replace entire file content:

```python
"""DEPRECATED — use puremacro.var.identify.cholesky.

Phase-2 shim (0.42.0). Removal target: 0.43.0.

Re-exports the canonical cholesky_svar but unpacks CholeskySVARResult
into the legacy 3-tuple (point, lower, upper) with the legacy
(n, n, H+1) IRF axis convention preserved for back-compat with
notebooks/R1_methods/R1_01_svar_menu.ipynb.
"""
import warnings as _w

from puremacro.var.identify.cholesky import cholesky_svar as _cholesky_svar

_w.warn(
    "puremacro.svar.identify_cholesky is deprecated since 0.42.0; "
    "use puremacro.var.identify.cholesky. Removal target: 0.43.0.",
    DeprecationWarning,
    stacklevel=2,
)


def cholesky_svar(*args, **kwargs):
    """Legacy 3-tuple wrapper. See puremacro.var.identify.cholesky.cholesky_svar."""
    r = _cholesky_svar(*args, **kwargs)
    def _T(a): return a.transpose(1, 2, 0)
    return _T(r.irf_point), _T(r.irf_lower), _T(r.irf_upper)
```

- [ ] **Step 5: Run both tests to confirm they pass**

```bash
python -m pytest tests/test_deprecation_warnings.py::test_shim_emits_one_deprecation_warning tests/test_shim_shape_preservation.py::test_cholesky_shim_returns_legacy_3tuple_with_legacy_axis -v
```

Expected: both pass.

- [ ] **Step 6: Run the broader var suite**

```bash
python -m pytest tests/test_var/ tests/test_cholesky_shocks.py tests/test_robustness.py -q
```

Expected: no regressions.

- [ ] **Step 7: Commit**

```bash
git add puremacro/svar/identify_cholesky.py tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py
git commit -m "$(cat <<'EOF'
refactor(svar): identify_cholesky → shim of var.identify.cholesky

Phase-2 shim. Transposes canonical (H+1,n,n) result back to legacy
(n,n,H+1) 3-tuple. Removal target: 0.43.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3c: svar/identify_bq.py → shim of var/identify/bq

**Files:**
- Modify: `puremacro/svar/identify_bq.py`
- Modify: `tests/test_deprecation_warnings.py`
- Modify: `tests/test_shim_shape_preservation.py`

**Key translation:** Same pattern as cholesky — legacy returns `(point, lower, upper)` shape `(n, n, H+1)`; canonical returns `BQSVARResult` with `(H+1, n, n)`. Shim transposes.

Additional legacy parameter: `permanent_var_idx: int = 0`. Pass-through.

- [ ] **Step 1: Add the shape-preservation test**

Append to `tests/test_shim_shape_preservation.py`:

```python
def test_bq_shim_returns_legacy_3tuple_with_legacy_axis(suppress_shim_warnings):
    """svar.identify_bq.bq_svar legacy contract: 3-tuple shape (n,n,H+1)."""
    from puremacro.svar.identify_bq import bq_svar as legacy
    from puremacro.var.identify.bq import bq_svar as canonical

    Y = _toy_var2(seed=0)
    p, H, B = 1, 4, 50

    leg = legacy(Y, p=p, horizon=H, n_boot=B, ci=0.9, seed=0,
                 permanent_var_idx=0)
    can = canonical(Y, p=p, horizon=H, n_boot=B, ci=0.9, seed=0,
                    permanent_var_idx=0)

    assert isinstance(leg, tuple) and len(leg) == 3
    leg_point, leg_lo, leg_hi = leg
    assert leg_point.shape == (2, 2, H + 1)

    np.testing.assert_allclose(leg_point, can.irf_point.transpose(1, 2, 0))
    np.testing.assert_allclose(leg_lo, can.irf_lower.transpose(1, 2, 0))
    np.testing.assert_allclose(leg_hi, can.irf_upper.transpose(1, 2, 0))
```

- [ ] **Step 2: Append the shim entry to `SHIMS`**

```python
    ("puremacro.svar.identify_bq", "puremacro.var.identify.bq"),
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
python -m pytest tests/test_deprecation_warnings.py::test_shim_emits_one_deprecation_warning tests/test_shim_shape_preservation.py::test_bq_shim_returns_legacy_3tuple_with_legacy_axis -v
```

Expected: both fail.

- [ ] **Step 4: Rewrite `puremacro/svar/identify_bq.py` as a shim**

```python
"""DEPRECATED — use puremacro.var.identify.bq.

Phase-2 shim (0.42.0). Removal target: 0.43.0.

Re-exports the canonical bq_svar but unpacks BQSVARResult into the
legacy 3-tuple (point, lower, upper) with the legacy (n, n, H+1)
IRF axis convention preserved for back-compat with
notebooks/R1_methods/R1_01_svar_menu.ipynb.
"""
import warnings as _w

from puremacro.var.identify.bq import bq_svar as _bq_svar

_w.warn(
    "puremacro.svar.identify_bq is deprecated since 0.42.0; "
    "use puremacro.var.identify.bq. Removal target: 0.43.0.",
    DeprecationWarning,
    stacklevel=2,
)


def bq_svar(*args, **kwargs):
    """Legacy 3-tuple wrapper. See puremacro.var.identify.bq.bq_svar."""
    r = _bq_svar(*args, **kwargs)
    def _T(a): return a.transpose(1, 2, 0)
    return _T(r.irf_point), _T(r.irf_lower), _T(r.irf_upper)
```

- [ ] **Step 5: Run both tests to confirm they pass**

```bash
python -m pytest tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py -v
```

Expected: all populated cases pass.

- [ ] **Step 6: Commit**

```bash
git add puremacro/svar/identify_bq.py tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py
git commit -m "$(cat <<'EOF'
refactor(svar): identify_bq → shim of var.identify.bq

Phase-2 shim. Transposes (H+1,n,n) → (n,n,H+1) and unpacks
BQSVARResult to 3-tuple. Removal target: 0.43.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3d: svar/identify_sign.py → shim of var/identify/sign

**Files:**
- Modify: `puremacro/svar/identify_sign.py`
- Modify: `tests/test_deprecation_warnings.py`
- Modify: `tests/test_shim_shape_preservation.py`

**Key translation:** Legacy returns `(median, lower, upper)` 3-tuple shape `(n, n, H+1)`. Canonical returns `SignRestrictionResult` with `irf_median/lower/upper` shape `(H+1, n, n)`. Shim transposes + unpacks. Note: canonical uses `safe_cholesky` and may have different numerical behaviour on degenerate Σ than legacy's bare `np.linalg.cholesky`; tolerate via `rtol=1e-5`.

- [ ] **Step 1: Add the shape-preservation test**

Append:

```python
def test_sign_shim_returns_legacy_3tuple_with_legacy_axis(suppress_shim_warnings):
    """svar.identify_sign.sign_restriction_svar legacy contract."""
    from puremacro.svar.identify_sign import sign_restriction_svar as legacy
    from puremacro.var.identify.sign import sign_restriction_svar as canonical

    Y = _toy_var2(seed=0)
    p, H = 1, 4
    restrictions = {0: [+1, +1]}

    leg = legacy(Y, p=p, horizon=H, restrictions=restrictions,
                 n_draws=200, ci=0.9, seed=0)
    can = canonical(Y, p=p, horizon=H, restrictions=restrictions,
                    n_draws=200, ci=0.9, seed=0)

    assert isinstance(leg, tuple) and len(leg) == 3
    leg_med, leg_lo, leg_hi = leg
    assert leg_med.shape == (2, 2, H + 1)

    # Canonical uses safe_cholesky; tolerate small numeric drift.
    np.testing.assert_allclose(leg_med, can.irf_median.transpose(1, 2, 0), rtol=1e-5)
    np.testing.assert_allclose(leg_lo, can.irf_lower.transpose(1, 2, 0), rtol=1e-5)
    np.testing.assert_allclose(leg_hi, can.irf_upper.transpose(1, 2, 0), rtol=1e-5)
```

- [ ] **Step 2: Append the shim entry to `SHIMS`**

```python
    ("puremacro.svar.identify_sign", "puremacro.var.identify.sign"),
```

- [ ] **Step 3: Confirm tests fail, then rewrite the shim**

Run:
```bash
python -m pytest tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py -v
```

Replace `puremacro/svar/identify_sign.py` with:

```python
"""DEPRECATED — use puremacro.var.identify.sign.

Phase-2 shim (0.42.0). Removal target: 0.43.0.
"""
import warnings as _w

from puremacro.var.identify.sign import sign_restriction_svar as _sign

_w.warn(
    "puremacro.svar.identify_sign is deprecated since 0.42.0; "
    "use puremacro.var.identify.sign. Removal target: 0.43.0.",
    DeprecationWarning,
    stacklevel=2,
)


def sign_restriction_svar(*args, **kwargs):
    """Legacy 3-tuple wrapper. Returns (median, lower, upper) shape (n,n,H+1)."""
    r = _sign(*args, **kwargs)
    def _T(a): return a.transpose(1, 2, 0)
    return _T(r.irf_median), _T(r.irf_lower), _T(r.irf_upper)
```

- [ ] **Step 4: Re-run tests and commit**

```bash
python -m pytest tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py -v
git add puremacro/svar/identify_sign.py tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py
git commit -m "$(cat <<'EOF'
refactor(svar): identify_sign → shim of var.identify.sign

Phase-2 shim. Removal target: 0.43.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3e: svar/identify_proxy.py → shim of var/identify/proxy

**Key translation:** Legacy returns `(point, lower, upper)` 3-tuple shape `(n, n, H+1)`. **Canonical also uses `(n, n, H+1)` axis** (per `ProxySVARResult.irf_point` docstring, because it routes through `wild_bootstrap_var`). **No transpose needed** — just unpack the dataclass's first three fields.

The canonical `ProxySVARResult` has extra fields (`B`, `first_stage_F`) that legacy didn't expose. Legacy callers won't see these, which is fine.

**Files:** as for 3d.

- [ ] **Step 1: Add the shape-preservation test**

```python
def test_proxy_shim_returns_legacy_3tuple_no_transpose(suppress_shim_warnings):
    """svar.identify_proxy.proxy_svar — both legacy and canonical use (n,n,H+1)."""
    from puremacro.svar.identify_proxy import proxy_svar as legacy
    from puremacro.var.identify.proxy import proxy_svar as canonical

    rng = np.random.default_rng(0)
    Y = _toy_var2(seed=0)
    z = rng.standard_normal(len(Y))
    p, H, B = 1, 4, 50

    leg = legacy(Y, p=p, horizon=H, instrument_series=z,
                 n_boot=B, ci=0.9, seed=0)
    can = canonical(Y, p=p, horizon=H, instrument_series=z,
                    n_boot=B, ci=0.9, seed=0)

    assert isinstance(leg, tuple) and len(leg) == 3
    leg_point, leg_lo, leg_hi = leg
    assert leg_point.shape == (2, 2, H + 1)
    # No transpose: both ProxySVARResult and legacy use (n,n,H+1).
    np.testing.assert_allclose(leg_point, can.irf_point)
    np.testing.assert_allclose(leg_lo, can.irf_lower)
    np.testing.assert_allclose(leg_hi, can.irf_upper)
```

- [ ] **Step 2: Append to `SHIMS`**

```python
    ("puremacro.svar.identify_proxy", "puremacro.var.identify.proxy"),
```

- [ ] **Step 3: Rewrite shim**

```python
"""DEPRECATED — use puremacro.var.identify.proxy.

Phase-2 shim (0.42.0). Removal target: 0.43.0.

Both legacy and canonical use (n, n, H+1) axis convention, so this
shim only unpacks the dataclass to a 3-tuple — no transpose.
"""
import warnings as _w

from puremacro.var.identify.proxy import proxy_svar as _proxy

_w.warn(
    "puremacro.svar.identify_proxy is deprecated since 0.42.0; "
    "use puremacro.var.identify.proxy. Removal target: 0.43.0.",
    DeprecationWarning,
    stacklevel=2,
)


def proxy_svar(*args, **kwargs):
    """Legacy 3-tuple wrapper."""
    r = _proxy(*args, **kwargs)
    return r.irf_point, r.irf_lower, r.irf_upper
```

- [ ] **Step 4: Run tests + commit**

```bash
python -m pytest tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py -v
git add puremacro/svar/identify_proxy.py tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py
git commit -m "$(cat <<'EOF'
refactor(svar): identify_proxy → shim of var.identify.proxy

Phase-2 shim; no transpose (both use (n,n,H+1)). Removal target: 0.43.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3f: svar/identify_heteroskedasticity.py → shim of var/identify/hetero

**Files:**
- Modify: `puremacro/svar/identify_heteroskedasticity.py`
- Modify: `tests/test_deprecation_warnings.py`
- Modify: `tests/test_shim_shape_preservation.py`

**Key complication:** Both legacy and canonical define a class named `HeteroResult`, but with different field axis conventions:
- Legacy `svar/identify_heteroskedasticity.HeteroResult`: `point`, `lower`, `upper` shape `(n, n, H+1)`.
- Canonical `var/identify/hetero.HeteroResult`: shape `(H+1, n, n)`.
- Both share `irfs`/`fevd` shape `(H+1, n, n)`.

Plus: external callers may do `from puremacro.svar.identify_heteroskedasticity import HeteroResult` and pattern-match the legacy shape. The shim must re-export a `HeteroResult` class with the legacy field axes.

- [ ] **Step 1: Add the shape-preservation test**

```python
def test_hetero_shim_returns_legacy_hetero_result(suppress_shim_warnings):
    """svar.identify_heteroskedasticity.rigobon_svar legacy contract."""
    from puremacro.svar.identify_heteroskedasticity import (
        rigobon_svar as legacy,
        HeteroResult as LegacyHetero,
    )
    from puremacro.var.identify.hetero import rigobon_svar as canonical

    Y = _toy_var2(seed=0)
    # Synthetic two-regime indicator: first half regime 0, second half regime 1.
    T = len(Y)
    p = 1
    regime = np.concatenate([np.zeros(T // 2 - p), np.ones(T - T // 2)])
    H = 4
    B = 20

    leg = legacy(Y, p=p, horizon=H, regime_indicator=regime,
                 n_boot=B, ci=0.9, seed=0)
    can = canonical(Y, p=p, horizon=H, regime_indicator=regime,
                    n_boot=B, ci=0.9, seed=0)

    assert isinstance(leg, LegacyHetero)
    assert leg.point.shape == (2, 2, H + 1)         # legacy axis
    assert can.point.shape == (H + 1, 2, 2)         # canonical axis
    np.testing.assert_allclose(leg.point, can.point.transpose(1, 2, 0))
    if can.lower is not None:
        np.testing.assert_allclose(leg.lower, can.lower.transpose(1, 2, 0))
        np.testing.assert_allclose(leg.upper, can.upper.transpose(1, 2, 0))
    # irfs and fevd share (H+1, n, n) in both:
    np.testing.assert_allclose(leg.irfs, can.irfs)
    np.testing.assert_allclose(leg.fevd, can.fevd)
```

- [ ] **Step 2: Append to `SHIMS`**

```python
    ("puremacro.svar.identify_heteroskedasticity", "puremacro.var.identify.hetero"),
```

- [ ] **Step 3: Rewrite the shim**

```python
"""DEPRECATED — use puremacro.var.identify.hetero.

Phase-2 shim (0.42.0). Removal target: 0.43.0.

Preserves the legacy HeteroResult dataclass shape (point/lower/upper
axes are (n, n, H+1)) by translating the canonical
(H+1, n, n) result back. The irfs/fevd fields share (H+1, n, n) in
both versions — no transpose applied to those.
"""
import warnings as _w
from dataclasses import dataclass
from typing import Optional

import numpy as np

from puremacro.var.identify.hetero import rigobon_svar as _rigobon

_w.warn(
    "puremacro.svar.identify_heteroskedasticity is deprecated since 0.42.0; "
    "use puremacro.var.identify.hetero. Removal target: 0.43.0.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class HeteroResult:
    """Legacy-axis Rigobon result. See puremacro.var.identify.hetero.HeteroResult."""

    B: np.ndarray
    variance_ratios: np.ndarray
    irfs: np.ndarray          # (H+1, n, n) — same as canonical
    fevd: np.ndarray          # (H+1, n, n) — same as canonical
    lower: Optional[np.ndarray]   # (n, n, H+1) — legacy convention
    upper: Optional[np.ndarray]   # (n, n, H+1)
    point: np.ndarray             # (n, n, H+1)


def rigobon_svar(*args, **kwargs):
    """Legacy-shaped wrapper."""
    r = _rigobon(*args, **kwargs)
    def _T(a): return None if a is None else a.transpose(1, 2, 0)
    return HeteroResult(
        B=r.B,
        variance_ratios=r.variance_ratios,
        irfs=r.irfs,
        fevd=r.fevd,
        lower=_T(r.lower),
        upper=_T(r.upper),
        point=_T(r.point),
    )
```

- [ ] **Step 4: Run tests + commit**

```bash
python -m pytest tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py -v
git add puremacro/svar/identify_heteroskedasticity.py tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py
git commit -m "$(cat <<'EOF'
refactor(svar): identify_heteroskedasticity → shim of var.identify.hetero

Phase-2 shim. Preserves legacy HeteroResult axis convention for
point/lower/upper while delegating compute to canonical.
Removal target: 0.43.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Phase-2.5 banner on panel_svar.py and identify_maxshare.py

**Files:**
- Modify: `puremacro/svar/panel_svar.py` (banner at top of docstring)
- Modify: `puremacro/svar/identify_maxshare.py` (banner)
- Modify: `ARCHITECTURE.md` (Phase-2.5 section)

**Why:** These two legacy files have no equivalent canonical home with the same surface. They stay alive at 0.42.0 with a banner documenting their Phase-2.5 deferral.

- [ ] **Step 1: Prepend banner to `puremacro/svar/panel_svar.py`**

At the very top of the file (before any existing docstring), prepend:

```python
# PHASE-2.5 DEFERRAL — 2026-05-17
# No canonical home in puremacro.var.identify yet. When a panel-SVAR
# estimator lands under var/identify/panel.py, this file will become a
# DeprecationWarning shim and retire alongside the rest of the legacy
# svar/* surface at 0.43.0.
```

- [ ] **Step 2: Prepend banner to `puremacro/svar/identify_maxshare.py`**

Same banner:

```python
# PHASE-2.5 DEFERRAL — 2026-05-17
# This file ships its own MaxShareResult + bootstrap pipeline; the
# canonical puremacro.var.identify.maxshare currently exports only
# the B0 identification step. Promote the canonical to full pipeline
# parity before turning this file into a shim. Retirement targeted
# alongside panel_svar.py at 0.43.0.
```

- [ ] **Step 3: Run tests to confirm no regression**

```bash
python -m pytest tests/ -q
```

Expected: same baseline count.

- [ ] **Step 4: Commit**

```bash
git add puremacro/svar/panel_svar.py puremacro/svar/identify_maxshare.py
git commit -m "$(cat <<'EOF'
docs(svar): banner panel_svar + identify_maxshare as Phase-2.5

No canonical sibling with equivalent surface; defer to 0.43.0 follow-up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Shim each lp/lp_*.py file that passes Task 0 parity audit

For each `lp_*` file that recorded `PARITY-OK` in Task 0, follow the shim pattern below. For each one that recorded `DEFER-2.5`, follow the banner pattern (mirror Task 4).

**Why:** Files where the legacy is a thin wrapper of the canonical can be safely shimmed. Files where they're independent implementations of the same idea need parity work that's out of scope here.

### Task 5a: lp/lp_jorda.py (template — copy for each PARITY-OK file)

**Files:**
- Modify: `puremacro/lp/lp_jorda.py`
- Modify: `tests/test_deprecation_warnings.py`

**Note:** `lp_jorda` defines its own `LPResult` dataclass. Confirm in the parity audit whether the canonical `lp/jorda.py` shares this exact dataclass or differs. If different (e.g. different field names or a `LPHACResult` instead), the shim must either re-export the canonical type with `LPResult` as an alias, OR pattern-match.

- [ ] **Step 1: Diff legacy vs canonical LPResult**

```bash
diff <(grep -nA10 "class LPResult" puremacro/lp/lp_jorda.py) \
     <(grep -nA20 "class LP" puremacro/lp/jorda.py)
```

Record whether the canonical exposes a `LPResult` with the same field set. If yes → Step 2. If no → drop to Task 5b's "DEFER-2.5" path.

- [ ] **Step 2: Append the shim entry to `SHIMS`**

In `tests/test_deprecation_warnings.py`, append:

```python
    ("puremacro.lp.lp_jorda", "puremacro.lp.jorda"),
```

- [ ] **Step 3: Run tests to confirm failure**

```bash
python -m pytest tests/test_deprecation_warnings.py -v
```

Expected: FAIL — legacy doesn't emit warning yet.

- [ ] **Step 4: Rewrite `puremacro/lp/lp_jorda.py` as a shim**

```python
"""DEPRECATED — use puremacro.lp.jorda.

Phase-2 shim (0.42.0). Removal target: 0.43.0.

The canonical puremacro.lp.jorda re-exports the same public surface
(LPResult and lp_hac).
"""
import warnings as _w

from puremacro.lp.jorda import *  # noqa: F401,F403
# Explicit re-export so introspection still works:
from puremacro.lp.jorda import lp_hac, LPResult  # noqa: F401

_w.warn(
    "puremacro.lp.lp_jorda is deprecated since 0.42.0; "
    "use puremacro.lp.jorda. Removal target: 0.43.0.",
    DeprecationWarning,
    stacklevel=2,
)
```

If the parity audit found a renamed dataclass (e.g. canonical has `LPHACResult`, legacy has `LPResult`), instead use:

```python
from puremacro.lp.jorda import lp_hac
from puremacro.lp.jorda import LPHACResult as LPResult  # legacy alias
```

- [ ] **Step 5: Run the deprecation test and the parity tests**

```bash
python -m pytest tests/test_deprecation_warnings.py tests/test_lp/ -v
```

Expected: deprecation test passes for `lp.lp_jorda`; existing `test_jorda_parity.py` still passes.

- [ ] **Step 6: Commit**

```bash
git add puremacro/lp/lp_jorda.py tests/test_deprecation_warnings.py
git commit -m "$(cat <<'EOF'
refactor(lp): lp_jorda → shim of lp.jorda

Phase-2 shim. Removal target: 0.43.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Tasks 5b–5h: lp_iv, lp_panel, lp_panel_dk, lp_state_dep, lp_smooth, lp_garch_state, lp_garch_in_mean

Repeat Task 5a's six-step pattern for each remaining file that passed the Task 0 parity audit.

**The shim template** (substitute `LEGACY` and `CANONICAL`):

```python
"""DEPRECATED — use puremacro.lp.CANONICAL.

Phase-2 shim (0.42.0). Removal target: 0.43.0.
"""
import warnings as _w

from puremacro.lp.CANONICAL import *  # noqa: F401,F403

_w.warn(
    "puremacro.lp.LEGACY is deprecated since 0.42.0; "
    "use puremacro.lp.CANONICAL. Removal target: 0.43.0.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Per-file specifics:**

| Legacy | Canonical | Sketch |
|---|---|---|
| `lp/lp_iv.py` | `lp/iv.py` | Pure-DataFrame return; `*`-import shim. |
| `lp/lp_panel.py` | `lp/panel.py` | Legacy uses `linearmodels.PanelOLS` lazily; canonical is pure-numpy. Confirm parity audit before shimming — likely PARITY-OK because `test_panel_parity.py` already pins them together. |
| `lp/lp_panel_dk.py` | `lp/panel_dk.py` | Same story. |
| `lp/lp_state_dep.py` | `lp/state_dep.py` | Legacy imports `inference.legacy.block_bootstrap`; verify canonical does too or has equivalent. |
| `lp/lp_smooth.py` | `lp/smooth.py` | Same. |
| `lp/lp_garch_state.py` | `lp/garch_state.py` | Lazy `statsmodels`; preserve in shim docstring. |
| `lp/lp_garch_in_mean.py` | `lp/garch_in_mean.py` | Same. |

**For each:** apply the 6-step Task 5a recipe (audit diff → SHIMS entry → fail test → rewrite as shim → re-run tests → commit). If any file fails parity, banner-defer (Task 4 pattern) instead and note in CHANGELOG that it stays at 0.42.0.

### Task 5i: lp/garch_utils.py — helper file, no canonical sibling

**Files:** `puremacro/lp/garch_utils.py`.

**Why:** `garch_utils.py` is a helper module imported by `lp/lp_garch_state.py` and `lp/lp_garch_in_mean.py`. No canonical sibling exists. Two reasonable choices:

- (i) Promote to `puremacro/lp/_garch_utils.py` (private helper); update both consumers to import from the new path.
- (ii) Leave at `garch_utils.py` (no leading underscore) since callers reach into it; just add a Phase-2.5 banner.

Default to (ii) for 0.42.0 — no churn beyond a banner.

- [ ] **Step 1: Prepend banner**

```python
# PHASE-2.5 NOTE — 2026-05-17
# This module is a private helper for the LP-GARCH variants. It has no
# canonical sibling under lp/. The leading-underscore relocation
# (lp/_garch_utils.py) is deferred; downstream callers reach into it
# from notebook code paths.
```

- [ ] **Step 2: No code change. Confirm no test regression and commit.**

```bash
python -m pytest tests/ -q
git add puremacro/lp/garch_utils.py
git commit -m "docs(lp): banner garch_utils as Phase-2.5 helper

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Update teaching/bq_canonical.py to canonical

**Files:**
- Modify: `puremacro/teaching/bq_canonical.py`

**Why:** The only in-repo non-test non-test caller still on the legacy path. Switching the import means callers of `puremacro.teaching.bq_canonical` get clean canonical behaviour without a DeprecationWarning surfacing in the teaching surface.

- [ ] **Step 1: Read the current import lines and tuple unpacks**

```bash
grep -n "svar\|bq_svar\|point\|hi\|lo" puremacro/teaching/bq_canonical.py | head -20
```

Confirm the two tuple-unpack sites at lines 67 and 99 (per earlier inspection):

```python
point, lo, hi = bq_svar(...)
```

- [ ] **Step 2: Modify the file**

Find:
```python
from puremacro.svar.identify_bq import bq_svar
```

Replace with:
```python
from puremacro.var.identify.bq import bq_svar as _bq_svar_canonical


def bq_svar(*args, **kwargs):
    """Local legacy-axis adapter (transposes canonical BQSVARResult back to
    (point, lower, upper) shape (n, n, H+1) so the rest of this teaching
    module's plotting code keeps working unchanged)."""
    r = _bq_svar_canonical(*args, **kwargs)
    def _T(a): return a.transpose(1, 2, 0)
    return _T(r.irf_point), _T(r.irf_lower), _T(r.irf_upper)
```

This keeps the tuple-unpack sites at lines 67 and 99 working as-is. (We don't switch them to attribute access because the docstring/comments around them refer to "point/lo/hi" and rewriting them is teaching-surface churn beyond the scope of this consolidation.)

- [ ] **Step 3: Run the teaching tests if any (and the broader test suite)**

```bash
python -m pytest tests/ -q -W "error::DeprecationWarning" --ignore=tests/test_deprecation_warnings.py
```

Expected: green. **This is the strongest gate** — it asserts no remaining in-repo caller imports a shim. If it red-fails on a file we haven't touched, that file also needs migration; loop back.

- [ ] **Step 4: Commit**

```bash
git add puremacro/teaching/bq_canonical.py
git commit -m "$(cat <<'EOF'
refactor(teaching): bq_canonical pinned to var.identify.bq

Routes through canonical (BQSVARResult) and applies the (H+1,n,n) →
(n,n,H+1) axis transpose locally so the existing teaching-surface
tuple-unpack call sites stay unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: inference/legacy/ retirement notes

**Files:**
- Modify: `puremacro/inference/legacy/bootstrap.py`
- Modify: `puremacro/inference/legacy/wild_bootstrap.py`
- Modify: `puremacro/inference/legacy/block_bootstrap.py`
- Modify: `puremacro/inference/legacy/weak_iv.py`

**Why:** These four files remain load-bearing at 0.42.0 (canonical `var/identify/cholesky.py`, `var/identify/bq.py`, `var/identify/proxy.py`, etc. still import from them). They retire at 0.43.0 alongside the shim layer. A retirement note at the top prevents premature deletion by a future contributor.

- [ ] **Step 1: For each file, prepend a one-line retirement note**

At the top of each file (before any existing docstring or import), prepend:

```python
# RETIREMENT NOTE — 2026-05-17
# Still imported by canonical var/identify/* and by the Phase-2 shims
# in svar/* and lp/lp_*.py at 0.42.0. Deletion target: 0.43.0, paired
# with the shim cutover and any required canonical migration.
```

- [ ] **Step 2: Run the test suite**

```bash
python -m pytest tests/ -q
```

Expected: green.

- [ ] **Step 3: Commit**

```bash
git add puremacro/inference/legacy/bootstrap.py puremacro/inference/legacy/wild_bootstrap.py puremacro/inference/legacy/block_bootstrap.py puremacro/inference/legacy/weak_iv.py
git commit -m "$(cat <<'EOF'
docs(inference): retirement notes on legacy/* files

These four files are still load-bearing for canonical var/identify/*
at 0.42.0. Removal scheduled with the shim cutover at 0.43.0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update ARCHITECTURE.md and CHANGELOG.md

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update `ARCHITECTURE.md` § "Known consolidation candidates"**

Replace the section content with the post-0.42.0 reality. Specifically rewrite the existing three bullets to:

```markdown
- **`svar/` → `var/identify/`** — done as shim layer at 0.42.0. The six files
  `identify_cholesky`, `identify_bq`, `identify_sign`, `identify_proxy`,
  `identify_heteroskedasticity`, and `estimate_var` are now thin
  DeprecationWarning shims. Deletion targeted for 0.43.0 with the next
  paper-figure refresh.
- **`lp/lp_*.py` → `lp/*.py`** — done as shim layer at 0.42.0 for the
  files that passed parity audit; the rest carry a Phase-2.5 banner.
  See `docs/plans/_phase2_audit_notes.md` for the per-file disposition.
  Deletion targeted for 0.43.0.
- **Result-object wrapping** — `VarEstimateResult` lands at 0.42.0.
  `panel_svar.py` and `identify_maxshare.py` are Phase-2.5 candidates
  (no equivalent canonical surface yet).
```

Also update the stability-tier table: change the "Stable but redundant" rows for `svar/*` and `lp/lp_*.py` to "Shim (DeprecationWarning) since 0.42.0; removal at 0.43.0".

- [ ] **Step 2: Prepend the 0.42.0 entry to `CHANGELOG.md`**

Add at the top of the file (before the current first entry):

```markdown
## 0.42.0 — 2026-05-17

Phase-2 consolidation: legacy `svar/*` and the parity-passing `lp/lp_*.py`
files are now thin DeprecationWarning shims of the canonical
`var/identify/*` and `lp/*.py` paths. No behaviour change for callers;
removal target 0.43.0 (next paper-figure refresh).

### Added
- `puremacro.var.VarEstimateResult` frozen dataclass; `var.estimate_var`
  now returns this dataclass (iterable to preserve 5-tuple unpack).

### Deprecated
- `puremacro.svar.estimate_var` → use `puremacro.var.estimate.estimate_var`.
- `puremacro.svar.identify_{cholesky,bq,sign,proxy,heteroskedasticity}` →
  use `puremacro.var.identify.{cholesky,bq,sign,proxy,hetero}`.
- `puremacro.lp.lp_{...}` (per parity audit) → use the prefix-free
  `puremacro.lp.*` siblings.
- Each emits exactly one `DeprecationWarning` on import. Removal target: 0.43.0.

### Internal
- `teaching.bq_canonical` re-pinned to canonical, applying the
  `(H+1,n,n)` → `(n,n,H+1)` axis transpose locally so its tuple-unpack
  call sites are unchanged.
- `inference/legacy/{bootstrap,wild_bootstrap,block_bootstrap,weak_iv}.py`
  marked with a 0.43.0 retirement note; still load-bearing for canonical
  `var/identify/*` at 0.42.0.
- `tests/test_deprecation_warnings.py` and `tests/test_shim_shape_preservation.py`
  gate the shim contract end-to-end.

### Not affected
- `svar/panel_svar.py` — banner only (Phase-2.5 candidate; no canonical
  `var/identify/panel.py` yet).
- `svar/identify_maxshare.py` — banner only (canonical maxshare is
  identification-only; legacy ships full pipeline).
- `lp/garch_utils.py` — banner only (private helper, no canonical
  sibling).
- Notebook deep imports — keep working unchanged via the shim layer;
  warnings surface on next execution.
```

- [ ] **Step 3: Run all tests to confirm no doc-only regression**

```bash
python -m pytest tests/ -q
```

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(phase2): ARCHITECTURE + CHANGELOG for 0.42.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Version bump to 0.42.0

**Files:**
- Modify: `puremacro/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_import.py`

- [ ] **Step 1: Bump `puremacro/__init__.py`**

Edit:
```python
__version__ = "0.41.0"
```
to:
```python
__version__ = "0.42.0"
```

- [ ] **Step 2: Bump `pyproject.toml`**

Edit:
```toml
version = "0.41.0"
```
to:
```toml
version = "0.42.0"
```

(Confirm the actual line via `grep -n 'version' pyproject.toml`; the current value may be different from 0.41.0 in the spec snapshot.)

- [ ] **Step 3: Update `tests/test_import.py`**

```bash
grep -n "__version__\|0\\.4" tests/test_import.py
```

Edit any assertion of the old version to `"0.42.0"`.

- [ ] **Step 4: Run the import test**

```bash
python -m pytest tests/test_import.py -v
```

Expected: passes against `0.42.0`.

- [ ] **Step 5: Regenerate public-API snapshot**

```bash
python - <<'EOF'
import json
import puremacro.var
# Check that VarEstimateResult is in puremacro.var.__all__
print("var __all__:", getattr(puremacro.var, "__all__", "no __all__"))
EOF
```

If `VarEstimateResult` is exported, regenerate the snapshot file. The snapshot script may be implicit — check `tests/test_public_api.py`:

```bash
grep -n "public_api_snapshot" tests/test_public_api.py
```

If a `--update` flag exists, run it; otherwise edit `tests/fixtures/public_api_snapshot.json` by hand to add the new class fields.

- [ ] **Step 6: Run `test_public_api.py`**

```bash
python -m pytest tests/test_public_api.py -v
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add puremacro/__init__.py pyproject.toml tests/test_import.py tests/fixtures/public_api_snapshot.json
git commit -m "$(cat <<'EOF'
chore(release): bump version to 0.42.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final verification gate

**Files:** read-only.

- [ ] **Step 1: Full pytest run**

```bash
python -m pytest tests/ -q
```

Expected: baseline_count + (N_shims + N_shape_tests). Record the new count.

- [ ] **Step 2: Strongest gate — no in-repo caller still on a shim**

```bash
python -m pytest tests/ -q -W "error::DeprecationWarning" \
  --ignore=tests/test_deprecation_warnings.py \
  --ignore=tests/test_shim_shape_preservation.py
```

Expected: green. **If this fails**, the failing test file imports a shim that we missed — fix that file's import and re-run.

- [ ] **Step 3: Pyodide-compat sweep**

```bash
python -m pytest tests/test_pyodide_compat.py -v
```

Expected: 2 passed.

- [ ] **Step 4: One-notebook smoke test**

Locate `notebooks/R1_methods/R1_01_svar_menu.ipynb` (under the sibling directory `../notebooks/R1_methods/`).

Execute end-to-end:

```bash
jupyter nbconvert --to notebook --execute \
  ../notebooks/R1_methods/R1_01_svar_menu.ipynb \
  --output /tmp/R1_01_smoke.ipynb \
  --ExecutePreprocessor.timeout=600
```

Inspect `/tmp/R1_01_smoke.ipynb` for:
1. At least one `DeprecationWarning` in cell outputs naming a `puremacro.svar.*` shim.
2. No other warnings.
3. Cell outputs (figures, tables) byte-identical to the committed `notebooks/R1_methods/R1_01_svar_menu.ipynb` where deterministic.

Per the `feedback_long_nbconvert_no_subagent` memory pin: if this nbconvert takes longer than 5 minutes, run as a background task in the controller, do not delegate to a subagent.

- [ ] **Step 5: Final state check**

```bash
git log --oneline -20
git status
```

Confirm:
- All commits in the Phase-2 chain are present and named per the per-task commit-message templates.
- No staged or unstaged files remain.

---

## Self-review checklist (run before declaring done)

- [ ] **Spec coverage:** every section of `docs/specs/2026-05-17-puremacro-phase2-consolidation-design.md` has a task that implements it. Note: the spec's claim "9 SVAR tests migrated" is superseded by the audit findings — no test migration required. The spec's claim that `experiment.py` / `gar/qar.py` need updating is also superseded — they were already on canonical.

- [ ] **Placeholder scan:** No "TBD", "TODO", "implement later", "fill in details" remain in this plan. The only conditional branches are explicit (Task 5 per-file PARITY-OK / DEFER-2.5).

- [ ] **Type consistency:** `VarEstimateResult` field names match across Tasks 1, 3a, and 9. `HeteroResult` legacy field convention (`point/lower/upper` shape `(n,n,H+1)`) matches across Task 3f. `CholeskySVARResult`/`BQSVARResult`/`SignRestrictionResult` canonical field names (`irf_point/irf_lower/irf_upper`) and `irf_median` (sign) match the actual code verified in the inspection pass.

- [ ] **Signature verification:** Per `feedback_plan_verify_api_signatures` memory pin, all `puremacro.*` import paths and dataclass field names in this plan were verified against the live code on 2026-05-17 (not transcribed from analogy). Per-file canonical → legacy axis translation:
  - cholesky: transpose YES
  - bq: transpose YES
  - sign: transpose YES (median/lower/upper)
  - proxy: transpose NO (canonical also uses `(n,n,H+1)`)
  - hetero: transpose YES on point/lower/upper only; irfs/fevd unchanged

- [ ] **Notebook hazard:** Plan defers all notebook editing to 0.43.0 per `feedback_notebook_builders_paired` and `feedback_builder_clobbers_outputs` memory pins. The one-notebook smoke test in Task 10 Step 4 reads from disk and writes to `/tmp/`, never overwriting the committed `.ipynb`.

- [ ] **Long nbconvert:** Task 10 Step 4 explicitly cites the `feedback_long_nbconvert_no_subagent` pin and instructs background-execution-in-controller if the run exceeds 5 minutes.
