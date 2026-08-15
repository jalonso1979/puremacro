# puremacro 0.43.0 — Canonical Promotion + Shim Retirement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `panel_svar` + `identify_maxshare` to canonical `var/identify/*`, harmonize the 8 `lp/lp_*.py` signatures (rewriting all callers to canonical kwargs), delete the entire shim layer (svar/* + lp/lp_*.py + inference/legacy/*), and re-execute the body-rewrite notebooks. Ship as **0.43.0**.

**Architecture:** Single-release, four-phase consolidation. Phase A promotes canonical surfaces. Phase B migrates callers (in-repo `tools/` + notebook `.ipynb` files + their paired builders, per the notebooks↔builders pairing memory pin). Phase C is essentially already done — only 4 files in `puremacro/` still import `inference.legacy`, all of which are being removed/shimmed in Phase A anyway. Phase D deletes everything, re-executes the body-rewrite notebooks, and tags 0.43.0.

**Tech Stack:** Python ≥3.10, numpy/scipy/pandas/matplotlib (Pyodide promise), pytest, frozen dataclasses.

**Source spec:** `docs/specs/2026-05-18-puremacro-043-canonical-promotion-design.md` (commit `f77eb57`).

**Pre-execution baselines (captured before Task 0):**
- HEAD: `2f64d7b` (0.42.0 release commit + docstring fix)
- pytest: 1274 passed, 10 failed (orthogonal pre-existing)
- 6 svar shims active; 8 lp/lp_*.py + garch_utils as Phase-2.5 banners
- `puremacro.svar` package: 8 files
- `puremacro.lp.lp_*`: 8 files + garch_utils
- `puremacro.inference.legacy`: 10 files (4 distinct + 6 byte-identical shims)

---

## File structure (post-0.43.0 target state)

**New files:**
- `puremacro/var/identify/panel.py` — `mean_group_svar` + helpers (ported from `svar/panel_svar.py`).
- `puremacro/var/identify/_results.py` — extend with `PanelSVARResult`, `MaxShareResult` frozen dataclasses.

**Modified files (substantive):**
- `puremacro/var/identify/maxshare.py` — extend with `identify_maxshare(...)` full pipeline.
- `puremacro/var/identify/__init__.py` — re-export new names.
- `puremacro/lp/panel.py` — port `lp_panel_regime_interaction` from legacy.
- `puremacro/lp/state_dep.py` — port `lp_smooth_transition_irf` from legacy.
- `puremacro/teaching/bq_canonical.py` — remove local adapter; switch call sites to `BQSVARResult` attribute access.
- `puremacro/inference/bootstrap.py` — potentially promote `residual_bootstrap_var` from legacy if any canonical caller still needs it (audit decides).
- Every `tools/run_*.py`, `tools/build_*.py`, `tools/make_notebook_*.py` that imports a legacy path — kwargs rewrite (LP) and import-path rewrite (svar).
- Notebook `.ipynb` files: 9 listed, plus any new ones discovered by Task 0's audit.
- `ARCHITECTURE.md`, `CHANGELOG.md`, `puremacro/__init__.py`, `pyproject.toml`, `tests/test_import.py`, `tests/fixtures/public_api_snapshot.json`.

**Deleted files:**
- `puremacro/svar/` — entire directory.
- All 8 `puremacro/lp/lp_*.py` files; `puremacro/lp/garch_utils.py` renamed to `_garch_utils.py`.
- `puremacro/inference/legacy/` — entire directory (after any unique surface promotion).
- `tests/test_deprecation_warnings.py`, `tests/test_shim_shape_preservation.py`.

---

## Task 0: Per-file audit (read-only, results captured in audit notes)

Resolves all the spec's B2 TBD entries + discovers any callers the survey missed. No code changes.

**Files (read-only):** `tools/run_*.py`, `tools/build_*.py`, `tools/make_notebook_*.py`, `notebooks/R1_methods/*.ipynb`, `notebooks/R2_subnational/*.ipynb`, `notebooks/T5_research_lab.ipynb`, `notebooks/T_us_national.ipynb`, `puremacro/svar/panel_svar.py`, `puremacro/svar/identify_maxshare.py`.

- [ ] **Step 1: Confirm baseline pytest count**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

Expected: 1274 passed, 10 failed (orthogonal pre-existing failures from the 0.42.0 baseline).

- [ ] **Step 2: Resolve `src/` mystery**

Check whether `src/svar/` exists as a separate legacy layer or whether the comment in `tools/make_notebook_R1_01.py` is stale:

```bash
ls /Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My\ Drive/MAV/uncertainty_examples/src/ 2>&1
grep -n "src\.svar\|src\.lp\|sys\.path" /Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My\ Drive/MAV/uncertainty_examples/tools/make_notebook_R1_01.py | head -10
```

Two possible outcomes:
- `src/` exists with a parallel svar package → out of 0.43.0 scope, document.
- `src/` doesn't exist; the comment is stale → R1_01 actually uses `puremacro.svar.*` (the shims). Plan proceeds normally.

- [ ] **Step 3: Catalog every legacy caller**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
echo "=== puremacro.svar.* callers ===" > /tmp/043_audit.txt
grep -rln "puremacro\.svar\." tools/ notebooks/ --include="*.py" --include="*.ipynb" 2>/dev/null | grep -v "_archive" >> /tmp/043_audit.txt
echo "" >> /tmp/043_audit.txt
echo "=== puremacro.lp.lp_* callers ===" >> /tmp/043_audit.txt
grep -rln "puremacro\.lp\.lp_" tools/ notebooks/ --include="*.py" --include="*.ipynb" 2>/dev/null | grep -v "_archive" >> /tmp/043_audit.txt
echo "" >> /tmp/043_audit.txt
echo "=== puremacro.inference.legacy callers ===" >> /tmp/043_audit.txt
grep -rln "puremacro\.inference\.legacy" tools/ notebooks/ --include="*.py" --include="*.ipynb" 2>/dev/null | grep -v "_archive" >> /tmp/043_audit.txt
echo "" >> /tmp/043_audit.txt
echo "=== puremacro.regress.lp callers ===" >> /tmp/043_audit.txt
grep -rln "puremacro\.regress\.lp" tools/ notebooks/ --include="*.py" --include="*.ipynb" 2>/dev/null | grep -v "_archive" >> /tmp/043_audit.txt
cat /tmp/043_audit.txt
```

- [ ] **Step 4: Per-notebook classification**

For each of the 9 listed notebooks, classify as **rename-only** or **body-rewrite** based on what imports it uses:

| Notebook | Body-rewrite trigger | Verdict |
|---|---|---|
| R1_methods/R1_01_svar_menu.ipynb | uses svar 3-tuple returns + identify_maxshare's MaxShareResult | RECORD |
| R1_methods/R1_02_lp_menu.ipynb | uses lp_smooth_irf / lp_state_dep_irf dataclasses → DataFrame switch | RECORD |
| R1_methods/R1_03_cross_country.ipynb | depends on its actual imports | RECORD |
| R1_methods/R1_04_dsge_compare.ipynb | depends | RECORD |
| R1_methods/R1_05_publication.ipynb | uses svar 3-tuple + lp_* dataclasses | RECORD |
| R2_subnational/R2_01_panels_and_data.ipynb | panel_lp_dk kwargs (outcome=→y=, etc.) | RECORD |
| R2_subnational/R2_02_lp_iv_bartik.ipynb | lp_iv_irf → lp_iv name + return type | RECORD |
| T5_research_lab.ipynb | depends; NO paired builder | RECORD |
| T_us_national.ipynb | depends; check for paired builder | RECORD |

For each, run:

```bash
jupyter nbconvert --to script "notebooks/<path>" --stdout 2>/dev/null | grep -E "puremacro\.svar|puremacro\.lp\.lp_|puremacro\.inference\.legacy" | head -10
```

Record per-notebook verdict + any non-import legacy usage (e.g. tuple unpacks like `point, lo, hi = bq_svar(...)`).

- [ ] **Step 5: Write audit notes file**

Create `docs/plans/_043_audit_notes.md`:

```markdown
# 0.43.0 audit notes — 2026-05-18

Baseline pytest: <result from Step 1>

## src/ status (Step 2)
<result>

## Legacy callers (Step 3)
### puremacro.svar.*
<list>

### puremacro.lp.lp_*
<list>

### puremacro.inference.legacy
<list>

### puremacro.regress.lp (out of scope but tracked)
<list>

## Notebook classification (Step 4)
| Notebook | Verdict | Paired builder | Notes |
|---|---|---|---|
| R1_01_svar_menu | body-rewrite | tools/make_notebook_R1_01.py | <notes> |
| ... | ... | ... | ... |

## Surprises
<anything unexpected>
```

- [ ] **Step 6: Commit audit notes**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
git add docs/plans/_043_audit_notes.md
git commit -m "$(cat <<'EOF'
docs(0.43.0): baseline + caller audit notes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: PanelSVARResult dataclass + port mean_group_svar

**Files:**
- Modify: `puremacro/var/identify/_results.py` — add `PanelSVARResult` frozen dataclass.
- Create: `puremacro/var/identify/panel.py` — `mean_group_svar` + helpers.
- Modify: `puremacro/var/identify/__init__.py` — re-export.
- Test: `tests/test_var/test_panel_svar.py`.

**Reference signatures (verified 2026-05-18):**

```python
# Legacy svar/panel_svar.py
@dataclass  # mutable
class PanelSVARResult:
    irf_mean: np.ndarray        # (n, n, H+1)
    irf_lo: np.ndarray          # (n, n, H+1)
    irf_hi: np.ndarray          # (n, n, H+1)
    country_irfs: np.ndarray    # (N, n, n, H+1)
    country_ids: list
    identification: str
    p: int
    horizon: int
    ci: float

def mean_group_svar(
    panel_data: dict[str, np.ndarray],
    *,
    p: int,
    horizon: int,
    identification: str = "cholesky",
    ci: float = 0.9,
    seed: int = 0,
    **id_kwargs,
) -> PanelSVARResult:
```

**Canonical convention:** axes `(H+1, n, n)`. We translate.

- [ ] **Step 1: Write the failing test**

Create `tests/test_var/test_panel_svar.py`:

```python
"""Tests for var.identify.panel.mean_group_svar + PanelSVARResult."""
import numpy as np
import pytest


def _toy_panel(n_countries: int = 3, T: int = 100, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    L = np.linalg.cholesky(Sigma)
    panel = {}
    for i in range(n_countries):
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t] = A @ Y[t-1] + L @ rng.standard_normal(2)
        panel[f"C{i}"] = Y
    return panel


def test_panel_svar_result_is_frozen():
    from puremacro.var.identify._results import PanelSVARResult

    H = 4; n = 2; N = 3
    res = PanelSVARResult(
        irf_mean=np.zeros((H + 1, n, n)),
        irf_lower=np.zeros((H + 1, n, n)),
        irf_upper=np.zeros((H + 1, n, n)),
        country_irfs=np.zeros((N, H + 1, n, n)),
        country_ids=["C0", "C1", "C2"],
        identification="cholesky",
        p=1,
        horizon=H,
        ci=0.9,
    )
    assert res.irf_mean.shape == (5, 2, 2)
    with pytest.raises(Exception):
        res.ci = 0.95  # frozen


def test_mean_group_svar_returns_dataclass():
    from puremacro.var.identify.panel import mean_group_svar
    from puremacro.var.identify._results import PanelSVARResult

    panel = _toy_panel()
    res = mean_group_svar(panel, p=1, horizon=4, identification="cholesky",
                          ci=0.9, seed=0)
    assert isinstance(res, PanelSVARResult)
    assert res.irf_mean.shape == (5, 2, 2)  # (H+1, n, n) canonical
    assert res.country_irfs.shape == (3, 5, 2, 2)
    assert set(res.country_ids) == {"C0", "C1", "C2"}
    assert res.identification == "cholesky"


def test_mean_group_svar_summary_smoke():
    from puremacro.var.identify.panel import mean_group_svar

    panel = _toy_panel()
    res = mean_group_svar(panel, p=1, horizon=4, ci=0.9, seed=0)
    s = res.summary()
    assert "Panel SVAR" in s
    assert "3 countries" in s or "N = 3" in s
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_var/test_panel_svar.py -v
```

Expected: 3 FAIL with `ImportError` or `AttributeError`.

- [ ] **Step 3: Add `PanelSVARResult` to `puremacro/var/identify/_results.py`**

Append to the existing file (after the existing dataclasses):

```python
@dataclass(frozen=True)
class PanelSVARResult:
    """Result of :func:`puremacro.var.identify.panel.mean_group_svar`.

    Canova-Ciccarelli mean-group panel SVAR result. Each country
    estimates its own SVAR(p); IRFs are averaged across countries
    (mean-group estimator) with cross-country percentile bands.

    Attributes
    ----------
    irf_mean : ndarray, shape (H+1, n, n)
        Mean-group IRF: simple cross-country average.
    irf_lower : ndarray, shape (H+1, n, n)
        Lower percentile band from cross-country distribution.
    irf_upper : ndarray, shape (H+1, n, n)
        Upper percentile band.
    country_irfs : ndarray, shape (N, H+1, n, n)
        Stacked country-level IRFs.
    country_ids : tuple of str
        Country identifiers in stacking order.
    identification : str
        Identification scheme used.
    p : int
        VAR lag order.
    horizon : int
        IRF horizon H.
    ci : float
        Coverage for percentile bands.
    """
    irf_mean: np.ndarray
    irf_lower: np.ndarray
    irf_upper: np.ndarray
    country_irfs: np.ndarray
    country_ids: tuple
    identification: str
    p: int
    horizon: int
    ci: float

    def summary(self) -> str:
        n = self.irf_mean.shape[1]
        H = self.irf_mean.shape[0] - 1
        N = len(self.country_ids)
        return (
            f"Panel SVAR (mean-group, {self.identification})\n"
            f"  countries (N)     : {N}\n"
            f"  variables (n)     : {n}\n"
            f"  horizon (H)       : {H}\n"
            f"  lag order (p)     : {self.p}\n"
            f"  CI level          : {self.ci:.2f}\n"
        )
```

- [ ] **Step 4: Create `puremacro/var/identify/panel.py`**

Read `puremacro/svar/panel_svar.py` for the full reference implementation, then port:

```python
"""Canova-Ciccarelli (2013) mean-group panel SVAR.

For each country i in the panel, estimate a SVAR(p) and compute
structural IRFs Phi_i^h for h = 0 … H. Aggregate via simple cross-
sectional average (mean-group). Uncertainty bands come from the
cross-country distribution of country-level IRFs.

References
----------
Canova, F. and Ciccarelli, M. (2013). Panel vector autoregressive
    models: a survey. Advances in Econometrics 32, 205-246.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..estimate import estimate_var
from ..irf import irf as compute_irf
from ._results import PanelSVARResult
from .cholesky import cholesky_factor
from .bq import _bq_impact


def _identify_country(Y: np.ndarray, A_list, Sigma, resid, *,
                      identification: str, horizon: int,
                      **id_kwargs) -> np.ndarray:
    """Run one country's identification. Returns IRF shape (H+1, n, n)."""
    if identification == "cholesky":
        B = cholesky_factor(Sigma)
    elif identification == "bq":
        B = _bq_impact(A_list, Sigma,
                       permanent_var_idx=id_kwargs.get("permanent_var_idx", 0))
    else:
        raise ValueError(
            f"Unsupported identification scheme: {identification}. "
            "Supported: 'cholesky', 'bq'. For 'proxy', 'maxshare', "
            "'rigobon', call the canonical var/identify/<scheme> for "
            "each country directly."
        )
    return compute_irf(A_list, B, horizon)


def mean_group_svar(
    panel_data: dict[str, np.ndarray],
    *,
    p: int,
    horizon: int,
    identification: str = "cholesky",
    ci: float = 0.9,
    seed: int = 0,
    **id_kwargs: Any,
) -> PanelSVARResult:
    """Mean-group panel SVAR estimator.

    Parameters
    ----------
    panel_data : dict mapping country_id -> ndarray (T_i, n)
        Per-country time series. All countries must share the same n.
    p : int
        VAR lag order (common across countries).
    horizon : int
        IRF horizon H.
    identification : str, default "cholesky"
        Identification scheme; one of 'cholesky', 'bq'.
    ci : float, default 0.9
        Coverage for cross-country percentile bands.
    seed : int, default 0
        RNG seed (currently unused but reserved for bootstrap extension).
    **id_kwargs
        Extra kwargs for the identification scheme (e.g.
        ``permanent_var_idx`` for BQ).

    Returns
    -------
    PanelSVARResult
    """
    country_ids = tuple(sorted(panel_data.keys()))
    country_irfs = []

    for cid in country_ids:
        Y = panel_data[cid]
        var_res = estimate_var(Y, p)
        A_list, _, Sigma, resid, _ = var_res
        irf = _identify_country(
            Y, A_list, Sigma, resid,
            identification=identification, horizon=horizon, **id_kwargs,
        )
        country_irfs.append(irf)

    country_irfs_arr = np.stack(country_irfs, axis=0)  # (N, H+1, n, n)
    irf_mean = country_irfs_arr.mean(axis=0)
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    irf_lower = np.quantile(country_irfs_arr, lo_q, axis=0)
    irf_upper = np.quantile(country_irfs_arr, hi_q, axis=0)

    return PanelSVARResult(
        irf_mean=irf_mean,
        irf_lower=irf_lower,
        irf_upper=irf_upper,
        country_irfs=country_irfs_arr,
        country_ids=country_ids,
        identification=identification,
        p=p,
        horizon=horizon,
        ci=ci,
    )


__all__ = ["mean_group_svar"]
```

Note: this canonical port intentionally supports only `cholesky` and `bq` (the two non-bootstrap identification schemes). The legacy `panel_svar.py` had proxy/maxshare/rigobon variants but they require bootstrap kwargs that are scheme-specific. Callers needing those should call the per-scheme canonical directly. Document this in the docstring's `Raises` section if any caller is found in Task 0 audit.

- [ ] **Step 5: Re-export from `puremacro/var/identify/__init__.py`**

Read the current `__init__.py`. Add:

```python
from .panel import mean_group_svar
from ._results import PanelSVARResult
```

…and append `"mean_group_svar"`, `"PanelSVARResult"` to `__all__`.

- [ ] **Step 6: Run the new test to confirm it passes**

```bash
python -m pytest tests/test_var/test_panel_svar.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Run broader var suite**

```bash
python -m pytest tests/test_var/ tests/test_cholesky_shocks.py tests/test_robustness.py -q
```

Expected: 78 + 3 = 81 passed.

- [ ] **Step 8: Pyodide compat sweep**

```bash
python -m pytest tests/test_pyodide_compat.py -v
```

Expected: 2 passed. `var/identify/panel.py` introduces no forbidden imports.

- [ ] **Step 9: Commit**

```bash
git add puremacro/var/identify/_results.py puremacro/var/identify/panel.py puremacro/var/identify/__init__.py tests/test_var/test_panel_svar.py
git commit -m "$(cat <<'EOF'
feat(var/identify): mean_group_svar + PanelSVARResult — 0.43.0 prep

Canonical port of svar/panel_svar.py: per-country SVAR with
cross-country percentile bands. Supports cholesky and bq
identification natively; other schemes call canonical
var/identify/<scheme> per-country directly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: MaxShareResult + extend canonical maxshare with full pipeline

**Files:**
- Modify: `puremacro/var/identify/_results.py` — add `MaxShareResult` frozen dataclass.
- Modify: `puremacro/var/identify/maxshare.py` — add `identify_maxshare(...)` full pipeline.
- Modify: `puremacro/var/identify/__init__.py` — re-export.
- Test: `tests/test_var/test_maxshare_pipeline.py`.

**Reference signatures (verified 2026-05-18):**

```python
# Legacy svar/identify_maxshare.py
@dataclass  # mutable
class MaxShareResult:
    B: np.ndarray
    q: np.ndarray
    fev_share_at_target: float
    irfs: np.ndarray          # (H+1, n, n)
    fevd: np.ndarray          # (H+1, n, n)
    max_fev_at: int
    irf_lo: Optional[np.ndarray] = None
    irf_hi: Optional[np.ndarray] = None

def identify_maxshare(
    Y, p=2, target_idx=0, max_fev_at=1, horizon=20,
    n_bootstrap=500, q_lo=16, q_hi=84,
    rng=None,
) -> MaxShareResult:
```

**Canonical extensions:** keep dataclass field order; rename `q_lo`/`q_hi` to a single `ci` (canonical convention). Match canonical axes already in `(H+1, n, n)`.

- [ ] **Step 1: Write failing test**

Create `tests/test_var/test_maxshare_pipeline.py`:

```python
"""Tests for var.identify.maxshare.identify_maxshare full pipeline + MaxShareResult."""
import numpy as np
import pytest


def _toy_var2(seed: int = 0):
    rng = np.random.default_rng(seed)
    T, n = 200, 2
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    L = np.array([[1.0, 0.0], [0.3, 0.9]])
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    return Y


def test_maxshare_result_is_frozen():
    from puremacro.var.identify._results import MaxShareResult

    H = 4; n = 2
    res = MaxShareResult(
        B=np.eye(n),
        q=np.array([1.0, 0.0]),
        fev_share_at_target=0.85,
        irfs=np.zeros((H + 1, n, n)),
        fevd=np.zeros((H + 1, n, n)),
        max_fev_at=1,
        irf_lower=None,
        irf_upper=None,
        ci=0.9,
    )
    assert res.B.shape == (n, n)
    with pytest.raises(Exception):
        res.ci = 0.95  # frozen


def test_identify_maxshare_full_pipeline_returns_dataclass():
    from puremacro.var.identify.maxshare import identify_maxshare
    from puremacro.var.identify._results import MaxShareResult

    Y = _toy_var2()
    res = identify_maxshare(Y, p=2, target_idx=0, max_fev_at=1,
                            horizon=4, n_bootstrap=50, ci=0.68, seed=0)
    assert isinstance(res, MaxShareResult)
    assert res.B.shape == (2, 2)
    assert res.irfs.shape == (5, 2, 2)
    assert res.fevd.shape == (5, 2, 2)
    assert res.irf_lower.shape == (5, 2, 2)
    assert res.irf_upper.shape == (5, 2, 2)
    assert 0.0 <= res.fev_share_at_target <= 1.0


def test_identify_maxshare_skips_bootstrap_when_n_bootstrap_is_zero():
    from puremacro.var.identify.maxshare import identify_maxshare

    Y = _toy_var2()
    res = identify_maxshare(Y, p=2, horizon=4, n_bootstrap=0, seed=0)
    assert res.irf_lower is None
    assert res.irf_upper is None
```

- [ ] **Step 2: Run test to verify fail**

```bash
python -m pytest tests/test_var/test_maxshare_pipeline.py -v
```

Expected: 3 FAIL — `identify_maxshare` doesn't exist in canonical yet.

- [ ] **Step 3: Add `MaxShareResult` to `_results.py`**

Append to `puremacro/var/identify/_results.py`:

```python
@dataclass(frozen=True)
class MaxShareResult:
    """Result of :func:`puremacro.var.identify.maxshare.identify_maxshare`.

    Faust-Uhlig (2003) max-share identification: pick the shock whose
    contribution to the target variable's forecast-error variance at
    horizon ``max_fev_at`` is maximised.

    Attributes
    ----------
    B : ndarray, shape (n, n)
        Full structural impact matrix. Column 0 is the max-share shock;
        columns 1..n-1 are orthogonal completions (no structural label).
    q : ndarray, shape (n,)
        Optimal rotation vector, ``B[:, 0] = chol(Sigma_u) @ q``.
    fev_share_at_target : float
        Achieved FEV share at ``max_fev_at`` for the target variable.
    irfs : ndarray, shape (H+1, n, n)
        Structural impulse responses.
    fevd : ndarray, shape (H+1, n, n)
        Forecast-error variance decomposition.
    max_fev_at : int
        Horizon used for identification.
    irf_lower : ndarray or None, shape (H+1, n, n)
        Lower bootstrap band; ``None`` if ``n_bootstrap == 0``.
    irf_upper : ndarray or None, shape (H+1, n, n)
        Upper bootstrap band; ``None`` if ``n_bootstrap == 0``.
    ci : float
        Coverage for bootstrap bands.
    """
    B: np.ndarray
    q: np.ndarray
    fev_share_at_target: float
    irfs: np.ndarray
    fevd: np.ndarray
    max_fev_at: int
    irf_lower: Optional[np.ndarray]
    irf_upper: Optional[np.ndarray]
    ci: float

    def summary(self) -> str:
        n = self.B.shape[0]
        H = self.irfs.shape[0] - 1
        band_str = ("no bootstrap" if self.irf_lower is None
                    else f"CI {self.ci:.2f}")
        return (
            f"Max-share SVAR (Faust-Uhlig)\n"
            f"  variables (n)        : {n}\n"
            f"  horizon (H)          : {H}\n"
            f"  FEV target horizon   : {self.max_fev_at}\n"
            f"  FEV share at target  : {self.fev_share_at_target:.4f}\n"
            f"  bands                : {band_str}\n"
        )
```

- [ ] **Step 4: Extend `puremacro/var/identify/maxshare.py`**

Read the legacy `svar/identify_maxshare.py` for the FEVD + bootstrap implementations. Port `_companion_irfs`, `_build_M_matrix`, `_complete_B`, `_compute_fevd_from_irfs`, and `identify_maxshare` into the canonical file alongside the existing `maxshare(...)` and `news_maxshare(...)`.

The canonical port:
- Uses `safe_cholesky` from `puremacro._linalg` (not `np.linalg.cholesky`) for the bootstrap loop, matching the canonical SVAR scheme convention.
- Routes the residual bootstrap through `puremacro.inference.bootstrap.residual_bootstrap` (NOT `inference.legacy.bootstrap`).
- Uses a single `ci: float = 0.9` kwarg (NOT `q_lo`/`q_hi` legacy). Internal mapping: `lo_q = (1-ci)/2`, `hi_q = 1 - lo_q`.
- Returns the new `MaxShareResult` (frozen) instead of legacy mutable.

Implementation sketch (the engineer should read the legacy file thoroughly first since it's 285 LOC):

```python
def identify_maxshare(
    Y: np.ndarray,
    *,
    p: int = 2,
    target_idx: int = 0,
    max_fev_at: int = 1,
    horizon: int = 20,
    n_bootstrap: int = 500,
    ci: float = 0.9,
    seed: int = 0,
) -> MaxShareResult:
    """Faust-Uhlig (2003) max-share identification with full pipeline.

    Parameters
    ----------
    Y : ndarray, shape (T, n)
        Data matrix.
    p : int
        VAR lag order.
    target_idx : int, default 0
        Index of the variable whose FEV share is maximised.
    max_fev_at : int, default 1
        Horizon h* at which FEV share is maximised. 1 = impact.
    horizon : int, default 20
        IRF horizon H for the output.
    n_bootstrap : int, default 500
        Residual bootstrap repetitions. 0 to skip bands.
    ci : float, default 0.9
        Bootstrap band coverage level.
    seed : int, default 0
        Bootstrap RNG seed.

    Returns
    -------
    MaxShareResult
    """
    from .._linalg import safe_cholesky
    # ... port from legacy svar/identify_maxshare.py ...
    # ... use safe_cholesky for the bootstrap conditioning gate ...
    # ... return MaxShareResult(B=..., q=..., fev_share_at_target=...,
    #     irfs=..., fevd=..., max_fev_at=max_fev_at,
    #     irf_lower=..., irf_upper=..., ci=ci) ...
```

Keep the existing `maxshare(...)` and `news_maxshare(...)` exports unchanged — they're the low-level building blocks. `identify_maxshare(...)` is the new full-pipeline orchestrator.

- [ ] **Step 5: Re-export from `var/identify/__init__.py`**

Add:

```python
from .maxshare import identify_maxshare
from ._results import MaxShareResult
```

…and append both to `__all__`.

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_var/test_maxshare_pipeline.py -v
python -m pytest tests/test_var/ -q
python -m pytest tests/test_pyodide_compat.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add puremacro/var/identify/_results.py puremacro/var/identify/maxshare.py puremacro/var/identify/__init__.py tests/test_var/test_maxshare_pipeline.py
git commit -m "$(cat <<'EOF'
feat(var/identify): identify_maxshare full pipeline + MaxShareResult — 0.43.0 prep

Promotes svar/identify_maxshare.py's full pipeline (FEVD + bootstrap)
to canonical var/identify/maxshare. Replaces legacy q_lo/q_hi kwargs
with single ci, routes bootstrap through inference.bootstrap (not
legacy), and uses safe_cholesky for the conditioning gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Port `lp_panel_regime_interaction` to canonical lp/panel.py

**Files:**
- Modify: `puremacro/lp/panel.py` — add `lp_panel_regime_interaction` function.
- Test: `tests/test_lp/test_regime_interaction.py`.

**Why first:** 7 callers in `tools/run_*.py` (verified) + 1 in `tools/build_cross_country_tightness_extended.py` block Task 5's tools/ migration until this function exists in canonical.

- [ ] **Step 1: Read the legacy implementation**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
grep -nA50 "^def lp_panel_regime_interaction" puremacro/lp/lp_panel.py | head -80
```

Record the signature and return type.

- [ ] **Step 2: Write failing test**

Create `tests/test_lp/test_regime_interaction.py`:

```python
"""Tests for lp.panel.lp_panel_regime_interaction (ported from legacy)."""
import numpy as np
import pandas as pd
import pytest


def _toy_regime_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in "ABCDE":
        for t in range(60):
            regime = 0 if t < 30 else 1
            rows.append({
                "code": c, "date": t,
                "shock": rng.standard_normal(),
                "y": rng.standard_normal(),
                "regime": regime,
            })
    return pd.DataFrame(rows)


def test_lp_panel_regime_interaction_returns_dataframe_with_per_regime_cols():
    from puremacro.lp.panel import lp_panel_regime_interaction

    df = _toy_regime_panel()
    # Use the canonical kwarg shape; if legacy used different names,
    # this test PINS the canonical kwarg names for callers to follow.
    result = lp_panel_regime_interaction(
        df, y="y", x="shock", regime_col="regime",
        horizons=range(0, 5), n_lags=1,
        entity_level="code", time_level="date",
    )
    # The function returns a long-form DataFrame keyed by (h, regime).
    assert isinstance(result, pd.DataFrame)
    assert {"h", "regime", "beta", "se"}.issubset(set(result.columns))
    assert set(result["regime"].unique()) == {0, 1}
```

The test's kwarg names anchor what every caller will use in Task 5. Update the test if the audit (Task 0 / step 1 above) found a different canonical kwarg convention.

- [ ] **Step 3: Run test to verify fail**

```bash
python -m pytest tests/test_lp/test_regime_interaction.py -v
```

Expected: FAIL — `lp_panel_regime_interaction` not in `lp.panel`.

- [ ] **Step 4: Port the function**

Read `puremacro/lp/lp_panel.py` for the legacy `lp_panel_regime_interaction` function. The legacy signature uses `outcome=`, `shock=`, `unit_col=`, `date_col=`, `regime_col=`. Port to `puremacro/lp/panel.py` with the canonical kwarg naming convention (`y=`, `x=`, `entity_level=`, `time_level=`, `regime_col=`).

If the legacy uses `linearmodels.PanelOLS` lazily, preserve that lazy import inside the function body. If it uses `inference.legacy.<x>`, switch to `puremacro.inference.<x>`.

Add to `__all__` in `puremacro/lp/panel.py`.

- [ ] **Step 5: Run test to verify pass**

```bash
python -m pytest tests/test_lp/test_regime_interaction.py -v
python -m pytest tests/test_lp/ -q
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add puremacro/lp/panel.py tests/test_lp/test_regime_interaction.py
git commit -m "$(cat <<'EOF'
feat(lp/panel): port lp_panel_regime_interaction from legacy — 0.43.0 prep

Unblocks 7 tools/ callers and the lp/lp_panel.py shim conversion.
Canonical kwargs: y/x/regime_col/entity_level/time_level.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Port `lp_smooth_transition_irf` to canonical lp/state_dep.py

**Files:**
- Modify: `puremacro/lp/state_dep.py` — add `lp_smooth_transition_irf` function.
- Test: `tests/test_lp/test_smooth_transition.py`.

Same pattern as Task 3 — port a function from legacy with no canonical equivalent. The legacy version lives in `puremacro/lp/lp_state_dep.py`.

- [ ] **Step 1: Read legacy**

```bash
grep -nA60 "^def lp_smooth_transition_irf" puremacro/lp/lp_state_dep.py | head -80
```

- [ ] **Step 2: Write failing test**

Create `tests/test_lp/test_smooth_transition.py`:

```python
"""Tests for lp.state_dep.lp_smooth_transition_irf (ported from legacy)."""
import numpy as np
import pandas as pd
import pytest


def _toy_series(T: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "y": rng.standard_normal(T).cumsum(),
        "shock": rng.standard_normal(T),
        "state_var": np.linspace(-2, 2, T) + 0.5 * rng.standard_normal(T),
    })


def test_lp_smooth_transition_irf_returns_dataframe():
    from puremacro.lp.state_dep import lp_smooth_transition_irf

    df = _toy_series()
    result = lp_smooth_transition_irf(
        df, y="y", x="shock", state_var="state_var",
        horizons=range(0, 5), n_lags=2, gamma=1.0,
    )
    assert isinstance(result, pd.DataFrame)
    # Columns expected: h, beta_high, beta_low, se_high, se_low (or similar).
    assert "h" in result.columns
```

- [ ] **Step 3: Verify failure, port function, re-test, commit**

```bash
python -m pytest tests/test_lp/test_smooth_transition.py -v   # fail
# ... port to puremacro/lp/state_dep.py ...
python -m pytest tests/test_lp/test_smooth_transition.py -v   # pass
python -m pytest tests/test_lp/ -q                             # no regressions
git add puremacro/lp/state_dep.py tests/test_lp/test_smooth_transition.py
git commit -m "$(cat <<'EOF'
feat(lp/state_dep): port lp_smooth_transition_irf from legacy — 0.43.0 prep

Unblocks lp/lp_state_dep.py shim conversion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Migrate tools/ scripts off legacy paths

**Files (per the Task 0 audit's caller list, the canonical set is):**
- `tools/run_state_bartik_urate_quartile.py`
- `tools/run_jolts_sectoral_lp.py`
- `tools/run_aus_state_vacancy_lp.py`
- `tools/run_bartik_surprises_lp.py`
- `tools/run_bartik_ltui_post2017_alone.py`
- `tools/run_bartik_horse_race_lp.py`
- `tools/run_jolts_state_bartik_lp.py`
- `tools/build_cross_country_tightness_extended.py`
- `tools/run_logurate_revision.py` (uses `puremacro.regress.lp` — out of scope unless audit demands; see Step 7)
- `tools/run_paper_extensions.py` (same as above)

Pattern per file:

- [ ] **Step 1: For each file in the list above (one commit per file), rewrite imports + call sites**

For files that import `lp_panel_regime_interaction`:

```python
# Before:
from puremacro.lp.lp_panel import lp_panel_regime_interaction

# After:
from puremacro.lp.panel import lp_panel_regime_interaction
```

If the call site uses legacy kwargs, update them in the same commit. For `lp_panel_regime_interaction`:

```python
# Before:
result = lp_panel_regime_interaction(
    df, outcome="y", shock="shock", regime_col="regime",
    unit_col="code", date_col="date", horizons=range(0, 13),
    dk_lag=4, ci_level=0.9,
)

# After:
result = lp_panel_regime_interaction(
    df, y="y", x="shock", regime_col="regime",
    entity_level="code", time_level="date", horizons=range(0, 13),
    n_lags=4, alpha=0.10,
)
```

(Map `dk_lag` → `n_lags` if semantically the same; verify with the function's Task 3 signature.)

- [ ] **Step 2: Per-file verification**

For each `tools/<script>.py`, ensure it imports cleanly:

```bash
python -c "import importlib.util, sys
spec = importlib.util.spec_from_file_location('s', '/Users/jalonso/.../tools/<script>.py')
m = importlib.util.module_from_spec(spec)
# Don't execute — just check syntax via compile
import ast
ast.parse(open('/Users/jalonso/.../tools/<script>.py').read())
print('OK')
"
```

A full functional smoke is not run here — the goal is import-path correctness; Task 11 covers full execution.

- [ ] **Step 3: Commit per script**

For each tools/ script, one commit:

```bash
git add tools/<script>.py
git commit -m "refactor(tools): <script> → canonical lp.panel imports

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Verify no remaining `puremacro.lp.lp_*` or `puremacro.svar.*` imports in tools/**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
grep -rln "puremacro\.lp\.lp_\|puremacro\.svar\." tools/ 2>/dev/null | grep -v _archive
```

Expected: only `tools/make_notebook_*.py` (handled in Task 6) and possibly `tools/run_logurate_revision.py` / `tools/run_paper_extensions.py` if they import `puremacro.regress.lp` (separately scoped).

- [ ] **Step 5: Resolve `puremacro.regress.lp` callers**

The `regress/` package is a thin re-export of `lp.panel`. Two options:
- (a) Update the two callers (`run_logurate_revision`, `run_paper_extensions`) to import directly from `puremacro.lp.panel`. Drop `regress/lp` re-export at the same time.
- (b) Leave `regress/` alone — declared out of scope in the spec.

Default to (a) since it's a small change and removes one more shim. If the regress package has callers we missed, defer to (b).

Per-file work + commit if (a).

---

## Task 6: Migrate make_notebook_*.py builders + .ipynb files (per Task 0 classification)

The audit in Task 0 produced a per-notebook verdict (rename-only vs body-rewrite) and a per-builder caller list. For each affected notebook, this task touches BOTH the builder and the `.ipynb` together (memory pin: notebooks ↔ builders are paired).

**Files (per Task 0 audit; the listed set is the starting point):**

Each pair gets its own commit:

| Builder | Notebook | Class |
|---|---|---|
| `tools/make_notebook_R1_01.py` | `notebooks/R1_methods/R1_01_svar_menu.ipynb` | body-rewrite |
| `tools/make_notebook_R1_02.py` | `notebooks/R1_methods/R1_02_lp_menu.ipynb` | body-rewrite |
| `tools/make_notebook_R1_03.py` | `notebooks/R1_methods/R1_03_cross_country.ipynb` | per audit |
| `tools/make_notebook_R1_04.py` | `notebooks/R1_methods/R1_04_dsge_compare.ipynb` | per audit |
| `tools/make_notebook_R1_05.py` | `notebooks/R1_methods/R1_05_publication.ipynb` | body-rewrite |
| `tools/make_notebook_R2_01.py` | `notebooks/R2_subnational/R2_01_panels_and_data.ipynb` | body-rewrite |
| `tools/make_notebook_R2_02.py` | `notebooks/R2_subnational/R2_02_lp_iv_bartik.ipynb` | body-rewrite |
| (no builder) | `notebooks/T5_research_lab.ipynb` | per audit; direct edit |
| (check audit) | `notebooks/T_us_national.ipynb` | per audit |

Per pair:

- [ ] **Step 1: Edit the `.ipynb` and its paired builder simultaneously**

Open both. Apply identical import-path renames and kwarg edits to:
- The `from puremacro.svar.<x> import ...` lines.
- The `from puremacro.lp.lp_<x> import ...` lines.
- The `point, lo, hi = bq_svar(...)` tuple-unpack lines (rewrite to `result = bq_svar(...); result.irf_point[h, i, j]` attribute access — the canonical returns `BQSVARResult` axes `(H+1, n, n)` instead of legacy `(n, n, H+1)`, so index order also flips).
- Any kwarg renames identified in Task 0 step 4.

For rename-only notebooks: only import lines change; do NOT re-execute (memory pin: renames don't need a rebuild). The committed outputs stay valid.

For body-rewrite notebooks: edit cells, then mark for re-execution in Task 11. Do NOT re-execute as part of this task — that's centralised in Task 11's controller-only background runs.

- [ ] **Step 2: Verify the .ipynb still parses**

```bash
jupyter nbconvert --to script "notebooks/<path>" --stdout 2>&1 | head -10
```

Expected: prints the script body, no syntax errors.

- [ ] **Step 3: Verify the builder still produces a valid notebook**

```bash
python tools/<builder>.py
git diff -- notebooks/<path>
```

For rename-only notebooks, the diff should be ONLY import-cell changes. For body-rewrite, also cell-source changes (no output changes yet — those come in Task 11).

If the builder mutates the notebook outputs (clobbers them) when re-run, do NOT commit those changes — `git checkout notebooks/<path>` and instead edit BOTH files in a single Edit-tool call, then commit. Memory pin: `feedback_builder_clobbers_outputs`.

- [ ] **Step 4: Commit the pair**

```bash
git add tools/<builder>.py notebooks/<path>
git commit -m "refactor(<notebook>): canonical imports

Paired update to <builder> and <notebook>. <verdict per Task 0>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

For T5_research_lab (no builder), just edit the `.ipynb` and commit alone.

---

## Task 7: Update teaching/bq_canonical.py to use canonical attribute access

**Files:**
- Modify: `puremacro/teaching/bq_canonical.py`.

The Phase-2 fix at commit `066d9fa` added a local `bq_svar` adapter that transposes canonical `BQSVARResult` arrays back to legacy `(n, n, H+1)` shape. 0.43.0 removes this adapter and switches the two call sites to canonical attribute access.

- [ ] **Step 1: Read the current state**

```bash
grep -nE "def bq_svar|bq_svar\(|point, lo, hi" puremacro/teaching/bq_canonical.py | head -10
```

- [ ] **Step 2: Remove the local adapter; switch call sites**

Replace the local adapter section:

```python
# REMOVE:
from puremacro.var.identify.bq import bq_svar as _bq_svar_canonical

def bq_svar(*args, **kwargs):
    """Local legacy-axis adapter ..."""
    r = _bq_svar_canonical(*args, **kwargs)
    def _T(a): return a.transpose(1, 2, 0)
    return _T(r.irf_point), _T(r.irf_lower), _T(r.irf_upper)
```

With direct canonical import:

```python
# ADD:
from puremacro.var.identify.bq import bq_svar
```

At the two call sites (around lines 80 and 112 per Phase-2 review), rewrite:

```python
# Before:
point, lo, hi = bq_svar(
    Y, p=2, horizon=20, n_boot=1000, ci=0.9, seed=0,
    permanent_var_idx=0,
)
# (downstream code uses point[i, j, h])

# After:
result = bq_svar(
    Y, p=2, horizon=20, n_boot=1000, ci=0.9, seed=0,
    permanent_var_idx=0,
)
point, lo, hi = result.irf_point, result.irf_lower, result.irf_upper
# (downstream code now needs index reordering: point[h, i, j] instead of point[i, j, h])
```

Audit the downstream code in `bq_canonical.py` for index-order assumptions. The legacy convention was `[response_var, shock, horizon]`; canonical is `[horizon, response_var, shock]`. Rewrite ALL index expressions to canonical ordering.

- [ ] **Step 3: Run the strongest gate**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/ -q -W "error::DeprecationWarning" \
  --ignore tests/test_deprecation_warnings.py \
  --ignore tests/test_shim_shape_preservation.py \
  --ignore tests/test_pyodide_compat.py \
  --ignore tests/test_public_api.py 2>&1 | tail -20
```

Expected: same pre-existing failure set as Phase-2 final review (no new ones).

- [ ] **Step 4: Smoke-test the teaching module**

```bash
python -c "
import warnings; warnings.simplefilter('error', DeprecationWarning)
from puremacro.teaching.bq_canonical import bq_gdp_urate, bq_employment_epu
import numpy as np
# These should not raise DeprecationWarning on import.
print('imports clean')
"
```

Expected: prints `imports clean`.

- [ ] **Step 5: Commit**

```bash
git add puremacro/teaching/bq_canonical.py
git commit -m "$(cat <<'EOF'
refactor(teaching): bq_canonical uses BQSVARResult directly — 0.43.0

Removes the local axis-translation adapter added in Phase 2. Call
sites now access result.irf_point/lower/upper directly with canonical
(H+1, n, n) indexing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Delete the entire shim layer

**Files (deletions):**
- `puremacro/svar/` — entire directory (8 files).
- `puremacro/lp/lp_jorda.py`, `lp/lp_iv.py`, `lp/lp_panel.py`, `lp/lp_panel_dk.py`, `lp/lp_state_dep.py`, `lp/lp_smooth.py`, `lp/lp_garch_state.py`, `lp/lp_garch_in_mean.py`.
- `puremacro/lp/garch_utils.py` → rename to `_garch_utils.py`, update internal callers.
- `puremacro/inference/legacy/` — entire directory.
- `tests/test_deprecation_warnings.py`.
- `tests/test_shim_shape_preservation.py`.

This is the destructive task; runs only after Tasks 1-7 are green.

- [ ] **Step 1: Strongest pre-deletion gate**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
echo "in-puremacro:" && grep -rln "puremacro\.svar\.\|puremacro\.lp\.lp_\|puremacro\.inference\.legacy" puremacro/puremacro/ 2>/dev/null
echo "in-tools:" && grep -rln "puremacro\.svar\.\|puremacro\.lp\.lp_\|puremacro\.inference\.legacy" tools/ 2>/dev/null | grep -v _archive
echo "in-notebooks:" && grep -rln "puremacro\.svar\.\|puremacro\.lp\.lp_\|puremacro\.inference\.legacy" notebooks/ 2>/dev/null | grep -v _archive
```

Expected: each line outputs nothing (only the `echo` text). If any caller remains, **STOP** — fix that caller before continuing.

- [ ] **Step 2: Delete `puremacro/svar/`**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
git rm -r puremacro/svar/
```

- [ ] **Step 3: Delete the 8 `lp/lp_*.py` files**

```bash
git rm puremacro/lp/lp_jorda.py puremacro/lp/lp_iv.py puremacro/lp/lp_panel.py puremacro/lp/lp_panel_dk.py puremacro/lp/lp_state_dep.py puremacro/lp/lp_smooth.py puremacro/lp/lp_garch_state.py puremacro/lp/lp_garch_in_mean.py
```

- [ ] **Step 4: Rename `garch_utils.py` to `_garch_utils.py` + update its callers**

```bash
git mv puremacro/lp/garch_utils.py puremacro/lp/_garch_utils.py
```

Update internal callers (`puremacro/lp/garch_state.py`, `puremacro/lp/garch_in_mean.py`) to import from the new path:

```python
# Before:
from puremacro.lp.garch_utils import fit_garch, make_regime_indicator

# After:
from ._garch_utils import fit_garch, make_regime_indicator
```

- [ ] **Step 5: Delete `puremacro/inference/legacy/`**

Pre-step: audit which canonical files (`puremacro/inference/<x>.py`) still need anything unique from `puremacro/inference/legacy/<x>.py`. The earlier Phase-2 work confirmed only `svar/*` shims used the legacy paths; the 4 distinct files have these unique surfaces:
- `inference/legacy/bootstrap.py`: `residual_bootstrap_var(...)` (different from `inference/bootstrap.py`'s `residual_bootstrap(...)`).
- `inference/legacy/wild_bootstrap.py`: `wild_bootstrap_var(...)`.
- `inference/legacy/block_bootstrap.py`: `block_bootstrap(...)`.
- `inference/legacy/weak_iv.py`: an older Cragg-Donald implementation.

After Task 8 step 4, the only callers of these unique functions are svar shims (which we just deleted) and lp_state_dep/lp_smooth (also deleted). Verify with:

```bash
grep -rn "residual_bootstrap_var\|wild_bootstrap_var\|legacy\.block_bootstrap" puremacro/ 2>/dev/null
```

Expected: zero hits.

Then delete:

```bash
git rm -r puremacro/inference/legacy/
```

- [ ] **Step 6: Delete shim test infrastructure**

```bash
git rm tests/test_deprecation_warnings.py tests/test_shim_shape_preservation.py
```

- [ ] **Step 7: Run the full suite**

```bash
python -m pytest tests/ -q --tb=line 2>&1 | tail -10
```

Expected: 1274 − 11 (shim tests deleted) + 6 (new Tasks 1+2 tests: 3 PanelSVARResult + 3 MaxShareResult) − 1 (deleted regression test that's no longer applicable) ≈ 1268 passed. The 10 pre-existing failures remain.

If failures other than the pre-existing 10 appear, **STOP** and triage.

- [ ] **Step 8: Strongest gate — no DeprecationWarning in any code path**

```bash
python -m pytest tests/ -q -W "error::DeprecationWarning" \
  --ignore tests/test_pyodide_compat.py 2>&1 | tail -20
```

Expected: same failure set as Phase 2's strict-mode baseline (the pre-existing `datetime.utcnow()` + `test_public_api` issues) but NO new failures from our removed shims. The shims that emitted DeprecationWarning no longer exist, so the relevant test cases don't run.

- [ ] **Step 9: Commit (one big deletion commit)**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore(0.43.0): delete shim layer — svar/, lp/lp_*.py, inference/legacy/

Per the 0.43.0 spec, all Phase-2 shim files and the Phase-2.5 banner
files are deleted now that callers (Tasks 5–7) have been migrated.

Removed:
- puremacro/svar/ (entire package).
- puremacro/lp/lp_{jorda,iv,panel,panel_dk,state_dep,smooth,garch_state,garch_in_mean}.py.
- puremacro/inference/legacy/ (entire directory).
- tests/test_deprecation_warnings.py + tests/test_shim_shape_preservation.py.

Renamed:
- puremacro/lp/garch_utils.py → puremacro/lp/_garch_utils.py (private).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Migrate canonical off `inference/legacy/*` (cleanup if needed)

This task is a no-op if Task 8's audit confirmed canonical paths no longer reference `inference/legacy/*`. Per the spec's verification on 2026-05-18 (`grep -n "inference\.legacy" puremacro/var/identify/*.py` returned nothing), this is already done.

If Task 0's audit surfaces any canonical file still importing legacy, handle it here:

- [ ] **Step 1: Re-verify**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
grep -rn "inference\.legacy" puremacro/ 2>/dev/null
```

Expected: zero hits (because `inference/legacy/` no longer exists per Task 8).

- [ ] **Step 2: If hits found, migrate**

For each remaining `from puremacro.inference.legacy.<x> import <y>`:
- Check whether `<y>` exists in `puremacro.inference.<x>` already.
- If yes: change import.
- If no: promote `<y>` from `inference/legacy/<x>.py` to `inference/<x>.py` first (re-introducing the function); then change import.

If Task 8 succeeded, this task has no work — mark it skipped in the task list and proceed.

---

## Task 10: ARCHITECTURE.md + CHANGELOG.md updates

**Files:**
- Modify: `ARCHITECTURE.md`.
- Modify: `CHANGELOG.md`.

- [ ] **Step 1: Update ARCHITECTURE.md**

Find the section "## Known consolidation candidates" (added in 0.42.0). Replace with:

```markdown
## Known consolidation candidates

Done at 0.43.0. The `svar/`, `lp/lp_*.py`, and `inference/legacy/` shim
layers were retired:

- `panel_svar` → `var/identify/panel.py` (frozen `PanelSVARResult`).
- `identify_maxshare` → `var/identify/maxshare.identify_maxshare`
  with full FEVD + bootstrap pipeline; frozen `MaxShareResult`.
- `lp_panel_regime_interaction` and `lp_smooth_transition_irf` ported
  from legacy `lp/lp_*` to canonical `lp/panel.py` and `lp/state_dep.py`.
- All 8 `lp/lp_*.py` files deleted. Callers updated to canonical kwargs.
- `inference/legacy/` directory deleted; canonical paths use
  `inference/<x>.py` directly.
- `tests/test_deprecation_warnings.py` and
  `tests/test_shim_shape_preservation.py` deleted (no shims to gate).
```

Also remove the "Phase 2 shim status" table — no shims remain.

Update the stability tier table: remove every row for `svar/*` and `lp/lp_*.py`. Add `var/identify/panel.py`, `var/identify/maxshare.identify_maxshare`, `lp/panel.lp_panel_regime_interaction`, `lp/state_dep.lp_smooth_transition_irf` as Stable.

Remove the retirement-note paragraph for `inference/legacy/`.

- [ ] **Step 2: Prepend the 0.43.0 entry to CHANGELOG.md**

Add at the top (before the 0.42.0 entry):

```markdown
## 0.43.0 — 2026-05-18

Shim retirement + canonical promotion. The `svar/`, `lp/lp_*.py`, and
`inference/legacy/` paths flagged in 0.42.0 are deleted entirely.
`panel_svar` and `identify_maxshare` promoted to `var/identify/*` with
frozen-dataclass results. LP signatures harmonized to canonical
kwargs; all callers updated.

### Added
- `puremacro.var.identify.panel.mean_group_svar` + `PanelSVARResult`
  (canonical port of `svar/panel_svar.py`; cholesky + bq schemes).
- `puremacro.var.identify.maxshare.identify_maxshare` + `MaxShareResult`
  (full FEVD + bootstrap pipeline; extends the existing low-level
  `maxshare(...)` and `news_maxshare(...)` exports).
- `puremacro.lp.panel.lp_panel_regime_interaction` (promoted from
  legacy `lp/lp_panel.py`).
- `puremacro.lp.state_dep.lp_smooth_transition_irf` (promoted from
  legacy `lp/lp_state_dep.py`).

### Removed (breaking)
- `puremacro.svar.*` — entire package.
- `puremacro.lp.lp_*` — all 8 files.
- `puremacro.inference.legacy.*` — entire directory (4 distinct files
  + 6 byte-identical shims).
- `puremacro.lp.garch_utils` — renamed to private `_garch_utils`.

### Changed
- LP signatures harmonized to canonical kwargs:
  `outcome→y`, `shock→x`, `unit_col→entity_level`,
  `date_col→time_level`, `dk_lag→n_lags`, `ci_level→alpha`.
- 8 `tools/run_*.py` + 4 `tools/build_*.py` scripts updated to canonical.
- 9 notebooks (R1_methods/*, R2_subnational/*, T5_research_lab,
  T_us_national) updated to canonical imports. Body-rewrite notebooks
  re-executed via paired builders.
- `teaching/bq_canonical.py` no longer has the Phase-2 local axis-
  translation adapter; call sites use `BQSVARResult` attribute access.
- `tests/test_deprecation_warnings.py` and
  `tests/test_shim_shape_preservation.py` deleted (no shims remain).
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -q --tb=no 2>&1 | tail -5
```

Expected: unchanged from Task 8 result.

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(0.43.0): ARCHITECTURE + CHANGELOG for shim retirement

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Re-execute body-rewrite notebooks

Per memory pin `feedback_long_nbconvert_no_subagent`, this task runs in the **controller's background**, NOT inside a subagent. Each notebook may take 5-15 minutes.

**Notebooks (per Task 0 classification — body-rewrite only):**
- `notebooks/R1_methods/R1_01_svar_menu.ipynb` (precondition: `data/processed/panel_Q.parquet` must exist)
- `notebooks/R1_methods/R1_02_lp_menu.ipynb`
- `notebooks/R1_methods/R1_05_publication.ipynb`
- `notebooks/R2_subnational/R2_01_panels_and_data.ipynb`
- `notebooks/R2_subnational/R2_02_lp_iv_bartik.ipynb`
- Any others classified body-rewrite in Task 0.

- [ ] **Step 1: Build the data cache if R1_01 is on the list**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
test -f data/processed/panel_Q.parquet || python -c "
from puremacro.build_panel import build_all
build_all(refresh=False, fast=True)
"
```

If `build_all` fails (network unreachable), defer R1_01 re-execution and document in `docs/plans/_043_audit_notes.md`. Proceed with the other notebooks.

- [ ] **Step 2: For each body-rewrite notebook, re-execute via its paired builder**

For notebooks with builders:

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
python tools/make_notebook_<R*>.py
```

For T5_research_lab (no builder):

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/T5_research_lab.ipynb --inplace \
  --ExecutePreprocessor.timeout=900
```

Run each command in the controller's background (`run_in_background=true` if using Bash tool). Wait for completion.

- [ ] **Step 3: Inspect diffs**

```bash
git diff -- notebooks/R1_methods/R1_01_svar_menu.ipynb | head -60
```

Expected: cell sources unchanged from Task 6's commit; outputs (figures, tables, prints) may shift slightly due to canonical implementations using `safe_cholesky` (different conditioning behaviour on degenerate Σ). Review each diff for sanity.

If any diff shows broken cells (Python errors in outputs), **STOP** and triage before committing.

- [ ] **Step 4: Commit per notebook (or batched if all clean)**

```bash
git add notebooks/<path>.ipynb
git commit -m "$(cat <<'EOF'
nb(<chapter>): re-execute against 0.43.0 canonical

Outputs regenerated after Task 6's import + kwarg migrations.
Pre-execution → post-execution diff reviewed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Version bump to 0.43.0

**Files:**
- Modify: `puremacro/__init__.py` (`__version__`).
- Modify: `pyproject.toml` (`version`).
- Modify: `tests/test_import.py` (expected version string).
- Modify: `tests/fixtures/public_api_snapshot.json` — regenerate for the new dataclasses.

- [ ] **Step 1: Bump version literals**

Edit `puremacro/__init__.py`: `__version__ = "0.42.0"` → `"0.43.0"`.

Edit `pyproject.toml`: `version = "0.42.0"` → `"0.43.0"`.

Edit `tests/test_import.py`: update the expected `"0.42.0"` → `"0.43.0"`.

- [ ] **Step 2: Regenerate public-API snapshot**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/test_public_api.py -v --update 2>&1 | tail -5
```

If the test doesn't support `--update`, regenerate by inspecting `tests/test_public_api.py` for its snapshot-generation function (often `_collect_current_api`) and run it manually to overwrite `tests/fixtures/public_api_snapshot.json`. The new entries should include `PanelSVARResult` and `MaxShareResult` under `puremacro.var.identify._results`, and the existing entries for the deleted svar/lp_* paths should be removed.

- [ ] **Step 3: Verify**

```bash
python -m pytest tests/test_import.py tests/test_public_api.py -v
```

Expected: 2-3 passed.

- [ ] **Step 4: Commit**

```bash
git add puremacro/__init__.py pyproject.toml tests/test_import.py tests/fixtures/public_api_snapshot.json
git commit -m "$(cat <<'EOF'
chore(release): bump version to 0.43.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Final verification gate

- [ ] **Step 1: Full pytest run**

```bash
python -m pytest tests/ -q --tb=line 2>&1 | tail -15
```

Expected: 1268 ± a few passed, 10 pre-existing failed. Record final counts.

- [ ] **Step 2: Strongest gate — no legacy imports anywhere**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git grep -E "puremacro\.svar\.|puremacro\.lp\.lp_|puremacro\.inference\.legacy" -- ':!docs/' ':!CHANGELOG.md' ':!*.bak'
```

Expected: zero hits.

- [ ] **Step 3: Strict DeprecationWarning gate**

```bash
python -m pytest tests/ -q -W "error::DeprecationWarning" \
  --ignore tests/test_pyodide_compat.py \
  --ignore tests/test_public_api.py 2>&1 | tail -10
```

Expected: same pre-existing failure set as Phase 2's baseline (the `datetime.utcnow()` clusters in narrative sources). Zero new failures introduced.

- [ ] **Step 4: Pyodide compat**

```bash
python -m pytest tests/test_pyodide_compat.py -v
```

Expected: 2 passed.

- [ ] **Step 5: One-notebook smoke (R1_01)**

If `data/processed/panel_Q.parquet` exists (per Task 11), confirm:

```bash
jupyter nbconvert --to notebook --execute \
  "notebooks/R1_methods/R1_01_svar_menu.ipynb" \
  --output /tmp/R1_01_smoke_$(date +%Y%m%d).ipynb \
  --ExecutePreprocessor.timeout=900 2>&1 | tail -3
```

Expected: writes the smoke notebook with no DeprecationWarning anywhere (because no shims exist).

- [ ] **Step 6: git log + status**

```bash
git log --oneline f77eb57^..HEAD | head -30
git status --short | head -10
```

Confirm all 0.43.0 commits are in chain and no uncommitted files remain.

---

## Self-Review (run before declaring complete)

- [ ] **Spec coverage:** every section of the 0.43.0 spec maps to a task above? Phase A → Tasks 1–4; Phase B → Tasks 5–7; Phase C → Task 9 (likely no-op); Phase D → Tasks 8 + 11.

- [ ] **Placeholder scan:** no "TBD", no "implement later", no "Similar to Task N — repeat the code". Notebook classification TBD entries in Task 6 are explicitly resolved by Task 0's audit, not deferred indefinitely.

- [ ] **Type consistency:** `PanelSVARResult.irf_mean/irf_lower/irf_upper/country_irfs/country_ids/identification/p/horizon/ci` matches across Task 1's dataclass def and test. `MaxShareResult.B/q/fev_share_at_target/irfs/fevd/max_fev_at/irf_lower/irf_upper/ci` matches across Task 2.

- [ ] **Signature verification:** every `puremacro.*` import path and dataclass field name in this plan was verified against live code on 2026-05-18 via `grep` + file reads (memory pin `feedback_plan_verify_api_signatures`). Specifically:
  - `svar/panel_svar.py` legacy `PanelSVARResult` has `(irf_mean, irf_lo, irf_hi, country_irfs, country_ids, identification, p, horizon, ci)` mutable; canonical port freezes + renames `irf_lo→irf_lower`, `irf_hi→irf_upper`.
  - `svar/identify_maxshare.py` legacy `MaxShareResult` has `(B, q, fev_share_at_target, irfs, fevd, max_fev_at, irf_lo, irf_hi)` mutable; canonical adds `ci` field and freezes.
  - Canonical lp signatures: `lp_hac(y, x, horizons, n_lags, ...)`, `panel_lp_dk(df_wide, y, x, horizons, n_lags, controls, alpha, entity_level, time_level)`.
  - Canonical `var/identify/*` does NOT import `inference.legacy.*` (verified 2026-05-18).

- [ ] **Notebook hazard:** Task 6 explicitly cites the `feedback_builder_clobbers_outputs` and `feedback_notebook_builders_paired` memory pins. Task 11 cites `feedback_long_nbconvert_no_subagent`.

- [ ] **Risk register coverage:** every spec §4 risk has a mitigation in a corresponding task step.
