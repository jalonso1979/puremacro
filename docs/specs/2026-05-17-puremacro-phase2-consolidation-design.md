# puremacro Phase 2 consolidation — design

**Status:** approved 2026-05-17. Target release: **0.42.0**. Removal of the shim layer targeted for **0.43.0** (next paper-figure refresh cycle).

## Why

`ARCHITECTURE.md` (post-Phase-5 rewrite at 0.41.0, dated 2026-05-14) flags three "known consolidation candidates" — all explicitly deferred to a future spec:

1. **`puremacro/svar/*` → `puremacro/var/identify/*`.** All 9 SVAR tests still import from `puremacro.svar`. The canonical home `var/identify/*` ships frozen-dataclass results, `safe_cholesky` error surfacing, and sign-robust / sign-zero / non-Gaussian variants the legacy path lacks.
2. **`puremacro/lp/lp_*.py` → `puremacro/lp/*.py`.** Nine notebooks under `notebooks/R1_methods/`, `notebooks/R2_subnational/`, and at the `notebooks/` root reach into the prefix versions by deep path.
3. **Two result-object violations in `svar/`.** `svar/estimate_var.estimate_var` returns a 5-tuple; `svar/identify_bq.bq_svar` returns a 3-tuple. Both are the only remaining tuple-return offenders above the 2-value carve-out.

These have been live as TODOs since 0.41.0. They drag on every new estimator because new code has to decide which path to wire to, and on every test addition because the test fixtures still import the legacy module.

This spec executes the consolidation **without re-executing notebooks** — the binding constraint flagged in the contributing convention ("Builders clobber executed outputs": `tools/make_notebook_NN.py` strips figures/tables/prints, so a wholesale notebook re-execution risks losing pinned outputs).

## Approach — shim and translate

Every retired file becomes a thin shim with three responsibilities:

1. **Re-export** the canonical implementation under the legacy import path.
2. **Translate return shapes** where the canonical signature has shifted — most notably the IRF tensor axis order, which differs between legacy `(n, n, H+1)` and canonical `(H+1, n, n)`.
3. **Emit a `DeprecationWarning`** on first import naming the canonical replacement and the planned removal version.

Canonical shim template (the `bq` case is the worst — both unpacks the dataclass *and* transposes axes):

```python
# puremacro/svar/identify_bq.py  (shim)
"""DEPRECATED — use puremacro.var.identify.bq.

Re-exports the canonical bq_svar but unpacks BQSVARResult into the
legacy 3-tuple (point, lower, upper) with the legacy IRF axis
convention (n, n, H+1) preserved for back-compat with
notebooks/R1_methods/R1_01_svar_menu.ipynb.

Removal target: 0.43.0 (next paper-figure refresh cycle).
"""
import warnings as _w
from puremacro.var.identify.bq import bq_svar as _bq

_w.warn(
    "puremacro.svar.identify_bq is deprecated since 0.42.0; "
    "use puremacro.var.identify.bq. Removal target: 0.43.0.",
    DeprecationWarning, stacklevel=2,
)

def bq_svar(*args, **kwargs):
    r = _bq(*args, **kwargs)
    # Transpose (H+1, n, n) → (n, n, H+1) and unpack to legacy tuple.
    def _T(a): return a.transpose(1, 2, 0)
    return _T(r.irf_point), _T(r.irf_lower), _T(r.irf_upper)
```

Simpler shims (no axis translation, no dataclass unpacking) are one-liners:

```python
# puremacro/lp/lp_smooth.py  (shim)
"""DEPRECATED — use puremacro.lp.smooth."""
import warnings as _w
from puremacro.lp.smooth import *  # noqa: F401,F403

_w.warn(
    "puremacro.lp.lp_smooth is deprecated since 0.42.0; "
    "use puremacro.lp.smooth. Removal target: 0.43.0.",
    DeprecationWarning, stacklevel=2,
)
```

## Scope by file

### In-repo callers cut to canonical (no shim)

These switch their import paths cleanly because they're under our control and have no external pinned callers:

| File | Action |
|---|---|
| Tests under `tests/test_var/`, `tests/test_cholesky_shocks.py`, `tests/test_svar_*.py` (9 total) | Re-import from `puremacro.var.identify.*`; rewrite tuple unpacks → dataclass attribute access. |
| `puremacro/experiment.py` | Top-level dispatcher; swap any `puremacro.svar.*` and `puremacro.inference.legacy.*` imports for canonical homes. |
| `puremacro/gar/qar.py` | Swap `puremacro.inference.legacy` import for canonical. |
| `puremacro/teaching/bq_canonical.py` | Pin to canonical `puremacro.var.identify.bq`. Teaching is a side-channel, allowed to mutate. |

### Files converted to shims

These keep their legacy import path but the body becomes a re-export + DeprecationWarning + (optional) return-shape translator:

| Legacy path | Canonical target | Return-shape translation |
|---|---|---|
| `puremacro/svar/identify_cholesky.py` | `puremacro/var/identify/cholesky.py` | Unpack `CholeskySVARResult` → 3-tuple; transpose IRF `(H+1, n, n)` → `(n, n, H+1)`. |
| `puremacro/svar/identify_bq.py` | `puremacro/var/identify/bq.py` | Same translator as cholesky. |
| `puremacro/svar/identify_sign.py` | `puremacro/var/identify/sign.py` | Same. |
| `puremacro/svar/identify_proxy.py` | `puremacro/var/identify/proxy.py` | Same; legacy `proxy` returns `(point, lower, upper, B, F)` — confirm field-by-field unpack from `ProxySVARResult`. |
| `puremacro/svar/identify_heteroskedasticity.py` | `puremacro/var/identify/hetero.py` | Unpack `HeteroResult`; verify axis. |
| `puremacro/svar/identify_maxshare.py` | `puremacro/var/identify/maxshare.py` | Unpack; preserves the lazy-statsmodels import inside the shim (Pyodide contract). |
| `puremacro/svar/panel_svar.py` | No canonical home yet — leave in place at 0.42.0, mark as Phase-2.5 candidate. |
| `puremacro/svar/estimate_var.py` | `puremacro/var/estimate.py` | Wrap canonical 5-tuple in `VarEstimateResult` (see §"Result-object cleanup"); shim unpacks back to 5-tuple. |
| `puremacro/lp/lp_jorda.py` | `puremacro/lp/jorda.py` | None — both return DataFrames. |
| `puremacro/lp/lp_iv.py` | `puremacro/lp/iv.py` | None. |
| `puremacro/lp/lp_panel.py` | `puremacro/lp/panel.py` | None. |
| `puremacro/lp/lp_panel_dk.py` | `puremacro/lp/panel_dk.py` | None. |
| `puremacro/lp/lp_state_dep.py` | `puremacro/lp/state_dep.py` | None. |
| `puremacro/lp/lp_smooth.py` | `puremacro/lp/smooth.py` | None. |
| `puremacro/lp/lp_garch_state.py` | `puremacro/lp/garch_state.py` | None. |
| `puremacro/lp/lp_garch_in_mean.py` | `puremacro/lp/garch_in_mean.py` | None. |
| `puremacro/lp/garch_utils.py` | Promote to `puremacro/lp/_garch_utils.py` (private helper); leave shim. | None. |

### Result-object cleanup (rides with above)

- **New `puremacro.var._results.VarEstimateResult`** — frozen dataclass with fields `(A_list: list[np.ndarray], c: np.ndarray, Sigma: np.ndarray, resid: np.ndarray, X: np.ndarray)`. Add `.n_obs` and `.summary()`.
- **`var/estimate.estimate_var`** — wraps its current 5-tuple return into `VarEstimateResult`. All canonical callers in `var/identify/*` switch to attribute access. Verified call sites: `bq.py`, `cholesky.py`, `proxy.py`, `hetero.py`, `maxshare.py`, `non_gaussian.py`, `sign.py`, `sign_robust.py`, `sign_zero.py`.
- **`svar/estimate_var.estimate_var`** shim — unpacks `VarEstimateResult` back to the legacy 5-tuple for back-compat with downstream legacy `svar/identify_*` modules (which are themselves shims). The shim-to-shim chain is fine; it dissolves at 0.43.0.
- **`var/identify/bq.py`** — already returns `BQSVARResult`. No change.

### Notebooks — deferred

The 9 notebooks below keep working unchanged via the shim layer. They emit one `DeprecationWarning` per legacy import the next time they're executed. **No rebuild.**

- `notebooks/R1_methods/R1_01_svar_menu.ipynb`
- `notebooks/R1_methods/R1_02_lp_menu.ipynb`
- `notebooks/R1_methods/R1_03_cross_country.ipynb`
- `notebooks/R1_methods/R1_04_dsge_compare.ipynb`
- `notebooks/R1_methods/R1_05_publication.ipynb`
- `notebooks/R2_subnational/R2_01_panels_and_data.ipynb`
- `notebooks/R2_subnational/R2_02_lp_iv_bartik.ipynb`
- `notebooks/T5_research_lab.ipynb`
- `notebooks/T_us_national.ipynb`

The 0.43.0 cutover that deletes the shims is paired with the next paper-figure refresh for each of these chapters — that's the natural point to update the deep imports and re-execute.

### `inference/legacy/` disposition at 0.42.0

After the in-repo cuts above, the only callers of `puremacro/inference/legacy/*` are the new shim files in `svar/*` and `lp/lp_*`. Two consequences:

- **`inference/legacy/` is untouched at 0.42.0.** It becomes a private support module for the deprecated shim layer.
- **A retirement note is added** at the top of each of the four legitimately-different files (`bootstrap.py`, `wild_bootstrap.py`, `block_bootstrap.py`, `weak_iv.py`): `# kept alive only for the deprecated svar/lp_* shims — removal scheduled with shims at 0.43.0`.
- The six byte-identical shims already in `inference/legacy/` (from the 0.41.0 consistency pass — `lp_block_bootstrap`, `moving_block_bootstrap`, `newey_west`, `pesaran_cce`, `swamy_test`, `balanced_panel`) stay as one-line `from puremacro.inference.<x> import *` shims. No change.

## DeprecationWarning convention

- Use `warnings.warn(<message>, DeprecationWarning, stacklevel=2)` at module top of every shim. Python's default `__warningregistry__` per-module collapses repeats, so each legacy import emits exactly one warning per process.
- **The shim lives in the deep file, not in `__init__.py`.** `puremacro/svar/__init__.py` and `puremacro/lp/__init__.py` stay as they are today (the `svar` `__init__.py` already documents the legacy status with a docstring; `lp/__init__.py` re-exports the canonical surface). Emitting from `__init__.py` would double-fire when a user does `from puremacro.svar import identify_cholesky`.
- Standard message:
  ```
  "puremacro.<legacy_path> is deprecated since 0.42.0; use puremacro.<canonical_path>. Removal target: 0.43.0."
  ```
- **New test file `tests/test_deprecation_warnings.py`**: parameterized over the shim file list. For each:
  1. Asserts the import emits exactly one `DeprecationWarning`.
  2. Asserts the message contains both `<legacy_path>` and `<canonical_path>`.
  3. Asserts the shim's exported callable returns the legacy shape (e.g. tuple, not dataclass) — captures regressions in the translation layer.

## Verification gates (before tagging 0.42.0)

1. `pytest tests/ -v` — full suite green. Expected count: `1220 + N_deprecation`, where `N_deprecation` is the count of shim files.
2. `pytest -W error::DeprecationWarning --ignore=tests/test_deprecation_warnings.py` — **the strongest single gate**: no in-repo caller still on a legacy path. Any test still hitting a shim becomes red here.
3. `pytest tests/test_pyodide_compat.py` — Pyodide promise unbroken. Shims must not introduce top-level forbidden imports.
4. `pytest tests/test_public_api.py` — `tests/fixtures/public_api_snapshot.json` regenerated for `VarEstimateResult`.
5. **One-notebook smoke test**: hand-execute `notebooks/R1_methods/R1_01_svar_menu.ipynb` (or one of the LP notebooks) end-to-end. Confirm:
   - all cell outputs match the committed `.ipynb` byte-for-byte where deterministic;
   - the first legacy import shows exactly one `DeprecationWarning` in the output;
   - no figure has shifted (compare pinned `.pdf` outputs under `notebooks/output_figures/` if applicable).

## Risk register

| Risk | Mitigation |
|---|---|
| **IRF-axis convention drift.** Shim must apply `.transpose(1, 2, 0)` to every `irf_point/lower/upper` array. Wrong axis silently produces a meaningless plot. | Explicit cross-shape test asserting `legacy_shim(Y, …) == transpose_then_unpack(canonical(Y, …))` on a fixed seed for every shimmed identification scheme. Lives in `tests/test_deprecation_warnings.py`. |
| **`filterwarnings = error` in `pyproject.toml` / `conftest.py`.** Would turn the shim's warning into a test-suite error and break the suite at import time. | Confirm `pyproject.toml [tool.pytest.ini_options].filterwarnings` does not promote `DeprecationWarning`. If it does, scope per shim test with `@pytest.mark.filterwarnings("default::DeprecationWarning")`. |
| **`experiment.py` is referenced by examples.** Switching its imports could shift dispatcher behavior subtly if anyone passes a tuple-shaped result through. | Audit `examples/*` for `experiment.lp`, `experiment.panel_lp`, `experiment.var_cholesky` call sites and assert each still passes after the cut. |
| **`inference/legacy/` files appear unused at 0.42.0** (no non-legacy callers) and may look dead to a future contributor. | Retirement note at the top of each kept-alive file. Listed in `ARCHITECTURE.md` § "Known consolidation candidates" with explicit "removal at 0.43.0" annotation. |
| **`narrative.types` still references "legacy" series of monthly-narrative magnitudes** (per existing comments) — distinct from this Phase-2 effort. | Out of scope. Flag in 0.42.0 CHANGELOG under "Not affected" to avoid confusion. |
| **Shim emits warning on every import** if Python's `__warningregistry__` is cleared (e.g. `importlib.reload`). | Acceptable — reloads in test code are rare; the user-facing surface is "one warning per process per legacy module." Document in the deprecation test header. |

## CHANGELOG entry (skeleton for 0.42.0)

```
## 0.42.0 — YYYY-MM-DD

Phase-2 consolidation: legacy svar/* and lp/lp_*.py are now thin
DeprecationWarning shims of the canonical var/identify/* and lp/*.py
paths. No behavior change for callers; removal target 0.43.0
(next paper-figure refresh).

### Added
- `puremacro.var.VarEstimateResult` frozen dataclass; `var.estimate_var`
  now returns this dataclass canonically.

### Deprecated
- `puremacro.svar.identify_{cholesky,bq,sign,proxy,heteroskedasticity,maxshare}`
  → use `puremacro.var.identify.*`.
- `puremacro.svar.estimate_var` → use `puremacro.var.estimate.estimate_var`.
- `puremacro.lp.lp_{jorda,iv,panel,panel_dk,state_dep,smooth,garch_state,garch_in_mean}`
  → use the prefix-free `puremacro.lp.*` siblings.
- Each emits one DeprecationWarning on import. Removal target: 0.43.0.

### Internal
- 9 SVAR tests migrated to var/identify/.
- experiment.py, gar/qar.py, teaching/bq_canonical.py pinned to canonical paths.
- inference/legacy/ kept alive privately as shim back-compat; targets
  retirement alongside the shim layer at 0.43.0.
- pyproject.toml: confirm filterwarnings doesn't promote DeprecationWarning
  to errors (no change expected, just verified).

### Not affected
- panel_svar.py (no canonical home yet — flagged as Phase-2.5 candidate).
- narrative.types legacy-series references (unrelated).
```

## Out of scope

- **`svar/panel_svar.py`** — no canonical home in `var/identify/`. Tracked as Phase-2.5; revisit when a panel-SVAR estimator lands under `var/identify/panel.py`.
- **Notebook re-execution and deep-import rewrites** in the 9 listed notebooks. Paired with the next paper-figure refresh at 0.43.0.
- **`inference/legacy/*` file-level deletion.** Scheduled for 0.43.0 alongside shim retirement.
- **CI / linter / Sphinx / type-checker.** Explicitly out per `CONTRIBUTING.md`.

## Acceptance

- Full pytest suite green with `N_deprecation` new passing tests.
- `pytest -W error::DeprecationWarning --ignore=tests/test_deprecation_warnings.py` passes.
- `tests/test_pyodide_compat.py` still green.
- `ARCHITECTURE.md` updated to mark the three consolidation candidates as "shim-and-deprecate at 0.42.0; removal at 0.43.0."
- 0.42.0 CHANGELOG entry above is committed.
- `puremacro/__init__.py::__version__` and `pyproject.toml::version` both read `0.42.0`.
- One-notebook smoke test passes on at least `R1_01_svar_menu.ipynb`.
