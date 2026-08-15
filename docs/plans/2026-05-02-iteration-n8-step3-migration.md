# Iteration N+8 Step 3 — Result-object migration sweep + public-API freeze + docstring pass

> **Working location:** This plan lives in the **WIP fork** at `uncertainty_examples/puremacro.step3-wip/`. All implementation runs in the WIP. The live `uncertainty_examples/puremacro/` continues to serve research scripts unaffected. Supersede happens at the very end (Task F) once WIP tests are fully green.

**Goal:** Migrate ~14 public functions across `var/identify`, `did`, `inference/weak_iv`, `garch`, `dsge/klein`, `gar/skewt` to result-object dataclasses per the standard set in iteration N+8 step 1. Add the public-API freeze test. Polish docstrings on the four 0.3.0 modules.

**Architecture:** The result-object standard from `ARCHITECTURE.md` (frozen dataclass, `<Method>Result` naming, `_results.py` per subpackage) propagates here. DataFrame returns from `lp/*` are explicitly EXCLUDED from migration — a DataFrame with named columns is already a structured result; wrapping it in a dataclass would be ceremony without value. This carve-out is added to the standard.

**Spec reference:** `docs/specs/2026-05-02-iteration-n8-design.md` § 2 (Section A — migration sweep), and the inventory below.

---

## Inventory (as of 2026-05-02, against WIP at 276 tests)

### Direct migration targets (must do — return 3+ fields, not yet a dataclass)

**var/identify/ (6 functions, 1 frozen conversion):**

| Function | Current | Result class | Notes |
|---|---|---|---|
| `cholesky_svar` | `(point, lo, hi)` 3-tuple | `CholeskySVARResult` | bootstrap with non-PD failure-counter; preserve warn-above-5% behaviour |
| `bq_svar` | `(point, lo, hi)` 3-tuple | `BQSVARResult` | cumsum-ed IRF; same shape as Cholesky |
| `sign_restriction_svar` | `(median, lo, hi)` 3-tuple | `SignRestrictionResult` | uses median over admissible draws, not point |
| `gk_robust_bands` | dict with 4 keys | `GKRobustBandsResult` | Giacomini-Kitagawa; both `gk_robust_bands` and `gk_robust_bands_from_gibbs` share this class |
| `non_gaussian_svar` | dict with 5 keys | `NonGaussianSVARResult` | LMS 2017 FastICA; has `B0`, `Q`, `kurtosis`, `irf`, `ordering_by_kurt` |
| `sign_zero` | dict with 3 keys OR `None` | `SignZeroResult` | **fix dict-or-None asymmetry**: result always returned, with `success: bool` flag |
| `HeteroResult` (Rigobon) | already dataclass, NOT frozen | `HeteroResult` | flip to `@dataclass(frozen=True)` |

**did/ (4 functions, all currently dict):**

| Function | Result class | Fields |
|---|---|---|
| `callaway_santanna` | `CallawaySantannaResult` | `att_gt: pd.DataFrame`, `att_event_study: pd.DataFrame`, `att_overall: float` |
| `sun_abraham` | `SunAbrahamResult` | same shape as CS |
| `borusyak_jaravel_spiess` | `BorusyakJaravelSpiessResult` | `tau_it: pd.DataFrame`, `att_event_study: pd.DataFrame`, `att_overall: float` |
| `synthetic_did` | `SyntheticDiDResult` | `tau`, `omega: pd.Series`, `lambda_w: pd.Series`, `se`, `lo`, `hi`, `treatment_time` (note: rename `lambda` → `lambda_w` since `lambda` is reserved) |

**inference/weak_iv.py (1 dict-return + missing `__all__`):**

| Function | Current | Result class |
|---|---|---|
| `anderson_rubin_test` | dict with 5 keys (`stat`, `p_value`, `df_num`, `df_den`, `residual_ss`) | `ARTestResult` |

Also: **add `__all__ = [...]` to `inference/weak_iv.py`** (currently missing).

**garch/ (2 fit functions, both dict-return):**

| Function | Current | Result class |
|---|---|---|
| `garch11_fit` | dict with 7 keys (`omega`, `alpha`, `beta`, `sigma: pd.Series`, `loglik`, `converged`, `persistence`) | `GARCH11Result` |
| `dcc_fit` | dict with 8 keys | `DCCResult` |

Callers needing updates: `lp/garch_state.py`, `lp/garch_in_mean.py` (both consume `garch11_fit`).

**dsge/, gar/ — frozen conversions only:**

- `dsge/klein.KleinSolution` — already a dataclass; flip to `@dataclass(frozen=True)`.
- `gar/skewt.SkewTFit` — same.

### Already compliant (skip)

- `var/identify/proxy.py:ProxySVARResult` (done in step 1).
- `puremacro.hfi.JKResult` (done in step 1).
- LP DataFrame returns (`lp_hac`, `panel_lp`, `lp_iv`, etc.) — DataFrames are already structured; **carve-out** documented in the standard.

### Excluded from this step (deferred)

- `nowcast/*` — single-DataFrame returns; no migration.
- `volatility/*` — same.

---

## Carve-out for the standard

Add a paragraph to `ARCHITECTURE.md`'s "Result-object standard" section:

> **DataFrame carve-out.** Functions that return a single `pandas.DataFrame` with named columns (e.g., `lp_hac`, `panel_lp`, every `lp/` estimator) do NOT need to wrap that DataFrame in a dataclass. The DataFrame is already a structured, self-documenting result. The standard applies to functions returning plain tuples, plain dicts, or unwrapped multi-array returns.

---

## Execution strategy

The migration pattern is the same for every function: create the dataclass in `<subpackage>/_results.py`, change the function's return statement, update internal callers and tests, expand `__all__`. Because the pattern repeats, one large subagent (Task C) handles the migration sweep. Then a second subagent (Task D) writes the freeze test, then a third (Task E) does docstrings, then a final verification + supersede (Task F).

---

## Task C — Migration sweep (single subagent, WIP only)

The subagent gets:
- This inventory.
- The result-object standard.
- The pattern: define dataclass → migrate function → update callers → update tests → add to `__all__`.
- TDD discipline: for each function, the existing tests should still pass after migration (with adjusted assertions); add at least one new test asserting `isinstance(res, <Result>)` and `frozen` semantics.

**Files created:**
- `puremacro/var/identify/_results.py` — extend with 6 new dataclasses (joining existing `ProxySVARResult`).
- `puremacro/did/_results.py` — new file with 4 dataclasses.
- `puremacro/inference/_results.py` — new file with `ARTestResult`.
- `puremacro/garch/_results.py` — new file with `GARCH11Result`, `DCCResult`.

**Files modified:**
- 6 functions in `var/identify/*.py` (cholesky, bq, sign, sign_robust, non_gaussian, sign_zero).
- `var/identify/hetero.py` (frozen=True on `HeteroResult`).
- 4 functions in `did/*.py`.
- `inference/weak_iv.py` (`anderson_rubin_test` migration + add `__all__`).
- `garch/fit.py`, `garch/dcc.py`.
- `dsge/klein.py` (frozen=True on `KleinSolution`).
- `gar/skewt.py` (frozen=True on `SkewTFit`).
- All `__init__.py` files for the subpackages above (re-export the new result classes; expand `__all__`).
- Internal callers: `lp/garch_state.py`, `lp/garch_in_mean.py`, `did/sun_abraham.py` (calls `callaway_santanna`).
- Examples that unpack the migrated dicts/tuples — search and update: `examples/bloom2009.py`, `examples/svariv_mertens_ravn.py` (some already updated for proxy_svar), and any did/ examples.
- Tests under `tests/test_var/`, `tests/test_did/`, `tests/test_inference/`, `tests/test_garch/` — adjust assertions from dict/tuple access to attribute access; add isinstance + frozen tests for each new result class.

**Acceptance:** all 276 tests pass in the WIP.

**Risk register:**
- The `lambda` → `lambda_w` rename in `synthetic_did` is a real API change for any caller that does `res["lambda"]`. Search for it.
- `sign_zero` returning `None` on no-admissible-draws is a UX choice some callers may rely on. The new contract returns a result with `success=False, n_draws_used=0, B0=None, Q=None`. Update any caller doing `if res is None`.
- `garch11_fit` is consumed by `lp/garch_state` and `lp/garch_in_mean`. Both unpack `res["sigma"]`. After migration: `res.sigma`. Surgical edit.
- DCC fit is heavier — has 8 fields including `R: ndarray (T, n, n)`, `H: ndarray (T, n, n)`. Make sure no test asserts on dict-iteration order.

---

## Task D — Public-API freeze test (single subagent, WIP only)

**File created:** `tests/test_public_api.py`.

**Spec:**

```python
"""Public-API freeze test.

Snapshots two things:
  1. ``__all__`` of every shippable subpackage.
  2. Field names of every public ``<MethodName>Result`` dataclass.

If either drifts, this test fails loudly and prints a diff. Regenerate the
snapshot deliberately when intentional API changes happen — never silently.
"""
import dataclasses
import importlib
import json
import pkgutil
from pathlib import Path

import pytest
import puremacro

SNAPSHOT = Path(__file__).parent / "fixtures" / "public_api_snapshot.json"


def _walk_subpackages():
    """Yield (qualname, module) for every shippable subpackage."""
    skip_prefixes = ("puremacro.examples", "puremacro.tests")
    for finder, name, is_pkg in pkgutil.walk_packages(
        puremacro.__path__, prefix="puremacro."
    ):
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        # Skip experimental network/LLM modules — see ARCHITECTURE.md
        if name.startswith("puremacro.narrative.sources"):
            continue
        if name == "puremacro.narrative.scoring.llm":
            continue
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        yield name, mod


def _collect_current_api():
    api = {"all": {}, "result_classes": {}}
    for name, mod in _walk_subpackages():
        if hasattr(mod, "__all__"):
            api["all"][name] = sorted(mod.__all__)
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            attr = getattr(mod, attr_name)
            if dataclasses.is_dataclass(attr) and isinstance(attr, type):
                # only count classes defined in this module (avoid double-counting re-exports)
                if attr.__module__ != name:
                    continue
                api["result_classes"][f"{name}.{attr_name}"] = sorted(
                    f.name for f in dataclasses.fields(attr)
                )
    return api


def test_public_api_matches_snapshot():
    if not SNAPSHOT.exists():
        pytest.fail(
            f"No snapshot at {SNAPSHOT}. Generate with:\n"
            f"  python -c \"from tests.test_public_api import _collect_current_api; "
            f"import json; print(json.dumps(_collect_current_api(), indent=2))\" "
            f"> {SNAPSHOT}"
        )
    expected = json.loads(SNAPSHOT.read_text())
    current = _collect_current_api()
    if current != expected:
        diff_lines = []
        for key in ("all", "result_classes"):
            added = set(current[key]) - set(expected[key])
            removed = set(expected[key]) - set(current[key])
            for k in sorted(added):
                diff_lines.append(f"  + {key}.{k} = {current[key][k]}")
            for k in sorted(removed):
                diff_lines.append(f"  - {key}.{k} (was {expected[key][k]})")
            for k in sorted(set(current[key]) & set(expected[key])):
                if current[key][k] != expected[key][k]:
                    diff_lines.append(
                        f"  ~ {key}.{k}: {expected[key][k]} -> {current[key][k]}"
                    )
        pytest.fail(
            "Public API drift detected:\n"
            + "\n".join(diff_lines)
            + "\n\nRegenerate snapshot only if the drift is intentional."
        )
```

**Snapshot file:** `tests/fixtures/public_api_snapshot.json`. Generated once after migration sweep is complete (not before — otherwise it locks in the pre-migration shape).

**Workflow:** subagent (a) writes `test_public_api.py`, (b) runs `_collect_current_api()` to generate the initial snapshot, (c) writes the snapshot to disk, (d) re-runs the test to confirm green.

**Also in this task:**
- Update `CHANGELOG.md` 0.4.0 block with the migration list.
- Update `ARCHITECTURE.md`: add the DataFrame carve-out paragraph; bump the result-object standard section's "0.4.0 release migrates existing 3+ field returns" line to past-tense.

---

## Task E — Docstring + type-hint pass on the four 0.3.0 modules

**Modules:** `puremacro/volatility/`, `puremacro/nowcast/`, `puremacro/gar/`, `puremacro/did/` (now with the new result classes from Task C).

**Pattern per public function:**

1. Top-line summary (1 sentence, imperative-mood ok).
2. Multi-paragraph description if non-obvious.
3. NumPy-style **Parameters** section: name + type + description.
4. NumPy-style **Returns** section: name + type + description. For result dataclasses, document the dataclass class.
5. **References** section if the function implements a published method.
6. Type hints on the signature: `def f(x: np.ndarray, h: int = 8) -> tuple[np.ndarray, np.ndarray]: ...`. Use `np.ndarray`, `pd.DataFrame`, `pd.Series`, builtins. Use `from __future__ import annotations` if not already.

**Out of scope** (do NOT touch in this pass):
- Function bodies — pure docstring/signature work.
- Examples / tests.
- Modules outside the four listed.

**Acceptance:** every public function in the four modules has Parameters/Returns/References. Type hints on signatures match the actual code paths.

---

## Task F — Verify WIP green; supersede live puremacro/

1. Run full suite in WIP: `cd <WIP> && PYTHONPATH=$PWD pytest tests/ -q --tb=short`. Must be green.
2. Run public-API freeze test: `pytest tests/test_public_api.py -v`. Must be green.
3. Run Pyodide-compat: `pytest tests/test_pyodide_compat.py`. Must be green.
4. Smoke-test the live examples: `python -m puremacro.examples.svariv_mertens_ravn`, `python -m puremacro.examples.narrative_ramey_2011` (with WIP on `PYTHONPATH`). Must run cleanly.
5. **Supersede:**
   - Snapshot live: `mv ../puremacro ../puremacro.before-step3-backup`.
   - Move WIP into place: `mv ../puremacro.step3-wip ../puremacro`.
   - Reinstall editable: `cd ../puremacro && pip install -e . --quiet` (so `__pycache__`/.egg-info refresh).
   - Re-run pytest at the new live location to confirm: `pytest tests/ -q --tb=short`.
6. Confirm imports resolve to the new location: `python -c "import puremacro, inspect; print(inspect.getfile(puremacro))"`.
7. Once user confirms the new live works in their workflow, delete `../puremacro.before-step3-backup`. (Do NOT delete in this task — leave the backup until user gives explicit OK.)
