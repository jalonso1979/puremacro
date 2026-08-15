# Architecture-consistency Phase 0 + 1 — design

**Status:** Phase 0+1 approved 2026-05-14. Phases 2–4 deferred to a later spec.

## Why

`puremacro/ARCHITECTURE.md` was written before Phase 5 (commits `c4482f2` → `03ecc41`, late Apr → early May 2026) absorbed ~22K LOC from `src/` into the package. The doc is now significantly out of date and the Pyodide-compatibility test it documents as load-bearing is failing.

## Audit findings (drift relative to ARCHITECTURE.md)

1. **Version mismatch.** `puremacro/__init__.py` says `__version__ = "0.40.0"` (matches CHANGELOG dated 2026-05-13); `pyproject.toml` still says `0.12.1`.
2. **`pypdf>=4.0` in runtime deps.** Used lazily inside `narrative/sources/_extractors.py`, but the Pyodide contract names only `numpy / scipy / pandas / matplotlib`. Should move to an optional-dependencies group.
3. **Six shippable modules import dev-only deps at top level**, breaking the Pyodide test:
   - `bartik/sensitivity.py` (`statsmodels.api`)
   - `build_panel.py` (`arch`)
   - `fetch/_seasonal.py` (`statsmodels.tsa.x13`)
   - `fetch/fred_states.py` (`arch`)
   - `sa/stl.py` (`statsmodels.tsa.seasonal.STL`)
   - `svar/identify_maxshare.py` (`statsmodels.tsa.api.VAR`)
4. **Four files under `teaching/`** also have hard imports (`garch_arch.py`, `lp_sm.py`, `panel_lm.py`, `var_sm.py`). These are intentionally statsmodels/linearmodels-backed teaching prototypes; they should be excluded from the Pyodide-test sweep the same way `examples/` already is.
5. **~30 modules absent from ARCHITECTURE.md's module map**, in three clusters:
   - data pipelines: `fetch/`, `build_panel.py`, `build_subnational_panel.py`, `bartik/`, `klems.py`, `bis_neer.py`, `long_panel.py`, `labor_share.py`, `vintages.py`, `_codes.py`, `_http.py`, `cache.py`
   - estimators: `cointegration_modern.py`, `korv_gmm.py`, `factor.py`, `midas.py`, `spectral.py`, `synthetic_control.py`, `wavelet.py`, `realized_vol.py`, `scale.py`
   - sub-packages: `uncertainty/`, `instruments/`, `regress/`, `sa/`, `teaching/`, `plotting/`, `svar/`, `sigma/`
6. **One canonical duplicate, deferred to Phase 2:** `svar/` is a less-complete sibling of `var/identify/`, but tests still use `svar/`. Not in Phase 0+1 scope; flagged in revised ARCHITECTURE.md as a known consolidation target.

## Phase 0 — stop the bleed (zero behavior change)

### Edits

- `pyproject.toml`: bump `version` to `0.40.0`; move `pypdf>=4.0` from `[project.dependencies]` to `[project.optional-dependencies.narrative]`.
- Six shippable files: move forbidden imports from module top to inside the function that uses them. Add a one-line comment `# lazy import: Pyodide contract — see ARCHITECTURE.md` at the point of use.
- `tests/test_pyodide_compat.py`: add `"puremacro.teaching"` to `_SKIP_PREFIXES`. Document it in the file's module docstring.
- `fetch/_seasonal.py`: the module-top `import as _x13_arima_analysis` exists for test monkeypatching; replace with an `_x13()` accessor function and have callers / monkeypatches target the accessor.

### Acceptance

- `pytest tests/test_pyodide_compat.py -v` passes both tests.
- Full suite `pytest` does not regress (some tests may already be flaky / data-dependent — we accept the pre-existing pass/fail set).

## Phase 1 — architecture realignment (docs-only)

Rewrite `ARCHITECTURE.md` to faithfully describe the Phase 5 reality:

1. Expand the **Module map** with the three new buckets: `data/loaders`, `data/construction`, plus the new top-level estimators and sub-packages. Group by intent, not alphabet.
2. Extend the **Stability-tier table** with rows for every newly-documented module. Mark Phase 5 additives as **Stable** when they have unit tests, **Best-effort** when they wrap a network resource, **Experimental** when no tests cover them.
3. Add the six Phase 0 lazy-loaded imports to a refreshed **Known leaks (now lazy)** subsection, with one-line context each.
4. Document the legitimate distinctions (so future contributors don't merge them):
   - `plot.py` (Pyodide-pure helpers) vs `plotting/` (full library styling).
   - `regime_dates.py` (constants) vs `regimes.py` (utilities).
   - `volatility/sigma.py` (canonical MATLAB port) vs `sigma/sigma_numpy.py` (minimal teaching stub).
   - `svar/` (legacy path, tests still depend on it; Phase 2 candidate) vs `var/identify/` (the canonical 0.4.0+ result-object path).
5. Document that `puremacro.teaching` and `puremacro.examples` are research side-channels excluded from the Pyodide promise.

## Out of scope

- Phase 2 (`svar/` → `var/identify/` migration), Phase 3 (8 result-object violations), Phase 4 (sigma consolidation). To be picked up after Phase 0+1 lands.
- README.md rewrite. Touch only `ARCHITECTURE.md` in this iteration.
