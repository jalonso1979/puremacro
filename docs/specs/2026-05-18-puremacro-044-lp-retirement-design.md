# puremacro 0.44.0 — LP shim retirement + notebook re-execution design

**Status:** approved 2026-05-18. Target release: **0.44.0**.

## Why

0.43.0 (`docs/specs/2026-05-18-puremacro-043-canonical-promotion-design.md`, shipped at commit `1bae0b2`) retired the entire `puremacro/svar/` package but explicitly deferred three follow-ups:

1. **`puremacro/lp/lp_*.py` (8 files)** still alive because `R1_02_lp_menu`, `R1_03_cross_country`, and `R1_05_publication` notebooks import legacy IRF function variants (`lp_irf`, `lp_panel_irf`, `lp_iv_irf`, `lp_smooth_irf`, `lp_state_dep_irf`) and result classes (`LPResult`, `PanelLPResult`, `LPIVResult`, `LPSmoothResult`, `LPGARCHStateResult`, `LPGIMResult`, `RegimeInteractionLPResult`) that have no canonical equivalents under the same names.
2. **`puremacro/inference/legacy/`** kept alive because `lp/lp_state_dep.py` and `lp/lp_smooth.py` import `block_bootstrap` from it.
3. **Body-rewrite + re-execution of 5 body-rewrite notebooks** (R1_01, R1_03, R1_04, R1_05, T_us_national) was deferred — source edits committed in 0.43.0 but `data/processed/panel_Q.parquet` was missing locally so re-execution was skipped.

0.44.0 closes all three out:

1. Body-rewrite `R1_02_lp_menu.ipynb` (the legacy lp menu demo) to a canonical `lp/*` menu — same chapter slot, same pedagogical role, new API.
2. Body-rewrite `R1_03_cross_country.ipynb` and `R1_05_publication.ipynb` off the remaining `lp_*` deep imports.
3. Build `data/processed/panel_Q.parquet`.
4. Re-execute the body-rewrite notebooks from BOTH releases: 0.43.0's (R1_01, R1_04, T_us_national) and 0.44.0's (R1_02, R1_03, R1_05), via their paired builders.
5. Delete `puremacro/lp/lp_*.py` (8 files), `puremacro/inference/legacy/` (10 files), and `puremacro/lp/garch_utils.py` if no callers remain after step 1.

`puremacro/regress/lp.py` stays deferred — independent pure-numpy implementation, not a re-export. Its retirement is a separate release once a canonical equivalent with the same signature ships.

## Pre-conditions

- 0.43.0 shipped (commit `1bae0b2` on `main`).
- `puremacro.var.identify.panel`, `puremacro.var.identify.maxshare.identify_maxshare`, `puremacro.lp.panel.lp_panel_regime_interaction`, `puremacro.lp.state_dep.lp_smooth_transition_irf` all canonical, all tested.
- Test baseline: 1277 passed, 10 pre-existing failed.

## Four phases (single release)

### Phase A — Body-rewrite the 3 lp-dependent notebooks

**A1. `R1_02_lp_menu.ipynb` + `tools/make_notebook_R1_02.py`**

Convert to canonical lp/* menu. Replace every legacy import:

| Legacy import | Canonical replacement |
|---|---|
| `from puremacro.lp.lp_jorda import lp_irf, LPResult` | `from puremacro.lp.jorda import lp_hac` (DataFrame return) |
| `from puremacro.lp.lp_panel import lp_panel_irf, PanelLPResult` | `from puremacro.lp.panel import panel_lp` |
| `from puremacro.lp.lp_iv import lp_iv_irf, LPIVResult, WeakInstrumentWarning` | `from puremacro.lp.iv import lp_iv` (DataFrame). `WeakInstrumentWarning` is in `puremacro.inference.weak_iv` if used. |
| `from puremacro.lp.lp_smooth import lp_smooth_irf, LPSmoothResult` | `from puremacro.lp.smooth import lp_smooth` |
| `from puremacro.lp.lp_state_dep import lp_state_dep_irf` | `from puremacro.lp.state_dep import lp_state_dep` |
| `from puremacro.lp.lp_garch_state import LPGARCHStateResult` | drop — canonical returns DataFrame |
| `from puremacro.lp.lp_garch_in_mean import LPGIMResult` | drop |

Cell bodies need rewrite where they access result-class attributes (`result.beta[h]`) — switch to DataFrame column access (`result.loc[result.h == h, "beta"].iloc[0]` or vectorized).

Update narrative text in markdown cells if it references "Result classes" by name; rewrite to mention "DataFrames" where appropriate. Preserve the section structure (`§1 Jordà LP`, `§2 Panel LP`, etc.).

**A2. `R1_03_cross_country.ipynb` + `tools/make_notebook_R1_03.py`**

Per Task 0 audit, R1_03 has 2 remaining legacy imports:
- `from puremacro.lp.lp_jorda import lp_irf` → `from puremacro.lp.jorda import lp_hac`
- `from puremacro.lp.lp_panel import RegimeInteractionLPResult  # legacy result type` → drop (canonical `lp_panel_regime_interaction` returns DataFrame; remove the `RegimeInteractionLPResult` reference from any type hints/isinstance checks).

Update call sites + result access. The notebook already uses canonical `mean_group_svar` (0.43.0); these LP imports are the last legacy holdovers.

**A3. `R1_05_publication.ipynb` + `tools/make_notebook_R1_05.py`**

Per audit, R1_05 has 1 remaining legacy import: `from puremacro.lp.lp_jorda import lp_irf`. Replace with `lp_hac` and rewrite the call site.

### Phase B — Build data + re-execute notebooks

**B1. Build `data/processed/panel_Q.parquet`**

```python
from puremacro.build_panel import build_all
build_all(refresh=False, fast=True)
```

This populates the panel cache. If fetchers are unreachable, document + skip (R1_01 re-execution is conditional). The cache file gates everything in B2 that uses `load_country(...)`.

**B2. Re-execute body-rewrite notebooks via paired builders**

Per memory pins `feedback_notebook_builders_paired` and `feedback_long_nbconvert_no_subagent`, run from the controller's background, never delegate to a subagent.

The 6 body-rewrite notebooks to re-execute (5 from 0.43.0 + 3 from 0.44.0 = 8, minus overlaps; R1_03 and R1_05 appear in both lists — they need re-execution only once with all 0.44.0 source edits applied):

1. `R1_methods/R1_01_svar_menu.ipynb` (0.43.0 svar edits already committed)
2. `R1_methods/R1_02_lp_menu.ipynb` (Phase A1 edits applied)
3. `R1_methods/R1_03_cross_country.ipynb` (0.43.0 + Phase A2 edits)
4. `R1_methods/R1_04_dsge_compare.ipynb` (0.43.0 edits)
5. `R1_methods/R1_05_publication.ipynb` (0.43.0 + Phase A3 edits)
6. `T_us_national.ipynb` (0.43.0 edits, no paired builder — direct nbconvert)

Per notebook: run `python tools/make_notebook_<X>.py` (or nbconvert for T_us). Inspect output cell diffs for sanity. If any cell raises an exception, stop and triage.

**Hazard:** memory pin `feedback_builder_clobbers_outputs` warns that builders strip existing outputs. That's actually the GOAL here (regenerating outputs against canonical) but it means the prior committed outputs are gone after re-execution. Diff review is essential.

### Phase C — Delete lp/lp_*.py + inference/legacy/

**C1. Pre-deletion gate**

```bash
git grep -E "puremacro\.lp\.lp_|puremacro\.inference\.legacy" -- ':!docs/' ':!CHANGELOG.md' ':!*.bak'
```

Expected after Phases A+B: zero hits outside `puremacro/lp/lp_*.py` itself and `puremacro/inference/legacy/` itself (i.e. all OUTSIDE callers migrated).

If hits remain, **STOP** — they need migration first.

**C2. Delete `puremacro/lp/lp_*.py`**

All 8 files:
- `lp_jorda.py`
- `lp_iv.py`
- `lp_panel.py`
- `lp_panel_dk.py`
- `lp_state_dep.py`
- `lp_smooth.py`
- `lp_garch_state.py`
- `lp_garch_in_mean.py`

**C3. Delete `puremacro/inference/legacy/`**

After C2 lands, the only callers of `inference/legacy/*` were inside the deleted `lp/lp_*.py` files. Verify:

```bash
git grep -E "from puremacro\.inference\.legacy|import puremacro\.inference\.legacy"
```

Expected: zero hits (after C2). Then `git rm -r puremacro/inference/legacy/`.

**C4. Decide `puremacro/lp/garch_utils.py`**

Per 0.43.0 ARCHITECTURE.md, `garch_utils.py` is a public-named helper. The canonical `lp/garch_state.py` and `lp/garch_in_mean.py` import from it. After Phase A, check:

```bash
git grep -E "from puremacro\.lp\.garch_utils|from \.garch_utils"
```

If only internal `puremacro/lp/*` callers remain (i.e. no notebook/tools caller), rename to `_garch_utils.py` (private). If external callers exist, keep as-is and add to the audit notes.

### Phase D — Release + verify

**D1. Update `ARCHITECTURE.md`**

Remove the "Known consolidation candidates" Phase-2.5 entries for `lp/lp_*.py` and `inference/legacy/`. The remaining candidate is `regress/lp.py` (independent implementation, its own future release).

Update the stability tier table to remove rows for the deleted files. Confirm `lp.panel.lp_panel_regime_interaction`, `lp.state_dep.lp_smooth_transition_irf`, `var.identify.panel.mean_group_svar`, `var.identify.maxshare.identify_maxshare` are listed as Stable.

**D2. CHANGELOG.md entry for 0.44.0**

Skeleton:

```markdown
## 0.44.0 — 2026-05-18

LP shim retirement + notebook re-execution. The lp/lp_*.py and
inference/legacy/ paths deferred in 0.43.0 are deleted. Three body-
rewrite notebooks (R1_02, R1_03, R1_05) migrated to canonical lp/*
APIs. All body-rewrite notebooks from 0.43.0 + 0.44.0 re-executed
against canonical.

### Removed (breaking)
- puremacro.lp.lp_* — all 8 files.
- puremacro.inference.legacy — entire directory.
- puremacro.lp.garch_utils (if no external callers) → renamed _garch_utils.

### Changed
- R1_02_lp_menu.ipynb body-rewritten as canonical lp/* menu. Preserves
  the chapter's pedagogical role (LP API reference) on the new surface.
- R1_03_cross_country.ipynb + R1_05_publication.ipynb migrated off the
  remaining lp_jorda + RegimeInteractionLPResult imports.
- 6 body-rewrite notebooks re-executed: R1_01, R1_02, R1_03, R1_04,
  R1_05, T_us_national. Outputs regenerated against canonical;
  pre-execution → post-execution diffs reviewed.

### Not affected (still deferred)
- puremacro/regress/lp.py — independent pure-numpy implementation.
  3 active callers in tools/. Its own follow-up release.
```

**D3. Version bump**

`0.43.0` → `0.44.0` in `__init__.py`, `pyproject.toml`, `tests/test_import.py`. Regenerate `tests/fixtures/public_api_snapshot.json` (lose all `lp.lp_*` entries).

**D4. Final verification gates**

1. `pytest tests/ -q` — green (1277 minus any lp/lp_*-test cases that get removed, plus any new R1_02-related tests).
2. `git grep -E "puremacro\.lp\.lp_|puremacro\.inference\.legacy" -- ':!docs/' ':!CHANGELOG.md'` — zero hits.
3. `pytest -W error::DeprecationWarning --ignore tests/test_pyodide_compat.py --ignore tests/test_public_api.py` — only the same pre-existing failures from 0.42.0/0.43.0 (skewt + narrative utcnow).
4. `pytest tests/test_pyodide_compat.py` — 2 passed.

## Risk register

| Risk | Mitigation |
|---|---|
| **R1_02 body-rewrite loses pedagogical value.** | Phase A1 explicitly preserves section structure (`§1 Jordà LP`, `§2 Panel LP`, etc.). Only the underlying API references change. Reviewer pass confirms the resulting notebook still functions as an LP menu. |
| **`data/processed/panel_Q.parquet` build fails** (fetcher network unreachable). | Document failure + skip R1_01 re-execution in Phase B2. The source edits stay committed; re-execution is non-blocking for the release. |
| **Notebook re-execution shifts outputs more than expected.** | Paper figures already verified independent of these 6 notebooks (per 0.43.0 spec). Diff review per notebook; commit per notebook so any single notebook can be reverted in isolation. |
| **Removing `LPGARCHStateResult` / `LPGIMResult` breaks isinstance checks** somewhere. | Phase A1 explicitly searches for and removes these references. Pre-deletion gate (C1) catches any survivors. |
| **`garch_utils.py` rename breaks an external caller.** | Pre-rename grep (Phase C4) gates. If hit, keep as `garch_utils.py` + document. |
| **`regress/lp.py` Phase-2.5 stayer** confuses future contributors. | 0.43.0 already added an inline `NOTE` banner at the top of the file. CHANGELOG re-states the deferral. |
| **R1_02's narrative cells reference the old legacy LP menu by name.** | Reviewer pass on the rewritten .ipynb's markdown cells; update text where it explicitly names the legacy API. |

## Out of scope

- `regress/lp.py` retirement — separate release.
- New estimators or new identification schemes.
- Pushing to `origin/main` — same local-only convention as 0.42.0/0.43.0 (the user pushes when they're ready).
- CI / linter / Sphinx docs — same exclusion as prior releases.
- `ProxySVARResult` axis fix (canonical inconsistency flagged in 0.43.0 final review) — bundle with future R1_04 cleanup.

## Acceptance

- All D4 verification gates pass.
- `git grep` for `puremacro.lp.lp_*` and `puremacro.inference.legacy` returns zero hits outside `docs/` and `CHANGELOG.md`.
- `puremacro/lp/lp_*.py` and `puremacro/inference/legacy/` directories no longer exist.
- 6 body-rewrite notebooks re-executed against canonical; outputs in git diff are sane.
- 0.44.0 CHANGELOG entry committed.
- `puremacro/__init__.py::__version__` and `pyproject.toml::version` both read `0.44.0`.
