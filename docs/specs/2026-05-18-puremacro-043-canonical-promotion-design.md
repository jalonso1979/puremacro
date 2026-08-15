# puremacro 0.43.0 — Canonical promotion + shim retirement design

**Status:** approved 2026-05-18. Target release: **0.43.0**.

## Why

Phase 2 (`docs/specs/2026-05-17-puremacro-phase2-consolidation-design.md`, shipped at 0.42.0) converted the legacy `svar/*` modules to thin DeprecationWarning shims and applied Phase-2.5 deferral banners to `svar/panel_svar.py`, `svar/identify_maxshare.py`, the 8 `lp/lp_*.py` files, and `lp/garch_utils.py`. The four `inference/legacy/{bootstrap, wild_bootstrap, block_bootstrap, weak_iv}.py` files carry retirement notes but stay alive because canonical `var/identify/*` still imports from them.

0.43.0 closes this out:

1. Promote `panel_svar` and `identify_maxshare` to canonical `var/identify/*`.
2. Harmonize the 8 `lp/lp_*.py` signatures with their canonical siblings, rewriting all callers (CONTRIBUTING.md: "rename freely; update consumers in the same commit").
3. Migrate canonical `var/identify/*` off `inference/legacy/*` so the legacy directory can go.
4. Re-port the 9 deep-import notebooks (paired with their `tools/make_notebook_*.py` builders where applicable).
5. Delete the entire shim layer: `puremacro/svar/`, `puremacro/lp/lp_*.py`, `puremacro/inference/legacy/`.

## Pre-conditions (already verified during context survey)

- **Paper figures are independent of the 9 affected notebooks.** `docs/paper/regime_uncertainty.tex` and `docs/paper/equipment.tex` `\includegraphics` paths reference T14, T7c, T17, T10 chapter families — none of R1_methods, R2_subnational, T5_research_lab, or T_us_national. Re-execution will not shift paper figures.
- **All Phase-2 shims are exercised**: `tests/test_deprecation_warnings.py` parametrizes over 6 shim modules; all green at 0.42.0.
- **CONTRIBUTING.md pre-1.0 policy applies**: "rename freely; update consumers in the same commit". The user explicitly picked option 1 (rewrite all callers to canonical kwargs) during 0.43.0 brainstorming.

## Four phases (single release)

### Phase A — Promote canonical surfaces

**A1. Promote `svar/panel_svar.py` → `puremacro/var/identify/panel.py`**

The legacy `svar/panel_svar.py` exports:
- `class PanelSVARResult` (mutable dataclass)
- `def mean_group_svar(...)` — per-country SVAR + averaged IRFs

Port to canonical with these adjustments:
- `@dataclass(frozen=True) PanelSVARResult` in `puremacro/var/identify/_results.py` (alongside `CholeskySVARResult`, etc.).
- Field set: `irfs_per_country` (dict[str, ndarray]), `irf_mean`, `irf_lower`, `irf_upper`, `n_countries`, `n_boot`, `ci`. Axes `(H+1, n, n)` per the canonical convention.
- `def mean_group_svar(...)` in `var/identify/panel.py`. Internally routes each country's SVAR through canonical `var/identify/{cholesky, bq, sign, proxy, hetero, maxshare}` rather than through `inference/legacy/*` like the legacy did.
- Add `.summary()` method.
- Re-export from `puremacro/var/identify/__init__.py`.

**A2. Promote `svar/identify_maxshare.py` → extend `puremacro/var/identify/maxshare.py`**

The canonical `var/identify/maxshare.py` (85 LOC) currently exports only the low-level `maxshare(...)` returning `B0: ndarray` and `news_maxshare(...)`. The legacy `svar/identify_maxshare.py` (285 LOC) wraps these with:
- `class MaxShareResult` (mutable dataclass) with `B0`, `irfs`, `fevd`, `lower`, `upper`, `point`.
- `def identify_maxshare(...)` — full pipeline: estimate VAR, identify, compute FEVD, bootstrap bands.

Port:
- `@dataclass(frozen=True) MaxShareResult` in `var/identify/_results.py`. Axes `(H+1, n, n)`.
- `def identify_maxshare(...)` in `var/identify/maxshare.py` alongside the existing low-level helpers. Internally routes its bootstrap through `inference.wild_bootstrap` or `inference.bootstrap` (whichever the canonical svar identification schemes already use), NOT `inference.legacy.*`.
- Re-export from `var/identify/__init__.py`.
- Keep the existing low-level `maxshare(...)` and `news_maxshare(...)` for callers that just need `B0`.

**A3. Harmonize 8 `lp/lp_*.py` signatures**

For each pair below, the per-file work is: (a) decide which kwargs the canonical should use, (b) audit every caller of the legacy file via `git grep` (in-repo + `tools/` + notebooks), (c) rewrite each caller, (d) make the legacy file a thin DeprecationWarning shim of the canonical.

| Legacy | Canonical | Headline mismatch |
|---|---|---|
| `lp/lp_jorda` | `lp/jorda` | Function name (`lp_irf` vs `lp_hac`); return type (`LPResult` dataclass vs `pd.DataFrame`). |
| `lp/lp_iv` | `lp/iv` | Function name (`lp_iv_irf` vs `lp_iv`); return type; `WeakInstrumentWarning` re-export. |
| `lp/lp_panel` | `lp/panel` | `lp_panel_regime_interaction` has no canonical equivalent — port it to `lp/panel.py` first. |
| `lp/lp_panel_dk` | `lp/panel_dk` | Kwargs mismatch (`outcome=/shock=/unit_col=/date_col=/dk_lag=/ci_level=` vs `y=/x=/entity_level=/time_level=/n_lags=/alpha=`). 8+ caller sites in `tools/`. |
| `lp/lp_state_dep` | `lp/state_dep` | Function name + return type; `lp_smooth_transition_irf` has no canonical equivalent — port it. |
| `lp/lp_smooth` | `lp/smooth` | Function name + return type. |
| `lp/lp_garch_state` | `lp/garch_state` | Same function name (`lp_garch_state`); return type (`LPGARCHStateResult` vs DataFrame). Audit if any caller uses the dataclass. |
| `lp/lp_garch_in_mean` | `lp/garch_in_mean` | Same function name (`lp_garch_in_mean`); return type (`LPGIMResult` vs DataFrame). |

The canonical should win on naming and return type by default (canonical convention is DataFrame for LP). Two exceptions where the legacy name is semantically richer or the canonical surface is missing are tracked above:
- `lp_panel_regime_interaction` (no canonical) — port from legacy to `lp/panel.py`.
- `lp_smooth_transition_irf` (no canonical) — port from legacy to `lp/state_dep.py`.

After A3, every `lp/lp_*.py` file is a thin shim emitting DeprecationWarning. Same pattern as Phase 2's svar shims.

### Phase B — Migrate callers off legacy

**B1. In-repo non-test callers**

- `tools/make_notebook_R1_01.py` — rewrites 7 svar imports to canonical (`var.estimate`, `var.identify.{cholesky, bq, sign, proxy, hetero, maxshare, panel}`). Note `identify_maxshare` becomes `var.identify.maxshare.identify_maxshare`; `mean_group_svar` becomes `var.identify.panel.mean_group_svar`.
- `tools/make_notebook_R1_02.py`, R1_03, R1_04, R1_05 — per-file audit + rewrite.
- `tools/make_notebook_R2_01.py` — rewrites `lp.lp_panel_dk` import + kwargs (8 call sites identified during Phase 2 review).
- `tools/make_notebook_R2_02.py` — rewrites `lp.lp_iv` → `lp.iv` with name/return-type change.
- Other `tools/` callers: per-file audit. Run `git grep -l "puremacro.svar\\|puremacro.lp.lp_\\|puremacro.inference.legacy" tools/` and rewrite each match.
- `puremacro/teaching/bq_canonical.py` — its local `bq_svar` adapter (from Phase 2) becomes unnecessary if call sites switch to `BQSVARResult` attribute access. Decision: **delete the adapter and rewrite the two call sites** to use `result.irf_point[h, i, j]` style. Consistent with "no adapters" policy at 0.43.0.

**B2. Notebook `.ipynb` files (9 notebooks)**

For each, classify upfront as **rename-only** or **body-rewrite**:

- **Rename-only**: only the import lines change; cell bodies don't. Edit both `.ipynb` AND paired builder `make_notebook_*.py` simultaneously (memory pin: notebooks ↔ builders paired). Do NOT re-execute — per memory pin `feedback_builder_clobbers_outputs`, renames don't need a rebuild.
- **Body-rewrite**: kwargs change or return type changes inside notebook cells. Edit both files. **Must re-execute** after edit.

Per-notebook classification, made during planning:

| Notebook | Paired builder | Likely class | Why |
|---|---|---|---|
| `R1_methods/R1_01_svar_menu.ipynb` | `tools/make_notebook_R1_01.py` | body-rewrite | svar shim returns 3-tuple `(point, lo, hi)` (n,n,H+1); canonical returns `CholeskySVARResult` (H+1,n,n). Cell bodies that index `point[i, j, h]` need rewrite to `result.irf_point[h, i, j]`. |
| `R1_methods/R1_02_lp_menu.ipynb` | `tools/make_notebook_R1_02.py` | body-rewrite | `lp.lp_smooth_irf` → DataFrame (canonical) — column access changes. |
| `R1_methods/R1_03_cross_country.ipynb` | `tools/make_notebook_R1_03.py` | TBD per audit | depends on which lp_* functions it uses. |
| `R1_methods/R1_04_dsge_compare.ipynb` | `tools/make_notebook_R1_04.py` | TBD per audit | likely svar callers — confirm. |
| `R1_methods/R1_05_publication.ipynb` | `tools/make_notebook_R1_05.py` | body-rewrite | uses svar 3-tuple + lp_* dataclasses per Phase 2 audit. |
| `R2_subnational/R2_01_panels_and_data.ipynb` | `tools/make_notebook_R2_01.py` | body-rewrite | `panel_lp_dk` kwargs change (`outcome=` → `y=`, etc.) — confirmed in Phase 2 lp_panel_dk revert. |
| `R2_subnational/R2_02_lp_iv_bartik.ipynb` | `tools/make_notebook_R2_02.py` | body-rewrite | `lp_iv_irf` returns `LPIVResult`; canonical `lp_iv` returns DataFrame. |
| `T5_research_lab.ipynb` | **no builder** | TBD per audit | direct `.ipynb` edit. |
| `T_us_national.ipynb` | (check for builder) | TBD per audit | direct `.ipynb` edit if no builder. |

The TBD entries are resolved during the planning phase (Task 0 of the plan: per-notebook audit).

### Phase C — Migrate canonical off `inference/legacy/*`

**C1. Audit canonical `var/identify/*` imports of `inference/legacy/*`**

Known imports (from Phase 2 inspection):

- `var/identify/cholesky.py` imports `..inference.legacy.bootstrap.residual_bootstrap_var`.
- `var/identify/bq.py` imports `..inference.legacy.bootstrap._irf_from_var`.
- `var/identify/proxy.py` imports `..inference.legacy.wild_bootstrap.wild_bootstrap_var`.
- `var/identify/sign.py` imports `..inference.legacy.bootstrap._irf_from_var`.
- `var/identify/hetero.py` imports `..inference.legacy.bootstrap._irf_from_var`.
- `var/identify/maxshare.py` (after A2): may add bootstrap import.
- Phase A's new `var/identify/panel.py` may import too.

For each `inference.legacy.<x>` import:

1. Check whether `puremacro/inference/<x>.py` already exists and has the same function.
2. If yes (byte-identical or close): change the canonical import to point at the non-legacy version; delete `inference/legacy/<x>.py`.
3. If no: rename `inference/legacy/<x>.py` to `puremacro/inference/_<x>_internal.py` (or merge with `inference/<x>.py` if the canonical exists but differs). Update the imports.
4. The 6 byte-identical shim files already in `inference/legacy/*` (from 0.41.0 consistency pass: `lp_block_bootstrap`, `moving_block_bootstrap`, `newey_west`, `pesaran_cce`, `swamy_test`, `balanced_panel`) all delete with no replacement — they were `from puremacro.inference.<x> import *` re-exports.
5. The 4 "legitimately different" files (`bootstrap.py`, `wild_bootstrap.py`, `block_bootstrap.py`, `weak_iv.py`) are merged with their non-legacy siblings or promoted to canonical with no `legacy` prefix.

**C2. Delete `puremacro/inference/legacy/` entirely**

Once C1 lands, the directory has zero remaining files (or is replaced by non-legacy promoted versions). Remove the directory.

### Phase D — Delete shim layer + verify

**D1. Delete every Phase-2 shim file**

Files to delete (not turn into a shim — delete entirely):

- `puremacro/svar/identify_cholesky.py`
- `puremacro/svar/identify_bq.py`
- `puremacro/svar/identify_sign.py`
- `puremacro/svar/identify_proxy.py`
- `puremacro/svar/identify_heteroskedasticity.py`
- `puremacro/svar/identify_maxshare.py` (after A2 + B1 migrate callers)
- `puremacro/svar/panel_svar.py` (after A1 + B1 migrate callers)
- `puremacro/svar/estimate_var.py`
- `puremacro/svar/__init__.py`
- All 8 `puremacro/lp/lp_*.py` files (after A3 + B1 migrate callers)
- `puremacro/lp/garch_utils.py` → promote to `puremacro/lp/_garch_utils.py` (private helper) and update the two callers (`lp/garch_state.py`, `lp/garch_in_mean.py`).
- `puremacro/inference/legacy/` (after C1).
- `puremacro/svar/` directory itself (now empty).

Test infrastructure deletes:
- `tests/test_deprecation_warnings.py` (no shims left to test).
- `tests/test_shim_shape_preservation.py` (no shape translations left).

**D2. Re-execute body-rewrite notebooks**

For each notebook classified body-rewrite in B2:

1. If it has a paired builder, run `python tools/make_notebook_<x>.py` to regenerate. This handles the cell-body re-execution.
2. If it has no paired builder (T5, possibly T_us_national), run `jupyter nbconvert --to notebook --execute <path> --inplace --ExecutePreprocessor.timeout=900`.
3. Per memory pin `feedback_long_nbconvert_no_subagent`, all of D2 runs in controller background, never delegated.
4. Pre-step for R1_01: ensure `data/processed/panel_Q.parquet` exists. If absent, run `python -c "from puremacro.build_panel import build_all; build_all(refresh=False)"` from the repo root. If fetchers fail (network issue), defer R1_01 re-execution and document.

### §3 Verification gates (before tagging 0.43.0)

1. **`pytest tests/ -q`** — green at the new count (1274 − ~11 [shim tests deleted] + N [new PanelSVARResult/MaxShareResult tests]).
2. **`pytest -W error::DeprecationWarning --ignore tests/test_pyodide_compat.py`** — green. No remaining shim emits warnings; they're deleted, not silenced. Pyodide compat excluded only because the sweep deliberately imports lots of modules.
3. **`pytest tests/test_pyodide_compat.py`** — green; new canonical files introduce no forbidden imports.
4. **`git grep -E "puremacro\\.svar\\.|puremacro\\.lp\\.lp_|puremacro\\.inference\\.legacy" -- ':!docs/' ':!CHANGELOG.md' ':!*.bak'`** — zero hits. The strongest "no callers on dead paths" gate.
5. **Each `tools/make_notebook_*.py` parses** + the import section runs end-to-end on a 10-row toy dataset (smoke test, not full re-execution).
6. **One full end-to-end notebook re-execution** for `R1_methods/R1_01_svar_menu.ipynb` (requires `data/processed/panel_Q.parquet` per §2-D2).
7. **`tests/test_public_api.py`** — snapshot regenerated for `PanelSVARResult` and `MaxShareResult`.

### §4 Risk register

| Risk | Mitigation |
|---|---|
| **Notebook re-execution shifts outputs** (memory pin `feedback_builder_clobbers_outputs`). | Paper figures verified independent of these 9 notebooks. Each notebook classified upfront (B2 table). Rename-only notebooks skip re-execution; body-rewrite notebooks have their pre-execution outputs preserved in git for diff review. |
| **`data/processed/panel_Q.parquet` missing** for R1_01. | Pre-execution step in plan (§2-D2). If fetchers unreachable, R1_01 re-execution deferred + documented; the import-rename portion still ships. |
| **`inference/legacy/{bootstrap,wild_bootstrap}.py` differ subtly** from `inference/<name>.py` siblings. | Per-pair `diff` audit during C1. Tests for each canonical SVAR scheme catch behavioural drift. |
| **8 LP callers per file × 8 files = up to 64 call-site edits in `tools/`**. | Per-file audit + mechanical `sed` where the kwarg rename is local; manual review where semantics shift. Each `tools/` script gets its own commit so a bad rename can be reverted in isolation. |
| **TBD notebook classifications in B2** (R1_03, R1_04, T5, T_us_national). | Resolved during planning (Task 0 of the implementation plan: per-notebook audit). The spec captures the policy; the plan does the per-file work. |
| **Re-executing R1_01 / R2_01 / R2_02 may exceed 5 min each**. | Per memory pin, controller-only background execution. Implementation plan's Task 10 (or equivalent) does these sequentially with explicit reporting. |
| **Public-API snapshot regenerates** might mask a real new addition. | The snapshot test is updated in a single commit alongside the dataclass additions; reviewer pass verifies the new entries match `PanelSVARResult`/`MaxShareResult` fields. |
| **`teaching/bq_canonical.py` adapter** removal might break downstream | `teaching/` is excluded from the Pyodide sweep + has limited test coverage. Verify by running its module-level import + one toy `bq_gdp_urate` call. |

### §5 Out of scope

- **New estimators or new identification schemes** — pure consolidation.
- **Pushing to `origin/main`** — local-only convention, same as 0.42.0.
- **Other Phase-2.5 files outside the lp_*/svar/inference scope** (e.g., `lp/_panel_helpers.py` rename considerations) — separate spec if needed.
- **CI / linter / Sphinx docs** — same exclusion as Phase 2.
- **Notebook deep imports beyond the 9 listed** — if a notebook surfaces during the plan-stage audit that's not in §2-B2's list, defer to a 0.44.0 follow-up rather than expanding scope.

### §6 CHANGELOG entry (skeleton for 0.43.0)

```
## 0.43.0 — YYYY-MM-DD

Shim retirement + canonical promotion. The svar/, lp/lp_*.py, and
inference/legacy/ paths flagged in 0.42.0 are deleted. panel_svar and
identify_maxshare are promoted to var/identify/. LP signatures are
harmonized to canonical kwargs; all callers updated.

### Added
- puremacro.var.identify.panel.mean_group_svar + PanelSVARResult.
- puremacro.var.identify.maxshare.identify_maxshare + MaxShareResult
  (extends the existing maxshare/news_maxshare low-level helpers).
- puremacro.lp.panel.lp_panel_regime_interaction (promoted from legacy).
- puremacro.lp.state_dep.lp_smooth_transition_irf (promoted).

### Removed (breaking)
- puremacro.svar.* (entire package).
- puremacro.lp.lp_* (8 files).
- puremacro.inference.legacy.* (4 distinct files merged or promoted
  to puremacro.inference.*; 6 byte-identical shims deleted outright).

### Changed
- LP signatures harmonized to canonical kwargs (y/x/n_lags/alpha/
  entity_level/time_level). 8+ callers in tools/ and 9 notebooks
  updated in the same release.
- 9 notebooks under R1_methods/, R2_subnational/, T5_research_lab,
  T_us_national updated to canonical imports. Body-rewrite notebooks
  re-executed; rename-only notebooks preserved as-is.
- teaching/bq_canonical no longer needs its local bq_svar adapter
  (deleted; call sites switched to canonical BQSVARResult).

### Internal
- puremacro.var.identify.* migrated off inference/legacy/* imports.
- tests/test_deprecation_warnings.py + tests/test_shim_shape_preservation.py
  deleted (no shims remain to gate).
```

## Acceptance

- All seven §3 verification gates pass.
- `git grep` for legacy paths returns zero hits outside `docs/` and `CHANGELOG.md`.
- `ARCHITECTURE.md` updated: the "Known consolidation candidates" section is empty (or removed); stability tier table marks `var/identify/panel`, `var/identify/maxshare.identify_maxshare`, `lp.panel.lp_panel_regime_interaction`, `lp.state_dep.lp_smooth_transition_irf` as Stable.
- `puremacro/__init__.py::__version__` and `pyproject.toml::version` both read `0.43.0`.
- 0.43.0 CHANGELOG entry committed.
- For each body-rewrite notebook listed in §2-B2 (and resolved at plan time): a follow-up commit shows pre-execution → post-execution diff.
