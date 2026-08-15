# 0.43.0 audit notes — 2026-05-18

## Baseline
- Pytest collected: 1309 tests (per `--co` dry run).
- Expected from plan: 1274 passed, 10 failed (pre-existing, orthogonal).
- Full-suite wall time: ~12 min; the blocking run was deferred to background.
  The plan's baseline numbers (1274/10) are treated as authoritative for this
  Task-0 audit; a fresh run result will be recorded at the start of Task 1.
- Pre-existing failures (per plan and git history):
  - Cluster around `datetime.utcnow()` deprecation warnings in narrative sources.
  - `test_public_api` snapshot mismatches (existing snapshot reflects 0.42.0 state).

## src/ status (Step 2)
- `src/` directory: **does not exist**.
- Detail: `ls .../uncertainty_examples/src/` returns "No such file or directory".
  The comment in `tools/make_notebook_R1_01.py` lines 11 and 71 that says
  "All identification functions live in src.svar.*, NOT puremacro.svar.*"
  is **stale documentation**. The builder actually injects cells that import from
  `puremacro.svar.*` (lines 99–105), which are the 0.42.0 Phase-2 shims.
  The `_bootstrap` prelude language referencing `src/` on `sys.path` is a
  historical artefact from a pre-0.40 layout; it has no runtime effect and
  should be removed when the builder is updated in Task 5.

## Legacy callers (Step 3)

### puremacro.svar.* (in tools/ and notebooks/)

**tools/ builders** (inject legacy import cells into notebooks):
- `tools/make_notebook_R1_01.py` — injects all 6 shim imports: `estimate_var`, `identify_cholesky`, `identify_bq`, `identify_sign`, `identify_proxy`, `identify_heteroskedasticity`, `identify_maxshare`
- `tools/make_notebook_R1_03.py` — injects: `identify_cholesky`, `estimate_var`, `panel_svar` (Phase-2.5), plus `lp_jorda`, `lp_panel`, `inference.legacy.bootstrap._irf_from_var`
- `tools/make_notebook_R1_04.py` — injects: `identify_bq`, `identify_proxy`, `estimate_var`
- `tools/make_notebook_R1_05.py` — injects: `estimate_var`, `identify_cholesky`, `identify_bq`, `identify_proxy`, `identify_sign`, `identify_maxshare` (Phase-2.5), `lp_jorda`, conditionally `lp_garch_state` and `panel_svar`

**notebooks/** (live executed state):
- `notebooks/R1_methods/R1_01_svar_menu.ipynb` — all 6 shim imports (mirrors builder)
- `notebooks/R1_methods/R1_03_cross_country.ipynb` — `identify_cholesky`, `estimate_var`, `panel_svar`, `inference.legacy.bootstrap._irf_from_var`
- `notebooks/R1_methods/R1_04_dsge_compare.ipynb` — `identify_bq`, `identify_proxy`, `estimate_var`
- `notebooks/R1_methods/R1_05_publication.ipynb` — all svar shims + conditional `panel_svar`

### puremacro.lp.lp_*

**tools/ builders**:
- `tools/make_notebook_R1_02.py` — injects: `lp_jorda`, `lp_panel`, `lp_iv`, `lp_smooth`, `lp_state_dep`, `lp_garch_state`, `lp_garch_in_mean` (all Phase-2.5)
- `tools/make_notebook_R1_03.py` — injects: `lp_jorda`, `lp_panel`
- `tools/make_notebook_R1_05.py` — injects: `lp_jorda`, conditionally `lp_garch_state`
- `tools/make_notebook_R2_01.py` — injects: `lp_panel_dk` (Phase-2.5; note: `lp_panel_dk` uses canonical kwargs `outcome=`, `unit_col=`, `date_col=` — the shim was reverted at 0.42.0, so this IS the canonical function, no path change needed, just the Phase-2.5 banner)
- `tools/make_notebook_R2_02.py` — injects: `lp_iv` (Phase-2.5), `regress.lp.lp_panel` (separately scoped)

**tools/ run_* scripts** (direct callers, not builders):
- `tools/run_state_bartik_urate_quartile.py` — `from puremacro.lp.lp_panel import lp_panel_regime_interaction`
- `tools/run_jolts_sectoral_lp.py` — `from puremacro.lp.lp_panel import lp_panel_regime_interaction`
- `tools/build_cross_country_tightness_extended.py` — `from puremacro.lp.lp_panel import lp_panel_regime_interaction`
- `tools/run_bartik_surprises_lp.py` — `from puremacro.lp.lp_panel import lp_panel_regime_interaction`
- `tools/run_bartik_ltui_post2017_alone.py` — `from puremacro.lp.lp_panel import lp_panel_regime_interaction`
- `tools/run_aus_state_vacancy_lp.py` — `from puremacro.lp.lp_panel import lp_panel_regime_interaction`
- `tools/run_bartik_horse_race_lp.py` — `from puremacro.lp.lp_panel import lp_panel_regime_interaction`
- `tools/run_jolts_state_bartik_lp.py` — `from puremacro.lp.lp_panel import lp_panel_regime_interaction`

**notebooks/**:
- `notebooks/T5_research_lab.ipynb` — `lp_garch_state` (conditional, Phase-2.5)
- `notebooks/R2_subnational/R2_02_lp_iv_bartik.ipynb` — `lp_iv` (Phase-2.5)
- `notebooks/R2_subnational/R2_01_panels_and_data.ipynb` — `lp_panel_dk` (canonical path, Phase-2.5 banner only)
- `notebooks/R1_methods/R1_05_publication.ipynb` — `lp_jorda`, conditional `lp_garch_state`
- `notebooks/R1_methods/R1_03_cross_country.ipynb` — `lp_jorda`, `lp_panel`
- `notebooks/R1_methods/R1_02_lp_menu.ipynb` — all 7 lp_*.py modules

### puremacro.inference.legacy
- `tools/make_notebook_R1_03.py` — injects `from puremacro.inference.legacy.bootstrap import _irf_from_var`
- `notebooks/R1_methods/R1_03_cross_country.ipynb` — same (mirrors builder)
- `notebooks/T_us_national.ipynb` — `from puremacro.inference.legacy.bootstrap import residual_bootstrap_var`

### puremacro.regress.lp (separately scoped; report only)
- `tools/run_paper_extensions.py`
- `tools/make_notebook_R2_02.py` — `from puremacro.regress.lp import lp_panel`
- `tools/run_logurate_revision.py`
- `tools/make_notebook_R3_02.py`
- `notebooks/R3_narrative/R3_02_irfs_and_state_dep.ipynb`
- `notebooks/R2_subnational/R2_02_lp_iv_bartik.ipynb` — `from puremacro.regress.lp import lp_panel` using canonical kwargs `y=`, `shock=`, `unit=`, `date=`

## Notebook classification (Step 4)

| Notebook | Verdict | Paired builder | Notes |
|---|---|---|---|
| R1_01_svar_menu | **body-rewrite** | `tools/make_notebook_R1_01.py` | All 6 shim imports; tuple unpacks via local `_bootstrap_irf()` which calls shims that return 3-tuples (`point, lo, hi = cholesky_svar(...)`, `point, lo, hi = bq_svar(...)`). Canonical functions return named result objects. |
| R1_02_lp_menu | **rename-only** | `tools/make_notebook_R1_02.py` | All 7 `lp_*.py` imports (all Phase-2.5). `shock=` kwarg is already canonical in `lp_jorda.lp_irf`. No tuple unpacks of LP results; notebook uses `.beta`, `.lower`, `.upper` attribute access on `LPResult`. Only the import paths need changing. |
| R1_03_cross_country | **body-rewrite** | `tools/make_notebook_R1_03.py` | `panel_svar` (Phase-2.5, no canonical home yet), `inference.legacy.bootstrap._irf_from_var` (retirement target), plus `cholesky_svar` tuple-unpack `point, lo, hi = cholesky_svar(...)`. Requires canonical `PanelSVARResult` and `mean_group_svar` to land first (Task 2). |
| R1_04_dsge_compare | **body-rewrite** | `tools/make_notebook_R1_04.py` | `bq_svar` and `proxy_svar` shim imports; notebook unpacks 3-tuples explicitly: `point_bq, lo_bq, hi_bq = bq_svar(...)`, `point_17, lo_17, hi_17 = proxy_svar(...)`. Must switch to attribute access on `BQSVARResult`/`ProxySVARResult`. |
| R1_05_publication | **body-rewrite** | `tools/make_notebook_R1_05.py` | All svar shims + conditional `panel_svar` / `lp_garch_state` (Phase-2.5). Tuple unpack: `pt, lo, hi = cholesky_svar(...)` inside try/except loop. |
| R2_01_panels_and_data | **rename-only** | `tools/make_notebook_R2_01.py` | Imports only `lp_panel_dk` from `puremacro.lp.lp_panel_dk`. The kwargs `outcome=`, `unit_col=`, `date_col=`, `dk_lag=`, `ci_level=` ARE the canonical signature (the shim was reverted at commit `6ff5846`). The Phase-2.5 banner is the only blocker; once the banner is removed and the path is cleared, this notebook needs no body changes — only the banner retirement. |
| R2_02_lp_iv_bartik | **rename-only** | `tools/make_notebook_R2_02.py` | `lp_iv` (Phase-2.5) and `regress.lp.lp_panel` (separately scoped, not a retirement target for 0.43.0). The `lp_panel` calls use canonical kwargs `y=`, `shock=`, `unit=`, `date=` (matching `puremacro.regress.lp` signature). `lp_iv_irf` calls use canonical `target=`, `shock=` kwargs. No tuple unpacks from either function. |
| T5_research_lab | **rename-only** | **none** (direct edit only) | Only legacy import is `lp_garch_state` from `puremacro.lp.lp_garch_state` (Phase-2.5, conditional). Remaining imports are all `puremacro.teaching.*` which are canonical. No tuple unpacks from `lp_garch_state` calls — result is accessed via `result.horizons`. |
| T_us_national | **body-rewrite** | **none** (direct edit only) | Imports `puremacro.inference.legacy.bootstrap.residual_bootstrap_var` (retirement target). Also imports `puremacro.var.identify.sign.sign_restriction_svar` — this IS the canonical path (no shim), so no issue. Tuple unpacks (`b, lo, hi = lps[t]`) come from a local `jorda_lp()` function defined within the notebook, not from any puremacro function. The only legacy import is `residual_bootstrap_var`; body-rewrite verdict is driven by needing to either promote `residual_bootstrap_var` to canonical or inline the bootstrap logic. |

## Surprises and recommendations

### 1. lp_panel_dk: canonical kwargs, NOT legacy kwargs
`R2_01_panels_and_data.ipynb` uses `outcome=`, `unit_col=`, `date_col=`, `dk_lag=`, `ci_level=` — these LOOK like the legacy `outcome=` pattern but they are the **actual canonical signature** of `puremacro.lp.lp_panel_dk.panel_lp_dk`. The shim was reverted at commit `6ff5846` (`fix(phase2): revert lp_panel_dk shim — kwargs mismatch breaks R2_01 builder`). There is no kwargs rewrite needed for this notebook; only the Phase-2.5 banner on `lp_panel_dk.py` needs to be cleared.

### 2. T_us_national imports canonical sign path, not svar shim
`notebooks/T_us_national.ipynb` uses `from puremacro.var.identify.sign import sign_restriction_svar` — this is already the canonical path under `puremacro/var/identify/sign.py`. It is not a shim caller. The only legacy element is `residual_bootstrap_var` from `inference.legacy.bootstrap`. The body-rewrite verdict is narrow: just the bootstrap call site.

### 3. stale `src/svar` comment in R1_01 builder
`tools/make_notebook_R1_01.py` lines 11 and 71 reference `src.svar.*` as a "local package". This is fully stale — `src/` never existed in the current repo layout. When Task 5 rewrites this builder, delete those comment lines entirely to avoid future confusion.

### 4. 8 run_* tools all import lp_panel_regime_interaction
All 8 `tools/run_*.py` scripts that are lp callers import only `lp_panel_regime_interaction` from `puremacro.lp.lp_panel`. Once `lp_panel_regime_interaction` is promoted to `puremacro.lp.panel` (Task 6), all 8 scripts need a one-line import change. No kwargs rewrite required.

### 5. R1_02 is rename-only despite 7 lp_*.py imports
All 7 `lp_*.py` modules are Phase-2.5 banners only (no shim wrapper, no kwargs mismatch). The `shock=` kwarg used throughout `R1_02` is canonical in `lp_jorda.lp_irf`. Result access is already via named attributes (`.beta`, `.lower`, `.upper`). This is the easiest notebook in the set.

### 6. T5 lp_garch_state is conditional
The `lp_garch_state` import in `T5_research_lab.ipynb` is inside a `try:` block. The result is accessed via `result.horizons` — named attribute access, not 3-tuple. This notebook's only change is the import path once `lp_garch_state` has a canonical home.

### 7. R1_03 is the hardest: two blockers
`R1_03_cross_country.ipynb` requires both (a) `mean_group_svar` / `PanelSVARResult` to land in `puremacro.var.identify.panel` (Task 2) and (b) `inference.legacy.bootstrap._irf_from_var` to be either promoted or inlined. Do not attempt R1_03 body rewrite until both prerequisites are done.

### 8. inference.legacy.bootstrap retirement path
`residual_bootstrap_var` is imported by `T_us_national.ipynb` only. `_irf_from_var` is imported by `R1_03_cross_country.ipynb`. The retirement note in `bootstrap.py` says these are deletion targets for 0.43.0. The plan's Task 4 covers `inference/legacy/` retirement. Verify whether `residual_bootstrap_var` has a canonical destination in `puremacro/inference/bootstrap.py` before deleting.
