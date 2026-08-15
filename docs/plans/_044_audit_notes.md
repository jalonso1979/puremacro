# 0.44.0 audit notes — 2026-05-18

## Baseline
- pytest (targeted: tests/test_var/ tests/test_lp/ test_cholesky_shocks.py test_robustness.py test_pyodide_compat.py): 120 passed, 0 failed.
- NOTE: Expected 122 per plan; actual is 120. Two tests may have been removed or merged since the plan was drafted.

## R1_02 legacy LP call sites (tools/make_notebook_R1_02.py)

### Import block (lines 108–117)
- L108: `from puremacro.lp.lp_jorda import lp_irf, LPResult` — §1
- L109: `from puremacro.lp.lp_panel import lp_panel_irf, PanelLPResult` — §2
- L110: `from puremacro.lp.lp_iv import lp_iv_irf, LPIVResult, WeakInstrumentWarning` — §3
- L111: `from puremacro.lp.lp_smooth import lp_smooth_irf, LPSmoothResult` — §4
- L112: `from puremacro.lp.lp_state_dep import lp_state_dep_irf` — §5
- L115: `from puremacro.lp.lp_garch_state import LPGARCHStateResult` — §6 result type import only
- L117: `from puremacro.lp.lp_garch_in_mean import LPGIMResult` — §7 result type import only

### Call sites
| Line | Function | Section | Kwargs used |
|------|----------|---------|-------------|
| 187  | `lp_irf(df, target=, shock=, horizon=, lags=, ci=, B=)` | §1 helper `_run_lp_hac` | target, shock, horizon, lags, ci, B |
| 235  | `lp_irf(df, target=, shock=, horizon=, lags=, ci=, B=)` | §4/§5 helper `_bootstrap_block` | target, shock, horizon, lags, ci, B |
| 411  | `lp_panel_irf(avail, target=, shock=, horizon=, lags=, entity_col=, time_col=, ci=)` | §2.1 | target, shock, horizon, lags, entity_col, time_col, ci |
| 510  | `lp_iv_irf(df_iv, target=, shock_variable=, instrument=, horizon=, lags=, ci=)` | §3.1 | target, shock_variable, instrument, horizon, lags, ci |
| 562  | `lp_iv_irf(df_rob_aligned, target=, shock_variable=, instrument=, horizon=, lags=, ci=)` | §3.3 robustness | target, shock_variable, instrument, horizon, lags, ci |
| 625  | `lp_irf(df_us_clean, target=, shock=, horizon=, lags=, ci=, B=)` | §4.1 plain baseline | target, shock, horizon, lags, ci, B |
| 628  | `lp_smooth_irf(df_us_clean, target=, shock=, horizon=, lags=, n_knots=, lambda_cv=True, ci=, B=)` | §4.1 | target, shock, horizon, lags, n_knots, lambda_cv, ci, B |
| 673  | `lp_smooth_irf(dfc, target=, shock=, horizon=, lags=, n_knots=, lambda_cv=True, ci=, B=)` | §4.2 cross-country loop | target, shock, horizon, lags, n_knots, lambda_cv, ci, B |
| 693  | `lp_smooth_irf(dfc, target=, shock=, horizon=, lags=, n_knots=, lambda_cv=True, ci=, B=)` | §4.2 WUI | same |
| 723  | `lp_irf(dfc, target=, shock=, horizon=, lags=, ci=, B=)` | §4.3 plain baseline | target, shock, horizon, lags, ci, B |
| 835  | `lp_state_dep_irf(df, target=, shock=, horizon=, lags=, state_col=, ci=)` | §5.2 state-dep loop | target, shock, horizon, lags, state_col, ci |

### Result attribute access patterns
- `lp_irf` result: `.beta[h]`, `.se[h]`, `.lower[h]`, `.upper[h]`, `.horizons`, `.ci_lo[h]`, `.ci_hi[h]`
  (MIXED: some call sites use `.lower`/`.upper`, others use `.ci_lo`/`.ci_hi` — see Surprises)
- `lp_panel_irf` result: `.beta[h]`, `.se[h]`, `.lower[h]`, `.upper[h]`, `.bandwidth`
- `lp_iv_irf` result: `.beta[h]`, `.se_wald[h]`, `.lower_ar[h]`, `.upper_ar[h]`, `.lower_wald[h]`, `.upper_wald[h]`, `.first_stage_f`, `.is_weak`
- `lp_smooth_irf` result: `.beta`, `.ci_lo`, `.ci_hi`, `.horizons`, `.lambda_used`, `.lambda_grid_mse`
- `lp_state_dep_irf` result: `.beta_H`, `.beta_L`, `.se_H`, `.se_L` (via `r`), `.ci_lo_H`, `.ci_hi_H`, `.ci_lo_L`, `.ci_hi_L`, `.horizons`
  (legacy `_run_lp_hac` results also accessed as `.beta_0`, `.beta_1`, `.ci_lo_0`, `.ci_hi_0`, `.ci_lo_1`, `.ci_hi_1` for hard-split results from `lp_state_dep_irf`)
- `LPGARCHStateResult` (§6, from `lp_garch_state` canonical): `.beta_H`, `.beta_L`, `.lower_H`, `.upper_H`, `.lower_L`, `.upper_L`, `.horizons`
- `LPGIMResult` (§7, from legacy `lp_garch_in_mean`): `.beta`, `.delta`, `.lower_beta`, `.upper_beta`, `.lower_delta`, `.upper_delta`, `.horizons`

## R1_03 legacy LP call sites (tools/make_notebook_R1_03.py)

| Line | Function | Section | Kwargs used |
|------|----------|---------|-------------|
| 414  | `lp_irf(df_lp, target=, shock=, horizon=, lags=, ci=)` | §5 proxy comparison loop | target, shock, horizon, lags, ci (no B) |

Import (L111): `from puremacro.lp.lp_panel import RegimeInteractionLPResult` — legacy result type referenced in import; however the actual call at L664 uses canonical `lp_panel_regime_interaction` with a `_RIAccessor` shim wrapping the DataFrame. The `RegimeInteractionLPResult` import is stale/unused in the execution path.

Result attribute access for `lp_irf` result:
- `.beta[h]`, `.lower[h]`, `.upper[h]` (sliced with `[:LP_HORIZON+1]`)

## R1_05 legacy LP call sites (tools/make_notebook_R1_05.py)

No legacy LP call sites found. `grep` returned empty output.

## WeakInstrumentWarning
Lives in **legacy module** `puremacro/puremacro/lp/lp_iv.py` (L54: `class WeakInstrumentWarning(UserWarning)`).
There is **no canonical path** — it was not ported to the canonical `lp/iv.py`.
Action: drop the import of `WeakInstrumentWarning` from `lp_iv` and replace the downstream `try/except`-style `warnings.catch_warnings` block in §3.1 with a plain `warnings.catch_warnings` that catches `UserWarning`, or drop the warning filter entirely (since `lp_iv` returns a plain DataFrame with `first_stage_f` column — no warning issued).

## Legacy → canonical mapping (verified 2026-05-18)

| Legacy fn | Canonical fn | Kwarg renames | Return type change |
|---|---|---|---|
| `lp_irf` | `lp_hac` | `target`→`y`, `shock`→`x`, `horizon`→`horizons` (int → `range(0,H+1)`), `lags`→`n_lags`, `ci`→`alpha` (flip: `alpha=1-ci`), `B` (bootstrap reps) → dropped (no bootstrap in canonical) | `LPResult` → `pd.DataFrame` |
| `lp_iv_irf` | `lp_iv` | `target`→`y`, `shock_variable`→`x`, `instrument`→`z`, `horizon`→`horizons`, `lags`→`n_lags`, `ci`→`alpha` | `LPIVResult` → `pd.DataFrame` |
| `lp_panel_irf` | `panel_lp` | `target`→`y`, `shock`→`x`, `horizon`→`horizons`, `lags`→`n_lags`, `entity_col`→`entity_level`, `time_col`→`time_level`, `ci`→`alpha` | `PanelLPResult` → `pd.DataFrame` |
| `lp_smooth_irf` | `lp_smooth` | `target`→`y`, `shock`→`x`, `horizon`→`horizons`, `lags`→`n_lags`, `ci`→`alpha`, `lambda_cv=True`→`lambda_=None` (GCV is default), `B` → dropped | `LPSmoothResult` → `pd.DataFrame` |
| `lp_state_dep_irf` | `lp_state_dep` | `target`→`y`, `shock`→`x`, `horizon`→`horizons`, `lags`→`n_lags`, `state_col`→`state`, `ci`→`alpha` | `LPStateDepResult` → `pd.DataFrame` |

## Canonical DataFrame column names (verified from source)

| Canonical fn | Columns |
|---|---|
| `lp_hac` | `h, beta, se, t, lo, hi` |
| `lp_iv` | `h, beta, se, t, lo, hi, first_stage_f` |
| `panel_lp` | `h, beta, se, t, lo, hi` (no `bandwidth` column — it's internal) |
| `lp_panel_regime_interaction` | `h, regime, beta, se, lo, hi, wald_pval, n_obs, n_entities` |
| `lp_smooth` | `h, beta, se, lo, hi, lambda` |
| `lp_state_dep` | `h, beta_H, se_H, lo_H, hi_H, beta_L, se_L, lo_L, hi_L` |
| `lp_smooth_transition_irf` (canonical alias in state_dep.py) | `h, beta_high, se_high, lo_high, hi_high, beta_low, se_low, lo_low, hi_low` |
| `lp_garch_state` | delegates to `lp_state_dep` → same columns as `lp_state_dep` above |
| `lp_garch_in_mean` | delegates to `lp_hac` → `h, beta, se, t, lo, hi` |

## Surprises

1. **MIXED `.lower`/`.upper` vs `.ci_lo`/`.ci_hi` on `lp_irf` results**: The `_plot_irf_grid` helper (L206-207) duck-types `getattr(res, 'lower', getattr(res, 'ci_lo', None))`. After migration to canonical, all results will be DataFrames — `.lower`/`.ci_lo` attribute access breaks entirely. Every plot helper must switch to `df["lo"].values` / `df["hi"].values`.

2. **`.bandwidth` attribute on `PanelLPResult` (L416)**: `res.bandwidth` is accessed after `lp_panel_irf`. Canonical `panel_lp` returns a plain DataFrame with NO `bandwidth` column. Migration must either drop this print or compute bandwidth separately from `floor(0.75 * T**(1/3))`.

3. **`lp_iv_irf` returns Wald + AR bands separately** (`.lower_wald`, `.upper_wald`, `.lower_ar`, `.upper_ar`, `.se_wald`, `.is_weak`): Canonical `lp_iv` only has `lo`/`hi` (Wald-style). There is NO AR-band column. The AR confidence band logic (Anderson-Rubin) must either be dropped, approximated, or the canonical `lp_iv` extended. This is a **feature gap**, not just a kwarg rename.

4. **`lp_smooth_irf` `.lambda_used` and `.lambda_grid_mse` attributes** (L631, L649-655): Canonical `lp_smooth` encodes `lambda` as a column (same value repeated for every row) and does NOT expose `lambda_grid_mse`. After migration, `lambda_used = df["lambda"].iloc[0]`; the CV curve plot (L649-655) must be dropped or rewritten.

5. **`LPGARCHStateResult` is still a legacy dataclass** (imported from `lp_garch_state.py`): The canonical `lp_garch_state` (from `garch_state.py`) now delegates to `lp_state_dep` and returns a DataFrame. The §6 result-type import is stale — `lp_garch_state` no longer returns `LPGARCHStateResult`.

6. **`LPGIMResult` is still a legacy dataclass** (imported from `lp_garch_in_mean.py`): The canonical `lp_garch_in_mean` (from `garch_in_mean.py`) delegates to `lp_hac` and returns a plain DataFrame with `.beta`/`.se`/`.lo`/`.hi`. Legacy `.delta`, `.lower_delta`, `.upper_delta`, `.delta_wald_stat`, `.delta_wald_p` attributes are gone in canonical. The §7 sigma-channel plot (L1145-L1152) accesses `.lower_beta`, `.upper_beta`, `.lower_delta`, `.upper_delta` — all need rewriting.

7. **`lp_state_dep_irf` hard-split vs smooth-transition**: The call site (L835) uses `state_col='regime_binary'` which is a binary 0/1 column — effectively `transition="threshold"`. Result attributes accessed are `.beta_0`/`.beta_1`/`.ci_lo_0`/`.ci_hi_0`/`.ci_lo_1`/`.ci_hi_1` (L840, L872-877). Canonical `lp_state_dep` uses `beta_H`/`beta_L`/`lo_H`/`hi_H`/`lo_L`/`hi_L`. Hard-split attribute names are legacy-specific.

8. **`RegimeInteractionLPResult` in R1_03 is a dead import**: At L111 it is imported but the actual interaction call (L664) uses canonical `lp_panel_regime_interaction` directly + a `_RIAccessor` shim. The import can simply be deleted.

9. **R1_05 has zero legacy LP call sites** — no work required for that notebook in Tasks 1-3.

10. **`horizon` kwarg takes a single int in legacy** (e.g. `horizon=HORIZON` where `HORIZON=12`): canonical `horizons` takes an iterable. Migration must convert `horizon=N` → `horizons=range(0, N+1)`.
