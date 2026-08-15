# Phase-2 audit notes — 2026-05-17

Baseline pytest count: 1259 passed, 11 failed, 26 skipped.

Pre-existing failures (orthogonal to Phase 2):
- tests/test_gar/test_qar_skewt_fci.py::test_fit_skewt_recovers_symmetric_normal
- tests/test_gar/test_qar_skewt_fci.py::test_fit_skewt_captures_left_skew
- tests/test_gar/test_qar_skewt_fci.py::test_skewt_fit_downside_quantile_consistent
- tests/test_gar/test_qar_skewt_fci.py::test_skewt_expected_shortfall_below_5pct
- tests/test_gar/test_qar_skewt_fci.py::test_skewt_fit_is_frozen
- tests/test_narrative_body_extractor_coverage.py::test_extract_body_bcb_from_pdf_fixture
- tests/test_narrative_body_extractor_coverage.py::test_extract_body_banxico_from_pdf_fixture
- tests/test_narrative_indices.py::test_lexicons_top_level_keys
- tests/test_public_api.py::test_public_api_matches_snapshot

Filterwarnings status: clean (no filter config in pyproject.toml or tests/conftest.py).

## Parity audit results (lp/lp_*.py vs lp/*.py)

| Legacy | Canonical | Top-level defs/classes match? | Verdict |
|---|---|---|---|
| lp_jorda    | jorda    | No — legacy exposes `lp_irf` (returns `LPResult` dataclass); canonical exposes `lp_hac` (returns `pd.DataFrame`). Function name differs, parameter names differ (`target/shock/horizon/lags/B` vs `y/x/horizons/n_lags/alpha`), and return type differs (dataclass vs DataFrame). `lp_irf` is imported in test files and multiple notebook builders. | DEFER-2.5 |
| lp_iv       | iv       | No — legacy exposes `lp_iv_irf` (returns `LPIVResult` dataclass) + `WeakInstrumentWarning`; canonical exposes `lp_iv` (returns `pd.DataFrame`). Function name differs, return type differs. `lp_iv_irf`, `LPIVResult`, and `WeakInstrumentWarning` are all imported in `tests/test_lp_iv.py` and notebook builders. | DEFER-2.5 |
| lp_panel    | panel    | No — legacy exposes `lp_panel_irf` (returns `PanelLPResult` dataclass) + `lp_panel_regime_interaction` (returns `RegimeInteractionLPResult`); canonical exposes only `panel_lp`. `lp_panel_regime_interaction` is actively used in 8+ `tools/run_*.py` scripts with direct imports. No canonical equivalent exists. | DEFER-2.5 |
| lp_panel_dk | panel_dk | Yes — both expose exactly `panel_lp_dk`. Same function name; canonical adds `__all__`. | PARITY-OK |
| lp_state_dep | state_dep | No — legacy exposes `lp_state_dep_irf` (returns `LPStateDepResult` dataclass) + `lp_smooth_transition_irf` (returns `LPSTResult`); canonical exposes `lp_state_dep` (returns `pd.DataFrame`). Name differs, return type differs. `lp_smooth_transition_irf` is imported in `tools/make_notebook_R1_02.py` and its notebook. `LPSTResult` appears in `public_api_snapshot.json`. | DEFER-2.5 |
| lp_smooth    | smooth    | No — legacy exposes `lp_smooth_irf` (returns `LPSmoothResult` dataclass); canonical exposes `lp_smooth` (returns `pd.DataFrame`). Name differs, return type differs. | DEFER-2.5 |
| lp_garch_state    | garch_state    | Partial — both expose `lp_garch_state`. Function name matches. Legacy adds `LPGARCHStateResult` dataclass (return type); canonical returns what `lp_state_dep` returns (pd.DataFrame via delegation). Class name can be aliased in a shim; function name matches. | PARITY-OK |
| lp_garch_in_mean  | garch_in_mean  | Partial — both expose `lp_garch_in_mean`. Function name matches. Legacy adds `LPGIMResult` dataclass (return type); canonical returns what `lp_hac` returns (pd.DataFrame via delegation). Class name can be aliased in a shim; function name matches. | PARITY-OK |

## Notes / observations

- **lp_panel is the most critical DEFER-2.5**: `lp_panel_regime_interaction` has no canonical equivalent and is called in at least 8 active `tools/` scripts (`run_state_bartik_urate_quartile.py`, `run_jolts_sectoral_lp.py`, `build_cross_country_tightness_extended.py`, `run_aus_state_vacancy_lp.py`, `run_bartik_surprises_lp.py`, `run_bartik_ltui_post2017_alone.py`, `run_bartik_horse_race_lp.py`, `run_jolts_state_bartik_lp.py`). A banner-only treatment is appropriate; cannot shim without a canonical target.

- **Return-type mismatch pattern**: The primary blocker for DEFER-2.5 files is that the legacy functions return named dataclasses (`LPResult`, `LPIVResult`, etc.) while the canonical functions return `pd.DataFrame`. Callers access `result.beta`, `result.se`, `result.lower`, `result.upper` on the dataclass objects. A shim wrapper would need to convert the DataFrame back to a dataclass, which is technically possible but fragile.

- **lp_garch_state and lp_garch_in_mean are PARITY-OK** because both pairs share the same function name. The legacy result classes (`LPGARCHStateResult`, `LPGIMResult`) are not referenced in any external files (tests or tools). A shim can alias these as empty stubs or re-export the function name directly.

- **lp_panel_dk is the cleanest PARITY-OK**: identical function name (`panel_lp_dk`), no result class difference, no external usage of a divergent API.

- **Pre-existing test_public_api failure**: The `test_public_api_matches_snapshot` failure pre-exists Phase 2 and must be kept in mind when regenerating the snapshot in later tasks — do not accidentally fix it in a way that masks new snapshot entries.

- **Pytest runtime**: ~12 minutes for the full suite. Plan accordingly for tasks that re-run tests.
