# puremacro examples gallery

Generated: 2026-08-15T20:39:45Z

| Status | Count |
|---|---|
| PASS | 61 |
| SKIP | 4 |
| FAIL | 8 |

## FAIL (8)

### asset_composition_dynamics

- **Status:** FAIL
- **Reason:** Augmented panel not found at /Users/jalonso/Documents/data_fetch/output/investment_by_asset_augmented.parquet. Run `python -m data_fetch.run_fill_gaps` first.
- **Runtime:** 0.468 s
- **Last run:** 2026-08-15T20:32:58Z
- **Figures:** none

### global_fiscal_stance

- **Status:** FAIL
- **Reason:** ValueError: Invalid frequency: A. Failed to parse with error message: ValueError("Invalid frequency: A. Failed to parse with error message: KeyError('A'). Did you mean Y?") Did you mean Y?
- **Runtime:** 0.517 s
- **Last run:** 2026-08-15T20:34:30Z
- **Figures:** none

### govt_vs_private_investment

- **Status:** FAIL
- **Reason:** Required data files missing. Run `python -m data_fetch.run_fill_gaps` first.
- **Runtime:** 0.478 s
- **Last run:** 2026-08-15T20:34:32Z
- **Figures:** none

### la_lp_pmw_demo

- **Status:** FAIL
- **Reason:** ValueError: Invalid frequency: Q. Failed to parse with error message: ValueError("'Q' is no longer supported for offsets. Please use 'QE' instead.")
- **Runtime:** 1.113 s
- **Last run:** 2026-08-15T20:38:08Z
- **Figures:** none

### narrative_panel_lp

- **Status:** FAIL
- **Reason:** Required input panels are missing.
- **Runtime:** 0.475 s
- **Last run:** 2026-08-15T20:39:11Z
- **Figures:** none

### regime_workflow

- **Status:** FAIL
- **Reason:** ValueError: Invalid frequency: Q. Failed to parse with error message: ValueError("'Q' is no longer supported for offsets. Please use 'QE' instead.")
- **Runtime:** 1.135 s
- **Last run:** 2026-08-15T20:39:23Z
- **Figures:** none

### report_workflow

- **Status:** FAIL
- **Reason:** ValueError: Invalid frequency: Q. Failed to parse with error message: ValueError("'Q' is no longer supported for offsets. Please use 'QE' instead.")
- **Runtime:** 1.127 s
- **Last run:** 2026-08-15T20:39:25Z
- **Figures:** none

### vulnerable_growth

- **Status:** FAIL
- **Reason:** ValueError: Invalid frequency: Q. Failed to parse with error message: ValueError("'Q' is no longer supported for offsets. Please use 'QE' instead.")
- **Runtime:** 1.162 s
- **Last run:** 2026-08-15T20:39:45Z
- **Figures:** none

## PASS (61)

### anderson_rubin_weak_iv_demo

- **Status:** PASS
- **Runtime:** 1.804 s
- **Last run:** 2026-08-15T20:32:58Z
- **Figures:**
  - `../puremacro/examples/output/anderson_rubin_weak_iv_demo.png`

### bandpass_cycles_comparison

- **Status:** PASS
- **Runtime:** 1.021 s
- **Last run:** 2026-08-15T20:32:59Z
- **Figures:**
  - `../puremacro/examples/output/bandpass_cycles_comparison.png`

### berkowitz_demo

- **Status:** PASS
- **Runtime:** 0.592 s
- **Last run:** 2026-08-15T20:33:00Z
- **Figures:** none

### bvar_fan_chart

- **Status:** PASS
- **Runtime:** 0.97 s
- **Last run:** 2026-08-15T20:33:02Z
- **Figures:**
  - `../puremacro/examples/output/bvar_fan_chart.png`

### cointegration_johansen

- **Status:** PASS
- **Runtime:** 1.161 s
- **Last run:** 2026-08-15T20:33:04Z
- **Figures:** none

### dcc_volatility

- **Status:** PASS
- **Runtime:** 2.259 s
- **Last run:** 2026-08-15T20:33:07Z
- **Figures:**
  - `../puremacro/examples/output/dcc_volatility.png`

### dfm_bai_ng

- **Status:** PASS
- **Runtime:** 0.939 s
- **Last run:** 2026-08-15T20:33:08Z
- **Figures:**
  - `../puremacro/examples/output/dfm_bai_ng.png`

### dfm_nowcast_kalman

- **Status:** PASS
- **Runtime:** 0.932 s
- **Last run:** 2026-08-15T20:33:09Z
- **Figures:**
  - `../puremacro/examples/output/dfm_nowcast_kalman.png`

### did_callaway_santanna_demo

- **Status:** PASS
- **Runtime:** 0.962 s
- **Last run:** 2026-08-15T20:33:10Z
- **Figures:** none

### diebold_yilmaz_spillover

- **Status:** PASS
- **Runtime:** 1.243 s
- **Last run:** 2026-08-15T20:33:11Z
- **Figures:** none

### dsge_ar1_demo

- **Status:** PASS
- **Runtime:** 55.842 s
- **Last run:** 2026-08-15T20:34:07Z
- **Figures:** none

### dsge_fertility_demo

- **Status:** PASS
- **Runtime:** 1.625 s
- **Last run:** 2026-08-15T20:34:08Z
- **Figures:** none

### dsge_rbc_klein

- **Status:** PASS
- **Runtime:** 1.645 s
- **Last run:** 2026-08-15T20:34:10Z
- **Figures:**
  - `../puremacro/examples/output/dsge_rbc_klein.png`

### exact_diffuse_kalman

- **Status:** PASS
- **Runtime:** 0.505 s
- **Last run:** 2026-08-15T20:34:10Z
- **Figures:**
  - `../puremacro/examples/output/exact_diffuse_kalman.png`

### fetch_fred_demo

- **Status:** PASS
- **Runtime:** 2.414 s
- **Last run:** 2026-08-15T20:34:13Z
- **Figures:**
  - `../puremacro/examples/output/fetch_fred_demo.png`

### fm_ols_dols_demo

- **Status:** PASS
- **Runtime:** 1.734 s
- **Last run:** 2026-08-15T20:34:15Z
- **Figures:**
  - `../puremacro/examples/output/fm_ols_dols.png`

### forecast_density_eval

- **Status:** PASS
- **Runtime:** 1.651 s
- **Last run:** 2026-08-15T20:34:16Z
- **Figures:**
  - `../puremacro/examples/output/forecast_density_eval.png`

### gali_1999_hours

- **Status:** PASS
- **Runtime:** 2.145 s
- **Last run:** 2026-08-15T20:34:18Z
- **Figures:**
  - `../puremacro/examples/output/gali_1999_hours.png`

### garch_midas_macro_volatility

- **Status:** PASS
- **Runtime:** 1.98 s
- **Last run:** 2026-08-15T20:34:20Z
- **Figures:**
  - `../puremacro/examples/output/garch_midas_macro_volatility.png`

### gfevd_spillover

- **Status:** PASS
- **Runtime:** 1.128 s
- **Last run:** 2026-08-15T20:34:22Z
- **Figures:** none

### gk_robust_from_gibbs

- **Status:** PASS
- **Runtime:** 4.781 s
- **Last run:** 2026-08-15T20:34:26Z
- **Figures:**
  - `../puremacro/examples/output/gk_robust_from_gibbs.png`

### gk_robust_signs

- **Status:** PASS
- **Runtime:** 3.024 s
- **Last run:** 2026-08-15T20:34:29Z
- **Figures:**
  - `../puremacro/examples/output/gk_robust_signs.png`

### glp_lambda_search

- **Status:** PASS
- **Runtime:** 1.202 s
- **Last run:** 2026-08-15T20:34:31Z
- **Figures:**
  - `../puremacro/examples/output/glp_lambda_search.png`

### har_realized_vol

- **Status:** PASS
- **Runtime:** 1.721 s
- **Last run:** 2026-08-15T20:34:33Z
- **Figures:**
  - `../puremacro/examples/output/har_realized_vol.png`

### hfi_gertler_karadi

- **Status:** PASS
- **Runtime:** 210.536 s
- **Last run:** 2026-08-15T20:38:04Z
- **Figures:** none

### kilian_2009_oil

- **Status:** PASS
- **Runtime:** 3.571 s
- **Last run:** 2026-08-15T20:38:07Z
- **Figures:**
  - `../puremacro/examples/output/kilian_2009_oil.png`

### labor_flows_demo

- **Status:** PASS
- **Runtime:** 0.512 s
- **Last run:** 2026-08-15T20:38:09Z
- **Figures:** none

### lp_asymmetric_tenreyro

- **Status:** PASS
- **Runtime:** 1.195 s
- **Last run:** 2026-08-15T20:38:10Z
- **Figures:** none

### lp_smooth_demo

- **Status:** PASS
- **Runtime:** 1.178 s
- **Last run:** 2026-08-15T20:38:13Z
- **Figures:** none

### lwui_wage_demo

- **Status:** PASS
- **Runtime:** 0.513 s
- **Last run:** 2026-08-15T20:38:13Z
- **Figures:** none

### mcmc_diagnostics

- **Status:** PASS
- **Runtime:** 0.97 s
- **Last run:** 2026-08-15T20:38:14Z
- **Figures:**
  - `../puremacro/examples/output/mcmc_diagnostics.png`

### midas_quarterly_monthly

- **Status:** PASS
- **Runtime:** 1.626 s
- **Last run:** 2026-08-15T20:38:16Z
- **Figures:**
  - `../puremacro/examples/output/midas_quarterly_monthly.png`

### mle_sandwich_demo

- **Status:** PASS
- **Runtime:** 1.034 s
- **Last run:** 2026-08-15T20:38:17Z
- **Figures:** none

### ms_var_business_cycle

- **Status:** PASS
- **Runtime:** 1.675 s
- **Last run:** 2026-08-15T20:38:18Z
- **Figures:**
  - `../puremacro/examples/output/ms_var_business_cycle.png`

### narrative_custom_corpus

- **Status:** PASS
- **Runtime:** 0.965 s
- **Last run:** 2026-08-15T20:38:19Z
- **Figures:**
  - `../puremacro/examples/output/narrative_custom_corpus.png`

### narrative_g7_panel

- **Status:** PASS
- **Runtime:** 9.48 s
- **Last run:** 2026-08-15T20:38:29Z
- **Figures:**
  - `../puremacro/examples/output/narrative_g7_panel.png`

### narrative_homogeneous_panel

- **Status:** PASS
- **Runtime:** 3.219 s
- **Last run:** 2026-08-15T20:38:32Z
- **Figures:**
  - `../puremacro/examples/output/homogeneous_panel.png`
  - `../puremacro/examples/output/narrative_iv_panel_consumption.csv`
  - `../puremacro/examples/output/narrative_iv_panel_dedup_audit.csv`
  - `../puremacro/examples/output/narrative_iv_panel_investment.csv`
  - `../puremacro/examples/output/narrative_iv_panel_long.csv`
  - `../puremacro/examples/output/narrative_iv_panel_quarterly.csv`

### narrative_indices_demo

- **Status:** PASS
- **Runtime:** 0.536 s
- **Last run:** 2026-08-15T20:38:33Z
- **Figures:** none

### narrative_llm_pipeline

- **Status:** PASS
- **Runtime:** 1.877 s
- **Last run:** 2026-08-15T20:38:34Z
- **Figures:** none

### narrative_local_llm

- **Status:** PASS
- **Runtime:** 0.53 s
- **Last run:** 2026-08-15T20:38:35Z
- **Figures:** none

### narrative_multilingual_live

- **Status:** PASS
- **Runtime:** 35.126 s
- **Last run:** 2026-08-15T20:39:10Z
- **Figures:**
  - `../puremacro/examples/output/narrative_multilingual_live.png`

### narrative_ramey_2011

- **Status:** PASS
- **Runtime:** 2.101 s
- **Last run:** 2026-08-15T20:39:13Z
- **Figures:**
  - `../puremacro/examples/output/narrative_ramey_2011.png`

### narrative_romer_romer_2010

- **Status:** PASS
- **Runtime:** 1.349 s
- **Last run:** 2026-08-15T20:39:14Z
- **Figures:**
  - `../puremacro/examples/output/narrative_rr_vs_ramey.png`

### narrative_sign_adrr

- **Status:** PASS
- **Runtime:** 1.948 s
- **Last run:** 2026-08-15T20:39:16Z
- **Figures:**
  - `../puremacro/examples/output/narrative_sign_adrr.png`

### narrative_us_canonical_compare

- **Status:** PASS
- **Runtime:** 1.512 s
- **Last run:** 2026-08-15T20:39:17Z
- **Figures:**
  - `../puremacro/examples/output/narrative_us_canonical_compare.png`

### non_gaussian_svar

- **Status:** PASS
- **Runtime:** 1.662 s
- **Last run:** 2026-08-15T20:39:19Z
- **Figures:**
  - `../puremacro/examples/output/non_gaussian_svar.png`

### posterior_predictive_fan

- **Status:** PASS
- **Runtime:** 1.014 s
- **Last run:** 2026-08-15T20:39:20Z
- **Figures:**
  - `../puremacro/examples/output/posterior_predictive_fan.png`

### ramey_zubairy_2018_multipliers

- **Status:** PASS
- **Runtime:** 2.09 s
- **Last run:** 2026-08-15T20:39:22Z
- **Figures:**
  - `../puremacro/examples/output/ramey_zubairy_2018_multipliers.png`

### shock_atlas_demo

- **Status:** PASS
- **Runtime:** 1.017 s
- **Last run:** 2026-08-15T20:39:26Z
- **Figures:** none

### sigma_decomposition

- **Status:** PASS
- **Runtime:** 1.145 s
- **Last run:** 2026-08-15T20:39:27Z
- **Figures:** none

### sign_restrictions_uhlig

- **Status:** PASS
- **Runtime:** 1.151 s
- **Last run:** 2026-08-15T20:39:28Z
- **Figures:** none

### spectral_business_cycle

- **Status:** PASS
- **Runtime:** 0.487 s
- **Last run:** 2026-08-15T20:39:28Z
- **Figures:**
  - `../puremacro/examples/output/spectral_business_cycle.png`

### svariv_mertens_ravn

- **Status:** PASS
- **Runtime:** 1.757 s
- **Last run:** 2026-08-15T20:39:30Z
- **Figures:**
  - `../puremacro/examples/output/svariv_mertens_ravn_irf.png`

### synthetic_control_demo

- **Status:** PASS
- **Runtime:** 1.315 s
- **Last run:** 2026-08-15T20:39:31Z
- **Figures:**
  - `../puremacro/examples/output/synthetic_control.png`

### synthetic_did_california_prop99

- **Status:** PASS
- **Runtime:** 2.452 s
- **Last run:** 2026-08-15T20:39:34Z
- **Figures:**
  - `../puremacro/examples/output/synthetic_did_prop99.png`

### tvar_threshold_demo

- **Status:** PASS
- **Runtime:** 0.874 s
- **Last run:** 2026-08-15T20:39:35Z
- **Figures:**
  - `../puremacro/examples/output/tvar_threshold.png`

### tvecm_demo

- **Status:** PASS
- **Runtime:** 0.894 s
- **Last run:** 2026-08-15T20:39:36Z
- **Figures:**
  - `../puremacro/examples/output/tvecm.png`

### tvp_var_demo

- **Status:** PASS
- **Runtime:** 2.635 s
- **Last run:** 2026-08-15T20:39:38Z
- **Figures:**
  - `../puremacro/examples/output/tvp_var.png`

### tvp_var_sv_demo

- **Status:** PASS
- **Runtime:** 4.169 s
- **Last run:** 2026-08-15T20:39:42Z
- **Figures:**
  - `../puremacro/examples/output/tvp_var_sv.png`

### vintage_revisions

- **Status:** PASS
- **Runtime:** 0.959 s
- **Last run:** 2026-08-15T20:39:43Z
- **Figures:**
  - `../puremacro/examples/output/vintage_revisions.png`

### wavelet_business_cycle

- **Status:** PASS
- **Runtime:** 0.503 s
- **Last run:** 2026-08-15T20:39:45Z
- **Figures:**
  - `../puremacro/examples/output/wavelet_business_cycle.png`

## SKIP (4)

### bloom2009

- **Status:** SKIP
- **Reason:** local data file missing
- **Runtime:** 1.203 s
- **Last run:** 2026-08-15T20:33:01Z
- **Figures:** none

### caldara_iacoviello

- **Status:** SKIP
- **Reason:** local data file missing
- **Runtime:** 1.211 s
- **Last run:** 2026-08-15T20:33:03Z
- **Figures:** none

### lp_panel_dk

- **Status:** SKIP
- **Reason:** local data file missing
- **Runtime:** 1.175 s
- **Last run:** 2026-08-15T20:38:11Z
- **Figures:** none

### narrative_event_study

- **Status:** SKIP
- **Reason:** requires local narrative-IV panel + MAV wedges file (not shipped)
- **Runtime:** 0.0 s
- **Last run:** 2026-08-15T20:38:19Z
- **Figures:** none
