> 🇬🇧 English · 🇪🇸 [Español](es/examples_gallery.md)

# puremacro examples gallery

Generated: 2026-11-05T00:48:27Z

| Status | Count |
|---|---|
| PASS | 86 |
| SKIP | 8 |
| FAIL | 1 |

## FAIL (1)

### romer_romer_2010_replication

- **Description:** Original-data Romer–Romer (2010) baseline and independent evidence.
- **Status:** FAIL
- **Reason:** RR2010 baseline: FAIL; h10=-3.08081276, t=-3.52499608; output=output/romer_romer_2010_replication
- **Runtime:** 1.458 s
- **Last run:** 2026-10-19T00:01:39Z
- **Figures:** none

## PASS (86)

### anderson_rubin_weak_iv_demo

- **Description:** Weak-instrument-robust Anderson-Rubin inference for local projections (LP-IV).
- **Status:** PASS
- **Runtime:** 3.281 s
- **Last run:** 2026-10-18T23:39:34Z
- **Figures:**
  - `../puremacro/examples/output/anderson_rubin_weak_iv_demo.png`

### bandpass_cycles_comparison

- **Description:** Business cycle filtering comparison: Baxter-King, Christiano-Fitzgerald, Beveridge-Nelson & Hamilton.
- **Status:** PASS
- **Runtime:** 1.645 s
- **Last run:** 2026-10-18T23:39:36Z
- **Figures:**
  - `../puremacro/examples/output/bandpass_cycles_comparison.png`

### berkowitz_demo

- **Description:** Berkowitz (2001) joint LR test: a more powerful PIT diagnostic.
- **Status:** PASS
- **Runtime:** 1.836 s
- **Last run:** 2026-10-18T23:39:38Z
- **Figures:** none

### bvar_fan_chart

- **Description:** Density forecast (fan chart) from a Minnesota-prior BVAR via Gibbs.
- **Status:** PASS
- **Runtime:** 2.647 s
- **Last run:** 2026-10-18T23:39:43Z
- **Figures:**
  - `../puremacro/examples/output/bvar_fan_chart.png`

### central_bank_narrative_sentiment

- **Description:** Central Bank Narrative Sentiment & Monetary Policy Transmission.
- **Status:** PASS
- **Runtime:** 3.199 s
- **Last run:** 2026-10-18T23:39:48Z
- **Figures:**
  - `../puremacro/examples/output/central_bank_narrative_sentiment.png`

### climate_dice_simulation

- **Description:** DICE Climate-Macroeconomic Simulation & Policy Analysis.
- **Status:** PASS
- **Runtime:** 1.397 s
- **Last run:** 2026-10-18T23:39:49Z
- **Figures:**
  - `../puremacro/examples/output/climate_dice_simulation.png`

### climate_sovereign_debt_risk

- **Description:** Climate Transition Risk and Sovereign Debt Sustainability.
- **Status:** PASS
- **Runtime:** 1.486 s
- **Last run:** 2026-10-18T23:39:51Z
- **Figures:**
  - `../puremacro/examples/output/climate_sovereign_debt_risk.png`

### cointegration_johansen

- **Description:** Engle-Granger and Johansen cointegration on a 3-variable simulated system.
- **Status:** PASS
- **Runtime:** 1.246 s
- **Last run:** 2026-10-18T23:39:52Z
- **Figures:** none

### dcc_volatility

- **Description:** Engle (2002) DCC: time-varying correlations among three return series.
- **Status:** PASS
- **Runtime:** 2.813 s
- **Last run:** 2026-10-18T23:39:55Z
- **Figures:**
  - `../puremacro/examples/output/dcc_volatility.png`

### dfm_bai_ng

- **Description:** High-dimensional dynamic factor model: Bai-Ng IC + PCA factors.
- **Status:** PASS
- **Runtime:** 1.106 s
- **Last run:** 2026-10-18T23:39:56Z
- **Figures:**
  - `../puremacro/examples/output/dfm_bai_ng.png`

### dfm_nowcast_kalman

- **Description:** Single-factor dynamic factor model via the Kalman filter.
- **Status:** PASS
- **Runtime:** 0.932 s
- **Last run:** 2026-10-18T23:39:57Z
- **Figures:**
  - `../puremacro/examples/output/dfm_nowcast_kalman.png`

### did_callaway_santanna_demo

- **Description:** Callaway-Sant'Anna + Sun-Abraham staggered DiD demo.
- **Status:** PASS
- **Runtime:** 1.518 s
- **Last run:** 2026-10-18T23:39:58Z
- **Figures:** none

### diebold_yilmaz_spillover

- **Description:** Diebold-Yilmaz (2012) volatility-spillover index on a synthetic 4-variable VAR.
- **Status:** PASS
- **Runtime:** 1.279 s
- **Last run:** 2026-10-18T23:40:00Z
- **Figures:** none

### distributional_trade_ge

- **Description:** Synthetic GE tariff experiment with synthetic household groups.
- **Status:** PASS
- **Runtime:** 1.769 s
- **Last run:** 2026-10-18T23:48:28Z
- **Figures:** none

### dsge_ar1_demo

- **Description:** Toy AR(1) state-space — demonstrates puremacro.dsge.estimate_dsge.
- **Status:** PASS
- **Runtime:** 55.334 s
- **Last run:** 2026-10-18T23:40:56Z
- **Figures:** none

### dsge_markov_switching

- **Description:** Markov-Switching DSGE Perturbation: Monetary Policy Regimes & Analytical GIRFs.
- **Status:** PASS
- **Runtime:** 2.345 s
- **Last run:** 2026-10-18T23:40:59Z
- **Figures:**
  - `../puremacro/examples/output/dsge_markov_switching.png`

### dsge_news_shocks

- **Description:** DSGE News & Anticipated Shocks Engine: State Augmentation & Variance Decomposition.
- **Status:** PASS
- **Runtime:** 2.255 s
- **Last run:** 2026-10-18T23:41:01Z
- **Figures:**
  - `../puremacro/examples/output/dsge_news_shocks.png`

### dsge_nk_sketchpad

- **Description:** Three-equation New Keynesian model, written as equations rather than matrices.
- **Status:** PASS
- **Runtime:** 1.674 s
- **Last run:** 2026-10-18T23:41:02Z
- **Figures:** none

### dsge_nuts_estimation

- **Description:** Bayesian DSGE Estimation via No-U-Turn Sampler (NUTS) Hamiltonian Monte Carlo.
- **Status:** PASS
- **Runtime:** 45.685 s
- **Last run:** 2026-10-18T23:41:48Z
- **Figures:**
  - `../puremacro/examples/output/dsge_nuts_autocorr.png`
  - `../puremacro/examples/output/dsge_nuts_estimation.png`
  - `../puremacro/examples/output/dsge_nuts_posterior.png`
  - `../puremacro/examples/output/dsge_nuts_trace.png`

### dsge_optimal_discretion

- **Description:** Optimal Monetary Policy: Markov-Perfect Discretion vs LQ Commitment.
- **Status:** PASS
- **Runtime:** 2.118 s
- **Last run:** 2026-10-18T23:41:50Z
- **Figures:**
  - `../puremacro/examples/output/dsge_optimal_discretion.png`

### dsge_particle_filter_sv

- **Description:** Nonlinear DSGE Particle Filtering with Stochastic Volatility (BPF vs Constant Baseline).
- **Status:** PASS
- **Runtime:** 2.324 s
- **Last run:** 2026-10-18T23:41:53Z
- **Figures:**
  - `../puremacro/examples/output/dsge_particle_filter_sv.png`

### dsge_rbc_klein

- **Description:** Linearised RBC model solved with the Klein (2000) QZ method.
- **Status:** PASS
- **Runtime:** 1.751 s
- **Last run:** 2026-10-18T23:41:54Z
- **Figures:**
  - `../puremacro/examples/output/dsge_rbc_klein.png`

### dsge_var_estimation

- **Description:** DSGE-VAR Hybrid Estimation & Marginal Data Density Optimization.
- **Status:** PASS
- **Runtime:** 2.279 s
- **Last run:** 2026-10-18T23:41:57Z
- **Figures:**
  - `../puremacro/examples/output/dsge_var_estimation.png`

### empirical_benchmark_replications

- **Description:** Empirical Benchmark Replications: Galí (1999) & Mertens-Ravn (2013).
- **Status:** PASS
- **Runtime:** 2.674 s
- **Last run:** 2026-10-18T23:41:59Z
- **Figures:**
  - `../puremacro/examples/output/empirical_benchmark_replications.png`

### exact_diffuse_kalman

- **Description:** Exact-diffuse Kalman initialisation (Koopman-Durbin 2003).
- **Status:** PASS
- **Runtime:** 0.718 s
- **Last run:** 2026-10-18T23:42:00Z
- **Figures:**
  - `../puremacro/examples/output/exact_diffuse_kalman.png`

### fetch_fred_demo

- **Description:** Live FRED fetch + LP + markdown report — full pipeline.
- **Status:** PASS
- **Runtime:** 1.667 s
- **Last run:** 2026-10-18T23:42:02Z
- **Figures:**
  - `../puremacro/examples/output/fetch_fred_demo.png`

### first_brief_fiscal

- **Description:** Fiscal policy brief: the US government spending multiplier at two and four years, with and without slack, on Ramey and Zubairy's historical data.
- **Status:** PASS
- **Runtime:** 1.705 s
- **Last run:** 2026-11-05T00:48:27Z
- **Figures:** none

### first_brief_tariffs

- **Description:** Your first policy brief: a 10% US tariff across all modeled Chinese sectors on the bundled OECD ICIO 2019 table, national EV with its price / factor-income / fiscal split, and synthetic household g...
- **Status:** PASS
- **Runtime:** 6.051 s
- **Last run:** 2026-10-29T05:09:05Z
- **Figures:** none

### fm_ols_dols_demo

- **Description:** Phillips-Hansen FM-OLS and Stock-Watson DOLS vs naive OLS on a known cointegrating DGP.
- **Status:** PASS
- **Runtime:** 1.643 s
- **Last run:** 2026-10-18T23:42:03Z
- **Figures:**
  - `../puremacro/examples/output/fm_ols_dols.png`

### forecast_density_eval

- **Description:** Density-forecast evaluation: PIT, CRPS, KLIC for two competing models.
- **Status:** PASS
- **Runtime:** 1.464 s
- **Last run:** 2026-10-18T23:42:05Z
- **Figures:**
  - `../puremacro/examples/output/forecast_density_eval.png`

### gali_1999_hours

- **Description:** Gali (1999 AER) technology shocks and the hours debate — BQ long-run SVAR on real US data.
- **Status:** PASS
- **Runtime:** 1.905 s
- **Last run:** 2026-10-18T23:42:07Z
- **Figures:**
  - `../puremacro/examples/output/gali_1999_hours.png`

### garch_midas_macro_volatility

- **Description:** GARCH-MIDAS mixed-frequency volatility model (Engle, Ghysels & Sohn 2013).
- **Status:** PASS
- **Runtime:** 1.672 s
- **Last run:** 2026-10-18T23:42:08Z
- **Figures:**
  - `../puremacro/examples/output/garch_midas_macro_volatility.png`

### gfevd_spillover

- **Description:** Diebold-Yilmaz spillover with GFEVD: order-invariance demo.
- **Status:** PASS
- **Runtime:** 1.004 s
- **Last run:** 2026-10-18T23:42:09Z
- **Figures:** none

### gk_robust_from_gibbs

- **Description:** Posterior-aware sign-restriction bands: Giacomini-Kitagawa 2021 fed by ``minnesota_gibbs`` posterior draws.
- **Status:** PASS
- **Runtime:** 4.633 s
- **Last run:** 2026-10-18T23:42:14Z
- **Figures:**
  - `../puremacro/examples/output/gk_robust_from_gibbs.png`

### gk_robust_signs

- **Description:** Giacomini-Kitagawa (2021) robust bands vs RWZ posterior bands on a sign-restricted SVAR.
- **Status:** PASS
- **Runtime:** 2.787 s
- **Last run:** 2026-10-18T23:42:17Z
- **Figures:**
  - `../puremacro/examples/output/gk_robust_signs.png`

### global_fiscal_stance

- **Description:** Global fiscal stance — sum of signed narrative magnitudes across the homogeneous IV panel, per quarter.
- **Status:** PASS
- **Runtime:** 0.943 s
- **Last run:** 2026-10-18T23:42:18Z
- **Figures:**
  - `../puremacro/examples/output/global_fiscal_stance.csv`
  - `../puremacro/examples/output/global_fiscal_stance.png`

### glp_lambda_search

- **Description:** Empirical-Bayes hyperparameter search for a Minnesota BVAR (Giannone-Lenza-Primiceri 2015, simplified).
- **Status:** PASS
- **Runtime:** 1.551 s
- **Last run:** 2026-10-18T23:42:19Z
- **Figures:**
  - `../puremacro/examples/output/glp_lambda_search.png`

### hank_sequence_space

- **Description:** Sequence-Space HANK Model Simulation & Multiplier Decomposition.
- **Status:** PASS
- **Runtime:** 2.219 s
- **Last run:** 2026-10-18T23:42:22Z
- **Figures:**
  - `../puremacro/examples/output/hank_sequence_space.png`

### hank_sequence_space_bridge

- **Description:** Heterogeneous-Agent New Keynesian (HANK) Sequence-Space Bridge (.mod).
- **Status:** PASS
- **Runtime:** 2.263 s
- **Last run:** 2026-10-18T23:42:24Z
- **Figures:**
  - `../puremacro/examples/output/hank_sequence_space_bridge.png`

### har_realized_vol

- **Description:** Corsi (2009) HAR-RV regression on simulated intra-day returns.
- **Status:** PASS
- **Runtime:** 1.624 s
- **Last run:** 2026-10-18T23:42:26Z
- **Figures:**
  - `../puremacro/examples/output/har_realized_vol.png`

### hfi_gertler_karadi

- **Description:** HFI monetary-policy shock: synthetic Gertler-Karadi 2015-style pipeline.
- **Status:** PASS
- **Runtime:** 1.564 s
- **Last run:** 2026-10-18T23:42:27Z
- **Figures:**
  - `../puremacro/examples/output/hfi_gertler_karadi.png`

### kilian_2009_oil

- **Description:** Kilian (2009 AER) "Not All Oil Price Shocks Are Alike" — recursive oil-market VAR with historical decomposition.
- **Status:** PASS
- **Runtime:** 2.686 s
- **Last run:** 2026-10-18T23:42:30Z
- **Figures:**
  - `../puremacro/examples/output/kilian_2009_oil.png`

### la_lp_pmw_demo

- **Description:** Lag-augmented LP (Montiel Olea & Plagborg-Møller 2021) vs Newey-West Jordà LP.
- **Status:** PASS
- **Runtime:** 2.008 s
- **Last run:** 2026-10-18T23:42:32Z
- **Figures:**
  - `../puremacro/examples/output/la_lp_pmw.png`

### labor_flows_demo

- **Description:** Labor-force transitions demo using the Shimer 2-state fallback.
- **Status:** PASS
- **Runtime:** 0.415 s
- **Last run:** 2026-10-18T23:42:33Z
- **Figures:** none

### lp_asymmetric_tenreyro

- **Description:** Tenreyro-Thwaites (2016)-style asymmetric local projection on synthetic data.
- **Status:** PASS
- **Runtime:** 1.16 s
- **Last run:** 2026-10-18T23:42:34Z
- **Figures:** none

### lp_smooth_demo

- **Description:** Barnichon-Brownlees (2019) smoothed local projections on synthetic data.
- **Status:** PASS
- **Runtime:** 1.178 s
- **Last run:** 2026-10-18T23:42:36Z
- **Figures:** none

### lwui_wage_demo

- **Description:** LWUI-wage demo: labor x uncertainty x wage triple co-occurrence.
- **Status:** PASS
- **Runtime:** 1.352 s
- **Last run:** 2026-10-18T23:42:38Z
- **Figures:** none

### mcmc_diagnostics

- **Description:** MCMC convergence diagnostics on a real Gibbs sampler.
- **Status:** PASS
- **Runtime:** 1.786 s
- **Last run:** 2026-10-18T23:42:40Z
- **Figures:**
  - `../puremacro/examples/output/mcmc_diagnostics.png`

### midas_quarterly_monthly

- **Description:** MIDAS nowcasting: quarterly y on monthly x.
- **Status:** PASS
- **Runtime:** 1.703 s
- **Last run:** 2026-10-18T23:42:41Z
- **Figures:**
  - `../puremacro/examples/output/midas_quarterly_monthly.png`

### mle_sandwich_demo

- **Description:** Bring-your-own-likelihood: Student-t MLE with sandwich SE.
- **Status:** PASS
- **Runtime:** 1.773 s
- **Last run:** 2026-10-18T23:42:43Z
- **Figures:** none

### ms_var_business_cycle

- **Description:** Two-regime MSIH MS-VAR for a recession dating exercise.
- **Status:** PASS
- **Runtime:** 2.063 s
- **Last run:** 2026-10-18T23:42:45Z
- **Figures:**
  - `../puremacro/examples/output/ms_var_business_cycle.png`

### narrative_bursts_and_transcripts

- **Description:** Narrative Macroeconomics: Transcripts, Dynamic Topics, Burst Anomalies, and SVARs.
- **Status:** PASS
- **Runtime:** 1.846 s
- **Last run:** 2026-10-18T23:42:47Z
- **Figures:**
  - `../puremacro/examples/output/narrative_bursts_and_transcripts.png`

### narrative_custom_corpus

- **Description:** Custom-corpus pipeline (workflow A): user-supplied texts → keyword scoring → quarterly IV → diagnostics. No network, no LLM.
- **Status:** PASS
- **Runtime:** 1.434 s
- **Last run:** 2026-10-18T23:42:48Z
- **Figures:**
  - `../puremacro/examples/output/narrative_custom_corpus.png`

### narrative_g7_panel

- **Description:** G7 narrative-instrument panel demo.
- **Status:** PASS
- **Runtime:** 1.508 s
- **Last run:** 2026-10-18T23:42:50Z
- **Figures:**
  - `../puremacro/examples/output/narrative_g7_panel.png`

### narrative_homogeneous_panel

- **Description:** Build the homogeneous cross-country fiscal-narrative-IV panel.
- **Status:** PASS
- **Runtime:** 1.764 s
- **Last run:** 2026-10-18T23:42:52Z
- **Figures:**
  - `../puremacro/examples/output/homogeneous_panel.png`
  - `../puremacro/examples/output/narrative_iv_panel_consumption.csv`
  - `../puremacro/examples/output/narrative_iv_panel_dedup_audit.csv`
  - `../puremacro/examples/output/narrative_iv_panel_investment.csv`
  - `../puremacro/examples/output/narrative_iv_panel_long.csv`
  - `../puremacro/examples/output/narrative_iv_panel_quarterly.csv`

### narrative_indices_demo

- **Description:** Narrative indices demo — assembles all six indices from a synthetic in-process corpus and prints summary statistics.
- **Status:** PASS
- **Runtime:** 1.244 s
- **Last run:** 2026-10-18T23:42:53Z
- **Figures:** none

### narrative_llm_pipeline

- **Description:** End-to-end LLM pipeline (workflow C): a small corpus → LLM scoring → ``NarrativeInstrument``.
- **Status:** PASS
- **Runtime:** 0.99 s
- **Last run:** 2026-10-18T23:42:54Z
- **Figures:** none

### narrative_local_llm

- **Description:** Run puremacro's narrative LLM features for FREE on a local model.
- **Status:** PASS
- **Runtime:** 0.999 s
- **Last run:** 2026-10-18T23:42:55Z
- **Figures:** none

### narrative_multilingual_live

- **Description:** Live multilingual fetch + IMF Article-IV + Google News across the G7 countries plus the IMF and ECB.
- **Status:** PASS
- **Runtime:** 1.438 s
- **Last run:** 2026-10-18T23:42:56Z
- **Figures:**
  - `../puremacro/examples/output/narrative_multilingual_live.png`

### narrative_ramey_2011

- **Description:** Replication-layer demo (workflow B): Ramey (2011) defense-news narrative instrument → ``proxy_svar`` on a small US-style VAR → spending-multiplier-style IRF.
- **Status:** PASS
- **Runtime:** 1.808 s
- **Last run:** 2026-10-18T23:42:58Z
- **Figures:**
  - `../puremacro/examples/output/narrative_ramey_2011.png`

### narrative_romer_romer_2010

- **Description:** Replication-layer demo (workflow B): Romer-Romer (2010) US tax narrative shocks → comparison against the Ramey defense-news series.
- **Status:** PASS
- **Runtime:** 1.478 s
- **Last run:** 2026-10-18T23:43:00Z
- **Figures:**
  - `../puremacro/examples/output/narrative_rr_vs_ramey.png`

### narrative_sign_adrr

- **Description:** Narrative sign restrictions (Antolín-Díaz & Rubio-Ramírez 2018) — the October 1979 Volcker episode as an identification input.
- **Status:** PASS
- **Runtime:** 1.75 s
- **Last run:** 2026-10-18T23:43:02Z
- **Figures:**
  - `../puremacro/examples/output/narrative_sign_adrr.png`

### narrative_us_canonical_compare

- **Description:** Link a user's US narrative-instrument series to the canonical published instruments: Ramey (2011) defense news, Romer-Romer (2010) exogenous tax, Mertens-Ravn (2013) unanticipated tax.
- **Status:** PASS
- **Runtime:** 1.86 s
- **Last run:** 2026-10-18T23:43:04Z
- **Figures:**
  - `../puremacro/examples/output/narrative_us_canonical_compare.png`

### non_gaussian_svar

- **Description:** SVAR identification by non-Gaussianity (Lanne-Meitz-Saikkonen 2017).
- **Status:** PASS
- **Runtime:** 1.254 s
- **Last run:** 2026-10-18T23:43:05Z
- **Figures:**
  - `../puremacro/examples/output/non_gaussian_svar.png`

### nowcasting_gdp_news

- **Description:** Mixed-Frequency GDP Nowcasting & News Decomposition.
- **Status:** PASS
- **Runtime:** 1.636 s
- **Last run:** 2026-10-18T23:43:06Z
- **Figures:**
  - `../puremacro/examples/output/nowcasting_gdp_news.png`

### penalized_macro_forecasting

- **Description:** High-Dimensional Penalized Macroeconomic Forecasting.
- **Status:** PASS
- **Runtime:** 1.565 s
- **Last run:** 2026-10-18T23:43:08Z
- **Figures:**
  - `../puremacro/examples/output/penalized_macro_forecasting.png`

### posterior_predictive_fan

- **Description:** Posterior-predictive fan chart from BVAR Gibbs draws.
- **Status:** PASS
- **Runtime:** 1.34 s
- **Last run:** 2026-10-18T23:43:09Z
- **Figures:**
  - `../puremacro/examples/output/posterior_predictive_fan.png`

### ramey_zubairy_2018_multipliers

- **Description:** Ramey-Zubairy (2018 JPE) spending multipliers — LP-IV with military news shocks on 126 years of US data.
- **Status:** PASS
- **Runtime:** 2.443 s
- **Last run:** 2026-10-18T23:43:12Z
- **Figures:**
  - `../puremacro/examples/output/ramey_zubairy_2018_multipliers.png`

### regime_workflow

- **Description:** End-to-end regime workflow: Bai-Perron breaks → regime indicator → regime-conditional LP.
- **Status:** PASS
- **Runtime:** 2.042 s
- **Last run:** 2026-10-18T23:43:14Z
- **Figures:**
  - `../puremacro/examples/output/regime_workflow.png`

### report_workflow

- **Description:** Render LP and IRF results as Markdown / LaTeX tables ready to drop into a paper or a Quarto / Jupyter notebook on iPad.
- **Status:** PASS
- **Runtime:** 0.997 s
- **Last run:** 2026-10-18T23:43:15Z
- **Figures:** none

### shock_atlas_demo

- **Description:** Shock-atlas registry demo.
- **Status:** PASS
- **Runtime:** 0.964 s
- **Last run:** 2026-10-18T23:43:18Z
- **Figures:** none

### sigma_decomposition

- **Description:** SigmaObject volatility decomposition — Python equivalent of ``MASTER_VolatilityDecomposition_Labor.m``.
- **Status:** PASS
- **Runtime:** 0.981 s
- **Last run:** 2026-10-18T23:43:19Z
- **Figures:** none

### sign_restrictions_uhlig

- **Description:** Uhlig (2005)-style sign-restricted SVAR on a synthetic 3-variable DGP.
- **Status:** PASS
- **Runtime:** 1.075 s
- **Last run:** 2026-10-18T23:43:20Z
- **Figures:** none

### spectral_business_cycle

- **Description:** Spectral analysis of a synthetic business-cycle-style series.
- **Status:** PASS
- **Runtime:** 0.686 s
- **Last run:** 2026-10-18T23:43:21Z
- **Figures:**
  - `../puremacro/examples/output/spectral_business_cycle.png`

### structural_irf_matching

- **Description:** Empirical LPs -> joint sampling covariance -> a three-equation NK model.
- **Status:** PASS
- **Runtime:** 1.688 s
- **Last run:** 2026-10-18T23:43:23Z
- **Figures:** none

### svariv_mertens_ravn

- **Description:** Mertens-Ravn (2013)-style proxy-SVAR replication on synthetic data.
- **Status:** PASS
- **Runtime:** 1.372 s
- **Last run:** 2026-10-18T23:43:24Z
- **Figures:**
  - `../puremacro/examples/output/svariv_mertens_ravn_irf.png`

### synthetic_control_demo

- **Description:** Synthetic control on a simulated treatment.
- **Status:** PASS
- **Runtime:** 0.958 s
- **Last run:** 2026-10-18T23:43:25Z
- **Figures:**
  - `../puremacro/examples/output/synthetic_control.png`

### synthetic_did_california_prop99

- **Description:** Synthetic Difference-in-Differences causal policy evaluation (Arkhangelsky et al. 2021).
- **Status:** PASS
- **Runtime:** 1.783 s
- **Last run:** 2026-10-18T23:43:27Z
- **Figures:**
  - `../puremacro/examples/output/synthetic_did_prop99.png`

### tvar_threshold_demo

- **Description:** Self-exciting threshold VAR: regime-dependent dynamics.
- **Status:** PASS
- **Runtime:** 1.585 s
- **Last run:** 2026-10-18T23:43:28Z
- **Figures:**
  - `../puremacro/examples/output/tvar_threshold.png`

### tvecm_demo

- **Description:** Threshold cointegration / TVECM demo.
- **Status:** PASS
- **Runtime:** 1.285 s
- **Last run:** 2026-10-18T23:43:30Z
- **Figures:**
  - `../puremacro/examples/output/tvecm.png`

### tvp_var_demo

- **Description:** Time-varying-parameter VAR (Primiceri-style FFBS Gibbs).
- **Status:** PASS
- **Runtime:** 3.758 s
- **Last run:** 2026-10-18T23:43:34Z
- **Figures:**
  - `../puremacro/examples/output/tvp_var.png`

### tvp_var_sv_demo

- **Description:** TVP-VAR with stochastic volatility (Primiceri 2005, KSC sampler).
- **Status:** PASS
- **Runtime:** 5.524 s
- **Last run:** 2026-10-18T23:43:39Z
- **Figures:**
  - `../puremacro/examples/output/tvp_var_sv.png`

### vintage_news_noise

- **Description:** Are GDP revisions news or noise? A cross-country answer.
- **Status:** PASS
- **Runtime:** 33.729 s
- **Last run:** 2026-10-18T23:49:22Z
- **Figures:** none

### vintage_revisions

- **Description:** Real-time vintages and forecast revisions.
- **Status:** PASS
- **Runtime:** 0.831 s
- **Last run:** 2026-10-18T23:43:45Z
- **Figures:**
  - `../puremacro/examples/output/vintage_revisions.png`

### vulnerable_growth

- **Description:** Adrian-Boyarchenko-Giannone (2019) vulnerable-growth demo.
- **Status:** PASS
- **Runtime:** 56.654 s
- **Last run:** 2026-10-18T23:44:42Z
- **Figures:**
  - `../puremacro/examples/output/vulnerable_growth.png`

### wavelet_business_cycle

- **Description:** Wavelet variance decomposition of a synthetic GDP-like series.
- **Status:** PASS
- **Runtime:** 0.632 s
- **Last run:** 2026-10-18T23:44:42Z
- **Figures:**
  - `../puremacro/examples/output/wavelet_business_cycle.png`

## SKIP (8)

### asset_composition_dynamics

- **Description:** Investment composition over time, per country, from the augmented asset-breakdown panel.
- **Status:** SKIP
- **Reason:** local data file missing
- **Runtime:** 0.738 s
- **Last run:** 2026-10-18T23:39:35Z
- **Figures:** none

### bloom2009

- **Description:** Bloom (2009)-style uncertainty-shock VAR replication on quarterly USA data.
- **Status:** SKIP
- **Reason:** local data file missing
- **Runtime:** 2.231 s
- **Last run:** 2026-10-18T23:39:40Z
- **Figures:** none

### caldara_iacoviello

- **Description:** Caldara-Iacoviello (2022)-style geopolitical-risk shock VAR on USA quarterly data.
- **Status:** SKIP
- **Reason:** local data file missing
- **Runtime:** 1.707 s
- **Last run:** 2026-10-18T23:39:45Z
- **Figures:** none

### govt_vs_private_investment

- **Description:** Government vs private investment shares from the augmented panel.
- **Status:** SKIP
- **Reason:** local data file missing
- **Runtime:** 0.377 s
- **Last run:** 2026-10-18T23:42:20Z
- **Figures:** none

### lp_panel_dk

- **Description:** Panel local projection of GDP on uncertainty with Driscoll-Kraay SE.
- **Status:** SKIP
- **Reason:** local data file missing
- **Runtime:** 1.474 s
- **Last run:** 2026-10-18T23:42:35Z
- **Figures:** none

### narrative_event_study

- **Description:** Event study around the largest narrative-IV episodes per country.
- **Status:** SKIP
- **Reason:** requires local narrative-IV panel + MAV wedges file (not shipped)
- **Runtime:** 0.0 s
- **Last run:** 2026-10-18T23:42:48Z
- **Figures:** none

### narrative_panel_lp

- **Description:** Cross-country fiscal-multiplier panel local projection using the homogeneous narrative-IV panel as the shock series and the wedges-panel RGDP as the outcome.
- **Status:** SKIP
- **Reason:** local data file missing
- **Runtime:** 0.329 s
- **Last run:** 2026-10-18T23:42:57Z
- **Figures:** none

### scf_aiyagari

- **Description:** Aiyagari (1994) calibrated to the Survey of Consumer Finances.
- **Status:** SKIP
- **Reason:** network unavailable
- **Runtime:** 1.193 s
- **Last run:** 2026-10-18T23:43:17Z
- **Figures:** none
