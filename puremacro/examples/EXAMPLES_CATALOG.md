# `puremacro.examples` — catalog

Standalone replication / demo scripts that each showcase one method. Every script exposes a `main()` entry-point and runs as:

```bash
python -m puremacro.examples.<name>
```

There are **83 scripts** in `puremacro/examples/` (66 described below; the remainder are listed at the end).

## Recommended starter shortlist

Twelve scripts that together span the methods catalogue. Read in this order if you want a one-week tour:

| # | Script | What it teaches |
|---|---|---|
| 1 | `bloom2009` | Cholesky VAR replication of Bloom 2009 uncertainty shock |
| 2 | `svariv_mertens_ravn` | Proxy-SVAR (external instrument) following Mertens-Ravn |
| 3 | `sign_restrictions_uhlig` | Sign-restricted SVAR following Uhlig 2005 |
| 4 | `lp_smooth_demo` | Barnichon-Brownlees smoothed local projections |
| 5 | `lp_panel_dk` | Panel LP with Driscoll-Kraay HAC standard errors |
| 6 | `ms_var_business_cycle` | Markov-switching VAR for two-regime business cycle |
| 7 | `dfm_bai_ng` | Dynamic factor model + Bai-Ng information criterion |
| 8 | `bvar_fan_chart` | Minnesota-prior BVAR via Gibbs -> density fan chart |
| 9 | `narrative_indices_demo` | Build LTUI/LWUI from central-bank speech corpora |
| 10 | `wavelet_business_cycle` | Wavelet decomposition of business-cycle frequencies |
| 11 | `vulnerable_growth` | Adrian-Boyarchenko-Giannone-style quantile growth-at-risk |
| 12 | `sigma_decomposition` | SigmaObject: var(w'X) into diagonal vs covariance terms |
| 13 | `did_callaway_santanna_demo` | Modern staggered DiD (CS 2021 + SA 2021) vs naive TWFE |

## All scripts, grouped by topic

### Narrative-text uncertainty (LTUI / LWUI / LUI)

| Script | LOC | Headline |
|---|---:|---|
| `narrative_custom_corpus` | 116 | Custom-corpus pipeline (workflow A): user-supplied texts → keyword |
| `narrative_event_study` | 171 | Event study around the largest narrative-IV episodes per country. |
| `narrative_g7_panel` | 193 | G7 narrative-instrument panel demo. |
| `narrative_homogeneous_panel` | 203 | Build the homogeneous cross-country fiscal-narrative-IV panel. |
| `narrative_indices_demo` | 68 | Narrative indices demo — assembles all six indices from a synthetic |
| `lwui_wage_demo` | 81 | LWUI-wage demo: labor x uncertainty x wage triple co-occurrence (NEW 2026-05-17) |
| `narrative_llm_pipeline` | 94 | End-to-end LLM pipeline (workflow C): a small corpus → LLM scoring → |
| `narrative_multilingual_live` | 145 | Live multilingual fetch + IMF Article-IV + Google News across the |
| `narrative_panel_lp` | 151 | Cross-country fiscal-multiplier panel local projection using the |
| `narrative_ramey_2011` | 159 | Replication-layer demo (workflow B): Ramey (2011) defense-news |
| `narrative_romer_romer_2010` | 145 | Replication-layer demo (workflow B): Romer-Romer (2010) US tax |
| `narrative_us_canonical_compare` | 195 | Link a user's US narrative-instrument series to the canonical |

### SVAR identification menu

| Script | LOC | Headline |
|---|---:|---|
| `narrative_sign_adrr` | 163 | Narrative sign restrictions (Antolín-Díaz & Rubio-Ramírez 2018): the October 1979 Volcker episode (as a `NarrativeEvent`) tightens a monetary SVAR's sign-identified set. Synthetic-but-calibrated DGP. |
| `non_gaussian_svar` | 95 | SVAR identification by non-Gaussianity (Lanne-Meitz-Saikkonen 2017). |
| `sign_restrictions_uhlig` | 84 | Uhlig (2005)-style sign-restricted SVAR on a synthetic 3-variable DGP. |
| `svariv_mertens_ravn` | 213 | Mertens-Ravn (2013)-style proxy-SVAR replication on synthetic data. |
| `gk_robust_from_gibbs` | 98 | Posterior-aware sign-restriction bands: Giacomini-Kitagawa 2021 |
| `gk_robust_signs` | 107 | Giacomini-Kitagawa (2021) robust bands vs RWZ posterior bands |
| `hfi_gertler_karadi` | 73 | HFI monetary-policy shock: synthetic Gertler-Karadi 2015-style pipeline. |

### Local projection methods

| Script | LOC | Headline |
|---|---:|---|
| `la_lp_pmw_demo` | 90 | Lag-augmented LP (Plagborg-Møller-Wolf 2021) vs Newey-West Jordà LP. |
| `lp_asymmetric_tenreyro` | 59 | Tenreyro-Thwaites (2016)-style asymmetric local projection on synthetic data. |
| `lp_panel_dk` | 53 | Panel local projection of GDP on uncertainty with Driscoll-Kraay SE. |
| `lp_smooth_demo` | 55 | Barnichon-Brownlees (2019) smoothed local projections on synthetic data. |
| `ramey_zubairy_2018_multipliers` | 392 | Ramey-Zubairy (2018 JPE) spending multipliers on frozen 1889-2015 US data: one-step LP-IV cumulative multipliers ~0.67-0.71, slack-state (unemp >= 6.5) variant via `lp_state_dep`, and the weak-IV core — per-horizon Olea-Pflueger F collapsing in slack, answered with `anderson_rubin_test` / `msw_bands`. |

### Regime / time-varying coefficients

| Script | LOC | Headline |
|---|---:|---|
| `ms_var_business_cycle` | 100 | Hamilton (1989)-style two-regime MS-VAR for a recession dating exercise. |
| `regime_workflow` | 108 | End-to-end regime workflow: Bai-Perron breaks → regime indicator |
| `tvar_threshold_demo` | 90 | Self-exciting threshold VAR: regime-dependent dynamics. |
| `tvp_var_demo` | 108 | Time-varying-parameter VAR (Primiceri-style FFBS Gibbs). |
| `tvp_var_sv_demo` | 101 | TVP-VAR with stochastic volatility (Primiceri 2005, KSC sampler). |

### Forecasting + DFM + nowcasting

| Script | LOC | Headline |
|---|---:|---|
| `bvar_fan_chart` | 106 | Density forecast (fan chart) from a Minnesota-prior BVAR via Gibbs. |
| `dfm_bai_ng` | 94 | High-dimensional dynamic factor model: Bai-Ng IC + PCA factors. |
| `dfm_nowcast_kalman` | 94 | Single-factor dynamic factor model via the Kalman filter. |
| `exact_diffuse_kalman` | 102 | Exact-diffuse Kalman initialisation (Koopman-Durbin 2003). |
| `forecast_density_eval` | 125 | Density-forecast evaluation: PIT, CRPS, KLIC for two competing models. |
| `glp_lambda_search` | 90 | Empirical-Bayes hyperparameter search for a Minnesota BVAR |
| `midas_quarterly_monthly` | 106 | MIDAS nowcasting: quarterly y on monthly x. |
| `posterior_predictive_fan` | 84 | Posterior-predictive fan chart from BVAR Gibbs draws. |
| `vintage_revisions` | 114 | Real-time vintages and forecast revisions. |

### Classic shock-VAR replications

| Script | LOC | Headline |
|---|---:|---|
| `bloom2009` | 104 | Bloom (2009)-style uncertainty-shock VAR replication on quarterly USA data. |
| `caldara_iacoviello` | 64 | Caldara-Iacoviello (2022)-style geopolitical-risk shock VAR on USA quarterly data. |
| `dsge_rbc_klein` | 139 | Linearised RBC model solved with the Klein (2000) QZ method. |
| `gali_1999_hours` | 210 | Gali (1999 AER) technology shocks and the hours debate: BQ long-run SVAR on frozen FRED data; hours fall on impact in differences, flip in levels (CEV critique). |
| `kilian_2009_oil` | 301 | Kilian (2009 AER) oil-market VAR on frozen FRED data: supply vs demand IRFs, real-price FEVD, and the stacked historical decomposition 1975-2007 (debut of `historical_decomp`). |

### Volatility, connectedness & spillover

| Script | LOC | Headline |
|---|---:|---|
| `dcc_volatility` | 79 | Engle (2002) DCC: time-varying correlations among three return series. |
| `diebold_yilmaz_spillover` | 59 | Diebold-Yilmaz (2012) volatility-spillover index on a synthetic 4-variable VAR. |
| `gfevd_spillover` | 86 | Diebold-Yilmaz spillover with GFEVD: order-invariance demo. |
| `har_realized_vol` | 103 | Corsi (2009) HAR-RV regression on simulated intra-day returns. |

### Cointegration & long-run

| Script | LOC | Headline |
|---|---:|---|
| `cointegration_johansen` | 68 | Engle-Granger and Johansen cointegration on a 3-variable simulated system. |
| `fm_ols_dols_demo` | 107 | Phillips-Hansen FM-OLS and Stock-Watson DOLS vs naive OLS on |
| `tvecm_demo` | 79 | Threshold cointegration / TVECM demo. |

### Frequency / wavelet / decompositions

| Script | LOC | Headline |
|---|---:|---|
| `sigma_decomposition` | 92 | SigmaObject volatility decomposition — Python equivalent of |
| `spectral_business_cycle` | 94 | Spectral analysis of a synthetic business-cycle-style series. |
| `wavelet_business_cycle` | 98 | Wavelet variance decomposition of a synthetic GDP-like series. |

### Inference diagnostics & density evaluation

| Script | LOC | Headline |
|---|---:|---|
| `berkowitz_demo` | 67 | Berkowitz (2001) joint LR test: a more powerful PIT diagnostic. |
| `mcmc_diagnostics` | 103 | MCMC convergence diagnostics on a real Gibbs sampler. |
| `mle_sandwich_demo` | 96 | Bring-your-own-likelihood: Student-t MLE with sandwich SE. |
| `vulnerable_growth` | 92 | Adrian-Boyarchenko-Giannone (2019) vulnerable-growth demo. |
| `synthetic_control_demo` | 102 | Synthetic control on a simulated treatment. |

### Applied panels & fiscal

| Script | LOC | Headline |
|---|---:|---|
| `asset_composition_dynamics` | 169 | Investment composition over time, per country, from the augmented |
| `did_callaway_santanna_demo` | — | Modern DiD: synthetic staggered panel; CS 2021 + SA 2021 overlay |
| `global_fiscal_stance` | 112 | Global fiscal stance — sum of signed narrative magnitudes across the |
| `govt_vs_private_investment` | 135 | Government vs private investment shares from the augmented panel. |
| `fetch_fred_demo` | 113 | Live FRED fetch + LP + markdown report — full pipeline. |
| `labor_flows_demo` | — | Labor markets: E/U/N transitions (Shimer 2-state or CPS flows) + supply-demand decomposition |
| `report_workflow` | 80 | Render LP and IRF results as Markdown / LaTeX tables ready to drop |

### Data infrastructure

| Script | LOC | Headline |
|---|---:|---|
| `shock_atlas_demo` | — | Data infrastructure: walks `puremacro.shock_atlas` registry; reports load status per shock |

## How to run

From the repository root, after `pip install -e ./puremacro`:

```bash
python -m puremacro.examples.bloom2009
python -m puremacro.examples.lp_smooth_demo
python -m puremacro.examples.narrative_indices_demo
```

Most scripts print summary stats and save a figure to `output_figures/`.
A few (the narrative-text and panel demos) require `data/processed/panel_Q.parquet`; see `README.md` for how to build it.

## Not yet described

Scripts that ship but have no entry above yet:

- `anderson_rubin_weak_iv_demo`
- `bandpass_cycles_comparison`
- `central_bank_narrative_sentiment`
- `climate_dice_simulation`
- `climate_sovereign_debt_risk`
- `dsge_ar1_demo`
- `dsge_fertility_demo`
- `dsge_nk_sketchpad`
- `empirical_benchmark_replications`
- `garch_midas_macro_volatility`
- `hank_sequence_space`
- `narrative_bursts_and_transcripts`
- `narrative_local_llm`
- `nowcasting_gdp_news`
- `penalized_macro_forecasting`
- `synthetic_did_california_prop99`
- `vintage_news_noise`
