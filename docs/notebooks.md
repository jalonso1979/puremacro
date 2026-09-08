> 🇬🇧 English · 🇪🇸 [Español](es/notebooks.md)

# Showcase Notebooks

`puremacro` ships interactive, publication-quality showcase notebooks covering the major heterogeneous-agent macro paradigms, empirical macroeconometrics, Bayesian estimation, and specialized climate/historical applications.

Every notebook is written in Jupytext percent format (`.py`), executes deterministically with fixed seeds, adheres to the Pyodide 4-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`), and is available with a paired Spanish edition (`_es.py`).

## Flagship Bayesian DSGE Showcase

### `42_dsge_bayesian_estimation_and_diagnostics`
- **Source**: `notebooks/42_dsge_bayesian_estimation_and_diagnostics.py`
- **Spanish Edition**: `notebooks/42_dsge_bayesian_estimation_and_diagnostics_es.py`
- **Benchmark Model**: Smets & Wouters (2007, AER 97(3):586–606) 7-observable, 7-shock US economy.
- **Core Capabilities Demonstrated**:
  1. **Model Specification & Data Ingestion**: Parsing Dynare `.mod` files (`sw07_pfeifer.mod`) with declared `varobs` and `estimated_params`, ingesting 156 quarters of real US macroeconomic data (`_sw07_data.csv`).
  2. **Multi-Algorithm Mode Search**: Comparing `"lbfgs"`, `"csminwel"` (Chris Sims' line-search with gradient-directed non-convex steps), and `"cmaes"` (Covariance Matrix Adaptation Evolution Strategy) to robustly locate posterior modes.
  3. **Visual Mode Diagnostics (`mode_check`)**: Curvature slice profiling across all parameter coordinates to ensure genuine local concavity and diagnose weak identification.
  4. **Bayesian MCMC Estimation**: Random-walk Metropolis-Hastings sampling with scale adaptation, split-$\hat{R}$ Gelman-Rubin convergence diagnostics, and prior-versus-posterior distribution plots.
  5. **Kalman Smoother & Structural Shock Decomposition**: Extracting smoothed unobserved states and structural innovations, validating the historical shock accounting decomposition across GDP growth, inflation, and policy rate.
  6. **Forecasting & Fan Charts**: Dynamic multi-period out-of-sample and conditional forecasting with 90% confidence fan charts.
  7. **Marginal Data Density & Model Comparison**: Computing marginal data densities via Laplace asymptotic approximation and Geweke (1999) modified harmonic mean across multiple truncation thresholds, coupled with formal Bayes factor model comparison.

---

## 2.6.0 Modernized Showcases

### `41_dynare_frontier_showcase`
- **Source**: `notebooks/41_dynare_frontier_showcase.py`
- **Spanish Edition**: `notebooks/41_dynare_frontier_showcase_es.py`
- **2.6.0 Upgrades**: Modernized to use puremacro's native `.smoother(data)` and `.estimate(data)` methods on `sw07_pfeifer.mod` with bundled quarterly US data, eliminating legacy toy likelihood wrappers. Demonstrates forecast error variance decompositions (FEVD), OccBin piecewise-linear zero lower bound (ZLB) regimes, and deterministic perfect-foresight Ramsey transitions.

### `macro_history_and_climate/N12_paleoclimate_eiv_and_simex`
- **Source**: `notebooks/macro_history_and_climate/N12_paleoclimate_eiv_and_simex.py`
- **2.6.0 Upgrades**: Completely purged external `statsmodels` dependencies in favor of `puremacro.regress.ols` with heteroskedasticity-consistent standard errors (`HC1`). Implements simulation extrapolation (SIMEX) and errors-in-variables (EIV) adjustments for paleoclimate proxy temperature reconstructions in pure NumPy.

---

## Complete Showcase Catalog

| Notebook | Topic & Methodology | Spanish Twin |
|---|---|---|
| `01_wealth_inequality` | Aiyagari & Huggett incomplete markets, permanent $\beta$-heterogeneity, Lorenz curves & Gini indices | `01_wealth_inequality_es` |
| `02_aggregate_shocks` | Krusell–Smith approximate aggregation, transition dynamics, representative-agent benchmark | `02_aggregate_shocks_es` |
| `03_life_cycle_and_demographics` | Finite-horizon life-cycle consumption-saving, cohort wealth by age, mortality hazard weighting | `03_life_cycle_and_demographics_es` |
| `04_firm_dynamics` | Hopenhayn industry equilibrium with endogenous entry/exit and selection | `04_firm_dynamics_es` |
| `05_portfolios_and_preferences` | Two-asset portfolio choice, Epstein–Zin recursive utility, EGM vs VFI | `05_portfolios_and_preferences_es` |
| `06_svar_identification` | Structural VAR identification: Cholesky recursive ordering vs sign restrictions | `06_svar_identification_es` |
| `07_local_projections` | Jordà Local Projections with Newey-West HAC inference & state-dependent IRFs | `07_local_projections_es` |
| `08_garch_volatility` | Pure-NumPy GARCH(1,1) MLE & Engle Dynamic Conditional Correlation (DCC) | `08_garch_volatility_es` |
| `09_growth_at_risk` | Adrian-Boyarchenko-Giannone Growth-at-Risk via quantile autoregression & skew-$t$ fitting | `09_growth_at_risk_es` |
| `10_staggered_did` | Modern Difference-in-Differences: Callaway-Sant'Anna & Sun-Abraham estimators | `10_staggered_did_es` |
| `11_narrative_uncertainty` | Pure-NumPy Economic Policy Uncertainty (EPU) text index construction | `11_narrative_uncertainty_es` |
| `12_validation_gallery` | Validation scorecard comparing puremacro implementations against independent references | `12_validation_gallery_es` |
| `13_build_your_own_index` | Multi-recipe uncertainty construction: Text $\to$ EPU, Panel $\to$ JLN, Financial $\to$ FCI | `13_build_your_own_index_es` |
| `14_tax_multiplier_three_ways` | US tax multiplier: Blanchard-Perotti SVAR, Romer-Romer narrative LP, Mertens-Ravn LP-IV | `14_tax_multiplier_three_ways_es` |
| `15_lp_did` | Local Projections Difference-in-Differences (LP-DiD) under staggered treatment | `15_lp_did_es` |
| `16_regime_girf` | Generalized Impulse Response Functions (GIRF) for threshold VAR / Markov-switching VAR | `16_regime_girf_es` |
| `17_identification_spec_curve` | Identification specification curves for sensitivity analysis | `17_identification_spec_curve_es` |
| `18_beveridge_curve` | Beveridge curve dynamics, matching efficiency, and vacancy-unemployment shifts | `18_beveridge_curve_es` |
| `19_model_confidence_set` | Hansen-Lunde-Nason Model Confidence Set (MCS) for competitive forecasting | `19_model_confidence_set_es` |
| `20_unit_roots_with_power` | Elliott-Rothenberg-Stock DF-GLS and Ng-Perron high-power unit root tests | `20_unit_roots_with_power_es` |
| `21_dynare_vfi_dsl` | Declarative VFI DSL specification with Howard policy iteration acceleration | `21_dynare_vfi_dsl_es` |
| `22_continuous_time_hjb` | Achdou et al. (2022) Continuous-Time Hamilton-Jacobi-Bellman finite difference scheme | `22_continuous_time_hjb_es` |
| `23_aiyagari_endogenous_labor` | General equilibrium incomplete markets with endogenous labor supply choice | `23_aiyagari_endogenous_labor_es` |
| `24_synthetic_control` | Abadie et al. Synthetic Control Method with in-space and in-time placebo tests | `24_synthetic_control_es` |
| `25_frequency_connectedness` | Baruník & Křehlík frequency-domain variance decomposition spillover networks | `25_frequency_connectedness_es` |
| `26_cycles_and_bandpass` | Baxter-King, Christiano-Fitzgerald, and Beveridge-Nelson cycle filtering | `26_cycles_and_bandpass_es` |
| `27_garch_midas_macro_risk` | Engle-Ghysels-Sohn GARCH-MIDAS mixed-frequency volatility modeling | `27_garch_midas_macro_risk_es` |
| `28_weak_iv_anderson_rubin` | Anderson-Rubin weak-instrument robust confidence sets for LP-IV | `28_weak_iv_anderson_rubin_es` |
| `29_synthetic_did` | Arkhangelsky et al. Synthetic Difference-in-Differences (SDID) | `29_synthetic_did_es` |
| `30_narrative_bursts_and_transcripts` | Kleinberg burst detection on central bank speech transcripts | `30_narrative_bursts_and_transcripts_es` |
| `31_sequence_space_hank` | Auclert et al. (2021) Sequence-Space Jacobian method for HANK models | `31_sequence_space_hank_es` |
| `32_climate_macro_dice` | Nordhaus DICE integrated assessment model & optimal carbon tax policy | `32_climate_macro_dice_es` |
| `33_gdp_nowcasting_news` | Giannone-Reichlin-Small Dynamic Factor Model GDP nowcasting & news decomposition | `33_gdp_nowcasting_news_es` |
| `34_penalized_macro_forecasting` | Elastic Net and Adaptive Lasso for high-dimensional macroeconomic forecasting | `34_penalized_macro_forecasting_es` |
| `35_empirical_benchmark_replications` | Replication scorecard across SVAR, LP, and DiD benchmark literatures | `35_empirical_benchmark_replications_es` |
| `36_climate_sovereign_debt_risk` | Physical and transition climate risk transmission to sovereign debt sustainability | `36_climate_sovereign_debt_risk_es` |
| `37_central_bank_narrative_sentiment` | Central bank communication sentiment scoring and high-frequency monetary LP | `37_central_bank_narrative_sentiment_es` |
| `38_real_time_vintages_and_revisions` | Real-time QNA vintages across 45+ countries & news vs. noise revision tests | `38_real_time_vintages_and_revisions_es` |
| `39_multilingual_narrative_harvesting` | Multi-source narrative harvesting (50+ connectors) and multilingual macro scoring | `39_multilingual_narrative_harvesting_es` |
| `40_quarterly_national_accounts` | Three approaches to GDP accounting: rebasing, identities, and growth contributions | `40_quarterly_national_accounts_es` |
| `41_dynare_frontier_showcase` | Smets-Wouters (2007) native 2.6.0 smoother & estimation, OccBin ZLB, Ramsey transitions | `41_dynare_frontier_showcase_es` |
| `42_dsge_bayesian_estimation_and_diagnostics` | Flagship Smets-Wouters Bayesian estimation, mode_check, MCMC, fan charts, MDD | `42_dsge_bayesian_estimation_and_diagnostics_es` |
| `00_whats_new_in_puremacro_2_0` | Overview of puremacro unified API, result classes, and publication exporters | `00_whats_new_in_puremacro_2_0_es` |

---

## Execution & Building Artifacts

Showcase notebooks can be executed and compiled to pre-rendered `.ipynb` artifacts using the built-in CLI:

```bash
# Execute and build all notebooks into .ipynb
python tools/build_notebooks.py

# Build a single notebook
python tools/build_notebooks.py 42_dsge_bayesian_estimation_and_diagnostics

# Execute without modifying files (fails if any notebook raises an exception)
python tools/build_notebooks.py --check
```
