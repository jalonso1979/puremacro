> 🇬🇧 English · 🇪🇸 [Español](es/notebooks.md)

# Showcase Notebooks

`puremacro` ships interactive, publication-quality showcase notebooks covering the major heterogeneous-agent macro paradigms, empirical macroeconometrics, Bayesian estimation, specialized climate/historical applications, and frontier applied macroeconomic policy suites.

Every notebook is written in Jupytext percent format (`.py`), executes deterministically with fixed seeds, adheres to the Pyodide 4-package contract (`numpy`, `scipy`, `pandas`, `matplotlib`), and is available with a paired Spanish edition (`_es.py`).

---

## The 7-Section Pedagogical Architecture

Following `notebooks/_TEMPLATE.md`, deepened and frontier showcase notebooks adhere to a consistent 7-cell structural flow:

1. **Motivating Question**: 1–2 sentences defining the economic problem.
2. **The Method in Math**: Governing structural and econometric equations in compact, rigorous LaTeX ($...$ / $$...$$).
3. **Intuition**: An explicit `**Intuition.**` section translating algebraic equations into intuitive economic mechanisms and identification logic.
4. **Worked Code**: Self-contained, pure-NumPy runnable code blocks with explanatory comments on *why* choices are made.
5. **Read the Output**: Dedicated markdown analysis directly interpreting headline numerical outputs, parameter estimates, and generated figures.
6. **Your Turn**: An interactive exploratory exercise with `# ← change this` knobs, runnable defaults with assertions, and graded challenge prompts.
7. **How Comprehensive Is This?**: Contextual cross-references connecting the showcase to related `puremacro` entry points and literature.

---

## Applied Macroeconomic Policy Showcases (puremacro 3.2)

Showcases `47` through `50` bridge theoretical macroeconometrics with applied central banking, treasury, and financial market policy practice.

### `47_applied_central_bank_policy_suite`
- **Source**: `notebooks/47_applied_central_bank_policy_suite.py` (Spanish: `_es.py`)
- **Key Capabilities**:
  - Real-time monetary policy stance evaluation against Taylor rules (1993, 1999) and inertia specifications.
  - Counterfactual policy simulation: evaluating macro trajectories under alternative policy paths.
  - Multi-period fan chart projection bands accounting for shock uncertainty and parameter posterior dispersion.
  - Textual sentiment and tone extraction from central bank statements (FOMC, ECB) via dictionary-based sentiment scoring.

### `48_applied_realtime_nowcasting_and_news`
- **Source**: `notebooks/48_applied_realtime_nowcasting_and_news.py` (Spanish: `_es.py`)
- **Key Capabilities**:
  - Dynamic Factor Model (DFM) nowcasting of quarterly GDP following Giannone, Reichlin & Small (2008).
  - Ragged-edge asynchronous data vintage ingestion handling mixed-frequency monthly and quarterly releases.
  - Release-day **news decomposition**: decomposing forecast revisions into surprise news components by data category (labor, output, surveys).
  - Mankiw-Shapiro (1986) news-versus-noise orthogonality tests on real-time data revisions.

### `49_applied_macroprudential_gar_and_stress`
- **Source**: `notebooks/49_applied_macroprudential_gar_and_stress.py` (Spanish: `_es.py`)
- **Key Capabilities**:
  - Growth-at-Risk (GaR) conditional quantile regression following Adrian, Boyarchenko & Giannone (2019).
  - Fitting parametric Azzalini skew-$t$ density distributions over predictive horizons to capture asymmetric downside tail risk.
  - Systemic risk connectedness networks via generalized forecast error variance decomposition (GFEVD) (Diebold & Yílmaz 2014).
  - Macroprudential capital buffer stress scenarios and probability-of-recession fan charts.

### `50_applied_fiscal_multipliers_and_debt_sustainability`
- **Source**: `notebooks/50_applied_fiscal_multipliers_and_debt_sustainability.py` (Spanish: `_es.py`)
- **Key Capabilities**:
  - Tri-method fiscal multiplier estimation on frozen datasets: Blanchard-Perotti SVAR, Romer-Romer narrative Local Projections, and Mertens-Ravn narrative LP-IV with effective first-stage $F$-statistics.
  - Stochastic sovereign Debt Sustainability Analysis (DSA) fan charts modeling joint growth, inflation, and interest-rate $(r - g)$ shocks.
  - Stress testing sovereign debt ratios under physical and transition climate risk scenarios.

---

## Bayesian DSGE & HANK Frontier Showcases (puremacro 3.0 & 3.1)

### `42_dsge_bayesian_estimation_and_diagnostics` (Flagship)
- **Source**: `notebooks/42_dsge_bayesian_estimation_and_diagnostics.py` (Spanish: `_es.py`)
- **Model**: Smets & Wouters (2007, AER) 7-observable, 7-shock US economy.
- **Key Capabilities**: Multi-algorithm mode search (`lbfgs`, `csminwel`, `cmaes`), `mode_check` curvature slices, MCMC sampling with Gelman-Rubin $\hat{R}$ diagnostics, Kalman smoother historical shock decomposition, fan chart dynamic forecasting, and Laplace / Geweke marginal data density comparison.

### `43_dsge_nuts_and_analytic_gradients`
- **Source**: `notebooks/43_dsge_nuts_and_analytic_gradients.py` (Spanish: `_es.py`)
- **Key Capabilities**: Exact analytic Kalman score recursion ($\nabla_\theta \ln L$) via generalized Sylvester solvers, No-U-Turn Sampler (NUTS) with dual averaging step-size adaptation, and energy BFMI diagnostics.

### `44_hank_sequence_space_bridge`
- **Source**: `notebooks/44_hank_sequence_space_bridge.py` (Spanish: `_es.py`)
- **Key Capabilities**: Heterogeneous-Agent Sequence-Space bridge parsed from Dynare `.mod` files (`hetagent_block`), stationary wealth distribution $\mathcal{D}^*(a)$, Fake-News Jacobians ($J_{C,r}, J_{C,Y}$), and nonlinear Broyden MIT transitions.

### `45_dsge_discretion_dsge_var_and_news_shocks`
- **Source**: `notebooks/45_dsge_discretion_dsge_var_and_news_shocks.py` (Spanish: `_es.py`)
- **Key Capabilities**: Optimal discretionary policy vs commitment (inflation/stabilization bias decomposition), Del Negro & Schorfheide (2004) DSGE-VAR($\lambda$) prior optimization, and anticipated news shock companion state-space augmentation.

### `46_dsge_particle_filtering_and_markov_switching`
- **Source**: `notebooks/46_dsge_particle_filtering_and_markov_switching.py` (Spanish: `_es.py`)
- **Key Capabilities**: Differentiable OccBin for NUTS, Foerster et al. (2016) Markov-Switching DSGE with closed-form analytical GIRFs, and vectorized sequential Monte Carlo particle filtering with stochastic volatility.

---

## Complete Showcase Catalog

| Notebook | Topic & Methodology | Spanish Twin |
|---|---|---|
| `00_whats_new_in_puremacro_3_0` | Milestone 3.0: Analytic Kalman gradients, NUTS HMC, and HANK Sequence-Space bridge | `00_whats_new_in_puremacro_3_0_es` |
| `00_whats_new_in_puremacro_2_0` | Milestone 2.0: Unified API (`lags`, `horizon`, `ci`), result objects, and exporters | `00_whats_new_in_puremacro_2_0_es` |
| `01_wealth_inequality` | Aiyagari & Huggett incomplete markets, permanent $\beta$-heterogeneity, Lorenz & Gini | `01_wealth_inequality_es` |
| `02_aggregate_shocks` | Krusell–Smith approximate aggregation, transition dynamics, representative-agent benchmark | `02_aggregate_shocks_es` |
| `03_life_cycle_and_demographics` | Finite-horizon life-cycle consumption-saving, cohort wealth by age, mortality weighting | `03_life_cycle_and_demographics_es` |
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
| `41_dynare_frontier_showcase` | Smets-Wouters (2007) native smoother & estimation, OccBin ZLB, Ramsey transitions | `41_dynare_frontier_showcase_es` |
| `42_dsge_bayesian_estimation_and_diagnostics` | Flagship Smets-Wouters Bayesian estimation, mode_check, MCMC, fan charts, MDD | `42_dsge_bayesian_estimation_and_diagnostics_es` |
| `43_dsge_nuts_and_analytic_gradients` | Bayesian DSGE via NUTS with exact analytic Kalman score recursion | `43_dsge_nuts_and_analytic_gradients_es` |
| `44_hank_sequence_space_bridge` | HANK Sequence-Space bridge from `.mod` files, Fake-News Jacobians, MIT transitions | `44_hank_sequence_space_bridge_es` |
| `45_dsge_discretion_dsge_var_and_news_shocks` | Discretionary policy vs commitment, DSGE-VAR prior optimization, news shocks | `45_dsge_discretion_dsge_var_and_news_shocks_es` |
| `46_dsge_particle_filtering_and_markov_switching` | Differentiable OccBin for NUTS, Markov-switching DSGE, particle filtering | `46_dsge_particle_filtering_and_markov_switching_es` |
| `47_applied_central_bank_policy_suite` | Monetary policy stance, counterfactual Taylor rules, fan charts, sentiment | `47_applied_central_bank_policy_suite_es` |
| `48_applied_realtime_nowcasting_and_news` | DFM nowcasting with ragged-edge vintages, release-day news decomposition | `48_applied_realtime_nowcasting_and_news_es` |
| `49_applied_macroprudential_gar_and_stress` | Growth-at-Risk quantile densities (skew-$t$), systemic connectedness networks | `49_applied_macroprudential_gar_and_stress_es` |
| `50_applied_fiscal_multipliers_and_debt_sustainability` | Multi-method fiscal multipliers, stochastic sovereign DSA under climate stress | `50_applied_fiscal_multipliers_and_debt_sustainability_es` |

---

## Execution & Building Artifacts

Showcase notebooks can be executed and compiled to pre-rendered `.ipynb` artifacts using the built-in CLI:

```bash
# Execute and build all notebooks into .ipynb
python tools/build_notebooks.py

# Build a single notebook
python tools/build_notebooks.py 47_applied_central_bank_policy_suite

# Execute without modifying files (fails if any notebook raises an exception)
python tools/build_notebooks.py --check
```
