# puremacro showcase notebooks

Start with the [English learning guide](../docs/notebooks.md) or the
[guía en español](../docs/es/notebooks.md) for suggested sequences, setup and
interpretation. Choose 06 for structural shocks, 10 for policy evaluation,
11 for text indices, or 34 for a first forecasting exercise.

Edit the Jupytext percent **`.py` sources**, then regenerate the `.ipynb` files
with executed outputs. Keep English and Spanish computations aligned. The
[template](_TEMPLATE.md) covers equations, intuition, data provenance, diagnostics
and exercises.

## Catalog

The titles open executed English notebooks. Every row also links the Spanish
notebook and both editable sources. Release tours `00` describe older versions.

| Notebook | Shows | Sources and Spanish edition |
|---|---|---|
| [00_whats_new_in_puremacro_3_0](00_whats_new_in_puremacro_3_0.ipynb) | Milestone 3.0: Analytic Kalman gradients, NUTS HMC, and HANK Sequence-Space bridge | [EN .py](00_whats_new_in_puremacro_3_0.py) · [ES .py](00_whats_new_in_puremacro_3_0_es.py) · [ES .ipynb](00_whats_new_in_puremacro_3_0_es.ipynb) |
| [00_whats_new_in_puremacro_2_0](00_whats_new_in_puremacro_2_0.ipynb) | Milestone 2.0: Unified API (`lags`, `horizon`, `ci`), result objects, and exporters | [EN .py](00_whats_new_in_puremacro_2_0.py) · [ES .py](00_whats_new_in_puremacro_2_0_es.py) · [ES .ipynb](00_whats_new_in_puremacro_2_0_es.ipynb) |
| [01_wealth_inequality](01_wealth_inequality.ipynb) | Aiyagari & Huggett incomplete markets, permanent $\beta$-heterogeneity, Lorenz & Gini | [EN .py](01_wealth_inequality.py) · [ES .py](01_wealth_inequality_es.py) · [ES .ipynb](01_wealth_inequality_es.ipynb) |
| [02_aggregate_shocks](02_aggregate_shocks.ipynb) | Krusell–Smith approximate aggregation, transition dynamics, representative-agent benchmark | [EN .py](02_aggregate_shocks.py) · [ES .py](02_aggregate_shocks_es.py) · [ES .ipynb](02_aggregate_shocks_es.ipynb) |
| [03_life_cycle_and_demographics](03_life_cycle_and_demographics.ipynb) | Finite-horizon life-cycle consumption-saving, cohort wealth by age, mortality weighting | [EN .py](03_life_cycle_and_demographics.py) · [ES .py](03_life_cycle_and_demographics_es.py) · [ES .ipynb](03_life_cycle_and_demographics_es.ipynb) |
| [04_firm_dynamics](04_firm_dynamics.ipynb) | Hopenhayn industry equilibrium with endogenous entry/exit and selection | [EN .py](04_firm_dynamics.py) · [ES .py](04_firm_dynamics_es.py) · [ES .ipynb](04_firm_dynamics_es.ipynb) |
| [05_portfolios_and_preferences](05_portfolios_and_preferences.ipynb) | Two-asset portfolio choice, Epstein–Zin recursive utility, EGM vs VFI | [EN .py](05_portfolios_and_preferences.py) · [ES .py](05_portfolios_and_preferences_es.py) · [ES .ipynb](05_portfolios_and_preferences_es.ipynb) |
| [06_svar_identification](06_svar_identification.ipynb) | Structural VAR identification: Cholesky recursive ordering vs sign restrictions | [EN .py](06_svar_identification.py) · [ES .py](06_svar_identification_es.py) · [ES .ipynb](06_svar_identification_es.ipynb) |
| [07_local_projections](07_local_projections.ipynb) | Jordà Local Projections with Newey-West HAC inference & state-dependent IRFs | [EN .py](07_local_projections.py) · [ES .py](07_local_projections_es.py) · [ES .ipynb](07_local_projections_es.ipynb) |
| [08_garch_volatility](08_garch_volatility.ipynb) | Pure-NumPy GARCH(1,1) MLE & Engle Dynamic Conditional Correlation (DCC) | [EN .py](08_garch_volatility.py) · [ES .py](08_garch_volatility_es.py) · [ES .ipynb](08_garch_volatility_es.ipynb) |
| [09_growth_at_risk](09_growth_at_risk.ipynb) | Adrian-Boyarchenko-Giannone Growth-at-Risk via quantile autoregression & skew-$t$ fitting | [EN .py](09_growth_at_risk.py) · [ES .py](09_growth_at_risk_es.py) · [ES .ipynb](09_growth_at_risk_es.ipynb) |
| [10_staggered_did](10_staggered_did.ipynb) | Staggered DiD with pointwise event-study bands and a unit-bootstrap interval for overall ATT | [EN .py](10_staggered_did.py) · [ES .py](10_staggered_did_es.py) · [ES .ipynb](10_staggered_did_es.ipynb) |
| [11_narrative_uncertainty](11_narrative_uncertainty.ipynb) | Pure-NumPy Economic Policy Uncertainty (EPU) text index construction | [EN .py](11_narrative_uncertainty.py) · [ES .py](11_narrative_uncertainty_es.py) · [ES .ipynb](11_narrative_uncertainty_es.ipynb) |
| [12_validation_gallery](12_validation_gallery.ipynb) | Validation scorecard comparing puremacro implementations against independent references | [EN .py](12_validation_gallery.py) · [ES .py](12_validation_gallery_es.py) · [ES .ipynb](12_validation_gallery_es.ipynb) |
| [13_build_your_own_index](13_build_your_own_index.ipynb) | Four index recipes: text EPU, simplified JLN-style uncertainty, FCI and comovement | [EN .py](13_build_your_own_index.py) · [ES .py](13_build_your_own_index_es.py) · [ES .ipynb](13_build_your_own_index_es.ipynb) |
| [14_tax_multiplier_three_ways](14_tax_multiplier_three_ways.ipynb) | US tax multiplier: Blanchard-Perotti SVAR, Romer-Romer narrative LP, Mertens-Ravn proxy-SVAR | [EN .py](14_tax_multiplier_three_ways.py) · [ES .py](14_tax_multiplier_three_ways_es.py) · [ES .ipynb](14_tax_multiplier_three_ways_es.ipynb) |
| [15_lp_did](15_lp_did.ipynb) | Local Projections Difference-in-Differences (LP-DiD) under staggered treatment | [EN .py](15_lp_did.py) · [ES .py](15_lp_did_es.py) · [ES .ipynb](15_lp_did_es.ipynb) |
| [16_regime_girf](16_regime_girf.ipynb) | Generalized Impulse Response Functions (GIRF) for threshold VAR / Markov-switching VAR | [EN .py](16_regime_girf.py) · [ES .py](16_regime_girf_es.py) · [ES .ipynb](16_regime_girf_es.ipynb) |
| [17_identification_spec_curve](17_identification_spec_curve.ipynb) | Identification specification curves for sensitivity analysis | [EN .py](17_identification_spec_curve.py) · [ES .py](17_identification_spec_curve_es.py) · [ES .ipynb](17_identification_spec_curve_es.ipynb) |
| [18_beveridge_curve](18_beveridge_curve.ipynb) | Beveridge curve dynamics, matching efficiency, and vacancy-unemployment shifts | [EN .py](18_beveridge_curve.py) · [ES .py](18_beveridge_curve_es.py) · [ES .ipynb](18_beveridge_curve_es.ipynb) |
| [19_model_confidence_set](19_model_confidence_set.ipynb) | Hansen-Lunde-Nason Model Confidence Set (MCS) for competitive forecasting | [EN .py](19_model_confidence_set.py) · [ES .py](19_model_confidence_set_es.py) · [ES .ipynb](19_model_confidence_set_es.ipynb) |
| [20_unit_roots_with_power](20_unit_roots_with_power.ipynb) | Elliott-Rothenberg-Stock DF-GLS and Ng-Perron high-power unit root tests | [EN .py](20_unit_roots_with_power.py) · [ES .py](20_unit_roots_with_power_es.py) · [ES .ipynb](20_unit_roots_with_power_es.ipynb) |
| [21_dynare_vfi_dsl](21_dynare_vfi_dsl.ipynb) | Declarative VFI DSL specification with Howard policy iteration acceleration | [EN .py](21_dynare_vfi_dsl.py) · [ES .py](21_dynare_vfi_dsl_es.py) · [ES .ipynb](21_dynare_vfi_dsl_es.ipynb) |
| [22_continuous_time_hjb](22_continuous_time_hjb.ipynb) | Achdou et al. (2022) Continuous-Time Hamilton-Jacobi-Bellman finite difference scheme | [EN .py](22_continuous_time_hjb.py) · [ES .py](22_continuous_time_hjb_es.py) · [ES .ipynb](22_continuous_time_hjb_es.ipynb) |
| [23_aiyagari_endogenous_labor](23_aiyagari_endogenous_labor.ipynb) | General equilibrium incomplete markets with endogenous labor supply choice | [EN .py](23_aiyagari_endogenous_labor.py) · [ES .py](23_aiyagari_endogenous_labor_es.py) · [ES .ipynb](23_aiyagari_endogenous_labor_es.ipynb) |
| [24_synthetic_control](24_synthetic_control.ipynb) | Abadie et al. Synthetic Control Method with in-space and in-time placebo tests | [EN .py](24_synthetic_control.py) · [ES .py](24_synthetic_control_es.py) · [ES .ipynb](24_synthetic_control_es.ipynb) |
| [25_frequency_connectedness](25_frequency_connectedness.ipynb) | Baruník & Křehlík frequency-domain variance decomposition spillover networks | [EN .py](25_frequency_connectedness.py) · [ES .py](25_frequency_connectedness_es.py) · [ES .ipynb](25_frequency_connectedness_es.ipynb) |
| [26_cycles_and_bandpass](26_cycles_and_bandpass.ipynb) | Baxter-King, Christiano-Fitzgerald, and Beveridge-Nelson cycle filtering | [EN .py](26_cycles_and_bandpass.py) · [ES .py](26_cycles_and_bandpass_es.py) · [ES .ipynb](26_cycles_and_bandpass_es.ipynb) |
| [27_garch_midas_macro_risk](27_garch_midas_macro_risk.ipynb) | Engle-Ghysels-Sohn GARCH-MIDAS mixed-frequency volatility modeling | [EN .py](27_garch_midas_macro_risk.py) · [ES .py](27_garch_midas_macro_risk_es.py) · [ES .ipynb](27_garch_midas_macro_risk_es.ipynb) |
| [28_weak_iv_anderson_rubin](28_weak_iv_anderson_rubin.ipynb) | Anderson-Rubin weak-instrument robust confidence sets for LP-IV | [EN .py](28_weak_iv_anderson_rubin.py) · [ES .py](28_weak_iv_anderson_rubin_es.py) · [ES .ipynb](28_weak_iv_anderson_rubin_es.ipynb) |
| [29_synthetic_did](29_synthetic_did.ipynb) | Arkhangelsky et al. Synthetic Difference-in-Differences (SDID) | [EN .py](29_synthetic_did.py) · [ES .py](29_synthetic_did_es.py) · [ES .ipynb](29_synthetic_did_es.ipynb) |
| [30_narrative_bursts_and_transcripts](30_narrative_bursts_and_transcripts.ipynb) | Kleinberg burst detection on central bank speech transcripts | [EN .py](30_narrative_bursts_and_transcripts.py) · [ES .py](30_narrative_bursts_and_transcripts_es.py) · [ES .ipynb](30_narrative_bursts_and_transcripts_es.ipynb) |
| [31_sequence_space_hank](31_sequence_space_hank.ipynb) | Auclert et al. (2021) Sequence-Space Jacobian method for HANK models | [EN .py](31_sequence_space_hank.py) · [ES .py](31_sequence_space_hank_es.py) · [ES .ipynb](31_sequence_space_hank_es.ipynb) |
| [32_climate_macro_dice](32_climate_macro_dice.ipynb) | Nordhaus DICE integrated assessment model & optimal carbon tax policy | [EN .py](32_climate_macro_dice.py) · [ES .py](32_climate_macro_dice_es.py) · [ES .ipynb](32_climate_macro_dice_es.ipynb) |
| [33_gdp_nowcasting_news](33_gdp_nowcasting_news.ipynb) | Single-vintage PCA/bridge demonstration; use 48 for a held-out target | [EN .py](33_gdp_nowcasting_news.py) · [ES .py](33_gdp_nowcasting_news_es.py) · [ES .ipynb](33_gdp_nowcasting_news_es.ipynb) |
| [34_penalized_macro_forecasting](34_penalized_macro_forecasting.ipynb) | Elastic Net and Adaptive Lasso with chronological holdout forecasts and a benchmark | [EN .py](34_penalized_macro_forecasting.py) · [ES .py](34_penalized_macro_forecasting_es.py) · [ES .ipynb](34_penalized_macro_forecasting_es.ipynb) |
| [35_empirical_benchmark_replications](35_empirical_benchmark_replications.ipynb) | Replication scorecard across SVAR, LP, and DiD benchmark literatures | [EN .py](35_empirical_benchmark_replications.py) · [ES .py](35_empirical_benchmark_replications_es.py) · [ES .ipynb](35_empirical_benchmark_replications_es.ipynb) |
| [36_climate_sovereign_debt_risk](36_climate_sovereign_debt_risk.ipynb) | Physical and transition climate risk transmission to sovereign debt sustainability | [EN .py](36_climate_sovereign_debt_risk.py) · [ES .py](36_climate_sovereign_debt_risk_es.py) · [ES .ipynb](36_climate_sovereign_debt_risk_es.ipynb) |
| [37_central_bank_narrative_sentiment](37_central_bank_narrative_sentiment.ipynb) | Central bank communication sentiment scoring and high-frequency monetary LP | [EN .py](37_central_bank_narrative_sentiment.py) · [ES .py](37_central_bank_narrative_sentiment_es.py) · [ES .ipynb](37_central_bank_narrative_sentiment_es.ipynb) |
| [38_real_time_vintages_and_revisions](38_real_time_vintages_and_revisions.ipynb) | Vintage and revision diagnostics; distinguish simulated examples from provider coverage | [EN .py](38_real_time_vintages_and_revisions.py) · [ES .py](38_real_time_vintages_and_revisions_es.py) · [ES .ipynb](38_real_time_vintages_and_revisions_es.ipynb) |
| [39_multilingual_narrative_harvesting](39_multilingual_narrative_harvesting.ipynb) | Multi-source narrative harvesting (50+ connectors) and multilingual macro scoring | [EN .py](39_multilingual_narrative_harvesting.py) · [ES .py](39_multilingual_narrative_harvesting_es.py) · [ES .ipynb](39_multilingual_narrative_harvesting_es.ipynb) |
| [40_quarterly_national_accounts](40_quarterly_national_accounts.ipynb) | Three approaches to GDP accounting: rebasing, identities, and growth contributions | [EN .py](40_quarterly_national_accounts.py) · [ES .py](40_quarterly_national_accounts_es.py) · [ES .ipynb](40_quarterly_national_accounts_es.ipynb) |
| [41_dynare_frontier_showcase](41_dynare_frontier_showcase.ipynb) | Smets-Wouters (2007) native smoother & estimation, OccBin ZLB, Ramsey transitions | [EN .py](41_dynare_frontier_showcase.py) · [ES .py](41_dynare_frontier_showcase_es.py) · [ES .ipynb](41_dynare_frontier_showcase_es.ipynb) |
| [42_dsge_bayesian_estimation_and_diagnostics](42_dsge_bayesian_estimation_and_diagnostics.ipynb) | Flagship Smets-Wouters Bayesian estimation, mode_check, MCMC, fan charts, MDD | [EN .py](42_dsge_bayesian_estimation_and_diagnostics.py) · [ES .py](42_dsge_bayesian_estimation_and_diagnostics_es.py) · [ES .ipynb](42_dsge_bayesian_estimation_and_diagnostics_es.ipynb) |
| [43_dsge_nuts_and_analytic_gradients](43_dsge_nuts_and_analytic_gradients.ipynb) | Bayesian DSGE via NUTS with exact analytic Kalman score recursion | [EN .py](43_dsge_nuts_and_analytic_gradients.py) · [ES .py](43_dsge_nuts_and_analytic_gradients_es.py) · [ES .ipynb](43_dsge_nuts_and_analytic_gradients_es.ipynb) |
| [44_hank_sequence_space_bridge](44_hank_sequence_space_bridge.ipynb) | HANK Sequence-Space bridge from `.mod` files, Fake-News Jacobians, MIT transitions | [EN .py](44_hank_sequence_space_bridge.py) · [ES .py](44_hank_sequence_space_bridge_es.py) · [ES .ipynb](44_hank_sequence_space_bridge_es.ipynb) |
| [45_dsge_discretion_dsge_var_and_news_shocks](45_dsge_discretion_dsge_var_and_news_shocks.ipynb) | Discretionary policy vs commitment, DSGE-VAR prior optimization, news shocks | [EN .py](45_dsge_discretion_dsge_var_and_news_shocks.py) · [ES .py](45_dsge_discretion_dsge_var_and_news_shocks_es.py) · [ES .ipynb](45_dsge_discretion_dsge_var_and_news_shocks_es.ipynb) |
| [46_dsge_particle_filtering_and_markov_switching](46_dsge_particle_filtering_and_markov_switching.ipynb) | Differentiable OccBin for NUTS, Markov-switching DSGE, particle filtering | [EN .py](46_dsge_particle_filtering_and_markov_switching.py) · [ES .py](46_dsge_particle_filtering_and_markov_switching_es.py) · [ES .ipynb](46_dsge_particle_filtering_and_markov_switching_es.ipynb) |
| [47_applied_central_bank_policy_suite](47_applied_central_bank_policy_suite.ipynb) | Monetary policy stance, counterfactual Taylor rules, fan charts, sentiment | [EN .py](47_applied_central_bank_policy_suite.py) · [ES .py](47_applied_central_bank_policy_suite_es.py) · [ES .ipynb](47_applied_central_bank_policy_suite_es.ipynb) |
| [48_applied_realtime_nowcasting_and_news](48_applied_realtime_nowcasting_and_news.ipynb) | Synthetic ragged-edge nowcast with held-out GDP, projection diagnostics and revision tests | [EN .py](48_applied_realtime_nowcasting_and_news.py) · [ES .py](48_applied_realtime_nowcasting_and_news_es.py) · [ES .ipynb](48_applied_realtime_nowcasting_and_news_es.ipynb) |
| [49_applied_macroprudential_gar_and_stress](49_applied_macroprudential_gar_and_stress.ipynb) | Growth-at-Risk quantile densities (skew-$t$), systemic connectedness networks | [EN .py](49_applied_macroprudential_gar_and_stress.py) · [ES .py](49_applied_macroprudential_gar_and_stress_es.py) · [ES .ipynb](49_applied_macroprudential_gar_and_stress_es.ipynb) |
| [50_applied_fiscal_multipliers_and_debt_sustainability](50_applied_fiscal_multipliers_and_debt_sustainability.ipynb) | Frozen US fiscal data, narrative tax instruments and stochastic debt scenarios with explicit units | [EN .py](50_applied_fiscal_multipliers_and_debt_sustainability.py) · [ES .py](50_applied_fiscal_multipliers_and_debt_sustainability_es.py) · [ES .ipynb](50_applied_fiscal_multipliers_and_debt_sustainability_es.ipynb) |
| [51_continuous_projection_collocation_and_fem](51_continuous_projection_collocation_and_fem.ipynb) | Continuous state-space DP: Chebyshev orthogonal collocation vs FEM Galerkin with borrowing kinks | [EN .py](51_continuous_projection_collocation_and_fem.py) · [ES .py](51_continuous_projection_collocation_and_fem_es.py) · [ES .ipynb](51_continuous_projection_collocation_and_fem_es.ipynb) |
| [52_continuous_transition_mit_shocks](52_continuous_transition_mit_shocks.ipynb) | Non-linear continuous transition dynamics & MIT shocks via coupled EGM-Young operator | [EN .py](52_continuous_transition_mit_shocks.py) · [ES .py](52_continuous_transition_mit_shocks_es.py) · [ES .ipynb](52_continuous_transition_mit_shocks_es.ipynb) |
| [53_exact_analytic_ift_gradients](53_exact_analytic_ift_gradients.ipynb) | IFT parameter sensitivities, adjoints and comparison with finite differences | [EN .py](53_exact_analytic_ift_gradients.py) · [ES .py](53_exact_analytic_ift_gradients_es.py) · [ES .ipynb](53_exact_analytic_ift_gradients_es.ipynb) |
| [54_deep_macro_pinns_high_dim](54_deep_macro_pinns_high_dim.ipynb) | High-dimensional dynamic models (10+ states) via pure-NumPy Physics-Informed Neural Networks | [EN .py](54_deep_macro_pinns_high_dim.py) · [ES .py](54_deep_macro_pinns_high_dim_es.py) · [ES .ipynb](54_deep_macro_pinns_high_dim_es.ipynb) |
| [55_quantitative_spatial_and_trade_ge](55_quantitative_spatial_and_trade_ge.ipynb) | Quantitative spatial & trade general equilibrium: Caliendo-Parro tariffs & Allen-Arkolakis geography | [EN .py](55_quantitative_spatial_and_trade_ge.py) · [ES .py](55_quantitative_spatial_and_trade_ge_es.py) · [ES .ipynb](55_quantitative_spatial_and_trade_ge_es.ipynb) |
| [56_implicit_hjb_and_continuous_kfe](56_implicit_hjb_and_continuous_kfe.ipynb) | Continuous-time HJB implicit upwind solver, adjoint KFE stationary wealth density & Aiyagari GE | [EN .py](56_implicit_hjb_and_continuous_kfe.py) · [ES .py](56_implicit_hjb_and_continuous_kfe_es.py) · [ES .ipynb](56_implicit_hjb_and_continuous_kfe_es.ipynb) |
| [57_multiconstraint_occbin_and_dml](57_multiconstraint_occbin_and_dml.ipynb) | Multi-constraint OccBin ($M \ge 2$: ZLB + credit limits) & Double Machine Learning (DML-PLR) | [EN .py](57_multiconstraint_occbin_and_dml.py) · [ES .py](57_multiconstraint_occbin_and_dml_es.py) · [ES .ipynb](57_multiconstraint_occbin_and_dml_es.ipynb) |
| [58_latin_america_realtime_macro](58_latin_america_realtime_macro.ipynb) | Simulated Latin American vintages, portable .pmz cartridges and news/noise diagnostics | [EN .py](58_latin_america_realtime_macro.py) · [ES .py](58_latin_america_realtime_macro_es.py) · [ES .ipynb](58_latin_america_realtime_macro_es.ipynb) |
| [59_latam_realtime_nowcast_and_news](59_latam_realtime_nowcast_and_news.ipynb) | Synthetic Latin America nowcasting, vintage news attribution and a separate PIT experiment | [EN .py](59_latam_realtime_nowcast_and_news.py) · [ES .py](59_latam_realtime_nowcast_and_news_es.py) · [ES .ipynb](59_latam_realtime_nowcast_and_news_es.ipynb) |
| [60_interactive_dml_irm_and_iv](60_interactive_dml_irm_and_iv.ipynb) | Interactive Double ML: IRM (ATE/ATT) with regularized logistic CD, propensity overlap & DML-IV | [EN .py](60_interactive_dml_irm_and_iv.py) · [ES .py](60_interactive_dml_irm_and_iv_es.py) · [ES .ipynb](60_interactive_dml_irm_and_iv_es.ipynb) |
| [61_quantitative_policy_simulators](61_quantitative_policy_simulators.ipynb) | Stylized trade preset and sequence-space HANK monetary experiments | [EN .py](61_quantitative_policy_simulators.py) · [ES .py](61_quantitative_policy_simulators_es.py) · [ES .ipynb](61_quantitative_policy_simulators_es.ipynb) |
| [62_flexible_trade_cge](62_flexible_trade_cge.ipynb) | ICIO inspection, flexible cost/demand/markup building blocks and synthetic benchmark GE | [EN .py](62_flexible_trade_cge.py) · [ES .py](62_flexible_trade_cge_es.py) · [ES .ipynb](62_flexible_trade_cge_es.ipynb) |
| [63_trade_wars_and_nash_tariffs](63_trade_wars_and_nash_tariffs.ipynb) | Hicksian tariff games: fixed baseline and actions, best-response regret and audited equilibrium recovery | [EN .py](63_trade_wars_and_nash_tariffs.py) · [ES .py](63_trade_wars_and_nash_tariffs_es.py) · [ES .ipynb](63_trade_wars_and_nash_tariffs_es.ipynb) |
| [64_singularities_and_keller_pac](64_singularities_and_keller_pac.ipynb) | Spectral bounds, scalar toy fold, small synthetic CGE continuation and a planted stiff matrix | [EN .py](64_singularities_and_keller_pac.py) · [ES .py](64_singularities_and_keller_pac_es.py) · [ES .ipynb](64_singularities_and_keller_pac_es.ipynb) |
| [65_gvc_cascades_and_welfare_decomposition](65_gvc_cascades_and_welfare_decomposition.ipynb) | Data provenance, cost propagation and Hicksian EV/CV with price, factor-income and fiscal-transfer attribution | [EN .py](65_gvc_cascades_and_welfare_decomposition.py) · [ES .py](65_gvc_cascades_and_welfare_decomposition_es.py) · [ES .ipynb](65_gvc_cascades_and_welfare_decomposition_es.ipynb) |
| [66_optimal_policy_cgg1999_replication](66_optimal_policy_cgg1999_replication.ipynb) | Replication of Clarida–Galí–Gertler (1999): discretion, simple-rule and timeless commitment against closed forms, and the loss criterion | [EN .py](66_optimal_policy_cgg1999_replication.py) · [ES .py](66_optimal_policy_cgg1999_replication_es.py) · [ES .ipynb](66_optimal_policy_cgg1999_replication_es.ipynb) |
| [67_bewley_classics_replication](67_bewley_classics_replication.ipynb) | Replication of Huggett (1993) and Aiyagari (1994): Table I exactly, the equilibrium tables' orderings, their decimals against numerical error, and the Markov chain | [EN .py](67_bewley_classics_replication.py) · [ES .py](67_bewley_classics_replication_es.py) · [ES .ipynb](67_bewley_classics_replication_es.ipynb) |
| [68_third_order_perturbation_trust](68_third_order_perturbation_trust.ipynb) | Third-order perturbation checked against closed forms and live Dynare 7.0, a tensor error no path can see, and what the checked third order says | [EN .py](68_third_order_perturbation_trust.py) · [ES .py](68_third_order_perturbation_trust_es.py) · [ES .ipynb](68_third_order_perturbation_trust_es.ipynb) |
| [local_llm_uncertainty](local_llm_uncertainty.ipynb) | Local inference with optional desktop engines; offline mock fallback | [EN .py](local_llm_uncertainty.py) · [ES .py](local_llm_uncertainty_es.py) · [ES .ipynb](local_llm_uncertainty_es.ipynb) |

## Rebuild and verify

Run from the repository root:

```bash
python -m pip install -e ".[notebooks]"
python tools/build_notebooks.py 34_penalized_macro_forecasting 34_penalized_macro_forecasting_es
python tools/build_notebooks.py --check 34_penalized_macro_forecasting 34_penalized_macro_forecasting_es
python tools/build_notebooks.py --validate-only
```

For an interactive interface, install `jupyterlab` in the same environment and
run `python -m jupyterlab notebooks`. Keep `_nbstyle.py` alongside the notebooks.

`--check` executes to temporary artifacts and checks outputs; `--validate-only`
checks existing outputs without running code. Without names the builder selects
both showcase and curso suites; use `--suite showcase` or `--suite curso` to
restrict discovery. Some Bayesian and simulation examples need substantially
more computation than introductory examples.

## Other collections and execution environments

- [`course/`](course/) contains a separate Spanish teaching sequence.
- [`curso/notebooks/`](../curso/notebooks/README.md) contains course companions.
- The [tablet guide](../docs/tablet.md) explains Pyodide, offline data and optional
  file-format dependencies. The numerical examples use NumPy, SciPy, pandas and
  Matplotlib; local LLM inference needs a desktop engine and otherwise uses a mock.

A numerical assertion checks the stated property of that experiment. It does
not turn a synthetic calibration into empirical evidence. See the learning guide
for interpretation notes and the [structural validation status](../docs/STRUCTURAL_VALIDATION_STATUS.md)
for the trade models' supported scope.
