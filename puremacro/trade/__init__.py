"""Multi-country multi-sector Computable General Equilibrium (CGE) trade model.

This subpackage implements a quantitative general equilibrium model of international
trade under short-run Leontief input-output linkages, Cobb-Douglas value added, and
multilateral Geary-Khamis Purchasing Power Parity (PPP) indexation.

The subpackage conforms strictly to the puremacro Pyodide runtime contract, importing
exclusively from NumPy, SciPy, Pandas, and Matplotlib.

Main Components
---------------
1. Data Ingestion (:mod:`puremacro.trade.data`):
   OECD ICIO table loading, 77-country ISO indexing, and 11-sector aggregation.
2. Calibration (:mod:`puremacro.trade.calibration`):
   Factor shares, technology scale parameters, technical coefficients, and endowments.
3. Nonlinear Equilibrium (:mod:`puremacro.trade.equilibrium`, :mod:`puremacro.trade.solver`):
   2,001-equation residual evaluation and general equilibrium solver.
4. Post-Processing & Flow Accounting (:mod:`puremacro.trade.postprocessing`):
   Reconstruction of bilateral flows, tariff accounting, and terms of trade.
5. Multilateral PPP (:mod:`puremacro.trade.geary_khamis`):
   Fixed-point Geary-Khamis international dollar reference prices and real GDP.
6. Scenario Engine (:mod:`puremacro.trade.scenarios`):
   Tariff shock specification (baseline, 10%, 25%, 54%, 75%, 125%, 145%).
7. LaTeX Tables & Reports (:mod:`puremacro.trade.tables`):
   Replication of publication impact tables.
8. Consistent accounting, Hicksian welfare and audited tariff policy
   (:mod:`puremacro.trade.welfare`, :mod:`puremacro.trade.policy_solver`).
9. Native MRIO tables with provenance (:mod:`puremacro.trade.mrio`):
   OECD ICIO, FIGARO and EXIOBASE readers, regularization, concordances and
   coarse tariff rules.
10. Condensed one-factor Leontief tariff model (:mod:`puremacro.trade.condensed`):
    exact 2N elimination, independent raw-flow certificate, numeraire-free measures.
11. Exact nested-CES block Newton (:mod:`puremacro.trade.ces_newton`) and
    audited parameter continuation (:mod:`puremacro.trade.continuation`).
12. Household demand systems and money-metric welfare (:mod:`puremacro.trade.household`).
13. Reduced local-stability diagnostic (:mod:`puremacro.trade.stability`, experimental).
14. Perfect-foresight dynamic MRIO with sector capital (:mod:`puremacro.trade.dynamic`,
    experimental).
"""
from __future__ import annotations

# Result containers
from ._results import (
    CapacityBottleneckResult,
    EVDecompositionResult,
    GearyKhamisResult,
    JCurveDynamicResult,
    NashTariffResult,
    OptimalTariffResult,
    RetaliationGameResult,
    RevenueRecyclingResult,
    ScenarioBatchResult,
    TheoremValidationReport,
    TradeCalibrationResult,
    TradeEquilibriumResult,
    WelfarePayoffMatrixResult,
)

# Data ingestion and canonical registries
from .data import (
    CANONICAL_COUNTRY_CODES,
    CANONICAL_FINAL_DEMAND_CODES,
    CANONICAL_SECTOR_CODES,
    CANONICAL_SECTOR_NAMES,
    EU_COUNTRY_CODES,
    ICIOData,
    get_country_codes,
    get_eu_country_codes,
    get_final_demand_codes,
    get_sector_codes,
    get_sector_names,
    load_icio_data,
)

# Benchmark calibration engine
from .calibration import (
    calibrate_trade_model,
)

# General equilibrium residual evaluation
from .equilibrium import (
    compute_equilibrium_residuals,
    evaluate_equilibrium_residuals,
    pack_equilibrium_vector,
    unpack_equilibrium_vector,
)

# Post-processing flow accounting
from .postprocessing import (
    compute_postprocessing_flows,
)

# Nonlinear general equilibrium solver
from .solver import (
    build_initial_guess,
    solve_trade_equilibrium,
)

# GPU & Apple MLX acceleration engine
from .gpu import (
    BatchedJacobianEvaluator,
    DeviceInfo,
    detect_device,
    device_context,
    get_memory_usage,
    reset_peak_memory,
    select_compute_device,
    solve_homotopy_continuation,
    solve_trade_equilibrium_gpu,
    solve_trade_equilibrium_mlx,
)

# Multilateral Geary-Khamis PPP engine
from .geary_khamis import (
    compute_geary_khamis,
    compute_geary_khamis_ppp,
    restore_capital_formation,
    solve_multilateral_ppp,
)

# Declarative tariff scenario engine and batch runner
from .scenarios import (
    CANONICAL_SCENARIOS,
    CANONICAL_SCENARIO_NAMES,
    SCENARIOS,
    ExtendedTariffScenario,
    TariffScenario,
    build_tariff_matrices,
    get_canonical_scenario,
    list_canonical_scenarios,
    run_scenario_batch,
    run_tariff_scenario,
)

# Advanced CGE extensions engine
from .extensions import (
    BottleneckConfig,
    DynamicJCurveConfig,
    FiscalRecyclingConfig,
    RetaliationConfig,
    compute_contractual_half_life,
    compute_j_curve_turning_point,
    decompose_harberger_welfare,
    run_bottleneck_scenario,
    run_dynamic_j_curve_simulation,
    run_fiscal_recycling_scenario,
    simulate_analytical_j_curve,
    solve_retaliation_game,
)

# Flexible CGE modeling engine
from .flexible import (
    FlexibleEquilibriumResult,
    FlexibleMarketStructureConfig,
    FlexiblePreferenceConfig,
    FlexibleTechnologyConfig,
    FlexibleTradeEquilibriumResult,
    FlexibleTradeModelConfig,
    compute_atkeson_burstein_markups,
    compute_benchmark_market_shares,
    compute_convergence_diagnostics,
    compute_dixit_stiglitz_varieties,
    compute_nested_ces_costs,
    compute_nested_factor_demands,
    compute_stone_geary_final_demand,
    compute_variety_price_scaling,
    smooth_subsistence_scaling,
    solve_flexible_trade_equilibrium,
)

# Game-theoretic optimal tariffs and Nash equilibrium engine
from . import game, optimal_tariffs
from .optimal_tariffs import (
    benchmark_real_world_tariffs,
    build_strategic_tariffs,
    compute_unilateral_optimal_tariff,
    compute_welfare_payoff_matrix,
    evaluate_national_welfare,
    get_country_code,
    resolve_country_indices,
    solve_multilateral_nash_tariffs,
)

# LaTeX table generation and export
from .tables import (
    export_latex_tables,
    generate_mean_by_scenario_table,
    generate_selected_country_table,
    generate_weighted_mean_by_scenario_table,
    to_latex_mean_by_scenario,
    to_latex_mean_by_scenario_table,
    to_latex_selected_country,
    to_latex_selected_country_table,
    to_latex_selected_country_with_row,
    to_latex_weighted_mean_by_scenario,
    to_latex_weighted_mean_by_scenario_table,
)

# Publication-grade plotting routines
from .plot import (
    plot_country_impacts,
    plot_scenario_distributions,
    plot_tariff_escalation_curve,
    plot_terms_of_trade_vs_welfare,
)

# Quantitative spatial trade & gravity general equilibrium engine
from .caliendo_parro import (
    CaliendoParroModel,
    CaliendoParroResult,
)
from puremacro.spatial.allen_arkolakis import (
    AllenArkolakisModel,
    AllenArkolakisResult,
)

# Trade policy analytics engine
from .policy_analytics import (
    EffectiveRateOfProtectionResult,
    SupplyChainVulnerabilityResult,
    TariffRevenueIncidenceResult,
    WelfareDecompositionResult,
    calculate_tariff_revenue_incidence,
    compute_effective_rate_of_protection,
    compute_supply_chain_vulnerability,
    decompose_hicksian_ev_3way,
    decompose_welfare_effects,
    verify_theorems_1_to_4,
)
from .welfare import HicksianWelfareResult, compute_hicksian_welfare
from .policy_solver import PolicyEquilibriumError, solve_policy_equilibrium
from .solver import ViabilityResult

# MRIO engines: module attributes plus the package-level names below.
from . import ces_newton, condensed, continuation, dynamic, household, mrio, stability
from .ces_newton import (
    CESBlockJacobian,
    CESBlockNewtonResult,
    CESNewtonError,
    NestedCESTechnology,
    certify_ces_equilibrium,
    continue_tariff_homotopy,
    solve_ces_block_newton,
)
from .household import (
    HouseholdCalibrationResult,
    HouseholdDemandResult,
    HouseholdDomainError,
    HouseholdPreferences,
    HouseholdWelfareResult,
    SupernumeraryFitResult,
    calibrate_household,
    compute_household_welfare,
    compute_household_welfare_from_results,
    fit_supernumerary_share,
    household_expenditure_from_calibration,
    household_prices_from_result,
)
from .distributional import (
    DistributionalWelfareResult,
    HouseholdGroups,
    compute_distributional_welfare,
    distributional_welfare_from_results,
    prepare_household_groups,
)
from .continuation import (
    DirectTargetResult,
    ParameterContinuationFailure,
    ParameterContinuationResult,
    continue_parameter,
    sigma_path,
    try_starts,
)
from .stability import StabilityError, StabilityResult, reduced_stability
from .mrio import (
    CoarseTariffResult,
    Concordance,
    MRIOAccountingReport,
    MRIOBuildReport,
    MRIOIntegrityError,
    MRIOTable,
    SourceRecord,
    aggregate_mrio,
    check_oecd_source,
    coarse_tariff_rates,
    identify_source,
    read_exiobase_native,
    read_figaro_native,
    read_oecd_native,
    regularize_table,
    to_calibration_matrix,
)
from .condensed import (
    BalancedIOTable,
    CondensedEquilibriumResult,
    CondensedMeasuresResult,
    CondensedSolveError,
    RawFlowCertificate,
    TariffWedges,
    build_tariff_wedges,
    calibrate_condensed,
    certify_raw_flows,
    compute_measures,
    solve_condensed,
)
from .dynamic import (
    ConsumptionEquivalentResult,
    DynamicAccounts,
    DynamicCalibration,
    DynamicDeterminacyError,
    DynamicEconomy,
    DynamicSolveError,
    DynamicStabilityResult,
    DynamicSteadyStateResult,
    DynamicTariff,
    DynamicTransitionResult,
    calibrate_dynamic,
    run_horizon_ladder,
    solve_dynamic_steady_state,
    solve_dynamic_transition,
    tariff_path,
)

__all__ = [
    "DistributionalWelfareResult",
    "HouseholdGroups",
    "compute_distributional_welfare",
    "distributional_welfare_from_results",
    "prepare_household_groups",
    # Result dataclasses
    "TradeCalibrationResult",
    "TradeEquilibriumResult",
    "GearyKhamisResult",
    "ScenarioBatchResult",
    "RetaliationGameResult",
    "JCurveDynamicResult",
    "RevenueRecyclingResult",
    "CapacityBottleneckResult",
    "OptimalTariffResult",
    "NashTariffResult",
    "WelfarePayoffMatrixResult",
    "TheoremValidationReport",
    "EVDecompositionResult",
    "HicksianWelfareResult",
    "compute_hicksian_welfare",
    "PolicyEquilibriumError",
    "solve_policy_equilibrium",
    # Data & Calibration entry points
    "load_icio_data",
    "calibrate_trade_model",
    "get_country_codes",
    "get_sector_codes",
    "get_final_demand_codes",
    "get_eu_country_codes",
    "get_sector_names",
    "ICIOData",
    "CANONICAL_COUNTRY_CODES",
    "EU_COUNTRY_CODES",
    "CANONICAL_SECTOR_CODES",
    "CANONICAL_SECTOR_NAMES",
    "CANONICAL_FINAL_DEMAND_CODES",
    # Equilibrium & Post-processing routines
    "compute_equilibrium_residuals",
    "evaluate_equilibrium_residuals",
    "unpack_equilibrium_vector",
    "pack_equilibrium_vector",
    "compute_postprocessing_flows",
    # Solver routines
    "solve_trade_equilibrium",
    "build_initial_guess",
    "ViabilityResult",
    # GPU & Apple MLX acceleration engine
    "solve_trade_equilibrium_gpu",
    "solve_trade_equilibrium_mlx",
    "solve_homotopy_continuation",
    "BatchedJacobianEvaluator",
    "detect_device",
    "select_compute_device",
    "device_context",
    "get_memory_usage",
    "reset_peak_memory",
    "DeviceInfo",
    # Multilateral PPP routines
    "compute_geary_khamis",
    "compute_geary_khamis_ppp",
    "restore_capital_formation",
    "solve_multilateral_ppp",
    # Scenario routines & specifications
    "TariffScenario",
    "ExtendedTariffScenario",
    "SCENARIOS",
    "CANONICAL_SCENARIOS",
    "CANONICAL_SCENARIO_NAMES",
    "get_canonical_scenario",
    "list_canonical_scenarios",
    "build_tariff_matrices",
    "run_tariff_scenario",
    "run_scenario_batch",
    # Advanced CGE extensions engine
    "RetaliationConfig",
    "solve_retaliation_game",
    "DynamicJCurveConfig",
    "compute_j_curve_turning_point",
    "compute_contractual_half_life",
    "simulate_analytical_j_curve",
    "run_dynamic_j_curve_simulation",
    "FiscalRecyclingConfig",
    "decompose_harberger_welfare",
    "run_fiscal_recycling_scenario",
    "BottleneckConfig",
    "run_bottleneck_scenario",
    # Tables & Reports
    "generate_selected_country_table",
    "generate_mean_by_scenario_table",
    "generate_weighted_mean_by_scenario_table",
    "to_latex_selected_country_table",
    "to_latex_selected_country",
    "to_latex_mean_by_scenario_table",
    "to_latex_mean_by_scenario",
    "to_latex_weighted_mean_by_scenario_table",
    "to_latex_weighted_mean_by_scenario",
    "to_latex_selected_country_with_row",
    "export_latex_tables",
    # Data Visualization
    "plot_country_impacts",
    "plot_scenario_distributions",
    "plot_tariff_escalation_curve",
    "plot_terms_of_trade_vs_welfare",
    # Game-theoretic Nash engine
    "optimal_tariffs",
    "game",
    # MRIO engine subpackages and modules
    "mrio",
    "condensed",
    "ces_newton",
    "household",
    "continuation",
    "stability",
    "dynamic",
    "resolve_country_indices",
    "get_country_code",
    "evaluate_national_welfare",
    "build_strategic_tariffs",
    "compute_unilateral_optimal_tariff",
    "solve_multilateral_nash_tariffs",
    "compute_welfare_payoff_matrix",
    "benchmark_real_world_tariffs",
    # Quantitative spatial trade & gravity general equilibrium engine
    "CaliendoParroModel",
    "CaliendoParroResult",
    "AllenArkolakisModel",
    "AllenArkolakisResult",
    # Trade policy analytics suite
    "EffectiveRateOfProtectionResult",
    "WelfareDecompositionResult",
    "SupplyChainVulnerabilityResult",
    "TariffRevenueIncidenceResult",
    "compute_effective_rate_of_protection",
    "decompose_welfare_effects",
    "compute_supply_chain_vulnerability",
    "calculate_tariff_revenue_incidence",
    "verify_theorems_1_to_4",
    "decompose_hicksian_ev_3way",
    # Flexible CGE modeling engine
    "FlexibleTechnologyConfig",
    "FlexiblePreferenceConfig",
    "FlexibleMarketStructureConfig",
    "FlexibleTradeModelConfig",
    "FlexibleTradeEquilibriumResult",
    "FlexibleEquilibriumResult",
    "compute_atkeson_burstein_markups",
    "compute_benchmark_market_shares",
    "compute_convergence_diagnostics",
    "compute_dixit_stiglitz_varieties",
    "compute_nested_ces_costs",
    "compute_nested_factor_demands",
    "compute_stone_geary_final_demand",
    "compute_variety_price_scaling",
    "smooth_subsistence_scaling",
    "solve_flexible_trade_equilibrium",
    # Exact nested-CES block Newton
    "NestedCESTechnology",
    "CESBlockJacobian",
    "CESBlockNewtonResult",
    "CESNewtonError",
    "solve_ces_block_newton",
    "continue_tariff_homotopy",
    "certify_ces_equilibrium",
    # Household demand systems and welfare
    "HouseholdDomainError",
    "HouseholdCalibrationResult",
    "HouseholdPreferences",
    "HouseholdDemandResult",
    "HouseholdWelfareResult",
    "SupernumeraryFitResult",
    "calibrate_household",
    "fit_supernumerary_share",
    "compute_household_welfare",
    "compute_household_welfare_from_results",
    "household_expenditure_from_calibration",
    "household_prices_from_result",
    # Audited parameter continuation
    "DirectTargetResult",
    "ParameterContinuationFailure",
    "ParameterContinuationResult",
    "continue_parameter",
    "sigma_path",
    "try_starts",
    # Reduced local stability (experimental)
    "StabilityError",
    "StabilityResult",
    "reduced_stability",
    # Native MRIO tables with provenance
    "MRIOTable",
    "SourceRecord",
    "MRIOAccountingReport",
    "MRIOBuildReport",
    "MRIOIntegrityError",
    "Concordance",
    "CoarseTariffResult",
    "identify_source",
    "check_oecd_source",
    "read_oecd_native",
    "read_figaro_native",
    "read_exiobase_native",
    "regularize_table",
    "aggregate_mrio",
    "coarse_tariff_rates",
    "to_calibration_matrix",
    # Condensed one-factor Leontief tariff model
    "BalancedIOTable",
    "calibrate_condensed",
    "build_tariff_wedges",
    "TariffWedges",
    "solve_condensed",
    "CondensedEquilibriumResult",
    "compute_measures",
    "CondensedMeasuresResult",
    "certify_raw_flows",
    "RawFlowCertificate",
    "CondensedSolveError",
    # Dynamic MRIO with sector capital (experimental)
    "DynamicAccounts",
    "DynamicCalibration",
    "calibrate_dynamic",
    "DynamicTariff",
    "tariff_path",
    "DynamicEconomy",
    "DynamicSolveError",
    "DynamicDeterminacyError",
    "solve_dynamic_steady_state",
    "solve_dynamic_transition",
    "run_horizon_ladder",
    "DynamicSteadyStateResult",
    "DynamicTransitionResult",
    "DynamicStabilityResult",
    "ConsumptionEquivalentResult",
]
