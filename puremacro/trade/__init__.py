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
"""
from __future__ import annotations

# Result containers
from ._results import (
    CapacityBottleneckResult,
    GearyKhamisResult,
    JCurveDynamicResult,
    NashTariffResult,
    OptimalTariffResult,
    RetaliationGameResult,
    RevenueRecyclingResult,
    ScenarioBatchResult,
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
    decompose_welfare_effects,
)

__all__ = [
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
]

