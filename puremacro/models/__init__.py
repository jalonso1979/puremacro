"""Structural models for puremacro.

Currently:
  - dmp_regime_dependent: DMP search-and-matching with a regime-dependent
    vacancy-cost multiplier (precautionary) and a regime-dependent
    discount factor (reaction-function expectation).
  - hank_sequence_space: Sequence-space heterogeneous-agent New Keynesian model
    (Auclert et al. 2021).
  - monetary_transmission: Comparative HANK vs RANK monetary and macroprudential
    transmission simulator (Kaplan, Moll & Violante 2018).
  - trade_policy: Multi-country multi-sector general equilibrium trade policy
    simulator (Caliendo & Parro 2015 exact hat algebra).
"""
from puremacro.models.dmp_regime_dependent import (
    DMPParameters,
    DMPState,
    dmp_irf,
    dmp_steady_state,
)
from puremacro.models.hank_sequence_space import (
    FakeNewsResult,
    FiscalTransferResult,
    NonlinearHANKResult,
    SequenceSpaceHANKResult,
    TwoAssetSequenceSpaceHANKResult,
    fake_news_algorithm,
    simulate_targeted_transfer,
    solve_hank_sequence_space,
    solve_nonlinear_transition,
    solve_two_asset_hank_sequence_space,
)
from puremacro.models.monetary_transmission import (
    MonetaryTransmissionResult,
    MonetaryTransmissionSimulator,
)
from puremacro.models.one_asset_hank import (
    OneAssetHANKResult,
    OneAssetHANKSteadyState,
    calibrate_one_asset_hank,
    solve_one_asset_hank,
)
from puremacro.models.trade_policy import (
    TradePolicySimulationResult,
    TradePolicySimulator,
)
from puremacro.models.hank_scenario_tree import (
    HANKScenarioTreeResult, monetary_scenario_tree, solve_hank_scenario_tree,
)
from puremacro.models.hank_equilibrium import (
    HANKEquilibriumWorkspace, solve_hank_equilibrium,
)
from puremacro.models.hank_recursive import (
    HANKDistributionBasis, HANKStochasticResult, HANKStochasticTreeResult,
    StochasticHANKPolicy, solve_hank_stochastic_policy, solve_hank_stochastic_tree,
)
from puremacro.models.hank_two_asset_equilibrium import (
    TwoAssetHANKSteadyState, calibrate_two_asset_equilibrium,
)
from puremacro.models.hank_reduction import (
    HANKMarginalBasis, ReducedStochasticHANKPolicy,
    solve_hank_reduced_policy, solve_hank_reduced_kernel,
)
from puremacro.models.hank_capital import (
    CapitalHANKSteadyState, calibrate_capital_hank, capital_production_block,
)
from puremacro.models.hank_nonlinear_continuation import (
    HANKNonlinearCorrection, HANKNonlinearSimulation, HANKContinuationAccuracyError,
    solve_hank_nonlinear_continuation, correct_hank_stochastic_state, simulate_hank_nonlinear,
)
from puremacro.models.information_nk import (
    InformationNKParameters, InformationNKSolution, InformationNKPath,
    solve_information_nk, audit_information_nk,
)

__all__ = [
    "DMPParameters",
    "DMPState",
    "dmp_steady_state",
    "dmp_irf",
    "SequenceSpaceHANKResult",
    "TwoAssetSequenceSpaceHANKResult",
    "FakeNewsResult",
    "FiscalTransferResult",
    "NonlinearHANKResult",
    "OneAssetHANKResult",
    "OneAssetHANKSteadyState",
    "calibrate_one_asset_hank",
    "solve_one_asset_hank",
    "solve_hank_sequence_space",
    "solve_two_asset_hank_sequence_space",
    "fake_news_algorithm",
    "simulate_targeted_transfer",
    "solve_nonlinear_transition",
    "MonetaryTransmissionSimulator",
    "MonetaryTransmissionResult",
    "TradePolicySimulator",
    "TradePolicySimulationResult",
    "HANKScenarioTreeResult", "monetary_scenario_tree", "solve_hank_scenario_tree",
    "HANKEquilibriumWorkspace", "solve_hank_equilibrium",
    "HANKDistributionBasis", "HANKStochasticResult", "HANKStochasticTreeResult",
    "StochasticHANKPolicy", "solve_hank_stochastic_policy", "solve_hank_stochastic_tree",
    "TwoAssetHANKSteadyState", "calibrate_two_asset_equilibrium",
    "HANKMarginalBasis", "ReducedStochasticHANKPolicy",
    "solve_hank_reduced_policy", "solve_hank_reduced_kernel",
    "CapitalHANKSteadyState", "calibrate_capital_hank", "capital_production_block",
    "HANKNonlinearCorrection", "HANKNonlinearSimulation", "HANKContinuationAccuracyError",
    "solve_hank_nonlinear_continuation", "correct_hank_stochastic_state", "simulate_hank_nonlinear",
    "InformationNKParameters", "InformationNKSolution", "InformationNKPath",
    "solve_information_nk", "audit_information_nk",
]
