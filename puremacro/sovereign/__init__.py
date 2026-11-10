"""Quantitative Sovereign Default, Endogenous Maturity & Sudden-Stop Suite.

Implements:
1. Arellano (2008) & Chatterjee-Eyigungor (2012) endogenous default with
   short-term and long-term maturing coupon bonds, extreme-value taste shocks,
   and PCHIP bond-price schedule derivatives dq/db'.
2. A fiscal block on the short-term solver: fixed tax share, government
   spending in utility, and the implied spending, primary-balance and
   debt-to-output policies (``solve_fiscal_sovereign_default``), with fiscal
   rules (hard/soft debt ceilings, deficit limits and recession escape
   clauses, ``FiscalRule``).
3. Mendoza (2010) sudden-stop collateral constraints with Fisherian debt-deflation
   fire-sale amplification and precautionary deleveraging.
4. Sovereign spread term-structure decomposition and debt restructuring
   counterfactuals (haircuts and maturity extensions).
5. Stochastic (reduced-form) and structural (solved fiscal model) debt
   sustainability analysis (DSA) simulations.
"""
from puremacro.sovereign._results import (
    FiscalSovereignResult,
    RestructuringResult,
    SovereignDefaultResult,
    StochasticDSAResult,
    SuddenStopResult,
)
from puremacro.sovereign.arellano import solve_sovereign_default
from puremacro.sovereign.adaptation import (
    SovereignAdaptationParameters,
    SovereignAdaptationResult,
    AdaptationDistribution,
    solve_sovereign_adaptation,
    audit_sovereign_adaptation,
    audit_adaptation_distribution,
)
from puremacro.sovereign.banking import (
    SovereignBankingParameters,
    SovereignBankingEquilibrium,
    SovereignBankingResult,
    sovereign_bank_accounts,
    solve_sovereign_banking,
    audit_sovereign_banking,
)
from puremacro.sovereign.dsa import (
    FiscalReactionEstimateResult,
    FiscalReactionFunction,
    compare_fiscal_rules_dsa,
    estimate_fiscal_reaction_function,
    fit_stochastic_dsa_from_data,
    simulate_fiscal_dsa,
    simulate_stochastic_dsa,
)
from puremacro.sovereign.fiscal import solve_fiscal_sovereign_default
from puremacro.sovereign.measurement import ergodic_moments, value_of_market_access
from puremacro.sovereign.mendoza import solve_sudden_stop
from puremacro.sovereign.restructuring import (
    decompose_spread_term_structure,
    evaluate_debt_restructuring,
)
from puremacro.sovereign.rules import FiscalRule
from puremacro.sovereign.smm import (
    SovereignSMMResult,
    calibrate_fiscal_sovereign_smm,
)

__all__ = [
    "SovereignAdaptationParameters",
    "SovereignAdaptationResult",
    "AdaptationDistribution",
    "solve_sovereign_adaptation",
    "audit_sovereign_adaptation",
    "audit_adaptation_distribution",
    "SovereignBankingParameters",
    "SovereignBankingEquilibrium",
    "SovereignBankingResult",
    "sovereign_bank_accounts",
    "solve_sovereign_banking",
    "audit_sovereign_banking",
    "SovereignDefaultResult",
    "FiscalSovereignResult",
    "FiscalRule",
    "SuddenStopResult",
    "RestructuringResult",
    "StochasticDSAResult",
    "solve_sovereign_default",
    "solve_fiscal_sovereign_default",
    "solve_sudden_stop",
    "decompose_spread_term_structure",
    "evaluate_debt_restructuring",
    "ergodic_moments",
    "value_of_market_access",
    "simulate_stochastic_dsa",
    "simulate_fiscal_dsa",
    "compare_fiscal_rules_dsa",
    "FiscalReactionFunction",
    "FiscalReactionEstimateResult",
    "estimate_fiscal_reaction_function",
    "fit_stochastic_dsa_from_data",
    "SovereignSMMResult",
    "calibrate_fiscal_sovereign_smm",
]
