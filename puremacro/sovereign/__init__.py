"""Quantitative Sovereign Default, Endogenous Maturity & Sudden-Stop Suite.

Implements:
1. Arellano (2008) & Chatterjee-Eyigungor (2012) endogenous default with
   short-term and long-term maturing coupon bonds, extreme-value taste shocks,
   and PCHIP bond-price schedule derivatives dq/db'.
2. A fiscal block on the short-term solver: fixed tax share, government
   spending in utility, and the implied spending, primary-balance and
   debt-to-output policies (``solve_fiscal_sovereign_default``).
3. Mendoza (2010) sudden-stop collateral constraints with Fisherian debt-deflation
   fire-sale amplification and precautionary deleveraging.
4. Sovereign spread term-structure decomposition and debt restructuring
   counterfactuals (haircuts and maturity extensions).
"""
from puremacro.sovereign._results import (
    FiscalSovereignResult,
    RestructuringResult,
    SovereignDefaultResult,
    SuddenStopResult,
)
from puremacro.sovereign.arellano import solve_sovereign_default
from puremacro.sovereign.fiscal import solve_fiscal_sovereign_default
from puremacro.sovereign.mendoza import solve_sudden_stop
from puremacro.sovereign.rules import FiscalRule
from puremacro.sovereign.restructuring import (
    decompose_spread_term_structure,
    evaluate_debt_restructuring,
)

__all__ = [
    "SovereignDefaultResult",
    "FiscalSovereignResult",
    "FiscalRule",
    "SuddenStopResult",
    "RestructuringResult",
    "solve_sovereign_default",
    "solve_fiscal_sovereign_default",
    "solve_sudden_stop",
    "decompose_spread_term_structure",
    "evaluate_debt_restructuring",
]
