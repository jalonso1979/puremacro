"""Practitioner & Hobbyist Portfolio Analysis Suite (``puremacro.portfolio``).

Provides Pyodide-compatible portfolio construction, covariance shrinkage,
Hierarchical Risk Parity, Black-Litterman Bayesian views, macro-quadrant
All-Weather allocation, Newey-West HAC factor attribution, rebalancing
backtests with transaction costs, and Monte Carlo retirement glidepaths.
"""
from __future__ import annotations

from puremacro.portfolio._results import (
    AllocationResult,
    FactorAttributionResult,
    MacroRegimePortfolioResult,
    PortfolioBacktestResult,
    WealthSimulationResult,
)
from puremacro.portfolio.allocation import (
    black_litterman,
    hierarchical_risk_parity,
    markowitz_frontier,
    optimize_portfolio,
    shrink_covariance,
)
from puremacro.portfolio.backtest import (
    backtest_portfolio,
    simulate_wealth_glidepath,
)
from puremacro.portfolio.macro_regime import (
    QUADRANTS,
    all_weather_portfolio,
    classify_macro_quadrants,
    factor_attribution,
)

__all__ = [
    "AllocationResult",
    "MacroRegimePortfolioResult",
    "FactorAttributionResult",
    "PortfolioBacktestResult",
    "WealthSimulationResult",
    "shrink_covariance",
    "optimize_portfolio",
    "markowitz_frontier",
    "hierarchical_risk_parity",
    "black_litterman",
    "QUADRANTS",
    "classify_macro_quadrants",
    "all_weather_portfolio",
    "factor_attribution",
    "backtest_portfolio",
    "simulate_wealth_glidepath",
]
