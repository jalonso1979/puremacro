"""Convenience alias module forwarding to puremacro.trade.optimal_tariffs."""
from __future__ import annotations

from puremacro.trade.optimal_tariffs import (
    benchmark_real_world_tariffs,
    build_strategic_tariffs,
    compute_unilateral_optimal_tariff,
    compute_welfare_payoff_matrix,
    evaluate_national_welfare,
    get_country_code,
    resolve_country_indices,
    solve_multilateral_nash_tariffs,
)

__all__ = [
    "resolve_country_indices",
    "get_country_code",
    "evaluate_national_welfare",
    "build_strategic_tariffs",
    "compute_unilateral_optimal_tariff",
    "solve_multilateral_nash_tariffs",
    "compute_welfare_payoff_matrix",
    "benchmark_real_world_tariffs",
]
