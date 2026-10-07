"""Dynamic multi-region input-output model with sector-specific capital.

A perfect-foresight, deterministic dynamic extension of the fixed-coefficient
(Leontief) multi-country input-output model in which every source
country-industry cell keeps its own installed capital stock with Tobin-q
adjustment costs. Ported from the IO research engine (``dynamic_model/native_*``)
into puremacro conventions: frozen result objects with the report quintet,
validating constructors, structured errors and no raising ``__post_init__``.

Workflow
--------
1. :class:`DynamicAccounts` from bundled or raw tables
   (``from_icio``, ``from_trade_calibration``, ``from_arrays``, or the
   analytic fixture :func:`analytic_two_country_accounts`).
2. :func:`calibrate_dynamic` turns one IO year into an exactly stationary
   economy and reports every transformation (:class:`DynamicCalibration`).
3. :class:`DynamicEconomy` evaluates states, residuals, analytic
   Jacobian-vector products and the 17-entry independent certificate.
4. :class:`DynamicTariff` and :func:`tariff_path` describe announced,
   delayed, permanent or temporary duties.
5. :func:`solve_dynamic_steady_state`, :func:`solve_dynamic_transition`,
   :func:`compare_horizons` and :func:`run_horizon_ladder` solve and accept.
6. :func:`consumption_equivalent_welfare` values paths in CRRA consumption
   equivalents; :func:`stability_report` is the Blanchard-Kahn diagnostic.

See ``docs/trade_dynamic.md`` for the model, the acceptance contract and the
documented limitations (fixed sourcing, inelastic labour, financial autarky,
interior investment, unidentified dynamic parameters, long horizons, and the
77x11 indeterminacy finding).
"""
from __future__ import annotations

from ._results import (
    ConsumptionEquivalentResult,
    DynamicCalibration,
    DynamicDeterminacyError,
    DynamicSolveError,
    DynamicStabilityResult,
    DynamicSteadyStateResult,
    DynamicTransitionResult,
    EconomicDomainError,
    HorizonComparisonResult,
    HorizonLadderResult,
)
from .accounts import DynamicAccounts, analytic_two_country_accounts
from .calibration import calibrate_dynamic
from .economy import DynamicEconomy, DynamicTariff, tariff_path
from .stability import stability_report
from .transition import (
    compare_horizons,
    discounted_endpoint_capital_value,
    run_horizon_ladder,
    solve_dynamic_steady_state,
    solve_dynamic_transition,
)
from .welfare import consumption_equivalent_welfare

__all__ = [
    "DynamicAccounts",
    "analytic_two_country_accounts",
    "DynamicCalibration",
    "calibrate_dynamic",
    "DynamicTariff",
    "tariff_path",
    "DynamicEconomy",
    "EconomicDomainError",
    "DynamicSolveError",
    "DynamicDeterminacyError",
    "solve_dynamic_steady_state",
    "solve_dynamic_transition",
    "compare_horizons",
    "run_horizon_ladder",
    "discounted_endpoint_capital_value",
    "consumption_equivalent_welfare",
    "stability_report",
    "DynamicSteadyStateResult",
    "DynamicTransitionResult",
    "HorizonComparisonResult",
    "HorizonLadderResult",
    "ConsumptionEquivalentResult",
    "DynamicStabilityResult",
]
