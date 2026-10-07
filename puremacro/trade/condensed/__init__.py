"""Condensed one-factor Leontief tariff model with an independent certificate and numeraire-free measures.

A static multi-country, multi-sector Leontief (fixed-coefficient) tariff model
with one composite factor per country, product taxes on output and on final
uses, and ad-valorem import duties rebated lump-sum. Spending is Cobb-Douglas
over three final baskets (C, G, X), each Leontief across origins; inventories
are exogenous quantities; trade balances are fixed shares of world factor
income (Dekle-Eaton-Kortum); the numeraire is world factor income. Prices and
outputs are eliminated exactly, leaving ``2N`` unknowns ``z = (log w, Y / Y0)``.

Ported from the IO research engine (``headlinePaper/rebuild/vendor/puremacro/
trade/corrected``) with puremacro conventions. It is a different model from
``solve_trade_equilibrium`` (two factors, fixed foreign saving, producer-price
numeraire) and from its ``accounting="consistent"`` mode; see
docs/trade_condensed.md.

Typical use::

    from puremacro.trade.condensed import (
        BalancedIOTable, calibrate_condensed, build_tariff_wedges, solve_condensed, compute_measures,
    )
    table = BalancedIOTable.from_arrays(Z, F, VA, TLS, TFD, country_codes=..., sector_codes=...,
                                        fd_codes=("C", "G", "X", "V"))
    calib = calibrate_condensed(table)
    wedges = build_tariff_wedges(calib, 0.10, importer="USA")
    result = solve_condensed(calib, wedges, run_arclength=True, run_multistart=True)
    measures = compute_measures(calib, wedges, result.state, country="USA")

Constants (``certify.CERTIFICATE_TOL``, ``table.FD_CODES``, ``table.ENDOGENOUS_FD``)
and the registry helper ``table.infer_merchandise_mask`` are importable from
their modules and are not re-exported here.
"""
from __future__ import annotations

from ._results import CondensedEquilibriumResult
from .calibration import (
    BaselineTariffAccounts,
    CondensedCalibration,
    ProductivityBounds,
    calibrate_condensed,
    certify_productivity,
    collatz_wielandt_bound,
    separate_baseline_tariffs,
)
from .certify import RawFlowCertificate, certify_raw_flows
from .errors import (
    CalibrationError,
    CertificationFailure,
    CondensedModelError,
    CondensedSolveError,
    ContinuationFailure,
    DataIntegrityError,
    EquilibriumNotFound,
    ExistenceViolation,
    FoldDetected,
    MultipleEquilibria,
    ProductivityUncertified,
    UnsupportedExtension,
)
from .measures import (
    CondensedMeasuresResult,
    EVSplit,
    aggregation_gaps,
    compute_measures,
    constant_price_rgdp,
    consumer_price_relative_to_wages,
    ev_split,
    pe_first_order_decomposition,
    pe_unit_cost_bound,
    real_national_income,
    sector_incidence,
    tariff_revenue_and_effective_rate,
    tot_fisher,
)
from .model import CondensedLeontiefModel, CondensedState
from .solve import (
    arclength,
    continuation,
    existence_gate,
    multistart_near,
    nested_fallback,
    newton,
    solve_condensed,
)
from .table import BalancedIOTable, TableBuildReport, table_from_arrays
from .tariffs import TariffWedges, build_tariff_wedges, scale_wedges

__all__ = [
    "BalancedIOTable",
    "BaselineTariffAccounts",
    "CalibrationError",
    "CertificationFailure",
    "CondensedCalibration",
    "CondensedEquilibriumResult",
    "CondensedLeontiefModel",
    "CondensedMeasuresResult",
    "CondensedModelError",
    "CondensedSolveError",
    "CondensedState",
    "ContinuationFailure",
    "DataIntegrityError",
    "EVSplit",
    "EquilibriumNotFound",
    "ExistenceViolation",
    "FoldDetected",
    "MultipleEquilibria",
    "ProductivityBounds",
    "ProductivityUncertified",
    "RawFlowCertificate",
    "TableBuildReport",
    "TariffWedges",
    "UnsupportedExtension",
    "aggregation_gaps",
    "arclength",
    "build_tariff_wedges",
    "calibrate_condensed",
    "certify_productivity",
    "certify_raw_flows",
    "collatz_wielandt_bound",
    "compute_measures",
    "constant_price_rgdp",
    "consumer_price_relative_to_wages",
    "continuation",
    "ev_split",
    "existence_gate",
    "multistart_near",
    "nested_fallback",
    "newton",
    "pe_first_order_decomposition",
    "pe_unit_cost_bound",
    "real_national_income",
    "scale_wedges",
    "sector_incidence",
    "separate_baseline_tariffs",
    "solve_condensed",
    "table_from_arrays",
    "tariff_revenue_and_effective_rate",
    "tot_fisher",
]
