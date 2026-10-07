"""Utility-consistent welfare for the inelastic-labour dynamic model.

Ported from the IO engine's ``dynamic_model/native_welfare.py``. The unit of
comparison is a permanent proportional change in benchmark consumption. This
is an annual consumption-equivalent variation valued at benchmark prices, not
a structural decomposition or a sum of output changes. The routine is
deliberately independent of the equilibrium residual and accepts physical
consumption quantities rather than nominal expenditures.

It is not comparable one-to-one with
:func:`puremacro.trade.compute_hicksian_welfare`: that is static conditional
consumption welfare from an expenditure function; this is discounted CRRA
lifetime utility with fixed baskets, inelastic labour, financial autarky and
government consumption inside the household basket.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._results import ConsumptionEquivalentResult

__all__ = ["consumption_equivalent_welfare"]


def _positive_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return array


def _readonly(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    out = np.array(value, dtype=float, copy=True)
    out.flags.writeable = False
    return out


def consumption_equivalent_welfare(
    consumption: Any,
    *,
    baseline_consumption: Any,
    beta: float,
    risk_aversion: float,
    terminal_consumption: Any | None = None,
    baseline_price: Any | None = None,
    baseline_gdp: Any | None = None,
    country_codes: Sequence[str] = (),
) -> ConsumptionEquivalentResult:
    """CRRA consumption equivalents by country, with an optional stationary tail.

    ``consumption[t, c]`` is the date-``t`` quantity, ``t = 0, ..., T-1``.
    When supplied, ``terminal_consumption`` is repeated at dates ``T, T+1, ...``
    with weight exactly ``beta**T / (1 - beta)``. Without it, utility is
    normalized by the *finite* sum of date weights and the result must not
    be interpreted as infinite-horizon welfare.

    Let ``x_t = C_t / C_0`` and ``u(x) = (x**(1-sigma) - 1) / (1-sigma)``
    (log at ``sigma = 1``; an ``expm1``/``log1p`` branch keeps small changes
    exact near one). The equivalent ``e`` solves
    ``sum_t beta**t u(e) = sum_t beta**t u(x_t)`` on the stated horizon, so it
    follows the same CRRA parameter as the capital Euler equation. If a model
    parameter is the intertemporal elasticity, pass its reciprocal.

    A supplied stationary tail is an approximation to the equilibrium path.
    This function does not establish its accuracy: the caller must separately
    certify the terminal equilibrium, terminal gap and horizon extension
    (:func:`compare_horizons`).

    ``baseline_price`` values the equivalent annual change ``(e - 1) C_0``;
    ``baseline_gdp`` expresses that annual amount in percent of benchmark GDP
    (``ev_pct_gdp``). The present value uses the same discounted horizon as
    utility. There is no cross-country aggregation.

    Raises ``ValueError`` for nonpositive, nonfinite or misaligned inputs,
    ``beta`` outside (0, 1), nonpositive ``risk_aversion``, or floating-point
    overflow of the CRRA transform.
    """
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("beta must be finite and in (0, 1)")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("risk_aversion must be finite and strictly positive")
    values = np.asarray(consumption, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("consumption must be a nonempty time-by-country matrix")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("consumption must be finite and strictly positive")
    horizon, countries = values.shape
    codes = tuple(str(c) for c in country_codes)
    if codes and len(codes) != countries:
        raise ValueError("country_codes must match the number of consumption columns")
    base = _positive_vector(baseline_consumption, name="baseline_consumption", size=countries)
    # Subtract logs instead of dividing, to avoid overflow under changes of units.
    log_ratios = np.log(values) - np.log(base)[None, :]
    weights = np.exp(np.arange(horizon, dtype=float) * np.log(beta))
    tail_weight = 0.0
    if terminal_consumption is not None:
        terminal = _positive_vector(terminal_consumption, name="terminal_consumption", size=countries)
        tail_weight = beta ** horizon / (1.0 - beta)
        log_ratios = np.vstack((log_ratios, np.log(terminal) - np.log(base)))
        weights = np.r_[weights, tail_weight]
    total_weight = float(weights.sum())
    normalized_weights = weights / total_weight
    power = 1.0 - risk_aversion
    if power == 0:
        average_utility = normalized_weights @ log_ratios
        log_equivalent = average_utility
    else:
        transformed = power * log_ratios
        # Around log utility, expm1/log1p preserve both the constant consumption
        # case and small welfare changes without a discontinuous sigma cutoff.
        if float(np.max(np.abs(transformed))) < 0.5:
            moment_minus_one = normalized_weights @ np.expm1(transformed)
            log_equivalent = np.log1p(moment_minus_one) / power
            average_utility = moment_minus_one / power
        else:
            maximum = np.max(transformed, axis=0)
            log_moment = maximum + np.log(normalized_weights @ np.exp(transformed - maximum))
            log_equivalent = log_moment / power
            with np.errstate(over="raise", invalid="raise"):
                try:
                    average_utility = np.expm1(log_moment) / power
                except FloatingPointError as error:
                    raise ValueError("CRRA utility exceeds floating-point range") from error
    with np.errstate(over="raise", invalid="raise"):
        try:
            equivalent = np.exp(log_equivalent)
            equivalent_change = np.expm1(log_equivalent)
        except FloatingPointError as error:
            raise ValueError("Consumption equivalent exceeds floating-point range") from error

    annual = present = gdp_pct = None
    if baseline_price is not None:
        prices = _positive_vector(baseline_price, name="baseline_price", size=countries)
        annual = equivalent_change * base * prices
        present = annual * total_weight
        if not np.all(np.isfinite(annual)) or not np.all(np.isfinite(present)):
            raise ValueError("Monetary welfare exceeds floating-point range")
    if baseline_gdp is not None:
        if baseline_price is None:
            raise ValueError("baseline_price is required when baseline_gdp is supplied")
        gdp = _positive_vector(baseline_gdp, name="baseline_gdp", size=countries)
        gdp_pct = 100.0 * annual / gdp
    tail = terminal_consumption is not None
    interpretation = (
        "Stationary-tail approximation to infinite-horizon CRRA consumption-equivalent welfare, "
        "with the constant tail beginning at date T. Terminal and horizon accuracy require separate "
        "validation; annual expenditure is valued at benchmark consumption prices."
        if tail else
        "Finite-horizon CRRA consumption equivalent normalized by the finite sum of discount weights; "
        "annual expenditure is valued at benchmark consumption prices."
    )
    return ConsumptionEquivalentResult(
        consumption_equivalent_ratio=_readonly(equivalent),
        consumption_equivalent_pct=_readonly(100.0 * equivalent_change),
        normalized_lifetime_utility=_readonly(total_weight * average_utility),
        equivalent_annual_expenditure=_readonly(annual),
        equivalent_present_value_expenditure=_readonly(present),
        ev_pct_gdp=_readonly(gdp_pct),
        beta=float(beta),
        risk_aversion=float(risk_aversion),
        horizon=int(horizon),
        tail_included=tail,
        tail_is_approximation=tail,
        terminal_discount_weight=float(tail_weight),
        total_discount_weight=total_weight,
        interpretation=interpretation,
        country_codes=codes,
        metadata={"utility": "CRRA, inelastic labour, no direct stock services",
                  "positive_is_gain": True},
    )
