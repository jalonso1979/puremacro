"""Finite-path household welfare conditional on prices and fiscal transfers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .distributional import HouseholdGroups, _values, prepare_household_groups
from .household import calibrate_household, compute_household_welfare

__all__ = ["HouseholdIncidencePath", "household_incidence_path"]


@dataclass(frozen=True)
class HouseholdIncidencePath:
    households: pd.DataFrame
    periods: pd.DataFrame
    utility: pd.DataFrame
    metadata: dict

    def to_frame(self) -> pd.DataFrame:
        return self.households.copy(deep=True)


def household_incidence_path(
    groups: HouseholdGroups,
    relative_prices: pd.DataFrame,
    *,
    transfers: pd.DataFrame | None = None,
    transfer_totals: pd.Series | None = None,
    discount: float = .99,
    risk_aversion: float = 2.,
    ces_elasticity: float = 1.,
) -> HouseholdIncidencePath:
    """Consumption-equivalent welfare of a finite CES consumption path.

    A household spends its observed baseline budget plus its signed transfer
    each date, with no saving or labor response. Its within-period CES bundle
    is normalized to one at baseline. Intertemporal utility is CRRA in that
    bundle, with the stated discount and curvature. CEV is the constant change
    in baseline consumption delivering the same utility over exactly these
    dates. There is no terminal continuation or stochastic expectation here.

    ``relative_prices`` has dates as rows and sector labels as columns.
    Transfer rows must match these dates and columns must match the household
    labels. Their weighted sum must equal the explicitly declared aggregate
    funding at each date (zero by default). Dates represent equally spaced
    periods in ``groups.period``; discount is per such period. Parameters and
    financing rules are assumptions; the sample alone does not identify them.
    """
    g = prepare_household_groups(
        groups.expenditure, groups.population_weights, monetary_unit=groups.monetary_unit,
        period=groups.period, population_unit=groups.population_unit, provenance=groups.provenance,
        income_exposure=groups.income_exposure, baseline_budget=groups.baseline_budget)
    if not isinstance(relative_prices, pd.DataFrame) or relative_prices.empty:
        raise ValueError("relative_prices must be a nonempty date-by-sector DataFrame")
    if not relative_prices.index.is_unique or not relative_prices.index.is_monotonic_increasing:
        raise ValueError("dates must be unique and increasing")
    if not relative_prices.columns.is_unique or set(relative_prices.columns) != set(g.expenditure.columns):
        raise ValueError("price columns must match sector labels exactly")
    p = _values(relative_prices.reindex(columns=g.expenditure.columns), "relative_prices")
    if np.any(p <= 0):
        raise ValueError("relative_prices must be positive")
    T, n = len(p), len(g.expenditure)
    tr = np.zeros((T, n))
    if transfers is not None:
        if (not isinstance(transfers, pd.DataFrame) or not transfers.index.equals(relative_prices.index)
                or not transfers.columns.is_unique or set(transfers.columns) != set(g.expenditure.index)):
            raise ValueError("transfers must match dates and household labels exactly")
        tr = _values(transfers.reindex(columns=g.expenditure.index), "transfers")
    target = np.zeros(T)
    if transfer_totals is not None:
        if not isinstance(transfer_totals, pd.Series) or not transfer_totals.index.equals(relative_prices.index):
            raise ValueError("transfer_totals must match the price dates")
        target = _values(transfer_totals, "transfer_totals")
    weights = g.population_weights.to_numpy()
    actual = tr @ weights
    scale = np.maximum.reduce([np.abs(target), np.abs(tr) @ weights, np.ones(T)])
    if np.any(np.abs(actual - target) > 1e-12 * scale):
        raise ValueError("transfers do not conserve the declared date-specific funding")
    if not np.isfinite(discount) or not 0 < discount <= 1:
        raise ValueError("discount must lie in (0, 1]")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("risk_aversion must be finite and positive")
    prefs = calibrate_household(g.expenditure.to_numpy().T, "ces", ces_elasticity=ces_elasticity,
                               sector_codes=g.expenditure.columns, country_codes=g.expenditure.index)
    m0 = g.baseline_budget.to_numpy()
    logu = np.empty((T, n))
    period_rows = []
    for t in range(T):
        result = compute_household_welfare(prefs, np.broadcast_to(p[t, :, None], prefs.x0.shape), m0 + tr[t])
        utility = (m0 + tr[t]) / (m0 * result.cost_index)
        if np.any(utility <= 0) or not np.isfinite(utility).all():
            raise ValueError("every household-date utility bundle must be positive and finite")
        logu[t] = np.log(utility)
        period_rows.append({"mean_ev": float(weights @ result.ev / weights.sum()),
                            "total_transfers": float(actual[t]),
                            "fiscal_residual": float(actual[t] - target[t]),
                            "mean_nominal_spending": float(weights @ (m0 + tr[t]) / weights.sum())})
    log_weights = np.arange(T) * np.log(discount)
    log_weights -= logsumexp(log_weights)
    power = 1 - risk_aversion
    if abs(power) < 1e-5:
        # Center the expansion to retain the log-utility limit without cancellation.
        center = np.exp(log_weights) @ logu
        if power == 0:
            equivalent = center
        else:
            equivalent = center + np.log1p(np.exp(log_weights) @ np.expm1(power * (logu - center))) / power
    else:
        equivalent = logsumexp(log_weights[:, None] + power * logu, axis=0) / power
    cev_fraction = np.expm1(equivalent)
    frame = pd.DataFrame({"population_weight": weights, "baseline_budget": m0,
                          "cev": m0 * cev_fraction, "cev_pct": 100 * cev_fraction},
                         index=g.expenditure.index.copy())
    return HouseholdIncidencePath(frame, pd.DataFrame(period_rows, index=relative_prices.index.copy()),
        pd.DataFrame(np.exp(logu), index=relative_prices.index.copy(), columns=g.expenditure.index.copy()),
        {"discount": float(discount), "risk_aversion": float(risk_aversion), "ces_elasticity": float(ces_elasticity),
         "monetary_unit": g.monetary_unit, "period": g.period, "population_unit": g.population_unit,
         "source": g.provenance["source"], "horizon": T,
         "scope": "finite deterministic consumption path; no saving, labor response or household feedback into GE",
         "welfare": "constant baseline consumption change over the same finite horizon; positive CEV is a gain",
         "terminal_condition": "no continuation beyond the recorded horizon",
         "funding": target.tolist(), "maximum_fiscal_residual": float(np.max(np.abs(actual-target)))})
