"""Explicitly financed, static household compensation programs.

The fiscal ledger is in the survey's currency, period and population units.
It specifies lump-sum incidence conditional on the supplied tax base; it does
not estimate elasticities or solve labor-supply and government-debt responses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .distributional import HouseholdGroups, _series, _values, prepare_household_groups

__all__ = ["CompensationPlan", "balanced_compensation"]


@dataclass(frozen=True)
class CompensationPlan:
    """Per-unit gross benefits, taxes, net transfers and aggregate ledger."""

    households: pd.DataFrame
    ledger: pd.Series
    metadata: dict

    @property
    def net_transfers(self) -> pd.Series:
        return self.households["net_transfer"].copy()

    def to_frame(self) -> pd.DataFrame:
        return self.households.copy(deep=True)


def balanced_compensation(
    groups: HouseholdGroups,
    gross_total: float,
    *,
    benefit_scores: pd.Series,
    tax_base: pd.Series,
) -> CompensationPlan:
    """Allocate a fixed gross budget and raise exactly that amount in taxes.

    Both ``benefit_scores`` and ``tax_base`` are nonnegative *per represented
    unit*. Benefits equal ``gross_total * score / sum(weight * score)``;
    taxes use the same formula with the supplied tax base. Expansion counts
    enter each denominator once. Scores can encode household size or an
    eligibility rule. The tax base is fixed before the counterfactual; this
    is a budget-neutral lump-sum incidence rule, not an optimal-tax model.
    Zero gross expenditure is allowed; negative budgets and empty bases are
    refused. Negative net resources are checked by the welfare evaluator.
    """
    g = prepare_household_groups(
        groups.expenditure, groups.population_weights,
        monetary_unit=groups.monetary_unit, period=groups.period,
        population_unit=groups.population_unit, provenance=groups.provenance,
        income_exposure=groups.income_exposure, baseline_budget=groups.baseline_budget,
    )
    amount = _values(gross_total, "gross_total")
    if amount.ndim or float(amount) < 0:
        raise ValueError("gross_total must be a finite nonnegative scalar")
    scores = _series(benefit_scores, g.expenditure.index, "benefit_scores").to_numpy()
    base = _series(tax_base, g.expenditure.index, "tax_base").to_numpy()
    w = g.population_weights.to_numpy()
    payments = []
    for name, x in (("benefit_scores", scores), ("tax_base", base)):
        if np.any(x < 0) or not np.any(x > 0):
            raise ValueError(f"{name} must be nonnegative with positive weighted mass")
        scaled = x / x.max()
        payments.append(float(amount) * scaled / np.dot(w, scaled))
    benefits, taxes = payments
    net = benefits - taxes
    gross = float(w @ benefits)
    collected = float(w @ taxes)
    residual = float(w @ net)
    if not np.isfinite(np.concatenate([benefits, taxes, net, [gross, collected, residual]])).all():
        raise ValueError("fiscal amounts overflowed; rescale currency or population units")
    if abs(residual) > 1e-12 * max(float(amount), 1.0):
        raise ValueError("compensation failed its aggregate budget identity")
    frame = pd.DataFrame({"benefit": benefits, "tax": taxes, "net_transfer": net,
                          "population_weight": w}, index=g.expenditure.index.copy())
    ledger = pd.Series({"gross_benefits": gross, "tax_revenue": collected,
                        "net_transfers": residual, "requested_gross_total": float(amount)})
    return CompensationPlan(frame, ledger, {
        "monetary_unit": g.monetary_unit, "period": g.period,
        "population_unit": g.population_unit, "source": g.provenance["source"],
        "financing": "balanced budget; fixed-base lump-sum taxes",
        "behavior": "no endogenous labor, saving, eligibility or tax-base response",
    })
