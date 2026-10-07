"""Static group incidence conditional on supplied prices and nominal incomes.

Group expenditure is mean spending per represented person or household, not
the group's total. Expansion weights count those represented units. This
module delegates preferences and exact EV/CV to :mod:`.household`; it neither
estimates microdata parameters nor feeds household demand back into a GE solve.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from .household import calibrate_household, compute_household_welfare, household_prices_from_result
from ._results import TradeCalibrationResult, TradeEquilibriumResult

__all__ = ["HouseholdGroups", "DistributionalWelfareResult", "prepare_household_groups",
           "compute_distributional_welfare", "distributional_welfare_from_results"]


def _labels(index: pd.Index, name: str) -> None:
    if not index.is_unique or any(not isinstance(x, str) or not x.strip() for x in index):
        raise ValueError(f"{name} must contain unique nonempty string labels")


def _values(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real and finite")
    try:
        out = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be real and finite")
    return out


def _series(value: Any, labels: pd.Index, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a labeled pandas Series")
    _labels(value.index, name)
    if set(value.index) != set(labels):
        raise ValueError(f"{name} labels must match {list(labels)} exactly")
    return pd.Series(_values(value.reindex(labels), name), index=labels, name=name)


def _frame(value: Any, rows: pd.Index, columns: pd.Index | None, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a labeled pandas DataFrame")
    _labels(value.index, f"{name} groups")
    _labels(value.columns, f"{name} columns")
    if set(value.index) != set(rows) or (columns is not None and set(value.columns) != set(columns)):
        raise ValueError(f"{name} labels must match the household groups and sectors exactly")
    aligned = value.reindex(index=rows, columns=columns if columns is not None else value.columns)
    return pd.DataFrame(_values(aligned, name), index=aligned.index, columns=aligned.columns)


@dataclass(frozen=True)
class HouseholdGroups:
    """Labeled survey aggregates; construct with :func:`prepare_household_groups`.

    ``income_exposure[g,f]`` is baseline nominal income from source ``f`` per
    represented unit. Sources can distinguish sector and factor, e.g.
    ``food_labor``. The observed budget need not equal income (saving and other
    resources may differ). Incremental exposed income is fully spent by
    assumption; baseline spending otherwise stays nominally fixed.
    DataFrames are copied by the adapter and revalidated at evaluation time.
    """

    expenditure: pd.DataFrame
    population_weights: pd.Series
    baseline_budget: pd.Series
    income_exposure: pd.DataFrame
    monetary_unit: str
    period: str
    population_unit: str
    provenance: dict[str, Any]


def prepare_household_groups(
    expenditure: pd.DataFrame,
    population_weights: pd.Series,
    *,
    monetary_unit: str,
    period: str,
    provenance: Mapping[str, Any],
    income_exposure: pd.DataFrame | None = None,
    population_unit: str = "households",
    baseline_budget: pd.Series | None = None,
) -> HouseholdGroups:
    """Validate mean group-by-sector expenditure and expansion weights.

    Monetary values are per ``population_unit`` per ``period`` in
    ``monetary_unit``. Weights are positive expansion counts, not spending
    weights. A normalized weight vector describes a population of size one;
    aggregate transfers must then use that same population scale. Expenditures
    must be nonnegative, every budget positive, and supplied budgets must equal
    row sums. Income exposure is nonnegative but may exceed spending because
    saving and other resources can differ. Full spending of incremental income
    is an explicit incidence assumption, not an estimated consumption response.

    For survey records, first compute each group's weighted mean sector
    spending using the *same complete household sample and denominator* for
    every sector, including zero purchases. Set its expansion count to the
    sum of household survey weights. Group means define representative groups,
    so nonlinear welfare of the means is not mean micro-household welfare.
    ``provenance`` must identify ``source``; empirical/synthetic tags are copied.
    """
    if not isinstance(expenditure, pd.DataFrame) or expenditure.empty:
        raise ValueError("expenditure must be a nonempty group-by-sector DataFrame")
    x = _frame(expenditure, expenditure.index, expenditure.columns, "expenditure")
    if np.any(x.to_numpy() < 0):
        raise ValueError("expenditure must be nonnegative")
    budget = x.sum(axis=1).rename("baseline_budget")
    if np.any(budget.to_numpy() <= 0):
        raise ValueError("every group must have a positive baseline budget")
    if baseline_budget is not None:
        supplied = _series(baseline_budget, x.index, "baseline_budget")
        if not np.allclose(supplied, budget, rtol=1e-10, atol=0):
            raise ValueError("baseline_budget must equal expenditure row sums")
    weights = _series(population_weights, x.index, "population_weights")
    if np.any(weights.to_numpy() <= 0) or not np.isfinite(weights.sum()):
        raise ValueError("population_weights must be positive expansion counts")
    exposure = (pd.DataFrame(index=x.index, dtype=float) if income_exposure is None
                else _frame(income_exposure, x.index, None, "income_exposure"))
    if np.any(exposure.to_numpy() < 0):
        raise ValueError("income_exposure must be nonnegative")
    for name, value in (("monetary_unit", monetary_unit), ("period", period),
                        ("population_unit", population_unit)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be explicitly specified as a nonempty string")
    if not isinstance(provenance, Mapping) or not isinstance(provenance.get("source"), str) or not provenance["source"].strip():
        raise ValueError("provenance must contain a nonempty source string")
    return HouseholdGroups(x, weights, budget, exposure, monetary_unit, period,
                           population_unit, deepcopy(dict(provenance)))


@dataclass(frozen=True)
class DistributionalWelfareResult:
    """Group means and population-weighted totals; positive EV/CV mean gains.

    ``groups`` contains monetary values per represented unit. ``aggregate``
    contains ``total_*`` monetary sums over expansion weights and population
    means. Aggregate ``ev_pct``/``cv_pct`` divide total welfare by total baseline
    spending; ``mean_ev_pct``/``mean_cv_pct`` instead average group percentages
    with population weights. Neither is a social welfare function.
    """

    groups: pd.DataFrame
    aggregate: pd.Series
    metadata: dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        return self.groups.copy(deep=True)

    def to_frame(self) -> pd.DataFrame:
        return self.to_dataframe()

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.to_dataframe(), **kwargs)

    def summary(self) -> str:
        return (f"Conditional household incidence ({self.metadata['rule']})\n"
                f"  groups: {len(self.groups)}; represented {self.metadata['population_unit']}: "
                f"{self.aggregate['total_population']:g}\n"
                f"  total EV: {self.aggregate['total_ev']:.6g} {self.metadata['monetary_unit']} "
                f"per {self.metadata['period']} ({self.aggregate['ev_pct']:.4g}% of baseline spending)\n"
                "  static partial-equilibrium incidence; no household feedback into GE\n")

    def plot(self, *, measure: str = "ev_pct", ax=None):
        """Plot group EV or CV in currency units or percent of group spending."""
        if measure not in ("ev", "cv", "ev_pct", "cv_pct"):
            raise ValueError("measure must be ev, cv, ev_pct or cv_pct")
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        ax.barh(self.groups.index, self.groups[measure], color="0.35")
        ax.axvline(0, color="0.1", linewidth=0.7)
        ax.set_xlabel("% of baseline group spending" if measure.endswith("_pct")
                      else f"{self.metadata['monetary_unit']} per {self.metadata['period']}")
        ax.set_title(f"Conditional group {measure.split('_')[0].upper()} (positive = gain)")
        ax.figure.tight_layout()
        return ax.figure


def compute_distributional_welfare(
    groups: HouseholdGroups,
    base_prices: pd.Series,
    prices: pd.Series,
    *,
    factor_income_changes: pd.Series | None = None,
    transfer_total: float = 0.0,
    transfer_weights: pd.Series | None = None,
    rule: Literal["fixed_baskets", "cobb_douglas", "ces", "stone_geary"] = "fixed_baskets",
    expenditure_elasticities: pd.DataFrame | None = None,
    supernumerary_share: float | pd.Series | None = None,
    ces_elasticity: float = 1.0,
) -> DistributionalWelfareResult:
    """Exact static EV/CV for each group using the existing preference engine.

    Prices are positive sector purchaser prices on a common numeraire. Their
    ratio rebases the observed baseline expenditure to benchmark unit prices.
    Income changes are fractions (``0.10`` means +10%), indexed by exactly the
    exposure sources, and bounded below by -1. Absent changes mean zero.

    ``transfer_total`` is the aggregate *change* in allocated revenue, in the
    groups' currency, period and represented-population scale. It can be
    negative. Allocation scores ``a_g`` are nonnegative per-unit entitlements:
    ``transfer_g = transfer_total * a_g / sum_h(weight_h * a_h)``. Defaults are
    equal per-unit transfers, not equal group totals. Thus weighted transfers
    conserve the supplied revenue. No fiscal revenue is inferred from incomes.

    Attribution telescopes anchored EV and CV along prices -> factor income ->
    transfers. It adds exactly but is order dependent and descriptive, not
    causal. Every intermediate state must satisfy the preference domain; LES
    can therefore reject a price-only state even if later transfers rescue it.
    LES elasticities and surplus shares are assumptions or external estimates,
    never identified from the group spending table.
    """
    if not isinstance(groups, HouseholdGroups):
        raise TypeError("groups must be a HouseholdGroups instance")
    g = prepare_household_groups(
        groups.expenditure, groups.population_weights, monetary_unit=groups.monetary_unit,
        period=groups.period, population_unit=groups.population_unit, provenance=groups.provenance,
        income_exposure=groups.income_exposure, baseline_budget=groups.baseline_budget)
    p0 = _series(base_prices, g.expenditure.columns, "base_prices")
    p1 = _series(prices, g.expenditure.columns, "prices")
    if np.any(p0.to_numpy() <= 0) or np.any(p1.to_numpy() <= 0):
        raise ValueError("base_prices and prices must be positive purchaser prices")
    changes = (pd.Series(0.0, index=g.income_exposure.columns) if factor_income_changes is None
               else _series(factor_income_changes, g.income_exposure.columns, "factor_income_changes"))
    if np.any(changes.to_numpy() < -1):
        raise ValueError("factor_income_changes are fractions and cannot be below -1")
    income = g.income_exposure.to_numpy() @ changes.to_numpy()
    scores = (pd.Series(1.0, index=g.expenditure.index) if transfer_weights is None
              else _series(transfer_weights, g.expenditure.index, "transfer_weights"))
    if np.any(scores.to_numpy() < 0) or not np.any(scores.to_numpy() > 0):
        raise ValueError("transfer_weights must be nonnegative with a positive weighted sum")
    revenue = _values(transfer_total, "transfer_total")
    if revenue.ndim != 0:
        raise ValueError("transfer_total must be a scalar aggregate amount")
    weights = g.population_weights.to_numpy()
    allocation = scores.to_numpy() / scores.max()
    transfer = float(revenue) * (allocation / np.dot(weights, allocation))
    eta = (None if expenditure_elasticities is None else
           _frame(expenditure_elasticities, g.expenditure.index, g.expenditure.columns,
                  "expenditure_elasticities").to_numpy().T)
    surplus = (_series(supernumerary_share, g.expenditure.index, "supernumerary_share").to_numpy()
               if isinstance(supernumerary_share, pd.Series) else supernumerary_share)
    if surplus is not None and not isinstance(supernumerary_share, pd.Series):
        surplus = _values(surplus, "supernumerary_share")
        if surplus.ndim != 0:
            raise ValueError("supernumerary_share must be a scalar or labeled group Series")
    prefs = calibrate_household(g.expenditure.to_numpy().T, rule,
                               expenditure_elasticities=eta, supernumerary_share=surplus,
                               ces_elasticity=ces_elasticity,
                               sector_codes=g.expenditure.columns, country_codes=g.expenditure.index)
    relative = np.broadcast_to((p1 / p0).to_numpy()[:, None], prefs.x0.shape)
    m0 = g.baseline_budget.to_numpy()
    price_state = compute_household_welfare(prefs, relative, m0)
    income_state = compute_household_welfare(prefs, relative, m0 + income)
    final = compute_household_welfare(prefs, relative, m0 + income + transfer)
    frame = pd.DataFrame({
        "population_weight": weights, "population_share": weights / weights.sum(),
        "baseline_budget": m0, "income_change": income, "transfer": transfer,
        "budget": final.budget, "cost_index": final.cost_index,
        "ev": final.ev, "cv": final.cv,
        "ev_pct": final.ev_pct_consumption, "cv_pct": final.cv_pct_consumption,
        "price_ev": price_state.ev, "income_ev": income_state.ev - price_state.ev,
        "transfer_ev": final.ev - income_state.ev,
        "price_cv": price_state.cv, "income_cv": income_state.cv - price_state.cv,
        "transfer_cv": final.cv - income_state.cv,
    }, index=g.expenditure.index.copy())
    frame.index.name = "group"
    totals = {f"total_{col}": float(np.dot(weights, frame[col]))
              for col in ("baseline_budget", "budget", "income_change", "transfer", "ev", "cv",
                          "price_ev", "income_ev", "transfer_ev", "price_cv", "income_cv", "transfer_cv")}
    totals.update({"total_population": float(weights.sum()),
                   "ev_pct": 100 * totals["total_ev"] / totals["total_baseline_budget"],
                   "cv_pct": 100 * totals["total_cv"] / totals["total_baseline_budget"],
                   "mean_ev": totals["total_ev"] / weights.sum(),
                   "mean_cv": totals["total_cv"] / weights.sum(),
                   "mean_ev_pct": float(np.average(frame["ev_pct"], weights=weights)),
                   "mean_cv_pct": float(np.average(frame["cv_pct"], weights=weights))})
    if not np.isfinite(frame.to_numpy()).all() or not np.isfinite(list(totals.values())).all():
        raise ValueError("incidence aggregation overflowed; rescale monetary or population units")
    if not np.isclose(totals["total_transfer"], float(revenue), rtol=1e-12, atol=1e-12 * abs(float(revenue))):
        raise ValueError("transfer allocation failed aggregate revenue conservation")
    metadata = {
        "source": g.provenance["source"], "provenance": deepcopy(g.provenance),
        "is_regression_fixture": bool(g.provenance.get("is_regression_fixture", False)),
        "monetary_unit": g.monetary_unit, "period": g.period, "population_unit": g.population_unit,
        "rule": rule, "preference_calibration": {
            **deepcopy(prefs.metadata),
            "target_source": prefs.calibration.target_source,
            "supernumerary_share": prefs.calibration.supernumerary_share.tolist(),
            "ces_elasticity": prefs.calibration.ces_elasticity,
            "adding_up_repair": prefs.calibration.adding_up_repair,
            "maximum_absolute_elasticity_adjustment": prefs.calibration.maximum_absolute_elasticity_adjustment,
            "repaired_expenditure_elasticities": prefs.calibration.repaired_expenditure_elasticities.tolist(),
            "identification": prefs.calibration.identification},
        "positive_is_gain": True, "income_change_unit": "fraction, 0.10 means +10%",
        "income_spending_assumption": "all incremental exposed nominal income is spent; baseline income need not equal observed consumption",
        "price_normalization": "counterfactual purchaser prices / supplied baseline purchaser prices",
        "attribution": "telescoping anchored EV/CV: prices -> factor income -> transfers; order dependent, descriptive, not causal",
        "scope": "static partial-equilibrium household incidence conditional on supplied GE prices/incomes; no household feedback or new GE welfare",
        "transfer_total": float(revenue), "transfer_conservation_error": totals["total_transfer"] - float(revenue),
        "baseline_prices": p0.to_dict(), "prices": p1.to_dict(),
        "factor_income_changes": changes.to_dict(), "transfer_weights": scores.to_dict(),
    }
    return DistributionalWelfareResult(frame, pd.Series(totals, name="aggregate"), metadata)


def distributional_welfare_from_results(
    groups: HouseholdGroups,
    calib: TradeCalibrationResult,
    baseline: TradeEquilibriumResult,
    counterfactual: TradeEquilibriumResult,
    *,
    country: str,
    factor_income_changes: pd.Series,
    baseline_tau_fd: Any | None = None,
    tau_fd: Any | None = None,
    category: int = 0,
    tol: float = 1e-8,
    **kwargs: Any,
) -> DistributionalWelfareResult:
    """Audit two GE states and use one country's household purchaser prices.

    ``factor_income_changes`` must explicitly map the group's income-source
    labels to fractional nominal changes on the GE prices' common numeraire.
    The caller supplies the factor/sector exposure reconciliation; national
    wages are never silently equated with household disposable income.
    ``transfer_total`` likewise remains an explicit argument in household
    units (GE tables may use millions while survey budgets use currency units).
    Other keyword arguments are passed to :func:`compute_distributional_welfare`.
    The bridge inherits the audited price helper's supported accounting and
    final-demand baskets. Flexible Armington results are not supported by that
    helper. Both states must have exactly the calibration's ordered labels.

    Within each sector, all survey groups inherit the selected GE country's
    fixed origin basket. The survey data do not identify group-specific import
    shares. Positive survey spending in a sector with no origin-basket support
    in that GE category is refused: the price helper's unit-price placeholder
    for a structural zero is not an observed household price. Supply explicit
    sector prices to :func:`compute_distributional_welfare` instead.
    """
    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError("calib must be a TradeCalibrationResult")
    if not isinstance(groups, HouseholdGroups):
        raise TypeError("groups must be a HouseholdGroups instance")
    countries = pd.Index(calib.country_codes)
    sectors = pd.Index(calib.sector_codes)
    _labels(countries, "calibration countries")
    _labels(sectors, "calibration sectors")
    if len(countries) != calib.n_countries or len(sectors) != calib.n_sectors:
        raise ValueError("calibration must provide complete country and sector labels")
    if country not in countries:
        raise ValueError(f"country {country!r} is absent from the calibration")
    if set(groups.expenditure.columns) != set(sectors):
        raise ValueError("household sector labels must match the calibration exactly")
    if factor_income_changes is None:
        raise ValueError("factor_income_changes must be an explicit labeled income mapping")
    for name, state in (("baseline", baseline), ("counterfactual", counterfactual)):
        if not isinstance(state, TradeEquilibriumResult):
            raise TypeError(f"{name} must be a TradeEquilibriumResult")
        if not state.converged:
            raise ValueError(f"{name} equilibrium did not converge")
        if tuple(state.country_codes) != tuple(countries) or tuple(state.sector_codes) != tuple(sectors):
            raise ValueError(f"{name} country/sector labels must match the calibration in order")
    if baseline.metadata.get("accounting", "legacy") != counterfactual.metadata.get("accounting", "legacy"):
        raise ValueError("both equilibria must use the same accounting mode")
    p0, _ = household_prices_from_result(baseline, calib, category=category,
                                         tau_fd=baseline_tau_fd, tol=tol)
    p1, _ = household_prices_from_result(counterfactual, calib, category=category,
                                         tau_fd=tau_fd, tol=tol)
    country_index = countries.get_loc(country)
    # The shared price helper returns one for a structural-zero sector, a
    # harmless placeholder only when consumption of that composite is zero.
    # A separately supplied survey basket can have positive spending there;
    # never interpret that placeholder as an unchanged empirical price.
    origin_weights = np.asarray(calib.afd, dtype=float)[:, category, country_index]
    sector_support = origin_weights.reshape(calib.n_countries, calib.n_sectors).sum(axis=0)
    survey_spending = _values(groups.expenditure.reindex(columns=sectors), "expenditure")
    unsupported = (sector_support <= 0) & np.any(survey_spending > 0, axis=0)
    if np.any(unsupported):
        missing = sectors[unsupported].tolist()
        raise ValueError(
            f"Positive household spending has no GE origin-basket support for country {country!r}, "
            f"category {category}, sectors {missing}. Supply explicit purchaser prices through "
            "compute_distributional_welfare; structural-zero placeholder prices cannot value this spending.")
    result = compute_distributional_welfare(
        groups, pd.Series(p0[:, country_index], index=sectors),
        pd.Series(p1[:, country_index], index=sectors), factor_income_changes=factor_income_changes, **kwargs)
    metadata = dict(result.metadata)
    metadata.update({"country": country, "category": category,
                     "ge_calibration_provenance": deepcopy(calib.metadata),
                     "ge_baseline_metadata": deepcopy(baseline.metadata),
                     "ge_counterfactual_metadata": deepcopy(counterfactual.metadata),
                     "ge_income_mapping": "caller-supplied nominal changes by household income source",
                     "ge_sector_price_assumption": "all groups inherit the selected country's fixed origin weights within each sector composite; group-specific import shares are not identified",
                     "ge_structural_zero_sectors": sectors[sector_support <= 0].tolist(),
                     "accounting": baseline.metadata.get("accounting", "legacy")})
    metadata["is_regression_fixture"] = (metadata["is_regression_fixture"] or
        any(bool(obj.metadata.get("is_regression_fixture", False)) for obj in (calib, baseline, counterfactual)))
    return DistributionalWelfareResult(result.groups, result.aggregate, metadata)
