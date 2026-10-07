"""Benchmark-normalized household demand with exact expenditure-function welfare.

Ported from the IO research module ``headlinePaper/preferences_2026-09-22/household.py``
(2026-09-22). The system is solver independent: it takes sector-composite
purchaser prices and a household budget per region and returns Marshallian and
Hicksian demand, the expenditure function, the cost-of-living index and exact
money-metric welfare. Origin substitution, taxes and the general-equilibrium
budget are external; two bridges recover prices and budgets from
:class:`~puremacro.trade.TradeEquilibriumResult` states.

Normalization
-------------
All arrays have shape ``(sector, region)`` except region budgets, utility and
welfare, which have shape ``(region,)``. A unit of a sector composite is the
quantity that costs one monetary unit at its benchmark purchaser price, so the
benchmark spending matrix ``x0`` is also the benchmark quantity matrix, and
benchmark prices and utility equal one. Regional budgets ``m0 = x0.sum(0)``,
shares ``w = x0 / m0``.

Preference rules (``rule``)
---------------------------
``fixed_baskets``
    ``q = x0 * m / (m0 * sum_i(w_i p_i))``, ``e(p, u) = m0 * u * sum_i(w_i p_i)``.
``cobb_douglas``
    LES with ``gamma = 0``: ``q_i = w_i m / p_i``, ``e(p, u) = m0 * u * prod_i p_i**w_i``.
``stone_geary``
    ``q_i = gamma_i + beta_i (m - sum_j p_j gamma_j) / p_i`` with ``sum_i beta_i = 1``
    and ``gamma_i >= 0``. Supernumerary price index ``P(p) = prod_i p_i**beta_i``;
    surplus ``S = m - sum_j p_j gamma_j``; ``S0 = lambda * m0``; benchmark-normalized
    utility ``u = S / (S0 * P(p))``. Expenditure function
    ``e(p, u) = sum_i p_i gamma_i + S0 * P(p) * u``; Hicksian demand
    ``h(p, u) = gamma + beta * S0 * P(p) * u / p = grad_p e``.
``ces``
    ``P(p) = (sum_i w_i p_i**(1 - sigma))**(1 / (1 - sigma))`` (an ``expm1``/``log1p``
    form is used away from ``sigma = 1``), ``q_i = w_i m p_i**(-sigma) P**(sigma - 1)``,
    ``e(p, u) = m0 * P(p) * u``. ``sigma = 0`` reproduces fixed baskets and
    ``sigma = 1`` Cobb-Douglas.

The full cost-of-living index at benchmark utility is ``e(p, 1) / m0``; ``P``
alone prices discretionary spending and is not that index.

Welfare
-------
With 0 the base state and 1 the comparison state, ``EV = e(p0, u1) - e(p0, u0)``
and ``CV = e(p1, u1) - e(p1, u0)``; both are positive for a welfare gain
(``cv`` is the amount removable at new prices, not the compensation required).
Against the calibration benchmark this is ``EV = S0 * (u1 - 1)``, valid for
every rule, and ``CV = m1 - e(p1, 1)``; for LES ``CV = P(p1) * EV``.
Percentages divide by the base household budget so cross-preference comparisons
share one denominator.

Calibration from expenditure elasticities (Stone-Geary)
-------------------------------------------------------
At the benchmark the LES expenditure elasticity is ``eta_i = beta_i / w_i``, so
``beta = w * eta``. Engel adding-up ``sum_i w_i eta_i = 1`` is imposed by dividing
each region's targets by their expenditure-weighted mean (recorded in the
calibration audit). ``gamma = x0 * (1 - lambda * eta)`` where ``lambda = S0 / m0``
is the supernumerary (discretionary) share; ``gamma >= 0`` requires
``0 < lambda <= 1 / max_i eta_i``. ``lambda = 1`` gives Cobb-Douglas. The IO
benchmark and Engel targets do not identify ``lambda``. A per-sector-and-region
``FlexiblePreferenceConfig.mu_s = 1 - lambda * eta`` (with the repaired ``eta``;
``1 - lambda`` for unit targets) reproduces the LES demand levels of
:mod:`puremacro.trade.flexible` at benchmark income only, and the adding-up
repair and the admissibility check ``lambda <= 1 / max(eta)`` must be performed
here first. The flexible general-equilibrium solver uses a fixed-proportion
supernumerary LES that nests the legacy basket (a uniform ``mu_s`` is neutral
there); the demand-level parity holds for ``compute_stone_geary_final_demand``
only. Only the demand levels coincide there: that module scales
subsistence with income (``tanh`` smoothing), so its demand has no expenditure
function and its finite-difference Slutsky matrix is asymmetric even at
benchmark income. The system here has an exact expenditure function.

Fitting ``lambda`` (:func:`fit_supernumerary_share`)
----------------------------------------------------
The LES benchmark compensated own-price elasticity is
``xi_ii = -lambda * eta_i * (1 - beta_i)``. Given targets and nonnegative weights
normalized per region, ``lambda_unc = -sum_i omega_i s_i xi_i / sum_i omega_i s_i**2``
with ``s_i = eta_i (1 - beta_i)``, clipped to ``[eps / max(eta), 1 / max(eta)]``.

Numerics
--------
Evaluation kernels deliberately retain complex inputs for complex-step
derivatives and never clip. Call :meth:`HouseholdPreferences.validate_domain`
at accepted real states or pass ``validate=True``: nonpositive prices,
nonfinite inputs and exhausted surplus raise :class:`HouseholdDomainError`.

Bridges
-------
:func:`household_prices_from_result` values sector composites at the prices
the equilibrium evaluator itself used: the audited ``_accounting.evaluate``
block for ``accounting="consistent"`` results, and the state prices in
``x_sol`` for legacy results (whose recorded ``p_fd`` was built from them).
Legacy results record neither the tariff schedule nor enough detail to verify
it: the supplied ``tau_fd`` is checked against the category-aggregate ``p_fd``,
which identifies national (origin- and sector-uniform) final-demand rates;
origin- or sector-specific legacy schedules raise unless the caller passes
``trust_tau_fd_detail=True``. Bridged EV/CV inherit the equilibrium's numeraire
and foreign-saving conventions.

Limitations quoted from the IO documentation
--------------------------------------------
MODEL.md: "This is a disclosed transfer and reconciliation, not native microdata
estimation"; "The inherited targets generally cannot be matched exactly by LES,
and a calibrated parameter is not thereby empirically identified for 2019";
"Its substitution restrictions are therefore substantive: matching income
elasticities and projecting compensated own-price targets does not reproduce a
general CDE substitution matrix"; "These remain static policy comparisons, not
lifetime welfare". README.md: "The IO accounts do not identify subsistence
requirements". Bundled OECD final-use category C is HFCE + NPISH + GGFC, so a
household calibrated from it values aggregate consumption including government
final consumption. See ``docs/trade_household.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from ._results import TradeCalibrationResult, TradeEquilibriumResult

__all__ = [
    "HouseholdDomainError",
    "HouseholdCalibrationResult",
    "HouseholdPreferences",
    "HouseholdDemandResult",
    "HouseholdWelfareResult",
    "SupernumeraryFitResult",
    "calibrate_household",
    "fit_supernumerary_share",
    "compute_household_welfare",
    "compute_household_welfare_from_results",
    "household_expenditure_from_calibration",
    "household_prices_from_result",
]

Rule = Literal["fixed_baskets", "cobb_douglas", "stone_geary", "ces"]
_RULES = ("fixed_baskets", "cobb_douglas", "stone_geary", "ces")


class HouseholdDomainError(ValueError):
    """A real household state lies outside the admissible preference domain."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _readonly(value: Any, dtype: Any = float) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


def _frozen(value: Any) -> np.ndarray:
    """Read-only copy that keeps a complex dtype (complex-step results)."""
    out = np.array(value, copy=True)
    out.flags.writeable = False
    return out


def _real_finite(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if np.iscomplexobj(result):
        raise HouseholdDomainError(f"{name} must be real for domain validation")
    if not np.all(np.isfinite(result)):
        raise HouseholdDomainError(f"{name} must be finite")
    return result


def _labels(codes: Sequence[str] | None, count: int, prefix: str, name: str) -> tuple[str, ...]:
    if codes is None:
        return tuple(f"{prefix}{i:02d}" for i in range(count))
    out = tuple(str(c) for c in codes)
    if len(out) != count:
        raise ValueError(f"{name} must have {count} entries, got {len(out)}")
    return out


def _index(codes: tuple[str, ...], count: int, prefix: str, name: str) -> pd.Index:
    """Labelled index; directly constructed results without codes get generated labels."""
    return pd.Index(list(codes) if codes else list(_labels(None, count, prefix, name)), name=name)


class _TableExports:
    """Markdown / LaTeX / Typst exporters routed through :mod:`puremacro.reports`."""

    def to_dataframe(self) -> pd.DataFrame:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.to_dataframe(), **kwargs)


# ---------------------------------------------------------------------------
# result objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HouseholdCalibrationResult(_TableExports):
    """Audit of :func:`calibrate_household` (the IO ``calibration`` dictionary).

    ``input_weighted_elasticity_mean`` is the expenditure-weighted mean of the
    supplied Engel targets before the adding-up repair; ``repaired_expenditure_elasticities``
    are the targets actually used (structural zeros stay zero);
    ``maximum_admissible_supernumerary_share`` is ``1 / max_i eta_i`` per region;
    ``subsistence_spending_share`` is ``sum_i gamma_i / m0``. ``identification``
    carries the IO qualification verbatim: surplus shares are supplied
    assumptions or external estimates, not identified by IO accounts and Engel
    targets.
    """

    rule: str
    target_source: str
    benchmark_price_normalization: str
    structural_zero_count: int
    adding_up_repair: str
    input_weighted_elasticity_mean: np.ndarray
    input_adding_up_max_error: float
    maximum_absolute_elasticity_adjustment: float
    repaired_expenditure_elasticities: np.ndarray
    supernumerary_share: np.ndarray
    maximum_admissible_supernumerary_share: np.ndarray
    subsistence_spending_share: np.ndarray
    ces_elasticity: float
    identification: str
    sector_codes: tuple[str, ...] = ()
    country_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Per-region calibration audit."""
        return pd.DataFrame({
            "weighted_elasticity_mean": self.input_weighted_elasticity_mean,
            "supernumerary_share": self.supernumerary_share,
            "maximum_admissible_share": self.maximum_admissible_supernumerary_share,
            "subsistence_spending_share": self.subsistence_spending_share,
        }, index=_index(self.country_codes, self.supernumerary_share.size, "C", "country"))

    def summary(self) -> str:
        """One-line description of the targets, repair and supernumerary shares."""
        return (f"Household calibration [{self.rule}]: targets {self.target_source}; "
                f"{self.structural_zero_count} structural zeros; adding-up max error "
                f"{self.input_adding_up_max_error:.3g}; max elasticity adjustment "
                f"{self.maximum_absolute_elasticity_adjustment:.3g}; supernumerary shares "
                f"{np.round(self.supernumerary_share, 4).tolist()}.")

    def as_dict(self) -> dict[str, Any]:
        """The IO ``calibration`` dictionary (lists, not arrays)."""
        return {
            "rule": self.rule,
            "target_source": self.target_source,
            "benchmark_price_normalization": self.benchmark_price_normalization,
            "structural_zero_count": self.structural_zero_count,
            "adding_up_repair": self.adding_up_repair,
            "input_weighted_elasticity_mean": self.input_weighted_elasticity_mean.tolist(),
            "input_adding_up_max_error": self.input_adding_up_max_error,
            "maximum_absolute_elasticity_adjustment": self.maximum_absolute_elasticity_adjustment,
            "repaired_expenditure_elasticities": self.repaired_expenditure_elasticities.tolist(),
            "supernumerary_share": self.supernumerary_share.tolist(),
            "maximum_admissible_supernumerary_share": self.maximum_admissible_supernumerary_share.tolist(),
            "subsistence_spending_share": self.subsistence_spending_share.tolist(),
            "ces_elasticity": self.ces_elasticity,
            "identification": self.identification,
        }


@dataclass(frozen=True)
class HouseholdDemandResult(_TableExports):
    """Marshallian state at prices ``p`` and budget ``m``.

    ``quantities`` has shape ``(sector, region)``; the other arrays have shape
    ``(region,)``. ``price_index`` is the rule's unit index ``P(p)`` (for LES the
    supernumerary index), ``cost_index`` the full cost-of-living index
    ``e(p, 1) / m0``, ``surplus`` the budget above subsistence cost
    (``m`` for homothetic rules) and ``utility`` the benchmark-normalized
    utility ``surplus / (surplus0 * P(p))``. Complex inputs give complex arrays.
    """

    quantities: np.ndarray
    price_index: np.ndarray
    cost_index: np.ndarray
    surplus: np.ndarray
    utility: np.ndarray
    budget: np.ndarray
    sector_codes: tuple[str, ...] = ()
    country_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Per-region budget, indices, surplus and utility."""
        return pd.DataFrame({
            "budget": self.budget, "price_index": self.price_index,
            "cost_index": self.cost_index, "surplus": self.surplus, "utility": self.utility,
        }, index=_index(self.country_codes, np.size(self.budget), "C", "country"))

    def quantities_frame(self) -> pd.DataFrame:
        """Composite quantities, sectors by regions."""
        ns, nc = np.shape(self.quantities)
        return pd.DataFrame(self.quantities, index=_index(self.sector_codes, ns, "S", "sector"),
                            columns=list(_index(self.country_codes, nc, "C", "country")))

    def summary(self) -> str:
        """One-line utility and cost-of-living summary (real parts)."""
        return (f"Household demand [{self.metadata.get('rule', '?')}]: utility "
                f"{np.round(np.real(self.utility), 6).tolist()}, cost index "
                f"{np.round(np.real(self.cost_index), 6).tolist()}.")


@dataclass(frozen=True)
class HouseholdWelfareResult(_TableExports):
    """Exact money-metric welfare per region; positive EV and CV are gains.

    ``EV = e(p0, u1) - e(p0, u0)`` and ``CV = e(p1, u1) - e(p1, u0)`` in the
    budget's value units. ``ev_pct_consumption`` and ``cv_pct_consumption``
    divide by the base household budget ``e(p0, u0)`` so both measures share one
    denominator. ``utility`` and ``base_utility`` are benchmark-normalized
    (one at the calibration point); ``cost_index`` is ``e(p1, 1) / m0`` and
    ``price_index`` the rule's unit index at ``p1``.
    """

    ev: np.ndarray
    cv: np.ndarray
    ev_pct_consumption: np.ndarray
    cv_pct_consumption: np.ndarray
    utility: np.ndarray
    base_utility: np.ndarray
    cost_index: np.ndarray
    price_index: np.ndarray
    budget: np.ndarray
    base_budget: np.ndarray
    country_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Per-region EV, CV, percentages, utilities, indices and budgets."""
        return pd.DataFrame({
            "ev": self.ev, "cv": self.cv,
            "ev_pct_consumption": self.ev_pct_consumption,
            "cv_pct_consumption": self.cv_pct_consumption,
            "utility": self.utility, "base_utility": self.base_utility,
            "cost_index": self.cost_index, "price_index": self.price_index,
            "budget": self.budget, "base_budget": self.base_budget,
        }, index=_index(self.country_codes, np.size(self.ev), "C", "country"))

    def summary(self) -> str:
        """One-line EV/CV summary; positive values denote gains."""
        return (f"Household welfare [{self.metadata.get('rule', '?')}] in "
                f"{self.metadata.get('unit', 'budget value units')}: EV "
                f"{np.round(self.ev, 6).tolist()} ({np.round(self.ev_pct_consumption, 4).tolist()}% "
                f"of base budget), CV {np.round(self.cv, 6).tolist()}; positive values denote gains.")


@dataclass(frozen=True)
class SupernumeraryFitResult(_TableExports):
    """Regional weighted-least-squares fit of the LES supernumerary share.

    ``fitted`` is the constrained estimate, ``unconstrained`` the unclipped
    optimum, ``predicted`` and ``residual`` the implied compensated own-price
    elasticities and their gap to the targets (``(sector, region)``), and the
    RMSE/maximum statistics use the normalized fit weights (zero-weight
    categories are excluded). ``metadata["qualification"]`` carries the IO
    wording: a conditional projection of supplied elasticities onto LES that
    need not reproduce external price responses or identify subsistence.
    """

    fitted: np.ndarray
    unconstrained: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    clipped: np.ndarray
    lower_bound_hit: np.ndarray
    upper_bound_hit: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    weighted_rmse: np.ndarray
    weighted_rmse_unconstrained: np.ndarray
    max_abs_residual: np.ndarray
    normalized_weights: np.ndarray
    input_weighted_expenditure_elasticity_mean: np.ndarray
    sector_codes: tuple[str, ...] = ()
    country_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Per-region fitted share, bounds, clipping flags and fit errors."""
        return pd.DataFrame({
            "fitted": self.fitted, "unconstrained": self.unconstrained,
            "lower_bound": self.lower_bound, "upper_bound": self.upper_bound,
            "clipped": self.clipped, "weighted_rmse": self.weighted_rmse,
            "weighted_rmse_unconstrained": self.weighted_rmse_unconstrained,
            "max_abs_residual": self.max_abs_residual,
        }, index=_index(self.country_codes, self.fitted.size, "C", "country"))

    def summary(self) -> str:
        """One-line fit summary: fitted shares, clipped regions and weighted RMSE."""
        return (f"Supernumerary share fit: fitted {np.round(self.fitted, 6).tolist()}, "
                f"{int(np.count_nonzero(self.clipped))} of {self.fitted.size} regions clipped, "
                f"weighted RMSE {np.round(self.weighted_rmse, 6).tolist()}.")

    def as_dict(self) -> dict[str, Any]:
        """The IO audit dictionary (lists, not arrays)."""
        return {
            "method": self.metadata.get("method"),
            "formula": self.metadata.get("formula"),
            "weight_source": self.metadata.get("weight_source"),
            "normalized_fit_weights": self.normalized_weights.tolist(),
            "zero_weight_targets": self.metadata.get("zero_weight_targets"),
            "input_weighted_expenditure_elasticity_mean": self.input_weighted_expenditure_elasticity_mean.tolist(),
            "unconstrained_supernumerary_share": self.unconstrained.tolist(),
            "fitted_supernumerary_share": self.fitted.tolist(),
            "lower_bound": self.lower_bound.tolist(),
            "upper_bound": self.upper_bound.tolist(),
            "lower_bound_hit": self.lower_bound_hit.tolist(),
            "upper_bound_hit": self.upper_bound_hit.tolist(),
            "clipped": self.clipped.tolist(),
            "predicted_compensated_own_price": self.predicted.tolist(),
            "compensated_own_price_residual": self.residual.tolist(),
            "weighted_root_mean_square_residual": self.weighted_rmse.tolist(),
            "weighted_root_mean_square_unconstrained_residual": self.weighted_rmse_unconstrained.tolist(),
            "maximum_absolute_residual": self.max_abs_residual.tolist(),
            "qualification": self.metadata.get("qualification"),
        }


# ---------------------------------------------------------------------------
# preferences
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HouseholdPreferences(_TableExports):
    """Calibrated benchmark-normalized household demand system.

    Construct with :func:`calibrate_household`. Arrays are read-only:
    ``x0`` benchmark spending (and quantity), ``m0`` budgets, ``shares``,
    ``beta`` marginal budget shares, ``gamma`` subsistence quantities,
    ``surplus0 = supernumerary_share * m0``, ``eta`` repaired expenditure
    elasticities (zero on structural zeros). Homothetic rules carry
    ``beta = shares``, ``gamma = 0``, ``supernumerary_share = 1``.

    The evaluation methods accept ``prices`` of shape ``(sector, region)`` and a
    scalar or ``(region,)`` ``budget``/``utility``; they keep complex inputs and
    never clip. ``validate=True`` (or :meth:`validate_domain`) rejects
    nonpositive prices, nonfinite inputs and exhausted surplus.
    """

    rule: str
    x0: np.ndarray
    m0: np.ndarray
    shares: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    surplus0: np.ndarray
    eta: np.ndarray
    supernumerary_share: np.ndarray
    ces_elasticity: float
    calibration: HouseholdCalibrationResult
    sector_codes: tuple[str, ...] = ()
    country_codes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_sectors(self) -> int:
        """Number of sector composites (rows of ``x0``)."""
        return int(self.x0.shape[0])

    @property
    def n_countries(self) -> int:
        """Number of regions (columns of ``x0``)."""
        return int(self.x0.shape[1])

    # -- input coercion ------------------------------------------------------
    def _prices(self, prices: Any) -> np.ndarray:
        p = np.asarray(prices)
        if p.shape != self.x0.shape:
            raise ValueError(f"prices must have shape {self.x0.shape}, got {p.shape}")
        return p

    def _regions(self, value: Any, name: str) -> np.ndarray:
        result = np.asarray(value)
        if result.ndim == 0:
            return np.broadcast_to(result, self.m0.shape)
        if result.shape != self.m0.shape:
            raise ValueError(f"{name} must be scalar or shape {self.m0.shape}, got {result.shape}")
        return result

    # -- domain --------------------------------------------------------------
    def validate_domain(self, prices: Any, budget: Any) -> None:
        """Reject nonpositive prices, nonfinite inputs, and exhausted surplus.

        Strictly positive surplus is required for an interior demand derivative
        and positive normalized utility; numerical extrapolation below the
        subsistence requirement is not an economically feasible state.
        """
        p = _real_finite(self._prices(prices), "prices")
        m = _real_finite(self._regions(budget, "budget"), "budget")
        if np.any(p <= 0):
            raise HouseholdDomainError("all sector composite prices must be strictly positive")
        surplus = m - np.sum(p * self.gamma, axis=0)
        if np.any(surplus <= 0):
            invalid = np.flatnonzero(surplus <= 0).tolist()
            raise HouseholdDomainError(f"household budget must exceed subsistence cost in regions {invalid}")

    def _validate_utility(self, p: np.ndarray, u: np.ndarray) -> None:
        _real_finite(p, "prices")
        _real_finite(u, "utility")
        if np.any(p <= 0) or np.any(u <= 0):
            raise HouseholdDomainError("prices and normalized utility must be strictly positive")

    # -- kernels -------------------------------------------------------------
    def price_index(self, prices: Any, *, validate: bool = False) -> np.ndarray:
        """Unit index ``P(p)`` for discretionary utility; see :meth:`cost_index` for ``e(p, 1) / m0``."""
        p = self._prices(prices)
        if validate:
            self._validate_utility(p, np.ones_like(self.m0))
        if self.rule == "fixed_baskets":
            return np.sum(self.shares * p, axis=0)
        if self.rule == "ces" and self.ces_elasticity != 1.0:
            power = 1.0 - self.ces_elasticity
            # The expm1/log1p form avoids cancellation close to sigma=1.
            return np.exp(np.log1p(np.sum(self.shares * np.expm1(power * np.log(p)), axis=0)) / power)
        return np.exp(np.sum(self.beta * np.log(p), axis=0))

    def expenditure(self, prices: Any, utility: Any, *, validate: bool = False) -> np.ndarray:
        """Minimum expenditure ``e(p, u)`` attaining utility normalized to one at benchmark."""
        p, u = self._prices(prices), self._regions(utility, "utility")
        if validate:
            self._validate_utility(p, u)
        return np.sum(p * self.gamma, axis=0) + self.surplus0 * self.price_index(p) * u

    def cost_index(self, prices: Any, *, validate: bool = False) -> np.ndarray:
        """Exact cost-of-living index at benchmark utility, ``e(p, 1) / m0``, including subsistence."""
        return self.expenditure(prices, 1.0, validate=validate) / self.m0

    def evaluate(self, prices: Any, budget: Any, *, validate: bool = False) -> HouseholdDemandResult:
        """Marshallian quantities, indices, surplus and utility at ``(p, m)``."""
        p, m = self._prices(prices), self._regions(budget, "budget")
        if validate:
            self.validate_domain(p, m)
        subsistence_cost = np.sum(p * self.gamma, axis=0)
        surplus = m - subsistence_cost
        index = self.price_index(p)
        if self.rule == "fixed_baskets":
            q = self.x0 * (m / (self.m0 * index))
        elif self.rule == "ces" and self.ces_elasticity != 1.0:
            sigma = self.ces_elasticity
            q = self.shares * m * np.exp(-sigma * np.log(p) + (sigma - 1.0) * np.log(index))
        else:
            q = self.gamma + self.beta * surplus / p
        utility = surplus / (self.surplus0 * index)
        cost_index = (subsistence_cost + self.surplus0 * index) / self.m0
        return HouseholdDemandResult(
            _frozen(q), _frozen(index), _frozen(cost_index), _frozen(surplus), _frozen(utility),
            _frozen(m), self.sector_codes, self.country_codes, {"rule": self.rule})

    def demand(self, prices: Any, budget: Any, *, validate: bool = False) -> np.ndarray:
        """Marshallian composite quantities, shape ``(sector, region)`` (a writable copy)."""
        return self.evaluate(prices, budget, validate=validate).quantities.copy()

    def hicksian(self, prices: Any, utility: Any, *, validate: bool = False) -> np.ndarray:
        """Compensated quantities ``h(p, u)`` (a writable copy), the gradient of expenditure in prices."""
        p, u = self._prices(prices), self._regions(utility, "utility")
        if validate:
            self._validate_utility(p, u)
        return self.demand(p, self.expenditure(p, u))

    def slutsky(self, prices: Any, budget: Any, *, validate: bool = False) -> np.ndarray:
        """Analytic substitution matrices ``S[i, j, k] = dh_i/dp_j`` at ``u(p, m)``, shape ``(sector, sector, region)``.

        Stone-Geary and Cobb-Douglas: ``S_ij = surplus * (beta_i beta_j / (p_i p_j) - delta_ij beta_i / p_i**2)``.
        CES: ``S_ij = sigma * (h_i h_j / m - delta_ij h_i / p_i)``. Fixed baskets: zero.
        Every matrix is symmetric, negative semidefinite and satisfies ``S @ p = 0``.
        """
        p, m = self._prices(prices), self._regions(budget, "budget")
        state = self.evaluate(p, m, validate=validate)
        ns, nc = self.x0.shape
        if self.rule == "fixed_baskets":
            return np.zeros((ns, ns, nc))
        eye = np.eye(ns)[:, :, None]
        if self.rule == "ces" and self.ces_elasticity != 1.0:
            h = state.quantities
            outer = h[:, None, :] * h[None, :, :] / np.asarray(m)[None, None, :]
            return self.ces_elasticity * (outer - eye * (h / p)[:, None, :])
        b = self.beta
        outer = b[:, None, :] * b[None, :, :] / (p[:, None, :] * p[None, :, :])
        return state.surplus[None, None, :] * (outer - eye * (b / p**2)[:, None, :])

    def welfare(self, prices: Any, budget: Any, *, validate: bool = False) -> HouseholdWelfareResult:
        """Exact EV/CV against the calibration benchmark (``p = 1``, ``m = m0``)."""
        return compute_household_welfare(self, prices, budget, validate=validate)

    # -- reporting -----------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Long table indexed by (sector, country): x0, share, beta, gamma, eta, active."""
        idx = pd.MultiIndex.from_product([list(_index(self.sector_codes, self.n_sectors, "S", "sector")),
                                          list(_index(self.country_codes, self.n_countries, "C", "country"))],
                                         names=["sector", "country"])
        return pd.DataFrame({
            "x0": self.x0.ravel(), "share": self.shares.ravel(), "beta": self.beta.ravel(),
            "gamma": self.gamma.ravel(), "eta": self.eta.ravel(), "active": (self.x0 > 0).ravel(),
        }, index=idx)

    def summary(self) -> str:
        """One-line description of the rule, dimensions, supernumerary shares and structural zeros."""
        extra = f", CES elasticity {self.ces_elasticity:g}" if self.rule == "ces" else ""
        return (f"Household preferences [{self.rule}]: {self.n_sectors} sectors x {self.n_countries} regions, "
                f"supernumerary shares {np.round(self.supernumerary_share, 4).tolist()}, "
                f"{self.calibration.structural_zero_count} structural zeros{extra}. "
                f"Benchmark prices and utility are one.")


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def calibrate_household(
    expenditure0: Any,
    rule: Rule = "stone_geary",
    *,
    expenditure_elasticities: Any | None = None,
    supernumerary_share: Any | None = None,
    ces_elasticity: float = 1.0,
    repair_adding_up: bool = True,
    sector_codes: Sequence[str] | None = None,
    country_codes: Sequence[str] | None = None,
) -> HouseholdPreferences:
    """Calibrate benchmark household demand without creating missing sectors.

    ``expenditure0`` is the ``(sector, region)`` benchmark purchaser spending
    matrix (nonnegative, every region positive). Stone-Geary requires positive
    local expenditure elasticity targets ``eta`` on observed sectors. By default
    ``eta`` is divided by its expenditure-weighted regional mean to impose Engel
    adding-up; original means, corrections and repaired targets are recorded in
    ``.calibration``. Set ``repair_adding_up=False`` to reject inconsistent
    targets (tolerance 1e-10). No substitute targets are inferred from IO
    expenditures alone: omitted targets explicitly mean unit elasticities.

    A supplied regional surplus fraction ``lambda`` (scalar or ``(region,)``)
    identifies subsistence through ``beta = w * eta``, ``gamma = x0 * (1 - lambda * eta)``.
    Nonnegative subsistence requires ``0 < lambda <= 1 / max(eta)`` in every
    region (checked with an 8 ulp allowance; roundoff is removed only at that
    boundary). Inactive sector targets are ignored and reported as structural
    zeros. The IO benchmark and ``eta`` do not identify ``lambda``: ``None`` (the
    default) uses the assumed value 0.5 and records
    ``metadata["supernumerary_share_default_is_assumption"] = True``; a supplied
    share records ``False`` and its external provenance belongs in run metadata.
    Cobb-Douglas sets ``lambda = 1``, ``eta = 1``, ``gamma = 0``. CES
    and fixed baskets are homothetic and accept no nonunit elasticity targets;
    ``ces_elasticity`` must be finite and nonnegative.
    """
    if rule not in _RULES:
        raise ValueError(f"unknown preference rule {rule!r}; expected one of {_RULES}")
    x0 = np.asarray(_real_finite(expenditure0, "benchmark expenditure"), dtype=float)
    if x0.ndim != 2 or min(x0.shape) == 0:
        raise ValueError("benchmark expenditure must be a nonempty (sector, region) matrix")
    if np.any(x0 < 0):
        raise ValueError("benchmark household expenditure cannot be negative")
    m0 = x0.sum(axis=0)
    if np.any(m0 <= 0):
        raise ValueError("every region must have positive benchmark household expenditure")
    ns, nc = x0.shape
    sectors = _labels(sector_codes, ns, "S", "sector_codes")
    countries = _labels(country_codes, nc, "C", "country_codes")
    active = x0 > 0
    shares = x0 / m0
    supplied = expenditure_elasticities is not None
    raw_eta = np.ones_like(x0) if not supplied else np.asarray(expenditure_elasticities)
    if raw_eta.shape != x0.shape:
        raise ValueError(f"expenditure_elasticities must have shape {x0.shape}")
    if np.iscomplexobj(raw_eta) or not np.all(np.isfinite(raw_eta[active])):
        raise ValueError("expenditure elasticity targets on observed sectors must be finite and real")
    if np.any(raw_eta[active] <= 0):
        raise ValueError("Stone-Geary interior calibration needs positive expenditure elasticities on observed sectors")
    # Unobserved goods remain structural zeros, including unspecified NaN targets.
    raw_eta = np.where(active, raw_eta, 0.0).astype(float)
    mean = np.sum(shares * raw_eta, axis=0)
    eta = raw_eta / mean
    adding_up_error = float(np.max(np.abs(mean - 1.0)))
    if not repair_adding_up and adding_up_error > 1e-10:
        raise ValueError("expenditure elasticity targets violate expenditure-weighted adding-up")
    if rule != "stone_geary":
        if supplied and np.any(np.abs(raw_eta[active] - 1.0) > 1e-10):
            raise ValueError("homothetic preferences require unit expenditure elasticity targets")
        eta = active.astype(float)
    sigma = float(ces_elasticity)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("CES elasticity must be finite and nonnegative")
    maximum = 1.0 / np.max(eta, axis=0)
    lam_default = supernumerary_share is None
    if rule == "stone_geary":
        lam = np.asarray(_real_finite(0.5 if lam_default else supernumerary_share, "supernumerary_share"),
                         dtype=float)
        if lam.ndim == 0:
            lam = np.full(m0.shape, lam)
        if lam.shape != m0.shape:
            raise ValueError(f"supernumerary_share must be scalar or shape {m0.shape}")
        if np.any(lam <= 0) or np.any(lam > maximum * (1 + 8 * np.finfo(float).eps)):
            raise ValueError("supernumerary_share must satisfy 0 < lambda <= 1/max(eta) in each region; "
                             f"maximum admissible shares are {maximum.tolist()}")
        # Remove roundoff only at the explicitly checked admissibility boundary.
        lam = np.minimum(lam, maximum)
        gamma = x0 * np.maximum(1.0 - lam * eta, 0.0)
        beta = shares * eta
    else:
        lam = np.ones_like(m0)
        gamma = np.zeros_like(x0)
        beta = shares.copy()
    surplus0 = lam * m0
    report = HouseholdCalibrationResult(
        rule=rule,
        target_source="provided" if supplied else "unit expenditure elasticities explicitly assumed",
        benchmark_price_normalization="one for each sector-region composite",
        structural_zero_count=int(np.count_nonzero(~active)),
        adding_up_repair="divide each region's targets by their benchmark-expenditure-weighted mean",
        input_weighted_elasticity_mean=_readonly(mean),
        input_adding_up_max_error=adding_up_error,
        maximum_absolute_elasticity_adjustment=float(np.max(np.abs(eta - raw_eta))),
        repaired_expenditure_elasticities=_readonly(eta),
        supernumerary_share=_readonly(lam),
        maximum_admissible_supernumerary_share=_readonly(maximum),
        subsistence_spending_share=_readonly(gamma.sum(axis=0) / m0),
        ces_elasticity=sigma,
        identification=("Stone-Geary surplus shares are supplied assumptions or external estimates, "
                        "not identified by IO accounts and Engel targets"),
        sector_codes=sectors, country_codes=countries,
    )
    return HouseholdPreferences(
        rule, _readonly(x0), _readonly(m0), _readonly(shares), _readonly(beta), _readonly(gamma),
        _readonly(surplus0), _readonly(eta), _readonly(lam), sigma, report, sectors, countries,
        {"source": "calibrate_household", "supernumerary_share_default_is_assumption": rule == "stone_geary" and lam_default})


def fit_supernumerary_share(
    expenditure0: Any,
    expenditure_elasticities: Any,
    compensated_own_price: Any,
    *,
    weights: Any | None = None,
    minimum_relative_share: float = 1e-6,
    sector_codes: Sequence[str] | None = None,
    country_codes: Sequence[str] | None = None,
) -> SupernumeraryFitResult:
    """Fit regional Stone-Geary surplus shares to compensated own-price targets.

    At the benchmark the LES Hicksian own-price elasticity of observed good i is
    ``-lambda * eta_i * (1 - beta_i)`` with ``beta_i = w_i * eta_i``. Conditional
    on normalized positive Engel targets, weighted least squares therefore
    identifies one scalar per region. Default weights are benchmark expenditure
    shares; custom weights must be nonnegative ``(sector, region)`` arrays.
    Targets are dimensionless elasticities (nonpositive on observed sectors),
    not price derivatives. Structural zero sectors are excluded.

    This is a projection of an external elasticity system onto LES, not an
    exact match to that system. Both unconstrained and constrained fits, bound
    hits and sector residuals are reported. The upper bound ``1 / max(eta)``
    ensures ``gamma >= 0``; the lower bound ``minimum_relative_share / max(eta)``
    keeps the calibration interior if the external targets would imply zero
    discretionary spending. A region with no weighted price response cannot
    identify ``lambda`` and raises ``ValueError``.
    """
    base = calibrate_household(expenditure0, "cobb_douglas", sector_codes=sector_codes,
                               country_codes=country_codes)
    active = base.x0 > 0
    raw_eta = np.asarray(expenditure_elasticities)
    target = np.asarray(compensated_own_price)
    if raw_eta.shape != base.x0.shape or target.shape != base.x0.shape:
        raise ValueError(f"elasticity arrays must have shape {base.x0.shape}")
    if np.iscomplexobj(raw_eta) or not np.all(np.isfinite(raw_eta[active])) or np.any(raw_eta[active] <= 0):
        raise ValueError("expenditure elasticities on observed sectors must be finite, real and positive")
    if np.iscomplexobj(target) or not np.all(np.isfinite(target[active])) or np.any(target[active] > 0):
        raise ValueError("compensated own-price targets on observed sectors must be finite, real and nonpositive")
    raw_eta = np.where(active, raw_eta, 0.).astype(float)
    mean_eta = np.sum(base.shares * raw_eta, axis=0)
    eta = raw_eta / mean_eta
    target = np.where(active, target, 0.).astype(float)
    if weights is None:
        fit_weights = base.shares
    else:
        fit_weights = np.asarray(weights)
        if fit_weights.shape != base.x0.shape or np.iscomplexobj(fit_weights):
            raise ValueError(f"weights must be real with shape {base.x0.shape}")
        if not np.all(np.isfinite(fit_weights[active])) or np.any(fit_weights[active] < 0):
            raise ValueError("weights on observed sectors must be finite and nonnegative")
        fit_weights = np.where(active, fit_weights, 0.).astype(float)
    weight_totals = fit_weights.sum(axis=0)
    if np.any(weight_totals <= 0):
        raise ValueError("each region needs at least one positive fit weight")
    fit_weights = fit_weights / weight_totals
    beta = base.shares * eta
    slope = eta * (1.0 - beta)
    denominator = np.sum(fit_weights * slope**2, axis=0)
    if np.any(denominator <= 1e-28):
        missing = np.flatnonzero(denominator <= 1e-28).tolist()
        raise ValueError(f"compensated own-price targets do not identify surplus shares in regions {missing}")
    unconstrained = -np.sum(fit_weights * slope * target, axis=0) / denominator
    if not np.isfinite(minimum_relative_share) or not 0 < minimum_relative_share < 1:
        raise ValueError("minimum_relative_share must be strictly between zero and one")
    upper = 1.0 / eta.max(axis=0)
    lower = minimum_relative_share * upper
    fitted = np.clip(unconstrained, lower, upper)
    predicted = -slope * fitted
    residual = predicted - target
    return SupernumeraryFitResult(
        fitted=_readonly(fitted), unconstrained=_readonly(unconstrained),
        lower_bound=_readonly(lower), upper_bound=_readonly(upper),
        clipped=_readonly(fitted != unconstrained, bool),
        lower_bound_hit=_readonly(unconstrained <= lower, bool),
        upper_bound_hit=_readonly(unconstrained >= upper, bool),
        predicted=_readonly(predicted), residual=_readonly(residual),
        weighted_rmse=_readonly(np.sqrt(np.sum(fit_weights * residual**2, axis=0))),
        weighted_rmse_unconstrained=_readonly(np.sqrt(np.sum(fit_weights * (-slope * unconstrained - target)**2, axis=0))),
        max_abs_residual=_readonly(np.max(np.where(fit_weights > 0, np.abs(residual), 0.), axis=0)),
        normalized_weights=_readonly(fit_weights),
        input_weighted_expenditure_elasticity_mean=_readonly(mean_eta),
        sector_codes=base.sector_codes, country_codes=base.country_codes,
        metadata={
            "method": "regional constrained weighted least squares of LES Hicksian own-price elasticities",
            "formula": "hicksian_own_i = -lambda * eta_i * (1-beta_i), beta_i = w_i*eta_i",
            "weight_source": ("benchmark expenditure shares" if weights is None
                              else "provided nonnegative weights, normalized regionally"),
            "zero_weight_targets": ("Excluded from the fit and reported fit-error statistics; "
                                    "their numerical target values can be placeholders."),
            "qualification": ("Conditional projection of supplied elasticities onto LES; constrained fit "
                              "need not reproduce external price responses or identify subsistence empirically"),
        })


# ---------------------------------------------------------------------------
# welfare
# ---------------------------------------------------------------------------

def compute_household_welfare(
    preferences: HouseholdPreferences,
    prices: Any,
    budget: Any,
    *,
    base_prices: Any | None = None,
    base_budget: Any | None = None,
    validate: bool = True,
) -> HouseholdWelfareResult:
    """Exact EV/CV of moving from a base state to ``(prices, budget)``.

    With both ``base_prices`` and ``base_budget`` omitted the base is the
    calibration benchmark (``p0 = 1``, ``m0``) and the IO closed forms are used:
    ``EV = surplus0 * (u1 - 1)`` (valid for every rule, including fixed baskets)
    and ``CV = m1 - e(p1, 1)``. Otherwise ``u0 = u(p0, m0)`` is evaluated and
    ``EV = e(p0, u1) - e(p0, u0)``, ``CV = e(p1, u1) - e(p1, u0)``. Both are
    positive for gains; percentages divide by ``e(p0, u0)``. ``validate=True``
    checks both states against the preference domain.
    """
    if not isinstance(preferences, HouseholdPreferences):
        raise TypeError("preferences must be a HouseholdPreferences instance")
    if (base_prices is None) != (base_budget is None):
        raise ValueError("base_prices and base_budget must be supplied together")
    prefs = preferences
    p1, m1 = prefs._prices(prices), prefs._regions(budget, "budget")
    state = prefs.evaluate(p1, m1, validate=validate)
    if base_prices is None:
        # e(p0,u1)-m0 simplifies to surplus0*(u1-1), including fixed baskets.
        ev = prefs.surplus0 * (state.utility - 1.0)
        cv = m1 - prefs.expenditure(p1, 1.0)
        base_utility = np.ones_like(prefs.m0)
        base_expense = prefs.m0
        base_budget_out = prefs.m0
        base = "calibration benchmark (unit prices, m0)"
    else:
        p0, m0b = prefs._prices(base_prices), prefs._regions(base_budget, "base_budget")
        base_state = prefs.evaluate(p0, m0b, validate=validate)
        base_utility = base_state.utility
        base_expense = prefs.expenditure(p0, base_utility)
        ev = prefs.expenditure(p0, state.utility) - base_expense
        cv = prefs.expenditure(p1, state.utility) - prefs.expenditure(p1, base_utility)
        base_budget_out = np.array(np.broadcast_to(m0b, prefs.m0.shape))
        base = "supplied base state"
    if validate:
        values = np.concatenate([np.ravel(ev), np.ravel(cv), np.ravel(base_expense)])
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)) or np.any(np.real(base_expense) <= 0):
            raise HouseholdDomainError("welfare requires finite real states with positive base expenditure")
    return HouseholdWelfareResult(
        ev=_frozen(ev), cv=_frozen(cv),
        ev_pct_consumption=_frozen(100 * ev / base_expense), cv_pct_consumption=_frozen(100 * cv / base_expense),
        utility=_frozen(state.utility), base_utility=_frozen(base_utility),
        cost_index=_frozen(state.cost_index), price_index=_frozen(state.price_index),
        budget=_frozen(m1), base_budget=_frozen(base_budget_out),
        country_codes=prefs.country_codes,
        metadata={"rule": prefs.rule, "base": base, "positive_is_gain": True,
                  "unit": "household budget value units",
                  "percent_denominator": "base household budget e(p0, u0)",
                  "definition": ("EV = e(p0,u1) - e(p0,u0); CV = e(p1,u1) - e(p1,u0); static comparison of "
                                 "the household composite basket, not lifetime welfare")})


# ---------------------------------------------------------------------------
# bridges to puremacro trade calibrations and equilibria
# ---------------------------------------------------------------------------

def _category(calib: TradeCalibrationResult, category: Any) -> int:
    if isinstance(category, bool) or not isinstance(category, Integral):
        raise TypeError("category must be an integer final-use index")
    k = int(category)
    nfd = calib.n_final_demand
    if not 0 <= k < nfd:
        raise ValueError(f"category {k} is out of range for {nfd} final-use categories")
    if nfd >= 2 and k == 1:
        raise ValueError("Investment category 1 subtracts foreign saving and does not follow a consumption "
                         "budget; it cannot be treated as a household")
    if nfd == 1 and np.any(np.asarray(calib.invforT) != 0):
        raise NotImplementedError("Single-category calibrations require zero foreign saving")
    return k


def _sector_weights(calib: TradeCalibrationResult, k: int) -> np.ndarray:
    """Sector shares of category ``k``'s Leontief origin basket, shape ``(ns, nc)``."""
    nc, ns = calib.n_countries, calib.n_sectors
    afd = np.asarray(calib.afd, dtype=float)
    if afd.shape != (nc * ns, calib.n_final_demand, nc) or not np.isfinite(afd).all():
        raise ValueError("calibration afd must be a finite (ns*nc, nfd, nc) array")
    return afd[:, k, :].reshape(nc, ns, nc).sum(axis=0)


def household_expenditure_from_calibration(calib: TradeCalibrationResult, category: int = 0) -> np.ndarray:
    """Benchmark purchaser spending by sector, shape ``(ns, nc)``, for one final-use category.

    ``x0[s, c] = A[s, c] * theta[0, k, c] * (l_endow + k_endow + T)[c]`` where
    ``A[s, c]`` is the share of sector ``s`` in the category's Leontief origin
    basket (``sum_i afd[i, s, k, c]``) and ``T`` the baseline transfers
    (``calib.T``, else ``TT + TTfd``). Purchaser spending includes the final-use
    tax, which is a uniform share of the category in both accounting modes, so
    sector shares of basic-price and purchaser spending coincide. The result is
    the ``x0`` expected by :func:`calibrate_household`; pass
    ``calib.sector_codes``/``calib.country_codes`` for labels.

    Category 1 (investment) is rejected: its spending subtracts foreign saving
    and is not a consumption budget. Negative cells raise ``ValueError`` naming
    them (the bundled 77x11 table has negative investment cells but a clean
    category 0). With the bundled OECD table, category 0 is HFCE + NPISH + GGFC,
    so the "household" values aggregate consumption including government final
    consumption.
    """
    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError("calib must be a TradeCalibrationResult")
    k = _category(calib, category)
    nc = calib.n_countries
    if calib.theta is None:
        raise ValueError("calibration lacks final-use expenditure shares (theta)")
    theta = np.asarray(calib.theta, dtype=float).reshape(calib.n_final_demand, nc)
    if calib.T is not None:
        transfers = np.asarray(calib.T, dtype=float).ravel()
    elif calib.TT is not None and calib.TTfd is not None:
        transfers = np.asarray(calib.TT, dtype=float).ravel() + np.asarray(calib.TTfd, dtype=float).ravel()
    else:
        raise ValueError("calibration lacks baseline transfers (T or TT and TTfd)")
    income0 = (np.asarray(calib.l_endow, dtype=float).ravel() + np.asarray(calib.k_endow, dtype=float).ravel()
               + transfers)
    if income0.shape != (nc,) or not np.isfinite(income0).all():
        raise ValueError("baseline income must be a finite (nc,) vector")
    spending = theta[k] * income0
    weight = _sector_weights(calib, k)
    if np.any(spending < 0):
        bad = [calib.country_codes[i] if calib.country_codes else i for i in np.flatnonzero(spending < 0)]
        raise ValueError(f"benchmark spending on category {k} is negative in {bad}; use a nonnegative "
                         "consumption category or aggregate the table")
    if np.any(weight < 0):
        cells = np.argwhere(weight < 0).tolist()
        raise ValueError(f"category {k} has negative sector basket weights at (sector, country) {cells}")
    x0 = weight * spending[None, :]
    if np.any(x0.sum(axis=0) <= 0):
        raise ValueError(f"every country must have positive benchmark spending on category {k}")
    return x0


def _sector_composites(calib: TradeCalibrationResult, pv: np.ndarray, tf: np.ndarray, k: int) -> np.ndarray:
    """Relative purchaser prices of sector composites, ``sum_i afd_is p_i tf_is / sum_i afd_is``."""
    nc, ns = calib.n_countries, calib.n_sectors
    afd = np.asarray(calib.afd, dtype=float)[:, k, :].reshape(nc, ns, nc)
    duty = np.asarray(tf, dtype=float)[:, k, :].reshape(nc, ns, nc)
    cost = np.einsum("isc,is->sc", afd * duty, pv.reshape(nc, ns))
    weight = afd.sum(axis=0)
    return np.divide(cost, weight, out=np.ones_like(cost), where=weight > 0)




def _restated(exc: Exception, message: str) -> Exception:
    """The same exception class with a clearer message (plain ``ValueError`` for exotic subclasses)."""
    if type(exc) in (ValueError, NotImplementedError, HouseholdDomainError):
        return type(exc)(message)
    return (NotImplementedError if isinstance(exc, NotImplementedError) else ValueError)(message)


def _consistent_block(result: TradeEquilibriumResult, calib: TradeCalibrationResult, tol: float) -> dict[str, Any]:
    from .welfare import _checked_state
    try:
        block, _ = _checked_state(result, calib, tol)
    except (ValueError, NotImplementedError) as exc:
        raise _restated(exc, f"{exc} (consistent-accounting audit shared with compute_hicksian_welfare)") from exc
    return block


def _nonuniform_destinations(tf_k: np.ndarray, afd_k: np.ndarray, ns: int, nc: int, tol: float) -> list[int]:
    """Destinations whose foreign final-demand multipliers vary across active origin cells.

    ``p_fd`` records one aggregate per (category, destination), so it identifies one
    foreign multiplier per destination and nothing finer.
    """
    origin = np.repeat(np.arange(nc), ns)
    varying = []
    for c in range(nc):
        cells = (origin != c) & (afd_k[:, c] > 0)
        if cells.any():
            values = tf_k[cells, c]
            if np.ptp(values) > tol * max(1., float(np.max(np.abs(values)))):
                varying.append(c)
    return varying


def _legacy_block(result: TradeEquilibriumResult, calib: TradeCalibrationResult, k: int,
                  tau_fd: Any | None, tol: float, trust_tau_fd_detail: bool) -> dict[str, Any]:
    from .equilibrium import unpack_equilibrium_vector
    from .solver import _resolve_tariffs
    meta = result.metadata
    if not result.converged:
        raise ValueError("household prices require converged equilibria")
    if meta.get("fiscal_closure", "lump_sum") not in ("lump_sum", "baseline", "") or meta.get("recycling_params"):
        raise NotImplementedError("legacy household budgets are available for lump-sum fiscal closures only")
    nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand
    n = nc * ns
    # Legacy results record neither the intermediate nor the final tariff schedule, so their
    # residuals cannot be re-evaluated here: the recorded residuals must meet the solve's tolerance.
    solve_tol = float(meta.get("tol", 1e-8))
    recorded_residuals = [np.nan if result.max_residual is None else float(result.max_residual)]
    if result.residuals is not None and np.size(result.residuals):
        recorded_residuals.append(float(np.max(np.abs(np.asarray(result.residuals, dtype=float)))))
    worst = float(np.max(recorded_residuals))
    if not (np.isfinite(solve_tol) and solve_tol > 0) or not np.isfinite(worst) or worst > solve_tol:
        raise ValueError(f"legacy result max_residual {worst:.3e} exceeds its solve tolerance {solve_tol:.3e}")
    try:
        state = unpack_equilibrium_vector(result.x_sol, ns=ns, nc=nc, nfd=nfd)
    except ValueError as exc:
        raise ValueError(f"legacy result state vector does not match the calibration: {exc}") from exc
    # The legacy evaluator builds p_fd, c_fd and the final-demand flows from the state prices in
    # x_sol; p_sol is the zero-profit price, which equals them only up to the solve residual.
    pv = np.asarray(state.p, dtype=float).ravel(order="F")
    w, r, T = (np.asarray(v, dtype=float).ravel() for v in (state.w, state.r, state.T))
    if not np.isfinite(pv).all() or np.any(pv <= 0) or not all(np.isfinite(v).all() for v in (w, r, T)):
        raise ValueError("Result state prices, factor prices and transfers must be finite with positive prices")
    for actual, expected, name in ((result.w_sol, w, "w_sol"), (result.r_sol, r, "r_sol"), (result.T_sol, T, "T_sol")):
        if (actual is None or np.size(actual) != nc
                or not np.allclose(np.asarray(actual, dtype=float).ravel(), expected, rtol=tol, atol=tol)):
            raise ValueError(f"Result field {name} disagrees with the state vector x_sol")
    p_sol = np.asarray(result.p_sol, dtype=float)
    slack = solve_tol + 64 * np.finfo(float).eps * max(1., float(np.max(pv)))
    if (p_sol.size != n or not np.isfinite(p_sol).all()
            or np.max(np.abs(p_sol.reshape(1, ns, nc).ravel(order="F") - pv)) > slack):
        raise ValueError("Recorded zero-profit prices p_sol differ from the state prices in x_sol by more than "
                         "the solve tolerance")
    _, tf, _, _ = _resolve_tariffs(calib, None, None if tau_fd is None else np.asarray(tau_fd, dtype=float))
    if tf.shape != (n, nfd, nc) or not np.isfinite(tf).all() or np.any(tf <= 0):
        raise ValueError("Final-demand tariff multipliers must resolve to a finite positive (ns*nc, nfd, nc) array")
    for c in range(nc):
        if not np.allclose(tf[c * ns:(c + 1) * ns, :, c], 1., rtol=0, atol=1e-14):
            raise ValueError("Import tariffs must be one on domestic transactions")
    afd = np.asarray(calib.afd, dtype=float)
    ppfd = np.tensordot(pv, afd * tf, axes=(0, 0))
    if result.p_fd is None or np.shape(result.p_fd) != (1, nfd, nc):
        raise ValueError("Result lacks composite purchaser prices p_fd; solve it again")
    recorded = np.asarray(result.p_fd, dtype=float)[0]
    if not np.isfinite(recorded).all() or not np.allclose(recorded, ppfd, rtol=tol, atol=tol):
        raise ValueError("The supplied final-demand tariff multipliers do not reproduce the recorded composite "
                         "purchaser prices p_fd; pass the tau_fd used in the solve (None means free trade)")
    varying = _nonuniform_destinations(tf[:, k, :], afd[:, k, :], ns, nc, tol)
    if varying and not trust_tau_fd_detail:
        names = [calib.country_codes[c] if len(calib.country_codes) == nc else c for c in varying]
        raise ValueError(
            f"tau_fd varies across foreign origin cells of destination(s) {names} in category {k}. A legacy "
            "result records only the category-aggregate composite price p_fd, which cannot verify origin- or "
            "sector-specific detail (a different schedule with the same aggregates would pass). Solve with "
            "accounting='consistent', which records the schedule, or pass trust_tau_fd_detail=True to accept "
            "the supplied detail with only its aggregates verified")
    verification = ("legacy: national final-demand rates identified by the recorded p_fd" if not varying else
                    "legacy: category aggregates verified against p_fd; origin/sector detail of tau_fd supplied "
                    "by the caller (trust_tau_fd_detail=True)")
    if calib.theta is None:
        raise ValueError("calibration lacks final-use expenditure shares (theta)")
    income = (w * np.asarray(calib.l_endow, dtype=float).ravel() + r * np.asarray(calib.k_endow, dtype=float).ravel()
              + T)
    expenditure = np.asarray(calib.theta, dtype=float).reshape(nfd, nc) * income[None, :]
    if result.c_fd is not None and np.shape(result.c_fd) == (1, nfd, nc):
        absorbed = np.asarray(result.c_fd, dtype=float)[0, k] * recorded[k]
        if not np.allclose(absorbed, expenditure[k], rtol=tol, atol=tol * max(1., np.max(np.abs(expenditure)))):
            raise ValueError("Recorded absorption disagrees with theta * income for the selected category")
    return {"p": pv, "tf": tf, "expenditure": expenditure[None], "accounting": "legacy",
            "verification": verification}


def _check_consistent_tau_fd(calib: TradeCalibrationResult, tau_fd: Any, recorded: np.ndarray, tol: float) -> None:
    """Resolve a user schedule (national rates or multipliers) and cross-check it against the recorded one."""
    from .solver import _resolve_tariffs
    user = np.asarray(tau_fd, dtype=float)
    if user.shape != recorded.shape:
        _, user, _, _ = _resolve_tariffs(calib, None, user)
    if user.shape != recorded.shape:
        raise ValueError(f"tau_fd must be national ad-valorem rates of shape ({calib.n_countries},) or final-demand "
                         f"tariff multipliers of shape {recorded.shape}, got shape {np.shape(tau_fd)}")
    if not np.isfinite(user).all() or not np.allclose(user, recorded, rtol=tol, atol=tol):
        raise ValueError("tau_fd disagrees with the tariff schedule recorded on the consistent result")


def _household_prices(result: Any, calib: Any, *, category: Any, tau_fd: Any | None, tol: float,
                      trust_tau_fd_detail: bool, caller: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Prices, budgets and the schedule-verification label; errors are prefixed with ``caller``."""
    if not isinstance(result, TradeEquilibriumResult):
        raise TypeError(f"{caller}: result must be a TradeEquilibriumResult")
    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError(f"{caller}: calib must be a TradeCalibrationResult")
    try:
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        k = _category(calib, category)
        accounting = result.metadata.get("accounting", "legacy")
        if accounting == "consistent":
            block = _consistent_block(result, calib, tol)
            if tau_fd is not None:
                _check_consistent_tau_fd(calib, tau_fd, block["tf"], tol)
            pv = np.asarray(block["p"], dtype=float).ravel(order="F")
            tf = block["tf"]
            verification = "consistent: recorded schedule re-audited"
        elif accounting == "legacy":
            block = _legacy_block(result, calib, k, tau_fd, tol, bool(trust_tau_fd_detail))
            pv, tf, verification = block["p"], block["tf"], block["verification"]
        else:
            raise ValueError(f"unsupported accounting mode {accounting!r}")
        prices = _sector_composites(calib, pv, tf, k)
        budget = np.asarray(block["expenditure"], dtype=float)[0, k].copy()
        if not np.isfinite(prices).all() or np.any(prices <= 0) or not np.isfinite(budget).all():
            raise ValueError("Household composite prices must be finite and positive with finite budgets")
    except (ValueError, NotImplementedError) as exc:
        raise _restated(exc, f"{caller}: {exc}") from exc
    return prices, budget, verification


def household_prices_from_result(
    result: TradeEquilibriumResult,
    calib: TradeCalibrationResult,
    *,
    category: int = 0,
    tau_fd: Any | None = None,
    trust_tau_fd_detail: bool = False,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Sector-composite purchaser prices ``(ns, nc)`` and household budgets ``(nc,)`` from an equilibrium.

    Prices are relative to the calibration benchmark (one at ``p = 1`` and unit
    tariff multipliers): ``P[s, c] = sum_i afd[i, s, k, c] p_i tf[i, s, k, c] / sum_i afd[i, s, k, c]``,
    the tariff-inclusive Leontief origin basket of each sector composite, valued
    at the producer prices the equilibrium evaluator used. The final-use tax is
    a constant share of the category and cancels in relative prices.
    Structural-zero sectors get price one. The budget is the category's
    purchaser expenditure ``theta[k] * (w l + r k + T)`` (investment subtracts
    foreign saving and is rejected).

    ``accounting="consistent"`` results are re-audited exactly as
    :func:`~puremacro.trade.compute_hicksian_welfare` does (recorded tariff
    schedules, residuals re-evaluated from ``x_sol``, fields) and valued from
    ``_accounting.evaluate``: ``tf`` is the recorded ``final_tariff_multipliers``
    and the budget is ``block["expenditure"]``. ``tau_fd`` is optional there
    and, when given as the solve's national rates ``(nc,)`` or as a full
    multiplier array, it is resolved with the solver's conventions and
    cross-checked against the recorded schedule (a disagreement raises).

    Legacy results record no tariff schedules, so their residuals cannot be
    re-evaluated here: they are accepted only if the recorded residuals meet the
    solve's own ``metadata["tol"]`` (1e-8 when absent). They are valued at the
    state prices in ``x_sol``, from which the legacy evaluator built ``p_fd``,
    ``c_fd`` and the final-demand flows; the recorded zero-profit prices
    ``p_sol`` must match them within the solve tolerance and ``w_sol``,
    ``r_sol``, ``T_sol`` within ``tol``. ``tau_fd`` must be the final-demand
    tariffs used in the solve (``None`` is free trade) and is audited against
    the stored ``p_fd`` (``ppfd_k = sum_i afd_ik p_i tf_ik``); a mismatch
    raises. That aggregate identifies one foreign multiplier per destination,
    so national (origin- and sector-uniform) schedules are verified. A schedule
    that varies across the foreign origin cells of a destination in the
    selected category cannot be verified from ``p_fd``: a different schedule
    with the same aggregates would pass and change EV/CV for every rule except
    fixed baskets. It therefore raises unless ``trust_tau_fd_detail=True``, in
    which case only its category aggregates are verified; prefer
    ``accounting="consistent"`` results, which record the schedule.
    ``trust_tau_fd_detail`` has no effect on consistent results. Legacy budgets
    are available for lump-sum closures only.

    Flexible-model results are not supported here: compute Armington composite
    prices with :mod:`puremacro.trade.flexible` and call
    :func:`compute_household_welfare` directly.

    Tolerances: ``tol`` (default 1e-8) is the relative and absolute tolerance
    for comparing recorded result fields with their recomputed values (the
    consistent audit's field checks; the legacy ``p_fd``, ``c_fd``, ``w_sol``,
    ``r_sol`` and ``T_sol``), a supplied ``tau_fd`` with a recorded consistent
    schedule, and the spread of a legacy schedule across foreign origins
    (relative to its largest multiplier). Equilibrium residuals are checked at
    the solver's recorded ``metadata["tol"]`` instead, and domestic final-demand
    multipliers must equal one to 1e-14. Error messages name this function.
    """
    prices, budget, _ = _household_prices(result, calib, category=category, tau_fd=tau_fd, tol=tol,
                                          trust_tau_fd_detail=trust_tau_fd_detail,
                                          caller="household_prices_from_result")
    return prices, budget


def _bridge_labels(pref_codes: tuple[str, ...], calib_codes: Any, prefix: str, name: str) -> tuple[str, ...]:
    """Calibration labels for a bridged result; mismatched non-default preference labels raise."""
    calib_codes = tuple(str(c) for c in (() if calib_codes is None else calib_codes))
    pref_codes = tuple(pref_codes)
    if len(calib_codes) != len(pref_codes):
        return pref_codes
    if pref_codes in (calib_codes, _labels(None, len(pref_codes), prefix, name)):
        return calib_codes
    raise ValueError(f"preferences {name}_codes {list(pref_codes)} differ from the calibration's "
                     f"{list(calib_codes)}; calibrate the household with {name}_codes=calib.{name}_codes")


def compute_household_welfare_from_results(
    preferences: HouseholdPreferences,
    calib: TradeCalibrationResult,
    eq_result: TradeEquilibriumResult,
    *,
    base_result: TradeEquilibriumResult,
    category: int = 0,
    tau_fd: Any | None = None,
    base_tau_fd: Any | None = None,
    trust_tau_fd_detail: bool = False,
    tol: float = 1e-8,
    validate: bool = True,
) -> HouseholdWelfareResult:
    """Exact household EV/CV between two equilibria of the same calibration.

    ``preferences`` must have been calibrated on
    ``household_expenditure_from_calibration(calib, category)`` (checked to
    ``1e-8`` relative), so that the result prices' benchmark normalization
    matches the household's. Preference ``country_codes``/``sector_codes`` must
    equal the calibration's unless they are the generated defaults (``C00``,
    ``S00``, ...); the result carries the calibration's country labels. Prices
    and budgets come from :func:`household_prices_from_result` for both states
    (``tau_fd`` and ``base_tau_fd`` are required for tariffed legacy results;
    for consistent results they are optional and cross-checked against the
    recorded schedules; ``trust_tau_fd_detail`` applies to both legacy states)
    and the welfare from :func:`compute_household_welfare` with the base state
    as base. Both results must use the same accounting mode. ``tol`` has the
    meaning documented in :func:`household_prices_from_result`; errors name the
    failing state (``eq_result`` or ``base_result``), and
    ``metadata["tariff_schedule_verification"]`` records how each state's
    final-demand schedule was verified. With ``rule="fixed_baskets"`` and a
    single consumption category this reproduces
    :func:`~puremacro.trade.compute_hicksian_welfare` (Leontief basket) exactly.
    """
    if not isinstance(preferences, HouseholdPreferences):
        raise TypeError("preferences must be a HouseholdPreferences instance")
    x0 = household_expenditure_from_calibration(calib, category)
    if preferences.x0.shape != x0.shape or not np.allclose(preferences.x0, x0, rtol=1e-8, atol=1e-8 * np.max(x0)):
        raise ValueError("preferences were not calibrated on this calibration's benchmark spending for the "
                         "selected category; rebuild them with household_expenditure_from_calibration")
    countries = _bridge_labels(preferences.country_codes, calib.country_codes, "C", "country")
    _bridge_labels(preferences.sector_codes, calib.sector_codes, "S", "sector")
    mode = eq_result.metadata.get("accounting", "legacy") if isinstance(eq_result, TradeEquilibriumResult) else None
    base_mode = base_result.metadata.get("accounting", "legacy") if isinstance(base_result, TradeEquilibriumResult) else None
    if mode != base_mode:
        raise ValueError("both equilibria must use the same accounting mode")
    caller = "compute_household_welfare_from_results"
    p1, m1, check1 = _household_prices(eq_result, calib, category=category, tau_fd=tau_fd, tol=tol,
                                       trust_tau_fd_detail=trust_tau_fd_detail, caller=f"{caller} (eq_result)")
    p0, m0, check0 = _household_prices(base_result, calib, category=category, tau_fd=base_tau_fd, tol=tol,
                                       trust_tau_fd_detail=trust_tau_fd_detail, caller=f"{caller} (base_result)")
    out = compute_household_welfare(preferences, p1, m1, base_prices=p0, base_budget=m0, validate=validate)
    meta = dict(out.metadata)
    meta.update({"accounting": mode, "category": int(category),
                 "unit": calib.metadata.get("unit", "calibration value units"),
                 "tariff_schedule_verification": {"eq_result": check1, "base_result": check0},
                 "category_note": ("bundled OECD category 0 is HFCE + NPISH + GGFC: aggregate consumption "
                                   "including government final consumption")})
    return HouseholdWelfareResult(
        ev=out.ev, cv=out.cv, ev_pct_consumption=out.ev_pct_consumption, cv_pct_consumption=out.cv_pct_consumption,
        utility=out.utility, base_utility=out.base_utility, cost_index=out.cost_index, price_index=out.price_index,
        budget=out.budget, base_budget=out.base_budget, country_codes=countries, metadata=meta)
