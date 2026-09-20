"""Hicksian consumption welfare for the consistent-accounting trade model.

The expenditure function is Cobb–Douglas across explicitly selected final-use
baskets, each Leontief across origins. Investment is excluded. EV and CV are
positive for gains. The endpoint Shapley attribution is accounting, not a causal
terms-of-trade or allocative-efficiency decomposition. See docs/trade_welfare.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from ._accounting import evaluate
from ._results import TradeCalibrationResult, TradeEquilibriumResult

__all__ = ["HicksianWelfareResult", "compute_hicksian_welfare"]


@dataclass(frozen=True)
class HicksianWelfareResult:
    """Consumption EV/CV in calibration value units and baseline-price attribution.

    Percent EV is supplied against both baseline consumption and baseline GDP.
    All components are levels, never percentages. Fiscal transfers include
    domestic taxes and import duties; their two subcomponents are not additional
    gains to add on top of the fiscal-transfer effect.
    """

    ev: float
    cv: float
    ev_pct_consumption: float
    ev_pct_gdp: float
    consumption_base: float
    consumption_counterfactual: float
    utility_ratio: float
    price_index: float
    price_effect: float
    factor_income_effect: float
    fiscal_transfer_effect: float
    tariff_rebate_effect: float
    domestic_tax_rebate_effect: float
    decomposition_residual: float
    country_code: str
    consumption_categories: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Main EV attribution in value units (CV uses counterfactual prices)."""
        rows = [("Purchaser prices", self.price_effect),
                ("Factor income", self.factor_income_effect),
                ("Fiscal transfers", self.fiscal_transfer_effect),
                ("Equivalent variation", self.ev),
                ("Decomposition residual", self.decomposition_residual)]
        return pd.DataFrame(rows, columns=["Component", "Value"]).set_index("Component")

    def summary(self) -> str:
        return (f"Hicksian consumption welfare [{self.country_code}]: "
                f"EV={self.ev:.6g} {self.metadata['unit']} "
                f"({self.ev_pct_consumption:.4f}% of baseline consumption), "
                f"CV={self.cv:.6g}; positive values denote gains.")

    def to_markdown(self, **kwargs: Any) -> str:
        return df_to_markdown(self.to_dataframe(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return df_to_latex(self.to_dataframe(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return df_to_typst(self.to_dataframe(), **kwargs)


def _checked_state(result, calib, tol):
    if not isinstance(result, TradeEquilibriumResult):
        raise TypeError("Both states must be TradeEquilibriumResult instances")
    meta = result.metadata
    if meta.get("accounting") != "consistent":
        raise ValueError('Hicksian welfare requires accounting="consistent" in both states')
    if not result.converged:
        raise ValueError("Hicksian welfare requires converged equilibria")
    if meta.get("fiscal_closure") != "lump_sum":
        raise NotImplementedError("Hicksian attribution currently requires lump-sum fiscal transfers")
    if any(name not in meta for name in ("intermediate_tariff_multipliers", "final_tariff_multipliers")):
        raise ValueError("Result lacks tariff schedules for validation; solve it again")
    block = evaluate(result.x_sol, calib, meta["intermediate_tariff_multipliers"],
                     meta["final_tariff_multipliers"], None, None, sigma=meta.get("sigma", 0.))
    equilibrium_tol = float(meta.get("tol", 1e-8))
    residuals = np.r_[block["residuals"], block["physical_residuals"]]
    if (not np.isfinite(equilibrium_tol) or equilibrium_tol <= 0
            or not np.isfinite(residuals).all()
            or np.max(np.abs(residuals)) > equilibrium_tol
            or np.any(block["expenditure"] < 0)):
        raise ValueError("Returned state fails the consistent equilibrium/accounting audit")
    # Reconstruct the returned quantities and prices: a cached convergence flag
    # or manually replaced result field cannot substitute for the demand system.
    pairs = ((result.p_sol, block["p"]), (result.y_sol, block["y"]),
             (result.w_sol, block["w"]), (result.r_sol, block["r"]),
             (result.T_sol, block["T"]), (result.Pfd_final, block["P"]),
             (result.c_fd, block["c"]), (result.tariffs, block["tariffs"]))
    for actual, expected in pairs:
        if (actual is None or np.shape(actual) != expected.shape
                or not np.isfinite(actual).all()
                or not np.allclose(actual, expected, rtol=tol, atol=tol)):
            raise ValueError("Result fields disagree with the recalculated equilibrium")
    factor_income = (block["w"].ravel()*calib.l_endow.ravel()
                     + block["r"].ravel()*calib.k_endow.ravel())
    if not np.allclose(result.gdp, factor_income+block["government"], rtol=tol, atol=tol):
        raise ValueError("GDP disagrees with factor income plus fiscal receipts")
    return block, factor_income


def compute_hicksian_welfare(
    calib: TradeCalibrationResult,
    eq_result: TradeEquilibriumResult,
    *,
    base_result: TradeEquilibriumResult,
    target_country: str | int = 0,
    consumption_categories: Sequence[int] = (0,),
    tol: float = 1e-8,
) -> HicksianWelfareResult:
    """Exact consumption EV/CV with an endpoint Shapley income/price attribution.

    Both equilibria must use ``accounting="consistent"`` and the same calibration.
    Category 0 is the default consumption aggregate. Additional non-investment
    categories may be selected explicitly; their fixed utility weights are the
    selected theta shares normalized to one. Multi-category investment (index 1)
    cannot be treated as consumption. A single-category model requires zero
    foreign saving. Values use the calibration's units, not an assumed USD unit.

    With U=prod(c_k**omega_k), e(P,U)=U*prod((P_k/omega_k)**omega_k).
    EV=e(P0,U1)-e(P0,U0); CV=e(P1,U1)-e(P1,U0), positive for gains.
    The three effects allocate the endpoint interaction symmetrically between
    purchaser prices, factor income and all fiscal transfers. They are not
    causal GE channels or a Harberger terms-of-trade/efficiency theorem.

    ``tol`` is the relative/absolute comparison tolerance for demand and welfare
    identities. Each equilibrium is independently rechecked at its solver's
    recorded absolute tolerance. Missing, stale, incompatible or failed states
    raise rather than producing a welfare certificate.
    """
    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError("calib must be a TradeCalibrationResult")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if isinstance(target_country, str):
        if target_country not in calib.country_codes:
            raise ValueError(f"Unknown country {target_country!r}")
        country = calib.country_codes.index(target_country)
    elif isinstance(target_country, Integral) and not isinstance(target_country, bool):
        country = int(target_country)
    else:
        raise TypeError("target_country must be a country code or integer index")
    if not 0 <= country < calib.nc:
        raise ValueError("target_country index is out of range")
    categories = tuple(consumption_categories)
    if (not categories or any(not isinstance(k, Integral) or isinstance(k, bool) for k in categories)
            or len(set(categories)) != len(categories)
            or any(k < 0 or k >= calib.n_final_demand for k in categories)):
        raise ValueError("consumption_categories must contain distinct valid integer indices")
    if calib.n_final_demand > 1 and 1 in categories:
        raise ValueError("Investment category 1 cannot be included in consumption welfare")
    if calib.n_final_demand == 1 and np.any(np.asarray(calib.invforT) != 0):
        raise NotImplementedError("Single-category welfare requires zero foreign saving")
    before, factor0 = _checked_state(base_result, calib, tol)
    after, factor1 = _checked_state(eq_result, calib, tol)
    idx = list(categories)
    weights = calib.theta[0, idx, country]
    share = float(weights.sum())
    if share <= 0:
        raise ValueError("Selected consumption must have a positive expenditure share")
    active = weights > 0
    omega = weights[active]/share
    p0, p1 = before["P"][0, idx, country][active], after["P"][0, idx, country][active]
    q0, q1 = before["c"][0, idx, country][active], after["c"][0, idx, country][active]
    if any(not np.isfinite(v).all() or np.any(v <= 0) for v in (p0, p1, q0, q1)):
        raise ValueError("Positive finite consumption and purchaser prices are required")
    # Compute both endpoint utilities and expenditure functions independently
    # of the attribution. Log ratios/expm1 preserve small welfare changes.
    log_u0, log_u1 = np.sum(omega*np.log(q0)), np.sum(omega*np.log(q1))
    log_unit0 = np.sum(omega*(np.log(p0)-np.log(omega)))
    log_unit1 = np.sum(omega*(np.log(p1)-np.log(omega)))
    log_ratio = float(np.sum(omega*(np.log(q1)-np.log(q0))))
    log_price = float(np.sum(omega*(np.log(p1)-np.log(p0))))
    with np.errstate(over="ignore", invalid="ignore"):
        expense0, expense1 = np.exp(log_u0+log_unit0), np.exp(log_u1+log_unit1)
        ev = float(expense0*np.expm1(log_ratio))
        cv = float(-expense1*np.expm1(-log_ratio))
        utility_ratio, price_index = np.exp(log_ratio), np.exp(log_price)
        inverse_index = np.exp(-log_price)
        price_effect = float(share*np.expm1(-log_price)*(
            factor0[country]+factor1[country]+before["T"].ravel()[country]+after["T"].ravel()[country])/2)
        income_weight = share*(1+inverse_index)/2
        factor_effect = float(income_weight*(factor1[country]-factor0[country]))
        fiscal_effect = float(income_weight*(after["T"].ravel()[country]-before["T"].ravel()[country]))
        tariff_effect = float(income_weight*(after["tariffs"][country]-before["tariffs"][country]))
        domestic0 = before["production_tax"].sum(1).ravel()+before["final_tax"].sum(1).ravel()
        domestic1 = after["production_tax"].sum(1).ravel()+after["final_tax"].sum(1).ravel()
        domestic_effect = float(income_weight*(domestic1[country]-domestic0[country]))
    values = [ev, cv, expense0, expense1, utility_ratio, price_index,
              price_effect, factor_effect, fiscal_effect, tariff_effect, domestic_effect]
    if not np.isfinite(values).all() or min(expense0, expense1, utility_ratio, price_index) <= 0:
        raise ValueError("Welfare calculation overflowed or underflowed; rescale the model")
    for state, prices, quantities, expense in ((before, p0, q0, expense0), (after, p1, q1, expense1)):
        expenditure = float(prices @ quantities)
        expected = share*state["income"].ravel()[country]
        if (not np.allclose(prices*quantities, omega*expenditure, rtol=tol, atol=tol)
                or not np.isclose(expense, expenditure, rtol=tol, atol=tol)
                or not np.isclose(expense, expected, rtol=tol, atol=tol)):
            raise ValueError("Consumption does not satisfy the declared expenditure function")
    residual = ev-price_effect-factor_effect-fiscal_effect
    scale = max(1., expense0, expense1, abs(ev))
    if (abs(residual) > tol*scale
            or abs(fiscal_effect-tariff_effect-domestic_effect) > tol*scale):
        raise ValueError("Hicksian attribution fails the expenditure/fiscal identity")
    gdp0 = float(np.asarray(base_result.gdp).ravel()[country])
    if not np.isfinite(gdp0) or gdp0 <= 0:
        raise ValueError("Baseline GDP must be positive for the GDP-normalized EV")
    code = calib.country_codes[country] if calib.country_codes else str(country)
    return HicksianWelfareResult(
        ev=ev, cv=cv, ev_pct_consumption=100*ev/expense0, ev_pct_gdp=100*ev/gdp0,
        consumption_base=float(expense0), consumption_counterfactual=float(expense1),
        utility_ratio=float(utility_ratio), price_index=float(price_index),
        price_effect=price_effect, factor_income_effect=factor_effect,
        fiscal_transfer_effect=fiscal_effect, tariff_rebate_effect=tariff_effect,
        domestic_tax_rebate_effect=domestic_effect, decomposition_residual=float(residual),
        country_code=code, consumption_categories=tuple(int(k) for k in categories),
        metadata={"unit": calib.metadata.get("unit", "calibration value units"),
                  "accounting": "consistent", "utility": "Cobb-Douglas over selected Leontief final-use baskets",
                  "weights": omega.copy(), "consumption_income_share": share,
                  "positive_is_gain": True, "attribution": "endpoint Shapley: purchaser prices, factor income, fiscal transfers",
                  "causal_decomposition": False, "tolerance": tol,
                  "numeraire": before.get("numeraire", "first country/sector producer price = 1")},
    )
