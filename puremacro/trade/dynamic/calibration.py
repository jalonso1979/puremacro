"""Stationary recalibration of one IO year into a sector-capital economy.

Ported from the IO engine's ``dynamic_model/native_economy.py::calibrate_native``.
A single year's IO table is not automatically a dynamic steady state; the
calibration records every transformation it applies in a report ledger
(``DynamicCalibration.report`` and ``.to_dataframe()``).

Mathematics (per active cell ``j`` in country ``c(j)``, benchmark prices one):

* factor income ``VA0_j = labour_j + capital_j`` (observed or reclassified),
  capital share ``alpha_j = capital_j / VA0_j``;
* output-tax wedge ``tax_j = (y0_j - inputs_j - VA0_j) / y0_j`` (product and
  production taxes, plus any declared source rounding reconciliation);
* investment purchaser price ``PI0_c = 1 + TFD_G,c / I_obs,c`` and the
  stationary rental ``R0_j = PI0_c (1/beta - 1 + delta)``;
* capital ``K0_j = alpha_j VA0_j / R0_j`` and replacement ``I0_j = delta K0_j``;
* the investment basket ``omegaI`` keeps the observed composition; the
  stationary replacement flow is reassigned inside observed absorption
  (``C + G + V + VAL`` by seller and destination) and the remainder, plus exempt
  purchases abroad ``X``, forms the consumption basket ``omegaC`` with taxable
  part ``omegaCtax``; ``PC0_c = 1 + (TFD_c,total - (PI0_c - 1) I_nat,c) / C0_c``;
* labour ``L0_j = (1 - alpha_j) VA0_j``; net-export transfers
  ``XN0_c = output_c - inputs_c - absorption_c`` sum to zero by construction.

Limitations quoted from the IO documentation (IMPLEMENTATION.md): "A single
IO table does not identify historical capital stocks, depreciation, adjustment
costs, or intertemporal preferences." (NATIVE_MODEL.md): "Recorded inventory
changes and valuables are explicitly reinterpreted in this stationary
absorption calibration. They do not identify inventory stocks."
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from ._results import DynamicCalibration

__all__ = ["calibrate_dynamic"]


def _national(x: np.ndarray, country: np.ndarray, count: int) -> np.ndarray:
    return np.bincount(country, weights=x, minlength=count)


def _maxabs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x), initial=0.0))


def _readonly(x: Any) -> np.ndarray:
    arr = np.array(x, dtype=float, copy=True)
    arr.flags.writeable = False
    return arr


def calibrate_dynamic(accounts: Any, *, beta: float = .96, delta: float = .08,
                      factor_policy: str = "strict",
                      investment_policy: str = "strict",
                      capital_share_bounds: tuple[float, float] = (.02, .98),
                      factor_floor: float = 1e-5,
                      accounting_tolerance: float = 1e-6,
                      accounting_policy: str = "strict",
                      accounting_world_tolerance: float = 1e-6) -> DynamicCalibration:
    """Reconcile observed flows with an exactly stationary capital economy.

    Parameters
    ----------
    accounts : DynamicAccounts
        Source accounts (see :class:`~puremacro.trade.dynamic.DynamicAccounts`).
        Any object with the same attribute names is accepted.
    beta, delta : float
        Discount factor and depreciation rate, both strictly inside (0, 1).
        Not identified by one IO year; the IO demonstrations use .96 and .08.
    factor_policy : {"strict", "reclassify_losses"}
        ``strict`` rejects cells with nonpositive labour or capital income.
        ``reclassify_losses`` is an explicit conditional calibration: negative
        observed factor components become net production subsidies, positive
        gross factor income is divided using ``capital_share_bounds`` and
        floored at ``factor_floor * y0``. Production cost accounts are
        unchanged; factor ownership is an assumption. Every altered cell is
        reported.
    investment_policy : {"strict", "reallocate"}
        ``reallocate`` changes the investment basket only where its observed
        composition cannot finance stationary replacement from nonnegative
        available absorption; it preserves the original support and relative
        shares until a source good's availability binds (capped proportional
        allocation, the relative-entropy projection onto the availability
        bounds), opening extra support only if necessary. Every such country
        is reported.
    accounting_tolerance : float
        Maximum relative production-cost gap ``|y0 - inputs - VA - taxes| / y0``
        accepted under ``strict`` accounting.
    accounting_policy : {"strict", "reconcile_rounding"}
        ``reconcile_rounding`` accepts source rounding gaps whose absolute sum
        is at most ``accounting_world_tolerance`` of world output and whose
        maximum is at most ``1e-6`` of the largest output, folding them into
        the output-tax wedge and reporting them.

    Returns
    -------
    DynamicCalibration
        Frozen calibration; ``report`` and ``to_dataframe()`` hold the ledger.

    Raises
    ------
    ValueError
        For inconsistent dimensions, nonpositive productive-cell output,
        unbalanced accounts, nonpositive factor income under ``strict``,
        infeasible stationary baskets under ``strict`` investment, or an
        unbalanced world transfer.
    """
    if not 0 < beta < 1 or not 0 < delta < 1:
        raise ValueError("beta and delta must lie strictly between zero and one")
    if factor_policy not in {"strict", "reclassify_losses"}:
        raise ValueError("Unknown factor_policy")
    if investment_policy not in {"strict", "reallocate"}:
        raise ValueError("Unknown investment_policy")
    if accounting_policy not in {"strict", "reconcile_rounding"}:
        raise ValueError("Unknown accounting_policy")
    if not np.isfinite(accounting_world_tolerance) or accounting_world_tolerance < 0:
        raise ValueError("accounting_world_tolerance must be finite and nonnegative")
    low, high = capital_share_bounds
    if not 0 < low < high < 1 or factor_floor <= 0:
        raise ValueError("Invalid factor regularization parameters")
    data = accounts
    countries, sectors = tuple(data.countries), tuple(data.sectors)
    nc = len(countries)
    Zraw = sparse.csr_matrix(data.Z, dtype=float)
    finals = [np.asarray(getattr(data, name), dtype=float) for name in
              ("final_consumption", "final_investment", "purchases_abroad",
               "inventory_changes", "valuables")]
    nr = Zraw.shape[0]
    if Zraw.shape != (nr, nr) or any(x.shape != (nr, nc) for x in finals):
        raise ValueError("Dynamic input dimensions disagree")
    if nr != nc * len(sectors):
        raise ValueError("Cells must use country-major source order")
    if np.any(Zraw.data < 0) or not np.all(np.isfinite(Zraw.data)):
        raise ValueError("Intermediate deliveries must be finite and nonnegative")
    if any(not np.all(np.isfinite(x)) for x in finals):
        raise ValueError("Nonfinite final deliveries")
    observed = np.asarray(data.output, dtype=float)
    all_final = sum(finals)
    raw_goods_output = np.asarray(Zraw.sum(axis=1)).ravel() + all_final.sum(axis=1)
    threshold = max(float(np.max(np.abs(raw_goods_output), initial=0.0)), 1.) * 1e-13
    active = np.flatnonzero(raw_goods_output > 0.)
    inactive = np.setdiff1d(np.arange(nr), active)
    dropped_abs = float(np.abs(Zraw[inactive, :].data).sum()
                        + np.abs(Zraw[:, inactive].data).sum()
                        + np.abs(all_final[inactive]).sum())
    global_scale = max(float(np.abs(raw_goods_output).sum()), 1.)
    if dropped_abs / global_scale > 1e-10:
        raise ValueError("Nonpositive-output cells contain material deliveries; explicit data repair is required")
    Z = Zraw[active, :][:, active].tocsr()
    fC, fI, fX, fS, fV = [x[active].copy() for x in finals]
    country = active // len(sectors)
    y0 = np.asarray(Z.sum(axis=1)).ravel() + (fC + fI + fX + fS + fV).sum(axis=1)
    if np.any(y0 <= 0):
        raise ValueError("Nonpositive productive-cell output")
    labor_raw, capital_raw = data.labor_compensation, data.operating_surplus
    if labor_raw is None or capital_raw is None:
        raise ValueError("A declared factor split is required; build the accounts with labour and capital rows")
    labor_raw = np.asarray(labor_raw, dtype=float)[active]
    capital_raw = np.asarray(capital_raw, dtype=float)[active]
    if not np.all(np.isfinite(labor_raw + capital_raw)):
        raise ValueError("Nonfinite factor accounts")
    raw_factor = labor_raw + capital_raw
    bad = (labor_raw <= 0) | (capital_raw <= 0)
    original_tax = (np.asarray(data.TLS, dtype=float)
                    + np.asarray(data.production_taxes, dtype=float))[active]
    input_bill = np.asarray(Z.sum(axis=0)).ravel()
    source_cost_gap = y0 - input_bill - raw_factor - original_tax
    source_gap_rel = np.abs(source_cost_gap) / np.maximum(y0, 1.)
    cost_gap_sum_relative = float(np.abs(source_cost_gap).sum()) / global_scale
    cost_gap_max_relative = _maxabs(source_cost_gap) / max(_maxabs(y0), 1.)
    rounding_admissible = (cost_gap_sum_relative <= accounting_world_tolerance
                           and cost_gap_max_relative <= 1e-6)
    if (_maxabs(source_gap_rel) > accounting_tolerance
            and not (accounting_policy == "reconcile_rounding" and rounding_admissible)):
        raise ValueError(f"Source production accounts do not balance: max relative gap {_maxabs(source_gap_rel):.3g}")
    if factor_policy == "strict" and np.any(bad):
        raise ValueError(f"{int(bad.sum())} active cells have nonpositive labor/capital income; choose an explicit factor policy")
    if factor_policy == "strict":
        va = raw_factor.copy()
        alpha = capital_raw / va
        altered = np.zeros(len(active), dtype=bool)
    else:
        gross_labor = np.maximum(labor_raw, 0.)
        gross_capital = np.maximum(capital_raw, 0.)
        va = np.maximum(gross_labor + gross_capital, factor_floor * y0)
        alpha_unbounded = np.divide(gross_capital, gross_labor + gross_capital,
                                    out=np.full_like(va, .5),
                                    where=(gross_labor + gross_capital) > 0)
        alpha = np.clip(alpha_unbounded, low, high)
        altered = bad | (va != raw_factor) | (alpha != alpha_unbounded)
    if np.any(va <= 0) or np.any((alpha <= 0) | (alpha >= 1)):
        raise ValueError("The interior capital model requires positive factor income and shares in (0,1)")
    # Tiny source rounding discrepancies and declared loss reclassification are
    # both visible in the report, rather than hidden in household transfers.
    production_tax = y0 - input_bill - va
    tax = production_tax / y0
    b = va / y0
    A = (Z @ sparse.diags(1. / y0)).tocsr()
    tfd = np.asarray(data.TFD, dtype=float)
    if tfd.shape != (nc, 5) or not np.all(np.isfinite(tfd)):
        raise ValueError("TFD must use the five named final-use categories")
    investment_observed = fI.sum(axis=0)
    if np.any(investment_observed <= 0):
        raise ValueError("Each country needs a positive observed investment basket")
    PI0 = 1. + tfd[:, 1] / investment_observed
    if np.any(PI0 <= 0):
        raise ValueError("Nonpositive benchmark investment purchaser price")
    tI = 1. - 1. / PI0
    rstar = 1. / beta - 1. + delta
    R0 = PI0[country] * rstar
    K0 = alpha * va / R0
    I0 = delta * K0
    Inational = _national(I0, country, nc)
    omegaI = fI / investment_observed[None, :]
    absorption = fC + fI + fS + fV
    if np.any(absorption < -threshold) or np.any(fX < -threshold):
        raise ValueError("Stationary consumption pooling has negative commodity deliveries")
    # Remove only sub-roundoff negative entries and expose their absolute mass.
    negative_final_rounding = float(-np.minimum(absorption, 0.).sum() - np.minimum(fX, 0.).sum())
    absorption = np.maximum(absorption, 0.)
    fX = np.maximum(fX, 0.)
    original_stationary_I = omegaI * Inational[None, :]
    feasible = np.min(absorption - original_stationary_I, axis=0) >= -threshold
    reallocated: list[str] = []
    investment_reallocation_records: list[dict[str, Any]] = []
    for c in np.flatnonzero(~feasible):
        if investment_policy == "strict":
            raise ValueError(f"Observed investment basket cannot support stationary replacement in {countries[c]}; choose investment_policy='reallocate'")
        available = float(absorption[:, c].sum())
        if Inational[c] >= available:
            raise ValueError(f"Replacement investment exhausts available absorption in {countries[c]}")
        capacity, weight = absorption[:, c], fI[:, c]
        support = weight > 0.
        support_capacity = float(capacity[support].sum())
        if support_capacity >= Inational[c]:
            # Capped proportional allocation is the relative-entropy projection
            # of observed investment shares onto physical availability bounds.
            lower, upper = 0., max(Inational[c] / investment_observed[c], 1.)
            while float(np.minimum(capacity, upper * weight).sum()) < Inational[c]:
                upper *= 2.
            for _ in range(80):
                middle = .5 * (lower + upper)
                if float(np.minimum(capacity, middle * weight).sum()) < Inational[c]:
                    lower = middle
                else:
                    upper = middle
            allocation = np.minimum(capacity, .5 * (lower + upper) * weight)
        else:
            allocation = np.where(support, capacity, 0.)
            extra = np.where(~support, capacity, 0.)
            allocation += extra * (Inational[c] - support_capacity) / float(extra.sum())
        # Roundoff correction is distributed only over remaining capacity.
        gap = Inational[c] - float(allocation.sum())
        if gap > 0:
            slack = np.maximum(capacity - allocation, 0.)
            allocation += gap * slack / float(slack.sum())
        elif gap < 0:
            allocation *= Inational[c] / float(allocation.sum())
        omegaI[:, c] = allocation / Inational[c]
        reallocated.append(countries[c])
        investment_reallocation_records.append(dict(
            country=countries[c], stationary_investment=float(Inational[c]),
            source_basket_absolute_change=float(np.abs(allocation - original_stationary_I[:, c]).sum()),
            capacity_bound_cells=int(np.count_nonzero(np.isclose(allocation, capacity, rtol=1e-10, atol=0.) & (capacity > 0))),
            new_source_cells=int(np.count_nonzero((allocation > 0) & ~support)),
            new_source_investment=float(allocation[~support].sum())))
    investment_deliveries = omegaI * Inational[None, :]
    taxable_C = absorption - investment_deliveries
    if np.any(taxable_C < -threshold):
        raise ValueError("Negative stationary consumption")
    taxable_C = np.maximum(taxable_C, 0.)
    C0 = (taxable_C + fX).sum(axis=0)
    if np.any(C0 <= 0):
        raise ValueError("Nonpositive stationary consumption aggregate")
    omegaC, omegaCtax = (taxable_C + fX) / C0, taxable_C / C0
    consumption_taxes = tfd.sum(axis=1) - (PI0 - 1.) * Inational
    PC0 = 1. + consumption_taxes / C0
    if np.any(PC0 <= 0):
        raise ValueError("Nonpositive benchmark consumption purchaser price")
    tC = 1. - 1. / PC0
    L0sector = (1. - alpha) * va
    L0 = _national(L0sector, country, nc)
    if np.any(L0 <= 0):
        raise ValueError("Every country needs positive labor supply")
    base_final = omegaC * C0 + investment_deliveries
    # Net exports are basic-price cross-border deliveries, not arbitrary budget
    # residuals.  Their sum is zero by construction of world trade accounts.
    output_by_country = _national(y0, country, nc)
    input_by_country = _national(input_bill, country, nc)
    XN0 = output_by_country - input_by_country - base_final.sum(axis=0)
    balanced_gap = float(XN0.sum())
    if abs(balanced_gap) > 1e-9 * global_scale:
        raise ValueError("World transfer source is not balanced")
    XN0[-1] -= balanced_gap
    Y0 = _national(va + production_tax, country, nc) + tfd.sum(axis=1)
    if np.any(Y0 <= 0):
        raise ValueError("Nonpositive country GDP; a separate country calibration is required")
    all_labels = tuple(getattr(data, "cell_labels", ()))
    labels = [str(all_labels[i]) if all_labels else
              f"{countries[int(i) // len(sectors)]}:{sectors[int(i) % len(sectors)]}" for i in active]
    adjustment_records = []
    for j in np.flatnonzero(altered):
        adjustment_records.append(dict(cell=labels[j], labor_observed=float(labor_raw[j]),
                                       capital_observed=float(capital_raw[j]),
                                       factor_income_calibrated=float(va[j]),
                                       capital_share_calibrated=float(alpha[j]),
                                       production_tax_change=float(production_tax[j] - original_tax[j])))
    report = dict(dataset=str(getattr(data, "dataset", "unknown")),
                  source_metadata=dict(getattr(data, "metadata", {}) or {}),
                  original_cells=nr, active_cells=len(active), inactive_cells=len(inactive),
                  dropped_inactive_absolute_flow=dropped_abs,
                  source_goods_output_gap_max=float(np.max(np.abs(observed - raw_goods_output))),
                  source_cost_gap_max_relative=_maxabs(source_gap_rel),
                  accounting_policy=accounting_policy,
                  accounting_world_tolerance=accounting_world_tolerance,
                  accounting_largest_output_tolerance=1e-6,
                  source_cost_gap_absolute_sum=float(np.abs(source_cost_gap).sum()),
                  source_cost_gap_sum_over_world_output=cost_gap_sum_relative,
                  source_cost_gap_max_over_largest_output=cost_gap_max_relative,
                  source_cost_reconciliation_total=float(source_cost_gap.sum()),
                  factor_policy=factor_policy, factor_adjusted_cells=len(adjustment_records),
                  factor_adjustments=adjustment_records,
                  factor_subsidy_reclassification_total=float((va - raw_factor).sum()),
                  capital_share_bounds=list(capital_share_bounds), factor_floor=factor_floor,
                  investment_policy=investment_policy,
                  investment_basket_reallocated_countries=reallocated,
                  investment_basket_reallocations=investment_reallocation_records,
                  observed_investment=investment_observed.tolist(),
                  stationary_investment=Inational.tolist(),
                  investment_reallocation_L1_over_observed_world_investment=(
                      sum(x["source_basket_absolute_change"] for x in investment_reallocation_records)
                      / float(investment_observed.sum())),
                  investment_to_consumption=(investment_observed - Inational).tolist(),
                  inventory_flow_reclassified_to_consumption=fS.sum(axis=0).tolist(),
                  valuables_reclassified_to_consumption=fV.sum(axis=0).tolist(),
                  negative_final_rounding_removed=negative_final_rounding,
                  source_delivery_reconciliation_max=_maxabs(base_final - (fC + fI + fX + fS + fV)),
                  final_tax_receipt_reconciliation_max=_maxabs(consumption_taxes + (PI0 - 1.) * Inational - tfd.sum(axis=1)),
                  transfer_world_sum=float(XN0.sum()),
                  financial_closure="financial autarky with balanced benchmark net-export transfers, indexed to last-country wage",
                  inventory_closure="observed inventory changes and valuables reclassified into stationary consumption; no endogenous stocks",
                  beta=beta, delta=delta)
    mask = getattr(data, "merchandise_mask", None)
    if mask is not None:
        mask = np.array(mask, dtype=bool, copy=True)
        mask.flags.writeable = False
    active_out = np.array(active, dtype=int)
    active_out.flags.writeable = False
    country_out = np.array(country, dtype=int)
    country_out.flags.writeable = False
    return DynamicCalibration(
        countries=countries, sectors=sectors, active_indices=active_out,
        country=country_out, A=A, Z=Z, y0=_readonly(y0), VA0=_readonly(va),
        b=_readonly(b), tax=_readonly(tax), alpha=_readonly(alpha), omegaC=_readonly(omegaC),
        omegaCtax=_readonly(omegaCtax), omegaI=_readonly(omegaI),
        qOther=_readonly(np.zeros((len(active), nc))), TV=_readonly(np.zeros(nc)),
        tC=_readonly(tC), tI=_readonly(tI), C0=_readonly(C0), I0=_readonly(I0), K0=_readonly(K0),
        R0=_readonly(R0), L0sector=_readonly(L0sector), L0=_readonly(L0), PC0=_readonly(PC0),
        PI0=_readonly(PI0), Y0=_readonly(Y0), XN0=_readonly(XN0), beta=float(beta), delta=float(delta),
        merchandise_mask=mask,
        report=report,
        metadata={"source": report["dataset"], "cell_labels": tuple(labels)})
