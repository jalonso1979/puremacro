"""Independent accounting and FOC reconstruction for trade reallocation."""
from __future__ import annotations

from typing import Any

import numpy as np


def audit(result: Any, *, tolerance: float) -> dict[str, Any]:
    """Reject malformed restored data before broadcasting or indexing it."""
    from .reallocation import ReallocationEconomy, ReallocationSteadyState, ReallocationTransitionResult

    if not isinstance(result, (ReallocationSteadyState, ReallocationTransitionResult)):
        raise ValueError("audit requires a reallocation steady state or transition")
    try:
        if not isinstance(result.economy, ReallocationEconomy):
            raise ValueError("invalid economy")
        rates, coordinates = result.tariffs, result.coordinates
        prefix: tuple[int, ...]
        if isinstance(result, ReallocationSteadyState):
            prefix, values = (), result.values
        else:
            if rates.ndim != 3 or not 2 <= len(rates) <= 2000:
                raise ValueError("invalid transition horizon")
            prefix, values = (len(rates),), result.paths
        if rates.shape != (*prefix, 2, 2) or coordinates.shape != (*prefix, 16):
            raise ValueError("invalid tariff or coordinate shape")
        if not np.isfinite(rates).all() or not np.isfinite(coordinates).all():
            raise ValueError("nonfinite tariffs or coordinates")
        country = ("consumption", "labor_costs", "gross_absorption", "price_index", "tariff_revenue", "rebates")
        sector = ("producer_prices", "capital", "capital_next", "labor", "supplier_log_ratio", "output",
                  "wages", "rentals", "investment", "capital_costs", "sourcing_costs", "sector_absorption",
                  "sector_prices", "import_shares", "capital_q")
        for names, shape in ((country, (2,)), (sector, (2, 2)), (("recipes", "bilateral_quantities"), (2, 2, 2))):
            for name in names:
                if name not in values or values[name].shape != (*prefix, *shape) or not np.isfinite(values[name]).all():
                    raise ValueError(f"missing, malformed or nonfinite retained flow: {name}")
        if isinstance(result, ReallocationTransitionResult):
            for name in ("initial", "terminal"):
                boundary = getattr(result, name)
                if (not isinstance(boundary, ReallocationSteadyState)
                        or not isinstance(boundary.economy, ReallocationEconomy)
                        or boundary.economy.fingerprint != result.economy.fingerprint
                        or not audit(boundary, tolerance=tolerance)["accepted"]):
                    raise ValueError(f"invalid {name} boundary or economy")
            if not np.array_equal(result.terminal.tariffs, rates[-1]):
                raise ValueError("terminal tariffs do not match final policy")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _audit_equations(result, tolerance=tolerance)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, FloatingPointError, OverflowError) as exc:
        return dict(accepted=False, max_error=None, checks={}, reason=str(exc), tolerance=tolerance)


def _audit_equations(result: Any, *, tolerance: float) -> dict[str, Any]:
    from .reallocation import ReallocationSteadyState, ReallocationTransitionResult

    stationary = isinstance(result, ReallocationSteadyState)
    if not stationary and not isinstance(result, ReallocationTransitionResult):
        raise ValueError("audit requires a reallocation steady state or transition")
    model = result.economy
    v = {k: x[None] for k, x in result.values.items()} if stationary else result.paths
    rates = result.tariffs[None] if stationary else result.tariffs
    coordinates = result.coordinates[None] if stationary else result.coordinates
    T = len(rates)
    p, C, K, Knext, L, r = (v[k] for k in
        ("producer_prices", "consumption", "capital", "capital_next", "labor", "supplier_log_ratio"))
    P, Ps, q, X, Q, Y = (v[k] for k in
        ("price_index", "sector_prices", "recipes", "sector_absorption", "bilateral_quantities", "output"))
    positive = (p, C, K, Knext, L, P, Ps, q, X, Q, Y, v["wages"], v["rentals"], v["capital_q"])
    if (any(not np.isfinite(x).all() or np.min(x) <= 0 for x in positive)
            or np.min(rates) < 0 or np.min(v["gross_absorption"]) <= 0):
        return dict(accepted=False, max_error=None, checks={"positive_domain": False},
                    reason="nonpositive price, quantity, marginal capital value, or negative tariff")
    errors: dict[str, np.ndarray] = {}

    def check(name, left, right=0.):
        a, b = np.asarray(left), np.asarray(right)
        delta = abs(a-b)/(1+abs(a)+abs(b))
        errors[name] = np.max(np.broadcast_to(delta, (T, *delta.shape[1:])).reshape(T, -1), axis=1)

    check("numeraire", p[:, 1, 1], 1.)
    check("labor_mass", L.sum(axis=-1), model.labor_supply)
    check("production", Y, model.productivity*K**model.capital_share*L**(1-model.capital_share))
    check("wage_marginal_product", v["wages"]*L, (1-model.capital_share)*p*Y)
    check("capital_marginal_product", v["rentals"]*K, model.capital_share*p*Y)
    check("factor_payments", v["wages"]*L+v["rentals"]*K, p*Y)

    foreign = np.array([[[0., 1.]], [[1., 0.]]])
    alpha = (1-foreign)*model.home_shares[..., None]+foreign*(1-model.home_shares[..., None])
    log_ratio = np.log(q/alpha)
    rho = (model.trade_elasticity-1)/model.trade_elasticity
    if rho == 0:
        log_output = np.sum(alpha*log_ratio, axis=-1)
        weights = np.broadcast_to(alpha, q.shape)
    else:
        terms = np.log(alpha)+rho*log_ratio
        shift = terms.max(axis=-1)
        log_sum = shift+np.log(np.exp(terms-shift[..., None]).sum(axis=-1))
        if abs(rho) < 1e-5:
            log_sum = np.log1p(np.sum(alpha*np.expm1(rho*log_ratio), axis=-1))
        log_output = log_sum/rho
        weights = np.exp(terms-log_sum[..., None])
    check("ces_unit_output", log_output)
    check("supplier_ratio", np.sum((2*foreign-1)*log_ratio, axis=-1), r)
    delivered = p.transpose(0, 2, 1)[:, None]*(1+rates[..., None]*foreign)
    check("sector_price", Ps, np.sum(delivered*q, axis=-1))
    check("composite_price", np.log(P), np.sum(model.sector_weights*np.log(Ps), axis=-1))
    check("sector_expenditure", Ps*X, model.sector_weights*P[..., None]*v["gross_absorption"][..., None])
    check("bilateral_deliveries", Q, X[..., None]*q)
    check("goods_markets", Y, Q.sum(axis=1).transpose(0, 2, 1))
    check("import_share", v["import_shares"], np.sum(delivered*Q*foreign, axis=-1)/(Ps*X))

    Lprev = L if stationary else np.concatenate([result.initial.values["labor"][None], L[:-1]])
    rprev = r if stationary else np.concatenate([result.initial.values["supplier_log_ratio"][None], r[:-1]])
    inherited = Knext if stationary else np.concatenate([result.initial.values["capital"][None], Knext[:-1]])
    check("inherited_capital", K, inherited)
    investment = Knext-(1-model.delta)*K
    growth = Knext/K-1
    costK = .5*model.capital_cost*(Knext-K)**2/K
    costL = .5*model.labor_cost*np.sum(((L-Lprev)/model.labor_supply[:, None])**2, axis=-1)*model.labor_supply
    costQ = .5*model.sourcing_cost*model.labor_supply[:, None]*model.sector_weights*(r-rprev)**2
    check("capital_law", Knext, (1-model.delta)*K+v["investment"])
    check("capital_adjustment_resources", v["capital_costs"], costK)
    check("labor_adjustment_resources", v["labor_costs"], costL)
    check("sourcing_adjustment_resources", v["sourcing_costs"], costQ)
    absorption = C+np.sum(investment+costK+costQ, axis=-1)+costL
    composite = np.exp(np.sum(model.sector_weights*np.log(X/model.sector_weights), axis=-1))
    check("physical_resources", composite, absorption)
    check("recorded_absorption", v["gross_absorption"], absorption)
    receipts = np.sum(rates[..., None]*foreign*Q*p.transpose(0, 2, 1)[:, None], axis=(2, 3))
    check("tariff_receipts", v["tariff_revenue"], receipts)
    check("government_budget", v["rebates"], receipts)
    income = np.sum(v["wages"]*L+v["rentals"]*K, axis=-1)+receipts
    check("all_household_budgets", income, P*absorption)
    for c in range(2):
        imports = np.sum(Q[:, c, :, 1-c]*p[:, 1-c], axis=-1)
        exports = np.sum(Q[:, 1-c, :, c]*p[:, c], axis=-1)
        check(f"external_balance_{c}", imports, exports)

    qK = 1+model.capital_cost*growth
    check("capital_marginal_cost", v["capital_q"], qK)
    wage_gain = (v["wages"][:, :, 0]-v["wages"][:, :, 1])/P
    dlogP = np.sum(delivered*q*(foreign-np.sum(weights*foreign, axis=-1)[..., None]), axis=-1)/Ps
    sourcing_gain = absorption[..., None]/model.labor_supply[:, None]*dlogP
    if stationary:
        check("capital_euler", qK, model.beta*(v["rentals"]/P[..., None]+1-model.delta))
        check("labor_optimality", wage_gain)
        check("sourcing_optimality", sourcing_gain)
    else:
        discount = model.beta*(C[1:]/C[:-1])**(-model.risk_aversion)
        marginal_return = v["rentals"][1:]/P[1:, :, None]+1-model.delta+model.capital_cost/2*((Knext[1:]/K[1:])**2-1)
        future_value = np.concatenate([discount[..., None]*marginal_return,
            np.broadcast_to((result.terminal.values["consumption"]/C[-1])[:, None]**(-model.risk_aversion), (2, 2))[None]])
        check("capital_euler_and_boundary", qK, future_value)
        labor_change = (L[:, :, 0]-Lprev[:, :, 0])/model.labor_supply
        labor_charge = 2*model.labor_cost*labor_change
        labor_credit = np.concatenate([discount*2*model.labor_cost*labor_change[1:], np.zeros((1, 2))])
        check("labor_optimality", wage_gain+labor_credit, labor_charge)
        source_charge = model.sourcing_cost*(r-rprev)
        source_credit = np.concatenate([discount[..., None]*source_charge[1:], np.zeros((1, 2, 2))])
        check("sourcing_optimality", sourcing_gain+source_charge, source_credit)

    rebuilt = np.column_stack([np.log(p).reshape(T, 4), np.log(C), np.log(Knext).reshape(T, 4),
                               np.log(L[:, :, 0]/L[:, :, 1]), r.reshape(T, 4)])
    check("coordinate_consistency", coordinates, rebuilt)
    table = np.column_stack(list(errors.values()))
    maxima = {name: float(np.max(value)) for name, value in errors.items()}
    maximum = float(np.max(table))
    accepted = bool(np.isfinite(table).all() and maximum <= tolerance)
    return dict(accepted=accepted, max_error=maximum, checks=maxima, check_names=tuple(errors),
                period_errors=table, tolerance=tolerance,
                minimum_investment=float(np.min(investment)), investment_is_reversible=True,
                scope="retained physical quantities, all budgets, fiscal accounts and local FOCs")
