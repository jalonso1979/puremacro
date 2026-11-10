"""Independent scalar accounting and exhaustive pure-equilibrium reconstruction."""
from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from puremacro.sovereign.banking import SovereignBankingResult, SovereignBankingParameters


def _state(p: SovereignBankingParameters, y: float, a: float, b: float, q: float, d: int) -> list[float]:
    """No call to the vectorized allocation or public accounts function."""
    reserve = p.initial_equity-q*p.domestic_share*b
    payment, loss = (1-d)*b, p.default_loss*d*y
    equity = p.gross_rate*reserve+(1-d)*p.domestic_share*b
    capacity = p.leverage*equity
    unconstrained = max(a/p.gross_rate-p.credit_scale, 0.)
    loans = min(capacity, unconstrained)
    loan_rate = max(p.gross_rate, a/(p.credit_scale+loans))
    funding = max(loans-equity, 0.)
    safe = max(equity-loans, 0.)
    production = a*math.log1p(loans/p.credit_scale)
    firm_profit = production-loan_rate*loans
    dividend = loan_rate*loans+p.gross_rate*safe-p.gross_rate*funding
    c1 = y-loss-payment
    c2 = p.terminal_income+production+p.gross_rate*safe-p.gross_rate*funding
    value = math.log(c1)+p.beta*math.log(c2) if c1 > 0 and c2 > 0 else -math.inf
    return [y, a, float(d), payment, loss, equity, loans, loan_rate, funding, safe,
            production, firm_profit, dividend, c1, c2, value]


def audit(result: SovereignBankingResult, tolerance: float) -> dict[str, Any]:
    from puremacro.sovereign.banking import SovereignBankingParameters, _specification
    errors: dict[str, float] = {}

    def check(name: str, actual: Any, reference: Any) -> None:
        a, b = np.asarray(actual), np.asarray(reference)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            errors[name] = math.inf
        else:
            error = float(np.max(np.abs(a-b)/(1+np.abs(b)), initial=0.))
            errors[name] = max(errors.get(name, 0.), error)

    try:
        p = SovereignBankingParameters(**asdict(result.parameters))
        y, a, weights, grid = _specification(result.income, result.productivity, result.probabilities,
                                            result.debt_grid, result.price_selection, result.decision_tolerance)
        n, nm = len(y), 2**len(y)
        if result.candidates.shape != (len(grid), nm) or result.selected_indices.shape != grid.shape:
            raise ValueError("invalid candidate or selection shapes")
        if (result.candidates.dtype.kind not in "iu" or result.selected_indices.dtype.kind not in "iu"
                or not np.isin(result.candidates, [0, 1, 2]).all()
                or not isinstance(result.selected_index, (int, np.integer))
                or isinstance(result.selected_index, (bool, np.bool_))):
            raise ValueError("invalid discrete classifications or selected index")
        for e in result.equilibria:
            if (not isinstance(e.debt_index, (int, np.integer)) or not isinstance(e.mask, (int, np.integer))
                    or e.default.dtype.kind != "b" or e.default.shape != y.shape):
                raise ValueError("invalid equilibrium indices or default mask")
        check("probability_normalization", result.probabilities, weights)
        records = {(e.debt_index, e.mask): (i, e) for i, e in enumerate(result.equilibria)}
        if len(records) != len(result.equilibria):
            raise ValueError("duplicate equilibria")
        expected_keys = set()
        selected = []
        near_ties, max_regret = 0, 0.
        for bi, b in enumerate(grid):
            available = []
            for mask in range(nm):
                default = [(mask >> j) & 1 for j in range(n)]
                q = math.fsum(float(weights[j]) for j in range(n) if not default[j])/p.gross_rate
                if mask == 0: q = 1/p.gross_rate
                reserve = p.initial_equity-q*p.domestic_share*b
                code = 2 if reserve < 0 else 0
                if code != 2:
                    rep = [_state(p, float(y[j]), float(a[j]), float(b), q, 0) for j in range(n)]
                    loss = [_state(p, float(y[j]), float(a[j]), float(b), q, 1) for j in range(n)]
                    implied = [int(loss[j][-1] > rep[j][-1]+result.decision_tolerance) for j in range(n)]
                    if default == implied:
                        code = 1
                        key = (bi, mask)
                        expected_keys.add(key)
                        if key not in records:
                            raise ValueError("missing equilibrium")
                        index, e = records[key]
                        rows = np.array([loss[j] if default[j] else rep[j] for j in range(n)])
                        c0 = p.initial_income-p.initial_equity+q*b
                        value = math.log(c0)+p.beta*math.fsum(float(weights[j])*rows[j, -1] for j in range(n))
                        available.append((index, q, value))
                        check("price", e.price, q)
                        check("initial_consumption", e.initial_consumption, c0)
                        check("initial_reserves", e.initial_reserves, reserve)
                        check("initial_resources", e.initial_consumption+e.initial_reserves,
                              p.initial_income+(1-p.domestic_share)*q*b)
                        check("value", e.value, value)
                        check("default_mask", np.asarray(e.default, dtype=int), default)
                        check("accounts", e.accounts, rows)
                        if e.accounts.shape != (n, 16):
                            raise ValueError("invalid account shape")
                        for j, row in enumerate(e.accounts):
                            Y, A, d, payment, burn, E, L, r, D, S, F, profit, div, c1, c2, v = row
                            check("government_household_date1", c1+payment+burn, Y)
                            check("bank_balance_sheet", L+S, E+D)
                            check("bank_settlement", E, p.gross_rate*reserve+p.domestic_share*payment)
                            check("bank_dividend", div+p.gross_rate*D, r*L+p.gross_rate*S)
                            check("household_date2", c2, p.terminal_income+profit+div)
                            check("resources_date1", c1+L+S,
                                  Y-burn+p.gross_rate*reserve-(1-p.domestic_share)*payment+D)
                            check("resources_date2", c2+p.gross_rate*D, p.terminal_income+F+p.gross_rate*S)
                            check("firm_profit", profit+r*L, F)
                            check("credit_complementarity", (r-p.gross_rate)*(p.leverage*E-L), 0.)
                            check("credit_feasibility", max(0., L-p.leverage*E, -L, -D, -S, -E), 0.)
                            check("firm_credit_demand", L*max(0., r-A/(p.credit_scale+L)), 0.)
                            check("bank_rate_floor", max(0., p.gross_rate-r), 0.)
                            if min(c1, c2) <= 0:
                                raise ValueError("infeasible selected consumption")
                            alternative = rep[j][-1] if default[j] else loss[j][-1]
                            max_regret = max(max_regret, alternative-v)
                            gap = abs(rep[j][-1]-loss[j][-1])
                            if gap <= 10*max(result.decision_tolerance, np.finfo(float).eps):
                                near_ties += 1
                check("candidate_classification", int(result.candidates[bi, mask]), code)
            if available:
                select = max if result.price_selection == "highest" else min
                chosen = select(available, key=lambda item: item[1])
                selected.append(chosen)
                check("selection_at_debt", result.selected_indices[bi], chosen[0])
            else:
                check("selection_at_debt", result.selected_indices[bi], -1)
        if set(records) != expected_keys:
            raise ValueError("extra or invalid equilibria")
        optimum = max(selected, key=lambda item: (item[2], -result.equilibria[item[0]].debt_index))
        # A scalar/vector sum may differ by roundoff at exact debt-value ties.
        available_indices = [item[0] for item in selected]
        if result.selected_index not in available_indices:
            raise ValueError("selected equilibrium violates declared price selection")
        contract_optimum = max(available_indices, key=lambda i: (result.equilibria[i].value,
                                                                -result.equilibria[i].debt_index))
        if result.selected_index != contract_optimum:
            raise ValueError("selected debt violates utility maximization or tie rule")
        selected_value = next(item[2] for item in selected if item[0] == result.selected_index)
        check("debt_optimality", max(0., optimum[2]-selected_value), 0.)
        check("default_regret", max(0., max_regret-result.decision_tolerance), 0.)
    except (ValueError, TypeError, IndexError, OverflowError, ZeroDivisionError, AttributeError) as exc:
        return dict(accepted=False, max_error=None, reason=str(exc))
    maximum = max(errors.values(), default=0.)
    return dict(accepted=bool(maximum <= tolerance), max_error=maximum if math.isfinite(maximum) else None,
                checks={k: v if math.isfinite(v) else None for k, v in errors.items()},
                tolerance=tolerance, near_indifference_state_decisions=near_ties,
                maximum_default_regret=max_regret, independently_enumerated_candidates=len(grid)*nm)
