"""Independent primal expenditure minimization against Hicksian trade welfare.

The economic equilibrium oracle uses the scalar derivation from the accounting
benchmark. This script independently minimizes purchases of origin goods needed
to deliver target consumption utility, including local taxes and duties. It
does not call the production expenditure-function calculation inside the oracle.
"""
from __future__ import annotations

import argparse
import json
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from tools.reference_validation.validate_trade_accounting import benchmark_table, scalar_reference


def consumption_benchmark():
    """Split C into two distinct baskets, preserving the original IO margins."""
    original, tau, old_fd = benchmark_table()
    data = np.zeros((5, 8))
    data[:, :2] = original[:, :2]
    fd = np.ones((2, 3, 2))
    for c in range(2):
        consumption = original[:2, 2+2*c]
        fractions = np.array([.8, .3]) if c == 0 else np.array([.2, .7])
        data[:2, 2+3*c] = fractions*consumption
        data[:2, 3+3*c] = original[:2, 3+2*c]
        data[:2, 4+3*c] = (1-fractions)*consumption
        tax = original[2, 2+2*c]
        data[2, 2+3*c:5+3*c] = [.6*tax, original[2, 3+2*c], .4*tax]
        fd[:, 0, c] = old_fd[:, 0, c]
        fd[:, 1, c] = old_fd[:, 1, c]
        fd[:, 2, c] = old_fd[:, 0, c]
    # Distinct shocks across consumption baskets as well as countries.
    fd[1, 2, 0] = 1.5
    fd[0, 2, 1] = 1.05
    return data, tau, fd


def primal_expenditure(prices, baskets, weights, utility):
    """Numerically minimize origin-level purchaser spending at target utility.

    Variables: origin deliveries x[i,k] and Leontief quantities q[k]. Constraints
    x[i,k] >= a[i,k]*q[k], product(q[k]**weights[k]) >= target. Normalize quantities
    by target utility and prices by their maximum to stabilize SLSQP.
    """
    n, k = baskets.shape
    unit = float(np.max(prices))
    costs = prices/unit
    start_q = np.ones(k)*1.2
    start = np.r_[(baskets*start_q[None]).ravel(), start_q]

    def objective(z):
        return float(np.sum(costs*z[:n*k].reshape(n, k)))

    def objective_jac(z):
        return np.r_[costs.ravel(), np.zeros(k)]

    def constraints(z):
        q = z[n*k:]
        return np.r_[z[:n*k]-(baskets*q[None]).ravel(), weights@np.log(q)]

    solution = minimize(objective, start, jac=objective_jac, method="SLSQP",
                        bounds=[(0, None)]*(n*k)+[(1e-12, None)]*k,
                        constraints={"type": "ineq", "fun": constraints},
                        options={"ftol": 1e-12, "maxiter": 500})
    if not solution.success or np.min(constraints(solution.x)) < -1e-9:
        raise AssertionError(f"Independent expenditure minimization failed: {solution.message}")
    return float(solution.fun*unit*utility)


def reference_welfare(data, tau, fd, country, categories=(0, 2)):
    before = scalar_reference(data, np.ones_like(tau), np.ones_like(fd))
    after = scalar_reference(data, tau, fd)
    nfd = fd.shape[1]
    F = data[:2, 2:].reshape(2, 2, nfd).transpose(0, 2, 1)
    tax = data[2, 2:].reshape(2, nfd).T
    baskets = F[:, categories, country]/F[:, categories, country].sum(0)
    spending0 = F[:, :, country].sum(0)+tax[:, country]
    weights = spending0[list(categories)]/spending0[list(categories)].sum()
    delta = tax[list(categories), country]/spending0[list(categories)]
    prices0 = np.ones_like(baskets)/(1-delta)[None]
    prices1 = after["p"][:, None]*fd[:, categories, country]/(1-delta)[None]
    # Utility is evaluated from actual delivered origin goods, using Leontief
    # minima. It never uses the runtime's CPI, income deflator or welfare total.
    q0 = np.min(before["F"][:, categories, country]/baskets, axis=0)
    q1 = np.min(after["F"][:, categories, country]/baskets, axis=0)
    u0, u1 = np.prod(q0**weights), np.prod(q1**weights)
    e00 = primal_expenditure(prices0, baskets, weights, u0)
    e01 = primal_expenditure(prices0, baskets, weights, u1)
    e10 = primal_expenditure(prices1, baskets, weights, u0)
    e11 = primal_expenditure(prices1, baskets, weights, u1)
    # Enumerate all six changes of order for an independent Shapley check.
    share = spending0[list(categories)].sum()/before["income"][country]
    income0 = before["income"][country]-before["T"][country]
    income1 = after["income"][country]-after["T"][country]
    blocks0 = [1., income0, before["T"][country]]
    blocks1 = [e00/e10, income1, after["T"][country]]
    attribution = np.zeros(3)
    def value(blocks):
        return share*blocks[0]*(blocks[1]+blocks[2])
    for order in permutations(range(3)):
        blocks = blocks0.copy()
        for j in order:
            previous = value(blocks)
            blocks[j] = blocks1[j]
            attribution[j] += (value(blocks)-previous)/6
    return {"ev": e01-e00, "cv": e11-e10,
            "price_effect": attribution[0], "factor_income_effect": attribution[1],
            "fiscal_transfer_effect": attribution[2],
            "expenditure": [e00, e01, e10, e11]}


def validate():
    from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, compute_hicksian_welfare
    data, tau, fd = consumption_benchmark()
    calib = calibrate_trade_model(data, ns=1, nc=2, nfd=3, country_codes=["A", "B"])
    base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-13)
    counter = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent", tol=1e-13)
    cases = {}
    for c in range(2):
        result = compute_hicksian_welfare(calib, counter, base_result=base,
                                          target_country=c, consumption_categories=(0, 2))
        ref = reference_welfare(data, tau, fd, c)
        comparisons = {key: {"actual": getattr(result, key), "reference": ref[key],
                             "max_abs_error": float(abs(getattr(result, key)-ref[key])),
                             "passed": bool(np.isclose(getattr(result, key), ref[key], atol=1e-8, rtol=1e-8))}
                       for key in ("ev", "cv", "price_effect", "factor_income_effect", "fiscal_transfer_effect")}
        cases[result.country_code] = {"comparisons": comparisons,
                                      "decomposition_residual": result.decomposition_residual,
                                      "expenditure_oracle": ref["expenditure"],
                                      "passed": all(v["passed"] for v in comparisons.values())}
    return {"passed": all(v["passed"] for v in cases.values()), "cases": cases}


def validate_oecd_consumption():
    from puremacro.trade import solve_trade_equilibrium, compute_hicksian_welfare
    from puremacro.trade.data import package_mrio_to_calibration_result
    from puremacro.trade._oecd_icio import condense_final_demand
    from tools.reference_validation.validate_oecd import load_fixture
    calib = package_mrio_to_calibration_result(condense_final_demand(load_fixture()))
    base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-5)
    rates = np.array([.1, 0., 0.])
    cf = solve_trade_equilibrium(calib, tau=rates, tau_fd=rates, accounting="consistent", tol=1e-5)
    records = {}
    for i, code in enumerate(calib.country_codes):
        r = compute_hicksian_welfare(calib, cf, base_result=base, target_country=i)
        # One fixed-composition consumption basket: the Hicksian quantity change
        # can independently be costed directly at baseline purchaser prices.
        direct = float(base.Pfd_final[0, 0, i]*(cf.c_fd[0, 0, i]-base.c_fd[0, 0, i]))
        records[code] = {"ev": r.ev, "cv": r.cv, "ev_pct_consumption": r.ev_pct_consumption,
                         "ev_pct_gdp": r.ev_pct_gdp, "price_effect": r.price_effect,
                         "factor_income_effect": r.factor_income_effect,
                         "fiscal_transfer_effect": r.fiscal_transfer_effect,
                         "decomposition_residual": r.decomposition_residual,
                         "direct_quantity_ev_error": abs(r.ev-direct),
                         "passed": bool(np.isclose(r.ev, direct, atol=1e-7, rtol=1e-10))}
    return {"passed": all(row["passed"] for row in records.values()),
            "unit": calib.metadata.get("unit"), "consumption_categories": [0],
            "consumption_mapping": "HFCE + NPISH + GGFC (model aggregate, not household-only data)",
            "native_archive_reloaded_this_run": False, "countries": records}


def validate_ces_solver_boundary():
    from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, compute_hicksian_welfare
    data, tau, fd = consumption_benchmark()
    calib = calibrate_trade_model(data, ns=1, nc=2, nfd=3)
    base = solve_trade_equilibrium(calib, accounting="consistent", sigma=.5, tol=1e-10)
    records = {}
    values = []
    for method in ("newton", "hybr", "keller_pac"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cf = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent",
                                         sigma=.5, tol=1e-10, method=method, max_steps=30)
        row = {"converged": bool(cf.converged), "warnings": sorted(set(str(w.message) for w in caught))}
        try:
            welfare = compute_hicksian_welfare(calib, cf, base_result=base, consumption_categories=(0, 2))
            row.update(ev=welfare.ev, welfare_available=True, passed=bool(cf.converged))
            values.append(welfare.ev)
        except ValueError as error:
            row.update(welfare_available=False, reason=str(error), passed=not cf.converged)
        records[method] = row
    agreement = len(values) >= 2 and np.allclose(values, values[0], atol=1e-8, rtol=1e-8)
    return {"passed": bool(agreement and all(r["passed"] for r in records.values())),
            "description": "Strong heterogeneous shock, CES elasticity 0.5; failed states must have no welfare",
            "methods": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate()
    result["oecd_aggregation"] = validate_oecd_consumption()
    result["ces_solver_boundary"] = validate_ces_solver_boundary()
    result["passed"] = result["passed"] and result["oecd_aggregation"]["passed"] and result["ces_solver_boundary"]["passed"]
    content = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(content)
    print(content)
    raise SystemExit(0 if result["passed"] else 1)
