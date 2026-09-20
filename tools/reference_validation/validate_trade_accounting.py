"""Independent scalar two-country oracle for producer/purchaser accounting.

At one sector per country, fixed factor endowments and Leontief top-level
production fix outputs. Intermediate origin sourcing can be Leontief (sigma=0)
or CES. For each relative producer price, solve household incomes algebraically
including fiscal rebates, then clear country 2's goods market with Brent's
method. This oracle never calls puremacro's residual evaluator or GE solver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq


def benchmark_table():
    Z = np.array([[10., 12.], [8., 14.]])
    F = np.array([[30., 8., 20., 20.], [15., 15., 50., 18.]])
    production_tax = np.array([4., 6.])
    final_tax = np.array([2., 1., 3., 2.])
    output = Z.sum(1)+F.sum(1)
    va = output-Z.sum(0)-production_tax
    data = np.vstack([np.hstack([Z, F]), np.r_[production_tax, final_tax],
                      np.r_[2*va/3, np.zeros(4)], np.r_[va/3, np.zeros(4)]])
    tau = np.ones((2, 1, 2)); tau[1, 0, 0] = 1.2; tau[0, 0, 1] = 1.1
    fd = np.ones((2, 2, 2)); fd[1, :, 0] = [1.3, 1.6]; fd[0, :, 1] = [1.1, 1.2]
    return data, tau, fd


def scalar_reference(data, tau, tau_fd, sigma=0.):
    nfd = tau_fd.shape[1]
    Z, raw_F = data[:2, :2], data[:2, 2:]
    y = Z.sum(1)+raw_F.sum(1)
    labor, capital = data[3, :2], data[4, :2]
    output_tax = data[2, :2]/y
    F = raw_F.reshape(2, 2, nfd).transpose(0, 2, 1)  # origin, category, destination
    a = F/F.sum(0)[None]
    final_tax0 = data[2, 2:].reshape(2, nfd).T
    spending0 = F.sum(0)+final_tax0
    tax_share = final_tax0/spending0
    # Compute baseline foreign saving directly from cross-border transactions.
    B0 = Z[0, 1]+F[0, :, 1].sum()-Z[1, 0]-F[1, :, 0].sum()
    B = np.array([B0, -B0])
    income0 = labor+capital+data[2, :2]+final_tax0.sum(0)
    theta = spending0.copy(); theta[1] += B; theta /= income0[None]

    def at_price(relative):
        p = np.array([1., relative])
        deliveries = Z
        if sigma != 0:
            purchaser = p[:, None]*tau[:, 0, :]
            weights = Z/Z.sum(0)[None]
            index = (np.exp(np.sum(weights*np.log(purchaser), axis=0)) if sigma == 1
                     else np.sum(weights*purchaser**(1-sigma), axis=0)**(1/(1-sigma)))
            deliveries = Z*(index[None]/purchaser)**sigma
        Q = np.sum(p[:, None, None]*tau_fd*a, axis=0)
        tariff_fraction = np.sum(p[:, None, None]*(tau_fd-1)*a, axis=0)/Q
        h = tax_share+(1-tax_share)*tariff_fraction
        available = p*y-(p[:, None]*deliveries).sum(0)
        income = (available-h[1]*B)/(1-np.sum(h*theta, axis=0))
        expenditure = theta*income[None]; expenditure[1] -= B
        quantities = (1-tax_share)*expenditure/Q
        final = a*quantities[None]
        va_income = (1-output_tax)*p*y-(p[:, None]*tau[:, 0, :]*deliveries).sum(0)
        wages, rents = (2/3)*va_income/labor, (1/3)*va_income/capital
        duties_i = ((tau[:, 0, :]-1)*p[:, None]*deliveries).sum(0)
        duties_f = ((tau_fd-1)*p[:, None, None]*final).sum(axis=(0, 1))
        domestic_tax = output_tax*p*y+(tax_share*expenditure).sum(0)
        return dict(p=p, y=y, w=wages, r=rents, income=income, expenditure=expenditure,
                    Q=Q, P=Q/(1-tax_share), F=final, Z=deliveries,
                    tariffs=duties_i+duties_f, T=domestic_tax+duties_i+duties_f,
                    B=B, goods_gap=y-deliveries.sum(1)-final.sum(axis=(1, 2)))

    root = brentq(lambda p: at_price(p)["goods_gap"][1], .2, 5., xtol=1e-14)
    result = at_price(root)
    if np.max(np.abs(result["goods_gap"])) > 1e-10:
        raise AssertionError("Independent scalar oracle violates goods clearing")
    return result


def validate():
    from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium

    data, tau, fd = benchmark_table()
    calibration = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"], sector_codes=["GOOD"])
    report = {}
    for name, ti, tf in (("baseline", np.ones_like(tau), np.ones_like(fd)),
                         ("heterogeneous_tariffs", tau, fd)):
        oracle = scalar_reference(data, ti, tf)
        result = solve_trade_equilibrium(calibration, tau=ti, tau_fd=tf, accounting="consistent", tol=1e-12, max_iter=100)
        pairs = {"producer_prices": (result.p_sol.ravel(), oracle["p"]),
                 "outputs": (result.y_sol.ravel(), oracle["y"]),
                 "wages": (result.w_sol.ravel(), oracle["w"]),
                 "rents": (result.r_sol.ravel(), oracle["r"]),
                 "rebates": (result.T_sol.ravel(), oracle["T"]),
                 "tariffs": (result.tariffs, oracle["tariffs"]),
                 "purchaser_prices": (result.Pfd_final[0], oracle["P"]),
                 "final_deliveries": (result.final_demand_flows[0], oracle["F"]),
                 "income": (result.metadata["household_income"].ravel(), oracle["income"])}
        comparisons = {field: {"max_abs_error": float(np.max(np.abs(actual-expected))),
                               "passed": bool(np.allclose(actual, expected, rtol=1e-10, atol=1e-10))}
                       for field, (actual, expected) in pairs.items()}
        report[name] = {"converged": bool(result.converged), "comparisons": comparisons,
                        "accounts": result.metadata["account_residuals"],
                        "producer_prices": oracle["p"].tolist(), "tariff_revenue": oracle["tariffs"].tolist(),
                        "passed": bool(result.converged and all(c["passed"] for c in comparisons.values()))}
    return {"passed": all(r["passed"] for r in report.values()), "cases": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate()
    content = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(content)
    print(content)
    raise SystemExit(0 if result["passed"] else 1)
