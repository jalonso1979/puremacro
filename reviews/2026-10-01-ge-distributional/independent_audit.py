"""Audit saved GE incidence evidence without importing application/GE/welfare code.

Run from the repository with .venv/bin/python reviews/.../independent_audit.py.
The only inputs are CSV, NPZ and manifest artifacts. Independent linear and
nonlinear expenditure minimization check both EV and CV for every scenario.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "application"
PACKAGE = ROOT.parents[1] / "puremacro"


def main():
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    for name, entry in manifest["artifacts"].items():
        payload = (ARTIFACTS / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"], name
        assert len(payload) == entry["bytes"], name
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == digest, name
    survey = pd.read_csv(ARTIFACTS / "observed_enigh_deciles.csv", index_col=0)
    incidence = pd.read_csv(ARTIFACTS / "all_scenarios.csv")
    mapping = pd.read_csv(ARTIFACTS / "income_mapping.csv", index_col=0)
    prices = pd.read_csv(ARTIFACTS / "purchaser_prices.csv", index_col=0)
    fiscal = pd.read_csv(ARTIFACTS / "fiscal_allocation.csv")
    aggregate = pd.read_csv(ARTIFACTS / "aggregate_results.csv", index_col=0)
    revenues = pd.read_csv(ARTIFACTS / "revenue_audit.csv")
    with np.load(ARTIFACTS / "equilibrium_evidence.npz", allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    sectors = list(manifest["observed_data"]["categories"])
    ns, nc = len(sectors), 2
    raw = arrays["calibration_table"]
    n = ns * nc
    counts = survey.households.to_numpy()
    spending = survey[sectors].to_numpy()
    budgets = spending.sum(axis=1)
    totals = counts @ spending
    total = float(counts @ budgets)
    shares = totals / total
    scale = manifest["units"]["mxn_per_model_unit"]
    tariff_rate = manifest["tariff_rate"]
    labor = np.array([raw[n+1, c*ns:(c+1)*ns].sum() for c in range(nc)])
    capital = np.array([raw[n+2, c*ns:(c+1)*ns].sum() for c in range(nc)])
    labor_share = float(counts @ survey.cash_wages / total)
    exposures = np.column_stack([survey.cash_wages, (1-labor_share) * budgets])
    np.testing.assert_allclose(exposures, mapping[["cash_wages", "other_factor_proxy"]], rtol=1e-13)
    np.testing.assert_allclose(raw[:n].sum(axis=1), raw[:, :n].sum(axis=0), atol=2e-13)
    state_audit = {}
    purchaser = {}
    for state in ("baseline", "counterfactual"):
        p = arrays[f"{state}_producer_prices"][0]
        inputs = arrays[f"{state}_intermediate_quantities"]
        final = arrays[f"{state}_final_quantities"]
        ti = (np.ones_like(arrays["intermediate_tariff_multipliers"]) if state == "baseline" else
              arrays["intermediate_tariff_multipliers"])
        tf = (np.ones_like(arrays["final_tariff_multipliers"]) if state == "baseline" else
              arrays["final_tariff_multipliers"])
        price = np.zeros((ns, nc))
        model_budget, duties_i, duties_f = np.zeros(nc), np.zeros(nc), np.zeros(nc)
        for destination in range(nc):
            for sector in range(ns):
                base_origin_amount = np.array([raw[origin*ns+sector, n+destination] for origin in range(nc)])
                origin_weights = base_origin_amount / base_origin_amount.sum()
                for origin in range(nc):
                    seller = origin*ns + sector
                    price[sector, destination] += origin_weights[origin] * p[sector, origin] * tf[seller, 0, destination]
                    final_value = final[sector, origin, 0, destination] * p[sector, origin]
                    model_budget[destination] += final_value * tf[seller, 0, destination]
                    duties_f[destination] += final_value * (tf[seller, 0, destination]-1)
                    for destination_sector in range(ns):
                        input_value = inputs[sector, origin, destination_sector, destination] * p[sector, origin]
                        duties_i[destination] += input_value * (ti[seller, destination_sector, destination]-1)
        wage = arrays[f"{state}_wages"].ravel()
        rent = arrays[f"{state}_rental_returns"].ravel()
        factor_income = wage*labor + rent*capital
        np.testing.assert_allclose(model_budget, factor_income+duties_i+duties_f, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(price[:, 0], prices[state], rtol=1e-13)
        exported = revenues.loc[revenues.state == state].set_index("country").loc[["HOME", "REST"]]
        np.testing.assert_allclose(duties_i+duties_f, exported.reconstructed_duties, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(duties_i+duties_f, exported.government_revenue, rtol=1e-10, atol=1e-10)
        purchaser[state] = price
        state_audit[state] = {"purchaser_budget_model_units": model_budget.tolist(),
            "factor_income_model_units": factor_income.tolist(),
            "intermediate_duties_model_units": duties_i.tolist(), "final_duties_model_units": duties_f.tolist(),
            "maximum_budget_residual_model_units": float(np.max(np.abs(model_budget-factor_income-duties_i-duties_f)))}
    # A second, closed-form oracle uses only the declared synthetic technology.
    factor_loss = (.35/.65)*.20*shares[sectors.index("food")]*tariff_rate
    for name in ("counterfactual_wages", "counterfactual_rental_returns"):
        np.testing.assert_allclose(arrays[name].ravel(), [1-factor_loss, 1.], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(arrays["counterfactual_producer_prices"], 1., atol=1e-10)
    np.testing.assert_allclose(purchaser["counterfactual"][:, 0],
        1+np.eye(1, ns, sectors.index("food")).ravel()*.25*tariff_rate, atol=1e-10)
    np.testing.assert_allclose(state_audit["counterfactual"]["intermediate_duties_model_units"],
                              [100*factor_loss, 0.], atol=1e-10)
    pool = scale * (sum(state_audit["counterfactual"][key][0] for key in
                         ("intermediate_duties_model_units", "final_duties_model_units")) -
                    sum(state_audit["baseline"][key][0] for key in
                         ("intermediate_duties_model_units", "final_duties_model_units")))
    np.testing.assert_allclose(pool, manifest["fiscal"]["incremental_tariff_revenue_mxn"], rtol=1e-12)
    wage_change = arrays["counterfactual_wages"].ravel()[0] / arrays["baseline_wages"].ravel()[0] - 1
    rent_change = arrays["counterfactual_rental_returns"].ravel()[0] / arrays["baseline_rental_returns"].ravel()[0] - 1
    income_change = exposures @ np.array([wage_change, rent_change])
    np.testing.assert_allclose(income_change, mapping.total_factor_income_change, rtol=1e-12)
    np.testing.assert_allclose(counts @ exposures, scale*np.array([labor[0], capital[0]]), rtol=1e-12)
    np.testing.assert_allclose(total, scale*state_audit["baseline"]["purchaser_budget_model_units"][0], rtol=1e-12)
    mapped_final = total + counts @ income_change + pool
    model_final = scale*state_audit["counterfactual"]["purchaser_budget_model_units"][0]
    np.testing.assert_allclose(mapped_final, model_final, rtol=1e-10)
    relative = purchaser["counterfactual"][:, 0] / purchaser["baseline"][:, 0]
    scenario_audit = []
    for (preference, rule), original in incidence.groupby(["preferences", "rule"]):
        original = original.set_index("group").loc[survey.index]
        transfer = np.zeros(len(survey))
        if rule == "equal_per_household":
            transfer[:] = pool/counts.sum()
        elif rule == "bottom_four_deciles":
            transfer[:4] = pool/counts[:4].sum()
        retained = pool-float(counts @ transfer)
        final_budgets = budgets + income_change + transfer
        np.testing.assert_allclose(original.income_change, income_change, rtol=1e-12)
        np.testing.assert_allclose(original.transfer, transfer, rtol=1e-12)
        np.testing.assert_allclose(counts @ final_budgets + retained, model_final, rtol=1e-10)
        analytic_ev, analytic_cv, numerical_ev, numerical_cv = [], [], [], []
        for i in range(len(survey)):
            weights = spending[i] / budgets[i]
            cost = weights @ relative if preference == "fixed_baskets" else np.exp(weights @ np.log(relative))
            analytic_ev.append(final_budgets[i]/cost - budgets[i])
            analytic_cv.append(final_budgets[i] - budgets[i]*cost)
            if preference == "fixed_baskets":
                utility1 = final_budgets[i]/(spending[i] @ relative)
                dual0 = linprog(np.ones(ns), bounds=list(zip(spending[i]*utility1, [None]*ns)), method="highs")
                dual1 = linprog(relative, bounds=list(zip(spending[i], [None]*ns)), method="highs")
                assert dual0.success and dual1.success
                e0, e1 = dual0.fun, dual1.fun
            else:
                q1 = weights*final_budgets[i]/relative
                log_u1 = float(weights @ np.log(q1/spending[i]))
                def expenditure(p, log_utility):
                    costs = p*weights
                    fit = minimize(lambda z: float(costs @ np.exp(z)), np.full(ns, log_utility),
                        jac=lambda z: costs*np.exp(z), method="SLSQP",
                        constraints={"type": "ineq", "fun": lambda z: float(weights @ z-log_utility),
                                     "jac": lambda z: weights},
                        options={"ftol": 1e-13, "maxiter": 300})
                    assert fit.success, fit.message
                    assert weights @ fit.x >= log_utility-1e-11
                    return budgets[i]*fit.fun
                e0, e1 = expenditure(np.ones(ns), log_u1), expenditure(relative, 0.)
            numerical_ev.append(e0-budgets[i])
            numerical_cv.append(final_budgets[i]-e1)
        np.testing.assert_allclose(original.ev, analytic_ev, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(original.cv, analytic_cv, rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(original.ev, numerical_ev, rtol=1e-8, atol=1e-6)
        np.testing.assert_allclose(original.cv, numerical_cv, rtol=1e-8, atol=1e-6)
        for welfare in ("ev", "cv"):
            np.testing.assert_allclose(original[[f"price_{welfare}", f"income_{welfare}", f"transfer_{welfare}"]].sum(axis=1),
                                      original[welfare], atol=1e-9)
            np.testing.assert_allclose(aggregate.loc[f"{preference}_{rule}", f"total_{welfare}"],
                                      counts @ original[welfare], rtol=1e-10, atol=.001)
        fiscal_row = fiscal.loc[(fiscal.preferences == preference) & (fiscal.rule == rule)].iloc[0]
        np.testing.assert_allclose(fiscal_row.household_rebate_mxn, counts @ transfer, rtol=1e-12)
        np.testing.assert_allclose(fiscal_row.retained_outside_households_mxn, retained, atol=1e-5)
        scenario_audit.append({"preferences": preference, "rule": rule,
            "maximum_ev_analytic_error_mxn": float(np.max(np.abs(original.ev-np.asarray(analytic_ev)))),
            "maximum_cv_analytic_error_mxn": float(np.max(np.abs(original.cv-np.asarray(analytic_cv)))),
            "maximum_ev_numerical_error_mxn": float(np.max(np.abs(original.ev-np.asarray(numerical_ev)))),
            "maximum_cv_numerical_error_mxn": float(np.max(np.abs(original.cv-np.asarray(numerical_cv))))})
    result = {"status": "passed", "independence": "artifact-only calculation; no imports of puremacro solvers, application or household welfare code",
        "artifact_hashes_verified": len(manifest["artifacts"]), "source_hashes_verified": len(manifest["source_sha256"]),
        "states": state_audit, "closed_form_factor_return_change": -factor_loss,
        "incremental_tariff_revenue_mxn": pool,
        "aggregate_baseline_income_minus_consumption_mxn": float(counts @ (exposures.sum(axis=1)-budgets)),
        "counterfactual_survey_minus_model_budget_mxn": float(mapped_final-model_final),
        "counterfactual_budget_relative_error": float(abs(mapped_final-model_final)/model_final),
        "scenarios": scenario_audit,
        "limitations": ["The no-rebate case retains revenue at the solved full-rebate GE prices; it is not another GE closure",
                        "Even full household rebates change group spending without feeding that demand back into GE",
                        "Other-factor exposures and common import-origin baskets are assumptions, not observed household accounts",
                        "EV and CV are monetary incidence measures; summing them does not identify a social welfare function"]}
    (ROOT / "independent_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
