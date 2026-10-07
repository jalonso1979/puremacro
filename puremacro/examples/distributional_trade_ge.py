"""Synthetic GE tariff experiment with observed ENIGH 2024 household baskets.

Run ``python -m puremacro.examples.distributional_trade_ge --output DIR``.
The production technology, import sourcing and tariff are scenario assumptions;
only the household baskets, expansion counts and cash wages are observed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from puremacro import __version__
from puremacro.datasets.enigh import load_enigh2024_deciles
from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium
from puremacro.trade.distributional import (
    distributional_welfare_from_results, prepare_household_groups,
)
from puremacro.trade.household import household_prices_from_result


def _calibration(data):
    """Construct a balanced, symmetric two-region economy, not an IO estimate."""
    sectors = list(data.attrs["categories"])
    total = data[sectors].T @ data.households
    shares = total.to_numpy() / total.sum()
    labor_share = float(data.households @ data.cash_wages / total.sum())
    if not 0 < labor_share < 1:
        raise ValueError("Observed cash wages must lie strictly between zero and consumption for this scenario")
    ns, nc, nfd = len(sectors), 2, 1
    n = ns * nc
    # These are scenario choices for broad consumption composites, not measured
    # industry import shares or an ENIGH-to-industry concordance.
    imports = np.array([.25, .55, .05, .35, .15, .40, .05, .15])
    raw = np.zeros((n + 3, n + nfd * nc))
    gross_output = 100 * shares / .65
    for destination in range(nc):
        for origin in range(nc):
            origin_slice = slice(origin * ns, (origin + 1) * ns)
            sourcing = .8 if origin == destination else .2
            raw[origin_slice, destination * ns:(destination + 1) * ns] = (
                .35 * sourcing * shares[:, None] * gross_output[None, :])
            raw[origin_slice, n + destination] = 100 * shares * (
                1 - imports if origin == destination else imports)
    raw[n + 1, :n] = np.tile(labor_share * .65 * gross_output, nc)
    raw[n + 2, :n] = np.tile((1 - labor_share) * .65 * gross_output, nc)
    calibration = calibrate_trade_model(
        raw, nc=nc, ns=ns, nfd=nfd, country_codes=["HOME", "REST"], sector_codes=sectors)
    assumptions = {
        "source": "synthetic balanced two-region scenario using observed ENIGH spending shares",
        "is_synthetic": True, "is_regression_fixture": True,
        "empirically_calibrated_trade_economy": False,
        "countries": "HOME and REST are hypothetical symmetric regions, not estimated Mexican/foreign accounts",
        "baseline_consumption_per_region_model_units": 100.,
        "intermediate_cost_share": .35, "intermediate_import_share": .20,
        "final_import_shares": dict(zip(sectors, imports.tolist())),
        "labor_share": labor_share,
        "labor_share_construction": "observed aggregate cash wages divided by observed monetary consumption; a scenario normalization, not a national-accounts factor share",
        "production": "fixed intermediate coefficients and Cobb-Douglas labor/capital value added",
        "final_demand": "one fixed-origin, fixed-sector consumption category; no investment or government category",
        "foreign_saving": "zero in both regions", "baseline_taxes_and_tariffs": "zero",
        "numeraire": "HOME food producer price equals one in both states",
    }
    calibration.metadata.update(assumptions)
    return calibration, raw, assumptions


def _duties(result, calibration, state):
    """Reconstruct duties from physical transactions and producer prices."""
    ns, nc = calibration.ns, calibration.nc
    n = ns * nc
    p = result.p_sol.ravel(order="F")
    intermediate = np.asarray(result.intermediate_flows).reshape(n, n, order="F")
    final = np.asarray(result.final_demand_flows).reshape(n, nc, order="F")
    ti = result.metadata["intermediate_tariff_multipliers"].reshape(n, n, order="F")
    tf = result.metadata["final_tariff_multipliers"].reshape(n, nc, order="F")
    rows = []
    for kind, quantities, schedule, uses in (
        ("intermediate", intermediate, ti, ns), ("final", final, tf, 1),
    ):
        for origin in range(nc):
            for sector in range(ns):
                i = origin * ns + sector
                for destination in range(nc):
                    for use in range(uses):
                        j = destination * uses + use
                        rate = float(schedule[i, j] - 1)
                        value = float(p[i] * quantities[i, j])
                        rows.append({
                            "state": state, "use": kind,
                            "origin": calibration.country_codes[origin],
                            "sector": calibration.sector_codes[sector],
                            "destination": calibration.country_codes[destination],
                            "destination_use": calibration.sector_codes[use] if kind == "intermediate" else "consumption",
                            "quantity": float(quantities[i, j]), "producer_price": float(p[i]),
                            "producer_value": value, "tariff_rate": rate,
                            "duty_model_units": rate * value,
                        })
    return pd.DataFrame(rows)


def _close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-8, atol=1e-8):
        raise RuntimeError(f"{label} failed reconciliation")


def run_application(output: Path, *, tariff_rate: float = .10):
    """Solve and export six conditional incidence scenarios with a fiscal ledger.

    The same solved GE prices and factor returns are held fixed across the
    household transfer and preference comparisons. The income proxy and the
    model-to-MXN scale are explicit assumptions. No household feedback is solved.
    """
    if not np.isfinite(tariff_rate) or tariff_rate < 0:
        raise ValueError("tariff_rate must be finite and nonnegative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_enigh2024_deciles()
    sectors = list(data.attrs["categories"])
    calibration, raw, assumptions = _calibration(data)
    base = solve_trade_equilibrium(calibration, accounting="consistent", tol=1e-10)
    ns, nc = calibration.ns, calibration.nc
    tau = np.ones((ns * nc, ns, nc))
    tau_fd = np.ones((ns * nc, 1, nc))
    # HOME levies the tariff only on REST food, for all intermediate uses and
    # the final consumption category; domestic transactions remain untaxed.
    tau[ns + sectors.index("food"), :, 0] += tariff_rate
    tau_fd[ns + sectors.index("food"), 0, 0] += tariff_rate
    cf = solve_trade_equilibrium(calibration, accounting="consistent", tau=tau,
                                 tau_fd=tau_fd, tol=1e-10)
    if not base.converged or not cf.converged:
        raise RuntimeError("Both GE states must converge before household incidence is evaluated")
    p0, budget0 = household_prices_from_result(base, calibration)
    p1, budget1 = household_prices_from_result(cf, calibration)
    expenditure = data[sectors].sum(axis=1)
    observed_total = float(data.households @ expenditure)
    scale = observed_total / float(budget0[0])
    other_share = 1 - assumptions["labor_share"]
    exposure = pd.DataFrame({"cash_wages": data.cash_wages,
                             "other_factor_proxy": other_share * expenditure})
    changes = pd.Series({"cash_wages": float(cf.w_sol.ravel()[0] / base.w_sol.ravel()[0] - 1),
                         "other_factor_proxy": float(cf.r_sol.ravel()[0] / base.r_sol.ravel()[0] - 1)})
    groups = prepare_household_groups(
        data[sectors], data.households, monetary_unit="MXN", period="quarter",
        income_exposure=exposure, provenance=dict(data.attrs),
    )
    duties = pd.concat([_duties(base, calibration, "baseline"),
                        _duties(cf, calibration, "counterfactual")], ignore_index=True)
    revenue_rows = []
    for name, result in (("baseline", base), ("counterfactual", cf)):
        reconstructed = duties[duties.state == name].groupby("destination").duty_model_units.sum()
        for country, index in zip(calibration.country_codes, range(nc)):
            receipt = float(reconstructed[country])
            reported = float(result.tariffs[index])
            government = float(result.T_sol.ravel()[index])
            _close(receipt, reported, f"{name} {country} bilateral tariff receipts")
            _close(receipt, government, f"{name} {country} zero-other-tax government budget")
            revenue_rows.append({"state": name, "country": country,
                                 "reconstructed_duties": receipt, "reported_tariffs": reported,
                                 "government_revenue": government,
                                 "duty_reconciliation_residual": receipt - reported,
                                 "government_reconciliation_residual": receipt - government})
    revenue_audit = pd.DataFrame(revenue_rows)
    base_receipt = revenue_audit.query("state == 'baseline' and country == 'HOME'").reconstructed_duties.iloc[0]
    cf_receipt = revenue_audit.query("state == 'counterfactual' and country == 'HOME'").reconstructed_duties.iloc[0]
    pool = float((cf_receipt - base_receipt) * scale)
    income_audit = exposure.copy()
    income_audit["baseline_consumption"] = expenditure
    income_audit["baseline_income_minus_consumption"] = exposure.sum(axis=1) - expenditure
    income_audit["cash_wage_change"] = exposure.cash_wages * changes.cash_wages
    income_audit["other_factor_change"] = exposure.other_factor_proxy * changes.other_factor_proxy
    income_audit["total_factor_income_change"] = exposure @ changes
    income_audit["households"] = data.households
    _close(data.households @ exposure.cash_wages,
           scale * base.w_sol.ravel()[0] * calibration.l_endow.ravel()[0], "baseline labor income")
    _close(data.households @ exposure.other_factor_proxy,
           scale * base.r_sol.ravel()[0] * calibration.k_endow.ravel()[0], "baseline other-factor proxy")
    _close(float(data.households @ exposure.sum(axis=1)), observed_total, "baseline aggregate income/spending")
    factor_change = float(data.households @ income_audit.total_factor_income_change)
    ge_income_change = float(scale * (budget1[0] - budget0[0]))
    _close(observed_total + factor_change + pool, scale * budget1[0], "counterfactual aggregate household budget")
    targeting = pd.Series([1., 1., 1., 1., 0., 0., 0., 0., 0., 0.], index=data.index)
    transfer_rules = {"no_household_rebate": (0., None),
                      "equal_per_household": (pool, None),
                      "bottom_four_deciles": (pool, targeting)}
    results, frames, fiscal_rows = {}, [], []
    for preference in ("fixed_baskets", "cobb_douglas"):
        for rule, (transfer, scores) in transfer_rules.items():
            result = distributional_welfare_from_results(
                groups, calibration, base, cf, country="HOME", factor_income_changes=changes,
                transfer_total=transfer, transfer_weights=scores, rule=preference)
            actual = float(data.households @ result.groups.transfer)
            retained = pool - transfer
            _close(actual + retained, pool, f"{rule} fiscal allocation")
            household_budget_change = float(data.households @ (result.groups.income_change + result.groups.transfer))
            _close(observed_total + household_budget_change + retained,
                   scale * budget1[0], f"{rule} household plus retained resources")
            fiscal_rows.append({
                "preferences": preference, "rule": rule, "incremental_tariff_revenue_mxn": pool,
                "household_rebate_mxn": actual, "retained_outside_households_mxn": retained,
                "fiscal_residual_mxn": actual + retained - pool,
                "factor_income_change_mxn": factor_change,
                "household_budget_change_mxn": household_budget_change,
                "scaled_ge_budget_change_mxn": ge_income_change,
                "income_residual_mxn": household_budget_change + retained - ge_income_change,
            })
            key = f"{preference}_{rule}"
            results[key] = result
            frame = result.to_frame().reset_index()
            frame.insert(0, "rule", rule)
            frame.insert(0, "preferences", preference)
            frames.append(frame)
    incidence = pd.concat(frames, ignore_index=True)
    fiscal = pd.DataFrame(fiscal_rows)
    aggregate = pd.DataFrame({key: value.aggregate for key, value in results.items()}).T
    prices = pd.DataFrame({"baseline": p0[:, 0], "counterfactual": p1[:, 0],
                           "change_pct": 100 * (p1[:, 0] / p0[:, 0] - 1)}, index=sectors)
    tables = {"observed_enigh_deciles": data, "synthetic_calibration_table": pd.DataFrame(raw),
              "all_scenarios": incidence, "aggregate_results": aggregate,
              "purchaser_prices": prices, "income_mapping": income_audit,
              "tariff_transactions": duties, "revenue_audit": revenue_audit, "fiscal_allocation": fiscal}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=name not in ("all_scenarios", "tariff_transactions", "revenue_audit", "fiscal_allocation"))
    np.savez_compressed(output / "equilibrium_evidence.npz", calibration_table=raw,
                        baseline_solution=base.x_sol, counterfactual_solution=cf.x_sol,
                        baseline_wages=base.w_sol, counterfactual_wages=cf.w_sol,
                        baseline_rental_returns=base.r_sol, counterfactual_rental_returns=cf.r_sol,
                        baseline_producer_prices=base.p_sol, counterfactual_producer_prices=cf.p_sol,
                        baseline_intermediate_quantities=base.intermediate_flows,
                        counterfactual_intermediate_quantities=cf.intermediate_flows,
                        baseline_final_quantities=base.final_demand_flows,
                        counterfactual_final_quantities=cf.final_demand_flows,
                        intermediate_tariff_multipliers=tau, final_tariff_multipliers=tau_fd)
    manifest = {
        "application": "synthetic GE tariff scenario with observed ENIGH baskets",
        "package_version": __version__, "observed_data": dict(data.attrs),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "is_synthetic": True, "is_regression_fixture": True,
        "model_assumptions": assumptions, "tariff_rate": tariff_rate,
        "policy": "HOME tariff on REST food inputs and final consumption; zero domestic tariffs",
        "units": {"households": "MXN per household per quarter", "model": "synthetic value units",
                  "mxn_per_model_unit": scale,
                  "conversion": "survey aggregate baseline consumption / HOME model baseline consumption; a normalization, not an exchange-rate estimate"},
        "income_mapping": {
            "cash_wages": "observed per-household cash wages times HOME GE wage change",
            "other_factor_proxy": "unobserved proxy allocated proportional to consumption, total chosen to reconcile remaining GE factor income; times HOME GE rental-rate change",
            "nominal_changes": changes.to_dict(),
            "baseline_group_residual": "income exposures minus consumption may be positive or negative by group; held fixed, sum to zero nationally",
            "spending_response": "all incremental nominal factor income is spent",
        },
        "fiscal": {"incremental_tariff_revenue_mxn": pool,
                   "revenue_source": "counterfactual minus baseline sum of tariff rate times solved producer value, across intermediate and final imports",
                   "no_double_counting": "income exposure contains factor earnings only; the already solved GE tariff receipt is distributed once through the transfer argument",
                   "no_household_rebate": "conditional benchmark retains receipts outside households; it is not a separately solved government-saving closure"},
        "ge_audit": {"baseline": base.metadata["account_residuals"],
                     "counterfactual": cf.metadata["account_residuals"],
                     "both_converged": True, "solve_tolerance": 1e-10, "accounting": "consistent"},
        "limitations": [
            "Synthetic GE technology and trade sourcing are not an empirical Mexican tariff model; survey data alone do not identify them",
            "Observed broad consumption categories are used as hypothetical model composites, not an estimated industry concordance",
            "Other-factor exposure and the labor-share normalization are modeling assumptions, not observed national accounts or survey capital income",
            "One solved GE price/income pair is held fixed across household preferences and allocations; household demand does not feed back into equilibrium",
            "All deciles inherit identical origin shares within each composite",
            "Fixed baskets and Cobb-Douglas are preference assumptions; no elasticity is estimated",
            "Group means omit within-decile heterogeneity and do not support sampling confidence intervals",
            "Summed monetary EV is not a social welfare function",
        ],
    }
    _report(output, incidence, aggregate, prices, fiscal, manifest)
    package = Path(__file__).resolve().parents[1]
    source_paths = ["examples/distributional_trade_ge.py", "datasets/enigh.py",
                    "trade/calibration.py", "trade/solver.py", "trade/_accounting.py",
                    "trade/household.py", "trade/distributional.py"]
    manifest["source_sha256"] = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                                for name in source_paths}
    artifacts = [output / f"{name}.csv" for name in tables]
    artifacts += [output / name for name in ("equilibrium_evidence.npz", "report.md", "conditional_incidence.png")]
    manifest["artifacts"] = {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "bytes": path.stat().st_size}
                             for path in sorted(artifacts)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"results": results, "calibration": calibration, "baseline": base, "counterfactual": cf,
            "revenue_audit": revenue_audit, "fiscal_allocation": fiscal, "manifest": manifest}


def _report(output, incidence, aggregate, prices, fiscal, manifest):
    import matplotlib.pyplot as plt
    from puremacro.reports import df_to_markdown

    order = list(load_enigh2024_deciles().index)
    fixed = incidence[incidence.preferences == "fixed_baskets"].pivot(index="group", columns="rule", values="ev_pct").reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labeled_prices = prices.change_pct.copy()
    labeled_prices.index = [name.replace("_", " ").capitalize() for name in prices.index]
    labeled_prices.plot.barh(ax=axes[0], color="#256D85")
    axes[0].set(title="Synthetic GE purchaser-price changes", xlabel="Percent", ylabel="Consumption composite")
    fixed.rename(columns={"bottom_four_deciles": "Bottom four deciles",
                          "equal_per_household": "Equal per household",
                          "no_household_rebate": "No household rebate"}).plot(ax=axes[1], marker="o")
    axes[1].axhline(0, color="gray", linewidth=.7)
    axes[1].set(title="Conditional incidence, observed ENIGH baskets", ylabel="EV (% of baseline consumption)", xlabel="Income decile")
    axes[1].set_xticks(np.arange(len(order)), order)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "conditional_incidence.png", dpi=170)
    plt.close(fig)
    text = (
        "# Synthetic GE tariff scenario with observed ENIGH baskets\n\n"
        "This is a reproducible pedagogical model experiment, not an estimated Mexican tariff effect. "
        "Official ENIGH 2024 spending, household counts and cash wages are observed; production technology, "
        "trade sourcing, other-factor ownership and the tariff are scenario assumptions.\n\n"
        f"HOME levies a {manifest['tariff_rate']:.1%} tariff on REST food inputs and final purchases. "
        "Both states converge under consistent accounting. Tariff receipts are reconstructed from every "
        "intermediate and final transaction and checked against the solved government budget.\n\n"
        "## Conditional equivalent variation\n\n"
        "Fixed baskets, percent of each decile's baseline monetary consumption:\n\n"
        + df_to_markdown(fixed)
        + "\n\nAggregate monetary EV, percent of observed aggregate consumption (not a social welfare ranking):\n\n"
        + df_to_markdown(aggregate[["ev_pct", "mean_ev_pct", "total_transfer"]])
        + "\n\n## Fiscal and income reconciliation\n\n"
        f"Incremental receipts are {manifest['fiscal']['incremental_tariff_revenue_mxn']:,.2f} MXN per quarter "
        "under the declared consumption-matching scale. Equal and bottom-four rebates exhaust the identical pool. "
        "The no-rebate comparison retains the pool outside household spending at the same GE prices; "
        "it is not a newly solved fiscal closure.\n\n"
        + df_to_markdown(fiscal[fiscal.preferences == "fixed_baskets"][["rule", "household_rebate_mxn", "retained_outside_households_mxn", "fiscal_residual_mxn", "income_residual_mxn"]].set_index("rule"))
        + "\n\nObserved cash wages follow the HOME wage. An explicitly unobserved other-factor proxy follows "
        "the rental return and is allocated proportional to baseline consumption. National exposed factor "
        "income matches the GE factor accounts; per-decile baseline income/spending residuals are exported "
        "and held fixed. Transfers allocate the existing tariff receipt once. With full rebates, mapped "
        "household budget changes sum to the scaled GE budget change.\n\n"
        "## Scope and reproduction\n\n"
        "Preferences and redistribution vary only in household postprocessing. Prices, factor returns and "
        "production remain those of the one solved GE counterfactual. Group demand does not feed back. "
        "All groups inherit the same origin basket within each hypothetical consumption composite.\n\n"
        "The manifest records every assumption, source provenance, unit conversion, equilibrium residual "
        "and artifact checksum. Transaction CSV and equilibrium NPZ preserve physical flows and tariff "
        "schedules for independent checks.\n\n"
        "Observed data: [INEGI ENIGH 2024, basic tables 3.2 and 4.2]("
        + manifest["observed_data"]["source_url"] + ").\n"
    )
    (output / "report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("research_output/distributional_trade_ge"))
    parser.add_argument("--tariff-rate", type=float, default=.10)
    args = parser.parse_args()
    study = run_application(args.output, tariff_rate=args.tariff_rate)
    print(study["results"]["fixed_baskets_equal_per_household"].summary())
    print(f"Saved six conditional scenarios, solved-flow evidence and fiscal reconciliation to {args.output}")


if __name__ == "__main__":
    main()
