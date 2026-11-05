"""Conditional supplier adjustment linked to observed ENIGH household baskets.

An illustrative two-origin food composite faces a temporary import tariff.
Origin shares, sourcing elasticity and switching costs are assumptions. Costs
are passed through fully to consumers; taxes finance household compensation.
No country equilibrium, employment reallocation or causal tariff estimate is
claimed. Both sourcing resources and household lifetime welfare are retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.examples.household_trade_policy import load_inputs
from puremacro.study.robust import compare_policy_payoffs
from puremacro.trade.adjustment import solve_sourcing_adjustment
from puremacro.trade.compensation import balanced_compensation
from puremacro.trade.incidence_transition import household_incidence_path


def run_study(*, horizon=12):
    if not isinstance(horizon, int) or horizon < 8:
        raise ValueError("horizon must be an integer >= 8")
    groups, data, source = load_inputs()
    weights = data.factor.to_numpy()
    poor = (data.quintile <= 2).to_numpy()
    gross = .01 * float(weights @ data.gasto_mon.to_numpy())
    scores = {"equal_per_household": pd.Series(1., index=data.index),
              "bottom_40_income": pd.Series(poor.astype(float), index=data.index)}
    plans = {k: balanced_compensation(groups, gross, benefit_scores=v, tax_base=data.gasto_mon)
             for k,v in scores.items()}
    sources, rows, period_rows, refinements = {}, [], [], []
    for tariff in (.1, .25):
        for cost in (0., 2.):
            key = f"tariff={tariff:g};adjustment={cost:g}"
            def solve(T):
                p = np.ones((T,2))
                p[2:6,1] += tariff
                return solve_sourcing_adjustment(pd.DataFrame(p, columns=["domestic","imported"]),
                    pd.Series({"domestic":.7,"imported":.3}), elasticity=3., adjustment_cost=cost)
            sourcing = solve(horizon)
            refined = solve(2*horizon)
            gap = float(np.max(np.abs(sourcing.accounts.total_unit_cost.to_numpy() -
                                     refined.accounts.total_unit_cost.to_numpy()[:horizon])))
            refinements.append({"state":key,"horizon":horizon,"extended_horizon":2*horizon,
                                "maximum_first_window_cost_change":gap,"tolerance":1e-4,"passed":gap<=1e-4})
            if gap>1e-4:
                raise RuntimeError(f"{key}: sourcing horizon comparison failed with {gap:g}")
            sources[key] = sourcing
            consumer_prices = pd.DataFrame({"food_beverages_tobacco":sourcing.accounts.total_unit_cost,
                                             "other_monetary_spending":1.})
            for name,plan in plans.items():
                tr = pd.DataFrame(np.broadcast_to(plan.net_transfers.to_numpy(), (horizon,len(data))),
                                  index=consumer_prices.index, columns=data.index)
                for gamma in (1.,2.):
                    result = household_incidence_path(groups,consumer_prices,transfers=tr,
                                                      discount=.99,risk_aversion=gamma,ces_elasticity=1.)
                    cev = result.households.cev.to_numpy()
                    state = key+f";curvature={gamma:g}"
                    rows.append({"policy":name,"state":state,"mean_cev":float(np.average(cev,weights=weights)),
                        "bottom40_mean_cev":float(np.average(cev[poor],weights=weights[poor])),
                        "winner_share":float(np.average(cev>0,weights=weights))})
                    period = result.periods.copy()
                    period["policy"],period["state"] = name,state
                    period["period"] = np.arange(horizon)
                    period_rows.append(period)
    aggregates = pd.DataFrame(rows)
    comparisons = {}
    for metric in ("mean_cev","bottom40_mean_cev"):
        comparisons[metric] = compare_policy_payoffs(aggregates.pivot(index="policy",columns="state",values=metric),
            objective=metric,units="baseline quarterly MXN per household", comparability={
                "population":"ENIGH positive-spending households"+(" in bottom two income quintiles" if metric.startswith("bottom") else ""),
                "horizon":f"same {horizon} quarters", "numeraire":source["monetary_unit"],
                "welfare_definition":"constant consumption equivalent of finite-path CES-CRRA utility"})
    return {"aggregate":aggregates,"periods":pd.concat(period_rows,ignore_index=True),
        "refinement":pd.DataFrame(refinements),"comparisons":comparisons,"sourcing":sources,
        "metadata":{"source":source,"horizon":horizon,"tariff_dates":[2,3,4,5],
            "origin_shares":{"domestic":.7,"imported":.3},"sourcing_elasticity":3.,
            "household_elasticity":1.,"discount":.99,"gross_compensation_share":.01,
            "identification":"origin shares, substitution elasticities, tariffs, and switching costs are assumptions",
            "transmission":"full pass-through of unit procurement plus switching resources to the survey alimentos composite",
            "financing":"baseline spending-proportional lump-sum taxes; tariff revenue is not allocated or counted as household income",
            "scope":"conditional sourcing and household incidence; no national GE, labor adjustment, or causal estimate",
            "welfare":"same finite horizon for all policy comparisons; no household saving or terminal tail"}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output must be a new directory")
    results=run_study()
    args.output.mkdir(parents=True)
    for key in ("aggregate","periods","refinement"):
        results[key].to_csv(args.output/f"{key}.csv",index=False)
    records={}
    for i,(key,result) in enumerate(results["sourcing"].items()):
        result.accounts.to_csv(args.output/f"sourcing_{i}_accounts.csv")
        result.quantities.to_csv(args.output/f"sourcing_{i}_quantities.csv")
        records[key]={"file_prefix":f"sourcing_{i}","diagnostics":result.diagnostics,"model":result.metadata}
    for key,result in results["comparisons"].items():
        result.summary.to_csv(args.output/f"{key}_robustness.csv")
        result.payoffs.to_csv(args.output/f"{key}_payoffs.csv")
    results["metadata"]["sourcing_records"]=records
    (args.output/"manifest.json").write_text(json.dumps(results["metadata"],indent=2)+"\n")
    print(results["refinement"].to_string(index=False))
    print(results["comparisons"]["bottom40_mean_cev"].summary.to_string())


if __name__=="__main__":
    main()
