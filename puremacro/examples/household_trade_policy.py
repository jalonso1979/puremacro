"""ENIGH 2022: food-price exposure and explicitly financed compensation.

Observed spending and weights, assumed counterfactual prices and preferences.
This is a conditional household policy experiment, not an estimated tariff
effect, an HANK calibration or a survey-based causal finding. Run offline:

    python -m puremacro.examples.household_trade_policy --output study_output
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro.study.robust import compare_policy_payoffs
from puremacro.trade.compensation import balanced_compensation
from puremacro.trade.distributional import prepare_household_groups, compute_distributional_welfare


def load_inputs():
    """Authenticate all retained source rows, then declare the analysis domain."""
    folder = Path(__file__).resolve().parents[1] / "trade" / "_datafiles"
    path = folder / "enigh2022_household_policy.npz"
    manifest = json.loads(path.with_suffix(".json").read_text())
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
        raise ValueError("ENIGH input checksum mismatch")
    with np.load(path, allow_pickle=False) as f:
        data = pd.DataFrame({k: f[k] for k in manifest["variables"]})
    if len(data) != manifest["source_rows"]:
        raise ValueError("ENIGH input row count mismatch")
    # Retain exclusions in the source payload/manifest, never impute budgets.
    data = data.loc[data.gasto_mon > 0].copy()
    data.index = pd.Index([f"source_row_{i}" for i in data.index], name="household")
    w = data.factor.to_numpy()
    rank_income = (data.ing_cor / data.tot_integ).to_numpy()
    order = np.argsort(rank_income, kind="stable")
    cdf = np.cumsum(w[order]) / w.sum()
    cutoffs = rank_income[order][np.searchsorted(cdf, [.2, .4, .6, .8])]
    # A tie stays together; realized population shares are recorded below.
    data["quintile"] = np.searchsorted(cutoffs, rank_income, side="left") + 1
    expenditure = pd.DataFrame({"food_beverages_tobacco": data.alimentos,
                                "other_monetary_spending": data.gasto_mon - data.alimentos})
    groups = prepare_household_groups(
        expenditure, data.factor, monetary_unit=manifest["monetary_unit"],
        period=manifest["period"], population_unit="households", provenance=manifest)
    manifest = {**manifest, "quintile_income_definition": "quarterly current income / household members",
                "quintile_weight": "household expansion factor; not person weights",
                "quintile_cutoffs": cutoffs.tolist(), "ties": "retain in lower quintile",
                "analysis_population": float(w.sum())}
    return groups, data, manifest


def run_study(*, price_changes=(.05, .10, .20), elasticities=(0., 1., 2.), gross_share=.01):
    """Evaluate household-level EV, then aggregate and compare policy rules.

    Household preferences are homothetic CES, calibrated separately to each
    observed basket. All net transfers are spent. Tax payments are lump sum,
    proportional to *baseline* monetary expenditure. These assumptions remain
    fixed across policies. Survey sampling uncertainty is not an uncertainty
    state: results here condition on the observed sample and its weights.
    """
    if not np.isfinite(gross_share) or not 0 <= gross_share < 1:
        raise ValueError("gross_share must be in [0, 1)")
    changes = np.asarray(price_changes, dtype=float)
    sigmas = np.asarray(elasticities, dtype=float)
    if changes.ndim != 1 or not changes.size or not np.isfinite(changes).all() or np.any(changes <= -1):
        raise ValueError("price_changes must be finite fractions above -1")
    if sigmas.ndim != 1 or not sigmas.size or not np.isfinite(sigmas).all() or np.any(sigmas < 0):
        raise ValueError("elasticities must be finite and nonnegative")
    if len(np.unique(changes)) != len(changes) or len(np.unique(sigmas)) != len(sigmas):
        raise ValueError("scenario coordinates must be unique")
    groups, data, manifest = load_inputs()
    w = data.factor.to_numpy()
    total = float(w @ data.gasto_mon.to_numpy())
    poor = data.quintile <= 2
    scores = {
        "no_compensation": pd.Series(1., index=data.index),
        "equal_per_household": pd.Series(1., index=data.index),
        "bottom_40_income": poor.astype(float),
        "food_expenditure": data.alimentos.copy(),
    }
    plans = {name: balanced_compensation(groups, 0. if name == "no_compensation" else gross_share * total,
                                         benefit_scores=score, tax_base=data.gasto_mon)
             for name, score in scores.items()}
    rows, aggregate_rows = [], []
    max_oracle_gap = 0.
    baseline_prices = pd.Series(1., index=groups.expenditure.columns)
    budget = data.gasto_mon.to_numpy()
    food_share = data.alimentos.to_numpy() / budget
    for change in changes:
        price = 1 + float(change)
        for sigma in sigmas:
            # Preserve distinct floating-point scenarios instead of rounding
            # their labels to the six digits used by the default :g format.
            change_label = np.format_float_positional(float(change), trim="-")
            sigma_label = np.format_float_positional(float(sigma), trim="-")
            state = f"food_change={change_label};household_sigma={sigma_label}"
            for name, plan in plans.items():
                result = compute_distributional_welfare(
                    groups, baseline_prices,
                    pd.Series([price, 1.], index=baseline_prices.index),
                    household_transfers=plan.net_transfers, rule="ces", ces_elasticity=float(sigma))
                ev = result.groups.ev.to_numpy()
                # Independent closed forms for this two-good, p_other=1 case.
                cost = (price ** food_share if sigma == 1 else
                        np.exp(np.log1p(food_share * np.expm1((1-sigma) * np.log(price))) / (1-sigma)))
                oracle = (budget + plan.net_transfers.to_numpy()) / cost - budget
                gap = float(np.max(np.abs(ev - oracle) / np.maximum(budget, 1.)))
                max_oracle_gap = max(max_oracle_gap, gap)
                if gap > 1e-10:
                    raise ArithmeticError("household EV disagrees with independent CES formula")
                aggregate_rows.append({"policy": name, "state": state,
                    "mean_ev": float(np.average(ev, weights=w)),
                    "bottom40_mean_ev": float(np.average(ev[poor], weights=w[poor])),
                    "mean_ev_pct": float(np.average(result.groups.ev_pct, weights=w)),
                    "winner_share": float(np.average(ev > 0, weights=w)),
                    "fiscal_residual": float(plan.ledger.net_transfers)})
                for quintile in range(1, 6):
                    mask = (data.quintile == quintile).to_numpy()
                    rows.append({"policy": name, "state": state, "quintile": quintile,
                        "population_share": float(w[mask].sum() / w.sum()),
                        "mean_ev": float(np.average(ev[mask], weights=w[mask])),
                        "mean_ev_pct": float(np.average(result.groups.ev_pct.to_numpy()[mask], weights=w[mask])),
                        "mean_benefit": float(np.average(plan.households.benefit.to_numpy()[mask], weights=w[mask])),
                        "mean_tax": float(np.average(plan.households.tax.to_numpy()[mask], weights=w[mask]))})
    aggregate = pd.DataFrame(aggregate_rows)
    comparisons = {}
    contract = {"population": "ENIGH 2022 positive-monetary-spending households",
                "horizon": "one quarter", "numeraire": manifest["monetary_unit"],
                "welfare_definition": "CES equivalent variation at each household's baseline prices"}
    for metric in ("mean_ev", "bottom40_mean_ev"):
        payoff = aggregate.pivot(index="policy", columns="state", values=metric)
        comparisons[metric] = compare_policy_payoffs(
            payoff, objective=metric, units="MXN per household per quarter",
            comparability={**contract, "population": contract["population"] +
                           ("; bottom two fixed income quintiles" if metric.startswith("bottom") else "")})
    baseline = pd.DataFrame([
        {"quintile": quintile, "sample_households": len(g), "population_households": g.factor.sum(),
         "mean_income": np.average(g.ing_cor, weights=g.factor),
         "mean_spending": np.average(g.gasto_mon, weights=g.factor),
         "mean_food_share": np.average(g.alimentos / g.gasto_mon, weights=g.factor)}
        for quintile, g in data.groupby("quintile", sort=True)]).set_index("quintile")
    metadata = {"source": manifest, "gross_budget_share": gross_share,
        "price_changes": changes.tolist(), "household_elasticities": sigmas.tolist(),
        "fiscal_rule": "gross benefits financed by lump-sum taxes proportional to baseline monetary spending",
        "fiscal_comparability": "all compensation rules share one gross budget; no_compensation is a zero-budget control",
        "income_rule": "baseline nominal resources fixed; all net transfers spent; no labor or saving response",
        "sector_definition": "alimentos includes food, beverages and tobacco; other=gasto_mon-alimentos",
        "scope": "observed-data conditional static incidence; no causal tariff identification or GE feedback",
        "uncertainty": "enumerated price and preference assumptions; no probabilities or survey confidence intervals",
        "aggregation": "evaluate nonlinear EV per household before survey-weighted aggregation",
        "maximum_scaled_independent_ev_gap": max_oracle_gap,
        "data_attribution": manifest["attribution"],
        "endorsement": "PureMacro analysis and transformations; not an official INEGI analysis or endorsement"}
    return {"baseline": baseline, "incidence": pd.DataFrame(rows), "aggregate": aggregate,
            "comparisons": comparisons, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output must be a new directory")
    result = run_study()
    args.output.mkdir(parents=True)
    for key in ("baseline", "incidence", "aggregate"):
        result[key].to_csv(args.output / f"{key}.csv", index=key == "baseline")
    for key, comparison in result["comparisons"].items():
        comparison.payoffs.to_csv(args.output / f"{key}_payoffs.csv")
        comparison.summary.to_csv(args.output / f"{key}_robustness.csv")
        comparison.regrets.to_csv(args.output / f"{key}_regrets.csv")
    folder = Path(__file__).resolve().parents[1] / "trade" / "_datafiles"
    for suffix in ("npz", "json"):
        source = folder / f"enigh2022_household_policy.{suffix}"
        (args.output / source.name).write_bytes(source.read_bytes())
    (args.output / "manifest.json").write_text(json.dumps(result["metadata"], indent=2) + "\n")
    print(result["baseline"].to_string())
    print(result["comparisons"]["bottom40_mean_ev"].summary.to_string())


if __name__ == "__main__":
    main()
