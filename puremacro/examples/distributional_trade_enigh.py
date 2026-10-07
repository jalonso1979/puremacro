"""Observed ENIGH 2024 baskets under explicit illustrative trade scenarios.

Run ``python -m puremacro.examples.distributional_trade_enigh --output DIR``.
Data are observed official income-decile means; import exposure, tariff
pass-through, cash-wage changes and the transfer pool are scenario assumptions.
No historical tariff effect or calibrated general-equilibrium result is claimed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from puremacro import __version__
from puremacro.datasets.enigh import load_enigh2024_deciles
from puremacro.trade.distributional import prepare_household_groups, compute_distributional_welfare


def run_application(output: Path, *, tariff_rate: float = .10,
                    food_import_share: float = .25, pass_through: float = 1.0,
                    cash_wage_change: float = 0.0, transfer_budget_share: float = .005):
    """Export incidence for observed deciles and explicitly assumed scenarios."""
    import numpy as np
    values = [tariff_rate, food_import_share, pass_through, cash_wage_change, transfer_budget_share]
    if not np.isfinite(values).all() or tariff_rate < 0 or not 0 <= food_import_share <= 1 or pass_through < 0 or cash_wage_change < -1 or transfer_budget_share < 0:
        raise ValueError("invalid tariff, import share, pass-through, wage change or transfer-budget assumption")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_enigh2024_deciles()
    sectors = list(data.attrs["categories"])
    groups = prepare_household_groups(
        data[sectors], data.households, monetary_unit="MXN", period="quarter",
        provenance=dict(data.attrs), income_exposure=data[["cash_wages"]],
    )
    base_prices = pd.Series(1., index=sectors)
    prices = base_prices.copy()
    # A deliberately explicit first-order pass-through *scenario mapping*,
    # not an equilibrium price equation or an estimated import-share response.
    prices["food"] += tariff_rate * food_import_share * pass_through
    pool = transfer_budget_share * float(data.households @ data[sectors].sum(axis=1))
    wage_change = pd.Series({"cash_wages": cash_wage_change})
    targeting = pd.Series([1., 1., 1., 1., 0., 0., 0., 0., 0., 0.], index=data.index)
    scenarios = {
        "no_transfer": (0., None),
        "equal_per_household": (pool, None),
        "bottom_four_deciles": (pool, targeting),
    }
    results, rows = {}, []
    for rule in ("fixed_baskets", "cobb_douglas"):
        for name, (transfer, scores) in scenarios.items():
            result = compute_distributional_welfare(
                groups, base_prices, prices, factor_income_changes=wage_change,
                transfer_total=transfer, transfer_weights=scores, rule=rule,
            )
            key = f"{rule}_{name}"
            results[key] = result
            frame = result.to_frame().reset_index()
            frame.insert(0, "scenario", name)
            frame.insert(0, "preferences", rule)
            rows.append(frame)
            result.to_frame().to_csv(output / f"{key}.csv")
    all_results = pd.concat(rows, ignore_index=True)
    all_results.to_csv(output / "all_scenarios.csv", index=False)
    data.to_csv(output / "observed_enigh_deciles.csv")
    aggregates = pd.DataFrame({name: r.aggregate for name, r in results.items()}).T
    aggregates.to_csv(output / "aggregate_results.csv")
    manifest = {
        "package_version": __version__, "application": "ENIGH 2024 conditional distributional trade scenarios",
        "observed_data": dict(data.attrs),
        "assumptions": {"tariff_rate": tariff_rate, "food_import_share": food_import_share,
                        "pass_through": pass_through, "cash_wage_change": cash_wage_change,
                        "transfer_budget_share_of_baseline_consumption": transfer_budget_share,
                        "transfer_pool_mxn_per_quarter": pool,
                        "prices": prices.to_dict(), "baseline_prices": base_prices.to_dict()},
        "price_mapping": "food purchaser-price ratio = 1 + assumed tariff * assumed import exposure * assumed pass-through",
        "scope": "observed group baskets with illustrative partial-equilibrium policy assumptions",
        "limitations": ["transfer pool is externally financed and is NOT inferred tariff revenue",
                        "no estimated sector concordance, import shares, elasticities or GE price responses",
                        "fixed baskets and Cobb-Douglas are alternative preference assumptions",
                        "all incremental cash wages are spent; no saving response",
                        "no sampling confidence intervals are available from group means",
                        "aggregate monetary EV sums do not define a social welfare function"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    from puremacro.reports import df_to_markdown
    summary = all_results[all_results.preferences == "fixed_baskets"].pivot(index="group", columns="scenario", values="ev_pct").reindex(data.index)
    (output / "report.md").write_text(
        "# Distributional trade scenarios: Mexico, ENIGH 2024\n\n"
        "Observed official spending baskets and expansion counts for ten national income deciles. "
        f"The illustrative tariff/import-exposure/pass-through mapping raises the food basket price by {(prices['food']-1)*100:.3g}%. "
        "These assumptions are editable; this is not an estimated effect of an actual tariff.\n\n"
        "Equivalent variation, percent of each decile's baseline monetary consumption, fixed baskets:\n\n"
        + df_to_markdown(summary)
        + "\n\nThe two transfer scenarios allocate the same externally supplied pool. "
        "The pool is not an estimate of tariff revenue. The eight-category basket excludes outward transfers "
        "and nonmonetary consumption. Cobb–Douglas sensitivity results, income/price/transfer attributions, "
        "population-weighted totals and full source/assumption metadata accompany this report.\n",
        encoding="utf-8",
    )
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    (data.food / data[sectors].sum(axis=1) * 100).plot.bar(ax=axes[0], color="#256D85")
    axes[0].set(title="Observed food budget share", ylabel="Percent of monetary consumption", xlabel="Income decile")
    summary.rename(columns={"bottom_four_deciles": "Transfer to bottom four deciles",
                            "equal_per_household": "Equal transfer per household",
                            "no_transfer": "No transfer"}).plot(
        ax=axes[1], marker="o", color=["#B36B39", "#3C847A", "#485B86"])
    axes[1].axhline(0, color="gray", linewidth=.7)
    axes[1].set(title="Illustrative price and transfer scenarios", ylabel="Equivalent variation (% of consumption)", xlabel="Income decile")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "distributional_incidence.png", dpi=170)
    plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("research_output/distributional_trade"))
    parser.add_argument("--tariff-rate", type=float, default=.10)
    parser.add_argument("--food-import-share", type=float, default=.25)
    parser.add_argument("--pass-through", type=float, default=1.)
    parser.add_argument("--cash-wage-change", type=float, default=0.)
    parser.add_argument("--transfer-budget-share", type=float, default=.005)
    args = parser.parse_args()
    results = run_application(args.output, tariff_rate=args.tariff_rate,
                              food_import_share=args.food_import_share, pass_through=args.pass_through,
                              cash_wage_change=args.cash_wage_change, transfer_budget_share=args.transfer_budget_share)
    print(results["fixed_baskets_no_transfer"].summary())
    print(f"Saved six scenarios and provenance to {args.output}")


if __name__ == "__main__":
    main()
