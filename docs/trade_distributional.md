> 🇬🇧 English · 🇪🇸 [Español](es/trade_distributional.md)

# Distributional household incidence

`puremacro.trade.distributional` combines observed group spending baskets with
supplied purchaser prices, nominal income changes and fiscal transfers. It
reports exact static equivalent variation (EV) and compensating variation (CV)
using the existing [household preference engine](trade_household.md).

This is **partial-equilibrium household incidence conditional on the supplied
prices and incomes**. Those prices can come from an audited GE solution or an
explicitly assumed scenario. Household demand is not fed back into production,
trade or factor prices. The output is not a new GE equilibrium or a causal
estimate of a tariff's effect. A survey basket alone does not identify price
pass-through, factor-income responses or preference elasticities.

## Supply group means and population counts

`prepare_household_groups` accepts:

| Input | Required meaning |
|---|---|
| `expenditure` | DataFrame indexed by group, with labeled sectors as columns; **mean monetary spending per represented unit**, not group totals |
| `population_weights` | Positive expansion counts by group, in the same represented unit |
| `baseline_budget` | Optional labeled Series, checked against expenditure row sums |
| `income_exposure` | Optional group-by-income-source DataFrame of baseline monetary income per represented unit |
| `monetary_unit`, `period`, `population_unit` | Explicit currency scale, expenditure period, and represented unit |
| `provenance` | Source dictionary; `source` is required, and synthetic/regression-fixture flags are retained |

Labels must be unique nonempty strings. Inputs are aligned by labels; missing,
extra or duplicate labels raise. Purchaser prices must be finite and positive.
Spending and income exposures must be finite and nonnegative; every group must
have positive total spending. No groups or missing observations are silently
dropped or imputed.

For survey observations with expansion weights `w_i`, construct each group mean
as `sum(w_i * spending_i_sector) / sum(w_i)` and its population count as
`sum(w_i)`. Use the same complete household sample and denominator for **every
sector**, retaining households with zero purchases. A published mean among
positive purchasers is not the unconditional group mean. Do not sum survey
weights repeatedly across sector records. Choose household or person weights
consistently with the spending unit.

Group means represent representative groups. Nonlinear welfare evaluated at
group means is not the survey-weighted mean welfare of individual households.
Use one row per household when that distinction matters and the data support it.

Income exposure may exceed observed spending: saving, noncash resources and
other income can differ. The implemented scenario assumes **all incremental
exposed nominal income is spent**, holding the remaining baseline spending
fixed. This is a disclosed behavioral assumption, not an estimated propensity
to consume. A source can distinguish both factor and sector, such as
`manufacturing_labor` or `agriculture_capital`.

## A small synthetic scenario

The following numbers are invented to demonstrate the interface. They are not
a calibrated economy, a survey estimate, or a tariff forecast.

```python
import pandas as pd
from puremacro.trade.distributional import (
    prepare_household_groups, compute_distributional_welfare,
)

spending = pd.DataFrame(
    [[60.0, 40.0], [40.0, 160.0]],
    index=["lower_income", "higher_income"], columns=["food", "other"],
)
groups = prepare_household_groups(
    spending,
    pd.Series([300.0, 100.0], index=spending.index),
    income_exposure=pd.DataFrame(
        {"labor": [80.0, 100.0], "capital": [20.0, 100.0]}, index=spending.index,
    ),
    monetary_unit="illustrative currency units", period="quarter",
    population_unit="households",
    provenance={"source": "two-group synthetic illustration", "is_synthetic": True},
)
result = compute_distributional_welfare(
    groups,
    base_prices=pd.Series({"food": 1.0, "other": 1.0}),
    prices=pd.Series({"food": 1.20, "other": 0.90}),
    factor_income_changes=pd.Series({"labor": 0.10, "capital": -0.05}),
    transfer_total=1400.0,
    transfer_weights=pd.Series({"lower_income": 2.0, "higher_income": 1.0}),
    rule="cobb_douglas",
)
print(result.groups[["baseline_budget", "income_change", "transfer", "ev", "ev_pct"]])
print(result.aggregate[["total_transfer", "total_ev", "ev_pct", "mean_ev_pct"]])
print(result.summary())
result.plot()
```

`factor_income_changes` uses **fractions**: `0.10` means +10%, and `-1` means a
complete loss of the exposed source. Its labels must exactly match the income
exposure columns. Omitting changes means no income shock.

The `prices / base_prices` ratio rebases sector prices to each supplied
baseline spending basket. Both states and nominal income shocks must use a
common numeraire. Uniformly scaling prices and all nominal resources leaves
real welfare unchanged.

## Transfers and aggregation

`transfer_total` is the aggregate **change** in revenue allocated to these
represented households, in the spending currency and period. A negative value
is an aggregate withdrawal. It is supplied explicitly; the bridge does not
infer it from GE government income or household factor earnings.

`transfer_weights` gives nonnegative **per-unit allocation scores** `a_g`:

```text
transfer_per_unit[g] = transfer_total * a_g / sum_h(population_weight[h] * a_h)
sum_g(population_weight[g] * transfer_per_unit[g]) = transfer_total
```

The default is equal transfers per represented unit, not equal totals per
group. In the example, each lower-income household receives four currency
units and each higher-income household two; `300*4 + 100*2 = 1400`.
Avoid counting fiscal transfers again inside the supplied income shocks.
Normalized population weights represent a population of size one, so the
transfer total must be expressed on that scale. Convert units explicitly when
a GE table is in millions but the survey uses currency per household.

`result.groups` reports monetary values per represented unit and percentages
relative to that group's baseline spending. `result.aggregate.total_ev` and
`total_cv` sum those values using expansion weights. Aggregate `ev_pct` divides
total EV by total baseline spending; `mean_ev_pct` instead averages group EV
percentages using population weights. They answer different questions. Neither
aggregation is a social welfare function or an interpersonal utility comparison.

## Preferences and exact attribution

Choose `rule="fixed_baskets"`, `"cobb_douglas"`, `"ces"` or `"stone_geary"`.
CES takes `ces_elasticity`. LES takes a labeled group-by-sector
`expenditure_elasticities` DataFrame and a scalar or group-Series
`supernumerary_share`. Omitted LES targets mean unit elasticities; its omitted
surplus share is the existing engine's explicit 0.5 assumption. The calibration
audit, adding-up repair, assumed-share flag and elasticity adjustments are
recorded in `metadata["preference_calibration"]`. Observed expenditure alone
does not estimate these preference parameters.

The household engine defines
`EV = e(p0,u1) - e(p0,u0)` and `CV = e(p1,u1) - e(p1,u0)`.
Both are positive for gains; CV is the amount removable at new prices.
Percentages use the original observed spending budget as denominator.

The columns `price_ev`, `income_ev`, and `transfer_ev` telescope exactly to
`ev`; the corresponding `_cv` columns telescope to `cv`. These differences
compare welfare anchored to the **same original baseline**, along the ordered
path **prices → factor income → transfers**. They are order-dependent,
descriptive attribution, not separate causal effects or a Shapley average.
Every stage is checked against the preference domain. A price-only LES stage
that exhausts subsistence resources raises even if a subsequent transfer
would make the final state feasible.

## Bridge to a solved trade model

`distributional_welfare_from_results(groups, calib, baseline, counterfactual,
country=..., factor_income_changes=..., **scenario_options)` uses
`household_prices_from_result` to recover one country's sector purchaser
prices and reuses its equilibrium audit. Both states must have converged, use
the same accounting mode, and carry exactly the calibration's ordered country
and sector labels. Household sector labels must match the calibration.

The explicit income mapping is a Series from household income-source labels
to fractional nominal changes. If a researcher maps national wage changes into
survey labor income, that reconciliation remains an explicit modeling choice.
The GE consumption-category budget is **not** substituted for observed group
budgets. `baseline_tau_fd` and `tau_fd` provide legacy final-demand schedules;
consistent results use their recorded, re-audited schedules. The supported
sector baskets and flexible-model restrictions are those of the
[household price bridge](trade_household.md).

Within each sector, every survey group inherits that country's GE origin
basket. Different sector spending shares do not identify group-specific
domestic/import sourcing within a sector. If the selected GE category has
zero origin-basket weight for a sector but a survey group spends on it, the
bridge raises an error. The shared helper's unit-price placeholder for an
unconsumed sector cannot value positive survey spending. To use a separately
estimated sector price or an explicit alternative origin mapping, construct
those purchaser prices and call `compute_distributional_welfare` directly.
The inherited-origin assumption and unused structural-zero sectors are
recorded in result metadata.

The result retains household provenance plus calibration and both equilibrium
metadata. A top-level `is_regression_fixture` tag is propagated when any of
these inputs carries it. Results derived from a regression fixture remain
regression-fixture exercises; attaching survey expenditure does not turn the
GE price scenario into empirical evidence.

Results support `summary()`, `to_frame()` / `to_dataframe()`, `to_markdown()`,
`to_latex()`, `to_typst()`, and `plot()`.
