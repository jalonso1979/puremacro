> 🇬🇧 English · 🇪🇸 [Español](es/distributional_trade_ge.md)

# A reconciled GE-to-household trade application

`puremacro.examples.distributional_trade_ge` connects two audited trade equilibria
to observed ENIGH 2024 decile baskets, nominal income exposures and an explicitly
reconciled tariff-revenue pool. It compares equal rebates and rebates to the
bottom four deciles, with a no-household-rebate reference, under fixed baskets
and Cobb–Douglas preferences.

**The trade economy is synthetic.** Official household spending, household counts
and cash wages are observed; technology, origin shares and the policy shock are
scenario assumptions. This is a reproducible pedagogical application, not an
estimate of an actual Mexican tariff's effects. Attaching ENIGH observations
does not turn a synthetic equilibrium into empirical trade evidence.

## Run the complete application

```bash
python -m puremacro.examples.distributional_trade_ge \
  --output research_output/distributional_trade_ge --tariff-rate 0.10
```

The command writes six decile-incidence scenarios, aggregate summaries, purchaser
prices, an income-mapping table, every bilateral tariff transaction, the fiscal
ledger, equilibrium arrays, a chart, a report and a checksummed manifest. It
requires no network access; the [ENIGH loader](trade_distributional.md) checks
the bundled observations against their recorded checksum. The manifest retains
the official source URL and hash, model assumptions, numerical source hashes,
runtime versions and equilibrium residuals.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from puremacro.examples.distributional_trade_ge import run_application

with TemporaryDirectory() as directory:
    study = run_application(Path(directory), tariff_rate=0.10)
    print(study["results"]["fixed_baskets_equal_per_household"].summary())
    print(study["fiscal_allocation"][[
        "preferences", "rule", "fiscal_residual_mxn", "income_residual_mxn",
    ]])
```

## What is solved and what is assumed

The model has two hypothetical symmetric regions, `HOME` and `REST`, and eight
consumption composites named after the survey categories. These are not an
estimated industry concordance. Baseline final spending is 100 model units per
region, with sector shares taken from population-weighted ENIGH spending.
Intermediate costs are 35% of output, and intermediate import sourcing is 20%.
Final import shares are specified separately in the manifest: food is 25%.
Production combines fixed intermediate coefficients with Cobb–Douglas labor and
capital. Baseline taxes, tariffs and foreign saving are zero.

The tariff applies to `REST` food delivered to `HOME`, for both intermediate
uses and final purchases. Both states use `accounting="consistent"`, with the
same numeraire: the `HOME` food producer price. The audited household bridge
recovers purchaser prices and checks convergence, labels, stored tariff
schedules and equilibrium equations. All groups inherit the same origin basket
within a sector. In this particular symmetric calibration, a 10% tariff raises
the food purchaser price by 2.5%; this is not an estimated pass-through rate.

## Income and monetary units

One model value unit is converted to MXN per quarter by dividing total observed
baseline consumption by `HOME` baseline model consumption. This is a disclosed
normalization, not an exchange rate or a national-accounts calibration.

The synthetic labor share is observed aggregate cash wages divided by observed
consumption, approximately 0.9119. The remaining factor income is represented by
an **unobserved other-factor proxy**, allocated across deciles proportional to
consumption. This proxy is not observed survey capital income.

| Household exposure | Counterfactual nominal change |
|---|---|
| Observed cash wages | `HOME` GE wage change |
| Assumed other-factor proxy | `HOME` GE rental-return change |
| Baseline exposure minus consumption | Held fixed within each decile |

All incremental factor income is spent. Decile exposure totals can exceed or
fall below baseline consumption; their residuals are exported and sum to zero
nationally. Population-weighted exposed income matches the scaled GE factor
accounts in both states. No tariff receipt is included in either factor exposure.

## Allocate the existing tariff receipt once

For every intermediate and final transaction, the application reconstructs
`duty = tariff_rate × producer_price × quantity`. Summed receipts must match the
reported tariff revenue and, because other taxes are zero, the government budget.
The household pool is the change in `HOME` receipts, converted using the same
monetary scale.

Equal rebates pay every household the same amount. Bottom-four targeting pays
only those deciles, with equal amounts per eligible household. Both exhaust
exactly the same pool. The ledger verifies:

```text
household rebates + receipts retained outside households = incremental revenue
household budget change + retained receipts = scaled GE budget change
```

The GE closure already includes fiscal receipts in national income. The example
maps factor earnings separately, then distributes those existing receipts once
through the household transfer argument. It never adds tariff revenue a second
time to GE total income.

The no-household-rebate reference retains the pool outside household spending
while holding GE prices fixed. It is a conditional comparison, not a separately
solved government-saving closure. Alternative targeting and household preferences
also remain postprocessing: household demand does not feed back into production,
trade, wages or rental returns.

## Read the default results with their scope

For the default synthetic 10% tariff, fixed-basket EV is +2.380% for decile I and
−0.677% for decile X under equal rebates. Bottom-four targeting changes those
figures to +8.084% and −1.238%. These are percentages of each group's baseline
monetary consumption, conditional on the scenario assumptions.

Aggregate monetary EV is close to zero and sensitive to preferences: −0.001055%
under fixed baskets with equal rebates, versus +0.006066% under Cobb–Douglas.
Population-weighted average percentage EV answers a different question. Neither
measure supplies a social welfare ranking. Group means omit within-decile
heterogeneity and provide no sampling confidence intervals.

See [distributional incidence](trade_distributional.md) for EV/CV definitions,
ordered price → income → transfer attribution and household input contracts.
