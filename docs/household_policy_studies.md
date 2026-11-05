> 🇬🇧 English · 🇪🇸 [Español](es/household_policy_studies.md)

# Observed household policy studies

These development examples connect observed household spending, explicitly
financed compensation and robust policy comparisons. They require the 6.1
development source. Their price changes and preference parameters are declared
scenarios, not estimates of a tariff's causal effect.

## Reproduce the ENIGH study offline

```sh
python -m puremacro.examples.household_trade_policy --output enigh-study
python -m puremacro.examples.trade_adjustment_households --output adjustment-study
```

Each output directory must be new. CSV tables and JSON manifests retain the
assumptions, inputs, financing rules and numerical comparisons. The first study
also copies its authenticated source payload into the output directory.

Source: [INEGI ENIGH 2022 nueva serie](https://www.inegi.org.mx/programas/enigh/nc/2022/),
`concentradohogar`. The retained columns contain all 90,102 original rows and
the expansion factor, stratum and PSU. Of these, 69 rows have zero monetary
spending and are outside the positive-budget preference domain. They represent
21,945 households and remain in the source and exclusion record. The analysis
uses 90,033 sampled households representing 37,538,178 households.

Money is quarterly MXN at August 2022 prices. `alimentos` covers food, beverages
and tobacco; the second category is `gasto_mon - alimentos`. Income quintiles
use household expansion weights and current income per member; ties stay in
the lower quintile. These are household-weighted groups, not population quintiles
weighted by persons. Source archive, CSV and payload hashes accompany the data.
The original all-household weighted mean current income is 63,695.46 pesos,
consistent with INEGI's rounded published figure.

The [INEGI free-use terms](https://www.inegi.org.mx/inegi/terminos.html) apply to
these data. Attribute INEGI and retain the source metadata. The transformations
and analysis are PureMacro's; INEGI does not endorse them.

## Household policies and financing

The static experiment compares 5%, 10% and 20% increases in the first
category's purchaser price. Each household has its own calibrated CES basket;
substitution elasticities 0, 1 and 2 are assumptions. Nonlinear welfare is
evaluated for each household before expansion-weighted aggregation.

Compensation rules allocate the same gross budget, 1% of aggregate baseline
monetary spending: equal benefits per household, benefits for the bottom two
income quintiles, or benefits proportional to baseline food expenditure.
Lump-sum taxes proportional to baseline monetary spending finance the benefits.
The zero-compensation case is a separate zero-budget control. There is no
endogenous labor, saving, tax-base or eligibility response; net transfers are
fully spent.

```python
from puremacro.trade import balanced_compensation
from puremacro.trade.distributional import compute_distributional_welfare

# groups, scores, tax_base and prices are explicitly labeled inputs.
plan = balanced_compensation(groups, gross_total,
                            benefit_scores=scores, tax_base=tax_base)
result = compute_distributional_welfare(
    groups, baseline_prices, counterfactual_prices,
    household_transfers=plan.net_transfers, transfer_total=0., rule="ces",
    ces_elasticity=1.,
)
```

The weighted sum of signed transfers must equal the declared aggregate funding.
Positive EV means a welfare gain measured in baseline purchasing power. The
studies separately compare mean EV for the whole analysis population and for
the bottom two income quintiles. Those objectives can rank policies differently;
neither is silently substituted for the other.

Survey strata and PSUs are preserved but these experiments report no survey
confidence intervals. The uncertainty set contains economic assumptions,
not sampling draws. A causal identification strategy and survey-aware inference
remain additional research work.

## Recorded results

The installed-wheel run retained in `tests/fixtures/household_trade_policy/`
illustrates the distinction between objectives. With a 10% first-category
price increase and CES elasticity one, mean EV in quarterly baseline MXN is:

| Policy | All households | Bottom two income quintiles |
|---|---:|---:|
| No compensation | -1,406.05 | -1,140.51 |
| Equal benefits | -1,408.01 | -1,018.04 |
| Targeted benefits | -1,409.81 | -444.96 |
| Food-spending benefits | -1,408.62 | -1,090.04 |

Targeting reduces the lower-income group's loss in every enumerated static
state. It does not eliminate that loss in this example. Redistributing nominal
resources across households with different price indexes also changes the
population mean of money-metric EV. The policy objectives therefore have
separate payoff and regret tables. These rankings condition on the stated
model and sample; they are not an empirical optimal-policy result.

The recorded independent two-good CES formula agrees with household EV to
`1.1e-15` scaled by baseline budgets. To regenerate both studies, source hashes,
CSV tables and checks into a new directory from a checkout:

```sh
PYTHONPATH=. python tools/reference_validation/check_household_trade.py --output study-evidence
```

## Robust comparisons

`puremacro.study.compare_policy_payoffs` takes a policy-by-state DataFrame.
It reports best/worst outcomes, dominance and maximum regret, where regret is
the gap from the best available policy in each state. All payoffs must share
an explicit population, horizon, numeraire and welfare definition. Missing,
nonfinite or failed cells block the comparison. Expected payoffs appear only
when the caller supplies labeled probabilities summing to one. Grid density
never implies probabilities. Tied policies remain tied.

## Trade adjustment and lifetime incidence

The second study uses [dynamic sourcing](trade_adjustment.md). A temporary
10% or 25% tariff raises the assumed imported input's delivered price at dates
2–5. A two-origin composite has assumed baseline shares 0.7/0.3 and substitution
elasticity 3. Switching costs are either zero or two. Full unit procurement and
adjustment costs pass through to the ENIGH first-category price by assumption.
Tariff revenue is not treated as extra household income or used to finance
compensation. This is not a national resource-accounting closure.

`household_incidence_path` computes a constant consumption equivalent over the
same finite horizon for every policy. It uses separable CRRA utility of each
household's CES bundle, quarterly discount 0.99 and curvature 1 or 2. Household
budgets and baskets repeat at baseline, with no saving. The same gross
compensation budget and its financing apply each quarter, including quarters
before and after the tariff. A twelve-quarter sourcing
path is compared with the first twelve dates of a 24-quarter sourcing solve;
the acceptance tolerance is `1e-4` in baseline unit cost. This comparison
concerns the sourcing path, not unmodeled lifetime utility after date twelve.

Tests include analytical CES welfare, conservation of financing, a discrete
dynamic-programming sourcing oracle, and the constant-path equivalence of
static EV and consumption-equivalent welfare. They do not establish empirical
elasticities or general-equilibrium validity.
