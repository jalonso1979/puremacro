# Audited GE-to-household distributional application

The application now connects a converged tariff equilibrium to household
purchaser prices, explicitly mapped nominal factor earnings and a reconciled
tariff-revenue pool. It produces six conditional incidence scenarios and preserves
the inputs needed to verify their accounting independently.

**The GE economy is synthetic; the ENIGH baskets are observed.** The two
hypothetical regions and eight consumption composites are a pedagogical model,
not an empirical Mexican trade calibration. Technology, import sourcing,
other-factor ownership and the 10% tariff are disclosed assumptions. Official
ENIGH 2024 spending, household counts and cash wages retain their original
provenance and checksum. This exercise advances the complete application and
its accounting, not empirical identification of an actual tariff's effects.

## Default conditional results

HOME taxes food imported from REST in both intermediate and final uses. Both
states converge under consistent accounting with solve tolerance `1e-10`.
The largest reported counterfactual accounting residual is `1.83e-11` model
units. Household comparisons keep the same solved prices and factor returns;
redistribution and preference changes do not feed back into equilibrium.

| Outcome | Default result |
|---|---:|
| Food purchaser-price change | +2.500% |
| HOME wage and rental-return change | −0.417370% |
| Scaled aggregate factor-income change | −7,519,580,047.46 MXN per quarter |
| Incremental tariff-revenue pool | 24,975,748,018.31 MXN per quarter |
| Scaled GE final-budget change | +17,456,167,970.83 MXN per quarter |

MXN magnitudes use a declared normalization: one model value unit equals
18,016,596,158.48 MXN per quarter, chosen to match the 100-unit HOME consumption
budget to the observed eight-category aggregate. This is not an exchange rate,
national-accounts estimate or empirical revenue forecast.

Fixed-basket equivalent variation, percent of each decile's baseline consumption:

| Decile | No household rebate | Equal per household | Bottom four deciles |
|---|---:|---:|---:|
| I | −1.422418 | +2.380233 | +8.084209 |
| IV | −1.452011 | +0.522241 | +3.483619 |
| VII | −1.427548 | −0.087919 | −1.427548 |
| X | −1.237983 | −0.676741 | −1.237983 |

Equal rebates make the lower six decile representatives better off in this
scenario; targeting the bottom four raises their gains and leaves the other
groups without compensation. These are representative-decile comparisons, not
fractions of individual households who gain or lose.

Aggregate monetary EV is small and changes sign with the preference assumption:

| Preferences | Equal rebate | Bottom-four rebate |
|---|---:|---:|
| Fixed baskets | −0.001055% | −0.003067% |
| Cobb–Douglas | +0.006066% | +0.004057% |

These aggregate monetary measures are not a social welfare ranking. The
population-weighted average of group percentage EV is a different statistic.
No sampling intervals or within-decile heterogeneity can be recovered from the
published means used here.

## Income and fiscal reconciliation

Observed cash wages receive the model wage change. The synthetic economy's
labor share is set to aggregate cash wages divided by observed consumption,
0.9119023. Remaining factor income is an explicitly **unobserved other-factor
proxy**, allocated proportional to group consumption and receiving the model
rental-return change. This balances the modeled national factor accounts; it
does not estimate survey capital ownership. Baseline exposure minus consumption
is recorded separately for every decile, held fixed, and sums to zero nationally.

The tariff pool is reconstructed directly as rate × producer price × physical
quantity for every intermediate and final delivery. It matches solved tariff
revenue and the government budget, which contains no other taxes in this
scenario. Income exposure includes factor earnings only. The existing solved
tariff receipt is allocated once through the household transfer argument.

The equal and targeted rules exhaust the same pool. For the no-rebate reference,
that pool remains outside household spending while prices are held fixed; this
is not a separately solved government-saving closure. Every rule verifies both:

```text
household rebate + retained receipts = incremental tariff revenue
household budget change + retained receipts = scaled GE budget change
```

The largest allocation residual is `3.82e-6` MXN. The largest aggregate income
reconciliation residual is about `0.0123` MXN against a baseline consumption
total of `1.80166e12` MXN, reflecting the scaled numerical equilibrium tolerance.
Residuals are exported rather than rounded to zero.

## Independent checks

[The targeted test log](tests.log) records **39 passing tests**, including 12
new application tests and the existing distributional/ENIGH checks. New tests
cover raw-table balance, zero-tariff recovery, convergence refusal, independent
physical-flow tariff reconstruction, factor-account reconciliation, transfer
allocation, analytic fixed-basket and Cobb–Douglas EV, artifact checksums and
preservation of observed versus synthetic provenance.

[A separate artifact-only audit](independent_audit.md) reconstructs purchaser
prices, both regions' budgets, both duty channels, factor mappings and all 60
EV/CV outcomes. It also verifies expenditure functions by independent linear
and nonlinear optimization. Its largest welfare discrepancy is `9.07e-9` MXN
per household; the aggregate budget discrepancy is `0.01172` MXN nationally.
[Both executable documentation examples pass](docs-execution-tests.log).

An additional closed-form check exploits this transparent symmetric model:
all producer prices and physical outputs remain unchanged, while HOME factor
returns fall by `(.35/.65) × .20 × food_budget_share × .10`. Intermediate-import
duties exactly offset that aggregate factor-income loss; final-import duties
fund the increased nominal cost of the unchanged aggregate final basket. The
numerical solution matches these analytic identities. Thus the 2.5% food price
increase follows this model's fixed coefficients and symmetric technology; it
is not inserted directly into household incidence.

## Reproduce and inspect

```bash
MPLBACKEND=Agg python -m puremacro.examples.distributional_trade_ge \
  --output research_output/distributional_trade_ge --tariff-rate 0.10
```

- [Readable application report](application/report.md) and [incidence figure](application/conditional_incidence.png).
- [Manifest](application/manifest.json): assumptions, official source, source/runtime fingerprints and artifact hashes.
- [Tariff transactions](application/tariff_transactions.csv) and [revenue audit](application/revenue_audit.csv).
- [Household income mapping](application/income_mapping.csv) and [fiscal allocation](application/fiscal_allocation.csv).
- [All 60 group outcomes](application/all_scenarios.csv), [aggregate results](application/aggregate_results.csv) and [purchaser prices](application/purchaser_prices.csv).
- [Physical equilibrium evidence](application/equilibrium_evidence.npz) and [synthetic input table](application/synthetic_calibration_table.csv).

The English and Spanish guides are `docs/distributional_trade_ge.md` and its
`docs/es/` counterpart. The application is offline and uses the existing
audited household bridge. A genuine empirical tariff application would still
require compatible production/trade accounts, measured origin shares, a sector
concordance and defensible ownership/income mappings.
