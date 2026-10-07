# Independent audit of the GE distributional application

**Result: passed; no blocking accounting or welfare issue found.**

The [audit script](independent_audit.py) reads only the exported CSV, NPZ and
manifest files. It does not import the application, GE solver, household-price
bridge or welfare routines. The [machine-readable results](independent_audit.json)
record all checks. Twelve artifact checksums and seven current numerical-source
checksums match the manifest.

## Prices and factor-income mapping

Purchaser prices were independently rebuilt by weighting the saved producer
prices and tariff multipliers by each composite's baseline origin basket.
They match the published price CSV: food rises 2.5%; the other seven composites
are unchanged up to numerical precision.

This result also matches a closed-form implication of the synthetic economy.
Writing `s_food` for the national food expenditure share, HOME's wage and rental
return both change by

`−(.35/.65) × .20 × s_food × .10 = −0.004173696286676192`.

All producer prices remain one. The HOME factor-income loss is offset by the
tariff receipts from intermediate inputs; final-food tariff receipts increase
the household expenditure budget. Thus the price and wage results follow the
declared fixed-coefficient symmetric economy, rather than an independently
estimated pass-through relationship.

Cash-wage exposures match the observed ENIGH cash wages. The other-factor
exposure was independently reconstructed as `(1 − aggregate wage share) ×
group consumption`. Its total matches the model capital account. It remains
an explicitly assumed ownership allocation, and per-decile baseline income
need not equal expenditure. Those group residuals are held fixed and sum to
zero nationally within floating-point precision.

## Physical transactions and fiscal accounts

Duties were summed directly over saved physical intermediate and final flows,
each multiplied by its producer price and applicable tariff rate. Only REST
food delivered to HOME is taxed. HOME accounts, in model units, are:

| Account | Baseline | Counterfactual |
|---|---:|---:|
| Factor income | 100.0000000000 | 99.5826303714 |
| Intermediate-import duties | 0 | 0.4173696287 |
| Final-import duties | 0 | 0.9688937808 |
| Purchaser expenditure | 100.0000000000 | 100.9688937809 |

REST's purchaser expenditure and factor income both remain 100. The maximum
country budget residual is `6.79e−13` model units. Summed duties reproduce both
the reported tariff receipts and the government account.

The incremental HOME pool is **24,975,748,018.307 MXN per quarter** under the
declared consumption-matching normalization. Exposed income contains only
factor earnings, so allocating this pool through the transfer term counts
the receipts once. Equal and bottom-four rebates exhaust the same pool.

Mapped household consumption plus factor-income changes plus the full pool
matches the independent purchaser-valued model budget to **0.01172 MXN**,
or `6.45e−15` relative error at the national scale. The no-rebate comparison
reconciles only after including the receipts retained outside household
spending. It is correctly described as conditional on the same GE prices,
not as a separately solved government-saving closure.

## Independent EV and CV

For every decile in all six scenarios, the audit reconstructs the relative
prices, factor-income changes and transfer allocations without using the
reported welfare values. Exact homothetic expenditure formulas reproduce
both EV and CV to at most `1.57e−11` MXN per household.

A second route solves expenditure minimization numerically: linear programs
for fixed baskets and constrained nonlinear optimization for Cobb–Douglas.
Across all 60 decile/scenario observations, the largest numerical difference
is **`9.07e−9` MXN per household**, for CV. EV/CV component decompositions and
household-weighted aggregate totals also reconcile.

## Interpretation limits

- Even the full-rebate comparisons hold the solved prices and factor returns
  fixed as preferences and redistribution change. Group demand is not fed
  back into GE; these are conditional household-incidence calculations.
- The technology, import sourcing, factor ownership proxy and tariff are
  assumptions. Observed ENIGH baskets do not make the resulting tariff
  effect an empirical estimate for Mexico.
- Group means omit within-decile heterogeneity. Monetary EV and CV totals
  do not establish a social welfare ranking.

Reproduce with:

```bash
.venv/bin/python reviews/2026-10-01-ge-distributional/independent_audit.py
```
