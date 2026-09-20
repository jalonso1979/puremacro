# Trade price and fiscal accounting correction

20 September 2026. Follow-up to the
[structural review](../2026-09-19-critical-evaluation/REVIEW.md) and
[live Dynare/native OECD validation](../2026-09-20-independent-validation/REPORT.md).

## Implemented result

`solve_trade_equilibrium(..., accounting="consistent")` now solves a coherent
producer/purchaser model with lump-sum fiscal rebates. The default remains
`accounting="legacy"` to preserve the historical MATLAB replication surface.
Selecting `tariff_revenue_mode="schedule"` alone does not select the new model.
The [equation documentation](../../docs/trade_accounting.md) describes the
calibration, closure, supported solvers and reporting fields.

The correction addresses five linked problems in the historical formulation:

1. Final deliveries and their duties used destination composite prices rather
   than the exporting good's producer price. Intermediate reporting also used
   implied unit costs in places where actual solved prices were required.
2. Production-tax receipts used output quantities despite being calibrated as
   a share of output revenue. Correct receipts are `tax * price * output`.
3. MATLAB's wage-expression precedence was not homogeneous in nominal prices.
   The corrected Cobb–Douglas expression raises the entire normalized wage to
   its labor-share exponent.
4. The legacy final-tax denominator includes foreign saving in investment.
   The consistent share uses actual tax-inclusive final purchases, excluding
   financial saving, and reproduces baseline taxes.
5. Correcting those equations exposes unspecified foreign-transfer choices and
   the overall nominal price scale. The new closure fixes baseline foreign
   saving in numeraire units and fixes the first producer price to one.

The final closure is a substantive economic assumption. It is recorded in
result metadata. The first goods equation is replaced in the square numerical
system by the numeraire equation; all goods equations, factor markets, fiscal
budgets and realized net exports are audited after solving. An infeasible or
inadequately balanced state cannot return `converged=True` merely because the
reduced numerical system converged.

Producer value plus duties plus local final tax reconciles with purchaser
spending. Import duties are rebated to the importing country exactly once.
Household and government budgets, fixed foreign balances and income versus
expenditure GDP are exposed in `metadata["account_residuals"]`. Nominal transaction
tables now contain nominal factor payments and tax receipts throughout.

## Independent two-country reference

The [reference program](../../tools/reference_validation/validate_trade_accounting.py)
starts from a hand-balanced table with two countries, one producing sector per
country, two final uses, nonzero foreign saving, production/final taxes, and
tariffs that differ across countries and final-use categories.

The oracle does not call puremacro's calibration, equilibrium evaluator or GE
solver. Factor clearing fixes output in this one-sector Leontief example.
Conditional on a relative producer price, it solves household incomes
algebraically, including rebates, then clears one goods market with a scalar
Brent solve. It independently checks the other goods market. The full GE
implementation is then compared with those separately derived prices, factor
payments, receipts, incomes and deliveries.

| Check, baseline and heterogeneous-tariff cases | Largest absolute discrepancy |
|---|---:|
| Producer prices | 4.89e-15 |
| Purchaser prices | 5.56e-15 |
| Tariff receipts | 3.56e-14 |
| Final deliveries | 7.82e-14 |
| Any compared field (household income) | 5.69e-13 |
| Any reported accounting/physical equation | 5.69e-14 |

Under the heterogeneous shock, the relative producer price of country B is
1.0328401217499743. Duties are 14.603438682042407 in A and
7.087466847293019 in B. Baseline producer prices, wages and rents recover one.
The reduced-system Jacobian has full numerical rank at this benchmark.
Results are in [two-country-results.json](two-country-results.json).

## Empirical OECD aggregation

The conserving 3-region × 3-sector fixture from the previously validated OECD
2023 regular / 2019 native archive was rerun with consistent accounting. This
run did **not** reload the full native archive and did not solve full-size GE.
USA, CHN and REST include every source economy; resources, manufacturing and
services include every source activity. The prior report and fixture manifest
retain source hashes, mappings and calibration adjustments.

Baseline recovery, an independent Leontief inversion, a 10% USA import tariff,
Newton/Keller continuation agreement, producer-price duty reconstruction and
million-to-billion USD scaling all pass. The counterfactual's maximum goods
error is 9.88e-8 million USD, foreign-balance error 8.10e-8, government-budget
error 1.87e-9, and income/expenditure GDP discrepancy 8.95e-8. Detailed results
are in [oecd-consistent-results.json](oecd-consistent-results.json).

The OECD production-column TLS row aggregates taxes less subsidies; interpreting
it as an output-revenue tax is an explicit modeling assumption. This does not
identify the original product/VAT schedule or validate published policy effects.
Regularization remains recorded; the adjusted calibration is not presented as
the untouched provider table.

## Tests and reproduction

The focused accounting suite has **33 passing tests**. It includes the separate
oracle, all supported solver methods, wrong-price-base counterexamples, nominal
homogeneity, closure rank, budget/GDP identities, CES sourcing at elasticities
0.5/1/2, a one-country zero-intermediate case, calibration without the original
table, currency scaling, invalid inputs and explicit unsupported-mode failures.

The broader regression command, including those tests, has **174 passing tests
and 9 skips**. The skips concern optional accelerator cases whose backends are
unavailable; they are not counted as successful validation.
The public-API check emits the existing private `lp.garch_utils` deprecation
warning. See [targeted-tests.log](targeted-tests.log) for the complete outcome.

An exploratory CES Newton solve at a 1e-10 requested tolerance met the square
system criterion but had a 1.80e-10 realized-balance discrepancy. The audit
correctly returned `converged=False`; solving at 1e-12 passed. A reduced-system
termination flag is deliberately insufficient, and tight tolerances can require
further refinement. No post-solve accounting tolerance was relaxed to accept it.

Reproduction commands are in [commands.txt](commands.txt). The technical report's
Markdown/LaTeX descriptions and PDF were updated. `git diff --check` passes.

## Scope still open

The new mode supports NumPy, Leontief/CES intermediate sourcing and lump-sum
rebates. Full Newton, sparse LU, Krylov, Broyden, SciPy hybrid/LM and full Keller
continuation share its equations. `method="condensed"` explicitly records its
fallback to full Newton. Quasi-condensed, GPU, capacity costs and other fiscal
recycling schemes are unavailable in consistent mode. Historical policy wrappers
retain their conventions unless they explicitly select the new accounting.

This change does not restore Hicksian EV decomposition or theorem certification.
Those endpoints remain quarantined. Full-size native GE, richer fiscal closures,
flexible markups and independent policy/welfare replication remain separate work.
