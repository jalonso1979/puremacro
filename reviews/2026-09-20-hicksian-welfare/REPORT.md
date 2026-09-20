# Hicksian consumption welfare

20 September 2026. Follow-up to the
[producer/purchaser accounting correction](../2026-09-20-trade-accounting/REPORT.md).

## Implemented result

`compute_hicksian_welfare(calibration, counterfactual, base_result=baseline)`
now provides expenditure-function equivalent variation (EV), compensating
variation (CV), explicit percentage denominators and an additive endpoint
attribution. Both states must use `accounting="consistent"`. The function
re-evaluates the recorded tariff schedules and checks equilibrium, reported
quantities/prices, fiscal receipts and consumption demand before returning welfare.

The default utility basket is category 0, conventionally C. Investment is
excluded. Additional non-investment categories may be declared explicitly;
their fixed utility weights are normalized calibrated expenditure shares.
Within each category, origin sourcing is Leontief. For selected weights omega:

```
U(c) = product(c_k ** omega_k)
e(P,U) = U * product((P_k / omega_k) ** omega_k)
EV = e(P0,U1) - e(P0,U0)
CV = e(P1,U1) - e(P1,U0)
```

P includes tariffs and local final-use taxes. Positive values denote gains.
Values use the calibration's units, with separate percentages relative to
baseline consumption and GDP. This is consumption-budget welfare: with the
fixed consumption-income share kappa, a total-income transfer constrained to
maintain that allocation would need to be EV/kappa. It does not value investment
or future utility. The [full derivation](../../docs/trade_welfare.md) records this
scope and the expenditure-function sign convention.

The independently computed total is compared with an endpoint Shapley allocation
to purchaser prices, factor income and fiscal transfers. Interactions are shared
symmetrically, equivalent to averaging all six possible block orderings. Import
duties and domestic-tax rebates are subcomponents of the fiscal term; they are
not added twice. No component is defined as the residual needed to manufacture
the welfare total.

The attribution depends on the fixed numeraire and is an endpoint accounting
exercise, not a causal general-equilibrium decomposition. Accordingly, the
historical `decompose_hicksian_ev_3way` interface with TOT/Alloc/TariffRec labels
remains unavailable and points users to the new function. Theorem certification
also remains unavailable. Legacy tariff-game and flexible-model welfare proxies
are not promoted to Hicksian measures by this change.

## Independent economic reference

The benchmark uses two countries, one producing sector each, two distinct
consumption baskets and a separate investment category. It includes nonzero
foreign saving, local taxes and heterogeneous tariffs. The equilibrium reference
is the separately derived scalar relative-price solve from the preceding
accounting work, generalized to the extra final-use category.

The welfare oracle numerically minimizes **origin-level purchaser expenditure**
over deliveries and composite quantities, subject to Leontief constraints and
the target Cobb–Douglas utility. It solves all four combinations of baseline/
counterfactual prices and utility. This primal optimization does not invoke the
runtime's expenditure formula or welfare evaluator. The Shapley oracle separately
enumerates six orderings.

| Independent comparison | Maximum absolute error across countries |
|---|---:|
| Equivalent variation | 7.31e-13 |
| Compensating variation | 6.36e-12 |
| Any Shapley attribution component | 1.51e-11 |
| Runtime EV minus sum of its independently computed components | 7.11e-15 |

EV is -12.9505151480 in A and +18.7692538407 in B, in the benchmark's value
units. CV is -19.7708885928 and +31.2231084527 respectively. These are results
for the constructed model, not empirical policy estimates.

## OECD aggregation

The frozen conserving 3-region × 3-sector OECD 2023/2019 aggregation was solved
with a 10% USA import tariff and consistent accounting. No full native archive
was reloaded and no full-size GE was solved. The default C category combines
HFCE, NPISH and GGFC; treating government consumption as part of the representative
consumption basket is an explicit model assumption, not household-only data.

| Region | EV (% of baseline model consumption) |
|---|---:|
| USA | +0.1663671 |
| CHN | -0.0033792 |
| REST | +0.0274043 |

For this single Leontief basket, directly valuing the quantity difference at
baseline purchaser prices provides an independent EV check. Its maximum
discrepancy is 6.43e-8 million USD. The maximum attribution residual is
6.90e-8 million USD. These checks validate the implemented calculation under
the specified closure and aggregation, not the empirical welfare ranking's
robustness to preferences, tax institutions or external-balance assumptions.

## Solver and failure boundary

For the strong heterogeneous shock with CES elasticity 0.5, default Newton
returned a nonconverged, nonfinite state. Welfare was unavailable, as required.
Hybrid and Keller continuation converged, with EV 6.53181685960925 and
6.531816859577174 respectively. This observed solver limitation is retained in
the reference JSON rather than being omitted from the validation record.

Additional tests reject missing schedules, legacy states, altered quantities,
altered prices/income/GDP, invalid consumption choices and actual failed solves.
They check zero-weight baskets without log floors, identity, reverse comparisons,
small shocks, subsidies, units, fiscal subcomponents, CES duality and the OECD
aggregation. Changing the reported Laspeyres CPI does not change Hicksian EV.

## Execution and artifacts

- Core regression: **210 passed, 9 skipped**. This includes 36 new welfare tests.
  Skips concern unavailable optional accelerators; the existing private LP-module
  deprecation warning is unrelated to these changes.
- English and Spanish notebook 65 rebuilt and executed with outputs. Its network
  illustrations retain explicit synthetic provenance. A separate hand-balanced
  table identifies consumption and investment for the welfare demonstration.
- Notebook regression: **100 passed, 2 expected failures**. The latter are
  non-executed historical EV/theorem tests for the quarantined interfaces.
- Markdown/LaTeX technical report updated and PDF rebuilt; local links and
  `git diff --check` checked before completion.

See [results.json](results.json), [reference-run.log](reference-run.log),
[regression-tests.log](regression-tests.log), [notebook-build.log](notebook-build.log),
[notebook-tests.log](notebook-tests.log), and [commands.txt](commands.txt).

The implementation is in [welfare.py](../../puremacro/trade/welfare.py), the
independent reference in
[validate_trade_welfare.py](../../tools/reference_validation/validate_trade_welfare.py),
and the active welfare tests in
[test_trade_hicksian_welfare.py](../../tests/test_trade_hicksian_welfare.py).

Native household-only welfare, intertemporal saving utility, other fiscal
closures, flexible markups and causal TOT/efficiency attribution require separate
economic derivations. Historical quarantined tests are not counted as evidence
for the new interface.
