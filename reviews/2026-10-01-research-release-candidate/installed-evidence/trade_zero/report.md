# Synthetic GE tariff scenario with observed ENIGH baskets

This is a reproducible pedagogical model experiment, not an estimated Mexican tariff effect. Official ENIGH 2024 spending, household counts and cash wages are observed; production technology, trade sourcing, other-factor ownership and the tariff are scenario assumptions.

HOME levies a 0.0% tariff on REST food inputs and final purchases. Both states converge under consistent accounting. Tariff receipts are reconstructed from every intermediate and final transaction and checked against the solved government budget.

## Conditional equivalent variation

Fixed baskets, percent of each decile's baseline monetary consumption:

| group | bottom_four_deciles | equal_per_household | no_household_rebate |
|-------|---------------------|---------------------|---------------------|
|     I |         2.22045e-14 |         2.22045e-14 |         2.22045e-14 |
|    II |         2.22045e-14 |         2.22045e-14 |         2.22045e-14 |
|   III |        -2.22045e-14 |        -2.22045e-14 |        -2.22045e-14 |
|    IV |                   0 |                   0 |                   0 |
|     V |                   0 |                   0 |                   0 |
|    VI |                   0 |                   0 |                   0 |
|   VII |                   0 |                   0 |                   0 |
|  VIII |         2.22045e-14 |         2.22045e-14 |         2.22045e-14 |
|    IX |                   0 |                   0 |                   0 |
|     X |                   0 |                   0 |                   0 |

Aggregate monetary EV, percent of observed aggregate consumption (not a social welfare ranking):

|                             index |      ev_pct | mean_ev_pct | total_transfer |
|-----------------------------------|-------------|-------------|----------------|
| fixed_baskets_no_household_rebate | 3.24621e-15 | 4.44089e-15 |              0 |
| fixed_baskets_equal_per_household | 3.24621e-15 | 4.44089e-15 |              0 |
| fixed_baskets_bottom_four_deciles | 3.24621e-15 | 4.44089e-15 |              0 |
|  cobb_douglas_no_household_rebate |           0 |           0 |              0 |
|  cobb_douglas_equal_per_household |           0 |           0 |              0 |
|  cobb_douglas_bottom_four_deciles |           0 |           0 |              0 |

## Fiscal and income reconciliation

Incremental receipts are 0.00 MXN per quarter under the declared consumption-matching scale. Equal and bottom-four rebates exhaust the identical pool. The no-rebate comparison retains the pool outside household spending at the same GE prices; it is not a newly solved fiscal closure.

|                rule | household_rebate_mxn | retained_outside_households_mxn | fiscal_residual_mxn | income_residual_mxn |
|---------------------|----------------------|---------------------------------|---------------------|---------------------|
| no_household_rebate |                    0 |                               0 |                   0 |                   0 |
| equal_per_household |                    0 |                               0 |                   0 |                   0 |
| bottom_four_deciles |                    0 |                               0 |                   0 |                   0 |

Observed cash wages follow the HOME wage. An explicitly unobserved other-factor proxy follows the rental return and is allocated proportional to baseline consumption. National exposed factor income matches the GE factor accounts; per-decile baseline income/spending residuals are exported and held fixed. Transfers allocate the existing tariff receipt once. With full rebates, mapped household budget changes sum to the scaled GE budget change.

## Scope and reproduction

Preferences and redistribution vary only in household postprocessing. Prices, factor returns and production remain those of the one solved GE counterfactual. Group demand does not feed back. All groups inherit the same origin basket within each hypothetical consumption composite.

The manifest records every assumption, source provenance, unit conversion, equilibrium residual and artifact checksum. Transaction CSV and equilibrium NPZ preserve physical flows and tariff schedules for independent checks.

Observed data: [INEGI ENIGH 2024, basic tables 3.2 and 4.2](https://www.inegi.org.mx/contenidos/programas/enigh/nc/2024/tabulados/enigh2024_ns_basicos_tabulados.xlsx).
