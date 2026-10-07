# Synthetic GE tariff scenario with observed ENIGH baskets

This is a reproducible pedagogical model experiment, not an estimated Mexican tariff effect. Official ENIGH 2024 spending, household counts and cash wages are observed; production technology, trade sourcing, other-factor ownership and the tariff are scenario assumptions.

HOME levies a 10.0% tariff on REST food inputs and final purchases. Both states converge under consistent accounting. Tariff receipts are reconstructed from every intermediate and final transaction and checked against the solved government budget.

## Conditional equivalent variation

Fixed baskets, percent of each decile's baseline monetary consumption:

| group | bottom_four_deciles | equal_per_household | no_household_rebate |
|-------|---------------------|---------------------|---------------------|
|     I |            8.084209 |            2.380233 |           -1.422418 |
|    II |            5.472816 |            1.324264 |           -1.441437 |
|   III |            4.273186 |            0.843641 |           -1.442722 |
|    IV |            3.483619 |            0.522241 |           -1.452011 |
|     V |           -1.451974 |            0.284416 |           -1.451974 |
|    VI |             -1.4345 |            0.077983 |             -1.4345 |
|   VII |           -1.427548 |           -0.087919 |           -1.427548 |
|  VIII |           -1.398854 |           -0.260386 |           -1.398854 |
|    IX |           -1.356625 |           -0.423269 |           -1.356625 |
|     X |           -1.237983 |           -0.676741 |           -1.237983 |

Aggregate monetary EV, percent of observed aggregate consumption (not a social welfare ranking):

|                             index |    ev_pct | mean_ev_pct |     total_transfer |
|-----------------------------------|-----------|-------------|--------------------|
| fixed_baskets_no_household_rebate |  -1.37284 |   -1.406607 |                  0 |
| fixed_baskets_equal_per_household | -0.001055 |    0.398446 | 24975748018.306995 |
| fixed_baskets_bottom_four_deciles | -0.003067 |    1.300635 |    24975748018.307 |
|  cobb_douglas_no_household_rebate | -1.365819 |   -1.399383 |                  0 |
|  cobb_douglas_equal_per_household |  0.006066 |    0.405805 | 24975748018.306995 |
|  cobb_douglas_bottom_four_deciles |  0.004057 |    1.308065 |    24975748018.307 |

## Fiscal and income reconciliation

Incremental receipts are 24,975,748,018.31 MXN per quarter under the declared consumption-matching scale. Equal and bottom-four rebates exhaust the identical pool. The no-rebate comparison retains the pool outside household spending at the same GE prices; it is not a newly solved fiscal closure.

|                rule | household_rebate_mxn | retained_outside_households_mxn | fiscal_residual_mxn | income_residual_mxn |
|---------------------|----------------------|---------------------------------|---------------------|---------------------|
| no_household_rebate |                    0 |                 24975748018.307 |                   0 |            0.012291 |
| equal_per_household |   24975748018.306995 |                               0 |         -3.8147e-06 |            0.012287 |
| bottom_four_deciles |      24975748018.307 |                               0 |                   0 |            0.012287 |

Observed cash wages follow the HOME wage. An explicitly unobserved other-factor proxy follows the rental return and is allocated proportional to baseline consumption. National exposed factor income matches the GE factor accounts; per-decile baseline income/spending residuals are exported and held fixed. Transfers allocate the existing tariff receipt once. With full rebates, mapped household budget changes sum to the scaled GE budget change.

## Scope and reproduction

Preferences and redistribution vary only in household postprocessing. Prices, factor returns and production remain those of the one solved GE counterfactual. Group demand does not feed back. All groups inherit the same origin basket within each hypothetical consumption composite.

The manifest records every assumption, source provenance, unit conversion, equilibrium residual and artifact checksum. Transaction CSV and equilibrium NPZ preserve physical flows and tariff schedules for independent checks.

Observed data: [INEGI ENIGH 2024, basic tables 3.2 and 4.2](https://www.inegi.org.mx/contenidos/programas/enigh/nc/2024/tabulados/enigh2024_ns_basicos_tabulados.xlsx).
