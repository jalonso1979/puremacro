# SW07 estimator controls: calibration

Completed 399 paired samples and 1596 variant fits; 16 variant fits remain unresolved.

The full published calibration generates stationary Gaussian data. All four variants use identical sample moments; finite expectations change the model map and oracle weights use the exact covariance at known generating parameters. The oracle is a diagnostic control, not a feasible covariance estimator. No original estimator is replaced and no parameter confidence intervals are reported.

## Naive chi-square reference and parameter recovery

| phase | variant | replications | usable_fits | unresolved_draws | boundary_fits | regular_fits | criterion_median | criterion_q95 | naive_events_lower | naive_events_upper | naive_rate_lower | naive_rate_upper | naive_mc95_lower | naive_mc95_upper | bias_crr | rmse_crr | bias_em | rmse_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | population_hac | 399 | 392 | 7 | 3 | 389 | 21.022 | 127.75 | 254 | 261 | 0.636591 | 0.654135 | 0.587268 | 0.700764 | 8.44804e-05 | 0.046751 | -0.0157575 | 0.059715 |
| calibration | finite_hac | 399 | 390 | 9 | 3 | 387 | 12.4958 | 54.5047 | 172 | 181 | 0.431078 | 0.453634 | 0.381902 | 0.503926 | -0.00487871 | 0.0447791 | -0.0159526 | 0.0504017 |
| calibration | population_oracle | 399 | 399 | 0 | 0 | 399 | 7.48063 | 15.2353 | 25 | 25 | 0.0626566 | 0.0626566 | 0.0409549 | 0.0911067 | -0.00405871 | 0.0265227 | -0.0028294 | 0.0255573 |
| calibration | finite_oracle | 399 | 399 | 0 | 0 | 399 | 6.16827 | 15.4699 | 25 | 25 | 0.0626566 | 0.0626566 | 0.0409549 | 0.0911067 | -0.0050238 | 0.0296595 | -0.00472973 | 0.0265216 |

Rate bounds count unresolved fits both ways. Monte Carlo intervals envelope Clopper–Pearson limits. Parameter bias/RMSE and criterion quantiles condition on usable fits. A numerically regular fit is not proof of valid inference.

## Paired comparisons

| left | right | difference | paired_usable | total_pairs | mean_squared_error_difference_crr | paired_mc_se_crr | mean_squared_error_difference_em | paired_mc_se_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_hac | finite_hac | right minus left | 383 | 399 | -0.00029502 | 6.1933e-05 | -0.00102113 | 0.000223929 |
| population_oracle | finite_oracle | right minus left | 399 | 399 | 0.000176232 | 2.05481e-05 | 5.02195e-05 | 7.62395e-06 |
| population_hac | population_oracle | right minus left | 392 | 399 | -0.00147516 | 0.000150892 | -0.00291622 | 0.000358368 |
| finite_hac | finite_oracle | right minus left | 390 | 399 | -0.00111757 | 0.000189887 | -0.00182821 | 0.00028471 |

Squared-error differences are right minus left on common usable samples only. The paired Monte Carlo SE measures simulation variation, not parameter uncertainty. Criteria under different weight matrices have different scales.

## Observed-data descriptive fits

| variant | crr | em | objective | boundary | unresolved |
| --- | --- | --- | --- | --- | --- |
| population_hac | 0.838258 | 0.254086 | 25.2554 | False | False |
| finite_hac | 0.875753 | 0.196029 | 27.5509 | False | False |
| population_oracle | 0.804625 | 0.288166 | 31.828 | False | False |
| finite_oracle | 0.808546 | 0.286695 | 37.2833 | False | False |

These use revised historical data and a fixed published calibration. They are not independently validated economic estimates.

All starts, seeds, sample hashes, moments and full HAC arrays are retained. Atomic checkpoints authenticate numerical sources, observed data, settings and dependency versions. A small replication count is a workflow check only.
