# SW07 estimator controls: validation

Completed 999 paired samples and 3996 variant fits; 30 variant fits remain unresolved.

The full published calibration generates stationary Gaussian data. All four variants use identical sample moments; finite expectations change the model map and oracle weights use the exact covariance at known generating parameters. The oracle is a diagnostic control, not a feasible covariance estimator. No original estimator is replaced and no parameter confidence intervals are reported.

## Naive chi-square reference and parameter recovery

| phase | variant | replications | usable_fits | unresolved_draws | boundary_fits | regular_fits | criterion_median | criterion_q95 | naive_events_lower | naive_events_upper | naive_rate_lower | naive_rate_upper | naive_mc95_lower | naive_mc95_upper | bias_crr | rmse_crr | bias_em | rmse_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | population_hac | 999 | 977 | 22 | 10 | 967 | 19.3341 | 110.393 | 615 | 637 | 0.615616 | 0.637638 | 0.584645 | 0.667503 | 0.00483177 | 0.0452573 | -0.0171518 | 0.0602465 |
| validation | finite_hac | 999 | 991 | 8 | 2 | 989 | 11.9351 | 58.531 | 409 | 417 | 0.409409 | 0.417417 | 0.378722 | 0.448703 | -6.98793e-05 | 0.042362 | -0.0156104 | 0.0504009 |
| validation | population_oracle | 999 | 999 | 0 | 1 | 998 | 7.00189 | 15.0077 | 77 | 77 | 0.0770771 | 0.0770771 | 0.0613038 | 0.0953938 | -0.0025547 | 0.0271628 | -0.000522005 | 0.0262416 |
| validation | finite_oracle | 999 | 999 | 0 | 1 | 998 | 5.82251 | 16.005 | 82 | 82 | 0.0820821 | 0.0820821 | 0.0658122 | 0.100863 | -0.00320299 | 0.0300803 | -0.00231852 | 0.0271681 |

Rate bounds count unresolved fits both ways. Monte Carlo intervals envelope Clopper–Pearson limits. Parameter bias/RMSE and criterion quantiles condition on usable fits. A numerically regular fit is not proof of valid inference.

## Paired comparisons

| left | right | difference | paired_usable | total_pairs | mean_squared_error_difference_crr | paired_mc_se_crr | mean_squared_error_difference_em | paired_mc_se_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_hac | finite_hac | right minus left | 969 | 999 | -0.000246751 | 3.97756e-05 | -0.00113859 | 0.000112644 |
| population_oracle | finite_oracle | right minus left | 999 | 999 | 0.000167009 | 1.31258e-05 | 4.94879e-05 | 7.91237e-06 |
| population_hac | population_oracle | right minus left | 977 | 999 | -0.00130415 | 0.000146594 | -0.0029335 | 0.000234974 |
| finite_hac | finite_oracle | right minus left | 991 | 999 | -0.000890257 | 0.000141706 | -0.0017982 | 0.000166195 |

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
