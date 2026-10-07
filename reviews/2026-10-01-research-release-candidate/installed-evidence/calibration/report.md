# SW07 estimator controls: calibration

Completed 2 paired samples and 8 variant fits; 1 variant fits remain unresolved.

The full published calibration generates stationary Gaussian data. All four variants use identical sample moments; finite expectations change the model map and oracle weights use the exact covariance at known generating parameters. The oracle is a diagnostic control, not a feasible covariance estimator. No original estimator is replaced and no parameter confidence intervals are reported.

## Naive chi-square reference and parameter recovery

| phase | variant | replications | usable_fits | unresolved_draws | boundary_fits | regular_fits | criterion_median | criterion_q95 | naive_events_lower | naive_events_upper | naive_rate_lower | naive_rate_upper | naive_mc95_lower | naive_mc95_upper | bias_crr | rmse_crr | bias_em | rmse_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | population_hac | 2 | 1 | 1 | 0 | 1 | 31.8077 | 31.8077 | 1 | 2 | 0.5 | 1 | 0.0125791 | 1 | 0.0517688 | 0.0517688 | -0.10751 | 0.10751 |
| calibration | finite_hac | 2 | 2 | 0 | 0 | 2 | 18.2231 | 22.9619 | 1 | 1 | 0.5 | 0.5 | 0.0125791 | 0.987421 | 0.0283058 | 0.0291335 | -0.0625111 | 0.0704405 |
| calibration | population_oracle | 2 | 2 | 0 | 0 | 2 | 9.69682 | 14.7221 | 1 | 1 | 0.5 | 0.5 | 0.0125791 | 0.987421 | 0.0153782 | 0.0153795 | -0.0141551 | 0.0210403 |
| calibration | finite_oracle | 2 | 2 | 0 | 0 | 2 | 7.83451 | 13.2045 | 0 | 0 | 0 | 0 | 0 | 0.841886 | 0.0143364 | 0.0143364 | -0.0153061 | 0.0218403 |

Rate bounds count unresolved fits both ways. Monte Carlo intervals envelope Clopper–Pearson limits. Parameter bias/RMSE and criterion quantiles condition on usable fits. A numerically regular fit is not proof of valid inference.

## Paired comparisons

| left | right | difference | paired_usable | total_pairs | mean_squared_error_difference_crr | paired_mc_se_crr | mean_squared_error_difference_em | paired_mc_se_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_hac | finite_hac | right minus left | 1 | 2 | -0.0014409 | unavailable | -0.00253716 | unavailable |
| population_oracle | finite_oracle | right minus left | 2 | 2 | -3.09948e-05 | 4.8521e-06 | 3.43036e-05 | 3.62221e-05 |
| population_hac | population_oracle | right minus left | 1 | 2 | -0.00243751 | unavailable | -0.0115563 | unavailable |
| finite_hac | finite_oracle | right minus left | 2 | 2 | -0.000643228 | 0.000389221 | -0.00448486 | 0.00453624 |

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
