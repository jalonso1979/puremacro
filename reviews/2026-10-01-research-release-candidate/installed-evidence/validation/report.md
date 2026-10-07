# SW07 estimator controls: validation

Completed 2 paired samples and 8 variant fits; 0 variant fits remain unresolved.

The full published calibration generates stationary Gaussian data. All four variants use identical sample moments; finite expectations change the model map and oracle weights use the exact covariance at known generating parameters. The oracle is a diagnostic control, not a feasible covariance estimator. No original estimator is replaced and no parameter confidence intervals are reported.

## Naive chi-square reference and parameter recovery

| phase | variant | replications | usable_fits | unresolved_draws | boundary_fits | regular_fits | criterion_median | criterion_q95 | naive_events_lower | naive_events_upper | naive_rate_lower | naive_rate_upper | naive_mc95_lower | naive_mc95_upper | bias_crr | rmse_crr | bias_em | rmse_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | population_hac | 2 | 2 | 0 | 0 | 2 | 17.3079 | 28.6342 | 1 | 1 | 0.5 | 0.5 | 0.0125791 | 0.987421 | -0.0455011 | 0.0516675 | 0.0296455 | 0.0496998 |
| validation | finite_hac | 2 | 2 | 0 | 0 | 2 | 9.01652 | 12.5115 | 0 | 0 | 0 | 0 | 0 | 0.841886 | -0.056774 | 0.0681662 | 0.0125364 | 0.0299708 |
| validation | population_oracle | 2 | 2 | 0 | 0 | 2 | 6.75606 | 6.76586 | 0 | 0 | 0 | 0 | 0 | 0.841886 | -0.0140778 | 0.016326 | 0.000207169 | 0.0117716 |
| validation | finite_oracle | 2 | 2 | 0 | 0 | 2 | 5.44774 | 6.11682 | 0 | 0 | 0 | 0 | 0 | 0.841886 | -0.0153631 | 0.0180554 | -0.00180776 | 0.0115837 |

Rate bounds count unresolved fits both ways. Monte Carlo intervals envelope Clopper–Pearson limits. Parameter bias/RMSE and criterion quantiles condition on usable fits. A numerically regular fit is not proof of valid inference.

## Paired comparisons

| left | right | difference | paired_usable | total_pairs | mean_squared_error_difference_crr | paired_mc_se_crr | mean_squared_error_difference_em | paired_mc_se_em |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_hac | finite_hac | right minus left | 2 | 2 | 0.0019771 | 0.00205628 | -0.00157182 | 0.00168257 |
| population_oracle | finite_oracle | right minus left | 2 | 2 | 5.94582e-05 | 5.86677e-05 | -4.38793e-06 | 4.62447e-05 |
| population_hac | population_oracle | right minus left | 2 | 2 | -0.002403 | 0.00199479 | -0.0023315 | 0.00236024 |
| finite_hac | finite_oracle | right minus left | 2 | 2 | -0.00432064 | 0.00399241 | -0.000764067 | 0.000723922 |

Squared-error differences are right minus left on common usable samples only. The paired Monte Carlo SE measures simulation variation, not parameter uncertainty. Criteria under different weight matrices have different scales.

## Observed-data descriptive fits

| variant | crr | em | objective | boundary | unresolved |
| --- | --- | --- | --- | --- | --- |
| population_hac | 0.838258 | 0.254086 | 25.2554 | False | False |
| finite_hac | 0.875753 | 0.196029 | 27.5509 | False | False |
| population_oracle | 0.804625 | 0.288166 | 31.828 | False | False |
| finite_oracle | 0.808546 | 0.286695 | 37.2833 | False | False |

These use revised historical data and a fixed published calibration. They are not independently validated economic estimates.

## Independent validation of calibration cutoffs

| variant | calibration_draws | validation_draws | calibration_unresolved | validation_unresolved | critical_rank | critical_lower | critical_upper | critical_lower_unbounded | critical_upper_unbounded | calibrated_events_lower | calibrated_events_upper | calibrated_rate_lower | calibrated_rate_upper | calibrated_mc95_lower | calibrated_mc95_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| population_hac | 2 | 2 | 1 | 0 | 3 | unavailable | unavailable | True | True | 0 | 0 | 0 | 0 | 0 | 0.841886 |
| finite_hac | 2 | 2 | 0 | 0 | 3 | unavailable | unavailable | True | True | 0 | 0 | 0 | 0 | 0 | 0.841886 |
| population_oracle | 2 | 2 | 0 | 0 | 3 | unavailable | unavailable | True | True | 0 | 0 | 0 | 0 | 0 | 0.841886 |
| finite_oracle | 2 | 2 | 0 | 0 | 3 | unavailable | unavailable | True | True | 0 | 0 | 0 | 0 | 0 | 0.841886 |

Cutoffs use calibration order rank ceil(0.95*(B+1)), with strict rejection above the cutoff. Unresolved calibration criteria lie between zero and infinity, producing lower/upper cutoffs. Validation rejection bounds additionally retain unresolved validation fits. Monte Carlo intervals condition on the realized calibration pool and do not include all cutoff-estimation uncertainty. Infinite cutoffs mean the reference sample is too small or too incomplete; JSON null and explicit unbounded flags distinguish this from a finite cutoff. This is a fixed-DGP experiment, not a composite-null test.

All starts, seeds, sample hashes, moments and full HAC arrays are retained. Atomic checkpoints authenticate numerical sources, observed data, settings and dependency versions. A small replication count is a workflow check only.
