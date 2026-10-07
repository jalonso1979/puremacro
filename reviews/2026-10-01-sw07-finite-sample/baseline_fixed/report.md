# SW07 finite-sample covariance-matching diagnostic

Computation status: **computational outcomes unresolved**. Unresolved draws: **4**.

Requested simulations per scenario: **399**. Seed: **20261001**. Bartlett HAC bandwidth used for fitting: **8**.

Each simulation repeats the 156-quarter sample, full-sample demeaning, common lag window, covariance estimation, parameter bounds, and four optimizer starts. Nine moments estimate monetary-policy smoothing and innovation volatility; six unused same-sample moments remain descriptive checks. Simulation starts from the stationary model distribution and excludes economic measurement error.

## Scenario results

| scenario | replications | usable_fits | unresolved_draws | boundary_fits | numerically_regular_fits | observed_objective | criterion_median | criterion_q95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_fixed | 399 | 395 | 4 | 91 | 304 | 396736 | 8.95506 | 43.6875 |

## Tail fractions and Monte Carlo uncertainty

| scenario | tail_exceedances | tail_fraction_lower | tail_fraction_upper | tail_mc_95_lower | tail_mc_95_upper |
| --- | --- | --- | --- | --- | --- |
| baseline_fixed | 0 | 0 | 0.0100251 | 0 | 0.025468 |

Fixed-calibration scenarios examine a specified Gaussian model and its fixed parameters. Fitted scenarios use values estimated from the observed data; their tail fractions are descriptive plug-in diagnostics, not validated bootstrap p-values or composite-null tests. Published calibration values use overlapping historical data and do not provide independent validation.

Binomial Monte Carlo intervals describe uncertainty from a finite number of simulation draws. They are not parameter confidence intervals and do not include calibration, model, historical-regime, or data-vintage uncertainty. A small simulation count supports only a coarse tail estimate; zero exceedances do not establish a zero population tail probability.

The lower and upper tail fractions count unresolved draws as no exceedances and all exceedances, respectively. The 95% Clopper–Pearson limits include both possibilities.

## Asymptotic-rule diagnostics

| scenario | naive_chi2_rate_lower | naive_chi2_rate_upper | numerically_regular_fits | regular_j_rate_conditional |
| --- | --- | --- | --- | --- |
| baseline_fixed | 0.333333 | 0.343358 | 304 | 0.240132 |

The naive chi-square(7) rule deliberately includes usable boundary fits and must not be interpreted as valid boundary inference. The regular J rejection rate uses only the reported numerically regular subset; selection changes its denominator. Neither rate proves general test size or coverage.

## Parameter recovery among usable fits

| scenario | true_crr | bias_crr_usable | rmse_crr_usable | true_em | bias_em_usable | rmse_em_usable |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_fixed | 0.8762 | 0.0209474 | 0.0685517 | 0.2397 | -0.077688 | 0.160713 |

Bias and RMSE are conditional on usable optimization and are not parameter confidence intervals.

## Observed-data fits

| calibration | crr | em | objective | optimizer_success | numerically_regular | boundary | j_pvalue | unresolved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.98 | 0.01 | 396736 | True | False | True | unavailable | False |

The criterion distribution plot shows the usable simulated criteria and the observed-data criterion on a common scale within each scenario. Read it together with the summary failure counts: unavailable draws are retained in draws.csv and must not be treated as successful non-exceedances.

## Moment and covariance evidence

moment_audit.csv compares observed moments, stationary population targets, exact finite-sample expectations, and simulated empirical moments. covariance_audit.csv compares exact Gaussian finite-sample covariance, simulation covariance of empirical moment estimators, and mean Bartlett HAC estimates. Bandwidths 4, 8, and 12 are audited without reestimating parameters for each audit bandwidth. These diagnostics expose finite-sample demeaning and weighting effects; they do not alter the original targets or declare a rejected model valid.

The declared Pfeifer initialization and the published SW07 mode represent different fixed calibrations. The independent data/mapping audit found no quarterly-percent unit mismatch. It identified the declared initialization's large risk-premium innovation variance as a major source of its poor empirical fit. Persistence and finite-sample demeaning remain separate diagnostic questions.

The source snapshot is revised FRED data for 1966Q1–2004Q4, not the original SW07 vintage. This exercise neither repeats the paper's Bayesian estimation nor externally identifies a monetary shock.

Files: summary.csv, draws.csv, observed_fits.csv, moment_audit.csv, covariance_audit.csv, draw_evidence.npz, criterion_distribution.png, and manifest.json. The manifest contains seeds, scenario definitions, source provenance, parameter choices, diagnostics, array shapes, and artifact hashes. The array_row column in draws.csv maps each replication to its NPZ row. Moment-audit draw counts can exceed usable fit counts when moments are available but optimization remains unresolved.

Sources: [SW07 model and data appendix](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf), [Pfeifer model source](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007.mod).
