# Observed-data conditional SW07 covariance matching

Study status: **nonregular_fit**. Objective: 396736. Conditional correct-specification null J p-value: unavailable (nonregular fit).

This study uses 156 observed US quarters, 1966Q1–2004Q4, from the revised FRED snapshot built 2026-09-30. It is not a reproduction of the original SW07 Bayesian estimates or data vintage. GDP growth, inflation and the funds rate use quarterly percent units. Nine covariance moments fit two parameters; six unused same-sample moments provide descriptive checks. The full 15×15 empirical-estimator covariance retains cross-moment dependence.

| parameter | estimate | se | ci_lower | ci_upper | boundary |
|-----------|----------|----|----------|----------|----------|
|       crr |     0.98 |    |          |          |     True |
|        em |     0.01 |    |          |          |     True |

Any displayed SE is a conditional correct-specification local asymptotic SE. Ordinary confidence intervals are withheld by default, and also for boundary, rank, optimization, flat-profile or rejected-specification cases. A missing J p-value is not evidence of fit; the usual reference law is unavailable. Passing J does not validate the calibration or establish correct specification.

All other structural parameters and independent latent-shock variances are held at the explicit Pfeifer model calibration in manifest.json. No externally identified monetary surprise enters this exercise. Population moments use an exact stationary Lyapunov solution and exclude the Kalman numerical measurement-error ridge.

profiles.csv contains nuisance-reoptimized objective slices, not confidence sets. bandwidth_sensitivity.csv uses Bartlett bandwidths 4, 8 and 12. sample_split_moments.csv compares the Great Inflation and Great Moderation periods descriptively, without a formal break test. Fixed-calibration uncertainty, vintage uncertainty, and possible regime instability remain outside ordinary inference.

An exploratory sensitivity added after the baseline boundary outcome fixes the remaining parameters to the full published SW07 posterior-mode columns in Tables 1a/b. It reestimates the same two parameters without replacing the declared baseline. calibration_sensitivity.csv and calibration_sensitivity_moments.csv show both cases. The published calibration used an overlapping original-vintage sample and is not independent validation. Residuals divided by target SEs are descriptive standardized discrepancies, not studentized fitted-residual tests.

Sources: [SW07 model and data definitions](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp722.pdf), [Pfeifer model](https://github.com/JohannesPfeifer/DSGE_mod/blob/master/Smets_Wouters_2007/Smets_Wouters_2007.mod), [Newey–West HAC](https://www.nber.org/papers/t0055), [Hansen's moment-estimation asymptotics](https://larspeterhansen.org/lph_research/large-sample-properties-of-generalized-method-of-moments-estimators/). Exact FRED series links, transformations, input hashes, calibration values and diagnostic reasons are recorded in manifest.json.
