# Independent saved-result audit

PASS. This script imports no puremacro production code and performs no refitting. It independently reconstructs statistics from the authenticated phase evidence.

Calibration contains 399 shared samples; validation contains 999 independently seeded shared samples. Each sample has exactly four variant records referring to the same seed, sample hash and NPZ array row. All numerical-source fingerprints and artifact hashes match the checkout; checkpoint arrays match the exported arrays exactly.

## Assessment of the predeclared comparisons

Exact finite-sample expectations reduce naive chi-square overrejection with HAC weights, from 61.56–63.76% to 40.94–41.74%, but leave severe distortion. Oracle covariance weights remove most of that distortion at this known DGP, but the naive rejection rates remain 7.71% and 8.21%; 5% lies below both corresponding 95% Monte Carlo intervals. Oracle weighting alone therefore does not restore the nominal chi-square reference.

After applying the frozen simulation-reference cutoffs, the two oracle validation fractions are 4.90% and 5.91%, with conditional Monte Carlo intervals containing 5%. The HAC variants have wider failure bounds. This evidence concerns the one fixed generating calibration; it is not a composite-null test or feasible estimated covariance.

Finite expectations improve paired parameter squared errors under HAC weighting, but increase them for both parameters under oracle weighting. The exact expectation map is therefore not a universal parameter-recovery improvement. Changing the covariance control gives the larger recovery improvements in this experiment.

All 46 unresolved variant fits are selected-result first-order stationarity failures. Every optimizer start reported success and successful-start objectives agreed, so an audit based on optimizer success flags alone would have missed these unresolved cases. They remain in the reported failure and rejection bounds.

## Naive chi-square reference and parameter recovery

All saved summary and paired-comparison numbers reproduce within serialization/numerical tolerance. Unresolved fits remain in rate denominators; parameter bias and RMSE condition on usable fits, and paired squared-error changes condition on common usable draws.

```text
      phase           variant  replications  usable_fits  unresolved_draws  boundary_fits  regular_fits  criterion_median  criterion_q95  naive_events_lower  naive_events_upper  naive_rate_lower  naive_rate_upper  naive_mc95_lower  naive_mc95_upper  bias_crr  rmse_crr   bias_em  rmse_em
calibration    population_hac           399          392                 7              3           389         21.022034     127.750073                 254                 261          0.636591          0.654135          0.587268          0.700764  0.000084  0.046751 -0.015757 0.059715
calibration        finite_hac           399          390                 9              3           387         12.495754      54.504664                 172                 181          0.431078          0.453634          0.381902          0.503926 -0.004879  0.044779 -0.015953 0.050402
calibration population_oracle           399          399                 0              0           399          7.480633      15.235250                  25                  25          0.062657          0.062657          0.040955          0.091107 -0.004059  0.026523 -0.002829 0.025557
calibration     finite_oracle           399          399                 0              0           399          6.168268      15.469869                  25                  25          0.062657          0.062657          0.040955          0.091107 -0.005024  0.029659 -0.004730 0.026522
 validation    population_hac           999          977                22             10           967         19.334133     110.393021                 615                 637          0.615616          0.637638          0.584645          0.667503  0.004832  0.045257 -0.017152 0.060246
 validation        finite_hac           999          991                 8              2           989         11.935061      58.530964                 409                 417          0.409409          0.417417          0.378722          0.448703 -0.000070  0.042362 -0.015610 0.050401
 validation population_oracle           999          999                 0              1           998          7.001886      15.007666                  77                  77          0.077077          0.077077          0.061304          0.095394 -0.002555  0.027163 -0.000522 0.026242
 validation     finite_oracle           999          999                 0              1           998          5.822513      16.004956                  82                  82          0.082082          0.082082          0.065812          0.100863 -0.003203  0.030080 -0.002319 0.027168
```

## Independent calibration cutoffs

The predeclared order rank is 380 of 399. Missing calibration criteria are bounded below all known criteria and above all known criteria. Validation uses strict exceedance, with missing validation fits counted both ways. Clopper–Pearson envelopes condition on the realized calibration pool and do not capture all cutoff-estimation uncertainty.

All four cutoffs match the frozen calibration_reference.json saved at 2026-10-01T21:53:48.308347+00:00, before validation finished. No validation outcomes or method selection enter that reference.

```text
          variant  calibration_draws  validation_draws  calibration_unresolved  validation_unresolved  critical_rank  critical_lower  critical_upper  critical_lower_unbounded  critical_upper_unbounded  calibrated_events_lower  calibrated_events_upper  calibrated_rate_lower  calibrated_rate_upper  calibrated_mc95_lower  calibrated_mc95_upper
   population_hac                399               999                       7                     22            380      127.973455      162.631702                     False                     False                       22                       55               0.022022               0.055055               0.013851               0.071062
       finite_hac                399               999                       9                      8            380       54.637589       86.675796                     False                     False                       23                       65               0.023023               0.065065               0.014649               0.082182
population_oracle                399               999                       0                      0            380       15.280468       15.280468                     False                     False                       49                       49               0.049049               0.049049               0.036504               0.064330
    finite_oracle                399               999                       0                      0            380       15.480962       15.480962                     False                     False                       59                       59               0.059059               0.059059               0.045259               0.075524
```

## Paired parameter recovery

Negative right-minus-left squared-error differences favor the right variant on the reported common usable sample. These comparisons are descriptive; no variant is selected after seeing validation and presented as independently validated.

```text
      phase              left             right       difference  paired_usable  total_pairs  mean_squared_error_difference_crr  paired_mc_se_crr  mean_squared_error_difference_em  paired_mc_se_em
calibration    population_hac        finite_hac right minus left            383          399                          -0.000295          0.000062                         -0.001021         0.000224
calibration population_oracle     finite_oracle right minus left            399          399                           0.000176          0.000021                          0.000050         0.000008
calibration    population_hac population_oracle right minus left            392          399                          -0.001475          0.000151                         -0.002916         0.000358
calibration        finite_hac     finite_oracle right minus left            390          399                          -0.001118          0.000190                         -0.001828         0.000285
 validation    population_hac        finite_hac right minus left            969          999                          -0.000247          0.000040                         -0.001139         0.000113
 validation population_oracle     finite_oracle right minus left            999          999                           0.000167          0.000013                          0.000049         0.000008
 validation    population_hac population_oracle right minus left            977          999                          -0.001304          0.000147                         -0.002933         0.000235
 validation        finite_hac     finite_oracle right minus left            991          999                          -0.000890          0.000142                         -0.001798         0.000166
```

## Observed descriptive fits

```text
          variant      crr       em  objective  boundary  unresolved  critical_lower  critical_upper
   population_hac 0.838258 0.254086  25.255410     False       False      127.973455      162.631702
       finite_hac 0.875753 0.196029  27.550883     False       False       54.637589       86.675796
population_oracle 0.804625 0.288166  31.828005     False       False       15.280468       15.280468
    finite_oracle 0.808546 0.286695  37.283335     False       False       15.480962       15.480962
```

Observed fits use revised historical data and the fixed published nuisance calibration. In particular, observed oracle-weighted criteria exceed their fixed-DGP calibration cutoffs. Improved null rejection behavior in model-generated data therefore does not establish adequacy for the historical sample. These are descriptive comparisons; no composite-null test, parameter interval or empirically validated feasible estimator is added. Criteria under different weights have different absolute scales.

## Scope

Raw simulated time series were not saved in the phase artifacts, so this audit authenticates seed/hash identity and shared moment/covariance references without independently resimulating each series. It verifies recorded start successes, objective agreement and selected-result consistency, but does not rederive stationarity diagnostics from unsaved gradients. Oracle covariance controls and fixed-DGP simulation cutoffs do not establish a feasible general-purpose estimator or composite-null inference.
