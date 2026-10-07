# Independent research benchmarks

Overall: **PASS**. Generated 2026-10-01T19:34:10.416318+00:00.

These results establish only the claims and domains stated in each case.

Environment: puremacro=4.3.0, python=3.14.6, numpy=2.5.2, scipy=1.18.1, pandas=3.0.5, platform=macOS-27.2-arm64-arm-64bit-Mach-O.

## linear_minimum_distance: GLS fit and full parameter covariance

**PASS** — analytical.

Sources: https://users.ssc.wisc.edu/~bhansen/econometrics/

Data: `{"design": [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 4.0]], "estimator_covariance": [[0.04000000000000001, 0.016, 0.004, 0.002], [0.016, 0.0964, -0.0104, 0.015799999999999998], [0.004, -0.0104, 0.0645, 0.005699999999999999], [0.002, 0.015799999999999998, 0.005699999999999999, 0.1635]], "observations": 4, "origin": "deterministic synthetic four-moment linear design", "values": [0.3, 0.85, 1.6, 2.75]}`

Sampling: No draws; supplied covariance is Cov(moment estimator), already scaled for its sample.

Transformations: Identity; fit intercept and slope with a correlated four-moment target.

Limitations: Checks linear optimal weighting and covariance. No empirical model identification or finite-sample coverage claim.

Tolerance: rtol=1e-07, atol=1e-09. Perturbed-output negative control rejected: True.

| Metric | Unit | Observed | Reference | Max absolute error | Pass |
|---|---|---|---|---:|---|
| parameters | level | `[0.30269655 0.62154637]` | `[0.30269655 0.62154637]` | 1.11022e-16 | True |
| parameter_covariance | level squared | `[[ 0.03539986 -0.01220062]  [-0.01220062  0.01030579]]` | `[[ 0.03539986 -0.01220062]  [-0.01220062  0.01030579]]` | 1.38778e-17 | True |
| objective | dimensionless | `0.09803322` | `0.09803322` | 0 | True |

Arrays are abbreviated here; benchmark_report.json contains every compared value.

## household_expenditure: Stone-Geary welfare versus independent primal optimization

**PASS** — numerical_oracle.

Sources: https://ocw.mit.edu/courses/14-661-labor-economics-i-fall-2024/mit14_661_f24_problem_set_0.pdf

Data: `{"base_budget": 10.2, "base_prices": [1.04, 0.95, 1.02], "baseline_spending": [3.0, 5.0, 2.0], "beta": [0.24, 0.52, 0.24], "budget": 10.8, "gamma": [1.56, 1.88, 0.56], "origin": "synthetic three-good household", "prices": [1.2, 0.9, 1.15]}`

Sampling: No sample or survey weights. One household; two deterministic states.

Transformations: Direct utility maximization and four expenditure minimizations use raw log Stone-Geary utility; no puremacro welfare reference.

Limitations: Independent SciPy SLSQP oracle for interior preferences. Not an empirical household calibration.

Tolerance: rtol=2e-06, atol=2e-07. Perturbed-output negative control rejected: True.

| Metric | Unit | Observed | Reference | Max absolute error | Pass |
|---|---|---|---|---:|---|
| ev | currency per period | `[0.14474689]` | `[0.14474689]` | 2.11919e-12 | True |
| cv | currency per period | `[0.14990565]` | `[0.14990565]` | 3.17968e-13 | True |
| base_quantities | benchmark-priced composite units | `[2.99547692 5.28485053 2.02362353]` | `[2.9954777  5.28485001 2.02362322]` | 7.7798e-07 | True |
| new_quantities | benchmark-priced composite units | `[2.8784     5.68871111 1.93572174]` | `[2.87840023 5.68871084 1.93572171]` | 2.71478e-07 | True |

Arrays are abbreviated here; benchmark_report.json contains every compared value.

## distributional_incidence: Household incidence, expansion weights and uniform rebates

**PASS** — analytical.

Sources: https://ocw.mit.edu/courses/14-661-labor-economics-i-fall-2024/mit14_661_f24_problem_set_0.pdf

Data: `{"expansion_counts": [3.0, 1.0], "group_spending": [[60.0, 40.0], [60.0, 140.0]], "origin": "synthetic grouped survey", "price_ratios": [1.2, 0.9], "transfer_total": 40.0}`

Sampling: Two representative groups expand to three low-budget and one high-budget household; no sampling inference.

Transformations: Cobb-Douglas exact cost indices, equal transfer per expanded household, and weighted EV totals.

Limitations: Conditional static incidence with fixed preferences and exogenous prices; no empirical Mexico or general-equilibrium claim.

Tolerance: rtol=1e-07, atol=1e-09. Perturbed-output negative control rejected: True.

| Metric | Unit | Observed | Reference | Max absolute error | Pass |
|---|---|---|---|---:|---|
| ev | currency per household-year | `[ 2.84588833 14.04010942]` | `[ 2.84588833 14.04010942]` | 2.84217e-14 | True |
| cv | currency per household-year | `[ 3.04385252 13.77509564]` | `[ 3.04385252 13.77509564]` | 2.84217e-14 | True |
| total_ev | currency per year | `22.57777441` | `22.57777441` | 7.10543e-15 | True |
| budget | currency per household-year | `[110. 210.]` | `[110. 210.]` | 0 | True |

Arrays are abbreviated here; benchmark_report.json contains every compared value.

## growth_analytical: Full-depreciation log-utility growth solution

**PASS** — analytical.

Sources: https://python.quantecon.org/cass_koopmans_1.html

Data: `{"alpha": 0.36, "beta": 0.96, "depreciation": 1.0, "model_source": "var c k a; varexo e;\nparameters alpha beta rho; alpha=.36; beta=.96; rho=.8;\nmodel;\n1/c=beta/c(+1)*alpha*exp(a(+1))*k^(alpha-1);\nk=exp(a)*k(-1)^alpha-c;\na=rho*a(-1)+e;\nend;\ninitval; c=.35; k=.19; a=0; end;\nshocks; var e; stderr .01; end;\n", "origin": "analytical growth economy", "rho": 0.8}`

Sampling: No sample. Deterministic steady state and first-order derivatives at zero technology.

Transformations: k' = alpha*beta*exp(a)*k**alpha; c = (1-alpha*beta)*exp(a)*k**alpha. Variable order c,k,a.

Limitations: Checks this special log-utility, Cobb-Douglas, full-depreciation case and its local derivatives only.

Tolerance: rtol=1e-07, atol=1e-09. Perturbed-output negative control rejected: True.

| Metric | Unit | Observed | Reference | Max absolute error | Pass |
|---|---|---|---|---:|---|
| steady_state | c,k: goods per capita; a: log technology | `[0.35999048 0.19011722 0.        ]` | `[0.35999048 0.19011722 0.        ]` | 5.55112e-17 | True |
| lagged_capital_derivative | output units per unit of lagged capital | `[0.68166667 0.36       0.        ]` | `[0.68166667 0.36       0.        ]` | 5.55112e-17 | True |
| technology_derivative | output units per unit log-technology innovation | `[0.35999048 0.19011722 1.        ]` | `[0.35999048 0.19011722 1.        ]` | 5.55112e-17 | True |

Arrays are abbreviated here; benchmark_report.json contains every compared value.

## dynare_rbc_order2: RBC decision rules and common-innovation moments versus Dynare 7

**PASS** — external_software.

Sources: https://www.dynare.org/manual/the-model-file.html#stochastic-solution-and-simulation; Package: puremacro.dsge/_references/DYNARE_FIXTURES.md

Data: `{"origin": "Live Dynare 7.0 in MATLAB R2026a, 2026-09-20", "provenance_manifest": "puremacro.dsge/_references/dynare_live/manifest.json", "reference": "puremacro.dsge/_references/dynare_live/rbc_order2.npz", "reference_sha256": "eb7ddf55b223fae31b2664e16ec7970848c816020544b46822eca05e52926d72", "run_id": "20260920_161322_166"}`

Sampling: 2,500 common Gaussian innovations from NumPy default_rng(20260920); discard first 500. Covariance uses ddof=1; lag-1 covariance uses separately centered endpoints and N-2.

Transformations: Match variable/state/shock labels; unfold derivative tensors without factorial rescaling; add deterministic steady state to simulated deviations. Validate model, innovations and reference checksums.

Limitations: External-software agreement for one small model at order two, conditional on supplied innovations. No live Dynare run, population-moment accuracy or published empirical replication is claimed.

Tolerance: rtol=1e-09, atol=1e-10. Perturbed-output negative control rejected: True.

| Metric | Unit | Observed | Reference | Max absolute error | Pass |
|---|---|---|---|---:|---|
| ys | model levels (y,c,k,a) | `[ 3.01532771  2.30661723 28.34841906  0.        ]` | `[ 3.01532771  2.30661723 28.34841906  0.        ]` | 0 | True |
| ghx | level decision-rule derivative in named state/shock coordinates | `[[3.51010101e-02 2.71379494e+00]  [4.80395296e-02 4.70773909e-01]  [9.62061480e-01 2.24302103e+00]  [1.86464105e-16 9.00000000e-01]]` | `[[0.03510101 2.71379494]  [0.04803953 0.47077391]  [0.96206148 2.24302103]  [0.         0.9       ]]` | 3.10862e-15 | True |
| ghu | level decision-rule derivative in named state/shock coordinates | `[[3.01532771]  [0.52308212]  [2.49224559]  [1.        ]]` | `[[3.01532771]  [0.52308212]  [2.49224559]  [1.        ]]` | 1.22125e-15 | True |
| ghxx | level decision-rule derivative in named state/shock coordinates | `[[-8.29593944e-04  3.15909091e-02  3.15909091e-02  2.44241544e+00]  [-5.53194861e-04  3.95172819e-03  3.95172819e-03  2.67140194e-01]  [-2.76399082e-04  2.76391809e-02  2.76391809e-02  2.17527525e+00]  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]]` | `[[-8.29593944e-04  3.15909091e-02  3.15909091e-02  2.44241544e+00]  [-5.53194861e-04  3.95172819e-03  3.95172819e-03  2.67140194e-01]  [-2.76399082e-04  2.76391809e-02  2.76391809e-02  2.17527525e+00]  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]]` | 1.37668e-14 | True |
| ghxu | level decision-rule derivative in named state/shock coordinates | `[[ 0.03510101  2.71379494]  [ 0.00439081  0.29682244]  [ 0.0307102   2.4169725 ]  [-0.         -0.        ]]` | `[[ 0.03510101  2.71379494]  [ 0.00439081  0.29682244]  [ 0.0307102   2.4169725 ]  [-0.         -0.        ]]` | 1.37668e-14 | True |
| ghuu | level decision-rule derivative in named state/shock coordinates | `[[ 3.01532771]  [ 0.32980271]  [ 2.685525  ]  [-0.        ]]` | `[[ 3.01532771]  [ 0.32980271]  [ 2.685525  ]  [-0.        ]]` | 1.15463e-14 | True |
| ghs2 | level decision-rule derivative in named state/shock coordinates | `[[ 0.00000000e+00]  [ 9.76460437e-05]  [-9.76460437e-05]  [ 0.00000000e+00]]` | `[[ 0.00000000e+00]  [ 9.76460437e-05]  [-9.76460437e-05]  [ 0.00000000e+00]]` | 2.44894e-17 | True |
| sample_mean | model levels | `[ 3.00344255e+00  2.29881754e+00  2.82142087e+01 -2.66228593e-03]` | `[ 3.00344255e+00  2.29881754e+00  2.82142087e+01 -2.66228593e-03]` | 1.06581e-14 | True |
| sample_covariance | products of model level units | `[[7.02116196e-03 3.14902289e-03 4.86886839e-02 1.80027254e-03]  [3.14902289e-03 1.90207117e-03 3.27220876e-02 6.70757179e-04]  [4.86886839e-02 3.27220876e-02 5.79612436e-01 9.44392294e-03]  [1.80027254e-03 6.70757179e-04 9.44392294e-03 5.00071202e-04]]` | `[[7.02116196e-03 3.14902289e-03 4.86886839e-02 1.80027254e-03]  [3.14902289e-03 1.90207117e-03 3.27220876e-02 6.70757179e-04]  [4.86886839e-02 3.27220876e-02 5.79612436e-01 9.44392294e-03]  [1.80027254e-03 6.70757179e-04 9.44392294e-03 5.00071202e-04]]` | 7.86038e-14 | True |
| sample_lag1_covariance | products of model level units | `[[6.56299517e-03 2.96425177e-03 4.59753873e-02 1.67698712e-03]  [3.18260682e-03 1.88738859e-03 3.22945561e-02 6.87670181e-04]  [5.08230829e-02 3.29561672e-02 5.78343972e-01 1.01931206e-02]  [1.61610572e-03 6.04338752e-04 8.52787862e-03 4.48280509e-04]]` | `[[6.56299517e-03 2.96425177e-03 4.59753873e-02 1.67698712e-03]  [3.18260682e-03 1.88738859e-03 3.22945561e-02 6.87670181e-04]  [5.08230829e-02 3.29561672e-02 5.78343972e-01 1.01931206e-02]  [1.61610572e-03 6.04338752e-04 8.52787862e-03 4.48280509e-04]]` | 7.80487e-14 | True |

Arrays are abbreviated here; benchmark_report.json contains every compared value.

## enigh2024_official_totals: Observed ENIGH decile baskets reproduce official national totals

**PASS** — official_data.

Sources: https://www.inegi.org.mx/contenidos/programas/enigh/nc/2024/tabulados/enigh2024_ns_basicos_tabulados.xlsx

Data: `{"accessed": "2026-10-01", "derived_csv": "puremacro.datasets/data/enigh2024_deciles.csv", "origin": "INEGI ENIGH 2024 national household income deciles; Cuadros 3.2 and 4.2", "reference_origin": "literal national published totals, not the derived CSV", "source_sha256": "7af7850495255fb1e6a9cb139cf531e5a62c7fd2a9a1de5ba0503f03aa82490b"}`

Sampling: Ten national household income deciles; expansion counts include all households. No sampling covariance is available.

Transformations: Published thousands of MXN multiplied by 1000; decile totals divided by all-household counts. Eight consumption categories plus outward transfers equal monetary expenditure. CSV decimal rounding motivates the tolerance.

Limitations: Validates the extraction, weights, units and selected accounting totals of actual survey aggregates. Does not replicate an empirical causal model, identify tariff responses or validate within-decile heterogeneity.

Tolerance: rtol=1e-11, atol=1e-05. Perturbed-output negative control rejected: True.

| Metric | Unit | Observed | Reference | Max absolute error | Pass |
|---|---|---|---|---:|---|
| households | expanded households | `38830230.` | `38830230.` | 0 | True |
| national_food | MXN per quarter | `6.98246719e+11` | `6.98246719e+11` | 0.482422 | True |
| national_monetary_expenditure | MXN per quarter | `1.85120662e+12` | `1.85120662e+12` | 1.15649 | True |
| expenditure_adding_up | MXN per household-quarter | `[-7.99627742e-09 -4.89962986e-08 -5.20049070e-08 ... -1.09990651e-07  -2.99914973e-08  2.80022505e-07]` | `[0. 0. 0. ... 0. 0. 0.]` | 2.80023e-07 | True |

Arrays are abbreviated here; benchmark_report.json contains every compared value.
