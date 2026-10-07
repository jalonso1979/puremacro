# Independent research benchmarks

Overall: **FAIL**. Generated 2026-10-01T20:52:49.431308+00:00.

These results establish only the claims and domains stated in each case.

Environment: puremacro=4.3.0, python=3.14.6, numpy=2.5.2, scipy=1.18.1, pandas=3.0.5, platform=macOS-27.2-arm64-arm-64bit-Mach-O.

## rr2010_baseline_software: Romer–Romer baseline versus independent software

**PASS** — external_software.

Sources: https://www.aeaweb.org/articles?id=10.1257/aer.100.3.763; https://eml.berkeley.edu/~dromer/papers/DataSet.zip; https://emlab.berkeley.edu/~dromer/papers/RomerandRomerAERJune2010.pdf

Data: `{"archive_sha256": "c6c3b4888f3de596abf56bc136e2175ec5a4aa0dab137e79e261984a0a84701c", "author_specification": "EXOGERNR.RAT", "csv_sha256": "059d7514fb14e5b8a8f2516e23c49b618032236c0b4cf4599082f3637e12fdad", "metadata_sha256": "06cf9cffee4213684b0abd9d05a158e00e880cc6260cfab632cf6a2793ee13d7", "origin": "Romer and Romer (2010) original author archive", "reference_loader": "puremacro.replication.load_rr2010_reference", "reference_sha256": "3c915d119158db1f7002efa80fa12812e2c4edaf225569960561fef29b915b8b", "reference_software": "statsmodels 0.14.6, Python 3.12.13; QR OLS"}`

Sampling: 1950Q1–2007Q4, 232 quarterly observations; 14 coefficients, 218 residual degrees of freedom.

Transformations: OLS of 100*delta log(real GDP) on intercept and lags 0–12 of 100*(DEFIC+LONGR)/NOMGDP; conventional residual-df OLS covariance. Cumulate tax coefficients and full covariance to obtain GDP-level responses.

Limitations: Frozen independent software port of the authors' baseline specification; not a live RATS run. Numerical agreement does not establish narrative shock exogeneity or reproduce robustness specifications.

Tolerance: rtol=1e-10, atol=1e-11. Perturbed-output negative control rejected: True.

| Metric | Unit | Observed | Reference | Max absolute error | Pass |
|---|---|---|---|---:|---|
| coefficients | GDP growth percentage points; intercept or per one-percent-of-GDP tax increase | `[ 0.79135443 -0.09341383 -0.2941718  ... -0.09250983  0.36308229   0.20003629]` | `[ 0.79135443 -0.09341383 -0.2941718  ... -0.09250983  0.36308229   0.20003629]` | 8.88178e-16 | True |
| coefficient_covariance | products of coefficient units | `[[ 0.00439678  0.00187209  0.00184419 ...  0.00184924  0.00192549    0.00216789]  [ 0.00187209  0.0844644   0.00131056 ...  0.00059979  0.00389212    0.00204827]  [ 0.00184419  0.00131056  0.08437749 ... -0.00024023  0.00058273    0.00407398]  ...  [ 0.00184924  0.00059979 -0.00024023 ...  0.0637433   0.00098504    0.00019461]  [ 0.00192549  0.00389212  0.00058273 ...  0.00098504  0.06381815    0.00120084]  [ 0.00216789  0.00204827  0.00407398 ...  0.00019461  0.00120084    0.06547253]]` | `[[ 0.00439678  0.00187209  0.00184419 ...  0.00184924  0.00192549    0.00216789]  [ 0.00187209  0.0844644   0.00131056 ...  0.00059979  0.00389212    0.00204827]  [ 0.00184419  0.00131056  0.08437749 ... -0.00024023  0.00058273    0.00407398]  ...  [ 0.00184924  0.00059979 -0.00024023 ...  0.0637433   0.00098504    0.00019461]  [ 0.00192549  0.00389212  0.00058273 ...  0.00098504  0.06381815    0.00120084]  [ 0.00216789  0.00204827  0.00407398 ...  0.00019461  0.00120084    0.06547253]]` | 6.93889e-17 | True |
| irf | GDP log-percent per one-percent-of-GDP tax increase | `[-0.09341383 -0.38758563 -0.23167967 ... -3.08081276 -2.71773047  -2.51769418]` | `[-0.09341383 -0.38758563 -0.23167967 ... -3.08081276 -2.71773047  -2.51769418]` | 1.33227e-15 | True |
| irf_covariance | squared GDP log-percent response units | `[[0.0844644  0.08577496 0.08533259 ... 0.08065899 0.08455111 0.08659939]  [0.08577496 0.17146302 0.17223356 ... 0.16264678 0.16712163 0.17324388]  [0.08533259 0.17223356 0.25712259 ... 0.24368579 0.24793044 0.25462925]  ...  [0.08065899 0.16264678 0.24368579 ... 0.76385924 0.76973031 0.77703188]  [0.08455111 0.16712163 0.24793044 ... 0.76973031 0.83941954 0.84792195]  [0.08659939 0.17324388 0.25462925 ... 0.77703188 0.84792195 0.92189688]]` | `[[0.0844644  0.08577496 0.08533259 ... 0.08065899 0.08455111 0.08659939]  [0.08577496 0.17146302 0.17223356 ... 0.16264678 0.16712163 0.17324388]  [0.08533259 0.17223356 0.25712259 ... 0.24368579 0.24793044 0.25462925]  ...  [0.08065899 0.16264678 0.24368579 ... 0.76385924 0.76973031 0.77703188]  [0.08455111 0.16712163 0.24793044 ... 0.76973031 0.83941954 0.84792195]  [0.08659939 0.17324388 0.25462925 ... 0.77703188 0.84792195 0.92189688]]` | 6.66134e-16 | True |
| standard_errors | GDP log-percent response units | `[0.29062759 0.41408093 0.50707257 ... 0.87399041 0.91619842 0.96015461]` | `[0.29062759 0.41408093 0.50707257 ... 0.87399041 0.91619842 0.96015461]` | 4.44089e-16 | True |
| t_statistics | dimensionless | `[-0.32142105 -0.93601418 -0.45689648 ... -3.52499608 -2.96631213  -2.62217579]` | `[-0.32142105 -0.93601418 -0.45689648 ... -3.52499608 -2.96631213  -2.62217579]` | 1.77636e-15 | True |
| nobs | quarters | `232.` | `232.` | 0 | True |
| df_resid | degrees of freedom | `218.` | `218.` | 0 | True |

Arrays are abbreviated here; benchmark_report.json contains every compared value.

## rr2010_published_peak: Romer–Romer published Figure 4 baseline trough

**FAIL** — published_empirical.

Sources: https://www.aeaweb.org/articles?id=10.1257/aer.100.3.763; https://eml.berkeley.edu/~dromer/papers/DataSet.zip; https://emlab.berkeley.edu/~dromer/papers/RomerandRomerAERJune2010.pdf

Data: `{"archive_sha256": "c6c3b4888f3de596abf56bc136e2175ec5a4aa0dab137e79e261984a0a84701c", "author_specification": "EXOGERNR.RAT", "csv_sha256": "059d7514fb14e5b8a8f2516e23c49b618032236c0b4cf4599082f3637e12fdad", "metadata_sha256": "06cf9cffee4213684b0abd9d05a158e00e880cc6260cfab632cf6a2793ee13d7", "origin": "Romer and Romer (2010) original author archive", "printed_page": 781, "reference_loader": "puremacro.replication.load_rr2010_reference", "reference_origin": "literal published values -3.08, -3.53 and horizon 10", "reference_sha256": "3c915d119158db1f7002efa80fa12812e2c4edaf225569960561fef29b915b8b", "reference_software": "statsmodels 0.14.6, Python 3.12.13; QR OLS"}`

Sampling: 1950Q1–2007Q4, 232 quarterly observations; 14 coefficients, 218 residual degrees of freedom.

Transformations: OLS of 100*delta log(real GDP) on intercept and lags 0–12 of 100*(DEFIC+LONGR)/NOMGDP; conventional residual-df OLS covariance. Cumulate tax coefficients and full covariance to obtain GDP-level responses.

Limitations: One original-data baseline result. Absolute tolerance 0.005 reflects two-decimal publication rounding; integer horizon must match exactly. The original-data t statistic misses its published rounding interval by about 0.000003925; this remains a failure, also present in the authors' RATS databanks. It is not a software-parity tolerance or independent causal validation.

Tolerance: rtol=0.0, atol=0.005. Perturbed-output negative control rejected: True.

| Metric | Unit | Observed | Reference | Max absolute error | Pass |
|---|---|---|---|---:|---|
| response_h10 | GDP log-percent per one-percent-of-GDP tax increase | `-3.08081276` | `-3.08` | 0.000812763 | True |
| t_h10 | dimensionless | `-3.52499608` | `-3.53` | 0.00500392 | False |
| trough_horizon | quarters after the tax increase | `10.` | `10.` | 0 | True |

Arrays are abbreviated here; benchmark_report.json contains every compared value.
