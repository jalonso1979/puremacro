# Independent finite-sample evidence review

No blocking implementation or reporting issue was found in the reviewed protocol,
sampling/aggregation code and completed experiment. This review confirms the
aggregation of the recorded experiment; it is not an independent SW07 solver
validation or an extension of the experiment to new calibrations.

Reproduce with:

```bash
.venv/bin/python reviews/2026-10-01-sw07-finite-sample/independent_review.py
```

The [script](independent_review.py) imports no production module, generates no
simulations and changes no model inputs. It authenticates the four manifests'
artifact hashes and independently recomputes counts, criterion quantiles,
parameter bias/RMSE, tail/rejection fractions and conservative Clopper–Pearson
intervals. From the saved arrays it recomputes all 15 moment means, the complete
15×15 empirical covariance and mean HAC8 covariance. All comparisons pass.
The [numerical record](independent_review.json) includes input and script hashes,
dependency versions, all eigenvalues and all 60 moment-mean comparisons. The
exact Gaussian reference is taken from the saved audit; separate mapping and
quadratic-form tests establish its correctness.

## Interpretation of the four scenarios

| Fixed simulated DGP | Unresolved / 399 | Observed-criterion tail bounds | Conservative 95% MC interval |
|---|---:|---:|---:|
| Baseline calibration | 4 | 0–1.003% | 0–2.547% |
| Published calibration | 9 | 41.604–43.860% | 36.720–48.884% |
| Baseline with fitted crr/em | 2 | 0–0.501% | 0–1.799% |
| Published with fitted crr/em | 7 | 39.348–41.103% | 34.524–46.107% |

The two fitted cases are plug-in diagnostics, not validated bootstrap or
composite-null p-values. Unresolved draws count as both no exceedances and all
exceedances in these bounds; they have not disappeared from denominators.
Zero usable exceedances do not establish zero probability or p<0.001.

Under the published fixed DGP, the nominal 5% chi-square(7) rule rejects
248/399 to 257/399 draws (**62.155–64.411%**, depending on unresolved outcomes).
Within the numerically regular subset, it rejects **241/383 = 62.924%**.
Boundary cases therefore cannot explain this poor finite-sample calibration.
The observed criterion **25.255** is ordinary relative to this particular
simulated distribution (usable-draw median **20.680**, 95th percentile
**115.086**). This comparison does not validate the economic model or restore
ordinary parameter confidence intervals.

## Covariance distortion is directional

At the published fixed DGP, mean HAC8/exact variance ratios are **0.556** for the
inflation-variance estimator and **0.525** for the interest-rate-variance
estimator. At lag four they are **0.484** and **0.470**, respectively. Bandwidth
12 improves these particular marginal ratios but leaves substantial differences.

The following table examines the complete covariance of the nine fitted moments.
Its entries are the minimum/median/maximum generalized eigenvalues of the mean
estimated covariance relative to the exact demeaned covariance. A value below
or above one identifies a linear combination whose variance is underestimated
or overestimated. These are not eigenvalues of the expected inverse weight.

| DGP | Empirical covariance | Mean HAC4 | Mean HAC8 | Mean HAC12 |
|---|---:|---:|---:|---:|
| Baseline fixed | 0.739 / 0.952 / 1.137 | 0.342 / 1.082 / 21.302 | 0.475 / 0.996 / 14.092 | 0.544 / 0.940 / 10.127 |
| Published fixed | 0.742 / 0.955 / 1.332 | 0.311 / 0.890 / 5.531 | 0.408 / 0.925 / 3.938 | 0.455 / 0.923 / 3.045 |
| Baseline fitted | 0.795 / 0.996 / 1.398 | 0.295 / 1.056 / 23.242 | 0.405 / 1.042 / 14.542 | 0.470 / 0.992 / 10.433 |
| Published fitted | 0.717 / 0.938 / 1.211 | 0.336 / 0.837 / 5.426 | 0.441 / 0.858 / 3.792 | 0.491 / 0.844 / 2.918 |

Thus HAC does not universally understate uncertainty. Demeaning bias, estimated
weighting, nonnormal quadratic forms and fitting remain entangled; this
experiment does not attribute a share of the rejection distortion to each.

## Mean correction and simulation agreement

Finite-sample demeaning cannot rescue the baseline: expected GDP-growth variance
changes **48.2591→48.2391**, versus observed **0.71689**. Under the published
calibration, expected inflation variance changes **0.29013→0.24345** and
interest-rate variance **0.36438→0.30016**, moving farther below observed
**0.37040/0.69875**. Those mean corrections are about **0.674/0.697 exact sampling
SDs**. The observed rate variance is **4.324 sampling SDs** above its
published-fixed expectation; this is a marginal descriptive discrepancy, not a
normal-theory p-value.

All 60 Monte Carlo moment means are within **2.123 MC standard errors** of their
exact expectations. Here each MC standard error is the exact sampling SD divided
by sqrt(399). The per-scenario maximum absolute standardized differences are
**1.701, 2.123, 1.549 and 0.736**, in the table's scenario order. This is a useful
descriptive simulation check, not a simultaneous normal test or a proof of
distributional equality.

The finite-sample mean correction was diagnosed rather than substituted into
the fitting target. Published parameters overlap the historical sample; the
dataset is a revised vintage; reserved moments share the fitted sample; and
fixed-calibration uncertainty, regime changes and departures from the stationary
Gaussian DGP remain outside this experiment.
