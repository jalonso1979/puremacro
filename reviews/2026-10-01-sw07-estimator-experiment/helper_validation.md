# Exact-expectation helper and estimator validation record

This record transcribes successful console tool output observed during this
session. The test and timing commands were not redirected to persistent log
files; this document is a record of that output, not a reconstructed raw log.
No tests were rerun solely to create this record.

## Tests observed

Standalone helper validation:

```bash
.venv/bin/python -m pytest tests/test_sw07_expectations.py -q
```

Observed result: **51 passed in 1.73 seconds**.

Helper plus independent experiment validation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest \
  tests/test_sw07_expectations.py tests/test_sw07_estimator_experiment.py -q
```

Observed result: **73 passed in 25.66 seconds**: 51 helper cases and 22 experiment
cases, including parameterized controls. The experiment smoke seed was **8931**,
separate from the recorded study seed. Calibration and validation each used two
draws for the integration fixture.

The helper cases compare both calibrations and all estimator-bound corners
against the previously audited full finite-sample oracle. Independent dense
VAR quadratic forms, cross-lag orientation, common-window changes, IID
demeaning bias, known-mean and zero-shock limits, and invalid/nonstationary
inputs are covered.

The experiment cases verify original population/HAC estimator equivalence,
finite-expectation criteria against the full exact oracle, constant oracle
weights while candidate moments vary, shared samples/HAC, independent phases,
interrupted and completed checkpoint reuse, seed/source mismatch rejection,
first-order stationarity failure handling, and failure bounds in both phases.
The 399-draw critical-rank calculation is exercised without simulating 399
draws in the tests.

The reporting inconsistency found during review—nonfinite calibration criteria
were bounded as unknown but omitted from the reported unresolved count—was
corrected before this successful combined run. A regression case covers it.

## Timing observed

The benchmark set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, and `MKL_NUM_THREADS=1`, then called each evaluator
directly at `_PUBLISHED_MODE` with its defaults: **156 observations and all 15
moments**. Each function received one warm-up call, followed by five
`timeit.repeat` batches of 50 calls. The reported statistic is the median batch
time divided by 50.

| Evaluator | Median milliseconds per call |
|---|---:|
| `sw07_covariance_moments` | 1.78012084 |
| `sw07_finite_sample_expectations` | 2.30790916 |
| `sw07_finite_sample_moments` | 34.12569832 |

The expectations-only evaluator was **14.7864 times faster** than the full
expectation-plus-covariance oracle, with approximately **0.528 ms** additional
time over population moments. These are observed local timings, not a portable
performance guarantee. The benchmark harness applies no interpolation or
memoization to these direct calls.

Per-batch milliseconds per call, preserved from the console output:

```text
population: [1.78012084, 1.79092500, 1.76439916, 1.77767334, 1.78733332]
expectations: [2.32209832, 2.30429332, 2.31015916, 2.30790916, 2.30026582]
full oracle: [33.53167916, 34.12569832, 33.26809000, 36.35189250, 38.40920832]
```

The executed timing code used this structure for each imported evaluator:

```python
function(_PUBLISHED_MODE)
timings = timeit.repeat(lambda: function(_PUBLISHED_MODE), number=50, repeat=5)
median_ms = float(np.median(timings) / 50 * 1000)
```

Scoped `git diff --check` returned exit code zero with no output for the helper
module and the two test modules. No existing numerical source files were
changed by the helper implementation or its independent review.
