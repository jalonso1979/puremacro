# Selected real-draw failure replay

**Result: all three selected cases reproduce their original classification and
objective. All 12 saved optimizer-start records reproduce exactly.** No study
output, failure, simulated draw or rate denominator was replaced.

The [script](failure_replay.py) authenticates all seven calibration artifacts,
all 11 frozen numerical-source files, and the recorded Python/NumPy/SciPy/Pandas
versions. It reconstructs complete 15-moment targets from the NPZ, using each
draw's full HAC covariance or the manifest's fixed true-DGP oracle covariance.
It then uses the original four starts, bounds and optimizer tolerances.
[Full replay diagnostics](failure_replay.json) include newly recomputed
first-order projections, tolerances, gradients and optimizer termination data.

## Selection and coverage

The predeclared replay selection was the earliest unresolved calibration draw
for every observed variant/cause pair, plus the earliest usable `finite_oracle`
control. Calibration contains seven unresolved `population_hac` draws and nine
unresolved `finite_hac` draws. All 16 fail the interior first-order stationarity
guard. There are no failed starts, start disagreements or unresolved oracle
draws in the 399-draw calibration phase, so those absent failure causes cannot
be selected for real-draw replay.

Replication indices below are zero based.

| Variant | Replication | Selected objective | Replay status | Selected projection / tolerance |
|---|---:|---:|---|---:|
| `population_hac` | 1 | 39.346234049271885 | unresolved, unchanged | 1.047722 |
| `finite_hac` | 62 | 20.367226403444775 | unresolved, unchanged | 1.029987 |
| `finite_oracle` | 0 | 13.801135665373558 | regular, unchanged | 0.024051 |

Every start reports SciPy success with `ftol` termination. Both unresolved
cases are interior and have local numerical rank two, but their selected
first-order projection is about 4.8% and 3.0% above the required tolerance,
respectively. Across their four starts the projection-to-tolerance ranges are
1.048–1.211 and 1.030–1.313. The usable oracle control has ratios 0.024–0.109.
Thus the unresolved cases are reproduced applications of the preserved
first-order guard, despite successful optimizer termination and closely
agreeing objectives.

The script also reconstructs each first-order projection independently from
a Cholesky whitening of the saved full fitted-moment covariance, the newly
recomputed derivative and the residual. These calculations agree with the
bridge's replay diagnostics. This checks the arithmetic of the guard; it is
not a separate analytic-derivative or global-optimality certificate.

## Exact evidence replay

The three selected objective differences are exactly zero. All 12 per-start
parameter vectors, objectives, success flags, evaluation counts, messages and
inference-reason lists match the original JSON records exactly. Selected
parameters differ from the CSV columns by at most `8.33e−17`, consistent with
decimal CSV parsing; the underlying JSON parameter records match exactly.

The unresolved sample for replication 1 was independently regenerated from
the saved seed `18399765841211511075`. Its raw float64 sample SHA-256 matches:

```text
2759eca8c35e34c9890d9989e8016dcac67da53a1d3310f7f3ec4826367d0eaf
```

All 15 sample moments and every entry of its 15×15 HAC covariance match the
authenticated NPZ **exactly**.

## Scope

This is a compact replay of selected actual calibration cases using the same
frozen model and optimizer. Original records contain first-order failure
reasons but not their numerical gradients or projections; those additional
diagnostics are recomputed here and are not claimed to have been stored in
the initial Monte Carlo output. The replay supports reproducibility of the
recorded handling. It neither converts failures to usable fits nor establishes
valid statistical inference or global optimization.

Reproduce from the repository root:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python reviews/2026-10-01-sw07-estimator-experiment/failure_replay.py
```
