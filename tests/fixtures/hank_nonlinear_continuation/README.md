# Nonlinear conditional HANK evidence

`record.json` is the final source-bound run. Its `completed` flag means that the
experiment finished; each case's `converged` flag separately reports numerical
and adaptive acceptance. Read both. The runner, source snapshots, artifact
hashes and exact seeds are retained. Source hashes refer to `sources/*.txt`
snapshots as well as the executable files in the checkout.

The training sample is seed 1201, 160 innovations with standard deviation
`1e-7`. The held-out sample is seed 9917, 48 innovations with standard deviation
`1e-4`. No local-policy coefficient is re-estimated on this held-out sequence.
The original fixed-policy audit threshold remains `1e-5`; nonlinear finite-tree
equations use `1e-8`, and both horizon and branching sensitivity use `1e-6`.

`full` uses all admissible distribution directions. `snapshots16` uses the
original 16-direction snapshot construction. Both retain and transport the
same full 80-by-3 household grid at every observed date. Corrections condition
on the actual distribution and previous prices and integrate model-implied
future branches without reading subsequent realized innovations.

CSV paths are level deviations; price CSVs are levels. NPZ files contain all
actual entering/post-action distributions, state vectors, expected household
marginals and root saving, consumption and hours policies. Arrays load with
`allow_pickle=False`. They support independent checks of resource accounting,
asset clearing, probability mass and observed-state transitions. The market
maximum in the record includes all conditional future nodes, so it can exceed
the maximum calculated from saved observed roots alone.

The focused comparisons hold relevant numerical dimensions fixed: full versus
compressed terminal states, horizon 8 versus 32, Gauss-Hermite orders 2/3/5,
and a dense versus matrix-free aggregate solve. Household grid sensitivity is
reported on separately recalibrated 40-by-2 and 80-by-3 grids; it is not an exact
grid-error bound. Coarse Euler/KKT errors remain separate from market residuals.

## Preserved unsuccessful refinements

* `prior_depth3/`: horizon limit 32 and branching-depth limit 3. Both full and
  compressed rollouts stop before action 28 because branching sensitivity
  exceeds `1e-6`, despite successfully solved finite market equations.
* `prior_horizon32/`: branching-depth limit 4. The full rollout completes;
  the compressed rollout stops before action 28 because the horizon 16-to-32
  expected-marginal change is `1.3321272138278045e-6`. Its branching criterion
  passes. The final design permits a 64-period tail without changing tolerances.

Each earlier directory retains its own original records, arrays and source
snapshots. Earlier versions' simulation-level status did not distinguish which
refinement failed; the authoritative per-failure diagnostics do, and the final
implementation preserves separate simulation statuses as well.

The final run records verified single-thread Accelerate control where available.
Earlier exploratory runs requested environment thread limits only. Wall times
are host observations, with other verification work sharing the machine; they
are not portable performance guarantees. Neither solved finite equations nor
empirical refinement acceptance certify an infinite-horizon/global stochastic
solution, quadrature-order convergence or household-grid accuracy.

## Final retained results

| Basis | Fixed-policy market error | Corrected market error | Accepted dates | Correction seconds |
|---|---:|---:|---:|---:|
| Full | 1.6507474900784302e-5 | 7.061013941296948e-9 | 48 | 43.1065 |
| Snapshots, rank 16 | 2.7380463044401893e-5 | 8.279097097406538e-9 | 48 | 978.7250 |

The full-basis run uses `(horizon, depth)` counts `(8,2):32`, `(8,3):9`,
`(16,4):7`. Rank 16 uses `(8,2):25`, `(8,3):2`, `(16,3):1`, `(64,2):7`,
`(64,3):6`, `(64,4):7`. Thus 20 rank-16 dates require the long tail.
Euler/KKT maxima remain approximately `9.95e-4` in both runs. The dense root
log-control gap is `1.4643e-11`; the largest reported quadrature-order metric is
`2.6311e-8`. The separate rank-32 fixture evaluates compression refinement
without changing this evidence or its acceptance thresholds.
