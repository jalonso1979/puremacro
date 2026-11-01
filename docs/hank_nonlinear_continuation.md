# Nonlinear stochastic HANK continuation

The local [recursive HANK policy](hank_recursive.md) provides a solved stochastic
terminal boundary. A nonlinear continuation correction solves household policies,
distribution transport and markets jointly on a conditional future tree, starting
from the **actual observed distribution and inherited prices**. Repeating this
calculation at each observation gives a nonanticipative nonlinear policy.

This is a finite conditional approximation with a local terminal policy. It is
not a global or second-order infinite-horizon stochastic solution. Passing market
equations does not certify household grids, distribution compression or omitted
future uncertainty.

## Use

```python
import numpy as np
from puremacro.models import calibrate_one_asset_hank, solve_hank_stochastic_policy
from puremacro.models.hank_nonlinear_continuation import (
    correct_hank_stochastic_state, simulate_hank_nonlinear,
    solve_hank_nonlinear_continuation,
)

ss = calibrate_one_asset_hank(n_a=40, n_e=2, a_max=120.)
local = solve_hank_stochastic_policy(ss, distribution_rank=12, shock_std=1e-4)

# One conditional equilibrium; its status covers finite equations only.
finite = solve_hank_nonlinear_continuation(
    local, distribution=ss.distribution,
    aggregate_state=np.array([0., -1e-4]), lag_q=1.,
    horizon=8, branch_depth=2,
)

# Require separate horizon and branching refinements at this observed state.
corrected = correct_hank_stochastic_state(
    local, ss.distribution, np.array([0., -1e-4]), lag_q=1.,
)

# Only the current innovation is revealed before each correction is solved.
history = simulate_hank_nonlinear(local, np.array([-1e-4, 4e-5, 1e-5]))
assert history.converged
```

The one-asset and consol aggregate states are the inherited nominal-rate
**deviation** and current monetary shock. Productive-capital kernels add inherited
log capital. Their `aggregate_state_names`, `prices_from_state` and
`next_aggregate_state` hooks determine the ordering and transition. Both full and
reduced local policies are accepted; household sweeps always use the full grid.

For `simulate_hank_nonlinear`, `initial_aggregate_state` is the conditional state
*before* the first supplied innovation. The next nominal rate, physical capital,
full transported distribution and actual quoted `q` come from the preceding
nonlinear solve. In particular, `q` is never re-inferred from an off-equilibrium
mean of illiquid wealth. `initial_q` and each correction's `lag_q` are price
levels, with default one.

## What is solved

The current date is the root. `horizon` counts dates including the root.
Gauss-Hermite innovations are integrated at each of the next `branch_depth`
dates. Each later finite-tree innovation is set to its conditional mean; the
terminal boundary again uses the solved local policy's stochastic expectation.
At nonterminal nodes the expected marginal value is the probability-weighted
mean of the **optimized nonlinear child policies**, rather than the original
fixed linear feedback. Prices, household choices and all node distributions
therefore respond jointly to the conditional shock branches.

Unknowns are aggregate log controls at each node and projected post-terminal
distribution coefficients. Backward household sweeps and forward full-grid
transport produce market equations and terminal consistency equations. Damped
Newton–Krylov products differentiate this entire nonlinear residual. Existing
one-asset and consol price preconditioners are reused. Other kernels use the
unpreconditioned operator unless extended separately. `method="hybr"` provides a
memory-bounded dense finite-difference comparison.

No coefficient is fitted to the supplied realized history. Later realized
innovations never enter an earlier conditional tree. The same current state and
settings yield the same current action even when later realizations change.

## Accuracy and failure

The default adaptive limits are `horizons=(4, 8, 16, 32, 64)` and
`branch_depths=(1, 2, 3, 4)`. A deeper tree uses only horizons longer than its
branch depth. At a candidate, the solver compares a shorter horizon with the
same branch depth and a shallower branching tree with the same horizon.

Each comparison reports three changes: maximum root aggregate log-control
change, maximum relative expected-marginal change over the household grid, and
the L1 change in the next full distribution. Their maximum must meet its
respective tolerance. Defaults are `1e-6` for each refinement criterion and
`1e-8` for the finite equations. **These are empirical sensitivity criteria, not
rigorous error bounds.** Gauss-Hermite order stays fixed during adaptation;
`quadrature_order_converged=None` makes this untested dimension explicit. Compare
orders with `solve_hank_nonlinear_continuation` separately.

| Dimension | Reported evidence | Meaning |
|---|---|---|
| Aggregate equations | `market_residual_max_abs`, budget, mass and grid-exit checks | Nonlinear equations on the chosen tree |
| Household grid | `euler_kkt_max`, `euler_kkt_mean_max`, `upper_grid_mass` | Interpolation and grid diagnostics; no grid-accuracy certificate |
| Distribution compression | Initial/terminal projection L1, minimum reconstructed terminal mass | Error in the local terminal representation; full actual transport is retained |
| Finite terminal horizon | `horizon_difference`, `horizon_converged` | Root sensitivity at fixed branching depth |
| Future shock branches | `branching_difference`, `branching_converged` | Root sensitivity at fixed horizon and quadrature order |
| Quadrature order | Separate same-tree order comparisons | Not certified by branch-depth acceptance |

With reduced household-control policies, the terminal marginal-value basis is
another approximation. Distribution L1 does not measure its omitted directions;
inspect that policy's omitted full marginal equations and reachable-shock RMS
(`full_marginal_stationary_rms_max`) separately. Full-grid interior household
sweeps do not certify the terminal control basis.

Actual distributions must remain nonnegative and retain unit mass. A projected
terminal distribution can have negative components: this is reported separately
and is never substituted for the actual distribution or silently clipped. The
terminal expected marginal values must remain positive and finite. Coarse-grid
Euler errors can remain near `1e-3` even when aggregate residuals are below
`1e-8`; `converged=True` does not erase that discrepancy.

If a refinement limit is exhausted, the default raises
`HANKContinuationAccuracyError`, whose `.result` retains the failed correction.
With `raise_on_failure=False`, a correction returns `converged=False` and its
individual statuses. A simulation stops **before transporting the failed
action** and preserves the failed period and diagnostics. Equation, horizon and
branching statuses remain separate, including a passed horizon on a
branching-only failure. `max_nodes`, `max_workspace_bytes` and iteration limits
bound tree size, estimated working arrays and iterations. The per-tree memory
estimate is not a process-RSS guarantee and excludes retained results from
other refinements or previous dates. Increasing branch depth or quadrature
order grows trees rapidly.

## Independent inspection

`HANKNonlinearCorrection.tree` contains node probabilities, paths, entering full
distributions and optimized household policies. `expected_marginals`,
`marginal_values`, `post_distributions`, `aggregate_states`,
`aggregate_coordinates` and `prices` allow independent expectation and budget
checks. Price and household arrays are **levels**; `tree.paths` contains level
deviations, and `aggregate_coordinates` contains log deviations.

The simulation retains root household policies, expectations, actual prices and
all transported distributions. `keep_trees=True` additionally retains each
conditional result, at a potentially substantial memory cost.

The frozen evidence and failed shorter refinements are in
`tests/fixtures/hank_nonlinear_continuation/`. Reproduce in a **new** directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/puremacro-nonlinear-mpl PYTHONPATH=. \
python tools/reference_validation/validate_hank_nonlinear_continuation.py \
  --output /tmp/puremacro-nonlinear-new
```

The runner records source and artifact hashes, numerical-library identities and
verified single-thread Accelerate status where its API is available. Timings
describe that run and are not portable speed guarantees. The held-out seed,
history and original `1e-5` fixed-policy audit threshold are unchanged; numerical
refinement adds horizon or branches rather than relaxing acceptance thresholds.

The retained 80-by-3, 48-date comparison gives:

| Terminal distribution basis | Fixed-policy market error | Corrected market error | Corrected dates | Observed correction time |
|---|---:|---:|---:|---:|
| Full admissible basis | `1.65e-5` | `7.06e-9` | 48/48 | 43.1 s |
| 16 snapshot directions | `2.74e-5` | `8.28e-9` | 48/48 | 978.7 s |

Both corrected histories pass the unchanged horizon and branching criteria.
The 16-direction boundary requires horizon 64 on 20 dates, while the full basis
requires at most 16. Smaller state dimension therefore does not automatically
reduce total nonlinear-solution cost. Euler/KKT maxima remain about `9.95e-4`.
The matched small-grid dense comparison differs by `1.46e-11` in root log
controls; order 2-to-3 and 3-to-5 quadrature comparisons differ by at most
`2.64e-8` across the reported root metrics. These are case-specific checks.
