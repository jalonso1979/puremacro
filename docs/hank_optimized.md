# Optimized, coupled HANK equilibrium

`HANKEquilibriumWorkspace` solves inflation, wages, output, household policies
and the distribution jointly at every node of a finite aggregate history tree.
Its matrix-free Newton–GMRES backend evaluates analytic Jacobian-vector
products through the household and distribution blocks. It supports the same
one-asset economics as [the finite-tree solver](hank_scenario_tree.md), with a
reusable sparse preconditioner and explicit warm starts.

This is a step toward scalable full-equilibrium HANK computation. The solved
object is still a **finite-history, one-asset equilibrium**. It has a steady
terminal marginal value, risk-neutral nominal firm valuation, a fixed real
debt stock, and idiosyncratic risk independent of aggregate innovations. It
does not solve a stationary stochastic tail, a policy over the space of
distributions, two-asset portfolio choice, or an effective lower bound.

## Repeated policy experiments

```python
from puremacro.models import (
    HANKEquilibriumWorkspace,
    calibrate_one_asset_hank,
    monetary_scenario_tree,
)

# Demonstration grid; research accuracy requires separate refinement checks.
ss = calibrate_one_asset_hank(n_a=240, n_e=7, a_max=150.)
tree = monetary_scenario_tree(
    horizon=48, innovations=(-.005, .005), probabilities=(.3, .7),
)
workspace = HANKEquilibriumWorkspace(ss, tree)
baseline = workspace.solve()
nearby = workspace.solve(
    monetary_shocks=1.02 * tree.monetary_shock,
    initial=baseline,
)
for result in (baseline, nearby):
    print(result.converged, result.diagnostics["market_residual_max_abs"])
    print(result.diagnostics["evaluations"],
          result.diagnostics["jacobian_vector_products"])
```

The workspace copies its inputs. Changing the original calibration or tree
does not alter a prepared problem. Monetary shocks can change on the existing
topology; a labeled Series must have the same index and row order. Changes to
parameters, grids, conditional probabilities or terminal data require a new
workspace. A result used as a warm start must match that workspace's input
fingerprint. There is no automatic reuse of the previous solution.

`solve_hank_equilibrium(ss, tree)` is the one-shot interface. Both interfaces
return `HANKScenarioTreeResult`, retaining node policies and conditional
distributions as well as aggregate paths and unconditional date expectations.

## Where the saving comes from

The [sequence-space approach of Auclert, Bardóczy, Rognlie and Straub
(2021)](https://doi.org/10.3982/ECTA17434) motivates separating the household,
distribution and aggregate-equilibrium mappings. The existing published-model
solver uses stationary sequence Jacobians for deterministic transitions. The
new backend differentiates the **current nonlinear iterate** on a tree:

1. A backward EGM pass computes household choices using conditional expected
   successor marginal values. Its tape stores interpolation brackets, slopes
   and borrowing-constraint regimes.
2. An analytic tangent pass differentiates expected marginal values,
   endogenous grid locations, consumption, labor and saving. Binding households
   use derivatives of the budget and intratemporal first-order condition.
3. Forward tangents move both the existing distribution and its asset-lottery
   weights. Changes in distribution composition enter aggregate assets and
   labor. Expectations use conditional probabilities before aggregation.
4. [GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html)
   solves the equilibrium correction using these products without assembling
   a dense aggregate Jacobian. A damped line search then evaluates the full
   nonlinear household/distribution problem again.

The preconditioner approximates local household price responses and the carry
of aggregate assets, together with the Phillips and policy-rate equations.
It uses a sparse LU factorization on the history graph, prepared once and
reused across experiments. It omits intertemporal and distribution-composition
effects **only inside the preconditioner**; the actual Jacobian-vector products
and final residuals include them. `preconditioner="none"` provides a comparison.

At a borrowing or interpolation kink, the current branch supplies a selected
derivative. A classical derivative across the kink need not exist. Tapes are
rebuilt after every accepted nonlinear step. Finite differences that cross
different branches need not match that linear map.

## Derivative and failure inspection

Coordinates concatenate `log(1 + inflation)`, `log(wage / steady_wage)` and
`log(output)` in tree row order. `workspace.residual(x)` returns Phillips,
asset and effective-labor residuals, in that block order.
`workspace.jacobian_vector_product(x, direction)` exposes the analytic product
for independent derivative checks. A plain coordinate vector can also be
passed as `initial`.

`converged` requires all three solved equations **and the goods-market
identity**, mass conservation, nonnegative distributions up to roundoff,
household budgets and negligible policy-grid exit mass. Euler/KKT errors,
upper-grid mass and minimum consumption are reported separately. Convergence
does not establish asset/income-grid accuracy, horizon accuracy, uniqueness,
or stationary stochastic welfare.

Invalid inputs and initial economic domains raise. Iteration limits, failed
linear solves and unsuccessful line searches return the last feasible iterate
with `converged=False`. `optimizer_message` and `history` explain termination;
each Newton entry records the independently recomputed linear residual and
accepted step length. A singular sparse preconditioner raises with the option
to try `preconditioner="none"`. Failures are never replaced by a linear IRF.

The dense reference remains available as `solve_hank_scenario_tree`. Its
256-node cap is absent from the new backend. Memory still grows with the number
of histories times the household grid size. The default one-GiB
`max_workspace_bytes` checks a conservative working-array estimate before
allocation; it is **not a process-RSS limit** and excludes sparse-factor,
Python and BLAS overhead. Large Krylov restart settings increase the estimate.

## Evidence and performance

Tests compare analytic tangents with central differences, including unequal
EIS and Frisch elasticities and binding households; check mass and the asset
first moment under differentiated distribution transport; compare nonlinear
equilibria with the dense reference and the published deterministic solver;
and solve a 279-node tree beyond the dense backend's cap. They also exercise
warm starts, input ownership, solver-budget failures and validation guards.

The reproducible benchmark is
`tools/reference_validation/benchmark_hank_equilibrium.py`. It records source
hashes, environment, calibration time, repeated cold solves including workspace
setup, explicit warm starts, equation errors, household-policy differences and
CSV paths. Dense and optimized methods alternate timing order. Numerical
equivalence is checked on identical grids; timing is not a CI pass criterion.

The frozen `tests/fixtures/hank_optimized/record.json` records the following
medians of three repetitions on arm64 with Python 3.11.17, NumPy
2.4.6 and SciPy 1.17.1. Both libraries use Apple Accelerate. The runner sets
and verifies its single-thread mode using
[`BLASSetThreading` / `BLASGetThreading`](https://developer.apple.com/documentation/accelerate/blassetthreading(_:))
on the calling thread, alongside OMP/OpenBLAS/MKL/vecLib environment limits.
All recorded solves converged at a market tolerance of `1e-8`. Largest
aggregate path difference from the dense reference: `9.07e-09`.

| Assets × incomes | Periods / nodes | Dense (s) | Optimized cold (s) | Ratio | Warm (s) |
|---|---|---:|---:|---:|---:|
| 240 × 7 | 48 / 95 | 19.38 | 1.18 | 16.4× | 1.03 |
| 480 × 31 | 48 / 95 | 58.78 | 6.01 | 9.8× | 5.14 |
| 240 × 7 | 96 / 191 | 76.71 | 3.61 | 21.3× | 3.10 |

Calibration is excluded from these solve timings. These observations are
machine- and case-specific; they do not establish grid accuracy. The retained
CSV paths support independent aggregate comparisons. Large household arrays
are not archived; reproducing their recorded maximum differences requires
rerunning the benchmark. `prior_uncontrolled/` retains the initial observations
and exact runner, whose environment settings did not explicitly pin Accelerate;
those timings are superseded by this verified-thread run.

## Next steps toward stationary stochastic HANK

The next difficult step is representing the distribution state without
expanding an exponentially growing tree. A practical extension should combine
a compressed distribution representation with a solved stochastic terminal
policy, then test held-out aggregate histories against the full distribution,
Euler/KKT conditions and market residuals. Moment fit alone is insufficient.

After that interface is validated, the same tangent/transport design can be
extended to liquid and illiquid assets, portfolio adjustment constraints and
richer fiscal rules. Two-asset household derivatives and asset-market closure
need their own independent references. Compiled or accelerator kernels should
be added only after profiling those blocks, keeping this NumPy/SciPy solver as
the portable numerical comparison.
