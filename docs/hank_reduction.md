# Reducing household controls before solving HANK equilibrium

`solve_hank_reduced_policy` compresses marginal-value deviations together with
distribution states. It applies household derivatives to retained directions
and constructs the equilibrium matrices in reduced coordinates. The two-asset
path never constructs the legacy full household derivative cache.

The stationary household solution, endogenous-grid backward steps, binding
asset constraints, and probability transport retain the full household grid.
This is a local first-order aggregate-risk approximation, with an infinite
stochastic continuation. It does not supply a precautionary shift, a global
policy, or a second-order welfare calculation.

```python
import numpy as np
from puremacro.models import calibrate_one_asset_hank
from puremacro.models.hank_reduction import solve_hank_reduced_policy

steady = calibrate_one_asset_hank()  # 500 asset points, 7 income states
policy = solve_hank_reduced_policy(
    steady, distribution_rank=96, marginal_rank=128, krylov_steps=96,
    rho=0.7, shock_std=0.0001,
)
linear = policy.simulate(np.r_[-1e-5, np.zeros(39)])
audit = policy.audit(np.r_[-1e-5, np.zeros(9)], tolerance=1e-5)
print(policy.diagnostics)
print(audit.converged, audit.diagnostics)
```

The distribution basis preserves mass, stationary income marginals, and the
market-observable directions of the existing recursive solver. The marginal
basis contains the independent responses to current aggregate prices. Further
directions come from repeated backward marginal-response applications and an
SVD. Thus the training space includes responses to anticipated prices, without
fitting equilibrium policy coefficients. `marginal_snapshots` can instead
contain positive full-grid marginal values with shape `(observations,
number_of_marginals)`; current price-response directions remain mandatory.

Requested marginal rank is capped at the independent training span and the
effective rank is recorded. `krylov_steps` controls the training horizon, so
rank refinement should also check that horizon. `marginal_rank=None` retains
all controls for small-grid comparisons. The dimension guard runs before any
full marginal identity matrix or Schur matrix is constructed.

## Equations and acceptance

Let `B` contain retained distribution directions and `V` retained marginal
directions. The policy uses `D = D_ss + B z` and `m = m_ss + V v`. Each reduced
coordinate is expanded before the full-grid directional household evaluation;
the resulting marginal equation is projected with `V.T`. Ordered QZ solves
this projected system. A wrong stable-root count, a unit root, a singular
state projection, or failed projected-equation checks raises an exception.
No root is discarded to manufacture determinacy.

`policy.converged` and linear `simulate(...).converged` certify **projected
equations only**. They must be read alongside these separate diagnostics:

| Diagnostic | Meaning |
| --- | --- |
| `projected_equilibrium_residual_max_abs` | Residual of the solved reduced matrices |
| `full_marginal_relative_residual_max_abs` | Maximum omitted household equation coefficient, divided by `1 + abs(m_ss)`, for a unit state direction |
| `full_marginal_stationary_relative_rms_max` | Largest normalized marginal residual RMS under the policy's stationary shock covariance |
| `directional_projection_error` | Difference between a fresh directional household evaluation and the assembled projected solution |
| `reduced_jacobian_bytes`, `basis_bytes` | Stored array sizes; these are not process peak memory |

Unit-coordinate coefficient errors depend on state normalization. The
stationary RMS and held-out path errors provide shock-scaled information.
Neither a small training reconstruction error nor a small projected residual
certifies the omitted equations.

`audit` transports the actual nonnegative full distribution. It solves
continuation/transport consistency while holding aggregate policy coefficients
fixed, and checks **every** marginal equation. Acceptance requires normalized
full marginal errors and market/goods errors below `tolerance`, household
budget error below `1e-8`, mass error below `1e-10`, no material
negative mass or grid exits, and successful continuation solves. Euler/KKT
interpolation errors are reported separately. Signed linear reconstructions
and nonpositive marginal values are never silently clipped.

## Refinement and limits

Compare several distribution ranks, marginal ranks, training horizons and
shock sizes. Use the existing dense stochastic solver as an independent
small-grid comparison, and long sequence-space transitions when available.
The 500-by-7 example above needs substantially more distribution directions
than the small-grid defaults: preliminary checks with ranks 4 through 32
failed determinacy, while 48 and larger solved. These failed cases are a
reason to inspect ranks rather than an instruction to relax the root check.

For a 40-by-40 two-asset grid, marginal refinement can make linear paths agree
closely while a larger-shock nonlinear audit still fails its full marginal
criterion. A small market residual alone is insufficient. Final benchmark
records must report these failures alongside successful small-excursion cases.

The approach follows the separation of stationary household functions from
reduced aggregate perturbations described by [Bayer and Lütticke (2020)](https://www.ralphluetticke.com/publication/bl_qe/).
This implementation uses a household-response Krylov/SVD basis, rather than
claiming to reproduce their sparse-cosine implementation or calibration.

## Additional aggregate closures

`solve_hank_reduced_kernel` accepts the shared backward/forward/market kernel
protocol. Optional hooks `aggregate_state_names`,
`prices_from_state(controls, aggregate_state, distribution, *, lag_q=None)`,
and `next_aggregate_state(prices, aggregate_state, rho)` support further
predetermined states. The monetary innovation is located by its name, not by
the last array position. The price hook must honor a supplied actual lagged
asset price during off-equilibrium audits. Kernels must provide directional
household derivatives; a legacy cached two-asset kernel is rejected before
construction begins.

`feedback` expands to all marginal controls followed by aggregate controls;
`reduced_feedback` and the projected matrices use marginal-basis coefficients.
`_forecast(next_state, current_nominal)` retains the existing full-marginal
stochastic-continuation semantics.
