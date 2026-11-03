> 🇬🇧 English · 🇪🇸 [Español](es/hank_recursive.md)

# Recursive HANK policies and two-asset equilibrium

`solve_hank_stochastic_policy` solves an infinite-horizon, **first-order local
rational-expectations policy** around a heterogeneous-household steady state.
The state contains distribution coefficients, the inherited nominal interest
rate and a persistent monetary disturbance. Household marginal values and
aggregate prices respond jointly to that state. Future aggregate innovations
continue indefinitely.

The construction follows the projection/perturbation distinction in
[Reiter (2009)](https://doi.org/10.1016/j.jedc.2008.08.010): solve nonlinear
household heterogeneity first, then linearize aggregate dynamics. The
[author's working paper](https://econ-papers.upf.edu/papers/972.pdf) explains
why a few distribution moments need not suffice. PureMacro uses constrained
distribution directions and an ordered generalized Schur decomposition.
This implementation is a local solution, with no second-order precautionary
shift, risk premium calculation or global stochastic welfare claim.

## Solve, compress and audit

```python
import numpy as np
from puremacro.models import (
    calibrate_one_asset_hank, solve_hank_stochastic_policy,
    solve_hank_stochastic_tree, monetary_scenario_tree,
)

# Demonstration grid; household-grid accuracy needs separate refinement.
ss = calibrate_one_asset_hank(n_a=80, n_e=3, a_max=150.)
full = solve_hank_stochastic_policy(ss, distribution_rank=None)
rng = np.random.default_rng(1201)
training = full.simulate(rng.normal(0, 1e-7, 160)).distributions
policy = solve_hank_stochastic_policy(
    ss, distribution_rank=16, snapshots=training,
    rho=.7, shock_std=1e-4,
)

# These innovations were not used to fit the distribution basis.
held_out = np.random.default_rng(9917).normal(0, 1e-4, 48)
audit = policy.audit(held_out, tolerance=1e-5)
print(audit.converged, audit.diagnostics)

tree = monetary_scenario_tree(
    horizon=8, persistence=.7,
    innovations=(-.001, .001), probabilities=(.3, .7),
)
equilibrium = solve_hank_stochastic_tree(policy, tree)
print(equilibrium.converged, equilibrium.diagnostics)
```

With no `snapshots`, the solver constructs directions from household price
responses and repeated distribution transport. Supplied snapshots instead
train an SVD basis around the steady distribution. Both retain the directions
needed to preserve contemporaneous market observables before reducing the
remaining variation. A rank below that minimum is rejected. Neither method fits the
equilibrium policy coefficients to a regression: those coefficients solve the
linearized household, distribution, market and price-setting equations.
`distribution_rank=None` retains the full constrained distribution, providing
a comparison for compression error.

The one-asset basis preserves each income marginal and aggregate inherited
assets. The two-asset basis preserves each income marginal and inherited
liquid assets; illiquid wealth can change with the consol price. The projection
also preserves aggregates of the steady household policies, the labor condition
and inherited illiquid wealth. Its columns are
orthonormal, and their coefficients describe **signed deviations** from the
steady distribution. `basis.decode()` checks positivity by default. There is
no clipping or renormalization. Linear simulations report negative
reconstruction mass explicitly; a good moment fit does not establish a valid
distribution approximation.

`policy.simulate(innovations)` solves the linear state recursion. Its aggregate
paths are **log deviations**, including `log(1+inflation)`; they are not the
level-deviation paths of the nonlinear solver. `policy.covariance()` solves
the first-order stationary covariance equation. Changing `shock_std` changes
that covariance, but not the first-order policy coefficients or steady means.
That certainty equivalence is a property of the approximation order.

`policy.audit()` uses the full household grid and transports the actual
distribution on every held-out date. It solves consistency between next
distribution coefficients and household continuation values while retaining
the fitted aggregate policy. It reports market and goods residuals, budgets,
mass, grid exits, Euler/KKT errors and projection loss. Its distribution array
includes the final transported distribution, so it has one more row than its
paths. An audit can fail even when the projected Schur equations converge.
Euler/KKT errors include household interpolation error and are separate from
the aggregate acceptance threshold.

## A solved stochastic terminal boundary

`solve_hank_stochastic_tree` optimizes households and clears markets at each
finite-history node. Each leaf has additional unknown distribution
coefficients. They must equal the projection of the distribution actually
transported beyond that leaf. The terminal policy then supplies successor
household marginal values, inflation and output, conditional on those
coefficients, the chosen nominal rate and the persistent disturbance.

Terminal Phillips expectations use two-point Gaussian quadrature. Expected
household marginal values integrate the linear terminal rule exactly. The
finite portion is nonlinear, while the terminal rule remains first order;
nonlinear arithmetic at the boundary does not turn this into a second-order
stochastic solution. Horizon, shock-size and compression comparisons remain
necessary.

The default correction uses damped Newton–GMRES with centered products of the
full nonlinear residual and sparse preconditioning. The preconditioner
approximates local responses and asset carry; household expectations and
distribution effects remain in the outer residual. `preconditioner="none"`
provides a check. `method="hybr"` uses a dense finite-difference reference.
This is a separate continuation backend; the original
[analytic finite-tree solver](hank_optimized.md) is unchanged.

`converged` requires finite-tree market and goods residuals, terminal
consistency, household budgets, conserved/nonnegative mass and negligible
policy-grid exits. Terminal projection loss, minimum reconstructed terminal
mass, Euler/KKT errors and upper-grid mass are reported separately. Small
finite-tree residuals do not certify the projected infinite tail. Iteration,
linear-probe and line-search failures retain their last feasible iterate and
history. Invalid initial domains raise; the dense comparison can also raise
if its unguarded MINPACK trial leaves the economic domain.

## Liquid bonds and an illiquid real consol

```python
from puremacro.models import calibrate_two_asset_equilibrium

two_ss = calibrate_two_asset_equilibrium(n_a=15, n_b=15, a_max=80.)
two_policy = solve_hank_stochastic_policy(two_ss, distribution_rank=16)
two_result = solve_hank_stochastic_tree(two_policy, tree)
print(two_result.paths[["liquid_mkt", "illiquid_mkt", "goods_mkt"]])
```

This closure has liquid nominal government bonds of fixed real quantity
\(B\), and an illiquid real consol with fixed unit supply \(S\), price
\(q_t\), and constant real coupon \(d\). The realized returns are

\[
1+r^b_t=\frac{1+i_{t-1}}{1+\pi_t},\qquad
1+r^a_t=\frac{q_t+d}{q_{t-1}}.
\]

The existing two-asset EGM household chooses liquid and illiquid savings,
paying its convex portfolio adjustment cost \(\chi\). Its budget is

\[
c+a'+b'+\chi=(1+r^a_t)a+(1+r^b_t)b+z_t e.
\]

The income process has the existing two productivity states. Wages, profits
and taxes are distributed proportionally to productivity, whose mean is one.
With labor-only production and common union hours \(N_t=Y_t\), flexible
labor supply satisfies
\(w_t\int e c^{-\gamma}\,dD_t=\varphi Y_t^{1/\nu}\).
Rotemberg pricing and the Taylor rule complete the aggregate block.

Taxes are \(T_t=r^b_tB+dS\). Consol capital gains enter household returns,
not government expenses. Total nonfinancial income is
\(z_t=Y_t-\text{price adjustment cost}_t-T_t\).
The solver clears **both** asset markets,
\(\int b'\,dD_t=B\) and \(\int a'\,dD_t=q_tS\), alongside labor supply
and the Phillips equation. It independently checks
\(Y_t=C_t+\int\chi\,dD_t+\text{price adjustment cost}_t\).

Calibration takes household preferences, adjustment costs, steady returns
and nonfinancial income as inputs. Household demands determine \(B\) and
\(S\); \(q=1\), the coupon equals the steady illiquid return, and labor
disutility matches steady output/hours. This is a consol economy, not the
productive-capital closure of a published two-asset HANK replication. The
steady state's `income` is distinct from its aggregate wage `w`.

## Numerical limits and evidence

One-asset policy derivatives are analytic. Two-asset policy derivatives use
cached centered household differences and differentiated bilinear transport;
tests compare their joint equations with independently evaluated nonlinear
period equations. Both models check Schur determinacy, stability, conditioning
and equation residuals. No pseudoinverse or arbitrary stable policy replaces
a failed determinacy check.

The distribution is reduced, but **all household marginal-value controls are
retained**. Ordered QZ is dense. `max_system_size` defaults to 1,800 and checks
the dimension before basis construction; it is not an RSS bound. The tree's
`max_workspace_bytes` bounds an estimated working-array footprint, excluding
the already prepared policy, sparse-factor and Python/BLAS overhead. Large
household grids will need further value-function reduction or an iterative
stable-subspace solver.
In particular, the existing one-asset calibration's default 500 × 7 grid
exceeds this Schur limit; the examples deliberately use smaller grids.
Raising the limit permits a larger dense allocation without changing its
computational complexity.

`tools/reference_validation/validate_hank_recursive.py` records full and
compressed policies, held-out audits, horizon comparisons, dense/Krylov
agreement and two-asset grid sensitivity. Its frozen artifacts live under
`tests/fixtures/hank_recursive/`: source and artifact hashes, state/feedback
matrices, seeds and CSV histories permit independent recomputation. Failed
audits remain in the record. Grid-sensitivity runs hold household preference
and price inputs fixed while recalibrating asset supplies; they are not a
certification of a fixed-supply economy's grid convergence.

The retained run uses 80 asset points and 3 income states for one asset, and
12 × 12 asset points and 2 income states for two assets. Its rank-16 snapshot
bases were trained on 160 small innovations and evaluated on 48 independent
innovations with standard deviation `1e-4`.

| Check | One asset | Two assets |
|---|---:|---:|
| Full → compressed distribution dimension | 236 → 16 | 285 → 16 |
| Full → compressed Schur dimension | 481 → 261 | 867 → 598 |
| Largest held-out linear aggregate gap | `4.44e-8` | `2.90e-10` |
| Full-policy nonlinear audit residual | `1.65e-5` (fails) | `8.52e-6` (passes) |
| Compressed-policy nonlinear audit residual | `2.74e-5` (fails) | `9.49e-6` (passes) |
| Compressed audit at one-quarter amplitude | `1.03e-6` (passes) | `5.33e-7` (passes) |
| Largest nonlinear tree residual, horizons 4/8/16/32 | `3.72e-10` | `5.54e-11` |
| Dense versus Krylov tree path gap | `2.07e-8` | `8.31e-10` |

The rollout audit threshold is `1e-5` in all audit rows. The one-asset failure also occurs
with the full constrained distribution, so a small linear compression gap
does not remove nonlinear approximation error. The two-asset result is near
the acceptance boundary; it is not a guarantee for other shock histories.
Audits use the previous **quoted** consol price when computing realized
returns, even when the approximate policy leaves an asset-market residual.
Maximum household Euler/KKT errors on these demonstration grids remain about
`1e-3` and `3e-3`, respectively. The tree convergence flag does not certify
those household-grid errors.

`prior_variance_only/` retains the first compression experiment, its exact
source bytes and failed determinacy checks. Preserving market observables
before truncation resolves those failures in the reported rank-8/rank-16
two-asset cases; rank 4 is now rejected explicitly because it omits required
directions. Timing observations in the JSON are descriptive, not a repeated
performance benchmark.
