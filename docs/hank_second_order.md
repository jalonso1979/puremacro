> English · [Español](es/hank_second_order.md)

# Experimental second-order one-asset HANK

`solve_hank_second_order` adds quadratic state terms and anticipated-variance
corrections to a small one-asset HANK policy. It retains every household
marginal control and the complete distribution basis subject to mass, income
and bond-market restrictions. Reduced bases, two assets, productive capital,
global accuracy and welfare are outside this implementation.

```python
import numpy as np
from puremacro.models import (
    calibrate_one_asset_hank, solve_hank_reduced_policy,
    solve_hank_second_order,
)

ss = calibrate_one_asset_hank(n_a=20, n_e=2, a_max=120.)
linear = solve_hank_reduced_policy(
    ss, distribution_rank=None, marginal_rank=None, shock_std=0.0005,
)
policy = solve_hank_second_order(linear)
x = policy.state(monetary_shock=0.0005)
audit = policy.audit(x, tolerance=1e-5, quadrature_order=3)
assert audit.accepted
assert audit.diagnostics["mass_error"] < 1e-10
assert audit.diagnostics["budget_error"] < 1e-8
assert audit.diagnostics["equation_error"] < 4e-7
assert audit.diagnostics["euler_kkt_is_grid_diagnostic"]
```

## Timing and approximation

The state includes the distribution coordinates, inherited nominal interest
rate and current monetary shock. It is observed after the current innovation.
For state `x`, the conditional policy is

\[
y = Gx + \tfrac12 G_{xx}[x,x] + \tfrac12 G_{\sigma\sigma}\sigma^2,
\qquad E[x'\mid x] = Hx + \tfrac12 H_{xx}[x,x]
                       + \tfrac12 H_{\sigma\sigma}\sigma^2.
\]

The next innovation enters only the monetary-shock coordinate. Household
expected marginal values are averaged **before** the current backward step;
the nonlinear Phillips continuation is averaged separately. The state and
variance coefficient equations are solved separately, including future policy
curvature in the variance equation. Tensors use the one-half Taylor convention
and do not already contain the chosen innovation variance.

Centered derivatives at `difference_step` and half that step must agree.
Richardson extrapolation follows this check. The solver rejects inconsistent
first derivatives, unresolved curvature, singular variance equations and
excessive state/workspace dimensions. It does not smooth policy kinks or add a
ridge. Agreement of two steps is evidence about the local calculation, not a
proof of global smoothness across borrowing or interpolation boundaries.

The state/variance expansion follows
[Schmitt-Grohé and Uribe](https://www.columbia.edu/~mu2166/2nd_order/2nd_order.pdf).
Simulation keeps first- and second-order state components separate, following
the order-consistent treatment discussed by
[Kim, Kim, Schaumburg and Sims](https://www.federalreserve.gov/pubs/feds/2003/200361/200361pap.pdf).
Quadratic terms use only first-order states; nonlinear feedback of quadratic
corrections would introduce unintended higher-order terms.

## Anticipated risk versus state dispersion

```python
mean = policy.mean_decomposition()
np.testing.assert_allclose(
    mean["state_mean"], mean["state_risk"] + mean["state_dispersion"],
)
np.testing.assert_allclose(
    mean["control_mean"], mean["control_risk"] + mean["control_dispersion"],
)
zero = policy.state()
np.testing.assert_allclose(
    policy.controls(zero, shock_std=0.0005),
    4 * policy.controls(zero, shock_std=0.00025),
)
path = policy.simulate(np.array([0.0002, -0.0001, 0., 0.]))
assert path.converged
assert path.diagnostics["pruned"]
```

`state_risk` is the limit under anticipated risk and zero realized innovations.
`state_dispersion` accounts for the curvature of the state rule evaluated at
the first-order stationary covariance. Their sum is the mean of the pruned
approximation. Control means are coordinates: the last three are log inflation,
log wage and log output deviations. They are not transformed level means or
household welfare. Reusing a nonlinear utility formula on these objects does
not supply an audited welfare measure.

## What the checks establish

`controls` and `next_state` evaluate polynomials; `audit` independently evaluates
the nonlinear **discrete** household, distribution and market equations using
Gaussian quadrature. `simulate` audits every pruned step. Failed equation,
probability or positive-domain checks remain failures. Distributions are never
clipped or renormalized to rescue a second-order path. Pruning does not ensure
distribution feasibility.

Euler/KKT residuals are reported separately from finite-grid equation acceptance.
At the demonstrated 20-asset grid they are around 0.004, despite much smaller
discrete equation errors. Household-grid convergence is not certified.

The checked 12- and 20-asset examples have cubic signed-shock equation errors
above their numerical floor. At 20 assets, shock/innovation scales of 0.001,
0.0005 and 0.00025 give maximum scaled errors of roughly 2.34e-6, 2.91e-7 and
3.62e-8. Exact polynomial fixtures and a separate dense coefficient solve check
the state/variance algebra. Independently solved finite nonlinear equilibria
provide an additional comparison; they retain their own first-order terminal
boundary. Stochastic horizon sensitivity remains material, and agreement with
one such tree does not establish infinite-horizon or global accuracy.

The stricter 1e-9 finite-reference check fails on the 12-asset calibration's
approximately 1.47e-9 equilibrium floor. Those failures are retained; the
20-asset references meet the same tolerance. No tolerance was relaxed.

Reproduce the amplitude, quadrature, mean and nonlinear-reference records with
`tools/reference_validation/benchmark_hank_second_order.py`. Inspect all
failed cases and horizon comparisons before extending the approximation.
