> 🇬🇧 English · 🇪🇸 [Español](es/equilibrium_linearization.md)

# Shared equilibrium derivatives

`puremacro.core.EquilibriumSystem` describes square equilibrium equations
\(F(x,\theta)=0\). `linearize_equilibrium` retains their local Jacobians
and solves \(F_x\,\partial x/\partial\theta=-F_\theta\).
This initial implementation uses dense matrices with a configurable state
limit. [Branch continuation](equilibrium_continuation.md) reuses these equations.
Statistical confidence sets remain a separate, unimplemented extension.

## Explicit equations

```python
import numpy as np
from puremacro.core import EquilibriumSystem, linearize_equilibrium

system = EquilibriumSystem(
    residual=lambda x, p: np.array([2*x[0] + x[1] - p[0], x[1] - p[0]]),
    state_names=("capital", "price"),
    parameter_names=("technology",),
)
lin = linearize_equilibrium(system, [0., 1.], [1.], branch_id="benchmark")
result = lin.sensitivity()
assert result.valid
print(result.to_frame())  # derivatives: capital=0, price=1
```

Optional callbacks `state_jacobian(x, theta)` and
`parameter_jacobian(x, theta)` are checked against centered residual
differences. Without callbacks, residual differences supply both matrices:
these are **numerical derivatives**, even though the implicit-function step
solves a linear system.

`lin.jvp(v)` and `lin.vjp(w)` apply \(F_x\) and \(F_x^T\).
`lin.solve(rhs)` and `lin.solve(rhs, transpose=True)` reuse one LU
factorization. These operations form dense matrices. The retained equations
are available through `lin.system.evaluate(x, theta)`.
Generic callbacks must be pure and remain caller-owned; the adapters below
snapshot their numerical inputs.

## Growth-model projection

```python
from puremacro.vfi import CollocationProblem, linearize_projection

kss = (.36 * .96)**(1 / (1 - .36))
problem = CollocationProblem(
    domain=(.5*kss, 1.5*kss), orders=6, method="euler", beta=.96,
    params={"alpha": .36, "delta": 1., "sigma": 1.},
    options={"tol": 1e-12, "max_iter": 500},
)
solution = problem.solve()
lin = linearize_projection(solution, problem, parameters=("alpha", "beta"))
result = lin.sensitivity()
assert result.valid, result.reasons
policy_derivative = solution.basis.evaluate(kss) @ result.jacobian
```

States are Chebyshev coefficients. Policy derivatives hold the evaluation
state, basis and domain fixed. Supported inputs are built-in one-dimensional
Euler growth equations, `beta`, and explicitly declared
`alpha`, `delta`, `sigma`/`gamma`, `z`/`A` parameters. A shadowed alias is
rejected. Bellman, custom callbacks, FEM and splines need another adapter or
explicit residual equations. Baseline physical clipping and continuation
outside the basis domain are rejected.

## Continuous Aiyagari

```python
from puremacro.vfi import solve_aiyagari_continuous, linearize_aiyagari

equilibrium = solve_aiyagari_continuous(
    beta=.95, n_a=30, N_k=60, n_z=3,
    xtol=1e-11, egm_tol=1e-11, tol_ge=1e-6,
)
lin = linearize_aiyagari(equilibrium, parameters=("alpha", "beta", "gamma", "delta"))
result = lin.sensitivity()
assert result.valid, result.reasons
print(result.to_frame().loc[["interest_rate"]])
```

Unknowns are consumption at household EGM nodes, histogram probabilities, and
the equilibrium interest rate, in that order. The equations combine the
Coleman fixed point, stationary transport with one equation replaced by mass
normalization, and capital-market clearing. Reshape blocks in row-major
order: asset index first, income state second.

The actual income chain is preserved, including custom chains. Supported
parameters are `alpha`, `beta`, `gamma`, `delta`; grids and income processes
remain fixed. The borrowing constraint uses a local active-set extension:
saving is identically zero at binding nodes while their consumption equation
enforces the budget. Stencils check binding sets, interpolation cells and
histogram lottery cells. An unresolved switch withholds ordinary derivatives.
Generic `continuous_stationary_equilibrium` outputs are unsupported.

There are `(n_a + N_k) * n_z + 1` states. This example has 271; the solver's
default grids exceed the default `max_states=1500` linearization budget.
Increasing this guard requires budgeting dense storage and cubic
factorization cost. Small example grids establish API behavior, not economic
grid accuracy or scalability to HANK.

## Diagnostics and scope

Positive `state_scale`, `parameter_scale`, `residual_scale` define coordinate
and equation units. Stencil steps are `step * coordinate_scale` (default
`step=1e-5`) and can shrink eight times. Singular values use
`diag(1/residual_scale) @ F_x @ diag(state_scale)`.
Supply corresponding scales when changing units; condition diagnostics
depend on those choices.

Defaults are `residual_tol=1e-7`, `derivative_tol=1e-3`, `singular_tol=1e-10`
and `condition_limit=1e10`. Failure reasons include `not_equilibrium`,
`regime_change`, `inconsistent_stencil`, `derivative_mismatch`, `singular`,
`ill_conditioned` and `linear_solve_failed`. Ineligible sensitivities
contain NaNs; linear solves refuse ineligible linearizations. No silent
regularization or equilibrium polishing occurs. Invalid inputs and
callback/domain exceptions raise directly; stencils require a two-sided
neighborhood in the declared coordinates.

Arrays are read-only snapshots. `snapshot_id` hashes numerical inputs,
matrices and scales, not source code. `branch_id` is a caller label, not
proof of branch discovery. Stencil checks and small residuals do not prove
smoothness, uniqueness, grid accuracy, statistical identification or valid
inference. Existing [specialized IFT APIs](vfi_analytic_gradients.md) retain
their behavior.

Results offer `summary`, `to_frame`/`to_dataframe`, `to_markdown`, `to_latex`,
`to_typst` and `plot(states=...)`. Plots never call `show` and require
at most 30 selected state rows.

Run `python -m puremacro.examples.equilibrium_linearization --output /tmp/equilibrium-derivatives`
for CSV/JSON records of growth-policy, aggregate-capital and interest-rate
sensitivities, plus a singular example. Tests compare independent model
re-solves across two growth calibrations and two income processes, including
the whole Aiyagari distribution and its zero total-mass derivative.
