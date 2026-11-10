> 🇬🇧 English · 🇪🇸 [Español](es/equilibrium_continuation.md)

# Equilibrium branch continuation

`puremacro.core.continue_equilibrium` follows a local branch of
\(F(x,\theta)=0\), varying one named parameter and holding the others fixed.
It uses the same [residual contract](equilibrium_linearization.md) and checked
Jacobians as the shared derivative interface.

The predictor advances along a tangent in scaled state/parameter coordinates;
a damped Newton corrector solves the equations plus a tangent hyperplane.
This is the pseudo-arclength construction described in the
[BifurcationKit documentation](https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/PALC/).
The augmented system can remain regular at a simple fold where the ordinary
parameter derivative fails; see
[Dickson et al.](https://arxiv.org/abs/math/0603716) for its conditioning.

## Trace and inspect a branch

```python
import numpy as np
from puremacro.core import EquilibriumSystem, continue_equilibrium

system = EquilibriumSystem(
    lambda x, p: x**3 - x - p,
    ("capital",), ("policy",),
)
branch = continue_equilibrium(
    system, [-1.5], [-1.875],
    step=.08, max_step=.12, max_steps=100,
    parameter_bounds=(-2., 2.), branch_id="cubic-example",
)
assert len(branch.folds) == 2
assert all(f.status == "simple_fold" for f in branch.folds)
print(branch.summary())
ax = branch.plot("capital")
```

The initial point must satisfy the supplied equations. `direction=1` or
`-1` selects the initial parameter direction; it may reverse at a fold.
With several parameters, specify `parameter="name"`. Invalid seeds, invalid
options and model callback/domain exceptions raise directly.

`states` and `parameters` store physical coordinates. `tangents` are unit
vectors in scaled state/selected-parameter coordinates, with the parameter
last. `arclength` is cumulative chord length in those coordinates;
`step_sizes` are the requested tangent-hyperplane distances. Arrays are
read-only snapshots. Generic callbacks remain caller-owned and must be pure;
built-in model adapters retain owned numerical inputs.

`to_frame`/`to_dataframe`, `summary`, `to_markdown`, `to_latex`,
`to_typst` and `plot` expose accepted points. Columns use `state:` and
`parameter:` prefixes. `attempts_frame()` includes accepted and rejected
attempts, iteration counts, residuals and reasons. Diagnostics unavailable
at rejected predictors are NaN. Plots never call `show`.

## Restart or reverse

```python
reverse = continue_equilibrium(
    branch.system, branch.states[-1], branch.parameters[-1],
    parameter=branch.parameter_name,
    initial_tangent=-branch.tangents[-1],
    step=.03, max_step=.05, max_steps=8,
)
assert reverse.states[-1, 0] < reverse.states[0, 0]
```

A supplied `initial_tangent` controls orientation and supersedes `direction`;
it must lie in the scaled augmented nullspace. Starting at a parameter turning
point requires this explicit orientation. Reversing hyperplane steps need not
visit identical numerical points.

## Reuse a model residual

```python
from puremacro.vfi import CollocationProblem, linearize_projection

kss = (.36*.96)**(1/(1-.36))
problem = CollocationProblem(
    domain=(.5*kss, 1.5*kss), orders=6, method="euler", beta=.96,
    params={"alpha": .36, "delta": 1., "sigma": 1.},
    options={"tol": 1e-12, "max_iter": 500},
)
solution = problem.solve()
lin = linearize_projection(solution, problem)
growth = continue_equilibrium(
    lin.system, lin.state, lin.parameters, parameter="beta",
    step=.002, max_step=.002, max_steps=8,
)
assert growth.status == "max_steps"
```

The same call accepts the system retained by `linearize_aiyagari`.
Consumption, stationary probabilities and interest rates move jointly.
Grids and the income chain stay fixed. This implementation stops at a
declared binding/interpolation/lottery regime change. Small smooth Aiyagari
traces are checked against independent re-solves; this does not establish
global heterogeneous-agent continuation across regime switches.

## Steps, termination and folds

Defaults are `step=.05`, `min_step=1e-5`, `max_step=.2`, `max_steps=100`,
`max_newton=12` and `max_retries=8`. Failed attempts halve the step;
successful steps adapt to corrector effort and tangent alignment. Large
corrections and sharp tangent changes trigger retries. Finite parameter
bounds stop the trace, possibly short of a boundary. Bounds constrain the
path, not derivative probes: equations still need a two-sided neighborhood.

Budget/bound termination uses `max_steps` or `parameter_bound`. Other statuses
retain the partial accepted path and identify the last diagnosed failure,
such as `regime_change`, `newton_failed`, `rank_deficient`, `rank_unresolved`,
`possible_branch_point`, `ill_conditioned` or `step_not_representable`.
Inspect status and attempts before using a path. A sign test can flag some
intervening branch points; it cannot detect every singularity between samples.

Default scaled equilibrium and hyperplane tolerances are `1e-9`.
`derivative_step=1e-5` and `derivative_tol=1e-3` govern checked Jacobians.
The augmented derivative must meet `rank_tol=1e-10` and
`condition_limit=1e10`. Near rank loss, a halved stencil checks whether its
smallest singular value exceeds the observed derivative discrepancy.

A parameter-tangent reversal brackets a fold candidate. Refinement uses up to
`fold_max_iter=40` corrections and scaled bracket tolerance `fold_tol=1e-7`.
Records include state corank, augmented conditioning, left-null
transversality and a quadratic directional coefficient. Curvature uses
`curvature_step=1e-4` and half that step; `degeneracy_tol=1e-5` controls
nondegeneracy checks. Records distinguish `simple_fold`,
`degenerate_or_unresolved` and `unresolved`. These are numerical
classifications at declared scales/tolerances, not mathematical certificates.
Refined folds are separate from the sampled path. Nearby folds or flat turning
points can be missed at finite resolution.

## Scope and reproducible example

The method is dense, with `max_states=1500` by default. It retains path
arrays, not every dense Jacobian. Coordinate/equation scales affect geometry
and conditioning. There is no automatic branch switching, disconnected-root
enumeration, global completeness claim, stability classification, Hopf
analysis or statistical inference. `branch_id` is only a caller label.

Run `python -m puremacro.examples.equilibrium_branches --output /tmp/equilibrium-branches`.
The stylized fixed-saving model has production
`f(k)=.05+k**2/(1+k**2)` and equilibrium `.1*k=s*f(k)`.
CSV/JSON and PNG/PDF records compare the traced path and folds with an
independent closed-form curve. At `s=.20`, independent scalar root solves
recover three equilibria and agree with the associated cubic polynomial.
This illustration is not an estimated economy or an optimal-saving model.

Tests also cover a cubic S-curve, coupled equations, unit transformations,
degenerate folds, branching seeds, retries and failures, fixed nonsmooth
parameters, and independent RBC/Aiyagari re-solves.
