# Composing the research APIs

[Español](es/research_api.md)

The research modules expose separate numerical and economic contracts. The
table below gives the public entry points for the nine initial research slices.
Imports through each parent package and the listed module refer to the same
objects. Existing model-specific interfaces remain available.

| Purpose | Public module and entry point | Acceptance means |
| --- | --- | --- |
| Local equilibrium derivatives | `core.linearize_equilibrium`; `vfi.linearize_projection`, `vfi.linearize_aiyagari` | `sensitivity().valid`: regular retained equations and tested stencils |
| Branch continuation | `core.continue_equilibrium` | Inspect termination `status`, rejected `attempts` and individual fold statuses |
| Simulation-aware estimation | `structural.fit_simulated_moments` | `success` concerns optimization; `inference_valid` additionally checks local inference |
| Identification-robust inference | `structural.simulated_moment_test`, `invert_quadratic_moments`, `evaluate_moment_grid` | A point test's `accepted=None` means unresolved; sampled grids never certify their interiors |
| HANK scalability | `models.solve_hank_reduced_policy` | First-order reduced solution; full-grid and nonlinear accuracy require their own audits |
| Integrated trade | `trade.solve_reallocation_transition` | `converged` concerns finite equations; use `run_reallocation_horizon_ladder` for tested horizon accuracy |
| Heterogeneous information | `expectations.build_gaussian_information`; `models.solve_information_nk` | Conditional Gaussian forecasts and finite NK coefficient equations under listed signals |
| Sovereign banking | `sovereign.solve_sovereign_banking` | Audited three-date pure equilibria with explicit `price_selection="highest"` or `"lowest"` |
| Adaptation and sovereign risk | `sovereign.solve_sovereign_adaptation` | Audited finite-horizon Bellman equations, lender prices and discrete choices |

See the detailed guides for [derivatives](equilibrium_linearization.md),
[continuation](equilibrium_continuation.md), [SMM](simulation_aware_smm.md),
[robust inference](identification_robust_inference.md), [HANK](hank_reduction.md),
[trade](trade_reallocation.md), [information](heterogeneous_expectations.md),
[banking](sovereign_banking.md) and [adaptation](sovereign_adaptation.md).

## Explicit composition

An equilibrium residual can provide sensitivities and a local branch. A caller
can then define measured model moments for SMM and use the retained simulation
batch for a point-null test. The measurement definition, units and statistical
assumptions must be supplied explicitly. This small example has equilibrium
`capital = 2 * technology` and an additive simulation error:

```python
import numpy as np
from puremacro.core import EquilibriumSystem, linearize_equilibrium, continue_equilibrium
from puremacro.structural import (
    MomentTargets, SimulationPlan, fit_simulated_moments, simulated_moment_test,
)

system = EquilibriumSystem(lambda x, theta: x - 2*theta, ("capital",), ("technology",))
linear = linearize_equilibrium(system, [2.], [1.])
assert linear.sensitivity().valid
branch = continue_equilibrium(system, [2.], [1.], max_steps=3)
np.testing.assert_allclose(branch.states, 2*branch.parameters, atol=1e-8)

targets = MomentTargets([2.], [[.04]], ("capital",), units="level")
plan = SimulationPlan(seed=12, replications=16, periods=20)

def simulator(theta, rng, budget):
    return 2*theta + rng.normal(size=budget.periods).mean()

def derivative(theta):
    return linearize_equilibrium(system, 2*theta, theta).sensitivity().jacobian

fit = fit_simulated_moments(simulator, targets, [.8], plan=plan,
    definition_id="capital-plus-sample-mean-v1", jac=derivative,
    parameter_names=("technology",), assume_independent=True, assume_smooth=True)
assert fit.success and fit.inference_valid
point_test = simulated_moment_test(targets, fit.simulation, assume_independent=True)
assert point_test.valid and point_test.accepted
```

The test at the fitted value is a descriptive compatibility check here. It is
not an overidentification test or a post-selection confidence statement. For
inference at a hypothesized parameter, simulate at that parameter and use its
covariance. Simulation plans retain every replication, including failures.
Generic callback exceptions propagate; invalid numerical moments remain
unresolved rather than becoming rejection or optimization penalties.

No automatic composition joins the trade, banking, adaptation or information
economies. Their household, financing, timing and terminal conditions differ.
Connecting their outputs requires an explicit new economic closure. Higher-order
HANK and the other deferred model extensions are outside these first slices.

## Ownership, controls and saved results

Equilibrium, SMM/robust-inference, trade, Gaussian/NK and sovereign result
snapshots have detached, immutable numeric storage. Obtain a writable working
array with `.copy()`. Existing HANK policy classes retain their own documented
result contract. Nested diagnostic mappings and sequences
are also frozen, and report tables are detached from retained numerical data.
Legacy metadata may contain arbitrary Python objects; their internal mutability
is outside the numeric-storage contract. Equilibrium tolerances and the scalar
primitives of the trade, NK and sovereign modules accept finite real scalars,
including zero-dimensional numeric NumPy arrays. Strings, booleans, complex
values and arrays with axes are rejected. The equilibrium `transpose` switch
requires a boolean; continuation `direction` requires integer -1 or 1.

Diagnostic mappings can be exported with an explicit JSON conversion:

```python
from collections.abc import Mapping
import json

def json_value(value):
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

encoded = json.dumps(dict(fit.diagnostics), default=json_value)
assert json.loads(encoded)["assume_independent"] is True
```

Result dataclasses are also used to reconstruct saved arrays. Construction
alone does not rerun an economic audit. Call the corresponding `audit_*`
function after restoring results and inspect `accepted`. Malformed retained
trade/information structures return a rejected audit with a reason; a wrong
top-level argument or invalid tolerance is an input error. Audit mappings are
fresh reports, so editing a returned audit cannot change a solution.

Model `fingerprint` values identify specifications, not solver output or source
code. Equilibrium `snapshot_id` values hash the retained point, matrices and
scales. Record source hashes, package/environment versions and output hashes
separately. Example CSV/NPZ/JSON bundles illustrate persistence; they are not a
versioned universal checkpoint format. A valid audit does not establish grid
convergence, empirical calibration, infinite-horizon accuracy or release readiness.
