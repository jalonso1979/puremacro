# Identification-robust moment confidence sets

The structural module now offers point-null moment tests and continuous
confidence sets for an explicitly supported scalar moment class. These methods
do not invert a model Jacobian or presume that parameters are strongly
identified. They complement [local structural inference](structural_bridge.md)
and [simulation-aware SMM](simulation_aware_smm.md).

## The point-null test

For empirical moments \(\hat m\), model moments \(m(\theta)\), and the full
covariance \(V(\theta)\) of their difference, the statistic is

\[
S(\theta)=[\hat m-m(\theta)]'V(\theta)^{-1}[\hat m-m(\theta)].
\]

Compare it with a chi-square cutoff with **q degrees of freedom**, where q
is the number of moments. Do not subtract the minimized objective or use q-p.
The covariance is already on the estimator scale; no sample-size multiplier
is applied. This implements a full point-null S-test in the spirit of
[Stock and Wright (2000), Theorem 2](https://users.ssc.wisc.edu/~behansen/718/StockWright2000.pdf).

```python
import numpy as np
from puremacro.structural import (
    MomentTargets, QuadraticMomentCurve, moment_restriction_test,
    invert_quadratic_moments, evaluate_moment_grid,
    SimulationPlan, simulate_moments, simulated_moment_test,
)

targets = MomentTargets([1.0], [[0.01]], ("second",))
point = moment_restriction_test(targets, [1.0], calibration="gaussian_known")
assert point.valid and point.accepted and point.df == 1
print(point.to_frame())
```

`calibration="gaussian_known"` is the caller's assertion that the null moment
error is exactly Gaussian with known positive definite covariance. It is not
automatically verified. Default `calibration="asymptotic"` instead requires a
valid null CLT and consistent covariance estimation; uniform validity requires
corresponding uniform assumptions. Neither checks correct specification.

Supply `covariance=...` only for the **total moment-error covariance at the
candidate**, including any model/simulation uncertainty. An arbitrary fitting
weight or a covariance evaluated at a potentially inconsistent estimate is not
a substitute. This first interface does not concentrate nuisance parameters;
all parameters entering a point-null prediction must be specified.

`valid` means numerically evaluated. Nonfinite model moments and singular or
numerically singular covariance produce unresolved results with `accepted=None`.
No pseudoinverse or ridge silently changes the test. Malformed labels, shapes
or non-PSD matrices raise errors. Report exports retain the status.

## Continuous inversion for a declared scalar curve

`QuadraticMomentCurve` supports

\[
m(\theta)=a+b(\ell z+c z^2),\qquad z=(\theta-\text{center})/\text{scale}.
\]

Here a and b are q-vectors, while the common index has scalar coefficients
`linear` and `quadratic`. This is a limited class, not an arbitrary vector of
quadratic functions. Covariance must be constant across the candidate parameter.
The interface solves for the accepted index interval and pulls it back through
the declared curve, preserving every numerically resolved component.

```python
curve = QuadraticMomentCurve(
    [0.0], [1.0], targets.labels, "gaussian-theta-squared-v1",
    linear=0.0, quadratic=1.0,
)
confidence = invert_quadratic_moments(curve, targets, calibration="gaussian_known")
assert confidence.resolved and len(confidence.intervals) == 2
assert confidence.contains(-1.0) and confidence.contains(1.0)
assert not confidence.contains(0.0)
print(confidence.summary())
print(confidence.to_frame())

clipped = invert_quadratic_moments(
    curve, targets, domain=(0.0, 1.0), calibration="gaussian_known",
)
assert clipped.truncated and clipped.boundary_contact == (False, True)
```

The example's set has two separate components near -1 and +1. It must not
be replaced by the interval between its outermost endpoints. The solver also
retains singleton tangencies, empty sets and the entire real line. An exactly
zero loading does not imply a narrow interval: the maintained moment test may
accept every parameter or reject all of them. Small nonzero loadings are never
silently set to zero using an identification threshold.

`global_intervals` records the analytic inverse image; `intervals` intersects
it with the declared `domain`, a search/display window. `global_geometry`
distinguishes bounded, unbounded/all-real, empty and unresolved cases.
`truncated` indicates that some global set lies outside the window, even if
the displayed set is empty. `boundary_contact` describes accepted finite
window endpoints. Unboundedness follows from the declared curve, not merely
from reaching a grid boundary. Coverage inside a window requires the true
parameter to lie in that prespecified window.

`resolved` concerns floating-point geometry. Ambiguous near-tangencies,
unrepresentable roots and failed endpoint residual checks remain unresolved;
`contains()` then raises an error. It does not turn numerical uncertainty into
an empty set. `summary()` reports status separately from component rows.
Markdown, LaTeX and Typst exports include status even for empty/unresolved sets.

## General models: sampled hypotheses remain sampled

`evaluate_moment_grid` accepts a callback returning `MomentTestResult`. It can
use a different covariance at every candidate and retain invalid evaluations.
It intentionally provides no continuous-completeness claim.

```python
test_at = lambda theta: moment_restriction_test(
    targets, curve(theta), calibration="gaussian_known",
)
sampled = evaluate_moment_grid(test_at, [-1.5, 0.0, 1.5])
assert not sampled.accepted.any()
assert not sampled.continuous_complete
assert len(sampled.unresolved_cells()) == 2
assert len(confidence.intervals) == 2
```

All three sampled points reject, although the continuous set has two accepted
islands. `accepted_runs()` therefore reports **sampled node spans**, not
confidence-interval endpoints. Every between-node cell remains unresolved,
including cells whose two endpoints reject. Invalid nodes carry `accepted=None`
in the table and separate accepted runs. Unexpected callback exceptions
propagate instead of becoming rejection penalties. All nodes must share alpha,
moment labels, q and calibration.

## Simulation uncertainty at the hypothesized parameter

The simulation adapter uses the actual joint batch at the candidate, including
its counts/statuses and full simulated-mean covariance. Assert independence from
data or supply cross-covariance as in the SMM interface.

```python
empirical = MomentTargets([0.3], [[0.02]], ("mean",))
plan = SimulationPlan(seed=618, replications=64, periods=20)

def simulator(theta, rng, budget):
    return [theta[0] + (1 + theta[0]**2) * rng.normal(size=budget.periods).mean()]

def simulated_test_at(theta):
    batch = simulate_moments(
        simulator, [theta], plan, labels=empirical.labels,
        definition_id="heteroskedastic-normal-mean-v1",
    )
    return simulated_moment_test(empirical, batch, assume_independent=True)

simulation_grid = evaluate_moment_grid(simulated_test_at, np.linspace(-0.5, 1.0, 31))
assert simulation_grid.valid.all()
assert simulation_grid.tests[0].calibration == "asymptotic"
print(simulation_grid.accepted_runs())
```

The total covariance is recomputed at each hypothesis; the example deliberately
changes simulation variance with the parameter. A fixed SMM pilot weight is
never reused as the test covariance. `simulated_moment_test` offers only
asymptotic calibration because covariance estimated from a finite number of
replications does not yield an exact chi-square pivot. Invalid batches remain
attached to unresolved results. Burn-in, finite-horizon-statistic and
discretization biases still require separate control.

## Interpretation and validation limits

The null maintains both the parameter and the moment restrictions. Empty sets
can reflect incompatible moments; a small set alone does not establish precise
identification. There is no automatic weak-identification diagnostic, arbitrary
nuisance profiling, conditional QLR test, singularity-robust inference or
continuous inversion of arbitrary callbacks in this first slice. In
`puremacro.inference.weak_iv`, `kleibergen_paap_f` computes the reduced-rank
Kleibergen–Paap (2006) `rk` Wald $F$ statistic for $k \ge 1$ endogenous
regressors, and `anderson_rubin_band` returns `(nan, nan)` with a
`RuntimeWarning` when the Anderson–Rubin acceptance set is empty on the supplied
grid instead of returning the grid endpoints. In `puremacro.narrative.validate`,
`weak_iv_summary` partials out the intercept and any `controls` from `endog`,
`outcome` and `Z` before evaluating Cragg–Donald, Kleibergen–Paap and
Anderson–Rubin diagnostics.

Run `python -m puremacro.examples.identification_robust_sets --output /tmp/weak-id`
for component tables, a missed-island demonstration, a simulation grid, report
exports, a figure and JSON provenance. Independent tests check Gaussian
quadratic forms, analytic endpoints, small nonzero loadings, tangencies,
truncation and invalid evaluations. The local acceptance study uses five
known-covariance Gaussian designs with 2,000 trials each, including strong,
weak, absent and turning-point identification. These references do not certify
arbitrary economic models or estimated-covariance finite-sample coverage.
