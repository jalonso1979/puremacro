# Simulation-aware SMM

`puremacro.structural.fit_simulated_moments` fits simulated moments while
accounting for both empirical sampling error and Monte Carlo error. This first
interface targets locally smooth moments with a fixed simulation budget. It
extends the [empirical-to-structural bridge](structural_bridge.md) and leaves
existing VFI, DMP and sovereign estimators unchanged.

## A reproducible fit

The empirical covariance is **Cov(the empirical estimate)**, already on its
sampling scale. The example below uses a synthetic observed mean with known
variance `1/1000`. Each simulation replication is a mean of ten independent
normal observations; 32 such means form the simulated estimator.

```python
from dataclasses import replace
import numpy as np
from puremacro.structural import (
    MomentReplication, MomentTargets, SimulationPlan,
    fit_simulated_moments, simulate_moments, moment_error_covariance,
)

targets = MomentTargets([0.4], [[1/1000]], ("mean",), units="level")
plan = SimulationPlan(seed=20261108, replications=32, periods=10)

def simulator(theta, rng, budget):
    values = theta[0] + rng.normal(size=budget.periods)
    return MomentReplication([values.mean()], counts=[budget.periods])

fit = fit_simulated_moments(
    simulator, targets, [0.0], plan=plan,
    definition_id="iid-normal-mean-v1", parameter_names=("mu",),
    jac=lambda theta: [[1.0]],
    assume_independent=True, assume_smooth=True,
    validation_plan=replace(plan, role="validation", replications=128),
)
assert fit.success and fit.inference_valid, fit.inference_reasons
print(fit.summary())
print(fit.moment_fit())
```

`success` describes the optimizer. `inference_valid` additionally requires
passing local numerical and derivative checks, conditional on the statistical
assumptions below. `summary()`, `to_frame()`, `to_dataframe()`, `to_markdown()`,
`to_latex()` and `to_typst()` expose the same parameter estimates and intervals.
`moment_fit(validation=True)` reports a separate descriptive comparison.

## Two different covariance conventions

For independent joint simulation vectors \(s_1,\ldots,s_R\), the simulated
mean is \(\bar s=R^{-1}\sum_r s_r\), with estimated covariance

\[
V_s=\frac{1}{R(R-1)}\sum_r(s_r-\bar s)(s_r-\bar s)' .
\]

There is **no extra division by periods**: the variability of each path's
statistic is already inside its replicate vector. Temporal dependence belongs
inside each simulated path; the outer replications must be independent.
By contrast, `MomentTargets.from_draws` estimates the covariance of an empirical
estimator from bootstrap/sampling draws and does **not** divide by their count.

```python
batch = simulate_moments(
    simulator, fit.theta, plan, labels=targets.labels, units=targets.units,
    definition_id="iid-normal-mean-v1",
)
np.testing.assert_array_equal(batch.replicates, fit.simulation.replicates)
np.testing.assert_allclose(batch.covariance,
                           batch.replicate_covariance / plan.replications)
total = moment_error_covariance(targets, batch, assume_independent=True)
np.testing.assert_allclose(total, targets.covariance + batch.covariance)
```

If data and simulation share randomness, supply
`cross_covariance=C`, where \(C=\operatorname{Cov}(\hat m,\bar s)\), instead of
asserting independence. The moment-error covariance is
\(V_e+V_s-C-C'\). The full joint covariance must be positive semidefinite;
labels and units must match exactly. Different seeds alone do not establish
independence if a callback reuses observed data. A supplied constant `C` must
be appropriate at the relevant parameter and for the fit replication budget.

## Fixed weights and separate variance contributions

The objective is \((\bar s(\theta)-\hat m)'W(\bar s(\theta)-\hat m)\).
The default fixes \(W=V_e^{-1}\). An explicit positive definite `weight` is
supported. `pilot_plan` instead estimates a weight at `theta0`, using the
pilot's replicate covariance divided by the **fit** replication count. Pilot
and fit periods/burn-in must agree. Weight and simulation effort stay fixed
throughout optimization. No ridge or eigenvalue clipping is applied.

```python
pilot_fit = fit_simulated_moments(
    simulator, targets, [0.0], plan=plan,
    pilot_plan=replace(plan, role="pilot", replications=64),
    definition_id="iid-normal-mean-v1", assume_independent=True,
    assume_smooth=True, jac=lambda theta: [[1.0]],
)
assert pilot_fit.inference_valid
np.testing.assert_allclose(
    pilot_fit.covariance,
    pilot_fit.sampling_covariance + pilot_fit.simulation_covariance
    + pilot_fit.cross_covariance_adjustment,
)
```

With \(G=\partial\bar s/\partial\theta'\) and
\(A=(G'WG)^{-1}G'W\), the local covariance is
\(A(V_e+V_s-C-C')A'\). The result retains its sampling, simulation and cross
contributions separately. `moment_covariance` is in moment units; these three
contributions and `covariance` are in parameter units. Final simulation
covariance is evaluated at the estimate, even when weights came from a pilot.

## Randomness and the callback contract

- `simulator(theta_copy, rng, plan)` returns a joint vector or
  `MomentReplication`. Use the supplied generator for **all** random choices.
  Implement `periods` retained observations and the stated `burn_in` explicitly.
- Each replication receives a fresh PCG64 generator. Its SeedSequence key is
  `[replication, stream, role_code, seed]`, with fit/pilot/validation codes
  0/1/2. The same plan replays streams regardless of evaluation order. Increasing
  replication count preserves existing streams. Roles separate random streams,
  not moment definitions; use the same statistic in pilot, fit and validation.
- `plan_id` hashes plan settings, stream schema, generator and NumPy version.
  It does not hash model source, guarantee cross-environment bit identity, or
  establish an economic timing convention. Preserve source and environment too.
- `definition_id` names the caller's measurement convention. Align conditioning,
  lag timing, transformations, sample windows, units and branch selection with
  the empirical targets. IDs and labels do not prove this alignment.
- Missing conditional observations should produce zero counts/nonfinite moments,
  not fabricated zeros. Nonfinite values, zero counts or status other than `ok`
  invalidate the entire joint batch. All rows remain available in `replicates`,
  `counts` and `statuses`; absent counts mean unknown. There is no filtering of
  surviving draws. Callback programming errors propagate unchanged.

`SimulationMomentError` retains `.batch` and, for replay failures, an available
`.reference`. Fitting checks exact replay at the start and estimate; pilot and
validation batches are replayed too. A bounded cache may reuse identical
parameter evaluations. `n_evals` counts actual batches, including checks and
pilot/validation; `n_replications_evaluated` sums their replication budgets.

## Inference scope and validation

Ordinary normal intervals require explicit `assume_smooth=True`. The estimator
withholds them for optimizer failure, nonstationarity, boundary estimates,
rank deficiency, excessive conditioning, singular moment-error covariance,
nonfinite covariance or inconsistent derivative probes. Supplied `jac` must
differentiate the **fixed-plan simulated mean**. Two finite-difference radii
and left/right slopes check consistency using common random numbers.
`derivative_step` controls the initial parameter-scaled radius and
`derivative_tol` the normalized discrepancy tolerance.

Passing these checks is not proof of smoothness, identification or statistical
coverage. Inference also assumes correct specification, regular interior
identification, consistent covariance/Jacobian estimates, and negligible
initialization, finite-horizon-statistic and discretization bias. Large Monte
Carlo error can make a local approximation poor. Discrete policies, branch
changes, weak identification and misspecification need additional methods.
There is no automatic J p-value or weak-identification-robust confidence set.

Fresh validation streams are descriptive and never pooled into the fitted
objective. Since the estimate depends on fit streams, the fit/validation
difference is not automatically a standard two-independent-samples test.
Changing a budget requires a new explicit plan and fit. The generic interface
does not audit the legacy sovereign wrapper's timing or conditional moments.

Run `python -m puremacro.examples.simulation_aware_smm --output /tmp/smm` for
CSV, Markdown, LaTeX, Typst and JSON records. The example checks stationary
AR(1) raw moments/finite-sample mean variance and matched two-state Markov
moments. Regression tests independently check full AR(1) joint covariance via
Gaussian product identities and correlated GLS covariance algebra.

Background: [Duffie and Singleton (1993)](https://web.stanford.edu/~duffie/jstorlinks/ds1.pdf)
on simulated moments; [NumPy's random-stream documentation](https://numpy.org/doc/stable/reference/random/parallel.html)
for deterministic stream identifiers. These references do not certify arbitrary
user simulation callbacks or finite-sample inference.
