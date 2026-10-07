> 🇬🇧 English · 🇪🇸 [Español](es/trade_continuation.md)

# Parameter continuation and multistart for policy equilibria

`puremacro.trade.continuation` adds two small helpers on top of
[`solve_policy_equilibrium`](trade_policy.md): an audited path in a model
parameter at a fixed, complete tariff policy, and a multistart at one exact
target. Both accept a stage only after the returned state has been re-audited
with the consistent-accounting evaluator, and both attach the same
qualification: an equilibrium reached along a path or from an alternative start
is not established as belonging to any other branch, and an unresolved search
proves neither a fold nor nonexistence.

```python
import numpy as np
from puremacro.trade import solve_policy_equilibrium
from puremacro.trade.continuation import sigma_path, try_starts
from puremacro.trade.data import generate_synthetic_mrio, package_mrio_to_calibration_result
from puremacro.trade._oecd_icio import condense_final_demand
from dataclasses import replace

# Three countries, three sectors, OECD final-use layout condensed to C/I/Cx.
raw = generate_synthetic_mrio("oecd", custom_c=3, custom_s=3, seed=2)
raw = replace(raw, taxes_less_subsidies_fd=np.zeros(raw.C*raw.K_F))
calibration = package_mrio_to_calibration_result(condense_final_demand(raw))

# A 10% import tariff by the first country on every foreign origin and use.
nc, ns, nfd = calibration.nc, calibration.ns, calibration.n_final_demand
tau, tau_fd = np.ones((ns*nc, ns, nc)), np.ones((ns*nc, nfd, nc))
tau[ns:, :, 0], tau_fd[ns:, :, 0] = 1.1, 1.1

# Follow the intermediate sourcing elasticity from Leontief (0) to 2 at that policy.
path = sigma_path(calibration, tau, tau_fd, 2., tol=1e-9)
print(path.summary())
print(path.to_dataframe()[["fraction", "sigma", "method", "iterations", "audited_max_residual"]])

# The endpoint is an ordinary consistent-accounting equilibrium.
direct = solve_policy_equilibrium(calibration, tau, tau_fd, sigma=2., tol=1e-9)
print(np.max(np.abs(path.final.x_sol - direct.x_sol)))  # 3.1e-11 on 2026-09-22

# Several warm starts at one exact target; the first audited root wins.
starts = [{"name": "calibrated", "x0": None},
          {"name": "from_path", "x0": path.final, "provenance": {"path": "sigma 0 -> 2"}}]
multi = try_starts(calibration, dict(tau=tau, tau_fd=tau_fd, sigma=2., tol=1e-9), starts)
print(multi.to_dataframe())
```

Seed 2 is used because the default synthetic seed produces a negative
expenditure share after condensation, which consistent accounting rejects.
The example condenses the six synthetic final uses with
`condense_final_demand` from the internal module `puremacro.trade._oecd_icio`
(not part of the public API, so it may change without deprecation), and it
passes zero final-use taxes because that helper does not accept a table
without them.

## What the helpers compute

`continue_parameter(solve, path, start, ...)` is the generic engine. `path(f)`
returns the solver keyword arguments at path fraction `f` in `[0, 1]`, for
example `{"tau": tau, "tau_fd": tau_fd, "sigma": 2*f, "tol": 1e-9}`. `solve`
is either a `TradeCalibrationResult`, in which case every trial calls
`solve_policy_equilibrium` with those arguments, `method="newton"` and the
previous accepted state as warm start (see the next section), or a callable
`solve(kwargs, x0)` returning a consistent-accounting `TradeEquilibriumResult`
(then `calib=` is required for the audit). `start` must already solve the
equations at `path(start_fraction)`: it is audited, never re-solved, so a saved
stage is resumed at its actual fraction.

`sigma_path(calib, tau, tau_fd, sigma_target, sigma_start=0., ...)` is the
convenience wrapper for the intermediate sourcing elasticity. The path is

```
sigma(f) = sigma_start + f (sigma_target - sigma_start),   f in [0, 1]
```

with the exact endpoints at `f = 0` and `f = 1`. `f` never means a tariff
fraction: the intermediate and final tariff schedules are complete and fixed
along the path. A supplied `start` is audited at `sigma(start_fraction)`
without re-solving (and `x0` is then refused). When `start` is omitted it is
solved at `sigma(start_fraction)` from `x0` with the full audited ladder of
`solve_policy_equilibrium` (`method="auto"` unless you name a method): the
start is audited like a supplied one, not followed, so the Newton-only rule
below applies to the trials only. The first stage records the method, seed and
fallback use of that solve, and `metadata["start_source"]` says `"solved"` or
`"supplied"`. A failed start solve raises `PolicyEquilibriumError` naming the
fraction, before any stage exists. When `sigma_start == sigma_target` the path
only audits the start.

`try_starts(solve, target, starts, max_iter=35)` tries named starts at the
unchanged target arguments. Each start is a mapping with a `name`, an `x0`
(an equilibrium vector, a `TradeEquilibriumResult`, or `None` for the
calibrated state) and optional `provenance`, stored verbatim. A start whose
vector is not finite or does not have the calibration's state length is
recorded as unresolved without a solve. The search stops at the first audited
root; a rejected start is never substituted for a root, and an unresolved
search returns `final=None`.

## Warm starts and `allow_fallback`

As in the IO helpers, every trial is by default a Newton solve from the
previous accepted state (in `try_starts`, from the named start), and any other
outcome is a rejected trial that halves the step. With the default
`allow_fallback=False`:

- the calibration adapter calls `solve_policy_equilibrium` with
  `method="newton"`, and refuses `method="auto"`, `"hybr"` and `"keller_pac"`
  with a `ValueError` before any solve, because `"auto"` runs the fallback
  ladder, `"hybr"` starts from the calibrated state and `"keller_pac"` starts
  from the zero-tariff schedule, so none of them follows the path;
- a returned state whose solver record shows the fallback ladder
  (`metadata["policy_solver_fallback_used"]`), or whose last recorded attempt
  was seeded from something other than the supplied start, is a rejected
  trial. This also applies to states returned by a custom `solve` callable.

`allow_fallback=True` opts into the ladder: the adapter then defaults to
`method="auto"`, and an accepted stage may be an audited root reached by
`hybr` from the calibrated state or by Keller tariff continuation from the
zero-tariff schedule. Such a stage solves the equations at the requested
parameters but was not reached by following the path. Every stage and every
`try_starts` attempt records `seed` (`"warm"` or `"calibrated"`) and
`fallback_used`, so a root credited to a named start whose vector was never
used is visible in the record.

## Step law

The stepping follows the IO helper `continue_origin`. With current fraction
`f` and step `h`:

```
trial     = min(1, f + h)
failure   -> h <- h / 2; if h < min_step: stop, keep the last audited state
success   -> f <- trial; if iterations <= fast_iterations: h <- min(max_step, growth h)
```

Defaults: `initial_step = 0.25`, `min_step = 1/1024`, `max_step = 0.5`,
`growth = 1.5`, `fast_iterations = 6`, the `continue_origin` values; the IO
`continue_ces` used a cap of 0.4, five fast iterations and `min_step = 1/128`,
which you can pass. `iterations` is the solver's own count: Newton iterations,
SciPy `hybr` function evaluations, or Keller continuation steps. Two
differences from the IO loop are deliberate. A trial within `1e-12` of
fraction one is taken as one, which avoids a duplicated final stage from
floating-point accumulation. The IO helpers also rejected every trial that
needed more than 25 Newton iterations (`continue_ces`: 12); here the per-trial
cap is the solver's `max_iter` (100 in `solve_policy_equilibrium`), so pass
`max_iter=25` to reproduce that rule. A stop raises
`ParameterContinuationFailure`, whose `state` is the last audited equilibrium
(never the rejected trial) and whose `result` is the partial
`ParameterContinuationResult` with `status="unresolved"`, every accepted stage
and every rejected trial.

## Acceptance and invariants

A trial is accepted only when the returned state

1. declares convergence,
2. carries consistent accounting, lump-sum transfers, and exactly the
   requested `sigma` and intermediate and final tariff schedules (by array
   equality),
3. passes the consistent-accounting re-evaluation of every equilibrium and
   accounting equation, including the omitted goods market and the foreign
   balances, at its recorded tolerance, with every reported field (prices,
   outputs, wages, rental rates, transfers, price indices, final demand,
   tariff revenue and GDP) equal to the recomputed equilibrium to
   `rtol = atol = 1e-8`,
4. has an audited maximum absolute residual at most the requested `tol`
   (default `1e-8`, the default of `solve_policy_equilibrium`; absolute in the
   calibration's units), and
5. unless `allow_fallback=True`, was neither reached through the solver's
   fallback ladder nor, according to its recorded last attempt, solved from a
   seed other than the warm start.

The audit is the model's own evaluator, the same one `solve_policy_equilibrium`
uses; it is not a separately derived oracle. A state that fails criterion 2 is
a broken contract and raises `ValueError` immediately; a state that fails any
other criterion is a rejected trial. `ValueError` or `TypeError` raised by the
solver mean invalid inputs and propagate; `RuntimeError` (for example
`PolicyEquilibriumError`), `ArithmeticError` and `LinAlgError` are numerical
failures that halve the step. The equilibria inherit the numeraire and
foreign-balance closure of `solve_policy_equilibrium` (see
[consistent accounting](trade_accounting.md)). On the synthetic table
(observed on 2026-09-23, not pinned by a test), rescaling every flow by 1000
with `tol` scaled alike, or permuting countries and sectors while keeping the
first country and sector in place, reproduced the path to about `1e-14`;
moving another country into first place, and so changing the numeraire,
changed outputs by about 1% and wages by 0.15.

By default every keyword argument that `path` leaves unchanged between
fractions 0 and 1 is fingerprinted (tariff schedules by their bytes, scalars by
value, equilibrium states by their digest) and must be identical at every
trial; `continued=` names the arguments allowed to change explicitly and
`invariant=` replaces the fingerprint. A violation raises before the trial is
solved. The fingerprint digest is stored as `invariant_id`. An identity path
(nothing continued) still audits the supplied state and reports fraction one.

## Results

`ParameterContinuationResult` holds `fractions`, `parameter_values`, the
JSON-serializable `stages` and `failures` (no state vectors), `final`, `status`,
`start_fraction`, `last_fraction`, `invariant_id`, `qualification` and
`metadata`. Each stage records the parameter, the method, the seed, whether
the fallback ladder was used, iterations, the solver's residual, the audited
residual, the requested tolerance, seconds, the step used, and minimum price,
output ratio, wage and final expenditure. Each rejected trial records the
error and, when the solver raised `PolicyEquilibriumError`, `solver_attempts`:
the method, seed, convergence flag, residuals and error of every attempt; the
message of `ParameterContinuationFailure` ends with the last of them.
`keep_states=True` retains every accepted `TradeEquilibriumResult` in `states`;
`stage_callback(state, stage)` receives each accepted state with an
independent copy of its record so a caller can persist it; `progress(event)`
receives trial, accepted and rejected events. `metadata=` is merged into the
result's metadata, and keys that the result records itself (`method`,
`solver`, `audit`, `sigma_target`, ...) are refused with a `ValueError`.
`to_dataframe()`, `to_markdown()`, `to_latex()`, `to_typst()` and `summary()`
follow the package conventions, and `plot(values, ax=...)` draws a scalar per
stage against the parameter: the audited residual by default, a stage key, a
sequence, or a callable applied to the retained states (for example a country's
Hicksian EV from `compute_hicksian_welfare`).

`DirectTargetResult` holds `attempts` (name, provenance, status, error or
solver diagnostics including `seed` and `fallback_used`), `accepted_start`,
`final`, `status`, the fingerprinted `target`, `qualification` and `metadata`,
with the same table exporters.

`final` is an ordinary consistent-accounting `TradeEquilibriumResult`:
[`compute_hicksian_welfare`](trade_welfare.md) and the post-processing
functions accept it unchanged.

## What is and is not validated

Tests in `tests/test_trade_continuation.py` check, on the synthetic 3 x 3
table above (`tol=1e-9`): the sigma path endpoint against a direct
`solve_policy_equilibrium(sigma=2)` (maximum difference in the state vector
`3.1e-11`, tolerance `1e-9`; accepted fractions 0, 0.25, 0.625, 1, every trial
a warm-started Newton solve); resume from a saved stage (`1.2e-11`); failure
retention of the start, with the solver attempts recorded; invariant
violations and parameter contract violations raised before or without any
solve; the identity path; settings, method and metadata validation; the
`allow_fallback` contract (fallback and calibrated-seed states rejected by
default and accepted and labelled with the opt-in); `try_starts` returning
only audited roots; JSON records, pickling, tables and plots; and the step law
under scripted solver outcomes. A parity test runs the IO `continue_origin`
and `try_direct_target` in a subprocess (skipped when the research volume is
absent) under the same scripted outcomes, including a stage at exactly
`fast_iterations` and a step capped at 0.5, and requires identical accepted
fractions, rejected trials, statuses and last fractions.

On the bundled 3-region x 3-sector OECD fixture in million USD,
`sigma_path(calib, tau, tau_fd, 0.5, tol=1e-5)` with a uniform 10% USA import
tariff agrees with `solve_policy_equilibrium(calib, tau, tau_fd, sigma=0.5,
tol=1e-5)` to a relative `6.6e-12` in prices and `6.7e-12` in GDP
(2026-09-23, after the consistent-mode solver fixes; the test requires
`1e-6`). Every stage, including the start at sigma 0, is a Newton solve (the
start takes 6 iterations to `4.8e-8`). Before those fixes plain Newton stopped
at sigma 0 with `converged=False` at a residual of `8.3e-6` because its
stopping rule did not test the audited physical equations, and the start came
from the Keller fallback. The path from 0 to 2 is **unresolved** on this
fixture: it stops at fraction 0.450 (sigma 0.900), and the path from 2 down to
0 stops at fraction 0.479 (sigma 1.041). Direct `solve_policy_equilibrium`
calls at sigma 0.91, 0.95, 1.0, 1.01 and 1.04 fail with Newton, hybrid and
Keller continuation; at sigma 1, 1.01 and 1.04 the Keller attempt reaches
residuals of `4.9e-8`, `9.9e-6` and `1.0e-7`, but at a root with negative
final expenditure, which the audit rejects as demand-infeasible. Sigma 0.9 is
solved by Newton and 1.05 through the Keller fallback. The tool reports the
region as audited stages plus a `ParameterContinuationFailure`; it does not
resolve it. This is an observation
about the existing solver stack near unit elasticity on this table and
tolerance, not a property of the path algorithm; a slow-marked test pins the
contract (audited stages, audited retained state) rather than the failure.

The bundled 77-country x 11-sector table cannot be used with these helpers:
its calibration has three negative investment cells, which consistent
accounting rejects before any solve (a test pins the `ValueError`). The
[accounting documentation](trade_accounting.md) explains why a negative signed
inventory aggregate needs aggregation or an explicit inventory model.

The IO helpers were written to recover low-elasticity GTAP equilibria and did
not: all eleven `recovery.json` records under
`retaliation_2026-09-22/results_latest/cases/*/runs/gtap/low/` are
`unresolved`, with the origin homotopy stopping between fractions 0.92 and 0.99
and every direct start rejected. The IO qualification reads as follows;
`result.qualification` attaches a version adapted to parameter paths:

> An admissible exact-target equilibrium found from an alternative start or
> production-elasticity path is not established as belonging to the forward
> low-elasticity tariff branch. Local stability is reported separately;
> neither convergence nor unsuccessful searches prove global uniqueness or
> nonexistence.

and, on early rejection of slow trials, "This can reject a slowly converging
trial; it does not diagnose a fold or prove nonexistence." puremacro does not
implement that early-rejection guard: a trial is rejected only when the solver
raises, the solver record shows a seed other than the warm start, or the audit
fails. Local stability of accepted states is not assessed by this module.

## Related documentation

- [Hicksian tariff policy and solver recovery](trade_policy.md): the audited
  solver each stage calls (Newton from the warm start by default; the full
  Newton, hybrid and Keller ladder with `allow_fallback=True`).
- [Consistent trade accounting](trade_accounting.md): the equations that the
  stage audit re-evaluates, and the numeraire and foreign-balance closure.
- [Hicksian consumption welfare](trade_welfare.md): welfare along a path from
  the retained states.
- [Structural validation status](STRUCTURAL_VALIDATION_STATUS.md).
