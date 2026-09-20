# Hicksian tariff policy and audited solver recovery

20 September 2026. Python 3.14.6, NumPy 2.5.2, SciPy 1.18.1; existing local
environment. Builds and tests use the NumPy backend.

## Delivered behavior

The explicit `metric="hicksian_ev"` path connects expenditure-function welfare
to unilateral tariff optimization, multilateral best responses and fixed-action
payoff matrices. Each search uses one audited zero-tariff baseline, one declared
consumption basket and consistent producer/purchaser accounting. Historical
defaults and utility aliases retain their previous meanings.

Level-valued objectives are monetary EV. Baseline EV is zero; gains and regret
therefore use fixed baseline selected consumption expenditure as denominator.
Reported gains are percentages; relative regret is a fraction. Bloc objectives
sum monetary EV without overlapping players. World welfare includes nonplayers.
Terms-of-trade diagnostics are unavailable for multi-country blocs rather than
substituting the first member's index.

Every positive matrix action is fixed across all four cells. Supplied rates,
including the historical `nash_*` arguments, carry no continuous-game Nash
verification. Conflicting action rates are rejected. The English/Spanish
tutorial's computed matrix does **not** satisfy the Prisoner's Dilemma conditions;
the notebook reports that outcome.

`solve_policy_equilibrium` tries Newton, hybrid and then full Keller continuation.
Recovery resets failed seeds and preserves complete intermediate/final tariff
schedules, technology, accounting closure and tolerance. Acceptance requires
solver convergence, matching requested schedules and a fresh audit of reported
fields and all physical/accounting equations. Metadata records each attempt.
Exhausted recovery raises `PolicyEquilibriumError` with attempts and, in policy
searches, the failed tariff profile. Failed candidates supply no payoff, enter
no cache and never seed the next solve.

Nash convergence requires final simultaneous best-response gaps and normalized
unilateral regret, rather than damped step size. Grid/refinement failures leave
optimization unresolved. A finite, audited candidate can still be returned with
`converged=False` when the outer iteration budget expires.

## Independent reference

[Reference code](../../tools/reference_validation/validate_hicksian_policy.py)
uses a scalar relative-price equation for a two-country CES economy, algebraic
fiscal accounts and origin-level primal expenditure minimization. The oracle
does not call the production GE residual, tariff builder or optimizer. It
evaluates all **1,681 profiles** on a 41-by-41 tariff grid, plus deviations at the
runtime candidate. The calibration has domestic taxes, nonzero foreign balances,
two distinct consumption baskets and a separate investment category.

| Comparison | Recorded result |
|---|---:|
| Unilateral optimum, A and B, on [0, 0.40] | Both at 0.40, matching independent grid |
| Maximum unilateral EV error | 4.64e-12 value units |
| Independent discrete Nash profile | (0.40, 0.40) |
| Runtime Nash tariffs | (0.399998976, 0.399998976) |
| Maximum Nash payoff error | 8.78e-11 value units |
| Independent maximum regret / baseline consumption | 3.08e-7 |
| Final simultaneous tariff gap | 1.024e-6 |
| Fixed-action matrix maximum percentage EV error | 1.51e-10 percentage points |
| Actual Newton-failure case: recovered hybrid residual | 4.62e-13 |
| Recovered producer-price error against scalar reference | 4.91e-14 |

The reference uses intermediate substitution elasticity 2; the actual recovery
case uses elasticity 0.5 and heterogeneous final-use tariffs. Tests separately
force the first two attempts to fail, then execute the real continuation solver.
They also check final-only and intermediate-only policy curves against the
independent scalar reference.

These optima reach the imposed ceiling. They do not identify interior optimal
tariffs, establish global continuous-game uniqueness or constitute empirical
policy estimates. A finite grid plus local refinement can miss unsampled peaks.

## Frozen OECD integration and tolerance scale

The conserving OECD 2023 regular / 2019 aggregation has three regions and three
sectors. This run reloads the frozen fixture, not the full native archive. Its C
aggregate includes household, NPISH and government consumption.

An explicit `ge_tol=1e-5`, in the table's million-USD scale, passes the unilateral
grid and a two-player game iteration. Direct baseline-price valuation of the
single consumption basket agrees with unilateral EV within 5.77e-9 value units.
World percentage EV, including REST as a nonplayer, agrees within 3.11e-14
percentage points. The one-iteration game candidate correctly remains
unconverged; its GE residual is 3.36e-8.

A separate stricter `tol=1e-8` baseline probe rejects all three methods, whose
reported residuals are approximately 2.05e-8 to 2.98e-8. That request produces
no payoff and is never silently relaxed. Absolute GE tolerances must fit the
data's numerical scale. Monetary rescaling may require changing the absolute
GE tolerance; percentage EV and consumption-normalized regret are unit invariant.
This is a multi-sector integration/accounting check, not independent replication
of an empirical optimal-tariff estimate.

## Verification and artifacts

- **241 passed, 9 skipped** in the trade, numerical-hardening, frozen structural
  reference and public-API regression selection. Skips concern unavailable
  optional accelerator backends. One existing private-module deprecation warning
  is recorded. See [regression log](regression-tests.log).
- **31 targeted policy tests passed**, including the independent grid, real and
  injected failures, false convergence, wrong schedules, stale states, currency
  scaling, tiny relaxation, overlapping players, fixed actions and strict supplied
  baseline acceptance. The final run follows the iteration-limit validation
  cleanup. See [policy log](policy-tests.log).
- **100 notebook checks passed, 2 expected failures** remain quarantined historical
  EV/theorem claims; these are not counted as economic validation. See
  [notebook test log](notebook-tests.log).
- Notebook 63 was rebuilt and executed in both languages, with five code cells
  each and no cell errors. Jupyter execution required local-kernel-port permission
  after sandbox denial. The Markdown/LaTeX technical report was updated and its
  PDF compiled twice. `git diff --check` passed.

Machine-readable comparisons: [results.json](results.json).
Reproduction commands: [commands.txt](commands.txt).
Usage and supported options: [trade policy documentation](../../docs/trade_policy.md).

The supported scope is consistent accounting, NumPy, Leontief/CES sourcing and
lump-sum rebates with conditional consumption welfare. Investment, flexible/GPU
models, gradient tariff games and the historical geopolitical scenario wrapper
are not covered by the new metric. The old TOT/Alloc/TariffRec decomposition and
theorem-certification interfaces remain unavailable. No commit or push was made.
