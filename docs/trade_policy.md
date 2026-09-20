# Hicksian tariff policy and audited equilibrium recovery

Select `metric="hicksian_ev"` explicitly in the unilateral tariff search,
multilateral Nash search or welfare payoff matrix. This uses
[consistent accounting](trade_accounting.md), lump-sum fiscal rebates and the
[consumption expenditure function](trade_welfare.md). Historical defaults and
the aliases `ev`, `equivalent_variation` and `consumption` retain their legacy
objectives. They do not select the new calculation.

```python
from puremacro.trade import (
    solve_policy_equilibrium, compute_unilateral_optimal_tariff,
    solve_multilateral_nash_tariffs, compute_welfare_payoff_matrix,
)

# Choose an absolute tolerance appropriate for the calibration units.
ge_tol = 1e-5  # used for the frozen OECD table in million USD
base = solve_policy_equilibrium(calibration, sigma=2., tol=ge_tol)
options = dict(metric="hicksian_ev", base_equilibrium=base, sigma=2.,
               consumption_categories=(0,), ge_tol=ge_tol)
opt = compute_unilateral_optimal_tariff(
    calibration, country_idx="USA", tariff_max=.30, num_grid=21, **options,
)
game = solve_multilateral_nash_tariffs(
    calibration, player_countries=["USA", "CHN"], tariff_max=.30,
    best_response_grid_size=13, tol=1e-4, regret_tol=1e-6, **options,
)
matrix = compute_welfare_payoff_matrix(
    calibration, player_a="USA", player_b="CHN",
    optimal_a=.10, optimal_b=.20, **options,
)
```

The named countries must exist in the calibration. `sigma` is intermediate
origin substitution; the final-use baskets remain Leontief. Consumption
categories default to `(0,)`. Select additional non-investment categories only
when they belong in the stated utility function. This is conditional consumption
welfare, including government consumption when the input data aggregate it.

## One comparison baseline

Each search holds the same zero-tariff equilibrium, preferences, numeraire,
foreign balances and consumption categories fixed for every candidate and
deviation. If `base_equilibrium` is omitted, the search solves it once. A supplied
baseline must pass the requested `ge_tol`, use the requested `sigma` and have
zero intermediate and final tariffs. Changes in domestic taxes are outside
this policy experiment.

Level-valued welfare fields and curves contain EV in calibration value units;
baseline EV is zero. Percentage gains use **baseline selected consumption
expenditure**, never baseline EV. Bloc payoffs sum their members' monetary EV;
strategic blocs must not overlap. World welfare includes all calibrated
countries, including countries that are not strategic players.
Terms-of-trade diagnostics are country-level indices; they are unavailable
(`NaN`) for a multi-country bloc because no external-trade bloc index is defined.

Nash metadata distinguishes `player_regrets` in value units from
`relative_max_regret`, a **fraction** of each player's fixed baseline consumption
expenditure. Thus `regret_tol=1e-6` means one millionth of that expenditure,
not one millionth of EV and not a percentage point. These conventions are
recorded alongside the baseline and solver runs in result metadata.

## What convergence establishes

Unilateral `method="grid"` evaluates the declared grid. `method="bounded"`
also refines every sampled local-maximum bracket, including endpoint brackets.
Nash `method="best_response"` uses this grid/refinement search for each player,
then evaluates simultaneous unilateral deviations at the final profile.
`converged=True` requires both the undamped best-response tariff gap and the
consumption-normalized regret to meet their tolerances. Small damped updates
alone cannot establish convergence. Reaching `max_iter` can return an audited GE
candidate with `converged=False` and finite final deviation diagnostics.

Read `best_response_boundaries`, the tariff ceiling and grid resolution. A
boundary solution is constrained by the imposed interval. Local refinement
can miss unsampled narrow peaks; these checks prove neither global optimality
nor existence or uniqueness of a continuous-game Nash equilibrium.

The payoff matrix uses the same two actions, zero and one fixed positive tariff,
for each player across all four cells. If omitted, the positive action is the
player's unilateral optimum against zero foreign tariffs. Supplied `nash_a` or
`nash_b` are accepted only as fixed actions, explicitly unverified as Nash rates.
Conflicting `optimal_*` and `nash_*` rates raise an error. Mutual positive tariffs
are not automatically a continuous-game Nash equilibrium or a Prisoner's Dilemma.

## Recovery and failed deviations

`solve_policy_equilibrium(method="auto")` tries Newton, SciPy hybrid, then Keller
pseudo-arclength continuation. The first attempt may use a previously accepted
warm start; subsequent attempts restart from the calibrated state. Continuation
starts at zero tariffs and uses the complete target intermediate and final-use
schedules. All attempts preserve `sigma`, the fiscal closure and the requested
tolerance. Recovery does not relax the acceptance threshold.

An accepted result must declare convergence, retain the requested schedules and
technology, and pass an independent re-evaluation of reported fields and all
equilibrium/accounting residuals, including the omitted goods market and foreign
balances. Attempt methods, seeds, warnings, errors and accepted residuals are
recorded in `equilibrium.metadata["policy_solver_attempts"]`.

If all attempts fail, `PolicyEquilibriumError` exposes `attempts`; policy searches
also attach the failing `profile`. The search raises immediately. It produces
no payoff or best-response certificate for that deviation, caches no failed
state and never uses it as the next warm start. Scalar-refinement failure also
leaves optimization unresolved. Use `ge_method`, `ge_max_iter`, `ge_max_steps`
and `ge_tol` to control policy equilibrium solves independently of outer-game
iteration limits. Unknown options are rejected.

`ge_tol` bounds absolute residuals in the calibration's units. It must be
appropriate for their numerical scale: the frozen OECD table in million USD
uses an explicitly requested `ge_tol=1e-5`. A request below floating-point
resolution can fail even at the baseline; recovery deliberately does not
silently loosen that request. Changing monetary units may require changing the
absolute GE tolerance, while percentage welfare and normalized regret remain
unit invariant.

## Evidence and scope

A separately derived two-country CES relative-price equation, algebraic fiscal
accounts and primal expenditure minimization supply the reference. All 1,681
profiles on a 41-by-41 tariff grid are evaluated independently. Unilateral
choices, the constrained Nash candidate, deviations and fixed-action payoffs
agree with that reference. Tests also exercise actual Newton failure with hybrid
recovery, forced continuation, exhausted recovery, false convergence, failed
deviation coverage, currency scaling and tiny relaxation.

See [recorded results](https://github.com/jalonso1979/puremacro/blob/v4.3.0/reviews/2026-09-20-hicksian-policy/REPORT.md) and
[notebook 63](https://github.com/jalonso1979/puremacro/blob/v4.3.0/notebooks/63_trade_wars_and_nash_tariffs.ipynb). The benchmark
reaches its 40% ceiling; it does not identify an interior optimal tariff or
validate an empirical trade-war forecast. Supported instruments are `universal`,
`final_only` and `intermediate_only`. Gradient games, other fiscal closures,
flexible/GPU models and the historical `benchmark_real_world_tariffs` wrapper
are unavailable for this metric. Evaluate explicit schedules with the public
solver and welfare evaluator instead of treating historical scenario labels
as current real-world policy data.
