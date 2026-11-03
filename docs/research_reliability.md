# Research reliability and the next PureMacro milestones

The agreed order is **research reliability and replications**, then **economic
models and policy applications**, then **teaching and usability**.

## Published corrections and current development

Version **6.0.1** was published to PyPI on 2026-10-29 and includes the
corrections below. Version **6.1.0** is under development; additions marked
unreleased require a source checkout. The 5.8.0, 5.9.0 and 6.0.0 source tags
were not published. See the [current roadmap and API policy](1.0_path.md).

Native model adapters now preserve the supplied equations and loadings.
First-order DSGE scores use Kalman sensitivities; DAG scores centrally difference
the native Whittle likelihood while holding the supplied steady state fixed.
Unsupported native operations raise errors. Explicit `model=None` demonstrations
warn about their scope. See [adapter details](autodiff_and_protocol.md).

The default nonlinear HANK DAG uses prescribed household responses. Its
aggregate-risk example uses prescribed moments and matrices. They are explicitly
demonstrations; a supplied model cannot be silently ignored. The legacy
`asymmetry_ratio` reports nonlinear/linear peak amplification, not two solved
opposite-shock paths. See [HANK scope](hank_ssj_and_aggregate_risk.md).

Sovereign corrections include deterministic default, stationary-mean default
costs, zero-debt reentry and unadjusted simulation statistics. A constant spread
has an undefined correlation, reported as `NaN` with a status field.

## Tests to run first

From the repository root, in a Python 3.11+ environment with `.[dev,notebooks]`:

```sh
python -m pytest tests/test_protocol_model_fidelity.py tests/test_research_reliability_references.py tests/test_sovereign/test_reference_reliability.py -q
python tools/release_check.py --no-tests
```

For an interactive tour, run notebook **72** in English or Spanish. Notebook
**73** explores sovereign debt; **74** covers synthetic distributional LPs,
Gaussian inference and reduced transitions. Their numerical scorecards have
the specific scope described in each notebook, not blanket economic validation.

## Independent evidence so far

| Comparison | Result on the tested grids | What remains open |
|---|---|---|
| Arellano short-term deterministic equations | Matched-grid equations pass; archived Fortran and MATLAB moments reproduce; public simulation timing and zero-debt reentry are corrected; public-calibration grids converge at 6,401 × 641, and the public solver reaches that grid with a certified monotone search | Published-moment discrepancies and extension-specific economic convergence |
| One-asset household block and published HANK equilibrium | Household policies and consumption match the SSJ authors' code within 5e-11; return Jacobians within 4e-4. The complete published equilibrium, both nonlinear shock signs and an extended asset/income/horizon ladder are independently checked; see the [HANK evidence](hank_equilibrium.md) | Household incidence with observed data and general higher-order aggregate risk |

The original equation/block references are frozen with hashes and provenance in
`tests/fixtures/research_reliability/`; the linked study guides locate the
additional equilibrium and grid records. Regeneration tools live in
`tools/reference_validation/`; the external SSJ package is not a runtime
dependency. Matched-grid agreement does not prove discretization accuracy.

## Sovereign study: baseline completed, public grids converged

The [original-code and published-moment comparison](arellano_replication.md)
now includes 16 independent simulations per case, nested debt grids and 21 to
321 income states. At the archived calibration, refinement moves the moments
far more than simulation uncertainty, and spread volatility halves. For the
public calibration, a pre-registered paired-seed criterion establishes grid
convergence at 6,401 debt nodes and 641 income states. A small Bellman residual
alone would not have justified that conclusion.

1. Public reentry timing, zero-debt initialization and measurement are now
   documented and tested. Keep the archive and public-calibration results
   distinct; they use different economic conventions.
2. Done for the public calibration: grids converge at 6,401 × 641 under the
   registered criterion, with sampling uncertainty kept separate from
   discretization error and every raw moment retained.
   `solve_sovereign_default(policy_search="monotone")` solves that grid,
   certified by an exhaustive check, and matches the reference arrays.
3. Study how income volatility, persistence and market-reentry probability
   change sustainable debt and default exposure in the validated short-term
   model. Report welfare and numerical uncertainty before adding maturity or
   restructuring mechanisms that require additional validation.

This creates a research result that can be reviewed from equations through
policy tables. It also supplies the model and output contracts for future work.

The [sovereign extension references](sovereign_extensions_validation.md) now
check the stated long-term smoothed operator, haircut/maturity states and
collateral-planner policies independently. Long-term simulation draws from the
same debt-choice distribution used in pricing. Collateral diagnostics are
measured from simulated paths, with explicit undefined amplification and no
prescribed fallback. The survival-spread residual is a compounding adjustment.
These are small-grid equation checks, not published extension-moment replications.

## Then: household incidence and teaching

The [published one-asset HANK equilibrium](hank_equilibrium.md) now has matched
fiscal rules, return timing and market clearing, independent nonlinear paths
for both shock signs, and a passing extended grid/horizon study. Next, link
household incidence to distributional LPs on observed data. A general
higher-order aggregate-risk solver remains a separate later task.

Use the validated study to build a short bilingual path: install a verified
wheel, reproduce one baseline, change one policy assumption, and inspect a
diagnostic table. Keep demonstrations visibly separate from empirical results.
