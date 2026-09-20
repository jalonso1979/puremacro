# Structural-model hardening following the critical evaluation

19 September 2026. This implements the repair priorities in [REVIEW.md](REVIEW.md).
The original review and its observations are retained as a record of the pre-repair
behavior. The changes are in the working tree; no commit, release or publication
was made. Pre-existing work, including the unrelated notebook 59 changes, was preserved.

## What changed

**Trade equations and experiment semantics.** Equilibrium evaluation and reported
flows now share the same economic calculation, including CES intermediate demand,
fiscal closure and capacity options. Keller PAC preserves intermediate and
final-demand schedules, retains adaptive step reductions and checks the residual
at the state it actually returns. Condensed PAC is explicitly unavailable. The
Caliendo–Parro identity shortcut requires tariffs, iceberg costs and deficit
closure to be individually unchanged.

**Fiscal compatibility.** `tariff_revenue_mode="legacy_national"` retains the
bundled MATLAB convention, which uses national rate vectors in the fiscal budget.
`"schedule"` applies the complete bilateral tariff schedule; new policy searches
use that mode. Metadata records the choice and fiscal receipts separately. This
preserves the external reference cases without treating their historical
accounting convention as a general welfare model. `quasi_condensed` rejects
schedule accounting because it does not implement that extension.

**Tariff games.** Best responses search the declared interval and refine sampled
local peaks. Acceptance requires final undamped best-response gaps, unilateral
regret and successful inner GE evaluations. The result records boundaries, search
resolution and failures. Welfare-domain errors also prevent certification.
Unilateral policy searches refuse failed equilibria. These are numerical searches,
not global optimality proofs; high-tariff experiments can remain unresolved.

**Unsupported economic claims.** `decompose_hicksian_ev_3way` and
`verify_theorems_1_to_4` now raise `NotImplementedError`. The former code constructed
its reported conclusions and cannot support the advertised certification.
Historical tests of those claims are retained as non-executed expected failures;
active regression tests check the explicit unavailable contract. None of those
historical tests is counted as economic validation.

**MRIO data.** Synthetic fallback requires explicit opt-in. Source, seed or file
hash, units, conversion assumptions and balancing adjustments survive packaging.
Machine-specific data-directory discovery was removed. A hand-authored labeled
FIGARO interchange fixture checks orientation, units, provenance and accounting.
Spectral acceptance uses Collatz–Wielandt bounds, with periodicity-safe iteration,
strongly connected blocks and a positive-resolvent-vector fallback for small,
badly scaled matrices. An unresolved interval does not certify productivity.

**DSGE and Dynare.** Third-order risk slopes now include the missing timing,
Hessian chain and variance-contraction terms. Analytical cubic and quadratic
benchmarks include multiple states and correlated innovations. The parity
checker compares steady states and every requested first/second-order tensor,
aligns labels and actually compares supplied moments. Missing/malformed data
cannot pass. Parameter-specific `oo_.dr.state_var` takes precedence when labeling
state columns; malformed risk vectors are rejected rather than resized. Failed
Markov-switching solves report unavailable impacts, stability and moments.

**VFI.** Euler projection flags require finite final residuals within the requested
tolerance. Auxiliary collocation values use model productivity and CRRA utility.
Deep Macro restores the current-state activation cache before the detached-target
weight update; a complete training-step derivative is checked by finite differences.
IFT sensitivities are described as semi-analytic, and unimplemented heterogeneous-
agent distribution/GE sensitivities now raise explicitly.

**Documentation and notebooks.** English and Spanish notebooks 63–65 were revised
and executed with new figures. They identify generated inputs, report numerical
search outcomes, separate a toy fold from economic claims, and withdraw the exact
EV/theorem narrative. The technical report sources and PDF were revised. See
[the validation matrix](../../docs/STRUCTURAL_VALIDATION_STATUS.md) for supported
calculations and remaining limits. Optional teaching imports and SMC formatting
were repaired, and the intentional public API additions were reconciled.

## Verification

| Check | Outcome |
|---|---|
| Broad affected suite: VFI, DSGE/Dynare, trade, OccBin, v3.0 additions, API and release checks | 3,120 passed, 30 skipped, 7 deselected, 274 expected failures; two spectral runtime tests initially failed. 469.45 seconds. |
| Final MRIO + independent regression + public API rerun | 96 passed. Both runtime failures pass after avoiding a sparse copy for strictly positive dense matrices. 5.19 seconds. |
| Final independent/Dynare/parity rerun | 107 passed, including the later malformed-vector and column-order checks. 1.74 seconds. |
| Notebook checks | 121 passed, two quarantined expected failures; 12 duplicate execution/timing tests deselected. |
| Actual notebook execution | All six EN/ES editions of 63–65 rebuilt through Jupytext/Jupyter; zero cell errors and fresh opaque figures. Kernel execution required permission for local ports. |
| Technical report | PDF rebuilt successfully: 32 pages; final pass has no LaTeX warnings. |
| Whitespace | `git diff --check` passes. |

The later targeted runs cover the final spectral and Dynare-loader changes. The
entire broad suite was not repeated after those targeted fixes. Counts overlap
between runs and must not be added together. Expected failures are unavailable
historical EV/theorem assertions, not successful numerical comparisons. Backend
and configured exclusions are likewise not validation of those configurations.

Logs are preserved beside this report under `hardening-*.log`; exact commands
are in [hardening-test-commands.txt](hardening-test-commands.txt).

The analytical regressions target independent equations and failure contracts;
the bundled MATLAB references remain unchanged. Fertility diagnostics fixtures
now use their documented pinned calibration and direct residual/generalized-
eigenvalue comparisons, rather than an optimizer endpoint that varied with BLAS.
Tests that assumed a particular Nash point, a Prisoner's Dilemma, or convergence
of a noncontractive iteration now test equations and explicit acceptance criteria.
PAC tests request the accuracy they assert. These are changes to invalid test
assumptions, not new independent empirical references.

## Remaining validation limits

- No new live Dynare third-order run was performed. Analytical success is not
  blanket higher-order parity. [Dynare's manual](https://www.dynare.org/manual/the-model-file.html)
  supplied the decision-rule ordering conventions; the new ordering tests are
  serialization checks, not external numerical oracles.
- Native archives from all five MRIO providers have not been validated. The
  labeled FIGARO interchange fixture is hand-authored and must not be described
  as an official native archive. Other positional adapters remain experimental.
- Exact Hicksian EV, theorem certification, condensed PAC, and heterogeneous-agent
  distribution sensitivities remain unavailable pending derivation and independent
  benchmarks.
- Deep Macro training can fail its held-out Euler criterion, including the default
  deterministic ten-country trajectory. Correct backpropagation does not guarantee
  successful training.
- Grid-based policy searches can miss narrow peaks, and legacy final-demand price
  valuation remains explicit. This pass does not establish general policy validity,
  off-grid VFI accuracy, browser/GPU parity, or full-package external replication.
