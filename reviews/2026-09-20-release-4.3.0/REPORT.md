# puremacro 4.3.0 release verification

Status: all five required pre-tag gates passed. Publication results are recorded
on the [4.3.0 release page](https://github.com/jalonso1979/puremacro/releases/tag/v4.3.0).

The implementation commit is `cd3a806c6d4feec1b5ed78a9181ed80a2f0e51eb`.
Commit `61b4c78` adds release-check repairs, refreshed notebooks and explicit
known-issue documentation. There are no package-source or dependency changes
between those commits. All 761 runtime Python modules are byte-for-byte
identical to the already tested wheel (`runtime-source-match.json`).
An unrelated local edit to notebook 59 (Spanish) is
excluded from the release.

## Completed checks

- Isolated Python 3.12.13 environment with NumPy 2.5.3, SciPy 1.18.1,
  pandas 3.0.6, Matplotlib 3.11.2 and statsmodels 0.14.6.
- Pyodide static contract, API snapshot, all five version pins and minimum
  Python 3.11 syntax: passed (`fast-gates.log`, `clean-export-gates.log`).
- Strict documentation build: passed (`docs-build-final.log`).
- SW07 10,000-draw, two-chain check: passed; the explicit multi-hour Table 1A
  opt-in was skipped (`sw07-slow.log`). This is the sampler contract, not a
  new claim of full posterior replication.
- Clean `git archive` build of implementation commit: wheel and sdist passed
  Twine. The wheel has 810 entries, the new modules, no PNGs and no MATLAB
  sources or MAT files (`artifact-inspection.json`, `twine-check.log`).
- Fresh installed-wheel environment, isolated from the checkout: version,
  consistent trade accounting, Hicksian welfare, unilateral/Nash search and
  fixed-action payoff smoke passed (`wheel-smoke.json`).
- The installed wheel passed all ten frozen Dynare model/order cases
  (`wheel-dynare-reference.json`).
- Focused release-check regressions: 165 passed, one skip
  (`release-blocker-fixes-final.log`).
- All six trade notebooks (63–65, EN/ES) executed successfully
  (`notebook-build-final.log`). The final corpus opacity/contrast/inventory checks
  passed all four tests (`notebook-corpus-final.log`).

- Complete JupyterLite/playground build from clean commit `61b4c78`: passed.
  All six trade notebooks contain the bootstrap installation cell; the policy
  guide and `puremacro-4.3.0` wheel are present in the generated site
  (`playground-build.log`, `playground-inspection.json`).

- Actual WebAssembly install and marked smoke suite using the repository
  harness: 31 passed, zero failed/skipped under Pyodide 0.28.3
  (`pyodide-smoke.json`). This harness version differs from the current
  playground kernel distribution; it is additional compatibility evidence,
  not complete browser-workload parity.
- The same WebAssembly runtime passed the new Hicksian unilateral/Nash/matrix
  flow (`pyodide-policy-smoke.json`) and all ten frozen Dynare model/order
  cases, including tensors, simulation paths and sample moments
  (`pyodide-dynare-reference.json`). Reproduction wrappers accompany the reports.

## Initial full-suite gate and repairs

The initial gate passed four checks and reported 28 failures outside its
stale 23-entry baseline (`release-gate-initial.log`). The two unrecovered
baseline failures make 30 failing tests in that first run. Focused reruns
preserve the exact failures in `full-gate-failures-recheck.log`.

Release-blocking discrepancies were resolved as follows:

- Restored the Spanish parity example, including externally supplied references,
  and removed its obsolete claim that no external third-order run existed.
- Notebook inventory checks now verify source/rendered pairs and retain the
  original corpus, then inspect all discovered images. Fixed corpus counts had
  stopped them at the six new notebooks before checking their content. Opacity,
  background, contrast and syntax assertions remain active. Removed four direct
  `plt.show()` calls and rebuilt the notebooks using the inline display hook.
- Plot-title checks use the active title location, including the notebook theme's
  left-aligned titles; they still assert the expected label text.
- MATLAB exclusion permits exactly two development-reference files. Neither is
  shipped in the wheel; runtime dependency/import checks remain active.
- The release-check worker closes stdin and detaches it before `communicate`,
  preventing a flush of a closed pipe and the consequent false worker failure.
- Hawkins–Simon tests accept an explicitly unresolved bound as a rejection and
  require an upper bound below one for acceptance. Reducibility does not imply
  a narrow global Collatz–Wielandt interval. No estimate is promoted to a proof.
- The extreme-shock DSGE test retains finite, bounded pruned paths for both
  shock signs. It no longer demands an unpruned path explode: that is not a
  necessary property, and its old hand-written recurrence omitted risk terms.
  Independent frozen-Dynare tensor/path comparisons remain the numerical oracle.

## Confirmed pre-existing failures

The previous `main`/4.2.0 CI run already failed:
https://github.com/jalonso1979/puremacro/actions/runs/35391309540

`prior-main-ci-failures.json` records its failing node IDs. A focused local
recheck reproduced 13 of them; two release-tool/reference-contract issues were
fixed, leaving 11 accepted baseline tests. `baseline-triage.json` and
`prior-failures-recheck.log` retain that evidence. Twenty-one obsolete recovered
entries were removed; no newly introduced runtime failure was added to the
baseline.

The remaining cases concern flexible-configuration alias replacement, singular
posterior covariance rejection by the harmonic-mean estimator, legacy flexible
welfare closure in large monetary units, and VIF/quantile floating-point parity.
`tests/known_failures.json` and CHANGELOG 4.3.0 state the individual reasons,
workarounds and 4.3.1 follow-up target. Raw cross-platform CI may remain red;
passing the release baseline does not mean every test passes.

## Final gate and publication

The final full run completed in 1620.75 seconds (27 minutes): **15,667 passed,
11 failed, 108 skipped, 186 deselected and 276 expected failures**. All eleven
failures exactly match the confirmed prior-release baseline. The baseline,
Pyodide contract, public API, version sync and minimum-Python syntax gates all
passed (`release-gate.log`). The package source was unchanged throughout this
run; subsequent changes only record its results.

The interrupted sandbox launch was stopped immediately and restarted with the
existing authorization for notebook-kernel ports. Its partial log is retained
separately and is not counted as a completed run.

Publication follows the commit of this verification record: an annotated tag on
that exact commit triggers PyPI, and the main-branch push triggers the site.
The release page records the final commit and workflow links. Verify the live
version independently with the installed-wheel smoke script; a passing baseline
gate does not make the raw cross-platform test suite universally green.
