# Unpublished research release candidate

Status: **software validation passed.** Artifacts, installed-wheel checks and
strict documentation passed. The first complete release gate passed four of five
gates and identified three unexpected notebook integrity failures. Those repairs
are complete, independently reviewed and frozen. The second complete gate also
passed four of five gates: two exact source/rendered notebook checks found
trailing-newline differences introduced by the display-only repair. Those
differences were corrected and independently verified. The
[third complete gate](release-gate-after-exact-parity.log) passed **all five
gates**: **18,441 passed, 11 failed (all known), 139 skipped, 202 deselected and
276 xfailed** in 2,395.48 seconds, with no unexpected failures. The
[third source comparison](final-source-match-third.json) confirms no drift from
the [third freeze](third-candidate-provenance.json), and the
[final summary](summary.json) records `software_validation_passed: true`. The
candidate is not tagged, published or deployed.

**Shipped as 4.4.0 (2026-10-02).** Version 4.3.0 was already on PyPI, so this
candidate was released as 4.4.0, together with cross-platform fixes found on
CI, the resolution of all eleven accepted test failures and a `vif` accuracy
fix (see the 4.4.0 changelog and `docs/ADVISORY.md`). Those later commits were
validated by a fresh full gate (18,453 passed, 0 failed) and a clean
`git archive` build; the evidence in this folder describes the frozen
candidate only.

All three originally unexpected failures now pass. The broader notebook checks
report **640 passed and one unchanged known failure**, with no unexpected
failures. Five rebuilt notebooks match their sources and contain no execution
errors. The remaining display-only repairs preserve the numerical code and
saved outputs; a fresh representative execution retains all three figures.
See the [notebook repair manifest](notebook-repair/final_manifest.json),
[display-only evidence](notebook-inline-display-repair.json) and
[fresh execution check](inline-display-execution.json).

The [amended freeze](amended-candidate-provenance.json) confirms that all **878
shipped package files** remain identical to the original candidate. The existing
wheel and source distribution therefore remain the candidate artifacts. The
original failed full-suite evidence is preserved; targeted checks are not
reported as a second full-suite pass.

The [second complete gate](release-gate-after-notebook-repairs.log) reported
**18,439 passed, 13 failed, 139 skipped, 202 deselected and 276 xfailed** in
1,908.73 seconds. Its 13 failures comprise the same eleven known failures and
two unexpected NB07 EN/ES exact cell-source comparisons. Pyodide, API, version
and Python 3.11 syntax gates passed again. The
[preserved second summary](second-validation-summary.json) records all five
outcomes; its [source comparison](second-final-source-match.json) confirms no
post-freeze source or documentation drift. A
[complete display-repair comparison](second-gate-all-display-parity.json)
identifies 66 cells in 30 rendered notebooks with trailing-newline differences
only. Calculations and saved outputs are unchanged.

The [exact-parity repair](notebook-exact-source-parity-repair.json) corrects
those 66 cells. An [independent audit](independent_exact_parity_audit.md) verifies
all 566 cell sources without stripping whitespace and reconstructs all 30 prior
notebook hashes, proving that outputs, metadata and calculations are unchanged.
The [targeted follow-up](exact-parity-targeted-tests.json) reports **464 passed,
one unchanged known failure and no unexpected failures**, including successful
checks of both previously failing comparisons. The
[third freeze](third-candidate-provenance.json) records the exact final inputs;
the third full rerun has a separate log and does not replace either earlier
failed run.

The [independent evidence audit](independent_evidence_audit.md) verifies both
archive hashes, all 883 wheel `RECORD` entries, 878 shipped file comparisons,
381 strict-documentation hashes and the preserved original failure records.

The candidate retains package version **4.3.0** and is identified separately by
a unique artifact directory, Git base and complete source hashes. It includes
the current working tree, including earlier changes; it is not a feature-only
release. Nothing is committed, tagged, published or deployed by these checks.

## Preflight

- Reconciled exactly nine additive, previously implemented API entries after
  reviewing their definitions, defaults and regression coverage. **286 tests
  passed**, with six optional-reference skips and four deselections. No prior
  export or field was removed. [API audit](API_AUDIT.md) and
  [exact reconciliation record](api-reconciliation.json).
- The static Pyodide contract, synchronized 4.3.0 version pins and minimum-Python
  syntax checks passed. The preflight API gate reported the expected stale
  entries plus research APIs being developed at that moment; the final gate
  runs after the deliberate snapshot updates. [Preflight log](preflight-gates.log).
- A fresh Python 3.12.13 validation environment was assembled offline from
  existing cached packages, including NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6 and
  statsmodels 0.14.6. This numerical stack matches the previous 4.3.0 release
  validation. Full-suite collection selected **18,830 tests**, deselected 202
  and had no collection errors. [Collection log](collection-preflight.log).

## Final validation procedure

1. Freeze and hash package/build inputs, tests, tooling and documentation.
2. Build a fresh wheel and source distribution from the frozen package copy;
   check both with Twine. Authenticate shipped source and bundled data.
3. Run all five default `tools/release_check.py` gates in the isolated validation
   environment. The test gate compares every observed failure with the existing
   known-failure list; this candidate does not expand that list.
4. Install the wheel outside the checkout. Block network operations and
   statsmodels imports, verify both blocking mechanisms, then run the new SW07
   experiment and the GE-to-household application, including their recorded
   controls and exported artifact checksums.
5. Run all eight independent research benchmarks from the installed wheel.
   Their existing **7/8 result and overall FAIL** remain visible: the RR2010
   printed-statistic discrepancy is not relaxed or reclassified.
6. Build documentation with strict warnings. Compare final package/test/tool
   sources with the freeze; any subsequent prose updates are listed and checked
   against the final documentation build separately.

The two-draw installed SW07 checks test installation, paired streams, exported
evidence and checkpoint replay. Scientific conclusions must use the full,
separately reported calibration and validation experiments. The GE application
combines observed ENIGH baskets with an explicitly synthetic economy; these
checks do not turn its assumptions into an empirical Mexican tariff estimate.

## Completed candidate checks

The frozen candidate is `/tmp/puremacro-research-candidate-20261001T214014Z`.
[candidate-provenance.json](candidate-provenance.json) authenticates **3,295**
package, build, test, tool, documentation and notebook files. The original
[source comparison](original-final-source-match.json) confirmed unchanged Python,
tests, tooling, configuration and packaged data before notebook repairs. Six documentation pages received
cross-links or completed-study results and passed a fresh strict build. Five
test-regenerated PNGs under `puremacro/examples/output/` are recorded separately
and are absent from the wheel. The full gate runs this frozen checkout,
preserving its fixture paths; package artifacts are built from a separate
copied source directory.

The [first full release gate](release-gate.log) completed in 2,096.77 seconds:
**18,438 passed, 14 failed, 139 skipped, 202 deselected and 276 xfailed**.
All eleven listed known failures remained, and three notebook-integrity checks
failed unexpectedly. The other four gates passed: Pyodide contract, public API,
all five 4.3.0 version pins and Python 3.11 syntax for 1,653 files.
[Original validation summary](original-validation-summary.json).

The [targeted traceback rerun](unexpected-notebook-failures.log) confirmed an
uncommented Spanish Markdown paragraph in notebook 00, section-order drift in
notebook 10 and mismatched English/Spanish cell counts in notebook 45. Fixing
the syntax also exposes the existing styling check's later assertion against
redundant display calls. These notebook repairs leave the package artifacts and
known-failure whitelist unchanged. Their complete path/hash manifest and the
second full-gate result will be recorded separately; the original failure
evidence remains intact.

- Wheel and source distribution both passed Twine. The wheel has 884 entries;
  all package Python modules are present, and 878 shipped source/resource files
  match their source hashes. Required SW07, ENIGH and RR2010 data are present.
  The source distribution contains the same 878 authenticated package files;
  the wheel has no PNGs, bytecode, shared libraries or MAT files.
  [Artifact hashes and locations](artifacts.json), [build log](build.log),
  [Twine log](twine-check.log). Retained files:
  [candidate wheel](artifacts/puremacro-research-candidate-20261001T214014Z/puremacro-4.3.0-py3-none-any.whl)
  and [source distribution](artifacts/puremacro-research-candidate-20261001T214014Z/puremacro-4.3.0.tar.gz).
- Installed-wheel validation passed with the checkout absent from `sys.path`.
  Network connections and statsmodels imports were blocked and their controls
  verified. Every loaded package module came from the isolated installation.
  [Installed summary](installed-smoke.json).
- The exact SW07 mean path matches the independent full covariance API.
  Two calibration and two validation samples produced 16 paired variant fits;
  seed streams are disjoint across phases and shared across variants. A completed
  checkpoint reproduces the draw rows and moment/covariance arrays exactly.
  The tiny reference pool retains explicitly unbounded critical values.
- All six GE household scenarios, zero-tariff control and the public CLI passed.
  CLI/API incidence agrees. Maximum fiscal allocation residual is
  `3.814697265625e-6` MXN and maximum income reconciliation residual is
  `0.01229095458984375` MXN against approximately 1.8 trillion MXN of baseline
  consumption. The synthetic-model and observed-data labels remain distinct.
- Installed independent benchmarks retain the expected **7/8, overall FAIL**;
  only RR2010's printed horizon-10 t-statistic misses its unchanged rounding
  tolerance. There are no benchmark execution errors.
- Strict MkDocs build passed. [Log](mkdocs-strict.log) and
  [documentation source hashes](docs-provenance.json).

The scientific evidence is reported separately: the
[SW07 saved-result audit](../2026-10-01-sw07-estimator-experiment/independent_audit.md)
authenticates all 399 calibration and 999 validation samples and independently
reconstructs their statistics without importing or refitting the production
estimator. The
[GE accounting and welfare audit](../2026-10-01-ge-distributional/independent_audit.md)
reconstructs prices, factor income, duties, both countries' budgets and household
welfare directly from exported transactions and quantities, with independent
optimization checks. Neither audit converts a conditional model experiment into
an empirically identified policy effect or composite-null inference.

Reproduction scripts: [preflight.py](preflight.py),
[validate_candidate.py](validate_candidate.py) and
[installed_smoke.py](installed_smoke.py).
