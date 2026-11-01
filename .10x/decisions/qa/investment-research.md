# Investment research validation

Self-review and executed evidence, not an independent-agent signoff.

- Accounting: hand-calculated round trip with fees and a separate dollar-holdings
  recursion; future observations cannot change earlier decisions or wealth.
- Allocation: analytical inverse-variance portfolios over three scales, an
  independent dense-grid solution, cap/turnover/infeasibility tests and LP gap checks.
- Indicators/inference: explicit seeds and warm-up periods, scalar RSI/EMA
  references, prefix invariance, and controls for constant/negative advantages
  and widening the supplied strategy family.
- Fundamentals: release/revision timing, reporting intervals/units, independent
  NPV/Gordon formulas, enterprise/equity bridge, undefined and overflowing ratios.
- Beliefs: independent scalar recurrence, fundamental equilibrium, zero-intensity
  solution, future-shock timing and divergence detection.
- Integration found and fixed timezone/DST and custom-holiday replay failures.
- Both notebooks executed with two PNGs and seven code cells each. Artifact checks
  compare code/source hashes and computational ASTs across languages.
- The five fast release-contract gates passed, including native and Windows-target
  typing under Python 3.11. Real Pyodide 0.28.3 source execution passed 52 tests.
- Final counts: 130 focused host checks and 54 new numerical/integration checks
  in real Pyodide. The full default sweep recorded 22,462 passed, four initial
  failures, 89 skipped and 258 deselected. Three CLI child imports needed the
  source PYTHONPATH; one notebook style failure required display-and-close calls.
  Re-executed notebooks and all affected checks passed (13-check combined run).
  No expectations were weakened; no second full clean sweep is claimed.
