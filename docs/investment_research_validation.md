> 🇬🇧 English · 🇪🇸 [Español](es/investment_research_validation.md)

# Investment research validation record

Validation date: 2026-11-01. Branch: `feat/investment-research`, based on main
`0b3caf3ec44dfbf94b4db47ea0c5b18b16ce6672`. This is unreleased source work.

| Layer | Verified result |
|---|---|
| Host numerical/API integration | 128 asset-pricing tests plus both executable guide pages: **130 passed**; 86 other/opt-in cases deselected |
| New numerical kernels in WebAssembly | **54 passed** in real Pyodide 0.28.3 / Python 3.13.2, using the source tree and distribution packages |
| Notebook execution | Both languages executed: seven code cells and two opaque figures each, zero cell errors; source hashes and computational parity checked |
| Required fast contract gates | **5 passed**: import restrictions, API snapshot, version synchronization, Python 3.11 syntax, native and Windows-target mypy |
| Documentation | `mkdocs build --strict` passed |
| Installed candidate wheel | Both guides executed outside the source import tree: **12 code blocks passed**, identical EN/ES results; all seven module hashes match source |
| Corrected broad-suite checks | **13 passed**: three CLI tests, eight showcase checks and two artifact checks |

## Repository sweep and corrections

<!-- BROAD_SUITE_RESULT -->
The complete default selection across 911 files ran in four file-disjoint
subprocesses: **22,462 passed, 4 failed, 89 skipped, 258 deselected**. All four
initial failures passed after the corrections below. Five subsequently added
regression/artifact cases are included in the final focused run. The longest
shard took about 17 minutes.

The sweep initially found three CLI subprocess import failures because the
worktree was not installed into the shared interpreter. Supplying the worktree
through `PYTHONPATH` made all three pass, without changing those tests or cache
code. It also caught the new notebooks' use of `plt.show()`, which violates the
existing showcase convention. Both notebooks now explicitly display and close
their figures; they were re-executed, and the full eight-test showcase file and
both artifact checks passed. Initial failures remain recorded; targeted reruns
are not described as a second complete clean sweep.

Other review findings were corrected before delivery: DST/timezone replay,
custom holiday calendars, duplicate fundamental columns, finite-input ratio
overflow, and nearly indefinite covariance producing negative reported risk.
The final focused and WebAssembly checks include these numerical corrections.

## Evidence and reproduction

- [Installed-wheel workflow receipt](assets/investment_research/installed_wheel.json)
  records module/guide hashes, interpreter, library versions, replay checksum,
  scenario outputs and the candidate wheel checksum.
- [Pyodide execution receipt](assets/investment_research/pyodide.json) records the
  actual runtime and its pytest result.
- [Combined validation record](assets/investment_research/validation.json) records
  the repository sweep, corrections and explicit validation boundaries.

Host environment: CPython 3.11.17 on macOS 27.2 arm64, NumPy 2.4.6, SciPy 1.17.1,
pandas 3.1.0. Existing pandas frequency deprecation warnings and headless plotting
warnings are visible in host logs; they are not test failures.

```bash
PYTHONPATH=. python -m pytest -q tests/test_asset_pricing
python -m pytest -q tests/test_docs_code_blocks.py -k investment_research
python tools/release_check.py --no-tests --mypy-floor
python -m mkdocs build --strict --site-dir /tmp/investment-docs
node tools/reference_validation/investment_research_pyodide.cjs
```

The Pyodide command uses `tools/pyodide/node_modules/pyodide`, installed through
that directory's existing package manifest, or an explicit module-directory
argument. It executes all six new numerical test files. Four workflow tests are
also marked for the existing release Pyodide smoke selection.

For installed-wheel validation, build the wheel, install it without dependencies
into an isolated target using the numerical environment above, then run from a
separate working directory:

```bash
PYTHONPATH=/path/to/installed python /checkout/tools/reference_validation/check_investment_research.py \
  --expected-package-root /path/to/installed \
  --wheel /path/to/puremacro-6.1.0-py3-none-any.whl \
  --output /tmp/investment-research.json
```

The checker imports the candidate package and asserts its location. It executes
trusted repository guide code; it is not an untrusted-code sandbox. Notebook
sources rebuild with `tools/build_notebooks.py investment_research
investment_research_es`; isolate Jupyter/IPython configuration when preserving a
user's existing kernel configuration.

## What the evidence establishes

Accounting uses hand calculations and a separate dollar-holdings recursion.
Allocation uses analytical solutions, independent grid search, a covariance
outer-product reference and optimization diagnostics. Indicator and belief tests
use scalar recurrences and future-input perturbations. Fundamental selection
tests filing availability/revisions; valuation uses independent present-value
identities. These are numerical and information-timing checks on synthetic inputs.

No empirical profitability, live data-provider normalization, survivor-free
market dataset, estimated belief dynamics or replication of a digitized chaotic
path is claimed. WebAssembly execution occurred in Node, not an iPad/browser UI.
Windows typing is not Windows runtime testing. The local candidate wheel was
tested, not published; the supported-platform CI matrix and release acceptance
remain separate requirements.
