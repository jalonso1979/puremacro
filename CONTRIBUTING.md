# Contributing to puremacro

This is a single-author research package. The conventions below exist so that **future-you** (or a collaborator) can land changes confidently without re-deriving the trade-offs that produced the current shape. They also keep the four cross-cutting promises of the package intact:

1. **Pyodide-compatible** — runs on iPad / juno.sh; no `statsmodels`, `linearmodels`, `arch` at runtime.
2. **Diagnostic over silent** — every numerical-failure path raises a named `LinAlgError` (or fires a `UserWarning`), never returns garbage.
3. **Public API curated in `__init__.py`** — top level exports `__version__`; subpackages re-export their stable names.
4. **Tests over types** — replication-validation + edge-case unit tests are the contract; there is no type checker, no linter, no CI.

If your change touches one of those, read `ARCHITECTURE.md` first.

---

## Where new code goes

| Adding… | Put it in… |
|---|---|
| A linear-algebra primitive (rank-aware inversion, Cholesky, etc.) | `puremacro/_linalg.py` and route every call site through it. |
| An OLS / panel / LP estimator | `puremacro/lp/` for time series, `puremacro/inference/` for SE machinery. Re-use `_ols_helpers.ols_hac` and `_panel_helpers.two_way_fe_within` rather than rewriting OLS. |
| A new SVAR identification scheme | `puremacro/var/identify/<scheme>.py`, expose from `var/identify/__init__.py`, follow the residual-bootstrap drop-and-warn pattern from `cholesky.py` (do **not** silently substitute the point estimate for failed draws). |
| A new HTTP source connector | `puremacro/narrative/sources/<name>.py`. **Use `_http.safe_get_*`** — never write your own `_safe_get`. Read `narrative/sources/RETRY_POLICY.md` first. |
| A new replication loader | `puremacro/narrative/replication/<dataset>.py`. Provide both `load(...)` (network-aware) and `<dataset>_csv_to_events` (pure-Python, offline-testable). Re-export from `narrative/replication/__init__.py` and from `narrative/__init__.py`. |
| A connector that needs an optional dep (LLM SDK, pandas-extension lib) | Lazy-import it inside the backend class or function — never at module top level. Add a `_HAS_X` flag if the import is conditional. |
| A new linear-RE-model solver | `puremacro/dsge/`. Pair with `state_space.py` for likelihood-based estimation and `mcmc.py` for diagnostics. |
| A teaching / replication script | `puremacro/examples/<name>.py`. Examples are not part of the wheel-shipped surface; they import the public API only. Keep one `main()`-style runnable. |

## Where new code does **not** go

- **`puremacro/__init__.py`** stays minimal. Don't add a top-level re-export unless every other submodule already does it.
- **No new files at the package root** unless they cross-cut every submodule (the only ones today are `_linalg.py`, `data.py`, `experiment.py`, `numerics.py`, `mcmc.py`, `posterior.py`, `state_space.py`). New "utilities" almost always belong inside a subpackage.
- **No silent regularisation** without a code comment explaining why the ridge is load-bearing (e.g. `bvar.py` ridges are intentional Bayesian Tikhonov — flagged in code).

## When to add a test

Always, when:

- You change a numerical algorithm — even a rename of a helper. Run `pytest tests/` before you commit.
- You add a public API surface (anything exported from a `__init__.py`). Add at minimum a documented-columns smoke test (look at `test_panel_parity.py` for the pattern).
- You harden a new failure path — add to `tests/test_robustness.py` mirroring the structure of the existing collinear-input / non-PD-Σ cases.
- You add a connector or replication loader — add an offline test in `test_narrative.py` exercising the CSV-to-events helper. Live-network tests go in `test_replication_*` and are allowed to skip.

## Making sure a test can fail

A green test that would still be green with the code deleted is worse than no
test, because it gets read as coverage. Three turned up in a single session, from
three different causes, so there are three different habits:

1. **Write the regression test against the broken tree first.** For a bug fix,
   `git archive <pre-fix-commit> | tar -x -C /tmp/before`, run the new test there,
   and require it to *fail*. If it passes, it is not testing the fix. This costs
   thirty seconds and is the single highest-return habit here.
2. **Do not patch the thing you are asserting about.** A test that monkeypatches
   `qna_panel` and then asserts something about seasonal adjustment is asserting
   something about the patch. Patch at the boundary instead — `get_sdmx_csv`, not
   the function under test. `tests/test_test_quality.py` enforces this for a
   registry of modules: the named test file must actually execute the named
   functions.
3. **Give any mechanism a positive control.** If a test installs an import hook,
   blocks the network, patches `builtins.__import__` or freezes a clock, add a
   test marked `@pytest.mark.mechanism_control` asserting the mechanism actually
   bites. This is not paranoia: `find_module` was removed in Python 3.12, so a
   blocker written against it is silently inert and every test depending on it
   passes while asserting nothing. Neither coverage nor mutation testing can find
   that — the defect is in the scaffolding, not the subject.
   `tests/test_test_quality.py` fails if a mechanism file has no control.

4. **Check the fixture can produce the condition.** A guard that filters rows
   down to a subset is untested if the fixture only ever generates that subset.
   `tools/mutation_check.py` automates exactly this — it deletes guards, flips
   comparisons and neutralises constants one at a time, reruns the tests, and
   reports every change nothing noticed:

   ```bash
   python tools/mutation_check.py puremacro/capital.py
   python tools/mutation_check.py puremacro/fetch/oecd_qna_panel.py \
       --tests tests/test_oecd_qna_panel.py --max-mutants 60
   ```

   It is a tool, not a gate: mutation cost is (mutants × suite time), so it runs
   on demand, when you add a module or when a test file looks suspiciously
   green. Not every survivor is a missing test — some mutations are
   semantically inert, and the report says so — but the guards a fixture never
   exercises show up immediately. Its first run on
   `oecd_qna_expenditure` found three untested branches (`codes=None`, an empty
   response, a response missing required columns); adding those tests took the
   survivor count from nine to five, and the five that remain genuinely do not
   change the output.

Of the three failure modes above, mutation testing finds (1) and (4) and cannot
find (3) — an inert mechanism defeats it, because mutating the subject changes
nothing about scaffolding that was never running. That is why all three exist.

## When to bump the version

| Change | Bump |
|---|---|
| New public function, no removal or signature change | minor (`0.X.0`) |
| Pure internal hardening / refactor / bug fix, **no** API change | patch (`0.X.Y`) |
| Removed or renamed a public symbol; changed a return schema | major (`X.0.0`) |

Update `pyproject.toml`, `puremacro/__init__.py`, `tests/test_import.py`, and add a `CHANGELOG.md` entry **in the same commit**.

## Before tagging a release

Run the release-gate from the repo root:

```bash
python tools/release_check.py
```

This runs four gates and exits 0 only if all pass:

1. **Test baseline** — pytest failing set must equal `tests/known_failures.json::entries[*].nodeid`. New failures → fail. Previously-red-now-green → pass with a warning that the whitelist should shrink.
2. **Pyodide contract** — `tests/test_pyodide_compat.py` green.
3. **Public API snapshot** — fresh introspection must equal `tests/fixtures/public_api_snapshot.json`. Regenerate the fixture deliberately when a public-API change is intentional; the gate never writes.
4. **Version sync** — `pyproject.toml`, `puremacro/__init__.py`, and the first `## X.Y.Z` heading in `CHANGELOG.md` must agree.

If a gate's failure is real and accepted (e.g. environmentally-gated test), add it to `tests/known_failures.json` with a populated `reason` / `since_version` / `owner_note`. The whitelist is the audit trail.

There is no enforcement beyond this command — the package's "tests-over-types, no CI by design" promise stands. The discipline is: run the gate before every `git tag`.

### Opt-in: examples-gallery health (Gate 5)

For a full release check including the examples-gallery health, run with `--examples`:

```bash
python tools/render_examples_gallery.py   # ~5-30 minutes depending on slow examples
git add docs/examples_gallery.json docs/examples_gallery.md puremacro/examples/output/
git commit -m "chore: refresh examples gallery"
python tools/release_check.py --examples
```

The `--examples` flag activates Gate 5, which fails on any example with status FAIL. The render step is slow; the gate itself just reads the committed JSON. Use `--only NAME` to re-render a single example if you only edited one.

### Opt-in: real Pyodide smoke (Gate 6)

For the strictest pre-tag check including Gate 6 (real Pyodide), add `--pyodide`:

```bash
cd tools/pyodide
npm install     # one-time, ~150 MB
cd ../..
python tools/release_check.py --examples --pyodide
```

The `--pyodide` flag builds the puremacro wheel, boots Pyodide via Node, installs the wheel via `micropip`, and runs `pytest -m pyodide_smoke` (currently 8 tests across 7 subpackages). Slow (~60-180s typical; ~6s observed on modern hardware); requires Node ≥18. Default gate run does not include this.

### Opt-in: slow tests (`pytest -m slow`)

Some tests are long-running (minutes). They are skipped by default via the `addopts = ["-m", "not slow"]` in `pyproject.toml`. Run them explicitly before tag if the change touches Bayesian estimation or DSGE code:

```bash
pytest -m slow tests/test_dsge/test_sw07_estimate_replication.py
```

The SW07 Bayesian replication test runs ~20-40 min.

## Diagnostic-error contract

Every public estimator that performs a matrix inversion or factorisation must:

1. Route through `_linalg.inv_xtx` or `_linalg.safe_cholesky` for X'X-shaped or PD-covariance work, **or**
2. Wrap the bare call in `try / except np.linalg.LinAlgError` and return a documented sentinel (`np.nan`, `-np.inf`, etc.), **or**
3. (Bootstrap loops only) count failures and warn above 5% / raise on 100%, never silently substitute the point estimate.

If none of (1)–(3) fits, you're probably designing a new edge case. Add a comment explaining why; future revisions will read it.

## How to run things

```bash
# Smoke check — should be fast.
python -m pytest tests/

# Pyodide compatibility — must pass after any new top-level import.
python -m pytest tests/test_pyodide_compat.py

# A single replication.
python -m puremacro.examples.bloom2009
```

There is no linter or formatter. Keep imports grouped (stdlib / third-party / first-party with a blank line between) and the line width loose-ish but not absurd. The existing files are the style guide.

## Out of scope

- **Async / threads.** Source connectors are synchronous on purpose; see `narrative/sources/RETRY_POLICY.md` §6.
- **Type checking.** Annotations are encouraged for documentation, not enforced. No `mypy --strict`.
- **Backwards-compat shims.** The package is pre-1.0. Rename freely; update consumers in the same commit. Promote private helpers to public when an example needs them, rather than keeping a `_` alias.
