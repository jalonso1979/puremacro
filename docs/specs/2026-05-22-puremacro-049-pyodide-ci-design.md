# puremacro 0.49.0 — real Pyodide CI (Gate 6)

**Status:** draft 2026-05-22. Target release: **0.49.0**.

## Why

The package's headline differentiator is its Pyodide-compatibility promise: "pure-numpy + scipy + pandas + matplotlib at runtime, runs on iPad / juno.sh." The existing Gate 2 (the `puremacro/tests/test_pyodide_compat.py` test) verifies this **statically** in CPython — it imports every shippable submodule and asserts no forbidden module (`statsmodels` / `linearmodels` / `arch`) leaked into `sys.modules`. Plus a pyproject-deps cross-check.

That static check catches the most common failure mode (someone adds a top-level `import statsmodels`), but it misses:

- **Runtime failures in Pyodide.** A module that uses a numpy / pandas API that Pyodide's bundled versions don't implement. Static-import test passes; the actual call fails.
- **Wheel-packaging bugs.** A `MANIFEST.in` oversight that leaves a needed file out of the wheel. CPython editable installs hide this; a real Pyodide install via `micropip` catches it.
- **Pyodide-specific filesystem assumptions.** Code that writes to `/tmp` with CPython behavior assumptions, or expects POSIX paths that Pyodide's virtual FS doesn't replicate exactly.

This spec adds **Gate 6** to `tools/release_check.py`: a real Pyodide-in-the-loop smoke that boots Pyodide, installs the freshly-built wheel via `micropip`, and runs a curated subset of pytest. This is one of the seven gates listed in `docs/1.0_path.md` § 4 ("Real Pyodide CI green").

This is the **P3** pitch from the 2026-05-22 brainstorm. P1 (Bayesian DSGE estimation) is still queued post-0.49.0.

## Scope

One release. Three functional units:

1. **`pyodide_smoke` pytest marker** — applied to 8 curated tests across 7 subpackages.
2. **Pyodide runner** (`tools/pyodide/`) — npm-installed Pyodide + a Node script that loads Pyodide, installs the wheel, runs the marked pytest subset, emits JSON.
3. **Gate 6** in `tools/release_check.py` — opt-in via `--pyodide` flag; calls a Python wrapper (`tools/pyodide_smoke.py`) that builds the wheel + invokes the Node runner + parses JSON.

PyPI publishing remains out of scope (separate spec; tied to 1.0).

## Pre-conditions

- 0.48.0 shipped at tag `v0.48.0` (commit `1089460`).
- `tools/release_check.py` has 5 gates with `--examples` opt-in for Gate 5.
- `docs/1.0_path.md` § 4 lists "Real Pyodide CI green" as a 1.0-blocking gate.
- Node + npm available on the maintainer's machine (verified locally: `/opt/homebrew/bin/node`, `/opt/homebrew/bin/npm`).
- `pyproject.toml::[tool.pytest.ini_options].markers` declares `network`; the new `pyodide_smoke` marker joins this list.

## Architecture

Three units, joined under the `--pyodide` opt-in flag in `release_check.py`.

```
  python tools/release_check.py --pyodide
            │
            ├── Gates 1-5 (existing; --examples for Gate 5 opt-in)
            │
            └── Gate 6 (NEW)
                   │
                   └── tools/pyodide_smoke.py::run()  ← Python wrapper
                          │
                          ├── python -m build --wheel -o /tmp/.../    ← ~10s
                          │
                          ├── node tools/pyodide/runner.js --wheel ...   ← ~60-180s
                          │       │
                          │       ├── require("pyodide") + loadPyodide()
                          │       ├── loadPackage([numpy, scipy, pandas, matplotlib, pytest, micropip])
                          │       ├── micropip.install("file:///tmp/.../puremacro-*.whl")
                          │       ├── pytest.main(["-m", "pyodide_smoke", "--tb=short", "-q"])
                          │       └── emit JSON envelope to stdout
                          │
                          └── parse JSON; return {"name": "pyodide_smoke", "passed": bool, "report": str}
```

The wheel is built **fresh per gate run** (no caching). The build takes ~10s for a pure-Python wheel; caching would save little and risks staleness.

Pyodide's `node_modules/` is **cached locally** but gitignored. The maintainer runs `cd tools/pyodide && npm install` once per checkout; subsequent gate runs reuse the cache.

## Components

### `pyodide_smoke` pytest marker

Declared in `puremacro/pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "network: tests requiring live network access (opt-in via `pytest -m network`)",
    "pyodide_smoke: tests safe to run under Pyodide; opt-in via `pytest -m pyodide_smoke`",
]
```

Applied with `@pytest.mark.pyodide_smoke` decorator on individual test functions. Initial set: **8 tests across 7 subpackages**, all known to be fast (<10s) under CPython, exercise numpy/scipy code paths, and avoid `puremacro.teaching` / `puremacro.fetch`. The exact `nodeid`s are confirmed at plan-writing time (Task 0 of the plan dumps the actual test inventory).

### `tools/pyodide/` directory

```
tools/pyodide/
├── package.json          ← declares pyodide npm dep at a pinned version
├── package-lock.json     ← committed (reproducibility)
├── runner.js             ← ~80 LOC Node script
├── README.md             ← one-time setup note
└── .gitignore            ← excludes node_modules/
```

**`package.json`:**

```json
{
  "name": "puremacro-pyodide-runner",
  "version": "0.0.0",
  "private": true,
  "description": "Headless Pyodide harness for puremacro Gate 6 (release_check.py --pyodide).",
  "dependencies": {
    "pyodide": "0.28.3"
  }
}
```

Version pin is exact (no `^` range). Bumping Pyodide is a deliberate maintainer act, not an auto-floating dependency.

**`runner.js`:** Loads Pyodide, installs the wheel, runs the marked tests, emits JSON. ~80 LOC. The full source is in the plan; this spec describes the contract:

- **Argv:** `node runner.js --wheel <absolute-path-to-wheel.whl>`.
- **Stdout:** exactly one JSON document of shape:
  ```json
  {
    "schema_version": 1,
    "pyodide_version": "0.28.3",
    "loaded_at": "2026-05-22T15:00:00Z",
    "wheel_installed": true,
    "wheel_path": "/tmp/.../puremacro-0.49.0-py3-none-any.whl",
    "pytest_returncode": 0,
    "passed": 8,
    "failed": 0,
    "skipped": 0,
    "runtime_s": 84.2,
    "stdout_tail": "============ 8 passed in 12.4s ============"
  }
  ```
- **Stderr:** human-readable progress lines; ignored by the wrapper but useful for live observation.
- **Exit code:** 0 if the JSON envelope was emitted (regardless of `pytest_returncode`); non-zero only if Pyodide failed to boot or `runner.js` itself crashed before producing JSON.

**`README.md`** explains the one-time `npm install` and lists supported Node versions (≥18, matching Pyodide's requirements).

**`.gitignore`** excludes `node_modules/`.

### `tools/pyodide_smoke.py`

Python wrapper, ~150 LOC, stdlib + subprocess. Single function `run(repo_root: Path) -> dict` returning `{"passed": bool, "report": str}` to match the existing gate-result contract.

**Algorithm:**

1. Verify Node available (`subprocess.run(["node", "--version"], ...)`). Missing → return fail dict with install guidance.
2. Verify `tools/pyodide/node_modules/` exists. Missing → return fail dict with `npm install` guidance.
3. Build the wheel: `subprocess.run([sys.executable, "-m", "build", "--wheel", "-o", tempdir], cwd=repo_root, timeout=120)`. Failure → return fail dict with build output.
4. Locate the produced wheel (`puremacro-*.whl` in the temp dir).
5. Invoke `node tools/pyodide/runner.js --wheel <wheel-path>` with `timeout=600`.
6. Parse JSON from stdout. Malformed → return fail dict naming the issue.
7. Inspect `wheel_installed` and `pytest_returncode`. Both clean → PASS. Either flag failure → FAIL with the runner's `stdout_tail` included.
8. Clean up the tempdir.

### Gate 6 in `tools/release_check.py`

`gate_pyodide_smoke(repo_root: Path) -> dict` — calls `tools/pyodide_smoke.py::run(repo_root)` and returns its dict augmented with `"name": "pyodide_smoke"`.

CLI flag added:

```python
parser.add_argument(
    "--pyodide",
    action="store_true",
    help="Also run Gate 6 (real Pyodide smoke). Builds the wheel + boots "
         "Pyodide via node tools/pyodide/runner.js. Slow (~3-5 min); "
         "requires node + one-time `npm install` in tools/pyodide/.",
)
```

Wired in `main()` after Gate 5:

```python
if args.pyodide:
    g6 = gate_pyodide_smoke(REPO_ROOT)
    gates.append(g6)
```

## Data flow

```
Maintainer
   │
   ├── (one-time setup) cd puremacro/tools/pyodide && npm install
   │
   ├── (regular) python tools/release_check.py --pyodide
   │       │
   │       ├── Gates 1-5 run (5 if --examples, 4 if not)
   │       │
   │       └── Gate 6: pyodide_smoke.run(repo_root)
   │               │
   │               ├── build wheel (~10s)
   │               ├── node runner.js (~60-180s)
   │               │       │
   │               │       ├── boot Pyodide
   │               │       ├── micropip install wheel
   │               │       └── pytest -m pyodide_smoke
   │               │
   │               └── parse JSON, return PASS/FAIL
   │
   └── exit code 0 if all gates pass
```

The runner is invoked exclusively through `pyodide_smoke.py`; it is never imported as a library. No persistent state between gate runs except `tools/pyodide/node_modules/` (the npm cache).

## Error handling

**`runner.js` errors:**
- Pyodide load failure → exit 2, no JSON. `pyodide_smoke.py` catches `returncode != 0` AND empty stdout as "Pyodide failed to boot" with stderr surfaced.
- Wheel install failure → JSON with `wheel_installed: false`, `pytest_returncode: -1`. Gate 6 reports FAIL with "wheel install failed in Pyodide" + the `stdout_tail`.
- pytest internal error → JSON with `pytest_returncode: 2` or higher; Gate 6 surfaces the non-zero code in the report.

**`pyodide_smoke.py` errors:**
- `node --version` fails (Node not on PATH) → `{"passed": False, "report": "Gate 6: FAIL — node not installed. Install Node.js (≥18) and run `cd tools/pyodide && npm install` once."}`
- `tools/pyodide/node_modules/` missing → `{"passed": False, "report": "Gate 6: FAIL — Pyodide not installed. Run `cd tools/pyodide && npm install` (one-time, ~150 MB)."}`
- `python -m build` fails → `{"passed": False, "report": "Gate 6: FAIL — wheel build failed. Last 20 lines:\n..."}`
- `subprocess.TimeoutExpired` (600s default) → `{"passed": False, "report": "Gate 6: FAIL — Pyodide runner exceeded 600s timeout."}`
- JSON parse failure → `{"passed": False, "report": "Gate 6: FAIL — runner emitted malformed output. Raw stdout:\n..."}`

**Gate 6** never raises out of `main()`; every error path returns a well-formed gate-result dict.

## Testing

### `puremacro/tests/test_pyodide_smoke_runner.py`

Unit tests for the Python wrapper. No actual Pyodide bootstraps — `subprocess.run` is monkeypatched.

- `test_run_pass` — monkeypatched JSON `{"pytest_returncode": 0, "passed": 8, "wheel_installed": true, ...}` → `{"passed": True, "report": "Gate 6 ... PASS — 8 passed in Pyodide"}`.
- `test_run_fail_test` — `{"pytest_returncode": 1, "passed": 7, "failed": 1, ...}` → `{"passed": False, ...}` with the failing-count in the report.
- `test_run_wheel_install_failed` — `{"wheel_installed": false, ...}` → FAIL with "wheel install failed in Pyodide".
- `test_run_wheel_build_fails` — `python -m build` mocked to return non-zero → FAIL with "wheel build failed" + tail.
- `test_run_node_not_installed` — `subprocess.run(["node", "--version"], ...)` raises `FileNotFoundError` → FAIL with install guidance.
- `test_run_node_modules_missing` — `tools/pyodide/node_modules/` does not exist → FAIL with `npm install` guidance; no other subprocess calls attempted.
- `test_run_json_malformed` — runner returns garbage stdout → FAIL with "malformed runner output" + raw stdout dump.
- `test_run_timeout` — `subprocess.TimeoutExpired` on the node call → FAIL with timeout reason.

### Gate 6 tests in `puremacro/tests/test_release_check.py`

- `test_gate6_pass` — monkeypatch `pyodide_smoke.run` to return `{"passed": True, "report": "..."}` → `gate_pyodide_smoke` returns `{"name": "pyodide_smoke", "passed": True, ...}`.
- `test_gate6_fail` — monkeypatch returns `{"passed": False, ...}` → gate dict has `passed=False`, report visible in `main()` output.
- `test_main_summary_with_pyodide_flag` — `main(["--examples", "--pyodide"])` runs Gates 1-6, all monkeypatched to pass, summary line says "all 6 gates PASS".

### NO tests for `runner.js`

The JavaScript runner is small (~80 LOC) and exercised by Gate 6's actual run. JS test infrastructure would dwarf its testable surface.

### Real Pyodide integration check

Acceptance criterion #10 is a **live Gate 6 run** on the maintainer's machine: `python tools/release_check.py --pyodide` exits 0. This is the integration test that proves the system works end-to-end.

## Acceptance criteria

1. `pyproject.toml::[tool.pytest.ini_options].markers` declares `pyodide_smoke`.
2. 8 tests carry `@pytest.mark.pyodide_smoke`, one per subpackage in: `var/identify/cholesky`, `var/identify/proxy`, `lp/jorda`, `inference/hac`, `dsge/klein`, `volatility`, `gar`, `cycles`. Exact `nodeid`s confirmed at plan-writing time.
3. `tools/pyodide/{package.json, runner.js, README.md, .gitignore}` exist; `package.json` pins `pyodide` exactly (no `^`).
4. `tools/pyodide/package-lock.json` committed.
5. `tools/pyodide_smoke.py` exists (~150 LOC), pure-stdlib + subprocess.
6. `tools/release_check.py` adds `gate_pyodide_smoke` and the `--pyodide` flag; default 4-gate run unchanged.
7. `puremacro/tests/test_pyodide_smoke_runner.py` — 8 unit tests, all green.
8. `puremacro/tests/test_release_check.py` — 3 new Gate 6 tests, all green.
9. Existing 5-gate run still green at HEAD (no regressions).
10. **Live integration:** `python tools/release_check.py --pyodide` exits 0 with Gate 6 PASS (8 tests passed in Pyodide).
11. `docs/1.0_path.md` § 4 — the "Real Pyodide CI green" gate marked `[x]` (one of seven 1.0 gates ticked).
12. `CONTRIBUTING.md` "Before tagging a release" — note `--pyodide` opt-in, slow (~3-5 min), one-time `npm install` setup, requires Node ≥18.
13. `CHANGELOG.md` 0.49.0 entry.
14. Version bumped: `pyproject.toml`, `puremacro/__init__.py`, `puremacro/tests/test_import.py`, CHANGELOG heading — all `0.49.0`.

## Risks and mitigations

1. **Pyodide-version drift.** Pyodide releases tie to specific numpy/scipy/pandas/matplotlib versions. Bumping Pyodide may surface a test regression that has nothing to do with puremacro itself. *Mitigation:* pin exactly (no `^`), document bumps as deliberate maintainer steps in `tools/pyodide/README.md`, and run Gate 6 immediately after bumping.

2. **One of the 8 curated tests is Pyodide-incompatible.** A test that depends on a numpy API not in Pyodide-numpy, or pandas behavior that diverges. *Mitigation:* Task 0 of the plan runs each candidate under Pyodide before locking the list; swap any incompatible test for a sibling. The marker is per-function, so the list is editable cheaply.

3. **Wheel-build catches a real packaging bug** on the first run. A `MANIFEST.in` oversight or `setuptools.packages.find` mismatch could leave files out of the wheel — CPython editable installs hide this. *Mitigation:* this is a feature, not a bug. If found, fix it in 0.49.0 scope. The fix is typically a one-line `MANIFEST.in` or `pyproject.toml` change.

4. **Slow gate gets skipped.** Opt-in `--pyodide` means a forgetful maintainer can tag without exercising Gate 6. *Mitigation:* `docs/1.0_path.md` § 4 elevates Real Pyodide CI to a mandatory 1.0 gate. Until then, the discipline is the same as Gate 5 (run before tag).

5. **Node + npm dependency drag.** Maintainers without Node installed cannot run Gate 6. *Mitigation:* `pyodide_smoke.py` detects missing Node and returns a clear install message; default 4-gate run doesn't need Node.

6. **`tools/pyodide/node_modules/` is heavy (~150 MB).** Gitignored, so no repo bloat, but the disk footprint is real. *Mitigation:* one-time cost per checkout; npm shares via global cache across multiple clones if configured.

7. **Pyodide can't install the wheel.** A wheel that builds for CPython might fail micropip's strict validation (e.g., missing pure-Python tag, dependency on a C extension by mistake). *Mitigation:* the package is already declared pure-Python in `pyproject.toml`; the build produces `py3-none-any` wheels; micropip handles those natively. If micropip rejects, Gate 6 surfaces the error and 0.49.0 expands scope to fix it.

## Out of scope (deferred)

- **Browser-Pyodide via Playwright** — npm-Pyodide and browser-Pyodide share the same Pyodide runtime, so tests passing under Node prove tests pass on juno.sh / iPad. Adding Playwright multiplies CI complexity for marginal additional coverage. If browser-specific bugs need catching later, a follow-on spec.
- **PyPI publishing** — the wheel is built locally for Gate 6 but not pushed to any registry. PyPI publishing is its own spec, listed in `docs/1.0_path.md` § 4 as a separate 1.0 gate.
- **GitHub Actions / hosted CI** — "no CI by design" stands. Gate 6 is a maintainer-discipline tool, run locally before tag.
- **Test count expansion beyond the initial 8.** Future releases can grow the `@pytest.mark.pyodide_smoke` set as confidence builds. Initial 8 is broad-but-shallow coverage; full Pyodide test parity is a 1.0+ target.
- **Bayesian DSGE estimation (P1), mixed-frequency BVAR (P2), numba JIT (P6).** Each has its own spec → plan → impl cycle. P1 is the obvious next pickup after 0.49.0 ships.
