# puremacro 0.49.0 — Real Pyodide CI (Gate 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Gate 6 — an opt-in `--pyodide` release-gate that boots Pyodide via Node (`tools/pyodide/runner.js`), installs the freshly-built puremacro wheel via `micropip`, mounts the `tests/` directory, and runs `pytest -m pyodide_smoke` on 8 curated tests. Tag as **0.49.0**, ticking the "Real Pyodide CI green" gate in `docs/1.0_path.md` § 4.

**Architecture:** Three units. (1) `@pytest.mark.pyodide_smoke` marker declared in `pyproject.toml`, applied to 8 fast Pyodide-safe tests. (2) `tools/pyodide/` — npm-installed Pyodide + `runner.js` Node script. (3) `tools/pyodide_smoke.py` Python wrapper + `gate_pyodide_smoke` in `tools/release_check.py`. Wheel built fresh per gate run; Pyodide cached in `tools/pyodide/node_modules/` (gitignored).

**Tech Stack:** Python ≥3.10 (stdlib + subprocess), Node ≥18 (`pyodide` npm package pinned to 0.28.3), `python -m build` for wheel construction.

**Source spec:** `docs/specs/2026-05-22-puremacro-049-pyodide-ci-design.md` (commit `e0586db`).

**Pre-execution state (HEAD `e0586db`, on `feature/subnational-labor-uncertainty-us`):**
- 0.48.0 shipped at tag `v0.48.0` (commit `1089460`).
- Current versions: all three sync files at `0.48.0`.
- `tests/known_failures.json::entries` = `[]`.
- `tools/release_check.py` has 5 gates with `--examples` opt-in for Gate 5.
- Node ≥18 + npm available on the maintainer's machine (`/opt/homebrew/bin/node`).

**Verified candidate tests (2026-05-22, all under outer `tests/`):**

| Test file | Statsmodels? | Picked function (at Task 1) |
|---|---|---|
| `tests/test_cholesky_shocks.py` | no | pick one fast `test_*` function |
| `tests/test_var/test_proxy_svar.py` | no | `test_proxy_svar_irf_shape` |
| `tests/test_lp/test_jorda_parity.py` | **no** (name misleading; pure-numpy parity, no statsmodels import) | pick one fast `test_*` function |
| `tests/test_inference/test_hac_fixed_b.py` | no | pick one fast `test_*` function |
| `tests/test_dsge/test_klein_unit_eigenvalue_fallback.py` | no | `test_klein_stable_system_uses_closed_form_path` |
| `tests/test_volatility/test_sigma.py` | no | pick one fast `test_*` function |
| `tests/test_gar/test_qar_skewt_fci.py` | no | pick one fast `test_*` function |
| `tests/test_cycles.py` | no | pick one fast `test_*` function |

Task 1 picks one specific test function per file by reading the file and selecting the smallest/fastest test that exercises a meaningful code path.

---

## File structure

**Created in this release:**

- `puremacro/tools/pyodide/package.json` — declares `pyodide` npm dep at pinned `0.28.3`.
- `puremacro/tools/pyodide/package-lock.json` — committed (reproducibility).
- `puremacro/tools/pyodide/runner.js` — ~80 LOC Node script: boots Pyodide, installs wheel, runs pytest, emits JSON.
- `puremacro/tools/pyodide/README.md` — one-time setup notes (node ≥18, `npm install`).
- `puremacro/tools/pyodide/.gitignore` — excludes `node_modules/`.
- `puremacro/tools/pyodide_smoke.py` — ~150 LOC Python wrapper.
- `puremacro/tests/test_pyodide_smoke_runner.py` — 8 unit tests for the wrapper.

**Modified:**

- `puremacro/pyproject.toml` — add `pyodide_smoke` marker.
- `puremacro/tools/release_check.py` — add `gate_pyodide_smoke` + `--pyodide` flag.
- `puremacro/tests/test_release_check.py` — 3 new Gate 6 tests.
- 8 test files (one per item above) — each gets `@pytest.mark.pyodide_smoke` on one specific test function.
- `puremacro/docs/1.0_path.md` — tick the "Real Pyodide CI green" gate.
- `puremacro/CONTRIBUTING.md` — note the `--pyodide` flag.
- `puremacro/CHANGELOG.md` — 0.49.0 entry.
- `puremacro/__init__.py`, `puremacro/pyproject.toml`, `puremacro/tests/test_import.py` — version bump.

**Untouched:**

- `puremacro/puremacro/...` — no estimator changes.
- `puremacro/tests/test_pyodide_compat.py` — the existing static check stays; Gate 6 complements it.
- Public-API snapshot fixture — wrapper lives in `tools/`, not the wheel.

---

## Working-directory convention

All paths relative to the **repo root**:

`/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/`

Within that root: subproject at `puremacro/`. Inside the subproject:
- Python package: `puremacro/puremacro/`
- Tools: `puremacro/tools/`
- Tests: `puremacro/tests/`
- Docs: `puremacro/docs/`

For brevity, the rest of this plan uses `cd puremacro` as a working assumption when convenient.

---

## Task 0: Pre-flight + branch creation + Node verification

- [ ] **Step 1: Verify clean state**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git status --short puremacro/ | grep -v "\.png$\|\.csv$" | head -10
```

Expected: empty.

- [ ] **Step 2: Confirm HEAD + 5-gate baseline**

```bash
git log --oneline -1
python puremacro/tools/release_check.py
```

Expected: HEAD on `feature/subnational-labor-uncertainty-us`, top commit `e0586db docs(0.49.0): spec — real Pyodide CI (Gate 6)`. Gate output: all 4 gates PASS at 0.48.0 (Gate 5 needs `--examples` to run, OK to skip here).

- [ ] **Step 3: Verify Node + npm available**

```bash
node --version
npm --version
```

Expected: Node ≥18 (e.g. `v20.x.x` or `v22.x.x`). npm any modern version. If node < 18 or missing, STOP and ask the user to install/upgrade Node.

- [ ] **Step 4: Verify `python -m build` works**

```bash
python -c "import build; print(build.__version__)"
```

Expected: prints a version string. If `ModuleNotFoundError: build`, install it:

```bash
pip install --user build
```

Then re-verify.

- [ ] **Step 5: Create the release branch**

```bash
git checkout -b release/0.49.0
git branch --show-current
```

Expected: `release/0.49.0`.

- [ ] **Step 6: No commit** — Task 0 is verification only.

---

## Task 1: Declare the `pyodide_smoke` marker + tag 8 tests

- [ ] **Step 1: Add the marker declaration to `pyproject.toml`**

In `puremacro/pyproject.toml`, find the existing `[tool.pytest.ini_options]` section:

```toml
[tool.pytest.ini_options]
markers = [
    "network: tests requiring live network access (opt-in via `pytest -m network`)",
]
```

Replace with:

```toml
[tool.pytest.ini_options]
markers = [
    "network: tests requiring live network access (opt-in via `pytest -m network`)",
    "pyodide_smoke: tests safe to run under Pyodide; opt-in via `pytest -m pyodide_smoke`",
]
```

- [ ] **Step 2: For each of the 8 candidate files, pick one fast test function**

For each file in the candidates table, open it and pick ONE test function that:
- Imports only numpy / scipy / pandas / matplotlib + puremacro symbols.
- Finishes in <10s under CPython (heuristic: small fixtures, small bootstrap n_boot, no Bayesian MCMC, no full SVAR identification on real data).
- Exercises a meaningful code path (not a trivial `assert True` smoke).

Procedure for each file:

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
grep -n "^def test_" tests/test_cholesky_shocks.py
# Pick the smallest one; verify by reading it.
```

Repeat for each of the 8 files. Build a list `picked = {file: function_name}`.

Then add `@pytest.mark.pyodide_smoke` to each picked function. Example pattern (you must `import pytest` if not already imported):

```python
import pytest

@pytest.mark.pyodide_smoke
def test_some_function():
    ...
```

Use `Edit` on each test file with sufficient context to make the match unique.

- [ ] **Step 3: Verify the marker is recognized by pytest**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/ -m pyodide_smoke --collect-only -q 2>&1 | tail -15
```

Expected: 8 tests collected (one from each file). If fewer, locate the missing marker.

- [ ] **Step 4: Run the 8 marked tests under CPython to verify they all pass**

```bash
python -m pytest tests/ -m pyodide_smoke -v 2>&1 | tail -15
```

Expected: 8 PASSED. Their runtime under CPython should sum to <30s (each test <10s).

If any test fails under CPython, it would also fail under Pyodide — that's a real test failure unrelated to this work. Fix or swap the test before continuing.

- [ ] **Step 5: Run the full default gate to confirm no regression**

```bash
python tools/release_check.py
```

Expected: all 4 gates PASS at 0.48.0 (or 5 if you choose to run Gate 5 with `--examples`).

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/pyproject.toml puremacro/tests/test_*.py puremacro/tests/test_*/test_*.py
git commit -m "$(cat <<'EOF'
feat(0.49.0): declare pyodide_smoke pytest marker + tag 8 tests

Marker declared in pyproject.toml alongside the existing network
marker. Applied to one fast test per file across 7 subpackages:
cholesky, proxy axis, lp jorda, inference HAC fixed-b, klein closed-
form path, sigma object, gar skew-t, cycles.

These are the smoke tests that will run under Pyodide via Gate 6
(landing in a later commit). All 8 pass under CPython today.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create `tools/pyodide/` scaffold

- [ ] **Step 1: Create the directory**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
mkdir -p tools/pyodide
```

- [ ] **Step 2: Write `tools/pyodide/package.json`**

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

Note the exact version pin — no `^`. Bumping Pyodide is a deliberate maintainer act.

- [ ] **Step 3: Write `tools/pyodide/.gitignore`**

```
node_modules/
```

- [ ] **Step 4: Write `tools/pyodide/README.md`**

```markdown
# Pyodide harness for puremacro Gate 6

This directory holds the headless Pyodide runner used by
`tools/release_check.py --pyodide` (Gate 6).

## One-time setup

```bash
cd tools/pyodide
npm install
```

This downloads the pinned Pyodide build (~150 MB into `node_modules/`,
gitignored). Subsequent gate runs reuse the cache.

## Requirements

- Node.js ≥ 18 (Pyodide's requirement).
- npm (bundled with Node).

## Pyodide version

Pinned exactly in `package.json` (no `^` range). Pyodide releases tie to
specific numpy / scipy / pandas / matplotlib versions, so a floating
range would invite hidden drift. Bumping is a deliberate maintainer
act; re-run Gate 6 immediately after bumping.

## Contract

`node runner.js --wheel <absolute-path-to-puremacro-*.whl>` emits one
JSON document to stdout:

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

Exit 0 if the JSON envelope was emitted (regardless of
`pytest_returncode`); non-zero only if Pyodide failed to boot or
`runner.js` itself crashed before producing JSON.
```

- [ ] **Step 5: Verify the scaffold doesn't include any other files yet**

```bash
ls puremacro/tools/pyodide/
```

Expected exactly: `package.json`, `.gitignore`, `README.md`. (`runner.js` is added in Task 3.)

- [ ] **Step 6: Commit**

```bash
git add puremacro/tools/pyodide/package.json puremacro/tools/pyodide/.gitignore puremacro/tools/pyodide/README.md
git commit -m "$(cat <<'EOF'
feat(0.49.0): scaffold tools/pyodide/ — package.json + README + gitignore

Declares pyodide npm dep pinned at 0.28.3 exact (no ^). README
explains the one-time `npm install` setup + the runner.js contract.
node_modules/ gitignored.

runner.js + the Python wrapper land in subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `npm install` + write `runner.js`

This task installs Pyodide (one-time, ~150 MB download) and writes the Node runner. The runner is exercised live in Task 5.

- [ ] **Step 1: Run `npm install`**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro/tools/pyodide"
npm install 2>&1 | tail -5
```

Expected: takes 30-60s. Output ends with `added N packages in T s`. Creates `node_modules/` + `package-lock.json`.

If npm errors out, investigate (network, Node version, npm registry, etc.).

- [ ] **Step 2: Verify Pyodide is loadable via Node**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro/tools/pyodide"
node -e "const { loadPyodide } = require('pyodide'); loadPyodide().then(p => { console.log('Pyodide loaded:', p.version); }).catch(e => { console.error('FAIL:', e); process.exit(1); });"
```

Expected: prints `Pyodide loaded: 0.28.3` (or whatever's pinned) after a few seconds. If this fails, the Pyodide install is broken — investigate.

- [ ] **Step 3: Write `tools/pyodide/runner.js`**

```javascript
#!/usr/bin/env node
// runner.js — headless Pyodide harness for puremacro Gate 6.
//
// Argv: --wheel <absolute-path-to-puremacro-*.whl>
//
// Stdout: exactly one JSON document with the schema in README.md.
// Stderr: human-readable progress lines.
// Exit code: 0 if JSON envelope emitted; non-zero if Pyodide failed to boot.

const { loadPyodide } = require("pyodide");
const path = require("path");
const fs = require("fs");

function parseArgs() {
    const argv = process.argv.slice(2);
    let wheel = null;
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === "--wheel" && i + 1 < argv.length) {
            wheel = argv[i + 1];
            i++;
        }
    }
    if (!wheel) {
        console.error("usage: node runner.js --wheel <path-to-wheel.whl>");
        process.exit(2);
    }
    if (!fs.existsSync(wheel)) {
        console.error(`error: wheel not found at ${wheel}`);
        process.exit(2);
    }
    return { wheel: path.resolve(wheel) };
}

async function main() {
    const t_start = Date.now();
    const { wheel } = parseArgs();

    // tests directory to mount: <repo>/puremacro/tests
    // Resolve relative to this script's location.
    const here = __dirname;  // .../puremacro/tools/pyodide
    const repo_subproject = path.resolve(here, "..", "..");  // .../puremacro
    const tests_dir = path.join(repo_subproject, "tests");
    if (!fs.existsSync(tests_dir)) {
        console.error(`error: tests dir not found at ${tests_dir}`);
        process.exit(2);
    }

    console.error("loading Pyodide ...");
    const pyodide = await loadPyodide();
    const pyodide_version = pyodide.version;
    const loaded_at = new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
    console.error(`Pyodide ${pyodide_version} loaded`);

    console.error("loading numpy / scipy / pandas / matplotlib / pytest ...");
    await pyodide.loadPackage(["numpy", "scipy", "pandas", "matplotlib", "pytest", "micropip"]);

    // Mount the host tests/ dir into Pyodide's FS at /mnt/tests (readonly).
    console.error(`mounting ${tests_dir} -> /mnt/tests`);
    pyodide.mountNodeFS("/mnt/tests", tests_dir);

    // Install the wheel via micropip from a file:// URL into Pyodide's FS.
    // We copy the wheel bytes into Pyodide first, then install from /tmp.
    console.error("installing puremacro wheel via micropip ...");
    const wheel_basename = path.basename(wheel);
    const wheel_bytes = fs.readFileSync(wheel);
    pyodide.FS.writeFile(`/tmp/${wheel_basename}`, wheel_bytes);

    let wheel_installed = false;
    try {
        await pyodide.runPythonAsync(`
import micropip
await micropip.install("emfs:/tmp/${wheel_basename}")
import puremacro
print("puremacro version:", puremacro.__version__)
        `);
        wheel_installed = true;
    } catch (e) {
        console.error("wheel install failed:", e.message);
    }

    // Run the marked pytest subset.
    console.error("running pytest -m pyodide_smoke ...");
    let pytest_returncode = -1;
    let passed = 0;
    let failed = 0;
    let skipped = 0;
    let stdout_tail = "";

    if (wheel_installed) {
        const py_out = await pyodide.runPythonAsync(`
import io, sys
import pytest
buf = io.StringIO()
old_stdout, old_stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = buf
try:
    rc = pytest.main(["/mnt/tests", "-m", "pyodide_smoke", "--tb=short", "-q"])
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr
out = buf.getvalue()
# Parse summary line (last line that looks like "N passed, M failed, K skipped in T s").
import re
m = re.search(r"(\\d+) passed", out)
p = int(m.group(1)) if m else 0
m = re.search(r"(\\d+) failed", out)
f = int(m.group(1)) if m else 0
m = re.search(r"(\\d+) skipped", out)
s = int(m.group(1)) if m else 0
tail_lines = out.strip().splitlines()[-3:]
tail = "\\n".join(tail_lines)
[int(rc), p, f, s, tail]
        `);
        const arr = py_out.toJs();
        pytest_returncode = arr[0];
        passed = arr[1];
        failed = arr[2];
        skipped = arr[3];
        stdout_tail = arr[4];
        py_out.destroy();
    }

    const runtime_s = (Date.now() - t_start) / 1000;
    const envelope = {
        schema_version: 1,
        pyodide_version,
        loaded_at,
        wheel_installed,
        wheel_path: wheel,
        pytest_returncode,
        passed,
        failed,
        skipped,
        runtime_s: Math.round(runtime_s * 10) / 10,
        stdout_tail,
    };
    console.log(JSON.stringify(envelope));
}

main().catch((e) => {
    console.error("runner.js fatal:", e.stack || e.message);
    process.exit(1);
});
```

Notes for the implementer:
- The `mountNodeFS` API is Node-Pyodide-specific (not browser-Pyodide). Required for our headless gate.
- `emfs:/` is Pyodide's URL scheme for files already in its emscripten FS — see Pyodide docs. If `emfs:` doesn't work, fall back to copying the wheel into `/tmp/` and installing via `f"/tmp/{wheel_basename}"` (no `emfs:` prefix).
- The pytest summary parse via regex is crude but adequate; if pytest emits an unexpected format, the JSON still produces sensible defaults (0/0/0).
- The runner is intentionally chatty on stderr to make live observation easy.

- [ ] **Step 4: Smoke-test runner.js with a built wheel**

First build a wheel:

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
rm -rf /tmp/puremacro_test_wheel
mkdir -p /tmp/puremacro_test_wheel
python -m build --wheel -o /tmp/puremacro_test_wheel 2>&1 | tail -5
ls /tmp/puremacro_test_wheel/
```

Expected: `puremacro-0.48.0-py3-none-any.whl` exists.

Then run runner.js:

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
node tools/pyodide/runner.js --wheel /tmp/puremacro_test_wheel/puremacro-0.48.0-py3-none-any.whl 2>/tmp/runner_stderr.log
```

Expected:
- Stderr (`/tmp/runner_stderr.log`) has progress lines like `loading Pyodide ...`, `Pyodide 0.28.3 loaded`, `loading numpy / ...`, `mounting ... -> /mnt/tests`, `installing puremacro wheel ...`, `running pytest ...`.
- Stdout has exactly one JSON document. Pipe to `jq` for inspection:

```bash
node tools/pyodide/runner.js --wheel /tmp/puremacro_test_wheel/puremacro-0.48.0-py3-none-any.whl 2>/dev/null | python -c "import json, sys; print(json.dumps(json.load(sys.stdin), indent=2))"
```

Expected JSON:
- `schema_version: 1`
- `wheel_installed: true`
- `pytest_returncode: 0`
- `passed: 8` (from Task 1's marker tags)
- `failed: 0`
- `runtime_s: ~60-180`

If `wheel_installed: false` or `failed > 0`, root-cause before continuing. Common failure modes:
- `emfs:` URL scheme not supported → swap for `/tmp/<wheel>`.
- Test imports something unavailable in Pyodide → swap the test for a sibling.

- [ ] **Step 5: Update `package-lock.json` if it wasn't committed yet**

The `npm install` from Step 1 created `package-lock.json`. Ensure it's tracked:

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
ls puremacro/tools/pyodide/package-lock.json
```

Expected: file exists.

- [ ] **Step 6: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/tools/pyodide/runner.js puremacro/tools/pyodide/package-lock.json
git commit -m "$(cat <<'EOF'
feat(0.49.0): tools/pyodide/runner.js — Node headless Pyodide harness

~80 LOC. Loads Pyodide via npm, loadPackage's numpy/scipy/pandas/
matplotlib/pytest/micropip, mounts host tests/ -> /mnt/tests via
mountNodeFS, installs the puremacro wheel via micropip, runs
pytest -m pyodide_smoke, emits a JSON envelope to stdout.

Smoke-verified locally: 8 tests pass under Pyodide in ~60-180s after
the one-time npm install. package-lock.json committed for
reproducibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `tools/pyodide_smoke.py` Python wrapper + 8 unit tests

- [ ] **Step 1: Write the failing unit tests**

Create `puremacro/tests/test_pyodide_smoke_runner.py`:

```python
"""Tests for tools/pyodide_smoke.py — the Python wrapper for Gate 6.

Tests monkeypatch subprocess.run to avoid actually booting Pyodide.
The runner's correctness is verified live by Gate 6 itself in
release_check.py.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "pyodide_smoke.py"

_spec = importlib.util.spec_from_file_location("pyodide_smoke", SCRIPT)
pyodide_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pyodide_smoke)


def _ok_json_envelope(returncode=0, passed=8, failed=0, wheel_installed=True):
    return json.dumps({
        "schema_version": 1,
        "pyodide_version": "0.28.3",
        "loaded_at": "2026-05-22T15:00:00Z",
        "wheel_installed": wheel_installed,
        "wheel_path": "/tmp/x.whl",
        "pytest_returncode": returncode,
        "passed": passed,
        "failed": failed,
        "skipped": 0,
        "runtime_s": 84.2,
        "stdout_tail": "8 passed in 12s",
    })


def test_run_pass(tmp_path, monkeypatch):
    """All-pass JSON envelope → gate dict passed=True."""
    # Stub Node + node_modules check + build to succeed.
    def fake_run(cmd, *a, **kw):
        if "node" in cmd[0] and "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="v20.0.0\n", stderr="")
        if "-m" in cmd and "build" in cmd:
            (tmp_path / "puremacro-0.49.0-py3-none-any.whl").write_text("")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "runner.js" in (cmd[1] if len(cmd) > 1 else ""):
            return subprocess.CompletedProcess(cmd, 0, stdout=_ok_json_envelope(), stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(pyodide_smoke.subprocess, "run", fake_run)
    monkeypatch.setattr(pyodide_smoke, "_node_modules_exists", lambda _r: True)
    monkeypatch.setattr(pyodide_smoke, "_build_wheel", lambda _r, _td: tmp_path / "puremacro-0.49.0-py3-none-any.whl")

    r = pyodide_smoke.run(REPO_ROOT)
    assert r["passed"] is True
    assert "8" in r["report"]


def test_run_fail_test(tmp_path, monkeypatch):
    """One failing test → gate dict passed=False."""
    monkeypatch.setattr(pyodide_smoke, "_node_modules_exists", lambda _r: True)
    monkeypatch.setattr(pyodide_smoke, "_check_node", lambda: True)
    monkeypatch.setattr(pyodide_smoke, "_build_wheel", lambda _r, _td: tmp_path / "puremacro.whl")
    def fake_runner(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 0,
            stdout=_ok_json_envelope(returncode=1, passed=7, failed=1),
            stderr="")
    monkeypatch.setattr(pyodide_smoke.subprocess, "run", fake_runner)
    r = pyodide_smoke.run(REPO_ROOT)
    assert r["passed"] is False
    assert "1" in r["report"] and "fail" in r["report"].lower()


def test_run_wheel_install_failed(tmp_path, monkeypatch):
    """wheel_installed=false → gate FAIL with explicit message."""
    monkeypatch.setattr(pyodide_smoke, "_node_modules_exists", lambda _r: True)
    monkeypatch.setattr(pyodide_smoke, "_check_node", lambda: True)
    monkeypatch.setattr(pyodide_smoke, "_build_wheel", lambda _r, _td: tmp_path / "puremacro.whl")
    def fake_runner(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 0,
            stdout=_ok_json_envelope(wheel_installed=False, returncode=-1),
            stderr="")
    monkeypatch.setattr(pyodide_smoke.subprocess, "run", fake_runner)
    r = pyodide_smoke.run(REPO_ROOT)
    assert r["passed"] is False
    assert "wheel install" in r["report"].lower()


def test_run_node_not_installed(monkeypatch):
    """node --version raises FileNotFoundError → gate FAIL with guidance."""
    def fake_run(cmd, *a, **kw):
        if "node" in cmd[0]:
            raise FileNotFoundError("node not found")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(pyodide_smoke.subprocess, "run", fake_run)
    r = pyodide_smoke.run(REPO_ROOT)
    assert r["passed"] is False
    assert "node" in r["report"].lower() and "install" in r["report"].lower()


def test_run_node_modules_missing(tmp_path, monkeypatch):
    """tools/pyodide/node_modules/ missing → gate FAIL with npm install guidance."""
    monkeypatch.setattr(pyodide_smoke, "_node_modules_exists", lambda _r: False)
    monkeypatch.setattr(pyodide_smoke, "_check_node", lambda: True)
    r = pyodide_smoke.run(REPO_ROOT)
    assert r["passed"] is False
    assert "npm install" in r["report"].lower()


def test_run_wheel_build_fails(tmp_path, monkeypatch):
    """python -m build returns non-zero → gate FAIL."""
    monkeypatch.setattr(pyodide_smoke, "_node_modules_exists", lambda _r: True)
    monkeypatch.setattr(pyodide_smoke, "_check_node", lambda: True)
    def fake_build(_r, _td):
        raise RuntimeError("wheel build failed: subprocess exited 1")
    monkeypatch.setattr(pyodide_smoke, "_build_wheel", fake_build)
    r = pyodide_smoke.run(REPO_ROOT)
    assert r["passed"] is False
    assert "wheel build" in r["report"].lower()


def test_run_json_malformed(tmp_path, monkeypatch):
    """Runner emits garbage instead of JSON → gate FAIL."""
    monkeypatch.setattr(pyodide_smoke, "_node_modules_exists", lambda _r: True)
    monkeypatch.setattr(pyodide_smoke, "_check_node", lambda: True)
    monkeypatch.setattr(pyodide_smoke, "_build_wheel", lambda _r, _td: tmp_path / "puremacro.whl")
    def fake_runner(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="not json", stderr="")
    monkeypatch.setattr(pyodide_smoke.subprocess, "run", fake_runner)
    r = pyodide_smoke.run(REPO_ROOT)
    assert r["passed"] is False
    assert "malformed" in r["report"].lower() or "json" in r["report"].lower()


def test_run_timeout(tmp_path, monkeypatch):
    """subprocess.TimeoutExpired on runner.js → gate FAIL."""
    monkeypatch.setattr(pyodide_smoke, "_node_modules_exists", lambda _r: True)
    monkeypatch.setattr(pyodide_smoke, "_check_node", lambda: True)
    monkeypatch.setattr(pyodide_smoke, "_build_wheel", lambda _r, _td: tmp_path / "puremacro.whl")
    def fake_runner(cmd, *a, **kw):
        raise subprocess.TimeoutExpired(cmd, 600)
    monkeypatch.setattr(pyodide_smoke.subprocess, "run", fake_runner)
    r = pyodide_smoke.run(REPO_ROOT)
    assert r["passed"] is False
    assert "timeout" in r["report"].lower()
```

- [ ] **Step 2: Run the tests — should fail (script doesn't exist)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/test_pyodide_smoke_runner.py -v 2>&1 | tail -5
```

Expected: 8 FAILED (collection error: `pyodide_smoke.py` missing).

- [ ] **Step 3: Write `puremacro/tools/pyodide_smoke.py`**

```python
"""tools/pyodide_smoke.py — Python wrapper for Gate 6.

Builds the puremacro wheel via `python -m build`, invokes
`node tools/pyodide/runner.js --wheel <wheel-path>`, parses the JSON
envelope, returns a {passed, report} dict.

See docs/specs/2026-05-22-puremacro-049-pyodide-ci-design.md.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _check_node() -> bool:
    """Return True if `node --version` succeeds."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _node_modules_exists(repo_root: Path) -> bool:
    """Return True if tools/pyodide/node_modules/ exists."""
    return (repo_root / "tools" / "pyodide" / "node_modules").is_dir()


def _build_wheel(repo_root: Path, out_dir: Path) -> Path:
    """Build a puremacro wheel into out_dir; return the wheel path.

    Raises RuntimeError if the build fails.
    """
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "-o", str(out_dir)],
        cwd=str(repo_root),
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-20:])
        raise RuntimeError(f"wheel build failed: returncode={result.returncode}\n{tail}")
    wheels = sorted(out_dir.glob("puremacro-*.whl"))
    if not wheels:
        raise RuntimeError(f"wheel build produced no .whl in {out_dir}")
    return wheels[-1]


def _invoke_runner(repo_root: Path, wheel_path: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    runner_path = repo_root / "tools" / "pyodide" / "runner.js"
    return subprocess.run(
        ["node", str(runner_path), "--wheel", str(wheel_path)],
        cwd=str(repo_root),
        capture_output=True, text=True, timeout=timeout,
    )


def run(repo_root: Path) -> dict:
    """Build the wheel + run the Pyodide smoke. Return {passed, report}.

    Returns a dict matching the existing gate-result contract:
        passed : bool
        report : str — human-readable gate report (matches Gate 1-5 style).
    """
    head = "  Gate 6 (pyodide smoke)"

    if not _check_node():
        return {
            "passed": False,
            "report": (
                f"{head}: FAIL — Node not installed or not on PATH\n"
                "    Install Node.js (>=18) and run `cd tools/pyodide && npm install` once."
            ),
        }

    if not _node_modules_exists(repo_root):
        return {
            "passed": False,
            "report": (
                f"{head}: FAIL — Pyodide not installed under tools/pyodide/\n"
                "    Run `cd tools/pyodide && npm install` (one-time, ~150 MB)."
            ),
        }

    with tempfile.TemporaryDirectory(prefix="puremacro_pyodide_") as td:
        td_path = Path(td)

        try:
            wheel = _build_wheel(repo_root, td_path)
        except RuntimeError as e:
            return {
                "passed": False,
                "report": f"{head}: FAIL — wheel build failed\n    {e}",
            }

        try:
            proc = _invoke_runner(repo_root, wheel)
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "report": f"{head}: FAIL — Pyodide runner exceeded 600s timeout",
            }

        if proc.returncode != 0 and not proc.stdout.strip():
            tail = "\n".join(proc.stderr.splitlines()[-10:])
            return {
                "passed": False,
                "report": (
                    f"{head}: FAIL — Pyodide failed to boot (runner exit {proc.returncode})\n"
                    f"    last stderr lines:\n    {tail}"
                ),
            }

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            stdout_preview = proc.stdout[:500] if proc.stdout else "(empty)"
            return {
                "passed": False,
                "report": (
                    f"{head}: FAIL — malformed runner output\n"
                    f"    json error: {e}\n"
                    f"    stdout preview: {stdout_preview}"
                ),
            }

    # Envelope is valid JSON — inspect contents.
    wheel_ok = envelope.get("wheel_installed", False)
    pytest_rc = envelope.get("pytest_returncode", -1)
    passed_n = envelope.get("passed", 0)
    failed_n = envelope.get("failed", 0)
    skipped_n = envelope.get("skipped", 0)
    runtime_s = envelope.get("runtime_s", 0.0)
    stdout_tail = envelope.get("stdout_tail", "")

    if not wheel_ok:
        return {
            "passed": False,
            "report": (
                f"{head}: FAIL — wheel install failed in Pyodide\n"
                f"    pyodide_version: {envelope.get('pyodide_version', '?')}\n"
                f"    stdout_tail: {stdout_tail}"
            ),
        }

    if pytest_rc != 0:
        return {
            "passed": False,
            "report": (
                f"{head}: FAIL — pytest exit {pytest_rc} in Pyodide "
                f"({passed_n} passed, {failed_n} failed, {skipped_n} skipped)\n"
                f"    stdout_tail: {stdout_tail}"
            ),
        }

    return {
        "passed": True,
        "report": (
            f"{head}: PASS — {passed_n} passed in Pyodide ({runtime_s}s, "
            f"pyodide {envelope.get('pyodide_version', '?')})"
        ),
    }
```

- [ ] **Step 4: Run unit tests — should pass**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest tests/test_pyodide_smoke_runner.py -v
```

Expected: 8 PASSED.

- [ ] **Step 5: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/tools/pyodide_smoke.py puremacro/tests/test_pyodide_smoke_runner.py
git commit -m "$(cat <<'EOF'
feat(0.49.0): tools/pyodide_smoke.py — Python wrapper for Gate 6

Builds wheel via `python -m build`, invokes node runner.js with the
wheel path, parses the JSON envelope, returns {passed, report} dict
matching the gate-result contract. 8 unit tests cover the major
paths: pass, fail-test, wheel-install-failed, no-node, no-node-modules,
build-fails, malformed-json, timeout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: First live Pyodide integration run

This is the moment-of-truth task. Run `tools/pyodide_smoke.py::run(...)` against the real repo + real Pyodide. **Controller-direct** (subagents will time out on a 60-180s subprocess).

- [ ] **Step 1: Smoke-test the Python wrapper end-to-end**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
python -c "
import sys
sys.path.insert(0, 'puremacro/tools')
import pyodide_smoke
from pathlib import Path
r = pyodide_smoke.run(Path('puremacro').resolve())
print('passed:', r['passed'])
print('report:')
print(r['report'])
"
```

Expected: takes 60-180s. Final output:
- `passed: True`
- Report says `Gate 6 (pyodide smoke): PASS — 8 passed in Pyodide (Xs, pyodide 0.28.3)`.

If `passed: False`, root-cause:
- **Wheel build failed**: usually a `pyproject.toml` issue or missing files. Fix and retry.
- **Wheel install failed**: micropip rejected the wheel. Likely a pyproject `dependencies` value Pyodide can't resolve. Check the runner's stderr for details.
- **N tests failed in Pyodide**: those tests use a Pyodide-incompatible API. Swap the @pytest.mark.pyodide_smoke decorator to a sibling test that doesn't have the issue. Update Task 1's commit message if the test ID changes.

- [ ] **Step 2: Document the runtime**

Note the actual `runtime_s` from the live run. Used for the CHANGELOG entry in Task 7 (e.g., "Gate 6 takes ~Ns wall on the maintainer's machine").

- [ ] **Step 3: No commit** — Task 5 is integration verification.

If everything passes, proceed to Task 6. If anything fails, fix the underlying issue (which may mean amending Task 1's marker placement, Task 3's runner.js, or Task 4's wrapper).

---

## Task 6: Gate 6 in `tools/release_check.py`

Wire the wrapper into the existing release-gate, behind a new `--pyodide` flag.

- [ ] **Step 1: Add the Gate 6 unit tests**

Append to `puremacro/tests/test_release_check.py`:

```python
def test_gate6_pass(monkeypatch, tmp_path):
    """gate_pyodide_smoke wraps pyodide_smoke.run; PASS path."""
    # Stub the wrapper so we don't actually boot Pyodide in unit tests.
    import sys
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import pyodide_smoke
    monkeypatch.setattr(
        pyodide_smoke, "run",
        lambda _r: {"passed": True, "report": "  Gate 6 (pyodide smoke): PASS — 8 passed"},
    )
    r = release_check.gate_pyodide_smoke(REPO_ROOT)
    assert r["passed"] is True
    assert r["name"] == "pyodide_smoke"


def test_gate6_fail(monkeypatch):
    """gate_pyodide_smoke surfaces wrapper FAIL."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import pyodide_smoke
    monkeypatch.setattr(
        pyodide_smoke, "run",
        lambda _r: {"passed": False, "report": "  Gate 6 (pyodide smoke): FAIL — 1 failed"},
    )
    r = release_check.gate_pyodide_smoke(REPO_ROOT)
    assert r["passed"] is False
    assert "fail" in r["report"].lower()
    assert r["name"] == "pyodide_smoke"


def test_main_summary_with_pyodide_flag(capsys, monkeypatch):
    """When --examples --pyodide is passed and all 6 gates pass, summary says 6."""
    monkeypatch.setattr(release_check, "gate_test_baseline", lambda _r: {"name": "test_baseline", "passed": True, "report": "  Gate 1: PASS"})
    monkeypatch.setattr(release_check, "gate_pyodide", lambda _r: {"name": "pyodide", "passed": True, "report": "  Gate 2: PASS"})
    monkeypatch.setattr(release_check, "gate_snapshot", lambda _r: {"name": "public_api_snapshot", "passed": True, "report": "  Gate 3: PASS"})
    monkeypatch.setattr(release_check, "gate_version_sync", lambda **kw: {"name": "version_sync", "passed": True, "report": "  Gate 4: PASS"})
    monkeypatch.setattr(
        release_check, "gate_examples_gallery",
        lambda _p, *, examples_source_dir: {"name": "examples_gallery", "passed": True, "report": "  Gate 5: PASS"},
    )
    monkeypatch.setattr(
        release_check, "gate_pyodide_smoke",
        lambda _r: {"name": "pyodide_smoke", "passed": True, "report": "  Gate 6: PASS"},
    )
    rc = release_check.main(["--examples", "--pyodide"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "all 6 gates pass" in captured.out.lower()
```

- [ ] **Step 2: Run tests — should fail (gate_pyodide_smoke not defined)**

```bash
python -m pytest tests/test_release_check.py -v 2>&1 | tail -10
```

Expected: 3 new tests FAIL.

- [ ] **Step 3: Add `gate_pyodide_smoke` to `tools/release_check.py`**

Insert AFTER `gate_examples_gallery` (so the file reads gates 1-6 in order):

```python
def gate_pyodide_smoke(repo_root: Path) -> dict:
    """Gate 6 — real Pyodide smoke.

    Delegates to tools/pyodide_smoke.py::run, which builds the wheel +
    invokes the Node runner. Slow gate (~60-180s); opt-in via --pyodide.
    """
    sys.path.insert(0, str(repo_root / "tools"))
    try:
        import pyodide_smoke
    finally:
        sys.path.pop(0)
    result = pyodide_smoke.run(repo_root)
    return {"name": "pyodide_smoke", **result}
```

- [ ] **Step 4: Add the `--pyodide` flag and wire Gate 6 into `main()`**

In the argparse block (after `--examples`):

```python
    parser.add_argument(
        "--pyodide",
        action="store_true",
        help="Also run Gate 6 (real Pyodide smoke). Builds the wheel + boots "
             "Pyodide via node tools/pyodide/runner.js. Slow (~60-180s); "
             "requires node + one-time `npm install` in tools/pyodide/.",
    )
```

In the gate-collection block, after the Gate 5 (`if args.examples:`) block:

```python
    if args.pyodide:
        g6 = gate_pyodide_smoke(REPO_ROOT)
        gates.append(g6)
```

- [ ] **Step 5: Verify unit tests pass**

```bash
python -m pytest tests/test_release_check.py -v
```

Expected: previously-passing tests + 3 new tests all PASS.

- [ ] **Step 6: Live smoke run — full gate with `--pyodide`**

This takes ~60-180s (Pyodide load + 8 tests). Controller-direct.

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python tools/release_check.py --pyodide 2>&1 | tail -12
```

Expected: all gates PASS at 0.48.0 (Gate 1 + Gate 2 + Gate 3 + Gate 4 + Gate 6). Final line: "all 5 gates PASS" (Gate 5 skipped without --examples).

To run all 6 gates:

```bash
python tools/release_check.py --examples --pyodide 2>&1 | tail -12
```

Expected: all 6 gates PASS. Wall-time ~4-6 min (Gate 1 ~2 min + Gate 6 ~2-3 min).

- [ ] **Step 7: Commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/tools/release_check.py puremacro/tests/test_release_check.py
git commit -m "$(cat <<'EOF'
feat(0.49.0): release_check Gate 6 — real Pyodide smoke

Delegates to tools/pyodide_smoke.py::run. Opt-in via --pyodide flag;
default 5-gate run unchanged (Gate 5 still --examples-opt-in).
3 new tests + 5-gate summary extends to 6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: CONTRIBUTING + 1.0_path tick + bump + CHANGELOG + final gate

- [ ] **Step 1: Update `puremacro/CONTRIBUTING.md`**

In the existing "Before tagging a release" section, after the Gate 5 `--examples` subsection, append:

```markdown

### Opt-in: real Pyodide smoke (Gate 6)

For the strictest pre-tag check including Gate 6 (real Pyodide), add `--pyodide`:

```bash
cd tools/pyodide
npm install     # one-time, ~150 MB
cd ../..
python tools/release_check.py --examples --pyodide
```

The `--pyodide` flag builds the puremacro wheel, boots Pyodide via Node, installs the wheel via `micropip`, and runs `pytest -m pyodide_smoke` (currently 8 tests across 7 subpackages). Slow (~60-180s); requires Node ≥18. Default gate run does not include this.
```

(Use literal triple backticks in the file, not escaped.)

- [ ] **Step 2: Tick the "Real Pyodide CI green" gate in `puremacro/docs/1.0_path.md`**

In `docs/1.0_path.md` § 4, find:

```markdown
- [ ] Real Pyodide CI green. Currently Gate 2 only checks
      `sys.modules`; 1.0 requires actually booting Pyodide,
      installing the wheel, and running at least one replication
      end-to-end. The Pyodide-CI design is the **P3** spec
      (not yet written; queued after 0.48.0).
```

Replace with:

```markdown
- [x] Real Pyodide CI green. Shipped at 0.49.0 as Gate 6 in
      `tools/release_check.py` (opt-in via `--pyodide`). Boots Pyodide
      via Node, installs the freshly-built wheel via micropip, runs
      `pytest -m pyodide_smoke` (8 curated tests across 7 subpackages).
      At 1.0 this becomes mandatory (not opt-in).
```

- [ ] **Step 3: Bump three version strings**

- `puremacro/pyproject.toml`: `version = "0.48.0"` → `"0.49.0"`.
- `puremacro/puremacro/__init__.py`: `__version__ = "0.48.0"` → `"0.49.0"`.
- `puremacro/tests/test_import.py`: `assert puremacro.__version__ == "0.48.0"` → `"0.49.0"`.

- [ ] **Step 4: Prepend CHANGELOG 0.49.0 entry**

Edit `puremacro/CHANGELOG.md` to insert after the `# Changelog` preamble and before the existing `## 0.48.0 — 2026-05-22` heading:

```markdown
## 0.49.0 — 2026-05-22

Real Pyodide CI (Gate 6).

Closes one of the seven 1.0-blocking gates documented in
`docs/1.0_path.md` § 4. Gate 6 builds the puremacro wheel, boots
Pyodide via Node (npm-installed, pinned at 0.28.3), installs the
wheel via `micropip`, mounts the host `tests/` directory into
Pyodide's FS, and runs `pytest -m pyodide_smoke` on 8 curated tests
across 7 subpackages. Opt-in via `--pyodide`; default gate run
unchanged.

### Added
- `@pytest.mark.pyodide_smoke` marker — declared in
  `pyproject.toml::[tool.pytest.ini_options].markers` alongside
  `network`. Applied to 8 tests across 7 subpackages: `cholesky`,
  `proxy axis`, `lp jorda`, `inference HAC fixed-b`, `klein
  closed-form path`, `sigma object`, `gar skew-t`, `cycles`.
- `tools/pyodide/package.json` — declares `pyodide` npm dep pinned
  at `0.28.3` (exact, no `^`). Bumping is a deliberate maintainer
  act.
- `tools/pyodide/runner.js` — ~80 LOC Node script that loads Pyodide,
  `loadPackage`'s the standard scientific stack + pytest + micropip,
  mounts the host `tests/` directory via `mountNodeFS`, installs the
  wheel via `micropip`, runs the marked pytest subset, emits a JSON
  envelope to stdout.
- `tools/pyodide/README.md` — one-time setup notes (`npm install`,
  Node ≥18, version-pin policy, JSON envelope contract).
- `tools/pyodide_smoke.py` — Python wrapper (~150 LOC) that builds
  the wheel via `python -m build`, invokes `node runner.js`, parses
  JSON, returns the gate-result contract dict.
- `tools/release_check.py` Gate 6 + `--pyodide` flag — opt-in slow
  gate; default 5-gate run unchanged.
- `puremacro/tests/test_pyodide_smoke_runner.py` — 8 unit tests for
  the Python wrapper (pass / fail-test / wheel-install-failed /
  no-node / no-node-modules / build-fails / malformed-json /
  timeout).

### Changed
- `docs/1.0_path.md` § 4 — "Real Pyodide CI green" gate ticked
  (`[x]`). One of seven 1.0-blocking gates now satisfied.
- `CONTRIBUTING.md` "Before tagging a release" subsection now
  documents the `--pyodide` opt-in workflow.

### Internal
- `tools/pyodide/.gitignore` excludes `node_modules/` (~150 MB
  after `npm install`).
- `tools/pyodide/package-lock.json` committed for reproducibility.
- `tools/release_check.py::main()` argparse adds `--pyodide`;
  default 4-gate / 5-gate runs unchanged.

### Out of scope (deferred to follow-on specs)
- PyPI publishing (still queued; gates 1.0 separately).
- Bayesian DSGE estimation (P1 pitch).
- Mixed-frequency BVAR (P2), numba JIT (P6) — not picked.
- Browser-Pyodide via Playwright — npm-Pyodide is the same runtime;
  tests passing here mean tests pass on iPad / juno.sh.

---
```

- [ ] **Step 5: Pre-commit gate verification**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python tools/release_check.py --examples --pyodide
```

Expected: all 6 gates PASS at 0.49.0, exit 0. Wall-time ~4-6 min.

If Gate 3 (public API snapshot) flags a diff, investigate WHY — the implementation should not have changed any public API. Most likely cause would be a new module accidentally exported.

- [ ] **Step 6: Commit the bump**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples"
git add puremacro/pyproject.toml puremacro/puremacro/__init__.py puremacro/tests/test_import.py puremacro/CHANGELOG.md puremacro/CONTRIBUTING.md puremacro/docs/1.0_path.md
git commit -m "$(cat <<'EOF'
chore(puremacro): bump 0.48.0 → 0.49.0 (real Pyodide CI ships)

Three version strings synced. CHANGELOG 0.49.0 entry covers Gate 6
(real Pyodide via npm-installed Pyodide + node runner + wheel build
+ pytest -m pyodide_smoke). docs/1.0_path.md § 4 ticks one of the
seven 1.0-blocking gates. CONTRIBUTING.md documents --pyodide opt-in.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Final gate run on the bump commit**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python tools/release_check.py --examples --pyodide
```

Expected: exit 0, all 6 gates PASS at 0.49.0.

- [ ] **Step 8: Hand off to the controller for merge + tag**

Do NOT auto-merge or tag. Report HEAD SHA + gate output; controller asks the user about tag + merge (same pattern as 0.46.0 / 0.47.0 / 0.48.0).

---

## Self-review notes

**Spec coverage** (from `docs/specs/2026-05-22-puremacro-049-pyodide-ci-design.md`):

- `pyodide_smoke` marker — Task 1 ✓
- 8 tests tagged — Task 1 ✓
- `tools/pyodide/` (package.json, runner.js, README, .gitignore) — Tasks 2-3 ✓
- `package-lock.json` committed — Task 3 ✓
- `tools/pyodide_smoke.py` — Task 4 ✓
- Gate 6 + `--pyodide` flag — Task 6 ✓
- Unit tests for Python wrapper — Task 4 (8 tests) ✓
- Unit tests for Gate 6 — Task 6 (3 tests) ✓
- Live Gate 6 PASS on maintainer machine — Task 5 (integration) + Task 7 Step 5 (final) ✓
- `docs/1.0_path.md` § 4 ticked — Task 7 Step 2 ✓
- CONTRIBUTING.md updated — Task 7 Step 1 ✓
- CHANGELOG 0.49.0 entry — Task 7 Step 4 ✓
- Version bump — Task 7 Step 3 ✓

**Placeholder scan:**
- Task 1 Step 2 has `pick one fast test function` — this is the genuine "audit then decide" branch (similar to 0.46.0's known-failures discovery). Not a TBD; explicit procedure.
- Task 3 Step 3's runner.js note "If `emfs:` doesn't work, fall back to copying the wheel into `/tmp/`" — documents a recoverable branch, not a placeholder.
- No other TBD / TODO / "implement later" patterns.

**Type consistency:**
- `pyodide_smoke.run(repo_root) -> dict` consistent across Tasks 4 (definition) and 6 (delegation).
- Gate-result contract `{"name": str, "passed": bool, "report": str}` consistent with existing Gates 1-5.
- `runner.js` JSON envelope schema declared in Task 2 README, locked in Task 3 implementation, consumed in Task 4 wrapper, tested in Task 4 unit tests.
- `--pyodide` and `--examples` flags both `action="store_true"`, defaulting OFF — consistent argparse pattern.

**Risks pulled forward from the spec:**
- **R1 (Pyodide version drift):** mitigated by exact pin in `package.json` + README warning. Task 3 Step 2 verifies the pinned version loads correctly.
- **R2 (curated test incompatible with Pyodide):** Task 5 (first live run) surfaces this. If a test fails under Pyodide, swap to a sibling and revisit Task 1.
- **R3 (wheel-packaging bug):** Task 5's first wheel build catches packaging issues; if found, fix in 0.49.0 scope.
- **R4 (forgetful maintainer skips Gate 6):** documented in CONTRIBUTING.md + 1.0_path.md; relies on maintainer discipline same as Gates 5+6 are opt-in.
- **R5 (Node dependency drag):** wrapper's actionable error messages catch missing Node + missing `node_modules/`.
- **R6 (node_modules/ heavy):** gitignored; one-time download.
- **R7 (Pyodide rejects the wheel):** Task 5 surfaces this; mitigation is to fix `pyproject.toml` / pyproject `dependencies` / wheel metadata.

**Out of scope (deferred):**
- PyPI publishing — separate spec.
- Browser-Pyodide via Playwright — npm-Pyodide is sufficient.
- Hosted CI / GitHub Actions — package's "no CI by design" stance stands.
- P1 (Bayesian DSGE estimation) — biggest remaining research-value pitch, queued after 0.49.0 ships.
