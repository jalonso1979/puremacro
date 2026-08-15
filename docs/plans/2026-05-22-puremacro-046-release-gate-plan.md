# puremacro 0.46.0 — Release-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `tools/release_check.py` + `tests/known_failures.json` so the maintainer can run a single command before `git tag` that catches the four failure modes that bit 0.41→0.42: stale `pyproject.toml::version`, regressed-but-tolerated tests, Pyodide-contract violations, and unsynced public-API snapshot. No `puremacro/*.py` files modified. Tag as **0.46.0**.

**Architecture:** Single-file ~150 LOC script in `tools/`, pure-stdlib + pytest. Four independent gates (`run_gate_*` functions), each returning a structured result; main aggregates and exits non-zero if any failed. Pure helpers (version parsing, failure-set comparison, snapshot diffing) live at module level so they're unit-testable. The script reads but never writes to repo files except by maintainer's explicit re-snapshot command.

**Tech Stack:** Python ≥3.10, pytest (existing dev dep), `re` + `json` + `pathlib` + `subprocess` from stdlib. No new dependencies.

**Source spec:** `docs/specs/2026-05-22-puremacro-046-047-consolidate-finish-design.md` (commit `a2065ff`).

**Pre-execution state (HEAD `4b9d851`):**
- 0.45.0 shipped, on `main`.
- `pyproject.toml::version` = `"0.45.0"`.
- `puremacro/__init__.py::__version__` = `"0.45.0"`.
- `tests/test_import.py` asserts `puremacro.__version__ == "0.44.0"` — **stale; will appear in baseline failures.** Resolved by Task 10's version bump.
- CHANGELOG.md's first `## X.Y.Z` heading: `## 0.45.0 — 2026-05-21`.
- `tests/fixtures/public_api_snapshot.json` regenerated at commit `4b9d851`.
- Three "named failing" test files run clean in isolation (118 passed, 6 skipped, 0 failed @ 2026-05-22). Full-suite red set unknown until Task 1's baseline runs.

---

## File structure

**Created:**
- `tools/release_check.py` — main script + pure helpers + `__main__`.
- `tests/test_release_check.py` — unit tests for pure helpers.
- `tests/known_failures.json` — seeded with Task 1's baseline.

**Modified:**
- `puremacro/__init__.py` — bump `__version__` to `"0.46.0"` (Task 10).
- `pyproject.toml` — bump `version` to `"0.46.0"` (Task 10).
- `tests/test_import.py` — update hardcoded assertion to `"0.46.0"` (Task 10).
- `CONTRIBUTING.md` — append "Before tagging a release" subsection citing `release_check.py` (Task 9).
- `CHANGELOG.md` — prepend `## 0.46.0 — 2026-05-22` entry (Task 10).

**Untouched:** every file under `puremacro/` except `__init__.py`. No estimator, no public API change.

---

## Task 0: Pre-flight + branch creation

Establish clean state. No code yet.

- [ ] **Step 1: Verify clean working tree (modulo Drive-sync detritus)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
git status --short
```

Expected: untracked + modified files in `../docs/paper/` and `../notebooks/output_*` are fine (pre-existing). No modifications to `puremacro/`, `tests/`, `tools/`, or `pyproject.toml`. If there are, pause and check with the user.

- [ ] **Step 2: Confirm HEAD is on the branch carrying 0.45.0**

```bash
git log --oneline -1
```

Expected: `4b9d851 chore(tests): regenerate public_api_snapshot for 0.45.0 additions` (or later if work has continued, in which case adjust).

- [ ] **Step 3: Confirm pre-existing 0.45.0 state**

```bash
grep '^version' pyproject.toml
grep '^__version__' puremacro/__init__.py
head -2 CHANGELOG.md
```

Expected:
- `version = "0.45.0"`
- `__version__ = "0.45.0"`
- `# Changelog` then blank-or-comment line.

- [ ] **Step 4: Create release branch**

```bash
git checkout -b release/0.46.0
```

Expected: switched to new branch.

- [ ] **Step 5: Commit nothing yet**

This task is read-only. No commit.

---

## Task 1: Baseline the red set

Discover the full-suite failing-test set on `release/0.46.0` HEAD. Long-running (~15–25 minutes for the full suite; 472s for 3 files alone). **Run from the controller's Bash with `run_in_background: true`** (per memory pin `feedback_long_nbconvert_no_subagent` — subagents time out on Monitor-wait).

- [ ] **Step 1: Run full suite (background)**

```bash
cd "/Users/jalonso/Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/uncertainty_examples/puremacro"
python -m pytest puremacro/tests/ tests/ -m "not network" --tb=no -q 2>&1 | tee /tmp/release_check_baseline.txt
```

Use `run_in_background: true` on the Bash call. Expected wall time: 15–25 min. You'll be notified when it completes — do not poll.

- [ ] **Step 2: Inspect the failure summary**

```bash
tail -50 /tmp/release_check_baseline.txt
```

Expected: pytest summary line like `N passed, M failed, K skipped`. Note the exact counts. If M == 0, the whitelist starts empty (best case). If M > 0, continue to Step 3.

- [ ] **Step 3: Extract failing nodeids**

```bash
grep -E "^FAILED " /tmp/release_check_baseline.txt | awk '{print $2}' | sort > /tmp/release_check_failing_nodeids.txt
wc -l /tmp/release_check_failing_nodeids.txt
cat /tmp/release_check_failing_nodeids.txt
```

Expected: one nodeid per line, total matching the `M failed` count from Step 2.

- [ ] **Step 4: Verify `tests/test_import.py::test_import_puremacro` is in the list**

```bash
grep test_import /tmp/release_check_failing_nodeids.txt
```

Expected: `tests/test_import.py::test_import_puremacro` present (it asserts `__version__ == "0.44.0"` against actual `"0.45.0"`). This entry will be removed by Task 10's bump → assertion update.

- [ ] **Step 5: Commit nothing yet**

The baseline is captured in `/tmp/`; it's an input to Task 2.

---

## Task 2: Create `tests/known_failures.json` with baseline + schema test

Seed the whitelist file. Add a tiny schema validator so future edits can't accidentally drop required fields.

- [ ] **Step 1: Write the failing schema test**

Create `tests/test_known_failures_schema.py`:

```python
"""Schema test for tests/known_failures.json — locks the contract used by tools/release_check.py."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATH = REPO_ROOT / "tests" / "known_failures.json"


def test_known_failures_file_exists():
    assert PATH.exists(), f"{PATH} must exist (tools/release_check.py reads it)"


def test_known_failures_top_level_shape():
    data = json.loads(PATH.read_text())
    assert isinstance(data, dict)
    assert data.get("schema_version") == 1
    assert isinstance(data.get("entries"), list)


def test_known_failures_entry_fields():
    data = json.loads(PATH.read_text())
    required = {"nodeid", "reason", "since_version", "owner_note"}
    for i, entry in enumerate(data["entries"]):
        missing = required - set(entry.keys())
        assert not missing, f"entry {i} missing fields: {missing}"
        assert isinstance(entry["nodeid"], str) and entry["nodeid"]
        assert isinstance(entry["reason"], str) and entry["reason"]
        assert isinstance(entry["since_version"], str) and entry["since_version"]
        assert isinstance(entry["owner_note"], str) and entry["owner_note"]


def test_known_failures_nodeids_unique():
    data = json.loads(PATH.read_text())
    nodeids = [e["nodeid"] for e in data["entries"]]
    assert len(nodeids) == len(set(nodeids)), "duplicate nodeid in known_failures.json"
```

- [ ] **Step 2: Run the schema test to verify it fails on missing file**

```bash
python -m pytest tests/test_known_failures_schema.py -v
```

Expected: 4 FAILED (file does not exist yet).

- [ ] **Step 3: Create `tests/known_failures.json` with Task 1's baseline**

Use the failing-nodeids list from Task 1 Step 3. For each entry, fill `reason` / `since_version` / `owner_note` with placeholder-but-honest values that survive lint. Example pattern below — the exact entry list depends on Task 1's discovery.

```json
{
  "schema_version": 1,
  "entries": [
    {
      "nodeid": "tests/test_import.py::test_import_puremacro",
      "reason": "asserts __version__ == 0.44.0 against actual 0.45.0",
      "since_version": "0.45.0",
      "owner_note": "resolved at 0.46.0 Task 10 when test_import.py is bumped to 0.46.0"
    }
  ]
}
```

**Required:** populate one `entries` element per nodeid from `/tmp/release_check_failing_nodeids.txt`. For nodeids whose root cause is unknown at write time, set `reason: "TBD — root-caused at 0.46.1"` and `owner_note: "Task B-N at 0.46.1"`. (This is the **only** place `TBD` is acceptable in this plan — it's documenting an unknown, not deferring task content.)

- [ ] **Step 4: Run the schema test to verify it passes**

```bash
python -m pytest tests/test_known_failures_schema.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Run the existing test_import to confirm baseline state**

```bash
python -m pytest tests/test_import.py -v
```

Expected: 1 failed (assertion `0.44.0 == 0.45.0`). Confirms the whitelist entry corresponds to a real failure.

- [ ] **Step 6: Commit**

```bash
git add tests/known_failures.json tests/test_known_failures_schema.py
git commit -m "$(cat <<'EOF'
feat(0.46.0): seed tests/known_failures.json with full-suite baseline

Captures the existing red-test set as of release/0.46.0 HEAD. Each entry
has reason, since_version, owner_note. Schema test locks the contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Scaffold `tools/release_check.py` with CLI skeleton

Create the script as a no-gate harness first. Tests assert exit behavior; gates come in Tasks 4–7.

- [ ] **Step 1: Write the failing skeleton tests**

Create `tests/test_release_check.py`:

```python
"""Tests for tools/release_check.py. Imports the module as a script-like module."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "release_check.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_script_exists():
    assert SCRIPT.exists(), f"{SCRIPT} must exist"


def test_help_flag():
    res = _run("--help")
    assert res.returncode == 0
    assert "release-gate" in res.stdout.lower() or "release check" in res.stdout.lower()


def test_default_run_emits_report_header():
    res = _run("--report-only")
    # --report-only short-circuits actual gate execution; still must print the header.
    assert "release_check" in res.stdout.lower() or "gate" in res.stdout.lower()
```

- [ ] **Step 2: Run tests to verify they fail (missing script)**

```bash
python -m pytest tests/test_release_check.py -v
```

Expected: 3 FAILED (`tools/release_check.py` does not exist).

- [ ] **Step 3: Create `tools/release_check.py` skeleton**

```python
"""tools/release_check.py — single pre-tag gate for puremacro releases.

Run me before `git tag X.Y.Z`. Four gates (all run; no fail-fast):

  Gate 1 (test baseline)   — pytest failures must equal tests/known_failures.json
  Gate 2 (Pyodide contract) — tests/test_pyodide_compat.py must be green
  Gate 3 (public API snap)  — re-generated snapshot must match the fixture
  Gate 4 (version sync)     — pyproject.toml / __init__.py / CHANGELOG agree

Exit 0 iff every gate passed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="release_check",
        description="Release-gate for puremacro — runs four gates pre-tag.",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip Gate 1 (test baseline). Useful for fast iteration on the other gates.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print which gates would run and exit 0, without executing any.",
    )
    args = parser.parse_args(argv)

    print("release_check — puremacro pre-tag gate")
    print(f"  repo root: {REPO_ROOT}")
    if args.report_only:
        print("  (report-only — no gates executed)")
        return 0

    # Gates land in Tasks 4–7.
    print("  (no gates implemented yet — wire-up tasks pending)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_release_check.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Smoke-run the script manually**

```bash
python tools/release_check.py --report-only
python tools/release_check.py --help
```

Expected: both exit 0, output matches the script.

- [ ] **Step 6: Commit**

```bash
git add tools/release_check.py tests/test_release_check.py
git commit -m "$(cat <<'EOF'
feat(0.46.0): scaffold tools/release_check.py CLI + smoke tests

No gates implemented yet — just argparse + report header. Gates land
in Tasks 4 through 7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement Gate 4 (version sync)

Simplest gate: pure file reading + string compare. Implement first to lock the gate-result data shape that later gates reuse.

- [ ] **Step 1: Write the failing helper tests**

Add to `tests/test_release_check.py`:

```python
import importlib.util
import textwrap

_spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
release_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_check)


def test_read_pyproject_version(tmp_path):
    f = tmp_path / "pyproject.toml"
    f.write_text(textwrap.dedent("""
        [build-system]
        requires = ["setuptools"]
        [project]
        name = "x"
        version = "1.2.3"
    """))
    assert release_check.read_pyproject_version(f) == "1.2.3"


def test_read_init_version(tmp_path):
    f = tmp_path / "__init__.py"
    f.write_text('__version__ = "9.8.7"\n')
    assert release_check.read_init_version(f) == "9.8.7"


def test_read_changelog_version(tmp_path):
    f = tmp_path / "CHANGELOG.md"
    f.write_text(textwrap.dedent("""
        # Changelog

        ## 0.46.0 — 2026-05-22
        Some content.

        ## 0.45.0 — 2026-05-21
        Older.
    """))
    assert release_check.read_changelog_version(f) == "0.46.0"


def test_gate4_version_sync_pass():
    r = release_check.gate_version_sync(
        pyproject_version="0.46.0",
        init_version="0.46.0",
        changelog_version="0.46.0",
    )
    assert r["passed"] is True


def test_gate4_version_sync_fail():
    r = release_check.gate_version_sync(
        pyproject_version="0.12.1",
        init_version="0.46.0",
        changelog_version="0.46.0",
    )
    assert r["passed"] is False
    assert "0.12.1" in r["report"]
    assert "0.46.0" in r["report"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_release_check.py -v
```

Expected: 5 new tests FAILED (helpers + gate function not defined).

- [ ] **Step 3: Add helpers + gate function to `tools/release_check.py`**

Insert above `main()`:

```python
import re

def read_pyproject_version(path: Path) -> str:
    text = Path(path).read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"version not found in {path}")
    return m.group(1)


def read_init_version(path: Path) -> str:
    text = Path(path).read_text()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"__version__ not found in {path}")
    return m.group(1)


def read_changelog_version(path: Path) -> str:
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^##\s+(\d+\.\d+\.\d+)", line)
        if m:
            return m.group(1)
    raise RuntimeError(f"no '## X.Y.Z' heading in {path}")


def gate_version_sync(
    *,
    pyproject_version: str,
    init_version: str,
    changelog_version: str,
) -> dict:
    versions = {
        "pyproject.toml": pyproject_version,
        "puremacro/__init__.py": init_version,
        "CHANGELOG.md": changelog_version,
    }
    passed = len(set(versions.values())) == 1
    if passed:
        report = f"  Gate 4 (version sync): PASS — all read {pyproject_version}"
    else:
        lines = ["  Gate 4 (version sync): FAIL"]
        for name, ver in versions.items():
            lines.append(f"    {name:<30} {ver}")
        report = "\n".join(lines)
    return {"name": "version_sync", "passed": passed, "report": report}
```

- [ ] **Step 4: Wire Gate 4 into `main()`**

Replace the `(no gates implemented yet ...)` line with:

```python
    gates = []
    g4 = gate_version_sync(
        pyproject_version=read_pyproject_version(REPO_ROOT / "pyproject.toml"),
        init_version=read_init_version(REPO_ROOT / "puremacro" / "__init__.py"),
        changelog_version=read_changelog_version(REPO_ROOT / "CHANGELOG.md"),
    )
    gates.append(g4)

    for g in gates:
        print(g["report"])

    return 0 if all(g["passed"] for g in gates) else 1
```

- [ ] **Step 5: Run helper tests to verify they pass**

```bash
python -m pytest tests/test_release_check.py -v
```

Expected: 8 PASSED (3 skeleton + 5 new).

- [ ] **Step 6: Run the script live**

```bash
python tools/release_check.py
echo "exit: $?"
```

Expected (current main state): Gate 4 reads `0.45.0` from all three files (assuming Task 10 hasn't bumped yet) → PASS, exit 0. If the three files disagree, the gate FAILs and the report lists which is mismatched.

- [ ] **Step 7: Commit**

```bash
git add tools/release_check.py tests/test_release_check.py
git commit -m "$(cat <<'EOF'
feat(0.46.0): release_check Gate 4 — version sync

Parses pyproject.toml + __init__.py + CHANGELOG first heading,
asserts byte-equal. Catches the 0.41→0.42 pyproject staleness pattern.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement Gate 1 (test baseline vs known_failures.json)

Most behavior-rich gate. Pure comparison logic + thin pytest-runner wrapper.

- [ ] **Step 1: Write the failing comparison tests**

Add to `tests/test_release_check.py`:

```python
def test_gate1_compare_exact_match():
    whitelist = {"a::test_x", "b::test_y"}
    failing = {"a::test_x", "b::test_y"}
    r = release_check.compare_failures(failing, whitelist)
    assert r["passed"] is True
    assert r["new"] == set()
    assert r["recovered"] == set()


def test_gate1_compare_new_failure():
    whitelist = {"a::test_x"}
    failing = {"a::test_x", "c::test_z"}
    r = release_check.compare_failures(failing, whitelist)
    assert r["passed"] is False
    assert r["new"] == {"c::test_z"}
    assert r["recovered"] == set()


def test_gate1_compare_recovered_test():
    whitelist = {"a::test_x", "b::test_y"}
    failing = {"a::test_x"}
    r = release_check.compare_failures(failing, whitelist)
    # Recovered is a warning, not a failure — gate still passes.
    assert r["passed"] is True
    assert r["new"] == set()
    assert r["recovered"] == {"b::test_y"}


def test_gate1_load_whitelist(tmp_path):
    f = tmp_path / "kf.json"
    f.write_text('{"schema_version": 1, "entries": [{"nodeid": "a::test_x", "reason": "x", "since_version": "0.1.0", "owner_note": "x"}]}')
    assert release_check.load_whitelist(f) == {"a::test_x"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_release_check.py::test_gate1_compare_exact_match tests/test_release_check.py::test_gate1_compare_new_failure tests/test_release_check.py::test_gate1_compare_recovered_test tests/test_release_check.py::test_gate1_load_whitelist -v
```

Expected: 4 FAILED.

- [ ] **Step 3: Add comparison helpers + gate runner to `tools/release_check.py`**

Insert above `gate_version_sync`:

```python
import json
import subprocess


def load_whitelist(path: Path) -> set[str]:
    data = json.loads(Path(path).read_text())
    return {e["nodeid"] for e in data.get("entries", [])}


def compare_failures(failing: set[str], whitelist: set[str]) -> dict:
    new = failing - whitelist
    recovered = whitelist - failing
    passed = len(new) == 0
    lines = []
    if passed and not recovered:
        lines.append(f"  Gate 1 (test baseline): PASS — {len(failing)} known red, no new")
    elif passed and recovered:
        lines.append(
            f"  Gate 1 (test baseline): PASS with warning — "
            f"{len(recovered)} previously-red now green; shrink whitelist."
        )
        for nid in sorted(recovered):
            lines.append(f"    recovered: {nid}")
    else:
        lines.append(f"  Gate 1 (test baseline): FAIL — {len(new)} new failure(s)")
        for nid in sorted(new):
            lines.append(f"    NEW: {nid}")
        if recovered:
            lines.append(f"    plus {len(recovered)} previously-red now green:")
            for nid in sorted(recovered):
                lines.append(f"    recovered: {nid}")
    return {
        "name": "test_baseline",
        "passed": passed,
        "report": "\n".join(lines),
        "new": new,
        "recovered": recovered,
    }


def run_pytest_collect_failures(repo_root: Path) -> set[str]:
    """Run the full suite (minus network) and return the failing-nodeid set."""
    cmd = [
        sys.executable, "-m", "pytest",
        "puremacro/tests/", "tests/",
        "-m", "not network",
        "--tb=no", "-q",
        "--no-header",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    failures = set()
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED "):
            nid = line[len("FAILED "):].split(" ", 1)[0]
            failures.add(nid)
    return failures


def gate_test_baseline(repo_root: Path) -> dict:
    whitelist = load_whitelist(repo_root / "tests" / "known_failures.json")
    failing = run_pytest_collect_failures(repo_root)
    return compare_failures(failing, whitelist)
```

- [ ] **Step 4: Wire Gate 1 into `main()` ahead of Gate 4**

```python
    gates = []
    if not args.no_tests:
        g1 = gate_test_baseline(REPO_ROOT)
        gates.append(g1)
    g4 = gate_version_sync(
        pyproject_version=read_pyproject_version(REPO_ROOT / "pyproject.toml"),
        init_version=read_init_version(REPO_ROOT / "puremacro" / "__init__.py"),
        changelog_version=read_changelog_version(REPO_ROOT / "CHANGELOG.md"),
    )
    gates.append(g4)
```

- [ ] **Step 5: Run helper tests to verify they pass**

```bash
python -m pytest tests/test_release_check.py -v
```

Expected: 12 PASSED (8 prior + 4 new).

- [ ] **Step 6: Smoke-run with `--no-tests` and without**

```bash
python tools/release_check.py --no-tests
echo "no-tests exit: $?"
# Full run — slow (~15–25 min). Run in background if iterating.
python tools/release_check.py
echo "full exit: $?"
```

Expected: `--no-tests` returns instantly with Gate 4 only. Full run executes pytest and reports Gate 1 vs. whitelist; should exit 0 because the whitelist was seeded from this exact baseline.

- [ ] **Step 7: Commit**

```bash
git add tools/release_check.py tests/test_release_check.py
git commit -m "$(cat <<'EOF'
feat(0.46.0): release_check Gate 1 — test baseline vs whitelist

Runs pytest collecting failing nodeids, compares to tests/known_failures.json.
New failures → exit 1. Previously-red-now-green → pass with warning.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Implement Gate 2 (Pyodide contract)

Thinnest gate: delegate to the existing `tests/test_pyodide_compat.py`. No new comparison logic.

- [ ] **Step 1: Write the failing gate test**

Add to `tests/test_release_check.py`:

```python
def test_gate2_pyodide_passthrough_pass(monkeypatch):
    # Smoke: gate_pyodide returns a dict with 'passed' / 'report'.
    r = release_check.gate_pyodide(REPO_ROOT)
    assert "passed" in r
    assert "report" in r
    assert r["name"] == "pyodide"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_release_check.py::test_gate2_pyodide_passthrough_pass -v
```

Expected: FAILED.

- [ ] **Step 3: Add `gate_pyodide` to `tools/release_check.py`**

Insert before `gate_test_baseline`:

```python
def gate_pyodide(repo_root: Path) -> dict:
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_pyodide_compat.py",
        "--tb=short", "-q",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    passed = proc.returncode == 0
    head = "  Gate 2 (Pyodide contract): " + ("PASS" if passed else "FAIL")
    tail = "" if passed else "\n" + "\n".join(
        f"    {line}" for line in proc.stdout.splitlines()[-20:]
    )
    return {"name": "pyodide", "passed": passed, "report": head + tail}
```

- [ ] **Step 4: Wire Gate 2 into `main()`**

Insert after Gate 1 / before Gate 4:

```python
    g2 = gate_pyodide(REPO_ROOT)
    gates.append(g2)
```

- [ ] **Step 5: Run helper test + smoke**

```bash
python -m pytest tests/test_release_check.py::test_gate2_pyodide_passthrough_pass -v
python tools/release_check.py --no-tests
```

Expected: helper test PASSED. Smoke-run prints Gate 2 result (should be PASS because 0.41.0 restored the Pyodide contract).

- [ ] **Step 6: Commit**

```bash
git add tools/release_check.py tests/test_release_check.py
git commit -m "$(cat <<'EOF'
feat(0.46.0): release_check Gate 2 — Pyodide contract passthrough

Delegates to tests/test_pyodide_compat.py. Surfaces any forbidden-import
regression at release time, not just at test time.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Implement Gate 3 (public API snapshot diff)

Regenerate the snapshot using the same code path `tests/test_public_api.py` uses, then diff against the on-disk fixture. The gate **never** writes to the fixture — the maintainer regenerates explicitly when intentional.

- [ ] **Step 1: Promote `tests/test_public_api.py::_collect_current_api` to a public name**

The existing test file has `_collect_current_api()` (returns the snapshot dict) and `_walk_subpackages()` (helper). Rename `_collect_current_api` → `collect_current_api` (drop the underscore) and update the single in-file reference in `test_public_api_matches_snapshot()`. Do not touch `_walk_subpackages` — it stays private.

```bash
grep -n "_collect_current_api\|collect_current_api" tests/test_public_api.py
```

Expected after rename: two references to `collect_current_api` (definition + test caller), zero to `_collect_current_api`.

- [ ] **Step 2: Write the failing gate test**

Add to `tests/test_release_check.py`:

```python
def test_gate3_snapshot_equal(tmp_path, monkeypatch):
    snap = {"all": {"puremacro._linalg": ["inv_xtx", "safe_cholesky"]}}
    f = tmp_path / "snap.json"
    f.write_text(json.dumps(snap))
    r = release_check.compare_snapshot(snap, f)
    assert r["passed"] is True


def test_gate3_snapshot_diff(tmp_path):
    on_disk = {"all": {"puremacro._linalg": ["inv_xtx"]}}
    fresh = {"all": {"puremacro._linalg": ["inv_xtx", "safe_cholesky"]}}
    f = tmp_path / "snap.json"
    f.write_text(json.dumps(on_disk))
    r = release_check.compare_snapshot(fresh, f)
    assert r["passed"] is False
    assert "safe_cholesky" in r["report"]
```

Note `import json` is already at module top in the test file (added in Task 5). Otherwise add it.

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_release_check.py::test_gate3_snapshot_equal tests/test_release_check.py::test_gate3_snapshot_diff -v
```

Expected: 2 FAILED.

- [ ] **Step 4: Add `compare_snapshot` + `gate_snapshot` to `tools/release_check.py`**

```python
def compare_snapshot(fresh: dict, fixture_path: Path) -> dict:
    on_disk = json.loads(Path(fixture_path).read_text())
    if fresh == on_disk:
        return {
            "name": "public_api_snapshot",
            "passed": True,
            "report": "  Gate 3 (public API snapshot): PASS",
        }
    lines = ["  Gate 3 (public API snapshot): FAIL — diff:"]
    fresh_modules = set(fresh.get("all", {}).keys())
    disk_modules = set(on_disk.get("all", {}).keys())
    added_modules = fresh_modules - disk_modules
    removed_modules = disk_modules - fresh_modules
    for mod in sorted(added_modules):
        lines.append(f"    + module {mod}")
    for mod in sorted(removed_modules):
        lines.append(f"    - module {mod}")
    for mod in sorted(fresh_modules & disk_modules):
        added = set(fresh["all"][mod]) - set(on_disk["all"][mod])
        removed = set(on_disk["all"][mod]) - set(fresh["all"][mod])
        for sym in sorted(added):
            lines.append(f"    + {mod}.{sym}")
        for sym in sorted(removed):
            lines.append(f"    - {mod}.{sym}")
    return {
        "name": "public_api_snapshot",
        "passed": False,
        "report": "\n".join(lines),
    }


def gate_snapshot(repo_root: Path) -> dict:
    # Import the helper from tests/ — Task 7 Step 1 promoted it to a public name.
    sys.path.insert(0, str(repo_root / "tests"))
    try:
        from test_public_api import collect_current_api  # type: ignore
    finally:
        sys.path.pop(0)
    fresh = collect_current_api()
    return compare_snapshot(fresh, repo_root / "tests" / "fixtures" / "public_api_snapshot.json")
```

- [ ] **Step 5: Wire Gate 3 into `main()`**

```python
    g3 = gate_snapshot(REPO_ROOT)
    gates.append(g3)
```

- [ ] **Step 6: Run helper tests + smoke**

```bash
python -m pytest tests/test_release_check.py::test_gate3_snapshot_equal tests/test_release_check.py::test_gate3_snapshot_diff -v
python tools/release_check.py --no-tests
```

Expected: 2 PASSED. Smoke-run: Gate 3 reports PASS (snapshot matches 0.45.0 fixture).

- [ ] **Step 7: Commit**

```bash
git add tools/release_check.py tests/test_release_check.py tests/test_public_api.py
git commit -m "$(cat <<'EOF'
feat(0.46.0): release_check Gate 3 — public API snapshot diff

Re-generates snapshot via tests/test_public_api.collect_current_api,
diffs against on-disk fixture. Never overwrites — re-snapshot is a
deliberate maintainer step. Promotes _collect_current_api → public name.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Aggregate report + exit code polish

The skeleton already aggregates with `all(g["passed"] for g in gates)`. Add a final summary line so the maintainer sees one-shot PASS/FAIL without scrolling.

- [ ] **Step 1: Write the failing aggregate test**

Add to `tests/test_release_check.py`:

```python
def test_main_emits_summary_pass(capsys, monkeypatch):
    # Force all gates to be no-ops returning pass.
    monkeypatch.setattr(release_check, "gate_test_baseline", lambda _r: {"name": "test_baseline", "passed": True, "report": "  Gate 1: PASS"})
    monkeypatch.setattr(release_check, "gate_pyodide", lambda _r: {"name": "pyodide", "passed": True, "report": "  Gate 2: PASS"})
    monkeypatch.setattr(release_check, "gate_snapshot", lambda _r: {"name": "public_api_snapshot", "passed": True, "report": "  Gate 3: PASS"})
    monkeypatch.setattr(release_check, "gate_version_sync", lambda **kw: {"name": "version_sync", "passed": True, "report": "  Gate 4: PASS"})
    rc = release_check.main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "all gates PASS" in captured.out.lower() or "all 4 gates pass" in captured.out.lower()


def test_main_emits_summary_fail(capsys, monkeypatch):
    monkeypatch.setattr(release_check, "gate_test_baseline", lambda _r: {"name": "test_baseline", "passed": True, "report": "  Gate 1: PASS"})
    monkeypatch.setattr(release_check, "gate_pyodide", lambda _r: {"name": "pyodide", "passed": True, "report": "  Gate 2: PASS"})
    monkeypatch.setattr(release_check, "gate_snapshot", lambda _r: {"name": "public_api_snapshot", "passed": True, "report": "  Gate 3: PASS"})
    monkeypatch.setattr(release_check, "gate_version_sync", lambda **kw: {"name": "version_sync", "passed": False, "report": "  Gate 4: FAIL"})
    rc = release_check.main([])
    captured = capsys.readouterr()
    assert rc == 1
    assert "fail" in captured.out.lower()
    assert "version_sync" in captured.out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_release_check.py::test_main_emits_summary_pass tests/test_release_check.py::test_main_emits_summary_fail -v
```

Expected: 2 FAILED (no summary line).

- [ ] **Step 3: Add summary block to `main()`**

After the gate-report print loop:

```python
    passed_count = sum(1 for g in gates if g["passed"])
    total = len(gates)
    failed = [g["name"] for g in gates if not g["passed"]]
    if not failed:
        print(f"\nall {total} gates PASS")
        return 0
    print(f"\n{passed_count}/{total} gates passed; FAIL: {', '.join(failed)}")
    return 1
```

- [ ] **Step 4: Run helper tests + smoke**

```bash
python -m pytest tests/test_release_check.py -v
python tools/release_check.py --no-tests
echo "exit: $?"
```

Expected: all tests PASS. Smoke-run prints all gate reports + summary line `all 3 gates PASS` (Gate 1 skipped via `--no-tests`).

- [ ] **Step 5: Commit**

```bash
git add tools/release_check.py tests/test_release_check.py
git commit -m "$(cat <<'EOF'
feat(0.46.0): release_check summary line + exit-code aggregation

One-shot PASS/FAIL line at the end. Exit 1 if any gate failed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `CONTRIBUTING.md` update — "Before tagging a release" subsection

- [ ] **Step 1: Read the current `CONTRIBUTING.md` "How to run things" section**

```bash
grep -n "How to run\|When to bump" CONTRIBUTING.md
```

- [ ] **Step 2: Insert new subsection between "When to bump the version" and "Diagnostic-error contract"**

Use Edit on `CONTRIBUTING.md`. Add this block before the `## Diagnostic-error contract` heading:

```markdown
## Before tagging a release

Run the release-gate from the repo root:

\`\`\`bash
python tools/release_check.py
\`\`\`

This runs four gates and exits 0 only if all pass:

1. **Test baseline** — pytest failing set must equal `tests/known_failures.json::entries[*].nodeid`. New failures → fail. Previously-red-now-green → pass with a warning that the whitelist should shrink.
2. **Pyodide contract** — `tests/test_pyodide_compat.py` green.
3. **Public API snapshot** — fresh introspection must equal `tests/fixtures/public_api_snapshot.json`. Regenerate the fixture deliberately when a public-API change is intentional; the gate never writes.
4. **Version sync** — `pyproject.toml`, `puremacro/__init__.py`, and the first `## X.Y.Z` heading in `CHANGELOG.md` must agree.

If a gate's failure is real and accepted (e.g. environmentally-gated test), add it to `tests/known_failures.json` with a populated `reason` / `since_version` / `owner_note`. The whitelist is the audit trail.

There is no enforcement beyond this command — the package's "tests-over-types, no CI by design" promise stands. The discipline is: run the gate before every `git tag`.
```

- [ ] **Step 3: Verify the markdown renders sensibly**

```bash
grep -A 20 "Before tagging" CONTRIBUTING.md
```

Expected: the new subsection appears between "When to bump the version" and "Diagnostic-error contract".

- [ ] **Step 4: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "$(cat <<'EOF'
docs(0.46.0): CONTRIBUTING — "Before tagging a release" gate procedure

Documents tools/release_check.py as the pre-tag step. No enforcement;
discipline is "run it before git tag" per the package's no-CI promise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Version bump + CHANGELOG + release tag

Final step: bump three version strings, write the CHANGELOG entry, run the gate end-to-end on the final commit, tag.

- [ ] **Step 1: Bump `pyproject.toml`**

Edit `pyproject.toml` line `version = "0.45.0"` → `version = "0.46.0"`.

- [ ] **Step 2: Bump `puremacro/__init__.py`**

Edit `__version__ = "0.45.0"` → `__version__ = "0.46.0"`.

- [ ] **Step 3: Bump `tests/test_import.py`**

Edit `assert puremacro.__version__ == "0.44.0"` → `assert puremacro.__version__ == "0.46.0"`. This **also** removes the long-standing baseline failure that was in `tests/known_failures.json`.

- [ ] **Step 4: Drop `tests/test_import.py::test_import_puremacro` from `tests/known_failures.json`**

Remove the entry whose `nodeid` is `tests/test_import.py::test_import_puremacro`.

- [ ] **Step 5: Prepend CHANGELOG 0.46.0 entry**

Edit `CHANGELOG.md`, inserting after the `# Changelog` header:

```markdown
## 0.46.0 — 2026-05-22

Release-gate consolidation: `tools/release_check.py` ships as the pre-tag
step. Four gates (test baseline / Pyodide contract / public API snapshot /
version sync). `tests/known_failures.json` is the explicit whitelist for
known-red tests; gate fails on any new failure not in the whitelist.

This release closes the structural gap that produced the 0.41 → 0.42
pyproject.toml staleness (caught by Gate 4) and that let "pre-existing
failures" become a meta-category across 0.42 → 0.45 (caught by Gate 1's
diff against an explicit whitelist).

### Added
- `tools/release_check.py` — single-command pre-tag gate.
- `tests/known_failures.json` — explicit whitelist with `reason` /
  `since_version` / `owner_note` per entry. Seeded with the live full-suite
  red set as of `release/0.46.0` HEAD (one less entry after Task 10's
  `tests/test_import.py` resync).
- `tests/test_release_check.py` — unit tests for gate helpers.
- `tests/test_known_failures_schema.py` — locks the JSON contract.

### Changed
- `tests/test_import.py` — bumped hardcoded assertion to `"0.46.0"`.
- `CONTRIBUTING.md` — new "Before tagging a release" subsection.

### Internal
- No `puremacro/*.py` files modified beyond `__init__.py::__version__`.
  Zero behavior change to the wheel-shipped surface.

---
```

- [ ] **Step 6: Run the gate on the staged state**

```bash
python tools/release_check.py
echo "exit: $?"
```

Expected: all 4 gates PASS. If Gate 1 reports a "recovered" warning naming `tests/test_import.py::test_import_puremacro`, that's because the entry was just removed from the whitelist — confirm and shrink. If Gate 1 reports a NEW failure, **do not tag**; root-cause first.

- [ ] **Step 7: Commit the bump**

```bash
git add pyproject.toml puremacro/__init__.py tests/test_import.py tests/known_failures.json CHANGELOG.md
git commit -m "$(cat <<'EOF'
chore(puremacro): bump 0.45.0 → 0.46.0 (release-gate ships)

Three version strings synced. CHANGELOG entry. test_import resync drops
its long-standing known-failure entry. Gate is now the pre-tag step.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 8: Final verification — gate passes on tag commit**

```bash
python tools/release_check.py
echo "exit: $?"
```

Expected: exit 0. If not, fix and re-commit before tagging.

- [ ] **Step 9: Tag the release**

```bash
git tag -a v0.46.0 -m "puremacro 0.46.0 — release-gate"
```

(Push the tag only when the user explicitly asks.)

- [ ] **Step 10: Merge to main (when user requests)**

```bash
git checkout main
git merge --no-ff release/0.46.0
```

Do not auto-execute Step 10 — wait for the user to request the merge.

---

## Self-review notes

Spec coverage (all from `docs/specs/2026-05-22-puremacro-046-047-consolidate-finish-design.md` § 0.46.0):

- Gate 1 (test baseline) — Task 5 ✓
- Gate 2 (Pyodide contract) — Task 6 ✓
- Gate 3 (public API snapshot) — Task 7 ✓
- Gate 4 (version sync) — Task 4 ✓
- `tests/known_failures.json` with schema — Task 2 ✓
- `tools/release_check.py` ≤ ~150 LOC — Tasks 3–8 total roughly ~150 LOC (verify in Task 8 Step 4)
- No `puremacro/*.py` modified beyond `__init__.py::__version__` — Tasks 0–10 file lists ✓
- `CONTRIBUTING.md` "Before tagging" — Task 9 ✓
- CHANGELOG 0.46.0 entry — Task 10 Step 5 ✓
- Version strings bumped — Task 10 Steps 1–3 ✓

Risks pulled forward from the spec:

- **R4 (Task 1's baseline larger than expected):** the plan handles arbitrary size — every nodeid in `/tmp/release_check_failing_nodeids.txt` gets a whitelist entry with `reason: "TBD — root-caused at 0.46.1"` if root-cause is unknown. This is the only TBD allowed in this plan.
- **R5 (gate is too burdensome and gets skipped):** Task 9's docs change is the structural mitigation. No further enforcement; per spec, discipline only.

Out of scope (deferred to follow-on plans):

- 0.46.1 plan (whitelist drain) — to be written after Task 1 baseline reveals the actual red set.
- 0.47.0 plan (regress/lp + garch_utils + ProxySVAR + Klein) — to be written after 0.46.1 ships clean.
