"""tools/release_check.py — single pre-tag gate for puremacro releases.

Run me before `git tag X.Y.Z`. Up to six gates (all run; no fail-fast):

  Gate 1 (test baseline)    — pytest failures must equal tests/known_failures.json
  Gate 2 (Pyodide contract)  — tests/test_pyodide_compat.py must be green (static)
  Gate 3 (public API snap)   — re-generated snapshot must match the fixture
  Gate 4 (version sync)      — pyproject.toml / __init__.py / CHANGELOG agree
  Gate 5 (examples gallery)  — opt-in via --examples; reads docs/examples_gallery.json
  Gate 6 (pyodide smoke)     — opt-in via --pyodide; boots Pyodide, runs marked tests

Exit 0 iff every gate run passed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_pyproject_version(path: Path) -> str:
    text = Path(path).read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise ValueError(f"version not found in {path}")
    return m.group(1)


def read_init_version(path: Path) -> str:
    text = Path(path).read_text()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise ValueError(f"__version__ not found in {path}")
    return m.group(1)


def read_changelog_version(path: Path) -> str:
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^##\s+(\d+\.\d+\.\d+)", line)
        if m:
            return m.group(1)
    raise ValueError(f"no '## X.Y.Z' heading in {path}")


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
    """Run the full suite (minus network) and return the failing-nodeid set.

    Raises:
        RuntimeError: if pytest itself errors out (collection error, internal error,
            usage error) rather than running tests normally.
        subprocess.TimeoutExpired: if the run exceeds the 600s budget.
    """
    cmd = [
        sys.executable, "-m", "pytest",
        "puremacro/tests/", "tests/",
        "-m", "not network and not slow",
        "--tb=no", "-q",
        "--no-header",
    ]
    proc = subprocess.run(
        cmd, cwd=repo_root, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"pytest exited with code {proc.returncode} "
            f"(expected 0 or 1). Last stderr lines:\n"
            + "\n".join(proc.stderr.splitlines()[-20:])
        )
    failures = set()
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED "):
            nid = line[len("FAILED "):].split(" ", 1)[0]
            failures.add(nid)
    return failures


def gate_pyodide(repo_root: Path) -> dict:
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_pyodide_compat.py",
        "--tb=short", "-q",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": "pyodide",
            "passed": False,
            "report": "  Gate 2 (Pyodide contract): FAIL — pytest exceeded 300s timeout",
        }
    passed = proc.returncode == 0
    head = "  Gate 2 (Pyodide contract): " + ("PASS" if passed else "FAIL")
    tail = "" if passed else "\n" + "\n".join(
        f"    {line}" for line in proc.stdout.splitlines()[-20:]
    )
    return {"name": "pyodide", "passed": passed, "report": head + tail}


def gate_test_baseline(repo_root: Path) -> dict:
    whitelist = load_whitelist(repo_root / "tests" / "known_failures.json")
    try:
        failing = run_pytest_collect_failures(repo_root)
    except subprocess.TimeoutExpired:
        return {
            "name": "test_baseline",
            "passed": False,
            "report": "  Gate 1 (test baseline): FAIL — pytest exceeded 600s timeout",
            "new": set(),
            "recovered": set(),
        }
    except RuntimeError as e:
        return {
            "name": "test_baseline",
            "passed": False,
            "report": f"  Gate 1 (test baseline): FAIL — pytest could not run\n    {e}",
            "new": set(),
            "recovered": set(),
        }
    return compare_failures(failing, whitelist)


def compare_snapshot(fresh: dict, fixture_path: Path) -> dict:
    on_disk = json.loads(Path(fixture_path).read_text())
    if fresh == on_disk:
        return {
            "name": "public_api_snapshot",
            "passed": True,
            "report": "  Gate 3 (public API snapshot): PASS",
        }
    lines = ["  Gate 3 (public API snapshot): FAIL — diff:"]

    # Diff the "all" map (module → symbol list).
    fresh_modules = set(fresh.get("all", {}).keys())
    disk_modules = set(on_disk.get("all", {}).keys())
    for mod in sorted(fresh_modules - disk_modules):
        lines.append(f"    + module {mod}")
    for mod in sorted(disk_modules - fresh_modules):
        lines.append(f"    - module {mod}")
    for mod in sorted(fresh_modules & disk_modules):
        added = set(fresh["all"][mod]) - set(on_disk["all"][mod])
        removed = set(on_disk["all"][mod]) - set(fresh["all"][mod])
        for sym in sorted(added):
            lines.append(f"    + {mod}.{sym}")
        for sym in sorted(removed):
            lines.append(f"    - {mod}.{sym}")

    # Diff the "result_classes" map (class qualname → field list).
    fresh_classes = set(fresh.get("result_classes", {}).keys())
    disk_classes = set(on_disk.get("result_classes", {}).keys())
    for cls in sorted(fresh_classes - disk_classes):
        lines.append(f"    + class {cls}")
    for cls in sorted(disk_classes - fresh_classes):
        lines.append(f"    - class {cls}")
    for cls in sorted(fresh_classes & disk_classes):
        added = set(fresh["result_classes"][cls]) - set(on_disk["result_classes"][cls])
        removed = set(on_disk["result_classes"][cls]) - set(fresh["result_classes"][cls])
        for field in sorted(added):
            lines.append(f"    + {cls}.{field}")
        for field in sorted(removed):
            lines.append(f"    - {cls}.{field}")

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


def gate_examples_gallery(json_path: Path, *, examples_source_dir: Path) -> dict:
    """Gate 5 — examples gallery health.

    Reads docs/examples_gallery.json. Fails on any FAIL entry. Warns
    (does not fail) if the JSON's generated_at is older than the newest
    *.py file under examples_source_dir.
    """
    name = "examples_gallery"
    if not Path(json_path).exists():
        return {
            "name": name,
            "passed": False,
            "report": (
                f"  Gate 5 (examples gallery): FAIL — examples gallery not rendered\n"
                f"    expected at {json_path}\n"
                f"    run: python tools/render_examples_gallery.py"
            ),
        }
    try:
        data = json.loads(Path(json_path).read_text())
    except json.JSONDecodeError as e:
        return {
            "name": name,
            "passed": False,
            "report": f"  Gate 5 (examples gallery): FAIL — JSON malformed: {e}",
        }
    examples = data.get("examples", {})
    counts = {"PASS": 0, "SKIP": 0, "FAIL": 0}
    fails = []
    for nm, e in examples.items():
        s = e.get("status", "FAIL")
        counts[s] = counts.get(s, 0) + 1
        if s == "FAIL":
            fails.append((nm, e.get("reason", "(no reason)")))

    # Stale-JSON warning (not a failure).
    warn_lines = []
    if examples_source_dir.exists():
        from datetime import datetime, timezone
        try:
            gen = datetime.strptime(data["generated_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            newest_src = max(
                (p.stat().st_mtime for p in examples_source_dir.glob("*.py")),
                default=0.0,
            )
            if newest_src > gen.timestamp():
                warn_lines.append(
                    "    stale: an example source file is newer than the gallery JSON; "
                    "consider re-rendering"
                )
        except (KeyError, ValueError):
            pass

    if fails:
        lines = [f"  Gate 5 (examples gallery): FAIL — {len(fails)} example(s) failed"]
        for nm, reason in sorted(fails):
            lines.append(f"    FAIL {nm} — {reason}")
        lines.extend(warn_lines)
        return {"name": name, "passed": False, "report": "\n".join(lines)}

    head = f"  Gate 5 (examples gallery): PASS — {counts['PASS']} PASS, {counts['SKIP']} SKIP, 0 FAIL"
    if warn_lines:
        return {"name": name, "passed": True, "report": "\n".join([head] + warn_lines)}
    return {"name": name, "passed": True, "report": head}


def gate_pyodide_smoke(repo_root: Path) -> dict:
    """Gate 6 — real Pyodide smoke.

    Delegates to tools/pyodide_smoke.py::run, which builds the wheel +
    invokes the Node runner. Slow gate (~60-180s typical; can be ~6s
    on modern hardware); opt-in via --pyodide.
    """
    sys.path.insert(0, str(repo_root / "tools"))
    try:
        import pyodide_smoke
    finally:
        sys.path.pop(0)
    result = pyodide_smoke.run(repo_root)
    return {"name": "pyodide_smoke", **result}


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="release_check",
        description="Release-gate for puremacro — runs 4 default gates + up to 2 opt-in pre-tag.",
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
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Also run Gate 5 (examples gallery health). Reads docs/examples_gallery.json.",
    )
    parser.add_argument(
        "--pyodide",
        action="store_true",
        help="Also run Gate 6 (real Pyodide smoke). Builds the wheel + boots "
             "Pyodide via node tools/pyodide/runner.js. Slow (~60-180s); "
             "requires node + one-time `npm install` in tools/pyodide/.",
    )
    args = parser.parse_args(argv)

    print("release_check — puremacro pre-tag gate")
    print(f"  repo root: {REPO_ROOT}")
    if args.report_only:
        print("  (report-only — no gates executed)")
        return 0

    gates = []
    if not args.no_tests:
        g1 = gate_test_baseline(REPO_ROOT)
        gates.append(g1)
    g2 = gate_pyodide(REPO_ROOT)
    gates.append(g2)
    g3 = gate_snapshot(REPO_ROOT)
    gates.append(g3)
    g4 = gate_version_sync(
        pyproject_version=read_pyproject_version(REPO_ROOT / "pyproject.toml"),
        init_version=read_init_version(REPO_ROOT / "puremacro" / "__init__.py"),
        changelog_version=read_changelog_version(REPO_ROOT / "CHANGELOG.md"),
    )
    gates.append(g4)
    if args.examples:
        g5 = gate_examples_gallery(
            REPO_ROOT / "docs" / "examples_gallery.json",
            examples_source_dir=REPO_ROOT / "puremacro" / "examples",
        )
        gates.append(g5)
    if args.pyodide:
        g6 = gate_pyodide_smoke(REPO_ROOT)
        gates.append(g6)

    for g in gates:
        print(g["report"])

    passed_count = sum(1 for g in gates if g["passed"])
    total = len(gates)
    failed = [g["name"] for g in gates if not g["passed"]]
    if not failed:
        print(f"\nall {total} gates PASS")
        return 0
    print(f"\n{passed_count}/{total} gates passed; FAIL: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
