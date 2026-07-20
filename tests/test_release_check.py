"""Tests for tools/release_check.py. Imports the module as a script-like module."""
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "release_check.py"

_spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
release_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_check)


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
    assert r["name"] == "version_sync"
    assert r["passed"] is True


def test_gate4_version_sync_fail():
    r = release_check.gate_version_sync(
        pyproject_version="0.12.1",
        init_version="0.46.0",
        changelog_version="0.46.0",
    )
    assert r["name"] == "version_sync"
    assert r["passed"] is False
    assert "0.12.1" in r["report"]
    assert "0.46.0" in r["report"]


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


def test_gate2_pyodide_passthrough_pass(monkeypatch):
    # Smoke: gate_pyodide returns a dict with 'passed' / 'report'.
    r = release_check.gate_pyodide(REPO_ROOT)
    assert "passed" in r
    assert "report" in r
    assert r["name"] == "pyodide"


def test_gate1_pytest_collect_error_path(monkeypatch, tmp_path):
    """When pytest exits with a non-0/1 code, gate_test_baseline must fail (not silently pass)."""
    def fake_run(*args, **kwargs):
        class P:
            returncode = 4  # pytest usage error
            stdout = ""
            stderr = "ERROR: pytest config invalid"
        return P()
    monkeypatch.setattr(release_check.subprocess, "run", fake_run)
    # Stub the whitelist loader so gate_test_baseline doesn't need a real file.
    monkeypatch.setattr(release_check, "load_whitelist", lambda _p: set())
    r = release_check.gate_test_baseline(tmp_path)
    assert r["passed"] is False
    assert "could not run" in r["report"] or "exited with code" in r["report"]


def test_gate3_snapshot_equal(tmp_path):
    snap = {"all": {"puremacro._linalg": ["inv_xtx", "safe_cholesky"]}}
    f = tmp_path / "snap.json"
    f.write_text(json.dumps(snap))
    r = release_check.compare_snapshot(snap, f)
    assert r["passed"] is True
    assert r["name"] == "public_api_snapshot"


def test_gate3_snapshot_diff(tmp_path):
    on_disk = {"all": {"puremacro._linalg": ["inv_xtx"]}}
    fresh = {"all": {"puremacro._linalg": ["inv_xtx", "safe_cholesky"]}}
    f = tmp_path / "snap.json"
    f.write_text(json.dumps(on_disk))
    r = release_check.compare_snapshot(fresh, f)
    assert r["passed"] is False
    assert "safe_cholesky" in r["report"]
    assert r["name"] == "public_api_snapshot"


def test_gate3_snapshot_diff_result_classes(tmp_path):
    """A result_classes-only change must produce a FAIL with a non-empty diff body."""
    on_disk = {
        "all": {"puremacro.foo": ["bar"]},
        "result_classes": {"puremacro.foo.BarResult": ["x"]},
    }
    fresh = {
        "all": {"puremacro.foo": ["bar"]},
        "result_classes": {"puremacro.foo.BarResult": ["x", "y"]},  # field added
    }
    f = tmp_path / "snap.json"
    f.write_text(json.dumps(on_disk))
    r = release_check.compare_snapshot(fresh, f)
    assert r["passed"] is False
    assert r["name"] == "public_api_snapshot"
    assert "BarResult.y" in r["report"]


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


def test_gate5_all_pass_or_skip(tmp_path):
    data = {
        "schema_version": 1,
        "generated_at": "2026-05-22T14:30:00Z",
        "examples": {
            "a": {"status": "PASS", "reason": None, "runtime_s": 1.0, "figures": [], "last_run": "2026-05-22T14:30:00Z"},
            "b": {"status": "SKIP", "reason": "network unavailable", "runtime_s": 0.5, "figures": [], "last_run": "2026-05-22T14:30:00Z"},
        },
    }
    f = tmp_path / "gallery.json"
    f.write_text(json.dumps(data))
    r = release_check.gate_examples_gallery(f, examples_source_dir=tmp_path / "no_examples")
    assert r["passed"] is True
    assert r["name"] == "examples_gallery"


def test_gate5_one_fail(tmp_path):
    data = {
        "schema_version": 1,
        "generated_at": "2026-05-22T14:30:00Z",
        "examples": {
            "a": {"status": "PASS", "reason": None, "runtime_s": 1.0, "figures": [], "last_run": "2026-05-22T14:30:00Z"},
            "b": {"status": "FAIL", "reason": "ValueError", "runtime_s": 0.1, "figures": [], "last_run": "2026-05-22T14:30:00Z"},
        },
    }
    f = tmp_path / "gallery.json"
    f.write_text(json.dumps(data))
    r = release_check.gate_examples_gallery(f, examples_source_dir=tmp_path / "no_examples")
    assert r["passed"] is False
    assert "b" in r["report"]


def test_gate5_missing_json(tmp_path):
    f = tmp_path / "gallery.json"  # does not exist
    r = release_check.gate_examples_gallery(f, examples_source_dir=tmp_path / "no_examples")
    assert r["passed"] is False
    assert "not rendered" in r["report"].lower() or "missing" in r["report"].lower()


def test_gate5_stale_json_warns(tmp_path):
    import time
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    (examples_dir / "a.py").write_text("# new\n")
    time.sleep(0.05)
    # JSON's generated_at is OLDER than the example file mtime.
    old_iso = "2020-01-01T00:00:00Z"
    data = {
        "schema_version": 1,
        "generated_at": old_iso,
        "examples": {
            "a": {"status": "PASS", "reason": None, "runtime_s": 1.0, "figures": [], "last_run": old_iso},
        },
    }
    f = tmp_path / "gallery.json"
    f.write_text(json.dumps(data))
    r = release_check.gate_examples_gallery(f, examples_source_dir=examples_dir)
    # Stale warns; does NOT fail.
    assert r["passed"] is True
    assert "stale" in r["report"].lower() or "consider re-rendering" in r["report"].lower()


def test_main_summary_5_gates_with_examples_flag(capsys, monkeypatch):
    """When --examples is passed and all 5 gates pass, summary line says 5."""
    monkeypatch.setattr(release_check, "gate_test_baseline", lambda _r: {"name": "test_baseline", "passed": True, "report": "  Gate 1: PASS"})
    monkeypatch.setattr(release_check, "gate_pyodide", lambda _r: {"name": "pyodide", "passed": True, "report": "  Gate 2: PASS"})
    monkeypatch.setattr(release_check, "gate_snapshot", lambda _r: {"name": "public_api_snapshot", "passed": True, "report": "  Gate 3: PASS"})
    monkeypatch.setattr(release_check, "gate_version_sync", lambda **kw: {"name": "version_sync", "passed": True, "report": "  Gate 4: PASS"})
    monkeypatch.setattr(
        release_check, "gate_examples_gallery",
        lambda _p, *, examples_source_dir: {"name": "examples_gallery", "passed": True, "report": "  Gate 5: PASS"},
    )
    rc = release_check.main(["--examples"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "all 5 gates pass" in captured.out.lower()


def test_gate6_pass(monkeypatch, tmp_path):
    """gate_pyodide_smoke wraps pyodide_smoke.run; PASS path."""
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
