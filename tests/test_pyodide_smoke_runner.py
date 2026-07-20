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
