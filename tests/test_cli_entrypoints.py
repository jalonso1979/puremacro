"""Contract tests for packaged CLI and MCP console-script entry points."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _resolve_entry_point(spec: str):
    mod_name, _, attr_path = spec.partition(":")
    assert mod_name and attr_path, f"invalid console script spec: {spec!r}"
    obj = importlib.import_module(mod_name)
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def _repo_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    mpl_dir = tmp_path / "mplconfig"
    mpl_dir.mkdir(exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))
    return env


def test_pyproject_scripts_declare_and_resolve_all_cli_entrypoints():
    """pyproject.toml [project.scripts] preserves 6.5.0 and exposes CLI & MCP entry points."""
    data = _load_pyproject()
    project = data["project"]
    assert project["version"] == "6.5.0"

    scripts = project["scripts"]
    expected = {
        "puremacro": "puremacro.cli:main",
        "puremacro-mcp": "puremacro.mcp_server:main",
        "puremacro-dynare": "puremacro.dsge.cli:main",
        "puremacro-dp": "puremacro.dp.cli:main",
    }
    for name, target in expected.items():
        assert scripts.get(name) == target
        fn = _resolve_entry_point(target)
        assert callable(fn)


def test_sys_argv_cli_and_mcp_main_callables(monkeypatch: pytest.MonkeyPatch, capsys, tmp_path: Path):
    """Both puremacro and puremacro-mcp main() entry points honor sys.argv when argv is None."""
    cli_main = _resolve_entry_point("puremacro.cli:main")
    mcp_main = _resolve_entry_point("puremacro.mcp_server:main")

    doctor_out = tmp_path / "doctor.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["puremacro", "doctor", "--format", "json", "-o", str(doctor_out), "--quiet"],
    )
    assert cli_main() == 0
    doctor_payload = json.loads(doctor_out.read_text(encoding="utf-8"))
    assert doctor_payload["puremacro_version"] == "6.5.0"
    assert doctor_payload["required_failures"] == []

    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["puremacro-mcp", "--list-tools"])
    assert mcp_main() == 0
    captured = capsys.readouterr()
    tools_payload = json.loads(captured.out)
    tool_names = {tool["name"] for tool in tools_payload["tools"]}
    assert {"preflight_panel", "lp_hac", "validation_scorecard"}.issubset(tool_names)


def test_module_cli_doctor_and_mcp_list_tools_subprocesses(tmp_path: Path):
    """python -m puremacro.cli doctor and python -m puremacro.mcp_server --list-tools succeed."""
    env = _repo_env(tmp_path)

    doc_proc = subprocess.run(
        [sys.executable, "-m", "puremacro.cli", "doctor", "--format", "json"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert doc_proc.returncode == 0, doc_proc.stderr
    doc_payload = json.loads(doc_proc.stdout)
    assert doc_payload["puremacro_version"] == "6.5.0"
    assert doc_payload["required_failures"] == []

    mcp_proc = subprocess.run(
        [sys.executable, "-m", "puremacro.mcp_server", "--list-tools"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert mcp_proc.returncode == 0, mcp_proc.stderr
    mcp_payload = json.loads(mcp_proc.stdout)
    tool_names = {tool["name"] for tool in mcp_payload["tools"]}
    assert {"preflight_panel", "lp_hac", "validation_scorecard"}.issubset(tool_names)
