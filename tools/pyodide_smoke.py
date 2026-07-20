"""tools/pyodide_smoke.py — Python wrapper for Gate 6.

Builds the puremacro wheel via `python -m build`, invokes
`node tools/pyodide/runner.js --wheel <wheel-path>`, parses the JSON
envelope, returns a {passed, report} dict.
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
