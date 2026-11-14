"""Tests for the new workbench/scorecards docs, updated tariff brief docs,
playground demo, and Pyodide-core import safety."""
from __future__ import annotations

import ast
import importlib
import os
import re
import runpy
import subprocess
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from puremacro.solvers.diagnostics import SolverDiagnosticRecord
from puremacro.trade import CompositeTariffScenario

ROOT = Path(__file__).resolve().parents[1]

_FENCED_PY_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

_FORBIDDEN_PYODIDE_IMPORTS = (
    "statsmodels",
    "linearmodels",
    "arch",
    "bs4",
    "pdfplumber",
    "pypdf",
    "pyarrow",
    "openpyxl",
    "yfinance",
    "fredapi",
    "requests",
    "urllib3",
    "httpx",
)


class _ForbiddenModuleFinder(MetaPathFinder):
    """Meta-path import blocker simulating an environment without native/network deps."""

    def __init__(self, forbidden: tuple[str, ...]) -> None:
        self.forbidden = set(forbidden)
        self.blocked_attempts: list[str] = []

    def find_spec(self, fullname: str, path=None, target=None):  # type: ignore[override]
        top = fullname.split(".")[0]
        if top in self.forbidden:
            self.blocked_attempts.append(fullname)
            raise ImportError(f"Blocked in Pyodide-core check: {fullname}")
        return None


def _extract_python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks: list[str] = []
    for match in _FENCED_PY_RE.finditer(text):
        code = match.group(1)
        ast.parse(code, filename=str(path))
        blocks.append(code)
    return blocks


def test_workbench_docs_bilingual_headers_and_structure() -> None:
    en_path = ROOT / "docs" / "workbench_and_scorecards.md"
    es_path = ROOT / "docs" / "es" / "workbench_and_scorecards.md"

    assert en_path.exists(), "Missing English workbench_and_scorecards.md"
    assert es_path.exists(), "Missing Spanish es/workbench_and_scorecards.md"

    en_text = en_path.read_text(encoding="utf-8")
    es_text = es_path.read_text(encoding="utf-8")

    assert "Español" in en_text[:600]
    assert "es/workbench_and_scorecards.md" in en_text[:600]
    assert "English" in es_text[:600]
    assert "../workbench_and_scorecards.md" in es_text[:600]

    en_blocks = _extract_python_blocks(en_path)
    es_blocks = _extract_python_blocks(es_path)
    assert len(en_blocks) == 6
    assert len(es_blocks) == 6

    required_tokens = (
        "preflight_panel",
        "lp_sensitivity_grid",
        "compare_irfs",
        "prior_posterior_comparison",
        "score_irf_against_targets",
        "ConvergenceStatus",
        "AccuracyStatus",
        "SupportStatus",
        "make_solver_diagnostic_record",
        "CompositeTariffScenario",
        "combine_scenarios",
        "summarize_scenario_batch",
    )
    for token in required_tokens:
        assert token in en_text, f"English guide missing {token}"
        assert token in es_text, f"Spanish guide missing {token}"


@pytest.mark.parametrize(
    ("rel_path", "expected_baseline_label"),
    [
        ("docs/workbench_and_scorecards.md", "LP (2 lags)"),
        ("docs/es/workbench_and_scorecards.md", "LP (2 rezagos)"),
    ],
)
def test_workbench_docs_code_blocks_execute(
    rel_path: str,
    expected_baseline_label: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc_path = ROOT / rel_path
    blocks = _extract_python_blocks(doc_path)
    assert len(blocks) == 6

    matplotlib.use("Agg", force=True)
    monkeypatch.chdir(tmp_path)
    ns: dict[str, object] = {}

    for idx, code in enumerate(blocks, 1):
        assert not any(
            line.strip().startswith("# requires:") for line in code.splitlines()
        ), f"Block {idx} in {rel_path} should execute unconditionally"
        exec(compile(code, f"{rel_path}#block{idx}", "exec"), ns, ns)
        plt.close("all")

    out = capsys.readouterr().out
    assert "status: passed" in out
    assert "pass record: PASS True" in out
    assert ns["report"].passed is True
    assert ns["report"].inferred_freq == "Q"
    assert len(ns["grid_res"].to_frame("metrics")) == 4
    assert expected_baseline_label in ns["cmp_res"].to_frame("metrics")["spec"].tolist()
    assert len(ns["pp"].to_frame("compact")) == 2
    assert int(ns["scorecard"].chi2_df) == 5
    assert isinstance(ns["pass_rec"], SolverDiagnosticRecord)
    assert ns["pass_rec"].is_pass is True
    assert ns["stall_status"] == "CONVERGED_INACCURATE"
    assert isinstance(ns["combined"], CompositeTariffScenario)
    assert len(ns["combined"].conflicts(ns["calib"].country_codes, ns["calib"].sector_codes)) == 1
    assert len(ns["summary"].to_payoff_matrix(metric="ev_pct_consumption")) == 4


def test_first_brief_tariffs_docs_cover_enigh_own_data_and_config_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    en_path = ROOT / "docs" / "first_brief_tariffs.md"
    es_path = ROOT / "docs" / "es" / "first_brief_tariffs.md"

    en_text = en_path.read_text(encoding="utf-8")
    es_text = es_path.read_text(encoding="utf-8")

    expected_flags_and_apis = (
        "--enigh",
        "--trade-elasticity",
        "--household-elasticity",
        "--experiment",
        "--groups",
        "TradeBriefConfig",
        "compute_brief",
        "load_enigh_groups",
        "load_groups",
        "paths.csv",
        "headline.json",
    )
    for token in expected_flags_and_apis:
        assert token in en_text, f"docs/first_brief_tariffs.md missing {token}"
        assert token in es_text, f"docs/es/first_brief_tariffs.md missing {token}"

    en_blocks = _extract_python_blocks(en_path)
    es_blocks = _extract_python_blocks(es_path)
    assert len(en_blocks) == 2
    assert len(es_blocks) == 2

    # Execute the newly added TradeBriefConfig / compute_brief block cleanly in tmp_path.
    monkeypatch.chdir(tmp_path)
    ns: dict[str, object] = {}
    exec(compile(en_blocks[1], "docs/first_brief_tariffs.md#block2", "exec"), ns, ns)
    paths = ns["paths"]
    assert isinstance(paths, pd.DataFrame)
    assert {"trade_elasticity", "household_elasticity", "group", "ev_pct_proportional"}.issubset(paths.columns)
    assert len(paths) == 6
    assert list(paths["group"]) == ["Q1", "Q2", "Q3", "Q4", "Q5", "all"]


def test_playground_workbench_demo_runs_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    demo_path = ROOT / "playground" / "01_workbench_demo.py"
    assert demo_path.exists()

    source = demo_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(demo_path))
    assert "# %% [markdown]" in source
    assert "# %%" in source
    assert "# %pip install puremacro" in source

    matplotlib.use("Agg", force=True)
    monkeypatch.chdir(tmp_path)
    ns = runpy.run_path(str(demo_path), run_name="__main__")
    plt.close("all")

    stdout = capsys.readouterr().out
    assert "=== 1. Panel Preflight ===" in stdout
    assert "=== 5. Solver Diagnostics ===" in stdout

    assert ns["preflight_report"].passed is True
    assert len(ns["lp_grid_result"].to_frame("metrics")) == 4
    assert ns["irf_comparison"].baseline == "LP (lags=2)"
    assert isinstance(ns["composite_scenario"], CompositeTariffScenario)
    assert len(ns["batch_summary"].to_payoff_matrix(metric="ev_pct_consumption")) == 4

    records = ns["solver_records"]
    assert records["loose_solve"].overall_status == "CONVERGED_INACCURATE"
    assert records["tight_solve"].overall_status == "PASS"
    assert records["constructed_tight"].is_pass is True
    assert records["manual_stall_status"] == "CONVERGED_INACCURATE"


@pytest.mark.mechanism_control
def test_forbidden_module_finder_control() -> None:
    finder = _ForbiddenModuleFinder(("statsmodels",))
    saved_modules = {
        k: v
        for k, v in sys.modules.items()
        if k == "statsmodels" or k.startswith("statsmodels.")
    }
    for key in saved_modules:
        sys.modules.pop(key, None)
    sys.meta_path.insert(0, finder)
    try:
        with pytest.raises(ImportError, match="Blocked in Pyodide-core check: statsmodels"):
            importlib.import_module("statsmodels.api")
        assert "statsmodels" in finder.blocked_attempts or "statsmodels.api" in finder.blocked_attempts
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        sys.modules.update(saved_modules)


def test_new_workflows_pyodide_core_import_safety(tmp_path: Path) -> None:
    demo_path = ROOT / "playground" / "01_workbench_demo.py"
    tree = ast.parse(demo_path.read_text(encoding="utf-8"), filename=str(demo_path))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots.isdisjoint(_FORBIDDEN_PYODIDE_IMPORTS)

    script = "\n".join(
        [
            "import runpy, sys",
            "from importlib.abc import MetaPathFinder",
            f"FORBIDDEN = {repr(_FORBIDDEN_PYODIDE_IMPORTS)}",
            "class _Blocker(MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname.split('.')[0] in FORBIDDEN:",
            "            raise ImportError('Blocked forbidden module: ' + fullname)",
            "        return None",
            "sys.meta_path.insert(0, _Blocker())",
            "import puremacro.data",
            "import puremacro.lp.sensitivity",
            "import puremacro.reports",
            "import puremacro.dsge.validation_scorecard",
            "import puremacro.solvers.diagnostics",
            "import puremacro.trade.scenarios",
            f"ns = runpy.run_path({repr(str(demo_path))}, run_name='__main__')",
            "loaded_forbidden = sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN)",
            "assert not loaded_forbidden, f'Forbidden modules loaded: {loaded_forbidden}'",
            "assert ns['solver_records']['tight_solve'].overall_status == 'PASS'",
        ]
    )
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    prev_pp = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{prev_pp}" if prev_pp else str(ROOT)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "tight_solve: overall=PASS" in proc.stdout
