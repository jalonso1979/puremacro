"""3.4.0 regression tests for tools/release_check.py.

Covers the three gate changes of the 3.4.0 pre-release review:

* Gate 7 (min-Python syntax): ``ast.parse(feature_version=(3, 11))`` on a
  3.12+ interpreter accepts PEP 701 f-strings, so the gate carries a tokenizer
  scan. The verdict table below was recorded from CPython 3.11.15 and is
  re-checked against a real ``python3.11`` whenever one is on PATH.
* Gate 1 now counts ``ERROR`` summary lines (setup errors), not only ``FAILED``.
* Gate 4 reads the playground wheel pin as a fifth version-bearing file.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "release_check.py"

_spec = importlib.util.spec_from_file_location("release_check_34fixes", SCRIPT)
release_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_check)

FLOOR = (3, 11)

#: (case, source, accepted by CPython 3.11?). The sources are ordinary string
#: literals here, so this file itself parses on 3.11.
_CASES = [
    ("same_quote", 'x = f"{d["k"]}"\n', False),
    ("backslash_raw", "x = f\"T ({s.replace('_', r'\\_')})\"\n", False),
    ("backslash_str", 'x = f"{"\\n".join(a)}"\n', False),
    ("comment", 'x = f"{a # c\n}"\n', False),
    ("newline_single", 'x = f"{a\n+ b}"\n', False),
    ("nested_fstring_same_quote", 'x = f"{f"{a}"}"\n', False),
    ("unicode_escape_in_field", 'x = f"{"\\N{DEGREE SIGN}"}"\n', False),
    ("nested_triple_in_single", "x = f'{'''a'''}'\n", False),
    ("type_stmt", "type X = int\n", False),
    ("generic_def", "def f[T](x: T) -> T:\n    return x\n", False),
    ("escaped_braces", 'x = f"{{literal}} {a!r:>{w}} {b}"\n', True),
    ("nested_diff_quote_ok", "x = f\"{d['k']}\"\n", True),
    ("triple_ok", 'x = f"""{a\n+ b}"""\n', True),
    ("triple_inner_same_char", 'x = f"""{d["k"]}"""\n', True),
    ("dict_in_field_ok", 'x = f"{ {1: 2}[1] }"\n', True),
    ("fmt_spec_nested", 'x = f"{a:{w}.{p}f}"\n', True),
    ("plain_str_in_field", "x = f\"{'a' + 'b'}\"\n", True),
    ("nested_fstring_diff_quote_ok", "x = f\"{f'{a}'}\"\n", True),
    ("backslash_in_literal_part_ok", 'x = f"a\\n{b}\\t"\n', True),
    ("escape_in_fmt_spec", 'x = f"{a:\\n}"\n', True),
    ("lambda_in_field_ok", 'x = f"{(lambda y: y + 1)(2)}"\n', True),
    ("walrus_in_field_ok", 'x = f"{(y := 2)}"\n', True),
    ("raw_fstring_ok", 'x = rf"\\d{n}"\n', True),
    ("implicit_concat_ok", 'x = f"{a}" "b" f"{c}"\n', True),
    ("debug_specifier_ok", 'x = f"{a=}"\n', True),
]


# ---------------------------------------------------------------------------
# Gate 7
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,src,accepted", _CASES, ids=[c[0] for c in _CASES])
def test_check_min_python_syntax_matches_the_311_verdict(tmp_path, name, src, accepted):
    """On 3.11 the native parser decides; on 3.12+ the PEP 701 scan must agree with it."""
    p = tmp_path / f"{name}.py"
    p.write_text(src, encoding="utf-8")
    problems = release_check.check_min_python_syntax(p, FLOOR, display=name)
    assert bool(problems) == (not accepted), problems
    for line in problems:
        assert line.startswith(f"{name}:"), line


_PY311 = shutil.which("python3.11")


@pytest.mark.skipif(_PY311 is None, reason="no python3.11 on PATH to re-derive the verdicts")
def test_verdict_table_is_what_cpython_311_says():
    """Guards the table above: a wrong expectation would make the scan test tautological."""
    probe = subprocess.run(
        [_PY311, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
        capture_output=True, text=True, timeout=60,
    )
    if probe.stdout.split() != ["3", "11"]:
        pytest.skip(f"python3.11 on PATH is not 3.11: {probe.stdout.strip()!r}")
    for name, src, accepted in _CASES:
        proc = subprocess.run(
            [_PY311, "-c", "import sys; compile(sys.stdin.read(), '<case>', 'exec')"],
            input=src, capture_output=True, text=True, timeout=60,
        )
        assert (proc.returncode == 0) == accepted, (name, proc.stderr.strip()[-200:])


def test_gate7_fails_on_a_file_the_floor_rejects_and_names_it(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nrequires-python = ">=3.11"\n', encoding="utf-8"
    )
    pkg = tmp_path / "puremacro"
    pkg.mkdir()
    (pkg / "good.py").write_text("x = f\"{d['k']}\"\n", encoding="utf-8")
    (pkg / "bad.py").write_text('x = f"{d["k"]}"\n', encoding="utf-8")
    cache = pkg / "__pycache__"
    cache.mkdir()
    (cache / "ignored.py").write_text("type X = int\n", encoding="utf-8")

    g = release_check.gate_min_python_syntax(tmp_path)
    assert g["name"] == "min_python_syntax"
    assert g["passed"] is False
    assert "puremacro/bad.py:1" in g["report"]
    assert "good.py" not in g["report"]
    assert "ignored.py" not in g["report"]

    (pkg / "bad.py").unlink()
    g = release_check.gate_min_python_syntax(tmp_path)
    assert g["passed"] is True, g["report"]
    assert "1 files parse" in g["report"]


def test_gate7_reads_the_floor_from_pyproject():
    assert release_check.read_requires_python_min(REPO_ROOT / "pyproject.toml") == (3, 11)


def test_gate7_parallel_workers_agree_with_the_serial_check(tmp_path):
    """The fan-out is an optimisation only: same problems, same order after sorting."""
    pkg = tmp_path / "puremacro"
    pkg.mkdir()
    for i in range(70):  # above the 64-file threshold that enables the fan-out
        (pkg / f"m{i:02d}.py").write_text(f"x{i} = f\"{{d['k']}}\"\n", encoding="utf-8")
    (pkg / "bad_a.py").write_text('x = f"{d["k"]}"\n', encoding="utf-8")
    (pkg / "bad_b.py").write_text("type X = int\n", encoding="utf-8")
    files = list(release_check.iter_syntax_files(tmp_path))
    serial = sorted(release_check._check_files(tmp_path, files, FLOOR, workers=1))
    parallel = sorted(release_check._check_files(tmp_path, files, FLOOR, workers=3))
    assert serial == parallel
    assert [p.split(":")[0] for p in serial] == ["puremacro/bad_a.py", "puremacro/bad_b.py"]


def test_gate7_fan_out_returns_none_when_a_worker_cannot_run(tmp_path):
    """A broken worker must never look like a clean pass."""
    cmd = [sys.executable, "-c", "import sys; sys.exit(3)"]
    assert release_check._fan_out(cmd, ["a.py", "b.py"], cwd=tmp_path, workers=2) is None
    cmd = [sys.executable, "-c", "print('no sentinel here')"]
    assert release_check._fan_out(cmd, ["a.py"], cwd=tmp_path, workers=1) is None
    cmd = [sys.executable, "-c", "import sys; sys.stdin.read(); print('CHECKED')"]
    assert release_check._fan_out(cmd, ["a.py"], cwd=tmp_path, workers=1) == []


@pytest.mark.skipif(sys.version_info < (3, 12), reason="the pre-filter only runs on 3.12+")
def test_gate7_prefilter_skips_the_tokenizer_for_plain_fstrings():
    """The AST pre-filter is what keeps Gate 7 at seconds; it must stay a superset."""
    import ast

    plain = 'x = f"{a:.2f} and {d[\'k\']!r}"\ny = "not an f-string"\n'
    assert release_check._may_have_pep701_fstrings(plain, ast.parse(plain)) is False
    for _name, src, accepted in _CASES:
        if accepted:
            continue
        try:
            tree = ast.parse(src, feature_version=FLOOR)
        except SyntaxError:
            continue  # rejected by ast.parse itself (PEP 695 etc.); no scan needed
        assert release_check._may_have_pep701_fstrings(src, tree) is True, src


def test_gate7_is_silent_on_the_gate_and_on_this_file():
    for p in (SCRIPT, Path(__file__)):
        assert release_check.check_min_python_syntax(p, FLOOR) == []


def test_gate7_runs_by_default(capsys, monkeypatch):
    """--no-tests must still run Gate 7: that is the configuration CI uses."""
    ok = {"passed": True, "report": "  PASS"}
    monkeypatch.setattr(release_check, "gate_pyodide", lambda _r: {"name": "pyodide", **ok})
    monkeypatch.setattr(release_check, "gate_snapshot", lambda _r: {"name": "public_api_snapshot", **ok})
    monkeypatch.setattr(release_check, "gate_version_sync", lambda **kw: {"name": "version_sync", **ok})
    seen = []
    monkeypatch.setattr(
        release_check, "gate_min_python_syntax",
        lambda root: seen.append(root) or {"name": "min_python_syntax", "passed": False, "report": "  Gate 7: FAIL"},
    )
    rc = release_check.main(["--no-tests"])
    assert seen == [release_check.REPO_ROOT]
    assert rc == 1
    assert "min_python_syntax" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Gate 1
# ---------------------------------------------------------------------------

def test_gate1_counts_setup_errors_as_red(monkeypatch):
    class _Proc:
        returncode = 1
        stderr = ""
        stdout = (
            "FAILED tests/a.py::test_x - AssertionError\n"
            "ERROR tests/b.py::TestC::test_y - FileNotFoundError: data_77c_11s.mat\n"
            "1 failed, 1 error in 1.0s\n"
        )

    monkeypatch.setattr(release_check.subprocess, "run", lambda *a, **k: _Proc())
    assert release_check.run_pytest_collect_failures(REPO_ROOT) == {
        "tests/a.py::test_x",
        "tests/b.py::TestC::test_y",
    }


# ---------------------------------------------------------------------------
# Gate 4
# ---------------------------------------------------------------------------

def test_gate4_reads_the_playground_wheel_pin(tmp_path):
    cfg = tmp_path / "jupyter_lite_config.json"
    cfg.write_text(
        json.dumps({"PipliteAddon": {"piplite_urls": ["./wheels/puremacro-3.4.0-py3-none-any.whl"]}}),
        encoding="utf-8",
    )
    assert release_check.read_playground_pin_version(cfg) == "3.4.0"


def test_gate4_fails_when_the_playground_pin_lags():
    g = release_check.gate_version_sync(
        pyproject_version="3.4.0", init_version="3.4.0",
        changelog_version="3.4.0", citation_version="3.4.0",
        playground_pin_version="1.9.0",
    )
    assert g["passed"] is False
    assert "jupyter_lite_config.json" in g["report"]


def test_the_repo_s_own_playground_pin_is_in_sync():
    pin = release_check.read_playground_pin_version(
        REPO_ROOT / "playground" / "jupyter_lite_config.json"
    )
    assert pin == release_check.read_pyproject_version(REPO_ROOT / "pyproject.toml")
