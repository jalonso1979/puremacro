"""The sovereign-risk policy brief example: scenarios, record builder, CLI."""
from __future__ import annotations

import copy
import inspect
import json
import os

import numpy as np
import pytest
from scipy.stats import t as student_t

from puremacro.examples import first_brief_sovereign as brief
from puremacro.sovereign import solve_sovereign_default
from puremacro.sovereign.measurement import MOMENTS

TINY_GRIDS = [(61, 11), (31, 11)]
TINY_DESIGN = {"periods": 5_000, "seeds": [1, 2, 3], "burn": 100}


@pytest.fixture(scope="module")
def tiny_record():
    return brief.build_record(TINY_GRIDS, design=TINY_DESIGN, log=lambda *a: None)


def test_scenarios_are_one_at_a_time():
    assert list(brief.SCENARIOS) == ["baseline", "theta=0.15", "theta=0.5", "sigma_y=0.02",
                                     "sigma_y=0.03", "rho_y=0.9", "rho_y=0.97"]
    assert brief.SCENARIOS["baseline"] == {}
    expected = {"theta": {.15, .50}, "sigma_y": {.020, .030}, "rho_y": {.90, .97}}
    seen = {key: set() for key in expected}
    for name, overrides in brief.SCENARIOS.items():
        if name == "baseline":
            continue
        assert len(overrides) == 1
        (key, value), = overrides.items()
        assert name == f"{key}={value:g}"
        assert value != brief.BASELINE[key]
        seen[key].add(value)
    assert seen == expected


def test_parse_scenario():
    assert brief.parse_scenario("theta=0.15") == ("theta=0.15", {"theta": .15})
    assert brief.parse_scenario("theta=0.50")[0] == "theta=0.5"
    assert brief.parse_scenario("baseline") == ("baseline", {})
    with pytest.raises(ValueError, match="theta"):
        brief.parse_scenario("beta=0.9")
    with pytest.raises(ValueError):
        brief.parse_scenario("theta=nan")
    with pytest.raises(ValueError):
        brief.parse_scenario("theta")


def test_baseline_is_the_public_calibration():
    # Literal copy of the kwargs of _public_monotone in tests/test_arellano_2008_study.py.
    assert brief.BASELINE == dict(beta=.95282, r_star=.017, theta=.282, rho_y=.945, sigma_y=.025,
                                  b_min=-.15, b_max=.33, sigma_eps=0, y_penalty_type="arellano",
                                  enforce_monotonicity=False, tol=1e-10, max_iter=10000,
                                  policy_search="monotone")
    assert brief.GRIDS == {"converged": (6401, 641), "error": (3201, 641), "coarse": (801, 161)}
    assert brief.DESIGN == {"periods": 500_000, "seeds": list(range(1979, 1995)), "burn": 1_000}


def test_tolerances_match_the_registered_study_values():
    from tools.reference_validation import check_arellano_grids

    for m in MOMENTS:
        assert brief.TOLERANCES[m] == check_arellano_grids.TOLERANCES["debt_pct"] == 0.10
    assert set(brief.TOLERANCES) == set(MOMENTS)


def _check_summary(case):
    sim = case["simulation"]
    n = len(sim["seeds"])
    t_crit = student_t.ppf(.975, n - 1)
    for m in MOMENTS:
        values = np.asarray(sim["per_seed"][m])
        assert len(values) == n
        se = values.std(ddof=1) / np.sqrt(n)
        summary = sim["summary"][m]
        assert summary["mean"] == pytest.approx(values.mean(), abs=1e-12)
        assert summary["se"] == pytest.approx(se, abs=1e-12)
        assert summary["ci95"] == pytest.approx([values.mean() - t_crit * se, values.mean() + t_crit * se], abs=1e-12)


def test_build_record_schema_and_recomputation(tiny_record, tmp_path):
    record = tiny_record
    for key in ("schema", "package_version", "versions", "source_sha256", "design", "baseline",
                "tolerances", "grids", "peak_rss_gb", "scenarios"):
        assert key in record, key
    assert record["schema"] == 1
    assert set(record["versions"]) == {"python", "numpy", "scipy", "pandas"}
    assert record["source_sha256"] == brief.source_hashes()
    assert record["grids"] == [[61, 11], [31, 11]]
    assert record["design"] == TINY_DESIGN and record["baseline"] == brief.BASELINE
    assert list(record["scenarios"]) == list(brief.SCENARIOS)
    # Schema 1 describes the brief's generated grids with scalar parameters.
    # The separate nonuniform study records and authenticates every explicit
    # coordinate; debt_grid=None adds no information to this uniform record.
    signature = inspect.signature(solve_sovereign_default)
    assert signature.parameters["debt_grid"].default is None
    solver_kwargs = set(signature.parameters) - {"debt_grid"}
    for name, scenario in record["scenarios"].items():
        assert scenario["overrides"] == brief.SCENARIOS[name]
        assert list(scenario["cases"]) == ["61x11", "31x11"]
        for case in scenario["cases"].values():
            assert set(case) == {"n_b", "n_y", "parameters", "solver", "simulation", "welfare_pct"}
            assert set(case["parameters"]) == solver_kwargs
            assert set(case["solver"]) == {"converged", "iterations", "residual", "monotone_certification_gap",
                                           "exhaustive_bellman_residual", "elapsed_s"}
            assert case["simulation"]["seeds"] == TINY_DESIGN["seeds"]
            _check_summary(case)
        expected = brief.paired_change(scenario["cases"]["31x11"], scenario["cases"]["61x11"])
        assert scenario["grid_change"] == expected
    text = json.dumps(record, allow_nan=False)
    for forbidden in (os.sep + "Users", "/home/", "/private/", str(tmp_path)):
        assert forbidden not in text


def test_paired_change_rule(tiny_record):
    cases = tiny_record["scenarios"]["baseline"]["cases"]
    coarse, fine = cases["31x11"], cases["61x11"]
    change = brief.paired_change(coarse, fine)
    for m in MOMENTS:
        d = np.asarray(fine["simulation"]["per_seed"][m]) - np.asarray(coarse["simulation"]["per_seed"][m])
        half = student_t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        entry = change["moments"][m]
        assert entry["change"] == pytest.approx(d.mean(), abs=1e-12)
        assert entry["absolute_change_bound"] == pytest.approx(abs(d.mean()) + half, abs=1e-12)
        bound, tol = entry["absolute_change_bound"], brief.TOLERANCES[m]
        assert entry["status"] == ("passed" if bound <= tol else "grid_shift_exceeds_tolerance"
                                   if abs(d.mean()) > tol else "sampling_precision_insufficient")
    assert change["passed"] == all(change["moments"][m]["status"] == "passed" for m in MOMENTS)
    assert change["welfare_change_pct"] == pytest.approx(fine["welfare_pct"] - coarse["welfare_pct"], abs=1e-15)
    other = copy.deepcopy(coarse)
    other["simulation"]["seeds"] = [1, 2, 4]
    with pytest.raises(ValueError, match="seed"):
        brief.paired_change(other, fine)


def test_freeze_resume_discards_stale_partial(tmp_path, monkeypatch):
    grids = [(31, 11)]
    calls = []
    real = brief.solve_scenario

    def counting(overrides, n_b, n_y):
        calls.append((tuple(overrides.items()), n_b, n_y))
        return real(overrides, n_b, n_y)

    monkeypatch.setattr(brief, "solve_scenario", counting)
    partial = tmp_path / "sweep.partial.json"

    # A complete run writes the partial after every case.
    first = brief.build_record(grids, design=TINY_DESIGN, partial=partial, log=lambda *a: None)
    assert len(calls) == len(brief.SCENARIOS)
    saved = json.loads(partial.read_text(encoding="utf-8"))
    assert saved["source_sha256"] == brief.source_hashes()

    # A partial from an older source is discarded: every case is re-solved.
    stale = copy.deepcopy(saved)
    stale["source_sha256"] = {k: "0" * 64 for k in stale["source_sha256"]}
    partial.write_text(json.dumps(stale), encoding="utf-8")
    calls.clear()
    messages = []
    second = brief.build_record(grids, design=TINY_DESIGN, partial=partial, log=messages.append)
    assert len(calls) == len(brief.SCENARIOS)
    assert any("discard" in str(m).lower() and partial.name in str(m) for m in messages)
    assert second["source_sha256"] == brief.source_hashes()

    # A matching partial with one finished case reuses it.
    matching = copy.deepcopy(saved)
    for name in list(matching["scenarios"])[1:]:
        del matching["scenarios"][name]
    partial.write_text(json.dumps(matching), encoding="utf-8")
    calls.clear()
    third = brief.build_record(grids, design=TINY_DESIGN, partial=partial, log=lambda *a: None)
    assert len(calls) == len(brief.SCENARIOS) - 1
    assert third["scenarios"]["baseline"]["cases"] == first["scenarios"]["baseline"]["cases"]

    # A partial with a different design is discarded too.
    other_design = {**TINY_DESIGN, "periods": 4_000}
    partial.write_text(json.dumps(saved), encoding="utf-8")
    calls.clear()
    brief.build_record(grids, design=other_design, partial=partial, log=lambda *a: None)
    assert len(calls) == len(brief.SCENARIOS)


# --- Task 4: renderers and CLI ------------------------------------------------

def test_single_scenario_cli_writes_outputs(tmp_path):
    code = brief.main(["--scenario", "theta=0.15", "--grid", "61x11", "--periods", "5000",
                       "--n-seeds", "2", "--output", str(tmp_path), "--lang", "es"])
    assert code == 0
    for name in ("table.md", "table.csv", "figure.png", "manifest.json"):
        assert (tmp_path / name).is_file(), name
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seeds"] == [1979, 1980]
    assert manifest["grids"] == [[61, 11]]
    assert manifest["periods"] == 5000 and manifest["burn"] == 1000
    assert manifest["measurement"] == "raw ergodic (ergodic_moments)"
    case = manifest["cases"]["theta=0.15"]["61x11"]
    assert {"monotone_certification_gap", "elapsed_s"} <= set(case)
    table = (tmp_path / "table.md").read_text(encoding="utf-8")
    assert "Escenario" in table
    assert any(ch.isdigit() and nxt == "," and after.isdigit()
               for ch, nxt, after in zip(table, table[1:], table[2:]))
    assert "Δ" not in table and "†" not in table


def test_sweep_requires_coarse(capsys):
    assert brief.main(["--sweep"]) == 2
    assert "--freeze" in capsys.readouterr().err


def test_no_arguments_prints_help(capsys, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no solve without arguments")

    monkeypatch.setattr(brief, "solve_scenario", refuse)
    assert brief.main([]) == 0
    assert "--scenario" in capsys.readouterr().out


def test_coarse_prints_caveat(capsys, tmp_path):
    assert brief.main(["--scenario", "baseline", "--coarse", "--grid", "31x11", "--periods", "3000",
                       "--n-seeds", "2", "--output", str(tmp_path)]) == 0
    assert "Coarse grid (801 x 161): for orientation only" in capsys.readouterr().out
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["coarse_grid_caveat"] is True


@pytest.mark.parametrize("lang,label,footnote", [("en", "Scenario", "†"), ("es", "Escenario", "†")])
def test_render_table_marks_failed_rows(tiny_record, lang, label, footnote):
    record = copy.deepcopy(tiny_record)
    for scenario in record["scenarios"].values():
        scenario["grid_change"]["passed"] = True
    record["scenarios"]["rho_y=0.97"]["grid_change"]["passed"] = False
    table = brief.render_table(record, lang)
    lines = table.splitlines()
    assert lines[0].startswith(f"| {label} |")
    rows = [line for line in lines if line.startswith("| ") and not line.startswith(f"| {label}")]
    assert len(rows) == 7
    marked = [row for row in rows if "†" in row.split("|")[1]]
    assert len(marked) == 1 and "0.97" in marked[0].replace(",", ".")
    assert rows[0].split("|")[1].strip().startswith("**")
    assert any(line.startswith("†") for line in lines), "footnote"
    assert "Δ" in table
    if lang == "es":
        assert "61.0" not in table  # no English decimal points in numbers
    clean = copy.deepcopy(record)
    clean["scenarios"]["rho_y=0.97"]["grid_change"]["passed"] = True
    assert "†" not in brief.render_table(clean, lang)


@pytest.mark.parametrize("lang", ["en", "es"])
def test_render_answer_is_one_paragraph_with_table_means(tiny_record, lang):
    text = brief.render_answer(tiny_record, lang)
    assert "\n\n" not in text.strip()
    debt = tiny_record["scenarios"]["theta=0.15"]["cases"]["61x11"]["simulation"]["summary"]["debt_output_pct"]["mean"]
    number = f"{debt:.3f}" if lang == "en" else f"{debt:.3f}".replace(".", ",")
    assert number in text
    words = ("raises", "lowers") if lang == "en" else ("sube", "baja")
    assert any(w in text for w in words)


@pytest.mark.parametrize("lang", ["en", "es"])
def test_render_figure_has_five_panels(tiny_record, lang):
    fig = brief.render_figure(tiny_record, lang)
    assert len(fig.axes) == 5


@pytest.mark.slow
def test_coarse_sweep_smoke(tmp_path, capsys):
    import time

    from _timing import budget

    start = time.perf_counter()
    assert brief.main(["--sweep", "--coarse", "--output", str(tmp_path)]) == 0
    elapsed = time.perf_counter() - start
    assert elapsed <= budget(1800), f"{elapsed:.0f} s > budget {budget(1800):.0f} s"
    assert "Coarse grid (801 x 161)" in capsys.readouterr().out
    rows = (tmp_path / "table.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 7
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["grids"] == [[801, 161]] and manifest["coarse_grid_caveat"] is True
    record = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    welfare = [record["scenarios"][n]["cases"]["801x161"]["welfare_pct"] for n in ("theta=0.15", "baseline", "theta=0.5")]
    assert welfare[0] > welfare[1] > welfare[2], welfare


# --- Final review fixes ---------------------------------------------------------

def _only(record, names, grids=None):
    out = copy.deepcopy(record)
    out["scenarios"] = {n: out["scenarios"][n] for n in names}
    if grids is not None:
        out["grids"] = [list(g) for g in grids]
        for s in out["scenarios"].values():
            s["cases"] = {f"{g[0]}x{g[1]}": s["cases"][f"{g[0]}x{g[1]}"] for g in grids}
            s.pop("grid_change", None)
    return out


@pytest.mark.parametrize("lang", ["en", "es"])
@pytest.mark.parametrize("names", [["theta=0.15"], ["baseline"], ["baseline", "theta=0.15"]])
def test_answer_is_empty_without_a_low_high_pair(tiny_record, lang, names):
    assert brief.render_answer(_only(tiny_record, names), lang) == ""


@pytest.mark.parametrize("lang,phrase", [("en", "grid change"), ("es", "cambio de malla")])
def test_answer_mentions_grid_change_only_with_two_grids(tiny_record, lang, phrase):
    assert phrase in brief.render_answer(tiny_record, lang)
    single = _only(tiny_record, list(brief.SCENARIOS), grids=[TINY_GRIDS[0]])
    text = brief.render_answer(single, lang)
    assert text and phrase not in text


def test_single_scenario_outputs_have_no_empty_answer(tmp_path, capsys):
    assert brief.main(["--scenario", "theta=0.15", "--grid", "61x11", "--periods", "3000",
                       "--n-seeds", "2", "--output", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    table = (tmp_path / "table.md").read_text(encoding="utf-8")
    for text in (out, table):
        assert ": ." not in text and "high value:" not in text
    assert table.startswith("| Scenario |")


@pytest.mark.parametrize("lang,column,banned", [
    ("en", "Debt (% of mean quarterly income)", ("debt/output", "Debt/output", "sustainable")),
    ("es", "Deuda (% del ingreso trimestral medio)", ("deuda/producto", "Deuda/producto", "sostenible")),
])
def test_debt_is_labelled_as_a_share_of_quarterly_income(tiny_record, lang, column, banned):
    table = brief.render_table(tiny_record, lang)
    answer = brief.render_answer(tiny_record, lang)
    fig = brief.render_figure(tiny_record, lang)
    assert column in table.splitlines()[0]
    assert fig.axes[0].get_title(loc="left") == column
    for word in banned:
        assert word not in table and word not in answer


# --- Windows portability ---------------------------------------------------------

def _text_io_without_encoding(path):
    """Calls of write_text/read_text/open in ``path`` that pass no encoding= (text mode)."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if name not in {"write_text", "read_text", "open"}:
            continue
        keywords = {k.arg for k in node.keywords}
        mode = next((a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)), "r")
        mode = next((k.value.value for k in node.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)), mode)
        if "b" in str(mode) or "encoding" in keywords:
            continue
        missing.append(f"{path.name}:{node.lineno} {name}")
    return missing


def test_every_text_io_in_the_brief_names_its_encoding():
    from pathlib import Path

    root = Path(brief.__file__).resolve().parents[2]
    files = [Path(brief.__file__), Path(__file__), root / "tests" / "test_sovereign_brief_record.py"]
    missing = [m for f in files for m in _text_io_without_encoding(f)]
    assert missing == [], missing


def test_cli_runs_without_getloadavg_or_resource(tmp_path, monkeypatch):
    # Windows has neither os.getloadavg nor the resource module.
    import sys

    monkeypatch.delattr(os, "getloadavg", raising=False)
    monkeypatch.setitem(sys.modules, "resource", None)  # makes `import resource` raise ImportError
    assert brief.main(["--scenario", "theta=0.15", "--grid", "31x11", "--periods", "3000",
                       "--n-seeds", "2", "--output", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["peak_rss_gb"] is None
    assert manifest["load_averages"] == {}
    assert manifest["load_average_status"] == "unavailable on this platform"


def test_brief_is_mypy_clean_on_windows():
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    if importlib.util.find_spec("mypy") is None:
        pytest.skip("mypy not installed")
    root = Path(brief.__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, "-m", "mypy", "--platform", "win32", "--ignore-missing-imports",
                           "--follow-imports=silent", "--no-incremental", str(Path(brief.__file__))],
                          cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("lang", ["en", "es"])
def test_custom_scenario_figure_contains_its_measured_values(tiny_record, lang):
    record = _only(tiny_record, ["baseline"])
    scenario = record["scenarios"].pop("baseline")
    scenario["overrides"] = {"theta": 0.3}
    record["scenarios"]["theta=0.3"] = scenario
    fig = brief.render_figure(record, lang)
    case = scenario["cases"]["61x11"]
    expected = [case["simulation"]["summary"][m]["mean"] for m in MOMENTS] + [case["welfare_pct"]]
    for ax, value in zip(fig.axes, expected):
        assert len(ax.containers) == 1
        np.testing.assert_allclose(np.asarray(ax.containers[0].lines[0].get_ydata(), dtype=float), [value])
        assert any(("0.3" if lang == "en" else "0,3") in t.get_text() for t in ax.get_xticklabels())


@pytest.mark.parametrize("lang", ["en", "es"])
@pytest.mark.parametrize("lo,hi,direction", [(9.0, 10.0, "up"), (10.0, 9.0, "down"),
                                            (-9.0, -10.0, "down"), (1.0001, 1.0002, "flat")])
def test_answer_compares_numbers_across_decimal_boundaries(tiny_record, lang, lo, hi, direction):
    record = _only(tiny_record, ["theta=0.15", "theta=0.5"])
    for name, value in [("theta=0.15", lo), ("theta=0.5", hi)]:
        record["scenarios"][name]["cases"]["61x11"]["simulation"]["summary"]["debt_output_pct"]["mean"] = value
    expected = brief._TEXT[lang][direction] + " " + brief._TEXT[lang]["outcome"]["debt_output_pct"]
    assert expected in brief.render_answer(record, lang)
