"""The frozen converged sovereign-risk brief sweep and the pages generated from it.

The record is produced once by the maintainer
(``python -m puremacro.examples.first_brief_sovereign --freeze
tests/fixtures/sovereign_brief/sweep.json``, 6,851 s or 1.9 h measured under
load on a shared workstation, including the error grid); these tests
read it and never re-run the sweep.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

from puremacro.examples import first_brief_sovereign as brief
from puremacro.sovereign.measurement import MOMENTS

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "tests" / "fixtures" / "sovereign_brief" / "sweep.json"


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def _cases(record):
    for name, scenario in record["scenarios"].items():
        for key, case in scenario["cases"].items():
            yield name, key, case


def test_record_design(record):
    assert list(record["scenarios"]) == list(brief.SCENARIOS)
    assert record["grids"] == [[6401, 641], [3201, 641]]
    assert record["design"] == brief.DESIGN
    assert record["tolerances"] == brief.TOLERANCES
    assert record["baseline"] == brief.BASELINE
    for name, scenario in record["scenarios"].items():
        assert scenario["overrides"] == brief.SCENARIOS[name]
        assert list(scenario["cases"]) == ["6401x641", "3201x641"]
    assert sum(1 for _ in _cases(record)) == 14


def test_record_solver_diagnostics(record):
    for name, key, case in _cases(record):
        solver = case["solver"]
        assert solver["converged"], (name, key)
        assert solver["monotone_certification_gap"] <= 1e-11, (name, key)
        assert solver["residual"] < 1e-10, (name, key)
        assert solver["exhaustive_bellman_residual"] < 2e-11, (name, key)


def test_record_recomputes_from_per_seed(record):
    for name, key, case in _cases(record):
        sim = case["simulation"]
        assert sim["seeds"] == brief.DESIGN["seeds"]
        n = len(sim["seeds"])
        t_crit = student_t.ppf(.975, n - 1)
        for m in MOMENTS:
            values = np.asarray(sim["per_seed"][m])
            se = values.std(ddof=1) / np.sqrt(n)
            summary = sim["summary"][m]
            assert summary["mean"] == pytest.approx(values.mean(), abs=1e-12), (name, key, m)
            assert summary["se"] == pytest.approx(se, abs=1e-12), (name, key, m)
            assert summary["ci95"] == pytest.approx([values.mean() - t_crit * se, values.mean() + t_crit * se],
                                                    abs=1e-12), (name, key, m)
    for name, scenario in record["scenarios"].items():
        expected = brief.paired_change(scenario["cases"]["3201x641"], scenario["cases"]["6401x641"])
        assert scenario["grid_change"]["passed"] == expected["passed"], name
        assert scenario["grid_change"]["welfare_change_pct"] == pytest.approx(expected["welfare_change_pct"], abs=1e-12)
        for m in MOMENTS:
            got, want = scenario["grid_change"]["moments"][m], expected["moments"][m]
            assert got["status"] == want["status"], (name, m)
            for field in ("change", "absolute_change_bound", "tolerance"):
                assert got[field] == pytest.approx(want[field], abs=1e-12), (name, m, field)
            assert got["ci95"] == pytest.approx(want["ci95"], abs=1e-12)


def test_record_has_no_local_paths():
    text = RECORD_PATH.read_text(encoding="utf-8")
    for forbidden in ("/Users/", "/home/", "/private/", "C:\\\\", "\\\\Users\\\\"):
        assert forbidden not in text, forbidden


def test_baseline_is_near_the_study_record(record):
    # A sanity link to the reference study at the same grid, not equality: the
    # study measures 74-quarter pre-default windows, this record raw ergodic
    # moments with a linear default annualization.
    summary = record["scenarios"]["baseline"]["cases"]["6401x641"]["simulation"]["summary"]
    assert abs(summary["debt_output_pct"]["mean"] - 4.003) <= 1.5
    assert abs(summary["default_annual_pct"]["mean"] - 2.997) <= 1.0


# --- The bilingual pages ------------------------------------------------------

PAGES = {"en": ROOT / "docs" / "brief_sovereign_risk.md", "es": ROOT / "docs" / "es" / "brief_sovereign_risk.md"}


def _first_paragraph(text: str) -> str:
    body = text.split("\n# ", 1)[1].split("\n", 1)[1].strip()
    return body.split("\n\n", 1)[0]


@pytest.mark.parametrize("lang", ["en", "es"])
def test_pages_carry_generated_table_and_answer(record, lang):
    page = PAGES[lang].read_text(encoding="utf-8")
    assert brief.render_table(record, lang) in page
    assert brief.render_answer(record, lang) in page


@pytest.mark.parametrize("lang,words", [("en", ("illustrative", "not a claim about any country")),
                                        ("es", ("ilustrativ", "ningún país"))])
def test_pages_frame_the_research_rule(lang, words):
    first = _first_paragraph(PAGES[lang].read_text(encoding="utf-8"))
    for word in words:
        assert word in first, word
    assert "arellano_replication.md" in first


@pytest.mark.parametrize("lang,headings", [
    ("en", ["## Question", "## Answer", "## Results", "## Run it yourself", "## Assumptions and limits"]),
    ("es", ["## Pregunta", "## Respuesta", "## Resultados", "## Ejecútelo usted mismo", "## Supuestos y límites"]),
])
def test_pages_have_required_sections(lang, headings):
    page = PAGES[lang].read_text(encoding="utf-8")
    lines = page.splitlines()
    positions = [lines.index(h) for h in headings]
    assert positions == sorted(positions)
    assert "python -m puremacro.examples.first_brief_sovereign --scenario theta=0.15" in page
    assert ("Español" if lang == "en" else "English") in page[:600]
    assert "```python" not in page


@pytest.mark.parametrize("lang,ref", [("en", "assets/brief_sovereign_risk.en.svg"),
                                      ("es", "../assets/brief_sovereign_risk.es.svg")])
def test_figure_assets_exist_and_are_referenced(lang, ref):
    assert (ROOT / "docs" / "assets" / f"brief_sovereign_risk.{lang}.svg").is_file()
    assert f"]({ref})" in PAGES[lang].read_text(encoding="utf-8")


def test_timings_are_quoted_consistently(record, capsys):
    # One timing source: the frozen theta=0.15 converged solve (measured on a
    # shared machine under load) and the full freeze wall time, 6,851 s (1.9 h).
    case = record["scenarios"]["theta=0.15"]["cases"]["6401x641"]
    minutes = round(case["solver"]["elapsed_s"] / 60)
    help_text = brief._parser().format_help()
    assert f"about {minutes} minutes" in " ".join(help_text.split())
    assert "under load" in " ".join(help_text.split())
    assert brief.main(["--sweep"]) == 2
    err = " ".join(capsys.readouterr().err.split())
    assert "1.9 h" in err and "under load" in err
    load = record["load_averages"]["theta=0.15 6401x641"]
    for lang, page_path in PAGES.items():
        page = " ".join(page_path.read_text(encoding="utf-8").split())
        assert (f"{minutes} minutes" if lang == "en" else f"{minutes} minutos") in page
        assert ("under load" if lang == "en" else "bajo carga") in page
        assert "6,851 s (1.9 h)" in page or "6.851 s (1,9 h)" in page
        loads = "/".join(f"{v:.0f}" for v in load)
        assert loads in page, loads
    assert "1.9 h" in " ".join(__doc__.split())


_LIBRARY_SOURCES = ("puremacro/sovereign/arellano.py", "puremacro/sovereign/_monotone.py",
                    "puremacro/sovereign/_results.py", "puremacro/sovereign/measurement.py")
_EXAMPLE = "puremacro/examples/first_brief_sovereign.py"
# The example's bytes changed after the freeze (record hash 4823e9ef…, commit
# 1752c83) without touching solve_scenario, measure_case or paired_change:
# 60a04ce0 added one type annotation, then the final-review fixes (caad961d,
# 38682a89, 072e549f) changed only answer/table/figure wording and CLI text;
# the Windows fixes add explicit utf-8 encodings and platform guards.
# Any further edit must be reviewed the same way and its hash added here.
_EXAMPLE_HASHES = {
    "51994374f6df3515d167147b93f6d455c561843edc81d3656489f4157cf1003d",  # resume validation only; solve and measurement are unchanged
    "4823e9efa4b98fd3f1cd309baf1d5f79ebe201b9c918e3f15f0a17b297fd5858",  # frozen (1752c83)
    "819b52aeac1762b1c93d3bd5ad18b7bd85d7de0e01097aa2a172725929b94d58",  # current: portable I/O and corrected presentation; numerical generation unchanged
}


# The fiscal-core extraction and added fiscal result class preserve the old
# short-term solver and simulator bit for bit. The pinned original/new pairs
# are supported by test_brief_scenarios_bit_identical_to_legacy and the full
# 6.0.1 parity harness; unknown edits still invalidate the frozen record.
# The later extension corrections change only long-term debt simulations and
# collateral diagnostics. Every brief calibration and short-term random path
# remains bit-identical in the parity tests; measurement parity is also tested.
_EQUIVALENT_LIBRARY_HASHES = {
    # Income batches preserve the pinned original search bit for bit, including
    # tied choices across batch boundaries (test_monotone_search.py).
    'puremacro/sovereign/_monotone.py': (
        'fb04bb672f59c89ec40bd51179df3a249df09e0ddeb7cd9102d6288e7b251464',
        'cb0655d94f4f5dc01c1f71979e8bb71551d98ed5ac90e18a79d4f420b3c7e467'),
    'puremacro/sovereign/arellano.py': (
        'd434bded913776d47a740ca5e35ce3b7a8e7047d3efbafce89b09d5e5b2655a2',
        # Explicit debt grids are optional; unchanged default calls retain the
        # pinned original equilibrium, metadata and simulation bit for bit.
        '2301827c26be1295a94d038c6a3a5c03afc459994b4cf2764b850a7d70ba795a'),
    'puremacro/sovereign/_results.py': (
        'fc97e14a0e10a450f9e2cacf965cffc8f42a1acc5e6d1c18fae9cb61a1cec3db',
        '21e0bd69e131b5bf3171c96ed79bc0ebf2ed38e86f115a8de82cf6491e922a98'),
    'puremacro/sovereign/measurement.py': (
        '104c41d31181121ae543d9544c908726435bbaa15a3f605f4b2c071bb2957365',
        '3a3588d95722c08b8c4392852d5bc41420d20bde29f12e514d876228b38c90a3'),
}


def _provenance_problems(recorded: dict, current: dict) -> list[str]:
    problems = [f"{path}: record {recorded.get(path)} != current {current[path]}"
                for path in _LIBRARY_SOURCES
                if recorded.get(path) != current[path]
                and (recorded.get(path), current[path]) != _EQUIVALENT_LIBRARY_HASHES.get(path)]
    if recorded.get(_EXAMPLE) not in _EXAMPLE_HASHES:
        problems.append(f"{_EXAMPLE}: record hash {recorded.get(_EXAMPLE)} not in the allow-list")
    if current[_EXAMPLE] not in _EXAMPLE_HASHES:
        problems.append(f"{_EXAMPLE}: current hash {current[_EXAMPLE]} not in the allow-list")
    if set(recorded) != set(_LIBRARY_SOURCES) | {_EXAMPLE}:
        problems.append(f"unexpected source keys {sorted(recorded)}")
    return problems


def test_record_sources_match_the_current_files(record):
    assert _provenance_problems(record["source_sha256"], brief.source_hashes()) == []


@pytest.mark.parametrize("changed", [None, "overrides", "missing_overrides", "baseline"])
def test_resume_reuses_only_the_same_scenario(record, tmp_path, monkeypatch, changed):
    saved = deepcopy(record)
    saved["source_sha256"] = brief.source_hashes()
    saved["scenarios"] = {"custom": saved["scenarios"]["theta=0.15"]}
    partial = tmp_path / "partial.json"
    overrides = {"theta": .15}
    if changed == "overrides":
        overrides = {"theta": .50}
    elif changed == "missing_overrides":
        del saved["scenarios"]["custom"]["overrides"]
    elif changed == "baseline":
        saved["baseline"]["beta"] = .90
    partial.write_text(json.dumps(saved), encoding="utf-8")
    calls = []
    fresh = deepcopy(record["scenarios"]["theta=0.5"]["cases"]["6401x641"])

    def solve(parameters, n_b, n_y):
        calls.append((parameters, n_b, n_y))
        return object(), 0.

    monkeypatch.setattr(brief, "solve_scenario", solve)
    monkeypatch.setattr(brief, "measure_case", lambda *args, **kwargs: fresh)
    resumed = brief.build_record([(6401, 641)], design=brief.DESIGN, partial=partial,
                                 scenarios={"custom": overrides}, log=lambda message: None)
    assert resumed["scenarios"]["custom"]["overrides"] == overrides
    result = resumed["scenarios"]["custom"]["cases"]["6401x641"]
    if changed is None:
        assert not calls
        assert result == saved["scenarios"]["custom"]["cases"]["6401x641"]
    else:
        assert calls == [(overrides, 6401, 641)]
        assert result == fresh


@pytest.mark.parametrize("path", [*_LIBRARY_SOURCES, _EXAMPLE])
def test_provenance_check_catches_a_changed_source(record, path):
    perturbed = dict(record["source_sha256"])
    perturbed[path] = "0" * 64
    assert _provenance_problems(perturbed, brief.source_hashes())
    current = brief.source_hashes()
    current[path] = "f" * 64
    assert _provenance_problems(record["source_sha256"], current)


def _paired(record, low: str, high: str, moment: str) -> tuple[float, float]:
    a = np.asarray(record["scenarios"][low]["cases"]["6401x641"]["simulation"]["per_seed"][moment])
    b = np.asarray(record["scenarios"][high]["cases"]["6401x641"]["simulation"]["per_seed"][moment])
    d = b - a
    return float(d.mean()), float(student_t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


@pytest.mark.parametrize("lang", ["en", "es"])
def test_volatility_sentence_quotes_the_record(record, lang):
    # All of the volatility effect on spread and default comes between
    # sigma_y 0.02 and 0.025; from 0.025 to 0.03 neither rises.
    page = " ".join(PAGES[lang].read_text(encoding="utf-8").split())
    fmt = (lambda v, f=".3f": format(v, f)) if lang == "en" else (lambda v, f=".3f": format(v, f).replace(".", ","))
    for moment in ("spread_mean_pct", "default_annual_pct"):
        means = [record["scenarios"][n]["cases"]["6401x641"]["simulation"]["summary"][moment]["mean"]
                 for n in ("sigma_y=0.02", "baseline", "sigma_y=0.03")]
        assert " / ".join(fmt(v) for v in means) in page, moment
        change, half = _paired(record, "baseline", "sigma_y=0.03", moment)
        digits = ".4f" if moment == "spread_mean_pct" else ".3f"
        assert f"{fmt(change, '+' + digits)} ± {fmt(half, digits)}" in page, (moment, change, half)


@pytest.mark.parametrize("lang,words", [
    ("en", ("1 - (1 - entries / repayment quarters)^4", "compounded", "2.989", "2.997")),
    ("es", ("1 - (1 - entradas / trimestres de pago)^4", "compuesta", "2,989", "2,997")),
])
def test_measurement_bullet_explains_the_default_convention(record, lang, words):
    limits = PAGES[lang].read_text(encoding="utf-8").split("\n## ")[-1]
    text = " ".join(limits.split())
    for word in words:
        assert word in text, word
    baseline = record["scenarios"]["baseline"]["cases"]["6401x641"]["simulation"]["summary"]
    assert f"{baseline['default_annual_pct']['mean']:.3f}" == "2.989"


@pytest.mark.parametrize("lang,phrase", [
    ("en", "The levels for θ = 0.5 and ρ_y = 0.97 (marked † in the table) are not grid-converged"),
    ("es", "Los niveles para θ = 0,5 y ρ_y = 0,97 (marcados con † en la tabla) no están convergidos en la malla"),
])
def test_answer_flags_unconverged_levels(record, lang, phrase):
    failing = [n for n, s in record["scenarios"].items() if not s["grid_change"]["passed"]]
    assert failing == ["theta=0.5", "rho_y=0.97"]
    page = PAGES[lang].read_text(encoding="utf-8")
    answer = page.split(brief.render_answer(record, lang), 1)[1].split("\n## ", 1)[0]
    assert phrase in " ".join(answer.split())


@pytest.mark.parametrize("lang,phrase", [("en", "averaged over the long-run income distribution"),
                                         ("es", "promediados sobre la distribución de largo plazo del ingreso")])
def test_question_defines_welfare_over_the_long_run_distribution(lang, phrase):
    page = PAGES[lang].read_text(encoding="utf-8")
    question = page.split("\n## ", 2)[1]
    assert phrase in " ".join(question.split())
