"""Paired-seed grid ladder for the illustrative fiscal calibration and its frozen record."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRID_RECORD = ROOT / "tests" / "fixtures" / "sovereign_fiscal" / "fiscal_grid_convergence.json"


def test_fiscal_grid_check_uses_the_public_certified_solver(monkeypatch):
    from tools.reference_validation import check_fiscal_grids as grids

    calls = []

    def recorder(**kwargs):
        calls.append(kwargs)
        return "solution"

    monkeypatch.setattr(grids, "solve_fiscal_sovereign_default", recorder)
    assert grids.solve_case(1601, 321) == "solution"
    assert calls == [dict(grids.ILLUSTRATIVE, n_b=1601, n_y=321, policy_search="monotone")]
    assert grids.ILLUSTRATIVE == dict(tau=.2, psi=1., gamma_c=2., gamma_g=2.)


def test_fiscal_tolerances_cover_the_registered_moments():
    from puremacro.sovereign._fiscal_paths import MOMENTS
    from tools.reference_validation.check_fiscal_grids import TOLERANCES

    assert list(TOLERANCES) == [m for m in MOMENTS if m != "boundary_choice_share_pct"]
    assert TOLERANCES["corr_pb_y"] == .02
    assert all(v == .10 for k, v in TOLERANCES.items() if not k.startswith("corr_"))


def test_paired_comparison_accepts_custom_tolerances():
    from tools.reference_validation.check_arellano_grids import paired_comparison

    def runs(shift):
        return dict(runs=[dict(seed=s, moments=dict(debt_pct=1. + shift + .001 * s, corr_pb_y=.3 + shift))
                          for s in (1, 2, 3)])

    result = paired_comparison(runs(0.), runs(.05), tolerances=dict(debt_pct=.10, corr_pb_y=.02))
    assert set(result["moments"]) == {"debt_pct", "corr_pb_y"}
    assert result["moments"]["debt_pct"]["status"] == "passed"
    assert result["moments"]["corr_pb_y"]["status"] == "grid_shift_exceeds_tolerance"
    assert not result["passed"]


def test_frozen_fiscal_grid_record_recomputes_from_its_seeds():
    from tools.reference_validation.check_arellano_grids import paired_comparison
    from tools.reference_validation.check_fiscal_grids import ILLUSTRATIVE, TOLERANCES

    record = json.loads(GRID_RECORD.read_text())
    assert record["tolerances"] == TOLERANCES
    assert {k: record["calibration"][k] for k in ILLUSTRATIVE} == ILLUSTRATIVE
    assert record["calibration"]["policy_search"] == "monotone"
    assert {(c["n_b"], c["n_y"]) for c in record["cases"].values()} == {
        (n_b, n_y) for n_b in (1601, 3201, 6401) for n_y in (321, 641)}
    for case in record["cases"].values():
        design, diagnostics = case["design"], case["diagnostics"]
        assert design["seeds"] == list(range(1979, 1995))
        assert (design["periods"], design["burn"]) == (500000, 1000)
        assert diagnostics["converged"] is True
        assert diagnostics["monotone_certification_gap"] <= 1e-11
        runs = case["simulation"]["runs"]
        assert [run["seed"] for run in runs] == design["seeds"]
        assert all(run["moments"]["boundary_choice_share_pct"] == 0 for run in runs)
    comparisons = record["comparisons"]
    assert set(comparisons) == {"final_asset_refinement", "final_income_refinement"}
    assert (comparisons["final_asset_refinement"]["coarse"], comparisons["final_asset_refinement"]["fine"]) \
        == ("fiscal_3201_641", "fiscal_6401_641")
    assert (comparisons["final_income_refinement"]["coarse"], comparisons["final_income_refinement"]["fine"]) \
        == ("fiscal_6401_321", "fiscal_6401_641")
    # Recompute from per-seed moments: the verdict must not rest on the stored summary.
    for comparison in comparisons.values():
        recomputed = paired_comparison(record["cases"][comparison["coarse"]]["simulation"],
                                       record["cases"][comparison["fine"]]["simulation"],
                                       tolerances=TOLERANCES)
        assert recomputed["passed"] == comparison["passed"]
        for name, value in recomputed["moments"].items():
            assert value["status"] == comparison["moments"][name]["status"]
            assert value["absolute_change_bound"] == pytest.approx(
                comparison["moments"][name]["absolute_change_bound"], abs=1e-12)
    all_passed = all(c["passed"] for c in comparisons.values())
    assert (record["status"] == "passed") == all_passed
    assert record["status"] in ("passed", "convergence_not_established")


@pytest.mark.parametrize("guide,thousands,decimal", [("docs/sovereign_fiscal_rules.md", ",", "."),
                                                     ("docs/es/sovereign_fiscal_rules.md", ".", ",")])
def test_fiscal_guides_tabulate_the_frozen_record(guide, thousands, decimal):
    from tools.reference_validation.check_fiscal_grids import GUIDE_COLUMNS

    assert GUIDE_COLUMNS == ("debt_pct", "spread_mean_pct", "annual_default_pct", "spread_std_pp",
                             "spending_pct", "corr_pb_y")
    record = json.loads(GRID_RECORD.read_text())
    text = (ROOT / guide).read_text(encoding="utf-8")
    for case in record["cases"].values():
        moments = case["simulation"]["moments"]
        nodes = f'{case["n_b"]:,}'.replace(",", thousands)
        row = f'| {nodes} × {case["n_y"]} | ' + " | ".join(
            f'{moments[k]["mean"]:.3f}'.replace(".", decimal) for k in GUIDE_COLUMNS) + " |"
        assert row in text, row
    verdict = {"passed": ("passed", "superada"), "convergence_not_established": (
        "convergence not established", "convergencia no establecida")}[record["status"]]
    assert verdict[0 if decimal == "." else 1] in text
    # The guide reports the remaining income-grid drift of the spread SD at the finest debt grid.
    sd = [record["cases"][name]["simulation"]["moments"]["spread_std_pp"]["mean"]
          for name in ("fiscal_6401_321", "fiscal_6401_641")]
    drift = " → ".join(f"{v:.3f}".replace(".", decimal) for v in sd)
    assert drift in text, drift


@pytest.mark.parametrize("changed", ["calibration", "search"])
def test_cached_solution_and_simulation_follow_effective_calibration(tmp_path, monkeypatch, changed):
    from argparse import Namespace
    from tools.reference_validation import check_fiscal_grids as grids

    args = Namespace(output=tmp_path, replications=2, periods=2000, burn=100)
    source_hash = "fixed-solver-source"
    first = grids.case(31, 7, args, source_hash)
    assert grids.case(31, 7, args, source_hash) == first
    if changed == "calibration":
        monkeypatch.setattr(grids, "ILLUSTRATIVE", {**grids.ILLUSTRATIVE, "tau": 0.3})
    else:
        monkeypatch.setattr(grids, "POLICY_SEARCH", "dense")
    second = grids.case(31, 7, args, source_hash)
    assert second["design"] != first["design"]
    manifest = json.loads((tmp_path / "fiscal_31_7_solution.json").read_text(encoding="utf-8"))
    assert manifest["calibration"] == grids._calibration()
    assert second["design"]["solution_identity"] == manifest["solution_identity"]
    assert manifest["scalars"]["tau"] == grids.ILLUSTRATIVE["tau"]


# ------------------------------------------------------------- registered fiscal-rule predictions
RULE_PROTOCOL = ROOT / "tests" / "fixtures" / "sovereign_fiscal" / "rule_predictions_protocol.json"
RULE_RECORD = ROOT / "tests" / "fixtures" / "sovereign_fiscal" / "rule_predictions_record.json"
RULE_ROWS = ("none", "ceiling", "ceiling_escape", "soft_ceiling", "deficit")
RULE_COLUMNS = ("debt_pct", "spread_mean_pct", "annual_default_pct", "spread_std_pp")


def test_rule_record_uses_the_registered_protocol():
    import hashlib

    text = RULE_PROTOCOL.read_text()
    record = json.loads(RULE_RECORD.read_text())
    assert record["protocol"] == json.loads(text)
    assert record["protocol_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert (record["protocol"]["design"]["grid"]["n_b"], record["protocol"]["design"]["grid"]["n_y"]) == (401, 21)
    for name, solve in record["solves"].items():
        assert solve["converged"] and solve["rule_infeasible_states"] == 0
        assert solve["policy_search"] == record["protocol"]["policy_search"][name]
        if solve["policy_search"] == "monotone":
            assert solve["monotone_certification_gap"] <= 1e-11
    for case in record["cases"].values():
        assert [run["seed"] for run in case["runs"]] == record["protocol"]["design"]["seeds"]


def test_escape_clause_within_registered_tolerance_of_ceiling():
    from tools.reference_validation.check_arellano_grids import paired_comparison

    record = json.loads(RULE_RECORD.read_text())
    tolerances = json.loads(RULE_PROTOCOL.read_text())["tolerances"]
    comparison = paired_comparison(record["cases"]["ceiling"], record["cases"]["ceiling_escape"],
                                   tolerances=tolerances)
    assert comparison["passed"], comparison
    cap = record["protocol"]["rules"]["ceiling"]["debt_ceiling"]
    assert all(v < cap for v in record["measurements"]["max_b_recession"].values())


def test_rule_prediction_verdicts_recompute_from_measurements():
    record = json.loads(RULE_RECORD.read_text())
    m, p = record["measurements"], record["protocol"]["predictions"]
    lo, hi = p["ceiling_binding_share_pct"]
    dlo, dhi = p["deficit_binding_share_pct"]
    assert record["predictions"]["ceiling_binding_share_pct"]["held"] == (lo <= m["ceiling_binding"]["binding_share_pct"] <= hi)
    assert record["predictions"]["deficit_binding_share_pct"]["held"] == (dlo <= m["deficit_binding"]["binding_share_pct"] <= dhi)
    assert record["predictions"]["ceiling_binding_in_booms"]["held"] == (m["ceiling_binding"]["binding_in_booms"] == 1.0)
    assert record["predictions"]["escape_inert"]["held"] == (
        all(v < 0.02 for v in m["max_b_recession"].values()) and m["escape_vs_ceiling"]["passed"])


@pytest.mark.parametrize("guide,decimal", [("docs/sovereign_fiscal_rules.md", "."),
                                           ("docs/es/sovereign_fiscal_rules.md", ",")])
def test_guides_report_the_rule_record(guide, decimal):
    record = json.loads(RULE_RECORD.read_text())
    text = (ROOT / guide).read_text(encoding="utf-8")

    def fmt(value, digits):
        return f"{value:.{digits}f}".replace(".", decimal)

    for name in RULE_ROWS:
        moments = record["cases"][name]["moments"]
        row = f"| `{name}` | " + " | ".join(fmt(moments[k], 3) for k in RULE_COLUMNS) + " |"
        assert row in text, row
    m = record["measurements"]
    for value, digits in ((m["penalty_output_equivalent"], 4), (m["ceiling_binding"]["binding_share_pct"], 1),
                          (m["deficit_binding"]["binding_share_pct"], 1), (m["max_b_recession"]["401x21"], 5),
                          (m["max_b_recession"]["801x41"], 5),
                          (m["soft_ceiling_policy_above_cap"]["min_current_debt"], 5)):
        assert fmt(value, digits) in text, fmt(value, digits)
    assert f'{m["soft_ceiling_policy_above_cap"]["repayment_states"]} ' in text
    assert ("COARSE 401 × 21" if decimal == "." else "MALLA GRUESA 401 × 21") in text
    assert ("not held" if decimal == "." else "no se cumplió") in text


def test_soft_ceiling_off_the_ergodic_set_is_recorded_and_recomputes():
    import numpy as np

    from puremacro.sovereign import FiscalRule, solve_fiscal_sovereign_default

    info = json.loads(RULE_RECORD.read_text())["measurements"]["soft_ceiling_policy_above_cap"]
    soft = solve_fiscal_sovereign_default(tau=.2, psi=1., n_b=401, n_y=21,
                                          rule=FiscalRule(debt_ceiling=.02, penalty=.05))
    hard = solve_fiscal_sovereign_default(tau=.2, psi=1., n_b=401, n_y=21, rule=FiscalRule(debt_ceiling=.02))
    above = (soft.policy_b > .02 + 1e-12) & (soft.default_prob < .5)
    rows = np.nonzero(above)[0]
    assert info["repayment_states"] == int(above.sum()) > 0
    assert info["min_current_debt"] == float(soft.b_grid[rows].min()) > .02
    at_or_below = soft.b_grid <= .02
    assert info["policies_equal_at_or_below_cap"] is True
    assert np.array_equal(soft.policy_index[at_or_below], hard.policy_index[at_or_below])


def test_rule_record_recomputes_from_code_for_one_seed():
    from tools.reference_validation import check_fiscal_rule_predictions as tool

    record = json.loads(RULE_RECORD.read_text())
    protocol = json.loads(RULE_PROTOCOL.read_text())
    design = dict(protocol["design"], seeds=[1979])
    grid = protocol["design"]["grid"]
    results = {}
    for name in ("none", "ceiling", "deficit"):
        res = tool.solve(protocol, name, grid["n_b"], grid["n_y"])
        assert res.iterations == record["solves"][name]["iterations"]
        paths = tool._paths(res, design)
        (run,) = tool._runs(paths, design)
        expected = record["cases"][name]["runs"][0]
        # Moments are sums over 100,000 quarters; allow last-ulp differences
        # across NumPy builds, keep every other field exact.
        assert run.keys() == expected.keys(), name
        for key, value in expected.items():
            if key == "moments":
                assert run[key].keys() == value.keys(), name
                for moment, number in value.items():
                    assert run[key][moment] == pytest.approx(number, rel=1e-12, abs=1e-15), (name, moment)
            else:
                assert run[key] == value, (name, key)
        results[name] = (res, paths)
    free, free_paths = results["none"]
    binding = tool._binding(free, free_paths, design["burn"], protocol["rules"]["ceiling"]["debt_ceiling"], None)
    assert binding["per_seed_pct"] == record["measurements"]["ceiling_binding"]["per_seed_pct"][:1]
    assert binding["per_seed_pct"][0] == pytest.approx(23.479, abs=5e-4)
