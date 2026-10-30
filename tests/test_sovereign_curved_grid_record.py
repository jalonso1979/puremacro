"""The curved-grid experiment is distinct, complete and reproducibly assessed."""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.stats import t as student_t

from tools.reference_validation import check_sovereign_curved_grids as study


def test_curved_grids_are_nested_with_the_original_bounds_and_exact_zero():
    for coarse, fine in ((1601, 3201), (3201, 6401), (6401, 12801)):
        a, b = study.debt_grid(coarse), study.debt_grid(fine)
        np.testing.assert_array_equal(a, b[::2])
        assert a[0] == -.15 and a[-1] == .33
        assert np.count_nonzero(a == 0.) == 1
        assert np.all(np.diff(a) > 0.)
        assert a[len(a) * 5 // 16 + 1] < .000001
    with pytest.raises(ValueError, match="16"):
        study.debt_grid(100)


@pytest.fixture
def record():
    # Synthetic repeated observations isolate the decision contract. They are
    # never written to a study output or represented as convergence evidence.
    frozen = json.loads(study.INITIAL.read_text(encoding="utf-8"))
    template = frozen["scenarios"]["theta=0.5"]["cases"]["6401x641"]
    result = {**study.protocol(), "cases": {}}
    for b, y in study.GRIDS:
        case = deepcopy(template)
        case.update(n_b=b, n_y=y, debt_grid=study.debt_grid(b).tolist(),
                    debt_grid_sha256=study.grid_hash(study.debt_grid(b)),
                    source_sha256=dict(result["source_sha256"]))
        case["parameters"].update(n_b=b, n_y=y)
        result["cases"][f"{b}x{y}"] = case
    return result


def test_zero_change_passes_only_with_complete_registered_design(record):
    assert study.assess(record)["passed"]


@pytest.mark.parametrize("change", ["missing_case", "missing_zero", "grid_hash", "sources",
                                   "parameters", "seeds", "short_sample", "nan", "tolerance"])
def test_altered_or_incomplete_curved_experiments_are_rejected(record, change):
    case = record["cases"]["12801x1281"]
    if change == "missing_case":
        del record["cases"]["6401x641"]
    elif change == "missing_zero":
        case["debt_grid"][4000] = 1e-12
    elif change == "grid_hash":
        case["debt_grid_sha256"] = "unknown"
    elif change == "sources":
        case["source_sha256"]["puremacro/sovereign/arellano.py"] = "unknown"
    elif change == "parameters":
        case["parameters"]["gamma"] = 5.
    elif change == "seeds":
        case["simulation"]["seeds"][0] = 0
    elif change == "short_sample":
        case["simulation"]["per_seed"]["spread_sd_pp"].pop()
    elif change == "nan":
        case["simulation"]["per_seed"]["spread_sd_pp"][0] = float("nan")
    else:
        record["tolerances"] = {m: 1. for m in study.TOLERANCES}
    with pytest.raises(ValueError):
        study.assess(record)


@pytest.mark.parametrize("change", ["solve", "certificate", "residual", "nonfinite", "final_income"])
def test_failed_solve_or_final_income_comparison_cannot_pass(record, change):
    solver = record["cases"]["12801x1281"]["solver"]
    if change == "solve":
        solver["converged"] = False
    elif change == "certificate":
        solver["monotone_certification_gap"] = 1e-6
    elif change == "residual":
        solver["residual"] = 1e-6
    elif change == "nonfinite":
        solver["exhaustive_bellman_residual"] = float("nan")
    else:
        record["cases"]["12801x641"]["simulation"]["per_seed"]["spread_sd_pp"] = [10.] * 16
    result = study.assess(record)
    assert not result["passed"]
    if change == "final_income":
        assert result["comparisons"]["final_asset_refinement"]["passed"]
        assert not result["comparisons"]["final_income_refinement"]["passed"]


def test_combine_rejects_different_environments_and_duplicate_cases(record, tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    a, b = deepcopy(record), deepcopy(record)
    a["cases"] = {k: c for k, c in a["cases"].items() if c["n_b"] == 6401}
    b["cases"] = {k: c for k, c in b["cases"].items() if c["n_b"] == 12801}
    first.write_text(json.dumps(a), encoding="utf-8")
    second.write_text(json.dumps(b), encoding="utf-8")
    assert study.combine([first, second])["passed"]
    b["versions"]["numpy"] = "unknown"
    second.write_text(json.dumps(b), encoding="utf-8")
    with pytest.raises(ValueError, match="environments"):
        study.combine([first, second])
    with pytest.raises(ValueError, match="duplicate"):
        study.combine([first, first])


def test_frozen_curved_evidence_recomputes_moments_and_every_verdict():
    path = study.INITIAL.with_name("curved_grid_convergence.json")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    checked = study.assess(deepcopy(frozen))
    assert (checked["passed"], checked["status"]) == (frozen["passed"], frozen["status"])
    for case in frozen["cases"].values():
        assert study.validate_case(case, case["n_b"], case["n_y"], study.study_sources())
        for moment, per_seed in case["simulation"]["per_seed"].items():
            values = np.asarray(per_seed)
            mean = values.mean()
            se = values.std(ddof=1) / np.sqrt(len(values))
            half = student_t.ppf(.975, len(values) - 1) * se
            summary = case["simulation"]["summary"][moment]
            assert summary["mean"] == pytest.approx(mean, abs=1e-12, rel=0)
            assert summary["se"] == pytest.approx(se, abs=1e-12, rel=0)
            assert summary["ci95"] == pytest.approx([mean - half, mean + half], abs=1e-12, rel=0)
    for label, comparison in checked["comparisons"].items():
        saved = frozen["comparisons"][label]
        assert comparison["passed"] == saved["passed"]
        for moment, values in comparison["moments"].items():
            assert values["status"] == saved["moments"][moment]["status"]
            for key in ("change", "ci95", "absolute_change_bound", "tolerance"):
                assert values[key] == pytest.approx(saved["moments"][moment][key], abs=1e-12, rel=0)
