"""Check retained HANK refinement evidence without erasing unsuccessful cases."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "tests/fixtures/hank_scenario_tree"


def _record():
    return json.loads((EVIDENCE / "record.json").read_text())


def test_tree_record_authenticates_sources_and_declares_omitted_arrays():
    record = _record()
    assert record["extension_complete"]
    for name, digest in record["sources"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    retention = json.loads((EVIDENCE / "retention.json").read_text())
    for name, digest in retention["retained"].items():
        assert hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest() == digest, name
    assert retention["omitted_household_arrays"]
    for name, digest in retention["omitted_household_arrays"].items():
        assert name.endswith(".npz") and not (EVIDENCE/name).exists()
        assert record["artifacts"][name] == digest


def test_failed_solutions_and_refinements_remain_failed():
    record = _record()
    assert not record["all_refinements_passed"]
    assert not record["cases"]["narrow11"]["converged"]
    assert record["cases"]["narrow11"]["diagnostics"]["policy_grid_exit_mass"] >= 1e-10
    assert not record["cases"]["narrow15"]["converged"]
    assert "invalid invariant distribution" in record["cases"]["narrow15"]["error"]
    for comparison in record["refinements"]:
        coarse, fine = comparison["coarse"], comparison["fine"]
        if "variable" not in comparison:
            assert not comparison["passed"] and comparison["reason"]
            continue
        left = pd.read_csv(EVIDENCE/f"{coarse}_nodes.csv", index_col=0)
        right = pd.read_csv(EVIDENCE/f"{fine}_nodes.csv", index_col=0)
        labels = left.index[left.period < comparison["window"]]
        variable = comparison["variable"]
        change = float((left.loc[labels, variable]-right.loc[labels, variable]).abs().max())
        tolerance = max(1e-6, .01*float(right.loc[labels, variable].abs().max()))
        assert change == pytest.approx(comparison["maximum_change"], abs=1e-15)
        assert tolerance == pytest.approx(comparison["tolerance"], abs=1e-15)
        solved = record["cases"][coarse]["converged"] and record["cases"][fine]["converged"]
        assert comparison["passed"] == bool(solved and change <= tolerance)


def test_recorded_expectations_use_history_probabilities_and_match_markets():
    for name, case in _record()["cases"].items():
        if not case["converged"]:
            continue
        nodes = pd.read_csv(EVIDENCE/f"{name}_nodes.csv", index_col=0)
        expected = pd.read_csv(EVIDENCE/f"{name}_expectations.csv", index_col=0)
        weighted = nodes[expected.columns].mul(nodes.probability, axis=0)
        np.testing.assert_allclose(weighted.groupby(nodes.period).sum(), expected, atol=1e-14, rtol=0)
        np.testing.assert_allclose(nodes.groupby("period").probability.sum(), 1, atol=1e-14)
        residual = nodes[["asset_mkt", "labor_mkt", "goods_mkt", "nkpc_res"]].abs().to_numpy().max()
        assert residual == pytest.approx(case["diagnostics"]["market_residual_max_abs"], abs=1e-15)
        assert residual <= case["diagnostics"]["tolerance"]


def test_expanded_bound_and_final_income_asset_horizon_comparisons_pass():
    record = _record()
    required = {("dense15", "bound600"), ("income21", "income31"),
                ("income31", "assets31"), ("income31", "horizon31")}
    checked = [c for c in record["refinements"] if (c["coarse"], c["fine"]) in required]
    assert len(checked) == len(required) * 3
    assert all(c["passed"] for c in checked)
    assert record["deterministic_oracle"]["passed"]
