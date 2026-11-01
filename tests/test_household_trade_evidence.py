"""Recorded runs must retain their inputs, accounting and actual acceptance tests."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "tests/fixtures/household_trade_policy"


def test_observed_study_record_matches_its_sources_and_artifacts():
    record = json.loads((EVIDENCE / "record.json").read_text())
    assert record["passed"]
    for path, digest in record["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    runner = ROOT / "tools/reference_validation/check_household_trade.py"
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == record["runner_sha256"]
    for name, digest in record["artifacts"].items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == digest, name
    assert all(check["passed"] for check in record["checks"].values())


def test_recorded_policy_objectives_are_distinct_and_complete():
    aggregate = pd.read_csv(EVIDENCE / "static_aggregate.csv")
    assert len(aggregate) == 4 * 9
    assert not aggregate.duplicated(["policy", "state"]).any()
    population = pd.read_csv(EVIDENCE / "static_mean_ev_summary.csv", index_col=0)
    bottom = pd.read_csv(EVIDENCE / "static_bottom40_mean_ev_summary.csv", index_col=0)
    assert population.index[population.undominated].tolist() == ["no_compensation"]
    assert bottom.index[bottom.undominated].tolist() == ["bottom_40_income"]
    # Recompute regrets from the recorded outcomes, independently of the API.
    for prefix, metric in (("static", "bottom40_mean_ev"), ("dynamic", "bottom40_mean_cev")):
        payoffs = pd.read_csv(EVIDENCE / f"{prefix}_{metric}_payoffs.csv", index_col=0)
        summary = pd.read_csv(EVIDENCE / f"{prefix}_{metric}_summary.csv", index_col=0)
        regret = payoffs.max(axis=0).to_numpy() - payoffs.to_numpy()
        np.testing.assert_allclose(regret.max(axis=1), summary.maximum_regret, atol=1e-9)


def test_recorded_accounts_conserve_money_and_separate_switching_costs():
    baseline = pd.read_csv(EVIDENCE / "static_baseline.csv")
    assert baseline.sample_households.sum() == 90033
    assert baseline.population_households.sum() == 37538178
    incidence = pd.read_csv(EVIDENCE / "static_incidence.csv")
    for _, group in incidence.groupby(["policy", "state"]):
        np.testing.assert_allclose(group.population_share.sum(), 1, atol=1e-14)
        assert np.dot(group.population_share, group.mean_benefit-group.mean_tax) == pytest.approx(0, abs=1e-10)
    record = json.loads((EVIDENCE / "record.json").read_text())
    for case in record["sourcing"].values():
        accounts = pd.read_csv(EVIDENCE / f"{case['file_prefix']}_accounts.csv", index_col=0)
        np.testing.assert_allclose(accounts.total_unit_cost,
                                   accounts.procurement_cost+accounts.adjustment_resources, atol=1e-14)
        assert accounts.output.min() >= 1-1e-6
        assert case["diagnostics"]["accepted"]
        if case["model"]["adjustment_cost"] > 0:
            assert accounts.adjustment_resources.max() > 0
