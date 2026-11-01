"""Distinguish adaptive failure, bounded work and a validated local domain."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"tests/fixtures/hank_capital_adaptive"


def read(name):
    return json.loads((DATA/name).read_text())


def test_adaptive_records_and_exact_generators_are_bound():
    manifest = read("manifest.json")
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == digest


def test_higher_risk_failures_are_not_replaced_by_the_smaller_risk_success():
    original = read("original.json")
    assert original["shock_std"] == 1e-4 and not original["converged"]
    d = original["diagnostics"]
    assert d["equation_converged"] and d["horizon_converged"]
    assert not d["branching_converged"]
    assert d["market_residual_max_abs"] < 1e-8
    assert d["branching_difference"]["maximum"] > 1e-6
    extended = read("refined_highrisk_bounded.json")
    assert extended["shock_std"] == 1e-4
    assert extended["status"] == "budget_exhausted"
    assert not extended["converged"]
    assert extended["pending_corner"] == {"horizon": 16, "branch_depth": 3}
    assert all(c["diagnostics"]["equation_converged"] for c in extended["corners"])


def test_declared_small_risk_adaptive_history_keeps_the_original_thresholds():
    r = read("refined_lowrisk.json")
    assert r["shock_std"] == 1e-5 and r["converged"]
    d = r["diagnostics"]
    assert d["completed_periods"] == d["requested_periods"] == 3
    assert d["equation_converged"] and d["horizon_converged"] and d["branching_converged"]
    assert d["market_residual_max_abs"] < 1e-8
    assert d["euler_kkt_max"] > 1e-3
    assert not d["global_accuracy_certified"] and not d["household_grid_accuracy_certified"]
    for period in d["period_diagnostics"]:
        assert period["horizon_tolerance"] == period["branching_tolerance"] == 1e-6
        assert period["horizon_difference"]["maximum"] < 1e-6
        assert period["branching_difference"]["maximum"] < 1e-6
    with np.load(DATA/"refined_lowrisk.npz", allow_pickle=False) as z:
        prices = {name: z["prices"][:, i] for i, name in enumerate(z["price_columns"])}
        assert len(z["states"]) == 3 and len(z["distributions"]) == 4
        np.testing.assert_allclose(z["distributions"].sum(axis=(1, 2, 3)), 1., atol=1e-12)
        assert z["distributions"].min() >= 0.
        np.testing.assert_allclose(prices["lag_q"], np.r_[1., prices["q"][:-1]], atol=1e-14)
        np.testing.assert_allclose(prices["K"][1:], prices["Knext"][:-1], atol=1e-12)
        payoff = prices["rental"]*prices["K"]-prices["investment_goods"]+prices["q"]*prices["Knext"]
        np.testing.assert_allclose((1+prices["r_a"])*prices["lag_q"]*prices["K"], payoff, atol=1e-11)
        assert r["independent_accounts_passed"]
