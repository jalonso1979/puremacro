"""Independent accounts for the separately bounded compression refinement."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT/"tests/fixtures/hank_nonlinear_rank32"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def record():
    return json.loads((FIXTURE/"record.json").read_text())


def test_compression_refinement_has_its_own_sources_and_unchanged_inputs(record):
    assert record["completed"]
    for name, digest in record["source_sha256"].items():
        assert sha(ROOT/name) == digest
        assert sha(FIXTURE/"sources"/(name+".txt")) == digest
    for name, digest in record["artifact_sha256"].items():
        assert sha(FIXTURE/name) == digest
    assert record["distribution_rank"] == 32
    assert record["held_out_seed"] == 9917 and record["held_out_periods"] == 48
    assert record["training_seed"] == 1201 and record["training_periods"] == 160
    assert record["numerical_tolerance"] == 1e-8
    assert record["horizon_tolerance"] == record["branching_tolerance"] == 1e-6
    assert record["original_audit_tolerance"] == 1e-5
    held = pd.read_csv(FIXTURE/"held_out_innovations.csv").innovation.to_numpy()
    np.testing.assert_allclose(held, np.random.default_rng(9917).normal(0., 1e-4, 48), atol=1e-18, rtol=1e-14)


def test_only_accepted_periods_are_counted_and_failures_are_explicit(record):
    count = record["completed_periods"]
    assert count == len(record["period_diagnostics"])
    assert record["converged"] == (count == 48)
    if not record["converged"]:
        assert record["failure"]["kind"] in ("time_budget_exhausted", "refinement_failure", "domain_or_solver_error")
        assert count < 48
    for period in record["period_diagnostics"]:
        assert period["equation_converged"] and period["horizon_converged"] and period["branching_converged"]
        assert period["horizon_difference"]["maximum"] <= 1e-6
        assert period["branching_difference"]["maximum"] <= 1e-6
        assert period["market_residual_max_abs"] <= 1e-8
        assert period["quadrature_order_converged"] is None
        assert not period["global_accuracy_certified"]
    assert record["euler_kkt_max"] > 1e-5


def test_actual_transport_budgets_prices_and_euler_errors_are_independent(record):
    data = np.load(FIXTURE/"nonlinear.npz", allow_pickle=False)
    p, count = record["calibration"], record["completed_periods"]
    prices = pd.read_csv(FIXTURE/"prices.csv")
    a_grid, e_grid, transition = data["asset_grid"], data["e_grid"], data["transition"]
    distributions, saving, consumption, hours = data["distributions"], data["a"], data["c"], data["n"]
    assert len(distributions) == count+1 and len(prices) == count
    assert np.min(distributions) >= 0 and np.min(consumption) > 0
    np.testing.assert_allclose(distributions.sum(axis=(1, 2)), 1., atol=1e-12)
    root_euler = []
    for t in range(count):
        d, a, c, n, v = distributions[t], saving[t], consumption[t], hours[t], prices.iloc[t]
        budget = c+a-(1+v.r)*a_grid[:, None]-v.w*e_grid*n-v.transfer*e_grid
        np.testing.assert_allclose(budget, 0., atol=1e-10)
        assert abs(np.sum(d*a)-p["B"]) < 1e-8
        assert abs(np.sum(d*n*e_grid)-v.Y) < 1e-8
        assert abs(v.Y-np.sum(d*c)-v.adj) < 1e-8
        assert abs(v.tax-p["B"]*v.r) < 1e-13
        assert abs(v.r-((1+p["r"]+data["states"][t, 32])/(1+v.pi)-1)) < 1e-13
        before_income = np.zeros_like(d)
        for j in range(len(e_grid)):
            choice = np.clip(a[:, j], a_grid[0], a_grid[-1])
            lo = np.clip(np.searchsorted(a_grid, choice, side="right")-1, 0, len(a_grid)-2)
            weight = (choice-a_grid[lo])/(a_grid[lo+1]-a_grid[lo])
            np.add.at(before_income[:, j], lo, d[:, j]*(1-weight))
            np.add.at(before_income[:, j], lo+1, d[:, j]*weight)
        np.testing.assert_allclose(before_income@transition, distributions[t+1], atol=2e-14)
        grid_rhs = p["beta"]*(data["expected_marginals"][t].reshape(d.shape)@transition.T)
        rhs = np.column_stack([np.interp(a[:, j], a_grid, grid_rhs[:, j]) for j in range(len(e_grid))])
        lhs = c**(-1/p["eis"])
        error = np.where(a <= a_grid[0]+1e-10, np.maximum(rhs-lhs, 0), abs(lhs-rhs))/lhs
        root_euler.append(float(error.max()))
    assert max(root_euler) > 1e-5
    assert max(root_euler) <= record["euler_kkt_max"]+1e-12
