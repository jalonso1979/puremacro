"""Bind measured same-grid solver comparisons to their sources and path files."""
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "hank_optimized"


def test_optimized_hank_evidence_hashes_and_scope():
    record = json.loads((FIXTURE / "record.json").read_text())
    assert record["schema_version"] == 1 and record["completed"]
    assert "not discretization" in record["scope"]
    assert record["repeats"] == 3
    assert set(record["blas_threads"].values()) == {"1"}
    assert record["blas_control"]["status"] == "verified_single_thread"
    assert record["blas_control"]["active"] == 1
    assert record["numpy_linear_algebra"]["blas"]["name"].lower() == "accelerate"
    assert record["scipy_linear_algebra"]["blas"]["name"].lower() == "accelerate"
    assert {c["name"] for c in record["cases"]} == {"medium", "fine", "long"}
    for name, digest in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for case in record["cases"]:
        for name, digest in case["artifact_sha256"].items():
            assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest, name
    # Preserve the earlier observations without presenting them as controlled
    # threading evidence or requiring historical sources to equal current code.
    prior = FIXTURE / "prior_uncontrolled"
    previous = json.loads((prior / "record.json").read_text())
    assert previous["completed"] and "blas_control" not in previous
    runner = "tools/reference_validation/benchmark_hank_equilibrium.py"
    assert hashlib.sha256((prior / "runner-used.py.txt").read_bytes()).hexdigest() == previous["source_sha256"][runner]
    for case in previous["cases"]:
        for name, digest in case["artifact_sha256"].items():
            assert hashlib.sha256((prior / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("name", ["medium", "fine", "long"])
def test_optimized_hank_equilibria_and_measured_timings(name):
    record = json.loads((FIXTURE / "record.json").read_text())
    case = next(c for c in record["cases"] if c["name"] == name)
    assert case["completed"] and case["all_converged"] and case["steady_state_converged"]
    assert len(case["runs"]) == record["repeats"]
    frames = {method: pd.read_csv(FIXTURE / f"{name}_{method}.csv", index_col=0)
              for method in ("reference", "optimized", "warm", "nearby_control")}
    accounts = ["goods_mkt", "asset_mkt", "labor_mkt", "nkpc_res"]
    columns = ["pi", "w", "Y", "C", "A", "NE", "r", "Div", "Tax", *accounts]
    for method, frame in frames.items():
        assert len(frame) == case["nodes"]
        assert np.isfinite(frame[columns].to_numpy()).all()
        assert np.max(np.abs(frame[accounts].to_numpy())) <= record["tolerance"]
        np.testing.assert_allclose(frame.groupby("period").probability.sum(), 1., atol=2e-14)
        p = case["calibration"]
        # Independent parent-rate timing and government debt-service accounts.
        for node, row in frame.iterrows():
            lag = p["r"] if pd.isna(row.parent) else (
                p["r"] + frame.loc[row.parent, "monetary_shock"] + p["phi"] * frame.loc[row.parent, "pi"])
            assert 1 + p["r"] + row.r == pytest.approx((1 + lag) / (1 + row.pi), abs=3e-14)
            assert row.Tax == pytest.approx(p["B"] * row.r, abs=3e-14)
    difference = float(np.max(np.abs(frames["reference"][columns].to_numpy() - frames["optimized"][columns].to_numpy())))
    warm_difference = float(np.max(np.abs(frames["warm"][columns].to_numpy() - frames["nearby_control"][columns].to_numpy())))
    assert difference == pytest.approx(case["runs"][0]["path_difference_max_abs"], abs=1e-14)
    assert warm_difference == pytest.approx(case["runs"][0]["warm_path_difference_max_abs"], abs=1e-14)
    for repeat, run in enumerate(case["runs"]):
        assert run["path_difference_max_abs"] < 5e-8
        assert run["warm_path_difference_max_abs"] < 5e-8
        assert max(run["household_difference_max_abs"].values()) < 5e-6
        assert run["order"] == (["reference", "optimized"] if repeat % 2 == 0 else ["optimized", "reference"])
        for method in frames:
            item = run[method]
            assert item["converged"] and item["diagnostics"]["market_residual_max_abs"] <= record["tolerance"]
            assert np.isfinite(item["seconds"]) and item["seconds"] > 0
        assert not run["optimized"]["diagnostics"]["dense_jacobian_assembled"]
        assert run["warm"]["diagnostics"]["preconditioner_reused"]
    for method in frames:
        assert case["median_seconds"][method] == median(r[method]["seconds"] for r in case["runs"])
    # Check the arithmetic, without asserting any hardware-dependent speed gate.
    assert case["cold_speed_ratio"] == pytest.approx(case["median_seconds"]["reference"] / case["median_seconds"]["optimized"])
