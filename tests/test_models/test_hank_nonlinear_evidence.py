"""Offline, source-bound checks of held-out nonlinear continuation evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT/"tests/fixtures/hank_nonlinear_continuation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def record():
    return json.loads((FIXTURE/"record.json").read_text())


def test_final_evidence_sources_artifacts_and_heldout_identity(record):
    assert record["completed"]
    for name, digest in record["source_sha256"].items():
        assert sha(ROOT/name) == digest
        assert sha(FIXTURE/"sources"/(name+".txt")) == digest
    for name, digest in record["artifact_sha256"].items():
        assert sha(FIXTURE/name) == digest
    assert sha(ROOT/"tests/fixtures/hank_recursive/record.json") == record["source_reference_sha256"]
    assert record["held_out_seed"] == 9917 and record["held_out_periods"] == 48
    assert record["held_out_std"] == 1e-4 and record["original_audit_tolerance"] == 1e-5
    assert record["training_seed"] == 1201 and record["training_periods"] == 160
    held = pd.read_csv(FIXTURE/"held_out_innovations.csv").innovation.to_numpy()
    expected = np.random.default_rng(9917).normal(0, 1e-4, 48)
    np.testing.assert_allclose(held, expected, atol=1e-18, rtol=1e-14)
    if record["blas_control"].get("status") == "verified_single_thread":
        assert record["blas_control"]["active"] == 1
        assert record["blas_control"]["scope"] == "calling thread"


def test_fixed_failure_and_corrected_equilibrium_have_distinct_accuracy_status(record):
    assert {case["name"] for case in record["cases"]} == {"full", "snapshots16"}
    for case in record["cases"]:
        assert not case["fixed"]["converged"]
        assert case["fixed"]["diagnostics"]["market_residual_max_abs"] > 1e-5
        corrected = case["nonlinear"]
        assert corrected["converged"]
        diagnostics = corrected["diagnostics"]
        assert diagnostics["completed_periods"] == diagnostics["requested_periods"] == 48
        assert diagnostics["market_residual_max_abs"] < 1e-8
        assert diagnostics["equation_converged"]
        assert diagnostics["horizon_converged"] and diagnostics["branching_converged"]
        assert diagnostics["quadrature_order_converged"] is None
        assert diagnostics["euler_kkt_max"] > 1e-5
        assert not diagnostics["household_grid_accuracy_certified"]
        assert not diagnostics["global_accuracy_certified"]
        for period in diagnostics["period_diagnostics"]:
            assert period["horizon_difference"]["maximum"] <= record["horizon_tolerance"]
            assert period["branching_difference"]["maximum"] <= record["branching_tolerance"]
            assert period["minimum_expected_marginal"] > 0


def test_independent_root_resource_assets_mass_and_observed_state_accounts(record):
    held = pd.read_csv(FIXTURE/"held_out_innovations.csv").innovation.to_numpy()
    p = record["calibration"]
    for case in record["cases"]:
        name = case["name"]
        arrays = np.load(FIXTURE/(name+"_nonlinear.npz"), allow_pickle=False)
        prices = pd.read_csv(FIXTURE/(name+"_prices.csv"))
        paths = pd.read_csv(FIXTURE/(name+"_nonlinear.csv"))
        distributions = arrays["distributions"]
        assert distributions.shape[0] == 49 and np.min(distributions) >= 0
        np.testing.assert_allclose(distributions.sum(axis=(1, 2)), 1., atol=1e-12)
        assert np.min(arrays["c"]) > 0 and np.min(arrays["expected_marginals"]) > 0
        assets = np.einsum("tij,tij->t", distributions[:-1], arrays["a"])
        consumption = np.einsum("tij,tij->t", distributions[:-1], arrays["c"])
        np.testing.assert_allclose(assets-p["B"], paths.asset_mkt, atol=2e-13)
        goods = prices.Y.to_numpy()-consumption-prices.adj.to_numpy()
        np.testing.assert_allclose(goods, paths.goods_mkt, atol=2e-13)
        assert np.max(abs(assets-p["B"])) < 1e-8 and np.max(abs(goods)) < 1e-8
        np.testing.assert_allclose(arrays["states"][1:, -2], prices.nominal.to_numpy()[:-1]-p["r"], atol=1e-14)
        shock = 0.
        for t, innovation in enumerate(held):
            shock = .7*shock+innovation
            assert arrays["states"][t, -1] == pytest.approx(shock, abs=1e-15)
    full_paths = pd.read_csv(FIXTURE/"full_nonlinear.csv")
    reduced_paths = pd.read_csv(FIXTURE/"snapshots16_nonlinear.csv")
    columns = ["pi", "w", "Y", "C"]
    np.testing.assert_allclose(full_paths[columns], reduced_paths[columns], atol=1e-7, rtol=0.)
    full_d = np.load(FIXTURE/"full_nonlinear.npz")["distributions"]
    reduced_d = np.load(FIXTURE/"snapshots16_nonlinear.npz")["distributions"]
    assert np.max(np.sum(abs(full_d-reduced_d), axis=(1, 2))) < 1e-6


def test_failed_shorter_refinements_are_preserved_with_their_original_sources():
    for directory in ("prior_depth3", "prior_horizon32"):
        root = FIXTURE/directory
        prior = json.loads((root/"record.json").read_text())
        assert prior["completed"]
        for name, digest in prior["source_sha256"].items():
            assert sha(root/"sources"/(name+".txt")) == digest
        for name, digest in prior["artifact_sha256"].items():
            assert sha(root/name) == digest
        reduced = next(case for case in prior["cases"] if case["name"] == "snapshots16")
        diagnostics = reduced["nonlinear"]["diagnostics"]
        assert not reduced["nonlinear"]["converged"]
        assert diagnostics["completed_periods"] == 28
        failed = diagnostics["failed_correction"]
        assert failed["period"] == 28 and failed["diagnostics"]["equation_converged"]
        if directory == "prior_depth3":
            assert not failed["diagnostics"]["branching_converged"]
        else:
            assert failed["diagnostics"]["branching_converged"]
            assert not failed["diagnostics"]["horizon_converged"]


def test_dense_horizon_grid_compression_and_quadrature_comparisons_stay_separate(record):
    focused = record["focused_comparisons"]
    for name in ("matched_full", "matched_snapshots16", "coarse_horizon8", "coarse_horizon32",
                 "coarse_quadrature3", "coarse_quadrature5", "coarse_krylov", "coarse_dense"):
        assert focused[name]["converged"]
        assert focused[name]["diagnostics"]["market_residual_max_abs"] < 1e-8
    assert focused["dense_difference"]["root_log_controls_max_abs"] < 1e-8
    assert focused["horizon_difference"]["root_log_controls_max_abs"] < 2e-6
    assert focused["matched_snapshots16"]["diagnostics"]["terminal_compression_l1"] > 0
    assert focused["household_grid_sensitivity"]["coarse_euler_kkt"] > 1e-5
    for name in ("quadrature_2_to_3", "quadrature_3_to_5"):
        assert all(np.isfinite(v) and v >= 0 for v in focused[name].values())
    assert focused["coarse_quadrature5"]["diagnostics"]["quadrature_order_converged"] is None
