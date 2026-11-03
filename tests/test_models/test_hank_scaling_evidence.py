"""Recompute reduced-policy equations and accuracy from retained scaling runs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import solve_discrete_lyapunov

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"tests/fixtures/hank_scaling"


def record():
    return json.loads((DATA/"record.json").read_text())


def case(name):
    return next(c for c in record()["cases"] if c["name"] == name)


def frame(name, kind="linear"):
    path = DATA/(name+"_"+kind+".csv")
    with path.open(newline="") as source:
        columns = next(csv.reader(source))[1:]
    return pd.read_csv(path, index_col=0, converters={name: float for name in columns})


def test_scaling_sources_artifacts_and_failed_configurations_are_preserved():
    r = record()
    assert r["completed"] and len(r["cases"]) == 12
    for name, expected in r["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    for name, expected in r["artifact_sha256"].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == expected
    assert all(value == "1" for value in r["threads"].values())
    for c in r["cases"]:
        assert c["completed"] and c["blas_control"]["status"] == "verified_single_thread"
        assert c["peak_resident_bytes"] > 0
    assert "25 stable roots for 26 states" in case("one_default_low_rank")["error"]["message"]
    assert case("two_fine")["error"]["message"] == "non-real Schur policy"
    assert not case("one_default_low_rank")["solved"] and not case("two_fine")["solved"]
    assert "regression" in r["timing_context"]


@pytest.mark.parametrize("name", ["one_small_reduced", "one_default", "two_small_reduced", "two_fine_coarse", "two_fine_more_controls"])
def test_reduced_solutions_expand_and_solve_the_retained_equations(name):
    c = case(name)
    with np.load(DATA/(name+"_policy.npz"), allow_pickle=False) as d:
        H, G, V = d["transition"], d["reduced_feedback"], d["marginal_basis"]
        nu, ns = 3 if c["settings"]["model"] == "one" else 4, len(H)
        S = np.vstack([np.eye(ns), G])
        np.testing.assert_allclose(d["F0"]@S+d["F1"]@S@H, 0., atol=1e-8, rtol=0)
        np.testing.assert_allclose(d["feedback"][:-nu], V@G[:-nu], atol=1e-10)
        np.testing.assert_allclose(V.T@V, np.eye(V.shape[1]), atol=1e-10)
        np.testing.assert_allclose(d["basis"].T@d["basis"], np.eye(d["basis"].shape[1]), atol=1e-10)
        np.testing.assert_allclose(d["constraints"].T@d["basis"], 0., atol=1e-10)
        assert np.max(abs(np.linalg.eigvals(H))) < 1.
        Q = np.zeros_like(H)
        Q[-1, -1] = 1e-8
        covariance = solve_discrete_lyapunov(H, Q)
        omitted = d["full_residual"][ns:ns+len(d["marginal_center"])]
        variance = np.maximum(np.einsum("ij,ji->i", omitted@covariance, omitted.T), 0.)
        normalized = np.max(np.sqrt(variance)/(1+abs(d["marginal_center"])))
        assert normalized == pytest.approx(c["policy"]["full_marginal_stationary_relative_rms_max"], abs=1e-13, rel=1e-9)
        eps = np.random.default_rng(9917).normal(0, 1e-4, 48)
        x, paths = np.zeros(ns), []
        for innovation in eps:
            x[-1] += innovation
            paths.append(d["feedback"][-nu:]@x)
            x = H@x
        np.testing.assert_allclose(paths, frame(name).to_numpy(), atol=1e-13, rtol=1e-10)
        assert c["policy"]["reduced_jacobian_bytes"] == d["F0"].nbytes+d["F1"].nbytes
        assert not c["policy"]["full_household_derivative_cache"]
        assert not c["policy"]["full_grid_schur_allocated"]


@pytest.mark.parametrize("left,right,tolerance", [
    ("one_small_full_controls", "one_small_reduced", 1e-8),
    ("two_small_full_controls", "two_small_reduced", 1e-9),
    ("two_fine_coarse", "two_fine_more_controls", 1e-8),
])
def test_same_grid_control_reduction_matches_independent_or_finer_controls(left, right, tolerance):
    gap = np.max(abs(frame(left).to_numpy()-frame(right).to_numpy()))
    assert gap < tolerance
    assert case(left)["settings"]["specification"] == case(right)["settings"]["specification"]
    assert case(left)["settings"]["distribution_rank"] == case(right)["settings"]["distribution_rank"]


def test_default_grid_has_independent_sequence_space_reference():
    c = case("one_default")
    assert c["policy"]["system_dimension"] == 229
    assert c["policy"]["full_marginal_dimension"] == 3500
    assert c["sequence_reference"]["converged"]
    left, right = frame("one_default", "sequence_reference"), frame("one_default", "recursive_impulse")
    gap = np.max(abs(left.iloc[:30].to_numpy()-right.iloc[:30].to_numpy()))
    assert gap == pytest.approx(c["sequence_reference"]["max_abs_gap"], abs=1e-13)
    assert gap < 1e-7
    assert case("two_fine_coarse")["policy"]["system_dimension"] == 78
    assert case("two_fine_coarse")["policy"]["full_marginal_dimension"] == 6400


def test_nonlinear_accuracy_failures_are_not_inferred_away_by_small_linear_errors():
    for name in ("one_small_reduced", "one_default", "two_fine_coarse", "two_fine_more_controls"):
        full, quarter = case(name)["audits"]
        assert not full["converged"]
        assert full["diagnostics"]["tolerance"] == quarter["diagnostics"]["tolerance"] == 1e-5
        assert full["diagnostics"]["household_budget_error"] < 1e-8
        assert full["diagnostics"]["mass_error"] < 1e-10
        assert full["diagnostics"]["policy_grid_exit_mass"] == 0.
        assert full["diagnostics"]["euler_kkt_max"] > 0.
        if name == "one_default":
            assert not quarter["converged"]
            assert quarter["diagnostics"]["market_residual_max_abs"] > 1e-5
        else:
            assert quarter["converged"]
