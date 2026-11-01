"""Recompute solved-policy and accounting checks from retained HANK evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/fixtures/hank_recursive"


def record(folder=DATA):
    return json.loads((folder / "record.json").read_text())


def frame(name):
    return pd.read_csv(DATA / f"{name}.csv", index_col=0, float_precision="round_trip")


def selected(case):
    return next(p for p in case["policies"] if p["method"] == "snapshots" and p["rank"] == 16)


@pytest.mark.parametrize("prior", [False, True])
def test_sources_and_artifacts_are_bound_to_the_record(prior):
    folder = DATA / "prior_variance_only" if prior else DATA
    r = record(folder)
    assert r["completed"] and all(c["completed"] for c in r["cases"])
    for name, expected in r["source_sha256"].items():
        path = folder / "sources" / (name + ".txt") if prior else ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    for name, expected in r["artifact_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected, name
    assert all(v == "1" for v in r["blas_threads"].values())


@pytest.mark.parametrize("name", ["one_asset", "two_asset"])
@pytest.mark.parametrize("version", ["full", "snapshots_16"])
def test_frozen_policy_solves_expectations_and_reproduces_held_out_paths(name, version):
    with np.load(DATA / f"{name}_{version}.npz", allow_pickle=False) as data:
        H, G, B, Q = [data[k] for k in ("transition", "feedback", "basis", "constraints")]
        solution = np.vstack([np.eye(len(H)), G])
        np.testing.assert_allclose(data["F0"] @ solution + data["F1"] @ solution @ H,
                                   0., atol=1e-8, rtol=0)
        np.testing.assert_allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-10, rtol=0)
        np.testing.assert_allclose(Q.T @ B, 0., atol=1e-10, rtol=0)
        np.testing.assert_allclose(B.sum(axis=0), 0., atol=1e-9, rtol=0)
        assert max(abs(np.linalg.eigvals(H))) < 1
        assert abs(data["center"].sum() - 1) < 1e-12
        eps = frame("held_out_innovations").innovation.to_numpy()
        observed = frame(f"{name}_{version}_linear")
        x, simulated = np.zeros(len(H)), []
        for innovation in eps:
            x[-1] += innovation
            simulated.append(G[-len(observed.columns):] @ x)
            x = H @ x
        np.testing.assert_allclose(simulated, observed, atol=1e-13, rtol=1e-10)


@pytest.mark.parametrize("name", ["one_asset", "two_asset"])
def test_compression_error_uses_independent_shock_history(name):
    r = record()
    case = next(c for c in r["cases"] if c["name"] == name)
    for label, seed, length, std in (("training", r["training_seed"], r["training_length"], r["training_std"]),
                                     ("held_out", r["held_out_seed"], r["held_out_length"], r["held_out_std"])):
        np.testing.assert_array_equal(frame(f"{label}_innovations").innovation,
                                      np.random.default_rng(seed).normal(0, std, length))
    assert r["training_seed"] != r["held_out_seed"]
    full = frame(f"{name}_full_linear").to_numpy()
    for item in case["policies"]:
        if item["completed"]:
            reduced = frame(f"{name}_{item['method']}_{item['rank']}_linear").to_numpy()
            assert np.max(abs(full-reduced)) == pytest.approx(item["linear_max_gap"], abs=1e-14)
    assert selected(case)["linear_max_gap"] < (5e-8 if name == "one_asset" else 5e-10)


@pytest.mark.parametrize("name", ["one_asset", "two_asset"])
def test_tail_equilibrium_and_independent_dense_comparison(name):
    case = next(c for c in record()["cases"] if c["name"] == name)
    assert [item["horizon"] for item in case["terminal"]] == [4, 8, 16, 32]
    for item in case["terminal"]:
        assert item["converged"]
        assert item["diagnostics"]["market_residual_max_abs"] < 1e-8
        assert item["diagnostics"]["terminal_consistency_error"] < 1e-8
    comparison = case["dense_comparison"]
    assert comparison["dense"]["converged"] and comparison["krylov"]["converged"]
    fast, dense = frame(f"{name}_branch_krylov"), frame(f"{name}_branch_dense")
    columns = list(fast.columns[5:])  # tree metadata precede level-deviation paths
    gap = np.max(abs(fast[columns].to_numpy()-dense[columns].to_numpy()))
    assert gap == pytest.approx(comparison["path_max_gap"], abs=1e-13)
    assert gap < 3e-8


def test_two_asset_fiscal_returns_and_resource_accounts_from_retained_paths():
    case = next(c for c in record()["cases"] if c["name"] == "two_asset")
    p, ss = case["calibration"], case["steady_state"]
    for name in ("two_asset_terminal_4", "two_asset_terminal_32", "two_asset_branch_krylov"):
        rows = frame(name)
        for _, row in rows.iterrows():
            parent = None if pd.isna(row.parent) else rows.loc[row.parent]
            lagq = 1. if parent is None else 1.+parent.q
            nominal = p["r"] if parent is None else p["r"]+parent.monetary_shock+p["phi"]*parent.pi
            q, Y = 1.+row.q, p["Y"]+row.Y
            rb, ra = p["r"]+row.r, p["r_a"]+row.r_a
            assert 1+rb == pytest.approx((1+nominal)/(1+row.pi), abs=1e-12)
            assert 1+ra == pytest.approx((q+p["coupon"])/lagq, abs=1e-12)
            assert ss["Tax"]+row.Tax == pytest.approx(rb*p["B"]+p["coupon"]*p["consol_units"], abs=1e-12)
            assert ss["A"]+row.A == pytest.approx(q*p["consol_units"], abs=1e-8)
            assert abs(row.B) < 1e-8
            cost = p["mu"]/(p["mu"]-1)/(2*p["kappa"])*np.log1p(row.pi)**2*Y
            goods = Y-ss["C"]-row.C-ss["CHI"]-row.CHI-cost
            assert goods == pytest.approx(row.goods_mkt, abs=1e-12)
            assert abs(goods) < 1e-8


def test_accuracy_boundary_and_failed_early_compression_are_retained():
    r = record()
    one, two = r["cases"]
    assert not one["full"]["audit"]["converged"]
    assert not selected(one)["audit"]["converged"]
    assert one["full"]["quarter_shock_audit"]["converged"]
    assert selected(one)["quarter_shock_audit"]["converged"]
    assert two["full"]["audit"]["converged"] and selected(two)["audit"]["converged"]
    for case in r["cases"]:
        for label, item in (("full", case["full"]), ("snapshots_16", selected(case))):
            audit = item["audit"]
            rows = frame(f"{case['name']}_{label}_audit")
            errors = rows[[c for c in rows if c.endswith("_mkt") or c in ("nkpc_res", "continuation_error")]]
            assert np.max(abs(errors.to_numpy())) == pytest.approx(audit["diagnostics"]["market_residual_max_abs"], abs=1e-12)
            assert audit["diagnostics"]["euler_kkt_max"] > 0  # not an exact household-grid solution
    old = record(DATA / "prior_variance_only")
    failures = [p["error"] for c in old["cases"] for p in c["policies"] if "error" in p]
    assert any("stable roots" in error for error in failures)
    assert all(item["completed"] for item in r["grid_sensitivity"])
