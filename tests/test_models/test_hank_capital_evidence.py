"""Independent accounting and probability checks on frozen capital evidence."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"tests/fixtures/hank_capital_equilibrium"


def record():
    return json.loads((DATA/"record.json").read_text())


def arrays(name):
    with np.load(DATA/name, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def frame(name):
    with (DATA/name).open(newline="") as source:
        columns = next(csv.reader(source))[1:]
    # Python's float parser preserves the round trip of the saved 17 digits,
    # without the deprecated pandas float_precision option.
    return pd.read_csv(DATA/name, index_col=0, converters={name: float for name in columns})


def grid(n):
    return next(item for item in record()["grids"] if item["n_a"] == n)


def joint(n):
    return next(item for item in record()["joint_histories"] if item["grid"] == n)


def cells(gridpoints, policy):
    lower = np.clip(np.searchsorted(gridpoints, policy, side="right")-1, 0, len(gridpoints)-2)
    upper = lower+1
    upper_weight = (policy-gridpoints[lower])/(gridpoints[upper]-gridpoints[lower])
    return lower, upper, 1-upper_weight, upper_weight


def transport(distribution, ap, bp, household):
    """Independent bilinear deposits followed by the earnings transition."""
    a0, a1, wa0, wa1 = cells(household["a_grid"], ap)
    b0, b1, wb0, wb1 = cells(household["b_grid"], bp)
    deposited = np.zeros_like(distribution)
    for ia, wa in ((a0, wa0), (a1, wa1)):
        for ib, wb in ((b0, wb0), (b1, wb1)):
            for e in range(distribution.shape[-1]):
                np.add.at(deposited[:, :, e], (ia[:, :, e], ib[:, :, e]),
                          (distribution*wa*wb)[:, :, e])
    return deposited@household["income_transition"]


def expected_marginal(marginal, ap, bp, household):
    a0, a1, wa0, wa1 = cells(household["a_grid"], ap)
    b0, b1, wb0, wb1 = cells(household["b_grid"], bp)
    result = np.zeros_like(ap)
    transition = household["income_transition"]
    for e in range(ap.shape[-1]):
        for future_e in range(ap.shape[-1]):
            for ia, wa in ((a0, wa0), (a1, wa1)):
                for ib, wb in ((b0, wb0), (b1, wb1)):
                    result[:, :, e] += transition[e, future_e]*wa[:, :, e]*wb[:, :, e]*marginal[
                        ia[:, :, e], ib[:, :, e], future_e]
    return result


def household_errors(distribution, ap, bp, consumption, cost, expected, prices, household, p):
    inherited_a = household["a_grid"][:, None, None]
    inherited_b = household["b_grid"][None, :, None]
    productivity = household["productivity"]
    deposit = ap-(1+prices["r_a"])*inherited_a
    chi = p["chi_0"]/(1+p["chi_1"])*abs(deposit)**(1+p["chi_1"])/(inherited_a+p["a_bar"])**p["chi_1"]
    np.testing.assert_allclose(cost, chi, atol=1e-10, rtol=1e-11)
    budget = consumption+ap+bp+cost-(1+prices["r_a"])*inherited_a \
        -(1+prices["r"])*inherited_b-prices["income"]*productivity
    assert abs(budget).max() < 1e-9
    assert consumption.min() > 0
    for policy, points in ((ap, household["a_grid"]), (bp, household["b_grid"])):
        assert policy.min() >= points[0]-1e-12 and policy.max() <= points[-1]+1e-12
    uc = consumption**(-p["gamma"])
    derivative = p["chi_0"]*np.sign(deposit)*(abs(deposit)/(inherited_a+p["a_bar"]))**p["chi_1"]
    marginal_a, marginal_b = np.split(expected, 2)
    errors = []
    for marginal, lhs, policy, points in (
            (marginal_a, uc*(1+derivative), ap, household["a_grid"]),
            (marginal_b, uc, bp, household["b_grid"])):
        rhs = p["beta"]*expected_marginal(marginal.reshape(distribution.shape), ap, bp, household)
        gap = np.where(policy <= points[0]+1e-10, np.maximum(rhs-lhs, 0.),
                       np.where(policy >= points[-1]-1e-10, np.maximum(lhs-rhs, 0.), abs(lhs-rhs)))
        errors.append(gap/uc)
    error = np.maximum(*errors)
    return float(error.max()), float(np.sum(distribution*error))


def test_capital_sources_artifacts_threading_and_scope_are_bound():
    r = record()
    assert r["completed"] and r["sources_unchanged_during_run"]
    assert r["verified_blas_threading"]
    assert r["blas_control"]["status"] == "verified_single_thread"
    assert all(value == "1" for value in r["blas_threads"].values())
    assert "Concurrent" in r["concurrent_host_workload"]
    assert "no isolated speed claims" in r["timing_scope"]
    assert {"adaptive horizon acceptance", "adaptive branching acceptance",
            "global aggregate-risk accuracy", "household-grid accuracy certificate"} <= set(r["exclusions"])
    for name, expected in r["source_sha256"].items():
        path = ROOT/r.get("source_snapshots", {}).get(name, name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    for name, expected in r["artifact_sha256"].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == expected, name
    assert {p.name for p in DATA.iterdir() if p.is_file()} == set(r["artifact_sha256"]) | {"record.json"}


def test_grid_and_determinacy_failures_are_preserved():
    r = record()
    assert [g["n_a"] for g in r["grids"]] == [10, 15, 20, 30]
    failed = grid(10)
    assert failed["status"] == "failed"
    assert "household solve failed" in failed["error"] and "household warnings" in failed["error"]
    assert len(r["rejected_ranks"]) == 2
    for item in r["rejected_ranks"]:
        assert (item["distribution_rank"], item["marginal_rank"]) == (16, 64)
        assert item["status"] == "failed" and "18 stable roots for 19 states" in item["error"]
    for n in (15, 20, 30):
        g = grid(n)
        assert g["status"] == "solved"
        for name, value in r["fixed_parameters"].items():
            assert g["settings"][name] == value
        for name in ("alpha", "delta", "epsI", "mu", "government_share"):
            assert g["calibration"][name] == r["fixed_parameters"][name]
        assert g["diagnostics"]["calibration_unknown"] == "beta"
    errors = [grid(n)["household_checks"]["euler_kkt_max"] for n in (15, 20, 30)]
    assert errors[0] > errors[1] > errors[2] > 1e-4


@pytest.mark.parametrize("n", [15, 20, 30])
def test_stationary_household_budget_transport_and_capital_accounts(n):
    g = grid(n)
    p, agg, h = g["calibration"], g["aggregates"], arrays(g["arrays"])
    distribution = h["distribution"]
    assert distribution.min() >= 0 and distribution.sum() == pytest.approx(1., abs=1e-12)
    np.testing.assert_allclose(transport(distribution, h["a"], h["b"], h), distribution, atol=1e-10)
    assert np.sum(distribution*h["productivity"]) == pytest.approx(1., abs=1e-12)
    for key in ("a", "b", "c", "chi"):
        label = {"a": "A", "b": "B", "c": "C", "chi": "CHI"}[key]
        assert np.sum(distribution*h[key]) == pytest.approx(agg[label], abs=1e-10)
    maximum, mean = household_errors(distribution, h["a"], h["b"], h["c"], h["chi"],
        h["marginal_values"], dict(r=p["r"], r_a=p["r_a"], income=p["income"]), h, p)
    assert maximum == pytest.approx(g["household_checks"]["euler_kkt_max"], abs=1e-8)
    assert mean == pytest.approx(g["household_checks"]["euler_kkt_mean"], abs=1e-9)
    assert agg["K"] == pytest.approx(agg["A"], abs=1e-10)
    assert p["r_a"]+p["delta"] == pytest.approx(p["alpha"]*agg["Y"]/p["mu"]/agg["K"], abs=1e-8)
    assert agg["Y"] == pytest.approx(agg["C"]+agg["CHI"]+p["delta"]*agg["K"]+agg["G"], abs=1e-9)
    assert agg["Tax"] == pytest.approx(p["r"]*agg["B"]+agg["G"], abs=1e-12)


@pytest.mark.parametrize("n", [15, 20, 30])
def test_reduced_policy_expectations_and_expansion_from_artifacts(n):
    entry = joint(n) if n != 30 else record()["medium_policy"]
    data = arrays(entry["policy_arrays"])
    H, G, V = data["transition"], data["reduced_feedback"], data["marginal_basis"]
    solution = np.vstack([np.eye(len(H)), G])
    np.testing.assert_allclose(data["F0"]@solution+data["F1"]@solution@H, 0., atol=1e-8, rtol=0)
    np.testing.assert_allclose(V.T@V, np.eye(V.shape[1]), atol=1e-10)
    np.testing.assert_allclose(data["feedback"][:-4], V@G[:-4], atol=1e-10)
    np.testing.assert_allclose(data["feedback"][-4:], G[-4:], atol=1e-12)
    B, Q = data["basis"], data["constraints"]
    np.testing.assert_allclose(B.T@B, np.eye(B.shape[1]), atol=1e-10)
    np.testing.assert_allclose(Q.T@B, 0., atol=1e-10)
    assert max(abs(np.linalg.eigvals(H))) < 1.
    assert entry["policy"]["full_household_derivative_cache"] is False
    assert entry["policy"]["full_grid_schur_allocated"] is False
    assert data["full_residual"].shape[0] > len(data["F0"])


def check_correction(n, item, zero_risk=False):
    g, entry = grid(n), joint(n)
    p, h = g["calibration"], arrays(g["arrays"])
    policy = arrays(entry["policy_arrays"])
    data, prices, paths = arrays(item["arrays"]), frame(item["prices"]), frame(item["paths"])
    parents, probability = data["parents"], data["probability"]
    k, nm = policy["basis"].shape[1], len(policy["marginal_center"])
    expected_errors, leaf_index = [], 0
    for i, price in enumerate(prices.to_dict("records")):
        parent = parents[i]
        d = data["distribution"][i]
        state = data["aggregate_state"][i]
        inherited_q = item["initial_q"] if parent < 0 else prices.iloc[parent].q
        K, q, Y, w = p["K"]*np.exp(state[2]), price["q"], price["Y"], price["w"]
        N = (Y/(p["Z"]*K**p["alpha"]))**(1/(1-p["alpha"]))
        mc, rental = w*N/((1-p["alpha"])*Y), p["alpha"]*w*N/((1-p["alpha"])*K)
        I = K*(p["delta"]+(q-1)/p["psi"])
        Knext = (1-p["delta"])*K+I
        Phi = .5*p["psi"]*(I/K-p["delta"])**2*K
        gross_a = (rental-(I+Phi)/K+q*Knext/K)/inherited_q
        gross_b = (1+p["r"]+state[0])/(1+price["pi"])
        nominal = p["r"]+state[1]+p["phi"]*price["pi"]
        cost = p["mu"]/(p["mu"]-1)/(2*p["kappa"])*np.log1p(price["pi"])**2*Y
        tax = (gross_b-1)*p["B"]+p["G"]
        div = Y-w*N-rental*K-cost
        income = w*N+div-tax
        for name, value in dict(K=K, Knext=Knext, I=I, N=N, mc=mc, rental=rental,
                investment_goods=I+Phi, capital_adjustment=Phi, r_a=gross_a-1,
                r=gross_b-1, nominal=nominal, tax=tax, div=div, income=income,
                lag_q=inherited_q, adj=cost).items():
            assert price[name] == pytest.approx(value, abs=1e-11), (i, name)
        ap, bp, consumption, chi = (data[name][i] for name in ("a", "b", "c", "chi"))
        expected_errors.append(household_errors(d, ap, bp, consumption, chi,
            data["expected_marginals"][i], price, h, p))
        transported = transport(d, ap, bp, h)
        np.testing.assert_allclose(transported, data["post_distribution"][i], atol=1e-12)
        assert transported.min() >= -1e-14 and transported.sum() == pytest.approx(1., abs=1e-12)
        C, A, B, CHI = (float(np.sum(d*value)) for value in (consumption, ap, bp, chi))
        assert A == pytest.approx(q*Knext, abs=1e-8)
        assert B == pytest.approx(p["B"], abs=1e-8)
        assert Y-C-CHI-I-Phi-cost-p["G"] == pytest.approx(paths.iloc[i].goods_mkt, abs=1e-11)
        assert abs(Y-C-CHI-I-Phi-cost-p["G"]) < 1e-8
        labor = w*np.sum(d*h["productivity"]*consumption**(-p["gamma"]))-p["vphi"]*N**(1/p["frisch"])
        assert abs(labor) < 1e-8
        children = np.flatnonzero(parents == i)
        nxt = np.array([nominal-p["r"], record()["rho"]*state[1], np.log(Knext/p["K"])])
        if i == 0:
            np.testing.assert_allclose(nxt, data["next_aggregate_state"], atol=1e-12)
        if len(children):
            np.testing.assert_allclose(data["distribution"][children],
                np.broadcast_to(transported, data["distribution"][children].shape), atol=1e-12)
            np.testing.assert_allclose(data["aggregate_state"][children, 0], nxt[0], atol=1e-12)
            np.testing.assert_allclose(data["aggregate_state"][children, 2], nxt[2], atol=1e-12)
            expected = sum(probability[j]*data["marginal_values"][j] for j in children)
            continuation = sum(probability[j]*prices.iloc[j].Y/Y*np.log1p(prices.iloc[j].pi)
                               /(1+prices.iloc[j].r) for j in children)
        else:
            terminal = data["terminal_coordinates"][leaf_index]
            leaf_index += 1
            np.testing.assert_allclose(terminal, policy["basis"].T@(transported-policy["center"]).ravel(), atol=1e-8)
            x = np.r_[terminal, nxt]
            expected = policy["marginal_center"]+policy["feedback"][:nm]@x
            continuation = 0.
            sigma = 0. if zero_risk else record()["shock_std"]
            for innovation in (-sigma, sigma):
                branch = x.copy()
                branch[k+1] += innovation
                u = policy["feedback"][-4:]@branch
                continuation += .5*p["Y"]*np.exp(u[2])/Y*u[0]*np.exp(u[0])/(1+nominal)
        np.testing.assert_allclose(expected, data["expected_marginals"][i], atol=1e-10)
        nkpc = p["kappa"]*(mc-1/p["mu"])-np.log1p(price["pi"])+continuation
        assert nkpc == pytest.approx(paths.iloc[i].nkpc_res, abs=1e-11)
        assert abs(nkpc) < 1e-8
    assert max(v[0] for v in expected_errors) == pytest.approx(item["diagnostics"]["euler_kkt_max"], abs=1e-10)
    assert max(v[1] for v in expected_errors) == pytest.approx(item["diagnostics"]["euler_kkt_mean_max"], abs=1e-10)
    assert item["converged"] and item["diagnostics"]["equation_converged"]
    assert item["diagnostics"]["horizon_converged"] is None
    assert item["diagnostics"]["branching_converged"] is None
    assert item["diagnostics"]["global_accuracy_certified"] is False


@pytest.mark.parametrize("n", [15, 20])
def test_finite_history_household_and_market_equations_from_arrays(n):
    entry = joint(n)
    assert entry["completed"] and entry["all_finite_equations_converged"]
    check_correction(n, entry["zero_risk_zero_shock"], zero_risk=True)
    check_correction(n, entry["dense_comparison"])
    for step in entry["steps"]:
        check_correction(n, step)


@pytest.mark.parametrize("n", [15, 20])
def test_actual_history_state_quotes_and_dense_reference(n):
    entry = joint(n)
    prior = None
    for step, innovation in zip(entry["steps"], record()["history_innovations"]):
        data, prices = arrays(step["arrays"]), frame(step["prices"])
        state = np.array(step["initial_aggregate_state"])
        if prior is None:
            np.testing.assert_allclose(state, [0., innovation, 0.], atol=1e-14)
            assert step["initial_q"] == 1.
        else:
            old_data, old_prices = prior
            expected_state = old_data["next_aggregate_state"].copy()
            expected_state[1] += innovation
            np.testing.assert_allclose(state, expected_state, atol=1e-14)
            assert step["initial_q"] == old_prices.iloc[0].q
            np.testing.assert_array_equal(data["distribution"][0], old_data["post_distribution"][0])
        prior = data, prices
    fast = arrays(entry["steps"][0]["arrays"])
    dense = arrays(entry["dense_comparison"]["arrays"])
    gap = abs(dense["aggregate_coordinates"][0]-fast["aggregate_coordinates"][0]).max()
    assert gap == pytest.approx(entry["dense_comparison"]["root_coordinate_max_gap"], abs=1e-15)
    assert gap < 1e-8
    gap = np.sum(abs(dense["post_distribution"][0]-fast["post_distribution"][0]))
    assert gap == pytest.approx(entry["dense_comparison"]["next_distribution_l1_gap"], abs=1e-15)
    assert gap < 1e-8
    zero = arrays(entry["zero_risk_zero_shock"]["arrays"])
    assert abs(zero["aggregate_coordinates"]).max() < 1e-8


def test_30_grid_linear_recursion_and_actual_distribution_accounts():
    r, g = record(), grid(30)
    entry, p, agg, h = r["medium_policy"], g["calibration"], g["aggregates"], arrays(g["arrays"])
    policy = arrays(entry["policy_arrays"])
    assert entry["completed"] and entry["linear"]["converged"] and entry["audit"]["converged"]
    innovations = frame(entry["innovations"]).innovation.to_numpy()
    np.testing.assert_array_equal(innovations,
        np.random.default_rng(r["audit_seed"]).normal(0., r["audit_std"], r["audit_length"]))
    linear, linear_arrays = frame(entry["linear"]["paths"]), arrays(entry["linear"]["arrays"])
    state, k = np.zeros(len(policy["transition"])), policy["basis"].shape[1]
    for i, innovation in enumerate(innovations):
        state[k+1] += innovation
        np.testing.assert_allclose(state, linear_arrays["states"][i], atol=1e-14)
        np.testing.assert_allclose(policy["feedback"][-4:]@state, linear.iloc[i], atol=1e-13)
        state = policy["transition"]@state
    audit, data = frame(entry["audit"]["paths"]), arrays(entry["audit"]["arrays"])
    assert len(data["distributions"]) == len(audit)+1
    quote, inherited_stock, lag_nominal, lag_shock = 1., p["K"], p["r"], 0.
    for i, row in audit.iterrows():
        state = data["states"][i, k:]
        assert state[0] == pytest.approx(lag_nominal-p["r"], abs=1e-14)
        assert state[1] == pytest.approx(r["rho"]*lag_shock+innovations[i], abs=1e-14)
        np.testing.assert_allclose(data["states"][i, :k],
            policy["basis"].T@(data["distributions"][i]-policy["center"]).ravel(), atol=1e-12)
        K, q, Y = p["K"]*np.exp(state[2]), 1+row.q, p["Y"]+row.Y
        assert K == pytest.approx(inherited_stock, abs=1e-11)
        assert K == pytest.approx(agg["K"]+row.K, abs=1e-11)
        I = K*(p["delta"]+(q-1)/p["psi"])
        Knext, Phi = (1-p["delta"])*K+I, .5*p["psi"]*(I/K-p["delta"])**2*K
        N = (Y/(p["Z"]*K**p["alpha"]))**(1/(1-p["alpha"]))
        rental = p["alpha"]*(p["w"]+row.w)*N/((1-p["alpha"])*K)
        assert 1+p["r_a"]+row.r_a == pytest.approx((rental-(I+Phi)/K+q*Knext/K)/quote, abs=1e-11)
        rb = (1+p["r"]+state[0])/(1+row.pi)-1
        assert p["r"]+row.r == pytest.approx(rb, abs=1e-12)
        assert agg["Tax"]+row.Tax == pytest.approx(rb*p["B"]+p["G"], abs=1e-12)
        adj = p["mu"]/(p["mu"]-1)/(2*p["kappa"])*np.log1p(row.pi)**2*Y
        goods = Y-agg["C"]-row.C-agg["CHI"]-row.CHI-I-Phi-adj-p["G"]
        assert goods == pytest.approx(row.goods_mkt, abs=1e-11)
        next_d = data["distributions"][i+1]
        assert next_d.min() >= 0 and next_d.sum() == pytest.approx(1., abs=1e-12)
        assert np.sum(next_d*h["a_grid"][:, None, None]) == pytest.approx(agg["A"]+row.A, abs=1e-10)
        assert np.sum(next_d*h["b_grid"][None, :, None]) == pytest.approx(agg["B"]+row.B, abs=1e-10)
        assert agg["A"]+row.A-q*Knext == pytest.approx(row.illiquid_mkt, abs=1e-10)
        assert agg["B"]+row.B-p["B"] == pytest.approx(row.liquid_mkt, abs=1e-12)
        quote, inherited_stock = q, Knext
        lag_nominal, lag_shock = p["r"]+state[1]+p["phi"]*row.pi, state[1]
    assert entry["audit"]["diagnostics"]["euler_kkt_max"] > 1e-4
    assert entry["audit"]["diagnostics"]["full_marginal_relative_error"] > 0
