"""Exact CES block Newton on consistent accounting: oracles, parity, certificates, homotopy.

The exact-inverse oracle (complex-step Jacobian of the module's own residual and
central differences of ``_accounting.evaluate`` at flat technology) gates the
solver tests, following the structure of the IO oracle suite
(``headlinePaper/rebuild/tests/test_ces_newton.py``). Exact parity with the IO
engine is not expected (one factor, DEK closure); the shared CES kernel is
compared in a subprocess when the research volume is mounted.
"""
from dataclasses import replace
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest
from scipy.optimize import root

from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, compute_hicksian_welfare
from puremacro.trade._accounting import evaluate, postprocess
from puremacro.trade._oecd_icio import condense_final_demand
from puremacro.trade.data import generate_synthetic_mrio, package_mrio_to_calibration_result
import puremacro.trade.ces_newton as ces_newton
from puremacro.trade.ces_newton import (CESBlockJacobian, CESBlockNewtonResult, CESNewtonError,
                                        NestedCESTechnology, certify_ces_equilibrium,
                                        continue_tariff_homotopy, solve_ces_block_newton,
                                        _build_model, _ces_index, _certificate)
from puremacro.trade.equilibrium import pack_equilibrium_vector
from puremacro.trade.solver import build_initial_guess
from tools.reference_validation.validate_oecd import load_fixture

GOLDEN = pathlib.Path(__file__).resolve().parent / "fixtures" / "trade" / "ces_newton" / "golden_oecd3x3.json"
# The IO research workspace; parity tests skip when it is absent. Override with PUREMACRO_IO_ROOT.
IO_ROOT = pathlib.Path(os.environ.get("PUREMACRO_IO_ROOT", "/Volumes/BIGDATA/Research/IO"))
TECHS = [(0, 0, 0, 1), (.1, .3, 1.5, 1), (1, 1, 1, 1), (.5, 1, 4, .7), (.1, .1, .5, .7)]
FLAT = [(0, .5, .5, 1), (0, 2, 2, 1), (0, 1, 1, 1)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def hand_table(absent_x=True):
    """Nonnegative nc=3, ns=2, nfd=3 table; country C has no purchases abroad when absent_x."""
    rng = np.random.default_rng(7)
    nc, ns, nfd = 3, 2, 3
    n = nc * ns
    Z = rng.integers(2, 12, size=(n, n)).astype(float)
    for c in range(nc):
        Z[c * ns:(c + 1) * ns, c * ns:(c + 1) * ns] *= 3
    F = rng.integers(5, 30, size=(n, nfd * nc)).astype(float)
    F[:, 1::nfd] *= 0.5
    F[:, 2::nfd] *= 0.2
    if absent_x:
        F[:, 2 + nfd * 2] = 0.0
    y = Z.sum(1) + F.sum(1)
    tax = 0.03 * y
    va = y - Z.sum(0) - tax
    assert np.all(va > 0)
    final_tax = 0.05 * F.sum(0)
    if absent_x:
        final_tax[2 + nfd * 2] = 0.0
    return np.vstack([np.hstack([Z, F]), np.r_[tax, final_tax], np.r_[2 * va / 3, np.zeros(nfd * nc)],
                      np.r_[va / 3, np.zeros(nfd * nc)]])


def hand_tariffs(calib):
    """Bilateral counter-tariffs: A taxes B 25%, B taxes A 40%, C free; category-specific final duties."""
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    n = nc * ns
    ta = np.ones((n, ns, nc)); tf = np.ones((n, nfd, nc))
    ta[2:4, :, 0] = 1.25; ta[0:2, :, 1] = 1.40
    tf[2:4, :, 0] = [[1.10, 1.30, 1.05]]; tf[0:2, :, 1] = [[1.20, 1.15, 1.00]]
    tf[4:6, 0, 0] = 1.08
    return ta, tf


def synthetic_calibration(nc, ns, seed=3, final_tax_rate=0.02, margin=0.05):
    """Seeded nonnegative RawIOData through package_mrio_to_calibration_result.

    The generator's six OECD final uses are condensed to C/I/X; investment of
    deficit countries is topped up with domestic purchases so that every
    expenditure share is positive (consistent accounting rejects negative
    signed investment aggregates); a 2% final tax is added on C.
    """
    raw = generate_synthetic_mrio("oecd", custom_c=nc, custom_s=ns, seed=seed)
    raw = condense_final_demand(replace(raw, taxes_less_subsidies_fd=np.zeros(raw.C * raw.K_F)))
    C, S, K = raw.C, raw.S, raw.K_F
    M = C * S
    Z = raw.Z.copy(); F = raw.F.reshape(M, C, K).copy()
    country = np.repeat(np.arange(C), S)
    flows = Z.reshape(C, S, C, S).sum((1, 3)) + F.reshape(C, S, C, K).sum((1, 3))
    np.fill_diagonal(flows, 0.)
    B = flows.sum(1) - flows.sum(0)
    for c in range(C):
        need = margin * abs(B[c]) - (F[:, c, 1].sum() + B[c])
        if need > 0:
            own = country == c
            F[own, c, 1] += need / own.sum()
    Y = Z.sum(1) + F.reshape(M, -1).sum(1)
    VA = Y - Z.sum(0) - raw.TLS
    tfd = np.zeros((C, K)); tfd[:, 0] = final_tax_rate * F[:, :, 0].sum(0)
    raw = replace(raw, final_demand_matrix=F.reshape(M, -1), gross_output=Y, value_added=VA,
                  taxes_less_subsidies_fd=tfd.ravel())
    return package_mrio_to_calibration_result(raw, regularize=False)


def random_schedule(calib, seed=5):
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    n = nc * ns
    ta = np.ones((n, ns, nc)); tf = np.ones((n, nfd, nc))
    rng = np.random.default_rng(seed)
    for c in range(nc):
        rate = rng.uniform(0, .3)
        ta[:, :, c] = 1 + rate; tf[:, :, c] = 1 + rate * rng.uniform(.5, 1.5)
        ta[c * ns:(c + 1) * ns, :, c] = 1; tf[c * ns:(c + 1) * ns, :, c] = 1
    return ta, tf


@pytest.fixture(scope="module")
def hand():
    data = hand_table(True)
    calib = calibrate_trade_model(data, ns=2, nc=3, nfd=3, country_codes=["A", "B", "C"])
    return dict(data=data, calib=calib, tariffs=hand_tariffs(calib))


@pytest.fixture(scope="module")
def hand_active():
    data = hand_table(False)
    calib = calibrate_trade_model(data, ns=2, nc=3, nfd=3, country_codes=["A", "B", "C"])
    return dict(data=data, calib=calib, tariffs=hand_tariffs(calib))


@pytest.fixture(scope="module")
def oecd3():
    calib = package_mrio_to_calibration_result(condense_final_demand(load_fixture()))
    rates = np.array([.1, 0., 0.])
    return dict(calib=calib, tariffs=(rates, rates))


@pytest.fixture(scope="module")
def rand20():
    calib = synthetic_calibration(20, 11)
    return dict(calib=calib, tariffs=random_schedule(calib))


@pytest.fixture
def case(request, hand, hand_active, oecd3, rand20):
    return {"hand": hand, "hand_active": hand_active, "oecd3": oecd3, "rand20": rand20}[request.param]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def complex_step_jacobian(model, x, h=1e-25):
    dim = len(x)
    jac = np.empty((dim, dim))
    for j in range(dim):
        z = np.asarray(x, dtype=complex).copy(); z[j] += 1j * h
        jac[:, j] = model.residual(z).imag / h
    return jac


def fd_jacobian_accounting(calib, x, ta, tf, sigma, rel=1e-6):
    """Central differences of ``_accounting.evaluate``; steps relative to level unknowns."""
    dim = len(x)
    jac = np.empty((dim, dim))
    for j in range(dim):
        h = rel * max(1.0, abs(x[j]))
        e = np.zeros(dim); e[j] = h
        up = evaluate(x + e, calib, ta, tf, None, None, sigma=sigma)["residuals"]
        dn = evaluate(x - e, calib, ta, tf, None, None, sigma=sigma)["residuals"]
        jac[:, j] = (up - dn) / (2 * h)
    return jac


def interior_point(model, rng):
    """A random admissible point that is not an equilibrium."""
    n, nc = model.n, model.nc
    x = build_initial_guess(model.calib)
    x[:n] += rng.uniform(-.06, .08, n); x[n:2 * n] += rng.uniform(-.05, .05, n)
    x[2 * n:2 * n + 2 * nc] += rng.uniform(-.04, .04, 2 * nc)
    x[2 * n + 2 * nc:2 * n + 3 * nc] *= rng.uniform(.95, 1.05, nc)
    x[2 * n + 3 * nc:] += rng.uniform(-.01, .01, nc - 1) * model.gdp0[:nc - 1]
    b = model.blocks(x)
    assert model.admissible(b)
    assert np.max(np.abs(b["residuals"] / model.scale)) > 1e-3
    return x


def tech_of(values):
    return NestedCESTechnology(*values)


# ---------------------------------------------------------------------------
# 1-2. Exact-inverse oracle (gate)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["hand", "hand_active", "oecd3"], indirect=True)
@pytest.mark.parametrize("values", TECHS + FLAT)
def test_exact_inverse_oracle_complex_step(case, values):
    """J_cs @ solve(r) = r and apply(solve(r)) = r to 3e-10 in scaled units at a non-equilibrium point."""
    calib = case["calib"]; ta, tf = case["tariffs"]
    tech = tech_of(values)
    model = _build_model(calib, ta, tf, None, None, tech)
    rng = np.random.default_rng(8213)
    x = interior_point(model, rng)
    J = complex_step_jacobian(model, x)
    for block_size in (2, 16):
        jac = CESBlockJacobian(calib, x, tau=ta, tau_fd=tf, technology=tech, block_size=block_size)
        assert jac.macro_size == 3 * calib.nc + 1
        for _ in range(2):
            rs = rng.normal(size=len(x))
            r = rs * model.scale
            v = jac.solve(r)
            np.testing.assert_allclose((J @ v) / model.scale, rs, rtol=0, atol=3e-10)
            np.testing.assert_allclose(jac.apply(v) / model.scale, rs, rtol=0, atol=3e-10)
            forward = J @ rs
            np.testing.assert_allclose(jac.apply(rs) / model.scale, forward / model.scale, rtol=1e-10,
                                       atol=1e-10 * max(1, np.max(np.abs(forward / model.scale))))


@pytest.mark.parametrize("values", [(0, 0, 0, 1), (.5, 1, 4, .7)])
def test_exact_inverse_oracle_random_20x11(rand20, values):
    calib = rand20["calib"]; ta, tf = rand20["tariffs"]
    tech = tech_of(values)
    model = _build_model(calib, ta, tf, None, None, tech)
    rng = np.random.default_rng(11)
    x = interior_point(model, rng)
    J = complex_step_jacobian(model, x)
    jac = CESBlockJacobian(calib, x, tau=ta, tau_fd=tf, technology=tech)
    rs = rng.normal(size=len(x)); r = rs * model.scale
    v = jac.solve(r)
    np.testing.assert_allclose((J @ v) / model.scale, rs, rtol=0, atol=3e-10)
    np.testing.assert_allclose(jac.apply(v) / model.scale, rs, rtol=0, atol=3e-10)


@pytest.mark.parametrize("case", ["hand", "hand_active", "oecd3"], indirect=True)
@pytest.mark.parametrize("sigma", [0., .5, 2.])
def test_flat_jacobian_matches_accounting_finite_differences(case, sigma):
    """Central differences of _accounting.evaluate invert the operator to 1e-6 at flat technology."""
    calib = case["calib"]; ta, tf = case["tariffs"]
    tech = NestedCESTechnology(0, sigma, sigma, 1)
    assert tech.is_flat
    model = _build_model(calib, ta, tf, None, None, tech)
    rng = np.random.default_rng(3)
    x = interior_point(model, rng)
    Jfd = fd_jacobian_accounting(calib, x, ta, tf, sigma)
    jac = CESBlockJacobian(calib, x, tau=ta, tau_fd=tf, technology=tech)
    rs = rng.normal(size=len(x)); r = rs * model.scale
    np.testing.assert_allclose((Jfd @ jac.solve(r)) / model.scale, rs, rtol=0, atol=1e-6)
    J = complex_step_jacobian(model, x)
    scale = np.max(np.abs(J / model.scale[:, None]))
    np.testing.assert_allclose(Jfd / model.scale[:, None], J / model.scale[:, None], rtol=0, atol=1e-6 * scale)


# ---------------------------------------------------------------------------
# 3-4. Row-by-row parity, Leontief limit, flat equilibria and welfare
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["hand", "hand_active", "oecd3", "rand20"], indirect=True)
@pytest.mark.parametrize("sigma", [0., .5, 1., 2.])
def test_flat_residual_rows_equal_accounting_evaluate(case, sigma):
    calib = case["calib"]; ta, tf = case["tariffs"]
    model = _build_model(calib, ta, tf, None, None, NestedCESTechnology(0, sigma, sigma, 1))
    x = interior_point(model, np.random.default_rng(2))
    theirs = evaluate(x, calib, ta, tf, None, None, sigma=sigma)
    mine = model.blocks(x)
    np.testing.assert_allclose(mine["residuals"], theirs["residuals"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mine["physical_residuals"], theirs["physical_residuals"], rtol=1e-12, atol=1e-12)
    for key in ("pp", "Z", "F", "labor", "capital", "tariffs", "government", "trade", "P", "c"):
        np.testing.assert_allclose(mine[key], theirs[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", ["hand", "oecd3", "rand20"], indirect=True)
def test_leontief_limit_equals_consistent_newton(case):
    """Technology (0,0,0,1) reproduces solve_trade_equilibrium(accounting="consistent").

    puremacro's finite-difference Newton reaches 1e-12 relative only on the small
    tables; on the 20x11 table it confirms the block-Newton root from a warm start.
    """
    calib = case["calib"]; ta, tf = case["tariffs"]
    mine = solve_ces_block_newton(calib, ta, tf, technology=NestedCESTechnology(0, 0, 0, 1))
    scale = np.max(_build_model(calib, ta, tf, None, None, NestedCESTechnology()).scale)
    cold = calib.nc <= 3
    theirs = solve_trade_equilibrium(calib, tau=ta, tau_fd=tf, accounting="consistent", method="newton",
                                     tol=max(1e-12 * scale, 1e-12), max_iter=100,
                                     x0=None if cold else mine.equilibrium.x_sol)
    assert theirs.converged
    np.testing.assert_allclose(mine.equilibrium.x_sol, theirs.x_sol, rtol=1e-10, atol=1e-10)
    for name in ("p_sol", "w_sol", "r_sol", "y_sol", "T_sol", "tariffs", "gdp"):
        np.testing.assert_allclose(getattr(mine.equilibrium, name), getattr(theirs, name), rtol=1e-10, atol=1e-10)
    if not cold:
        assert theirs.iterations == 0


@pytest.mark.parametrize("case", ["hand", "hand_active"], indirect=True)
@pytest.mark.parametrize("sigma", [.5, 2.])
def test_flat_ces_equilibrium_parity_postprocessing_and_welfare(case, sigma):
    """Flat CES equilibria agree with the finite-difference consistent Newton to 1e-10 (card item 4).

    The reference runs at the same absolute tolerance as the Leontief test
    (1e-12 times the largest row scale) and reaches the block-Newton root to
    about 3e-12 on these tables.
    """
    calib = case["calib"]; ta, tf = case["tariffs"]
    tech = NestedCESTechnology(0, sigma, sigma, 1)
    mine = solve_ces_block_newton(calib, ta, tf, technology=tech, tol=1e-13)
    scale = np.max(_build_model(calib, ta, tf, None, None, tech).scale)
    theirs = solve_trade_equilibrium(calib, tau=ta, tau_fd=tf, accounting="consistent", sigma=sigma,
                                     method="newton", tol=max(1e-12 * scale, 1e-12), max_iter=100)
    assert theirs.converged
    np.testing.assert_allclose(mine.equilibrium.x_sol, theirs.x_sol, rtol=1e-10, atol=1e-10)
    # Postprocessed flows equal _accounting.postprocess at the same state.
    pp = postprocess(evaluate(mine.equilibrium.x_sol, calib, ta, tf, None, None, sigma=sigma), calib, None)
    for key in ("intermediate_flows", "final_demand_flows", "gdp", "cpi", "tariffs", "exports", "imports", "terms_of_trade", "P_fd", "c_fd"):
        np.testing.assert_allclose(getattr(mine.equilibrium, key if key not in ("P_fd",) else "Pfd_final"), pp[key], rtol=1e-10, atol=1e-10)
    assert max(mine.equilibrium.metadata["account_residuals"].values()) <= 1e-10
    assert mine.equilibrium.metadata["accounting"] == "consistent"
    assert mine.equilibrium.metadata["effective_method"] == "ces_block_newton"
    assert mine.equilibrium.metadata["sigma"] == sigma
    assert mine.equilibrium.metadata["hicksian_welfare_supported"]
    base = solve_ces_block_newton(calib, technology=tech, tol=1e-13)
    w = compute_hicksian_welfare(calib, mine.equilibrium, base_result=base.equilibrium, target_country="A",
                                 consumption_categories=(0, 2))
    base_ref = solve_trade_equilibrium(calib, accounting="consistent", sigma=sigma, tol=1e-12, max_iter=100)
    ref = compute_hicksian_welfare(calib, theirs, base_result=base_ref, target_country="A", consumption_categories=(0, 2))
    assert w.ev == pytest.approx(ref.ev, abs=1e-7, rel=1e-7)
    assert w.ev_pct_consumption == pytest.approx(ref.ev_pct_consumption, abs=1e-7)


def test_nested_result_is_rejected_by_hicksian_welfare_audit(hand):
    calib = hand["calib"]; ta, tf = hand["tariffs"]
    tech = NestedCESTechnology(.3, .5, 1.5, .8)
    counter = solve_ces_block_newton(calib, ta, tf, technology=tech)
    base = solve_ces_block_newton(calib, technology=tech)
    assert counter.equilibrium.metadata["hicksian_welfare_supported"] is False
    with pytest.raises(ValueError, match="audit"):
        compute_hicksian_welfare(calib, counter.equilibrium, base_result=base.equilibrium, target_country="A",
                                 consumption_categories=(0, 2))


# ---------------------------------------------------------------------------
# 5-6. Benchmark reproduction, Shephard and Euler
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["hand", "hand_active", "oecd3", "rand20"], indirect=True)
@pytest.mark.parametrize("values", TECHS + FLAT)
def test_benchmark_reproduction(case, values):
    calib = case["calib"]
    model = _build_model(calib, None, None, None, None, tech_of(values))
    b = model.blocks(build_initial_guess(calib))
    assert np.max(np.abs(b["residuals"] / model.scale)) <= 1e-12
    np.testing.assert_allclose(b["tech"]["a"], model.a0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b["tech"]["bL"], model.b0L, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b["tech"]["bK"], model.b0K, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b["tech"]["c"], 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("values", TECHS)
def test_shephard_euler_identities(hand, values):
    calib = hand["calib"]; ta, tf = hand["tariffs"]
    model = _build_model(calib, ta, tf, None, None, tech_of(values))
    x = interior_point(model, np.random.default_rng(9))
    p, y, r, w, T, XN, B = model.unpack(x)
    t = model.technology_at(p, r, w)
    net = (1 - model.t) * t["c"]
    outlay = np.sum(t["a"] * t["rr"], axis=0) + t["bL"] * w[model.country] + t["bK"] * r[model.country]
    np.testing.assert_allclose(outlay, net, rtol=1e-12, atol=1e-12)
    shares = t["a"] * t["rr"] / net[None]
    total = shares.sum(0) + (t["bL"] * w[model.country] + t["bK"] * r[model.country]) / net
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-14)
    # d((1-t_j) c_j)/d p_i = a_ij tau_ij by central differences
    h = 1e-6
    for i in (0, 3, 5):
        up = p.copy(); dn = p.copy(); up[i] *= 1 + h; dn[i] *= 1 - h
        d = ((1 - model.t) * (model.technology_at(up, r, w)["c"] - model.technology_at(dn, r, w)["c"])) / (2 * h * p[i])
        np.testing.assert_allclose(d, t["a"][i] * model.tau[i], rtol=1e-7, atol=1e-9)
    for i in (1,):
        up = w.copy(); dn = w.copy(); up[i] *= 1 + h; dn[i] *= 1 - h
        d = ((1 - model.t) * (model.technology_at(p, r, up)["c"] - model.technology_at(p, r, dn)["c"])) / (2 * h * w[i])
        np.testing.assert_allclose(d, np.where(model.country == i, t["bL"], 0.0), rtol=1e-7, atol=1e-9)


def _ces_scalar(values, shares, sigma):
    """Plain-loop CES index for one nest (one on an empty nest)."""
    values = np.asarray(values, float); shares = np.asarray(shares, float)
    if shares.sum() <= 0:
        return 1.0
    if abs(sigma - 1) < 1e-12:
        return float(np.exp(np.sum(shares * np.log(values))))
    return float(np.sum(shares * values ** (1 - sigma)) ** (1 / (1 - sigma)))


@pytest.mark.parametrize("values", [(.3, .5, 1.5, .8), (.5, 1, 4, .7), (1, .2, 3, 2), (0, 2, 2, 1)])
def test_nest_orientation_pinned_by_independent_loops(hand_active, values):
    """Origins (countries) nest within each material sector at sigma_origins; sectors at sigma_sectors.

    Every other check differentiates or re-solves the module's own residual, so
    an axis swap (sectors within each origin country) would pass them; here
    m_j, c_j and a_ij are rebuilt with explicit per-buyer loops at a far
    interior point and the swapped orientation is shown to differ whenever
    sigma_sectors != sigma_origins (both collapse to one flat index when equal).
    """
    calib = hand_active["calib"]; ta, tf = hand_active["tariffs"]
    sv, ss, so, rho = values
    model = _build_model(calib, ta, tf, None, None, tech_of(values))
    nc, ns, n = model.nc, model.ns, model.n
    rng = np.random.default_rng(5)
    x = build_initial_guess(calib)
    x[:2 * n] += rng.uniform(-.5, .5, 2 * n); x[2 * n:2 * n + 2 * nc] += rng.uniform(-.5, .5, 2 * nc)
    p, y, r, w, T, XN, B = model.unpack(x)
    t = model.technology_at(p, r, w)
    a0, tau = model.a0, model.tau
    m_ind = np.empty(n); m_swapped = np.empty(n); c_ind = np.empty(n); a_ind = np.empty((n, n))
    for j in range(n):
        rr = p * tau[:, j]
        A0 = a0[:, j].sum()
        h = np.empty(ns); mass = np.empty(ns)
        for s in range(ns):                                   # origins within material sector s
            idx = [cc * ns + s for cc in range(nc)]
            mass[s] = a0[idx, j].sum()
            h[s] = _ces_scalar(rr[idx], a0[idx, j] / mass[s] if mass[s] > 0 else np.zeros(nc), so)
        m_ind[j] = _ces_scalar(h, mass / A0, ss)
        g = np.empty(nc); massc = np.empty(nc)
        for cc in range(nc):                                  # swapped: sectors within origin country cc
            idx = [cc * ns + s for s in range(ns)]
            massc[cc] = a0[idx, j].sum()
            g[cc] = _ces_scalar(rr[idx], a0[idx, j] / massc[cc] if massc[cc] > 0 else np.zeros(ns), so)
        m_swapped[j] = _ces_scalar(g, massc / A0, ss)
        cj = model.country[j]
        vhat = _ces_scalar([w[cj], r[cj]], [1 - model.alpha[j], model.alpha[j]], rho)
        c_ind[j] = _ces_scalar([vhat, m_ind[j]], [model.v0[j] / (1 - model.t[j]), A0 / (1 - model.t[j])], sv)
        for i in range(n):
            s = i % ns
            a_ind[i, j] = a0[i, j] * (c_ind[j] / m_ind[j]) ** sv * (m_ind[j] / h[s]) ** ss * (h[s] / rr[i]) ** so
    np.testing.assert_allclose(t["m"], m_ind, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(t["c"], c_ind, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(t["a"], a_ind, rtol=1e-13, atol=1e-13)
    if ss != so:
        assert np.max(np.abs(m_swapped - t["m"])) > 1e-6
    else:
        np.testing.assert_allclose(m_swapped, t["m"], rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# 7-8. Certificates, ledgers, homogeneity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["hand", "hand_active", "oecd3"], indirect=True)
@pytest.mark.parametrize("values", TECHS + FLAT)
def test_certificate_and_accounting_ledger_on_every_solution(case, values):
    calib = case["calib"]; ta, tf = case["tariffs"]
    tech = tech_of(values)
    if calib.nc == 3 and calib.ns == 3 and values[2] == 1:
        pytest.skip("unit origin elasticity is a near-singular point of the OECD fixture's Jacobian; "
                    "see test_unit_origin_elasticity_is_near_singular_on_oecd_fixture")
    res = solve_ces_block_newton(calib, ta, tf, technology=tech)
    assert res.converged and res.equilibrium.converged
    assert max(res.certificate.values()) <= 1e-8
    again = certify_ces_equilibrium(calib, res.equilibrium.x_sol, tau=ta, tau_fd=tf, technology=tech)
    assert again == res.certificate
    ledger = res.equilibrium.metadata["account_residuals"]
    scale = np.max(_build_model(calib, ta, tf, None, None, tech).scale)
    assert max(ledger.values()) <= 1e-10 * max(1.0, scale)       # relative to the largest benchmark scale
    assert res.equilibrium.max_residual <= res.equilibrium.metadata["tol"]
    # The loop's acceptance equals the postprocessing audit, implied rows included.
    assert np.max(np.abs(res.equilibrium.metadata["physical_residuals"])) <= res.equilibrium.metadata["tol"]
    if tech.is_flat:
        block = evaluate(res.equilibrium.x_sol, calib, ta, tf, None, None, sigma=tech.sigma_origins)
        their_ledger = postprocess(block, calib, None)["accounting_metadata"]["account_residuals"]
        assert max(their_ledger.values()) <= 1e-10 * max(1.0, scale)
        assert res.certificate["dropped_equation"] <= 1e-8      # 3e-10 observed on the OECD fixture


def test_unit_origin_elasticity_is_near_singular_on_oecd_fixture(oecd3):
    """The fixture with a 10% US tariff has a nearly singular Jacobian at sigma_origins = 1.

    The singularity is a property of the consistent-accounting model, shown here
    independently of this module: central differences of ``_accounting.evaluate``
    at the benchmark give a determinant that changes sign between origin
    elasticities .7 and .8 and again between .98 and .99, with smallest singular
    value 4.8e-6 at .99 against 5e-3 at 0 and 2 (rows divided by their benchmark
    scales, T and XN columns multiplied by benchmark income). puremacro's
    finite-difference Newton and hybr also failed there when this was written
    (observed, not asserted: that is another module's behaviour). The block
    Newton reports the condition-number growth and raises, and the homotopy
    certifies the (1,1,1,1) endpoint while the (0,1,1,1) path stalls near the
    benchmark. Neighbouring elasticities solve.
    """
    calib = oecd3["calib"]; rates, _ = oecd3["tariffs"]
    x0 = build_initial_guess(calib)
    model = _build_model(calib, None, None, None, None, NestedCESTechnology())
    ta0, tf0 = model.ta, model.tf                                   # the benchmark: no tariffs
    n, nc = model.n, model.nc
    col = np.ones(model.dim); col[2 * n + 2 * nc:] = model.gdp0[np.r_[0:nc, 0:nc - 1]]
    smin, sign = {}, {}
    for sigma in (0., .7, .8, .98, .99, 2.):
        J = fd_jacobian_accounting(calib, x0, ta0, tf0, sigma, rel=1e-7) / model.scale[:, None] * col[None, :]
        smin[sigma] = np.linalg.svd(J, compute_uv=False)[-1]
        sign[sigma] = np.sign(np.linalg.det(J))
    assert smin[.99] < 2e-5 and min(smin[0.], smin[2.]) > 1e-3
    assert sign[.7] != sign[.8] and sign[.98] != sign[.99]
    for values in ((0, 1, 1, 1), (1, 1, 1, 1)):
        with pytest.raises(CESNewtonError, match="macro condition") as failure:
            solve_ces_block_newton(calib, rates, rates, technology=tech_of(values), max_iter=30)
        assert failure.value.history and np.isfinite(failure.value.macro_condition)
    path = continue_tariff_homotopy(calib, rates, rates, technology=tech_of((1, 1, 1, 1)), max_iter=30)
    assert path.converged and path.stages[-1]["fraction"] == 1.0
    assert max(certify_ces_equilibrium(calib, path.equilibrium.x_sol, tau=rates, tau_fd=rates,
                                       technology=tech_of((1, 1, 1, 1))).values()) <= 1e-8
    with pytest.raises(CESNewtonError, match="unresolved") as failure:
        continue_tariff_homotopy(calib, rates, rates, technology=tech_of((0, 1, 1, 1)), max_iter=30)
    assert failure.value.failed_stages and failure.value.last_result is not None
    for values in ((0, .9, .9, 1), (0, 2, 2, 1)):
        assert solve_ces_block_newton(calib, rates, rates, technology=tech_of(values), max_iter=30).converged


def test_nominal_homogeneity_and_currency_units(hand):
    data, calib = hand["data"], hand["calib"]; ta, tf = hand["tariffs"]
    tech = NestedCESTechnology(.3, .5, 1.5, .8)
    res = solve_ces_block_newton(calib, ta, tf, technology=tech, tol=1e-13)
    model = _build_model(calib, ta, tf, None, None, tech)
    e = res.equilibrium
    scale = 3.7
    xx = pack_equilibrium_vector(e.p_sol * scale, e.y_sol, e.r_sol * scale, e.w_sol * scale, e.T_sol * scale, e.XN_sol * scale)
    base, scaled = model.blocks(e.x_sol), model.blocks(xx)
    for key in ("pp", "income", "expenditure", "tariffs", "government"):
        np.testing.assert_allclose(scaled[key], base[key] * scale, rtol=1e-10, atol=1e-10)
    active = model.active[None]                       # absent categories keep the placeholder index one
    for key in ("Q", "P"):
        np.testing.assert_allclose(scaled[key][active], base[key][active] * scale, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(scaled[key][~active], 1.0)
    for key in ("Z", "F", "c", "labor", "capital"):
        np.testing.assert_allclose(scaled[key], base[key], rtol=1e-10, atol=1e-10)
    # Currency units: the table in thousandths gives identical prices and welfare percentages.
    calib2 = calibrate_trade_model(data * .001, ns=2, nc=3, nfd=3, country_codes=["A", "B", "C"])
    res2 = solve_ces_block_newton(calib2, ta, tf, technology=tech, tol=1e-13)
    for name in ("p_sol", "w_sol", "r_sol", "cpi", "terms_of_trade"):
        np.testing.assert_allclose(getattr(res2.equilibrium, name), getattr(e, name), rtol=1e-10, atol=1e-10)
    for name in ("y_sol", "T_sol", "gdp", "tariffs"):
        np.testing.assert_allclose(getattr(res2.equilibrium, name), getattr(e, name) * .001, rtol=1e-10, atol=1e-10)
    flat = NestedCESTechnology(0, .5, .5, 1)
    w1 = compute_hicksian_welfare(calib, solve_ces_block_newton(calib, ta, tf, technology=flat, tol=1e-13).equilibrium,
                                  base_result=solve_ces_block_newton(calib, technology=flat, tol=1e-13).equilibrium,
                                  target_country="B", consumption_categories=(0, 2))
    w2 = compute_hicksian_welfare(calib2, solve_ces_block_newton(calib2, ta, tf, technology=flat, tol=1e-13).equilibrium,
                                  base_result=solve_ces_block_newton(calib2, technology=flat, tol=1e-13).equilibrium,
                                  target_country="B", consumption_categories=(0, 2))
    assert w2.ev_pct_consumption == pytest.approx(w1.ev_pct_consumption, abs=1e-8)
    assert w2.ev == pytest.approx(.001 * w1.ev, abs=1e-10)


def test_country_permutation_invariance_given_the_numeraire(hand):
    """Reordering countries changes nothing when the numeraire cell is kept, or when the fixed
    foreign saving is restated in the new numeraire's units.

    The closure fixes B0 (``calib.invforT``) in units of the first producer
    price, so a permutation that moves the numeraire is a different closure
    unless ``invforT`` is divided by the original solution's price of the new
    numeraire cell; with that rescaling the solver is exactly permutation
    invariant (1e-10 asserted, 1e-14 observed).
    """
    data, calib = hand["data"], hand["calib"]; ta, tf = hand["tariffs"]
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    n = nc * ns
    tech = NestedCESTechnology(.3, .5, 1.5, .8)
    codes = ["A", "B", "C"]

    def permuted(perm):
        cell = np.concatenate([np.arange(c * ns, (c + 1) * ns) for c in perm])
        fdcol = np.concatenate([np.arange(c * nfd, (c + 1) * nfd) for c in perm])
        top = np.hstack([data[:n, :n][np.ix_(cell, cell)], data[:n, n:][np.ix_(cell, fdcol)]])
        rows = np.hstack([data[n:, :n][:, cell], data[n:, n:][:, fdcol]])
        calib_p = calibrate_trade_model(np.vstack([top, rows]), ns=ns, nc=nc, nfd=nfd,
                                        country_codes=[codes[c] for c in perm])
        return cell, calib_p, ta[cell][:, :, perm], tf[cell][:, :, perm]

    def check(a, b, cell, perm, numeraire_cell):
        ea, eb = a.equilibrium, b.equilibrium
        pa, ya = ea.p_sol.ravel(order="F"), ea.y_sol.ravel(order="F")
        num = pa[numeraire_cell]
        np.testing.assert_allclose(eb.p_sol.ravel(order="F"), pa[cell] / num, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(eb.y_sol.ravel(order="F"), ya[cell], rtol=1e-10, atol=0)
        np.testing.assert_allclose(eb.w_sol.ravel(), ea.w_sol.ravel()[perm] / num, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(eb.r_sol.ravel(), ea.r_sol.ravel()[perm] / num, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(eb.T_sol.ravel(), ea.T_sol.ravel()[perm] / num, rtol=1e-10, atol=0)
        np.testing.assert_allclose(eb.tariffs, ea.tariffs[perm] / num, rtol=1e-10, atol=1e-10 * np.max(np.abs(ea.tariffs)))
        np.testing.assert_allclose(eb.cpi, ea.cpi[perm], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(eb.terms_of_trade, ea.terms_of_trade[perm], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(eb.intermediate_flows.reshape(n, n, order="F"),
                                   ea.intermediate_flows.reshape(n, n, order="F")[np.ix_(cell, cell)],
                                   rtol=1e-10, atol=1e-10)

    a = solve_ces_block_newton(calib, ta, tf, technology=tech, tol=1e-13)
    cell, calib_p, ta_p, tf_p = permuted([0, 2, 1])            # the numeraire cell stays first
    check(a, solve_ces_block_newton(calib_p, ta_p, tf_p, technology=tech, tol=1e-13), cell, [0, 2, 1], 0)
    cell, calib_p, ta_p, tf_p = permuted([1, 2, 0])            # the numeraire moves to B's first sector
    p_b0 = a.equilibrium.p_sol.ravel(order="F")[ns]
    calib_q = replace(calib_p, invforT=np.asarray(calib_p.invforT) / p_b0)
    check(a, solve_ces_block_newton(calib_q, ta_p, tf_p, technology=tech, tol=1e-13), cell, [1, 2, 0], ns)
    with pytest.raises(AssertionError):                        # without the rescaling it is a different closure
        check(a, solve_ces_block_newton(calib_p, ta_p, tf_p, technology=tech, tol=1e-13), cell, [1, 2, 0], ns)


# ---------------------------------------------------------------------------
# 9. Admissibility and failure contract
# ---------------------------------------------------------------------------

def test_admissibility_and_failure_contract(hand):
    calib = hand["calib"]; ta, tf = hand["tariffs"]
    tech = NestedCESTechnology(.3, .5, 1.5, .8)
    x0 = build_initial_guess(calib)
    bad = x0.copy(); bad[-1] = 1e6            # foreign saving above income: negative absorption
    with pytest.raises(CESNewtonError, match="inadmissible"):
        solve_ces_block_newton(calib, ta, tf, technology=tech, x0=bad)
    model = _build_model(calib, ta, tf, None, None, tech)
    b = model.blocks(x0)
    assert model.admissible(b)
    b["quantity"] = b["quantity"].copy(); b["quantity"][~model.active] = 1e-3   # positive on an absent category
    assert not model.admissible(b)
    with pytest.raises(CESNewtonError, match="iteration limit") as failure:
        solve_ces_block_newton(calib, ta, tf, technology=tech, max_iter=0)
    assert failure.value.history == ()
    with pytest.raises(CESNewtonError, match="iteration limit") as failure:
        solve_ces_block_newton(calib, ta, tf, technology=tech, max_iter=1)
    assert len(failure.value.history) == 1 and np.isfinite(failure.value.macro_condition)
    with pytest.raises(CESNewtonError, match="certificate") as failure:
        solve_ces_block_newton(calib, ta, tf, technology=tech, certificate_tol=1e-300)
    assert failure.value.certificate is not None
    with pytest.raises(CESNewtonError, match="line search"):
        solve_ces_block_newton(calib, ta, tf, technology=tech, max_backtracks=1, x0=x0 + 3.0)
    # Input validation never reaches the solver.
    wrong = ta.copy(); wrong[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="domestic"):
        solve_ces_block_newton(calib, wrong, tf, technology=tech)
    with pytest.raises(ValueError, match="nonnegative"):
        NestedCESTechnology.validated(sigma_origins=-1)
    with pytest.raises(ValueError, match="rho_va"):
        NestedCESTechnology.validated(rho_va=0)
    with pytest.raises(ValueError, match="finite"):
        NestedCESTechnology.validated(sigma_sectors=np.inf)
    with pytest.raises(TypeError):
        solve_ces_block_newton(calib, ta, tf, technology=(0, 0, 0, 1))
    with pytest.raises(ValueError, match="max_cells"):
        CESBlockJacobian(calib, x0, tau=ta, tau_fd=tf, technology=tech, max_cells=5)
    with pytest.raises(ValueError, match="block_size"):
        CESBlockJacobian(calib, x0, tau=ta, tau_fd=tf, technology=tech, block_size=0)
    jac = CESBlockJacobian(calib, x0, tau=ta, tau_fd=tf, technology=tech)
    jac.L_r = jac.L_r.copy(); jac.L_r[0, 0] = np.nan          # what a singular price factorization would leave
    with pytest.raises(CESNewtonError, match="Nonfinite CES macro") as failure:
        jac._assemble_macro(16)
    assert failure.value.macro_condition == np.inf
    with pytest.raises(ValueError, match="tol"):
        solve_ces_block_newton(calib, ta, tf, technology=tech, tol=0)
    with pytest.raises(TypeError, match="Newton options"):
        continue_tariff_homotopy(calib, ta, tf, technology=tech, tolerance=1e-9)


def test_newton_stops_only_where_the_postprocessing_audit_holds(hand, hand_active):
    """One acceptance test: the loop stops only where the level audit of postprocessing holds.

    Regression: the loop used to stop on the scaled imposed rows while postprocessing
    (and ``compute_hicksian_welfare``) audit every level row, including the omitted
    goods equation and the realized foreign balances that only Walras' law imposes.
    At this schedule the old loop stopped at scaled residual 1.45e-11 with a net-export
    gap of 4.8e-9 against the bound 3.9e-9 and raised "Postprocessed state fails the
    absolute physical/accounting audit"; now it takes one more (polishing) step.
    """
    calib = hand["calib"]; ta, tf = hand["tariffs"]
    s = 0.8653681121103338
    res = solve_ces_block_newton(calib, 1 + s * (ta - 1), 1 + s * (tf - 1), technology=NestedCESTechnology(.1, .1, .5, 1))
    meta = res.equilibrium.metadata
    last = res.history[-1]
    assert last["residual_before"] <= 2e-11 and last["level_audit_before"] > meta["tol"]    # the polishing step
    assert np.max(np.abs(meta["physical_residuals"])) <= meta["tol"] and res.equilibrium.max_residual <= meta["tol"]
    # The homotopy variant raised the same error outside its try block, with no stages and no last result.
    calib2 = hand_active["calib"]; ta2, tf2 = hand_active["tariffs"]
    path = continue_tariff_homotopy(calib2, 1 + .787 * (ta2 - 1), 1 + .787 * (tf2 - 1),
                                    technology=NestedCESTechnology(0, 0, 0, 1), tol=5e-12)
    assert path.converged and path.stages[-1]["fraction"] == 1.0
    # Sweep (66 solves, four of which need a polishing step): every solve returns an audited state.
    for case in (hand, hand_active):
        calib, (ta, tf) = case["calib"], case["tariffs"]
        for s in list(np.linspace(.2, 1.5, 8)) + [0.8653681121103338, .374, .787]:
            for values in ((0, 0, 0, 1), (.1, .1, .5, 1), (.3, .5, 1.5, .8)):
                res = solve_ces_block_newton(calib, 1 + s * (ta - 1), 1 + s * (tf - 1), technology=tech_of(values))
                meta = res.equilibrium.metadata
                assert res.equilibrium.converged and max(res.certificate.values()) <= 1e-8
                assert np.max(np.abs(meta["physical_residuals"])) <= meta["tol"]
                assert np.max(np.abs(res.equilibrium.residuals)) <= meta["tol"] and meta["demand_feasible"]


def test_certificate_detects_the_violations_it_names(hand, monkeypatch):
    """Each certificate key responds to the violation it names; 1e-8 is the acceptance gate.

    Perturbing one unknown of a certified state by 1e-5 (log units, or 1e-5 of benchmark
    income for T and XN) trips the named keys (observed 2e-7 to 1.2e-5); faults injected
    into the vectorized assembly trip the keys that compare against it; and a technology
    that violates the Euler identity (capital requirement scaled by 1 + 1e-6) still has a
    root of the imposed rows, where the certificate reports the cost adding-up, the
    omitted goods equation and the current account, and the solver refuses the state.
    """
    calib = hand["calib"]; ta, tf = hand["tariffs"]
    tech = NestedCESTechnology(.3, .5, 1.5, .8)
    res = solve_ces_block_newton(calib, ta, tf, technology=tech, tol=1e-13)
    model = _build_model(calib, ta, tf, None, None, tech)
    n, nc = model.n, model.nc
    x = res.equilibrium.x_sol
    b = model.blocks(x)
    assert max(_certificate(model, b).values()) <= 1e-12
    expected = {3: ("zero_profit", "goods_value", "root_residual"),                    # log p of cell 3
                n + 2: ("goods_value", "factor_clearing"),                            # log y of cell 2
                2 * n + 1: ("zero_profit", "factor_clearing"),                        # log r of B
                2 * n + nc + 2: ("zero_profit", "factor_clearing"),                   # log w of C
                2 * n + 2 * nc: ("government_budget", "gdp_identity", "dropped_equation"),   # T of A
                2 * n + 3 * nc + 1: ("current_account", "household_budget")}          # XN of B
    for index, keys in expected.items():
        xx = x.copy(); xx[index] += 1e-5 * (1.0 if index < 2 * n + 2 * nc else model.gdp0[0])
        cert = _certificate(model, model.blocks(xx))
        for key in keys:
            assert cert[key] > 1e-7, (index, key, cert)
    faulty = _certificate(model, dict(b, tariffs=b["tariffs"] * (1 + 1e-5)))          # recorded duty receipts
    assert faulty["tariff_revenue"] > 1e-7
    assert max(v for k, v in faulty.items() if k != "tariff_revenue") <= 1e-12
    faulty = _certificate(model, dict(b, exp_mat=b["exp_mat"] * (1 + 1e-5)))          # final expenditure
    assert faulty["household_budget"] > 1e-7 and faulty["gdp_identity"] > 1e-7
    real = ces_newton._Model.technology_at

    def euler_violating(self, p, r, w):
        out = dict(real(self, p, r, w))
        out["bK"] = out["bK"] * (1 + 1e-6)
        return out

    monkeypatch.setattr(ces_newton._Model, "technology_at", euler_violating)
    xm = x.copy()
    for _ in range(8):                                    # Newton on the imposed rows only
        bm = model.blocks(xm)
        xm = xm - CESBlockJacobian(calib, xm, _model=model, _blocks=bm).solve(bm["residuals"])
    bm = model.blocks(xm)
    assert np.max(np.abs(bm["residuals"] / model.scale)) < 1e-12
    cert = _certificate(model, bm)
    for key in ("cost_adding_up", "dropped_equation", "current_account"):
        assert cert[key] > 5e-8, (key, cert)             # observed 1.9e-7, 8.4e-7, 5.5e-7
    with pytest.raises(CESNewtonError, match="level audit"):
        solve_ces_block_newton(calib, ta, tf, technology=tech)


def test_line_search_failures_name_their_cause(rand20, hand_active):
    """The error separates an admissibility-bound direction from an Armijo failure and a rounding floor."""
    calib = rand20["calib"]
    ta, tf = random_schedule(calib, seed=0)
    with pytest.raises(CESNewtonError, match="left the admissible domain") as failure:
        solve_ces_block_newton(calib, ta, tf, technology=NestedCESTechnology(0, 2, 2, 1), max_backtracks=10)
    message = str(failure.value)
    assert "final quantity" in message and "investment" in message
    assert any(f" in {code} " in message for code in calib.country_codes)
    assert failure.value.history[-1]["rejected_inadmissible"] > 0
    # Prohibitive multipliers on final and intermediate imports from B into A: with fixed final
    # origin baskets the rebated duties reach about 3e5 (1e6) and 3e7 (1e8) times benchmark
    # income, so rounding in the implied rows exceeds tol * max(scale) at the default tol.
    calib = hand_active["calib"]
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    # Whether that rounding crosses tol depends on the BLAS build (one case solved cleanly
    # on Windows CI), so the invariant is: refuse by name, or meet the acceptance contract,
    # whose independent flow certificate is bounded by certificate_tol (default 1e-8), not tol.
    for multiplier, tol in ((1e8, 2e-11), (1e8, 1e-9)):
        ta = np.ones((nc * ns, ns, nc)); tf = np.ones((nc * ns, nfd, nc))
        ta[ns:2 * ns, :, 0] = multiplier; tf[ns:2 * ns, :, 0] = multiplier
        try:
            res = solve_ces_block_newton(calib, ta, tf, technology=NestedCESTechnology(0, 4, 4, 1), tol=tol, max_iter=40)
        except CESNewtonError as error:
            assert "level audit" in str(error)
        else:
            assert max(res.certificate.values()) <= 1e-8
    ta = np.ones((nc * ns, ns, nc)); tf = np.ones((nc * ns, nfd, nc))
    ta[ns:2 * ns, :, 0] = 1e6; tf[ns:2 * ns, :, 0] = 1e6
    res = solve_ces_block_newton(calib, ta, tf, technology=NestedCESTechnology(0, 4, 4, 1), tol=1e-9, max_iter=40)
    assert res.equilibrium.T_sol.ravel()[0] > 1e7 and max(res.certificate.values()) <= 1e-8


def test_homotopy_aborts_carry_their_record(hand, monkeypatch):
    """Every abort carries stages, failed stages and the last result; packaging failures are attempts."""
    calib = hand["calib"]; ta, tf = hand["tariffs"]
    tech = NestedCESTechnology(.3, .5, 1.5, .8)
    with pytest.raises(CESNewtonError, match="start schedule unresolved") as failure:
        continue_tariff_homotopy(calib, ta, tf, technology=tech, tau_start=ta, tau_fd_start=tf, max_iter=0)
    assert failure.value.stages == () and failure.value.failed_stages == () and failure.value.last_result is None
    real = ces_newton._package

    def flaky(model, b, info, **kw):
        stages = kw.get("stages", ())
        if stages and stages[-1]["fraction"] == .25:
            raise CESNewtonError("Postprocessed state fails the absolute physical/accounting audit (injected)")
        return real(model, b, info, **kw)

    monkeypatch.setattr(ces_newton, "_package", flaky)
    res = continue_tariff_homotopy(calib, ta, tf, technology=tech)
    assert [f["fraction"] for f in res.failed_stages] == [.25] and "injected" in res.failed_stages[0]["error"]
    fractions = [s["fraction"] for s in res.stages]
    assert .25 not in fractions and fractions[:2] == [0.0, .125] and fractions[-1] == 1.0


# ---------------------------------------------------------------------------
# 10. Homotopy and operator behaviours
# ---------------------------------------------------------------------------

def test_homotopy_certifies_every_stage_and_matches_direct_solve(hand):
    calib = hand["calib"]; ta, tf = hand["tariffs"]
    tech = NestedCESTechnology(.3, .5, 1.5, .8)
    seen = []
    res = continue_tariff_homotopy(calib, ta, tf, technology=tech,
                                   stage_callback=lambda r, rec: seen.append((r, rec)))
    fractions = [s["fraction"] for s in res.stages]
    assert fractions[0] == 0.0 and fractions[-1] == 1.0 and np.all(np.diff(fractions) > 0)
    assert all(s["certificate_max"] <= 1e-8 for s in res.stages)
    assert res.failed_stages == ()
    assert len(seen) == len(res.stages) and all(isinstance(r, CESBlockNewtonResult) and r.converged for r, _ in seen)
    assert seen[-1][1]["fraction"] == 1.0
    direct = solve_ces_block_newton(calib, ta, tf, technology=tech)
    np.testing.assert_allclose(res.equilibrium.x_sol, direct.equilibrium.x_sol, rtol=1e-10, atol=1e-10)
    assert res.calls == sum(s["calls"] for s in res.stages)
    assert len(res.stages_frame()) == len(res.stages)
    # Forced failures: every stage fails, the step halves to the floor and the last certified state is kept.
    with pytest.raises(CESNewtonError, match="unresolved") as failure:
        continue_tariff_homotopy(calib, ta, tf, technology=tech, max_iter=0, min_step=1 / 8)
    assert len(failure.value.failed_stages) == 2
    assert [f["fraction"] for f in failure.value.failed_stages] == [.25, .125]
    assert failure.value.last_result.stages[-1]["fraction"] == 0.0
    assert len(failure.value.last_result.failed_stages) == 2
    # A partial path can be resumed from a non-benchmark start schedule.
    half = 1 + .5 * (ta - 1), 1 + .5 * (tf - 1)
    resumed = continue_tariff_homotopy(calib, ta, tf, technology=tech, tau_start=half[0], tau_fd_start=half[1],
                                       x0=solve_ces_block_newton(calib, *half, technology=tech).equilibrium.x_sol)
    np.testing.assert_allclose(resumed.equilibrium.x_sol, direct.equilibrium.x_sol, rtol=1e-10, atol=1e-10)


def test_homotopy_resolves_high_origin_elasticity_on_random_20x11(rand20):
    """A direct Newton from the benchmark fails or agrees; the homotopy certifies the endpoint either way.

    Observed 2026-09-23 (not asserted): the direct Newton fails in the line search with 25 Armijo
    rejections at scaled residual 9.2e-2 and macro condition 8.4e10; the homotopy stages are
    [0, .25, .625, 1].
    """
    calib = rand20["calib"]; ta, tf = rand20["tariffs"]
    tech = NestedCESTechnology(.5, 1, 4, .7)
    path = continue_tariff_homotopy(calib, ta, tf, technology=tech)
    assert path.converged and path.stages[-1]["fraction"] == 1.0
    assert all(s["certificate_max"] <= 1e-8 for s in path.stages)
    audit = certify_ces_equilibrium(calib, path.equilibrium.x_sol, tau=ta, tau_fd=tf, technology=tech)
    assert max(audit.values()) <= 1e-8
    try:
        direct = solve_ces_block_newton(calib, ta, tf, technology=tech)
    except CESNewtonError as failure:
        assert failure.history or "inadmissible" in str(failure) or "iteration limit" in str(failure)
    else:
        np.testing.assert_allclose(direct.equilibrium.x_sol, path.equilibrium.x_sol, rtol=1e-9, atol=1e-9)


def test_block_size_independence_and_linear_operators(hand):
    calib = hand["calib"]; ta, tf = hand["tariffs"]
    tech = NestedCESTechnology(.5, 1, 4, .7)
    model = _build_model(calib, ta, tf, None, None, tech)
    x = interior_point(model, np.random.default_rng(4))
    r = np.random.default_rng(1).normal(size=len(x)) * model.scale
    sols = [CESBlockJacobian(calib, x, tau=ta, tau_fd=tf, technology=tech, block_size=bs).solve(r) for bs in (1, 8, 16, 64)]
    for s in sols[1:]:
        np.testing.assert_allclose(s, sols[0], rtol=1e-12, atol=1e-12)
    jac = CESBlockJacobian(calib, x, tau=ta, tau_fd=tf, technology=tech)
    np.testing.assert_array_equal(jac.as_preconditioner() @ r, jac.solve(r))
    np.testing.assert_array_equal(jac.as_operator() @ sols[0], jac.apply(sols[0]))
    assert jac.shape == (len(x), len(x)) and np.isfinite(jac.macro_condition)
    assert jac.n == calib.nc * calib.ns and jac.dim == jac.shape[0] == 2 * jac.n + 4 * calib.nc - 1


def test_damped_newton_matches_independent_dense_root(hand):
    """IO structure: the block Newton endpoint equals a dense hybr root of the same residual."""
    calib = hand["calib"]; ta, tf = hand["tariffs"]
    tech = NestedCESTechnology(.5, .8, 3., .9)
    model = _build_model(calib, ta, tf, None, None, tech)
    res = solve_ces_block_newton(calib, ta, tf, technology=tech)
    sol = root(lambda x: model.residual(x) / model.scale, build_initial_guess(calib), method="hybr",
               jac=lambda x: complex_step_jacobian(model, x) / model.scale[:, None],
               options={"xtol": 1e-13, "maxfev": 2000})
    assert sol.success
    np.testing.assert_allclose(res.equilibrium.x_sol, sol.x, rtol=1e-9, atol=1e-10)
    assert max(res.certificate.values()) < 1e-9


# ---------------------------------------------------------------------------
# 11-13. Scale, golden, contract
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_scale_77x11_random_table():
    calib = synthetic_calibration(77, 11)
    rates = np.zeros(calib.nc); rates[0] = .1
    tech = NestedCESTechnology(.1, .1, .5, 1)
    x0 = build_initial_guess(calib)
    model = _build_model(calib, rates, rates, None, None, tech)
    tic = time.perf_counter(); jac = CESBlockJacobian(calib, x0, tau=rates, tau_fd=rates, technology=tech)
    build = time.perf_counter() - tic
    assert jac.shape == (2 * 847 + 4 * 77 - 1, 2 * 847 + 4 * 77 - 1) and jac.macro_size == 232
    b = model.blocks(x0)
    tic = time.perf_counter(); v = jac.solve(b["residuals"]); solve = time.perf_counter() - tic
    assert build < 10 and solve < 10
    np.testing.assert_allclose(jac.apply(v) / model.scale, b["residuals"] / model.scale, rtol=0, atol=1e-9)
    res = solve_ces_block_newton(calib, rates, rates, technology=tech)
    assert res.converged and res.iterations <= 6 and max(res.certificate.values()) <= 1e-8
    ta, tf = random_schedule(calib)
    res = solve_ces_block_newton(calib, ta, tf, technology=NestedCESTechnology(.5, 1, 4, .7))
    assert res.converged and max(res.certificate.values()) <= 1e-8   # 10 iterations observed: no bound claimed


def test_golden_oecd3x3(oecd3):
    calib = oecd3["calib"]; rates, _ = oecd3["tariffs"]
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    tech = NestedCESTechnology(**golden["technology"])
    res = solve_ces_block_newton(calib, rates, rates, technology=tech)
    i = list(calib.country_codes).index("USA")
    assert res.equilibrium.w_sol.ravel()[i] == pytest.approx(golden["us_wage"], abs=1e-8)
    assert res.equilibrium.r_sol.ravel()[i] == pytest.approx(golden["us_rent"], abs=1e-8)   # equals the wage by construction (uniform alpha, rho_va = 1)
    assert res.equilibrium.gdp[i] == pytest.approx(golden["us_gdp"], rel=1e-8)
    assert res.equilibrium.tariffs[i] == pytest.approx(golden["us_tariff_revenue"], rel=1e-8)
    assert res.macro_condition == pytest.approx(golden["macro_condition"], rel=1e-6)
    assert res.iterations == golden["iterations"]
    assert res.residual_max <= 2e-11 and max(res.certificate.values()) <= 1e-8


def test_result_contract_and_renderers(hand):
    calib = hand["calib"]; ta, tf = hand["tariffs"]
    tech = NestedCESTechnology.from_flexible(type("Cfg", (), {"sigma_y": .2, "sigma_inter": .8, "rho_va": .9})())
    assert tech.elasticities == (.2, .8, .8, .9) and not tech.is_flat
    assert NestedCESTechnology(0, None, .5, 1).is_flat and NestedCESTechnology(0, .4, .5, 1).is_flat is False
    res = solve_ces_block_newton(calib, ta, tf, technology=tech, progress=lambda rec: None)
    assert isinstance(res, CESBlockNewtonResult) and res.technology == tech
    assert res.country_codes == ("A", "B", "C") and len(res.sector_codes) == 2
    frame = res.to_dataframe()
    assert list(frame.columns) == ["iteration", "residual_before", "step", "macro_condition", "residual_after"]
    assert len(frame) == res.iterations
    for text in (res.to_markdown(), res.to_latex(), res.to_typst(), res.summary()):
        assert isinstance(text, str) and text
    assert "converged" in res.summary()
    assert set(res.certificate_frame().index) == set(res.certificate)
    e = res.equilibrium
    assert e.metadata["accounting"] == "consistent" and e.metadata["method"] == "ces_block_newton"
    assert e.metadata["technology"] == tech.to_dict() and e.metadata["fiscal_closure"] == "lump_sum"
    assert "intermediate_tariff_multipliers" in e.metadata and "certificate" in e.metadata
    assert e.metadata["tol"] == pytest.approx(2e-11 * np.max(_build_model(calib, ta, tf, None, None, tech).scale))
    assert res.metadata["macro_size"] == 3 * calib.nc + 1
    assert e.converged and e.max_residual <= e.metadata["tol"]


def test_module_imports_only_the_pyodide_core():
    import puremacro.trade.ces_newton as module
    src = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    for name in ("torch", "numba", "statsmodels", "matplotlib"):
        assert f"import {name}" not in src
    heads = [ln.split()[1].split(".")[0] for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not set(heads) & {"pandas", "matplotlib", "torch", "numba", "statsmodels"}


# ---------------------------------------------------------------------------
# Parity with the IO kernel (subprocess, skipped without the research volume)
# ---------------------------------------------------------------------------

def test_ces_index_matches_io_kernel(tmp_path):
    vendor = IO_ROOT / "headlinePaper" / "rebuild" / "vendor"
    if not (vendor / "puremacro" / "trade" / "corrected" / "ces.py").exists():
        pytest.skip("IO research volume not mounted")
    rng = np.random.default_rng(21)
    prices = rng.uniform(.5, 2., size=(4, 6)); shares = rng.uniform(0, 1, size=(4, 6))
    shares[:, 2] = 0.0                                   # one empty nest
    shares /= np.where(shares.sum(0) > 0, shares.sum(0), 1.)[None]
    np.save(tmp_path / "prices.npy", prices); np.save(tmp_path / "shares.npy", shares)
    code = ("import sys, json, numpy as np; sys.path.insert(0, sys.argv[1]); "
            "from puremacro.trade.corrected.ces import ces_index; "
            "p = np.load(sys.argv[2]); s = np.load(sys.argv[3]); "
            "print(json.dumps({str(sig): ces_index(p, s, sig).tolist() for sig in (0., .5, 1., 2., 4.)}))")
    # -B: never write bytecode into the read-only research volume.
    run = subprocess.run([sys.executable, "-B", "-c", code, str(vendor), str(tmp_path / "prices.npy"),
                          str(tmp_path / "shares.npy")], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-500:]
    reference = json.loads(run.stdout.strip().splitlines()[-1])
    for sig, expected in reference.items():
        np.testing.assert_allclose(_ces_index(prices, shares, float(sig)), expected, rtol=1e-14, atol=1e-14)
