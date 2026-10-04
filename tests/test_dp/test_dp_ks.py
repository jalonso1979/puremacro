"""Aggregate risk in dp models: the Krusell-Smith loop with a firm written as equations.

Parity is exact against ``vfi.krusell_smith`` when the dp model is given the
same grids and initial distribution (same household, chain, path and updates).
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from puremacro import dp
from puremacro.dp import ModelSpecError
from puremacro.vfi.discretize import markov_stationary
from puremacro.vfi.distribution import stationary_distribution
from puremacro.vfi.problem import VFIProblem

ksm = importlib.import_module("puremacro.vfi.krusell_smith")
ALPHA, DELTA, BETA, GAMMA = 0.36, 0.025, 0.99, 1.0
E = np.array([0.5, 1.1])
P_E = np.array([[0.6, 0.4], [0.2, 0.8]])
Z = np.array([0.99, 1.01])
P_Z = np.array([[0.875, 0.125], [0.125, 0.875]])
SMALL = dict(T=400, burn_in=100, max_outer=4, tol=1e-6)


def reference_grids(n_a=60, n_K=5):
    """The grids and starting distribution krusell_smith() builds internally."""
    L = float(markov_stationary(P_E) @ E)
    opts = dict(tol=1e-7, n_howard=40)
    K_cm = L * (ALPHA / (1 / BETA - 1 + DELTA)) ** (1 / (1 - ALPHA))
    a = np.linspace(0.0, 4 * K_cm, n_a)
    K_star = ksm._no_agg_risk_K((0.4 * K_cm, 4 * K_cm), z_vals=E, P_z=P_E, L=L, alpha=ALPHA, delta=DELTA,
                                beta=BETA, gamma=GAMMA, a_grid=a, options=opts)
    r0, w0 = ksm._firm_prices(1.0, K_star, L, ALPHA, DELTA)

    def rf0(ap, a_, zc, xp=np):
        c = (1 + r0) * a_ + w0 * zc - ap
        return xp.where(c > 0, ksm._util(c, GAMMA, xp), -np.inf)

    sol0 = VFIProblem(a_grid=a, z_grid=E, P_z=P_E, return_fn=rf0, beta=BETA, options=opts).solve("numpy")
    return L, a, np.linspace(0.75 * K_star, 1.25 * K_star, n_K), stationary_distribution(sol0.policy_aprime, P_E)


def ks_model(L, a, K_grid, firm=("r = alpha*Z*(K/L)^(alpha - 1) - delta", "w = (1 - alpha)*Z*(K/L)^alpha")):
    m = dp.Model("krusell-smith")
    m.parameters(beta=BETA, gamma=GAMMA, alpha=ALPHA, delta=DELTA, L=L)
    m.exogenous("e", dp.Markov(E, P_E))
    m.aggregate_shock("Z", dp.Markov(Z, P_Z))
    m.state("a", a)
    m.aggregate_state("K", K_grid, mean="a")
    m.local(*firm)
    m.local("c = (1 + r)*a + w*e - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    return m


def test_cobb_douglas_matches_krusell_smith_exactly():
    ref = ksm.krusell_smith(n_a=60, n_K=5, **SMALL)
    L, a, K_grid, mu0 = reference_grids()
    sol = ks_model(L, a, K_grid).solve(tol=1e-7, howard=40, ks=dict(SMALL, mu0=mu0))
    eq = sol.equilibrium
    np.testing.assert_array_equal(eq.b0, ref.b0)
    np.testing.assert_array_equal(eq.b1, ref.b1)
    np.testing.assert_array_equal(eq.K_path, ref.K_path)
    assert sol.method == "vfi+ks" and sol.aggregates["K"] == pytest.approx(ref.mean_K)
    assert sol.policy("c").shape == (60, 2 * 2 * 5)


def test_any_firm_written_as_equations():
    # CES technology with elasticity 0.8 instead of Cobb-Douglas
    L, a, K_grid, mu0 = reference_grids()
    ces = ("Y = Z*(alpha*K^s + (1 - alpha)*L^s)^(1/s)",
           "r = alpha*Z^s*(Y/K)^(1 - s) - delta",
           "w = (1 - alpha)*Z^s*(Y/L)^(1 - s)")
    m = ks_model(L, a, K_grid, firm=ces)
    m.parameters(s=-0.25)
    sol = m.solve(tol=1e-7, howard=40, ks=dict(SMALL, mu0=mu0))
    eq = sol.equilibrium
    assert np.all(np.isfinite(eq.b0)) and np.all(np.isfinite(eq.K_path))
    # r and w are the CES marginal products at every (Z, K) node
    s, Kc, Zc = -0.25, K_grid[2], Z[1]
    Y = Zc * (ALPHA * Kc ** s + (1 - ALPHA) * L ** s) ** (1 / s)
    r = sol.policy("r")[0].reshape(2, 2, 5)[0, 1, 2]
    assert r == pytest.approx(ALPHA * Zc ** s * (Y / Kc) ** (1 - s) - DELTA, rel=1e-12)


def test_aggregate_risk_specification_errors():
    L, a, K_grid, _ = reference_grids(n_a=10, n_K=3)
    m = ks_model(L, a, K_grid)
    m.prices(r=(0.0, 0.01))
    with pytest.raises(ModelSpecError, match="no prices"):
        m.solve()
    m = ks_model(L, a, K_grid)
    m._agg_state = ("K", K_grid, "c")
    with pytest.raises(ModelSpecError, match="mean must be the state"):
        m.solve()
    m = ks_model(L, a, K_grid)
    m._agg_state = None
    with pytest.raises(ModelSpecError, match="needs an aggregate_state"):
        m.solve()
    with pytest.raises(ModelSpecError, match="positive"):
        dp.Model().aggregate_state("K", [0.0, 1.0], mean="a")


def test_egm_households_give_the_krusell_smith_forecast_rule():
    # with continuous EGM policies aggregate capital responds to TFP and the
    # forecast rule converges to the familiar near-unit-root, R^2 ~ 1 law
    L, a, K_grid, _ = reference_grids(n_a=100, n_K=7)
    sol = ks_model(L, a, K_grid).solve("egm", tol=1e-6, max_iter=100_000,
                                       ks=dict(T=1000, burn_in=200, max_outer=40, tol=1e-3))
    eq = sol.equilibrium
    assert eq.converged and sol.method == "egm+ks"
    assert np.all((eq.b1 > 0.9) & (eq.b1 < 1.0))
    assert np.all(eq.r_squared > 0.999)
    assert np.std(eq.K_path[200:]) > 0.3
    assert K_grid[0] < eq.mean_K < K_grid[-1]
