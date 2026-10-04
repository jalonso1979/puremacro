"""Parity: models written with puremacro.dp reproduce the hand-coded vfi examples.

A dp spec compiles to the same solver, grids and chain as the example it
mirrors, so prices, aggregates, value functions and policy indices must agree
exactly (up to float noise from a different summation order).
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro import dp
from puremacro.vfi.examples import (
    aiyagari_steady_state,
    huggett_steady_state,
    life_cycle_profile,
)


def aiyagari_spec(beta=0.96):
    m = dp.Model("aiyagari")
    m.parameters(beta=beta, gamma=1.0, alpha=0.36, delta=0.08)
    m.prices(r=(0.005, 1.0 / beta - 1.0 - 0.002))
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
    m.state("a", np.linspace(1e-4, 80.0, 150))
    m.local("w = (1 - alpha)*((alpha/(r + delta))^(1/(1 - alpha)))^alpha")
    m.local("c = w*exp(z) + (1 + r)*a - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(K="a", L="exp(z)")
    m.clear("K - L*(alpha/(r + delta))^(1/(1 - alpha))")
    return m


@pytest.fixture(scope="module")
def aiyagari_pair():
    return aiyagari_steady_state(), aiyagari_spec().solve(tol=1e-9, howard=40)


def test_aiyagari_prices_and_aggregates_match(aiyagari_pair):
    ref, sol = aiyagari_pair
    assert sol.prices["r"] == pytest.approx(ref["r"], abs=1e-12)
    assert sol.aggregates["K"] == pytest.approx(ref["K"], abs=1e-10)
    assert sol.aggregates["L"] == pytest.approx(ref["L"], abs=1e-12)


def test_aiyagari_value_policy_and_distribution_match(aiyagari_pair):
    ref, sol = aiyagari_pair
    eq = ref["equilibrium"]
    np.testing.assert_array_equal(sol.raw.policy_aprime, eq.solution.policy_aprime)
    np.testing.assert_allclose(sol.V, eq.solution.V, rtol=0, atol=1e-10)
    np.testing.assert_allclose(sol.distribution, eq.distribution, rtol=0, atol=1e-12)
    assert sol.mean("a") == pytest.approx(ref["K"], abs=1e-10)


def test_aiyagari_consumption_policy_obeys_budget(aiyagari_pair):
    _, sol = aiyagari_pair
    r, alpha, delta = sol.prices["r"], 0.36, 0.08
    w = (1 - alpha) * (alpha / (r + delta)) ** (alpha / (1 - alpha))
    a, z = sol.grids["a"], sol.grids["z"]
    expected = w * np.exp(z)[None, :] + (1 + r) * a[:, None] - sol.policy("a(+1)")
    np.testing.assert_allclose(sol.policy("c"), expected, atol=1e-12)
    assert np.all(sol.policy("c") > 0)


def test_huggett_matches():
    beta = 0.96
    ref = huggett_steady_state()
    m = dp.Model("huggett")
    m.parameters(beta=beta, gamma=1.5)
    m.prices(r=(-0.5, 1.0 / beta - 1.0 - 0.002))
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=7))
    m.state("b", np.linspace(-2.0, 24.0, 200))
    m.local("c = (1 + r)*b + exp(z) - b(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(B="b")
    m.clear("B")
    sol = m.solve(tol=1e-9, howard=40)
    assert sol.prices["r"] == pytest.approx(ref["r"], abs=1e-12)
    assert sol.aggregates["B"] == pytest.approx(ref["mean_assets"], abs=1e-10)
    frac = float(sol.distribution[sol.grids["b"] < 0].sum())
    assert frac == pytest.approx(ref["frac_borrowing"], abs=1e-12)


def test_life_cycle_matches():
    ref = life_cycle_profile()
    T = 40
    ages = np.arange(T)
    m = dp.Model("life cycle")
    m.parameters(beta=0.96, gamma=2.0, r=0.04, kappa=np.exp(0.1 * ages - 0.0025 * ages ** 2))
    m.exogenous("z", dp.AR1(rho=0.95, sigma=0.15, n=5))
    m.state("a", np.linspace(0.0, 40.0, 120))
    m.local("c = (1 + r)*a + kappa*exp(z) - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.horizon(T)
    sol = m.solve()
    np.testing.assert_array_equal(sol.raw.policy_aprime, ref["solution"].policy_aprime)
    np.testing.assert_allclose(sol.V, ref["solution"].V, rtol=0, atol=1e-10)
    np.testing.assert_allclose(sol.distribution, ref["life_cycle_dist"], atol=1e-14)
    np.testing.assert_allclose(sol.mean("a"), ref["assets_by_age"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sol.mean("c"), ref["consumption_by_age"], rtol=1e-12, atol=1e-12)
