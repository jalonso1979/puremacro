"""dp models compiled to the endogenous grid method (``solve(method="egm")``).

Parity is exact against hand-called ``vfi.solve_egm`` (same algorithm, grids and
chain), for CRRA through the closed-form inverse and for other felicities
through the symbolic derivative and the numeric inverse.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from puremacro import dp
from puremacro.dp import ModelSpecError
from puremacro.vfi.distribution import lottery_distribution
from puremacro.vfi.egm import solve_egm

ALPHA, DELTA, BETA = 0.36, 0.08, 0.96
A_GRID = np.linspace(1e-4, 80.0, 150)


def wage(r):
    return (1 - ALPHA) * ((ALPHA / (r + DELTA)) ** (1 / (1 - ALPHA))) ** ALPHA


def household(reward="crra(c, gamma)", gamma=1.0, r=0.03, **params):
    m = dp.Model("household")
    m.parameters(beta=BETA, gamma=gamma, alpha=ALPHA, delta=DELTA, r=r, **params)
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
    m.state("a", A_GRID)
    m.local("w = (1 - alpha)*((alpha/(r + delta))^(1/(1 - alpha)))^alpha")
    m.local("c = w*exp(z) + (1 + r)*a - a(+1)")
    m.reward(reward)
    m.subject_to("c > 0")
    return m


def reference(sol, r=0.03, **kw):
    z = sol.grids["z"]
    return solve_egm(A_GRID, z, wage(r) * np.exp(z), sol.P, beta=BETA, r=r, tol=1e-10, **kw)


def test_crra_household_matches_solve_egm():
    sol = household().solve("egm", tol=1e-10)
    ref = reference(sol, gamma=1.0)
    assert sol.method == "egm"
    np.testing.assert_allclose(sol.raw.c, ref.c, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sol.policy("a(+1)"), ref.aprime, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sol.policy("c"), ref.c, rtol=0, atol=1e-12)
    assert sol.raw.R == pytest.approx(1.03, abs=1e-12)


def test_explicit_power_utility_uses_numeric_inverse():
    sol = household("(c^(1 - gamma) - 1)/(1 - gamma)", gamma=2.0).solve("egm", tol=1e-10)
    ref = reference(sol, gamma=2.0)
    np.testing.assert_allclose(sol.raw.c, ref.c, rtol=0, atol=1e-9)


def test_cara_utility_matches_solve_egm_with_custom_marginal_utility():
    theta = 0.7
    sol = household("-exp(-theta*c)/theta", theta=theta).solve("egm", tol=1e-10)
    ref = reference(sol, u_prime=lambda c: np.exp(-theta * c),
                    u_prime_inv=lambda x: -np.log(x) / theta)
    np.testing.assert_allclose(sol.raw.c, ref.c, rtol=0, atol=1e-9)


def test_value_function_solves_the_policy_bellman_equation():
    sol = household(gamma=2.0).solve("egm", tol=1e-10)
    a, ap, P = A_GRID, sol.raw.aprime, sol.P
    EV = np.stack([np.interp(ap[:, z], a, sol.V @ P[z]) for z in range(P.shape[0])], axis=1)
    c = sol.raw.c
    u = (c ** -1.0 - 1.0) / -1.0
    np.testing.assert_allclose(sol.V, u + BETA * EV, rtol=0, atol=1e-9)


def test_value_function_approaches_fine_grid_vfi():
    m = household(gamma=2.0)
    m._states = [("a", np.linspace(1e-4, 80.0, 600))]
    e, v = m.solve("egm", tol=1e-10), m.solve(tol=1e-10, howard=40)
    assert np.max(np.abs(e.V - v.V)) < 0.1     # 1.55 at 150 points, 0.016 at 1500


def aiyagari():
    m = dp.Model("aiyagari")
    m.parameters(beta=BETA, gamma=1.0, alpha=ALPHA, delta=DELTA)
    m.prices(r=(0.005, 1.0 / BETA - 1.0 - 0.002))
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
    m.state("a", A_GRID)
    m.local("w = (1 - alpha)*((alpha/(r + delta))^(1/(1 - alpha)))^alpha")
    m.local("c = w*exp(z) + (1 + r)*a - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(K="a", L="exp(z)")
    m.clear("K - L*(alpha/(r + delta))^(1/(1 - alpha))")
    return m


def test_aiyagari_equilibrium_matches_hand_written_egm_loop():
    sol = aiyagari().solve("egm", tol=1e-10, xtol=1e-10)
    z, P = sol.grids["z"], sol.P
    L = float(np.linalg.matrix_power(P.T, 2000)[:, 0] @ np.exp(z))

    def resid(r):
        e = solve_egm(A_GRID, z, wage(r) * np.exp(z), P, beta=BETA, r=r, gamma=1.0, tol=1e-10)
        mu = lottery_distribution(e.aprime, A_GRID, P)
        return float(np.sum(mu * A_GRID[:, None])) - L * (ALPHA / (r + DELTA)) ** (1 / (1 - ALPHA))

    r_ref = brentq(resid, 0.005, 1.0 / BETA - 1.0 - 0.002, xtol=1e-10)
    assert sol.prices["r"] == pytest.approx(r_ref, abs=1e-8)
    assert sol.equilibrium.residual == pytest.approx(0.0, abs=1e-4)
    assert sol.aggregates["L"] == pytest.approx(L, abs=1e-10)
    vfi = aiyagari().solve(tol=1e-9, howard=40)
    assert sol.prices["r"] == pytest.approx(vfi.prices["r"], abs=1e-3)


def test_huggett_with_negative_borrowing_limit():
    m = dp.Model("huggett")
    m.parameters(beta=BETA, gamma=1.5)
    m.prices(r=(-0.5, 1.0 / BETA - 1.0 - 0.002))
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=7))
    m.state("b", np.linspace(-2.0, 24.0, 200))
    m.local("c = (1 + r)*b + exp(z) - b(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(B="b")
    m.clear("B")
    e = m.solve("egm", tol=1e-10, xtol=1e-10)
    v = m.solve(tol=1e-9, howard=40)
    assert abs(e.aggregates["B"]) < 1e-6
    assert e.prices["r"] == pytest.approx(v.prices["r"], abs=2e-3)


@pytest.mark.parametrize("change, match", [
    (lambda m: m.state("k", np.linspace(0.1, 1.0, 5)), "exactly one state"),
    (lambda m: m.local("c = w*exp(z) + (1 + r)*a - a(+1) - 0.01*a(+1)^2"), "not to depend on a"),
    (lambda m: m.local("c = w*exp(z) + (1 + r)*a + 0.01*a^2 - a(+1)"), "linear in a"),
    (lambda m: m.subject_to("a(+1) >= 1"), "exactly the constraint"),
    (lambda m: m.reward("crra(c, gamma) + 0.1*a"), "consumption as a local"),
    (lambda m: m.reward("crra(c, gamma)*exp(z)"), "otherwise on parameters"),
    (lambda m: m.reward("c"), "strictly concave"),
])
def test_models_outside_egm_are_rejected(change, match):
    m = household()
    if "local" in getattr(change, "__code__").co_names:
        m._locals = m._locals[:1]
    change(m)
    with pytest.raises(ModelSpecError, match=match):
        m.solve("egm")


# ---------------------------------------------------------------- life cycle
def two_period(survival=None, terminal=None, T=2):
    m = dp.Model("two period")
    m.parameters(beta=BETA, r=0.03, y=1.0)
    m.state("a", np.linspace(0.0, 20.0, 201))
    m.local("c = (1 + r)*a + y - a(+1)")
    m.reward("log(c)")
    m.subject_to("c > 0")
    m.horizon(T, survival=survival, terminal=terminal)
    return m


@pytest.mark.parametrize("s0", [1.0, 0.5])
def test_two_period_log_utility_is_exact(s0):
    sol = two_period(survival=[s0, 1.0]).solve("egm")
    R, y, b = 1.03, 1.0, BETA * s0
    a = sol.grids["a"]
    c = sol.policy("c")[:, :, 0]
    np.testing.assert_allclose(c[1], R * a + y, rtol=0, atol=1e-12)
    c0 = np.minimum((R * a + y + y / R) / (1 + b), R * a + y)     # kink where a' hits 0
    np.testing.assert_allclose(c[0], c0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sol.V[1, :, 0], np.log(R * a + y), rtol=0, atol=1e-12)


def test_terminal_value_enters_through_its_slope():
    a = np.linspace(0.0, 20.0, 2001)
    k, R, y = 3.0, 1.03, 1.0
    m = two_period(terminal=(k * np.log(a + 1.0))[:, None], T=1)
    m._states = [("a", a)]
    sol = m.solve("egm")
    c_exact = np.minimum((R * a + y + 1.0) / (1 + BETA * k), R * a + y)
    np.testing.assert_allclose(sol.policy("c")[0, :, 0], c_exact, rtol=0, atol=1e-4)


def life_cycle(n_a=120):
    T = 40
    ages = np.arange(T)
    m = dp.Model("life cycle")
    m.parameters(beta=0.96, gamma=2.0, r=0.04, kappa=np.exp(0.1 * ages - 0.0025 * ages ** 2))
    m.exogenous("z", dp.AR1(rho=0.95, sigma=0.15, n=5))
    m.state("a", np.linspace(0.0, 40.0, n_a))
    m.local("c = (1 + r)*a + kappa*exp(z) - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.horizon(T)
    return m


def test_life_cycle_egm_is_closer_to_fine_vfi_than_coarse_vfi():
    e, v = life_cycle().solve("egm"), life_cycle().solve()
    fine = life_cycle(n_a=119 * 10 + 1).solve()          # contains the coarse nodes
    V_ref = fine.V[:, ::10, :]
    assert np.median(np.abs(e.V - V_ref)) < 0.6 * np.median(np.abs(v.V - V_ref))
    assert np.max(np.abs(e.mean("a") - fine.mean("a"))) < 0.03
    assert np.max(np.abs(v.mean("a") - fine.mean("a"))) > 0.1
    np.testing.assert_allclose(e.distribution.sum(axis=(1, 2)), 1.0, atol=1e-12)
    assert e.method == "egm" and e.V.shape == (40, 120, 5)


def test_life_cycle_age_varying_return_is_read_per_age():
    T = 3
    m = dp.Model("varying r")
    m.parameters(beta=BETA, r=np.array([0.0, 0.05, 0.10]), y=1.0)
    m.state("a", np.linspace(0.0, 10.0, 101))
    m.local("c = (1 + r)*a + y - a(+1)")
    m.reward("log(c)")
    m.subject_to("c > 0")
    m.horizon(T)
    sol = m.solve("egm")
    np.testing.assert_allclose(sol.raw.R, [1.0, 1.05, 1.10], atol=1e-12)
    # age 1 is a two-period problem with R1 = 1.05 today and R2 = 1.10 tomorrow
    a = sol.grids["a"]
    R1, R2 = 1.05, 1.10
    c1 = np.minimum((R1 * a + 1.0 + 1.0 / R2) / (1 + BETA), R1 * a + 1.0)
    np.testing.assert_allclose(sol.policy("c")[1, :, 0], c1, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- discrete choice (DC-EGM)
def labour(n_a=100, a_max=60.0, T=None, discrete=True, chi=0.5):
    m = dp.Model("labour")
    m.parameters(beta=0.95, gamma=2.0, r=0.03, chi=chi, b=0.3)
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
    m.state("a", np.linspace(0.0, a_max, n_a))
    if discrete:
        m.discrete("h", [0.0, 1.0])
        m.local("c = (1 + r)*a + h*exp(z) + (1 - h)*b - a(+1)")
        m.reward("crra(c, gamma) - chi*h")
    else:
        m.local("c = (1 + r)*a + exp(z) - a(+1)")
        m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(H="h" if discrete else "1", A="a")
    if T:
        m.horizon(T)
    return m


def test_a_dominated_option_reproduces_plain_egm():
    # working pays the same as not working (b = 1 = exp(0) is not used: income is
    # exp(z) either way) but costs chi, so h = 0 everywhere and DC-EGM is EGM
    m = labour(n_a=200, a_max=200.0)
    m._locals = []
    m.local("c = (1 + r)*a + exp(z) - a(+1)")
    dc = m.solve("egm", tol=1e-10)
    plain = labour(n_a=200, a_max=200.0, discrete=False).solve("egm", tol=1e-10)
    assert dc.raw.policy_d.max() == 0
    np.testing.assert_allclose(dc.raw.c, plain.raw.c, rtol=0, atol=1e-8)
    np.testing.assert_allclose(dc.V, plain.V, rtol=0, atol=1e-7)


def test_dc_egm_infinite_horizon_agrees_with_vfi_on_a_fine_grid():
    e = labour(n_a=401).solve("egm", tol=1e-8)
    v = labour(n_a=401).solve(tol=1e-8)
    assert 0.2 < e.aggregates["H"] < 0.95
    assert e.aggregates["H"] == pytest.approx(v.aggregates["H"], abs=0.02)   # mass sits on nodes
    # VFI itself moves A by 0.2 between 401 and 793 points; both approach about 8.0
    assert e.aggregates["A"] == pytest.approx(v.aggregates["A"], abs=0.3)
    assert np.median(np.abs(e.V - v.V)) < 0.02


def test_dc_egm_life_cycle_is_closer_to_fine_vfi_than_coarse_vfi():
    T, k = 30, 8
    e, v = labour(T=T).solve("egm"), labour(T=T).solve()
    fine = labour(n_a=99 * k + 1, T=T).solve()            # contains the coarse nodes
    V_ref = fine.V[:, ::k, :]
    assert np.median(np.abs(e.V - V_ref)) < 0.5 * np.median(np.abs(v.V - V_ref))
    e_fine = labour(n_a=99 * k + 1, T=T).solve("egm")
    np.testing.assert_allclose(e_fine.mean("h"), fine.mean("h"), atol=0.02)
    np.testing.assert_allclose(e_fine.mean("a"), fine.mean("a"), atol=0.05)
    assert e.raw.policy_d.shape == (T, 100, 5)


def test_return_that_depends_on_the_discrete_choice_is_rejected():
    m = labour()
    m._locals = []
    m.local("c = (1 + r + 0.01*h)*a + h*exp(z) + (1 - h)*b - a(+1)")
    with pytest.raises(ModelSpecError, match="not to depend on the discrete choice"):
        m.solve("egm")


# ---------------------------------------------------------------- taste shocks
def retire(T, sigma, n_a=2001, terminal=None):
    m = dp.Model("work or not")
    m.parameters(beta=BETA, r=0.03, chi=0.4, sig=sigma)
    m.state("a", np.linspace(0.0, 20.0, n_a))
    m.discrete("h", [0.0, 1.0])
    m.local("c = (1 + r)*a + 0.3 + 0.9*h - a(+1)")
    m.reward("log(c) - chi*h")
    m.subject_to("c > 0")
    m.taste_shocks("sig")
    m.horizon(T, terminal=terminal)
    return m


def logsum(v, s):
    top = np.max(v, axis=0)
    return top + s * np.log(np.sum(np.exp((v - top) / s), axis=0))


def test_taste_shocks_last_period_is_the_closed_form_logit():
    s = 0.2
    sol = retire(1, s, n_a=11).solve("egm")
    a = sol.grids["a"]
    v = np.stack([np.log(1.03 * a + 0.3), np.log(1.03 * a + 1.2) - 0.4])
    np.testing.assert_allclose(sol.V[0, :, 0], logsum(v, s), rtol=0, atol=1e-12)
    p1 = 1.0 / (1.0 + np.exp((v[0] - v[1]) / s))
    np.testing.assert_allclose(sol.raw.choice_prob[0, 1, :, 0], p1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sol.policy("h")[0, :, 0], p1, rtol=0, atol=1e-12)


def test_taste_shocks_two_periods_match_brute_force():
    s = 0.2
    sol = retire(2, s).solve("egm")
    a = sol.grids["a"]
    R, beta = 1.03, BETA

    def V1(x):
        return logsum(np.stack([np.log(R * x + 0.3), np.log(R * x + 1.2) - 0.4]), s)

    ap = np.linspace(0.0, 20.0, 200_001)
    v0 = []
    for y, cost in ((0.3, 0.0), (1.2, 0.4)):
        coh = R * a[:, None] + y
        with np.errstate(all="ignore"):
            obj = np.where(coh - ap[None, :] > 0, np.log(coh - ap[None, :]), -np.inf) - cost + beta * V1(ap)[None, :]
        v0.append(obj.max(axis=1))
    np.testing.assert_allclose(sol.V[0, :, 0], logsum(np.stack(v0), s), rtol=0, atol=1e-5)


def test_vanishing_taste_shocks_approach_the_deterministic_choice():
    m0 = retire(5, 0.0, n_a=401)
    m0._taste = None
    det = m0.solve("egm")
    near = retire(5, 1e-4, n_a=401).solve("egm")
    assert np.max(np.abs(near.V - det.V)) < 1e-3
    assert near.raw.choice_prob is not None and det.raw.choice_prob is None


def test_taste_shocks_infinite_horizon_distribution_and_means():
    m = labour(n_a=150)
    m.parameters(sig=0.05)
    m.taste_shocks("sig")
    sol = m.solve("egm", tol=1e-9)
    probs = sol.raw.choice_prob
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-12)
    assert sol.distribution.sum() == pytest.approx(1.0, abs=1e-10)
    assert sol.aggregates["H"] == pytest.approx(float(np.sum(sol.distribution * probs[1])), abs=1e-12)
    det = labour(n_a=150).solve("egm", tol=1e-9)
    assert np.all(sol.V >= det.V - 1e-8)          # the log-sum is at least the max
    assert 0.0 < sol.aggregates["H"] < 1.0


def test_taste_shocks_need_egm_and_a_discrete_choice():
    with pytest.raises(ModelSpecError, match="method='egm' only"):
        retire(2, 0.2, n_a=11).solve()
    m = household()
    m.taste_shocks(0.1)
    with pytest.raises(ModelSpecError, match="needs a discrete choice"):
        m.solve("egm")
