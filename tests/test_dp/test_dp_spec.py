"""puremacro.dp: multi-state, discrete-choice and multi-shock compilation, and spec errors."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro import dp
from puremacro.dp import ModelSpecError
from puremacro.vfi.discretize import tauchen
from puremacro.vfi.examples import two_asset_profile
from puremacro.vfi.problem import VFIProblem


def test_two_asset_matches_example():
    ref = two_asset_profile()
    m = dp.Model("two asset")
    m.parameters(beta=0.96, gamma=2.0, rl=0.01, ri=0.05, kappa=0.05)
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.25, n=5))
    m.state("m", np.linspace(0.0, 15.0, 20))
    m.state("k", np.linspace(0.0, 15.0, 20))
    m.local("c = (1 + rl)*m + (1 + ri)*k + exp(z) - m(+1) - k(+1) - kappa*abs(k(+1) - k)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(M="m", K="k")
    sol = m.solve(tol=1e-9, howard=30)
    np.testing.assert_array_equal(sol.raw.policy_aprime, ref["solution"].policy_aprime)
    np.testing.assert_allclose(sol.V, ref["solution"].V, rtol=0, atol=1e-10)
    assert sol.aggregates["M"] == pytest.approx(ref["mean_liquid"], abs=1e-10)
    assert sol.aggregates["K"] == pytest.approx(ref["mean_illiquid"], abs=1e-10)


def test_discrete_choice_and_two_shocks_match_hand_built_problem():
    a = np.linspace(0.0, 10.0, 40)
    z1, P1 = tauchen(n=3, rho=0.8, sigma=0.2)
    z2, P2 = np.array([0.9, 1.1]), np.array([[0.7, 0.3], [0.4, 0.6]])
    h = np.array([0.0, 1.0])

    def rf(d, ap, aa, x1, x2, r, chi, xp=np):
        c = (1 + r) * aa + d * np.exp(x1) * x2 + 0.2 - ap
        u = xp.log(c) - chi * d
        return xp.where(c > 0, u, -np.inf)

    with np.errstate(all="ignore"):
        ref = VFIProblem(a_grid=a, z_grid=[z1, z2], P_z=np.kron(P1, P2), return_fn=rf, beta=0.95,
                         params={"r": 0.02, "chi": 0.3}, d_grid=h,
                         options=dict(tol=1e-9, n_howard=20)).solve()

    m = dp.Model("labour")
    m.parameters(beta=0.95, r=0.02, chi=0.3)
    m.exogenous("e", dp.AR1(rho=0.8, sigma=0.2, n=3))
    m.exogenous("q", dp.Markov(z2, P2))
    m.state("a", a)
    m.discrete("h", h)
    m.local("c = (1 + r)*a + h*exp(e)*q + 0.2 - a(+1)")
    m.reward("log(c) - chi*h")
    m.subject_to("c > 0")
    sol = m.solve(tol=1e-9)
    np.testing.assert_array_equal(sol.raw.policy_aprime, ref.policy_aprime)
    np.testing.assert_array_equal(sol.raw.policy_d, ref.policy_d)
    np.testing.assert_allclose(sol.V, ref.V, atol=1e-12)
    assert sol.policy("h").shape == (40, 6)
    np.testing.assert_array_equal(sol.policy("h"), h[ref.policy_d])
    assert sol.distribution.sum() == pytest.approx(1.0)


def test_crra_is_log_at_one():
    c = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(dp.crra(c, 1.0), np.log(c))
    np.testing.assert_allclose(dp.crra(c, 1.0 + 1e-6), np.log(c), atol=1e-5)


def _base():
    m = dp.Model("base")
    m.parameters(beta=0.95, r=0.02)
    m.exogenous("z", dp.AR1(rho=0.5, sigma=0.1, n=3))
    m.state("a", np.linspace(0.0, 5.0, 10))
    m.local("c = (1 + r)*a + exp(z) - a(+1)")
    m.reward("log(c)")
    m.subject_to("c > 0")
    return m


def test_base_model_compiles():
    sol = _base().solve()
    assert sol.V.shape == (10, 3)


@pytest.mark.parametrize("mutate, message", [
    (lambda m: m.reward("log(c) + zeta"), "unknown name 'zeta'"),
    (lambda m: m.reward("log(c) + z(+1)"), "only states take a timing index"),
    (lambda m: m.reward("log(a(+2))"), r"only a and a\(\+1\)"),
    (lambda m: m.subject_to("c"), "must be a comparison"),
    (lambda m: m.reward("c > 0"), "must be an expression"),
    (lambda m: m.reward("normcdf(c)"), "not supported"),
    (lambda m: m.parameters(kappa=np.ones(4)), "need horizon"),
    (lambda m: m.prices(r2=(0.0, 0.1)), "no clear"),
    (lambda m: m.parameters(z=1.0), "already a state"),
    (lambda m: m.discount("rho"), "scalar parameter"),
    (lambda m: m.choose("b(+1)"), "chooses every state"),
])
def test_spec_errors_name_the_problem(mutate, message):
    m = _base()
    mutate(m)
    with pytest.raises(ModelSpecError, match=message):
        m.check()


def test_missing_pieces():
    with pytest.raises(ModelSpecError, match="at least one state"):
        dp.Model().reward("1").check()
    m = dp.Model().parameters(beta=0.9).state("a", [0.0, 1.0])
    with pytest.raises(ModelSpecError, match="reward"):
        m.check()
    with pytest.raises(ModelSpecError, match="strictly increasing"):
        dp.Model().state("a", [1.0, 0.0])
    with pytest.raises(ModelSpecError, match="cannot parse"):
        _base().reward("log(c +").check()


def test_model_without_shocks_and_finite_horizon_with_survival():
    det = (dp.Model("deterministic").parameters(beta=0.9, R=1.05)
           .state("w", np.linspace(0.0, 2.0, 100))
           .local("c = 0.2 + R*w - w(+1)").reward("log(c)").subject_to("c > 0"))
    sol = det.solve()
    assert sol.V.shape == (100, 1)
    np.testing.assert_allclose(sol.policy("c"), 0.2 + 1.05 * sol.grids["w"][:, None] - sol.policy("w(+1)"))

    lc = (dp.Model("survival").parameters(beta=0.95)
          .state("a", np.linspace(0.0, 5.0, 50))
          .local("c = 1 + 1.1*a - a(+1)").reward("log(c)").subject_to("c > 0"))
    v_short = lc.horizon(10, survival=np.full(10, 0.5)).solve().V[0]
    v_long = lc.horizon(10).solve().V[0]
    assert np.all(v_short < v_long)   # mortality shortens expected life, lowering V at age 0
