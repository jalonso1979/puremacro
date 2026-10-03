"""CRRA curvature of the built-in growth models of the spline and Smolyak solvers.

Oracle: the Euler equation of the neoclassical growth model with CRRA(g)
utility and full depreciation, written out here by hand,

    1 = beta (c / c')^g * alpha_m Y' / k'_m,

evaluated off the collocation nodes at the solver's policy. Up to 4.4.0
``puremacro.vfi.splines`` and ``puremacro.vfi.smolyak`` read only
``params["sigma"]`` (the ``gamma`` spelling of ``CollocationProblem`` and the
discrete engines silently gave log utility), the spline Bellman solver used
log utility whatever the curvature, and the Smolyak Bellman solver iterated on
the closed-form log-utility, full-depreciation policy ``k'_m = alpha_m beta Y``.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi.smolyak import solve_smolyak
from puremacro.vfi.splines import SplineCollocationProblem

ALPHA, BETA, G = 0.36, 0.96, 2.0
K_TEST = np.linspace(0.12, 0.35, 400)

ALPHAS = (0.18, 0.18)
Z2 = 1.0 / (ALPHAS[0] * BETA)
S_TEST = np.random.default_rng(0).uniform(0.85, 1.15, (500, 2))


def _euler_1d(policy, k, g):
    kp = policy(k)
    kpp = policy(kp)
    c = k ** ALPHA - kp
    cp = kp ** ALPHA - kpp
    return 1.0 - BETA * (c / cp) ** g * ALPHA * kp ** (ALPHA - 1.0)


def _euler_2d(policy, s, g):
    kp = policy(s)
    kpp = policy(kp)
    Y = Z2 * s[:, 0] ** ALPHAS[0] * s[:, 1] ** ALPHAS[1]
    Yp = Z2 * kp[:, 0] ** ALPHAS[0] * kp[:, 1] ** ALPHAS[1]
    C = Y - kp.sum(axis=1)
    Cp = Yp - kpp.sum(axis=1)
    return np.column_stack([1.0 - BETA * (C / Cp) ** g * ALPHAS[m] * Yp / kp[:, m] for m in range(2)])


def _spline(method, key):
    prob = SplineCollocationProblem(
        domain=(0.1, 0.4), n_knots=16, spline_type="cubic", bc_type="clamped", beta=BETA,
        method=method, params={"alpha": ALPHA, "delta": 1.0, "z": 1.0, key: G},
    )
    return prob.solve(backend="numpy")


@pytest.mark.parametrize("key", ["sigma", "gamma"])
def test_spline_euler_solves_the_crra_model(key):
    # 4.4.0 with gamma=2: hand-written Euler residual 0.151 (the log-utility policy).
    sol = _spline("euler", key)
    assert sol.converged
    assert np.max(np.abs(_euler_1d(sol.policy, K_TEST, G))) < 1e-4
    # The solution's own diagnostics use the same curvature.
    np.testing.assert_allclose(sol.euler_residual(K_TEST), _euler_1d(sol.policy, K_TEST, G), atol=1e-6)


@pytest.mark.parametrize("key", ["sigma", "gamma"])
def test_spline_bellman_solves_the_crra_model(key):
    # 4.4.0: residual 0.151 for sigma=2 and gamma=2 (log utility in the Bellman step).
    sol = _spline("bellman", key)
    assert sol.converged
    assert np.max(np.abs(_euler_1d(sol.policy, K_TEST, G))) < 5e-3


def test_spline_marginal_value_uses_the_alias():
    sol = _spline("euler", "gamma")
    kp = sol.policy(K_TEST)
    expected = (K_TEST ** ALPHA - kp) ** (-G) * ALPHA * K_TEST ** (ALPHA - 1.0)
    np.testing.assert_allclose(sol.marginal_value(K_TEST), expected, rtol=1e-6)


def test_smolyak_euler_honours_gamma():
    # 4.4.0 with gamma=2: hand-written Euler residual 0.0345 (the log-utility policy).
    sol = solve_smolyak(domain=((0.8, 1.2), (0.8, 1.2)), mu=3,
                        params={"alphas": list(ALPHAS), "z": Z2, "beta": BETA, "gamma": G})
    assert np.max(np.abs(_euler_2d(sol.policy, S_TEST, G))) < 1e-5
    np.testing.assert_allclose(sol.euler_residual(S_TEST), _euler_2d(sol.policy, S_TEST, G), atol=1e-8)


@pytest.mark.parametrize("extra", [{"sigma": G}, {"gamma": G}, {"delta": 0.1}])
def test_smolyak_bellman_refuses_models_it_does_not_solve(extra):
    # 4.4.0 returned converged=True with the log-utility, full-depreciation policy
    # (hand-written Euler residual 0.0345 at sigma=2).
    params = {"alphas": list(ALPHAS), "z": Z2, "beta": BETA, **extra}
    with pytest.raises(NotImplementedError, match="method='euler'"):
        solve_smolyak(domain=((0.8, 1.2), (0.8, 1.2)), mu=2, method="bellman", params=params)


def test_smolyak_bellman_log_full_depreciation_is_unchanged():
    sol = solve_smolyak(domain=((0.8, 1.2), (0.8, 1.2)), mu=2, method="bellman",
                        params={"alphas": list(ALPHAS), "z": Z2, "beta": BETA})
    assert sol.converged
    assert np.max(np.abs(_euler_2d(sol.policy, S_TEST, 1.0))) < 1e-4
