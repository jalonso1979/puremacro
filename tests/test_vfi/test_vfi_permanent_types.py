from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem, tauchen
from puremacro.vfi.permanent_types import (
    PermanentTypesSolution,
    solve_permanent_types,
)


def _assets(aprime, a, z, xp=np):
    return a + xp.zeros_like(aprime)          # state-based: mean assets


def _make_problem(beta):
    z_grid, P = tauchen(n=5, rho=0.9, sigma=0.2)
    a_grid = np.linspace(0.0, 60.0, 150)

    def rf(ap, a, z, xp=np):
        c = 1.03 * a + np.exp(z) - ap
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)
    return VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                      beta=beta, options=dict(tol=1e-9, n_howard=30))


def test_single_type_matches_plain():
    prob = _make_problem(0.95)
    sol = prob.solve("numpy")
    mu = prob.stationary_distribution(sol)
    plain_K = prob.aggregate(sol, mu, _assets)

    pt = solve_permanent_types(lambda t: _make_problem(0.95), weights=[1.0])
    assert isinstance(pt, PermanentTypesSolution)
    assert pt.aggregate(_assets) == pytest.approx(plain_K, rel=1e-9)
    np.testing.assert_allclose(pt.distributions[0], mu, atol=1e-12)


def test_two_identical_types_equal_single():
    pt = solve_permanent_types(lambda t: _make_problem(0.95), weights=[0.5, 0.5])
    K = pt.aggregate(_assets)
    per = pt.type_aggregates(_assets)
    assert per.shape == (2,)
    np.testing.assert_allclose(per[0], per[1], atol=1e-12)         # identical types
    assert K == pytest.approx(per[0], rel=1e-12)                    # weighted == single
    # mixture distribution equals the single-type distribution
    np.testing.assert_allclose(pt.mixture_distribution(), pt.distributions[0], atol=1e-12)
    np.testing.assert_allclose(pt.mixture_distribution().sum(), 1.0, atol=1e-12)


def test_higher_beta_saves_more():
    betas = [0.90, 0.97]
    pt = solve_permanent_types(lambda t: _make_problem(betas[t]), weights=[0.5, 0.5])
    per = pt.type_aggregates(_assets)
    assert per[1] > per[0]                                          # patient type holds more assets
    K = pt.aggregate(_assets)
    assert per[0] < K < per[1]                                      # population mean is between


def test_weights_validation():
    with pytest.raises(ValueError, match="sum to 1"):
        solve_permanent_types(lambda t: _make_problem(0.95), weights=[0.5, 0.4])
    with pytest.raises(ValueError, match="nonnegative"):
        solve_permanent_types(lambda t: _make_problem(0.95), weights=[1.5, -0.5])


def test_mixture_requires_matching_grids():
    def build(t):
        # type 1 has a different (longer) a-grid -> mixture is undefined
        n_a = 150 if t == 0 else 120
        z_grid, P = tauchen(n=5, rho=0.9, sigma=0.2)
        a_grid = np.linspace(0.0, 60.0, n_a)

        def rf(ap, a, z, xp=np):
            c = 1.03 * a + np.exp(z) - ap
            return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)
        return VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                          beta=0.95, options=dict(tol=1e-9, n_howard=30))

    pt = solve_permanent_types(build, weights=[0.5, 0.5])
    # type_aggregates still works (each uses its own grid)
    assert pt.type_aggregates(_assets).shape == (2,)
    with pytest.raises(ValueError, match="same shape"):
        pt.mixture_distribution()


def test_build_problem_must_return_vfiproblem():
    with pytest.raises(TypeError, match="VFIProblem"):
        solve_permanent_types(lambda t: "not a problem", weights=[1.0])


def test_ptype_exported():
    from puremacro.vfi import PermanentTypesSolution as S
    from puremacro.vfi import solve_permanent_types as fn

    assert fn is solve_permanent_types
    assert S is PermanentTypesSolution
