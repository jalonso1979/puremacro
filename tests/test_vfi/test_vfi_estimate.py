from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import VFIProblem
from puremacro.vfi.estimate import EstimationResult, estimate_method_of_moments


def test_identity_recovery():
    # moments_at(theta) = theta ; data = theta_true -> recover theta_true
    theta_true = np.array([0.3, -1.2])

    def moments_at(theta):
        return np.asarray(theta, dtype=float)

    res = estimate_method_of_moments(moments_at, theta_true, theta0=np.array([0.0, 0.0]))
    assert isinstance(res, EstimationResult)
    np.testing.assert_allclose(res.theta, theta_true, atol=1e-4)
    assert res.objective < 1e-8
    np.testing.assert_allclose(res.moments, theta_true, atol=1e-4)


def test_nonlinear_recovery():
    # moments_at(theta) = [theta0^2, theta0*theta1]; pick data with a known root
    # at theta=(2, 3): moments=(4, 6).
    def moments_at(theta):
        t0, t1 = float(theta[0]), float(theta[1])
        return np.array([t0 ** 2, t0 * t1])

    data = np.array([4.0, 6.0])
    res = estimate_method_of_moments(moments_at, data, theta0=np.array([1.5, 1.0]),
                                     bounds=[(0.1, 5.0), (-5.0, 5.0)])
    np.testing.assert_allclose(res.moments, data, atol=1e-3)
    np.testing.assert_allclose(res.theta, [2.0, 3.0], atol=1e-2)


def test_weight_matrix_used():
    # over-identified (2 moments, 1 param). With a weight emphasizing moment 0,
    # the estimate fits moment 0 better than with the identity weight.
    def moments_at(theta):
        t = float(theta[0])
        return np.array([t, 2.0 * t])          # the two moments are t and 2t

    data = np.array([1.0, 1.0])                # inconsistent: t=1 vs t=0.5
    res_id = estimate_method_of_moments(moments_at, data, theta0=np.array([0.5]))
    W = np.diag([100.0, 1.0])                   # weight moment 0 heavily
    res_w = estimate_method_of_moments(moments_at, data, theta0=np.array([0.5]), weight=W)
    # the W-estimate pulls moment 0 (=t) closer to its target (1.0) than identity
    assert abs(res_w.moments[0] - 1.0) < abs(res_id.moments[0] - 1.0)


def test_model_based_recovery_beta_from_mean_assets():
    # SMM using the full solve -> stationary distribution -> aggregate stack:
    # mean household assets is increasing in beta (precautionary savings motive),
    # so beta is recovered from a target mean-assets moment (partial equilibrium,
    # fixed prices). Uses n=7, sigma=0.4 to create sufficient income uncertainty
    # for a non-trivial precautionary savings demand across beta in [0.88, 0.97].
    from puremacro.vfi import tauchen

    z_grid, P = tauchen(n=7, rho=0.9, sigma=0.4)
    a_grid = np.linspace(1e-3, 60.0, 100)

    def moments_at(theta):
        beta = float(theta[0])

        def rf(ap, a, z, xp=np):
            c = 1.0 * np.exp(z) + 1.03 * a - ap
            return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -1e10)

        prob = VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                          beta=beta, options=dict(tol=1e-9, n_howard=40))
        sol = prob.solve("numpy")
        mu = prob.stationary_distribution(sol)
        return np.array([float(np.sum(mu * a_grid[:, None]))])   # mean assets

    beta_true = 0.95
    data = moments_at([beta_true])
    # Use Nelder-Mead explicitly: the moment function is computed via discrete VFI,
    # so finite-difference gradients (needed by L-BFGS-B) are zero at eps=1e-8.
    res = estimate_method_of_moments(moments_at, data, theta0=np.array([0.90]),
                                     method="Nelder-Mead")
    assert abs(res.theta[0] - beta_true) < 5e-3
    assert res.objective < 1e-4


def test_validation():
    def moments_at(theta):
        return np.asarray(theta, dtype=float)

    with pytest.raises(ValueError, match="weight"):
        estimate_method_of_moments(moments_at, np.array([1.0, 2.0]),
                                   theta0=np.array([0.0, 0.0]),
                                   weight=np.eye(3))  # 3x3 vs 2 moments


def test_estimate_exported():
    from puremacro.vfi import estimate_method_of_moments as fn

    assert fn is estimate_method_of_moments
