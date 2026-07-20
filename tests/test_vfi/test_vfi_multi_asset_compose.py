"""Multiple endogenous states compose with the higher-level workflows (PType, GE)
for free -- those reuse solve / stationary_distribution / aggregate, all of which
are multi-asset aware. These tests pin that composability."""
from __future__ import annotations

import numpy as np

from puremacro.vfi import (
    VFIProblem,
    markov_stationary,
    solve_permanent_types,
    stationary_equilibrium,
    tauchen,
)


def _total(m_p, k_p, m, k, z, *params, xp=np):
    return m + k


def test_permanent_types_with_multi_asset():
    z_grid, P = tauchen(3, 0.8, 0.2)
    m_grid = np.linspace(0.0, 10.0, 11)
    k_grid = np.linspace(0.0, 10.0, 11)
    betas = [0.94, 0.97]

    def build(t):
        def rf(m_p, k_p, m, k, z, xp=np):
            c = 1.02 * m + 1.04 * k + np.exp(z) - m_p - k_p
            return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)
        return VFIProblem(a_grid=[m_grid, k_grid], z_grid=z_grid, P_z=P, return_fn=rf,
                          beta=betas[t], options=dict(tol=1e-9, n_howard=30))

    pt = solve_permanent_types(build, [0.5, 0.5])
    assert pt.distributions[0].shape == (121, 3)          # joint product distribution
    per = pt.type_aggregates(_total)
    K = pt.aggregate(_total)
    assert per.shape == (2,)
    assert per[1] > per[0]                                 # patient type holds more total wealth
    assert per[0] < K < per[1]


def test_general_equilibrium_with_multi_asset():
    # a two-asset Aiyagari where both assets are capital (total m+k = K); confirms
    # the GE root-find composes with the multi-asset solve / dist / aggregate.
    alpha, delta, beta = 0.36, 0.08, 0.96
    z_grid, P = tauchen(5, 0.9, 0.2)
    L = float(markov_stationary(P) @ np.exp(z_grid))
    m_grid = np.linspace(0.0, 30.0, 16)
    k_grid = np.linspace(0.0, 30.0, 16)

    def KL(r):
        return (alpha / (r + delta)) ** (1.0 / (1.0 - alpha))

    def wage(r):
        return (1.0 - alpha) * KL(r) ** alpha

    def build(r):
        w = wage(r)

        def rf(m_p, k_p, m, k, z, r_, w_, xp=np):
            c = (1.0 + r_) * (m + k) + w_ * np.exp(z) - m_p - k_p
            return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)
        return VFIProblem(a_grid=[m_grid, k_grid], z_grid=z_grid, P_z=P, return_fn=rf,
                          beta=beta, params={"r": r, "w": w},
                          options=dict(tol=1e-9, n_howard=40))

    def resid(r, sol, mu, prob):
        K_supply = prob.aggregate(sol, mu, _total)
        return K_supply - L * KL(r)

    eq = stationary_equilibrium(build, resid, (0.005, 1.0 / beta - 1.0 - 0.002))
    assert 0.0 < eq.price < 1.0 / beta - 1.0                # sane interest rate
    assert abs(eq.residual) < 1.0                           # capital market clears (grid-coarse)
