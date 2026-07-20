from __future__ import annotations

import numpy as np

from puremacro.vfi import VFIProblem, stationary_distribution, tauchen
from puremacro.vfi.aggregate import aggregate, evaluate_on_grid


def _two_asset_solution():
    z_grid, P = tauchen(3, 0.8, 0.2)
    beta, r_m, r_k = 0.96, 0.01, 0.05
    m_grid = np.linspace(0.0, 12.0, 13)
    k_grid = np.linspace(0.0, 12.0, 13)

    def rf(m_p, k_p, m, k, z, xp=np):
        c = (1.0 + r_m) * m + (1.0 + r_k) * k + np.exp(z) - m_p - k_p
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)

    prob = VFIProblem(a_grid=[m_grid, k_grid], z_grid=z_grid, P_z=P, return_fn=rf,
                      beta=beta, options=dict(tol=1e-9, n_howard=30))
    sol = prob.solve("numpy")
    mu = stationary_distribution(sol.policy_aprime, P)
    return prob, sol, mu, m_grid, k_grid


def test_evaluate_single_asset_unchanged():
    z_grid, P = tauchen(3, 0.8, 0.2)
    a_grid = np.linspace(0.0, 10.0, 12)
    pol = np.tile(np.arange(12)[:, None], (1, 3))

    def fn(ap, a, z, xp=np):
        return ap + 2.0 * a

    vals = evaluate_on_grid(fn, pol, a_grid, z_grid)
    expected = a_grid[pol] + 2.0 * a_grid[:, None]
    np.testing.assert_allclose(vals, expected, atol=1e-12)


def test_aggregate_two_asset_total_wealth():
    prob, sol, mu, m_grid, k_grid = _two_asset_solution()

    def total(m_p, k_p, m, k, z, *params, xp=np):
        return m + k                          # state-based total wealth

    K_total = aggregate(total, mu, sol.policy_aprime, [m_grid, k_grid],
                        prob.z_grid)
    mm, kk = np.meshgrid(m_grid, k_grid, indexing="ij")
    mk = (mm + kk).reshape(-1)
    direct = float(np.sum(mu * mk[:, None]))
    assert K_total == direct


def test_aggregate_two_asset_realized_next_assets():
    prob, sol, mu, m_grid, k_grid = _two_asset_solution()

    def saved(m_p, k_p, m, k, z, *params, xp=np):
        return m_p + k_p                      # realized next total assets

    val = aggregate(saved, mu, sol.policy_aprime, [m_grid, k_grid], prob.z_grid)
    mm, kk = np.meshgrid(m_grid, k_grid, indexing="ij")
    comp_m = mm.reshape(-1)[sol.policy_aprime]
    comp_k = kk.reshape(-1)[sol.policy_aprime]
    direct = float(np.sum(mu * (comp_m + comp_k)))
    assert val == direct


def test_problem_aggregate_two_asset():
    prob, sol, mu, m_grid, k_grid = _two_asset_solution()

    def total(m_p, k_p, m, k, z, *params, xp=np):
        return m + k

    via_method = prob.aggregate(sol, mu, total)
    via_func = aggregate(total, mu, sol.policy_aprime, [m_grid, k_grid], prob.z_grid)
    assert via_method == via_func
