from __future__ import annotations

import numpy as np

from puremacro.vfi import tauchen
from puremacro.vfi.finite_horizon import FiniteHorizonProblem, life_cycle_distribution
from puremacro.vfi.olg import olg_aggregate, stationary_age_weights


def test_age_weights_uniform_no_mortality():
    w = stationary_age_weights(5)
    np.testing.assert_allclose(w, np.full(5, 0.2), atol=1e-12)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)


def test_age_weights_decline_with_mortality_and_growth():
    surv = np.full(6, 0.95)
    w = stationary_age_weights(6, survival=surv)
    assert np.all(np.diff(w) < 0)                       # mortality -> fewer old
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
    wg = stationary_age_weights(6, pop_growth=0.02)
    assert np.all(np.diff(wg) < 0)                      # growth -> fewer old
    np.testing.assert_allclose(wg.sum(), 1.0, atol=1e-12)


def _labor_fh():
    # a small life-cycle problem WITH a labor decision d (hours, incl. 0)
    z_grid, P = tauchen(3, 0.7, 0.2)
    a_grid = np.linspace(0.0, 10.0, 20)
    d_grid = np.array([0.0, 0.5, 1.0])
    J = 6

    def rf(d, ap, a, z, age, xp=np):
        c = 1.03 * a + np.exp(z) * d - ap
        u = xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)
        return u - 0.5 * d ** 2
    sol = FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                               beta=0.96, horizon=J, d_grid=d_grid).solve("numpy")
    lcd = life_cycle_distribution(sol, P)
    return sol, lcd, a_grid, z_grid, d_grid, J


def test_olg_aggregate_assets_matches_direct():
    sol, lcd, a_grid, z_grid, d_grid, J = _labor_fh()
    w = stationary_age_weights(J)

    def assets(d, ap, a, z, age, xp=np):
        return a + 0.0 * ap

    agg = olg_aggregate(assets, lcd, w, sol, a_grid, z_grid, d_grid=d_grid)
    direct = sum(float(w[j]) * float(np.sum(lcd[j] * a_grid[:, None])) for j in range(J))
    assert agg == direct


def test_olg_aggregate_labor_uses_policy_d():
    sol, lcd, a_grid, z_grid, d_grid, J = _labor_fh()
    w = stationary_age_weights(J)

    def labor(d, ap, a, z, age, xp=np):
        return np.exp(z) * d                          # effective labor (uses chosen hours d)

    L = olg_aggregate(labor, lcd, w, sol, a_grid, z_grid, d_grid=d_grid)
    direct = 0.0
    for j in range(J):
        hours = d_grid[sol.policy_d[j]]               # (n_a, n_z)
        direct += float(w[j]) * float(np.sum(lcd[j] * (np.exp(z_grid)[None, :] * hours)))
    assert L == direct
    assert L > 0.0


def test_olg_aggregate_exported():
    from puremacro.vfi import olg_aggregate as a
    from puremacro.vfi import stationary_age_weights as b

    assert a is olg_aggregate and b is stationary_age_weights
