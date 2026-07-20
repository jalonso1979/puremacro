"""Known-answer economic models: the engine must reproduce analytic solutions.

Both have INTERIOR closed-form policies (bounded away from the grid floor, so
the infeasibility penalty never contaminates V), and we assert within ~grid
spacing on the interior band where the optimum is not pinned to a boundary. The
stochastic Brock-Mirman variant additionally exercises the exogenous Markov
state. (Cake-eating is intentionally NOT used: its continuous closed form
a'=beta*a does not transfer to a bounded grid, where the floor's infeasibility
penalty makes the discrete optimum save-almost-everything instead.)
"""
from __future__ import annotations

import numpy as np

from puremacro.vfi import VFIProblem, tauchen


def test_stochastic_brock_mirman_closed_form_policy():
    # y = exp(z)*k^alpha, full depreciation, log utility.  With log utility the
    # optimal savings rate is the constant alpha*beta regardless of the shock,
    # so the closed-form policy is k' = alpha*beta*exp(z)*k^alpha (interior; the
    # grid floor never binds).  Also exercises the Markov z state.
    alpha, beta = 0.3, 0.95
    z_grid, P = tauchen(n=5, rho=0.7, sigma=0.05)
    kss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    k_grid = np.linspace(0.5 * kss, 2.5 * kss, 700)

    def rf(kp, k, z, alpha, xp=np):
        c = xp.exp(z) * k ** alpha - kp
        return xp.where(c > 1e-12, xp.log(xp.maximum(c, 1e-12)), -1e10)

    prob = VFIProblem(a_grid=k_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                      beta=beta, params={"alpha": alpha},
                      options=dict(tol=1e-12, n_howard=60))
    sol = prob.solve("numpy")
    spacing = k_grid[1] - k_grid[0]
    lo, hi = k_grid[3], k_grid[-4]  # interior band: avoid grid-edge clipping
    checked = 0
    for iz, zval in enumerate(z_grid):
        kp = k_grid[sol.policy_aprime[:, iz]]
        closed = alpha * beta * np.exp(zval) * k_grid ** alpha
        mask = (closed > lo) & (closed < hi)
        if mask.any():
            np.testing.assert_allclose(kp[mask], closed[mask], atol=3.0 * spacing)
            checked += int(mask.sum())
    assert checked > 100  # confirm we verified a meaningful number of (k,z) points


def test_brock_mirman_closed_form_policy():
    # f(k)=k^alpha, full depreciation, log utility.  Closed form: k' = alpha*beta*k^alpha.
    alpha, beta = 0.3, 0.95
    n_k = 500
    kss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    k_grid = np.linspace(0.5 * kss, 1.5 * kss, n_k)
    z = np.array([0.0])
    P = np.array([[1.0]])

    def rf(kp, k, z, alpha, xp=np):
        c = k ** alpha - kp
        return xp.where(c > 1e-12, xp.log(xp.maximum(c, 1e-12)), -1e10)

    prob = VFIProblem(a_grid=k_grid, z_grid=z, P_z=P, return_fn=rf, beta=beta,
                      params={"alpha": alpha}, options=dict(tol=1e-12, n_howard=60))
    sol = prob.solve("numpy")
    kp = k_grid[sol.policy_aprime[:, 0]]
    closed = alpha * beta * k_grid ** alpha
    spacing = k_grid[1] - k_grid[0]
    mask = (k_grid > 0.6 * kss) & (k_grid < 1.4 * kss)
    np.testing.assert_allclose(kp[mask], closed[mask], atol=2.0 * spacing)
