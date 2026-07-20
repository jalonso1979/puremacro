from __future__ import annotations

import numpy as np

from puremacro.vfi import tauchen
from puremacro.vfi.finite_horizon import FiniteHorizonProblem
from puremacro.vfi.olg import (
    OLGEquilibrium,
    olg_aggregate,
    olg_stationary_equilibrium,
    stationary_age_weights,
)


def test_olg_endogenous_labor_clears_both_margins():
    # Life-cycle OLG with endogenous hours d in {0, .5, 1} (0 = non-participation):
    # work (hump productivity) then retire (pension floor). Clears in
    # capital-labor-ratio space -- the standard OLG algorithm: guess KL -> firm
    # prices r(KL), w(KL) -> households choose -> require K_supply/L_supply == KL.
    # The identity LHS ranges widely while the household K/L is bounded, so a
    # crossing is guaranteed (unlike clearing on r with a fixed firm KL demand).
    alpha, delta, beta = 0.36, 0.08, 0.96
    z_grid, P = tauchen(3, 0.8, 0.2)
    a_grid = np.linspace(0.0, 60.0, 30)
    d_grid = np.array([0.0, 0.5, 1.0])               # 0 = non-participation
    J, R = 30, 24                                     # work 0..23, retire 24..29
    ages = np.arange(J)
    kappa = np.where(ages < R, np.exp(0.06 * ages - 0.0015 * ages ** 2), 0.0)
    pension = np.where(ages >= R, 0.4, 0.0)           # floor so retirees can consume
    chi, nu, fixed_cost = 1.0, 1.0, 0.05
    weights = stationary_age_weights(J)

    def r_of_KL(KL):
        return alpha * KL ** (alpha - 1.0) - delta

    def w_of_KL(KL):
        return (1.0 - alpha) * KL ** alpha

    def build(KL):                                    # the "price" is the K/L ratio
        r, w = r_of_KL(KL), w_of_KL(KL)

        def rf(d, ap, a, z, age, xp=np):
            income = w * kappa[age] * np.exp(z) * d + pension[age]
            c = (1.0 + r) * a + income - ap
            disutil = chi * d ** (1.0 + 1.0 / nu) / (1.0 + 1.0 / nu) + fixed_cost * (d > 0.0)
            u = xp.where(c > 0.0, 1.0 - 1.0 / xp.maximum(c, 1e-12), -np.inf)
            return u - disutil
        return FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                                    beta=beta, horizon=J, d_grid=d_grid)

    def assets(d, ap, a, z, age, xp=np):
        return a + 0.0 * ap

    def labor(d, ap, a, z, age, xp=np):
        return kappa[age] * np.exp(z) * d            # effective labor units

    def participation(d, ap, a, z, age, xp=np):
        return (d > 0.0) + 0.0 * a

    def resid(KL, sol, lcd, w_age, prob):
        K = olg_aggregate(assets, lcd, w_age, sol, a_grid, z_grid, d_grid=d_grid)
        L = olg_aggregate(labor, lcd, w_age, sol, a_grid, z_grid, d_grid=d_grid)
        return K / L - KL                            # household K/L must equal the firm's KL

    eq = olg_stationary_equilibrium(build, resid, (0.7, 2.0), age_weights=weights)
    assert isinstance(eq, OLGEquilibrium)
    assert eq.price > 0.0                            # equilibrium capital-labor ratio
    assert abs(eq.residual) < 0.05                   # capital market clears (grid-coarse)

    L = olg_aggregate(labor, eq.life_cycle_dist, weights, eq.solution, a_grid, z_grid,
                      d_grid=d_grid)
    part = olg_aggregate(participation, eq.life_cycle_dist, weights, eq.solution,
                         a_grid, z_grid, d_grid=d_grid)
    assert L > 0.0
    assert 0.0 < part < 1.0                          # BOTH margins: some h=0, some h>0


def test_olg_result_fields_and_export():
    from puremacro.vfi import OLGEquilibrium as E
    from puremacro.vfi import olg_stationary_equilibrium as fn

    assert fn is olg_stationary_equilibrium and E is OLGEquilibrium
