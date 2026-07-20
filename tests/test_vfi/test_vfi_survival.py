from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi import tauchen
from puremacro.vfi.finite_horizon import (
    FiniteHorizonProblem,
    age_profile,
    life_cycle_distribution,
)


def _income_rf(r):
    def rf(ap, a, z, age, r_, xp=np):
        c = (1.0 + r_) * a + np.exp(z) - ap
        return xp.where(c > 0.0, xp.log(xp.maximum(c, 1e-12)), -np.inf)
    return rf


def test_survival_ones_reduces_to_no_survival():
    z_grid, P = tauchen(3, 0.8, 0.2)
    a_grid = np.linspace(0.0, 20.0, 30)
    J = 12
    rf = _income_rf(0.03)
    kw = dict(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf, beta=0.96,
              horizon=J, params={"r": 0.03})
    s0 = FiniteHorizonProblem(**kw).solve("numpy")
    s1 = FiniteHorizonProblem(**kw, survival=np.ones(J)).solve("numpy")
    np.testing.assert_allclose(s1.V, s0.V, atol=1e-12)          # exact reduction
    np.testing.assert_array_equal(s1.policy_aprime, s0.policy_aprime)


def test_mortality_reduces_saving():
    # lower survival -> effective discount beta*s < beta -> less patient -> save less
    z_grid, P = tauchen(3, 0.9, 0.2)
    a_grid = np.linspace(0.0, 30.0, 60)
    J = 30
    ages = np.arange(J)
    kappa = np.exp(0.1 * ages - 0.004 * ages ** 2)

    def rf(ap, a, z, age, xp=np):
        c = 1.03 * a + kappa[age] * np.exp(z) - ap
        return xp.where(c > 0.0, -1.0 / xp.maximum(c, 1e-12), -np.inf)

    def build(surv):
        return FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                                    beta=0.97, horizon=J, survival=surv)

    full = build(np.ones(J)).solve("numpy")
    mort = build(np.full(J, 0.95)).solve("numpy")
    a_full = age_profile(life_cycle_distribution(full, P), a_grid)
    a_mort = age_profile(life_cycle_distribution(mort, P), a_grid)
    assert a_mort.max() < a_full.max()                          # mortality lowers peak assets


def test_survival_validation():
    z_grid, P = tauchen(2, 0.5, 0.2)
    a_grid = np.linspace(0.0, 5.0, 6)

    def rf(ap, a, z, age, xp=np):
        return -ap

    with pytest.raises(ValueError, match="survival"):           # wrong length
        FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                             beta=0.95, horizon=5, survival=np.ones(4))
    with pytest.raises(ValueError, match="survival"):           # > 1
        FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                             beta=0.95, horizon=5,
                             survival=np.array([1.0, 1.0, 1.0, 1.0, 1.5]))
