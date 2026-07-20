from __future__ import annotations

import numpy as np

from puremacro.vfi import markov_stationary
from puremacro.vfi.krusell_smith import (
    KSEquilibrium,
    krusell_smith,
    ks_simulate,
)


def test_ks_simulate_constant_policy_holds_K():
    # identity policy a'=a: each agent stays put, so mean capital is constant.
    n_a, nz, nZ, nK = 10, 2, 2, 4
    a_grid = np.linspace(0.0, 9.0, n_a)
    P_z = np.array([[0.7, 0.3], [0.4, 0.6]])
    K_grid = np.linspace(2.0, 8.0, nK)
    pol = np.broadcast_to(np.arange(n_a)[:, None, None, None],
                          (n_a, nz, nZ, nK)).copy()
    Z_path = np.array([0, 1, 0, 1, 1, 0, 0, 1] * 4)
    mu0 = np.zeros((n_a, nz))
    mu0[3, :] = markov_stationary(P_z)                 # all mass at a_grid[3] = 3.0
    K = ks_simulate(pol, a_grid, P_z, K_grid, Z_path, mu0=mu0)
    assert K.shape == (Z_path.size + 1,)
    np.testing.assert_allclose(K, a_grid[3], atol=1e-9)   # K stays at 3.0


def test_krusell_smith_converges_high_r2_and_mean_K():
    eq = krusell_smith(n_a=150, n_K=5, T=1500, burn_in=200, seed=1)
    assert isinstance(eq, KSEquilibrium)
    assert eq.converged                                    # forecast-rule fixed point found
    assert np.all(eq.r_squared > 0.99)                     # KS "approximate aggregation"
    # mean capital sits near the no-aggregate-risk (Aiyagari) level
    assert abs(eq.mean_K - eq.no_agg_risk_K) / eq.no_agg_risk_K < 0.05
    assert eq.mean_K > 0.0
    assert np.all(eq.b1 > 0.0)                             # sensible persistence in the rule


def test_krusell_smith_degenerate_reduces_to_aiyagari():
    # a SINGLE aggregate state (no aggregate uncertainty) => the simulated mean
    # capital approximates the no-aggregate-risk Aiyagari capital K* (the
    # reduction). The gap is the coarse aggregate-K-grid discretization (the
    # K-folded household interpolates the policy over n_K points), not a bug.
    eq = krusell_smith(Z_vals=(1.0,), P_Z=((1.0,),), n_a=150, n_K=7, T=800,
                       burn_in=150, seed=2)
    assert abs(eq.mean_K - eq.no_agg_risk_K) / eq.no_agg_risk_K < 0.06
    # and the no-aggregate-risk path settles to a steady state (tiny tail variation)
    tail = eq.K_path[400:]
    assert (tail.max() - tail.min()) / eq.mean_K < 0.02


def test_krusell_smith_exported():
    from puremacro.vfi import krusell_smith as f1
    from puremacro.vfi import ks_simulate as f2
    from puremacro.vfi import KSEquilibrium as C

    assert f1 is krusell_smith and f2 is ks_simulate and C is KSEquilibrium
