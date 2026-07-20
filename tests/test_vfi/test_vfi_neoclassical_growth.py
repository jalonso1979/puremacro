from __future__ import annotations

import numpy as np

from puremacro.vfi.examples import neoclassical_growth


def test_steady_state_and_ergodic_mean():
    out = neoclassical_growth(n_k=300)
    K_ss = out["K_ss"]
    # analytical deterministic steady state K_ss = (alpha*beta/(1-beta(1-delta)))^(1/(1-alpha))
    expected = (0.35 * 0.984 / (1.0 - 0.984 * 0.99)) ** (1.0 / 0.65)
    assert K_ss == np.float64(expected) or abs(K_ss - expected) < 1e-9
    # under the tiny TFP shocks (sigma=0.005) the ergodic capital concentrates at K_ss
    assert abs(out["mean_capital"] - K_ss) / K_ss < 0.01
    np.testing.assert_allclose(out["distribution"].sum(), 1.0, atol=1e-10)
    for key in ("K_ss", "solution", "distribution", "mean_capital", "k_grid"):
        assert key in out


def test_policy_monotone_in_capital():
    out = neoclassical_growth(n_k=300)
    pol = out["solution"].policy_aprime
    assert np.all(np.diff(pol, axis=0) >= 0)        # k'(k) rises with k


def test_neoclassical_growth_exported():
    from puremacro.vfi import neoclassical_growth as fn

    assert fn is neoclassical_growth
