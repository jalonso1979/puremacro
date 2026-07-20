from __future__ import annotations

import numpy as np

from puremacro.vfi.examples import aiyagari_steady_state


def test_aiyagari_steady_state_log():
    res = aiyagari_steady_state(n_a=100, n_z=5)
    r_max = 1.0 / 0.96 - 1.0
    assert 0.0 < res["r"] < r_max              # precautionary wedge: r below 1/beta-1
    assert res["K"] > 0.0 and np.isfinite(res["K"])
    assert res["Y"] > 0.0
    assert 0.0 < res["wealth_gini"] < 1.0      # nondegenerate wealth inequality
    assert abs(res["equilibrium"].residual) < 1e-2
    # firm wage consistent with the equilibrium K/L
    alpha = 0.36
    np.testing.assert_allclose(
        res["w"], (1 - alpha) * (res["K"] / res["L"]) ** alpha, rtol=1e-2
    )


def test_aiyagari_crra_gamma2_runs():
    res = aiyagari_steady_state(n_a=80, n_z=5, gamma=2.0)
    assert 0.0 < res["r"] < 1.0 / 0.96 - 1.0
    assert res["K"] > 0.0
    assert 0.0 < res["wealth_gini"] < 1.0


def test_aiyagari_exported_from_package():
    from puremacro.vfi import aiyagari_steady_state as fn

    assert fn is aiyagari_steady_state
