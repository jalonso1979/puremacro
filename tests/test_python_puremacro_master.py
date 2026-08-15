"""Master test suite for puremacro Python package features."""
import pytest
import numpy as np

from puremacro.vfi.model import Model
from puremacro.vfi.hjb_achdou import solve_hjb_achdou
from puremacro.vfi.aiyagari_endogenous import solve_aiyagari_endogenous
from puremacro.did.callaway_santanna import callaway_santanna
from puremacro.causal.synthetic_control import synthetic_control
from puremacro.connectedness.bk2018 import barunik_krehlik


def test_dynare_vfi_model():
    m = Model("Test Model")
    m.set_param("beta", 0.95).set_param("gamma", 1.5)
    m.add_state("k", 0.2, 10.0, 25)
    m.add_shock("z", 0.85, 0.10, 3)
    m.set_utility(lambda c, p: (c**(1 - p["gamma"])) / (1 - p["gamma"]))
    m.set_budget(lambda k, kp, z, p: np.exp(z) * (k**0.35) + 0.90 * k - kp)

    res = m.solve(howard_steps=10)
    assert res["V"].shape == (25, 3)
    assert res["mean_assets"] > 0
    assert 0 <= res["gini"] <= 1.0


def test_hjb_achdou():
    res = solve_hjb_achdou(Na=50)
    assert res["V"].shape == (50, 2)
    assert res["n_iter"] <= 500


def test_aiyagari_endogenous():
    res = solve_aiyagari_endogenous(Na=30, Ne=3)
    assert res["K_star"] > 0
    assert res["L_star"] > 0


def test_callaway_santanna():
    np.random.seed(42)
    N_units = 30
    T_periods = 5
    group_assignment = np.array([np.inf]*10 + [3]*10 + [4]*10)

    y_vec, g_vec, t_vec, u_vec = [], [], [], []
    for u in range(N_units):
        g_val = group_assignment[u]
        for t in range(1, T_periods + 1):
            y_val = 2.0 + (t >= g_val) * 1.5 + np.random.randn() * 0.2
            y_vec.append(y_val)
            g_vec.append(g_val)
            t_vec.append(t)
            u_vec.append(u)

    res = callaway_santanna(y_vec, g_vec, t_vec, u_vec)
    assert not np.isnan(res["overall_att"])


def test_synthetic_control():
    np.random.seed(44)
    Y_panel = 10.0 + np.random.randn(40, 8)
    Y_panel[20:, 0] += 3.0

    res = synthetic_control(Y_panel, 0, 20)
    assert len(res["weights"]) == 7
    assert res["att_post"] > 1.0


def test_barunik_krehlik():
    np.random.seed(55)
    ret = np.random.randn(100, 3)
    res = barunik_krehlik(ret, var_lags=2, fevd_horizon=50)
    assert 0 <= res["total_spillover"] <= 100
