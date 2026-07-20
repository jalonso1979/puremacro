from __future__ import annotations

import numpy as np

from puremacro.vfi.examples import huggett_steady_state


def test_huggett_clears_bond_market():
    out = huggett_steady_state(beta=0.96, gamma=1.5, n_z=5, n_a=150)
    # bond in zero net supply: aggregate net assets clear to ~0 (grid-bounded)
    assert abs(out["mean_assets"]) < 0.05
    assert out["mean_assets"] == out["mean_assets"]            # not NaN
    # precautionary saving pins the equilibrium rate strictly below the rate of
    # time preference 1/beta - 1
    assert out["r"] < 1.0 / 0.96 - 1.0
    assert out["r"] > -0.05


def test_huggett_has_borrowers_and_savers():
    out = huggett_steady_state(beta=0.96, gamma=1.5, n_z=5, n_a=150)
    # genuine trade: some agents borrow (a<0), some save (a>0)
    assert 0.0 < out["frac_borrowing"] < 1.0


def test_huggett_residual_small_and_keys():
    out = huggett_steady_state(beta=0.96, gamma=1.5, n_z=5, n_a=150)
    assert abs(out["equilibrium"].residual) < 0.05
    for k in ("equilibrium", "r", "mean_assets", "frac_borrowing"):
        assert k in out


def test_huggett_more_risk_lowers_rate():
    # higher idiosyncratic risk -> stronger precautionary bond demand -> the bond
    # price rises, i.e. the equilibrium interest rate falls.
    lo = huggett_steady_state(beta=0.96, gamma=2.0, sigma=0.15, n_z=5, n_a=120)
    hi = huggett_steady_state(beta=0.96, gamma=2.0, sigma=0.30, n_z=5, n_a=120)
    assert hi["r"] < lo["r"]


def test_huggett_exported():
    from puremacro.vfi import huggett_steady_state as fn

    assert fn is huggett_steady_state
