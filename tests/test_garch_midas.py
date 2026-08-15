"""Tests for GARCH-MIDAS mixed-frequency volatility model (Engle, Ghysels & Sohn 2013)."""
import numpy as np
import pandas as pd
import pytest

from puremacro.midas import garch_midas, GarchMidasResult
from puremacro.volatility import garch_midas as garch_midas_vol


def test_garch_midas_exports_match():
    assert garch_midas is garch_midas_vol


def test_garch_midas_basic_rv():
    rng = np.random.default_rng(42)
    N = 800  # 40 periods of K=20
    K = 20
    L = 6
    eps = rng.normal(0, 1, N)
    returns = 0.005 + 0.02 * eps

    res = garch_midas(returns, x_lf=None, K=K, L=L)
    assert isinstance(res, GarchMidasResult)
    assert res.converged
    assert res.n_obs == N
    assert res.n_low == N // K
    assert len(res.weights) == L
    assert np.isclose(np.sum(res.weights), 1.0)
    assert len(res.tau) == N
    assert len(res.g) == N
    assert len(res.sigma) == N
    assert np.all(res.tau > 0)
    assert np.all(res.g > 0)
    assert np.all(res.sigma > 0)
    assert np.allclose(res.sigma, np.sqrt(res.tau * res.g))
    assert res.alpha >= 0
    assert res.beta >= 0
    assert res.alpha + res.beta < 1.0


def test_garch_midas_with_macro_variable():
    rng = np.random.default_rng(101)
    T = 50
    K = 22
    N = T * K
    L = 8

    macro_lf = rng.normal(1.5, 0.5, T)
    returns_hf = rng.normal(0.001, 0.015, N)

    res = garch_midas(returns_hf, x_lf=macro_lf, K=K, L=L)
    assert res.converged
    assert res.n_low == T
    assert res.n_obs == N
    assert res.w2 >= 1.0
    summary = res.summary()
    assert "GARCH-MIDAS (Engle-Ghysels-Sohn 2013)" in summary
    assert "persistence" in summary


def test_garch_midas_pandas_series():
    rng = np.random.default_rng(77)
    dates = pd.date_range("2020-01-01", periods=600, freq="B")
    r_series = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)

    res = garch_midas(r_series, K=20, L=5)
    assert isinstance(res.sigma, pd.Series)
    assert isinstance(res.tau, pd.Series)
    assert isinstance(res.g, pd.Series)
    assert (res.sigma.index == dates[: res.n_obs]).all()


def test_garch_midas_invalid_inputs():
    rng = np.random.default_rng(0)
    # Too short series: T <= L
    r_short = rng.normal(0, 1, 50)  # K=20 -> T=2, L=6 -> T <= L
    with pytest.raises(ValueError, match="requires T > L"):
        garch_midas(r_short, K=20, L=6)

    # Macro series too short
    r_long = rng.normal(0, 1, 200)  # T=10
    macro_short = rng.normal(0, 1, 5)  # len=5 < 10
    with pytest.raises(ValueError, match="x_lf length"):
        garch_midas(r_long, x_lf=macro_short, K=20, L=4)
