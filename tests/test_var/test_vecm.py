import numpy as np
import pandas as pd
import pytest

from puremacro.var.vecm import engle_granger, johansen, fit_vecm


def test_engle_granger_cointegrated_series():
    """Two I(1) series sharing a common trend → EG rejects unit root in residuals."""
    rng = np.random.default_rng(11)
    T = 300
    common = np.cumsum(rng.standard_normal(T))  # I(1)
    y = common + rng.standard_normal(T) * 0.5
    x = common + rng.standard_normal(T) * 0.5
    out = engle_granger(y, x)
    assert out["adf_p"] < 0.10
    assert "cointegrating_beta" in out


def test_engle_granger_independent_series():
    """Two independent random walks: no cointegration."""
    rng = np.random.default_rng(13)
    T = 300
    y = np.cumsum(rng.standard_normal(T))
    x = np.cumsum(rng.standard_normal(T))
    out = engle_granger(y, x)
    # Should fail to reject unit root in residuals at 5%
    assert out["adf_p"] > 0.05


def test_johansen_rank_one():
    """Two I(1) series with a single cointegrating vector → Johansen finds rank 1."""
    rng = np.random.default_rng(17)
    T = 400
    common = np.cumsum(rng.standard_normal(T))
    y0 = common + rng.standard_normal(T) * 0.3
    y1 = common + rng.standard_normal(T) * 0.3
    Y = pd.DataFrame({"y0": y0, "y1": y1})
    out = johansen(Y, p=2)
    assert out["r_hat"] == 1
    assert out["beta"].shape == (2, 1)


def test_fit_vecm_runs():
    rng = np.random.default_rng(19)
    T = 300
    common = np.cumsum(rng.standard_normal(T))
    y0 = common + rng.standard_normal(T) * 0.3
    y1 = 0.5 * common + rng.standard_normal(T) * 0.3
    Y = pd.DataFrame({"y0": y0, "y1": y1})
    out = fit_vecm(Y, p=2, r=1)
    for key in ("alpha", "beta", "Pi", "residuals"):
        assert key in out
    assert out["alpha"].shape == (2, 1)
    assert out["beta"].shape == (2, 1)
    assert out["Pi"].shape == (2, 2)
