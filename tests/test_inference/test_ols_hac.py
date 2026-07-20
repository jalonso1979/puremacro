import numpy as np
import pandas as pd
import pytest

from puremacro.inference._ols_helpers import ols_hac


def test_ols_hac_parity_with_statsmodels():
    rng = np.random.default_rng(7)
    T = 200
    X = np.column_stack([np.ones(T), rng.standard_normal((T, 2))])
    beta_true = np.array([0.5, 1.0, -0.5])
    y = X @ beta_true + rng.standard_normal(T)

    # Reference: statsmodels OLS w/ HAC, L=4
    import statsmodels.api as sm
    sm_res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    out = ols_hac(y, X, lags=4)
    assert np.allclose(out["beta"], sm_res.params, atol=1e-10)
    assert np.allclose(out["se"], sm_res.bse, atol=1e-6)


def test_ols_hac_returns_documented_keys():
    rng = np.random.default_rng(13)
    T = 100
    X = rng.standard_normal((T, 3))
    y = X @ np.array([1.0, -0.5, 0.2]) + rng.standard_normal(T)
    out = ols_hac(y, X, lags=2)
    for key in ("beta", "se", "t", "vcov", "residuals", "n_obs"):
        assert key in out
    assert out["n_obs"] == T
    assert out["beta"].shape == (3,)
    assert out["se"].shape == (3,)
    assert out["vcov"].shape == (3, 3)
