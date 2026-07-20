import numpy as np
import pandas as pd

from puremacro.lp.iv_helpers import romer_romer_clean, gertler_karadi_surprise


def test_romer_romer_returns_residuals_orthogonal_to_controls():
    rng = np.random.default_rng(11)
    T = 200
    idx = pd.date_range("2000-01-01", periods=T, freq="MS")
    z1 = rng.standard_normal(T)
    z2 = rng.standard_normal(T)
    raw = 0.5 * z1 + 0.3 * z2 + rng.standard_normal(T) * 0.5
    raw_s = pd.Series(raw, index=idx)
    ctrl = pd.DataFrame({"z1": z1, "z2": z2}, index=idx)
    cleaned = romer_romer_clean(raw_s, ctrl)
    # Cleaned series should be (nearly) uncorrelated with controls
    assert abs(np.corrcoef(cleaned.values, z1)[0, 1]) < 0.05
    assert abs(np.corrcoef(cleaned.values, z2)[0, 1]) < 0.05


def test_gertler_karadi_surprise_no_market_proxy_passthrough():
    rng = np.random.default_rng(13)
    s = pd.Series(rng.standard_normal(50),
                  index=pd.date_range("2010-01-01", periods=50, freq="MS"))
    out = gertler_karadi_surprise(s)
    pd.testing.assert_series_equal(out, s)


def test_gertler_karadi_surprise_orthogonalises_market():
    rng = np.random.default_rng(17)
    T = 100
    idx = pd.date_range("2010-01-01", periods=T, freq="MS")
    market = rng.standard_normal(T)
    raw = 0.6 * market + 0.4 * rng.standard_normal(T)
    s = pd.Series(raw, index=idx); m = pd.Series(market, index=idx)
    cleaned = gertler_karadi_surprise(s, m)
    assert abs(np.corrcoef(cleaned.values, market)[0, 1]) < 0.05
