"""Tests for LPResult and standardized keywords across LP estimators."""
import numpy as np
import pandas as pd
import pytest

from puremacro.lp import (
    LPResult,
    lp_hac,
    panel_lp,
    panel_lp_dk,
    lp_iv,
    lp_asymmetric,
    lp_smooth,
    lp_quantile,
    la_lp,
    lp_iv_lewbel,
    cce_panel_lp,
    mean_group_panel_lp,
    lp_garch_state,
    lp_garch_in_mean,
)


def test_lp_result_properties_and_dataframe_behavior():
    rows = [
        {"h": 0, "beta": 1.0, "se": 0.2, "t": 5.0, "lo": 0.67, "hi": 1.33},
        {"h": 1, "beta": 0.5, "se": 0.25, "t": 2.0, "lo": 0.09, "hi": 0.91},
        {"h": 2, "beta": 0.0, "se": 0.3, "t": 0.0, "lo": -0.49, "hi": 0.49},
    ]
    res = LPResult(rows)
    res.index = res["h"]
    res.y_name = "output"
    res.x_name = "mp_shock"
    res.method = "LP-HAC"

    # Subclass of DataFrame
    assert isinstance(res, pd.DataFrame)
    assert isinstance(res, LPResult)
    assert len(res) == 3
    assert list(res.columns) == ["h", "beta", "se", "t", "lo", "hi"]

    # Properties
    np.testing.assert_allclose(res.point, [1.0, 0.5, 0.0])
    np.testing.assert_allclose(res.se, [0.2, 0.25, 0.3])
    np.testing.assert_allclose(res.se_arr, [0.2, 0.25, 0.3])
    np.testing.assert_allclose(res.ci_lower, [0.67, 0.09, -0.49])
    np.testing.assert_allclose(res.ci_upper, [1.33, 0.91, 0.49])
    np.testing.assert_allclose(res.t_stat, [5.0, 2.0, 0.0])
    np.testing.assert_allclose(res.horizons, [0, 1, 2])

    # Metadata & conversion
    meta = res.metadata
    assert meta["y_name"] == "output"
    assert meta["x_name"] == "mp_shock"
    assert meta["method"] == "LP-HAC"
    df = res.to_frame()
    assert type(df) is pd.DataFrame
    assert type(res.frame) is pd.DataFrame

    # Text summary
    s = res.summary()
    assert "LP-HAC" in s
    assert "beta" in s

    # Lazy plot
    fig = res.plot(title="Test IRF")
    assert fig is not None


def test_lp_hac_returns_lp_result():
    rng = np.random.default_rng(42)
    y = np.cumsum(rng.standard_normal(100))
    x = rng.standard_normal(100)

    res = lp_hac(y, x, horizon=4, lags=2, ci=0.90)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 5
    assert len(res.point) == 5
    assert res.method == "LP-HAC"


def test_panel_lp_standardized_keywords_and_result():
    rng = np.random.default_rng(123)
    codes = ["US", "DE"]
    dates = pd.date_range("2010-01-01", periods=30, freq="QE")
    idx = pd.MultiIndex.from_product([codes, dates], names=["code", "date"])
    df = pd.DataFrame(
        {
            "y": rng.standard_normal(len(idx)),
            "x": rng.standard_normal(len(idx)),
        },
        index=idx,
    )

    res = panel_lp(df, y="y", x="x", horizon=3, lags=1, ci=0.95)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 4
    assert res.method == "panel_lp"
    fig = res.plot()
    assert fig is not None


def test_panel_lp_dk_standardized_keywords_and_result():
    rng = np.random.default_rng(123)
    codes = ["US", "DE"]
    dates = pd.date_range("2010-01-01", periods=30, freq="QE")
    idx = pd.MultiIndex.from_product([codes, dates], names=["code", "date"])
    df = pd.DataFrame(
        {
            "y": rng.standard_normal(len(idx)),
            "x": rng.standard_normal(len(idx)),
        },
        index=idx,
    )

    res = panel_lp_dk(df, y="y", x="x", horizon=3, lags=1, ci=0.90)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 4


def test_lp_iv_standardized_keywords_and_result():
    rng = np.random.default_rng(42)
    n = 100
    z = rng.standard_normal(n)
    v = rng.standard_normal(n)
    x = 0.8 * z + 0.5 * v + rng.standard_normal(n) * 0.1
    y = np.cumsum(0.5 * x + v + rng.standard_normal(n) * 0.2)
    df = pd.DataFrame({"y": y, "x": x, "z": z})

    res = lp_iv(df, y="y", x="x", z="z", horizon=3, lags=1, ci=0.90)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 4
    assert res.method == "LP-IV"
    assert "first_stage_f" in res.columns


def test_lp_asymmetric_standardized_keywords_and_result():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"y": rng.standard_normal(100), "x": rng.standard_normal(100)})
    res = lp_asymmetric(df, y="y", x="x", horizon=3, lags=1, ci=0.90)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 4
    assert res.method == "LP-asymmetric"


def test_lp_smooth_standardized_keywords_and_result():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"y": rng.standard_normal(100), "x": rng.standard_normal(100)})
    res = lp_smooth(df, y="y", x="x", horizon=4, lags=1, ci=0.90)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 5
    assert res.method == "LP-smooth"


def test_lp_quantile_standardized_keywords_and_result():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"y": rng.standard_normal(80), "x": rng.standard_normal(80)})
    res = lp_quantile(df, y="y", x="x", quantiles=[0.5], horizon=2, lags=1, ci=0.90, n_boot=5)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    assert res.method == "LP-quantile"


def test_la_lp_standardized_keywords_and_result():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"y": rng.standard_normal(100), "x": rng.standard_normal(100)})
    res = la_lp(df, y="y", x="x", horizon=3, lags=1, ci=0.90)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 4
    assert res.method == "la_lp"


def test_lp_iv_lewbel_standardized_keywords_and_result():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(3):
        for t in range(40):
            rows.append({
                "code": f"E{i}",
                "date": pd.Timestamp("2020-01-01") + pd.DateOffset(months=t),
                "y": rng.standard_normal(),
                "x": rng.standard_normal(),
                "z": rng.standard_normal(),
            })
    df = pd.DataFrame(rows)
    res = lp_iv_lewbel(df, y="y", x_endog="x", heterosk_source="z", horizon=2, lags=1, ci=0.90)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    assert res.method == "LP-IV-Lewbel"


def test_cce_panel_lp_standardized_keywords_and_result():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(3):
        for t in range(30):
            rows.append({
                "code": f"E{i}",
                "date": pd.Timestamp("2020-01-01") + pd.DateOffset(months=t),
                "y": rng.standard_normal(),
                "x": rng.standard_normal(),
            })
    df = pd.DataFrame(rows).set_index(["code", "date"]).sort_index()
    res = cce_panel_lp(df, y="y", x="x", horizon=2, lags=1, ci=0.90)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    assert res.method == "cce_panel_lp"


def test_mean_group_panel_lp_standardized_keywords_and_result():
    rng = np.random.default_rng(42)
    rows = []
    for i in range(3):
        for t in range(35):
            rows.append({
                "code": f"E{i}",
                "date": pd.Timestamp("2020-01-01") + pd.DateOffset(months=t),
                "y": rng.standard_normal(),
                "x": rng.standard_normal(),
            })
    df = pd.DataFrame(rows).set_index(["code", "date"]).sort_index()
    res = mean_group_panel_lp(df, y="y", x="x", horizon=2, lags=1, ci=0.90, min_obs=20)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    assert res.method == "mean_group_panel_lp"


def test_lp_garch_state_standardized_keywords_and_result():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"y": rng.standard_normal(60), "x": rng.standard_normal(60)})
    res = lp_garch_state(df, y="y", x="x", horizon=2, lags=1, ci=0.90)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    assert res.method == "LP-garch-state"


def test_lp_garch_in_mean_standardized_keywords_and_result():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"y": rng.standard_normal(60), "x": rng.standard_normal(60)})
    res = lp_garch_in_mean(df, y="y", x="x", horizon=2, lags=1, ci=0.90)
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 3
    assert res.method == "LP-garch-in-mean"
