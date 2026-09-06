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


# ---------------------------------------------------------------------------
# Regression tests for the 2.3.x audit: presentation layer on regime / sign /
# quantile results (C13, C32, M33) and ci/alpha validation.
# ---------------------------------------------------------------------------
from matplotlib.figure import Figure

from puremacro.lp import lp_state_dep, lp_state_dep_iv
from puremacro.plot import plot_irf_multi


def _regime_df(seed: int = 11, T: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = rng.standard_normal(T)
    x = rng.standard_normal(T)
    z = 0.8 * x + 0.3 * rng.standard_normal(T)
    high = (s > 0).astype(float)
    y = np.cumsum((1.0 * high + 0.2 * (1 - high)) * x + 0.3 * rng.standard_normal(T))
    return pd.DataFrame({"y": y, "x": x, "z": z, "s": s})


@pytest.mark.parametrize("make", [
    lambda df: lp_state_dep(df, y="y", x="x", state="s", horizon=3, lags=1, ci=0.90),
    lambda df: lp_state_dep(df, y="y", x="x", state="s", horizon=3, lags=1,
                            transition="threshold"),
    lambda df: lp_state_dep_iv(df, y="y", x="x", z="z", state="s", horizon=3, lags=1),
    lambda df: lp_garch_state(df, y="y", x="x", horizon=3, lags=1),
])
def test_regime_results_plot_point_summary(make):
    """lp_state_dep / lp_state_dep_iv / lp_garch_state results have
    beta_H/beta_L columns. Their .plot() raised KeyError 'beta', .point
    raised KeyError, .se was an empty array, .ci_lower was None and
    summary() printed an all-NaN body (C13, C32, M33)."""
    res = make(_regime_df())
    assert res.labels == ["H", "L"]

    point = res.point
    assert isinstance(point, pd.DataFrame) and list(point.columns) == ["H", "L"]
    assert point.index.name == "h" and list(point.index) == [0, 1, 2, 3]
    np.testing.assert_allclose(point["H"].values, res["beta_H"].values)
    for prop, col in (("se", "se"), ("ci_lower", "lo"), ("ci_upper", "hi")):
        val = getattr(res, prop)
        assert isinstance(val, pd.DataFrame) and list(val.columns) == ["H", "L"]
        np.testing.assert_allclose(val["L"].values, res[f"{col}_L"].values)
    t = res.t_stat
    assert isinstance(t, pd.DataFrame)
    np.testing.assert_allclose(t["H"].values, res["beta_H"].values / res["se_H"].values)

    fig = res.plot(title="regimes")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    labels = [ln.get_label() for ln in ax.lines]
    assert "H" in labels and "L" in labels
    assert len(ax.collections) == 2                # one band per regime
    assert ax.get_legend() is not None

    s = res.summary()
    body = [ln for ln in s.splitlines() if ln[:4].strip().isdigit()]
    assert len(body) == 2 * len(res)               # one row per horizon and regime
    assert not any("nan" in ln for ln in body)
    assert "coefficients: H, L" in s
    assert any(ln.rstrip().endswith("*") for ln in body)   # h=0 response is significant

    md = res.to_markdown()
    assert "beta_H" in md and "beta_L" in md


def test_lp_asymmetric_plot_and_point_frame():
    """lp_asymmetric returns beta_pos/beta_neg; .plot() raised KeyError 'beta'."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"y": np.cumsum(rng.standard_normal(200)), "x": rng.standard_normal(200)})
    res = lp_asymmetric(df, y="y", x="x", horizon=3, lags=1)
    assert res.labels == ["pos", "neg"]
    assert list(res.point.columns) == ["pos", "neg"]
    fig = res.plot()
    assert sorted(ln.get_label() for ln in fig.axes[0].lines
               if not ln.get_label().startswith("_")) == ["neg", "pos"]
    s = res.summary()
    assert "nan" not in s.lower().split("significance")[0]
    # plot_irf_multi expands regime frames into one line per label
    fig2 = plot_irf_multi({"asym": res, "plain": lp_hac(df, y="y", x="x", horizon=3, lags=1)})
    labels = [ln.get_label() for ln in fig2.axes[0].lines]
    assert "asym [pos]" in labels and "asym [neg]" in labels and "plain" in labels


def test_lp_quantile_summary_t_stat_and_plot():
    """lp_quantile has no se column: summary() printed an all-NaN body and
    .t_stat raised ValueError (shape mismatch with the empty se array)."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"y": np.cumsum(rng.standard_normal(120)), "x": rng.standard_normal(120)})
    res = lp_quantile(df, y="y", x="x", quantiles=(0.25, 0.75), horizon=1, lags=1, n_boot=8)
    assert res.labels == []
    assert res.se.shape == (len(res),) and np.isnan(res.se).all()
    t = res.t_stat
    assert t.shape == (len(res),) and np.isnan(t).all()
    s = res.summary()
    body = [ln for ln in s.splitlines() if ln[:4].strip().isdigit()]
    assert len(body) == len(res)
    assert all("tau" in s.splitlines()[2] for _ in [0])
    # beta / lo / hi cells are real numbers even though se is nan
    assert all(ln.split()[2] != "nan" for ln in body)
    fig = res.plot()
    labels = [ln.get_label() for ln in fig.axes[0].lines if not ln.get_label().startswith("_")]
    assert labels == ["tau=0.25", "tau=0.75"]


def test_summary_has_significance_flags():
    """docs/lp.md promises 'significance flags' in summary(); the old table
    had none."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(300)
    y = np.cumsum(1.0 * x + 0.3 * rng.standard_normal(300))
    res = lp_hac(pd.DataFrame({"y": y, "x": x}), y="y", x="x", horizon=2, lags=1, ci=0.90)
    s = res.summary()
    assert "sig" in s.splitlines()[2]
    assert "bands: 90%" in s
    first = [ln for ln in s.splitlines() if ln.lstrip().startswith("0 ")][0]
    assert first.rstrip().endswith("***")
    assert "*** p<0.01" in s


@pytest.mark.parametrize("call", [
    lambda df: lp_hac(df, y="y", x="x", horizon=2, lags=1, ci=90),
    lambda df: lp_hac(df, y="y", x="x", horizons=[0, 1], alpha=1.5),
    lambda df: lp_iv(df, y="y", x="x", z="z", horizon=2, lags=1, ci=95),
    lambda df: lp_asymmetric(df, y="y", x="x", horizon=2, lags=1, ci=0.0),
    lambda df: lp_quantile(df, y="y", x="x", quantiles=(0.5,), horizon=1, lags=1, n_boot=2, ci=1.0),
    lambda df: la_lp(df, y="y", x="x", horizon=2, lags=1, ci=90),
    lambda df: lp_garch_in_mean(df, y="y", x="x", horizon=2, lags=1, alpha=0),
    lambda df: lp_hac(df, y="y", x="x", horizon=-1, lags=1),
    lambda df: lp_hac(df, y="y", x="x", horizon=2, lags=-1),
])
def test_ci_alpha_horizon_lags_are_validated(call):
    """ci=90 / alpha=1.5 were accepted silently with all-NaN lo/hi bands."""
    df = _regime_df()
    with pytest.raises(ValueError):
        call(df)


def test_panel_ci_validation():
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([["A", "B", "C"], range(40)], names=["code", "date"])
    df = pd.DataFrame({"y": rng.standard_normal(120), "x": rng.standard_normal(120)}, index=idx)
    for fn in (panel_lp, panel_lp_dk, cce_panel_lp, mean_group_panel_lp):
        with pytest.raises(ValueError, match="ci"):
            fn(df, y="y", x="x", horizon=1, lags=1, ci=90)


def test_lp_result_slice_without_beta_keeps_none_ci():
    res = lp_hac(_regime_df(), y="y", x="x", horizon=2, lags=1)
    sub = res[["h", "se"]]
    assert sub.ci_lower is None and sub.ci_upper is None
    with pytest.raises(KeyError):
        _ = sub.point
