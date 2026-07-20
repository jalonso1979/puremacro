"""Coverage tests for puremacro.uncertainty.composite.

Tests exercise: _rescale, _drop_existing, _infer_freq_suffix, _pc1_from_wide,
_ar2_residual, build_backbone, build_composite, build_innovation.

Strategy:
- Known-value tests using synthetic data with a closed-form correct answer.
- Property tests (output shape, column names, sa_source, mean/sd contracts,
  idempotency, sign consistency, tier annotation, freq routing).
- Error handling (bad freq arg on build_innovation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.uncertainty.composite import (
    DEFAULT_PROXIES,
    GPR_SIGN_FLOOR,
    LONG_PROXY_BASES,
    MIN_OBS,
    TARGET_MEAN,
    TARGET_SD,
    _ar2_residual,
    _drop_existing,
    _infer_freq_suffix,
    _pc1_from_wide,
    _rescale,
    build_backbone,
    build_composite,
    build_innovation,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal synthetic panels
# ---------------------------------------------------------------------------

def _make_monthly_proxy_panel(
    codes: list[str],
    proxies: list[str],
    n: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a long-form panel with one common factor + noise per country."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    rows = []
    for code in codes:
        f = rng.standard_normal(n)
        for proxy in proxies:
            vals = f + 0.3 * rng.standard_normal(n)
            for d, v in zip(dates, vals):
                rows.append(
                    {
                        "code": code,
                        "date": d,
                        "variable": proxy,
                        "value": float(v),
                        "sa_source": "raw",
                        "source": proxy,
                    }
                )
    return pd.DataFrame(rows)


def _make_gpr_panel(codes: list[str], n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    rows = []
    for code in codes:
        vals = rng.uniform(50.0, 200.0, n)
        for d, v in zip(dates, vals):
            rows.append(
                {
                    "code": code,
                    "date": d,
                    "variable": "gpr_m",
                    "value": float(v),
                    "sa_source": "raw",
                    "source": "gpr",
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# _rescale
# ===========================================================================


def test_rescale_mean_sd_contract():
    """_rescale must produce mean == TARGET_MEAN and sd == TARGET_SD (ddof=0)."""
    rng = np.random.default_rng(1)
    s = pd.Series(rng.standard_normal(50) * 5 + 100)
    r = _rescale(s)
    assert abs(r.mean() - TARGET_MEAN) < 1e-10
    assert abs(r.std(ddof=0) - TARGET_SD) < 1e-10


def test_rescale_constant_series_returns_target_mean():
    """Constant series has sd=0; _rescale should return TARGET_MEAN everywhere."""
    s = pd.Series([7.0] * 30)
    r = _rescale(s)
    assert (r == TARGET_MEAN).all()
    assert len(r) == len(s)


def test_rescale_preserves_index():
    idx = pd.date_range("2005-01-01", periods=10, freq="MS")
    s = pd.Series(range(10), index=idx, dtype=float)
    r = _rescale(s)
    pd.testing.assert_index_equal(r.index, idx)


def test_rescale_known_value():
    """_rescale of [0,1,2,3,4] should produce [60, 80, 100, 120, 140]."""
    s = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    r = _rescale(s)
    # mean=2, sd=sqrt(2) (ddof=0), rescaled = (x-2)/sqrt(2) * 20 + 100
    expected = pd.Series(
        [(x - 2) / np.sqrt(2) * TARGET_SD + TARGET_MEAN for x in range(5)]
    )
    np.testing.assert_allclose(r.values, expected.values, rtol=1e-12)


# ===========================================================================
# _drop_existing
# ===========================================================================


def test_drop_existing_removes_target_variable():
    df = pd.DataFrame(
        {
            "code": ["USA", "USA", "DEU"],
            "variable": ["gpr_m", "unc_m", "unc_m"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    out = _drop_existing(df, "unc_m")
    assert (out["variable"] == "gpr_m").all()
    assert len(out) == 1


def test_drop_existing_no_match_returns_all_rows():
    df = pd.DataFrame({"code": ["A", "B"], "variable": ["x", "y"], "value": [1, 2]})
    out = _drop_existing(df, "z")
    assert len(out) == len(df)


def test_drop_existing_resets_index():
    df = pd.DataFrame(
        {"code": ["A", "B", "C"], "variable": ["x", "y", "x"], "value": [1, 2, 3]}
    )
    out = _drop_existing(df, "y")
    assert list(out.index) == list(range(len(out)))


# ===========================================================================
# _infer_freq_suffix
# ===========================================================================


def test_infer_freq_suffix_all_monthly():
    assert _infer_freq_suffix(["gpr_m", "epu_m", "rv_m"]) == "_m"


def test_infer_freq_suffix_all_quarterly():
    assert _infer_freq_suffix(["gpr_q", "epu_q"]) == "_q"


def test_infer_freq_suffix_mixed_defaults_to_monthly():
    assert _infer_freq_suffix(["gpr_m", "epu_q"]) == "_m"


def test_infer_freq_suffix_empty_defaults_to_monthly():
    assert _infer_freq_suffix([]) == "_m"


def test_infer_freq_suffix_no_underscore_defaults_to_monthly():
    assert _infer_freq_suffix(["gpronly", "epuonly"]) == "_m"


# ===========================================================================
# _pc1_from_wide
# ===========================================================================


def test_pc1_from_wide_returns_series_and_columns():
    rng = np.random.default_rng(0)
    n = 30
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    f = rng.standard_normal(n)
    wide = pd.DataFrame(
        {
            "gpr_m": f + 0.1 * rng.standard_normal(n),
            "epu_m": f + 0.1 * rng.standard_normal(n),
        },
        index=idx,
    )
    result = _pc1_from_wide(wide, "_m")
    assert result is not None
    pc1, cols = result
    assert isinstance(pc1, pd.Series)
    assert isinstance(cols, list)
    assert len(cols) == 2


def test_pc1_from_wide_too_few_obs_returns_none():
    wide = pd.DataFrame({"gpr_m": [1, 2, 3], "epu_m": [2, 3, 4]})
    assert _pc1_from_wide(wide, "_m") is None


def test_pc1_from_wide_single_proxy_returns_none():
    """After z-scoring, only 1 column survives -> fewer than 2 -> None."""
    rng = np.random.default_rng(0)
    n = 30
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    # One real proxy, one constant (sd=0, will be dropped)
    wide = pd.DataFrame(
        {
            "gpr_m": rng.standard_normal(n),
            "epu_m": np.ones(n),
        },
        index=idx,
    )
    assert _pc1_from_wide(wide, "_m") is None


def test_pc1_from_wide_sign_positively_correlated_with_gpr():
    """PC1 should be positively correlated with gpr_m when its loading is informative."""
    rng = np.random.default_rng(7)
    n = 30
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    f = rng.standard_normal(n)
    wide = pd.DataFrame(
        {
            "gpr_m": f + 0.1 * rng.standard_normal(n),
            "epu_m": f + 0.1 * rng.standard_normal(n),
            "rv_m":  f + 0.1 * rng.standard_normal(n),
        },
        index=idx,
    )
    result = _pc1_from_wide(wide, "_m")
    assert result is not None
    pc1, _ = result
    corr = np.corrcoef(pc1.values, wide["gpr_m"].values)[0, 1]
    # Sign anchor: correlation with gpr should be positive
    assert corr > 0


def test_pc1_from_wide_balanced_window_uses_dropna():
    """_pc1_from_wide drops rows with any NaN before SVD."""
    rng = np.random.default_rng(3)
    n = 50
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    f = rng.standard_normal(n)
    col1 = f + 0.1 * rng.standard_normal(n)
    col2 = f + 0.1 * rng.standard_normal(n)
    col1[:5] = np.nan  # first 5 rows missing
    wide = pd.DataFrame({"gpr_m": col1, "epu_m": col2}, index=idx)
    result = _pc1_from_wide(wide, "_m")
    assert result is not None
    pc1, cols = result
    # PC1 index should not include the NaN rows
    assert len(pc1) == n - 5


# ===========================================================================
# _ar2_residual
# ===========================================================================


def test_ar2_residual_first_two_are_nan():
    rng = np.random.default_rng(0)
    s = pd.Series(
        rng.standard_normal(50), index=pd.date_range("2000-01-01", periods=50, freq="MS")
    )
    resid = _ar2_residual(s)
    assert np.isnan(resid.iloc[0]) and np.isnan(resid.iloc[1])


def test_ar2_residual_rest_are_finite():
    rng = np.random.default_rng(0)
    s = pd.Series(
        rng.standard_normal(50), index=pd.date_range("2000-01-01", periods=50, freq="MS")
    )
    resid = _ar2_residual(s)
    assert np.isfinite(resid.iloc[2:]).all()


def test_ar2_residual_too_short_returns_all_nan():
    s = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        index=pd.date_range("2000-01-01", periods=5, freq="MS"),
    )
    resid = _ar2_residual(s)
    assert np.isnan(resid).all()


def test_ar2_residual_preserves_index():
    idx = pd.date_range("2005-06-01", periods=20, freq="MS")
    s = pd.Series(range(20), index=idx, dtype=float)
    resid = _ar2_residual(s)
    pd.testing.assert_index_equal(resid.index, idx)


def test_ar2_residual_near_zero_mean():
    """Residuals from fitting AR(2) to its own DGP should have mean near zero."""
    rng = np.random.default_rng(99)
    n = 200
    e = rng.standard_normal(n)
    s = np.zeros(n)
    for t in range(2, n):
        s[t] = 0.5 * s[t - 1] + 0.2 * s[t - 2] + e[t]
    series = pd.Series(
        s, index=pd.date_range("2000-01-01", periods=n, freq="MS")
    )
    resid = _ar2_residual(series).dropna()
    assert abs(resid.mean()) < 0.1


# ===========================================================================
# build_backbone
# ===========================================================================


def test_build_backbone_adds_unc_m():
    panel = _make_gpr_panel(["USA", "DEU"], n=40)
    out = build_backbone(panel)
    assert "unc_m" in out["variable"].unique()


def test_build_backbone_unc_m_rescaled_to_target():
    """unc_m per country must have mean=100, sd=20 (ddof=0)."""
    panel = _make_gpr_panel(["USA", "DEU"], n=40, seed=5)
    out = build_backbone(panel)
    unc = out[out["variable"] == "unc_m"]
    for code in ["USA", "DEU"]:
        vals = unc[unc["code"] == code]["value"]
        assert abs(vals.mean() - TARGET_MEAN) < 1e-9
        assert abs(vals.std(ddof=0) - TARGET_SD) < 1e-9


def test_build_backbone_no_gpr_m_returns_panel_unchanged():
    """If panel has no gpr_m, return panel as-is."""
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {
            "code": ["USA"] * 30,
            "date": pd.date_range("2000-01-01", periods=30, freq="MS"),
            "variable": ["epu_m"] * 30,
            "value": rng.standard_normal(30),
            "sa_source": ["raw"] * 30,
            "source": ["epu"] * 30,
        }
    )
    out = build_backbone(panel)
    assert len(out) == len(panel)
    assert "unc_m" not in out["variable"].unique()


def test_build_backbone_too_few_obs_skips_country():
    """Countries with fewer than MIN_OBS gpr_m observations are skipped."""
    rng = np.random.default_rng(0)
    n = MIN_OBS - 1  # 23 < 24
    panel = pd.DataFrame(
        {
            "code": ["USA"] * n,
            "date": pd.date_range("2000-01-01", periods=n, freq="MS"),
            "variable": ["gpr_m"] * n,
            "value": rng.uniform(50, 200, n),
            "sa_source": ["raw"] * n,
            "source": ["gpr"] * n,
        }
    )
    out = build_backbone(panel)
    assert "unc_m" not in out["variable"].unique()


def test_build_backbone_output_columns():
    panel = _make_gpr_panel(["USA"], n=30)
    out = build_backbone(panel)
    unc = out[out["variable"] == "unc_m"]
    expected_cols = {"code", "date", "variable", "value", "sa_source", "source"}
    assert expected_cols.issubset(set(unc.columns))


def test_build_backbone_sa_source_is_derived():
    panel = _make_gpr_panel(["USA"], n=30)
    out = build_backbone(panel)
    unc = out[out["variable"] == "unc_m"]
    assert (unc["sa_source"] == "derived").all()


def test_build_backbone_idempotent():
    """Running build_backbone twice does not duplicate unc_m rows."""
    panel = _make_gpr_panel(["USA"], n=30)
    out1 = build_backbone(panel)
    out2 = build_backbone(out1)
    cnt1 = (out1["variable"] == "unc_m").sum()
    cnt2 = (out2["variable"] == "unc_m").sum()
    assert cnt1 == cnt2


def test_build_backbone_exact_min_obs_included():
    """Exactly MIN_OBS observations should produce unc_m (boundary case)."""
    rng = np.random.default_rng(0)
    n = MIN_OBS
    panel = pd.DataFrame(
        {
            "code": ["USA"] * n,
            "date": pd.date_range("2000-01-01", periods=n, freq="MS"),
            "variable": ["gpr_m"] * n,
            "value": rng.uniform(50, 200, n),
            "sa_source": ["raw"] * n,
            "source": ["gpr"] * n,
        }
    )
    out = build_backbone(panel)
    assert "unc_m" in out["variable"].unique()


# ===========================================================================
# build_composite
# ===========================================================================


def test_build_composite_adds_unc_pc_m():
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m", "epu_m", "rv_m"])
    out = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    assert "unc_pc_m" in out["variable"].unique()


def test_build_composite_unc_pc_m_rescaled_to_target():
    """unc_pc_m per country must have mean=100, sd=20 (ddof=0)."""
    panel = _make_monthly_proxy_panel(["USA", "DEU"], ["gpr_m", "epu_m", "rv_m"])
    out = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    unc = out[out["variable"] == "unc_pc_m"]
    for code in ["USA", "DEU"]:
        vals = unc[unc["code"] == code]["value"]
        assert abs(vals.mean() - TARGET_MEAN) < 1e-9
        assert abs(vals.std(ddof=0) - TARGET_SD) < 1e-9


def test_build_composite_quarterly_proxies_produce_unc_pc_q():
    rng = np.random.default_rng(10)
    n = 40
    dates = pd.date_range("2000-01-01", periods=n, freq="QS")
    rows = []
    f = rng.standard_normal(n)
    for proxy in ["gpr_q", "epu_q", "rv_q"]:
        vals = f + 0.2 * rng.standard_normal(n)
        for d, v in zip(dates, vals):
            rows.append(
                {"code": "USA", "date": d, "variable": proxy,
                 "value": float(v), "sa_source": "raw", "source": proxy}
            )
    panel = pd.DataFrame(rows)
    out = build_composite(panel, proxies=["gpr_q", "epu_q", "rv_q"])
    assert "unc_pc_q" in out["variable"].unique()
    assert "unc_pc_m" not in out["variable"].unique()


def test_build_composite_t1_tier_with_three_or_more_proxies():
    panel = _make_monthly_proxy_panel(
        ["USA"], ["gpr_m", "epu_m", "rv_m", "mpu_m"], n=40
    )
    out = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m", "mpu_m"])
    src = out[out["variable"] == "unc_pc_m"]["source"].iloc[0]
    assert "tier=T1" in src


def test_build_composite_t2_tier_with_two_proxies():
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m", "epu_m"], n=40)
    out = build_composite(panel, proxies=["gpr_m", "epu_m"])
    src = out[out["variable"] == "unc_pc_m"]["source"].iloc[0]
    assert "tier=T2" in src


def test_build_composite_t3_tier_single_proxy_fallback():
    """When only 1 proxy is available, tier=T3 and no PC1 is computed."""
    rng = np.random.default_rng(0)
    n = 40
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    rows = [
        {"code": "USA", "date": d, "variable": "gpr_m", "value": float(v),
         "sa_source": "raw", "source": "gpr_m"}
        for d, v in zip(dates, rng.standard_normal(n))
    ]
    panel = pd.DataFrame(rows)
    out = build_composite(panel, proxies=["gpr_m", "epu_m"])
    src = out[out["variable"] == "unc_pc_m"]["source"].iloc[0]
    assert "tier=T3" in src
    assert "single_proxy" in src


def test_build_composite_t3_no_overlap_fallback():
    """Two proxies present but disjoint time windows → no_overlap T3 fallback."""
    rng = np.random.default_rng(42)
    n = 60
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    rows = []
    # gpr_m in first 30 dates, epu_m in last 30 dates (no overlap)
    for d, v in zip(dates[:30], rng.standard_normal(30)):
        rows.append({"code": "USA", "date": d, "variable": "gpr_m",
                     "value": float(v), "sa_source": "raw", "source": "gpr_m"})
    for d, v in zip(dates[30:], rng.standard_normal(30)):
        rows.append({"code": "USA", "date": d, "variable": "epu_m",
                     "value": float(v), "sa_source": "raw", "source": "epu_m"})
    panel = pd.DataFrame(rows)
    out = build_composite(panel, proxies=["gpr_m", "epu_m"])
    unc = out[out["variable"] == "unc_pc_m"]
    assert len(unc) > 0
    src = unc["source"].iloc[0]
    assert "no_overlap" in src
    assert "tier=T3" in src


def test_build_composite_no_matching_proxies_returns_panel_unchanged():
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m"], n=30)
    out = build_composite(panel, proxies=["epu_m", "rv_m"])
    assert len(out) == len(panel)


def test_build_composite_too_few_obs_skips_country():
    rng = np.random.default_rng(0)
    n = MIN_OBS - 2
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    rows = [
        {"code": "USA", "date": d, "variable": "gpr_m", "value": float(v),
         "sa_source": "raw", "source": "gpr_m"}
        for d, v in zip(dates, rng.standard_normal(n))
    ]
    panel = pd.DataFrame(rows)
    out = build_composite(panel, proxies=["gpr_m", "epu_m"])
    assert "unc_pc_m" not in out["variable"].unique()


def test_build_composite_output_columns():
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m", "epu_m", "rv_m"])
    out = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    unc = out[out["variable"] == "unc_pc_m"]
    expected_cols = {"code", "date", "variable", "value", "sa_source", "source"}
    assert expected_cols.issubset(set(unc.columns))
    assert (unc["sa_source"] == "derived").all()


def test_build_composite_idempotent():
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m", "epu_m", "rv_m"])
    proxies = ["gpr_m", "epu_m", "rv_m"]
    out1 = build_composite(panel, proxies=proxies)
    out2 = build_composite(out1, proxies=proxies)
    cnt1 = (out1["variable"] == "unc_pc_m").sum()
    cnt2 = (out2["variable"] == "unc_pc_m").sum()
    assert cnt1 == cnt2


def test_build_composite_multiple_countries():
    panel = _make_monthly_proxy_panel(
        ["USA", "DEU", "GBR"], ["gpr_m", "epu_m", "rv_m"], n=40
    )
    out = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    unc = out[out["variable"] == "unc_pc_m"]
    assert set(unc["code"].unique()) == {"USA", "DEU", "GBR"}


def test_build_composite_extend_history_adds_rows():
    """With extend_history=True, pre-rich-window rows appear; False shrinks them."""
    rng = np.random.default_rng(42)
    # Long proxies (gpr, epu) available for 51 periods
    # wui_m (post-2008 proxy) available only for last 31 periods
    n_long = 51
    dates_all = pd.date_range("2000-01-01", periods=n_long, freq="MS")
    f = rng.standard_normal(n_long)
    rows = []
    for proxy in ["gpr_m", "epu_m"]:
        vals = f + 0.3 * rng.standard_normal(n_long)
        for d, v in zip(dates_all, vals):
            rows.append({"code": "USA", "date": d, "variable": proxy,
                         "value": float(v), "sa_source": "raw", "source": proxy})
    for proxy in ["wui_m"]:
        vals = f[20:] + 0.3 * rng.standard_normal(n_long - 20)
        for d, v in zip(dates_all[20:], vals):
            rows.append({"code": "USA", "date": d, "variable": proxy,
                         "value": float(v), "sa_source": "raw", "source": proxy})
    panel = pd.DataFrame(rows)
    proxies = ["gpr_m", "epu_m", "wui_m"]
    out_ext = build_composite(panel, proxies=proxies, extend_history=True)
    out_no = build_composite(panel, proxies=proxies, extend_history=False)
    cnt_ext = (out_ext["variable"] == "unc_pc_m").sum()
    cnt_no = (out_no["variable"] == "unc_pc_m").sum()
    assert cnt_ext > cnt_no


def test_build_composite_rich_segment_identical_with_and_without_extension():
    """The rich-window values must be bit-identical regardless of extend_history."""
    rng = np.random.default_rng(42)
    n_long = 51
    dates_all = pd.date_range("2000-01-01", periods=n_long, freq="MS")
    f = rng.standard_normal(n_long)
    rows = []
    for proxy in ["gpr_m", "epu_m"]:
        vals = f + 0.3 * rng.standard_normal(n_long)
        for d, v in zip(dates_all, vals):
            rows.append({"code": "USA", "date": d, "variable": proxy,
                         "value": float(v), "sa_source": "raw", "source": proxy})
    for proxy in ["wui_m"]:
        vals = f[20:] + 0.3 * rng.standard_normal(n_long - 20)
        for d, v in zip(dates_all[20:], vals):
            rows.append({"code": "USA", "date": d, "variable": proxy,
                         "value": float(v), "sa_source": "raw", "source": proxy})
    panel = pd.DataFrame(rows)
    proxies = ["gpr_m", "epu_m", "wui_m"]
    out_ext = build_composite(panel, proxies=proxies, extend_history=True)
    out_no = build_composite(panel, proxies=proxies, extend_history=False)
    vals_ext = (
        out_ext[out_ext["variable"] == "unc_pc_m"]
        .set_index("date")["value"]
        .rename("ext")
    )
    vals_no = (
        out_no[out_no["variable"] == "unc_pc_m"]
        .set_index("date")["value"]
        .rename("no")
    )
    common = vals_ext.index.intersection(vals_no.index)
    diff = (vals_ext.loc[common] - vals_no.loc[common]).abs().max()
    assert diff == 0.0  # bit-identical on overlapping dates


def test_build_composite_source_contains_proxy_names():
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m", "epu_m", "rv_m"])
    out = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    src = out[out["variable"] == "unc_pc_m"]["source"].iloc[0]
    assert "gpr_m" in src
    assert "epu_m" in src


# ===========================================================================
# build_innovation
# ===========================================================================


def test_build_innovation_bad_freq_raises_value_error():
    panel = pd.DataFrame(
        {
            "code": ["USA"],
            "date": [pd.Timestamp("2000-01-01")],
            "variable": ["unc_pc_m"],
            "value": [100.0],
            "sa_source": ["derived"],
            "source": ["test"],
        }
    )
    with pytest.raises(ValueError, match="freq"):
        build_innovation(panel, freq="x")


def test_build_innovation_missing_pc_var_returns_panel_unchanged():
    """No unc_pc_q in panel -> returned panel unchanged."""
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m"], n=30)
    out = build_innovation(panel, freq="q")
    assert len(out) == len(panel)
    assert "unc_innov_q" not in out["variable"].unique()


def test_build_innovation_adds_unc_innov_m():
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m", "epu_m", "rv_m"])
    panel = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    out = build_innovation(panel, freq="m")
    assert "unc_innov_m" in out["variable"].unique()


def test_build_innovation_standardized_mean_zero_sd_one():
    """Innovation must be standardized to mean~0, sd~1 per country."""
    panel = _make_monthly_proxy_panel(
        ["USA", "DEU"], ["gpr_m", "epu_m", "rv_m"], n=60
    )
    panel = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    out = build_innovation(panel, freq="m")
    innov = out[out["variable"] == "unc_innov_m"]
    for code in ["USA", "DEU"]:
        vals = innov[innov["code"] == code]["value"]
        assert abs(vals.mean()) < 1e-10
        assert abs(vals.std(ddof=0) - 1.0) < 1e-10


def test_build_innovation_output_columns():
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m", "epu_m", "rv_m"])
    panel = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    out = build_innovation(panel, freq="m")
    innov = out[out["variable"] == "unc_innov_m"]
    expected_cols = {"code", "date", "variable", "value", "sa_source", "source"}
    assert expected_cols.issubset(set(innov.columns))
    assert (innov["sa_source"] == "derived").all()


def test_build_innovation_source_carries_tier_tag():
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m", "epu_m", "rv_m"])
    panel = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    out = build_innovation(panel, freq="m")
    innov = out[out["variable"] == "unc_innov_m"]
    src = innov["source"].iloc[0]
    # Source should carry a tier tag (T1/T2/T3 or Tx as fallback)
    assert any(tag in src for tag in ("T1", "T2", "T3", "Tx"))
    assert "AR(2)" in src


def test_build_innovation_idempotent():
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m", "epu_m", "rv_m"])
    panel = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    out1 = build_innovation(panel, freq="m")
    out2 = build_innovation(out1, freq="m")
    cnt1 = (out1["variable"] == "unc_innov_m").sum()
    cnt2 = (out2["variable"] == "unc_innov_m").sum()
    assert cnt1 == cnt2


def test_build_innovation_first_two_dates_absent_due_to_ar2():
    """AR(2) loses the first 2 observations, so the innovation has 2 fewer rows
    than the underlying composite (after dropping NaN residuals)."""
    panel = _make_monthly_proxy_panel(["USA"], ["gpr_m", "epu_m", "rv_m"], n=50)
    panel = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    n_pc = (panel["variable"] == "unc_pc_m").sum()
    out = build_innovation(panel, freq="m")
    n_innov = (out["variable"] == "unc_innov_m").sum()
    # AR(2) drops first 2 pre-residual + residual from dropna → 2 fewer
    assert n_innov == n_pc - 2


def test_build_innovation_freq_q_works():
    """build_innovation should work with freq='q' on a quarterly composite."""
    rng = np.random.default_rng(5)
    n = 40
    dates = pd.date_range("2000-01-01", periods=n, freq="QS")
    rows = []
    f = rng.standard_normal(n)
    for proxy in ["gpr_q", "epu_q", "rv_q"]:
        vals = f + 0.2 * rng.standard_normal(n)
        for d, v in zip(dates, vals):
            rows.append(
                {"code": "USA", "date": d, "variable": proxy,
                 "value": float(v), "sa_source": "raw", "source": proxy}
            )
    panel = pd.DataFrame(rows)
    panel = build_composite(panel, proxies=["gpr_q", "epu_q", "rv_q"])
    out = build_innovation(panel, freq="q")
    assert "unc_innov_q" in out["variable"].unique()


def test_build_innovation_multiple_countries():
    panel = _make_monthly_proxy_panel(
        ["USA", "DEU", "GBR"], ["gpr_m", "epu_m", "rv_m"], n=50
    )
    panel = build_composite(panel, proxies=["gpr_m", "epu_m", "rv_m"])
    out = build_innovation(panel, freq="m")
    innov = out[out["variable"] == "unc_innov_m"]
    assert set(innov["code"].unique()) == {"USA", "DEU", "GBR"}


# ===========================================================================
# Additional edge-case tests to cover remaining branches
# ===========================================================================


def test_pc1_sign_anchor_falls_back_to_dominant_when_gpr_loading_small():
    """Line 103: When GPR loading < GPR_SIGN_FLOOR, sign anchors on dominant proxy.

    Construct a panel where gpr is noise (small loading) and epu/rv drive PC1.
    The sign of PC1 should still be meaningful (positively correlated with epu).
    """
    rng = np.random.default_rng(99)
    n = 30
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    f = rng.standard_normal(n)
    # gpr_m is pure noise (orthogonal to f -> loading below GPR_SIGN_FLOOR)
    wide = pd.DataFrame(
        {
            "gpr_m": rng.standard_normal(n),
            "epu_m": f + 0.01 * rng.standard_normal(n),
            "rv_m":  f + 0.01 * rng.standard_normal(n),
        },
        index=idx,
    )
    result = _pc1_from_wide(wide, "_m")
    assert result is not None
    pc1, cols = result
    # Sign should align with epu/rv (dominant proxies), not GPR
    corr_epu = np.corrcoef(pc1.values, wide["epu_m"].values)[0, 1]
    assert corr_epu > 0.9  # epu is the dominant proxy; must be positively correlated


def test_build_composite_all_proxies_have_too_few_nonnan_obs_skips():
    """Line 159: After keep_cols filter, if no columns survive, country is skipped.

    Set up two proxies where each appears but in disjoint, short windows
    (each below MIN_OBS). The wide pivot will show both but each column
    has too few valid observations.
    """
    rng = np.random.default_rng(42)
    n = MIN_OBS - 2  # too few per proxy
    dates1 = pd.date_range("2000-01-01", periods=n, freq="MS")
    dates2 = pd.date_range("2010-01-01", periods=n, freq="MS")
    rows = []
    for d, v in zip(dates1, rng.standard_normal(n)):
        rows.append({"code": "USA", "date": d, "variable": "gpr_m",
                     "value": float(v), "sa_source": "raw", "source": "gpr_m"})
    for d, v in zip(dates2, rng.standard_normal(n)):
        rows.append({"code": "USA", "date": d, "variable": "epu_m",
                     "value": float(v), "sa_source": "raw", "source": "epu_m"})
    panel = pd.DataFrame(rows)
    out = build_composite(panel, proxies=["gpr_m", "epu_m"])
    assert "unc_pc_m" not in out["variable"].unique()


def test_build_composite_no_overlap_fallback_top_proxy_also_too_short():
    """Line 184: In the no_overlap path, if best proxy also has < min_obs, skip.

    Both gpr_m and epu_m exist but with very few obs (< MIN_OBS) in
    disjoint windows. PC1 cannot be computed; no fallback single proxy
    meets the min_obs threshold either.
    """
    rng = np.random.default_rng(7)
    n = MIN_OBS - 5  # well below threshold
    dates1 = pd.date_range("2000-01-01", periods=n, freq="MS")
    dates2 = pd.date_range("2015-01-01", periods=n, freq="MS")
    rows = []
    for d, v in zip(dates1, rng.standard_normal(n)):
        rows.append({"code": "USA", "date": d, "variable": "gpr_m",
                     "value": float(v), "sa_source": "raw", "source": "gpr_m"})
    for d, v in zip(dates2, rng.standard_normal(n)):
        rows.append({"code": "USA", "date": d, "variable": "epu_m",
                     "value": float(v), "sa_source": "raw", "source": "epu_m"})
    panel = pd.DataFrame(rows)
    out = build_composite(panel, proxies=["gpr_m", "epu_m"])
    assert "unc_pc_m" not in out["variable"].unique()


def test_build_composite_extend_history_post_rich_tail_added():
    """Line 234: long-segment post-rich tail is appended when it exists.

    Build a setup where gpr/epu are available both before AND after
    the wui_m window (which is in the middle), to force the post tail.
    """
    rng = np.random.default_rng(100)
    n_total = 80
    dates_all = pd.date_range("2000-01-01", periods=n_total, freq="MS")
    f = rng.standard_normal(n_total)
    rows = []
    # gpr_m and epu_m (long proxies) for the entire span
    for proxy in ["gpr_m", "epu_m"]:
        vals = f + 0.3 * rng.standard_normal(n_total)
        for d, v in zip(dates_all, vals):
            rows.append({"code": "USA", "date": d, "variable": proxy,
                         "value": float(v), "sa_source": "raw", "source": proxy})
    # wui_m only in the middle 40 periods (indices 20..59)
    wui_vals = f[20:60] + 0.3 * rng.standard_normal(40)
    for d, v in zip(dates_all[20:60], wui_vals):
        rows.append({"code": "USA", "date": d, "variable": "wui_m",
                     "value": float(v), "sa_source": "raw", "source": "wui_m"})
    panel = pd.DataFrame(rows)
    proxies = ["gpr_m", "epu_m", "wui_m"]
    out_ext = build_composite(panel, proxies=proxies, extend_history=True)
    out_no = build_composite(panel, proxies=proxies, extend_history=False)
    cnt_ext = (out_ext["variable"] == "unc_pc_m").sum()
    cnt_no = (out_no["variable"] == "unc_pc_m").sum()
    # Extended history should cover more dates (pre + post)
    assert cnt_ext > cnt_no


def test_build_innovation_too_few_pc_obs_per_country_skips():
    """Line 306: build_innovation skips countries with < MIN_OBS unc_pc values."""
    n = MIN_OBS - 1  # 23 observations
    rng = np.random.default_rng(42)
    rows = []
    for d, v in zip(pd.date_range("2000-01-01", periods=n, freq="MS"),
                    rng.standard_normal(n)):
        rows.append({"code": "USA", "date": d, "variable": "unc_pc_m",
                     "value": float(v), "sa_source": "derived",
                     "source": "PC1(gpr_m; tier=T1; rescaled mean=100.0, sd=20.0)"})
    panel = pd.DataFrame(rows)
    out = build_innovation(panel, freq="m")
    assert "unc_innov_m" not in out["variable"].unique()


def test_build_innovation_constant_composite_produces_zero_residual_and_is_skipped():
    """Line 309: If residual std==0 (e.g., constant composite), country is skipped.

    A constant unc_pc_m series gives zero AR(2) residuals, so std==0 → skip.
    """
    n = 30
    rows = []
    for d in pd.date_range("2000-01-01", periods=n, freq="MS"):
        rows.append({"code": "USA", "date": d, "variable": "unc_pc_m",
                     "value": 100.0, "sa_source": "derived",
                     "source": "PC1(gpr_m; tier=T1; rescaled mean=100.0, sd=20.0)"})
    panel = pd.DataFrame(rows)
    out = build_innovation(panel, freq="m")
    assert "unc_innov_m" not in out["variable"].unique()


def test_build_innovation_no_valid_countries_returns_panel_unchanged():
    """Line 328: When no country produces a valid innovation, panel is returned as-is.

    Use a constant unc_pc_m for a single country so the innovation loop
    produces no rows, and verify the returned object has the same shape.
    """
    n = 30
    rows = []
    for d in pd.date_range("2000-01-01", periods=n, freq="MS"):
        rows.append({"code": "USA", "date": d, "variable": "unc_pc_m",
                     "value": 100.0, "sa_source": "derived",
                     "source": "PC1(gpr_m; tier=T1; rescaled mean=100.0, sd=20.0)"})
    panel = pd.DataFrame(rows)
    out = build_innovation(panel, freq="m")
    assert len(out) == len(panel)
