"""Regression tests for review defect GK (2026-09-30 notebook review).

Three contracts are pinned here:

1. ``GertlerKaradiResult.to_frame()`` returns **level** deviations
   ``x_t - x_ss`` (the model is linearised in levels), and
   ``to_frame(units="pct")`` returns percent deviations
   ``100 * (x_t - x_ss) / x_ss``.  The course notebook T02_F multiplied the
   level frame by 100 and labelled it "% dev.", printing capital at -28.66 %
   on impact when the true figure is -5.06 %.
2. ``gk2015_surprise`` returns the surprise in the futures-implied *rate*
   (Gertler-Karadi 2015, NBER WP 20224, eq. (19) and footnote 6), positive
   for a tightening, whether the inputs are quoted as implied rates
   (``quote="rate"``, the default) or as CME prices ``100 - rate``
   (``quote="price"``).
3. ``aggregate_to_period(..., method="gk2015")`` reproduces the monthly
   averaging of GK (2015) footnote 11: cumulate the FOMC-day surprises into a
   daily series, average it within each month, first-difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.gertler_karadi import solve_gertler_karadi, solve_steady_state
from puremacro.hfi.surprises import aggregate_to_period, gk2015_surprise


# ---------------------------------------------------------------------------
# 1. GertlerKaradiResult.to_frame units
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gk_default():
    return solve_gertler_karadi()          # occbin, -5% capital quality, H=40


@pytest.fixture(scope="module")
def gk_klein():
    return solve_gertler_karadi(method="klein")


def test_to_frame_default_is_level_deviation_and_unchanged(gk_default):
    df = gk_default.to_frame()
    pd.testing.assert_frame_equal(df, gk_default.irf)
    pd.testing.assert_frame_equal(gk_default.to_frame(units="level"), gk_default.irf)
    assert df.attrs.get("units") == "level"


def test_level_frame_satisfies_linear_resource_constraint(gk_default):
    """Y = C + I + g*Y_ss is linear, so its level deviations add up exactly.

    Read as fractional deviations the same identity misses by ~0.09, which is
    what shows the frame is in levels.
    """
    df = gk_default.to_frame()
    assert np.max(np.abs(df["Y"] - df["C"] - df["I"])) < 1e-10
    pct = gk_default.to_frame(units="pct")
    assert np.max(np.abs(pct["Y"] - pct["C"] - pct["I"])) > 1.0


def test_pct_frame_is_percent_deviation_from_steady_state(gk_default):
    ss = gk_default.steady_state
    lvl = gk_default.to_frame()
    pct = gk_default.to_frame(units="pct")
    assert pct.attrs.get("units") == "pct"
    assert list(pct.columns) == list(lvl.columns)
    for v in ["Y", "C", "I", "K", "N", "Q", "phi", "prem", "R", "Rn", "Pi"]:
        np.testing.assert_allclose(pct[v].to_numpy(),
                                   100.0 * lvl[v].to_numpy() / ss[v], rtol=1e-12)
    # Canonical numbers (occbin, -5% capital quality): the notebook printed the
    # level deviation times 100 (-28.66 for K, -67.88 for N, -1.01 for I).
    assert pct["K"].iloc[0] == pytest.approx(-5.0640, abs=5e-4)
    assert pct["K"].min() == pytest.approx(-13.4946, abs=5e-4)
    assert pct["N"].iloc[0] == pytest.approx(-49.1531, abs=5e-4)
    assert pct["I"].iloc[0] == pytest.approx(-7.1027, abs=5e-4)
    assert pct["Y"].iloc[0] == pytest.approx(-1.5964, abs=5e-4)
    assert pct["C"].iloc[0] == pytest.approx(-0.6509, abs=5e-4)
    assert 100 * lvl["K"].iloc[0] == pytest.approx(-28.6625, abs=5e-4)
    # Q_ss = 1, so its percent deviation coincides with 100 x level deviation.
    np.testing.assert_allclose(pct["Q"], 100 * lvl["Q"], rtol=1e-12)
    # Weighted by steady-state shares, the resource constraint holds in pct.
    lhs = ss["Y"] * pct["Y"]
    rhs = ss["C"] * pct["C"] + ss["I"] * pct["I"]
    assert np.max(np.abs(lhs - rhs)) < 1e-8


def test_pct_frame_zero_steady_state_is_nan(gk_default):
    """psi has a zero steady state: a percent deviation is undefined."""
    assert gk_default.steady_state["psi"] == 0.0
    pct = gk_default.to_frame(units="pct")
    assert pct["psi"].isna().all()
    assert "psi" in pct.attrs.get("pct_undefined", [])


def test_pct_frame_klein(gk_klein):
    pct = gk_klein.to_frame(units="pct")
    assert pct["K"].iloc[0] == pytest.approx(-5.1441, abs=5e-4)
    assert pct["N"].iloc[0] == pytest.approx(-61.7177, abs=5e-4)


def test_to_frame_rejects_unknown_units(gk_default):
    with pytest.raises(ValueError, match="units"):
        gk_default.to_frame(units="log")


def test_exports_and_plot_accept_units(gk_default):
    md = gk_default.to_markdown(units="pct")
    assert isinstance(md, str) and len(md) > 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = gk_default.plot(variables=["Y", "K"], units="pct", style="default")
    try:
        ax = fig.axes[0]
        np.testing.assert_allclose(ax.lines[0].get_ydata(),
                                   gk_default.to_frame(units="pct")["Y"].to_numpy())
        assert "%" in ax.get_ylabel()
    finally:
        plt.close(fig)


def test_summary_states_level_units(gk_default):
    text = gk_default.summary()
    assert "x_t - x_ss" in text


def test_steady_state_matches_solver(gk_default):
    ss = solve_steady_state()
    assert gk_default.steady_state["K"] == pytest.approx(ss["K"])


# ---------------------------------------------------------------------------
# 2. gk2015_surprise quote convention and sign
# ---------------------------------------------------------------------------

def test_gk2015_surprise_rate_quote_is_tightening_positive():
    # The current-month implied rate rises 25 bp in the window with 15 of 30
    # days left; GK (2015) fn. 6 scales by T/(T - t) = 30/15, so the
    # target-rate surprise is 0.25 * 2 = +0.5 pp (a tightening).
    s = gk2015_surprise([5.00], [5.25], [15], 30, quote="rate")
    np.testing.assert_allclose(s, [0.5])
    # Default quote is "rate" (backward compatible).
    np.testing.assert_allclose(gk2015_surprise([5.00], [5.25], [15], 30), [0.5])


def test_gk2015_surprise_price_quote_negates_price_change():
    # CME quotes 100 - rate: the same hike is a price fall 95.00 -> 94.75.
    s_price = gk2015_surprise([95.00], [94.75], [15], 30, quote="price")
    np.testing.assert_allclose(s_price, [0.5])
    # Without the quote flag the price change comes out sign-reversed.
    np.testing.assert_allclose(gk2015_surprise([95.00], [94.75], [15], 30), [-0.5])


def test_gk2015_surprise_price_and_rate_agree():
    rng = np.random.default_rng(0)
    r_pre = 5.0 + rng.standard_normal(50)
    r_post = r_pre + 0.1 * rng.standard_normal(50)
    rem = rng.integers(1, 29, size=50)
    M = np.full(50, 30)
    a = gk2015_surprise(r_pre, r_post, rem, M, quote="rate")
    b = gk2015_surprise(100 - r_pre, 100 - r_post, rem, M, quote="price")
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_gk2015_surprise_scaling_is_T_over_T_minus_t():
    """Footnote 6: factor T/(T - t), t = days elapsed *before* the meeting.

    A meeting on day k of a T-day month has t = k - 1 days elapsed and
    T - t = T - k + 1 days remaining including the meeting day.
    """
    T = 31
    for k in (1, 10, 31):
        rem = T - k + 1
        s = gk2015_surprise([1.0], [1.1], [rem], T)
        np.testing.assert_allclose(s, [0.1 * T / (T - (k - 1))])


def test_gk2015_surprise_rejects_bad_quote_and_day_counts():
    with pytest.raises(ValueError, match="quote"):
        gk2015_surprise([5.0], [5.25], [15], 30, quote="yield")
    with pytest.raises(ValueError, match="days_remaining_in_month"):
        gk2015_surprise([5.0], [5.25], [31], 30)


def test_price_based_surprise_recovers_planted_shock_sign():
    """With quote='price' the instrument correlates positively with the
    planted tightening shock; mislabelled prices flip it."""
    rng = np.random.default_rng(1)
    eps = rng.standard_normal(400)
    rem = rng.integers(3, 29, size=400)
    M = 30
    target = 0.25 * eps
    implied_move = target * rem / M          # only the rest of the month moves
    price_pre = 100 - 5.0 * np.ones(400)
    price_post = price_pre - implied_move
    z = gk2015_surprise(price_pre, price_post, rem, M, quote="price")
    np.testing.assert_allclose(z, target, atol=1e-12)
    z_bad = gk2015_surprise(price_pre, price_post, rem, M)
    assert np.corrcoef(z, eps)[0, 1] > 0.99
    assert np.corrcoef(z_bad, eps)[0, 1] < -0.99


# ---------------------------------------------------------------------------
# 3. GK (2015) footnote-11 monthly averaging
# ---------------------------------------------------------------------------

def _gk_footnote11_bruteforce(surprises, dates, freq="M"):
    """Literal transcription of GK (2015) footnote 11, second recipe:
    cumulative daily surprise series -> period averages -> first difference."""
    dates = pd.to_datetime(dates).normalize()
    first = dates.min().to_period(freq)
    last = dates.max().to_period(freq) + 1
    days = pd.date_range(first.start_time, last.end_time.normalize(), freq="D")
    daily = pd.Series(0.0, index=days)
    for s, d in zip(surprises, dates):
        daily.loc[d] += s
    cum = daily.cumsum()
    avg = cum.groupby(cum.index.to_period(freq)).mean()
    return avg.diff().fillna(avg.iloc[0])


@pytest.mark.parametrize("freq", ["M", "Q"])
def test_aggregate_gk2015_matches_footnote11_bruteforce(freq):
    rng = np.random.default_rng(3)
    dates = pd.to_datetime(["2001-01-03", "2001-01-31", "2001-03-20",
                            "2001-05-15", "2001-05-16", "2001-08-21",
                            "2001-09-17", "2001-12-11"])
    s = rng.standard_normal(len(dates))
    got = aggregate_to_period(s, dates, freq=freq, method="gk2015")
    ref = _gk_footnote11_bruteforce(s, dates, freq=freq)
    assert list(got.index) == list(ref.index)
    np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), atol=1e-12)
    # The scheme only moves surprise mass between adjacent periods.
    assert got.sum() == pytest.approx(s.sum())


def test_aggregate_gk2015_hand_weights():
    # Jan 1 (k=1 of 31): all of it lands in January.
    out = aggregate_to_period([1.0], pd.to_datetime(["2001-01-01"]), method="gk2015")
    assert out.loc["2001-01"] == pytest.approx(1.0)
    assert out.loc["2001-02"] == pytest.approx(0.0)
    # Jan 31 (k=31 of 31): 1/31 in January, 30/31 carried into February.
    out = aggregate_to_period([1.0], pd.to_datetime(["2001-01-31"]), method="gk2015")
    assert out.loc["2001-01"] == pytest.approx(1 / 31)
    assert out.loc["2001-02"] == pytest.approx(30 / 31)
    # Feb 15 2001 (k=15 of 28): 14/28 in February, 14/28 in March.
    out = aggregate_to_period([2.0], pd.to_datetime(["2001-02-15"]), method="gk2015")
    assert out.loc["2001-02"] == pytest.approx(2.0 * 14 / 28)
    assert out.loc["2001-03"] == pytest.approx(2.0 * 14 / 28)


def test_aggregate_default_is_still_sum():
    s = np.array([0.10, -0.05, 0.20])
    dates = pd.to_datetime(["2020-01-15", "2020-01-29", "2020-02-12"])
    a = aggregate_to_period(s, dates, freq="M")
    b = aggregate_to_period(s, dates, freq="M", method="sum")
    pd.testing.assert_series_equal(a, b)
    assert a.loc["2020-01"] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="method"):
        aggregate_to_period(s, dates, method="mean")


# ---------------------------------------------------------------------------
# 4. Catalog loader pass-through
# ---------------------------------------------------------------------------

def test_catalog_loader_quote_and_aggregation():
    from puremacro.instruments._catalog import _load_gk2015_ffr_surprise

    dates = pd.to_datetime(["2001-01-03", "2001-01-31"])
    kw = dict(days_remaining_in_month=[29, 1], dates=dates, days_in_month=31)
    by_rate = _load_gk2015_ffr_surprise(ff_futures_pre=[6.50, 6.00],
                                        ff_futures_post=[6.00, 6.00 - 0.5 / 31],
                                        **kw)
    by_price = _load_gk2015_ffr_surprise(ff_futures_pre=[93.50, 94.00],
                                         ff_futures_post=[94.00, 94.00 + 0.5 / 31],
                                         quote="price", **kw)
    pd.testing.assert_series_equal(by_rate.series, by_price.series)
    assert by_rate.series.loc["2001-01"] == pytest.approx(-0.5 * 31 / 29 - 0.5)
    assert by_price.metadata["quote"] == "price"
    assert by_rate.metadata["quote"] == "rate"
    assert by_rate.metadata["aggregation"] == "sum"
    gk = _load_gk2015_ffr_surprise(ff_futures_pre=[6.50, 6.00],
                                   ff_futures_post=[6.00, 6.00 - 0.5 / 31],
                                   aggregation="gk2015", **kw)
    assert gk.metadata["aggregation"] == "gk2015"
    assert gk.series.sum() == pytest.approx(by_rate.series.sum())


# ---------------------------------------------------------------------------
# 5. The shipped example: the instrument must move the policy rate
# ---------------------------------------------------------------------------

def test_example_instrument_moves_policy_rate():
    from puremacro.examples import hfi_gertler_karadi as ex

    out = ex.run_pipeline(n_boot=50)
    res = out["result"]
    ffr = out["var_names"].index("FFR")
    # Strong instrument, positively correlated with the planted MP shock
    # (the window adds equal-variance unrelated news, so corr is ~0.7).
    assert res.first_stage_F > 10.0
    assert out["corr_with_true_shock"] > 0.6
    # The identified shock raises the policy rate on impact ...
    assert res.irf_point[0, ffr, 0] > 0.0
    # ... and recovers the planted impact column up to scale.
    b_true = out["b_true"]
    b_hat = res.B[:, 0]
    cos = b_hat @ b_true / (np.linalg.norm(b_hat) * np.linalg.norm(b_true))
    assert cos > 0.85
    # Feeding the prices without quote="price" reverses the instrument.
    assert out["corr_with_true_shock_if_misquoted"] == pytest.approx(
        -out["corr_with_true_shock"])
