"""RTNOW: ``realtime_nowcast(as_of=...)`` must respect the information set.

A historical replay made "as of" a date may only use vintages published on or
before that date (the dataset a researcher had in hand that day, see
``VintagePanel.as_of``). The Banbura-Modugno news decomposition compares two
consecutive information sets Omega_{v-1} subset Omega_v (Banbura & Modugno 2010,
ECB WP 1189, sec. 2.3; Banbura, Giannone, Modugno & Reichlin 2013, ECB WP 1564,
eq. 8), so the default baseline is the last vintage strictly before the as_of
vintage, never one published later.

The central oracle is invariance: a replay with ``as_of`` on the full panel must
be identical, field by field, to running on the panel truncated at ``as_of``,
and must not move when values first published after ``as_of`` are perturbed.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.realtime import VintagePanel
from puremacro.nowcast.realtime_nowcast import RealtimeNowcastResult, realtime_nowcast

VINTAGES = pd.to_datetime(
    ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]
)
PLANTED = 555.0  # gdp(2020-06-01) as "revised" only from vintage v3 onward


def _rows(seed: int = 0, future_shift: float = 0.0, shift_after=None):
    """5 monthly vintages; vintage k publishes one more month than k-1.

    ``gdp`` lags the indicators by one month (ragged edge). A revision to
    gdp(2020-06-01) = ``PLANTED`` exists only in vintages v3 and v4.
    ``future_shift`` is added to every row whose vintage is > ``shift_after``
    (used to prove that data published after ``as_of`` is never read).
    """
    rng = np.random.default_rng(seed)
    T = 40
    dates = pd.date_range("2020-01-01", periods=T, freq="MS")
    f = np.cumsum(rng.normal(scale=0.3, size=T))
    truth = {
        v: 1.0 + (0.8 + 0.2 * i) * f + rng.normal(scale=0.2, size=T)
        for i, v in enumerate(["gdp", "ip", "emp"])
    }
    rows = []
    for k, vint in enumerate(VINTAGES):
        last = 34 + k
        for var, y in truth.items():
            for t in range(last + 1 - (1 if var == "gdp" else 0)):
                val = float(y[t])
                if var == "gdp" and t == 5 and k >= 3:
                    val = PLANTED
                # A small revision to ip(2020-03) in every later vintage, so
                # consecutive-vintage decompositions have a revision term.
                if var == "ip" and t == 2:
                    val += 0.05 * k
                if shift_after is not None and vint > pd.Timestamp(shift_after):
                    val += future_shift
                rows.append(dict(country="MEX", variable=var, date=dates[t],
                                 vintage=vint, value=val, provider="mock",
                                 series_id=var, units="rate"))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def vp() -> VintagePanel:
    return VintagePanel(_rows())


def _nowcast(panel, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return realtime_nowcast("MEX", panel=panel, method=kw.pop("method", "dfm"),
                                n_factors=1, **kw)


def _assert_news_equal(a, b):
    assert (a is None) == (b is None)
    if a is None:
        return
    assert a.forecast_old == pytest.approx(b.forecast_old, abs=1e-12, rel=0)
    assert a.forecast_new == pytest.approx(b.forecast_new, abs=1e-12, rel=0)
    assert a.revision == pytest.approx(b.revision, abs=1e-12, rel=0)
    pd.testing.assert_frame_equal(a.news_table.reset_index(drop=True),
                                  b.news_table.reset_index(drop=True))
    pd.testing.assert_frame_equal(a.revision_table.reset_index(drop=True),
                                  b.revision_table.reset_index(drop=True))


def _assert_results_equal(a: RealtimeNowcastResult, b: RealtimeNowcastResult):
    assert pd.Timestamp(a.latest_vintage) == pd.Timestamp(b.latest_vintage)
    assert a.nowcast == pytest.approx(b.nowcast, abs=1e-12, rel=0)
    assert a.forecast_sd == pytest.approx(b.forecast_sd, abs=1e-12, rel=0)
    _assert_news_equal(a.news_decomposition, b.news_decomposition)
    pd.testing.assert_frame_equal(a.vintage_history.reset_index(drop=True),
                                  b.vintage_history.reset_index(drop=True))
    assert repr(a.news_vs_noise_test) == repr(b.news_vs_noise_test)


# ---------------------------------------------------------------------------
# Default baseline vintage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3])
def test_default_previous_vintage_is_last_vintage_before_as_of(vp, k):
    """as_of = v_k -> the default news baseline is v_{k-1}, never a later one."""
    res = _nowcast(vp, as_of=VINTAGES[k])
    oracle = _nowcast(vp, as_of=VINTAGES[k], previous_vintage=VINTAGES[k - 1])
    assert pd.Timestamp(res.latest_vintage) == VINTAGES[k]
    _assert_news_equal(res.news_decomposition, oracle.news_decomposition)
    nd = res.news_decomposition
    # Omega_{v-1} subset Omega_v: the identity is exact and there is news.
    assert nd.decomposition_error < 1e-10
    assert len(nd.news_table) > 0
    assert abs(nd.revision) > 0.0


def test_no_future_value_in_news_tables(vp):
    """The planted gdp value exists only from v3; as_of=v1 must never show it."""
    res = _nowcast(vp, as_of=VINTAGES[1])
    nd = res.news_decomposition
    for tbl in (nd.revision_table, nd.news_table):
        for col in ("previous_val", "updated_val", "actual", "value"):
            if col in tbl.columns:
                assert not np.isclose(tbl[col].astype(float), PLANTED).any()


def test_as_of_second_to_last_vintage_is_not_degenerate(vp):
    """Old default used all_vintages[-2] == as_of vintage -> revision exactly 0."""
    res = _nowcast(vp, as_of=VINTAGES[3])
    nd = res.news_decomposition
    assert nd is not None
    assert nd.revision != 0.0
    assert len(nd.news_table) > 0
    oracle = _nowcast(vp, as_of=VINTAGES[3], previous_vintage=VINTAGES[2])
    _assert_news_equal(nd, oracle.news_decomposition)


def test_as_of_first_vintage_has_no_news(vp):
    res = _nowcast(vp, as_of=VINTAGES[0])
    assert pd.Timestamp(res.latest_vintage) == VINTAGES[0]
    assert res.news_decomposition is None
    assert res.vintage_history.empty


def test_as_of_none_keeps_latest_pair(vp):
    """Unchanged path: without as_of the baseline is the second-to-last vintage."""
    res = _nowcast(vp)
    oracle = _nowcast(vp, previous_vintage=VINTAGES[3])
    assert pd.Timestamp(res.latest_vintage) == VINTAGES[4]
    _assert_news_equal(res.news_decomposition, oracle.news_decomposition)


# ---------------------------------------------------------------------------
# Invariance: replay on full panel == run on the truncated panel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["dfm", "mfvar"])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_replay_equals_truncated_panel(vp, k, method):
    as_of = VINTAGES[k] + pd.Timedelta(days=10)  # between vintages
    full = _nowcast(vp, as_of=as_of, method=method, compute_news_vs_noise=True)
    trunc_panel = VintagePanel(vp.df[vp.df["vintage"] <= as_of].copy())
    trunc = _nowcast(trunc_panel, method=method, compute_news_vs_noise=True)
    _assert_results_equal(full, trunc)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_replay_invariant_to_future_data(k):
    """Perturbing every value first published after as_of changes nothing."""
    base = VintagePanel(_rows())
    moved = VintagePanel(_rows(future_shift=50.0, shift_after=VINTAGES[k]))
    a = _nowcast(base, as_of=VINTAGES[k], compute_news_vs_noise=True)
    b = _nowcast(moved, as_of=VINTAGES[k], compute_news_vs_noise=True)
    _assert_results_equal(a, b)


def _realtime_rows(n_vint: int = 24, seed: int = 7) -> pd.DataFrame:
    """Monthly vintages; vintage v publishes reference months up to v - 1 month
    and revises the last 3 published months, so first releases are observable
    and Mankiw-Shapiro has enough (preliminary, final) pairs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", "2023-12-01", freq="MS")
    vints = pd.date_range("2022-01-15", periods=n_vint, freq="MS") + pd.Timedelta(days=14)
    f = np.cumsum(rng.normal(scale=0.3, size=len(dates)))
    truth = {v: 1.0 + (0.8 + 0.2 * i) * f + rng.normal(scale=0.2, size=len(dates))
             for i, v in enumerate(["gdp", "ip", "emp"])}
    shocks = rng.normal(scale=0.1, size=(len(vints), len(dates), 3))
    rows = []
    for k, vint in enumerate(vints):
        for j, (var, y) in enumerate(truth.items()):
            for t, d in enumerate(dates):
                if d >= vint - pd.DateOffset(months=1):
                    break
                age = (vint.year - d.year) * 12 + vint.month - d.month
                val = float(y[t]) + (shocks[k, t, j] if age <= 3 else 0.0)
                rows.append(dict(country="MEX", variable=var, date=d, vintage=vint,
                                 value=val, provider="mock", series_id=var,
                                 units="rate"))
    return pd.DataFrame(rows)


def test_news_vs_noise_uses_only_vintages_up_to_as_of():
    """Mankiw-Shapiro on a replay equals the test on the truncated panel,
    and differs from the full-panel test (which uses later editions)."""
    base = VintagePanel(_realtime_rows())
    vints = sorted(base.df["vintage"].unique())
    as_of = pd.Timestamp(vints[11])
    trunc = VintagePanel(base.df[base.df["vintage"] <= as_of].copy())
    expected = trunc.news_or_noise("MEX", "gdp")
    full_expected = base.news_or_noise("MEX", "gdp")
    assert repr(expected) != repr(full_expected)

    res = _nowcast(base, as_of=as_of, compute_news_vs_noise=True)
    assert res.news_vs_noise_test is not None
    assert repr(res.news_vs_noise_test) == repr(expected)
    full = _nowcast(base, compute_news_vs_noise=True)
    assert repr(full.news_vs_noise_test) == repr(full_expected)


# ---------------------------------------------------------------------------
# Explicit previous_vintage must lie in the past of the as_of vintage
# ---------------------------------------------------------------------------
def test_explicit_previous_vintage_after_as_of_raises(vp):
    with pytest.raises(ValueError, match="after the information cutoff"):
        _nowcast(vp, as_of=VINTAGES[1], previous_vintage=VINTAGES[3])


def test_explicit_previous_vintage_not_before_current_raises(vp):
    # Same information set as the current vintage -> news identically zero.
    with pytest.raises(ValueError, match="strictly before"):
        _nowcast(vp, as_of=VINTAGES[1], previous_vintage=VINTAGES[1])
    with pytest.raises(ValueError, match="strictly before"):
        _nowcast(vp, as_of=VINTAGES[1] + pd.Timedelta(days=5),
                 previous_vintage=VINTAGES[1] + pd.Timedelta(days=2))
    with pytest.raises(ValueError, match="after the information cutoff"):
        _nowcast(vp, previous_vintage=VINTAGES[4] + pd.Timedelta(days=1))


def test_explicit_previous_vintage_between_vintages_uses_that_information_set(vp):
    """A date between v0 and v1 means 'what was known then' = Omega_{v0}."""
    a = _nowcast(vp, as_of=VINTAGES[2], previous_vintage=VINTAGES[0] + pd.Timedelta(days=3))
    b = _nowcast(vp, as_of=VINTAGES[2], previous_vintage=VINTAGES[0])
    _assert_news_equal(a.news_decomposition, b.news_decomposition)


# ---------------------------------------------------------------------------
# vintage_history: only past vintages, honestly labelled
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["dfm", "mfvar"])
def test_vintage_history_only_past_and_honest_columns(vp, method):
    as_of = VINTAGES[2]
    res = _nowcast(vp, as_of=as_of, method=method)
    vh = res.vintage_history
    assert list(vh.columns) == ["vintage", "last_observed_period", "last_observed_value"]
    assert "nowcast" not in vh.columns
    assert len(vh) == 3
    assert pd.to_datetime(vh["vintage"]).max() <= as_of
    # Each value is the last non-missing gdp observation of that vintage.
    for _, r in vh.iterrows():
        sl = vp.as_of(r["vintage"]).xs("MEX")["gdp"].dropna()
        assert r["last_observed_period"] == sl.index[-1]
        assert r["last_observed_value"] == sl.iloc[-1]


def test_vintage_history_full_panel_without_as_of(vp):
    res = _nowcast(vp)
    vh = res.vintage_history
    assert len(vh) == 5
    assert pd.to_datetime(vh["vintage"]).max() == VINTAGES[4]


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def test_summary_and_plot_use_honest_labels(vp):
    res = _nowcast(vp, as_of=VINTAGES[2])
    s = res.summary()
    assert "Nowcast Progression" not in s
    assert "last observed" in s.lower()
    assert "(< 1e-10)" in s  # the identity holds for a valid baseline
    fig = res.plot()
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert not any(lbl == "Nowcast Path" for lbl in labels)
    plt.close(fig)


def test_summary_prints_baseline_and_consistent_update(vp):
    """forecast_old + revision = forecast_new is visible; baseline is named."""
    res = _nowcast(vp, as_of=VINTAGES[2])
    nd = res.news_decomposition
    assert pd.Timestamp(res.previous_vintage) == VINTAGES[1]
    s = res.summary()
    assert f"Baseline Vintage (v-1)      : {res.previous_vintage}" in s
    assert f"Updated Nowcast (v)         : {nd.forecast_new:+.4f}" in s
    # gdp is unobserved at the target period here -> nowcast == forecast_new
    assert res.nowcast == pytest.approx(nd.forecast_new, abs=1e-12)
    assert "Point Nowcast is the published figure" not in s
    # as_of on the first vintage: no decomposition, no baseline
    assert _nowcast(vp, as_of=VINTAGES[0]).previous_vintage is None


def test_summary_notes_observed_target(vp):
    """If the target period is already published, the point nowcast is the
    datum while the news refers to the model fit: the summary says so."""
    res = _nowcast(vp, as_of=VINTAGES[2])
    per = res.vintage_history["last_observed_period"].iloc[-1]
    res_obs = _nowcast(vp, as_of=VINTAGES[2], target_period=per)
    assert res_obs.nowcast != pytest.approx(res_obs.news_decomposition.forecast_new, abs=1e-8)
    assert "Point Nowcast is the published figure" in res_obs.summary()


def test_summary_does_not_claim_identity_when_it_fails(vp):
    res = _nowcast(vp, as_of=VINTAGES[2])
    fake = SimpleNamespace(forecast_old=1.0, forecast_new=1.07, revision=0.07,
                           impact_releases={}, impact_revisions={},
                           decomposition_error=0.07)
    bad = RealtimeNowcastResult(**{**res.__dict__, "news_decomposition": fake})
    s = bad.summary()
    assert "(< 1e-10)" not in s
    assert "7.00e-02" in s


def test_wide_frame_with_as_of_raises():
    df_wide = pd.DataFrame({"gdp": [1.0, 2.0, 3.0], "ip": [1.0, 2.0, 2.5]},
                           index=pd.date_range("2024-01-01", periods=3, freq="MS"))
    with pytest.raises(ValueError, match="vintage"):
        realtime_nowcast("MEX", panel=df_wide, target_variable="gdp",
                         as_of="2024-02-01")
    with pytest.raises(ValueError, match="vintage"):
        realtime_nowcast("MEX", panel=df_wide, target_variable="gdp",
                         previous_vintage="2024-02-01")
