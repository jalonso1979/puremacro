"""Regression tests for the NOWCAST fixes (notebook review 2026-09-30).

1. RealtimeNowcastResult.plot_fan_chart drew an invented history
   (nowcast - 0.2, - 0.1, + 0.05) and an invented path (x1.01, x1.02; s.d.
   x1.2, x1.4). The fan now comes from the DFM's state space and the history
   is the published data.
2. NewsDecompositionResult.plot drew levels from 0, so impacts of ~0.1 on a
   level of ~100 were invisible. It now draws the update relative to the
   previous nowcast.
3. PITUniformityResult.summary said "PASS: ... well-calibrated"; plot ignored
   title= when ax was passed.
4. QNAVintagePanel.revision_stats used b_0 = -1 as the noise null. Under
   classical noise b_0 = -Var(v)/Var(y_0) lies in (-1, 0); the noise null is
   b_T = 0 on the final release.
5. fontweight="semibold" made matplotlib log "findfont: Failed to find font
   weight semibold" under the notebook font.

Every panel is simulated here; nothing is fetched.
"""
from __future__ import annotations

import dataclasses
import inspect

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.fetch import QNAVintagePanel
from puremacro.fetch.realtime import VintagePanel
from puremacro.nowcast import pit_uniformity_test, realtime_nowcast
from puremacro.vintages import mankiw_shapiro


# ----------------------------------------------------------------------------
# Fixtures: a two-vintage ragged-edge panel like notebook 59's
# ----------------------------------------------------------------------------

V1, V2 = pd.Timestamp("2025-01-15"), pd.Timestamp("2025-02-15")
PUB_LAG = {"gdp": 2, "activity": 1, "ip": 1, "cpi": 0}


def _ragged_panel(seed: int = 42) -> VintagePanel:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    f = np.cumsum(rng.normal(scale=0.25, size=len(dates)))
    rows = []
    for var, lag in PUB_LAG.items():
        load = rng.uniform(0.7, 1.3)
        y = 100.0 + 1.5 * f * load + rng.normal(scale=0.15, size=len(dates))
        for vint in (V1, V2):
            last = (vint.to_period("M") - 1 - lag).to_timestamp()
            for t, d in enumerate(dates):
                if d > last:
                    continue
                val = float(y[t])
                if vint == V2 and var == "activity" and t == 20:
                    val += 0.45  # a revision of an old month
                rows.append(dict(country="MEX", variable=var, date=d, vintage=vint,
                                 value=val, provider="sim", series_id=var))
    return VintagePanel(pd.DataFrame(rows))


@pytest.fixture(scope="module")
def nowcast_result():
    vp = _ragged_panel()
    res = realtime_nowcast(country="MEX", panel=vp, method="dfm", n_factors=1)
    return vp, res


# ----------------------------------------------------------------------------
# 1. Fan chart: observed history + state-space variances
# ----------------------------------------------------------------------------

def _hand_fan(res, horizon):
    """The fan computed independently from the fitted DFM's matrices."""
    fit = res.model_result
    j = list(fit.columns).index("gdp")
    m = fit.A.shape[0]
    lam = np.zeros(m)
    lam[: fit.n_factors] = fit.loadings[j]
    a_sm = fit.smoother_out["a_smooth"]
    P_sm = fit.smoother_out["P_smooth"]
    obs = res.observed_panel["gdp"].to_numpy()
    last_pub = int(np.flatnonzero(~np.isnan(obs))[-1])
    means, sds = [], []
    for t in range(last_pub + 1, len(a_sm)):
        means.append(fit.means[j] + fit.stds[j] * lam @ a_sm[t])
        sds.append(fit.stds[j] * np.sqrt(lam @ P_sm[t] @ lam + fit.H[j, j]))
    a, P = a_sm[-1], P_sm[-1]
    for _ in range(horizon):
        a = fit.A @ a
        P = fit.A @ P @ fit.A.T + fit.Q
        means.append(fit.means[j] + fit.stds[j] * lam @ a)
        sds.append(fit.stds[j] * np.sqrt(lam @ P @ lam + fit.H[j, j]))
    return np.array(means), np.array(sds), last_pub


def test_fan_chart_is_model_based_not_invented(nowcast_result):
    vp, res = nowcast_result
    fc = res.fan_chart(horizon=3)
    means, sds, last_pub = _hand_fan(res, 3)

    np.testing.assert_allclose(fc.forecast_mean.to_numpy(), means, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fc.quantiles["upper_90%"].to_numpy() - means,
                               1.6448536269514722 * sds, rtol=0, atol=1e-9)

    # The in-sample part starts after the last published gdp month and holds the nowcast
    idx = res.observed_panel.index
    assert fc.forecast_mean.index[0] == idx[last_pub + 1]
    assert fc.forecast_mean.loc[res.target_period] == pytest.approx(res.nowcast, abs=1e-10)
    sd_at_target = sds[list(fc.forecast_mean.index).index(res.target_period)]
    assert sd_at_target == pytest.approx(res.forecast_sd, abs=1e-10)

    # Beyond the panel: the model's own forecast path, dated at the panel frequency
    beyond = fc.forecast_mean.index[-3:]
    assert list(beyond) == list(pd.date_range(idx[-1], periods=4, freq="MS")[1:])
    pred = res.model_result.predict(steps=3)["gdp"].to_numpy()
    np.testing.assert_allclose(fc.forecast_mean.to_numpy()[-3:], pred, rtol=0, atol=1e-10)
    assert np.all(np.diff(sds[-4:]) > 0), "the fan widens with the horizon"

    # The old invented path is gone
    assert not np.isclose(fc.forecast_mean.iloc[-2], res.nowcast * 1.01)
    assert not np.isclose(sds[-1], res.forecast_sd * 1.4)

    # History = the gdp values published in the current vintage, nothing else
    published = vp.as_of(V2).xs("MEX")["gdp"].dropna()
    pd.testing.assert_series_equal(fc.history, published.iloc[-12:], check_names=False,
                                   check_freq=False, check_index_type=False)


def test_plot_fan_chart_draws_published_history_and_model_path(nowcast_result):
    _, res = nowcast_result
    fig = res.plot_fan_chart(horizon=2)
    ax = fig.axes[0]
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    hist = lines["Historical Data"]
    published = res.observed_panel["gdp"].dropna().iloc[-12:]
    np.testing.assert_allclose(np.asarray(hist.get_ydata(), float), published.to_numpy())
    central = np.asarray(lines["Central Projection"].get_ydata(), float)
    fc = res.fan_chart(horizon=2)
    np.testing.assert_allclose(central, [published.iloc[-1], *fc.forecast_mean.to_numpy()])
    assert ax.get_ylabel() == "gdp"
    plt.close(fig)


def test_fan_chart_follows_the_style_colours(nowcast_result):
    """History in the style's foreground colour; any matplotlib colour as palette."""
    from matplotlib.colors import to_rgba
    _, res = nowcast_result
    with plt.rc_context({"text.color": "#dddddd"}):
        fig = res.plot_fan_chart(palette="#888888")
        ax = fig.axes[0]
        lines = {ln.get_label(): ln for ln in ax.get_lines()}
        assert to_rgba(lines["Historical Data"].get_color()) == to_rgba("#dddddd")
        assert to_rgba(lines["Central Projection"].get_color()) == to_rgba("#888888")
        plt.close(fig)


def test_fan_chart_refuses_without_state_space(nowcast_result):
    _, res = nowcast_result
    no_model = dataclasses.replace(res, model_result=None, method="mfvar")
    with pytest.raises(ValueError, match="state space"):
        no_model.plot_fan_chart()
    with pytest.raises(ValueError, match="horizon"):
        res.fan_chart(horizon=-1)


# ----------------------------------------------------------------------------
# 2. News waterfall relative to the previous nowcast
# ----------------------------------------------------------------------------

def test_news_plot_draws_impacts_relative_to_previous_nowcast(nowcast_result):
    _, res = nowcast_result
    nd = res.news_decomposition
    fig = nd.plot()
    ax = fig.axes[0]
    bars = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]

    impacts = {}
    for s, v in nd.impact_releases.items():
        impacts[s] = impacts.get(s, 0.0) + v
    for s, v in nd.impact_revisions.items():
        impacts[f"{s} (rev)"] = impacts.get(f"{s} (rev)", 0.0) + v
    widths = np.array([b.get_width() for b in bars])
    lefts = np.array([b.get_x() for b in bars])
    expected = np.array([*impacts.values(), nd.revision])
    np.testing.assert_allclose(widths, expected, atol=1e-12)
    # Each impact bar starts where the previous ended; the total starts at 0
    np.testing.assert_allclose(lefts[:-1], np.concatenate([[0.0], np.cumsum(expected[:-2])]), atol=1e-12)
    assert lefts[-1] == 0.0
    assert lefts[-2] + widths[-2] == pytest.approx(nd.revision, abs=1e-10)

    # The axis is a change axis: it holds 0 and the revision, not the ~100 level
    lo, hi = ax.get_xlim()
    assert lo <= min(0.0, nd.revision) and hi >= max(0.0, nd.revision)
    assert hi - lo < 1.0 < abs(nd.forecast_old)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[-1] == "Total revision"
    plt.close(fig)

    fig, ax = plt.subplots()
    nd.plot(ax=ax, title="custom")
    assert "custom" in {ax.get_title(loc) for loc in ("left", "center", "right")}
    plt.close(fig)


def test_news_summary_reports_identity_honestly(nowcast_result):
    _, res = nowcast_result
    nd = res.news_decomposition
    assert "(< 1e-10)" in nd.summary()
    broken = dataclasses.replace(nd, decomposition_error=1e-3)
    assert "IDENTITY FAILS" in broken.summary()


# ----------------------------------------------------------------------------
# 3. PIT summary wording and title=
# ----------------------------------------------------------------------------

def _gaussian_forecasts(bias=0.0, seed=0, T=60):
    g = np.random.default_rng(seed)
    mu = g.normal(2.0, 0.5, T)
    sd = g.uniform(0.4, 0.8, T)
    y = mu + bias * sd + sd * g.normal(size=T)
    return y, mu, sd


def test_pit_summary_says_cannot_reject_not_pass():
    y, mu, sd = _gaussian_forecasts()
    res = pit_uniformity_test(y, mu=mu, sigma=sd)
    assert res.is_uniform
    text = res.summary()
    assert "Cannot reject uniformity or independence" in text
    for overclaim in ("PASS", "well-calibrated", "satisfies both", "FAIL"):
        assert overclaim not in text

    y, mu, sd = _gaussian_forecasts(bias=1.0)
    bad = pit_uniformity_test(y, mu=mu, sigma=sd)
    assert not bad.is_uniform
    assert "Rejects i.i.d. uniform PITs" in bad.summary()


def test_pit_alpha_is_used_in_verdicts():
    y, mu, sd = _gaussian_forecasts(bias=0.3, seed=3)
    r05 = pit_uniformity_test(y, mu=mu, sigma=sd, alpha=0.05)
    r50 = pit_uniformity_test(y, mu=mu, sigma=sd, alpha=0.5)
    assert r05.alpha == 0.05 and r50.alpha == 0.5
    assert "at 50%" in r50.summary()
    p = r50.lr_pvalue
    assert r50.to_frame().loc[0, "Verdict"] == ("Reject" if p <= 0.5 else "Fail to Reject")


def test_pit_plot_honours_title_with_ax():
    y, mu, sd = _gaussian_forecasts()
    res = pit_uniformity_test(y, mu=mu, sigma=sd)
    fig, (ax_a, ax_b) = plt.subplots(1, 2)
    fig.suptitle("caller's figure title")
    res.plot(ax=ax_b, title="PIT panel")
    assert "PIT panel" in {ax_b.get_title(loc) for loc in ("left", "center", "right")}
    assert fig._suptitle.get_text() == "caller's figure title"
    plt.close(fig)

    fig = res.plot(title="own figure")
    assert fig._suptitle.get_text() == "own figure"
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. No 'semibold' font weight in the nowcast plots
# ----------------------------------------------------------------------------

def test_no_semibold_font_weight():
    import puremacro.nowcast.dfm as dfm
    import puremacro.nowcast.evaluation as evaluation
    import puremacro.nowcast.news as news
    import puremacro.nowcast.realtime_nowcast as rt
    for mod in (dfm, evaluation, news, rt):
        assert "semibold" not in inspect.getsource(mod), mod.__name__


# ----------------------------------------------------------------------------
# 4. revision_stats: the noise null is b_T = 0, not b_0 = -1
# ----------------------------------------------------------------------------

def _revision_panel(dgp: str, seed: int, T: int = 40, sd_true: float = 1.5, sd_v: float = 0.6):
    """Two editions per quarter: a first release and a final value one quarter later.

    noise: y_0 = y* + v, final = y*   (classical measurement error)
    news : y_0 = y*,     final = y* + v (the revision is new information)
    """
    r = np.random.default_rng(seed)
    obs = pd.date_range("2000-01-01", periods=T, freq="QS")
    vin = pd.date_range("2000-04-01", periods=T + 2, freq="QS")
    y_star = r.normal(2.2, sd_true, T)
    v = r.normal(0.0, sd_v, T)
    first = y_star + v if dgp == "noise" else y_star
    final = y_star if dgp == "noise" else y_star + v
    rows = []
    for i, d in enumerate(obs):
        rows.append(dict(country="USA", variable="gdp_real", date=d, vintage=vin[i], value=first[i]))
        rows.append(dict(country="USA", variable="gdp_real", date=d, vintage=vin[i + 1], value=final[i]))
    return QNAVintagePanel(df=pd.DataFrame(rows))


def test_revision_stats_pure_noise_is_not_labelled_news_or_mixed():
    """Noise share 0.36 / 2.61 = 0.138, T = 40, 200 replications.

    The old rule ('noise' only if |b_0 + 1| < 0.3) labelled all 200 of these
    'news' or 'mixed' and none 'noise'. The two-leg test says 'news' or
    'mixed' only when it falsely rejects the true noise null, about 5% of the
    time.
    """
    labels = pd.Series([
        _revision_panel("noise", s).revision_stats("USA", "gdp_real")["hypothesis"]
        for s in range(200)
    ])
    counts = labels.value_counts()
    false_news = counts.get("news", 0) + counts.get("mixed", 0)
    assert false_news / 200 <= 0.10, counts.to_dict()
    assert counts.idxmax() == "noise", counts.to_dict()
    assert set(labels) <= {"news", "noise", "mixed", "indeterminate"}


def test_revision_stats_pure_news_is_not_labelled_noise_or_mixed():
    labels = pd.Series([
        _revision_panel("news", s).revision_stats("USA", "gdp_real")["hypothesis"]
        for s in range(200)
    ])
    counts = labels.value_counts()
    assert (counts.get("noise", 0) + counts.get("mixed", 0)) / 200 <= 0.10, counts.to_dict()
    assert counts.idxmax() == "news", counts.to_dict()


def test_revision_stats_large_sample_recovers_noise_slopes():
    st = _revision_panel("noise", 7, T=2000).revision_stats("USA", "gdp_real")
    share = 0.6**2 / (1.5**2 + 0.6**2)
    assert st["hypothesis"] == "noise"
    assert st["mankiw_shapiro_beta"] == pytest.approx(-share, abs=0.03)
    assert -1.0 < st["mankiw_shapiro_beta"] < 0.0
    assert st["mankiw_shapiro_beta_final"] == pytest.approx(0.0, abs=0.03)
    assert st["noise_share"] == pytest.approx(share, abs=0.03)
    assert st["rejects_news"] and not st["rejects_noise"]


def test_revision_stats_delegates_to_mankiw_shapiro():
    panel = _revision_panel("noise", 11)
    st = panel.revision_stats("USA", "gdp_real", hac_lags="auto", significance=0.1)
    ms = mankiw_shapiro(panel.first_release("USA", "gdp_real"),
                        panel.latest_release("USA", "gdp_real"),
                        hac_lags="auto", significance=0.1)
    assert st["mankiw_shapiro_beta"] == ms.beta_on_preliminary
    assert st["mankiw_shapiro_se"] == ms.se_beta_on_preliminary
    assert st["mankiw_shapiro_pvalue"] == ms.p_beta_on_preliminary
    assert st["mankiw_shapiro_beta_final"] == ms.beta_on_final
    assert st["mankiw_shapiro_pvalue_final"] == ms.p_beta_on_final
    assert st["hac_lags"] == ms.hac_lags and st["significance"] == 0.1
    mapping = {"news": "news", "noise": "noise", "neither": "mixed", "indeterminate": "indeterminate"}
    assert st["hypothesis"] == mapping[ms.verdict]


def test_revision_stats_edge_cases_and_docstring():
    short = QNAVintagePanel(df=_revision_panel("noise", 1, T=3).df)
    st = short.revision_stats("USA", "gdp_real")
    assert st["hypothesis"] == "insufficient_data" and np.isnan(st["mankiw_shapiro_beta"])

    df = _revision_panel("noise", 1, T=10).df.copy()
    df = df.drop_duplicates(subset=["date"], keep="first")  # one edition per quarter
    st = QNAVintagePanel(df=df).revision_stats("USA", "gdp_real")
    assert st["hypothesis"] == "no_revisions" and st["mean_revision"] == 0.0

    doc = inspect.getdoc(QNAVintagePanel.revision_stats)
    assert "-Var(v) / Var(y_0)" in doc and "not the noise null" in doc
    assert "b_T = 0" in doc
