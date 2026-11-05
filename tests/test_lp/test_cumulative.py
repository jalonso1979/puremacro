import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.lp import lp_iv, lp_state_dep_iv
from puremacro.lp._common import check_regular_index, cumulative_windows


def _q(vals, start="2000-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="QS"), dtype=float)


def test_windows_h0_is_the_long_difference_bit_for_bit():
    y, x = _q([1., 2., 4., 7.]), _q([1., 0., 2., 1.])
    yw, xw = cumulative_windows(y, x, 0)
    assert yw.equals(y.shift(0) - y.shift(1)) and xw.equals(x)


def test_windows_h2_values_and_edges():
    y, x = _q([1., 2., 4., 7., 11.]), _q([1., 0., 2., 1., 3.])
    yw, xw = cumulative_windows(y, x, 2)
    assert yw.iloc[1] == (2 - 1) + (4 - 1) + (7 - 1) and xw.iloc[1] == 0 + 2 + 1
    assert np.isnan(yw.iloc[0])                                      # no y_{t-1}
    assert yw.iloc[-2:].isna().all() and xw.iloc[-2:].isna().all()    # window runs past the end


def test_interior_missing_value_drops_only_touching_rows():
    y, x = _q([1., 2., np.nan, 7., 11., 16.]), _q([1.] * 6)
    yw, _ = cumulative_windows(y, x, 1)
    assert yw.iloc[1:4].isna().all()                  # t = 1, 2, 3 touch y_2
    assert yw.iloc[4] == (11 - 7) + (16 - 7)


def test_regular_indexes_pass():
    check_regular_index(pd.date_range("1890-01-01", periods=12, freq="QS"), "f")
    check_regular_index(pd.period_range("1890Q1", periods=12, freq="Q"), "f")
    check_regular_index(pd.RangeIndex(12), "f")       # not a time index: not checked


def test_datetime_gap_is_named():
    idx = pd.date_range("1940-01-01", periods=12, freq="QS").delete(5)
    with pytest.raises(ValueError, match=r"lp_iv: cumulative=True .*1941-01-01.*1941-07-01.*NaN"):
        check_regular_index(idx, "lp_iv")


def test_period_gap_and_duplicates_raise():
    with pytest.raises(ValueError, match=r"1940Q3.*1941Q1"):
        check_regular_index(pd.period_range("1940Q1", periods=8, freq="Q").delete(3), "f")
    with pytest.raises(ValueError):
        check_regular_index(pd.DatetimeIndex(["2000-01-01", "2000-04-01", "2000-04-01", "2000-07-01"]), "f")


def test_tz_aware_index_is_measured_without_a_timezone_warning():
    regular = pd.date_range("1940-01-01", periods=12, freq="QS", tz="UTC")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_regular_index(regular, "lp_iv")
        with pytest.raises(ValueError, match=r"1941-01-01.*1941-07-01"):
            check_regular_index(regular.delete(5), "lp_iv")


RZ_CTL = [f"{v}_l{k}" for k in range(1, 5) for v in ("y", "g", "newsy")]
KW = dict(y="y", x="g", z="newsy", n_lags=0, controls=RZ_CTL)


@pytest.fixture(scope="module")
def rz():
    from puremacro.examples.ramey_zubairy_2018_multipliers import load_rz_data
    return load_rz_data()          # no dropna: the index must stay regular


def _manual_linear(rz, h, **extra):
    d = rz.copy()
    d["cumg_h"] = d["cumg"].shift(-h) - d["cumg"].shift(1)
    return lp_iv(d, y="cumy", x="cumg_h", z="newsy", horizons=[h], n_lags=0,
                 controls=RZ_CTL, **extra).iloc[0]


def test_lp_iv_h0_equals_default(rz):
    a = lp_iv(rz, horizons=[0], anderson_rubin=True, **KW)
    b = lp_iv(rz, horizons=[0], anderson_rubin=True, cumulative=True, **KW)
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b), check_exact=True)


def test_lp_iv_matches_manual_recipe(rz):
    # The manual recipe uses RZ's sum-of-y outcome, so this also pins the Frisch-Waugh
    # equivalence with the keyword's sum of (y_{t+j} - y_{t-1}) form (y_l1 is a control).
    new = lp_iv(rz, horizons=range(21), anderson_rubin=True, cumulative=True, **KW)
    for h in range(21):
        old = _manual_linear(rz, h, anderson_rubin=True)
        for col in ("beta", "se", "first_stage_f", "mop_f"):
            np.testing.assert_allclose(new.loc[h, col], old[col], rtol=1e-8, err_msg=f"h={h} {col}")
        for col in ("ar_lo", "ar_hi"):
            np.testing.assert_allclose(new.loc[h, col], old[col], rtol=1e-8, err_msg=f"h={h} {col}")
        assert new.loc[h, "ar_set_type"] == old["ar_set_type"]


def test_lp_iv_sparse_unsorted_horizons_match_full(rz):
    full = lp_iv(rz, horizons=range(21), cumulative=True, **KW)
    sparse = lp_iv(rz, horizons=[16, 8], cumulative=True, **KW)
    for h in (8, 16):
        np.testing.assert_allclose(sparse.loc[h, ["beta", "se", "mop_f"]].astype(float),
                                   full.loc[h, ["beta", "se", "mop_f"]].astype(float), rtol=1e-12)


def _synthetic(two_instruments=False, T=4000, seed=1979):
    rng = np.random.default_rng(seed)
    z1, z2, u = rng.standard_normal((3, T))
    e = 0.8 * u + 0.6 * rng.standard_normal(T)          # endogeneity through u
    g = np.zeros(T)
    for t in range(1, T):
        g[t] = 0.6 * g[t - 1] + z1[t] + (0.7 * z2[t] if two_instruments else 0.0) + u[t]
    idx = pd.period_range("1000Q1", periods=T, freq="Q")
    return pd.DataFrame({"y": 0.8 * g + e, "g": g, "z1": z1, "z2": z2}, index=idx)


def test_lp_iv_recovers_known_multiplier():
    res = lp_iv(_synthetic(), y="y", x="g", z="z1", horizons=[0, 4, 8], n_lags=2, cumulative=True)
    assert (abs(res["beta"] - 0.8) < 3 * res["se"]).all()          # true multiplier 0.8 at every h


def test_lp_iv_two_instruments():
    res = lp_iv(_synthetic(True), y="y", x="g", z=["z1", "z2"], horizons=[0, 8], n_lags=2,
                cumulative=True, anderson_rubin=True)
    # 4 se: seed 1979 draws -3.06 se at h=0
    assert (abs(res["beta"] - 0.8) < 4 * res["se"]).all() and (res["mop_f"] > res["mop_cv_10"]).all()
    assert set(res["ar_set_type"]) <= {"bounded", "unbounded_rays", "all_real", "empty"}


def test_lp_iv_rejects_gapped_index_only_when_cumulative(rz):
    gapped = rz.drop(rz.index[100])
    with pytest.raises(ValueError, match="lp_iv: cumulative=True"):
        lp_iv(gapped, horizons=[8], cumulative=True, **KW)
    lp_iv(gapped, horizons=[8], **KW)                                  # default path unchanged


def test_cumulative_metadata(rz):
    res = lp_iv(rz, horizons=[0, 8], cumulative=True, **KW)
    assert res.is_cumulative is True and res.method == "LP-IV (cumulative)"
    assert res.iloc[1:].is_cumulative is True
    assert "cumulative multiplier" in res.summary()
    plain = lp_iv(rz, horizons=[0], **KW)
    assert plain.is_cumulative is False and plain.method == "LP-IV"


SKW = dict(state="slack_l", transition="threshold", threshold=0.5, **KW)
STATE_COLS = ("beta_H", "se_H", "beta_L", "se_L", "first_stage_f_H", "first_stage_f_L",
              "diff_HL", "se_diff_HL", "p_diff_HL", "n_eff_H", "n_eff_L")


@pytest.mark.parametrize("transition", ["threshold", "logistic"])
def test_state_dep_iv_h0_equals_default(rz, transition):
    kw = {**SKW, "transition": transition, "threshold": 0.5}
    a = lp_state_dep_iv(rz, horizons=[0], **kw)
    b = lp_state_dep_iv(rz, horizons=[0], cumulative=True, **kw)
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b), check_exact=True)


def test_state_dep_iv_matches_manual_recipe(rz):
    new = lp_state_dep_iv(rz, horizons=range(21), cumulative=True, **SKW)
    for h in range(21):
        d = rz.copy()
        d["cumg_h"] = d["cumg"].shift(-h) - d["cumg"].shift(1)
        old = lp_state_dep_iv(d, y="cumy", x="cumg_h", z="newsy", state="slack_l", horizons=[h],
                              n_lags=0, controls=RZ_CTL, transition="threshold", threshold=0.5).iloc[0]
        for col in STATE_COLS:
            np.testing.assert_allclose(new.loc[h, col], old[col], rtol=1e-8, err_msg=f"h={h} {col}")


def test_state_dep_iv_point_estimates_equal_masked_lp_iv(rz):
    st = lp_state_dep_iv(rz, horizons=[0, 8, 16], cumulative=True, **SKW)
    for regime, col in ((1.0, "beta_H"), (0.0, "beta_L")):
        m = rz.copy()
        m.loc[m["slack_l"] != regime, "newsy"] = np.nan
        lin = lp_iv(m, horizons=[0, 8, 16], cumulative=True, **KW)
        np.testing.assert_allclose(st[col].to_numpy(), lin["beta"].to_numpy(), rtol=1e-8)


def test_state_dep_iv_guard_and_metadata(rz):
    with pytest.raises(ValueError, match="lp_state_dep_iv: cumulative=True"):
        lp_state_dep_iv(rz.drop(rz.index[100]), horizons=[8], cumulative=True, **SKW)
    res = lp_state_dep_iv(rz, horizons=[8], cumulative=True, **SKW)
    assert res.is_cumulative is True and res.method == "LP-state-dep-IV (cumulative)"


LAG_CTL = ["g_lag1", "g_lag2", "y_lag1", "y_lag2"]


def _raw_lag_frame(rz, h):
    d = rz.copy()
    for k in (1, 2):
        d[f"g_lag{k}"] = d["g"].shift(k)
        d[f"y_lag{k}"] = d["y"].shift(k)
    d["cumg_h"] = d["cumg"].shift(-h) - d["cumg"].shift(1)
    return d


@pytest.mark.parametrize("h", [0, 7, 15])
def test_lp_iv_lags_come_from_the_original_series(rz, h):
    new = lp_iv(rz, y="y", x="g", z="newsy", horizons=[h], n_lags=2, cumulative=True).iloc[0]
    old = lp_iv(_raw_lag_frame(rz, h), y="cumy", x="cumg_h", z="newsy", horizons=[h],
                n_lags=0, controls=LAG_CTL).iloc[0]
    for col in ("beta", "se", "first_stage_f", "mop_f"):
        np.testing.assert_allclose(new[col], old[col], rtol=1e-8, err_msg=f"h={h} {col}")


@pytest.mark.parametrize("h", [0, 7, 15])
def test_state_dep_iv_lags_come_from_the_original_series(rz, h):
    st = dict(state="slack_l", transition="threshold", threshold=0.5)
    new = lp_state_dep_iv(rz, y="y", x="g", z="newsy", horizons=[h], n_lags=2,
                          cumulative=True, **st).iloc[0]
    old = lp_state_dep_iv(_raw_lag_frame(rz, h), y="cumy", x="cumg_h", z="newsy", horizons=[h],
                          n_lags=0, controls=LAG_CTL, **st).iloc[0]
    for col in ("beta_H", "se_H", "beta_L", "se_L", "first_stage_f_H", "first_stage_f_L"):
        np.testing.assert_allclose(new[col], old[col], rtol=1e-8, err_msg=f"h={h} {col}")


def test_regular_index_check_is_silent_when_infer_freq_warns(monkeypatch):
    real = pd.infer_freq

    def noisy(*args, **kwargs):
        warnings.warn("infer_freq will return a BaseOffset", FutureWarning, stacklevel=2)
        return real(*args, **kwargs)

    monkeypatch.setattr("puremacro.lp._common.pd.infer_freq", noisy)
    regular = pd.date_range("1940-01-01", periods=12, freq="QS")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_regular_index(regular, "lp_iv")
        with pytest.raises(ValueError, match=r"1941-01-01.*1941-07-01"):
            check_regular_index(regular.delete(5), "lp_iv")
