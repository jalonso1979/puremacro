"""Contract, chronological-index, alias, estimator-honesty, and specification-provenance tests for puremacro.lp."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from puremacro._linalg import inv_xtx
from puremacro.examples.ramey_zubairy_2018_multipliers import RZ_CONTROLS, load_rz_data
from puremacro.inference.hac import newey_west_se
from puremacro.lp import (
    estimate_distributional_lp,
    estimate_quantile_lp,
    lp_hac,
    lp_iv,
    panel_lp,
    panel_lp_dk,
    panel_lp_iv,
)
from puremacro.lp._common import check_regular_index


def _dated_macro_frame(n: int = 100, freq: str = "Q", seed: int = 1234) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    w = rng.standard_normal(n)
    u = rng.standard_normal(n) * 0.4
    x = 0.8 * z + 0.3 * w + u
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.55 * y[t - 1] + 0.45 * x[t - 1] + 0.2 * w[t] + 0.3 * u[t] + 0.25 * rng.standard_normal()
    idx = (
        pd.period_range("1995Q1", periods=n, freq="Q")
        if freq == "Q"
        else pd.date_range("1995-01-01", periods=n, freq="QS")
    )
    return pd.DataFrame({"y": y, "x": x, "z": z, "w": w}, index=idx)


@pytest.mark.parametrize("freq", ["Q", "QS"])
def test_chronological_reversal_and_unsorted_dated_index_rejected_with_sort_recovery(freq: str) -> None:
    df = _dated_macro_frame(freq=freq)
    expected = lp_hac(df, y="y", x="x", controls=["w"], horizons=[0, 1, 2], n_lags=2)

    df_rev = df.iloc[::-1].copy()
    with pytest.raises(ValueError, match=r"sort_index\(\)"):
        lp_hac(df_rev, y="y", x="x", controls=["w"], horizons=[0, 1, 2], n_lags=2)
    with pytest.raises(ValueError, match=r"sort_index\(\)"):
        lp_iv(df_rev, y="y", x="x", z="z", controls=["w"], horizons=[0, 1, 2], n_lags=2)
    with pytest.raises(ValueError, match=r"sort_index\(\)"):
        estimate_quantile_lp(df_rev, shock="x", outcome="y", horizons=[0, 1], n_lags=1)

    # Interior swap (mostly sorted except two adjacent quarters) also fails loudly
    order = list(range(len(df)))
    order[25], order[26] = order[26], order[25]
    df_swapped = df.iloc[order].copy()
    with pytest.raises(ValueError, match=r"sort_index\(\)"):
        lp_hac(df_swapped, y="y", x="x", horizons=[0, 1], n_lags=1)

    # Explicit sort_index() recovers bit-identical estimates
    recovered = lp_hac(df_rev.sort_index(), y="y", x="x", controls=["w"], horizons=[0, 1, 2], n_lags=2)
    pd.testing.assert_frame_equal(pd.DataFrame(expected), pd.DataFrame(recovered))


def test_duplicate_and_length_two_dated_indexes_rejected() -> None:
    dup_dt = pd.DatetimeIndex(["2020-01-01", "2020-01-01"])
    rev_dt = pd.DatetimeIndex(["2020-04-01", "2020-01-01"])
    dup_pr = pd.PeriodIndex(["2020Q1", "2020Q1"], freq="Q")
    rev_pr = pd.PeriodIndex(["2020Q2", "2020Q1"], freq="Q")

    for idx in (dup_dt, rev_dt, dup_pr, rev_pr):
        with pytest.raises(ValueError):
            check_regular_index(idx, "lp_hac", cumulative=False)
        with pytest.raises(ValueError):
            check_regular_index(idx, "lp_iv", cumulative=True)
        tiny = pd.DataFrame({"y": [1.0, 2.0], "x": [0.5, -0.5]}, index=idx)
        with pytest.raises(ValueError):
            lp_hac(tiny, y="y", x="x", horizons=[0], n_lags=0)

    nat_idx = pd.DatetimeIndex(["2020-01-01", pd.NaT, "2020-07-01"])
    with pytest.raises(ValueError, match="NaT"):
        check_regular_index(nat_idx, "lp_hac", cumulative=False)


def test_gapped_dated_index_rejected_by_default_and_nan_row_pattern_matches() -> None:
    df = _dated_macro_frame(n=90, freq="Q")
    df_gap = df.drop(df.index[35])

    with pytest.raises(ValueError, match="allow_index_gaps=True"):
        lp_hac(df_gap, y="y", x="x", horizons=[0, 1, 2], n_lags=2)
    with pytest.raises(ValueError, match="allow_index_gaps=True"):
        lp_iv(df_gap, y="y", x="x", z="z", horizons=[0, 1, 2], n_lags=2)

    # Recommended exclusion pattern: set excluded periods to NaN while keeping rows
    df_nan = df.copy()
    df_nan.iloc[35] = np.nan
    res_nan = lp_hac(df_nan, y="y", x="x", horizons=[0, 1, 2], n_lags=2)
    assert np.all(np.isfinite(res_nan["beta"]))
    # Explicit opt-in works for non-cumulative LP when user acknowledges pre-built shifts
    res_opt = lp_hac(df_gap, y="y", x="x", horizons=[0, 1], n_lags=1, allow_index_gaps=True)
    assert res_opt.attrs["specification"]["allow_index_gaps"] is True


def test_series_mode_alignment_and_dated_index_contracts() -> None:
    df = _dated_macro_frame(n=70, freq="Q")
    # 1-D Series mode preserves dated index regularity checks and sample bounds (both positional and keyword x)
    res_ser_pos = lp_hac(df["y"], df["x"], controls=df[["w"]], horizons=[0, 1], n_lags=1)
    res_ser = lp_hac(df["y"], x=df["x"], controls=df[["w"]], horizons=[0, 1], n_lags=1)
    res_df = lp_hac(df, y="y", x="x", controls=["w"], horizons=[0, 1], n_lags=1)
    np.testing.assert_allclose(res_ser_pos["beta"], res_df["beta"], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(res_ser["beta"], res_df["beta"], rtol=1e-13, atol=1e-14)
    assert res_ser.attrs["specification"]["sample_start_by_horizon"] == res_df.attrs["specification"]["sample_start_by_horizon"]

    # Reversed Series index raises
    with pytest.raises(ValueError, match=r"sort_index\(\)"):
        lp_hac(df["y"].iloc[::-1], x=df["x"].iloc[::-1], horizons=[0, 1], n_lags=1)

    # Mismatched Series indexes raise even when lengths match
    shifted_x = pd.Series(df["x"].to_numpy(), index=df.index + 1)
    with pytest.raises(ValueError, match="mismatched lengths or indexes"):
        lp_hac(df["y"], x=shifted_x, horizons=[0, 1], n_lags=1)

    # Mismatched array lengths raise cleanly
    with pytest.raises(ValueError, match="same length"):
        lp_hac(df["y"].to_numpy(), x=df["x"].to_numpy()[:-3], horizons=[0, 1], n_lags=1)
    with pytest.raises(ValueError, match="controls row count"):
        lp_hac(df["y"].to_numpy(), x=df["x"].to_numpy(), controls=df[["w"]].to_numpy()[:-2], horizons=[0, 1], n_lags=1)


def test_nonfinite_inf_rejected_while_nan_is_supported() -> None:
    df = _dated_macro_frame(n=80, freq="Q")
    for col, bad_val in (("y", np.inf), ("x", -np.inf), ("z", np.inf), ("w", -np.inf)):
        bad = df.copy()
        bad.iloc[20, bad.columns.get_loc(col)] = bad_val
        if col != "z":
            with pytest.raises(ValueError, match="non-finite"):
                lp_hac(bad, y="y", x="x", controls=["w"], horizons=[0, 1], n_lags=1)
        with pytest.raises(ValueError, match="non-finite"):
            lp_iv(bad, y="y", x="x", z="z", controls=["w"], horizons=[0, 1], n_lags=1)

    # 1-D array with ±inf also rejected
    arr_bad = df["x"].to_numpy().copy()
    arr_bad[10] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        lp_hac(df["y"].to_numpy(), x=arr_bad, horizons=[0, 1], n_lags=1)

    # Duplicate DataFrame columns rejected
    dup_cols = pd.concat([df[["y", "x"]], df[["x"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate column"):
        lp_hac(dup_cols, y="y", x="x", horizons=[0, 1], n_lags=1)


def test_conflicting_canonical_and_alias_arguments_rejected_everywhere() -> None:
    df = _dated_macro_frame(n=80, freq="Q")

    conflict_kwargs = [
        {"n_lags": 4, "lags": 1},
        {"n_lags": 2, "lags": 4},  # canonical explicitly passed at default value!
        {"horizons": [0, 2, 4], "horizon": 4},
        {"horizons": range(0, 21), "horizon": 4},  # canonical explicitly passed at default value!
        {"alpha": 0.05, "ci": 0.90},
        {"alpha": 0.10, "ci": 0.95},  # canonical explicitly passed at default value!
    ]
    for kw in conflict_kwargs:
        with pytest.raises(ValueError, match="conflicting"):
            lp_hac(df, y="y", x="x", **kw)
        with pytest.raises(ValueError, match="conflicting"):
            lp_iv(df, y="y", x="x", z="z", **kw)

    # Matching canonical + alias pairs are accepted cleanly
    ok = lp_hac(df, y="y", x="x", horizons=range(0, 4), horizon=3, n_lags=3, lags=3, alpha=0.05, ci=0.95)
    assert list(ok["h"]) == [0, 1, 2, 3]
    assert ok.attrs["specification"]["n_lags"] == 3
    assert ok.attrs["specification"]["alpha"] == pytest.approx(0.05)

    # Panel estimators also reject conflicting aliases
    pan = pd.DataFrame(
        {
            "code": ["A"] * 30 + ["B"] * 30,
            "date": list(range(30)) * 2,
            "y": df["y"].to_numpy()[:60],
            "x": df["x"].to_numpy()[:60],
            "z": df["z"].to_numpy()[:60],
        }
    ).set_index(["code", "date"])
    for fn, extra in ((panel_lp, {}), (panel_lp_dk, {}), (panel_lp_iv, {"z": "z"})):
        with pytest.raises(ValueError, match="conflicting"):
            fn(pan, y="y", x="x", n_lags=2, lags=1, **extra)
        with pytest.raises(ValueError, match="conflicting"):
            fn(pan, y="y", x="x", alpha=0.10, ci=0.95, **extra)

    # Distributional and quantile LP conflict checks
    df_dist = df.copy()
    df_dist["b1"] = 0.4 + 0.02 * df_dist["x"]
    df_dist["b2"] = 1.0 - df_dist["b1"]
    with pytest.raises(ValueError, match="conflicting"):
        estimate_distributional_lp(df_dist, shock="x", bin_cols=["b1", "b2"], n_lags=2, lags=4)
    with pytest.raises(ValueError, match="conflicting"):
        estimate_distributional_lp(df_dist, shock="x", bin_cols=["b1", "b2"], ci=0.90, alpha=0.05)
    with pytest.raises(ValueError, match="conflicting"):
        estimate_quantile_lp(df, shock="x", x="w", outcome="y")
    with pytest.raises(ValueError, match="conflicting"):
        estimate_quantile_lp(df, shock="x", outcome="y", y="w")
    with pytest.raises(ValueError, match="conflicting"):
        estimate_quantile_lp(df, shock="x", outcome="y", n_lags=2, lags=1)


def test_estimate_quantile_lp_rejects_panel_and_unsupported_cov_type() -> None:
    df = _dated_macro_frame(n=60, freq="Q")
    pan_long = pd.DataFrame(
        {
            "unit": ["U1"] * 30 + ["U2"] * 30,
            "time": list(pd.period_range("2000Q1", periods=30, freq="Q")) * 2,
            "y": df["y"].to_numpy(),
            "x": df["x"].to_numpy(),
        }
    )
    with pytest.raises(ValueError, match="panel inputs"):
        estimate_quantile_lp(pan_long, shock="x", outcome="y", unit_col="unit", time_col="time")
    with pytest.raises(ValueError, match="panel inputs"):
        estimate_quantile_lp(pan_long.set_index(["unit", "time"]), shock="x", outcome="y")
    with pytest.raises(ValueError, match="cov_type"):
        estimate_quantile_lp(df, shock="x", outcome="y", cov_type="driscoll_kraay")

    # Supported time-series call with time_col works and normalizes cov_type='newey_west' -> 'hac'
    ts_long = df.reset_index().rename(columns={"index": "quarter"})
    res_q = estimate_quantile_lp(
        ts_long, shock="x", outcome="y", quantiles=(0.25, 0.75), horizons=[0, 1], time_col="quarter", cov_type="newey_west"
    )
    assert res_q.cov_type == "hac"


def test_specification_provenance_reconstructs_exact_regression_and_survives_slicing() -> None:
    df = _dated_macro_frame(n=96, freq="Q")
    res = lp_hac(df, y="y", x="x", controls=["w"], horizons=[0, 2, 5], n_lags=3, alpha=0.05)

    spec = res.attrs["specification"]
    assert res.metadata["specification"] == spec
    assert spec["estimator"] == "lp_hac"
    assert spec["y_name"] == "y" and spec["x_name"] == "x"
    assert spec["controls"] == ["w"]
    assert spec["n_lags"] == 3
    assert spec["horizons"] == [0, 2, 5]
    assert spec["estimand"] == "diff_from_t_minus_1"
    assert spec["kernel"] == "bartlett"
    assert spec["bandwidth_by_horizon"] == {0: 1, 2: 3, 5: 6}
    assert spec["small_sample_correction"] is False
    assert spec["alpha"] == pytest.approx(0.05)
    assert spec["ci_level"] == pytest.approx(0.95)
    assert spec["interval_method"] == "normal_wald"

    # Reconstruct the exact estimation purely from `spec` and `df`
    y_col = spec["y_name"]
    x_col = spec["x_name"]
    ctls = spec["controls"]
    p = spec["n_lags"]
    z_crit = float(norm.ppf(1.0 - spec["alpha"] / 2.0))
    for h in spec["horizons"]:
        work = pd.DataFrame(index=df.index)
        work["dy"] = df[y_col].shift(-h) - df[y_col].shift(1)
        work["x"] = df[x_col]
        reg_cols = ["x"]
        for lag in range(1, p + 1):
            work[f"x_L{lag}"] = df[x_col].shift(lag)
            work[f"y_L{lag}"] = df[y_col].shift(lag)
            reg_cols += [f"x_L{lag}", f"y_L{lag}"]
            for c in ctls:
                work[f"{c}_L{lag}"] = df[c].shift(lag)
                reg_cols.append(f"{c}_L{lag}")
        for c in ctls:
            work[c] = df[c]
            reg_cols.append(c)
        sub = work[["dy", *reg_cols]].assign(y_t=df[y_col]).dropna()
        assert len(sub) == spec["n_obs_by_horizon"][h]
        assert str(sub.index[0]) == spec["sample_start_by_horizon"][h]
        assert str(sub.index[-1]) == spec["sample_end_by_horizon"][h]

        X = np.column_stack([np.ones(len(sub)), sub[reg_cols].to_numpy()])
        yv = sub["dy"].to_numpy()
        b_vec = inv_xtx(X) @ (X.T @ yv)
        u_vec = yv - X @ b_vec
        se_vec = newey_west_se(X, u_vec, spec["bandwidth_by_horizon"][h])
        assert float(res.loc[h, "beta"]) == pytest.approx(float(b_vec[1]), rel=1e-13, abs=1e-14)
        assert float(res.loc[h, "se"]) == pytest.approx(float(se_vec[1]), rel=1e-13, abs=1e-14)
        assert float(res.loc[h, "lo"]) == pytest.approx(float(b_vec[1] - z_crit * se_vec[1]), rel=1e-13, abs=1e-14)

    # Provenance survives DataFrame slicing
    sliced = res.iloc[1:]
    assert sliced.attrs["specification"]["estimator"] == "lp_hac"
    assert sliced.metadata["n_lags"] == 3


def test_weak_iv_summary_warnings_and_line_geometry() -> None:
    rz = load_rz_data()
    rz_iv = lp_iv(
        rz, y="y", x="g", z="newsy", horizons=[0, 7, 15], n_lags=0, controls=RZ_CONTROLS, cumulative=True
    )
    spec_iv = rz_iv.attrs["specification"]
    assert spec_iv["estimator"] == "lp_iv"
    assert spec_iv["estimand"] == "cumulative_multiplier"
    assert len(spec_iv["weak_iv_horizons_10"]) >= 1

    s_plain = rz_iv.summary()
    lines_plain = s_plain.splitlines()
    assert lines_plain[0].startswith("Local Projection Result")
    assert "lags: 0" in lines_plain[1] and "SE: 2SLS HAC Bartlett(h+1)" in lines_plain[1]
    assert lines_plain[2].startswith("estimand: cumulative multiplier")
    assert "Weak-IV warning:" in lines_plain[-1]
    assert "anderson_rubin=True" in lines_plain[-1]

    rz_ar = lp_iv(
        rz,
        y="y",
        x="g",
        z="newsy",
        horizons=[7, 15],
        n_lags=0,
        controls=RZ_CONTROLS,
        cumulative=True,
        anderson_rubin=True,
    )
    assert "ar_lo/ar_hi/ar_set_type" in rz_ar.summary().splitlines()[-1]

    # Non-cumulative summary preserves lines[2] as the table header
    df = _dated_macro_frame(n=90, freq="Q")
    s_ols = lp_hac(df, y="y", x="x", controls=["w"], horizons=[0, 1, 2], n_lags=2).summary()
    ols_lines = s_ols.splitlines()
    assert "SE: HAC Bartlett(h+1)" in ols_lines[1]
    assert "beta" in ols_lines[2] and "sig" in ols_lines[2]
    assert "Weak-IV warning" not in s_ols
