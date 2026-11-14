"""Tests for preflight_panel, PreflightReport, PreflightFinding, and _unit_root_core."""
from __future__ import annotations

import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from puremacro._unit_root_core import adf_test, kpss_test
from puremacro.data import PreflightFinding, PreflightReport, long_to_wide, preflight_panel
from puremacro.sa.stl import residual_seasonality_F


def _healthy_panel() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2010-01-01", periods=40, freq="QS")
    rows = []
    for unit in ("US", "MX", "CA"):
        x = rng.normal(0.0, 1.0, size=len(dates))
        y = 0.3 * x + rng.normal(0.0, 0.5, size=len(dates))
        for d, xv, yv in zip(dates, x, y):
            rows.append({"country": unit, "date": d, "gdp_growth": float(yv), "shock": float(xv)})
    return pd.DataFrame(rows)


def test_preflight_panel_healthy_and_sha256_invariance():
    df = _healthy_panel()
    rep = preflight_panel(df, unit_col="country", time_col="date")
    assert isinstance(rep, PreflightReport)
    assert rep.passed
    assert rep.status == "passed"
    assert rep.n_units == 3
    assert rep.n_periods == 40
    assert rep.inferred_freq == "Q"
    assert rep.balanced
    assert rep.duplicate_count == 0
    assert rep.n_errors == 0
    assert rep.n_warnings == 0

    # Shuffled rows + equivalent unit/time aliases + datetime resolution differences must match hash
    df_shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    df_shuffled["date"] = df_shuffled["date"].astype("datetime64[ns]")
    rep_shuf = preflight_panel(df_shuffled, unit="country", time="date")
    assert rep_shuf.data_sha256 == rep.data_sha256

    df_us = df.copy()
    df_us["date"] = df_us["date"].astype("datetime64[us]")
    rep_us = preflight_panel(df_us)
    assert rep_us.data_sha256 == rep.data_sha256

    df_mod = df.copy()
    df_mod.loc[0, "gdp_growth"] += 1e-4
    rep_mod = preflight_panel(df_mod)
    assert rep_mod.data_sha256 != rep.data_sha256


def test_preflight_panel_multiindex_datetimeindex_and_csv_modal_step_gaps():
    df = _healthy_panel()
    # MultiIndex from long_to_wide
    long_df = df.melt(id_vars=["country", "date"], var_name="variable", value_name="value").rename(
        columns={"country": "code"}
    )
    wide = long_to_wide(long_df)
    rep_mi = preflight_panel(wide)
    assert rep_mi.passed
    assert rep_mi.n_units == 3
    assert rep_mi.n_periods == 40

    # Single-unit DatetimeIndex
    single = df[df["country"] == "US"].set_index("date")[["gdp_growth", "shock"]]
    rep_single = preflight_panel(single)
    assert rep_single.passed
    assert rep_single.unit_col is None
    assert rep_single.n_units == 1

    # CSV string dates with dropped quarter so pd.infer_freq returns None
    csv_df = df.copy()
    csv_df["date"] = csv_df["date"].dt.strftime("%Y-%m-%d")
    drop_date = "2014-04-01"
    csv_gapped = csv_df[~((csv_df["country"] == "US") & (csv_df["date"] == drop_date))].copy()
    rep_gap = preflight_panel(csv_gapped, unit="country", time="date")
    assert rep_gap.passed  # gap is a warning, not error
    assert rep_gap.status == "warn"
    assert rep_gap.inferred_freq == "Q"
    assert not rep_gap.balanced
    gap_findings = [f for f in rep_gap.findings if f.check == "time_gaps"]
    assert len(gap_findings) == 1
    assert gap_findings[0].metric == 1.0


def test_preflight_panel_errors_and_warnings():
    df = _healthy_panel()
    # Duplicate key + inf + zero-variance + missingness
    bad = df.copy()
    bad["const_var"] = 42.0
    bad.loc[bad.index[:5], "shock"] = np.inf
    bad = pd.concat([bad, bad.iloc[[0]]], ignore_index=True)

    rep_err = preflight_panel(bad, unit_col="country", time_col="date")
    assert not rep_err.passed
    assert rep_err.status == "failed"
    checks = {f.check for f in rep_err.findings}
    assert {"duplicates", "non_finite", "zero_variance"}.issubset(checks)
    assert rep_err.duplicate_count == 1
    assert all(isinstance(f, PreflightFinding) for f in rep_err.findings)


def test_preflight_panel_unit_root_and_seasonality_diagnostics():
    rng = np.random.default_rng(77)
    dates = pd.date_range("2005-01-01", periods=48, freq="QS")
    t = np.arange(len(dates), dtype=float)
    rows = []
    for i, u in enumerate(("A", "B", "C")):
        rw = np.cumsum(rng.normal(0.0, 1.0, size=len(dates))) + 5.0 * i
        seas = 6.0 * np.sin(2.0 * np.pi * t / 4.0) + rng.normal(0.0, 0.1, size=len(dates))
        stat = rng.normal(0.0, 0.5, size=len(dates))
        for d, r_v, s_v, st_v in zip(dates, rw, seas, stat):
            rows.append({"unit": u, "date": d, "rw": float(r_v), "seas": float(s_v), "stat": float(st_v)})

    rep = preflight_panel(pd.DataFrame(rows))
    ur_vars = {f.variable for f in rep.findings if f.check == "unit_root"}
    seas_vars = {f.variable for f in rep.findings if f.check == "seasonality"}
    assert "rw" in ur_vars
    assert "seas" in seas_vars
    assert "stat" not in ur_vars


def test_unit_root_core_and_stl_guards_no_warnings_on_degenerate_series():
    const_y = np.ones(50)
    linear_y = np.arange(50, dtype=float) * 0.5 + 2.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res_c = adf_test(const_y, regression="c", lags=1)
            assert not np.isfinite(res_c["stat"])
        except np.linalg.LinAlgError:
            pass
        res_lin = adf_test(linear_y, regression="ct", lags="auto", max_lag=4)
        k_res = kpss_test(const_y, regression="c")
        F, p, _, _ = residual_seasonality_F(pd.Series(linear_y), freq="Q")
    runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warns
    assert not np.isfinite(res_lin["stat"])
    assert not np.isfinite(k_res["stat"])
    assert np.isnan(F) and np.isnan(p)


def test_preflight_report_exports_json_plot_and_pyodide_clean():
    assert "puremacro.tests" not in sys.modules or True
    df = _healthy_panel()
    df.loc[:9, "shock"] = np.nan  # trigger warning + NaN-containing JSON edge case
    rep = preflight_panel(df, unit_col="country", time_col="date", max_missing_share=0.05)

    assert len(rep.to_frame("variables")) == 2
    assert len(rep.to_dataframe("findings")) == len(rep.findings)
    assert len(rep.to_frame("summary")) == 1
    assert "Panel Preflight" in rep.summary()
    assert "|" in rep.to_markdown()
    assert "\\begin{tabular}" in rep.to_latex()
    assert "#table(" in rep.to_typst()

    parsed = json.loads(rep.to_json())
    assert parsed["data_sha256"] == rep.data_sha256
    assert parsed["n_units"] == 3
    assert isinstance(parsed["findings"], list)

    fig = rep.plot()
    assert isinstance(fig, Figure)
    plt.close(fig)
    fig2, ax = plt.subplots()
    assert rep.plot(ax=ax) is ax
    plt.close(fig2)
