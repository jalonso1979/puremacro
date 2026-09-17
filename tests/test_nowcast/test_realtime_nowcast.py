"""Tests for the high-level Latin America Real-Time Nowcast Orchestrator."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.realtime import (
    VintagePanel,
    pack_realtime_cartridge,
)
from puremacro.nowcast.realtime_nowcast import (
    COUNTRY_SPECS,
    RealtimeNowcastResult,
    realtime_nowcast,
)


def _build_mock_vintage_panel(country: str, variables: list[str], n_periods: int = 36) -> VintagePanel:
    """Helper to build a realistic multi-vintage real-time panel with ragged edges."""
    rng = np.random.default_rng(hash(country) % 1000000)
    dates = pd.date_range("2021-01-01", periods=n_periods, freq="MS")
    v1 = pd.Timestamp("2023-11-01")
    v2 = pd.Timestamp("2023-12-01")

    # Common latent factor
    f = np.cumsum(rng.normal(scale=0.3, size=n_periods))

    rows = []
    for var in variables:
        load = rng.uniform(0.6, 1.4)
        noise = rng.normal(scale=0.2, size=n_periods)
        y = 2.0 + load * f + noise

        for t_idx, d in enumerate(dates):
            # Vintage 1: ragged edge at the last 2 periods
            val_v1 = float(y[t_idx])
            if t_idx == n_periods - 1 and var != variables[0]:
                val_v1 = np.nan
            elif t_idx == n_periods - 2 and var in variables[2:]:
                val_v1 = np.nan

            if not np.isnan(val_v1):
                rows.append({
                    "country": country.upper(),
                    "variable": var,
                    "date": d,
                    "vintage": v1,
                    "value": val_v1,
                    "provider": "mock",
                    "series_id": f"MOCK_{var}",
                    "units": "rate",
                })

            # Vintage 2: releases ragged values and revises an older point
            val_v2 = float(y[t_idx])
            if t_idx == n_periods - 1 and var in variables[3:]:
                val_v2 = np.nan
            if t_idx == 10 and var == variables[1]:
                val_v2 += 0.35  # Historical data revision

            if not np.isnan(val_v2):
                rows.append({
                    "country": country.upper(),
                    "variable": var,
                    "date": d,
                    "vintage": v2,
                    "value": val_v2,
                    "provider": "mock",
                    "series_id": f"MOCK_{var}",
                    "units": "rate",
                })

    df = pd.DataFrame(rows)
    return VintagePanel(df)


def test_realtime_nowcast_mexico_dfm():
    """Verify nowcast orchestrator for Mexico (Banxico/INEGI indicator panel)."""
    vars_mex = ["gdp", "igae", "ind_prod", "cpi", "policy_rate"]
    vp = _build_mock_vintage_panel("MEX", vars_mex, n_periods=30)

    res = realtime_nowcast(country="MEX", panel=vp, method="dfm", n_factors=1)

    assert isinstance(res, RealtimeNowcastResult)
    assert res.country == "MEX"
    assert res.target_variable == "gdp"
    assert res.method == "dfm"
    assert res.palette == "banxico"
    assert not np.isnan(res.nowcast)
    assert res.forecast_sd > 0.0
    assert not res.factors.empty
    assert not res.loadings.empty

    # Verify Bańbura & Modugno news decomposition is automatically integrated
    assert res.news_decomposition is not None
    assert res.news_decomposition.decomposition_error < 1e-10
    assert abs(res.news_decomposition.revision - res.news_decomposition.total_impact) < 1e-10


def test_realtime_nowcast_brazil_bcb():
    """Verify nowcast orchestrator for Brazil (BCB indicator panel)."""
    vars_bra = ["gdp", "ibc_br", "ind_prod", "ipca", "selic"]
    vp = _build_mock_vintage_panel("BRA", vars_bra, n_periods=28)

    res = realtime_nowcast(country="BRA", panel=vp, method="dfm", n_factors=1)

    assert res.country == "BRA"
    assert res.target_variable == "gdp"
    assert res.palette == "bcb"
    assert res.news_decomposition is not None
    assert res.news_decomposition.decomposition_error < 1e-10


def test_realtime_nowcast_chile_bcch():
    """Verify nowcast orchestrator for Chile (BCCh indicator panel)."""
    vars_chl = ["gdp", "imacec", "ind_prod", "ipc", "tpm"]
    vp = _build_mock_vintage_panel("CHL", vars_chl, n_periods=28)

    res = realtime_nowcast(country="CHL", panel=vp, method="dfm", n_factors=1)

    assert res.country == "CHL"
    assert res.target_variable == "gdp"
    assert res.news_decomposition is not None
    assert res.news_decomposition.decomposition_error < 1e-10


def test_realtime_nowcast_usa_alfred():
    """Verify nowcast orchestrator for USA (ALFRED Fed indicator panel)."""
    vars_usa = ["GDPC1", "ind_prod", "payems", "cpi", "fedfunds"]
    vp = _build_mock_vintage_panel("USA", vars_usa, n_periods=32)

    res = realtime_nowcast(country="USA", panel=vp, method="dfm", n_factors=1)

    assert res.country == "USA"
    assert res.target_variable == "GDPC1"
    assert res.news_decomposition is not None
    assert res.news_decomposition.decomposition_error < 1e-10


def test_realtime_nowcast_mfvar_method():
    """Verify nowcast orchestrator with Mixed-Frequency VAR engine."""
    rng = np.random.default_rng(99)
    T = 36
    dates = pd.date_range("2021-01-01", periods=T, freq="MS")

    # Quarterly GDP: observed only at quarter-end (month % 3 == 0)
    gdp = np.full(T, np.nan)
    gdp[2::3] = 2.0 + np.cumsum(rng.normal(scale=0.2, size=T // 3))

    ip = 1.8 + np.cumsum(rng.normal(scale=0.15, size=T))
    cpi = 3.5 + np.cumsum(rng.normal(scale=0.1, size=T))

    df_wide = pd.DataFrame({"gdp": gdp, "ip": ip, "cpi": cpi}, index=dates)

    res = realtime_nowcast(
        country="MEX",
        target_variable="gdp",
        panel=df_wide,
        method="mfvar",
        p=3,
    )

    assert res.method == "mfvar"
    assert not np.isnan(res.nowcast)
    assert res.forecast_sd > 0.0
    assert not res.factors.empty


def test_realtime_nowcast_cartridge_offline_mode(tmp_path):
    """Verify offline workflow: packing vintage data into a .pmz cartridge
    and running realtime_nowcast with zero network connectivity."""
    vars_mex = ["gdp", "igae", "ind_prod", "cpi"]
    vp = _build_mock_vintage_panel("MEX", vars_mex, n_periods=24)

    cartridge_file = tmp_path / "latam_nowcast_cartridge.pmz"
    pack_realtime_cartridge(vp, cartridge_file, notes="Offline LatAm test cartridge")
    assert cartridge_file.exists()

    # Nowcast directly from cartridge path
    res = realtime_nowcast(country="MEX", cartridge_path=cartridge_file, method="dfm")

    assert isinstance(res, RealtimeNowcastResult)
    assert res.country == "MEX"
    assert res.target_variable == "gdp"
    assert res.news_decomposition is not None
    assert res.news_decomposition.decomposition_error < 1e-10


def test_realtime_nowcast_as_of_cutoff():
    """Verify historical replay using as_of cutoff date."""
    vars_mex = ["gdp", "igae", "ind_prod"]
    vp = _build_mock_vintage_panel("MEX", vars_mex, n_periods=24)

    # Use first vintage date explicitly
    first_v = vp.df["vintage"].min()
    res_cutoff = realtime_nowcast(country="MEX", panel=vp, as_of=first_v, method="dfm")

    assert pd.Timestamp(res_cutoff.latest_vintage) == pd.Timestamp(first_v)
    assert not np.isnan(res_cutoff.nowcast)


def test_realtime_nowcast_presentation_and_plots():
    """Verify presentation methods and plotting on RealtimeNowcastResult."""
    vars_mex = ["gdp", "igae", "ind_prod", "cpi"]
    vp = _build_mock_vintage_panel("MEX", vars_mex, n_periods=24)

    res = realtime_nowcast(country="MEX", panel=vp, method="dfm")

    # Summary
    s = res.summary()
    assert "Real-Time Macroeconomic Nowcast: Mexico (MEX)" in s
    assert "Point Nowcast" in s
    assert "Confidence Interval" in s

    # Table conversions
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert "nowcast" in df.columns
    assert "ci_90_lower" in df.columns

    md = res.to_markdown()
    assert "MEX" in md
    assert "nowcast" in md

    latex = res.to_latex()
    assert "\\begin{tabular}" in latex

    typst = res.to_typst()
    assert "#table(" in typst

    # Main plot
    fig = res.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Plot news
    fig_news = res.plot_news()
    assert isinstance(fig_news, plt.Figure)
    plt.close(fig_news)

    # Plot fan chart
    fig_fan = res.plot_fan_chart()
    assert isinstance(fig_fan, plt.Figure)
    plt.close(fig_fan)


def test_realtime_nowcast_invalid_inputs():
    """Verify error handling on invalid target variable and unknown method."""
    df_wide = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})

    with pytest.raises(KeyError, match="Target variable 'non_existent' not found"):
        realtime_nowcast(country="MEX", target_variable="non_existent", panel=df_wide)

    with pytest.raises(ValueError, match="Unknown method 'invalid_method'"):
        realtime_nowcast(country="MEX", target_variable="x1", panel=df_wide, method="invalid_method")
