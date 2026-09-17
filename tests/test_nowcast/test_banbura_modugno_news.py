"""Tests for Bańbura & Modugno (2014) analytical news decomposition."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.nowcast.dfm import DynamicFactorModel, kalman_dfm
from puremacro.nowcast.news import NewsDecompositionResult, banbura_modugno_news


@pytest.fixture
def nowcast_panel_dgp():
    """Generates synthetic macroeconomic panel with common factor and known loadings."""
    rng = np.random.default_rng(101)
    T = 48
    n = 6
    dates = pd.date_range("2020-01-01", periods=T, freq="MS")
    cols = ["gdp", "ind_prod", "retail", "cpi", "unemployment", "rate"]

    # Factor AR(1)
    F = np.zeros(T)
    for t in range(1, T):
        F[t] = 0.75 * F[t - 1] + rng.normal(scale=0.5)

    # Loadings: GDP, IP, Retail are pro-cyclical; Unemployment is counter-cyclical
    loadings = np.array([1.2, 0.9, 0.7, 0.2, -0.8, 0.3])
    idio_sd = np.array([0.2, 0.3, 0.25, 0.2, 0.35, 0.15])

    X = np.zeros((T, n))
    for j in range(n):
        X[:, j] = 2.0 + loadings[j] * F + rng.normal(scale=idio_sd[j], size=T)

    df = pd.DataFrame(X, index=dates, columns=cols)
    return df, loadings


def test_banbura_modugno_pure_releases_identity(nowcast_panel_dgp):
    """Verify |revision - total_impact| < 10^-10 when new releases occur at the ragged edge."""
    df_true, _ = nowcast_panel_dgp
    df_old = df_true.copy()
    # Ragged edge at latest month: series 1 to 5 missing
    df_old.iloc[-1, 1:] = np.nan
    # Ragged edge at second-to-last month: series 3 to 5 missing
    df_old.iloc[-2, 3:] = np.nan

    model = DynamicFactorModel(n_factors=1, p=1, standardize=True).fit(df_old)

    # New vintage: releases for ind_prod and retail at T-1, and cpi at T-2
    df_new = df_old.copy()
    df_new.iloc[-2, 3] = df_true.iloc[-2, 3]
    df_new.iloc[-1, 1] = df_true.iloc[-1, 1]
    df_new.iloc[-1, 2] = df_true.iloc[-1, 2]

    res = banbura_modugno_news(model, df_old, df_new, target_series="gdp")

    assert isinstance(res, NewsDecompositionResult)
    assert res.target_variable == "gdp"
    # Mandatory precision requirement
    assert res.decomposition_error < 1e-10
    assert abs(res.revision - res.total_impact) < 1e-10

    # No revisions occurred, so revision impacts must be empty / zero
    assert len(res.revision_table) == 0
    assert len(res.news_table) == 3
    assert abs(sum(res.impact_releases.values()) - res.total_impact) < 1e-10


def test_banbura_modugno_pure_revisions_identity(nowcast_panel_dgp):
    """Verify |revision - total_impact| < 10^-10 when only historical data are revised."""
    df_true, _ = nowcast_panel_dgp
    df_old = df_true.copy()
    df_old.iloc[-1, 1:] = np.nan

    model = DynamicFactorModel(n_factors=1, p=1, standardize=True).fit(df_old)

    # New vintage revises historical values 6 and 12 months ago
    df_new = df_old.copy()
    df_new.iloc[-6, 1] += 0.75  # IP revised up
    df_new.iloc[-12, 2] -= 0.40  # Retail revised down

    res = banbura_modugno_news(model, df_old, df_new, target_series="gdp")

    assert res.decomposition_error < 1e-10
    assert abs(res.revision - res.total_impact) < 1e-10
    assert len(res.news_table) == 0
    assert len(res.revision_table) == 2
    assert abs(sum(res.impact_revisions.values()) - res.total_impact) < 1e-10


def test_banbura_modugno_combined_releases_and_revisions(nowcast_panel_dgp):
    """Verify analytical identity holds under simultaneous new releases and historical revisions."""
    df_true, _ = nowcast_panel_dgp
    df_old = df_true.copy()
    df_old.iloc[-1, 1:] = np.nan
    df_old.iloc[-2, 3:] = np.nan

    model = DynamicFactorModel(n_factors=2, p=1, standardize=True).fit(df_old)

    df_new = df_old.copy()
    # Revisions
    df_new.iloc[-5, 1] += 0.5
    df_new.iloc[-10, 4] -= 0.3
    # Releases
    df_new.iloc[-2, 3] = df_true.iloc[-2, 3]
    df_new.iloc[-1, 1] = df_true.iloc[-1, 1]
    df_new.iloc[-1, 4] = df_true.iloc[-1, 4]

    res = banbura_modugno_news(model, df_old, df_new, target_series="gdp")

    assert res.decomposition_error < 1e-10
    assert abs(res.revision - res.total_impact) < 1e-10
    assert len(res.news_table) == 3
    assert len(res.revision_table) == 2


def test_banbura_modugno_pro_and_counter_cyclical_signs(nowcast_panel_dgp):
    """Verify positive news on pro-cyclical indicator raises GDP nowcast,
    while positive news on counter-cyclical indicator lowers GDP nowcast."""
    df_true, _ = nowcast_panel_dgp
    df_old = df_true.copy()
    df_old.iloc[-1, 1:] = np.nan

    model = DynamicFactorModel(n_factors=1, p=1, standardize=True).fit(df_old)

    # 1. Release positive surprise for ind_prod (pro-cyclical)
    df_new_ip = df_old.copy()
    df_new_ip.iloc[-1, 1] = df_old["ind_prod"].mean() + 4.0 * df_old["ind_prod"].std()
    res_ip = banbura_modugno_news(model, df_old, df_new_ip, target_series="gdp")

    # Positive surprise on pro-cyclical variable should increase nowcast
    ip_news = res_ip.news_table[res_ip.news_table["series"] == "ind_prod"].iloc[0]
    assert ip_news["surprise"] > 0
    assert ip_news["weight"] > 0
    assert ip_news["impact"] > 0
    assert res_ip.revision > 0

    # 2. Release positive surprise for unemployment (counter-cyclical)
    df_new_unemp = df_old.copy()
    df_new_unemp.iloc[-1, 4] = df_old["unemployment"].mean() + 4.0 * df_old["unemployment"].std()
    res_unemp = banbura_modugno_news(model, df_old, df_new_unemp, target_series="gdp")

    unemp_news = res_unemp.news_table[res_unemp.news_table["series"] == "unemployment"].iloc[0]
    assert unemp_news["surprise"] > 0
    assert unemp_news["weight"] < 0
    assert unemp_news["impact"] < 0
    assert res_unemp.revision < 0


def test_banbura_modugno_multi_step_vintages(nowcast_panel_dgp):
    """Verify cumulative revisions across sequence of vintages equals sum of impacts."""
    df_true, _ = nowcast_panel_dgp
    df_v1 = df_true.copy()
    df_v1.iloc[-1, 1:] = np.nan
    df_v1.iloc[-2, 2:] = np.nan

    model = DynamicFactorModel(n_factors=1, p=1, standardize=True).fit(df_v1)

    # Vintage 2: release at T-2
    df_v2 = df_v1.copy()
    df_v2.iloc[-2, 2] = df_true.iloc[-2, 2]
    res_1_2 = banbura_modugno_news(model, df_v1, df_v2, target_series="gdp")

    # Vintage 3: release at T-1
    df_v3 = df_v2.copy()
    df_v3.iloc[-1, 1] = df_true.iloc[-1, 1]
    res_2_3 = banbura_modugno_news(model, df_v2, df_v3, target_series="gdp")

    # Step 1 -> 3 directly
    res_1_3 = banbura_modugno_news(model, df_v1, df_v3, target_series="gdp")

    assert res_1_2.decomposition_error < 1e-10
    assert res_2_3.decomposition_error < 1e-10
    assert res_1_3.decomposition_error < 1e-10
    # Additivity across steps
    assert abs((res_1_2.revision + res_2_3.revision) - res_1_3.revision) < 1e-10


def test_banbura_modugno_presentation_and_plots(nowcast_panel_dgp):
    """Verify summary, to_frame, to_markdown, to_latex, to_typst, and plot."""
    df_true, _ = nowcast_panel_dgp
    df_old = df_true.copy()
    df_old.iloc[-1, 1:] = np.nan

    model = DynamicFactorModel(n_factors=1, p=1, standardize=True).fit(df_old)
    df_new = df_old.copy()
    df_new.iloc[-1, 1] = df_true.iloc[-1, 1]
    df_new.iloc[-3, 2] += 0.2

    res = banbura_modugno_news(model, df_old, df_new, target_series="gdp")

    # Summary string
    s = res.summary()
    assert "Bańbura & Modugno (2014)" in s
    assert "New Releases (Innovations):" in s
    assert "Data Revisions" in s
    assert "Decomposition Identity Error" in s

    # Table converters
    df_frame = res.to_frame()
    assert isinstance(df_frame, pd.DataFrame)
    assert not df_frame.empty
    assert "impact" in df_frame.columns

    md = res.to_markdown()
    assert "| release |" in md or "| revision |" in md

    latex = res.to_latex()
    assert "\\begin{tabular}" in latex

    typst = res.to_typst()
    assert "#table(" in typst

    # Plot
    fig = res.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_model_news_convenience_wrapper(nowcast_panel_dgp):
    """Verify DynamicFactorModel.news(...) delegates to banbura_modugno_news."""
    df_true, _ = nowcast_panel_dgp
    df_old = df_true.copy()
    df_old.iloc[-1, 1:] = np.nan

    model = DynamicFactorModel(n_factors=1, p=1, standardize=True).fit(df_old)
    df_new = df_old.copy()
    df_new.iloc[-1, 1] = df_true.iloc[-1, 1]

    res = model.news(df_old, df_new, target_series="gdp")
    assert isinstance(res, NewsDecompositionResult)
    assert res.decomposition_error < 1e-10


def test_banbura_modugno_no_change(nowcast_panel_dgp):
    """Verify handling when old and new vintages are identical."""
    df_true, _ = nowcast_panel_dgp
    model = DynamicFactorModel(n_factors=1, p=1, standardize=True).fit(df_true)
    res = banbura_modugno_news(model, df_true, df_true, target_series="gdp")

    assert res.revision == 0.0
    assert res.total_impact == 0.0
    assert res.decomposition_error == 0.0
    assert len(res.news_table) == 0
    assert len(res.revision_table) == 0


def test_banbura_modugno_raw_arrays():
    """Verify support for raw numpy arrays without pandas Index."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4))
    X_old = X.copy()
    X_old[-1, 2:] = np.nan
    X_new = X_old.copy()
    X_new[-1, 2] = X[-1, 2]

    model = DynamicFactorModel(n_factors=1, p=1, standardize=False).fit(X_old)
    res = banbura_modugno_news(model, X_old, X_new, target_series=0)

    assert res.decomposition_error < 1e-10
    assert abs(res.revision - res.total_impact) < 1e-10
