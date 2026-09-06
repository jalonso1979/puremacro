"""Tests for Mixed-Frequency Dynamic Factor Model GDP Nowcasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.nowcast.dfm_nowcast import NowcastResult, nowcast_gdp


@pytest.fixture
def synthetic_nowcast_data():
    rng = np.random.default_rng(42)
    n_months = 60
    n_series = 10
    
    dates_m = pd.date_range("2019-01-01", periods=n_months, freq="MS")
    # True factor
    F_true = np.zeros(n_months)
    for t in range(1, n_months):
        F_true[t] = 0.8 * F_true[t-1] + rng.normal()

    # Monthly indicators
    X = np.zeros((n_months, n_series))
    for i in range(n_series):
        load = rng.uniform(0.5, 1.5)
        X[:, i] = load * F_true + rng.normal(scale=0.3, size=n_months)

    df_X = pd.DataFrame(X, index=dates_m, columns=[f"Series_{i+1}" for i in range(n_series)])
    
    # Introduce ragged edges in the last month: series 5 to 9 are missing
    df_X.iloc[-1, 5:] = np.nan

    # Quarterly GDP
    dates_q = pd.date_range("2019-01-01", periods=n_months // 3, freq="QS")
    # Quarterly average factor
    F_q = df_X.resample("QE").mean().to_numpy().mean(axis=1)[:len(dates_q)]
    gdp = 2.0 + 1.2 * F_q + rng.normal(scale=0.2, size=len(dates_q))
    s_gdp = pd.Series(gdp, index=dates_q.to_period("Q").astype(str), name="GDP")

    return df_X, s_gdp


def test_nowcast_gdp_basic(synthetic_nowcast_data):
    df_X, s_gdp = synthetic_nowcast_data
    res = nowcast_gdp(df_X, s_gdp, n_factors=2)

    assert isinstance(res, NowcastResult)
    assert isinstance(res.nowcast, float)
    assert not np.isnan(res.nowcast)
    assert res.factors.shape == (60, 2)
    assert res.loadings.shape == (10, 2)
    assert 0.0 <= res.model_r2 <= 1.0
    assert not res.news_decomposition.empty

    s = res.summary()
    assert "Dynamic Factor Model GDP Nowcasting" in s
    assert "GDP Growth Nowcast" in s
    assert "Quarterly Bridge Regression R²" in s


# ---------------------------------------------------------------------------
# Regression tests (v2.3.x audit fixes)
# ---------------------------------------------------------------------------
import warnings


def _one_factor_panel(seed, T_m=120, N=10, phi=0.85, sd_f=0.5, sd_e=0.3):
    """One-factor monthly panel with known loadings and a quarterly GDP
    series driven by the three-month factor average."""
    rng = np.random.default_rng(seed)
    F = np.zeros(T_m)
    for t in range(1, T_m):
        F[t] = phi * F[t - 1] + rng.normal(scale=sd_f)
    lam = rng.uniform(0.5, 1.5, N)
    X = F[:, None] * lam[None, :] + rng.normal(scale=sd_e, size=(T_m, N))
    dates = pd.date_range("2015-01-01", periods=T_m, freq="MS")
    monthly = pd.DataFrame(X, index=dates, columns=[f"ind_{i}" for i in range(N)])
    nq = T_m // 3
    g = 1.0 + 2.0 * F.reshape(nq, 3).mean(axis=1) + rng.normal(scale=0.1, size=nq)
    qidx = pd.period_range("2015Q1", periods=nq, freq="Q").astype(str)
    return monthly, F, lam, g, qidx


def test_em_imputation_uses_rank_k_reconstruction_ragged_edge_factor_unbiased():
    """Regression for the EM scaling bug: the old EM step imputed missing
    entries with ``sqrt(T) U_k V_k'`` (singular values dropped), so every
    imputed value was ~sqrt(T)/S_k (about a third) of the right size and
    the factor of a partially-published month converged to a shrunken
    fixed point (slope 0.27 on the complete-panel factor; nowcast RMSE
    0.49 against 0.17 with the complete panel). With the rank-k
    reconstruction ``U_k S_k V_k'`` the slope is ~1 and the RMSE is close
    to the complete-panel one."""
    rows = []
    for s in range(40):
        monthly, F, lam, g, qidx = _one_factor_panel(200 + s)
        mon = monthly.copy()
        mon.iloc[-1, 2:] = np.nan                 # 2 of 10 series published
        gdp = pd.Series(g[:-1], index=qidx[:-1])
        r_full = nowcast_gdp(monthly, gdp, n_factors=1)
        r_mask = nowcast_gdp(mon, gdp, n_factors=1)
        sgn = np.sign(np.corrcoef(r_full.factors.iloc[:100, 0], r_mask.factors.iloc[:100, 0])[0, 1])
        # imputed standardised values vs the truth (std scale)
        true_std = ((monthly - mon.mean()) / mon.std()).iloc[-1, 2:].to_numpy()
        imputed = (r_mask.factors.iloc[-1, 0] * r_mask.loadings.iloc[2:, 0].to_numpy())
        rows.append((r_full.factors.iloc[-1, 0], sgn * r_mask.factors.iloc[-1, 0],
                     np.mean(imputed / true_std),
                     r_full.nowcast - g[-1], r_mask.nowcast - g[-1]))
    rows = np.array(rows)
    slope = np.polyfit(rows[:, 0], rows[:, 1], 1)[0]
    rmse_full = np.sqrt(np.mean(rows[:, 3] ** 2))
    rmse_mask = np.sqrt(np.mean(rows[:, 4] ** 2))
    assert slope > 0.85, f"ragged-edge factor slope {slope:.3f} (old code: 0.27)"
    assert np.median(rows[:, 2]) > 0.6, f"imputed/true ratio {np.median(rows[:, 2]):.3f} (old code: 0.08)"
    assert rmse_mask < 1.6 * rmse_full and rmse_mask < 0.35, (rmse_full, rmse_mask)


def test_loadings_reconstruct_panel_and_news_forecast_on_data_scale():
    """``F @ Lambda.T`` must be the rank-K reconstruction of the standardised
    panel, and the news ``forecast`` column must be on the data scale (the
    old code returned fitted values ~0.34x too small)."""
    monthly, F, lam, g, qidx = _one_factor_panel(11)
    gdp = pd.Series(g[:-1], index=qidx[:-1])
    res = nowcast_gdp(monthly, gdp, n_factors=1)
    Xs = ((monthly - monthly.mean()) / monthly.std()).to_numpy()
    recon = res.factors.to_numpy() @ res.loadings.to_numpy().T
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    rank1 = (U[:, :1] * S[:1]) @ Vt[:1]
    assert np.allclose(recon, rank1, atol=1e-8)
    r2_recon = 1.0 - ((Xs - recon) ** 2).sum() / (Xs ** 2).sum()
    assert r2_recon > 0.8, r2_recon
    news = res.news_decomposition
    assert len(news) == monthly.shape[1]
    expected = monthly.mean().to_numpy() + monthly.std().to_numpy() * recon[-1]
    assert np.allclose(news["forecast"].to_numpy(), expected)
    # Old code: forecast = mean + std * sqrt(T)/S_1 * recon -> ~0.34x too small.
    assert np.allclose(news["surprise"], news["actual"] - news["forecast"])
    assert np.allclose(news["contribution"], news["weight"] * news["surprise"])


def test_p_factor_lags_is_used_and_factor_var_returned():
    """Regression: ``p_factor_lags`` was accepted and never read (1 vs 5
    gave bit-identical results). It now fits the factor VAR(p) that
    completes the target quarter."""
    monthly, F, lam, g, qidx = _one_factor_panel(7)
    mon = monthly.iloc[:-1]                       # target quarter has 2 months
    gdp = pd.Series(g[:-1], index=qidx[:-1])
    r1 = nowcast_gdp(mon, gdp, n_factors=1, p_factor_lags=1)
    r3 = nowcast_gdp(mon, gdp, n_factors=1, p_factor_lags=3)
    assert r1.factor_var is not None and r1.factor_var.shape == (1, 1)
    assert r3.factor_var is not None and r3.factor_var.shape == (3, 1)
    assert not np.isclose(r1.nowcast, r3.nowcast)
    assert 0.6 < r1.factor_var[0, 0] < 1.0        # true AR coefficient 0.85
    with pytest.raises(ValueError, match="p_factor_lags"):
        nowcast_gdp(mon, gdp, n_factors=1, p_factor_lags=0)


def test_target_quarter_completed_by_factor_var_forecast():
    """A frame ending in the first month of a quarter gets two forecast
    months, so the target quarter's factor average is a three-month
    average as the bridge was estimated on."""
    monthly, F, lam, g, qidx = _one_factor_panel(3)
    mon = monthly.iloc[:-2]                       # ends in the first month of 2024Q4
    gdp = pd.Series(g[:-1], index=qidx[:-1])
    res = nowcast_gdp(mon, gdp, n_factors=1)
    assert len(res.factor_forecast) == 2
    assert list(res.factor_forecast.index) == list(pd.DatetimeIndex(["2024-11-01", "2024-12-01"]))
    assert res.target_quarter == "2024Q4"
    assert res.factors.shape[0] == len(mon)       # `factors` keeps the frame's rows
    full = nowcast_gdp(monthly, gdp, n_factors=1)
    assert full.factor_forecast.empty
    # The completion is the VAR forecast from the last observed factor.
    B = res.factor_var
    f_last = res.factors.iloc[-1].to_numpy()
    assert np.allclose(res.factor_forecast.iloc[0].to_numpy(), f_last @ B)
    assert np.allclose(res.factor_forecast.iloc[1].to_numpy(), (f_last @ B) @ B)
    # Nowcast reproduces from the bridge coefficients and the completed quarter.
    q_avg = np.vstack([res.factors.iloc[-1:].to_numpy(), res.factor_forecast.to_numpy()]).mean(axis=0)
    b = res.bridge_coefficients.to_numpy()
    assert np.isclose(res.nowcast, b[0] + b[1:] @ q_avg)


def test_all_nan_padding_rows_equivalent_to_truncation():
    """Appending all-NaN rows for the remaining months of the target quarter
    must give the same nowcast as truncating the frame: both are completed
    by the factor VAR. (The old EM dragged padded rows to the panel mean
    and diluted the quarter average.)"""
    monthly, F, lam, g, qidx = _one_factor_panel(5)
    mon = monthly.iloc[:-2]
    gdp = pd.Series(g[:-1], index=qidx[:-1])
    padded = monthly.copy()
    padded.iloc[-2:, :] = np.nan
    r_trunc = nowcast_gdp(mon, gdp, n_factors=1)
    r_pad = nowcast_gdp(padded, gdp, n_factors=1)
    assert np.isclose(r_trunc.nowcast, r_pad.nowcast, rtol=1e-4, atol=1e-6)
    assert r_pad.factor_forecast.empty and r_pad.factors.shape[0] == len(padded)
    assert r_pad.news_decomposition.empty
    assert np.allclose(r_pad.factors.iloc[-2:].to_numpy() / r_trunc.factor_forecast.to_numpy(),
                       r_pad.factors.iloc[-3, 0] / r_trunc.factors.iloc[-1, 0], rtol=1e-4)


def test_period_and_datetime_gdp_index_align_by_label():
    """Regression: a PeriodIndex on ``quarterly_gdp`` crashed with
    ``LinAlgError: Incompatible dimensions`` and a DatetimeIndex silently
    fell back to positional alignment (wrong bridge when the GDP series
    starts later than the panel). Both now align by quarter label."""
    monthly, F, lam, g, qidx = _one_factor_panel(9)
    mon = monthly.iloc[:-1]
    gdp = pd.Series(g[:-1], index=qidx[:-1])
    ref = nowcast_gdp(mon, gdp, n_factors=1)
    gdp_p = gdp.copy(); gdp_p.index = pd.PeriodIndex(gdp.index, freq="Q")
    gdp_d = gdp.copy(); gdp_d.index = pd.date_range("2015-03-31", periods=len(gdp), freq="QE")
    gdp_qs = gdp.copy(); gdp_qs.index = pd.date_range("2015-01-01", periods=len(gdp), freq="QS")
    for other in (gdp_p, gdp_d, gdp_qs):
        r = nowcast_gdp(mon, other, n_factors=1)
        assert np.isclose(r.nowcast, ref.nowcast) and np.isclose(r.model_r2, ref.model_r2)
    # GDP starting four quarters after the panel: no positional fallback.
    late_ref = nowcast_gdp(mon, gdp.iloc[4:], n_factors=1)
    late_d = nowcast_gdp(mon, gdp_d.iloc[4:], n_factors=1)
    assert np.isclose(late_d.nowcast, late_ref.nowcast)
    assert late_d.model_r2 > 0.9


def test_label_mismatch_raises_instead_of_positional_fallback():
    """Regression: fewer than four matching quarter labels used to trigger a
    silent positional slice of the first min(len) rows."""
    monthly, F, lam, g, qidx = _one_factor_panel(2)
    gdp = pd.Series(g[:-1], index=qidx[:-1])
    with pytest.raises(ValueError, match="fewer than 4 quarter labels"):
        nowcast_gdp(monthly.reset_index(drop=True), gdp, n_factors=1)
    with pytest.raises(ValueError, match="fewer than 4 quarter labels"):
        nowcast_gdp(monthly, gdp.iloc[:3], n_factors=1)
    # A positional frame works with positional 'Q1', 'Q2', ... labels.
    gdp_pos = pd.Series(g[:-1], index=[f"Q{i + 1}" for i in range(len(g) - 1)])
    r = nowcast_gdp(monthly.reset_index(drop=True), gdp_pos, n_factors=1)
    assert r.target_quarter == "Q40" and r.model_r2 > 0.9


def test_zero_variance_gdp_reports_zero_r2_and_summary_has_no_annualized_claim():
    monthly, F, lam, g, qidx = _one_factor_panel(4)
    gdp = pd.Series(2.0, index=qidx[:-1])
    res = nowcast_gdp(monthly, gdp, n_factors=1)
    assert res.model_r2 == 0.0
    assert "annualized" not in res.summary()
    assert "units of quarterly_gdp" in res.summary()


def test_nowcast_result_presentation_contract():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    monthly, F, lam, g, qidx = _one_factor_panel(6)
    mon = monthly.iloc[:-1].copy()
    mon.iloc[-1, [3, 4]] = np.nan
    gdp = pd.Series(g[:-1], index=qidx[:-1])
    res = nowcast_gdp(mon, gdp, n_factors=2)
    md = res.to_markdown()
    lines = [l for l in md.splitlines() if "|" in l]
    assert len(lines) >= 2 and "---" in lines[1] and "surprise" in lines[0]
    assert "tabular" in res.to_latex()
    assert "#table(" in res.to_typst()
    assert res.to_frame().shape == (8, 5)
    fig = res.plot()
    assert isinstance(fig, Figure)
    plt.close("all")
    # empty news (all-NaN last row) still renders
    mon2 = mon.copy(); mon2.iloc[-1, :] = np.nan
    res2 = nowcast_gdp(mon2, gdp, n_factors=1)
    assert res2.news_decomposition.empty
    assert "|" in res2.to_markdown() and "tabular" in res2.to_latex()
    plt.close(res2.plot())


def test_invalid_inputs_raise():
    monthly, F, lam, g, qidx = _one_factor_panel(8)
    gdp = pd.Series(g[:-1], index=qidx[:-1])
    with pytest.raises(ValueError, match="n_factors"):
        nowcast_gdp(monthly, gdp, n_factors=0)
    with pytest.raises(ValueError, match="n_factors"):
        nowcast_gdp(monthly, gdp, n_factors=11)
    with pytest.raises(TypeError):
        nowcast_gdp(monthly.to_numpy(), gdp, n_factors=1)
