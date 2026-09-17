"""Tests for forecast evaluation: PIT uniformity, Berkowitz LR test, scoring rules, and fan charts."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.nowcast.evaluation import (
    FanChartResult,
    PITUniformityResult,
    brier_score,
    crps_ensemble,
    crps_gaussian,
    fan_chart,
    log_score_gaussian,
    pit_histogram,
    pit_uniformity_test,
)


def test_pit_uniformity_calibrated():
    """Verify calibrated predictive distribution passes Berkowitz LR and KS tests."""
    rng = np.random.default_rng(42)
    T = 250
    mu = rng.normal(size=T)
    sigma = rng.uniform(0.8, 1.5, size=T)
    y = mu + sigma * rng.normal(size=T)

    res = pit_uniformity_test(y, mu=mu, sigma=sigma)

    assert isinstance(res, PITUniformityResult)
    assert res.is_uniform is True
    assert res.lr_pvalue > 0.05
    assert res.ks_pvalue > 0.05
    assert abs(res.mu) < 0.2
    assert 0.8 < res.sigma < 1.2
    assert abs(res.rho) < 0.2
    assert len(res.hist_counts) == 10


def test_pit_uniformity_biased_forecast():
    """Verify biased forecast distribution is sharply rejected by Berkowitz test."""
    rng = np.random.default_rng(43)
    T = 200
    mu_true = np.zeros(T)
    sigma_true = np.ones(T)
    y = rng.normal(size=T)

    # Forecast has positive bias (+1.5)
    mu_fc = mu_true + 1.5
    res = pit_uniformity_test(y, mu=mu_fc, sigma=sigma_true)

    assert res.is_uniform is False
    assert res.lr_pvalue < 1e-4
    # The transformed errors will have negative mean
    assert res.mu < -0.8


def test_pit_uniformity_underdispersed():
    """Verify underdispersed forecast (too narrow intervals) is rejected with high sigma."""
    rng = np.random.default_rng(44)
    T = 200
    y = rng.normal(scale=1.5, size=T)

    # Forecaster assumes standard deviation is only 0.5
    mu_fc = np.zeros(T)
    sigma_fc = np.full(T, 0.5)

    res = pit_uniformity_test(y, mu=mu_fc, sigma=sigma_fc)

    assert res.is_uniform is False
    assert res.lr_pvalue < 1e-4
    # Estimated sigma of z should be significantly greater than 1
    assert res.sigma > 1.5


def test_pit_uniformity_overdispersed():
    """Verify overdispersed forecast (too wide intervals) is rejected with low sigma."""
    rng = np.random.default_rng(45)
    T = 200
    y = rng.normal(scale=0.5, size=T)

    # Forecaster assumes standard deviation is 2.0
    mu_fc = np.zeros(T)
    sigma_fc = np.full(T, 2.0)

    res = pit_uniformity_test(y, mu=mu_fc, sigma=sigma_fc)

    assert res.is_uniform is False
    assert res.lr_pvalue < 1e-4
    # Estimated sigma of z should be significantly less than 1
    assert res.sigma < 0.6


def test_pit_uniformity_autocorrelated_errors():
    """Verify persistent forecast errors are rejected through the AR(1) coefficient."""
    rng = np.random.default_rng(46)
    T = 300
    # True error has AR(1) persistence
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = 0.6 * u[t - 1] + rng.normal(scale=np.sqrt(1 - 0.6**2))

    y = u
    # Unconditional forecast: N(0, 1)
    res = pit_uniformity_test(y, mu=np.zeros(T), sigma=np.ones(T))

    assert res.rho > 0.4
    assert res.lr_pvalue < 0.05


def test_pit_uniformity_ensemble_input():
    """Verify PIT test works with ensemble draws matrix."""
    rng = np.random.default_rng(47)
    T, M = 150, 200
    mu = rng.normal(size=T)
    sigma = rng.uniform(0.7, 1.3, size=T)
    y = mu + sigma * rng.normal(size=T)

    ensemble = mu[:, None] + sigma[:, None] * rng.normal(size=(T, M))
    res = pit_uniformity_test(y, ensemble_or_pit=ensemble)

    assert isinstance(res, PITUniformityResult)
    assert res.is_uniform is True
    assert res.lr_pvalue > 0.05


def test_pit_uniformity_presentation_and_plots():
    """Verify PITUniformityResult methods: summary, to_frame, to_markdown, to_latex, to_typst, plot."""
    rng = np.random.default_rng(48)
    y = rng.normal(size=100)
    res = pit_uniformity_test(y, mu=np.zeros(100), sigma=np.ones(100))

    s = res.summary()
    assert "Probability Integral Transform (PIT) Calibration" in s
    assert "Berkowitz (2001) Likelihood Ratio Test" in s
    assert "Kolmogorov-Smirnov" in s

    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert "Statistic" in df.columns
    assert "P-Value" in df.columns

    md = res.to_markdown()
    assert "Berkowitz LR" in md

    latex = res.to_latex()
    assert "\\begin{tabular}" in latex

    typst = res.to_typst()
    assert "#table(" in typst

    fig = res.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_scoring_rules_crps_and_log_score():
    """Verify strictly proper scoring rules (CRPS and log score)."""
    rng = np.random.default_rng(49)
    T = 50
    y = rng.normal(loc=2.0, scale=1.0, size=T)

    # Model A: true distribution N(2.0, 1.0)
    crps_a = crps_gaussian(y, mu=np.full(T, 2.0), sigma=np.full(T, 1.0)).mean()
    ls_a = log_score_gaussian(y, mu=np.full(T, 2.0), sigma=np.full(T, 1.0)).mean()

    # Model B: biased distribution N(4.0, 1.0)
    crps_b = crps_gaussian(y, mu=np.full(T, 4.0), sigma=np.full(T, 1.0)).mean()
    ls_b = log_score_gaussian(y, mu=np.full(T, 4.0), sigma=np.full(T, 1.0)).mean()

    # Lower CRPS is better; higher log score is better
    assert crps_a < crps_b
    assert ls_a > ls_b

    # Ensemble CRPS
    ensemble_a = 2.0 + rng.normal(size=(T, 100))
    crps_ens = crps_ensemble(y, ensemble_a).mean()
    assert abs(crps_ens - crps_a) < 0.15

    # Brier score
    y_bin = (y > 2.0).astype(float)
    p_good = np.full(T, 0.5)
    p_bad = np.full(T, 0.9)
    assert brier_score(y_bin, p_good).mean() < brier_score(y_bin, p_bad).mean()


def test_fan_chart_gaussian_intervals():
    """Verify fan chart calculation, interval nesting, and quantile monotonicity."""
    history = pd.Series([1.5, 1.8, 2.1, 2.0], index=["2023Q1", "2023Q2", "2023Q3", "2023Q4"])
    forecast_mean = pd.Series([2.2, 2.4, 2.5, 2.6], index=["2024Q1", "2024Q2", "2024Q3", "2024Q4"])
    forecast_sd = pd.Series([0.3, 0.45, 0.6, 0.75], index=forecast_mean.index)

    res = fan_chart(history, forecast_mean, forecast_sd, levels=(0.3, 0.6, 0.9), palette="banxico")

    assert isinstance(res, FanChartResult)
    assert res.palette == "banxico"
    assert len(res.intervals) == 3

    # Check nesting: 30% interval must be inside 60% interval, which is inside 90%
    lo_30, hi_30 = res.intervals[0.3]
    lo_60, hi_60 = res.intervals[0.6]
    lo_90, hi_90 = res.intervals[0.9]

    assert (lo_90 < lo_60).all()
    assert (lo_60 < lo_30).all()
    assert (hi_30 < hi_60).all()
    assert (hi_60 < hi_90).all()


def test_fan_chart_ensemble_input():
    """Verify fan chart constructed from empirical ensemble draws."""
    rng = np.random.default_rng(50)
    H, M = 4, 500
    history = pd.Series([1.0, 1.2, 1.3])
    fc_mean = pd.Series([1.5, 1.7, 1.8, 2.0])
    ensemble = fc_mean.values[:, None] + rng.normal(scale=0.4, size=(H, M))

    res = fan_chart(history, fc_mean, ensemble=ensemble, levels=(0.5, 0.8), palette="bcb")

    assert isinstance(res, FanChartResult)
    assert res.palette == "bcb"
    lo_50, hi_50 = res.intervals[0.5]
    lo_80, hi_80 = res.intervals[0.8]
    assert (lo_80 < lo_50).all()
    assert (hi_50 < hi_80).all()


def test_fan_chart_presentation_and_plots():
    """Verify summary, table exports, and rendering across central bank themes."""
    history = pd.Series([2.0, 2.2], index=["t-1", "t"])
    fc_mean = pd.Series([2.3, 2.4], index=["t+1", "t+2"])
    fc_sd = pd.Series([0.2, 0.3], index=fc_mean.index)

    for palette in ("banxico", "bcb", "bank_of_england", "default"):
        res = fan_chart(history, fc_mean, fc_sd, palette=palette)
        s = res.summary()
        assert f"{palette.upper()} Theme" in s
        assert "Confidence levels" in s

        df = res.to_frame()
        assert not df.empty
        assert "mean" in df.columns

        md = res.to_markdown()
        assert "| mean |" in md

        latex = res.to_latex()
        assert "\\begin{tabular}" in latex

        typst = res.to_typst()
        assert "#table(" in typst

        fig = res.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
