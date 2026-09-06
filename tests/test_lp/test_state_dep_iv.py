"""Unit tests for State-Dependent LP-IV (Ramey & Zubairy 2018)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.lp import lp_state_dep_iv, LPResult


@pytest.fixture
def rz_style_data():
    rng = np.random.default_rng(2018)
    T = 150
    # State variable (e.g. unemployment rate around 6.5%)
    state = 6.5 + 1.2 * rng.standard_normal(T)
    # Military news instrument
    z = rng.standard_normal(T)
    # Government spending: strong first stage with news
    g = 0.7 * z + 0.3 * rng.standard_normal(T)
    # Outcome: state-dependent spending response
    # High slack (state > 6.5) vs low slack (state <= 6.5)
    high_state = (state > 6.5).astype(float)
    y = np.cumsum(0.8 * (high_state * g) + 0.5 * ((1.0 - high_state) * g) + 0.2 * rng.standard_normal(T))
    return pd.DataFrame({"y": y, "g": g, "news": z, "unemp": state})


def test_lp_state_dep_iv_threshold(rz_style_data):
    df = rz_style_data
    res = lp_state_dep_iv(
        df,
        y="y",
        x="g",
        z="news",
        state="unemp",
        threshold=6.5,
        transition="threshold",
        horizon=4,
        lags=2,
        ci=0.90,
    )

    assert isinstance(res, LPResult)
    assert len(res) == 5  # h = 0..4
    expected_cols = [
        "h", "beta_H", "se_H", "lo_H", "hi_H",
        "beta_L", "se_L", "lo_L", "hi_L",
        "first_stage_f_H", "first_stage_f_L",
    ]
    for col in expected_cols:
        assert col in res.columns

    # Estimates should be finite numbers
    assert np.all(np.isfinite(res["beta_H"].values))
    assert np.all(np.isfinite(res["beta_L"].values))
    assert np.all(res["se_H"].values > 0)
    assert np.all(res["se_L"].values > 0)


def test_lp_state_dep_iv_logistic(rz_style_data):
    # ``threshold`` is on the raw scale of the state (6.5 % unemployment),
    # exactly as in the threshold-transition call above. (This test used to
    # pass threshold=0.0, which only "worked" because the old code silently
    # compared the cutoff to the standardised state.)
    df = rz_style_data
    res = lp_state_dep_iv(
        df,
        y="y",
        x="g",
        z="news",
        state="unemp",
        threshold=6.5,
        transition="logistic",
        gamma=2.5,
        horizon=2,
        lags=1,
    )

    assert isinstance(res, LPResult)
    assert len(res) == 3
    assert np.all(np.isfinite(res["beta_H"].values))
    assert np.all(np.isfinite(res["beta_L"].values))
    # The default (threshold=None) splits at the sample mean and runs too.
    res_mean = lp_state_dep_iv(df, y="y", x="g", z="news", state="unemp",
                               transition="logistic", gamma=2.5, horizon=2, lags=1)
    assert len(res_mean) == 3


def test_lp_state_dep_iv_export_methods(rz_style_data):
    df = rz_style_data
    res = lp_state_dep_iv(
        df,
        y="y",
        x="g",
        z="news",
        state="unemp",
        horizon=2,
        lags=1,
    )

    md = res.to_markdown()
    assert "| h |" in md or "| beta_H |" in md

    ltx = res.to_latex()
    assert "\\begin{tabular}" in ltx

    typ = res.to_typst()
    assert "#table(" in typ
