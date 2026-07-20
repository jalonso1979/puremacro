"""Tests for lp.state_dep.lp_smooth_transition_irf (ported from legacy)."""
import numpy as np
import pandas as pd
import pytest


def _toy_series(T: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "y": rng.standard_normal(T).cumsum(),
        "shock": rng.standard_normal(T),
        "state_var": np.linspace(-2, 2, T) + 0.5 * rng.standard_normal(T),
    })


def test_lp_smooth_transition_irf_returns_dataframe():
    """Function returns a long-form DataFrame with per-regime betas."""
    from puremacro.lp.state_dep import lp_smooth_transition_irf

    df = _toy_series()
    result = lp_smooth_transition_irf(
        df, y="y", x="shock", state_var="state_var",
        horizons=range(0, 5), n_lags=2, gamma=1.0,
    )
    assert isinstance(result, pd.DataFrame)
    assert "h" in result.columns
    # Expect per-regime beta columns (high/low state).
    cols = set(result.columns)
    assert ({"beta_high", "beta_low"}.issubset(cols)
            or {"regime", "beta"}.issubset(cols))


def test_lp_smooth_transition_irf_gamma_scaling():
    """Larger gamma should produce a sharper transition (different betas)."""
    from puremacro.lp.state_dep import lp_smooth_transition_irf

    df = _toy_series()
    soft = lp_smooth_transition_irf(df, y="y", x="shock", state_var="state_var",
                                     horizons=range(0, 3), n_lags=1, gamma=0.5)
    hard = lp_smooth_transition_irf(df, y="y", x="shock", state_var="state_var",
                                     horizons=range(0, 3), n_lags=1, gamma=5.0)
    # Should not be identical — gamma changes the transition function.
    assert not soft.equals(hard)
