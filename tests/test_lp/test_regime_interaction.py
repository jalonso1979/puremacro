"""Tests for lp.panel.lp_panel_regime_interaction (ported from legacy)."""
import numpy as np
import pandas as pd
import pytest


def _toy_regime_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in "ABCDE":
        for t in range(60):
            regime = 0 if t < 30 else 1
            rows.append({
                "code": c, "date": t,
                "shock": rng.standard_normal(),
                "y": rng.standard_normal(),
                "regime": regime,
            })
    return pd.DataFrame(rows)


def test_lp_panel_regime_interaction_returns_long_dataframe():
    """The function returns a long-form DataFrame keyed by (h, regime).

    PINS the canonical kwarg names: y/x/regime_col/horizons/n_lags/
    entity_level/time_level/alpha. Task 5 rewrites callers to match.
    """
    from puremacro.lp.panel import lp_panel_regime_interaction

    df = _toy_regime_panel()
    result = lp_panel_regime_interaction(
        df,
        y="y", x="shock",
        regime_col="regime",
        horizons=range(0, 5),
        n_lags=1,
        entity_level="code",
        time_level="date",
    )
    assert isinstance(result, pd.DataFrame)
    # Must expose per-regime beta + SE at each horizon.
    assert "h" in result.columns
    assert "regime" in result.columns
    assert "beta" in result.columns
    assert "se" in result.columns
    assert set(result["regime"].unique()) == {0, 1}
    assert set(result["h"].unique()) == set(range(0, 5))


def test_lp_panel_regime_interaction_respects_alpha_kwarg():
    """alpha=0.10 -> 90% CI columns lo/hi populated."""
    from puremacro.lp.panel import lp_panel_regime_interaction

    df = _toy_regime_panel()
    result = lp_panel_regime_interaction(
        df, y="y", x="shock", regime_col="regime",
        horizons=range(0, 3), n_lags=1,
        entity_level="code", time_level="date",
        alpha=0.10,
    )
    # CI band columns must be present.
    assert {"lo", "hi"}.issubset(set(result.columns)) or \
           {"ci_lo", "ci_hi"}.issubset(set(result.columns))
