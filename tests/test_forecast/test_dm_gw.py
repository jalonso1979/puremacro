import numpy as np
import pytest

from puremacro.forecast.compare import diebold_mariano, giacomini_white


def test_dm_equal_loss_uniform_p_value():
    """Two i.i.d. forecast-error series with the same loss -> DM p-value
    distributed near 0.5 (large samples)."""
    rng = np.random.default_rng(11)
    e1 = rng.standard_normal(500)
    e2 = rng.standard_normal(500)
    out = diebold_mariano(e1, e2, h=1, loss="mse")
    assert "stat" in out and "p_value" in out
    # Same distribution -> should NOT reject equal predictive ability
    assert 0.05 < out["p_value"] < 0.95


def test_dm_detects_dominant_forecast():
    """If e1 has variance 1 and e2 has variance 4, DM should reject equal
    loss in favour of method 1."""
    rng = np.random.default_rng(13)
    e1 = rng.standard_normal(300)
    e2 = 2.0 * rng.standard_normal(300)
    out = diebold_mariano(e1, e2, h=1, loss="mse")
    assert out["p_value"] < 0.05
    # Negative stat means e1 has lower MSE
    assert out["stat"] < 0


def test_giacomini_white_returns_documented_keys():
    rng = np.random.default_rng(17)
    e1 = rng.standard_normal(200)
    e2 = rng.standard_normal(200)
    cond = rng.standard_normal((200, 2))
    out = giacomini_white(e1, e2, h=1, conditioning_vars=cond)
    for k in ("stat", "p_value", "df"):
        assert k in out
