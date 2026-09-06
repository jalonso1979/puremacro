"""Regression tests: lp_did accepts the 2.0 keyword aliases
``horizon`` / ``ci`` / ``lags`` like every other puremacro.lp estimator.

Before the fix ``lp_did(panel, ..., horizon=4)`` raised ``TypeError:
unexpected keyword argument 'horizon'`` (M34), so docs/lp.md's claim that
every estimator accepts lags/horizon/ci was false for lp_did.
"""
import numpy as np
import pandas as pd
import pytest

from puremacro.lp import lp_did


def _staggered_panel(seed: int = 0, N: int = 40, T: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    adopt = rng.choice([8, 12, 10 ** 6], size=N, p=[0.35, 0.35, 0.30])
    for i in range(N):
        alpha_i = rng.standard_normal()
        for t in range(T):
            D = float(t >= adopt[i])
            since = t - adopt[i]
            eff = 1.0 + 0.2 * since if D else 0.0
            rows.append({"unit": i, "time": t, "D": D,
                         "y": alpha_i + 0.1 * t + eff + 0.3 * rng.standard_normal()})
    return pd.DataFrame(rows)


def test_horizon_and_ci_aliases_match_canonical_arguments():
    pan = _staggered_panel()
    canon = lp_did(pan, "y", "D", "unit", "time", range(0, 5), pre_window=0, alpha=0.05)
    alias = lp_did(pan, "y", "D", "unit", "time", horizon=4, pre_window=0, ci=0.95)
    pd.testing.assert_frame_equal(canon.estimates, alias.estimates)
    assert alias.alpha == pytest.approx(0.05)


def test_lags_adds_lagged_outcome_changes_as_controls():
    """``lags=k`` adds Δy_{i,t-1..t-k} as pre-determined controls
    (DGJT's dylags). lags=0 is the baseline; k>0 changes the estimates,
    keeps them finite and drops the observations without history."""
    pan = _staggered_panel()
    base = lp_did(pan, "y", "D", "unit", "time", range(0, 4), pre_window=0)
    zero = lp_did(pan, "y", "D", "unit", "time", range(0, 4), pre_window=0, lags=0)
    pd.testing.assert_frame_equal(base.estimates, zero.estimates)
    lagged = lp_did(pan, "y", "D", "unit", "time", range(0, 4), pre_window=0, lags=2)
    assert np.all(np.isfinite(lagged.estimates["beta"].values))
    assert not np.allclose(base.estimates["beta"].values, lagged.estimates["beta"].values)
    assert (lagged.estimates["n_obs"].values <= base.estimates["n_obs"].values).all()
    # Same as constructing the lagged changes by hand and passing them as controls
    p = pan.sort_values(["unit", "time"]).copy()
    g = p.groupby("unit")["y"]
    p["dy1"] = g.shift(1) - g.shift(2)
    p["dy2"] = g.shift(2) - g.shift(3)
    manual = lp_did(p, "y", "D", "unit", "time", range(0, 4), pre_window=0,
                    controls=["dy1", "dy2"])
    np.testing.assert_allclose(lagged.estimates["beta"].values, manual.estimates["beta"].values)
    np.testing.assert_allclose(lagged.estimates["se"].values, manual.estimates["se"].values)
    # ATT at h=0 remains close to the true effect (1.0) with the extra controls
    assert abs(float(lagged.estimates.loc[lagged.estimates["h"] == 0, "beta"].iloc[0]) - 1.0) < 0.3


def test_lp_did_validates_alias_scale():
    pan = _staggered_panel()
    with pytest.raises(ValueError, match="ci"):
        lp_did(pan, "y", "D", "unit", "time", [0], pre_window=0, ci=90)
    with pytest.raises(ValueError, match="lags"):
        lp_did(pan, "y", "D", "unit", "time", [0], pre_window=0, lags=-1)
    with pytest.raises(ValueError, match="horizon"):
        lp_did(pan, "y", "D", "unit", "time", pre_window=0, horizon=-2)
