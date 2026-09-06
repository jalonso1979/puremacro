import numpy as np
import pandas as pd

from puremacro.lp.state_dep import lp_state_dep


def test_lp_state_dep_returns_two_regimes():
    rng = np.random.default_rng(17)
    T = 240
    state = rng.standard_normal(T)
    x = rng.standard_normal(T)
    y = -0.2 * x + 0.4 * state * x + rng.standard_normal(T)
    df = pd.DataFrame({"y": y, "x": x, "state": state},
                      index=pd.date_range("1990-01-01", periods=T, freq="MS"))
    out = lp_state_dep(df, y="y", x="x", state="state",
                       horizons=range(0, 6), n_lags=2)
    assert {"beta_H", "beta_L", "se_H", "se_L"}.issubset(out.columns)
    assert np.all(np.isfinite(out[["beta_H", "beta_L"]].values))
    assert len(out) == 6


def test_lp_state_dep_threshold_mode():
    rng = np.random.default_rng(19)
    T = 200
    state = rng.standard_normal(T)
    x = rng.standard_normal(T)
    y = (state > 0).astype(float) * 0.5 * x + (state <= 0).astype(float) * (-0.3) * x + rng.standard_normal(T) * 0.2
    df = pd.DataFrame({"y": y, "x": x, "state": state},
                      index=pd.date_range("1990-01-01", periods=T, freq="MS"))
    out = lp_state_dep(df, y="y", x="x", state="state",
                       horizons=[0], n_lags=0, transition="threshold")
    # With clean state, beta_H (high state) ≈ 0.5, beta_L (low) ≈ -0.3
    assert out["beta_H"].iloc[0] > 0.2
    assert out["beta_L"].iloc[0] < 0.0


# ---------------------------------------------------------------------------
# Regression tests for the 2.3.x audit (state-dependent LP semantics & aliases)
# ---------------------------------------------------------------------------
import pytest

from puremacro.lp import lp_state_dep_iv, LPResult


def _regime_dgp(seed: int = 5, T: int = 400):
    """Unemployment-style state (mean 6, sd 1.5, in percent); the response
    to x is 1.0 when unemployment > 6.5 and 0.2 otherwise."""
    rng = np.random.default_rng(seed)
    u = 6.0 + 1.5 * rng.standard_normal(T)
    x = rng.standard_normal(T)
    z = 0.7 * x + 0.3 * rng.standard_normal(T)      # instrument for the IV test
    high = (u > 6.5).astype(float)
    y = np.cumsum((1.0 * high + 0.2 * (1.0 - high)) * x + 0.3 * rng.standard_normal(T))
    return pd.DataFrame({"y": y, "x": x, "z": z, "u_rate": u})


def test_lp_state_dep_accepts_2_0_aliases_and_state_var():
    """lp_state_dep used to be the only exported estimator without the
    keyword-only lags/horizon/ci aliases and rejected state_var= (C1, M34):
    ``lp_state_dep(df, ..., state_var=..., horizon=12, lags=2)`` raised
    TypeError. The aliases must give the same table as the legacy names."""
    df = _regime_dgp()
    legacy = lp_state_dep(df, y="y", x="x", state="u_rate", horizons=range(0, 5),
                          n_lags=2, alpha=0.10, transition="threshold", threshold=6.5)
    modern = lp_state_dep(df, y="y", x="x", state_var="u_rate", horizon=4,
                          lags=2, ci=0.90, transition="threshold", threshold=6.5)
    pd.testing.assert_frame_equal(pd.DataFrame(legacy), pd.DataFrame(modern))
    assert isinstance(modern, LPResult)
    assert list(modern.index) == [0, 1, 2, 3, 4]          # indexed by h like lp_hac
    assert modern.method == "LP-state-dep"
    assert modern.metadata["y_name"] == "y" and modern.metadata["x_name"] == "x"
    with pytest.raises(ValueError, match="state"):
        lp_state_dep(df, y="y", x="x", horizon=2)                      # no state at all
    with pytest.raises(ValueError, match="disagree"):
        lp_state_dep(df, y="y", x="x", state="u_rate", state_var="x", horizon=2)


def test_lp_state_dep_threshold_honoured_under_logistic():
    """Under transition='logistic' the old code dropped ``threshold``
    entirely (M31): threshold=0.0 and threshold=1.0 returned identical
    tables. Now the cutoff shifts the logistic centre, and the default
    (None) is the sample mean of the state."""
    rng = np.random.default_rng(3)
    T = 300
    s = rng.standard_normal(T)
    x = rng.standard_normal(T)
    y = np.cumsum(0.5 * x + 0.3 * s * x + 0.4 * rng.standard_normal(T))
    df = pd.DataFrame({"y": y, "x": x, "s": s})
    a = lp_state_dep(df, y="y", x="x", state="s", horizons=range(0, 3), threshold=0.0)
    b = lp_state_dep(df, y="y", x="x", state="s", horizons=range(0, 3), threshold=1.0)
    assert not np.allclose(a["beta_H"], b["beta_H"])
    at_mean = lp_state_dep(df, y="y", x="x", state="s", horizons=range(0, 3),
                           threshold=float(df["s"].mean()))
    default = lp_state_dep(df, y="y", x="x", state="s", horizons=range(0, 3))
    pd.testing.assert_frame_equal(pd.DataFrame(at_mean), pd.DataFrame(default))
    # Steeper gamma converges to the sharp threshold at the same cutoff.
    sharp = lp_state_dep(df, y="y", x="x", state="s", horizons=[0],
                         transition="threshold", threshold=0.3)
    steep = lp_state_dep(df, y="y", x="x", state="s", horizons=[0],
                         transition="logistic", gamma=400.0, threshold=0.3)
    np.testing.assert_allclose(steep["beta_H"].values, sharp["beta_H"].values, atol=1e-2)


def test_lp_state_dep_threshold_is_on_the_raw_state_scale():
    """The docs example passes threshold=6.5 on an unemployment rate.
    The old code compared the cutoff to the *standardised* state, so 6.5
    put every observation in one regime and crashed with a singular X'X
    (M32). Now the raw-scale cutoff recovers the regime-specific slopes."""
    df = _regime_dgp()
    res = lp_state_dep(df, y="y", x="x", state="u_rate", horizons=[0], n_lags=1,
                       transition="threshold", threshold=6.5)
    assert abs(float(res["beta_H"].iloc[0]) - 1.0) < 0.15
    assert abs(float(res["beta_L"].iloc[0]) - 0.2) < 0.15
    # lp_state_dep_iv shares the convention (raw scale, None = sample mean).
    riv = lp_state_dep_iv(df, y="y", x="x", z="z", state="u_rate", horizon=0,
                          lags=1, transition="threshold", threshold=6.5)
    assert abs(float(riv["beta_H"].iloc[0]) - 1.0) < 0.2
    assert abs(float(riv["beta_L"].iloc[0]) - 0.2) < 0.2
    riv_default = lp_state_dep_iv(df, y="y", x="x", z="z", state="u_rate", horizon=1, lags=1)
    riv_mean = lp_state_dep_iv(df, y="y", x="x", z="z", state="u_rate", horizon=1, lags=1,
                               threshold=float(df["u_rate"].mean()))
    pd.testing.assert_frame_equal(pd.DataFrame(riv_default), pd.DataFrame(riv_mean))
    assert list(riv.index) == [0] and riv.method == "LP-state-dep-IV"


def test_lp_state_dep_degenerate_threshold_raises_clear_error():
    """A cutoff outside the state's range used to surface as
    ``LinAlgError: X'X is singular`` (threshold mode) or a near-singular
    fit (logistic, e.g. threshold=0.0 on a ~6 % unemployment rate). Both
    now raise a ValueError naming the state's range."""
    df = _regime_dgp()
    with pytest.raises(ValueError, match="raw scale"):
        lp_state_dep(df, y="y", x="x", state="u_rate", horizons=[0],
                     transition="threshold", threshold=-3.0)
    with pytest.raises(ValueError, match="raw scale"):
        lp_state_dep(df, y="y", x="x", state="u_rate", horizons=[0],
                     transition="logistic", gamma=3.0, threshold=0.0)
    with pytest.raises(ValueError, match="raw scale"):
        lp_state_dep_iv(df, y="y", x="x", z="z", state="u_rate", horizons=[0],
                        transition="threshold", threshold=20.0)
    with pytest.raises(ValueError, match="transition"):
        lp_state_dep(df, y="y", x="x", state="u_rate", horizons=[0], transition="banana")


def test_lp_state_dep_validates_ci_and_alpha():
    """ci=90 (a percentage) and alpha=1.5 were accepted silently and gave
    NaN bands; they must raise."""
    df = _regime_dgp()
    with pytest.raises(ValueError, match="ci"):
        lp_state_dep(df, y="y", x="x", state="u_rate", horizon=2, ci=90)
    with pytest.raises(ValueError, match="alpha"):
        lp_state_dep(df, y="y", x="x", state="u_rate", horizons=[0], alpha=1.5)
    with pytest.raises(ValueError, match="ci"):
        lp_state_dep_iv(df, y="y", x="x", z="z", state="u_rate", horizon=2, ci=95)
