"""Regime interaction in the state-dependent LPs (``interact=``).

Ramey-Zubairy (2018, eq. 2) let every coefficient vary with the state. These
tests pin: (i) with a threshold, ``interact='all'`` equals one regression per
regime; (ii) ``interact='none'`` is the pooled regression (the behaviour before
the option existed); (iii) pooling the intercept biases the regime slopes when
outcome levels and shock means differ by regime; (iv) the new difference test
and effective sample sizes; (v) failures are reported, not swallowed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.lp import lp_iv, lp_state_dep, lp_state_dep_iv


def _regime_dgp(T: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Same true effect (1.0) in both regimes; the regime shifts the outcome
    level (+3) and the mean of the shock / instrument (+1)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = s[t - 1] if rng.random() < 0.95 else 1 - s[t - 1]
    z = 1.0 * s + rng.standard_normal(T)
    v = rng.standard_normal(T)
    x = z + v
    u = 0.5 * v + rng.standard_normal(T)          # x endogenous, z a valid instrument
    xo = 1.0 * s + rng.standard_normal(T)         # exogenous shock for the OLS version
    return pd.DataFrame({
        "Y": np.cumsum(3.0 * s + 1.0 * x + u),    # long difference at h=0 = level
        "x": x, "z": z, "reg": s.astype(float),
        "Yo": np.cumsum(3.0 * s + 1.0 * xo + rng.standard_normal(T)),
        "xo": xo,
    })


@pytest.mark.parametrize("mode", ["intercept", "all"])
def test_regime_intercept_removes_pooling_bias(mode):
    df = _regime_dgp()
    kw = dict(state="reg", horizons=[0], n_lags=0, transition="threshold", threshold=0.5)
    iv_pooled = lp_state_dep_iv(df, y="Y", x="x", z="z", interact="none", **kw).iloc[0]
    ols_pooled = lp_state_dep(df, y="Yo", x="xo", interact="none", **kw).iloc[0]
    # Pooling loads the regime level gap onto the high-regime slope (~2 instead of 1).
    assert abs(iv_pooled["beta_H"] - 1.0) > 0.5
    assert abs(ols_pooled["beta_H"] - 1.0) > 0.5
    iv = lp_state_dep_iv(df, y="Y", x="x", z="z", interact=mode, **kw).iloc[0]
    ols = lp_state_dep(df, y="Yo", x="xo", interact=mode, **kw).iloc[0]
    for r in (iv, ols):
        assert abs(r["beta_H"] - 1.0) < 0.1
        assert abs(r["beta_L"] - 1.0) < 0.1


def test_default_is_all():
    df = _regime_dgp(T=800, seed=3)
    kw = dict(y="Y", x="x", z="z", state="reg", horizons=[0, 2], n_lags=1,
              transition="threshold", threshold=0.5)
    pd.testing.assert_frame_equal(pd.DataFrame(lp_state_dep_iv(df, **kw)),
                                  pd.DataFrame(lp_state_dep_iv(df, interact="all", **kw)))


def _ols_by_hand(df: pd.DataFrame, h: int, n_lags: int, rows: np.ndarray | None) -> np.ndarray:
    """OLS of y_{t+h}-y_{t-1} on [x, lags of x and y, 1], on all rows or a subset."""
    d = pd.DataFrame({"dy": df["Yo"].shift(-h) - df["Yo"].shift(1), "x": df["xo"], "reg": df["reg"]})
    cols = ["x"]
    for lag in range(1, n_lags + 1):
        d[f"xL{lag}"] = df["xo"].shift(lag); d[f"yL{lag}"] = df["Yo"].shift(lag)
        cols += [f"xL{lag}", f"yL{lag}"]
    d = d.dropna()
    if rows is not None:
        d = d[d["reg"] == rows]
    X = np.column_stack([d[cols].to_numpy(), np.ones(len(d))])
    return np.linalg.lstsq(X, d["dy"].to_numpy(), rcond=None)[0]


def test_ols_all_threshold_equals_one_regression_per_regime():
    df = _regime_dgp(T=1500, seed=11)
    res = lp_state_dep(df, y="Yo", x="xo", state="reg", horizons=[0, 3], n_lags=2,
                       transition="threshold", threshold=0.5, interact="all")
    for h in (0, 3):
        assert res.loc[h, "beta_H"] == pytest.approx(_ols_by_hand(df, h, 2, 1.0)[0], rel=1e-8, abs=1e-10)
        assert res.loc[h, "beta_L"] == pytest.approx(_ols_by_hand(df, h, 2, 0.0)[0], rel=1e-8, abs=1e-10)


def test_ols_none_is_the_pooled_regression():
    df = _regime_dgp(T=1500, seed=11)
    h, n_lags = 3, 2
    res = lp_state_dep(df, y="Yo", x="xo", state="reg", horizons=[h], n_lags=n_lags,
                       transition="threshold", threshold=0.5, interact="none")
    d = pd.DataFrame({"dy": df["Yo"].shift(-h) - df["Yo"].shift(1), "x": df["xo"], "F": df["reg"]})
    cols = []
    for lag in range(1, n_lags + 1):
        d[f"xL{lag}"] = df["xo"].shift(lag); d[f"yL{lag}"] = df["Yo"].shift(lag)
        cols += [f"xL{lag}", f"yL{lag}"]
    d = d.dropna()
    X = np.column_stack([d["F"] * d["x"], (1 - d["F"]) * d["x"], np.ones(len(d)), d[cols].to_numpy()])
    b = np.linalg.lstsq(X, d["dy"].to_numpy(), rcond=None)[0]
    assert res.loc[h, "beta_H"] == pytest.approx(b[0], rel=1e-8, abs=1e-10)
    assert res.loc[h, "beta_L"] == pytest.approx(b[1], rel=1e-8, abs=1e-10)


def test_iv_all_threshold_equals_one_lp_iv_per_regime_on_rz_data():
    """On Ramey-Zubairy's own snapshot: the one-step integral multiplier by
    regime equals lp_iv run on a frame with the instrument blanked outside the
    regime (the masking idiom keeps the windows contiguous)."""
    from puremacro.examples.ramey_zubairy_2018_multipliers import _CTL, load_rz_data

    df = load_rz_data()
    h = 8
    d = df.copy()
    d["cumg_h"] = d["cumg"].shift(-h) - d["cumg"].shift(1)
    d = d.dropna(subset=["slack_l"])
    res = lp_state_dep_iv(d, y="cumy", x="cumg_h", z="newsy", state="slack_l", horizons=[h],
                          n_lags=0, controls=_CTL, transition="threshold", threshold=0.5).iloc[0]
    for col, reg in (("beta_H", 1.0), ("beta_L", 0.0)):
        dr = d.copy()
        dr.loc[dr["slack_l"] != reg, "newsy"] = np.nan
        ref = lp_iv(dr, y="cumy", x="cumg_h", z="newsy", horizons=[h], n_lags=0, controls=_CTL)
        assert res[col] == pytest.approx(float(ref["beta"].iloc[0]), rel=1e-8, abs=1e-10)
    # RZ (2018, Table 1, news shock, 2 years): 0.60 (high unemployment) and 0.59 (low).
    assert 0.5 < res["beta_H"] < 0.7 and 0.5 < res["beta_L"] < 0.7
    assert res["p_diff_HL"] > 0.5


def test_difference_test_and_effective_sizes():
    df = _regime_dgp(T=1200, seed=5)
    for fn, kw in ((lp_state_dep, dict(y="Yo", x="xo")), (lp_state_dep_iv, dict(y="Y", x="x", z="z"))):
        r = fn(df, state="reg", horizons=[0], n_lags=1, transition="threshold", threshold=0.5, **kw).iloc[0]
        assert r["diff_HL"] == pytest.approx(r["beta_H"] - r["beta_L"])
        assert r["se_diff_HL"] > 0 and 0.0 <= r["p_diff_HL"] <= 1.0
        n = r["n_eff_H"] + r["n_eff_L"]
        assert r["n_eff_H"] == pytest.approx(float(df["reg"].iloc[1:].sum()), abs=2)
        assert n == pytest.approx(len(df) - 1, abs=2)


@pytest.mark.parametrize("mode", ["all", "intercept", "none"])
def test_logistic_transition_runs_in_every_mode(mode):
    rng = np.random.default_rng(0)
    df = _regime_dgp(T=600, seed=2)
    df["u"] = 5.0 + 2.0 * df["reg"] + 0.5 * rng.standard_normal(len(df))
    r = lp_state_dep_iv(df, y="Y", x="x", z="z", state="u", horizons=[0, 1], n_lags=1,
                        transition="logistic", gamma=3.0, interact=mode)
    assert np.isfinite(r[["beta_H", "beta_L", "se_H", "se_L", "diff_HL"]].to_numpy()).all()
    o = lp_state_dep(df, y="Yo", x="xo", state="u", horizons=[0, 1], n_lags=1,
                     transition="logistic", interact=mode)
    assert np.isfinite(o[["beta_H", "beta_L", "se_H", "se_L"]].to_numpy()).all()


def test_unknown_interact_raises():
    df = _regime_dgp(T=200, seed=1)
    with pytest.raises(ValueError, match="interact"):
        lp_state_dep(df, y="Yo", x="xo", state="reg", horizons=[0], interact="both")
    with pytest.raises(ValueError, match="interact"):
        lp_state_dep_iv(df, y="Y", x="x", z="z", state="reg", horizons=[0], interact="lags")


def test_iv_failure_is_reported_not_swallowed(monkeypatch):
    import puremacro.lp.state_dep as sd

    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(sd, "tsls_hac", boom)
    df = _regime_dgp(T=300, seed=4)
    with pytest.warns(RuntimeWarning, match=r"h=0 could not be estimated .*singular matrix"):
        r = lp_state_dep_iv(df, y="Y", x="x", z="z", state="reg", horizons=[0], n_lags=1,
                            transition="threshold", threshold=0.5)
    assert np.isnan(r["beta_H"].iloc[0]) and np.isnan(r["diff_HL"].iloc[0])
