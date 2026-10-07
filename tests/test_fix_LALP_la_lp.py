"""Regression tests for the lag-augmented LP fix (review key LALP, 2026-09-30).

Montiel Olea, J. L. and M. Plagborg-Møller (2021), "Local Projection
Inference Is Simpler and More Robust Than You Think", Econometrica 89(4),
1789-1823 (arXiv:2007.13888v4, read for this fix):

* §2.1, eq. (3), p. 8: the AR(1) lag-augmented LP regresses y_{t+h} on y_t
  and "uses y_{t-1} as an additional control variable";
* §4.1, p. 20: in a VAR(p) the LA-LP controls for p lags of *all* variables,
  "Thus, we are including one additional lag" - the same at every horizon;
* eq. (5): Eicker-Huber-White standard errors, no HAR correction;
* Table 1, p. 12: AR(1) with intercept, rho = 0.95, T = 240, 90% intervals,
  LP-LA coverage .878/.838/.806/.814/.833 at h = 1/6/12/36/60
  (5,000 Monte Carlo repetitions).

Before the fix ``la_lp`` defaulted to ``extra_lags = max(horizons)`` (so the
estimate at a given h depended on the largest horizon requested, and
``horizons=[0]`` had no augmentation at all), gave user controls only
``n_lags`` lags, and credited the method to Plagborg-Møller & Wolf (2021),
the LP = VAR paper, with a "p + h" lag rule that no source states.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import importlib

import puremacro.lp as lp_pkg
from puremacro.lp.la_lp import la_lp, la_lp_iv

# ``import puremacro.lp.la_lp as m`` would bind the re-exported *function*.
la_mod = importlib.import_module("puremacro.lp.la_lp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _var_data(T: int = 260, seed: int = 0) -> pd.DataFrame:
    """Bivariate recursive DGP plus one persistent control."""
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    y = np.zeros(T)
    c = np.zeros(T)
    for t in range(1, T):
        c[t] = 0.8 * c[t - 1] + rng.standard_normal()
        x[t] = 0.5 * x[t - 1] + 0.2 * c[t] + rng.standard_normal()
        y[t] = 0.9 * y[t - 1] + 0.5 * x[t] + 0.1 * c[t - 1] + rng.standard_normal()
    return pd.DataFrame({"y": y, "x": x, "c": c})


def _ref_la_lp(df, y, x, h, n_x_y_lags, controls=(), control_lags=0):
    """Independent numpy LA-LP: OLS of y_{t+h} on
    [1, x_{t-l}, y_{t-l} (l=1..L), c_{t-l} (l=1..q), c_t, x_t], HC0 SE."""
    Y = df[y].to_numpy(float)
    X = df[x].to_numpy(float)
    T = len(Y)
    start = max([n_x_y_lags] + ([control_lags] if controls else []))
    rows = range(start, T - h)
    cols = [np.ones(len(rows))]
    for lag in range(1, n_x_y_lags + 1):
        cols.append(np.array([X[t - lag] for t in rows]))
        if y != x:
            cols.append(np.array([Y[t - lag] for t in rows]))
    for cname in controls:
        C = df[cname].to_numpy(float)
        for lag in range(1, control_lags + 1):
            cols.append(np.array([C[t - lag] for t in rows]))
        cols.append(np.array([C[t] for t in rows]))
    cols.append(np.array([X[t] for t in rows]))
    M = np.column_stack(cols)
    dep = np.array([Y[t + h] for t in rows])
    XtXi = np.linalg.inv(M.T @ M)
    b = XtXi @ M.T @ dep
    e = dep - M @ b
    S = (M * e[:, None]).T @ (M * e[:, None])
    V = XtXi @ S @ XtXi
    return float(b[-1]), float(np.sqrt(V[-1, -1])), len(rows)


# ---------------------------------------------------------------------------
# Default = MOPM's single extra lag, fixed across horizons
# ---------------------------------------------------------------------------


def test_default_adds_exactly_one_lag_at_every_horizon():
    df = _var_data()
    res = la_lp(df, y="y", x="x", horizons=range(0, 21), n_lags=4)
    assert (res["p_aug"] == 5).all()
    # extra_lags is recorded on the result
    assert res.extra_lags == 1
    assert res.n_lags == 4
    assert res.attrs["extra_lags"] == 1
    assert res.attrs["n_lags"] == 4
    assert res.attrs["p_aug"] == 5
    # attrs survive ordinary pandas operations
    assert res.set_index("h", drop=False).attrs["extra_lags"] == 1


def test_estimate_at_h_does_not_depend_on_max_horizon():
    df = _var_data()
    short = la_lp(df, y="y", x="x", horizons=range(0, 5), n_lags=4).set_index("h")
    long_ = la_lp(df, y="y", x="x", horizons=range(0, 21), n_lags=4).set_index("h")
    only = la_lp(df, y="y", x="x", horizons=[4], n_lags=4).set_index("h")
    for other in (long_, only):
        assert other.loc[4, "beta"] == pytest.approx(short.loc[4, "beta"], abs=1e-12)
        assert other.loc[4, "se"] == pytest.approx(short.loc[4, "se"], abs=1e-12)


def test_horizon_zero_alone_is_still_lag_augmented():
    df = _var_data()
    res = la_lp(df, y="y", x="x", horizons=[0], n_lags=2)
    assert int(res["p_aug"].iloc[0]) == 3


def test_matches_independent_hc0_regression_with_controls():
    """Default call = OLS on n_lags + 1 lags of x, y AND of every control
    (MOPM §4: 'controls for p lags of all the time series that enter into
    the VAR model'), EHW (HC0) SE per MOPM eq. (5)."""
    df = _var_data()
    raw = la_lp(df, y="y", x="x", horizons=[0, 3, 8], n_lags=2, controls=["c"])
    assert raw.control_lags == 3
    res = raw.set_index("h")
    assert res.attrs["control_lags"] == 3
    for h in (0, 3, 8):
        b, s, _ = _ref_la_lp(df, "y", "x", h, 3, controls=["c"], control_lags=3)
        assert res.loc[h, "beta"] == pytest.approx(b, rel=1e-9, abs=1e-12)
        assert res.loc[h, "se"] == pytest.approx(s, rel=1e-9, abs=1e-12)
        z = norm.ppf(0.95)
        assert res.loc[h, "lo"] == pytest.approx(b - z * s, rel=1e-9)
        assert res.loc[h, "hi"] == pytest.approx(b + z * s, rel=1e-9)


def test_legacy_specification_is_an_explicit_opt_in():
    """extra_lags=max(H) with control_lags=n_lags reproduces the <=4.3.0
    default exactly (outcome y_{t+h}-y_{t-1} and level y_{t+h} give the same
    coefficient on x_t because y_{t-1} is a regressor)."""
    df = _var_data()
    H = 8
    raw = la_lp(df, y="y", x="x", horizons=range(0, H + 1), n_lags=2,
                extra_lags=H, controls=["c"], control_lags=2)
    assert raw.extra_lags == H and raw.control_lags == 2
    res = raw.set_index("h")
    assert (res["p_aug"] == 2 + H).all()
    assert res.attrs["extra_lags"] == H
    for h in (0, 4, 8):
        b, s, _ = _ref_la_lp(df, "y", "x", h, 2 + H, controls=["c"], control_lags=2)
        assert res.loc[h, "beta"] == pytest.approx(b, rel=1e-9, abs=1e-12)
        assert res.loc[h, "se"] == pytest.approx(s, rel=1e-9, abs=1e-12)


def test_explicit_none_means_the_mopm_default():
    df = _var_data()
    a = la_lp(df, y="y", x="x", horizons=[2], n_lags=3, extra_lags=None)
    b = la_lp(df, y="y", x="x", horizons=[2], n_lags=3)
    assert int(a["p_aug"].iloc[0]) == 4
    assert float(a["beta"].iloc[0]) == float(b["beta"].iloc[0])


def test_la_lp_iv_default_is_one_extra_lag():
    rng = np.random.default_rng(3)
    T = 300
    z = rng.standard_normal(T)
    x = 0.8 * z + rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 0.4 * x[t] + rng.standard_normal()
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    res = la_lp_iv(df, y="y", x="x", z="z", horizons=range(0, 9), n_lags=2)
    assert (res["p_aug"] == 3).all()
    assert res.extra_lags == 1


@pytest.mark.parametrize("bad", [-1, 1.5, "2"])
def test_invalid_extra_lags_raise(bad):
    df = _var_data()
    with pytest.raises(ValueError, match="extra_lags"):
        la_lp(df, y="y", x="x", horizons=[0], n_lags=2, extra_lags=bad)


def test_no_augmentation_warns():
    df = _var_data()
    with pytest.warns(UserWarning, match="lag augmentation"):
        la_lp(df, y="y", x="x", horizons=[0, 1], n_lags=2, extra_lags=0)


def test_default_call_emits_no_warning():
    df = _var_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        la_lp(df, y="y", x="x", horizons=[0, 1], n_lags=2)


# ---------------------------------------------------------------------------
# Univariate AR case (y == x): MOPM eq. (3)
# ---------------------------------------------------------------------------


def test_own_response_is_mopm_ar1_regression():
    rng = np.random.default_rng(11)
    T = 240
    u = rng.standard_normal(T)
    yv = np.zeros(T)
    prev = 0.0
    for t in range(T):
        prev = 0.95 * prev + u[t]
        yv[t] = prev
    df = pd.DataFrame({"y": yv})
    res = la_lp(df, y="y", x="y", horizons=[1, 6, 12], n_lags=0).set_index("h")
    assert (res["p_aug"] == 1).all()
    for h in (1, 6, 12):
        # regress y_{t+h} on (1, y_{t-1}, y_t), t = 1..T-1-h (0-based)
        b, s, n = _ref_la_lp(df, "y", "y", h, 1)
        assert n == T - 1 - h
        assert res.loc[h, "beta"] == pytest.approx(b, rel=1e-9)
        assert res.loc[h, "se"] == pytest.approx(s, rel=1e-9)


# ---------------------------------------------------------------------------
# Attribution and the "p + h" rule
# ---------------------------------------------------------------------------


def test_docstrings_credit_montiel_olea_plagborg_moller():
    for doc in (la_lp.__doc__, la_lp_iv.__doc__):
        assert "Montiel Olea" in doc and "Plagborg-Møller" in doc
        assert "p + h" not in doc and "p+h" not in doc
        assert "PMW" not in doc
        assert "Wolf" not in doc
    mod_doc = la_mod.__doc__
    assert mod_doc.startswith(
        "Lag-augmented local projections (Montiel Olea & Plagborg-Møller, 2021)")
    assert "Econometrica* 89(4), 1789-1823" in mod_doc
    assert "PMW" not in mod_doc
    assert "Plagborg-Møller-Wolf" not in mod_doc
    assert "p_aug = p + h" not in mod_doc
    assert "Montiel Olea & Plagborg-Møller 2021" in lp_pkg.__doc__
    assert "PMW" not in lp_pkg.__doc__


# ---------------------------------------------------------------------------
# Monte Carlo: MOPM (2021) Table 1, rho = 0.95, T = 240, LP-LA column
# ---------------------------------------------------------------------------

# Table 1, p. 12 of arXiv:2007.13888v4 (read 2026-09-30).
_MOPM_T1_RHO095_LPLA = {1: 0.878, 6: 0.838, 12: 0.806, 36: 0.814, 60: 0.833}
_MOPM_REPS = 5000


def _ar1(T, rho, rng):
    u = rng.standard_normal(T)
    y = np.empty(T)
    prev = 0.0  # y_0 = 0, MOPM eq. (1)
    for t in range(T):
        prev = rho * prev + u[t]
        y[t] = prev
    return y


def test_mopm_table1_rho095_coverage_within_mc_error():
    R = 5000
    T, rho = 240, 0.95
    H = sorted(_MOPM_T1_RHO095_LPLA)
    rng = np.random.default_rng(20070225)
    hits = {h: 0 for h in H}
    for _ in range(R):
        df = pd.DataFrame({"y": _ar1(T, rho, rng)})
        res = la_lp(df, y="y", x="y", horizons=H, n_lags=0, ci=0.90)
        for h, lo, hi in zip(res["h"], res["lo"], res["hi"]):
            hits[int(h)] += lo <= rho ** int(h) <= hi
    for h in H:
        cov = hits[h] / R
        p = _MOPM_T1_RHO095_LPLA[h]
        mc_se = np.sqrt(p * (1 - p) * (1 / R + 1 / _MOPM_REPS))
        assert abs(cov - p) < 3.5 * mc_se, (h, cov, p, mc_se)
