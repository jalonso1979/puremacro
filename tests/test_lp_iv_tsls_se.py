"""LP-IV standard errors must come from 2SLS residuals built with the *actual* regressor.

Regression for ``lp_iv``, ``la_lp_iv`` and ``lp_state_dep_iv`` in 4.0.1: each ran a plain
OLS variance on the second-stage regression, so the residual in the sandwich was
``y - X_hat beta = u + beta (x - x_hat)`` rather than the structural error ``u``. The
variance was off by ``2 beta cov(u, v) + beta^2 var(v)``: 90% bands covered the truth
100% of the time when ``beta`` and ``cov(u, v)`` share a sign, and too rarely otherwise.
"""
import numpy as np
import pandas as pd
import pytest

from puremacro.inference._ols_helpers import tsls_hac
from puremacro.lp import lp_iv
from puremacro.lp.la_lp import la_lp_iv
from puremacro.lp.state_dep import lp_state_dep_iv


def _textbook_2sls_hac(y, X, Z, lags):
    """Matrix-form 2SLS with a Bartlett long-run variance, written with the T x T projection."""
    T = len(y)
    P = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    A = X.T @ P @ X
    beta = np.linalg.solve(A, X.T @ P @ y)
    u = y - X @ beta
    t = np.arange(T)
    K = np.clip(1.0 - np.abs(t[:, None] - t[None, :]) / (lags + 1.0), 0.0, None)
    PX = P @ X
    S = PX.T @ (np.outer(u, u) * K) @ PX
    A_inv = np.linalg.inv(A)
    V = A_inv @ S @ A_inv
    return beta, np.sqrt(np.diag(V))


@pytest.mark.parametrize("lags", [0, 1, 3])
def test_tsls_hac_matches_matrix_form(lags):
    rng = np.random.default_rng(3)
    T = 160
    z1, z2, v, e = rng.normal(size=(4, T))
    u = 0.7 * v + 0.5 * e
    x = 0.6 * z1 - 0.4 * z2 + v
    y = 2.0 - 1.3 * x + u
    X = np.column_stack([np.ones(T), x])
    Z = np.column_stack([np.ones(T), z1, z2])
    X_hat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]

    out = tsls_hac(y, X, X_hat, lags=lags)
    beta, se = _textbook_2sls_hac(y, X, Z, lags)
    np.testing.assert_allclose(out["beta"], beta, rtol=1e-10)
    np.testing.assert_allclose(out["se"], se, rtol=1e-10)


@pytest.mark.parametrize("h", [0, 2])
def test_lp_iv_se_equals_2sls_hac_on_its_design(h):
    rng = np.random.default_rng(11)
    T = 240
    z, v, e = rng.normal(size=(3, T))
    x = 0.8 * z + v
    y = np.cumsum(-0.9 * x + 0.8 * v + 0.6 * e)
    df = pd.DataFrame({"y": y, "x": x, "z": z})

    res = lp_iv(df, y="y", x="x", z="z", horizons=[h], n_lags=1)

    d = df.copy()
    d["dy"] = d["y"].shift(-h) - d["y"].shift(1)
    d["x_L1"], d["y_L1"] = d["x"].shift(1), d["y"].shift(1)
    d = d.dropna()
    n = len(d)
    X = np.column_stack([np.ones(n), d["x"], d["x_L1"], d["y_L1"]])
    Z = np.column_stack([np.ones(n), d["z"], d["x_L1"], d["y_L1"]])
    beta, se = _textbook_2sls_hac(d["dy"].to_numpy(), X, Z, lags=h + 1)

    assert float(res["beta"].iloc[0]) == pytest.approx(beta[1], rel=1e-9)
    assert float(res["se"].iloc[0]) == pytest.approx(se[1], rel=1e-9)


def _simulate(T, seed, beta, cov_uv):
    rng = np.random.default_rng(seed)
    z, v, e, s = rng.normal(size=(4, T))
    u = cov_uv * v + np.sqrt(1.0 - cov_uv**2) * e
    x = 0.7 * z + v
    y = np.cumsum(beta * x + u)  # y_t - y_{t-1} = beta x_t + u_t
    return pd.DataFrame({"y": y, "x": x, "z": z, "s": s})


# beta * cov(u, v) > 0 made the old bands too wide; < 0 made them too narrow.
@pytest.mark.parametrize("beta", [1.0, -1.0])
def test_lp_iv_bands_have_nominal_coverage(beta):
    R, hits = 300, {"lp_iv": 0, "la_lp_iv": 0, "state_H": 0, "state_L": 0}
    for seed in range(R):
        df = _simulate(300, seed, beta, cov_uv=0.8)
        for name, fn in (("lp_iv", lp_iv), ("la_lp_iv", la_lp_iv)):
            r = fn(df, y="y", x="x", z="z", horizons=[0], n_lags=1)
            hits[name] += float(r["lo"].iloc[0]) <= beta <= float(r["hi"].iloc[0])
        r = lp_state_dep_iv(df, y="y", x="x", z="z", state="s", horizons=[0], n_lags=1)
        hits["state_H"] += float(r["lo_H"].iloc[0]) <= beta <= float(r["hi_H"].iloc[0])
        hits["state_L"] += float(r["lo_L"].iloc[0]) <= beta <= float(r["hi_L"].iloc[0])
    # 300 draws: the Monte Carlo standard error of a 90% coverage rate is about 0.017
    for name, k in hits.items():
        assert 0.84 <= k / R <= 0.96, f"{name}: coverage {k / R:.3f} of a nominal 90% band"
