"""Regression tests for puremacro.inference.over_id.hansen_j (audit major).

Before the fix the function computed the homoskedastic Sargan statistic
``T * R^2`` of the 2SLS residual on the instruments while its name and
docstring promised the Hansen J of 2SLS / IV-GMM. The two coincide only
under conditional homoskedasticity; under heteroskedasticity the Sargan
statistic is not chi^2(l - k). ``robust=True`` (default) is now the two-step
GMM J with the HC0 weight matrix, and ``robust=False`` keeps the Sargan
statistic under its own name.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from puremacro.inference.over_id import hansen_j


def _two_step_gmm_j(y, X, Z, W):
    """Hayashi (2000, ch. 3) two-step GMM J with the HC0 weight matrix."""
    n = len(y)
    Zf = np.column_stack([W, Z])
    Xf = np.column_stack([X, W])
    Xh = Zf @ np.linalg.lstsq(Zf, Xf, rcond=None)[0]
    b1 = np.linalg.solve(Xh.T @ Xf, Xh.T @ y)
    u1 = y - Xf @ b1
    S = (Zf.T * u1 ** 2) @ Zf / n
    Si = np.linalg.inv(S)
    ZX, Zy = Zf.T @ Xf, Zf.T @ y
    b2 = np.linalg.solve(ZX.T @ Si @ ZX, ZX.T @ Si @ Zy)
    gb = Zf.T @ (y - Xf @ b2) / n
    return float(n * gb @ Si @ gb)


def _sargan(y, X, Z, W):
    n = len(y)
    Zf = np.column_stack([W, Z])
    Xf = np.column_stack([X, W])
    Xh = Zf @ np.linalg.lstsq(Zf, Xf, rcond=None)[0]
    b = np.linalg.solve(Xh.T @ Xf, Xh.T @ y)
    u = y - Xf @ b
    proj = Zf @ np.linalg.lstsq(Zf, u, rcond=None)[0]
    r2 = 1 - np.sum((u - proj) ** 2) / np.sum((u - u.mean()) ** 2)
    return float(n * r2)


def _hetero_design(n=400, seed=77):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 3))
    v = rng.standard_normal(n)
    x = Z @ [0.5, 0.4, 0.3] + v
    e = rng.standard_normal(n) * (0.5 + np.abs(Z[:, 0]))   # heteroskedastic
    y = 1.0 * x + 0.5 * v + e
    return y, x, Z


def test_robust_j_equals_two_step_gmm_with_hc0_weights():
    """Reference values on this design: two-step GMM J = 2.349711 (also
    linearmodels IVGMM(weight_type='robust').fit(iter_limit=2).j_stat);
    the old code returned the Sargan 3.013365 under the Hansen name."""
    y, x, Z = _hetero_design()
    W = np.ones((len(y), 1))
    res = hansen_j(y, x, Z)
    assert res["stat"] == pytest.approx(_two_step_gmm_j(y, x[:, None], Z, W), rel=1e-10)
    assert res["stat"] == pytest.approx(2.349711, abs=1e-5)
    assert res["df"] == 2
    assert res["stat"] != pytest.approx(_sargan(y, x[:, None], Z, W), rel=1e-3)


def test_robust_false_is_the_sargan_statistic():
    y, x, Z = _hetero_design()
    W = np.ones((len(y), 1))
    res = hansen_j(y, x, Z, robust=False)
    assert res["stat"] == pytest.approx(_sargan(y, x[:, None], Z, W), rel=1e-10)
    assert res["stat"] == pytest.approx(3.013365, abs=1e-5)


def test_robust_j_with_controls_matches_manual():
    rng = np.random.default_rng(3)
    n = 500
    Z = rng.standard_normal((n, 3))
    w = rng.standard_normal(n)
    x = Z @ [0.6, 0.3, 0.2] + 0.4 * w + rng.standard_normal(n)
    y = 2.0 * x + 0.7 * w + rng.standard_normal(n) * (1 + w ** 2) ** 0.5
    res = hansen_j(y, x, Z, W=w)
    W = np.column_stack([np.ones(n), w])
    assert res["stat"] == pytest.approx(_two_step_gmm_j(y, x[:, None], Z, W), rel=1e-10)
    assert res["df"] == 2


def test_just_identified_j_is_zero_and_p_is_nan():
    rng = np.random.default_rng(99)
    n = 300
    z = rng.standard_normal(n)
    x = 0.7 * z + 0.3 * rng.standard_normal(n)
    y = 2.0 * x + rng.standard_normal(n)
    res = hansen_j(y, x, z)
    assert abs(res["stat"]) < 1e-10
    assert res["df"] == 0
    assert math.isnan(res["p_value"])


def test_two_endogenous_regressors():
    rng = np.random.default_rng(11)
    n = 600
    Z = rng.standard_normal((n, 4))
    X = np.column_stack([Z[:, 0] + 0.5 * rng.standard_normal(n),
                         Z[:, 1] + 0.5 * rng.standard_normal(n)])
    y = 2 * X[:, 0] + 1.5 * X[:, 1] + rng.standard_normal(n)
    res = hansen_j(y, X, Z)
    W = np.ones((n, 1))
    assert res["stat"] == pytest.approx(_two_step_gmm_j(y, X, Z, W), rel=1e-10)
    assert res["df"] == 2
    assert 0.0 <= res["p_value"] <= 1.0


def test_robust_j_has_chi2_size_under_heteroskedasticity():
    """Under H0 with heteroskedastic errors the robust J rejects at ~5%;
    the Sargan statistic over-rejects on this design."""
    rej_robust = rej_sargan = 0
    R = 300
    for seed in range(R):
        rng = np.random.default_rng(seed)
        n = 400
        Z = rng.standard_normal((n, 3))
        v = rng.standard_normal(n)
        x = Z @ [0.5, 0.4, 0.3] + v
        e = rng.standard_normal(n) * (0.3 + 1.5 * Z[:, 0] ** 2)
        y = x + 0.5 * v + e
        rej_robust += hansen_j(y, x, Z)["p_value"] < 0.05
        rej_sargan += hansen_j(y, x, Z, robust=False)["p_value"] < 0.05
    assert 0.02 <= rej_robust / R <= 0.10, rej_robust / R
    assert rej_sargan / R > rej_robust / R


def test_invalid_instrument_is_rejected():
    rng = np.random.default_rng(1234)
    n = 5000
    z1 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    z2 = 0.9 * u + 0.1 * rng.standard_normal(n)
    x = 0.7 * z1 + 0.5 * z2 + 0.3 * rng.standard_normal(n)
    y = 2.0 * x + u
    assert hansen_j(y, x, np.column_stack([z1, z2]))["p_value"] < 0.01


def test_length_mismatch_raises():
    y, x, Z = _hetero_design()
    with pytest.raises(ValueError, match="rows"):
        hansen_j(y[:-1], x, Z)
