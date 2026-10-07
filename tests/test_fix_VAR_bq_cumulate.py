"""Regression tests for ``bq_svar(cumulate=...)`` (review key BQ).

``bq_svar`` cumulated every response variable, which is right when every
column of ``Y`` is differenced (Gali 1999's difference specification) and
wrong for a variable in levels, such as the unemployment rate of the
canonical Blanchard-Quah (1989) system X = (dlog GNP, U)'. Its response then
converged to ((I - A(1))^-1 B)[u, :] instead of returning to 0, and its bands
could not be repaired afterwards. ``cumulate=`` selects the rows to cumulate,
for the point estimate and for every bootstrap draw.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.var.estimate import estimate_var
from puremacro.var.identify.bq import _bq_impact, bq_svar
from puremacro.var.irf import irf as compute_irf

# Known VAR(1) in X = (dy, u): u stationary in levels, dy stationary growth.
_A = np.array([[0.3, -0.2], [0.1, 0.8]])
# Long-run matrix C(1) = (I - A)^-1 B chosen lower triangular: shock 1
# (demand) has no long-run effect on the level of y.
_C1 = np.array([[1.0, 0.0], [0.5, 0.7]])
_B = np.linalg.solve(np.linalg.inv(np.eye(2) - _A), _C1)


def _simulate(T=20000, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((T + 200, 2)) @ _B.T
    X = np.zeros((T + 200, 2))
    for t in range(1, T + 200):
        X[t] = _A @ X[t - 1] + e[t]
    return X[200:]


@pytest.fixture(scope="module")
def Y():
    return _simulate()


def test_population_impact_is_recovered():
    B_hat = _bq_impact([_A], _B @ _B.T, 0)
    np.testing.assert_allclose(B_hat, _B, atol=1e-12)


def test_default_still_cumulates_every_variable(Y):
    """Backward compatibility: the default equals the old cumsum-all output."""
    res = bq_svar(Y, p=1, horizon=40, n_boot=0)
    A_list, _, Sigma, _, _ = estimate_var(Y, 1)
    raw = compute_irf(A_list, _bq_impact(A_list, Sigma, 0), 40)
    np.testing.assert_array_equal(res.irf_point, np.cumsum(raw, axis=0))
    res_true = bq_svar(Y, p=1, horizon=40, n_boot=0, cumulate=True)
    np.testing.assert_array_equal(res.irf_point, res_true.irf_point)


def test_levels_variable_returns_to_zero_with_cumulate_first_only(Y):
    """Canonical BQ spec: cumulate only dy. The u response must be the raw
    response (-> 0), the y row the partial sum (-> C(1) row 0)."""
    H = 40
    res = bq_svar(Y, p=1, horizon=H, n_boot=0, cumulate=[0])
    # population responses
    raw_true = np.stack([np.linalg.matrix_power(_A, h) @ _B for h in range(H + 1)])
    y_level_true = np.cumsum(raw_true[:, 0, :], axis=0)
    np.testing.assert_allclose(res.irf_point[:, 0, :], y_level_true, atol=0.05)
    np.testing.assert_allclose(res.irf_point[:, 1, :], raw_true[:, 1, :], atol=0.05)
    assert np.all(np.abs(res.irf_point[H, 1, :]) < 1e-3)          # u -> 0
    np.testing.assert_allclose(res.irf_point[H, 0, :], _C1[0], atol=0.05)
    # the old behaviour: u response converges to C(1)[1, :], not 0
    old = bq_svar(Y, p=1, horizon=H, n_boot=0)
    np.testing.assert_allclose(old.irf_point[H, 1, :], _C1[1], atol=0.05)
    assert np.abs(old.irf_point[H, 1, :]).max() > 0.4


@pytest.mark.parametrize("spec", [[0], (0,), np.array([0]), [True, False],
                                  np.array([True, False])])
def test_index_list_and_mask_are_equivalent(Y, spec):
    ref = bq_svar(Y[:2000], p=1, horizon=12, n_boot=0, cumulate=[0])
    res = bq_svar(Y[:2000], p=1, horizon=12, n_boot=0, cumulate=spec)
    np.testing.assert_array_equal(res.irf_point, ref.irf_point)


@pytest.mark.parametrize("spec", [False, [], np.array([], dtype=int)])
def test_no_cumulation_returns_raw_irfs(Y, spec):
    Ys = Y[:2000]
    res = bq_svar(Ys, p=1, horizon=12, n_boot=0, cumulate=spec)
    A_list, _, Sigma, _, _ = estimate_var(Ys, 1)
    raw = compute_irf(A_list, _bq_impact(A_list, Sigma, 0), 12)
    np.testing.assert_array_equal(res.irf_point, raw)


def test_bands_use_the_same_transformation_as_the_point(Y):
    """With one seed the bootstrap draws are identical across calls, so the
    cumulate=[0] bands must equal the cumulate-all bands on row 0 and the raw
    bands on row 1 — quantiles of the reported object, not diffs of bands."""
    Ys = Y[:600]
    kw = dict(p=1, horizon=16, n_boot=60, ci=0.9, seed=3)
    mixed = bq_svar(Ys, cumulate=[0], **kw)
    allc = bq_svar(Ys, cumulate=True, **kw)
    none = bq_svar(Ys, cumulate=False, **kw)
    for band in ("irf_lower", "irf_upper"):
        np.testing.assert_array_equal(getattr(mixed, band)[:, 0], getattr(allc, band)[:, 0])
        np.testing.assert_array_equal(getattr(mixed, band)[:, 1], getattr(none, band)[:, 1])
    assert np.all(mixed.irf_lower <= mixed.irf_upper)
    # At h = 16 the band of the levels variable has collapsed towards 0
    # (0.8**16 ~ 0.03); the cumulated band sits near C(1)[1, :] = (0.5, 0.7).
    assert np.all(np.abs(mixed.irf_lower[-1, 1]) < 0.02)
    assert np.all(np.abs(mixed.irf_upper[-1, 1]) < 0.02)
    assert np.all(allc.irf_lower[-1, 1] > 0.3)
    # np.diff of cumulated bands is not the raw band
    diffed = np.diff(allc.irf_upper[:, 1], axis=0, prepend=0.0)
    assert not np.allclose(diffed, none.irf_upper[:, 1])


@pytest.mark.parametrize("bad,exc", [([2], ValueError), ([-1], ValueError),
                                     ([True], ValueError), ([0.0], TypeError),
                                     ("y", TypeError), (1, TypeError),
                                     ([[0]], ValueError)])
def test_invalid_cumulate_raises(Y, bad, exc):
    with pytest.raises(exc):
        bq_svar(Y[:500], p=1, horizon=4, n_boot=0, cumulate=bad)
