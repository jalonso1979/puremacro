"""Size and centring tests for :func:`puremacro.inference.swamy.swamy_test`.

The only test this function had was ``test_imports.py``'s ``callable(...)``,
which is why it shipped for several releases centred on the arithmetic mean of
the per-unit coefficients instead of the precision-weighted one. The two agree
exactly when every unit is estimated with the same precision — which is what a
smoke fixture naturally builds — and diverge without limit as the precisions
spread apart. So the test that had to exist is a size test.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from puremacro.inference.swamy import swamy_test


def _draw(sds, reps, seed=0, beta_true=(1.0, -0.5)):
    """Replications drawn strictly under the null: every unit shares beta."""
    rng = np.random.default_rng(seed)
    K = len(beta_true)
    bt = np.asarray(beta_true, dtype=float)
    sigma = np.stack([s ** 2 * np.eye(K) for s in sds])
    for _ in range(reps):
        beta = np.stack([rng.multivariate_normal(bt, S) for S in sigma])
        yield beta, sigma


def _size(sds, reps=4000, seed=0):
    N, K = len(sds), 2
    crit = stats.chi2.ppf(0.95, K * (N - 1))
    hits = sum(swamy_test(b, s)[0] > crit for b, s in _draw(sds, reps, seed))
    return hits / reps


@pytest.mark.parametrize("label, sds", [
    ("equal precision", [1.0] * 10),
    ("2x spread", [1.0] * 5 + [2.0] * 5),
    ("30x spread", [0.1] * 5 + [3.0] * 5),
])
def test_size_is_nominal_however_unequal_the_per_unit_precisions_are(label, sds):
    """Under H0 the test must reject 5% of the time, at every dispersion.

    Centred on the plain mean the measured sizes were 0.050 / 0.078 / 0.975 —
    the last one rejecting slope homogeneity on a panel where homogeneity is
    literally true. The distortion is monotone in the spread of per-unit
    precision and never goes the other way, because the quadratic form is
    being evaluated away from its own minimum.
    """
    assert _size(sds) == pytest.approx(0.05, abs=0.015), label


def test_the_statistic_is_centred_on_the_precision_weighted_mean():
    """S is the *minimised* quadratic form; that is where its df come from."""
    rng = np.random.default_rng(1)
    N, K = 6, 2
    sds = [0.2, 0.3, 1.0, 2.0, 3.0, 5.0]
    sigma = np.stack([s ** 2 * np.eye(K) for s in sds])
    beta = np.stack([rng.multivariate_normal([1.0, -0.5], S) for S in sigma])
    S, _, df = swamy_test(beta, sigma)
    assert df == K * (N - 1)

    def q(b):
        return sum(float((bi - b) @ np.linalg.solve(Si, (bi - b)))
                   for bi, Si in zip(beta, sigma))

    prec = [np.linalg.inv(S_) for S_ in sigma]
    beta_w = np.linalg.inv(sum(prec)) @ sum(P @ b for P, b in zip(prec, beta))
    assert S == pytest.approx(q(beta_w), rel=1e-10)
    # and it really is the minimum: the arithmetic mean scores strictly worse
    assert q(beta.mean(axis=0)) > S
    for _ in range(20):
        assert q(beta_w + 0.05 * rng.standard_normal(K)) > S


def test_perfectly_homogeneous_slopes_give_a_zero_statistic():
    beta = np.tile(np.array([1.0, -0.5]), (5, 1))
    sigma = np.stack([(0.5 * (i + 1)) ** 2 * np.eye(2) for i in range(5)])
    S, p, df = swamy_test(beta, sigma)
    assert S == pytest.approx(0.0, abs=1e-20)
    assert p == pytest.approx(1.0)
    assert df == 2 * 4


def test_a_non_positive_definite_unit_covariance_is_diagnosed():
    """CONTRIBUTING: named LinAlgError, never a pseudo-inverse in the sum."""
    beta = np.array([[1.0, -0.5], [1.1, -0.4]])
    sigma = np.stack([np.eye(2), np.zeros((2, 2))])
    with pytest.raises(np.linalg.LinAlgError, match="Swamy Sigma_hat"):
        swamy_test(beta, sigma)
