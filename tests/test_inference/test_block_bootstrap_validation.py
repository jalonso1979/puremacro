"""Regression tests for puremacro.inference.block_bootstrap input validation
(audit garch-vol-inference: 'block_bootstrap block_length > T').

Before the fix ``block_length > T`` fell through to ``rng.integers(0, T - ell
+ 1)`` and surfaced as numpy's bare ``ValueError: high <= 0``; the docstring
also claimed the residuals are 're-centered inside' when they are used as
given.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.inference.block_bootstrap import block_bootstrap


def _mean(e):
    return np.array([e.mean()])


def test_block_length_larger_than_T_raises_named_error():
    residuals = np.random.default_rng(0).standard_normal(20)
    with pytest.raises(ValueError, match=r"block_length=25 must satisfy 1 <= block_length <= T=20"):
        block_bootstrap(residuals, refit_fn=_mean, B=5, block_length=25,
                        rng=np.random.default_rng(1))


@pytest.mark.parametrize("bad", [0, -3])
def test_non_positive_block_length_raises(bad):
    residuals = np.random.default_rng(0).standard_normal(20)
    with pytest.raises(ValueError, match="block_length"):
        block_bootstrap(residuals, refit_fn=_mean, B=5, block_length=bad,
                        rng=np.random.default_rng(1))


def test_block_length_equal_to_T_is_allowed():
    residuals = np.random.default_rng(0).standard_normal(20)
    draws = block_bootstrap(residuals, refit_fn=_mean, B=4, block_length=20,
                            rng=np.random.default_rng(1))
    assert draws.shape == (4, 1)
    np.testing.assert_allclose(draws[:, 0], residuals.mean())


def test_empty_and_multidimensional_residuals_raise():
    with pytest.raises(ValueError, match="empty"):
        block_bootstrap(np.array([]), refit_fn=_mean, B=2)
    with pytest.raises(ValueError, match="1-D"):
        block_bootstrap(np.ones((5, 2)), refit_fn=_mean, B=2)


def test_residuals_are_used_as_given_not_recentred():
    """The docstring now says so; a constant array bootstraps to itself."""
    residuals = np.full(30, 2.5)
    draws = block_bootstrap(residuals, refit_fn=_mean, B=3, block_length=5,
                            rng=np.random.default_rng(2))
    np.testing.assert_allclose(draws, 2.5)
