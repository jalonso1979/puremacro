"""Tests for puremacro.inference.weak_iv."""
import numpy as np
import pytest

from puremacro.inference.weak_iv import olea_pflueger_f


def test_olea_pflueger_f_strong_instrument():
    """Olea-Pflueger F on a strong instrument should clear the F=23 cutoff."""
    rng = np.random.default_rng(0)
    T = 500
    Z = rng.standard_normal((T, 1))
    eps = rng.standard_normal(T) * 0.5
    X = 2.0 * Z[:, 0] + eps                # strong first stage, beta=2
    f = olea_pflueger_f(X, Z)
    assert f > 23.0, f"strong instrument should clear F=23, got {f:.2f}"


def test_olea_pflueger_f_weak_instrument():
    """Weak instrument should produce a small F."""
    rng = np.random.default_rng(1)
    T = 500
    Z = rng.standard_normal((T, 1))
    X = 0.05 * Z[:, 0] + rng.standard_normal(T)
    f = olea_pflueger_f(X, Z)
    assert f < 5.0, f"weak instrument expected F < 5, got {f:.2f}"


def test_olea_pflueger_f_matches_homoskedastic_first_stage_f():
    """Under homoskedasticity with one instrument, OP-F approximately equals
    the standard first-stage F."""
    rng = np.random.default_rng(2)
    T = 1000
    Z = rng.standard_normal((T, 1))
    X = 1.5 * Z[:, 0] + rng.standard_normal(T)
    f_op = olea_pflueger_f(X, Z)
    # Standard first-stage F: (R^2 / (1 - R^2)) * (T - K)
    # Demean both sides (matches OP-F convention; intercept implied).
    Xd = X - X.mean()
    Zd = Z - Z.mean(axis=0, keepdims=True)
    Pi = np.linalg.solve(Zd.T @ Zd, Zd.T @ Xd)
    fitted = Zd @ Pi
    R2 = 1 - np.sum((Xd - fitted) ** 2) / np.sum(Xd ** 2)
    K = Z.shape[1]
    f_classical = (R2 / (1 - R2)) * (T - K)
    # rtol=0.15 because OP-F uses HC0 (heteroskedasticity-robust) variance
    # while the classical F uses a homoskedastic sigma^2 estimate; they
    # differ by sampling noise of order O(1/sqrt(T)) even under homoskedasticity.
    np.testing.assert_allclose(f_op, f_classical, rtol=0.15)
