import numpy as np
import pytest

from puremacro.inference.weak_iv import anderson_rubin_test, msw_bands, ARTestResult


def _strong_iv_design(rng, T=300, beta=-0.5):
    z = rng.standard_normal(T)
    x = 2.0 * z + 0.5 * rng.standard_normal(T)  # KP-F ~~ 144
    eta = rng.standard_normal(T)
    y = beta * x + 0.5 * eta + 0.3 * rng.standard_normal(T)
    return y, x, z


def _weak_iv_design(rng, T=300, beta=-0.5, pi=0.1):
    """Pi=0.1 means tiny first-stage coefficient → KP-F well below 10."""
    z = rng.standard_normal(T)
    x = pi * z + rng.standard_normal(T)
    eta = rng.standard_normal(T)
    y = beta * x + 0.5 * eta + rng.standard_normal(T)
    return y, x, z


def test_anderson_rubin_accepts_true_beta():
    rng = np.random.default_rng(11)
    y, x, z = _strong_iv_design(rng, T=300, beta=-0.5)
    out = anderson_rubin_test(beta0=-0.5, y=y, x_endog=x, z=z)
    assert isinstance(out, ARTestResult)
    assert out.p_value > 0.10  # accept H_0 at the true β
    assert hasattr(out, "stat") and hasattr(out, "p_value")


def test_anderson_rubin_rejects_distant_beta():
    rng = np.random.default_rng(13)
    y, x, z = _strong_iv_design(rng, T=300, beta=-0.5)
    out = anderson_rubin_test(beta0=2.0, y=y, x_endog=x, z=z)
    assert isinstance(out, ARTestResult)
    assert out.p_value < 0.01  # firmly reject far-from-truth


def test_anderson_rubin_result_is_frozen():
    rng = np.random.default_rng(15)
    y, x, z = _strong_iv_design(rng, T=200, beta=-0.5)
    out = anderson_rubin_test(beta0=-0.5, y=y, x_endog=x, z=z)
    assert isinstance(out, ARTestResult)
    with pytest.raises(Exception):
        out.stat = 0.0


def test_anderson_rubin_summary_smoke():
    rng = np.random.default_rng(17)
    y, x, z = _strong_iv_design(rng, T=200, beta=-0.5)
    out = anderson_rubin_test(beta0=-0.5, y=y, x_endog=x, z=z)
    s = out.summary()
    assert "Anderson-Rubin" in s


def test_msw_bands_strong_iv_match_delta_method():
    """Under strong identification, MSW bands ≈ standard delta-method IV bands."""
    rng = np.random.default_rng(17)
    y, x, z = _strong_iv_design(rng, T=300, beta=-0.5)
    # Standard 2SLS β
    pi_z = (z @ x) / (z @ z)
    x_hat = pi_z * z
    beta_2sls = (x_hat @ y) / (x_hat @ x_hat)
    # MSW bands
    grid = np.linspace(-1.5, 0.5, 401)
    lo, hi = msw_bands(grid=grid, y=y, x_endog=x, z=z, alpha=0.10)
    assert lo <= beta_2sls <= hi
    # Width should be similar to delta-method (i.e., reasonably tight under strong IV)
    assert hi - lo < 0.5


def test_msw_bands_weak_iv_are_wide():
    """Under weak identification, MSW should give wide bands."""
    rng = np.random.default_rng(19)
    y, x, z = _weak_iv_design(rng, T=300, beta=-0.5, pi=0.1)
    grid = np.linspace(-5, 5, 1001)
    lo, hi = msw_bands(grid=grid, y=y, x_endog=x, z=z, alpha=0.10)
    # Bands should be substantially wider than under strong IV
    assert hi - lo > 1.0
