import numpy as np
import pytest

from puremacro.inference.hac_fixed_b import hac_fixed_b, llsw_critical_value


@pytest.mark.pyodide_smoke
def test_llsw_cv_table_lookup():
    # Tabulated value at b=0.10, alpha=0.10
    assert abs(llsw_critical_value(0.10, alpha=0.10) - 1.86) < 1e-6
    # Linearly interpolated at b=0.15 between 0.10 and 0.20
    cv_15 = llsw_critical_value(0.15, alpha=0.10)
    assert 1.86 < cv_15 < 1.93


def test_hac_fixed_b_returns_documented_keys():
    rng = np.random.default_rng(11)
    T = 200
    X = np.column_stack([np.ones(T), rng.standard_normal((T, 2))])
    beta = np.array([0.5, 1.0, -0.5])
    y = X @ beta + rng.standard_normal(T)
    out = hac_fixed_b(y, X, b=0.10)
    for k in ("beta", "se", "t", "vcov", "residuals", "n_obs", "b", "lags", "llsw_cv_90"):
        assert k in out
    assert out["lags"] == 20  # ⌊0.10 * 200⌋
    assert abs(out["llsw_cv_90"] - 1.86) < 1e-6


def test_hac_fixed_b_bandwidth_lower_bound():
    # Very short series: bandwidth should still be ≥ 1.
    rng = np.random.default_rng(13)
    T = 5
    X = np.column_stack([np.ones(T), rng.standard_normal(T)])
    y = rng.standard_normal(T)
    out = hac_fixed_b(y, X, b=0.05)
    assert out["lags"] >= 1
