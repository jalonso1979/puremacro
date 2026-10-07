import numpy as np
import pytest

from puremacro.inference.hac_fixed_b import hac_fixed_b, llsw_critical_value

# Kiefer & Vogelsang (2005, Econometric Theory 21(6)) Table I, Bartlett kernel,
# 95% row (two-sided 10%), read from CAE WP 05-08, printed p. 24 (PDF p. 26).
_KV95 = {0.10: 1.861, 0.12: 1.902, 0.14: 1.944, 0.16: 1.988}
# Same table, 97.5% row (two-sided 5%).
_KV975 = {0.10: 2.235, 0.12: 2.296}


@pytest.mark.pyodide_smoke
def test_llsw_cv_table_lookup():
    # Tabulated value at b=0.10, two-sided alpha=0.10: the 95% quantile of KV Table I.
    # (Until the FIXB fix this was 1.86 from a table that matched no published source.)
    assert abs(llsw_critical_value(0.10, alpha=0.10) - _KV95[0.10]) < 1e-9
    # Linearly interpolated at b=0.15 between the grid points b=0.14 and b=0.16.
    cv_15 = llsw_critical_value(0.15, alpha=0.10)
    assert abs(cv_15 - 0.5 * (_KV95[0.14] + _KV95[0.16])) < 1e-9


def test_hac_fixed_b_returns_documented_keys():
    rng = np.random.default_rng(11)
    T = 200
    X = np.column_stack([np.ones(T), rng.standard_normal((T, 2))])
    beta = np.array([0.5, 1.0, -0.5])
    y = X @ beta + rng.standard_normal(T)
    out = hac_fixed_b(y, X, b=0.10)
    for k in (
        "beta", "se", "t", "vcov", "residuals", "n_obs", "b", "lags", "llsw_cv_90",
        "bandwidth", "b_eff", "fixed_b_cv_90", "fixed_b_cv_95", "fixed_b_cv_99",
    ):
        assert k in out
    assert out["lags"] == 20  # ⌊0.10 * 200⌋
    # Newey-West weights 1 - l/(L+1) are the Bartlett kernel with M = L + 1, and
    # KV (2005, section 3.3) look the critical value up at b = M/T.
    assert out["bandwidth"] == 21
    assert out["b_eff"] == 21 / 200
    cv90 = _KV95[0.10] + 0.25 * (_KV95[0.12] - _KV95[0.10])  # b = 0.105: 1.87125
    cv95 = _KV975[0.10] + 0.25 * (_KV975[0.12] - _KV975[0.10])  # 2.25025
    assert abs(out["llsw_cv_90"] - cv90) < 1e-12
    assert abs(out["llsw_cv_90"] - llsw_critical_value(21 / 200, alpha=0.10)) < 1e-12
    assert out["fixed_b_cv_90"] == out["llsw_cv_90"]
    assert abs(out["fixed_b_cv_95"] - cv95) < 1e-12
    # Two-sided 1% needs the 99.5% quantile, which Table I lacks; puremacro
    # computes it (see the module docstring).
    assert abs(out["fixed_b_cv_99"] - llsw_critical_value(21 / 200, alpha=0.01)) < 1e-12


def test_hac_fixed_b_bandwidth_lower_bound():
    # Very short series: bandwidth should still be ≥ 1.
    rng = np.random.default_rng(13)
    T = 5
    X = np.column_stack([np.ones(T), rng.standard_normal(T)])
    y = rng.standard_normal(T)
    out = hac_fixed_b(y, X, b=0.05)
    assert out["lags"] >= 1
