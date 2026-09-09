"""Empirical adversarial stress test suite for puremacro 2.9.0 Tier 3 Milestone 1.

Authored by m1_challenger_2 to empirically challenge:
1. simul_replic = 1000 Monte Carlo 3-sigma confidence bounds across multiple DGP configurations.
2. Autocorrelation matrices R(k) diagonal matching and [-1, 1] bounds up to high lags.
3. Contemporaneous correlation positive semi-definiteness, exact symmetry, and 1.0 diagonal under collinearity.
4. Comprehensive rejection of invalid arguments with informative ValueErrors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    build,
    one_sided_hp_filter,
    StochSimulResult,
    TheoreticalMomentsResult,
)


# ============================================================================
# 1. simul_replic = 1000 Empirical Simulated Moments across Multiple DGPs
# ============================================================================

def test_simul_replic_1000_mc_confidence_bounds_multi_dgp():
    """Verify that sample means and variances fall strictly within 3-sigma MC confidence bounds.

    Tested across 5 diverse DGP configurations:
    - DGP 1: AR(1) low persistence (rho=0.2)
    - DGP 2: AR(1) negative persistence (rho=-0.4)
    - DGP 3: AR(1) non-zero steady-state level (a_ss=3.5, rho=0.7)
    - DGP 4: Bivariate VAR(1) with bidirectional cross-variable feedback
    - DGP 5: Quasi-white noise (rho=0.001)
    """
    # DGP 1: AR(1) low persistence (rho=0.2)
    def eqs_ar1(xp, x, e, p):
        return [xp.a - p.rho * x.a - e.eps]

    m1 = build(
        eqs_ar1,
        variables=["a"],
        states=["a"],
        shocks=["eps"],
        params={"rho": 0.2},
        steady_state={"a": 0.0},
    )
    res1 = m1.stoch_simul(periods=1000, simul_replic=1000, seed=101)
    tm1 = res1.theoretical_moments.moments.loc["a"]
    sm1 = res1.simulated_moments.loc["a"]
    se_var1 = res1.simulated_moments.attrs["mc_se_var"]["a"]
    assert abs(sm1["Mean"] - tm1["Mean"]) <= 3.0 * sm1["MC Std.Err."]
    assert abs(sm1["Variance"] - tm1["Variance"]) <= 3.0 * se_var1

    # DGP 2: AR(1) negative persistence (rho=-0.4)
    m2 = build(
        eqs_ar1,
        variables=["a"],
        states=["a"],
        shocks=["eps"],
        params={"rho": -0.4},
        steady_state={"a": 0.0},
    )
    res2 = m2.stoch_simul(periods=1000, simul_replic=1000, seed=202)
    tm2 = res2.theoretical_moments.moments.loc["a"]
    sm2 = res2.simulated_moments.loc["a"]
    se_var2 = res2.simulated_moments.attrs["mc_se_var"]["a"]
    assert abs(sm2["Mean"] - tm2["Mean"]) <= 3.0 * sm2["MC Std.Err."]
    assert abs(sm2["Variance"] - tm2["Variance"]) <= 3.0 * se_var2

    # DGP 3: AR(1) with non-zero steady state (a_ss=3.5, rho=0.7)
    def eqs_ss(xp, x, e, p):
        return [xp.a - (1.0 - p.rho) * p.a_ss - p.rho * x.a - e.eps]

    m3 = build(
        eqs_ss,
        variables=["a"],
        states=["a"],
        shocks=["eps"],
        params={"rho": 0.7, "a_ss": 3.5},
        steady_state={"a": 3.5},
    )
    res3 = m3.stoch_simul(periods=1000, simul_replic=1000, seed=303)
    tm3 = res3.theoretical_moments.moments.loc["a"]
    sm3 = res3.simulated_moments.loc["a"]
    se_var3 = res3.simulated_moments.attrs["mc_se_var"]["a"]
    assert tm3["Mean"] == pytest.approx(3.5)
    assert abs(sm3["Mean"] - tm3["Mean"]) <= 3.0 * sm3["MC Std.Err."]
    assert abs(sm3["Variance"] - tm3["Variance"]) <= 3.0 * se_var3

    # DGP 4: Bivariate VAR(1) with cross-feedback
    def eqs_var(xp, x, e, p):
        return [
            xp.x1 - p.a11 * x.x1 - p.a12 * x.x2 - e.e1,
            xp.x2 - p.a21 * x.x1 - p.a22 * x.x2 - e.e2,
        ]

    m4 = build(
        eqs_var,
        variables=["x1", "x2"],
        states=["x1", "x2"],
        shocks=["e1", "e2"],
        params={"a11": 0.4, "a12": 0.2, "a21": 0.1, "a22": 0.5},
        steady_state={"x1": 0.0, "x2": 0.0},
    )
    res4 = m4.stoch_simul(periods=1000, simul_replic=1000, seed=404)
    tm4 = res4.theoretical_moments.moments
    sm4 = res4.simulated_moments
    se_var4 = res4.simulated_moments.attrs["mc_se_var"]
    for v in ["x1", "x2"]:
        assert abs(sm4.loc[v, "Mean"] - tm4.loc[v, "Mean"]) <= 3.0 * sm4.loc[v, "MC Std.Err."]
        assert abs(sm4.loc[v, "Variance"] - tm4.loc[v, "Variance"]) <= 3.0 * se_var4[v]

    # DGP 5: Quasi-white noise (rho=0.001)
    m5 = build(
        eqs_ar1,
        variables=["a"],
        states=["a"],
        shocks=["eps"],
        params={"rho": 0.001},
        steady_state={"a": 0.0},
    )
    res5 = m5.stoch_simul(periods=1000, simul_replic=1000, seed=505)
    tm5 = res5.theoretical_moments.moments.loc["a"]
    sm5 = res5.simulated_moments.loc["a"]
    se_var5 = res5.simulated_moments.attrs["mc_se_var"]["a"]
    assert abs(sm5["Mean"] - tm5["Mean"]) <= 3.0 * sm5["MC Std.Err."]
    assert abs(sm5["Variance"] - tm5["Variance"]) <= 3.0 * se_var5


# ============================================================================
# 2. Autocorrelation Matrices R(k): Diagonals and Bounds
# ============================================================================

def test_autocorr_matrices_diagonals_and_bounds_high_lags():
    """Verify that R(k) diagonals match univariate autocorr coefficients and stay bounded in [-1, 1]."""
    def eqs_rbc(xp, x, e, p):
        return [
            x.c**-p.sigma - p.beta * xp.c**-p.sigma * (p.alpha * xp.z * xp.k**(p.alpha - 1) + 1 - p.delta),
            x.c + xp.k - x.z * x.k**p.alpha - (1 - p.delta) * x.k,
            xp.z - (1.0 - p.rho) - p.rho * x.z - e.eps,
        ]

    beta, delta, alpha = 0.99, 0.025, 0.33
    r_ss = 1.0 / beta - 1.0
    k_ss = (alpha / (r_ss + delta)) ** (1.0 / (1.0 - alpha))
    y_ss = k_ss**alpha
    c_ss = y_ss - delta * k_ss

    m_rbc = build(
        eqs_rbc,
        variables=["c", "k", "z"],
        states=["k", "z"],
        shocks=["eps"],
        params=dict(alpha=alpha, beta=beta, delta=delta, sigma=1.0, rho=0.95),
        steady_state=dict(c=c_ss, k=k_ss, z=1.0),
    )

    # Test up to lag 10
    res = m_rbc.theoretical_moments(ar=10)
    assert len(res.autocorr_matrices) == 10
    assert len(res.autocorrelation_matrices) == 10

    for k in range(1, 11):
        R_k = res.autocorr_matrix(k).to_numpy()
        diag = np.diag(R_k)
        expected_diag = res.autocorr[f"Lag {k}"].to_numpy()
        np.testing.assert_allclose(diag, expected_diag, atol=1e-12)
        assert np.all(R_k >= -1.0)
        assert np.all(R_k <= 1.0)

    # Lag 1 cross-autocorrelation is not symmetric in general: R(1) != R(1).T
    R_1 = res.autocorr_matrix(1).to_numpy()
    assert not np.allclose(R_1, R_1.T, atol=1e-3)


def test_autocorr_matrices_oscillatory_system():
    """Verify R(k) properties in a system with complex oscillatory roots."""
    def eqs_rot(xp, x, e, p):
        return [
            xp.x1 - (0.3 * x.x1 - 0.4 * x.x2) - e.e1,
            xp.x2 - (0.4 * x.x1 + 0.3 * x.x2) - e.e2,
        ]

    m_rot = build(
        eqs_rot,
        variables=["x1", "x2"],
        states=["x1", "x2"],
        shocks=["e1", "e2"],
        params={},
        steady_state={"x1": 0.0, "x2": 0.0},
    )

    res = m_rot.theoretical_moments(ar=6)
    for k in range(1, 7):
        R_k = res.autocorr_matrix(k).to_numpy()
        np.testing.assert_allclose(np.diag(R_k), res.autocorr[f"Lag {k}"].to_numpy(), atol=1e-12)
        assert np.all(R_k >= -1.0)
        assert np.all(R_k <= 1.0)


# ============================================================================
# 3. Contemporaneous Correlation: Positive Semi-Definiteness under Collinearity
# ============================================================================

@pytest.mark.parametrize("delta", [1e-2, 1e-4, 1e-6, 1e-8, 1e-12, 0.0])
def test_contemporaneous_correlation_collinear_psd(delta):
    """Verify PSD, exact symmetry, and 1.0 diagonal when variables are near- or perfectly collinear."""
    def eqs_collinear(xp, x, e, p):
        return [
            xp.x1 - 0.7 * x.x1 - e.e1,
            x.x2 - x.x1 - p.delta * e.e2,
        ]

    m = build(
        eqs_collinear,
        variables=["x1", "x2"],
        states=["x1"],
        shocks=["e1", "e2"],
        params={"delta": delta},
        steady_state={"x1": 0.0, "x2": 0.0},
    )

    res = m.theoretical_moments(contemporaneous_correlation=True)
    corr = res.correlation.to_numpy()

    # 1. Exact symmetry
    np.testing.assert_allclose(corr, corr.T, atol=1e-14)
    # 2. Diagonal is strictly 1.0
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-14)
    # 3. Positive semi-definite (min eigenvalue >= -1e-12)
    eigvals = np.linalg.eigvalsh(corr)
    assert eigvals.min() >= -1e-12, f"Min eigenvalue {eigvals.min()} < -1e-12"
    # 4. Values strictly within [-1, 1]
    assert np.all(corr >= -1.0) and np.all(corr <= 1.0)


def test_contemporaneous_correlation_5var_singular():
    """Verify PSD and symmetry on a 5-variable rank-deficient singular system."""
    def eqs_multi_collinear(xp, x, e, p):
        return [
            xp.x1 - 0.6 * x.x1 - e.e1,
            xp.x2 - 0.4 * x.x2 - e.e2,
            x.x3 - (0.5 * x.x1 + 0.5 * x.x2 + 1e-8 * e.e3),
            x.x4 - (-x.x1 + 1e-10 * e.e4),
            x.x5 - (2.0 * x.x3 - x.x2),
        ]

    m = build(
        eqs_multi_collinear,
        variables=["x1", "x2", "x3", "x4", "x5"],
        states=["x1", "x2"],
        shocks=["e1", "e2", "e3", "e4"],
        params={},
        steady_state={"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0, "x5": 0.0},
    )

    # Theoretical correlation matrix
    res_theo = m.theoretical_moments(contemporaneous_correlation=True, ar=4)
    corr_theo = res_theo.correlation.to_numpy()
    np.testing.assert_allclose(corr_theo, corr_theo.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(corr_theo), 1.0, atol=1e-14)
    eigvals = np.linalg.eigvalsh(corr_theo)
    assert eigvals.min() >= -1e-12
    assert (corr_theo >= -1.0).all() and (corr_theo <= 1.0).all()

    # Simulated correlation matrix under simul_replic
    res_sim = m.stoch_simul(periods=200, simul_replic=50, seed=42)
    corr_sim = res_sim.contemporaneous_correlation.to_numpy()
    np.testing.assert_allclose(corr_sim, corr_sim.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(corr_sim), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(corr_sim).min() >= -1e-12
    assert (corr_sim >= -1.0).all() and (corr_sim <= 1.0).all()


# ============================================================================
# 4. Graceful Rejection of Invalid Arguments
# ============================================================================

def test_invalid_arguments_graceful_rejection():
    """Verify that all invalid argument configurations are rejected with informative ValueErrors."""
    def eqs(xp, x, e, p):
        return [xp.a - p.rho * x.a - e.eps]

    m = build(
        eqs,
        variables=["a"],
        states=["a"],
        shocks=["eps"],
        params={"rho": 0.5},
        steady_state={"a": 0.0},
    )

    # 1. simul_replic < 0
    with pytest.raises(ValueError, match="simul_replic must be non-negative"):
        m.stoch_simul(periods=100, simul_replic=-1)
    with pytest.raises(ValueError, match="simul_replic must be non-negative"):
        m.stoch_simul(periods=100, simul_replic=-10)

    # 2. simul_replic > 0 with periods = 0
    with pytest.raises(ValueError, match="simul_replic > 0 requires periods > 0"):
        m.stoch_simul(periods=0, simul_replic=100)

    # 3. ar < 0 in both stoch_simul and theoretical_moments
    with pytest.raises(ValueError, match="ar must be a non-negative integer"):
        m.stoch_simul(periods=100, ar=-1)
    with pytest.raises(ValueError, match="ar must be a non-negative integer"):
        m.theoretical_moments(ar=-4)

    # 4. hp_filter <= 0
    with pytest.raises(ValueError, match="hp_filter parameter lambda must be positive"):
        m.stoch_simul(periods=100, hp_filter=-1600.0)
    with pytest.raises(ValueError, match="hp_filter parameter lambda must be positive"):
        m.stoch_simul(periods=100, hp_filter=0.0)
    with pytest.raises(ValueError, match="hp_filter parameter lambda must be positive"):
        m.theoretical_moments(hp_filter=-100.0)
    with pytest.raises(ValueError, match="hp_filter parameter lambda must be positive"):
        m.theoretical_moments(hp_filter=0.0)

    # 5. bandpass_filter invalid configurations
    bad_bps = [(32, 6), (-6, 32), (0, 32), (6, 6), (6,), (6, 12, 32)]
    for bp in bad_bps:
        with pytest.raises(ValueError, match="bandpass_filter requires 0 < low < high"):
            m.stoch_simul(periods=100, bandpass_filter=bp)
        with pytest.raises(ValueError, match="bandpass_filter requires 0 < low < high"):
            m.theoretical_moments(bandpass_filter=bp)

    # 6. Multiple filters simultaneously
    with pytest.raises(ValueError, match="Only one filter can be specified"):
        m.stoch_simul(periods=100, hp_filter=1600.0, bandpass_filter=(6, 32))
    with pytest.raises(ValueError, match="Only one filter can be specified"):
        m.stoch_simul(periods=100, hp_filter=1600.0, one_sided_hp_filter=True)
    with pytest.raises(ValueError, match="Only one filter can be specified"):
        m.stoch_simul(periods=100, bandpass_filter=(6, 32), one_sided_hp_filter=True)

    # 7. one_sided_hp_filter with theoretical moments (periods=0 or in theoretical_moments)
    dynare_guard = "disp_th_moments:: theoretical moments incompatible with one-sided HP filter. Use simulated moments instead."
    with pytest.raises(ValueError, match=dynare_guard):
        m.theoretical_moments(one_sided_hp_filter=True)
    with pytest.raises(ValueError, match=dynare_guard):
        m.stoch_simul(periods=0, one_sided_hp_filter=True)

    # 8. one_sided_hp_filter function guards
    with pytest.raises(ValueError, match="one_sided_hp_filter requires at least 4 observations"):
        one_sided_hp_filter(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="one_sided_hp_filter parameter lambda must be positive"):
        one_sided_hp_filter(np.array([1.0, 2.0, 3.0, 4.0]), lamb=-10.0)
    with pytest.raises(ValueError, match="one_sided_hp_filter parameter lambda must be positive"):
        one_sided_hp_filter(np.array([1.0, 2.0, 3.0, 4.0]), lamb=0.0)
