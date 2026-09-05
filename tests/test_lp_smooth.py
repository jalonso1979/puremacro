"""Comprehensive Unit, Interface, and Empirical Verification Tests for Smooth LP.

Verifies Barnichon & Brownlees (2019) Smooth Local Projections:
1. Public API standardization: `smooth_lp` and `lp_smooth` in puremacro.lp.
2. Interface contracts: LPResult subclass of DataFrame, `.summary()`, `.plot()`,
   `.to_markdown()`, `.to_latex()`, `.to_typst()`.
3. True joint penalized estimation with B-spline basis and roughness penalty matrix P.
4. Data-driven lambda selection via AIC, BIC, GCV, and Cross-Validation (CV).
5. Analytical sandwich HAC covariance and moving block bootstrap inference.
6. Empirical verification on synthetic AR(2) DGP:
   - Strictly lower empirical IRF variance and MSE than standard unpenalized LP.
   - Unbiasedness as lambda -> 0.
   - Linear shrinkage as lambda -> inf.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.lp import smooth_lp, lp_smooth, LPResult
from puremacro.lp.smooth import (
    _build_bspline_basis,
    _difference_penalty_matrix,
    _prepare_lp_data,
)


def _simulate_ar2(
    T: int = 250,
    phi1: float = 0.6,
    phi2: float = -0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate an AR(2) time series: y_t = phi1*y_{t-1} + phi2*y_{t-2} + eps_t."""
    rng = np.random.default_rng(seed)
    burn = 100
    total = T + burn
    eps = rng.standard_normal(total)
    y = np.zeros(total, dtype=float)
    for t in range(2, total):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]

    y_sample = y[burn:]
    eps_sample = eps[burn:]
    dates = pd.date_range("1995-01-01", periods=T, freq="MS")
    df = pd.DataFrame({"y": y_sample, "x": eps_sample}, index=dates)

    # Analytical MA(inf) true IRF
    H_max = 25
    psi = np.zeros(H_max + 1, dtype=float)
    psi[0] = 1.0
    psi[1] = phi1
    for h in range(2, H_max + 1):
        psi[h] = phi1 * psi[h - 1] + phi2 * psi[h - 2]

    return df, psi


# =====================================================================
# 1. Public API Standardization & Export Verification
# =====================================================================
def test_public_api_and_exports():
    """Verify smooth_lp and lp_smooth exports and alias identity."""
    from puremacro.lp import smooth_lp as fn1, lp_smooth as fn2
    from puremacro.lp.smooth import smooth_lp as fn3, lp_smooth as fn4

    assert fn1 is fn2
    assert fn1 is fn3
    assert fn3 is fn4
    assert callable(smooth_lp)


# =====================================================================
# 2. Interface Contract & Result Object Methods
# =====================================================================
def test_lpresult_contract_and_presentation_methods():
    """Verify LPResult contract: DataFrame subclass with all required presentation methods."""
    df, _ = _simulate_ar2(T=200, seed=10)
    res = smooth_lp(df, y="y", x="x", horizons=12, n_lags=2, lam="auto", selection="aic")

    # LPResult type inheritance
    assert isinstance(res, LPResult)
    assert isinstance(res, pd.DataFrame)

    # Columns contract
    expected_cols = {"h", "beta", "se", "lo", "hi", "lambda", "t"}
    assert expected_cols.issubset(set(res.columns))
    assert np.array_equal(res.index, res["h"])

    # Vector property accessors
    assert isinstance(res.point, np.ndarray) and len(res.point) == 13
    assert isinstance(res.se, np.ndarray) and len(res.se) == 13
    assert isinstance(res.ci_lower, np.ndarray) and len(res.ci_lower) == 13
    assert isinstance(res.ci_upper, np.ndarray) and len(res.ci_upper) == 13
    assert isinstance(res.t_stat, np.ndarray) and len(res.t_stat) == 13
    assert isinstance(res.horizons, np.ndarray) and len(res.horizons) == 13

    # Internal estimation attributes
    assert hasattr(res, "optimal_lambda") and res.optimal_lambda > 0
    assert hasattr(res, "df_lambda") and 0 < res.df_lambda <= 13
    assert hasattr(res, "selection_criterion") and res.selection_criterion == "aic"
    assert hasattr(res, "ci_type") and res.ci_type == "analytic"
    assert hasattr(res, "theta") and isinstance(res.theta, np.ndarray)
    assert hasattr(res, "vcov") and res.vcov.shape == (13, 13)
    assert hasattr(res, "vcov_theta") and isinstance(res.vcov_theta, np.ndarray)
    assert hasattr(res, "B") and res.B.shape[0] == 13
    assert hasattr(res, "P") and res.P.shape[0] == res.B.shape[1]

    # Required presentation methods
    summary_str = res.summary()
    assert isinstance(summary_str, str)
    assert "Local Projection Result" in summary_str
    assert "beta" in summary_str and "se" in summary_str

    md_str = res.to_markdown()
    assert isinstance(md_str, str)
    assert "|" in md_str

    latex_str = res.to_latex()
    assert isinstance(latex_str, str)
    assert "\\begin{tabular}" in latex_str

    typst_str = res.to_typst()
    assert isinstance(typst_str, str)
    assert "#table" in typst_str

    # Plot method returning matplotlib figure/axes
    fig, ax = plt.subplots(figsize=(7, 4))
    ret_ax = res.plot(ax=ax, title="Test Smooth LP IRF")
    assert ret_ax is not None
    plt.close(fig)


# =====================================================================
# 3. B-spline Basis & Penalty Matrix Mathematical Properties
# =====================================================================
def test_bspline_basis_and_penalty_properties():
    """Verify B-spline partition of unity and penalty matrix PSD structure."""
    horizons = np.arange(0, 21, dtype=float)
    B, n_basis = _build_bspline_basis(horizons, n_knots=6, degree=3)

    assert B.shape == (21, n_basis)
    assert not np.isnan(B).any()
    assert not np.isinf(B).any()

    # Partition of unity: each row must sum to 1.0
    row_sums = B.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    # Difference penalty matrix P = D_d' D_d
    P = _difference_penalty_matrix(n_basis, order=2)
    assert P.shape == (n_basis, n_basis)

    # P must be symmetric
    np.testing.assert_allclose(P, P.T, atol=1e-14)

    # P must be positive semi-definite: all eigenvalues >= -1e-12
    eigvals = np.linalg.eigvalsh(P)
    assert np.all(eigvals >= -1e-12)

    # For second difference, null space has dimension 2 (constant + linear)
    near_zeros = np.sum(np.abs(eigvals) < 1e-10)
    assert near_zeros == 2


# =====================================================================
# 4. Data-Driven Lambda Selection (AIC, BIC, GCV, CV)
# =====================================================================
@pytest.mark.parametrize("criterion", ["aic", "bic", "gcv", "cv"])
def test_lambda_selection_all_criteria(criterion):
    """Verify automated data-driven selection runs and finds positive finite lambda."""
    df, _ = _simulate_ar2(T=220, seed=101)
    res = smooth_lp(
        df,
        y="y",
        x="x",
        horizons=15,
        n_lags=2,
        lam="auto",
        selection=criterion,
    )
    assert res.optimal_lambda > 0.0
    assert np.isfinite(res.optimal_lambda)
    assert res.df_lambda > 0.0
    assert res["beta"].diff().abs().max() < 2.0


def test_bic_penalizes_more_than_aic():
    """BIC applies heavier penalty log(N) vs 2, typically selecting >= lambda."""
    df, _ = _simulate_ar2(T=250, seed=77)
    res_aic = smooth_lp(df, y="y", x="x", horizons=16, n_lags=2, lam="auto", selection="aic")
    res_bic = smooth_lp(df, y="y", x="x", horizons=16, n_lags=2, lam="auto", selection="bic")

    assert res_bic.optimal_lambda >= res_aic.optimal_lambda or np.isclose(
        res_bic.optimal_lambda, res_aic.optimal_lambda
    )


def test_explicit_lambda_overrides_selection():
    """Supplying explicit float lam sets optimal_lambda exactly."""
    df, _ = _simulate_ar2(T=200, seed=33)
    res = smooth_lp(df, y="y", x="x", horizons=10, n_lags=2, lam=0.042)
    assert res.optimal_lambda == 0.042
    np.testing.assert_allclose(res["lambda"].values, 0.042)


# =====================================================================
# 5. Inference: Analytical Sandwich HAC and Moving Block Bootstrap
# =====================================================================
def test_analytical_sandwich_hac_inference():
    """Verify analytical sandwich HAC standard errors and confidence bands."""
    df, _ = _simulate_ar2(T=220, seed=55)
    res = smooth_lp(df, y="y", x="x", horizons=12, n_lags=2, ci_type="analytic", alpha=0.05)

    assert np.all(res.se > 0)
    assert np.all(np.isfinite(res.se))
    assert np.all(res.ci_lower < res.point)
    assert np.all(res.point < res.ci_upper)


def test_moving_block_bootstrap_inference():
    """Verify moving block bootstrap produces finite, strictly positive SEs and valid bands."""
    df, _ = _simulate_ar2(T=200, seed=66)
    res = smooth_lp(
        df,
        y="y",
        x="x",
        horizons=10,
        n_lags=2,
        ci_type="bootstrap",
        n_boot=150,
        seed=123,
    )

    assert np.all(res.se > 0)
    assert np.all(np.isfinite(res.se))
    assert np.all(res.ci_lower <= res.ci_upper)
    assert res.ci_type == "bootstrap"


def test_confidence_level_scaling():
    """99% confidence interval (alpha=0.01) must be strictly wider than 90% (alpha=0.10)."""
    df, _ = _simulate_ar2(T=200, seed=99)
    res_90 = smooth_lp(df, y="y", x="x", horizons=10, n_lags=2, alpha=0.10)
    res_99 = smooth_lp(df, y="y", x="x", horizons=10, n_lags=2, alpha=0.01)

    width_90 = res_90.ci_upper - res_90.ci_lower
    width_99 = res_99.ci_upper - res_99.ci_lower
    assert np.all(width_99 > width_90)


# =====================================================================
# 6. Controls, Lag Configurations & Array Inputs
# =====================================================================
def test_smooth_lp_with_exogenous_controls():
    """Verify partialling out exogenous controls alongside autoregressive lags."""
    rng = np.random.default_rng(42)
    T = 200
    df, _ = _simulate_ar2(T=T, seed=42)
    df["c1"] = rng.standard_normal(T)
    df["c2"] = 0.5 * df["y"] + rng.standard_normal(T) * 0.2

    res = smooth_lp(df, y="y", x="x", horizons=8, n_lags=3, controls=["c1", "c2"])
    assert isinstance(res, LPResult)
    assert len(res) == 9
    assert np.all(np.isfinite(res.point))
    assert np.all(res.se > 0)


def test_smooth_lp_with_array_inputs():
    """Verify compatibility when y and x are passed as 1D numpy arrays."""
    rng = np.random.default_rng(123)
    T = 180
    y = rng.standard_normal(T)
    x = 0.3 * np.roll(y, 1) + rng.standard_normal(T)

    res = smooth_lp(y, y=None, x=x, horizons=6, n_lags=1)
    assert isinstance(res, LPResult)
    assert len(res) == 7
    assert np.all(np.isfinite(res.point))


def test_gls_weighting():
    """Verify smooth_lp with gls=True cross-horizon GLS weighting."""
    df, _ = _simulate_ar2(T=200, seed=88)
    res_pls = smooth_lp(df, y="y", x="x", horizons=10, n_lags=2, gls=False)
    res_gls = smooth_lp(df, y="y", x="x", horizons=10, n_lags=2, gls=True)

    assert isinstance(res_gls, LPResult)
    assert np.all(np.isfinite(res_gls.point))
    assert np.all(res_gls.se > 0)
    # Both estimators remain close to each other
    np.testing.assert_allclose(res_pls.point, res_gls.point, atol=0.25)


# =====================================================================
# 7. Empirical Verification on Synthetic AR(2) DGP (Acceptance Criterion §R3)
# =====================================================================
def test_empirical_variance_and_mse_reduction_over_unpenalized_lp():
    """Empirical Monte Carlo validation per Barnichon & Brownlees (2019):

    Smooth Local Projections achieves:
    1. Strictly lower empirical IRF variance than standard unpenalized OLS LP across horizons.
    2. Strictly lower total Mean Squared Error (MSE) than unpenalized OLS LP.
    """
    phi1, phi2 = 0.6, -0.25
    H = 15
    # True analytical IRF
    psi = np.zeros(H + 1, dtype=float)
    psi[0] = 1.0
    psi[1] = phi1
    for h in range(2, H + 1):
        psi[h] = phi1 * psi[h - 1] + phi2 * psi[h - 2]

    n_mc = 80
    T = 180
    rng = np.random.default_rng(2026)

    ols_estimates = np.zeros((n_mc, H + 1), dtype=float)
    smooth_estimates = np.zeros((n_mc, H + 1), dtype=float)

    for m in range(n_mc):
        burn = 60
        total = T + burn
        eps = rng.standard_normal(total)
        y = np.zeros(total, dtype=float)
        for t in range(2, total):
            y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]

        y_s = y[burn:]
        x_s = eps[burn:]
        df_sim = pd.DataFrame({"y": y_s, "x": x_s})

        # 1. Unpenalized OLS LP (estimated on common sample via prepare_lp_data)
        _, _, _, b_ols, _, _ = _prepare_lp_data(
            df=df_sim, y="y", x="x", horizons=list(range(H + 1)), n_lags=2, controls=None
        )
        ols_estimates[m, :] = b_ols

        # 2. Smooth LP with data-driven selection
        res_smooth = smooth_lp(
            df_sim,
            y="y",
            x="x",
            horizons=H,
            n_lags=2,
            lam="auto",
            selection="aic",
        )
        smooth_estimates[m, :] = res_smooth.point

    # Empirical variance across Monte Carlo draws
    var_ols = np.var(ols_estimates, axis=0).mean()
    var_smooth = np.var(smooth_estimates, axis=0).mean()

    # Empirical MSE against true analytical IRF
    mse_ols = np.mean((ols_estimates - psi) ** 2)
    mse_smooth = np.mean((smooth_estimates - psi) ** 2)

    # Acceptance Criteria §R3:
    # 1. Strictly lower empirical variance
    assert var_smooth < var_ols, (
        f"Smooth LP variance ({var_smooth:.6f}) was not lower than OLS LP ({var_ols:.6f})!"
    )

    # 2. Significant variance reduction (at least 20%)
    var_reduction_pct = (1.0 - var_smooth / var_ols) * 100.0
    assert var_reduction_pct > 20.0, (
        f"Variance reduction ({var_reduction_pct:.1f}%) was below 20% threshold!"
    )

    # 3. Strictly lower MSE
    assert mse_smooth < mse_ols, (
        f"Smooth LP MSE ({mse_smooth:.6f}) was not lower than OLS LP ({mse_ols:.6f})!"
    )


def test_unbiasedness_as_lambda_approaches_zero():
    """As lambda -> 0, Smooth LP penalty vanishes:
    1. With saturated basis (n_knots=H+1), point estimates match unpenalized OLS LP.
    2. With standard basis, point estimates match the unpenalized least squares projection.
    3. Over Monte Carlo simulations on AR(2) DGP, mean estimate recovers true analytical IRF.
    """
    df, psi = _simulate_ar2(T=250, seed=1234)
    H = 10
    _, _, _, b_ols, _, _ = _prepare_lp_data(
        df=df, y="y", x="x", horizons=list(range(H + 1)), n_lags=2, controls=None
    )

    # 1. Saturated basis exact equivalence to OLS
    res_sat = smooth_lp(df, y="y", x="x", horizons=H, n_lags=2, n_knots=H + 1, lam=1e-8)
    np.testing.assert_allclose(res_sat.point, b_ols, atol=1e-5)

    # 2. General basis matches unpenalized projection B (B'B)^{-1} B' b_ols
    res_zero = smooth_lp(df, y="y", x="x", horizons=H, n_lags=2, lam=1e-8)
    B = res_zero.B
    proj_ols = B @ np.linalg.solve(B.T @ B, B.T @ b_ols)
    np.testing.assert_allclose(res_zero.point, proj_ols, atol=1e-5)

    # 3. Monte Carlo unbiasedness against true population IRF
    rng = np.random.default_rng(888)
    n_mc = 40
    mc_estimates = np.zeros((n_mc, H + 1))
    for m in range(n_mc):
        total = 250 + 50
        e = rng.standard_normal(total)
        y = np.zeros(total)
        for t in range(2, total):
            y[t] = 0.6 * y[t - 1] - 0.25 * y[t - 2] + e[t]
        d_sim = pd.DataFrame({"y": y[50:], "x": e[50:]})
        res_m = smooth_lp(d_sim, y="y", x="x", horizons=H, n_lags=2, lam=1e-6)
        mc_estimates[m] = res_m.point

    mean_est = np.mean(mc_estimates, axis=0)
    bias = np.abs(mean_est - psi[:H + 1])
    assert np.max(bias) < 0.10, f"Max bias {np.max(bias)} exceeded tolerance 0.10!"


def test_shrinkage_to_linearity_as_lambda_approaches_infinity():
    """As lambda -> inf, the second difference penalty forces the spline coefficients to linearity."""
    df, _ = _simulate_ar2(T=200, seed=5678)
    res_low = smooth_lp(df, y="y", x="x", horizons=10, n_lags=2, lam=0.001)
    res_inf = smooth_lp(df, y="y", x="x", horizons=10, n_lags=2, lam=1e9)

    # Second differences of spline coefficients must be zero
    d2_theta = np.diff(res_inf.theta, n=2)
    np.testing.assert_allclose(d2_theta, 0.0, atol=1e-8)

    # Curvature (variance of second differences) of beta is reduced by orders of magnitude
    curv_low = float(np.var(np.diff(res_low.point, n=2)))
    curv_inf = float(np.var(np.diff(res_inf.point, n=2)))
    assert curv_inf < curv_low * 0.05


# =====================================================================
# 8. Error Handling & Edge Cases
# =====================================================================
def test_zero_variance_shock_raises_error():
    """Constant shock with zero variance must raise ValueError."""
    T = 100
    df = pd.DataFrame({"y": np.random.randn(T), "x": np.ones(T)})
    with pytest.raises(ValueError, match="near-zero variance"):
        smooth_lp(df, y="y", x="x", horizons=5, n_lags=1)


def test_insufficient_sample_raises_error():
    """Dataset shorter than horizon + lags must raise ValueError."""
    T = 15
    df = pd.DataFrame({"y": np.random.randn(T), "x": np.random.randn(T)})
    with pytest.raises(ValueError, match="Insufficient effective observations"):
        smooth_lp(df, y="y", x="x", horizons=20, n_lags=4)


def test_invalid_selection_criterion_raises_error():
    """Unsupported selection criterion raises ValueError."""
    df, _ = _simulate_ar2(T=150, seed=99)
    with pytest.raises(ValueError, match="Unknown selection criterion"):
        smooth_lp(df, y="y", x="x", horizons=5, n_lags=1, selection="invalid_crit")
