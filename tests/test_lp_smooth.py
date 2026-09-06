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

from puremacro.lp import smooth_lp, lp_smooth, LPResult, lp_hac
from puremacro.lp.smooth import (
    SmoothLPResult,
    _build_bspline_basis,
    _difference_penalty_matrix,
    _normal_equations,
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
    """Verify compatibility when y and x are passed as 1D numpy arrays.

    Both ``smooth_lp(y_arr, x_arr)`` (the lp_hac convention: response first,
    shock second) and ``smooth_lp(y_arr, x=x_arr)`` must give the DataFrame
    result.
    """
    rng = np.random.default_rng(123)
    T = 180
    y = rng.standard_normal(T)
    x = 0.3 * np.roll(y, 1) + rng.standard_normal(T)
    ref = smooth_lp(pd.DataFrame({"y": y, "x": x}), y="y", x="x", horizons=6, n_lags=1)

    res_kw = smooth_lp(y, y=None, x=x, horizons=6, n_lags=1)
    res_pos = smooth_lp(y, x, horizons=6, n_lags=1)
    for res in (res_kw, res_pos):
        assert isinstance(res, LPResult)
        assert len(res) == 7
        assert np.all(np.isfinite(res.point))
        np.testing.assert_allclose(res.point, ref.point, atol=1e-12)
        assert res.y_name == "y" and res.x_name == "x"


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
    _, _, s_ww, b_ols, _, _ = _prepare_lp_data(
        df=df, y="y", x="x", horizons=list(range(H + 1)), n_lags=2, controls=None
    )

    # 1. Saturated basis (H - degree interior knots -> H + 1 basis functions)
    #    reproduces the horizon-by-horizon OLS LP exactly.
    res_sat = smooth_lp(df, y="y", x="x", horizons=H, n_lags=2, n_knots=H - 3, lam=1e-8)
    assert res_sat.n_basis == H + 1
    np.testing.assert_allclose(res_sat.point, b_ols, atol=1e-5)

    # 2. General basis matches the unpenalized (sample-size weighted) projection
    #    B (B'SB)^{-1} B'S b_ols, S = diag(w~_h' w~_h) -- each horizon has its own
    #    sample, so the horizons enter the least-squares fit with weights s_h.
    res_zero = smooth_lp(df, y="y", x="x", horizons=H, n_lags=2, lam=1e-8)
    B = res_zero.B
    S = np.diag(s_ww)
    proj_ols = B @ np.linalg.solve(B.T @ S @ B, B.T @ S @ b_ols)
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
    """As lambda -> inf, the second difference penalty forces the spline coefficients to linearity.

    ``lam`` is the lambda of the stacked objective ||Y - X theta||^2 + lam theta'P theta,
    whose sum of squares is O(T); lam=1e12 is therefore the "infinite" penalty here.
    """
    df, _ = _simulate_ar2(T=200, seed=5678)
    res_low = smooth_lp(df, y="y", x="x", horizons=10, n_lags=2, lam=0.001)
    res_inf = smooth_lp(df, y="y", x="x", horizons=10, n_lags=2, lam=1e12)

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


# =====================================================================
# 9. Regression tests for the 2.3.x audit (r3-smooth-lp / review-r3-smooth)
# =====================================================================
def _simulate_shock_dgp(T: int = 200, seed: int = 0, with_control: bool = False) -> pd.DataFrame:
    """y_t = 0.6 y_{t-1} + x_t + 0.7 z_t + e_t with an exogenous shock x."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    z = rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.6 * y[t - 1] + x[t] + (0.7 * z[t] if with_control else 0.0) + 0.4 * rng.standard_normal()
    df = pd.DataFrame({"y": y, "x": x})
    if with_control:
        df["z"] = z
    return df


def test_positional_array_call_follows_lp_hac_convention():
    """smooth_lp(y_arr, x_arr) must treat the first array as the response and the second as the shock.

    Old behaviour (audit C6): with two positional arrays both the response and
    the shock were taken from the second argument, so the IRF was the shock
    regressed on itself (~[1, 0, 0, ...]) and x_name/y_name became the array repr.
    """
    df = _simulate_shock_dgp(T=200, seed=3)
    y_arr = df["y"].to_numpy()
    x_arr = df["x"].to_numpy()

    ref = smooth_lp(df, "y", "x", horizons=6, n_lags=2, lam=1e-8, n_knots=3)
    pos = smooth_lp(y_arr, x_arr, horizons=6, n_lags=2, lam=1e-8, n_knots=3)
    np.testing.assert_allclose(pos.point, ref.point, atol=1e-10)
    assert pos.y_name == "y" and pos.x_name == "x"
    assert len(pos.x_name) < 10 and len(pos.y_name) < 10

    # ... and coincides with lp_hac on the same (per-horizon) samples: the
    # shock-on-itself IRF would be ~1 at h=0 and ~0 afterwards.
    raw = lp_hac(y_arr, x_arr, horizons=range(7), n_lags=2)
    np.testing.assert_allclose(pos.point, raw["beta"].to_numpy(), atol=1e-8)
    assert abs(pos.point[1]) > 0.3

    # Series names propagate
    named = smooth_lp(df["y"].rename("gdp"), df["x"].rename("mp_shock"), horizons=6, n_lags=2)
    assert named.y_name == "gdp" and named.x_name == "mp_shock"


def test_mixed_dataframe_and_array_inputs():
    """smooth_lp(df, 'y', x_array) and smooth_lp(df, y_array, 'x') must work.

    Old behaviour: ValueError 'could not convert string to float: y'.
    """
    df = _simulate_shock_dgp(T=200, seed=4)
    ref = smooth_lp(df, "y", "x", horizons=6, n_lags=2, lam=0.5)
    mixed_x = smooth_lp(df, "y", df["x"].to_numpy(), horizons=6, n_lags=2, lam=0.5)
    mixed_y = smooth_lp(df, df["y"].to_numpy(), "x", horizons=6, n_lags=2, lam=0.5)
    np.testing.assert_allclose(mixed_x.point, ref.point, atol=1e-12)
    np.testing.assert_allclose(mixed_y.point, ref.point, atol=1e-12)
    assert mixed_x.x_name == "x" and mixed_y.y_name == "y"


def test_ambiguous_or_missing_array_inputs_raise():
    """Array input must never silently alias the shock to the response."""
    df = _simulate_shock_dgp(T=120, seed=5)
    y_arr = df["y"].to_numpy()
    x_arr = df["x"].to_numpy()
    with pytest.raises(ValueError, match="Ambiguous"):
        smooth_lp(y_arr, x_arr, x=x_arr, horizons=4, n_lags=1)
    with pytest.raises(ValueError, match="Shock array missing"):
        smooth_lp(y_arr, horizons=4, n_lags=1)
    with pytest.raises(ValueError, match="require df to be a DataFrame"):
        smooth_lp(y_arr, "x", horizons=4, n_lags=1)
    with pytest.raises(ValueError, match="not found in df columns"):
        smooth_lp(df, "y", "nope", horizons=4, n_lags=1)
    with pytest.raises(ValueError, match="required when df is a DataFrame"):
        smooth_lp(df, "y", horizons=4, n_lags=1)
    with pytest.raises(ValueError, match="Length mismatch"):
        smooth_lp(y_arr, x_arr[:-1], horizons=4, n_lags=1)


def test_controls_as_ndarray_with_dataframe_input():
    """controls may be a (T,) or (T, k) array (or DataFrame) alongside a DataFrame.

    Old behaviour (audit M15): ValueError 'The truth value of an array with more
    than one element is ambiguous' from ``list(controls or [])``.
    """
    df = _simulate_shock_dgp(T=200, seed=6, with_control=True)
    rng = np.random.default_rng(1)
    df["z2"] = rng.standard_normal(len(df))
    ref = smooth_lp(df, "y", "x", horizons=6, n_lags=2, lam=1.0, controls=["z", "z2"])

    arr2 = smooth_lp(df, "y", "x", horizons=6, n_lags=2, lam=1.0, controls=df[["z", "z2"]].to_numpy())
    np.testing.assert_allclose(arr2.point, ref.point, atol=1e-12)

    frame = smooth_lp(df, "y", "x", horizons=6, n_lags=2, lam=1.0, controls=df[["z", "z2"]])
    np.testing.assert_allclose(frame.point, ref.point, atol=1e-12)

    ref1 = smooth_lp(df, "y", "x", horizons=6, n_lags=2, lam=1.0, controls=["z"])
    arr1 = smooth_lp(df, "y", "x", horizons=6, n_lags=2, lam=1.0, controls=df["z"].to_numpy())
    np.testing.assert_allclose(arr1.point, ref1.point, atol=1e-12)

    # array-branch controls agree with the DataFrame branch
    arr_branch = smooth_lp(
        df["y"].to_numpy(), df["x"].to_numpy(), horizons=6, n_lags=2, lam=1.0,
        controls=df[["z", "z2"]].to_numpy(),
    )
    np.testing.assert_allclose(arr_branch.point, ref.point, atol=1e-12)

    with pytest.raises(ValueError, match="rows"):
        smooth_lp(df, "y", "x", horizons=6, n_lags=2, controls=np.ones(10))


def test_gls_cv_uses_the_pgls_estimator_inside_the_folds():
    """With gls=True and selection='cv' the fold estimator must be the same PGLS estimator.

    Old behaviour (audit M12/M113): the inner solve used the GLS left-hand side
    B'W^{-1}B but the OLS right-hand side B'b_tr, so the CV curve was wrong and
    lambda was pinned at the grid maximum (1e5) on every seed.
    """
    H, nl = 8, 2
    for seed in (11, 12, 13):
        df = _simulate_shock_dgp(T=150, seed=seed, with_control=True)
        res = smooth_lp(df, "y", "x", horizons=H, n_lags=nl, controls=["z"], selection="cv", gls=True)
        grid = res.lambda_grid
        assert res.optimal_lambda not in (grid[0], grid[-1]), "lambda pinned at the grid edge"

        # Independent replica of K-fold CV with the consistent PGLS fold estimator
        W, Y, s, b, U, t_eff = _prepare_lp_data(
            df, "y", "x", list(range(H + 1)), nl, ["z"], balanced=True
        )
        Omega = (U.T @ U) / t_eff.min()
        Omega = Omega + 1e-4 * np.trace(Omega) / (H + 1) * np.eye(H + 1)
        W_inv = np.linalg.inv(Omega)
        B, K = _build_bspline_basis(np.arange(H + 1, dtype=float))
        P = _difference_penalty_matrix(K, 2)
        T0 = W.shape[0]
        folds = np.array_split(np.arange(T0), min(5, max(2, T0 // 10)))

        def cv_score(lam: float) -> float:
            err = 0.0
            for fold in folds:
                tr = np.setdiff1d(np.arange(T0), fold)
                A, c = _normal_equations(W[tr], Y[tr], W_inv)
                beta = B @ np.linalg.solve(B.T @ A @ B + lam * P, B.T @ c)
                err += float(np.sum((Y[fold] - W[fold] * beta[None, :]) ** 2))
            return err

        lam_rep = grid[int(np.argmin([cv_score(l) for l in grid]))]
        assert np.isclose(res.optimal_lambda, lam_rep)


def test_lambda_is_on_the_documented_stacked_objective_scale():
    """``lam`` must be the lambda of min ||Y - X theta||^2 + lam theta'P theta with X = B (x) w~.

    Old behaviour (audit M16/M115): the code solved (B'B + lam P) theta = B' b_ols,
    i.e. the effective penalty was lam * (w~'w~), so the same lam smoothed
    differently as T changed and did not match the documented objective.
    """
    df = _simulate_shock_dgp(T=200, seed=7, with_control=True)
    H, nl = 8, 2
    horizons = list(range(H + 1))
    ctl = ["z"]

    # Explicit stacked design with horizon-specific unpenalised controls on
    # horizon-specific samples (the documented PLS problem).
    sub = df.copy()
    for lag in range(1, nl + 1):
        for v in ("x", "y", "z"):
            sub[f"{v}L{lag}"] = sub[v].shift(lag)
    zcols = [f"{v}L{lag}" for lag in range(1, nl + 1) for v in ("x", "y", "z")] + ctl
    B, nb = _build_bspline_basis(np.array(horizons, dtype=float))
    P = _difference_penalty_matrix(nb, 2)
    n_z = len(zcols) + 1
    blocks, ys = [], []
    for h in horizons:
        s_h = sub.copy()
        s_h["lead"] = s_h["y"].shift(-h)
        s_h = s_h.dropna()
        Zh = np.column_stack([np.ones(len(s_h))] + [s_h[c].to_numpy() for c in zcols])
        Xh = np.zeros((len(s_h), nb + (H + 1) * n_z))
        Xh[:, :nb] = np.outer(s_h["x"].to_numpy(), B[h, :])
        Xh[:, nb + h * n_z: nb + (h + 1) * n_z] = Zh
        blocks.append(Xh)
        ys.append(s_h["lead"].to_numpy())
    X = np.vstack(blocks)
    Y = np.concatenate(ys)
    P_big = np.zeros((X.shape[1], X.shape[1]))
    P_big[:nb, :nb] = P

    for lam in (0.01, 0.5, 10.0, 500.0):
        res = smooth_lp(df, "y", "x", horizons=H, n_lags=nl, controls=ctl, lam=lam)
        coef = np.linalg.solve(X.T @ X + lam * P_big, X.T @ Y)
        np.testing.assert_allclose(res.point, B @ coef[:nb], atol=1e-10)
        # effective df = trace of the stacked hat matrix minus the control df
        hat_tr = np.trace(np.linalg.solve(X.T @ X + lam * P_big, X.T @ X))
        assert np.isclose(res.df_lambda, hat_tr - (H + 1) * n_z, atol=1e-8)

    # The automatic grid is reported on the same scale
    auto = smooth_lp(df, "y", "x", horizons=H, n_lags=nl, controls=ctl)
    assert auto.optimal_lambda in auto.lambda_grid
    assert auto.lambda_grid[0] < auto.optimal_lambda < auto.lambda_grid[-1]


def test_per_horizon_samples_match_horizon_by_horizon_lp():
    """Each horizon uses its own sample t + h <= T (Barnichon-Brownlees), not one balanced sample.

    Old behaviour (audit M119): every horizon was truncated to the h=H sample
    (T - H - n_lags observations), so h=0 discarded H usable observations and
    the lambda -> 0 limit did not coincide with lp_hac.
    """
    df = _simulate_shock_dgp(T=200, seed=8, with_control=True)
    H, nl = 8, 2
    res = smooth_lp(df, "y", "x", horizons=H, n_lags=nl, controls=["z"], lam=0.0, n_knots=H - 3)
    np.testing.assert_array_equal(res.n_obs, [200 - nl - h for h in range(H + 1)])
    assert res.sample == "per-horizon"

    raw = lp_hac(df, y="y", x="x", horizons=range(H + 1), n_lags=nl, controls=["z"])
    np.testing.assert_allclose(res.point, raw["beta"].to_numpy(), atol=1e-10)

    # gls=True needs a balanced residual panel: common sample, documented
    res_gls = smooth_lp(df, "y", "x", horizons=H, n_lags=nl, controls=["z"], gls=True)
    assert res_gls.sample == "balanced"
    np.testing.assert_array_equal(res_gls.n_obs, [200 - nl - H] * (H + 1))


def test_invalid_ci_type_raises():
    """ci_type must be validated.

    Old behaviour: ci_type='block_bootstrap' or 'wild' silently produced the
    analytic result and res.ci_type echoed the bogus string.
    """
    df = _simulate_shock_dgp(T=120, seed=9)
    with pytest.raises(ValueError, match="Unknown ci_type"):
        smooth_lp(df, "y", "x", horizons=5, n_lags=1, ci_type="wild")
    res = smooth_lp(df, "y", "x", horizons=5, n_lags=1, ci_type="boot", n_boot=50, seed=0)
    assert res.ci_type == "bootstrap"
    res = smooth_lp(df, "y", "x", horizons=5, n_lags=1, ci_type="Analytic")
    assert res.ci_type == "analytic"


def test_invalid_selection_raises_even_with_fixed_lambda():
    """An unknown selection criterion must raise even when lam is a number.

    Old behaviour: lam=0.1, selection='not_a_criterion' ran silently and
    res.selection_criterion recorded the bogus string.
    """
    df = _simulate_shock_dgp(T=120, seed=10)
    with pytest.raises(ValueError, match="Unknown selection criterion"):
        smooth_lp(df, "y", "x", horizons=5, n_lags=1, lam=0.1, selection="not_a_criterion")
    res = smooth_lp(df, "y", "x", horizons=5, n_lags=1, lam=0.1, selection="bic")
    assert res.selection_criterion == "fixed"
    assert res.optimal_lambda == 0.1


def test_lambda_argument_validation():
    """lam='AUTO' is accepted (case-insensitive); negative, NaN or other strings raise.

    Old behaviour: lam='AUTO' -> 'could not convert string to float'; lam=-5.0
    was accepted silently.
    """
    df = _simulate_shock_dgp(T=150, seed=11)
    ref = smooth_lp(df, "y", "x", horizons=6, n_lags=1, lam="auto")
    for lam in ("AUTO", " Auto ", None):
        res = smooth_lp(df, "y", "x", horizons=6, n_lags=1, lam=lam)
        assert res.optimal_lambda == ref.optimal_lambda
        assert res.selection_criterion == "aic"
    for bad in (-5.0, float("nan"), float("inf"), "gcv"):
        with pytest.raises(ValueError, match="lam must be"):
            smooth_lp(df, "y", "x", horizons=6, n_lags=1, lam=bad)
    res0 = smooth_lp(df, "y", "x", horizons=6, n_lags=1, lam=0.0)
    assert res0.optimal_lambda == 0.0 and np.all(np.isfinite(res0.point))


def test_horizons_validation_gives_clean_messages():
    """horizons=0 (or any single horizon) must raise a clear ValueError.

    Old behaviour: raw scipy error 'Need at least 4 knots for degree 1'.
    """
    df = _simulate_shock_dgp(T=120, seed=12)
    with pytest.raises(ValueError, match="at least two horizons"):
        smooth_lp(df, "y", "x", horizons=0, n_lags=1)
    with pytest.raises(ValueError, match="at least two distinct horizons"):
        smooth_lp(df, "y", "x", horizons=[3], n_lags=1)
    with pytest.raises(ValueError, match="non-negative"):
        smooth_lp(df, "y", "x", horizons=[-1, 0, 1], n_lags=1)
    res = smooth_lp(df, "y", "x", horizons=1, n_lags=1)
    assert len(res) == 2 and np.all(np.isfinite(res.point))


def test_n_knots_counts_interior_knots_and_clipping_warns():
    """n_knots is the number of interior knots (basis size n_knots + degree + 1); clipping warns.

    Old behaviour (audit M118): n_knots counted the boundary knots (n_knots=6 gave
    4 interior knots) and any request above H - degree was silently replaced.
    """
    horizons = np.arange(0, 21, dtype=float)
    B, n_basis = _build_bspline_basis(horizons, n_knots=6, degree=3)
    assert n_basis == 6 + 3 + 1
    B1, n_basis1 = _build_bspline_basis(horizons, n_knots=0, degree=3)
    assert n_basis1 == 4  # a single cubic polynomial
    with pytest.warns(UserWarning, match="reduced to n_knots=17"):
        B_clip, n_clip = _build_bspline_basis(horizons, n_knots=30, degree=3)
    assert n_clip == 21

    df = _simulate_shock_dgp(T=200, seed=13)
    res = smooth_lp(df, "y", "x", horizons=10, n_lags=1, n_knots=2)
    assert res.n_knots == 2 and res.n_basis == 6 and res.degree == 3
    with pytest.warns(UserWarning, match="more than the 11 horizons"):
        res_clip = smooth_lp(df, "y", "x", horizons=10, n_lags=1, n_knots=11)
    assert res_clip.n_knots == 7 and res_clip.n_basis == 11
    with pytest.raises(ValueError, match="non-negative"):
        smooth_lp(df, "y", "x", horizons=10, n_lags=1, n_knots=-1)


def test_summary_reports_lambda_and_metadata_survive_pandas_ops():
    """summary() must report the selected lambda; estimation metadata must survive pandas ops.

    Old behaviour: summary() was the plain h/beta/se/lo/hi table and
    optimal_lambda/theta/vcov/B/P were lost after .copy(), .iloc[:3] or column selection.
    """
    df = _simulate_shock_dgp(T=200, seed=14)
    res = smooth_lp(df, "y", "x", horizons=10, n_lags=2, selection="bic")
    assert isinstance(res, SmoothLPResult) and isinstance(res, LPResult)
    txt = res.summary()
    assert "Local Projection Result" in txt
    assert f"lambda = {res.optimal_lambda:.4g}" in txt
    assert "selected by BIC" in txt
    assert "effective degrees of freedom" in txt
    assert "per-horizon" in txt

    fixed = smooth_lp(df, "y", "x", horizons=10, n_lags=2, lam=2.0)
    assert "fixed by the user" in fixed.summary()

    for obj in (res.copy(), res.iloc[:3], res[["h", "beta"]]):
        assert isinstance(obj, SmoothLPResult)
        assert obj.optimal_lambda == res.optimal_lambda
        assert obj.df_lambda == res.df_lambda
        for name in ("theta", "vcov", "vcov_theta", "B", "P", "lambda_grid", "n_obs"):
            assert getattr(obj, name) is not None
        assert obj.selection_criterion == "bic" and obj.ci_type == "analytic"
        assert obj.n_knots == res.n_knots and obj.degree == 3 and obj.penalty_order == 2
        assert obj.gls is False
    assert res.metadata["optimal_lambda"] == res.optimal_lambda
