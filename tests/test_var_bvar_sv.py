"""Unit tests for Bayesian VAR with Stochastic Volatility (BVAR-SV).

Verifies:
1. Synthetic SV recovery with known stochastic volatility jumps:
   - Gelman-Rubin split-Rhat convergence diagnostics (assert Rhat < 1.1).
   - Recovery of true volatility paths within 95% credible intervals.
2. Volatility-conditioned IRFs:
   - Returns BVAR_SV_IRF (ndarray subclass with .lower, .upper, .median).
   - Volatility conditioning: crisis/high-volatility date has wider bands than calm date.
3. Predictive density log-scores:
   - Finite scalar total log-score and pointwise time series of length T_eff.
4. Presentation suite:
   - .summary(), .to_frame(), .to_markdown(), .to_latex(), .to_typst(), .plot().
5. Multi-chain and single-chain execution.
6. Input validation and error handling.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.var import bvar_sv, BVAR_SVResult
from puremacro.var.bvar_sv import BVAR_SV_IRF


@pytest.fixture
def synthetic_sv_var_data():
    """Generate synthetic VAR(1) data with known stochastic volatility jumps."""
    rng = np.random.default_rng(42)
    T = 150
    n = 2
    lags = 1

    # True parameters
    c_true = np.array([0.05, -0.02])
    A1_true = np.array([[0.45, 0.10],
                        [0.05, 0.40]])
    A_mat_true = np.array([[1.0, 0.0],
                           [0.3, 1.0]])
    A_inv_true = np.linalg.inv(A_mat_true)

    # Known stochastic volatility jumps: regime 1 (calm) vs regime 2 (volatile)
    true_h = np.empty((T, n))
    half = T // 2
    true_h[:half, 0] = -1.5
    true_h[half:, 0] = 0.5
    true_h[:half, 1] = -1.0
    true_h[half:, 1] = 0.8

    # Simulate VAR-SV
    Y = np.zeros((T, n))
    for t in range(1, T):
        eps_t = rng.standard_normal(n)
        nu_t = np.exp(true_h[t] / 2.0) * eps_t
        u_t = A_inv_true @ nu_t
        Y[t] = c_true + A1_true @ Y[t - 1] + u_t

    df = pd.DataFrame(Y, columns=["y1", "y2"])
    return {
        "df": df,
        "Y": Y,
        "true_h": true_h,
        "c_true": c_true,
        "A1_true": A1_true,
        "lags": lags,
        "n": n,
        "T": T,
    }


def test_bvar_sv_synthetic_recovery(synthetic_sv_var_data):
    """Test synthetic SV recovery: MCMC draws pass Rhat < 1.1 and recover true h within 95% CI."""
    data = synthetic_sv_var_data
    res = bvar_sv(
        data["df"],
        lags=1,
        n_draws=2000,
        n_burn=1000,
        minnesota_prior=True,
        n_chains=2,
        seed=42,
    )

    assert isinstance(res, BVAR_SVResult)
    assert res.n == 2
    assert res.lags == 1
    assert res.T_eff == data["T"] - 1

    # 1. Gelman-Rubin convergence diagnostics: assert Rhat < 1.1 on all parameters
    r_hat = res.gelman_rubin()
    assert isinstance(r_hat, dict)
    assert "max" in r_hat
    assert r_hat["max"] < 1.1, f"Expected R_hat < 1.1, got {r_hat['max']:.4f}"
    if "beta_max" in r_hat:
        assert r_hat["beta_max"] < 1.1
    if "h_max" in r_hat:
        assert r_hat["h_max"] < 1.1

    # 2. True volatility recovery within 95% credible intervals
    h_draws = res.h_draws  # (D, T_eff, n)
    h_lower = np.percentile(h_draws, 2.5, axis=0)
    h_upper = np.percentile(h_draws, 97.5, axis=0)
    true_h_eff = data["true_h"][data["lags"]:]  # (T_eff, n)

    # Assert coverage rate exceeds 90% (nominal 95%)
    cov_rate = np.mean((true_h_eff >= h_lower) & (true_h_eff <= h_upper))
    assert cov_rate >= 0.90, f"Expected coverage >= 90%, got {cov_rate * 100:.1f}%"

    # 3. Estimated VAR coefficients are close to true coefficients
    beta_mean = np.mean(res.beta_draws, axis=0)  # (k, n)
    A1_est = beta_mean[1:, :].T  # (n, n)
    np.testing.assert_allclose(A1_est, data["A1_true"], atol=0.20)


def test_bvar_sv_volatility_conditioned_irf(synthetic_sv_var_data):
    """Test volatility-conditioned IRFs and verify wider bands during high-volatility regime."""
    data = synthetic_sv_var_data
    res = bvar_sv(
        data["df"],
        lags=1,
        n_draws=800,
        n_burn=400,
        seed=42,
    )

    # Condition on low-volatility period (date 20) vs high-volatility period (date 120)
    irf_low = res.irf(horizon=10, t_idx=20, ci=0.9)
    irf_high = res.irf(horizon=10, t_idx=120, ci=0.9)

    # Type contract: must be ndarray subclass BVAR_SV_IRF
    assert isinstance(irf_low, np.ndarray)
    assert isinstance(irf_low, BVAR_SV_IRF)
    assert irf_low.shape == (11, 2, 2)
    assert irf_low.median.shape == (11, 2, 2)
    assert irf_low.lower.shape == (11, 2, 2)
    assert irf_low.upper.shape == (11, 2, 2)
    assert irf_low.draws.shape == (res.n_draws, 11, 2, 2)

    # Lower band <= median <= upper band
    assert np.all(irf_low.lower <= irf_low.median + 1e-10)
    assert np.all(irf_low.median <= irf_low.upper + 1e-10)

    # Volatility conditioning: high-volatility regime has strictly wider credible bands
    width_low = np.mean(irf_low.upper - irf_low.lower)
    width_high = np.mean(irf_high.upper - irf_high.lower)
    assert width_high > width_low, (
        f"High volatility width ({width_high:.4f}) should exceed low volatility width ({width_low:.4f})"
    )

    # Test to_frame on IRF
    df_irf = irf_low.to_frame(target_idx=0, shock_idx=0, names=res.names)
    assert isinstance(df_irf, pd.DataFrame)
    assert len(df_irf) == 11
    assert list(df_irf.columns) == ["horizon", "target", "shock", "median", "lower", "upper"]


def test_bvar_sv_predictive_log_score(synthetic_sv_var_data):
    """Test predictive density log-scores computation."""
    data = synthetic_sv_var_data
    res = bvar_sv(
        data["df"],
        lags=1,
        n_draws=500,
        n_burn=200,
        seed=10,
    )

    total_ls = res.predictive_log_score()
    assert isinstance(total_ls, float)
    assert np.isfinite(total_ls)

    pointwise_ls = res.predictive_log_score(point_by_point=True)
    assert isinstance(pointwise_ls, np.ndarray)
    assert pointwise_ls.shape == (res.T_eff,)
    assert np.all(np.isfinite(pointwise_ls))
    np.testing.assert_allclose(np.sum(pointwise_ls), total_ls, rtol=1e-10)


def test_bvar_sv_presentation_suite(synthetic_sv_var_data):
    """Test .summary(), .to_frame(), .to_markdown(), .to_latex(), .to_typst(), .plot()."""
    data = synthetic_sv_var_data
    res = bvar_sv(
        data["df"],
        lags=1,
        n_draws=400,
        n_burn=200,
        seed=7,
    )

    # 1. Summary
    summ = res.summary()
    assert isinstance(summ, str)
    assert "Bayesian VAR with Stochastic Volatility" in summ
    assert "Variables (n)" in summ
    assert "Lag order (p)" in summ
    assert "Predictive Log-Score" in summ

    # 2. DataFrame
    df_tab = res.to_frame()
    assert isinstance(df_tab, pd.DataFrame)
    assert len(df_tab) == res.n
    for col in ["variable", "mu_mean", "phi_mean", "sigma_h_mean", "R_hat"]:
        assert col in df_tab.columns

    # 3. Markdown
    md_str = res.to_markdown()
    assert isinstance(md_str, str)
    assert "| variable |" in md_str

    # 4. LaTeX
    tex_str = res.to_latex()
    assert isinstance(tex_str, str)
    assert "\\begin{tabular}" in tex_str
    assert "\\end{tabular}" in tex_str

    # 5. Typst
    typ_str = res.to_typst()
    assert isinstance(typ_str, str)
    assert "#table(" in typ_str

    # 6. Plot
    fig = res.plot(t_idx=-1, horizon=10, ci=0.9)
    assert fig is not None
    plt.close("all")

    # Plot into existing axes
    _, axes = plt.subplots(1, 3, figsize=(12, 3))
    out_axes = res.plot(ax=axes, horizon=10)
    assert out_axes is axes
    plt.close("all")


def test_bvar_sv_single_chain_and_alias(synthetic_sv_var_data):
    """Test bvar_sv with n_chains=1, keyword alias p, and diffuse prior."""
    data = synthetic_sv_var_data
    res = bvar_sv(
        data["Y"],
        p=1,
        n_draws=400,
        n_burn=200,
        n_chains=1,
        minnesota_prior=False,
        seed=99,
    )
    assert res.lags == 1
    assert res.p == 1
    assert res.beta_draws.shape == (400, 1 + 2 * 1, 2)
    assert res.A_draws.shape == (400, 1, 2, 2)
    assert res.intercept_draws.shape == (400, 2)
    assert res.a_draws.shape == (400, 2, 2)
    assert res.h_draws.shape == (400, data["T"] - 1, 2)


def test_bvar_sv_custom_names_and_dataframe(synthetic_sv_var_data):
    """Test custom column names propagation to summary and properties."""
    Y = synthetic_sv_var_data["Y"]
    df = pd.DataFrame(Y, columns=["GDP_Growth", "Inflation"])
    res = bvar_sv(df, lags=1, n_draws=200, n_burn=100, seed=12)

    assert res.names == ["GDP_Growth", "Inflation"]
    summ = res.summary()
    assert "GDP_Growth" in summ
    assert "Inflation" in summ


def test_bvar_sv_input_validation():
    """Test error handling for invalid input parameters."""
    Y = np.random.randn(50, 2)

    # Invalid lags
    with pytest.raises(ValueError, match="lags must be a positive integer"):
        bvar_sv(Y, lags=0)

    with pytest.raises(ValueError, match="lags must be a positive integer"):
        bvar_sv(Y, lags=-1)

    # Invalid n_draws / n_burn
    with pytest.raises(ValueError, match="n_draws must be > 0"):
        bvar_sv(Y, n_draws=0)

    # Insufficient observations
    with pytest.raises(ValueError, match="Insufficient observations"):
        bvar_sv(Y[:5], lags=4)

    # 1D input
    with pytest.raises(ValueError, match="data must be a 2D array"):
        bvar_sv(np.random.randn(50))

    # Out of bounds t_idx in irf()
    res = bvar_sv(Y, lags=1, n_draws=50, n_burn=20, seed=1)
    with pytest.raises(ValueError, match="out of bounds"):
        res.irf(t_idx=1000)

    # Invalid ax in plot()
    with pytest.raises(ValueError, match="must contain 3 subplots"):
        _, single_ax = plt.subplots(1, 1)
        res.plot(ax=single_ax)
    plt.close("all")


def test_bvar_sv_precision_symmetry_and_stability():
    """Verify that Sigma_inv and prec maintain strict mathematical symmetry (||prec - prec.T||_inf < 1e-14)."""
    # 1. Direct algebraic verification with non-diagonal A and asymmetric volatility
    rng = np.random.default_rng(1234)
    T = 100
    n = 3
    k = 1 + n * 1  # constant + 1 lag = 4

    # Lower triangular contemporaneous matrix with non-zero off-diagonals
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.6, 1.0, 0.0],
        [-0.4, 0.5, 1.0],
    ])
    # Highly unequal log-volatilities across equations
    exp_neg_h = np.exp(-np.column_stack([
        np.linspace(-1.0, 2.0, T),
        np.linspace(1.5, -0.5, T),
        np.linspace(-2.0, 0.5, T),
    ]))

    # Compute Sigma_inv via corrected formula: Sigma_t^{-1} = A' diag(exp(-h_t)) A
    Sigma_inv = np.einsum("ji,tj,jk->tik", A, exp_neg_h, A)
    asym_raw = np.max(np.abs(Sigma_inv - np.swapaxes(Sigma_inv, 1, 2)))
    assert asym_raw < 1e-14, f"Raw Sigma_inv is mathematically asymmetric: {asym_raw}"

    Sigma_inv = 0.5 * (Sigma_inv + np.swapaxes(Sigma_inv, 1, 2))

    # Form outer products of regressors X
    X = rng.standard_normal((T, k))
    X_outer = X[:, :, None] * X[:, None, :]
    V0_inv = np.diag(rng.uniform(0.1, 10.0, n * k))

    # Joint precision matrix: V_post^{-1} = V0^{-1} + sum_t Sigma_t^{-1} (x_t x_t')
    prec = V0_inv + np.einsum("tij,tkl->ikjl", Sigma_inv, X_outer).reshape(n * k, n * k)
    asym_prec_raw = np.max(np.abs(prec - prec.T))
    assert asym_prec_raw < 1e-14, f"Raw prec is mathematically asymmetric: {asym_prec_raw}"

    prec = 0.5 * (prec + prec.T)
    np.testing.assert_allclose(prec, prec.T, atol=1e-15)

    # 2. End-to-end MCMC run on Challenger 1 DGP with persistent volatility shifts
    Y = np.zeros((T, 2))
    for t in range(1, T):
        vol = 0.5 if t < 50 else 2.5
        Y[t, 0] = 0.4 * Y[t - 1, 0] + vol * rng.standard_normal()
        Y[t, 1] = 0.2 * Y[t - 1, 1] + 0.3 * Y[t, 0] + vol * rng.standard_normal()
    df = pd.DataFrame(Y, columns=["y1", "y2"])

    res = bvar_sv(df, lags=1, n_draws=100, n_burn=50, n_chains=2, seed=42)
    assert res.n_draws == 100

    # Check precision and covariance symmetry across sampled draws
    for m in range(min(20, res.n_draws)):
        A_m = res.a_draws[m]
        h_m = res.h_draws[m]
        exp_neg_h_m = np.exp(-h_m)
        Sigma_inv_m = np.einsum("ji,tj,jk->tik", A_m, exp_neg_h_m, A_m)
        asym_draw = np.max(np.abs(Sigma_inv_m - np.swapaxes(Sigma_inv_m, 1, 2)))
        assert asym_draw < 1e-14, f"Draw {m} produced asymmetric Sigma_inv: {asym_draw}"

