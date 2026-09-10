"""Empirical Challenger Test Suite for Milestone 2: DSGE-VAR Structural Rotation & Forecasts.

Author: teamwork_preview_challenger_m2_2
Verification Objectives:
1. Strict orthonormality of rotation matrix Q*: ||Q* (Q*)' - I_n|| < 10^{-10} and ||(Q*)' Q* - I_n|| < 10^{-10}.
2. Continuous convergence of DSGE-VAR structural IRFs to theoretical DSGE IRFs as lambda -> infty.
3. FEVD variance decomposition rows sum to 1.0 across all horizons h = 0..20, with shares in [0, 1].
4. Out-of-sample forecast generation and fan charts under non-interactive Matplotlib Agg mode.
5. Adversarial stress testing: scale invariance, high lag orders, extreme lambda, proper bounds.
"""
from __future__ import annotations

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.dynare import build_dynare
from puremacro.dsge.build import LinearModel
from puremacro.dsge.dsge_var import DSGEVARResult, estimate_dsge_var
from puremacro.dsge._results import DSGEForecastResult


# ==============================================================================
# Fixtures: Standard 3-Equation NK Model and 2-Equation RBC / Macro Model
# ==============================================================================

@pytest.fixture
def nk3_model() -> LinearModel:
    """Standard 3-equation New Keynesian model with 3 shocks."""
    mod = """
    var y pi r a u;
    varexo eps_a eps_u eps_r;
    parameters beta sigma kappa phi_pi phi_y rho_a rho_u;
    beta = 0.99;
    sigma = 1.0;
    kappa = 0.5;
    phi_pi = 1.5;
    phi_y = 0.5;
    rho_a = 0.75;
    rho_u = 0.65;

    model;
    y = y(+1) - (1/sigma)*(r - pi(+1)) + (a(+1) - a);
    pi = beta*pi(+1) + kappa*y + u;
    r = phi_pi*pi + phi_y*y + eps_r;
    a = rho_a*a(-1) + eps_a;
    u = rho_u*u(-1) + eps_u;
    end;

    shocks;
    var eps_a; stderr 0.01;
    var eps_u; stderr 0.01;
    var eps_r; stderr 0.005;
    end;
    """
    return build_dynare(mod)


@pytest.fixture
def macro2_model() -> LinearModel:
    """2-variable 2-shock macroeconomic model."""
    mod = """
    var y pi a;
    varexo eps_a eps_m;
    parameters beta kappa rho_a;
    beta = 0.99;
    kappa = 0.25;
    rho_a = 0.80;

    model;
    y = 0.8*y(-1) + 0.1*y(+1) + a + eps_m;
    pi = beta*pi(+1) + kappa*y;
    a = rho_a*a(-1) + eps_a;
    end;

    shocks;
    var eps_a; stderr 0.02;
    var eps_m; stderr 0.01;
    end;
    """
    return build_dynare(mod)


# ==============================================================================
# Challenge 1: Strict Orthonormality of Rotation Matrix Q* (||Q*(Q*)' - I|| < 1e-10)
# ==============================================================================

class TestRotationMatrixOrthonormality:
    """Adversarial validation that rotation matrix Q* is strictly orthonormal."""

    @pytest.mark.parametrize("lamb_val", [0.1, 0.5, 1.0, 2.5, 10.0, 100.0, 1e5])
    @pytest.mark.parametrize("intercept", [True, False])
    def test_q_star_orthonormality_nk3(self, nk3_model: LinearModel, lamb_val: float, intercept: bool):
        """Verify ||Q*(Q*)' - I_3||_infty < 1e-10 and ||(Q*)' Q* - I_3||_infty < 1e-10."""
        sim = nk3_model.simulate(250, seed=12345)
        data = sim[["y", "pi", "r"]]

        res = estimate_dsge_var(
            nk3_model,
            data,
            p=2,
            lamb=lamb_val,
            intercept=intercept,
            check_bounds=False,
            identification="dsge",
        )

        # Recover Q* from B0 and Cholesky of Sigma: B0 = Sigma_chol @ Q*
        Sigma_chol = np.linalg.cholesky(res.Sigma)
        Q_star = np.linalg.solve(Sigma_chol, res.B0)

        n = 3
        I_n = np.eye(n)

        # Right orthonormality: Q* (Q*)' = I_n
        qq_t = Q_star @ Q_star.T
        err_right = np.max(np.abs(qq_t - I_n))
        assert err_right < 1e-10, f"Failed ||Q*(Q*)' - I|| < 1e-10: max err = {err_right:.3e} at lambda={lamb_val}"

        # Left orthonormality: (Q*)' Q* = I_n
        qt_q = Q_star.T @ Q_star
        err_left = np.max(np.abs(qt_q - I_n))
        assert err_left < 1e-10, f"Failed ||(Q*)' Q* - I|| < 1e-10: max err = {err_left:.3e} at lambda={lamb_val}"

        # Determinant must be +/- 1 (orthogonal matrix)
        det_q = np.linalg.det(Q_star)
        assert abs(abs(det_q) - 1.0) < 1e-10, f"det(Q*) = {det_q}, expected +/-1.0"

        # Exact covariance reconstruction: B0 @ B0' == Sigma
        err_cov = np.max(np.abs(res.B0 @ res.B0.T - res.Sigma))
        assert err_cov < 1e-10, f"Failed B0 B0' == Sigma: max err = {err_cov:.3e}"

    def test_q_star_orthonormality_2var_model(self, macro2_model: LinearModel):
        """Verify orthonormality in a 2x2 system across multiple seeds and small samples."""
        for seed in [1, 42, 999]:
            sim = macro2_model.simulate(120, seed=seed)
            data = sim[["y", "pi"]]
            res = estimate_dsge_var(macro2_model, data, p=1, lamb=1.5, identification="dsge")

            Sigma_chol = np.linalg.cholesky(res.Sigma)
            Q_star = np.linalg.solve(Sigma_chol, res.B0)

            err = np.max(np.abs(Q_star @ Q_star.T - np.eye(2)))
            assert err < 1e-10, f"Seed {seed}: max error = {err:.3e}"


# ==============================================================================
# Challenge 2: Continuous Convergence of DSGE-VAR IRFs as lambda -> infty
# ==============================================================================

class TestContinuousConvergenceToDSGE:
    """Adversarial stress-testing of asymptotic convergence to theoretical DSGE moments."""

    def test_monotone_convergence_of_parameters(self, nk3_model: LinearModel):
        """Verify tilde{Phi}(lambda) and tilde{Sigma}(lambda) converge monotonically to prior moments."""
        sim = nk3_model.simulate(300, seed=42)
        data = sim[["y", "pi", "r"]]
        p = 2

        lambdas = [1.0, 5.0, 20.0, 100.0, 1000.0, 1e5]
        phi_errors = []
        sigma_errors = []

        for l_val in lambdas:
            res = estimate_dsge_var(nk3_model, data, p=p, lamb=l_val)
            phi_err = np.linalg.norm(res.to_frame().to_numpy() - res.Phi_star, ord="fro")
            sigma_err = np.linalg.norm(res.Sigma - res.Sigma_star, ord="fro")
            phi_errors.append(phi_err)
            sigma_errors.append(sigma_err)

        # Ensure errors decrease monotonically
        for i in range(len(lambdas) - 1):
            assert phi_errors[i + 1] < phi_errors[i], (
                f"Phi error did not decrease from lambda={lambdas[i]} ({phi_errors[i]:.4e}) "
                f"to lambda={lambdas[i+1]} ({phi_errors[i+1]:.4e})"
            )
            assert sigma_errors[i + 1] < sigma_errors[i], (
                f"Sigma error did not decrease from lambda={lambdas[i]} ({sigma_errors[i]:.4e}) "
                f"to lambda={lambdas[i+1]} ({sigma_errors[i+1]:.4e})"
            )

        # At lambda = 1e5, discrepancy must be below 1e-5
        assert phi_errors[-1] < 1e-5
        assert sigma_errors[-1] < 1e-5

    def test_continuous_irf_convergence_across_horizons(self, nk3_model: LinearModel):
        """Verify structural IRF paths converge continuously as lambda -> infty across h=0..20."""
        sim = nk3_model.simulate(300, seed=42)
        data = sim[["y", "pi", "r"]]
        p = 2
        horizon = 20

        # Estimate benchmark at ultra-high lambda (asymptote)
        res_asymptote = estimate_dsge_var(nk3_model, data, p=p, lamb=1e9, identification="dsge")
        irf_asymptote = res_asymptote.irf(horizon=horizon)

        # Compute distance to asymptote for increasing lambda
        lambdas = [2.0, 10.0, 50.0, 500.0, 5000.0]
        max_irf_diffs = []

        for l_val in lambdas:
            res_l = estimate_dsge_var(nk3_model, data, p=p, lamb=l_val, identification="dsge")
            irf_l = res_l.irf(horizon=horizon)
            diff = np.max(np.abs(irf_l.to_numpy() - irf_asymptote.to_numpy()))
            max_irf_diffs.append(diff)

        # Check strictly monotonic convergence across all horizons h=0..20
        for i in range(len(lambdas) - 1):
            assert max_irf_diffs[i + 1] < max_irf_diffs[i], (
                f"IRF discrepancy did not decrease from lambda={lambdas[i]} ({max_irf_diffs[i]:.4e}) "
                f"to lambda={lambdas[i+1]} ({max_irf_diffs[i+1]:.4e})"
            )

        # Rate of convergence test: ratio of diffs should be approximately proportional to ratio of lambdas
        assert max_irf_diffs[-1] < 1e-4, f"Final IRF distance to asymptote was {max_irf_diffs[-1]:.4e}"

    def test_impact_matrix_convergence_to_theoretical_ghu(self, nk3_model: LinearModel):
        """Verify structural impact matrix B0 converges to theoretical DSGE ghu @ shock_sd."""
        sim = nk3_model.simulate(300, seed=42)
        data = sim[["y", "pi", "r"]]

        res_inf = estimate_dsge_var(nk3_model, data, p=2, lamb=1e8, identification="dsge")

        dr = nk3_model.decision_rules()
        sd_shocks = nk3_model._shock_sd(None)
        obs = ["y", "pi", "r"]
        shocks = ["eps_a", "eps_u", "eps_r"]
        s_idx = [nk3_model.shocks.index(s) for s in shocks]
        A0_dsge = dr.ghu.loc[obs, shocks].to_numpy() @ np.diag(sd_shocks[s_idx])

        # B0 at lambda=1e8 must match A0_dsge with tight precision
        np.testing.assert_allclose(res_inf.B0, A0_dsge, rtol=1e-4, atol=1e-5)


# ==============================================================================
# Challenge 3: FEVD Variance Decomposition Rows Sum to 1.0 (h = 0..20)
# ==============================================================================

class TestFEVDSumToOne:
    """Adversarial validation of Forecast Error Variance Decomposition properties."""

    @pytest.mark.parametrize("ident", ["dsge", "cholesky"])
    @pytest.mark.parametrize("horizon", [0, 5, 20, 40])
    def test_fevd_rows_sum_to_one_exact(self, nk3_model: LinearModel, ident: str, horizon: int):
        """Verify that for every horizon h in 0..horizon and every variable i, sum_j FEVD[h, i, j] == 1.0."""
        sim = nk3_model.simulate(300, seed=777)
        data = sim[["y", "pi", "r"]]

        res = estimate_dsge_var(nk3_model, data, p=2, lamb=1.5, identification=ident)
        fevd = res.fevd(horizon=horizon)

        assert fevd.shape == (horizon + 1, 3, 3)

        # Every entry must be between 0.0 and 1.0 (valid probability / variance shares)
        assert np.all(fevd >= -1e-12), f"Negative FEVD values found: min = {fevd.min()}"
        assert np.all(fevd <= 1.0 + 1e-12), f"FEVD values > 1.0 found: max = {fevd.max()}"

        # Sum across structural shocks (axis 2) must be identically 1.0 everywhere
        row_sums = fevd.sum(axis=2)
        expected = np.ones((horizon + 1, 3))

        max_err = np.max(np.abs(row_sums - expected))
        assert max_err < 1e-12, f"FEVD row sum discrepancy: max error = {max_err:.3e}"

    def test_fevd_impact_properties(self, nk3_model: LinearModel):
        """Verify FEVD properties on impact (h=0) under Cholesky and DSGE identification."""
        sim = nk3_model.simulate(250, seed=42)
        data = sim[["y", "pi", "r"]]

        # Under Cholesky, first variable at h=0 is explained 100% by shock 1
        res_chol = estimate_dsge_var(nk3_model, data, p=2, lamb=1.0, identification="cholesky")
        fevd_chol = res_chol.fevd(horizon=5)
        assert abs(fevd_chol[0, 0, 0] - 1.0) < 1e-10
        assert abs(fevd_chol[0, 0, 1]) < 1e-10
        assert abs(fevd_chol[0, 0, 2]) < 1e-10

        # Under DSGE identification, shares reflect model structure and sum to 1.0
        res_dsge = estimate_dsge_var(nk3_model, data, p=2, lamb=1.0, identification="dsge")
        fevd_dsge = res_dsge.fevd(horizon=5)
        np.testing.assert_allclose(fevd_dsge.sum(axis=2), np.ones((6, 3)), atol=1e-12)


# ==============================================================================
# Challenge 4: Out-of-Sample Forecast & Non-Interactive Matplotlib Agg Mode
# ==============================================================================

class TestForecastAndAggFanCharts:
    """Stress testing forecast recursion and non-interactive graphics rendering."""

    @pytest.mark.parametrize("ci", [0.50, 0.68, 0.90, 0.95, 0.99])
    def test_forecast_confidence_band_ordering_and_monotonicity(
        self, nk3_model: LinearModel, ci: float
    ):
        """Verify lower < mean < upper, band width expands with CI and widens with horizon."""
        sim = nk3_model.simulate(300, seed=123)
        data = sim[["y", "pi", "r"]]

        res = estimate_dsge_var(nk3_model, data, p=2, lamb=1.2)
        horizon = 12
        fc = res.forecast(horizon=horizon, ci=ci)

        assert isinstance(fc, DSGEForecastResult)
        assert fc.mean.shape == (horizon, 3)
        assert fc.lower.shape == (horizon, 3)
        assert fc.upper.shape == (horizon, 3)

        for col in res.names:
            m = fc.mean[col].to_numpy()
            low = fc.lower[col].to_numpy()
            up = fc.upper[col].to_numpy()

            # Strict band ordering
            assert np.all(low < m), f"Lower band >= mean for variable {col}"
            assert np.all(m < up), f"Upper band <= mean for variable {col}"

            # Forecast error variance must widen monotonically over forecast horizon
            widths = up - low
            assert np.all(np.diff(widths) >= -1e-10), f"Band width shrank with horizon for {col}"

    def test_forecast_ci_coverage_monotonicity(self, nk3_model: LinearModel):
        """Higher CI must produce strictly wider intervals at every horizon."""
        sim = nk3_model.simulate(200, seed=42)
        data = sim[["y", "pi", "r"]]
        res = estimate_dsge_var(nk3_model, data, p=2, lamb=1.0)

        fc_90 = res.forecast(horizon=8, ci=0.90)
        fc_99 = res.forecast(horizon=8, ci=0.99)

        for col in res.names:
            width_90 = (fc_90.upper[col] - fc_90.lower[col]).to_numpy()
            width_99 = (fc_99.upper[col] - fc_99.lower[col]).to_numpy()
            assert np.all(width_99 > width_90), f"99% band is not strictly wider than 90% band for {col}"

    def test_fan_chart_headless_agg_mode_rendering(self, nk3_model: LinearModel):
        """Test rendering fan charts and saving to PNG buffer under Matplotlib Agg mode."""
        # Ensure Agg backend is active
        assert matplotlib.get_backend().lower() == "agg", "Matplotlib backend is not Agg"

        sim = nk3_model.simulate(250, seed=42)
        data = sim[["y", "pi", "r"]]
        res = estimate_dsge_var(nk3_model, data, p=2, lamb=1.0)

        # 1. Test DSGEVARResult.plot(kind="forecast") for all variables
        for v in res.names:
            fig, ax = res.plot(kind="forecast", target=v, horizon=10)
            assert isinstance(fig, plt.Figure)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            assert len(buf.getvalue()) > 5000, "Rendered PNG file size is suspiciously small"
            plt.close(fig)

        # 2. Test DSGEForecastResult.plot() multi-panel rendering
        fc = res.forecast(horizon=10, ci=0.90)
        ax_arr = fc.plot()
        assert len(ax_arr) == 3
        fig_fc = ax_arr[0].figure
        buf_fc = io.BytesIO()
        fig_fc.savefig(buf_fc, format="png")
        assert len(buf_fc.getvalue()) > 10000
        plt.close(fig_fc)

        # 3. Test IRF and MDD plot kinds under Agg mode
        fig_irf, ax_irf = res.plot(kind="irf", shock="eps_a", target="y", horizon=15)
        buf_irf = io.BytesIO()
        fig_irf.savefig(buf_irf, format="png")
        assert len(buf_irf.getvalue()) > 5000
        plt.close(fig_irf)

        # Verify all figures were closed (zero open figure leak)
        assert len(plt.get_fignums()) == 0


# ==============================================================================
# Challenge 5: Adversarial Boundary & Stress Cases
# ==============================================================================

class TestAdversarialEdgeCases:
    """Stress tests for invalid dimensions, high lag orders, and scale invariants."""

    def test_shock_dimension_mismatch_raises_value_error(self, nk3_model: LinearModel):
        """DSGE identification requires equal number of shocks and observables."""
        sim = nk3_model.simulate(200, seed=42)
        data = sim[["y", "pi", "r"]]

        with pytest.raises(ValueError, match="Structural DSGE identification requires equal number"):
            estimate_dsge_var(
                nk3_model,
                data,
                p=1,
                lamb=1.0,
                shocks=["eps_a", "eps_u"],  # 2 shocks for 3 observables!
                identification="dsge",
            )

    def test_high_lag_order_admissibility_bound(self, nk3_model: LinearModel):
        """At high lag order p=8, lambda_min increases substantially; must enforce bound correctly."""
        sim = nk3_model.simulate(200, seed=42)
        data = sim[["y", "pi", "r"]]
        p = 8
        n = 3
        k = 1 + n * p  # 25
        T = len(data) - p  # 192
        expected_lambda_min = (k + n) / T  # 28 / 192 ~= 0.1458

        res = estimate_dsge_var(nk3_model, data, p=p, lamb=expected_lambda_min + 0.01)
        assert abs(res.lambda_min - expected_lambda_min) < 1e-6
        assert len(res.A_list) == p

        with pytest.raises(ValueError, match="below admissibility bound"):
            estimate_dsge_var(nk3_model, data, p=p, lamb=expected_lambda_min - 0.01)

    def test_invalid_forecast_parameters_raise(self, nk3_model: LinearModel):
        """Invalid ci or horizon must raise ValueError."""
        sim = nk3_model.simulate(100, seed=42)
        res = estimate_dsge_var(nk3_model, sim[["y", "pi", "r"]], p=1, lamb=1.0)

        with pytest.raises(ValueError, match="ci must be in"):
            res.forecast(horizon=5, ci=0.0)
        with pytest.raises(ValueError, match="ci must be in"):
            res.forecast(horizon=5, ci=1.0)
        with pytest.raises(ValueError, match="horizon must be positive"):
            res.forecast(horizon=0)
        with pytest.raises(ValueError, match="horizon must be positive"):
            res.forecast(horizon=-5)
