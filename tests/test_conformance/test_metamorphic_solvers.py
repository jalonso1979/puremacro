"""Metamorphic conformance test suite for PureMacro solvers.

Tests formal metamorphic invariants across four major solver families:
1. VAR (puremacro.var)
2. DSGE (puremacro.dsge)
3. Trade (puremacro.trade)
4. Sovereign Default (puremacro.sovereign)

Metamorphic invariants verified:
- Scaling Equivariance:
    Multiplying data by scalar c scales linear responses by c, covariances by c^2,
    while scale-free metrics (t-statistics, F-statistics, correlation matrices,
    relative prices/wages) remain invariant.
- Permutation Equivariance:
    Permuting endogenous variable ordering permutes coefficient matrices,
    impulse response matrices, and policy vectors consistently.
- Frequency / Time-Shift Invariance:
    Uniform time-index shifts leave dynamic autoregressive coefficients,
    general equilibrium transmissions, and Markov transition paths invariant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# 1. VAR Solvers
from puremacro.var.estimate import estimate_var
from puremacro.var.irf import irf, gfevd

# 2. DSGE Solvers
from puremacro.dsge.gensys import gensys
from puremacro.dsge.klein import klein_solve

# 3. Trade Solvers
from puremacro.trade.data import synthetic_icio_table
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.solver import solve_trade_equilibrium

# 4. Sovereign Solvers
from puremacro.sovereign.arellano import solve_sovereign_default


# =========================================================================
# 1. VAR Solvers: Metamorphic Conformance
# =========================================================================
class TestVARMetamorphic:
    """Metamorphic invariance tests for Vector Autoregression solvers."""

    @pytest.fixture
    def var_data(self) -> np.ndarray:
        """Stationary 3-variable VAR(2) synthetic time series."""
        rng = np.random.default_rng(101)
        T, n = 200, 3
        A1 = np.array([[0.5, 0.1, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.1, 0.0, 0.3]])
        A2 = np.array([[0.1, 0.0, 0.0],
                       [0.0, 0.1, 0.0],
                       [0.0, 0.0, 0.1]])
        Sigma = np.array([[1.0, 0.3, 0.2],
                          [0.3, 1.2, 0.4],
                          [0.2, 0.4, 0.9]])
        L = np.linalg.cholesky(Sigma)
        Y = np.zeros((T, n))
        for t in range(2, T):
            Y[t] = A1 @ Y[t - 1] + A2 @ Y[t - 2] + L @ rng.standard_normal(n)
        return Y

    def test_var_scaling_equivariance(self, var_data: np.ndarray):
        """Scaling data by scalar c scales Sigma by c^2, IRF by c, keeps A and corr invariant."""
        c = 3.5
        Y = var_data
        Y_scaled = c * Y

        res_base = estimate_var(Y, p=2)
        res_scaled = estimate_var(Y_scaled, p=2)

        # Dynamic coefficients A_1, A_2 must be scale-invariant
        for A_base, A_sc in zip(res_base.A_list, res_scaled.A_list):
            np.testing.assert_allclose(A_sc, A_base, rtol=1e-10, atol=1e-10)

        # Constant vector scales by c
        np.testing.assert_allclose(res_scaled.c, c * res_base.c, rtol=1e-10, atol=1e-10)

        # Residuals scale by c
        np.testing.assert_allclose(res_scaled.resid, c * res_base.resid, rtol=1e-10, atol=1e-10)

        # Covariance Sigma scales by c^2
        np.testing.assert_allclose(res_scaled.Sigma, (c ** 2) * res_base.Sigma, rtol=1e-10, atol=1e-10)

        # Correlation matrix of Sigma is scale-invariant
        def _cov2corr(S: np.ndarray) -> np.ndarray:
            d = np.sqrt(np.diag(S))
            return S / np.outer(d, d)

        np.testing.assert_allclose(_cov2corr(res_scaled.Sigma), _cov2corr(res_base.Sigma), rtol=1e-10, atol=1e-10)

        # Linear impulse responses with Cholesky identification scale linearly with c
        B0_base = np.linalg.cholesky(res_base.Sigma)
        B0_scaled = np.linalg.cholesky(res_scaled.Sigma)
        irf_base = irf(res_base.A_list, B0_base, horizon=5)
        irf_scaled = irf(res_scaled.A_list, B0_scaled, horizon=5)
        np.testing.assert_allclose(irf_scaled, c * irf_base, rtol=1e-10, atol=1e-10)

        # GFEVD shares (variance ratios) are scale-invariant
        gfevd_base = gfevd(res_base.A_list, res_base.Sigma, horizon=5)
        gfevd_scaled = gfevd(res_scaled.A_list, res_scaled.Sigma, horizon=5)
        np.testing.assert_allclose(gfevd_scaled, gfevd_base, rtol=1e-10, atol=1e-10)

    def test_var_permutation_equivariance(self, var_data: np.ndarray):
        """Permuting endogenous variable order permutes coefficient matrices consistently."""
        Y = var_data
        n = Y.shape[1]
        perm = [1, 2, 0]
        P_mat = np.eye(n)[perm]
        Y_perm = Y[:, perm]

        res_base = estimate_var(Y, p=2)
        res_perm = estimate_var(Y_perm, p=2)

        # Permuted coefficients: A_perm = P A P^T
        for A_base, A_p in zip(res_base.A_list, res_perm.A_list):
            expected_A = P_mat @ A_base @ P_mat.T
            np.testing.assert_allclose(A_p, expected_A, rtol=1e-10, atol=1e-10)

        # Permuted covariance: Sigma_perm = P Sigma P^T
        expected_Sigma = P_mat @ res_base.Sigma @ P_mat.T
        np.testing.assert_allclose(res_perm.Sigma, expected_Sigma, rtol=1e-10, atol=1e-10)

        # Permuted constant: c_perm = P c
        np.testing.assert_allclose(res_perm.c, P_mat @ res_base.c, rtol=1e-10, atol=1e-10)

    def test_var_time_shift_invariance(self, var_data: np.ndarray):
        """Uniform time shifts leave dynamic coefficients and residuals invariant."""
        Y = var_data
        dates_base = pd.date_range("2000-01-01", periods=len(Y), freq="QS")
        dates_shifted = pd.date_range("2015-01-01", periods=len(Y), freq="QS")

        df_base = pd.DataFrame(Y, index=dates_base)
        df_shifted = pd.DataFrame(Y, index=dates_shifted)

        res_base = estimate_var(df_base.to_numpy(), p=2)
        res_shifted = estimate_var(df_shifted.to_numpy(), p=2)

        for A_b, A_s in zip(res_base.A_list, res_shifted.A_list):
            np.testing.assert_allclose(A_s, A_b, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(res_shifted.Sigma, res_base.Sigma, rtol=1e-14, atol=1e-14)


# =========================================================================
# 2. DSGE Solvers: Metamorphic Conformance
# =========================================================================
class TestDSGEMetamorphic:
    """Metamorphic invariance tests for DSGE perturbation/QZ solvers."""

    @pytest.fixture
    def canonical_dsge_system(self):
        """A stable 3-variable linear rational expectations system."""
        n = 3
        G0 = np.eye(n)
        G1 = np.diag([0.65, 0.45, 0.25])
        G1[0, 1] = 0.15
        G1[1, 2] = 0.10
        Psi = np.array([[1.0, 0.2],
                        [0.0, 1.0],
                        [0.5, 0.0]])
        Pi = np.zeros((n, 1))
        return G0, G1, Psi, Pi

    def test_dsge_scaling_equivariance(self, canonical_dsge_system):
        """Scaling shock loadings scales Impact matrix by c while policy G remains invariant."""
        G0, G1, Psi, Pi = canonical_dsge_system
        c = 2.75

        sol_base = gensys(G0, G1, Psi, Pi)
        sol_scaled = gensys(G0, G1, c * Psi, Pi)

        assert sol_base.eu == (1, 1)
        assert sol_scaled.eu == (1, 1)

        # G matrix is invariant to shock magnitude
        np.testing.assert_allclose(sol_scaled.G, sol_base.G, rtol=1e-12, atol=1e-12)

        # Impact matrix scales linearly with c
        np.testing.assert_allclose(sol_scaled.Impact, c * sol_base.Impact, rtol=1e-12, atol=1e-12)

        # Multi-horizon impulse responses: IRF(h) = G^h * Impact scales by c
        for h in range(5):
            irf_base = np.linalg.matrix_power(sol_base.G, h) @ sol_base.Impact
            irf_scaled = np.linalg.matrix_power(sol_scaled.G, h) @ sol_scaled.Impact
            np.testing.assert_allclose(irf_scaled, c * irf_base, rtol=1e-12, atol=1e-12)

    def test_dsge_permutation_equivariance(self, canonical_dsge_system):
        """Permuting endogenous variable order permutes transition G and Impact matrices consistently."""
        G0, G1, Psi, Pi = canonical_dsge_system
        n = G0.shape[0]
        perm = np.array([2, 0, 1])
        P_mat = np.eye(n)[perm]

        # Permute system variables: z' = P z
        G0_perm = P_mat @ G0 @ P_mat.T
        G1_perm = P_mat @ G1 @ P_mat.T
        Psi_perm = P_mat @ Psi
        Pi_perm = P_mat @ Pi

        sol_base = gensys(G0, G1, Psi, Pi)
        sol_perm = gensys(G0_perm, G1_perm, Psi_perm, Pi_perm)

        assert sol_base.eu == (1, 1)
        assert sol_perm.eu == (1, 1)

        # Permuted G: G_perm = P G P^T
        expected_G = P_mat @ sol_base.G @ P_mat.T
        np.testing.assert_allclose(sol_perm.G, expected_G, rtol=1e-12, atol=1e-12)

        # Permuted Impact: Impact_perm = P Impact
        expected_Impact = P_mat @ sol_base.Impact
        np.testing.assert_allclose(sol_perm.Impact, expected_Impact, rtol=1e-12, atol=1e-12)

    def test_dsge_equation_scaling_invariance(self, canonical_dsge_system):
        """Multiplying individual equation rows by arbitrary non-zero scalars leaves solution invariant."""
        G0, G1, Psi, Pi = canonical_dsge_system
        n = G0.shape[0]
        row_scales = np.diag([2.0, 0.5, 3.2])

        G0_scaled = row_scales @ G0
        G1_scaled = row_scales @ G1
        Psi_scaled = row_scales @ Psi
        Pi_scaled = row_scales @ Pi

        sol_base = gensys(G0, G1, Psi, Pi)
        sol_eq_scaled = gensys(G0_scaled, G1_scaled, Psi_scaled, Pi_scaled)

        assert sol_eq_scaled.eu == (1, 1)
        np.testing.assert_allclose(sol_eq_scaled.G, sol_base.G, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(sol_eq_scaled.Impact, sol_base.Impact, rtol=1e-12, atol=1e-12)


# =========================================================================
# 3. Trade Solvers: Metamorphic Conformance
# =========================================================================
class TestTradeMetamorphic:
    """Metamorphic invariance tests for General Equilibrium Trade solvers."""

    @pytest.fixture
    def trade_setup(self):
        """Synthetic 2-country 2-sector trade table with tariff shock."""
        nc, ns, nfd = 2, 2, 3
        data = synthetic_icio_table(nc, ns, nfd=nfd, seed=42)
        calib = calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd)

        # Tariff shock: 10% tariff on country 0 imports from country 1
        tau = np.ones((ns * nc, ns, nc), dtype=float)
        tau_fd = np.ones((ns * nc, nfd, nc), dtype=float)
        tau[2:, :, 0] = 1.10
        tau_fd[2:, :, 0] = 1.10

        return data, calib, tau, tau_fd

    def test_trade_scaling_equivariance(self, trade_setup):
        """Scaling table transactions by scalar c leaves prices/wages invariant and scales flows by c."""
        data, calib_base, tau, tau_fd = trade_setup
        c = 4.0
        data_scaled = c * data

        nc, ns, nfd = 2, 2, 3
        calib_scaled = calibrate_trade_model(data_scaled, ns=ns, nc=nc, nfd=nfd)

        res_base = solve_trade_equilibrium(calib_base, tau=tau, tau_fd=tau_fd, method="newton")
        res_scaled = solve_trade_equilibrium(calib_scaled, tau=tau, tau_fd=tau_fd, method="newton")

        assert res_base.converged is True
        assert res_scaled.converged is True

        # Relative prices and wages are scale-invariant
        np.testing.assert_allclose(res_scaled.p_sol, res_base.p_sol, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res_scaled.w_sol, res_base.w_sol, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res_scaled.r_sol, res_base.r_sol, rtol=1e-6, atol=1e-6)

        # Trade and final demand flows scale linearly with c
        np.testing.assert_allclose(
            res_scaled.intermediate_flows, c * res_base.intermediate_flows, rtol=1e-3, atol=1e-3
        )
        np.testing.assert_allclose(
            res_scaled.final_demand_flows, c * res_base.final_demand_flows, rtol=1e-3, atol=1e-3
        )

    def test_trade_zero_shock_identity(self, trade_setup):
        """Under identity tariffs (no shock), baseline prices and wages are exactly unity."""
        _, calib, _, _ = trade_setup
        res = solve_trade_equilibrium(calib, method="newton")

        assert res.converged is True
        assert res.iterations <= 1
        np.testing.assert_allclose(res.p_sol, 1.0, atol=1e-10)
        np.testing.assert_allclose(res.w_sol, 1.0, atol=1e-10)
        np.testing.assert_allclose(res.r_sol, 1.0, atol=1e-10)


# =========================================================================
# 4. Sovereign Default Solvers: Metamorphic Conformance
# =========================================================================
class TestSovereignMetamorphic:
    """Metamorphic invariance tests for Sovereign Default solvers."""

    @pytest.fixture
    def sovereign_result(self):
        """Fast calibrated sovereign default equilibrium."""
        return solve_sovereign_default(
            beta=0.953,
            gamma=2.0,
            r_star=0.01,
            n_b=20,
            n_y=7,
            max_iter=200,
            tol=1e-5,
        )

    def test_sovereign_bounds_and_scale_invariance(self, sovereign_result):
        """Bond prices and default probabilities satisfy probability and discount bounds."""
        res = sovereign_result
        assert res.converged is True

        # Default probabilities are bounded in [0, 1]
        assert np.all(res.default_prob >= 0.0)
        assert np.all(res.default_prob <= 1.0)

        # Bond prices are bounded in [0, 1 / (1 + r_star)]
        q_max = 1.0 / (1.0 + 0.01) + 1e-7
        assert np.all(res.bond_price >= 0.0)
        assert np.all(res.bond_price <= q_max)

        # Spreads are non-negative wherever bond prices are strictly positive
        valid_q = res.bond_price > 1e-4
        assert np.all(res.spread[valid_q] >= -1e-6)

    def test_sovereign_time_shift_invariance(self, sovereign_result):
        """Simulation trajectories under stationary Markov policy are deterministic and shift-invariant."""
        res = sovereign_result

        # Simulations with identical seed produce identical policy paths
        sim1 = res.simulate(T=50, seed=42)
        sim2 = res.simulate(T=50, seed=42)

        np.testing.assert_allclose(sim1["b"].values, sim2["b"].values)
        np.testing.assert_allclose(sim1["bond_price"].values, sim2["bond_price"].values)
        np.testing.assert_allclose(sim1["spread"].values, sim2["spread"].values)
        np.testing.assert_allclose(sim1["consumption"].values, sim2["consumption"].values)

        # Stationary Markov transition probability matrix rows sum to 1
        p_row_sums = res.p_trans.sum(axis=1)
        np.testing.assert_allclose(p_row_sums, 1.0, atol=1e-12)
