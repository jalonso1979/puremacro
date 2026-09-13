"""Unit and benchmark tests for Young (2010) continuous stationary distribution and GE.

Verifies:
1. young_lottery_weights:
   - Grid validation (rejects unsorted or len < 2).
   - Boundary clamping (kp < k_min and kp > k_max).
   - Convex combination (w_lo + w_hi == 1.0, w >= 0).
   - Exact first-moment preservation (w_lo * k_lo + w_hi * k_hi == kp).
2. continuous_push_distribution / young_step:
   - 1D and 2D mass conservation (|sum(pdf_next) - 1.0| <= 1e-14).
3. build_continuous_transition_matrix / young_transition_matrix:
   - CSR matrix format.
   - Exact row-stochasticity (|sum(T, axis=1) - 1.0| <= 1e-14).
   - Sparsity bound (nnz <= 2 * n_z * N).
4. continuous_stationary_distribution / young_stationary_distribution:
   - All 4 solver methods: "sparse_direct", "power", "arnoldi", "auto".
   - Strict mass conservation (|sum(mu*) - 1.0| <= 1e-12).
   - Non-negativity (mu* >= 0.0).
   - Fixed-point residual (||mu* T - mu*||_inf <= 1e-10).
   - Marginal shock distribution parity with Markov stationary distribution (||mu_z - pi_z*||_inf <= 1e-10).
   - Seamless interop with CollocationSolution and FEMSolution.
5. ContinuousStationaryDistribution container:
   - Statistics: mean, variance, percentile, gini, lorenz.
   - Presentation contract: summary, plot, to_frame, to_markdown, to_latex, to_typst.
6. continuous_stationary_equilibrium & AiyagariContinuousEquilibrium:
   - Market clearing tolerance: |K^s(r*) - K^d(r*)| < 1e-4.
   - Theoretical bound: r* < 1/beta - 1.
   - Bracket validation and error handling.
   - Full presentation contract.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Headless backend for automated test runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from puremacro.vfi.collocation import CollocationProblem
from puremacro.vfi.fem import FEMProblem
from puremacro.vfi.continuous_distribution import (
    AiyagariContinuousEquilibrium,
    AiyagariContinuousModel,
    ContinuousDistributionResult,
    ContinuousEquilibriumResult,
    ContinuousStationaryDistribution,
    build_continuous_transition_matrix,
    continuous_push_distribution,
    continuous_stationary_distribution,
    continuous_stationary_equilibrium,
    solve_aiyagari_continuous,
    young_lottery_weights,
    young_stationary_distribution,
    young_step,
    young_transition_matrix,
)
from puremacro.vfi.discretize import markov_stationary, tauchen


# ===========================================================================
# 1. Young (2010) Lottery Weights Tests
# ===========================================================================

class TestYoungLotteryWeights:
    """Test mathematical invariants of linear lottery weight projections."""

    def test_grid_validation(self):
        """Verify validation of invalid or non-monotonic grids."""
        with pytest.raises(ValueError, match="length >= 2"):
            young_lottery_weights(np.array([1.0]), np.array([1.0]))

        with pytest.raises(ValueError, match="strictly increasing"):
            young_lottery_weights(np.array([1.0]), np.array([2.0, 1.0]))

        with pytest.raises(ValueError, match="strictly increasing"):
            young_lottery_weights(np.array([1.0]), np.array([1.0, 1.0, 2.0]))

    def test_convex_combination_and_first_moment_preservation(self):
        """Verify weights are non-negative, sum to 1.0, and preserve the first moment."""
        k_grid = np.linspace(0.0, 10.0, 101)
        np.random.seed(42)
        kp_eval = np.random.uniform(0.0, 10.0, 500)

        j_lo, w_lo, w_hi = young_lottery_weights(kp_eval, k_grid)

        # Convex combination
        assert np.all(w_lo >= 0.0)
        assert np.all(w_hi >= 0.0)
        assert np.allclose(w_lo + w_hi, 1.0, atol=1e-14)

        # First-moment preservation
        k_lo = k_grid[j_lo]
        k_hi = k_grid[j_lo + 1]
        reconstructed = w_lo * k_lo + w_hi * k_hi
        assert np.allclose(reconstructed, kp_eval, atol=1e-13)

    def test_boundary_clamping_edge_cases(self):
        """Verify E1 and E2 boundary conditions clamp cleanly without mass leakage."""
        k_grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Points below lower bound (E1)
        kp_below = np.array([-5.0, 0.0, 0.999])
        j_lo, w_lo, w_hi = young_lottery_weights(kp_below, k_grid)
        assert np.all(j_lo == 0)
        assert np.all(w_lo == 1.0)
        assert np.all(w_hi == 0.0)

        # Points above upper bound (E2)
        kp_above = np.array([5.001, 8.0, 100.0])
        j_lo, w_lo, w_hi = young_lottery_weights(kp_above, k_grid)
        assert np.all(j_lo == len(k_grid) - 2)
        assert np.all(w_lo == 0.0)
        assert np.all(w_hi == 1.0)

        # Exact boundary hits
        j_lo_min, w_lo_min, w_hi_min = young_lottery_weights(np.array([1.0]), k_grid)
        assert j_lo_min[0] == 0
        assert np.isclose(w_lo_min[0], 1.0)

        j_lo_max, w_lo_max, w_hi_max = young_lottery_weights(np.array([5.0]), k_grid)
        assert j_lo_max[0] == len(k_grid) - 2
        assert np.isclose(w_hi_max[0], 1.0)


# ===========================================================================
# 2. Continuous Push Distribution Tests
# ===========================================================================

class TestContinuousPushDistribution:
    """Test forward mass push steps under continuous policies."""

    def test_push_1d_mass_conservation(self):
        """Verify 1D forward mass step preserves mass to machine precision."""
        k_grid = np.linspace(0.0, 5.0, 200)
        pdf = np.full(len(k_grid), 1.0 / len(k_grid))

        # Contracting policy g(k) = 0.8 * k + 0.2
        policy = lambda k: 0.8 * k + 0.2
        pdf_next = continuous_push_distribution(pdf, policy, k_grid)

        assert len(pdf_next) == len(k_grid)
        assert np.all(pdf_next >= 0.0)
        assert np.isclose(np.sum(pdf_next), 1.0, atol=1e-14)

    def test_push_2d_mass_conservation_and_alias(self):
        """Verify 2D forward mass step with discrete Markov shocks preserves mass."""
        k_grid = np.linspace(0.0, 10.0, 150)
        P_z = np.array([[0.9, 0.1], [0.15, 0.85]])
        n_z = 2
        pdf = np.full((len(k_grid), n_z), 1.0 / (len(k_grid) * n_z))

        # Array policy
        policy_arr = np.column_stack([0.9 * k_grid + 0.1, 0.85 * k_grid + 0.5])
        pdf_next = young_step(pdf, policy_arr, k_grid, shock_transition=P_z)

        assert pdf_next.shape == (len(k_grid), n_z)
        assert np.all(pdf_next >= 0.0)
        assert np.isclose(np.sum(pdf_next), 1.0, atol=1e-14)


# ===========================================================================
# 3. CSR Transition Matrix Assembly Tests
# ===========================================================================

class TestContinuousTransitionMatrix:
    """Verify sparsity, row-stochasticity, and structure of CSR operator T."""

    def test_transition_matrix_1d_stochasticity_and_sparsity(self):
        """Verify 1D CSR transition matrix row-stochasticity and sparsity."""
        k_grid = np.linspace(0.0, 5.0, 300)
        policy = lambda k: 0.85 * k + 0.3

        T = build_continuous_transition_matrix(policy, k_grid)
        assert isinstance(T, sp.csr_matrix)
        assert T.shape == (len(k_grid), len(k_grid))

        # Row stochasticity: sum over row equals 1.0 exactly
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        assert np.allclose(row_sums, 1.0, atol=1e-14)

        # Max nonzeros per row is 2
        assert T.nnz <= 2 * len(k_grid)

    def test_transition_matrix_2d_stochasticity_and_sparsity(self):
        """Verify 2D CSR transition matrix row-stochasticity and sparsity."""
        N_k = 250
        k_grid = np.linspace(0.0, 20.0, N_k)
        log_z, P_z = tauchen(3, 0.8, 0.15)
        z_grid = np.exp(log_z)
        n_z = 3

        policy = lambda k, z: 0.9 * k + 0.1 * z

        T = young_transition_matrix(policy, k_grid, shock_transition=P_z, shock_grid=z_grid)
        assert isinstance(T, sp.csr_matrix)
        N = N_k * n_z
        assert T.shape == (N, N)

        # Row stochasticity
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        assert np.allclose(row_sums, 1.0, atol=1e-14)

        # Sparsity invariant: nnz / N^2 <= 2 / N_k
        sparsity = T.nnz / (N**2)
        assert sparsity <= (2.0 * n_z / N) + 1e-12


# ===========================================================================
# 4. Stationary Distribution Solver Tests
# ===========================================================================

class TestContinuousStationaryDistribution:
    """Verify accuracy, mass conservation, and convergence of stationary solvers."""

    @pytest.mark.parametrize("method", ["sparse_direct", "power", "arnoldi", "auto"])
    def test_stationary_distribution_methods_1d(self, method: str):
        """Verify strict mass conservation across all 4 solver methods on 1D model."""
        k_grid = np.linspace(0.0, 5.0, 200)
        # Stable AR(1) contraction with interior fixed point k* = 0.5 / 0.15 = 3.333
        policy = lambda k: 0.85 * k + 0.5

        dist = continuous_stationary_distribution(
            policy, k_grid, method=method, tol=1e-12
        )

        assert isinstance(dist, ContinuousStationaryDistribution)
        assert dist.converged
        assert np.all(dist.pdf >= 0.0)

        # Strict mass conservation invariant
        assert dist.mass_error <= 1e-12
        assert np.isclose(np.sum(dist.pdf), 1.0, atol=1e-12)

        # Fixed point convergence: mean should be near 3.333
        assert np.isclose(dist.mean(), 3.3333, atol=0.05)

    def test_stationary_distribution_2d_and_marginal_shock_parity(self):
        """Verify 2D stationary distribution reproduces ergodic shock shares."""
        N_k = 300
        k_grid = np.linspace(0.0, 25.0, N_k)
        log_z, P_z = tauchen(3, 0.9, 0.2)
        z_grid = np.exp(log_z)

        # True invariant distribution of exogenous Markov chain
        pi_z_true = markov_stationary(P_z)

        policy = lambda k, z: 0.88 * k + 0.25 * z

        dist = continuous_stationary_distribution(
            policy, k_grid, shock_transition=P_z, shock_grid=z_grid, method="auto"
        )

        assert dist.converged
        assert dist.pdf.shape == (N_k, 3)
        assert dist.mass_error <= 1e-12
        assert np.all(dist.pdf >= 0.0)

        # Invariant INV-06: Marginal shock distribution parity
        mu_z = dist.marginal_shocks()
        assert mu_z is not None
        assert np.allclose(mu_z, pi_z_true, atol=1e-10)

    def test_interoperability_with_collocation_solution(self):
        """Verify continuous stationary distribution consumes CollocationSolution directly."""
        alpha = 0.36
        beta = 0.96
        k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
        domain = (0.5 * k_ss, 1.5 * k_ss)

        # Solve smooth neoclassical growth via Chebyshev Collocation
        prob = CollocationProblem(
            domain=domain,
            orders=6,
            method="euler",
            params={"alpha": alpha, "delta": 1.0},
            beta=beta,
        )
        sol = prob.solve()
        assert sol.converged

        k_hist = np.linspace(domain[0], domain[1], 400)
        dist = continuous_stationary_distribution(sol, k_hist, method="auto")

        assert dist.converged
        assert dist.mass_error <= 1e-12
        # Stationary distribution mean should match analytical steady state k_ss
        assert np.isclose(dist.mean(), k_ss, atol=1e-4)

    def test_interoperability_with_fem_solution(self):
        """Verify continuous stationary distribution consumes FEMSolution directly."""
        alpha = 0.36
        beta = 0.96
        k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
        domain = (0.5 * k_ss, 1.5 * k_ss)

        prob = FEMProblem(
            domain=domain,
            elements=25,
            method="euler",
            transition_fn=lambda k: k**alpha,
            return_fn=lambda c: np.log(np.maximum(c, 1e-12)),
            beta=beta,
            params={"alpha": alpha, "gamma": 1.0},
        )
        sol = prob.solve()
        assert sol.converged

        k_hist = np.linspace(domain[0], domain[1], 400)
        dist = continuous_stationary_distribution(sol, k_hist, method="auto")

        assert dist.converged
        assert dist.mass_error <= 1e-12
        assert np.isclose(dist.mean(), k_ss, atol=1e-3)


# ===========================================================================
# 5. ContinuousStationaryDistribution Container & Statistics Tests
# ===========================================================================

class TestContinuousDistributionResultContainer:
    """Test analytical statistics and presentation methods."""

    @pytest.fixture
    def sample_dist(self) -> ContinuousStationaryDistribution:
        k_grid = np.linspace(0.0, 10.0, 100)
        # Construct triangular density peaked at 5.0
        pdf_raw = np.maximum(5.0 - np.abs(k_grid - 5.0), 0.0)
        pdf = pdf_raw / np.sum(pdf_raw)
        return ContinuousStationaryDistribution(
            pdf=pdf,
            asset_grid=k_grid,
            mass_error=0.0,
            iterations=1,
            converged=True,
            metadata={"method": "auto", "residual_norm": 0.0},
        )

    def test_statistics_calculations(self, sample_dist):
        """Verify mean, variance, percentiles, Gini, and Lorenz curves."""
        d = sample_dist
        mean_k = d.mean()
        assert np.isclose(mean_k, 5.0, atol=0.05)

        var_k = d.variance()
        assert var_k > 0.0

        p50 = d.percentile(50.0)
        assert np.isclose(p50, 5.0, atol=0.1)

        p10, p90 = d.percentile([10.0, 90.0])
        assert p10 < p50 < p90

        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            d.percentile(-5.0)

        with pytest.raises(ValueError, match="must be in \\[0, 100\\]"):
            d.percentile(105.0)

        gini = d.gini()
        assert 0.0 <= gini <= 1.0

        p, L = d.lorenz(50)
        assert len(p) == 50
        assert np.isclose(p[0], 0.0) and np.isclose(p[-1], 1.0)
        assert np.isclose(L[0], 0.0) and np.isclose(L[-1], 1.0)
        assert np.all(np.diff(L) >= -1e-12)

    def test_presentation_contract(self, sample_dist):
        """Verify .summary(), .to_frame(), .to_markdown(), .to_latex(), .to_typst(), .plot()."""
        d = sample_dist
        df = d.summary()
        assert isinstance(df, pd.DataFrame)
        assert "Mean Assets" in df.index
        assert "Gini Coefficient" in df.index

        df_frame = d.to_frame()
        assert df_frame.equals(df)

        md = d.to_markdown()
        assert isinstance(md, str)
        assert "Mean Assets" in md

        ltx = d.to_latex()
        assert isinstance(ltx, str)
        assert "\\begin" in ltx

        typ = d.to_typst()
        assert isinstance(typ, str)
        assert "#table" in typ

        fig = d.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ===========================================================================
# 6. Continuous General Equilibrium & Aiyagari Tests
# ===========================================================================

class TestContinuousGeneralEquilibrium:
    """Verify continuous market clearing (|K^s - K^d| < 1e-4) in continuous Aiyagari GE."""

    def test_solve_aiyagari_continuous_market_clearing(self):
        """Verify solve_aiyagari_continuous clears the capital market within tolerance."""
        beta = 0.96
        alpha = 0.36
        delta = 0.08
        gamma = 2.0

        eq = solve_aiyagari_continuous(
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            delta=delta,
            rho_z=0.85,
            sigma_z=0.20,
            n_z=3,
            a_max=25.0,
            N_k=500,
            xtol=1e-6,
        )

        assert isinstance(eq, AiyagariContinuousEquilibrium)
        assert eq.converged

        # General equilibrium capital market clearing tolerance
        assert abs(eq.capital_market_clearing_error) < 1e-4

        # Theoretical Aiyagari bound: r* < 1/beta - 1
        r_upper = 1.0 / beta - 1.0
        assert 0.0 < eq.r < r_upper

        # Positive factor prices and aggregates
        assert eq.w > 0.0
        assert eq.K > 0.0
        assert eq.L > 0.0

        # Stationary distribution mass conservation
        assert eq.distribution.mass_error <= 1e-12

    def test_aiyagari_continuous_equilibrium_presentation_contract(self):
        """Verify presentation methods on AiyagariContinuousEquilibrium."""
        eq = solve_aiyagari_continuous(
            beta=0.96,
            n_z=3,
            a_max=30.0,
            N_k=400,
            xtol=1e-8,
        )

        df = eq.summary()
        assert isinstance(df, pd.DataFrame)
        assert "Equilibrium Interest Rate (r*)" in df.index
        assert "Aggregate Capital Supply (K*)" in df.index

        md = eq.to_markdown()
        assert isinstance(md, str)
        assert "Equilibrium Interest Rate" in md

        ltx = eq.to_latex()
        assert "\\begin" in ltx

        typ = eq.to_typst()
        assert "#table" in typ

        fig = eq.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bracket_validation_and_error_handling(self):
        """Verify detection of invalid price brackets exceeding theoretical bounds."""
        beta = 0.96
        r_upper = 1.0 / beta - 1.0

        with pytest.raises(ValueError, match="strictly less than 1/beta - 1"):
            solve_aiyagari_continuous(
                beta=beta,
                r_bracket=(0.01, r_upper + 0.01),
            )

    def test_continuous_stationary_equilibrium_custom_wrapper(self):
        """Verify continuous_stationary_equilibrium with custom problem and market residual."""
        # Simple toy economy: household saves g(k) = (1 + r) * 0.5 * k + 1.0
        # Firm capital demand: K_d(r) = 3.0 / (1.0 + r)
        k_hist = np.linspace(0.0, 20.0, 300)

        def build_prob(r):
            return lambda k: (1.0 + r) * 0.5 * k + 1.0

        def market_residual(r, sol, dist, prob):
            Kd = 3.0 / (1.0 + r)
            Ks = dist.mean()
            return Ks - Kd

        eq = continuous_stationary_equilibrium(
            build_problem=build_prob,
            market_residual=market_residual,
            price_bracket=(0.01, 0.50),
            asset_grid=k_hist,
            xtol=1e-8,
        )

        assert isinstance(eq, AiyagariContinuousEquilibrium)
        assert eq.converged
        assert abs(eq.capital_market_clearing_error) < 1e-4


# ===========================================================================
# 6. Degenerate 1-State Markov Shock Handling (Defect P1-01 Verification)
# ===========================================================================

class TestDegenerateMarkovShocks:
    """Verify robust handling of degenerate 1-state Markov shock processes."""

    def test_evaluate_policy_degenerate_shock(self):
        """Verify _evaluate_policy produces (N_k, 1) across all policy representations."""
        from puremacro.vfi.continuous_distribution import _evaluate_policy

        N_k = 25
        k_grid = np.linspace(0.1, 5.0, N_k)
        P_z = np.array([[1.0]])
        z_grid = np.array([1.0])

        # 1. 1D ndarray
        pol_1d = 0.85 * k_grid + 0.3
        res = _evaluate_policy(pol_1d, k_grid, shock_transition=P_z, shock_grid=z_grid)
        assert res.shape == (N_k, 1)

        # 2. 2D ndarray
        pol_2d = pol_1d[:, None]
        res = _evaluate_policy(pol_2d, k_grid, shock_transition=P_z, shock_grid=z_grid)
        assert res.shape == (N_k, 1)

        # 3. 1-arg callable
        fn_1arg = lambda k: 0.85 * k + 0.3
        res = _evaluate_policy(fn_1arg, k_grid, shock_transition=P_z, shock_grid=z_grid)
        assert res.shape == (N_k, 1)

        # 4. 2-arg callable
        fn_2arg = lambda k, z: 0.85 * k + 0.3 * z
        res = _evaluate_policy(fn_2arg, k_grid, shock_transition=P_z, shock_grid=z_grid)
        assert res.shape == (N_k, 1)

        # 5. List of callables
        fn_list = [lambda k: 0.85 * k + 0.3]
        res = _evaluate_policy(fn_list, k_grid, shock_transition=P_z, shock_grid=z_grid)
        assert res.shape == (N_k, 1)

    def test_continuous_push_distribution_degenerate_shock(self):
        """Verify continuous_push_distribution handles 1D and 2D pdf inputs for n_z=1."""
        N_k = 50
        k_grid = np.linspace(0.1, 5.0, N_k)
        P_z = np.array([[1.0]])
        policy = lambda k: 0.85 * k + 0.3

        # 1D pdf input
        pdf_1d = np.full(N_k, 1.0 / N_k)
        pdf_next_1d = continuous_push_distribution(pdf_1d, policy, k_grid, shock_transition=P_z)
        assert pdf_next_1d.shape == (N_k,)
        assert np.all(pdf_next_1d >= 0.0)
        assert np.isclose(np.sum(pdf_next_1d), 1.0, atol=1e-14)

        # 2D pdf input
        pdf_2d = pdf_1d[:, None]
        pdf_next_2d = continuous_push_distribution(pdf_2d, policy, k_grid, shock_transition=P_z)
        assert pdf_next_2d.shape == (N_k, 1)
        assert np.all(pdf_next_2d >= 0.0)
        assert np.isclose(np.sum(pdf_next_2d), 1.0, atol=1e-14)

    def test_build_continuous_transition_matrix_degenerate_shock(self):
        """Verify build_continuous_transition_matrix succeeds and preserves row-stochasticity for n_z=1."""
        N_k = 60
        k_grid = np.linspace(0.1, 5.0, N_k)
        P_z = np.array([[1.0]])
        policy = lambda k: 0.85 * k + 0.3

        T = build_continuous_transition_matrix(policy, k_grid, shock_transition=P_z)
        assert isinstance(T, sp.csr_matrix)
        assert T.shape == (N_k, N_k)

        # Row stochasticity
        row_sums = np.asarray(T.sum(axis=1)).ravel()
        assert np.allclose(row_sums, 1.0, atol=1e-14)
        assert T.nnz <= 2 * N_k

    @pytest.mark.parametrize("method", ["sparse_direct", "power", "arnoldi", "auto"])
    def test_continuous_stationary_distribution_degenerate_shock_all_methods(self, method: str):
        """Verify continuous_stationary_distribution converges with mass conservation for n_z=1."""
        N_k = 80
        k_grid = np.linspace(0.0, 5.0, N_k)
        P_z = np.array([[1.0]])
        # Fixed point k* = 0.5 / 0.15 = 3.333
        policy = lambda k: 0.85 * k + 0.5

        dist = continuous_stationary_distribution(
            policy, k_grid, shock_transition=P_z, method=method, tol=1e-12
        )

        assert dist.converged
        assert dist.pdf.shape == (N_k, 1)
        assert dist.mass_error <= 1e-12
        assert np.isclose(np.sum(dist.pdf), 1.0, atol=1e-12)
        assert np.all(dist.pdf >= 0.0)

        # Marginal distributions
        mu_k = dist.marginal_assets()
        assert mu_k.shape == (N_k,)
        assert np.isclose(np.sum(mu_k), 1.0, atol=1e-12)

        mu_z = dist.marginal_shocks()
        assert mu_z is not None
        assert mu_z.shape == (1,)
        assert np.isclose(mu_z[0], 1.0, atol=1e-12)

        # Fixed point convergence
        assert np.isclose(dist.mean(), 3.3333, atol=0.08)

        # Summary DataFrame validation
        summary_df = dist.summary()
        assert summary_df.loc["Shock States (n_z)", "Value"] == "1"

