"""Comprehensive test suite for DSGE 3rd-order perturbation and pruning.

Covers Requirement R1 (Andreasen, Fernández-Villaverde, and Rubio-Ramírez 2018):
- Symbolic 3rd-order dynamic derivatives with 6th-order numerical finite difference validation
- 3-fold Sylvester solver via Kronecker-Schur back-substitution
- Andreasen et al. (2018) 3rd-order pruned state-space simulation, stability, and GIRFs
- Presentation contract (.summary(), .to_markdown(), .to_latex(), .to_typst())
- Dynare integration and LinearModel solve(order=3)
"""

import time
import pytest
import numpy as np
import pandas as pd
import scipy.linalg

from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge._symbolic import (
    compile_derivatives,
    CompiledDerivatives,
    SparseDynamicTensor3D,
)
from puremacro.dsge._sylvester import (
    solve_order3_sylvester_kronecker,
    solve_order1_sylvester,
)
from puremacro.dsge.pruning import (
    Order3PrunedSolution,
    Order3TheoreticalMomentsResult,
    PrunedSimulationResult,
    canonical_growth_3rd_order,
)
from puremacro.dsge.dynare import (
    build_dynare,
    load_mod,
    solve_dynare_3rd_order,
)
from puremacro.dsge.build import LinearModel


# ============================================================================
# 1. TestSymbolicThirdOrderDerivatives
# ============================================================================

class TestSymbolicThirdOrderDerivatives:
    """Verification of analytical 3rd derivatives, permutation symmetry, and sparse tensor operations."""

    def test_euler_equation_3rd_derivative_high_precision_6th_order_stencil(self):
        """Test 3rd analytical derivatives match 6th-order central difference stencil to <= 1e-9."""
        src = """
        var c k;
        varexo e;
        parameters beta alpha delta gamma;
        beta = 0.99; delta = 0.025; alpha = 0.33; gamma = 2.0;
        model;
        c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * k(+1)^(alpha - 1.0) + 1.0 - delta);
        k = k(-1)^alpha - c + (1.0 - delta) * k(-1) + e;
        end;
        """
        dag = parse_mod_to_dag(src)
        compiled = compile_derivatives(dag)
        assert compiled.has_third_order

        lead = np.array([1.2, 10.5])
        curr = np.array([1.1, 10.2])
        lag = np.array([1.0, 10.0])
        shocks = np.array([0.0])
        pvec = np.array([dag.parameter_values[p] for p in dag.parameters])

        sparse_3rd = compiled.eval_third_order(lead, curr, lag, shocks, pvec)
        assert isinstance(sparse_3rd, SparseDynamicTensor3D)

        dense_3rd = sparse_3rd.to_dense()

        # Equation 0 is Euler equation. Lead c is index 0 in [lead, curr, lag, shocks].
        # Analytical 3rd derivative w.r.t lead c (0, 0, 0, 0):
        analytical_val = dense_3rd[0, 0, 0, 0]

        def eval_Hf_00(c_lead_val):
            lp = lead.copy()
            lp[0] = c_lead_val
            Hf = compiled.eval_second_order(lp, curr, lag, shocks, pvec)
            return Hf[0, 0, 0]

        # 6th-order central difference stencil for 1st derivative of H_f:
        # f'(x) = (f(x+3h) - 9f(x+2h) + 45f(x+h) - 45f(x-h) + 9f(x-2h) - f(x-3h)) / (60h)
        h = 1e-3
        x0 = lead[0]
        num_6th = (
            eval_Hf_00(x0 + 3 * h)
            - 9.0 * eval_Hf_00(x0 + 2 * h)
            + 45.0 * eval_Hf_00(x0 + h)
            - 45.0 * eval_Hf_00(x0 - h)
            + 9.0 * eval_Hf_00(x0 - 2 * h)
            - eval_Hf_00(x0 - 3 * h)
        ) / (60.0 * h)

        diff = abs(analytical_val - num_6th)
        assert diff <= 1e-9, f"Analytical {analytical_val} vs numerical {num_6th}, diff={diff} > 1e-9"

    def test_permutation_symmetry_and_sparse_contraction(self):
        """Verify permutation symmetry p <= q <= r and sparse 3D contraction with matrices."""
        n_eq = 2
        dim = 4
        entries = {
            (0, 0, 1, 2): 2.5,
            (0, 1, 1, 1): 1.8,
            (1, 2, 2, 3): -3.2,
        }
        tensor = SparseDynamicTensor3D(entries=entries, shape=(n_eq, dim, dim, dim))

        # Check dense representation preserves symmetry
        dense = tensor.to_dense()
        assert dense.shape == (2, 4, 4, 4)
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        for p, q, r in perms:
            assert np.isclose(dense[0, p, q, r], 2.5)

        # Test contraction: C_{i, a, b, c} = sum_{p, q, r} T_{i, p, q, r} M1_{p, a} M2_{q, b} M3_{r, c}
        np.random.seed(10)
        M1 = np.random.randn(dim, 3)
        M2 = np.random.randn(dim, 2)
        M3 = np.random.randn(dim, 2)

        contracted_sparse = tensor.contract(M1, M2, M3)
        assert contracted_sparse.shape == (n_eq, 3 * 2 * 2)

        # Contract dense using np.einsum for comparison
        contracted_dense = np.einsum("ipqr,pa,qb,rc->iabc", dense, M1, M2, M3).reshape(n_eq, -1)
        np.testing.assert_allclose(contracted_sparse, contracted_dense, rtol=1e-12, atol=1e-12)

    def test_sparse_tensor_memory_guard(self):
        """Verify to_dense memory guard raises error when dense size exceeds limit."""
        entries = {(0, 0, 0, 0): 1.0}
        huge_tensor = SparseDynamicTensor3D(entries=entries, shape=(10, 200, 200, 200))
        with pytest.raises((MemoryError, ValueError)):
            huge_tensor.to_dense(max_elements=100_000)


# ============================================================================
# 2. TestOrder3SylvesterSolvers
# ============================================================================

class TestOrder3SylvesterSolvers:
    """Verification of 3-fold Sylvester solver and order-1 Sylvester solver for risk correction."""

    def test_order3_sylvester_residual_accuracy(self):
        """Residual ||A_hat X + A_plus X (h_x kron h_x kron h_x) + K_xxx|| <= 1e-12."""
        np.random.seed(123)
        n_vars = 4
        n_x = 2

        H = np.random.randn(n_x, n_x) * 0.4
        h_x = scipy.linalg.schur(H, output="complex")[0] * 0.7
        h_x = np.real(h_x)

        A_hat = np.eye(n_vars) + np.random.randn(n_vars, n_vars) * 0.1
        A_plus = np.random.randn(n_vars, n_vars) * 0.2
        K_xxx = np.random.randn(n_vars, n_x**3)

        X = solve_order3_sylvester_kronecker(A_hat, A_plus, h_x, K_xxx)
        assert X.shape == (n_vars, n_x**3)

        hx3 = np.kron(h_x, np.kron(h_x, h_x))
        residual = A_hat @ X + A_plus @ X @ hx3 + K_xxx
        res_norm = np.linalg.norm(residual, ord="fro")
        assert res_norm < 1e-12, f"Order-3 Sylvester residual {res_norm} exceeded 1e-12"

    def test_order1_sylvester_residual_accuracy(self):
        """Residual ||A_hat X + A_plus X h_x + K_x_ss|| <= 1e-12."""
        np.random.seed(456)
        n_vars = 5
        n_x = 3

        h_x = np.array([[0.8, 0.1, 0.0], [0.0, 0.7, 0.05], [0.0, 0.0, 0.6]])
        A_hat = np.eye(n_vars) + np.random.randn(n_vars, n_vars) * 0.05
        A_plus = np.random.randn(n_vars, n_vars) * 0.1
        K_x_ss = np.random.randn(n_vars, n_x)

        X = solve_order1_sylvester(A_hat, A_plus, h_x, K_x_ss)
        assert X.shape == (n_vars, n_x)

        residual = A_hat @ X + A_plus @ X @ h_x + K_x_ss
        res_norm = np.linalg.norm(residual, ord="fro")
        assert res_norm < 1e-12, f"Order-1 Sylvester residual {res_norm} exceeded 1e-12"

    def test_degenerate_empty_state_space(self):
        """Verify solvers handle static model with n_x = 0 returning (N, 0)."""
        N = 3
        A_hat = np.eye(N)
        A_plus = np.zeros((N, N))
        h_x = np.zeros((0, 0))
        K_xxx = np.zeros((N, 0))
        K_x_ss = np.zeros((N, 0))

        X3 = solve_order3_sylvester_kronecker(A_hat, A_plus, h_x, K_xxx)
        assert X3.shape == (N, 0)

        X1 = solve_order1_sylvester(A_hat, A_plus, h_x, K_x_ss)
        assert X1.shape == (N, 0)

    def test_sw07_dimension_benchmark_performance_and_memory(self):
        """SW07-sized system (N=14, n_x=7) solves in <= 0.8s with < 5MB memory."""
        import tracemalloc

        N = 14
        n_x = 7

        np.random.seed(789)
        A_hat = np.eye(N) + np.random.randn(N, N) * 0.02
        A_plus = np.random.randn(N, N) * 0.05
        h_x = np.diag([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6])
        K_xxx = np.random.randn(N, n_x**3)

        tracemalloc.start()
        t0 = time.perf_counter()

        X = solve_order3_sylvester_kronecker(A_hat, A_plus, h_x, K_xxx)

        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert elapsed <= 0.8, f"Solve time {elapsed:.4f}s exceeded 0.8s limit"
        assert peak_mb < 5.0, f"Peak memory {peak_mb:.2f} MB exceeded 5.0 MB limit"


# ============================================================================
# 3. TestOrder3PruningSimulation
# ============================================================================

class TestOrder3PruningSimulation:
    """Verification of Andreasen et al. (2018) 3rd-order pruning simulation and moments."""

    @pytest.fixture
    def canonical_growth_model(self):
        """Instantiate canonical growth model with 3rd-order pruning."""
        return canonical_growth_3rd_order()

    def test_andreasen_state_decomposition_identity(self, canonical_growth_model):
        """Verify x_t = x_t^{(1)} + x_t^{(2)} + x_t^{(3)} holds at every time step."""
        sol = canonical_growth_model
        sim = sol.simulate(periods=50, seed=42, burn=10)

        assert isinstance(sim, PrunedSimulationResult)
        assert sim.states_3rd is not None
        assert sim.controls_3rd is not None

        total_states = sim.states.to_numpy()
        sum_components = (sim.states_1st + sim.states_2nd + sim.states_3rd).to_numpy()
        np.testing.assert_allclose(total_states, sum_components, rtol=1e-14, atol=1e-14)

        total_controls = sim.controls.to_numpy()
        sum_ctrl_components = (sim.controls_1st + sim.controls_2nd + sim.controls_3rd).to_numpy()
        np.testing.assert_allclose(total_controls, sum_ctrl_components, rtol=1e-14, atol=1e-14)

    def test_10k_simulation_boundedness_and_finite_kurtosis(self, canonical_growth_model):
        """Verify 10,000-period simulation executes with 0 explosions and finite stationary kurtosis."""
        sol = canonical_growth_model
        sim = sol.simulate(periods=10_000, seed=123, burn=200)

        assert len(sim) == 10_000
        x_vals = sim["k"].to_numpy()

        assert not np.any(np.isnan(x_vals))
        assert not np.any(np.isinf(x_vals))

        var_k = float(np.var(x_vals))
        assert var_k > 0.0

        kurt_k = float(np.mean((x_vals - np.mean(x_vals))**4) / (var_k**2))
        assert 1.0 < kurt_k < 50.0

    def test_girf_sign_and_scale_asymmetry(self, canonical_growth_model):
        """Verify Generalized Impulse Response Functions (GIRF) exhibit sign and scale asymmetry."""
        sol = canonical_growth_model

        girf_pos1 = sol.girf(shock_name="eps", shock_size=1.0, periods=20, n_draws=100, seed=7)
        girf_neg1 = sol.girf(shock_name="eps", shock_size=-1.0, periods=20, n_draws=100, seed=7)
        girf_pos2 = sol.girf(shock_name="eps", shock_size=2.0, periods=20, n_draws=100, seed=7)

        c_pos1 = girf_pos1["c"].to_numpy()
        c_neg1 = girf_neg1["c"].to_numpy()
        c_pos2 = girf_pos2["c"].to_numpy()

        sum_asym = np.max(np.abs(c_pos1 + c_neg1))
        assert sum_asym > 1e-6, "Order-3 GIRF failed to show sign asymmetry"

        scale_asym = np.max(np.abs(c_pos2 - 2.0 * c_pos1))
        assert scale_asym > 1e-6, "Order-3 GIRF failed to show scale asymmetry"

    def test_presentation_contract(self, canonical_growth_model):
        """Verify presentation methods: .summary(), .to_markdown(), .to_latex(), .to_typst()."""
        sol = canonical_growth_model

        summary_txt = sol.summary()
        assert "Order-3 Pruned DSGE Solution (Andreasen et al. 2018)" in summary_txt
        assert "Predetermined States" in summary_txt
        assert "Control Variables" in summary_txt

        md_txt = sol.to_markdown()
        assert "| SteadyState |" in md_txt
        assert "| index |" in md_txt

        latex_txt = sol.to_latex()
        assert "begin{tabular}" in latex_txt
        assert "end{tabular}" in latex_txt

        typst_txt = sol.to_typst()
        assert "#table(" in typst_txt

    def test_theoretical_moments_and_ergodic_moments(self, canonical_growth_model):
        """Verify theoretical and ergodic moments return valid containers with mean, var, skew, kurt."""
        sol = canonical_growth_model

        moments = sol.theoretical_moments()
        assert isinstance(moments, Order3TheoreticalMomentsResult)
        assert "k" in moments.mean
        assert "c" in moments.variance
        assert "k" in moments.skewness
        assert "c" in moments.kurtosis

        dr = sol.decision_rules()
        assert hasattr(dr, "ghxxx")
        assert isinstance(dr.to_frame(), pd.DataFrame)
        assert "k" in dr.to_frame().index
        assert "c" in dr.to_frame().index

    def test_canonical_growth_3rd_order_function(self):
        """Verify canonical_growth_3rd_order convenience factory function returns Order3PrunedSolution."""
        sol = canonical_growth_3rd_order()
        assert isinstance(sol, Order3PrunedSolution)
        assert sol.is_stable
        assert sol.spectral_radius < 1.0


# ============================================================================
# 4. TestDynareOrder3Integration
# ============================================================================

class TestDynareOrder3Integration:
    """Verification of end-to-end integration via load_mod, build_dynare, and LinearModel.solve(order=3)."""

    def test_load_mod_order3_execution(self):
        """Verify load_mod with order=3 produces Order3PrunedSolution."""
        mod_code = """
        var c k z;
        varexo eps;
        parameters beta alpha delta rho sigma_pref sigma_eps;
        beta = 0.99;
        alpha = 0.33;
        delta = 0.025;
        rho = 0.95;
        sigma_pref = 1.0;
        sigma_eps = 0.01;

        model;
        exp(-sigma_pref * c) - beta * exp(-sigma_pref * c(+1)) * (alpha * exp(z(+1)) * exp((alpha - 1.0) * k) + 1.0 - delta);
        exp(c) + exp(k) - exp(z) * exp(alpha * k(-1)) - (1.0 - delta) * exp(k(-1));
        z - rho * z(-1) - sigma_eps * eps;
        end;

        initval;
        k = 3.8;
        c = 0.8;
        z = 0.0;
        end;
        """

        sol = load_mod(mod_code, order=3)
        assert isinstance(sol, Order3PrunedSolution)
        assert sol.g_xxx is not None
        assert sol.g_x_ss is not None

        sim = sol.simulate(periods=100, seed=99)
        assert len(sim) == 100
        assert not sim.states.isna().any().any()

    def test_build_dynare_python_callable_order3(self):
        """Verify build_dynare with equations callable and order=3 produces Order3PrunedSolution."""
        def eqs(lead, curr, lag, shocks, p):
            return [
                np.exp(-p.sigma_pref * curr.c)
                - p.beta * np.exp(-p.sigma_pref * lead.c)
                * (p.alpha * np.exp(lead.z) * np.exp((p.alpha - 1.0) * curr.k) + 1.0 - p.delta),
                np.exp(curr.c) + np.exp(curr.k)
                - np.exp(curr.z) * np.exp(p.alpha * lag.k) - (1.0 - p.delta) * np.exp(lag.k),
                curr.z - p.rho * lag.z - p.sigma_eps * shocks.eps,
            ]

        params = {"beta": 0.99, "alpha": 0.33, "delta": 0.025, "rho": 0.95, "sigma_pref": 1.0, "sigma_eps": 0.01}
        guess = {"c": 0.8, "k": 3.8, "z": 0.0}

        sol = build_dynare(
            eqs,
            variables=["c", "k", "z"],
            shocks=["eps"],
            params=params,
            guess=guess,
            states=["k", "z"],
            order=3,
        )
        assert isinstance(sol, Order3PrunedSolution)
        assert sol.H_xxx is not None

    def test_linear_model_solve_order3_and_stoch_simul(self):
        """Verify LinearModel.solve(order=3) and LinearModel.stoch_simul(order=3)."""
        mod_code = """
        var c k z;
        varexo eps;
        parameters beta alpha delta rho sigma_pref sigma_eps;
        beta = 0.99;
        alpha = 0.33;
        delta = 0.025;
        rho = 0.95;
        sigma_pref = 1.0;
        sigma_eps = 0.01;

        model;
        exp(-sigma_pref * c) - beta * exp(-sigma_pref * c(+1)) * (alpha * exp(z(+1)) * exp((alpha - 1.0) * k) + 1.0 - delta);
        exp(c) + exp(k) - exp(z) * exp(alpha * k(-1)) - (1.0 - delta) * exp(k(-1));
        z - rho * z(-1) - sigma_eps * eps;
        end;

        initval;
        k = 3.8;
        c = 0.8;
        z = 0.0;
        end;
        """
        model = load_mod(mod_code, order=1)
        assert isinstance(model, LinearModel)

        sol = model.solve(order=3)
        assert isinstance(sol, Order3PrunedSolution)

        sol_alias = model.solve_third_order()
        assert isinstance(sol_alias, Order3PrunedSolution)

        res = model.stoch_simul(order=3, periods=100, seed=1)
        assert res.order == 3
        assert hasattr(res.dr, "ghxxx")
        assert res.simulated_moments is not None
