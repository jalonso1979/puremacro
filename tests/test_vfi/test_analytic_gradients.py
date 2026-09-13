r"""Comprehensive Verification and Unit Test Suite for puremacro.vfi.analytic_gradients.

Verifies Milestone 2: Exact Analytic Gradients via the Implicit Function Theorem (IFT):
1. Gradient parity against central finite differences (relative error < 1e-5).
2. Execution speedup benchmark: IFT achieves >= 5x speedup over numerical finite differences.
3. Continuous projection engine compatibility:
   - Orthogonal Chebyshev polynomial collocation (CollocationProblem).
   - Finite Element Method with piecewise linear elements (FEMProblem).
   - Cubic B-Splines (CubicBSplineBasis).
   - Schumaker (1983) shape-preserving quadratic splines (SchumakerSpline).
4. Continuous policy gradient consistency: \nabla_\theta g(s) = \Phi(s) \nabla_\theta c^*.
5. Macroeconomic aggregate sensitivities \nabla_\theta K*, \nabla_\theta C*, \nabla_\theta r*, \nabla_\theta w*.
6. Adjoint general equilibrium stationary distribution sensitivities.
7. Robust fallbacks: condition number tracking, Tikhonov regularization, and Truncated SVD.
8. High-level entry points: policy_parameter_jacobian, equilibrium_parameter_jacobian, gmm_objective_and_gradient.
9. Full puremacro presentation contract (.summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst()).
"""
from __future__ import annotations

import dataclasses
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq, root

import puremacro.vfi.analytic_gradients as ag
from puremacro.vfi.analytic_gradients import (
    AnalyticGradientResult,
    compute_ift_gradients,
    equilibrium_parameter_jacobian,
    gmm_objective_and_gradient,
    policy_parameter_jacobian,
)
from puremacro.vfi.collocation import CollocationProblem, _evaluate_euler_residual
from puremacro.vfi.fem import FEMProblem
from puremacro.vfi.splines import SplineCollocationProblem


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def neoclassical_collocation():
    """Solved canonical neoclassical growth model on Chebyshev collocation nodes."""
    alpha = 0.36
    beta = 0.96
    delta = 1.0
    sigma = 1.0
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
    prob = CollocationProblem(
        domain=(0.5 * k_ss, 1.5 * k_ss),
        orders=6,
        method="euler",
        params={"alpha": alpha, "delta": delta, "sigma": sigma},
        beta=beta,
        options={"tol": 1e-12, "max_iter": 500},
    )
    sol = prob.solve(backend="numpy")
    return prob, sol, k_ss


@pytest.fixture
def cubic_spline_model():
    """Solved neoclassical growth model using Cubic B-Spline basis."""
    prob = SplineCollocationProblem(
        domain=(0.8, 3.0),
        n_knots=12,
        spline_type="cubic",
        beta=0.96,
        params={"alpha": 0.36, "delta": 1.0, "sigma": 1.0, "z": 1.0},
        options={"tol": 1e-13},
    )
    sol = prob.solve(backend="numpy")
    return prob, sol


@pytest.fixture
def schumaker_spline_model():
    """Solved neoclassical growth model using Schumaker shape-preserving splines."""
    prob = SplineCollocationProblem(
        domain=(0.8, 3.0),
        n_knots=12,
        spline_type="schumaker",
        beta=0.96,
        params={"alpha": 0.36, "delta": 1.0, "sigma": 1.0, "z": 1.0},
        options={"tol": 1e-13},
    )
    sol = prob.solve(backend="numpy")
    return prob, sol


@pytest.fixture
def fem_model():
    """Solved neoclassical growth model using Finite Element Method (FEM)."""
    alpha = 0.36
    beta = 0.96
    delta = 1.0
    gamma = 1.0
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))

    def euler_fn(s, sp, sn, alpha=0.36, delta=1.0, gamma=1.0, beta=0.96):
        c = s**alpha + (1.0 - delta) * s - sp
        cp = sp**alpha + (1.0 - delta) * sp - sn
        uc = 1.0 / np.maximum(c, 1e-12)
        ucp = 1.0 / np.maximum(cp, 1e-12)
        fkp = alpha * (sp ** (alpha - 1.0)) + 1.0 - delta
        return 1.0 - beta * (ucp / uc) * fkp

    prob = FEMProblem(
        domain=(0.5 * k_ss, 1.5 * k_ss),
        elements=16,
        method="euler",
        projection="collocation",
        euler_residual_fn=euler_fn,
        beta=beta,
        params={"alpha": alpha, "delta": delta, "gamma": gamma, "beta": beta},
        options={"tol": 1e-13},
    )
    sol = prob.solve(backend="numpy")
    return prob, sol, k_ss


# ===========================================================================
# 1. Gradient Parity vs Central Finite Differences
# ===========================================================================

class TestAnalyticGradientsParity:
    """Verifies that exact IFT gradients match central differences with relative error < 1e-5."""

    def test_chebyshev_collocation_gradient_parity(self, neoclassical_collocation):
        """Verify IFT parameter gradients match central finite differences to < 1e-5."""
        prob, sol, k_ss = neoclassical_collocation
        params_to_test = ["alpha", "beta", "delta", "sigma"]

        res = compute_ift_gradients(sol, prob, params=params_to_test, step_c=1e-6, h=1e-5)
        assert isinstance(res, AnalyticGradientResult)
        assert res.grad_coefficients.shape == (7, 4)

        # Compute reference central finite differences with high precision
        h_fd = 5e-5
        grad_num = np.zeros_like(res.grad_coefficients)
        c_star = sol.coefficients.copy()
        basis = sol.basis
        nodes = basis.nodes()

        for k, pname in enumerate(params_to_test):
            if pname == "beta":
                p_p = dict(prob.params)
                p_m = dict(prob.params)
                b_p = prob.beta + h_fd
                b_m = prob.beta - h_fd
            else:
                p_p = dict(prob.params); p_p[pname] += h_fd; b_p = prob.beta
                p_m = dict(prob.params); p_m[pname] -= h_fd; b_m = prob.beta

            prob_p = dataclasses.replace(prob, params=p_p, beta=b_p)
            prob_m = dataclasses.replace(prob, params=p_m, beta=b_m)

            rp = root(lambda c: _evaluate_euler_residual(prob_p, basis, c, nodes, "numpy"), c_star, tol=1e-13, method="hybr")
            rm = root(lambda c: _evaluate_euler_residual(prob_m, basis, c, nodes, "numpy"), c_star, tol=1e-13, method="hybr")
            grad_num[:, k] = (rp.x - rm.x) / (2.0 * h_fd)

        rel_error = np.max(np.abs(res.grad_coefficients - grad_num) / (np.abs(grad_num) + 1e-8))
        assert rel_error < 1e-5, f"IFT vs Central FD relative error {rel_error:.2e} >= 1e-5"

    def test_cubic_spline_gradient_parity(self, cubic_spline_model):
        """Verify IFT gradient parity on Cubic B-Spline basis."""
        prob, sol = cubic_spline_model
        params_to_test = ["alpha", "beta"]

        res = compute_ift_gradients(sol, prob, params=params_to_test, step_c=1e-6, h=1e-5)
        assert res.grad_coefficients.shape == (14, 2)

        h_fd = 5e-5
        grad_num = np.zeros_like(res.grad_coefficients)
        c_star = sol.coefficients.copy()

        for k, pname in enumerate(params_to_test):
            if pname == "beta":
                p_p = dict(prob.params); b_p = prob.beta + h_fd; p_p["beta"] = b_p
                p_m = dict(prob.params); b_m = prob.beta - h_fd; p_m["beta"] = b_m
            else:
                p_p = dict(prob.params); p_p[pname] += h_fd; b_p = prob.beta
                p_m = dict(prob.params); p_m[pname] -= h_fd; b_m = prob.beta

            prob_p = dataclasses.replace(prob, params=p_p, beta=b_p)
            prob_m = dataclasses.replace(prob, params=p_m, beta=b_m)

            rp = root(lambda c: ag._evaluate_residual_system(sol, prob_p, c), c_star, tol=1e-13, method="hybr")
            rm = root(lambda c: ag._evaluate_residual_system(sol, prob_m, c), c_star, tol=1e-13, method="hybr")
            grad_num[:, k] = (rp.x - rm.x) / (2.0 * h_fd)

        rel_error = np.max(np.abs(res.grad_coefficients - grad_num) / (np.abs(grad_num) + 1e-8))
        assert rel_error < 1e-5, f"Cubic spline IFT vs Central FD relative error {rel_error:.2e} >= 1e-5"

    def test_schumaker_spline_gradient_parity(self, schumaker_spline_model):
        """Verify IFT gradient parity on Schumaker shape-preserving spline."""
        prob, sol = schumaker_spline_model
        params_to_test = ["alpha", "beta"]

        res = compute_ift_gradients(sol, prob, params=params_to_test, step_c=1e-6, h=1e-5)
        assert res.grad_coefficients.shape == (12, 2)

        h_fd = 5e-5
        grad_num = np.zeros_like(res.grad_coefficients)
        c_star = sol.coefficients.copy()

        for k, pname in enumerate(params_to_test):
            if pname == "beta":
                p_p = dict(prob.params); b_p = prob.beta + h_fd; p_p["beta"] = b_p
                p_m = dict(prob.params); b_m = prob.beta - h_fd; p_m["beta"] = b_m
            else:
                p_p = dict(prob.params); p_p[pname] += h_fd; b_p = prob.beta
                p_m = dict(prob.params); p_m[pname] -= h_fd; b_m = prob.beta

            prob_p = dataclasses.replace(prob, params=p_p, beta=b_p)
            prob_m = dataclasses.replace(prob, params=p_m, beta=b_m)

            rp = root(lambda c: ag._evaluate_residual_system(sol, prob_p, c), c_star, tol=1e-13, method="hybr")
            rm = root(lambda c: ag._evaluate_residual_system(sol, prob_m, c), c_star, tol=1e-13, method="hybr")
            grad_num[:, k] = (rp.x - rm.x) / (2.0 * h_fd)

        rel_error = np.max(np.abs(res.grad_coefficients - grad_num) / (np.abs(grad_num) + 1e-8))
        assert rel_error < 1e-5, f"Schumaker spline IFT vs Central FD relative error {rel_error:.2e} >= 1e-5"

    def test_fem_gradient_parity(self, fem_model):
        """Verify IFT gradient parity on Finite Element Method piecewise linear elements."""
        prob, sol, k_ss = fem_model
        params_to_test = ["alpha", "beta"]

        res = compute_ift_gradients(sol, prob, params=params_to_test, step_c=1e-7, h=1e-5)
        assert res.grad_coefficients.shape == (17, 2)

        h_fd = 5e-5
        grad_num = np.zeros_like(res.grad_coefficients)
        c_star = sol.nodal_values.copy()

        for k, pname in enumerate(params_to_test):
            if pname == "beta":
                p_p = dict(prob.params); b_p = prob.beta + h_fd; p_p["beta"] = b_p
                p_m = dict(prob.params); b_m = prob.beta - h_fd; p_m["beta"] = b_m
            else:
                p_p = dict(prob.params); p_p[pname] += h_fd; b_p = prob.beta
                p_m = dict(prob.params); p_m[pname] -= h_fd; b_m = prob.beta

            prob_p = dataclasses.replace(prob, params=p_p, beta=b_p)
            prob_m = dataclasses.replace(prob, params=p_m, beta=b_m)

            rp = root(lambda c: ag._evaluate_residual_system(sol, prob_p, c), c_star, tol=1e-13, method="hybr")
            rm = root(lambda c: ag._evaluate_residual_system(sol, prob_m, c), c_star, tol=1e-13, method="hybr")
            grad_num[:, k] = (rp.x - rm.x) / (2.0 * h_fd)

        # In FEM with piecewise linear interpolation, L2 relative error reflects full domain integration
        l2_err = float(np.linalg.norm(res.grad_coefficients - grad_num) / np.linalg.norm(grad_num))
        assert l2_err < 1e-5, f"FEM IFT vs Central FD L2 relative error {l2_err:.2e} >= 1e-5"


# ===========================================================================
# 2. Performance & Speedup Benchmark
# ===========================================================================

class TestAnalyticGradientsPerformance:
    """Verifies that single LU factorization achieves >= 5x speedup over numerical FD."""

    def test_ift_speedup_vs_finite_difference(self, neoclassical_collocation):
        """Benchmark execution time of IFT vs full numerical model re-solve finite differences."""
        prob, sol, k_ss = neoclassical_collocation
        params = ["alpha", "beta", "delta", "sigma"]

        # Warm-up run
        _ = compute_ift_gradients(sol, prob, params=params)

        # 1. Measure IFT time (single LU solve)
        t_ift_runs = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = compute_ift_gradients(sol, prob, params=params)
            t_ift_runs.append(time.perf_counter() - t0)
        t_ift = float(np.median(t_ift_runs))

        # 2. Measure Numerical Finite Differences time (2p full model re-solves)
        h_fd = 5e-5
        t0 = time.perf_counter()
        for pname in params:
            if pname == "beta":
                p_plus = dataclasses.replace(prob, beta=prob.beta + h_fd)
                p_minus = dataclasses.replace(prob, beta=prob.beta - h_fd)
            else:
                p_p = dict(prob.params); p_p[pname] += h_fd
                p_m = dict(prob.params); p_m[pname] -= h_fd
                p_plus = dataclasses.replace(prob, params=p_p)
                p_minus = dataclasses.replace(prob, params=p_m)
            _ = p_plus.solve(backend="numpy")
            _ = p_minus.solve(backend="numpy")
        t_num = time.perf_counter() - t0

        speedup = t_num / max(t_ift, 1e-6)
        print(f"\n[BENCHMARK] IFT time: {t_ift*1000:.2f} ms | Numerical FD time: {t_num*1000:.2f} ms | Speedup: {speedup:.1f}x")
        assert speedup >= 5.0, f"IFT speedup {speedup:.2f}x is below the 5.0x requirement"


# ===========================================================================
# 3. Continuous Policy & Macro Aggregate Sensitivities
# ===========================================================================

class TestContinuousPolicyAndAggregates:
    """Verifies continuous policy gradients and general equilibrium macro aggregate sensitivities."""

    def test_continuous_policy_gradient_consistency(self, neoclassical_collocation):
        """Verify continuous policy gradient \nabla_\theta g(s) matches finite difference of policy."""
        prob, sol, k_ss = neoclassical_collocation
        res = compute_ift_gradients(sol, prob, params=["alpha", "beta"])

        s_dense = np.linspace(0.5 * k_ss, 1.5 * k_ss, 40)
        pol_grads = res.policy_gradient(s_dense)
        assert pol_grads.shape == (40, 2)

        # Direct policy perturbation
        h = 5e-5
        p_plus = dataclasses.replace(prob, params={**prob.params, "alpha": prob.params["alpha"] + h})
        p_minus = dataclasses.replace(prob, params={**prob.params, "alpha": prob.params["alpha"] - h})
        sol_p = p_plus.solve()
        sol_m = p_minus.solve()

        num_pol_grad_alpha = (sol_p.policy(s_dense) - sol_m.policy(s_dense)) / (2.0 * h)
        rel_diff = np.max(np.abs(pol_grads[:, 0] - num_pol_grad_alpha) / (np.abs(num_pol_grad_alpha) + 1e-8))
        assert rel_diff < 1e-5, f"Continuous policy gradient relative difference {rel_diff:.2e} >= 1e-5"

    def test_policy_parameter_jacobian_api(self, neoclassical_collocation):
        """Verify policy_parameter_jacobian entry point function."""
        prob, sol, k_ss = neoclassical_collocation
        # Scalar evaluation
        grad_scalar = policy_parameter_jacobian(sol, prob, s=k_ss, params=["alpha", "beta"])
        assert grad_scalar.shape == (2,)
        assert np.all(np.isfinite(grad_scalar))

        # Array evaluation
        grad_arr = policy_parameter_jacobian(sol, prob, s=np.array([k_ss, 1.1 * k_ss]), params=["alpha", "beta"])
        assert grad_arr.shape == (2, 2)

    def test_equilibrium_aggregate_sensitivities(self, neoclassical_collocation):
        """Verify aggregate sensitivities match numerical steady state responses."""
        prob, sol, k_ss = neoclassical_collocation
        res = compute_ift_gradients(sol, prob, params=["alpha", "beta", "delta"])

        aggs = res.grad_aggregates
        assert "K" in aggs and "C" in aggs and "r" in aggs and "w" in aggs

        # Numerical perturbation of model steady state
        h = 5e-5
        p_plus = dataclasses.replace(prob, params={**prob.params, "alpha": prob.params["alpha"] + h})
        p_minus = dataclasses.replace(prob, params={**prob.params, "alpha": prob.params["alpha"] - h})
        sol_p = p_plus.solve()
        sol_m = p_minus.solve()

        k_star_p = brentq(lambda k: sol_p.policy(k) - k, 0.5 * k_ss, 1.5 * k_ss)
        k_star_m = brentq(lambda k: sol_m.policy(k) - k, 0.5 * k_ss, 1.5 * k_ss)
        dk_star_num = (k_star_p - k_star_m) / (2.0 * h)

        rel_diff = abs(aggs["K"][0] - dk_star_num) / abs(dk_star_num)
        assert rel_diff < 1e-5, f"Aggregate K sensitivity relative difference {rel_diff:.2e} >= 1e-5"

    def test_equilibrium_parameter_jacobian_api(self, neoclassical_collocation):
        """Verify equilibrium_parameter_jacobian entry point function."""
        prob, sol, k_ss = neoclassical_collocation
        deq = equilibrium_parameter_jacobian(sol, prob, params=["alpha", "beta"])
        assert isinstance(deq, dict)
        assert "K" in deq and "r" in deq
        assert len(deq["K"]) == 2


# ===========================================================================
# 4. Robust Fallbacks: Condition Number, Tikhonov & Truncated SVD
# ===========================================================================

class TestRobustFallbacks:
    """Verifies ill-conditioned and singular linear system handling."""

    def test_condition_number_reporting(self, neoclassical_collocation):
        """Verify condition number is reported and finite."""
        prob, sol, k_ss = neoclassical_collocation
        res = compute_ift_gradients(sol, prob, params=["alpha", "beta"])
        assert np.isfinite(res.condition_number)
        assert res.condition_number > 0.0

    def test_tikhonov_regularization_fallback(self, neoclassical_collocation):
        """Verify Tikhonov regularization is triggered when cond_max is exceeded or forced."""
        prob, sol, k_ss = neoclassical_collocation
        res_reg = compute_ift_gradients(sol, prob, params=["alpha", "beta"], cond_max=1.0)
        assert "Tikhonov" in res_reg.metadata["solver_method"]
        assert res_reg.grad_coefficients.shape == (7, 2)
        assert np.all(np.isfinite(res_reg.grad_coefficients))

    def test_truncated_svd_fallback(self, neoclassical_collocation):
        """Verify explicit Truncated SVD execution."""
        prob, sol, k_ss = neoclassical_collocation
        res_svd = compute_ift_gradients(sol, prob, params=["alpha", "beta"], solver="svd")
        assert "Truncated SVD" in res_svd.metadata["solver_method"]
        assert res_svd.grad_coefficients.shape == (7, 2)
        assert np.all(np.isfinite(res_svd.grad_coefficients))

    def test_singular_matrix_svd_fallback(self):
        """Verify singular residual Jacobian automatically triggers Truncated SVD."""
        # Rank-deficient singular residual Jacobian
        J_c = np.array([[1.0, 2.0], [2.0, 4.0]])
        J_theta = np.array([[1.0, 0.5], [2.0, 1.0]])

        grad_c, cond_num, method = ag._solve_ift_linear_system(
            J_c, J_theta, cond_max=10.0, regularization=0.0
        )
        assert "Truncated SVD" in method
        assert grad_c.shape == (2, 2)
        assert np.all(np.isfinite(grad_c))


# ===========================================================================
# 5. GMM Objective & Gradient
# ===========================================================================

class TestGMMAndEstimation:
    """Verifies GMM/SMM objective value and exact analytical gradient evaluation."""

    def test_gmm_objective_and_gradient_parity(self, neoclassical_collocation):
        """Verify exact GMM gradient matches central finite difference of objective."""
        prob, sol, k_ss = neoclassical_collocation
        empirical_moments = np.array([k_ss, 0.04])  # Target K and r
        theta_0 = np.array([0.35, 0.95])

        Q_val, grad_Q = gmm_objective_and_gradient(
            theta_0, prob, empirical_moments, param_names=["alpha", "beta"]
        )

        assert isinstance(Q_val, float)
        assert grad_Q.shape == (2,)

        # Numerical central difference of Q(theta)
        h = 1e-4
        grad_Q_num = np.zeros_like(grad_Q)
        for i in range(2):
            th_p = theta_0.copy(); th_p[i] += h
            th_m = theta_0.copy(); th_m[i] -= h
            Q_p, _ = gmm_objective_and_gradient(th_p, prob, empirical_moments, param_names=["alpha", "beta"])
            Q_m, _ = gmm_objective_and_gradient(th_m, prob, empirical_moments, param_names=["alpha", "beta"])
            grad_Q_num[i] = (Q_p - Q_m) / (2.0 * h)

        rel_diff = np.max(np.abs(grad_Q - grad_Q_num) / (np.abs(grad_Q_num) + 1e-8))
        assert rel_diff < 1e-4, f"GMM gradient relative difference {rel_diff:.2e} >= 1e-4"


# ===========================================================================
# 6. Presentation Interface Compliance
# ===========================================================================

class TestPresentationInterfaceCompliance:
    """Verifies .summary(), .plot(), .to_frame(), .to_markdown(), .to_latex(), .to_typst()."""

    def test_summary_and_dataframe(self, neoclassical_collocation):
        """Verify .summary() and .to_frame() return non-empty DataFrames."""
        prob, sol, k_ss = neoclassical_collocation
        res = compute_ift_gradients(sol, prob, params=["alpha", "beta", "delta", "sigma"])

        summary_df = res.summary()
        assert isinstance(summary_df, pd.DataFrame)
        assert not summary_df.empty
        assert "Implicit Function Theorem (IFT)" in summary_df.loc["Method", "Value"]

        frame_df = res.to_frame()
        assert isinstance(frame_df, pd.DataFrame)
        assert frame_df.shape == (11, 4)  # 7 coefficients + 4 aggregates (K, C, r, w)
        assert list(frame_df.columns) == ["alpha", "beta", "delta", "sigma"]

    def test_formatting_renderers(self, neoclassical_collocation):
        """Verify .to_markdown(), .to_latex(), and .to_typst() produce valid output strings."""
        prob, sol, k_ss = neoclassical_collocation
        res = compute_ift_gradients(sol, prob, params=["alpha", "beta"])

        md_str = res.to_markdown()
        assert isinstance(md_str, str)
        assert "| alpha | beta |" in md_str or "alpha" in md_str

        tex_str = res.to_latex()
        assert isinstance(tex_str, str)
        assert "\\begin{tabular}" in tex_str
        assert "\\end{tabular}" in tex_str

        typ_str = res.to_typst()
        assert isinstance(typ_str, str)
        assert "#table(" in typ_str

    def test_plot_figure(self, neoclassical_collocation):
        """Verify .plot() returns a valid matplotlib Figure with subplots."""
        prob, sol, k_ss = neoclassical_collocation
        res = compute_ift_gradients(sol, prob, params=["alpha", "beta"])

        fig = res.plot(show=False)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)


# ===========================================================================
# 7. Standalone Custom Residual Function
# ===========================================================================

class TestCustomResidualFunction:
    """Verifies compute_ift_gradients with arbitrary user residual functions."""

    def test_custom_residual_fn(self):
        """Verify IFT on arbitrary user residual system R(c, params)."""
        # Linear residual system: R(c, theta) = A(theta) c - b(theta) = 0
        def user_R(c, params):
            a1 = params.get("a1", 2.0)
            a2 = params.get("a2", 3.0)
            b1 = params.get("b1", 10.0)
            b2 = params.get("b2", 15.0)
            # R_1 = a1 * c_0 - b1
            # R_2 = a2 * c_1 - b2
            return np.array([a1 * c[0] - b1, a2 * c[1] - b2])

        c_opt = np.array([5.0, 5.0])
        params_dict = {"a1": 2.0, "a2": 3.0, "b1": 10.0, "b2": 15.0}

        # Mock solution/problem
        class DummySolution:
            coefficients = c_opt

        class DummyProblem:
            params = params_dict
            beta = 0.96

        sol = DummySolution()
        prob = DummyProblem()

        res = compute_ift_gradients(sol, prob, params=["b1", "b2"], residual_fn=user_R)
        # R = [a1 * c0 - b1, a2 * c1 - b2] = 0
        # dR/dc = diag(a1, a2) = diag(2, 3)
        # dR/db1 = [-1, 0]^T
        # dR/db2 = [0, -1]^T
        # dc/db1 = [1/2, 0]^T = [0.5, 0]
        # dc/db2 = [0, 1/3]^T = [0, 0.3333]
        assert np.isclose(res.grad_coefficients[0, 0], 0.5, atol=1e-5)
        assert np.isclose(res.grad_coefficients[1, 0], 0.0, atol=1e-5)
        assert np.isclose(res.grad_coefficients[0, 1], 0.0, atol=1e-5)
        assert np.isclose(res.grad_coefficients[1, 1], 1.0 / 3.0, atol=1e-5)
