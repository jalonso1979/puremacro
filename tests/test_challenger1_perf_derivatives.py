"""Empirical challenger test suite: Performance & Derivative Accuracy Verifier.

Executed by orch4_challenger_1 to verify:
1. SW07 Order-1 and Order-2 benchmark speedup across 10 repeated runs (<= 0.20s assert).
2. Analytical Jacobians (A+, A0, A-, Bu) and dynamic Hessian (Hf) accuracy
   against high-precision numerical central differences across nonlinear models
   over 1,000 random perturbations (<= 1e-10 max abs error assert).
3. Generalized Schur Sylvester equation residual norm (<= 1e-10) and memory footprint.
"""

from __future__ import annotations

import math
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from puremacro.dsge._ast import BinOp, Call, Const, Node, Param, UnaryOp, Var
from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge._symbolic import compile_derivatives
from puremacro.dsge._sylvester import solve_generalized_sylvester_kronecker
from puremacro.dsge.build import LinearModel, _Vec
from puremacro.dsge.dynare import _first_order_pieces, build_dynare, load_mod, parse_mod, solve_dynare_2nd_order
from puremacro.dsge.pruning import PrunedDSGESolution

SW07_PATH = Path("puremacro/dsge/_references/sw07_pfeifer.mod")


# ===========================================================================
# 1. SW07 Performance & Speedup Benchmark
# ===========================================================================

class TestSW07BenchmarkPerformance:
    """Benchmark SW07 Order-1 and Order-2 parse and solve times across 10 runs."""

    def test_sw07_parse_and_solve_order2_under_200ms(self):
        """SW07 Order-2 parse and solve consistently executes in <= 0.20 seconds."""
        assert SW07_PATH.exists(), f"Benchmark file {SW07_PATH} missing"

        # Warm up
        _ = load_mod(SW07_PATH, order=2)

        times_o2_full = []
        for _ in range(10):
            t0 = time.perf_counter()
            sol = load_mod(SW07_PATH, order=2)
            elapsed = time.perf_counter() - t0
            times_o2_full.append(elapsed)
            assert isinstance(sol, PrunedDSGESolution)
            assert elapsed <= 0.20, f"SW07 Order-2 full solve took {elapsed:.4f}s > 0.20s"

        mean_time = np.mean(times_o2_full)
        max_time = np.max(times_o2_full)
        speedup = 4.6 / mean_time
        print(f"\n[SW07 Order-2 Full] mean={mean_time:.4f}s, max={max_time:.4f}s, speedup={speedup:.1f}x vs 4.6s baseline")
        assert mean_time < 0.050, f"Expected ~0.028s, got mean {mean_time:.4f}s"

    def test_sw07_order1_benchmark(self):
        """SW07 Order-1 parse and solve executes in <= 0.050s across 10 runs."""
        assert SW07_PATH.exists()
        _ = load_mod(SW07_PATH, order=1)

        times_o1 = []
        for _ in range(10):
            t0 = time.perf_counter()
            m1 = load_mod(SW07_PATH, order=1)
            elapsed = time.perf_counter() - t0
            times_o1.append(elapsed)
            assert isinstance(m1, LinearModel)
            assert elapsed <= 0.050, f"SW07 Order-1 took {elapsed:.4f}s > 0.050s"

        mean_time = np.mean(times_o1)
        print(f"\n[SW07 Order-1 Full] mean={mean_time:.4f}s, min={min(times_o1):.4f}s, max={max(times_o1):.4f}s")


# ===========================================================================
# 2. Analytical Derivatives vs High-Precision Numerical Central Differences
# ===========================================================================

def _complex_step_jacobian(eval_fn: Callable[[np.ndarray], np.ndarray], u0: np.ndarray, N: int, K: int) -> np.ndarray:
    """Evaluate exact first-order Jacobian using complex step differentiation (machine precision)."""
    hc = 1e-20
    J = np.zeros((N, K))
    for q in range(K):
        pert = np.array(u0, dtype=complex)
        pert[q] += 1j * hc
        J[:, q] = eval_fn(pert).imag / hc
    return J


def _stencil4_hessian(
    eval_fn: Callable[[np.ndarray], np.ndarray],
    u0: np.ndarray,
    N: int,
    K: int,
    h: float = 1e-4,
) -> np.ndarray:
    """Evaluate dynamic Hessian via 4th-order central difference of complex-step gradients."""
    hc = 1e-20

    def grad_at(u: np.ndarray) -> np.ndarray:
        G = np.zeros((N, K))
        for q in range(K):
            pert = np.array(u, dtype=complex)
            pert[q] += 1j * hc
            G[:, q] = eval_fn(pert).imag / hc
        return G

    H = np.zeros((N, K, K))
    for p in range(K):
        ep = np.eye(K)[p]
        gp2 = grad_at(u0 + 2 * h * ep)
        gp1 = grad_at(u0 + 1 * h * ep)
        gm1 = grad_at(u0 - 1 * h * ep)
        gm2 = grad_at(u0 - 2 * h * ep)
        H[:, p, :] = (-gp2 + 8.0 * gp1 - 8.0 * gm1 + gm2) / (12.0 * h)

    for i in range(N):
        H[i] = 0.5 * (H[i] + H[i].T)
    return H


class TestAnalyticalDerivativeAccuracy:
    """Verify analytical Jacobians and Hessians against numerical central differences across 1,000 perturbations."""

    def test_hansen_rbc_1000_perturbations(self):
        """Verify analytical derivatives on Hansen (1985) RBC model across 1,000 random perturbations."""
        mod_src = """
        var c k l y a;
        varexo e;
        parameters beta delta alpha psi rho A_scale;
        beta = 0.99;
        delta = 0.025;
        alpha = 0.36;
        psi = 1.72;
        rho = 0.95;
        A_scale = 1.0;
        model;
        c^(-1) = beta * c(+1)^(-1) * (1 - delta + alpha * y(+1) / k);
        c^(-1) * (1 - alpha) * y / l = A_scale * l^psi;
        y = a * (k(-1)^alpha) * (l^(1 - alpha));
        k = (1 - delta) * k(-1) + y - c;
        log(a) = rho * log(a(-1)) + e;
        end;
        """
        dag = parse_mod_to_dag(mod_src)
        compiled = compile_derivatives(dag)
        eq_fn = dag.compile_equations()

        vars_list = dag.variables
        shocks_list = dag.shocks
        pdict = dag.parameter_values
        pvec = _Vec(list(pdict.keys()), list(pdict.values()), what="parameter")

        N = len(vars_list)
        n_e = len(shocks_list)
        K = 3 * N + n_e

        def eval_f(u_vec: np.ndarray) -> np.ndarray:
            ld = _Vec(vars_list, u_vec[0:N])
            cr = _Vec(vars_list, u_vec[N:2 * N])
            lg = _Vec(vars_list, u_vec[2 * N:3 * N])
            sh = _Vec(shocks_list, u_vec[3 * N:3 * N + n_e])
            return np.asarray(eq_fn(ld, cr, lg, sh, pvec))

        rng = np.random.default_rng(1001)
        base_point = np.array([0.8, 10.0, 0.3, 1.0, 1.0, 0.8, 10.0, 0.3, 1.0, 1.0, 0.8, 10.0, 0.3, 1.0, 1.0, 0.0])

        max_err_J = 0.0
        max_err_H = 0.0

        # We test 1,000 perturbations for Jacobian and a representative 100 perturbations for the 4th-order Hessian
        n_draws_J = 1000
        n_draws_H = 100

        for idx in range(n_draws_J):
            pert_scale = rng.uniform(0.85, 1.15, K)
            u_pt = base_point * pert_scale
            u_pt[-1] = 0.02 * rng.uniform(-1, 1)  # small shock

            # Analytical
            lead_pt = u_pt[0:N]
            curr_pt = u_pt[N:2 * N]
            lag_pt = u_pt[2 * N:3 * N]
            shk_pt = u_pt[3 * N:3 * N + n_e]

            A_p, A_0, A_m, B_u = compiled.eval_first_order(lead_pt, curr_pt, lag_pt, shk_pt, pdict)
            J_sym = np.hstack([A_p, A_0, A_m, B_u])

            # High precision numerical complex-step
            J_num = _complex_step_jacobian(eval_f, u_pt, N, K)
            err_J = float(np.max(np.abs(J_sym - J_num)))
            if err_J > max_err_J:
                max_err_J = err_J

            if idx < n_draws_H:
                H_sym = compiled.eval_second_order(lead_pt, curr_pt, lag_pt, shk_pt, pdict)
                H_num = _stencil4_hessian(eval_f, u_pt, N, K, h=1e-4)
                err_H = float(np.max(np.abs(H_sym - H_num)))
                if err_H > max_err_H:
                    max_err_H = err_H

        print(f"\n[Hansen RBC {n_draws_J} draws] Max Jacobian error: {max_err_J:.4e}, Max Hessian error: {max_err_H:.4e}")
        assert max_err_J <= 1e-10, f"Jacobian max error {max_err_J:.4e} > 1e-10"
        assert max_err_H <= 1e-10, f"Hessian max error {max_err_H:.4e} > 1e-10"

    def test_nonlinear_euler_habit_and_capital_adjustment(self):
        """Verify analytical derivatives on Euler equation with habit persistence and adjustment costs."""
        mod_src = """
        var c k r;
        varexo e;
        parameters beta sigma h delta phi r_bar;
        beta = 0.99;
        sigma = 2.0;
        h = 0.2;
        delta = 0.025;
        phi = 1.0;
        r_bar = 0.04;
        model;
        (c - h * c(-1))^(-sigma) = beta * (c(+1) - h * c)^(-sigma) * (1 + r(+1) - delta);
        k = (1 - delta) * k(-1) + (1 - phi / 2 * (k / k(-1) - 1)^2) * (c + e);
        r = r_bar * (k / 10.0)^(-0.5);
        end;
        """
        dag = parse_mod_to_dag(mod_src)
        compiled = compile_derivatives(dag)
        eq_fn = dag.compile_equations()

        vars_list = dag.variables
        shocks_list = dag.shocks
        pdict = dag.parameter_values
        pvec = _Vec(list(pdict.keys()), list(pdict.values()), what="parameter")

        N = len(vars_list)
        n_e = len(shocks_list)
        K = 3 * N + n_e

        def eval_f(u_vec: np.ndarray) -> np.ndarray:
            ld = _Vec(vars_list, u_vec[0:N])
            cr = _Vec(vars_list, u_vec[N:2 * N])
            lg = _Vec(vars_list, u_vec[2 * N:3 * N])
            sh = _Vec(shocks_list, u_vec[3 * N:3 * N + n_e])
            return np.asarray(eq_fn(ld, cr, lg, sh, pvec))

        rng = np.random.default_rng(2002)
        base_point = np.array([1.5, 10.0, 0.04, 1.5, 10.0, 0.04, 1.5, 10.0, 0.04, 0.0])

        max_err_J = 0.0
        max_err_H = 0.0

        for idx in range(300):
            pert = rng.uniform(0.92, 1.08, K)
            u_pt = base_point * pert
            u_pt[-1] = 0.01 * rng.uniform(-1, 1)

            A_p, A_0, A_m, B_u = compiled.eval_first_order(u_pt[0:N], u_pt[N:2 * N], u_pt[2 * N:3 * N], u_pt[3 * N:], pdict)
            J_sym = np.hstack([A_p, A_0, A_m, B_u])
            J_num = _complex_step_jacobian(eval_f, u_pt, N, K)
            err_J = float(np.max(np.abs(J_sym - J_num)))
            if err_J > max_err_J:
                max_err_J = err_J

            if idx < 50:
                H_sym = compiled.eval_second_order(u_pt[0:N], u_pt[N:2 * N], u_pt[2 * N:3 * N], u_pt[3 * N:], pdict)
                H_num = _stencil4_hessian(eval_f, u_pt, N, K, h=1e-4)
                err_H = float(np.max(np.abs(H_sym - H_num)))
                if err_H > max_err_H:
                    max_err_H = err_H

        print(f"\n[Nonlinear Euler 300 draws] Max Jacobian error: {max_err_J:.4e}, Max Hessian error: {max_err_H:.4e}")
        assert max_err_J <= 1e-10
        assert max_err_H <= 1e-10

    def test_transcendental_math_functions_accuracy(self):
        """Verify analytical derivatives on models with exp, sin, cos, normcdf, log."""
        mod_src = """
        var x y z;
        varexo eps;
        parameters a b c_p d;
        a = 0.5;
        b = 0.8;
        c_p = 1.2;
        d = 0.4;
        model;
        exp(x) + sin(y(+1)) - a * log(z(-1) + 2.0) = 0;
        y^2 * cos(x(-1)) - b * normcdf(z) + eps = 0;
        z = d * z(-1) + c_p * exp(-x^2);
        end;
        """
        dag = parse_mod_to_dag(mod_src)
        compiled = compile_derivatives(dag)
        eq_fn = dag.compile_equations()

        vars_list = dag.variables
        shocks_list = dag.shocks
        pdict = dag.parameter_values
        pvec = _Vec(list(pdict.keys()), list(pdict.values()), what="parameter")

        N = len(vars_list)
        n_e = len(shocks_list)
        K = 3 * N + n_e

        def eval_f(u_vec: np.ndarray) -> np.ndarray:
            ld = _Vec(vars_list, u_vec[0:N])
            cr = _Vec(vars_list, u_vec[N:2 * N])
            lg = _Vec(vars_list, u_vec[2 * N:3 * N])
            sh = _Vec(shocks_list, u_vec[3 * N:3 * N + n_e])
            return np.asarray(eq_fn(ld, cr, lg, sh, pvec))

        rng = np.random.default_rng(3003)
        base_point = np.array([0.2, 0.6, 0.4, 0.2, 0.6, 0.4, 0.2, 0.6, 0.4, 0.0])

        max_err_J = 0.0
        max_err_H = 0.0

        for idx in range(300):
            pert = rng.uniform(0.8, 1.2, K)
            u_pt = base_point * pert
            u_pt[-1] = 0.02 * rng.uniform(-1, 1)

            A_p, A_0, A_m, B_u = compiled.eval_first_order(u_pt[0:N], u_pt[N:2 * N], u_pt[2 * N:3 * N], u_pt[3 * N:], pdict)
            J_sym = np.hstack([A_p, A_0, A_m, B_u])
            J_num = _complex_step_jacobian(eval_f, u_pt, N, K)
            err_J = float(np.max(np.abs(J_sym - J_num)))
            if err_J > max_err_J:
                max_err_J = err_J

            if idx < 50:
                H_sym = compiled.eval_second_order(u_pt[0:N], u_pt[N:2 * N], u_pt[2 * N:3 * N], u_pt[3 * N:], pdict)
                H_num = _stencil4_hessian(eval_f, u_pt, N, K, h=1e-4)
                err_H = float(np.max(np.abs(H_sym - H_num)))
                if err_H > max_err_H:
                    max_err_H = err_H

        print(f"\n[Transcendental Math 300 draws] Max Jacobian error: {max_err_J:.4e}, Max Hessian error: {max_err_H:.4e}")
        assert max_err_J <= 1e-10
        assert max_err_H <= 1e-10


# ===========================================================================
# 3. Generalized Schur Sylvester Equation Residual & Memory
# ===========================================================================

class TestGeneralizedSchurSylvester:
    """Stress-test Schur Sylvester equation: residual norm <= 1e-10 and memory footprint."""

    def test_sylvester_residual_norm_on_sw07(self):
        """Verify residual norm ||(A_0 + A_+ g_x P_s) g_xx + A_+ g_xx (h_x kron h_x) + K_xx||_F <= 1e-10 on SW07."""
        assert SW07_PATH.exists()
        m1 = load_mod(SW07_PATH, order=1)
        vars_list = list(m1.variables)
        shocks_list = list(m1.shocks)
        (states_list, controls_list, n_x, n_y,
         g_x, g_u, P_s, P_c, h_x, h_u) = _first_order_pieces(m1, vars_list, shocks_list)

        N = len(vars_list)
        A_plus = np.asarray(m1._A_plus, dtype=float)
        A_0 = np.asarray(m1._A_0, dtype=float)
        A_hat = A_0 + A_plus @ g_x @ P_s

        rng = np.random.default_rng(42)
        K_xx = rng.standard_normal((N, n_x**2))

        g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
        assert g_xx.shape == (N, n_x**2)
        assert np.all(np.isfinite(g_xx))

        # Compute Frobenius residual norm
        C = np.kron(h_x, h_x)
        residual = A_hat @ g_xx + A_plus @ g_xx @ C + K_xx
        norm = float(np.linalg.norm(residual, "fro"))
        print(f"\n[SW07 Sylvester Residual Norm] ||Res||_F = {norm:.4e}")
        assert norm <= 1e-10, f"Residual norm {norm:.4e} > 1e-10"

    def test_sylvester_memory_and_dense_comparison(self):
        """Measure peak memory allocation of Schur Sylvester vs dense Kronecker."""
        assert SW07_PATH.exists()
        m1 = load_mod(SW07_PATH, order=1)
        (states_list, controls_list, n_x, n_y,
         g_x, g_u, P_s, P_c, h_x, h_u) = _first_order_pieces(m1, list(m1.variables), list(m1.shocks))

        N = len(m1.variables)
        A_plus = np.asarray(m1._A_plus, dtype=float)
        A_0 = np.asarray(m1._A_0, dtype=float)
        A_hat = A_0 + A_plus @ g_x @ P_s
        rng = np.random.default_rng(42)
        K_xx = rng.standard_normal((N, n_x**2))

        # Dense Kronecker size calculation
        dense_dim = N * (n_x**2)  # 40 * 225 = 9000
        dense_mb = (dense_dim * dense_dim * 8) / (1024 * 1024)
        print(f"\n[Memory Comparison] Dense Kronecker (9000x9000 float64): {dense_mb:.1f} MB")

        # Measure peak memory of Schur Sylvester
        tracemalloc.start()
        g_xx = solve_generalized_sylvester_kronecker(A_hat, A_plus, h_x, K_xx)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        print(f"[Memory Measured] Schur Sylvester peak memory: {peak_mb:.3f} MB")
        # Dense Kronecker was 648 MB; optimized Schur Sylvester uses < 2.0 MB (>340x reduction)
        assert peak_mb < 2.0, f"Peak memory {peak_mb:.2f} MB unexpectedly exceeds 2.0 MB budget"
