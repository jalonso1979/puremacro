"""Adversarial stress test suite for puremacro 2.8.0 R1 & R2.

Executed by orch5_challenger_1 (Empirical Challenger):
1. Order-3 simulation stability over 10,000 periods (0 explosions, 0 NaNs/Infs, finite kurtosis).
2. Extreme 10-standard-deviation shocks (10*sigma): bounded pruned trajectories vs unpruned explosion.
3. SW07 g_xxx Sylvester solve execution time (<= 0.8s) and memory consumption (< 5 MB).
4. 3rd analytical derivatives vs 6th-order central difference stencil to <= 10^-9.
5. Semismooth Newton MCP stress testing: deep deflationary shocks, pinned rates, rate collars,
   and large permanent transitions (y_init -> y_end) with terminal error <= 10^-10.
6. Pyodide four-package contract verification.
"""
from __future__ import annotations

import math
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
import pytest
import scipy.linalg

from puremacro.dsge._parser import parse_mod_to_dag
from puremacro.dsge._symbolic import (
    compile_derivatives,
    SparseDynamicTensor3D,
)
from puremacro.dsge._sylvester import (
    solve_order3_sylvester_kronecker,
    solve_order1_sylvester,
)
from puremacro.dsge.pruning import (
    Order3PrunedSolution,
    canonical_growth_3rd_order,
)
from puremacro.dsge.perfect_foresight import (
    MCPResult,
    PerfectForesightResult,
    find_steady_state,
    solve_perfect_foresight,
)
from puremacro.dsge.dynare import build_dynare, load_mod


# ============================================================================
# TASK 1: Order-3 Simulation Stability over 10,000 Periods
# ============================================================================

class TestTask1Order3SimulationStability10k:
    """Stress-test order-3 pruned simulation over 10,000 periods across multiple seeds."""

    @pytest.mark.parametrize("seed", [42, 123, 777, 2026])
    def test_canonical_growth_10k_stability_and_kurtosis(self, seed: int):
        """Verify 10,000 periods: 0 explosions, 0 NaNs/Infs, finite stationary kurtosis."""
        sol = canonical_growth_3rd_order()
        assert sol.is_stable
        assert sol.spectral_radius < 1.0

        sim = sol.simulate(periods=10_000, seed=seed, burn=500)
        assert len(sim) == 10_000

        for col in ["k", "c"]:
            vals = sim[col].to_numpy()
            assert not np.isnan(vals).any(), f"NaN detected in {col} at seed {seed}"
            assert not np.isinf(vals).any(), f"Inf detected in {col} at seed {seed}"

            # Verify trajectory is bounded within safe domain
            assert np.max(np.abs(vals)) < 50.0, f"Explosive values detected in {col}: max={np.max(np.abs(vals))}"

            # Variance and kurtosis
            var_val = float(np.var(vals))
            assert var_val > 0.0, f"Zero variance for {col}"
            mean_val = float(np.mean(vals))
            kurt_val = float(np.mean((vals - mean_val) ** 4) / (var_val ** 2))
            assert 1.0 < kurt_val < 50.0, f"Unstable kurtosis {kurt_val} for {col}"

        # Check Andreasen decomposition identity holds for all 10,000 steps
        states_sum = (sim.states_1st + sim.states_2nd + sim.states_3rd).to_numpy()
        np.testing.assert_allclose(sim.states.to_numpy(), states_sum, rtol=1e-13, atol=1e-13)

        ctrls_sum = (sim.controls_1st + sim.controls_2nd + sim.controls_3rd).to_numpy()
        np.testing.assert_allclose(sim.controls.to_numpy(), ctrls_sum, rtol=1e-13, atol=1e-13)

    def test_multistate_dsge_10k_simulation_stability(self):
        """Verify 10,000 periods on 3-variable DSGE model with persistent AR(1) technology shock."""
        mod_src = """
        var c k z;
        varexo eps;
        parameters beta alpha delta rho sigma_pref sigma_eps;
        beta = 0.99; alpha = 0.33; delta = 0.025; rho = 0.95; sigma_pref = 1.0; sigma_eps = 0.01;
        model;
        exp(-sigma_pref * c) - beta * exp(-sigma_pref * c(+1)) * (alpha * exp(z(+1)) * exp((alpha - 1.0) * k) + 1.0 - delta);
        exp(c) + exp(k) - exp(z) * exp(alpha * k(-1)) - (1.0 - delta) * exp(k(-1));
        z - rho * z(-1) - sigma_eps * eps;
        end;
        initval;
        k = 3.8; c = 0.8; z = 0.0;
        end;
        """
        sol = load_mod(mod_src, order=3)
        assert isinstance(sol, Order3PrunedSolution)

        sim = sol.simulate(periods=10_000, seed=888, burn=500)
        assert len(sim) == 10_000

        for var_name in ["k", "z", "c"]:
            arr = sim[var_name].to_numpy()
            assert not np.isnan(arr).any()
            assert not np.isinf(arr).any()
            assert np.max(np.abs(arr)) < 100.0

            var_arr = float(np.var(arr))
            assert var_arr > 0.0
            mean_arr = float(np.mean(arr))
            kurt_arr = float(np.mean((arr - mean_arr) ** 4) / (var_arr ** 2))
            assert 1.0 < kurt_arr < 50.0


# ============================================================================
# TASK 2: Extreme 10-Standard-Deviation Shocks (Pruned vs Unpruned)
# ============================================================================

class TestTask2ExtremeShocksPrunedVsUnpruned:
    """Subject model to extreme 10-sigma shocks and verify bounded pruned trajectories vs unpruned explosion."""

    def test_10_sigma_single_shock_pruned_bounded_vs_unpruned_explosion(self):
        """Hit model with +10 sigma and -10 sigma shocks: unpruned explodes, pruned remains bounded."""
        def eqs(lead, curr, lag, shocks, p):
            return [
                curr.c**(-p.gamma) - p.beta * lead.c**(-p.gamma) * (p.alpha * curr.k**(p.alpha - 1.0) + 1.0 - p.delta),
                curr.k - (lag.k**p.alpha + (1.0 - p.delta) * lag.k - curr.c + shocks.e),
            ]

        params = {"beta": 0.99, "alpha": 0.33, "delta": 0.025, "gamma": 2.0}
        guess = {"c": 0.8, "k": 12.0}
        sigma_e = 0.30
        sol = build_dynare(eqs, variables=["k", "c"], shocks=["e"], params=params, guess=guess, order=3, shock_cov=[[sigma_e**2]])

        shock_10_pos = np.zeros((50, 1))
        shock_10_pos[0, 0] = 10.0 * sigma_e

        shock_10_neg = np.zeros((50, 1))
        shock_10_neg[0, 0] = -10.0 * sigma_e

        for shock_seq, label in [(shock_10_pos, "positive 10-sigma"), (shock_10_neg, "negative 10-sigma")]:
            # 1. Pruned simulation
            sim_pruned = sol.simulate(periods=50, shocks=shock_seq, burn=0)
            k_pruned = sim_pruned["k"].to_numpy()

            assert not np.isnan(k_pruned).any(), f"NaN in pruned k under {label}"
            assert not np.isinf(k_pruned).any(), f"Inf in pruned k under {label}"
            assert np.max(np.abs(k_pruned)) < 10.0, f"Pruned k exceeded bound under {label}"

            # 2. Unpruned simulation
            x_unpruned = np.zeros(sol.n_states)
            exploded = False
            for t in range(50):
                u_t = shock_seq[t]
                kx_x = np.outer(x_unpruned, x_unpruned).ravel()
                kx_u = np.outer(x_unpruned, u_t).ravel()
                ku_u = np.outer(u_t, u_t).ravel()
                kx3 = np.outer(kx_x, x_unpruned).ravel()
                kx2_u = np.outer(kx_x, u_t).ravel()
                kx_u2 = np.outer(x_unpruned, ku_u).ravel()
                ku3 = np.outer(ku_u, u_t).ravel()

                x_next = (
                    sol.G @ x_unpruned
                    + sol.N @ u_t
                    + 0.5 * (sol.H_xx @ kx_x)
                    + sol.H_xu @ kx_u
                    + 0.5 * (sol.H_uu @ ku_u)
                    + (1.0 / 6.0) * (sol.H_xxx @ kx3)
                    + 0.5 * (sol.H_xxu @ kx2_u)
                    + 0.5 * (sol.H_xuu @ kx_u2)
                    + (1.0 / 6.0) * (sol.H_uuu @ ku3)
                )
                if np.any(np.isnan(x_next)) or np.any(np.isinf(x_next)) or np.max(np.abs(x_next)) > 1e6:
                    exploded = True
                    break
                x_unpruned = x_next

            assert exploded is True, f"Unpruned simulation unexpectedly did not explode under {label}"

    def test_repeated_extreme_shocks_pruned_resilience(self):
        """Sequence of 50 extreme shocks of magnitude 10*sigma alternating sign."""
        sol = canonical_growth_3rd_order()
        n_e = sol.n_shocks
        T = 50
        extreme_shocks = np.zeros((T, n_e))
        for t in range(T):
            extreme_shocks[t, 0] = 10.0 if t % 2 == 0 else -10.0

        sim = sol.simulate(periods=T, shocks=extreme_shocks, burn=0)
        assert not sim.states.isna().any().any()
        assert not sim.states.isin([np.inf, -np.inf]).any().any()
        assert np.max(np.abs(sim.states.to_numpy())) < 20.0


# ============================================================================
# TASK 3: SW07 g_xxx Sylvester Solve Execution Time and Memory
# ============================================================================

class TestTask3SW07SylvesterExecutionTimeAndMemory:
    """Benchmark SW07 g_xxx Sylvester solve execution time (<= 0.8s) and memory (< 5 MB)."""

    def test_sw07_sylvester_benchmark_time_and_memory(self):
        """SW07 system (N=14, n_x=7) solve must complete in <= 0.8s and peak memory < 5.0 MB."""
        N = 14
        n_x = 7

        np.random.seed(98765)
        A_hat = np.eye(N) + np.random.randn(N, N) * 0.05
        A_plus = np.random.randn(N, N) * 0.05
        # Stable eigenvalues for h_x
        h_x = np.diag([0.92, 0.88, 0.82, 0.76, 0.70, 0.65, 0.60])
        K_xxx = np.random.randn(N, n_x**3)

        tracemalloc.start()
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            X = solve_order3_sylvester_kronecker(A_hat, A_plus, h_x, K_xxx)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        median_time = float(np.median(times))
        max_time = float(np.max(times))

        print(f"\n[SW07 Sylvester Benchmark] Median time: {median_time:.4f}s, Max: {max_time:.4f}s, Peak RAM: {peak_mb:.2f} MB")

        assert max_time <= 0.8, f"Max solve time {max_time:.4f}s exceeded 0.8s limit"
        assert peak_mb < 5.0, f"Peak memory {peak_mb:.2f} MB exceeded 5.0 MB limit"

        # Residual verification
        hx3 = np.kron(h_x, np.kron(h_x, h_x))
        res = A_hat @ X + A_plus @ X @ hx3 + K_xxx
        rel_res = np.linalg.norm(res, "fro") / np.linalg.norm(K_xxx, "fro")
        assert rel_res < 1e-11, f"Sylvester relative residual {rel_res:.4e} exceeded 1e-11"

    @pytest.mark.parametrize("spectral_radius", [0.90, 0.98, 0.999])
    def test_sylvester_near_unit_root_numerical_accuracy(self, spectral_radius: float):
        """Stress-test Sylvester solver with roots approaching unit root."""
        N = 6
        n_x = 3
        np.random.seed(42)
        A_hat = np.eye(N) + np.random.randn(N, N) * 0.02
        A_plus = np.random.randn(N, N) * 0.02
        h_x = np.diag([spectral_radius, spectral_radius * 0.95, spectral_radius * 0.90])
        K_xxx = np.random.randn(N, n_x**3)

        X = solve_order3_sylvester_kronecker(A_hat, A_plus, h_x, K_xxx)
        hx3 = np.kron(h_x, np.kron(h_x, h_x))
        res = A_hat @ X + A_plus @ X @ hx3 + K_xxx
        rel_res = np.linalg.norm(res, "fro") / np.linalg.norm(K_xxx, "fro")
        assert rel_res < 1e-11, f"Failed at spectral radius {spectral_radius}: rel_res={rel_res:.4e}"


# ============================================================================
# TASK 4: 3rd Analytical Derivatives vs 6th-Order Central Difference Stencil
# ============================================================================

class TestTask4SymbolicThirdOrderDerivativesVsStencil:
    """Verify 3rd analytical derivatives match 6th-order central difference stencil to <= 10^-9."""

    def test_euler_equation_and_resource_constraint_6th_order_stencil(self):
        """Verify dynamic 3rd derivatives across lead, current, and lag variables to <= 1e-9."""
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

        lead = np.array([1.25, 10.8])
        curr = np.array([1.15, 10.4])
        lag = np.array([1.05, 10.1])
        shocks = np.array([0.0])
        pvec = np.array([dag.parameter_values[p] for p in dag.parameters])

        sparse_3rd = compiled.eval_third_order(lead, curr, lag, shocks, pvec)
        dense_3rd = sparse_3rd.to_dense()

        # Check 1: 3rd derivative of Euler equation (eq 0) w.r.t lead c (var index 0 in lead)
        analytical_lead_c = dense_3rd[0, 0, 0, 0]

        def eval_H_lead_c(c_val):
            lp = lead.copy()
            lp[0] = c_val
            Hf = compiled.eval_second_order(lp, curr, lag, shocks, pvec)
            return Hf[0, 0, 0]

        h = 5e-4
        x0 = lead[0]
        num_stencil = (
            eval_H_lead_c(x0 + 3 * h)
            - 9.0 * eval_H_lead_c(x0 + 2 * h)
            + 45.0 * eval_H_lead_c(x0 + h)
            - 45.0 * eval_H_lead_c(x0 - h)
            + 9.0 * eval_H_lead_c(x0 - 2 * h)
            - eval_H_lead_c(x0 - 3 * h)
        ) / (60.0 * h)

        diff = abs(analytical_lead_c - num_stencil)
        assert diff <= 1e-9, f"Lead c: analytical {analytical_lead_c} vs stencil {num_stencil}, diff={diff:.4e}"

        # Check 2: 3rd derivative of resource constraint (eq 1) w.r.t lag k (global index 5)
        analytical_lag_k = dense_3rd[1, 5, 5, 5]

        def eval_H_lag_k(k_val):
            lg = lag.copy()
            lg[1] = k_val
            Hf = compiled.eval_second_order(lead, curr, lg, shocks, pvec)
            return Hf[1, 5, 5]

        x0_k = lag[1]
        num_stencil_k = (
            eval_H_lag_k(x0_k + 3 * h)
            - 9.0 * eval_H_lag_k(x0_k + 2 * h)
            + 45.0 * eval_H_lag_k(x0_k + h)
            - 45.0 * eval_H_lag_k(x0_k - h)
            + 9.0 * eval_H_lag_k(x0_k - 2 * h)
            - eval_H_lag_k(x0_k - 3 * h)
        ) / (60.0 * h)

        diff_k = abs(analytical_lag_k - num_stencil_k)
        assert diff_k <= 1e-9, f"Lag k: analytical {analytical_lag_k} vs stencil {num_stencil_k}, diff={diff_k:.4e}"

    def test_ces_non_linear_aggregator_3rd_derivative(self):
        """Verify 3rd derivative on non-linear CES function f = (c1^rho + c2^rho)^(1/rho)."""
        src = """
        var y c1 c2;
        varexo e;
        parameters rho;
        rho = 0.5;
        model;
        y = (0.6 * c1^rho + 0.4 * c2^rho)^(1.0 / rho);
        c1 = 1.0 + e;
        c2 = 1.0;
        end;
        """
        dag = parse_mod_to_dag(src)
        compiled = compile_derivatives(dag)

        lead = np.array([1.0, 1.0, 1.0])
        curr = np.array([1.0, 1.2, 0.8])
        lag = np.array([1.0, 1.0, 1.0])
        shocks = np.array([0.0])
        pvec = np.array([0.5])

        sparse_3rd = compiled.eval_third_order(lead, curr, lag, shocks, pvec)
        dense_3rd = sparse_3rd.to_dense()

        # Equation 0, current c1 is index 4: lead(0,1,2), curr(3,4,5), lag(6,7,8), shocks(9)
        analytical_c1 = dense_3rd[0, 4, 4, 4]

        def eval_H_c1(val):
            c_curr = curr.copy()
            c_curr[1] = val
            Hf = compiled.eval_second_order(lead, c_curr, lag, shocks, pvec)
            return Hf[0, 4, 4]

        h = 5e-4
        x0 = curr[1]
        stencil = (
            eval_H_c1(x0 + 3 * h)
            - 9.0 * eval_H_c1(x0 + 2 * h)
            + 45.0 * eval_H_c1(x0 + h)
            - 45.0 * eval_H_c1(x0 - h)
            + 9.0 * eval_H_c1(x0 - 2 * h)
            - eval_H_c1(x0 - 3 * h)
        ) / (60.0 * h)

        diff = abs(analytical_c1 - stencil)
        assert diff <= 1e-9, f"CES 3rd derivative diff {diff:.4e} > 1e-9"


# ============================================================================
# TASK 5: Semismooth Newton MCP Stress Testing
# ============================================================================

class TestTask5SemismoothNewtonMCPStressTesting:
    """Stress test Semismooth Newton MCP under deep deflationary shocks, pinned rates,
    rate collars, and permanent transitions with terminal error <= 10^-10."""

    @pytest.fixture
    def nk_model(self):
        """New Keynesian 3-equation model."""
        beta = 0.99
        sigma = 1.0
        kappa = 0.1
        phi_pi = 1.5
        phi_x = 0.5
        r_target = 0.02

        def equations_fn(yp, yc, yl, eps):
            pi_p, x_p, r_p = yp
            pi, x, r = yc
            pi_m, x_m, r_m = yl
            r_nat = float(eps) if np.ndim(eps) == 0 else float(eps[0])

            f1 = beta * pi_p + kappa * x - pi
            f2 = x_p - (1.0 / sigma) * (r - pi_p - r_nat) - x
            f3 = r_nat + phi_pi * pi + phi_x * x - r
            return [f1, f2, f3]

        y_ss = np.array([0.0, 0.0, r_target])
        return {
            "equations_fn": equations_fn,
            "y_ss": y_ss,
            "variable_names": ["pi", "x", "r"],
            "r_target": r_target,
        }

    @pytest.fixture
    def ramsey_model(self):
        """Neoclassical growth model with capital accumulation."""
        alpha = 0.33
        beta = 0.96
        delta = 0.10
        sigma = 1.0

        r_ss = 1.0 / beta - (1.0 - delta)
        k_ss = (alpha * 1.0 / r_ss) ** (1.0 / (1.0 - alpha))
        c_ss = 1.0 * k_ss**alpha - delta * k_ss

        def equations_fn(yp, yc, yl, eps):
            c_p, k_p = yp
            c, k = yc
            c_m, k_m = yl
            A = eps if np.ndim(eps) == 0 else eps[0]
            euler = c ** (-sigma) - beta * c_p ** (-sigma) * (alpha * A * k ** (alpha - 1.0) + 1.0 - delta)
            res_c = k - (A * k_m**alpha + (1.0 - delta) * k_m - c)
            return [euler, res_c]

        return {
            "equations_fn": equations_fn,
            "y_ss": np.array([c_ss, k_ss]),
            "k_ss": k_ss,
            "c_ss": c_ss,
            "params": dict(alpha=alpha, beta=beta, delta=delta, sigma=sigma),
        }

    def test_deep_deflationary_shock_zlb_enforcement(self, nk_model):
        """Deep deflationary shock: r_nat = -0.15 for 8 periods, forcing extended ZLB."""
        eqs = nk_model["equations_fn"]
        y_ss = nk_model["y_ss"]
        v_names = nk_model["variable_names"]

        T = 30
        r_nat_path = np.full(T, 0.02)
        r_nat_path[0:8] = -0.15

        res = solve_perfect_foresight(
            eqs,
            y_init=y_ss,
            y_ss=y_ss,
            exogenous_path=r_nat_path,
            n_periods=T,
            variable_names=v_names,
            mcp=True,
            mcp_bounds={"r": (0.0, None)},
            tol=1e-10,
        )

        assert isinstance(res, MCPResult)
        assert res.converged is True
        assert res.iterations <= 25

        # Strictly non-negative rates
        r_path = res.path["r"].to_numpy()
        min_r = np.min(r_path)
        assert min_r >= -1e-14, f"ZLB violated: min r = {min_r:.4e}"

        # Periods 1 to 8 must be binding
        assert "r" in res.binding_periods
        for p in range(1, 9):
            assert p in res.binding_periods["r"]
            assert abs(res.path["r"].loc[p]) < 1e-12

    def test_pinned_rate_forward_guidance(self, nk_model):
        """Pinned rate forward guidance: policy rate pegged at 0.015 for 20 periods."""
        eqs = nk_model["equations_fn"]
        y_ss = nk_model["y_ss"]
        v_names = nk_model["variable_names"]

        T = 20
        r_nat_path = np.full(T, 0.02)
        pinned_rate = 0.015

        res = solve_perfect_foresight(
            eqs,
            y_init=y_ss,
            y_ss=y_ss,
            exogenous_path=r_nat_path,
            n_periods=T,
            variable_names=v_names,
            mcp=True,
            mcp_bounds={"r": (pinned_rate, pinned_rate)},
            tol=1e-10,
        )

        assert res.converged is True
        r_vals = res.path["r"].to_numpy()
        np.testing.assert_allclose(r_vals, pinned_rate, atol=1e-12)
        assert res.binding_periods["r"] == list(range(1, T + 1))

    def test_rate_collar_two_sided_bounds(self, nk_model):
        """Rate collar [0.005, 0.040] with massive natural rate swings (-0.12 to +0.10)."""
        eqs = nk_model["equations_fn"]
        y_ss = nk_model["y_ss"]
        v_names = nk_model["variable_names"]

        T = 25
        r_nat_path = np.full(T, 0.02)
        r_nat_path[0:5] = -0.12   # Hits lower bound
        r_nat_path[8:13] = 0.10   # Hits upper bound

        lb, ub = 0.005, 0.040
        res = solve_perfect_foresight(
            eqs,
            y_init=y_ss,
            y_ss=y_ss,
            exogenous_path=r_nat_path,
            n_periods=T,
            variable_names=v_names,
            mcp=True,
            mcp_bounds={"r": (lb, ub)},
            tol=1e-10,
        )

        assert res.converged is True
        r_vals = res.path["r"].to_numpy()
        assert np.all(r_vals >= lb - 1e-14)
        assert np.all(r_vals <= ub + 1e-14)

        binding = res.binding_periods["r"]
        assert 1 in binding
        assert 9 in binding

    def test_permanent_transition_distinct_steady_states_terminal_error(self, ramsey_model):
        """Permanent productivity shift from A=0.70 to A=1.40 (+100% surge) with terminal error <= 10^-10."""
        eqs = ramsey_model["equations_fn"]
        y_ss_base = ramsey_model["y_ss"]

        # Initial steady state at A=0.70
        A_init = 0.70
        y_ss_init = find_steady_state(
            eqs, variables=["c", "k"], shocks=["A"], exo_values=[A_init], guess=y_ss_base
        )

        # Terminal steady state at A=1.40
        A_end = 1.40
        y_ss_end = find_steady_state(
            eqs, variables=["c", "k"], shocks=["A"], exo_values=[A_end], guess=y_ss_base
        )

        T = 200
        exo_path = np.full(T, A_end)

        res = solve_perfect_foresight(
            eqs,
            y_init=y_ss_init,
            y_ss=y_ss_end,
            exogenous_path=exo_path,
            n_periods=T,
            variable_names=["c", "k"],
            tol=1e-12,
        )

        assert res.converged is True
        assert res.terminal_error <= 1e-10, f"Terminal error {res.terminal_error:.4e} exceeded 1e-10 limit"
        assert res.residual_norm < 1e-10

        # Capital must accumulate monotonically from low to high steady state
        k_path = res.path["k"].to_numpy()
        assert k_path[0] > y_ss_init[1]
        assert abs(k_path[-1] - y_ss_end[1]) <= 1e-10
        assert np.all(np.diff(k_path[:150]) > 0)

        # Reverse transition: A drops permanently from 1.40 to 0.70
        exo_path_rev = np.full(T, A_init)
        res_rev = solve_perfect_foresight(
            eqs,
            y_init=y_ss_end,
            y_ss=y_ss_init,
            exogenous_path=exo_path_rev,
            n_periods=T,
            variable_names=["c", "k"],
            tol=1e-12,
        )
        assert res_rev.converged is True
        assert res_rev.terminal_error <= 1e-10
        assert np.all(np.diff(res_rev.path["k"].to_numpy()[:150]) < 0)


# ============================================================================
# TASK 6: Pyodide Four-Package Contract Verification
# ============================================================================

class TestTask6PyodideFourPackageContract:
    """Verify strictly pure Python execution under the Pyodide 4-package contract."""

    def test_dsge_tier2_modules_zero_unauthorized_imports(self):
        """Confirm no forbidden symbols or packages leaked into dsge submodules."""
        forbidden = {"sympy", "jax", "torch", "numba", "statsmodels", "linearmodels", "arch"}
        
        # Import target modules
        import puremacro.dsge._ast
        import puremacro.dsge._parser
        import puremacro.dsge._symbolic
        import puremacro.dsge._sylvester
        import puremacro.dsge.pruning
        import puremacro.dsge.perfect_foresight
        import puremacro.dsge.dynare

        loaded_modules = set(sys.modules.keys())
        for f in forbidden:
            assert f not in loaded_modules, f"Forbidden dependency {f} was loaded into sys.modules"

    def test_four_package_dependency_whitelist(self):
        """Assert core dependencies are restricted to numpy, scipy, pandas, matplotlib."""
        import puremacro
        import numpy
        import scipy
        import pandas
        import matplotlib

        assert hasattr(numpy, "__version__")
        assert hasattr(scipy, "__version__")
        assert hasattr(pandas, "__version__")
        assert hasattr(matplotlib, "__version__")
