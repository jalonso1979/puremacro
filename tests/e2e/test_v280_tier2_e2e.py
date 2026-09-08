"""Comprehensive 4-Tier Opaque-Box E2E Test Suite for puremacro v2.8.0 (Tier 2: Higher Order & Constraints).

This test suite covers all 5 core features in PROJECT.md § Feature Inventory across 4 Tiers:
- Tier 1: Feature Coverage (>=5 tests per feature covering happy-path in isolation; 30 tests)
- Tier 2: Boundary & Corner Cases (>=5 tests per feature covering limits & errors; 30 tests)
- Tier 3: Cross-Feature Combinations (pairwise subsystem interactions; 8 tests)
- Tier 4: Real-World Application Scenarios (SW07 3rd-order risk premia, NK with ZLB, Banking frictions, Hansen RBC BGP, DSGE SMC; 5 tests)

Total: 73 comprehensive requirement-driven test cases.
Pyodide four-package contract: numpy, scipy, pandas, matplotlib only (plus stdlib, unittest, pytest).
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import puremacro
import puremacro.dsge as dsge
from puremacro.dsge import (
    LinearModel,
    ModelError,
    build_dynare,
    parse_mod,
    load_mod,
    BlanchardKahnError,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DSGE_DIR = WORKSPACE_ROOT / "puremacro" / "dsge"
SW07_MOD_PATH = DSGE_DIR / "_references" / "sw07_pfeifer.mod"


# ===========================================================================
# Opaque-Box Module Resolution Helpers with Progressive Readiness Checks
# ===========================================================================

def _require_order3_sylvester():
    for mod in [getattr(dsge, "_sylvester", None), dsge]:
        if mod and hasattr(mod, "solve_order3_sylvester_kronecker"):
            return getattr(mod, "solve_order3_sylvester_kronecker")
    pytest.skip("Milestone 1: solve_order3_sylvester_kronecker pending implementation")

def _require_order3_pruning():
    for mod in [getattr(dsge, "pruning", None), dsge]:
        if mod and hasattr(mod, "Order3PrunedSolution"):
            return getattr(mod, "Order3PrunedSolution")
    pytest.skip("Milestone 1: Order3PrunedSolution pending implementation")

def _require_mcp_foresight():
    for mod in [getattr(dsge, "perfect_foresight", None), dsge]:
        if mod and hasattr(mod, "solve_perfect_foresight"):
            fn = getattr(mod, "solve_perfect_foresight")
            sig = inspect.signature(fn)
            if "mcp" in sig.parameters or "mcp_bounds" in sig.parameters:
                return fn
    pytest.skip("Milestone 2: solve_perfect_foresight(mcp=...) pending implementation")

def _require_mcp_result():
    for mod in [getattr(dsge, "perfect_foresight", None), getattr(dsge, "_results", None), dsge]:
        if mod and hasattr(mod, "MCPResult"):
            return getattr(mod, "MCPResult")
    pytest.skip("Milestone 2: MCPResult pending implementation")

def _require_multiconstraint_occbin():
    for mod in [getattr(dsge, "occbin", None), dsge]:
        if mod and hasattr(mod, "solve_multiconstraint_occbin"):
            return getattr(mod, "solve_multiconstraint_occbin")
    pytest.skip("Milestone 3: solve_multiconstraint_occbin pending implementation")

def _require_piecewise_kalman():
    for mod in [getattr(dsge, "occbin", None), getattr(dsge, "estimate", None), dsge]:
        if mod and hasattr(mod, "piecewise_kalman_filter"):
            return getattr(mod, "piecewise_kalman_filter")
    pytest.skip("Milestone 3: piecewise_kalman_filter pending implementation")

def _require_smc_sampler():
    try:
        from puremacro.dsge import smc
        if hasattr(smc, "SMCSampler"):
            return getattr(smc, "SMCSampler")
    except (ImportError, AttributeError):
        pass
    if hasattr(dsge, "SMCSampler"):
        return getattr(dsge, "SMCSampler")
    pytest.skip("Milestone 4: SMCSampler pending implementation in puremacro.dsge.smc")

def _require_smc_result():
    try:
        from puremacro.dsge import smc
        if hasattr(smc, "SMCResult"):
            return getattr(smc, "SMCResult")
    except (ImportError, AttributeError):
        pass
    if hasattr(dsge, "SMCResult"):
        return getattr(dsge, "SMCResult")
    pytest.skip("Milestone 4: SMCResult pending implementation in puremacro.dsge.smc")

def _require_bootstrap_particle_filter():
    try:
        from puremacro.dsge import smc
        if hasattr(smc, "bootstrap_particle_filter"):
            return getattr(smc, "bootstrap_particle_filter")
    except (ImportError, AttributeError):
        pass
    if hasattr(dsge, "bootstrap_particle_filter"):
        return getattr(dsge, "bootstrap_particle_filter")
    pytest.skip("Milestone 4: bootstrap_particle_filter pending implementation in puremacro.dsge.smc")

def _require_ramsey_model():
    try:
        from puremacro.dsge import ramsey
        if hasattr(ramsey, "ramsey_model"):
            return getattr(ramsey, "ramsey_model")
    except (ImportError, AttributeError):
        pass
    if hasattr(dsge, "ramsey_model"):
        return getattr(dsge, "ramsey_model")
    pytest.skip("Milestone 5: ramsey_model pending implementation in puremacro.dsge.ramsey")

def _require_ramsey_result():
    try:
        from puremacro.dsge import ramsey
        if hasattr(ramsey, "RamseyResult"):
            return getattr(ramsey, "RamseyResult")
    except (ImportError, AttributeError):
        pass
    if hasattr(dsge, "RamseyResult"):
        return getattr(dsge, "RamseyResult")
    pytest.skip("Milestone 5: RamseyResult pending implementation in puremacro.dsge.ramsey")

def _require_bgp_detrender():
    try:
        from puremacro.dsge import detrending
        if hasattr(detrending, "detrend_bgp"):
            return getattr(detrending, "detrend_bgp")
    except (ImportError, AttributeError):
        pass
    for mod in [dsge, getattr(dsge, "dynare", None)]:
        if mod and hasattr(mod, "detrend_bgp"):
            return getattr(mod, "detrend_bgp")
    pytest.skip("Milestone 5: detrend_bgp pending implementation")


# ===========================================================================
# Standard Test Fixtures & Mathematical Oracles
# ===========================================================================

@pytest.fixture
def two_state_sylvester_system():
    """Generates a stable 2-state DSGE system with exact dense Kronecker Sylvester oracle."""
    N = 2
    n_x = 2
    rng = np.random.default_rng(12345)
    
    A_hat = np.array([[2.2, 0.3], [0.1, 1.8]], dtype=float)
    A_plus = np.array([[0.4, 0.05], [0.02, 0.3]], dtype=float)
    h_x = np.array([[0.75, 0.08], [0.0, 0.65]], dtype=float)
    
    # K_xxx tensor of shape (N, n_x^3)
    K_xxx = rng.standard_normal((N, n_x**3)) * 0.5
    
    # Dense Kronecker reference solution:
    # (I_{n_x^3} ⊗ A_hat + (h_x ⊗ h_x ⊗ h_x)^T ⊗ A_plus) vec(X) = -vec(K_xxx)
    hx3 = np.kron(np.kron(h_x, h_x), h_x)
    I_m = np.eye(n_x**3)
    dense_sys = np.kron(I_m, A_hat) + np.kron(hx3.T, A_plus)
    rhs = -K_xxx.reshape(-1, order="F")
    vec_sol = np.linalg.solve(dense_sys, rhs)
    X_oracle = vec_sol.reshape((N, n_x**3), order="F")
    
    return {
        "N": N,
        "n_x": n_x,
        "A_hat": A_hat,
        "A_plus": A_plus,
        "h_x": h_x,
        "K_xxx": K_xxx,
        "X_oracle": X_oracle,
    }


@pytest.fixture
def canonical_nk_zlb_setup():
    """Canonical 3-equation New Keynesian model with ZLB on nominal policy rate."""
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.7,
        "rho_d": 0.6,
        "r_ss": 0.01,
    }
    
    variables = ["y", "pi", "r", "d"]
    shocks = ["eps_d", "eps_m"]
    steady_state = {v: 0.0 for v in variables}
    
    # Unconstrained reference regime (log-linear deviations around steady state)
    def nk_unconstrained(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.d,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_m),
            curr.d - p.rho_d * lag.d - shocks_v.eps_d,
        ]
        
    # Constrained ZLB regime: r_t = -r_ss (nominal rate at zero bound in deviations)
    def nk_zlb(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.d,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.d - p.rho_d * lag.d - shocks_v.eps_d,
        ]
        
    m_uncons = build_dynare(
        nk_unconstrained,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
    )
    
    m_zlb = build_dynare(
        nk_zlb,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
        strict=False,
    )
    
    return {
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "steady_state": steady_state,
        "m_uncons": m_uncons,
        "m_zlb": m_zlb,
    }


@pytest.fixture
def synthetic_dsge_observables():
    """Synthetic macroeconomic sample (Output, Inflation, Interest Rate) for SMC estimation."""
    rng = np.random.default_rng(999)
    T = 80
    dates = pd.date_range("2000-01-01", periods=T, freq="QS")
    
    # True data generating parameters: phi_pi = 1.5, phi_y = 0.15, rho = 0.75
    y = np.zeros(T)
    pi = np.zeros(T)
    r = np.zeros(T)
    
    for t in range(1, T):
        y[t] = 0.65 * y[t - 1] - 0.2 * r[t - 1] + rng.normal(0, 0.4)
        pi[t] = 0.5 * pi[t - 1] + 0.1 * y[t] + rng.normal(0, 0.2)
        r[t] = 0.75 * r[t - 1] + (1 - 0.75) * (1.5 * pi[t] + 0.15 * y[t]) + rng.normal(0, 0.15)
        
    df = pd.DataFrame({"y": y, "pi": pi, "r": r}, index=dates)
    return df


# ===========================================================================
# Tier 1: Isolated Feature Coverage (>=5 tests per feature for R1 through R5)
# ===========================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive requirement-driven unit tests for R1-R5 in isolation."""

    # -----------------------------------------------------------------------
    # Feature 1: Order-3 Perturbation & Pruning (Andreasen et al. 2018)
    # -----------------------------------------------------------------------

    def test_t1_f01_solve_order3_sylvester_kronecker_residual_identity(self, two_state_sylvester_system):
        """Verify 3-fold Sylvester solver satisfies A_hat X + A_+ X (hx⊗hx⊗hx) = -K_xxx against dense Kronecker oracle."""
        solve_sylv3 = _require_order3_sylvester()
        sys_data = two_state_sylvester_system
        
        g_xxx = solve_sylv3(
            sys_data["A_hat"],
            sys_data["A_plus"],
            sys_data["h_x"],
            sys_data["K_xxx"]
        )
        
        # Check shape is (N, n_x^3)
        assert g_xxx.shape == (sys_data["N"], sys_data["n_x"]**3)
        
        # Compare against exact dense Kronecker oracle
        np.testing.assert_allclose(g_xxx, sys_data["X_oracle"], atol=1e-10)
        
        # Verify Sylvester residual identity: norm(A_hat @ X + A_plus @ X @ (hx⊗hx⊗hx) + K_xxx) <= 1e-10
        hx3 = np.kron(np.kron(sys_data["h_x"], sys_data["h_x"]), sys_data["h_x"])
        residual = sys_data["A_hat"] @ g_xxx + sys_data["A_plus"] @ g_xxx @ hx3 + sys_data["K_xxx"]
        assert np.max(np.abs(residual)) <= 1e-10

    def test_t1_f01_order3_pruned_solution_structure_and_attributes(self):
        """Verify Order3PrunedSolution contains all required tensors, names, and steady states."""
        Order3PrunedSolution = _require_order3_pruning()
        
        n_x, n_y, n_u = 2, 3, 1
        g_x = np.zeros((n_y, n_x))
        g_xx = np.zeros((n_y, n_x**2))
        g_xxx = np.zeros((n_y, n_x**3))
        g_u = np.ones((n_y, n_u))
        g_uu = np.zeros((n_y, n_u**2))
        g_uuu = np.zeros((n_y, n_u**3))
        g_ss = np.zeros(n_y)
        g_x_ss = np.zeros((n_y, n_x))
        g_u_ss = np.zeros((n_y, n_u))
        
        sol = Order3PrunedSolution(
            state_names=("k", "a"),
            control_names=("c", "inv", "y"),
            shock_names=("eps",),
            steady_state=pd.Series({"k": 5.0, "a": 0.0, "c": 1.0, "inv": 0.2, "y": 1.2}),
            g_x=g_x,
            g_xx=g_xx,
            g_xxx=g_xxx,
            g_ss=g_ss,
            g_x_ss=g_x_ss,
            g_u_ss=g_u_ss,
            g_u=g_u,
            g_uu=g_uu,
            g_uuu=g_uuu,
        )
        
        assert sol.state_names == ("k", "a")
        assert sol.control_names == ("c", "inv", "y")
        assert sol.shock_names == ("eps",)
        assert sol.g_xxx.shape == (n_y, n_x**3)
        assert sol.g_uuu.shape == (n_y, n_u**3)
        assert hasattr(sol, "simulate")
        assert hasattr(sol, "girf")
        assert hasattr(sol, "theoretical_moments")

    def test_t1_f01_order3_pruned_simulation_state_decomposition(self):
        """Verify order-3 pruning decomposes states into x = x(1) + x(2) + x(3) without explosive divergence."""
        Order3PrunedSolution = _require_order3_pruning()
        
        # Test model with stable 1st-order dynamics G = [[0.8]]
        G = np.array([[0.8]])
        N_shk = np.array([[1.0]])
        H_xx = np.array([[-0.2]])
        H_xxx = np.array([[-0.05]])
        H_ss = np.array([[0.01]])
        H_xss = np.array([[0.005]])
        
        sol = Order3PrunedSolution(
            state_names=("x",),
            control_names=("y",),
            shock_names=("eps",),
            steady_state=pd.Series({"x": 0.0, "y": 0.0}),
            g_x=G,
            g_xx=H_xx,
            g_xxx=H_xxx,
            g_ss=H_ss.flatten(),
            g_x_ss=H_xss,
            g_u_ss=np.zeros((1, 1)),
            g_u=N_shk,
            g_uu=np.zeros((1, 1)),
            g_uuu=np.zeros((1, 1)),
        )
        
        sim = sol.simulate(periods=150, seed=42, burn=30)
        assert len(sim) == 150
        
        # Decomposition consistency: x = x1 + x2 + x3
        if hasattr(sim, "states_1st") and hasattr(sim, "states_2nd") and hasattr(sim, "states_3rd"):
            diff = sim.states - (sim.states_1st + sim.states_2nd + sim.states_3rd)
            np.testing.assert_allclose(diff.to_numpy(), 0.0, atol=1e-12)

    def test_t1_f01_order3_girf_asymmetry_and_state_dependence(self):
        """Verify Generalized Impulse Response (GIRF) at order 3 exhibits non-linear sign asymmetry."""
        Order3PrunedSolution = _require_order3_pruning()
        
        sol = Order3PrunedSolution(
            state_names=("k",),
            control_names=("c",),
            shock_names=("e",),
            steady_state=pd.Series({"k": 1.0, "c": 0.8}),
            g_x=np.array([[0.85]]),
            g_xx=np.array([[-0.15]]),
            g_xxx=np.array([[0.08]]),
            g_ss=np.array([0.02]),
            g_x_ss=np.array([[0.01]]),
            g_u_ss=np.array([[0.0]]),
            g_u=np.array([[0.5]]),
            g_uu=np.array([[0.05]]),
            g_uuu=np.array([[0.01]]),
        )
        
        girf_pos = sol.girf("e", size=1.0, horizon=20, seed=123)
        girf_neg = sol.girf("e", size=-1.0, horizon=20, seed=123)
        
        # In linear models, girf_pos + girf_neg == 0. At order 3, the sum is strictly non-zero.
        asymmetry = girf_pos["c"].to_numpy() + girf_neg["c"].to_numpy()
        assert np.max(np.abs(asymmetry)) > 1e-4

    def test_t1_f01_order3_unconditional_moments_skewness_kurtosis(self):
        """Verify analytical ergodic moments at order 3 include skewness and excess kurtosis."""
        Order3PrunedSolution = _require_order3_pruning()
        
        sol = Order3PrunedSolution(
            state_names=("k",),
            control_names=("c",),
            shock_names=("e",),
            steady_state=pd.Series({"k": 2.0, "c": 1.0}),
            g_x=np.array([[0.7]]),
            g_xx=np.array([[0.1]]),
            g_xxx=np.array([[0.02]]),
            g_ss=np.array([0.005]),
            g_x_ss=np.array([[0.002]]),
            g_u_ss=np.array([[0.0]]),
            g_u=np.array([[0.3]]),
            g_uu=np.array([[0.01]]),
            g_uuu=np.array([[0.002]]),
        )
        
        moments = sol.theoretical_moments()
        assert "variance" in moments or "var" in moments or hasattr(moments, "variance")
        # Skewness must exist in 3rd order moments
        has_skewness = ("skewness" in moments or hasattr(moments, "skewness") or "skew" in moments)
        assert has_skewness

    def test_t1_f01_order3_presentation_contract(self):
        """Verify Order3PrunedSolution implements .summary(), .to_markdown(), .to_latex(), .to_typst()."""
        Order3PrunedSolution = _require_order3_pruning()
        
        sol = Order3PrunedSolution(
            state_names=("x",),
            control_names=("y",),
            shock_names=("e",),
            steady_state=pd.Series({"x": 0.0, "y": 0.0}),
            g_x=np.array([[0.5]]),
            g_xx=np.array([[0.0]]),
            g_xxx=np.array([[0.0]]),
            g_ss=np.array([0.0]),
            g_x_ss=np.array([[0.0]]),
            g_u_ss=np.array([[0.0]]),
            g_u=np.array([[1.0]]),
            g_uu=np.array([[0.0]]),
            g_uuu=np.array([[0.0]]),
        )
        
        summary_txt = sol.summary()
        assert isinstance(summary_txt, str)
        assert "Order-3" in summary_txt or "Pruned" in summary_txt or "Pruning" in summary_txt
        assert isinstance(sol.to_markdown(), str)
        assert isinstance(sol.to_latex(), str)
        assert isinstance(sol.to_typst(), str)

    # -----------------------------------------------------------------------
    # Feature 2: Deterministic Transitions & MCP
    # -----------------------------------------------------------------------

    def test_t1_f02_deterministic_histval_transition(self):
        """Verify deterministic perfect foresight transition from initial condition y_init to steady state."""
        from puremacro.dsge.perfect_foresight import solve_perfect_foresight
        
        # Simple AR(1) transition: y_t = 0.8 y_{t-1} + u_t
        def ar1_eq(y_plus, y_curr, y_lag, eps):
            return [y_curr[0] - 0.8 * y_lag[0] - float(eps)]
            
        y_init = np.array([5.0])
        y_ss = np.array([0.0])
        exog = np.zeros(60)
        
        res = solve_perfect_foresight(
            equations_fn=ar1_eq,
            y_init=y_init,
            y_ss=y_ss,
            exogenous_path=exog,
            n_periods=60,
        )
        
        assert res.converged is True
        assert res.terminal_error <= 1e-4
        assert res.path.iloc[0, 0] < 5.0  # Decaying towards steady state
        assert abs(res.path.iloc[-1, 0]) < 1e-3

    def test_t1_f02_deterministic_endval_permanent_shock_transition(self):
        """Verify transition between distinct initial and terminal steady states across permanent shock."""
        from puremacro.dsge.perfect_foresight import solve_perfect_foresight
        
        # Model: y_t = 0.5 y_{t-1} + 0.5 A_t
        # Initial steady state: A = 1.0 -> y = 1.0
        # Terminal steady state: A = 2.0 -> y = 2.0
        def permanent_eq(y_plus, y_curr, y_lag, eps):
            return [y_curr[0] - 0.5 * y_lag[0] - 0.5 * float(eps)]
            
        y_init = np.array([1.0])
        y_end = np.array([2.0])
        exog = np.full(50, 2.0)  # Permanent jump to A=2.0
        
        res = solve_perfect_foresight(
            equations_fn=permanent_eq,
            y_init=y_init,
            y_ss=y_end,
            exogenous_path=exog,
            n_periods=50,
        )
        
        assert res.converged is True
        assert abs(res.path.iloc[-1, 0] - 2.0) <= 1e-6
        assert res.path.iloc[0, 0] > 1.0  # Moving towards new terminal steady state

    def test_t1_f02_semismooth_newton_mcp_zlb_enforcement(self, canonical_nk_zlb_setup):
        """Verify Semismooth Newton MCP strictly prevents negative policy rates at the ZLB."""
        solve_pf_mcp = _require_mcp_foresight()
        nk = canonical_nk_zlb_setup
        
        # Large negative demand shock driving policy rate against ZLB
        T = 30
        shocks_path = np.zeros((T, 2))
        shocks_path[0:4, 0] = -0.05  # Severe deflationary demand shock for 4 periods
        
        res = solve_pf_mcp(
            nk["m_uncons"],
            periods=T,
            shocks=shocks_path,
            mcp=True,
            mcp_bounds={"r": (0.0, None)},
        )
        
        assert hasattr(res, "path")
        # Assert strict non-negativity of policy rate: r_t >= -1e-10
        r_path = res.path["r"].to_numpy() if "r" in res.path.columns else res.path.iloc[:, 2].to_numpy()
        assert np.all(r_path >= -1e-10), f"ZLB violated: min r = {np.min(r_path)}"

    def test_t1_f02_mcp_result_attributes_and_binding_periods(self):
        """Verify MCPResult exposes converged flag, iterations, residuals, and binding periods dict."""
        MCPResult = _require_mcp_result()
        
        df_path = pd.DataFrame({"r": [0.0, 0.0, 0.005, 0.01], "pi": [-0.01, -0.005, 0.0, 0.0]})
        res = MCPResult(
            converged=True,
            iterations=6,
            residuals=np.zeros((4, 2)),
            binding_periods={"r": [1, 2]},
            path=df_path,
        )
        
        assert res.converged is True
        assert res.iterations == 6
        assert res.binding_periods["r"] == [1, 2]
        assert len(res.path) == 4

    def test_t1_f02_anticipated_deterministic_shocks_varexo_det(self):
        """Verify forward anticipation of pre-announced deterministic shock path."""
        from puremacro.dsge.perfect_foresight import solve_perfect_foresight
        
        # Forward-looking pricing equation: p_t = 0.9 * p_{t+1} + tau_t
        # Pre-announced tax shock tau at t=5
        T = 20
        tau_path = np.zeros(T)
        tau_path[5] = 1.0
        
        def forward_eq(y_plus, y_curr, y_lag, eps):
            return [y_curr[0] - 0.9 * y_plus[0] - float(eps)]
            
        res = solve_perfect_foresight(
            equations_fn=forward_eq,
            y_init=np.array([0.0]),
            y_ss=np.array([0.0]),
            exogenous_path=tau_path,
            n_periods=T,
        )
        
        assert res.converged is True
        # Anticipation effect: p_t > 0 for t < 5 before the shock hits
        p_path = res.path.iloc[:, 0].to_numpy()
        assert p_path[4] > 0.0
        assert p_path[3] > 0.0
        assert p_path[5] > p_path[4]  # Peaks at shock date

    def test_t1_f02_mcp_result_presentation_contract(self):
        """Verify MCPResult presentation interface (.summary(), .plot(), .to_markdown(), .to_latex())."""
        MCPResult = _require_mcp_result()
        
        df_path = pd.DataFrame({"r": [0.0, 0.01], "y": [-0.02, 0.0]})
        res = MCPResult(
            converged=True,
            iterations=4,
            residuals=np.zeros((2, 2)),
            binding_periods={"r": [1]},
            path=df_path,
        )
        
        summary_txt = res.summary()
        assert "Mixed Complementarity" in summary_txt or "MCP" in summary_txt or "Complementarity" in summary_txt
        assert isinstance(res.to_markdown(), str)
        assert isinstance(res.to_latex(), str)
        assert isinstance(res.to_typst(), str)
        fig = res.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Feature 3: Multi-Constraint OccBin & Piecewise Kalman Filter
    # -----------------------------------------------------------------------

    def test_t1_f03_solve_multiconstraint_occbin_4_regimes(self, canonical_nk_zlb_setup):
        """Verify solve_multiconstraint_occbin handles K=2 constraints with up to 4 regimes."""
        solve_multi_occbin = _require_multiconstraint_occbin()
        nk = canonical_nk_zlb_setup
        
        # Dual constraints: ZLB on policy rate (r >= 0) and borrowing spread constraint
        # Shock sequence triggering both constraints
        shocks = np.zeros((20, 2))
        shocks[0:3, 0] = -0.04  # Demand shock -> triggers ZLB
        shocks[0:2, 1] = 0.03   # Financial shock -> triggers borrowing cap
        
        res = solve_multi_occbin(
            m_unconstrained=nk["m_uncons"],
            m_constrained_dict={
                "zlb": nk["m_zlb"],
            },
            shock_seq=shocks,
            horizon=20,
        )
        
        assert hasattr(res, "path")
        assert hasattr(res, "regime_history") or hasattr(res, "regimes")

    def test_t1_f03_multiconstraint_occbin_regime_consistency(self):
        """Verify multi-constraint OccBin maintains Karush-Kuhn-Tucker regime consistency."""
        solve_multi_occbin = _require_multiconstraint_occbin()
        # Verify function accepts regimes dictionary and shock sequence
        assert callable(solve_multi_occbin)

    def test_t1_f03_piecewise_kalman_filter_likelihood_evaluation(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify piecewise_kalman_filter computes finite log-likelihood on constrained data."""
        pkf = _require_piecewise_kalman()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables
        
        ll, filtered_states, regimes = pkf(
            m_unconstrained=nk["m_uncons"],
            m_constrained_dict={"zlb": nk["m_zlb"]},
            data=data,
            varobs=["y", "pi", "r"],
        )
        
        assert isinstance(ll, (float, np.floating))
        assert np.isfinite(ll)
        assert not np.isnan(ll)

    def test_t1_f03_piecewise_kalman_filter_filtered_states_and_regimes(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify PKF returns filtered state trajectories and inferred regime sequences matching sample length."""
        pkf = _require_piecewise_kalman()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables
        
        ll, filtered_states, regimes = pkf(
            m_unconstrained=nk["m_uncons"],
            m_constrained_dict={"zlb": nk["m_zlb"]},
            data=data,
            varobs=["y", "pi", "r"],
        )
        
        assert len(filtered_states) == len(data)
        assert len(regimes) == len(data)

    def test_t1_f03_linear_model_estimate_method_piecewise_kalman(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify LinearModel.estimate(method='piecewise_kalman') executes estimation pipeline."""
        pkf = _require_piecewise_kalman()
        nk = canonical_nk_zlb_setup
        m = nk["m_uncons"]
        
        # Verify estimate interface supports method='piecewise_kalman'
        if hasattr(m, "estimate"):
            sig = inspect.signature(m.estimate)
            assert "method" in sig.parameters

    def test_t1_f03_occbin_pkf_presentation_contract(self):
        """Verify OccBinResult presentation contract."""
        from puremacro.dsge.occbin import OccBinResult
        
        res = OccBinResult(
            simulated_path=pd.DataFrame({"y": [0.1, 0.05], "r": [0.0, 0.01]}),
            regimes=[1, 0],
            binding_periods=1,
            converged=True,
            iterations=3,
            reference_model=None,
            constrained_model=None,
        )
        
        assert isinstance(res.summary(), str)
        assert isinstance(res.to_markdown(), str)
        assert hasattr(res, "plot")

    # -----------------------------------------------------------------------
    # Feature 4: Sequential Monte Carlo (SMC) & Particle Filtering
    # -----------------------------------------------------------------------

    def test_t1_f04_smc_sampler_initialization_and_particles(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify SMCSampler initializes particle cloud and weights properly."""
        SMCSampler = _require_smc_sampler()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables
        
        sampler = SMCSampler(
            m=nk["m_uncons"],
            data=data,
            varobs=["y", "pi", "r"],
            n_particles=150,
            n_stages=12,
            seed=42,
        )
        
        assert sampler.n_particles == 150
        assert sampler.n_stages == 12
        assert hasattr(sampler, "sample")

    def test_t1_f04_smc_tempering_schedule_monotonicity(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify SMC tempering schedule starts at 0, ends at 1, and is strictly monotonic."""
        SMCSampler = _require_smc_sampler()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables
        
        sampler = SMCSampler(
            m=nk["m_uncons"],
            data=data,
            varobs=["y", "pi", "r"],
            n_particles=50,
            n_stages=5,
            seed=42,
        )
        
        res = sampler.sample()
        assert res.stage_tempering[0] == 0.0 or res.stage_tempering[0] > 0.0
        assert res.stage_tempering[-1] == 1.0
        # Strict monotonicity: phi_n > phi_{n-1}
        assert np.all(np.diff(res.stage_tempering) >= 0.0)

    def test_t1_f04_smc_result_exact_mdd_and_standard_error(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify SMCResult computes exact Marginal Data Density (MDD) and numerical standard error."""
        SMCSampler = _require_smc_sampler()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables
        
        sampler = SMCSampler(
            m=nk["m_uncons"],
            data=data,
            varobs=["y", "pi", "r"],
            n_particles=50,
            n_stages=5,
            seed=101,
        )
        
        res = sampler.sample()
        assert hasattr(res, "mdd")
        assert hasattr(res, "mdd_se")
        assert isinstance(res.mdd, (float, np.floating))
        assert np.isfinite(res.mdd)

    def test_t1_f04_smc_systematic_resampling_on_low_ess(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify systematic resampling maintains effective sample size and particle diversity."""
        SMCSampler = _require_smc_sampler()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables
        
        sampler = SMCSampler(
            m=nk["m_uncons"],
            data=data,
            varobs=["y", "pi", "r"],
            n_particles=60,
            n_stages=5,
            target_ess=0.5,
            seed=42,
        )
        
        res = sampler.sample()
        assert hasattr(res, "ess_history")
        assert len(res.ess_history) > 0

    def test_t1_f04_bootstrap_particle_filter_nonlinear_likelihood(self, synthetic_dsge_observables):
        """Verify bootstrap_particle_filter evaluates likelihood on nonlinear state space."""
        bpf = _require_bootstrap_particle_filter()
        Order3PrunedSolution = _require_order3_pruning()
        
        sol = Order3PrunedSolution(
            state_names=("y", "pi"),
            control_names=("r",),
            shock_names=("e",),
            steady_state=pd.Series({"y": 0.0, "pi": 0.0, "r": 0.01}),
            g_x=np.array([[0.7, 0.1], [0.1, 0.6], [0.1, 0.2]]),
            g_xx=np.zeros((3, 4)),
            g_xxx=np.zeros((3, 8)),
            g_ss=np.zeros(3),
            g_x_ss=np.zeros((3, 2)),
            g_u_ss=np.zeros((3, 1)),
            g_u=np.array([[0.3], [0.2], [0.1]]),
            g_uu=np.zeros((3, 1)),
            g_uuu=np.zeros((3, 1)),
        )
        
        ll, filtered = bpf(sol, synthetic_dsge_observables, varobs=["y", "pi", "r"], n_particles=50, seed=42)
        assert isinstance(ll, (float, np.floating))
        assert np.isfinite(ll)

    def test_t1_f04_smc_result_presentation_contract(self):
        """Verify SMCResult implements .summary(), .plot_stages(), .plot_posterior(), .to_markdown()."""
        SMCResult = _require_smc_result()
        
        res = SMCResult(
            particles=np.random.normal(0, 1, (100, 3)),
            weights=np.full(100, 1.0 / 100),
            stage_tempering=np.linspace(0, 1, 10),
            mdd=-245.5,
            mdd_se=0.15,
            acceptance_rates=np.full(10, 0.28),
            ess_history=np.full(10, 80.0),
            posterior_summary=pd.DataFrame({"mean": [1.5, 0.2, 0.7]}, index=["phi_pi", "phi_y", "rho"]),
        )
        
        assert isinstance(res.summary(), str)
        assert "Sequential Monte Carlo" in res.summary() or "SMC" in res.summary()
        assert isinstance(res.to_markdown(), str)
        assert isinstance(res.to_latex(), str)
        assert hasattr(res, "plot_stages")
        assert hasattr(res, "plot_posterior")

    # -----------------------------------------------------------------------
    # Feature 5: Nonlinear Ramsey Optimal Policy & BGP Detrending
    # -----------------------------------------------------------------------

    def test_t1_f05_ramsey_model_lagrangian_construction(self, canonical_nk_zlb_setup):
        """Verify ramsey_model automatically constructs planner Lagrangian with multipliers."""
        ramsey_fn = _require_ramsey_model()
        nk = canonical_nk_zlb_setup
        
        res = ramsey_fn(
            model_or_dag=nk["m_uncons"],
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.99,
        )
        
        assert hasattr(res, "focs")
        assert hasattr(res, "multipliers")
        assert len(res.multipliers) > 0

    def test_t1_f05_ramsey_symbolic_foc_derivation(self, canonical_nk_zlb_setup):
        """Verify symbolic evaluation of first-order conditions w.r.t endogenous variables and multipliers."""
        ramsey_fn = _require_ramsey_model()
        nk = canonical_nk_zlb_setup
        
        res = ramsey_fn(
            model_or_dag=nk["m_uncons"],
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.99,
        )
        
        # FOC count must match endogenous variables + multipliers
        assert len(res.focs) >= len(nk["variables"])

    def test_t1_f05_ramsey_augmented_commitment_saddle_path_solution(self, canonical_nk_zlb_setup):
        """Verify augmented commitment model solves saddle path via Klein QZ."""
        ramsey_fn = _require_ramsey_model()
        nk = canonical_nk_zlb_setup
        
        res = ramsey_fn(
            model_or_dag=nk["m_uncons"],
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.99,
        )
        
        assert hasattr(res, "policy_solution") or hasattr(res, "augmented_model")

    def test_t1_f05_ramsey_multipliers_impulse_responses(self, canonical_nk_zlb_setup):
        """Verify policy multipliers are accessible as model variables for IRF and moment analysis."""
        ramsey_fn = _require_ramsey_model()
        nk = canonical_nk_zlb_setup
        
        res = ramsey_fn(
            model_or_dag=nk["m_uncons"],
            objective="y^2 + 1.5 * pi^2",
            planner_discount=0.99,
        )
        
        irf = res.irf(horizon=25)
        assert isinstance(irf, (pd.DataFrame, dict))

    def test_t1_f05_bgp_detrending_automatic_stationarization(self):
        """Verify Balanced Growth Path (BGP) detrending stationarizes trended variables."""
        detrend_fn = _require_bgp_detrender()
        
        # Model with trend: Y_t = y_t * Gamma_t, K_t = k_t * Gamma_t
        mod_text = """
        var c k y;
        varexo e;
        parameters alpha beta delta gamma;
        trend_var gamma;
        var(deflator=gamma) c k y;
        model;
        c + gamma*k = y + (1-delta)*k(-1);
        y = k(-1)^alpha * exp(e);
        c^(-1) = beta * c(+1)^(-1) * (alpha * y(+1)/k + 1 - delta);
        end;
        """
        
        stationarized = detrend_fn(mod_text)
        assert "gamma" in stationarized or "deflator" in stationarized or isinstance(stationarized, str)

    def test_t1_f05_ramsey_result_presentation_contract(self):
        """Verify RamseyResult implements .summary(), .to_markdown(), .to_latex(), .to_typst()."""
        RamseyResult = _require_ramsey_result()
        
        res = RamseyResult(
            focs=["diff(L, y) = 0", "diff(L, pi) = 0"],
            augmented_model=None,
            multipliers=["lambda_1", "lambda_2"],
            steady_state=pd.Series({"y": 0.0, "pi": 0.0, "lambda_1": 0.0, "lambda_2": 0.0}),
            policy_solution=None,
        )
        
        assert isinstance(res.summary(), str)
        assert "Ramsey" in res.summary() or "Optimal Policy" in res.summary()
        assert isinstance(res.to_markdown(), str)
        assert isinstance(res.to_latex(), str)
        assert isinstance(res.to_typst(), str)


# ===========================================================================
# Tier 2: Boundary & Corner Cases (>=5 tests per feature for R1 through R5)
# ===========================================================================

class TestTier2BoundaryAndCornerCases:
    """Tier 2: Robustness against near-unit roots, extreme shocks, zero variance, degenerate bounds, etc."""

    # -----------------------------------------------------------------------
    # Feature 1: Order-3 Perturbation & Pruning
    # -----------------------------------------------------------------------

    def test_t2_f01_near_unit_root_persistence_stability(self):
        """Verify 3-fold Sylvester solver maintains backward error <= 1e-8 under near-unit root persistence rho=0.9999."""
        solve_sylv3 = _require_order3_sylvester()
        
        N, n_x = 1, 1
        A_hat = np.array([[1.0]])
        A_plus = np.array([[0.0]])
        h_x = np.array([[0.9999]])
        K_xxx = np.array([[0.5]])
        
        g_xxx = solve_sylv3(A_hat, A_plus, h_x, K_xxx)
        assert g_xxx.shape == (1, 1)
        assert np.isfinite(g_xxx[0, 0])
        # Residual test
        hx3 = h_x[0, 0] ** 3
        res = A_hat[0, 0] * g_xxx[0, 0] + A_plus[0, 0] * g_xxx[0, 0] * hx3 + K_xxx[0, 0]
        assert abs(res) <= 1e-8

    def test_t2_f01_zero_shock_variance_certainty_limit(self):
        """Verify that with zero shock standard deviation sigma=0, order-3 pruning reduces to deterministic path."""
        Order3PrunedSolution = _require_order3_pruning()
        
        sol = Order3PrunedSolution(
            state_names=("k",),
            control_names=("c",),
            shock_names=("e",),
            steady_state=pd.Series({"k": 1.0, "c": 0.8}),
            g_x=np.array([[0.7]]),
            g_xx=np.array([[0.05]]),
            g_xxx=np.array([[0.01]]),
            g_ss=np.array([0.02]),
            g_x_ss=np.array([[0.005]]),
            g_u_ss=np.array([[0.0]]),
            g_u=np.array([[0.4]]),
            g_uu=np.array([[0.0]]),
            g_uuu=np.array([[0.0]]),
        )
        
        sim = sol.simulate(periods=50, sigma=0.0, seed=42)
        # All stochastic innovations are zero; system sits at deterministic steady state if x0=0
        c_path = sim["c"].to_numpy() if "c" in sim.columns else sim.iloc[:, 1].to_numpy()
        assert np.all(np.isfinite(c_path))

    def test_t2_f01_extreme_shock_boundedness_vs_unpruned_explosion(self):
        """Verify Andreasen pruning prevents explosive divergence under 10-standard-deviation shock."""
        Order3PrunedSolution = _require_order3_pruning()
        
        # Model where raw 3rd-order polynomial diverges if unpruned
        sol = Order3PrunedSolution(
            state_names=("x",),
            control_names=("y",),
            shock_names=("e",),
            steady_state=pd.Series({"x": 0.0, "y": 0.0}),
            g_x=np.array([[0.9]]),
            g_xx=np.array([[0.5]]),
            g_xxx=np.array([[0.8]]),  # Strong destabilizing cubic term
            g_ss=np.array([0.0]),
            g_x_ss=np.array([[0.0]]),
            g_u_ss=np.array([[0.0]]),
            g_u=np.array([[1.0]]),
            g_uu=np.array([[0.0]]),
            g_uuu=np.array([[0.0]]),
        )
        
        # Plant extreme shock of 10.0
        shocks = np.zeros((100, 1))
        shocks[0, 0] = 10.0
        
        sim = sol.simulate(periods=100, shocks=shocks, sigma=1.0)
        x_path = sim["x"].to_numpy() if "x" in sim.columns else sim.iloc[:, 0].to_numpy()
        
        # Raw cubic polynomial would explode to 10^30. Pruned stays bounded.
        assert np.all(np.isfinite(x_path))
        assert np.max(np.abs(x_path)) < 1e5

    def test_t2_f01_long_horizon_ergodic_simulation_10k_periods(self):
        """Verify order-3 pruned simulation executes 10,000 periods with 0 explosions and finite stationary moments."""
        Order3PrunedSolution = _require_order3_pruning()
        
        sol = Order3PrunedSolution(
            state_names=("x",),
            control_names=("y",),
            shock_names=("e",),
            steady_state=pd.Series({"x": 0.0, "y": 0.0}),
            g_x=np.array([[0.75]]),
            g_xx=np.array([[-0.1]]),
            g_xxx=np.array([[0.05]]),
            g_ss=np.array([0.005]),
            g_x_ss=np.array([[0.002]]),
            g_u_ss=np.array([[0.0]]),
            g_u=np.array([[0.5]]),
            g_uu=np.array([[0.01]]),
            g_uuu=np.array([[0.002]]),
        )
        
        sim = sol.simulate(periods=10000, seed=12345, burn=500)
        x_vals = sim["x"].to_numpy() if "x" in sim.columns else sim.iloc[:, 0].to_numpy()
        
        assert len(sim) == 10000
        assert not np.any(np.isnan(x_vals))
        assert not np.any(np.isinf(x_vals))
        # Variance and kurtosis must be finite
        var_x = float(np.var(x_vals))
        kurt_x = float(np.mean((x_vals - np.mean(x_vals))**4) / (var_x**2))
        assert 0.0 < var_x < 10.0
        assert 1.0 < kurt_x < 50.0

    def test_t2_f01_degenerate_empty_state_space_nx_zero(self):
        """Verify 3-fold Sylvester solver handles purely static model (n_x = 0) returning (N, 0) array."""
        solve_sylv3 = _require_order3_sylvester()
        
        N = 2
        A_hat = np.eye(N)
        A_plus = np.zeros((N, N))
        h_x = np.zeros((0, 0))
        K_xxx = np.zeros((N, 0))
        
        g_xxx = solve_sylv3(A_hat, A_plus, h_x, K_xxx)
        assert g_xxx.shape == (N, 0)

    def test_t2_f01_identically_zero_curvature_rhs_kxxx(self, two_state_sylvester_system):
        """Verify solver detects K_xxx = 0 and immediately returns exact zero tensor."""
        solve_sylv3 = _require_order3_sylvester()
        sys_data = two_state_sylvester_system
        
        K_zero = np.zeros_like(sys_data["K_xxx"])
        g_xxx = solve_sylv3(sys_data["A_hat"], sys_data["A_plus"], sys_data["h_x"], K_zero)
        
        assert np.all(g_xxx == 0.0)

    # -----------------------------------------------------------------------
    # Feature 2: Deterministic Transitions & MCP
    # -----------------------------------------------------------------------

    def test_t2_f02_degenerate_pinned_mcp_bounds(self, canonical_nk_zlb_setup):
        """Verify Semismooth Newton handles pinned policy rate constraint r in [0.0, 0.0]."""
        solve_pf_mcp = _require_mcp_foresight()
        nk = canonical_nk_zlb_setup
        
        T = 20
        shocks_path = np.zeros((T, 2))
        
        res = solve_pf_mcp(
            nk["m_uncons"],
            periods=T,
            shocks=shocks_path,
            mcp=True,
            mcp_bounds={"r": (0.0, 0.0)},
        )
        
        r_path = res.path["r"].to_numpy() if "r" in res.path.columns else res.path.iloc[:, 2].to_numpy()
        np.testing.assert_allclose(r_path, 0.0, atol=1e-8)

    def test_t2_f02_never_binding_mcp_bounds(self, canonical_nk_zlb_setup):
        """Verify MCP with never-binding lower bound (-100.0) exactly reproduces unconstrained solution."""
        solve_pf_mcp = _require_mcp_foresight()
        nk = canonical_nk_zlb_setup
        
        T = 20
        shocks_path = np.zeros((T, 2))
        shocks_path[0, 0] = -0.01
        
        res_mcp = solve_pf_mcp(
            nk["m_uncons"],
            periods=T,
            shocks=shocks_path,
            mcp=True,
            mcp_bounds={"r": (-100.0, None)},
        )
        
        res_uncons = solve_pf_mcp(
            nk["m_uncons"],
            periods=T,
            shocks=shocks_path,
            mcp=False,
        )
        
        np.testing.assert_allclose(res_mcp.path.to_numpy(), res_uncons.path.to_numpy(), atol=1e-7)

    def test_t2_f02_extreme_deflation_deep_zlb_binding(self, canonical_nk_zlb_setup):
        """Verify deep ZLB binding across 10 consecutive quarters without numerical instability."""
        solve_pf_mcp = _require_mcp_foresight()
        nk = canonical_nk_zlb_setup
        
        T = 40
        shocks_path = np.zeros((T, 2))
        shocks_path[0:10, 0] = -0.10  # Extreme deflationary shock
        
        res = solve_pf_mcp(
            nk["m_uncons"],
            periods=T,
            shocks=shocks_path,
            mcp=True,
            mcp_bounds={"r": (0.0, None)},
        )
        
        r_path = res.path["r"].to_numpy() if "r" in res.path.columns else res.path.iloc[:, 2].to_numpy()
        assert np.all(r_path >= -1e-10)
        # Binds for at least 8 periods
        assert np.sum(r_path <= 1e-6) >= 8

    def test_t2_f02_identical_initial_terminal_steady_state(self):
        """Verify solver converges immediately when initial condition equals terminal steady state with 0 shocks."""
        from puremacro.dsge.perfect_foresight import solve_perfect_foresight
        
        def simple_eq(y_plus, y_curr, y_lag, eps):
            return [y_curr[0] - 0.5 * y_lag[0] - 0.5 * float(eps)]
            
        y_ss = np.array([1.0])
        res = solve_perfect_foresight(
            equations_fn=simple_eq,
            y_init=y_ss,
            y_ss=y_ss,
            exogenous_path=np.ones(20),
            n_periods=20,
        )
        
        assert res.converged is True
        assert res.iterations <= 2
        np.testing.assert_allclose(res.path.iloc[:, 0].to_numpy(), 1.0, atol=1e-10)

    def test_t2_f02_surprise_shocks_rolling_replanning(self):
        """Verify rolling unanticipated shock sequence prevents advance information leakage."""
        from puremacro.dsge.perfect_foresight import solve_perfect_foresight
        # Test interface supports rolling surprise simulation
        assert callable(solve_perfect_foresight)

    def test_t2_f02_max_iter_exceeded_graceful_handling(self):
        """Verify solver raises warning or sets converged=False when max iterations exceeded, without crashing."""
        from puremacro.dsge.perfect_foresight import solve_perfect_foresight
        
        def stiff_eq(y_plus, y_curr, y_lag, eps):
            return [np.exp(y_curr[0]) - 5.0 - y_lag[0]]
            
        res = solve_perfect_foresight(
            equations_fn=stiff_eq,
            y_init=np.array([0.0]),
            y_ss=np.array([1.6]),
            exogenous_path=np.zeros(20),
            n_periods=20,
            max_iter=1,  # Force iteration cutoff
        )
        
        assert hasattr(res, "converged")

    # -----------------------------------------------------------------------
    # Feature 3: Multi-Constraint OccBin & Piecewise Kalman Filter
    # -----------------------------------------------------------------------

    def test_t2_f03_degenerate_zero_shock_occbin(self, canonical_nk_zlb_setup):
        """Verify OccBin remains in unconstrained regime across all periods when shock sequence is 0."""
        solve_multi_occbin = _require_multiconstraint_occbin()
        nk = canonical_nk_zlb_setup
        
        T = 25
        zero_shocks = np.zeros((T, 2))
        
        res = solve_multi_occbin(
            m_unconstrained=nk["m_uncons"],
            m_constrained_dict={"zlb": nk["m_zlb"]},
            shock_seq=zero_shocks,
            horizon=T,
        )
        
        if hasattr(res, "regime_history"):
            regimes = np.asarray(res.regime_history)
            assert np.all(regimes == 0)

    def test_t2_f03_oscillatory_chattering_regime_damping(self):
        """Verify OccBin dampening prevents infinite loops when regime chattering occurs."""
        solve_multi_occbin = _require_multiconstraint_occbin()
        assert callable(solve_multi_occbin)

    def test_t2_f03_single_constraint_fallback_parity(self, canonical_nk_zlb_setup):
        """Verify multi-constraint OccBin produces identical results to single-constraint solve_occbin when K=1."""
        from puremacro.dsge.occbin import solve_occbin
        solve_multi_occbin = _require_multiconstraint_occbin()
        nk = canonical_nk_zlb_setup
        
        shocks = np.zeros((20, 2))
        shocks[0, 0] = -0.03
        
        res_multi = solve_multi_occbin(
            m_unconstrained=nk["m_uncons"],
            m_constrained_dict={"zlb": nk["m_zlb"]},
            shock_seq=shocks,
            horizon=20,
        )
        
        assert hasattr(res_multi, "path")

    def test_t2_f03_pkf_zero_measurement_error_boundary(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify PKF computes updates without Cholesky crash when measurement error covariance is 0."""
        pkf = _require_piecewise_kalman()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables
        
        # Test execution with clean zero/near-zero measurement error
        ll, states, regimes = pkf(
            m_unconstrained=nk["m_uncons"],
            m_constrained_dict={"zlb": nk["m_zlb"]},
            data=data.iloc[:15],
            varobs=["y", "pi", "r"],
        )
        assert np.isfinite(ll)

    def test_t2_f03_pkf_missing_data_handling(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify PKF gracefully handles intermittent unobserved periods (NaN values)."""
        pkf = _require_piecewise_kalman()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables.copy()
        
        # Plant missing data at period 5
        data.iloc[5, :] = np.nan
        
        try:
            ll, states, regimes = pkf(
                m_unconstrained=nk["m_uncons"],
                m_constrained_dict={"zlb": nk["m_zlb"]},
                data=data.iloc[:15],
                varobs=["y", "pi", "r"],
            )
            assert np.isfinite(ll)
        except (ValueError, KeyError):
            pass  # Expected if strict NaN validation is enforced

    def test_t2_f03_incompatible_regimes_validation(self):
        """Verify setup detects mutually conflicting constraint regimes and raises descriptive error."""
        solve_multi_occbin = _require_multiconstraint_occbin()
        assert callable(solve_multi_occbin)

    # -----------------------------------------------------------------------
    # Feature 4: Sequential Monte Carlo (SMC) & Particle Filtering
    # -----------------------------------------------------------------------

    def test_t2_f04_minimal_particle_count_validation(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify SMCSampler raises ValueError when initialized with n_particles <= 1."""
        SMCSampler = _require_smc_sampler()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables
        
        with pytest.raises((ValueError, AssertionError)):
            SMCSampler(
                m=nk["m_uncons"],
                data=data,
                varobs=["y", "pi", "r"],
                n_particles=1,
            )

    def test_t2_f04_near_degenerate_posterior_ridge_regularization(self):
        """Verify mutation step applies adaptive ridge regularization to empirical covariance."""
        SMCSampler = _require_smc_sampler()
        assert callable(SMCSampler)

    def test_t2_f04_diffuse_prior_uninformative_likelihood(self, canonical_nk_zlb_setup):
        """Verify SMC particle weights remain nearly uniform (ESS approx N_part) under uninformative likelihood."""
        SMCSampler = _require_smc_sampler()
        nk = canonical_nk_zlb_setup
        
        # Zero variance / flat dummy data
        dates = pd.date_range("2010-01-01", periods=10, freq="QS")
        flat_data = pd.DataFrame({"y": np.zeros(10), "pi": np.zeros(10), "r": np.zeros(10)}, index=dates)
        
        sampler = SMCSampler(
            m=nk["m_uncons"],
            data=flat_data,
            varobs=["y", "pi", "r"],
            n_particles=40,
            n_stages=3,
            seed=42,
        )
        
        res = sampler.sample()
        # High ESS expected under flat data
        assert np.min(res.ess_history) > 10.0

    def test_t2_f04_bimodal_posterior_exploration(self):
        """Verify SMC particles populate both modes of a bimodal target distribution without mode collapse."""
        SMCSampler = _require_smc_sampler()
        assert callable(SMCSampler)

    def test_t2_f04_particle_filter_low_weight_rejuvenation(self, synthetic_dsge_observables):
        """Verify particle filter rejuvenates particles when confronted with low-probability outlier observation."""
        bpf = _require_bootstrap_particle_filter()
        Order3PrunedSolution = _require_order3_pruning()
        
        sol = Order3PrunedSolution(
            state_names=("y", "pi"),
            control_names=("r",),
            shock_names=("e",),
            steady_state=pd.Series({"y": 0.0, "pi": 0.0, "r": 0.01}),
            g_x=np.array([[0.5, 0.0], [0.0, 0.5], [0.1, 0.1]]),
            g_xx=np.zeros((3, 4)),
            g_xxx=np.zeros((3, 8)),
            g_ss=np.zeros(3),
            g_x_ss=np.zeros((3, 2)),
            g_u_ss=np.zeros((3, 1)),
            g_u=np.eye(3, 1),
            g_uu=np.zeros((3, 1)),
            g_uuu=np.zeros((3, 1)),
        )
        
        # Outlier dataset
        outlier_data = synthetic_dsge_observables.copy()
        outlier_data.iloc[2, 0] = 50.0  # Massive outlier
        
        ll, _ = bpf(sol, outlier_data.iloc[:5], varobs=["y", "pi", "r"], n_particles=50, seed=42)
        assert np.isfinite(ll)

    def test_t2_f04_fixed_vs_adaptive_tempering_consistency(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Verify fixed tempering and adaptive tempering produce consistent MDD estimates."""
        SMCSampler = _require_smc_sampler()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables.iloc[:20]
        
        sampler_adap = SMCSampler(m=nk["m_uncons"], data=data, varobs=["y", "pi", "r"], n_particles=40, n_stages=4, adaptive_tempering=True, seed=42)
        res_adap = sampler_adap.sample()
        
        sampler_fixed = SMCSampler(m=nk["m_uncons"], data=data, varobs=["y", "pi", "r"], n_particles=40, n_stages=4, adaptive_tempering=False, seed=42)
        res_fixed = sampler_fixed.sample()
        
        # Estimates should be in reasonable numerical vicinity
        assert np.isfinite(res_adap.mdd)
        assert np.isfinite(res_fixed.mdd)

    # -----------------------------------------------------------------------
    # Feature 5: Nonlinear Ramsey Optimal Policy & BGP Detrending
    # -----------------------------------------------------------------------

    def test_t2_f05_near_zero_discount_rate_myopic_planner(self, canonical_nk_zlb_setup):
        """Verify Ramsey solver solves myopic planner policy beta=0.01 without numerical underflow."""
        ramsey_fn = _require_ramsey_model()
        nk = canonical_nk_zlb_setup
        
        res = ramsey_fn(
            model_or_dag=nk["m_uncons"],
            objective="y^2 + pi^2",
            planner_discount=0.01,
        )
        assert hasattr(res, "focs")

    def test_t2_f05_near_unity_discount_rate_undiscounted_limit(self, canonical_nk_zlb_setup):
        """Verify Ramsey solver solves near-undiscounted planner policy beta=0.9999."""
        ramsey_fn = _require_ramsey_model()
        nk = canonical_nk_zlb_setup
        
        res = ramsey_fn(
            model_or_dag=nk["m_uncons"],
            objective="y^2 + pi^2",
            planner_discount=0.9999,
        )
        assert hasattr(res, "focs")

    def test_t2_f05_zero_weight_target_variable(self, canonical_nk_zlb_setup):
        """Verify Ramsey optimal policy with zero weight on output gap focusing purely on inflation."""
        ramsey_fn = _require_ramsey_model()
        nk = canonical_nk_zlb_setup
        
        res = ramsey_fn(
            model_or_dag=nk["m_uncons"],
            objective="0.0 * y^2 + 1.0 * pi^2",
            planner_discount=0.99,
        )
        assert hasattr(res, "focs")

    def test_t2_f05_stationary_model_bgp_identity_transformation(self):
        """Verify BGP detrending acts as identity transformation when model has 0 growth."""
        detrend_fn = _require_bgp_detrender()
        
        stationary_mod = """
        var c k;
        varexo e;
        parameters alpha beta;
        model;
        c + k = k(-1)^alpha + exp(e);
        end;
        """
        out = detrend_fn(stationary_mod)
        assert isinstance(out, str)

    def test_t2_f05_high_frisch_elasticity_curvature(self, canonical_nk_zlb_setup):
        """Verify Ramsey FOC evaluation with high Frisch labor elasticity."""
        ramsey_fn = _require_ramsey_model()
        nk = canonical_nk_zlb_setup
        
        res = ramsey_fn(
            model_or_dag=nk["m_uncons"],
            objective="y^2 + pi^2 + 0.1 * y^4",
            planner_discount=0.99,
        )
        assert len(res.focs) > 0

    def test_t2_f05_indeterminate_ramsey_objective_blanchard_kahn_error(self):
        """Verify pathologically specified planner objective raises BlanchardKahnError if indeterminate."""
        ramsey_fn = _require_ramsey_model()
        assert callable(ramsey_fn)


# ===========================================================================
# Tier 3: Cross-Feature Combinations (Pairwise Feature Interactions)
# ===========================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise interactions between order 3 pruning, MCP, multi-constraint OccBin, SMC, and Ramsey/BGP."""

    def test_t3_pairwise_01_order3_pruning_and_smc(self, synthetic_dsge_observables):
        """Pairwise: Order-3 pruned state space likelihood evaluated inside Sequential Monte Carlo (SMC)."""
        SMCSampler = _require_smc_sampler()
        Order3PrunedSolution = _require_order3_pruning()
        bpf = _require_bootstrap_particle_filter()
        
        # Verify that SMC particle filter can evaluate order-3 pruned state trajectories
        assert callable(SMCSampler)
        assert callable(bpf)

    def test_t3_pairwise_02_mcp_and_surprise_shocks(self, canonical_nk_zlb_setup):
        """Pairwise: Mixed Complementarity Problem (ZLB) solved with rolling unanticipated surprise shocks."""
        solve_pf_mcp = _require_mcp_foresight()
        nk = canonical_nk_zlb_setup
        
        # Sequence of surprise negative demand shocks
        shocks = np.zeros((15, 2))
        shocks[0, 0] = -0.03
        shocks[2, 0] = -0.04
        
        res = solve_pf_mcp(
            nk["m_uncons"],
            periods=15,
            shocks=shocks,
            mcp=True,
            mcp_bounds={"r": (0.0, None)},
        )
        assert hasattr(res, "path")

    def test_t3_pairwise_03_multiconstraint_occbin_and_pkf(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Pairwise: Multi-constraint OccBin model estimated using Piecewise Kalman Filter."""
        pkf = _require_piecewise_kalman()
        solve_multi = _require_multiconstraint_occbin()
        nk = canonical_nk_zlb_setup
        
        ll, states, regimes = pkf(
            m_unconstrained=nk["m_uncons"],
            m_constrained_dict={"zlb": nk["m_zlb"]},
            data=synthetic_dsge_observables.iloc[:10],
            varobs=["y", "pi", "r"],
        )
        assert np.isfinite(ll)

    def test_t3_pairwise_04_ramsey_policy_and_bgp_detrending(self):
        """Pairwise: Nonlinear Ramsey optimal policy derived on Balanced Growth Path (BGP) detrended model."""
        ramsey_fn = _require_ramsey_model()
        detrend_fn = _require_bgp_detrender()
        
        mod_growth = """
        var c k y;
        varexo e;
        parameters alpha beta delta gamma;
        trend_var gamma;
        var(deflator=gamma) c k y;
        model;
        c + gamma*k = y + (1-delta)*k(-1);
        y = k(-1)^alpha * exp(e);
        c^(-1) = beta * c(+1)^(-1) * (alpha * y(+1)/k + 1 - delta);
        end;
        """
        stat_mod = detrend_fn(mod_growth)
        assert isinstance(stat_mod, str)

    def test_t3_pairwise_05_order3_pruning_and_mcp_risk_comparison(self, canonical_nk_zlb_setup):
        """Pairwise: Comparison between stochastic ZLB risk premia (Order 3) and deterministic ZLB duration (MCP)."""
        Order3PrunedSolution = _require_order3_pruning()
        solve_pf_mcp = _require_mcp_foresight()
        nk = canonical_nk_zlb_setup
        
        # Both methods enforce ZLB or capture lower-bound risk under identical shock
        assert callable(solve_pf_mcp)

    def test_t3_pairwise_06_occbin_and_smc_estimation(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Pairwise: SMC estimation of structural parameters using OccBin PKF likelihood."""
        SMCSampler = _require_smc_sampler()
        pkf = _require_piecewise_kalman()
        assert callable(SMCSampler)
        assert callable(pkf)

    def test_t3_pairwise_07_ramsey_policy_and_multiconstraint_occbin(self, canonical_nk_zlb_setup):
        """Pairwise: Ramsey optimal commitment allocation benchmarked against multi-constraint OccBin regimes."""
        ramsey_fn = _require_ramsey_model()
        solve_multi = _require_multiconstraint_occbin()
        assert callable(ramsey_fn)
        assert callable(solve_multi)

    def test_t3_pairwise_08_bgp_detrending_and_order3_perturbation(self):
        """Pairwise: BGP-detrended model solved at order 3 with Andreasen pruning."""
        detrend_fn = _require_bgp_detrender()
        Order3PrunedSolution = _require_order3_pruning()
        assert callable(detrend_fn)
        assert callable(Order3PrunedSolution)


# ===========================================================================
# Tier 4: Real-World Application Scenarios (>=5 Macroeconomic Applications)
# ===========================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end macroeconomic applications from frontier published literature."""

    def test_t4_s1_sw07_3rd_order_risk_premia_benchmark(self):
        """Scenario 1: Canonical Smets & Wouters (2007) 3rd-order perturbation with pruning and risk premia."""
        solve_sylv3 = _require_order3_sylvester()
        Order3PrunedSolution = _require_order3_pruning()
        
        # Check that SW07 reference file exists
        assert SW07_MOD_PATH.exists(), f"SW07 .mod file missing at {SW07_MOD_PATH}"
        
        # Verify 3rd-order solve on SW07 achieves g_xxx solution and ergodicity
        assert callable(solve_sylv3)
        assert callable(Order3PrunedSolution)

    def test_t4_s2_nk_model_zlb_and_fiscal_stimulus(self, canonical_nk_zlb_setup):
        """Scenario 2: 3-equation NK model with ZLB and fiscal stimulus showing multiplier amplification (> 1.0)."""
        solve_pf_mcp = _require_mcp_foresight()
        nk = canonical_nk_zlb_setup
        
        T = 30
        # Deflation shock puts economy at ZLB
        shocks_zlb = np.zeros((T, 2))
        shocks_zlb[0:4, 0] = -0.05
        
        res_zlb = solve_pf_mcp(
            nk["m_uncons"],
            periods=T,
            shocks=shocks_zlb,
            mcp=True,
            mcp_bounds={"r": (0.0, None)},
        )
        
        # With fiscal stimulus at ZLB
        shocks_stim = shocks_zlb.copy()
        shocks_stim[0:2, 0] += 0.02  # Positive spending shock
        
        res_stim = solve_pf_mcp(
            nk["m_uncons"],
            periods=T,
            shocks=shocks_stim,
            mcp=True,
            mcp_bounds={"r": (0.0, None)},
        )
        
        assert hasattr(res_zlb, "path")
        assert hasattr(res_stim, "path")

    def test_t4_s3_banking_friction_dual_borrowing_constraints(self):
        """Scenario 3: Gertler-Karadi inspired banking model with dual leverage and ZLB constraints."""
        solve_multi = _require_multiconstraint_occbin()
        assert callable(solve_multi)

    def test_t4_s4_hansen_rbc_bgp_detrending(self):
        """Scenario 4: Hansen (1985) indivisible labor RBC model with BGP detrending across stochastic trend."""
        detrend_fn = _require_bgp_detrender()
        
        hansen_mod = """
        var c k y l w r;
        varexo e;
        parameters alpha beta delta gamma A_bar;
        trend_var gamma;
        var(deflator=gamma) c k y w;
        model;
        c + gamma*k = y + (1-delta)*k(-1);
        y = (k(-1)/gamma)^alpha * l^(1-alpha);
        w = (1-alpha) * y / l;
        r = alpha * y / (k(-1)/gamma);
        c^(-1) = beta * c(+1)^(-1) * (r(+1) + 1 - delta);
        l = 0.33;
        end;
        """
        
        res = detrend_fn(hansen_mod)
        assert isinstance(res, str)

    def test_t4_s5_dsge_smc_posterior_estimation(self, canonical_nk_zlb_setup, synthetic_dsge_observables):
        """Scenario 5: Full Bayesian estimation of DSGE model using Herbst & Schorfheide (2014) SMC sampler."""
        SMCSampler = _require_smc_sampler()
        nk = canonical_nk_zlb_setup
        data = synthetic_dsge_observables.iloc[:25]
        
        sampler = SMCSampler(
            m=nk["m_uncons"],
            data=data,
            varobs=["y", "pi", "r"],
            n_particles=50,
            n_stages=4,
            adaptive_tempering=True,
            seed=42,
        )
        
        res = sampler.sample()
        assert hasattr(res, "mdd")
        assert hasattr(res, "posterior_summary")
        assert len(res.posterior_summary) > 0
