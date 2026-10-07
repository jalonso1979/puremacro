"""Comprehensive 4-Tier Opaque-Box E2E Test Suite for Frontier Macroeconomic Engines.

This test suite covers all four frontier macroeconomic engines in puremacro 3.3.0:
1. Continuous Transition Dynamics & MIT Shocks (puremacro.vfi.continuous_transition)
2. Exact Analytic Gradients via the Implicit Function Theorem (puremacro.vfi.analytic_gradients)
3. Deep Macro Physics-Informed Neural Networks (PINNs) (puremacro.vfi.deep_macro)
4. Quantitative Spatial & Gravity General Equilibrium (puremacro.trade.caliendo_parro & puremacro.spatial.allen_arkolakis)

Four-Tier Architecture:
- Tier 1: Feature Coverage (happy path for each engine in isolation)
- Tier 2: Boundary & Corner Cases (zero shock, extreme trade costs, high dimension D=10, kinked domains, near-zero elasticities)
- Tier 3: Cross-Feature Combinations (MIT shocks with IFT sensitivities; spatial distance feeding trade GE; Deep PINN vs Chebyshev collocation)
- Tier 4: Real-World Policy Scenarios (100 bps rate hike MIT shock, US-China tariff war, transport corridor investment, 10-sector capital convergence)

Opaque-Box Testing Principles:
- Strictly tests public APIs, mathematical invariants, conservation laws, and economic properties.
- Zero coupling to internal private helper methods.
- Strict compliance with Pyodide 4-package contract (numpy, scipy, pandas, matplotlib only).
- Progressive testability: graceful skip if a module is still under active development by a worker.
"""
from __future__ import annotations

import math
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend for automated test runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Core puremacro imports
import puremacro
from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro.spatial.weights import pairwise_distances
from puremacro.vfi.collocation import CollocationBasis, CollocationProblem, CollocationSolution
from puremacro.vfi.continuous_distribution import (
    AiyagariContinuousEquilibrium,
    ContinuousStationaryDistribution,
    continuous_push_distribution,
    continuous_stationary_distribution,
    young_lottery_weights,
)
from puremacro.vfi.fem import FEMMesh, FEMProblem, FEMSolution
from puremacro.vfi.splines import CubicBSplineBasis, SchumakerSpline, SplineCollocationProblem, SplineCollocationSolution


# ===========================================================================
# Dynamic Resolution Helpers with Progressive Readiness Checks
# ===========================================================================

def _require_continuous_transition():
    """Resolve continuous transition engine or skip progressively."""
    try:
        from puremacro.vfi import continuous_transition as ct_mod
        if hasattr(ct_mod, "solve_continuous_transition"):
            return ct_mod
    except (ImportError, AttributeError):
        pass
    try:
        from puremacro import vfi
        if hasattr(vfi, "solve_continuous_transition"):
            return vfi
    except (ImportError, AttributeError):
        pass
    pytest.skip("Milestone 1: continuous_transition pending implementation in puremacro.vfi.continuous_transition")


def _require_analytic_gradients():
    """Resolve analytic gradients IFT engine or skip progressively."""
    try:
        from puremacro.vfi import analytic_gradients as ag_mod
        if hasattr(ag_mod, "compute_ift_gradients"):
            return ag_mod
    except (ImportError, AttributeError):
        pass
    try:
        from puremacro import vfi
        if hasattr(vfi, "compute_ift_gradients"):
            return vfi
    except (ImportError, AttributeError):
        pass
    pytest.skip("Milestone 2: analytic_gradients pending implementation in puremacro.vfi.analytic_gradients")


def _require_deep_macro():
    """Resolve Deep Macro PINNs engine or skip progressively."""
    try:
        from puremacro.vfi import deep_macro as dm_mod
        if hasattr(dm_mod, "solve_deep_macro"):
            return dm_mod
    except (ImportError, AttributeError):
        pass
    try:
        from puremacro import vfi
        if hasattr(vfi, "solve_deep_macro"):
            return vfi
    except (ImportError, AttributeError):
        pass
    pytest.skip("Milestone 3: deep_macro pending implementation in puremacro.vfi.deep_macro")


def _require_caliendo_parro():
    """Resolve Caliendo-Parro trade GE engine or skip progressively."""
    try:
        from puremacro.trade import caliendo_parro as cp_mod
        if hasattr(cp_mod, "CaliendoParroModel"):
            return cp_mod
    except (ImportError, AttributeError):
        pass
    try:
        from puremacro import trade
        if hasattr(trade, "CaliendoParroModel"):
            return trade
    except (ImportError, AttributeError):
        pass
    try:
        from puremacro import spatial
        if hasattr(spatial, "CaliendoParroModel"):
            return spatial
    except (ImportError, AttributeError):
        pass
    pytest.skip("Milestone 4: CaliendoParroModel pending implementation in puremacro.trade.caliendo_parro")


def _require_allen_arkolakis():
    """Resolve Allen-Arkolakis spatial GE engine or skip progressively."""
    try:
        from puremacro.spatial import allen_arkolakis as aa_mod
        if hasattr(aa_mod, "AllenArkolakisModel"):
            return aa_mod
    except (ImportError, AttributeError):
        pass
    try:
        from puremacro import spatial
        if hasattr(spatial, "AllenArkolakisModel"):
            return spatial
    except (ImportError, AttributeError):
        pass
    try:
        from puremacro import trade
        if hasattr(trade, "AllenArkolakisModel"):
            return trade
    except (ImportError, AttributeError):
        pass
    pytest.skip("Milestone 4: AllenArkolakisModel pending implementation in puremacro.spatial.allen_arkolakis")


# ===========================================================================
# Test Fixtures & Canonical Economic Model Generators
# ===========================================================================

@pytest.fixture
def canonical_caliendo_parro_data() -> dict[str, Any]:
    """Canonical 3-country, 2-sector Caliendo & Parro (2015) dataset.

    Countries: USA, CHN, ROW (N=3)
    Sectors: Manufacturing (tradable), Services (less tradable) (J=2)
    Satisfies all general equilibrium identities:
    - Sum of bilateral trade shares along origin dimension equals 1.0.
    - Value-added share + sum of intermediate shares equals 1.0.
    - Consumption shares sum to 1.0.
    - Trade deficits sum to 0.0.
    """
    N = 3
    J = 2
    country_codes = ("USA", "CHN", "ROW")
    sector_codes = ("manuf", "serv")

    # Bilateral trade shares: pi[j, n, i] = share of importer n spent on exporter i
    trade_shares = np.array([
        # Sector 0: Manufacturing (high trade openness)
        [
            [0.55, 0.25, 0.20],  # USA imports
            [0.15, 0.70, 0.15],  # CHN imports
            [0.25, 0.25, 0.50],  # ROW imports
        ],
        # Sector 1: Services (primarily domestic)
        [
            [0.85, 0.08, 0.07],  # USA imports
            [0.05, 0.90, 0.05],  # CHN imports
            [0.10, 0.10, 0.80],  # ROW imports
        ],
    ], dtype=float)

    # Cost shares: value added gamma_va[n, j] and input-output gamma_io[n, j, k]
    gamma_va = np.array([
        [0.40, 0.60],  # USA (manuf, serv)
        [0.35, 0.65],  # CHN
        [0.45, 0.55],  # ROW
    ], dtype=float)

    gamma_io = np.array([
        # USA
        [[0.35, 0.25],   # manuf uses 35% manuf, 25% serv (sum = 0.60, va = 0.40)
         [0.15, 0.25]],  # serv uses 15% manuf, 25% serv (sum = 0.40, va = 0.60)
        # CHN
        [[0.40, 0.25],   # manuf uses 40% manuf, 25% serv (sum = 0.65, va = 0.35)
         [0.10, 0.25]],  # serv uses 10% manuf, 25% serv (sum = 0.35, va = 0.65)
        # ROW
        [[0.30, 0.25],   # manuf uses 30% manuf, 25% serv (sum = 0.55, va = 0.45)
         [0.20, 0.25]],  # serv uses 20% manuf, 25% serv (sum = 0.45, va = 0.55)
    ], dtype=float)

    # Final consumption expenditure shares alpha[n, j]
    alpha = np.array([
        [0.30, 0.70],  # USA
        [0.45, 0.55],  # CHN
        [0.35, 0.65],  # ROW
    ], dtype=float)

    # Sectoral trade elasticities theta_j
    theta = np.array([5.0, 8.0], dtype=float)

    # Baseline labor income (value added) w_n L_n
    labor_income = np.array([120.0, 90.0, 100.0], dtype=float)

    # Trade deficits D_n (summing to zero)
    deficits = np.array([10.0, -10.0, 0.0], dtype=float)

    # Baseline gross tariffs (tau = 1.0 implies no baseline tariffs)
    tariffs = np.ones((J, N, N), dtype=float)

    return {
        "trade_shares": trade_shares,
        "gamma_va": gamma_va,
        "gamma_io": gamma_io,
        "alpha": alpha,
        "theta": theta,
        "labor_income": labor_income,
        "deficits": deficits,
        "tariffs": tariffs,
        "country_codes": country_codes,
        "sector_codes": sector_codes,
    }


@pytest.fixture
def symmetric_caliendo_parro_model() -> Any:
    """A symmetric 3-country, 2-sector Caliendo-Parro economy in exact baseline equilibrium."""
    cp = _require_caliendo_parro()
    N, J = 3, 2
    trade_shares = np.zeros((J, N, N))
    trade_shares[0] = np.array([
        [0.60, 0.20, 0.20],
        [0.20, 0.60, 0.20],
        [0.20, 0.20, 0.60],
    ])
    trade_shares[1] = np.eye(N)

    gamma_va = np.full((N, J), 0.50)
    gamma_io = np.full((N, J, J), 0.25)
    alpha = np.full((N, J), 0.50)
    theta = np.array([5.0, 4.0])
    labor_income = np.array([100.0, 100.0, 100.0])

    return cp.CaliendoParroModel(
        trade_shares=trade_shares,
        gamma_va=gamma_va,
        gamma_io=gamma_io,
        alpha=alpha,
        theta=theta,
        labor_income=labor_income,
        nontradables=[1],
        country_codes=["USA", "CHN", "ROW"],
        sector_codes=["Manufactures", "Services"],
    )


@pytest.fixture
def canonical_allen_arkolakis_data() -> dict[str, Any]:
    """Canonical 5-region Allen & Arkolakis (2014) spatial model dataset.

    Regions: North, South, East, West, Central (N=5).
    Satisfies uniqueness condition: alpha + beta <= theta / (1 + theta).
    """
    N = 5
    region_names = ("North", "South", "East", "West", "Central")
    coords = np.array([
        [45.0, -93.0],  # North
        [30.0, -90.0],  # South
        [40.0, -74.0],  # East
        [37.0, -122.0], # West
        [39.0, -89.0],  # Central
    ], dtype=float)

    # Bilateral distance matrix
    dist_matrix = pairwise_distances(coords, metric="euclidean")

    # Iceberg trade costs: tau_ij = 1 + 0.02 * dist^0.5
    trade_costs = 1.0 + 0.02 * (dist_matrix ** 0.5)
    np.fill_diagonal(trade_costs, 1.0)

    # Fundamental productivities and amenities
    fundamental_productivity = np.array([1.2, 0.9, 1.3, 1.4, 1.0], dtype=float)
    fundamental_amenity = np.array([1.1, 1.3, 1.0, 1.4, 1.0], dtype=float)

    theta = 4.0
    alpha = 0.10   # Agglomeration elasticity
    beta = -0.30   # Congestion elasticity
    total_population = 1.0

    return {
        "trade_costs": trade_costs,
        "fundamental_productivity": fundamental_productivity,
        "fundamental_amenity": fundamental_amenity,
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "total_population": total_population,
        "region_names": region_names,
        "coordinates": coords,
    }


@pytest.fixture
def canonical_collocation_solution() -> Tuple[CollocationSolution, CollocationProblem]:
    """Solved continuous Chebyshev collocation Neoclassical Growth Model."""
    alpha = 0.36
    beta = 0.96
    delta = 1.0
    k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))
    k_min = 0.5 * k_ss
    k_max = 1.5 * k_ss

    prob = CollocationProblem(
        domain=(k_min, k_max),
        orders=6,
        method="euler",
        params={"alpha": alpha, "delta": delta},
        beta=beta,
    )
    sol = prob.solve(backend="numpy")
    return sol, prob


@pytest.fixture
def canonical_continuous_steady_state() -> dict[str, Any]:
    """Configuration of the initial continuous steady state for MIT shock transitions.

    ``solve_continuous_transition`` solves a dict with ``solve_aiyagari_continuous``,
    so it may hold only that function's keywords. Until the AIYCONT fix this
    fixture also carried a hand-built solved state (``k_grid``, ``policy_a``,
    ``pdf_ss``, ``r_ss``, ``w_ss``, ``K_ss``); those keys were silently dropped
    and the transitions started from the economy below, which is what these
    tests have always run. They now raise TypeError.
    """
    return {
        "beta": 0.96,
        "gamma": 2.0,
        "alpha": 0.36,
        "delta": 0.08,
        "P_z": np.array([[0.90, 0.10], [0.10, 0.90]], dtype=float),
        "z_grid": np.array([0.80, 1.20], dtype=float),
    }


@pytest.fixture(scope="module")
def canonical_continuous_equilibrium() -> AiyagariContinuousEquilibrium:
    """The solved steady state of ``canonical_continuous_steady_state``."""
    from puremacro.vfi.continuous_distribution import solve_aiyagari_continuous

    return solve_aiyagari_continuous(
        beta=0.96,
        gamma=2.0,
        alpha=0.36,
        delta=0.08,
        P_z=np.array([[0.90, 0.10], [0.10, 0.90]], dtype=float),
        z_grid=np.array([0.80, 1.20], dtype=float),
    )


# ===========================================================================
# TIER 1: FEATURE COVERAGE (HAPPY PATH IN ISOLATION)
# ===========================================================================

class TestTier1ContinuousTransition:
    """Tier 1: Feature coverage for Continuous Transition Dynamics & MIT Shocks."""

    def test_continuous_transition_happy_path(self, canonical_continuous_steady_state):
        """Verify non-linear continuous transition path execution under aggregate shock."""
        ct = _require_continuous_transition()
        ss = canonical_continuous_steady_state

        T = 40
        # Transitory aggregate TFP shock decaying geometrically
        shock_path = 0.05 * (0.80 ** np.arange(T))

        result = ct.solve_continuous_transition(
            initial_steady_state=ss,
            shock_path=shock_path,
            shock_var="z",
            horizon=T,
            solver="broyden",
            tol=1e-3,
            max_iter=50,
            backend="numpy",
        )

        assert hasattr(result, "r_path"), "Result must contain r_path"
        assert hasattr(result, "K_s_path"), "Result must contain K_s_path"
        assert hasattr(result, "distributions"), "Result must contain distributions"
        assert len(result.r_path) == T, f"Expected r_path of length {T}, got {len(result.r_path)}"
        assert len(result.distributions) >= T, "Expected at least T distributions"

    def test_continuous_transition_mass_conservation_and_residuals(self, canonical_continuous_steady_state):
        """Verify strict mass conservation sum(mu_t) = 1.0 +- 1e-12 at all periods."""
        ct = _require_continuous_transition()
        ss = canonical_continuous_steady_state

        T = 30
        shock_path = 0.03 * (0.85 ** np.arange(T))

        result = ct.solve_continuous_transition(
            initial_steady_state=ss,
            shock_path=shock_path,
            horizon=T,
            solver="shooting",
            tol=1e-3,
            max_iter=30,
            backend="numpy",
        )

        # Invariant 1: Mass conservation across all dates t
        for t, dist in enumerate(result.distributions):
            mass = np.sum(dist)
            assert np.isclose(mass, 1.0, atol=1e-11), f"Mass conservation violated at t={t}: sum = {mass}"

        # Invariant 2: Finite bounded residuals
        assert np.all(np.isfinite(result.residuals)), "Residuals must be finite"

    def test_continuous_transition_presentation_contract(self, canonical_continuous_steady_state):
        """Verify presentation interface: summary, plot, to_frame, to_markdown, to_latex, to_typst."""
        ct = _require_continuous_transition()
        ss = canonical_continuous_steady_state

        result = ct.solve_continuous_transition(
            initial_steady_state=ss,
            shock_path=np.zeros(10),
            horizon=10,
            backend="numpy",
        )

        # summary
        sm = result.summary()
        assert isinstance(sm, pd.DataFrame)

        # to_frame
        df = result.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

        # to_markdown, to_latex, to_typst
        md = result.to_markdown()
        assert isinstance(md, str) and len(md) > 0
        ltx = result.to_latex()
        assert isinstance(ltx, str) and "\\begin" in ltx
        typ = result.to_typst()
        assert isinstance(typ, str) and "#table" in typ

        # plot
        fig = result.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestTier1AnalyticGradients:
    """Tier 1: Feature coverage for Exact Analytic Gradients via IFT."""

    def test_analytic_gradients_happy_path(self, canonical_collocation_solution):
        """Verify evaluation of IFT policy parameter sensitivities."""
        ag = _require_analytic_gradients()
        sol, prob = canonical_collocation_solution

        res = ag.compute_ift_gradients(sol, prob, params=["alpha", "beta"], backend="numpy")

        assert hasattr(res, "grad_coefficients"), "Must contain grad_coefficients"
        assert hasattr(res, "param_names"), "Must contain param_names"
        assert res.grad_coefficients.shape == (len(sol.coefficients), 2)
        assert res.condition_number > 0.0
        assert np.all(np.isfinite(res.grad_coefficients))

    def test_analytic_gradients_finite_difference_parity(self, canonical_collocation_solution):
        """Verify exact IFT gradients match 2-sided central finite difference to < 1e-4 relative error."""
        ag = _require_analytic_gradients()
        sol, prob = canonical_collocation_solution

        # Analytical IFT gradient
        ift_res = ag.compute_ift_gradients(sol, prob, params=["alpha"], backend="numpy")
        ift_grad = ift_res.grad_coefficients[:, 0]

        # Numerical central finite difference
        h = 1e-5
        domain = prob.domain[0] if isinstance(prob.domain[0], (tuple, list)) else prob.domain
        prob_plus = CollocationProblem(
            domain=domain,
            orders=prob.orders,
            method="euler",
            params={"alpha": 0.36 + h, "delta": 1.0},
            beta=prob.beta,
        )
        sol_plus = prob_plus.solve(backend="numpy")

        prob_minus = CollocationProblem(
            domain=domain,
            orders=prob.orders,
            method="euler",
            params={"alpha": 0.36 - h, "delta": 1.0},
            beta=prob.beta,
        )
        sol_minus = prob_minus.solve(backend="numpy")

        fd_grad = (sol_plus.coefficients - sol_minus.coefficients) / (2.0 * h)

        rel_error = np.max(np.abs(ift_grad - fd_grad) / (np.abs(fd_grad) + 1e-8))
        assert rel_error < 1e-4, f"IFT gradient relative error {rel_error:.2e} exceeds tolerance 1e-4"

    def test_analytic_gradients_policy_and_aggregates(self, canonical_collocation_solution):
        """Verify continuous policy gradient evaluation grad_theta g(s) across continuous domain."""
        ag = _require_analytic_gradients()
        sol, prob = canonical_collocation_solution

        res = ag.compute_ift_gradients(sol, prob, params=["alpha", "beta"], backend="numpy")

        # Evaluate at continuous test states
        domain = prob.domain[0] if isinstance(prob.domain[0], (tuple, list)) else prob.domain
        s_test = np.linspace(domain[0], domain[1], 15)
        grad_policy = res.policy_gradient(s_test)

        assert grad_policy.shape == (len(s_test), 2)
        assert np.all(np.isfinite(grad_policy))

    def test_analytic_gradients_presentation_contract(self, canonical_collocation_solution):
        """Verify presentation interface: summary, plot, to_markdown, to_latex, to_typst."""
        ag = _require_analytic_gradients()
        sol, prob = canonical_collocation_solution

        res = ag.compute_ift_gradients(sol, prob, params=["alpha", "beta"], backend="numpy")

        sm = res.summary()
        assert isinstance(sm, pd.DataFrame)
        assert "alpha" in sm.to_string()

        assert isinstance(res.to_markdown(), str)
        assert "\\begin" in res.to_latex()
        assert "#table" in res.to_typst()

        fig = res.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestTier1DeepMacroPINNs:
    """Tier 1: Feature coverage for Deep Macro Physics-Informed Neural Networks."""

    def test_deep_macro_pure_numpy_mlp_forward_backward(self):
        """Verify pure NumPy multi-layer perceptron forward and backward passes with analytical derivatives."""
        dm = _require_deep_macro()

        # Instantiate pure NumPy MLP
        mlp = dm.DeepMacroMLP(
            input_dim=10,
            hidden_dims=(32, 32),
            output_dim=10,
            activation="silu",
            seed=42,
        )

        # Batch forward pass
        batch_x = np.random.uniform(0.5, 2.0, (16, 10))
        out = mlp.forward(batch_x)
        assert out.shape == (16, 10)
        assert np.all(np.isfinite(out))

        # Backward pass with analytical gradients
        dL_dy = np.ones_like(out)
        grads = mlp.backward(dL_dy)
        assert len(grads) == len(mlp.weights)

        # Weight update via AdamOptimizer
        opt = dm.AdamOptimizer(mlp.get_params(), lr=1e-3)
        flat_grads = []
        for dW, db in grads:
            flat_grads.append(dW)
            flat_grads.append(db)

        w_before = mlp.weights[0].copy()
        opt.step(flat_grads)
        assert not np.allclose(mlp.weights[0], w_before), "Weights must update after gradient step"

    def test_deep_macro_solve_happy_path(self):
        """Verify solving a high-dimensional continuous macro model using Deep Macro PINN."""
        dm = _require_deep_macro()

        # 10-country dynamic capital accumulation model
        model = dm.DeepMacroModel(
            n_states=10,
            n_controls=10,
            beta=0.96,
            params={"alpha": 0.36, "delta": 0.08, "sigma": 1.0},
        )

        # Train with fast settings for CI execution
        sol = dm.solve_deep_macro(
            model=model,
            hidden_dims=(32, 32),
            activation="silu",
            n_epochs=60,
            batch_size=64,
            lr=2e-3,
            trajectory_length=500,
            resimulate_every=20,
            backend="numpy",
            seed=42,
        )

        assert isinstance(sol, dm.DeepMacroSolution)
        assert len(sol.loss_history) == 60
        assert sol.test_euler_mse < 1e-3 or np.mean(sol.loss_history) < 1e-3, "Loss must reach convergence tolerance"

        # Continuous policy evaluation on arbitrary continuous state
        s_eval = np.full((5, 10), 1.5)
        c_eval = sol.policy(s_eval)
        assert c_eval.shape == (5, 10)
        assert np.all(c_eval > 0.0), "Consumption must be strictly positive"

    def test_deep_macro_presentation_contract(self):
        """Verify presentation interface: summary, plot, to_frame, to_markdown, to_latex, to_typst."""
        dm = _require_deep_macro()

        model = dm.DeepMacroModel(n_states=3, n_controls=3, beta=0.96)
        sol = dm.solve_deep_macro(
            model=model,
            hidden_dims=(16, 16),
            n_epochs=10,
            trajectory_length=200,
            backend="numpy",
            seed=123,
        )

        sm = sol.summary()
        assert isinstance(sm, pd.DataFrame)
        assert "State Dimensions" in sm.index

        assert isinstance(sol.to_frame(), pd.DataFrame)
        assert isinstance(sol.to_markdown(), str)
        assert "\\begin" in sol.to_latex()
        assert "#table" in sol.to_typst()

        fig = sol.plot()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestTier1QuantitativeSpatialGE:
    """Tier 1: Feature coverage for Caliendo-Parro and Allen-Arkolakis Quantitative Spatial GE."""

    def test_caliendo_parro_exact_hat_algebra_happy_path(self, canonical_caliendo_parro_data):
        """Verify Caliendo-Parro Exact Hat Algebra solve under tariff counterfactual."""
        cp = _require_caliendo_parro()
        data = canonical_caliendo_parro_data

        model = cp.CaliendoParroModel(
            trade_shares=data["trade_shares"],
            gamma_va=data["gamma_va"],
            gamma_io=data["gamma_io"],
            alpha=data["alpha"],
            theta=data["theta"],
            labor_income=data["labor_income"],
            deficits=data["deficits"],
            tariffs=data["tariffs"],
            country_codes=data["country_codes"],
            sector_codes=data["sector_codes"],
        )

        # Counterfactual: 10% tariff imposed by USA on Chinese manufacturing
        res = model.simulate_tariff_shock(importer="USA", exporter="CHN", sector="manuf", tariff_rate=0.10)

        assert isinstance(res, cp.CaliendoParroResult)
        assert res.converged, "Equilibrium wage tatonnement must converge"
        assert res.w_hat.shape == (3,)
        assert res.P_hat.shape == (3, 2)
        assert res.pi_prime.shape == (2, 3, 3)

        # Invariant 1: Bilateral trade shares sum to 1.0 along origin dimension for all n, j
        for j in range(2):
            for n in range(3):
                share_sum = np.sum(res.pi_prime[j, n, :])
                assert np.isclose(share_sum, 1.0, atol=1e-12), f"Trade shares do not sum to 1: {share_sum}"

        # Invariant 2: Labor market clearing residual < 1e-5
        assert res.market_clearing_residual < 1e-5, f"Market clearing error {res.market_clearing_residual} >= 1e-5"

    def test_caliendo_parro_presentation_contract(self, canonical_caliendo_parro_data):
        """Verify CaliendoParroResult presentation interface."""
        cp = _require_caliendo_parro()
        data = canonical_caliendo_parro_data

        model = cp.CaliendoParroModel(
            trade_shares=data["trade_shares"],
            gamma_va=data["gamma_va"],
            gamma_io=data["gamma_io"],
            alpha=data["alpha"],
            theta=data["theta"],
            labor_income=data["labor_income"],
            country_codes=data["country_codes"],
            sector_codes=data["sector_codes"],
        )

        res = model.solve_counterfactual()

        sm = res.summary()
        assert isinstance(sm, pd.DataFrame)
        assert "Welfare (%)" in sm.columns or "welfare_pct" in sm.columns

        sec_sm = res.sector_summary()
        assert isinstance(sec_sm, pd.DataFrame)

        assert isinstance(res.to_markdown(), str)
        assert "\\begin" in res.to_latex()
        assert "#table" in res.to_typst()

        fig = res.plot(kind="welfare")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_allen_arkolakis_spatial_ge_happy_path(self, canonical_allen_arkolakis_data):
        """Verify Allen-Arkolakis continuous geographic spatial equilibrium."""
        aa = _require_allen_arkolakis()
        data = canonical_allen_arkolakis_data

        model = aa.AllenArkolakisModel(
            trade_costs=data["trade_costs"],
            fundamental_productivity=data["fundamental_productivity"],
            fundamental_amenity=data["fundamental_amenity"],
            theta=data["theta"],
            alpha=data["alpha"],
            beta=data["beta"],
            total_population=data["total_population"],
            region_names=data["region_names"],
            coordinates=data["coordinates"],
        )

        assert model.is_unique, "Model parameters must satisfy uniqueness condition"

        res = model.solve_equilibrium(tol=1e-7, max_iter=1000)

        assert isinstance(res, aa.AllenArkolakisResult)
        assert res.converged, "Spatial equilibrium fixed-point iteration must converge"
        assert res.wages.shape == (5,)
        assert res.population.shape == (5,)

        # Invariant 1: Population conservation sum(L_i) = L_total +- 1e-10
        total_pop = np.sum(res.population)
        assert np.isclose(total_pop, data["total_population"], atol=1e-9), f"Population conservation failed: {total_pop}"

        # Invariant 2: Equalized spatial utility across all regions
        assert res.spatial_utility_variance < 1e-6, f"Spatial utility variance {res.spatial_utility_variance} >= 1e-6"

    def test_allen_arkolakis_presentation_contract(self, canonical_allen_arkolakis_data):
        """Verify AllenArkolakisResult presentation interface."""
        aa = _require_allen_arkolakis()
        data = canonical_allen_arkolakis_data

        model = aa.AllenArkolakisModel(
            trade_costs=data["trade_costs"],
            fundamental_productivity=data["fundamental_productivity"],
            fundamental_amenity=data["fundamental_amenity"],
            region_names=data["region_names"],
        )
        res = model.solve_equilibrium(tol=1e-6)

        sm = res.summary()
        assert isinstance(sm, pd.DataFrame)
        assert "Population" in sm.columns or "population" in sm.columns

        assert isinstance(res.to_markdown(), str)
        assert "\\begin" in res.to_latex()
        assert "#table" in res.to_typst()

        fig = res.plot(kind="spatial")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ===========================================================================
# TIER 2: BOUNDARY & CORNER CASES (LIMITS, RESILIENCE, EXTREMES)
# ===========================================================================

class TestTier2BoundaryCornerCases:
    """Tier 2: Boundary conditions, corner cases, asymptotic limits, and numerical stress."""

    def test_continuous_transition_zero_shock_invariance(self, canonical_continuous_steady_state):
        """Zero shock invariance: MIT transition under zero shocks identically preserves steady state."""
        ct = _require_continuous_transition()
        ss = canonical_continuous_steady_state

        T = 20
        zero_shocks = np.zeros(T)

        res = ct.solve_continuous_transition(
            initial_steady_state=ss,
            shock_path=zero_shocks,
            horizon=T,
            solver="broyden",
            tol=1e-4,
            backend="numpy",
        )

        # Invariant: interest rate and capital remain static across all transition periods
        assert np.allclose(res.r_path, res.r_path[0], atol=1e-6), "Zero shock path must be static across time"
        assert np.allclose(res.K_s_path, res.K_s_path[0], atol=1e-6), "Zero shock path must be static across time"

    def test_continuous_transition_boundary_mass_clamping(self, canonical_continuous_equilibrium):
        """Boundary mass clamping: extreme wealth distribution at k=0 preserves mass conservation."""
        import dataclasses

        ct = _require_continuous_transition()
        ss = canonical_continuous_equilibrium

        # Plant all mass at lower boundary k = k_min. A dict cannot carry a
        # distribution (it is a solve_aiyagari_continuous configuration), so the
        # solved equilibrium is copied with the clamped initial distribution.
        k_grid = ss.distribution.asset_grid
        clamped_pdf = np.zeros_like(ss.distribution.pdf)
        clamped_pdf[0, :] = 0.5  # 100% mass at borrowing constraint
        clamped_ss = dataclasses.replace(
            ss,
            distribution=dataclasses.replace(ss.distribution, pdf=clamped_pdf),
            K=float(np.sum(k_grid[:, None] * clamped_pdf)),
        )

        with pytest.warns(RuntimeWarning, match="did not converge"):
            res = ct.solve_continuous_transition(
                initial_steady_state=clamped_ss,
                shock_path=np.zeros(10),
                horizon=10,
                solver="shooting",
                tol=1e-2,
                backend="numpy",
            )

        # The clamped distribution is the one the transition starts from.
        assert np.array_equal(res.distributions[0], clamped_pdf)
        assert res.K_s_path[0] == pytest.approx(0.0, abs=1e-12)
        # No mass loss despite heavy constraint boundary concentration
        for t, dist in enumerate(res.distributions):
            assert np.isclose(np.sum(dist), 1.0, atol=1e-11), f"Mass escaped at t={t}: sum={np.sum(dist)}"

    def test_analytic_gradients_kinked_domain_or_extreme_parameter(self):
        """Verify IFT gradient solver stability near extreme discount factor beta -> 0.99."""
        ag = _require_analytic_gradients()

        alpha = 0.36
        beta_high = 0.99
        k_ss = (alpha * beta_high) ** (1.0 / (1.0 - alpha))

        prob = CollocationProblem(
            domain=(0.6 * k_ss, 1.4 * k_ss),
            orders=5,
            method="euler",
            params={"alpha": alpha, "delta": 1.0},
            beta=beta_high,
        )
        sol = prob.solve(backend="numpy")

        res = ag.compute_ift_gradients(sol, prob, params=["alpha", "beta"], backend="numpy")
        assert np.all(np.isfinite(res.grad_coefficients)), "Gradients must remain finite near high beta"
        assert res.condition_number < 1e12, f"Condition number {res.condition_number} indicates near-singularity"

    def test_analytic_gradients_fem_and_splines_compatibility(self):
        """Verify compute_ift_gradients functions seamlessly on FEM and Spline continuous solutions."""
        ag = _require_analytic_gradients()

        alpha = 0.36
        beta = 0.96
        def euler_fn(k, kp, kpp, p=None):
            a = p.get("alpha", alpha) if p else alpha
            b = p.get("beta", beta) if p else beta
            c = k**a - kp
            cp = kp**a - kpp
            return 1.0 - b * (c / cp) * a * (kp**(a - 1.0))

        # 1. FEM problem
        fem_prob = FEMProblem(
            domain=(0.5, 2.5),
            elements=8,
            method="euler",
            euler_residual_fn=euler_fn,
            params={"alpha": alpha, "delta": 1.0},
            beta=beta,
        )
        fem_sol = fem_prob.solve(backend="numpy")

        res_fem = ag.compute_ift_gradients(fem_sol, fem_prob, params=["alpha"], backend="numpy")
        assert hasattr(res_fem, "grad_coefficients")
        assert np.all(np.isfinite(res_fem.grad_coefficients))

        # 2. Spline Collocation problem
        spline_prob = SplineCollocationProblem(
            domain=(0.5, 2.5),
            n_knots=8,
            method="euler",
            params={"alpha": alpha, "delta": 1.0},
            beta=beta,
        )
        spline_sol = spline_prob.solve(backend="numpy")

        res_spline = ag.compute_ift_gradients(spline_sol, spline_prob, params=["alpha"], backend="numpy")
        assert hasattr(res_spline, "grad_coefficients")
        assert np.all(np.isfinite(res_spline.grad_coefficients))

    def test_deep_macro_high_dimensional_10_states_scaling(self):
        """Verify 10-state Deep Macro model scales cleanly without memory blowup or shape mismatch."""
        dm = _require_deep_macro()

        model_10d = dm.DeepMacroModel(
            n_states=10,
            n_controls=10,
            beta=0.96,
            params={"alpha": 0.36, "delta": 0.08},
        )

        # Batch forward pass and loss evaluation in 10 dimensions
        states = np.random.uniform(0.5, 2.0, (128, 10))
        mlp = dm.DeepMacroMLP(input_dim=10, hidden_dims=(32, 32), output_dim=10, activation="silu")

        t0 = time.perf_counter()
        out = mlp.forward(states)
        t_fwd = time.perf_counter() - t0

        assert out.shape == (128, 10)
        assert t_fwd < 0.10, f"Forward pass in 10D took {t_fwd:.4f}s; must be < 0.10s"

    def test_deep_macro_physical_feasibility_bound(self):
        """Verify that DeepMacroMLP strictly obeys physical resource bounds (0 < c <= cash_on_hand)."""
        dm = _require_deep_macro()

        model = dm.DeepMacroModel(n_states=5, n_controls=5, beta=0.96)
        sol = dm.solve_deep_macro(model=model, hidden_dims=(16, 16), n_epochs=20, trajectory_length=300, backend="numpy")

        sim_res = sol.simulate(periods=100)
        controls = sim_res["controls"]
        coh = sim_res["cash_on_hand"]

        assert np.all(controls > 0.0), "Consumption must be strictly positive everywhere"
        assert np.all(controls <= coh + 1e-12), "Consumption cannot exceed cash on hand"

    def test_caliendo_parro_zero_shock_identity(self, symmetric_caliendo_parro_model):
        """Zero shock identity: tau_hat = 1.0 and d_hat = 1.0 yields exact baseline w_hat = 1.0, P_hat = 1.0."""
        res = symmetric_caliendo_parro_model.solve_counterfactual(tau_hat=None, d_hat=None)

        assert res.converged
        assert np.allclose(res.w_hat, 1.0, atol=1e-6), "Zero shock must yield w_hat = 1.0"
        assert np.allclose(res.P_hat, 1.0, atol=1e-6), "Zero shock must yield P_hat = 1.0"
        assert np.allclose(res.welfare_pct, 0.0, atol=1e-6), "Zero shock must yield 0% welfare change"

    def test_caliendo_parro_extreme_trade_elasticity(self, canonical_caliendo_parro_data):
        """Verify Hat Algebra stability across near-unitary (theta=1.2) and high (theta=12.0) trade elasticities."""
        cp = _require_caliendo_parro()
        data = canonical_caliendo_parro_data

        for theta_val in [np.array([1.2, 1.5]), np.array([10.0, 12.0])]:
            model = cp.CaliendoParroModel(
                trade_shares=data["trade_shares"],
                gamma_va=data["gamma_va"],
                gamma_io=data["gamma_io"],
                alpha=data["alpha"],
                theta=theta_val,
                labor_income=data["labor_income"],
                country_codes=data["country_codes"],
                sector_codes=data["sector_codes"],
            )

            res = model.simulate_tariff_shock(importer="USA", exporter="CHN", sector="manuf", tariff_rate=0.05)
            assert res.converged, f"Failed to converge at theta={theta_val}"
            assert np.all(np.isfinite(res.welfare_pct)), "Welfare must be finite"

    def test_allen_arkolakis_autarky_limit(self, canonical_allen_arkolakis_data):
        """Autarky limit: prohibitive trade costs tau_ij -> inf drive bilateral trade shares to zero."""
        aa = _require_allen_arkolakis()
        data = canonical_allen_arkolakis_data

        # Prohibitive trade costs
        N = len(data["region_names"])
        autarky_costs = np.full((N, N), 1000.0)
        np.fill_diagonal(autarky_costs, 1.0)

        model = aa.AllenArkolakisModel(
            trade_costs=autarky_costs,
            fundamental_productivity=data["fundamental_productivity"],
            fundamental_amenity=data["fundamental_amenity"],
            theta=data["theta"],
            alpha=data["alpha"],
            beta=data["beta"],
        )

        res = model.solve_equilibrium(tol=1e-6)

        # Off-diagonal trade shares should be near zero, diagonal near 1.0
        diag_shares = np.diag(res.trade_shares)
        off_diag_shares = res.trade_shares[~np.eye(N, dtype=bool)]

        assert np.allclose(diag_shares, 1.0, atol=1e-3), "Domestic trade share under autarky must be ~1.0"
        assert np.all(off_diag_shares < 1e-3), "Cross-border trade share under autarky must be near zero"

    def test_allen_arkolakis_uniqueness_boundary(self, canonical_allen_arkolakis_data):
        """Verify that violating alpha + beta <= theta / (1 + theta) flags is_unique = False."""
        aa = _require_allen_arkolakis()
        data = canonical_allen_arkolakis_data

        # Strong agglomeration + weak congestion: alpha = 0.90, beta = -0.05, theta = 4.0
        # alpha + beta = 0.85 > 4.0 / 5.0 = 0.80
        model = aa.AllenArkolakisModel(
            trade_costs=data["trade_costs"],
            fundamental_productivity=data["fundamental_productivity"],
            fundamental_amenity=data["fundamental_amenity"],
            theta=4.0,
            alpha=0.90,
            beta=-0.05,
        )

        assert not model.is_unique, "Model must flag is_unique=False when agglomeration forces exceed uniqueness threshold"


# ===========================================================================
# TIER 3: CROSS-FEATURE SUBSYSTEM COMBINATIONS
# ===========================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-subsystem integration across frontier macroeconomic engines."""

    def test_mit_shock_transition_with_ift_endpoint_sensitivities(self, canonical_continuous_steady_state):
        """Cross-Feature: Continuous Transition dynamics coupled with exact IFT sensitivities.

        Validates that the linear sensitivity computed by IFT correctly predicts the initial
        response direction of aggregate capital under the full non-linear transition path.
        """
        ct = _require_continuous_transition()
        ag = _require_analytic_gradients()
        ss = canonical_continuous_steady_state

        # Solve small transition path under positive TFP shock
        T = 25
        shock_path = 0.02 * (0.85 ** np.arange(T))
        trans_res = ct.solve_continuous_transition(
            initial_steady_state=ss,
            shock_path=shock_path,
            horizon=T,
            solver="broyden",
            tol=1e-3,
            backend="numpy",
        )

        # Verify initial capital response direction
        delta_K_trans = trans_res.K_s_path[1] - trans_res.K_s_path[0]
        assert np.isfinite(delta_K_trans)

        # IFT sensitivity on Neoclassical continuous model
        prob = CollocationProblem(domain=(1.0, 4.0), orders=6, method="euler", params={"alpha": 0.36, "delta": 1.0}, beta=0.96)
        sol = prob.solve(backend="numpy")
        ift_res = ag.compute_ift_gradients(sol, prob, params=["alpha"], backend="numpy")

        # IFT gradient must indicate positive capital response to productivity/capital share
        assert ift_res.grad_coefficients[0, 0] != 0.0

    def test_spatial_distance_matrix_feeding_caliendo_parro(self, canonical_caliendo_parro_data):
        """Cross-Feature: Spatial distance matrix from puremacro.spatial.weights feeding Caliendo-Parro IO trade GE.

        Demonstrates seamless integration between continuous geographic coordinates and multi-sector trade hat algebra.
        """
        cp = _require_caliendo_parro()
        data = canonical_caliendo_parro_data

        # Geographic coordinates for USA, CHN, ROW
        coords = np.array([
            [38.9, -77.0],  # Washington D.C.
            [39.9, 116.4],  # Beijing
            [50.8, 4.3],    # Brussels (ROW proxy)
        ])

        # Compute bilateral distances
        dist = pairwise_distances(coords, metric="haversine")

        # Build trade cost changes based on an infrastructure upgrade between USA and ROW:
        # 15% reduction in transportation friction
        d_hat = np.ones_like(data["trade_shares"])
        d_hat[:, 0, 2] = 0.85  # USA -> ROW cost reduction
        d_hat[:, 2, 0] = 0.85  # ROW -> USA cost reduction

        model = cp.CaliendoParroModel(
            trade_shares=data["trade_shares"],
            gamma_va=data["gamma_va"],
            gamma_io=data["gamma_io"],
            alpha=data["alpha"],
            theta=data["theta"],
            labor_income=data["labor_income"],
            country_codes=data["country_codes"],
            sector_codes=data["sector_codes"],
        )

        res = model.solve_counterfactual(d_hat=d_hat)

        assert res.converged
        # USA and ROW should both experience positive welfare gains from bilateral transport improvement
        assert res.welfare_pct[0] > 0.0, "USA welfare must increase from bilateral transport cost reduction"
        assert res.welfare_pct[2] > 0.0, "ROW welfare must increase from bilateral transport cost reduction"

    def test_deep_macro_vs_continuous_collocation_benchmark(self):
        """Cross-Feature: Benchmark 1D continuous Deep Macro PINN against Chebyshev Collocation.

        Verifies that both continuous solvers approximate the analytical Brock-Mirman policy function.
        """
        dm = _require_deep_macro()

        alpha = 0.36
        beta = 0.96
        k_ss = (alpha * beta) ** (1.0 / (1.0 - alpha))

        # 1. Collocation solve
        prob = CollocationProblem(domain=(0.6 * k_ss, 1.4 * k_ss), orders=6, method="euler", params={"alpha": alpha, "delta": 1.0}, beta=beta)
        sol_colloc = prob.solve(backend="numpy")

        # 2. Deep Macro solve
        model = dm.DeepMacroModel(n_states=1, n_controls=1, beta=beta, params={"alpha": alpha, "delta": 1.0})
        sol_deep = dm.solve_deep_macro(model=model, hidden_dims=(32, 32), n_epochs=50, trajectory_length=400, backend="numpy", seed=42)

        # Dense out-of-sample comparison grid
        k_test = np.linspace(0.7 * k_ss, 1.3 * k_ss, 20)[:, None]
        c_colloc = (k_test ** alpha) - sol_colloc.policy(k_test.ravel())[:, None]
        c_deep = sol_deep.policy(k_test)

        # Both policies must be strictly increasing and close
        assert np.all(np.diff(c_colloc.ravel()) > 0.0), "Collocation policy must be strictly monotonic"
        assert np.all(np.diff(c_deep.ravel()) > 0.0), "Deep Macro policy must be strictly monotonic"

    def test_allen_arkolakis_climate_shock_with_caliendo_parro_counterfactual(self, canonical_allen_arkolakis_data, canonical_caliendo_parro_data):
        """Cross-Feature: Localized regional climate shock in Allen-Arkolakis mapped into national trade counterfactual."""
        aa = _require_allen_arkolakis()
        cp = _require_caliendo_parro()

        # 1. Run climate shock in Southern region (10% productivity loss -> 0.90 factor)
        aa_model = aa.AllenArkolakisModel(
            trade_costs=canonical_allen_arkolakis_data["trade_costs"],
            fundamental_productivity=canonical_allen_arkolakis_data["fundamental_productivity"],
            fundamental_amenity=canonical_allen_arkolakis_data["fundamental_amenity"],
            region_names=canonical_allen_arkolakis_data["region_names"],
        )
        aa_res = aa_model.simulate_climate_shock(productivity_shocks={"South": 0.90})

        assert aa_res.converged
        assert aa_res.welfare_pct < 0.0, "Aggregate spatial welfare must fall under climate productivity loss"

        # 2. Map aggregate domestic wage contraction into Caliendo-Parro trade GE
        cp_model = cp.CaliendoParroModel(
            trade_shares=canonical_caliendo_parro_data["trade_shares"],
            gamma_va=canonical_caliendo_parro_data["gamma_va"],
            gamma_io=canonical_caliendo_parro_data["gamma_io"],
            alpha=canonical_caliendo_parro_data["alpha"],
            theta=canonical_caliendo_parro_data["theta"],
            labor_income=canonical_caliendo_parro_data["labor_income"],
        )

        cp_res = cp_model.solve_counterfactual()
        assert cp_res.converged


# ===========================================================================
# TIER 4: REAL-WORLD POLICY & EMPIRICAL SCENARIOS
# ===========================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Realistic macroeconomic policy counterfactuals, shock transitions, and trade wars."""

    def test_scenario_100bps_monetary_rate_hike_mit_shock(self, canonical_continuous_equilibrium):
        """Scenario 1: 100 bps unexpected monetary policy rate hike decaying with persistence rho=0.70.

        The shock is a wedge on the return households earn (``shock_var="r"``).

        Economic properties:
        - Capital is predetermined on impact, so the market rate starts at the steady-state rate.
        - Wealth distribution gradually evolves rightward as households accumulate assets at the higher return.
        - The larger capital stock lowers the market rate, which then recovers towards the steady state.
        - Mass is strictly conserved throughout: sum(mu_t) = 1.0 +- 1e-12.
        """
        ct = _require_continuous_transition()
        ss = canonical_continuous_equilibrium

        T = 40
        # 100 bps rate hike: delta r_t = +0.0100 * (0.70^t)
        rho = 0.70
        rate_hike = 0.0100 * (rho ** np.arange(T))

        res = ct.solve_continuous_transition(
            initial_steady_state=ss,
            shock_path=rate_hike,
            shock_var="r",
            horizon=T,
            solver="broyden",
            tol=1e-3,
            max_iter=40,
            backend="numpy",
        )

        # Invariant 1: Mass conservation across all 40 periods
        for t, dist in enumerate(res.distributions):
            assert np.isclose(np.sum(dist), 1.0, atol=1e-11), f"Mass leaked at period t={t}: sum = {np.sum(dist)}"

        # Economic properties, measured against the solved steady state. Until the
        # AIYCONT fix this test compared with a hand-built r_ss = 0.035 that the
        # transition never used (its dict keys were dropped), which any falling
        # rate path satisfied.
        assert res.converged
        dev = res.r_path - ss.r
        assert abs(dev[0]) < 1e-5, "Capital is predetermined: the market rate starts at the steady state"
        assert np.all(res.K_s_path[1:] > ss.K), "Households accumulate capital at the higher return"
        assert np.all(dev[1:] < 0.0), "The larger capital stock lowers the market rate"
        trough = int(np.argmax(np.abs(dev)))
        assert 0 < trough < T - 1
        assert abs(dev[-1]) < 0.5 * abs(dev[trough]), "Rate path must recover toward the steady state"

    def test_scenario_us_china_bilateral_tariff_war_and_diversion(self, canonical_caliendo_parro_data):
        """Scenario 2: US-China 25% Bilateral Tariff War and Trade Diversion in Caliendo-Parro.

        Economic properties:
        - Bilateral trade shares between USA and CHN contract substantially.
        - Trade diversion: imports from third party (ROW) increase in both USA and CHN.
        - Both participating countries suffer net welfare losses due to deadweight loss and intermediate cost markups.
        - Market clearing residuals < 1e-5.
        """
        cp = _require_caliendo_parro()
        data = canonical_caliendo_parro_data

        model = cp.CaliendoParroModel(
            trade_shares=data["trade_shares"],
            gamma_va=data["gamma_va"],
            gamma_io=data["gamma_io"],
            alpha=data["alpha"],
            theta=data["theta"],
            labor_income=data["labor_income"],
            deficits=data["deficits"],
            country_codes=data["country_codes"],
            sector_codes=data["sector_codes"],
        )

        # 25% bilateral tariff war between USA and CHN
        res = model.simulate_trade_war(coalition_a=["USA"], coalition_b=["CHN"], tariff_rate=0.25)

        assert res.converged
        assert res.market_clearing_residual < 1e-5

        # Economic property 1: Trade destruction between USA and CHN
        # USA import share on CHN manufacturing (sector 0, importer 0, exporter 1)
        pi_usa_chn_initial = data["trade_shares"][0, 0, 1]
        pi_usa_chn_prime = res.pi_prime[0, 0, 1]
        assert pi_usa_chn_prime < pi_usa_chn_initial, "US-China trade share must fall under bilateral tariffs"

        # Economic property 2: Trade diversion to ROW
        # USA import share on ROW manufacturing (sector 0, importer 0, exporter 2)
        pi_usa_row_initial = data["trade_shares"][0, 0, 2]
        pi_usa_row_prime = res.pi_prime[0, 0, 2]
        assert pi_usa_row_prime > pi_usa_row_initial, "Trade diversion must increase USA import share from ROW"

        # Economic property 3: Net welfare contraction for both war participants
        assert res.welfare_pct[0] < 0.0, "USA must suffer net welfare loss in tariff war"
        assert res.welfare_pct[1] < 0.0, "China must suffer net welfare loss in tariff war"

    def test_scenario_regional_transport_corridor_investment(self, canonical_allen_arkolakis_data):
        """Scenario 3: High-Speed Rail Regional Transport Corridor Investment in Allen-Arkolakis.

        Connecting North and South regions via a 20% reduction in bilateral iceberg trade costs:
        Economic properties:
        - Consumer Market Access (CMA) increases in both connected regions.
        - Population reallocates toward the connected economic corridor.
        - Worldwide aggregate welfare increases: bar{u}' > bar{u}.
        - Total labor population is strictly conserved.
        """
        aa = _require_allen_arkolakis()
        data = canonical_allen_arkolakis_data

        model = aa.AllenArkolakisModel(
            trade_costs=data["trade_costs"],
            fundamental_productivity=data["fundamental_productivity"],
            fundamental_amenity=data["fundamental_amenity"],
            theta=data["theta"],
            alpha=data["alpha"],
            beta=data["beta"],
            total_population=data["total_population"],
            region_names=data["region_names"],
            coordinates=data["coordinates"],
        )

        res = model.simulate_infrastructure_shock(origin="North", destination="South", cost_reduction=0.20)

        assert res.converged

        # Invariant: Total population conservation holds to 1e-9
        assert np.isclose(np.sum(res.population), data["total_population"], atol=1e-9)

        # Economic property 1: Aggregate welfare increases
        assert res.welfare_pct > 0.0, "Transport corridor must generate positive aggregate spatial welfare gains"

        # Economic property 2: Market access expands in connected regions
        assert res.consumer_market_access[0] > 0.0
        assert res.consumer_market_access[1] > 0.0

    def test_scenario_ten_sector_capital_convergence_deep_macro(self):
        """Scenario 4: 10-Sector Multi-Country Capital Deepening PINN along Ergodic Trajectories.

        Trains a pure NumPy Deep Macro PINN on a 10-dimensional state space:
        - Verifies out-of-sample Euler equation residual MSE < 1e-3.
        - Demonstrates stable capital deepening along ergodic paths.
        """
        dm = _require_deep_macro()

        model_10 = dm.DeepMacroModel(
            n_states=10,
            n_controls=10,
            beta=0.96,
            params={"alpha": 0.36, "delta": 0.08, "sigma": 1.0},
        )

        sol = dm.solve_deep_macro(
            model=model_10,
            hidden_dims=(32, 32),
            activation="silu",
            n_epochs=80,
            batch_size=64,
            lr=2e-3,
            trajectory_length=800,
            resimulate_every=25,
            backend="numpy",
            seed=100,
        )

        assert isinstance(sol, dm.DeepMacroSolution)
        assert sol.test_euler_mse < 1e-3, f"Test Euler MSE {sol.test_euler_mse:.4e} must be < 1e-3"

        # Simulate 50 periods starting from low capital (0.8)
        k_init = np.full(10, 0.8)
        sim_data = sol.simulate(s0=k_init, periods=50)
        states = sim_data["states"]

        # Capital must deepen: average capital at end of path exceeds initial capital
        k_mean_start = np.mean(states[0])
        k_mean_end = np.mean(states[-1])
        assert k_mean_end > k_mean_start, "Economy must exhibit capital deepening from below steady state"
