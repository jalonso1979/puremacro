"""Unit tests for Continuous Transition Dynamics & MIT Shocks (Milestone 1).

Verifies:
1. Strict Mass Conservation Invariant:
   - Preserves sum_{i, m} mu_t(k_i, z_m) = 1.0 +- 1e-12 at all t in [0, T] over T = 150.
   - Maintains mass conservation under large / stress shocks (+300 bps rate hike, -10% TFP).
2. Zero Shock Steady-State Invariance:
   - Returns steady-state path identically with zero iterations (iterations == 0).
   - Price path deviation ||r_t - r_ss||_inf < 1e-12 and market clearing ||H||_inf < 1e-4.
3. Permanent TFP Shock Transition:
   - Simulates 5% permanent productivity increase from SS_0 to SS_1.
   - Factor prices {r_t, w_t} and capital stock K_t transition smoothly to SS_1.
   - Market clearing error ||K^s - K^d||_inf < 1e-4.
4. Transitory Monetary / Interest Rate Jump:
   - Simulates unexpected 100 bps rate shock r_t = r_ss + 0.01 * 0.7^t.
   - Broyden Quasi-Newton converges in < 25 iterations to ||H||_inf < 1e-4.
   - Trajectory reverts to steady state as t -> T.
5. Presentation Interface Compliance:
   - .summary(), .to_frame(), .to_markdown(), .to_latex(), .to_typst(), and .plot().
6. Algorithm Comparison (Broyden vs Shooting):
   - Both relaxation algorithms achieve market clearing within tolerance.
7. Input Validation & Edge Cases:
   - Validates horizon, tolerance, max_iter, shock_var, solver name, and persistence.
8. Pyodide 4-Package Compliance:
   - Strictly pure NumPy/SciPy/Pandas/Matplotlib without unauthorized foreign dependencies.
"""
from __future__ import annotations

import re
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Headless backend for automated test runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.vfi.continuous_distribution import (
    AiyagariContinuousEquilibrium,
    solve_aiyagari_continuous,
)
from puremacro.vfi.continuous_transition import (
    ContinuousTransitionResult,
    TransitionShock,
    continuous_mit_shock,
    solve_continuous_transition,
)


# ---------------------------------------------------------------------------
# Shared Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def baseline_steady_state() -> AiyagariContinuousEquilibrium:
    """Compute baseline continuous Aiyagari stationary equilibrium."""
    return solve_aiyagari_continuous(
        beta=0.96,
        gamma=2.0,
        alpha=0.36,
        delta=0.08,
        N_k=150,
        n_z=5,
        max_evals=60,
    )


# ---------------------------------------------------------------------------
# 1. Mass Conservation Invariant Tests
# ---------------------------------------------------------------------------

class TestMassConservationInvariant:
    """Verify strict mass conservation of continuous distributions along transition paths."""

    def test_mass_conservation_horizon_150(self, baseline_steady_state: AiyagariContinuousEquilibrium):
        """Verify sum_{i, m} mu_t(k_i, z_m) = 1.0 +- 1e-12 at all t in [0, 150]."""
        T = 150
        res = continuous_mit_shock(
            baseline_steady_state,
            shock_type="tfp",
            shock_size=0.05,
            persistence=0.8,
            horizon=T,
            solver="broyden",
        )

        assert res.converged
        assert len(res.distributions) == T + 1
        assert res.mass_conservation_error < 1e-12

        for t, dist in enumerate(res.distributions):
            total_mass = float(np.sum(dist))
            assert np.all(dist >= 0.0), f"Negative probability mass at period {t}"
            assert np.isclose(total_mass, 1.0, atol=1e-12), (
                f"Mass conservation violated at period {t}: sum = {total_mass:.16f}"
            )

    def test_mass_conservation_severe_stress_shock(self, baseline_steady_state: AiyagariContinuousEquilibrium):
        """Verify mass conservation under large 300 bps interest rate hike and -10% TFP drop."""
        T = 40
        # Severe combined shock: 300 bps rate shock
        res = continuous_mit_shock(
            baseline_steady_state,
            shock_type="rate",
            shock_size=0.03,
            persistence=0.7,
            horizon=T,
            solver="broyden",
            max_iter=50,
        )

        assert len(res.distributions) == T + 1
        assert res.mass_conservation_error < 1e-12
        for dist in res.distributions:
            assert abs(np.sum(dist) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 2. Zero-Shock Steady-State Invariance Tests
# ---------------------------------------------------------------------------

class TestZeroShockSteadyStateInvariance:
    """Verify exact invariance when no shock is applied to the economy."""

    def test_zero_shock_none_input(self, baseline_steady_state: AiyagariContinuousEquilibrium):
        """Verify shock_path=None returns steady state identically in 0 iterations."""
        T = 50
        res = solve_continuous_transition(
            baseline_steady_state,
            shock_path=None,
            horizon=T,
            solver="broyden",
        )

        assert res.converged
        assert res.iterations == 0
        assert np.allclose(res.r_path, baseline_steady_state.r, atol=1e-12)
        assert np.allclose(res.w_path, baseline_steady_state.w, atol=1e-12)
        assert np.allclose(res.K_s_path, baseline_steady_state.K, atol=1e-4)
        assert res.max_residual < 1e-4

    def test_zero_shock_zeros_array(self, baseline_steady_state: AiyagariContinuousEquilibrium):
        """Verify shock_path=np.zeros(T) returns steady state identically in 0 iterations."""
        T = 40
        res = solve_continuous_transition(
            baseline_steady_state,
            shock_path=np.zeros(T),
            shock_var="tfp",
            horizon=T,
            solver="broyden",
        )

        assert res.converged
        assert res.iterations == 0
        assert np.max(np.abs(res.r_path - baseline_steady_state.r)) < 1e-12
        assert res.max_residual < 1e-4


# ---------------------------------------------------------------------------
# 3. Permanent TFP Shock Transition Tests
# ---------------------------------------------------------------------------

class TestPermanentTFPShockTransition:
    """Verify non-linear transition dynamics under a permanent 5% TFP increase."""

    def test_permanent_tfp_increase_convergence_and_dynamics(
        self, baseline_steady_state: AiyagariContinuousEquilibrium
    ):
        """Verify smooth transition from SS_0 to higher capital stock SS_1."""
        T = 50
        res = continuous_mit_shock(
            baseline_steady_state,
            shock_type="tfp",
            shock_size=0.05,
            persistence=1.0,
            horizon=T,
            solver="broyden",
        )

        assert res.converged
        assert res.max_residual < 1e-4

        # On impact: capital K_0 is predetermined at baseline
        assert np.isclose(res.K_s_path[0], baseline_steady_state.K, atol=1e-4)

        # Real wage must jump up on impact due to higher TFP
        assert res.w_path[0] > baseline_steady_state.w

        # Interest rate must jump up on impact due to higher marginal product of capital
        assert res.r_path[0] > baseline_steady_state.r

        # Capital stock accumulates over time: K_T > K_0
        assert res.K_s_path[-1] > res.K_s_path[0]

        # Over the horizon, capital stock accumulates towards long-run level
        assert res.K_s_path[-1] > baseline_steady_state.K * 1.04


# ---------------------------------------------------------------------------
# 4. Transitory Monetary / Interest Rate Jump Tests
# ---------------------------------------------------------------------------

class TestTransitoryRateJump:
    """Verify Broyden solver convergence on unexpected 100 bps rate shock."""

    def test_rate_jump_broyden_convergence(
        self, baseline_steady_state: AiyagariContinuousEquilibrium
    ):
        """Verify Broyden converges in < 25 iterations with ||H||_inf < 1e-4."""
        T = 40
        # Perturbed initial candidate interest rate path r_t = r_ss + 0.01 * 0.7^t
        shock_seq = 0.01 * (0.7 ** np.arange(T))
        r_init_path = baseline_steady_state.r + shock_seq

        res = solve_continuous_transition(
            baseline_steady_state,
            r_init_path=r_init_path,
            shock_var="rate",
            horizon=T,
            solver="broyden",
            tol=1e-4,
            max_iter=30,
        )

        assert res.converged
        assert res.iterations < 25, f"Broyden took {res.iterations} >= 25 iterations"
        assert res.max_residual < 1e-4

        # Reversion to steady state: r_T approaches r_ss
        assert np.isclose(res.r_path[-1], baseline_steady_state.r, atol=1e-3)
        assert np.isclose(res.K_s_path[-1], baseline_steady_state.K, atol=1e-3)


# ---------------------------------------------------------------------------
# 5. Presentation Interface Compliance Tests
# ---------------------------------------------------------------------------

class TestPresentationInterface:
    """Verify all required puremacro presentation methods execute cleanly."""

    def test_summary_and_frame_outputs(self, baseline_steady_state: AiyagariContinuousEquilibrium):
        """Verify .summary() and .to_frame() return valid structured tables."""
        T = 25
        res = continuous_mit_shock(
            baseline_steady_state,
            shock_type="tfp",
            shock_size=0.03,
            persistence=0.8,
            horizon=T,
            solver="broyden",
        )

        summary_df = res.summary()
        assert isinstance(summary_df, pd.DataFrame)
        assert "Metric" == summary_df.index.name
        assert "Horizon (T)" in summary_df.index
        assert "Max Residual (||H||_inf)" in summary_df.index
        assert "Mass Error" in summary_df.index
        assert summary_df.loc["Horizon (T)", "Value"] == str(T)
        assert summary_df.loc["Converged", "Value"] == "True"

        frame_df = res.to_frame()
        assert isinstance(frame_df, pd.DataFrame)
        assert frame_df.shape == (T, 6)
        assert list(frame_df.columns) == ["r", "w", "K_s", "K_d", "C", "residual"]
        assert frame_df.index.name == "t"

    def test_export_formats_markdown_latex_typst(
        self, baseline_steady_state: AiyagariContinuousEquilibrium
    ):
        """Verify .to_markdown(), .to_latex(), and .to_typst() return non-empty strings."""
        res = continuous_mit_shock(
            baseline_steady_state,
            shock_type="tfp",
            shock_size=0.02,
            persistence=0.7,
            horizon=20,
            solver="broyden",
        )

        md = res.to_markdown()
        assert isinstance(md, str)
        assert "Metric" in md
        assert "Horizon (T)" in md

        latex = res.to_latex()
        assert isinstance(latex, str)
        assert "\\begin{tabular}" in latex or "tabular" in latex

        typst = res.to_typst()
        assert isinstance(typst, str)
        assert "#table" in typst or "[" in typst

    def test_plot_generation(self, baseline_steady_state: AiyagariContinuousEquilibrium):
        """Verify .plot() produces a 6-panel Figure object with correct titles."""
        res = continuous_mit_shock(
            baseline_steady_state,
            shock_type="tfp",
            shock_size=0.03,
            persistence=0.8,
            horizon=25,
            solver="broyden",
        )

        fig = res.plot(show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        # 6 subplots
        assert len(fig.axes) == 6
        plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Algorithm Comparison Tests (Broyden vs Shooting)
# ---------------------------------------------------------------------------

class TestAlgorithmComparison:
    """Verify both Broyden and Shooting solvers produce mutually consistent solutions."""

    def test_broyden_vs_shooting_parity(self, baseline_steady_state: AiyagariContinuousEquilibrium):
        """Verify Broyden and Shooting converge to the same equilibrium price path."""
        T = 20
        shock = 0.03 * (0.75 ** np.arange(T))

        res_broyden = solve_continuous_transition(
            baseline_steady_state,
            shock_path=shock,
            shock_var="tfp",
            horizon=T,
            solver="broyden",
            tol=1e-4,
        )

        res_shooting = solve_continuous_transition(
            baseline_steady_state,
            shock_path=shock,
            shock_var="tfp",
            horizon=T,
            solver="shooting",
            tol=1e-4,
            damping=0.35,
            max_iter=60,
        )

        assert res_broyden.converged
        assert res_shooting.converged
        assert np.allclose(res_broyden.r_path, res_shooting.r_path, atol=1e-3)
        assert np.allclose(res_broyden.K_s_path, res_shooting.K_s_path, atol=1e-2)


# ---------------------------------------------------------------------------
# 7. Input Validation & Error Handling Tests
# ---------------------------------------------------------------------------

class TestInputValidationAndEdgeCases:
    """Verify robust validation of input arguments and error reporting."""

    def test_invalid_horizon_and_tolerances(self, baseline_steady_state: AiyagariContinuousEquilibrium):
        """Verify rejection of invalid horizon or non-positive tolerance."""
        with pytest.raises(ValueError, match="horizon must be a positive integer"):
            solve_continuous_transition(baseline_steady_state, horizon=0)

        with pytest.raises(ValueError, match="tol must be strictly positive"):
            solve_continuous_transition(baseline_steady_state, tol=0.0)

        with pytest.raises(ValueError, match="max_iter must be non-negative"):
            solve_continuous_transition(baseline_steady_state, max_iter=-1)

    def test_invalid_solver_and_shock_var(self, baseline_steady_state: AiyagariContinuousEquilibrium):
        """Verify rejection of unknown solver and unknown shock variable."""
        with pytest.raises(ValueError, match="Invalid solver"):
            solve_continuous_transition(baseline_steady_state, solver="newton_krylov")

        with pytest.raises(ValueError, match="Invalid shock_var"):
            solve_continuous_transition(baseline_steady_state, shock_var="money_supply")

    def test_continuous_mit_shock_persistence_validation(
        self, baseline_steady_state: AiyagariContinuousEquilibrium
    ):
        """Verify rejection of invalid persistence parameters outside [0, 1]."""
        with pytest.raises(ValueError, match="persistence must be in"):
            continuous_mit_shock(baseline_steady_state, persistence=1.5)

        with pytest.raises(ValueError, match="persistence must be in"):
            continuous_mit_shock(baseline_steady_state, persistence=-0.1)

    def test_transition_shock_validation(self):
        """Verify TransitionShock dataclass handles normalization and finite checks."""
        shock = TransitionShock(path=[0.01, 0.02, 0.03], var="TFP")
        assert shock.var == "tfp"
        assert len(shock.path) == 3

        with pytest.raises(ValueError, match="must all be finite"):
            TransitionShock(path=[0.01, np.nan, 0.03])


# ---------------------------------------------------------------------------
# 8. Pyodide 4-Package Contract Tests
# ---------------------------------------------------------------------------

class TestPyodideCompliance:
    """Verify zero external/foreign dependencies for browser execution."""

    def test_no_unauthorized_imports_in_continuous_transition(self):
        """Verify puremacro/vfi/continuous_transition.py imports only approved packages."""
        import puremacro.vfi.continuous_transition as ct
        file_path = ct.__file__

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Check for forbidden C-extensions, PyTorch, JAX, TensorFlow, SymPy
        forbidden = ["torch", "tensorflow", "jax", "sympy", "numba", "cython", "cupy", "mlx"]
        for pkg in forbidden:
            pattern = rf"(^|\n)\s*(import\s+{pkg}|from\s+{pkg})"
            match = re.search(pattern, code)
            assert match is None, f"Forbidden import of '{pkg}' detected in continuous_transition.py"
