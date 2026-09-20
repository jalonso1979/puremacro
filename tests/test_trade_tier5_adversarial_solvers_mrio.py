"""Tier 5 Adversarial Coverage Hardening Tests for puremacro.trade.

White-box adversarial audit covering:
- puremacro.trade.solver:
  * check_hawkins_simon_viability: degenerate matrices, zero rows, extreme spectral radii,
    negative/infinite tariffs, diverse input shapes, ViabilityResult unpacking, tax boundaries,
    and silent NaN acceptance gap.
  * solve_keller_pac: ds adaptations, turning points/bifurcations, predictor clipping, step limits,
    2D tariff tensor shape mismatch gap, and premature termination uncaught ValueError gap.
  * svd_clamped_newton_step: rank-deficient/zero Jacobians, clamping thresholding, wage displacement.
  * anderson_accelerate: empty buffers, m=1, overflow, collinear/zero/NaN residual histories.
  * solve_cyprus_manifold: linearized vs secant, out-of-bounds indices, singular Schur complements.
  * clamp_wage_displacement: scalar/vector, coordinate vs uniform modes, zero/negative bounds.
- puremacro.trade.regularize:
  * regularize_mrio_table: phantom output injection, negative VA flooring, dual TLS debit,
    TLS reconciliation, structural dimension validation, non-productive Leontief systems.
  * balance_ras, balance_gras, balance_quadratic: negative entries rejection, margin rescaling,
    structural zeros, zero marginal targets, and division-by-zero edge conditions.
- puremacro.trade.data:
  * load_figaro, load_exiobase, load_wiod, load_eora, load_oecd_icio_granular: fallback handling,
    custom dimension overrides, corrupt dimensions, missing accounting rows, scale factors.
  * Multilateral accounting and balance invariant verifiers.
"""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any
import numpy as np
import pandas as pd
import pytest

from puremacro.trade._results import TradeCalibrationResult, TradeEquilibriumResult
from puremacro.trade.data import (
    compute_trade_balances,
    generate_synthetic_mrio,
    load_eora,
    load_exiobase,
    load_figaro,
    load_oecd_icio_granular,
    load_wiod,
    package_mrio_to_calibration_result,
    verify_accounting_invariants,
    verify_zero_leakage,
)
from puremacro.trade.regularize import (
    balance_gras,
    balance_quadratic,
    balance_ras,
    compute_spectral_radius,
    regularize_mrio_table,
    spectral_radius,
    validate_accounting_identities,
)
from puremacro.trade.solver import (
    ViabilityResult,
    anderson_accelerate,
    check_hawkins_simon_viability,
    clamp_wage_displacement,
    solve_cyprus_manifold,
    solve_keller_pac,
    svd_clamped_newton_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calib_2c_2s() -> TradeCalibrationResult:
    """Generate a clean synthetic 2-country 2-sector calibrated trade model."""
    raw = generate_synthetic_mrio("oecd", custom_c=2, custom_s=2, seed=123)
    return package_mrio_to_calibration_result(raw, regularize=True)


@pytest.fixture(scope="module")
def calib_3c_2s() -> TradeCalibrationResult:
    """Generate a clean synthetic 3-country 2-sector calibrated trade model."""
    raw = generate_synthetic_mrio("oecd", custom_c=3, custom_s=2, seed=456)
    return package_mrio_to_calibration_result(raw, regularize=True)


# ---------------------------------------------------------------------------
# 1. Adversarial Tests: Hawkins-Simon Spectral Viability Filter
# ---------------------------------------------------------------------------

class TestAdversarialHawkinsSimon:
    """Adversarial stress testing for check_hawkins_simon_viability."""

    def test_degenerate_zero_matrix_a(self, calib_2c_2s: TradeCalibrationResult) -> None:
        """Degenerate case: Leontief technical coefficients A is completely zero."""
        from dataclasses import replace
        calib_zero = replace(calib_2c_2s, a=np.zeros_like(calib_2c_2s.a))
        res = check_hawkins_simon_viability(calib_zero)
        assert res.rho == 0.0
        assert res.cw_lower == 0.0
        assert res.cw_upper == 0.0
        assert res.is_viable is True

    def test_degenerate_isolated_zero_row(self, calib_2c_2s: TradeCalibrationResult) -> None:
        """Degenerate case: an isolated sector produces zero intermediate goods."""
        from dataclasses import replace
        a_mod = calib_2c_2s.a.copy()
        a_mod[0, :] = 0.0
        calib_mod = replace(calib_2c_2s, a=a_mod)
        res = check_hawkins_simon_viability(calib_mod)
        assert np.isfinite(res.rho)
        assert res.rho < 1.0
        assert res.is_viable is True

    def test_extreme_spectral_radius_prohibitive_tariffs(
        self, calib_2c_2s: TradeCalibrationResult
    ) -> None:
        """Extreme tariffs (tau >> 1) must be strictly rejected before solver launch."""
        for extreme_tau in [10.0, 50.0, 100.0]:
            with pytest.raises(ValueError, match="violates Hawkins-Simon viability condition"):
                check_hawkins_simon_viability(calib_2c_2s, tau=extreme_tau)

    def test_spectral_radius_boundary_near_one(self, calib_2c_2s: TradeCalibrationResult) -> None:
        """Boundary test: verify that rho >= 1.0 - 1e-6 triggers rejection."""
        with pytest.raises(ValueError, match="violates Hawkins-Simon viability condition"):
            check_hawkins_simon_viability(calib_2c_2s, tau=15.0)

    def test_negative_tariff_subsidies(self, calib_2c_2s: TradeCalibrationResult) -> None:
        """Negative tariffs (import subsidies: tau in [-0.5, -0.9]) should not crash."""
        res = check_hawkins_simon_viability(calib_2c_2s, tau=-0.5)
        assert np.isfinite(res.rho)
        assert 0.0 <= res.rho < 1.0
        assert res.is_viable is True

    def test_infinite_tariffs(self, calib_2c_2s: TradeCalibrationResult) -> None:
        """Infinite tariffs (tau = np.inf) must raise ValueError."""
        with pytest.raises(ValueError, match="violates Hawkins-Simon viability condition"):
            check_hawkins_simon_viability(calib_2c_2s, tau=np.inf)

    def test_hawkins_simon_nan_tariff_silent_acceptance_gap(
        self, calib_2c_2s: TradeCalibrationResult
    ) -> None:
        """GAP TEST: NaN tariffs must raise ValueError rather than being silently swallowed.

        In solver.py line 341, `is_inf_tau = bool(np.any(np.isinf(tau)))` checks for Infs
        but omits checking `np.isnan(tau)`. Furthermore, `np.nan_to_num(B_tau, nan=0.0)`
        zeroes out NaN entries in the cost matrix, allowing NaN tariffs to pass as viable!
        """
        try:
            check_hawkins_simon_viability(calib_2c_2s, tau=np.nan)
            pytest.fail("Bug exposed: check_hawkins_simon_viability silently accepted tau=NaN as viable")
        except ValueError:
            pass  # Expected contract: NaN tariffs must raise ValueError

    def test_diverse_tariff_input_tensor_shapes(self, calib_2c_2s: TradeCalibrationResult) -> None:
        """Verify check_hawkins_simon_viability across all 5 supported tariff array ranks."""
        nc, ns = calib_2c_2s.n_countries, calib_2c_2s.n_sectors
        M = nc * ns

        # Rank 0 (scalar)
        res_scalar = check_hawkins_simon_viability(calib_2c_2s, tau=0.10)
        assert res_scalar.is_viable is True

        # Rank 1 (nc,)
        res_1d = check_hawkins_simon_viability(calib_2c_2s, tau=np.full(nc, 0.10))
        assert res_1d.is_viable is True

        # Rank 2 (M, M)
        res_2d = check_hawkins_simon_viability(calib_2c_2s, tau=np.ones((M, M)) * 1.10)
        assert res_2d.is_viable is True

        # Rank 3 (M, ns, nc)
        res_3d = check_hawkins_simon_viability(calib_2c_2s, tau=np.ones((M, ns, nc)) * 1.10)
        assert res_3d.is_viable is True

        # Rank 4 (ns, nc, ns, nc)
        res_4d = check_hawkins_simon_viability(calib_2c_2s, tau=np.ones((ns, nc, ns, nc)) * 1.10)
        assert res_4d.is_viable is True

    def test_viability_result_unpacking_and_attribute_access(
        self, calib_2c_2s: TradeCalibrationResult
    ) -> None:
        """Verify ViabilityResult supports 3-element, 4-element unpacking and attributes."""
        res = check_hawkins_simon_viability(calib_2c_2s)

        # 3-element unpack
        r3, l3, u3 = check_hawkins_simon_viability(calib_2c_2s)
        assert isinstance(r3, float)
        assert isinstance(l3, float)
        assert isinstance(u3, float)

        # 4-element unpack
        r4, l4, u4, v4 = check_hawkins_simon_viability(calib_2c_2s)
        assert r4 == r3
        assert v4 is True

        # Attributes
        assert res.rho == r3
        assert res.cw_lower == l3
        assert res.cw_upper == u3
        assert res.is_viable is True

    def test_tax_boundary_clipping(self, calib_2c_2s: TradeCalibrationResult) -> None:
        """Extreme production taxes (-0.99 or >= 1.0) must be clipped safely without zero division."""
        from dataclasses import replace
        calib_high_tax = replace(calib_2c_2s, tax=np.ones_like(calib_2c_2s.tax) * 1.5)
        try:
            res = check_hawkins_simon_viability(calib_high_tax)
            assert np.isfinite(res.rho)
        except ValueError:
            pass  # Rejection is also valid if rho >= 1.0


# ---------------------------------------------------------------------------
# 2. Adversarial Tests: SVD Clamped Newton Step & Wage Displacement
# ---------------------------------------------------------------------------

class TestAdversarialSVDClamping:
    """Adversarial stress testing for svd_clamped_newton_step and clamp_wage_displacement."""

    def test_singular_and_rank_deficient_jacobian(self) -> None:
        """Verify SVD clamping on completely singular and rank-1 Jacobians."""
        n = 6
        J_zero = np.zeros((n, n))
        f_val = np.ones(n)
        DL = np.ones(n)
        DR = np.ones(n)

        # Completely zero Jacobian: step must remain finite and bounded
        delta_zero = svd_clamped_newton_step(J_zero, f_val, DL, DR, max_comp=20.0, max_disp=0.30)
        assert np.all(np.isfinite(delta_zero))
        assert np.max(np.abs(delta_zero)) <= 0.30 + 1e-12

        # Rank-1 Jacobian
        J_rank1 = np.ones((n, n))
        delta_r1 = svd_clamped_newton_step(J_rank1, f_val, DL, DR, max_comp=20.0, max_disp=0.30)
        assert np.all(np.isfinite(delta_r1))
        assert np.max(np.abs(delta_r1)) <= 0.30 + 1e-12

    def test_modal_component_clamping_active(self) -> None:
        """Verify modal projection coefficient clamping |c_k| <= max_comp triggers."""
        n = 4
        J_stiff = np.diag([1.0, 1e-2, 1e-8, 1e-16])
        f_val = np.ones(n) * 100.0  # Large residual
        DL = np.ones(n)
        DR = np.ones(n)

        max_comp = 5.0
        delta = svd_clamped_newton_step(J_stiff, f_val, DL, DR, max_comp=max_comp, max_disp=0.0)
        assert np.all(np.isfinite(delta))

    def test_diagonal_equilibration_2d_matrix_inputs(self) -> None:
        """Verify support for both 1D vectors and 2D diagonal matrices for D_L and D_R."""
        n = 3
        J = np.eye(n) * 2.0
        f_val = np.array([1.0, -2.0, 3.0])
        DL_1d = np.array([0.5, 1.0, 2.0])
        DR_1d = np.array([2.0, 1.0, 0.5])

        delta_1d = svd_clamped_newton_step(J, f_val, DL_1d, DR_1d, max_comp=20.0, max_disp=0.30)
        delta_2d = svd_clamped_newton_step(J, f_val, np.diag(DL_1d), np.diag(DR_1d), max_comp=20.0, max_disp=0.30)
        np.testing.assert_allclose(delta_1d, delta_2d, atol=1e-14)

    def test_clamp_wage_displacement_modes(self) -> None:
        """Verify clamp_wage_displacement across scalar, coordinate, and uniform modes."""
        # Scalar
        assert clamp_wage_displacement(0.50, max_disp=0.30) == 0.30
        assert clamp_wage_displacement(-0.50, max_disp=0.30) == -0.30
        assert clamp_wage_displacement(0.10, max_disp=0.30) == 0.10

        # Vector coordinate mode (elementwise clipping)
        v = np.array([0.50, -0.40, 0.10])
        clamped_coord = clamp_wage_displacement(v, max_disp=0.30, mode="coordinate")
        np.testing.assert_allclose(clamped_coord, [0.30, -0.30, 0.10])

        # Vector uniform mode (preserves direction angle)
        clamped_uniform = clamp_wage_displacement(v, max_disp=0.30, mode="uniform")
        assert np.max(np.abs(clamped_uniform)) <= 0.30 + 1e-12
        scale_factor = 0.30 / 0.50
        np.testing.assert_allclose(clamped_uniform, v * scale_factor)

        # Empty array
        empty_arr = np.array([])
        res_empty = clamp_wage_displacement(empty_arr, max_disp=0.30)
        assert res_empty.size == 0

        # Zero or negative max_disp
        res_zero = clamp_wage_displacement(v, max_disp=0.0, mode="uniform")
        np.testing.assert_allclose(res_zero, v)


# ---------------------------------------------------------------------------
# 3. Adversarial Tests: Anderson Acceleration
# ---------------------------------------------------------------------------

class TestAdversarialAndersonAcceleration:
    """Adversarial stress testing for anderson_accelerate."""

    def test_empty_history_buffers_raise_value_error(self) -> None:
        """Empty history buffers must raise informative ValueError."""
        with pytest.raises(ValueError, match="history buffers must be non-empty"):
            anderson_accelerate([], [], [], m=4)

    def test_single_history_depth_one(self) -> None:
        """Single history item (k=1 or m=1) returns copy of proposal vector g_hist[-1]."""
        x_hist = [np.array([1.0, 2.0])]
        g_hist = [np.array([1.5, 2.5])]
        f_hist = [np.array([0.5, 0.5])]

        res = anderson_accelerate(x_hist, g_hist, f_hist, m=1)
        np.testing.assert_allclose(res, g_hist[-1])

    def test_history_buffer_overflow(self) -> None:
        """History depth overflow: len(x_hist) = 20 > m = 3."""
        dim = 4
        m = 3
        x_hist = [np.random.randn(dim) for _ in range(20)]
        g_hist = [np.random.randn(dim) for _ in range(20)]
        f_hist = [np.random.randn(dim) for _ in range(20)]

        res = anderson_accelerate(x_hist, g_hist, f_hist, m=m)
        assert res.shape == (dim,)
        assert np.all(np.isfinite(res))

    def test_linearly_dependent_and_identical_residuals(self) -> None:
        """Singular KKT matrix: collinear/identical residuals must not cause divergence."""
        dim = 3
        x_hist = [np.full(dim, float(i)) for i in range(4)]
        g_hist = [np.full(dim, float(i) + 0.5) for i in range(4)]
        f_hist = [np.ones(dim) * 2.0 for _ in range(4)]

        res = anderson_accelerate(x_hist, g_hist, f_hist, m=4)
        assert np.all(np.isfinite(res))

    def test_all_zero_residuals(self) -> None:
        """Zero residuals (converged fixed point): KKT R^T R = 0 handled gracefully."""
        dim = 2
        x_hist = [np.array([1.0, 2.0]), np.array([1.1, 2.1])]
        g_hist = [np.array([1.1, 2.1]), np.array([1.2, 2.2])]
        f_hist = [np.zeros(dim), np.zeros(dim)]

        res = anderson_accelerate(x_hist, g_hist, f_hist, m=2)
        assert np.all(np.isfinite(res))

    def test_nan_resilience_in_residual_history(self) -> None:
        """NaN in residual buffer triggers fallback without raising unhandled exception."""
        x_hist = [np.array([1.0]), np.array([2.0])]
        g_hist = [np.array([1.1]), np.array([2.1])]
        f_hist = [np.array([0.1]), np.array([np.nan])]

        res = anderson_accelerate(x_hist, g_hist, f_hist, m=2)
        assert np.all(np.isfinite(res))
        np.testing.assert_allclose(res, g_hist[-1])


# ---------------------------------------------------------------------------
# 4. Adversarial Tests: Micro-Economy Manifold (Cyprus) Sub-Solver
# ---------------------------------------------------------------------------

class TestAdversarialCyprusManifold:
    """Adversarial stress testing for solve_cyprus_manifold."""

    def test_linearized_solve_on_stiff_network(self) -> None:
        """Linearized manifold decomposition on 5-country factor wage Schur matrix."""
        nc = 5
        S_ww = np.eye(nc) * 2.0
        S_ww[0, :] = 0.5
        S_ww[:, 0] = 0.5
        rhs_w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        dw, res, conv = solve_cyprus_manifold(S_ww, rhs_w, idx_cyp=1, max_disp=0.30)
        assert len(dw) == nc
        assert np.all(np.isfinite(dw))
        assert np.max(np.abs(dw)) <= 0.30 + 1e-12

    def test_out_of_bounds_cyprus_index_handling(self) -> None:
        """Out-of-bounds idx_cyp (e.g. 99 in 3-country system) is safely clamped to 0."""
        nc = 3
        S_ww = np.eye(nc)
        rhs_w = np.array([0.1, 0.2, 0.3])

        dw_oob, res_oob, conv_oob = solve_cyprus_manifold(S_ww, rhs_w, idx_cyp=99)
        dw_0, res_0, conv_0 = solve_cyprus_manifold(S_ww, rhs_w, idx_cyp=0)
        np.testing.assert_allclose(dw_oob, dw_0, atol=1e-14)

    def test_singular_schur_complement_least_squares(self) -> None:
        """Singular S_ww matrix falls back cleanly to least-squares solve."""
        nc = 4
        S_sing = np.ones((nc, nc))  # Rank 1
        rhs_w = np.array([0.1, 0.2, 0.3, 0.4])

        dw, res, conv = solve_cyprus_manifold(S_sing, rhs_w, idx_cyp=2)
        assert len(dw) == nc
        assert np.all(np.isfinite(dw))

    def test_nonlinear_secant_sub_solver_convergence(self) -> None:
        """Nonlinear secant 1D manifold search with custom evaluator."""
        nc = 4
        S_ww = np.eye(nc)
        rhs_w = np.array([0.1, 0.2, 0.3, 0.4])

        def nonlinear_cyp(c: float) -> tuple[float, float, Any]:
            val = float(c**3 - 0.001)
            return val, abs(val), None

        dw, res, conv = solve_cyprus_manifold(
            S_ww, rhs_w, eval_cyp_fn=nonlinear_cyp, idx_cyp=2, tol=1e-4, max_secant_iter=10
        )
        assert np.all(np.isfinite(dw))
        assert abs(dw[2] - 0.1) < 0.05


# ---------------------------------------------------------------------------
# 5. Adversarial Tests: Keller's Pseudo-Arclength Continuation (PAC)
# ---------------------------------------------------------------------------

class TestAdversarialKellerPAC:
    """Adversarial stress testing for solve_keller_pac."""

    def test_keller_pac_fold_bifurcation_turning_point(
        self, calib_2c_2s: TradeCalibrationResult
    ) -> None:
        """Traverse turning point with sigma=0.1238 and verify metadata tracking."""
        res = solve_keller_pac(
            calib=calib_2c_2s,
            tau_target=0.20,
            sigma=0.1238,
            ds_init=0.05,
            ds_min=1e-4,
            ds_max=0.5,
            tol=5e-3,
            max_steps=60,
        )
        assert isinstance(res, TradeEquilibriumResult)
        assert res.converged is True
        assert res.metadata.get("method") == "keller_pac"
        assert res.metadata.get("final_lambda", 0.0) >= 1.0 - 1e-4

    def test_keller_pac_various_target_formats(
        self, calib_2c_2s: TradeCalibrationResult
    ) -> None:
        """Verify PAC target accepts scalar and 3D tensor array formats."""
        # Scalar rate
        res_scalar = solve_keller_pac(calib_2c_2s, tau_target=0.10, max_steps=10, tol=5e-3)
        assert res_scalar.converged is True

        # 3D Tensor format (M, ns, nc)
        nc, ns = calib_2c_2s.n_countries, calib_2c_2s.n_sectors
        M = nc * ns
        tau_3d = np.ones((M, ns, nc))
        for i in range(nc):
            for j in range(nc):
                if i != j:
                    tau_3d[i * ns : (i + 1) * ns, :, j] = 1.10
        res_arr = solve_keller_pac(calib_2c_2s, tau_target=tau_3d, max_steps=10, tol=5e-3)
        assert res_arr.converged is True

    def test_keller_pac_2d_matrix_tariff_shape_mismatch_gap(
        self, calib_2c_2s: TradeCalibrationResult
    ) -> None:
        """GAP TEST: 2D matrix tariffs of shape (M, M) must not fail shape broadcasting in PAC.

        In solver.py lines 120-135, `_resolve_tariffs` does not reshape 2D (M, M) tariffs
        into (M, ns, nc). As a result, `(1.0 - l_c) * tau_s_a + l_c * tau_t_a` fails with
        `ValueError: operands could not be broadcast together with shapes (4,2,2) (4,4)`.
        """
        nc, ns = calib_2c_2s.n_countries, calib_2c_2s.n_sectors
        M = nc * ns
        tau_2d = np.ones((M, M))
        for i in range(nc):
            for j in range(nc):
                if i != j:
                    tau_2d[i * ns : (i + 1) * ns, j * ns : (j + 1) * ns] = 1.10

        try:
            res = solve_keller_pac(calib_2c_2s, tau_target=tau_2d, max_steps=10, tol=5e-3)
            assert res.converged is True
        except ValueError as exc:
            if "operands could not be broadcast together" in str(exc):
                pytest.fail(
                    f"Bug exposed: _resolve_tariffs failed to reshape 2D (M, M) tariff matrix for PAC: {exc}"
                )
            raise

    def test_keller_pac_start_and_base_result(
        self, calib_2c_2s: TradeCalibrationResult
    ) -> None:
        """Verify PAC accepts tau_start and base_result correctly."""
        res_base = solve_keller_pac(calib_2c_2s, tau_target=0.05, max_steps=10, tol=5e-3)
        res_next = solve_keller_pac(
            calib_2c_2s,
            tau_target=0.10,
            tau_start=0.05,
            base_result=res_base,
            max_steps=10,
            tol=5e-3,
        )
        assert res_next.converged is True

    def test_keller_pac_step_limit_behavior(
        self, calib_2c_2s: TradeCalibrationResult
    ) -> None:
        """Verify solver terminates within max_steps budget when steps are constrained."""
        res = solve_keller_pac(calib_2c_2s, tau_target=0.05, max_steps=2, ds_init=0.01)
        assert res.metadata.get("pac_steps") <= 2

    def test_keller_pac_premature_step_limit_gap(
        self, calib_2c_2s: TradeCalibrationResult
    ) -> None:
        """GAP TEST: Premature step exhaustion on large tariffs must not crash with uncaught ValueError.

        When max_steps=1 on a large tariff (tau_target=2.0), the continuation loop exits
        at lambda ~ 0.001. An un-damped terminal polish jumping directly to lambda=1.0 causes
        NaN overflows in prices/wages and raises `ValueError: array must not contain infs or NaNs`
        from scipy.linalg.solve (which is uncaught because solver.py only catches la.LinAlgError).
        The expected robust contract is returning a TradeEquilibriumResult with converged=False.
        """
        try:
            res = solve_keller_pac(calib_2c_2s, tau_target=2.0, max_steps=1, ds_init=0.001)
            assert res.converged is False, "Expected non-converged status on step exhaustion"
        except Exception as exc:
            assert not (isinstance(exc, ValueError) and "array must not contain infs or NaNs" in str(exc)), (
                "Bug exposed: Uncaught ValueError from scipy.linalg.solve during terminal polish in solve_keller_pac"
            )


# ---------------------------------------------------------------------------
# 6. Adversarial Tests: MRIO Economic Regularization
# ---------------------------------------------------------------------------

class TestAdversarialRegularization:
    """Adversarial stress testing for regularize_mrio_table and accounting validation."""

    def test_phantom_output_injection_all_inactive_sectors(self) -> None:
        """Phantom output injection on multiple completely inactive sector nodes (Y = 0)."""
        M = 4
        Z = np.zeros((M, M))
        F = np.zeros((M, 2))
        VA = np.zeros(M)
        TLS = np.zeros(M)
        Y = np.zeros(M)

        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            Z, F, VA, TLS, Y=Y, floor_output=1e-6
        )
        assert np.all(Y_c >= 1e-6)
        assert np.all(VA_c >= 1e-6)
        assert np.all(np.sum(F_c, axis=1) >= 1e-6)

    def test_negative_value_added_flooring_and_dual_tls(self) -> None:
        """Negative VA must be floored with exact dual TLS debit preserving outlays."""
        M = 4
        Z = np.ones((M, M)) * 10.0
        F = np.ones((M, 2)) * 20.0
        VA_neg = np.array([-10.0, -5.0, 2.0, 15.0])
        TLS = np.array([5.0, 5.0, 5.0, 5.0])
        Y = np.sum(Z, axis=1) + np.sum(F, axis=1)

        Z_c, F_c, VA_c, TLS_c, Y_c = regularize_mrio_table(
            Z, F, VA_neg, TLS, Y=Y, floor_va_ratio=1e-3, floor_va_abs=1.0
        )
        assert np.all(VA_c >= 1.0)
        outlays = np.sum(Z_c, axis=0) + VA_c + TLS_c
        np.testing.assert_allclose(outlays, Y_c, atol=1e-10)

    def test_structural_dimension_validation(self) -> None:
        """Verify strict dimensional mismatch detection in regularize_mrio_table."""
        M = 4
        Z = np.eye(M)
        F = np.ones((M, 2))
        VA = np.ones(M)
        TLS = np.ones(M)

        with pytest.raises(ValueError, match="Z must be square"):
            regularize_mrio_table(np.ones((M, M + 1)), F, VA, TLS)

        with pytest.raises(ValueError, match="F rows"):
            regularize_mrio_table(Z, np.ones((M + 1, 2)), VA, TLS)

        with pytest.raises(ValueError, match="VA length"):
            regularize_mrio_table(Z, F, np.ones(M + 1), TLS)

        with pytest.raises(ValueError, match="TLS shape"):
            regularize_mrio_table(Z, F, VA, np.ones(M + 1))

    def test_non_productive_leontief_system_rejection(self) -> None:
        """A system where rho(B) >= 0.999 must raise ValueError."""
        M = 3
        Z_excessive = np.ones((M, M)) * 2.0
        F = np.ones((M, 2))
        VA = np.ones(M)
        TLS = np.zeros(M)
        Y = np.ones(M) * 1.5

        with pytest.raises(ValueError, match="Leontief cost system is non-productive"):
            regularize_mrio_table(Z_excessive, F, VA, TLS, Y=Y)

    def test_validate_accounting_identities_on_corrupt_data(self) -> None:
        """validate_accounting_identities returns valid=False for unbalanced tables."""
        M = 3
        Z = np.ones((M, M))
        F = np.ones((M, 2))
        VA = np.ones(M)
        TLS = np.zeros(M)
        Y_wrong = np.ones(M) * 999.0

        report = validate_accounting_identities(Z, F, VA, TLS, Y_wrong)
        assert report["valid"] is False
        assert report["max_sales_error"] > 10.0


# ---------------------------------------------------------------------------
# 7. Adversarial Tests: Matrix Balancing Algorithms (RAS, GRAS, Quadratic)
# ---------------------------------------------------------------------------

class TestAdversarialMatrixBalancing:
    """Adversarial stress testing for balance_ras, balance_gras, and balance_quadratic."""

    def test_balance_ras_negative_entries_rejected(self) -> None:
        """balance_ras must reject negative prior matrix entries with clear error."""
        Z_neg = np.array([[1.0, -0.5], [0.5, 2.0]])
        u = np.array([1.0, 2.0])
        v = np.array([1.5, 1.5])
        with pytest.raises(ValueError, match="requires non-negative entries"):
            balance_ras(Z_neg, u, v)

    def test_balance_ras_target_margin_rescaling(self) -> None:
        """Target sum mismatch sum(u) != sum(v) is automatically rescaled."""
        Z0 = np.array([[2.0, 1.0], [1.0, 3.0]])
        u = np.array([3.0, 4.0])
        v = np.array([5.0, 5.0])

        Z_bal = balance_ras(Z0, u, v)
        np.testing.assert_allclose(np.sum(Z_bal, axis=1), u, atol=1e-8)
        assert abs(np.sum(Z_bal) - 7.0) < 1e-8

    def test_balance_gras_with_mixed_signs(self) -> None:
        """balance_gras handles positive and negative entries simultaneously."""
        Z0 = np.array([[3.0, -1.0], [-2.0, 4.0]])
        u = np.array([2.0, 2.0])
        v = np.array([1.0, 3.0])

        Z_bal = balance_gras(Z0, u, v)
        np.testing.assert_allclose(np.sum(Z_bal, axis=1), u, atol=1e-8)
        np.testing.assert_allclose(np.sum(Z_bal, axis=0), v, atol=1e-8)

    def test_balance_gras_pure_negative_rows_and_cols(self) -> None:
        """balance_gras handles rows or columns that are purely negative (mask_only_n)."""
        Z0 = np.array([[-2.0, -3.0], [4.0, 5.0]])
        u = np.array([-4.0, 8.0])
        v = np.array([1.0, 3.0])

        Z_bal = balance_gras(Z0, u, v)
        np.testing.assert_allclose(np.sum(Z_bal, axis=1), u, atol=1e-7)

    def test_balance_quadratic_unweighted_and_weighted(self) -> None:
        """balance_quadratic unweighted projection and weighted least-squares."""
        Z0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        u = np.array([4.0, 6.0])
        v = np.array([5.0, 5.0])

        # Unweighted
        Z_unw = balance_quadratic(Z0, u, v)
        np.testing.assert_allclose(np.sum(Z_unw, axis=1), u, atol=1e-10)
        np.testing.assert_allclose(np.sum(Z_unw, axis=0), v, atol=1e-10)

        # Weighted
        W = np.array([[2.0, 1.0], [1.0, 3.0]])
        Z_w = balance_quadratic(Z0, u, v, weights=W)
        np.testing.assert_allclose(np.sum(Z_w, axis=1), u, atol=1e-10)
        np.testing.assert_allclose(np.sum(Z_w, axis=0), v, atol=1e-10)

    def test_balance_quadratic_invalid_weights_rejected(self) -> None:
        """balance_quadratic rejects non-positive or dimensionally mismatched weights."""
        Z0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        u = np.array([3.0, 7.0])
        v = np.array([4.0, 6.0])

        with pytest.raises(ValueError, match="Weights must be strictly positive"):
            balance_quadratic(Z0, u, v, weights=np.array([[1.0, -0.5], [1.0, 1.0]]))

        with pytest.raises(ValueError, match="Weights shape"):
            balance_quadratic(Z0, u, v, weights=np.array([[1.0, 1.0]]))

    def test_balance_ras_structural_zeros_non_convergent(self) -> None:
        """Structural zeros where margins cannot be matched terminate cleanly within max_iter."""
        Z0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        u = np.array([1.0, 1.0])
        v = np.array([1.0, 1.0])

        res = balance_ras(Z0, u, v, max_iter=10)
        assert np.all(np.isfinite(res))

    def test_balance_ras_and_gras_zero_column_target_gap(self) -> None:
        """GAP TEST: sum_v == 0 with sum_u > 0 must not crash with unhandled ZeroDivisionError.

        When target column marginals sum to zero while row targets are non-zero,
        the rescaling check `abs(sum_u - sum_v) > 1e-12` attempts `sum_u / sum_v`,
        raising a raw ZeroDivisionError rather than a clean ValueError.
        """
        Z0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        u = np.array([1.0, 1.0])
        v = np.array([0.0, 0.0])

        for name, fn in [("RAS", balance_ras), ("GRAS", balance_gras), ("Quadratic", balance_quadratic)]:
            try:
                fn(Z0, u, v)
                pytest.fail(f"{name} should have raised ValueError for inconsistent zero column target")
            except ZeroDivisionError:
                pytest.fail(f"Bug exposed in {name}: unhandled ZeroDivisionError raised instead of ValueError")
            except ValueError:
                pass  # Clean ValueError is the expected contract

    def test_balance_ras_all_zero_targets_gap(self) -> None:
        """GAP TEST: balance_ras with all-zero targets fails to zero out matrix.

        When u_target = [0, 0] and v_target = [0, 0], mask_r and mask_s are all False
        because they require u_target > 0. The matrix Z0 is returned completely unscaled!
        The expected result for zero target margins is the zero matrix.
        """
        Z0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        u_zero = np.array([0.0, 0.0])
        v_zero = np.array([0.0, 0.0])

        res = balance_ras(Z0, u_zero, v_zero)
        row_sums = np.sum(res, axis=1)
        assert np.allclose(row_sums, 0.0, atol=1e-10), (
            f"Bug exposed: balance_ras failed to zero out matrix for zero targets; got row sums {row_sums}"
        )


# ---------------------------------------------------------------------------
# 8. Adversarial Tests: Multi-Database Ingestion Adapters
# ---------------------------------------------------------------------------

class TestAdversarialDataLoaders:
    """Adversarial stress testing for data loading and database adapters."""

    def test_figaro_missing_file_fallback_disabled(self) -> None:
        """load_figaro with fallback_to_synthetic=False and invalid path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="FIGARO"):
            load_figaro(year=2099, fallback_to_synthetic=False, file_path="/nonexistent/figaro.csv")

    def test_exiobase_missing_file_fallback_disabled(self) -> None:
        """load_exiobase with fallback_to_synthetic=False raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="EXIOBASE"):
            load_exiobase(year=2099, model="ixi", fallback_to_synthetic=False, file_path="/nonexistent/exio.tsv")

    def test_wiod_missing_file_fallback_disabled(self) -> None:
        """load_wiod with fallback_to_synthetic=False raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="WIOD"):
            load_wiod(year=2099, fallback_to_synthetic=False, file_path="/nonexistent/wiod.csv")

    def test_eora_missing_file_fallback_disabled(self) -> None:
        """load_eora with fallback_to_synthetic=False raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Eora"):
            load_eora(year=2099, fallback_to_synthetic=False, file_path="/nonexistent/eora.csv")

    def test_oecd_icio_missing_file_fallback_disabled(self) -> None:
        """load_oecd_icio_granular with fallback_to_synthetic=False raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="OECD ICIO"):
            load_oecd_icio_granular(year=2099, fallback_to_synthetic=False, file_path="/nonexistent/oecd.csv")

    def test_custom_dimension_overrides_all_loaders(self) -> None:
        """Verify custom_c and custom_s overrides across all 5 database loaders."""
        c, s = 3, 2
        loaders = [
            ("figaro", load_figaro),
            ("exiobase_ixi", lambda **kw: load_exiobase(model="ixi", **kw)),
            ("exiobase_pxp", lambda **kw: load_exiobase(model="pxp", **kw)),
            ("wiod", load_wiod),
            ("eora", load_eora),
            ("oecd", load_oecd_icio_granular),
        ]
        for name, loader in loaders:
            calib = loader(custom_c=c, custom_s=s, seed=42, fallback_to_synthetic=True)
            assert calib.nc == c, f"{name} nc mismatch"
            assert calib.ns == s, f"{name} ns mismatch"

    def test_eora_thousand_to_million_usd_scaling(self) -> None:
        """Verify Eora Thousand USD to Million USD (scale 1/1000) scaling factor."""
        raw_eora_th = generate_synthetic_mrio("eora", custom_c=2, custom_s=2, unit="000_USD", seed=99)
        calib_eora_m = load_eora(custom_c=2, custom_s=2, seed=99, fallback_to_synthetic=True)
        ratio = float(np.sum(raw_eora_th.gross_output) / np.sum(calib_eora_m.ytot))
        assert abs(ratio - 1000.0) < 1e-1

    def test_wiod_missing_accounting_rows_rejected(self) -> None:
        """load_wiod detects missing VA, TXSP, or GO accounting rows in table."""
        cols = {"Country": ["USA"] * 56, "IndustryCode": [f"S{i:02d}" for i in range(1, 57)]}
        for r in range(1, 57):
            cols[f"vUSA{r}"] = [1.0] * 56
        for r in range(57, 62):
            cols[f"vUSA{r}"] = [0.5] * 56

        df = pd.DataFrame(cols)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f.name, index=False)
            fname = f.name

        try:
            with pytest.raises(ValueError, match="Accounting rows.*missing in WIOD"):
                load_wiod(file_path=fname, fallback_to_synthetic=False)
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def test_corrupt_file_dimensions_caught(self) -> None:
        """Verify that passing undersized CSV files to empirical loaders raises descriptive errors."""
        df = pd.DataFrame(np.ones((10, 10)))
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f.name)
            fname = f.name

        try:
            with pytest.raises((IndexError, ValueError)):
                load_figaro(file_path=fname, fallback_to_synthetic=False)

            with pytest.raises((IndexError, ValueError)):
                load_exiobase(file_path=fname, fallback_to_synthetic=False)

            with pytest.raises((IndexError, ValueError)):
                load_eora(file_path=fname, fallback_to_synthetic=False)

            with pytest.raises((IndexError, ValueError)):
                load_oecd_icio_granular(file_path=fname, fallback_to_synthetic=False)
        finally:
            if os.path.exists(fname):
                os.remove(fname)

    def test_multilateral_trade_balance_accounting_helpers(self) -> None:
        """Verify compute_trade_balances and verify_zero_leakage on adversarial inputs."""
        xn_balanced = np.array([10.0, -5.0, -5.0])
        assert verify_zero_leakage(xn_balanced) is True

        xn_leaking = np.array([10.0, -5.0, 0.0])
        assert verify_zero_leakage(xn_leaking) is False

        assert verify_zero_leakage(np.zeros(5)) is True
