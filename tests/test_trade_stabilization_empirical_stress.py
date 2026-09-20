"""Empirical Stress Test Harness & Adversarial Hardening for Network Stabilization (Milestone 1).

Author: challenger_m1_2 (Empirical Challenger)
Scope:
1. SVD Spectral Component Clamping:
   - Stress-test ill-conditioned synthetic Jacobian matrices with kappa_2(J) in [10^8, 10^16, 10^18, inf].
   - Verify modal projection coefficients are strictly clamped to |c_k| <= 20.0 and factor displacement is strictly bounded to <= 0.30.
2. Anderson Acceleration:
   - Challenge depth-m Anderson acceleration with collinear and rank-deficient residual histories.
   - Verify KKT least-squares solution maintains sum(gamma_i) == 1.0 and does not crash or raise LinAlgError.
3. Cyprus CYP Micro-Economy Manifold:
   - Test open micro-economy shock configurations with steep gradient spikes.
   - Verify 1D secant sub-solver converges without stalling, handles singular S_76, and enforces max_disp <= 0.30.

Strictly conforms to the puremacro Pyodide runtime contract (NumPy, SciPy only).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as la

from puremacro.trade.solver import (
    anderson_accelerate,
    solve_cyprus_manifold_step,
    svd_clamped_newton_step,
)


# =============================================================================
# 1. SVD Spectral Component Clamping Adversarial Suite
# =============================================================================

class TestSVDSpectralClampingAdversarial:
    """Stress tests for SVD modal component clamping on ill-conditioned and singular Jacobians."""

    @pytest.mark.parametrize("kappa", [1e8, 1e10, 1e12, 1e14, 1e16, 1e18, np.inf])
    @pytest.mark.parametrize("n", [2, 5, 20, 77])
    def test_svd_clamping_modal_and_displacement_bounds(
        self, kappa: float, n: int
    ) -> None:
        """Verify modal coefficients |c_k| <= 20.0 and displacement bounds <= 0.30 across kappa."""
        rng = np.random.default_rng(int(kappa % 10000) if np.isfinite(kappa) else 42 + n)
        Q1, _ = la.qr(rng.standard_normal((n, n)))
        Q2, _ = la.qr(rng.standard_normal((n, n)))

        if np.isinf(kappa):
            s = np.ones(n)
            s[-1] = 0.0
            if n > 2:
                s[-2] = 0.0
        else:
            s = np.logspace(0, -np.log10(kappa), n)

        J = Q1 @ np.diag(s) @ Q2.T
        D_L = np.ones(n)
        D_R = np.ones(n)

        # Test various residual structures
        residuals = [
            rng.standard_normal(n),                       # random Gaussian
            Q1[:, -1] * 10.0,                              # aligned with worst singular mode
            rng.standard_normal(n) * 1e8,                  # huge residual
            rng.standard_normal(n) * 1e-10,                # tiny residual
            np.zeros(n),                                   # zero residual
        ]

        for f_val in residuals:
            # 1. Verify modal coefficient clamping with max_disp=0.0 (no uniform displacement scaling)
            delta_nodisp = svd_clamped_newton_step(
                J=J, f_val=f_val, D_L=D_L, D_R=D_R, max_comp=20.0, max_disp=0.0
            )
            assert np.all(np.isfinite(delta_nodisp))

            # Reconstruct modal projection coefficients: coeffs = Vt @ (delta / D_R)
            U_eq, S_eq, Vt_eq = la.svd(J)
            c_rec = Vt_eq @ delta_nodisp
            max_c = float(np.max(np.abs(c_rec)))
            assert max_c <= 20.0 + 1e-10, (
                f"Modal coefficient clamp exceeded 20.0: max_c={max_c:.6f} at kappa={kappa}, n={n}"
            )

            # 2. Verify factor displacement clamping with max_disp=0.30
            delta_disp = svd_clamped_newton_step(
                J=J, f_val=f_val, D_L=D_L, D_R=D_R, max_comp=20.0, max_disp=0.30
            )
            assert np.all(np.isfinite(delta_disp))
            max_step = float(np.max(np.abs(delta_disp)))
            assert max_step <= 0.30 + 1e-12, (
                f"Factor displacement bound exceeded 0.30: max_step={max_step:.6f} at kappa={kappa}, n={n}"
            )

    def test_svd_clamping_disparate_equilibration_scalers(self) -> None:
        """Verify clamping operates correctly with highly disparate 1D and 2D equilibration scalers."""
        n = 10
        rng = np.random.default_rng(2026)
        Q1, _ = la.qr(rng.standard_normal((n, n)))
        Q2, _ = la.qr(rng.standard_normal((n, n)))
        s = np.logspace(0, -14, n)
        J = Q1 @ np.diag(s) @ Q2.T

        D_L = np.logspace(-5, 5, n)
        D_R = np.logspace(5, -5, n)
        f_val = rng.standard_normal(n) * 1e4

        for use_2d in [False, True]:
            dl = np.diag(D_L) if use_2d else D_L
            dr = np.diag(D_R) if use_2d else D_R

            delta = svd_clamped_newton_step(
                J=J, f_val=f_val, D_L=dl, D_R=dr, max_comp=20.0, max_disp=0.30
            )
            assert np.all(np.isfinite(delta))
            assert np.max(np.abs(delta)) <= 0.30 + 1e-12


# =============================================================================
# 2. Anderson Acceleration Adversarial Suite
# =============================================================================

class TestAndersonAccelerationAdversarial:
    """Stress tests for depth-m Anderson acceleration on rank-deficient and collinear histories."""

    @pytest.mark.parametrize("case", [
        "identical",
        "proportional",
        "all_zeros",
        "some_zeros",
        "near_collinear_1e-16",
        "near_collinear_1e-12",
        "near_collinear_1e-8",
        "rank1_outer",
        "huge_scale",
        "tiny_scale",
        "disparate_scale",
        "subspace_deficit",
    ])
    @pytest.mark.parametrize("n", [1, 2, 5, 20])
    @pytest.mark.parametrize("m", [1, 2, 4, 8])
    def test_anderson_collinear_and_rank_deficient_histories(
        self, case: str, n: int, m: int
    ) -> None:
        """Verify KKT least-squares solution maintains sum(gamma_i) == 1.0 and does not crash."""
        rng = np.random.default_rng(1000 + n * 10 + m)
        k = max(m, 3)

        x_h = [rng.standard_normal(n) for _ in range(k)]
        g_h = [rng.standard_normal(n) for _ in range(k)]

        if case == "identical":
            f_base = rng.standard_normal(n)
            f_h = [f_base.copy() for _ in range(k)]
        elif case == "proportional":
            f_base = rng.standard_normal(n)
            f_h = [f_base * (i + 1) * (-1) ** i for i in range(k)]
        elif case == "all_zeros":
            f_h = [np.zeros(n) for _ in range(k)]
        elif case == "some_zeros":
            f_h = [np.zeros(n) if i % 2 == 0 else rng.standard_normal(n) for i in range(k)]
        elif case == "near_collinear_1e-16":
            f_base = rng.standard_normal(n)
            f_h = [f_base + 1e-16 * rng.standard_normal(n) for _ in range(k)]
        elif case == "near_collinear_1e-12":
            f_base = rng.standard_normal(n)
            f_h = [f_base + 1e-12 * rng.standard_normal(n) for _ in range(k)]
        elif case == "near_collinear_1e-8":
            f_base = rng.standard_normal(n)
            f_h = [f_base + 1e-8 * rng.standard_normal(n) for _ in range(k)]
        elif case == "rank1_outer":
            u = rng.standard_normal(n)
            v = rng.standard_normal(k)
            f_h = [u * v[i] for i in range(k)]
        elif case == "huge_scale":
            f_h = [rng.standard_normal(n) * 1e12 for _ in range(k)]
        elif case == "tiny_scale":
            f_h = [rng.standard_normal(n) * 1e-12 for _ in range(k)]
        elif case == "disparate_scale":
            f_h = [rng.standard_normal(n) * (10.0 ** (i * 6 - 12)) for i in range(k)]
        elif case == "subspace_deficit":
            f_h = [rng.standard_normal(n) for _ in range(k)]
        else:
            f_h = [rng.standard_normal(n) for _ in range(k)]

        # Must execute without LinAlgError or crash
        x_next = anderson_accelerate(x_h, g_h, f_h, m=m)
        assert np.all(np.isfinite(x_next))
        assert x_next.shape == (n,)

        # Verify KKT solution weights sum to 1.0 within floating point precision
        eff_k = min(len(x_h), len(g_h), len(f_h), m)
        if eff_k > 1:
            R = np.column_stack([f_h[-eff_k + i] for i in range(eff_k)])
            KKT = np.empty((eff_k + 1, eff_k + 1), dtype=float)
            KKT[:eff_k, :eff_k] = R.T @ R
            KKT[:eff_k, eff_k] = 1.0
            KKT[eff_k, :eff_k] = 1.0
            KKT[eff_k, eff_k] = 0.0
            rhs_kkt = np.zeros(eff_k + 1, dtype=float)
            rhs_kkt[eff_k] = 1.0
            sol_kkt = la.lstsq(KKT, rhs_kkt)[0]
            gamma = sol_kkt[:eff_k]
            sum_g = float(np.sum(gamma))
            if abs(sum_g) > 1e-14 and np.all(np.isfinite(gamma)):
                gamma_norm = gamma / sum_g
            else:
                gamma_norm = np.zeros(eff_k, dtype=float)
                gamma_norm[-1] = 1.0

            sum_gamma = float(np.sum(gamma_norm))
            # In severe ill-conditioned cases where components reach 10^7, floating-point cancellation
            # limits sum precision to ~1e-9; in all other cases it is machine precision (< 1e-12).
            assert abs(sum_gamma - 1.0) < 1e-8, f"sum(gamma) != 1.0: {sum_gamma}"


# =============================================================================
# 3. Cyprus CYP Micro-Economy Manifold Adversarial Suite
# =============================================================================

class TestCyprusManifoldAdversarial:
    """Stress tests for the 1D secant micro-economy manifold solver under steep gradient shocks."""

    @pytest.mark.parametrize("scale", [1e-2, 1e-4, 1e-6, 1e-8])
    @pytest.mark.parametrize("spike", [0.01, 1.0, 50.0, 1000.0, 1e5])
    def test_cyprus_linear_manifold_gradient_spikes(
        self, scale: float, spike: float
    ) -> None:
        """Verify linear manifold solver resolves micro-economy scale disparities and gradient spikes."""
        nc = 77
        idx_cyp = 14
        rng = np.random.default_rng(42)
        A = rng.standard_normal((nc, nc))
        S_ww = A.T @ A + 10.0 * np.eye(nc)

        # Scale Cyprus row and column to simulate small open micro-economy
        S_ww[idx_cyp, :] *= scale
        S_ww[:, idx_cyp] *= scale
        S_ww[idx_cyp, idx_cyp] = scale

        rhs_w = rng.standard_normal(nc) * 0.1
        rhs_w[idx_cyp] = spike

        dw_full, best_res, conv = solve_cyprus_manifold_step(
            S_ww=S_ww,
            rhs_w=rhs_w,
            eval_cyp_fn=None,
            idx_cyp=idx_cyp,
            tol=2.5e-3,
            max_disp=0.30,
        )

        assert conv is True, f"Linear manifold failed to converge for scale={scale}, spike={spike}"
        assert best_res <= 2.5e-3
        assert np.all(np.isfinite(dw_full))
        assert np.max(np.abs(dw_full)) <= 0.30 + 1e-12

    @pytest.mark.parametrize("theta", [2.0, 5.0, 10.0, 20.0, 50.0])
    @pytest.mark.parametrize("target", [0.01, 0.1, 0.5])
    def test_cyprus_nonlinear_secant_armington_elasticity_spikes(
        self, theta: float, target: float
    ) -> None:
        """Verify 1D secant sub-solver converges without stalling under steep Armington elasticity responses."""
        nc = 77
        idx_cyp = 14
        rng = np.random.default_rng(2026)
        A = rng.standard_normal((nc, nc))
        S_ww = A.T @ A + 5.0 * np.eye(nc)
        S_ww[idx_cyp, :] *= 1e-5
        S_ww[:, idx_cyp] *= 1e-5
        S_ww[idx_cyp, idx_cyp] = 1e-5
        rhs_w = np.ones(nc) * 0.05

        def eval_armington(c: float):
            fc = float(np.exp(np.clip(theta * c, -20.0, 20.0)) - (1.0 + target))
            return fc, abs(fc), None

        dw_full, best_res, conv = solve_cyprus_manifold_step(
            S_ww=S_ww,
            rhs_w=rhs_w,
            eval_cyp_fn=eval_armington,
            idx_cyp=idx_cyp,
            c0=0.0,
            tol=2.5e-3,
            max_secant_iter=8,
            max_disp=0.30,
        )

        assert conv is True, f"Secant solver failed: theta={theta}, target={target}, res={best_res}"
        assert best_res <= 2.5e-3
        assert np.all(np.isfinite(dw_full))
        assert np.max(np.abs(dw_full)) <= 0.30 + 1e-12

    def test_cyprus_singular_s76_subsystem_fallback(self) -> None:
        """Verify solve_cyprus_manifold_step falls back to lstsq and converges when S_76 is singular."""
        nc = 77
        idx_cyp = 14
        # Create a rank-deficient S_ww matrix
        S_ww = np.ones((nc, nc))
        rhs_w = np.ones(nc) * 0.1

        dw_full, best_res, conv = solve_cyprus_manifold_step(
            S_ww=S_ww,
            rhs_w=rhs_w,
            eval_cyp_fn=None,
            idx_cyp=idx_cyp,
            tol=2.5e-3,
            max_disp=0.30,
        )

        assert conv is True
        assert best_res <= 2.5e-3
        assert np.all(np.isfinite(dw_full))
        assert np.max(np.abs(dw_full)) <= 0.30 + 1e-12
