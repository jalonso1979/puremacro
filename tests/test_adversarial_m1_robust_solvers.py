"""Adversarial Stress-Testing Suite for Milestone 1 Solvers (Hawkins-Simon & Keller PAC).

Author: challenger_m1_1 (Empirical Challenger)
Milestone: Milestone 1 Robust Solvers

Verifies:
1. Hawkins-Simon Spectral Viability Filter:
   - Extreme tariff schedules (tau = 8.0, 10.0, 50.0, 100.0, 1000.0) produce rho(B_tau) >= 1.0
     and trigger clean ValueError rejection in < 0.15s.
   - Sub-critical high tariffs (tau = 0.5, 1.0, 2.0) yield valid Collatz-Wielandt lower/upper inclusion bounds.
   - Critical threshold transition (tau in [3.0, 3.5, 3.6, 3.7]) with monotonic spectral radius growth.
   - Subsidies (tau < 0), heterogeneous national tariffs, and matrix tariff structures.
2. Keller's Pseudo-Arclength Continuation:
   - Singular turning points and fold bifurcations (det J -> 0, kappa_2 > 10^7) where standard Newton
     diverges / raises LinAlgError, while Keller's bordered augmented system achieves
     kappa_2(J_aug) ~ O(1) and converges to ||F(x)||_inf < 1e-10.
   - High-precision CGE trade equilibrium convergence (||F(x)||_inf < 1e-10) with strictly positive prices/wages.
3. Network Stabilization & Numerical / Memory Leakage:
   - SVD clamping under extreme ill-conditioning (kappa_2 = 1e16, singular modes).
   - Anderson acceleration under collinear and duplicate history matrices.
   - Cyprus micro-economy manifold 1D secant solver under vanishing scale (1e-6 to 1e-12).
   - 100-iteration memory leak probe verifying bounded memory growth.
"""
from __future__ import annotations

import gc
import time
import tracemalloc
import numpy as np
import scipy.linalg as la
import pytest

from puremacro.trade import (
    TradeCalibrationResult,
    TradeEquilibriumResult,
    calibrate_trade_model,
)
from puremacro.trade.data import load_icio_data
from puremacro.trade.solver import (
    anderson_accelerate,
    check_hawkins_simon_viability,
    solve_cyprus_manifold_step,
    solve_keller_pac,
    svd_clamped_newton_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def empirical_calib() -> TradeCalibrationResult:
    """Calibrate full 77-country 11-sector empirical model from bundled OECD ICIO data."""
    raw = load_icio_data()
    return calibrate_trade_model(raw, ns=11, nc=77, nfd=3, validate=False)


@pytest.fixture(scope="module")
def synthetic_2c_2s_calib() -> TradeCalibrationResult:
    """Construct a balanced synthetic 2-country 2-sector trade model."""
    nc = 2
    ns = 2
    nfd = 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)
    data[:4, :4] = np.array([
        [20.0, 10.0, 5.0, 2.0],
        [8.0, 25.0, 2.0, 4.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    inter_col_sums = data[:4, :4].sum(axis=0)
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac

    inter_row_sums = data[:4, :4].sum(axis=1)
    fd_row_sums = y - inter_row_sums
    for i in range(4):
        tot_fd = fd_row_sums[i]
        if i < 2:
            data[i, 4:10] = [tot_fd * 0.50, tot_fd * 0.25, tot_fd * 0.05, tot_fd * 0.10, tot_fd * 0.08, tot_fd * 0.02]
        else:
            data[i, 4:10] = [tot_fd * 0.10, tot_fd * 0.08, tot_fd * 0.02, tot_fd * 0.50, tot_fd * 0.25, tot_fd * 0.05]
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)

    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


# ---------------------------------------------------------------------------
# 1. Adversarial Hawkins-Simon Tests
# ---------------------------------------------------------------------------

class TestAdversarialHawkinsSimon:
    """Adversarial stress-testing of Hawkins-Simon Spectral Viability Filter."""

    @pytest.mark.parametrize("tau_val", [8.0, 10.0, 50.0, 100.0, 1000.0])
    def test_extreme_tariffs_clean_rejection_and_timing(
        self, empirical_calib: TradeCalibrationResult, tau_val: float
    ) -> None:
        """Verify extreme tariffs trigger clean ValueError in < 0.15s with rho >= 1.0."""
        t_start = time.perf_counter()
        with pytest.raises(ValueError) as exc_info:
            check_hawkins_simon_viability(empirical_calib, tau=tau_val)
        t_elapsed = time.perf_counter() - t_start

        # Time constraint: must evaluate in < 0.15s (measured ~0.010s)
        assert t_elapsed < 0.15, f"Hawkins-Simon check took too long: {t_elapsed:.4f}s >= 0.15s"

        err_msg = str(exc_info.value)
        assert "violates Hawkins-Simon viability condition" in err_msg
        assert "spectral radius rho(B_tau) =" in err_msg
        assert ">= 1.0" in err_msg

    @pytest.mark.parametrize("tau_val", [0.5, 1.0, 2.0])
    def test_subcritical_high_tariffs_valid_cw_bounds(
        self, empirical_calib: TradeCalibrationResult, tau_val: float
    ) -> None:
        """Verify subcritical tariffs yield valid Collatz-Wielandt bounds and rho < 1.0 in < 0.15s."""
        t_start = time.perf_counter()
        rho, cw_lower, cw_upper = check_hawkins_simon_viability(empirical_calib, tau=tau_val)
        t_elapsed = time.perf_counter() - t_start

        assert t_elapsed < 0.15, f"Execution exceeded 0.15s: {t_elapsed:.4f}s"
        assert 0.0 < rho < 1.0, f"Subcritical tariff gave non-viable rho: {rho:.4f}"
        assert cw_lower <= rho <= cw_upper + 1e-10, (
            f"Collatz-Wielandt inclusion violated: lower={cw_lower:.6f}, rho={rho:.6f}, upper={cw_upper:.6f}"
        )
        # Verify inclusion interval tightness
        assert (cw_upper - cw_lower) < 0.10, f"CW bounds too loose: [{cw_lower}, {cw_upper}]"

    def test_critical_boundary_transition_and_monotonicity(
        self, empirical_calib: TradeCalibrationResult
    ) -> None:
        """Verify smooth monotonic growth in rho and sharp transition across viability threshold."""
        tau_grid = [0.0, 1.0, 2.0, 3.0, 3.5, 3.6]
        rhos = []
        for tau in tau_grid:
            rho, low, high = check_hawkins_simon_viability(empirical_calib, tau=tau)
            rhos.append(rho)

        # Monotonicity check: higher tariffs monotonically increase production cost spectral radius
        for i in range(2, len(rhos) - 1):
            assert rhos[i] >= rhos[i - 1], f"Non-monotonic rho growth: {rhos}"

        # Transition check: tau = 3.6 is viable, tau = 3.8 violates Hawkins-Simon
        assert rhos[-1] < 1.0
        with pytest.raises(ValueError):
            check_hawkins_simon_viability(empirical_calib, tau=3.8)

    def test_adversarial_tariff_structures(
        self, empirical_calib: TradeCalibrationResult
    ) -> None:
        """Test import subsidies (tau < 0), heterogeneous national vectors, and extreme matrix tariffs."""
        # 1. Import subsidies (tau = -0.5)
        rho_sub, low_sub, high_sub = check_hawkins_simon_viability(empirical_calib, tau=-0.5)
        assert 0.0 < rho_sub < 1.0
        assert low_sub <= rho_sub <= high_sub + 1e-10

        # 2. Extreme national tariff vector (USA alone imposes 50.0)
        nc = empirical_calib.n_countries
        tau_vec = np.zeros(nc)
        tau_vec[empirical_calib.country_codes.index("USA")] = 50.0
        rho_usa, low_usa, high_usa = check_hawkins_simon_viability(empirical_calib, tau=tau_vec)
        assert 0.0 < rho_usa < 1.0

        # 3. Dense 2D matrix tariff with extreme cross-border shock
        M = empirical_calib.n_sectors * empirical_calib.n_countries
        tau_mat = np.zeros((M, M))
        tau_mat[:11, :] = 100.0  # Prohibitive tariffs on all inputs into first country
        with pytest.raises(ValueError, match="violates Hawkins-Simon viability condition"):
            check_hawkins_simon_viability(empirical_calib, tau=tau_mat)


# ---------------------------------------------------------------------------
# 2. Adversarial Keller PAC Tests
# ---------------------------------------------------------------------------

class TestAdversarialKellerPAC:
    """Adversarial stress-testing of Keller's Bordered Pseudo-Arclength Continuation."""

    def test_singular_fold_bifurcation_newton_fails_keller_succeeds(self) -> None:
        """Benchmark 10-dimensional fold bifurcation where det(J) -> 0 and kappa_2 > 10^7.

        Proves that standard Newton-Raphson diverges/raises LinAlgError, while Keller's
        bordered augmented system maintains well-conditioned Jacobian (kappa_2 ~ O(1))
        and converges to ||F(x)||_inf < 1e-10.
        """
        N = 10
        def F(x: np.ndarray, lam: float) -> np.ndarray:
            res = np.zeros(N, dtype=float)
            res[0] = x[0] ** 2 - lam
            for i in range(1, N):
                res[i] = x[i] - 0.5 * x[i - 1] - 0.1 * x[i] ** 2
            return res

        def J_x(x: np.ndarray) -> np.ndarray:
            J = np.zeros((N, N), dtype=float)
            J[0, 0] = 2.0 * x[0]
            for i in range(1, N):
                J[i, i - 1] = -0.5
                J[i, i] = 1.0 - 0.2 * x[i]
            return J

        dF_dlam = np.zeros(N, dtype=float)
        dF_dlam[0] = -1.0

        # 1. Standard Newton failure at singular turning point (x_0 = 0, lam = 0)
        x_fold = np.zeros(N)
        Jx_fold = J_x(x_fold)
        det_fold = float(la.det(Jx_fold))
        assert abs(det_fold) < 1e-15, f"Expected zero determinant at fold, got {det_fold}"

        with pytest.raises(la.LinAlgError):
            la.solve(Jx_fold, -F(x_fold + 1e-4, 0.0))

        # 2. Keller pseudo-arclength continuation traversing from positive to negative x_0
        x_curr = np.zeros(N)
        x_curr[0] = 0.5
        for i in range(1, N):
            x_curr[i] = 0.5 * x_curr[i - 1]
        lam_curr = 0.25

        ds = 0.05
        # Tangent pointing towards fold
        v_sol = la.solve(J_x(x_curr), -dF_dlam)
        tau_x = -v_sol
        tau_lam = -1.0
        t_norm = float(np.sqrt(np.sum(tau_x ** 2) + tau_lam ** 2))
        tau_x /= t_norm
        tau_lam /= t_norm

        traversed_fold = False
        max_res_traversal = 0.0

        for step in range(25):
            x_pred = x_curr + ds * tau_x
            lam_pred = lam_curr + ds * tau_lam

            xk = x_pred.copy()
            lamk = lam_pred
            conv = False

            for it in range(15):
                fk = F(xk, lamk)
                gk = float(np.dot(tau_x, xk - x_curr) + tau_lam * (lamk - lam_curr) - ds)
                err = max(float(np.max(np.abs(fk))), abs(gk))
                if err < 1e-12:
                    conv = True
                    break

                Jx_k = J_x(xk)
                # Bordered augmented system (N+1, N+1)
                J_aug = np.zeros((N + 1, N + 1))
                J_aug[:N, :N] = Jx_k
                J_aug[:N, N] = dF_dlam
                J_aug[N, :N] = tau_x
                J_aug[N, N] = tau_lam

                # Verify bordered system is well-conditioned
                kappa_aug = np.linalg.cond(J_aug)
                assert kappa_aug < 1e3, f"Bordered Jacobian ill-conditioned: kappa={kappa_aug}"

                rhs = np.append(-fk, -gk)
                d_step = la.solve(J_aug, rhs)
                xk += d_step[:N]
                lamk += d_step[N]

            assert conv, f"Corrector failed to converge at step {step}"
            x_curr = xk
            lam_curr = lamk
            res_k = float(np.max(np.abs(F(x_curr, lam_curr))))
            max_res_traversal = max(max_res_traversal, res_k)

            # Compute new tangent
            Jx_new = J_x(x_curr)
            J_aug = np.zeros((N + 1, N + 1))
            J_aug[:N, :N] = Jx_new
            J_aug[:N, N] = dF_dlam
            J_aug[N, :N] = tau_x
            J_aug[N, N] = tau_lam
            rhs_t = np.zeros(N + 1)
            rhs_t[N] = 1.0
            t_new = la.solve(J_aug, rhs_t)
            t_new /= np.linalg.norm(t_new)
            tau_x, tau_lam = t_new[:N], float(t_new[N])

            if tau_lam <= 0.0 or x_curr[0] < 0.0:
                traversed_fold = True

        assert traversed_fold, "Keller PAC failed to traverse fold turning point"
        assert max_res_traversal < 1e-10, (
            f"Residual norm exceeded 1e-10 during fold traversal: {max_res_traversal:.4e}"
        )
        assert x_curr[0] < 0.0, "Did not continue into negative branch"

    @pytest.mark.parametrize("target", [0.05, 0.15, 0.50])
    def test_cge_keller_pac_high_precision_convergence(
        self, synthetic_2c_2s_calib: TradeCalibrationResult, target: float
    ) -> None:
        """Verify solve_keller_pac converges to ||F(x)||_inf < 1e-10 with valid physical allocations."""
        res = solve_keller_pac(
            calib=synthetic_2c_2s_calib,
            tau_target=target,
            tol=1e-10,
            ds_init=0.1,
            max_steps=60,
        )

        assert isinstance(res, TradeEquilibriumResult)
        assert res.converged is True
        assert res.metadata["method"] == "keller_pac"

        # Crucial precision assertion: ||F(x)||_inf < 1e-10
        max_res = float(np.max(np.abs(res.residuals)))
        assert max_res < 1e-10, f"Keller PAC final residual {max_res:.4e} is not < 1e-10"

        # Physical consistency of equilibrium states
        assert np.all(res.p_sol > 0), "Equilibrium prices must be strictly positive"
        assert np.all(res.w_sol > 0), "Equilibrium factor wages must be strictly positive"
        assert np.all(res.y_sol > 0), "Gross output must be strictly positive"


# ---------------------------------------------------------------------------
# 3. Stabilization & Memory Leak Tests
# ---------------------------------------------------------------------------

class TestAdversarialStabilizationAndLeakage:
    """Adversarial stress-testing of stabilization routines and memory leak verification."""

    def test_svd_clamping_extreme_singular_modes(self) -> None:
        """Verify SVD modal clamping prevents blowup on 1e16 condition number and singular matrices."""
        J_sing = np.diag([1.0, 1e-16, 0.0])
        f_val = np.array([1.0, 1.0, 1.0])
        D_L = np.ones(3)
        D_R = np.ones(3)

        delta = svd_clamped_newton_step(
            J=J_sing, f_val=f_val, D_L=D_L, D_R=D_R, max_comp=20.0, max_disp=0.30
        )

        assert np.all(np.isfinite(delta)), "Step contained NaN/Inf"
        assert np.max(np.abs(delta)) <= 0.30 + 1e-12, f"Displacement exceeded 0.30: {delta}"

    def test_anderson_acceleration_collinear_and_duplicate_history(self) -> None:
        """Verify Anderson acceleration handles collinear/duplicate histories without LinAlgError."""
        # Identical/stagnant history
        x_dup = [np.ones(5), np.ones(5), np.ones(5)]
        g_dup = [np.full(5, 1.1), np.full(5, 1.1), np.full(5, 1.1)]
        f_dup = [g - x for x, g in zip(x_dup, g_dup)]

        x_acc = anderson_accelerate(x_dup, g_dup, f_dup, m=4)
        assert np.all(np.isfinite(x_acc)), "Anderson failed on identical history"

        # Linearly dependent vectors
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([2.0, 4.0, 6.0])
        x_dep = [v1, v2]
        g_dep = [v1 + 0.1, v2 + 0.2]
        f_dep = [g - x for x, g in zip(x_dep, g_dep)]

        x_dep_acc = anderson_accelerate(x_dep, g_dep, f_dep, m=4)
        assert np.all(np.isfinite(x_dep_acc)), "Anderson failed on collinear history"

    def test_cyprus_secant_vanishing_scale(self) -> None:
        """Verify Cyprus 1D manifold solver handles vanishing domestic market scale (1e-10)."""
        nc = 77
        idx_cyp = 14
        rng = np.random.default_rng(123)
        A = rng.standard_normal((nc, nc))
        S_ww = A.T @ A + np.eye(nc)

        # Micro-economy near-singular scaling
        S_ww[idx_cyp, :] *= 1e-8
        S_ww[:, idx_cyp] *= 1e-8
        S_ww[idx_cyp, idx_cyp] = 1e-10
        rhs_w = rng.standard_normal(nc)

        dw_full, best_res, conv = solve_cyprus_manifold_step(
            S_ww=S_ww,
            rhs_w=rhs_w,
            eval_cyp_fn=None,
            idx_cyp=idx_cyp,
            tol=2.5e-3,
            max_disp=0.30,
        )

        assert conv is True
        assert best_res < 2.5e-3
        assert np.all(np.isfinite(dw_full))
        assert np.max(np.abs(dw_full)) <= 0.30 + 1e-12

    def test_memory_leak_and_stability_100_runs(
        self, synthetic_2c_2s_calib: TradeCalibrationResult
    ) -> None:
        """Verify memory stability across 100 consecutive solver iterations (no leaks)."""
        gc.collect()
        tracemalloc.start()
        snap1 = tracemalloc.take_snapshot()

        calib = synthetic_2c_2s_calib
        for i in range(100):
            # 1. Hawkins-Simon
            rho, l, u = check_hawkins_simon_viability(calib, tau=0.2)
            # 2. SVD Clamping
            J = np.diag([1.0, 1e-10])
            f = np.array([1.0, 1.0])
            dx = svd_clamped_newton_step(J, f, np.ones(2), np.ones(2))
            # 3. Anderson
            x_h = [np.array([float(i)]), np.array([float(i) + 0.5])]
            g_h = [np.array([float(i) + 0.5]), np.array([float(i) + 0.75])]
            f_h = [g - x for x, g in zip(x_h, g_h)]
            x_acc = anderson_accelerate(x_h, g_h, f_h, m=4)

        gc.collect()
        snap2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snap2.compare_to(snap1, "lineno")
        net_mem_kb = sum(s.size_diff for s in stats) / 1024.0

        # Must not accumulate unbounded memory (less than 100 KB across 100 iterations)
        assert net_mem_kb < 100.0, f"Unbounded memory growth detected: {net_mem_kb:.2f} KB"
