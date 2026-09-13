"""Adversarial Empirical Verification Suite for VFI Frontier Engines (Challenger 1).

Covers:
1. Continuous Transition (puremacro.vfi.continuous_transition):
   - Severe MIT shocks: 500 bps jump, 1000 bps jump, and severe negative TFP shock (-20%).
   - Strict mass conservation sum mu_t = 1.0 +- 1e-12 across all dates.
   - Broyden convergence and r_max bound analysis.
2. Exact IFT Gradients (puremacro.vfi.analytic_gradients):
   - Near boundary parameters (beta -> 0.999, 0.9995).
   - Tikhonov and Truncated SVD regularized fallbacks on ill-conditioned (cond > 1e13) and singular Jacobians.
   - Solver mode selection and condition number bounds.
3. Deep Macro PINNs (puremacro.vfi.deep_macro):
   - 10-state dynamic growth model stress testing.
   - 1000-period simulation from extreme initial state perturbations (0.01x to 5.0x steady state).
   - Strict verification of consumption feasibility (c > 0, k' > 0, c + k' = W).
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.vfi.analytic_gradients import (
    compute_ift_gradients,
    _solve_ift_linear_system,
)
from puremacro.vfi.collocation import CollocationProblem
from puremacro.vfi.continuous_distribution import solve_aiyagari_continuous
from puremacro.vfi.continuous_transition import (
    ContinuousTransitionResult,
    _evaluate_transition_system,
    _solve_transition_broyden,
    solve_continuous_transition,
)
from puremacro.vfi.deep_macro import DeepMacroModel, solve_deep_macro


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def aiyagari_baseline_ss():
    """Solved baseline continuous Aiyagari steady state."""
    return solve_aiyagari_continuous(
        beta=0.96,
        gamma=2.0,
        alpha=0.36,
        delta=0.08,
        a_max=30.0,
        n_a=40,
        n_z=3,
        N_k=60,
        tol=1e-4,
        backend="numpy",
    )


@pytest.fixture(scope="module")
def trained_10_state_pinn():
    """Trained 10-state dynamic neoclassical growth PINN solution."""
    model = DeepMacroModel.multi_country_growth(n_countries=10)
    sol = solve_deep_macro(
        model,
        hidden_dims=(32, 32),
        n_epochs=40,
        batch_size=64,
        lr=2e-3,
        backend="numpy",
        seed=123,
    )
    return model, sol


# ===========================================================================
# 1. Continuous Transition Dynamics & MIT Shocks
# ===========================================================================

class TestContinuousTransitionMITStress:
    """Adversarial stress testing of continuous transition dynamics under severe MIT shocks."""

    def test_rate_jump_500bps_broyden_convergence_and_mass_conservation(
        self, aiyagari_baseline_ss
    ):
        """Verify 500 bps transitory rate hike converges smoothly under Broyden with exact mass conservation."""
        ss0 = aiyagari_baseline_ss
        T = 40
        # 500 bps jump decaying at rate 0.75
        shock_rate = 0.05 * (0.75 ** np.arange(T))

        res = solve_continuous_transition(
            initial_steady_state=ss0,
            shock_path=shock_rate,
            shock_var="rate",
            horizon=T,
            solver="broyden",
            tol=1e-4,
            max_iter=50,
        )

        assert res.converged, f"Broyden failed to converge on 500 bps shock: max_res={res.max_residual}"
        assert res.max_residual < 1e-4
        assert res.iterations <= 20

        # Verify mass conservation everywhere across all T+1 distributions
        assert res.mass_conservation_error < 1e-12
        for t, dist in enumerate(res.distributions):
            mass_t = float(np.sum(dist))
            assert abs(mass_t - 1.0) < 1e-12, f"Mass conservation violated at t={t}: mass={mass_t}"

    def test_rate_jump_1000bps_extreme_stress(self, aiyagari_baseline_ss):
        """Verify extreme 1000 bps rate hike strictly conserves distribution mass sum mu_t = 1.0 +- 1e-12."""
        ss0 = aiyagari_baseline_ss
        T = 40
        shock_rate = 0.10 * (0.70 ** np.arange(T))

        res = solve_continuous_transition(
            initial_steady_state=ss0,
            shock_path=shock_rate,
            shock_var="rate",
            horizon=T,
            solver="broyden",
            tol=1e-3,
            max_iter=50,
        )

        assert res.mass_conservation_error < 1e-12
        for t, dist in enumerate(res.distributions):
            assert abs(float(np.sum(dist)) - 1.0) < 1e-12

    def test_negative_tfp_shock_broyden_with_transitional_rate_bound(
        self, aiyagari_baseline_ss
    ):
        """Verify that when transitional r_max is set appropriately, Broyden converges smoothly on -20% TFP shock."""
        ss0 = aiyagari_baseline_ss
        hh_init = ss0.household_solution
        P_z = np.asarray(hh_init.P_z, dtype=np.float64)
        z_grid = np.asarray(hh_init.z_grid, dtype=np.float64)
        K_hist = np.asarray(hh_init.a_grid, dtype=np.float64)
        n_a = len(hh_init.policy_c)
        a_max = float(K_hist[-1])
        a_grid_dense = a_max * (np.linspace(0.0, 1.0, n_a) ** 1.5)
        c_term = np.asarray(hh_init.policy_c, dtype=np.float64)

        T = 40
        Z_path = 1.0 - 0.20 * (0.8 ** np.arange(T))
        alpha = 0.36
        delta = 0.08
        L_agg = float(ss0.L)
        r_ss_init = float(ss0.r)
        K_ss_init = float(ss0.K)

        r0_implied = alpha * Z_path[0] * ((K_ss_init / L_agg) ** (alpha - 1.0)) - delta
        r_init = r_ss_init + (r0_implied - r_ss_init) * (0.85 ** np.arange(T))

        def eval_fn(r_arr):
            return _evaluate_transition_system(
                r_path=r_arr,
                Z_path=Z_path,
                mu_0=ss0.distribution.pdf,
                c_term=c_term,
                a_grid_dense=a_grid_dense,
                K_hist=K_hist,
                P_z=P_z,
                z_grid=z_grid,
                alpha=alpha,
                delta=delta,
                L_agg=L_agg,
                beta=0.96,
                gamma=2.0,
                r_term=r_ss_init,
            )

        # With r_max allowed to reach transitional levels (0.15), Broyden converges in <= 10 iterations
        r_sol, last_res, iters, converged = _solve_transition_broyden(
            eval_fn=eval_fn,
            r_init=r_init,
            alpha=alpha,
            delta=delta,
            tol=1e-4,
            max_iter=50,
            r_min=1e-4,
            r_max=0.15,
        )

        assert converged, f"Broyden failed with r_max=0.15: max_res={np.max(np.abs(last_res[0]))}"
        assert np.max(np.abs(last_res[0])) < 1e-4
        assert iters <= 10

        # Check mass conservation
        dists = last_res[5]
        for t, d in enumerate(dists):
            assert abs(float(np.sum(d)) - 1.0) < 1e-12


# ===========================================================================
# 2. Exact IFT Gradients & Robust Regularized Fallbacks
# ===========================================================================

class TestAnalyticGradientsBoundaryStress:
    """Adversarial stress testing of IFT gradients near parameter boundaries."""

    @pytest.mark.parametrize("beta_val", [0.99, 0.999, 0.9995])
    def test_near_boundary_discount_factor(self, beta_val):
        """Verify IFT computes finite, bounded parameter gradients near beta -> 1."""
        alpha = 0.36
        delta = 0.08
        sigma = 1.0
        k_ss = float((alpha * beta_val / (1.0 - beta_val * (1.0 - delta))) ** (1.0 / (1.0 - alpha)))

        prob = CollocationProblem(
            domain=(0.7 * k_ss, 1.3 * k_ss),
            orders=6,
            method="euler",
            params={"alpha": alpha, "delta": delta, "sigma": sigma},
            beta=beta_val,
            options={"tol": 1e-12, "max_iter": 500},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged

        # Evaluate IFT gradients
        res = compute_ift_gradients(sol, prob, solver="auto")
        assert np.all(np.isfinite(res.grad_coefficients))
        assert res.condition_number < 1e6
        assert np.all(np.isfinite(res.policy_gradient(k_ss)))

    def test_tikhonov_fallback_on_ill_conditioned_jacobian(self):
        """Verify Tikhonov regularization prevents solver divergence on ill-conditioned J_c."""
        N = 12
        p = 4
        rng = np.random.default_rng(101)

        U, _ = np.linalg.qr(rng.standard_normal((N, N)))
        V, _ = np.linalg.qr(rng.standard_normal((N, N)))
        # Artificially ill-conditioned singular values spanning 1e-14
        S_ill = np.logspace(0, -14, N)
        J_c_ill = U @ np.diag(S_ill) @ V.T
        J_theta = rng.standard_normal((N, p))

        grad_tikh, cond_num, method = _solve_ift_linear_system(
            J_c_ill, J_theta, solver="tikhonov", regularization=1e-6
        )

        assert "Tikhonov" in method
        assert np.all(np.isfinite(grad_tikh))
        assert np.max(np.abs(grad_tikh)) < 1e6

    def test_truncated_svd_fallback_on_rank_deficient_jacobian(self):
        """Verify Truncated SVD produces finite gradients on strictly singular J_c."""
        N = 10
        p = 3
        rng = np.random.default_rng(202)

        U, _ = np.linalg.qr(rng.standard_normal((N, N)))
        V, _ = np.linalg.qr(rng.standard_normal((N, N)))
        S_sing = np.ones(N)
        S_sing[-3:] = 0.0  # Rank N-3
        J_c_sing = U @ np.diag(S_sing) @ V.T
        J_theta = rng.standard_normal((N, p))

        # Explicit SVD with small regularization to bypass Tikhonov
        grad_svd, cond_num, method = _solve_ift_linear_system(
            J_c_sing, J_theta, solver="svd", regularization=0.0, rcond=1e-8
        )

        assert "SVD" in method
        assert np.all(np.isfinite(grad_svd))
        assert np.max(np.abs(grad_svd)) < 1e6


# ===========================================================================
# 3. Deep Macro PINNs Feasibility Stress Testing
# ===========================================================================

class TestDeepMacroFeasibilityStress:
    """Adversarial stress testing of 10-state PINN for physical viability and feasibility."""

    def test_10_state_growth_model_random_perturbations_feasibility(
        self, trained_10_state_pinn
    ):
        """Verify consumption feasibility (c > 0, k' > 0, c + k' = W) across 1000 periods from extreme perturbations."""
        model, sol = trained_10_state_pinn
        k_ss, _ = model.steady_state()
        assert len(k_ss) == 10

        rng = np.random.default_rng(42)

        # 5 extreme perturbations
        perturbations = [
            ("Near zero collapse (0.05x ss)", k_ss * 0.05),
            ("Massive capital glut (4.0x ss)", k_ss * 4.0),
            ("Highly asymmetric (0.1x to 3.0x ss)", k_ss * rng.uniform(0.1, 3.0, size=10)),
            ("Extreme uniform random (0.01x to 5.0x ss)", k_ss * rng.uniform(0.01, 5.0, size=10)),
            ("Single collapsed country (0.01x vs 1.0x)", np.array([k_ss[0] * 0.01] + list(k_ss[1:]))),
        ]

        for desc, s0 in perturbations:
            sim = sol.simulate(s0=s0, periods=1000, seed=42)
            states = sim["states"]        # (1001, 10)
            controls = sim["controls"]    # (1000, 10)
            coh = sim["cash_on_hand"]     # (1000, 10)

            # Invariant 1: Strictly positive consumption (c > 0)
            min_c = float(np.min(controls))
            assert min_c > 0.0, f"{desc}: Negative or zero consumption detected: min_c={min_c}"

            # Invariant 2: Strictly positive capital (k' > 0)
            min_k = float(np.min(states))
            assert min_k > 0.0, f"{desc}: Negative or zero capital stock detected: min_k={min_k}"

            # Invariant 3: Exact resource constraint feasibility c + k' = W
            k_next = states[1:]
            budget_resids = np.abs((controls + k_next) - coh)
            max_budget_err = float(np.max(budget_resids))
            assert max_budget_err < 1e-12, (
                f"{desc}: Resource constraint violated: max |c + k' - W| = {max_budget_err}"
            )

            # Invariant 4: No NaN, Inf, or non-finite values
            assert np.all(np.isfinite(states)), f"{desc}: Non-finite values in state trajectory"
            assert np.all(np.isfinite(controls)), f"{desc}: Non-finite values in controls trajectory"
            assert sim["physically_viable"] is True
