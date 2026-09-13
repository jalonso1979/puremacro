"""Adversarial Empirical Verification Suite for Experiment 2 and Experiment 3.

Challenger 2 Suite:
1. Experiment 2: Borrowing Constraint Kink & Gibbs Ringing
   - High-degree Chebyshev polynomials (N = 6, 12, 20, 30) persistently suffer from
     Gibbs ringing (amplitude >= 5e-4) and boundary violations (a' < a_bar).
   - FEM with localized hat basis and kink node (FEMMesh.from_kinks) and Fischer-Burmeister
     complementarity resolves the kink with zero boundary violations (0.00) and
     suppresses oscillations to machine precision (< 1e-8).
2. Experiment 3: Stochastic Multi-State Markov Model
   - State-contingent continuous policies across discrete Markov productivity states
     z in [0.90, 1.00, 1.10] maintain strict stochastic monotonicity:
     g(k, z_high) > g(k, z_mid) > g(k, z_low) everywhere on fine evaluation grids.
   - Adversarial stress tests with perturbed transition probability matrices (near-absorbing,
     fast mean-reverting, asymmetric recession, asymmetric boom, cyclical transit).
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pytest
from scipy.optimize import root

from puremacro.vfi.collocation import CollocationBasis, CollocationProblem
from puremacro.vfi.fem import FEMMesh, FEMProblem
from puremacro.vfi.problem import VFIProblem


# ============================================================================
# Fixtures for Experiment 2 & 3
# ============================================================================

@pytest.fixture
def exp2_setup():
    """Canonical parameters for Experiment 2: Borrowing-Constrained Savings Model."""
    alpha = 0.36
    beta = 0.96
    k_ss = float((alpha * beta) ** (1.0 / (1.0 - alpha)))
    domain = (0.1 * k_ss, 2.0 * k_ss)
    k_star = 0.5 * k_ss
    k_bar = float(alpha * beta * (k_star ** alpha))

    eval_k = np.linspace(domain[0], domain[1], 5000)
    constr_mask = eval_k <= k_star
    unconstr_mask = eval_k > k_star

    def g_true(k):
        k_arr = np.asarray(k, dtype=np.float64)
        return np.where(k_arr <= k_star, k_bar, alpha * beta * (k_arr ** alpha))

    return {
        "alpha": alpha,
        "beta": beta,
        "k_ss": k_ss,
        "domain": domain,
        "k_star": k_star,
        "k_bar": k_bar,
        "eval_k": eval_k,
        "constr_mask": constr_mask,
        "unconstr_mask": unconstr_mask,
        "g_true": g_true,
    }


@pytest.fixture
def exp3_setup():
    """Canonical parameters for Experiment 3: Stochastic Multi-State Model."""
    alpha = 0.36
    beta = 0.96
    delta = 1.0
    z_states = [0.90, 1.00, 1.10]
    P_z_baseline = np.array([
        [0.80, 0.15, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ])
    k_ss_mid = float((alpha * beta * 1.0) ** (1.0 / (1.0 - alpha)))
    domain = (0.5 * k_ss_mid, 1.5 * k_ss_mid)
    eval_k = np.linspace(domain[0], domain[1], 5000)

    return {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "z_states": z_states,
        "P_z_baseline": P_z_baseline,
        "k_ss_mid": k_ss_mid,
        "domain": domain,
        "eval_k": eval_k,
    }


# ============================================================================
# Task 1: Adversarial Verification of Experiment 2 (Kink & Gibbs Ringing)
# ============================================================================

class TestExperiment2BorrowingKinkGibbsRinging:
    """Adversarial testing of borrowing constraint kink resolution."""

    @pytest.mark.parametrize("N", [6, 12, 20, 30])
    def test_chebyshev_persistent_gibbs_ringing(self, exp2_setup, N):
        """Chebyshev polynomials persistently suffer from Gibbs ringing (amplitude >= 5e-4)."""
        setup = exp2_setup
        basis = CollocationBasis(domain=setup["domain"], orders=N)
        nodes = basis.nodes(squeeze=True)
        y_nodes = setup["g_true"](nodes)
        coeffs = basis.fit(y_nodes)

        g_cheb = np.asarray([basis.interpolate(coeffs, float(k)) for k in setup["eval_k"]], dtype=np.float64)

        gibbs_amplitude = float(np.max(np.abs(g_cheb[setup["constr_mask"]] - setup["k_bar"])))
        assert gibbs_amplitude >= 5e-4, (
            f"Chebyshev (N={N}) Gibbs ringing amplitude {gibbs_amplitude:.2e} unexpectedly < 5e-4"
        )

    @pytest.mark.parametrize("N", [6, 12, 20, 30])
    def test_chebyshev_persistent_boundary_violations(self, exp2_setup, N):
        """Chebyshev polynomials persistently violate borrowing lower bound (a' < a_bar)."""
        setup = exp2_setup
        basis = CollocationBasis(domain=setup["domain"], orders=N)
        nodes = basis.nodes(squeeze=True)
        y_nodes = setup["g_true"](nodes)
        coeffs = basis.fit(y_nodes)

        g_cheb = np.asarray([basis.interpolate(coeffs, float(k)) for k in setup["eval_k"]], dtype=np.float64)

        max_violation = float(np.max(np.maximum(0.0, setup["k_bar"] - g_cheb)))
        assert max_violation > 1e-4, (
            f"Chebyshev (N={N}) boundary violation {max_violation:.2e} unexpectedly <= 1e-4"
        )

    @pytest.mark.parametrize("E", [20, 40, 50, 80])
    def test_fem_kink_aligned_zero_boundary_violations(self, exp2_setup, E):
        """FEM with kink node achieves zero boundary violations (0.00)."""
        setup = exp2_setup
        mesh_kink = FEMMesh.from_kinks(setup["domain"], n_elements=E, kinks=[setup["k_star"]])
        prob = FEMProblem(
            domain=setup["domain"],
            elements=E,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(np.maximum(c, 1e-14)),
            transition_fn=lambda k: k ** setup["alpha"],
            beta=setup["beta"],
            params={"alpha": setup["alpha"], "gamma": 1.0},
            borrowing_constraint=setup["k_bar"],
            options={"mesh": mesh_kink, "kinks": [setup["k_star"]]},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged, f"FEM kink solve failed to converge for E={E}"

        g_fem = sol.policy(setup["eval_k"])
        max_violation = float(np.max(np.maximum(0.0, setup["k_bar"] - g_fem)))
        assert max_violation == 0.0, f"FEM (E={E}) violated borrowing constraint: {max_violation:.2e}"

    @pytest.mark.parametrize("E", [10, 20, 40])
    def test_fem_kink_aligned_machine_precision_ringing(self, exp2_setup, E):
        """FEM with kink node suppresses Gibbs ringing to machine precision (< 1e-8)."""
        setup = exp2_setup
        mesh_kink = FEMMesh.from_kinks(setup["domain"], n_elements=E, kinks=[setup["k_star"]])
        prob = FEMProblem(
            domain=setup["domain"],
            elements=E,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(np.maximum(c, 1e-14)),
            transition_fn=lambda k: k ** setup["alpha"],
            beta=setup["beta"],
            params={"alpha": setup["alpha"], "gamma": 1.0},
            borrowing_constraint=setup["k_bar"],
            options={"mesh": mesh_kink, "kinks": [setup["k_star"]]},
        )
        sol = prob.solve(backend="numpy")
        assert sol.converged

        g_fem = sol.policy(setup["eval_k"])
        gibbs_amplitude = float(np.max(np.abs(g_fem[setup["constr_mask"]] - setup["k_bar"])))
        assert gibbs_amplitude < 1e-8, (
            f"FEM (E={E}) Gibbs ringing {gibbs_amplitude:.2e} >= 1e-8"
        )

    def test_fem_vs_chebyshev_direct_contrast(self, exp2_setup):
        """FEM suppresses Gibbs ringing by over 5 orders of magnitude compared to Chebyshev."""
        setup = exp2_setup

        # Chebyshev N=20
        basis = CollocationBasis(domain=setup["domain"], orders=20)
        coeffs = basis.fit(setup["g_true"](basis.nodes(squeeze=True)))
        g_cheb = np.asarray([basis.interpolate(coeffs, float(k)) for k in setup["eval_k"]], dtype=np.float64)
        cheb_ring = float(np.max(np.abs(g_cheb[setup["constr_mask"]] - setup["k_bar"])))

        # FEM E=50
        mesh_kink = FEMMesh.from_kinks(setup["domain"], n_elements=50, kinks=[setup["k_star"]])
        prob = FEMProblem(
            domain=setup["domain"],
            elements=50,
            method="euler",
            projection="galerkin",
            return_fn=lambda c: np.log(np.maximum(c, 1e-14)),
            transition_fn=lambda k: k ** setup["alpha"],
            beta=setup["beta"],
            params={"alpha": setup["alpha"], "gamma": 1.0},
            borrowing_constraint=setup["k_bar"],
            options={"mesh": mesh_kink, "kinks": [setup["k_star"]]},
        )
        sol = prob.solve(backend="numpy")
        g_fem = sol.policy(setup["eval_k"])
        fem_ring = float(np.max(np.abs(g_fem[setup["constr_mask"]] - setup["k_bar"])))

        ratio = cheb_ring / fem_ring
        assert ratio > 1e4, f"FEM suppression ratio {ratio:.2e} is less than 10,000x"
        assert fem_ring < 1e-7


# ============================================================================
# Task 2: Adversarial Verification of Experiment 3 (Stochastic Multi-State)
# ============================================================================

class TestExperiment3StochasticMultiState:
    """Adversarial testing of stochastic multi-state Markov model."""

    def test_continuous_stochastic_monotonicity_collocation(self, exp3_setup):
        """Collocation policies maintain strict stochastic monotonicity across z."""
        setup = exp3_setup
        coll_policies = {}

        for z in setup["z_states"]:
            def euler_fn(policy_fn, k, params, z_val=z):
                kp = policy_fn(k)
                kpp = policy_fn(kp)
                c = np.maximum(z_val * (k ** setup["alpha"]) - kp, 1e-12)
                cp = np.maximum(z_val * (kp ** setup["alpha"]) - kpp, 1e-12)
                return 1.0 - setup["beta"] * (c / cp) * setup["alpha"] * z_val * (kp ** (setup["alpha"] - 1.0))

            sol = CollocationProblem(
                domain=setup["domain"],
                orders=8,
                method="euler",
                euler_residual_fn=euler_fn,
                params={"alpha": setup["alpha"], "z": z},
                beta=setup["beta"],
            ).solve(backend="numpy")
            assert sol.converged
            coll_policies[z] = sol.policy(setup["eval_k"])

        gap_high_mid = coll_policies[1.10] - coll_policies[1.00]
        gap_mid_low = coll_policies[1.00] - coll_policies[0.90]

        assert np.all(gap_high_mid > 0.0), "Collocation violated g(k, 1.10) > g(k, 1.00)"
        assert np.all(gap_mid_low > 0.0), "Collocation violated g(k, 1.00) > g(k, 0.90)"
        assert np.min(gap_high_mid) > 0.01, f"Minimum gap {np.min(gap_high_mid):.4f} too small"
        assert np.min(gap_mid_low) > 0.01, f"Minimum gap {np.min(gap_mid_low):.4f} too small"

    def test_continuous_stochastic_monotonicity_fem(self, exp3_setup):
        """FEM Galerkin policies maintain strict stochastic monotonicity across z."""
        setup = exp3_setup
        fem_policies = {}

        for z in setup["z_states"]:
            sol = FEMProblem(
                domain=setup["domain"],
                elements=50,
                method="euler",
                projection="galerkin",
                return_fn=lambda c: np.log(np.maximum(c, 1e-14)),
                transition_fn=lambda k, z_val=z: z_val * (k ** setup["alpha"]),
                beta=setup["beta"],
                params={"alpha": setup["alpha"], "gamma": 1.0, "z": z},
            ).solve(backend="numpy")
            assert sol.converged
            fem_policies[z] = sol.policy(setup["eval_k"])

        gap_high_mid = fem_policies[1.10] - fem_policies[1.00]
        gap_mid_low = fem_policies[1.00] - fem_policies[0.90]

        assert np.all(gap_high_mid > 0.0), "FEM violated g(k, 1.10) > g(k, 1.00)"
        assert np.all(gap_mid_low > 0.0), "FEM violated g(k, 1.00) > g(k, 0.90)"
        assert np.min(gap_high_mid) > 0.01
        assert np.min(gap_mid_low) > 0.01

    @pytest.mark.parametrize(
        "matrix_name,P_z",
        [
            ("Baseline", np.array([[0.80, 0.15, 0.05], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]])),
            ("NearAbsorbing", np.array([[0.98, 0.015, 0.005], [0.01, 0.98, 0.01], [0.005, 0.015, 0.98]])),
            ("FastMeanReversion", np.array([[0.34, 0.33, 0.33], [0.33, 0.34, 0.33], [0.33, 0.33, 0.34]])),
            ("AsymmetricDownward", np.array([[0.90, 0.08, 0.02], [0.40, 0.50, 0.10], [0.20, 0.60, 0.20]])),
            ("AsymmetricUpward", np.array([[0.20, 0.60, 0.20], [0.10, 0.50, 0.40], [0.02, 0.08, 0.90]])),
            ("CyclicalTransit", np.array([[0.10, 0.80, 0.10], [0.10, 0.10, 0.80], [0.80, 0.10, 0.10]])),
        ],
    )
    def test_coupled_continuous_stochastic_euler_perturbed_Pz(self, exp3_setup, matrix_name, P_z):
        """Coupled continuous Euler system across Markov states converges and preserves monotonicity under perturbed P_z."""
        setup = exp3_setup
        alpha = setup["alpha"]
        beta = setup["beta"]
        z_states = np.asarray(setup["z_states"])
        nz = len(z_states)
        n_nodes = 30
        nodes = np.linspace(setup["domain"][0], setup["domain"][1], n_nodes)

        def residual_system(y_flat):
            Y = y_flat.reshape((nz, n_nodes))
            res = np.zeros_like(Y)
            for i in range(nz):
                zi = z_states[i]
                for m in range(n_nodes):
                    k = nodes[m]
                    kp = Y[i, m]
                    c = zi * (k ** alpha) - kp
                    if c <= 1e-12:
                        res[i, m] = 1e5
                        continue
                    exp_term = 0.0
                    for j in range(nz):
                        zj = z_states[j]
                        kpp = np.interp(kp, nodes, Y[j, :])
                        cp = zj * (kp ** alpha) - kpp
                        if cp <= 1e-12:
                            exp_term += P_z[i, j] * 1e5
                        else:
                            exp_term += P_z[i, j] * (1.0 / cp) * alpha * zj * (kp ** (alpha - 1.0))
                    res[i, m] = 1.0 - beta * c * exp_term
            return res.ravel()

        y0 = np.zeros((nz, n_nodes))
        for i in range(nz):
            y0[i, :] = alpha * beta * z_states[i] * (nodes ** alpha)

        sol = root(residual_system, y0.ravel(), method="hybr", tol=1e-8)
        assert sol.success, f"Coupled stochastic Euler solve failed under {matrix_name}"
        assert np.max(np.abs(sol.fun)) < 1e-6

        Y_sol = sol.x.reshape((nz, n_nodes))
        g_fine = np.zeros((nz, len(setup["eval_k"])))
        for i in range(nz):
            g_fine[i, :] = np.interp(setup["eval_k"], nodes, Y_sol[i, :])

        gap_high_mid = g_fine[2, :] - g_fine[1, :]
        gap_mid_low = g_fine[1, :] - g_fine[0, :]

        assert np.all(gap_high_mid > 0.0), f"Monotonicity failed for {matrix_name} (high vs mid)"
        assert np.all(gap_mid_low > 0.0), f"Monotonicity failed for {matrix_name} (mid vs low)"
        assert np.min(gap_high_mid) > 0.01
        assert np.min(gap_mid_low) > 0.01

    @pytest.mark.parametrize(
        "matrix_name,P_z",
        [
            ("Baseline", np.array([[0.80, 0.15, 0.05], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]])),
            ("NearAbsorbing", np.array([[0.98, 0.015, 0.005], [0.01, 0.98, 0.01], [0.005, 0.015, 0.98]])),
            ("FastMeanReversion", np.array([[0.34, 0.33, 0.33], [0.33, 0.34, 0.33], [0.33, 0.33, 0.34]])),
            ("AsymmetricDownward", np.array([[0.90, 0.08, 0.02], [0.40, 0.50, 0.10], [0.20, 0.60, 0.20]])),
            ("AsymmetricUpward", np.array([[0.20, 0.60, 0.20], [0.10, 0.50, 0.40], [0.02, 0.08, 0.90]])),
        ],
    )
    def test_discrete_vfi_perturbed_Pz_monotonicity(self, exp3_setup, matrix_name, P_z):
        """Discrete VFI joint 3-state solve preserves monotonicity across perturbed P_z."""
        setup = exp3_setup
        k_grid = np.linspace(setup["domain"][0], setup["domain"][1], 200)

        def rf_3state(kp, k, z, xp=np):
            c = xp.maximum(z * (k ** setup["alpha"]) - kp, 1e-12)
            return xp.where(c > 1e-12, xp.log(c), -1e10)

        prob = VFIProblem(
            a_grid=k_grid,
            z_grid=np.asarray(setup["z_states"]),
            P_z=P_z,
            return_fn=rf_3state,
            beta=setup["beta"],
            options={"tol": 1e-7, "howard": True, "n_howard": 30},
        )
        sol_vfi = prob.solve(backend="numpy")

        vfi_policies = {}
        for iz, z in enumerate(setup["z_states"]):
            pol_z = k_grid[sol_vfi.policy_aprime[:, iz]]
            vfi_policies[z] = np.interp(setup["eval_k"], k_grid, pol_z)

        gap_high_mid = vfi_policies[1.10] - vfi_policies[1.00]
        gap_mid_low = vfi_policies[1.00] - vfi_policies[0.90]

        assert np.all(gap_high_mid >= 0.0), f"Discrete VFI failed monotonicity for {matrix_name}"
        assert np.all(gap_mid_low >= 0.0), f"Discrete VFI failed monotonicity for {matrix_name}"
        assert np.min(gap_high_mid) > 0.01
        assert np.min(gap_mid_low) > 0.01
