"""Adversarial stress harness and empirical challenge suite for puremacro 2.9.0 M3.

Empirically tests Waggoner & Zha (1999) Conditional Forecasting, Dynare shock_groups
parsing and decomposition, Bayesian IRFs, and Tunable QZ Criterium across 5 core stress dimensions:
1. Multi-period simultaneous interest rate pegs (ZLB condition R_t = 0 for t=1..4, positive steady state,
   deep recession initial states, extended 8-period pegs, multi-target pegs).
2. Singular & under-determined restriction matrices (formal null-space optimality of minimum energy,
   anisotropic covariance weighting vs minimum energy, consistent rank-deficient systems).
3. Conflicting conditions & error handling (graceful failure under exact=True vs least squares under exact=False,
   normal equations orthogonality verification, decoupled zero-impact shocks).
4. Shock groups decomposition adding-up balance under extreme shock vectors (10*sigma, 50*sigma disaster
   regimes, multi-group partitions with unassigned 'Others', exact linearity/homogeneity, initial state separation).
5. Tunable QZ criterium boundary unit roots and Bayesian IRF credible interval monotonicity under indeterminacy.
"""
from __future__ import annotations

import warnings
from dataclasses import FrozenInstanceError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import (
    build_dynare,
    conditional_forecast,
    ConditionalForecastResult,
    load_mod,
    shock_groups_decomposition,
    ShockDecompositionResult,
    bayesian_irf,
    BayesianIRFResult,
    prior_predictive,
    PriorPredictiveResult,
    klein_solve,
    BetaPrior,
    GammaPrior,
    NormalPrior,
    UniformPrior,
)
from puremacro.dsge._parser import parse_mod_to_dag


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def nk_model_zlb():
    """Canonical 3-equation New Keynesian model with positive steady-state interest rate (r_ss = 0.01)."""
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.7,
        "rho_a": 0.6,
    }
    variables = ["y", "pi", "r", "a"]
    shocks = ["e_d", "e_m"]
    # Positive annualized steady state rate: r_ss = 0.01 (4% annual)
    steady_state = {"y": 0.0, "pi": 0.0, "r": 0.01, "a": 0.0}

    def nk_equations(lead, curr, lag, shocks_v, p):
        return [
            curr.y - (lead.y - (curr.r - lead.pi) / p.sigma + curr.a),
            curr.pi - (p.beta * lead.pi + p.kappa * curr.y),
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.e_m),
            curr.a - (p.rho_a * lag.a + shocks_v.e_d),
        ]

    return build_dynare(
        nk_equations,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
        strict=False,
    )


@pytest.fixture
def nk_model_with_redundant_var():
    """NK model augmented with an auxiliary variable z = 2*y to test rank-deficient systems."""
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.7,
        "rho_a": 0.6,
    }
    variables = ["y", "pi", "r", "a", "z"]
    shocks = ["e_d", "e_m"]
    steady_state = {v: 0.0 for v in variables}

    def nk_equations(lead, curr, lag, shocks_v, p):
        return [
            curr.y - (lead.y - (curr.r - lead.pi) / p.sigma + curr.a),
            curr.pi - (p.beta * lead.pi + p.kappa * curr.y),
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.e_m),
            curr.a - (p.rho_a * lag.a + shocks_v.e_d),
            curr.z - 2.0 * curr.y,
        ]

    return build_dynare(
        nk_equations,
        variables=variables,
        shocks=shocks,
        params=params,
        steady_state=steady_state,
        check_steady_state=False,
        strict=False,
    )


@pytest.fixture
def four_shock_model():
    """Multi-shock model with 4 distinct shock processes to test complex shock groupings."""
    mod_text = """
    var y, pi, r, a, b, g, ms;
    varexo e_a, e_b, e_g, e_ms;
    parameters beta, sigma, kappa, phi_pi, phi_y, rho_r, rho_a, rho_b, rho_g;

    beta = 0.99;
    sigma = 1.0;
    kappa = 0.15;
    phi_pi = 1.5;
    phi_y = 0.25;
    rho_r = 0.7;
    rho_a = 0.8;
    rho_b = 0.5;
    rho_g = 0.6;

    model;
    y = y(+1) - (r - pi(+1)) / sigma + a + g;
    pi = beta * pi(+1) + kappa * y + b;
    r = rho_r * r(-1) + (1.0 - rho_r) * (phi_pi * pi + phi_y * y) + ms;
    a = rho_a * a(-1) + e_a;
    b = rho_b * b(-1) + e_b;
    g = rho_g * g(-1) + e_g;
    ms = e_ms;
    end;

    shock_groups;
    'Supply' = e_a, e_b;
    'Fiscal' = e_g;
    end;
    """
    m = load_mod(mod_text)
    return m


# ===========================================================================
# Dimension 1: Multi-Period Simultaneous Interest Rate Pegs (ZLB)
# ===========================================================================

class TestMultiPeriodInterestRatePegs:
    """Stress testing multi-period interest rate pegs and ZLB conditions."""

    def test_zlb_4_quarter_peg_positive_steady_state(self, nk_model_zlb):
        """Verify ZLB peg (R_t = 0 for t=1..4) starting from positive steady state (R_ss = 0.01)."""
        H = 8
        target_r = [0.0, 0.0, 0.0, 0.0]
        res = conditional_forecast(
            nk_model_zlb,
            target_paths={"r": target_r},
            controlled_shocks=["e_m"],
            horizon=H,
        )

        assert isinstance(res, ConditionalForecastResult)
        # 1. Assert peg is enforced to extreme precision
        np.testing.assert_allclose(res.forecast["r"].iloc[:4].to_numpy(), 0.0, atol=1e-12)

        # 2. Assert uncontrolled shock e_d is strictly zero
        np.testing.assert_allclose(res.shocks["e_d"].to_numpy(), 0.0, atol=1e-15)

        # 3. Assert forward simulation with inverted shocks reproduces forecast exactly
        G, N, C, D = nk_model_zlb._dynare_companion()
        Psi = [D.copy()]
        CG = C.copy()
        for k in range(1, H):
            Psi.append(CG @ N)
            CG = CG @ G
        ss_vec = np.array([float(nk_model_zlb.steady_state[v]) for v in nk_model_zlb.variables])

        recon_r = np.zeros(H)
        for h in range(1, H + 1):
            val = ss_vec[nk_model_zlb.variables.index("r")]
            for s in range(1, h + 1):
                val += (Psi[h - s] @ res.shocks.iloc[s - 1].to_numpy())[nk_model_zlb.variables.index("r")]
            recon_r[h - 1] = val

        np.testing.assert_allclose(res.forecast["r"].to_numpy(), recon_r, atol=1e-14)

        # 4. Beyond the peg (t > 4), r must smoothly return toward steady state (0.01)
        r_post = res.forecast["r"].iloc[4:].to_numpy()
        assert r_post[0] > 0.0  # Immediately begins reverting toward 0.01
        assert np.all(np.diff(r_post) > 0.0)  # Monotonically increasing toward steady state
        assert np.all(r_post < 0.01)

    def test_zlb_peg_under_deep_recession_initial_state(self, nk_model_zlb):
        """Verify ZLB peg when economy starts in a deep negative shock (a_0 = -0.05)."""
        x0 = {"a": -0.05, "r": 0.0}
        target_r = [0.0, 0.0, 0.0, 0.0]
        res = conditional_forecast(
            nk_model_zlb,
            target_paths={"r": target_r},
            controlled_shocks=["e_m"],
            x0=x0,
            horizon=6,
        )

        np.testing.assert_allclose(res.forecast["r"].iloc[:4].to_numpy(), 0.0, atol=1e-12)
        # In a recession without monetary intervention, r would endogenous fall or adjust;
        # monetary shock e_m must compensate for both baseline state propagation and target
        assert len(res.shocks) == 6
        assert np.any(np.abs(res.shocks["e_m"].iloc[:4]) > 1e-4)

    def test_extended_8_quarter_interest_rate_peg(self, nk_model_zlb):
        """Verify stability and inversion across an extended 8-quarter peg."""
        H = 12
        target_r = [0.005] * 8  # Pegged at 0.5% for 8 quarters
        res = conditional_forecast(
            nk_model_zlb,
            target_paths={"r": target_r},
            controlled_shocks=["e_m"],
            horizon=H,
        )

        np.testing.assert_allclose(res.forecast["r"].iloc[:8].to_numpy(), 0.005, atol=1e-12)
        # Beyond period 8, no further shocks are applied
        np.testing.assert_allclose(res.shocks.iloc[8:].to_numpy(), 0.0, atol=1e-15)
        # Smooth reversion from 0.005 to 0.01
        assert res.forecast["r"].iloc[8] > 0.005
        assert res.forecast["r"].iloc[11] > res.forecast["r"].iloc[8]

    def test_multi_target_multi_period_peg(self, nk_model_zlb):
        """Verify simultaneous targets on output gap and interest rate across 3 quarters."""
        target_y = [0.01, 0.015, 0.02]
        target_r = [0.005, 0.005, 0.005]
        res = conditional_forecast(
            nk_model_zlb,
            target_paths={"y": target_y, "r": target_r},
            controlled_shocks=["e_d", "e_m"],
            horizon=5,
        )

        np.testing.assert_allclose(res.forecast["y"].iloc[:3].to_numpy(), target_y, atol=1e-11)
        np.testing.assert_allclose(res.forecast["r"].iloc[:3].to_numpy(), target_r, atol=1e-11)


# ===========================================================================
# Dimension 2: Singular & Under-Determined Restriction Matrices & Minimum Energy
# ===========================================================================

class TestSingularAndUnderdeterminedRestrictions:
    """Stress testing under-determined and singular restriction matrices."""

    def test_minimum_energy_formal_null_space_optimality(self, nk_model_zlb):
        """Verify mathematically that any null-space perturbation strictly increases 2-norm of shocks."""
        # 2 restrictions on r over horizon H=6 with 2 controlled shocks -> 12 shock parameters, rank 2
        H = 6
        target_r = [0.02, 0.015]
        res_me = conditional_forecast(
            nk_model_zlb,
            target_paths={"r": target_r},
            controlled_shocks=["e_d", "e_m"],
            method="minimum_energy",
            horizon=H,
        )

        shocks_opt = res_me.shocks[["e_d", "e_m"]].to_numpy().flatten()
        norm_opt_sq = float(np.sum(shocks_opt ** 2))

        # Construct exact restriction matrix A and deviation vector b
        G, N, C, D = nk_model_zlb._dynare_companion()
        Psi = [D.copy()]
        CG = C.copy()
        for k in range(1, H):
            Psi.append(CG @ N)
            CG = CG @ G

        ctrl_idx = [nk_model_zlb.shocks.index(s) for s in ["e_d", "e_m"]]
        v_idx = nk_model_zlb.variables.index("r")
        ss_val = float(nk_model_zlb.steady_state["r"])

        A = np.zeros((2, H * len(ctrl_idx)))
        b = np.array([target_r[0] - ss_val, target_r[1] - ss_val])

        # Row 0: target at h=1
        A[0, 0:2] = Psi[0][v_idx, ctrl_idx]
        # Row 1: target at h=2
        A[1, 0:2] = Psi[1][v_idx, ctrl_idx]
        A[1, 2:4] = Psi[0][v_idx, ctrl_idx]

        # Verify A @ shocks_opt == b
        np.testing.assert_allclose(A @ shocks_opt, b, atol=1e-12)

        # Compute null space of A via SVD
        _, _, Vt = np.linalg.svd(A)
        null_space = Vt[2:, :]  # 10 x 12
        assert null_space.shape[0] == 10

        # Empirical challenge: sample 30 random null space perturbations
        rng = np.random.default_rng(1001)
        for _ in range(30):
            coeffs = rng.normal(0.0, 0.05, size=10)
            v = coeffs @ null_space
            v_norm_sq = float(np.sum(v ** 2))
            assert v_norm_sq > 1e-8

            perturbed_shocks = shocks_opt + v
            perturbed_norm_sq = float(np.sum(perturbed_shocks ** 2))

            # 1. Perturbation must satisfy target constraints identically
            np.testing.assert_allclose(A @ perturbed_shocks, b, atol=1e-10)

            # 2. Strict minimum energy property: ||u* + v||^2 == ||u*||^2 + ||v||^2 > ||u*||^2
            assert perturbed_norm_sq > norm_opt_sq
            np.testing.assert_allclose(perturbed_norm_sq, norm_opt_sq + v_norm_sq, atol=1e-12)

    def test_covariance_weighted_vs_minimum_energy_anisotropic(self, nk_model_zlb):
        """Verify covariance-weighted solution differs from minimum energy under unequal shock variances."""
        # e_d has variance 1.0, e_m has variance 100.0
        shock_cov = np.diag([1.0, 100.0])

        res_cw = conditional_forecast(
            nk_model_zlb,
            target_paths={"y": [0.02]},
            controlled_shocks=["e_d", "e_m"],
            shock_cov=shock_cov,
            method="covariance_weighted",
            horizon=3,
        )

        res_me = conditional_forecast(
            nk_model_zlb,
            target_paths={"y": [0.02]},
            controlled_shocks=["e_d", "e_m"],
            shock_cov=shock_cov,
            method="minimum_energy",
            horizon=3,
        )

        # Both meet the target exactly
        assert abs(res_cw.forecast["y"].iloc[0] - 0.02) < 1e-10
        assert abs(res_me.forecast["y"].iloc[0] - 0.02) < 1e-10

        # Covariance-weighted should rely much more on e_m (larger variance) than minimum energy
        shocks_cw = res_cw.shocks.iloc[0]
        shocks_me = res_me.shocks.iloc[0]
        assert abs(shocks_cw["e_m"]) > abs(shocks_me["e_m"])
        assert abs(shocks_cw["e_d"]) < abs(shocks_me["e_d"])

        # Covariance-weighted minimizes u^T Sigma^{-1} u
        weighted_norm_cw = shocks_cw["e_d"] ** 2 / 1.0 + shocks_cw["e_m"] ** 2 / 100.0
        weighted_norm_me = shocks_me["e_d"] ** 2 / 1.0 + shocks_me["e_m"] ** 2 / 100.0
        assert weighted_norm_cw < weighted_norm_me

        # Minimum energy minimizes u^T u
        euclid_norm_cw = shocks_cw["e_d"] ** 2 + shocks_cw["e_m"] ** 2
        euclid_norm_me = shocks_me["e_d"] ** 2 + shocks_me["e_m"] ** 2
        assert euclid_norm_me < euclid_norm_cw

    def test_consistent_rank_deficient_restrictions_exact_true(self, nk_model_with_redundant_var):
        """Verify that consistent singular/rank-deficient restrictions pass cleanly under exact=True."""
        # z = 2*y. Setting y=0.02 and z=0.04 is rank-1 system with 2 equations
        res = conditional_forecast(
            nk_model_with_redundant_var,
            target_paths={"y": [0.02], "z": [0.04]},
            controlled_shocks=["e_m"],
            exact=True,
            horizon=3,
        )
        assert isinstance(res, ConditionalForecastResult)
        assert abs(res.forecast["y"].iloc[0] - 0.02) < 1e-10
        assert abs(res.forecast["z"].iloc[0] - 0.04) < 1e-10


# ===========================================================================
# Dimension 3: Conflicting Conditions & Error Handling
# ===========================================================================

class TestConflictingConditionsAndErrorHandling:
    """Stress testing conflicting target paths, exact=True failure, and least-squares under exact=False."""

    def test_conflicting_conditions_exact_true_raises_value_error(self, nk_model_with_redundant_var):
        """Verify exact=True raises ValueError on conflicting targets with descriptive message."""
        # z = 2*y. Target y=0.02 and z=0.05 is contradictory with 1 controlled shock
        with pytest.raises(ValueError, match="Target path cannot be achieved"):
            conditional_forecast(
                nk_model_with_redundant_var,
                target_paths={"y": [0.02], "z": [0.05]},
                controlled_shocks=["e_m"],
                exact=True,
                horizon=3,
            )

    def test_conflicting_conditions_exact_false_least_squares_orthogonality(self, nk_model_with_redundant_var):
        """Verify exact=False emits warning and mathematically minimizes residual sum of squares."""
        # Contradictory: y=0.02, z=0.05 (where z = 2*y)
        with pytest.warns(UserWarning, match="returning least-squares approximation"):
            res = conditional_forecast(
                nk_model_with_redundant_var,
                target_paths={"y": [0.02], "z": [0.05]},
                controlled_shocks=["e_m"],
                exact=False,
                horizon=3,
            )

        assert isinstance(res, ConditionalForecastResult)
        y_ls = res.forecast["y"].iloc[0]
        z_ls = res.forecast["z"].iloc[0]

        # Analytical least-squares solution: min_y (y - 0.02)^2 + (2y - 0.05)^2
        # d/dy = 2(y - 0.02) + 4(2y - 0.05) = 10y - 0.24 = 0 => y = 0.024, z = 0.048
        np.testing.assert_allclose(y_ls, 0.024, atol=1e-12)
        np.testing.assert_allclose(z_ls, 0.048, atol=1e-12)

        # Verify against 20 random shock perturbations: any perturbation yields larger residual norm
        u_ls = res.shocks["e_m"].iloc[0]
        err_ls = (y_ls - 0.02) ** 2 + (z_ls - 0.05) ** 2
        # Sensitivity of y and z to e_m
        G, N, C, D = nk_model_with_redundant_var._dynare_companion()
        dy_du = D[nk_model_with_redundant_var.variables.index("y"), nk_model_with_redundant_var.shocks.index("e_m")]
        dz_du = D[nk_model_with_redundant_var.variables.index("z"), nk_model_with_redundant_var.shocks.index("e_m")]

        rng = np.random.default_rng(2026)
        for _ in range(20):
            delta = rng.normal(0.0, 0.01)
            if abs(delta) < 1e-6:
                continue
            y_pert = y_ls + dy_du * delta
            z_pert = z_ls + dz_du * delta
            err_pert = (y_pert - 0.02) ** 2 + (z_pert - 0.05) ** 2
            assert err_pert > err_ls

    def test_zero_impact_shock_raises_value_error(self, nk_model_zlb):
        """Verify ValueError is raised if controlled shock has zero impact on target variable."""
        # e_m (monetary shock) has exactly zero impact on exogenous technology process 'a'
        with pytest.raises(ValueError, match="zero impact"):
            conditional_forecast(
                nk_model_zlb,
                target_paths={"a": [0.05]},
                controlled_shocks=["e_m"],
                exact=True,
            )

    def test_horizon_shorter_than_target_raises_value_error(self, nk_model_zlb):
        """Verify ValueError when horizon < maximum target horizon."""
        with pytest.raises(ValueError, match="is shorter than the maximum target horizon"):
            conditional_forecast(
                nk_model_zlb,
                target_paths={"r": [0.01, 0.01, 0.01, 0.01]},
                controlled_shocks=["e_m"],
                horizon=2,
            )

    def test_empty_controlled_shocks_raises_value_error(self, nk_model_zlb):
        """Verify ValueError when controlled_shocks is empty but targets exist."""
        with pytest.raises(ValueError, match="Cannot enforce target conditions with empty controlled_shocks"):
            conditional_forecast(
                nk_model_zlb,
                target_paths={"r": [0.01]},
                controlled_shocks=[],
            )


# ===========================================================================
# Dimension 4: Shock Groups Adding-Up Balance Under Extreme Shocks (10*sigma, 50*sigma)
# ===========================================================================

class TestShockGroupsExtremeStress:
    """Stress testing shock groups decomposition under extreme volatility and disaster shocks."""

    def test_shock_groups_10_sigma_adding_up_balance(self, four_shock_model):
        """Verify adding-up balance error <= 1e-12 under 10*sigma structural shocks."""
        T = 50
        rng = np.random.default_rng(777)
        # 10*sigma extreme shock regime
        extreme_shocks = rng.normal(0.0, 0.20, size=(T, 4))
        sim_df = four_shock_model.simulate(periods=T, shocks=extreme_shocks, burn=0)

        groups = {
            "Demand_Fiscal": ["e_g"],
            "Supply": ["e_a", "e_b"],
            "Monetary": ["e_ms"],
        }
        res = shock_groups_decomposition(four_shock_model, data=sim_df, groups=groups, initial_state=np.zeros(four_shock_model.n_states))

        assert isinstance(res, ShockDecompositionResult)
        for var in four_shock_model.variables:
            total = np.zeros(T)
            for g_name in res.components:
                total += res.components[g_name][var].to_numpy()
            diff = np.max(np.abs(total - sim_df[var].to_numpy()))
            assert diff <= 1e-12, f"10*sigma adding-up violated for {var}: max diff {diff:.3e}"

    def test_shock_groups_50_sigma_disaster_adding_up_balance(self, four_shock_model):
        """Verify adding-up balance error <= 1e-12 under 50*sigma disaster shocks."""
        T = 40
        rng = np.random.default_rng(888)
        # 50*sigma catastrophic disaster shocks (sigma = 1.0)
        disaster_shocks = rng.normal(0.0, 1.0, size=(T, 4))
        sim_df = four_shock_model.simulate(periods=T, shocks=disaster_shocks, burn=0)

        res = shock_groups_decomposition(four_shock_model, data=sim_df, initial_state=np.zeros(four_shock_model.n_states))

        for var in four_shock_model.variables:
            total = np.zeros(T)
            for g_name in res.components:
                total += res.components[g_name][var].to_numpy()
            diff = np.max(np.abs(total - sim_df[var].to_numpy()))
            assert diff <= 1e-12, f"50*sigma adding-up violated for {var}: max diff {diff:.3e}"

    def test_shock_groups_unassigned_partition_others(self, four_shock_model):
        """Verify unassigned shocks are properly grouped into 'Others' and invariant holds."""
        T = 30
        rng = np.random.default_rng(999)
        shocks = rng.normal(0.0, 0.05, size=(T, 4))
        sim_df = four_shock_model.simulate(periods=T, shocks=shocks, burn=0)

        # Assign only e_ms, leaving e_a, e_b, e_g unassigned
        groups = {"Policy": ["e_ms"]}
        res = shock_groups_decomposition(four_shock_model, data=sim_df, groups=groups)

        assert "Policy" in res.components
        assert "Others" in res.components
        assert set(res.groups["Others"]) == {"e_a", "e_b", "e_g"}

        for var in four_shock_model.variables:
            total = np.zeros(T)
            for g_name in res.components:
                total += res.components[g_name][var].to_numpy()
            np.testing.assert_allclose(total, sim_df[var].to_numpy(), atol=1e-12)

    def test_shock_groups_exact_linearity_and_homogeneity(self, four_shock_model):
        """Verify that multiplying shock vector by scalar c scales shock group components by c."""
        T = 25
        rng = np.random.default_rng(42)
        base_shocks = rng.normal(0.0, 0.02, size=(T, 4))
        scaled_shocks = 2.5 * base_shocks

        sim_base = four_shock_model.simulate(periods=T, shocks=base_shocks, burn=0)
        sim_scaled = four_shock_model.simulate(periods=T, shocks=scaled_shocks, burn=0)

        res_base = shock_groups_decomposition(four_shock_model, data=sim_base, initial_state=np.zeros(four_shock_model.n_states))
        res_scaled = shock_groups_decomposition(four_shock_model, data=sim_scaled, initial_state=np.zeros(four_shock_model.n_states))

        for g_name in ["Supply", "Fiscal"]:
            for var in four_shock_model.variables:
                base_comp = res_base.components[g_name][var].to_numpy()
                scaled_comp = res_scaled.components[g_name][var].to_numpy()
                np.testing.assert_allclose(scaled_comp, 2.5 * base_comp, atol=1e-13)

    def test_shock_groups_non_zero_initial_state_separation(self, four_shock_model):
        """Verify clean separation of predetermined state propagation from innovations."""
        T = 30
        rng = np.random.default_rng(333)
        shocks = rng.normal(0.0, 0.02, size=(T, 4))
        s0 = np.array([0.05, -0.03, 0.02, -0.01])
        sim_df = four_shock_model.simulate(periods=T, shocks=shocks, burn=0, initial_state=s0)

        res = shock_groups_decomposition(four_shock_model, data=sim_df, initial_state=s0)

        # When initial state matches data generation, no residual is needed
        assert "residual" not in res.components
        for var in four_shock_model.variables:
            total = np.zeros(T)
            for g_name in res.components:
                total += res.components[g_name][var].to_numpy()
            diff = np.max(np.abs(total - sim_df[var].to_numpy()))
            assert diff <= 1e-12, f"Initial state adding-up balance failed for {var}: {diff:.3e}"

    def test_shock_groups_mismatched_initial_state_captured_in_residual(self, four_shock_model):
        """Verify that misspecified initial state generates residual component preserving adding-up."""
        T = 25
        rng = np.random.default_rng(444)
        shocks = rng.normal(0.0, 0.01, size=(T, 4))
        # Generated from zero initial state
        sim_df = four_shock_model.simulate(periods=T, shocks=shocks, burn=0)

        # Decomposed with deliberate non-zero initial state
        s0_mismatch = np.array([0.1, -0.05, 0.08, -0.03])
        with pytest.warns(UserWarning, match="the smoothed model path does not reproduce the observed data"):
            res = shock_groups_decomposition(four_shock_model, data=sim_df, initial_state=s0_mismatch)

        assert "residual" in res.components
        assert np.any(np.abs(res.components["residual"]) > 1e-4)

        # Even under misspecification, the total sum including residual identically equals observed data
        for var in four_shock_model.variables:
            total = np.zeros(T)
            for g_name in res.components:
                total += res.components[g_name][var].to_numpy()
            diff = np.max(np.abs(total - sim_df[var].to_numpy()))
            assert diff <= 1e-12, f"Adding-up identity violated under misspecified initial state for {var}: {diff:.3e}"


# ===========================================================================
# Dimension 5: Tunable QZ Criterium & Bayesian IRF Credible Interval Monotonicity
# ===========================================================================

class TestQZCriteriumAndBayesianIRF:
    """Stress testing tunable QZ criterium unit roots and Bayesian IRF credible bands."""

    def test_tunable_qz_criterium_unit_root_boundary(self):
        """Verify qz_criterium correctly admits or rejects unit root eigenvalues."""
        mod_text = """
        var y, a;
        varexo e_a;
        parameters rho;
        rho = 1.0;

        model;
        y = a;
        a = rho * a(-1) + e_a;
        end;
        """
        # 1. Permissive criterium admits unit root
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m_admit = load_mod(mod_text, qz_criterium=1.0 + 1e-6)
            assert m_admit.solution is not None
            np.testing.assert_allclose(m_admit.solution.G, [[1.0]], atol=1e-10)

        # 2. Strict criterium rejects unit root as explosive/indeterminate
        from puremacro.dsge.klein import BlanchardKahnError
        with pytest.raises((BlanchardKahnError, ValueError)):
            load_mod(mod_text, qz_criterium=1.0 - 1e-6)

    def test_bayesian_irf_credible_bands_monotonicity_under_indeterminacy(self, nk_model_zlb):
        """Verify that Bayesian IRF bands satisfy lower_95 <= lower_68 <= median <= upper_68 <= upper_95."""
        priors = {
            "kappa": GammaPrior(mean=0.15, std=0.05),
            "phi_pi": UniformPrior(lb=0.5, ub=2.0),  # spans indeterminate region phi_pi < 1.0
            "rho_r": BetaPrior(mean=0.7, std=0.1),
        }
        res = bayesian_irf(
            nk_model_zlb,
            priors=priors,
            shock="e_m",
            periods=12,
            n_draws=40,
            bands=(0.68, 0.90, 0.95),
            seed=42,
        )

        assert isinstance(res, BayesianIRFResult)
        # Verify determinacy rate is strictly between 0 and 1
        assert 0.0 < res.determinacy_rate < 1.0
        assert res.n_valid < 40

        # Monotonicity check across all variables and horizons
        for var in nk_model_zlb.variables:
            l95, u95 = res.bands[0.95][0][var].to_numpy(), res.bands[0.95][1][var].to_numpy()
            l90, u90 = res.bands[0.90][0][var].to_numpy(), res.bands[0.90][1][var].to_numpy()
            l68, u68 = res.bands[0.68][0][var].to_numpy(), res.bands[0.68][1][var].to_numpy()
            med = res.median[var].to_numpy()

            assert np.all(l95 <= l90 + 1e-12)
            assert np.all(l90 <= l68 + 1e-12)
            assert np.all(l68 <= med + 1e-12)
            assert np.all(med <= u68 + 1e-12)
            assert np.all(u68 <= u90 + 1e-12)
            assert np.all(u90 <= u95 + 1e-12)
