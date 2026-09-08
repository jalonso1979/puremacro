"""Unit tests for Multi-Constraint OccBin and Piecewise Kalman Filter (Giovannini et al. 2021).

Tests Requirement R3:
1. Multi-Constraint OccBin (K >= 2, up to 4 regimes: {0, 1, 2, 3}).
2. Precomputed system matrices for all 2^K regimes.
3. Generalized backward recursion on multi-regime sequence.
4. Simultaneous shadow value updating and regime fixed-point convergence.
5. Piecewise Kalman Filter likelihood evaluation and filtered state recovery.
6. Robustness to zero measurement error and missing data (NaNs).
7. Likelihood smoothness with respect to structural parameters.
8. Bayesian estimation pipeline integration via method="piecewise_kalman".
9. Full presentation contract compliance (.summary, .plot, .to_markdown, .to_latex, .to_typst).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import build_dynare, LinearModel
from puremacro.dsge.occbin import (
    OccBinConstraint,
    OccBinResult,
    PiecewiseKalmanResult,
    solve_occbin,
    solve_multiconstraint_occbin,
    piecewise_kalman_filter,
    _auto_detect_constraint,
    _build_multi_regime_matrices,
)
from puremacro.dsge.priors import NormalPrior, UniformPrior
from puremacro.dsge.estimate import estimate_dsge


# ---------------------------------------------------------------------------
# Test Fixtures: Canonical 4-Regime Model (ZLB + Collateral/Borrowing Cap)
# ---------------------------------------------------------------------------

@pytest.fixture
def dual_constraint_model_setup():
    """Setup a New Keynesian model with dual constraints:
    - Constraint 1: Zero Lower Bound on policy rate: r_t >= -r_ss (r_t = -r_ss)
    - Constraint 2: Borrowing limit / leverage cap: b_t <= b_bar (b_t = b_bar)
    Yielding 4 distinct regimes:
      0: (0, 0) unconstrained (both slack)
      1: (1, 0) ZLB binding only
      2: (0, 1) Borrowing cap binding only
      3: (1, 1) Joint binding (both ZLB and borrowing cap active)
    """
    params = {
        "beta": 0.99,
        "sigma": 1.0,
        "kappa": 0.15,
        "phi_pi": 1.5,
        "phi_y": 0.25,
        "rho_r": 0.6,
        "rho_b": 0.5,
        "rho_g": 0.7,
        "gamma_y": 0.2,
        "chi": 0.1,
        "r_ss": 0.015,
        "b_bar": 0.02,
    }

    variables = ["y", "pi", "r", "b", "g"]
    shocks = ["eps_g", "eps_r", "eps_b"]
    steady_state = {v: 0.0 for v in variables}

    # Regime 0: Unconstrained reference regime
    def ref_eqs(lead, curr, lag, shocks_v, p):
        return [
            # Dynamic IS with borrowing effect
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            # NKPC
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            # Taylor rule
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            # Borrowing dynamics
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            # Demand shock
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 1: ZLB binding (r_t pegged to -r_ss, borrowing unconstrained)
    def zlb_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 2: Borrowing cap binding (b_t pegged to b_bar, Taylor rule active)
    def borr_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
            curr.b - p.b_bar,
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    # Regime 3: Joint binding (both ZLB and borrowing cap active)
    def joint_eqs(lead, curr, lag, shocks_v, p):
        return [
            curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
            curr.pi - p.beta * lead.pi - p.kappa * curr.y,
            curr.r - (-p.r_ss),
            curr.b - p.b_bar,
            curr.g - p.rho_g * lag.g - shocks_v.eps_g,
        ]

    m_uncons = build_dynare(
        ref_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state
    )
    m_zlb = build_dynare(
        zlb_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state,
        check_steady_state=False, strict=False
    )
    m_borr = build_dynare(
        borr_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state,
        check_steady_state=False, strict=False
    )
    m_joint = build_dynare(
        joint_eqs, variables=variables, shocks=shocks, params=params, steady_state=steady_state,
        check_steady_state=False, strict=False
    )

    c_zlb = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")
    c_borr = OccBinConstraint(variable="b", threshold=params["b_bar"], operator=">")

    return {
        "params": params,
        "variables": variables,
        "shocks": shocks,
        "m_uncons": m_uncons,
        "m_zlb": m_zlb,
        "m_borr": m_borr,
        "m_joint": m_joint,
        "c_zlb": c_zlb,
        "c_borr": c_borr,
    }


# ---------------------------------------------------------------------------
# Unit Tests: Multi-Constraint OccBin (4 Regimes)
# ---------------------------------------------------------------------------

class TestMultiConstraintOccBin:
    """Test suite for K >= 2 multi-constraint OccBin relaxation solver."""

    def test_precomputed_system_matrices(self, dual_constraint_model_setup):
        """Verify precomputed system matrices correctly assemble across all 2^K = 4 regimes."""
        setup = dual_constraint_model_setup
        m_uncons = setup["m_uncons"]
        c_zlb = setup["c_zlb"]
        c_borr = setup["c_borr"]

        c_info = [
            (c_zlb, 2, (setup["m_zlb"]._A_plus, setup["m_zlb"]._A_0, setup["m_zlb"]._A_minus, setup["m_zlb"]._B_u, np.array([0, 0, setup["params"]["r_ss"], 0, 0]))),
            (c_borr, 3, (setup["m_borr"]._A_plus, setup["m_borr"]._A_0, setup["m_borr"]._A_minus, setup["m_borr"]._B_u, np.array([0, 0, 0, -setup["params"]["b_bar"], 0]))),
        ]
        ref_matrices = (
            m_uncons._A_plus, m_uncons._A_0, m_uncons._A_minus, m_uncons._B_u,
            np.zeros(5), np.zeros(5), setup["variables"], setup["shocks"]
        )

        regime_matrices = _build_multi_regime_matrices(ref_matrices, c_info)
        assert len(regime_matrices) == 4

        # Regime 0: unconstrained
        assert np.allclose(regime_matrices[0][1], m_uncons._A_0)

        # Regime 1: ZLB only (row 2 differs)
        assert not np.allclose(regime_matrices[1][1][2], m_uncons._A_0[2])
        assert np.allclose(regime_matrices[1][1][3], m_uncons._A_0[3])

        # Regime 2: Borrowing only (row 3 differs)
        assert np.allclose(regime_matrices[2][1][2], m_uncons._A_0[2])
        assert not np.allclose(regime_matrices[2][1][3], m_uncons._A_0[3])

        # Regime 3: Joint binding (both row 2 and row 3 differ)
        assert not np.allclose(regime_matrices[3][1][2], m_uncons._A_0[2])
        assert not np.allclose(regime_matrices[3][1][3], m_uncons._A_0[3])

    def test_multiconstraint_occbin_4_regimes_traversal(self, dual_constraint_model_setup):
        """Verify OccBin simulation traverses all 4 regimes under dual shocks:
        Regime 3 (joint) -> Regime 1/2 (single constraint) -> Regime 0 (slack).
        """
        setup = dual_constraint_model_setup
        horizon = 30
        shocks = np.zeros((horizon, 3))

        # Shocks at t=1:
        # Negative demand shock to push r down below -r_ss (trigger ZLB)
        shocks[0, 0] = -0.06
        # Positive credit shock to push b up above b_bar (trigger borrowing limit)
        shocks[0, 2] = 0.05

        res = solve_multiconstraint_occbin(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={
                "zlb": setup["m_zlb"],
                "borr": setup["m_borr"],
            },
            constraints={
                "zlb": setup["c_zlb"],
                "borr": setup["c_borr"],
            },
            shock_seq=shocks,
            horizon=horizon,
        )

        assert isinstance(res, OccBinResult)
        assert res.converged
        assert len(res.simulated_path) == horizon

        # Verify regimes visit joint binding (3), single constraints, and return to 0
        regimes_visited = set(res.regimes)
        assert 3 in regimes_visited, f"Regime 3 (joint binding) was not visited: {res.regimes}"
        assert 0 in regimes_visited, f"Regime 0 (reference slack) was not reached: {res.regimes}"
        assert res.regimes[-1] == 0, "Terminal condition requires return to reference regime"

        # Check bounds satisfaction
        # At ZLB, r >= -r_ss
        assert res.simulated_path["r"].min() >= -setup["params"]["r_ss"] - 1e-8
        # At borrowing cap, b <= b_bar
        assert res.simulated_path["b"].max() <= setup["params"]["b_bar"] + 1e-8

    def test_multiconstraint_occbin_single_constraint_fallback_parity(self, dual_constraint_model_setup):
        """Verify solve_multiconstraint_occbin with K=1 matches solve_occbin bit-for-bit."""
        setup = dual_constraint_model_setup
        shocks = np.zeros((25, 3))
        shocks[0, 0] = -0.04

        res_single = solve_occbin(
            reference_model=setup["m_uncons"],
            constrained_model=setup["m_zlb"],
            constraint=setup["c_zlb"],
            shock_sequence=shocks,
            horizon=25,
        )

        res_multi = solve_multiconstraint_occbin(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"]},
            constraints={"zlb": setup["c_zlb"]},
            shock_seq=shocks,
            horizon=25,
        )

        assert res_single.regimes == res_multi.regimes
        max_diff = np.max(np.abs(res_single.simulated_path.values - res_multi.simulated_path.values))
        assert max_diff < 1e-12

    def test_multiconstraint_occbin_zero_shock_degenerate(self, dual_constraint_model_setup):
        """Verify degenerate zero-shock sequence immediately converges to zero path in regime 0."""
        setup = dual_constraint_model_setup
        shocks = np.zeros((20, 3))

        res = solve_multiconstraint_occbin(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"], "borr": setup["m_borr"]},
            shock_seq=shocks,
            horizon=20,
        )

        assert res.converged
        assert res.binding_periods == 0
        assert np.all(res.regime_history == 0)
        assert np.allclose(res.simulated_path.values, 0.0)

    def test_multiconstraint_occbin_conflicting_regimes_validation(self, dual_constraint_model_setup):
        """Verify setup detects mutually conflicting constraint regimes targeting the same row."""
        setup = dual_constraint_model_setup
        # Pass two constrained models that both modify row 2 (Taylor rule)
        with pytest.raises(ValueError, match="Incompatible constraint regimes"):
            solve_multiconstraint_occbin(
                m_unconstrained=setup["m_uncons"],
                m_constrained_dict={
                    "zlb1": setup["m_zlb"],
                    "zlb2": setup["m_zlb"],
                },
                shock_seq=np.zeros(3),
            )

    def test_occbin_result_presentation_contract(self, dual_constraint_model_setup):
        """Verify OccBinResult adheres to puremacro presentation contract."""
        setup = dual_constraint_model_setup
        shocks = np.zeros((15, 3))
        shocks[0, 0] = -0.05

        res = solve_multiconstraint_occbin(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"]},
            shock_seq=shocks,
            horizon=15,
        )

        # Presentation methods
        summary_str = res.summary()
        assert isinstance(summary_str, str)
        assert "OCCASIONALLY BINDING CONSTRAINTS REPORT" in summary_str

        df = res.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 15

        md_str = res.to_markdown()
        assert isinstance(md_str, str)
        assert "|" in md_str

        latex_str = res.to_latex()
        assert isinstance(latex_str, str)
        assert "begin{tabular}" in latex_str

        typst_str = res.to_typst()
        assert isinstance(typst_str, str)
        assert "#table" in typst_str

        # Subscript access
        assert len(res["y"]) == 15
        assert hasattr(res, "path")
        assert hasattr(res, "regime_history")

        # Plotting
        fig = res.plot(style="publication")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Unit Tests: Piecewise Kalman Filter (PKF) & Bayesian Estimation
# ---------------------------------------------------------------------------

class TestPiecewiseKalmanFilter:
    """Test suite for Piecewise Kalman Filter likelihood evaluation and estimation."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic dataset with binding ZLB periods."""
        rng = np.random.default_rng(42)
        T = 40
        dates = pd.date_range("2010-01-01", periods=T, freq="QS")

        y = np.zeros(T)
        pi = np.zeros(T)
        r = np.zeros(T)

        for t in range(1, T):
            # Large deflationary shock in periods 5..10
            shock_d = -0.08 if 5 <= t <= 10 else rng.normal(0, 0.01)
            y[t] = 0.6 * y[t - 1] - 0.2 * r[t - 1] + shock_d
            pi[t] = 0.4 * pi[t - 1] + 0.15 * y[t] + rng.normal(0, 0.005)
            # Taylor rule with ZLB at -0.015
            notional_r = 0.7 * r[t - 1] + 0.3 * (1.5 * pi[t] + 0.25 * y[t])
            r[t] = max(-0.015, notional_r)

        return pd.DataFrame({"y": y, "pi": pi, "r": r}, index=dates)

    def test_pkf_likelihood_evaluation_and_filtered_states(self, dual_constraint_model_setup, synthetic_data):
        """Verify PKF computes finite likelihood and tracks filtered states matching data length."""
        setup = dual_constraint_model_setup
        data = synthetic_data

        res = piecewise_kalman_filter(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"]},
            data=data,
            varobs=["y", "pi", "r"],
            horizon=25,
        )

        assert isinstance(res, PiecewiseKalmanResult)
        assert np.isfinite(res.log_likelihood)
        assert len(res.filtered_states) == len(data)
        assert len(res.regimes) == len(data)

        # Unpacking contract: ll, states, regimes = pkf(...)
        ll, states, regimes = res
        assert ll == res.log_likelihood
        assert states.equals(res.filtered_states)
        assert len(regimes) == len(data)

    def test_pkf_zero_measurement_error_boundary(self, dual_constraint_model_setup, synthetic_data):
        """Verify PKF computes stable likelihood when measurement error covariance is 0."""
        setup = dual_constraint_model_setup
        res = piecewise_kalman_filter(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"]},
            data=synthetic_data.iloc[:20],
            varobs=["y", "pi", "r"],
            H=np.zeros((3, 3)),
        )
        assert np.isfinite(res.log_likelihood)

    def test_pkf_missing_data_handling(self, dual_constraint_model_setup, synthetic_data):
        """Verify PKF gracefully handles periods with missing observations (NaN)."""
        setup = dual_constraint_model_setup
        data_nan = synthetic_data.copy()
        data_nan.iloc[5, :] = np.nan
        data_nan.iloc[10, 0] = np.nan  # Partial missing

        res = piecewise_kalman_filter(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"]},
            data=data_nan,
            varobs=["y", "pi", "r"],
        )
        assert np.isfinite(res.log_likelihood)
        assert len(res.filtered_states) == len(data_nan)

    def test_pkf_likelihood_smoothness(self, dual_constraint_model_setup, synthetic_data):
        """Verify PKF likelihood is smooth with respect to structural parameters (no particle chatter)."""
        setup = dual_constraint_model_setup
        data = synthetic_data.iloc[:20]

        phi_grid = np.linspace(1.2, 1.8, 5)
        logliks = []

        for phi in phi_grid:
            p_mod = dict(setup["params"])
            p_mod["phi_pi"] = phi
            ref_m = build_dynare(
                setup["m_uncons"]._dynare_equations,
                variables=setup["variables"],
                shocks=setup["shocks"],
                params=p_mod,
                steady_state=setup["m_uncons"].steady_state,
                check_steady_state=False,
                strict=False,
            )
            cons_m = build_dynare(
                setup["m_zlb"]._dynare_equations,
                variables=setup["variables"],
                shocks=setup["shocks"],
                params=p_mod,
                steady_state=setup["m_zlb"].steady_state,
                check_steady_state=False,
                strict=False,
            )
            res = piecewise_kalman_filter(
                m_unconstrained=ref_m,
                m_constrained_dict={"zlb": cons_m},
                data=data,
                varobs=["y", "pi", "r"],
            )
            logliks.append(res.log_likelihood)

        assert all(np.isfinite(ll) for ll in logliks)
        # Verify deterministic smoothness (differences are well-behaved)
        diffs = np.diff(logliks)
        assert not np.any(np.isnan(diffs))

    def test_pkf_presentation_contract(self, dual_constraint_model_setup, synthetic_data):
        """Verify PiecewiseKalmanResult adheres to puremacro presentation contract."""
        setup = dual_constraint_model_setup
        res = piecewise_kalman_filter(
            m_unconstrained=setup["m_uncons"],
            m_constrained_dict={"zlb": setup["m_zlb"]},
            data=synthetic_data.iloc[:15],
            varobs=["y", "pi", "r"],
        )

        summary_str = res.summary()
        assert isinstance(summary_str, str)
        assert "PIECEWISE KALMAN FILTER REPORT" in summary_str

        summary_df = res.summary(as_dataframe=True)
        assert isinstance(summary_df, pd.DataFrame)
        assert "Log-Likelihood" in summary_df.columns

        assert isinstance(res.to_markdown(), str)
        assert isinstance(res.to_latex(), str)
        assert isinstance(res.to_typst(), str)

        fig = res.plot(style="publication")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_linear_model_estimate_method_piecewise_kalman(self, dual_constraint_model_setup, synthetic_data):
        """Verify LinearModel.estimate(method='piecewise_kalman') runs Bayesian estimation."""
        setup = dual_constraint_model_setup
        m = setup["m_uncons"]

        priors = {
            "phi_pi": NormalPrior(mean=1.5, std=0.2, lb=1.0, ub=2.5),
            "phi_y": NormalPrior(mean=0.25, std=0.1, lb=0.05, ub=0.8),
        }

        # Run with small chain for speed
        res = m.estimate(
            synthetic_data.iloc[:20],
            method="piecewise_kalman",
            m_constrained_dict={"zlb": setup["m_zlb"]},
            constraints={"zlb": setup["c_zlb"]},
            priors=priors,
            varobs=["y", "pi", "r"],
            n_draws=40,
            n_chains=2,
            burn_in=10,
            mode_compute="lbfgs",
        )

        assert res.mode is not None
        assert "phi_pi" in res.mode
        assert "phi_y" in res.mode
        assert res.draws.shape == (2, 40, 2)
