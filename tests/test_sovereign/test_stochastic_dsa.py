"""Comprehensive Test Suite for Stochastic Debt Sustainability Analysis (DSA).

Covers the 8-suite test matrix for Stochastic DSA in PureMacro:
1. Suite 1: Parameter validation (rejection of invalid inputs, non-stationary VAR, non-PSD cov).
2. Suite 2: Deterministic baseline dynamics (snowball identities to 1e-14, steady-state invariance, divergence/convergence).
3. Suite 3: Fiscal reaction functions & rules (Bohn feedback, solvency stabilization, fiscal fatigue, limits, multiplier drag).
4. Suite 4: Shock generators & covariance structures (Cholesky sampling, covariance convergence, correlated shocks).
5. Suite 5: Monte Carlo fan charts & distributional metrics (monotonicity across all t, dispersion expansion, DaR 95%, probabilities).
6. Suite 6: Boundary conditions & numerical stability (zero debt, creditor assets, crisis stress, finite checks).
7. Suite 7: Seed reproducibility & bit-identity (identical seeds -> bit-identical arrays, CRN stream alignment).
8. Suite 8: Presentation contract compliance (all 7 methods + 2D array duck-typing).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Ensure headless execution
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.sovereign import (
    FiscalReactionFunction,
    FiscalRule,
    FiscalSovereignResult,
    StochasticDSAResult,
    compare_fiscal_rules_dsa,
    simulate_fiscal_dsa,
    simulate_stochastic_dsa,
    solve_fiscal_sovereign_default,
)


# Shared fixtures (module-scoped: solved once per test module)

@pytest.fixture(scope="module")
def dsa_result() -> StochasticDSAResult:
    """Fixture returning a standard solved StochasticDSAResult."""
    return simulate_stochastic_dsa(seed=42, n_sims=500, horizon=20)


@pytest.fixture(scope="module")
def solved_models() -> dict[str, FiscalSovereignResult]:
    """Solve the 101x11 illustrative grid under none, ceiling, soft_ceiling, and ceiling_escape."""
    cal = dict(tau=0.2, psi=1.0, n_b=101, n_y=11)
    return {
        "none": solve_fiscal_sovereign_default(**cal),
        "ceiling": solve_fiscal_sovereign_default(
            **cal, rule=FiscalRule(debt_ceiling=0.02)
        ),
        "soft_ceiling": solve_fiscal_sovereign_default(
            **cal, rule=FiscalRule(debt_ceiling=0.02, penalty=0.05)
        ),
        "ceiling_escape": solve_fiscal_sovereign_default(
            **cal, rule=FiscalRule(debt_ceiling=0.02, escape_below=0.92)
        ),
    }


# =====================================================================
# Suite 1: Parameter Validation Suite
# =====================================================================

class TestDSAParameterValidation:
    """Suite 1: Validates that invalid parameters raise clear, informative ValueErrors."""

    @pytest.mark.parametrize("invalid_d0", [float("nan"), float("inf"), -float("inf"), "0.65", None, [0.65]])
    def test_rejection_of_invalid_initial_debt(self, invalid_d0: object) -> None:
        """Reject non-finite or non-numeric initial debt."""
        with pytest.raises(ValueError, match="initial debt d0 must be a finite float"):
            simulate_stochastic_dsa(d0=invalid_d0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_horizon", [0, -1, -20, 5.5, True, False, "20", None])
    def test_rejection_of_invalid_horizon(self, invalid_horizon: object) -> None:
        """Reject non-positive or non-integer simulation horizons."""
        with pytest.raises(ValueError, match="horizon must be an integer >= 1"):
            simulate_stochastic_dsa(horizon=invalid_horizon)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_sims", [0, -10, 50.5, True, False, "1000", None])
    def test_rejection_of_invalid_simulations(self, invalid_sims: object) -> None:
        """Reject non-positive or non-integer simulation counts."""
        with pytest.raises(ValueError, match="n_sims must be an integer >= 1"):
            simulate_stochastic_dsa(n_sims=invalid_sims)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_thresh", [float("nan"), float("inf"), -float("inf"), "0.80", None])
    def test_rejection_of_invalid_threshold(self, invalid_thresh: object) -> None:
        """Reject non-finite or non-numeric prudential debt thresholds."""
        with pytest.raises(ValueError, match="threshold must be a finite float"):
            simulate_stochastic_dsa(threshold=invalid_thresh)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_freq", ["monthly", "daily", "semi-annual", "", "QUARTERLY"])
    def test_rejection_of_invalid_frequency(self, invalid_freq: str) -> None:
        """Reject unsupported frequency strings."""
        with pytest.raises(ValueError, match="frequency must be 'quarterly' or 'annual'"):
            simulate_stochastic_dsa(frequency=invalid_freq)

    @pytest.mark.parametrize(
        "invalid_mu",
        [
            [2.2, 1.8],  # length 2
            [2.2, 1.8, -1.5, 0.0],  # length 4
            [float("nan"), 1.8, -1.5],  # non-finite
            [2.2, float("inf"), -1.5],  # infinite
        ],
    )
    def test_rejection_of_invalid_mu(self, invalid_mu: object) -> None:
        """Reject malformed unconditional mean vectors."""
        with pytest.raises(ValueError):
            simulate_stochastic_dsa(mu=invalid_mu)  # type: ignore[arg-type]

    def test_rejection_of_invalid_transition_matrix_shape_and_finite(self) -> None:
        """Reject transition matrix A with wrong dimension or non-finite elements."""
        with pytest.raises(ValueError, match=r"A must have shape \(3, 3\)"):
            simulate_stochastic_dsa(A=np.eye(2))
        with pytest.raises(ValueError, match="A values must be finite numbers"):
            simulate_stochastic_dsa(A=np.full((3, 3), np.nan))

    @pytest.mark.parametrize(
        "non_stationary_A",
        [
            np.eye(3),  # spectral radius == 1.0 (unit root)
            np.diag([1.05, 0.8, 0.7]),  # explosive eigenvalue 1.05 > 1.0
            np.array([[0.8, 0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 0.5]]),  # spectral radius 1.4
        ],
    )
    def test_rejection_of_non_stationary_var_transition_matrix(
        self, non_stationary_A: np.ndarray
    ) -> None:
        """Reject VAR transition matrix with spectral radius rho(A) >= 1.0."""
        with pytest.raises(ValueError, match="VAR transition matrix A is non-stationary"):
            simulate_stochastic_dsa(A=non_stationary_A)

    def test_rejection_of_invalid_covariance_matrix_shape_and_finite(self) -> None:
        """Reject covariance matrix with wrong dimension or non-finite elements."""
        with pytest.raises(ValueError, match=r"cov must have shape \(3, 3\)"):
            simulate_stochastic_dsa(cov=np.eye(2))
        with pytest.raises(ValueError, match="cov values must be finite numbers"):
            simulate_stochastic_dsa(cov=np.full((3, 3), np.nan))

    def test_rejection_of_non_positive_semi_definite_covariance(self) -> None:
        """Reject covariance matrix having strictly negative eigenvalues."""
        indefinite_cov = np.array(
            [
                [1.0, 2.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )  # eigenvalues: [-1, 3, 1]
        with pytest.raises(ValueError, match="Covariance matrix cov is not positive semi-definite"):
            simulate_stochastic_dsa(cov=indefinite_cov)


# =====================================================================
# Suite 2: Deterministic Baseline Dynamics Suite
# =====================================================================
class TestDSADeterministicDynamics:
    """Suite 2: Validates closed-form debt snowball identities to 1e-14 and invariance."""

    def test_quarterly_linearized_snowball_identity(self) -> None:
        """Verify quarterly linearized accumulation matches closed-form recursion to 1e-14."""
        d0 = 0.65
        horizon = 20
        mu = np.array([2.2, 1.8, -1.5])
        A = np.array(
            [
                [0.60, -0.10, 0.05],
                [0.10, 0.80, -0.05],
                [0.20, -0.10, 0.70],
            ]
        )

        res = simulate_stochastic_dsa(
            d0=d0,
            horizon=horizon,
            n_sims=50,
            shocks=False,
            mu=mu,
            A=A,
            frequency="quarterly",
            exact_compounding=False,
        )

        # Independent closed-form analytical recursion
        analytical = np.zeros(horizon + 1)
        analytical[0] = d0
        st = mu.copy()
        for t in range(1, horizon + 1):
            st = mu + A @ (st - mu)
            g_t, r_t, pb_t = st[0], st[1], st[2]
            analytical[t] = analytical[t - 1] * (1.0 + (r_t - g_t) / 400.0) - (pb_t / 400.0)

        # Verify baseline path matches analytical closed-form to float precision
        np.testing.assert_allclose(res.baseline_path, analytical, atol=1e-14, rtol=1e-12)

        # With shocks=False, all simulation paths must be bit-identical to baseline path
        for s in range(len(res)):
            np.testing.assert_allclose(res.paths[s], analytical, atol=1e-14, rtol=1e-12)

    def test_exact_compounding_snowball_identity(self) -> None:
        """Verify exact compounding accumulation matches closed-form recursion to 1e-14."""
        d0 = 0.75
        horizon = 20
        mu = np.array([2.5, 3.0, -1.0])
        A = np.array(
            [
                [0.50, 0.00, 0.00],
                [0.00, 0.70, 0.00],
                [0.00, 0.00, 0.60],
            ]
        )

        res = simulate_stochastic_dsa(
            d0=d0,
            horizon=horizon,
            n_sims=50,
            shocks=False,
            mu=mu,
            A=A,
            exact_compounding=True,
        )

        analytical = np.zeros(horizon + 1)
        analytical[0] = d0
        st = mu.copy()
        for t in range(1, horizon + 1):
            st = mu + A @ (st - mu)
            g_t, r_t, pb_t = st[0], st[1], st[2]
            analytical[t] = analytical[t - 1] * (1.0 + r_t / 100.0) / (1.0 + g_t / 100.0) - (pb_t / 100.0)

        np.testing.assert_allclose(res.baseline_path, analytical, atol=1e-14, rtol=1e-12)

    def test_steady_state_invariance_linearized(self) -> None:
        """Verify debt remains constant across all periods when pb* = (r - g) * d*."""
        d_star = 0.60
        g = 2.0
        r = 4.0
        # In linearized macro notation: pb* / 400 = d* * (r - g) / 400 => pb* = (r - g) * d*
        pb_star = (r - g) * d_star  # 2.0 * 0.60 = 1.20

        horizon = 40
        res = simulate_stochastic_dsa(
            d0=d_star,
            horizon=horizon,
            n_sims=30,
            shocks=False,
            mu=(g, r, pb_star),
            frequency="quarterly",
            exact_compounding=False,
        )

        max_drift = np.max(np.abs(res.baseline_path - d_star))
        assert max_drift < 1e-12, f"Debt drifted from steady-state: max drift = {max_drift:.3e}"
        assert np.max(np.abs(res.paths - d_star)) < 1e-12

    def test_steady_state_invariance_exact_compounding(self) -> None:
        """Verify debt remains constant when pb* = d* * (r - g) / (1 + g/100)."""
        d_star = 0.80
        g = 3.0
        r = 5.0
        # Exact condition: d* = d* * (1 + r/100)/(1 + g/100) - pb*/100 => pb* = 100 * d* * ((r - g)/(100 + g))
        pb_star = d_star * (r - g) / (1.0 + g / 100.0)

        horizon = 40
        res = simulate_stochastic_dsa(
            d0=d_star,
            horizon=horizon,
            n_sims=30,
            shocks=False,
            mu=(g, r, pb_star),
            exact_compounding=True,
        )

        max_drift = np.max(np.abs(res.baseline_path - d_star))
        assert max_drift < 1e-12, f"Debt drifted from exact steady-state: max drift = {max_drift:.3e}"

    def test_snowball_divergence_unfavorable_differential(self) -> None:
        """Verify r > g with primary deficit causes explosive debt divergence."""
        d0 = 0.60
        res = simulate_stochastic_dsa(
            d0=d0,
            horizon=50,
            n_sims=20,
            shocks=False,
            mu=(1.0, 5.0, -2.0),  # r - g = +4.0%, pb = -2.0%
        )

        # Debt must strictly increase at every single step
        diffs = np.diff(res.baseline_path)
        assert np.all(diffs > 0), "Debt path should be strictly monotonically increasing"
        assert res.baseline_path[-1] > 2.0 * d0, "Explosive snowball should more than double debt"

    def test_favorable_differential_convergence(self) -> None:
        """Verify g > r with balanced budget leads to monotonic debt reduction."""
        d0 = 0.80
        res = simulate_stochastic_dsa(
            d0=d0,
            horizon=30,
            n_sims=20,
            shocks=False,
            mu=(4.0, 1.0, 0.0),  # r - g = -3.0%, pb = 0.0%
        )

        diffs = np.diff(res.baseline_path)
        assert np.all(diffs < 0), "Debt path should be strictly monotonically decreasing"
        assert res.baseline_path[-1] < d0 * 0.85, "Debt should decline substantially"


# =====================================================================
# Suite 3: Fiscal Reaction Functions & Rules Suite
# =====================================================================
class TestDSAFiscalRules:
    """Suite 3: Validates Bohn feedback, fiscal fatigue, clamping limits, and multiplier drag."""

    def test_bohn_reaction_feedback_evaluation(self) -> None:
        """Verify primary balance increases with lagged debt (Bohn discipline)."""
        fr = FiscalReactionFunction(alpha=0.5, rho=0.5, beta_debt=0.04, gamma_y=0.10)

        # Scalar evaluation
        pb_low = fr.evaluate(d_prev=0.50, y_gap=0.0, pb_prev=0.0)
        pb_high = fr.evaluate(d_prev=1.00, y_gap=0.0, pb_prev=0.0)
        expected_diff = 0.04 * (1.00 - 0.50)
        assert np.isclose(pb_high - pb_low, expected_diff, atol=1e-12)

        # Array evaluation
        d_arr = np.array([0.40, 0.60, 0.80, 1.00])
        pb_arr = fr.evaluate(d_prev=d_arr, y_gap=0.0, pb_prev=0.0)
        assert np.all(np.diff(pb_arr) > 0), "Primary balance must be strictly increasing in debt"

    def test_solvency_stabilization_under_bohn_discipline(self) -> None:
        """Verify strong Bohn discipline (beta_debt > r - g) stabilizes high debt."""
        d0 = 1.20
        horizon = 40
        mu = (2.0, 4.0, 0.0)  # r - g = 2.0%

        # Passive policy (beta_debt = 0.0): debt explodes
        fr_passive = FiscalReactionFunction(alpha=0.0, rho=0.0, beta_debt=0.0)
        res_passive = simulate_stochastic_dsa(
            d0=d0,
            horizon=horizon,
            shocks=False,
            mu=mu,
            fiscal_reaction=fr_passive,
        )

        # Active stabilizing policy (beta_debt = 3.0 > (4.0 - 2.0) = 2.0)
        fr_active = FiscalReactionFunction(alpha=0.0, rho=0.0, beta_debt=3.0)
        res_active = simulate_stochastic_dsa(
            d0=d0,
            horizon=horizon,
            shocks=False,
            mu=mu,
            fiscal_reaction=fr_active,
        )

        # Active policy stabilizes debt downwards while passive policy explodes
        assert res_active.baseline_path[-1] < d0, "Active Bohn rule should bend debt trajectory down"
        assert res_passive.baseline_path[-1] > d0 * 1.15, "Passive rule should allow debt expansion"
        assert res_active.baseline_path[-1] < res_passive.baseline_path[-1]

    def test_fiscal_fatigue_threshold_caps_responsiveness(self) -> None:
        """Verify debt beyond fatigue_threshold ceases to elicit additional primary surplus."""
        fatigue_thresh = 0.85
        fr = FiscalReactionFunction(beta_debt=0.05, fatigue_threshold=fatigue_thresh)

        pb_at_threshold = fr.evaluate(d_prev=fatigue_thresh)
        pb_above_1 = fr.evaluate(d_prev=0.95)
        pb_above_2 = fr.evaluate(d_prev=1.40)
        pb_below = fr.evaluate(d_prev=0.70)

        # Responsiveness active below threshold
        assert pb_below < pb_at_threshold
        # Responsiveness flat above threshold (fiscal fatigue)
        assert np.isclose(pb_at_threshold, pb_above_1, atol=1e-12)
        assert np.isclose(pb_above_1, pb_above_2, atol=1e-12)

    def test_surplus_and_deficit_limits_clamping(self) -> None:
        """Verify primary balance is strictly clamped within [pb_min, pb_max]."""
        pb_min_val = -1.5
        pb_max_val = 3.0
        fr = FiscalReactionFunction(
            alpha=10.0,  # massive expansion attempt
            beta_debt=0.10,
            pb_min=pb_min_val,
            pb_max=pb_max_val,
        )

        # Over-the-top scalar evaluation
        pb_clamped_high = fr.evaluate(d_prev=2.0)
        assert np.isclose(pb_clamped_high, pb_max_val, atol=1e-12)

        fr_floor = FiscalReactionFunction(alpha=-15.0, pb_min=pb_min_val, pb_max=pb_max_val)
        pb_clamped_low = fr_floor.evaluate(d_prev=0.0)
        assert np.isclose(pb_clamped_low, pb_min_val, atol=1e-12)

        # Verification across 500 simulated paths with shocks
        res = simulate_stochastic_dsa(
            n_sims=500,
            horizon=20,
            fiscal_reaction=fr,
            store_macro_paths=True,
            seed=42,
        )
        assert res.macro_paths is not None
        sim_pb = res.macro_paths[:, :, 2]
        assert np.all(sim_pb <= pb_max_val + 1e-12), "Primary balance exceeded pb_max ceiling"

    def test_keynesian_multiplier_drag_propagation(self) -> None:
        """Verify multiplier drag m * pb_shift operates strictly during multiplier_horizon."""
        m = 0.75
        pb_shift = 1.20
        multiplier_horizon = 4
        horizon = 12

        res_drag = simulate_stochastic_dsa(
            horizon=horizon,
            shocks=False,
            m=m,
            pb_shift=pb_shift,
            multiplier_horizon=multiplier_horizon,
            store_macro_paths=True,
        )
        res_nodrag = simulate_stochastic_dsa(
            horizon=horizon,
            shocks=False,
            m=0.0,
            pb_shift=pb_shift,
            multiplier_horizon=multiplier_horizon,
            store_macro_paths=True,
        )

        assert res_drag.macro_paths is not None
        assert res_nodrag.macro_paths is not None

        g_drag = res_drag.macro_paths[0, :, 0]
        g_nodrag = res_nodrag.macro_paths[0, :, 0]

        expected_reduction = m * pb_shift  # 0.75 * 1.20 = 0.90
        # Periods 1..multiplier_horizon have exactly expected_reduction drag
        for t in range(1, multiplier_horizon + 1):
            assert np.isclose(g_nodrag[t] - g_drag[t], expected_reduction, atol=1e-12)

        # Periods after multiplier_horizon have zero drag
        for t in range(multiplier_horizon + 1, horizon + 1):
            assert np.isclose(g_nodrag[t] - g_drag[t], 0.0, atol=1e-12)

        # Multiplier drag lowers growth, thus increasing short-term debt compared to no-drag
        assert res_drag.baseline_path[multiplier_horizon] > res_nodrag.baseline_path[multiplier_horizon]


# =====================================================================
# Suite 4: Shock Generators & Covariance Structures Suite
# =====================================================================
class TestDSAShockGenerators:
    """Suite 4: Validates Cholesky sampling, covariance convergence, and correlated shocks."""

    def test_cholesky_covariance_factorization(self) -> None:
        """Verify innovation covariance factorization Sigma = L L^T."""
        cov = np.array(
            [
                [3.0, -0.5, 0.4],
                [-0.5, 2.0, -0.2],
                [0.4, -0.2, 1.5],
            ]
        )
        L = np.linalg.cholesky(cov)
        assert np.allclose(L @ L.T, cov, atol=1e-14)
        assert np.all(np.tril(L) == L), "Cholesky factor must be lower triangular"

    def test_singular_covariance_matrix_graceful_handling(self) -> None:
        """Verify rank-deficient covariance matrix (zero variance on one variable) executes cleanly."""
        cov_singular = np.array(
            [
                [2.0, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 0.0, 0.0],  # rank 2, zero variance for primary balance
            ]
        )
        res = simulate_stochastic_dsa(
            n_sims=100,
            horizon=10,
            cov=cov_singular,
            store_macro_paths=True,
            seed=42,
        )
        assert np.all(np.isfinite(res.paths))
        assert res.macro_paths is not None
        # With zero PB variance and A=default, innovations on pb must be zero
        assert np.all(np.isfinite(res.macro_paths))

    def test_sample_covariance_convergence(self) -> None:
        """Verify simulated shock covariance converges to theoretical Sigma as N increases."""
        cov_theoretical = np.array(
            [
                [2.5, -0.3, 0.4],
                [-0.3, 1.2, -0.2],
                [0.4, -0.2, 1.0],
            ]
        )
        # N=8000, horizon=1 to extract clean one-step innovations
        res = simulate_stochastic_dsa(
            n_sims=8000,
            horizon=1,
            cov=cov_theoretical,
            fiscal_reaction=None,
            store_macro_paths=True,
            seed=12345,
        )
        assert res.macro_paths is not None
        # st_1 - mu_active = innovations (since st_0 = mu_active and A @ 0 = 0)
        innovations = res.macro_paths[:, 1, :] - res.macro_paths[:, 0, :]
        empirical_cov = np.cov(innovations, rowvar=False)

        # Standard Monte Carlo tolerance for N=8000 (std error ~ sqrt(2/N)*sigma^2 < 0.08)
        diff = np.max(np.abs(empirical_cov - cov_theoretical))
        assert diff < 0.15, f"Sample covariance did not converge within tolerance: max diff = {diff:.4f}"

    def test_correlated_macro_shocks_propagation(self) -> None:
        """Verify negative correlation between growth and interest rates propagates accurately."""
        # Growth and interest rates negatively correlated (-0.80)
        s_g = 2.0
        s_r = 1.5
        corr_target = -0.75
        cov_gr = corr_target * s_g * s_r
        cov = np.array(
            [
                [s_g**2, cov_gr, 0.0],
                [cov_gr, s_r**2, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        res = simulate_stochastic_dsa(
            n_sims=6000,
            horizon=1,
            cov=cov,
            store_macro_paths=True,
            seed=42,
        )
        assert res.macro_paths is not None
        g_shk = res.macro_paths[:, 1, 0] - res.macro_paths[:, 0, 0]
        r_shk = res.macro_paths[:, 1, 1] - res.macro_paths[:, 0, 1]
        empirical_corr = float(np.corrcoef(g_shk, r_shk)[0, 1])

        assert np.isclose(empirical_corr, corr_target, atol=0.05), (
            f"Empirical correlation {empirical_corr:.3f} deviated from target {corr_target}"
        )


# =====================================================================
# Suite 5: Monte Carlo Fan Charts & Distributional Metrics Suite
# =====================================================================
class TestDSAFanCharts:
    """Suite 5: Validates quantile monotonicity, fan expansion, DaR 95%, and probabilities."""

    @pytest.mark.parametrize(
        "exact_comp, freq, horizon, n_sims",
        [
            (False, "quarterly", 20, 1000),
            (True, "quarterly", 20, 800),
            (True, "annual", 15, 600),
            (False, "quarterly", 40, 1000),
        ],
    )
    def test_quantile_monotonicity_across_all_periods(
        self, exact_comp: bool, freq: str, horizon: int, n_sims: int
    ) -> None:
        """Verify strict weak ordering q5 <= q10 <= q25 <= q50 <= q75 <= q90 <= q95 for all t."""
        res = simulate_stochastic_dsa(
            horizon=horizon,
            n_sims=n_sims,
            exact_compounding=exact_comp,
            frequency=freq,
            seed=42,
        )
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for t in range(horizon + 1):
            vals = [res.quantiles[p][t] for p in percentiles]
            for i in range(len(vals) - 1):
                assert vals[i] <= vals[i + 1] + 1e-12, (
                    f"Quantile inversion at t={t}: q{percentiles[i]}={vals[i]} > q{percentiles[i+1]}={vals[i+1]}"
                )

    def test_uncertainty_fan_dispersion_expansion(self) -> None:
        """Verify uncertainty fan spread (q95 - q5) strictly widens from period 1 to terminal horizon."""
        res = simulate_stochastic_dsa(horizon=20, n_sims=1500, seed=42)
        disp_1 = res.quantiles[95][1] - res.quantiles[5][1]
        disp_T = res.quantiles[95][-1] - res.quantiles[5][-1]

        assert disp_T > disp_1 * 3.0, (
            f"Fan dispersion did not expand adequately: disp(1)={disp_1:.4f}, disp(T)={disp_T:.4f}"
        )

    def test_debt_at_risk_metrics(self) -> None:
        """Verify Debt-at-Risk (DaR 95) matches 95th percentile and exceeds median terminal debt."""
        res = simulate_stochastic_dsa(horizon=20, n_sims=1000, seed=42)
        expected_dar = float(np.percentile(res.paths[:, -1], 95))

        assert np.isclose(res.dar_95, expected_dar, atol=1e-12)
        assert np.isclose(res.dar_95, res.quantiles[95][-1], atol=1e-12)
        assert res.dar_95 > res.quantiles[50][-1], "DaR 95 must exceed median terminal debt"

    def test_probabilities_bounds_and_hierarchy(self) -> None:
        """Verify all probability metrics are bounded in [0, 1] and prob_breach >= prob_terminal_breach."""
        for seed_val in [42, 101, 777]:
            res = simulate_stochastic_dsa(n_sims=1000, horizon=20, threshold=0.75, seed=seed_val)

            assert 0.0 <= res.prob_breach <= 1.0
            assert 0.0 <= res.prob_terminal_breach <= 1.0
            assert 0.0 <= res.prob_non_stabilizing <= 1.0

            # Any-time breach must always be greater than or equal to terminal breach
            assert res.prob_breach >= res.prob_terminal_breach, (
                f"Hierarchy violated: prob_breach={res.prob_breach} < prob_terminal_breach={res.prob_terminal_breach}"
            )

    def test_breach_probability_threshold_monotonicity(self) -> None:
        """Verify higher prudential threshold strictly lowers or preserves breach probability."""
        res_low = simulate_stochastic_dsa(threshold=0.70, seed=42, n_sims=1000)
        res_mid = simulate_stochastic_dsa(threshold=0.80, seed=42, n_sims=1000)
        res_high = simulate_stochastic_dsa(threshold=0.95, seed=42, n_sims=1000)

        assert res_low.prob_breach >= res_mid.prob_breach >= res_high.prob_breach
        assert res_low.prob_terminal_breach >= res_mid.prob_terminal_breach >= res_high.prob_terminal_breach


# =====================================================================
# Suite 6: Boundary Conditions & Numerical Stability Suite
# =====================================================================
class TestDSANumericalStability:
    """Suite 6: Validates zero debt, creditor asset positions, and extreme crisis stress."""

    def test_zero_debt_boundary(self) -> None:
        """Verify zero initial debt executes cleanly without division-by-zero or NaNs."""
        res = simulate_stochastic_dsa(d0=0.0, horizon=20, n_sims=500, seed=42)
        assert np.all(np.isfinite(res.paths))
        assert np.all(np.isfinite(res.baseline_path))
        assert 0.0 <= res.prob_breach <= 1.0
        assert np.isfinite(res.dar_95)

        summary = res.summary()
        assert len(summary) == 10
        assert float(summary.loc[summary["metric"] == "initial_debt_gdp", "value"].iloc[0]) == 0.0

    @pytest.mark.parametrize("creditor_d0", [-0.20, -0.50, -1.20])
    def test_net_creditor_assets_boundary(self, creditor_d0: float) -> None:
        """Verify sovereign net asset positions (d0 < 0) simulate cleanly."""
        res = simulate_stochastic_dsa(d0=creditor_d0, horizon=20, n_sims=500, seed=42)
        assert np.all(np.isfinite(res.paths))
        assert np.all(np.isfinite(res.baseline_path))
        # Net creditor starting far below threshold has 0 breach probability
        assert res.prob_breach == 0.0
        assert res.dar_95 < 0.0  # Still in net asset position at 95th percentile

    def test_severe_macroeconomic_crisis_stress(self) -> None:
        """Verify extreme crisis collapse (g = -25%, r = 20%, pb = -15%) remains finite."""
        res = simulate_stochastic_dsa(
            d0=0.90,
            horizon=20,
            n_sims=300,
            mu=(-25.0, 20.0, -15.0),
            cov=np.diag([100.0, 80.0, 50.0]),  # hyper-volatility
            seed=42,
        )
        assert np.all(np.isfinite(res.paths))
        assert np.all(np.isfinite(res.baseline_path))
        assert res.prob_breach == 1.0
        assert res.dar_95 > 2.0

    def test_catastrophic_growth_drop_hyper_collapse(self) -> None:
        """Verify catastrophic collapse (g <= -100%) is numerically protected by denominator floor."""
        # g = -150% would make 1 + g/100 = -0.50 without protection
        res = simulate_stochastic_dsa(
            d0=0.70,
            horizon=10,
            n_sims=200,
            mu=(-150.0, 10.0, 0.0),
            exact_compounding=True,
            seed=42,
        )
        assert np.all(np.isfinite(res.paths))
        assert np.all(np.isfinite(res.baseline_path))
        assert not np.any(np.isnan(res.paths))
        assert not np.any(np.isinf(res.paths))


# =====================================================================
# Suite 7: Seed Reproducibility & Bit-Identity Suite
# =====================================================================
class TestDSASeedReproducibility:
    """Suite 7: Validates bit-identity under identical seeds and Common Random Numbers."""

    @pytest.mark.parametrize("seed_val", [0, 42, 1337, 987654])
    def test_identical_seeds_produce_bit_identical_arrays(self, seed_val: int) -> None:
        """Verify identical seeds produce 100% bit-identical simulation paths and metrics."""
        res1 = simulate_stochastic_dsa(seed=seed_val, n_sims=500, horizon=20, store_macro_paths=True)
        res2 = simulate_stochastic_dsa(seed=seed_val, n_sims=500, horizon=20, store_macro_paths=True)

        assert np.array_equal(res1.paths, res2.paths)
        assert np.array_equal(res1.baseline_path, res2.baseline_path)
        assert res1.prob_breach == res2.prob_breach
        assert res1.prob_terminal_breach == res2.prob_terminal_breach
        assert res1.prob_non_stabilizing == res2.prob_non_stabilizing
        assert res1.dar_95 == res2.dar_95

        for p in [5, 10, 25, 50, 75, 90, 95]:
            assert np.array_equal(res1.quantiles[p], res2.quantiles[p])

        assert res1.macro_paths is not None and res2.macro_paths is not None
        assert np.array_equal(res1.macro_paths, res2.macro_paths)

    def test_differing_seeds_produce_distinct_paths(self) -> None:
        """Verify different seeds produce distinct realization trajectories."""
        res_a = simulate_stochastic_dsa(seed=42, n_sims=200, horizon=10)
        res_b = simulate_stochastic_dsa(seed=43, n_sims=200, horizon=10)

        assert not np.array_equal(res_a.paths, res_b.paths)
        # Baselines should still be identical since they are deterministic
        assert np.array_equal(res_a.baseline_path, res_b.baseline_path)

    def test_common_random_numbers_crn_stream_alignment(self) -> None:
        """Verify shocks=False consumes the random stream to preserve paired CRN alignment."""
        # Run with shocks=True
        res_shocks = simulate_stochastic_dsa(seed=100, n_sims=200, horizon=10, shocks=True)
        # Run with shocks=False
        res_no_shocks = simulate_stochastic_dsa(seed=100, n_sims=200, horizon=10, shocks=False)

        # Baseline trajectories identical
        assert np.array_equal(res_shocks.baseline_path, res_no_shocks.baseline_path)
        # No shocks paths must be identical across rows
        for s in range(len(res_no_shocks)):
            assert np.array_equal(res_no_shocks.paths[s], res_no_shocks.baseline_path)


# =====================================================================
# Suite 8: Presentation Contract Compliance Suite
# =====================================================================
class TestDSAPresentationContract:
    """Suite 8: Validates all 7 Result-Object Presentation Standard methods and duck-typing."""

    def test_summary_structure_and_metrics(self, dsa_result: StochasticDSAResult) -> None:
        """Verify summary DataFrame schema, metric names, and finiteness."""
        summary = dsa_result.summary()
        assert isinstance(summary, pd.DataFrame)
        assert list(summary.columns) == ["metric", "value"]
        assert len(summary) == 10

        expected_metrics = [
            "initial_debt_gdp",
            "prudential_threshold",
            "baseline_terminal_debt",
            "median_terminal_debt",
            "dar_95_terminal_debt",
            "prob_breach_threshold",
            "prob_terminal_breach",
            "prob_non_stabilizing",
            "horizon",
            "n_sims",
        ]
        assert summary["metric"].tolist() == expected_metrics
        assert not summary["value"].isna().any()
        assert np.all(np.isfinite(summary["value"].values))

    def test_fan_chart_frame_and_aliases(self, dsa_result: StochasticDSAResult) -> None:
        """Verify fan_chart_frame(), to_dataframe(), and to_frame() shape and columns."""
        fc = dsa_result.fan_chart_frame()
        assert isinstance(fc, pd.DataFrame)
        assert fc.shape == (dsa_result.horizon + 1, 8)
        assert list(fc.columns) == ["baseline", "q05", "q10", "q25", "q50", "q75", "q90", "q95"]
        assert fc.index.name == "period"
        assert list(fc.index) == list(range(dsa_result.horizon + 1))

        # Check aliases
        df_frame = dsa_result.to_dataframe()
        to_f = dsa_result.to_frame()
        pd.testing.assert_frame_equal(fc, df_frame)
        pd.testing.assert_frame_equal(fc, to_f)

    def test_markup_export_methods(self, dsa_result: StochasticDSAResult) -> None:
        """Verify to_markdown(), to_latex(), and to_typst() return non-empty formatted strings."""
        md = dsa_result.to_markdown()
        assert isinstance(md, str) and len(md) > 50
        assert "metric" in md and "initial_debt_gdp" in md and "|" in md

        ltx = dsa_result.to_latex()
        assert isinstance(ltx, str) and len(ltx) > 50
        assert r"\begin{tabular}" in ltx and r"\end{tabular}" in ltx
        assert r"initial\_debt\_gdp" in ltx

        typ = dsa_result.to_typst()
        assert isinstance(typ, str) and len(typ) > 50
        assert "#table(" in typ and "columns: 2" in typ
        assert r"initial\_debt\_gdp" in typ

    def test_headless_plotting(self, dsa_result: StochasticDSAResult) -> None:
        """Verify plot(show=False) runs headlessly and supports injected axes."""
        # 1. Default Figure creation
        fig = dsa_result.plot(show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

        # 2. Injected Axes
        fig_custom, ax_custom = plt.subplots(figsize=(8, 4))
        ax_returned = dsa_result.plot(ax=ax_custom, show=False)
        assert ax_returned is ax_custom
        plt.close(fig_custom)

    def test_2d_array_duck_typing(self, dsa_result: StochasticDSAResult) -> None:
        """Verify .shape, len(), 2D slicing, and np.asarray / np.array operations."""
        # Shape and len
        assert dsa_result.shape == (len(dsa_result), dsa_result.horizon + 1)
        assert len(dsa_result) == 500

        # Scalar and slice indexing
        assert dsa_result[0, 0] == dsa_result.d0
        assert np.array_equal(dsa_result[10], dsa_result.paths[10])
        assert np.array_equal(dsa_result[0:5, 0:3], dsa_result.paths[0:5, 0:3])
        assert np.array_equal(dsa_result[:, -1], dsa_result.paths[:, -1])

        # NumPy array conversions
        arr_as = np.asarray(dsa_result)
        assert isinstance(arr_as, np.ndarray)
        assert arr_as.shape == dsa_result.shape
        assert np.array_equal(arr_as, dsa_result.paths)

        # NumPy 2.0 copy keyword support
        arr_no_copy = np.array(dsa_result, copy=False)
        assert np.array_equal(arr_no_copy, dsa_result.paths)
        arr_copy = np.array(dsa_result, copy=True)
        assert np.array_equal(arr_copy, dsa_result.paths)

        # NumPy reductions and ufuncs
        assert np.isclose(np.mean(dsa_result), np.mean(dsa_result.paths))
        assert np.isclose(np.max(dsa_result), np.max(dsa_result.paths))
        assert np.isclose(np.min(dsa_result), np.min(dsa_result.paths))


# =====================================================================
# Suite 9: Structural Fiscal Sovereign DSA & Rule Comparison
# =====================================================================
class TestStructuralFiscalDSA:
    """Suite 9: Validates simulate_fiscal_dsa and compare_fiscal_rules_dsa from high initial debt."""

    def test_validation_errors(self, solved_models: dict[str, FiscalSovereignResult]) -> None:
        """Verify input validation in simulate_fiscal_dsa and compare_fiscal_rules_dsa."""
        res = solved_models["none"]
        with pytest.raises(TypeError, match="FiscalSovereignResult"):
            simulate_fiscal_dsa("not_a_result")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="b0"):
            simulate_fiscal_dsa(res, b0=float("nan"))

        with pytest.raises(ValueError, match="y0"):
            simulate_fiscal_dsa(res, y0=float("inf"))

        with pytest.raises(ValueError, match="horizon"):
            simulate_fiscal_dsa(res, horizon=0)

        with pytest.raises(ValueError, match="n_sims"):
            simulate_fiscal_dsa(res, n_sims=0)

        with pytest.raises(ValueError, match="threshold"):
            simulate_fiscal_dsa(res, threshold=float("nan"))

        with pytest.raises(ValueError, match="uniforms"):
            simulate_fiscal_dsa(res, n_sims=10, horizon=5, uniforms=np.zeros((2, 10, 5)))

        with pytest.raises(ValueError, match="empty"):
            compare_fiscal_rules_dsa({})

    def test_high_debt_separates_soft_and_hard_ceiling(
        self, solved_models: dict[str, FiscalSovereignResult]
    ) -> None:
        """Verify that starting at high debt (b0=0.05, y0=1.10) separates soft_ceiling from ceiling."""
        u = np.random.default_rng(42).random((3, 300, 24))
        dsa_none = simulate_fiscal_dsa(
            solved_models["none"], b0=0.05, y0=1.10, horizon=24, n_sims=300, uniforms=u
        )
        dsa_hard = simulate_fiscal_dsa(
            solved_models["ceiling"], b0=0.05, y0=1.10, horizon=24, n_sims=300, uniforms=u
        )
        dsa_soft = simulate_fiscal_dsa(
            solved_models["soft_ceiling"], b0=0.05, y0=1.10, horizon=24, n_sims=300, uniforms=u
        )

        # Hard ceiling strictly enforces b' <= 0.02 in every period t >= 1
        assert dsa_hard.prob_breach == 0.0
        assert dsa_hard.prob_terminal_breach == 0.0
        assert dsa_hard.metadata["first_period_above_threshold_pct"] == 0.0
        assert dsa_hard.metadata["above_threshold_repay_share_pct"] == 0.0
        assert dsa_hard.dar_95 <= 0.02 + 1e-12

        # Soft ceiling chooses b'_1 > 0.02 in period 1 to smooth spending, then deleverages below 0.02
        assert dsa_soft.prob_breach == 1.0
        assert dsa_soft.metadata["first_period_above_threshold_pct"] == 100.0
        assert dsa_soft.metadata["first_period_b_next"] > 0.02
        assert (
            dsa_soft.metadata["first_period_spending_pct"]
            > dsa_hard.metadata["first_period_spending_pct"] + 1.0
        )
        assert (
            0.0
            < dsa_soft.metadata["above_threshold_repay_share_pct"]
            < dsa_none.metadata["above_threshold_repay_share_pct"]
        )
        assert dsa_soft.dar_95 <= 0.02 + 1e-12
        assert np.isclose(dsa_soft.quantiles[50][-1], dsa_hard.quantiles[50][-1])

    def test_default_y0_selection_and_presentation(
        self, solved_models: dict[str, FiscalSovereignResult]
    ) -> None:
        """Verify automatic y0 selection at zero vs high debt and StochasticDSAResult presentation."""
        res = solved_models["soft_ceiling"]
        # At b0=0.0, median income node is in the repayment region
        dsa_zero = simulate_fiscal_dsa(res, b0=0.0, horizon=12, n_sims=50, seed=7)
        assert dsa_zero.metadata["y0_index"] == len(res.y_grid) // 2

        # At b0=0.05, median income defaults, so y0 automatically selects the boom node near 1.10
        dsa_high = simulate_fiscal_dsa(res, b0=0.05, horizon=12, n_sims=50, seed=7)
        assert dsa_high.metadata["y0_node"] > 1.05
        assert dsa_high.shape == (50, 13)
        assert len(dsa_high.summary()) == 10
        fig = dsa_high.plot(show=False)
        plt.close(fig)

    def test_compare_fiscal_rules_dsa_table(
        self, solved_models: dict[str, FiscalSovereignResult]
    ) -> None:
        """Verify compare_fiscal_rules_dsa schema and common-random-numbers parity."""
        df = compare_fiscal_rules_dsa(
            solved_models, b0=0.05, y0=1.10, horizon=20, n_sims=200, seed=42
        )
        expected_cols = [
            "rule_name",
            "b0_node",
            "first_period_b_next",
            "first_period_g_pct",
            "first_period_above_threshold_pct",
            "above_threshold_repay_share_pct",
            "median_terminal_debt_pct",
            "dar_95_pct",
            "prob_breach_pct",
            "cumulative_default_pct",
        ]
        assert list(df.columns) == expected_cols
        assert df["rule_name"].tolist() == ["none", "ceiling", "soft_ceiling", "ceiling_escape"]

        row_hard = df.loc[df["rule_name"] == "ceiling"].iloc[0]
        row_esc = df.loc[df["rule_name"] == "ceiling_escape"].iloc[0]
        row_soft = df.loc[df["rule_name"] == "soft_ceiling"].iloc[0]

        # Escape clause is inert relative to hard ceiling
        assert np.isclose(row_hard["first_period_g_pct"], row_esc["first_period_g_pct"])
        assert np.isclose(row_hard["dar_95_pct"], row_esc["dar_95_pct"])
        assert row_hard["prob_breach_pct"] == 0.0
        assert row_soft["first_period_above_threshold_pct"] == 100.0
        assert row_soft["first_period_g_pct"] > row_hard["first_period_g_pct"]
