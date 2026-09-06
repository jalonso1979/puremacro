"""Coverage tests for puremacro.state_space.

Targets the branches and functions NOT hit by the existing test suite
(~45% coverage before this file). Specifically covers:

  - StateSpaceModel.__post_init__ defaults (R, c, d = None)
  - kalman_filter: 1-D y auto-expansion, all-missing period (n_obs==0),
    exact-diffuse initialisation (diffuse_states=[...]), Cholesky augment
    fallback, custom a0/P0, intercepts c/d, partial missing obs
  - kalman_smoother: variance reduction, terminal boundary, PSD of P_smooth
  - simulation_smoother: reproducibility, shapes, posterior moment matching
  - Known-value: local-level steady-state Riccati convergence (golden ratio)
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from puremacro.state_space import (
    StateSpaceModel,
    kalman_filter,
    kalman_smoother,
    simulation_smoother,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_level(sigma_eta: float = 1.0, sigma_eps: float = 1.0) -> StateSpaceModel:
    """Simplest univariate local-level model (scalar state)."""
    return StateSpaceModel(
        T=np.array([[1.0]]),
        Z=np.array([[1.0]]),
        Q=np.array([[sigma_eta ** 2]]),
        H=np.array([[sigma_eps ** 2]]),
    )


def _ar1_bivariate() -> StateSpaceModel:
    """2-state, 2-observation model for multivariate tests."""
    return StateSpaceModel(
        T=np.array([[0.9, 0.1], [-0.05, 0.7]]),
        Z=np.array([[1.0, 0.5], [0.3, 1.0]]),
        Q=np.eye(2) * 0.3,
        H=np.eye(2) * 0.5,
    )


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# StateSpaceModel.__post_init__ (lines 69-74)
# ---------------------------------------------------------------------------

class TestStateSpaceModelDefaults:
    """Lines 69-74: R, c, d None-defaults."""

    def test_R_none_becomes_identity(self):
        m = _local_level()
        assert m.R is not None
        assert np.allclose(m.R, np.eye(1))

    def test_c_none_becomes_zeros(self):
        m = _local_level()
        assert m.c is not None
        assert np.allclose(m.c, np.zeros(1))

    def test_d_none_becomes_zeros(self):
        m = _local_level()
        assert m.d is not None
        assert np.allclose(m.d, np.zeros(1))

    def test_defaults_multivariate(self):
        """For 3-state, 2-obs model all defaults must have correct shapes."""
        m = StateSpaceModel(
            T=np.eye(3),
            Z=np.ones((2, 3)),
            Q=np.eye(3),
            H=np.eye(2),
        )
        assert m.R.shape == (3, 3)
        assert np.allclose(m.R, np.eye(3))
        assert m.c.shape == (3,)
        assert m.d.shape == (2,)

    def test_explicit_R_not_overwritten(self):
        """If R is provided, __post_init__ must not replace it."""
        R_given = np.eye(2) * 3.0
        m = StateSpaceModel(
            T=np.eye(2),
            Z=np.ones((1, 2)),
            Q=np.eye(2),
            H=np.ones((1, 1)),
            R=R_given,
            c=np.array([0.1, 0.2]),
            d=np.array([0.5]),
        )
        assert np.allclose(m.R, R_given)
        assert np.allclose(m.c, [0.1, 0.2])
        assert np.allclose(m.d, [0.5])


# ---------------------------------------------------------------------------
# kalman_filter: 1-D y auto-expansion (line 113)
# ---------------------------------------------------------------------------

class TestKalmanFilter1D:
    """1-D y input is silently promoted to (T, 1)."""

    def test_1d_and_2d_give_same_loglik(self):
        model = _local_level()
        y = _rng(1).normal(0, 1, 20)
        out_1d = kalman_filter(y, model)
        out_2d = kalman_filter(y.reshape(-1, 1), model)
        assert np.isclose(out_1d["loglik"], out_2d["loglik"])

    def test_1d_shapes_match_expectation(self):
        model = _local_level()
        y = _rng(2).normal(0, 1, 15)
        out = kalman_filter(y, model)
        assert out["a_filt"].shape == (15, 1)
        assert out["innov"].shape == (15, 1)
        assert out["P_filt"].shape == (15, 1, 1)


# ---------------------------------------------------------------------------
# kalman_filter: all-missing period (lines 170-174)
# ---------------------------------------------------------------------------

class TestKalmanFilterAllMissing:
    """n_obs == 0 branch: a_filt[t] = a_pred[t], no loglik contribution."""

    def test_all_nan_period_sets_innov_nan(self):
        model = _ar1_bivariate()
        y = _rng(10).normal(0, 1, (20, 2))
        y[5] = np.nan    # fully missing
        out = kalman_filter(y, model)
        assert np.all(np.isnan(out["innov"][5]))

    def test_all_nan_period_a_filt_equals_a_pred(self):
        """When t is fully missing, a_filt[t] == a_pred[t] (no update)."""
        model = _ar1_bivariate()
        y = _rng(11).normal(0, 1, (20, 2))
        y[7] = np.nan
        out = kalman_filter(y, model)
        assert np.allclose(out["a_filt"][7], out["a_pred"][7])

    def test_all_nan_period_P_filt_equals_P_pred(self):
        model = _ar1_bivariate()
        y = _rng(12).normal(0, 1, (20, 2))
        y[3] = np.nan
        out = kalman_filter(y, model)
        assert np.allclose(out["P_filt"][3], out["P_pred"][3])

    def test_prediction_through_missing_period(self):
        """a_pred[t+1] = T @ a_filt[t] + c when t is fully missing."""
        model = StateSpaceModel(
            T=np.array([[0.9, 0.0], [0.0, 0.8]]),
            Z=np.eye(2),
            Q=np.eye(2) * 0.5,
            H=np.eye(2) * 0.5,
        )
        y = _rng(13).normal(0, 1, (20, 2))
        y[5] = np.nan
        out = kalman_filter(y, model)
        expected_a_pred_6 = model.T @ out["a_filt"][5] + model.c
        assert np.allclose(out["a_pred"][6], expected_a_pred_6)

    def test_multiple_consecutive_missing_periods(self):
        model = _local_level()
        y = _rng(14).normal(0, 1, 20)
        y[4] = np.nan
        y[5] = np.nan
        y[6] = np.nan
        out = kalman_filter(y, model)
        assert np.isfinite(out["loglik"])
        for t in (4, 5, 6):
            assert np.all(np.isnan(out["innov"][t]))

    def test_univariate_partial_missing(self):
        """For a 2-obs model, one obs missing per period still updates."""
        model = StateSpaceModel(
            T=np.array([[0.9]]),
            Z=np.array([[1.0], [0.8]]),
            Q=np.array([[0.5]]),
            H=np.eye(2) * 0.5,
        )
        y = _rng(77).normal(0, 1, (20, 2))
        y[3, 0] = np.nan   # first obs missing, second present
        y[7, 1] = np.nan   # second obs missing, first present
        y[11] = np.nan     # both missing
        out = kalman_filter(y, model)
        assert np.isfinite(out["loglik"])
        assert np.isnan(out["innov"][3, 0])
        assert np.isfinite(out["innov"][3, 1])
        assert np.isfinite(out["innov"][7, 0])
        assert np.isnan(out["innov"][7, 1])
        assert np.all(np.isnan(out["innov"][11]))


# ---------------------------------------------------------------------------
# kalman_filter: exact-diffuse initialisation (lines 137-148, 182-225)
# ---------------------------------------------------------------------------

class TestKalmanFilterDiffuse:
    """diffuse_states=[...] branch — exact Koopman-Durbin diffuse init."""

    def test_diffuse_loglik_is_finite(self):
        model = _local_level()
        y = _rng(20).normal(0, 1, 20)
        out = kalman_filter(y, model, diffuse_states=[0])
        assert np.isfinite(out["loglik"])

    def test_diffuse_shapes_unchanged(self):
        model = _local_level()
        y = _rng(21).normal(0, 1, 15)
        out = kalman_filter(y, model, diffuse_states=[0])
        assert out["a_filt"].shape == (15, 1)
        assert out["P_filt"].shape == (15, 1, 1)
        assert out["innov"].shape == (15, 1)

    def test_diffuse_and_non_diffuse_finite(self):
        """Both standard and diffuse initialisation return finite loglik."""
        model = _local_level()
        y = _rng(22).normal(0, 1, 15)
        ll_std = kalman_filter(y, model)["loglik"]
        ll_diff = kalman_filter(y, model, diffuse_states=[0])["loglik"]
        assert np.isfinite(ll_std)
        assert np.isfinite(ll_diff)

    def test_diffuse_two_states(self):
        """Both states diffuse should still give a finite loglik."""
        model = StateSpaceModel(
            T=np.array([[1.0, 0.0], [0.0, 0.8]]),
            Z=np.array([[1.0, 1.0]]),
            Q=np.array([[0.1, 0.0], [0.0, 0.3]]),
            H=np.array([[0.5]]),
        )
        y = _rng(23).normal(0, 2, 20)
        out = kalman_filter(y, model, diffuse_states=[0, 1])
        assert np.isfinite(out["loglik"])
        assert out["a_filt"].shape == (20, 2)

    def test_diffuse_single_state_non_diagonal(self):
        """One diffuse, one non-diffuse state in 2-state model."""
        model = StateSpaceModel(
            T=np.array([[1.0, 0.0], [0.0, 0.8]]),
            Z=np.array([[1.0, 1.0]]),
            Q=np.array([[0.1, 0.0], [0.0, 0.3]]),
            H=np.array([[0.5]]),
        )
        y = _rng(24).normal(0, 2, 20)
        out = kalman_filter(y, model, diffuse_states=[0])
        assert np.isfinite(out["loglik"])

    def test_P_inf_does_not_infect_finite_states(self):
        """After diffuse burnin, P_filt diagonals should be finite (not huge)."""
        model = _local_level()
        y = _rng(25).normal(0, 1, 30)
        out = kalman_filter(y, model, diffuse_states=[0])
        # After the first few observations the diffuse component collapses.
        # The late-period P_filt must be much smaller than the initial diffuse scale.
        late_P = out["P_filt"][20:, 0, 0]
        assert (late_P < 1e4).all(), "P_filt did not collapse after diffuse burnin"


# ---------------------------------------------------------------------------
# kalman_filter: Cholesky augmentation fallback (lines 231-236)
# ---------------------------------------------------------------------------

class TestKalmanFilterCholeskyFallback:
    """Lines 231-236: np.linalg.cholesky raises LinAlgError -> augment and retry."""

    def test_fallback_still_produces_finite_loglik(self):
        """Patch np.linalg.cholesky to raise on the first call; filter must recover."""
        model = _local_level()
        y = _rng(30).normal(0, 1, 10)

        orig = np.linalg.cholesky
        call_count = [0]

        def mock_chol(A):
            call_count[0] += 1
            if call_count[0] == 1:
                raise np.linalg.LinAlgError("injected failure")
            return orig(A)

        with patch("numpy.linalg.cholesky", mock_chol):
            out = kalman_filter(y, model)

        assert np.isfinite(out["loglik"])
        assert call_count[0] > 1  # fallback was reached


# ---------------------------------------------------------------------------
# kalman_filter: custom initial conditions (a0, P0)
# ---------------------------------------------------------------------------

class TestKalmanFilterInitialConditions:
    """Custom a0 / P0 propagate correctly."""

    def test_custom_a0_appears_in_a_pred0(self):
        model = _local_level()
        y = _rng(40).normal(0, 1, 20)
        a0 = np.array([5.0])
        out = kalman_filter(y, model, a0=a0)
        assert np.allclose(out["a_pred"][0], a0)

    def test_custom_P0_appears_in_P_pred0(self):
        model = _local_level()
        y = _rng(41).normal(0, 1, 20)
        P0 = np.array([[0.25]])
        out = kalman_filter(y, model, P0=P0)
        assert np.allclose(out["P_pred"][0], P0)

    def test_informative_P0_changes_loglik(self):
        model = _local_level()
        y = _rng(42).normal(0, 1, 20)
        ll_diffuse = kalman_filter(y, model)["loglik"]
        ll_info = kalman_filter(y, model, P0=np.array([[0.1]]))["loglik"]
        # They must differ (different initialisation -> different likelihood)
        assert not np.isclose(ll_diffuse, ll_info)


# ---------------------------------------------------------------------------
# kalman_filter: state/measurement intercepts (c, d)
# ---------------------------------------------------------------------------

class TestKalmanFilterIntercepts:
    """State intercept c and measurement intercept d."""

    def test_state_intercept_propagates(self):
        """a_pred[1] - a_filt[0] must equal c for a model with K=0 at t=0."""
        # For a model with very small Q (nearly K=0), the predicted state
        # approximately equals T @ a_filt + c.
        model = StateSpaceModel(
            T=np.array([[1.0]]),
            Z=np.array([[1.0]]),
            Q=np.array([[1e-6]]),
            H=np.array([[1000.0]]),  # large H -> tiny K
            c=np.array([0.5]),
        )
        y = _rng(50).normal(5, 1, 20)
        out = kalman_filter(y, model, a0=np.array([0.0]), P0=np.array([[0.0]]))
        # With P0=0 the Kalman gain K≈0, so a_pred[1] = T @ a_filt[0] + c exactly.
        expected = model.T @ out["a_filt"][0] + model.c
        assert np.allclose(out["a_pred"][1], expected, atol=1e-6)

    def test_measurement_intercept_shifts_innovation(self):
        """Innovation = y - Z @ a_pred - d. A nonzero d shifts the innovation."""
        d_val = np.array([3.0])
        model_no_d = StateSpaceModel(
            T=np.array([[0.9]]),
            Z=np.array([[1.0]]),
            Q=np.array([[0.5]]),
            H=np.array([[1.0]]),
            d=np.zeros(1),
        )
        model_d = StateSpaceModel(
            T=np.array([[0.9]]),
            Z=np.array([[1.0]]),
            Q=np.array([[0.5]]),
            H=np.array([[1.0]]),
            d=d_val,
        )
        # y shifted by d so innovations should be the same
        y_base = _rng(51).normal(0, 1, 20)
        y_shifted = y_base + d_val[0]
        out_base = kalman_filter(y_base, model_no_d)
        out_shifted = kalman_filter(y_shifted, model_d)
        assert np.allclose(
            out_base["innov"], out_shifted["innov"], atol=1e-8
        ), "Measurement intercept d must shift data, leaving innovations unchanged."


# ---------------------------------------------------------------------------
# kalman_filter: known-value (golden-ratio steady state)
# ---------------------------------------------------------------------------

class TestKalmanFilterKnownValues:
    """Local-level model: steady-state Riccati solution is the golden ratio."""

    def test_P_pred_converges_to_golden_ratio(self):
        """P* = (1 + sqrt(5)) / 2 for sigma_eta = sigma_eps = 1 (golden ratio)."""
        model = _local_level(sigma_eta=1.0, sigma_eps=1.0)
        y = _rng(60).normal(0, 2, 300)
        out = kalman_filter(y, model)
        P_ss = (1 + np.sqrt(5)) / 2
        P_pred_late = out["P_pred"][-10:, 0, 0]
        assert np.allclose(P_pred_late, P_ss, atol=1e-6), (
            f"Predicted variance did not converge to golden ratio {P_ss:.6f}; "
            f"got {P_pred_late.mean():.6f}"
        )

    def test_K_pred_converges_to_golden_ratio_minus_one(self):
        """Steady-state Kalman gain K* = (sqrt(5)-1)/2 ≈ 0.618."""
        model = _local_level(sigma_eta=1.0, sigma_eps=1.0)
        y = _rng(61).normal(0, 2, 300)
        out = kalman_filter(y, model)
        K_ss = (np.sqrt(5) - 1) / 2
        K_late = out["K"][-10:, 0, 0]
        assert np.allclose(K_late, K_ss, atol=1e-6)

    def test_loglik_is_finite_and_negative(self):
        model = _local_level()
        y = _rng(62).normal(0, 1, 20)
        out = kalman_filter(y, model)
        assert np.isfinite(out["loglik"])
        assert out["loglik"] < 0.0

    def test_innovations_approximately_mean_zero(self):
        """For a correctly specified model on a long sample, E[v_t] ≈ 0."""
        model = _local_level(sigma_eta=0.5, sigma_eps=1.0)
        rng = _rng(63)
        T_obs = 500
        alpha = np.cumsum(rng.normal(0, 0.5, T_obs))
        y = alpha + rng.normal(0, 1.0, T_obs)
        out = kalman_filter(y, model)
        innov = out["innov"][:, 0]
        assert np.isfinite(innov).all()
        assert abs(innov.mean()) < 0.15  # sample mean should be near 0


# ---------------------------------------------------------------------------
# kalman_filter: output structure
# ---------------------------------------------------------------------------

class TestKalmanFilterOutputStructure:
    """Shapes and keys of the returned dict."""

    def test_output_keys(self):
        model = _local_level()
        y = _rng(70).normal(0, 1, 10)
        out = kalman_filter(y, model)
        assert set(out.keys()) == {
            "a_pred", "P_pred", "a_filt", "P_filt",
            "innov", "F", "K", "loglik"
        }

    def test_a_pred_shape(self):
        T_obs, m = 15, 2
        model = _ar1_bivariate()
        y = _rng(71).normal(0, 1, (T_obs, 2))
        out = kalman_filter(y, model)
        # a_pred has T+1 rows (includes t=0 prior)
        assert out["a_pred"].shape == (T_obs + 1, m)

    def test_F_and_K_shapes(self):
        T_obs, n, m = 12, 2, 2
        model = _ar1_bivariate()
        y = _rng(72).normal(0, 1, (T_obs, n))
        out = kalman_filter(y, model)
        assert out["F"].shape == (T_obs, n, n)
        assert out["K"].shape == (T_obs, m, n)

    def test_P_pred_first_equals_P0(self):
        model = _local_level()
        P0 = np.array([[2.5]])
        y = _rng(73).normal(0, 1, 10)
        out = kalman_filter(y, model, P0=P0)
        assert np.allclose(out["P_pred"][0], P0)


# ---------------------------------------------------------------------------
# kalman_smoother (lines 284-310)
# ---------------------------------------------------------------------------

class TestKalmanSmoother:
    """RTS smoother properties."""

    def test_smoother_keys(self):
        model = _local_level()
        y = _rng(80).normal(0, 1, 10)
        out = kalman_smoother(y, model)
        assert "a_smooth" in out
        assert "P_smooth" in out

    def test_smoother_shapes(self):
        T_obs, m = 20, 2
        model = _ar1_bivariate()
        y = _rng(81).normal(0, 1, (T_obs, 2))
        out = kalman_smoother(y, model)
        assert out["a_smooth"].shape == (T_obs, m)
        assert out["P_smooth"].shape == (T_obs, m, m)

    def test_terminal_boundary_condition(self):
        """At t = T-1, smoother == filter (no future information)."""
        model = _local_level()
        y = _rng(82).normal(0, 1, 20)
        out = kalman_smoother(y, model)
        assert np.allclose(out["a_smooth"][-1], out["a_filt"][-1])
        assert np.allclose(out["P_smooth"][-1], out["P_filt"][-1])

    def test_smoother_reduces_variance(self):
        """P_smooth[t, i, i] <= P_filt[t, i, i] for all t, i (uncertainty shrinks)."""
        model = _ar1_bivariate()
        y = _rng(83).normal(0, 1, (30, 2))
        out = kalman_smoother(y, model)
        for i in range(2):
            filt_diag = out["P_filt"][:, i, i]
            sm_diag = out["P_smooth"][:, i, i]
            assert (sm_diag <= filt_diag + 1e-10).all(), (
                f"P_smooth[{i},{i}] exceeds P_filt[{i},{i}] somewhere"
            )

    def test_P_smooth_is_PSD_everywhere(self):
        """Smoothed covariance must be positive semi-definite at every t."""
        model = _ar1_bivariate()
        y = _rng(84).normal(0, 1, (25, 2))
        out = kalman_smoother(y, model)
        for t in range(25):
            eigs = np.linalg.eigvalsh(out["P_smooth"][t])
            assert (eigs >= -1e-10).all(), f"P_smooth[{t}] not PSD (min eig={eigs.min():.2e})"

    def test_P_filt_is_PSD_everywhere(self):
        model = _ar1_bivariate()
        y = _rng(85).normal(0, 1, (25, 2))
        out = kalman_smoother(y, model)
        for t in range(25):
            eigs = np.linalg.eigvalsh(out["P_filt"][t])
            assert (eigs >= -1e-10).all(), f"P_filt[{t}] not PSD (min eig={eigs.min():.2e})"

    def test_smoother_loglik_equals_filter_loglik(self):
        """kalman_smoother augments the filter dict; loglik must be unchanged."""
        model = _local_level()
        y = _rng(86).normal(0, 1, 20)
        ll_f = kalman_filter(y, model)["loglik"]
        ll_s = kalman_smoother(y, model)["loglik"]
        assert np.isclose(ll_f, ll_s)

    def test_smoother_with_missing_observations(self):
        """Smoother must handle NaN in y gracefully."""
        model = _local_level()
        y = _rng(87).normal(0, 1, 20)
        y[5] = np.nan
        y[10] = np.nan
        out = kalman_smoother(y, model)
        assert np.isfinite(out["loglik"])
        assert np.isfinite(out["a_smooth"]).all()

    def test_smoother_with_diffuse_states(self):
        """Smoother accepts diffuse_states and returns finite outputs."""
        model = _local_level()
        y = _rng(88).normal(0, 1, 20)
        out = kalman_smoother(y, model, diffuse_states=[0])
        assert np.isfinite(out["loglik"])
        assert np.isfinite(out["a_smooth"]).all()
        assert out["a_smooth"].shape == (20, 1)


# ---------------------------------------------------------------------------
# simulation_smoother (lines 331-370)
# ---------------------------------------------------------------------------

class TestSimulationSmoother:
    """Durbin-Koopman simulation smoother."""

    def test_output_shape_univariate(self):
        model = _local_level()
        y = _rng(90).normal(0, 1, 15)
        draw = simulation_smoother(y, model, rng=_rng(0))
        assert draw.shape == (15, 1)

    def test_output_shape_multivariate(self):
        model = _ar1_bivariate()
        y = _rng(91).normal(0, 1, (20, 2))
        draw = simulation_smoother(y, model, rng=_rng(1))
        assert draw.shape == (20, 2)

    def test_reproducibility_same_seed(self):
        """Same RNG seed must produce identical draws."""
        model = _local_level()
        y = _rng(92).normal(0, 1, 12)
        draw1 = simulation_smoother(y, model, rng=np.random.default_rng(7))
        draw2 = simulation_smoother(y, model, rng=np.random.default_rng(7))
        assert np.allclose(draw1, draw2)

    def test_different_seed_produces_different_draw(self):
        model = _local_level()
        y = _rng(93).normal(0, 1, 12)
        draw1 = simulation_smoother(y, model, rng=np.random.default_rng(7))
        draw2 = simulation_smoother(y, model, rng=np.random.default_rng(99))
        assert not np.allclose(draw1, draw2)

    def test_default_rng_runs_without_error(self):
        """rng=None should use a fresh default_rng and not crash."""
        model = _local_level()
        y = _rng(94).normal(0, 1, 10)
        draw = simulation_smoother(y, model)  # no rng arg
        assert draw.shape == (10, 1)
        assert np.isfinite(draw).all()

    def test_all_draws_finite(self):
        model = _local_level()
        y = _rng(95).normal(0, 1, 20)
        for seed in range(5):
            draw = simulation_smoother(y, model, rng=np.random.default_rng(seed))
            assert np.isfinite(draw).all(), f"Non-finite draw at seed {seed}"

    def test_1d_y_input(self):
        """Simulation smoother accepts 1-D y just like the filter."""
        model = _local_level()
        y_1d = _rng(96).normal(0, 1, 10)
        y_2d = y_1d.reshape(-1, 1)
        draw_1d = simulation_smoother(y_1d, model, rng=np.random.default_rng(3))
        draw_2d = simulation_smoother(y_2d, model, rng=np.random.default_rng(3))
        assert np.allclose(draw_1d, draw_2d)

    def test_custom_a0_P0(self):
        """Custom a0 / P0 must be accepted without error."""
        model = _local_level()
        y = _rng(97).normal(0, 1, 15)
        draw = simulation_smoother(
            y, model,
            rng=np.random.default_rng(5),
            a0=np.array([2.0]),
            P0=np.array([[0.1]]),
        )
        assert draw.shape == (15, 1)
        assert np.isfinite(draw).all()

    @pytest.mark.slow
    def test_posterior_mean_close_to_smoothed_mean(self):
        """Monte Carlo mean of simulation-smoother draws converges to a_smooth."""
        model = _local_level()
        rng = _rng(98)
        T_obs = 40
        alpha_true = np.cumsum(rng.normal(0, 1, T_obs))
        y = alpha_true + rng.normal(0, 1, T_obs)

        sm_out = kalman_smoother(y, model)
        a_sm = sm_out["a_smooth"].flatten()

        N = 800
        draws = np.stack([
            simulation_smoother(y, model, rng=np.random.default_rng(s)).flatten()
            for s in range(N)
        ])
        sample_mean = draws.mean(axis=0)
        # The MC mean should approximate the posterior mean within simulation error.
        # With N=800, atol=0.15 is a conservative bound (std of mean ~ P^0.5 / sqrt(N)).
        assert np.abs(sample_mean - a_sm).max() < 0.15, (
            "Simulation smoother MC mean deviates too much from a_smooth"
        )


# ---------------------------------------------------------------------------
# __all__ membership
# ---------------------------------------------------------------------------

def test_all_exports():
    from puremacro import state_space
    assert "StateSpaceModel" in state_space.__all__
    assert "kalman_filter" in state_space.__all__
    assert "kalman_smoother" in state_space.__all__
    assert "simulation_smoother" in state_space.__all__


# ---------------------------------------------------------------------------
# System-matrix validation (v2.3.x regression): time-invariant 2-D only
# ---------------------------------------------------------------------------

class TestSystemMatrixValidation:
    """The module docstring used to claim 3-D ``(T, ...)`` time-varying
    matrices were accepted; they crashed deep inside the recursions
    (``ValueError: Input must be 1- or 2-d`` / matmul mismatch). The
    docstring now says time-invariant only and the container raises a
    clear error up front."""

    def test_time_varying_Z_raises_clear_error(self):
        with pytest.raises(ValueError, match="time-invariant"):
            StateSpaceModel(T=np.eye(1), Z=np.ones((200, 1, 1)),
                            Q=np.array([[0.09]]), H=np.array([[0.49]]))

    def test_time_varying_T_raises_clear_error(self):
        with pytest.raises(ValueError, match="time-invariant"):
            StateSpaceModel(T=np.tile(np.eye(1), (200, 1, 1)), Z=np.eye(1),
                            Q=np.array([[0.09]]), H=np.array([[0.49]]))

    def test_inconsistent_shapes_raise(self):
        with pytest.raises(ValueError, match="Z must be"):
            StateSpaceModel(T=np.eye(2), Z=np.ones((1, 3)), Q=np.eye(2), H=np.eye(1))
        with pytest.raises(ValueError, match="H must be"):
            StateSpaceModel(T=np.eye(2), Z=np.ones((1, 2)), Q=np.eye(2), H=np.eye(2))
        with pytest.raises(ValueError, match="Q must be"):
            StateSpaceModel(T=np.eye(2), Z=np.ones((1, 2)), Q=np.eye(1), H=np.eye(1))
        with pytest.raises(ValueError, match="R must be"):
            StateSpaceModel(T=np.eye(2), Z=np.ones((1, 2)), Q=np.eye(1), H=np.eye(1),
                            R=np.ones((3, 1)))
        with pytest.raises(ValueError, match="c must be"):
            StateSpaceModel(T=np.eye(2), Z=np.ones((1, 2)), Q=np.eye(2), H=np.eye(1),
                            c=np.zeros(3))

    def test_module_docstring_no_longer_claims_time_varying_support(self):
        import puremacro.state_space as ss
        assert "time-invariant" in ss.__doc__
        assert "not supported" in ss.__doc__
        assert "(time-varying)" not in ss.__doc__

    def test_valid_2d_model_with_selection_matrix_still_works(self):
        m = StateSpaceModel(T=np.array([[1.2, -0.4], [1.0, 0.0]]), Z=np.array([[1.0, 0.0]]),
                            Q=np.array([[1.0]]), H=np.array([[0.3]]), R=np.array([[1.0], [0.0]]))
        out = kalman_filter(np.zeros(10), m)
        assert np.isfinite(out["loglik"])
