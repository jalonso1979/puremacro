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
# kalman_filter: exact-diffuse checked against the kappa -> inf limit,
# the GLS posterior and invariance to the diffuse state's units
# ---------------------------------------------------------------------------

_KAPPA = 1e8


def _gls_period0(model, y0, a0, P0, diffuse):
    """Exact posterior of alpha_1 given y_1 alone, with zero prior precision
    on the diffuse indices (GLS; independent of the Kalman code)."""
    m = model.T.shape[0]
    finite = [i for i in range(m) if i not in diffuse]
    prior_prec = np.zeros((m, m))
    prior_prec[np.ix_(finite, finite)] = np.linalg.inv(P0[np.ix_(finite, finite)])
    Z, Hinv = model.Z, np.linalg.inv(model.H)
    P_post = np.linalg.inv(Z.T @ Hinv @ Z + prior_prec)
    a_post = P_post @ (Z.T @ Hinv @ (y0 - model.d) + prior_prec @ a0)
    return a_post, P_post


def _rescale_state(model, i, s):
    """The same model with state i measured in units s times larger
    (alpha_i' = s alpha_i)."""
    S = np.eye(model.T.shape[0])
    S[i, i] = s
    S_inv = np.linalg.inv(S)
    return StateSpaceModel(T=S @ model.T @ S_inv, Z=model.Z @ S_inv, R=S @ model.R,
                           Q=model.Q, H=model.H, c=S @ model.c, d=model.d)


class TestExactDiffuseNonUnitLoadings:
    """The exact-diffuse filter is the kappa -> inf limit of the proper prior
    P0 = kappa * P_inf + P_*. The P_* update used to scale K0 K0' by
    F_* / F_inf instead of F_*, which is only right when F_inf = 1 (every
    test above loads the diffuse state with Z = 1)."""

    @staticmethod
    def _local_level_z2() -> StateSpaceModel:
        return StateSpaceModel(T=np.array([[1.0]]), Z=np.array([[2.0]]),
                               Q=np.array([[0.5]]), H=np.array([[1.0]]))

    @staticmethod
    def _two_obs_model() -> StateSpaceModel:
        # Both observables load the diffuse random-walk state 0, with
        # weights 2 and -0.7; state 1 is a stationary AR(1) with a proper prior.
        return StateSpaceModel(T=np.diag([1.0, 0.6]),
                               Z=np.array([[2.0, 1.0], [-0.7, 0.5]]),
                               Q=np.diag([0.3, 0.5]), H=np.diag([0.8, 1.3]))

    def test_local_level_first_period_posterior(self):
        """One observation y = 2 alpha + eps under a flat prior: Var = H / Z^2."""
        y = _rng(0).normal(0, 1, 15)
        out = kalman_filter(y, self._local_level_z2(), diffuse_states=[0])
        assert out["P_filt"][0, 0, 0] == pytest.approx(1.0 / 2.0 ** 2, rel=1e-12)
        assert out["a_filt"][0, 0] == pytest.approx(y[0] / 2.0, rel=1e-12)

    def test_local_level_matches_large_P0(self):
        model = self._local_level_z2()
        y = _rng(0).normal(0, 1, 15)
        exact = kalman_filter(y, model, diffuse_states=[0])
        approx = kalman_filter(y, model, P0=np.array([[_KAPPA]]))
        np.testing.assert_allclose(exact["a_filt"][1:], approx["a_filt"][1:], atol=1e-6)
        np.testing.assert_allclose(exact["P_filt"][1:], approx["P_filt"][1:], atol=1e-6)

    def test_local_level_loglik_is_kappa_limit(self):
        """Diffuse loglik = lim_{kappa -> inf} [loglik(kappa) + (1/2) log kappa]."""
        model = self._local_level_z2()
        y = _rng(0).normal(0, 1, 15)
        exact = kalman_filter(y, model, diffuse_states=[0])["loglik"]
        approx = kalman_filter(y, model, P0=np.array([[_KAPPA]]))["loglik"]
        assert exact == pytest.approx(approx + 0.5 * np.log(_KAPPA), abs=1e-6)

    def test_two_observables_first_period_posterior(self):
        """Observations are processed one at a time; the period-0 posterior
        must still be the GLS posterior with zero prior precision on state 0."""
        model = self._two_obs_model()
        y = _rng(1).normal(0, 1, (12, 2))
        a0 = np.array([0.0, 0.4])
        P0 = np.diag([1.0, 0.9])
        out = kalman_filter(y, model, a0=a0, P0=P0, diffuse_states=[0])
        a_post, P_post = _gls_period0(model, y[0], a0, P0, [0])
        np.testing.assert_allclose(out["P_filt"][0], P_post, rtol=1e-10)
        np.testing.assert_allclose(out["a_filt"][0], a_post, rtol=1e-10)

    def test_two_observables_match_large_P0(self):
        model = self._two_obs_model()
        y = _rng(1).normal(0, 1, (12, 2))
        a0 = np.array([0.0, 0.4])
        exact = kalman_smoother(y, model, a0=a0, P0=np.diag([1.0, 0.9]),
                                diffuse_states=[0])
        approx = kalman_smoother(y, model, a0=a0, P0=np.diag([_KAPPA, 0.9]))
        for key in ("a_filt", "P_filt", "a_smooth", "P_smooth"):
            np.testing.assert_allclose(exact[key], approx[key], atol=1e-6, err_msg=key)
        assert exact["loglik"] == pytest.approx(
            approx["loglik"] + 0.5 * np.log(_KAPPA), abs=1e-6)


class TestExactDiffuseEdgeCases:
    """Exact-diffuse filter: correlated measurement errors, fully missing
    periods inside the diffuse phase, and loadings far from unit scale."""

    def test_correlated_measurement_errors(self):
        """The diffuse pass processes a period's observations one at a time,
        which is only valid once their errors are uncorrelated."""
        model = StateSpaceModel(T=np.diag([1.0, 0.6]),
                                Z=np.array([[2.0, 1.0], [-0.7, 0.5]]),
                                Q=np.diag([0.3, 0.5]),
                                H=np.array([[0.8, 0.5], [0.5, 1.3]]))
        y = _rng(2).normal(0, 1, (12, 2))
        a0 = np.array([0.0, 0.4])
        P0 = np.diag([1.0, 0.9])
        exact = kalman_filter(y, model, a0=a0, P0=P0, diffuse_states=[0])
        a_post, P_post = _gls_period0(model, y[0], a0, P0, [0])
        np.testing.assert_allclose(exact["P_filt"][0], P_post, rtol=1e-10)
        np.testing.assert_allclose(exact["a_filt"][0], a_post, rtol=1e-10)
        approx = kalman_filter(y, model, a0=a0, P0=np.diag([_KAPPA, 0.9]))
        np.testing.assert_allclose(exact["a_filt"], approx["a_filt"], atol=1e-6)
        np.testing.assert_allclose(exact["P_filt"], approx["P_filt"], atol=1e-6)
        assert exact["loglik"] == pytest.approx(
            approx["loglik"] + 0.5 * np.log(_KAPPA), abs=1e-6)

    def test_missing_period_moves_diffuse_direction(self):
        """A fully missing period in the diffuse phase still propagates
        P_inf <- T P_inf T'. Here T swaps the states, so at t = 1 the diffuse
        part sits on the unobserved state 1, and y[1] is not a diffuse update."""
        model = StateSpaceModel(T=np.array([[0.0, 1.0], [1.0, 0.0]]),
                                Z=np.array([[1.0, 0.0]]),
                                Q=np.diag([0.4, 0.3]), H=np.array([[0.6]]))
        y = _rng(3).normal(0, 1, 10)
        y[0] = np.nan
        exact = kalman_filter(y, model, a0=np.zeros(2), P0=np.eye(2), diffuse_states=[0])
        approx = kalman_filter(y, model, a0=np.zeros(2), P0=np.diag([_KAPPA, 1.0]))
        # Both states are pinned down from t = 2 on.
        np.testing.assert_allclose(exact["a_filt"][2:], approx["a_filt"][2:], atol=1e-6)
        np.testing.assert_allclose(exact["P_filt"][2:], approx["P_filt"][2:], atol=1e-6)
        assert exact["loglik"] == pytest.approx(
            approx["loglik"] + 0.5 * np.log(_KAPPA), abs=1e-6)

    def test_missing_period_stationary_diffuse_state_loglik(self):
        """A stationary diffuse state shrinks P_inf by T^2 over a missing period,
        which changes the diffuse term -0.5 log F_inf at the next observation."""
        model = StateSpaceModel(T=np.array([[0.5]]), Z=np.array([[1.5]]),
                                Q=np.array([[0.4]]), H=np.array([[0.6]]))
        y = _rng(4).normal(0, 1, 10)
        y[0] = np.nan
        exact = kalman_filter(y, model, diffuse_states=[0])["loglik"]
        approx = kalman_filter(y, model, P0=np.array([[_KAPPA]]))["loglik"]
        assert exact == pytest.approx(approx + 0.5 * np.log(_KAPPA), abs=1e-6)

    @pytest.mark.parametrize("scale", [1e-7, 1e3])
    def test_invariant_to_units_of_diffuse_state(self, scale):
        """Measuring the diffuse state in other units (alpha' = s alpha) maps
        a -> S a, P -> S P S' and adds log|s| to the diffuse loglik. With
        s = 1e-7, F_inf = 1e-14 is a genuine diffuse direction below the old
        absolute 1e-12 cut-off; with s = 1e3, the rounding residual of the
        collapsed P_inf exceeded it and triggered spurious diffuse updates."""
        unit = StateSpaceModel(T=np.diag([1.0, 0.6]),
                               Z=np.array([[1.7, 1.0], [1.1, 0.5]]),
                               Q=np.diag([0.3, 0.5]), H=np.diag([0.8, 1.3]))
        scaled = _rescale_state(unit, 0, 1.0 / scale)    # Z[:, 0] *= scale
        y = _rng(3).normal(0, 1, (10, 2))
        a0, P0 = np.zeros(2), np.eye(2)
        ref = kalman_filter(y, unit, a0=a0, P0=P0, diffuse_states=[0])
        out = kalman_filter(y, scaled, a0=a0, P0=P0, diffuse_states=[0])
        S = np.diag([scale, 1.0])
        np.testing.assert_allclose(out["a_filt"] @ S, ref["a_filt"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(S @ out["P_filt"] @ S, ref["P_filt"], rtol=1e-9, atol=1e-12)
        assert out["loglik"] == pytest.approx(ref["loglik"] - np.log(scale), abs=1e-9)

    def test_nearly_collinear_loadings_resolve_both_diffuse_states(self):
        """Rows [1, 0] and [1, 1e-5] are linearly independent, so both diffuse
        states are pinned down in period 0: F_inf = 1e-10 at the second row
        is a genuine diffuse step, not a rounding residual."""
        unit = StateSpaceModel(T=np.eye(2), Z=np.array([[1.0, 0.0], [1.0, 1.0]]),
                               Q=np.diag([0.3, 0.5]), H=np.diag([0.8, 1.3]))
        scaled = _rescale_state(unit, 1, 1e5)            # Z[:, 1] *= 1e-5
        y = _rng(5).normal(0, 1, (8, 2))
        out = kalman_filter(y, scaled, diffuse_states=[0, 1])
        Z_inv = np.linalg.inv(scaled.Z)
        np.testing.assert_allclose(out["a_filt"][0], Z_inv @ y[0], rtol=1e-9)
        np.testing.assert_allclose(out["P_filt"][0], Z_inv @ scaled.H @ Z_inv.T, rtol=1e-9)
        ref = kalman_filter(y, unit, diffuse_states=[0, 1])
        S = np.diag([1.0, 1e5])
        np.testing.assert_allclose(out["a_filt"], ref["a_filt"] @ S, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(out["P_filt"], S @ ref["P_filt"] @ S, rtol=1e-9, atol=1e-12)
        assert out["loglik"] == pytest.approx(ref["loglik"] + np.log(1e5), abs=1e-9)

    def test_small_diffuse_component_survives_until_observed(self):
        """With T = diag(1, 1e-5) and a missing period, P_inf = diag(1, 1e-10).
        Observing only state 0 resolves that direction and must leave the
        diffuse part of state 1 in place, however small, until state 1 is
        observed; it then has the flat-prior posterior Var = H_11 / Z_11^2."""
        model = StateSpaceModel(T=np.diag([1.0, 1e-5]), Z=np.diag([1.0, 2.0]),
                                Q=np.diag([0.3, 0.5]), H=np.diag([0.8, 1.3]))
        y = _rng(6).normal(0, 1, (8, 2))
        y[0] = np.nan
        y[1, 1] = np.nan
        out = kalman_filter(y, model, diffuse_states=[0, 1])
        assert out["P_filt"][2, 1, 1] == pytest.approx(1.3 / 2.0 ** 2, rel=1e-9)
        assert out["a_filt"][2, 1] == pytest.approx(y[2, 1] / 2.0, rel=1e-9)
        # The states are independent: the bivariate run is two univariate ones.
        univ = [kalman_filter(y[:, i], StateSpaceModel(
                    T=model.T[i:i + 1, i:i + 1], Z=model.Z[i:i + 1, i:i + 1],
                    Q=model.Q[i:i + 1, i:i + 1], H=model.H[i:i + 1, i:i + 1]),
                    diffuse_states=[0]) for i in range(2)]
        for i in range(2):
            np.testing.assert_allclose(out["a_filt"][:, i], univ[i]["a_filt"][:, 0],
                                       rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(out["P_filt"][:, i, i], univ[i]["P_filt"][:, 0, 0],
                                       rtol=1e-12, atol=1e-14)
        assert out["loglik"] == pytest.approx(univ[0]["loglik"] + univ[1]["loglik"], abs=1e-9)


# ---------------------------------------------------------------------------
# kalman_smoother: exact initial smoothing over the diffuse phase, checked
# against the batch GLS posterior of alpha_{1..T}
# ---------------------------------------------------------------------------

def _gls_smoother(model, y, a0, P0, diffuse):
    """Posterior mean and marginal covariances of alpha_{1..T} given y_{1..T},
    with zero prior precision on the diffuse indices of alpha_1. Built in
    information form from the prior, the transitions and the observations
    (needs R Q R' and H invertible; independent of the Kalman code)."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n_t, m = y.shape[0], model.T.shape[0]
    Tm, Z, c, d = model.T, model.Z, model.c, model.d
    W = np.linalg.inv(model.R @ model.Q @ model.R.T)
    Lam = np.zeros((n_t * m, n_t * m))
    b = np.zeros(n_t * m)

    def blk(t):
        return slice(t * m, (t + 1) * m)

    finite = [i for i in range(m) if i not in diffuse]
    prior_prec = np.zeros((m, m))
    prior_prec[np.ix_(finite, finite)] = np.linalg.inv(P0[np.ix_(finite, finite)])
    Lam[blk(0), blk(0)] += prior_prec
    b[blk(0)] += prior_prec @ a0
    for t in range(n_t - 1):
        # alpha_{t+1} - T alpha_t - c ~ N(0, R Q R')
        Lam[blk(t), blk(t)] += Tm.T @ W @ Tm
        Lam[blk(t), blk(t + 1)] -= Tm.T @ W
        Lam[blk(t + 1), blk(t)] -= W @ Tm
        Lam[blk(t + 1), blk(t + 1)] += W
        b[blk(t)] -= Tm.T @ W @ c
        b[blk(t + 1)] += W @ c
    for t in range(n_t):
        obs = ~np.isnan(y[t])
        if not obs.any():
            continue
        Z_o = Z[obs]
        H_inv = np.linalg.inv(model.H[np.ix_(obs, obs)])
        Lam[blk(t), blk(t)] += Z_o.T @ H_inv @ Z_o
        b[blk(t)] += Z_o.T @ H_inv @ (y[t, obs] - d[obs])
    cov = np.linalg.inv(Lam)
    a_post = (cov @ b).reshape(n_t, m)
    P_post = np.stack([cov[blk(t), blk(t)] for t in range(n_t)])
    return a_post, P_post


class TestExactDiffuseSmoother:
    """The filter stores only the finite part P_* in P_pred / P_filt while the
    diffuse phase lasts, so an RTS gain built from them is wrong in every
    period whose filtered P_inf is not yet zero. Those periods must follow the
    exact initial smoother (Durbin and Koopman 2012, §5.3)."""

    @staticmethod
    def _assert_matches_gls(model, y, a0, P0, diffuse):
        out = kalman_smoother(y, model, a0=a0, P0=P0, diffuse_states=diffuse)
        a_post, P_post = _gls_smoother(model, y, a0, P0, diffuse)
        np.testing.assert_allclose(out["a_smooth"], a_post, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(out["P_smooth"], P_post, rtol=1e-9, atol=1e-10)

    @staticmethod
    def _local_linear_trend() -> StateSpaceModel:
        return StateSpaceModel(T=np.array([[1.0, 1.0], [0.0, 1.0]]),
                               Z=np.array([[1.0, 0.0]]),
                               Q=np.diag([0.2, 0.1]), H=np.array([[0.5]]))

    def test_local_linear_trend_both_states_diffuse(self):
        """Level and slope diffuse, one observable: the slope stays diffuse
        after t = 0, where RTS was off by 0.18 in a_smooth."""
        y = np.cumsum(_rng(5).normal(0, 1, 30))
        self._assert_matches_gls(self._local_linear_trend(), y,
                                 np.zeros(2), np.eye(2), [0, 1])

    def test_diffuse_phase_reaches_last_period(self):
        """With y[1] missing the diffuse phase lasts through the last period,
        so the exact recursion starts from r = 0, N = 0."""
        y = np.cumsum(_rng(5).normal(0, 1, 3))
        y[1] = np.nan
        self._assert_matches_gls(self._local_linear_trend(), y,
                                 np.zeros(2), np.eye(2), [0, 1])

    def test_stationary_diffuse_state_missing_first_period(self):
        """AR(0.5) state loaded with Z = 1.5, diffuse, y[0] missing."""
        model = StateSpaceModel(T=np.array([[0.5]]), Z=np.array([[1.5]]),
                                Q=np.array([[0.4]]), H=np.array([[0.6]]))
        y = _rng(4).normal(0, 1, 10)
        y[0] = np.nan
        self._assert_matches_gls(model, y, np.zeros(1), np.eye(1), [0])

    def test_missing_period_moves_diffuse_direction(self):
        """T swaps the states; with y[0] missing the diffuse part passes
        through the unobserved state and is resolved only at t = 2."""
        model = StateSpaceModel(T=np.array([[0.0, 1.0], [1.0, 0.0]]),
                                Z=np.array([[1.0, 0.0]]),
                                Q=np.diag([0.4, 0.3]), H=np.array([[0.6]]))
        y = _rng(3).normal(0, 1, 10)
        y[0] = np.nan
        self._assert_matches_gls(model, y, np.zeros(2), np.eye(2), [0])

    def test_correlated_measurement_errors(self):
        """Non-diagonal H: the diffuse pass works on the eigen-rotated
        observations, and the smoother must use the same rotation. Both
        observables load the level but not the slope, so the slope is still
        diffuse after t = 0; y[1, 1] is missing inside the diffuse phase."""
        model = StateSpaceModel(T=np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.6]]),
                                Z=np.array([[1.0, 0.0, 1.0], [2.0, 0.0, -0.5]]),
                                Q=np.diag([0.2, 0.1, 0.5]),
                                H=np.array([[0.8, 0.5], [0.5, 1.3]]),
                                c=np.array([0.1, 0.0, 0.2]), d=np.array([0.3, -0.4]))
        y = _rng(6).normal(0, 1, (12, 2))
        y[1, 1] = np.nan
        self._assert_matches_gls(model, y, np.array([0.0, 0.0, 0.4]),
                                 np.diag([1.0, 1.0, 0.9]), [0, 1])


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
    """Fixed-interval smoother properties."""

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

    def test_smoother_matches_the_batch_posterior_with_missing_data(self):
        """E[alpha_t | y] and Var[alpha_t | y] from one GLS projection of the
        stacked states on the observed entries, against the backward recursion
        through partially observed and fully missing periods with intercepts."""
        rng = _rng(89)
        T_obs, m = 12, 2
        model = StateSpaceModel(
            T=np.array([[0.8, 0.2], [-0.1, 0.5]]), Z=np.array([[1.0, 0.4], [0.2, 1.0]]),
            R=np.array([[1.0], [0.6]]), Q=np.array([[0.4]]), H=np.diag([0.3, 0.6]),
            c=np.array([0.1, -0.2]), d=np.array([1.0, -0.5]))
        a0, P0 = np.array([0.3, -0.1]), np.array([[1.0, 0.2], [0.2, 0.7]])
        y = rng.normal(0, 1, (T_obs, 2))
        y[2, 0] = y[5, 1] = np.nan
        y[8] = np.nan

        mu, V = [a0], [P0]
        for _ in range(1, T_obs):
            mu.append(model.T @ mu[-1] + model.c)
            V.append(model.T @ V[-1] @ model.T.T + model.R @ model.Q @ model.R.T)
        Sig = np.zeros((T_obs * m, T_obs * m))
        for t in range(T_obs):
            for s in range(t + 1):
                C = np.linalg.matrix_power(model.T, t - s) @ V[s]
                Sig[t*m:(t+1)*m, s*m:(s+1)*m] = C
                Sig[s*m:(s+1)*m, t*m:(t+1)*m] = C.T
        rows = [(t, i) for t in range(T_obs) for i in range(2) if not np.isnan(y[t, i])]
        G = np.zeros((len(rows), T_obs * m))
        Hs = np.zeros((len(rows), len(rows)))
        for q, (t, i) in enumerate(rows):
            G[q, t*m:(t+1)*m] = model.Z[i]
            Hs[q, q] = model.H[i, i]
        resid = np.array([y[t, i] - model.Z[i] @ mu[t] - model.d[i] for t, i in rows])
        gain = np.linalg.solve(G @ Sig @ G.T + Hs, G @ Sig).T
        mean = (np.concatenate(mu) + gain @ resid).reshape(T_obs, m)
        cov = Sig - gain @ G @ Sig

        out = kalman_smoother(y, model, a0=a0, P0=P0)
        np.testing.assert_allclose(out["a_smooth"], mean, rtol=0, atol=1e-12)
        for t in range(T_obs):
            np.testing.assert_allclose(out["P_smooth"][t], cov[t*m:(t+1)*m, t*m:(t+1)*m],
                                       rtol=0, atol=1e-12)

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
