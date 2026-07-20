"""Coverage tests for puremacro.inference.bootstrap.

Module under test
-----------------
    _ols_var(Y, p) -> (A_list, c, Sigma, resid, X)
    _irf_from_var(A_list, horizon, impact) -> ndarray (n, n, horizon+1)
    residual_bootstrap(residuals, Y, A_list, intercept, ...) -> dict

Coverage strategy
-----------------
All three functions are currently at 0% coverage (lines 13-108 missing).

_ols_var
  - Shape contract: A_list has p elements each (n,n); c shape (n,); Sigma
    shape (n,n); resid shape (T-p, n); X shape (T-p, 1+n*p).
  - OLS identity: c and A_list reproduce the lstsq fit exactly.
  - Residuals sum near zero (OLS with intercept property).
  - Sigma is symmetric.
  - VAR(1) and VAR(2) cases.
  - Known-value: AR(1) DGP, estimated coefficient near truth.

_irf_from_var
  - h=0 always returns impact matrix (Phi[0] = I => ir[:,0] = I @ impact = impact).
  - Zero A_list: IRF is zero for all h>0 (no dynamics).
  - Scalar AR(1) rho: IRF at horizon h equals rho^h * impact.
  - VAR(2) known recursion at h=2: Phi[2] = A1*A1 + A2.
  - Shape (n, n_impact, horizon+1) for multivariate impact.

residual_bootstrap
  - Output is a dict with key "draws".
  - Draws list has exactly n_draws elements.
  - Each draw has shape (horizon+1, n, n) when irf_fn returns (H+1,n,n).
  - irf_fn=None branch: each draw is zeros((horizon+1, n, n)).
  - irf_fn is called with (Y_star, p, horizon) and exactly n_draws times.
  - Y_star passed to irf_fn has shape (T, n) and initial rows match Y[:p].
  - rng=None branch: runs without error, correct structure.
  - Reproducibility: same seed => identical draws.
  - Different seeds => different Y_star simulations.
  - rng argument is consumed correctly (seed propagates).
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.inference.bootstrap import (
    _ols_var,
    _irf_from_var,
    residual_bootstrap,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_var_data(T: int, n: int, p: int, seed: int = 0) -> np.ndarray:
    """White-noise VAR data (T x n)."""
    return np.random.default_rng(seed).standard_normal((T, n))


def _make_ar1_data(T: int, rho: float, seed: int = 0) -> np.ndarray:
    """Univariate AR(1): y_t = rho*y_{t-1} + e_t."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T)
    y = np.zeros((T, 1))
    for t in range(1, T):
        y[t, 0] = rho * y[t - 1, 0] + e[t]
    return y


# ---------------------------------------------------------------------------
# Tests for _ols_var
# ---------------------------------------------------------------------------

class TestOlsVar:
    """Tests for _ols_var(Y, p) -> (A_list, c, Sigma, resid, X)."""

    # --- Shape contracts -------------------------------------------------------

    def test_var1_shapes_n2(self):
        """VAR(1), n=2: verify shapes of all five returned objects."""
        T, n, p = 60, 2, 1
        Y = _make_var_data(T, n, p=p, seed=1)
        A_list, c, Sigma, resid, X = _ols_var(Y, p)
        assert len(A_list) == p
        assert A_list[0].shape == (n, n)
        assert c.shape == (n,)
        assert Sigma.shape == (n, n)
        assert resid.shape == (T - p, n)
        assert X.shape == (T - p, 1 + n * p)

    def test_var2_shapes_n3(self):
        """VAR(2), n=3: A_list has 2 elements each (3,3); X has 1+3*2=7 cols."""
        T, n, p = 80, 3, 2
        Y = _make_var_data(T, n, p=p, seed=2)
        A_list, c, Sigma, resid, X = _ols_var(Y, p)
        assert len(A_list) == p
        for Ai in A_list:
            assert Ai.shape == (n, n)
        assert c.shape == (n,)
        assert Sigma.shape == (n, n)
        assert resid.shape == (T - p, n)
        assert X.shape == (T - p, 1 + n * p)

    def test_var1_shapes_n1(self):
        """Univariate VAR(1): all shapes use n=1."""
        T, n, p = 40, 1, 1
        Y = _make_ar1_data(T, rho=0.5, seed=3)
        A_list, c, Sigma, resid, X = _ols_var(Y, p)
        assert len(A_list) == 1
        assert A_list[0].shape == (1, 1)
        assert c.shape == (1,)
        assert Sigma.shape == (1, 1)
        assert resid.shape == (T - p, 1)
        assert X.shape == (T - p, 2)  # 1 const + 1*p = 2

    # --- OLS identity ----------------------------------------------------------

    def test_coefficients_match_lstsq(self):
        """A_list and c must reproduce the direct lstsq solution."""
        T, n, p = 80, 3, 2
        Y = _make_var_data(T, n, p=p, seed=4)
        A_list, c, Sigma, resid, X = _ols_var(Y, p)
        # Direct lstsq
        B = np.linalg.lstsq(X, Y[p:], rcond=None)[0]
        c_ref = B[0]
        A1_ref = B[1 : 1 + n].T
        A2_ref = B[1 + n : 1 + 2 * n].T
        np.testing.assert_allclose(c, c_ref, atol=1e-10)
        np.testing.assert_allclose(A_list[0], A1_ref, atol=1e-10)
        np.testing.assert_allclose(A_list[1], A2_ref, atol=1e-10)

    def test_residuals_match_lstsq(self):
        """resid must equal Y[p:] - X @ B exactly."""
        T, n, p = 60, 2, 1
        Y = _make_var_data(T, n, p=p, seed=5)
        A_list, c, Sigma, resid, X = _ols_var(Y, p)
        B = np.linalg.lstsq(X, Y[p:], rcond=None)[0]
        resid_ref = Y[p:] - X @ B
        np.testing.assert_allclose(resid, resid_ref, atol=1e-10)

    # --- OLS properties --------------------------------------------------------

    def test_residuals_sum_near_zero(self):
        """OLS with intercept: residuals sum to near zero in each equation."""
        T, n, p = 100, 2, 1
        Y = _make_var_data(T, n, p=p, seed=6)
        A_list, c, Sigma, resid, X = _ols_var(Y, p)
        np.testing.assert_allclose(resid.sum(axis=0), 0.0, atol=1e-10)

    def test_sigma_is_symmetric(self):
        """Sigma = resid.T @ resid / dof must be symmetric."""
        T, n, p = 80, 3, 2
        Y = _make_var_data(T, n, p=p, seed=7)
        A_list, c, Sigma, resid, X = _ols_var(Y, p)
        np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)

    def test_sigma_formula(self):
        """Sigma = resid.T @ resid / (T - p - 1 - n*p)."""
        T, n, p = 70, 2, 1
        Y = _make_var_data(T, n, p=p, seed=8)
        A_list, c, Sigma, resid, X = _ols_var(Y, p)
        dof = T - p - 1 - n * p
        Sigma_ref = resid.T @ resid / dof
        np.testing.assert_allclose(Sigma, Sigma_ref, atol=1e-12)

    def test_x_first_col_is_ones(self):
        """First column of X must be all ones (the constant term)."""
        T, n, p = 60, 2, 1
        Y = _make_var_data(T, n, p=p, seed=9)
        A_list, c, Sigma, resid, X = _ols_var(Y, p)
        np.testing.assert_allclose(X[:, 0], 1.0, atol=1e-12)

    # --- Known-value: AR(1) recovery ------------------------------------------

    def test_ar1_coefficient_recovery(self):
        """Univariate AR(1) with rho=0.7, T=400: estimated coefficient near 0.7."""
        rho_true = 0.7
        T = 400
        Y = _make_ar1_data(T, rho=rho_true, seed=10)
        A_list, c, Sigma, resid, X = _ols_var(Y, p=1)
        rho_hat = A_list[0][0, 0]
        assert abs(rho_hat - rho_true) < 0.05, (
            f"Estimated rho={rho_hat:.4f}, expected near {rho_true}"
        )

    def test_zero_dynamics_gives_small_coefficients(self):
        """White-noise VAR: A estimates should be near zero with large T."""
        T, n, p = 300, 2, 1
        Y = _make_var_data(T, n, p=p, seed=11)
        A_list, c, Sigma, resid, X = _ols_var(Y, p)
        assert np.max(np.abs(A_list[0])) < 0.25, (
            f"White-noise A estimate too large: {A_list[0]}"
        )


# ---------------------------------------------------------------------------
# Tests for _irf_from_var
# ---------------------------------------------------------------------------

class TestIrfFromVar:
    """Tests for _irf_from_var(A_list, horizon, impact) -> (n, n_impact, H+1)."""

    # --- Shape ----------------------------------------------------------------

    def test_shape_square_impact(self):
        """Square impact (n x n): output shape is (n, n, horizon+1)."""
        n = 2
        A_list = [np.zeros((n, n))]
        impact = np.eye(n)
        ir = _irf_from_var(A_list, horizon=4, impact=impact)
        assert ir.shape == (n, n, 5)

    def test_shape_non_square_impact(self):
        """Impact with k columns: output shape is (n, k, horizon+1)."""
        n = 3
        k = 2
        A_list = [np.zeros((n, n))]
        impact = np.random.default_rng(20).standard_normal((n, k))
        ir = _irf_from_var(A_list, horizon=3, impact=impact)
        assert ir.shape == (n, k, 4)

    def test_horizon_zero_shape(self):
        """horizon=0: output has 1 time point, shape (n, n, 1)."""
        n = 2
        A_list = [np.zeros((n, n))]
        impact = np.eye(n)
        ir = _irf_from_var(A_list, horizon=0, impact=impact)
        assert ir.shape == (n, n, 1)

    # --- Known value: h=0 always equals impact --------------------------------

    def test_h0_equals_impact_eye(self):
        """Phi[0] = I => ir[:,:,0] = I @ impact = impact."""
        n = 3
        A_list = [np.random.default_rng(21).standard_normal((n, n)) * 0.1]
        impact = np.eye(n)
        ir = _irf_from_var(A_list, horizon=5, impact=impact)
        np.testing.assert_allclose(ir[:, :, 0], impact, atol=1e-12)

    def test_h0_equals_impact_general(self):
        """For a general impact matrix, ir[:,:,0] == impact exactly."""
        n = 2
        rng = np.random.default_rng(22)
        A_list = [rng.standard_normal((n, n)) * 0.1]
        impact = rng.standard_normal((n, n))
        ir = _irf_from_var(A_list, horizon=6, impact=impact)
        np.testing.assert_allclose(ir[:, :, 0], impact, atol=1e-12)

    # --- Known value: zero VAR ------------------------------------------------

    def test_zero_var_irf_after_h0(self):
        """A_list = [0]: IRF is zero for all h > 0."""
        n = 2
        A_list = [np.zeros((n, n))]
        impact = np.eye(n)
        ir = _irf_from_var(A_list, horizon=5, impact=impact)
        np.testing.assert_allclose(ir[:, :, 0], impact, atol=1e-12)
        np.testing.assert_allclose(ir[:, :, 1:], 0.0, atol=1e-12)

    def test_scalar_zero_var_irf(self):
        """Univariate, A=0: IR = [1, 0, 0, ...]."""
        n = 1
        A_list = [np.zeros((n, n))]
        impact = np.ones((n, n))
        ir = _irf_from_var(A_list, horizon=4, impact=impact)
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(ir.flatten(), expected, atol=1e-12)

    # --- Known value: scalar AR(1) geometric decay ----------------------------

    def test_ar1_geometric_decay(self):
        """Scalar AR(1): Phi[h] = rho^h so ir[h] = rho^h * impact."""
        rho = 0.5
        n = 1
        A_list = [np.array([[rho]])]
        impact = np.ones((n, n))
        H = 5
        ir = _irf_from_var(A_list, horizon=H, impact=impact)
        for h in range(H + 1):
            expected = rho ** h
            np.testing.assert_allclose(
                ir[0, 0, h], expected, atol=1e-10,
                err_msg=f"h={h}: expected {expected:.6f}, got {ir[0,0,h]:.6f}",
            )

    def test_multivariate_ar1_geometric_decay(self):
        """VAR(1) with A = rho*I: each diagonal IRF decays as rho^h."""
        rho = 0.6
        n = 2
        A_list = [np.eye(n) * rho]
        impact = np.eye(n)
        H = 4
        ir = _irf_from_var(A_list, horizon=H, impact=impact)
        for h in range(H + 1):
            expected_diag = rho ** h
            np.testing.assert_allclose(
                np.diag(ir[:, :, h]), expected_diag, atol=1e-10,
            )
            # Off-diagonals should be zero (diagonal A, diagonal impact)
            np.testing.assert_allclose(
                ir[:, :, h] - np.diag(np.diag(ir[:, :, h])), 0.0, atol=1e-10,
            )

    # --- Known value: VAR(2) recursion ----------------------------------------

    def test_var2_h2_recursion(self):
        """VAR(2): Phi[2] = A1*Phi[1] + A2*Phi[0] = A1^2 + A2."""
        n = 1
        A1 = np.array([[0.5]])
        A2 = np.array([[0.2]])
        impact = np.eye(n)
        ir = _irf_from_var([A1, A2], horizon=3, impact=impact)
        # Phi[1] = A1, Phi[2] = A1@A1 + A2@I = A1^2 + A2
        h2_expected = (A1 @ A1 + A2 @ np.eye(n)) @ impact
        np.testing.assert_allclose(ir[:, :, 2], h2_expected, atol=1e-10)

    def test_var2_h3_recursion(self):
        """VAR(2): Phi[3] = A1*Phi[2] + A2*Phi[1]."""
        n = 2
        rng = np.random.default_rng(23)
        A1 = rng.standard_normal((n, n)) * 0.2
        A2 = rng.standard_normal((n, n)) * 0.1
        impact = np.eye(n)
        ir = _irf_from_var([A1, A2], horizon=5, impact=impact)
        Phi0 = np.eye(n)
        Phi1 = A1 @ Phi0
        Phi2 = A1 @ Phi1 + A2 @ Phi0
        Phi3 = A1 @ Phi2 + A2 @ Phi1
        h3_expected = Phi3 @ impact
        np.testing.assert_allclose(ir[:, :, 3], h3_expected, atol=1e-10)

    # --- Properties -----------------------------------------------------------

    def test_all_finite_for_stable_var(self):
        """Stable VAR with small coefficients: all IRF values must be finite."""
        n = 3
        rng = np.random.default_rng(24)
        A_list = [rng.standard_normal((n, n)) * 0.1]
        impact = np.eye(n)
        ir = _irf_from_var(A_list, horizon=20, impact=impact)
        assert np.all(np.isfinite(ir)), "Non-finite IRF values for stable VAR"


# ---------------------------------------------------------------------------
# Tests for residual_bootstrap
# ---------------------------------------------------------------------------

class TestResidualBootstrap:
    """Tests for residual_bootstrap(...) -> dict."""

    def _setup_var1(self, T: int = 60, n: int = 2, seed: int = 0):
        """Return (residuals, Y, A_list, c) from _ols_var for a white-noise VAR."""
        Y = _make_var_data(T, n, p=1, seed=seed)
        A_list, c, Sigma, resid, X = _ols_var(Y, p=1)
        return resid, Y, A_list, c, n

    # --- Output structure -----------------------------------------------------

    def test_output_is_dict_with_draws_key(self):
        """Result must be a dict containing exactly the 'draws' key."""
        resid, Y, A_list, c, n = self._setup_var1(seed=30)
        result = residual_bootstrap(
            resid, Y, A_list, c, n_draws=3, horizon=2, rng=np.random.default_rng(1)
        )
        assert isinstance(result, dict)
        assert "draws" in result

    def test_draws_list_length(self):
        """draws list must contain exactly n_draws elements."""
        resid, Y, A_list, c, n = self._setup_var1(seed=31)
        n_draws = 7
        result = residual_bootstrap(
            resid, Y, A_list, c, n_draws=n_draws, horizon=3,
            rng=np.random.default_rng(2),
        )
        assert len(result["draws"]) == n_draws

    def test_draw_shape_with_irf_fn(self):
        """When irf_fn returns (horizon+1, n, n), each draw has that shape."""
        resid, Y, A_list, c, n = self._setup_var1(seed=32)
        H = 5

        def irf_fn(Y_star, p, horizon):
            return np.ones((horizon + 1, n, n))

        result = residual_bootstrap(
            resid, Y, A_list, c, n_draws=4, horizon=H, irf_fn=irf_fn,
            rng=np.random.default_rng(3),
        )
        for draw in result["draws"]:
            assert draw.shape == (H + 1, n, n), f"Expected ({H+1},{n},{n}), got {draw.shape}"

    # --- irf_fn=None branch ---------------------------------------------------

    def test_irf_fn_none_returns_zeros(self):
        """irf_fn=None: each draw must be zeros of shape (horizon+1, n, n)."""
        resid, Y, A_list, c, n = self._setup_var1(seed=33)
        H = 4
        result = residual_bootstrap(
            resid, Y, A_list, c, n_draws=5, horizon=H, irf_fn=None,
            rng=np.random.default_rng(4),
        )
        for draw in result["draws"]:
            assert draw.shape == (H + 1, n, n)
            np.testing.assert_allclose(draw, 0.0, atol=1e-12)

    # --- irf_fn call contract -------------------------------------------------

    def test_irf_fn_called_n_draws_times(self):
        """irf_fn must be called exactly n_draws times."""
        resid, Y, A_list, c, n = self._setup_var1(seed=34)
        call_count = [0]

        def irf_fn(Y_star, p, horizon):
            call_count[0] += 1
            return np.zeros((horizon + 1, n, n))

        n_draws = 11
        residual_bootstrap(
            resid, Y, A_list, c, n_draws=n_draws, horizon=3, irf_fn=irf_fn,
            rng=np.random.default_rng(5),
        )
        assert call_count[0] == n_draws

    def test_irf_fn_receives_correct_arguments(self):
        """irf_fn must receive (Y_star, p, horizon) with correct values."""
        resid, Y, A_list, c, n = self._setup_var1(seed=35)
        p = len(A_list)
        H = 7
        T = Y.shape[0]
        received = []

        def irf_fn(Y_star, p_arg, horizon_arg):
            received.append((Y_star.shape, p_arg, horizon_arg))
            return np.zeros((horizon_arg + 1, n, n))

        residual_bootstrap(
            resid, Y, A_list, c, n_draws=3, horizon=H, irf_fn=irf_fn,
            rng=np.random.default_rng(6),
        )
        for shape, p_arg, horizon_arg in received:
            assert shape == (T, n), f"Y_star shape {shape} != {(T, n)}"
            assert p_arg == p
            assert horizon_arg == H

    def test_y_star_initial_rows_match_y(self):
        """Y_star[:p] must equal Y[:p] (initial conditions preserved)."""
        resid, Y, A_list, c, n = self._setup_var1(seed=36)
        p = len(A_list)
        captured_inits = []

        def irf_fn(Y_star, p_arg, horizon):
            captured_inits.append(Y_star[:p_arg].copy())
            return np.zeros((horizon + 1, n, n))

        residual_bootstrap(
            resid, Y, A_list, c, n_draws=4, horizon=3, irf_fn=irf_fn,
            rng=np.random.default_rng(7),
        )
        for init in captured_inits:
            np.testing.assert_allclose(init, Y[:p], atol=1e-12)

    # --- rng=None branch ------------------------------------------------------

    def test_rng_none_does_not_raise(self):
        """rng=None should create a fresh generator and run successfully."""
        resid, Y, A_list, c, n = self._setup_var1(seed=37)
        result = residual_bootstrap(
            resid, Y, A_list, c, n_draws=3, horizon=2, irf_fn=None, rng=None
        )
        assert "draws" in result
        assert len(result["draws"]) == 3

    # --- Reproducibility ------------------------------------------------------

    def test_same_seed_identical_draws(self):
        """Same rng seed must produce identical Y_star simulations."""
        resid, Y, A_list, c, n = self._setup_var1(seed=38)
        captured1 = []
        captured2 = []

        def irf_fn1(Y_star, p, horizon):
            captured1.append(Y_star.copy())
            return np.zeros((horizon + 1, n, n))

        def irf_fn2(Y_star, p, horizon):
            captured2.append(Y_star.copy())
            return np.zeros((horizon + 1, n, n))

        kw = dict(n_draws=5, horizon=3)
        residual_bootstrap(resid, Y, A_list, c, **kw, irf_fn=irf_fn1, rng=np.random.default_rng(99))
        residual_bootstrap(resid, Y, A_list, c, **kw, irf_fn=irf_fn2, rng=np.random.default_rng(99))

        for ys1, ys2 in zip(captured1, captured2):
            np.testing.assert_array_equal(ys1, ys2)

    def test_different_seeds_different_draws(self):
        """Different rng seeds should (almost certainly) produce different Y_star."""
        resid, Y, A_list, c, n = self._setup_var1(seed=39)
        captured1 = []
        captured2 = []

        def irf_fn1(Y_star, p, horizon):
            captured1.append(Y_star.copy())
            return np.zeros((horizon + 1, n, n))

        def irf_fn2(Y_star, p, horizon):
            captured2.append(Y_star.copy())
            return np.zeros((horizon + 1, n, n))

        kw = dict(n_draws=5, horizon=3)
        residual_bootstrap(resid, Y, A_list, c, **kw, irf_fn=irf_fn1, rng=np.random.default_rng(100))
        residual_bootstrap(resid, Y, A_list, c, **kw, irf_fn=irf_fn2, rng=np.random.default_rng(200))

        # At least one draw must differ (draws drawn from different random streams)
        any_different = any(
            not np.allclose(ys1, ys2) for ys1, ys2 in zip(captured1, captured2)
        )
        assert any_different, "Expected different draws with different seeds"

    # --- VAR(2) variant -------------------------------------------------------

    def test_var2_draws_correct_count(self):
        """residual_bootstrap works for VAR(2) with correct draw count."""
        T, n, p = 80, 2, 2
        Y = _make_var_data(T, n, p=p, seed=40)
        A_list, c, Sigma, resid, X = _ols_var(Y, p=2)
        n_draws = 6
        result = residual_bootstrap(
            resid, Y, A_list, c, n_draws=n_draws, horizon=4,
            rng=np.random.default_rng(8),
        )
        assert len(result["draws"]) == n_draws

    # --- n_draws=1 edge case --------------------------------------------------

    def test_n_draws_1(self):
        """n_draws=1 should return a list with exactly one element."""
        resid, Y, A_list, c, n = self._setup_var1(seed=41)
        result = residual_bootstrap(
            resid, Y, A_list, c, n_draws=1, horizon=2, irf_fn=None,
            rng=np.random.default_rng(9),
        )
        assert len(result["draws"]) == 1

    # --- Slow: large n_draws --------------------------------------------------

    @pytest.mark.slow
    def test_large_n_draws_shape(self):
        """n_draws=200: verify list length and all draws have correct shape."""
        resid, Y, A_list, c, n = self._setup_var1(T=80, seed=42)
        H = 8
        result = residual_bootstrap(
            resid, Y, A_list, c, n_draws=200, horizon=H,
            irf_fn=lambda Y_s, p, h: np.zeros((h + 1, n, n)),
            rng=np.random.default_rng(10),
        )
        assert len(result["draws"]) == 200
        for draw in result["draws"]:
            assert draw.shape == (H + 1, n, n)
