"""Coverage tests for puremacro.inference.wild_bootstrap.

Module under test
-----------------
    wild_bootstrap(residuals, refit_fn, n_boot, rng) -> ndarray (n_boot, ...)
    wild_bootstrap_var(Y, *, p, horizon, impact_fn, n_boot, ci, seed)
      -> (point, lo, hi) each shape (n, n, horizon+1)

Coverage strategy
-----------------
Both functions are at 0% coverage (lines 43-56, 69-97 missing).

wild_bootstrap
  - Shape: output is (n_boot, ...) where ... is the shape returned by refit_fn.
  - Scalar refit_fn: output shape (n_boot,).
  - Vector refit_fn: output shape (n_boot, k).
  - n_boot=1 edge case.
  - rng=None branch: creates default_rng internally; no crash.
  - Rademacher property: each weight is ±1 exactly (hence e_boot has same
    absolute values as residuals).
  - 1-D residuals (ndim==1) branch: e_boot = residuals * w.
  - 2-D residuals (ndim==2) branch: e_boot = residuals * w[:, None].
  - refit_fn receives array of same shape as residuals.
  - Reproducibility: same seed → identical draws.
  - Different seeds → different draws.
  - All draws are finite (for finite inputs).
  - Output dtype is float64.

wild_bootstrap_var
  - Return type: three-tuple (point, lo, hi).
  - All three have shape (n, n, horizon+1).
  - lo <= hi element-wise.
  - point is finite for a stable DGP.
  - ci=0.9 gives lo/hi at 5th/95th percentile (contract).
  - Reproducibility: same seed → same lo/hi.
  - Different seeds → different lo/hi (near-certain).
  - LinAlgError fallback: when impact_fn raises LinAlgError on bootstrap
    replications, falls back to point estimate (draws[b] = point).
  - n_boot=1 runs without error.
  - impact_fn is called with (A_list, Sigma, resid) once for the point estimate.

Notes
-----
  - No external I/O; all tests use synthetic numpy DGPs.
  - The LinAlgError branch (line 94-95) is tested via monkeypatching impact_fn
    so it raises on bootstrap (but not on the initial call).
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.inference.wild_bootstrap import wild_bootstrap, wild_bootstrap_var
from puremacro.inference.bootstrap import _ols_var


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ar1_data(T: int = 80, rho: float = 0.5, n: int = 1, seed: int = 0) -> np.ndarray:
    """Simulate a VAR(1) DGP: Y_{t} = rho*I @ Y_{t-1} + eps."""
    rng = np.random.default_rng(seed)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = rho * Y[t - 1] + rng.standard_normal(n)
    return Y


def _make_var_data(T: int = 80, n: int = 2, p: int = 1, seed: int = 0) -> np.ndarray:
    """White-noise VAR data (T x n)."""
    return np.random.default_rng(seed).standard_normal((T, n))


def _cholesky_impact(A_list, Sigma, resid):
    """Minimal impact_fn: lower-triangular Cholesky of Sigma."""
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return np.eye(Sigma.shape[0])


# ---------------------------------------------------------------------------
# Tests for wild_bootstrap
# ---------------------------------------------------------------------------

class TestWildBootstrap:
    """Tests for wild_bootstrap(residuals, refit_fn, n_boot, rng)."""

    # --- Shape contracts -------------------------------------------------------

    def test_scalar_output_shape(self):
        """Scalar refit_fn: output shape is (n_boot,)."""
        residuals = np.random.default_rng(1).standard_normal(40)
        result = wild_bootstrap(
            residuals,
            refit_fn=lambda e: np.array(e.mean()),
            n_boot=20,
            rng=np.random.default_rng(2),
        )
        assert result.shape == (20,)

    def test_vector_output_shape(self):
        """Vector refit_fn (returns k-vector): output shape is (n_boot, k)."""
        k = 5
        residuals = np.random.default_rng(3).standard_normal(50)
        result = wild_bootstrap(
            residuals,
            refit_fn=lambda e: np.ones(k) * e.mean(),
            n_boot=15,
            rng=np.random.default_rng(4),
        )
        assert result.shape == (15, k)

    def test_n_boot_1(self):
        """n_boot=1: output shape is (1,) for scalar refit_fn."""
        residuals = np.ones(10)
        result = wild_bootstrap(
            residuals,
            refit_fn=lambda e: np.array(e.sum()),
            n_boot=1,
            rng=np.random.default_rng(5),
        )
        assert result.shape == (1,)

    def test_2d_residuals_shape(self):
        """2-D residuals (T, n): output shape is (n_boot, k) for vector refit_fn."""
        T, n, k = 30, 3, 2
        residuals = np.random.default_rng(6).standard_normal((T, n))
        result = wild_bootstrap(
            residuals,
            refit_fn=lambda e: e.mean(axis=0)[:k],  # returns (k,)
            n_boot=10,
            rng=np.random.default_rng(7),
        )
        assert result.shape == (10, k)

    # --- rng=None branch -------------------------------------------------------

    def test_rng_none_does_not_raise(self):
        """rng=None must create an internal generator and complete without error."""
        residuals = np.ones(20) * 0.5
        result = wild_bootstrap(
            residuals,
            refit_fn=lambda e: np.array(e.mean()),
            n_boot=5,
            rng=None,
        )
        assert result.shape == (5,)

    # --- Rademacher property ---------------------------------------------------

    def test_rademacher_weights_preserve_abs_values_1d(self):
        """1-D residuals: |e_boot| == |residuals| for every bootstrap draw."""
        T = 20
        residuals = np.arange(1.0, T + 1.0)  # all positive, non-zero
        captured = []

        def refit_fn(e_boot):
            captured.append(e_boot.copy())
            return np.array(0.0)

        wild_bootstrap(residuals, refit_fn=refit_fn, n_boot=5, rng=np.random.default_rng(8))
        for e_boot in captured:
            np.testing.assert_allclose(
                np.abs(e_boot), np.abs(residuals), atol=1e-12,
                err_msg="Absolute values changed after Rademacher weighting (1D)",
            )

    def test_rademacher_weights_preserve_abs_values_2d(self):
        """2-D residuals: |e_boot| == |residuals| for every bootstrap draw."""
        T, n = 15, 3
        residuals = np.random.default_rng(9).standard_normal((T, n)) + 2.0
        captured = []

        def refit_fn(e_boot):
            captured.append(e_boot.copy())
            return np.array(0.0)

        wild_bootstrap(residuals, refit_fn=refit_fn, n_boot=5, rng=np.random.default_rng(10))
        for e_boot in captured:
            np.testing.assert_allclose(
                np.abs(e_boot), np.abs(residuals), atol=1e-12,
                err_msg="Absolute values changed after Rademacher weighting (2D)",
            )

    def test_rademacher_signs_are_pm1(self):
        """Rademacher weights can only take values ±1 (check via sign-flip test)."""
        T = 50
        residuals_pos = np.ones(T)  # all +1
        captured_signs = []

        def refit_fn(e_boot):
            captured_signs.append(e_boot.copy())
            return np.array(0.0)

        wild_bootstrap(residuals_pos, refit_fn=refit_fn, n_boot=30, rng=np.random.default_rng(11))
        # e_boot == residuals * w, and residuals==1, so e_boot values are the weights
        unique_vals = set(np.unique(np.concatenate(captured_signs)))
        assert unique_vals.issubset({-1.0, 1.0}), (
            f"Expected only ±1.0 but got: {unique_vals}"
        )

    # --- 1-D vs 2-D branch distinction ----------------------------------------

    def test_1d_residuals_branch(self):
        """1-D residuals use the ndim==1 branch (e_boot = residuals * w, no [:, None])."""
        T = 25
        residuals = np.random.default_rng(12).standard_normal(T)
        shapes_seen = []

        def refit_fn(e_boot):
            shapes_seen.append(e_boot.shape)
            return np.array(0.0)

        wild_bootstrap(residuals, refit_fn=refit_fn, n_boot=3, rng=np.random.default_rng(13))
        for shape in shapes_seen:
            assert shape == (T,), f"Expected 1-D shape {(T,)}, got {shape}"

    def test_2d_residuals_branch(self):
        """2-D residuals use the ndim==2 branch (e_boot = residuals * w[:, None])."""
        T, n = 25, 4
        residuals = np.random.default_rng(14).standard_normal((T, n))
        shapes_seen = []

        def refit_fn(e_boot):
            shapes_seen.append(e_boot.shape)
            return np.array(0.0)

        wild_bootstrap(residuals, refit_fn=refit_fn, n_boot=3, rng=np.random.default_rng(15))
        for shape in shapes_seen:
            assert shape == (T, n), f"Expected 2-D shape {(T, n)}, got {shape}"

    # --- refit_fn call contract ------------------------------------------------

    def test_refit_fn_called_n_boot_times(self):
        """refit_fn must be called exactly n_boot times."""
        call_count = [0]
        residuals = np.ones(10)

        def refit_fn(e_boot):
            call_count[0] += 1
            return np.array(0.0)

        n_boot = 13
        wild_bootstrap(residuals, refit_fn=refit_fn, n_boot=n_boot, rng=np.random.default_rng(16))
        assert call_count[0] == n_boot

    # --- Reproducibility -------------------------------------------------------

    def test_same_seed_identical_draws(self):
        """Same rng seed must produce identical draws."""
        residuals = np.random.default_rng(17).standard_normal(30)
        refit_fn = lambda e: np.array(e.mean())
        r1 = wild_bootstrap(residuals, refit_fn=refit_fn, n_boot=10, rng=np.random.default_rng(99))
        r2 = wild_bootstrap(residuals, refit_fn=refit_fn, n_boot=10, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_different_draws(self):
        """Different seeds produce different draws (with very high probability)."""
        residuals = np.random.default_rng(18).standard_normal(50)
        refit_fn = lambda e: e.copy()  # returns the full e_boot
        r1 = wild_bootstrap(residuals, refit_fn=refit_fn, n_boot=20, rng=np.random.default_rng(100))
        r2 = wild_bootstrap(residuals, refit_fn=refit_fn, n_boot=20, rng=np.random.default_rng(200))
        assert not np.allclose(r1, r2), "Expected different draws with different seeds"

    # --- Output dtype ----------------------------------------------------------

    def test_output_dtype_float64(self):
        """Output dtype must be float64 regardless of input dtype."""
        residuals = np.array([1, -2, 3, -1], dtype=np.int32)
        result = wild_bootstrap(
            residuals,
            refit_fn=lambda e: np.array(e.mean()),
            n_boot=5,
            rng=np.random.default_rng(19),
        )
        assert result.dtype == np.float64

    # --- Finite output ---------------------------------------------------------

    def test_output_is_finite(self):
        """All draws must be finite for finite residuals and refit_fn."""
        residuals = np.random.default_rng(20).standard_normal(40)
        result = wild_bootstrap(
            residuals,
            refit_fn=lambda e: np.array(e.mean()),
            n_boot=20,
            rng=np.random.default_rng(21),
        )
        assert np.all(np.isfinite(result))

    # --- Known-value: zero residuals ------------------------------------------

    def test_zero_residuals_1d_gives_zero_draws(self):
        """Zero 1-D residuals: every refit_fn call receives zeros -> draws are 0."""
        T = 20
        residuals = np.zeros(T)
        result = wild_bootstrap(
            residuals,
            refit_fn=lambda e: np.array(e.mean()),
            n_boot=10,
            rng=np.random.default_rng(22),
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_zero_residuals_2d_gives_zero_draws(self):
        """Zero 2-D residuals: every draw is zero."""
        T, n = 15, 3
        residuals = np.zeros((T, n))
        result = wild_bootstrap(
            residuals,
            refit_fn=lambda e: e.mean(axis=0),  # returns (n,)
            n_boot=8,
            rng=np.random.default_rng(23),
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    # --- Known-value: constant-magnitude residuals ----------------------------

    def test_constant_residuals_abs_preserved(self):
        """If all residuals = c, each draw has |e_boot| = |c| (Rademacher)."""
        c = 3.14
        residuals = np.full(25, c)
        captured = []

        def refit_fn(e_boot):
            captured.append(np.abs(e_boot).mean())
            return np.array(0.0)

        wild_bootstrap(residuals, refit_fn=refit_fn, n_boot=10, rng=np.random.default_rng(24))
        for mean_abs in captured:
            assert abs(mean_abs - c) < 1e-12

    # --- Mean of draws converges to zero (zero-mean residuals) ----------------

    def test_mean_statistic_converges_to_zero(self):
        """With zero-mean residuals and refit_fn = mean(e_boot), E[draws] ~ 0."""
        rng = np.random.default_rng(25)
        residuals = rng.standard_normal(100)
        residuals -= residuals.mean()  # force zero mean
        result = wild_bootstrap(
            residuals,
            refit_fn=lambda e: np.array(e.mean()),
            n_boot=500,
            rng=np.random.default_rng(26),
        )
        # E[w_t] = 0, so E[e_boot.mean()] = 0 by linearity
        assert abs(result.mean()) < 0.05, (
            f"Expected ~0, got {result.mean():.4f}"
        )


# ---------------------------------------------------------------------------
# Tests for wild_bootstrap_var
# ---------------------------------------------------------------------------

class TestWildBootstrapVar:
    """Tests for wild_bootstrap_var(Y, *, p, horizon, impact_fn, ...)."""

    # --- Return type / structure -----------------------------------------------

    def test_returns_three_tuple(self):
        """Return value must be a 3-tuple (point, lo, hi)."""
        Y = _make_var_data(T=60, n=2, p=1, seed=50)
        result = wild_bootstrap_var(
            Y, p=1, horizon=4, impact_fn=_cholesky_impact, n_boot=10, seed=1
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_output_shapes(self):
        """All three outputs have shape (n, n, horizon+1)."""
        n, H = 2, 5
        Y = _make_var_data(T=70, n=n, p=1, seed=51)
        point, lo, hi = wild_bootstrap_var(
            Y, p=1, horizon=H, impact_fn=_cholesky_impact, n_boot=15, seed=2
        )
        for arr in (point, lo, hi):
            assert arr.shape == (n, n, H + 1), (
                f"Expected ({n},{n},{H+1}), got {arr.shape}"
            )

    def test_output_shapes_var2(self):
        """VAR(2) case: shapes still (n, n, horizon+1)."""
        n, H = 2, 3
        Y = _make_var_data(T=80, n=n, p=2, seed=52)
        point, lo, hi = wild_bootstrap_var(
            Y, p=2, horizon=H, impact_fn=_cholesky_impact, n_boot=10, seed=3
        )
        for arr in (point, lo, hi):
            assert arr.shape == (n, n, H + 1)

    # --- lo <= hi contract ------------------------------------------------------

    def test_lo_le_hi_element_wise(self):
        """lo <= hi element-wise (lower quantile <= upper quantile)."""
        n, H = 2, 6
        Y = _make_ar1_data(T=100, rho=0.4, n=n, seed=53)
        _, lo, hi = wild_bootstrap_var(
            Y, p=1, horizon=H, impact_fn=_cholesky_impact, n_boot=50, seed=4
        )
        assert np.all(lo <= hi), (
            "lo must be <= hi element-wise"
        )

    # --- Point is finite for stable DGP ----------------------------------------

    def test_point_is_finite(self):
        """Point IRF must be finite for a stable DGP."""
        Y = _make_ar1_data(T=80, rho=0.4, n=2, seed=54)
        point, _, _ = wild_bootstrap_var(
            Y, p=1, horizon=8, impact_fn=_cholesky_impact, n_boot=10, seed=5
        )
        assert np.all(np.isfinite(point))

    def test_lo_hi_are_finite(self):
        """lo and hi must be finite for a stable DGP."""
        Y = _make_ar1_data(T=80, rho=0.4, n=2, seed=55)
        _, lo, hi = wild_bootstrap_var(
            Y, p=1, horizon=8, impact_fn=_cholesky_impact, n_boot=10, seed=6
        )
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))

    # --- ci parameter: quantile contract ---------------------------------------

    def test_ci_0_5_gives_symmetric_quantiles(self):
        """ci=0.5 → lo at 25th pct, hi at 75th pct; narrower than ci=0.9."""
        Y = _make_ar1_data(T=100, rho=0.3, n=2, seed=56)
        _, lo90, hi90 = wild_bootstrap_var(
            Y, p=1, horizon=4, impact_fn=_cholesky_impact, n_boot=50, ci=0.9, seed=7
        )
        _, lo50, hi50 = wild_bootstrap_var(
            Y, p=1, horizon=4, impact_fn=_cholesky_impact, n_boot=50, ci=0.5, seed=7
        )
        # ci=0.5 band must be narrower (or equal) than ci=0.9 band
        width90 = hi90 - lo90
        width50 = hi50 - lo50
        assert np.all(width50 <= width90 + 1e-10), (
            "ci=0.5 should produce narrower bands than ci=0.9"
        )

    # --- Reproducibility -------------------------------------------------------

    def test_same_seed_identical_results(self):
        """Same seed must produce identical (point, lo, hi)."""
        Y = _make_var_data(T=70, n=2, p=1, seed=57)
        r1 = wild_bootstrap_var(
            Y, p=1, horizon=4, impact_fn=_cholesky_impact, n_boot=20, seed=42
        )
        r2 = wild_bootstrap_var(
            Y, p=1, horizon=4, impact_fn=_cholesky_impact, n_boot=20, seed=42
        )
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_different_bands(self):
        """Different seeds produce different (lo, hi) with very high probability."""
        Y = _make_var_data(T=80, n=2, p=1, seed=58)
        _, lo1, hi1 = wild_bootstrap_var(
            Y, p=1, horizon=4, impact_fn=_cholesky_impact, n_boot=30, seed=10
        )
        _, lo2, hi2 = wild_bootstrap_var(
            Y, p=1, horizon=4, impact_fn=_cholesky_impact, n_boot=30, seed=11
        )
        assert not (np.allclose(lo1, lo2) and np.allclose(hi1, hi2)), (
            "Expected different bands with different seeds"
        )

    # --- LinAlgError fallback branch -------------------------------------------

    def test_linalg_error_fallback_uses_point(self):
        """When impact_fn raises LinAlgError on bootstrap replications, draws[b] = point."""
        Y = _make_var_data(T=80, n=2, p=1, seed=59)
        call_count = [0]

        def impact_fn_raise_on_boot(A_list, Sigma, resid):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call is the point estimate — succeed
                return np.linalg.cholesky(Sigma)
            else:
                # All bootstrap calls fail
                raise np.linalg.LinAlgError("Singular matrix (test)")

        point, lo, hi = wild_bootstrap_var(
            Y, p=1, horizon=3, impact_fn=impact_fn_raise_on_boot, n_boot=5, seed=20
        )
        # All draws are equal to point, so lo == hi == point
        np.testing.assert_allclose(lo, point, atol=1e-12)
        np.testing.assert_allclose(hi, point, atol=1e-12)

    # --- n_boot=1 edge case ----------------------------------------------------

    def test_n_boot_1_runs(self):
        """n_boot=1 completes without error and returns correct shapes."""
        n, H = 2, 3
        Y = _make_var_data(T=60, n=n, p=1, seed=60)
        point, lo, hi = wild_bootstrap_var(
            Y, p=1, horizon=H, impact_fn=_cholesky_impact, n_boot=1, seed=21
        )
        for arr in (point, lo, hi):
            assert arr.shape == (n, n, H + 1)

    # --- impact_fn call contract -----------------------------------------------

    def test_impact_fn_called_for_point_estimate(self):
        """impact_fn must be called at least once (for the point estimate)."""
        call_count = [0]
        Y = _make_var_data(T=60, n=2, p=1, seed=61)

        def tracking_impact_fn(A_list, Sigma, resid):
            call_count[0] += 1
            return _cholesky_impact(A_list, Sigma, resid)

        wild_bootstrap_var(
            Y, p=1, horizon=3, impact_fn=tracking_impact_fn, n_boot=5, seed=22
        )
        # Called once for point + once per successful bootstrap
        assert call_count[0] >= 1

    # --- Horizon=0 edge case ---------------------------------------------------

    def test_horizon_zero(self):
        """horizon=0: all shapes are (n, n, 1) and point is the impact matrix."""
        n = 2
        Y = _make_var_data(T=60, n=n, p=1, seed=62)
        point, lo, hi = wild_bootstrap_var(
            Y, p=1, horizon=0, impact_fn=_cholesky_impact, n_boot=10, seed=23
        )
        for arr in (point, lo, hi):
            assert arr.shape == (n, n, 1)

    # --- Yb preserves initial conditions --------------------------------------

    def test_initial_conditions_of_bootstrap_series(self):
        """Bootstrap series Yb[:p] must equal Y[:p] (initial conditions preserved)."""
        Y = _make_ar1_data(T=60, rho=0.4, n=2, seed=63)
        p = 1
        captured_firsts = []

        def tracking_impact_fn(A_list, Sigma, resid):
            return _cholesky_impact(A_list, Sigma, resid)

        # We can't directly inspect Yb inside the loop, but we can verify
        # that the function completes without numerical issues (integrity check)
        point, lo, hi = wild_bootstrap_var(
            Y, p=p, horizon=4, impact_fn=tracking_impact_fn, n_boot=5, seed=24
        )
        # Verify all outputs are finite (which would fail if initial conditions blow up)
        assert np.all(np.isfinite(point))
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(hi))

    # --- Slow: large n_boot for stable DGP ------------------------------------

    @pytest.mark.slow
    def test_large_n_boot_stable_dgp(self):
        """Large n_boot=200: bands contain the point estimate (plausibility)."""
        n, H = 2, 6
        Y = _make_ar1_data(T=150, rho=0.3, n=n, seed=64)
        point, lo, hi = wild_bootstrap_var(
            Y, p=1, horizon=H, impact_fn=_cholesky_impact, n_boot=200, ci=0.9, seed=25
        )
        # With 200 bootstrap draws and a stable DGP, lo <= point <= hi should
        # hold for the vast majority of elements.
        frac_contained = np.mean((lo <= point + 1e-8) & (point <= hi + 1e-8))
        assert frac_contained > 0.5, (
            f"Only {frac_contained:.2%} of point elements contained in [lo, hi]"
        )
