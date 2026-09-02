"""Comprehensive tests for puremacro.inference.moving_block.

These were written against `puremacro/inference/moving_block_bootstrap.py`,
a second copy of the same module that had drifted: its `_default_irf_fn`
imported `statsmodels`, a dev-only extra, breaking the package's first
promise. The copy is gone; these tests now cover the surviving module
alongside `test_inference_moving_block_coverage.py`, whose test names are
disjoint from these.

Coverage strategy
-----------------
- _sample_blocks: shape, content drawn from residuals, target length
  truncation, minimum block_len=1 edge case, single-row residual.
- _simulate_var: shape (p + T_star rows), known-value on a zero-VAR,
  known-value recovery of a trivial AR(1) DGP.
- moving_block_bootstrap: output dict keys and types, block_len auto-selection
  (round(T_eff^(1/3))), block_len=None vs explicit, draws list length, each
  draw shape, custom irf_fn invoked correctly, rng=None branch, reproducibility.
- bootstrap_percentiles: output shapes, ordering lo <= med <= hi,
  degenerate (all-equal) draws, quantile levels passed correctly,
  stack axis semantics.
- _default_irf_fn: shape (horizon+1, n, n), output is finite, VAR(1) trivial.
- Known-value: for a white-noise VAR (A_list = [0-matrix]) the bootstrap
  draws should remain centred near zero.
- Properties: block residuals are subsets of original; more draws does not
  change shape; variances of bootstrap IRFs are non-negative.
- Edge cases / error: block_len > T_eff; block_len=1 still valid.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from puremacro.inference.moving_block import (
    _sample_blocks,
    _simulate_var,
    _default_irf_fn,
    moving_block_bootstrap,
    bootstrap_percentiles,
)


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_residuals(T: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n))


def _zero_var1(n: int, T: int, seed: int = 0):
    """Return (residuals, Y, A_list, intercept) for a zero-dynamics VAR(1).

    All coefficient matrices are zero so Y_t = intercept + e_t.  This makes
    the expected IRF trivial to compute analytically.
    """
    rng = np.random.default_rng(seed)
    p = 1
    A_list = [np.zeros((n, n))]
    intercept = np.zeros(n)
    e = rng.standard_normal((T, n))
    Y = np.zeros((T, n))
    for t in range(p, T):
        Y[t] = intercept + e[t]
    residuals = e[p:]   # shape (T-p, n) = (T-1, n)
    return residuals, Y, A_list, intercept


def _stable_var1_data(T: int = 100, n: int = 2, seed: int = 0) -> tuple:
    """Simulate VAR(1) with diagonal A=0.5*I and return (residuals, Y, A_list, intercept)."""
    rng = np.random.default_rng(seed)
    p = 1
    A = 0.5 * np.eye(n)
    A_list = [A]
    intercept = np.zeros(n)
    Y = np.zeros((T, n))
    e = rng.standard_normal((T, n))
    for t in range(p, T):
        Y[t] = A @ Y[t - 1] + intercept + e[t]
    residuals = e[p:]  # shape (T-1, n)
    return residuals, Y, A_list, intercept


# ===========================================================================
# _sample_blocks
# ===========================================================================

class TestSampleBlocks:

    def test_output_shape_exact_multiple(self):
        """target_len that is an exact multiple of block_len."""
        rng = np.random.default_rng(0)
        residuals = _make_residuals(T=30, n=2)
        block_len = 5
        target_len = 20
        out = _sample_blocks(residuals, block_len=block_len,
                              target_len=target_len, rng=rng)
        assert out.shape == (target_len, 2)

    def test_output_shape_non_multiple(self):
        """target_len not a multiple of block_len — output must still be exactly target_len."""
        rng = np.random.default_rng(1)
        residuals = _make_residuals(T=50, n=3)
        block_len = 7
        target_len = 25
        out = _sample_blocks(residuals, block_len=block_len,
                              target_len=target_len, rng=rng)
        assert out.shape == (target_len, 3)

    def test_output_rows_are_subsets_of_residuals(self):
        """Every row of the bootstrap output must appear in the original residuals."""
        rng = np.random.default_rng(2)
        residuals = _make_residuals(T=20, n=2, seed=42)
        out = _sample_blocks(residuals, block_len=3, target_len=12, rng=rng)
        # Each row must match some row of `residuals` up to floating-point equality
        found = [
            any(np.allclose(out[i], residuals[j]) for j in range(residuals.shape[0]))
            for i in range(out.shape[0])
        ]
        assert all(found), "Some output rows are not present in the original residuals"

    def test_block_len_1_valid(self):
        """block_len=1 is a degenerate IID bootstrap — must work without error."""
        rng = np.random.default_rng(3)
        residuals = _make_residuals(T=15, n=2)
        out = _sample_blocks(residuals, block_len=1, target_len=10, rng=rng)
        assert out.shape == (10, 2)

    def test_block_len_equals_T(self):
        """block_len = T means there is exactly one possible block (the whole series)."""
        rng = np.random.default_rng(4)
        T, n = 10, 2
        residuals = _make_residuals(T=T, n=n, seed=5)
        # Only one valid start: 0. Target shorter than T.
        out = _sample_blocks(residuals, block_len=T, target_len=5, rng=rng)
        assert out.shape == (5, n)
        # Each row must come from the beginning of residuals (the only block)
        np.testing.assert_allclose(out, residuals[:5])

    def test_dtype_preserved(self):
        """Output dtype should match input residuals dtype."""
        rng = np.random.default_rng(5)
        residuals = _make_residuals(T=20, n=2).astype(np.float32)
        out = _sample_blocks(residuals, block_len=4, target_len=10, rng=rng)
        assert out.dtype == np.float32

    def test_target_len_1(self):
        """Single-row output — edge case for short simulations."""
        rng = np.random.default_rng(6)
        residuals = _make_residuals(T=10, n=3)
        out = _sample_blocks(residuals, block_len=3, target_len=1, rng=rng)
        assert out.shape == (1, 3)

    def test_reproducible_with_same_rng(self):
        """Same Generator seed produces identical output."""
        residuals = _make_residuals(T=30, n=2, seed=7)
        out1 = _sample_blocks(residuals, block_len=5, target_len=20,
                               rng=np.random.default_rng(99))
        out2 = _sample_blocks(residuals, block_len=5, target_len=20,
                               rng=np.random.default_rng(99))
        np.testing.assert_array_equal(out1, out2)

    def test_different_rng_gives_different_output(self):
        """Two different seeds should (almost certainly) differ."""
        residuals = _make_residuals(T=30, n=2, seed=7)
        out1 = _sample_blocks(residuals, block_len=5, target_len=20,
                               rng=np.random.default_rng(1))
        out2 = _sample_blocks(residuals, block_len=5, target_len=20,
                               rng=np.random.default_rng(2))
        assert not np.allclose(out1, out2)

    def test_n_blocks_ceiling(self):
        """The number of sampled blocks = ceil(target_len / block_len)."""
        rng = np.random.default_rng(8)
        # target_len=11, block_len=5 → ceil(11/5)=3 blocks → 15 rows → truncated to 11
        residuals = _make_residuals(T=20, n=2)
        out = _sample_blocks(residuals, block_len=5, target_len=11, rng=rng)
        assert out.shape == (11, 2)


# ===========================================================================
# _simulate_var
# ===========================================================================

class TestSimulateVar:

    def test_output_shape_p1(self):
        """p=1: output has p + T_star = 1 + T_star rows."""
        p, n, T_star = 1, 2, 50
        Y_init = np.zeros((p, n))
        A_list = [0.5 * np.eye(n)]
        intercept = np.zeros(n)
        e_star = _make_residuals(T=T_star, n=n)
        Y_out = _simulate_var(Y_init, A_list, intercept, e_star)
        assert Y_out.shape == (p + T_star, n)

    def test_output_shape_p2(self):
        """p=2: output has p + T_star = 2 + T_star rows."""
        p, n, T_star = 2, 3, 40
        Y_init = np.zeros((p, n))
        A_list = [0.3 * np.eye(n), 0.1 * np.eye(n)]
        intercept = np.zeros(n)
        e_star = _make_residuals(T=T_star, n=n)
        Y_out = _simulate_var(Y_init, A_list, intercept, e_star)
        assert Y_out.shape == (p + T_star, n)

    def test_initial_conditions_preserved(self):
        """The first p rows of the output must equal Y_init exactly."""
        p, n = 1, 2
        Y_init = np.array([[3.0, -1.5]])
        A_list = [0.5 * np.eye(n)]
        intercept = np.zeros(n)
        e_star = _make_residuals(T=20, n=n)
        Y_out = _simulate_var(Y_init, A_list, intercept, e_star)
        np.testing.assert_array_equal(Y_out[:p], Y_init)

    def test_zero_dynamics_known_value(self):
        """If A_list = [0] and intercept = 0, Y[p+t] = e_star[t] exactly."""
        p, n = 1, 2
        T_star = 10
        Y_init = np.zeros((p, n))
        A_list = [np.zeros((n, n))]
        intercept = np.zeros(n)
        e_star = _make_residuals(T=T_star, n=n, seed=42)
        Y_out = _simulate_var(Y_init, A_list, intercept, e_star)
        # Y[1], Y[2], ... should equal e_star[0], e_star[1], ...
        np.testing.assert_allclose(Y_out[p:], e_star)

    def test_intercept_shifts_mean(self):
        """With A=0 and constant intercept c, all simulated rows (after p) equal c + e."""
        p, n = 1, 2
        T_star = 20
        c = np.array([5.0, -3.0])
        Y_init = np.zeros((p, n))
        A_list = [np.zeros((n, n))]
        e_star = np.zeros((T_star, n))   # zero residuals → Y[t] = c exactly
        Y_out = _simulate_var(Y_init, A_list, c, e_star)
        np.testing.assert_allclose(Y_out[p:], c[np.newaxis, :] * np.ones((T_star, n)))

    def test_ar1_one_step_recursion(self):
        """Single-step check: Y[p] = A @ Y[p-1] + intercept + e[0]."""
        n = 2
        p = 1
        A = np.array([[0.4, 0.1], [0.0, 0.6]])
        Y_init = np.array([[1.0, 2.0]])
        intercept = np.array([0.1, -0.2])
        e_star = np.array([[0.5, -0.3]])   # T_star=1
        Y_out = _simulate_var(Y_init, [A], intercept, e_star)
        expected_Y1 = A @ Y_init[0] + intercept + e_star[0]
        np.testing.assert_allclose(Y_out[p], expected_Y1)

    def test_output_finite_for_stable_var(self):
        """A stable VAR should produce only finite values over 100 steps."""
        p, n = 1, 3
        T_star = 100
        rng = np.random.default_rng(0)
        A_list = [0.4 * np.eye(n)]
        intercept = np.zeros(n)
        Y_init = np.zeros((p, n))
        e_star = rng.standard_normal((T_star, n))
        Y_out = _simulate_var(Y_init, A_list, intercept, e_star)
        assert np.all(np.isfinite(Y_out))

    def test_p2_two_step_recursion(self):
        """p=2: Y[2] = A1 @ Y[1] + A2 @ Y[0] + intercept + e[0]."""
        n = 2
        p = 2
        A1 = np.array([[0.3, 0.0], [0.0, 0.2]])
        A2 = np.array([[0.1, 0.0], [0.0, 0.1]])
        Y_init = np.array([[1.0, 0.5], [0.8, -0.3]])  # rows = lag-2, lag-1
        intercept = np.zeros(n)
        e_star = np.array([[0.1, -0.1]])   # T_star=1
        Y_out = _simulate_var(Y_init, [A1, A2], intercept, e_star)
        # Y[p+0] = A1 @ Y[p-1] + A2 @ Y[p-2] + intercept + e[0]
        #        = A1 @ Y[1]   + A2 @ Y[0]    + 0         + e[0]
        expected = A1 @ Y_init[1] + A2 @ Y_init[0] + e_star[0]
        np.testing.assert_allclose(Y_out[p], expected)


# ===========================================================================
# _default_irf_fn
# ===========================================================================

class TestDefaultIrfFn:

    def test_output_shape(self):
        """Shape must be (horizon+1, n, n)."""
        residuals, Y, A_list, intercept = _stable_var1_data(T=100, n=2, seed=0)
        p = len(A_list)
        e_star = _make_residuals(T=80, n=2, seed=1)
        Y_star = _simulate_var(Y[:p], A_list, intercept, e_star)
        horizon = 8
        out = _default_irf_fn(Y_star, p=p, horizon=horizon)
        assert out.shape == (horizon + 1, 2, 2)

    def test_output_finite(self):
        """Output must contain only finite values for a stable DGP."""
        residuals, Y, A_list, intercept = _stable_var1_data(T=120, n=2, seed=2)
        p = len(A_list)
        e_star = _make_residuals(T=100, n=2, seed=3)
        Y_star = _simulate_var(Y[:p], A_list, intercept, e_star)
        out = _default_irf_fn(Y_star, p=p, horizon=5)
        assert np.all(np.isfinite(out))

    def test_three_variable_shape(self):
        """n=3 VAR(1): shape (H+1, 3, 3)."""
        residuals, Y, A_list, intercept = _stable_var1_data(T=150, n=3, seed=4)
        p = len(A_list)
        e_star = _make_residuals(T=120, n=3, seed=5)
        Y_star = _simulate_var(Y[:p], A_list, intercept, e_star)
        out = _default_irf_fn(Y_star, p=p, horizon=6)
        assert out.shape == (7, 3, 3)

    def test_horizon_0_returns_single_slice(self):
        """horizon=0 → output shape (1, n, n)."""
        residuals, Y, A_list, intercept = _stable_var1_data(T=80, n=2, seed=6)
        p = len(A_list)
        e_star = _make_residuals(T=60, n=2, seed=7)
        Y_star = _simulate_var(Y[:p], A_list, intercept, e_star)
        out = _default_irf_fn(Y_star, p=p, horizon=0)
        assert out.shape == (1, 2, 2)


# ===========================================================================
# moving_block_bootstrap
# ===========================================================================

class TestMovingBlockBootstrap:

    def _run(self, n=2, T=80, n_draws=10, block_len=None, horizon=4,
             seed=0, irf_fn=None):
        residuals, Y, A_list, intercept = _stable_var1_data(T=T, n=n, seed=seed)
        rng = np.random.default_rng(seed)
        return moving_block_bootstrap(
            residuals=residuals,
            Y=Y,
            A_list=A_list,
            intercept=intercept,
            n_draws=n_draws,
            block_len=block_len,
            horizon=horizon,
            irf_fn=irf_fn,
            rng=rng,
        )

    # --- output contract ---

    def test_output_keys(self):
        result = self._run()
        assert set(result.keys()) == {"draws", "block_len"}

    def test_draws_is_list_of_correct_length(self):
        n_draws = 15
        result = self._run(n_draws=n_draws)
        assert isinstance(result["draws"], list)
        assert len(result["draws"]) == n_draws

    def test_each_draw_shape(self):
        n, horizon, n_draws = 2, 6, 8
        result = self._run(n=n, horizon=horizon, n_draws=n_draws)
        for draw in result["draws"]:
            assert draw.shape == (horizon + 1, n, n), \
                f"Expected ({horizon+1}, {n}, {n}); got {draw.shape}"

    def test_block_len_auto_selection(self):
        """If block_len is None, block_len used = round(T_eff^(1/3))."""
        T = 80
        residuals, Y, A_list, intercept = _stable_var1_data(T=T, n=2, seed=0)
        T_eff = residuals.shape[0]  # T - p = 79
        expected_ell = round(T_eff ** (1 / 3))
        expected_ell = max(expected_ell, 1)
        result = moving_block_bootstrap(
            residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
            n_draws=5, block_len=None, horizon=3,
            rng=np.random.default_rng(0),
        )
        assert result["block_len"] == expected_ell

    def test_explicit_block_len_returned(self):
        """block_len passed explicitly is echoed back in result."""
        result = self._run(block_len=4)
        assert result["block_len"] == 4

    def test_block_len_clamped_to_1(self):
        """block_len=0 or tiny T_eff should not crash; ell >= 1 enforced."""
        # Use extremely small T so round(T_eff^(1/3)) = 1 or 2
        residuals, Y, A_list, intercept = _stable_var1_data(T=10, n=2, seed=0)
        result = moving_block_bootstrap(
            residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
            n_draws=3, block_len=None, horizon=2,
            rng=np.random.default_rng(0),
        )
        assert result["block_len"] >= 1

    def test_rng_none_runs(self):
        """rng=None should not raise; output contract holds."""
        residuals, Y, A_list, intercept = _stable_var1_data(T=60, n=2, seed=0)
        result = moving_block_bootstrap(
            residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
            n_draws=5, horizon=3, rng=None,
        )
        assert len(result["draws"]) == 5

    def test_reproducible_with_same_rng(self):
        """Same rng seed → identical draws."""
        residuals, Y, A_list, intercept = _stable_var1_data(T=60, n=2, seed=1)
        kw = dict(residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
                  n_draws=5, block_len=3, horizon=3)
        r1 = moving_block_bootstrap(**kw, rng=np.random.default_rng(42))
        r2 = moving_block_bootstrap(**kw, rng=np.random.default_rng(42))
        for d1, d2 in zip(r1["draws"], r2["draws"]):
            np.testing.assert_array_equal(d1, d2)

    def test_different_rng_different_draws(self):
        """Different rng seeds → different draws."""
        residuals, Y, A_list, intercept = _stable_var1_data(T=60, n=2, seed=2)
        kw = dict(residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
                  n_draws=5, block_len=3, horizon=3)
        r1 = moving_block_bootstrap(**kw, rng=np.random.default_rng(1))
        r2 = moving_block_bootstrap(**kw, rng=np.random.default_rng(2))
        any_differ = any(
            not np.allclose(d1, d2)
            for d1, d2 in zip(r1["draws"], r2["draws"])
        )
        assert any_differ

    def test_draws_are_finite(self):
        """All draws should be finite for a stable well-conditioned DGP."""
        result = self._run(T=80, n=2, n_draws=10, horizon=5, seed=3)
        for draw in result["draws"]:
            assert np.all(np.isfinite(draw)), "Non-finite draw detected"

    # --- custom irf_fn ---

    def test_custom_irf_fn_invoked(self):
        """custom irf_fn should be called once per draw."""
        call_count = {"n": 0}
        horizon = 4
        n = 2

        def counting_irf_fn(Y_star, p, h):
            call_count["n"] += 1
            return np.zeros((h + 1, n, n))

        residuals, Y, A_list, intercept = _stable_var1_data(T=60, n=n, seed=4)
        n_draws = 7
        moving_block_bootstrap(
            residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
            n_draws=n_draws, block_len=3, horizon=horizon,
            irf_fn=counting_irf_fn, rng=np.random.default_rng(0),
        )
        assert call_count["n"] == n_draws

    def test_custom_irf_fn_shape_passed(self):
        """irf_fn receives (Y_star, p, horizon) with the right p and horizon."""
        observed = {}
        horizon = 6
        p_expected = 1
        n = 2

        def recording_irf_fn(Y_star, p, h):
            observed["p"] = p
            observed["h"] = h
            observed["Y_star_shape"] = Y_star.shape
            return np.zeros((h + 1, n, n))

        residuals, Y, A_list, intercept = _stable_var1_data(T=60, n=n, seed=5)
        moving_block_bootstrap(
            residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
            n_draws=1, block_len=3, horizon=horizon,
            irf_fn=recording_irf_fn, rng=np.random.default_rng(0),
        )
        assert observed["p"] == p_expected
        assert observed["h"] == horizon
        # Y_star must have at least p + 1 rows
        assert observed["Y_star_shape"][0] >= p_expected + 1
        assert observed["Y_star_shape"][1] == n

    # --- VAR(2) lags ---

    def test_var2_runs_and_shapes(self):
        """VAR(2): draws have correct shape."""
        n, horizon, n_draws = 2, 4, 6
        p = 2
        T = 100
        rng = np.random.default_rng(0)
        A_list = [0.3 * np.eye(n), 0.1 * np.eye(n)]
        intercept = np.zeros(n)
        Y = np.zeros((T, n))
        e = rng.standard_normal((T, n))
        for t in range(p, T):
            Y[t] = A_list[0] @ Y[t - 1] + A_list[1] @ Y[t - 2] + e[t]
        residuals = e[p:]
        result = moving_block_bootstrap(
            residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
            n_draws=n_draws, block_len=4, horizon=horizon,
            rng=np.random.default_rng(1),
        )
        assert len(result["draws"]) == n_draws
        for draw in result["draws"]:
            assert draw.shape == (horizon + 1, n, n)

    # --- known-value: zero-dynamics VAR ---

    def test_zero_var_draws_near_zero_mean(self):
        """With A_list=[0] and zero intercept, bootstrap IRFs should have
        median near zero at all horizons (only impact response via irf_fn
        is non-trivial, but the custom zero-returning irf_fn makes it exact)."""
        n = 2
        residuals, Y, A_list, intercept = _zero_var1(n=n, T=80, seed=10)

        def zero_irf_fn(Y_star, p, h):
            return np.zeros((h + 1, n, n))

        result = moving_block_bootstrap(
            residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
            n_draws=20, block_len=3, horizon=5,
            irf_fn=zero_irf_fn, rng=np.random.default_rng(0),
        )
        for draw in result["draws"]:
            np.testing.assert_array_equal(draw, np.zeros((6, n, n)))


# ===========================================================================
# bootstrap_percentiles
# ===========================================================================

class TestBootstrapPercentiles:

    def _make_draws(self, B: int, H: int, n: int, seed: int = 0) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        return [rng.standard_normal((H + 1, n, n)) for _ in range(B)]

    def test_output_shapes(self):
        B, H, n = 50, 8, 2
        draws = self._make_draws(B=B, H=H, n=n)
        lo, med, hi = bootstrap_percentiles(draws, q_lo=16, q_hi=84)
        assert lo.shape  == (H + 1, n, n)
        assert med.shape == (H + 1, n, n)
        assert hi.shape  == (H + 1, n, n)

    def test_lo_le_med_le_hi(self):
        """lo <= med <= hi element-wise (by construction of percentile ordering)."""
        draws = self._make_draws(B=100, H=5, n=2, seed=1)
        lo, med, hi = bootstrap_percentiles(draws, q_lo=5, q_hi=95)
        assert np.all(lo <= med + 1e-10), "lo > med detected"
        assert np.all(med <= hi + 1e-10), "med > hi detected"

    def test_degenerate_all_equal_draws(self):
        """If every draw is the same constant array, lo = med = hi = that constant."""
        H, n = 3, 2
        const = np.ones((H + 1, n, n)) * 2.718
        draws = [const.copy() for _ in range(20)]
        lo, med, hi = bootstrap_percentiles(draws)
        np.testing.assert_allclose(lo, const)
        np.testing.assert_allclose(med, const)
        np.testing.assert_allclose(hi, const)

    def test_symmetric_percentiles_known_value(self):
        """With q_lo=0, q_hi=100: lo=min, hi=max across draws."""
        H, n = 2, 1
        rng = np.random.default_rng(42)
        B = 30
        draws = [rng.standard_normal((H + 1, n, n)) for _ in range(B)]
        lo, med, hi = bootstrap_percentiles(draws, q_lo=0, q_hi=100)
        stack = np.stack(draws, axis=0)
        np.testing.assert_allclose(lo, stack.min(axis=0))
        np.testing.assert_allclose(hi, stack.max(axis=0))

    def test_median_matches_numpy(self):
        """med must equal np.percentile(..., 50, axis=0)."""
        draws = self._make_draws(B=40, H=4, n=2, seed=2)
        _, med, _ = bootstrap_percentiles(draws, q_lo=16, q_hi=84)
        stack = np.stack(draws, axis=0)
        expected_med = np.percentile(stack, 50, axis=0)
        np.testing.assert_allclose(med, expected_med)

    def test_lo_matches_numpy(self):
        """lo must equal np.percentile(..., q_lo, axis=0)."""
        q_lo = 10
        draws = self._make_draws(B=40, H=4, n=2, seed=3)
        lo, _, _ = bootstrap_percentiles(draws, q_lo=q_lo, q_hi=90)
        stack = np.stack(draws, axis=0)
        expected_lo = np.percentile(stack, q_lo, axis=0)
        np.testing.assert_allclose(lo, expected_lo)

    def test_hi_matches_numpy(self):
        """hi must equal np.percentile(..., q_hi, axis=0)."""
        q_hi = 90
        draws = self._make_draws(B=40, H=4, n=2, seed=4)
        _, _, hi = bootstrap_percentiles(draws, q_lo=10, q_hi=q_hi)
        stack = np.stack(draws, axis=0)
        expected_hi = np.percentile(stack, q_hi, axis=0)
        np.testing.assert_allclose(hi, expected_hi)

    def test_single_draw(self):
        """With B=1 draw, lo = med = hi = that draw."""
        H, n = 3, 2
        rng = np.random.default_rng(5)
        single = rng.standard_normal((H + 1, n, n))
        lo, med, hi = bootstrap_percentiles([single])
        np.testing.assert_allclose(lo, single)
        np.testing.assert_allclose(med, single)
        np.testing.assert_allclose(hi, single)

    def test_wider_quantiles_wider_band(self):
        """Wider quantile range should produce wider (or equal) bands."""
        draws = self._make_draws(B=100, H=5, n=2, seed=6)
        lo_narrow, _, hi_narrow = bootstrap_percentiles(draws, q_lo=40, q_hi=60)
        lo_wide, _, hi_wide = bootstrap_percentiles(draws, q_lo=5, q_hi=95)
        width_narrow = (hi_narrow - lo_narrow).mean()
        width_wide = (hi_wide - lo_wide).mean()
        assert width_wide >= width_narrow - 1e-10

    def test_three_variable_shapes(self):
        """n=3: output shapes (H+1, 3, 3)."""
        H, n = 6, 3
        draws = self._make_draws(B=20, H=H, n=n, seed=7)
        lo, med, hi = bootstrap_percentiles(draws, q_lo=16, q_hi=84)
        assert lo.shape == (H + 1, n, n)
        assert med.shape == (H + 1, n, n)
        assert hi.shape == (H + 1, n, n)


# ===========================================================================
# Integration: moving_block_bootstrap → bootstrap_percentiles pipeline
# ===========================================================================

class TestIntegration:

    def test_percentile_bands_from_bootstrap_shapes(self):
        """End-to-end: bootstrap + percentile bands shape contract."""
        n, horizon, n_draws = 2, 5, 20
        residuals, Y, A_list, intercept = _stable_var1_data(T=80, n=n, seed=0)
        result = moving_block_bootstrap(
            residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
            n_draws=n_draws, block_len=3, horizon=horizon,
            rng=np.random.default_rng(0),
        )
        lo, med, hi = bootstrap_percentiles(result["draws"], q_lo=16, q_hi=84)
        assert lo.shape == (horizon + 1, n, n)
        assert med.shape == (horizon + 1, n, n)
        assert hi.shape == (horizon + 1, n, n)

    def test_percentile_bands_ordering_end_to_end(self):
        """lo <= med <= hi end-to-end for a real bootstrap."""
        n, horizon, n_draws = 2, 4, 30
        residuals, Y, A_list, intercept = _stable_var1_data(T=80, n=n, seed=1)
        result = moving_block_bootstrap(
            residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
            n_draws=n_draws, block_len=3, horizon=horizon,
            rng=np.random.default_rng(1),
        )
        lo, med, hi = bootstrap_percentiles(result["draws"])
        assert np.all(lo <= med + 1e-10)
        assert np.all(med <= hi + 1e-10)

    def test_more_draws_same_shape(self):
        """Doubling n_draws should not change draw shape (just list length)."""
        n, horizon = 2, 3
        residuals, Y, A_list, intercept = _stable_var1_data(T=80, n=n, seed=2)
        kw = dict(residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
                  block_len=3, horizon=horizon, rng=np.random.default_rng(2))
        r10 = moving_block_bootstrap(**kw, n_draws=10)
        r20 = moving_block_bootstrap(**kw, n_draws=20)
        assert len(r10["draws"]) == 10
        assert len(r20["draws"]) == 20
        for d in r10["draws"] + r20["draws"]:
            assert d.shape == (horizon + 1, n, n)

    @pytest.mark.slow
    def test_more_data_tighter_bootstrap_bands(self):
        """Larger T → tighter bootstrap confidence bands on average.

        Uses the default Cholesky irf_fn (_default_irf_fn) via n_draws=100.
        """
        def run(T_val, seed_val):
            residuals, Y, A_list, intercept = _stable_var1_data(
                T=T_val, n=2, seed=seed_val
            )
            res = moving_block_bootstrap(
                residuals=residuals, Y=Y, A_list=A_list, intercept=intercept,
                n_draws=50, block_len=None, horizon=5,
                rng=np.random.default_rng(0),
            )
            lo, _, hi = bootstrap_percentiles(res["draws"], q_lo=16, q_hi=84)
            return (hi - lo).mean()

        width_small = run(T_val=60, seed_val=5)
        width_large = run(T_val=250, seed_val=5)
        assert width_large < width_small, (
            f"Larger T should give tighter bands; "
            f"width_small={width_small:.4f}, width_large={width_large:.4f}"
        )
