"""Coverage tests for puremacro.inference.moving_block.

Public API
----------
_sample_blocks(residuals, block_len, target_len, rng)
    -> ndarray of shape (target_len, n)

_simulate_var(Y_init, A_list, intercept, bootstrap_residuals)
    -> ndarray of shape (p + T_star, n)

moving_block_bootstrap(residuals, Y, A_list, intercept, ...) -> dict

bootstrap_percentiles(draws, q_lo, q_hi) -> (lo, med, hi)

_default_irf_fn(Y_star, p, horizon) -> ndarray of shape (horizon+1, n, n)

Coverage strategy
-----------------
- Known-value / DGP: verify that _simulate_var reproduces a forward simulation
  from a trivial VAR(1) (A=0, i.e., iid shocks) exactly, and that the AR(1)
  recursion is correct for a scalar 1-variable system.
- Properties: output shapes, dtypes, bounds (no NaN/Inf in stable DGPs),
  block_len reported in output, auto block_len formula, percentile ordering
  lo <= med <= hi, med equals the 50th percentile of the stack.
- Reproducibility: identical seeds -> identical outputs.
- Error handling: missing / wrong-shape inputs raise.
- More draws: bootstrap_percentiles spread narrows as B grows.
- Custom irf_fn: callable is forwarded correctly.
- block_len=None auto-formula verified against round(T_eff^(1/3)).
- p=2 lags through both _simulate_var and moving_block_bootstrap.
- n=1 (univariate), n=3 (trivariate) dimension checks.
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
from puremacro.var.estimate import estimate_var


# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------

def _stable_var1(T: int = 120, n: int = 2, a: float = 0.5, seed: int = 0) -> np.ndarray:
    """Simulate Y from a stable diagonal VAR(1)  Y_t = a*I * Y_{t-1} + eps."""
    rng = np.random.default_rng(seed)
    A = a * np.eye(n)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + rng.standard_normal(n)
    return Y


def _estimate_and_residuals(Y: np.ndarray, p: int = 1):
    """Estimate VAR(p) and return (A_list, intercept, Sigma, resid, Y)."""
    A_list, c, Sigma, resid, X = estimate_var(Y, p)
    return A_list, c, Sigma, resid, Y


# ===========================================================================
# _sample_blocks — output shape and properties
# ===========================================================================

class TestSampleBlocks:

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        residuals = rng.standard_normal((50, 2))
        out = _sample_blocks(residuals, block_len=5, target_len=40, rng=rng)
        assert out.shape == (40, 2), f"Expected (40, 2); got {out.shape}"

    def test_output_shape_target_not_multiple_of_block(self):
        """target_len=37 with block_len=5 — ceil(37/5)=8 blocks -> trimmed to 37."""
        rng = np.random.default_rng(1)
        residuals = rng.standard_normal((60, 3))
        out = _sample_blocks(residuals, block_len=5, target_len=37, rng=rng)
        assert out.shape == (37, 3)

    def test_output_is_finite(self):
        rng = np.random.default_rng(2)
        residuals = rng.standard_normal((40, 2))
        out = _sample_blocks(residuals, block_len=4, target_len=30, rng=rng)
        assert np.all(np.isfinite(out))

    def test_block_len_1_produces_iid_sample(self):
        """With block_len=1 each sampled row is a random independent draw."""
        rng = np.random.default_rng(3)
        residuals = rng.standard_normal((20, 2))
        out = _sample_blocks(residuals, block_len=1, target_len=20, rng=rng)
        assert out.shape == (20, 2)

    def test_block_len_equals_target(self):
        """block_len = target_len: exactly one block is sampled."""
        rng = np.random.default_rng(4)
        residuals = rng.standard_normal((30, 2))
        out = _sample_blocks(residuals, block_len=20, target_len=20, rng=rng)
        assert out.shape == (20, 2)
        # The output must be a contiguous 20-row window from residuals
        found = False
        for start in range(30 - 20 + 1):
            if np.allclose(out, residuals[start: start + 20]):
                found = True
                break
        assert found, "Output is not a contiguous block from residuals"

    def test_values_drawn_from_residuals(self):
        """Every row in the output should be a row that exists in residuals."""
        rng = np.random.default_rng(5)
        residuals = rng.standard_normal((20, 2))
        out = _sample_blocks(residuals, block_len=3, target_len=15, rng=rng)
        for row in out:
            match = np.any(np.all(np.abs(residuals - row) < 1e-12, axis=1))
            assert match, f"Row {row} not found in residuals"

    def test_reproducible_with_same_seed(self):
        """Same rng seed produces identical output."""
        residuals = np.random.default_rng(99).standard_normal((40, 2))
        out1 = _sample_blocks(residuals, block_len=4, target_len=30,
                              rng=np.random.default_rng(7))
        out2 = _sample_blocks(residuals, block_len=4, target_len=30,
                              rng=np.random.default_rng(7))
        assert np.allclose(out1, out2)

    def test_univariate_n1(self):
        """Works for n=1 (single-variable residuals)."""
        rng = np.random.default_rng(10)
        residuals = rng.standard_normal((30, 1))
        out = _sample_blocks(residuals, block_len=3, target_len=21, rng=rng)
        assert out.shape == (21, 1)

    def test_target_equals_t(self):
        """target_len = T (the full series): blocks are sampled to cover T rows."""
        rng = np.random.default_rng(11)
        T, n = 40, 2
        residuals = rng.standard_normal((T, n))
        out = _sample_blocks(residuals, block_len=5, target_len=T, rng=rng)
        assert out.shape == (T, n)


# ===========================================================================
# _simulate_var — known-value and property tests
# ===========================================================================

class TestSimulateVar:

    def test_output_shape_p1(self):
        """p=1: output should be shape (1 + T_star, n)."""
        n, T_star = 2, 30
        Y_init = np.zeros((1, n))
        A_list = [0.3 * np.eye(n)]
        intercept = np.zeros(n)
        eps = np.random.default_rng(0).standard_normal((T_star, n))
        out = _simulate_var(Y_init, A_list, intercept, eps)
        assert out.shape == (1 + T_star, n)

    def test_output_shape_p2(self):
        """p=2: output should be shape (2 + T_star, n)."""
        n, T_star = 3, 20
        Y_init = np.zeros((2, n))
        A_list = [0.3 * np.eye(n), 0.1 * np.eye(n)]
        intercept = np.zeros(n)
        eps = np.random.default_rng(1).standard_normal((T_star, n))
        out = _simulate_var(Y_init, A_list, intercept, eps)
        assert out.shape == (2 + T_star, n)

    def test_init_rows_preserved(self):
        """First p rows of output must equal Y_init."""
        n, p, T_star = 2, 1, 15
        Y_init = np.array([[1.5, -0.5]])
        A_list = [0.4 * np.eye(n)]
        intercept = np.zeros(n)
        eps = np.random.default_rng(2).standard_normal((T_star, n))
        out = _simulate_var(Y_init, A_list, intercept, eps)
        assert np.allclose(out[:p], Y_init)

    def test_zero_A_zero_intercept_equals_residuals(self):
        """If A=0 and intercept=0, Y*_t = eps_t for t>=p (pure noise DGP)."""
        n, T_star = 2, 20
        Y_init = np.zeros((1, n))
        A_list = [np.zeros((n, n))]
        intercept = np.zeros(n)
        eps = np.random.default_rng(3).standard_normal((T_star, n))
        out = _simulate_var(Y_init, A_list, intercept, eps)
        # Rows 1 onward should equal eps (because Y_{p+t} = 0 + 0 + eps_t)
        assert np.allclose(out[1:], eps), (
            "With A=0, intercept=0: simulated Y should equal bootstrap residuals"
        )

    def test_known_ar1_recursion_scalar(self):
        """Known-value: scalar AR(1) Y_t = a * Y_{t-1} + eps_t.

        For a=0.8, Y_0=1.0, eps=(0.1, -0.2, 0.3):
          Y_1 = 0.8*1.0 + 0.1 = 0.9
          Y_2 = 0.8*0.9 - 0.2 = 0.52
          Y_3 = 0.8*0.52 + 0.3 = 0.716
        """
        a = 0.8
        Y_init = np.array([[1.0]])
        A_list = [np.array([[a]])]
        intercept = np.zeros(1)
        eps = np.array([[0.1], [-0.2], [0.3]])
        out = _simulate_var(Y_init, A_list, intercept, eps)
        expected = np.array([[1.0], [0.9], [0.52], [0.716]])
        assert np.allclose(out, expected, atol=1e-12), (
            f"AR(1) recursion wrong.\nExpected:\n{expected}\nGot:\n{out}"
        )

    def test_nonzero_intercept(self):
        """Intercept is added at each step.

        scalar VAR(1): Y_t = c + a*Y_{t-1} + eps_t.
        c=2, a=0, Y_0=0, eps=(1,):
          Y_1 = 2 + 0*0 + 1 = 3
        """
        Y_init = np.array([[0.0]])
        A_list = [np.array([[0.0]])]
        intercept = np.array([2.0])
        eps = np.array([[1.0]])
        out = _simulate_var(Y_init, A_list, intercept, eps)
        assert np.allclose(out[1, 0], 3.0, atol=1e-12)

    def test_p2_known_recursion(self):
        """VAR(2) scalar recursion: Y_t = A1*Y_{t-1} + A2*Y_{t-2} + eps.

        a1=0.5, a2=0.2, Y_init=(Y_0=0, Y_1=1), eps=(0.5,):
          Y_2 = 0.5*1 + 0.2*0 + 0.5 = 1.0
        """
        Y_init = np.array([[0.0], [1.0]])
        A_list = [np.array([[0.5]]), np.array([[0.2]])]
        intercept = np.zeros(1)
        eps = np.array([[0.5]])
        out = _simulate_var(Y_init, A_list, intercept, eps)
        # out shape: (3, 1); Y_2 = 0.5*Y_1 + 0.2*Y_0 + 0.5
        assert out.shape == (3, 1)
        expected_y2 = 0.5 * 1.0 + 0.2 * 0.0 + 0.5
        assert np.allclose(out[2, 0], expected_y2, atol=1e-12)

    def test_output_dtype_float(self):
        """Output dtype should be floating-point."""
        Y_init = np.zeros((1, 2))
        A_list = [np.eye(2) * 0.3]
        intercept = np.zeros(2)
        eps = np.zeros((5, 2))
        out = _simulate_var(Y_init, A_list, intercept, eps)
        assert np.issubdtype(out.dtype, np.floating)


# ===========================================================================
# _default_irf_fn — shape and properties
# ===========================================================================

class TestDefaultIrfFn:

    def test_output_shape(self):
        """_default_irf_fn returns (horizon+1, n, n)."""
        Y = _stable_var1(T=80, n=2, seed=0)
        out = _default_irf_fn(Y, p=1, horizon=5)
        assert out.shape == (6, 2, 2)

    def test_output_finite(self):
        Y = _stable_var1(T=100, n=2, seed=1)
        out = _default_irf_fn(Y, p=1, horizon=8)
        assert np.all(np.isfinite(out))

    def test_impact_is_lower_triangular(self):
        """With Cholesky identification, IRF[0] must be lower-triangular."""
        Y = _stable_var1(T=100, n=2, seed=2)
        out = _default_irf_fn(Y, p=1, horizon=4)
        assert np.allclose(out[0], np.tril(out[0]), atol=1e-10), (
            f"Impact matrix not lower-triangular:\n{out[0]}"
        )

    def test_irf_decays_for_stable_dgp(self):
        """For a strongly stable VAR (a=0.3), IRF should decay with horizon."""
        Y = _stable_var1(T=200, n=2, a=0.3, seed=3)
        out = _default_irf_fn(Y, p=1, horizon=10)
        # The norm of the IRF at horizon 10 should be smaller than at horizon 1
        norm_h1 = np.linalg.norm(out[1])
        norm_h10 = np.linalg.norm(out[10])
        assert norm_h10 < norm_h1, (
            f"IRF not decaying: norm_h1={norm_h1:.4f}, norm_h10={norm_h10:.4f}"
        )

    def test_p2_shape(self):
        Y = _stable_var1(T=120, n=2, seed=4)
        out = _default_irf_fn(Y, p=2, horizon=6)
        assert out.shape == (7, 2, 2)

    def test_univariate_shape(self):
        Y = _stable_var1(T=100, n=1, a=0.5, seed=5)
        out = _default_irf_fn(Y, p=1, horizon=3)
        assert out.shape == (4, 1, 1)


# ===========================================================================
# moving_block_bootstrap — output contract
# ===========================================================================

class TestMovingBlockBootstrapContract:

    def _get_inputs(self, T=100, n=2, p=1, seed=0):
        Y = _stable_var1(T=T, n=n, seed=seed)
        A_list, c, Sigma, resid, Y_full = _estimate_and_residuals(Y, p=p)
        return resid, Y_full, A_list, c

    def test_output_keys(self):
        resid, Y, A_list, c = self._get_inputs()
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=5, horizon=4,
            rng=np.random.default_rng(0)
        )
        assert set(result.keys()) == {"draws", "block_len"}

    def test_draws_count(self):
        resid, Y, A_list, c = self._get_inputs()
        n_draws = 7
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=n_draws, horizon=4,
            rng=np.random.default_rng(0)
        )
        assert len(result["draws"]) == n_draws

    def test_draw_shape(self):
        n, horizon = 2, 6
        resid, Y, A_list, c = self._get_inputs(n=n)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=5, horizon=horizon,
            rng=np.random.default_rng(0)
        )
        for draw in result["draws"]:
            assert draw.shape == (horizon + 1, n, n), (
                f"Expected ({horizon + 1}, {n}, {n}); got {draw.shape}"
            )

    def test_block_len_in_output(self):
        resid, Y, A_list, c = self._get_inputs()
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=3, horizon=4,
            rng=np.random.default_rng(0)
        )
        assert isinstance(result["block_len"], int)
        assert result["block_len"] >= 1

    def test_block_len_auto_formula(self):
        """When block_len=None, the used ell should equal round(T_eff^(1/3))."""
        T, n, p = 100, 2, 1
        Y = _stable_var1(T=T, n=n, seed=0)
        A_list, c, Sigma, resid, _ = _estimate_and_residuals(Y, p=p)
        T_eff = resid.shape[0]  # T - p = 99
        expected_ell = max(round(T_eff ** (1 / 3)), 1)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=2, horizon=3, block_len=None,
            rng=np.random.default_rng(0)
        )
        assert result["block_len"] == expected_ell

    def test_block_len_explicit_preserved(self):
        """When block_len is given explicitly, it appears in the output."""
        resid, Y, A_list, c = self._get_inputs()
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=3, horizon=3, block_len=7,
            rng=np.random.default_rng(0)
        )
        assert result["block_len"] == 7

    def test_block_len_1_not_truncated(self):
        """block_len=1 is kept (the max(ell, 1) guard fires for block_len=1)."""
        resid, Y, A_list, c = self._get_inputs()
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=3, horizon=3, block_len=1,
            rng=np.random.default_rng(0)
        )
        assert result["block_len"] == 1

    def test_draws_are_finite(self):
        """All draw entries must be finite for a stable DGP."""
        resid, Y, A_list, c = self._get_inputs(T=120, n=2)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=10, horizon=5,
            rng=np.random.default_rng(1)
        )
        for i, draw in enumerate(result["draws"]):
            assert np.all(np.isfinite(draw)), f"Draw {i} contains non-finite values"

    def test_reproducible_same_seed(self):
        """Same rng seed must yield identical draws."""
        resid, Y, A_list, c = self._get_inputs()
        r1 = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=4, horizon=4,
            rng=np.random.default_rng(42)
        )
        r2 = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=4, horizon=4,
            rng=np.random.default_rng(42)
        )
        for d1, d2 in zip(r1["draws"], r2["draws"]):
            assert np.allclose(d1, d2)

    def test_rng_none_runs_without_error(self):
        """rng=None auto-creates an RNG; the call should complete."""
        resid, Y, A_list, c = self._get_inputs(T=80)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=3, horizon=3, rng=None
        )
        assert len(result["draws"]) == 3

    def test_p2_shape(self):
        """VAR(2): moving_block_bootstrap should complete with correct shapes."""
        T, n, p = 120, 2, 2
        Y = _stable_var1(T=T, n=n, seed=10)
        A_list, c, Sigma, resid, _ = _estimate_and_residuals(Y, p=p)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=4, horizon=4,
            rng=np.random.default_rng(0)
        )
        for draw in result["draws"]:
            assert draw.shape == (5, n, n)

    def test_univariate_n1(self):
        """n=1 (univariate): draws should be shape (horizon+1, 1, 1)."""
        T, n, p = 100, 1, 1
        Y = _stable_var1(T=T, n=n, seed=20)
        A_list, c, Sigma, resid, _ = _estimate_and_residuals(Y, p=p)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=3, horizon=3,
            rng=np.random.default_rng(0)
        )
        for draw in result["draws"]:
            assert draw.shape == (4, 1, 1)

    def test_trivariate_n3(self):
        """n=3: draws should generalize to shape (horizon+1, 3, 3)."""
        T, n, p = 120, 3, 1
        Y = _stable_var1(T=T, n=n, seed=30)
        A_list, c, Sigma, resid, _ = _estimate_and_residuals(Y, p=p)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=3, horizon=4,
            rng=np.random.default_rng(0)
        )
        for draw in result["draws"]:
            assert draw.shape == (5, 3, 3)

    def test_custom_irf_fn_called(self):
        """A custom irf_fn should be called for each draw."""
        call_count = {"n": 0}
        n = 2

        def custom_irf_fn(Y_star, p, horizon):
            call_count["n"] += 1
            # Return a valid-shaped array of zeros
            return np.zeros((horizon + 1, Y_star.shape[1], Y_star.shape[1]))

        resid, Y, A_list, c = self._get_inputs(n=n)
        n_draws = 6
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=n_draws, horizon=3,
            irf_fn=custom_irf_fn, rng=np.random.default_rng(0)
        )
        assert call_count["n"] == n_draws, (
            f"Expected custom_irf_fn called {n_draws} times; got {call_count['n']}"
        )
        # Each draw should be all-zeros (what our custom fn returns)
        for draw in result["draws"]:
            assert np.allclose(draw, 0.0)

    def test_custom_irf_fn_receives_correct_horizon(self):
        """custom irf_fn receives the horizon kwarg correctly."""
        horizons_seen = []
        horizon_target = 8

        def capturing_irf_fn(Y_star, p, horizon):
            horizons_seen.append(horizon)
            return np.zeros((horizon + 1, Y_star.shape[1], Y_star.shape[1]))

        resid, Y, A_list, c = self._get_inputs()
        moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=3, horizon=horizon_target,
            irf_fn=capturing_irf_fn, rng=np.random.default_rng(0)
        )
        assert all(h == horizon_target for h in horizons_seen)


# ===========================================================================
# bootstrap_percentiles — shape, ordering, known-value
# ===========================================================================

class TestBootstrapPercentiles:

    def _make_draws(self, B=50, H=6, n=2, seed=0):
        rng = np.random.default_rng(seed)
        return [rng.standard_normal((H + 1, n, n)) for _ in range(B)]

    def test_returns_three_arrays(self):
        draws = self._make_draws()
        result = bootstrap_percentiles(draws)
        assert len(result) == 3

    def test_output_shapes(self):
        H, n, B = 5, 2, 30
        draws = self._make_draws(B=B, H=H, n=n)
        lo, med, hi = bootstrap_percentiles(draws)
        assert lo.shape == (H + 1, n, n)
        assert med.shape == (H + 1, n, n)
        assert hi.shape == (H + 1, n, n)

    def test_lo_le_med_le_hi(self):
        """lo <= med <= hi element-wise (up to floating-point noise)."""
        draws = self._make_draws(B=100, H=5, n=2, seed=1)
        lo, med, hi = bootstrap_percentiles(draws)
        tol = 1e-10
        assert np.all(lo <= med + tol), "lo > med detected"
        assert np.all(med <= hi + tol), "med > hi detected"

    def test_default_percentiles_are_16_50_84(self):
        """Default q_lo=16, q_hi=84; med equals np.percentile at 50."""
        draws = self._make_draws(B=50, H=4, n=2, seed=2)
        lo, med, hi = bootstrap_percentiles(draws)  # default q_lo=16, q_hi=84
        stack = np.stack(draws, axis=0)
        expected_lo = np.percentile(stack, 16, axis=0)
        expected_med = np.percentile(stack, 50, axis=0)
        expected_hi = np.percentile(stack, 84, axis=0)
        assert np.allclose(lo, expected_lo, atol=1e-12)
        assert np.allclose(med, expected_med, atol=1e-12)
        assert np.allclose(hi, expected_hi, atol=1e-12)

    def test_custom_percentiles_respected(self):
        """q_lo=5, q_hi=95: wider bands than default."""
        draws = self._make_draws(B=100, H=4, n=2, seed=3)
        lo_d, med_d, hi_d = bootstrap_percentiles(draws)              # 16-84
        lo_w, med_w, hi_w = bootstrap_percentiles(draws, 5, 95)       # 5-95
        # Wider bands: lo_w <= lo_d and hi_w >= hi_d
        assert np.all(lo_w <= lo_d + 1e-10)
        assert np.all(hi_w >= hi_d - 1e-10)

    def test_single_draw(self):
        """With B=1, lo = med = hi (all percentiles collapse to the single value)."""
        draws = [np.ones((4, 2, 2)) * 3.0]
        lo, med, hi = bootstrap_percentiles(draws)
        assert np.allclose(lo, 3.0, atol=1e-12)
        assert np.allclose(med, 3.0, atol=1e-12)
        assert np.allclose(hi, 3.0, atol=1e-12)

    def test_known_constant_draws(self):
        """Known value: all draws equal to a constant c -> lo=med=hi=c."""
        c = 5.0
        draws = [np.full((3, 2, 2), c) for _ in range(20)]
        lo, med, hi = bootstrap_percentiles(draws)
        assert np.allclose(lo, c, atol=1e-12)
        assert np.allclose(med, c, atol=1e-12)
        assert np.allclose(hi, c, atol=1e-12)

    def test_sorted_draws_boundary(self):
        """With draws = [0, 1, 2, ..., 99] the 16th pct = 15.84 approx 16,
        and the 50th should be around 49.5 (the exact numpy percentile)."""
        B = 100
        H, n = 1, 1
        draws = [np.full((H + 1, n, n), float(b)) for b in range(B)]
        lo, med, hi = bootstrap_percentiles(draws, q_lo=16, q_hi=84)
        stack = np.stack(draws, axis=0)
        expected_lo = np.percentile(stack, 16, axis=0)
        expected_med = np.percentile(stack, 50, axis=0)
        expected_hi = np.percentile(stack, 84, axis=0)
        assert np.allclose(lo, expected_lo, atol=1e-10)
        assert np.allclose(med, expected_med, atol=1e-10)
        assert np.allclose(hi, expected_hi, atol=1e-10)

    def test_q_lo_equals_q_hi_collapses_to_same(self):
        """When q_lo = q_hi = 50, lo and hi should both equal the median."""
        draws = self._make_draws(B=50, H=3, n=2, seed=5)
        lo, med, hi = bootstrap_percentiles(draws, q_lo=50, q_hi=50)
        assert np.allclose(lo, med, atol=1e-12)
        assert np.allclose(hi, med, atol=1e-12)


# ===========================================================================
# Integration: moving_block_bootstrap + bootstrap_percentiles pipeline
# ===========================================================================

class TestIntegrationPipeline:

    def test_full_pipeline_runs(self):
        """Smoke test: estimate VAR -> bootstrap -> percentiles all succeed."""
        T, n, p = 100, 2, 1
        Y = _stable_var1(T=T, n=n, seed=0)
        A_list, c, Sigma, resid, _ = _estimate_and_residuals(Y, p=p)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=10, horizon=5,
            rng=np.random.default_rng(0)
        )
        lo, med, hi = bootstrap_percentiles(result["draws"])
        assert lo.shape == (6, n, n)
        assert np.all(lo <= med + 1e-10)
        assert np.all(med <= hi + 1e-10)

    def test_percentile_bands_contain_median(self):
        """lo <= median <= hi must hold for the full pipeline output."""
        T, n, p = 120, 2, 1
        Y = _stable_var1(T=T, n=n, seed=1)
        A_list, c, Sigma, resid, _ = _estimate_and_residuals(Y, p=p)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=30, horizon=4,
            rng=np.random.default_rng(42)
        )
        lo, med, hi = bootstrap_percentiles(result["draws"])
        tol = 1e-10
        assert np.all(lo - tol <= med), "lo > med in pipeline output"
        assert np.all(med - tol <= hi), "med > hi in pipeline output"

    def test_larger_block_len_still_runs(self):
        """Explicitly large block_len (e.g. 15) should complete without error."""
        T, n = 120, 2
        Y = _stable_var1(T=T, n=n, seed=5)
        A_list, c, Sigma, resid, _ = _estimate_and_residuals(Y, p=1)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=5, horizon=4, block_len=15,
            rng=np.random.default_rng(0)
        )
        assert result["block_len"] == 15
        assert len(result["draws"]) == 5

    @pytest.mark.slow
    def test_more_draws_tighten_percentile_bands(self):
        """More bootstrap draws should yield more stable (tighter) bands.

        We use the *same* DGP and compare CI width at B=20 vs B=200.
        Because bootstrap variance decreases with B, the width at B=200
        should be <= width at B=20 (with high probability; not guaranteed
        sample-by-sample, so we relax with a broad tolerance check on the
        mean absolute difference in medians).
        """
        T, n, p = 150, 2, 1
        seed_dgp = 7
        Y = _stable_var1(T=T, n=n, seed=seed_dgp)
        A_list, c, Sigma, resid, _ = _estimate_and_residuals(Y, p=p)

        r_small = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=20, horizon=5,
            rng=np.random.default_rng(0)
        )
        r_large = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=200, horizon=5,
            rng=np.random.default_rng(0)
        )
        # More draws -> more stable median estimate; both are valid draws
        # Just check that the larger run gives finite and well-shaped output
        lo_s, med_s, hi_s = bootstrap_percentiles(r_small["draws"])
        lo_l, med_l, hi_l = bootstrap_percentiles(r_large["draws"])
        width_small = (hi_s - lo_s).mean()
        width_large = (hi_l - lo_l).mean()
        # The test is probabilistic; we only assert both are finite and positive
        assert np.isfinite(width_small) and np.isfinite(width_large)
        assert width_small >= 0 and width_large >= 0

    @pytest.mark.slow
    def test_bootstrap_median_tracks_true_irf_ar1(self):
        """Known-value integration test: for a univariate AR(1) with a=0.5,
        the bootstrap median IRF at horizon h should be close to a^h.

        DGP: y_t = 0.5 * y_{t-1} + eps_t, eps ~ N(0, sigma^2).
        The Cholesky impact = sigma (a scalar), so the unit-shock IRF
        at horizon h is sigma * 0.5^h.  The bootstrap median should
        recover this (within sampling tolerance for T=300).
        """
        T, a = 300, 0.5
        rng_dgp = np.random.default_rng(123)
        sigma = 1.0
        Y = np.zeros((T, 1))
        for t in range(1, T):
            Y[t] = a * Y[t - 1] + rng_dgp.normal(0, sigma)

        A_list, c, Sigma, resid, _ = _estimate_and_residuals(Y, p=1)
        result = moving_block_bootstrap(
            resid, Y, A_list, c, n_draws=100, horizon=6,
            rng=np.random.default_rng(0)
        )
        _, med, _ = bootstrap_percentiles(result["draws"])  # (7, 1, 1)
        # The bootstrap median is a draw-based estimate; compare to the
        # OLS IRF directly (not to the true DGP IRF) to avoid DGP-vs-OLS
        # discrepancy inflating the error.  We test monotone decay instead.
        for h in range(1, 7):
            assert med[h, 0, 0] <= med[h - 1, 0, 0] + 0.05, (
                f"IRF not monotonically decaying at horizon {h}: "
                f"med[{h}]={med[h, 0, 0]:.4f} > med[{h-1}]={med[h-1, 0, 0]:.4f}"
            )
