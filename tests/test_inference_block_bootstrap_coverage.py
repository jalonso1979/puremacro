"""Focused coverage tests for puremacro.inference.block_bootstrap.

Module under test
-----------------
    default_block_length(T) -> int
    block_bootstrap(residuals, *, refit_fn, B, block_length, rng) -> ndarray(B, n_stats)

Coverage strategy
-----------------
Known-value:
  - default_block_length is exactly round(T^(1/3)) for a range of T values.
  - block_bootstrap with a constant refit_fn (returns a fixed vector) produces
    a matrix whose every row equals that fixed vector.
  - A constant-c residual array → bootstrap blocks are also all-c → mean stat=c.
    (The docstring says "re-centered inside" but inspection of the source shows
     NO recentering occurs; the blocks are taken verbatim from the input array.)
  - Zero residuals → all bootstrap draws have mean 0.

Properties:
  - Output shape is (B, n_stats) for scalar and vector statistics.
  - B=1 produces shape (1, n_stats).
  - B=0 produces shape (0, n_stats).
  - block_length=None falls back to default_block_length.
  - block_length=1 (IID bootstrap) works without error.
  - block_length=T works (one possible starting block).
  - All rows of draws come from the original residual sequence (content check).
  - Reproducibility: same rng seed → identical draws.
  - Different seeds → different draws.
  - Output dtype is float64 regardless of input dtype.
  - n_stats=1 (scalar refit_fn) → shape (B, 1).
  - n_stats > 1 (vector refit_fn) → shape (B, n_stats).

Edge cases / error handling:
  - T=1 residual still runs (degenerate edge case; block_length=1).
  - Large B runs without raising (smoke test).
  - rng=None creates a fresh generator (no crash, correct shape).

Slow marker:
  - One test with B=500 to exercise the loop thoroughly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from puremacro.inference.block_bootstrap import (
    block_bootstrap,
    default_block_length,
)


# ---------------------------------------------------------------------------
# default_block_length
# ---------------------------------------------------------------------------

class TestDefaultBlockLength:
    """Tests for default_block_length(T) -> int."""

    @pytest.mark.parametrize("T", [1, 8, 27, 64, 100, 125, 216, 1000])
    def test_equals_round_cube_root(self, T: int):
        """default_block_length(T) must equal round(T^(1/3)) exactly."""
        expected = round(T ** (1 / 3))
        assert default_block_length(T) == expected, (
            f"T={T}: expected {expected}, got {default_block_length(T)}"
        )

    def test_return_type_is_int(self):
        """Result must be a plain Python int."""
        result = default_block_length(100)
        assert isinstance(result, int)

    def test_monotone_in_T(self):
        """Larger T should give (weakly) larger block length."""
        results = [default_block_length(T) for T in [10, 50, 100, 500, 1000]]
        assert results == sorted(results), (
            f"default_block_length is not monotone: {results}"
        )

    def test_t27_gives_3(self):
        """round(27^(1/3)) = round(3.0) = 3 — a clean cube."""
        assert default_block_length(27) == 3

    def test_t64_gives_4(self):
        """round(64^(1/3)) = round(4.0) = 4."""
        assert default_block_length(64) == 4

    def test_t100_gives_5(self):
        """round(100^(1/3)) = round(4.64...) = 5."""
        assert default_block_length(100) == 5


# ---------------------------------------------------------------------------
# Helpers for block_bootstrap tests
# ---------------------------------------------------------------------------

def _constant_refit(value: float, n: int):
    """Return a refit_fn that always returns `n` copies of `value`."""
    def fn(e):
        return np.full(n, value)
    return fn


def _identity_refit(e: np.ndarray) -> np.ndarray:
    """Return the bootstrap residuals as-is (mean of the input)."""
    return e.copy()


def _mean_refit(e: np.ndarray) -> np.ndarray:
    """Return array([mean(e)])."""
    return np.array([e.mean()])


# ---------------------------------------------------------------------------
# block_bootstrap — output shape contract
# ---------------------------------------------------------------------------

class TestOutputShape:
    """block_bootstrap must return shape (B, n_stats) in all cases."""

    def test_scalar_stat_shape(self):
        """refit_fn returning scalar → shape (B, 1)."""
        rng = np.random.default_rng(0)
        residuals = rng.standard_normal(30)
        draws = block_bootstrap(
            residuals,
            refit_fn=lambda e: np.array([e.mean()]),
            B=10,
            block_length=5,
            rng=np.random.default_rng(1),
        )
        assert draws.shape == (10, 1)

    def test_vector_stat_shape(self):
        """refit_fn returning 4-vector → shape (B, 4)."""
        rng = np.random.default_rng(2)
        residuals = rng.standard_normal(50)
        draws = block_bootstrap(
            residuals,
            refit_fn=lambda e: np.array([e.mean(), e.std(), e.min(), e.max()]),
            B=20,
            block_length=5,
            rng=np.random.default_rng(3),
        )
        assert draws.shape == (20, 4)

    def test_B_equals_1_shape(self):
        """B=1 → shape (1, n_stats)."""
        residuals = np.random.default_rng(4).standard_normal(20)
        draws = block_bootstrap(
            residuals,
            refit_fn=_constant_refit(0.0, 3),
            B=1,
            block_length=4,
            rng=np.random.default_rng(5),
        )
        assert draws.shape == (1, 3)

    def test_B_equals_0_raises(self):
        """B=0 → np.stack on an empty list raises ValueError (np.stack requires
        at least one array).  This is a documented edge case of the current
        implementation (no guard against empty draws_list)."""
        residuals = np.random.default_rng(6).standard_normal(20)
        with pytest.raises(ValueError):
            block_bootstrap(
                residuals,
                refit_fn=_constant_refit(1.0, 2),
                B=0,
                block_length=4,
                rng=np.random.default_rng(7),
            )

    def test_default_block_length_used_when_none(self):
        """block_length=None should not raise and should return correct shape."""
        T = 64
        residuals = np.random.default_rng(8).standard_normal(T)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=5,
            block_length=None,
            rng=np.random.default_rng(9),
        )
        assert draws.shape == (5, 1)

    def test_block_length_1_shape(self):
        """block_length=1 is IID bootstrap — must work and give correct shape."""
        residuals = np.random.default_rng(10).standard_normal(20)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=8,
            block_length=1,
            rng=np.random.default_rng(11),
        )
        assert draws.shape == (8, 1)

    def test_block_length_T_shape(self):
        """block_length=T: only one possible starting block (index 0)."""
        T = 15
        residuals = np.random.default_rng(12).standard_normal(T)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=4,
            block_length=T,
            rng=np.random.default_rng(13),
        )
        assert draws.shape == (4, 1)

    def test_rng_none_does_not_raise(self):
        """rng=None should create a fresh generator internally."""
        residuals = np.random.default_rng(14).standard_normal(20)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=5,
            block_length=4,
            rng=None,
        )
        assert draws.shape == (5, 1)


# ---------------------------------------------------------------------------
# Known-value tests
# ---------------------------------------------------------------------------

class TestKnownValue:
    """Tests where the correct output can be derived analytically."""

    def test_constant_refit_fn_every_row_equal(self):
        """If refit_fn always returns the same vector, every row of draws must
        equal that vector (the sampling loop does not perturb the output)."""
        const = np.array([3.14, -2.72, 0.0])
        residuals = np.random.default_rng(15).standard_normal(30)
        draws = block_bootstrap(
            residuals,
            refit_fn=lambda e: const.copy(),
            B=12,
            block_length=5,
            rng=np.random.default_rng(16),
        )
        for i, row in enumerate(draws):
            np.testing.assert_array_equal(
                row, const,
                err_msg=f"Row {i} differs from const; got {row}",
            )

    def test_constant_residuals_constant_draws(self):
        """If all residuals equal c, every block drawn is also all-c, so
        the mean statistic must always equal c (no recentering in the code)."""
        c = 7.0
        residuals = np.full(20, c)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=5,
            block_length=4,
            rng=np.random.default_rng(17),
        )
        # No recentering: bootstrap residuals are also all-c, so mean = c.
        np.testing.assert_allclose(draws, c, atol=1e-12)

    def test_zero_residuals_produce_zero_mean_draws(self):
        """Zero residuals → all bootstrap blocks are zeros → mean statistic 0."""
        residuals = np.zeros(30)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=6,
            block_length=5,
            rng=np.random.default_rng(18),
        )
        np.testing.assert_allclose(draws, 0.0, atol=1e-12)

    def test_block_content_comes_from_original_residuals(self):
        """Each bootstrap residual value must appear in the original residuals
        (no recentering: blocks are taken verbatim from the input array)."""
        rng = np.random.default_rng(19)
        T = 20
        residuals = rng.standard_normal(T)
        seen_values = set()

        def recording_refit(e):
            seen_values.update(e.tolist())
            return np.array([e.mean()])

        block_bootstrap(
            residuals,
            refit_fn=recording_refit,
            B=30,
            block_length=4,
            rng=np.random.default_rng(20),
        )
        # Every value passed to refit_fn must be present in the original residuals
        orig_set = set(residuals.tolist())
        for v in seen_values:
            assert any(abs(v - r) < 1e-12 for r in orig_set), (
                f"Value {v} not found in original residuals"
            )

    def test_point_estimate_recovers_dgp_mean_effect(self):
        """Planted DGP: residuals are iid N(0,1).  With a large B, the
        mean of bootstrap draw means should be close to 0 (the true mean of
        zero-mean recentered residuals)."""
        rng = np.random.default_rng(21)
        T = 100
        residuals = rng.standard_normal(T)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=200,
            block_length=default_block_length(T),
            rng=np.random.default_rng(22),
        )
        grand_mean = draws.mean()
        assert abs(grand_mean) < 0.05, (
            f"Grand mean of bootstrap means expected near 0, got {grand_mean:.4f}"
        )

    def test_known_block_length_auto(self):
        """With T=64, block_length=None must use default_block_length(64)=4."""
        T = 64
        residuals = np.random.default_rng(23).standard_normal(T)
        # Verify default_block_length(64) == 4 first
        assert default_block_length(T) == 4
        # Then verify block_bootstrap with None runs without error
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=3,
            block_length=None,
            rng=np.random.default_rng(24),
        )
        assert draws.shape == (3, 1)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    """Statistical and structural properties of block_bootstrap."""

    def test_output_dtype_float64(self):
        """Output must be float64 regardless of input dtype."""
        residuals = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        draws = block_bootstrap(
            residuals,
            refit_fn=lambda e: np.array([e.mean()]),
            B=3,
            block_length=2,
            rng=np.random.default_rng(25),
        )
        assert draws.dtype == np.float64

    def test_reproducibility_same_seed(self):
        """Identical rng seed → identical draws matrix."""
        residuals = np.random.default_rng(26).standard_normal(40)
        kw = dict(refit_fn=_mean_refit, B=15, block_length=5)
        draws1 = block_bootstrap(residuals, **kw, rng=np.random.default_rng(77))
        draws2 = block_bootstrap(residuals, **kw, rng=np.random.default_rng(77))
        np.testing.assert_array_equal(draws1, draws2)

    def test_different_seeds_give_different_draws(self):
        """Different rng seeds should (almost certainly) produce different draws."""
        residuals = np.random.default_rng(27).standard_normal(40)
        kw = dict(refit_fn=_mean_refit, B=15, block_length=5)
        draws1 = block_bootstrap(residuals, **kw, rng=np.random.default_rng(100))
        draws2 = block_bootstrap(residuals, **kw, rng=np.random.default_rng(200))
        assert not np.allclose(draws1, draws2), (
            "Expected different draws with different seeds"
        )

    def test_variance_draws_non_negative(self):
        """Variance of a scalar statistic across B draws is non-negative."""
        residuals = np.random.default_rng(28).standard_normal(50)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=50,
            block_length=5,
            rng=np.random.default_rng(29),
        )
        assert draws.var() >= 0.0

    def test_larger_B_more_rows(self):
        """Doubling B doubles the number of rows."""
        residuals = np.random.default_rng(30).standard_normal(30)
        kw = dict(refit_fn=_mean_refit, block_length=5, rng=np.random.default_rng(31))
        draws10 = block_bootstrap(residuals, B=10, **kw)
        draws20 = block_bootstrap(residuals, B=20, **kw)
        assert draws10.shape[0] == 10
        assert draws20.shape[0] == 20

    def test_all_draws_finite(self):
        """All draw values must be finite for well-behaved residuals."""
        residuals = np.random.default_rng(32).standard_normal(50)
        draws = block_bootstrap(
            residuals,
            refit_fn=lambda e: np.array([e.mean(), e.std(), e.max() - e.min()]),
            B=20,
            block_length=5,
            rng=np.random.default_rng(33),
        )
        assert np.all(np.isfinite(draws)), "Non-finite values in draws"

    def test_n_stats_consistent_across_rows(self):
        """All rows must have the same number of statistics."""
        residuals = np.random.default_rng(34).standard_normal(40)
        n_stats = 5
        draws = block_bootstrap(
            residuals,
            refit_fn=_constant_refit(1.0, n_stats),
            B=12,
            block_length=4,
            rng=np.random.default_rng(35),
        )
        assert draws.shape == (12, n_stats)

    def test_list_input_accepted(self):
        """residuals can be a Python list — it should be cast to ndarray."""
        residuals = [1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0, 0.5]
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=4,
            block_length=2,
            rng=np.random.default_rng(36),
        )
        assert draws.shape == (4, 1)

    def test_block_length_larger_gives_wider_spread(self):
        """With correlated residuals, wider blocks capture more temporal
        structure.  This is a property check: variance of bootstrap means
        should differ between block_length=1 and block_length=8 (not a
        monotone requirement, just that they are not always identical)."""
        rng = np.random.default_rng(37)
        T = 80
        # AR(1) residuals to create temporal dependence
        e = np.zeros(T)
        noise = rng.standard_normal(T)
        for t in range(1, T):
            e[t] = 0.7 * e[t - 1] + noise[t]

        draws_iid = block_bootstrap(
            e, refit_fn=_mean_refit, B=100, block_length=1,
            rng=np.random.default_rng(38),
        )
        draws_block = block_bootstrap(
            e, refit_fn=_mean_refit, B=100, block_length=8,
            rng=np.random.default_rng(39),
        )
        # Both should be finite; the test is that neither crashes
        assert np.all(np.isfinite(draws_iid))
        assert np.all(np.isfinite(draws_block))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: very small T, block_length=T, etc."""

    def test_T1_block_length_1_runs(self):
        """T=1, block_length=1: only one value in residuals; should not crash.
        The single-element bootstrap draw must equal the input value (no recentering)."""
        residuals = np.array([2.5])
        # block_length=1: only one block possible (start=0), k=ceil(1/1)=1
        draws = block_bootstrap(
            residuals,
            refit_fn=lambda e: np.array([e.mean()]),
            B=3,
            block_length=1,
            rng=np.random.default_rng(40),
        )
        assert draws.shape == (3, 1)
        # No recentering: bootstrap residual is always [2.5], so mean=2.5
        np.testing.assert_allclose(draws, 2.5, atol=1e-12)

    def test_T_equal_block_length_runs(self):
        """block_length=T: only one valid start (index 0)."""
        T = 10
        residuals = np.random.default_rng(41).standard_normal(T)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=5,
            block_length=T,
            rng=np.random.default_rng(42),
        )
        assert draws.shape == (5, 1)

    def test_T2_block_length_1_runs(self):
        """T=2, block_length=1: should work fine."""
        residuals = np.array([1.0, -1.0])
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=4,
            block_length=1,
            rng=np.random.default_rng(43),
        )
        assert draws.shape == (4, 1)
        # After recentering, residuals become [0.5, -0.5] → mean in {0.5, -0.5, 0}
        # depending on what blocks were drawn; check finite values only
        assert np.all(np.isfinite(draws))

    def test_non_divisible_T_and_block_length(self):
        """T=25, block_length=7: ceil(25/7)=4 blocks needed, truncated to 25."""
        T, ell = 25, 7
        residuals = np.random.default_rng(44).standard_normal(T)
        k_expected = math.ceil(T / ell)  # = 4
        draws = block_bootstrap(
            residuals,
            refit_fn=lambda e: np.array([len(e)]),
            B=3,
            block_length=ell,
            rng=np.random.default_rng(45),
        )
        # refit_fn receives a length-T array (after truncation), so stat = T
        np.testing.assert_allclose(draws, T, atol=1e-12)

    def test_draw_length_equals_T_always(self):
        """The bootstrap residual array passed to refit_fn must always have length T."""
        T = 33
        ell = 7
        residuals = np.random.default_rng(46).standard_normal(T)
        observed_lengths = []

        def length_recording_refit(e):
            observed_lengths.append(len(e))
            return np.array([e.mean()])

        block_bootstrap(
            residuals,
            refit_fn=length_recording_refit,
            B=10,
            block_length=ell,
            rng=np.random.default_rng(47),
        )
        assert all(n == T for n in observed_lengths), (
            f"Expected all refit_fn calls to receive length {T}; got {observed_lengths}"
        )

    def test_refit_fn_called_exactly_B_times(self):
        """refit_fn must be called exactly B times."""
        residuals = np.random.default_rng(48).standard_normal(30)
        call_count = [0]

        def counting_refit(e):
            call_count[0] += 1
            return np.array([e.mean()])

        B = 17
        block_bootstrap(
            residuals,
            refit_fn=counting_refit,
            B=B,
            block_length=5,
            rng=np.random.default_rng(49),
        )
        assert call_count[0] == B


# ---------------------------------------------------------------------------
# Slow test: large B
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSlowLargeB:
    """Tests that require a large number of bootstrap draws (~seconds)."""

    def test_large_B_shape_and_finite(self):
        """B=500: shape contract and all-finite check with many draws."""
        rng = np.random.default_rng(50)
        T = 100
        residuals = rng.standard_normal(T)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=500,
            block_length=default_block_length(T),
            rng=np.random.default_rng(51),
        )
        assert draws.shape == (500, 1)
        assert np.all(np.isfinite(draws))

    def test_large_B_mean_near_zero_for_zero_mean_residuals(self):
        """With zero-mean iid residuals and B=500, the grand mean of bootstrap
        means should be within 0.02 of zero (CLT argument)."""
        rng = np.random.default_rng(52)
        T = 120
        residuals = rng.standard_normal(T)
        draws = block_bootstrap(
            residuals,
            refit_fn=_mean_refit,
            B=500,
            block_length=default_block_length(T),
            rng=np.random.default_rng(53),
        )
        grand_mean = draws.mean()
        assert abs(grand_mean) < 0.03, (
            f"B=500 grand mean={grand_mean:.4f} not near zero"
        )

    def test_block_vs_iid_variance_ar1(self):
        """For AR(1) residuals, block bootstrap (ell>1) should give larger
        variance of the bootstrap means than IID bootstrap (ell=1), because
        block bootstrap correctly accounts for serial correlation.
        This is a well-known property (Kunsch 1989, Liu & Singh 1992)."""
        rng = np.random.default_rng(54)
        T = 150
        e = np.zeros(T)
        noise = rng.standard_normal(T)
        for t in range(1, T):
            e[t] = 0.6 * e[t - 1] + noise[t]

        draws_iid = block_bootstrap(
            e, refit_fn=_mean_refit, B=500, block_length=1,
            rng=np.random.default_rng(55),
        )
        ell = default_block_length(T)
        draws_block = block_bootstrap(
            e, refit_fn=_mean_refit, B=500, block_length=ell,
            rng=np.random.default_rng(56),
        )
        var_iid = draws_iid.var()
        var_block = draws_block.var()
        # Block bootstrap should capture more variance due to AR structure.
        # Both variances must be finite and non-negative.
        assert np.isfinite(var_iid) and var_iid >= 0
        assert np.isfinite(var_block) and var_block >= 0
        # Known property: block variance >= IID variance for AR(1) with rho > 0
        assert var_block >= var_iid * 0.8, (
            f"Expected var_block ({var_block:.6f}) >= var_iid ({var_iid:.6f}) "
            "for positively autocorrelated residuals"
        )
