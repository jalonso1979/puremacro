"""Meaningful coverage tests for puremacro.posterior.

Covers both public functions:
- simulate_var_paths   — output shape, initial condition, VAR(1) DGP recovery,
                         deterministic (zero-noise) path, multi-lag, rng seeding.
- fan_chart_quantiles  — output shape, quantile ordering (monotonicity),
                         bounds, median recovery for known DGP, custom levels.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.posterior import simulate_var_paths, fan_chart_quantiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_var1_draws(
    D: int,
    n: int,
    A_diag: float = 0.5,
    sigma_diag: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build constant (D, p, n, n) / (D, n, n) / (D, n) draws for a VAR(1).

    All draws share the same AR coefficient matrix (diagonal A_diag) and
    error covariance (diagonal sigma_diag^2) so results are predictable.
    """
    rng = np.random.default_rng(seed)
    A_draws = np.tile(A_diag * np.eye(n), (D, 1, 1, 1))  # (D, 1, n, n)
    Sigma_draws = np.tile((sigma_diag ** 2) * np.eye(n), (D, 1, 1))  # (D, n, n)
    intercept_draws = np.zeros((D, n))
    y_init = rng.standard_normal((1, n))
    return A_draws, Sigma_draws, intercept_draws, y_init


# ---------------------------------------------------------------------------
# simulate_var_paths — shape tests
# ---------------------------------------------------------------------------

class TestSimulateVarPathsShapes:

    def test_basic_shape(self):
        """Output must be (D, H+1, n)."""
        D, n, horizon = 20, 2, 5
        A, S, c, y0 = _make_var1_draws(D, n)
        rng = np.random.default_rng(0)
        paths = simulate_var_paths(A, S, c, y0, horizon, rng=rng)
        assert paths.shape == (D, horizon + 1, n)

    def test_horizon_zero_shape(self):
        """horizon=0 => shape (D, 1, n) — only the initial condition."""
        D, n = 10, 3
        A, S, c, y0 = _make_var1_draws(D, n)
        rng = np.random.default_rng(1)
        paths = simulate_var_paths(A, S, c, y0, 0, rng=rng)
        assert paths.shape == (D, 1, n)

    def test_single_draw_shape(self):
        D, n, horizon = 1, 2, 8
        A, S, c, y0 = _make_var1_draws(D, n)
        rng = np.random.default_rng(2)
        paths = simulate_var_paths(A, S, c, y0, horizon, rng=rng)
        assert paths.shape == (1, horizon + 1, n)

    def test_univariate_shape(self):
        """n=1 should work."""
        D, horizon = 15, 4
        A, S, c, y0 = _make_var1_draws(D, n=1)
        rng = np.random.default_rng(3)
        paths = simulate_var_paths(A, S, c, y0, horizon, rng=rng)
        assert paths.shape == (D, horizon + 1, 1)

    def test_dtype_is_float(self):
        D, n, horizon = 5, 2, 3
        A, S, c, y0 = _make_var1_draws(D, n)
        rng = np.random.default_rng(4)
        paths = simulate_var_paths(A, S, c, y0, horizon, rng=rng)
        assert paths.dtype == float

    def test_two_lag_shape(self):
        """p=2: A_draws is (D, 2, n, n), y_init must have 2 rows."""
        D, n, p, horizon = 10, 2, 2, 6
        rng_gen = np.random.default_rng(5)
        A_draws = np.tile(0.3 * np.eye(n), (D, p, 1, 1))
        Sigma_draws = np.tile(0.01 * np.eye(n), (D, 1, 1))
        intercept_draws = np.zeros((D, n))
        y_init = rng_gen.standard_normal((p, n))
        rng = np.random.default_rng(5)
        paths = simulate_var_paths(A_draws, Sigma_draws, intercept_draws, y_init, horizon, rng=rng)
        assert paths.shape == (D, horizon + 1, n)

    def test_1d_y_init_promoted(self):
        """A 1-D y_init should be accepted and promoted to (1, n)."""
        D, n, horizon = 5, 2, 3
        A, S, c, _ = _make_var1_draws(D, n)
        y_init_1d = np.array([1.0, 2.0])
        rng = np.random.default_rng(6)
        paths = simulate_var_paths(A, S, c, y_init_1d, horizon, rng=rng)
        assert paths.shape == (D, horizon + 1, n)


# ---------------------------------------------------------------------------
# simulate_var_paths — initial condition
# ---------------------------------------------------------------------------

class TestSimulateVarPathsInitialCondition:

    def test_period_zero_equals_y_init_last_row(self):
        """paths[d, 0] must equal y_init[-1] for every draw d."""
        D, n, horizon = 10, 3, 4
        rng_gen = np.random.default_rng(7)
        y_init = rng_gen.standard_normal((1, n))
        A, S, c, _ = _make_var1_draws(D, n)
        rng = np.random.default_rng(8)
        paths = simulate_var_paths(A, S, c, y_init, horizon, rng=rng)
        for d in range(D):
            np.testing.assert_array_equal(paths[d, 0], y_init[-1])

    def test_period_zero_equals_y_init_last_of_two_rows(self):
        """With p=2 y_init rows, paths[d, 0] must equal y_init[1] (most recent)."""
        D, n, p, horizon = 5, 2, 2, 3
        rng_gen = np.random.default_rng(9)
        y_init = rng_gen.standard_normal((p, n))
        A_draws = np.tile(0.3 * np.eye(n), (D, p, 1, 1))
        Sigma_draws = np.tile(0.01 * np.eye(n), (D, 1, 1))
        intercept_draws = np.zeros((D, n))
        rng = np.random.default_rng(10)
        paths = simulate_var_paths(A_draws, Sigma_draws, intercept_draws,
                                   y_init, horizon, rng=rng)
        for d in range(D):
            np.testing.assert_array_equal(paths[d, 0], y_init[-1])


# ---------------------------------------------------------------------------
# simulate_var_paths — deterministic path (zero noise)
# ---------------------------------------------------------------------------

class TestSimulateVarPathsDeterministic:

    def test_zero_sigma_gives_deterministic_path(self):
        """With Sigma ~ 0, every draw approximately follows the VAR recursion: y_{t+1} = A y_t + c.

        The code adds 1e-12 * I to Sigma before Cholesky, so noise is O(1e-6) per step.
        We use a very loose tolerance to account for accumulated numerical noise.
        """
        D, n, horizon = 5, 2, 8
        A_val = 0.7
        c_val = 0.1
        y0 = np.array([[1.0, 2.0]])
        A_draws = np.tile(A_val * np.eye(n), (D, 1, 1, 1))
        # Near-zero sigma — the code adds 1e-12*I, so effective sigma ~ 1e-6 per draw
        Sigma_draws = np.tile(1e-20 * np.eye(n), (D, 1, 1))
        intercept_draws = np.full((D, n), c_val)
        # Use a fixed rng
        rng = np.random.default_rng(11)
        paths = simulate_var_paths(A_draws, Sigma_draws, intercept_draws,
                                   y0, horizon, rng=rng)
        # Compute expected deterministic path
        y_expected = np.empty((horizon + 1, n))
        y_expected[0] = y0[0]
        for h in range(1, horizon + 1):
            y_expected[h] = A_val * y_expected[h - 1] + c_val
        for d in range(D):
            # Allow tolerance for noise accumulated from the 1e-12*I jitter
            np.testing.assert_allclose(paths[d], y_expected, atol=1e-4)

    def test_zero_intercept_zero_sigma_decays_to_zero(self):
        """With A=0.5*I, c=0, Sigma~0, starting from 1 → path decays geometrically."""
        D, n, horizon = 3, 1, 10
        A_val = 0.5
        y0 = np.array([[4.0]])
        A_draws = np.tile(A_val * np.eye(n), (D, 1, 1, 1))
        Sigma_draws = np.tile(1e-20 * np.eye(n), (D, 1, 1))
        intercept_draws = np.zeros((D, n))
        rng = np.random.default_rng(12)
        paths = simulate_var_paths(A_draws, Sigma_draws, intercept_draws,
                                   y0, horizon, rng=rng)
        for d in range(D):
            for h in range(horizon + 1):
                expected = 4.0 * (A_val ** h)
                assert abs(paths[d, h, 0] - expected) < 1e-5, (
                    f"draw {d}, horizon {h}: expected {expected}, got {paths[d, h, 0]}"
                )


# ---------------------------------------------------------------------------
# simulate_var_paths — stochastic / statistical properties
# ---------------------------------------------------------------------------

class TestSimulateVarPathsStochastic:

    def test_rng_none_does_not_crash(self):
        """When rng=None, a default rng should be created internally."""
        D, n, horizon = 5, 2, 3
        A, S, c, y0 = _make_var1_draws(D, n)
        paths = simulate_var_paths(A, S, c, y0, horizon, rng=None)
        assert paths.shape == (D, horizon + 1, n)

    def test_two_calls_same_rng_give_different_paths(self):
        """Different rng seeds produce different paths (not identical)."""
        D, n, horizon = 20, 2, 5
        A, S, c, y0 = _make_var1_draws(D, n)
        rng1 = np.random.default_rng(100)
        rng2 = np.random.default_rng(200)
        paths1 = simulate_var_paths(A, S, c, y0, horizon, rng=rng1)
        paths2 = simulate_var_paths(A, S, c, y0, horizon, rng=rng2)
        assert not np.allclose(paths1, paths2), "Different seeds must yield different paths"

    def test_same_seed_gives_identical_paths(self):
        """Same rng seed produces identical paths."""
        D, n, horizon = 10, 2, 6
        A, S, c, y0 = _make_var1_draws(D, n)
        paths1 = simulate_var_paths(A, S, c, y0, horizon,
                                    rng=np.random.default_rng(42))
        paths2 = simulate_var_paths(A, S, c, y0, horizon,
                                    rng=np.random.default_rng(42))
        np.testing.assert_array_equal(paths1, paths2)

    @pytest.mark.slow
    def test_mean_path_close_to_deterministic_for_large_D(self):
        """With large D, the cross-draw mean should approach the deterministic path.

        Known DGP: y_{t+1} = A y_t + c + eps, eps ~ N(0, Sigma).
        E[y_t] = A^t y_0 + (I + A + ... + A^{t-1}) c.
        For c=0, E[y_t] = A^t y_0.
        """
        D, n, horizon = 3000, 2, 6
        A_val = 0.6
        sigma_val = 0.3
        y0 = np.array([[1.0, -1.0]])
        A_draws = np.tile(A_val * np.eye(n), (D, 1, 1, 1))
        Sigma_draws = np.tile((sigma_val ** 2) * np.eye(n), (D, 1, 1))
        intercept_draws = np.zeros((D, n))
        rng = np.random.default_rng(99)
        paths = simulate_var_paths(A_draws, Sigma_draws, intercept_draws,
                                   y0, horizon, rng=rng)
        # E[paths[:, h, :]] = A_val^h * y0
        mean_paths = paths.mean(axis=0)  # (horizon+1, n)
        for h in range(horizon + 1):
            expected = (A_val ** h) * y0[0]
            np.testing.assert_allclose(mean_paths[h], expected, atol=0.05)


# ---------------------------------------------------------------------------
# fan_chart_quantiles — shape and ordering
# ---------------------------------------------------------------------------

class TestFanChartQuantilesShapes:

    def test_default_levels_shape(self):
        """Default 5 levels -> (5, H+1, n)."""
        D, H, n = 200, 4, 2
        rng = np.random.default_rng(13)
        paths = rng.standard_normal((D, H + 1, n))
        q = fan_chart_quantiles(paths)
        assert q.shape == (5, H + 1, n)

    def test_custom_levels_shape(self):
        """Custom k levels -> (k, H+1, n)."""
        D, H, n = 100, 3, 3
        rng = np.random.default_rng(14)
        paths = rng.standard_normal((D, H + 1, n))
        levels = (0.05, 0.25, 0.50, 0.75, 0.95)
        q = fan_chart_quantiles(paths, levels=levels)
        assert q.shape == (len(levels), H + 1, n)

    def test_single_level(self):
        """Single level -> (1, H+1, n)."""
        D, H, n = 50, 5, 2
        rng = np.random.default_rng(15)
        paths = rng.standard_normal((D, H + 1, n))
        q = fan_chart_quantiles(paths, levels=(0.5,))
        assert q.shape == (1, H + 1, n)

    def test_single_variable(self):
        """n=1 should work."""
        D, H = 100, 4
        rng = np.random.default_rng(16)
        paths = rng.standard_normal((D, H + 1, 1))
        q = fan_chart_quantiles(paths)
        assert q.shape == (5, H + 1, 1)


# ---------------------------------------------------------------------------
# fan_chart_quantiles — monotonicity (ordering of quantiles)
# ---------------------------------------------------------------------------

class TestFanChartQuantilesMonotonicity:

    def test_quantiles_are_nondecreasing_in_level(self):
        """For every horizon h and variable j, quantiles must be non-decreasing
        in the probability level."""
        D, H, n = 500, 6, 3
        rng = np.random.default_rng(17)
        paths = rng.standard_normal((D, H + 1, n))
        levels = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        q = fan_chart_quantiles(paths, levels=levels)
        # q[i+1, h, j] >= q[i, h, j] for all i, h, j
        for i in range(len(levels) - 1):
            assert np.all(q[i + 1] >= q[i] - 1e-12), (
                f"Quantile ordering violated between levels {levels[i]} and {levels[i+1]}"
            )

    def test_median_between_lower_and_upper(self):
        """Median (q=0.50) must be >= q=0.10 and <= q=0.90 pointwise."""
        D, H, n = 300, 5, 2
        rng = np.random.default_rng(18)
        paths = rng.standard_normal((D, H + 1, n))
        levels = (0.10, 0.50, 0.90)
        q = fan_chart_quantiles(paths, levels=levels)
        q10, q50, q90 = q[0], q[1], q[2]
        assert np.all(q50 >= q10 - 1e-12)
        assert np.all(q90 >= q50 - 1e-12)


# ---------------------------------------------------------------------------
# fan_chart_quantiles — known-value tests
# ---------------------------------------------------------------------------

class TestFanChartQuantilesKnownValues:

    def test_constant_paths_give_exact_quantiles(self):
        """If all paths[d, h, j] = constant, every quantile must equal that constant."""
        D, H, n = 50, 3, 2
        constant = 7.5
        paths = np.full((D, H + 1, n), constant)
        q = fan_chart_quantiles(paths)
        np.testing.assert_allclose(q, constant, atol=1e-12)

    def test_median_recovers_true_median_for_gaussian(self):
        """For a large draw count from N(mu, 1), the median quantile should be
        close to mu — the true median of the distribution."""
        D, H, n = 5000, 1, 1
        mu = 3.14
        rng = np.random.default_rng(19)
        paths = rng.normal(mu, 1.0, (D, H + 1, n))
        q = fan_chart_quantiles(paths, levels=(0.50,))
        # Median of many N(mu, 1) draws should be within 0.05 of mu
        np.testing.assert_allclose(q[0], mu, atol=0.05)

    def test_symmetry_of_quantiles_for_symmetric_distribution(self):
        """For a symmetric distribution centred at mu, quantile at p and (1-p)
        should be equidistant from mu: q(p) + q(1-p) ≈ 2*mu.

        Uses a large D=20000 and loose atol=0.1 to make this robust to MC noise.
        """
        D, H, n = 20000, 2, 2
        mu = 0.0
        rng = np.random.default_rng(20)
        paths = rng.normal(mu, 1.0, (D, H + 1, n))
        levels = (0.10, 0.25, 0.75, 0.90)
        q = fan_chart_quantiles(paths, levels=levels)
        # q[0] + q[3] ~ 2*mu, q[1] + q[2] ~ 2*mu
        np.testing.assert_allclose(q[0] + q[3], 2 * mu, atol=0.1)
        np.testing.assert_allclose(q[1] + q[2], 2 * mu, atol=0.1)

    def test_known_uniform_distribution_quantiles(self):
        """For paths drawn from U(0,1), the 0.10 quantile ~ 0.10, etc."""
        D, H, n = 10000, 1, 1
        rng = np.random.default_rng(21)
        paths = rng.uniform(0, 1, (D, H + 1, n))
        levels = (0.10, 0.25, 0.50, 0.75, 0.90)
        q = fan_chart_quantiles(paths, levels=levels)
        for i, lev in enumerate(levels):
            np.testing.assert_allclose(q[i, 0, 0], lev, atol=0.01)

    def test_horizon_zero_quantile_equals_y_init(self):
        """When all paths share the same period-0 value, q[:, 0, :] must equal it."""
        D, H, n = 30, 5, 2
        y_anchor = np.array([1.0, -2.0])
        rng = np.random.default_rng(22)
        paths = rng.standard_normal((D, H + 1, n))
        # Overwrite period 0 for all draws
        paths[:, 0, :] = y_anchor[np.newaxis, :]
        q = fan_chart_quantiles(paths)
        for i in range(q.shape[0]):
            np.testing.assert_allclose(q[i, 0, :], y_anchor, atol=1e-12)


# ---------------------------------------------------------------------------
# Integration: simulate_var_paths -> fan_chart_quantiles
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_pipeline_shape(self):
        """Full pipeline: gibbs draws -> paths -> quantiles; shape is correct."""
        D, n, p, horizon = 50, 2, 1, 8
        A_draws = np.tile(0.5 * np.eye(n), (D, 1, 1, 1))
        Sigma_draws = np.tile(0.01 * np.eye(n), (D, 1, 1))
        intercept_draws = np.zeros((D, n))
        y_init = np.zeros((1, n))
        rng = np.random.default_rng(23)
        paths = simulate_var_paths(A_draws, Sigma_draws, intercept_draws,
                                   y_init, horizon, rng=rng)
        q = fan_chart_quantiles(paths)
        assert q.shape == (5, horizon + 1, n)

    def test_period_zero_all_quantiles_equal_y_init(self):
        """Since paths[d, 0] = y_init[-1] for all d, all quantiles at h=0 equal y_init."""
        D, n, p, horizon = 100, 2, 1, 4
        y_init = np.array([[3.0, -1.5]])
        A_draws = np.tile(0.4 * np.eye(n), (D, 1, 1, 1))
        Sigma_draws = np.tile(0.05 * np.eye(n), (D, 1, 1))
        intercept_draws = np.zeros((D, n))
        rng = np.random.default_rng(24)
        paths = simulate_var_paths(A_draws, Sigma_draws, intercept_draws,
                                   y_init, horizon, rng=rng)
        q = fan_chart_quantiles(paths)
        for i in range(q.shape[0]):
            np.testing.assert_allclose(q[i, 0, :], y_init[0], atol=1e-10)

    def test_quantile_band_width_increases_with_horizon(self):
        """For a stationary VAR with noise, the 90%-10% band should be wider
        at horizon H than at horizon 1 (uncertainty accumulates)."""
        D, n, horizon = 1000, 1, 10
        A_draws = np.tile(0.5 * np.eye(n), (D, 1, 1, 1))
        Sigma_draws = np.tile(0.1 * np.eye(n), (D, 1, 1))
        intercept_draws = np.zeros((D, n))
        y_init = np.zeros((1, n))
        rng = np.random.default_rng(25)
        paths = simulate_var_paths(A_draws, Sigma_draws, intercept_draws,
                                   y_init, horizon, rng=rng)
        q = fan_chart_quantiles(paths, levels=(0.10, 0.90))
        # Band width at each horizon
        width = q[1] - q[0]  # (horizon+1, n)
        # Width at h=1 should be less than width at h=horizon (uncertainty grows)
        assert width[horizon, 0] > width[1, 0], (
            f"Expected wider band at h={horizon} than h=1; "
            f"got {width[horizon, 0]:.4f} vs {width[1, 0]:.4f}"
        )

    @pytest.mark.slow
    def test_fan_chart_median_tracks_deterministic_path(self):
        """For a very large D, the median fan-chart path should closely track
        the noise-free deterministic VAR path (law of large numbers)."""
        D, n, horizon = 4000, 2, 8
        A_val = 0.6
        y0 = np.array([[2.0, -1.0]])
        A_draws = np.tile(A_val * np.eye(n), (D, 1, 1, 1))
        Sigma_draws = np.tile(0.1 * np.eye(n), (D, 1, 1))
        intercept_draws = np.zeros((D, n))
        rng = np.random.default_rng(26)
        paths = simulate_var_paths(A_draws, Sigma_draws, intercept_draws,
                                   y0, horizon, rng=rng)
        q = fan_chart_quantiles(paths, levels=(0.50,))
        # Expected deterministic median (c=0, symmetric noise => median = mean = A^h y0)
        for h in range(horizon + 1):
            expected = (A_val ** h) * y0[0]
            np.testing.assert_allclose(q[0, h, :], expected, atol=0.1)
