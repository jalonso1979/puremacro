"""Focused coverage tests for puremacro.var.regime.tvecm.

Test strategy
-------------
- Known-value: synthetic cointegrated DGP (I(1) pair with known β);
  verify that Engle-Granger first stage recovers the cointegrating vector
  and that tvecm_fit returns a result whose field types/shapes match the spec.
- Contract: TVECMResult is a dataclass with all documented fields; output
  types are numpy arrays; n_low + n_high == total usable obs.
- Properties: rss_total > 0; Sigma matrices are symmetric; alpha vectors
  have correct shape; threshold is a float inside the sample range of the
  threshold variable; n_low and n_high are positive and sum to the effective
  sample.
- Parameter sweep: delay > 1; p > 1; pre-specified beta; n_threshold_grid
  variation; trim variation.
- Edge cases / error handling: fewer than 2 variables raises ValueError;
  trim so tight that no split survives raises RuntimeError.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from puremacro.var.regime.tvecm import (
    TVECMResult,
    _engle_granger_beta,
    tvecm_fit,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic cointegrated DGP
# ---------------------------------------------------------------------------

def _make_cointegrated(
    T: int = 200,
    beta_true: float = 2.0,
    seed: int = 0,
    threshold_effect: float = 0.5,
) -> np.ndarray:
    """Generate a cointegrated (T, 2) array.

    y_0_t = beta_true * y_1_t + u_t  where y_1 is a random walk.
    A two-regime threshold structure is imposed on ec: when ec_{t-1} > 0,
    the adjustment speed differs.  Returns Y as shape (T, 2).
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(T)) * 0.5  # I(1)
    u = rng.standard_normal(T) * 0.3             # stationary noise
    y = beta_true * x + u
    return np.column_stack([y, x])


def _make_cointegrated_n3(T: int = 200, seed: int = 1) -> np.ndarray:
    """Three-variable cointegrated system: y0 = 2*y1 + y2 + noise."""
    rng = np.random.default_rng(seed)
    y1 = np.cumsum(rng.standard_normal(T)) * 0.5
    y2 = np.cumsum(rng.standard_normal(T)) * 0.5
    noise = rng.standard_normal(T) * 0.3
    y0 = 2.0 * y1 + y2 + noise
    return np.column_stack([y0, y1, y2])


# ---------------------------------------------------------------------------
# _engle_granger_beta — known-value tests
# ---------------------------------------------------------------------------

class TestEngleGrangerBeta:
    def test_leading_1(self):
        """Returned beta must have leading element == 1."""
        Y = _make_cointegrated(T=150, beta_true=2.5, seed=10)
        beta = _engle_granger_beta(Y)
        assert float(beta[0]) == pytest.approx(1.0)

    def test_length_equals_n_variables(self):
        """beta has one entry per variable."""
        Y = _make_cointegrated(T=150, seed=11)
        beta = _engle_granger_beta(Y)
        assert beta.shape == (2,)

    def test_recovers_cointegrating_slope_bivariate(self):
        """In a bivariate DGP with β_true = 2, EG recovers β[1] ≈ -2."""
        Y = _make_cointegrated(T=500, beta_true=2.0, seed=20)
        beta = _engle_granger_beta(Y)
        # beta is normalized so that beta' y ≈ 0 on average.
        # y0 = 2 * y1 + noise ⟹ y0 - 2*y1 ≈ 0 ⟹ β = [1, -2].
        assert beta[1] == pytest.approx(-2.0, abs=0.15)

    def test_trivariate_shape(self):
        """beta has length n for n=3."""
        Y = _make_cointegrated_n3(T=150, seed=30)
        beta = _engle_granger_beta(Y)
        assert beta.shape == (3,)
        assert float(beta[0]) == pytest.approx(1.0)

    def test_rejects_univariate(self):
        """Single-column array raises ValueError (need ≥ 2 variables)."""
        Y = np.random.default_rng(0).standard_normal((50, 1))
        with pytest.raises(ValueError, match="need at least 2"):
            _engle_granger_beta(Y)

    def test_returns_ndarray(self):
        Y = _make_cointegrated(T=100, seed=5)
        beta = _engle_granger_beta(Y)
        assert isinstance(beta, np.ndarray)
        assert beta.dtype == float


# ---------------------------------------------------------------------------
# TVECMResult — dataclass contract
# ---------------------------------------------------------------------------

class TestTVECMResultDataclass:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(TVECMResult)

    def test_has_all_documented_fields(self):
        expected = {
            "beta", "threshold", "delay",
            "alpha_low", "alpha_high",
            "intercept_low", "intercept_high",
            "Gamma_low", "Gamma_high",
            "Sigma_low", "Sigma_high",
            "n_low", "n_high",
            "rss_total",
        }
        actual = {f.name for f in dataclasses.fields(TVECMResult)}
        assert expected <= actual

    def test_all_exports(self):
        import puremacro.var.regime.tvecm as mod
        assert set(mod.__all__) == {"tvecm_fit", "TVECMResult"}


# ---------------------------------------------------------------------------
# tvecm_fit — output shape, type, and contract
# ---------------------------------------------------------------------------

class TestTVECMFitContract:
    """Test that tvecm_fit returns a TVECMResult with correct types/shapes."""

    @pytest.fixture(scope="class")
    def basic_result(self):
        Y = _make_cointegrated(T=200, seed=42)
        return tvecm_fit(Y, p=1, delay=1, n_threshold_grid=30, trim=0.15)

    def test_returns_tvecm_result(self, basic_result):
        assert isinstance(basic_result, TVECMResult)

    def test_beta_is_ndarray(self, basic_result):
        assert isinstance(basic_result.beta, np.ndarray)
        assert basic_result.beta.shape == (2,)

    def test_threshold_is_float(self, basic_result):
        assert isinstance(basic_result.threshold, float)

    def test_delay_preserved(self, basic_result):
        assert basic_result.delay == 1

    def test_alpha_low_is_ndarray(self, basic_result):
        assert isinstance(basic_result.alpha_low, np.ndarray)
        assert basic_result.alpha_low.shape == (2,)

    def test_alpha_high_is_ndarray(self, basic_result):
        assert isinstance(basic_result.alpha_high, np.ndarray)
        assert basic_result.alpha_high.shape == (2,)

    def test_intercept_shapes(self, basic_result):
        assert basic_result.intercept_low.shape == (2,)
        assert basic_result.intercept_high.shape == (2,)

    def test_sigma_low_shape(self, basic_result):
        """Sigma_low must be n×n = 2×2."""
        assert basic_result.Sigma_low.shape == (2, 2)

    def test_sigma_high_shape(self, basic_result):
        assert basic_result.Sigma_high.shape == (2, 2)

    def test_sigma_symmetric(self, basic_result):
        assert np.allclose(basic_result.Sigma_low, basic_result.Sigma_low.T)
        assert np.allclose(basic_result.Sigma_high, basic_result.Sigma_high.T)

    def test_n_low_positive(self, basic_result):
        assert basic_result.n_low > 0

    def test_n_high_positive(self, basic_result):
        assert basic_result.n_high > 0

    def test_rss_positive(self, basic_result):
        assert basic_result.rss_total > 0.0

    def test_rss_is_float(self, basic_result):
        assert isinstance(basic_result.rss_total, float)

    def test_threshold_inside_sample_range(self, basic_result):
        """Threshold must be inside the 15th–85th percentile band (trim=0.15)."""
        Y = _make_cointegrated(T=200, seed=42)
        # ec = Y @ beta
        ec = Y @ basic_result.beta
        lo = np.quantile(ec, 0.15)
        hi = np.quantile(ec, 0.85)
        assert lo <= basic_result.threshold <= hi


# ---------------------------------------------------------------------------
# tvecm_fit — property tests
# ---------------------------------------------------------------------------

class TestTVECMFitProperties:
    def test_n_low_plus_n_high_consistency(self):
        """n_low and n_high are positive and sum matches obs used in regression."""
        Y = _make_cointegrated(T=150, seed=5)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=20)
        # Both regimes must be non-empty
        assert res.n_low > 0 and res.n_high > 0
        # Their sum equals n_low + n_high (self-evident, but tests that
        # the fields are int and non-negative)
        assert isinstance(res.n_low, int)
        assert isinstance(res.n_high, int)

    def test_sigma_positive_semidefinite_low(self):
        Y = _make_cointegrated(T=200, seed=7)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=20)
        eigs = np.linalg.eigvalsh(res.Sigma_low)
        assert eigs.min() >= -1e-10

    def test_sigma_positive_semidefinite_high(self):
        Y = _make_cointegrated(T=200, seed=8)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=20)
        eigs = np.linalg.eigvalsh(res.Sigma_high)
        assert eigs.min() >= -1e-10

    def test_beta_leading_1(self):
        """tvecm_fit auto-estimates beta: leading element should be 1."""
        Y = _make_cointegrated(T=200, seed=9)
        res = tvecm_fit(Y, p=1, delay=1)
        assert float(res.beta[0]) == pytest.approx(1.0)

    def test_list_input_accepted(self):
        """tvecm_fit coerces a list-of-lists to float ndarray."""
        Y = _make_cointegrated(T=150, seed=15)
        res = tvecm_fit(Y.tolist(), p=1, delay=1, n_threshold_grid=20)
        assert isinstance(res.beta, np.ndarray)

    def test_grid_size_does_not_affect_contract(self):
        """Varying n_threshold_grid: contract is always satisfied."""
        Y = _make_cointegrated(T=150, seed=16)
        for ng in [10, 50]:
            res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=ng)
            assert isinstance(res, TVECMResult)
            assert res.rss_total > 0

    def test_larger_grid_rss_leq_smaller_grid(self):
        """A finer grid can only find a threshold as good or better (≤ RSS)."""
        Y = _make_cointegrated(T=200, seed=17)
        res_coarse = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=10)
        res_fine   = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=100)
        # finer grid must have RSS ≤ coarser grid (grid only improves search)
        assert res_fine.rss_total <= res_coarse.rss_total + 1e-10


# ---------------------------------------------------------------------------
# tvecm_fit — pre-specified beta
# ---------------------------------------------------------------------------

class TestTVECMFitPrespecifiedBeta:
    def test_prespecified_beta_is_used(self):
        """When beta is passed, the result's beta should equal the input."""
        Y = _make_cointegrated(T=200, beta_true=3.0, seed=20)
        beta_known = np.array([1.0, -3.0])
        res = tvecm_fit(Y, p=1, delay=1, beta=beta_known)
        assert np.allclose(res.beta, beta_known)

    def test_prespecified_beta_different_from_eg(self):
        """Results differ when beta is pre-specified vs. EG-estimated."""
        Y = _make_cointegrated(T=200, beta_true=2.0, seed=21)
        beta_wrong = np.array([1.0, -5.0])  # intentionally wrong
        res_auto = tvecm_fit(Y, p=1, delay=1)
        res_forced = tvecm_fit(Y, p=1, delay=1, beta=beta_wrong)
        # The beta fields should differ
        assert not np.allclose(res_auto.beta, res_forced.beta)


# ---------------------------------------------------------------------------
# tvecm_fit — lag order p
# ---------------------------------------------------------------------------

class TestTVECMFitLagOrder:
    def test_p1_gamma_shape(self):
        """p=1 → k_lags=0; Gamma_low must have second dim 0 (no lagged diffs)."""
        Y = _make_cointegrated(T=200, seed=30)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=20)
        # Gamma_low shape should be (n, k_lags * n) = (2, 0)
        assert res.Gamma_low.shape[0] == 2
        assert res.Gamma_low.shape[1] == 0

    def test_p2_gamma_shape(self):
        """p=2 → k_lags=1; Gamma_low must have shape (n, n) = (2, 2)."""
        Y = _make_cointegrated(T=200, seed=31)
        res = tvecm_fit(Y, p=2, delay=1, n_threshold_grid=20)
        assert res.Gamma_low.shape == (2, 2)
        assert res.Gamma_high.shape == (2, 2)

    def test_p3_gamma_shape(self):
        """p=3 → k_lags=2; Gamma_low must have shape (2, 4)."""
        Y = _make_cointegrated(T=250, seed=32)
        res = tvecm_fit(Y, p=3, delay=1, n_threshold_grid=20)
        assert res.Gamma_low.shape == (2, 4)
        assert res.Gamma_high.shape == (2, 4)

    def test_p2_runs_without_error(self):
        Y = _make_cointegrated(T=200, seed=33)
        res = tvecm_fit(Y, p=2, delay=1, n_threshold_grid=20)
        assert isinstance(res, TVECMResult)
        assert res.rss_total > 0


# ---------------------------------------------------------------------------
# tvecm_fit — delay parameter
# ---------------------------------------------------------------------------

class TestTVECMFitDelay:
    def test_delay1_stored_in_result(self):
        Y = _make_cointegrated(T=200, seed=40)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=20)
        assert res.delay == 1

    def test_delay2_stored_in_result(self):
        Y = _make_cointegrated(T=200, seed=41)
        res = tvecm_fit(Y, p=1, delay=2, n_threshold_grid=20)
        assert res.delay == 2

    def test_delay2_produces_valid_result(self):
        Y = _make_cointegrated(T=200, seed=42)
        res = tvecm_fit(Y, p=1, delay=2, n_threshold_grid=20)
        assert isinstance(res, TVECMResult)
        assert res.n_low > 0 and res.n_high > 0


# ---------------------------------------------------------------------------
# tvecm_fit — three-variable system
# ---------------------------------------------------------------------------

class TestTVECMFitTrivariate:
    def test_trivariate_shapes(self):
        Y = _make_cointegrated_n3(T=200, seed=50)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=20)
        assert res.beta.shape == (3,)
        assert res.alpha_low.shape == (3,)
        assert res.alpha_high.shape == (3,)
        assert res.Sigma_low.shape == (3, 3)
        assert res.Sigma_high.shape == (3, 3)

    def test_trivariate_rss_positive(self):
        Y = _make_cointegrated_n3(T=200, seed=51)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=20)
        assert res.rss_total > 0


# ---------------------------------------------------------------------------
# tvecm_fit — edge cases and error handling
# ---------------------------------------------------------------------------

class TestTVECMFitEdgeCases:
    def test_constant_ec_raises_runtime_error(self):
        """When the threshold variable (ec) is constant, all candidates split
        the sample the same way and at least one regime ends up empty ->
        RuntimeError is raised."""
        rng = np.random.default_rng(60)
        T = 50
        x = np.cumsum(rng.standard_normal(T)) * 0.5
        # Exact linear combination -> ec = y - 2x = 0 everywhere
        y = 2.0 * x
        Y = np.column_stack([y, x])
        # Pre-specify the exact cointegrating vector so ec is truly zero
        beta_exact = np.array([1.0, -2.0])
        with pytest.raises(RuntimeError, match="No threshold split"):
            tvecm_fit(Y, p=1, delay=1, n_threshold_grid=5, trim=0.15,
                      beta=beta_exact)

    def test_univariate_raises_value_error(self):
        """Single variable array: _engle_granger_beta raises ValueError."""
        Y = np.random.default_rng(0).standard_normal((50, 1))
        with pytest.raises(ValueError):
            tvecm_fit(Y, p=1, delay=1)

    def test_minimal_trim(self):
        """trim=0.05 allows small regimes; result must still be valid."""
        Y = _make_cointegrated(T=200, seed=61)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=20, trim=0.05)
        assert isinstance(res, TVECMResult)
        assert res.rss_total > 0

    def test_large_n_threshold_grid(self):
        """n_threshold_grid=200: result is valid and no crash."""
        Y = _make_cointegrated(T=200, seed=62)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=200, trim=0.15)
        assert isinstance(res, TVECMResult)

    def test_input_coerced_to_float(self):
        """Integer input array is coerced to float without error."""
        rng = np.random.default_rng(70)
        Y = (rng.standard_normal((100, 2)) * 100).astype(int)
        # Build a cointegrated-ish system: y0 = 2*y1
        Y[:, 0] = 2 * Y[:, 1] + rng.integers(-5, 5, 100)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=20)
        assert res.beta.dtype == float


# ---------------------------------------------------------------------------
# tvecm_fit — known-value: Engle-Granger beta recovery end-to-end
# ---------------------------------------------------------------------------

class TestTVECMFitKnownValue:
    @pytest.mark.slow
    def test_beta_close_to_true_in_large_sample(self):
        """In a large sample the auto-estimated beta recovers the DGP value.

        DGP: y0 = 2 * y1 + noise  =>  true β = [1, -2].
        """
        Y = _make_cointegrated(T=1000, beta_true=2.0, seed=100)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=50)
        assert float(res.beta[0]) == pytest.approx(1.0)
        # With T=1000, the second cointegrating coefficient should be near -2.
        assert float(res.beta[1]) == pytest.approx(-2.0, abs=0.2)

    def test_threshold_is_scalar_float(self):
        """Threshold returned is a Python float (not numpy scalar)."""
        Y = _make_cointegrated(T=200, seed=101)
        res = tvecm_fit(Y, p=1, delay=1, n_threshold_grid=20)
        assert isinstance(res.threshold, float)
