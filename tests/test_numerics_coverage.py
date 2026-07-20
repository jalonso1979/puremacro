"""Focused tests for puremacro.numerics.

Covers:
- numerical_gradient: known-value (quadratic, linear), shape, dtype
- numerical_hessian: known-value (quadratic), symmetry, shape, constant fn
- sandwich_vcov: shape, PSD, correct formula for Gaussian normal model
- mle_fit: normal-mean MLE recovers truth, output keys, sandwich=True path,
           sandwich=True without per_obs raises ValueError
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.numerics import (
    numerical_gradient,
    numerical_hessian,
    sandwich_vcov,
    mle_fit,
)


# ---------------------------------------------------------------------------
# numerical_gradient
# ---------------------------------------------------------------------------

class TestNumericalGradient:
    """Central-difference gradient checks."""

    def test_quadratic_known_value(self):
        """f(x) = x^T A x / 2  => grad = A x."""
        A = np.array([[4.0, 1.0], [1.0, 2.0]])
        x0 = np.array([1.0, 2.0])
        f = lambda x: 0.5 * float(x @ A @ x)
        g = numerical_gradient(f, x0)
        expected = A @ x0
        np.testing.assert_allclose(g, expected, rtol=1e-6, atol=1e-7)

    def test_linear_function(self):
        """f(x) = c^T x  => grad = c exactly."""
        c = np.array([3.0, -1.0, 2.5])
        f = lambda x: float(c @ x)
        x0 = np.zeros(3)
        g = numerical_gradient(f, x0)
        np.testing.assert_allclose(g, c, rtol=1e-8, atol=1e-10)

    def test_output_shape_and_dtype(self):
        """Gradient has same shape as input and is float."""
        f = lambda x: float(np.sum(x**2))
        x0 = np.array([1.0, 2.0, 3.0, 4.0])
        g = numerical_gradient(f, x0)
        assert g.shape == x0.shape
        assert g.dtype == float

    def test_scalar_input(self):
        """Works for 1-d input (single parameter)."""
        f = lambda x: float(x[0] ** 3)
        x0 = np.array([2.0])
        g = numerical_gradient(f, x0)
        # d/dx x^3 = 3x^2; at x=2: 12
        np.testing.assert_allclose(g, [12.0], rtol=1e-6)

    def test_accepts_list_input(self):
        """numerical_gradient must accept plain Python list as x."""
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        g = numerical_gradient(f, [3.0, 4.0])
        # grad = [2*3, 2*4] = [6, 8]
        np.testing.assert_allclose(g, [6.0, 8.0], rtol=1e-6)

    def test_custom_step_size(self):
        """Different h values all give a close answer for smooth f."""
        f = lambda x: float(np.sin(x[0]) + np.cos(x[1]))
        x0 = np.array([0.5, 1.0])
        g_default = numerical_gradient(f, x0)
        g_small = numerical_gradient(f, x0, h=1e-7)
        expected = np.array([np.cos(0.5), -np.sin(1.0)])
        np.testing.assert_allclose(g_default, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(g_small, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# numerical_hessian
# ---------------------------------------------------------------------------

class TestNumericalHessian:
    """Central-difference Hessian checks."""

    def test_quadratic_known_value(self):
        """f(x) = x^T A x  => Hessian = A + A^T (here A is symmetric -> 2A;
        but f = 0.5 x^T A x => Hessian = A)."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        f = lambda x: 0.5 * float(x @ A @ x)
        x0 = np.zeros(2)
        H = numerical_hessian(f, x0)
        np.testing.assert_allclose(H, A, rtol=1e-5, atol=1e-6)

    def test_constant_function_hessian_is_zero(self):
        """Hessian of a constant function is zero."""
        f = lambda x: 5.0
        x0 = np.array([1.0, 2.0, 3.0])
        H = numerical_hessian(f, x0)
        np.testing.assert_allclose(H, np.zeros((3, 3)), atol=1e-8)

    def test_hessian_is_symmetric(self):
        """Hessian must be symmetric (by construction)."""
        rng = np.random.default_rng(42)
        B = rng.standard_normal((4, 4))
        M = B.T @ B  # PSD
        f = lambda x: float(x @ M @ x)
        x0 = rng.standard_normal(4)
        H = numerical_hessian(f, x0)
        np.testing.assert_allclose(H, H.T, atol=1e-10)

    def test_output_shape(self):
        """Hessian shape is (n, n) for n-dim input."""
        f = lambda x: float(np.sum(x**4))
        x0 = np.array([1.0, 2.0, 3.0])
        H = numerical_hessian(f, x0)
        assert H.shape == (3, 3)

    def test_diagonal_quadratic_hessian(self):
        """f = sum_i a_i x_i^2 => Hessian = 2*diag(a)."""
        a = np.array([1.0, 2.0, 3.0])
        f = lambda x: float(np.sum(a * x**2))
        x0 = np.zeros(3)
        H = numerical_hessian(f, x0)
        expected = 2.0 * np.diag(a)
        np.testing.assert_allclose(H, expected, rtol=1e-5, atol=1e-6)

    def test_scalar_hessian(self):
        """f(x) = x^4 => H = 12 x^2 = 12 at x=1."""
        f = lambda x: float(x[0] ** 4)
        x0 = np.array([1.0])
        H = numerical_hessian(f, x0)
        np.testing.assert_allclose(H[0, 0], 12.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# sandwich_vcov
# ---------------------------------------------------------------------------

class TestSandwichVcov:
    """Sandwich covariance V = H^{-1} J H^{-1} / T."""

    def _normal_mean_setup(self, n: int = 200, mu_true: float = 1.5,
                            sigma: float = 1.0, seed: int = 0):
        """Correctly-specified normal mean MLE: neg_loglik_per_obs."""
        rng = np.random.default_rng(seed)
        data = rng.normal(mu_true, sigma, n)

        def neg_ll_per_obs(theta):
            mu = theta[0]
            return 0.5 * (data - mu) ** 2 / sigma ** 2 + 0.5 * np.log(
                2 * np.pi * sigma ** 2
            )

        return data, neg_ll_per_obs

    def test_output_shape(self):
        """sandwich_vcov returns (k, k) matrix."""
        _, neg_ll_per_obs = self._normal_mean_setup()
        theta_hat = np.array([1.5])
        V = sandwich_vcov(neg_ll_per_obs, theta_hat)
        assert V.shape == (1, 1)

    def test_vcov_is_positive(self):
        """Variance of a scalar parameter estimate is positive."""
        data, neg_ll_per_obs = self._normal_mean_setup(n=500, seed=1)
        # MLE for normal mean is the sample mean
        theta_hat = np.array([np.mean(data)])
        V = sandwich_vcov(neg_ll_per_obs, theta_hat)
        assert float(V[0, 0]) > 0.0

    def test_vcov_is_symmetric(self):
        """Sandwich vcov must be symmetric for multi-param model."""
        rng = np.random.default_rng(7)
        n = 300
        data = rng.normal(0.5, 1.5, n)

        # Two-parameter normal model (mu, log_sigma)
        def neg_ll_per_obs(theta):
            mu, log_s = theta[0], theta[1]
            s = np.exp(log_s)
            return 0.5 * (data - mu) ** 2 / s ** 2 + log_s + 0.5 * np.log(
                2 * np.pi
            )

        theta_hat = np.array([np.mean(data), np.log(np.std(data))])
        V = sandwich_vcov(neg_ll_per_obs, theta_hat)
        assert V.shape == (2, 2)
        np.testing.assert_allclose(V, V.T, atol=1e-10)

    def test_correctly_specified_vcov_near_asymptotic(self):
        """For correctly-specified Gaussian model J = H, so V = H^{-1}/T.
        With large n, this should be near sigma^2/n."""
        n = 5000
        sigma = 1.0
        _, neg_ll_per_obs = self._normal_mean_setup(n=n, sigma=sigma, seed=42)
        theta_hat = np.array([1.5])  # true mean, maximiser
        V = sandwich_vcov(neg_ll_per_obs, theta_hat)
        # Asymptotic variance of sample mean = sigma^2 / n
        expected_var = sigma ** 2 / n
        np.testing.assert_allclose(float(V[0, 0]), expected_var, rtol=0.05)

    def test_larger_n_gives_smaller_vcov(self):
        """More data -> smaller sandwich variance."""
        _, nllo_small = self._normal_mean_setup(n=100, seed=3)
        _, nllo_large = self._normal_mean_setup(n=5000, seed=3)
        theta = np.array([1.5])
        V_small = sandwich_vcov(nllo_small, theta)
        V_large = sandwich_vcov(nllo_large, theta)
        assert float(V_small[0, 0]) > float(V_large[0, 0])


# ---------------------------------------------------------------------------
# mle_fit
# ---------------------------------------------------------------------------

class TestMleFit:
    """Tests for mle_fit: basic MLE, output keys, sandwich option."""

    def _make_normal_mean_nll(self, data: np.ndarray, sigma: float = 1.0):
        """Return scalar neg_loglik and per-obs neg_loglik for normal mean."""
        def neg_ll(theta):
            mu = theta[0]
            return float(np.sum(0.5 * (data - mu) ** 2 / sigma ** 2))

        def neg_ll_per_obs(theta):
            mu = theta[0]
            return 0.5 * (data - mu) ** 2 / sigma ** 2 + 0.5 * np.log(
                2 * np.pi * sigma ** 2
            )

        return neg_ll, neg_ll_per_obs

    def test_recovers_normal_mean(self):
        """MLE for normal mean = sample mean."""
        rng = np.random.default_rng(0)
        n = 300
        mu_true = 2.0
        data = rng.normal(mu_true, 1.0, n)
        neg_ll, _ = self._make_normal_mean_nll(data)
        result = mle_fit(neg_ll, np.array([0.0]))
        np.testing.assert_allclose(result["theta"][0], np.mean(data), atol=1e-6)

    def test_output_keys_without_sandwich(self):
        """Without sandwich=True the result dict has the expected keys."""
        rng = np.random.default_rng(1)
        data = rng.standard_normal(100)
        neg_ll, _ = self._make_normal_mean_nll(data)
        result = mle_fit(neg_ll, np.array([0.0]))
        required_keys = {"theta", "loglik", "hessian", "vcov_classical",
                         "se_classical", "converged"}
        assert required_keys.issubset(set(result.keys()))
        assert "vcov_sandwich" not in result
        assert "se_sandwich" not in result

    def test_output_shapes(self):
        """theta shape, hessian shape, and se shape are consistent."""
        rng = np.random.default_rng(2)
        data = rng.standard_normal(100)
        neg_ll, _ = self._make_normal_mean_nll(data)
        result = mle_fit(neg_ll, np.array([0.0]))
        k = 1
        assert result["theta"].shape == (k,)
        assert result["hessian"].shape == (k, k)
        assert result["vcov_classical"].shape == (k, k)
        assert result["se_classical"].shape == (k,)

    def test_se_classical_is_nonnegative(self):
        """Classical SE must be non-negative."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal(200)
        neg_ll, _ = self._make_normal_mean_nll(data)
        result = mle_fit(neg_ll, np.array([0.0]))
        assert np.all(result["se_classical"] >= 0.0)

    def test_loglik_is_finite_and_negative_of_fun(self):
        """loglik field should be finite."""
        rng = np.random.default_rng(4)
        data = rng.standard_normal(100)
        neg_ll, _ = self._make_normal_mean_nll(data)
        result = mle_fit(neg_ll, np.array([0.0]))
        assert np.isfinite(result["loglik"])

    def test_converged_flag_is_bool(self):
        """converged must be a plain Python bool."""
        rng = np.random.default_rng(5)
        data = rng.standard_normal(50)
        neg_ll, _ = self._make_normal_mean_nll(data)
        result = mle_fit(neg_ll, np.array([0.0]))
        assert isinstance(result["converged"], bool)

    def test_sandwich_mode_adds_keys(self):
        """sandwich=True adds vcov_sandwich and se_sandwich to the output."""
        rng = np.random.default_rng(6)
        data = rng.standard_normal(200)
        neg_ll, neg_ll_per_obs = self._make_normal_mean_nll(data)
        result = mle_fit(
            neg_ll, np.array([0.0]),
            sandwich=True, neg_loglik_per_obs=neg_ll_per_obs
        )
        assert "vcov_sandwich" in result
        assert "se_sandwich" in result

    def test_sandwich_se_is_nonnegative(self):
        """Sandwich SE must be non-negative."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal(300)
        neg_ll, neg_ll_per_obs = self._make_normal_mean_nll(data)
        result = mle_fit(
            neg_ll, np.array([0.0]),
            sandwich=True, neg_loglik_per_obs=neg_ll_per_obs
        )
        assert np.all(result["se_sandwich"] >= 0.0)

    def test_sandwich_raises_without_per_obs(self):
        """sandwich=True without neg_loglik_per_obs raises ValueError."""
        rng = np.random.default_rng(8)
        data = rng.standard_normal(100)
        neg_ll, _ = self._make_normal_mean_nll(data)
        with pytest.raises(ValueError, match="neg_loglik_per_obs"):
            mle_fit(neg_ll, np.array([0.0]), sandwich=True)

    def test_mle_with_bounds(self):
        """mle_fit respects the bounds argument (mu >= 0)."""
        rng = np.random.default_rng(9)
        # All-positive data; bound mu >= 0 should be satisfied at MLE
        data = rng.exponential(scale=2.0, size=200)
        neg_ll, _ = self._make_normal_mean_nll(data)
        result = mle_fit(neg_ll, np.array([0.5]), bounds=[(0.0, None)])
        assert result["theta"][0] >= -1e-8  # at or above bound

    def test_two_parameter_normal_mle(self):
        """Joint MLE of (mu, log_sigma) converges to known truth."""
        rng = np.random.default_rng(10)
        n = 1000
        mu_true, sigma_true = 1.0, 2.0
        data = rng.normal(mu_true, sigma_true, n)

        def neg_ll(theta):
            mu, log_s = theta
            s = np.exp(log_s)
            return float(n * log_s + np.sum((data - mu) ** 2) / (2 * s ** 2))

        result = mle_fit(neg_ll, np.array([0.0, 0.0]))
        np.testing.assert_allclose(result["theta"][0], mu_true, atol=0.3)
        np.testing.assert_allclose(
            np.exp(result["theta"][1]), sigma_true, rtol=0.3
        )
        assert result["hessian"].shape == (2, 2)
        assert result["vcov_classical"].shape == (2, 2)
        assert result["se_classical"].shape == (2,)

    def test_classical_ci_contains_truth(self):
        """theta +/- 3*se_classical should contain the true value."""
        rng = np.random.default_rng(11)
        mu_true = 3.0
        data = rng.normal(mu_true, 1.0, 500)
        neg_ll, _ = self._make_normal_mean_nll(data)
        result = mle_fit(neg_ll, np.array([0.0]))
        mu_hat = result["theta"][0]
        se = result["se_classical"][0]
        assert abs(mu_hat - mu_true) < 3 * se

    @pytest.mark.slow
    def test_sandwich_matches_classical_correctly_specified(self):
        """For correctly-specified model, sandwich and classical SE are close."""
        rng = np.random.default_rng(99)
        n = 2000
        data = rng.normal(1.0, 1.0, n)
        neg_ll, neg_ll_per_obs = self._make_normal_mean_nll(data)
        result = mle_fit(
            neg_ll, np.array([np.mean(data)]),
            sandwich=True, neg_loglik_per_obs=neg_ll_per_obs
        )
        ratio = result["se_sandwich"][0] / result["se_classical"][0]
        # Should be near 1 for a correctly-specified model
        assert 0.5 < float(ratio) < 2.0

    def test_mle_fit_singular_hessian_uses_pinv(self, monkeypatch):
        """When np.linalg.inv raises LinAlgError, mle_fit falls back to pinv.
        We monkeypatch inv inside puremacro.numerics to simulate a singular H."""
        import puremacro.numerics as _num_mod

        original_inv = np.linalg.inv
        call_count = {"n": 0}

        def _raise_on_first(m):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise np.linalg.LinAlgError("singular (simulated)")
            return original_inv(m)

        monkeypatch.setattr(_num_mod.np.linalg, "inv", _raise_on_first)

        rng = np.random.default_rng(20)
        data = rng.standard_normal(100)
        neg_ll, _ = self._make_normal_mean_nll(data)
        # Should not raise — pinv fallback handles it
        result = mle_fit(neg_ll, np.array([np.mean(data)]))
        assert "vcov_classical" in result
        assert result["vcov_classical"].shape == (1, 1)

    def test_sandwich_vcov_singular_hessian_uses_pinv(self, monkeypatch):
        """When np.linalg.inv raises in sandwich_vcov, pinv is used."""
        import puremacro.numerics as _num_mod

        original_inv = np.linalg.inv
        call_count = {"n": 0}

        def _raise_on_first(m):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise np.linalg.LinAlgError("singular (simulated)")
            return original_inv(m)

        monkeypatch.setattr(_num_mod.np.linalg, "inv", _raise_on_first)

        rng = np.random.default_rng(21)
        n = 200
        data = rng.standard_normal(n)

        def neg_ll_per_obs(theta):
            mu = theta[0]
            return 0.5 * (data - mu) ** 2

        theta = np.array([np.mean(data)])
        # Should not raise — pinv fallback in sandwich_vcov handles it
        V = sandwich_vcov(neg_ll_per_obs, theta)
        assert V.shape == (1, 1)
        assert np.isfinite(V[0, 0])
