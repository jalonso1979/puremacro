"""Focused tests for puremacro.var.regime.ms_var (MS-VAR via EM).

Test strategy
-------------
- known_value: synthetic data from a known 2-regime DGP; recover the
  regime-specific variance ratio (signal is large enough to survive EM
  label-switching and be asserted numerically).
- property: output shapes, dtype, probability bounds, P-row sums,
  Sigma symmetry / PSD, log-likelihood finiteness, seed reproducibility,
  final filtered == final smoothed.
- contract: internal helper contracts (_gauss_density known value,
  _hamilton_filter + _kim_smoother probability sum constraints).
- edge: K=1 degenerate regime, list/DataFrame input coercion, verbose
  flag, short n_iter, zero-density fallback in filter.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from puremacro.var.regime.ms_var import (
    MSVARResult,
    _gauss_density,
    _hamilton_filter,
    _kim_smoother,
    ms_var_fit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_regime_var_data(
    T: int = 300,
    rng: np.random.Generator | None = None,
    mu_diff: float = 0.0,
    sigma_low: float = 0.5,
    sigma_high: float = 2.0,
    p_stay: float = 0.9,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate T observations from a univariate 2-regime MS model.

    Regime 0 ~ N(0, sigma_low^2), Regime 1 ~ N(mu_diff, sigma_high^2).
    Returns (Y, true_regimes).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    P_true = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
    regimes = np.zeros(T, dtype=int)
    s = 0
    for t in range(1, T):
        s = rng.choice(2, p=P_true[s])
        regimes[t] = s
    sigma = np.array([sigma_low, sigma_high])
    mu = np.array([0.0, mu_diff])
    Y = np.zeros((T, 1))
    for t in range(T):
        Y[t, 0] = mu[regimes[t]] + sigma[regimes[t]] * rng.standard_normal()
    return Y, regimes


# ---------------------------------------------------------------------------
# _gauss_density — known-value tests
# ---------------------------------------------------------------------------

class TestGaussDensity:
    def test_univariate_standard_normal_at_zero(self):
        """density(0 | mu=0, Sigma=I) == 1/sqrt(2*pi) for n=1."""
        y = np.array([0.0])
        mean = np.array([0.0])
        cov = np.array([[1.0]])
        result = _gauss_density(y, mean, cov)
        expected = 1.0 / np.sqrt(2.0 * np.pi)
        assert abs(result - expected) < 1e-6

    def test_bivariate_standard_normal_at_zero(self):
        """density(0 | mu=0, Sigma=I) == 1/(2*pi) for n=2."""
        y = np.zeros(2)
        mean = np.zeros(2)
        cov = np.eye(2)
        result = _gauss_density(y, mean, cov)
        expected = 1.0 / (2.0 * np.pi)
        assert abs(result - expected) < 1e-6

    def test_density_decreases_with_distance(self):
        """density at mean > density far from mean."""
        cov = np.array([[1.0]])
        d_center = _gauss_density(np.array([0.0]), np.array([0.0]), cov)
        d_far = _gauss_density(np.array([5.0]), np.array([0.0]), cov)
        assert d_center > d_far

    def test_density_is_nonneg(self):
        """Density is always >= 0."""
        rng = np.random.default_rng(0)
        y = rng.standard_normal(3)
        mean = rng.standard_normal(3)
        L = rng.standard_normal((3, 3))
        cov = L @ L.T + 0.1 * np.eye(3)
        assert _gauss_density(y, mean, cov) >= 0.0

    def test_near_singular_cov_returns_nonneg(self):
        """Degenerate (near-singular) covariance: fallback returns nonneg value, not crash."""
        y = np.array([1.0])
        mean = np.array([0.0])
        cov = np.array([[1e-300]])
        result = _gauss_density(y, mean, cov)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# _hamilton_filter — property and known-value tests
# ---------------------------------------------------------------------------

class TestHamiltonFilter:
    def test_filtered_probs_sum_to_one(self):
        """At every t, filtered probs sum to 1."""
        rng = np.random.default_rng(1)
        T, K = 15, 3
        densities = np.abs(rng.standard_normal((T, K))) + 0.01
        P = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        pi0 = np.full(K, 1.0 / K)
        filt, pred, ll = _hamilton_filter(densities, P, pi0)
        assert filt.shape == (T, K)
        assert np.allclose(filt.sum(axis=1), 1.0)

    def test_filtered_probs_in_unit_interval(self):
        rng = np.random.default_rng(2)
        T, K = 10, 2
        densities = np.abs(rng.standard_normal((T, K))) + 0.01
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        pi0 = np.array([0.5, 0.5])
        filt, pred, ll = _hamilton_filter(densities, P, pi0)
        assert np.all(filt >= 0)
        assert np.all(filt <= 1)

    def test_loglik_is_finite(self):
        rng = np.random.default_rng(3)
        T, K = 20, 2
        densities = np.abs(rng.standard_normal((T, K))) + 0.01
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        pi0 = np.array([0.5, 0.5])
        _, _, ll = _hamilton_filter(densities, P, pi0)
        assert np.isfinite(ll)

    def test_clear_signal_assigns_correct_regime(self):
        """When densities clearly favor regime 0, filtered probs should be near [1, 0]."""
        T, K = 5, 2
        densities = np.zeros((T, K))
        densities[:, 0] = 1.0       # regime 0 overwhelmingly preferred
        densities[:, 1] = 1e-10
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        pi0 = np.array([0.5, 0.5])
        filt, pred, ll = _hamilton_filter(densities, P, pi0)
        # After t=1, regime 0 prob should be very close to 1
        assert filt[1, 0] > 0.99

    def test_all_zero_densities_fallback(self):
        """All-zero densities trigger fallback; filter must still return row-stochastic output."""
        T, K = 3, 2
        densities = np.zeros((T, K))
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        pi0 = np.array([0.5, 0.5])
        filt, pred, ll = _hamilton_filter(densities, P, pi0)
        assert filt.shape == (T, K)
        assert np.allclose(filt.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# _kim_smoother — property tests
# ---------------------------------------------------------------------------

class TestKimSmoother:
    def _get_filt(self, T: int = 10, K: int = 2, seed: int = 0) -> tuple:
        rng = np.random.default_rng(seed)
        densities = np.abs(rng.standard_normal((T, K))) + 0.01
        # Build a K×K transition matrix with diagonal 0.8 and off-diagonal 0.2/(K-1)
        if K == 2:
            P = np.array([[0.85, 0.15], [0.15, 0.85]])
        else:
            off = 0.2 / (K - 1)
            P = np.full((K, K), off)
            np.fill_diagonal(P, 0.8)
        pi0 = np.full(K, 1.0 / K)
        return _hamilton_filter(densities, P, pi0), P

    def test_smoothed_probs_sum_to_one(self):
        (filt, _, _), P = self._get_filt(T=10, K=2)
        sm, pairwise = _kim_smoother(filt, P)
        assert np.allclose(sm.sum(axis=1), 1.0)

    def test_pairwise_probs_sum_to_one(self):
        (filt, _, _), P = self._get_filt(T=10, K=2)
        sm, pairwise = _kim_smoother(filt, P)
        assert pairwise.shape == (9, 2, 2)
        assert np.allclose(pairwise.sum(axis=(1, 2)), 1.0)

    def test_smoothed_probs_in_unit_interval(self):
        (filt, _, _), P = self._get_filt(T=15, K=3, seed=5)
        sm, _ = _kim_smoother(filt, P)
        assert np.all(sm >= 0)
        assert np.all(sm <= 1)

    def test_last_smoothed_equals_last_filtered(self):
        """Kim smoother boundary condition: at T-1, smoothed == filtered."""
        (filt, _, _), P = self._get_filt(T=8, K=2)
        sm, _ = _kim_smoother(filt, P)
        assert np.allclose(sm[-1], filt[-1])

    def test_output_shapes_k3(self):
        rng = np.random.default_rng(9)
        T, K = 12, 3
        densities = np.abs(rng.standard_normal((T, K))) + 0.01
        P = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        pi0 = np.full(K, 1.0 / K)
        filt, _, _ = _hamilton_filter(densities, P, pi0)
        sm, pairwise = _kim_smoother(filt, P)
        assert sm.shape == (T, K)
        assert pairwise.shape == (T - 1, K, K)


# ---------------------------------------------------------------------------
# ms_var_fit — output shapes and contract
# ---------------------------------------------------------------------------

class TestMSVarFitShapes:
    """Output shape / type contract for ms_var_fit."""

    def test_univariate_k2_p1_shapes(self):
        rng = np.random.default_rng(0)
        T, n, K, p = 80, 1, 2, 1
        Y = rng.standard_normal((T, n))
        res = ms_var_fit(Y, K=K, p=p, n_iter=20)
        assert res.A.shape == (n, n * p)
        assert res.mu.shape == (K, n)
        assert res.Sigma.shape == (K, n, n)
        assert res.P.shape == (K, K)
        assert res.smoothed_probs.shape == (T - p, K)
        assert res.filtered_probs.shape == (T - p, K)

    def test_multivariate_k2_p2_shapes(self):
        rng = np.random.default_rng(1)
        T, n, K, p = 120, 3, 2, 2
        Y = rng.standard_normal((T, n))
        res = ms_var_fit(Y, K=K, p=p, n_iter=20)
        assert res.A.shape == (n, n * p)
        assert res.mu.shape == (K, n)
        assert res.Sigma.shape == (K, n, n)
        assert res.P.shape == (K, K)
        assert res.smoothed_probs.shape == (T - p, K)
        assert res.filtered_probs.shape == (T - p, K)

    def test_k3_n2_p1_shapes(self):
        rng = np.random.default_rng(2)
        T, n, K, p = 90, 2, 3, 1
        Y = rng.standard_normal((T, n))
        res = ms_var_fit(Y, K=K, p=p, n_iter=15)
        assert res.A.shape == (n, n * p)
        assert res.mu.shape == (K, n)
        assert res.Sigma.shape == (K, n, n)
        assert res.P.shape == (K, K)
        assert res.smoothed_probs.shape == (T - p, K)
        assert res.filtered_probs.shape == (T - p, K)

    def test_k_attribute_equals_requested_k(self):
        rng = np.random.default_rng(3)
        Y = rng.standard_normal((60, 1))
        for K in [2, 3]:
            res = ms_var_fit(Y, K=K, p=1, n_iter=10)
            assert res.K == K

    def test_returns_msvar_result_dataclass(self):
        rng = np.random.default_rng(4)
        Y = rng.standard_normal((60, 2))
        res = ms_var_fit(Y, K=2, p=1, n_iter=10)
        assert isinstance(res, MSVARResult)
        assert dataclasses.is_dataclass(res)

    def test_all_required_fields_present(self):
        rng = np.random.default_rng(5)
        Y = rng.standard_normal((60, 2))
        res = ms_var_fit(Y, K=2, p=1, n_iter=10)
        expected = {
            "K", "A", "mu", "Sigma", "P",
            "smoothed_probs", "filtered_probs",
            "loglik", "n_iter", "converged",
        }
        actual = {f.name for f in dataclasses.fields(res)}
        assert expected <= actual

    def test_list_input_coercion(self):
        """ms_var_fit accepts a plain Python list and converts to float array."""
        rng = np.random.default_rng(6)
        Y = rng.standard_normal((60, 2)).tolist()
        res = ms_var_fit(Y, K=2, p=1, n_iter=10)
        assert isinstance(res.A, np.ndarray)
        assert res.A.dtype == float

    def test_dataframe_input_coercion(self):
        """ms_var_fit accepts a pandas DataFrame."""
        rng = np.random.default_rng(7)
        Y = pd.DataFrame(rng.standard_normal((60, 2)), columns=["a", "b"])
        res = ms_var_fit(Y, K=2, p=1, n_iter=10)
        assert isinstance(res.A, np.ndarray)


# ---------------------------------------------------------------------------
# ms_var_fit — probability and linear-algebra properties
# ---------------------------------------------------------------------------

class TestMSVarFitProperties:
    def test_smoothed_probs_sum_to_one(self):
        rng = np.random.default_rng(10)
        Y = rng.standard_normal((100, 2))
        res = ms_var_fit(Y, K=2, p=1, n_iter=30)
        assert np.allclose(res.smoothed_probs.sum(axis=1), 1.0)

    def test_filtered_probs_sum_to_one(self):
        rng = np.random.default_rng(11)
        Y = rng.standard_normal((100, 2))
        res = ms_var_fit(Y, K=2, p=1, n_iter=30)
        assert np.allclose(res.filtered_probs.sum(axis=1), 1.0)

    def test_smoothed_probs_in_unit_interval(self):
        rng = np.random.default_rng(12)
        Y = rng.standard_normal((80, 1))
        res = ms_var_fit(Y, K=2, p=1, n_iter=20)
        assert np.all(res.smoothed_probs >= 0)
        assert np.all(res.smoothed_probs <= 1)

    def test_filtered_probs_in_unit_interval(self):
        rng = np.random.default_rng(13)
        Y = rng.standard_normal((80, 1))
        res = ms_var_fit(Y, K=2, p=1, n_iter=20)
        assert np.all(res.filtered_probs >= 0)
        assert np.all(res.filtered_probs <= 1)

    def test_transition_matrix_rows_sum_to_one(self):
        rng = np.random.default_rng(14)
        Y = rng.standard_normal((80, 2))
        res = ms_var_fit(Y, K=2, p=1, n_iter=20)
        assert np.allclose(res.P.sum(axis=1), 1.0)

    def test_transition_matrix_entries_nonneg(self):
        rng = np.random.default_rng(15)
        Y = rng.standard_normal((80, 2))
        res = ms_var_fit(Y, K=2, p=1, n_iter=20)
        assert np.all(res.P >= 0)

    def test_sigma_symmetric(self):
        rng = np.random.default_rng(16)
        T, n, K = 100, 3, 2
        Y = rng.standard_normal((T, n))
        res = ms_var_fit(Y, K=K, p=1, n_iter=30)
        for k in range(K):
            assert np.allclose(res.Sigma[k], res.Sigma[k].T)

    def test_sigma_positive_semidefinite(self):
        rng = np.random.default_rng(17)
        T, n, K = 100, 3, 2
        Y = rng.standard_normal((T, n))
        res = ms_var_fit(Y, K=K, p=1, n_iter=30)
        for k in range(K):
            eigs = np.linalg.eigvalsh(res.Sigma[k])
            assert eigs.min() >= -1e-10  # allow for floating-point tolerance

    def test_loglik_is_finite(self):
        rng = np.random.default_rng(18)
        Y = rng.standard_normal((80, 2))
        res = ms_var_fit(Y, K=2, p=1, n_iter=20)
        assert np.isfinite(res.loglik)

    def test_final_smoothed_equals_final_filtered(self):
        """Kim smoother boundary: at the final time step, smoothed == filtered."""
        rng = np.random.default_rng(19)
        Y = rng.standard_normal((80, 2))
        res = ms_var_fit(Y, K=2, p=1, n_iter=30)
        assert np.allclose(res.smoothed_probs[-1], res.filtered_probs[-1])

    def test_seed_reproducibility(self):
        """Same seed + same data → identical results."""
        rng = np.random.default_rng(20)
        Y = rng.standard_normal((80, 2))
        res1 = ms_var_fit(Y, K=2, p=1, n_iter=30, seed=77)
        res2 = ms_var_fit(Y, K=2, p=1, n_iter=30, seed=77)
        assert np.allclose(res1.mu, res2.mu)
        assert np.allclose(res1.P, res2.P)
        assert np.allclose(res1.A, res2.A)

    def test_different_seeds_yield_valid_results(self):
        """Different seeds both produce valid (probability-feasible) results.

        Note: EM may converge to the same local optimum for different seeds on
        some data, so we only assert output validity (not that results differ).
        """
        rng = np.random.default_rng(21)
        Y = rng.standard_normal((100, 2))
        for seed in [1, 999, 12345]:
            res = ms_var_fit(Y, K=2, p=1, n_iter=30, seed=seed)
            assert np.allclose(res.smoothed_probs.sum(axis=1), 1.0)
            assert np.allclose(res.P.sum(axis=1), 1.0)

    def test_n_iter_respected_when_no_convergence(self):
        """With very low max iterations, n_iter <= n_iter_requested."""
        rng = np.random.default_rng(22)
        Y = rng.standard_normal((80, 1))
        limit = 5
        res = ms_var_fit(Y, K=2, p=1, n_iter=limit)
        assert res.n_iter <= limit
        assert not res.converged


# ---------------------------------------------------------------------------
# ms_var_fit — known-value / regime-recovery
# ---------------------------------------------------------------------------

class TestMSVarFitKnownValue:
    @pytest.mark.slow
    def test_variance_ratio_recovered(self):
        """EM recovers the variance ratio between regimes on a known DGP.

        DGP: univariate MS model, K=2, regime-specific variance only
        (no mean difference).  Sigma_low = 0.5^2 = 0.25, Sigma_high = 2.0^2 = 4.0.
        We verify the estimated ratio is >= 5 (true ratio = 16) and the sorted
        Sigma values are close to the true values.
        """
        rng = np.random.default_rng(42)
        Y, _ = _two_regime_var_data(
            T=300, rng=rng,
            sigma_low=0.5, sigma_high=2.0,
            mu_diff=0.0,
        )
        res = ms_var_fit(Y, K=2, p=1, n_iter=200, tol=1e-6)
        s0, s1 = res.Sigma[0, 0, 0], res.Sigma[1, 0, 0]
        sigma_sorted = np.sort([s0, s1])

        # True values: 0.25 and 4.0
        assert np.allclose(sigma_sorted, [0.25, 4.0], atol=0.5), (
            f"Expected ~[0.25, 4.0], got {sigma_sorted}"
        )
        # Ratio should be large
        assert sigma_sorted[1] / sigma_sorted[0] > 5.0

    @pytest.mark.slow
    def test_transition_matrix_diagonal_persistent(self):
        """Persistent regimes (p_stay=0.9) → estimated P diagonal entries > 0.7."""
        rng = np.random.default_rng(42)
        Y, _ = _two_regime_var_data(
            T=300, rng=rng,
            sigma_low=0.5, sigma_high=2.0,
            mu_diff=0.0, p_stay=0.9,
        )
        res = ms_var_fit(Y, K=2, p=1, n_iter=200, tol=1e-6)
        P_diag = np.diag(res.P)
        assert np.all(P_diag > 0.7), (
            f"Expected both diagonal P entries > 0.7, got {P_diag}"
        )


# ---------------------------------------------------------------------------
# ms_var_fit — degenerate / edge cases
# ---------------------------------------------------------------------------

class TestMSVarFitEdgeCases:
    def test_k1_degenerate_regime(self):
        """K=1: single regime, P must be [[1]], smoothed probs all ones."""
        rng = np.random.default_rng(30)
        Y = rng.standard_normal((60, 2))
        res = ms_var_fit(Y, K=1, p=1, n_iter=15)
        assert res.K == 1
        assert res.P.shape == (1, 1)
        assert np.allclose(res.P, [[1.0]])
        assert np.allclose(res.smoothed_probs, 1.0)

    def test_verbose_flag_does_not_break_output(self, capsys):
        """verbose=True produces per-iteration output but result is unchanged."""
        rng = np.random.default_rng(31)
        Y = rng.standard_normal((60, 1))
        res = ms_var_fit(Y, K=2, p=1, n_iter=5, verbose=True)
        captured = capsys.readouterr()
        assert "loglik" in captured.out
        assert res.K == 2

    def test_large_p_shapes_correct(self):
        """p=4 with moderate T: shapes are (n, n*4) for A."""
        rng = np.random.default_rng(32)
        T, n, K, p = 120, 2, 2, 4
        Y = rng.standard_normal((T, n))
        res = ms_var_fit(Y, K=K, p=p, n_iter=10)
        assert res.A.shape == (n, n * p)
        assert res.smoothed_probs.shape == (T - p, K)

    def test_short_series_k2_p1(self):
        """Minimal T: T=p+2 should run without exception (T_eff=2 after lagging)."""
        rng = np.random.default_rng(33)
        T, n, K, p = 10, 1, 2, 1
        Y = rng.standard_normal((T, n))
        res = ms_var_fit(Y, K=K, p=p, n_iter=5)
        assert res.smoothed_probs.shape[0] == T - p

    def test_high_k_shapes(self):
        """K=5 does not crash and yields correct shapes."""
        rng = np.random.default_rng(34)
        T, n, K, p = 100, 2, 5, 1
        Y = rng.standard_normal((T, n))
        res = ms_var_fit(Y, K=K, p=p, n_iter=10)
        assert res.mu.shape == (K, n)
        assert res.Sigma.shape == (K, n, n)
        assert res.P.shape == (K, K)

    def test_n_iter_1_does_not_converge(self):
        """n_iter=1 should not set converged=True."""
        rng = np.random.default_rng(35)
        Y = rng.standard_normal((80, 1))
        res = ms_var_fit(Y, K=2, p=1, n_iter=1)
        assert res.n_iter == 1
        assert not res.converged


# ---------------------------------------------------------------------------
# Defensive-branch coverage — monkeypatching internal helpers
# ---------------------------------------------------------------------------

class TestDefensiveBranches:
    """Cover defensive fallback branches that require crafted or patched inputs."""

    def test_gauss_density_linalg_error_fallback(self, monkeypatch):
        """When safe_cholesky raises LinAlgError inside _gauss_density,
        the function must return 1e-12 rather than propagate the error."""
        import puremacro.var.regime.ms_var as ms_var_mod

        def _fail_cholesky(*args, **kwargs):
            raise np.linalg.LinAlgError("forced failure for test")

        monkeypatch.setattr(ms_var_mod, "safe_cholesky", _fail_cholesky)
        result = _gauss_density(
            np.array([1.0]), np.array([0.0]), np.array([[1.0]])
        )
        assert result == 1e-12

    def test_kim_smoother_pred_next_zero_branch(self):
        """A degenerate filtered row (all zeros) triggers the pred_next[j] <= 0
        guard in _kim_smoother.  The smoother must not crash and must produce
        finite output.
        """
        T, K = 6, 2
        # Row t=2 is all zeros → after t=2, filt[2] @ P = [0, 0] → pred_next = 0
        filt = np.array([
            [0.7, 0.3],
            [0.6, 0.4],
            [0.0, 0.0],   # degenerate row
            [0.5, 0.5],
            [0.8, 0.2],
            [0.9, 0.1],
        ])
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        sm, pairwise = _kim_smoother(filt, P)
        assert sm.shape == (T, K)
        assert pairwise.shape == (T - 1, K, K)
        # All entries are finite (no NaN/Inf produced by the divide-by-zero guard)
        assert np.all(np.isfinite(sm))


# ---------------------------------------------------------------------------
# Public API / __all__
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_all_exports(self):
        import puremacro.var.regime.ms_var as mod
        assert set(mod.__all__) == {"ms_var_fit", "MSVARResult"}

    def test_msvar_result_is_dataclass(self):
        assert dataclasses.is_dataclass(MSVARResult)

    def test_msvar_result_fields(self):
        fields = {f.name for f in dataclasses.fields(MSVARResult)}
        expected = {
            "K", "A", "mu", "Sigma", "P",
            "smoothed_probs", "filtered_probs",
            "loglik", "n_iter", "converged",
        }
        assert expected <= fields
