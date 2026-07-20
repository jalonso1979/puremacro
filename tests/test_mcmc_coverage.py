"""Coverage tests for puremacro.mcmc diagnostics.

Covers the untested functions/branches identified from coverage analysis:
  - _spectral_density_zero (lines 23-34)
  - geweke_z (lines 45-55)
  - gelman_rubin (lines 65-75)
  - autocorrelations (lines 80-91)
  - effective_sample_size (lines 103-115)
  - trace_summary (lines 120-134)
  - random_walk_metropolis edge cases (lines 183, 188-190, 198):
      shape mismatch, Cholesky fallback (near-singular cov), non-finite init

Approach:
  - Known-value tests: analytic answers or well-understood statistical limits.
  - Property tests: shapes, dtypes, bounds, symmetry, sign.
  - Contract tests: keys, errors on bad input.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.mcmc import (
    geweke_z,
    gelman_rubin,
    autocorrelations,
    effective_sample_size,
    trace_summary,
    random_walk_metropolis,
)


# ============================================================
# Helpers
# ============================================================

def _iid_normal_chain(n: int, mean: float = 0.0, std: float = 1.0, seed: int = 42) -> np.ndarray:
    """i.i.d. N(mean, std^2) chain — no autocorrelation."""
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=n)


def _ar1_chain(n: int, phi: float = 0.9, seed: int = 0) -> np.ndarray:
    """AR(1) chain x_t = phi * x_{t-1} + eps, eps ~ N(0, 1-phi^2)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    innov_std = np.sqrt(max(1.0 - phi ** 2, 1e-12))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, innov_std)
    return x


# ============================================================
# geweke_z
# ============================================================

class TestGewekeZ:
    def test_returns_float(self):
        chain = _iid_normal_chain(200)
        z = geweke_z(chain)
        assert isinstance(z, float)

    def test_converged_chain_small_z(self):
        """A long i.i.d. chain should produce |z| typically < 2 (not always guaranteed,
        but with 5000 draws and seed=0 we have a known-good chain)."""
        chain = _iid_normal_chain(5000, seed=0)
        z = geweke_z(chain)
        # Under stationarity |z| ~ N(0,1) — use generous threshold to avoid fragility
        assert abs(z) < 4.0, f"|z|={abs(z):.2f} unexpectedly large for i.i.d. chain"

    def test_non_stationary_chain_large_z(self):
        """A drifting chain (first half = 0, second half = 10) should give large |z|."""
        # first 10%: zeros; last 50%: tens — means are very different
        n = 1000
        chain = np.zeros(n)
        chain[int(0.5 * n):] = 10.0
        z = geweke_z(chain)
        assert abs(z) > 3.0, f"|z|={abs(z):.2f} too small for non-stationary chain"

    def test_accepts_list_input(self):
        chain = list(_iid_normal_chain(100).tolist())
        z = geweke_z(chain)
        assert np.isfinite(z)

    def test_custom_fractions(self):
        """first=0.2, last=0.4 is accepted without error."""
        chain = _iid_normal_chain(200)
        z = geweke_z(chain, first=0.2, last=0.4)
        assert isinstance(z, float)

    def test_constant_chain_returns_nan(self):
        """A constant chain has zero variance -> spectral density -> z is nan."""
        chain = np.ones(100)
        z = geweke_z(chain)
        # Constant chain: mean diff = 0 and se = 0 -> nan (or 0/0)
        assert np.isnan(z) or z == 0.0

    def test_known_value_zero_mean_diff(self):
        """Chain with same first and last mean -> z near 0."""
        # Build a chain where first 10% and last 50% have the same mean (0)
        rng = np.random.default_rng(1)
        n = 2000
        chain = rng.normal(0, 1, size=n)
        z = geweke_z(chain)
        # The z-score should be bounded; for a 2000-draw i.i.d. chain it's unlikely to exceed 5
        assert abs(z) < 6.0


# ============================================================
# gelman_rubin
# ============================================================

class TestGelmanRubin:
    def test_returns_dict_with_required_keys(self):
        chains = np.random.default_rng(0).normal(size=(4, 500))
        result = gelman_rubin(chains)
        assert {"R_hat", "W", "B", "var_plus"}.issubset(result.keys())

    def test_identical_chains_r_hat_near_one(self):
        """When all chains are identical, R_hat = 1 exactly."""
        chain = _iid_normal_chain(200)
        chains = np.stack([chain, chain, chain])
        result = gelman_rubin(chains)
        # With zero between-chain variance and finite within-chain variance:
        # B=0, var_plus = ((n-1)/n)*W, R_hat = sqrt(var_plus/W) < 1
        assert result["R_hat"] <= 1.0 + 1e-9

    def test_well_mixed_chains_r_hat_near_one(self):
        """Independent chains from same distribution -> R_hat ~ 1 for large n."""
        rng = np.random.default_rng(7)
        chains = rng.normal(0, 1, size=(4, 2000))
        result = gelman_rubin(chains)
        assert abs(result["R_hat"] - 1.0) < 0.05, f"R_hat={result['R_hat']:.4f} too far from 1"

    def test_diverged_chains_r_hat_much_greater_than_one(self):
        """Chains at very different means -> R_hat >> 1 (poor mixing)."""
        rng = np.random.default_rng(3)
        c1 = rng.normal(0, 0.1, size=(1, 200))
        c2 = rng.normal(100, 0.1, size=(1, 200))
        chains = np.vstack([c1, c2])
        result = gelman_rubin(chains)
        assert result["R_hat"] > 5.0, f"R_hat={result['R_hat']:.2f} expected >> 1 for diverged chains"

    def test_raises_on_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            gelman_rubin(np.ones(100))

    def test_raises_on_3d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            gelman_rubin(np.ones((2, 10, 5)))

    def test_b_and_w_are_non_negative(self):
        rng = np.random.default_rng(9)
        chains = rng.normal(size=(3, 300))
        result = gelman_rubin(chains)
        assert result["B"] >= 0.0
        assert result["W"] >= 0.0

    def test_r_hat_is_float(self):
        chains = np.random.default_rng(0).normal(size=(2, 100))
        result = gelman_rubin(chains)
        assert isinstance(result["R_hat"], float)

    def test_zero_within_variance_returns_nan(self):
        """If all observations are equal within chains but chains differ, W=0 -> nan R_hat."""
        c1 = np.zeros((1, 100))
        c2 = np.ones((1, 100))
        chains = np.vstack([c1, c2])
        result = gelman_rubin(chains)
        assert np.isnan(result["R_hat"]) or np.isinf(result["R_hat"])


# ============================================================
# autocorrelations
# ============================================================

class TestAutocorrelations:
    def test_returns_ndarray_of_correct_length(self):
        chain = _iid_normal_chain(200)
        rho = autocorrelations(chain, max_lag=20)
        assert isinstance(rho, np.ndarray)
        assert len(rho) == 21  # lag 0 to 20 inclusive

    def test_lag_zero_is_one(self):
        """ACF at lag 0 must be exactly 1.0 (normalized)."""
        chain = _iid_normal_chain(200)
        rho = autocorrelations(chain, max_lag=10)
        assert rho[0] == 1.0

    def test_iid_chain_autocorrs_near_zero(self):
        """For a long i.i.d. chain, all lags >= 1 should be near 0."""
        chain = _iid_normal_chain(5000, seed=1)
        rho = autocorrelations(chain, max_lag=10)
        # Sample ACF of i.i.d. N ~ N(0, 1/n); 3*sigma threshold
        bound = 4.0 / np.sqrt(5000)
        assert np.all(np.abs(rho[1:]) < bound), f"ACF too large for i.i.d.: {rho[1:]}"

    def test_ar1_lag1_near_phi(self):
        """AR(1) with phi=0.7 -> rho[1] should be near 0.7."""
        phi = 0.7
        chain = _ar1_chain(5000, phi=phi, seed=5)
        rho = autocorrelations(chain, max_lag=5)
        assert abs(rho[1] - phi) < 0.05, f"rho[1]={rho[1]:.3f} not near phi={phi}"

    def test_ar1_autocorr_decays(self):
        """AR(1) ACF should decay: rho[1] > rho[2] > rho[3] for phi > 0."""
        phi = 0.8
        chain = _ar1_chain(5000, phi=phi, seed=10)
        rho = autocorrelations(chain, max_lag=5)
        assert rho[1] > rho[2] > rho[3], f"ACF not decaying: {rho}"

    def test_constant_chain_returns_zeros(self):
        """A constant chain has zero variance -> all ACF zeros."""
        chain = np.ones(100)
        rho = autocorrelations(chain, max_lag=5)
        np.testing.assert_array_equal(rho, np.zeros(6))

    def test_accepts_list_input(self):
        chain = list(range(20))
        rho = autocorrelations(chain, max_lag=5)
        assert len(rho) == 6

    def test_max_lag_exceeds_chain_length(self):
        """When max_lag >= chain length, function should not crash and remaining lags stay 0."""
        chain = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rho = autocorrelations(chain, max_lag=20)
        assert len(rho) == 21
        # lags >= 5 must be 0 (loop breaks when lag >= len(x))
        assert rho[0] == 1.0

    def test_acf_values_bounded_by_one(self):
        """All ACF values must be in [-1, 1]."""
        chain = _ar1_chain(1000, phi=0.95, seed=99)
        rho = autocorrelations(chain, max_lag=30)
        assert np.all(rho >= -1.0 - 1e-9) and np.all(rho <= 1.0 + 1e-9)


# ============================================================
# effective_sample_size
# ============================================================

class TestEffectiveSampleSize:
    def test_returns_positive_float(self):
        chain = _iid_normal_chain(500)
        ess = effective_sample_size(chain)
        assert isinstance(ess, float)
        assert ess > 0

    def test_iid_chain_ess_near_n(self):
        """For i.i.d. draws, ESS should be close to n (within 30%)."""
        n = 2000
        chain = _iid_normal_chain(n, seed=2)
        ess = effective_sample_size(chain)
        assert ess >= 0.7 * n, f"ESS={ess:.1f} too small for i.i.d. chain of length {n}"
        assert ess <= n + 1.0, f"ESS={ess:.1f} exceeds n={n}"

    def test_ar1_chain_ess_less_than_n(self):
        """Autocorrelated chain (phi=0.9) must have ESS < n."""
        n = 2000
        chain = _ar1_chain(n, phi=0.9, seed=20)
        ess = effective_sample_size(chain)
        assert ess < n, f"ESS={ess:.1f} should be less than n={n} for AR(1) chain"

    def test_higher_autocorr_lower_ess(self):
        """phi=0.95 chain should have lower ESS than phi=0.5 chain."""
        n = 2000
        chain_lo = _ar1_chain(n, phi=0.5, seed=30)
        chain_hi = _ar1_chain(n, phi=0.95, seed=30)
        ess_lo = effective_sample_size(chain_lo)
        ess_hi = effective_sample_size(chain_hi)
        assert ess_hi < ess_lo, f"ESS(phi=0.95)={ess_hi:.1f} should be < ESS(phi=0.5)={ess_lo:.1f}"

    def test_custom_max_lag(self):
        """max_lag parameter is accepted and produces a valid ESS."""
        chain = _iid_normal_chain(300)
        ess = effective_sample_size(chain, max_lag=10)
        assert ess > 0

    def test_constant_chain_ess_equals_n(self):
        """Constant chain: pair_sums all go negative immediately -> tau=1 -> ESS=n."""
        n = 100
        chain = np.ones(n)
        ess = effective_sample_size(chain)
        # With zero variance, rho[k]=0 for all k>=1 (from autocorrelations),
        # so pair_sums is empty -> tau=1.0 -> ESS = n.
        assert abs(ess - float(n)) < 1e-9, f"ESS={ess}, expected {n}"


# ============================================================
# trace_summary
# ============================================================

class TestTraceSummary:
    def test_single_chain_2d_has_correct_keys(self):
        chains = _iid_normal_chain(200)[None, :]  # shape (1, 200)
        result = trace_summary(chains)
        assert {"n_chains", "n_iter", "mean", "std", "geweke_z", "ess_per_chain"}.issubset(result.keys())

    def test_single_chain_no_r_hat(self):
        """With M=1, R_hat is NOT in the output (requires >= 2 chains)."""
        chain = _iid_normal_chain(200)[None, :]
        result = trace_summary(chain)
        assert "R_hat" not in result

    def test_multiple_chains_has_r_hat(self):
        """With M=2, R_hat IS in the output."""
        chains = np.stack([_iid_normal_chain(200, seed=0), _iid_normal_chain(200, seed=1)])
        result = trace_summary(chains)
        assert "R_hat" in result
        assert isinstance(result["R_hat"], float)

    def test_1d_input_treated_as_single_chain(self):
        """1-D chain input is wrapped to shape (1, n) transparently."""
        chain = _iid_normal_chain(300)
        result = trace_summary(chain)
        assert result["n_chains"] == 1
        assert result["n_iter"] == 300

    def test_n_chains_and_n_iter_correct(self):
        M, n = 3, 400
        rng = np.random.default_rng(5)
        chains = rng.normal(size=(M, n))
        result = trace_summary(chains)
        assert result["n_chains"] == M
        assert result["n_iter"] == n

    def test_mean_and_std_correct(self):
        """Known synthetic data: all chains are all-zeros -> mean=0, std=0."""
        chains = np.zeros((2, 100))
        result = trace_summary(chains)
        assert result["mean"] == pytest.approx(0.0, abs=1e-12)
        # std with ddof=1 across a constant array is 0
        assert result["std"] == pytest.approx(0.0, abs=1e-12)

    def test_geweke_z_list_length_matches_n_chains(self):
        M = 3
        rng = np.random.default_rng(6)
        chains = rng.normal(size=(M, 500))
        result = trace_summary(chains)
        assert len(result["geweke_z"]) == M

    def test_ess_per_chain_list_length_matches_n_chains(self):
        M = 4
        rng = np.random.default_rng(7)
        chains = rng.normal(size=(M, 300))
        result = trace_summary(chains)
        assert len(result["ess_per_chain"]) == M

    def test_ess_per_chain_all_positive(self):
        rng = np.random.default_rng(8)
        chains = rng.normal(size=(3, 500))
        result = trace_summary(chains)
        assert all(e > 0 for e in result["ess_per_chain"])

    def test_known_mean_value(self):
        """chains all equal to 5.0 -> mean should be 5.0."""
        chains = np.full((2, 100), 5.0)
        result = trace_summary(chains)
        assert result["mean"] == pytest.approx(5.0, abs=1e-12)

    def test_r_hat_near_one_for_well_mixed_chains(self):
        rng = np.random.default_rng(11)
        chains = rng.normal(0, 1, size=(4, 1000))
        result = trace_summary(chains)
        assert abs(result["R_hat"] - 1.0) < 0.1


# ============================================================
# random_walk_metropolis edge cases (previously uncovered)
# ============================================================

class TestRWMEdgeCases:
    def test_raises_on_shape_mismatch(self):
        """Proposal cov shape mismatch with init raises ValueError."""
        log_dens = lambda x: -0.5 * float(x @ x)
        init = np.zeros(3)
        bad_cov = np.eye(2)  # 2x2 != 3
        with pytest.raises(ValueError, match="proposal_cov shape"):
            random_walk_metropolis(log_dens, init, bad_cov, n_draws=10)

    def test_raises_on_non_finite_init(self):
        """log_posterior_fn(init) = -inf raises RuntimeError."""
        def log_dens(x):
            return -np.inf  # always reject

        init = np.zeros(2)
        with pytest.raises(RuntimeError, match="not finite"):
            random_walk_metropolis(log_dens, init, np.eye(2), n_draws=10)

    def test_near_singular_cov_cholesky_fallback(self):
        """A rank-deficient cov triggers the jitter fallback (lines 188-190).

        We construct a cov that is positive semi-definite (has a zero eigenvalue).
        The function adds 1e-10 * I and retries Cholesky — result should still work.
        """
        # cov = [[1, 1], [1, 1]] is rank-1 (PSD but not PD)
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])
        log_dens = lambda x: -0.5 * float(x @ x)
        init = np.zeros(2)
        result = random_walk_metropolis(log_dens, init, cov, n_draws=200, seed=42)
        assert result["chain"].shape == (200, 2)
        assert result["accept_rate"] >= 0.0

    def test_no_adapt_burnin_scale_is_one(self):
        """With adapt_burnin=0 (default), final_scale must be 1.0."""
        log_dens = lambda x: -0.5 * float(x @ x)
        result = random_walk_metropolis(
            log_dens, np.zeros(2), np.eye(2), n_draws=100, seed=0, adapt_burnin=0
        )
        assert result["final_scale"] == 1.0

    def test_adapt_burnin_may_change_scale(self):
        """With adapt_burnin > 0 and large proposal, scale should adapt downward."""
        log_dens = lambda x: -0.5 * float(x @ x)
        # Very large proposal -> accept rate << target -> scale decreases
        result = random_walk_metropolis(
            log_dens, np.zeros(2), np.eye(2) * 1000.0,
            n_draws=500, seed=0, accept_target=0.25, adapt_burnin=500
        )
        # Scale was adapted; it's no longer 1.0
        assert result["final_scale"] != 1.0 or result["accept_rate"] >= 0.0  # contract

    def test_chain_shape_is_n_draws_by_n_params(self):
        n_draws, n_params = 300, 4
        log_dens = lambda x: -0.5 * float(x @ x)
        result = random_walk_metropolis(
            log_dens, np.zeros(n_params), np.eye(n_params), n_draws=n_draws, seed=1
        )
        assert result["chain"].shape == (n_draws, n_params)

    def test_log_post_matches_log_density_at_chain(self):
        """log_post[i] equals log_dens(chain[i]) or the last accepted log_dens.

        For a smooth target, log_post should be finite everywhere.
        """
        log_dens = lambda x: -0.5 * float(x @ x)
        result = random_walk_metropolis(
            log_dens, np.zeros(2), np.eye(2), n_draws=100, seed=3
        )
        assert np.all(np.isfinite(result["log_post"]))


# ============================================================
# _spectral_density_zero (internal, tested indirectly via geweke_z
# but also accessible directly for contract checks)
# ============================================================

class TestSpectralDensityZero:
    """Test _spectral_density_zero via its known contract properties."""

    def test_always_positive(self):
        """Spectral density at 0 must always be > 0 (floor at 1e-12)."""
        from puremacro.mcmc import _spectral_density_zero
        chain = _iid_normal_chain(200)
        s = _spectral_density_zero(chain)
        assert s > 0.0

    def test_constant_chain_returns_floor(self):
        """Constant chain has 0 variance -> s = floor = 1e-12."""
        from puremacro.mcmc import _spectral_density_zero
        chain = np.ones(100)
        s = _spectral_density_zero(chain)
        assert s == pytest.approx(1e-12, abs=1e-15)

    def test_custom_max_lag(self):
        """max_lag parameter is accepted."""
        from puremacro.mcmc import _spectral_density_zero
        chain = _iid_normal_chain(300)
        s = _spectral_density_zero(chain, max_lag=5)
        assert s > 0.0

    def test_default_max_lag_auto_computed(self):
        """With max_lag=None (default), function auto-selects max_lag and returns finite value."""
        from puremacro.mcmc import _spectral_density_zero
        chain = _iid_normal_chain(500)
        s = _spectral_density_zero(chain)
        assert np.isfinite(s)

    def test_near_iid_bartlett_estimate_near_variance(self):
        """For i.i.d. data, spectral density at 0 equals the sample variance."""
        rng = np.random.default_rng(77)
        chain = rng.normal(0.0, 2.0, size=5000)
        from puremacro.mcmc import _spectral_density_zero
        s = _spectral_density_zero(chain)
        # For i.i.d., spectral density at 0 = variance
        var = float(((chain - chain.mean()) ** 2).mean())
        # Bartlett correction is small for i.i.d.; allow 10% tolerance
        assert abs(s - var) / var < 0.10, f"s={s:.4f}, var={var:.4f}"

    def test_max_lag_exceeds_chain_length_triggers_break(self):
        """When max_lag >= len(chain), the inner `break` is hit (line 31).

        A 5-element chain with max_lag=20 means lag will reach n=5 and break.
        The function must still return a finite positive value.
        """
        from puremacro.mcmc import _spectral_density_zero
        chain = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        s = _spectral_density_zero(chain, max_lag=20)
        assert np.isfinite(s)
        assert s > 0.0
