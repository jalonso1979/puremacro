"""Tests for puremacro.var.tvp — TVP-VAR and TVP-VAR-SV Gibbs samplers.

Coverage strategy
-----------------
- Output shape/dtype contracts for both estimators (tvp_var_fit, tvp_var_sv_fit).
- Key structural properties: Sigma/Q positive-definite, A lower-triangular,
  exp(log_h) > 0, W > 0.
- Known-value: posterior mean of intercept recovers the DGP mean.
- Monotonicity: looser Q prior -> larger Q posterior draws.
- Reproducibility: same seed -> identical draws.
- rng=None branch executes without error.
- Private helpers: _inv_wishart (shape, symmetry, PD), _sample_mixture_indicators
  (range 0..6, valid probabilities), _kalman_ffbs (shape, finiteness, tight-Q
  property), _ffbs_sv (shape, finiteness).
- KSC mixture constants: 7 components, positive weights/variances summing to 1.
- Edge: short T (barely more than burn-in variables), p=2, univariate n=1.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.var.tvp import (
    TVPVARResult,
    TVPVARSVResult,
    _KSC_MEANS,
    _KSC_VARS,
    _KSC_WEIGHTS,
    _ffbs_sv,
    _inv_wishart,
    _kalman_ffbs,
    _kalman_ffbs_tv,
    _sample_mixture_indicators,
    tvp_var_fit,
    tvp_var_sv_fit,
)


# ---------------------------------------------------------------------------
# Shared data generator helpers
# ---------------------------------------------------------------------------

def _make_var_data(T: int = 80, n: int = 2, p: int = 1, seed: int = 42) -> np.ndarray:
    """Simulate a stable VAR(p) with fixed coefficients."""
    rng = np.random.default_rng(seed)
    A = 0.4 * np.eye(n)
    Y = np.zeros((T, n))
    for t in range(p, T):
        for k in range(1, p + 1):
            Y[t] += A @ Y[t - k]
        Y[t] += 0.4 * rng.standard_normal(n)
    return Y


# ===========================================================================
# KSC mixture constants
# ===========================================================================

def test_ksc_weights_sum_to_one():
    assert abs(_KSC_WEIGHTS.sum() - 1.0) < 1e-10


def test_ksc_weights_positive():
    assert np.all(_KSC_WEIGHTS > 0)


def test_ksc_variances_positive():
    assert np.all(_KSC_VARS > 0)


def test_ksc_seven_components():
    assert len(_KSC_WEIGHTS) == 7
    assert len(_KSC_MEANS) == 7
    assert len(_KSC_VARS) == 7


# ===========================================================================
# _inv_wishart helper
# ===========================================================================

def test_inv_wishart_shape():
    rng = np.random.default_rng(0)
    n = 3
    W = _inv_wishart(np.eye(n) * 2, df=n + 5, rng=rng)
    assert W.shape == (n, n)


def test_inv_wishart_symmetric():
    rng = np.random.default_rng(1)
    n = 4
    W = _inv_wishart(np.eye(n), df=n + 3, rng=rng)
    assert np.allclose(W, W.T), "Inverse-Wishart draw must be symmetric"


def test_inv_wishart_positive_definite():
    rng = np.random.default_rng(2)
    n = 3
    W = _inv_wishart(np.eye(n), df=n + 10, rng=rng)
    eigvals = np.linalg.eigvalsh(W)
    assert np.all(eigvals > 0), f"Inverse-Wishart not PD; min eigenvalue = {eigvals.min()}"


def test_inv_wishart_larger_scale_larger_mean():
    """Scaling S up should shift the IW draw up (in Frobenius norm) on average."""
    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(10)
    n, df = 3, 10
    draws_small = np.array([_inv_wishart(np.eye(n), df, rng1) for _ in range(30)])
    draws_large = np.array([_inv_wishart(2 * np.eye(n), df, rng2) for _ in range(30)])
    mean_small = np.trace(draws_small.mean(0))
    mean_large = np.trace(draws_large.mean(0))
    assert mean_large > mean_small, "Larger S should produce larger IW draws"


# ===========================================================================
# _sample_mixture_indicators
# ===========================================================================

def test_mixture_indicators_shape():
    rng = np.random.default_rng(3)
    T = 50
    y_star = rng.standard_normal(T)
    h = rng.standard_normal(T) * 0.5
    s = _sample_mixture_indicators(y_star, h, rng)
    assert s.shape == (T,)


def test_mixture_indicators_valid_range():
    rng = np.random.default_rng(4)
    T = 200
    y_star = rng.standard_normal(T) - 1.27
    h = rng.standard_normal(T)
    s = _sample_mixture_indicators(y_star, h, rng)
    assert np.all((s >= 0) & (s <= 6)), "Mixture indicators must be in {0, ..., 6}"


def test_mixture_indicators_integer_type():
    rng = np.random.default_rng(5)
    T = 30
    y_star = rng.standard_normal(T)
    h = np.zeros(T)
    s = _sample_mixture_indicators(y_star, h, rng)
    # Result should be integer-valued (numpy fancy indexing depends on this)
    assert s.dtype.kind == 'i' or np.all(s == s.astype(int))


# ===========================================================================
# _ffbs_sv helper
# ===========================================================================

def test_ffbs_sv_shape():
    rng = np.random.default_rng(6)
    T = 40
    y_star = rng.standard_normal(T) * 2 - 1.27
    m_t = np.zeros(T)
    v_t = np.ones(T)
    h_draw = _ffbs_sv(y_star, m_t, v_t, w=0.1, h0=0.0, p0=10.0, rng=rng)
    assert h_draw.shape == (T,)


def test_ffbs_sv_finite():
    rng = np.random.default_rng(7)
    T = 50
    y_star = rng.standard_normal(T) - 1.27
    m_t = rng.standard_normal(T) * 0.5
    v_t = np.ones(T) * 0.5
    h_draw = _ffbs_sv(y_star, m_t, v_t, w=0.05, h0=0.0, p0=5.0, rng=rng)
    assert np.all(np.isfinite(h_draw)), "FFBS-SV draw must be finite"


def test_ffbs_sv_tight_w_near_constant():
    """With w -> 0 (no SV innovation), log h path should barely drift."""
    rng = np.random.default_rng(8)
    T = 40
    y_star = rng.standard_normal(T) - 1.27
    m_t = np.zeros(T)
    v_t = np.ones(T) * 0.5
    h_draw = _ffbs_sv(y_star, m_t, v_t, w=1e-10, h0=0.0, p0=0.001, rng=rng)
    std_across_time = h_draw.std()
    assert std_across_time < 0.1, f"Expected near-constant h under tight w; got std={std_across_time}"


# ===========================================================================
# _kalman_ffbs helper
# ===========================================================================

def test_kalman_ffbs_shape():
    rng = np.random.default_rng(9)
    T_eff, n, k = 30, 2, 4
    Y_dep = rng.standard_normal((T_eff, n)) * 0.3
    X_full = rng.standard_normal((T_eff, n, k)) * 0.1
    Sigma = np.eye(n)
    Q = 0.01 * np.eye(k)
    a0 = np.zeros(k)
    P0 = np.eye(k)
    beta = _kalman_ffbs(Y_dep, X_full, Sigma, Q, a0, P0, rng)
    assert beta.shape == (T_eff, k)


def test_kalman_ffbs_finite():
    rng = np.random.default_rng(10)
    T_eff, n, k = 25, 2, 4
    Y_dep = rng.standard_normal((T_eff, n)) * 0.5
    X_full = rng.standard_normal((T_eff, n, k)) * 0.2
    Sigma = np.eye(n)
    Q = 0.01 * np.eye(k)
    a0 = np.zeros(k)
    P0 = np.eye(k)
    beta = _kalman_ffbs(Y_dep, X_full, Sigma, Q, a0, P0, rng)
    assert np.all(np.isfinite(beta)), "FFBS draw must be finite"


def test_kalman_ffbs_tight_Q_gives_small_drift():
    """With near-zero process noise Q, coefficients should barely change over time."""
    rng = np.random.default_rng(11)
    T_eff, n, k = 40, 2, 4
    Y_dep = rng.standard_normal((T_eff, n)) * 0.2
    X_full = np.random.default_rng(0).standard_normal((T_eff, n, k)) * 0.1
    Sigma = 0.01 * np.eye(n)
    Q_tight = 1e-8 * np.eye(k)
    a0 = np.zeros(k)
    P0 = 0.01 * np.eye(k)
    beta = _kalman_ffbs(Y_dep, X_full, Sigma, Q_tight, a0, P0, rng)
    std_over_time = beta.std(axis=0).max()
    assert std_over_time < 0.01, f"Tight Q should give nearly constant beta; got std={std_over_time}"


# ===========================================================================
# _kalman_ffbs_tv helper
# ===========================================================================

def test_kalman_ffbs_tv_shape():
    rng = np.random.default_rng(12)
    T_eff, n, k = 20, 2, 4
    Y_dep = rng.standard_normal((T_eff, n)) * 0.3
    X_full = rng.standard_normal((T_eff, n, k)) * 0.1
    Sigma_t = np.stack([np.eye(n)] * T_eff)
    Q = 0.01 * np.eye(k)
    a0 = np.zeros(k)
    P0 = np.eye(k)
    beta = _kalman_ffbs_tv(Y_dep, X_full, Sigma_t, Q, a0, P0, rng)
    assert beta.shape == (T_eff, k)


def test_kalman_ffbs_tv_finite():
    rng = np.random.default_rng(13)
    T_eff, n, k = 25, 2, 4
    Y_dep = rng.standard_normal((T_eff, n)) * 0.5
    X_full = rng.standard_normal((T_eff, n, k)) * 0.2
    Sigma_t = np.stack([0.5 * np.eye(n) for _ in range(T_eff)])
    Q = 0.01 * np.eye(k)
    a0 = np.zeros(k)
    P0 = np.eye(k)
    beta = _kalman_ffbs_tv(Y_dep, X_full, Sigma_t, Q, a0, P0, rng)
    assert np.all(np.isfinite(beta))


# ===========================================================================
# tvp_var_fit — output contract
# ===========================================================================

def test_tvp_var_fit_returns_result_type():
    Y = _make_var_data(T=60, n=2, p=1)
    result = tvp_var_fit(Y, p=1, n_draws=5, burn=5, rng=np.random.default_rng(0))
    assert isinstance(result, TVPVARResult)


def test_tvp_var_fit_shape_contract():
    T, n, p = 60, 2, 1
    n_draws = 8
    Y = _make_var_data(T=T, n=n, p=p)
    result = tvp_var_fit(Y, p=p, n_draws=n_draws, burn=5, rng=np.random.default_rng(0))
    T_eff = T - p
    k_x = 1 + n * p
    k_state = n * k_x
    assert result.beta_draws.shape == (n_draws, T_eff, k_state)
    assert result.Sigma_draws.shape == (n_draws, n, n)
    assert result.Q_draws.shape == (n_draws, k_state, k_state)
    assert result.n == n
    assert result.p == p
    assert result.T_eff == T_eff
    assert result.n_draws == n_draws


def test_tvp_var_fit_p2_shape():
    """Shape with p=2 lags."""
    T, n, p = 80, 3, 2
    n_draws = 5
    Y = _make_var_data(T=T, n=n, p=p, seed=1)
    result = tvp_var_fit(Y, p=p, n_draws=n_draws, burn=5, rng=np.random.default_rng(1))
    T_eff = T - p
    k_x = 1 + n * p    # 1 + 3*2 = 7
    k_state = n * k_x  # 3 * 7 = 21
    assert result.beta_draws.shape == (n_draws, T_eff, k_state)


def test_tvp_var_fit_univariate():
    """n=1 univariate case should work without error."""
    T = 50
    Y = np.random.default_rng(2).standard_normal((T, 1)) * 0.5
    result = tvp_var_fit(Y, p=1, n_draws=5, burn=5, rng=np.random.default_rng(2))
    assert result.n == 1
    assert result.beta_draws.shape[2] == 2  # k_x = 1+1*1 = 2; k_state = 1*2 = 2


def test_tvp_var_fit_sigma_symmetric_pd():
    """Sigma draws must be symmetric and positive definite."""
    Y = _make_var_data(T=80, n=2)
    result = tvp_var_fit(Y, p=1, n_draws=20, burn=10, rng=np.random.default_rng(3))
    for d in range(result.n_draws):
        S = result.Sigma_draws[d]
        assert np.allclose(S, S.T, atol=1e-12), f"Sigma[{d}] not symmetric"
        eigs = np.linalg.eigvalsh(S)
        assert np.all(eigs > 0), f"Sigma[{d}] not PD; min eig = {eigs.min()}"


def test_tvp_var_fit_Q_symmetric_pd():
    """Q draws must be symmetric and positive definite."""
    Y = _make_var_data(T=80, n=2)
    result = tvp_var_fit(Y, p=1, n_draws=20, burn=10, rng=np.random.default_rng(4))
    for d in range(result.n_draws):
        Q = result.Q_draws[d]
        assert np.allclose(Q, Q.T, atol=1e-12), f"Q[{d}] not symmetric"
        eigs = np.linalg.eigvalsh(Q)
        assert np.all(eigs > 0), f"Q[{d}] not PD; min eig = {eigs.min()}"


def test_tvp_var_fit_beta_finite():
    Y = _make_var_data(T=80, n=2)
    result = tvp_var_fit(Y, p=1, n_draws=10, burn=5, rng=np.random.default_rng(5))
    assert np.all(np.isfinite(result.beta_draws)), "beta_draws contains non-finite values"


def test_tvp_var_fit_reproducible():
    Y = _make_var_data(T=60, n=2, seed=7)
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    r1 = tvp_var_fit(Y, p=1, n_draws=5, burn=5, rng=rng_a)
    r2 = tvp_var_fit(Y, p=1, n_draws=5, burn=5, rng=rng_b)
    assert np.allclose(r1.beta_draws, r2.beta_draws)
    assert np.allclose(r1.Sigma_draws, r2.Sigma_draws)
    assert np.allclose(r1.Q_draws, r2.Q_draws)


def test_tvp_var_fit_rng_none_runs():
    """rng=None should auto-create an RNG without error."""
    Y = _make_var_data(T=50, n=2)
    result = tvp_var_fit(Y, p=1, n_draws=3, burn=3, rng=None)
    assert result.n_draws == 3


def test_tvp_var_fit_different_seeds_differ():
    """Two independent seeds should produce different posterior draws."""
    Y = _make_var_data(T=60, n=2)
    r1 = tvp_var_fit(Y, p=1, n_draws=5, burn=5, rng=np.random.default_rng(0))
    r2 = tvp_var_fit(Y, p=1, n_draws=5, burn=5, rng=np.random.default_rng(1))
    assert not np.allclose(r1.beta_draws, r2.beta_draws)


def test_tvp_var_fit_Q_prior_monotonicity():
    """Larger Q_prior_scale → larger posterior Q diagonal entries."""
    Y = _make_var_data(T=100, n=2, seed=11)
    r_tight = tvp_var_fit(Y, p=1, n_draws=30, burn=10,
                           Q_prior_scale=1e-6, rng=np.random.default_rng(1))
    r_loose = tvp_var_fit(Y, p=1, n_draws=30, burn=10,
                           Q_prior_scale=0.1, rng=np.random.default_rng(1))
    mean_q_tight = np.mean([np.diag(r_tight.Q_draws[d]).mean() for d in range(r_tight.n_draws)])
    mean_q_loose = np.mean([np.diag(r_loose.Q_draws[d]).mean() for d in range(r_loose.n_draws)])
    assert mean_q_loose > mean_q_tight, (
        f"Larger Q_prior should yield larger Q draws; got tight={mean_q_tight}, loose={mean_q_loose}"
    )


def test_tvp_var_fit_recovers_intercept():
    """Posterior mean of intercept coefficients should be close to the DGP mean.

    Known-value test: Y_t = mu + eps_t, so the VAR intercept should track mu.
    k_x = 1 + n*p = 3 (n=2, p=1). Intercept for eq 0 is beta[:, :, 0];
    intercept for eq 1 is beta[:, :, 3].
    """
    rng = np.random.default_rng(42)
    T, n, p = 200, 2, 1
    mu = np.array([0.5, -0.3])
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = mu + 0.3 * rng.standard_normal(n)

    result = tvp_var_fit(Y, p=p, n_draws=100, burn=200, rng=rng)
    k_x = 1 + n * p  # = 3
    mean_c0 = result.beta_draws[:, :, 0].mean()
    mean_c1 = result.beta_draws[:, :, k_x].mean()
    assert abs(mean_c0 - mu[0]) < 0.15, f"Intercept eq0 off: {mean_c0:.3f} vs {mu[0]}"
    assert abs(mean_c1 - mu[1]) < 0.15, f"Intercept eq1 off: {mean_c1:.3f} vs {mu[1]}"


def test_tvp_var_fit_short_series():
    """Short T (barely more than p) must run without error."""
    Y = _make_var_data(T=20, n=2, p=1)
    result = tvp_var_fit(Y, p=1, n_draws=5, burn=5, rng=np.random.default_rng(0))
    assert result.T_eff == 19


# ===========================================================================
# tvp_var_sv_fit — output contract
# ===========================================================================

def test_tvp_var_sv_fit_returns_result_type():
    Y = _make_var_data(T=60, n=2)
    result = tvp_var_sv_fit(Y, p=1, n_draws=5, burn=5, rng=np.random.default_rng(0))
    assert isinstance(result, TVPVARSVResult)


def test_tvp_var_sv_fit_shape_contract():
    T, n, p = 60, 2, 1
    n_draws = 8
    Y = _make_var_data(T=T, n=n, p=p)
    result = tvp_var_sv_fit(Y, p=p, n_draws=n_draws, burn=5, rng=np.random.default_rng(0))
    T_eff = T - p
    k_x = 1 + n * p
    k_state = n * k_x
    assert result.beta_draws.shape == (n_draws, T_eff, k_state)
    assert result.log_h_draws.shape == (n_draws, T_eff, n)
    assert result.A_draws.shape == (n_draws, n, n)
    assert result.Q_draws.shape == (n_draws, k_state, k_state)
    assert result.W_draws.shape == (n_draws, n)
    assert result.n == n
    assert result.p == p
    assert result.T_eff == T_eff
    assert result.n_draws == n_draws


def test_tvp_var_sv_fit_h_positive():
    """exp(log_h) must be strictly positive."""
    Y = _make_var_data(T=70, n=2)
    result = tvp_var_sv_fit(Y, p=1, n_draws=10, burn=10, rng=np.random.default_rng(1))
    h = np.exp(result.log_h_draws)
    assert np.all(h > 0), "h = exp(log_h) must be strictly positive"


def test_tvp_var_sv_fit_W_positive():
    """W draws (variances of log-h innovations) must be positive."""
    Y = _make_var_data(T=70, n=2)
    result = tvp_var_sv_fit(Y, p=1, n_draws=10, burn=10, rng=np.random.default_rng(2))
    assert np.all(result.W_draws > 0), "W draws must be strictly positive"


def test_tvp_var_sv_fit_A_lower_triangular():
    """A_draws must be lower triangular with ones on the diagonal."""
    T, n = 70, 3
    Y = _make_var_data(T=T, n=n)
    result = tvp_var_sv_fit(Y, p=1, n_draws=10, burn=10, rng=np.random.default_rng(3))
    for d in range(result.n_draws):
        A = result.A_draws[d]
        # Diagonal must be exactly 1 (not updated in Gibbs)
        assert np.allclose(np.diag(A), 1.0), f"A[{d}] diagonal not all 1"
        # Upper triangle must be 0
        upper = A[np.triu_indices(n, 1)]
        assert np.allclose(upper, 0.0, atol=1e-12), f"A[{d}] has nonzero upper triangle"


def test_tvp_var_sv_fit_beta_finite():
    Y = _make_var_data(T=70, n=2)
    result = tvp_var_sv_fit(Y, p=1, n_draws=10, burn=10, rng=np.random.default_rng(4))
    assert np.all(np.isfinite(result.beta_draws))
    assert np.all(np.isfinite(result.log_h_draws))


def test_tvp_var_sv_fit_reproducible():
    Y = _make_var_data(T=60, n=2, seed=3)
    rng_a = np.random.default_rng(77)
    rng_b = np.random.default_rng(77)
    r1 = tvp_var_sv_fit(Y, p=1, n_draws=5, burn=5, rng=rng_a)
    r2 = tvp_var_sv_fit(Y, p=1, n_draws=5, burn=5, rng=rng_b)
    assert np.allclose(r1.beta_draws, r2.beta_draws)
    assert np.allclose(r1.log_h_draws, r2.log_h_draws)


def test_tvp_var_sv_fit_rng_none_runs():
    Y = _make_var_data(T=50, n=2)
    result = tvp_var_sv_fit(Y, p=1, n_draws=3, burn=3, rng=None)
    assert result.n_draws == 3


def test_tvp_var_sv_fit_Q_symmetric_pd():
    """Q draws from the SV model must also be symmetric and PD."""
    Y = _make_var_data(T=70, n=2)
    result = tvp_var_sv_fit(Y, p=1, n_draws=10, burn=10, rng=np.random.default_rng(5))
    for d in range(result.n_draws):
        Q = result.Q_draws[d]
        assert np.allclose(Q, Q.T, atol=1e-12)
        eigs = np.linalg.eigvalsh(Q)
        assert np.all(eigs > 0), f"Q_sv[{d}] not PD; min eig = {eigs.min()}"


def test_tvp_var_sv_fit_short_series():
    Y = _make_var_data(T=20, n=2)
    result = tvp_var_sv_fit(Y, p=1, n_draws=5, burn=5, rng=np.random.default_rng(6))
    assert result.T_eff == 19


@pytest.mark.slow
def test_tvp_var_fit_sigma_prior_monotonicity():
    """Larger Sigma_prior_scale -> larger Sigma draws (expected posterior mean increases).

    Marked slow because it uses more draws (100) for reliable comparison.
    """
    Y = _make_var_data(T=100, n=2, seed=22)
    r_small = tvp_var_fit(Y, p=1, n_draws=100, burn=50,
                           Sigma_prior_scale=0.01, rng=np.random.default_rng(9))
    r_large = tvp_var_fit(Y, p=1, n_draws=100, burn=50,
                           Sigma_prior_scale=5.0, rng=np.random.default_rng(9))
    trace_small = np.mean([np.trace(r_small.Sigma_draws[d]) for d in range(r_small.n_draws)])
    trace_large = np.mean([np.trace(r_large.Sigma_draws[d]) for d in range(r_large.n_draws)])
    assert trace_large > trace_small, (
        f"Larger Sigma_prior should give larger Sigma draws; got small={trace_small:.4f}, "
        f"large={trace_large:.4f}"
    )


@pytest.mark.slow
def test_tvp_var_sv_fit_three_variable():
    """n=3, p=1: checks that the Gibbs completes with the higher-dimensional A step."""
    Y = _make_var_data(T=80, n=3, p=1, seed=5)
    result = tvp_var_sv_fit(Y, p=1, n_draws=20, burn=20, rng=np.random.default_rng(5))
    assert result.n == 3
    assert result.beta_draws.shape[2] == 3 * (1 + 3 * 1)  # 3 * 4 = 12
    assert np.all(np.isfinite(result.beta_draws))
    assert np.all(np.isfinite(result.log_h_draws))


# ===========================================================================
# Branches: else beta0=zeros when T is too short for pre-sample OLS
# n_pre = min(40, T_eff//4 + 1); must be < k_x+1 to hit else.
# For T=10, n=2, p=1: T_eff=9, n_pre=3, k_x=3, k_x+1=4 → else branch taken.
# ===========================================================================

def test_tvp_var_fit_very_short_T_else_branch():
    """T so small that n_pre < k_x+1 forces the zero-initialisation else branch."""
    T, n, p = 10, 2, 1
    Y = np.random.default_rng(0).standard_normal((T, n)) * 0.5
    result = tvp_var_fit(Y, p=p, n_draws=3, burn=3, rng=np.random.default_rng(0))
    # n_pre = min(40, 9//4+1) = 3 < k_x+1=4 → zero beta0 path
    assert result.T_eff == T - p
    assert np.all(np.isfinite(result.beta_draws))


def test_tvp_var_sv_fit_very_short_T_else_branch():
    """Same short-T else branch in tvp_var_sv_fit."""
    T, n, p = 10, 2, 1
    Y = np.random.default_rng(0).standard_normal((T, n)) * 0.5
    result = tvp_var_sv_fit(Y, p=p, n_draws=3, burn=3, rng=np.random.default_rng(0))
    assert result.T_eff == T - p
    assert np.all(np.isfinite(result.beta_draws))
