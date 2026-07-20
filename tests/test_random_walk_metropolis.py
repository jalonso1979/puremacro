"""Tests for puremacro.mcmc.random_walk_metropolis."""
import numpy as np
import pytest
from scipy import stats


def _bivariate_normal_log_density(mean, cov):
    """Return a closure log_dens(x) -> log N(x | mean, cov)."""
    inv = np.linalg.inv(cov)
    log_det = np.log(np.linalg.det(cov))
    k = len(mean)
    norm_const = -0.5 * (k * np.log(2 * np.pi) + log_det)
    def log_dens(x):
        diff = x - mean
        return float(norm_const - 0.5 * diff @ inv @ diff)
    return log_dens


def test_metropolis_recovers_2d_normal():
    """Target: bivariate N([1, -2], diag([2, 0.5])). 20K draws after burn-in
    should give empirical mean within 0.1 and empirical cov within 15%."""
    from puremacro.mcmc import random_walk_metropolis

    true_mean = np.array([1.0, -2.0])
    true_cov = np.array([[2.0, 0.0], [0.0, 0.5]])
    log_dens = _bivariate_normal_log_density(true_mean, true_cov)

    init = np.zeros(2)
    proposal_cov = np.eye(2)

    result = random_walk_metropolis(
        log_dens, init, proposal_cov, n_draws=20_000,
        seed=42, accept_target=0.30, adapt_burnin=2_000,
    )
    chain = result["chain"]
    assert chain.shape == (20_000, 2)
    emp_mean = chain.mean(axis=0)
    emp_cov = np.cov(chain.T)
    np.testing.assert_allclose(emp_mean, true_mean, atol=0.15)
    np.testing.assert_allclose(emp_cov, true_cov, rtol=0.20, atol=0.05)


def test_metropolis_accept_target_adaptation():
    """After adapt_burnin, accept_rate is within ±15pp of accept_target=0.25."""
    from puremacro.mcmc import random_walk_metropolis

    log_dens = _bivariate_normal_log_density(np.zeros(2), np.eye(2))
    init = np.zeros(2)
    result = random_walk_metropolis(
        log_dens, init, np.eye(2) * 100, n_draws=5_000,
        seed=0, accept_target=0.25, adapt_burnin=2_000,
    )
    assert 0.10 <= result["accept_rate"] <= 0.40


def test_metropolis_handles_minus_inf():
    """log_dens returns -inf outside the unit ball; chain stays inside."""
    from puremacro.mcmc import random_walk_metropolis

    def log_dens(x):
        if np.dot(x, x) > 1.0:
            return -np.inf
        return 0.0

    init = np.array([0.1, 0.1])
    result = random_walk_metropolis(
        log_dens, init, np.eye(2) * 0.1, n_draws=2_000, seed=7,
    )
    norms_sq = (result["chain"] ** 2).sum(axis=1)
    assert (norms_sq <= 1.0 + 1e-9).all()


def test_metropolis_returns_documented_keys():
    """Result dict has chain, log_post, accept_rate, final_scale."""
    from puremacro.mcmc import random_walk_metropolis
    log_dens = lambda x: -0.5 * float(x @ x)
    result = random_walk_metropolis(
        log_dens, np.zeros(3), np.eye(3), n_draws=500, seed=1,
    )
    assert set(result.keys()) >= {"chain", "log_post", "accept_rate", "final_scale"}
    assert result["chain"].shape == (500, 3)
    assert result["log_post"].shape == (500,)
    assert 0.0 <= result["accept_rate"] <= 1.0
    assert result["final_scale"] > 0


def test_metropolis_seed_reproducibility():
    """Same seed → same chain."""
    from puremacro.mcmc import random_walk_metropolis
    log_dens = lambda x: -0.5 * float(x @ x)
    init = np.zeros(2)
    cov = np.eye(2)
    r1 = random_walk_metropolis(log_dens, init, cov, n_draws=200, seed=11)
    r2 = random_walk_metropolis(log_dens, init, cov, n_draws=200, seed=11)
    np.testing.assert_array_equal(r1["chain"], r2["chain"])
