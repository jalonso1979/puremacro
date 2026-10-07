"""Regression tests for the Minnesota NIW dummy-observation fix (review key MINN).

Before the fix, ``_build_minnesota_dummies`` wrote the same cell of the
(lag k, variable j) dummy row once per equation, so the last equation won:
every variable except the last got the cross-lag precision on its own lag.
The implied prior mean of the own first lag was ``lambda2`` (0.5 by default)
instead of 1 for all but the last variable, and the prior depended on the
column order. The Normal-inverse-Wishart (Kronecker) prior cannot encode a
cross-variable factor at all: Banbura, Giannone and Reichlin (2010) impose it
"under the condition that theta = 1" (ECB WP 966, p. 11). These tests pin

* the BGR eq. (5) dummy block, built here by hand;
* a random-walk prior mean for every variable, whatever the ordering;
* the tight-prior limit of ``minnesota_gibbs`` (A_1 -> I);
* equality of the NIW posterior mean and the equation-by-equation
  ``minnesota_posterior`` at lambda2 = 1;
* the lambda2 = 1 defaults and the warning for any other value on the NIW
  paths, while ``minnesota_posterior`` keeps honouring lambda2;
* a validation case that fails on the pre-fix builder.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.var import bvar as bvar_mod
from puremacro.var.bvar import (
    _build_minnesota_dummies,
    _minnesota_log_marginal_likelihood,
    _univariate_sigma,
    minnesota_gibbs,
    minnesota_optimal_lambda,
    minnesota_posterior,
)


def _var1_panel(T: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    A = np.array([[0.6, 0.10, 0.0], [0.05, 0.5, 0.10], [0.0, 0.05, 0.55]])
    scale = np.array([0.5, 1.5, 0.2])  # unequal scales make sigma_j matter
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + scale * rng.standard_normal(3)
    return pd.DataFrame(Y, columns=list("ABC"))


def _bgr_eq5_block(sigmas, p, lambda1, lambda3, ips):
    """BGR (2010) eq. (5) written out by hand, with J_p = diag(1..p)
    generalised to diag(k**lambda3), plus the sigma and intercept rows."""
    n = len(sigmas)
    k_reg = 1 + n * p
    Yd = np.zeros((n * p + n + 1, n))
    Xd = np.zeros((n * p + n + 1, k_reg))
    Yd[:n, :] = np.diag(sigmas) / lambda1
    Jp = np.diag(np.arange(1, p + 1, dtype=float) ** lambda3)
    Xd[: n * p, 1:] = np.kron(Jp, np.diag(sigmas)) / lambda1
    Yd[n * p: n * p + n, :] = np.diag(sigmas)
    Xd[-1, 0] = 1.0 / ips
    return Yd, Xd


def _niw_prior_mean(Yd, Xd):
    """B0 = (Xd'Xd)^-1 Xd'Yd from the dummies alone (no data)."""
    return np.linalg.lstsq(Xd, Yd, rcond=None)[0]


# ---------------------------------------------------------------------------
# The dummy block itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,lambda3", [(1, 1.0), (2, 1.0), (3, 2.0)])
def test_dummy_block_equals_bgr_eq5(p, lambda3):
    Y = np.random.default_rng(1).standard_normal((60, 3))
    sigmas = np.array([1.0, 2.0, 3.0])
    _, _, Yd, Xd = _build_minnesota_dummies(Y, p, sigmas, 0.2, 1.0, lambda3, 1e3)
    Yd_ref, Xd_ref = _bgr_eq5_block(sigmas, p, 0.2, lambda3, 1e3)
    np.testing.assert_allclose(Yd, Yd_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Xd, Xd_ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("lambda2", [1.0, 0.5, 0.1])
def test_dummy_prior_mean_is_a_random_walk_for_every_variable(lambda2):
    """Pre-fix: diag(0.5, 0.5, 1.0) at lambda2 = 0.5 (only the last variable
    was centred on a random walk)."""
    Y = np.random.default_rng(0).standard_normal((50, 3))
    sigmas = np.array([1.0, 2.0, 3.0])
    _, _, Yd, Xd = _build_minnesota_dummies(Y, 2, sigmas, 0.2, lambda2, 1.0, 1e3)
    B0 = _niw_prior_mean(Yd, Xd)
    np.testing.assert_allclose(B0[0], 0.0, atol=1e-10)             # intercept
    np.testing.assert_allclose(B0[1:4].T, np.eye(3), atol=1e-10)   # A_1 = I
    np.testing.assert_allclose(B0[4:7].T, 0.0, atol=1e-10)         # A_2 = 0


def test_dummy_own_lag_prior_scale_is_lambda1_for_every_variable():
    """Under NIW, Var(B[r, i]) = Psi_ii * [(Xd'Xd)^-1]_rr; with the sigma
    dummies E[Psi_ii] is proportional to sigma_i^2, so the own-lag prior sd is
    lambda1 / k^lambda3 for every variable (pre-fix: lambda1*lambda2 for all but
    the last one)."""
    sigmas = np.array([1.0, 2.0, 3.0])
    lambda1, lambda3, p = 0.2, 1.0, 2
    Y = np.zeros((20, 3))
    _, _, _, Xd = _build_minnesota_dummies(Y, p, sigmas, lambda1, 0.5, lambda3, 1e3)
    Om = np.linalg.inv(Xd.T @ Xd)
    for k in range(1, p + 1):
        for i in range(3):
            r = 1 + (k - 1) * 3 + i
            sd_own = sigmas[i] * np.sqrt(Om[r, r])
            assert sd_own == pytest.approx(lambda1 / k ** lambda3, rel=1e-12)


# ---------------------------------------------------------------------------
# minnesota_gibbs
# ---------------------------------------------------------------------------

def test_gibbs_tight_prior_limit_is_random_walk_for_every_variable():
    """lambda1 -> 0 must give A_1 -> I. Pre-fix (lambda2 = 0.5 requested):
    diag(0.5, 0.5, 1.0)."""
    df = _var1_panel()
    with pytest.warns(UserWarning, match="lambda2"):
        g = minnesota_gibbs(df, 1, n_draws=300, burn=50, lambda1=1e-4, lambda2=0.5,
                            rng=np.random.default_rng(1))
    np.testing.assert_allclose(g["A_draws"][:, 0].mean(0), np.eye(3), atol=2e-3)
    np.testing.assert_allclose(g["A_mean"][0], np.eye(3), atol=1e-4)


def test_gibbs_posterior_mean_equals_equation_by_equation_at_lambda2_one():
    """At theta = lambda2 = 1 the NIW posterior mean (matrix-t mean) is the
    augmented OLS, which minnesota_posterior computes equation by equation."""
    df = _var1_panel()
    for lambda1 in (0.05, 0.2, 1.0):
        g = minnesota_gibbs(df, 2, n_draws=10, burn=0, lambda1=lambda1,
                            rng=np.random.default_rng(0))
        mp = minnesota_posterior(df, 2, lambda1=lambda1, lambda2=1.0)
        for k in range(2):
            np.testing.assert_allclose(g["A_mean"][k], mp["A_list"][k], atol=1e-10)
        np.testing.assert_allclose(g["intercept_mean"], mp["intercept"], atol=1e-10)


def test_gibbs_draws_are_centred_on_the_posterior_mean():
    df = _var1_panel()
    g = minnesota_gibbs(df, 1, n_draws=6000, burn=0, rng=np.random.default_rng(3))
    mc_se = g["A_draws"][:, 0].std(0) / np.sqrt(6000)
    assert np.all(np.abs(g["A_draws"][:, 0].mean(0) - g["A_mean"][0]) < 5 * mc_se + 1e-12)


def test_gibbs_is_invariant_to_variable_ordering():
    """Pre-fix the variable listed last was the only one centred on 1."""
    df = _var1_panel()
    perm = [2, 0, 1]
    g = minnesota_gibbs(df, 1, n_draws=5, burn=0, lambda1=0.05,
                        rng=np.random.default_rng(0))
    gp = minnesota_gibbs(df.iloc[:, perm], 1, n_draws=5, burn=0, lambda1=0.05,
                         rng=np.random.default_rng(0))
    A = g["A_mean"][0]
    Ap = gp["A_mean"][0]
    np.testing.assert_allclose(Ap, A[np.ix_(perm, perm)], atol=1e-10)


def test_niw_paths_default_lambda2_is_one():
    assert inspect.signature(minnesota_gibbs).parameters["lambda2"].default == 1.0
    assert inspect.signature(_minnesota_log_marginal_likelihood).parameters["lambda2"].default == 1.0
    assert tuple(inspect.signature(minnesota_optimal_lambda).parameters["lambda2_grid"].default) == (1.0,)
    # the equation-by-equation posterior keeps its per-equation lambda2
    assert inspect.signature(minnesota_posterior).parameters["lambda2"].default == 0.5


def test_gibbs_default_does_not_warn_and_reports_lambda2_one():
    df = _var1_panel(T=80)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = minnesota_gibbs(df, 1, n_draws=5, burn=0, rng=np.random.default_rng(0))
    assert g["lambda2"] == 1.0


def test_gibbs_warns_and_uses_theta_one_when_lambda2_is_not_one():
    df = _var1_panel(T=80)
    with pytest.warns(UserWarning, match=r"lambda2=0\.3"):
        g = minnesota_gibbs(df, 1, n_draws=5, burn=0, lambda2=0.3,
                            rng=np.random.default_rng(0))
    g1 = minnesota_gibbs(df, 1, n_draws=5, burn=0, rng=np.random.default_rng(0))
    assert g["lambda2"] == 1.0
    np.testing.assert_array_equal(g["A_draws"], g1["A_draws"])


# ---------------------------------------------------------------------------
# Marginal likelihood and the GLP grid
# ---------------------------------------------------------------------------

def test_log_ml_is_invariant_to_variable_ordering():
    Y = _var1_panel().to_numpy()
    perm = [2, 0, 1]
    for l1 in (0.1, 0.3):
        a = _minnesota_log_marginal_likelihood(Y, 2, l1)
        b = _minnesota_log_marginal_likelihood(Y[:, perm], 2, l1)
        assert a == pytest.approx(b, rel=1e-10, abs=1e-8)


def test_log_ml_warns_for_lambda2_not_one_and_equals_theta_one_value():
    Y = _var1_panel().to_numpy()
    ref = _minnesota_log_marginal_likelihood(Y, 1, 0.2, 1.0, 1.0, 1e3)
    with pytest.warns(UserWarning, match="lambda2"):
        val = _minnesota_log_marginal_likelihood(Y, 1, 0.2, 0.5, 1.0, 1e3)
    assert val == ref


def test_optimal_lambda_default_grid_reports_lambda2_one():
    df = _var1_panel(T=120)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = minnesota_optimal_lambda(df, p=1, lambda1_grid=(0.1, 0.3),
                                       lambda3_grid=(1.0,))
    assert out["lambda2"] == 1.0
    assert {row["lambda2"] for row in out["grid"]} == {1.0}


def test_optimal_lambda_drops_non_unit_lambda2_with_a_warning():
    df = _var1_panel(T=120)
    with pytest.warns(UserWarning, match="lambda2_grid"):
        out = minnesota_optimal_lambda(df, p=1, lambda1_grid=(0.1, 0.3),
                                       lambda2_grid=(0.5, 1.0), lambda3_grid=(1.0,))
    assert out["lambda2"] == 1.0
    assert len(out["grid"]) == 2
    assert {row["lambda2"] for row in out["grid"]} == {1.0}


# ---------------------------------------------------------------------------
# minnesota_posterior keeps honouring lambda2 (independent closed form)
# ---------------------------------------------------------------------------

def _theil_posterior_mean(Y, p, lambda1, lambda2, lambda3, ips):
    """Theil-Goldberger mixed estimator per equation from the Minnesota prior
    moments (Litterman 1986): prior mean 1 on the own first lag, 0 elsewhere;
    prior sd lambda1/k^lambda3 on own lags and lambda1*lambda2*sigma_i /
    (k^lambda3 sigma_j) on cross lags; intercept sd sigma_i * ips."""
    Y = np.asarray(Y, float)
    T, n = Y.shape
    s = np.array([_univariate_sigma(Y[:, i], p) for i in range(n)])
    X = np.column_stack([np.ones(T - p)] + [Y[p - k: T - k] for k in range(1, p + 1)])
    A = np.zeros((p, n, n))
    for i in range(n):
        b0 = np.zeros(1 + n * p)
        v = np.empty(1 + n * p)
        v[0] = (s[i] * ips) ** 2
        for k in range(1, p + 1):
            for j in range(n):
                r = 1 + (k - 1) * n + j
                f = 1.0 if j == i else lambda2
                v[r] = (lambda1 * f * s[i] / (k ** lambda3 * s[j])) ** 2
                b0[r] = 1.0 if (k == 1 and j == i) else 0.0
        P = X.T @ X / s[i] ** 2 + np.diag(1.0 / v)
        beta = np.linalg.solve(P, X.T @ Y[p:, i] / s[i] ** 2 + b0 / v)
        for k in range(p):
            A[k, i] = beta[1 + k * n: 1 + (k + 1) * n]
    return A


@pytest.mark.parametrize("lambda2", [0.5, 0.2])
def test_minnesota_posterior_honours_lambda2(lambda2):
    df = _var1_panel()
    mp = minnesota_posterior(df, 2, lambda1=0.2, lambda2=lambda2, lambda3=1.0)
    ref = _theil_posterior_mean(df, 2, 0.2, lambda2, 1.0, 1e3)
    for k in range(2):
        np.testing.assert_allclose(mp["A_list"][k], ref[k], atol=1e-9)


# ---------------------------------------------------------------------------
# The validation case can now see the bug
# ---------------------------------------------------------------------------

def _prefix_builder(Y_arr, p, sigmas, lambda1, lambda2, lambda3, intercept_prior_std):
    """Verbatim lag-block logic of the pre-fix builder (last equation wins)."""
    T, n = Y_arr.shape
    k_reg = 1 + n * p
    Y_dep = Y_arr[p:]
    X = np.column_stack([np.ones(T - p)] + [Y_arr[p - k: T - k] for k in range(1, p + 1)])
    Yd = np.zeros((n * p + n + 1, n))
    Xd = np.zeros((n * p + n + 1, k_reg))
    row = 0
    for k in range(1, p + 1):
        for j in range(n):
            own = sigmas[j] * (k ** lambda3) / lambda1
            cross = sigmas[j] * (k ** lambda3) / (lambda1 * lambda2)
            for eq in range(n):
                if eq == j:
                    Xd[row, 1 + (k - 1) * n + j] = own
                    Yd[row, eq] = (1.0 if k == 1 else 0.0) * own
                else:
                    Xd[row, 1 + (k - 1) * n + j] = cross
            row += 1
    for eq in range(n):
        Yd[row, eq] = sigmas[eq]
        row += 1
    Xd[row, 0] = 1.0 / intercept_prior_std
    return Y_dep, X, Yd, Xd


def _run_case(case_id):
    from puremacro.validation import run_all

    res = {r.id: r for r in run_all(subsystem="var")}
    return res[case_id]


def test_validation_case_passes_on_the_fixed_builder():
    r = _run_case("var.bvar_minnesota_analytical_posterior")
    assert r.passed, (r.max_margin, r.error)


def test_validation_case_fails_on_the_prefix_builder(monkeypatch):
    monkeypatch.setattr(bvar_mod, "_build_minnesota_dummies", _prefix_builder)
    r = _run_case("var.bvar_minnesota_analytical_posterior")
    assert not r.passed


# ---------------------------------------------------------------------------
# Posterior degrees of freedom of Sigma (BGR 2010 eq. 7)
# ---------------------------------------------------------------------------

def test_sigma_posterior_degrees_of_freedom_follow_bgr_eq7():
    """BGR (2010, ECB WP 966 p. 12) eq. (7): with the improper prior
    |Psi|^-(n+3)/2, Psi | Y ~ iW(S~, T_d + 2 + T - k). Up to 4.4.0 the sampler
    used T_d + T - k, which inflates E[Sigma | Y] = S~ / (nu - n - 1) by
    (nu - n + 1) / (nu - n - 1): 5.3% on this short sample.

    The oracle is the paper's formula evaluated on the eq. (5) dummies built
    by hand, and the inverse-Wishart mean S / (nu - n - 1).
    """
    df = _var1_panel(T=40, seed=5)
    p, lambda1, ips = 1, 0.2, 1e3
    Y = df.to_numpy()
    n = Y.shape[1]
    sigmas = np.array([_univariate_sigma(Y[:, i], p) for i in range(n)])
    Yd, Xd = _bgr_eq5_block(sigmas, p, lambda1, 1.0, ips)
    X = np.column_stack([np.ones(len(Y) - p), Y[:-1]])
    Ys = np.vstack([Y[p:], Yd])
    Xs = np.vstack([X, Xd])
    B = np.linalg.solve(Xs.T @ Xs, Xs.T @ Ys)
    S = (Ys - Xs @ B).T @ (Ys - Xs @ B)
    T_d, T, k = Yd.shape[0], len(Y) - p, Xd.shape[1]
    nu = T_d + 2 + T - k

    n_draws = 20000
    g = minnesota_gibbs(df, p, n_draws=n_draws, burn=0, lambda1=lambda1,
                        intercept_prior_std=ips, rng=np.random.default_rng(11))
    assert g["nu_post"] == nu
    mean_exact = S / (nu - n - 1)
    draws = g["Sigma_draws"]
    mc_se = draws.std(axis=0) / np.sqrt(n_draws)
    assert np.all(np.abs(draws.mean(axis=0) - mean_exact) < 5 * mc_se)
    # The old degrees of freedom are far outside the Monte Carlo band.
    mean_old = S / (nu - 2 - n - 1)
    assert np.all(np.abs(np.diag(mean_old - mean_exact)) > 20 * np.diag(mc_se))
