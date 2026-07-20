"""BVAR with Minnesota prior — posterior mean via dummy observations.

Implementation follows Banbura-Giannone-Reichlin (2010): the Minnesota
prior on VAR coefficients is equivalent to OLS on an augmented dataset
that stacks the original observations with prior dummy observations.

Prior structure (Litterman 1986, with Doan-Litterman-Sims 1984 lag decay):
- Own first lag: prior mean 1, prior std λ₁ * σ_i / σ_i = λ₁.
- Own k-th lag (k > 1): prior mean 0, prior std λ₁ / k^λ₃.
- Cross-variable lag: prior mean 0, prior std λ₂ * λ₁ * σ_i / (k^λ₃ * σ_j).
- Intercept: diffuse prior (very large variance).
- σ_i estimated from a univariate AR(p) on each y_i.

Full posterior sampling (Gibbs) is deferred to v2. This function returns
the posterior MEAN coefficients only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._linalg import inv_xtx


def _univariate_sigma(y: np.ndarray, p: int) -> float:
    """RMSE from a univariate AR(p) on y (used to scale the prior)."""
    T = len(y)
    if T <= p + 1:
        return float(np.std(y, ddof=0)) or 1.0
    Y_lag = np.column_stack([y[p - k - 1: T - k - 1] for k in range(p)])
    X = np.column_stack([np.ones(len(Y_lag)), Y_lag])
    y_dep = y[p:]
    beta, *_ = np.linalg.lstsq(X, y_dep, rcond=None)
    resid = y_dep - X @ beta
    return float(np.std(resid, ddof=0)) or 1.0


def minnesota_posterior(
    Y: pd.DataFrame,
    p: int,
    lambda1: float = 0.2,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    intercept_prior_std: float = 1e3,
) -> dict:
    """Compute the posterior mean of a Minnesota-prior BVAR via dummy
    observations.

    Parameters
    ----------
    Y : pd.DataFrame
        T × n panel of endogenous variables.
    p : int
        Number of lags.
    lambda1 : float
        Overall prior tightness. Smaller = tighter (more shrinkage to RW).
    lambda2 : float
        Cross-variable shrinkage. Smaller = more shrinkage to AR(1).
    lambda3 : float
        Lag-decay exponent. Larger = faster decay of prior std with k.
    intercept_prior_std : float
        Diffuse prior std on the intercept (large = nearly uninformative).

    Returns
    -------
    dict with keys
        A_list : list of (n, n) — posterior mean coefficient matrices,
                 A_list[k][i,j] is coefficient of y_j at lag (k+1) in eq i.
        intercept : (n,) — posterior mean intercept
        Sigma : (n, n) — residual covariance from posterior mean
        residuals : (T-p, n) — residuals at posterior mean
        lambda1, lambda2, lambda3 : floats — hyperparameters used
    """
    Y_arr = np.asarray(Y, dtype=float)
    T, n = Y_arr.shape

    # Estimate per-variable sigma_i from univariate AR(p)
    sigmas = np.array([_univariate_sigma(Y_arr[:, i], p) for i in range(n)])

    # Build the OLS design matrix: X has columns [const, y_{t-1}, ..., y_{t-p}]
    # X is (T-p, 1 + n*p), Y_dep is (T-p, n)
    Y_dep = Y_arr[p:]
    X_rows = []
    for t in range(p, T):
        row = [1.0]
        for k in range(1, p + 1):
            row.extend(Y_arr[t - k])
        X_rows.append(row)
    X = np.array(X_rows)  # (T-p, 1 + n*p)

    k_reg = 1 + n * p  # number of regressors per equation

    # Build dummy observations per equation (Banbura-Giannone-Reichlin 2010).
    # For each equation eq_idx, we solve a separate OLS:
    #   y_eq_aug = X_aug @ beta_eq
    # where y_eq_aug = [Y_dep[:, eq_idx]; dummy_y] and X_aug = [X; dummy_X].
    #
    # Dummy construction for equation i (= eq_idx):
    #   For lag k (1..p), variable j (0..n-1):
    #     dummy_y scalar = σ_j if (k == 1 and j == i) else 0
    #     dummy_x row:  the (1 + (k-1)*n + j)-th element = σ_j * k^λ₃ / scale
    #       where scale = λ₁          if j == i (own-lag)
    #                     λ₁ * λ₂     if j != i (cross-lag)
    #
    # This encodes:
    #   prior mean β_{ij,k} = (dummy_y) / (dummy_x[1+(k-1)*n+j])
    #                        = 1 if j==i and k==1, else 0  ✓
    #   prior std  ∝ 1 / dummy_x[1+(k-1)*n+j]
    #              = λ₁ / (σ_j * k^λ₃)          for own lag
    #              = λ₁ * λ₂ / (σ_j * k^λ₃)     for cross lag
    #
    # Sigma dummies (1 row per equation, targeting residual variance σ_i²):
    #   dummy_y = σ_i, dummy_x = 0 vector → adds a prior residual observation.
    #
    # Intercept dummy (1 row shared, diffuse):
    #   dummy_y = 0, dummy_x[0] = 1 / intercept_prior_std  → diffuse prior.

    # Intercept dummy (same for all equations)
    Xd_intercept = np.zeros((1, k_reg))
    Xd_intercept[0, 0] = 1.0 / intercept_prior_std

    A_list = [np.zeros((n, n)) for _ in range(p)]
    intercept = np.zeros(n)

    for eq_idx in range(n):
        # Lag dummies — typed as Any to allow list→ndarray reassignment below.
        Yd_eq: Any = []
        Xd_eq: Any = []
        for k in range(1, p + 1):
            for j in range(n):
                # Prior precision scale
                scale = lambda1 if j == eq_idx else lambda1 * lambda2
                xd_coeff = sigmas[j] * (k ** lambda3) / scale
                xd_row = np.zeros(k_reg)
                xd_row[1 + (k - 1) * n + j] = xd_coeff
                # Prior mean: 1 on own first lag, 0 elsewhere.
                # Dummy encodes prior_mean = yd_scalar / xd_coeff, so
                # yd_scalar = prior_mean * xd_coeff.
                prior_mean = 1.0 if (k == 1 and j == eq_idx) else 0.0
                yd_scalar = prior_mean * xd_coeff
                Yd_eq.append(yd_scalar)
                Xd_eq.append(xd_row)
        Yd_eq = np.array(Yd_eq)       # (n*p,)
        Xd_eq = np.array(Xd_eq)       # (n*p, k_reg)

        # Sigma dummy for this equation
        Yd_sigma_eq = np.array([sigmas[eq_idx]])   # (1,)
        Xd_sigma_eq = np.zeros((1, k_reg))          # all zeros → targets residual

        # Stack augmented system
        y_eq = np.concatenate([Y_dep[:, eq_idx], Yd_eq, Yd_sigma_eq, np.array([0.0])])
        X_eq = np.vstack([X, Xd_eq, Xd_sigma_eq, Xd_intercept])

        # OLS on augmented system: β = (X'X)^{-1} X'y
        beta_eq, *_ = np.linalg.lstsq(X_eq, y_eq, rcond=None)
        intercept[eq_idx] = beta_eq[0]
        for k in range(p):
            # beta_eq[1 + k*n : 1 + (k+1)*n] are coefficients on y_{t-k-1}
            A_list[k][eq_idx, :] = beta_eq[1 + k * n: 1 + (k + 1) * n]

    # Compute residuals and Sigma at the posterior mean
    Y_hat = np.zeros_like(Y_dep)
    for t in range(len(Y_dep)):
        Y_hat[t] = intercept + sum(
            A_list[k] @ Y_arr[p + t - k - 1] for k in range(p)
        )
    residuals = Y_dep - Y_hat
    Sigma = (residuals.T @ residuals) / max(1, len(residuals) - 1)

    return {
        "A_list": A_list,
        "intercept": intercept,
        "Sigma": Sigma,
        "residuals": residuals,
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "lambda3": float(lambda3),
    }


def _build_minnesota_dummies(
    Y_arr: np.ndarray,
    p: int,
    sigmas: np.ndarray,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    intercept_prior_std: float,
):
    """Construct the (Y_d, X_d) Banbura-Giannone-Reichlin dummy block.

    The augmented system
        [Y; Y_d] = [X; X_d] @ B + [u; u_d]
    has Normal-Inverse-Wishart conjugate posterior. Returns the augmented
    OLS posterior moments needed by both ``minnesota_posterior`` and
    ``minnesota_gibbs``.
    """
    T, n = Y_arr.shape
    k_reg = 1 + n * p
    Y_dep = Y_arr[p:]
    X_rows = []
    for t in range(p, T):
        row = [1.0]
        for k in range(1, p + 1):
            row.extend(Y_arr[t - k])
        X_rows.append(row)
    X = np.array(X_rows)

    # Build dummy block: prior on coefficients + sigma + intercept.
    Yd_rows: list[Any] = []
    Xd_rows: list[Any] = []
    for k in range(1, p + 1):
        for j in range(n):
            xd_row = np.zeros(k_reg)
            for eq in range(n):
                if eq != j:
                    continue
                # Per-equation own-vs-cross scale (we encode the diagonal
                # of the prior precision; off-diagonal uses lambda1*lambda2).
                pass
    # We build dummies in a *stacked* form so the same X_d / Y_d feed
    # all n equations simultaneously (BGR equation-by-equation: extend
    # X_d with one block per coefficient slot).
    Yd = np.zeros((n * p * n + n + 1, n))
    Xd = np.zeros((n * p * n + n + 1, k_reg))
    row_idx: int = 0
    for k in range(1, p + 1):
        for j in range(n):
            xd_coeff = sigmas[j] * (k ** lambda3) / lambda1
            cross_coeff = sigmas[j] * (k ** lambda3) / (lambda1 * lambda2)
            for eq in range(n):
                # Coefficient of y_j at lag k in equation eq:
                # diagonal slot in the regressor row; prior mean = 1 if k==1 and j==eq else 0.
                if eq == j:
                    Xd[row_idx, 1 + (k - 1) * n + j] = xd_coeff
                    Yd[row_idx, eq] = (1.0 if k == 1 else 0.0) * xd_coeff
                else:
                    Xd[row_idx, 1 + (k - 1) * n + j] = cross_coeff
                    Yd[row_idx, eq] = 0.0
            row_idx += 1
    # Sigma dummies: one per equation.
    for eq in range(n):
        Yd[row_idx, eq] = sigmas[eq]
        row_idx += 1
    # Intercept dummy.
    Xd[row_idx, 0] = 1.0 / intercept_prior_std
    row_idx += 1

    return Y_dep, X, Yd[:row_idx], Xd[:row_idx]


def minnesota_gibbs(
    Y: pd.DataFrame,
    p: int,
    *,
    n_draws: int = 1000,
    burn: int = 500,
    lambda1: float = 0.2,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    intercept_prior_std: float = 1e3,
    rng: np.random.Generator | None = None,
) -> dict:
    """Posterior draws from a Minnesota-prior BVAR via the Normal-Inverse-
    Wishart conjugate posterior (Banbura-Giannone-Reichlin 2010).

    The Minnesota prior is encoded as dummy observations; the augmented
    OLS system has a closed-form NIW posterior:

        Sigma | Y ~ IW(S_post, nu_post)
        vec(B) | Sigma, Y ~ N(vec(B_post), Sigma kron (X*'X*)^{-1})

    where (Y*, X*) stack data and dummies.

    Parameters
    ----------
    n_draws : int — number of post-burn-in draws.
    burn    : int — burn-in (kept for interface symmetry; the conjugate
              posterior does not require it but burning improves
              numerical stability across cold starts).
    rng     : numpy Generator.

    Returns
    -------
    dict with
        A_draws       : (n_draws, p, n, n)
        Sigma_draws   : (n_draws, n, n)
        intercept_draws : (n_draws, n)
        nu_post       : posterior degrees of freedom
    """
    if rng is None:
        rng = np.random.default_rng()
    Y_arr = np.asarray(Y, dtype=float)
    T, n = Y_arr.shape
    sigmas = np.array([_univariate_sigma(Y_arr[:, i], p) for i in range(n)])
    Y_dep, X, Yd, Xd = _build_minnesota_dummies(
        Y_arr, p, sigmas, lambda1, lambda2, lambda3, intercept_prior_std,
    )
    Y_aug = np.vstack([Y_dep, Yd])
    X_aug = np.vstack([X, Xd])
    XtX_inv = inv_xtx(X_aug, name="minnesota_bvar")
    B_post = XtX_inv @ X_aug.T @ Y_aug                       # (k, n)
    resid = Y_aug - X_aug @ B_post
    S_post = resid.T @ resid                                  # (n, n)
    nu_post = Y_aug.shape[0] - X_aug.shape[1]

    # Pre-Cholesky once; redraw each iter
    L_kron = np.linalg.cholesky(XtX_inv + 1e-12 * np.eye(XtX_inv.shape[0]))

    A_draws = np.empty((n_draws, p, n, n))
    Sigma_draws = np.empty((n_draws, n, n))
    intercept_draws = np.empty((n_draws, n))

    total = n_draws + burn
    keep_idx = 0
    for it in range(total):
        # Sigma | Y ~ IW(S_post, nu_post): sample via Bartlett.
        Sigma_draw = _inv_wishart(S_post, nu_post, rng)
        # vec(B) | Sigma ~ N(vec(B_post), Sigma kron XtX_inv)
        L_sig = np.linalg.cholesky(Sigma_draw + 1e-12 * np.eye(n))
        Z = rng.standard_normal(B_post.shape)               # (k, n)
        B_draw = B_post + L_kron @ Z @ L_sig.T

        if it < burn:
            continue
        intercept_draws[keep_idx] = B_draw[0]
        for k in range(p):
            A_draws[keep_idx, k] = B_draw[1 + k * n: 1 + (k + 1) * n].T
        Sigma_draws[keep_idx] = Sigma_draw
        keep_idx += 1

    return {
        "A_draws": A_draws,
        "Sigma_draws": Sigma_draws,
        "intercept_draws": intercept_draws,
        "nu_post": int(nu_post),
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "lambda3": float(lambda3),
    }


def _inv_wishart(S: np.ndarray, df: int, rng: np.random.Generator) -> np.ndarray:
    """Draw Sigma ~ IW(S, df) via Bartlett decomposition of W = inv(Sigma) ~ W(inv(S), df)."""
    n = S.shape[0]
    S_inv = np.linalg.inv(S + 1e-12 * np.eye(n))
    L = np.linalg.cholesky(S_inv)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = np.sqrt(rng.chisquare(df - i))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    W = L @ A @ A.T @ L.T
    return np.linalg.inv(W)


def _minnesota_log_marginal_likelihood(
    Y_arr: np.ndarray,
    p: int,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    intercept_prior_std: float,
) -> float:
    """Compute the log marginal likelihood of a Minnesota-prior BVAR
    via the dummy-observation representation (Banbura-Giannone-Reichlin
    2010 / Giannone-Lenza-Primiceri 2015 eq. 3).

    For the conjugate Normal-Inverse-Wishart prior implied by the
    Minnesota dummy observations, the marginal likelihood is

        log p(Y | λ) = const + (T_d - k)/2 * log|S_post| - (T - k)/2 * log|S_aug|
                       + log Γ_n((T_aug - k)/2) - log Γ_n((T_d - k)/2)
                       + ...

    Up to an additive constant we collect the data-dependent terms; the
    relative argmax over hyperparameters is what we need.
    """
    T, n = Y_arr.shape
    sigmas = np.array([_univariate_sigma(Y_arr[:, i], p) for i in range(n)])
    Y_dep, X, Yd, Xd = _build_minnesota_dummies(
        Y_arr, p, sigmas, lambda1, lambda2, lambda3, intercept_prior_std,
    )
    # Augmented data (data + dummies) — the prior alone is just dummies.
    Y_aug = np.vstack([Y_dep, Yd])
    X_aug = np.vstack([X, Xd])
    k = X.shape[1]
    T_eff = X.shape[0]
    T_aug = X_aug.shape[0]
    # OLS on augmented system.
    XtX_dummy = Xd.T @ Xd
    try:
        XtX_aug_inv = inv_xtx(X_aug, name="bvar_marginal_likelihood")
        XtX_dummy_inv = np.linalg.inv(XtX_dummy + 1e-8 * np.eye(k))
    except np.linalg.LinAlgError:
        # Marginal-likelihood loop must keep iterating on bad
        # hyperparameters; return -inf so the optimiser rejects them.
        return -np.inf
    XtX_aug = X_aug.T @ X_aug
    B_aug = XtX_aug_inv @ X_aug.T @ Y_aug
    resid_aug = Y_aug - X_aug @ B_aug
    S_post = resid_aug.T @ resid_aug
    # Prior moment: regress dummy "y" on dummy "X" alone.
    B_pri = XtX_dummy_inv @ Xd.T @ Yd
    resid_pri = Yd - Xd @ B_pri
    S_pri = resid_pri.T @ resid_pri
    sign_post, logdet_post = np.linalg.slogdet(S_post + 1e-12 * np.eye(n))
    sign_pri, logdet_pri = np.linalg.slogdet(S_pri + 1e-12 * np.eye(n))
    sign_xa, logdet_xa = np.linalg.slogdet(XtX_aug)
    sign_xd, logdet_xd = np.linalg.slogdet(XtX_dummy + 1e-8 * np.eye(k))
    if sign_post <= 0 or sign_pri <= 0 or sign_xa <= 0 or sign_xd <= 0:
        return -np.inf
    # Up to an additive constant
    nu_post = T_aug - k
    nu_pri = Xd.shape[0] - k
    from scipy.special import gammaln
    log_mvgamma = lambda v, p_: (p_ * (p_ - 1) / 4) * np.log(np.pi) + sum(
        gammaln(v / 2 + (1 - i) / 2) for i in range(1, p_ + 1)
    )
    val = (
        log_mvgamma(nu_post, n) - log_mvgamma(nu_pri, n)
        - 0.5 * n * (logdet_xa - logdet_xd)
        - 0.5 * nu_post * logdet_post + 0.5 * nu_pri * logdet_pri
        - 0.5 * n * T_eff * np.log(np.pi)
    )
    return float(val)


def minnesota_optimal_lambda(
    Y: pd.DataFrame,
    p: int,
    *,
    intercept_prior_std: float = 1e3,
    lambda1_grid=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0),
    lambda2_grid=(0.5, 1.0),
    lambda3_grid=(1.0, 2.0),
) -> dict:
    """Empirical-Bayes choice of Minnesota hyperparameters via marginal
    likelihood maximisation (Giannone-Lenza-Primiceri 2015).

    Grids over ``λ_1, λ_2, λ_3`` and returns the triple maximising the
    log marginal likelihood. This is the cheap "good default" version
    of GLP 2015; their full hierarchical procedure also optimises over
    a Sum-of-Coefficients prior tightness, which we omit for brevity.
    """
    Y_arr = np.asarray(Y, dtype=float)
    best: tuple[float, Any] = (-np.inf, None)
    grid = []
    for l1 in lambda1_grid:
        for l2 in lambda2_grid:
            for l3 in lambda3_grid:
                ml = _minnesota_log_marginal_likelihood(
                    Y_arr, p, l1, l2, l3, intercept_prior_std
                )
                grid.append({"lambda1": l1, "lambda2": l2, "lambda3": l3,
                              "log_ml": ml})
                if ml > best[0]:
                    best = (ml, (l1, l2, l3))
    if best[1] is None:
        raise RuntimeError("Marginal-likelihood evaluation failed for all "
                           "grid points — try different priors.")
    l1, l2, l3 = best[1]
    return {
        "lambda1": float(l1),
        "lambda2": float(l2),
        "lambda3": float(l3),
        "log_ml": float(best[0]),
        "grid": grid,
    }


__all__ = [
    "minnesota_posterior", "minnesota_gibbs", "minnesota_optimal_lambda",
]
