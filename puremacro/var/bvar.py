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

Two posteriors are offered, and they differ in what λ₂ can do:

- :func:`minnesota_posterior` solves one Theil-Goldberger mixed regression
  per equation, so the own/cross factor λ₂ is honoured exactly. It returns
  the posterior MEAN only.
- :func:`minnesota_gibbs`, :func:`minnesota_optimal_lambda` (and the private
  ``_minnesota_log_marginal_likelihood``) use the conjugate
  Normal-inverse-Wishart (NIW) prior ``vec(B) | Ψ ~ N(vec(B₀), Ψ ⊗ Ω₀)``.
  Its Kronecker structure gives every equation the same prior covariance up
  to the scale Ψ_ii, so a cross-variable factor cannot be represented:
  BGR impose the Minnesota prior on the NIW "under the condition that
  θ = 1" (ECB WP 966, 2008, p. 11), i.e. λ₂ = 1. These paths default to
  λ₂ = 1 and warn, then use 1, when another value is requested. Their
  dummy block is BGR eq. (5) with lag decay ``k^λ₃`` in place of ``k``.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .._linalg import inv_xtx, safe_cholesky


def _niw_lambda2(lambda2: float, caller: str, stacklevel: int = 3) -> float:
    """Return the only λ₂ the conjugate NIW prior can encode (1.0).

    Warns when the caller asked for anything else. BGR (2010) impose the
    Minnesota prior on the Normal-inverse-Wishart "under the condition that
    θ = 1" because ``Ψ ⊗ Ω₀`` has no own-vs-cross factor per equation.
    """
    if not np.isclose(float(lambda2), 1.0, rtol=0.0, atol=1e-12):
        warnings.warn(
            f"{caller}: lambda2={lambda2!r} cannot be represented by the "
            "conjugate Normal-inverse-Wishart Minnesota prior, whose Kronecker "
            "covariance has no own-vs-cross factor (Banbura, Giannone and "
            "Reichlin 2010 require theta = 1). Using lambda2=1.0. For a "
            "cross-variable shrinkage factor use minnesota_posterior, which "
            "solves each equation separately.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return 1.0


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
    p: int | None = None,
    lambda1: float = 0.2,
    lambda2: float = 0.5,
    lambda3: float = 1.0,
    intercept_prior_std: float = 1e3,
    *,
    lags: int | None = None,
) -> dict:
    """Compute the posterior mean of a Minnesota-prior BVAR via dummy
    observations.

    Parameters
    ----------
    Y : pd.DataFrame
        T × n panel of endogenous variables.
    p : int, optional
        Number of lags.
    lambda1 : float
        Overall prior tightness. Smaller = tighter (more shrinkage to RW).
    lambda2 : float
        Cross-variable shrinkage. Smaller = more shrinkage to AR(1).
        Honoured exactly: each equation is a separate Theil-Goldberger
        mixed regression, so the own and cross prior sds can differ. (The
        NIW paths, :func:`minnesota_gibbs` and :func:`minnesota_optimal_lambda`,
        can only use ``lambda2 = 1``; at that value their posterior mean
        equals this function's.)
    lambda3 : float
        Lag-decay exponent. Larger = faster decay of prior std with k.
    intercept_prior_std : float
        Diffuse prior std on the intercept (large = nearly uninformative).
    lags : int, optional
        Keyword alias for ``p``.


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
    if lags is not None:
        if p is not None and p != lags:
            raise ValueError(f"Conflicting lag arguments: p={p}, lags={lags}")
        p = lags
    if p is None:
        raise ValueError("Must specify lag order via positional `p` or keyword `lags=...`")

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
    """Construct the Banbura-Giannone-Reichlin (2010) eq. (5) dummy block
    for the conjugate Normal-inverse-Wishart Minnesota prior.

    The augmented system ``[Y; Y_d] = [X; X_d] B + [u; u_d]`` has the NIW
    posterior used by :func:`minnesota_gibbs` and by the marginal
    likelihood. (:func:`minnesota_posterior` does not use this block; it
    builds its own per-equation dummies so that it can honour λ₂.)

    Rows, in order (``n`` variables, ``p`` lags, ``k_reg = 1 + n p``):

    * ``n p`` lag rows, one per (lag ``k``, variable ``j``), row
      ``(k-1) n + j``: ``X_d[row, 1+(k-1)n+j] = σ_j k^λ₃ / λ₁`` and
      ``Y_d[row, j] = σ_j / λ₁`` if ``k = 1`` else 0. The implied prior mean
      ``B₀ = (X_d'X_d)⁻¹ X_d'Y_d`` is 1 on every own first lag and 0
      elsewhere; the implied prior sd of coefficient (i, j, k) is
      ``λ₁ σ_i / (k^λ₃ σ_j)`` — the Minnesota moments with λ₂ = 1.
      With λ₃ = 1 this is BGR eq. (5), ``J_p ⊗ diag(σ)/λ``.
    * ``n`` rows ``Y_d = diag(σ)``, ``X_d = 0`` for the residual covariance.
    * 1 intercept row ``X_d[:, 0] = 1 / intercept_prior_std``.

    ``lambda2`` is accepted for signature compatibility and ignored: the
    Kronecker prior covariance ``Ψ ⊗ Ω₀`` cannot scale a coefficient
    differently in its own equation and in the others, so BGR impose
    θ = λ₂ = 1 (ECB WP 966, p. 11). The public NIW callers warn when
    λ₂ ≠ 1 is requested.
    """
    del lambda2  # not representable under NIW; see docstring
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

    n_rows = n * p + n + 1
    Yd = np.zeros((n_rows, n))
    Xd = np.zeros((n_rows, k_reg))
    # Lag rows: one per (k, j), a single non-zero regressor each.
    for k in range(1, p + 1):
        for j in range(n):
            row_idx = (k - 1) * n + j
            xd_coeff = sigmas[j] * (k ** lambda3) / lambda1
            Xd[row_idx, 1 + (k - 1) * n + j] = xd_coeff
            if k == 1:
                # prior mean δ_j = 1 (random walk) on the own first lag
                Yd[row_idx, j] = xd_coeff
    # Sigma dummies: one per equation.
    for eq in range(n):
        Yd[n * p + eq, eq] = sigmas[eq]
    # Intercept dummy.
    Xd[n_rows - 1, 0] = 1.0 / intercept_prior_std

    return Y_dep, X, Yd, Xd


def minnesota_gibbs(
    Y: pd.DataFrame,
    p: int | None = None,
    *,
    lags: int | None = None,
    n_draws: int = 1000,
    burn: int = 500,
    lambda1: float = 0.2,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    intercept_prior_std: float = 1e3,
    rng: np.random.Generator | None = None,
) -> dict:
    """Posterior draws from a Minnesota-prior BVAR via the Normal-Inverse-
    Wishart conjugate posterior (Banbura-Giannone-Reichlin 2010).

    The Minnesota prior is encoded as the BGR eq. (5) dummy observations
    (see ``_build_minnesota_dummies``); the augmented OLS system has a
    closed-form NIW posterior:

        Sigma | Y ~ IW(S_post, nu_post),  nu_post = T_d + 2 + T - k
        vec(B) | Sigma, Y ~ N(vec(B_post), Sigma kron (X*'X*)^{-1})

    (BGR eq. 7; ``T_d`` dummy rows, ``T`` observations after the ``p``
    initial lags, ``k = 1 + n p``; the ``+ 2`` comes from the improper prior
    ``|Sigma|^{-(n+3)/2}``. puremacro 4.4.0 and earlier used
    ``T_d + T - k``.)

    where (Y*, X*) stack data and dummies, ``B`` is ``(1 + n p, n)`` with the
    intercept in row 0 and ``A_k[i, j]`` (coefficient of ``y_j`` at lag ``k``
    in equation ``i``) in row ``1 + (k-1) n + j``, column ``i``.

    Prior moments (Minnesota with θ = λ₂ = 1): mean 1 on every own first
    lag and 0 elsewhere; sd ``λ₁ σ_i / (k^λ₃ σ_j)`` for the coefficient of
    ``y_j`` at lag ``k`` in equation ``i``; ``σ_i`` is the residual sd of a
    univariate AR(p) on ``y_i``, in the units of ``Y``.

    Parameters
    ----------
    lambda1 : float — overall tightness λ₁ (smaller = closer to a random walk).
    lambda2 : float — cross-variable factor. The NIW prior can only encode
              λ₂ = 1 (BGR 2010, "under the condition that θ = 1"); any other
              value raises a ``UserWarning`` and 1.0 is used. Use
              :func:`minnesota_posterior` for λ₂ ≠ 1 (posterior mean only).
              In puremacro 4.3.0 and earlier the default was 0.5 and a bug
              in the dummy block centred the own first lag of every variable
              but the last on λ₂ instead of 1.
    lambda3 : float — lag-decay exponent λ₃.
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
        A_mean        : (p, n, n) exact posterior mean of the A_k (B_post)
        intercept_mean : (n,) exact posterior mean of the intercept
        nu_post       : posterior degrees of freedom of Sigma, T_d + 2 + T - k
        lambda1, lambda2, lambda3 : hyperparameters actually used
    """
    if lags is not None:
        if p is not None and p != lags:
            raise ValueError(f"Conflicting lag arguments: p={p}, lags={lags}")
        p = lags
    if p is None:
        raise ValueError("Must specify lag order via positional `p` or keyword `lags=...`")

    lambda2 = _niw_lambda2(lambda2, "minnesota_gibbs")
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
    # BGR (2010) eq. (7): with the improper prior |Psi|^-(n+3)/2 that makes the
    # prior mean of Psi exist, Psi | Y ~ iW(S_post, T_d + 2 + T - k). Up to
    # 4.4.0 the "+ 2" was missing (docs/ADVISORY.md).
    nu_post = Y_aug.shape[0] + 2 - X_aug.shape[1]

    # Pre-Cholesky once with scale-invariant relative jitter; redraw each iter
    diag_kron = np.diag(XtX_inv)
    jitter_kron = 1e-12 * float(np.max(diag_kron)) if diag_kron.size else 0.0
    L_kron = safe_cholesky(XtX_inv, name="bvar_XtX_inv", jitter=jitter_kron)

    A_draws = np.empty((n_draws, p, n, n))
    Sigma_draws = np.empty((n_draws, n, n))
    intercept_draws = np.empty((n_draws, n))

    total = n_draws + burn
    # Batched vectorized sampling from NIW conjugate posterior
    Sigma_all = _inv_wishart(S_post, nu_post, rng, size=total)
    mean_diag = np.mean(np.diagonal(Sigma_all, axis1=1, axis2=2))
    jitter_sig = 1e-12 * float(mean_diag) if mean_diag > 0 else 1e-12
    L_sig_all = np.linalg.cholesky(Sigma_all + jitter_sig * np.eye(n))
    Z_all = rng.standard_normal((total, B_post.shape[0], n))
    B_all = B_post[None, :, :] + L_kron @ Z_all @ L_sig_all.transpose(0, 2, 1)

    B_draws = B_all[burn:]
    Sigma_draws = Sigma_all[burn:]
    intercept_draws = B_draws[:, 0, :]
    A_draws = np.empty((n_draws, p, n, n))
    for k in range(p):
        A_draws[:, k] = B_draws[:, 1 + k * n : 1 + (k + 1) * n, :].transpose(0, 2, 1)
    A_mean = np.stack([B_post[1 + k * n: 1 + (k + 1) * n, :].T for k in range(p)])

    return {
        "A_draws": A_draws,
        "Sigma_draws": Sigma_draws,
        "intercept_draws": intercept_draws,
        "A_mean": A_mean,
        "intercept_mean": B_post[0, :].copy(),
        "nu_post": int(nu_post),
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "lambda3": float(lambda3),
    }


def _inv_wishart(
    S: np.ndarray,
    df: int,
    rng: np.random.Generator,
    size: int | None = None,
) -> np.ndarray:
    """Draw Sigma ~ IW(S, df) via Bartlett decomposition of W = inv(Sigma) ~ W(inv(S), df)."""
    n = S.shape[0]
    S_inv = np.linalg.inv(S + 1e-12 * np.eye(n))
    L = np.linalg.cholesky(S_inv)
    if size is None:
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = np.sqrt(rng.chisquare(df - i))
            for j in range(i):
                A[i, j] = rng.standard_normal()
        W = L @ A @ A.T @ L.T
        return np.linalg.inv(W)
    else:
        A_batch = np.zeros((size, n, n))
        for i in range(n):
            A_batch[:, i, i] = np.sqrt(rng.chisquare(df - i, size=size))
            for j in range(i):
                A_batch[:, i, j] = rng.standard_normal(size=size)
        LA = L @ A_batch
        W_batch = LA @ LA.transpose(0, 2, 1)
        return np.linalg.inv(W_batch)


def _minnesota_log_marginal_likelihood(
    Y_arr: np.ndarray,
    p: int,
    lambda1: float,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    intercept_prior_std: float = 1e3,
) -> float:
    """Compute the log marginal likelihood of a Minnesota-prior BVAR
    via the dummy-observation representation (Banbura-Giannone-Reichlin
    2010 / Giannone-Lenza-Primiceri 2015 eq. 3).

    For the conjugate Normal-Inverse-Wishart prior implied by the
    BGR eq. (5) dummies (``T_d`` rows, ``k = 1 + n p`` regressors;
    ``*`` marks data stacked on dummies, ``ν = rows - k``), the value is

        log p(Y | λ) = log Γ_n(ν*/2) - log Γ_n(ν_d/2)
                       - n/2 (log|X*'X*| - log|X_d'X_d|)
                       - ν*/2 log|S*| + ν_d/2 log|S_d| - n T/2 log π

    with ``S`` the residual cross-product of each OLS. Only the argmax
    over hyperparameters is used. ``lambda2`` must be 1 (the NIW prior
    cannot encode another value; a ``UserWarning`` is raised and 1 used).
    """
    _niw_lambda2(lambda2, "_minnesota_log_marginal_likelihood")
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
    p: int | None = None,
    *,
    lags: int | None = None,
    intercept_prior_std: float = 1e3,
    lambda1_grid=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0),
    lambda2_grid=(1.0,),
    lambda3_grid=(1.0, 2.0),
) -> dict:
    """Empirical-Bayes choice of Minnesota hyperparameters via marginal
    likelihood maximisation (Giannone-Lenza-Primiceri 2015).

    Grids over ``λ_1`` and ``λ_3`` and returns the pair maximising the
    log marginal likelihood of the conjugate NIW Minnesota prior. This is
    the cheap "good default" version of GLP 2015; their full hierarchical
    procedure also optimises over a Sum-of-Coefficients prior tightness,
    which we omit for brevity.

    ``λ_2`` is fixed at 1: the NIW prior cannot encode a cross-variable
    factor (BGR 2010 impose θ = 1), so the marginal likelihood does not
    depend on it. ``lambda2_grid`` is kept for backward compatibility;
    values other than 1.0 are dropped with a ``UserWarning`` (in
    puremacro 4.3.0 and earlier the default grid was ``(0.5, 1.0)`` and
    the 0.5 entry evaluated a prior mis-centred by a dummy-block bug).
    """
    if lags is not None:
        if p is not None and p != lags:
            raise ValueError(f"Conflicting lag arguments: p={p}, lags={lags}")
        p = lags
    if p is None:
        raise ValueError("Must specify lag order via positional `p` or keyword `lags=...`")

    dropped = [l2 for l2 in lambda2_grid
               if not np.isclose(float(l2), 1.0, rtol=0.0, atol=1e-12)]
    if dropped:
        warnings.warn(
            f"minnesota_optimal_lambda: lambda2_grid values {dropped} dropped; "
            "the conjugate Normal-inverse-Wishart Minnesota prior can only "
            "encode lambda2 = 1 (Banbura, Giannone and Reichlin 2010 require "
            "theta = 1), so the grid over lambda2 is (1.0,).",
            UserWarning,
            stacklevel=2,
        )
    lambda2_grid = (1.0,)

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
