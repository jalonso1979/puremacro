"""Lewbel (2012) heteroskedasticity-based constructed IVs.

Given y = X_endog · β + X_exog · γ + u where X_endog is endogenous and no
external instrument is available, Lewbel constructs instruments from
heteroskedasticity in the auxiliary regression of X_endog on observed
``heterosk_source``. The constructed IVs are valid under the assumption
that ``Cov(heterosk_source · ν, u) = 0`` where ν is the first-stage
residual.

References
----------
Lewbel, A. (2012). Using heteroscedasticity to identify and estimate
    mismeasured and endogenous regressor models. JBES 30(1), 67-80.
"""
from __future__ import annotations

import warnings

import numpy as np

from ._results import LewbelIVResult
from .._linalg import inv_xtx


def _residualise(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return X minus its OLS projection onto W. Assumes W has a constant."""
    XtX_inv = inv_xtx(W, name="lewbel_iv Frisch-Waugh")
    return X - W @ (XtX_inv @ W.T @ X)


def _lewbel_diagnostic(X_endog_res: np.ndarray, Z: np.ndarray) -> dict:
    """Breusch-Pagan-style test: does Z drive heteroskedasticity in X_endog_res?

    Stacks all k_endog columns of X_endog_res into one vector of squared
    residuals and regresses on Z (+ constant), then tests the joint
    significance of Z's coefficients. The test statistic is
    ``LM = T · k_endog · R²`` ~ χ²(k_z) under the null.

    Stacked specification caveat: this test assumes a SHARED
    heteroskedasticity function across all k_endog equations. When two
    endogenous regressors have opposite-sign heteroskedasticity in the same
    Z (e.g., Var(e1) ∝ exp(+z) but Var(e2) ∝ exp(-z)), the contributions
    partially cancel and the test may report a deceptively large p-value
    even though each individual regressor is strongly heteroskedastic. For
    k_endog > 1, inspect per-column relevance separately if the joint test
    is weak.

    Parameters
    ----------
    X_endog_res : (T, k_endog)
        Endogenous regressors residualised against the exogenous controls.
    Z : (T, k_z)
        Original (non-residualised) heteroskedasticity source variables.

    Returns dict with ``stat`` (LM) and ``p_value`` (chi-squared)."""
    from scipy.stats import chi2
    T, k_e = X_endog_res.shape
    k_z = Z.shape[1]
    u2 = (X_endog_res ** 2).reshape(-1)
    Z_stack = np.tile(Z, (k_e, 1))
    Z_aug = np.column_stack([np.ones(Z_stack.shape[0]), Z_stack])
    XtX_inv = inv_xtx(Z_aug, name="lewbel_iv BP")
    beta = XtX_inv @ Z_aug.T @ u2
    pred = Z_aug @ beta
    ss_tot = float(np.sum((u2 - u2.mean()) ** 2))
    ss_res = float(np.sum((u2 - pred) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    stat = T * k_e * r2
    p = float(1.0 - chi2.cdf(stat, df=k_z))
    return {"stat": float(stat), "p_value": p}


def lewbel_iv(
    y: np.ndarray,
    X_endog: np.ndarray,
    X_exog: np.ndarray,
    heterosk_source: np.ndarray,
) -> LewbelIVResult:
    """2SLS with Lewbel-constructed instruments.

    Parameters
    ----------
    y : (T,) outcome.
    X_endog : (T, k_endog) endogenous regressors.
    X_exog : (T, k_exog) exogenous regressors (must include a constant column).
    heterosk_source : (T, k_z) observed drivers of heteroskedasticity in X_endog.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X_endog = np.asarray(X_endog, dtype=float)
    X_exog = np.asarray(X_exog, dtype=float)
    Z_source = np.asarray(heterosk_source, dtype=float)
    if X_endog.ndim == 1:
        X_endog = X_endog.reshape(-1, 1)
    if X_exog.ndim == 1:
        X_exog = X_exog.reshape(-1, 1)
    if Z_source.ndim == 1:
        Z_source = Z_source.reshape(-1, 1)

    T = y.size
    k_endog = X_endog.shape[1]
    k_exog = X_exog.shape[1]
    k_z = Z_source.shape[1]

    # Verify X_exog includes a constant column (required for Frisch-Waugh).
    has_constant = np.any(np.all(np.isclose(X_exog, X_exog[0:1, :], atol=1e-12), axis=0))
    if not has_constant:
        warnings.warn(
            "lewbel_iv: X_exog does not appear to contain a constant column; "
            "Lewbel-IV identification requires E[Z·ν] = 0, which is most "
            "reliably achieved with a constant in X_exog. Results may be biased.",
            stacklevel=2,
        )

    # 1. Frisch-Waugh residualise X_endog and Z against X_exog.
    X_endog_res = _residualise(X_endog, X_exog)
    Z_res = _residualise(Z_source, X_exog)

    # 2. Construct Lewbel IVs: Z_res * centred_endog (defensive centring;
    #    centred_endog is already mean-zero by Frisch-Waugh when X_exog
    #    contains a constant — re-centring guards user misuse).
    centred_endog = X_endog_res - X_endog_res.mean(axis=0, keepdims=True)
    # Column (k, j) of Z_constructed = Z_res[:, k] * centred_endog[:, j]
    Z_constructed = np.einsum("tk,tj->tkj", Z_res, centred_endog).reshape(T, k_z * k_endog)

    # 3. Diagnostic on identifying strength.
    diag = _lewbel_diagnostic(X_endog_res, Z_source)
    if diag["p_value"] > 0.10:
        warnings.warn(
            f"lewbel_iv: weak Lewbel diagnostic (p={diag['p_value']:.3f}). "
            "Heteroskedasticity source may not drive sufficient variation; "
            "treat results with caution.",
            stacklevel=2,
        )

    # 4. 2SLS. Endogenous regressors first, then exogenous (in the result).
    X_full = np.column_stack([X_endog, X_exog])
    Z_full = np.column_stack([Z_constructed, X_exog])  # instruments: constructed + exogenous

    # First stage: regress each X_endog column on Z_full
    ZtZ_inv = inv_xtx(Z_full, name="lewbel_iv first stage")
    Pi = ZtZ_inv @ Z_full.T @ X_endog  # shape (n_iv + k_exog, k_endog)
    X_endog_hat = Z_full @ Pi
    X_full_hat = np.column_stack([X_endog_hat, X_exog])

    # Second stage: regress y on (X_endog_hat | X_exog)
    XhX_inv = inv_xtx(X_full_hat, name="lewbel_iv second stage")
    beta = XhX_inv @ X_full_hat.T @ y
    resid = y - X_full @ beta  # residuals from original X, not projection
    sigma2 = float(resid @ resid / max(T - X_full.shape[1], 1))
    vcov = sigma2 * XhX_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    t = np.where(se > 0, beta / np.maximum(se, 1e-300), 0.0)

    # First-stage F: joint significance of Z_constructed in regression of
    # first endogenous regressor on Z_full.
    q = Z_constructed.shape[1]
    kU = Z_full.shape[1]
    ssr_u = float(((X_endog[:, 0] - X_endog_hat[:, 0]) ** 2).sum())
    Z_restr = X_exog
    XtX_r = inv_xtx(Z_restr, name="lewbel_iv F restricted")
    pi_r = XtX_r @ Z_restr.T @ X_endog[:, 0]
    ssr_r = float(((X_endog[:, 0] - Z_restr @ pi_r) ** 2).sum())
    if ssr_u > 0 and q > 0 and (T - kU) > 0:
        F = ((ssr_r - ssr_u) / q) / (ssr_u / (T - kU))
    else:
        F = float("nan")

    return LewbelIVResult(
        beta=beta,
        se=se,
        t=t,
        n_obs=int(T),
        n_iv_constructed=int(q),
        first_stage_F=float(F),
        lewbel_diagnostic=diag,
    )


__all__ = ["lewbel_iv"]
