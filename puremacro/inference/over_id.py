"""Over-identification: Hansen J test for 2SLS / IV-GMM, plus a small
Stock-Yogo critical-value lookup for weak-instrument decisions.

Hansen J (``robust=True``, the default):
    Two-step GMM with the heteroskedasticity-robust (HC0) weight matrix.
    Step 1 is 2SLS; its residuals ``u_1`` give
    ``S = (1/T) sum_t z_t z_t' u_1t^2``. Step 2 is the efficient GMM
    estimator ``b_2 = (X'Z S^{-1} Z'X)^{-1} X'Z S^{-1} Z'y`` and the
    statistic is

        J = T * gbar(b_2)' S^{-1} gbar(b_2),   gbar(b) = Z'(y - X b) / T,

    which is ``chi^2(l - k)`` under valid instruments and conditional
    heteroskedasticity of unknown form (Hansen 1982; Hayashi 2000, ch. 3;
    what ``ivreg2, gmm2s robust`` and ``linearmodels.IVGMM`` report).

Sargan (``robust=False``):
    The homoskedastic special case: regress the 2SLS residual on the
    instrument matrix (and the included controls); ``J = T * R^2``. This is
    what the function computed -- under the name Hansen J -- before 2.3.1.

Here ``l`` = number of excluded instruments and ``k`` = number of
endogenous regressors; the included exogenous controls ``W`` (constant
included) enter both the instrument set and the regressor set and do not
change the degrees of freedom.

Stock-Yogo critical values are tabulated for the maximal IV bias
relative to OLS (Stock-Yogo 2005, Tables 5.2 / 5.3). We expose a
small lookup for the single-endogenous-regressor case at the most
common "max bias" thresholds.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .._linalg import inv_xtx


def hansen_j(
    y: np.ndarray,
    X_endog: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray | None = None,
    *,
    robust: bool = True,
) -> dict:
    """Hansen J (robust two-step GMM) or Sargan over-identification statistic.

    Parameters
    ----------
    y       : (T,) outcome.
    X_endog : (T, k) endogenous regressors.
    Z       : (T, l) excluded instruments (l > k for the test to be meaningful).
    W       : (T, m) included exogenous controls; a constant is prepended
              when the first column is not all ones. ``None`` means a
              constant only.
    robust  : if True (default) the heteroskedasticity-robust Hansen J
              from two-step GMM with the HC0 weight matrix; if False the
              homoskedastic Sargan statistic ``T * R^2`` of the 2SLS residual
              on the instruments.

    Returns
    -------
    dict with stat (J), p_value, df (= l - k), n_obs.
    ``p_value`` is NaN when df = 0 (just identified: J is identically 0).

    References
    ----------
    Hansen, L.P. (1982). Large sample properties of generalized method of
        moments estimators. Econometrica 50(4), 1029-1054.
    Sargan, J.D. (1958). The estimation of economic relationships using
        instrumental variables. Econometrica 26(3), 393-415.
    Hayashi, F. (2000). Econometrics. Princeton University Press, ch. 3.
    """
    y = np.asarray(y, dtype=float).ravel()
    X_endog = np.asarray(X_endog, dtype=float)
    Z = np.asarray(Z, dtype=float)
    X = X_endog.reshape(-1, 1) if X_endog.ndim == 1 else X_endog
    Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
    T = len(y)
    k = X.shape[1]
    l = Z.shape[1]
    if X.shape[0] != T or Z.shape[0] != T:
        raise ValueError(
            f"hansen_j: y has {T} rows but X_endog has {X.shape[0]} and Z has {Z.shape[0]}"
        )

    if W is None:
        W = np.ones((T, 1))
    else:
        W = np.asarray(W, dtype=float)
        W = W.reshape(-1, 1) if W.ndim == 1 else W
        # ensure a constant is present
        if not np.allclose(W[:, 0], 1.0):
            W = np.column_stack([np.ones(T), W])

    full_Z = np.column_stack([W, Z])       # (T, m + l) instrument set
    XW = np.column_stack([X, W])           # (T, k + m) regressors

    # Step 1: 2SLS.  X_hat = P_Z X without forming the T x T projection.
    ZtZ_inv = inv_xtx(full_Z, name="hansen_j (Z'Z)")
    X_hat = full_Z @ (ZtZ_inv @ (full_Z.T @ X))
    XW_hat = np.column_stack([X_hat, W])
    beta_2sls = np.linalg.solve(XW_hat.T @ XW, XW_hat.T @ y)
    u_1 = y - XW @ beta_2sls

    if not robust:
        # Sargan: T * R^2 of the 2SLS residual on the instruments.
        proj = full_Z @ (ZtZ_inv @ (full_Z.T @ u_1))
        rss = float(np.sum((u_1 - proj) ** 2))
        tss = float(np.sum((u_1 - u_1.mean()) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else 0.0
        J = T * r2
    else:
        # HC0 weight matrix from the first-step residuals.
        S = (full_Z.T * u_1 ** 2) @ full_Z / T
        try:
            S_inv_ZX = np.linalg.solve(S, full_Z.T @ XW)
            S_inv_Zy = np.linalg.solve(S, full_Z.T @ y)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "hansen_j: the HC0 weight matrix S = Z' diag(u^2) Z / T is "
                "singular (collinear instruments, or residuals that are "
                "exactly zero)"
            ) from exc
        # Step 2: efficient GMM with weight S^{-1}.
        ZX = full_Z.T @ XW
        beta_gmm = np.linalg.solve(ZX.T @ S_inv_ZX, ZX.T @ S_inv_Zy)
        u_2 = y - XW @ beta_gmm
        gbar = full_Z.T @ u_2 / T
        J = float(T * gbar @ np.linalg.solve(S, gbar))

    df = l - k
    p_value = chi2.sf(J, df=df) if df > 0 else np.nan
    return {
        "stat": float(J),
        "p_value": float(p_value),
        "df": int(df),
        "n_obs": int(T),
    }


# Stock-Yogo (2005) Table 5.2 — single endogenous regressor (k=1),
# 2SLS critical values for "max bias relative to OLS" thresholds.
# Keys: (n_instruments, max_bias_pct) -> first-stage F critical value.
_STOCK_YOGO_BIAS_K1 = {
    # (l, threshold%) -> CV
    (3,  5):  13.91, (3,  10):  9.08, (3,  20): 6.46, (3,  30): 5.39,
    (4,  5):  16.85, (4,  10): 10.27, (4,  20): 6.71, (4,  30): 5.34,
    (5,  5):  18.37, (5,  10): 10.83, (5,  20): 6.77, (5,  30): 5.25,
    (6,  5):  19.28, (6,  10): 11.12, (6,  20): 6.76, (6,  30): 5.15,
    (7,  5):  19.86, (7,  10): 11.29, (7,  20): 6.73, (7,  30): 5.07,
    (8,  5):  20.25, (8,  10): 11.39, (8,  20): 6.69, (8,  30): 4.99,
}


def stock_yogo_cv(n_instruments: int, max_bias_pct: int = 10) -> float | None:
    """Look up the Stock-Yogo critical value for first-stage F.

    Compare your first-stage F-stat against this CV; reject "weak
    instruments" if F exceeds it.

    Parameters
    ----------
    n_instruments : int   — number of excluded instruments l.
    max_bias_pct  : int   — maximal IV bias relative to OLS, in % (5/10/20/30).

    Returns
    -------
    Critical value (float) or None if (l, max_bias_pct) is not tabulated.
    The classic Staiger-Stock rule of thumb of "F > 10" corresponds to
    n_instruments=3, max_bias_pct=10 in this table.
    """
    return _STOCK_YOGO_BIAS_K1.get((int(n_instruments), int(max_bias_pct)))


__all__ = ["hansen_j", "stock_yogo_cv"]
