"""High-dimensional factor models: PCA factors + Bai-Ng (2002) IC for k.

Static factor model:
    X_t = Λ F_t + e_t,   X_t ∈ R^n,   F_t ∈ R^k

Estimator: PCA on the (T × n) standardised data matrix. Factors are
the first k left singular vectors (rescaled by sqrt(T)); loadings come
from the right singular vectors weighted by the singular values.

Bai-Ng (2002) provide three penalty-based ICs that are consistent for
the true k as both n and T grow.

References
----------
Stock, J.H. and Watson, M.W. (2002). Forecasting using principal
    components from a large number of predictors. JASA 97.
Bai, J. and Ng, S. (2002). Determining the number of factors in
    approximate factor models. Econometrica 70(1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _standardise(X: np.ndarray, demean: bool, standardize: bool) -> np.ndarray:
    if demean:
        X = X - X.mean(axis=0, keepdims=True)
    if standardize:
        sd = X.std(axis=0, keepdims=True, ddof=0)
        X = X / np.where(sd == 0, 1.0, sd)
    return X


def pca_factors(
    X: np.ndarray,
    k: int = 1,
    *,
    demean: bool = True,
    standardize: bool = True,
) -> dict:
    """Extract the first k principal-component factors.

    Parameters
    ----------
    X : (T, n) ndarray.
    k : number of factors to extract.

    Returns
    -------
    dict with
        factors : (T, k) — F_t with unit variance per column.
        loadings : (n, k) — Λ such that X ≈ F Λ'.
        eigvals  : (k,)   — variance of each factor (cumulative explanatory).
        variance_share : (k,) — share of total variance explained.
    """
    X = np.asarray(X, dtype=float)
    X = _standardise(X, demean, standardize)
    T, n = X.shape
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    factors = np.sqrt(T) * U[:, :k]
    loadings = (Vt[:k, :] * (S[:k] / np.sqrt(T))[:, None]).T  # (n, k)
    eigvals = (S[:k] ** 2) / T
    total_var = float((S ** 2 / T).sum())
    var_share = eigvals / total_var if total_var > 0 else np.zeros(k)
    return {
        "factors": factors,
        "loadings": loadings,
        "eigvals": eigvals,
        "variance_share": var_share,
        "cumulative_share": np.cumsum(var_share),
    }


def bai_ng_ic(
    X: np.ndarray,
    kmax: int = 10,
    *,
    demean: bool = True,
    standardize: bool = True,
) -> dict:
    """Bai-Ng (2002) ICs for choosing the number of factors.

    For each candidate k (0 to kmax), compute V(k) = ||X - X̂_k||² / (nT)
    where X̂_k is the best rank-k approximation. The three ICs balance
    fit against a complexity penalty:

        IC1(k) = log V(k) + k * (n + T)/(nT) * log((nT)/(n + T))
        IC2(k) = log V(k) + k * (n + T)/(nT) * log(min(n, T))
        IC3(k) = log V(k) + k * log(min(n, T)) / min(n, T)

    Returns the full table plus the argmin of each IC.
    """
    X = np.asarray(X, dtype=float)
    X = _standardise(X, demean, standardize)
    T, n = X.shape
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    rows = []
    for k in range(0, kmax + 1):
        if k == 0:
            X_hat = np.zeros_like(X)
        else:
            X_hat = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        V_k = float(((X - X_hat) ** 2).sum() / (n * T))
        log_V = np.log(V_k) if V_k > 0 else -np.inf
        ic1 = log_V + k * (n + T) / (n * T) * np.log((n * T) / (n + T)) if (n + T) > 0 else log_V
        ic2 = log_V + k * (n + T) / (n * T) * np.log(min(n, T)) if min(n, T) > 1 else log_V
        ic3 = log_V + k * np.log(min(n, T)) / min(n, T) if min(n, T) > 1 else log_V
        rows.append({"k": k, "V_k": V_k, "IC1": ic1, "IC2": ic2, "IC3": ic3})
    df = pd.DataFrame(rows)
    return {
        "table": df,
        "k_hat_IC1": int(df.loc[df["IC1"].idxmin(), "k"]),
        "k_hat_IC2": int(df.loc[df["IC2"].idxmin(), "k"]),
        "k_hat_IC3": int(df.loc[df["IC3"].idxmin(), "k"]),
    }


def static_dfm_fit(
    X: np.ndarray,
    *,
    k: int | None = None,
    kmax: int = 8,
    demean: bool = True,
    standardize: bool = True,
) -> dict:
    """Convenience: choose k by Bai-Ng IC2 (if not supplied) and run PCA."""
    X = np.asarray(X, dtype=float)
    if k is None:
        ic = bai_ng_ic(X, kmax=kmax, demean=demean, standardize=standardize)
        k = ic["k_hat_IC2"]
        ic_table = ic["table"]
    else:
        ic_table = None
    out = pca_factors(X, k=k, demean=demean, standardize=standardize)
    out["k"] = int(k)
    out["bai_ng_table"] = ic_table
    return out


__all__ = ["pca_factors", "bai_ng_ic", "static_dfm_fit"]
