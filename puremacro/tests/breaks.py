"""Bai-Perron (1998, 2003) multiple structural breaks via DP.

Estimates the optimal partition for a regression y = X β + ε with up
to `max_breaks` breaks, subject to a minimum-segment-length constraint
(`trim` fraction of T).

For v1 we restrict attention to a constant-regression model
(y_t = μ_g + ε_t) — i.e., the "mean-shift" case. Extending to a
general design X is straightforward (replace the OLS in each segment
with the user's regressors), but adds boilerplate without changing
the DP structure.

The algorithm:
  1. Pre-compute SSR(s, t) — the residual SS of OLS on segment [s, t].
  2. For each m ∈ {1, …, max_breaks}, find the partition into m+1
     segments that minimises total SSR, subject to each segment having
     length ≥ trim*T. Use dynamic programming.
  3. Sequential F-test (Bai-Perron 2003): test H_0(m) vs H_1(m+1) by
     comparing nested SSRs. Default selection: pick m̂ = largest m with
     significant F at α=0.05; user can override via `force_m`.

Sequential F critical values (5%) are taken from Bai-Perron (2003)
Table 1, simplified for n_regressors = 1 (constant only). For larger
designs, swap in the appropriate column.
"""
from __future__ import annotations

import numpy as np

# Bai-Perron (2003) Table 1, sequential F 5% CV for n_regressors=1
# Indexed by ell+1 (number of breaks under the alternative).
_BP_F_CV_5PCT = {1: 8.58, 2: 7.22, 3: 5.96, 4: 4.99, 5: 4.10}


def _segment_ssr(y: np.ndarray) -> np.ndarray:
    """SSR(s, t) = Σ_{i=s..t} (y_i - mean(y[s..t+1]))^2 for all (s, t)."""
    T = len(y)
    csum = np.concatenate([[0.0], np.cumsum(y)])
    csum2 = np.concatenate([[0.0], np.cumsum(y ** 2)])
    ssr = np.full((T, T + 1), np.inf)
    for s in range(T):
        for t in range(s + 1, T + 1):
            n = t - s
            mean = (csum[t] - csum[s]) / n
            sumsq = csum2[t] - csum2[s]
            ssr[s, t] = sumsq - n * mean ** 2
    return ssr


def _dp_partition(ssr: np.ndarray, m: int, min_len: int) -> tuple[list[int], float]:
    """Find the optimal m-break partition (i.e., m+1 segments).

    Returns (break_dates, total_SSR).
    """
    T = ssr.shape[0]
    # cost[k][t] = best total SSR partitioning [0, t] into k+1 segments.
    # We want cost[m][T].
    cost = np.full((m + 1, T + 1), np.inf)
    last = np.full((m + 1, T + 1), -1, dtype=int)  # back-pointer
    # k = 0: one segment [0, t]
    for t in range(min_len, T + 1):
        cost[0, t] = ssr[0, t]
    # k breaks: best of cost[k-1, s] + ssr[s, t]
    for k in range(1, m + 1):
        for t in range((k + 1) * min_len, T + 1):
            for s in range(k * min_len, t - min_len + 1):
                if cost[k - 1, s] + ssr[s, t] < cost[k, t]:
                    cost[k, t] = cost[k - 1, s] + ssr[s, t]
                    last[k, t] = s
    # Recover break dates
    breaks: list[int] = []
    t = T
    for k in range(m, 0, -1):
        s = int(last[k, t])
        if s < 0:
            break
        breaks.append(s)
        t = s
    breaks.reverse()
    return breaks, float(cost[m, T])


def bai_perron(y, max_breaks: int = 5, trim: float = 0.15,
                force_m: int | None = None, alpha: float = 0.05) -> dict:
    """Bai-Perron multiple structural breaks for a constant-regression model.

    Returns
    -------
    dict with keys:
        breaks : dict[int, list[int]] — for each m ∈ {0, …, max_breaks},
                 the optimal break dates (rising indices, exclusive — a
                 break at index k means segments [..k) and [k..])
        ssrs   : dict[int, float] — total SSR at each m
        selected_m : int — m̂ chosen by sequential F at level alpha
        max_breaks : int — input echoed
    """
    y = np.asarray(y, dtype=float).ravel()
    T = len(y)
    min_len = max(1, int(np.floor(trim * T)))
    ssr = _segment_ssr(y)

    breaks: dict[int, list[int]] = {0: []}
    ssrs: dict[int, float] = {0: ssr[0, T]}
    for m in range(1, max_breaks + 1):
        if (m + 1) * min_len > T:
            breaks[m] = breaks[m - 1]
            ssrs[m] = ssrs[m - 1]
            continue
        b, total = _dp_partition(ssr, m, min_len)
        breaks[m] = b
        ssrs[m] = total

    # Sequential F-test selection (Bai-Perron 2003 sup-F_T(ell+1 | ell))
    # F_stat = T * (SSR(ell) - SSR(ell+1)) / SSR(ell+1) approximated.
    selected_m = 0
    if force_m is not None:
        selected_m = int(force_m)
    else:
        for ell in range(0, max_breaks):
            ssr_ell = ssrs[ell]; ssr_next = ssrs[ell + 1]
            if ssr_next <= 0:
                break
            f_stat = T * (ssr_ell - ssr_next) / ssr_next
            cv = _BP_F_CV_5PCT.get(ell + 1, 4.10)
            if f_stat > cv:
                selected_m = ell + 1
            else:
                break

    return {
        "breaks": breaks,
        "ssrs": ssrs,
        "selected_m": int(selected_m),
        "max_breaks": int(max_breaks),
    }


def _segment_ssr_regression(y: np.ndarray, X: np.ndarray, min_len: int) -> np.ndarray:
    """SSR(s, t) of OLS y[s:t] on X[s:t] for all valid (s, t) with t-s >= min_len."""
    T, k = X.shape
    ssr = np.full((T, T + 1), np.inf)
    for s in range(T):
        # Iterate t expanding from s + min_len so every segment has identifiable OLS.
        for t in range(s + min_len, T + 1):
            Xs = X[s:t]
            ys = y[s:t]
            try:
                beta, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
                resid = ys - Xs @ beta
                ssr[s, t] = float(resid @ resid)
            except np.linalg.LinAlgError:
                continue
    return ssr


def bai_perron_regression(y, X, max_breaks: int = 5, trim: float = 0.15,
                          force_m: int | None = None) -> dict:
    """Bai-Perron multiple breaks for a generic regression y = X beta + u.

    Identical DP recursion to ``bai_perron``, but each segment is fit by
    full OLS on the supplied design matrix X (e.g., y_t = alpha + x_t' beta + u_t
    where the intercept and slopes both shift across regimes).

    Parameters
    ----------
    y : (T,) ndarray
    X : (T, k) ndarray (include a constant column if you want intercept shifts)
    max_breaks : int
    trim : float — minimum-segment fraction.
    force_m : int or None — override sequential-F selection.

    Returns
    -------
    Same dict as ``bai_perron``.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(X if np.asarray(X).ndim > 1 else np.asarray(X)[:, None]).astype(float)
    T = len(y)
    k = X.shape[1]
    min_len = max(k + 1, int(np.floor(trim * T)))
    ssr = _segment_ssr_regression(y, X, min_len)

    breaks: dict[int, list[int]] = {0: []}
    ssrs: dict[int, float] = {0: ssr[0, T]}
    for m in range(1, max_breaks + 1):
        if (m + 1) * min_len > T:
            breaks[m] = breaks[m - 1]
            ssrs[m] = ssrs[m - 1]
            continue
        b, total = _dp_partition(ssr, m, min_len)
        breaks[m] = b
        ssrs[m] = total

    selected_m = 0
    if force_m is not None:
        selected_m = int(force_m)
    else:
        for ell in range(0, max_breaks):
            ssr_ell = ssrs[ell]; ssr_next = ssrs[ell + 1]
            if ssr_next <= 0:
                break
            f_stat = (T - (ell + 2) * k) * (ssr_ell - ssr_next) / ssr_next
            cv = _BP_F_CV_5PCT.get(ell + 1, 4.10)
            if f_stat > cv:
                selected_m = ell + 1
            else:
                break

    return {
        "breaks": breaks,
        "ssrs": ssrs,
        "selected_m": int(selected_m),
        "max_breaks": int(max_breaks),
    }


__all__ = ["bai_perron", "bai_perron_regression"]
