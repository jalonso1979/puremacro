"""High-frequency surprise construction.

Public functions:
    - :func:`gk2015_surprise`  — Gertler-Karadi 2015 month-end-adjusted FFR-futures change.
    - :func:`ns2018_first_pc`  — Nakamura-Steinsson 2018 first-PC of multiple policy contracts.
    - :func:`aggregate_to_period` — sum announcement-day surprises into monthly/quarterly bins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def gk2015_surprise(
    ff_futures_pre: np.ndarray,
    ff_futures_post: np.ndarray,
    days_remaining_in_month: np.ndarray,
    days_in_month: int | np.ndarray = 30,
) -> np.ndarray:
    """Gertler-Karadi (2015) high-frequency monetary surprise.

    Computes ``(post - pre) * M / (M - d_elapsed)`` where ``d_elapsed = M - days_remaining``.
    The scaling factor adjusts for the fact that federal-funds-futures payoff
    is the *average* effective FFR over the month: only the post-announcement
    portion of the month carries the shock.

    Parameters
    ----------
    ff_futures_pre : ndarray, shape (n_announce,)
        Futures price (or implied rate) immediately before the announcement.
    ff_futures_post : ndarray, shape (n_announce,)
        Futures price (or implied rate) immediately after the announcement.
    days_remaining_in_month : ndarray of int, shape (n_announce,)
        Calendar days remaining in the announcement month *including* the
        announcement day. Must be > 0.
    days_in_month : int or ndarray, default 30
        Total calendar days in the announcement month. Pass an array if the
        month length varies across announcements.

    Returns
    -------
    surprise : ndarray, shape (n_announce,)
        Scale-adjusted high-frequency surprise (same units as the inputs).

    References
    ----------
    Gertler, M. and Karadi, P. (2015). Monetary policy surprises, credit
        costs, and economic activity. AEJ:Macro 7(1), 44-76.
    """
    pre = np.asarray(ff_futures_pre, dtype=float)
    post = np.asarray(ff_futures_post, dtype=float)
    rem = np.asarray(days_remaining_in_month, dtype=float)
    M = np.asarray(days_in_month, dtype=float)
    if np.any(rem <= 0):
        raise ValueError(
            "gk2015_surprise: days_remaining_in_month must be > 0 "
            "(announcement on or after the last day of the month is invalid)."
        )
    return (post - pre) * (M / rem)


def ns2018_first_pc(
    surprise_matrix: np.ndarray,
    scale_to_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nakamura-Steinsson (2018) first-PC of multiple announcement-window changes.

    Computes the first principal component of K policy-sensitive contracts'
    announcement-window changes, rescaled so a unit of the PC corresponds to a
    unit change in the contract at ``scale_to_idx``.

    Parameters
    ----------
    surprise_matrix : ndarray, shape (n_announce, K)
        Matrix of K contracts' surprises across n_announce announcements.
    scale_to_idx : int, default 0
        Index of the contract to which the PC is rescaled. The recovered
        loading on this contract is positive by construction.

    Returns
    -------
    pc : ndarray, shape (n_announce,)
        First-PC time series, scaled in the units of the target contract.
    loadings : ndarray, shape (K,)
        Recovered factor loadings (one entry per contract).

    Notes
    -----
    Uses SVD of the demeaned surprise matrix. The PC is sign-normalized so
    the loading on the target contract is positive.

    References
    ----------
    Nakamura, E. and Steinsson, J. (2018). High-frequency identification
        of monetary non-neutrality: the information effect.
        QJE 133(3), 1283-1330.
    """
    X = np.asarray(surprise_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            f"ns2018_first_pc: surprise_matrix must be 2-D, got shape {X.shape}"
        )
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc_raw = U[:, 0] * S[0]                # length n_announce
    loadings_raw = Vt[0, :]                # length K
    # Sign-normalise so loading at target contract is positive
    if loadings_raw[scale_to_idx] < 0:
        pc_raw = -pc_raw
        loadings_raw = -loadings_raw
    # Rescale: PC in units of target contract → loading[scale_to_idx] = 1 / scale
    target_loading = loadings_raw[scale_to_idx]
    if abs(target_loading) < 1e-12:
        raise np.linalg.LinAlgError(
            f"ns2018_first_pc: target contract idx {scale_to_idx} has zero "
            f"loading on first PC; choose a different scale_to_idx."
        )
    pc = pc_raw * target_loading
    loadings = loadings_raw / target_loading
    return pc, loadings


def aggregate_to_period(
    surprises: np.ndarray,
    dates,
    freq: str = "M",
) -> pd.Series:
    """Sum announcement-day surprises into period bins (monthly, quarterly).

    Periods with no announcement appear with value ``0.0``, not dropped — this
    matches the convention used in macro VARs where a "no announcement" period
    is informationally equivalent to a zero-surprise period.

    Parameters
    ----------
    surprises : ndarray, shape (n_announce,)
        Surprise series (e.g., output of :func:`gk2015_surprise`).
    dates : array-like of datetimes, length n_announce
        Announcement dates.
    freq : str, default "M"
        Pandas offset alias. ``"M"`` = month-end, ``"Q"`` = quarter-end.

    Returns
    -------
    pd.Series
        Indexed by period (PeriodIndex), values are the sum of all surprises
        falling within that period; periods between min(dates) and max(dates)
        with no announcement are zero-filled.
    """
    s = pd.Series(np.asarray(surprises, dtype=float), index=pd.to_datetime(dates))
    grouped = s.groupby(s.index.to_period(freq)).sum()
    full_idx = pd.period_range(grouped.index.min(), grouped.index.max(), freq=freq)
    return grouped.reindex(full_idx, fill_value=0.0)
