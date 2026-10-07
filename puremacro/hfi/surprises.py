"""High-frequency surprise construction.

Public functions:
    - :func:`gk2015_surprise`  — Gertler-Karadi 2015 month-end-adjusted FFR-futures implied-rate change (positive = tightening).
    - :func:`ns2018_first_pc`  — Nakamura-Steinsson 2018 first-PC of multiple policy contracts.
    - :func:`aggregate_to_period` — sum (or GK 2015 period-average) announcement-day surprises into monthly/quarterly bins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_GK2015_QUOTES = ("rate", "price")


def gk2015_surprise(
    ff_futures_pre: np.ndarray,
    ff_futures_post: np.ndarray,
    days_remaining_in_month: np.ndarray,
    days_in_month: int | np.ndarray = 30,
    *,
    quote: str = "rate",
) -> np.ndarray:
    """Gertler-Karadi (2015) high-frequency monetary surprise.

    Returns the surprise in the futures-implied **rate**, scaled for the
    within-month timing of the meeting:

    .. math::

        s = (f^{post} - f^{pre}) \\cdot \\frac{T}{T - t},

    where ``f`` is the implied rate of the current-month federal-funds futures
    contract, ``T = days_in_month`` and ``t`` is the number of days of the
    month elapsed *before* the meeting, so ``T - t = days_remaining_in_month``
    counted *including* the meeting day (a meeting on day ``k`` has
    ``t = k - 1``). This is GK (2015) eq. (19),
    ``(E_t i_t)^u = f_t - f_{t,-1}``, with the footnote-6 factor ``T/(T - t)``
    that they take from Kuttner (2001): the contract pays the *average* rate
    of the month, so only the part of the month after the meeting carries the
    surprise.

    **Sign convention.** The result is an interest-rate surprise: **positive
    means a tightening** (the market-implied policy rate rose). CME quotes
    fed-funds futures as a *price* ``100 - implied rate``, so a tightening
    lowers the quoted price. Pass ``quote="price"`` for price inputs; the
    function then returns minus the scaled price change, which is the same
    rate surprise. Passing prices with the default ``quote="rate"`` returns the
    surprise with the wrong sign, which reverses the sign of a proxy-SVAR's
    identified shock and swaps the Jarocinski-Karadi (2020) monetary and
    information shocks.

    Parameters
    ----------
    ff_futures_pre : ndarray, shape (n_announce,)
        Contract quote just before the announcement (GK use a 30-minute window
        around it), in the convention named by ``quote``.
    ff_futures_post : ndarray, shape (n_announce,)
        Contract quote just after the announcement, same convention.
    days_remaining_in_month : ndarray of int, shape (n_announce,)
        ``T - t``: calendar days remaining in the announcement month
        *including* the announcement day. Must satisfy
        ``0 < days_remaining_in_month <= days_in_month``.
    days_in_month : int or ndarray, default 30
        ``T``: calendar days in the announcement month. Pass an array if the
        month length varies across announcements.
    quote : {'rate', 'price'}, default 'rate'
        How the two quotes are expressed.

        * ``'rate'`` -- the futures-implied rate (``100 - price``), e.g. in
          percent per annum. The surprise is ``post - pre``, scaled.
        * ``'price'`` -- the exchange price ``100 - implied rate``. The
          surprise is ``-(post - pre)``, scaled.

    Returns
    -------
    surprise : ndarray, shape (n_announce,)
        Scaled implied-rate surprise, positive for a tightening. Units are the
        rate units of the inputs: percentage points when the quotes are in
        percent (``0.25`` = 25 bp).

    Notes
    -----
    The factor ``T/(T - t)`` grows as the meeting approaches the end of the
    month (``T`` on the last day), which amplifies measurement noise in the
    last days of a month. This function applies it as written and does not
    switch to the next month's contract.

    References
    ----------
    Gertler, M. and Karadi, P. (2015). Monetary policy surprises, credit
        costs, and economic activity. AEJ:Macro 7(1), 44-76. Working-paper
        version: NBER WP 20224 (2014), eq. (19) and footnotes 5-6, pp. 13-14.
    Kuttner, K. N. (2001). Monetary policy surprises and interest rates:
        evidence from the Fed funds futures market. JME 47(3), 523-544.

    Examples
    --------
    A 25 bp move in the implied rate with 15 of 30 days left is a 0.5 pp
    target-rate surprise. The same move quoted as a price (95.00 -> 94.75):

    >>> gk2015_surprise([5.00], [5.25], [15], 30)
    array([0.5])
    >>> gk2015_surprise([95.00], [94.75], [15], 30, quote="price")
    array([0.5])
    """
    if quote not in _GK2015_QUOTES:
        raise ValueError(
            f"gk2015_surprise: unknown quote {quote!r}; expected 'rate' "
            "(futures-implied rate, 100 - price) or 'price' (exchange price "
            "quoted as 100 - rate)."
        )
    pre = np.asarray(ff_futures_pre, dtype=float)
    post = np.asarray(ff_futures_post, dtype=float)
    rem = np.asarray(days_remaining_in_month, dtype=float)
    M = np.asarray(days_in_month, dtype=float)
    if np.any(rem <= 0):
        raise ValueError(
            "gk2015_surprise: days_remaining_in_month must be > 0 "
            "(announcement on or after the last day of the month is invalid)."
        )
    if np.any(rem > M):
        raise ValueError(
            "gk2015_surprise: days_remaining_in_month cannot exceed days_in_month "
            "(it counts the days from the announcement day to the month end, "
            "announcement day included)."
        )
    change = post - pre
    if quote == "price":
        change = -change
    return change * (M / rem)


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


_AGGREGATION_METHODS = ("sum", "gk2015")


def aggregate_to_period(
    surprises: np.ndarray,
    dates,
    freq: str = "M",
    method: str = "sum",
) -> pd.Series:
    """Aggregate announcement-day surprises into period bins (monthly, quarterly).

    Periods with no announcement appear with value ``0.0``, not dropped — this
    matches the convention used in macro VARs where a "no announcement" period
    is informationally equivalent to a zero-surprise period.

    Parameters
    ----------
    surprises : ndarray, shape (n_announce,)
        Surprise series (e.g., output of :func:`gk2015_surprise`).
    dates : array-like of datetimes, length n_announce
        Announcement dates (the time of day is ignored).
    freq : str, default "M"
        Pandas offset alias. ``"M"`` = month-end, ``"Q"`` = quarter-end.
    method : {'sum', 'gk2015'}, default 'sum'
        * ``'sum'`` -- each surprise is added to the period that contains its
          announcement. Appropriate when the VAR's policy indicator is an
          end-of-period rate.
        * ``'gk2015'`` -- Gertler-Karadi (2015) monthly *average* surprise
          (NBER WP 20224, footnote 11, pp. 20-21): cumulate the FOMC-day
          surprises into a daily series, average that series over the days of
          each period, and take the first difference of the averages. GK use
          it because their VAR uses period-average rates, on which a surprise
          late in the month has only a small effect. In closed form, a
          surprise ``s`` announced on day ``k`` of a period with ``D`` days
          contributes ``s * (D - k + 1) / D`` to that period and
          ``s * (k - 1) / D`` to the next one (the announcement day counts as
          already affected, consistent with ``days_remaining_in_month`` in
          :func:`gk2015_surprise`). The weights sum to one, so the total
          surprise is preserved. The output index runs one period past the
          last announcement's period, which holds the carried-over part.
          GK state the recipe two ways -- for each day, sum the FOMC surprises
          of the last 31 days and average within the month, or,
          "equivalently", difference the monthly averages of the cumulative
          daily series. This function implements the second form.

    Returns
    -------
    pd.Series
        Indexed by period (PeriodIndex). With ``method='sum'`` the index runs
        from the first to the last announcement period; with
        ``method='gk2015'`` it extends one period further. Periods without an
        announcement are zero-filled.

    References
    ----------
    Gertler, M. and Karadi, P. (2015). Monetary policy surprises, credit
        costs, and economic activity. AEJ:Macro 7(1), 44-76. Working-paper
        version: NBER WP 20224 (2014), footnote 11.
    """
    if method not in _AGGREGATION_METHODS:
        raise ValueError(
            f"aggregate_to_period: unknown method {method!r}; expected 'sum' "
            "(add each surprise to its own period) or 'gk2015' (Gertler-Karadi "
            "2015 period-average surprise)."
        )
    values = np.asarray(surprises, dtype=float)
    idx = pd.to_datetime(dates)
    s = pd.Series(values, index=idx)
    if method == "sum":
        grouped = s.groupby(s.index.to_period(freq)).sum()
        full_idx = pd.period_range(grouped.index.min(), grouped.index.max(), freq=freq)
        return grouped.reindex(full_idx, fill_value=0.0)

    # method == "gk2015": split each surprise between its period and the next
    # by the share of the period's days on or after the announcement day.
    days = pd.DatetimeIndex(idx).normalize()
    periods = days.to_period(freq)
    starts = periods.start_time.normalize()
    next_starts = (periods + 1).start_time.normalize()
    n_days = np.asarray((next_starts - starts).days, dtype=float)
    k = np.asarray((days - starts).days, dtype=float) + 1.0   # 1-based day in period
    w_own = (n_days - k + 1.0) / n_days
    full_idx = pd.period_range(periods.min(), periods.max() + 1, freq=freq)
    out = pd.Series(0.0, index=full_idx)
    own = pd.Series(values * w_own, index=periods).groupby(level=0).sum()
    carry = pd.Series(values * (1.0 - w_own), index=periods + 1).groupby(level=0).sum()
    out = out.add(own, fill_value=0.0).add(carry, fill_value=0.0)
    return out.reindex(full_idx, fill_value=0.0)
