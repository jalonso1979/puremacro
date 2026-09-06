"""Mixed-frequency VAR — Mariano-Murasawa (2003) style.

The classic nowcasting setup: ``n`` monthly indicators plus one
quarterly variable whose value at quarter-end is the 3-month average
of an *unobserved* monthly path. Cast as a Gaussian state-space
companion VAR(p), the Kalman smoother gives both (a) interpolated
monthly values for the quarterly variable and (b) a real-time
nowcast at the most recent month.

Specification
-------------
Stack monthly variables ``m_t = (m_{1,t}, …, m_{n-1,t}, m_n^*_t)`` where
``m_n^*`` is the *latent* monthly counterpart of the published quarterly
``y^Q``. Assume

    m_t = c + A_1 m_{t-1} + … + A_p m_{t-p} + ε_t,    ε ~ N(0, Σ).

The observation equation gives the monthly variables directly and
imposes the temporal-aggregation constraint at quarter-end months:

    y^Q_t = (1/3)(m_n^*_t + m_n^*_{t-1} + m_n^*_{t-2}),    t mod 3 == 0,

where intra-quarter months mark ``y^Q`` as NaN. ``p`` must be ≥ 3 so
the 3-lag aggregation lives in the companion state. Quarterly values
stamped at the first or middle month of their quarter
(``quarter_end_offset`` 0 or 1) are moved to the quarter-end month
internally, so the constraint always binds the three months of the
quarter the value belongs to.

References
----------
Mariano, R.S. and Murasawa, Y. (2003). A new coincident index of
    business cycles based on monthly and quarterly series. JAE 18(4).
Schorfheide, F. and Song, D. (2015). Real-time forecasting with a
    mixed-frequency VAR. JBES 33(3), 366-380.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..state_space import StateSpaceModel, kalman_smoother


def mf_var(
    df: pd.DataFrame,
    *,
    quarterly_col: str,
    p: int = 3,
    quarter_end_offset: int = 2,
    diffuse_scale: float = 1e6,
) -> dict:
    """Fit a mixed-frequency VAR with one quarterly variable.

    Parameters
    ----------
    df : DataFrame
        Monthly-indexed panel. ``quarterly_col`` carries the published
        quarterly value once per quarter (at the month given by
        ``quarter_end_offset``) and NaN elsewhere; the other columns are
        observed every month.
    quarterly_col : str
        Name of the column holding the quarterly variable.
    p : int, default 3
        VAR lag length. Must be ≥ 3 because the 3-month aggregation
        constraint requires three lags of the latent monthly state.
    quarter_end_offset : int, default 2
        Month of the quarter at which the quarterly value is stamped:
        ``2`` = last month of the quarter (end-of-quarter dating, the
        default), ``1`` = middle month, ``0`` = first month. A value
        stamped at month ``t`` with offset ``o`` is the average of months
        ``t - o, …, t - o + 2``; internally the observation is moved to
        ``t + (2 - o)`` so that the backward-looking aggregation row of the
        state-space form binds exactly those three months. When the last
        stamped value's quarter extends past the end of ``df`` the state
        is run ``2 - o`` months beyond the frame and truncated back.
    diffuse_scale : float, default 1e6
        Initial state-covariance scale (approximately diffuse prior).

    Returns
    -------
    dict with
        ``A`` (n·p, n·p), ``Q`` (n·p, n·p), ``Z`` (n, n·p), ``H`` (n, n)
        — state-space matrices,
        ``factors_monthly`` (T, n) — smoothed monthly path of every
            variable (including the latent monthly version of
            ``quarterly_col``),
        ``df_monthly`` — DataFrame view of ``factors_monthly``,
        ``df_filled`` — original df with the quarterly column
            interpolated to monthly via the smoother.
    """
    if p < 3:
        raise ValueError(
            "p must be >= 3 so the 3-month aggregation lives in the "
            "companion state."
        )
    if quarter_end_offset not in (0, 1, 2):
        raise ValueError(
            f"quarter_end_offset must be 0, 1 or 2; got {quarter_end_offset!r}"
        )
    cols = list(df.columns)
    if quarterly_col not in cols:
        raise ValueError(f"quarterly_col {quarterly_col!r} not in df.columns")
    monthly_cols = [c for c in cols if c != quarterly_col]
    n = len(cols)
    n_monthly = len(monthly_cols)

    Y_data = df[monthly_cols + [quarterly_col]].values.astype(float)
    T = Y_data.shape[0]
    q_col_idx = n - 1

    # Re-stamp the quarterly observations at the quarter-end month so the
    # backward-looking aggregation row (t, t-1, t-2) covers the quarter the
    # value belongs to. Values whose quarter runs past the frame are kept by
    # extending the observation matrix with all-NaN monthly rows.
    shift = 2 - int(quarter_end_offset)
    if shift > 0:
        Y_obs = np.full((T + shift, n), np.nan)
        Y_obs[:T, :n_monthly] = Y_data[:, :n_monthly]
        Y_obs[shift:, q_col_idx] = Y_data[:, q_col_idx]
    else:
        Y_obs = Y_data.copy()
    T_work = Y_obs.shape[0]

    # 1) Initialise: VAR on the *imputed* panel.
    # For the quarterly variable, naively fill NaN by forward-filling the
    # last published quarterly value divided by 3 — gives a plausible
    # starting monthly path while the EM-Kalman picks up the true one.
    Y_init = Y_obs.copy()
    last_q = np.nan
    for t in range(T_work):
        if not np.isnan(Y_init[t, q_col_idx]):
            last_q = Y_init[t, q_col_idx] / 3.0
        Y_init[t, q_col_idx] = last_q
    # Drop rows still carrying a NaN in any column (leading months before
    # the first quarterly observation, holes in monthly columns, and the
    # all-NaN extension rows).
    valid = ~np.isnan(Y_init).any(axis=1)
    if valid.sum() < p + 2:
        raise ValueError(
            "too few months with both monthly and quarterly observations "
            "after imputation; check quarterly_col coverage"
        )
    Y_fit = Y_init[valid]

    # OLS VAR(p) on Y_fit.
    Y_dep = Y_fit[p:]
    X = np.column_stack(
        [Y_fit[p - lag - 1: Y_fit.shape[0] - lag - 1] for lag in range(p)]
    )
    beta, *_ = np.linalg.lstsq(X, Y_dep, rcond=None)
    resid = Y_dep - X @ beta
    Sigma = (resid.T @ resid) / max(1, Y_fit.shape[0] - p)

    # 2) Companion-form state-space matrices.
    A = np.zeros((n * p, n * p))
    A[:n] = beta.T
    if p > 1:
        A[n:, :-n] = np.eye(n * (p - 1))
    Q = np.zeros((n * p, n * p))
    Q[:n, :n] = Sigma

    # Observation matrix.
    Z = np.zeros((n, n * p))
    # Monthly variables: Z[i, i] = 1 for i in 0..n-2.
    for i in range(n_monthly):
        Z[i, i] = 1.0
    # Quarterly variable: aggregation constraint, applied at every t.
    # At non-quarter-end months we mask y[t, q] as NaN, so Z's row for
    # the quarterly variable is harmless except at quarter ends.
    Z[q_col_idx, q_col_idx] = 1.0 / 3.0          # m_n^*_t
    Z[q_col_idx, q_col_idx + n] = 1.0 / 3.0      # m_n^*_{t-1}
    Z[q_col_idx, q_col_idx + 2 * n] = 1.0 / 3.0  # m_n^*_{t-2}

    # Tiny measurement noise (helps numerical stability).
    H = 1e-8 * np.eye(n)

    ssm = StateSpaceModel(T=A, Z=Z, Q=Q, H=H)

    # 3) Kalman smoother. NaN entries in Y_obs are honoured.
    sm = kalman_smoother(Y_obs, ssm, diffuse_scale=diffuse_scale)
    a_smooth = sm["a_smooth"]                                # (T_work, n*p)
    factors_monthly = a_smooth[:T, :n]                       # (T, n)

    # 4) Build a "filled" DataFrame: for the quarterly column we report
    # the smoothed monthly path of m_n^*; for monthly columns we leave
    # the original observations.
    df_monthly = pd.DataFrame(
        factors_monthly, index=df.index,
        columns=monthly_cols + [f"{quarterly_col}_monthly"],
    )
    df_filled = df.copy()
    df_filled[f"{quarterly_col}_monthly"] = factors_monthly[:, q_col_idx]

    return {
        "A":               A,
        "Q":               Q,
        "Z":               Z,
        "H":               H,
        "factors_monthly": factors_monthly,
        "df_monthly":      df_monthly,
        "df_filled":       df_filled,
    }


__all__ = ["mf_var"]
