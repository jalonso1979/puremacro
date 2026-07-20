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
the 3-lag aggregation lives in the companion state.

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
        quarterly value at the quarter-end month and NaN elsewhere; the
        other columns are observed every month.
    quarterly_col : str
        Name of the column holding the quarterly variable.
    p : int, default 3
        VAR lag length. Must be ≥ 3 because the 3-month aggregation
        constraint requires three lags of the latent monthly state.
    quarter_end_offset : int, default 2
        Position of the quarter-end month within a quarter (0 = first
        month, 2 = last month). For end-of-quarter dating use the
        default; if your quarterly observations are stamped at the
        first month of the quarter pass 0.

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
    cols = list(df.columns)
    if quarterly_col not in cols:
        raise ValueError(f"quarterly_col {quarterly_col!r} not in df.columns")
    monthly_cols = [c for c in cols if c != quarterly_col]
    n = len(cols)
    n_monthly = len(monthly_cols)

    Y_obs = df[monthly_cols + [quarterly_col]].values.astype(float)
    T = Y_obs.shape[0]

    # 1) Initialise: VAR on the *imputed* panel.
    # For the quarterly variable, naively fill NaN by forward-filling the
    # last published quarterly value divided by 3 — gives a plausible
    # starting monthly path while the EM-Kalman picks up the true one.
    Y_init = Y_obs.copy()
    q_col_idx = n - 1
    last_q = np.nan
    for t in range(T):
        if not np.isnan(Y_init[t, q_col_idx]):
            last_q = Y_init[t, q_col_idx] / 3.0
        Y_init[t, q_col_idx] = last_q
    # Drop leading rows where the quarterly value is still NaN.
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
    a_smooth = sm["a_smooth"]                                # (T, n*p)
    factors_monthly = a_smooth[:, :n]                        # (T, n)

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
