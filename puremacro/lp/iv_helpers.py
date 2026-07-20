"""Instrument-cleanup helpers for LP-IV.

- romer_romer_clean(narrative, controls): orthogonalise a narrative
  shock against pre-determined controls (Romer-Romer 2004).
- gertler_karadi_surprise(window_returns, market_proxy): extract a
  pure monetary-surprise instrument from a high-frequency window
  (Gertler-Karadi 2015 / Nakamura-Steinsson 2018).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def romer_romer_clean(narrative_series, controls_df) -> pd.Series:
    """Project a narrative-shock series onto pre-determined controls and
    return the residuals — the "cleaned" instrument that's orthogonal
    to the controls.

    Mirrors the Romer-Romer (2004) construction of the FFR shock series:
    regress the candidate shock on macro controls available at the time
    the FOMC decision was made, take the residuals.

    Parameters
    ----------
    narrative_series : pd.Series
        Raw narrative shocks (indexed by date).
    controls_df : pd.DataFrame
        Predetermined control variables (same index).

    Returns
    -------
    pd.Series of residuals, aligned to narrative_series's index.
    """
    s = pd.Series(narrative_series).copy()
    X = pd.DataFrame(controls_df).copy()
    df = pd.concat([s.rename("__y__"), X], axis=1).dropna()
    if df.empty:
        return pd.Series(np.nan, index=s.index)
    y = df["__y__"].values
    Xmat = np.column_stack([np.ones(len(df)), df.drop(columns="__y__").values])
    beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
    resid = y - Xmat @ beta
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[df.index] = resid
    return out


def gertler_karadi_surprise(
    window_returns: pd.Series,
    market_proxy: pd.Series | None = None,
) -> pd.Series:
    """Extract a monetary-policy surprise instrument from high-frequency
    window returns around FOMC announcements.

    If `market_proxy` is supplied (e.g., S&P futures returns over the
    same window), we project window_returns onto the market proxy and
    return the residuals — the component orthogonal to broad-market
    movements (Nakamura-Steinsson 2018 style cleaning of the GK shock).

    If `market_proxy` is None, return window_returns as-is (raw
    high-frequency surprise).

    Parameters
    ----------
    window_returns : pd.Series
        e.g., change in FFR futures across the announcement window.
    market_proxy : pd.Series, optional
        Same-window broad-market return.

    Returns
    -------
    pd.Series of cleaned surprises, aligned to window_returns's index.
    """
    s = pd.Series(window_returns).copy()
    if market_proxy is None:
        return s
    m = pd.Series(market_proxy)
    df = pd.concat([s.rename("__y__"), m.rename("__m__")], axis=1).dropna()
    if df.empty:
        return pd.Series(np.nan, index=s.index)
    Xmat = np.column_stack([np.ones(len(df)), df["__m__"].values])
    beta, *_ = np.linalg.lstsq(Xmat, df["__y__"].values, rcond=None)
    resid = df["__y__"].values - Xmat @ beta
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[df.index] = resid
    return out


__all__ = ["romer_romer_clean", "gertler_karadi_surprise"]
