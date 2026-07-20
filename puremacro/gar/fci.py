"""Financial Conditions Index — NFCI-style PCA aggregation.

Constructs a single-factor index from a wide panel of financial
indicators (rates, spreads, prices, vol, credit aggregates). The
estimator is the first principal component of the standardised panel,
sign-normalised so that **higher values mean tighter financial
conditions**: the convention used by the Chicago Fed NFCI, the
Bloomberg US Financial Conditions Index, and most academic GaR work.

Two routines:

  ``fci`` — fit on a complete-cases prefix; project the full panel.
  ``fci_rolling`` — rolling-window FCI, useful when loadings drift.

References
----------
Brave, S. and Butters, R.A. (2011). Monitoring financial stability:
    a financial conditions index approach. ChiFed Economic Perspectives.
Hatzius, J. et al. (2010). Financial conditions indexes: a fresh look
    after the financial crisis. NBER WP 16150.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Xs = (X - mu) / sd
    return Xs, mu, sd


def fci(
    df: pd.DataFrame,
    *,
    tightening_columns: Sequence[str] | None = None,
    handle_nan: str = "drop",
) -> dict:
    """Single-factor Financial Conditions Index.

    Parameters
    ----------
    df : DataFrame
        Panel of financial indicators, time-indexed. Columns where a
        higher value indicates tighter conditions (spreads, vol) and
        looser conditions (asset prices) are mixed; the function infers
        the sign-normalisation from the loading on the first PC unless
        ``tightening_columns`` is supplied.
    tightening_columns : optional sequence of column names known to
        load positively under "tighter conditions". Used to fix the
        sign of the first PC if its loading-sum on these is negative.
    handle_nan : ``"drop"`` (default) drops rows with any NaN before
        fitting; ``"zero"`` replaces NaN with 0 after standardisation
        (useful for ragged-edge panels but biases toward 0).

    Returns
    -------
    dict with
        ``index``    (T,)  — the FCI series (mean-zero, unit-variance),
        ``loadings`` (n,)  — first-PC loadings on each input,
        ``var_explained`` — share of standardised-data variance captured.
    """
    X = df.values.astype(float)
    Xs_raw, _, _ = _standardise(X)

    if handle_nan == "drop":
        complete = ~np.isnan(Xs_raw).any(axis=1)
        if complete.sum() < 5:
            raise ValueError(
                f"too few complete-cases rows ({int(complete.sum())}) "
                "for FCI fitting — drop sparse columns or use "
                'handle_nan="zero"'
            )
        Xs_fit = Xs_raw[complete]
    elif handle_nan == "zero":
        Xs_fit = np.where(np.isnan(Xs_raw), 0.0, Xs_raw)
    else:
        raise ValueError(f"unknown handle_nan {handle_nan!r}")

    # First PC via SVD on the standardised, mean-zero matrix.
    Xs_fit_centred = Xs_fit - Xs_fit.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xs_fit_centred, full_matrices=False)
    loadings = Vt[0]

    # Sign-normalise: tightening loadings should be positive.
    if tightening_columns:
        cols = list(df.columns)
        idx = [cols.index(c) for c in tightening_columns if c in cols]
        if idx and loadings[idx].sum() < 0:
            loadings = -loadings
    else:
        # Default convention: the largest absolute loading carries the
        # sign of "tightening" — flip if negative.
        if loadings[np.argmax(np.abs(loadings))] < 0:
            loadings = -loadings

    # Project the *full* panel (not just complete cases) onto the loading.
    if handle_nan == "zero":
        Xs_proj = np.where(np.isnan(Xs_raw), 0.0, Xs_raw)
    else:
        Xs_proj = np.where(np.isnan(Xs_raw), 0.0, Xs_raw)
    index = Xs_proj @ loadings

    # Re-standardise the index for interpretability.
    index = (index - index.mean()) / max(index.std(ddof=0), 1e-12)

    var_explained = float((s[0] ** 2) / (s ** 2).sum())
    return {
        "index":         pd.Series(index, index=df.index, name="FCI"),
        "loadings":      pd.Series(loadings, index=df.columns, name="loading"),
        "var_explained": var_explained,
    }


def fci_rolling(
    df: pd.DataFrame,
    window: int,
    *,
    tightening_columns: Sequence[str] | None = None,
) -> pd.Series:
    """Rolling-window Financial Conditions Index.

    Refits the first PC inside a moving window and reads off the FCI
    value at the window's right edge. Useful when the loading structure
    drifts (regime breaks, addition of new indicators); the trade-off
    is increased noise from each short-window PCA.

    Parameters
    ----------
    df : DataFrame
        Panel of financial indicators, time-indexed.
    window : int
        Rolling-window length in periods (must be ``>= 30``).
    tightening_columns : sequence of str, optional
        Columns known to load positively under "tighter conditions";
        forwarded to :func:`fci` for sign normalisation.

    Returns
    -------
    pandas.Series
        FCI value at each window's right edge, NaN before the first
        full window.
    """
    n = len(df)
    if window < 30:
        raise ValueError("window must be ≥ 30 for stable PCA")
    out = pd.Series(np.nan, index=df.index, name="FCI_rolling")
    for end in range(window, n + 1):
        sub = df.iloc[end - window: end]
        try:
            res = fci(sub, tightening_columns=tightening_columns,
                      handle_nan="zero")
        except Exception:
            continue
        out.iloc[end - 1] = float(res["index"].iloc[-1])
    return out


__all__ = ["fci", "fci_rolling"]
