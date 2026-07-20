"""Shared GARCH helpers for LP-GARCH modules.

These helpers are called by:
  - src/lp/lp_garch_state.py  (LP-GARCH ii: state-dependent)
  - src/lp/lp_garch_in_mean.py (LP-GARCH iii: GARCH-in-mean)

Design notes
------------
* ``fit_garch`` wraps ``arch.arch_model`` and returns a 4-tuple so callers
  receive everything they need without re-fitting.
* ``make_regime_indicator`` is stateless: it takes an already-fitted sigma
  series (typically the precomputed ``garch_sigma_<u>`` panel column) and
  applies a threshold rule, so the same function works whether the sigma
  comes from the panel or from a fresh ``fit_garch`` call.
* ``align_series_for_lp`` is a thin DataFrame alignment helper that drops
  any rows where *both* the shock and sigma are present but the index does
  not overlap.  It is intentionally minimal; LP-specific lag construction
  lives in lp_garch_state and lp_garch_in_mean.
"""

from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
# `arch.arch_model` is lazy-imported inside fit_garch below (Pyodide contract
# — see ARCHITECTURE.md). `ARCHModelResult` is referenced only in fit_garch's
# return annotation; thanks to `from __future__ import annotations` above,
# type hints are strings and never eagerly resolved at import time.
if False:  # pragma: no cover — type-checker visibility only
    from arch.univariate.base import ARCHModelResult  # noqa: F401


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_ArchDist = Literal[
    "normal", "gaussian", "t", "studentst", "skewstudent", "skewt",
    "ged", "generalized error",
]


def fit_garch(
    series: pd.Series,
    model: Literal["GARCH11", "GJR11"] = "GARCH11",
    *,
    dist: _ArchDist = "normal",
    rescale: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, ARCHModelResult]:
    """Fit a univariate GARCH(1,1) or GJR-GARCH(1,1) to *series*.

    Parameters
    ----------
    series:
        Time series to fit.  NaNs are dropped before fitting; the returned
        conditional volatility is re-indexed to the original index.
    model:
        ``'GARCH11'`` for the symmetric GARCH(1,1); ``'GJR11'`` for the
        GJR-GARCH(1,1) (asymmetric, Glosten-Jagannathan-Runkle).
    dist:
        Innovation distribution passed to ``arch_model`` (``'normal'`` by
        default; ``'t'`` is a useful robustness alternative).
    rescale:
        Whether to let ``arch`` rescale the series for numerical stability.
        Default ``True``.

    Returns
    -------
    cond_vol : pd.Series
        One-step-ahead conditional standard deviations, same index as
        *series* (NaN where *series* was NaN).
    innovations : pd.Series
        Fitted model residuals (mean-corrected series), same index.
    residuals : pd.Series
        Standardised residuals (innovations / cond_vol), same index.
    fit_result : ARCHModelResult
        Full ``arch`` result object for further inspection.
    """
    from arch import arch_model  # lazy: Pyodide contract

    clean = series.dropna()

    if model == "GARCH11":
        am = arch_model(clean, vol="GARCH", p=1, q=1, dist=dist, rescale=rescale)
    elif model == "GJR11":
        # arch >= 5 uses vol="GARCH" with o=1 for the asymmetric (GJR) term.
        am = arch_model(clean, vol="GARCH", p=1, o=1, q=1, dist=dist, rescale=rescale)
    else:
        raise ValueError(f"model must be 'GARCH11' or 'GJR11', got {model!r}")

    res = am.fit(disp="off", show_warning=False)

    # arch stores conditional variance; take sqrt for standard deviation.
    cond_var = res.conditional_volatility ** 2  # already sigma^2 in some versions
    # Defensive: conditional_volatility is sigma (not sigma^2) in arch >= 5.
    cond_sigma = res.conditional_volatility  # this is σ_t
    innovations_clean = res.resid

    # Re-index to the original series index (NaN for any missing dates).
    cond_vol = pd.Series(np.nan, index=series.index, name="cond_vol", dtype=float)
    innovations = pd.Series(np.nan, index=series.index, name="innovations", dtype=float)
    residuals = pd.Series(np.nan, index=series.index, name="residuals", dtype=float)

    cond_vol.loc[clean.index] = cond_sigma.values  # type: ignore[union-attr]  # arch stubs widen to ndarray|Any; runtime is Series
    innovations.loc[clean.index] = innovations_clean.values  # type: ignore[union-attr]  # arch stubs widen to ndarray|Any; runtime is Series
    residuals.loc[clean.index] = (innovations_clean / cond_sigma).values  # type: ignore[union-attr]  # arch stubs widen to ndarray|Any; runtime is Series

    return cond_vol, innovations, residuals, res


def make_regime_indicator(
    sigma: pd.Series,
    rule: Literal["median", "p75"] = "median",
) -> pd.Series:
    """Return a Boolean Series marking the high-volatility regime.

    Parameters
    ----------
    sigma:
        Conditional volatility series (any positive time series works).
    rule:
        ``'median'`` → high when ``σ_t > median(σ)``; approximately 50 %
        of observations are classified as high.
        ``'p75'`` → high when ``σ_t > p75(σ)``; approximately 25 % of
        observations are classified as high (sharper contrast, used as
        robustness check).

    Returns
    -------
    pd.Series of dtype bool, same index as *sigma*.
    """
    if rule == "median":
        threshold = sigma.median()
    elif rule == "p75":
        threshold = sigma.quantile(0.75)
    else:
        raise ValueError(f"rule must be 'median' or 'p75', got {rule!r}")

    return (sigma > threshold).rename(f"regime_high_{rule}")


def align_series_for_lp(
    df: pd.DataFrame,
    shock_col: str,
    sigma_col: str,
) -> pd.DataFrame:
    """Return a DataFrame with *shock_col* and *sigma_col* on a common index.

    Drops rows where either column is NaN so that downstream LP regressions
    receive a balanced sample.  Preserves all other columns in *df*.

    Parameters
    ----------
    df:
        Wide DataFrame containing at least *shock_col* and *sigma_col*.
    shock_col:
        Name of the shock column.
    sigma_col:
        Name of the conditional-volatility column.

    Returns
    -------
    pd.DataFrame with the same columns as *df*, restricted to rows where
    both *shock_col* and *sigma_col* are non-NaN.
    """
    mask = df[shock_col].notna() & df[sigma_col].notna()
    return df.loc[mask].copy()
