"""State-dependent LP where the regime indicator is GARCH(1,1) σ_t."""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from .state_dep import lp_state_dep
from ._common import resolve_lp_kwargs
from ._results import LPResult
from ..garch.fit import garch11_fit


def lp_garch_state(
    df: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    sigma_col: str | None = None,
    transition: str = "logistic",
    gamma: float = 3.0,
    threshold: float | None = None,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> LPResult:
    """State-dependent LP with conditional volatility as the state.

    If ``sigma_col`` is None, fit GARCH(1,1) on the first differences of
    ``x`` and use the resulting σ_t as the regime state; otherwise use the
    column ``sigma_col``. Regime weights follow :func:`lp_state_dep`:
    ``threshold`` is on the raw scale of the state (``None`` = sample mean
    of σ_t, i.e. high-volatility regime above average volatility) and
    ``gamma`` is the logistic speed in standard deviations of the state.

    Returns an :class:`LPResult` indexed by ``h`` with columns
    ``beta_H, se_H, lo_H, hi_H, beta_L, se_L, lo_L, hi_L``.
    """
    horizons, n_lags, alpha = resolve_lp_kwargs(
        horizons, n_lags, alpha, lags=lags, horizon=horizon, ci=ci, name="lp_garch_state")
    df = df.copy()
    if sigma_col is None:
        eps = df[x].diff().dropna()
        garch = garch11_fit(eps)
        df["__sigma__"] = garch.sigma.reindex(df.index).ffill().bfill()
        sigma_col = "__sigma__"
    out = lp_state_dep(df, y=y, x=x, state=sigma_col, horizons=horizons,
                       n_lags=n_lags, transition=transition, gamma=gamma,
                       threshold=threshold, controls=controls, alpha=alpha)
    res = LPResult(out)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "LP-garch-state"
    res.ci_level = 1.0 - alpha
    return res


__all__ = ["lp_garch_state"]
