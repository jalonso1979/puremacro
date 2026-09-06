"""GARCH-in-mean LP: include σ_t directly as a regressor."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .jorda import lp_hac
from ._results import LPResult
from ..garch.fit import garch11_fit
from ._common import resolve_lp_kwargs


def lp_garch_in_mean(
    df: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
) -> LPResult:
    """Fit GARCH(1,1) on Δx, then run lp_hac with σ_t added to controls."""
    horizons, n_lags, alpha = resolve_lp_kwargs(
        horizons, n_lags, alpha, lags=lags, horizon=horizon, ci=ci, name="lp_garch_in_mean")
    df = df.copy()
    eps = df[x].diff().dropna()
    garch = garch11_fit(eps)
    df["__sigma__"] = garch.sigma.reindex(df.index).ffill().bfill()
    extra = list(controls or []) + ["__sigma__"]
    out = lp_hac(df, y=y, x=x, horizons=horizons, n_lags=n_lags,
                 controls=extra, alpha=alpha)
    res = LPResult(out)
    if "h" in res.columns:
        res.index = res["h"]
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "LP-garch-in-mean"
    res.ci_level = 1.0 - alpha
    return res


__all__ = ["lp_garch_in_mean"]
