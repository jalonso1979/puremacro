"""GARCH-in-mean LP: include σ_t directly as a regressor."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .jorda import lp_hac
from ..garch.fit import garch11_fit


def lp_garch_in_mean(
    df: pd.DataFrame,
    y: str,
    x: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
):
    """Fit GARCH(1,1) on Δx, then run lp_hac with σ_t added to controls."""
    df = df.copy()
    eps = df[x].diff().dropna()
    garch = garch11_fit(eps)
    df["__sigma__"] = garch.sigma.reindex(df.index).ffill().bfill()
    extra = list(controls or []) + ["__sigma__"]
    return lp_hac(df, y=y, x=x, horizons=horizons, n_lags=n_lags,
                   controls=extra, alpha=alpha)


__all__ = ["lp_garch_in_mean"]
