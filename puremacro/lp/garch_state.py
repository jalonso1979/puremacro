"""State-dependent LP where the regime indicator is GARCH(1,1) σ_t."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .state_dep import lp_state_dep
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
    threshold: float = 0.0,
    controls: Sequence[str] | None = None,
    alpha: float = 0.10,
):
    """If sigma_col is None, fit GARCH(1,1) on x's first differences and
    use the resulting σ_t as the regime state."""
    df = df.copy()
    if sigma_col is None:
        eps = df[x].diff().dropna()
        garch = garch11_fit(eps)
        df["__sigma__"] = garch.sigma.reindex(df.index).ffill().bfill()
        sigma_col = "__sigma__"
    return lp_state_dep(df, y=y, x=x, state=sigma_col, horizons=horizons,
                        n_lags=n_lags, transition=transition, gamma=gamma,
                        threshold=threshold, controls=controls, alpha=alpha)


__all__ = ["lp_garch_state"]
