"""Panel LP-IV: project x on z within FE, then panel LP with x_hat."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .panel import panel_lp
from ._panel_helpers import two_way_fe_within, as_panel_index
from ._common import resolve_lp_kwargs


def panel_lp_iv(
    df_wide: pd.DataFrame,
    y: str,
    x: str,
    z: str,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] = (),
    alpha: float = 0.10,
    entity_level: str = "code",
    time_level: str = "date",
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    unit_col: str | None = None,
    time_col: str | None = None,
) -> pd.DataFrame:
    """First stage: x_t ~ z_t + entity FE + time FE (within transform).
    Second stage: panel LP with the fitted x̂ in place of x.
    Returns LPResult with [h, beta, se, lo, hi, first_stage_f]."""
    horizons, n_lags, alpha = resolve_lp_kwargs(
        horizons, n_lags, alpha, lags=lags, horizon=horizon, ci=ci, name="panel_lp_iv")
    df_wide, entity_level, time_level = as_panel_index(
        df_wide, entity_level=entity_level, time_level=time_level,
        unit_col=unit_col, time_col=time_col, func="panel_lp_iv")
    df = df_wide.copy()
    df.index = df.index.set_names([entity_level, time_level])
    df = df.sort_index()

    # First stage in within-projection
    fs_sub = df[[x, z] + list(controls)].dropna().copy()
    fs_out = two_way_fe_within(fs_sub, y_col=x, x_cols=[z] + list(controls),
                                entity_level=entity_level, time_level=time_level)
    pi_z = float(fs_out["beta"][0])
    se_z = float(fs_out["se"][0])
    first_stage_f = (pi_z / se_z) ** 2 if se_z > 0 else np.nan

    # x_hat (using only the contemporaneous z component; entity/time FE
    # handled by the second stage)
    df["__xhat__"] = pi_z * df[z]
    out = panel_lp(df, y=y, x="__xhat__", horizons=horizons, n_lags=n_lags,
                   controls=list(controls), alpha=alpha,
                   entity_level=entity_level, time_level=time_level)
    out["first_stage_f"] = first_stage_f
    return out


__all__ = ["panel_lp_iv"]
