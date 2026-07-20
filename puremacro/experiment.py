"""Single-call orchestrator for IRF experiments (puremacro edition).

Dispatches to :func:`puremacro.lp.lp_hac`, :func:`puremacro.lp.panel_lp`,
or a Cholesky-identified VAR built from :mod:`puremacro.var` primitives.
Default-rescales results to "% (or pp) response per 1-SD shock" with
90% bands; pass ``normalize='raw'`` for unrescaled coefficients.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ._linalg import safe_cholesky
from .lp.jorda import lp_hac
from .lp.panel import panel_lp
from .scale import is_rate_outcome, to_1sd_pct
from .var.bootstrap import bootstrap_bands
from .var.estimate import estimate_var, select_lag_bic
from .var.identify.cholesky import cholesky_factor
from .var.irf import irf as compute_irf

_VALID_METHODS = ("lp", "panel_lp", "var_cholesky")
_VALID_NORMALIZE = ("raw", "1sd_pct")


def _slice_country(wide: pd.DataFrame, code: str) -> pd.DataFrame:
    """Return the per-country wide slice with a sorted DatetimeIndex."""
    if code not in wide.index.get_level_values("code"):
        raise KeyError(f"country {code!r} not in panel")
    sub = wide.xs(code, level="code").copy()
    sub.index = pd.DatetimeIndex(sub.index)
    return sub.sort_index()


def run_experiment(
    wide: pd.DataFrame,
    *,
    outcome: str,
    proxy: str,
    method: str,
    country: str | None = None,
    countries: list[str] | None = None,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 4,
    controls: list[str] | None = None,
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.10,
    normalize: str = "1sd_pct",
) -> dict:
    """Dispatch an IRF experiment and return a unified result dict.

    Parameters mirror :func:`src.teaching.experiment.run_experiment`. The
    panel must be a ``(code, date)`` MultiIndexed wide frame.

    Result keys
    -----------
    method      : str
    irf_point   : pd.DataFrame (var_cholesky) or pd.Series (LP / panel-LP)
    irf_lower   : same shape as irf_point
    irf_upper   : same shape as irf_point
    n_obs       : int
    n_countries : int (panel_lp only)
    shock_sd    : float (when ``normalize='1sd_pct'``)
    y_units     : '%' or 'pp' (when ``normalize='1sd_pct'``)
    alpha       : float
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {_VALID_METHODS}; got {method!r}"
        )
    if normalize not in _VALID_NORMALIZE:
        raise ValueError(
            f"normalize must be one of {_VALID_NORMALIZE}; got {normalize!r}"
        )

    horizons = list(horizons)

    if method == "lp":
        if country is None:
            raise ValueError("method='lp' requires country=")
        sub = _slice_country(wide, country)
        irf_df = lp_hac(sub, y=outcome, x=proxy, horizons=horizons,
                        n_lags=n_lags, controls=controls, alpha=alpha)
        irf_df = irf_df.set_index("h")
        result = {
            "method": "lp",
            "irf_point": irf_df["beta"],
            "irf_lower": irf_df["lo"],
            "irf_upper": irf_df["hi"],
            "n_obs": int(sub[[outcome, proxy]].dropna().shape[0]),
            "alpha": alpha,
        }
        if normalize == "1sd_pct":
            x_sd = float(sub[proxy].dropna().std(ddof=0))
            result = to_1sd_pct(result, x_sd=x_sd, outcome_name=outcome)
        return result

    if method == "panel_lp":
        sub = wide
        if countries is not None:
            sub = sub.loc[sub.index.get_level_values("code").isin(countries)]
        irf_df = panel_lp(sub, y=outcome, x=proxy, horizons=horizons,
                          n_lags=n_lags, controls=controls or [], alpha=alpha)
        irf_df = irf_df.set_index("h")
        n_countries = sub.index.get_level_values("code").nunique()
        result = {
            "method": "panel_lp",
            "irf_point": irf_df["beta"],
            "irf_lower": irf_df["lo"],
            "irf_upper": irf_df["hi"],
            "n_obs": int(len(sub)),
            "n_countries": int(n_countries),
            "alpha": alpha,
        }
        if normalize == "1sd_pct":
            x_sd = float(sub[proxy].dropna().std(ddof=0))
            result = to_1sd_pct(result, x_sd=x_sd, outcome_name=outcome)
        return result

    # var_cholesky
    if country is None:
        raise ValueError("method='var_cholesky' requires country=")
    sub = _slice_country(wide, country)
    sys_vars = [proxy, outcome] + (controls or [])
    Y = sub[sys_vars].dropna()
    Y_arr = Y.values
    # Match teaching's fit_var semantics: n_lags is the *maxlag* and BIC selects
    # within [1, n_lags]. Without this, h=0 Cholesky impacts diverge from the
    # teaching reference whenever BIC picks a smaller lag than n_lags.
    p = select_lag_bic(Y_arr, max_p=int(n_lags))
    H = int(max(horizons))

    A_list, _, Sigma, _, _ = estimate_var(Y_arr, p)
    B0 = cholesky_factor(Sigma)
    point_arr = compute_irf(A_list, B0, H)  # (H+1, n, n)

    bands = bootstrap_bands(
        Y_arr, p,
        identify_fn=lambda A_list, Sigma: safe_cholesky(
            Sigma, name="experiment var_cholesky bootstrap"),
        horizon=H, n_boot=n_boot, alpha=alpha, method="recursive",
        rng=np.random.default_rng(seed),
    )
    lower_arr = bands["lower"]
    upper_arr = bands["upper"]

    k = sys_vars.index(proxy)
    h_idx = pd.Index(range(H + 1), name="h")
    point_df = pd.DataFrame(point_arr[:, :, k], index=h_idx, columns=sys_vars)
    lower_df = pd.DataFrame(lower_arr[:, :, k], index=h_idx, columns=sys_vars)
    upper_df = pd.DataFrame(upper_arr[:, :, k], index=h_idx, columns=sys_vars)

    result = {
        "method": "var_cholesky",
        "irf_point": point_df,
        "irf_lower": lower_df,
        "irf_upper": upper_df,
        "n_obs": int(len(Y)),
        "alpha": alpha,
    }
    if normalize == "1sd_pct":
        for key in ("irf_point", "irf_lower", "irf_upper"):
            df = result[key].copy()
            for col in df.columns:
                if not is_rate_outcome(col):
                    df[col] = df[col] * 100.0
            result[key] = df
        result["shock_sd"] = None
        result["y_units"] = "%/pp (per-column)"
    return result


__all__ = ["run_experiment"]
