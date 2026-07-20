"""Single-call orchestrator for IRF experiments.

Used by T5 templates so each cell stays short:

    out = run_experiment(wide, outcome="log_gfcf_real", proxy="unc_q",
                         method="panel_lp", horizons=range(0, 21))
    plot_irf_single(out["irf_point"], ...)

Dispatches to ``lp_ols_hac``, ``panel_lp``, or ``fit_var + cholesky_irf``.

Default-rescales results to "% (or pp) response per 1-SD shock" with
90% bands, the convention used in Bloom (2009) / Caldara et al. (2016).
Pass ``normalize="raw"`` to recover the unrescaled coefficients.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .data import slice_country
from .irf_scale import to_1sd_pct
from .lp_sm import lp_ols_hac
from .panel_lm import panel_lp
from .var_sm import cholesky_irf, fit_var

_VALID_METHODS = ("lp", "panel_lp", "var_cholesky")
_VALID_NORMALIZE = ("raw", "1sd_pct")


def _wide_x_sd(wide: pd.DataFrame, proxy: str, code: str | None = None) -> float:
    """Sample SD of *proxy* (one country) for use as the 1-SD shock size."""
    if proxy not in wide.columns:
        return float("nan")
    s = (
        wide[proxy]
        if code is None
        else wide.loc[wide.index.get_level_values("code") == code, proxy]
    )
    return float(s.dropna().std(ddof=0))


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

    Parameters
    ----------
    wide : pd.DataFrame
        Long-form panel pivoted to ``(code, date) × variable``.
    outcome, proxy : str
        Names of the response variable and the shock variable.
    method : {"lp", "panel_lp", "var_cholesky"}
    country, countries : str, list[str], optional
        Country selector. ``country`` is required for single-economy
        methods (``lp``, ``var_cholesky``); ``countries`` filters the
        panel for ``panel_lp``.
    horizons, n_lags, controls : standard LP / VAR settings.
    n_boot, seed : Monte-Carlo bootstrap settings (VAR Cholesky only).
    alpha : float, default ``0.10``
        Two-sided significance level for confidence bands. ``0.10`` →
        90% bands; ``0.05`` → 95%; ``0.32`` → ±1σ.
    normalize : {"1sd_pct", "raw"}, default ``"1sd_pct"``
        ``"1sd_pct"`` rescales to "% (pp) of y per 1-SD of x", with the
        sign of x's SD applied (LP / panel-LP) or with Cholesky's built-in
        1-SD shock interpretation (var_cholesky). ``"raw"`` returns the
        un-rescaled coefficients.

    Result keys
    -----------
    method      : str
    irf_point   : pd.DataFrame (var_cholesky) or pd.Series (LP / panel-LP)
    irf_lower   : same shape as irf_point
    irf_upper   : same shape as irf_point
    n_obs       : int
    n_countries : int (panel_lp only)
    shock_sd    : float (when ``normalize='1sd_pct'``); the SD applied
    y_units     : '%' or 'pp' (when ``normalize='1sd_pct'``)
    alpha       : float — significance level used for the bands
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
        sub = slice_country(wide, country)
        irf = lp_ols_hac(sub, y=outcome, x=proxy, horizons=horizons,
                         n_lags=n_lags, controls=controls, alpha=alpha)
        result = {
            "method": "lp",
            "irf_point": irf["beta"],
            "irf_lower": irf["lo"],
            "irf_upper": irf["hi"],
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
        irf = panel_lp(sub, y=outcome, x=proxy, horizons=horizons,
                       n_lags=n_lags, controls=controls or [], alpha=alpha)
        n_countries = sub.index.get_level_values("code").nunique()
        result = {
            "method": "panel_lp",
            "irf_point": irf["beta"],
            "irf_lower": irf["lo"],
            "irf_upper": irf["hi"],
            "n_obs": int(len(sub)),
            "n_countries": int(n_countries),
            "alpha": alpha,
        }
        if normalize == "1sd_pct":
            # Use the within-panel pooled SD (across all included countries).
            x_sd = float(sub[proxy].dropna().std(ddof=0))
            result = to_1sd_pct(result, x_sd=x_sd, outcome_name=outcome)
        return result

    # var_cholesky
    if country is None:
        raise ValueError("method='var_cholesky' requires country=")
    sub = slice_country(wide, country)
    sys_vars = [proxy, outcome] + (controls or [])
    Y = sub[sys_vars].dropna()
    res = fit_var(Y, maxlags=n_lags, ic="bic")
    irf = cholesky_irf(res, horizon=max(horizons), shock=proxy,
                       n_boot=n_boot, seed=seed, signif=alpha)
    result = {
        "method": "var_cholesky",
        "irf_point": irf["point"],
        "irf_lower": irf["lower"],
        "irf_upper": irf["upper"],
        "n_obs": int(len(Y)),
        "alpha": alpha,
    }
    if normalize == "1sd_pct":
        # Cholesky already encodes a 1-SD structural shock; we only
        # convert the response from log to percent (where applicable).
        # is_rate_outcome inspection is per-column: for a multi-column
        # response we apply the same scaling per column based on its name.
        from .irf_scale import is_rate_outcome
        for key in ("irf_point", "irf_lower", "irf_upper"):
            df = result[key].copy()
            for col in df.columns:
                if not is_rate_outcome(col):
                    df[col] = df[col] * 100.0
            result[key] = df
        result["shock_sd"] = None  # implicit in Cholesky construction
        result["y_units"] = "%/pp (per-column)"
    return result


__all__ = ["run_experiment"]
