"""Single-country Jordà (2005) local projection with Newey-West HAC SE.

Pure-numpy port of src/lp/lp_jorda.py — replaces statsmodels.OLS w/ HAC.
Uses puremacro.inference._ols_helpers.ols_hac under the hood.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..inference._ols_helpers import ols_hac
from ._common import (
    _DEFAULT_ALPHA_10,
    _DEFAULT_HORIZONS_21,
    _DEFAULT_N_LAGS_2,
    check_regular_index,
    format_sample_bounds,
    resolve_lp_kwargs,
    validate_lp_columns_finite,
)


def lp_hac(
    df: pd.DataFrame | pd.Series | np.ndarray,
    y: str | pd.Series | np.ndarray | None = None,
    x: str | pd.Series | np.ndarray | None = None,
    horizons: Iterable[int] = _DEFAULT_HORIZONS_21,
    n_lags: int = _DEFAULT_N_LAGS_2,
    controls: Sequence[str] | pd.DataFrame | pd.Series | np.ndarray | None = None,
    alpha: float = _DEFAULT_ALPHA_10,
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    allow_index_gaps: bool = False,
) -> pd.DataFrame:
    """Estimate local projection with Newey-West HAC standard errors.

    Accepts either DataFrame input `lp_hac(df, y='y_col', x='x_col', ...)`
    or 1D array/series input `lp_hac(y_series, shock_series, horizons=..., lags=...)`.

    At horizon ``h`` the outcome is ``y_{t+h} - y_{t-1}``, regressed on a
    constant, ``x_t``, ``n_lags`` lags of ``x`` and ``y`` (and of each
    control) and the contemporaneous controls; ``beta`` is the ``x_t``
    coefficient.

    For dated ``DatetimeIndex`` / ``PeriodIndex`` inputs, duplicate,
    non-increasing (unsorted/reversed), and gapped periods raise
    ``ValueError`` by default. Set ``allow_index_gaps=True`` only when
    periods were intentionally removed after pre-constructing lag/horizon
    columns. Plain ``RangeIndex`` and ``ndarray`` inputs use positional row
    semantics.

    Standard errors are Newey-West (1987): Bartlett kernel with
    truncation lag ``h + 1`` at horizon ``h`` (weights ``1 - l/(h + 2)``),
    no small-sample correction. The truncation lag is a rule of thumb: the
    ``h``-step-ahead residual is serially correlated up to order ``h``, so
    the kernel must reach at least that far. No reference is claimed for
    it. (Earlier releases credited it to Plagborg-Møller and Wolf 2021,
    whose paper proves that LPs and VARs estimate the same impulse
    responses and makes no HAC recommendation.) For intervals that need no
    HAC correction, use :func:`puremacro.lp.la_lp` (Montiel Olea and
    Plagborg-Møller 2021).
    Bands at level (1 - alpha) — default 90 %.

    Returns
    -------
    LPResult (subclass of pd.DataFrame) with columns [h, beta, se, t, lo, hi] indexed by h,
    carrying reconstructible specification metadata in ``res.attrs['specification']``
    and ``res.metadata``.
    """
    horizons_list, n_lags_int, alpha_val = resolve_lp_kwargs(
        horizons,
        n_lags,
        alpha,
        lags=lags,
        horizon=horizon,
        ci=ci,
        name="lp_hac",
        strict_aliases=True,
    )
    if not horizons_list:
        raise ValueError("lp_hac: horizons cannot be empty.")
    if any(h < 0 for h in horizons_list):
        raise ValueError(
            f"lp_hac: horizons must be non-negative integers >= 0; got {horizons_list!r}"
        )
    z_crit = norm.ppf(1 - alpha_val / 2)

    if isinstance(df, pd.DataFrame):
        if not (isinstance(y, str) and isinstance(x, str)):
            raise ValueError(
                "lp_hac: DataFrame input requires string column names y='...' and x='...'."
            )
        y_col = y
        x_col = x
        if controls is not None and not isinstance(controls, (list, tuple)):
            ctl = list(controls)  # type: ignore[arg-type]
        else:
            ctl = [str(c) for c in (controls or [])]
        work_df = df
    else:
        if y is not None and x is not None:
            raise ValueError(
                "lp_hac: 1-D array/Series mode expects lp_hac(y_series, x_series, ...) "
                "or lp_hac(y_series, x=x_series, ...); do not pass both y and x."
            )
        x_input = y if y is not None else x
        if x_input is None:
            raise ValueError(
                "lp_hac: 1-D array/Series mode requires both y and x series."
            )
        series_index: pd.Index | None = None
        if isinstance(df, pd.Series) and isinstance(x_input, pd.Series):
            if len(df) != len(x_input) or not df.index.equals(x_input.index):
                raise ValueError(
                    "lp_hac: Series inputs y and x have mismatched lengths or indexes; "
                    "align them explicitly before estimation."
                )
            series_index = df.index
        elif isinstance(df, pd.Series):
            series_index = df.index
        elif isinstance(x_input, pd.Series):
            series_index = x_input.index

        try:
            y_vals = np.asarray(df, dtype=float)
            x_vals = np.asarray(x_input, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("lp_hac: y and x inputs must be numeric.") from exc

        if y_vals.ndim != 1 or x_vals.ndim != 1:
            raise ValueError(
                f"lp_hac: 1-D array/Series inputs must be 1-dimensional; "
                f"got shapes y={y_vals.shape}, x={x_vals.shape}."
            )
        if len(y_vals) != len(x_vals):
            raise ValueError(
                f"lp_hac: y and x must have the same length; got {len(y_vals)} and {len(x_vals)}."
            )

        data: dict[str, np.ndarray] = {"y": y_vals, "x": x_vals}
        ctl_names: list[str] = []
        if controls is not None:
            if (
                isinstance(controls, (pd.Series, pd.DataFrame))
                and series_index is not None
                and not controls.index.equals(series_index)
            ):
                raise ValueError(
                    "lp_hac: controls index does not match y/x Series index."
                )
            try:
                C = np.asarray(controls, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("lp_hac: controls must be numeric.") from exc
            if C.ndim == 1:
                if len(C) != len(y_vals):
                    raise ValueError(
                        f"lp_hac: controls length ({len(C)}) must match y length ({len(y_vals)})."
                    )
                data["c0"] = C
                ctl_names.append("c0")
            elif C.ndim == 2:
                if C.shape[0] != len(y_vals):
                    raise ValueError(
                        f"lp_hac: controls row count ({C.shape[0]}) must match y length ({len(y_vals)})."
                    )
                for j in range(C.shape[1]):
                    cname = f"c{j}"
                    data[cname] = C[:, j]
                    ctl_names.append(cname)
            else:
                raise ValueError(
                    f"lp_hac: controls array must be 1-D or 2-D; got {C.ndim}-D."
                )
        work_df = pd.DataFrame(data, index=series_index)
        y_col = "y"
        x_col = "x"
        ctl = ctl_names

    check_regular_index(
        work_df.index, "lp_hac", cumulative=False, allow_gaps=bool(allow_index_gaps)
    )
    validate_lp_columns_finite(work_df, [y_col, x_col] + ctl, "lp_hac")

    rows: list[dict[str, Any]] = []
    n_obs_by_h: dict[int, int] = {}
    start_by_h: dict[int, str | int | None] = {}
    end_by_h: dict[int, str | int | None] = {}
    bw_by_h: dict[int, int] = {}

    for h in horizons_list:
        bw_by_h[h] = int(h + 1)
        sub = work_df[[y_col, x_col] + ctl].copy()
        sub["dy_h"] = sub[y_col].shift(-h) - sub[y_col].shift(1)
        for lag in range(1, n_lags_int + 1):
            sub[f"{x_col}_L{lag}"] = sub[x_col].shift(lag)
            sub[f"{y_col}_L{lag}"] = sub[y_col].shift(lag)
            for c in ctl:
                sub[f"{c}_L{lag}"] = sub[c].shift(lag)
        sub = sub.dropna()
        n = int(len(sub))
        n_obs_by_h[h] = n
        s_lo, s_hi = format_sample_bounds(sub.index)
        start_by_h[h] = s_lo
        end_by_h[h] = s_hi
        if sub.empty:
            rows.append(
                {
                    "h": h,
                    "beta": np.nan,
                    "se": np.nan,
                    "t": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                }
            )
            continue
        regressors = [np.ones(n), sub[x_col].values]
        for lag in range(1, n_lags_int + 1):
            regressors.append(sub[f"{x_col}_L{lag}"].values)
            regressors.append(sub[f"{y_col}_L{lag}"].values)
            for c in ctl:
                regressors.append(sub[f"{c}_L{lag}"].values)
        for c in ctl:
            regressors.append(sub[c].values)
        X = np.column_stack(regressors)
        out = ols_hac(sub["dy_h"].values, X, lags=h + 1)
        beta_h = float(out["beta"][1])  # x_t coefficient (col 1 after const)
        se_h = float(out["se"][1])
        rows.append(
            {
                "h": h,
                "beta": beta_h,
                "se": se_h,
                "t": beta_h / se_h if se_h > 0 else np.nan,
                "lo": beta_h - z_crit * se_h,
                "hi": beta_h + z_crit * se_h,
            }
        )

    from ._results import LPResult

    res = LPResult(rows)
    res.index = res["h"]
    res.y_name = str(y_col)
    res.x_name = str(x_col)
    res.method = "LP-HAC"
    res.ci_level = 1.0 - alpha_val
    res._set_specification(
        {
            "estimator": "lp_hac",
            "method": "LP-HAC",
            "y_name": str(y_col),
            "x_name": str(x_col),
            "controls": list(ctl),
            "n_lags": int(n_lags_int),
            "horizons": list(horizons_list),
            "estimand": "diff_from_t_minus_1",
            "outcome_formula": "y_{t+h} - y_{t-1}",
            "cov_type": "HAC",
            "kernel": "bartlett",
            "bandwidth_rule": "h + 1",
            "bandwidth_by_horizon": bw_by_h,
            "small_sample_correction": False,
            "cov_summary": "HAC Bartlett(h+1)",
            "alpha": float(alpha_val),
            "ci_level": float(1.0 - alpha_val),
            "interval_method": "normal_wald",
            "index_type": type(work_df.index).__name__,
            "allow_index_gaps": bool(allow_index_gaps),
            "n_obs_by_horizon": n_obs_by_h,
            "sample_start_by_horizon": start_by_h,
            "sample_end_by_horizon": end_by_h,
        }
    )
    return res


__all__ = ["lp_hac"]
