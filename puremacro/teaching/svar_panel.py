"""Per-country SVAR + cross-country IRF distribution helpers.

Used by the new T1/T2 sections that estimate a country-by-country SVAR on
a small system, then summarise the response distribution across countries
via mean-group ± cross-country fan bands.

Identification schemes wrapped:
  - "cholesky" — recursive ordering as listed in `var_names` (puremacro.var.identify.cholesky)
  - "sign"     — Rubio-Ramírez-Waggoner-Zha sign restrictions (puremacro.var.identify.sign)
  - "bq"       — Blanchard-Quah long-run zero (puremacro.var.identify.bq)
  - "maxshare" — Faust/Uhlig FEVD-max-share (puremacro.var.identify.maxshare)

Each scheme returns IRF arrays of shape (H+1, n, n) (point, lo, hi).

The only structure imposed across schemes: shock index `s` and response
index `r` correspond to the variable order in `var_names`. Shock-label
interpretation depends on the scheme — caller is responsible for
labelling.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# Make sure puremacro is importable when the caller hasn't already done so.
_PM = Path(__file__).resolve().parents[2] / "puremacro"
if _PM.exists() and str(_PM) not in sys.path:
    sys.path.insert(0, str(_PM))

try:
    from puremacro.plotting.bw_style import bw_colors
except ImportError:
    def bw_colors(n: int = 7) -> list[str]:
        shades = ["0.05", "0.40", "0.65", "0.25", "0.55", "0.10", "0.75"]
        return shades[:n] if n <= len(shades) else (shades * ((n // len(shades)) + 1))[:n]

# Back-compat re-export: peak_summary now lives in puremacro.var.peak.
from puremacro.var.peak import peak_summary  # noqa: F401  (re-export)


def prep_country_system(
    panel_wide: pd.DataFrame,
    code: str,
    *,
    vars_levels: Sequence[str] = (),
    vars_diff: Sequence[str] = (),
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build the (T, n) `Y` array for one country's SVAR.

    Filters panel_wide to `code`, drops rows where any required variable
    is missing, first-differences the `vars_diff` variables (the I(1)
    block), and keeps `vars_levels` untouched (the I(0) block, e.g.
    pre-residualised innovations like `unc_innov_q`).

    Returns (df, Y_array, var_names) where var_names is `[*vars_levels, *vars_diff]`
    in that order — i.e. levels precede differenced variables in the
    SVAR system. If the country panel has fewer than 30 observations
    after dropna+diff, returns an empty Y array.
    """
    var_names = list(vars_levels) + list(vars_diff)
    if not var_names:
        raise ValueError("must specify at least one of vars_levels / vars_diff")

    if code not in panel_wide.index.get_level_values(0):
        return pd.DataFrame(), np.empty((0, len(var_names))), var_names

    sub = panel_wide.xs(code, level=0)[var_names].sort_index()
    if start is not None:
        sub = sub.loc[pd.Timestamp(start):]
    if end is not None:
        sub = sub.loc[:pd.Timestamp(end)]
    sub = sub.dropna()   # drop rows where any required col is NaN

    if len(sub) < 6:
        return sub, np.empty((0, len(var_names))), var_names

    Y = sub.copy()
    for c in vars_diff:
        Y[c] = Y[c].diff()
    Y = Y.dropna()

    return Y, Y.values.astype(float), var_names


def fit_svar_country(
    Y: np.ndarray,
    *,
    p: int,
    horizon: int,
    scheme: str,
    n_boot: int = 500,
    ci: float = 0.9,
    seed: int = 0,
    **scheme_kwargs,
) -> dict | None:
    """Fit a single-country SVAR with the requested identification.

    Returns a dict ``{"point", "lo", "hi", "scheme", "n_obs"}`` or None
    if estimation failed (insufficient data, singular Σ, etc.). For
    ``scheme == "sign"`` the dict additionally carries
    ``"n_accepted"`` (admissible rotations across ``n_draws`` prior
    samples) and ``"n_draws"`` (the requested draw count).
    """
    if Y.shape[0] < p + 6:
        return None

    try:
        res: Any
        if scheme == "cholesky":
            from puremacro.var.identify.cholesky import cholesky_svar
            res = cholesky_svar(
                Y, p=p, horizon=horizon, n_boot=n_boot, ci=ci, seed=seed,
                ordering=scheme_kwargs.get("ordering"),
            )
            # Accept both legacy tuple (point, lo, hi) and new CholeskySVARResult dataclass.
            if hasattr(res, "irf_point"):
                point, lo, hi = res.irf_point, res.irf_lower, res.irf_upper
            else:
                point, lo, hi = res[0], res[1], res[2]
        elif scheme == "sign":
            from puremacro.var.identify.sign import sign_restriction_svar
            restrictions = scheme_kwargs["restrictions"]
            n_draws = scheme_kwargs.get("n_draws", 2000)
            res = sign_restriction_svar(
                Y, p=p, horizon=horizon, restrictions=restrictions,
                n_draws=n_draws, ci=ci, seed=seed,
            )
            # Accept dataclass or legacy tuple; sign uses irf_median as point estimate.
            if hasattr(res, "irf_point"):
                point, lo, hi = res.irf_point, res.irf_lower, res.irf_upper
                _n_accepted = res.n_accepted
                _n_draws    = res.n_draws
            elif hasattr(res, "irf_median"):
                point, lo, hi = res.irf_median, res.irf_lower, res.irf_upper
                _n_accepted = res.n_accepted
                _n_draws    = res.n_draws
            else:
                if res[0] is None:
                    return None
                point, lo, hi = res[0], res[1], res[2]
                _n_accepted, _n_draws = None, n_draws
        elif scheme == "bq":
            from puremacro.var.identify.bq import bq_svar
            permanent_var_idx = scheme_kwargs.get("permanent_var_idx", 0)
            res = bq_svar(
                Y, p=p, horizon=horizon,
                permanent_var_idx=permanent_var_idx,
                n_boot=n_boot, ci=ci, seed=seed,
            )
            # Accept both legacy tuple (point, lo, hi) and new BQSVARResult dataclass.
            if hasattr(res, "irf_point"):
                point, lo, hi = res.irf_point, res.irf_lower, res.irf_upper
            else:
                point, lo, hi = res[0], res[1], res[2]
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
    except Exception:
        return None

    out_dict: dict = {"point": point, "lo": lo, "hi": hi, "scheme": scheme, "n_obs": Y.shape[0]}
    if scheme == "sign":
        out_dict["n_accepted"] = _n_accepted
        out_dict["n_draws"]    = _n_draws
    return out_dict


def fit_per_country(
    panel_wide: pd.DataFrame,
    codes: Iterable[str],
    *,
    vars_levels: Sequence[str] = (),
    vars_diff: Sequence[str] = (),
    p: int = 2,
    horizon: int = 20,
    scheme: str = "cholesky",
    start: str | None = None,
    end: str | None = None,
    min_obs: int = 24,
    **scheme_kwargs,
) -> dict[str, dict]:
    """Run :func:`fit_svar_country` for each `code`. Skips countries whose
    cleaned sample is shorter than `min_obs`. Returns ``{code: result}``.
    """
    out: dict[str, dict] = {}
    for code in codes:
        _, Y, _ = prep_country_system(
            panel_wide, code,
            vars_levels=vars_levels, vars_diff=vars_diff,
            start=start, end=end,
        )
        if Y.shape[0] < min_obs:
            continue
        res = fit_svar_country(
            Y, p=p, horizon=horizon, scheme=scheme,
            n_boot=scheme_kwargs.pop("n_boot", 300),
            ci=scheme_kwargs.pop("ci", 0.9),
            seed=hash((code, scheme)) & 0xFFFFFFFF,
            **scheme_kwargs,
        )
        if res is not None:
            res["code"] = code
            out[code] = res
    return out


def cross_country_irf(
    results: dict[str, dict],
    *,
    shock_idx: int,
    response_idx: int,
    scale: float = 100.0,
    cumulative: bool = False,
) -> pd.DataFrame:
    """Stack per-country IRFs to (shock, response) and return per-horizon
    cross-country summary (mean, median, sd, q10, q90, n).

    `cumulative=True` returns the cumulative-sum IRF along the horizon
    axis — useful when the variable of interest is a first-difference
    and the structural concept is the level (e.g. dlog_y → log_y).
    """
    if not results:
        return pd.DataFrame()

    H = next(iter(results.values()))["point"].shape[0]
    arr = np.full((len(results), H), np.nan)
    for i, (code, r) in enumerate(results.items()):
        irf_h = r["point"][:, response_idx, shock_idx]
        if cumulative:
            irf_h = np.cumsum(irf_h)
        arr[i] = irf_h * scale

    df = pd.DataFrame({
        "h": np.arange(H),
        "mean":   np.nanmean(arr, axis=0),
        "median": np.nanmedian(arr, axis=0),
        "sd":     np.nanstd(arr, axis=0, ddof=1),
        "q10":    np.nanpercentile(arr, 10, axis=0),
        "q90":    np.nanpercentile(arr, 90, axis=0),
        "n":      np.sum(~np.isnan(arr), axis=0),
    })
    return df


_METRIC_BOUND_SUFFIX = {"peak": ("peak_lo", "peak_hi"),
                        "accum_h16": ("accum_h16_lo", "accum_h16_hi")}


def lp_short_vs_long_forest(
    ax,
    lp_results,
    *,
    h_short: int = 4,
    h_long: int = 16,
    scale: float = 100.0,
    sort_by: str = "long",
    show_zero: bool = True,
    short_style: dict | None = None,
    long_style: dict | None = None,
    xlabel: str = "",
    title: str = "",
):
    """Forest plot comparing short-run (h_short) vs long-run (h_long) LP IRFs.

    Parameters
    ----------
    ax : matplotlib axis
    lp_results : dict[str, pd.DataFrame] OR pd.DataFrame
        Either a dict mapping a label (e.g. an outcome name or country code)
        to a per-horizon LP DataFrame, or a single DataFrame indexed by
        horizon. Each DataFrame must contain either:
          * columns ``beta``, ``lo``, ``hi`` (the convention used by
            `lp_hac` / `panel_lp` / `mean_group_panel_lp`), OR
          * columns ``irf_point``, ``irf_lower``, ``irf_upper`` (the
            convention used by `run_experiment`).
        The horizon is taken from ``h`` column or from the index if
        the DataFrame is horizon-indexed.
    h_short, h_long : int
        Two horizons to compare. Long run defaults to h=16 quarters.
    scale : float
        Multiplier on β / lo / hi (e.g. 100 to convert log-units to %).
    sort_by : "short" | "long" | "diff"
        Sort labels by the chosen quantity at that horizon (or by
        `long − short` if "diff").

    Renders one row per label with two markers (short = solid black
    circle, long = open gray square by default), HAC CIs as horizontal
    error bars, and a faint horizontal divider for the mean-group ± SD
    summary at the top.
    """
    if isinstance(lp_results, pd.DataFrame):
        lp_results = {"": lp_results}

    # Pick the right column convention per DataFrame.
    def _cols(df):
        if "beta" in df.columns:
            return "beta", "lo", "hi"
        if "irf_point" in df.columns:
            return "irf_point", "irf_lower", "irf_upper"
        raise ValueError("LP DataFrame must have beta/lo/hi or irf_point/irf_lower/irf_upper columns")

    def _fetch(df, h):
        b, lo, hi = _cols(df)
        # Index by horizon column if present, else assume horizon == row index.
        if "h" in df.columns:
            row = df[df["h"] == h]
            if row.empty: return (np.nan, np.nan, np.nan)
            r = row.iloc[0]
            return float(r[b]) * scale, float(r[lo]) * scale, float(r[hi]) * scale
        if h in df.index:
            r = df.loc[h]
            return float(r[b]) * scale, float(r[lo]) * scale, float(r[hi]) * scale
        # Else positional
        if h < len(df):
            r = df.iloc[h]
            return float(r[b]) * scale, float(r[lo]) * scale, float(r[hi]) * scale
        return (np.nan, np.nan, np.nan)

    rows = []
    for lbl, df in lp_results.items():
        b_s, lo_s, hi_s = _fetch(df, h_short)
        b_l, lo_l, hi_l = _fetch(df, h_long)
        rows.append({
            "label": lbl,
            "short": b_s, "short_lo": lo_s, "short_hi": hi_s,
            "long":  b_l, "long_lo": lo_l, "long_hi": hi_l,
            "diff": b_l - b_s if (np.isfinite(b_l) and np.isfinite(b_s)) else np.nan,
        })
    sl = pd.DataFrame(rows)
    if sl.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return sl

    sort_key = {"short": "short", "long": "long", "diff": "diff"}[sort_by]
    sl = sl.sort_values(sort_key).reset_index(drop=True)

    short_style = short_style or {"marker": "o", "color": "0.10",
                                   "edgecolor": "black", "label": f"h={h_short}"}
    long_style  = long_style  or {"marker": "s", "color": "0.55",
                                   "edgecolor": "black", "label": f"h={h_long}"}

    ys = np.arange(len(sl))
    # Short markers slightly above; long below
    for offset, sty, key in [(+0.18, short_style, "short"),
                              (-0.18, long_style, "long")]:
        x = sl[key].values
        lo = sl[f"{key}_lo"].values
        hi = sl[f"{key}_hi"].values
        err_lo = np.maximum(0.0, x - lo)
        err_hi = np.maximum(0.0, hi - x)
        ax.errorbar(x, ys + offset, xerr=[err_lo, err_hi],
                    fmt=sty["marker"], color=sty["color"],
                    ecolor=sty["color"], elinewidth=0.7, capsize=2, ms=5,
                    markeredgecolor=sty.get("edgecolor", "black"),
                    markeredgewidth=0.4, label=sty["label"])

    # Mean-group ± SD summary at top
    yy = len(sl) + 1.0
    for offset, sty, key in [(+0.18, short_style, "short"),
                              (-0.18, long_style, "long")]:
        vals = sl[key].dropna()
        if vals.empty: continue
        mu, sd = float(vals.mean()), float(vals.std(ddof=1))
        ax.errorbar([mu], [yy + offset], xerr=[[sd], [sd]],
                    fmt=sty["marker"], color=sty["color"],
                    ecolor=sty["color"], elinewidth=1.2, capsize=3, ms=7,
                    markeredgecolor=sty.get("edgecolor", "black"),
                    markeredgewidth=0.6)

    if show_zero:
        ax.axvline(0, color="0.5", lw=0.5, ls=":")
    ax.axhline(len(sl) + 0.5, color="0.7", lw=0.4, ls="-")
    ax.set_yticks(list(ys) + [yy])
    ax.set_yticklabels(list(sl["label"]) + ["Mean-group ± SD"], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    # Dedup legend
    h_, l_ = ax.get_legend_handles_labels()
    seen = {}
    for hi_, li_ in zip(h_, l_):
        if li_ not in seen: seen[li_] = hi_
    ax.legend(seen.values(), seen.keys(), loc="best", fontsize=7, frameon=False)
    return sl


def forest_plot_id_compare(
    ax,
    peaks_by_scheme,
    *,
    metric: str = "peak",
    sort_scheme: str | None = None,
    show_mean_group: bool = True,
    show_ci: bool = True,
    clip_outliers: bool = True,
    outlier_factor: float = 3.0,
    schemes_styles: dict | None = None,
    xlabel: str = "",
    title: str = "",
    max_countries: int | None = None,
):
    """Forest plot of country-level peak (or accumulated) IRFs across identification schemes.

    Parameters
    ----------
    show_ci
        If True (default) and the metric has matching ``*_lo``/``*_hi`` columns,
        draw horizontal error bars per country. Otherwise points only.
    clip_outliers
        If True, detect outliers via ``IQR × outlier_factor`` on the pooled
        distribution of point estimates across countries × schemes, clip the
        x-axis to the inlier range with a 10% margin, and annotate any
        out-of-range value as ``"→ {true value}"`` at the clipped edge.
    outlier_factor
        Multiplier on the IQR for outlier detection. Default 3.0 — only
        truly extreme observations get flagged.
    """
    if not peaks_by_scheme:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return

    schemes = list(peaks_by_scheme.keys())
    sort_scheme = sort_scheme or schemes[0]

    sort_df = peaks_by_scheme[sort_scheme]
    if metric not in sort_df.columns:
        raise ValueError(f"metric {metric!r} not in peak_summary columns; "
                          f"have {list(sort_df.columns)}")

    # CI columns
    lo_col, hi_col = _METRIC_BOUND_SUFFIX.get(metric, (None, None))
    has_ci = (
        show_ci
        and lo_col is not None
        and all(lo_col in df.columns and hi_col in df.columns
                for df in peaks_by_scheme.values())
    )

    countries = list(sort_df.sort_values(metric).index)
    if max_countries is not None and len(countries) > max_countries:
        countries = countries[:max_countries // 2] + countries[-max_countries // 2:]

    default_markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]
    try:
        default_colors = bw_colors(len(schemes))
    except Exception:
        default_colors = ["0.10", "0.45", "0.65", "0.30"][:len(schemes)] or ["0.10"]
    if schemes_styles is None:
        schemes_styles = {
            s: {
                "marker": default_markers[i % len(default_markers)],
                "color":  default_colors[i % len(default_colors)],
            }
            for i, s in enumerate(schemes)
        }

    # Outlier-detection bounds. Use IQR on pooled point estimates.
    pool_pts = np.concatenate([
        peaks_by_scheme[s][metric].dropna().to_numpy() for s in schemes
    ]) if any(metric in peaks_by_scheme[s].columns for s in schemes) else np.array([])
    if clip_outliers and pool_pts.size >= 6:
        q25, q75 = np.percentile(pool_pts, [25, 75])
        iqr = q75 - q25
        x_lo = q25 - outlier_factor * iqr
        x_hi = q75 + outlier_factor * iqr
        # Pad by 10% for visual breathing room.
        span = max(x_hi - x_lo, 1e-6)
        x_lo -= 0.10 * span
        x_hi += 0.10 * span
    else:
        x_lo, x_hi = None, None

    n_c = len(countries)
    y_country = np.arange(n_c)
    n_s = len(schemes)
    jitter = np.linspace(-0.22, 0.22, n_s) if n_s > 1 else [0.0]

    def _clip_value(v: float) -> tuple[float, bool]:
        """Return (clipped_value, is_clipped)."""
        if x_lo is None:
            return v, False
        if v < x_lo:
            return x_lo, True
        if v > x_hi:
            return x_hi, True
        return v, False

    for k, scheme in enumerate(schemes):
        df = peaks_by_scheme[scheme]
        sty = schemes_styles[scheme]
        for ci, code in enumerate(countries):
            if code not in df.index or pd.isna(df.loc[code, metric]):
                continue
            v = float(df.loc[code, metric])
            v_clip, was_clipped = _clip_value(v)
            yy = y_country[ci] + jitter[k]
            if has_ci and not was_clipped:
                lo_v = float(df.loc[code, lo_col])
                hi_v = float(df.loc[code, hi_col])
                # Clip CI ends so they don't blow up the axis on outlier ends.
                lo_clip, _ = _clip_value(lo_v)
                hi_clip, _ = _clip_value(hi_v)
                err_lo = max(0.0, v_clip - lo_clip)
                err_hi = max(0.0, hi_clip - v_clip)
                ax.errorbar(
                    [v_clip], [yy], xerr=[[err_lo], [err_hi]],
                    marker=sty["marker"], color=sty["color"],
                    ecolor=sty["color"], elinewidth=0.7, capsize=2,
                    ms=5, markeredgecolor="black", markeredgewidth=0.4,
                    label=scheme if ci == 0 else None,
                )
            else:
                ax.scatter(
                    [v_clip], [yy], marker=sty["marker"], color=sty["color"],
                    s=22, edgecolor="black", linewidth=0.4,
                    label=scheme if ci == 0 else None,
                )
            if was_clipped:
                # Annotate with the actual value at the clipped edge.
                ha = "left" if v < x_lo else "right"
                arrow = "←" if v < x_lo else "→"
                ax.annotate(
                    f"{arrow} {v:+.2f}",
                    (v_clip, yy), xytext=(2 if v < x_lo else -2, 0),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=6, color=sty["color"], style="italic",
                )

    if show_mean_group:
        for k, scheme in enumerate(schemes):
            df = peaks_by_scheme[scheme]
            vals = df[metric].dropna()
            if vals.empty:
                continue
            mu, sd = float(vals.mean()), float(vals.std(ddof=1))
            sty = schemes_styles[scheme]
            yy = n_c + 1.0 + jitter[k]
            mu_clip, _ = _clip_value(mu)
            ax.errorbar(
                [mu_clip], [yy],
                xerr=[[max(0.0, mu_clip - _clip_value(mu - sd)[0])],
                      [max(0.0, _clip_value(mu + sd)[0] - mu_clip)]],
                marker=sty["marker"], color=sty["color"],
                ecolor=sty["color"], elinewidth=1.2, capsize=3, ms=7,
                markeredgecolor="black", markeredgewidth=0.6,
            )

    ax.axvline(0, color="0.5", lw=0.5, ls=":")
    ax.set_yticks(list(y_country) + ([n_c + 1.0] if show_mean_group else []))
    ax.set_yticklabels(countries + (["Mean-group ± SD"] if show_mean_group else []),
                       fontsize=8)
    if show_mean_group:
        ax.axhline(n_c + 0.5, color="0.7", lw=0.4, ls="-")
    if x_lo is not None:
        ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    # Deduplicate legend.
    h, l = ax.get_legend_handles_labels()
    if h:
        seen = {}
        for hi_, li_ in zip(h, l):
            if li_ not in seen:
                seen[li_] = hi_
        ax.legend(seen.values(), seen.keys(), loc="best", fontsize=7, frameon=False)


def plot_mean_group_fan(
    ax,
    summary: pd.DataFrame,
    *,
    label: str = "",
    color: str = "0.10",
    linestyle: str = "-",
    fan_alpha: float = 0.30,
    fan_color: str = "0.65",
):
    """Plot mean-group IRF + 10/90 cross-country fan on `ax`."""
    if summary.empty:
        ax.text(0.5, 0.5, "no countries", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        return
    h = summary["h"]
    ax.fill_between(h, summary["q10"], summary["q90"],
                     color=fan_color, alpha=fan_alpha, lw=0)
    ax.plot(h, summary["mean"], color=color, linestyle=linestyle, lw=1.5,
             label=label or None)
    ax.axhline(0, color="0.3", lw=0.5, ls=":")
