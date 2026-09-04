"""Pyodide-friendly grayscale matplotlib helpers for puremacro.

Self-contained: no dependency on ``src.plotting.bw_style``. All helpers
accept an optional ``ax`` for composition into pre-built figure grids
and return a ``matplotlib.figure.Figure``. Defaults are grayscale so
figures print legibly without color.
"""
from __future__ import annotations

from typing import Mapping, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

_GRAYS = ["0.00", "0.45", "0.25", "0.65", "0.15", "0.80", "0.35", "0.55"]
_LINESTYLES = ["-", (0, (4, 2)), (0, (1, 1)), "-.",
               (0, (3, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1)),
               (0, (1, 2)), (0, (6, 3))]


def _palette(n: int) -> list[str]:
    if n <= len(_GRAYS):
        return _GRAYS[:n]
    return [f"{g:.3f}" for g in np.linspace(0.0, 0.8, n)]


def _styles(n: int) -> list:
    if n <= len(_LINESTYLES):
        return _LINESTYLES[:n]
    return (_LINESTYLES * ((n // len(_LINESTYLES)) + 1))[:n]


def _new_ax(ax, figsize=(5.5, 3.2)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def _irf_to_frame(irf, target_idx: int = 0, shock_idx: int = 0) -> pd.DataFrame:
    """Coerce dict-like, dataclass Result, or DataFrame IRFs into a DataFrame[h, beta, lo?, hi?]."""
    if isinstance(irf, pd.DataFrame) and "h" in irf.columns and "beta" in irf.columns:
        return irf
    raw_point = None
    for attr in ("irf_point", "irf_median", "irf_mean", "irfs", "point", "irf"):
        if hasattr(irf, attr) and getattr(irf, attr) is not None:
            raw_point = getattr(irf, attr)
            break
    if raw_point is not None:
        point = np.asarray(raw_point)
        if point.ndim == 3:
            point = point[:, target_idx, shock_idx]
        elif point.ndim == 2:
            point = point[:, target_idx]
        elif point.ndim != 1:
            raise ValueError(f"irf array has unexpected ndim {point.ndim}")
        H = len(point)
        df = pd.DataFrame({"h": np.arange(H), "beta": point})
        raw_lo = getattr(irf, "irf_lower", getattr(irf, "lower", None))
        if raw_lo is not None:
            lo = np.asarray(raw_lo)
            if lo.ndim == 3:
                lo = lo[:, target_idx, shock_idx]
            elif lo.ndim == 2:
                lo = lo[:, target_idx]
            df["lo"] = lo
        raw_hi = getattr(irf, "irf_upper", getattr(irf, "upper", None))
        if raw_hi is not None:
            hi = np.asarray(raw_hi)
            if hi.ndim == 3:
                hi = hi[:, target_idx, shock_idx]
            elif hi.ndim == 2:
                hi = hi[:, target_idx]
            df["hi"] = hi
        return df
    if isinstance(irf, Mapping) and "irf_point" in irf:
        point = irf["irf_point"]
        if isinstance(point, pd.Series):
            df = pd.DataFrame({"h": np.asarray(point.index), "beta": point.values})
            if irf.get("irf_lower") is not None:
                df["lo"] = np.asarray(irf["irf_lower"]).ravel()
            if irf.get("irf_upper") is not None:
                df["hi"] = np.asarray(irf["irf_upper"]).ravel()
            return df
        arr = np.asarray(point)
        if arr.ndim == 3:
            arr = arr[:, target_idx, shock_idx]
        elif arr.ndim == 2:
            arr = arr[:, target_idx]
        elif arr.ndim != 1:
            raise TypeError("dict-form irf_point must be a pd.Series or 1D/2D/3D ndarray")
        df = pd.DataFrame({"h": np.arange(len(arr)), "beta": arr})
        if irf.get("irf_lower") is not None:
            lo = np.asarray(irf["irf_lower"])
            if lo.ndim == 3:
                lo = lo[:, target_idx, shock_idx]
            elif lo.ndim == 2:
                lo = lo[:, target_idx]
            df["lo"] = lo.ravel()
        if irf.get("irf_upper") is not None:
            hi = np.asarray(irf["irf_upper"])
            if hi.ndim == 3:
                hi = hi[:, target_idx, shock_idx]
            elif hi.ndim == 2:
                hi = hi[:, target_idx]
            df["hi"] = hi.ravel()
        return df
    raise TypeError("irf must be a DataFrame[h,beta,...], a result dataclass, or a dict with irf_point")


def plot_irf_single(irf, *, title: str = "", ylabel: str = "Response",
                    scale: float = 1.0, target_idx: int = 0, shock_idx: int = 0,
                    ax=None) -> Figure:
    """Plot a single IRF with optional shaded band; series scaled by ``scale``."""
    df = _irf_to_frame(irf, target_idx=target_idx, shock_idx=shock_idx)
    fig, ax = _new_ax(ax)
    ax.plot(df["h"], df["beta"] * scale, color="0.0", linewidth=1.4, label="point")
    if {"lo", "hi"}.issubset(df.columns):
        ax.fill_between(df["h"], df["lo"] * scale, df["hi"] * scale,
                        color="0.75", alpha=0.5, linewidth=0, label="band")
    ax.axhline(0.0, color="0.3", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def plot_irf_multi(irf_by_label: Mapping[str, Any], *, title: str = "",
                   ylabel: str = "Response", show_bands: bool = False,
                   scale: float | Mapping[str, float] = 1.0,
                   target_idx: int = 0, shock_idx: int = 0,
                   ax=None) -> Figure:
    """Overlay several IRFs (point + optional bands) using grayscale styles."""
    fig, ax = _new_ax(ax, figsize=(6.2, 3.5))
    n = len(irf_by_label)
    colors = _palette(n)
    styles = _styles(n)
    for (label, raw_irf), c, ls in zip(irf_by_label.items(), colors, styles):
        df = _irf_to_frame(raw_irf, target_idx=target_idx, shock_idx=shock_idx)
        s = scale[label] if isinstance(scale, Mapping) else scale
        ax.plot(df["h"], df["beta"] * s, color=c, linestyle=ls,
                linewidth=1.3, label=label)
        if show_bands and {"lo", "hi"}.issubset(df.columns):
            ax.fill_between(df["h"], df["lo"] * s, df["hi"] * s,
                            color=c, alpha=0.15, linewidth=0)
    ax.axhline(0.0, color="0.3", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    return fig


def plot_fevd(fevd_array: np.ndarray, var_names: Sequence[str], *,
              title: str = "", ax=None) -> Figure:
    """Stacked-area plot of one variable's FEVD across n shocks (rows sum to 1)."""
    arr = np.asarray(fevd_array, dtype=float)
    if arr.ndim != 2:
        raise ValueError("fevd_array must be 2-D (H+1, n)")
    n_h, n = arr.shape
    if len(var_names) != n:
        raise ValueError("var_names length must equal fevd_array.shape[1]")
    fig, ax = _new_ax(ax, figsize=(6.0, 3.4))
    horizons = np.arange(n_h)
    ax.stackplot(horizons, arr.T, labels=list(var_names), colors=_palette(n),
                 edgecolor="white", linewidth=0.4)
    ax.set_xlim(horizons[0], horizons[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("Variance share")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    return fig


def _pick_quantile(df: pd.DataFrame, level: float) -> tuple[str, str] | None:
    """Find row labels for the lo/hi quantiles bracketing a central ``level``."""
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    cand_lo = [f"q{int(round(lo_q * 100)):02d}", f"{lo_q:.2f}",
               f"lo_{int(round(level * 100))}"]
    cand_hi = [f"q{int(round(hi_q * 100)):02d}", f"{hi_q:.2f}",
               f"hi_{int(round(level * 100))}"]
    idx = df.index.astype(str)
    lo = next((c for c in cand_lo if c in idx), None)
    hi = next((c for c in cand_hi if c in idx), None)
    return (lo, hi) if (lo is not None and hi is not None) else None


def plot_fan(forecast_df: pd.DataFrame, *,
             levels: Sequence[float] = (0.5, 0.8, 0.95),
             title: str = "", ax=None) -> Figure:
    """Fan chart: ``forecast_df`` has horizons as columns; rows are mean/quantiles."""
    fig, ax = _new_ax(ax, figsize=(6.0, 3.4))
    horizons = np.asarray(forecast_df.columns)
    sorted_levels = sorted(levels, reverse=True)
    shades = np.linspace(0.85, 0.55, max(len(sorted_levels), 1))
    for shade, level in zip(shades, sorted_levels):
        match = _pick_quantile(forecast_df, level)
        if match is None:
            continue
        lo_lab, hi_lab = match
        ax.fill_between(horizons, forecast_df.loc[lo_lab].values,
                        forecast_df.loc[hi_lab].values,
                        color=f"{shade:.3f}", linewidth=0,
                        label=f"{int(round(level * 100))}%")
    centre = "mean" if "mean" in forecast_df.index.astype(str) else (
        "median" if "median" in forecast_df.index.astype(str) else None)
    if centre is not None:
        ax.plot(horizons, forecast_df.loc[centre].values,
                color="0.0", linewidth=1.4, label=centre)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Forecast")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    return fig


def plot_fan_from_paths(
    paths: np.ndarray,
    *,
    var_index: int = 0,
    levels: Sequence[float] = (0.50, 0.80, 0.95),
    h_axis: Union[Sequence, np.ndarray, None] = None,
    title: str = "",
    ax=None,
) -> Figure:
    """Fan chart from a (D, H+1, n) bundle of posterior-predictive paths.

    Companion to ``puremacro.posterior.simulate_var_paths``. The
    median is overlaid as a black line; bands are stacked from outermost
    to innermost (so the densest shade is the narrowest band).
    """
    fig, ax = _new_ax(ax, figsize=(6.5, 3.4))
    if paths.ndim != 3:
        raise ValueError("paths must be (n_draws, H+1, n).")
    p = paths[:, :, var_index]
    H = p.shape[1]
    if h_axis is None:
        h_axis = np.arange(H)
    sorted_levels = sorted(levels, reverse=True)
    shades = np.linspace(0.85, 0.55, max(len(sorted_levels), 1))
    for shade, level in zip(shades, sorted_levels):
        lo = np.percentile(p, 100 * (1 - level) / 2, axis=0)
        hi = np.percentile(p, 100 * (1 - (1 - level) / 2), axis=0)
        ax.fill_between(h_axis, lo, hi,
                        color=f"{shade:.3f}", linewidth=0,
                        label=f"{int(round(level * 100))}%")
    median = np.percentile(p, 50, axis=0)
    ax.plot(h_axis, median, color="0.0", linewidth=1.2, label="median")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Forecast")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    return fig


def plot_spillover(spill_df: pd.DataFrame, *, title: str = "", ax=None) -> Figure:
    """Heatmap of a square pairwise spillover matrix (rows=from, cols=to)."""
    if spill_df.shape[0] != spill_df.shape[1]:
        raise ValueError("spill_df must be square")
    fig, ax = _new_ax(ax, figsize=(5.0, 4.4))
    im = ax.imshow(spill_df.values.astype(float), cmap="Greys", aspect="auto")
    ax.set_xticks(range(spill_df.shape[1]))
    ax.set_xticklabels(list(spill_df.columns), rotation=45, ha="right")
    ax.set_yticks(range(spill_df.shape[0]))
    ax.set_yticklabels(list(spill_df.index))
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


__all__ = [
    "plot_irf_single",
    "plot_irf_multi",
    "plot_fevd",
    "plot_fan",
    "plot_fan_from_paths",
    "plot_spillover",
]
