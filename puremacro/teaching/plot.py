"""Thin grayscale plotting helpers for IRFs. Wraps ``src.plotting.bw_style``.

Unit convention: ``scale`` lets callers express IRFs in interpretable
units. For Cholesky/SVAR responses to a unit-variance structural shock
on a log-level variable, ``scale=100`` turns the response into
percentage points. For LP regressions where the shock is :math:`\\Delta\\log`
of a proxy, pass ``scale = sigma_shock * 100`` to express the response
as "% of outcome per 1-σ shock".
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from puremacro.plotting.bw_style import apply_bw_style, bw_colors, bw_linestyles


def plot_irf_single(
    irf: pd.DataFrame,
    title: str = "",
    ylabel: str = r"Respuesta (% por 1-$\sigma$ shock)",
    scale: float = 1.0,
):
    """Plot a single IRF with shaded 68% band.

    `irf` must have columns h, beta, lo, hi. Series are multiplied by
    ``scale`` before plotting.
    """
    apply_bw_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(irf["h"], irf["beta"] * scale, color="0.0", linewidth=1.4, label="punto")
    if {"lo", "hi"}.issubset(irf.columns):
        ax.fill_between(irf["h"], irf["lo"] * scale, irf["hi"] * scale,
                         color="0.75", alpha=0.5, linewidth=0)
    ax.axhline(0, color="0.3", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Horizonte (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def plot_irf_multi(
    irf_by_proxy: dict[str, pd.DataFrame],
    title: str = "",
    ylabel: str = r"Respuesta (% por 1-$\sigma$ shock)",
    show_bands: bool = False,
    scale: float | dict[str, float] = 1.0,
):
    """Overlay IRFs (point only, or with bands if `show_bands`) from multiple proxies.

    `irf_by_proxy` maps a label → DataFrame with columns h, beta, [lo, hi].
    ``scale`` can be a scalar applied to every series, or a dict
    ``{label: factor}`` for per-proxy SD scaling.
    """
    apply_bw_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    colors = bw_colors(len(irf_by_proxy))
    styles = bw_linestyles(len(irf_by_proxy))
    for (label, df), c, ls in zip(irf_by_proxy.items(), colors, styles):
        s = scale[label] if isinstance(scale, dict) else scale
        ax.plot(df["h"], df["beta"] * s, color=c, linestyle=ls, linewidth=1.3, label=label)
        if show_bands and {"lo", "hi"}.issubset(df.columns):
            ax.fill_between(df["h"], df["lo"] * s, df["hi"] * s,
                             color=c, alpha=0.15, linewidth=0)
    ax.axhline(0, color="0.3", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Horizonte (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    return fig
