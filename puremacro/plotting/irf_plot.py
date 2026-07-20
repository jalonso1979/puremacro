"""Shared IRF plotter — duck-typed over SVAR and LP output arrays.

Accepts ``point``, ``lower``, ``upper`` arrays shaped ``(n_targets, n_shocks, H+1)``.
LP-style output with a single shock can be passed with shape ``(n_targets, 1, H+1)``.

Normalization options:
    - ``normalize=None``       : raw IRFs (default; no scaling).
    - ``normalize='pct'``      : multiply by 100 (log-target → percent response).
    - ``normalize='sd_target'``: divide each row by the passed ``target_sd`` vector,
                                 so the response is in SDs of each target variable.
    - ``normalize='sd_shock'`` : multiply by the passed ``shock_sd`` (expresses the
                                 IRF as the response to a 1-SD shock when the raw
                                 impact matrix was unit-normalized).

Black-and-white mode (``bw=True``): uses the grayscale palette + line-style cycle
from ``src.plotting.bw_style``, suitable for publication / print.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _apply_normalization(
    arrs: tuple[np.ndarray | None, ...],
    normalize: str | None,
    target_sd: np.ndarray | None = None,
    shock_sd: np.ndarray | None = None,
) -> tuple[np.ndarray | None, ...]:
    if normalize is None:
        return arrs
    out: List[Optional[np.ndarray]] = []
    for a in arrs:
        if a is None:
            out.append(None)
            continue
        a = np.asarray(a, dtype=float)
        if normalize == "pct":
            out.append(a * 100.0)
        elif normalize == "sd_target":
            if target_sd is None:
                raise ValueError("normalize='sd_target' requires target_sd array")
            t_sd = np.asarray(target_sd, dtype=float).reshape(-1, 1, 1)
            out.append(a / t_sd)
        elif normalize == "sd_shock":
            if shock_sd is None:
                raise ValueError("normalize='sd_shock' requires shock_sd array")
            s_sd = np.asarray(shock_sd, dtype=float).reshape(1, -1, 1)
            out.append(a * s_sd)
        else:
            raise ValueError(f"Unknown normalize option: {normalize!r}")
    return tuple(out)


def plot_irf(
    *,
    point: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    target_labels: Sequence[str],
    shock_labels: Sequence[str],
    title: str = "",
    band_alpha: float = 0.2,
    bw: bool = False,
    normalize: str | None = None,
    target_sd: np.ndarray | None = None,
    shock_sd: np.ndarray | None = None,
    ylabel_suffix: str | None = None,
) -> Figure:
    """Plot a grid of IRF panels.

    Parameters
    ----------
    point, lower, upper
        IRF arrays shaped ``(n_targets, n_shocks, H+1)``.
    target_labels, shock_labels
        Axis labels.
    title
        Figure-level title.
    band_alpha
        Transparency of the CI fill (ignored if ``bw=True`` — uses hatch instead).
    bw
        If True, render in grayscale with line-style cycle and hatched CI bands.
    normalize
        ``None | 'pct' | 'sd_target' | 'sd_shock'``.
    target_sd, shock_sd
        Required when the corresponding ``normalize`` mode is selected.
    ylabel_suffix
        Appended to each row label (e.g. " (%)" when normalize='pct').
    """
    if point.ndim != 3:
        raise ValueError("point must be (n_targets, n_shocks, H+1)")
    n_t, n_s, h1 = point.shape
    assert len(target_labels) == n_t and len(shock_labels) == n_s

    _norm_point, lower, upper = _apply_normalization(
        (point, lower, upper),
        normalize=normalize,
        target_sd=target_sd,
        shock_sd=shock_sd,
    )
    # _apply_normalization preserves non-None inputs; point was ndarray so result is ndarray.
    assert _norm_point is not None
    point = _norm_point

    # Style choices
    line_colors: List[Optional[str]]
    if bw:
        from .bw_style import bw_colors, bw_linestyles

        _bw_colors: List[Optional[str]] = list(bw_colors(max(n_s, 1)))
        line_colors = _bw_colors
        line_styles = bw_linestyles(max(n_s, 1))
        band_face = "0.8"
        band_edge = "0.5"
    else:
        line_colors = [None] * n_s   # matplotlib default cycle
        line_styles = ["-"] * n_s
        band_face = None
        band_edge = None

    horizon = np.arange(h1)
    fig, axes = plt.subplots(
        n_t, n_s,
        figsize=(3.2 * n_s, 2.2 * n_t),
        sharex=True,
        squeeze=False,
    )
    for i in range(n_t):
        for j in range(n_s):
            ax = axes[i, j]
            ax.axhline(0, color="black" if bw else "k", lw=0.5)
            if lower is not None and upper is not None:
                if bw:
                    ax.fill_between(
                        horizon, lower[i, j], upper[i, j],
                        facecolor=band_face, edgecolor=band_edge,
                        linewidth=0.3, alpha=0.5,
                    )
                else:
                    ax.fill_between(horizon, lower[i, j], upper[i, j], alpha=band_alpha)
            ax.plot(
                horizon, point[i, j],
                lw=1.4,
                color=line_colors[j] if line_colors[j] is not None else None,
                linestyle=line_styles[j],
            )
            if i == 0:
                ax.set_title(f"← {shock_labels[j]}", fontsize=10)
            if j == 0:
                label = target_labels[i]
                if ylabel_suffix:
                    label = f"{label}{ylabel_suffix}"
                ax.set_ylabel(label, fontsize=9)
            ax.grid(True, linestyle=":" if bw else "-", alpha=0.5 if bw else 0.2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
