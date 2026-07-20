"""Black-and-white publication-grade matplotlib style.

Use via:

    from puremacro.plotting.bw_style import apply_bw_style, bw_colors, bw_linestyles
    apply_bw_style()                    # global rcParams
    colors  = bw_colors(n=6)            # grayscale palette, well-separated
    styles  = bw_linestyles(n=6)        # line styles (solid/dashed/dotted/…)

Figures are rendered at 600 DPI by default; toggle ``dpi`` if you need smaller files.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler as mpl_cycler
import numpy as np


# Grayscale palette: 0.0 = black, 1.0 = white. Avoid pure white for visibility.
_GRAYS = [0.00, 0.45, 0.25, 0.65, 0.15, 0.80, 0.35, 0.55]

# A compact cycle of line styles that stay distinguishable when printed on paper.
_LINESTYLES = [
    "-",          # solid
    (0, (4, 2)),  # long-dash
    (0, (1, 1)),  # dotted
    "-.",         # dashdot
    (0, (3, 1, 1, 1)),     # dash-dot-dot
    (0, (5, 1, 1, 1, 1, 1)),  # dash-dot-dot-dot
    (0, (1, 2)),  # sparse-dotted
    (0, (6, 3)),  # long-dash wide
]

# Hatch patterns for fill_between bands — keeps CI bands legible in grayscale print.
_HATCHES = ["", "///", "...", "\\\\\\", "xxx", "|||", "---", "+++"]


def bw_colors(n: int = 4) -> list[str]:
    """Return ``n`` grayscale colors as matplotlib-compatible shade strings."""
    shades = _GRAYS[:n] if n <= len(_GRAYS) else np.linspace(0.0, 0.8, n).tolist()
    return [f"{g:.3f}" for g in shades]


def bw_linestyles(n: int = 4) -> list:
    """Return ``n`` line styles compatible with matplotlib ``linestyle`` arg."""
    if n <= len(_LINESTYLES):
        return _LINESTYLES[:n]
    return (_LINESTYLES * ((n // len(_LINESTYLES)) + 1))[:n]


def bw_hatches(n: int = 4) -> list[str]:
    """Return ``n`` hatch patterns for ``fill_between``."""
    if n <= len(_HATCHES):
        return _HATCHES[:n]
    return (_HATCHES * ((n // len(_HATCHES)) + 1))[:n]


def apply_bw_style(*, dpi: int = 600, serif: bool = True) -> None:
    """Install the black-and-white publication style globally."""
    rc = {
        # Colors: everything defaults to black; the property cycle uses grayscale.
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "grid.color": "0.7",
        "grid.linestyle": ":",
        "grid.linewidth": 0.4,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.prop_cycle": mpl_cycler(color=bw_colors(8)),
        # High DPI for raster outputs.
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Lines: thicker + well-scaled markers.
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        # Axes spines: hide top+right for cleaner look.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        # Ticks: inside+tight.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Legends: smaller, no frame.
        "legend.fontsize": 8,
        "legend.frameon": False,
        # Fonts.
        "font.size": 10,
        "font.family": "serif" if serif else "sans-serif",
        "mathtext.fontset": "cm",  # Computer Modern for math
        # PDF-friendly: don't embed fonts as paths (keeps file size small).
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    # matplotlib >= 3.11 stubs key rcParams by a Literal of every rc name;
    # a str-keyed dict is correct at runtime on all supported versions.
    mpl.rcParams.update(rc)  # type: ignore[arg-type]


@contextmanager
def bw_context(*, dpi: int = 600, serif: bool = True):
    """Context manager: apply bw style inside a ``with`` block; restore on exit."""
    prev = mpl.rcParams.copy()
    try:
        apply_bw_style(dpi=dpi, serif=serif)
        yield
    finally:
        mpl.rcParams.update(prev)


def save_pub(fig, path_no_ext: str, dpi: int = 600) -> list[str]:
    """Save ``fig`` as both PDF and high-DPI PNG; return the list of filepaths."""
    import os
    os.makedirs(os.path.dirname(path_no_ext) or ".", exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = f"{path_no_ext}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    return paths
