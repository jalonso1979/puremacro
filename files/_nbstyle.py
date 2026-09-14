"""Shared grayscale style + tiny palette helpers for the puremacro.vfi
showcase notebooks.

Kept intentionally thin: anything reused widely should graduate into a
``puremacro.vfi.plot`` module (improvement backlog #1). Mirrors the aesthetic
of ``puremacro.plot`` (grayscale, linestyle cycling) without importing its
private helpers.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

GRAYS = ["0.00", "0.45", "0.25", "0.65", "0.15", "0.80", "0.35", "0.55"]
LINESTYLES = ["-", (0, (4, 2)), (0, (1, 1)), "-.", (0, (3, 1, 1, 1))]


def apply_style() -> None:
    """Set publication-oriented grayscale rcParams for the notebook suite."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.figsize": (6.2, 3.8),
    })


def palette(n: int) -> list[str]:
    """n grayscale colors (extends past the base set by even spacing)."""
    if n <= len(GRAYS):
        return GRAYS[:n]
    return [f"{g:.3f}" for g in np.linspace(0.0, 0.8, n)]


def styles(n: int) -> list:
    """n line styles, cycling the base set if needed."""
    if n <= len(LINESTYLES):
        return LINESTYLES[:n]
    return (LINESTYLES * (n // len(LINESTYLES) + 1))[:n]
