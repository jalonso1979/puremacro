"""Portable .pmz cartridge integration for Latin American real-time macro data.

Uses `puremacro.pocket` to package and load offline self-verifying cartridges
of vintage panels without requiring network or pyarrow.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from puremacro import pocket
from ._base import VintagePanel, normalize_vintage_frame


def pack_realtime_cartridge(
    panel_or_df: VintagePanel | pd.DataFrame,
    path: str | Path,
    *,
    source: str = "LatAm Regional Real-Time Ecosystem",
    vintage: str | None = None,
    notes: str = "Latin America central bank real-time macroeconomic vintage cartridge",
) -> Path:
    """Pack a VintagePanel or vintage DataFrame into a portable .pmz cartridge.

    Parameters
    ----------
    panel_or_df : VintagePanel | pd.DataFrame
        The vintage panel or tidy vintage DataFrame to package.
    path : str | Path
        Destination path for the .pmz file.
    source : str
        Upstream source description.
    vintage : str | None
        Vintage date stamp. Defaults to current UTC date.
    notes : str
        Provenance notes.

    Returns
    -------
    Path
        Path to the written .pmz file.
    """
    df = panel_or_df.df if isinstance(panel_or_df, VintagePanel) else panel_or_df
    if vintage is None:
        vintage = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    out_path = Path(path)
    pocket.pack(df, out_path, source=source, vintage=vintage, notes=notes)
    return out_path


def load_realtime_cartridge(
    path: str | Path,
    *,
    verify: bool = True,
) -> VintagePanel:
    """Load a VintagePanel from a portable .pmz cartridge.

    Parameters
    ----------
    path : str | Path
        Path to the .pmz file.
    verify : bool
        Whether to verify SHA-256 digests of frames. Defaults to True.

    Returns
    -------
    VintagePanel
        The reconstructed vintage panel with cartridge provenance metadata.
    """
    cart = pocket.load(path)
    if verify:
        cart.verify()
    df = cart.frame()
    return VintagePanel(
        df=df,
        metadata={
            "cartridge_path": str(path),
            "provenance_source": cart.provenance.source,
            "provenance_vintage": cart.provenance.vintage,
            "provenance_notes": cart.provenance.notes,
        },
    )


__all__ = [
    "pack_realtime_cartridge",
    "load_realtime_cartridge",
]
