"""Portable .pmz cartridge integration for Latin American real-time macro data.

Uses `puremacro.pocket` to package and load offline self-verifying
cartridges of vintage panels without requiring network or pyarrow. A
cartridge is a plain zip holding ``manifest.json`` plus one ``.npz``
per frame, read with ``allow_pickle=False``: no pickle, no code
execution on load.

Two frames are written: ``data`` (the tidy vintage frame) and, when
the packed object is a :class:`VintagePanel` with metadata,
``vintage_metadata`` — a key / JSON-value table that
:func:`load_realtime_cartridge` turns back into ``panel.metadata`` so
``failed``, ``missing``, ``freq``, ``provider_used`` and friends
survive the round trip.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from puremacro import pocket
from ._base import VintagePanel

_DATA_FRAME = "data"
_METADATA_FRAME = "vintage_metadata"


def _metadata_frame(metadata: dict[str, Any]) -> pd.DataFrame:
    """``{key: value}`` -> two string columns, values JSON-encoded.

    Values that are not JSON-native (a DataFrame of dropped editions,
    a Timestamp) are stored through ``str`` rather than dropped, so the
    cartridge still says what was there.
    """
    keys, values = [], []
    for key, value in metadata.items():
        try:
            encoded = json.dumps(value, default=str)
        except (TypeError, ValueError):
            encoded = json.dumps(str(value))
        keys.append(str(key))
        values.append(encoded)
    return pd.DataFrame({"key": keys, "json": values})


def _metadata_from_frame(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, encoded in zip(frame["key"], frame["json"]):
        try:
            out[str(key)] = json.loads(encoded)
        except (TypeError, ValueError):
            out[str(key)] = encoded
    return out


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
        The vintage panel or tidy vintage DataFrame to package. A
        panel's ``metadata`` is packed alongside its frame.
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
    if isinstance(panel_or_df, VintagePanel):
        df, metadata = panel_or_df.df, panel_or_df.metadata
    else:
        df, metadata = panel_or_df, {}
    if vintage is None:
        vintage = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    frames: dict[str, pd.DataFrame] = {_DATA_FRAME: df}
    if metadata:
        frames[_METADATA_FRAME] = _metadata_frame(metadata)
    out_path = Path(path)
    pocket.pack(frames, out_path, source=source, vintage=vintage, notes=notes)
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
        Whether to verify SHA-256 digests of frames while reading.
        Defaults to True; ``False`` skips the check entirely.

    Returns
    -------
    VintagePanel
        The reconstructed vintage panel. ``metadata`` holds whatever the
        packed panel carried, plus ``cartridge_path`` and the
        ``provenance_*`` fields of the manifest.
    """
    cart = pocket.load(path, verify=verify)
    df = cart[_DATA_FRAME] if _DATA_FRAME in cart else cart.frame()
    metadata: dict[str, Any] = (
        _metadata_from_frame(cart[_METADATA_FRAME])
        if _METADATA_FRAME in cart else {}
    )
    metadata.update({
        "cartridge_path": str(path),
        "provenance_source": cart.provenance.source,
        "provenance_vintage": cart.provenance.vintage,
        "provenance_notes": cart.provenance.notes,
    })
    return VintagePanel(df=df, metadata=metadata)


__all__ = [
    "pack_realtime_cartridge",
    "load_realtime_cartridge",
]
