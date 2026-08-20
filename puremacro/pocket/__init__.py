"""Offline data cartridges — research data that travels.

A cartridge (``.pmz``) is a portable, self-verifying bundle of
DataFrames plus the provenance that explains them. Pack one on the
machine that has network and pyarrow; open it on the one that has
neither.

    >>> from puremacro import pocket
    >>> pocket.pack(panel, "g7.pmz", source="OECD QNA",
    ...                vintage="2026-08-19")            # doctest: +SKIP
    >>> cart = pocket.load("g7.pmz")                 # doctest: +SKIP
    >>> print(cart.summary())                           # doctest: +SKIP

See :mod:`puremacro.pocket._cartridge` for the format description.
"""
from puremacro.pocket._cartridge import (
    FORMAT_VERSION,
    Cartridge,
    CartridgeError,
    FrameRecord,
    Provenance,
    from_base64,
    inspect_cartridge,
    load,
    loads,
    pack,
    packs,
    snapshot,
    to_base64,
)

__all__ = [
    "FORMAT_VERSION",
    "Cartridge", "CartridgeError", "FrameRecord", "Provenance",
    "pack", "packs", "load", "loads", "inspect_cartridge", "snapshot",
    "to_base64", "from_base64",
]
