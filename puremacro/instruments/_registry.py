"""Self-describing registry of available identified-shock instruments.

The registry is a process-wide dict of :class:`InstrumentSpec` entries
populated by :mod:`._catalog`. Public functions :func:`list_available`,
:func:`load`, and :func:`describe` provide ergonomic access. Use
:func:`register` to add a new spec at runtime (rare — most additions
should be in the catalog file).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from ._core import Instrument, VALID_CATEGORIES


@dataclass(frozen=True)
class InstrumentSpec:
    """Catalog entry describing one identified-shock series.

    Attributes
    ----------
    key : str
        Unique snake_case identifier.
    name : str
        Human-readable display name.
    category : str
        One of :data:`puremacro.instruments.VALID_CATEGORIES`.
    description : str
        One-paragraph what the series represents.
    reference : str
        Full citation (Author Year, journal, vol).
    loader : Callable[..., Instrument]
        Constructs the Instrument; may take kwargs.
    country : str | None
        ISO3 or None for cross-country.
    frequency : str
        Pandas-style frequency code: ``"M"``, ``"Q"``, ``"A"``.
    requires_network : bool
        True if loader needs HTTP.
    requires_fixture : bool
        True if loader needs a user-supplied CSV.
    """

    key: str
    name: str
    category: str
    description: str
    reference: str
    loader: Callable[..., Instrument]
    country: str | None
    frequency: str
    requires_network: bool
    requires_fixture: bool


_REGISTRY: dict[str, InstrumentSpec] = {}


def register(spec: InstrumentSpec) -> None:
    """Add a spec to the process-wide registry.

    Raises
    ------
    ValueError
        If ``spec.category`` is not in :data:`VALID_CATEGORIES`.

    Warns
    -----
    UserWarning
        If a spec with the same ``key`` was already registered (the
        previous entry is overwritten).
    """
    if spec.category not in VALID_CATEGORIES:
        raise ValueError(
            f"category {spec.category!r} not in {sorted(VALID_CATEGORIES)}"
        )
    if spec.key in _REGISTRY:
        warnings.warn(
            f"Registry key {spec.key!r} already exists; overwriting "
            f"the previous entry.",
            UserWarning,
            stacklevel=2,
        )
    _REGISTRY[spec.key] = spec


def _is_available(spec: InstrumentSpec) -> bool:
    """An entry is 'available' if it needs neither live network nor a
    user-supplied fixture."""
    return not spec.requires_network and not spec.requires_fixture


def list_available(
    *,
    category: str | None = None,
    country: str | None = None,
    include_unavailable: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of catalogued instruments, one row per spec.

    Columns: ``key``, ``name``, ``category``, ``country``, ``frequency``,
    ``reference``, ``available``, ``requires_network``, ``requires_fixture``.

    Parameters
    ----------
    category : str | None
        Filter to one category if provided.
    country : str | None
        Filter to ISO3 country if provided.
    include_unavailable : bool, default False
        If False, drop entries with ``requires_network=True`` or
        ``requires_fixture=True``.
    """
    rows = []
    for spec in _REGISTRY.values():
        if category is not None and spec.category != category:
            continue
        if country is not None and spec.country != country:
            continue
        avail = _is_available(spec)
        if not include_unavailable and not avail:
            continue
        rows.append({
            "key": spec.key,
            "name": spec.name,
            "category": spec.category,
            "country": spec.country,
            "frequency": spec.frequency,
            "reference": spec.reference,
            "available": avail,
            "requires_network": spec.requires_network,
            "requires_fixture": spec.requires_fixture,
        })
    return pd.DataFrame(rows, columns=[
        "key", "name", "category", "country", "frequency",
        "reference", "available", "requires_network", "requires_fixture",
    ])


def load(key: str, **kwargs: Any) -> Instrument:
    """Construct an Instrument by registry key. Forwards kwargs to the loader."""
    if key not in _REGISTRY:
        raise KeyError(
            f"Instrument key {key!r} not found in registry. "
            f"Use list_available(include_unavailable=True) to see all."
        )
    return _REGISTRY[key].loader(**kwargs)


def describe(key: str) -> str:
    """Return a multi-line human-readable description of the spec at key."""
    if key not in _REGISTRY:
        raise KeyError(f"Instrument key {key!r} not found in registry.")
    s = _REGISTRY[key]
    return (
        f"Instrument: {s.name}  ({s.key})\n"
        f"  category          : {s.category}\n"
        f"  country           : {s.country or '(cross-country)'}\n"
        f"  frequency         : {s.frequency}\n"
        f"  requires_network  : {s.requires_network}\n"
        f"  requires_fixture  : {s.requires_fixture}\n"
        f"  reference         : {s.reference}\n"
        f"  description       : {s.description}\n"
    )


__all__ = [
    "InstrumentSpec", "register",
    "list_available", "load", "describe",
]
