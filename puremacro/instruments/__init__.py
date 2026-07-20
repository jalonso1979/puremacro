"""Unified Instrument protocol + discovery registry.

See :class:`Instrument` for the canonical wrapper, :class:`InstrumentLike`
for the Protocol, and :func:`list_available` / :func:`load` for the
discovery registry. :func:`compose` combines multiple Instruments into
a composite series.
"""
from ._core import Instrument, InstrumentLike, VALID_CATEGORIES
from ._compose import compose
from ._registry import (
    InstrumentSpec, register,
    list_available, load, describe,
)
from . import _catalog  # noqa: F401  — populates _REGISTRY at import time

__all__ = [
    "Instrument", "InstrumentLike", "VALID_CATEGORIES",
    "compose",
    "InstrumentSpec", "register",
    "list_available", "load", "describe",
]
