"""Pydantic schema validation for source-connector 4-tuples (roadmap D4).

Every ``iter_*`` connector under :mod:`puremacro.narrative.sources`
must yield records shaped as:

    (date, text, url, metadata)

where ``date`` is convertible to ``pd.Timestamp``, ``text`` and ``url``
are strings, and ``metadata`` is a dict (commonly carrying ``country``,
``language``, ``doctype``, ``bank_code``, plus connector-specific extras).

This module provides ``validate_records(iterable)`` which wraps any
connector iterator and (a) raises ``SourceRecordValidationError`` on
the first ill-shaped record, or (b) is opt-in tolerant: pass
``strict=False`` to emit a warning and drop the bad record. Useful
when wiring new connectors into NB31 — catches the "3-tuple from
iter_google_news but corpus loop unpacks 4" class of bugs at the
boundary.

Pydantic isn't a runtime dependency of puremacro (yet). Falls back
to manual checks if pydantic isn't installed.
"""
from __future__ import annotations

import warnings
from typing import Any, Iterable, Iterator

import pandas as pd


class SourceRecordValidationError(TypeError):
    """Raised when a connector record fails the 4-tuple shape check."""


def _validate_one(rec: Any, *, ix: int) -> tuple:
    """Coerce + validate a single record. Returns a 5-tuple
    (date, text, url, meta, magnitude); ``magnitude`` is ``None`` for
    legacy 4-tuple inputs."""
    if not isinstance(rec, tuple):
        raise SourceRecordValidationError(
            f"record {ix}: expected tuple, got {type(rec).__name__}"
        )
    if len(rec) not in (4, 5):
        raise SourceRecordValidationError(
            f"record {ix}: expected 4-tuple or 5-tuple "
            f"(date, text, url, metadata[, magnitude]); got {len(rec)}-tuple"
        )
    if len(rec) == 4:
        date, text, url, meta = rec
        magnitude = None
    else:
        date, text, url, meta, magnitude = rec

    try:
        date = pd.Timestamp(date)
    except (TypeError, ValueError) as e:
        raise SourceRecordValidationError(
            f"record {ix}: date={date!r} not convertible to Timestamp ({e})"
        ) from e
    if not isinstance(text, str):
        raise SourceRecordValidationError(
            f"record {ix}: text must be str, got {type(text).__name__}"
        )
    if not isinstance(url, str):
        raise SourceRecordValidationError(
            f"record {ix}: url must be str, got {type(url).__name__}"
        )
    if not isinstance(meta, dict):
        raise SourceRecordValidationError(
            f"record {ix}: metadata must be dict, got {type(meta).__name__}"
        )
    if magnitude is not None:
        if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)):
            raise SourceRecordValidationError(
                f"record {ix}: magnitude must be numeric or None, "
                f"got {type(magnitude).__name__}"
            )
        if magnitude < 0:
            raise SourceRecordValidationError(
                f"record {ix}: magnitude must be ≥ 0, got {magnitude}"
            )
        magnitude = float(magnitude)
    return (date, text, url, meta, magnitude)


def validate_records(
    iterable: Iterable[Any],
    *,
    strict: bool = True,
) -> Iterator[tuple]:
    """Yield validated 4-tuples; raise (strict=True) or warn (strict=False)
    on the first malformed record.

    When ``strict=False``, malformed records are dropped silently after
    emitting a ``UserWarning``. This is the mode appropriate for NB31's
    corpus-assembly loop, where you want to keep going even if one
    upstream connector misbehaves.
    """
    for i, rec in enumerate(iterable):
        try:
            yield _validate_one(rec, ix=i)
        except SourceRecordValidationError as e:
            if strict:
                raise
            warnings.warn(str(e), UserWarning, stacklevel=2)


__all__ = ["validate_records", "SourceRecordValidationError"]
