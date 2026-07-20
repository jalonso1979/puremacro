"""Deprecated: renamed to :mod:`ecb_decision` in 0.6.1.

This shim keeps any pre-0.6.1 imports working but emits a
DeprecationWarning when the function is called. The new connector
yields a 4-tuple ``(date, text, source_url, metadata)``; the legacy
3-tuple shape is no longer offered.
"""
from __future__ import annotations

import warnings
from typing import Iterator

from .ecb_decision import iter_ecb_decision
from ._schema_check import assert_landmarks, ParserSchemaMismatchError


PARSER_SCHEMA_VERSION = 1


def iter_ecb_press(
    *, language: str = "en", feed_url: str | None = None,
    fetch_body: bool = False,
) -> Iterator[tuple]:
    warnings.warn(
        "iter_ecb_press is deprecated; use iter_ecb_decision instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _checked = False
    for record in iter_ecb_decision(language=language, feed_url=feed_url,
                                    fetch_body=fetch_body):
        if not _checked:
            # record is (date, text, url, metadata); check the text field.
            _, text, *_ = record
            try:
                assert_landmarks(
                    text, source="ecb_press",
                    expected_version=PARSER_SCHEMA_VERSION,
                    landmarks=["European Central Bank", "Press release"],
                )
            except ParserSchemaMismatchError as e:
                from ._telemetry import log_event
                log_event(source="ecb_press", outcome="parser_schema_mismatch",
                          fallback_used="none")
                warnings.warn(
                    f"puremacro.narrative.sources.ecb_press: schema mismatch "
                    f"on first body: {e}",
                    UserWarning, stacklevel=2,
                )
                return
            _checked = True
        yield record


__all__ = ["iter_ecb_press", "PARSER_SCHEMA_VERSION"]
