"""Federal Reserve speech archive (RSS)."""
from __future__ import annotations

import warnings
from typing import Iterator

from ._speeches import iter_speeches_rss
from ..._http import safe_get_bytes
from ._schema_check import assert_landmarks, ParserSchemaMismatchError


PARSER_SCHEMA_VERSION = 1

_FEED_URL = "https://www.federalreserve.gov/feeds/speeches.xml"


def iter_fed_speeches(*, fetch_body: bool = False) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for Fed speeches via RSS."""
    # Fetch the raw RSS bytes once to run the landmark check before
    # delegating to the shared speeches iterator.
    try:
        raw_bytes = safe_get_bytes(_FEED_URL)
    except Exception:
        return
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    try:
        assert_landmarks(
            raw_text, source="fed_speeches",
            expected_version=PARSER_SCHEMA_VERSION,
            landmarks=["Speeches", "Federal Reserve"],
        )
    except ParserSchemaMismatchError as e:
        from ._telemetry import log_event
        log_event(source="fed_speeches", outcome="parser_schema_mismatch",
                  fallback_used="none")
        warnings.warn(
            f"puremacro.narrative.sources.fed_speeches: schema mismatch "
            f"on first body: {e}",
            UserWarning, stacklevel=2,
        )
        return
    yield from iter_speeches_rss(
        _FEED_URL, bank_code="FED", country="USA", language="en",
        fetch_body=fetch_body,
    )


__all__ = ["iter_fed_speeches", "PARSER_SCHEMA_VERSION"]
