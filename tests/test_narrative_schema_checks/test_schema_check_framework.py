"""F2.3 — ParserSchemaMismatchError + assert_landmarks framework."""
from __future__ import annotations

import pytest


def test_passes_when_all_landmarks_present():
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = "<html><body><h1>Beige Book</h1><p>Summary of Commentary</p></body></html>"
    # Should not raise.
    assert_landmarks(
        text, source="beige_book", expected_version=1,
        landmarks=["Beige Book", "Summary of Commentary"],
    )


def test_raises_on_missing_substring_landmark():
    from puremacro.narrative.sources._schema_check import (
        assert_landmarks, ParserSchemaMismatchError,
    )
    text = "<html><body><h1>Beige Book</h1></body></html>"
    with pytest.raises(ParserSchemaMismatchError) as exc_info:
        assert_landmarks(
            text, source="beige_book", expected_version=1,
            landmarks=["Beige Book", "MISSING SENTENCE"],
        )
    msg = str(exc_info.value)
    assert "beige_book" in msg
    assert "MISSING SENTENCE" in msg
    assert "version=1" in msg


def test_supports_tuple_landmark_form():
    """(selector_hint, expected_text) tuples — selector is informational,
    the check is `expected_text in text`."""
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = "<html><h1>Beige Book</h1></html>"
    assert_landmarks(
        text, source="beige_book", expected_version=1,
        landmarks=[("h1", "Beige Book")],
    )


def test_raises_on_missing_tuple_landmark():
    from puremacro.narrative.sources._schema_check import (
        assert_landmarks, ParserSchemaMismatchError,
    )
    text = "<html><h1>Something Else</h1></html>"
    with pytest.raises(ParserSchemaMismatchError) as exc_info:
        assert_landmarks(
            text, source="beige_book", expected_version=1,
            landmarks=[("h1", "Beige Book")],
        )
    assert "h1" in str(exc_info.value)
    assert "Beige Book" in str(exc_info.value)


def test_parser_schema_mismatch_is_runtimeerror():
    from puremacro.narrative.sources._schema_check import ParserSchemaMismatchError
    assert issubclass(ParserSchemaMismatchError, RuntimeError)
