"""F2.3 — Each connector's golden fixture parses without raising."""
from __future__ import annotations

import pathlib

import pytest


_FIXTURE_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "puremacro" / "narrative" / "sources" / "_fixtures"
)


def _fixture_text(name: str, ext: str) -> str:
    return (_FIXTURE_DIR / f"{name}_v1.{ext}").read_text(encoding="utf-8")


def test_beige_book_fixture_passes_landmark_check():
    from puremacro.narrative.sources import beige_book
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("beige_book", "html")
    assert_landmarks(
        text, source="beige_book",
        expected_version=beige_book.PARSER_SCHEMA_VERSION,
        landmarks=["Beige Book"],
    )


def test_eu_eurlex_fixture_passes_landmark_check():
    from puremacro.narrative.sources import eu_eurlex
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("eu_eurlex", "html")
    assert_landmarks(
        text, source="eu_eurlex",
        expected_version=eu_eurlex.PARSER_SCHEMA_VERSION,
        landmarks=["CELEX", "EUR-Lex"],
    )


def test_eu_parliament_fixture_passes_landmark_check():
    from puremacro.narrative.sources import eu_parliament
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("eu_parliament", "html")
    assert_landmarks(
        text, source="eu_parliament",
        expected_version=eu_parliament.PARSER_SCHEMA_VERSION,
        landmarks=["European Parliament", "plenary"],
    )


def test_us_cbo_fixture_passes_landmark_check():
    from puremacro.narrative.sources import us_cbo
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("us_cbo", "xml")
    assert_landmarks(
        text, source="us_cbo",
        expected_version=us_cbo.PARSER_SCHEMA_VERSION,
        landmarks=["<rss", "Congressional Budget Office"],
    )


def test_fed_minutes_fixture_passes_landmark_check():
    from puremacro.narrative.sources import fed_minutes
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("fed_minutes", "html")
    assert_landmarks(
        text, source="fed_minutes",
        expected_version=fed_minutes.PARSER_SCHEMA_VERSION,
        landmarks=["Federal Open Market Committee", "Minutes"],
    )


def test_fed_speeches_fixture_passes_landmark_check():
    from puremacro.narrative.sources import fed_speeches
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("fed_speeches", "html")
    assert_landmarks(
        text, source="fed_speeches",
        expected_version=fed_speeches.PARSER_SCHEMA_VERSION,
        landmarks=["Speeches", "Federal Reserve"],
    )


def test_bluesky_fixture_passes_landmark_check():
    from puremacro.narrative.sources import bluesky
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("bluesky", "json")
    assert_landmarks(
        text, source="bluesky",
        expected_version=bluesky.PARSER_SCHEMA_VERSION,
        landmarks=['"$type":', "app.bsky.feed.post"],
    )


def test_ecb_press_fixture_passes_landmark_check():
    from puremacro.narrative.sources import ecb_press
    from puremacro.narrative.sources._schema_check import assert_landmarks
    text = _fixture_text("ecb_press", "html")
    assert_landmarks(
        text, source="ecb_press",
        expected_version=ecb_press.PARSER_SCHEMA_VERSION,
        landmarks=["European Central Bank", "Press release"],
    )
