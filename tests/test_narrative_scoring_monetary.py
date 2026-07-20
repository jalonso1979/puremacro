"""Tests for monetary-kind keyword + LLM scoring (Slice 1)."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.scoring import score_keyword


def _records(*pairs):
    """Build (date, text, url) records."""
    return [(pd.Timestamp(d), t, "https://test/" + d) for d, t in pairs]


def test_keyword_monetary_hawkish_signal():
    """A clear rate-hike signal should yield a +1 (hawkish) monetary event."""
    records = _records(
        ("2022-03-16", "The FOMC voted to raise the federal funds rate by 25 basis points."),
    )
    events = score_keyword(records, kind="monetary", country="USA")
    assert len(events) == 1
    e = events[0]
    assert e.kind == "monetary"
    assert e.target == "policy_rate"
    assert e.sign == +1
    assert e.country == "USA"


def test_keyword_monetary_past_tense_hike():
    """Past-tense inflections must match (real FOMC opener phrasing)."""
    records = _records(
        ("2022-05-04", "The Committee raised the target range for the federal funds rate by 50 basis points."),
        ("2007-09-18", "The FOMC lowered the federal funds rate by 50 basis points."),
        ("2008-12-16", "The Committee cut the target range to 0 to 1/4 percent."),
    )
    events = score_keyword(records, kind="monetary", country="USA")
    assert len(events) == 3
    assert events[0].sign == +1   # raised
    assert events[1].sign == -1   # lowered
    assert events[2].sign == -1   # cut


def test_keyword_monetary_dovish_signal():
    records = _records(
        ("2020-03-15", "The Fed announced a rate cut and asset-purchase expansion."),
    )
    events = score_keyword(records, kind="monetary", country="USA")
    assert len(events) >= 1
    assert events[0].sign == -1
    assert events[0].kind == "monetary"


def test_keyword_monetary_no_signal():
    records = _records(
        ("2020-03-15", "Inflation remained elevated through the quarter."),
    )
    events = score_keyword(records, kind="monetary", country="USA")
    assert events == []


def test_keyword_fiscal_kind_unchanged():
    """Existing fiscal-keyword path must still work with kind='fiscal' default."""
    records = _records(
        ("2020-04-01", "Congress approved a $500 billion infrastructure package."),
    )
    events = score_keyword(records, country="USA")  # kind defaults fiscal
    assert len(events) == 1
    assert events[0].kind == "fiscal"
    assert events[0].target == "investment"


def test_keyword_invalid_kind_raises():
    records = _records(("2020-01-01", "anything"))
    with pytest.raises(ValueError, match="kind"):
        score_keyword(records, kind="not_a_kind", country="USA")


# ---------------------------------------------------------------------------
# LLM prompt-dispatch tests (no API key required — uses dry_run)
# ---------------------------------------------------------------------------
from puremacro.narrative.scoring.llm import (
    _PROMPTS, _build_prompt, score_llm,
)


def test_prompt_registry_has_five_kinds():
    assert set(_PROMPTS) == {"fiscal", "monetary", "macropru", "fx", "structural"}


def test_build_prompt_fiscal_contains_legacy_text():
    p = _build_prompt(kind="fiscal", language="en", country="USA",
                      date="2020-01-01", text="hello")
    assert "fiscal-policy events" in p
    assert "USA" in p
    assert "hello" in p


def test_build_prompt_monetary_contains_bps_and_hawkish_dovish():
    p = _build_prompt(kind="monetary", language="en", country="USA",
                      date="2022-03-16", text="rate hike")
    assert "basis points" in p.lower() or "bps" in p.lower()
    assert "hawkish" in p.lower()
    assert "dovish" in p.lower()


def test_build_prompt_includes_language_hint_for_non_english():
    p = _build_prompt(kind="fiscal", language="es", country="MEX",
                      date="2020-01-01", text="hola")
    assert "es" in p or "language" in p.lower()


def test_score_llm_dry_run_returns_empty_list():
    """Dry run should not call the network; just print cost estimate."""
    records = [(pd.Timestamp("2020-01-01"), "x", "u")]
    out = score_llm(records, backend=None, kind="fiscal", dry_run=True)
    assert out == []


def test_score_llm_invalid_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        score_llm([], backend=None, kind="not_a_kind", dry_run=True)


def test_score_llm_accepts_4_tuple_records():
    """Backwards-compat: 4-tuple SourceRecord must work alongside 3-tuple."""
    records = [
        (pd.Timestamp("2020-01-01"), "x", "u", {"doctype": "decision", "language": "en"}),
        (pd.Timestamp("2020-02-01"), "y", "u"),
    ]
    out = score_llm(records, backend=None, kind="fiscal", dry_run=True)
    assert out == []
