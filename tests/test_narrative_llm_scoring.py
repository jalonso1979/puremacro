"""Tests for puremacro.narrative.scoring.llm timing extraction.

Uses a stub backend (no network calls) to feed canned LLM responses
through score_llm and inspect the resulting NarrativeEvent.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from puremacro.narrative.scoring.llm import score_llm


class _StubBackend:
    """Returns canned response strings in FIFO order. One call per text."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.model = "stub"
        self.max_tokens = 0
        self.temperature = 0.0

    def call(self, prompt: str) -> str:
        if not self._responses:
            raise RuntimeError("stub backend out of canned responses")
        return self._responses.pop(0)


def _one_event_response(**fields) -> str:
    """Build a JSON-array response string with one event dict."""
    base = {
        "magnitude_usd_bn": 50.0,
        "target": "consumption",
        "subtarget": "tax",
        "sign": +1,
        "confidence": 0.9,
        "excerpt": "test excerpt",
    }
    base.update(fields)
    return json.dumps([base])


def _one_text_input(date: str = "2010-01-15") -> list[tuple]:
    return [(pd.Timestamp(date), "press text", "https://example.org/x")]


def test_score_llm_extracts_implementation_date():
    """LLM returns implementation_date='2010-04-01' → profile points there."""
    stub = _StubBackend([_one_event_response(implementation_date="2010-04-01")])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == [(pd.Timestamp("2010-04-01"), 1.0)]


def test_score_llm_extracts_horizon_quarters_when_date_null():
    """implementation_date null + horizon_quarters=2, announced 2010-01-15 (Q1).
    Profile should land on 2010-07-01 (Q3, two quarters out)."""
    stub = _StubBackend([_one_event_response(
        implementation_date=None, horizon_quarters=2,
    )])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert events[0].implementation_profile == [(pd.Timestamp("2010-07-01"), 1.0)]


def test_score_llm_implementation_date_wins_when_both_present():
    """When LLM returns both, implementation_date wins; horizon_quarters ignored."""
    stub = _StubBackend([_one_event_response(
        implementation_date="2010-04-01", horizon_quarters=8,
    )])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert events[0].implementation_profile == [(pd.Timestamp("2010-04-01"), 1.0)]


def test_score_llm_default_empty_profile_when_both_null():
    """Both fields null → profile is empty list; effective_profile falls back."""
    stub = _StubBackend([_one_event_response(
        implementation_date=None, horizon_quarters=None,
    )])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert events[0].implementation_profile == []
    # Fallback semantics: effective_profile uses announcement date.
    assert events[0].effective_profile == [(pd.Timestamp("2010-01-15"), 1.0)]


def test_score_llm_skips_malformed_date_string():
    """Bad implementation_date string → profile empty; event still emitted."""
    stub = _StubBackend([_one_event_response(implementation_date="not a date")])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == []


def test_score_llm_skips_out_of_range_horizon():
    """horizon_quarters=20 (out of 0..16) → profile empty; event still emitted."""
    stub = _StubBackend([_one_event_response(
        implementation_date=None, horizon_quarters=20,
    )])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == []


def test_score_llm_rejects_implementation_predating_announcement():
    """implementation_date < announcement_date (Q comparison) → profile empty."""
    stub = _StubBackend([_one_event_response(implementation_date="2009-01-01")])
    events = score_llm(_one_text_input("2010-06-01"),
                       backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == []


def test_score_llm_legacy_response_without_timing_fields_works():
    """Old-schema response with NO timing keys → profile empty; event ships."""
    # Build a response missing both new fields entirely (not just null).
    legacy = json.dumps([{
        "magnitude_usd_bn": 50.0,
        "target": "consumption",
        "subtarget": "tax",
        "sign": +1,
        "confidence": 0.9,
        "excerpt": "legacy",
    }])
    stub = _StubBackend([legacy])
    events = score_llm(_one_text_input("2010-01-15"),
                       backend=stub, country="USA")
    assert len(events) == 1
    assert events[0].implementation_profile == []
