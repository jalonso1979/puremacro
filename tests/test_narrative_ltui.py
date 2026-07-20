"""Slice (b): ltui() helper smoke tests."""
from __future__ import annotations

import pandas as pd
import pytest


def test_ltui_three_way_match_scores_positive():
    """A doc with labor + uncertainty + tech in a single paragraph scores > 0."""
    from puremacro.narrative.indices import ltui
    text = "Workers face displacement from rapid ai automation. Uncertainty about jobs is rising."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = ltui(records, country="USA", language="en", normalize="raw")
    val = float(series.series.dropna().iloc[0])
    assert val > 0.0


def test_ltui_zero_when_no_tech():
    """LUI-style labor + uncertainty without any tech term: LTUI = 0."""
    from puremacro.narrative.indices import ltui
    text = "Unemployment is rising and the labor market is weakening."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = ltui(records, country="USA", language="en", normalize="raw")
    val = float(series.series.dropna().iloc[0])
    assert val == 0.0


def test_ltui_zero_when_only_tech_without_labor_uncertainty():
    """Tech adoption news with no labor + uncertainty co-occurrence: LTUI = 0."""
    from puremacro.narrative.indices import ltui
    text = "AI adoption has accelerated. Cloud computing investments grew."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = ltui(records, country="USA", language="en", normalize="raw")
    val = float(series.series.dropna().iloc[0])
    assert val == 0.0


def test_ltui_lexicon_override_accepts_dict():
    from puremacro.narrative.indices import ltui
    custom = {
        "labor_domain": frozenset({"workers"}),
        "uncertainty_tone": frozenset({"struggle"}),
        "tech_domain": frozenset({"bots"}),
    }
    text = "Workers struggle as bots take their jobs."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = ltui(records, country="USA", language="en",
                  lexicon=custom, normalize="raw")
    val = float(series.series.dropna().iloc[0])
    assert val > 0.0


def test_ltui_metadata_includes_index_name():
    from puremacro.narrative.indices import ltui
    records = [(pd.Timestamp("2024-01-01"),
                "Workers face automation displacement.",
                "url", {"language": "en"})]
    series = ltui(records, country="USA", language="en", normalize="raw")
    assert series.metadata["index"] == "ltui"
