"""Slice (c): ltui_up / ltui_down + upside/downside lexicon checks."""
from __future__ import annotations

import importlib

import pandas as pd
import pytest


ALL_LANGS = ["en", "es", "pt", "de", "fr", "it", "ja", "zh"]


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_upside_downside_lexicons_non_empty(lang):
    mod = importlib.import_module("puremacro.narrative.indices._lexicons")
    up = getattr(mod, f"_TECH_LABOR_UPSIDE_{lang.upper()}")
    down = getattr(mod, f"_TECH_LABOR_DOWNSIDE_{lang.upper()}")
    assert isinstance(up, frozenset) and len(up) >= 10
    assert isinstance(down, frozenset) and len(down) >= 10


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_upside_downside_disjoint(lang):
    """A term must not legitimately fall in both upside and downside."""
    mod = importlib.import_module("puremacro.narrative.indices._lexicons")
    up = getattr(mod, f"_TECH_LABOR_UPSIDE_{lang.upper()}")
    down = getattr(mod, f"_TECH_LABOR_DOWNSIDE_{lang.upper()}")
    overlap = up & down
    assert overlap == frozenset(), f"upside/downside overlap [{lang}]: {overlap}"


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_lexicons_ltui_up_down_registry(lang):
    from puremacro.narrative.indices._lexicons import LEXICONS
    for key in ("ltui_up", "ltui_down"):
        assert key in LEXICONS
        entry = LEXICONS[key][lang]
        assert set(entry.keys()) == {
            "labor_domain", "uncertainty_tone", "tech_domain", "polarity",
        }
        for v in entry.values():
            assert isinstance(v, frozenset) and len(v) > 0


def test_ltui_up_scores_positive_on_upside_narrative():
    from puremacro.narrative.indices import ltui_up
    text = (
        "Workers face uncertainty as ai automation rolls out. "
        "But productivity gains from augmentation will lift wages."
    )
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    s = ltui_up(records, country="USA", language="en", normalize="raw")
    assert float(s.series.dropna().iloc[0]) > 0.0


def test_ltui_down_scores_positive_on_downside_narrative():
    from puremacro.narrative.indices import ltui_down
    text = (
        "Workers face uncertain risk of displacement from ai automation. "
        "Many jobs become obsolete and layoffs follow."
    )
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    s = ltui_down(records, country="USA", language="en", normalize="raw")
    assert float(s.series.dropna().iloc[0]) > 0.0


def test_ltui_up_zero_on_downside_only():
    from puremacro.narrative.indices import ltui_up
    text = "Workers face uncertain risk of displacement from ai. Jobs become obsolete."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    s = ltui_up(records, country="USA", language="en", normalize="raw")
    assert float(s.series.dropna().iloc[0]) == 0.0


def test_ltui_metadata_distinguishes_subindex():
    from puremacro.narrative.indices import ltui_up, ltui_down
    text = "Workers, automation, displacement, productivity gains."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    s_up = ltui_up(records, country="USA", language="en", normalize="raw")
    s_dn = ltui_down(records, country="USA", language="en", normalize="raw")
    assert s_up.metadata["index"] == "ltui_up"
    assert s_dn.metadata["index"] == "ltui_down"
