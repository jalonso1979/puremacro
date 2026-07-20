"""Slice (e): LWUI helper + war lexicon checks."""
from __future__ import annotations

import importlib

import pandas as pd
import pytest


ALL_LANGS = ["en", "es", "pt", "de", "fr", "it", "ja", "zh"]


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_war_domain_lexicons_non_empty(lang):
    mod = importlib.import_module("puremacro.narrative.indices._lexicons")
    war = getattr(mod, f"_WAR_DOMAIN_{lang.upper()}")
    assert isinstance(war, frozenset) and len(war) >= 25


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_war_domain_disjoint_from_labor_and_tone(lang):
    mod = importlib.import_module("puremacro.narrative.indices._lexicons")
    war = getattr(mod, f"_WAR_DOMAIN_{lang.upper()}")
    labor = getattr(mod, f"_LABOR_DOMAIN_{lang.upper()}")
    tone = getattr(mod, f"_UNCERTAINTY_TONE_{lang.upper()}")
    assert war & labor == frozenset()
    assert war & tone == frozenset()


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_lexicons_lwui_registry(lang):
    from puremacro.narrative.indices._lexicons import LEXICONS
    assert "lwui" in LEXICONS
    entry = LEXICONS["lwui"][lang]
    assert set(entry.keys()) == {"labor_domain", "uncertainty_tone", "war_domain"}
    for v in entry.values():
        assert isinstance(v, frozenset) and len(v) > 0


def test_lwui_three_way_match_scores_positive():
    from puremacro.narrative.indices import lwui
    text = (
        "Workers in occupied territories face uncertain risk of displacement. "
        "The war economy and conscription drain the labor force."
    )
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    s = lwui(records, country="USA", language="en", normalize="raw")
    assert float(s.series.dropna().iloc[0]) > 0.0


def test_lwui_zero_when_no_war_terms():
    from puremacro.narrative.indices import lwui
    text = "Workers face uncertainty about wages and employment."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    s = lwui(records, country="USA", language="en", normalize="raw")
    assert float(s.series.dropna().iloc[0]) == 0.0


def test_lwui_metadata_index_name():
    from puremacro.narrative.indices import lwui
    text = "Workers face conscription risk in the war economy."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    s = lwui(records, country="USA", language="en", normalize="raw")
    assert s.metadata["index"] == "lwui"
