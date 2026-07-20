"""Slice (b): _TECH_DOMAIN_<lang> lexicon sanity checks."""
from __future__ import annotations

import pytest


def test_tech_domain_en_non_empty_and_typed():
    from puremacro.narrative.indices._lexicons import _TECH_DOMAIN_EN
    assert isinstance(_TECH_DOMAIN_EN, frozenset)
    assert len(_TECH_DOMAIN_EN) >= 50
    assert "ai" in _TECH_DOMAIN_EN
    assert "automation" in _TECH_DOMAIN_EN
    assert "robot" in _TECH_DOMAIN_EN or "robots" in _TECH_DOMAIN_EN


def test_tech_domain_en_disjoint_from_labor_domain():
    from puremacro.narrative.indices._lexicons import (
        _TECH_DOMAIN_EN, _LABOR_DOMAIN_EN,
    )
    overlap = _TECH_DOMAIN_EN & _LABOR_DOMAIN_EN
    assert overlap == frozenset(), f"unexpected overlap: {overlap}"


def test_tech_domain_en_disjoint_from_uncertainty_tone():
    from puremacro.narrative.indices._lexicons import (
        _TECH_DOMAIN_EN, _UNCERTAINTY_TONE_EN,
    )
    overlap = _TECH_DOMAIN_EN & _UNCERTAINTY_TONE_EN
    assert overlap == frozenset(), f"unexpected overlap: {overlap}"


LANGS_OTHER = ["es", "pt", "de", "fr", "it", "ja", "zh"]


@pytest.mark.parametrize("lang", LANGS_OTHER)
def test_tech_domain_other_languages_non_empty(lang):
    import importlib
    mod = importlib.import_module("puremacro.narrative.indices._lexicons")
    lex = getattr(mod, f"_TECH_DOMAIN_{lang.upper()}")
    assert isinstance(lex, frozenset)
    assert len(lex) >= 30, f"_TECH_DOMAIN_{lang.upper()} only has {len(lex)} terms"


@pytest.mark.parametrize("lang", LANGS_OTHER)
def test_tech_domain_disjoint_from_other_groups(lang):
    import importlib
    mod = importlib.import_module("puremacro.narrative.indices._lexicons")
    tech = getattr(mod, f"_TECH_DOMAIN_{lang.upper()}")
    labor = getattr(mod, f"_LABOR_DOMAIN_{lang.upper()}")
    tone = getattr(mod, f"_UNCERTAINTY_TONE_{lang.upper()}")
    assert tech & labor == frozenset()
    assert tech & tone == frozenset()


ALL_LANGS = ["en", "es", "pt", "de", "fr", "it", "ja", "zh"]


@pytest.mark.parametrize("lang", ALL_LANGS)
def test_lexicons_ltui_has_three_keys_per_language(lang):
    from puremacro.narrative.indices._lexicons import LEXICONS
    assert "ltui" in LEXICONS
    entry = LEXICONS["ltui"][lang]
    assert set(entry.keys()) == {"labor_domain", "uncertainty_tone", "tech_domain"}
    for k, v in entry.items():
        assert isinstance(v, frozenset), f"{lang}/{k} is {type(v)}"
        assert len(v) > 0
