"""Tests for Spanish macroeconomic lexicons and stance scoring."""
from __future__ import annotations

import pytest

from puremacro.narrative.scoring.lexicons_es import (
    SPANISH_MACRO_LEXICON,
    score_spanish_macro_sentiment,
)


def test_score_spanish_hawkish():
    text = (
        "La Junta de Gobierno decidió por unanimidad incrementar la tasa de interés objetivo "
        "ante la presencia de presiones inflacionarias y la necesidad de anclar las expectativas."
    )
    res = score_spanish_macro_sentiment(text)
    assert res.net_stance > 0.0
    assert res.hawkish_count >= 2
    assert res.dovish_count == 0
    assert "anclar las expectativas" in res.matched_hawkish or "presiones inflacionarias" in res.matched_hawkish


def test_score_spanish_dovish():
    text = (
        "El balance de riesgos para la actividad económica se encuentra sesgado a la baja, "
        "con una marcada desaceleración económica y espacio para un recorte de tasas."
    )
    res = score_spanish_macro_sentiment(text)
    assert res.net_stance < 0.0
    assert res.dovish_count >= 2
    assert res.hawkish_count == 0


def test_score_spanish_neutral():
    text = "El banco central sostuvo una reunión ordinaria de política monetaria en la sede institucional."
    res = score_spanish_macro_sentiment(text)
    assert res.net_stance == 0.0
    assert res.hawkish_count == 0
    assert res.dovish_count == 0
