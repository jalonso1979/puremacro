"""Tests for Shapley keyword and sentence explainability."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.scoring.explain import (
    shapley_keyword_attribution,
    explain_sentence_contributions,
)


def _toy_sentiment_scorer(text: str) -> float:
    text_l = text.lower()
    pos = sum(text_l.count(w) for w in ["inflation", "hiking", "tightening", "restrictive"])
    neg = sum(text_l.count(w) for w in ["cutting", "easing", "dovish", "slack"])
    return float(pos - neg)


def test_shapley_keyword_attribution():
    text = "The committee noted rising inflation and decided on hiking rates to maintain a restrictive stance."
    df = shapley_keyword_attribution(text, _toy_sentiment_scorer, top_k=5)
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "word" in df.columns
    assert "marginal_contribution" in df.columns
    assert "direction" in df.columns
    
    # "inflation", "hiking", "restrictive" should have positive contributions
    top_words = list(df["word"])
    assert any(w in top_words for w in ["inflation", "hiking", "restrictive"])


def test_explain_sentence_contributions():
    text = (
        "Economic activity expanded at a moderate pace. "
        "However, inflation remained elevated and above the target. "
        "Consequently, the board favored further policy tightening."
    )
    df = explain_sentence_contributions(text, _toy_sentiment_scorer, top_k=3)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 3
    assert "sentence" in df.columns
    assert "marginal_contribution" in df.columns
