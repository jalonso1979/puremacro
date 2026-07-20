"""Slice 6a: focused integration tests on the new LUI methodology."""
from __future__ import annotations
import pandas as pd


def test_lui_basic_sentence_cooccurrence_signal():
    """Doc with one labor-uncertainty sentence + one neutral sentence
    should score ~0.5."""
    from puremacro.narrative.indices import lui
    text = "Employment risk has risen materially. The economy expanded."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = lui(records, country="USA", language="en", normalize="raw")
    # Quarterly aggregation of one record gets that record's score.
    val = float(series.series.dropna().iloc[0])
    assert 0.4 < val < 0.6   # ~0.5


def test_lui_phrase_shortcut():
    """A doc whose only matching sentence has a phrase but no separate
    labor + uncertainty co-occurrence still gets credit."""
    from puremacro.narrative.indices import lui
    text = "Rising unemployment is a concern."  # phrase only
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = lui(records, country="USA", language="en", normalize="raw")
    val = float(series.series.dropna().iloc[0])
    assert val == 1.0


def test_lui_pure_labor_discussion_scores_low():
    """A doc that talks about labor positively (no uncertainty markers)
    should score 0."""
    from puremacro.narrative.indices import lui
    text = "Employment grew strongly. Wages rose by 4 percent. Hiring continued."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = lui(records, country="USA", language="en", normalize="raw")
    val = float(series.series.dropna().iloc[0])
    assert val == 0.0


def test_lui_lexicon_override_accepts_dict():
    """The lexicon kwarg accepts the dict-of-frozensets shape."""
    from puremacro.narrative.indices import lui
    custom = {
        "labor_domain": frozenset({"workers"}),
        "uncertainty_tone": frozenset({"struggle"}),
        "phrases": frozenset({"job pain"}),
    }
    text = "Workers struggle right now. Job pain is widespread."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    series = lui(records, country="USA", language="en",
                 lexicon=custom, normalize="raw")
    val = float(series.series.dropna().iloc[0])
    assert val == 1.0  # both sentences match
