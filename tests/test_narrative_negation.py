"""Sprint-1 / B2: negation-aware keyword counting."""
from __future__ import annotations

import pandas as pd
import pytest


def test_count_keywords_default_off_keeps_existing_behavior():
    from puremacro.narrative.indices._kernels import count_keywords
    text = "Inflation is not uncertain."
    terms = frozenset({"uncertain"})
    # Default: negation=False → counts the hit.
    assert count_keywords(text, terms, language="en") == 1


def test_count_keywords_negation_suppresses_after_not():
    from puremacro.narrative.indices._kernels import count_keywords
    text = "Inflation is not uncertain."
    terms = frozenset({"uncertain"})
    assert count_keywords(text, terms, language="en", negation=True) == 0


def test_count_keywords_negation_handles_multiple_negators():
    from puremacro.narrative.indices._kernels import count_keywords
    text = "The outlook is neither weak nor uncertain."
    terms = frozenset({"weak", "uncertain"})
    # 'neither' and 'nor' both fire on respective terms.
    assert count_keywords(text, terms, language="en", negation=True) == 0


def test_count_keywords_negation_keeps_distant_hits():
    """A negator that's > 4 tokens away does NOT suppress."""
    from puremacro.narrative.indices._kernels import count_keywords
    text = "We are not yet ready to call this anything. The outlook is uncertain."
    terms = frozenset({"uncertain"})
    assert count_keywords(text, terms, language="en", negation=True) == 1


def test_count_keywords_negation_spanish():
    from puremacro.narrative.indices._kernels import count_keywords
    text = "La inflación no es incierta."
    terms = frozenset({"incierta"})
    assert count_keywords(text, terms, language="es", negation=True) == 0


def test_count_keywords_negation_german():
    from puremacro.narrative.indices._kernels import count_keywords
    text = "Die Inflation ist nicht unsicher."
    terms = frozenset({"unsicher"})
    assert count_keywords(text, terms, language="de", negation=True) == 0


def test_count_keywords_negation_japanese_prepositive():
    """Japanese has both pre- and post-posed negation; we only handle pre-
    (e.g., 不 prefix). Postposed ない is a known limitation."""
    from puremacro.narrative.indices._kernels import count_keywords
    text = "不確実な見通しはありません。"  # uncertain outlook (no negator before)
    terms = frozenset({"不確実"})
    # No preceding negator → hit kept.
    assert count_keywords(text, terms, language="ja", negation=True) == 1


def test_lui_negation_changes_score_on_negated_uncertainty():
    """LUI sentence has labor + (uncertainty | negated-uncertainty);
    enabling negation removes the negated case."""
    from puremacro.narrative.indices import lui
    # Use 'wages' (labor) + 'uncertain' (uncertainty-tone), not phrase
    # shortcuts. Negator 'not' is 1 token before 'uncertain'.
    text = "Output is steady. Wages are not uncertain this quarter."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    s_neg = lui(records, country="USA", language="en",
                normalize="raw", negation=True)
    s_no  = lui(records, country="USA", language="en",
                normalize="raw", negation=False)
    val_neg = float(s_neg.series.dropna().iloc[0])
    val_no  = float(s_no.series.dropna().iloc[0])
    # Without negation: sentence 2 matches (wages + uncertain) → 0.5.
    # With negation: 'uncertain' suppressed by 'not' → sentence drops → 0.0.
    assert val_no == pytest.approx(0.5)
    assert val_neg == 0.0


def test_negation_window_not_too_wide():
    """A negator more than 4 tokens before the hit should not fire."""
    from puremacro.narrative.indices._kernels import count_keywords
    # 'not' is 5 tokens before 'uncertain'.
    text = "I do not at this present moment feel the outlook is uncertain."
    terms = frozenset({"uncertain"})
    # Window=4 → 'not' is too far → hit kept.
    assert count_keywords(text, terms, language="en", negation=True) == 1
