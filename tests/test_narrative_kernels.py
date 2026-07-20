"""Unit tests for sentence splitter, sentence_cooccurrence_kernel,
and length_normalize on keyword_count_kernel (Slice 6a)."""
from __future__ import annotations
import pandas as pd


# -----------------------------------------------------------------------------
# _split_sentences
# -----------------------------------------------------------------------------

def test_split_sentences_english_basic():
    from puremacro.narrative.indices._kernels import _split_sentences
    out = _split_sentences("First sentence. Second sentence! Third?", "en")
    assert len(out) == 3
    assert out[0].startswith("First")
    assert out[1].startswith("Second")
    assert out[2].startswith("Third")


def test_split_sentences_english_empty():
    from puremacro.narrative.indices._kernels import _split_sentences
    assert _split_sentences("", "en") == []


def test_split_sentences_english_no_punctuation():
    """A doc with no boundary punctuation is treated as one sentence."""
    from puremacro.narrative.indices._kernels import _split_sentences
    out = _split_sentences("just one sentence with no end mark", "en")
    assert len(out) == 1


def test_split_sentences_chinese():
    from puremacro.narrative.indices._kernels import _split_sentences
    out = _split_sentences("第一句话。第二句话！第三句话？", "zh")
    assert len(out) == 3


def test_split_sentences_japanese():
    from puremacro.narrative.indices._kernels import _split_sentences
    out = _split_sentences("第一文。第二文！第三文？", "ja")
    assert len(out) == 3


# -----------------------------------------------------------------------------
# sentence_cooccurrence_kernel
# -----------------------------------------------------------------------------

def test_sentence_cooccurrence_all_match():
    """Every sentence has both labor + uncertainty term → score 1.0."""
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    text = "Employment risk has risen. The labor market is weak."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {})]
    labor = frozenset({"employment", "labor market"})
    unc = frozenset({"risk", "weak"})
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[labor, unc], language="en",
    ))
    assert len(out) == 1
    assert out[0][1] == 1.0


def test_sentence_cooccurrence_partial_match():
    """Only one of two sentences has co-occurrence → score 0.5.

    Note: text is crafted so S1 has labor+uncertainty co-occurrence and
    S2 has only an uncertainty term without a matching labor term.
    """
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    text = "Employment risk grew strongly last quarter. Wages rose."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {})]
    labor = frozenset({"employment", "labor market"})
    unc = frozenset({"risk"})
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[labor, unc], language="en",
    ))
    assert out[0][1] == 0.5


def test_sentence_cooccurrence_no_match():
    """Labor terms appear but no uncertainty in same sentence → 0.0."""
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    text = "Employment grew. Wages rose."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {})]
    labor = frozenset({"employment", "wages"})
    unc = frozenset({"risk", "uncertain"})
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[labor, unc], language="en",
    ))
    assert out[0][1] == 0.0


def test_sentence_cooccurrence_phrase_shortcut():
    """A sentence with a curated phrase matches even without co-occurrence."""
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    text = "Rising unemployment was discussed. Employment is fine."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {})]
    labor = frozenset({"employment"})  # the second sentence has labor
    unc = frozenset({"risk"})           # but no uncertainty term
    phrases = frozenset({"rising unemployment"})
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[labor, unc], phrases=phrases, language="en",
    ))
    # First sentence matches via phrase (rising unemployment).
    # Second sentence: has "employment" but no uncertainty → no match.
    # Score = 1/2 = 0.5
    assert out[0][1] == 0.5


def test_sentence_cooccurrence_empty_doc():
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    records = [(pd.Timestamp("2024-01-01"), "", "url", {})]
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[frozenset({"x"}), frozenset({"y"})],
        language="en",
    ))
    assert out[0][1] == 0.0


def test_sentence_cooccurrence_multiple_groups():
    """Generalized to >2 groups (e.g. labor ∩ uncertainty ∩ time-ref)."""
    from puremacro.narrative.indices._kernels import sentence_cooccurrence_kernel
    text = "Employment risk rose this quarter."
    records = [(pd.Timestamp("2024-01-01"), text, "url", {})]
    g1 = frozenset({"employment"})
    g2 = frozenset({"risk"})
    g3 = frozenset({"quarter"})
    out = list(sentence_cooccurrence_kernel(
        records, term_groups=[g1, g2, g3], language="en",
    ))
    assert out[0][1] == 1.0


# -----------------------------------------------------------------------------
# keyword_count_kernel — length normalization
# -----------------------------------------------------------------------------

def test_keyword_count_length_normalize_doubles_text_halves_score():
    """Doubling text length with same hits halves the per-1000-word score."""
    from puremacro.narrative.indices._kernels import keyword_count_kernel
    text_short = "uncertainty " * 5 + "the " * 95     # 100 words, 5 hits
    text_long = "uncertainty " * 5 + "the " * 195    # 200 words, 5 hits
    records_short = [(pd.Timestamp("2024-01-01"), text_short, "u", {})]
    records_long = [(pd.Timestamp("2024-01-01"), text_long, "u", {})]
    terms = frozenset({"uncertainty"})
    s_short = list(keyword_count_kernel(
        records_short, terms=terms, language="en", length_normalize=True,
    ))[0][1]
    s_long = list(keyword_count_kernel(
        records_long, terms=terms, language="en", length_normalize=True,
    ))[0][1]
    assert abs(s_short - 50.0) < 1.0   # 5/100 * 1000 = 50
    assert abs(s_long - 25.0) < 1.0    # 5/200 * 1000 = 25


def test_keyword_count_length_normalize_default_off():
    """Default behavior unchanged: returns raw hit count."""
    from puremacro.narrative.indices._kernels import keyword_count_kernel
    text = "uncertainty " * 5 + "x " * 95
    records = [(pd.Timestamp("2024-01-01"), text, "u", {})]
    terms = frozenset({"uncertainty"})
    score = list(keyword_count_kernel(records, terms=terms, language="en"))[0][1]
    assert score == 5.0


def test_keyword_count_length_normalize_empty_doc():
    """Empty doc with length_normalize → 0.0 (no division by zero)."""
    from puremacro.narrative.indices._kernels import keyword_count_kernel
    records = [(pd.Timestamp("2024-01-01"), "", "u", {})]
    score = list(keyword_count_kernel(
        records, terms=frozenset({"x"}), language="en", length_normalize=True,
    ))[0][1]
    assert score == 0.0
