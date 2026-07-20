"""Slice (b): triple-co-occurrence kernel + paragraph splitter."""
from __future__ import annotations

import pandas as pd
import pytest


def test_split_paragraphs_latin_blank_line():
    from puremacro.narrative.indices._kernels import _split_paragraphs
    text = "First paragraph.\nStill first.\n\nSecond paragraph.\n\n  \nThird paragraph."
    parts = _split_paragraphs(text, language="en")
    assert parts == [
        "First paragraph.\nStill first.",
        "Second paragraph.",
        "Third paragraph.",
    ]


def test_split_paragraphs_cjk_double_newline():
    from puremacro.narrative.indices._kernels import _split_paragraphs
    text = "第一段。続きあり。\n\n第二段です。"
    parts = _split_paragraphs(text, language="ja")
    assert parts == ["第一段。続きあり。", "第二段です。"]


def test_split_paragraphs_no_break_returns_whole_text():
    from puremacro.narrative.indices._kernels import _split_paragraphs
    text = "Just one paragraph with no blank lines."
    parts = _split_paragraphs(text, language="en")
    assert parts == ["Just one paragraph with no blank lines."]


def test_split_paragraphs_empty_returns_empty_list():
    from puremacro.narrative.indices._kernels import _split_paragraphs
    assert _split_paragraphs("", language="en") == []
    assert _split_paragraphs("   \n  \n   ", language="en") == []


def test_triple_cooccurrence_three_groups_sentence_window():
    from puremacro.narrative.indices._kernels import triple_cooccurrence_kernel
    text = (
        "Workers face displacement from AI automation.   "  # all three groups
        "The weather is nice today."                        # none
    )
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    out = list(triple_cooccurrence_kernel(
        records,
        term_groups=[
            frozenset({"workers", "jobs"}),
            frozenset({"displacement", "uncertainty"}),
            frozenset({"ai", "automation"}),
        ],
        weight_group_idx=None,
        window="sentence",
        language="en",
    ))
    assert len(out) == 1
    date, score = out[0]
    assert score == pytest.approx(0.5)  # 1 of 2 sentences matched


def test_triple_cooccurrence_two_groups_matches_sentence_cooccurrence():
    """K=2 with weight_group_idx=None should match sentence_cooccurrence_kernel
    (with phrases=None) on the same input."""
    from puremacro.narrative.indices._kernels import (
        triple_cooccurrence_kernel,
        sentence_cooccurrence_kernel,
    )
    records = [(pd.Timestamp("2024-01-01"),
                "Employment is uncertain. The sky is blue.",
                "url", {"language": "en"})]
    groups = [frozenset({"employment"}), frozenset({"uncertain"})]
    a = list(triple_cooccurrence_kernel(
        records, term_groups=groups, weight_group_idx=None,
        window="sentence", language="en",
    ))
    b = list(sentence_cooccurrence_kernel(
        records, term_groups=groups, phrases=None, language="en",
    ))
    assert a == b


def test_triple_cooccurrence_no_match_yields_zero():
    from puremacro.narrative.indices._kernels import triple_cooccurrence_kernel
    records = [(pd.Timestamp("2024-01-01"),
                "Nothing matches at all.", "url", {"language": "en"})]
    out = list(triple_cooccurrence_kernel(
        records,
        term_groups=[frozenset({"workers"}), frozenset({"risk"}), frozenset({"ai"})],
        weight_group_idx=None, window="sentence", language="en",
    ))
    assert out[0][1] == 0.0


def test_triple_cooccurrence_empty_doc_yields_zero():
    from puremacro.narrative.indices._kernels import triple_cooccurrence_kernel
    out = list(triple_cooccurrence_kernel(
        [(pd.Timestamp("2024-01-01"), "", "url", {"language": "en"})],
        term_groups=[frozenset({"a"}), frozenset({"b"}), frozenset({"c"})],
        weight_group_idx=None, window="sentence", language="en",
    ))
    assert out[0][1] == 0.0


def test_triple_cooccurrence_paragraph_window_treats_paragraph_as_unit():
    """Paragraph window: all-three-groups must coexist within one paragraph,
    not just within the document overall."""
    from puremacro.narrative.indices._kernels import triple_cooccurrence_kernel
    text = (
        "Workers face displacement from automation in this paragraph.\n\n"
        "The economy expanded last quarter."
    )
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    out = list(triple_cooccurrence_kernel(
        records,
        term_groups=[
            frozenset({"workers"}),
            frozenset({"displacement"}),
            frozenset({"automation"}),
        ],
        weight_group_idx=None,
        window="paragraph",
        language="en",
    ))
    assert out[0][1] == pytest.approx(0.5)  # 1 of 2 paragraphs


def test_triple_cooccurrence_paragraph_window_no_match_when_split():
    """If labor+uncertainty+tech are spread across paragraphs (not co-located),
    no paragraph matches and the score is 0."""
    from puremacro.narrative.indices._kernels import triple_cooccurrence_kernel
    text = (
        "Workers across many sectors are watched closely.\n\n"
        "Displacement risks rise.\n\n"
        "Automation is advancing quickly."
    )
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    out = list(triple_cooccurrence_kernel(
        records,
        term_groups=[
            frozenset({"workers"}),
            frozenset({"displacement"}),
            frozenset({"automation"}),
        ],
        weight_group_idx=None,
        window="paragraph",
        language="en",
    ))
    assert out[0][1] == 0.0


def test_triple_cooccurrence_density_weighting_increases_with_more_hits():
    from puremacro.narrative.indices._kernels import triple_cooccurrence_kernel
    text_a = "Workers face displacement from ai. " + ("filler " * 17)
    text_b = "Workers face displacement from ai automation digital. " + ("filler " * 13)
    base_kwargs = dict(
        term_groups=[
            frozenset({"workers"}),
            frozenset({"displacement"}),
            frozenset({"ai", "automation", "digital"}),
        ],
        weight_group_idx=2,
        window="paragraph",
        language="en",
    )
    records_a = [(pd.Timestamp("2024-01-01"), text_a, "url", {"language": "en"})]
    records_b = [(pd.Timestamp("2024-01-01"), text_b, "url", {"language": "en"})]
    [(_, s_a)] = list(triple_cooccurrence_kernel(records_a, **base_kwargs))
    [(_, s_b)] = list(triple_cooccurrence_kernel(records_b, **base_kwargs))
    assert s_b > s_a


def test_triple_cooccurrence_weight_group_idx_none_is_unweighted():
    from puremacro.narrative.indices._kernels import triple_cooccurrence_kernel
    text = "Workers face displacement from ai automation. " + ("filler " * 15)
    records = [(pd.Timestamp("2024-01-01"), text, "url", {"language": "en"})]
    common = dict(
        term_groups=[
            frozenset({"workers"}),
            frozenset({"displacement"}),
            frozenset({"ai", "automation"}),
        ],
        window="paragraph", language="en",
    )
    [(_, unw)] = list(triple_cooccurrence_kernel(
        records, weight_group_idx=None, **common,
    ))
    assert unw == pytest.approx(1.0)


def test_triple_cooccurrence_invalid_weight_group_idx_raises():
    from puremacro.narrative.indices._kernels import triple_cooccurrence_kernel
    records = [(pd.Timestamp("2024-01-01"), "x", "url", {"language": "en"})]
    with pytest.raises(ValueError, match="weight_group_idx=3"):
        list(triple_cooccurrence_kernel(
            records,
            term_groups=[frozenset({"a"}), frozenset({"b"}), frozenset({"c"})],
            weight_group_idx=3, window="paragraph", language="en",
        ))


def test_triple_cooccurrence_invalid_window_raises():
    from puremacro.narrative.indices._kernels import triple_cooccurrence_kernel
    records = [(pd.Timestamp("2024-01-01"), "x", "url", {"language": "en"})]
    with pytest.raises(ValueError, match="window must be"):
        list(triple_cooccurrence_kernel(
            records,
            term_groups=[frozenset({"a"}), frozenset({"b"})],
            window="document", language="en",
        ))
