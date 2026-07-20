"""Cluster B (NLP modernization) — kernel-level unit tests.

Each kernel is tested with a synthetic / mocked dependency so the
tests are network- and ML-stack-free.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# B3 — embedding_similarity_kernel
# ---------------------------------------------------------------------------
def _synthetic_embed(texts):
    """Tiny hand-crafted embedder: 3-D vectors keyed on word-boundary
    keyword presence. Word-bounded to avoid false hits like 'uncert*ai*n'
    matching 'ai'."""
    import re
    def _has_word(tl, *words):
        return any(re.search(rf"\b{w}\b", tl) for w in words)
    out = []
    for t in texts:
        tl = (t or "").lower()
        out.append([
            1.0 if _has_word(tl, "uncertain", "concern", "concerns") else 0.0,
            1.0 if _has_word(tl, "labor", "labour", "employment") else 0.0,
            1.0 if _has_word(tl, "automation", "ai", "robot") else 0.0,
        ])
    return np.asarray(out, dtype=float)


def test_embedding_similarity_kernel_prototype_match():
    from puremacro.narrative.indices import embedding_similarity_kernel
    seeds = ["labor concern uncertain"]  # vector [1, 1, 0]
    records = [
        (pd.Timestamp("2024-01-01"), "employment concerns are rising", "u", {}),  # [1, 1, 0] → cos = 1
        (pd.Timestamp("2024-01-02"), "the weather is nice today",       "u", {}),  # [0, 0, 0] → cos = 0
    ]
    out = list(embedding_similarity_kernel(
        records, seed_prototype=seeds, embed_fn=_synthetic_embed,
    ))
    assert len(out) == 2
    assert out[0][1] == pytest.approx(1.0)
    assert out[1][1] == pytest.approx(0.0)


def test_embedding_similarity_kernel_handles_empty_doc():
    from puremacro.narrative.indices import embedding_similarity_kernel
    records = [(pd.Timestamp("2024-01-01"), "", "u", {})]
    out = list(embedding_similarity_kernel(
        records, seed_prototype=["uncertainty"], embed_fn=_synthetic_embed,
    ))
    assert out[0][1] == 0.0


def test_embedding_similarity_kernel_accepts_precomputed_prototype():
    from puremacro.narrative.indices import embedding_similarity_kernel
    proto = np.array([1.0, 1.0, 0.0])
    records = [(pd.Timestamp("2024-01-01"), "labor uncertain", "u", {})]
    out = list(embedding_similarity_kernel(
        records, seed_prototype=proto, embed_fn=_synthetic_embed,
    ))
    assert out[0][1] == pytest.approx(1.0)


def test_build_seed_prototype_strips_empty_seeds():
    from puremacro.narrative.indices import build_seed_prototype
    proto = build_seed_prototype(
        ["uncertain labor", "", "  "], embed_fn=_synthetic_embed,
    )
    assert proto.shape == (3,)
    # 1 valid seed, so prototype == its embedding
    np.testing.assert_array_equal(proto, np.array([1.0, 1.0, 0.0]))


def test_build_seed_prototype_raises_on_all_empty():
    from puremacro.narrative.indices import build_seed_prototype
    with pytest.raises(ValueError, match="empty"):
        build_seed_prototype(["", "  ", "\n"], embed_fn=_synthetic_embed)


def test_make_sentence_transformer_embedder_lazy_imports():
    from puremacro.narrative.indices import make_sentence_transformer_embedder
    # If sentence-transformers IS installed, ensure factory returns a callable.
    # If it isn't installed, expect ImportError with a useful message.
    try:
        embedder = make_sentence_transformer_embedder()
        assert callable(embedder)
    except ImportError as e:
        assert "sentence-transformers" in str(e).lower()


# ---------------------------------------------------------------------------
# B1 — mnl_kernel
# ---------------------------------------------------------------------------
def test_mnl_kernel_dict_of_dicts_weights():
    from puremacro.narrative.indices import mnl_kernel
    # 2 categories: 'uncertain' and 'neutral'.
    # 'concern' pulls toward uncertain; 'growth' pulls toward neutral.
    weights = {
        "beta": {
            "concern": {"uncertain": 4.0, "neutral": 0.0},
            "growth":  {"uncertain": 0.0, "neutral": 4.0},
        },
        "bias": {"uncertain": 0.0, "neutral": 0.0},
    }
    records = [
        (pd.Timestamp("2024-01-01"), "Inflation concern dominates the outlook.", "u", {}),
        (pd.Timestamp("2024-01-02"), "Strong growth is broad-based.", "u", {}),
    ]
    out = list(mnl_kernel(records, weights=weights, category="uncertain",
                          window="sentence"))
    assert out[0][1] > 0.9   # 'concern' fires strongly → P(uncertain) ≈ exp(4)/(exp(4)+1) ≈ 0.98
    assert out[1][1] < 0.1   # 'growth' fires → P(uncertain) ≈ 0.02


def test_mnl_kernel_array_form_weights():
    from puremacro.narrative.indices import mnl_kernel
    weights = {
        "terms": ["concern", "growth"],
        "categories": ["uncertain", "neutral"],
        "beta": [[4.0, 0.0], [0.0, 4.0]],
        "bias": [0.0, 0.0],
    }
    records = [(pd.Timestamp("2024-01-01"), "Concern about inflation.", "u", {})]
    out = list(mnl_kernel(records, weights=weights, category="uncertain",
                          window="sentence"))
    assert out[0][1] > 0.9


def test_mnl_kernel_unknown_category_raises():
    from puremacro.narrative.indices import mnl_kernel
    weights = {"beta": {"x": {"a": 1.0}}, "bias": {"a": 0.0}}
    records = [(pd.Timestamp("2024-01-01"), "x", "u", {})]
    with pytest.raises(ValueError, match="not in"):
        list(mnl_kernel(records, weights=weights, category="MISSING"))


def test_mnl_kernel_canonicalize_array_form_validates_shape():
    from puremacro.narrative.indices import canonicalize_weights
    with pytest.raises(ValueError, match="beta shape"):
        canonicalize_weights({
            "terms": ["a", "b"],
            "categories": ["x", "y"],
            "beta": [[1, 0]],  # only 1 row, should be 2
            "bias": [0, 0],
        })


def test_mnl_kernel_empty_doc_yields_zero():
    from puremacro.narrative.indices import mnl_kernel
    weights = {"beta": {"x": {"a": 1.0}}, "bias": {"a": 0.0}}
    records = [(pd.Timestamp("2024-01-01"), "", "u", {})]
    out = list(mnl_kernel(records, weights=weights, category="a"))
    assert out[0][1] == 0.0


# ---------------------------------------------------------------------------
# B4 — llm_prob_kernel
# ---------------------------------------------------------------------------
def test_llm_prob_kernel_with_mock_provider(tmp_path, monkeypatch):
    from puremacro.narrative.indices import llm_prob_kernel, MockProvider
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(tmp_path / "scores.sqlite"))
    provider = MockProvider(hot_keywords=["uncertain"])
    records = [
        (pd.Timestamp("2024-01-01"), "Inflation outlook is uncertain. Risks remain.", "u", {}),
        (pd.Timestamp("2024-01-02"), "Growth is steady. Hiring is robust.", "u", {}),
    ]
    out = list(llm_prob_kernel(records, provider=provider, category="uncertainty",
                                window="sentence"))
    # Doc 1: 2 sentences, 1 has 'uncertain' → mean = 0.5
    assert out[0][1] == pytest.approx(0.5)
    # Doc 2: 0 'uncertain' hits → mean = 0
    assert out[1][1] == pytest.approx(0.0)


def test_llm_prob_kernel_cache_avoids_repeat_calls(tmp_path, monkeypatch):
    from puremacro.narrative.indices import llm_prob_kernel, MockProvider
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(tmp_path / "scores.sqlite"))

    class CountingMock(MockProvider):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def score_paragraph(self, text, category):
            self.calls += 1
            return super().score_paragraph(text, category)

    provider = CountingMock()
    records = [(pd.Timestamp("2024-01-01"), "Inflation outlook is uncertain.", "u", {})]
    # First pass: should make one API call.
    list(llm_prob_kernel(records, provider=provider, category="uncertainty",
                          window="sentence"))
    first_calls = provider.calls
    # Second pass: should hit cache, no new API calls.
    list(llm_prob_kernel(records, provider=provider, category="uncertainty",
                          window="sentence"))
    assert provider.calls == first_calls


def test_llm_prob_kernel_respects_max_calls(tmp_path, monkeypatch):
    from puremacro.narrative.indices import llm_prob_kernel, MockProvider
    monkeypatch.setenv("PUREMACRO_LLM_CACHE", str(tmp_path / "scores.sqlite"))

    class CountingMock(MockProvider):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def score_paragraph(self, text, category):
            self.calls += 1
            return super().score_paragraph(text, category)

    provider = CountingMock()
    # 3 sentences, 1 window each, budget = 1 call.
    records = [(pd.Timestamp("2024-01-01"),
                "One is uncertain. Two is uncertain. Three is uncertain.",
                "u", {})]
    list(llm_prob_kernel(records, provider=provider, category="uncertainty",
                          window="sentence", max_calls=1))
    assert provider.calls == 1


def test_anthropic_provider_raises_if_sdk_missing():
    from puremacro.narrative.indices import AnthropicProvider
    # If the SDK is installed in the env, this test is a no-op — pass anyway.
    try:
        import anthropic  # type: ignore  # noqa: F401
        pytest.skip("anthropic SDK is installed; cannot exercise the ImportError path.")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="anthropic"):
        AnthropicProvider()
