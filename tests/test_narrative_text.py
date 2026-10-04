"""NarrativeCorpus.dtm / .segment and the sparse path through narrative.topics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from puremacro.narrative.harvest import NarrativeCorpus, NarrativeDocument
from puremacro.narrative.topics import NMF, DynamicTopicModel, TfidfVectorizer
from puremacro.text import DocumentTermMatrix, RegexTokenizer


def _doc(doc_id, text, country="US", language="en", date="2024-01-05", **meta):
    return NarrativeDocument(
        doc_id=doc_id, source="test", country=country, date=pd.Timestamp(date),
        url=f"https://x/{doc_id}", title=f"T {doc_id}", text=text, language=language,
        policy_domain="monetary", doc_type="statement", metadata=meta,
    )


@pytest.fixture
def corpus():
    return NarrativeCorpus(docs=[
        _doc("a", "Rates rose 25 basis points.\n\nInflation is high. Mr. Powell spoke.",
             speaker="Powell"),
        _doc("b", "Rates were held.", date="2024-01-25"),
        _doc("c", "La inflación de los precios bajó.", country="MX", language="es",
             date="2024-02-02"),
    ])


def test_corpus_dtm_uses_document_language(corpus):
    m = corpus.dtm()
    assert isinstance(m, DocumentTermMatrix)
    assert m.docs["doc_id"].tolist() == ["a", "b", "c"]
    assert "text" not in m.docs.columns and "content_hash" in m.docs.columns
    # Spanish stop words removed from the Spanish document only.
    assert "los" not in m.vocab and "inflación" in m.vocab and "25" in m.vocab


def test_corpus_dtm_options(corpus):
    m = corpus.dtm(metadata="promote")
    assert "meta_speaker" in m.docs.columns
    assert corpus.dtm(min_df=2).vocab == ["rates"]
    assert corpus.dtm(weighting="tfidf").weighting == "tfidf"
    assert corpus.dtm(weighting="binary").X.max() == 1
    assert "<num>" in corpus.dtm(tokenizer=RegexTokenizer(numbers="mask")).vocab
    with pytest.raises(ValueError):
        corpus.dtm(weighting="bm25")


def test_corpus_dtm_aggregates_to_country_month_panel(corpus):
    p = corpus.dtm().aggregate("country", freq="MS")
    assert p.docs[["country", "n_docs"]].values.tolist() == [["MX", 1], ["US", 2]]
    assert p.X[:, p.term_index["rates"]].toarray().ravel().tolist() == [0, 2]


def test_segment_sentences_keeps_parent_fields(corpus):
    s = corpus.segment("sentence")
    assert [d.doc_id for d in s] == ["a:0", "a:1", "a:2", "b:0", "c:0"]
    a1 = s.docs[1]
    assert a1.text == "Inflation is high."
    assert a1.metadata == {"speaker": "Powell", "parent_id": "a", "position": 1,
                           "segment": "sentence"}
    assert (a1.country, a1.date, a1.language) == ("US", pd.Timestamp("2024-01-05"), "en")
    assert s.metadata["segmented_by"] == "sentence"
    # The parent corpus is not modified.
    assert corpus.docs[0].metadata == {"speaker": "Powell"}


def test_segment_paragraphs_and_callable(corpus):
    p = corpus.segment("paragraph")
    assert [d.doc_id for d in p] == ["a:0", "a:1", "b:0", "c:0"]
    assert p.segment("sentence").docs[1].metadata["parent_id"] == "a:1"

    def halves(text):
        return [text[: len(text) // 2], text[len(text) // 2:]]

    h = corpus.segment(halves, min_chars=5)
    assert h.metadata["segmented_by"] == "halves"
    assert all(len(d.text) >= 5 for d in h)


def test_segmented_corpus_round_trips_through_frame(corpus):
    s = corpus.segment("sentence")
    back = NarrativeCorpus.from_frame(s.to_frame())
    assert [d.metadata["parent_id"] for d in back] == ["a", "a", "a", "b", "c"]


# ── topics on the sparse matrix ───────────────────────────────────────────
TEXTS = [
    "Inflation pressure remains high due to supply shocks and energy prices.",
    "Central bank raised policy rates to anchor inflation expectations.",
    "Labor market shows signs of cooling with slower employment growth.",
    "Financial market stress and credit spreads widening across banks.",
    "Inflation convergence towards target allows potential policy rate cuts.",
    "Economic activity rebounding with strong consumer spending and investment.",
]


def test_tfidf_vectorizer_sparse_output_equals_dense():
    dense = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform(TEXTS)
    sp = TfidfVectorizer(min_df=1, ngram_range=(1, 2), sparse_output=True).fit_transform(TEXTS)
    assert isinstance(dense, np.ndarray) and sparse.issparse(sp)
    np.testing.assert_allclose(sp.toarray(), dense, atol=1e-15)


def test_tfidf_vectorizer_legacy_tokenizer_drops_numbers():
    vec = TfidfVectorizer(min_df=1).fit(["rates rose 25 basis points", "rates fell"])
    assert "25" not in vec.get_feature_names_out()
    vec = TfidfVectorizer(min_df=1, tokenizer=RegexTokenizer()).fit(
        ["rates rose 25 basis points", "rates fell"])
    assert "25" in vec.get_feature_names_out()


def test_tfidf_vectorizer_matches_pre_sparse_reference():
    # Reference numbers from the dense implementation this replaced.
    vec = TfidfVectorizer(max_features=5, min_df=1)
    X = vec.fit_transform(TEXTS)
    assert vec.get_feature_names_out() == ["inflation", "market", "policy", "across", "activity"]
    N, df = 6, np.array([3, 2, 2, 1, 1], dtype=float)
    idf = np.log((1 + N) / (1 + df)) + 1
    C = np.array([[1 if w in t.lower().split() else 0 for w in vec.get_feature_names_out()]
                  for t in TEXTS], dtype=float)
    R = C * idf
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    norms[norms == 0] = 1
    np.testing.assert_allclose(X, R / norms, atol=1e-15)


def test_nmf_sparse_matches_dense():
    rng = np.random.default_rng(1)
    X = np.abs(rng.normal(size=(20, 9)))
    X[X < 0.5] = 0.0
    d = NMF(n_components=3, random_state=0)
    s = NMF(n_components=3, random_state=0)
    Wd = d.fit_transform(X)
    Ws = s.fit_transform(sparse.csr_matrix(X))
    np.testing.assert_allclose(Ws, Wd, atol=1e-10)
    np.testing.assert_allclose(s.components_, d.components_, atol=1e-10)
    assert s.reconstruction_err_ == pytest.approx(d.reconstruction_err_, rel=1e-9)
    np.testing.assert_allclose(s.transform(sparse.csr_matrix(X[:3])), d.transform(X[:3]),
                               atol=1e-10)
    with pytest.raises(ValueError):
        NMF().fit(sparse.csr_matrix(-X))


def test_dynamic_topic_model_runs_on_sparse_matrix():
    res = DynamicTopicModel(n_topics=3).fit_transform_corpus(
        TEXTS, pd.date_range("2024-01-01", periods=6, freq="MS"))
    assert res.document_topics.shape == (6, 3)
    # The last text shares no term with the others (min_df=2), so its row is 0.
    np.testing.assert_allclose(res.document_topics.sum(axis=1), [1, 1, 1, 1, 1, 0])
