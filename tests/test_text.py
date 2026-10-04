"""puremacro.text: tokenizer, segmentation, sparse document-term matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from puremacro.text import (
    DocumentTermMatrix, RegexTokenizer, build_dtm, ngrams, segment,
    split_paragraphs, split_sentences, stopwords,
)
from puremacro.text.stopwords import normalize_language, resolve_stop_words


# ── tokenizer ─────────────────────────────────────────────────────────────
def test_tokenizer_keeps_numbers_and_percentages():
    toks = RegexTokenizer()("The FOMC raised rates by 25 basis points to 5.25%.", "en")
    assert toks == ["fomc", "raised", "rates", "25", "basis", "points", "5.25%"]


def test_tokenizer_joins_spaced_percent_and_keeps_decimal_comma():
    assert RegexTokenizer()("subió al 4,2 % en marzo", "es") == ["subió", "4,2%", "marzo"]


def test_tokenizer_number_modes():
    text = "cut by 50 basis points to 4.5%"
    assert RegexTokenizer(numbers="mask")(text) == ["cut", "<num>", "basis", "points", "<pct>"]
    assert RegexTokenizer(numbers="drop")(text) == ["cut", "basis", "points"]
    with pytest.raises(ValueError):
        RegexTokenizer(numbers="round")


def test_stop_words_follow_document_language():
    tok = RegexTokenizer()
    assert tok("la inflación de los precios", "es") == ["inflación", "precios"]
    # Under English rules the Spanish articles survive.
    assert "la" in tok("la inflación de los precios", "en")
    assert tok("die Inflation und der Zins", "de") == ["inflation", "zins"]
    assert tok("la inflation et les taux", "fr") == ["inflation", "taux"]


def test_auto_stop_words_fall_back_to_default_language():
    assert RegexTokenizer()("the rate") == ["rate"]
    assert RegexTokenizer(default_language="es")("the rate") == ["the", "rate"]


def test_fixed_extra_and_no_stop_words():
    assert RegexTokenizer(stop_words="es")("la tasa", "en") == ["tasa"]
    assert RegexTokenizer(stop_words=None)("the rate") == ["the", "rate"]
    assert RegexTokenizer(extra_stop_words={"Committee"})("the committee met", "en") == ["met"]
    assert RegexTokenizer(stop_words={"rate"})("the rate") == ["the"]
    with pytest.raises(ValueError, match="no stop-word list"):
        RegexTokenizer(stop_words="klingon")


def test_elisions_apostrophes_and_unicode():
    tok = RegexTokenizer()
    assert tok("L’économie ralentit", "fr") == ["économie", "ralentit"]
    assert tok("dell'inflazione", "it") == ["inflazione"]
    # Curly apostrophe is normalised so "don’t" hits the English stop list.
    assert tok("Don’t hike", "en") == ["hike"]
    assert RegexTokenizer(strip_accents=True)("Inflación está", "es") == ["inflacion"]


def test_ngrams():
    assert ngrams(["a", "b", "c"], (1, 2)) == ["a", "b", "c", "a b", "b c"]
    assert ngrams(["a", "b", "c"], (3, 3)) == ["a b c"]
    with pytest.raises(ValueError):
        ngrams(["a"], (2, 1))


def test_language_normalisation():
    assert normalize_language("en-US") == "en"
    assert normalize_language("pt_BR") == "pt"
    assert normalize_language("Spanish") == "es"
    assert normalize_language(None) is None
    assert stopwords("xx") == frozenset()
    assert resolve_stop_words("auto") is None
    assert resolve_stop_words(None) == frozenset()


def test_function_satisfies_tokenizer_protocol():
    from puremacro.text import Tokenizer

    def whitespace(text, language=None):
        return text.split()

    assert isinstance(whitespace, Tokenizer)
    assert isinstance(RegexTokenizer(), Tokenizer)


# ── segmentation ──────────────────────────────────────────────────────────
def test_split_paragraphs():
    assert split_paragraphs("a\nb\n\n \n c \n\n") == ["a b", "c"]
    assert split_paragraphs("short\n\nlonger one", min_chars=6) == ["longer one"]


def test_split_sentences_respects_abbreviations_and_decimals():
    text = ("Inflation rose 2.5% in the U.S. economy. Mr. Powell spoke! "
            "\"Rates will rise,\" he said. Chair J. Powell agreed? Yes.")
    assert split_sentences(text) == [
        "Inflation rose 2.5% in the U.S. economy.",
        "Mr. Powell spoke!",
        "\"Rates will rise,\" he said.",
        "Chair J. Powell agreed?",
        "Yes.",
    ]


def test_split_sentences_spanish_openers():
    assert split_sentences("Subió la tasa. ¿Bajará pronto? ¡No!") == [
        "Subió la tasa.", "¿Bajará pronto?", "¡No!"]


def test_segment_frame():
    df = segment(["p1.\n\np2.", "", "x"], "paragraph", ids=["A", "B", "C"])
    assert df.to_dict("list") == {
        "parent_id": ["A", "A", "C"], "position": [0, 1, 0], "text": ["p1.", "p2.", "x"]}
    df = segment(["a;b"], lambda t: t.split(";"))
    assert df["text"].tolist() == ["a", "b"] and df["parent_id"].tolist() == [0, 0]
    with pytest.raises(ValueError):
        segment(["a"], "chapter")


# ── document-term matrix ──────────────────────────────────────────────────
TEXTS = [
    "Inflation rose to 3.5% as rates rose.",
    "Rates were cut by 25 basis points.",
    "La inflación bajó al 4%.",
]
DOCS = pd.DataFrame({
    "doc_id": ["a", "b", "c"],
    "country": ["US", "US", "MX"],
    "date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-01"]),
})


@pytest.fixture
def dtm():
    return build_dtm(TEXTS, languages=["en", "en", "es"], docs=DOCS)


def _dense(m):
    return pd.DataFrame(m.X.toarray(), columns=m.vocab)


def test_build_dtm_counts(dtm):
    assert sparse.isspmatrix_csr(dtm.X)
    assert dtm.shape == (3, 11)
    assert dtm.vocab == sorted(dtm.vocab)
    d = _dense(dtm)
    assert d.loc[0, "rose"] == 2 and d.loc[1, "rates"] == 1 and d.loc[0, "3.5%"] == 1
    assert d.loc[2, "inflación"] == 1 and "la" not in dtm.vocab
    assert dtm.X.dtype == np.int64
    assert dtm.docs["doc_id"].tolist() == ["a", "b", "c"]
    assert "3 docs x 11 terms" in repr(dtm)


def test_build_dtm_is_independent_of_document_order():
    a = build_dtm(TEXTS, languages="en")
    b = build_dtm(TEXTS[::-1], languages="en")
    assert a.vocab == b.vocab
    assert (a.X.toarray() == b.X.toarray()[::-1]).all()


def test_build_dtm_validates_lengths():
    with pytest.raises(ValueError):
        build_dtm(TEXTS, languages=["en"])
    with pytest.raises(ValueError):
        build_dtm(TEXTS, docs=DOCS.iloc[:2])


def test_ngrams_in_dtm():
    m = build_dtm(["basis points cut", "basis points hike"], ngram_range=(1, 2))
    assert "basis points" in m.vocab
    assert m.document_frequency()["basis points"] == 2


def test_filter(dtm):
    assert dtm.filter(min_df=2).vocab == ["rates"]
    assert dtm.filter(max_df=0.5).n_terms == 10
    # Ties at the cut-off are broken alphabetically.
    assert dtm.filter(max_features=3).vocab == ["25", "rates", "rose"]
    assert dtm.filter(terms=["rose", "nope", "25"]).vocab == ["rose", "25"]
    assert "rose" not in dtm.filter(drop_terms={"rose"}).vocab
    with pytest.raises(ValueError):
        dtm.filter(min_df=1.5)
    m = build_dtm(TEXTS, languages=["en", "en", "es"], min_df=2)
    assert m.vocab == ["rates"]


def test_tfidf_matches_sklearn_definition(dtm):
    C = dtm.X.toarray().astype(float)
    n = C.shape[0]
    df = (C > 0).sum(axis=0)
    for smooth, sub in [(True, False), (False, True)]:
        tf = np.where(C > 0, 1 + np.log(np.where(C > 0, C, 1)), 0) if sub else C
        idf = (np.log((1 + n) / (1 + df)) if smooth else np.log(n / df)) + 1
        R = tf * idf
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        got = dtm.tfidf(smooth_idf=smooth, sublinear_tf=sub).X.toarray()
        np.testing.assert_allclose(got, R, rtol=0, atol=1e-14)
    l1 = dtm.tfidf(norm="l1").X.toarray()
    np.testing.assert_allclose(l1.sum(axis=1), 1.0)
    assert dtm.tfidf().weighting == "tfidf"
    with pytest.raises(ValueError):
        dtm.tfidf().tfidf()


def test_binary(dtm):
    b = dtm.binary()
    assert b.X.max() == 1 and b.X.nnz == dtm.X.nnz and b.weighting == "binary"


def test_aggregate_country_month(dtm):
    p = dtm.aggregate("country", freq="MS")
    assert p.docs.to_dict("list") == {
        "country": ["MX", "US"],
        "date": [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-01-01")],
        "n_docs": [1, 2],
    }
    d = _dense(p)
    assert d.loc[1, "rates"] == 2 and d.loc[1, "rose"] == 2 and d.loc[0, "rates"] == 0
    np.testing.assert_array_equal(p.X.sum(axis=0), dtm.X.sum(axis=0))
    q = dtm.aggregate(freq="QS")
    assert q.n_docs == 1 and q.docs["n_docs"].tolist() == [3]
    with pytest.raises(ValueError):
        dtm.aggregate()
    with pytest.raises(ValueError, match="aggregate the count matrix"):
        dtm.tfidf().aggregate("country")


def test_aggregate_mixed_timezones():
    docs = pd.DataFrame({"doc_id": [0, 1], "date": [
        pd.Timestamp("2024-01-31T23:00", tz="America/New_York"),
        pd.Timestamp("2024-02-15T10:00", tz="UTC")]})
    p = build_dtm(["a rate", "b rate"], docs=docs).aggregate(freq="MS")
    # 23:00 New York on 31 Jan is 1 Feb UTC: both documents land in February.
    assert p.docs["n_docs"].tolist() == [2]


def test_transform_uses_fitted_vocabulary(dtm):
    t = dtm.transform(["rates and inflation, rates, and unseen words"], languages="en")
    assert t.vocab == dtm.vocab
    d = _dense(t)
    assert d.loc[0, "rates"] == 2 and d.loc[0, "inflation"] == 1 and d.values.sum() == 3


def test_tidy_and_frame(dtm):
    tidy = dtm.to_tidy()
    assert list(tidy.columns) == ["doc_id", "term", "value"]
    assert len(tidy) == dtm.X.nnz
    assert tidy.query("doc_id == 'a' and term == 'rose'")["value"].item() == 2
    wide = dtm.to_frame()
    assert list(wide.index) == ["a", "b", "c"] and wide.loc["a", "rose"] == 2
    assert build_dtm(["x y"], docs=pd.DataFrame({"k": [1]})).to_tidy().columns[0] == "row"


def test_save_load_roundtrip(dtm, tmp_path):
    out = dtm.save(tmp_path / "m")
    assert {p.name for p in out.iterdir()} == {"X.mtx", "vocab.txt", "docs.csv", "dtm.json"}
    back = DocumentTermMatrix.load(out)
    assert back.vocab == dtm.vocab
    assert (back.X != dtm.X).nnz == 0
    assert back.docs["doc_id"].tolist() == ["a", "b", "c"]
    assert "integer" in (out / "X.mtx").read_text().splitlines()[0]
    t = dtm.tfidf().save(tmp_path / "t")
    assert "real" in (t / "X.mtx").read_text().splitlines()[0]
    np.testing.assert_allclose(DocumentTermMatrix.load(t).X.toarray(),
                               dtm.tfidf().X.toarray(), atol=1e-12)


def test_custom_tokenizer():
    m = build_dtm(["A-B c", "c"], tokenizer=lambda text, language=None: text.split())
    assert m.vocab == ["A-B", "c"]


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        DocumentTermMatrix(X=sparse.csr_matrix((2, 2)), vocab=["a"], docs=pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(ValueError):
        DocumentTermMatrix(X=sparse.csr_matrix((1, 1)), vocab=["a"],
                           docs=pd.DataFrame({"x": [1]}), weighting="bm25")


def test_empty_corpus():
    m = build_dtm([])
    assert m.shape == (0, 0)
    m = build_dtm(["", None])
    assert m.shape == (2, 0)
