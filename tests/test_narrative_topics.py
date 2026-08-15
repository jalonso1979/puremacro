"""Tests for pure-Python TF-IDF, NMF, and DynamicTopicModel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.narrative.topics import TfidfVectorizer, NMF, DynamicTopicModel


def test_tfidf_vectorizer_basic():
    corpus = [
        "The Federal Reserve increased interest rates to fight high inflation.",
        "Interest rates and inflation expectations remain elevated.",
        "Labor market conditions are tightening and wage growth is strong.",
        "The central bank discussed monetary policy and inflation risks.",
    ]
    vec = TfidfVectorizer(max_features=20, min_df=1, stop_words="english")
    X = vec.fit_transform(corpus)
    
    assert X.shape[0] == 4
    assert X.shape[1] > 0
    assert np.all(X >= 0)
    # Check L2 normalization
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms, 1.0)
    
    # Check feature names
    features = vec.get_feature_names_out()
    assert "inflation" in features or "rates" in features


def test_nmf_factorization():
    rng = np.random.default_rng(42)
    # Non-negative matrix 10 x 8
    X = np.abs(rng.normal(loc=1.0, scale=0.5, size=(10, 8)))
    nmf = NMF(n_components=3, max_iter=100, random_state=42)
    W = nmf.fit_transform(X)
    
    assert W.shape == (10, 3)
    assert nmf.components_.shape == (3, 8)
    assert np.all(W >= 0)
    assert np.all(nmf.components_ >= 0)
    
    # Check transform
    W_new = nmf.transform(X[:2])
    assert W_new.shape == (2, 3)
    assert np.all(W_new >= 0)


def test_dynamic_topic_model_time_series():
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    texts = [
        "Inflation pressure remains high due to supply shocks and energy prices.",
        "Central bank raised policy rates to anchor inflation expectations.",
        "Labor market shows signs of cooling with slower employment growth.",
        "Financial market stress and credit spreads widening across banks.",
        "Inflation convergence towards target allows potential policy rate cuts.",
        "Economic activity rebounding with strong consumer spending and investment.",
    ]
    
    dtm = DynamicTopicModel(n_topics=3, random_state=42)
    res = dtm.fit_transform_corpus(texts, dates, freq="MS")
    
    assert res.topic_shares.shape[0] == 6
    assert res.topic_shares.shape[1] == 3
    assert "Topic_1" in res.topic_keywords
    assert len(res.topic_keywords["Topic_1"]) > 0
    assert res.document_topics.shape == (6, 3)
