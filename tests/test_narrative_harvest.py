"""Test suite for puremacro.narrative.harvest."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.harvest import (
    SOURCE_REGISTRY,
    NarrativeCorpus,
    NarrativeDocument,
    harvest_narrative_corpus,
)
from puremacro.narrative.types import NarrativeEvent


def test_source_registry_coverage():
    """Verify source registry covers major central banks and fiscal agencies."""
    assert len(SOURCE_REGISTRY) >= 30

    # Key sources present
    assert "fed_decision" in SOURCE_REGISTRY
    assert "fed_minutes" in SOURCE_REGISTRY
    assert "ecb_decision" in SOURCE_REGISTRY
    assert "boe_decision" in SOURCE_REGISTRY
    assert "boj_decision" in SOURCE_REGISTRY
    assert "banxico" in SOURCE_REGISTRY
    assert "bcb" in SOURCE_REGISTRY
    assert "us_cbo" in SOURCE_REGISTRY
    assert "imf_articleiv" in SOURCE_REGISTRY

    for name, (mod, fn, country, domain, dt, lang) in SOURCE_REGISTRY.items():
        assert country
        assert domain in {"monetary", "fiscal", "macropru", "fx", "structural", "general"}
        assert dt
        assert lang


def test_narrative_document_and_corpus_operations():
    """Verify NarrativeDocument and NarrativeCorpus filter, to_frame, summary, to_events."""
    docs = [
        NarrativeDocument(
            doc_id="d1",
            source="fed_minutes",
            country="USA",
            date=pd.Timestamp("2021-03-17"),
            url="https://federalreserve.gov/m1",
            title="Minutes March 2021",
            text="The committee noted that the fiscal stimulus supported household spending.",
            language="en",
            policy_domain="monetary",
            doc_type="minutes",
        ),
        NarrativeDocument(
            doc_id="d2",
            source="us_cbo",
            country="USA",
            date=pd.Timestamp("2021-06-01"),
            url="https://cbo.gov/r1",
            title="CBO Report",
            text="Infrastructure investment package of 1.2% of GDP was signed into law.",
            language="en",
            policy_domain="fiscal",
            doc_type="report",
            magnitude=1.2,
        ),
        NarrativeDocument(
            doc_id="d3",
            source="banxico",
            country="MEX",
            date=pd.Timestamp("2021-08-15"),
            url="https://banxico.org.mx/m1",
            title="Minuta Banxico",
            text="La junta decidió un alza de tasa de interés para contener las presiones inflacionarias.",
            language="es",
            policy_domain="monetary",
            doc_type="minutes",
        ),
    ]

    corpus = NarrativeCorpus(docs=docs)
    assert len(corpus) == 3
    assert corpus.countries == ["MEX", "USA"]
    assert corpus.policy_domains == ["fiscal", "monetary"]

    # 1. to_frame
    df = corpus.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "text" in df.columns

    # 2. filter
    us_corpus = corpus.filter(countries=["USA"])
    assert len(us_corpus) == 2
    assert us_corpus.countries == ["USA"]

    fiscal_corpus = corpus.filter(policy_domains=["fiscal"])
    assert len(fiscal_corpus) == 1
    assert fiscal_corpus.docs[0].source == "us_cbo"

    date_filtered = corpus.filter(start_date="2021-05-01")
    assert len(date_filtered) == 2

    # 3. summary
    summ = corpus.summary()
    assert isinstance(summ, pd.DataFrame)
    assert "n_docs" in summ.columns

    # 4. to_events
    events = corpus.to_events()
    assert len(events) >= 2
    for ev in events:
        assert isinstance(ev, NarrativeEvent)
        assert ev.country in ("usa", "mex")


def test_harvest_narrative_corpus_mock(monkeypatch):
    """Test harvest_narrative_corpus orchestrator with mock connector."""
    def _fake_fed_decision():
        yield (
            "2020-03-23",
            "The Federal Open Market Committee announced aggressive asset purchase program to support liquidity.",
            "https://fake.url/fed",
            {"country": "USA", "policy_domain": "monetary", "doc_type": "decision"},
            None,
        )

    import puremacro.narrative.sources.fed_decision as fed_mod
    monkeypatch.setattr(fed_mod, "iter_fed_decision", _fake_fed_decision)

    corpus = harvest_narrative_corpus(
        sources=["fed_decision"],
        max_docs_per_source=10,
    )

    assert len(corpus) == 1
    doc = corpus.docs[0]
    assert doc.source == "fed_decision"
    assert doc.country == "USA"
    assert doc.policy_domain == "monetary"
    assert "asset purchase" in doc.text
