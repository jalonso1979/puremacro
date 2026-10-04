"""Corpus persistence, export and harvest reporting (puremacro.narrative.store/harvest)."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from puremacro.narrative.harvest import (
    QUERY_SOURCE_REGISTRY,
    SOURCE_REGISTRY,
    NarrativeCorpus,
    NarrativeDocument,
    harvest_narrative_corpus,
)
import puremacro.narrative.sources as sources_pkg
from puremacro.narrative.store import CorpusStore


def _doc(doc_id, text="Rates were held at 5.25%.", **kw):
    base = dict(
        doc_id=doc_id, source="fed_decision", country="USA",
        date=pd.Timestamp("2024-01-31"), url=f"https://x/{doc_id}", title=f"T {doc_id}",
        text=text, language="en", policy_domain="monetary", doc_type="decision",
        metadata={"bank_code": "FED", "speakers": ["Powell"], "n": 3},
    )
    base.update(kw)
    return NarrativeDocument(**base)


@pytest.fixture
def corpus():
    return NarrativeCorpus(docs=[
        _doc("a"),
        _doc("b", text="Mantuvo la tasa.", source="banxico", country="MEX", language="es",
             date=pd.Timestamp("2024-02-08"), magnitude=0.25, metadata={"bank_code": "BANXICO"}),
    ])


# ── registry ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted({**SOURCE_REGISTRY, **QUERY_SOURCE_REGISTRY}))
def test_every_registered_source_resolves_to_a_function(name):
    # Parsed, not imported: connectors pull scraper deps (bs4, pdfplumber) the
    # test environment need not have. Three entries once named functions that
    # did not exist and silently harvested nothing.
    mod, fn = (SOURCE_REGISTRY.get(name) or QUERY_SOURCE_REGISTRY[name])[:2]
    path = Path(sources_pkg.__file__).parent / f"{mod}.py"
    assert path.exists(), f"{name}: module {mod}.py missing"
    defined = {n.name for n in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(n, ast.FunctionDef)}
    assert fn in defined, f"{name}: {mod}.{fn} is not defined"


# ── harvest ──────────────────────────────────────────────────────────────────

def test_three_tuple_records_are_kept_and_doctype_is_read(monkeypatch, tmp_path):
    import puremacro.narrative.sources.uk_obr as mod

    def fake(**kw):
        yield (pd.Timestamp("2024-03-06"), "Spring budget forecast", "https://obr/1")
        yield ("2024-03-07", "EFO", "https://obr/2", {"doctype": "efo", "title": "EFO"})
        yield ("not a date", "x", "u", {})

    monkeypatch.setattr(mod, "iter_obr_publications", fake)
    c = harvest_narrative_corpus(sources=["uk_obr"], store=CorpusStore(tmp_path / "c.db"))
    assert len(c) == 2
    assert {d.doc_type for d in c} == {"report", "efo"}
    row = c.harvest_report.iloc[0]
    assert (row.status, row.n_docs, row.n_invalid) == ("ok", 2, 1)


def test_failing_connector_is_reported_not_swallowed(monkeypatch):
    import puremacro.narrative.sources.fed_decision as mod

    def boom():
        raise ConnectionError("site down")
        yield  # pragma: no cover

    monkeypatch.setattr(mod, "iter_fed_decision", boom)
    with pytest.warns(UserWarning, match="1 of 1 narrative sources failed"):
        c = harvest_narrative_corpus(sources=["fed_decision"], use_cache=False)
    assert c.harvest_report.loc[0, "status"] == "error"
    assert "site down" in c.harvest_report.loc[0, "error"]
    with pytest.raises(RuntimeError, match="site down"):
        harvest_narrative_corpus(sources=["fed_decision"], use_cache=False, on_error="raise")


def test_query_sources_need_kwargs(monkeypatch):
    import puremacro.narrative.sources.reddit as mod
    seen = {}

    def fake(subreddit, **kw):
        seen["sub"] = subreddit
        yield ("2024-05-01", "inflation thread", "https://r/1", {"title": "t1"})

    monkeypatch.setattr(mod, "iter_reddit", fake)
    with pytest.raises(ValueError, match="query-driven"):
        harvest_narrative_corpus(sources=["reddit"], use_cache=False)
    c = harvest_narrative_corpus(sources=["reddit"], use_cache=False,
                                 source_kwargs={"reddit": {"subreddit": "economics"}})
    assert seen["sub"] == "economics" and len(c) == 1 and c.docs[0].doc_type == "social"
    with pytest.raises(ValueError, match="unknown"):
        harvest_narrative_corpus(sources=["nope"], use_cache=False)


def test_same_key_different_text_gets_distinct_ids():
    recs = [("2024-01-01", "strike A", "https://landing", {}),
            ("2024-01-01", "strike B", "https://landing", {})]
    c = NarrativeCorpus.from_records(recs, source="x")
    assert len({d.doc_id for d in c}) == 2


# ── store ────────────────────────────────────────────────────────────────────

def test_store_keeps_latest_text_and_flags_revisions(tmp_path):
    with CorpusStore(tmp_path / "c.db") as st:
        assert st.upsert([_doc("a"), _doc("b")]) == {"new": 2, "updated": 0, "unchanged": 0}
        assert st.upsert([_doc("a")]) == {"new": 0, "updated": 0, "unchanged": 1}
        assert st.upsert([_doc("a", text="Rates were raised.")])["updated"] == 1
        assert len(st) == 2
        rev = st.revised()
        assert list(rev.doc_id) == ["a"] and rev.n_revisions[0] == 1
        assert rev.previous_content_hash[0] == _doc("a").content_hash
        loaded = st.load(sources=["fed_decision"])
    assert len(loaded) == 2
    a = next(d for d in loaded if d.doc_id == "a")
    assert a.text == "Rates were raised." and a.metadata["speakers"] == ["Powell"]


def test_harvest_writes_to_store_and_logs_run(monkeypatch, tmp_path):
    import puremacro.narrative.sources.fed_decision as mod
    monkeypatch.setattr(mod, "iter_fed_decision",
                        lambda: iter([("2020-03-15", "Cut to zero.", "https://f/1", {})]))
    path = tmp_path / "c.db"
    harvest_narrative_corpus(sources=["fed_decision"], db_path=path)
    harvest_narrative_corpus(sources=["fed_decision"], db_path=path)
    with CorpusStore(path) as st:
        assert len(st) == 1
        runs = st.runs()
    assert len(runs) == 2
    assert json.loads(runs.report_json[0])[0]["n_docs"] == 1


def test_store_load_date_bounds_with_mixed_timezones(tmp_path):
    with CorpusStore(tmp_path / "c.db") as st:
        st.upsert([_doc("a", date=pd.Timestamp("2024-01-31T14:00:00+00:00")),
                   _doc("b", date=pd.Timestamp("2023-06-01"))])
        assert [d.doc_id for d in st.load(start_date="2024-01-01")] == ["a"]


# ── export ───────────────────────────────────────────────────────────────────

def test_to_frame_metadata_layouts(corpus):
    assert isinstance(corpus.to_frame().metadata[0], dict)
    flat = corpus.to_frame(metadata="promote")
    assert {"meta_bank_code", "meta_n", "metadata_json", "content_hash"} <= set(flat.columns)
    assert "meta_speakers" not in flat.columns  # list-valued, stays in JSON
    assert json.loads(flat.metadata_json[0]) == {"speakers": ["Powell"]}
    assert "metadata" not in corpus.to_frame(metadata="drop").columns
    with pytest.raises(ValueError):
        corpus.to_frame(metadata="bogus")
    back = NarrativeCorpus.from_frame(flat)
    assert {d.doc_id: d.metadata for d in back} == {d.doc_id: d.metadata for d in corpus}


def test_jsonl_round_trip_and_manifest(corpus, tmp_path):
    path = corpus.to_jsonl(tmp_path / "out" / "corpus.jsonl")
    back = NarrativeCorpus.from_jsonl(path)
    assert [d.to_dict() for d in back] == [d.to_dict() for d in corpus]
    man = json.loads((tmp_path / "out" / "corpus.manifest.json").read_text())
    assert man["n_docs"] == 2 and man["sources"] == {"banxico": 1, "fed_decision": 1}
    assert man["documents"][0]["content_hash"] == corpus.docs[0].content_hash


def test_parquet_round_trip(corpus, tmp_path):
    pytest.importorskip("pyarrow")
    corpus.to_parquet(tmp_path / "corpus.parquet")
    back = NarrativeCorpus.from_parquet(tmp_path / "corpus.parquet")
    assert sorted(d.doc_id for d in back) == ["a", "b"]
    b = next(d for d in back if d.doc_id == "b")
    assert b.magnitude == 0.25 and b.metadata == {"bank_code": "BANXICO"}
    corpus.to_parquet(tmp_path / "parts", partition_cols=["source"], manifest=False)
    assert len(NarrativeCorpus.from_parquet(tmp_path / "parts")) == 2


def test_verify_manifest(corpus):
    man = corpus.manifest()
    rehydrated = NarrativeCorpus(docs=[_doc("a", text="edited")])
    status = rehydrated.verify_manifest(man).set_index("doc_id").status.to_dict()
    assert status == {"a": "changed", "b": "missing"}
    assert set(corpus.verify_manifest(man).status) == {"match"}


def test_connector_submodule_is_imported_on_demand(tmp_path, monkeypatch):
    # The sources package loads connector submodules lazily, so a plain
    # getattr(package, "fed_decision") is None in a fresh session; harvest
    # used to skip every connector not already imported.
    import sys
    from puremacro.narrative.harvest import _harvest_source

    pkg_dir = tmp_path / "fakesrc_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "fed_decision.py").write_text(
        "def iter_fed_decision():\n"
        "    yield ('2024-01-31', 'Held rates.', 'https://f/1', {})\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = importlib.import_module("fakesrc_pkg")
    assert getattr(pkg, "fed_decision", None) is None
    docs, report = _harvest_source("fed_decision", pkg, None, True)
    assert report["status"] == "ok" and len(docs) == 1
    sys.modules.pop("fakesrc_pkg.fed_decision", None)
    sys.modules.pop("fakesrc_pkg", None)
