"""A-ext + D2/D4 — connector smokes + ergonomics infra."""
from __future__ import annotations

import warnings
from unittest import mock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Schema validation (D4)
# ---------------------------------------------------------------------------
def test_validate_records_passes_valid_4tuples():
    from puremacro.narrative.sources import validate_records
    iterable = [
        (pd.Timestamp("2024-01-01"), "text", "url", {"language": "en"}),
        (pd.Timestamp("2024-04-01"), "more text", "url2", {}),
    ]
    out = list(validate_records(iterable))
    assert len(out) == 2


def test_validate_records_raises_strict_on_3tuple():
    from puremacro.narrative.sources import validate_records, SourceRecordValidationError
    bad = [(pd.Timestamp("2024-01-01"), "text", "url")]  # 3-tuple!
    with pytest.raises(SourceRecordValidationError, match="4-tuple"):
        list(validate_records(bad, strict=True))


def test_validate_records_warns_when_lenient():
    from puremacro.narrative.sources import validate_records
    bad = [
        (pd.Timestamp("2024-01-01"), "ok", "url", {}),
        (pd.Timestamp("2024-04-01"), "bad", "url"),  # 3-tuple
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = list(validate_records(bad, strict=False))
    assert len(out) == 1
    assert any("4-tuple" in str(w.message) for w in caught)


def test_validate_records_rejects_bad_meta_type():
    from puremacro.narrative.sources import validate_records, SourceRecordValidationError
    with pytest.raises(SourceRecordValidationError, match="metadata"):
        list(validate_records([(pd.Timestamp("2024-01-01"), "t", "u", "not-a-dict")]))


# ---------------------------------------------------------------------------
# Disk cache (D2)
# ---------------------------------------------------------------------------
def test_disk_cache_caches_dataframe(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_CACHE_DIR", str(tmp_path))
    from puremacro.cache import disk_cache

    calls = {"n": 0}
    def loader():
        calls["n"] += 1
        return pd.DataFrame({"x": [1, 2, 3]})

    df1 = disk_cache("test_key", loader, namespace="unittest")
    df2 = disk_cache("test_key", loader, namespace="unittest")  # cache hit
    assert calls["n"] == 1
    pd.testing.assert_frame_equal(df1, df2)


def test_disk_cache_caches_dict(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_CACHE_DIR", str(tmp_path))
    from puremacro.cache import disk_cache
    calls = {"n": 0}
    def loader():
        calls["n"] += 1
        return {"alpha": 1, "beta": [1, 2, 3]}
    val = disk_cache("dict_key", loader, namespace="unittest")
    val2 = disk_cache("dict_key", loader, namespace="unittest")
    assert calls["n"] == 1
    assert val == val2


def test_disk_cache_refetch_bypasses(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_CACHE_DIR", str(tmp_path))
    from puremacro.cache import disk_cache
    calls = {"n": 0}
    def loader():
        calls["n"] += 1
        return pd.DataFrame({"x": [1]})
    disk_cache("k", loader, namespace="u")
    disk_cache("k", loader, namespace="u", refetch=True)
    assert calls["n"] == 2


def test_disk_cache_env_disable(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PUREMACRO_CACHE_DISABLE", "1")
    from puremacro.cache import disk_cache
    calls = {"n": 0}
    def loader():
        calls["n"] += 1
        return {"x": 1}
    disk_cache("k", loader)
    disk_cache("k", loader)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# NBER WP connector (mocked HTTP)
# ---------------------------------------------------------------------------
def test_iter_nber_wp_parses_items(monkeypatch):
    fake_xml = b"""<?xml version="1.0"?><rss><channel>
        <item>
            <title>A Cool Paper -- by Alice, Bob</title>
            <description>This paper studies labor uncertainty.</description>
            <link>https://www.nber.org/papers/w12345#fromrss</link>
        </item>
    </channel></rss>"""
    from puremacro.narrative.sources import nber_wp as mod
    monkeypatch.setattr(mod, "safe_get_bytes", lambda *_a, **_kw: fake_xml)
    recs = list(mod.iter_nber_wp())
    assert len(recs) == 1
    d, text, url, meta = recs[0]
    assert "A Cool Paper" in text
    assert "labor uncertainty" in text
    assert meta["authors"] == "Alice, Bob"
    assert meta["bank_code"] == "NBER"
    assert "w12345" in url


# ---------------------------------------------------------------------------
# Hacker News connector (mocked API)
# ---------------------------------------------------------------------------
def test_iter_hackernews_parses_hits(monkeypatch):
    page1 = {"hits": [
        {
            "title": "Cloudflare lays off 1100",
            "story_text": "",
            "url": "https://example.com",
            "objectID": "12345",
            "author": "u123",
            "points": 200,
            "num_comments": 50,
            "created_at_i": 1714780800,  # 2024-05-04
        },
        {
            "title": "AI is killing my job",
            "story_text": "Lengthy rant about AI...",
            "url": None,
            "objectID": "67890",
            "author": "u456",
            "points": 50,
            "num_comments": 12,
            "created_at_i": 1714800000,
        },
    ]}
    empty = {"hits": []}
    responses = [page1, empty]
    call = {"i": 0}
    def fake_get(*args, **kwargs):
        idx = call["i"]
        call["i"] += 1
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = lambda: None
        resp.json = (lambda i=idx: responses[i])
        return resp

    from puremacro.narrative.sources import hackernews as mod
    monkeypatch.setattr(mod.requests, "get", fake_get)
    recs = list(mod.iter_hackernews("layoff"))
    assert len(recs) == 2
    assert all(r[3]["bank_code"] == "HN" for r in recs)
    assert recs[0][3]["score"] == 200
    assert "Cloudflare" in recs[0][1]


def test_iter_hackernews_min_score_filters(monkeypatch):
    page1 = {"hits": [
        {"title": "popular", "url": "x", "objectID": "1", "author": "a",
         "points": 500, "num_comments": 100, "created_at_i": 1714780800},
        {"title": "ignored", "url": "y", "objectID": "2", "author": "b",
         "points": 2, "num_comments": 0, "created_at_i": 1714780900},
    ]}
    empty = {"hits": []}
    responses = [page1, empty]
    call = {"i": 0}
    def fake_get(*args, **kwargs):
        idx = call["i"]
        call["i"] += 1
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = lambda: None
        resp.json = (lambda i=idx: responses[i])
        return resp

    from puremacro.narrative.sources import hackernews as mod
    monkeypatch.setattr(mod.requests, "get", fake_get)
    recs = list(mod.iter_hackernews("anything", min_score=100))
    assert len(recs) == 1
    assert recs[0][3]["score"] == 500


# ---------------------------------------------------------------------------
# Reddit connector (mocked API)
# ---------------------------------------------------------------------------
def test_iter_reddit_parses_posts(monkeypatch):
    fake_resp = {
        "data": {"children": [
            {"data": {
                "title": "I just got laid off",
                "selftext": "Was at the company for 5 years...",
                "permalink": "/r/LayOffs/comments/abc/i_just_got_laid_off/",
                "url": "https://reddit.com/r/LayOffs/...",
                "score": 150,
                "num_comments": 25,
                "author": "throwaway123",
                "created_utc": 1714780800,
            }},
            {"data": {
                "title": "Nothing important",
                "selftext": "",
                "permalink": "/r/LayOffs/comments/xyz/x/",
                "url": "https://reddit.com/r/LayOffs/...",
                "score": -5,
                "num_comments": 0,
                "author": "u",
                "created_utc": 1714790800,
            }},
        ]}
    }
    def fake_get(*args, **kwargs):
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = lambda: None
        resp.json = lambda: fake_resp
        return resp

    from puremacro.narrative.sources import reddit as mod
    monkeypatch.setattr(mod.requests, "get", fake_get)
    recs = list(mod.iter_reddit("LayOffs", min_score=0))
    assert len(recs) == 1  # the -5 post is filtered
    _, text, url, meta = recs[0]
    assert "laid off" in text.lower()
    assert "5 years" in text
    assert meta["bank_code"] == "REDDIT"
    assert meta["subreddit"] == "LayOffs"
    assert meta["score"] == 150


def test_iter_reddit_rejects_unknown_sort():
    from puremacro.narrative.sources import iter_reddit
    with pytest.raises(ValueError, match="sort"):
        list(iter_reddit("LayOffs", sort="banana"))
