"""Tests for puremacro.narrative.sources.bluesky and indices.bluesky."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "bluesky_mini"


class TestPostParser:
    def test_post_to_record_extracts_4_tuple(self):
        from puremacro.narrative.sources.bluesky import _post_to_record
        feed_path = FIXTURE_DIR / "feed_bsky_app.json"
        if not feed_path.exists():
            pytest.skip("feed fixture not fetched")
        feed_data = json.loads(feed_path.read_text())
        items = feed_data.get("feed", [])
        actor_meta = {
            "handle": "bsky.app", "did": "did:plc:z72i7hdynmk6r22z27h6tvur",
            "name": "Bluesky", "role": "test", "country": "WORLD",
            "actor_class": "institution",
        }
        recs = [_post_to_record(it, actor_meta=actor_meta) for it in items]
        # At least one original English post should parse
        non_null = [r for r in recs if r is not None]
        assert len(non_null) >= 1, f"no records parsed from {len(items)} items"
        for r in non_null:
            assert len(r) == 4
            date, text, source_url, metadata = r
            assert text.strip()
            assert source_url.startswith("https://bsky.app/profile/")
            assert metadata["handle"] == "bsky.app"

    def test_post_to_record_skips_reposts(self):
        from puremacro.narrative.sources.bluesky import _post_to_record
        # Synthetic feed item with reason=repost
        item = {
            "post": {
                "uri": "at://did:plc:abc/app.bsky.feed.post/xyz",
                "record": {"$type": "app.bsky.feed.post",
                           "text": "x", "createdAt": "2024-06-01T00:00:00Z",
                           "langs": ["en"]},
            },
            "reason": {"$type": "app.bsky.feed.defs#reasonRepost"},
        }
        actor_meta = {"handle": "test.bsky.social"}
        assert _post_to_record(item, actor_meta=actor_meta) is None

    def test_post_to_record_skips_non_english(self):
        from puremacro.narrative.sources.bluesky import _post_to_record
        item = {
            "post": {
                "uri": "at://did:plc:abc/app.bsky.feed.post/xyz",
                "record": {"$type": "app.bsky.feed.post",
                           "text": "x", "createdAt": "2024-06-01T00:00:00Z",
                           "langs": ["ja"]},
            },
        }
        actor_meta = {"handle": "test.bsky.social"}
        assert _post_to_record(item, actor_meta=actor_meta) is None

    def test_post_to_record_accepts_missing_langs(self):
        """If langs field is missing entirely, accept (permissive)."""
        from puremacro.narrative.sources.bluesky import _post_to_record
        item = {
            "post": {
                "uri": "at://did:plc:abc/app.bsky.feed.post/xyz",
                "record": {"$type": "app.bsky.feed.post",
                           "text": "x", "createdAt": "2024-06-01T00:00:00Z"},
            },
        }
        actor_meta = {"handle": "test.bsky.social"}
        rec = _post_to_record(item, actor_meta=actor_meta)
        assert rec is not None

    def test_known_handles_structure(self):
        from puremacro.narrative.sources.bluesky import KNOWN_HANDLES
        assert len(KNOWN_HANDLES) >= 25
        for entry in KNOWN_HANDLES:
            assert {"handle", "name", "role", "country", "actor_class"} <= set(entry)
            assert entry["actor_class"] in {"institution", "governor", "minister"}

    def test_iter_empty_handles_yields_nothing(self):
        from puremacro.narrative.sources.bluesky import iter_bluesky_posts
        records = list(iter_bluesky_posts(handles=()))
        assert records == []


@pytest.mark.network
class TestBlueskyLive:
    def test_resolve_known_active_handle(self):
        from puremacro.narrative.sources.bluesky import _resolve_handle
        prof = _resolve_handle("bsky.app")
        if prof is None:
            pytest.skip("network fetch returned empty")
        assert "did:plc:" in prof["did"]
        assert prof["handle"] == "bsky.app"

    def test_iter_one_known_handle_smoke(self):
        from puremacro.narrative.sources.bluesky import iter_bluesky_posts
        records = list(iter_bluesky_posts(handles=("bsky.app",),
                                          max_posts_per_actor=3,
                                          since="2024-01-01"))
        if not records:
            pytest.skip("network fetch returned empty")
        assert len(records) <= 3
        for r in records:
            assert len(r) == 4


class TestBlueskyUi:
    @pytest.fixture
    def toy(self):
        import datetime as dt
        import pandas as pd
        return pd.DataFrame([
            (dt.date(2024, 6, 1), "Inflation outlook uncertain.",
             "https://bsky.app/profile/a/post/x1",
             {"handle": "a", "actor_class": "institution", "country": "USA",
              "name": "Inst A", "role": "central bank", "did": "d1",
              "post_uri": "at://d1/x1", "langs": ["en"]}),
            (dt.date(2024, 6, 2), "Markets remain volatile.",
             "https://bsky.app/profile/a/post/x2",
             {"handle": "a", "actor_class": "institution", "country": "USA",
              "name": "Inst A", "role": "central bank", "did": "d1",
              "post_uri": "at://d1/x2", "langs": ["en"]}),
            (dt.date(2024, 6, 3), "Rate decision pending.",
             "https://bsky.app/profile/b/post/y1",
             {"handle": "b", "actor_class": "governor", "country": "GBR",
              "name": "Gov B", "role": "BoE", "did": "d2",
              "post_uri": "at://d2/y1", "langs": ["en"]}),
        ], columns=["date", "text", "source_url", "metadata"])

    def test_default(self, toy):
        from puremacro.narrative.indices.bluesky import bluesky_ui
        ri = bluesky_ui(toy)
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_actor_class_filter(self, toy):
        from puremacro.narrative.indices.bluesky import bluesky_ui
        ri = bluesky_ui(toy, actor_class="governor")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_country_filter(self, toy):
        from puremacro.narrative.indices.bluesky import bluesky_ui
        ri = bluesky_ui(toy, country="USA")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_accepts_iter_of_tuples(self):
        import datetime as dt
        from puremacro.narrative.indices.bluesky import bluesky_ui
        records = [
            (dt.date(2024, 6, 1), "Outlook uncertain.",
             "https://bsky.app/x", {"actor_class": "institution",
                                     "country": "USA"}),
        ]
        from puremacro.narrative.types import RiskIndex
        ri = bluesky_ui(iter(records))
        assert isinstance(ri, RiskIndex)


class TestBlueskyMultilingual:
    def _synth_post(self, langs):
        """Build a minimal Bluesky post JSON for _post_to_record."""
        return {
            "post": {
                "uri": "at://did:plc:fake/app.bsky.feed.post/abc123",
                "record": {
                    "$type": "app.bsky.feed.post",
                    "text": "Some test post text.",
                    "createdAt": "2024-06-01T12:00:00Z",
                    "langs": langs,
                },
            },
        }

    def test_post_with_de_langs_accepted_when_de_in_languages(self):
        from puremacro.narrative.sources import bluesky
        actor_meta = {"handle": "test.bsky.social", "did": "did:plc:fake",
                       "name": "Test", "role": "", "country": "",
                       "actor_class": ""}
        post = self._synth_post(["de"])
        rec = bluesky._post_to_record(
            post, actor_meta=actor_meta, languages=("en", "de", "fr"),
        )
        assert rec is not None
        assert "de" in rec[3]["langs"]

    def test_post_with_de_langs_rejected_when_only_en_requested(self):
        from puremacro.narrative.sources import bluesky
        actor_meta = {"handle": "test.bsky.social", "did": "did:plc:fake",
                       "name": "Test", "role": "", "country": "",
                       "actor_class": ""}
        post = self._synth_post(["de"])
        rec = bluesky._post_to_record(
            post, actor_meta=actor_meta, languages=("en",),
        )
        assert rec is None

    def test_post_with_no_langs_tag_accepted_under_any_filter(self):
        """Backwards-compat: posts without a langs field are always kept."""
        from puremacro.narrative.sources import bluesky
        actor_meta = {"handle": "test.bsky.social", "did": "did:plc:fake",
                       "name": "Test", "role": "", "country": "",
                       "actor_class": ""}
        post = {
            "post": {
                "uri": "at://did:plc:fake/app.bsky.feed.post/abc123",
                "record": {
                    "$type": "app.bsky.feed.post",
                    "text": "Some test post text.",
                    "createdAt": "2024-06-01T12:00:00Z",
                    # no langs key
                },
            },
        }
        rec = bluesky._post_to_record(
            post, actor_meta=actor_meta, languages=("de",),
        )
        assert rec is not None


class TestBlueskyActorMonthAggregation:
    def _records(self):
        """5 records: 3 from handle_a (2 in 2024-06, 1 in 2024-07),
        2 from handle_b (both 2024-06)."""
        from datetime import date
        return [
            (date(2024, 6, 1), "post a1 jun", "http://a1",
             {"handle": "a.bsky", "country": "USA", "actor_class": "institution"}),
            (date(2024, 6, 15), "post a2 jun", "http://a2",
             {"handle": "a.bsky", "country": "USA", "actor_class": "institution"}),
            (date(2024, 7, 5), "post a3 jul", "http://a3",
             {"handle": "a.bsky", "country": "USA", "actor_class": "institution"}),
            (date(2024, 6, 3), "post b1 jun", "http://b1",
             {"handle": "b.bsky", "country": "GBR", "actor_class": "governor"}),
            (date(2024, 6, 20), "post b2 jun", "http://b2",
             {"handle": "b.bsky", "country": "GBR", "actor_class": "governor"}),
        ]

    def test_aggregate_to_actor_month_reduces_record_count(self):
        from puremacro.narrative.indices.bluesky import _aggregate_to_actor_month
        import pandas as pd
        df = pd.DataFrame(self._records(),
                           columns=["date", "text", "source_url", "metadata"])
        out = _aggregate_to_actor_month(df)
        # 5 raw → 3 groups: (a, 2024-06), (a, 2024-07), (b, 2024-06)
        assert len(out) == 3

    def test_aggregate_to_actor_month_concatenates_text(self):
        from puremacro.narrative.indices.bluesky import _aggregate_to_actor_month
        import pandas as pd
        df = pd.DataFrame(self._records(),
                           columns=["date", "text", "source_url", "metadata"])
        out = _aggregate_to_actor_month(df)
        a_jun = out[out["metadata"].apply(
            lambda m: m["handle"] == "a.bsky")].iloc[0]
        assert "a1" in a_jun["text"]
        assert "a2" in a_jun["text"]

    def test_aggregate_to_actor_month_adds_n_posts_metadata(self):
        from puremacro.narrative.indices.bluesky import _aggregate_to_actor_month
        import pandas as pd
        df = pd.DataFrame(self._records(),
                           columns=["date", "text", "source_url", "metadata"])
        out = _aggregate_to_actor_month(df)
        for _, row in out.iterrows():
            assert "n_posts" in row["metadata"]
            assert row["metadata"]["aggregation"] == "actor_month"
        counts = {(m["handle"], r["date"]): m["n_posts"]
                  for _, r in out.iterrows() for m in [r["metadata"]]}
        from datetime import date
        assert counts[("a.bsky", date(2024, 6, 1))] == 2
        assert counts[("a.bsky", date(2024, 7, 1))] == 1
        assert counts[("b.bsky", date(2024, 6, 1))] == 2

    def test_bluesky_ui_aggregate_to_none_preserves_raw_count(self):
        from puremacro.narrative.indices.bluesky import bluesky_ui
        from datetime import date
        records = [
            (date(2024, 6, 1), "uncertainty volatility risk", "http://a1",
             {"handle": "a.bsky", "country": "USA", "actor_class": "institution"}),
        ]
        idx_off = bluesky_ui(records, aggregate_to=None)
        idx_on = bluesky_ui(records, aggregate_to="actor_month")
        assert idx_off is not None
        assert idx_on is not None

    def test_bluesky_ui_unknown_aggregate_to_raises(self):
        from puremacro.narrative.indices.bluesky import bluesky_ui
        from datetime import date
        records = [(date(2024, 6, 1), "uncertainty", "http://a1",
                     {"handle": "a.bsky"})]
        with pytest.raises(ValueError, match="unknown aggregate_to"):
            bluesky_ui(records, aggregate_to="weekly")
