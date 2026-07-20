"""Tests for puremacro.narrative.sources.us_erp,
puremacro.narrative.sources.us_sotu,
and their respective parsers."""
from __future__ import annotations

from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "us_executive_mini"


class TestErpParser:
    def test_split_chapters_handles_inline_chapter_headers(self):
        """Inline 'CHAPTER N: Title' format (older ERPs / fallback path)."""
        from puremacro.narrative.sources.us_erp import _split_chapters

        # Simulate an older ERP where chapter headers are inline.
        # Each element is one page's text.
        pages = [
            "PRESIDENT'S MESSAGE\nPresident foo bar baz.",
            "CHAPTER 1: Economic Outlook\nThe outlook is strong.",
            "More outlook text continues here.",
            "CHAPTER 2: Labor Markets\nLabor markets tightened in 2023.",
            "Wages rose 4 percent.",
        ]
        chapters = _split_chapters(pages)
        labels = [c[0] for c in chapters]
        assert "front matter" in labels
        assert any("Chapter 1" in lbl for lbl in labels)
        assert any("Chapter 2" in lbl for lbl in labels)
        assert all(c[1].strip() for c in chapters)

    def test_split_chapters_handles_page_start_headers(self):
        """Page-start 'Chapter N' format (modern ERPs, primary path)."""
        from puremacro.narrative.sources.us_erp import _split_chapters

        # Simulate modern ERP: chapter number alone on first line of page,
        # title on second line, body on subsequent lines.
        pages = [
            "Contents\nChapter 1: The Benefits ...... 21\nChapter 2: Review .... 61",
            "Letter of Transmittal\nDear President,\nHerewith.",
            "Chapter 1\nThe Benefits of Full Employment\nThis chapter covers ...",
            "Employment continued to be strong throughout the year.",
            "Chapter 2\nThe Year in Review\nAt the start of the year forecasters ...",
            "Growth was stronger than expected.",
        ]
        chapters = _split_chapters(pages)
        labels = [c[0] for c in chapters]
        assert "front matter" in labels
        assert any("Chapter 1" in lbl for lbl in labels)
        assert any("Chapter 2" in lbl for lbl in labels)
        assert all(c[1].strip() for c in chapters)
        # Verify TOC "Chapter 1: ..." did NOT create a spurious chapter split
        # (only 3 entries: front matter + 2 chapters)
        assert len(chapters) == 3

    def test_iter_erp_empty_window_yields_nothing(self):
        from puremacro.narrative.sources.us_erp import iter_erp
        records = list(iter_erp(since="1900-01-01", until="1900-12-31"))
        assert records == []

    def test_fixture_pdf_extracts(self):
        from puremacro.narrative.sources.us_erp import (
            _extract_erp_pages, _split_chapters,
        )
        pdf_path = FIXTURE_DIR / "erp_2024.pdf"
        if not pdf_path.exists():
            pytest.skip("erp_2024.pdf fixture not yet fetched")
        pages = _extract_erp_pages(pdf_path.read_bytes())
        assert len(pages) > 100, f"expected 100+ pages, got {len(pages)}"
        chapters = _split_chapters(pages)
        assert len(chapters) >= 3, (
            f"expected 3+ chapter splits, got {len(chapters)}"
        )
        # Verify chapter labels follow expected pattern
        chapter_labels = [c[0] for c in chapters if c[0] != "front matter"]
        assert len(chapter_labels) >= 7, (
            f"ERP-2024 has 7 chapters; got {chapter_labels}"
        )
        # Every chapter body must be non-trivially long
        for label, body in chapters:
            assert len(body) > 200, (
                f"Chapter {label!r} body too short ({len(body)} chars)"
            )


@pytest.mark.network
class TestErpLiveFetch:
    def test_fetch_recent_erp(self):
        from puremacro.narrative.sources.us_erp import iter_erp
        records = list(iter_erp(since="2023-01-01", until="2024-12-31",
                                granularity="per_document"))
        if not records:
            pytest.skip("network fetch returned empty — endpoint may be unavailable")
        for r in records:
            assert len(r) == 4
            date, text, source_url, metadata = r
            assert text.strip(), "ERP body text must be non-empty"
            assert "ERP" in source_url or "erp" in source_url.lower()


class TestSotuParser:
    def test_parse_date_handles_common_formats(self):
        from puremacro.narrative.sources.us_sotu import _parse_date
        import datetime as dt
        assert _parse_date("January 23, 2024") == dt.date(2024, 1, 23)
        assert _parse_date("2024-01-23") == dt.date(2024, 1, 23)
        assert _parse_date("") is None
        assert _parse_date(None) is None

    def test_parse_sotu_fixture(self):
        from puremacro.narrative.sources.us_sotu import _parse_sotu_page
        html_path = FIXTURE_DIR / "sotu_2024.html"
        if not html_path.exists():
            pytest.skip("sotu_2024.html fixture not yet fetched")
        rec = _parse_sotu_page(html_path.read_text(),
                                source_url="https://test")
        assert rec is not None, "SOTU fixture failed to parse"
        date, text, source_url, metadata = rec
        assert len(text) > 1000, f"SOTU body text suspiciously short: {len(text)}"
        assert "year" in metadata


@pytest.mark.network
class TestSotuLiveFetch:
    def test_listing_returns_urls(self):
        from puremacro.narrative.sources.us_sotu import _sotu_listing
        urls = _sotu_listing()
        if not urls:
            pytest.skip("listing returned empty; UCSB endpoint may be down")
        assert any("ucsb" in u.lower() for u in urls)
        assert len(urls) >= 5


class TestCboParser:
    def test_parse_rss_yields_items(self):
        from puremacro.narrative.sources.us_cbo import _parse_rss
        rss_path = FIXTURE_DIR / "cbo_rss_sample.xml"
        if not rss_path.exists():
            pytest.skip("cbo_rss_sample.xml fixture not yet fetched")
        items = _parse_rss(rss_path.read_text())
        assert len(items) >= 5, f"expected 5+ items, got {len(items)}"
        first = items[0]
        assert "title" in first
        assert first["link"].startswith("https://www.cbo.gov/publication/")
        assert first["pubDate"] is not None
        assert first["pub_id"] is not None

    def test_find_pdf_link_handles_fixture(self):
        from puremacro.narrative.sources.us_cbo import _find_pdf_link
        import glob
        candidates = sorted(glob.glob(str(FIXTURE_DIR / "cbo_pub_*.html")))
        if not candidates:
            pytest.skip("no cbo_pub_*.html fixture")
        html = open(candidates[0]).read()
        # Extract pub_id from filename
        import re
        m = re.search(r"cbo_pub_(\d+)\.html", candidates[0])
        pub_id = int(m.group(1)) if m else None
        pdf_url = _find_pdf_link(html, pub_id=pub_id)
        assert pdf_url is not None
        assert pdf_url.endswith(".pdf")
        # Must resolve to a real cbo.gov URL (Wayback wrappers stripped)
        assert "cbo.gov" in pdf_url

    def test_extract_cbo_pdf_text_on_fixture(self):
        from puremacro.narrative.sources.us_cbo import _extract_cbo_pdf_text
        import glob
        candidates = sorted(glob.glob(str(FIXTURE_DIR / "cbo_pub_*.pdf")))
        if not candidates:
            pytest.skip("no cbo_pub_*.pdf fixture")
        pdf_bytes = open(candidates[0], "rb").read()
        text = _extract_cbo_pdf_text(pdf_bytes)
        assert len(text) > 1000, f"CBO PDF body text suspiciously short: {len(text)}"


@pytest.mark.network
class TestCboLiveFetch:
    def test_recent_window(self):
        from puremacro.narrative.sources.us_cbo import iter_cbo
        import datetime as dt
        today = dt.date.today()
        records = list(iter_cbo(since=(today - dt.timedelta(days=14)).isoformat(),
                                 until=today.isoformat(),
                                 max_items=3))
        if not records:
            pytest.skip("network fetch returned empty in recent window")
        for r in records:
            assert len(r) == 4
            date, text, source_url, metadata = r
            assert text.strip(), "CBO body text must be non-empty"
            assert "/publication/" in source_url


class TestCboWaybackFallback:
    def test_fetch_publication_body_falls_back_to_wayback(
        self, monkeypatch, tmp_path,
    ):
        """When direct cbo.gov fetch returns empty (DataDome challenge),
        the connector retries through the Wayback Machine."""
        from puremacro.narrative.sources import us_cbo

        # Hermetic against the SQLite HTTP cache: under marker mixes that
        # run the reference/replication tests first (the CI coverage job),
        # a sibling test can cache a non-fixture body for the same pub URL,
        # and the fallback layer's cache hit would bypass every mock below.
        monkeypatch.setenv("PUREMACRO_HTTP_NO_CACHE", "1")
        import glob, re
        from pathlib import Path

        # Find the Wayback-snapshot fixture from Phase 2.
        fix_dir = Path(__file__).parent / "fixtures" / "us_executive_mini"
        cands = sorted(glob.glob(str(fix_dir / "cbo_pub_*.html")))
        if not cands:
            pytest.skip("no cbo_pub_*.html fixture")
        fixture_html = Path(cands[0]).read_text()

        # Stub: direct fetch returns empty; Wayback fetch returns fixture.
        wb_calls = []
        def fake_get_text(url, *a, **kw):
            if "web.archive.org" in url and "/web/" in url:
                return fixture_html
            return ""  # direct cbo.gov returns empty (simulate DataDome 403)
        def fake_wayback_url(target, *a, **kw):
            wb_calls.append(target)
            return f"https://web.archive.org/web/20240115/{target}"

        monkeypatch.setattr(us_cbo, "safe_get_text", fake_get_text)
        monkeypatch.setattr(us_cbo, "wayback_snapshot_url", fake_wayback_url)
        monkeypatch.setattr(us_cbo, "safe_get_bytes",
                            lambda url, *a, **kw: b"")  # PDF irrelevant here
        # The governed fallback layer resolves its fetchers from ITS OWN
        # namespace, not us_cbo's — without these the HTML stage does real
        # cbo.gov + Wayback-CDX I/O and the test flakes when CDX throttles.
        from puremacro.narrative.sources import _fallback
        monkeypatch.setattr(_fallback, "safe_get_text", fake_get_text)
        monkeypatch.setattr(_fallback, "safe_get_text_cached", fake_get_text)
        monkeypatch.setattr(_fallback, "wayback_snapshot_url", fake_wayback_url)

        # _fetch_publication_body should: try direct → empty → call
        # wayback_snapshot_url → re-fetch via Wayback → get the fixture HTML
        body, pdf_url = us_cbo._fetch_publication_body(
            "https://www.cbo.gov/publication/60039", pub_id=60039,
        )
        # We at minimum see Wayback was queried for the HTML URL.
        assert wb_calls and wb_calls[0].startswith("https://www.cbo.gov/")


class TestIndices:
    @pytest.fixture
    def toy_records(self):
        import datetime as dt
        import pandas as pd
        return pd.DataFrame([
            (dt.date(2020, 1, 15), "Labor markets tightened amid uncertainty.",
             "https://test/erp-2020", {"chapter": "Chapter 2: Labor", "year": 2020}),
            (dt.date(2021, 1, 15), "Prices rose amid persistent inflation uncertainty.",
             "https://test/erp-2021", {"chapter": "Chapter 3: Prices", "year": 2021}),
            (dt.date(2022, 1, 15), "Labor markets stayed strong despite uncertainty.",
             "https://test/erp-2022", {"chapter": "Chapter 2: Labor", "year": 2022}),
            (dt.date(2023, 1, 15), "Overall outlook remains uncertain.",
             "https://test/erp-2023", {"chapter": "Chapter 1: Outlook", "year": 2023}),
        ], columns=["date", "text", "source_url", "metadata"])

    def test_erpui_default(self, toy_records):
        from puremacro.narrative.indices.us_executive import erpui
        ri = erpui(toy_records)
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)
        assert len(ri.series) >= 1

    def test_erpui_chapter_filter(self, toy_records):
        from puremacro.narrative.indices.us_executive import erpui
        ri = erpui(toy_records, chapter="Labor")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_sotuui_default(self, toy_records):
        from puremacro.narrative.indices.us_executive import sotuui
        ri = sotuui(toy_records)
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_cboui_topic_filter(self, toy_records):
        import pandas as pd
        import datetime as dt
        cbo_rows = pd.DataFrame([
            (dt.date(2024, 5, 1), "Estimates of budget effects of HR 1.",
             "https://test/pub-1", {"title": "Cost Estimate: HR 1",
                                      "pub_id": 1, "description": "", "pdf_url": ""}),
            (dt.date(2024, 5, 15), "Economic outlook revision.",
             "https://test/pub-2", {"title": "Budget and Economic Outlook",
                                      "pub_id": 2, "description": "", "pdf_url": ""}),
        ], columns=["date", "text", "source_url", "metadata"])
        from puremacro.narrative.indices.us_executive import cboui
        ri = cboui(cbo_rows, topic="Outlook")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_ensure_dataframe_accepts_empty_iterable(self):
        from puremacro.narrative.indices.us_executive import _ensure_dataframe
        df = _ensure_dataframe(iter([]))
        assert list(df.columns) == ["date", "text", "source_url", "metadata"]
        assert len(df) == 0
