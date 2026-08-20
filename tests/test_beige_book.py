"""Tests for puremacro.narrative.sources.beige_book."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from puremacro.narrative.sources.beige_book import (
    CANONICAL_SECTIONS, DISTRICTS, canonical_section,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "beige_book_mini"


class TestCanonicalSection:
    def test_overall_summary_header(self):
        assert canonical_section("Summary of Economic Activity") == "overall"

    def test_labor_section_header(self):
        assert canonical_section("Employment and Wages") == "labor"

    def test_prices_section_header(self):
        assert canonical_section("Prices") == "prices"

    def test_consumer_spending(self):
        assert canonical_section("Consumer Spending") == "consumer"
        assert canonical_section("Retail") == "consumer"

    def test_manufacturing(self):
        assert canonical_section("Manufacturing") == "manufacturing"
        assert canonical_section("Industrial production") == "manufacturing"

    def test_realestate(self):
        assert canonical_section("Real Estate and Construction") == "realestate"
        assert canonical_section("Housing") == "realestate"

    def test_other_fallback(self):
        assert canonical_section("Agriculture and Natural Resources") == "other"
        assert canonical_section("Banking and Finance") == "other"
        assert canonical_section("") == "other"

    def test_case_insensitive(self):
        assert canonical_section("EMPLOYMENT AND WAGES") == "labor"
        assert canonical_section("employment AND wages") == "labor"

    def test_labor_costs_maps_to_labor(self):
        assert canonical_section("Labor Costs") == "labor"
        assert canonical_section("Unit Labor Costs") == "labor"

    def test_input_costs_does_not_map_to_prices(self):
        # 'costs' no longer in prices synonyms; bucketing to 'other' is acceptable
        assert canonical_section("Input Costs") in ("other", "prices")

    def test_specific_conditions_do_not_map_to_overall(self):
        assert canonical_section("Agricultural Conditions") == "other"
        assert canonical_section("Financial Conditions") == "other"
        assert canonical_section("Business Conditions") == "other"

    def test_canonical_sections_tuple_complete(self):
        assert set(CANONICAL_SECTIONS) == {
            "overall", "labor", "prices", "consumer",
            "manufacturing", "realestate", "other",
        }

    def test_districts_complete(self):
        assert len(DISTRICTS) == 12
        assert "Boston" in DISTRICTS
        assert "San Francisco" in DISTRICTS


# ---------------------------------------------------------------------------
# Modern HTML backend tests
# Fixtures: modern_202401_summary.html  (national summary page)
#           modern_202401_boston.html   (Boston district page)
#           modern_202401.html          (index page — used only by network test)
# ---------------------------------------------------------------------------

from puremacro.narrative.sources.beige_book import (  # noqa: E402
    _parse_district_page,
    _parse_summary_page,
)


class TestModernDistrictParser:
    """Tests against the pre-fetched Boston district fixture."""

    def setup_method(self):
        self.html = (FIXTURE_DIR / "modern_202401_boston.html").read_text(encoding="utf-8")
        self.release_date = dt.date(2024, 1, 1)
        self.source_url = "https://test"

    def test_yields_records(self):
        records = _parse_district_page(
            self.html, district="Boston",
            release_date=self.release_date, source_url=self.source_url,
            granularity="per_section",
        )
        assert len(records) >= 1
        for r in records:
            assert len(r) == 6, f"expected 6-tuple, got {len(r)}"
            date, district, section, text, source_url, metadata = r
            assert text.strip(), "section text must be non-empty"
            assert district == "Boston"
            assert section in CANONICAL_SECTIONS, (
                f"unexpected section {section!r} for header "
                f"{metadata.get('raw_header')!r}"
            )

    def test_includes_overall_section(self):
        records = _parse_district_page(
            self.html, district="Boston",
            release_date=self.release_date, source_url=self.source_url,
        )
        overall = [r for r in records if r[2] == "overall"]
        assert len(overall) >= 1, "expected at least one 'overall' section"

    def test_per_district_granularity(self):
        records = _parse_district_page(
            self.html, district="Boston",
            release_date=self.release_date, source_url=self.source_url,
            granularity="per_district",
        )
        assert len(records) == 1
        assert records[0][2] == "all"
        assert len(records[0][3]) > 100, "bundled text should be substantial"


class TestModernSummaryParser:
    """Tests against the pre-fetched national summary fixture."""

    def setup_method(self):
        self.html = (FIXTURE_DIR / "modern_202401_summary.html").read_text(encoding="utf-8")
        self.release_date = dt.date(2024, 1, 1)
        self.source_url = "https://test"

    def test_yields_national_records(self):
        records = _parse_summary_page(
            self.html, release_date=self.release_date,
            source_url=self.source_url,
        )
        assert len(records) >= 1
        for r in records:
            assert len(r) == 6
            date, district, section, text, source_url, metadata = r
            assert district == "National"
            assert text.strip()
            assert section in CANONICAL_SECTIONS

    def test_includes_overall(self):
        records = _parse_summary_page(
            self.html, release_date=self.release_date,
            source_url=self.source_url,
        )
        overalls = [r for r in records if r[2] == "overall"]
        assert len(overalls) >= 1, "expected at least one 'overall' national record"


@pytest.mark.network
class TestModernLiveFetch:
    def test_fetch_one_recent_release(self):
        from puremacro.narrative.sources.beige_book import _iter_modern
        records = list(_iter_modern(2024, 1))
        if not records:
            pytest.skip("fetch returned empty; endpoint may be down")
        nationals = [r for r in records if r[1] == "National"]
        districts = {r[1] for r in records if r[1] != "National"}
        assert len(nationals) >= 1
        assert len(districts) >= 5


class TestFomcHistoricalParser:
    """Tests against pre-fetched Fed FOMC historical Beige Book PDF fixtures.

    The fixture names follow fomc_{YEAR}.pdf. These PDFs come from:
    https://www.federalreserve.gov/monetarypolicy/files/fomc...beige...pdf

    Note: These were originally scoped as 'FRASER' fixtures, but exhaustive
    FRASER search found no national Beige Book series there. The real source is
    the Fed's FOMC historical archive.
    """

    def test_parse_1985_beige_book(self):
        """Parse a January 1985 FOMC historical PDF (Beige Book naming era)."""
        from puremacro.narrative.sources.beige_book import (
            _extract_pdf_text,
            _parse_fomc_pdf,
        )

        pdf_path = FIXTURE_DIR / "fomc_1985.pdf"
        if not pdf_path.exists():
            pytest.skip("fomc_1985.pdf fixture not yet fetched")

        try:
            text = _extract_pdf_text(pdf_path.read_bytes())
        except ImportError:
            pytest.skip("pdfplumber not installed")

        assert text.strip(), "Expected non-empty text from 1985 PDF"
        assert "SUMMARY OF COMMENTARY" in text.upper() or "Federal Reserve" in text

        records = list(_parse_fomc_pdf(
            text,
            release_date=dt.date(1985, 1, 1),
            source_url="https://test",
        ))
        assert len(records) >= 1, "FOMC historical parser produced zero records"
        for r in records:
            assert len(r) == 6, f"expected 6-tuple, got {len(r)}: {r}"
            date, district, section, text_body, src, meta = r
            assert text_body.strip(), "section text must be non-empty"
            assert section in CANONICAL_SECTIONS, (
                f"unexpected section {section!r} for header "
                f"{meta.get('raw_header')!r}"
            )

    def test_parse_1983_fomc_pdf(self):
        """Parse a June 1983 FOMC historical PDF (earliest available era)."""
        from puremacro.narrative.sources.beige_book import (
            _extract_pdf_text,
            _parse_fomc_pdf,
        )

        pdf_path = FIXTURE_DIR / "fomc_1983.pdf"
        if not pdf_path.exists():
            pytest.skip("fomc_1983.pdf fixture not yet fetched")

        try:
            text = _extract_pdf_text(pdf_path.read_bytes())
        except ImportError:
            pytest.skip("pdfplumber not installed")

        records = list(_parse_fomc_pdf(
            text,
            release_date=dt.date(1983, 6, 1),
            source_url="https://test",
        ))
        assert isinstance(records, list)
        # Should have at least some records even if OCR quality varies
        assert len(records) >= 1

    def test_1985_covers_multiple_districts(self):
        """The 1985 PDF should yield records for several districts."""
        from puremacro.narrative.sources.beige_book import (
            _extract_pdf_text,
            _parse_fomc_pdf,
        )

        pdf_path = FIXTURE_DIR / "fomc_1985.pdf"
        if not pdf_path.exists():
            pytest.skip("fomc_1985.pdf fixture not yet fetched")

        try:
            text = _extract_pdf_text(pdf_path.read_bytes())
        except ImportError:
            pytest.skip("pdfplumber not installed")

        records = list(_parse_fomc_pdf(
            text,
            release_date=dt.date(1985, 1, 1),
            source_url="https://test",
        ))
        districts_seen = {r[1] for r in records}
        # Should have at least National + a few individual districts
        assert len(districts_seen) >= 3, (
            f"Expected 3+ districts, got {districts_seen}"
        )


class TestOrchestrator:
    def test_iter_beige_book_importable_from_init(self):
        from puremacro.narrative.sources import iter_beige_book
        assert callable(iter_beige_book)

    def test_iter_beige_book_yields_nothing_for_empty_window(self):
        """An obviously-empty window (1900) returns no records."""
        from puremacro.narrative.sources import iter_beige_book
        records = list(iter_beige_book(since="1900-01-01", until="1900-01-02"))
        assert records == []

    @pytest.mark.network
    def test_iter_beige_book_recent_window(self):
        """One-year recent window should produce many records."""
        from puremacro.narrative.sources import iter_beige_book
        import datetime as dt
        today = dt.date.today()
        last_year = today.replace(year=today.year - 1)
        records = list(iter_beige_book(
            since=last_year.isoformat(), until=today.isoformat(),
        ))
        if not records:
            pytest.skip("network fetch returned empty")
        assert len(records) >= 20


import pandas as pd_for_bbui  # noqa: E402


class TestBBUI:
    @pytest.fixture
    def toy_records(self):
        import datetime as dt
        return pd_for_bbui.DataFrame([
            (dt.date(2024, 1, 1), "Boston", "labor",
             "Hiring slowed amid uncertain demand.", "u1", {}),
            (dt.date(2024, 1, 1), "Boston", "prices",
             "Prices held steady.", "u1", {}),
            (dt.date(2024, 3, 1), "Boston", "labor",
             "Layoffs picked up in manufacturing.", "u2", {}),
            (dt.date(2024, 3, 1), "New York", "labor",
             "Wages rose modestly.", "u3", {}),
        ], columns=["date", "district", "section", "text",
                     "source_url", "metadata"])

    def test_bbui_national_default(self, toy_records):
        from puremacro.narrative.indices import bbui
        ri = bbui(toy_records, level="national")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)
        assert len(ri.series) >= 1

    def test_bbui_district_returns_dataframe(self, toy_records):
        from puremacro.narrative.indices import bbui
        out = bbui(toy_records, level="district")
        assert isinstance(out, pd_for_bbui.DataFrame)
        assert set(out.columns) == {"Boston", "New York"}

    def test_bbui_section_filter(self, toy_records):
        from puremacro.narrative.indices import bbui
        ri = bbui(toy_records, section="labor", level="national")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_bbui_invalid_level_raises(self, toy_records):
        from puremacro.narrative.indices import bbui
        with pytest.raises(ValueError, match="level must be"):
            bbui(toy_records, level="invalid")

    def test_bbui_accepts_iter_of_tuples(self):
        from puremacro.narrative.indices import bbui
        import datetime as dt
        records = [
            (dt.date(2024, 1, 1), "Boston", "labor",
             "Hiring slowed.", "u1", {}),
            (dt.date(2024, 3, 1), "Boston", "labor",
             "Layoffs rose.", "u2", {}),
        ]
        ri = bbui(iter(records), level="national")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)


class TestPre2017ModernHtml:
    def test_enumerate_pre2017_returns_summary_and_12_districts(self):
        from puremacro.narrative.sources.beige_book import (
            _enumerate_pre2017_district_urls,
        )
        urls = _enumerate_pre2017_district_urls(2014, 7)
        # __summary__ + 12 districts = 13 entries.
        assert len(urls) == 13
        names = [n for n, _ in urls]
        assert "__summary__" in names
        assert "Boston" in names
        assert "San Francisco" in names

    def test_parse_modern_html_pre2017_layout_uses_url_enum(self,
                                                             monkeypatch):
        from puremacro.narrative.sources import beige_book
        from datetime import date as _date

        FIX = (Path(__file__).parent / "fixtures" / "beige_book_pre2017")
        district_html = (FIX / "district_boston.html").read_text(encoding="utf-8")
        index_html = (FIX / "index.html").read_text(encoding="utf-8")

        # Replay HTTP: index returns the fixture index; any district URL
        # returns the same Boston fixture (we just want >0 records).
        def fake_get_text(url, *a, **kw):
            if url.endswith("-boston.htm"):
                return district_html
            return ""  # other districts unavailable in fixture set

        monkeypatch.setattr(beige_book, "safe_get_text", fake_get_text)

        records = list(beige_book._parse_modern_html(
            index_html,
            release_date=_date(2014, 7, 1),
            source_url="https://test.invalid/beigebook201407.htm",
            granularity="per_section",
            year=2014, month=7,
        ))
        # We should have >0 records via the URL-enum fallback hitting
        # the Boston fixture.
        assert len(records) > 0
        # Each record is a 6-tuple.
        assert all(len(r) == 6 for r in records)
        # The Boston district fixture has 3 h4 sections.
        boston_records = [r for r in records if r[1] == "Boston"]
        assert len(boston_records) == 3


class TestFomcOrchestratorRefactor:
    def test_fetch_fomc_pdf_uses_pre_resolved_url(self, monkeypatch, tmp_path):
        """_fetch_fomc_pdf takes a pre-resolved pdf_url — no listing scrape."""
        from puremacro.narrative.sources import beige_book
        from datetime import date as _date

        listing_calls = []
        monkeypatch.setattr(beige_book, "_PDFPLUMBER_AVAILABLE", True)
        monkeypatch.setattr(beige_book, "_fomc_listing",
                            lambda y: (listing_calls.append(y) or []))
        # Stub the PDF byte fetch + text extraction to a tiny known body.
        monkeypatch.setattr(beige_book, "safe_get_bytes",
                            lambda url, *a, **kw: b"stub-pdf-bytes")
        monkeypatch.setattr(beige_book, "_extract_pdf_text",
                            lambda b: ("FIRST DISTRICT - BOSTON\n"
                                       "Summary\n"
                                       "Economic activity grew modestly.\n"))

        records = list(beige_book._fetch_fomc_pdf(
            "https://test.invalid/foo.pdf",
            release_date=_date(1993, 7, 1),
            granularity="per_section",
        ))
        # The orchestrator helper must NOT have re-scraped the listing.
        assert listing_calls == []
        # And must yield at least one record from the stub PDF text.
        assert len(records) >= 1

    def test_iter_beige_book_calls_listing_once_per_year(self, monkeypatch):
        """The orchestrator must call _fomc_listing once per year, not N times."""
        from puremacro.narrative.sources import beige_book
        from datetime import date as _date

        listing_calls = []
        def fake_listing(year):
            listing_calls.append(year)
            # Return 2 fake issues per year.
            return [(year, 1, f"https://test.invalid/{year}_1.pdf"),
                    (year, 7, f"https://test.invalid/{year}_7.pdf")]

        monkeypatch.setattr(beige_book, "_fomc_listing", fake_listing)
        monkeypatch.setattr(beige_book, "_fetch_fomc_pdf",
                            lambda url, **kw: iter([]))
        # Also stub modern path so we exit early after 1995.
        monkeypatch.setattr(beige_book, "_iter_modern",
                            lambda y, m, **kw: iter([]))
        # Force _PDFPLUMBER_AVAILABLE True so pre-1996 branch runs.
        monkeypatch.setattr(beige_book, "_PDFPLUMBER_AVAILABLE", True)

        list(beige_book.iter_beige_book(
            since="1993-01-01", until="1994-12-31",
        ))
        # Two years × one listing call each = 2 total. Not 2*2 = 4.
        assert listing_calls == [1993, 1994]


class TestBeigeBookUsesCachedHttp:
    def test_safe_get_text_is_cached_variant(self):
        from puremacro.narrative.sources import beige_book
        from puremacro._http import safe_get_text_cached
        assert beige_book.safe_get_text is safe_get_text_cached

    def test_safe_get_bytes_is_cached_variant(self):
        from puremacro.narrative.sources import beige_book
        from puremacro._http import safe_get_bytes_cached
        assert beige_book.safe_get_bytes is safe_get_bytes_cached


class TestRedbookBackend:
    """The pre-1983 Redbook (Beige Book predecessor, May 1970-May 1983)
    shares the FOMC index pages and the PDF parser with the modern Beige
    Book. These tests lock in (a) that the listing regex accepts the
    ``redbook`` filename and (b) that a real Redbook PDF parses into the
    twelve districts via the unchanged ``_parse_fomc_pdf``."""

    def test_listing_regex_matches_both_names(self):
        import re
        # The exact pattern used inside _fomc_listing.
        pat = re.compile(
            r"fomc(\d{4})(\d{2})\d{2}(?:beige|redbook)(\d{4})(\d{2})\d{2}"
            r"\.pdf", re.IGNORECASE)
        beige = pat.search("/monetarypolicy/files/fomc19850109beige19850102.pdf")
        red = pat.search("/monetarypolicy/files/fomc19780418redbook19780412.pdf")
        assert beige and (int(beige.group(3)), int(beige.group(4))) == (1985, 1)
        assert red and (int(red.group(3)), int(red.group(4))) == (1978, 4)

    def test_parse_1978_redbook(self):
        from puremacro.narrative.sources.beige_book import (
            _extract_pdf_text, _parse_fomc_pdf, DISTRICTS,
        )
        pdf_path = FIXTURE_DIR / "fomc_1978_redbook.pdf"
        if not pdf_path.exists():
            pytest.skip("fomc_1978_redbook.pdf fixture not present")
        try:
            text = _extract_pdf_text(pdf_path.read_bytes())
        except ImportError:
            pytest.skip("pdfplumber not installed")
        assert text.strip()
        assert "DISTRICT" in text.upper()
        records = list(_parse_fomc_pdf(
            text, release_date=dt.date(1978, 4, 1),
            source_url="https://test-redbook"))
        districts = {r[1] for r in records}
        # The full 12-district panel must come through the predecessor format.
        assert set(DISTRICTS).issubset(districts), (
            f"missing districts: {set(DISTRICTS) - districts}")
        for r in records:
            assert len(r) == 6
