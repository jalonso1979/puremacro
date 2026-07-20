"""Unit tests for CA WARN historical PDF parsing."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.sources.us_warn import _normalize_ca_pdf_table


def test_normalize_maps_pdf_columns_to_xlsx_schema():
    raw_rows = [
        ["Notice Date", "Effective Date", "Received Date", "Company",
         "City", "County", "No. Of Employees", "Layoff/Closure"],
        ["07/15/2019", "09/15/2019", "07/15/2019", "Acme Corp",
         "Los Angeles", "Los Angeles", "150", "Layoff"],
        ["08/01/2019", "10/01/2019", "08/01/2019", "Beta LLC",
         "San Diego", "San Diego", "75", "Closure"],
    ]
    df = _normalize_ca_pdf_table(raw_rows, source_url="http://example/fy19-20.pdf")
    assert {"Notice Date", "Company", "County", "No. Of Employees",
            "Layoff/Closure", "_source_url"}.issubset(df.columns)
    assert len(df) == 2
    assert df.iloc[0]["Company"] == "Acme Corp"
    assert df.iloc[0]["No. Of Employees"] in (150, "150")


def test_normalize_handles_multiline_headers():
    raw_rows = [
        ["Notice\nDate", "Effective\nDate", "Company", "County",
         "No. Of\nEmployees", "Layoff/\nClosure"],
        ["12/03/2020", "02/03/2021", "Gamma Inc", "Orange", "200", "Layoff"],
    ]
    df = _normalize_ca_pdf_table(raw_rows, source_url="u")
    assert "Notice Date" in df.columns
    assert "No. Of Employees" in df.columns


def test_normalize_empty_table_returns_empty_df():
    df = _normalize_ca_pdf_table([], source_url="u")
    assert df.empty
    assert isinstance(df, pd.DataFrame)


def test_normalize_drops_rows_with_no_date():
    raw_rows = [
        ["Notice Date", "Company", "County", "No. Of Employees"],
        ["", "Empty Row", "Whatever", "0"],
        ["07/15/2019", "Real Co", "LA", "100"],
    ]
    df = _normalize_ca_pdf_table(raw_rows, source_url="u")
    assert len(df) == 1
    assert df.iloc[0]["Company"] == "Real Co"


# ────────────────────────────────────────────────────────────────────
# _load_ca_historical (network + pdfplumber mocked)
# ────────────────────────────────────────────────────────────────────
from unittest import mock
from contextlib import contextmanager


class _FakePage:
    def __init__(self, tables):
        self._tables = tables
    def extract_tables(self):
        return self._tables


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def test_load_ca_historical_returns_concatenated_frame(tmp_path):
    from puremacro.narrative.sources import us_warn

    fake_table = [
        ["Notice Date", "Effective Date", "Company", "County",
         "No. Of Employees", "Layoff/Closure"],
        ["07/15/2019", "09/15/2019", "Acme Corp", "Los Angeles", "150", "Layoff"],
        ["08/01/2019", "10/01/2019", "Beta LLC", "San Diego", "75", "Closure"],
    ]
    fake_pdf = _FakePdf(pages=[_FakePage(tables=[fake_table])])

    with mock.patch.object(us_warn, "safe_get_bytes", return_value=b"FAKEPDF"), \
         mock.patch.object(us_warn.pdfplumber, "open", return_value=fake_pdf), \
         mock.patch.object(us_warn, "_CA_HISTORICAL_PDF_URLS",
                            ["http://example/fy19-20.pdf"]), \
         mock.patch.object(us_warn, "_CA_HIST_CACHE", tmp_path / "hist.parquet"):
        df = us_warn._load_ca_historical(refetch=True)
    assert not df.empty, "expected at least one row from mocked PDF"
    assert "Company" in df.columns
    companies = df["Company"].tolist()
    assert "Acme Corp" in companies
    assert "Beta LLC" in companies


def test_load_ca_historical_handles_unreachable_url(tmp_path):
    from puremacro.narrative.sources import us_warn
    with mock.patch.object(us_warn, "safe_get_bytes",
                            side_effect=RuntimeError("boom")), \
         mock.patch.object(us_warn, "_CA_HISTORICAL_PDF_URLS",
                            ["http://example/fy19-20.pdf"]), \
         mock.patch.object(us_warn, "_CA_HIST_CACHE", tmp_path / "hist.parquet"):
        df = us_warn._load_ca_historical(refetch=True)
    assert df.empty
    assert isinstance(df, pd.DataFrame)


def test_load_ca_merges_live_and_historical_dedups_overlap(tmp_path):
    """Live xlsx and the most recent historical PDF will both contain the
    last few filings — _load_ca must dedup so each filing appears once."""
    from puremacro.narrative.sources import us_warn
    live = pd.DataFrame([
        {"Notice Date": pd.Timestamp("2024-12-01"),
         "Company": "Acme Corp", "County": "LA",
         "No. Of Employees": 100, "_source_url": "live.xlsx"},
    ])
    hist = pd.DataFrame([
        {"Notice Date": pd.Timestamp("2024-12-01"),
         "Company": "Acme Corp", "County": "LA",
         "No. Of Employees": 100,
         "_source_url": "fy24-25.pdf"},
        {"Notice Date": pd.Timestamp("2019-07-15"),
         "Company": "Beta LLC", "County": "SD",
         "No. Of Employees": 75,
         "_source_url": "fy19-20.pdf"},
    ])
    with mock.patch.object(us_warn, "_load_ca_live", return_value=live), \
         mock.patch.object(us_warn, "_load_ca_historical", return_value=hist):
        df = us_warn._load_ca(refetch=False)
    assert len(df) == 2
    assert set(df["Company"]) == {"Acme Corp", "Beta LLC"}


def test_load_ca_returns_empty_when_both_sources_empty():
    from puremacro.narrative.sources import us_warn
    with mock.patch.object(us_warn, "_load_ca_live", return_value=pd.DataFrame()), \
         mock.patch.object(us_warn, "_load_ca_historical", return_value=pd.DataFrame()):
        df = us_warn._load_ca(refetch=False)
    assert df.empty
    assert isinstance(df, pd.DataFrame)


# ────────────────────────────────────────────────────────────────────
# NY historical archive (HTML table scrape)
# ────────────────────────────────────────────────────────────────────

_NY_2024_FIXTURE_HTML = """\
<html><body>
<h1>2024 WARN Notices</h1>
<table>
  <thead>
    <tr><th>Company Name</th><th>Region</th><th>Date Posted</th><th>Notice Dated</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="/warn-spencerarl-new-york-inc-region-north-country-12312024">SpencerARL New York, Inc.</a></td>
      <td>North Country</td>
      <td>12/31/2024</td>
      <td>7/9/2024 (Amended 10/28/2024)</td>
    </tr>
    <tr>
      <td><a href="/warn-mv-transportation-12312024">MV Transportation, Inc.</a></td>
      <td>New York City</td>
      <td>12/31/2024</td>
      <td>11/11/2024</td>
    </tr>
    <tr>
      <td><a href="/warn-party-city-12272024">Party City Corporation</a></td>
      <td>New York City</td>
      <td>12/27/2024</td>
      <td>12/20/2024</td>
    </tr>
  </tbody>
</table>
</body></html>
"""


def test_parse_ny_archive_html_returns_records():
    from puremacro.narrative.sources.us_warn import _parse_ny_archive_html
    rows = _parse_ny_archive_html(_NY_2024_FIXTURE_HTML.encode(),
                                   source_url="http://example/2024")
    assert len(rows) == 3
    assert rows[0]["Company"] == "SpencerARL New York, Inc."
    assert rows[0]["Region"] == "North Country"
    assert isinstance(rows[0]["Date Posted"], pd.Timestamp)
    assert rows[0]["Date Posted"].year == 2024
    # Notice Dated parses to a Timestamp even with amendment text
    nd = rows[0]["Notice Dated"]
    assert pd.isna(nd) or isinstance(nd, pd.Timestamp)


def test_parse_ny_archive_html_handles_empty_table():
    from puremacro.narrative.sources.us_warn import _parse_ny_archive_html
    html = "<html><body><table><thead><tr><th>A</th></tr></thead><tbody></tbody></table></body></html>"
    rows = _parse_ny_archive_html(html.encode(), source_url="u")
    assert rows == []


def test_load_ny_historical_returns_concatenated_frame(tmp_path):
    from puremacro.narrative.sources import us_warn
    with mock.patch.object(us_warn, "safe_get_bytes",
                           return_value=_NY_2024_FIXTURE_HTML.encode()), \
         mock.patch.object(us_warn, "_NY_HISTORICAL_HTML_URLS",
                           ["http://example/2024"]), \
         mock.patch.object(us_warn, "_NY_HIST_CACHE", tmp_path / "ny_hist.parquet"):
        df = us_warn._load_ny_historical(refetch=True)
    assert len(df) == 3
    assert "Company" in df.columns
    assert set(df["Company"]) >= {"SpencerARL New York, Inc.",
                                   "MV Transportation, Inc.",
                                   "Party City Corporation"}


def test_load_ny_historical_handles_unreachable_url(tmp_path):
    from puremacro.narrative.sources import us_warn
    with mock.patch.object(us_warn, "safe_get_bytes",
                           side_effect=RuntimeError("boom")), \
         mock.patch.object(us_warn, "_NY_HISTORICAL_HTML_URLS",
                           ["http://example/2024"]), \
         mock.patch.object(us_warn, "_NY_HIST_CACHE", tmp_path / "ny_hist.parquet"):
        df = us_warn._load_ny_historical(refetch=True)
    assert df.empty
    assert isinstance(df, pd.DataFrame)


def test_load_ny_merges_live_and_historical_dedups(tmp_path):
    """Live uses 'Business Legal Name'; historical uses 'Company'. The
    merge must harmonise these before dedup so an overlapping filing
    isn't double-counted."""
    from puremacro.narrative.sources import us_warn
    live = pd.DataFrame([
        # Live xlsx schema mirrors the real Tableau CSV
        {"Date Posted": pd.Timestamp("2025-01-15"),
         "Business Legal Name": "Acme NY",
         "Impacted Site County": "Westchester",
         "_source_url": "live.csv"},
    ])
    hist = pd.DataFrame([
        # Same filing as live but in the historical HTML schema
        {"Date Posted": pd.Timestamp("2025-01-15"),
         "Company": "Acme NY", "Region": "Hudson Valley",
         "_source_url": "2024.html"},
        # Older filing only in historical
        {"Date Posted": pd.Timestamp("2024-06-15"),
         "Company": "Beta NY", "Region": "Western NY",
         "_source_url": "2024.html"},
    ])
    with mock.patch.object(us_warn, "_load_ny_live", return_value=live), \
         mock.patch.object(us_warn, "_load_ny_historical", return_value=hist):
        df = us_warn._load_ny(refetch=False)
    assert len(df) == 2
    # After harmonisation, company name lives in 'Business Legal Name'
    assert set(df["Business Legal Name"]) == {"Acme NY", "Beta NY"}


def test_iter_us_warn_wa_yields_records_when_cache_present():
    """If data/processed/warn_wa.parquet is on disk (built one-shot by
    tools/scrape_wa_warn.py), iter_us_warn(state='WA') yields properly
    structured 5-tuples. If the cache isn't present we skip — the
    scraper is the prerequisite."""
    from puremacro.narrative.sources import us_warn
    if not us_warn._WA_CACHE.exists():
        pytest.skip(f"WA cache not built yet at {us_warn._WA_CACHE}")
    records = list(us_warn.iter_us_warn(state="WA"))
    if not records:
        pytest.skip("WA cache empty")
    assert len(records) >= 500, f"WA only {len(records)} records"
    r0 = records[0]
    assert len(r0) == 5
    assert isinstance(r0[0], pd.Timestamp)
    assert r0[3]["state"] == "WA"


def test_iter_us_warn_rejects_unknown_state_with_three_options():
    from puremacro.narrative.sources.us_warn import iter_us_warn
    with pytest.raises(ValueError, match="CA.*NY.*WA"):
        list(iter_us_warn(state="ZZ"))


def test_load_ca_historical_skips_repeated_header_on_page_2(tmp_path):
    """When a PDF has multiple pages whose tables all start with the same
    header row, the body of subsequent pages should be appended without
    re-including the header as a data row."""
    from puremacro.narrative.sources import us_warn
    header = ["Notice Date", "Company", "County"]
    page1_table = [header, ["07/15/2019", "Acme", "LA"]]
    page2_table = [header, ["08/01/2019", "Beta", "SD"]]
    fake_pdf = _FakePdf(pages=[_FakePage([page1_table]),
                                _FakePage([page2_table])])
    with mock.patch.object(us_warn, "safe_get_bytes", return_value=b"FAKE"), \
         mock.patch.object(us_warn.pdfplumber, "open", return_value=fake_pdf), \
         mock.patch.object(us_warn, "_CA_HISTORICAL_PDF_URLS", ["x"]), \
         mock.patch.object(us_warn, "_CA_HIST_CACHE", tmp_path / "h.parquet"):
        df = us_warn._load_ca_historical(refetch=True)
    assert len(df) == 2
    assert set(df["Company"]) == {"Acme", "Beta"}
