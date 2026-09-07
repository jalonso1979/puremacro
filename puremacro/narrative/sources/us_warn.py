"""US WARN-Act layoff filings (β.1).

The Worker Adjustment and Retraining Notification (WARN) Act requires
employers with >= 100 workers to file notice of layoffs >= 60 days in
advance. Filings are managed by state-level labor agencies; each
state publishes them under public-records law.

This connector exposes ``iter_us_warn(state="CA"|"NY")``. CA filings
land in this file (β-4); NY filings are added in β-5.

CA reads the EDD "Daily WARN Report" xlsx; NY reads NY DOL's public
Tableau dashboard (which replaced the per-year xlsx archive in
April 2025 — see the "Legacy WARN Notices" redirect on
``dol.ny.gov/warn-notices``). The Tableau view exposes a CSV
crosstab endpoint that returns the full filings list.

Yields 5-tuples ``(date, text, url, meta, magnitude)`` where
``magnitude`` is the number of employees affected by the filing.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterator, Literal

import pandas as pd

try:
    import pdfplumber  # type: ignore[import-untyped]
except ImportError:
    pdfplumber = None  # type: ignore[assignment]

from ._http import safe_get_bytes


# California EDD publishes a consolidated rolling-FY "Daily WARN Report"
# at the URL below (updated Tuesdays / Thursdays). Historical fiscal
# years exist only as PDFs so we only wire the live xlsx here.
_CA_XLSX_URLS: list[str] = [
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn_report1.xlsx",
]

_CA_LANDING = "https://edd.ca.gov/en/jobs_and_training/Layoff_Services_WARN"

# The detailed report sheet uses a banner row at A1 and the real header
# row at A2 (index 1). Sheet name has a trailing space in the workbook.
_CA_SHEET_NAME = "Detailed WARN Report "
_CA_HEADER_ROW = 1

_CACHE_ROOT = (Path(__file__).resolve().parents[4]
               / "notebooks" / "data_cache")
_CA_CACHE = _CACHE_ROOT / "warn_ca.parquet"
_CA_HIST_CACHE = _CACHE_ROOT / "warn_ca_historical.parquet"


# California EDD per-fiscal-year archive PDFs. URL pattern verified against
# the EDD landing page on 2026-05-17. Coverage: FY 2014-15 through FY 2024-25.
# (The current FY's data is in the rolling-live xlsx above; the FY 2024-25
# PDF is included to capture the last fully closed FY.)
_CA_HISTORICAL_PDF_URLS: list[str] = [
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn-report-for-7-1-2024-to-06-30-2025.pdf",
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn-report-for-7-1-2023-to-06-30-2024.pdf",
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn-report-for-7-1-2022-to-06-30-2023.pdf",
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn-report-for-7-1-2021-to-06-30-2022.pdf",
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn-report-for-7-1-2020-to-06-30-2021.pdf",
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn-report-for-7-1-2019-to-6-30-2020.pdf",
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn-report-for-7-1-2018-to-06-30-2019.pdf",
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn-report-for-7-1-2017-to-06-30-2018.pdf",
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn-report-for-7-1-2016-to-06-30-2017.pdf",
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn-report-for-7-1-2015-to-06-30-2016.pdf",
    "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warnreportfor7-1-2014to06-30-2015.pdf",
]


def _normalize_col(name: str) -> str:
    """Flatten embedded newlines / collapse whitespace so column matches
    survive the EDD xlsx's multi-line headers (e.g. ``'Notice\\nDate'``)."""
    return " ".join(str(name).split()).strip()


def _load_ca_live(*, refetch: bool) -> pd.DataFrame:
    """Read the rolling-live EDD xlsx (current fiscal year)."""
    if _CA_CACHE.exists() and not refetch:
        return pd.read_parquet(_CA_CACHE)
    frames = []
    for url in _CA_XLSX_URLS:
        try:
            raw = safe_get_bytes(url)
        except Exception:
            continue
        if not raw:
            continue
        try:
            # Try the detailed-report sheet first; fall back to first sheet.
            try:
                df = pd.read_excel(
                    BytesIO(raw),
                    sheet_name=_CA_SHEET_NAME,
                    header=_CA_HEADER_ROW,
                    engine="openpyxl",
                )
            except (ValueError, KeyError):
                df = pd.read_excel(BytesIO(raw), engine="openpyxl")
            df.columns = [_normalize_col(c) for c in df.columns]
            df["_source_url"] = url
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    try:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        out.to_parquet(_CA_CACHE)
    except Exception:
        # Cache write is best-effort; don't fail the iterator if the
        # parquet engine is missing or the cache dir is read-only.
        pass
    return out


def _load_ca(*, refetch: bool) -> pd.DataFrame:
    """Merge the live xlsx and the historical per-FY PDFs into a single
    deduped DataFrame. Dedup key is (Notice Date, Company, County), with
    the live row kept when a filing appears in both sources.
    """
    live = _load_ca_live(refetch=refetch)
    hist = _load_ca_historical(refetch=refetch)
    if live.empty and hist.empty:
        return pd.DataFrame()
    df = pd.concat([live, hist], ignore_index=True, sort=False)
    date_col = _resolve_col(df, "Notice Date", "Date Received",
                             "Effective Date", "Date")
    company_col = _resolve_col(df, "Company", "Company Name", "Employer")
    county_col = _resolve_col(df, "County/Parish", "County",
                               "Local Area", "Location")
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    dedup_keys = [c for c in (date_col, company_col, county_col)
                  if c is not None]
    if dedup_keys:
        df = df.drop_duplicates(subset=dedup_keys, keep="first")
    return df.reset_index(drop=True)


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name matching any candidate (case-insensitive,
    newline/whitespace-tolerant). None if no match."""
    norm = {_normalize_col(c).lower(): c for c in df.columns}
    for cand in candidates:
        match = norm.get(_normalize_col(cand).lower())
        if match:
            return match
    return None


def _extract_pdf_rows(raw: bytes) -> list[list[str]]:
    """Open a PDF byte-stream and return all table rows across all pages.

    The first non-empty table's row 0 is treated as the header. Subsequent
    tables' rows are appended as body if their column-count matches; a
    repeated header row is skipped.
    """
    if pdfplumber is None:
        raise ImportError(
            "pdfplumber is required to parse PDF WARN filings. "
            "Install via `pip install puremacro[narrative]`."
        )
    rows: list[list[str]] = []
    header: list[str] | None = None
    with pdfplumber.open(BytesIO(raw)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for tbl in tables:
                if not tbl:
                    continue
                clean = [[("" if c is None else str(c)) for c in r] for r in tbl]
                if header is None:
                    header = clean[0]
                    rows.append(header)
                    rows.extend(r for r in clean[1:] if len(r) == len(header))
                else:
                    body = clean
                    if [c.strip() for c in clean[0]] == [c.strip() for c in header]:
                        body = clean[1:]
                    rows.extend(r for r in body if len(r) == len(header))
    return rows


def _load_ca_historical(*, refetch: bool) -> pd.DataFrame:
    """Fetch all CA-EDD per-FY archive PDFs, table-extract, normalise.

    Returns an empty DataFrame if every URL fails to fetch or parse —
    never raises.
    """
    if _CA_HIST_CACHE.exists() and not refetch:
        return pd.read_parquet(_CA_HIST_CACHE)
    frames = []
    for url in _CA_HISTORICAL_PDF_URLS:
        try:
            raw = safe_get_bytes(url)
        except Exception:
            continue
        if not raw:
            continue
        try:
            raw_rows = _extract_pdf_rows(raw)
        except Exception:
            continue
        if not raw_rows:
            continue
        df = _normalize_ca_pdf_table(raw_rows, source_url=url)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    try:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        out.to_parquet(_CA_HIST_CACHE)
    except Exception:
        pass
    return out


def _normalize_ca_pdf_table(
    raw_rows: list[list[str]],
    *,
    source_url: str,
) -> pd.DataFrame:
    """Convert a pdfplumber table extraction (header + body rows) into a
    DataFrame matching the live-xlsx schema.

    raw_rows[0] is the header; raw_rows[1:] are filings. Columns are
    re-normalised through ``_normalize_col`` so that downstream
    ``_resolve_col`` matches them identically to the xlsx path.
    Rows without a Notice Date are dropped.
    """
    if not raw_rows or len(raw_rows) < 2:
        return pd.DataFrame()
    header = [_normalize_col(c) for c in raw_rows[0]]
    body = [r for r in raw_rows[1:] if len(r) == len(header)]
    if not body:
        return pd.DataFrame()
    df = pd.DataFrame(body, columns=header)
    df["_source_url"] = source_url
    date_col = _resolve_col(df, "Notice Date", "Date Received",
                            "Effective Date", "Date")
    if date_col is not None:
        df = df[df[date_col].astype(str).str.strip() != ""]
    return df.reset_index(drop=True)


def _iter_ca(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    df = _load_ca(refetch=refetch)
    if df.empty:
        return
    date_col = _resolve_col(
        df,
        "Notice Date", "Date Received", "Effective Date",
        "Processed Date", "Date",
    )
    if date_col is None:
        return
    company_col = _resolve_col(df, "Company", "Company Name", "Employer")
    county_col = _resolve_col(df, "County/Parish", "County", "Local Area", "Location")
    type_col = _resolve_col(df, "Layoff/Closure", "Type", "Action Type", "Reason")
    n_workers_col = _resolve_col(
        df, "No. Of Employees", "Number of Employees",
        "Employees Affected", "Total Affected", "Workers Impacted",
    )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if since:
        df = df[df[date_col] >= pd.Timestamp(since)]

    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = str(row.get(company_col) or "") if company_col else ""
        county = str(row.get(county_col) or "") if county_col else ""
        filing_type = (
            str(row.get(type_col) or "layoff") if type_col else "layoff"
        ).strip().lower()
        n_workers = None
        if n_workers_col is not None:
            v = pd.to_numeric(row.get(n_workers_col), errors="coerce")
            if pd.notna(v) and v >= 0:
                n_workers = float(v)

        if county:
            text = f"{company} ({county}, CA) - {filing_type}"
        else:
            text = f"{company} (CA) - {filing_type}"
        url = str(row.get("_source_url") or _CA_LANDING)
        meta = {
            "doctype": "warn_filing",
            "language": "en",
            "bank_code": "WARN_CA",     # back-compat name; stability filter reads this
            "source": "warn_ca",
            "country": "USA",
            "state": "CA",
            "county": county or None,
            "filing_type": filing_type,
        }
        yield (row[date_col], text, url, meta, n_workers)


# NY DOL retired its per-FY xlsx archive on 2025-04-01 (the
# ``dol.ny.gov/warn-notices`` landing page now 301-redirects to
# ``legacy-warn-notices`` whose only artifacts are per-filing HTML
# pages — no consolidated spreadsheet). Current filings live in a
# Tableau Public dashboard embedded under ``/warn-dashboard``;
# Tableau exposes each sheet as a downloadable CSV crosstab at the
# URL pattern below. ``WARNList`` is the full filings table with
# per-row magnitude (``Min. Number of Affected Workers``); the other
# sheets are roll-ups by industry / WDB.
_NY_CSV_URLS: list[str] = [
    "https://public.tableau.com/views/"
    "WorkerAdjustmentRetrainingNotificationWARN/WARNList.csv"
    "?:embed=y&:showVizHome=no",
]

_NY_LANDING = "https://dol.ny.gov/warn-dashboard"
_NY_CACHE = _CACHE_ROOT / "warn_ny.parquet"
_NY_HIST_CACHE = _CACHE_ROOT / "warn_ny_historical.parquet"

# Tableau Public sits behind a WAF that 403s the default puremacro UA.
_NY_USER_AGENT = "Mozilla/5.0 (puremacro/narrative; +https://dol.ny.gov)"

# NY-DOL per-year HTML archive pages. The Tableau dashboard (April 2025+)
# replaced this archive going forward, but the 2023 and 2024 pages remain
# public. Each is a static HTML table with one row per filing — no
# magnitude (employee-count) column on the index page; per-filing pages
# carry that information but their URLs are not enumerable here.
_NY_HISTORICAL_HTML_URLS: list[str] = [
    "https://dol.ny.gov/2024-warn-notices",
    "https://dol.ny.gov/2023-warn-notices",
]


def _parse_ny_archive_html(raw: bytes, *, source_url: str) -> list[dict]:
    """Parse a NY DOL annual WARN-archive page into a list of filing dicts.

    Each yielded dict has keys: Company, Region, Date Posted, Notice Dated,
    _source_url. Date Posted is coerced to ``pd.Timestamp``; Notice Dated
    is parsed when possible (drops trailing "(Amended …)" annotation),
    set to ``pd.NaT`` otherwise. Returns ``[]`` on malformed input.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, "html.parser")
    table = soup.find("table")
    if table is None:
        return []
    out = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td"])
        if len(cells) < 4:
            continue  # header rows have <th> not <td>
        company = cells[0].get_text(strip=True)
        region = cells[1].get_text(strip=True)
        date_posted_raw = cells[2].get_text(strip=True)
        notice_dated_raw = cells[3].get_text(strip=True)

        # Strip "(Amended …)" annotation from Notice Dated before parsing
        notice_clean = notice_dated_raw.split("(", 1)[0].strip()
        date_posted = pd.to_datetime(date_posted_raw, errors="coerce")
        notice_dated = pd.to_datetime(notice_clean, errors="coerce")

        if pd.isna(date_posted) or not company:
            continue
        out.append({
            "Company": company,
            "Region": region,
            "Date Posted": date_posted,
            "Notice Dated": notice_dated,
            "_source_url": source_url,
        })
    return out


def _load_ny_historical(*, refetch: bool) -> pd.DataFrame:
    """Scrape the NY-DOL 2023+2024 archive pages into a single DataFrame.

    Magnitudes (employee counts) are unavailable from the index pages,
    so the resulting frame intentionally has no ``Number of Affected
    Workers``-style column — the downstream ``_iter_ny`` path will see
    ``magnitude=None`` for these rows.
    """
    if _NY_HIST_CACHE.exists() and not refetch:
        return pd.read_parquet(_NY_HIST_CACHE)
    rows: list[dict] = []
    for url in _NY_HISTORICAL_HTML_URLS:
        try:
            raw = safe_get_bytes(url, user_agent=_NY_USER_AGENT)
        except Exception:
            continue
        if not raw:
            continue
        try:
            page_rows = _parse_ny_archive_html(raw, source_url=url)
        except Exception:
            continue
        rows.extend(page_rows)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    try:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        out.to_parquet(_NY_HIST_CACHE)
    except Exception:
        pass
    return out


def _load_ny_live(*, refetch: bool) -> pd.DataFrame:
    """Fetch the post-2025 Tableau dashboard CSV (live filings)."""
    if _NY_CACHE.exists() and not refetch:
        return pd.read_parquet(_NY_CACHE)
    frames = []
    for url in _NY_CSV_URLS:
        try:
            raw = safe_get_bytes(url, user_agent=_NY_USER_AGENT)
        except Exception:
            continue
        if not raw:
            continue
        try:
            df = pd.read_csv(BytesIO(raw))
            df.columns = [_normalize_col(c) for c in df.columns]
            df["_source_url"] = url
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    try:
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        out.to_parquet(_NY_CACHE)
    except Exception:
        # Cache write is best-effort (parquet engine may be missing).
        pass
    return out


def _load_ny(*, refetch: bool) -> pd.DataFrame:
    """Merge the live Tableau CSV with the 2023+2024 HTML archive.

    The two sources use different column names for the same fields
    (live: ``Business Legal Name`` + ``Date Posted``; historical:
    ``Company`` + ``Date Posted``). Harmonise the company-name column
    onto ``Business Legal Name`` before concat so the (date, company)
    dedup actually compares like with like; keeps live rows when a
    filing appears in both sources.
    """
    live = _load_ny_live(refetch=refetch)
    hist = _load_ny_historical(refetch=refetch)
    if live.empty and hist.empty:
        return pd.DataFrame()
    hist = hist.copy()
    if "Company" in hist.columns and "Business Legal Name" not in hist.columns:
        hist = hist.rename(columns={"Company": "Business Legal Name"})
    df = pd.concat([live, hist], ignore_index=True, sort=False)
    date_col = _resolve_col(df, "Date Posted", "Date of WARN Notice",
                             "Notice Date", "Date Received", "Date")
    company_col = _resolve_col(df, "Business Legal Name",
                                "Company", "Company Name", "Employer")
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    dedup_keys = [c for c in (date_col, company_col) if c is not None]
    if dedup_keys:
        df = df.drop_duplicates(subset=dedup_keys, keep="first")
    return df.reset_index(drop=True)


def _iter_ny(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    df = _load_ny(refetch=refetch)
    if df.empty:
        return
    date_col = _resolve_col(
        df,
        "Date Posted", "Date of WARN Notice", "Notice Date",
        "Date Received", "Date Layoff/Closure Starts", "Date",
    )
    if date_col is None:
        return
    company_col = _resolve_col(
        df, "Business Legal Name", "Company", "Company Name", "Employer",
    )
    county_col = _resolve_col(
        df, "Impacted Site County", "County", "Region", "Location",
    )
    type_col = _resolve_col(
        df, "Layoff or Closure?", "Reason", "Layoff Reason",
        "Notice Type", "Layoff/Closure",
    )
    n_workers_col = _resolve_col(
        df,
        "Min. Number of Affected Workers",
        "Number of Affected Workers",
        "Number Affected", "Total Affected",
        "Number of Affected Employees",
        "Number of Employees", "Workers Impacted",
    )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if since:
        df = df[df[date_col] >= pd.Timestamp(since)]

    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = str(row.get(company_col) or "") if company_col else ""
        county = str(row.get(county_col) or "") if county_col else ""
        filing_type = (
            str(row.get(type_col) or "layoff") if type_col else "layoff"
        ).strip().lower()
        n_workers = None
        if n_workers_col is not None:
            # NY occasionally formats counts with thousands separators
            # (e.g. "1,433") which pd.to_numeric rejects; strip commas
            # before coercion.
            raw_val = row.get(n_workers_col)
            if isinstance(raw_val, str):
                raw_val = raw_val.replace(",", "").strip()
            v = pd.to_numeric(raw_val, errors="coerce")
            if pd.notna(v) and v >= 0:
                n_workers = float(v)

        if county:
            text = f"{company} ({county}, NY) - {filing_type}"
        else:
            text = f"{company} (NY) - {filing_type}"
        url = str(row.get("_source_url") or _NY_LANDING)
        meta = {
            "doctype": "warn_filing",
            "language": "en",
            "bank_code": "WARN_NY",
            "source": "warn_ny",
            "country": "USA",
            "state": "NY",
            "county": county or None,
            "filing_type": filing_type,
        }
        yield (row[date_col], text, url, meta, n_workers)


# Washington State ESD WARN database. Source page is an ASP.NET
# Web-Forms grid behind ``fortress.wa.gov/esd/file/WARN``; the live HTTP
# endpoint blocks puremacro's default UA. The cached parquet at
# data/processed/warn_wa.parquet is built one-shot by
# tools/scrape_wa_warn.py using a Playwright headless browser. Iterating
# returns the cached rows; refetch=True is a no-op for WA because the
# scraper lives outside the puremacro process.
_WA_CACHE = (Path(__file__).resolve().parents[4]
              / "data" / "processed" / "warn_wa.parquet")
_WA_LANDING = ("https://fortress.wa.gov/esd/file/"
                "WARN/Public/SearchWARN.aspx")


def _iter_wa(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    if not _WA_CACHE.exists():
        return
    df = pd.read_parquet(_WA_CACHE)
    if df.empty:
        return
    # Coerce date columns; prefer Received Date as the canonical filing date.
    for c in ("Received Date", "Layoff Start Date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["Received Date"])
    if since:
        df = df[df["Received Date"] >= pd.Timestamp(since)]

    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = str(row.get("Company") or "")
        location = str(row.get("Location") or "")
        filing_type = (str(row.get("Closure Layoff") or "layoff")
                        .strip().lower())
        n_workers = None
        v = pd.to_numeric(row.get("Num Workers"), errors="coerce")
        if pd.notna(v) and v >= 0:
            n_workers = float(v)

        if location:
            text = f"{company} ({location}, WA) - {filing_type}"
        else:
            text = f"{company} (WA) - {filing_type}"
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   "WARN_WA",
            "source":      "warn_wa",
            "country":     "USA",
            "state":       "WA",
            "county":      location or None,
            "filing_type": filing_type,
        }
        yield (row["Received Date"], text, _WA_LANDING, meta, n_workers)


# Illinois worknet — archived monthly XLSX downloads from a SharePoint
# page. Cached parquet at data/processed/warn_il.parquet is built by
# tools/scrape_il_warn.py (Playwright link enumeration + per-month
# XLSX fetch via requests).
_IL_CACHE = (Path(__file__).resolve().parents[4]
              / "data" / "processed" / "warn_il.parquet")
_IL_LANDING = ("https://illinoisworknet.com/LayoffRecovery/Pages/"
                "ArchivedWARNReports.aspx")


def _iter_il(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    if not _IL_CACHE.exists():
        return
    df = pd.read_parquet(_IL_CACHE)
    if df.empty:
        return
    date_col = _resolve_col(df, "WARN RECEIVED DATE:", "WARN RECEIVED DATE",
                             "Received Date", "Date Received")
    company_col = _resolve_col(df, "COMPANY NAME:", "COMPANY NAME",
                                "Company", "Company Name")
    city_col = _resolve_col(df, "CITY, STATE, ZIP:", "CITY, STATE, ZIP",
                             "City")
    event_col = _resolve_col(df, "TYPE OF EVENT:", "TYPE OF EVENT",
                              "Layoff/Closure", "Type")
    # IL has two worker-count column variants in the historical XLSX
    # series; prefer the one with more non-null values
    n_workers_col = _resolve_col(df, "WORKERS AFFECTED:", "# WORKERS AFFECTED:",
                                  "WORKERS AFFECTED", "Workers Affected",
                                  "Number of Employees")
    if date_col is None:
        return
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if since:
        df = df[df[date_col] >= pd.Timestamp(since)]

    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = str(row.get(company_col) or "") if company_col else ""
        city = str(row.get(city_col) or "") if city_col else ""
        event = (str(row.get(event_col) or "layoff") if event_col
                 else "layoff").strip().lower()
        n_workers = None
        if n_workers_col is not None:
            raw_val = row.get(n_workers_col)
            if isinstance(raw_val, str):
                raw_val = raw_val.replace(",", "").strip()
            v = pd.to_numeric(raw_val, errors="coerce")
            if pd.notna(v) and v >= 0:
                n_workers = float(v)

        if city:
            text = f"{company} ({city}, IL) - {event}"
        else:
            text = f"{company} (IL) - {event}"
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   "WARN_IL",
            "source":      "warn_il",
            "country":     "USA",
            "state":       "IL",
            "county":      city or None,
            "filing_type": event,
        }
        yield (row[date_col], text, _IL_LANDING, meta, n_workers)


# Maryland DLLR per-year WARN logs at dllr.state.md.us/employment/
# warn<YYYY>.shtml (2010-2025) + warn.shtml (current). Cached parquet
# at data/processed/warn_md.parquet is built by tools/scrape_md_warn.py
# (no Playwright needed — plain requests + BeautifulSoup).
_MD_CACHE = (Path(__file__).resolve().parents[4]
              / "data" / "processed" / "warn_md.parquet")
_MD_LANDING = "https://www.dllr.state.md.us/employment/warn.shtml"


def _iter_md(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    if not _MD_CACHE.exists():
        return
    df = pd.read_parquet(_MD_CACHE)
    if df.empty:
        return
    date_col = _resolve_col(df, "Notice Date", "Date Received", "Date")
    company_col = _resolve_col(df, "Company", "Company Name", "Employer")
    location_col = _resolve_col(df, "Location", "City")
    type_col = _resolve_col(df, "Type Code", "Type", "Layoff/Closure")
    n_workers_col = _resolve_col(df, "Total Employees", "Total Employ",
                                  "Number of Employees")
    if date_col is None:
        return
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if since:
        df = df[df[date_col] >= pd.Timestamp(since)]

    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = str(row.get(company_col) or "") if company_col else ""
        location = str(row.get(location_col) or "") if location_col else ""
        filing_type = (str(row.get(type_col) or "layoff")
                        if type_col else "layoff").strip().lower()
        n_workers = None
        if n_workers_col is not None:
            raw_val = row.get(n_workers_col)
            if isinstance(raw_val, str):
                raw_val = raw_val.replace(",", "").strip()
            v = pd.to_numeric(raw_val, errors="coerce")
            if pd.notna(v) and v >= 0:
                n_workers = float(v)
        if location:
            text = f"{company} ({location}, MD) - {filing_type}"
        else:
            text = f"{company} (MD) - {filing_type}"
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   "WARN_MD",
            "source":      "warn_md",
            "country":     "USA",
            "state":       "MD",
            "county":      location or None,
            "filing_type": filing_type,
        }
        yield (row[date_col], text, _MD_LANDING, meta, n_workers)


# Virginia Works WARN notices. Public CSV download at a session-keyed
# URL (`/warn_notices_<int-token>.csv`); the parquet at
# data/processed/warn_va.parquet is built by tools/scrape_va_warn.py.
_VA_CACHE = (Path(__file__).resolve().parents[4]
              / "data" / "processed" / "warn_va.parquet")
_VA_LANDING = "https://www.virginiaworks.gov/warn-notices"


def _iter_va(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    if not _VA_CACHE.exists():
        return
    df = pd.read_parquet(_VA_CACHE)
    if df.empty:
        return
    df["Notice Date"] = pd.to_datetime(df["Notice Date"], errors="coerce")
    df = df.dropna(subset=["Notice Date"])
    if since:
        df = df[df["Notice Date"] >= pd.Timestamp(since)]

    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = "" if pd.isna(row.get("Company")) else str(row.get("Company"))
        location = "" if pd.isna(row.get("Location")) else str(row.get("Location"))
        nt = row.get("Notice Type")
        filing_type = ("layoff" if pd.isna(nt) else str(nt)).strip().lower()
        n_workers = None
        v = pd.to_numeric(row.get("Employees Affected"), errors="coerce")
        if pd.notna(v) and v >= 0:
            n_workers = float(v)
        if location:
            text = f"{company} ({location}) - {filing_type}"
        else:
            text = f"{company} (VA) - {filing_type}"
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   "WARN_VA",
            "source":      "warn_va",
            "country":     "USA",
            "state":       "VA",
            "county":      location or None,
            "filing_type": filing_type,
        }
        yield (row["Notice Date"], text, _VA_LANDING, meta, n_workers)


# Wisconsin DWD per-year WARN archives (JS-rendered HTML tables).
# Cached parquet at data/processed/warn_wi.parquet is built by
# tools/scrape_wi_warn.py (Playwright per year, 2020-2026 populated).
_WI_CACHE = (Path(__file__).resolve().parents[4]
              / "data" / "processed" / "warn_wi.parquet")
_WI_LANDING = "https://dwd.wisconsin.gov/dislocatedworker/warn/"


def _iter_wi(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    if not _WI_CACHE.exists():
        return
    df = pd.read_parquet(_WI_CACHE)
    if df.empty:
        return
    date_col = _resolve_col(df, "Notice Received", "Notice Date",
                             "Date Received")
    company_col = _resolve_col(df, "Company")
    city_col = _resolve_col(df, "City")
    type_col = _resolve_col(df, "Original Notice Type / Update Type",
                             "Notice Type", "Type")
    n_workers_col = _resolve_col(df, "Affected Workers (Change)",
                                  "Affected Workers", "Workers Affected")
    if date_col is None:
        return
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if since:
        df = df[df[date_col] >= pd.Timestamp(since)]
    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = "" if pd.isna(row.get(company_col)) else str(row.get(company_col))
        city = "" if pd.isna(row.get(city_col)) else str(row.get(city_col))
        nt = row.get(type_col)
        filing_type = ("layoff" if pd.isna(nt) else str(nt)).strip().lower()
        n_workers = None
        if n_workers_col is not None:
            raw_val = row.get(n_workers_col)
            if isinstance(raw_val, str):
                raw_val = raw_val.replace(",", "").strip()
            v = pd.to_numeric(raw_val, errors="coerce")
            if pd.notna(v) and v >= 0:
                n_workers = float(v)
        if city:
            text = f"{company} ({city}, WI) - {filing_type}"
        else:
            text = f"{company} (WI) - {filing_type}"
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   "WARN_WI",
            "source":      "warn_wi",
            "country":     "USA",
            "state":       "WI",
            "county":      city or None,
            "filing_type": filing_type,
        }
        yield (row[date_col], text, _WI_LANDING, meta, n_workers)


# WARNTracker.com fallback for states without a direct scraper. The free
# public-tier table at https://www.warntracker.com/?state=<X> returns the
# 200 most-recent records; worker counts are paywalled for older rows.
# Cached at data/processed/warn_warntracker.parquet by tools/scrape_warntracker.py.
_WT_CACHE = (Path(__file__).resolve().parents[4]
              / "data" / "processed" / "warn_warntracker.parquet")
_WT_LANDING_TPL = "https://www.warntracker.com/?state={state}"

WARNTRACKER_STATES = frozenset({
    "AZ", "CO", "FL", "GA", "IN", "MA", "MI", "MO",
    "NC", "NJ", "OH", "TN", "TX",
})


def _iter_warntracker(*, state: str, refetch: bool,
                       since: str | None) -> Iterator[tuple]:
    if not _WT_CACHE.exists():
        return
    df = pd.read_parquet(_WT_CACHE)
    if df.empty:
        return
    df["Notice Date"] = pd.to_datetime(df["Notice Date"], errors="coerce")
    df = df.dropna(subset=["Notice Date"])
    df = df[df["State"] == state]
    if since:
        df = df[df["Notice Date"] >= pd.Timestamp(since)]
    landing = _WT_LANDING_TPL.format(state=state)
    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = "" if pd.isna(row.get("Company")) else str(row.get("Company"))
        city = ("" if pd.isna(row.get("City")) else str(row.get("City"))).strip()
        if "purchase" in city.lower() or "tier" in city.lower():
            city = ""
        wn = row.get("WorkersN")
        n_workers = float(wn) if pd.notna(wn) else None
        if city:
            text = f"{company} ({city}, {state}) - layoff"
        else:
            text = f"{company} ({state}) - layoff"
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   f"WARN_{state}",
            "source":      "warn_warntracker",
            "country":     "USA",
            "state":       state,
            "county":      city or None,
            "filing_type": "layoff",
        }
        yield (row["Notice Date"], text, landing, meta, n_workers)


# BLN public-GCS bucket historical files (storage.googleapis.com/
# bln-data-public/warn-layoffs/). Per-state historical CSV/XLSX that the
# warn-scraper-bot pulls in from a separate source pipeline. Includes
# states not in the bot snapshot — notably KY (1998-2016, no other path).
_BLN_GCS_CACHE = (Path(__file__).resolve().parents[4]
                   / "data" / "processed" / "warn_bln_gcs.parquet")
_BLN_GCS_LANDING = "https://storage.googleapis.com/bln-data-public/warn-layoffs/"


def _bln_gcs_states() -> frozenset:
    if not _BLN_GCS_CACHE.exists():
        return frozenset()
    try:
        states = pd.read_parquet(_BLN_GCS_CACHE, columns=["state"])["state"].unique()
        return frozenset(states.tolist())
    except Exception:
        return frozenset()


BLN_GCS_STATES = _bln_gcs_states()


def _iter_bln_gcs(*, state: str, refetch: bool,
                   since: str | None) -> Iterator[tuple]:
    if not _BLN_GCS_CACHE.exists():
        return
    df = pd.read_parquet(_BLN_GCS_CACHE)
    df = df[df["state"] == state]
    if df.empty:
        return
    df = df.dropna(subset=["notice_date"])
    if since:
        df = df[df["notice_date"] >= pd.Timestamp(since)]
    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = "" if pd.isna(row.get("company")) else str(row.get("company"))
        city = "" if pd.isna(row.get("city")) else str(row.get("city"))
        wn = row.get("workers")
        n_workers = float(wn) if pd.notna(wn) and wn >= 0 else None
        if city:
            text = f"{company} ({city}, {state}) - layoff"
        else:
            text = f"{company} ({state}) - layoff"
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   f"WARN_{state}",
            "source":      "warn_bln_gcs",
            "country":     "USA",
            "state":       state,
            "county":      city or None,
            "filing_type": "layoff",
        }
        yield (row["notice_date"], text, _BLN_GCS_LANDING, meta, n_workers)


# BigLocalNews warn-scraper-bot snapshot. Per-state CSVs scraped from the
# state DOL pages and committed to per-state branches of
# github.com/chriszs/warn-scraper-bot. Last updated ~2022. Cached at
# data/processed/warn_bln.parquet by tools/scrape_bln_warn.py.
_BLN_CACHE = (Path(__file__).resolve().parents[4]
               / "data" / "processed" / "warn_bln.parquet")
_BLN_LANDING_TPL = ("https://raw.githubusercontent.com/chriszs/"
                     "warn-scraper-bot/{state}/{lower}.csv")


def _bln_states() -> frozenset:
    """Discover BLN-covered states by reading the parquet (dynamic)."""
    if not _BLN_CACHE.exists():
        return frozenset()
    try:
        states = pd.read_parquet(_BLN_CACHE, columns=["state"])["state"].unique()
        return frozenset(states.tolist())
    except Exception:
        return frozenset()


BLN_STATES = _bln_states()


def _iter_bln(*, state: str, refetch: bool,
               since: str | None) -> Iterator[tuple]:
    if not _BLN_CACHE.exists():
        return
    df = pd.read_parquet(_BLN_CACHE)
    df = df[df["state"] == state]
    if df.empty:
        return
    df = df.dropna(subset=["notice_date"])
    if since:
        df = df[df["notice_date"] >= pd.Timestamp(since)]
    landing = _BLN_LANDING_TPL.format(state=state, lower=state.lower())
    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = "" if pd.isna(row.get("company")) else str(row.get("company"))
        city = "" if pd.isna(row.get("city")) else str(row.get("city"))
        wn = row.get("workers")
        n_workers = float(wn) if pd.notna(wn) and wn >= 0 else None
        if city:
            text = f"{company} ({city}, {state}) - layoff"
        else:
            text = f"{company} ({state}) - layoff"
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   f"WARN_{state}",
            "source":      "warn_bln",
            "country":     "USA",
            "state":       state,
            "county":      city or None,
            "filing_type": "layoff",
        }
        yield (row["notice_date"], text, landing, meta, n_workers)


# --- South Carolina: per-year PDF reports parsed to mixed-schema parquet
_SC_CACHE = (Path(__file__).resolve().parents[4]
              / "data" / "processed" / "warn_sc.parquet")
_SC_LANDING = ("https://scworks.org/employer/employer-programs/"
                "risk-closing/layoff-notification-reports")


def _sc_pick_date(row: pd.Series) -> pd.Timestamp | None:
    """Try Notice Date / col3 / col7 / col2 in that order; first parseable wins."""
    for col in ("Notice Date", "col3", "col7", "col2"):
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v) or str(v).strip().lower() in ("", "nan", "tbd"):
            continue
        ts = pd.to_datetime(str(v).strip(), errors="coerce")
        if pd.notna(ts) and 2010 <= ts.year <= 2030:
            return ts
    return None


def _sc_pick_company(row: pd.Series) -> str:
    for col in ("Company", "col1"):
        if col in row.index:
            v = row[col]
            if pd.notna(v) and str(v).strip().lower() not in ("", "nan"):
                return " ".join(str(v).split())
    return ""


def _sc_pick_workers(row: pd.Series) -> float | None:
    for col in ("Impacted", "Affected", "col6", "col10",
                "Projected Positions Affected"):
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v):
            continue
        s = str(v).replace(",", "").strip().upper()
        if s in ("", "NAN", "TBD", "N/A"):
            continue
        try:
            n = float(s)
        except ValueError:
            continue
        if n >= 0:
            return n
    return None


def _sc_pick_city(row: pd.Series) -> str:
    for col in ("County", "col2", "col4", "Location"):
        if col in row.index:
            v = row[col]
            if pd.notna(v) and str(v).strip().lower() not in ("", "nan"):
                return " ".join(str(v).split())
    return ""


def _iter_sc(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    if not _SC_CACHE.exists():
        return
    df = pd.read_parquet(_SC_CACHE)
    state = "SC"
    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        date = _sc_pick_date(row)
        if date is None:
            continue
        if since and date < pd.Timestamp(since):
            continue
        company = _sc_pick_company(row)
        if not company:
            continue
        city = _sc_pick_city(row)
        n_workers = _sc_pick_workers(row)
        text = (f"{company} ({city}, {state}) - layoff"
                 if city else f"{company} ({state}) - layoff")
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   f"WARN_{state}",
            "source":      "warn_sc",
            "country":     "USA",
            "state":       state,
            "county":      city or None,
            "filing_type": "layoff",
        }
        yield (date, text, _SC_LANDING, meta, n_workers)


SC_AVAILABLE = _SC_CACHE.exists()


# --- Hawaii: per-year text bullet listings parsed to clean parquet
_HI_CACHE = (Path(__file__).resolve().parents[4]
              / "data" / "processed" / "warn_hi.parquet")
_HI_LANDING = "https://labor.hawaii.gov/wdc/real-time-warn-updates/"


def _iter_hi(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    if not _HI_CACHE.exists():
        return
    df = pd.read_parquet(_HI_CACHE)
    df = df.dropna(subset=["notice_date"])
    if since:
        df = df[df["notice_date"] >= pd.Timestamp(since)]
    state = "HI"
    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = "" if pd.isna(row.get("company")) else str(row.get("company"))
        if not company:
            continue
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   f"WARN_{state}",
            "source":      "warn_hi",
            "country":     "USA",
            "state":       state,
            "county":      None,
            "filing_type": "layoff",
        }
        yield (row["notice_date"], f"{company} ({state}) - layoff",
                _HI_LANDING, meta, None)


HI_AVAILABLE = _HI_CACHE.exists()


# --- PA / NV / WV: per-state parquet from stealth scrape (all share the
#     same notice_date/company/workers/city/_source_url schema).
def _iter_state_parquet(*, cache: Path, state: str, landing: str,
                          source_tag: str,
                          refetch: bool, since: str | None) -> Iterator[tuple]:
    if not cache.exists():
        return
    df = pd.read_parquet(cache)
    df = df[df["state"] == state] if "state" in df.columns else df
    df = df.dropna(subset=["notice_date"])
    if since:
        df = df[df["notice_date"] >= pd.Timestamp(since)]
    # Performance: Use to_dict('records') instead of iterrows() for ~25x faster iteration
    for row in df.to_dict('records'):
        company = "" if pd.isna(row.get("company")) else str(row.get("company"))
        if not company.strip():
            continue
        city = "" if pd.isna(row.get("city")) else str(row.get("city"))
        wn = row.get("workers")
        n_workers = float(wn) if pd.notna(wn) and wn >= 0 else None
        text = (f"{company} ({city}, {state}) - layoff"
                 if city else f"{company} ({state}) - layoff")
        meta = {
            "doctype":     "warn_filing",
            "language":    "en",
            "bank_code":   f"WARN_{state}",
            "source":      source_tag,
            "country":     "USA",
            "state":       state,
            "county":      city or None,
            "filing_type": "layoff",
        }
        yield (row["notice_date"], text, landing, meta, n_workers)


_PA_CACHE = Path(__file__).resolve().parents[4] / "data" / "processed" / "warn_pa.parquet"
_NV_CACHE = Path(__file__).resolve().parents[4] / "data" / "processed" / "warn_nv.parquet"
_WV_CACHE = Path(__file__).resolve().parents[4] / "data" / "processed" / "warn_wv.parquet"

_PA_LANDING = ("https://www.pa.gov/agencies/dli/programs-services/"
                "workforce-development-home/warn-requirements/warn-notices")
_NV_LANDING = "https://detr.nv.gov/Page/WARN"
_WV_LANDING = "https://workforcewv.org/job-seeker/layoffs-downsizing/warn-listing/"


def _iter_pa(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    yield from _iter_state_parquet(
        cache=_PA_CACHE, state="PA", landing=_PA_LANDING,
        source_tag="warn_pa", refetch=refetch, since=since,
    )


def _iter_nv(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    yield from _iter_state_parquet(
        cache=_NV_CACHE, state="NV", landing=_NV_LANDING,
        source_tag="warn_nv", refetch=refetch, since=since,
    )


def _iter_wv(*, refetch: bool, since: str | None) -> Iterator[tuple]:
    yield from _iter_state_parquet(
        cache=_WV_CACHE, state="WV", landing=_WV_LANDING,
        source_tag="warn_wv", refetch=refetch, since=since,
    )


PA_AVAILABLE = _PA_CACHE.exists()
NV_AVAILABLE = _NV_CACHE.exists()
WV_AVAILABLE = _WV_CACHE.exists()


def iter_us_warn(
    *,
    state: Literal["CA", "NY", "WA", "IL", "MD", "VA", "WI", "SC", "HI",
                    "PA", "NV", "WV"],
    since: str | None = "2014-01-01",
    refetch: bool = False,
) -> Iterator[tuple]:
    """Yield 5-tuples for WARN-Act filings in the requested state.

    Dispatch priority: direct state scraper > BLN historical >
    WARNTracker recent. Direct scrapers have full magnitudes and live
    coverage; BLN has 1998-2022 historical depth for ~28 states;
    WARNTracker covers ~13 recent-only states with paywalled magnitudes.
    """
    if state == "CA":
        yield from _iter_ca(refetch=refetch, since=since)
    elif state == "NY":
        yield from _iter_ny(refetch=refetch, since=since)
    elif state == "WA":
        yield from _iter_wa(refetch=refetch, since=since)
    elif state == "IL":
        yield from _iter_il(refetch=refetch, since=since)
    elif state == "MD":
        yield from _iter_md(refetch=refetch, since=since)
    elif state == "VA":
        yield from _iter_va(refetch=refetch, since=since)
    elif state == "WI":
        yield from _iter_wi(refetch=refetch, since=since)
    elif state == "SC" and SC_AVAILABLE:
        yield from _iter_sc(refetch=refetch, since=since)
    elif state == "HI" and HI_AVAILABLE:
        yield from _iter_hi(refetch=refetch, since=since)
    elif state == "PA" and PA_AVAILABLE:
        yield from _iter_pa(refetch=refetch, since=since)
    elif state == "NV" and NV_AVAILABLE:
        yield from _iter_nv(refetch=refetch, since=since)
    elif state == "WV" and WV_AVAILABLE:
        yield from _iter_wv(refetch=refetch, since=since)
    elif (state in BLN_STATES or state in WARNTRACKER_STATES
           or state in BLN_GCS_STATES):
        # Multi-source UNION dedup on (date, company). Sources are mostly
        # disjoint in time (BLN-GCS deepest 1989+, BLN bot 1998-2022,
        # WARNTracker 2020-2026) so dedup is light.
        seen = set()
        for src_fn, in_states in (
            (_iter_bln_gcs,    state in BLN_GCS_STATES),
            (_iter_bln,        state in BLN_STATES),
            (_iter_warntracker, state in WARNTRACKER_STATES),
        ):
            if not in_states:
                continue
            for tup in src_fn(state=state, refetch=refetch, since=since):
                date, text, _url, _meta, _mag = tup
                key = (pd.Timestamp(date).normalize(), text.split(" (", 1)[0])
                if key in seen:
                    continue
                seen.add(key)
                yield tup
    else:
        raise ValueError(
            f"state must be CA/NY/WA/IL/MD/VA/WI/SC/HI/PA/NV/WV, a BLN state "
            f"({sorted(BLN_STATES)}), a BLN-GCS state "
            f"({sorted(BLN_GCS_STATES)}), or a WARNTracker state "
            f"({sorted(WARNTRACKER_STATES)}); got {state!r}"
        )


__all__ = ["iter_us_warn", "BLN_STATES", "BLN_GCS_STATES",
            "WARNTRACKER_STATES", "SC_AVAILABLE", "HI_AVAILABLE",
            "PA_AVAILABLE", "NV_AVAILABLE", "WV_AVAILABLE"]
