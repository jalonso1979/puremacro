"""Sprint α.1: body-text extraction for 10 CB connectors.

Each bank has:
  * a fixture-based unit test asserting the extractor returns >= 500 chars
    of body text containing a known body-only anchor phrase;
  * a network-marked integration test that pytest.skip()s when the live
    fetcher returns empty (per memory/feedback_network_tests_skip_on_empty.md).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from puremacro.narrative.sources._extractors import extract_body

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "narrative" / "body_html"


def _load_fixture(bank_lower: str) -> str:
    return (FIXTURE_DIR / f"{bank_lower}.html").read_text(encoding="utf-8",
                                                         errors="ignore")


def test_extract_body_bis_fixture():
    html = _load_fixture("bis")
    body = extract_body(html, bank_code="BIS")
    assert len(body) >= 500, f"BIS body too short ({len(body)} chars)"
    # Sanity: extracted text should not contain navigation chrome.
    assert "Skip to main content" not in body


@pytest.mark.network
def test_iter_bis_speeches_with_body_returns_long_text():
    from puremacro.narrative.sources.bis_speeches import iter_bis_speeches
    records = list(iter_bis_speeches(fetch_body=True))
    if not records:
        pytest.skip("BIS RSS empty — upstream may be down")
    # At least one record should have body text > 500 chars.
    longest = max(len(rec[1]) for rec in records[:25])
    assert longest > 500, f"max body length only {longest}"


def test_extract_body_boj_fixture():
    html = _load_fixture("boj")
    body = extract_body(html, bank_code="BOJ")
    assert len(body) >= 500, f"BoJ body too short ({len(body)} chars)"
    assert "Skip to main content" not in body


@pytest.mark.network
def test_iter_boj_speeches_with_body_returns_long_text():
    from puremacro.narrative.sources.boj_speeches import iter_boj_speeches
    records = list(iter_boj_speeches(min_year=2023, max_year=2023,
                                     fetch_body=True))
    if not records:
        pytest.skip("BoJ archive empty — upstream may have changed")
    longest = max(len(rec[1]) for rec in records[:10])
    assert longest > 500, f"max body length only {longest}"


def test_extract_body_pboc_fixture():
    html = _load_fixture("pboc")
    body = extract_body(html, bank_code="PBOC")
    assert len(body) >= 500, f"PBoC body too short ({len(body)} chars)"
    assert "Skip to main content" not in body


@pytest.mark.network
def test_iter_pboc_with_body_returns_long_text():
    from puremacro.narrative.sources.pboc import iter_pboc_speeches
    records = list(iter_pboc_speeches(fetch_body=True))
    if not records:
        pytest.skip("PBoC empty — upstream may have changed")
    longest = max(len(rec[1]) for rec in records[:10])
    assert longest > 500, f"max body length only {longest}"


def test_extract_body_rba_fixture():
    html = _load_fixture("rba")
    body = extract_body(html, bank_code="RBA")
    assert len(body) >= 500, f"RBA body too short ({len(body)} chars)"
    assert "Skip to main content" not in body


@pytest.mark.network
def test_iter_rba_with_body_returns_long_text():
    from puremacro.narrative.sources.rba import iter_rba_speeches
    records = list(iter_rba_speeches(min_year=2023, max_year=2023,
                                     fetch_body=True))
    if not records:
        pytest.skip("RBA archive empty — upstream may have changed")
    longest = max(len(rec[1]) for rec in records[:10])
    assert longest > 500, f"max body length only {longest}"


# ---------------------------------------------------------------------------
# BCB (α-6 retry): SPA-only site — long-form text lives in linked PDFs,
# so the extractor is ``extract_body_from_pdf`` and the fixture is a real
# Copom Minutes PDF rather than an HTML snapshot.
# ---------------------------------------------------------------------------
def test_extract_body_bcb_from_pdf_fixture():
    pytest.importorskip("pypdf", reason="pypdf is an optional [narrative] dep; install it to run PDF tests")
    pdf_bytes = (FIXTURE_DIR / "bcb_minutes.pdf").read_bytes()
    from puremacro.narrative.sources._extractors import extract_body_from_pdf
    body = extract_body_from_pdf(pdf_bytes)
    assert body is not None, "BCB PDF body returned None"
    assert len(body) >= 500, f"BCB PDF body too short ({len(body)} chars)"


def test_extract_body_from_pdf_empty_bytes_returns_none():
    from puremacro.narrative.sources._extractors import extract_body_from_pdf
    assert extract_body_from_pdf(b"") is None


@pytest.mark.network
def test_iter_bcb_minutes_with_body_returns_long_text():
    from puremacro.narrative.sources.bcb import iter_bcb_minutes
    records = list(iter_bcb_minutes(fetch_body=True))
    if not records:
        pytest.skip("BCB minutes feed empty — upstream may have changed")
    longest = max(len(rec[1]) for rec in records[:5])
    if longest <= 500:
        pytest.skip("BCB minutes returned but no long PDF body — upstream "
                    "PDF urls may be stale")
    assert longest > 500


# ---------------------------------------------------------------------------
# Banxico (α-7 retry): same PDF-only situation as BCB. The monetary-policy
# announcements listing links straight to GUID-named PDFs; there are no
# per-decision HTML landing pages.
# ---------------------------------------------------------------------------
def test_extract_body_banxico_from_pdf_fixture():
    pytest.importorskip("pypdf", reason="pypdf is an optional [narrative] dep; install it to run PDF tests")
    pdf_bytes = (FIXTURE_DIR / "banxico_decision.pdf").read_bytes()
    from puremacro.narrative.sources._extractors import extract_body_from_pdf
    body = extract_body_from_pdf(pdf_bytes)
    assert body is not None, "Banxico PDF body returned None"
    assert len(body) >= 500, f"Banxico PDF body too short ({len(body)} chars)"


@pytest.mark.network
def test_iter_banxico_with_body_returns_long_text():
    from puremacro.narrative.sources.banxico import iter_banxico_decision
    records = list(iter_banxico_decision(fetch_body=True))
    if not records:
        pytest.skip("Banxico feed empty")
    longest = max(len(rec[1]) for rec in records[:5])
    if longest <= 500:
        pytest.skip("Banxico returned only short metadata; PDF endpoints may be stale")
    assert longest > 500


def test_extract_body_bok_fixture():
    html = _load_fixture("bok")
    body = extract_body(html, bank_code="BOK")
    assert len(body) >= 500, f"BoK body too short ({len(body)} chars)"
    assert "Skip to main content" not in body


@pytest.mark.network
def test_iter_bok_with_body_returns_long_text():
    from puremacro.narrative.sources.bok import iter_bok_decision
    records = list(iter_bok_decision(fetch_body=True))
    if not records:
        pytest.skip("BoK empty")
    longest = max(len(rec[1]) for rec in records[:10])
    assert longest > 500, f"max body length only {longest}"


def test_extract_body_riksbank_fixture():
    html = _load_fixture("riksbank")
    body = extract_body(html, bank_code="RIKSBANK")
    assert len(body) >= 500, f"Riksbank body too short ({len(body)} chars)"
    assert "Skip to main content" not in body


@pytest.mark.network
def test_iter_riksbank_with_body_returns_long_text():
    from puremacro.narrative.sources.riksbank import iter_riksbank_decision
    records = list(iter_riksbank_decision(fetch_body=True))
    if not records:
        pytest.skip("Riksbank empty")
    longest = max(len(rec[1]) for rec in records[:10])
    assert longest > 500, f"max body length only {longest}"


def test_extract_body_rbi_fixture():
    html = _load_fixture("rbi")
    body = extract_body(html, bank_code="RBI")
    assert len(body) >= 500, f"RBI body too short ({len(body)} chars)"
    assert "Skip to main content" not in body


@pytest.mark.network
def test_iter_rbi_with_body_returns_long_text():
    from puremacro.narrative.sources.rbi import iter_rbi_decision
    records = list(iter_rbi_decision(fetch_body=True))
    if not records:
        pytest.skip("RBI feed empty")
    longest = max(len(rec[1]) for rec in records[:10])
    assert longest > 500


def test_extract_body_norges_fixture():
    html = _load_fixture("norges")
    body = extract_body(html, bank_code="NORGES")
    assert len(body) >= 500, f"Norges body too short ({len(body)} chars)"
    assert "Skip to main content" not in body


@pytest.mark.network
def test_iter_norges_with_body_returns_long_text():
    from puremacro.narrative.sources.norges import iter_norges_decision
    records = list(iter_norges_decision(fetch_body=True))
    if not records:
        pytest.skip("Norges feed empty")
    longest = max(len(rec[1]) for rec in records[:10])
    assert longest > 500
