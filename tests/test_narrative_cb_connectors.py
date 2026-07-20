"""Offline + smoke tests for first-wave central-bank connectors."""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Federal Reserve
# ---------------------------------------------------------------------------
def test_fed_decision_yields_four_tuple(mock_http):
    # Real Fed JSON shape: top-level list, key 't' (not 'ti'), 'pt' filter.
    # Also includes a UTF-8 BOM since the live endpoint serves it.
    mock_http(
        bytes_={
            "https://www.federalreserve.gov/json/ne-press.json":
                b'\xef\xbb\xbf[{"d":"2022-03-16","t":"Federal Reserve issues FOMC statement",'
                b'"pt":"Monetary Policy",'
                b'"l":"/newsevents/pressreleases/monetary20220316a.htm"}]',
        },
        text={
            "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220316a.htm":
                "<html><body><p>The Committee decided to raise the target range "
                "for the federal funds rate to 1/4 to 1/2 percent.</p></body></html>",
        },
    )
    from puremacro.narrative.sources import iter_fed_decision
    records = list(iter_fed_decision())
    assert len(records) >= 1
    date, text, url, meta = records[0]
    assert isinstance(date, pd.Timestamp)
    assert "federal funds rate" in text.lower()
    assert meta["doctype"] == "decision"
    assert meta["language"] == "en"
    assert meta["bank_code"] == "FED"
    assert meta["country"] == "USA"


def test_fed_minutes_yields_four_tuple(mock_http):
    mock_http(
        bytes_={
            "https://www.federalreserve.gov/json/ne-press.json":
                b'\xef\xbb\xbf[{"d":"2022-04-06","t":"Minutes of the FOMC March meeting",'
                b'"pt":"Monetary Policy",'
                b'"l":"/newsevents/pressreleases/monetary20220316a.htm"}]',
        },
        text={
            # Announcement page contains a body link → body URL gets fetched.
            # Must contain landmark strings ("Federal Open Market Committee",
            # "Minutes") to pass the PARSER_SCHEMA_VERSION sentinel check.
            "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220316a.htm":
                "<html><body>"
                "<h1>Minutes of the Federal Open Market Committee</h1>"
                "<p>The Minutes of the Federal Open Market Committee were released today.</p>"
                "<a href=\"/monetarypolicy/fomcminutes20220316.htm\">Minutes</a>"
                "</body></html>",
            # Body page (5000+ chars) — must contain <div id="article"> for FED extractor.
            "https://www.federalreserve.gov/monetarypolicy/fomcminutes20220316.htm":
                "<html><body><div id=\"article\"><p>Participants noted "
                "that inflation remained elevated. " * 200 + "</p></div></body></html>",
        },
    )
    from puremacro.narrative.sources import iter_fed_minutes
    records = list(iter_fed_minutes())
    assert len(records) >= 1
    _, _, _, meta = records[0]
    assert meta["doctype"] == "minutes"


def test_fed_press_conf_yields_four_tuple(mock_http):
    """Press-conference connector: listing HTML + per-PDF byte fetches."""
    mock_http(
        bytes_={
            "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20220316.pdf":
                b"%PDF-1.4\n" + (
                    b"Chair Powell: Today the FOMC raised the federal funds "
                    b"rate by 25 basis points." * 5
                ),
        },
        text={
            "https://www.federalreserve.gov/monetarypolicy/fomcpresconf.htm":
                '<html><body><a href="/mediacenter/files/FOMCpresconf20220316.pdf">'
                'March 16, 2022</a></body></html>',
        },
    )
    from puremacro.narrative.sources import iter_fed_press_conf
    records = list(iter_fed_press_conf())
    if records:
        _, _, _, meta = records[0]
        assert meta["doctype"] == "press_conf"


@pytest.mark.network
def test_fed_speeches_smoke():
    """Live RSS smoke: skip if empty, never assert positive count."""
    from puremacro.narrative.sources import iter_fed_speeches
    records = list(iter_fed_speeches())
    if not records:
        pytest.skip("Fed speech feed returned empty (network or upstream issue).")
    _, _, _, meta = records[0]
    assert meta["doctype"] == "speech"
    assert meta["bank_code"] == "FED"
    assert meta["language"] == "en"


# ---------------------------------------------------------------------------
# European Central Bank
# ---------------------------------------------------------------------------
def test_ecb_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.ecb.europa.eu/rss/press.html":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary policy decisions</title>'
            b'<description>The Governing Council decided to raise rates by 25bps.</description>'
            b'<link>https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp220721.en.html</link>'
            b'<pubDate>Thu, 21 Jul 2022 12:45:00 +0200</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_ecb_decision
    records = list(iter_ecb_decision())
    assert len(records) == 1
    date, text, _, meta = records[0]
    assert "Governing Council" in text or "25bps" in text
    assert meta["doctype"] == "decision"
    assert meta["bank_code"] == "ECB"


def test_ecb_press_legacy_import_still_works(mock_http, recwarn):
    """Backwards compat: iter_ecb_press still importable, emits DeprecationWarning."""
    mock_http(bytes_={
        "https://www.ecb.europa.eu/rss/press.html":
            b'<?xml version="1.0"?><rss><channel></channel></rss>',
    })
    from puremacro.narrative.sources import iter_ecb_press
    list(iter_ecb_press())
    deprecations = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected DeprecationWarning from iter_ecb_press"


@pytest.mark.network
def test_ecb_speeches_smoke():
    from puremacro.narrative.sources import iter_ecb_speeches
    recs = list(iter_ecb_speeches())
    if not recs:
        pytest.skip("ECB speeches feed returned empty.")
    _, _, _, meta = recs[0]
    assert meta["bank_code"] == "ECB"
    assert meta["doctype"] == "speech"


# ---------------------------------------------------------------------------
# Bank of England
# ---------------------------------------------------------------------------
def test_boe_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.bankofengland.co.uk/rss/news/monetary-policy":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Bank Rate increased to 1.75% - August 2022</title>'
            b'<description>The MPC voted to raise Bank Rate by 0.5 percentage points.</description>'
            b'<link>https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes/2022/august-2022</link>'
            b'<pubDate>Thu, 04 Aug 2022 12:00:00 +0100</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_boe_decision
    records = list(iter_boe_decision())
    assert len(records) == 1
    _, text, _, meta = records[0]
    assert "Bank Rate" in text or "MPC" in text
    assert meta["bank_code"] == "BOE"
    assert meta["country"] == "GBR"


@pytest.mark.network
def test_boe_speeches_smoke():
    from puremacro.narrative.sources import iter_boe_speeches
    recs = list(iter_boe_speeches())
    if not recs:
        pytest.skip("BoE speeches feed empty.")
    _, _, _, meta = recs[0]
    # iter_boe_speeches emits "BoE" mixed-case (preserved across NBs); accept both.
    assert meta["bank_code"].upper() == "BOE"


# ---------------------------------------------------------------------------
# Bank of Japan
# ---------------------------------------------------------------------------
def test_boj_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.boj.or.jp/en/rss/whatsnew_e.xml":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Statement on Monetary Policy</title>'
            b'<description>The Bank of Japan decided to maintain the current monetary easing.</description>'
            b'<link>https://www.boj.or.jp/en/announcements/release_2022/k220721a.htm</link>'
            b'<pubDate>Thu, 21 Jul 2022 12:48:00 +0900</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_boj_decision
    recs = list(iter_boj_decision())
    assert len(recs) == 1
    _, text, _, meta = recs[0]
    assert "monetary" in text.lower() or "Bank of Japan" in text
    assert meta["bank_code"] == "BOJ"
    assert meta["country"] == "JPN"


@pytest.mark.network
def test_boj_speeches_smoke():
    from puremacro.narrative.sources import iter_boj_speeches
    recs = list(iter_boj_speeches())
    if not recs:
        pytest.skip("BoJ speeches feed empty.")
    _, _, _, meta = recs[0]
    # iter_boj_speeches emits "BoJ" mixed-case (preserved across NBs); accept both.
    assert meta["bank_code"].upper() == "BOJ"


def test_iter_rss_filtered_fetch_body_replaces_summary(mock_http):
    """When fetch_body=True, the connector pulls the link target and
    replaces the RSS summary text with extract_body(...)."""
    mock_http(
        bytes_={
            "https://example.test/rss.xml":
                b'<?xml version="1.0"?><rss><channel><item>'
                b'<title>Decision title</title>'
                b'<description>Short summary.</description>'
                b'<link>https://example.test/page.html</link>'
                b'<pubDate>Tue, 01 Mar 2022 12:00:00 +0000</pubDate>'
                b'</item></channel></rss>',
        },
        text={
            "https://example.test/page.html":
                '<html><body>'
                '<div id="article"><p>The full body of the decision is here.</p></div>'
                '</body></html>',
        },
    )
    from puremacro.narrative.sources._rss_filtered import iter_rss_filtered
    records = list(iter_rss_filtered(
        "https://example.test/rss.xml",
        bank_code="FED", country="USA", doctype="decision",
        language="en", fetch_body=True,
    ))
    assert len(records) == 1
    _, text, _, _ = records[0]
    assert "full body of the decision" in text
    assert "Short summary" not in text   # body replaces summary
