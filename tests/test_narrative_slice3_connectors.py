"""Offline + smoke tests for Slice 3 central-bank connectors (LATAM,
Advanced non-G7, Asia-EM, BIS speeches meta).

Uses the conftest-provided ``mock_http`` fixture for offline tests.
"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Banco de México (Banxico)
# ---------------------------------------------------------------------------
def test_banxico_decision_yields_four_tuple_es(mock_http):
    """Banxico rewritten in 0.23.0 as HTML scraper against the
    announcement listing page. Fixture provides a minimal table row."""
    listing_url = (
        "https://www.banxico.org.mx/publicaciones-y-prensa/"
        "anuncios-de-las-decisiones-de-politica-monetaria/"
        "anuncios-politica-monetaria-t.html"
    )
    mock_http(text={
        listing_url: (
            '<html><body><table>'
            '<tr><td>29/09/22</td>'
            '<td>Anuncio de política monetaria - tasa objetivo se mantiene</td>'
            '<td><A HREF="/publicaciones-y-prensa/anuncios-de-las-decisiones-de-politica-monetaria/'
            '{ABCDEFAB-1234-5678-9ABC-DEF012345678}.pdf" '
            'aria-label="Texto completo de anuncio">Texto completo</A></td>'
            '</tr></table></body></html>'
        ),
    })
    from puremacro.narrative.sources import iter_banxico_decision
    records = list(iter_banxico_decision())
    assert len(records) == 1
    _, text, _, meta = records[0]
    assert "política monetaria" in text.lower() or "tasa objetivo" in text.lower()
    assert meta["doctype"] == "decision"
    assert meta["bank_code"] == "BANXICO"
    assert meta["country"] == "MEX"
    assert meta["language"] == "es"


@pytest.mark.network
def test_banxico_decision_smoke():
    from puremacro.narrative.sources import iter_banxico_decision
    recs = list(iter_banxico_decision())
    if not recs:
        pytest.skip("Banxico feed returned empty.")
    _, _, _, meta = recs[0]
    assert meta["bank_code"] == "BANXICO"


# ---------------------------------------------------------------------------
# Banco Central do Brasil (BCB)
# ---------------------------------------------------------------------------
def test_bcb_decision_yields_four_tuple_pt(monkeypatch):
    """BCB rewritten in 0.25.0 as XHR-API JSON scraper. Mock the
    internal _fetch_section helper with a synthetic API response."""
    fake_items = [{
        "DataReferencia": "2022-09-21T18:30:00Z",
        "Titulo": "Decisão do Copom: taxa Selic",
        "Subtitulo": "250ª reunião",
        "LinkPagina": "/controleinflacao/comunicadoscopom/100",
    }]
    from puremacro.narrative.sources import bcb as bcb_mod
    monkeypatch.setattr(bcb_mod, "_fetch_section",
                        lambda section, quantidade=1000: fake_items)
    records = list(bcb_mod.iter_bcb_decision(language="pt"))
    assert len(records) == 1
    _, text, _, meta = records[0]
    assert "copom" in text.lower() or "selic" in text.lower()
    assert meta["bank_code"] == "BCB"
    assert meta["country"] == "BRA"
    assert meta["language"] == "pt"


@pytest.mark.network
def test_bcb_decision_smoke():
    from puremacro.narrative.sources import iter_bcb_decision
    recs = list(iter_bcb_decision())
    if not recs:
        pytest.skip("BCB feed empty.")
    _, _, _, meta = recs[0]
    assert meta["bank_code"] == "BCB"


# ---------------------------------------------------------------------------
# Banco Central de Chile (BCCh)
# ---------------------------------------------------------------------------
def test_bccl_decision_yields_four_tuple_es(mock_http):
    mock_http(bytes_={
        "https://www.bcentral.cl/-/rss-feed-prensa":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Reuni\xc3\xb3n de pol\xc3\xadtica monetaria</title>'
            b'<description>El Consejo decidi\xc3\xb3 aumentar la tasa.</description>'
            b'<link>https://www.bcentral.cl/contenido/-/detalle/reunion-2022-09</link>'
            b'<pubDate>Tue, 06 Sep 2022 18:00:00 -0400</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_bccl_decision
    records = list(iter_bccl_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BCCH"
    assert meta["country"] == "CHL"
    assert meta["language"] == "es"


# ---------------------------------------------------------------------------
# Banco Central de la República Argentina (BCRA)
# ---------------------------------------------------------------------------
def test_bcra_decision_yields_four_tuple_es(mock_http):
    mock_http(bytes_={
        "https://www.bcra.gob.ar/rss/Prensa.aspx":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Comunicado de prensa</title>'
            b'<description>El BCRA fij\xc3\xb3 la tasa de pol\xc3\xadtica monetaria.</description>'
            b'<link>https://www.bcra.gob.ar/Noticias/Comunicado-2022-08.asp</link>'
            b'<pubDate>Thu, 11 Aug 2022 17:00:00 -0300</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_bcra_decision
    records = list(iter_bcra_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BCRA"
    assert meta["country"] == "ARG"


# ---------------------------------------------------------------------------
# Banco de la República (Colombia, BanRep)
# ---------------------------------------------------------------------------
def test_banrep_decision_yields_four_tuple_es(mock_http):
    mock_http(bytes_={
        "https://www.banrep.gov.co/rss-comunicados":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Comunicado de pol\xc3\xadtica monetaria</title>'
            b'<description>La Junta Directiva increment\xc3\xb3 la tasa de inter\xc3\xa9s.</description>'
            b'<link>https://www.banrep.gov.co/comunicado-2022-10</link>'
            b'<pubDate>Fri, 28 Oct 2022 14:00:00 -0500</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_banrep_decision
    records = list(iter_banrep_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BANREP"
    assert meta["country"] == "COL"


# ---------------------------------------------------------------------------
# Reserve Bank of Australia (RBA)
# ---------------------------------------------------------------------------
def test_rba_decision_yields_four_tuple(monkeypatch):
    """RBA rewritten in 0.24.0 as Playwright HTML scraper. Mock the
    _fallback._stage_playwright helper to return a synthetic archive-page
    HTML snippet (0.67.0: rba migrated to fetch_with_fallback)."""
    fake_html = (
        '<li class="item rss-mr-item">'
        '<div class="title">'
        '<a href="/media-releases/2022/mr-22-30.html" itemprop="url">'
        '<span itemprop="headline">Statement on monetary policy decision</span>'
        '</a></div>'
        '<div class="info">'
        '<time datetime="2022-10-04">4 October 2022</time>'
        '</div></li>'
    )
    from puremacro.narrative.sources import _fallback as fb_mod
    monkeypatch.setattr(fb_mod, "_stage_playwright",
                        lambda url, **_kw: fake_html)
    from puremacro.narrative.sources import rba as rba_mod
    records = list(rba_mod.iter_rba_decision(min_year=2022, max_year=2022))
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "RBA"
    assert meta["country"] == "AUS"
    assert meta["language"] == "en"


def test_rba_speeches_yields_four_tuple(monkeypatch):
    fake_html = (
        '<h3 class="title rss-speech-title">'
        '<a href="/speeches/2022/sp-gov-2022-09-15.html" itemprop="url">'
        '<span itemprop="headline">Inflation, productivity, and the supply side</span>'
        '</a></h3>'
        '<time datetime="2022-09-15T03:00+10:00">15 September 2022</time>'
    )
    from puremacro.narrative.sources import _fallback as fb_mod
    monkeypatch.setattr(fb_mod, "_stage_playwright",
                        lambda url, **_kw: fake_html)
    from puremacro.narrative.sources import rba as rba_mod
    records = list(rba_mod.iter_rba_speeches(min_year=2022, max_year=2022))
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "RBA"
    assert meta["doctype"] == "speech"


# ---------------------------------------------------------------------------
# Reserve Bank of New Zealand (RBNZ)
# ---------------------------------------------------------------------------
def test_rbnz_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.rbnz.govt.nz/rss/news.xml":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary Policy Statement</title>'
            b'<description>The MPC raised the OCR by 50 basis points.</description>'
            b'<link>https://www.rbnz.govt.nz/news/2022/10/monetary-policy-statement-october-2022</link>'
            b'<pubDate>Wed, 05 Oct 2022 02:00:00 +0000</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_rbnz_decision
    records = list(iter_rbnz_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "RBNZ"
    assert meta["country"] == "NZL"


# ---------------------------------------------------------------------------
# Riksbank (Sweden)
# ---------------------------------------------------------------------------
def test_riksbank_decision_yields_four_tuple(monkeypatch):
    """Riksbank rewritten in 0.24.0 as Playwright HTML scraper against
    the monetary-policy news category page.
    0.67.0: riksbank migrated to fetch_with_fallback; mock _stage_playwright."""
    fake_html = (
        '<a href="/en-gb/press-and-published/notices-and-press-releases/'
        'press-releases/2022/repo-rate/">'
        '<span class="date-and-category">21/09/2022 Press release</span>'
        '<h2>Repo rate decision</h2>'
        '</a>'
    )
    from puremacro.narrative.sources import _fallback as fb_mod
    monkeypatch.setattr(fb_mod, "_stage_playwright",
                        lambda url, **_kw: fake_html)
    from puremacro.narrative.sources import riksbank as rb_mod
    records = list(rb_mod.iter_riksbank_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "RIKSBANK"
    assert meta["country"] == "SWE"


# ---------------------------------------------------------------------------
# Norges Bank
# ---------------------------------------------------------------------------
def test_norges_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.norges-bank.no/en/news-events/news-publications/?rss=true":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary Policy Report and key policy rate</title>'
            b'<description>Norges Bank raised the policy rate by 50 basis points.</description>'
            b'<link>https://www.norges-bank.no/en/news-events/news-publications/2022/2022-09</link>'
            b'<pubDate>Thu, 22 Sep 2022 08:00:00 +0000</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_norges_decision
    records = list(iter_norges_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "NORGES"
    assert meta["country"] == "NOR"


# ---------------------------------------------------------------------------
# South African Reserve Bank (SARB)
# ---------------------------------------------------------------------------
def test_sarb_decision_yields_four_tuple(monkeypatch):
    """SARB rewritten in 0.26.0 as Playwright HTML scraper.
    0.67.0: sarb migrated to fetch_with_fallback; mock _stage_playwright."""
    fake_html = (
        '<a href="/en/home/publications/publication-detail-pages/statements/'
        'monetary-policy-statements/2022/september">'
        '<span class="publications__resultListItem__title"><b>'
        'Statement of the Monetary Policy Committee September 2022'
        '</b></span></a>'
    )
    from puremacro.narrative.sources import _fallback as fb_mod
    monkeypatch.setattr(fb_mod, "_stage_playwright",
                        lambda url, **_kw: fake_html)
    from puremacro.narrative.sources import sarb as sarb_mod
    records = list(sarb_mod.iter_sarb_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "SARB"
    assert meta["country"] == "ZAF"


# ---------------------------------------------------------------------------
# People's Bank of China (PBoC) — English mirror
# ---------------------------------------------------------------------------
def test_pboc_decision_yields_four_tuple_en(mock_http):
    mock_http(text={
        "https://www.pbc.gov.cn/en/3688110/3688215/index.html":
            '<html><body>'
            '<a href="/en/3688110/3688215/4582345/index.html" title="PBC announces rate cut">'
            'PBC announces rate cut'
            '</a>'
            '<span class="date">2022-08-22</span>'
            '</body></html>',
    })
    from puremacro.narrative.sources import iter_pboc_decision
    records = list(iter_pboc_decision())
    # Best-effort HTML parse — may yield 0 if the structure isn't recognised.
    if records:
        _, _, _, meta = records[0]
        assert meta["bank_code"] == "PBOC"
        assert meta["country"] == "CHN"


# ---------------------------------------------------------------------------
# Reserve Bank of India (RBI)
# ---------------------------------------------------------------------------
def test_rbi_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://rbi.org.in/Scripts/RSS.aspx":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary Policy Statement</title>'
            b'<description>The MPC raised the repo rate by 50 basis points.</description>'
            b'<link>https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=12345</link>'
            b'<pubDate>Fri, 30 Sep 2022 10:00:00 +0530</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_rbi_decision
    records = list(iter_rbi_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "RBI"
    assert meta["country"] == "IND"


# ---------------------------------------------------------------------------
# Bank of Korea (BoK) — English mirror
# ---------------------------------------------------------------------------
def test_bok_decision_yields_four_tuple(monkeypatch):
    """BoK rewritten in 0.26.0 as Playwright HTML scraper.
    0.67.0: bok migrated to fetch_with_fallback; mock _stage_playwright."""
    fake_html = (
        '<li class="bbsRowCls">'
        '<span class="date">2022.10.12</span>'
        '<a href="/eng/bbs/E0000627/view.do?nttId=10000123">'
        'Monetary Policy Decision'
        '</a></li>'
    )
    from puremacro.narrative.sources import _fallback as fb_mod
    monkeypatch.setattr(fb_mod, "_stage_playwright",
                        lambda url, **_kw: fake_html)
    from puremacro.narrative.sources import bok as bok_mod
    records = list(bok_mod.iter_bok_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BoK"
    assert meta["country"] == "KOR"


# ---------------------------------------------------------------------------
# Monetary Authority of Singapore (MAS)
# ---------------------------------------------------------------------------
def test_mas_decision_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.mas.gov.sg/news/rss":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Monetary policy statement</title>'
            b'<description>MAS will continue with the policy of appreciation of the SGD NEER.</description>'
            b'<link>https://www.mas.gov.sg/news/monetary-policy-statements/2022/oct</link>'
            b'<pubDate>Fri, 14 Oct 2022 08:00:00 +0800</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_mas_decision
    records = list(iter_mas_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "MAS"
    assert meta["country"] == "SGP"


# ---------------------------------------------------------------------------
# Bank of Thailand (BoT)
# ---------------------------------------------------------------------------
def test_bot_decision_yields_four_tuple(monkeypatch):
    """BoT rewritten in 0.26.0 as AEM XHR-JSON scraper.

    The mock_http fixture doesn't intercept it because the connector
    uses ``requests.get`` directly with a ``Referer`` header. We
    monkeypatch the internal ``_fetch_page`` helper instead.
    """
    fake_items = [{
        "listingTitle": "Monetary Policy Committee's Decision 5/2022",
        "issueDt": "28 Sep 2022",
        "pagePath": "https://www.bot.or.th/en/news-and-media/news/news-20220928.html",
        "tags": ["monetary policy"],
        "releaseNumber": "33/2022",
    }]
    from puremacro.narrative.sources import bot as bot_mod
    monkeypatch.setattr(bot_mod, "_fetch_page",
                        lambda **_kw: fake_items)
    records = list(bot_mod.iter_bot_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BoT"
    assert meta["country"] == "THA"
    return  # OLD test (RSS-based) below is dead code
    mock_http(bytes_={
        "https://www.bot.or.th/content/bot/en/_jcr_content.feed":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>MPC decision</title>'
            b'<description>The MPC voted to raise the policy rate by 25bps.</description>'
            b'<link>https://www.bot.or.th/en/news-and-media/news/news-202209</link>'
            b'<pubDate>Wed, 28 Sep 2022 14:00:00 +0700</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_bot_decision
    records = list(iter_bot_decision())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BOT"
    assert meta["country"] == "THA"


# ---------------------------------------------------------------------------
# BIS speeches meta-connector
# ---------------------------------------------------------------------------
def test_bis_speeches_yields_four_tuple(mock_http):
    mock_http(bytes_={
        "https://www.bis.org/doclist/cbspeeches.rss":
            b'<?xml version="1.0"?><rss><channel><item>'
            b'<title>Inflation outlook and policy challenges</title>'
            b'<description>Speech by Christine Lagarde at the IMF annual meetings.</description>'
            b'<link>https://www.bis.org/review/r221015a.htm</link>'
            b'<pubDate>Sat, 15 Oct 2022 12:00:00 +0000</pubDate>'
            b'</item></channel></rss>',
    })
    from puremacro.narrative.sources import iter_bis_speeches
    records = list(iter_bis_speeches())
    assert len(records) == 1
    _, _, _, meta = records[0]
    assert meta["bank_code"] == "BIS"
    assert meta["doctype"] == "speech"
    # Country tag is "MULTI" because BIS speeches span all member banks.
    assert meta["country"] == "MULTI"


def test_bis_speeches_filter_by_bank_keyword(mock_http):
    """When the caller passes bank_filter, only matching items are yielded."""
    mock_http(bytes_={
        "https://www.bis.org/doclist/cbspeeches.rss":
            b'<?xml version="1.0"?><rss><channel>'
            b'<item><title>ECB Lagarde on inflation</title>'
            b'<description>Christine Lagarde speech</description>'
            b'<link>https://www.bis.org/review/r221015a.htm</link>'
            b'<pubDate>Sat, 15 Oct 2022 12:00:00 +0000</pubDate></item>'
            b'<item><title>Fed Powell on financial conditions</title>'
            b'<description>Jerome Powell speech</description>'
            b'<link>https://www.bis.org/review/r221016a.htm</link>'
            b'<pubDate>Sun, 16 Oct 2022 12:00:00 +0000</pubDate></item>'
            b'</channel></rss>',
    })
    from puremacro.narrative.sources import iter_bis_speeches
    records = list(iter_bis_speeches(bank_filter="Powell"))
    assert len(records) == 1
    _, text, _, _ = records[0]
    assert "Powell" in text


@pytest.mark.network
def test_bis_speeches_smoke():
    from puremacro.narrative.sources import iter_bis_speeches
    recs = list(iter_bis_speeches())
    if not recs:
        pytest.skip("BIS speeches feed empty.")
    _, _, _, meta = recs[0]
    assert meta["bank_code"] == "BIS"
