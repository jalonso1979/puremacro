"""Source connectors. Each emits an iterable of ``(date, text, source_url)``
records ready for any scoring backend. Connectors do *not* score —
separation of concerns.

Country-scope expansion to the eventual 33-country wedges panel is
purely additive: drop in a sibling file (``uk_obr.py``, ``de_bmf.py``,
…) implementing the same iterator contract, and the scoring,
aggregation, and validation layers consume it without changes.

**Lazy loading (PEP 562).** The connectors import scraper/network deps
(``bs4``, ``requests``, ``pdfplumber``) at module level, which are NOT
available under Pyodide. Importing this package therefore does NOT eager-load
the connectors; each is imported on first attribute access. This keeps
``import puremacro.narrative`` (and the text-uncertainty indices) browser-clean
while the actual scrapers pull their heavy deps only when called. Importing a
connector *submodule* directly (``from puremacro.narrative.sources import
beige_book``) still loads it eagerly — by design, that is opting in.
"""
from __future__ import annotations

import importlib

# Public name -> submodule that defines it. Each connector module is imported
# lazily on first access to one of its names (see ``__getattr__``).
_MODULE_OF = {
    "iter_beige_book": "beige_book",
    "iter_bluesky_posts": "bluesky",
    "iter_local_csv": "local_csv",
    "iter_federal_register": "us_federal_register",
    "iter_treasury_press": "us_treasury",
    "iter_dod_contracts": "us_dod_contracts",
    "iter_erp": "us_erp",
    "iter_sotu": "us_sotu",
    "iter_cbo": "us_cbo",
    "iter_gdelt_v2": "news_api",
    "iter_google_news": "google_news",
    "LOCALE_BY_COUNTRY": "google_news",
    "iter_hmt_press": "uk_hmt",
    "iter_obr_publications": "uk_obr",
    "iter_bmf_press": "de_bmf",
    "iter_tresor_press": "fr_tresor",
    "iter_mef_press": "it_mef",
    "iter_mof_press": "jp_mof",
    "iter_dof_press": "ca_dof",
    "iter_ecfin_press": "eu_ecfin",
    "iter_eurlex": "eu_eurlex",
    "iter_ep_debates": "eu_parliament",
    "iter_ecb_decision": "ecb_decision",
    "iter_ecb_minutes": "ecb_minutes",
    "iter_ecb_press_conf": "ecb_press_conf",
    "iter_ecb_speeches": "ecb_speeches",
    "iter_ecb_press": "ecb_press",
    "iter_boe_decision": "boe_decision",
    "iter_boe_minutes": "boe_minutes",
    "iter_boe_speeches": "boe_speeches",
    "iter_boj_decision": "boj_decision",
    "iter_boj_speeches": "boj_speeches",
    "iter_imf_articleiv": "imf_articleiv",
    "iter_imf_listing": "imf_articleiv",
    "clear_listing_cache": "imf_articleiv",
    "iter_imf_news": "imf_news",
    "iter_oecd_surveys": "oecd_surveys",
    "iter_oecd_listing": "oecd_surveys",
    "iter_fed_decision": "fed_decision",
    "iter_fed_minutes": "fed_minutes",
    "iter_fed_press_conf": "fed_press_conf",
    "iter_fed_speeches": "fed_speeches",
    "iter_banxico_decision": "banxico",
    "iter_bcb_decision": "bcb",
    "iter_bcb_minutes": "bcb",
    "iter_bccl_decision": "bccl",
    "iter_bcra_decision": "bcra",
    "iter_banrep_decision": "banrep",
    "iter_rba_decision": "rba",
    "iter_rba_speeches": "rba",
    "iter_rbnz_decision": "rbnz",
    "iter_riksbank_decision": "riksbank",
    "iter_norges_decision": "norges",
    "iter_sarb_decision": "sarb",
    "iter_pboc_decision": "pboc",
    "iter_pboc_speeches": "pboc",
    "iter_rbi_decision": "rbi",
    "iter_bok_decision": "bok",
    "iter_bok_minutes": "bok",
    "iter_mas_decision": "mas",
    "iter_bot_decision": "bot",
    "iter_bot_news": "bot",
    "iter_bis_speeches": "bis_speeches",
    "iter_nber_wp": "nber_wp",
    "iter_hackernews": "hackernews",
    "iter_reddit": "reddit",
    "iter_macro_blogs": "macro_blogs",
    "MACRO_BLOG_FEEDS": "macro_blogs",
    "parse_transcript": "transcripts",
    "TranscriptDocument": "transcripts",
    "DialogueTurn": "transcripts",
    "extract_corporate_macro_risks": "sec_edgar",
    "CorporateMacroRiskMatch": "sec_edgar",
    "CORPORATE_MACRO_PATTERNS": "sec_edgar",
    "iter_us_warn": "us_warn",
    "iter_frdb_strikes": "frdb_strikes",
    "validate_records": "_schema",
    "SourceRecordValidationError": "_schema",
}

# ``BLUESKY_KNOWN_HANDLES`` is re-exported under a different name than its
# definition (``bluesky.KNOWN_HANDLES``), so it gets a small special case below.
_ALIASES = {
    "BLUESKY_KNOWN_HANDLES": ("bluesky", "KNOWN_HANDLES"),
}

__all__ = sorted([*_MODULE_OF, *_ALIASES])


def __getattr__(name):
    if name in _ALIASES:
        mod, attr = _ALIASES[name]
    elif name in _MODULE_OF:
        mod, attr = _MODULE_OF[name], name
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{mod}", __name__)
    value = getattr(module, attr)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__():
    return __all__
