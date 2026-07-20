"""Bank for International Settlements speech archive (multi-bank).

The BIS republishes speeches by senior officials of its ~60 member
central banks at https://www.bis.org/cbspeeches/. We pull the master
RSS feed and tag every item with ``bank_code="BIS"``.

Country attribution: BIS speech descriptions consistently include the
speaker's affiliation (e.g., "Governor of the Bank of Korea",
"Executive Board of the ECB"). ``_resolve_country()`` searches the
text against a known issuing-bank → ISO3 dict; if a match is found,
the record's ``country`` metadata is overridden from ``"MULTI"`` to
the inferred ISO3. Records with no recognised affiliation keep
``country="MULTI"``.

Callers who want bank-specific filtering can pass ``bank_filter`` (a
string matched against the title, e.g., ``"Lagarde"``, ``"Fed"``).
"""
from __future__ import annotations

from typing import Iterator

from ._rss_filtered import iter_rss_filtered


_FEED = "https://www.bis.org/doclist/cbspeeches.rss"


# Issuing-institution → ISO3 country. Longest match wins to disambiguate
# (e.g., "Bank of England" vs "European Central Bank"). Entries are
# matched as plain case-insensitive substrings against the speech text
# (title + description).
_BANK_TO_ISO3: dict[str, str] = {
    # Eurosystem (EA20)
    "European Central Bank": "EA20",
    "Banco Central Europeo": "EA20",
    "Banque centrale européenne": "EA20",
    "Europäische Zentralbank": "EA20",
    "Executive Board of the ECB": "EA20",
    "Deutsche Bundesbank": "DEU",
    "Bundesbank": "DEU",
    "Banque de France": "FRA",
    "Bank of France": "FRA",
    "Banca d'Italia": "ITA",
    "Bank of Italy": "ITA",
    "Banco de España": "ESP",
    "Bank of Spain": "ESP",
    "Banco de Portugal": "PRT",
    "Bank of Portugal": "PRT",
    "De Nederlandsche Bank": "NLD",
    "Nationale Bank van België": "BEL",
    "National Bank of Belgium": "BEL",
    "Banque nationale de Belgique": "BEL",
    "Oesterreichische Nationalbank": "AUT",
    "Bank of Finland": "FIN",
    "Suomen Pankki": "FIN",
    "Bank of Greece": "GRC",
    "Central Bank of Ireland": "IRL",
    "Banque centrale du Luxembourg": "LUX",
    "Central Bank of Cyprus": "CYP",
    "Central Bank of Malta": "MLT",
    "Banka Slovenije": "SVN",
    "Národná banka Slovenska": "SVK",
    "Eesti Pank": "EST",
    "Latvijas Banka": "LVA",
    "Lietuvos bankas": "LTU",
    "Hrvatska narodna banka": "HRV",
    # Other Europe
    "Bank of England": "GBR",
    "Swiss National Bank": "CHE",
    "Schweizerische Nationalbank": "CHE",
    "Banque nationale suisse": "CHE",
    "Norges Bank": "NOR",
    "Sveriges Riksbank": "SWE",
    "Riksbank": "SWE",
    "Danmarks Nationalbank": "DNK",
    "Seðlabanki Íslands": "ISL",
    "Central Bank of Iceland": "ISL",
    "Czech National Bank": "CZE",
    "Česká národní banka": "CZE",
    "Magyar Nemzeti Bank": "HUN",
    "Narodowy Bank Polski": "POL",
    "National Bank of Poland": "POL",
    "Banca Naţională a României": "ROU",
    "National Bank of Romania": "ROU",
    "National Bank of Serbia": "SRB",
    "Bank of Russia": "RUS",
    "Central Bank of Russia": "RUS",
    # North America
    "Federal Reserve": "USA",
    "Bank of Canada": "CAN",
    "Banque du Canada": "CAN",
    "Banco de México": "MEX",
    "Bank of Mexico": "MEX",
    # Latin America
    "Banco Central de la República Argentina": "ARG",
    "Central Bank of Argentina": "ARG",
    "Banco Central do Brasil": "BRA",
    "Central Bank of Brazil": "BRA",
    "Banco Central de Chile": "CHL",
    "Central Bank of Chile": "CHL",
    "Banco de la República": "COL",
    "Banco Central de Reserva del Perú": "PER",
    "Central Reserve Bank of Peru": "PER",
    "Banco Central de Reserva de El Salvador": "SLV",
    "Banco de Guatemala": "GTM",
    "Banco Central del Uruguay": "URY",
    "Banco Central de Venezuela": "VEN",
    "Banco Central de Bolivia": "BOL",
    # Asia-Pacific
    "Reserve Bank of Australia": "AUS",
    "Reserve Bank of New Zealand": "NZL",
    "Bank of Japan": "JPN",
    "Bank of Korea": "KOR",
    "People's Bank of China": "CHN",
    "Hong Kong Monetary Authority": "HKG",
    "Monetary Authority of Singapore": "SGP",
    "Reserve Bank of India": "IND",
    "Bank Indonesia": "IDN",
    "Bank Negara Malaysia": "MYS",
    "Central Bank of Malaysia": "MYS",
    "Bank of Thailand": "THA",
    "Bangko Sentral ng Pilipinas": "PHL",
    "State Bank of Vietnam": "VNM",
    "Bank of Mongolia": "MNG",
    "Central Bank of Sri Lanka": "LKA",
    "Bangladesh Bank": "BGD",
    "State Bank of Pakistan": "PAK",
    # Middle East & Africa
    "Saudi Central Bank": "SAU",
    "Saudi Arabian Monetary Authority": "SAU",
    "Central Bank of the United Arab Emirates": "ARE",
    "Bank of Israel": "ISR",
    "Central Bank of the Republic of Turkey": "TUR",
    "Türkiye Cumhuriyet Merkez Bankası": "TUR",
    "Central Bank of Egypt": "EGY",
    "South African Reserve Bank": "ZAF",
    "Bank of Zambia": "ZMB",
    "Bank of Uganda": "UGA",
    "Bank of Ghana": "GHA",
    "Bank of Mauritius": "MUS",
    "Central Bank of Kenya": "KEN",
    "Central Bank of Nigeria": "NGA",
    "Reserve Bank of Malawi": "MWI",
    "Bank of Tanzania": "TZA",
    "Banque Centrale des États de l'Afrique de l'Ouest": "BCEAO",
    "Banque des États de l'Afrique Centrale": "BEAC",
    # Caribbean / small jurisdictions
    "Central Bank of Curaçao and Sint Maarten": "CUW",
    "Central Bank of Curacao and Sint Maarten": "CUW",
    "Central Bank of Trinidad and Tobago": "TTO",
    "Central Bank of Barbados": "BRB",
    "Bank of Jamaica": "JAM",
}

_BANK_KEYS_SORTED = sorted(_BANK_TO_ISO3.keys(), key=len, reverse=True)


def _resolve_country(text: str) -> str:
    """Return ISO3 (or "MULTI") for a speech via longest-prefix search of
    the speaker's affiliation in the text. Empty / unrecognised → "MULTI"."""
    if not text:
        return "MULTI"
    lowered = text.lower()
    for key in _BANK_KEYS_SORTED:
        if key.lower() in lowered:
            return _BANK_TO_ISO3[key]
    return "MULTI"


def iter_bis_speeches(*, bank_filter: str | None = None,
                      fetch_body: bool = True) -> Iterator[tuple]:
    """Yield (date, text, url, metadata) for BIS-republished CB speeches.

    Country attribution: each record's ``metadata["country"]`` is set to
    the speaker's issuing CB's ISO3 (via ``_resolve_country``) instead
    of the default ``"MULTI"``. Records where the affiliation cannot be
    parsed keep ``country="MULTI"``.

    Parameters
    ----------
    bank_filter : optional substring to match against the title (case-
        insensitive). Useful to narrow to one institution's speeches
        (e.g., ``"Lagarde"`` for ECB, ``"Powell"`` for Fed).
    fetch_body : default ``True``. If ``True``, fetch the underlying
        BIS review page for each RSS item and replace the short RSS
        summary with the full speech body (via the registered BIS
        body extractor). Forwarded verbatim to ``iter_rss_filtered``.
    """
    title_keywords = [bank_filter] if bank_filter else None
    for date, text, link, meta in iter_rss_filtered(
        _FEED,
        bank_code="BIS", country="MULTI",
        doctype="speech", language="en",
        title_keywords=title_keywords,
        fetch_body=fetch_body,
    ):
        country = _resolve_country(text or "")
        new_meta = dict(meta or {})
        new_meta["country"] = country
        new_meta["bis_resolved"] = country != "MULTI"
        yield (date, text, link, new_meta)


__all__ = ["iter_bis_speeches"]
