"""Which series carries which variable, for each real-time provider.

Three things live here: the canonical variable vocabulary (with SDMX
aliases, so ``series="B1GQ"`` works as the issue asks), a
provider-by-country map from a variable to that provider's series
identifier, and — load-bearing — the **units** each of those series is
published in.

WHY UNITS ARE IN THE TABLE
--------------------------
FRED carries, for the same country and concept, both level series
(``CLVMNACSCAB1GQDE``, millions of chained euros) and growth-rate
series (``IRLGDPRQPSMEI``, "growth rate same period previous year").
Nothing in the identifier says which. Feed a growth-rate series to a
revision test configured for levels and it silently differences an
already-differenced series; the output is well-formed and wrong.

So every entry declares its units, :func:`resolve_spec` exposes them,
and the assembled panel carries them so a cross-country regression can
refuse to mix them.

ON THE HONESTY OF THIS TABLE
----------------------------
Every entry is a claim that an identifier exists, is still maintained,
and carries more than one archived edition. Identifiers rot: this
package previously resolved every non-US country to FRED's OECD-MEI
codes (``NAEXKP01<ISO2>Q652S``), a family that stopped updating in
January 2024 — which is why cross-country revision work through it
collapsed to a couple of usable countries.

So the table is self-auditing. Run::

    pytest -m network tests/test_realtime_providers/test_catalog_live.py

to walk every entry, fetch its vintage list, and report which no
longer return two or more editions. That is the difference between
"these countries do not publish vintages" and "this table went stale".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


#: Canonical variable names and what they mean.
CANONICAL_VARIABLES: dict[str, str] = {
    "gdp_real": "Gross domestic product, chained volume / constant prices, SA",
    "gdp_nom": "Gross domestic product, current prices, SA",
    "deflator": "GDP implicit price deflator, SA",
    "con_real": "Household final consumption expenditure, real, SA",
    "govcon_real": "Government final consumption expenditure, real, SA",
    "gfcf_real": "Gross fixed capital formation, real, SA",
    "exports_real": "Exports of goods and services, real, SA",
    "imports_real": "Imports of goods and services, real, SA",
    "employment": "Total employment, SA",
    "unemployment_rate": "Unemployment rate, SA",
    "cpi": "Consumer price index, all items",
    "ip": "Industrial production index, SA",
}

#: Accepted spellings, including the ESA/SDMX transaction codes the
#: issue's ``series="B1GQ"`` example uses.
VARIABLE_ALIASES: dict[str, str] = {
    "B1GQ": "gdp_real", "B1G": "gdp_real",
    "P3": "con_real", "P31_S14": "con_real", "P31_S14_S15": "con_real",
    "P3_S13": "govcon_real", "P51G": "gfcf_real",
    "P6": "exports_real", "P7": "imports_real",
    "gdp": "gdp_real", "rgdp": "gdp_real", "real_gdp": "gdp_real",
    "gdp_deflator": "deflator", "consumption": "con_real",
    "investment": "gfcf_real", "government": "govcon_real",
    "exports": "exports_real", "imports": "imports_real",
    "log_gdp_real": "gdp_real", "log_gdp_nom": "gdp_nom",
    "log_con_real": "con_real", "log_govcon_real": "govcon_real",
    "log_gfcf_real": "gfcf_real", "log_exports_real": "exports_real",
    "log_imports_real": "imports_real", "log_deflator": "deflator",
}

#: Units a catalogued series can be published in, and the revision
#: transform each implies. ``level`` series must be differenced before
#: a news/noise test; series already expressed as growth must not be.
UNITS_TRANSFORM: dict[str, str] = {
    "level": "log_diff_pct",
    "index": "log_diff_pct",
    "growth_qoq": "level",
    "growth_yoy": "level",
    "rate": "level",
}


@dataclass(frozen=True)
class SeriesSpec:
    """One provider's series for one (country, variable)."""
    series_id: str
    units: str = "level"
    source: str = ""
    note: str = ""

    def default_transform(self) -> str:
        """The revision transform these units imply."""
        return UNITS_TRANSFORM.get(self.units, "log_diff_pct")


def canonical_variable(variable: str) -> str:
    """Map any accepted spelling onto a canonical variable name.

    Raises
    ------
    ValueError
        On an unknown name. A silent fallthrough would produce an empty
        panel reading as "this country has no vintages" rather than
        "you made a typo".
    """
    for probe in (variable, variable.upper(), variable.lower()):
        if probe in CANONICAL_VARIABLES:
            return probe
        key = VARIABLE_ALIASES.get(probe)
        if key is not None:
            return key
    raise ValueError(
        f"unknown variable {variable!r}. Canonical names: "
        f"{sorted(CANONICAL_VARIABLES)}. Aliases include "
        f"{sorted(VARIABLE_ALIASES)[:8]}..."
    )


# ---------------------------------------------------------------------------
# ALFRED (FRED archive)
# ---------------------------------------------------------------------------
# Verified against the FRED API on 2026-08-23: every id below returned
# >= 8 archived vintage dates. Eurostat's quarterly national accounts,
# replicated into FRED, use a mechanical scheme —
#     CLV MNAC SCA B1GQ <geo>
# chain-linked volumes, national currency, seasonally + calendar
# adjusted, GDP — so euro-area members share one template. Note the
# geo codes are Eurostat's, not ISO 3166: Greece is EL and the United
# Kingdom is UK.
_EUROSTAT_SCA = {
    "AUT": "AT", "BEL": "BE", "CHE": "CH", "CZE": "CZ", "DEU": "DE",
    "DNK": "DK", "ESP": "ES", "EST": "EE", "FIN": "FI", "FRA": "FR",
    "GRC": "EL", "HUN": "HU", "ITA": "IT", "LTU": "LT", "LUX": "LU",
    "LVA": "LV", "NLD": "NL", "NOR": "NO", "POL": "PL", "PRT": "PT",
    "SVN": "SI", "SWE": "SE",
}

# Seasonally adjusted but not calendar adjusted — the only variant
# Eurostat/FRED publishes for these.
_EUROSTAT_SA = {"ISL": "IS"}

# IMF IFS replication, for economies outside the Eurostat perimeter.
# Vintage history starts 2021-06-14 for this whole family.
_IMF = {
    "AUS": "AU", "BRA": "BR", "CAN": "CA", "IDN": "ID", "KOR": "KR",
    "MEX": "MX", "TUR": "TR", "ZAF": "ZA",
}

_IMF_NOTE = "IMF IFS replication; ALFRED vintages begin 2021-06-14"

_ALFRED_SPECS: dict[str, dict[str, SeriesSpec]] = {}

for _iso3, _geo in _EUROSTAT_SCA.items():
    _ALFRED_SPECS[_iso3] = {"gdp_real": SeriesSpec(
        f"CLVMNACSCAB1GQ{_geo}", "level", "eurostat")}
for _iso3, _geo in _EUROSTAT_SA.items():
    _ALFRED_SPECS[_iso3] = {"gdp_real": SeriesSpec(
        f"CLVMNACSAB1GQ{_geo}", "level", "eurostat",
        "seasonally but not calendar adjusted")}
for _iso3, _geo in _IMF.items():
    _ALFRED_SPECS[_iso3] = {"gdp_real": SeriesSpec(
        f"NGDPRSAXDC{_geo}Q", "level", "imf", _IMF_NOTE)}

_ALFRED_SPECS["EA19"] = {"gdp_real": SeriesSpec(
    "CLVMNACSCAB1GQEA19", "level", "eurostat", "euro area, 19 countries")}
_ALFRED_SPECS["GBR"] = {"gdp_real": SeriesSpec(
    "CLVMNACSCAB1GQUK", "level", "eurostat",
    "DISCONTINUED 2021-02: Eurostat stopped receiving UK data after "
    "Brexit. 40 vintages, 2016-04 to 2021-02. For current UK vintages "
    "use the ONS provider.")}
_ALFRED_SPECS["JPN"] = {"gdp_real": SeriesSpec(
    "JPNRGDPEXP", "level", "national", "Cabinet Office via FRED")}
_ALFRED_SPECS["USA"] = {
    "gdp_real": SeriesSpec("GDPC1", "level", "bea"),
    "gdp_nom": SeriesSpec("GDP", "level", "bea"),
    "deflator": SeriesSpec("GDPDEF", "index", "bea"),
    "con_real": SeriesSpec("PCECC96", "level", "bea"),
    "govcon_real": SeriesSpec("GCEC1", "level", "bea"),
    "gfcf_real": SeriesSpec("GPDIC1", "level", "bea"),
    "exports_real": SeriesSpec("EXPGSC1", "level", "bea"),
    "imports_real": SeriesSpec("IMPGSC1", "level", "bea"),
    "employment": SeriesSpec("PAYEMS", "level", "bls"),
    "unemployment_rate": SeriesSpec("UNRATE", "rate", "bls"),
    "cpi": SeriesSpec("CPIAUCSL", "index", "bls"),
    "ip": SeriesSpec("INDPRO", "index", "frb"),
}

#: Countries with no usable ALFRED *level* series for real GDP, and
#: why. Kept explicit so "missing" is a documented fact rather than a
#: silent absence — see :func:`known_gaps`.
ALFRED_KNOWN_GAPS: dict[str, str] = {
    "IRL": "CLVMNACSAB1GQIE was discontinued in 2016 (ends 2015Q4); the "
           "only maintained Irish series on FRED (IRLGDPRQPSMEI) is a "
           "year-on-year growth rate, not a level.",
    "SVK": "CLVMNACSAB1GQSK was discontinued in 2020 (ends 2020Q3).",
    "NZL": "Only NZLGDPRQPSMEI, a year-on-year growth rate, is archived.",
    "ISR": "Only ISRGDPRQPSMEI, a year-on-year growth rate, is archived.",
    "CHL": "Only CHLGDPRQPSMEI, a year-on-year growth rate, is archived.",
    "IND": "Only INDGDPRQPSMEI (growth rate) and an NSA level series.",
    "COL": "No archived quarterly real-GDP series found.",
}


#: provider -> country -> variable -> SeriesSpec. Providers other than
#: ALFRED register themselves at import time as their modules load.
_CATALOGS: dict[str, dict[str, dict[str, SeriesSpec]]] = {
    "alfred": _ALFRED_SPECS,
}

#: provider -> country -> reason it is unavailable there.
_GAPS: dict[str, dict[str, str]] = {"alfred": ALFRED_KNOWN_GAPS}


def register_catalog(
    provider: str,
    mapping: dict[str, dict[str, SeriesSpec | str]],
    *,
    gaps: dict[str, str] | None = None,
) -> None:
    """Attach ``provider``'s country -> variable -> series mapping.

    Bare strings are accepted and promoted to level-units specs, which
    keeps simple connectors terse.
    """
    table = _CATALOGS.setdefault(provider, {})
    for country, by_var in mapping.items():
        target = table.setdefault(str(country).upper(), {})
        for variable, spec in by_var.items():
            target[canonical_variable(variable)] = (
                spec if isinstance(spec, SeriesSpec) else SeriesSpec(str(spec))
            )
    if gaps:
        _GAPS.setdefault(provider, {}).update(gaps)


def _normalise_override(table: dict) -> dict:
    """Apply :func:`register_catalog`'s key normalisation to an override.

    A user-supplied ``catalog=`` dict is written by hand, so its keys
    are whatever the user typed — ``{"deu": {"B1GQ": ...}}``. Without
    this the override would look empty and the country would be
    reported missing, which reads as "no vintages exist" rather than
    "your override key did not match".
    """
    out: dict[str, dict[str, object]] = {}
    for country, by_var in (table or {}).items():
        target = out.setdefault(str(country).upper(), {})
        for variable, spec in (by_var or {}).items():
            try:
                target[canonical_variable(variable)] = spec
            except ValueError:
                target[variable] = spec
    return out


def resolve_spec(
    provider: str, country: str, variable: str, *, catalog: dict | None = None,
) -> SeriesSpec | None:
    """The full :class:`SeriesSpec` for one (provider, country, variable)."""
    var = canonical_variable(variable)
    if catalog is not None:
        table = _normalise_override(catalog.get(provider, {}))
    else:
        table = _CATALOGS.get(provider, {})
    spec = table.get(str(country).upper(), {}).get(var)
    if spec is None or isinstance(spec, SeriesSpec):
        return spec
    return SeriesSpec(str(spec))


def resolve_series(
    provider: str, country: str, variable: str, *, catalog: dict | None = None,
) -> str | None:
    """The provider's identifier for one (country, variable), or None.

    ``catalog`` overrides the built-in table entirely — the escape
    hatch for a series this package does not know about yet::

        vintage_panel(["DEU"], catalog={"alfred": {"DEU": {"gdp_real": "..."}}})
    """
    spec = resolve_spec(provider, country, variable, catalog=catalog)
    return None if spec is None else spec.series_id


def provider_country_codes(provider: str) -> list[str]:
    """Countries ``provider`` claims to cover."""
    return sorted(_CATALOGS.get(provider, {}))


def providers_for(country: str, variable: str = "gdp_real") -> list[str]:
    """Every provider that can serve this (country, variable)."""
    var = canonical_variable(variable)
    c = str(country).upper()
    return sorted(p for p, table in _CATALOGS.items() if var in table.get(c, {}))


def known_gaps(provider: str | None = None) -> dict[str, str]:
    """Countries a provider deliberately does not cover, and why."""
    if provider is not None:
        return dict(_GAPS.get(provider, {}))
    merged: dict[str, str] = {}
    for prov, gaps in _GAPS.items():
        for country, reason in gaps.items():
            merged[f"{prov}:{country}"] = reason
    return merged


def vintage_catalog(provider: str | None = None) -> pd.DataFrame:
    """The whole table as a DataFrame, for inspection.

    Columns ``[provider, country, variable, series_id, units,
    default_transform, source, note, description]``.
    """
    rows = []
    for prov, table in sorted(_CATALOGS.items()):
        if provider is not None and prov != provider:
            continue
        for country, mapping in sorted(table.items()):
            for variable, spec in sorted(mapping.items()):
                rows.append({
                    "provider": prov,
                    "country": country,
                    "variable": variable,
                    "series_id": spec.series_id,
                    "units": spec.units,
                    "default_transform": spec.default_transform(),
                    "source": spec.source,
                    "note": spec.note,
                    "description": CANONICAL_VARIABLES.get(variable, ""),
                })
    return pd.DataFrame(rows, columns=[
        "provider", "country", "variable", "series_id", "units",
        "default_transform", "source", "note", "description",
    ])


__all__ = [
    "CANONICAL_VARIABLES",
    "VARIABLE_ALIASES",
    "UNITS_TRANSFORM",
    "SeriesSpec",
    "ALFRED_KNOWN_GAPS",
    "canonical_variable",
    "register_catalog",
    "resolve_spec",
    "resolve_series",
    "provider_country_codes",
    "providers_for",
    "known_gaps",
    "vintage_catalog",
]
