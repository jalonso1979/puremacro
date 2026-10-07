r"""One-call cross-country annual national accounts: the annual twin of ``qna_panel``.

``ana_panel(["USA", "FRA", "MEX"])`` returns an annual panel indexed by
``(code, date)``, one row per country and year with the date at 1 January, and
the expenditure side of the national accounts in **current prices** (millions
of national currency), the matching **implicit deflators** (2020 = 100) and,
by default, the **chain-linked volumes** at 2020 prices::

    gdp  cons_hh  cons_gov  inv  capform  exports  imports  discrepancy_exp
    gdp_defl  cons_hh_defl  ...      gdp_real  cons_hh_real  ...

so that ``gdp_real = 100 * gdp / gdp_defl`` and

.. math:: Y = C_{hh} + C_{gov} + I_{cap} + X - M + \text{YA0}

holds to rounding in current prices. Seven switches join the rest of the
annual accounts on the same index: the income approach (with gross operating
surplus and mixed income split apart, which the quarterly accounts never do),
the output approach by A10 activity, household consumption by durability, gross
fixed capital formation by asset and by institutional sector (public
investment), fixed capital stocks, and employment, hours and population.

Why an annual panel, when the quarterly one exists
--------------------------------------------------
Because the annual tables start decades earlier and cover more countries. The
quarterly flow begins in the 1990s for most of Europe; the annual tables give
France, the United Kingdom, Sweden and India from 1950, Korea from 1953 and
some thirty countries from 1970 or before, 64 countries in all against 49.
They also carry what the quarterly accounts do not publish at all: mixed income
separately from operating surplus (the Gollin correction to the labour share
needs exactly that split), consumption of fixed capital, public investment and
public capital stocks, and net and gross capital stocks with volumes.

How the volumes and deflators are built
---------------------------------------
The OECD publishes each series at current prices (``V``), as a chain-linked
volume at the country's own reference year (``L``: Mexico 2018, the United
States 2017, ...) and as the same volume **rebased to 2020** (``LR``). The
panel takes ``LR`` wherever it exists, so every ``_real`` column is in 2020
prices and every deflator ``<x>_defl = 100 * V / LR`` equals 100 in 2020 for
every country; that is the OECD's own published deflator (``PRICE_BASE=DR``),
to the last digit. Where a series has ``L`` but no ``LR`` (the capital stocks,
some durability splits) the ``L`` volume is rebased here to 2020 by the ratio
``V / L`` in 2020, which is exactly what ``LR`` is; if 2020 is missing it is
kept at its own reference year and ``attrs["meta"]`` says so. Chain-linked
volumes are **not additive**: the sum of the components' ``_real`` columns is
not ``gdp_real`` except in 2020.

Mechanics
---------
One SDMX request per flow per chunk of at most ten countries, every key pinned
with :func:`puremacro.fetch._oecd_sdmx.oecd_key` so the dimension count is
right by construction (``DSD_NAMAIN10`` has 12 dimensions, ``DSD_NASEC10``
13). The main aggregates, the asset split, the income and the output side all
come from the one table ``DF_TABLE1`` (tables 0101, 0102 and 0103 together),
so switching on ``assets``, ``income`` or ``output`` widens that request rather
than adding one. GDP appears in all three tables; each series (country,
variable, price base) is read whole from the best table that has it, the
expenditure table 0102 first, so the approaches are never spliced year by
year, and ``meta["tables_not_t0102"]`` lists the exceptions (Russia's volumes
come from 0101). Requests go through
:func:`puremacro.fetch._oecd_sdmx.oecd_csv` (urllib, paced, cached on disk
outside the repository). A provider failure never raises: the block is left
out, the request is listed in ``attrs["missing"]`` with its status (``HTTP
429``, ``timeout``, ``IncompleteRead``, ...), and a warning says so; a re-run
re-sends only those. A key the OECD answers with nothing (``HTTP 404`` or no
rows) goes to ``attrs["chunks_empty"]`` instead, since re-running cannot
change it. ``attrs["requests"]`` logs every request sent, with its key.

Source: OECD SDMX, agency ``OECD.SDD.NAD``: ``DSD_NAMAIN10@DF_TABLE1``,
``DF_TABLE5A_T117``, ``DF_TABLE5_T117``, ``DF_TABLE9A``, ``DF_TABLE3_EMPDC``,
``DF_TABLE3_POP_EMPNC``; ``DSD_NASEC10@DF_TABLE14``, ``DF_TABLE14_GFCF``,
``DF_TABLE9B`` (https://sdmx.oecd.org/public/rest).
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ._hours import hours_scale_factors
from ._oecd_sdmx import OECD_AGGREGATES, oecd_availability, oecd_csv, oecd_key

# ---------------------------------------------------------------------------
# Dataflows and their dimension orders
# ---------------------------------------------------------------------------

#: Dimension order of ``DSD_NAMAIN10`` (annual main aggregates), 12 dimensions.
NAMAIN10_DIMS: tuple[str, ...] = (
    "FREQ", "REF_AREA", "SECTOR", "COUNTERPART_SECTOR", "TRANSACTION",
    "INSTR_ASSET", "ACTIVITY", "EXPENDITURE", "UNIT_MEASURE", "PRICE_BASE",
    "TRANSFORMATION", "TABLE_IDENTIFIER")

#: Dimension order of ``DSD_NASEC10`` (annual sector accounts), 13 dimensions:
#: ``ACCOUNTING_ENTRY`` after the counterpart sector, ``VALUATION`` after the
#: unit, and no ``ACTIVITY``.
NASEC10_DIMS: tuple[str, ...] = (
    "FREQ", "REF_AREA", "SECTOR", "COUNTERPART_SECTOR", "ACCOUNTING_ENTRY",
    "TRANSACTION", "INSTR_ASSET", "EXPENDITURE", "UNIT_MEASURE", "VALUATION",
    "PRICE_BASE", "TRANSFORMATION", "TABLE_IDENTIFIER")

_A = "OECD.SDD.NAD,"

#: Short name -> ``agency,DSD@flow,version``. Versions are pinned: a blank one
#: resolves to the latest, but a stale explicit one answers with fewer
#: reference areas, and ``latest`` is refused with HTTP 400.
ANA_FLOWS: dict[str, str] = {
    "table1":       _A + "DSD_NAMAIN10@DF_TABLE1,2.0",
    "table5a":      _A + "DSD_NAMAIN10@DF_TABLE5A_T117,2.0",
    "table5":       _A + "DSD_NAMAIN10@DF_TABLE5_T117,1.0",
    "table9a":      _A + "DSD_NAMAIN10@DF_TABLE9A,2.0",
    "empdc":        _A + "DSD_NAMAIN10@DF_TABLE3_EMPDC,2.0",
    "pop":          _A + "DSD_NAMAIN10@DF_TABLE3_POP_EMPNC,2.0",
    "table14":      _A + "DSD_NASEC10@DF_TABLE14,1.1",
    "table14_gfcf": _A + "DSD_NASEC10@DF_TABLE14_GFCF,1.1",
    "table9b":      _A + "DSD_NASEC10@DF_TABLE9B,1.1",
}

#: The flow :func:`ana_countries` asks which reference areas publish GDP.
_COUNTRIES_FLOW = _A + "DSD_NAMAIN10@DF_TABLE1_EXPENDITURE,2.0"

# ---------------------------------------------------------------------------
# Registries: column name -> (pinned SDMX dimensions, description)
# ---------------------------------------------------------------------------

#: The expenditure side, from ``DF_TABLE1`` (table 0102). ``discrepancy_exp``
#: is the statistical discrepancy ``YA0``: in Mexico it is 1-5 % of GDP in most
#: years, so ``C + G + I + X - M`` does not close without it.
ANA_MAIN: dict[str, tuple[dict[str, str], str]] = {
    "gdp":      ({"TRANSACTION": "B1GQ", "SECTOR": "S1"},
                 "Gross domestic product (expenditure approach)"),
    "cons_hh":  ({"TRANSACTION": "P3", "SECTOR": "S1M"},
                 "Final consumption of households and NPISH"),
    "cons_gov": ({"TRANSACTION": "P3", "SECTOR": "S13"},
                 "General government final consumption"),
    "inv":      ({"TRANSACTION": "P51G", "SECTOR": "S1", "INSTR_ASSET": "N11G",
                  "ACTIVITY": "_T"}, "Gross fixed capital formation"),
    "capform":  ({"TRANSACTION": "P5", "SECTOR": "S1", "INSTR_ASSET": "N1G",
                  "ACTIVITY": "_T"}, "Gross capital formation (GFCF + inventories + valuables)"),
    "exports":  ({"TRANSACTION": "P6", "SECTOR": "S1"}, "Exports of goods and services"),
    "imports":  ({"TRANSACTION": "P7", "SECTOR": "S1"}, "Imports of goods and services"),
    "discrepancy_exp": ({"TRANSACTION": "YA0"},
                        "Statistical discrepancy, expenditure approach (YA0)"),
}

#: The income approach, from ``DF_TABLE1`` (table 0103), current prices only:
#: :math:`Y = D1 + B2A3G + (D2 - D3) + \text{YA2}` holds exactly where all four
#: are published. ``subsidies`` is a positive number that enters negatively.
ANA_INCOME: dict[str, tuple[dict[str, str], str]] = {
    "gdp_income":    ({"TRANSACTION": "B1GQ", "TABLE_IDENTIFIER": "T0103"},
                      "GDP as the income table publishes it"),
    "comp_emp":      ({"TRANSACTION": "D1", "ACTIVITY": "_T"},
                      "Compensation of employees"),
    "surplus_mixed": ({"TRANSACTION": "B2A3G", "ACTIVITY": "_T"},
                      "Gross operating surplus and mixed income"),
    "taxes_prod_imp": ({"TRANSACTION": "D2"}, "Taxes on production and imports"),
    "subsidies":     ({"TRANSACTION": "D3"}, "Subsidies (positive; enter negatively)"),
    "taxes_prod_imp_net": ({"TRANSACTION": "D2X3"},
                           "Taxes less subsidies on production and imports"),
    "discrepancy_inc": ({"TRANSACTION": "YA2"},
                        "Statistical discrepancy, income approach (YA2)"),
}

#: The part of the income block that only the sector accounts publish, from
#: ``DSD_NASEC10@DF_TABLE14`` for the total economy (``S1``). ``surplus_gross
#: + mixed_income == surplus_mixed`` exactly; mixed income is a fifth to a
#: third of it (Germany 0.21, United States 0.23, Mexico 0.33 in 2018-24),
#: which is the self-employed labour income the Gollin (2002) correction to
#: ``comp_emp / gdp`` is about. The table books most items twice, once as a
#: resource (``C``) and once as a use (``D``); the uses side is the one that
#: matches table 0103, Mexico publishes mixed income only there, and VAT
#: exists only as a resource, hence the pins.
ANA_INCOME_SECTOR: dict[str, tuple[dict[str, str], str]] = {
    "surplus_gross": ({"TRANSACTION": "B2G", "ACCOUNTING_ENTRY": "D", "SECTOR": "S1"},
                      "Gross operating surplus (B2G)"),
    "mixed_income":  ({"TRANSACTION": "B3G", "ACCOUNTING_ENTRY": "D", "SECTOR": "S1"},
                      "Gross mixed income (B3G)"),
    "cfc":           ({"TRANSACTION": "P51C", "ACCOUNTING_ENTRY": "D", "SECTOR": "S1"},
                      "Consumption of fixed capital (P51C)"),
    "vat":           ({"TRANSACTION": "D211", "ACCOUNTING_ENTRY": "C", "SECTOR": "S1"},
                      "Value added type taxes (D211)"),
}

#: The output approach by A10 activity, from ``DF_TABLE1`` (table 0101):
#: :math:`Y = \sum_j \text{VA}_j + \text{D21X31} + \text{YA1}`. ``va_mfg`` is a
#: memo item inside ``va_ind``; sum :data:`ANA_VA_ADDITIVE`, not every
#: ``va_*`` column. ``va_services`` is not published at this level and is
#: built here as the sum of the seven service activities, only in years where
#: all seven are present.
ANA_OUTPUT: dict[str, tuple[dict[str, str], str]] = {
    "gdp_output":  ({"TRANSACTION": "B1GQ", "TABLE_IDENTIFIER": "T0101"},
                    "GDP as the output table publishes it"),
    "va_total":    ({"TRANSACTION": "B1G", "ACTIVITY": "_T"}, "Gross value added, all activities"),
    "va_agri":     ({"TRANSACTION": "B1G", "ACTIVITY": "A"}, "Agriculture, forestry and fishing"),
    "va_ind":      ({"TRANSACTION": "B1G", "ACTIVITY": "BTE"}, "Industry except construction (B-E)"),
    "va_mfg":      ({"TRANSACTION": "B1G", "ACTIVITY": "C"}, "Manufacturing (memo: inside va_ind)"),
    "va_constr":   ({"TRANSACTION": "B1G", "ACTIVITY": "F"}, "Construction"),
    "va_trade":    ({"TRANSACTION": "B1G", "ACTIVITY": "GTI"},
                    "Trade, transport, accommodation and food (G-I)"),
    "va_ict":      ({"TRANSACTION": "B1G", "ACTIVITY": "J"}, "Information and communication"),
    "va_fin":      ({"TRANSACTION": "B1G", "ACTIVITY": "K"}, "Financial and insurance activities"),
    "va_realest":  ({"TRANSACTION": "B1G", "ACTIVITY": "L"}, "Real estate activities"),
    "va_prof":     ({"TRANSACTION": "B1G", "ACTIVITY": "M_N"},
                    "Professional, scientific, administrative (M-N)"),
    "va_public":   ({"TRANSACTION": "B1G", "ACTIVITY": "OTQ"},
                    "Public administration, education, health (O-Q)"),
    "va_other":    ({"TRANSACTION": "B1G", "ACTIVITY": "RTU"}, "Other services (R-U)"),
    "taxes_prod":  ({"TRANSACTION": "D21X31"}, "Taxes less subsidies on products"),
    "discrepancy_out": ({"TRANSACTION": "YA1"},
                        "Statistical discrepancy, output approach (YA1)"),
}

#: The value-added columns that sum to ``va_total``.
ANA_VA_ADDITIVE: tuple[str, ...] = (
    "va_agri", "va_ind", "va_constr", "va_trade", "va_ict", "va_fin",
    "va_realest", "va_prof", "va_public", "va_other")

#: The seven service activities ``va_services`` is the sum of.
_VA_SERVICES: tuple[str, ...] = (
    "va_trade", "va_ict", "va_fin", "va_realest", "va_prof", "va_public", "va_other")

#: Gross fixed capital formation by asset, from ``DF_TABLE1`` (table 0102).
#: ``inv_transp``, ``inv_ict`` and ``inv_othmach`` are the three parts of
#: ``inv_equip``; the additive split of ``inv`` is dwellings, other
#: structures, equipment, cultivated biological resources and IPP.
ANA_ASSETS: dict[str, tuple[dict[str, str], str]] = {
    "inv_dwell":   ({"TRANSACTION": "P51G", "INSTR_ASSET": "N111G"}, "Dwellings"),
    "inv_struct":  ({"TRANSACTION": "P51G", "INSTR_ASSET": "N112G"},
                    "Other buildings and structures"),
    "inv_equip":   ({"TRANSACTION": "P51G", "INSTR_ASSET": "N11MG"},
                    "Machinery and equipment and weapons systems"),
    "inv_transp":  ({"TRANSACTION": "P51G", "INSTR_ASSET": "N1131G"},
                    "Transport equipment (memo: inside inv_equip)"),
    "inv_ict":     ({"TRANSACTION": "P51G", "INSTR_ASSET": "N1132G"},
                    "ICT equipment (memo: inside inv_equip)"),
    "inv_othmach": ({"TRANSACTION": "P51G", "INSTR_ASSET": "N11OG"},
                    "Other machinery and equipment (memo: inside inv_equip)"),
    "inv_bio":     ({"TRANSACTION": "P51G", "INSTR_ASSET": "N115G"},
                    "Cultivated biological resources"),
    "inv_ipp":     ({"TRANSACTION": "P51G", "INSTR_ASSET": "N117G"},
                    "Intellectual property products"),
}

#: Household consumption by durability, households only (``S14``; the
#: headline ``cons_hh`` is ``S1M``, households and NPISH). Two tables publish
#: it: ``DF_TABLE5A_T117`` (COICOP 2018, current everywhere it exists) and the
#: older ``DF_TABLE5_T117`` (COICOP 1999, which the EU members stopped updating
#: in 2022-23). Each country is served whole by one table, 5A first, and
#: ``attrs["meta"]`` records which. Chile, Indonesia and New Zealand publish no
#: semi-durables.
ANA_DURABILITY: dict[str, tuple[dict[str, str], str]] = {
    "cons_dur":     ({"TRANSACTION": "P311"}, "Durable goods"),
    "cons_semidur": ({"TRANSACTION": "P312"}, "Semi-durable goods"),
    "cons_nondur":  ({"TRANSACTION": "P313"}, "Non-durable goods"),
    "cons_serv":    ({"TRANSACTION": "P314"}, "Services"),
}

#: Gross fixed capital formation by institutional sector, from
#: ``DSD_NASEC10@DF_TABLE14_GFCF``, current prices only: the OECD publishes a
#: volume for four countries at most, so these carry no deflator; deflate
#: public investment with ``inv_defl`` (or ``inv_struct_defl``) and say so.
#: ``inv_hh`` is households (``S14``) where published and households plus
#: NPISH (``S1M``) otherwise, per country; ``attrs["meta"]`` says which.
#: Switzerland and Norway are not in the table and Turkey has no ``S13``.
ANA_SECTORS: dict[str, tuple[dict[str, str], str]] = {
    "inv_gov":  ({"SECTOR": "S13"}, "GFCF of general government (public investment)"),
    "inv_corp": ({"SECTOR": "S11"}, "GFCF of non-financial corporations"),
    "inv_fin":  ({"SECTOR": "S12"}, "GFCF of financial corporations"),
    "inv_hh":   ({"SECTOR": "S14"}, "GFCF of households (S14, else S14+S15)"),
    "_inv_hh_s1m": ({"SECTOR": "S1M"}, "GFCF of households and NPISH (fallback)"),
}

#: Closing stocks of fixed assets, total economy, from ``DF_TABLE9A`` (net and
#: gross, current prices and chain-linked volumes, 36 countries), and the
#: general government's net stock from the balance sheets ``DSD_NASEC10@
#: DF_TABLE9B`` (current prices only, 34 countries) -- the anchor a perpetual
#: inventory of public capital needs.
ANA_STOCKS: dict[str, tuple[dict[str, str], str]] = {
    "k_net":   ({"TRANSACTION": "LE", "INSTR_ASSET": "N11N", "ACTIVITY": "_T"},
                "Net fixed capital stock, end of year"),
    "k_gross": ({"TRANSACTION": "LE", "INSTR_ASSET": "N11G", "ACTIVITY": "_T"},
                "Gross fixed capital stock, end of year"),
}
ANA_STOCKS_GOV: dict[str, tuple[dict[str, str], str]] = {
    "k_net_gov": ({"SECTOR": "S13", "INSTR_ASSET": "N11N"},
                  "Net fixed capital stock of general government, end of year"),
}

#: Labour input, domestic concept, from ``DF_TABLE3_EMPDC``: persons in
#: thousands (``PS``) and hours in millions (``H``), for the whole economy and,
#: with suffixes from :data:`ANA_LABOR_ACTIVITIES`, for agriculture and for
#: public administration, education and health (the two blocks a market-sector
#: aggregate removes, as in :func:`puremacro.fetch.ana_by_activity`).
ANA_LABOR: dict[str, tuple[dict[str, str], str]] = {
    "emp":             ({"TRANSACTION": "EMP", "UNIT_MEASURE": "PS"},
                        "Total employment, persons (thousands)"),
    "emp_employees":   ({"TRANSACTION": "SAL", "UNIT_MEASURE": "PS"},
                        "Employees, persons (thousands)"),
    "emp_selfemp":     ({"TRANSACTION": "SELF", "UNIT_MEASURE": "PS"},
                        "Self-employed, persons (thousands)"),
    "hours":           ({"TRANSACTION": "EMP", "UNIT_MEASURE": "H"},
                        "Hours worked, all employed (millions)"),
    "hours_employees": ({"TRANSACTION": "SAL", "UNIT_MEASURE": "H"},
                        "Hours worked by employees (millions)"),
    "hours_selfemp":   ({"TRANSACTION": "SELF", "UNIT_MEASURE": "H"},
                        "Hours worked by the self-employed (millions)"),
}

#: ISIC activity -> column suffix for the labour block; ``_T`` keeps the plain
#: names. Same suffixes as ``va_agri`` and ``va_public``.
ANA_LABOR_ACTIVITIES: dict[str, str] = {"_T": "", "A": "_agri", "OTQ": "_public"}

#: Population, national concept, from ``DF_TABLE3_POP_EMPNC`` (thousands).
ANA_POP: dict[str, tuple[dict[str, str], str]] = {
    "pop": ({"TRANSACTION": "POP", "UNIT_MEASURE": "PS"}, "Population (thousands)"),
}

#: Power of ten each unit is put on: money in millions, persons in thousands,
#: hours in millions (the OECD publishes ``UNIT_MULT`` 6, 3 and 6 today).
_UNIT_TARGET: dict[str, int] = {"XDC": 6, "PS": 3, "H": 6}

#: Annual hours per worker outside which a reference area's hours are taken to
#: be on the wrong time base (New Zealand publishes hours per week), and the
#: band a correction has to land in -- the bands of ``ana_by_activity``.
_HOURS_IMPLAUSIBLE = (600.0, 4000.0)
_HOURS_PLAUSIBLE = (1200.0, 2600.0)
_HOURS_SCALES: dict[float, str] = {52.0: "weekly", 4.0: "quarterly"}

#: Preference among the tables of ``DF_TABLE1`` when one series is published
#: in several (GDP is in all three): expenditure, then output, then income.
_TABLE_RANK: dict[str, int] = {"T0102": 0, "T0101": 1, "T0103": 2}

#: Fallback for :func:`ana_countries` when the availability endpoint cannot be
#: reached: the 64 countries with annual GDP at current prices in
#: ``DF_TABLE1_EXPENDITURE`` as probed on 6-Oct-2026.
_ANA_COUNTRIES_FALLBACK: tuple[str, ...] = (
    "ALB", "ARG", "AUS", "AUT", "BEL", "BGR", "BRA", "CAN", "CHE", "CHL",
    "CHN", "CMR", "COL", "CPV", "CRI", "CYP", "CZE", "DEU", "DNK", "ESP",
    "EST", "FIN", "FRA", "GBR", "GEO", "GRC", "HKG", "HRV", "HUN", "IDN",
    "IND", "IRL", "ISL", "ISR", "ITA", "JPN", "KAZ", "KOR", "LTU", "LUX",
    "LVA", "MAR", "MDG", "MEX", "MKD", "MLT", "NLD", "NOR", "NZL", "POL",
    "PRT", "ROU", "RUS", "SAU", "SEN", "SGP", "SRB", "SVK", "SVN", "SWE",
    "TUR", "USA", "ZAF", "ZMB")

#: Column names with no price dimension, or no volume published: no deflator
#: and no ``_real`` column for any of them.
_NO_DEFLATOR: frozenset[str] = (
    frozenset(ANA_INCOME) | frozenset(ANA_INCOME_SECTOR) | frozenset(ANA_SECTORS)
    | frozenset(ANA_STOCKS_GOV) | frozenset(ANA_POP)
    | frozenset(f"{n}{s}" for n in ANA_LABOR for s in ANA_LABOR_ACTIVITIES.values())
    | {"discrepancy_exp", "discrepancy_out", "gdp_output"})

_REF_DATE = pd.Timestamp("2020-01-01")
_CHUNK = 10

#: Seconds between requests to ``sdmx.oecd.org`` (passed to ``oecd_csv``).
#: Module-level so a caller sharing the endpoint with others can slow down.
_PAUSE: float = 5.0

_META_KEYS = ("code", "currency", "units", "unit_mult", "sa", "first", "last", "n",
              "volume_base", "volume_ref_year", "volume_from_l", "tables_not_t0102",
              "durability_table",
              "inv_hh_sector", "hours_scale", "absent")


# ---------------------------------------------------------------------------
# HTTP seams (tests patch these two names)
# ---------------------------------------------------------------------------

def _csv(agency_flow: str, key: str, *, start: str, refresh: bool) -> pd.DataFrame:
    """One OECD data request; an empty frame with ``attrs["status"]`` on failure.

    Guarded here as well as in ``oecd_csv``: a truncated transfer raises
    ``http.client.IncompleteRead`` (not an ``OSError``) and a truncated gzip
    raises ``EOFError``, and neither may escape :func:`ana_panel`.
    """
    try:
        return oecd_csv(agency_flow, key, start_period=start, refresh=refresh,
                        pause=_PAUSE)
    except Exception as exc:  # noqa: BLE001 - the never-raise contract
        out = pd.DataFrame()
        out.attrs["status"] = type(exc).__name__
        return out


def _availability(agency_flow: str, key: str, *, refresh: bool) -> dict:
    """The ``availableconstraint`` JSON for a key; ``{}`` on failure."""
    try:
        return oecd_availability(agency_flow, key, refresh=refresh)
    except Exception:  # noqa: BLE001 - the never-raise contract
        return {}


# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------

def ana_countries(*, refresh: bool = False) -> list[str]:
    """Every country with annual GDP at current prices in the OECD annual accounts.

    Asks the SDMX availability endpoint which reference areas carry
    ``B1GQ`` in national currency at current prices in
    ``DSD_NAMAIN10@DF_TABLE1_EXPENDITURE``, and drops the aggregates (euro
    area, EU, OECD, West Germany ``DEU_F``) by the explicit list
    :data:`puremacro.fetch._oecd_sdmx.OECD_AGGREGATES`.

    Parameters
    ----------
    refresh
        Re-query instead of reading the on-disk cache.

    Returns
    -------
    list of str
        Sorted ISO3 codes (64 as of October 2026). If the endpoint cannot be
        reached, the frozen list of 6-Oct-2026 is returned with a warning, so
        this never raises and never returns empty.

    Examples
    --------
    >>> codes = ana_countries()            # doctest: +SKIP
    >>> panel = ana_panel(codes)           # doctest: +SKIP
    """
    key = oecd_key(NAMAIN10_DIMS, FREQ="A", SECTOR="S1", TRANSACTION="B1GQ",
                   UNIT_MEASURE="XDC", PRICE_BASE="V", TRANSFORMATION="N")
    codes: list[str] = []
    try:
        payload = _availability(_COUNTRIES_FLOW, key, refresh=refresh)
        regions = payload["data"]["contentConstraints"][0]["cubeRegions"][0]
        for kv in regions["keyValues"]:
            if kv["id"] == "REF_AREA":
                codes = [str(v) for v in kv["values"]]
    except (KeyError, IndexError, TypeError):
        codes = []
    if not codes:
        warnings.warn("ana_countries: the OECD availability endpoint did not "
                      "answer; returning the frozen list of 6-Oct-2026",
                      stacklevel=2)
        codes = list(_ANA_COUNTRIES_FALLBACK)
    return sorted({c for c in codes if c not in OECD_AGGREGATES})


# ---------------------------------------------------------------------------
# Request bookkeeping
# ---------------------------------------------------------------------------

#: Statuses that mean "the OECD publishes nothing for that key": recorded in
#: ``attrs["chunks_empty"]``, not in ``attrs["missing"]``, because a re-run
#: cannot change them.
_EMPTY_STATUSES: frozenset[str] = frozenset({"HTTP 404", "no rows"})


class _Requests:
    """Sends the requests of one panel build and remembers how each one went.

    ``log`` has one dict per request sent (flow, key, codes, status, rows);
    ``missing`` the failures (rate limit, timeout, transport error, a body
    that is not SDMX-CSV); ``empty`` the requests the OECD answered with
    nothing (HTTP 404 or a header-only CSV).
    """

    def __init__(self, start: str, refresh: bool) -> None:
        self.start = start
        self.refresh = refresh
        self.missing: list[dict] = []
        self.empty: list[dict] = []
        self.log: list[dict] = []
        self.flows: list[str] = []
        self.last_status: str = ""

    def get(self, flow: str, dims: Sequence[str], chunk: Sequence[str],
            block: str, **pins: str) -> pd.DataFrame:
        agency_flow = ANA_FLOWS[flow]
        key = oecd_key(dims, FREQ="A", REF_AREA="+".join(chunk), **pins)
        if agency_flow not in self.flows:
            self.flows.append(agency_flow)
        raw = _csv(agency_flow, key, start=self.start, refresh=self.refresh)
        status = str(raw.attrs.get("status", "ok" if not raw.empty else "empty response"))
        if status == "ok":
            if raw.empty:
                status = "no rows"
            elif "OBS_VALUE" not in raw.columns:
                status = "unexpected response (no OBS_VALUE)"
        entry = {"what": "request", "block": block, "flow": agency_flow,
                 "key": key, "codes": tuple(chunk), "status": status}
        self.log.append({**entry, "rows": int(len(raw)) if status == "ok" else 0})
        self.last_status = status
        if status == "ok":
            return raw
        (self.empty if status in _EMPTY_STATUSES else self.missing).append(entry)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Tidying
# ---------------------------------------------------------------------------

def _numeric(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed rows: ``code``, ``date`` (1 January), ``value`` on the target scale."""
    d = raw.rename(columns={"REF_AREA": "code"}).copy()
    d["code"] = d["code"].astype(str)
    d = d[~d["code"].isin(OECD_AGGREGATES)]
    year = pd.to_numeric(d["TIME_PERIOD"].astype(str).str[:4], errors="coerce")
    value = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
    d = d.assign(year=year, value=value).dropna(subset=["year", "value"])
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["year"].astype(int).astype(str) + "-01-01")
    unit = d["UNIT_MEASURE"].astype(str) if "UNIT_MEASURE" in d else pd.Series("XDC", index=d.index)
    target = unit.map(_UNIT_TARGET).astype(float)
    mult = (pd.to_numeric(d["UNIT_MULT"], errors="coerce") if "UNIT_MULT" in d
            else pd.Series(np.nan, index=d.index))
    # A blank multiplier means "already on the target scale"; a unit with no
    # target is left as published. Either way the factor is 1, not NaN.
    target, mult = target.fillna(mult), mult.fillna(target)
    d["value"] = d["value"] * np.power(10.0, (mult - target).fillna(0.0))
    return d


def _select(raw: pd.DataFrame, registry: dict[str, tuple[dict[str, str], str]], *,
            units: tuple[str, ...] = ("XDC",),
            price_bases: tuple[str, ...] = ("V", "L", "LR"),
            suffix_by: tuple[str, dict[str, str]] | None = None) -> pd.DataFrame:
    """Rows of ``raw`` matched to registry names, one per (code, date, name, PRICE_BASE).

    A registry entry matches every row whose dimensions equal its pins; a
    dimension it does not pin is free. One row may serve two names (GDP is
    both ``gdp`` and ``gdp_output``), which is why this loops over entries
    rather than mapping rows. ``suffix_by=(dim, {code: suffix})`` multiplies
    every entry across the values of one more dimension (the labour block's
    activities).
    """
    if raw.empty:
        return pd.DataFrame()
    d = _numeric(raw)
    if d.empty:
        return pd.DataFrame()
    if "UNIT_MEASURE" in d:
        d = d[d["UNIT_MEASURE"].astype(str).isin(units)]
    if "PRICE_BASE" in d and price_bases:
        d = d[d["PRICE_BASE"].astype(str).isin(price_bases)]
    if "TRANSFORMATION" in d:
        t = d["TRANSFORMATION"]
        d = d[t.isna() | t.astype(str).isin(["N", "_Z"])]
    if d.empty:
        return pd.DataFrame()
    pinned = {dim for pins, _ in registry.values() for dim in pins}
    if suffix_by is not None:
        pinned.add(suffix_by[0])
    d = d.assign(**{dim: d[dim].astype(str) for dim in pinned if dim in d.columns})
    variants = [("", None)] if suffix_by is None else [
        (sfx, (suffix_by[0], val)) for val, sfx in suffix_by[1].items()]
    parts = []
    for name, (pins, _) in registry.items():
        for sfx, extra in variants:
            mask = pd.Series(True, index=d.index)
            for dim, val in pins.items():
                if dim not in d.columns:
                    mask &= False
                    break
                mask &= d[dim] == val
            if extra is not None:
                mask &= d[extra[0]] == extra[1]
            if mask.any():
                parts.append(d[mask].assign(name=name + sfx))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "PRICE_BASE" not in out:
        out["PRICE_BASE"] = "V"
    out["PRICE_BASE"] = out["PRICE_BASE"].astype(str).replace({"_Z": "V"})
    rank = (out["TABLE_IDENTIFIER"].astype(str).map(_TABLE_RANK).fillna(9)
            if "TABLE_IDENTIFIER" in out else pd.Series(0.0, index=out.index))
    if "EXPENDITURE" in out:
        # A total (`_T`) or not-applicable (`_Z`) breakdown before any detail.
        rank = rank + 10.0 * ~out["EXPENDITURE"].astype(str).isin(["_T", "_Z"])
    # One table per series: the best-ranked table with any row for that
    # (code, name, PRICE_BASE) serves every year, so the output and the
    # expenditure approaches are never spliced year by year.
    series = ["code", "name", "PRICE_BASE"]
    out = out.assign(_rank=rank)
    out = out[out["_rank"] == out.groupby(series)["_rank"].transform("min")]
    out = (out.sort_values("_rank", kind="stable")
              .drop_duplicates(subset=["code", "date", "name", "PRICE_BASE"], keep="first"))
    out["TABLE"] = (out["TABLE_IDENTIFIER"].astype(str) if "TABLE_IDENTIFIER" in out
                    else "")
    keep = ["code", "date", "name", "PRICE_BASE", "value", "TABLE"]
    if "CURRENCY" in out:
        keep.append("CURRENCY")
    return out[keep].reset_index(drop=True)


def _wide(tidy: pd.DataFrame, price_base: str) -> pd.DataFrame:
    part = tidy[tidy["PRICE_BASE"] == price_base]
    if part.empty:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["code", "date"]))
    w = part.pivot_table(index=["code", "date"], columns="name", values="value",
                         aggfunc="first")
    w.columns.name = None
    return w


def _volumes(nominal: pd.DataFrame, lr: pd.DataFrame, l_: pd.DataFrame
             ) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Per (country, series): ``LR`` where published, else ``L`` rebased to 2020.

    Returns the volume frame and, per country, which series came from ``L``
    and whether they could be rebased (``{code: {name: "L->2020" | "L"}}``).
    Rebasing a chain-linked ``L`` by ``V / L`` in 2020 is what the OECD does
    to make ``LR``, so the two kinds of column are on the same footing.
    """
    how: dict[str, dict] = {}
    if l_.empty:
        return lr, how
    have: set[tuple[str, str]] = set()
    if not lr.empty:
        any_lr = lr.notna().groupby(level="code").any()
        have = {(c, n) for c in any_lr.index for n in any_lr.columns if any_lr.loc[c, n]}
    extra: dict[str, list[pd.Series]] = {}
    for name in l_.columns:
        for code, s in l_[name].dropna().groupby(level="code"):
            if s.empty or (code, name) in have:
                continue
            v20 = (nominal[name].get((code, _REF_DATE), np.nan)
                   if name in nominal.columns else np.nan)
            l20 = s.get((code, _REF_DATE), np.nan)
            if np.isfinite(v20) and np.isfinite(l20) and l20 > 0 and v20 > 0:
                s = s * (v20 / l20)
                how.setdefault(code, {})[name] = "L->2020"
            else:
                how.setdefault(code, {})[name] = "L"
            extra.setdefault(name, []).append(s)
    if not extra:
        return lr, how
    add = pd.DataFrame({n: pd.concat(v) for n, v in extra.items()})
    vol = add if lr.empty else lr.combine_first(add)
    vol.index = vol.index.set_names(["code", "date"])
    return vol, how


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def _table1(req: _Requests, chunks: list[list[str]], *, assets: bool,
            income: bool, output: bool) -> pd.DataFrame:
    """Main aggregates (+ assets, income, output) from ``DF_TABLE1``."""
    trans = ["B1GQ", "P3", "P51G", "P5", "P6", "P7", "YA0"]
    instr = ["_Z", "N11G", "N1G"]
    acts = ["_Z", "_T"]
    registry = dict(ANA_MAIN)
    if income:
        trans += ["D1", "B2A3G", "D2", "D3", "D2X3", "YA2"]
        registry.update(ANA_INCOME)
    if output:
        trans += ["B1G", "D21X31", "YA1"]
        acts += [p["ACTIVITY"] for p, _ in ANA_OUTPUT.values()
                 if "ACTIVITY" in p and p["ACTIVITY"] != "_T"]
        registry.update(ANA_OUTPUT)
    if assets:
        instr += [p["INSTR_ASSET"] for p, _ in ANA_ASSETS.values()]
        registry.update(ANA_ASSETS)
    parts = []
    for ck in chunks:
        # 12 dims: FREQ.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.
        # INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.PRICE_BASE.
        # TRANSFORMATION.TABLE_IDENTIFIER. SECTOR is left open on purpose: the
        # discrepancy YA0 is not booked to S1 everywhere.
        raw = req.get("table1", NAMAIN10_DIMS, ck, "main",
                      TRANSACTION=trans, INSTR_ASSET=instr, ACTIVITY=acts,
                      UNIT_MEASURE="XDC", PRICE_BASE="V+L+LR", TRANSFORMATION="N")
        parts.append(_select(raw, registry))
    return _concat(parts)


def _durability(req: _Requests, chunks: list[list[str]]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Durability split: ``TABLE5A_T117`` first, ``TABLE5_T117`` for the rest."""
    parts, served = [], {}
    pins = dict(SECTOR="S14", TRANSACTION="P311+P312+P313+P314",
                UNIT_MEASURE="XDC", PRICE_BASE="V+L+LR", TRANSFORMATION="N")
    for ck in chunks:
        t5a = _select(req.get("table5a", NAMAIN10_DIMS, ck, "durability", **pins),
                      ANA_DURABILITY)
        if req.last_status not in {"ok"} | _EMPTY_STATUSES:
            # 5A failed (rate limit, timeout...): sending its countries to the
            # stale old-COICOP TABLE5 would freeze the wrong source; leave the
            # chunk absent so a re-run fills it from 5A.
            continue
        got = set(t5a.loc[t5a["PRICE_BASE"] == "V", "code"]) if not t5a.empty else set()
        if got:
            parts.append(t5a[t5a["code"].isin(got)])
            served.update({c: "TABLE5A_T117" for c in got})
        rest = [c for c in ck if c not in got]
        if rest:
            t5 = _select(req.get("table5", NAMAIN10_DIMS, rest, "durability", **pins),
                         ANA_DURABILITY)
            if not t5.empty:
                parts.append(t5)
                served.update({c: "TABLE5_T117" for c in
                               set(t5.loc[t5["PRICE_BASE"] == "V", "code"])})
    return _concat(parts), served


def _sectors(req: _Requests, chunks: list[list[str]]) -> tuple[pd.DataFrame, dict[str, str]]:
    """GFCF by institutional sector from ``DSD_NASEC10@DF_TABLE14_GFCF``."""
    parts = []
    for ck in chunks:
        # 13 dims: FREQ.REF_AREA.SECTOR.COUNTERPART_SECTOR.ACCOUNTING_ENTRY.
        # TRANSACTION.INSTR_ASSET.EXPENDITURE.UNIT_MEASURE.VALUATION.
        # PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
        raw = req.get("table14_gfcf", NASEC10_DIMS, ck, "sectors",
                      SECTOR="S11+S12+S13+S14+S1M", ACCOUNTING_ENTRY="D",
                      TRANSACTION="P51G", INSTR_ASSET="N11G", UNIT_MEASURE="XDC",
                      PRICE_BASE="V")
        parts.append(_select(raw, ANA_SECTORS, price_bases=("V",)))
    tidy = _concat(parts)
    which: dict[str, str] = {}
    if tidy.empty:
        return tidy, which
    has_s14 = set(tidy.loc[tidy["name"] == "inv_hh", "code"])
    fallback = tidy[(tidy["name"] == "_inv_hh_s1m") & ~tidy["code"].isin(has_s14)]
    fallback = fallback.assign(name="inv_hh")
    which.update({c: "S14" for c in has_s14})
    which.update({c: "S1M" for c in set(fallback["code"])})
    tidy = pd.concat([tidy[tidy["name"] != "_inv_hh_s1m"], fallback], ignore_index=True)
    return tidy, which


def _income_sector(req: _Requests, chunks: list[list[str]]) -> pd.DataFrame:
    """B2G, B3G, P51C and D211 of the total economy from ``DF_TABLE14``."""
    parts = []
    for ck in chunks:
        raw = req.get("table14", NASEC10_DIMS, ck, "income",
                      SECTOR="S1", ACCOUNTING_ENTRY="C+D",
                      TRANSACTION="B2G+B3G+P51C+D211", UNIT_MEASURE="XDC",
                      PRICE_BASE="V")
        parts.append(_select(raw, ANA_INCOME_SECTOR, price_bases=("V",)))
    return _concat(parts)


def _stocks(req: _Requests, chunks: list[list[str]]) -> pd.DataFrame:
    parts = []
    for ck in chunks:
        raw = req.get("table9a", NAMAIN10_DIMS, ck, "stocks",
                      TRANSACTION="LE", INSTR_ASSET="N11N+N11G", ACTIVITY="_T",
                      UNIT_MEASURE="XDC", PRICE_BASE="V+L")
        parts.append(_select(raw, ANA_STOCKS))
        raw = req.get("table9b", NASEC10_DIMS, ck, "stocks",
                      SECTOR="S13", INSTR_ASSET="N11N", UNIT_MEASURE="XDC",
                      PRICE_BASE="V")
        parts.append(_select(raw, ANA_STOCKS_GOV, price_bases=("V",)))
    return _concat(parts)


def _labor(req: _Requests, chunks: list[list[str]]) -> tuple[pd.DataFrame, dict[str, float]]:
    parts = []
    for ck in chunks:
        raw = req.get("empdc", NAMAIN10_DIMS, ck, "labor",
                      TRANSACTION="EMP+SAL+SELF", ACTIVITY="+".join(ANA_LABOR_ACTIVITIES),
                      UNIT_MEASURE="PS+H")
        parts.append(_select(raw, ANA_LABOR, units=("PS", "H"), price_bases=(),
                             suffix_by=("ACTIVITY", ANA_LABOR_ACTIVITIES)))
        raw = req.get("pop", NAMAIN10_DIMS, ck, "labor",
                      TRANSACTION="POP", UNIT_MEASURE="PS")
        parts.append(_select(raw, ANA_POP, units=("PS",), price_bases=()))
    tidy = _concat(parts)
    if tidy.empty:
        return tidy, {}
    tidy["PRICE_BASE"] = "V"
    heads = tidy[tidy["name"] == "emp"].set_index(["code", "date"])["value"]
    hours = tidy[tidy["name"] == "hours"].set_index(["code", "date"])["value"]
    scales = hours_scale_factors(heads, hours, implausible=_HOURS_IMPLAUSIBLE,
                                 plausible=_HOURS_PLAUSIBLE, scales=_HOURS_SCALES)
    if scales:
        # Every hours column of that country, the branches included, or
        # `hours - hours_agri` would subtract two different time bases.
        is_h = tidy["name"].str.startswith("hours")
        tidy.loc[is_h, "value"] = (tidy.loc[is_h, "value"]
                                   * tidy.loc[is_h, "code"].map(scales).fillna(1.0))
    return tidy, scales


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _check_codes(codes) -> list[str] | None:
    if codes is None:
        return None
    if isinstance(codes, str):
        codes = [codes]
    out = []
    for c in codes:
        c = str(c).strip().upper()
        if not c:
            continue
        if c in OECD_AGGREGATES:
            raise ValueError(f"{c!r} is an aggregate, not a country; ana_panel takes "
                             f"ISO3 country codes (see ana_countries())")
        if not re.fullmatch(r"[A-Z]{3}", c):
            raise ValueError(f"{c!r} is not an ISO3 country code; valid codes are "
                             f"those returned by ana_countries(), e.g. "
                             f"{', '.join(_ANA_COUNTRIES_FALLBACK[:6])}, ...")
        if c not in out:
            out.append(c)
    if not out:
        raise ValueError("codes is empty; pass None for every country or ISO3 "
                         "codes from ana_countries()")
    return out


def _check_start(start) -> str:
    s = str(start).strip()
    if not re.fullmatch(r"\d{4}", s) or not 1900 <= int(s) <= 2100:
        raise ValueError(f"start must be a year such as '1950', got {start!r}")
    return s


def _variables_registry() -> dict[str, tuple[str, str, str]]:
    """name -> (flow short name, units, description) for every column."""
    money = "millions of national currency, current prices"
    out: dict[str, tuple[str, str, str]] = {}
    for reg, flow in ((ANA_MAIN, "table1"), (ANA_INCOME, "table1"),
                      (ANA_OUTPUT, "table1"), (ANA_ASSETS, "table1"),
                      (ANA_INCOME_SECTOR, "table14"), (ANA_SECTORS, "table14_gfcf"),
                      (ANA_STOCKS, "table9a"), (ANA_STOCKS_GOV, "table9b")):
        for name, (_, desc) in reg.items():
            if not name.startswith("_"):
                out[name] = (flow, money, desc)
    out["va_services"] = ("table1", money, "All services, G-U (built here: sum of seven)")
    for name, (_, desc) in ANA_DURABILITY.items():
        out[name] = ("table5a|table5", money, desc)
    for name, (pins, desc) in ANA_LABOR.items():
        unit = "thousands of persons" if pins["UNIT_MEASURE"] == "PS" else "millions of hours"
        for act, sfx in ANA_LABOR_ACTIVITIES.items():
            out[name + sfx] = ("empdc", unit, f"{desc}, ISIC {act}")
    out["pop"] = ("pop", "thousands of persons", ANA_POP["pop"][1])
    return out


def _ordered_names(*, assets, durability, income, output, labor, sectors, stocks) -> list[str]:
    names = list(ANA_MAIN)
    if income:
        names += list(ANA_INCOME) + list(ANA_INCOME_SECTOR)
    if output:
        names += [n for n in ANA_OUTPUT if n not in ("taxes_prod", "discrepancy_out")]
        names += ["va_services", "taxes_prod", "discrepancy_out"]
    if durability:
        names += list(ANA_DURABILITY)
    if assets:
        names += list(ANA_ASSETS)
    if sectors:
        names += [n for n in ANA_SECTORS if not n.startswith("_")]
    if stocks:
        names += list(ANA_STOCKS) + list(ANA_STOCKS_GOV)
    if labor:
        names += [n + s for s in ANA_LABOR_ACTIVITIES.values() for n in ANA_LABOR]
        names += ["pop"]
    return names


def _empty(long: bool, *, missing=(), chunks_empty=(), requests=(), source: str = "",
           fetched_at: str = "") -> pd.DataFrame:
    if long:
        out = pd.DataFrame({"code": pd.Series(dtype=object),
                            "date": pd.Series(dtype="datetime64[ns]"),
                            "variable": pd.Series(dtype=object),
                            "value": pd.Series(dtype=float)})
    else:
        idx = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype=object), pd.DatetimeIndex([])], names=["code", "date"])
        out = pd.DataFrame(index=idx)
    out.attrs.update(meta=(), variables=(), missing=tuple(missing), corrections=(),
                     chunks_empty=tuple(chunks_empty), requests=tuple(requests),
                     source=source or " + ".join(ANA_FLOWS.values()),
                     fetched_at=fetched_at or _now())
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ana_panel(codes: Iterable[str] | None = None, *,
              start: str = "1950",
              assets: bool = False,
              durability: bool = False,
              income: bool = False,
              output: bool = False,
              labor: bool = False,
              sectors: bool = False,
              stocks: bool = False,
              real: bool = True,
              long: bool = False,
              refresh: bool = False) -> pd.DataFrame:
    r"""Annual national accounts panel: current prices, deflators and volumes.

    Parameters
    ----------
    codes
        ISO3 country codes, e.g. ``["USA", "FRA", "MEX"]``. ``None`` (default)
        takes every country :func:`ana_countries` lists, 64 today, in
        requests of ten. Aggregates (``OECD``, ``EA20``, ...) are refused.
    start
        First year requested, as ``"1950"``. The OECD starts France, the United
        Kingdom, Sweden and India in 1950 and most of the OECD in 1970.
    assets
        Join gross fixed capital formation by asset (:data:`ANA_ASSETS`):
        ``inv_dwell``, ``inv_struct``, ``inv_equip`` (with its parts
        ``inv_transp``, ``inv_ict``, ``inv_othmach``), ``inv_bio``,
        ``inv_ipp``, each with a deflator. Widens the main request.
    durability
        Join household consumption by durability (:data:`ANA_DURABILITY`),
        households only (``S14``). Up to two extra requests per chunk.
    income
        Join the income approach (:data:`ANA_INCOME`) and, from the sector
        accounts, gross operating surplus and mixed income separately plus
        consumption of fixed capital and VAT (:data:`ANA_INCOME_SECTOR`).
        Current prices only. One extra request per chunk.
    output
        Join value added by A10 activity, taxes less subsidies on products and
        the output discrepancy (:data:`ANA_OUTPUT`), with deflators. Widens
        the main request.
    labor
        Join employment, employees, self-employed and hours, for the whole
        economy and for agriculture (``_agri``) and public administration,
        education and health (``_public``) (:data:`ANA_LABOR`), and
        population (``pop``). Persons in thousands, hours in millions; New
        Zealand's hours, published per week, are put on an annual basis.
        Two extra requests per chunk.
    sectors
        Join gross fixed capital formation by institutional sector
        (:data:`ANA_SECTORS`): ``inv_gov`` (public investment), ``inv_corp``,
        ``inv_fin``, ``inv_hh``. Current prices only. One extra request.
    stocks
        Join the closing net and gross fixed capital stocks ``k_net`` and
        ``k_gross`` (with volumes) and the government's net stock
        ``k_net_gov`` (:data:`ANA_STOCKS`). Two extra requests per chunk.
    real
        Keep the ``<x>_real`` volume columns (default ``True``): chain-linked
        volumes at 2020 prices, ``LR`` where published and ``L`` rebased to
        2020 otherwise. ``False`` keeps current prices and deflators only.
    long
        Return ``code, date, variable, value`` rows instead of the wide frame.
    refresh
        Re-download instead of reading the on-disk cache.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(code, date)``, ``date`` at 1 January of the year. Money
        columns are millions of national currency at current prices;
        ``<x>_defl`` are implicit deflators ``100 * V / LR`` (2020 = 100);
        ``<x>_real`` are chain-linked volumes in millions of national currency
        at 2020 prices. ``attrs``: ``meta`` (one dict per country: currency,
        first and last year, volume base, durability table, ...; read it with
        :func:`ana_panel_meta`), ``variables`` (one dict per column: flow,
        units, description and the SDMX ``keys`` that served it), ``source``,
        ``fetched_at`` (UTC), ``missing`` (every request that failed, with its
        status, and every requested country with no main aggregates: what a
        re-run may fix), ``chunks_empty`` (requests the OECD answered with
        nothing, ``HTTP 404`` or no rows) and ``requests`` (every request
        sent: block, flow, key, codes, status, rows). On a provider failure of
        the main request the frame is empty, with the same index names and
        ``attrs`` keys, and a warning is issued; nothing is raised.

    Raises
    ------
    ValueError
        On caller errors only: a code that is not ISO3, an aggregate, or a
        ``start`` that is not a year.

    Examples
    --------
    >>> p = ana_panel(["USA", "FRA", "MEX"], income=True)   # doctest: +SKIP
    >>> p.loc["FRA", ["gdp", "gdp_defl", "mixed_income"]].head()  # doctest: +SKIP
    """
    codes_list = _check_codes(codes)
    start = _check_start(start)
    fetched_at = _now()
    if codes_list is None:
        codes_list = ana_countries(refresh=refresh)
    chunks = [codes_list[i:i + _CHUNK] for i in range(0, len(codes_list), _CHUNK)]
    req = _Requests(start, refresh)

    tidy = _table1(req, chunks, assets=assets, income=income, output=output)
    if tidy.empty or not (tidy["name"] == "gdp").any():
        statuses = (sorted({m["status"] for m in req.missing + req.empty})
                    or ["no GDP rows"])
        warnings.warn(f"ana_panel: no main aggregates obtained from the OECD "
                      f"({', '.join(statuses)}); returning an empty frame, see "
                      f"attrs['missing']", stacklevel=2)
        missing = list(req.missing) + [
            {"what": "country", "code": c, "status": "no GDP at current prices in DF_TABLE1"}
            for c in codes_list]
        return _empty(long, missing=missing, chunks_empty=req.empty, requests=req.log,
                      source=" + ".join(req.flows), fetched_at=fetched_at)

    blocks = [tidy]
    durability_table: dict[str, str] = {}
    inv_hh_sector: dict[str, str] = {}
    hours_scales: dict[str, float] = {}
    if income:
        blocks.append(_income_sector(req, chunks))
    if durability:
        t, durability_table = _durability(req, chunks)
        blocks.append(t)
    if sectors:
        t, inv_hh_sector = _sectors(req, chunks)
        blocks.append(t)
    if stocks:
        blocks.append(_stocks(req, chunks))
    if labor:
        t, hours_scales = _labor(req, chunks)
        blocks.append(t)
    tidy = _concat(blocks)
    tidy = tidy[tidy["code"].isin(codes_list)]

    nominal = _wide(tidy, "V")
    lr, l_ = _wide(tidy, "LR"), _wide(tidy, "L")
    vol, vol_how = _volumes(nominal, lr, l_)
    vol = vol[[c for c in vol.columns if c not in _NO_DEFLATOR]]

    out = nominal.join(vol.add_suffix("_real"), how="outer").sort_index()
    if output and set(_VA_SERVICES) <= set(out.columns):
        out["va_services"] = out[list(_VA_SERVICES)].sum(axis=1, min_count=len(_VA_SERVICES))
        out.loc[out[list(_VA_SERVICES)].isna().any(axis=1), "va_services"] = np.nan

    names = _ordered_names(assets=assets, durability=durability, income=income,
                           output=output, labor=labor, sectors=sectors, stocks=stocks)
    for name in names:
        if name in _NO_DEFLATOR or name not in out or f"{name}_real" not in out:
            continue
        num, den = out[name], out[f"{name}_real"]
        ok = (num > 0) & (den > 0)
        out[f"{name}_defl"] = np.where(ok, 100.0 * num / den, np.nan)

    cols = [n for n in names if n in out]
    cols += [f"{n}_defl" for n in names if f"{n}_defl" in out]
    if real:
        cols += [f"{n}_real" for n in names if f"{n}_real" in out]
    out = out[cols].dropna(how="all")
    out.index = out.index.set_names(["code", "date"])
    out, corrections = _fix_stock_scale(out)

    missing = list(req.missing)
    for c in codes_list:
        if (c not in out.index.get_level_values("code") or "gdp" not in out
                or not out.loc[c, "gdp"].notna().any()):
            missing.append({"what": "country", "code": c,
                            "status": "no GDP at current prices in DF_TABLE1"})
    _warn_missing(missing, "requested items")

    meta = _build_meta(tidy, out, names, vol_how, durability_table, inv_hh_sector,
                       hours_scales, durability, sectors, labor)
    registry = _variables_registry()
    keys_by_flow: dict[str, tuple[str, ...]] = {}
    for r in req.log:
        fl = next((k for k, v in ANA_FLOWS.items() if v == r["flow"]), r["flow"])
        keys_by_flow[fl] = keys_by_flow.get(fl, ()) + (r["key"],)
    variables = []
    for c in out.columns:
        base = c[:-5] if c.endswith(("_defl", "_real")) else c
        flow, units, desc = registry.get(base, ("", "", ""))
        if c.endswith("_defl"):
            units, desc = "index, 2020 = 100", f"Implicit deflator of {base}"
        elif c.endswith("_real"):
            units = "millions of national currency, chain-linked volume at 2020 prices"
            desc = f"Volume of {base}"
        series = out[c].dropna()
        variables.append({
            "name": c, "flow": ANA_FLOWS.get(flow.split("|")[0], flow) if flow else "",
            "units": units, "unit_mult": _unit_mult(units), "sa": "not applicable (annual)",
            "description": desc, "keys": tuple(k for f in flow.split("|") for k in keys_by_flow.get(f, ())), "n_countries": int(series.index.get_level_values("code").nunique()),
            "first": series.index.get_level_values("date").min().year if len(series) else None,
            "last": series.index.get_level_values("date").max().year if len(series) else None,
            "n": int(len(series))})

    if long:
        out = (out.stack().dropna().rename("value").reset_index()
                  .rename(columns={"level_2": "variable"}))
    out.attrs.update(meta=tuple(meta), variables=tuple(variables), missing=tuple(missing),
                     corrections=tuple(corrections),
                     chunks_empty=tuple(req.empty), requests=tuple(req.log),
                     source=" + ".join(req.flows), fetched_at=fetched_at)
    return out


#: A published capital stock is a few years of GDP. A country whose median
#: stock-to-GDP ratio lies in this band publishes its stocks a thousand times
#: too large: ``UNIT_MULT`` says millions while the values are thousands
#: (Iceland in DF_TABLE9A / DF_TABLE9B, verified on 7 Oct 2026: net stock about
#: 2,950 times GDP). The band is three orders of magnitude above any plausible
#: ratio, so the rule fires only on that error and stops firing once the OECD
#: corrects its file.
_STOCK_SCALE_BAND = (300.0, 30000.0)


def _fix_stock_scale(out: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Rescale by 1e-3 the stock columns of a country published in thousands."""
    stock_cols = [c for c in out.columns
                  if c.split("_real")[0] in ANA_STOCKS or c.split("_real")[0] in ANA_STOCKS_GOV]
    if "gdp" not in out or not stock_cols:
        return out, []
    corrections = []
    lo, hi = _STOCK_SCALE_BAND
    for name in [c for c in ("k_net", "k_gross", "k_net_gov") if c in out]:
        ratio = (out[name] / out["gdp"]).groupby(level="code").median()
        for code in ratio[(ratio > lo) & (ratio < hi)].index:
            cols = [c for c in stock_cols if c.split("_real")[0] == name]
            out.loc[code, cols] = out.loc[code, cols].to_numpy() * 1e-3
            corrections.append({"code": code, "variables": tuple(cols), "factor": 1e-3,
                                "median_ratio_to_gdp": float(ratio[code]),
                                "reason": "OECD publishes this stock in thousands "
                                          "under UNIT_MULT 6 (millions)"})
    if corrections:
        warnings.warn("ana_panel: capital stocks rescaled by 1e-3 for "
                      + ", ".join(sorted({c["code"] for c in corrections}))
                      + " (published in thousands); see attrs['corrections']", stacklevel=3)
    return out, corrections


def _unit_mult(units: str) -> int:
    if units.startswith("index"):
        return 0
    return 3 if units.startswith("thousands") else 6


def _warn_missing(missing: list[dict], what: str) -> None:
    """Warn about everything in ``attrs["missing"]`` (empty keys are not there)."""
    bad = list(missing)
    if bad:
        shown = "; ".join(f"{m.get('block', m.get('code'))}: {m.get('status')}"
                          for m in bad[:6])
        warnings.warn(f"ana_panel: {len(bad)} {what} not obtained ({shown}"
                      f"{' ...' if len(bad) > 6 else ''}); see attrs['missing']",
                      stacklevel=3)


def _build_meta(tidy, out, names, vol_how, durability_table, inv_hh_sector,
                hours_scales, durability, sectors, labor) -> list[dict]:
    rows = []
    tables_not_t0102: dict[str, tuple[str, ...]] = {}
    if "TABLE" in tidy:
        t1 = tidy[tidy["TABLE"].isin(_TABLE_RANK) & (tidy["TABLE"] != "T0102")
                  & tidy["name"].isin(list(ANA_MAIN) + list(ANA_ASSETS))]
        for (code, name, pb, tab), _ in t1.groupby(["code", "name", "PRICE_BASE", "TABLE"]):
            tables_not_t0102[code] = tables_not_t0102.get(code, ()) + (f"{name}:{pb}={tab}",)
    codes = list(dict.fromkeys(out.index.get_level_values("code")))
    for code in codes:
        g = out.loc[code]
        years = g.index[g.notna().any(axis=1)]
        cur = tidy.loc[(tidy["code"] == code) & (tidy["PRICE_BASE"].isin(["V", "LR", "L"]))]
        currency = ""
        if "CURRENCY" in cur:
            vals = [v for v in cur["CURRENCY"].dropna().astype(str) if v not in ("_Z", "nan", "")]
            currency = vals[0] if vals else ""
        how = vol_how.get(code, {})
        if "gdp_real" in g and g["gdp_real"].notna().any():
            gdp_base = how.get("gdp", "LR")
        else:
            gdp_base = ""
        ref_year = 2020 if gdp_base in ("LR", "L->2020") else (
            _implied_ref_year(g) if gdp_base == "L" else None)
        absent = tuple(n for n in names if n not in g.columns or not g[n].notna().any())
        rows.append({
            "code": code, "currency": currency,
            # Units differ by column (money, persons, hours, index): they are
            # per variable in attrs["variables"], not per country.
            "units": None, "unit_mult": None, "sa": "not applicable (annual)",
            "first": int(years.min().year) if len(years) else None,
            "last": int(years.max().year) if len(years) else None,
            "n": int(len(years)),
            "volume_base": gdp_base, "volume_ref_year": ref_year,
            "volume_from_l": tuple(sorted(how)),
            "tables_not_t0102": tables_not_t0102.get(code, ()),
            "durability_table": durability_table.get(code, "") if durability else "",
            "inv_hh_sector": inv_hh_sector.get(code, "") if sectors else "",
            "hours_scale": hours_scales.get(code, 1.0) if labor else None,
            "absent": absent,
        })
    return rows


def _implied_ref_year(g: pd.DataFrame) -> int | None:
    """Year in which the GDP deflator is closest to 100 (volume at own base)."""
    if "gdp_defl" not in g:
        return None
    s = g["gdp_defl"].dropna()
    if s.empty:
        return None
    return int((s - 100.0).abs().idxmin().year)


def ana_panel_meta(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-country metadata of an :func:`ana_panel` result, as a DataFrame.

    The records live in ``panel.attrs["meta"]`` as plain dicts, because a
    DataFrame inside ``attrs`` makes ``pd.concat`` of two slices raise.
    """
    return pd.DataFrame(list(panel.attrs.get("meta", ())), columns=list(_META_KEYS))


__all__ = [
    "ANA_ASSETS", "ANA_DURABILITY", "ANA_FLOWS", "ANA_INCOME", "ANA_INCOME_SECTOR",
    "ANA_LABOR", "ANA_LABOR_ACTIVITIES", "ANA_MAIN", "ANA_OUTPUT", "ANA_POP",
    "ANA_SECTORS", "ANA_STOCKS", "ANA_STOCKS_GOV", "ANA_VA_ADDITIVE",
    "NAMAIN10_DIMS", "NASEC10_DIMS", "ana_countries", "ana_panel", "ana_panel_meta",
]
