r"""One-call cross-country quarterly national accounts: nominal SA + deflators.

``qna_panel(["USA", "ESP", "MEX"])`` returns a ready-to-use quarterly panel
indexed by ``(code, date)`` with the expenditure side of the national accounts
in **current prices, seasonally and calendar adjusted** (millions of national
currency) plus the matching **implicit price deflators**::

    gdp  cons_hh  cons_gov  inv  capform  exports  imports
    gdp_defl  cons_hh_defl  cons_gov_defl  inv_defl  capform_defl ...

so that ``real = nominal / deflator * 100`` and the expenditure identity

.. math:: Y = C_{hh} + C_{gov} + I + X - M

closes up to the statistical discrepancy. ``output=True`` and ``income=True``
join the other two approaches to the same GDP — value added by industry, and
the income it is paid out as — so all three of

.. math::

    Y = C + G + I + X - M, \qquad
    Y = \sum_j \text{VA}_j + (D21 - D31), \qquad
    Y = D1 + B2A3G + (D2 - D3)

sit in one frame and can be scored against each other with
:func:`puremacro.fetch.qna_identity`. Everything is one SDMX round-trip
per institutional sector per chunk of ten countries, on-disk cached by
:func:`puremacro.fetch._oecd_sdmx.get_sdmx_csv`.

Why a dedicated fetcher
-----------------------
:func:`puremacro.fetch.oecd_qna_expenditure.fetch_qna_expenditure` hard-filters
``PRICE_BASE == "L"`` (chain-linked volumes) and returns logs. That silently
yields an empty frame for the economies that publish **fixed-base** volumes
only — Mexico, Argentina, Indonesia, India, South Africa — and it never exposes
current prices, so a deflator cannot be built from it. This module keeps the
current-price series (``PRICE_BASE == "V"``) as the primary object and picks
the volume measure per country (``L`` when published, else ``Q``) purely to
divide it out into a deflator.

Source: OECD SDMX dataflow ``OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR``.
"""
from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ._hours import hours_scale_factors

_AGENCY_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR,"

#: column name -> (SDMX TRANSACTION, SDMX SECTOR, human description)
QNA_COMPONENTS: dict[str, tuple[str, str, str]] = {
    "gdp":      ("B1GQ", "S1",  "Gross domestic product (expenditure approach)"),
    "cons_hh":  ("P3",   "S1M", "Household final consumption (households + NPISH)"),
    "cons_gov": ("P3",   "S13", "General government final consumption"),
    "inv":      ("P51G", "S1",  "Gross fixed capital formation"),
    "capform":  ("P5",   "S1",  "Gross capital formation (GFCF + inventories)"),
    "exports":  ("P6",   "S1",  "Exports of goods and services"),
    "imports":  ("P7",   "S1",  "Imports of goods and services"),
}

_ASSET_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GFCF_ASSET,"

#: column name -> (SDMX INSTR_ASSET, human description). Gross fixed capital
#: formation split by asset, which is what separates the *tradable* part of
#: investment (machinery, equipment, ICT — largely imported) from the part
#: that is produced on site and barely trades at all (buildings, civil works).
QNA_ASSETS: dict[str, tuple[str, str]] = {
    "inv_equip":  ("N11MG", "Machinery and equipment and weapons systems"),
    "inv_struct": ("N112G", "Other buildings and structures"),
    "inv_dwell":  ("N111G", "Dwellings"),
    "inv_ipp":    ("N117G", "Intellectual property products (R&D, software)"),
}

_DURABILITY_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_DURABILITY,"

#: column name -> (SDMX TRANSACTION, human description). Household final
#: consumption split by durability (institutional sector ``S14``, households
#: only — the headline ``cons_hh`` is ``S1M``, households + NPISH). Durables
#: are the part national accounts book as consumption but macro theory treats
#: as household capital (Cooley & Prescott 1995; Gomme & Rupert 2007).
QNA_DURABILITY: dict[str, tuple[str, str]] = {
    "cons_dur":     ("P311", "Durable goods"),
    "cons_semidur": ("P312", "Semi-durable goods"),
    "cons_nondur":  ("P313", "Non-durable goods"),
    "cons_serv":    ("P314", "Services"),
}

_OUTPUT_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_BY_ACTIVITY_OUTPUT,"

#: column name -> (SDMX TRANSACTION, SDMX ACTIVITY, human description). The
#: **output (production) approach**: gross value added by ISIC Rev.4 activity,
#: plus the taxes that stand between value added and GDP at market prices.
#:
#: .. math:: Y = \sum_j \text{VA}_j + (D21 - D31) + \text{YA1}
#:
#: **Two of these columns are memo items, not addends.** ``va_mfg``
#: (manufacturing) sits *inside* ``va_ind`` (industry), and ``va_services`` is
#: the sum of the seven service activities that already appear separately.
#: Adding every column would count roughly a third of the economy twice, so
#: the additive subset is published separately as :data:`QNA_VA_ADDITIVE`.
QNA_ACTIVITIES: dict[str, tuple[str, str, str]] = {
    "gdp_output":  ("B1GQ",   "_Z",  "GDP as this flow publishes it"),
    "va_total":    ("B1G",    "_T",  "Gross value added, all activities"),
    "va_agri":     ("B1G",    "A",   "Agriculture, forestry and fishing"),
    "va_ind":      ("B1G",    "BTE", "Industry except construction (B-E)"),
    "va_mfg":      ("B1G",    "C",   "Manufacturing (memo: inside va_ind)"),
    "va_constr":   ("B1G",    "F",   "Construction"),
    "va_trade":    ("B1G",    "GTI", "Trade, transport, accommodation, food (G-I)"),
    "va_ict":      ("B1G",    "J",   "Information and communication"),
    "va_fin":      ("B1G",    "K",   "Financial and insurance activities"),
    "va_realest":  ("B1G",    "L",   "Real estate activities"),
    "va_prof":     ("B1G",    "M_N", "Professional, scientific, admin (M-N)"),
    "va_public":   ("B1G",    "OTQ", "Public admin, education, health (O-Q)"),
    "va_other":    ("B1G",    "RTU", "Other services (R-U)"),
    "va_services": ("B1G",    "GTU", "All services (memo: sum of G-U above)"),
    "taxes_prod":  ("D21X31", "_Z",  "Taxes less subsidies on products"),
    "chainlink_disc": ("YA1", "_Z",  "Chain-linking discrepancy, where published"),
}

#: The value-added columns that actually sum to ``va_total``. Excludes the two
#: memo items in :data:`QNA_ACTIVITIES` (``va_mfg``, ``va_services``).
QNA_VA_ADDITIVE: tuple[str, ...] = (
    "va_agri", "va_ind", "va_constr", "va_trade", "va_ict", "va_fin",
    "va_realest", "va_prof", "va_public", "va_other")

#: Value-added columns that are subsets or aggregates of the additive ones.
QNA_VA_MEMO: tuple[str, ...] = ("va_mfg", "va_services")

_INCOME_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_INCOME,"

#: column name -> (SDMX TRANSACTION, SDMX ACTIVITY, human description). The
#: **income approach**: what GDP is paid out as.
#:
#: .. math:: Y = D1 + B2A3G + (D2 - D3)
#:
#: ``comp_emp`` over GDP is the *unadjusted* labour share. It is unadjusted in
#: a way that matters: the labour income of the self-employed is not in ``D1``
#: at all, it sits inside ``surplus_mixed`` (B2A3G is gross operating surplus
#: **and mixed income** together, and the accounts do not split them), which is
#: exactly the problem Gollin (2002) is about. Italy reads 39% and the United
#: States 54% largely because of how much self-employment each has.
#:
#: These series exist **only in current prices** — there is no volume measure
#: of compensation of employees — so they get no deflator and no ``_real``
#: column, and ``real=True`` does not change that.
QNA_INCOME: dict[str, tuple[str, str, str]] = {
    "gdp_income":    ("B1GQ",  "_Z", "GDP as this flow publishes it"),
    "comp_emp":      ("D1",    "_T", "Compensation of employees"),
    "surplus_mixed": ("B2A3G", "_T", "Gross operating surplus and mixed income"),
    "taxes_prod_imp_net": ("D2X3", "_Z",
                           "Taxes less subsidies on production and imports"),
    "taxes_prod_imp": ("D2",   "_Z", "Taxes on production and imports"),
    "subsidies":     ("D3",    "_Z", "Subsidies (enter negatively)"),
}

_LABOR_FLOW = "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_BY_ACTIVITY_EMPDC,"

#: column name -> (SDMX TRANSACTION, SDMX UNIT_MEASURE, human description). The
#: **labour input** the same accounts measure: heads and hours, split into
#: employees and the self-employed, on the *domestic* concept — employment in
#: resident production units, which is the concept GDP is measured on, so
#: ``gdp_real / hours`` is a productivity measure and not a mismatch of two
#: populations.
#:
#: These carry no price and therefore no deflator: persons are reported in
#: thousands and hours in millions, and :data:`QNA_LABOR_UNITS` records which
#: is which. The split matters twice over — ``emp_selfemp / emp`` is the share
#: of the workforce whose labour income the accounts book inside
#: ``surplus_mixed`` rather than ``comp_emp``, which is the Gollin (2002)
#: correction to the labour share of :data:`QNA_INCOME`; and ``hours / emp``
#: is average hours per worker, the margin that moves in European recessions
#: where employment does not.
#:
#: With ``income=True`` and ``output=True`` alongside, the panel now carries
#: every column :func:`puremacro.labor_share.gollin_adjusted_ls` asks for —
#: ``comp_emp``, ``va_total``, ``surplus_mixed``, ``emp_employees`` and
#: ``emp_selfemp`` — which before this block had no single source.
QNA_LABOR: dict[str, tuple[str, str, str]] = {
    "emp":             ("EMP",  "PS", "Total employment, persons (thousands)"),
    "emp_employees":   ("SAL",  "PS", "Employees, persons (thousands)"),
    "emp_selfemp":     ("SELF", "PS", "Self-employed, persons (thousands)"),
    "hours":           ("EMP",  "H",  "Total hours worked (millions)"),
    "hours_employees": ("SAL",  "H",  "Hours worked by employees (millions)"),
    "hours_selfemp":   ("SELF", "H",  "Hours worked by the self-employed (millions)"),
}

#: ISIC Rev.4 activity -> column suffix, for the by-activity breakdown of the
#: labour block. ``_T`` is the whole economy and keeps the plain names; every
#: other activity appends its suffix, so ``hours`` gains ``hours_agri`` and
#: ``hours_public`` alongside it. The two named here are the two the accounts
#: measure differently from everything else, and they are the two you have to
#: remove to get a *market* sector: agriculture, where most of the labour is
#: self-employed and the output is the weather, and O–Q, where value added is
#: **defined** in the SNA as compensation plus consumption of fixed capital, so
#: its measured productivity growth is near zero by construction rather than by
#: finding. The suffixes match :data:`QNA_ACTIVITIES` (``va_agri``,
#: ``va_public``) so the two sides of :math:`Y/H` line up by name.
#:
#: Every activity of one unit of measure shares a seasonal adjustment, because
#: the whole point of the breakdown is ``hours - hours_agri - hours_public``
#: and a total adjusted at source minus a raw part is not a subtraction of
#: anything. See ``sa_family`` in :func:`_tidy`.
QNA_LABOR_ACTIVITIES: dict[str, str] = {"A": "agri", "OTQ": "public"}

#: Every labour column name the panel can carry, the by-activity ones included.
#: Three separate consumers need exactly this set — the deflator exclusion
#: (:data:`_NO_DEFLATOR`), the hours rescale (:func:`_rescale_hours`) and the
#: metadata builder (:func:`_build_meta`) — and the first version of the
#: by-activity change widened two of the three by hand and missed the last, so
#: it is built once here and read from there.
_LABOR_NAMES: frozenset[str] = (
    frozenset(QNA_LABOR)
    | frozenset(f"{n}_{s}" for n in QNA_LABOR
                for s in QNA_LABOR_ACTIVITIES.values()))

#: Scale each labour unit of measure is normalised to, as a power of ten:
#: persons in thousands, hours in millions. Every reference area currently
#: publishes ``UNIT_MULT`` 3 for ``PS`` and 6 for ``H``, so today this mapping
#: is a no-op; it is applied anyway because the money blocks have already been
#: caught publishing a scale the panel did not expect, and a silent factor of
#: a thousand in an employment series is not something a caller would spot.
QNA_LABOR_UNITS: dict[str, int] = {"PS": 3, "H": 6}

#: Hours per worker per quarter that no real economy falls outside. Across the
#: 31 reference areas that publish heads and hours on the same basis, *every
#: observation ever published* sits in 304–572; this band is deliberately far
#: wider, so it can only ever fire on an order-of-magnitude error and never on
#: a country that merely works short weeks.
_HOURS_IMPLAUSIBLE = (150.0, 1000.0)

#: Where a correction has to land before it is accepted. Tighter than the
#: detection band: a rescaling that does not produce a believable working year
#: is not applied at all, and the series is left as published.
_HOURS_PLAUSIBLE = (250.0, 700.0)

#: Factors that turn a mislabelled basis into a quarterly one, with the basis
#: each implies. Chile publishes hours per week (13 weeks to a quarter) and
#: Costa Rica at an annual rate (4 quarters to a year); both are labelled
#: exactly like everyone else's quarterly figure, so only the magnitude
#: distinguishes them.
_HOURS_SCALES: dict[float, str] = {13.0: "weekly", 0.25: "annual rate"}

_SECTORS = sorted({s for _, s, _ in QNA_COMPONENTS.values()})
_TRANSACTIONS = {t for t, _, _ in QNA_COMPONENTS.values()}

#: series with no price dimension at all, so no implicit deflator can be
#: built for them: the income flows exist only in current prices, and the
#: labour block is counts of people and hours.
_NO_DEFLATOR: frozenset[str] = frozenset(QNA_INCOME) | _LABOR_NAMES

#: SDMX reference areas that are country groupings rather than countries.
#: The QNA dataflows publish them alongside the members, and a panel that
#: silently mixes ``OECD`` in with ``USA`` double-counts every aggregate it
#: touches, so :func:`qna_countries` drops them unless asked for them.
QNA_AGGREGATES: frozenset[str] = frozenset(
    {"EA19", "EA20", "EU15", "EU27_2020", "EU28", "G7", "G20",
     "NAFTA", "OECD", "OECD26", "OECDE", "W"})

#: Fallback for :func:`qna_countries` when the availability endpoint cannot be
#: reached. Every reference area the OECD published quarterly expenditure
#: accounts for as of the 2026-Q2 vintage; the live query is authoritative.
_QNA_COUNTRIES_FALLBACK: tuple[str, ...] = (
    "ARG", "AUS", "AUT", "BEL", "BGR", "BRA", "CAN", "CHE", "CHL", "CHN",
    "COL", "CRI", "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR",
    "GRC", "HRV", "HUN", "IDN", "IND", "IRL", "ISL", "ISR", "ITA", "JPN",
    "KOR", "LTU", "LUX", "LVA", "MEX", "NLD", "NOR", "NZL", "POL", "PRT",
    "ROU", "RUS", "SAU", "SVK", "SVN", "SWE", "TUR", "USA", "ZAF")

_AVAILABILITY = ("https://sdmx.oecd.org/public/rest/availableconstraint/"
                 "{flow}1.0/Q............?mode=exact&format=jsondata")


def qna_countries(*, include_aggregates: bool = False,
                  flow: str = _AGENCY_FLOW,
                  refresh: bool = False) -> list[str]:
    """Every reference area the OECD publishes quarterly accounts for.

    Asks the SDMX *availability* endpoint which values of ``REF_AREA`` carry
    data in ``flow``, so a panel can be built for the largest set the source
    actually supports instead of a hand-maintained list that goes stale the
    next time the OECD onboards a country.

    Parameters
    ----------
    include_aggregates
        Keep the country groupings (``OECD``, ``EA20``, ``G7``, ...). Default
        ``False`` returns individual countries only — see :data:`QNA_AGGREGATES`.
    flow
        Agency/dataflow to ask about, trailing comma included. Defaults to the
        expenditure-in-national-currency flow behind :func:`qna_panel`; pass a
        sibling flow (assets, durability) to see its own thinner coverage.
    refresh
        Bypass the on-disk HTTP cache and re-query.

    Returns
    -------
    list of str
        Sorted ISO3 codes, ready to hand to :func:`qna_panel`. Falls back to
        the frozen list in ``_QNA_COUNTRIES_FALLBACK`` if the endpoint cannot
        be reached, so this never raises and never returns empty.

    Examples
    --------
    >>> codes = qna_countries()            # doctest: +SKIP
    >>> panel = qna_panel(codes)           # doctest: +SKIP
    """
    codes = _fetch_ref_areas(flow, refresh)
    if not codes:
        codes = list(_QNA_COUNTRIES_FALLBACK)
        if include_aggregates:
            codes += sorted(QNA_AGGREGATES)
    if not include_aggregates:
        codes = [c for c in codes if c not in QNA_AGGREGATES]
    return sorted(set(codes))


def _fetch_ref_areas(flow: str, refresh: bool) -> list[str]:
    """REF_AREA values with data in ``flow``; empty list on any failure."""
    import json

    url = _AVAILABILITY.format(flow=flow)
    try:
        from ._http import cached_get

        raw = cached_get(url, refresh=refresh, timeout=180,
                         headers={"Accept": "application/vnd.sdmx.structure"
                                            "+json;version=1.0"})
        payload = json.loads(raw)
        regions = payload["data"]["contentConstraints"][0]["cubeRegions"][0]
        for kv in regions["keyValues"]:
            if kv["id"] == "REF_AREA":
                return [str(v) for v in kv["values"]]
    except Exception:
        # Availability is an optimisation, not a dependency: any failure
        # (offline, rate limit, a schema change at the OECD) degrades to the
        # frozen list rather than taking the caller's panel down with it.
        return []
    return []

def get_sdmx_csv(agency_flow: str, key: str, start_period: str,
                 *, refresh: bool = False) -> pd.DataFrame:
    """Thin indirection over :func:`puremacro.fetch._oecd_sdmx.get_sdmx_csv`.

    The import is deferred so that ``import puremacro.fetch`` stays free of
    ``requests`` (it has to work under Pyodide, where the scraper stack is
    absent), and so tests can monkeypatch this name.
    """
    from ._oecd_sdmx import get_sdmx_csv as _impl

    return _impl(agency_flow, key, start_period, refresh=refresh)


_META_COLS = ["code", "currency", "units", "price_base", "price_ref_year",
              "sa", "sa_detail", "sa_labor", "hours_scale",
              "n_obs", "first", "last"]

#: What ``qna_meta``'s ``units`` column describes: the money columns. The
#: labour block is counts, not currency — persons in thousands and hours in
#: millions, per :data:`QNA_LABOR_UNITS` — and is not covered by this string.
_UNITS = "millions of national currency, current prices, SA"


def _quarter_to_date(s: pd.Series) -> pd.Series:
    return pd.PeriodIndex(s.astype(str), freq="Q").to_timestamp(how="start")


def qna_meta(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-country metadata of a :func:`qna_panel` result, as a DataFrame.

    The records live in ``panel.attrs["meta"]`` as a tuple of plain dicts
    rather than as a DataFrame: pandas compares ``.attrs`` for equality when
    it propagates them, and a DataFrame in there makes ``pd.concat`` on any
    slice of the panel raise ``ValueError: The truth value of a DataFrame is
    ambiguous``. This helper is the intended way to read them back.
    """
    return pd.DataFrame(list(panel.attrs.get("meta", ())), columns=_META_COLS)


def _empty(long: bool) -> pd.DataFrame:
    if long:
        out = pd.DataFrame(columns=["code", "date", "variable", "value"])
    else:
        idx = pd.MultiIndex.from_arrays([[], []], names=["code", "date"])
        out = pd.DataFrame(index=idx)
    out.attrs["meta"] = ()
    out.attrs["source"] = _AGENCY_FLOW.rstrip(",")
    return out


def _code_chunks(codes: Sequence[str] | None) -> list[str]:
    """OECD truncates very wide REF_AREA filters — request ten at a time."""
    if codes is None:
        return [""]
    return ["+".join(codes[i:i + 10]) for i in range(0, len(codes), 10)]


def _download(codes: Sequence[str] | None, start: str, refresh: bool) -> pd.DataFrame:
    """Raw expenditure rows, one request per institutional sector per chunk."""
    parts: list[pd.DataFrame] = []
    for code_key in _code_chunks(codes):
        for sector in _SECTORS:
            # 13 dims: FREQ.ADJUSTMENT.REF_AREA.SECTOR.COUNTERPART_SECTOR.
            # TRANSACTION.INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.
            # PRICE_BASE.TRANSFORMATION.TABLE_IDENTIFIER
            key = f"Q..{code_key}.{sector}.........."
            raw = get_sdmx_csv(_AGENCY_FLOW, key, start, refresh=refresh)
            if not raw.empty:
                parts.append(raw)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _download_flow(flow: str, codes: Sequence[str] | None, start: str,
                   refresh: bool, *, tail: str = "..........") -> pd.DataFrame:
    """Raw rows from a sibling QNA dataflow (same 13-dim key), one per chunk.

    ``tail`` is everything after ``REF_AREA`` in the key. It is left wide open
    by default; the labour flow overrides it to pin ACTIVITY to the total
    economy, which is the difference between one response and twelve times one
    response (that flow publishes every ISIC section).
    """
    parts: list[pd.DataFrame] = []
    for code_key in _code_chunks(codes):
        raw = get_sdmx_csv(flow, f"Q..{code_key}{tail}", start,
                           refresh=refresh)
        if not raw.empty:
            parts.append(raw)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _tidy(raw: pd.DataFrame, lookup: dict[tuple, str],
          dims: tuple[str, ...], *, sa: str = "prefer",
          min_gain: int | None = None,
          units: tuple[str, ...] = ("XDC",),
          price_bases: tuple[str, ...] = ("V", "L", "Q"),
          mult_target: int | dict[str, int] = 6,
          sa_family: dict[str, str] | None = None) -> pd.DataFrame:
    """Filter the raw SDMX rows down to one row per (code, date, name, price_base).

    ``lookup`` maps a tuple of ``dims`` values (e.g. ``("B1GQ", "S1")``) to the
    output column name; rows whose dimension tuple is absent are dropped.

    ``units``, ``price_bases`` and ``mult_target`` describe the measurement the
    flow uses. The money flows are all ``XDC`` at one of three price bases and
    are put in millions; the labour flow is counts (``PS``) and hours (``H``)
    at no price base at all, and each goes on its own scale — hence
    ``mult_target`` also accepts a per-unit mapping.

    ``sa_family`` groups output columns that have to share one seasonal
    adjustment, mapping each name to a family label. Without it the adjusted /
    unadjusted choice is made per series, which is right for blocks that are
    just a list of series and wrong for one sold as a decomposition: Korea
    publishes total employment adjusted but its employee and self-employed
    components raw, so a per-series choice returns an SA total with NSA parts
    and ``emp_employees + emp_selfemp`` no longer sums to ``emp``. Naming a
    family makes the whole family fall back together, so the identity holds.
    """
    needed = {"PRICE_BASE", "UNIT_MEASURE", "ADJUSTMENT",
              "REF_AREA", "TIME_PERIOD", "OBS_VALUE", *dims}
    if not needed.issubset(raw.columns):
        return pd.DataFrame()

    df = raw[raw["UNIT_MEASURE"].isin(units)
             & raw["PRICE_BASE"].isin(price_bases)].copy()
    if "ACTIVITY" in df.columns and "ACTIVITY" not in dims:
        # Expenditure-side flows publish one row per activity and we want the
        # economy-wide total. The output flow keys *on* activity, so there the
        # filter would throw away the entire point of the request.
        df = df[df["ACTIVITY"].isin(["_T", "_Z"]) | df["ACTIVITY"].isna()]
    if "TRANSFORMATION" in df.columns:
        df = df[(df["TRANSFORMATION"] == "N") | df["TRANSFORMATION"].isna()]
    if df.empty:
        return pd.DataFrame()

    dim_tuples = zip(*(df[d] for d in dims))
    df["name"] = [lookup.get(k) for k in dim_tuples]
    df = df[df["name"].notna()]
    if df.empty:
        return pd.DataFrame()

    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["value"])
    # Put every series on one scale (UNIT_MULT is the power of ten the
    # published figure carries; 6 = millions), because reference areas differ
    # on which one they publish. The target has to be computed first: a
    # missing multiplier must fall back to *this block's* scale, not to the
    # money block's 6, or a persons row with a blank UNIT_MULT is silently
    # multiplied by a thousand.
    if "UNIT_MULT" in df.columns:
        mult = pd.to_numeric(df["UNIT_MULT"], errors="coerce")
    else:
        mult = pd.Series(np.nan, index=df.index)
    if isinstance(mult_target, dict):
        target = df["UNIT_MEASURE"].map(mult_target).astype(float)
    else:
        target = pd.Series(float(mult_target), index=df.index)
    # A blank multiplier, or a unit this block has no target for, means
    # "already on the right scale": each falls back to the other so the factor
    # is 1. Rescaling by a NaN would delete the series instead.
    target, mult = target.fillna(mult), mult.fillna(target)
    df["value"] = df["value"] * np.power(10.0, (mult - target).fillna(0.0))
    df["date"] = _quarter_to_date(df["TIME_PERIOD"])
    df = df.rename(columns={"REF_AREA": "code"})

    keys = ["code", "name", "PRICE_BASE"]
    df["_adj_rank"] = np.where(df["ADJUSTMENT"] == "Y", 0, 1)
    df = df.sort_values(keys + ["date", "_adj_rank"])
    # Which series must share one adjustment. Absent `sa_family` every series
    # is its own family and both branches below reduce to the per-series rule
    # they used before.
    df["_fam"] = (df["name"] if sa_family is None
                  else df["name"].map(sa_family).fillna(df["name"]))
    fam_keys = ["code", "_fam", "PRICE_BASE"]

    if sa == "x13":
        # Default: the source's own adjusted series always wins where it
        # exists; the raw series is taken only for blocks the source never
        # adjusts (CHN, SAU, and the asset/durability splits of MEX, JPN,
        # TUR), and those get adjusted here.
        #
        # `min_gain` additionally trades an official adjustment for ours when
        # the raw series is that many quarters longer. It is off by default,
        # and deliberately so: US durables volumes run from 2002 unadjusted
        # but only 2007 adjusted, and taking the longer raw series triples the
        # quarterly volatility of adjusted US services consumption relative to
        # the OECD's own adjustment. Five extra years is not worth a worse
        # series. Turn it on only after checking what it does to your series.
        n = (df.groupby(keys + ["ADJUSTMENT"], sort=False)["date"]
               .transform("nunique"))
        cover = (df.assign(_n=n)
                   .drop_duplicates(subset=keys + ["ADJUSTMENT"])
                   .pivot_table(index=keys, columns="ADJUSTMENT", values="_n",
                                fill_value=0))
        n_y = cover["Y"] if "Y" in cover else 0 * cover.iloc[:, 0]
        n_n = cover["N"] if "N" in cover else 0 * cover.iloc[:, 0]
        take_raw = (n_y.to_numpy() == 0) & (n_n.to_numpy() > 0)
        if min_gain is not None:
            take_raw |= ((n_n >= n_y + min_gain) & (n_n > 0)).to_numpy()
        pick = pd.DataFrame({"ADJUSTMENT": np.where(take_raw, "N", "Y")},
                            index=cover.index).reset_index()
        if sa_family is not None:
            # One member taking the raw series pulls its whole family with it,
            # so the parts and the total are all adjusted by the same engine.
            pick["_fam"] = pick["name"].map(sa_family).fillna(pick["name"])
            raw_fam = (pick.assign(_raw=take_raw)
                           .groupby(fam_keys, sort=False)["_raw"]
                           .transform("max"))
            pick["ADJUSTMENT"] = np.where(raw_fam.to_numpy(), "N", "Y")
            pick = pick.drop(columns="_fam")
        df = df.merge(pick, on=keys + ["ADJUSTMENT"], how="inner")
    else:
        # Adjusted-at-source wins outright; unadjusted rows only survive for a
        # block that has no adjusted data at all — where `sa_family` groups
        # several names into one block, one member missing an adjusted series
        # sends the whole family to the raw one rather than mixing the two.
        best = df.groupby(keys, sort=False)["_adj_rank"].transform("min")
        fam_best = (df.assign(_best=best).groupby(fam_keys, sort=False)["_best"]
                      .transform("max"))
        df = _warn_family_losses(df, df[df["_adj_rank"] == fam_best], keys)

    return (df.drop(columns="_fam")
              .drop_duplicates(subset=keys + ["date"], keep="first"))


def _warn_family_losses(before: pd.DataFrame, after: pd.DataFrame,
                        keys: list[str]) -> pd.DataFrame:
    """Say so when the family rule erased a series outright.

    The family rule is a *filter*, not a preference: one member published
    unadjusted-only sends the whole family to the raw series, and any member
    that has no unadjusted rows then matches nothing and disappears. That is
    the right call when the alternative is mixing two adjustment engines inside
    one decomposition — but the member that vanishes can be the total, and a
    decomposition returned without its total is the one outcome this machinery
    exists to prevent. It is silent today, and it is silent on the ``x13``
    branch too.

    No reference area in the current OECD flows triggers it, which is why it
    has never been seen; widening the labour family from three names to nine
    (:data:`QNA_LABOR_ACTIVITIES`) triples the number of ways it could be. So
    the series still goes -- mixing adjustments would be worse -- but it goes
    with its name on a warning rather than without a trace.
    """
    if after.empty or len(after) == len(before):
        return after
    lost = (before[~before.set_index(keys).index.isin(after.set_index(keys).index)]
            .drop_duplicates(subset=keys))
    if not lost.empty:
        names = ", ".join(sorted({f"{r['code']}/{r['name']}"
                                  for _, r in lost.iterrows()})[:12])
        warnings.warn(
            f"seasonal-adjustment family rule dropped {len(lost)} series that "
            f"publish no unadjusted edition: {names}. A family takes one "
            f"adjustment or none, so these had no variant to fall back to; if "
            f"one of them is a total, its parts are now returned without it.",
            UserWarning, stacklevel=3)
    return after


def _labor_activity_lookup(activities: bool | Sequence[str]
                           ) -> tuple[dict[tuple, str], tuple[str, ...]]:
    """``(TRANSACTION, UNIT_MEASURE, ACTIVITY) -> column``, and the activities.

    ``activities`` is False for the total economy alone (today's behaviour),
    True for :data:`QNA_LABOR_ACTIVITIES`, or an explicit sequence of ISIC
    codes. ``_T`` is always requested: the breakdown is only ever used by
    subtracting from the total, so returning a part without its whole would be
    returning something no caller can use.
    """
    if activities is False or activities is None:
        acts: tuple[str, ...] = ("_T",)
    elif activities is True:
        acts = ("_T", *QNA_LABOR_ACTIVITIES)
    else:
        # `dict.fromkeys` rather than a set: order is the column order, and a
        # repeated code would otherwise be expanded twice by `_ordered_columns`
        # into a frame with duplicate labels, which `long=True` cannot stack.
        acts = tuple(dict.fromkeys(("_T", *activities)))
    unknown = [a for a in acts[1:] if a not in QNA_LABOR_ACTIVITIES]
    if unknown:
        raise ValueError(
            f"no column name for ISIC activity {', '.join(unknown)}; add it to "
            f"QNA_LABOR_ACTIVITIES (have: {', '.join(QNA_LABOR_ACTIVITIES)})")
    lookup = {}
    for name, (t, u, _) in QNA_LABOR.items():
        for act in acts:
            suffix = QNA_LABOR_ACTIVITIES.get(act)
            lookup[(t, u, act)] = name if act == "_T" else f"{name}_{suffix}"
    return lookup, acts


def _labor_tidy(codes: Sequence[str] | None, start: str, refresh: bool, *,
                sa: str = "prefer", sa_min_gain: int | None = None,
                hours_rescale: bool = True,
                activities: bool | Sequence[str] = False
                ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Download and tidy the labour block alone. Shared by both entry points."""
    lookup_l, acts = _labor_activity_lookup(activities)
    # Same 13-dim key as the money flows, but ACTIVITY pinned to what was
    # asked for: this flow publishes every ISIC section, and asking for all of
    # them is twelve times the response for eleven series nobody wanted.
    raw_l = _download_flow(_LABOR_FLOW, codes, start, refresh,
                           tail=f".....{'+'.join(acts)}.....")
    if raw_l.empty:
        return pd.DataFrame(), {}
    tidy_l = _tidy(raw_l, lookup_l, ("TRANSACTION", "UNIT_MEASURE", "ACTIVITY"),
                   sa=sa, min_gain=sa_min_gain,
                   units=tuple(QNA_LABOR_UNITS),
                   price_bases=("_Z",),
                   mult_target=QNA_LABOR_UNITS,
                   # Heads are one family and hours another: the total, its two
                   # institutional parts and its activity parts have to carry
                   # the same adjustment or they stop adding up — and the
                   # activity columns exist only to be subtracted from the
                   # total, which is the subtraction that would break first.
                   sa_family={c: u for (_, u, _a), c in lookup_l.items()})
    if tidy_l.empty:
        return pd.DataFrame(), {}
    if hours_rescale:
        tidy_l, scales = _rescale_hours(tidy_l)
        return tidy_l, scales
    return tidy_l, {}


def qna_labor(codes: Iterable[str] | None = None, start: str = "1995", *,
              sa: str = "prefer", sa_min_gain: int | None = None,
              hours_rescale: bool = True, activities: bool | Sequence[str] = False,
              refresh: bool = False) -> pd.DataFrame:
    """The labour block on its own, without the expenditure block.

    :func:`qna_panel` with ``labor=True`` joins employment and hours onto the
    national accounts, which means it also downloads the expenditure block and,
    under ``sa="x13"``, seasonally adjusts it. When the labour series are all
    you want that is three extra SDMX round-trips per chunk of ten reference
    areas and a lot of X-13 you did not
    ask for — and it couples the result to a flow you are not using, because a
    reference area publishing labour but no expenditure is dropped and an
    expenditure request that comes back empty takes the labour rows with it.

    Returns a long frame with columns ``code``, ``date``, ``variable``,
    ``value`` and ``sa_source``, the last being ``oecd`` / ``puremacro`` /
    ``none`` **per series** rather than per country. Values are levels:
    thousands of persons and millions of hours, per :data:`QNA_LABOR_UNITS`.
    ``df.attrs["hours_scale"]`` records any time-base correction, as in
    :func:`qna_panel`; see ``hours_rescale`` there for what that means.

    ``activities=True`` adds the ISIC breakdown of :data:`QNA_LABOR_ACTIVITIES`
    — every series again for agriculture (``_agri``) and for public
    administration, education and health (``_public``) — which is what turns
    the whole-economy total into a *market* sector by subtraction. It costs
    nothing extra: same request, three activities instead of one.
    """
    if sa not in ("prefer", "x13"):
        raise ValueError(f"sa must be 'prefer' or 'x13', got {sa!r}")
    codes_list = None if codes is None else [c.upper() for c in codes if c]
    tidy_l, scales = _labor_tidy(codes_list, start, refresh, sa=sa,
                                 sa_min_gain=sa_min_gain,
                                 hours_rescale=hours_rescale,
                                 activities=activities)
    out_cols = ["code", "date", "variable", "value", "sa_source"]
    if tidy_l.empty:
        out = pd.DataFrame(columns=out_cols)
        out.attrs["hours_scale"] = {}
        return out
    engines: dict = {}
    if sa == "x13":
        tidy_l, engines = _adjust_unadjusted(tidy_l)
    label = {"Y": "oecd", "X": "puremacro", "N": "none"}
    out = pd.DataFrame({
        "code": tidy_l["code"].to_numpy(),
        "date": tidy_l["date"].to_numpy(),
        "variable": tidy_l["name"].to_numpy(),
        "value": tidy_l["value"].to_numpy(),
        "sa_source": [label.get(a, "none") for a in tidy_l["ADJUSTMENT"]],
    })
    out = (out.sort_values(["code", "variable", "date"])
              .reset_index(drop=True)[out_cols])
    out.attrs["hours_scale"] = dict(scales)
    out.attrs["sa_engines"] = engines
    out.attrs["source"] = _LABOR_FLOW.rstrip(",")
    return out


def _rescale_hours(tidy_l: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Put every reference area's hours on a quarterly basis.

    Two of them are not, and nothing in the SDMX message says so: Chile
    publishes hours *per week* and Costa Rica at an *annual rate*, under the
    same ``UNIT_MEASURE`` and ``UNIT_MULT`` as everyone else. Left alone they
    make ``hours / emp`` wrong by 13x and 4x — silently, and in the one
    direction a caller is least likely to sanity-check, because the series
    still moves correctly and only its level is absurd.

    Detection is by ``hours / emp`` against :data:`_HOURS_IMPLAUSIBLE`, and a
    candidate factor is accepted only if it lands the median inside
    :data:`_HOURS_PLAUSIBLE`. A reference area that publishes hours but no
    heads (Canada) cannot be checked and is left alone. Returns the frame plus
    the factor applied per code, which :func:`qna_meta` reports as
    ``hours_scale``.
    """
    scales: dict[str, float] = {}
    if tidy_l.empty or "hours" not in set(tidy_l["name"]):
        return tidy_l, scales

    heads = tidy_l[tidy_l["name"] == "emp"].set_index(["code", "date"])["value"]
    hours = tidy_l[tidy_l["name"] == "hours"].set_index(["code", "date"])["value"]
    # The detection itself is shared with the annual flow, which has the same
    # defect on a different country and a different period: see
    # `puremacro.fetch._hours`. Only the bands and the candidate factors are
    # quarterly, and they stay here with the flow they describe.
    scales = hours_scale_factors(heads, hours,
                                 implausible=_HOURS_IMPLAUSIBLE,
                                 plausible=_HOURS_PLAUSIBLE,
                                 scales=_HOURS_SCALES)
    if scales:
        # every hours column, the by-activity ones included: a country whose
        # total is on the wrong time base has its branches on it too, and
        # rescaling only the total would break `hours - hours_agri`.
        is_hours = tidy_l["name"].isin(
            {n for n in _LABOR_NAMES if n.split("_")[0] == "hours"})
        factors = tidy_l["code"].map(scales).fillna(1.0)
        tidy_l = tidy_l.copy()
        tidy_l.loc[is_hours, "value"] = (tidy_l.loc[is_hours, "value"]
                                         * factors[is_hours])
    return tidy_l, scales


def _adjust_unadjusted(tidy: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Seasonally adjust every series that arrived unadjusted, in place.

    Uses :func:`puremacro.sa.deseasonalize_x13`, whose first choice is the
    X-13ARIMA-SEATS binary and whose fallback is puremacro's own pure-Python
    X-11/ARIMA engine — so this works with no binary installed (iPad, Pyodide,
    a bare container). Returns the frame plus the per-series engine report.
    """
    nsa = tidy["ADJUSTMENT"] != "Y"
    if not nsa.any():
        return tidy, {}
    from ..sa import deseasonalize_x13

    work = tidy.loc[nsa, ["code", "name", "PRICE_BASE", "date", "value"]].copy()
    work["_unit"] = (work["code"] + "|" + work["name"] + "|" + work["PRICE_BASE"])
    # Adjust in logs: national accounts seasonality is multiplicative, and the
    # levels here are strictly positive by construction.
    positive = work["value"] > 0
    work.loc[positive, "value"] = np.log(work.loc[positive, "value"])
    work = work[positive]
    engines: dict = {}
    adj = deseasonalize_x13(work, "value", by="_unit", date_col="date",
                            freq="Q", engines=engines)
    ok = adj.notna()
    tidy = tidy.copy()
    tidy.loc[work.index[ok], "value"] = np.exp(adj[ok].to_numpy())
    tidy.loc[work.index[ok], "ADJUSTMENT"] = "X"      # adjusted by puremacro
    return tidy, engines


def _pick_volume_base(tidy: pd.DataFrame) -> dict[str, str]:
    """Per country: 'L' (chain-linked) when published, else 'Q' (fixed base)."""
    out: dict[str, str] = {}
    have = tidy.groupby(["code", "PRICE_BASE"]).size().unstack(fill_value=0)
    for code, row in have.iterrows():
        if row.get("L", 0) > 0:
            out[code] = "L"
        elif row.get("Q", 0) > 0:
            out[code] = "Q"
    return out


def qna_panel(codes: Iterable[str] | None = None,
              *,
              start: str = "1995",
              assets: bool = False,
              durability: bool = False,
              output: bool = False,
              income: bool = False,
              labor: bool = False,
              labor_activities: bool | Sequence[str] = False,
              hours_rescale: bool = True,
              sa: str = "prefer",
              sa_min_gain: int | None = None,
              real: bool = False,
              long: bool = False,
              refresh: bool = False) -> pd.DataFrame:
    r"""Quarterly national accounts panel: nominal SA levels + implicit deflators.

    Parameters
    ----------
    codes
        ISO3 country codes, e.g. ``["USA", "ESP", "MEX"]``. ``None`` asks the
        OECD for every reference area it publishes (slower, and includes
        aggregates such as ``EA20``).
    start
        First year requested, e.g. ``"1995"``.
    assets
        Also split gross fixed capital formation by asset — ``inv_equip``
        (machinery and equipment), ``inv_struct`` (other buildings and
        structures), ``inv_dwell``, ``inv_ipp`` — with their own deflators.
        Costs one extra SDMX request per chunk of ten countries, and coverage
        is thinner than for the headline aggregates: a country missing an
        asset simply has NaN in that column.
    durability
        Also split household consumption by durability — ``cons_dur``,
        ``cons_semidur``, ``cons_nondur``, ``cons_serv`` — with their own
        deflators. Same cost and same sparsity caveat as ``assets``. Note the
        institutional sector is ``S14`` (households) rather than the ``S1M``
        (households + NPISH) of the headline ``cons_hh``, and that the United
        States and Chile publish no semi-durable category.
    output
        Also join the **output (production) approach**: gross value added by
        ISIC Rev.4 activity plus taxes less subsidies on products, with their
        own deflators, so that
        :math:`Y = \sum_j \text{VA}_j + (D21 - D31) + \text{YA1}`.

        Coverage is 46 of the 49 reference areas — Argentina, Iceland and
        **the United States** publish nothing in this flow at all, the US
        industry accounts being a separate BEA release rather than part of the
        OECD QNA. Of those 46, four (Australia, Canada, Israel, New Zealand)
        publish value added **only in volume terms**, so they arrive with
        ``va_*_real`` columns and no current-price ones, and the output
        identity can be scored in current prices for 42.

        Beware the two memo columns: ``va_mfg`` is inside ``va_ind`` and
        ``va_services`` is the sum of the seven service activities already
        listed, so sum :data:`QNA_VA_ADDITIVE`, not every ``va_*`` column.

        Note also that this flow publishes **its own GDP** (``gdp_output``),
        which is not always the same number as the expenditure flow's ``gdp``
        — see :data:`puremacro.fetch.APPROACH_GDP`.
    income
        Also join the **income approach**: compensation of employees, gross
        operating surplus and mixed income, and taxes less subsidies on
        production and imports, so that :math:`Y = D1 + B2A3G + (D2 - D3)`.

        Coverage is 40 of 49 reference areas, 39 of which publish
        compensation of employees (New Zealand is in the flow but does not).
        Japan publishes GDP and compensation but neither operating surplus nor
        net taxes, so its income identity cannot be closed at all — which
        :func:`~puremacro.fetch.qna_identity` reports as NaN rather than as a
        large fake gap. These series exist only in current prices —
        there is no volume measure of compensation of employees — so they
        carry no deflator and no ``_real`` column even with ``real=True``.
        ``comp_emp / gdp`` is the *unadjusted* labour share; see
        :data:`QNA_INCOME` for why that adjective is load-bearing.
    labor
        Also join the **labour input** the same accounts measure: employment
        and hours worked, each split into employees and the self-employed, on
        the domestic concept — see :data:`QNA_LABOR`. Persons arrive in
        thousands and hours in millions; neither carries a price, so neither
        gets a deflator or a ``_real`` column.

        Coverage is 38 of 49 reference areas for heads and 34 for hours, and
        it is ragged in a way worth knowing before you divide one by the
        other. **The United States**, Japan, Argentina, Brazil, Canada,
        China, Colombia, Indonesia, India, Saudi Arabia and Turkey publish no
        head count here — the first ten of those publish nothing in this flow
        at all, at any level of activity, so there is no key that recovers
        them. Canada publishes hours without heads; Australia, Switzerland,
        Korea, Russia and South Africa publish heads without hours.

        Two reference areas are on a different time base from everyone
        else, and nothing in the SDMX message says so: **Chile** reports hours
        *per week* and **Costa Rica** at an *annual rate*, under the same unit
        and multiplier as everybody. Both are put back on a quarterly basis —
        see ``hours_rescale`` — and ``qna_meta``'s ``hours_scale`` records the
        factor applied.

        Ten reference areas publish this block with no adjusted variant at
        all — Australia, Canada, Switzerland, Chile, Costa Rica, Iceland,
        Mexico, New Zealand, Russia and South Africa — several of which do
        publish adjusted aggregates, so ``sa="x13"`` is what makes their
        labour columns comparable with the rest. The meta column ``sa_labor``
        reports who adjusted this block, separately from the aggregates.

        **Korea** is adjusted at source for total employment but not for its
        employee and self-employed components. Heads and hours are each
        resolved as one family rather than series by series, so Korea falls
        back to the raw series for all three and the decomposition still adds
        up; the price is that ``sa_labor`` reads ``none`` for Korea under the
        default. Taking the adjusted total with raw parts instead would put a
        1.1pp seasonal artefact straight into ``emp_selfemp / emp``.
    labor_activities
        Split that block by ISIC activity as well as by institutional sector.
        ``True`` takes :data:`QNA_LABOR_ACTIVITIES` — agriculture (``_agri``)
        and public administration, education and health (``_public``) — and a
        sequence takes the ISIC codes you name. Requires ``labor=True``, and
        raises if it is off rather than returning a panel without the columns
        you asked for.

        Every stem comes back again per activity, so ``hours`` gains
        ``hours_agri`` and ``hours_public``: twelve extra columns on the
        default. The whole economy (``_T``) is always requested alongside
        them, because the point of the breakdown is ``hours - hours_agri -
        hours_public``, the subtraction that turns a whole-economy :math:`Y/H`
        into the market-sector ratio the United States publishes as its
        nonfarm business sector. A part without its whole is not usable.

        It costs nothing extra — same request, three activities instead of
        one — and 34 reference areas publish hours for all three. Every
        activity of one unit of measure is resolved as a single seasonal
        family, since a source-adjusted total minus a raw part is not a
        subtraction of anything.
    hours_rescale
        Put every reference area's hours on a quarterly basis (default
        ``True``). Chile publishes hours per week and Costa Rica at an annual
        rate, both labelled exactly like everyone else's quarterly figure, so
        ``hours / emp`` reads ~41 and ~2,157 hours per worker against ~430 for
        everyone else — wrong by 13x and 4x, silently, in the direction a
        caller is least likely to check, because the series still moves
        correctly and only its level is absurd.

        Detection is by the level itself: a reference area is rescaled only if
        its median ``hours / emp`` is outside 150–1000 *and* a candidate
        factor lands it inside 250–700. Across the 31 areas publishing both on
        the same basis, every observation ever published sits in 304–572, so
        the band cannot fire on a country that merely works short weeks.
        Canada publishes hours but no heads, so the ratio cannot be formed and
        its hours are left alone. ``qna_meta`` reports the factor applied per
        country in ``hours_scale``: ``1.0`` for taken as published, ``13.0``
        for weekly, ``0.25`` for an annual rate, blank where there are no
        hours. Set ``False`` to see the numbers exactly as the OECD sends them.
    sa
        How to handle seasonal adjustment. ``"prefer"`` (default) takes the
        series the source publishes adjusted and falls back to the unadjusted
        one only where no adjusted series exists. ``"x13"`` instead takes
        whichever of the two covers more quarters and seasonally adjusts it
        here when the source did not — via :func:`puremacro.sa.deseasonalize_x13`,
        which prefers the X-13ARIMA-SEATS binary and falls back to puremacro's
        own pure-Python X-11/ARIMA engine, so it needs no binary installed.
        This brings in the reference areas that publish nothing adjusted at
        all (CHN, SAU), the asset / durability splits that several countries
        publish raw (MEX, JPN, TUR), and — with ``labor=True`` — the ten
        reference areas that publish the whole labour block raw. See
        ``sa_min_gain`` to also trade a short official series for a longer
        raw one.
        Series adjusted here are labelled ``puremacro`` in the metadata's
        ``sa`` / ``sa_detail`` / ``sa_labor`` columns, and the engine that ran
        for each is reported in ``df.attrs["sa_engines"]``.
    sa_min_gain
        With ``sa="x13"``, also prefer the *raw* series over the source's
        adjusted one when it offers at least this many extra quarters, and
        adjust it here. ``None`` (default) never does this: an official
        adjustment is kept wherever one exists, and only blocks the source
        leaves entirely unadjusted are adjusted here. The default is
        conservative on purpose — for US durables the raw volume series starts
        five years earlier, but adjusting it here triples the quarterly
        volatility of US services consumption against the OECD's own
        adjustment, which is a bad trade.
    real
        Also return the volume measures as ``<component>_real`` columns
        (chain-linked where published, fixed-base otherwise).
    long
        Return tidy long form ``code, date, variable, value`` instead of the
        wide ``(code, date) x variable`` frame.
    refresh
        Bypass the on-disk HTTP cache and re-download.

    Returns
    -------
    pandas.DataFrame
        Wide frame with a ``(code, date)`` MultiIndex. Current-price columns
        (``gdp``, ``cons_hh``, ``cons_gov``, ``inv``, ``capform``, ``exports``,
        ``imports``) are in millions of national currency, seasonally and
        calendar adjusted; ``<component>_defl`` are the implicit price
        deflators, 100 in the country's price reference year.
        With ``assets=True`` the GFCF asset split and its deflators are joined
        on. The labour columns added by ``labor=True`` are the exception to
        all of the above: they are counts, not money, in thousands of persons
        and millions of hours, and ``meta``'s ``units`` string does not
        describe them. ``df.attrs["meta"]`` documents currency, adjustment and price base per
        country; ``df.attrs["source"]`` the SDMX dataflow. Empty frame (never
        an exception) if the download fails.

    Examples
    --------
    >>> panel = qna_panel(["USA", "ESP", "MEX"], start="2000")  # doctest: +SKIP
    >>> panel.loc["MEX", ["gdp", "cons_hh", "gdp_defl"]].tail(2)  # doctest: +SKIP
    """
    codes_list = None if codes is None else [c.upper() for c in codes if c]
    raw = _download(codes_list, start, refresh)
    if raw.empty:
        return _empty(long)

    if sa not in ("prefer", "x13"):
        raise ValueError(f"sa must be 'prefer' or 'x13', got {sa!r}")
    # The breakdown rides on the labour block, so asking for it without asking
    # for the block used to be a silent no-op that also skipped the ISIC
    # validation below -- and the CHANGELOG documents the feature as
    # `qna_panel(labor_activities=True)`, which is exactly that call.
    if labor_activities and not labor:
        raise ValueError("labor_activities requires labor=True: the ISIC "
                         "breakdown is of the labour block, which is off")
    _labor_activity_lookup(labor_activities)   # validate the ISIC codes early
    exp_lookup = {(t, sec): name for name, (t, sec, _) in QNA_COMPONENTS.items()}
    tidy = _tidy(raw, exp_lookup, ("TRANSACTION", "SECTOR"), sa=sa,
                 min_gain=sa_min_gain)
    if codes_list is not None and not tidy.empty:
        tidy = tidy[tidy["code"].isin(codes_list)]
    if tidy.empty:
        return _empty(long)
    core_codes = set(tidy["code"].unique())

    # Sibling dataflows: same key shape, joined on (code, date). Each is kept
    # on the headline panel's country set so a stray reference area in one of
    # them cannot widen the index.
    for wanted, flow, registry, dim in (
            (assets, _ASSET_FLOW, QNA_ASSETS, "INSTR_ASSET"),
            (durability, _DURABILITY_FLOW, QNA_DURABILITY, "TRANSACTION"),
            (output, _OUTPUT_FLOW, QNA_ACTIVITIES, ("TRANSACTION", "ACTIVITY")),
            (income, _INCOME_FLOW, QNA_INCOME, ("TRANSACTION", "ACTIVITY"))):
        if not wanted:
            continue
        raw_x = _download_flow(flow, codes_list, start, refresh)
        if raw_x.empty:
            continue
        lookup_x: dict[tuple, str]
        dims_x: tuple[str, ...]
        if isinstance(dim, tuple):
            # Two-dimensional key: the registry stores (TRANSACTION, ACTIVITY).
            lookup_x = {(val[0], val[1]): name for name, val in registry.items()}
            dims_x = dim
        else:
            lookup_x = {(val[0],): name for name, val in registry.items()}
            dims_x = (dim,)
        tidy_x = _tidy(raw_x, lookup_x, dims_x, sa=sa, min_gain=sa_min_gain)
        if not tidy_x.empty:
            tidy = pd.concat([tidy, tidy_x[tidy_x["code"].isin(core_codes)]],
                             ignore_index=True)

    hours_scales: dict[str, float] = {}
    if labor:
        tidy_l, hours_scales = _labor_tidy(codes_list, start, refresh, sa=sa,
                                           sa_min_gain=sa_min_gain,
                                           hours_rescale=hours_rescale,
                                           activities=labor_activities)
        if not tidy_l.empty:
            tidy = pd.concat([tidy, tidy_l[tidy_l["code"].isin(core_codes)]],
                             ignore_index=True)

    engines: dict = {}
    if sa == "x13":
        tidy, engines = _adjust_unadjusted(tidy)

    vol_base = _pick_volume_base(tidy)
    # "_Z" is the labour block: no price, so it is a level by default and can
    # never be mistaken for a volume measure of anything.
    nominal = tidy[tidy["PRICE_BASE"].isin(["V", "_Z"])]
    # `map` on the code column instead of a Python listcomp over every row of
    # the tidy frame, which on a full 49-country panel is hundreds of thousands
    # of dict lookups in the interpreter. A code with no chosen volume base
    # maps to NaN, which compares unequal to every PRICE_BASE -- the same
    # False the `vol_base.get(c) -> None` comparison produced.
    volume = tidy[tidy["PRICE_BASE"] == tidy["code"].map(vol_base)]

    def _wide(part: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if part.empty:
            return pd.DataFrame()
        w = part.pivot_table(index=["code", "date"], columns="name", values="value")
        w.columns = [f"{c}{suffix}" for c in w.columns]
        w.columns.name = None
        return w

    nom_w = _wide(nominal, "")
    vol_w = _wide(volume, "_real")
    if nom_w.empty or vol_w.empty:
        return _empty(long)

    out = nom_w.join(vol_w, how="outer").sort_index()

    # Implicit deflator: 100 * current prices / volume, only where both legs
    # are strictly positive (inventory-inclusive aggregates can cross zero).
    for name in (list(QNA_COMPONENTS) + list(QNA_ASSETS) + list(QNA_DURABILITY)
             + list(QNA_ACTIVITIES) + list(QNA_INCOME) + list(QNA_LABOR)):
        if name in _NO_DEFLATOR or name not in out or f"{name}_real" not in out:
            continue
        num, den = out[name], out[f"{name}_real"]
        ok = (num > 0) & (den > 0)
        out[f"{name}_defl"] = np.where(ok, 100.0 * num / den, np.nan)

    meta = _build_meta(tidy, nominal, vol_base, out, hours_scales)
    if not real:
        out = out.drop(columns=[c for c in out.columns if c.endswith("_real")])
    out = out[[c for c in _ordered_columns(real, assets, durability,
                                          output, income, labor,
                                          labor_activities)
               if c in out.columns]]
    out = out.dropna(how="all")

    if long:
        out = (out.stack().rename("value").reset_index()
                  .rename(columns={"level_2": "variable"}))
    out.attrs["meta"] = tuple(meta.to_dict("records"))
    out.attrs["source"] = _AGENCY_FLOW.rstrip(",")
    out.attrs["sa_engines"] = engines
    return out


def _ordered_columns(real: bool, assets: bool = False,
                     durability: bool = False, output: bool = False,
                     income: bool = False, labor: bool = False,
                     labor_activities: bool | Sequence[str] = False) -> list[str]:
    labor_names = list(QNA_LABOR) if labor else []
    if labor and labor_activities:
        _, acts = _labor_activity_lookup(labor_activities)
        labor_names += [f"{n}_{QNA_LABOR_ACTIVITIES[a]}"
                        for a in acts[1:] for n in QNA_LABOR]
    names = (list(QNA_COMPONENTS)
             + (list(QNA_ASSETS) if assets else [])
             + (list(QNA_DURABILITY) if durability else [])
             + (list(QNA_ACTIVITIES) if output else [])
             + (list(QNA_INCOME) if income else [])
             + labor_names)
    cols = list(names)
    cols += [f"{c}_defl" for c in names if c not in _NO_DEFLATOR]
    if real:
        cols += [f"{c}_real" for c in names]
    return cols


def _sa_label(g: pd.DataFrame) -> str:
    """'oecd' / 'puremacro' / 'mixed' / 'none' for a block of series."""
    if not len(g):
        return ""
    kinds = set(g["ADJUSTMENT"].unique())
    mapped = {"Y": "oecd", "X": "puremacro", "N": "none"}
    labels = {mapped.get(k, "none") for k in kinds}
    if len(labels) == 1:
        return labels.pop()
    return "mixed"


def _implied_ref_year(panel: pd.DataFrame, code: str) -> float:
    """Year in which the GDP deflator sits closest to 100 (the volume base)."""
    if panel is None or "gdp_defl" not in panel.columns or code not in panel.index:
        return np.nan
    s = panel.loc[code, "gdp_defl"].dropna()
    if s.empty:
        return np.nan
    annual = s.groupby(s.index.year).mean()
    return float((annual - 100.0).abs().idxmin())


def _build_meta(tidy: pd.DataFrame, nominal: pd.DataFrame,
                vol_base: dict[str, str],
                panel: pd.DataFrame | None = None,
                hours_scales: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []
    for code, g in nominal.groupby("code"):
        g_core = g[g["name"].isin(QNA_COMPONENTS)]
        g_ast = g[g["name"].isin(list(QNA_ASSETS) + list(QNA_DURABILITY)
                                 + list(QNA_ACTIVITIES) + list(QNA_INCOME))]
        g_lab = g[g["name"].isin(_LABOR_NAMES)]
        gv = tidy[(tidy["code"] == code) & (tidy["PRICE_BASE"] == vol_base.get(code))]
        # `.get` on a missing column returns None, and `to_numeric(None)` is a
        # scalar NaN rather than None -- so the `is not None` guard below never
        # fired and the next line raised AttributeError instead of falling back.
        ref_year = (pd.to_numeric(gv["REF_YEAR_PRICE"], errors="coerce")
                    if "REF_YEAR_PRICE" in gv.columns else None)
        if ref_year is not None and ref_year.notna().any():
            ref = float(ref_year.dropna().iloc[0])
        else:
            # Some reference areas (e.g. AUS) leave REF_YEAR_PRICE blank; the
            # base is still recoverable as the year the deflator equals 100.
            ref = _implied_ref_year(panel, code)
        rows.append({
            "code": code,
            # From the money block only. The labour rows do carry a
            # CURRENCY, but it is the SDMX not-applicable code "_Z", which
            # survives `dropna` and would be reported as this country's
            # currency whenever it sorted first.
            "currency": (g_core["CURRENCY"].dropna().iloc[0]
                         if "CURRENCY" in g_core and g_core["CURRENCY"].notna().any()
                         else ""),
            "units": _UNITS,
            "price_base": vol_base.get(code, ""),
            "price_ref_year": ref,
            # `sa` covers the headline aggregates; the detailed splits
            # (assets, durability) are reported separately because several
            # countries (MEX, JPN, TUR) publish them unadjusted even though
            # their aggregates are adjusted.
            "sa": _sa_label(g_core),
            "sa_detail": _sa_label(g_ast) if len(g_ast) else "",
            "sa_labor": _sa_label(g_lab) if len(g_lab) else "",
            # 1.0 = published quarterly and taken as-is; 13.0 = published
            # weekly, 0.25 = published at an annual rate. Blank where the
            # reference area publishes no hours at all.
            "hours_scale": ((hours_scales or {}).get(code, 1.0)
                            if len(g_lab) and g_lab["name"].str.startswith("hours").any()
                            else ""),
            # Span of the panel, not of the headline block: Luxembourg and
            # South Africa publish the labour block one quarter beyond their
            # expenditure block, and a `last` that stopped at the money rows
            # would understate a panel that does carry that quarter.
            "n_obs": int(g["date"].nunique()),
            "first": g["date"].min(),
            "last": g["date"].max(),
        })
    return pd.DataFrame(rows, columns=_META_COLS).sort_values("code").reset_index(drop=True)


__all__ = ["qna_panel", "qna_labor", "qna_meta", "qna_countries",
           "QNA_COMPONENTS", "QNA_ASSETS", "QNA_AGGREGATES",
           "QNA_DURABILITY", "QNA_ACTIVITIES", "QNA_INCOME",
           "QNA_LABOR", "QNA_LABOR_UNITS",
           "QNA_VA_ADDITIVE", "QNA_VA_MEMO"]
