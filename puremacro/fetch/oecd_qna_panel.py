"""One-call cross-country quarterly national accounts: nominal SA + deflators.

``qna_panel(["USA", "ESP", "MEX"])`` returns a ready-to-use quarterly panel
indexed by ``(code, date)`` with the expenditure side of the national accounts
in **current prices, seasonally and calendar adjusted** (millions of national
currency) plus the matching **implicit price deflators**::

    gdp  cons_hh  cons_gov  inv  capform  exports  imports
    gdp_defl  cons_hh_defl  cons_gov_defl  inv_defl  capform_defl ...

so that ``real = nominal / deflator * 100`` and the expenditure identity

.. math:: Y = C_{hh} + C_{gov} + I + X - M

closes up to the statistical discrepancy. Everything is one SDMX round-trip
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

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

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

_SECTORS = sorted({s for _, s, _ in QNA_COMPONENTS.values()})
_TRANSACTIONS = {t for t, _, _ in QNA_COMPONENTS.values()}

#: components whose volume can legitimately cross zero, so no implicit
#: deflator is published for them (inventory changes swing sign).
_NO_DEFLATOR: frozenset[str] = frozenset()

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
              "sa", "sa_detail", "n_obs", "first", "last"]

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
                   refresh: bool) -> pd.DataFrame:
    """Raw rows from a sibling QNA dataflow (same 13-dim key), one per chunk."""
    parts: list[pd.DataFrame] = []
    for code_key in _code_chunks(codes):
        raw = get_sdmx_csv(flow, f"Q..{code_key}..........", start,
                           refresh=refresh)
        if not raw.empty:
            parts.append(raw)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _tidy(raw: pd.DataFrame, lookup: dict[tuple, str],
          dims: tuple[str, ...], *, sa: str = "prefer",
          min_gain: int | None = None) -> pd.DataFrame:
    """Filter the raw SDMX rows down to one row per (code, date, name, price_base).

    ``lookup`` maps a tuple of ``dims`` values (e.g. ``("B1GQ", "S1")``) to the
    output column name; rows whose dimension tuple is absent are dropped.
    """
    needed = {"PRICE_BASE", "UNIT_MEASURE", "ADJUSTMENT",
              "REF_AREA", "TIME_PERIOD", "OBS_VALUE", *dims}
    if not needed.issubset(raw.columns):
        return pd.DataFrame()

    df = raw[(raw["UNIT_MEASURE"] == "XDC")
             & (raw["PRICE_BASE"].isin(["V", "L", "Q"]))].copy()
    if "ACTIVITY" in df.columns:
        df = df[df["ACTIVITY"].isin(["_T", "_Z"]) | df["ACTIVITY"].isna()]
    if "TRANSFORMATION" in df.columns:
        df = df[(df["TRANSFORMATION"] == "N") | df["TRANSFORMATION"].isna()]
    if df.empty:
        return pd.DataFrame()

    keys = zip(*(df[d] for d in dims))
    df["name"] = [lookup.get(k) for k in keys]
    df = df[df["name"].notna()]
    if df.empty:
        return pd.DataFrame()

    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["value"])
    # Put every series in millions of national currency (UNIT_MULT is the
    # power of ten the published figure carries; 6 = millions).
    mult = pd.to_numeric(df.get("UNIT_MULT", 6), errors="coerce").fillna(6.0)
    df["value"] = df["value"] * np.power(10.0, mult - 6.0)
    df["date"] = _quarter_to_date(df["TIME_PERIOD"])
    df = df.rename(columns={"REF_AREA": "code"})

    keys = ["code", "name", "PRICE_BASE"]
    df["_adj_rank"] = np.where(df["ADJUSTMENT"] == "Y", 0, 1)
    df = df.sort_values(keys + ["date", "_adj_rank"])

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
        df = df.merge(pick, on=keys + ["ADJUSTMENT"], how="inner")
    else:
        # Adjusted-at-source wins outright; unadjusted rows only survive for a
        # (code, name, price_base) block that has no adjusted data at all.
        best = df.groupby(keys, sort=False)["_adj_rank"].transform("min")
        df = df[df["_adj_rank"] == best]

    return df.drop_duplicates(subset=keys + ["date"], keep="first")


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
              sa: str = "prefer",
              sa_min_gain: int | None = None,
              real: bool = False,
              long: bool = False,
              refresh: bool = False) -> pd.DataFrame:
    """Quarterly national accounts panel: nominal SA levels + implicit deflators.

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
    sa
        How to handle seasonal adjustment. ``"prefer"`` (default) takes the
        series the source publishes adjusted and falls back to the unadjusted
        one only where no adjusted series exists. ``"x13"`` instead takes
        whichever of the two covers more quarters and seasonally adjusts it
        here when the source did not — via :func:`puremacro.sa.deseasonalize_x13`,
        which prefers the X-13ARIMA-SEATS binary and falls back to puremacro's
        own pure-Python X-11/ARIMA engine, so it needs no binary installed.
        This brings in the reference areas that publish nothing adjusted at
        all (CHN, SAU) and the asset / durability splits that several
        countries publish raw (MEX, JPN, TUR). See ``sa_min_gain`` to also
        trade a short official series for a longer raw one.
        Series adjusted here are labelled ``puremacro`` in the metadata's
        ``sa`` / ``sa_detail`` columns, and the engine that ran for each is
        reported in ``df.attrs["sa_engines"]``.
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
        on. ``df.attrs["meta"]`` documents currency, adjustment and price base per
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
            (durability, _DURABILITY_FLOW, QNA_DURABILITY, "TRANSACTION")):
        if not wanted:
            continue
        raw_x = _download_flow(flow, codes_list, start, refresh)
        if raw_x.empty:
            continue
        lookup_x = {(v,): name for name, (v, _) in registry.items()}
        tidy_x = _tidy(raw_x, lookup_x, (dim,), sa=sa, min_gain=sa_min_gain)
        if not tidy_x.empty:
            tidy = pd.concat([tidy, tidy_x[tidy_x["code"].isin(core_codes)]],
                             ignore_index=True)

    engines: dict = {}
    if sa == "x13":
        tidy, engines = _adjust_unadjusted(tidy)

    vol_base = _pick_volume_base(tidy)
    nominal = tidy[tidy["PRICE_BASE"] == "V"]
    volume = tidy[[pb == vol_base.get(c) for c, pb in
                   zip(tidy["code"], tidy["PRICE_BASE"])]]

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
    for name in list(QNA_COMPONENTS) + list(QNA_ASSETS) + list(QNA_DURABILITY):
        if name in _NO_DEFLATOR or name not in out or f"{name}_real" not in out:
            continue
        num, den = out[name], out[f"{name}_real"]
        ok = (num > 0) & (den > 0)
        out[f"{name}_defl"] = np.where(ok, 100.0 * num / den, np.nan)

    meta = _build_meta(tidy, nominal, vol_base, out)
    if not real:
        out = out.drop(columns=[c for c in out.columns if c.endswith("_real")])
    out = out[[c for c in _ordered_columns(real, assets, durability)
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
                     durability: bool = False) -> list[str]:
    names = (list(QNA_COMPONENTS)
             + (list(QNA_ASSETS) if assets else [])
             + (list(QNA_DURABILITY) if durability else []))
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
                panel: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for code, g in nominal.groupby("code"):
        g_core = g[g["name"].isin(QNA_COMPONENTS)]
        g_ast = g[g["name"].isin(list(QNA_ASSETS) + list(QNA_DURABILITY))]
        gv = tidy[(tidy["code"] == code) & (tidy["PRICE_BASE"] == vol_base.get(code))]
        ref_year = pd.to_numeric(gv.get("REF_YEAR_PRICE"), errors="coerce")
        if ref_year is not None and ref_year.notna().any():
            ref = float(ref_year.dropna().iloc[0])
        else:
            # Some reference areas (e.g. AUS) leave REF_YEAR_PRICE blank; the
            # base is still recoverable as the year the deflator equals 100.
            ref = _implied_ref_year(panel, code)
        rows.append({
            "code": code,
            "currency": (g["CURRENCY"].dropna().iloc[0]
                         if "CURRENCY" in g and g["CURRENCY"].notna().any() else ""),
            "units": _UNITS,
            "price_base": vol_base.get(code, ""),
            "price_ref_year": ref,
            # `sa` covers the headline aggregates; the detailed splits
            # (assets, durability) are reported separately because several
            # countries (MEX, JPN, TUR) publish them unadjusted even though
            # their aggregates are adjusted.
            "sa": _sa_label(g_core),
            "sa_detail": _sa_label(g_ast) if len(g_ast) else "",
            "n_obs": int(g_core["date"].nunique()),
            "first": g_core["date"].min(),
            "last": g_core["date"].max(),
        })
    return pd.DataFrame(rows, columns=_META_COLS).sort_values("code").reset_index(drop=True)


__all__ = ["qna_panel", "qna_meta", "qna_countries",
           "QNA_COMPONENTS", "QNA_ASSETS", "QNA_AGGREGATES",
           "QNA_DURABILITY"]
