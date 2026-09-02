r"""Annual national accounts by activity: value added and the labour that made it.

``ana_by_activity(["USA", "JPN", "DEU"])`` returns an annual panel indexed by
``(code, year)`` with, for each ISIC Rev.4 grouping the caller asks for, the
**value added** at current prices and at the previous year's prices, and the
**labour input** — heads and hours, employees and self-employed — from the same
accounts::

    va_total  va_agri  va_public       va_total_py  va_agri_py  va_public_py
    emp       emp_agri emp_public      hours        hours_agri  hours_public
    hours_employees  hours_employees_agri  hours_employees_public  ...

Why this exists, when :func:`puremacro.fetch.qna_panel` already does quarterly
--------------------------------------------------------------------------
Because the quarterly flow does not cover the two economies the business-cycle
literature is written about. ``DSD_NAMAIN1@DF_QNA_BY_ACTIVITY_EMPDC`` returns
**nothing at all** for the United States and Japan — not a missing activity, not
a short sample: zero rows on a key left open in every dimension. Neither does
the quarterly output flow for the United States. So there is no quarterly
national-accounts denominator for either, and dividing their GDP by a labour
input taken from somewhere else is the mismatch this package exists to avoid.

The **annual** tables do cover them, and with the breakdown that matters:
``DSD_NAMAIN10@DF_TABLE3_EMPDC`` publishes hours worked *by employees* per ISIC
grouping — the whole-economy total back to 1970 for the United States and 1980
for Japan, the branches from 1998 and 1994 — and ``DSD_NAMAIN10@DF_TABLE6``
publishes value added by ISIC section for both. That is enough to build a
*market sector* — the whole economy less agriculture and less public
administration, education and health — on the same concept for every country,
which is the only basis on which the United States and Japan can be compared
with anyone at all. Check ``market_first`` in :func:`ana_meta` before you claim
a window: the branches, not the totals, are what bind.

Two mechanics worth knowing before you read a number
----------------------------------------------------
**Aggregates come out by subtraction, never by addition.** The value-added table
publishes ISIC *sections* (A, B, C, … U) rather than the groupings the labour
table uses, so ``public`` here is O+P+Q summed. But the retained sector is
always ``_T`` minus what is removed, because Japan does not publish sections E,
N, S, T or U at all: summing what is kept drops Japan, while subtracting what
is removed keeps it. Same arithmetic where both work, and only one of the two
answers the question for every country.

**Volumes are yours to chain.** This returns current prices (``V``) and the
previous year's prices (``Y``) and stops there, because chain-linked volumes are
not additive and an aggregate built by adding them is wrong in a way that does
not announce itself. Previous-year prices *are* additive within a year, so the
volume growth of any aggregate you build is ``va_X_py[t] / va_X[t-1]`` and
chaining it is one cumulative product. :func:`chain_volume` does that.

Source: OECD SDMX ``DSD_NAMAIN10@DF_TABLE6`` and ``DSD_NAMAIN10@DF_TABLE3_EMPDC``.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import warnings

import numpy as np
import pandas as pd

from ._hours import hours_scale_factors

_VA_FLOW = "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE6,"
_LABOR_FLOW = "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE3_EMPDC,"

#: Column suffix -> (ISIC grouping in the **labour** table, ISIC sections in the
#: **value added** table). The two tables do not use the same code list — the
#: labour one publishes the A*10 groupings, the value-added one publishes
#: sections — so every grouping needs both spellings, and the sections are what
#: gets summed on the value-added side.
#:
#: ``total`` is the whole economy. ``agri`` is agriculture, forestry and fishing,
#: where most of the labour is self-employed and most of the year-to-year output
#: is weather. ``public`` is public administration and defence, education and
#: human health and social work — the block whose value added the SNA *defines*
#: as compensation plus consumption of fixed capital, so that its measured
#: productivity growth is near zero by construction rather than by finding.
#: Removing the two is what turns a whole-economy ratio into the market-sector
#: ratio the United States publishes as its nonfarm business sector.
ANA_ACTIVITIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "total":  ("_T",  ("_T",)),
    "agri":   ("A",   ("A",)),
    "public": ("OTQ", ("O", "P", "Q")),
}

#: column stem -> (SDMX TRANSACTION, SDMX UNIT_MEASURE). Same six series as the
#: quarterly labour block, on the same domestic concept: employment in resident
#: production units, which is the concept GDP is measured on.
#:
#: **Read the coverage before the level.** ``hours`` (all employed persons) is
#: what you want and is not what the United States, Japan or Korea publish by
#: activity: the US publishes it for the total economy only, and Japan does not
#: publish it at all. ``hours_employees`` is published by activity by those
#: two, which is why the market-sector ratio has to be built on it. The
#: substitution is not free — the numerator still contains the output of the
#: self-employed — and :func:`ana_hours_wedge` measures what it costs on the
#: countries that publish both.
#:
#: **Korea is not in that set.** It publishes no hours by activity at all, for
#: any transaction — not all-persons hours, not employee hours — so it has no
#: branch-level denominator of any kind and :func:`ana_meta` reports
#: ``market_n`` of 0 for it. It is a country you cannot put in this panel, not
#: one you put in with a caveat.
ANA_LABOR: dict[str, tuple[str, str]] = {
    "emp":             ("EMP",  "PS"),
    "emp_employees":   ("SAL",  "PS"),
    "emp_selfemp":     ("SELF", "PS"),
    "hours":           ("EMP",  "H"),
    "hours_employees": ("SAL",  "H"),
    "hours_selfemp":   ("SELF", "H"),
}

#: Scale each unit is normalised to, as a power of ten: persons in thousands,
#: hours in millions — the same convention as the quarterly block, so the two
#: can be compared without a factor hiding between them.
ANA_LABOR_UNITS: dict[str, int] = {"PS": 3, "H": 6}

#: Hours per worker per **year** that no real economy falls outside. Every
#: reference area in the annual table sits in 1,400-2,350; this band is
#: deliberately far wider, so it can only fire on an order-of-magnitude error.
#: The quarterly siblings are in :mod:`~puremacro.fetch.oecd_qna_panel`.
_HOURS_IMPLAUSIBLE = (600.0, 4000.0)

#: Where a correction has to land before it is accepted -- tighter than the
#: detection band, so a rescaling that does not produce a believable working
#: year is not applied at all and the series is left as published.
_HOURS_PLAUSIBLE = (1200.0, 2600.0)

#: Factors that turn a mislabelled basis into an annual one. New Zealand
#: publishes these hours per week (52 weeks to a year), labelled exactly like
#: everyone else's annual figure; only the magnitude distinguishes it.
_HOURS_SCALES: dict[float, str] = {52.0: "weekly", 4.0: "quarterly"}

_META_COLS = ["code", "currency", "n_obs", "first", "last", "hours_scale",
              "hours_by_activity", "market_first", "market_last", "market_n"]


def get_sdmx_csv(agency_flow: str, key: str, start_period: str,
                 *, refresh: bool = False) -> pd.DataFrame:
    """Thin indirection over :func:`._oecd_sdmx.get_sdmx_csv`.

    Deferred so ``import puremacro.fetch`` stays free of ``requests`` (Pyodide
    has no scraper stack), and so tests can monkeypatch this name.
    """
    from ._oecd_sdmx import get_sdmx_csv as _impl

    return _impl(agency_flow, key, start_period, refresh=refresh)


def _chunks(codes: Sequence[str] | None, n: int = 8) -> list[str]:
    """The OECD truncates very wide REF_AREA filters — ask in small batches."""
    if codes is None:
        return [""]
    return ["+".join(codes[i:i + n]) for i in range(0, len(codes), n)]


def _empty(chunks_empty: Sequence[str] = ()) -> pd.DataFrame:
    """The no-data frame, carrying the diagnostics that say *why* it is empty.

    ``chunks_empty`` is not optional decoration here: an empty frame is exactly
    the case where the caller cannot tell an OECD rate limit from a batch of
    countries that publish nothing, and it is the case the attribute exists
    for. Returning without it made the documented ``df.attrs["chunks_empty"]``
    raise ``KeyError`` on the one path it was written to serve.
    """
    idx = pd.MultiIndex.from_arrays([[], []], names=["code", "year"])
    out = pd.DataFrame(index=idx)
    out.attrs["meta"] = ()
    out.attrs["chunks_empty"] = tuple(chunks_empty)
    out.attrs["missing_columns"] = ()
    out.attrs["source"] = f"{_VA_FLOW.rstrip(',')} + {_LABOR_FLOW.rstrip(',')}"
    return out


def _numeric(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["value"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce")
    d["year"] = pd.to_numeric(d["TIME_PERIOD"], errors="coerce")
    d = d.dropna(subset=["value", "year"])
    d["year"] = d["year"].astype(int)
    return d.rename(columns={"REF_AREA": "code"})


def _rescale(d: pd.DataFrame, target: dict[str, int] | int) -> pd.Series:
    """Put every row on one scale. ``UNIT_MULT`` is the power of ten published."""
    mult = (pd.to_numeric(d["UNIT_MULT"], errors="coerce")
            if "UNIT_MULT" in d.columns else pd.Series(np.nan, index=d.index))
    if isinstance(target, dict):
        tgt = d["UNIT_MEASURE"].map(target).astype(float)
    else:
        tgt = pd.Series(float(target), index=d.index)
    # A blank multiplier, or a unit with no target, means "already right": each
    # falls back to the other so the factor is 1 rather than NaN, which would
    # delete the series instead of leaving it alone.
    tgt, mult = tgt.fillna(mult), mult.fillna(tgt)
    return d["value"] * np.power(10.0, (mult - tgt).fillna(0.0))


def _fetch_va(codes: Sequence[str] | None, start: str, refresh: bool,
              sections: Sequence[str], empty: list[str] | None = None
              ) -> pd.DataFrame:
    """Value added by ISIC section, current prices and previous year's prices."""
    acts = "+".join(sections)
    parts = []
    for ck in _chunks(codes):
        for price_base, tag in (("V", ""), ("Y", "_py")):
            # 12 dims: FREQ.REF_AREA.SECTOR.COUNTERPART_SECTOR.TRANSACTION.
            # INSTR_ASSET.ACTIVITY.EXPENDITURE.UNIT_MEASURE.PRICE_BASE.
            # TRANSFORMATION.TABLE_IDENTIFIER
            raw = get_sdmx_csv(_VA_FLOW, f"A.{ck}...B1G..{acts}...{price_base}..",
                               start, refresh=refresh)
            d = _numeric(raw) if not raw.empty else raw
            if not d.empty:
                d = d[d["UNIT_MEASURE"] == "XDC"]
            if d.empty:
                # Per *request*, not per chunk. The two price bases are two
                # separate requests against a ~20/hour endpoint, and the
                # likeliest partial failure is losing the second one: that
                # costs every `_py` column and therefore all volume
                # information, while a chunk-level counter still reports
                # nothing empty because the first request succeeded.
                if empty is not None:
                    empty.append(f"va:{ck}:{price_base}")
                continue
            d["value"] = _rescale(d, 6)
            d["tag"] = tag
            parts.append(d[["code", "year", "ACTIVITY", "tag", "value",
                            "CURRENCY"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _fetch_labor(codes: Sequence[str] | None, start: str, refresh: bool,
                 groupings: Sequence[str], empty: list[str] | None = None
                 ) -> pd.DataFrame:
    """Heads and hours by ISIC grouping, domestic concept."""
    acts = "+".join(groupings)
    parts = []
    for ck in _chunks(codes):
        raw = get_sdmx_csv(_LABOR_FLOW, f"A.{ck}.....{acts}.....", start,
                           refresh=refresh)
        d = _numeric(raw) if not raw.empty else raw
        if not d.empty:
            d = d[d["UNIT_MEASURE"].isin(ANA_LABOR_UNITS)]
        if d.empty:
            # A chunk that comes back empty under a rate limit looks exactly
            # like a chunk of countries that publish nothing, and the second
            # is a finding while the first is an outage. Record which chunks
            # these were so the caller can tell them apart.
            if empty is not None:
                empty.append(f"labor:{ck}")
            continue
        d["value"] = _rescale(d, ANA_LABOR_UNITS)
        parts.append(d[["code", "year", "ACTIVITY", "TRANSACTION",
                        "UNIT_MEASURE", "value"]])
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # The key leaves several dimensions open, so one (code, year, activity,
    # transaction, unit) can arrive more than once. The value-added side is
    # deduplicated by `pivot_table(aggfunc="first")`; without the same here a
    # repeated row becomes a duplicated `(code, year)` in the returned panel --
    # silently, and `.loc[code, year]` then hands back two rows.
    return out.drop_duplicates(
        subset=["code", "year", "ACTIVITY", "TRANSACTION", "UNIT_MEASURE"],
        keep="first")


def _rescale_hours(lab: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Put every reference area's annual hours on an annual basis.

    The quarterly flow has this defect for Chile and Costa Rica and
    :func:`puremacro.fetch.oecd_qna_panel._rescale_hours` corrects it; the
    annual flow has it for **New Zealand**, which publishes these hours *per
    week* under the same ``UNIT_MEASURE`` and ``UNIT_MULT`` as everyone else.
    Left alone, ``hours / emp`` reads 33 hours per worker per year against
    1,400-2,350 for every other reference area in the table -- wrong by 52x,
    silently, and only in the level, so the series still moves correctly.

    The detection is :func:`puremacro.fetch._hours.hours_scale_factors`, shared
    with the quarterly flow; only the bands are annual.
    """
    if lab.empty:
        return lab, {}
    tot = lab[lab["ACTIVITY"] == "_T"]
    heads = tot[(tot["TRANSACTION"] == "EMP") & (tot["UNIT_MEASURE"] == "PS")]
    hours = tot[(tot["TRANSACTION"] == "EMP") & (tot["UNIT_MEASURE"] == "H")]
    if heads.empty or hours.empty:
        return lab, {}
    scales = hours_scale_factors(
        heads.set_index(["code", "year"])["value"],
        hours.set_index(["code", "year"])["value"],
        implausible=_HOURS_IMPLAUSIBLE, plausible=_HOURS_PLAUSIBLE,
        scales=_HOURS_SCALES)
    if scales:
        # every hours row, the branches included: a country whose total is on
        # the wrong time base has its branches on it too, and rescaling only
        # the total would break `hours - hours_agri`.
        lab = lab.copy()
        is_h = lab["UNIT_MEASURE"] == "H"
        lab.loc[is_h, "value"] = (lab.loc[is_h, "value"]
                                  * lab.loc[is_h, "code"].map(scales).fillna(1.0))
    return lab, scales


def ana_by_activity(codes: Iterable[str] | None = None, start: str = "1970", *,
                    activities: Iterable[str] | None = None,
                    refresh: bool = False) -> pd.DataFrame:
    """Annual value added and labour input by ISIC activity, one frame.

    ``codes`` are ISO3 reference areas, or ``None`` for everything the OECD
    publishes. ``activities`` selects from :data:`ANA_ACTIVITIES` and defaults
    to all of it; ``total`` is always included, because every aggregate here is
    built by subtracting from the total and a part without its whole is not
    usable.

    Returns a frame indexed by ``(code, year)``. Per activity ``X``:
    ``va_X`` at current prices and ``va_X_py`` at the previous year's prices,
    both in millions of national currency; then every stem of
    :data:`ANA_LABOR` — ``emp_X``, ``hours_X``, ``hours_employees_X`` and so on,
    with the ``total`` suffix dropped so the whole-economy columns keep the
    plain names of the quarterly block.

    Per-country metadata is in ``df.attrs["meta"]``, read it back with
    :func:`ana_meta`. Two columns there decide what you may claim.
    ``hours_by_activity`` is ``False`` when a country publishes total hours for
    the whole economy only (the United States) or not at all (Japan), so the
    market-sector ratio has to be built on ``hours_employees`` — and also when
    a country publishes a branch at least as large as its own total, so the
    retained sector is not positive (Australia). Korea publishes no hours by
    activity at all and gets ``market_n`` of 0 rather than a caveat.
    ``market_first`` / ``market_last`` / ``market_n`` are the years where every
    column that ratio needs exists at once — always shorter than the longest
    column, and the honest sample to quote.

    Never raises on an empty response: an OECD rate limit or an unknown code
    gives an empty frame, as everywhere else in this package. But it does say
    so — ``df.attrs["chunks_empty"]`` lists every request that came back with
    nothing, because under a rate limit that is indistinguishable from a batch
    of countries which publish nothing, and one of the two is an outage you
    want to retry rather than a fact you want to write down.
    """
    codes_list = None if codes is None else [c.upper() for c in codes if c]
    wanted = ["total"] + [a for a in (activities or ANA_ACTIVITIES)
                          if a != "total"]
    unknown = [a for a in wanted if a not in ANA_ACTIVITIES]
    if unknown:
        raise ValueError(f"unknown activity {', '.join(unknown)}; "
                         f"have {', '.join(ANA_ACTIVITIES)}")
    groupings = [ANA_ACTIVITIES[a][0] for a in wanted]
    sections = sorted({s for a in wanted for s in ANA_ACTIVITIES[a][1]})

    empty: list[str] = []
    va = _fetch_va(codes_list, start, refresh, sections, empty)
    lab = _fetch_labor(codes_list, start, refresh, groupings, empty)
    if va.empty and lab.empty:
        return _empty(empty)
    lab, hours_scales = _rescale_hours(lab)

    cols: dict[str, pd.Series] = {}
    currency: dict[str, str] = {}
    if not va.empty:
        wide = va.pivot_table(index=["code", "year", "tag"], columns="ACTIVITY",
                              values="value", aggfunc="first")
        for name in wanted:
            secs = [s for s in ANA_ACTIVITIES[name][1] if s in wide.columns]
            if len(secs) != len(ANA_ACTIVITIES[name][1]):
                continue
            # every section present, or nothing: a partial sum of O+P+Q is a
            # smaller government, not a missing one, and would read as a fact.
            ok = wide[secs].notna().all(axis=1)
            s = wide[secs].sum(axis=1).where(ok).unstack("tag")
            for tag in ("", "_py"):
                if tag in s.columns:
                    cols[f"va_{name}{tag}"] = s[tag]
        cur = va.dropna(subset=["CURRENCY"]).groupby("code")["CURRENCY"].first()
        currency = cur.to_dict()

    hours_by_activity: dict[str, bool] = {}
    if not lab.empty:
        lookup = {v: k for k, v in ANA_LABOR.items()}
        lab = lab.copy()
        lab["stem"] = [lookup.get((t, u)) for t, u
                       in zip(lab["TRANSACTION"], lab["UNIT_MEASURE"])]
        lab = lab.dropna(subset=["stem"])
        act_name = {ANA_ACTIVITIES[a][0]: a for a in wanted}
        lab["act"] = lab["ACTIVITY"].map(act_name)
        lab = lab.dropna(subset=["act"])
        for (stem, act), g in lab.groupby(["stem", "act"]):
            name = stem if act == "total" else f"{stem}_{act}"
            cols[name] = g.set_index(["code", "year"])["value"]
        # which countries publish total hours *by activity*, not just for the
        # whole economy: the difference between "you can build the ratio" and
        # "you have to fall back to employee hours".
        h = lab[lab["stem"] == "hours"]
        for code, g in h.groupby("code"):
            hours_by_activity[code] = set(g["act"]) >= set(wanted)

    if not cols:
        return _empty(empty)
    out = pd.DataFrame(cols)
    out.index.names = ["code", "year"]
    order = ([f"va_{a}{t}" for a in wanted for t in ("", "_py")]
             + [stem if a == "total" else f"{stem}_{a}"
                for a in wanted for stem in ANA_LABOR])
    out = out[[c for c in order if c in out.columns]].sort_index().dropna(how="all")

    # What the market-sector ratio actually spans, which is never what the
    # longest column spans: the US publishes whole-economy employee hours from
    # 1970 and the branches only from 1998, so quoting 1970 would be quoting a
    # column nobody can divide by.
    #
    # `need` is built from `wanted` and NOT filtered by what the panel happens
    # to contain. A required column absent because nobody in this request
    # published it means the ratio cannot be formed at all -- filtering it out
    # of the requirement made that read as "satisfied", so Korea (which
    # publishes no hours by activity) came back with a 13-year market window
    # asked for alone and a 0-year one when Germany joined the same call. The
    # window is a property of the country, never of its company. Reindexing
    # makes an absent column all-NaN, which is all-False, which is the truth.
    need = (["va_total", "va_total_py"]
            + [f"va_{a}{t}" for a in wanted if a != "total" for t in ("", "_py")]
            + ["hours_employees"]
            + [f"hours_employees_{a}" for a in wanted if a != "total"])
    missing = [c for c in need if c not in out.columns]
    have = out.reindex(columns=need).notna().all(axis=1)

    # Presence is not consistency. Australia publishes EMP/H for OTQ identical
    # to its whole-economy total, so `hours - hours_public` is negative in
    # every year -- and the meta certified all 31 of them as usable. A year
    # whose retained sector is negative is not a year the ratio can be quoted
    # on, whatever columns exist.
    have &= _nonneg(out, wanted, hours_by_activity)

    meta = []
    for code, g in out.groupby(level="code"):
        yrs = g.index.get_level_values("year")
        m = have.loc[code]
        m_yrs = m.index[m.to_numpy()]
        meta.append({"code": code, "currency": currency.get(code, ""),
                     "n_obs": len(g), "first": int(yrs.min()),
                     "last": int(yrs.max()),
                     "hours_scale": hours_scales.get(code, 1.0),
                     "hours_by_activity": bool(hours_by_activity.get(code, False)),
                     "market_first": int(m_yrs.min()) if len(m_yrs) else None,
                     "market_last": int(m_yrs.max()) if len(m_yrs) else None,
                     "market_n": int(len(m_yrs))})
    out.attrs["meta"] = tuple(meta)
    out.attrs["chunks_empty"] = tuple(empty)
    #: Columns the market-sector ratio needs that no reference area in this
    #: request published -- the reason a `market_n` of 0 is 0.
    out.attrs["missing_columns"] = tuple(missing)
    out.attrs["hours_scale"] = dict(hours_scales)
    out.attrs["source"] = f"{_VA_FLOW.rstrip(',')} + {_LABOR_FLOW.rstrip(',')}"
    return out


def _nonneg(out: pd.DataFrame, wanted: Sequence[str],
            hours_by_activity: dict[str, bool]) -> pd.Series:
    """``True`` where total minus every removed activity is still non-negative.

    Aggregating by subtraction is what lets Japan through, and it is also what
    lets a mislabelled branch through: if a reference area publishes a part
    equal to (or larger than) its own total, the retained sector is zero or
    negative and the documented recipe divides by it. Australia does exactly
    that for hours in O-Q. Flags the years, warns by name, and -- because the
    flag feeds ``have`` -- keeps them out of the market window.
    """
    ok = pd.Series(True, index=out.index)
    parts = [a for a in wanted if a != "total"]
    if not parts:
        return ok
    checks: list[tuple[str, list[str]]] = [
        ("va_total", [f"va_{a}" for a in parts]),
        ("va_total_py", [f"va_{a}_py" for a in parts]),
    ] + [(stem, [f"{stem}_{a}" for a in parts]) for stem in ANA_LABOR]

    bad: dict[str, set[str]] = {}
    for total, cols in checks:
        if total not in out.columns or any(c not in out.columns for c in cols):
            continue
        resid = out[total] - out[cols].sum(axis=1)
        neg = resid.notna() & (resid < 0)
        if not neg.any():
            continue
        ok &= ~neg
        for code in out.index[neg].get_level_values("code").unique():
            bad.setdefault(str(code), set()).add(total)
            if total.startswith("hours"):
                # the field the docstring tells a caller to check before
                # building the ratio must be False when the ratio is
                # unbuildable, not merely when the columns are absent.
                hours_by_activity[str(code)] = False
    if bad:
        detail = "; ".join(f"{c}: {', '.join(sorted(s))}"
                           for c, s in sorted(bad.items()))
        warnings.warn(
            "the retained sector is negative for some reference areas -- a "
            "published activity is at least as large as its own total, so "
            "`total - parts` cannot be divided by. Those years are excluded "
            f"from market_first/market_last/market_n. Affected: {detail}",
            UserWarning, stacklevel=3)
    return ok


def ana_meta(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-country metadata of an :func:`ana_by_activity` result, as a frame.

    Lives in ``panel.attrs["meta"]`` as a tuple of dicts and not as a DataFrame,
    for the same reason :func:`puremacro.fetch.qna_meta` does: pandas compares
    ``.attrs`` when it propagates them, and a DataFrame in there makes
    ``pd.concat`` on any slice of the panel raise.
    """
    return pd.DataFrame(list(panel.attrs.get("meta", ())), columns=_META_COLS)


def chain_volume(current: pd.Series, previous_year: pd.Series) -> pd.Series:
    """Chain an annual volume index from current and previous-year prices.

    The growth of a volume aggregate between ``t-1`` and ``t`` is the year ``t``
    figure valued at year ``t-1`` prices over the year ``t-1`` figure at its own
    prices — which is why both columns exist and why previous-year prices are
    the ones that can be added up across activities. The result is an index,
    arbitrary in level and exact in growth.

    **A break does not carry.** A missing year, or a missing figure on either
    side, means one link cannot be computed — and a chain that simply skipped it
    would put the level after the hole on the level before it, as if the gap had
    grown at zero. So the index is chained **within each unbroken run** and each
    run gets its own base. Levels are therefore comparable inside a run and not
    across one, which is the honest statement; growth is exact everywhere,
    because ``np.log(index).diff()`` is ``NaN`` at every run boundary rather
    than bridging it. Each break costs the one link inside it and nothing else.

    ``current`` and ``previous_year`` are indexed by ``(code, year)`` or by
    ``year``; the return matches.
    """
    def _one(v: pd.Series, y: pd.Series) -> pd.Series:
        v, y = v.sort_index(), y.sort_index()
        growth = y / v.shift(1)
        gap = pd.Series(v.index, index=v.index).diff() != 1
        # Positivity, not just finiteness. A link of zero or a sign change is
        # perfectly finite, so it used to survive the mask and enter the run --
        # but `np.log` of it is -inf or NaN and `Series.cumsum` *skips* NaN, so
        # the bad link contributed nothing and the level carried straight over
        # it as if it had grown at zero. That is verbatim the failure this
        # docstring says cannot happen. A non-positive link now ends its run.
        growth = growth.mask(gap | ~np.isfinite(growth) | (growth <= 0))
        live = growth.notna().to_numpy()
        idx = pd.Series(np.nan, index=v.index, dtype=float)
        if not live.any():
            return idx
        # maximal runs of consecutive computable links
        edges = np.flatnonzero(np.diff(np.r_[False, live, False]))
        for k, (lo, hi) in enumerate(zip(edges[::2], edges[1::2])):
            # cumsum on the numpy array, which propagates NaN, rather than on
            # the Series, which skips it: belt and braces behind the mask.
            idx.iloc[lo:hi] = np.exp(
                np.cumsum(np.log(growth.iloc[lo:hi].to_numpy())))
            if k == 0 and lo > 0:
                # the first run can also carry its own base year, one row back,
                # because that row is exactly what made its first link valid
                idx.iloc[lo - 1] = 1.0
        return idx

    if isinstance(current.index, pd.MultiIndex):
        prev_codes = set(previous_year.index.get_level_values(0))
        out = {}
        for code in current.index.get_level_values(0).unique():
            cur = current.loc[code]
            # A code present in one series and not the other is a country whose
            # previous-year prices were never published (or were lost to a
            # rate limit); it gets an all-NaN chain rather than a KeyError.
            prev = (previous_year.loc[code] if code in prev_codes
                    else pd.Series(np.nan, index=cur.index, dtype=float))
            out[code] = _one(cur, prev)
        return pd.concat(out, names=current.index.names)
    return _one(current, previous_year)


def ana_hours_wedge(panel: pd.DataFrame, activities: Sequence[str] = ("agri", "public"),
                    start: int | None = None, end: int | None = None) -> pd.DataFrame:
    """What using employee hours instead of all hours costs, country by country.

    The United States and Japan publish hours by activity for **employees**
    only, so a market-sector :math:`Y/H` that includes them has to be built on
    ``hours_employees``. This measures the substitution where it can be
    measured: on every country that publishes both, over the same years and the
    same retained sector, it returns the average annual growth of ``Y/H`` under
    each denominator and the difference between them.

    A small, centred wedge means the substitution moves the level and not the
    trend — the self-employed share is roughly flat, so it differences away. A
    large one is a country whose self-employment is collapsing or exploding, and
    there the two denominators are answering different questions.
    """
    keep = [a for a in activities
            if f"va_{a}" in panel.columns and f"va_{a}_py" in panel.columns]
    if not {"va_total", "va_total_py"} <= set(panel.columns):
        return pd.DataFrame()
    rows = []
    for code, g in panel.groupby(level="code"):
        g = g.droplevel("code").sort_index()
        if start is not None or end is not None:
            g = g.loc[(start or g.index.min()):(end or g.index.max())]
        va = g["va_total"].copy()
        va_py = g["va_total_py"].copy()
        for a in keep:
            va = va - g[f"va_{a}"]
            va_py = va_py - g[f"va_{a}_py"]
        vol = chain_volume(va, va_py)

        # One ratio per denominator, each on a year-complete index so that a
        # missing year stays a hole. `.dropna()` before `.diff()` would close
        # the hole and difference two non-adjacent years -- and across a
        # `chain_volume` run boundary the two runs have different arbitrary
        # bases, so the bridged link is not merely a two-year growth rate but
        # a meaningless one. On a 25-year sample a single bridged link moves
        # the reported mean by more than the entire wedge distribution.
        years = range(int(g.index.min()), int(g.index.max()) + 1)
        ratios: dict[str, pd.Series] = {}
        for tag, stem in (("all", "hours"), ("employees", "hours_employees")):
            if stem not in g.columns:
                continue
            h = g[stem].copy()
            for a in keep:
                col = f"{stem}_{a}"
                if col not in g.columns:
                    h = None
                    break
                h = h - g[col]
            if h is None:
                continue
            # A reference area whose retained sector is non-positive (Australia
            # publishes O-Q hours equal to its own total) makes this log
            # undefined. The mask below is the answer; `errstate` only stops
            # numpy narrating it, since a bare RuntimeWarning is not the named
            # diagnostic this package promises -- `ana_by_activity` already
            # warned about that reference area by name.
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.log(vol / h).reindex(years)
            r = r.mask(~np.isfinite(r))
            # A column that exists on the panel because *another* country
            # publishes it, and is all-NaN for this one, is not a denominator.
            # It must not enter the intersection below, or it would empty the
            # sample of the denominator that does exist.
            if r.notna().any():
                ratios[tag] = r

        row: dict[str, object] = {"code": code}
        # Both denominators are averaged over the SAME years. The whole point
        # is the substitution's cost, and employee hours routinely run decades
        # longer than all-persons hours (that asymmetry is what is being
        # measured), so two independent windows put a growth-regime difference
        # into the wedge and call it a substitution cost.
        common = None
        if len(ratios) == 2:
            d_all, d_emp = ratios["all"].diff(), ratios["employees"].diff()
            common = d_all.notna() & d_emp.notna()
        for tag, r in ratios.items():
            d = r.diff()
            d = (d[common] if common is not None else d).dropna()
            if len(d) < 5:
                continue
            row[f"n_{tag}"] = len(d)
            row[f"g_{tag}"] = float(100 * d.mean())
        if len(row) > 1:
            rows.append(row)
    out = pd.DataFrame(rows).set_index("code") if rows else pd.DataFrame()
    if {"g_employees", "g_all"} <= set(out.columns):
        out["wedge"] = out["g_employees"] - out["g_all"]
    return out
