"""Guajardo-Leigh-Pescatori (2011) AER + IMF EM-extension dataset.

Extends the 17 DGLP advanced economies with 11 emerging markets:
BRA, CHL, COL, IDN, IND, KOR, MEX, POL, THA, TUR, ZAF.

Source: Guajardo, J., Leigh, D., and Pescatori, A. (2011).
"Expansionary Austerity? International Evidence." AER 104(3) + IMF
extension table for EM countries.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument

_MIRROR = (
    "https://raw.githubusercontent.com/qmd-co/macro-data/main/"
    "guajardo_2011_aeipf_em_extension.csv"
)

_EM_COUNTRIES = {
    "BRA", "CHL", "COL", "IDN", "IND", "KOR",
    "MEX", "POL", "THA", "TUR", "ZAF",
}

_DGLP_17 = {
    "AUS", "AUT", "BEL", "CAN", "DNK", "FIN", "FRA", "DEU", "IRL",
    "ITA", "JPN", "NLD", "PRT", "ESP", "SWE", "GBR", "USA",
}
_GUAJARDO_COUNTRIES = _DGLP_17 | _EM_COUNTRIES

_CATEGORY_TARGET = {
    "tax":         ("consumption", "tax"),
    "expenditure": ("both",        "expenditure"),
    "spending":    ("both",        "expenditure"),
    "combined":    ("both",        "general"),
    "general":     ("both",        "general"),
}


def _guajardo_extract_rows(df: pd.DataFrame) -> list[dict]:
    """Normalise raw DataFrame rows into a list of dicts for grouping.

    Expected columns (case-insensitive):
        country          ISO3
        year             int (may come as float from pd.read_csv)
        quarter          int 1-4 (optional)
        action_pct_gdp   float, positive = consolidation
        category / type / kind   "tax" / "expenditure" / "combined" (optional)
        narrative        text excerpt (optional)
    """
    cols = {c.lower(): c for c in df.columns}
    country_col = cols.get("country") or cols.get("iso3")
    year_col    = cols.get("year") or cols.get("date")
    val_col     = (cols.get("action_pct_gdp") or cols.get("action")
                   or cols.get("magnitude"))
    cat_col     = cols.get("category") or cols.get("type") or cols.get("kind")
    text_col    = cols.get("narrative") or cols.get("source_text")

    if country_col is None or year_col is None or val_col is None:
        raise ValueError(
            "Guajardo CSV requires country, year, and action_pct_gdp columns."
        )

    columns_list = list(df.columns)
    country_idx = columns_list.index(country_col)
    year_idx = columns_list.index(year_col)
    val_idx = columns_list.index(val_col)
    cat_idx = columns_list.index(cat_col) if cat_col else None
    text_idx = columns_list.index(text_col) if text_col else None

    out: list[dict] = []
    # Optimize: Use itertuples instead of iterrows for faster iteration
    for row in df.itertuples(index=False, name=None):
        v = float(row[val_idx])
        if v == 0.0 or pd.isna(v):
            continue

        # Normalize float years ("1992.0" → "1992") from pd.read_csv.
        year_val = str(row[year_idx]).strip()
        if "." in year_val:
            year_val = year_val.split(".")[0]
        if year_val.isdigit():
            year = int(year_val)
        else:
            try:
                year = pd.Period(year_val, freq="Q").year
            except Exception:
                year = pd.Timestamp(year_val).year

        raw_cat = str(row[cat_idx]).lower().strip() if cat_idx is not None else "general"
        target, subtarget = _CATEGORY_TARGET.get(raw_cat, ("both", "general"))

        narrative = (
            str(row[text_idx])[:400]
            if text_idx is not None and not pd.isna(row[text_idx])
            else "Guajardo et al. (2011) action-based fiscal consolidation."
        )

        out.append({
            "country": str(row[country_idx]).upper(),
            "year": year,
            "value": v,
            "target": target,
            "subtarget": subtarget,
            "narrative": narrative,
        })
    return out


def _guajardo_group_consecutive(rows: list[dict]) -> list[dict]:
    """Group consecutive years for the same (country, subtarget) into one dict."""
    def _key(r):
        return (r["country"], r["subtarget"])

    sorted_rows = sorted(rows, key=lambda r: (_key(r), r["year"]))
    groups: list[dict] = []
    current: dict | None = None
    for r in sorted_rows:
        k = _key(r)
        if current is None or current["_key"] != k or r["year"] != current["last_year"] + 1:
            if current is not None:
                current.pop("_key", None)
                groups.append(current)
            current = {
                "_key": k, "country": r["country"], "subtarget": r["subtarget"],
                "target": r["target"],
                "first_year": r["year"], "last_year": r["year"],
                "tranches": [(r["year"], r["value"])],
                "narrative": r["narrative"],
            }
        else:
            current["last_year"] = r["year"]
            current["tranches"].append((r["year"], r["value"]))
    if current is not None:
        current.pop("_key", None)
        groups.append(current)
    return groups


def guajardo_csv_to_events(
    df: pd.DataFrame,
    *,
    within_year_rule: str = "uniform",
) -> list[NarrativeEvent]:
    """Coerce a Guajardo-format CSV into NarrativeEvents.

    Consecutive years for the same (country, subtarget) collapse into
    one event with a multi-quarter ``implementation_profile``. All
    events are entered as contractionary (sign=-1).

    Expected columns (case-insensitive):
        country          ISO3
        year             int (may come as float from pd.read_csv)
        quarter          int 1-4 (optional)
        action_pct_gdp   float, positive = consolidation
        category / type / kind   "tax" / "expenditure" / "combined" (optional)
    """
    from ._within_year import annualize_to_quarters

    rows = _guajardo_extract_rows(df)
    groups = _guajardo_group_consecutive(rows)

    out: list[NarrativeEvent] = []
    for g in groups:
        total = sum(v for _, v in g["tranches"])
        if total == 0.0:
            continue
        profile_pairs: list[tuple[pd.Timestamp, float]] = []
        for year, val in g["tranches"]:
            year_weight = val / total
            country_for_fy = g["country"] if within_year_rule == "fiscal_year" else None
            profile_pairs.extend(
                annualize_to_quarters(year, weight=year_weight,
                                      rule=within_year_rule,
                                      country=country_for_fy)
            )
        out.append(NarrativeEvent(
            date=pd.Timestamp(g["first_year"], 1, 1),
            country=g["country"],
            magnitude=abs(total),
            magnitude_unit="pct_gdp",
            target=g["target"],
            subtarget=g["subtarget"],
            sign=-1,
            confidence=1.0,
            source_text=g["narrative"],
            source_url=(
                "https://www.aeaweb.org/articles?id=10.1257/aer.104.3.832"
            ),
            scoring_method="manual",
            metadata={
                "replication": "guajardo_2011",
                "tranche_years": [y for y, _ in g["tranches"]],
                "within_year_rule": within_year_rule,
            },
            implementation_profile=profile_pairs,
        ))
    return out


def load(
    *,
    countries: list[str] | None = None,
    csv_path: str | Path | None = None,
    within_year_rule: str = "uniform",
) -> dict[str, NarrativeInstrument]:
    """Load the Guajardo 2011 panel (17 advanced + 11 EM countries).

    All events are entered as contractionary (sign=-1). The panel
    includes: AUS, AUT, BEL, CAN, DNK, FIN, FRA, DEU, IRL, ITA, JPN,
    NLD, PRT, ESP, SWE, GBR, USA (advanced) + BRA, CHL, COL, IDN, IND,
    KOR, MEX, POL, THA, TUR, ZAF (emerging).

    Parameters
    ----------
    countries : optional ISO3 filter.
    csv_path : optional local path; default attempts mirror download.
    within_year_rule : ``"uniform"`` | ``"front_loaded"`` | ``"fiscal_year"``.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch Guajardo 2011 CSV. Download from AER "
                "replication archive and pass csv_path=."
            ) from e

    events = guajardo_csv_to_events(df, within_year_rule=within_year_rule)
    by_country: dict[str, list[NarrativeEvent]] = {}
    for ev in events:
        if ev.country in _GUAJARDO_COUNTRIES:
            by_country.setdefault(ev.country, []).append(ev)
    if countries is not None:
        wanted = {c.upper() for c in countries}
        by_country = {c: v for c, v in by_country.items() if c in wanted}
    return {
        c: NarrativeInstrument.from_events(evs, target=None, aggregation="sum")
        for c, evs in by_country.items()
    }


__all__ = ["load", "guajardo_csv_to_events"]
