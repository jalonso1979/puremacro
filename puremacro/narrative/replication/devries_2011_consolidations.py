"""Devries et al. (2011) action-based fiscal-consolidation dataset.

Reads the long-format IMF WP appendix CSV where each row is one
fiscal event (in contrast to dglp_2011_consolidations.py which reads
a wide format with separate tax/expenditure columns per country-year).

Source: Devries, P., Guajardo, J., Leigh, D., and Pescatori, A. (2011).
"A New Action-Based Dataset of Fiscal Consolidation." IMF WP 11/158.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument

_MIRROR = (
    "https://raw.githubusercontent.com/qmd-co/macro-data/main/"
    "devries_2011_fiscal_consolidations_long.csv"
)

_SUBTARGET_MAP = {
    "tax": "tax",
    "expenditure": "expenditure",
    "spending": "expenditure",
    "expense": "expenditure",
}
_TARGET_MAP = {
    "tax": "consumption",
    "expenditure": "both",
    "general": "both",
}


def _devries_extract_rows(df: pd.DataFrame) -> list[dict]:
    cols = {c.lower(): c for c in df.columns}
    country_col = cols.get("country") or cols.get("iso3")
    year_col    = cols.get("year") or cols.get("date")
    quarter_col = cols.get("quarter")
    val_col     = (cols.get("action_pct_gdp") or cols.get("action")
                   or cols.get("magnitude") or cols.get("size_pct_gdp"))
    type_col    = cols.get("type") or cols.get("category") or cols.get("kind")
    text_col    = cols.get("narrative") or cols.get("description") or cols.get("note")

    if country_col is None or year_col is None or val_col is None:
        raise ValueError(
            "Devries CSV missing required columns; need country, year/date, "
            "and action_pct_gdp (or action / magnitude / size_pct_gdp)."
        )

    out: list[dict] = []
    for _, row in df.iterrows():
        v = float(row[val_col])
        if v == 0.0 or pd.isna(v):
            continue
        year_val = str(row[year_col]).strip()
        if "." in year_val:
            year_val = year_val.split(".")[0]
        if year_val.isdigit():
            year = int(year_val)
        else:
            try:
                year = pd.Period(year_val, freq="Q").year
            except Exception:
                year = pd.Timestamp(year_val).year
        raw_type = str(row[type_col]).lower().strip() if type_col else "general"
        subtarget = _SUBTARGET_MAP.get(raw_type, "general")
        target    = _TARGET_MAP.get(subtarget, "both")
        narrative = (str(row[text_col])
                     if text_col and not pd.isna(row[text_col])
                     else "Devries (2011) action-based fiscal consolidation.")
        out.append({
            "country": str(row[country_col]).upper(),
            "year": year,
            "value": v,
            "target": target,
            "subtarget": subtarget,
            "narrative": narrative[:400],
        })
    return out


def _devries_group_consecutive(rows: list[dict]) -> list[dict]:
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


def devries_csv_to_events(
    df: pd.DataFrame,
    *,
    within_year_rule: str = "uniform",
) -> list[NarrativeEvent]:
    """Coerce a long-format Devries-style CSV into NarrativeEvents.

    Consecutive years for the same (country, subtarget) collapse into
    one event with a multi-quarter ``implementation_profile``. All
    events are entered as contractionary (sign=-1), matching the
    consolidation-as-tightening convention.
    """
    from ._within_year import annualize_to_quarters

    rows = _devries_extract_rows(df)
    groups = _devries_group_consecutive(rows)

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
                "https://www.imf.org/en/Publications/WP/Issues/2016/12/31/"
                "A-New-Action-Based-Dataset-of-Fiscal-Consolidation-25022"
            ),
            scoring_method="manual",
            metadata={
                "replication": "devries_2011",
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
    """Load the Devries et al. (2011) long-format fiscal-consolidation panel.

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
                "Could not fetch Devries 2011 long-format CSV. "
                "Download the replication archive from IMF WP 11/158 and "
                "pass csv_path=."
            ) from e
    events = devries_csv_to_events(df, within_year_rule=within_year_rule)
    by_country: dict[str, list[NarrativeEvent]] = {}
    for ev in events:
        by_country.setdefault(ev.country, []).append(ev)
    if countries is not None:
        wanted = {c.upper() for c in countries}
        by_country = {c: v for c, v in by_country.items() if c in wanted}
    return {
        c: NarrativeInstrument.from_events(evs, target=None, aggregation="sum")
        for c, evs in by_country.items()
    }


__all__ = ["load", "devries_csv_to_events"]
