"""Mitchell (2013) International Historical Statistics — government expenditure.

Annual government expenditure as % of GDP, 1900–1979. Not event-based:
each year is converted to 4 equal quarterly anchors. Use as magnitude-
only pre-series before action-based datasets begin (typically 1978).

Source: Mitchell, B.R. (2013). International Historical Statistics.
Palgrave Macmillan.

Implementation timing
---------------------
Mitchell's data is annual and pre-dates the modern budget-execution
calendar; the source provides no implementation-quarter information
distinct from announcement. Events emitted by this loader carry the
default empty ``implementation_profile``, so ``effective_profile``
falls back to ``[(announcement_date, 1.0)]``.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..sources._http import safe_get_bytes
from ..types import NarrativeEvent, NarrativeInstrument

_MIRROR = (
    "https://raw.githubusercontent.com/qmd-co/macro-data/main/"
    "mitchell_2013_govt_expenditure_pct_gdp.csv"
)


def mitchell_csv_to_events(df: pd.DataFrame) -> list[NarrativeEvent]:
    """Coerce Mitchell-format CSV into quarterly NarrativeEvents.

    Expected columns (case-insensitive):
        country    ISO3
        year       int (may come as float from pd.read_csv)
        gdp_share  float in (0, 1] — government expenditure as fraction of GDP.
                   (If values > 1, treated as percentage directly.)
    """
    cols = {c.lower(): c for c in df.columns}
    country_col = cols.get("country") or cols.get("iso3")
    year_col    = cols.get("year")
    val_col     = (cols.get("gdp_share") or cols.get("govt_pct_gdp")
                   or cols.get("g_pct_gdp") or cols.get("value"))

    if country_col is None or year_col is None or val_col is None:
        raise ValueError(
            "Mitchell CSV requires country, year, and gdp_share columns."
        )

    out: list[NarrativeEvent] = []
    for _, row in df.iterrows():
        v = float(row[val_col])
        if pd.isna(v) or v <= 0:
            continue
        # Normalise: treat values > 1 as already in percent.
        pct_gdp = v if v > 1 else v * 100.0
        quarterly_mag = pct_gdp / 4.0

        # Normalize float years from pd.read_csv ("1960.0" → "1960").
        year_val = str(row[year_col]).strip()
        if "." in year_val:
            year_val = year_val.split(".")[0]
        year = int(year_val)
        country = str(row[country_col]).upper()

        for q in range(4):
            date = pd.Timestamp(f"{year}-{q * 3 + 1:02d}-01")
            out.append(NarrativeEvent(
                date=date,
                country=country,
                magnitude=quarterly_mag,
                magnitude_unit="pct_gdp",
                target="both",
                subtarget="general",
                sign=+1,
                confidence=0.6,
                source_text="Mitchell (2013) historical expenditure.",
                source_url="https://link.springer.com/book/9780230005099",
                scoring_method="manual",
                metadata={"replication": "mitchell_2013"},
            ))
    return out


def load(
    *,
    countries: list[str] | None = None,
    csv_path: str | Path | None = None,
    start_year: int = 1960,
    end_year: int = 1979,
) -> dict[str, NarrativeInstrument]:
    """Load Mitchell (2013) historical government expenditure panel.

    Annual government expenditure is distributed equally across 4 quarters.
    All events have confidence=0.6 and sign=+1 (expenditure levels, not shocks).
    Intended as pre-1978 magnitude anchors before action-based datasets begin.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_MIRROR)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch Mitchell 2013 CSV. "
                "Supply csv_path= with the expenditure table."
            ) from e

    # Apply year range filter.
    cols = {c.lower(): c for c in df.columns}
    year_col = cols.get("year")
    if year_col:
        year_series = df[year_col].astype(str).str.split(".").str[0].astype(int)
        df = df[(year_series >= start_year) & (year_series <= end_year)]

    events = mitchell_csv_to_events(df)
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


__all__ = ["load", "mitchell_csv_to_events"]
