"""State × month / state × quarter US births series.

Source today: NBER NCHS Natality state×month aggregations 1968-2002,
already produced by the sibling Fertility project's
`climate_fertility/fetch_cdc_births.py` and saved to
`Fertility/data/PROCESSED/nber_state_month_births.csv`. This loader
reads that file, normalizes state_abbr → state_fips, and exposes
monthly and quarterly accessors.

Future extension: CDC WONDER Natality (D27 / D66 datasets) covers
2003-present at state×month. Adding it would require an XML POST to
`https://wonder.cdc.gov/controller/datarequest/...` and is *not*
implemented here — see TODO at the bottom of the module.

Public API:
    fetch_monthly() -> pd.DataFrame[state_fips, date, births_state_m]
    fetch_quarterly() -> pd.DataFrame[state_fips, date, births_state_q]
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


_FERTILITY_DATA = Path.home() / (
    "Library/CloudStorage/GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/Fertility/data/PROCESSED"
)
_NBER_STATE_MONTH = _FERTILITY_DATA / "nber_state_month_births.csv"
_CANONICAL_MONTHLY = _FERTILITY_DATA / "state_month_births.csv"

_STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56",
}


def _load_nber_state_month(path: Path | None = None) -> pd.DataFrame:
    """Load Fertility/.../nber_state_month_births.csv → (state_fips, date, births)."""
    if path is None:
        path = _NBER_STATE_MONTH
    if not path.exists():
        logger.warning("state-month source not found: %s", path)
        return pd.DataFrame(columns=["state_fips", "date", "births_state_m"])
    raw = pd.read_csv(path)
    raw["state_fips"] = raw["state"].map(_STATE_ABBR_TO_FIPS)
    unmapped = raw["state_fips"].isna().sum()
    if unmapped > 0:
        logger.warning("dropping %d rows with unmapped state codes: %s",
                       unmapped, raw.loc[raw["state_fips"].isna(), "state"].unique())
        raw = raw.dropna(subset=["state_fips"])
    raw["date"] = pd.to_datetime(
        {"year": raw["year"], "month": raw["month"], "day": 1}
    )
    out = raw[["state_fips", "date", "births"]].rename(columns={"births": "births_state_m"})
    return out.sort_values(["state_fips", "date"]).reset_index(drop=True)


def _load_socrata_state_month(*, refresh: bool = False) -> pd.DataFrame:
    """Pull state × month from CDC Socrata VSRR provisional dataset (2023+).

    Cached at data/raw/cdc_socrata_natality/state_month_provisional.csv.
    Network call is single-page-iterating (the dataset is small, ~2k rows).
    """
    cache_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "cdc_socrata_natality"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / "state_month_provisional.csv"
    if cp.exists() and not refresh:
        return pd.read_csv(cp, dtype={"state_fips": str}, parse_dates=["date"])
    try:
        from . import cdc_socrata_natality
        df = cdc_socrata_natality.fetch_state_month()
    except Exception as e:
        logger.warning("CDC Socrata fetch failed: %s", e)
        return pd.DataFrame(columns=["state_fips", "date", "births_state_m"])
    if not df.empty:
        df.to_csv(cp, index=False)
    return df


def fetch_monthly(
    *, write_artifact: bool = True, include_socrata: bool = True, refresh: bool = False,
) -> pd.DataFrame:
    """Return a (state_fips, date, births_state_m) frame.

    Sources:
      - NBER NCHS Natality state×month aggregates (1968-2002).
      - CDC Socrata `hmz2-vwda` (VSRR provisional state×month, 2023-current)
        when ``include_socrata=True``.

    There is a programmatic **gap from 2003 through 2022** because CDC
    WONDER's API forbids state-level grouping for natality and the NCHS
    All-County Natality file is RDC-restricted. Manual WONDER web export
    is the workaround for that window.
    """
    nber = _load_nber_state_month()
    parts = [nber] if not nber.empty else []
    if include_socrata:
        try:
            socrata = _load_socrata_state_month(refresh=refresh)
            if not socrata.empty:
                if not nber.empty:
                    nber_years = set(nber["date"].dt.year.unique())
                    socrata = socrata[~socrata["date"].dt.year.isin(nber_years)]
                parts.append(socrata)
        except Exception as e:
            logger.warning("Socrata append failed: %s — using NBER only", e)
    if not parts:
        return pd.DataFrame(columns=["state_fips", "date", "births_state_m"])
    df = pd.concat(parts, ignore_index=True).sort_values(
        ["state_fips", "date"]
    ).reset_index(drop=True)

    if write_artifact and _FERTILITY_DATA.exists() and not df.empty:
        out = df.rename(columns={"births_state_m": "births"}).copy()
        out["year"] = out["date"].dt.year
        out["month"] = out["date"].dt.month
        out = out[["state_fips", "year", "month", "births"]]
        _CANONICAL_MONTHLY.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(_CANONICAL_MONTHLY, index=False)
        logger.info("wrote %s (%d rows, %d-%d)",
                    _CANONICAL_MONTHLY, len(out), out["year"].min(), out["year"].max())
    return df


def fetch_yearly(*, refresh: bool = False) -> pd.DataFrame:
    """State × year live births, 1968-2025.

    Combines NBER state-month aggregations (1968-2002, summed over 12
    months) with Census PEP state totals (2001-2025; partial vintage
    base years filtered out). Where both sources have a year, Census
    wins because it explicitly publishes calendar-year totals; NBER
    sums could differ slightly due to monthly imputations.
    """
    parts = []
    nber = _load_nber_state_month()
    if not nber.empty:
        n = nber.copy()
        n["year"] = n["date"].dt.year
        n_year = n.groupby(["state_fips", "year"], as_index=False)["births_state_m"].sum()
        n_year = n_year.rename(columns={"births_state_m": "births_state_a"})
        n_year["source"] = "nber_state_month_aggregated"
        parts.append(n_year)

    try:
        from . import census_pep_births
        pep = census_pep_births.fetch_state_year(refresh=refresh)
        pep = pep.rename(columns={"births": "births_state_a"})
        if parts:
            pep_years = set(pep["year"].unique())
            parts[0] = parts[0][~parts[0]["year"].isin(pep_years)]
        parts.append(pep)
    except Exception as e:
        logger.warning("Census PEP state-year fetch failed: %s", e)

    if not parts:
        return pd.DataFrame(columns=["state_fips", "year", "births_state_a", "source"])
    df = pd.concat(parts, ignore_index=True)
    df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)
    return df.sort_values(["state_fips", "year"]).reset_index(drop=True)


def fetch_quarterly() -> pd.DataFrame:
    """Aggregate state × month → state × quarter.

    Drops quarters with fewer than 3 observed months so partial-year edges
    don't pollute the panel.
    """
    m = fetch_monthly(write_artifact=False)
    if m.empty:
        return pd.DataFrame(columns=["state_fips", "date", "births_state_q"])
    m = m.copy()
    m["period"] = m["date"].dt.to_period("Q")
    counts = m.groupby(["state_fips", "period"], as_index=False).agg(
        births_state_q=("births_state_m", "sum"),
        n_months=("births_state_m", "size"),
    )
    counts = counts[counts["n_months"] == 3].drop(columns="n_months")
    counts["date"] = counts["period"].dt.to_timestamp(how="start")
    return counts[["state_fips", "date", "births_state_q"]].reset_index(drop=True)


# TODO (2003-present extension): wrap CDC WONDER Natality D66 (2007+) and
# D27 (2003-2006) via XML POST to
#   https://wonder.cdc.gov/controller/datarequest/{D66,D27}
# parsing the tab-delimited table (rows 0..N then "---" then totals).
# Set `accept_datause_restrictions=true` in the request XML. Append the
# returned rows to _CANONICAL_MONTHLY.
