"""BEA CAINC local area personal income — wage & salary disbursements.

Two ingestion paths:

1. ``parse_cainc(path)`` — parse a manually-downloaded county CSV from
   the BEA Regional bulk-download page
   (https://apps.bea.gov/regional/downloadzip.cfm). Either CAINC4 or
   CAINC5N work since both expose LineCode 50 (wages and salaries) and
   60 (supplements).

2. ``fetch_cainc4(...)`` — pull CAINC4 from the BEA API. CAINC4 is the
   right table for historical work (covers 1969–present, single source,
   stable line codes). The newer CAINC5N goes back only to 2001 and is
   NAICS-conditioned, so it is not usable for pre-1990 pre-trends.

Both return the same long schema:
    county_fips (str, 5-digit) | year (int) | wages_salaries | supplements | source
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Union, IO

import pandas as pd

logger = logging.getLogger(__name__)

_LINE_WAGES = 50          # Wages and salaries (thousands of $)
_LINE_SUPPL = 60          # Supplements to wages and salaries
_WANT = {_LINE_WAGES, _LINE_SUPPL}

_API = "https://apps.bea.gov/api/data"
_DEFAULT_TABLE = "CAINC4"
_DEFAULT_YEARS = tuple(range(1969, 2025))   # CAINC4 coverage as of 2026


def parse_cainc(path: Union[str, Path, IO[bytes]]) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"GeoFIPS": str, "LineCode": int})
    raw["county_fips"] = raw["GeoFIPS"].str.replace('"', '').str.strip()
    raw = raw[raw["LineCode"].isin(_WANT)]

    year_cols = [c for c in raw.columns if c.isdigit()]
    melted = raw.melt(
        id_vars=["county_fips", "LineCode"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    melted["year"] = melted["year"].astype(int)
    melted["measure"] = melted["LineCode"].map(
        {_LINE_WAGES: "wages_salaries", _LINE_SUPPL: "supplements"}
    )

    # pivot_table with aggfunc="first" defends against duplicate
    # (county_fips, year, measure) rows in real CAINC dumps.
    out = melted.pivot_table(
        index=["county_fips", "year"],
        columns="measure",
        values="value",
        aggfunc="first",
    ).reset_index()
    out.columns.name = None
    out["county_fips"] = out["county_fips"].astype(object)
    out["source"] = "bea_cainc"
    return out


def _coerce_value(s: pd.Series) -> pd.Series:
    """Strip thousands separators and turn BEA suppression codes into NaN.

    Suppression codes seen in CAINC4: '(D)' (disclosure-suppressed),
    '(NA)' / '(L)' (not available / less than $50). All become NaN.
    """
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _fetch_api_chunk(
    *,
    api_key: str,
    table: str,
    line_code: int,
    years: Iterable[int],
    timeout: int,
) -> pd.DataFrame:
    import requests

    year_str = ",".join(str(y) for y in years)
    params = {
        "UserID": api_key,
        "method": "GetData",
        "datasetname": "Regional",
        "TableName": table,
        "LineCode": str(line_code),
        "GeoFips": "COUNTY",
        "Year": year_str,
        "ResultFormat": "json",
    }
    r = requests.get(_API, params=params, timeout=timeout)
    r.raise_for_status()
    res = r.json().get("BEAAPI", {}).get("Results", {})
    if isinstance(res, list):
        res = res[0] if res else {}
    if "Error" in res:
        raise RuntimeError(
            f"BEA API error for {table} line={line_code} years={year_str}: {res['Error']}"
        )
    data = res.get("Data", [])
    if not data:
        return pd.DataFrame(columns=["county_fips", "year", "value"])
    df = pd.DataFrame(data)
    out = pd.DataFrame({
        "county_fips": df["GeoFips"].astype(str).str.zfill(5),
        "year": df["TimePeriod"].astype(int),
        "value": _coerce_value(df["DataValue"]),
    })
    return out


def fetch_cainc4(
    years: Iterable[int] = _DEFAULT_YEARS,
    *,
    api_key: str | None = None,
    batch_size: int = 5,
    timeout: int = 120,
    table: str = _DEFAULT_TABLE,
) -> pd.DataFrame:
    """Pull CAINC4 county wages and salaries (line 50) + supplements (line 60).

    Returns the same long schema as :func:`parse_cainc`. Years are batched
    to stay within the BEA API per-request size cap (~30k rows).
    """
    from puremacro import credentials
    api_key = credentials.require("bea", explicit=api_key)

    years = sorted(set(int(y) for y in years))
    chunks_by_measure: dict[str, list[pd.DataFrame]] = {
        "wages_salaries": [],
        "supplements": [],
    }
    for line_code, measure in [(_LINE_WAGES, "wages_salaries"), (_LINE_SUPPL, "supplements")]:
        for i in range(0, len(years), batch_size):
            batch = years[i:i + batch_size]
            logger.info("BEA CAINC4 line=%s years=%s..%s", line_code, batch[0], batch[-1])
            chunks_by_measure[measure].append(
                _fetch_api_chunk(
                    api_key=api_key, table=table, line_code=line_code,
                    years=batch, timeout=timeout,
                )
            )

    wages = pd.concat(chunks_by_measure["wages_salaries"], ignore_index=True)
    suppl = pd.concat(chunks_by_measure["supplements"], ignore_index=True)
    wages = wages.rename(columns={"value": "wages_salaries"})
    suppl = suppl.rename(columns={"value": "supplements"})

    out = wages.merge(suppl, on=["county_fips", "year"], how="outer")
    out = out.sort_values(["county_fips", "year"]).reset_index(drop=True)
    out["source"] = f"bea_{table.lower()}"
    return out[["county_fips", "year", "wages_salaries", "supplements", "source"]]
