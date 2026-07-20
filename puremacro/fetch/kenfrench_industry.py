"""Ken French 30-industry value-weighted monthly returns.

URL: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_CSV.zip
Output: tidy long-form (date, industry_name, ret).
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pandas as pd

from ._http import cached_get

_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_CSV.zip"
_CACHE = Path(__file__).resolve().parents[2] / "data" / "raw" / "kenfrench" / "industry_portfolios_monthly.csv"


def _download_and_parse() -> pd.DataFrame:
    raw = cached_get(_URL)
    zf = zipfile.ZipFile(BytesIO(raw))
    csv_name = next(n for n in zf.namelist() if n.upper().endswith(".CSV"))
    text = zf.read(csv_name).decode("utf-8", errors="ignore")
    lines = text.splitlines()
    # Find first numeric header row (6-digit YYYYMM in first col).
    header_idx = None
    for i, ln in enumerate(lines):
        fields = [f.strip() for f in ln.split(",")]
        if fields and fields[0].isdigit() and len(fields[0]) == 6:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not locate numeric block in Ken-French CSV")
    col_header = [c for c in lines[header_idx - 1].split(",") if c.strip()]
    block_rows = []
    for ln in lines[header_idx:]:
        fields = [f.strip() for f in ln.split(",")]
        if not fields or not fields[0].isdigit() or len(fields[0]) != 6:
            break
        block_rows.append(fields)
    df = pd.DataFrame(block_rows, columns=["ym"] + col_header)
    df["date"] = pd.to_datetime(df["ym"], format="%Y%m").dt.to_period("M").dt.to_timestamp()
    long = df.melt(id_vars=["date"], value_vars=col_header,
                   var_name="industry_name", value_name="ret")
    long["ret"] = pd.to_numeric(long["ret"], errors="coerce") / 100.0
    return long.dropna().sort_values(["date", "industry_name"]).reset_index(drop=True)


def fetch(refresh: bool = False) -> pd.DataFrame:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    if _CACHE.exists() and not refresh:
        return pd.read_csv(_CACHE, parse_dates=["date"])
    df = _download_and_parse()
    df.to_csv(_CACHE, index=False)
    return df
