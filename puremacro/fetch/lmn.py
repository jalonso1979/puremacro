"""Ludvigson-Ma-Ng (2021) real vs financial uncertainty decomposition (US, monthly).

Home: https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes

As of 2026-02 the data is distributed as a single zip archive containing
three xlsx files.  This fetcher downloads the zip and extracts the Real and
Financial uncertainty files, each using only the h=1 horizon, returned as
variables lmn_real and lmn_fin respectively.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from ._http import cached_get

_LMN_ZIP_URL = (
    "https://www.sydneyludvigson.com/s/MacroFinanceUncertainty_202602Update-3klg.zip"
)
_REAL_FILENAME = "RealUncertaintyToCirculate.xlsx"
_FIN_FILENAME = "FinancialUncertaintyToCirculate.xlsx"


def _extract_h1(zf: zipfile.ZipFile, filename: str, var_name: str) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(zf.read(filename)))

    if "Date" not in raw.columns:
        raise ValueError(
            f"LMN file {filename!r} missing 'Date' column. Got: {list(raw.columns)}"
        )

    # Use h=1 horizon; fall back to the first non-Date column if label differs.
    col = "h=1" if "h=1" in raw.columns else raw.columns[1]

    return pd.DataFrame(
        {
            "code": "USA",
            "date": pd.to_datetime(raw["Date"]),
            "variable": var_name,
            "value": raw[col].astype(float),
            "sa_source": "none",
            "source": "LMN",
        }
    ).dropna(subset=["value"])


def fetch(refresh: bool = False) -> pd.DataFrame:
    """Return LMN real and financial uncertainty in long form.

    Columns: code, date, variable, value, sa_source, source.
    Variables: lmn_real (real uncertainty h=1), lmn_fin (financial uncertainty h=1).
    """
    content = cached_get(_LMN_ZIP_URL, refresh=refresh)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        real_df = _extract_h1(zf, _REAL_FILENAME, "lmn_real")
        fin_df = _extract_h1(zf, _FIN_FILENAME, "lmn_fin")

    return pd.concat([real_df, fin_df], ignore_index=True)
