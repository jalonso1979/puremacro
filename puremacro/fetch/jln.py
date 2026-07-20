"""Jurado-Ludvigson-Ng macro uncertainty (US, monthly).

Home: https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes

As of 2026-02 the data is distributed as a single zip archive containing
three xlsx files (Macro, Real, Financial).  This fetcher downloads the zip,
extracts MacroUncertaintyToCirculate.xlsx, and emits all three horizons
(h=1, h=3, h=12) as separate long-form rows.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from ._http import cached_get

_JLN_ZIP_URL = (
    "https://www.sydneyludvigson.com/s/MacroFinanceUncertainty_202602Update-3klg.zip"
)
_MACRO_FILENAME = "MacroUncertaintyToCirculate.xlsx"


def fetch(refresh: bool = False) -> pd.DataFrame:
    """Return JLN macro uncertainty in long form.

    Columns: code, date, variable, value, sa_source, source.
    Variables: jln_macro_1, jln_macro_3, jln_macro_12 (horizons h=1,3,12).
    """
    content = cached_get(_JLN_ZIP_URL, refresh=refresh)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        raw_bytes = zf.read(_MACRO_FILENAME)

    raw = pd.read_excel(io.BytesIO(raw_bytes))

    if "Date" not in raw.columns:
        raise ValueError(
            f"JLN macro file missing 'Date' column. Got: {list(raw.columns)}"
        )

    dates = pd.to_datetime(raw["Date"])
    frames = []
    for h in (1, 3, 12):
        col = f"h={h}"
        if col not in raw.columns:
            raise ValueError(
                f"JLN macro file missing column '{col}'. Got: {list(raw.columns)}"
            )
        frames.append(
            pd.DataFrame(
                {
                    "code": "USA",
                    "date": dates,
                    "variable": f"jln_macro_{h}",
                    "value": raw[col].astype(float),
                    "sa_source": "none",
                    "source": "JLN",
                }
            )
        )

    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])
