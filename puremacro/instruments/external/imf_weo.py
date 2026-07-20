"""IMF World Economic Outlook (WEO) panel loader.

The WEO is published twice yearly as a single tab-delimited file
containing all countries × all macro indicators × all years. The
schema includes ``ISO`` (country code), ``WEO Subject Code`` (indicator
code), and one column per year. Common indicators: ``GGXWDG_NGDP``
(general government gross debt as % of GDP), ``GGXONLB_NGDP``
(primary balance as % of GDP), ``NGDP_RPCH`` (real GDP growth %).

This loader fetches the bulk file, filters to one (indicator, country)
row, and returns the year-by-year time series as an :class:`Instrument`.

Reference
---------
International Monetary Fund. World Economic Outlook Database.
https://www.imf.org/en/Publications/WEO
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .._core import Instrument
from ...narrative.sources._http import safe_get_bytes


# WEO publishes the latest archive at a versioned URL; this is the
# October 2024 release path. The user can pass csv_path= with any
# WEO archive (the schema is stable across releases).
_DEFAULT_MIRROR = (
    "https://www.imf.org/-/media/Files/Publications/WEO/WEO-Database/"
    "2024/October/WEOOct2024all.xls"
)

_REFERENCE = (
    "International Monetary Fund. World Economic Outlook Database. "
    "https://www.imf.org/en/Publications/WEO"
)


def _read_weo_tsv(source) -> pd.DataFrame:
    """Read a WEO tab-separated file, trying UTF-8 then Latin-1.

    Real WEO archives are published in Windows-1252 (country names like
    "Côte d'Ivoire" contain non-UTF-8 bytes). Fall back to Latin-1, which
    is a single-byte encoding that always decodes.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(source, sep="\t", encoding=enc)
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            continue
    raise ValueError("Could not decode WEO file with utf-8 or latin-1.")


def load(
    *,
    indicator: str,
    country: str,
    csv_path: str | Path | None = None,
    frequency: str = "A",
) -> Instrument:
    """Load one (indicator, country) WEO time series as an :class:`Instrument`.

    Parameters
    ----------
    indicator : str
        WEO subject code (e.g. ``"GGXWDG_NGDP"``, ``"GGXONLB_NGDP"``,
        ``"NGDP_RPCH"``).
    country : str
        ISO3 country code (e.g. ``"USA"``, ``"GBR"``).
    csv_path : str | Path | None
        Optional local path to a WEO bulk file (tab-separated). When
        None, attempt the canonical IMF mirror download.
    frequency : str, default ``"A"``
        WEO is published annually.

    Returns
    -------
    Instrument
        Annual series spanning the WEO archive's year range, name
        ``f"imf_weo_{indicator}_{country}"``.
    """
    if csv_path is not None:
        df = _read_weo_tsv(csv_path)
    else:
        try:
            raw = safe_get_bytes(_DEFAULT_MIRROR)
            df = _read_weo_tsv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                "Could not fetch IMF WEO bulk file. Download a copy from "
                "https://www.imf.org/en/Publications/WEO/weo-database/ "
                "and pass csv_path=."
            ) from e

    expected_cols = {"ISO", "WEO Subject Code"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"WEO file missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)[:10]}... "
            f"Expected long-format with ISO and 'WEO Subject Code' columns."
        )

    mask = (df["ISO"] == country) & (df["WEO Subject Code"] == indicator)
    sub = df[mask]
    if sub.empty:
        raise ValueError(
            f"WEO row not found for indicator={indicator!r}, country={country!r}."
        )
    if len(sub) > 1:
        import warnings
        warnings.warn(
            f"WEO file has {len(sub)} rows for indicator={indicator!r}, "
            f"country={country!r}; using first row.",
            UserWarning, stacklevel=2,
        )
        sub = sub.iloc[[0]]

    # Year columns are integer-typed strings ("2015", "2016", ...). Pick
    # them out via regex.
    year_pattern = re.compile(r"^\d{4}$")
    year_cols = [c for c in sub.columns if year_pattern.match(str(c))]
    if not year_cols:
        raise ValueError(
            "No year columns (4-digit integer-named) found in WEO file."
        )

    # Build a Series indexed by year-start date.
    row = sub.iloc[0]
    raw_values = []
    dates = []
    for c in year_cols:
        val = row[c]
        # WEO uses "n/a" or "--" or NaN for missing; pd.read_csv may
        # return either string or NaN. Coerce to float-or-NaN.
        if pd.isna(val):
            num = float("nan")
        elif isinstance(val, str) and val.strip() in ("n/a", "--", ""):
            num = float("nan")
        else:
            try:
                num = float(val)
            except (TypeError, ValueError):
                num = float("nan")
        raw_values.append(num)
        dates.append(pd.Timestamp(f"{c}-01-01"))

    series = pd.Series(
        raw_values,
        index=pd.DatetimeIndex(dates),
        name=f"imf_weo_{indicator}_{country}",
    ).sort_index()

    return Instrument(
        series=series,
        name=f"imf_weo_{indicator}_{country}",
        source=f"IMF WEO {indicator} ({country})",
        category="external_csv",
        frequency=frequency,
        metadata={
            "reference": _REFERENCE,
            "indicator": indicator,
            "country": country,
        },
    )


__all__ = ["load"]
