"""BIS (Bank for International Settlements) statistics loader.

The BIS publishes cross-country financial statistics — credit-to-GDP
gaps, effective exchange rates, total credit to non-financial sectors,
etc. — at https://www.bis.org/statistics/. Most are quarterly panels
in long-format CSV (one row per country × period).

This v1 loader pulls a single country slice from a single statistical
series. The bulk CSV URL is hardcoded for the credit-to-GDP gap; pass
``csv_path=`` to use a local download or to point at a different BIS
release.

Reference
---------
Bank for International Settlements. BIS Statistics. https://www.bis.org/statistics/
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .._core import Instrument
from ...narrative.sources._http import safe_get_bytes


# BIS publishes the credit-to-GDP gap as part of the totcredit release.
# The exact mirror path has rotated across BIS website redesigns; the
# user can always pass csv_path= with a manual download.
_DEFAULT_MIRRORS = {
    "credit_to_gdp_gap": "https://www.bis.org/statistics/totcredit/credit-gap.csv",
}

_REFERENCE = (
    "Bank for International Settlements. BIS Statistics — Credit-to-GDP "
    "gap. https://www.bis.org/statistics/totcredit.htm"
)


def load(
    *,
    series_id: str = "credit_to_gdp_gap",
    country: str,
    csv_path: str | Path | None = None,
    frequency: str = "Q",
) -> Instrument:
    """Load a BIS country-slice series as an :class:`Instrument`.

    Parameters
    ----------
    series_id : str, default ``"credit_to_gdp_gap"``
        Identifier of the BIS statistical release. Currently only
        ``"credit_to_gdp_gap"`` has a default mirror URL; for other
        series pass ``csv_path=``.
    country : str
        ISO-2 country code matching the ``ISO`` column of the BIS CSV.
        Required (no sensible default).
    csv_path : str | Path | None
        Optional local path to the BIS CSV. When None, attempt the
        default mirror download.
    frequency : str, default ``"Q"``
        Pandas-style frequency code. Most BIS stats are quarterly.

    Returns
    -------
    Instrument
        Country-filtered series, name ``f"bis_{series_id}_{country}"``.
    """
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        mirror = _DEFAULT_MIRRORS.get(series_id)
        if mirror is None:
            raise RuntimeError(
                f"BIS series {series_id!r} has no default mirror; pass "
                f"csv_path= with a local download from "
                f"https://www.bis.org/statistics/."
            )
        try:
            raw = safe_get_bytes(mirror)
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch BIS {series_id!r} from {mirror}. "
                f"Download a local copy from https://www.bis.org/statistics/ "
                f"and pass csv_path=."
            ) from e

    expected_cols = {"ISO", "date", "value"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"BIS CSV missing expected columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}. The default schema is "
            f"long-format with ISO/date/value columns."
        )

    available = sorted(df["ISO"].dropna().unique().tolist())
    if country not in available:
        raise ValueError(
            f"country {country!r} not present in BIS CSV; available: {available}"
        )

    sub = df[df["ISO"] == country].copy()
    sub = sub.dropna(subset=["date", "value"])
    # BIS dates are typically "1999-Q1" or ISO. pd.to_datetime handles "1999Q1"
    # natively; convert "1999-Q1" by stripping the dash.
    # Normalize "1999-Q1", "1999Q1", "1999-Q01" → "1999Q1".
    raw_dates = (
        sub["date"].astype(str)
        .str.replace("-Q", "Q", regex=False)
        .str.replace(r"Q0+(\d)", r"Q\1", regex=True)
    )
    dates = pd.PeriodIndex(raw_dates, freq="Q").to_timestamp(how="start")
    series = pd.Series(
        sub["value"].astype(float).values,
        index=dates,
        name=f"bis_{series_id}_{country}",
    ).sort_index()

    return Instrument(
        series=series,
        name=f"bis_{series_id}_{country}",
        source=f"BIS {series_id} ({country})",
        category="external_csv",
        frequency=frequency,
        metadata={
            "reference": _REFERENCE,
            "series_id": series_id,
            "country": country,
        },
    )


__all__ = ["load"]
