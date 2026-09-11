"""Global commodity benchmark price series and price indices.

Provides standardized monthly and quarterly commodity benchmarks across:
- Energy: Brent crude, WTI crude, natural gas US Henry Hub, natural gas Europe TTF,
  coal Australia, coal South Africa
- Industrial metals: copper, aluminum, iron ore
- Precious metals: gold, silver
- Agricultural staples & Fertilizers: wheat, maize, rice, soybeans, phosphate rock, DAP, urea
- Commodity Price Indices (base 2010=100): Total, Energy, Non-energy, Agriculture,
  Metals & Minerals, Fertilizers

Output schema conforms to puremacro long-form specification:
[code='WLD', date, variable, value, sa_source, source]
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .wb_pink_sheet import (
    COMMODITY_CATEGORIES,
    fetch_commodity_benchmarks,
    fetch_indices,
    fetch_prices,
)


def fetch_energy_benchmarks(
    *,
    frequency: str = "M",
    start_date: str = "1960-01-01",
    refresh: bool = False,
    workbook_bytes: bytes | None = None,
    file_path: str | Path | None = None,
    variable_suffix: bool | None = None,
) -> pd.DataFrame:
    """Convenience fetcher for standardized energy commodity benchmarks."""
    return fetch_commodity_benchmarks(
        frequency=frequency,
        start_date=start_date,
        categories=["energy"],
        include_indices=False,
        refresh=refresh,
        workbook_bytes=workbook_bytes,
        file_path=file_path,
        variable_suffix=variable_suffix,
    )


def fetch_metal_benchmarks(
    *,
    frequency: str = "M",
    start_date: str = "1960-01-01",
    refresh: bool = False,
    workbook_bytes: bytes | None = None,
    file_path: str | Path | None = None,
    variable_suffix: bool | None = None,
) -> pd.DataFrame:
    """Convenience fetcher for industrial and precious metal commodity benchmarks."""
    return fetch_commodity_benchmarks(
        frequency=frequency,
        start_date=start_date,
        categories=["industrial_metals", "precious_metals"],
        include_indices=False,
        refresh=refresh,
        workbook_bytes=workbook_bytes,
        file_path=file_path,
        variable_suffix=variable_suffix,
    )


def fetch_agriculture_benchmarks(
    *,
    frequency: str = "M",
    start_date: str = "1960-01-01",
    refresh: bool = False,
    workbook_bytes: bytes | None = None,
    file_path: str | Path | None = None,
    variable_suffix: bool | None = None,
) -> pd.DataFrame:
    """Convenience fetcher for agricultural staple and fertilizer benchmarks."""
    return fetch_commodity_benchmarks(
        frequency=frequency,
        start_date=start_date,
        categories=["agriculture", "fertilizers"],
        include_indices=False,
        refresh=refresh,
        workbook_bytes=workbook_bytes,
        file_path=file_path,
        variable_suffix=variable_suffix,
    )


def fetch_commodity_indices(
    *,
    frequency: str = "M",
    start_date: str = "1960-01-01",
    refresh: bool = False,
    workbook_bytes: bytes | None = None,
    file_path: str | Path | None = None,
    variable_suffix: bool | None = None,
) -> pd.DataFrame:
    """Convenience fetcher for World Bank monthly and quarterly commodity indices (2010=100)."""
    return fetch_commodity_benchmarks(
        frequency=frequency,
        start_date=start_date,
        categories=["indices"],
        include_indices=True,
        refresh=refresh,
        workbook_bytes=workbook_bytes,
        file_path=file_path,
        variable_suffix=variable_suffix,
    )


__all__ = [
    "fetch_commodity_benchmarks",
    "fetch_energy_benchmarks",
    "fetch_metal_benchmarks",
    "fetch_agriculture_benchmarks",
    "fetch_commodity_indices",
    "fetch_prices",
    "fetch_indices",
    "COMMODITY_CATEGORIES",
]
