"""World Trade Uncertainty Index (WTUI) and World Policy Uncertainty
Index (WPUI) from Ahir-Bloom-Furceri's WUI dataset.

Both series are constructed by counting the frequency of "uncertainty"
(or its variants) **near** a topic-specific keyword set in EIU country
reports — Trade (T6) for WTUI, Policy / Politics (T7) for WPUI. They
share country coverage with the WUI itself (~71 countries, monthly
since 2008-01).

The dataset is cached on the user's Drive at
``MAV/WUI_M_dataset_2025_08.xlsx``. Tab T6 = WTUI, Tab T7 = WPUI; both
are wide (date × country code).

Returns the same long-form schema as other fetchers in this package:
``code, date, variable, value, sa_source, source``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_DEFAULT_DRIVE = Path(
    "/Users/jalonso/Library/CloudStorage/"
    "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV"
)
_DEFAULT_PATH = _DEFAULT_DRIVE / "WUI_M_dataset_2025_08.xlsx"


def _resolve_path(path: str | Path | None) -> Path:
    if path:
        return Path(path)
    if "WUI_EXTRAS_XLSX" in os.environ:
        return Path(os.environ["WUI_EXTRAS_XLSX"])
    return _DEFAULT_PATH


def _wide_to_long(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        # The first column is the date.
        first = df.columns[0]
        df = df.rename(columns={first: "date"})
    df["date"] = pd.to_datetime(df["date"])
    long = df.melt(id_vars="date", var_name="code", value_name="value")
    long = long.dropna(subset=["value"])
    long["variable"] = variable
    long["sa_source"] = "none"
    long["source"] = "Ahir-Bloom-Furceri WUI dataset"
    return long[["code", "date", "variable", "value", "sa_source", "source"]]


def fetch_wui_m_full(path: str | Path | None = None) -> pd.DataFrame:
    """World Uncertainty Index (general, monthly), Tab T1 — 71 countries."""
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"WUI dataset not found at {p}.")
    df = pd.read_excel(p, sheet_name="T1")
    return _wide_to_long(df, "wui_m")


def fetch_wtui_m(path: str | Path | None = None) -> pd.DataFrame:
    """World Trade Uncertainty Index, monthly. Tab T6 of the WUI dataset."""
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"WUI dataset not found at {p}. Set WUI_EXTRAS_XLSX env var or "
            f"pass an explicit path."
        )
    df = pd.read_excel(p, sheet_name="T6")
    return _wide_to_long(df, "wtui_m")


def fetch_wpui_m(path: str | Path | None = None) -> pd.DataFrame:
    """World Policy Uncertainty Index, monthly. Tab T7 of the WUI dataset."""
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"WUI dataset not found at {p}. Set WUI_EXTRAS_XLSX env var or "
            f"pass an explicit path."
        )
    df = pd.read_excel(p, sheet_name="T7")
    return _wide_to_long(df, "wpui_m")


def fetch_extras_m(path: str | Path | None = None) -> pd.DataFrame:
    """Convenience: returns WTUI and WPUI stacked in long form."""
    return pd.concat([fetch_wtui_m(path), fetch_wpui_m(path)], ignore_index=True)


__all__ = ["fetch_wui_m_full", "fetch_wtui_m", "fetch_wpui_m", "fetch_extras_m"]
