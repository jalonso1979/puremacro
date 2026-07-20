"""Panel loaders and slicing helpers for the teaching layer.

The research library stores the panel in long form (columns:
``code``, ``date``, ``variable``, ``value``). For teaching we prefer
wide form: a ``(code, date)`` MultiIndex with one column per variable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"


def load_long(freq: str = "Q") -> pd.DataFrame:
    """Load ``panel_Q.parquet`` or ``panel_M.parquet`` in long form."""
    if freq.upper() == "Q":
        return pd.read_parquet(_DATA_DIR / "panel_Q.parquet")
    if freq.upper() == "M":
        return pd.read_parquet(_DATA_DIR / "panel_M.parquet")
    raise ValueError(f"freq must be 'Q' or 'M', got {freq!r}")


def long_to_wide(panel_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-form panel to a (code, date) × variable wide frame."""
    wide = (
        panel_long.pivot_table(index=["code", "date"], columns="variable", values="value")
        .sort_index()
    )
    wide.columns.name = None
    return wide


def slice_country(wide: pd.DataFrame, code: str) -> pd.DataFrame:
    """Return the per-country wide slice with a DatetimeIndex."""
    if code not in wide.index.get_level_values("code"):
        raise KeyError(f"country {code!r} not in panel")
    sub = wide.xs(code, level="code").copy()
    sub.index = pd.DatetimeIndex(sub.index)
    return sub.sort_index()


def align_yx(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return `(y, x)` both non-null, inner-joined in time, within `[start, end]`."""
    sub = df[[y_col, x_col]].dropna()
    if start is not None:
        sub = sub[sub.index >= pd.Period(start, freq="Q").to_timestamp()]
    if end is not None:
        sub = sub[sub.index <= pd.Period(end, freq="Q").to_timestamp(how="end")]
    return sub[y_col], sub[x_col]


def load_quarterly(countries: list[str] | None = None) -> pd.DataFrame:
    """Load panel_Q in wide form, optionally filtered to a country list."""
    wide = long_to_wide(load_long("Q"))
    if countries is not None:
        wide = wide.loc[wide.index.get_level_values("code").isin(countries)]
    return wide


def load_monthly(countries: list[str] | None = None) -> pd.DataFrame:
    """Load panel_M in wide form, optionally filtered to a country list."""
    wide = long_to_wide(load_long("M"))
    if countries is not None:
        wide = wide.loc[wide.index.get_level_values("code").isin(countries)]
    return wide
