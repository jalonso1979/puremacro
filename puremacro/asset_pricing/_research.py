"""Validation and reporting shared by the experimental investment kernels."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..inference._results import _ReportMixin

__all__: list[str] = []


def dated_frame(value: Any, name: str, *, missing: bool = False) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise ValueError(f"{name} must be a nonempty dated DataFrame or Series")
    if not isinstance(value.index, pd.DatetimeIndex):
        raise ValueError(f"{name} requires a DatetimeIndex")
    if value.index.hasnans or value.index.has_duplicates or not value.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be unique, nonmissing and increasing")
    if value.columns.has_duplicates or not all(isinstance(c, str) and c for c in value.columns):
        raise ValueError(f"{name} requires unique nonempty string column names")
    out = value.astype(float).copy()
    a = out.to_numpy()
    if np.isinf(a).any() or (not missing and np.isnan(a).any()):
        raise ValueError(f"{name} contains missing or infinite values")
    return out


def integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def finite(value: float, name: str, minimum: float | None = None) -> float:
    x = float(value)
    if not np.isfinite(x) or (minimum is not None and x < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return x


def readonly(value: Any) -> np.ndarray:
    a = np.array(value, dtype=float, copy=True)
    a.setflags(write=False)
    return a


class ResearchReport(_ReportMixin):
    """Tabular report convention; plots are supplied by individual results."""

    def summary(self) -> str:
        return self.to_dataframe().to_string(index=False)

