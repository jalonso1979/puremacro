"""Universal data ingestion and coercion helpers for puremacro.

Provides robust array and DataFrame coercion across the numerical core
(NumPy / SciPy / Pandas) preserving variable names, time indices, and
frequency metadata into result objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _CoercedData:
    """Standardized output of :func:`coerce_array_or_frame`.

    Attributes
    ----------
    values : np.ndarray
        Underlying numerical array of required dtype and dimension.
    names : list[str]
        Variable / feature names.
    index : Any | None
        Time or observation index if the input was a pandas object, else None.
    freq : str | None
        Inferred or explicit frequency of the time index, if available.
    """

    values: np.ndarray
    names: list[str]
    index: Any | None = None
    freq: str | None = None

    def __len__(self) -> int:
        return len(self.values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape


def coerce_array_or_frame(
    data: Any,
    *,
    required_dim: int | None = None,
    name: str = "var",
    dtype: type = float,
) -> CoercedData:
    """Coerce array-like, Series, or DataFrame into a standardized CoercedData container.

    Parameters
    ----------
    data : array_like, pd.Series, or pd.DataFrame
        Input data.
    required_dim : int, optional
        Required number of dimensions (typically 1 or 2). If 1 and input is a
        single-column DataFrame or 2D array of shape (T, 1), it is flattened to (T,).
        If 2 and input is a 1D Series or 1D array of shape (T,), it is reshaped to (T, 1).
    name : str, default "var"
        Default variable name prefix when input has no column or Series name.
    dtype : type, default float
        Target numeric data type for the values array.

    Returns
    -------
    CoercedData
        Object containing `.values`, `.names`, `.index`, and `.freq`.
    """
    if data is None:
        raise ValueError(f"'{name}' cannot be None.")

    index = None
    freq = None
    names: list[str] = []

    if isinstance(data, pd.DataFrame):
        index = data.index
        freq = getattr(index, "freqstr", getattr(index, "inferred_freq", None))
        names = [str(c) for c in data.columns]
        arr = np.asarray(data.values, dtype=dtype)
        if required_dim == 1:
            if arr.shape[1] == 1:
                arr = arr.ravel()
            else:
                raise ValueError(
                    f"'{name}' DataFrame has {arr.shape[1]} columns, but required_dim=1."
                )
    elif isinstance(data, pd.Series):
        index = data.index
        freq = getattr(index, "freqstr", getattr(index, "inferred_freq", None))
        s_name = str(data.name) if data.name is not None else name
        names = [s_name]
        arr = np.asarray(data.values, dtype=dtype)
        if required_dim == 2:
            arr = arr[:, None]
    else:
        arr = np.asarray(data, dtype=dtype)
        if required_dim == 1 and arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        elif required_dim == 2 and arr.ndim == 1:
            arr = arr[:, None]

        if arr.ndim == 1:
            names = [name]
        elif arr.ndim == 2:
            names = [f"{name}_{i}" for i in range(arr.shape[1])]

    if required_dim is not None and arr.ndim != required_dim:
        raise ValueError(
            f"Expected {required_dim}-dimensional data for '{name}', got shape {arr.shape}."
        )

    return _CoercedData(values=arr, names=names, index=index, freq=freq)
