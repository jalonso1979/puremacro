"""Private ownership and scalar-validation helpers for numerical APIs."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import numpy as np


def real_scalar(value: Any, name: str) -> float:
    """Accept finite real scalars, including zero-dimensional numeric arrays."""
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        answer = float(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(answer):
        raise ValueError(f"{name} must be a finite real scalar")
    return answer


def readonly_array(value: Any, *, dtype: Any = None) -> np.ndarray:
    """Detach storage; numeric arrays have an immutable bytes backing buffer.

    Object arrays in legacy metadata retain their copy-only contract. Numeric
    result arrays never use object storage. Calling setflags on any numeric
    view or its base cannot restore write access.
    """
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        result = deepcopy(array)
        result.setflags(write=False)
        return result
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def real_vector(value: Any, name: str) -> np.ndarray:
    """Validate a nonempty real vector and return a detached working copy."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or not len(raw):
        raise ValueError(f"{name} must be a nonempty real vector")
    answer = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(answer).all():
        raise ValueError(f"{name} must be finite")
    return answer


def freeze(value: Any) -> Any:
    """Own nested metadata; freeze mappings, sequences and numeric arrays."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return readonly_array(value)
    return deepcopy(value)


def thaw(value: Any) -> Any:
    """Return detached ordinary containers for reports and JSON encoders."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.copy()
    return deepcopy(value)
