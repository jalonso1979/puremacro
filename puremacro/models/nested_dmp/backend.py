"""Back-compat shim: backend dispatch now lives in puremacro._backend.

Kept so existing imports — `from puremacro.models.nested_dmp import backend`
and `from puremacro.models.nested_dmp.backend import ...` — keep working
unchanged. The shared module additionally knows the "cupy" backend.
"""
from __future__ import annotations

from puremacro._backend import (  # noqa: F401
    SUPPORTED,
    available_backends,
    backend_available,
    get_array_namespace,
    to_numpy,
)

__all__ = [
    "SUPPORTED",
    "backend_available",
    "available_backends",
    "get_array_namespace",
    "to_numpy",
]
