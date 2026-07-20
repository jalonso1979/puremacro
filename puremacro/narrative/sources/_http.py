"""Backwards-compat shim: HTTP helpers moved to :mod:`puremacro._http`
in 0.6.0.

This re-export keeps any pre-0.6.0 imports
(e.g. ``from puremacro.narrative.sources._http import safe_get_bytes``)
working. New code should import from the promoted location:
``from puremacro._http import safe_get_bytes``.
"""
from ..._http import (
    USER_AGENT,
    DEFAULT_TIMEOUT,
    safe_get_bytes,
    safe_get_text,
    safe_get_json,
)

__all__ = [
    "USER_AGENT", "DEFAULT_TIMEOUT",
    "safe_get_bytes", "safe_get_text", "safe_get_json",
]
