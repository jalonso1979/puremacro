"""Backwards-compat shim: ``_csv_to_instrument`` moved to
:mod:`puremacro.instruments._helpers` in 0.5.2.

This re-export keeps any pre-0.5.2 imports working. New code should
import from the promoted location directly.
"""
from .._helpers import _csv_to_instrument

__all__ = ["_csv_to_instrument"]
