"""Newey-West HAC standard errors (compatibility alias).

The implementation lives in :mod:`puremacro.inference.hac`; this module
re-exports it so that ``from puremacro.inference.newey_west import
newey_west_se`` keeps working. It used to hold a byte-for-byte duplicate of
the routine that inverted ``X'X`` with a bare ``np.linalg.inv`` (no
singularity diagnostic), so the two copies could drift apart.
"""
from __future__ import annotations

from .hac import newey_west_se

__all__ = ["newey_west_se"]
