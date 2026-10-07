"""puremacro replication gallery — reproduce published papers' headline results.

>>> from puremacro.replication import run_all, scorecard
>>> df = scorecard()           # puremacro vs each paper's published target
>>> assert df["passed"].all()

Pyodide-pure: model-based cases solve live; data-backed cases (added by later
families) read vendored snapshots. No statsmodels needed.
"""
from __future__ import annotations

from ._model import CaseResult, ReplicationCase, TargetKind, Tol, run
from .runner import run_all, scorecard
from .romer_romer_2010 import (
    RR2010Result, estimate_rr2010_baseline, load_rr2010_data, load_rr2010_reference,
)

__all__ = [
    "run",
    "run_all",
    "scorecard",
    "ReplicationCase",
    "CaseResult",
    "TargetKind",
    "Tol",
    "RR2010Result",
    "estimate_rr2010_baseline",
    "load_rr2010_data",
    "load_rr2010_reference",
]
