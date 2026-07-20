"""puremacro validation gallery — re-verify the library against trusted references.

>>> from puremacro.validation import run_all, scorecard
>>> df = scorecard()              # puremacro vs reference, per case
>>> assert df["passed"].all()

Pyodide-pure: ``run_all`` reads frozen goldens for PACKAGE/PUBLISHED cases and
computes ANALYTICAL/SCIPY/INTERNAL references live. No statsmodels needed.
"""
from __future__ import annotations

from ._model import CaseResult, Mechanism, Tol, ValidationCase, run
from .runner import run_all, scorecard

__all__ = [
    "run",
    "run_all",
    "scorecard",
    "ValidationCase",
    "CaseResult",
    "Mechanism",
    "Tol",
]
