"""Discover and run validation cases; render a scorecard.

Discovery: every module named ``cases_*`` in this package exposes a module-level
``CASES: list[ValidationCase]``. ``run_all`` concatenates them. Adding a
subsystem = dropping in a new ``cases_<name>.py`` — no registry edit needed.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Optional

from ._model import CaseResult, ValidationCase, run


def _discover_cases() -> list[ValidationCase]:
    import puremacro.validation as pkg

    cases: list[ValidationCase] = []
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        if mod_info.name.startswith("cases_"):
            mod = importlib.import_module(f"puremacro.validation.{mod_info.name}")
            cases.extend(getattr(mod, "CASES", []))
    return cases


def run_all(subsystem: Optional[str] = None) -> list[CaseResult]:
    cases = _discover_cases()
    if subsystem is not None:
        cases = [c for c in cases if c.subsystem == subsystem]
    return [run(c) for c in cases]


def scorecard(results: Optional[list[CaseResult]] = None):
    """Return a tidy pandas DataFrame summarising case results."""
    import pandas as pd

    if results is None:
        results = run_all()
    rows = [
        {
            "id": r.id,
            "subsystem": r.subsystem,
            "mechanism": r.mechanism,
            "tol": r.tol,
            "passed": r.passed,
            "max_margin": r.max_margin,
            "citation": r.citation,
        }
        for r in results
    ]
    return pd.DataFrame(
        rows,
        columns=["id", "subsystem", "mechanism", "tol", "passed", "max_margin", "citation"],
    )
