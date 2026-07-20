"""Discover and run replication cases; render a scorecard.

Discovery mirrors the validation gallery: every ``cases_*`` module in this
package exposes ``CASES: list[ReplicationCase]``. Adding a paper family = drop
in a new ``cases_<family>.py`` — no registry edit needed.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Optional

from ._model import CaseResult, ReplicationCase, run


def _discover_cases() -> list[ReplicationCase]:
    import puremacro.replication as pkg

    cases: list[ReplicationCase] = []
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        if mod_info.name.startswith("cases_"):
            mod = importlib.import_module(f"puremacro.replication.{mod_info.name}")
            cases.extend(getattr(mod, "CASES", []))
    return cases


def run_all(family: Optional[str] = None) -> list[CaseResult]:
    cases = _discover_cases()
    if family is not None:
        cases = [c for c in cases if c.family == family]
    return [run(c) for c in cases]


def scorecard(results: Optional[list[CaseResult]] = None):
    """Return a tidy pandas DataFrame summarising replication results."""
    import pandas as pd

    if results is None:
        results = run_all()
    rows = [
        {
            "id": r.id, "family": r.family, "paper": r.paper, "source": r.source,
            "target_kind": r.target_kind, "tol": r.tol, "passed": r.passed,
            "margin": r.margin, "network": r.network, "citation": r.citation,
        }
        for r in results
    ]
    return pd.DataFrame(
        rows,
        columns=["id", "family", "paper", "source", "target_kind", "tol", "passed", "margin", "network", "citation"],
    )
