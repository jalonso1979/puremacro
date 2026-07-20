"""Regression: the narrative package and its text-uncertainty indices import in
a Pyodide-like environment where the scraper / network deps (bs4, requests,
pdfplumber, pypdf, lxml) are ABSENT.

The source connectors that genuinely need those deps must load LAZILY, so a
browser user can build an uncertainty index from text without the scraper stack.
Each target is imported in a fresh subprocess (clean sys.modules) with the deps
blocked via a meta_path finder, mirroring how they are simply not installed in
Pyodide.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_PROBE = textwrap.dedent(
    """
    import sys
    _BLOCKED = {"bs4", "requests", "pdfplumber", "pypdf", "lxml"}
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _BLOCKED:
                raise ModuleNotFoundError("blocked (simulated Pyodide): " + name)
            return None
    sys.meta_path.insert(0, _Blocker())
    import importlib
    importlib.import_module(sys.argv[1])
    leaked = sorted(m for m in _BLOCKED if m in sys.modules)
    assert not leaked, "leaked scraper deps on import path: " + repr(leaked)
    print("OK")
    """
)

_TARGETS = [
    "puremacro.narrative",
    "puremacro.narrative.indices",
    "puremacro.narrative.indices.lui",
    "puremacro.narrative.indices.epu",
    # The lazy sources/__init__ also unblocks these in the browser (the
    # instrument catalog + the SDMX/fetch package no longer eager-load the
    # scraper connectors). A future connector that eager-imports a scraper dep
    # on one of these paths would (correctly) fail here.
    "puremacro.instruments",
    "puremacro.fetch",
]


@pytest.mark.parametrize("target", _TARGETS)
def test_narrative_imports_without_scraper_deps(target):
    r = subprocess.run(
        [sys.executable, "-c", _PROBE, target],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (
        f"{target} failed to import under simulated Pyodide "
        f"(scraper deps absent):\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    )
