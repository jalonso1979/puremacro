"""F2.4 — coverage assertion: the 7 fallback connectors all call
fetch_with_fallback(...). Fails the build if any of them regresses to
a direct safe_get_text / fetch_with_playwright call."""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest


_FALLBACK_CONNECTORS = (
    "eu_eurlex", "eu_parliament", "us_cbo",
    "rba", "bok", "riksbank", "sarb",
)


def _module_source(name: str) -> str:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return pathlib.Path(mod.__file__).read_text(encoding="utf-8")


def _has_fetch_with_fallback_call(name: str) -> bool:
    src = _module_source(name)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "fetch_with_fallback":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "fetch_with_fallback":
                return True
    return False


def test_every_fallback_connector_calls_fetch_with_fallback():
    missing = [
        n for n in _FALLBACK_CONNECTORS if not _has_fetch_with_fallback_call(n)
    ]
    assert not missing, (
        f"F2.4 contract violation: these connectors do not call "
        f"fetch_with_fallback(...): {missing}. Either add the call or "
        f"remove the connector from the F2.4 scope list."
    )
