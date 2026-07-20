"""F2.3 — per-connector PARSER_SCHEMA_VERSION + assert_landmarks call."""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest


# All 8 connectors in the Slice A rollout. Tasks 17 + 18 add the
# rollouts; this test grows green incrementally as each is committed.
_CONNECTORS = [
    "beige_book", "eu_eurlex", "eu_parliament", "us_cbo",
    "fed_minutes", "fed_speeches", "bluesky", "ecb_press",
]


def _module_for(name: str):
    return importlib.import_module(f"puremacro.narrative.sources.{name}")


@pytest.mark.parametrize("name", _CONNECTORS)
def test_connector_declares_parser_schema_version(name):
    mod = _module_for(name)
    assert hasattr(mod, "PARSER_SCHEMA_VERSION"), (
        f"{name}.py must declare PARSER_SCHEMA_VERSION (F2.3 contract)"
    )
    assert isinstance(mod.PARSER_SCHEMA_VERSION, int)
    assert mod.PARSER_SCHEMA_VERSION >= 1


@pytest.mark.parametrize("name", _CONNECTORS)
def test_connector_imports_assert_landmarks(name):
    """AST scan: the module must import or reference `assert_landmarks`."""
    mod = _module_for(name)
    src = pathlib.Path(mod.__file__).read_text()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(alias.name == "assert_landmarks" for alias in node.names):
                found = True
                break
        if isinstance(node, ast.Name) and node.id == "assert_landmarks":
            found = True
            break
    assert found, (
        f"{name}.py must import or reference `assert_landmarks` "
        f"(F2.3 contract)"
    )
