"""F1 Slice A — each of the 6 SE Asia + Africa CB connectors declares
PARSER_SCHEMA_VERSION (adopting the Slice A schema-versioning contract
from inception)."""
from __future__ import annotations

import importlib

import pytest


_F1A_CONNECTORS = [
    "bi", "bnm", "bsp", "cbn", "cbe", "cbk",
]


@pytest.mark.parametrize("name", _F1A_CONNECTORS)
def test_connector_declares_parser_schema_version(name):
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    assert hasattr(mod, "PARSER_SCHEMA_VERSION"), (
        f"{name}.py must declare PARSER_SCHEMA_VERSION (F1 Slice A contract)"
    )
    assert isinstance(mod.PARSER_SCHEMA_VERSION, int)
    assert mod.PARSER_SCHEMA_VERSION >= 1


@pytest.mark.parametrize("name", _F1A_CONNECTORS)
def test_connector_imports_assert_landmarks(name):
    """AST scan: the module imports or references assert_landmarks."""
    import ast
    import pathlib
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
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
        f"(F1 Slice A contract)"
    )
