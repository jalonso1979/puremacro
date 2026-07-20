"""F2.3 — coverage assertion: the 8 named connectors all declare
PARSER_SCHEMA_VERSION and call assert_landmarks. Fails the build if
any Slice-A connector regresses."""
from __future__ import annotations

import ast
import importlib
import inspect
import pathlib

import pytest


_SLICE_A_CONNECTORS = (
    "beige_book", "eu_eurlex", "eu_parliament", "us_cbo",
    "fed_minutes", "fed_speeches", "bluesky", "ecb_press",
)


def _module_source(name: str) -> str:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return pathlib.Path(mod.__file__).read_text()


def _has_assert_landmarks_call(name: str) -> bool:
    src = _module_source(name)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "assert_landmarks":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "assert_landmarks":
                return True
    return False


def _has_parser_schema_version(name: str) -> bool:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return hasattr(mod, "PARSER_SCHEMA_VERSION") and isinstance(
        mod.PARSER_SCHEMA_VERSION, int
    )


def test_every_slice_a_connector_has_parser_schema_version():
    missing = [n for n in _SLICE_A_CONNECTORS if not _has_parser_schema_version(n)]
    assert not missing, (
        f"F2.3 contract violation: connectors missing PARSER_SCHEMA_VERSION: {missing}"
    )


def test_every_slice_a_connector_calls_assert_landmarks():
    missing = [n for n in _SLICE_A_CONNECTORS if not _has_assert_landmarks_call(n)]
    assert not missing, (
        f"F2.3 contract violation: connectors not calling assert_landmarks(): {missing}"
    )
