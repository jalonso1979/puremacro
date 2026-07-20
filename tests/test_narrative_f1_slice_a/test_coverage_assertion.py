"""F1 Slice A — coverage assertion: the 6 named connectors all
declare PARSER_SCHEMA_VERSION + call assert_landmarks + call
fetch_with_fallback. Fails the build if any of them regresses."""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest


_F1A_CONNECTORS = ("bi", "bnm", "bsp", "cbn", "cbe", "cbk")


def _module_source(name: str) -> str:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return pathlib.Path(mod.__file__).read_text()


def _has_call(name: str, fn_name: str) -> bool:
    src = _module_source(name)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == fn_name:
                return True
            if isinstance(func, ast.Attribute) and func.attr == fn_name:
                return True
    return False


def _has_parser_schema_version(name: str) -> bool:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return hasattr(mod, "PARSER_SCHEMA_VERSION") and isinstance(
        mod.PARSER_SCHEMA_VERSION, int
    )


def _has_fallback_policy(name: str) -> bool:
    mod = importlib.import_module(f"puremacro.narrative.sources.{name}")
    return hasattr(mod, "FALLBACK_POLICY") and isinstance(
        mod.FALLBACK_POLICY, tuple
    )


def test_every_f1a_connector_has_parser_schema_version():
    missing = [n for n in _F1A_CONNECTORS if not _has_parser_schema_version(n)]
    assert not missing, (
        f"F1 Slice A contract violation: connectors missing "
        f"PARSER_SCHEMA_VERSION: {missing}"
    )


def test_every_f1a_connector_has_fallback_policy():
    missing = [n for n in _F1A_CONNECTORS if not _has_fallback_policy(n)]
    assert not missing, (
        f"F1 Slice A contract violation: connectors missing "
        f"FALLBACK_POLICY: {missing}"
    )


def test_every_f1a_connector_calls_assert_landmarks():
    missing = [n for n in _F1A_CONNECTORS if not _has_call(n, "assert_landmarks")]
    assert not missing, (
        f"F1 Slice A contract violation: connectors not calling "
        f"assert_landmarks: {missing}"
    )


def test_every_f1a_connector_calls_fetch_with_fallback():
    missing = [n for n in _F1A_CONNECTORS if not _has_call(n, "fetch_with_fallback")]
    assert not missing, (
        f"F1 Slice A contract violation: connectors not calling "
        f"fetch_with_fallback: {missing}"
    )
