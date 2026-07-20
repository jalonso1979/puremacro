"""F2.0 — AST lint: no direct `os.environ.get("*_API_KEY")` in fetcher /
narrative-scoring / narrative-indices / instruments code. All such
lookups must route through puremacro.credentials.get/require."""
from __future__ import annotations

import ast
import pathlib

import pytest


_TARGET_DIRS = [
    "puremacro/puremacro/fetch",
    "puremacro/puremacro/narrative/scoring",
    "puremacro/puremacro/narrative/indices",
    "puremacro/puremacro/instruments",
]


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "puremacro" / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not find puremacro/ repo root")


def _python_files(root: pathlib.Path):
    for d in _TARGET_DIRS:
        for p in (root / d).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


def _violations(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return list of (lineno, key) for any `os.environ.get("..._API_KEY")`."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
            continue
        target = node.func.value
        is_environ = (
            (isinstance(target, ast.Attribute) and target.attr == "environ"
             and isinstance(target.value, ast.Name) and target.value.id == "os")
            or (isinstance(target, ast.Name) and target.id == "environ")
        )
        if not is_environ:
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        key = node.args[0].value
        if isinstance(key, str) and "_API_KEY" in key:
            out.append((node.lineno, key))
    return out


def test_no_direct_env_get_for_api_keys():
    root = _repo_root()
    offenders: list[str] = []
    for f in _python_files(root):
        for lineno, key in _violations(f):
            offenders.append(f"{f.relative_to(root)}:{lineno}: os.environ.get({key!r})")
    assert not offenders, (
        "F2.0 contract violation: these files read API-key env vars "
        "directly. Route through `puremacro.credentials.require(service)`:\n  "
        + "\n  ".join(offenders)
    )
