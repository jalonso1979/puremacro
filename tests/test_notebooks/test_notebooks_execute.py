"""Slow gate: re-execute every showcase notebook end-to-end (opt-in).

Run with:  pytest -m slow tests/test_notebooks/test_notebooks_execute.py
NOTE: build the whole suite from the CONTROLLER (background), not a waiting
subagent -- see the plan's execution note.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_all_notebooks_execute():
    bn = _load(PROJ / "tools" / "build_notebooks.py", "build_notebooks_slow")
    srcs = bn.discover_sources()
    assert len(srcs) == 62, [p.name for p in srcs]
    rc = bn.main(["--check"])
    assert rc == 0, "one or more notebooks failed to execute (see output above)"
