"""F1 Slice A — each module imports cleanly and exposes iter_<cb>_decision."""
from __future__ import annotations

import importlib

import pytest


_F1A_CONNECTORS = ["bi", "bnm", "bsp", "cbn", "cbe", "cbk"]


@pytest.mark.parametrize("cb", _F1A_CONNECTORS)
def test_module_imports_cleanly(cb):
    importlib.import_module(f"puremacro.narrative.sources.{cb}")


@pytest.mark.parametrize("cb", _F1A_CONNECTORS)
def test_iter_decision_function_exists(cb):
    mod = importlib.import_module(f"puremacro.narrative.sources.{cb}")
    iter_name = f"iter_{cb}_decision"
    assert hasattr(mod, iter_name), (
        f"{cb}.py must export {iter_name} (F1 Slice A contract)"
    )
    assert callable(getattr(mod, iter_name))
