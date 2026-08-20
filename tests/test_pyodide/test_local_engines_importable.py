# tests/test_pyodide/test_local_engines_importable.py
"""Regression: the local-engine layer and the two LLM call sites import in a
Pyodide-like environment where the optional inference engines (mlx_lm,
llama_cpp) are ABSENT. The engines must load LAZILY (inside methods) so the
HTTP path and the rest of the package import browser-clean.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_PROBE = textwrap.dedent(
    """
    import sys
    _BLOCKED = {"mlx_lm", "llama_cpp", "mlx"}
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _BLOCKED:
                raise ModuleNotFoundError("blocked (simulated Pyodide): " + name)
            return None
    sys.meta_path.insert(0, _Blocker())
    import importlib
    importlib.import_module(sys.argv[1])
    leaked = sorted(m for m in _BLOCKED if m in sys.modules)
    assert not leaked, "leaked engine deps on import path: " + repr(leaked)
    print("OK")
    """
)

_TARGETS = [
    "puremacro.narrative._local_engines",
    "puremacro.narrative.scoring",
    "puremacro.narrative.indices",
]


@pytest.mark.parametrize("target", _TARGETS)
def test_local_engine_imports_without_inference_deps(target):
    r = subprocess.run(
        [sys.executable, "-c", _PROBE, target],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert r.returncode == 0, (
        f"{target} failed to import with engines absent:\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}"
    )
