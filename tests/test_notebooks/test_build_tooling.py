"""Fast tests for the notebook build tooling and shared style helpers."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]   # uncertainty_examples/puremacro


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_nbstyle_helpers():
    nb = _load(PROJ / "notebooks" / "_nbstyle.py", "_nbstyle_under_test")
    nb.apply_style()                       # must not raise
    assert len(nb.palette(3)) == 3
    assert len(nb.palette(20)) == 20       # extends beyond the base grays
    assert len(nb.styles(2)) == 2
    assert len(nb.styles(9)) == 9          # cycles beyond the base styles


def test_discover_excludes_underscore():
    bn = _load(PROJ / "tools" / "build_notebooks.py", "build_notebooks_under_test")
    srcs = bn.discover_sources()
    assert all(p.suffix == ".py" for p in srcs)
    assert all(not p.name.startswith("_") for p in srcs)   # _nbstyle.py excluded
    # deterministic build order: sorted by full path (course/ notebooks sort
    # as a block after the top-level ones, so names alone are NOT sorted)
    assert srcs == sorted(srcs)
    assert any(p.parent.name == "course" for p in srcs)    # course/ included


def test_cli_list_runs():
    rc = subprocess.run(
        [sys.executable, str(PROJ / "tools" / "build_notebooks.py"), "--list"],
        cwd=PROJ, capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
