"""playground/inject_pip_cell.py: every playground notebook installs puremacro
first, and the ones whose code reads parquet also install pyarrow, which left the
base install for the `io` extra in 3.3.0 but ships with the playground's Pyodide."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "inject_pip_cell",
    Path(__file__).resolve().parents[1] / "playground" / "inject_pip_cell.py",
)
inject_pip_cell = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inject_pip_cell)


def _notebook(tmp_path: Path, *code: str) -> Path:
    cells = [{"cell_type": "code", "metadata": {}, "execution_count": None,
              "outputs": [], "source": [c]} for c in code]
    path = tmp_path / "10_lesson.ipynb"
    path.write_text(json.dumps({"cells": cells, "metadata": {},
                                "nbformat": 4, "nbformat_minor": 5}))
    return path


def _first_source(path: Path) -> list[str]:
    return json.loads(path.read_text())["cells"][0]["source"]


def test_plain_notebook_gets_the_plain_install(tmp_path):
    nb = _notebook(tmp_path, "import puremacro\n")
    inject_pip_cell.inject(nb)
    assert _first_source(nb) == ["%pip install puremacro\n"]


def test_parquet_notebook_also_installs_pyarrow(tmp_path):
    nb = _notebook(tmp_path, "import pandas as pd\n",
                   "tr = pd.read_parquet(DATA / 'enoe.parquet')\n")
    inject_pip_cell.inject(nb)
    assert _first_source(nb) == ["%pip install puremacro pyarrow\n"]


def test_injection_is_idempotent(tmp_path):
    nb = _notebook(tmp_path, "import puremacro\n")
    inject_pip_cell.inject(nb)
    inject_pip_cell.inject(nb)
    cells = json.loads(nb.read_text())["cells"]
    assert sum("%pip install puremacro" in "".join(c["source"]) for c in cells) == 1
