"""Prepend a `%pip install puremacro` code cell to every .ipynb passed on argv.

In JupyterLite each notebook gets a fresh kernel, so each showcase notebook must
install puremacro before its `import puremacro...` cells. We inject the cell at
build time so the canonical notebooks/ sources stay clean.

The install resolves offline because every base dependency ships with Pyodide
(the playground disables PyPI fallback). Notebooks whose code reads parquet also
get `pyarrow`: it is not a base dependency (it lives in the `io` extra), but the
playground's Pyodide (314.x) distributes it, so it resolves the same way -- and
only the notebooks that need it pay for the download.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PIP = "%pip install puremacro"
_PARQUET_MARKERS = ("read_parquet", "to_parquet", ".parquet")


def _needs_parquet(nb: dict) -> bool:
    code = "".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"
    )
    return any(marker in code for marker in _PARQUET_MARKERS)


def inject(path: Path) -> None:
    nb = json.loads(path.read_text())
    first = nb["cells"][0] if nb["cells"] else {}
    if first.get("cell_type") == "code" and _PIP in "".join(first.get("source", [])):
        return  # already injected (idempotent)
    line = _PIP + (" pyarrow" if _needs_parquet(nb) else "")
    cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n"],
    }
    nb["cells"].insert(0, cell)
    path.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        inject(Path(arg))
