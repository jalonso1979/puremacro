"""Prepend a `%pip install puremacro` code cell to every .ipynb passed on argv.

In JupyterLite each notebook gets a fresh kernel, so each showcase notebook must
install puremacro before its `import puremacro...` cells. We inject the cell at
build time so the canonical notebooks/ sources stay clean.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PIP_SOURCE = ["%pip install puremacro\n"]


def inject(path: Path) -> None:
    nb = json.loads(path.read_text())
    first = nb["cells"][0] if nb["cells"] else {}
    if first.get("cell_type") == "code" and "%pip install puremacro" in "".join(
        first.get("source", [])
    ):
        return  # already injected (idempotent)
    cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _PIP_SOURCE,
    }
    nb["cells"].insert(0, cell)
    path.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        inject(Path(arg))
