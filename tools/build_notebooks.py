"""Build the puremacro.vfi showcase notebooks from their jupytext .py sources.

Source of truth: ``notebooks/NN_topic.py`` (percent format). This converts +
executes each to ``notebooks/NN_topic.ipynb`` (committed, WITH outputs). The
.ipynb is a build artifact -- never hand-edit it; edit the .py and rebuild.

    python tools/build_notebooks.py                      # build all
    python tools/build_notebooks.py 01_wealth_inequality # build one (by stem or filename)
    python tools/build_notebooks.py --list               # list discovered sources
    python tools/build_notebooks.py --check              # execute all to a temp file, fail on error

Requires the ``notebooks`` extra:  pip install -e ".[notebooks]"
The kernel cwd is ``notebooks/`` so notebooks can ``import _nbstyle`` directly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJ_ROOT / "notebooks"
KERNEL_NAME = "python3"


def discover_sources() -> list[Path]:
    """Sorted notebook sources (``notebooks/*.py`` and ``notebooks/course/*.py``,
    excluding ``_*.py`` helpers)."""
    top = [p for p in NB_DIR.glob("*.py") if not p.name.startswith("_")]
    course = [p for p in (NB_DIR / "course").glob("*.py") if not p.name.startswith("_")]
    return sorted(top + course)


def ensure_kernel() -> None:
    """Register a ``python3`` Jupyter kernelspec for the current interpreter.

    jupytext --execute needs a registered kernelspec; a bare ``ipykernel``
    install does not create one. Idempotent (re-registering overwrites). Points
    the standard ``python3`` kernel at ``sys.executable`` so the notebooks build
    with the same Python that has puremacro installed.
    """
    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--user", "--name", KERNEL_NAME],
        capture_output=True,
    )


def _jupytext(args: list[str]) -> int:
    cmd = [sys.executable, "-m", "jupytext", *args]
    print("›", " ".join(cmd))
    return subprocess.run(cmd, cwd=NB_DIR).returncode


def build_one(src: Path, *, check: bool = False) -> int:
    """Convert+execute one source. check=True writes to a temp file (discarded).

    ``--run-path NB_DIR`` pins the kernel's working directory to ``notebooks/``
    regardless of where the output ``.ipynb`` is written. Without it, jupytext
    derives the run path from the OUTPUT file's parent dir, so ``--check`` (which
    writes to a temp dir) would execute the kernel there and ``import _nbstyle``
    would fail.
    """
    base = ["--to", "ipynb", "--execute", "--set-kernel", KERNEL_NAME,
            "--run-path", str(NB_DIR)]
    # Path relative to NB_DIR (the subprocess cwd), so ``course/<name>.py`` sources
    # build correctly while top-level sources stay just ``<name>.py``.
    rel = str(src.relative_to(NB_DIR))
    if check:
        out = Path(tempfile.gettempdir()) / f"{src.stem}.check.ipynb"
        rc = _jupytext([*base, "--output", str(out), rel])
        out.unlink(missing_ok=True)
        return rc
    return _jupytext([*base, rel])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build puremacro.vfi showcase notebooks.")
    ap.add_argument("names", nargs="*", help="source stems/filenames to build (default: all)")
    ap.add_argument("--list", action="store_true", help="list discovered sources and exit")
    ap.add_argument("--check", action="store_true", help="execute to temp, discard, fail on error")
    ns = ap.parse_args(argv)

    srcs = discover_sources()
    if ns.names:
        wanted = set(ns.names)
        srcs = [s for s in srcs if s.stem in wanted or s.name in wanted]
        if not srcs:
            print(f"no sources match {sorted(wanted)}", file=sys.stderr)
            return 2
    if ns.list:
        for s in srcs:
            print(s.relative_to(PROJ_ROOT))
        return 0
    ensure_kernel()
    rc_total = 0
    for s in srcs:
        rc = build_one(s, check=ns.check)
        rc_total = rc_total or rc
    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
