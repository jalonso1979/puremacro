"""Per-case timing loop of puremacro's validation gallery, shared by both platforms.

``tools/pyodide/gallery_runner.js`` loads this file into the Pyodide interpreter
with ``pyodide.runPython(source)`` and calls ``discover()`` / ``run_index(i)``
from JavaScript; ``tools/pyodide_gallery.py`` runs the very same file as a
desktop CPython subprocess (``python tools/pyodide/gallery_cases.py --json
PATH``). One copy of the loop means the two platforms are timed the same way:
``time.perf_counter()`` around ``puremacro.validation.run(case)`` for every case,
one case at a time, in discovery order.

Nothing beyond the standard library is imported until ``discover()`` runs, so
the module loads before the wheel is installed. ``MPLBACKEND`` is forced to
``Agg`` because Pyodide's default matplotlib backend expects a browser DOM that
Node does not provide (the gallery itself draws nothing).

Record emitted per case (JSON object)::

    {"id", "subsystem", "mechanism", "tol", "passed", "max_margin", "error", "seconds"}

``max_margin`` is ``null`` when the case raised (``CaseResult.max_margin`` is
``inf`` then) because strict JSON has no infinity; ``error`` carries the
exception text in that situation.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

SCHEMA_VERSION = 1

_cases: list = []


def discover() -> int:
    """Import puremacro, discover the gallery cases; return their number."""
    from puremacro.validation.runner import _discover_cases

    _cases[:] = _discover_cases()
    return len(_cases)


def run_index(i: int) -> str:
    """Run case ``i`` of the discovered list; return its record as a JSON string."""
    from puremacro.validation._model import run

    case = _cases[i]
    t0 = time.perf_counter()
    result = run(case)  # never raises: a broken case is a failed case
    seconds = time.perf_counter() - t0
    margin = float(result.max_margin)
    return json.dumps(
        {
            "id": result.id,
            "subsystem": result.subsystem,
            "mechanism": result.mechanism,
            "tol": result.tol,
            "passed": bool(result.passed),
            "max_margin": margin if math.isfinite(margin) else None,
            "error": result.error or "",
            "seconds": round(seconds, 6),
        }
    )


def versions() -> str:
    """Interpreter, puremacro and core-dependency versions as a JSON string."""
    from importlib import metadata

    import puremacro

    packages: dict = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "requests"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return json.dumps(
        {
            "python_version": sys.version.split()[0],
            "puremacro_version": puremacro.__version__,
            "puremacro_file": puremacro.__file__,
            "packages": packages,
        }
    )


def progress_line(i: int, n: int, rec: dict) -> str:
    verdict = "PASS" if rec["passed"] else "FAIL"
    line = f"[{i + 1}/{n}] {verdict} {rec['id']} ({rec['seconds']:.3f} s)"
    if rec["error"]:
        line += f" -- {rec['error']}"
    return line


def main(argv=None) -> int:
    """Desktop entry point: run every case here and print one JSON envelope."""
    import argparse
    import datetime as _dt
    import platform

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", default=None, help="also write the envelope (indented) to this path")
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    loaded_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    n = discover()
    import_s = time.perf_counter() - t0

    cases = []
    for i in range(n):
        rec = json.loads(run_index(i))
        cases.append(rec)
        print(progress_line(i, n, rec), file=sys.stderr, flush=True)

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "runtime": "cpython",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "loaded_at": loaded_at,
        **json.loads(versions()),
        "import_s": round(import_s, 3),
        "n_cases": n,
        "n_passed": sum(1 for c in cases if c["passed"]),
        "n_failed": sum(1 for c in cases if not c["passed"]),
        "gallery_s": round(sum(c["seconds"] for c in cases), 3),
        "runtime_s": round(time.perf_counter() - t_start, 3),
        "cases": cases,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(envelope, fh, indent=2)
            fh.write("\n")
    print(json.dumps(envelope))
    return 0


# Under Pyodide this file is executed with runPython() in the __main__ namespace,
# so the guard on sys.platform keeps the desktop entry point from running there.
if __name__ == "__main__" and sys.platform != "emscripten":
    sys.exit(main())
