"""Standard-library modules that some Python builds leave out.

Pyodide 0.28 ships ``sqlite3`` and ``ssl`` as *unvendored* packages that a page
must load on request, so a plain ``import sqlite3`` at module level makes every
importer of ``puremacro.narrative`` fail there with ``ModuleNotFoundError``
(found by the 2026-10-03 headless gallery run: one case of 110). Pyodide 314
vendors both again, and every CPython build has them. Importing them through
this module keeps the import of the *library* working everywhere and moves the
failure to the first actual use, with a message that says what to do.

Usage::

    from puremacro._optional_stdlib import sqlite3, ssl

On a build that has the module, the name *is* the real module. On one that
does not, it is a placeholder whose attribute access raises ``ImportError``.
"""
from __future__ import annotations

import importlib
from types import ModuleType


class _MissingStdlibModule:
    """Placeholder for a standard-library module absent from this Python build."""

    def __init__(self, name: str, hint: str) -> None:
        self._name = name
        self._hint = hint

    def __getattr__(self, attr: str):
        raise ImportError(
            f"the standard-library module {self._name!r} is not available in this Python "
            f"build (needed for {attr!r}). {self._hint}"
        )

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<missing stdlib module {self._name!r}>"


def _optional(name: str, hint: str) -> ModuleType | _MissingStdlibModule:
    try:
        return importlib.import_module(name)
    except ImportError:
        return _MissingStdlibModule(name, hint)


_PYODIDE_HINT = ("In Pyodide 0.28 run `await pyodide.loadPackage({pkg!r})` (or `%pip install {pkg}` in "
                 "JupyterLite) before using this feature; Pyodide 314 and CPython have it built in.")

sqlite3 = _optional("sqlite3", _PYODIDE_HINT.format(pkg="sqlite3"))
"""``sqlite3`` or a placeholder: used by the HTTP/data caches and the narrative LLM kernels."""

ssl = _optional("ssl", _PYODIDE_HINT.format(pkg="ssl"))
"""``ssl`` or a placeholder: used only to build TLS contexts for live fetches."""

__all__ = ["sqlite3", "ssl"]
