"""Optional file-format engines: the ``io`` extra (``pip install "puremacro[io]"``).

``pyarrow`` (parquet) and ``openpyxl`` (.xlsx) stopped being base dependencies in
3.3.0: neither ships with every Pyodide distribution, and the JupyterLite
playground resolves its install with PyPI fallback disabled. Code that cannot
work without them calls :func:`require_engines` before doing any work, so a
missing engine surfaces as one clear ``ImportError`` naming the extra -- not as
a failure swallowed by a builder's ``except Exception: print(...)``, which is how
a bare install once produced panels silently missing most of their series.

Direct ``pandas.read_parquet`` / ``read_excel`` calls need no wrapper: pandas
already raises a clear ``ImportError`` of its own.
"""
from __future__ import annotations

from importlib.util import find_spec as _find_spec

INSTALL_HINT = 'pip install "puremacro[io]"'

# Engine kind -> (modules any one of which provides it, name shown to the user).
# pandas accepts either parquet engine, so fastparquet counts as well.
_ENGINES = {
    "parquet": (("pyarrow", "fastparquet"), "a parquet engine (pyarrow)"),
    "excel": (("openpyxl",), "the .xlsx engine openpyxl"),
}


class MissingEngineError(ImportError):
    """A file-format engine from the ``io`` extra is not installed."""


def _installed(module: str) -> bool:
    try:
        return _find_spec(module) is not None
    except (ImportError, ValueError):  # a blocking finder, or a half-initialised module
        return False


def has_engine(kind: str) -> bool:
    """True if the ``kind`` engine (``"parquet"`` or ``"excel"``) is installed."""
    try:
        modules, _ = _ENGINES[kind]
    except KeyError:
        raise ValueError(
            f"unknown engine kind {kind!r}; expected one of {sorted(_ENGINES)}"
        ) from None
    return any(_installed(m) for m in modules)


def require_engines(*kinds: str, feature: str) -> None:
    """Raise :class:`MissingEngineError` unless every engine in ``kinds`` is installed."""
    missing = [_ENGINES[k][1] for k in kinds if not has_engine(k)]
    if missing:
        verb = "is" if len(missing) == 1 else "are"
        raise MissingEngineError(
            f"{feature} needs {' and '.join(missing)}, which {verb} not installed. "
            f"Install the file-format engines with:  {INSTALL_HINT}"
        )
