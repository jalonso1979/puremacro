"""What can this machine actually do?

`puremacro`'s headline promise is that the estimator core runs on an
iPad (juno.sh / the Juno app). The promise has always been *static* —
``tests/test_pyodide_compat.py`` asserts no forbidden module lands in
``sys.modules``. This module makes it *dynamic*: at run time it reports
which of the capabilities the rest of the package quietly assumes are
actually present.

Four assumptions break on a tablet, and each one has a field here:

============  =====================================================
``sockets``   ``requests`` / ``urllib`` need a real socket. Pyodide
              has none, so every ``fetch.*`` call dies with a
              connection error until :mod:`puremacro.runtime.transport`
              routes it through the browser's fetch instead.
``parquet``   ``pyarrow`` has no Pyodide wheel, so ``read_parquet``
              (``cache``, ``fetch.labor*``, ``shock_atlas``,
              ``build_panel``) is unavailable. :mod:`puremacro.runtime.store`
              is the pure-numpy way around it.
``threads``   Pyodide is single-threaded; the numba / mlx / cupy
              backends are absent. Compute budgets shrink accordingly
              (:mod:`puremacro.runtime.budget`).
``writable``  A read-only or ephemeral filesystem means checkpoints
              (:mod:`puremacro.longrun`) have nowhere to live.
============  =====================================================

Detection is heuristic by necessity — no Python API tells you "you are
inside Juno". Every field can be overridden with an environment
variable (see :data:`_ENV_OVERRIDES`) when the heuristic guesses wrong,
and :func:`refresh` re-runs detection after an override changes.

Pure stdlib + numpy. Nothing here imports anything the Pyodide contract
forbids.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "Capabilities",
    "capabilities",
    "refresh",
    "report",
    "is_pyodide",
    "is_tablet",
]

# Host families. "pyodide" is the browser/WASM kernel (juno.sh, JupyterLite);
# "cpython" is any real interpreter, including the one Juno bundles on iPadOS.
HOSTS = ("cpython", "pyodide", "unknown")

# Device classes, ordered by how much compute they can be asked for.
DEVICES = ("workstation", "tablet", "browser", "unknown")

# Environment overrides. Each maps to the Capabilities field of the same
# name; values are parsed by _coerce below. Set these when the heuristic
# is wrong (e.g. `PUREMACRO_DEVICE=tablet` to force tablet-sized budgets
# on a laptop while you test a notebook you plan to run on the iPad).
_ENV_OVERRIDES = {
    "PUREMACRO_HOST": "host",
    "PUREMACRO_DEVICE": "device",
    "PUREMACRO_SOCKETS": "sockets",
    "PUREMACRO_PARQUET": "parquet",
}


@dataclass(frozen=True)
class Capabilities:
    """A snapshot of what this environment supports.

    Attributes
    ----------
    host : str
        One of :data:`HOSTS`.
    device : str
        One of :data:`DEVICES` — the compute class, which is what
        :mod:`puremacro.runtime.budget` keys off.
    python : str
        ``sys.version.split()[0]``.
    machine : str
        ``platform.machine()`` (``arm64``, ``x86_64``, ``wasm32``, ...).
    sockets : bool
        True if ``urllib`` / ``requests`` can open a TCP connection.
        False under Pyodide, where :mod:`puremacro.runtime.transport`
        is the only way out.
    js_fetch : bool
        True if the JavaScript ``fetch`` / ``XMLHttpRequest`` bridge is
        reachable (i.e. the ``js`` module imports).
    parquet : bool
        True if a ``pandas.read_parquet`` engine is installed.
    threads : bool
        True if the interpreter has working OS threads.
    writable_fs : bool
        True if the temporary directory accepts writes.
    backends : tuple[str, ...]
        Installed compute backends, from :mod:`puremacro._backend`.
    cpu_count : int
        ``os.cpu_count() or 1``.
    memory_mb : int | None
        Physical memory where the platform reports it, else None.
    overridden : tuple[str, ...]
        Fields whose value came from an environment variable rather
        than from detection.
    """

    host: str
    device: str
    python: str
    machine: str
    sockets: bool
    js_fetch: bool
    parquet: bool
    threads: bool
    writable_fs: bool
    backends: tuple[str, ...]
    cpu_count: int
    memory_mb: int | None
    overridden: tuple[str, ...]

    def as_dict(self) -> dict:
        """Plain-dict view, for logging into a cartridge manifest."""
        return asdict(self)


def _detect_host() -> str:
    # Pyodide sets sys.platform to "emscripten"; the `pyodide` module is
    # also importable, but checking sys.platform avoids the import cost.
    if sys.platform == "emscripten" or "pyodide" in sys.modules:
        return "pyodide"
    if sys.implementation.name == "cpython":
        return "cpython"
    return "unknown"


def _looks_like_ios() -> bool:
    """Heuristics for 'this is an iPad', across both Juno flavours.

    * CPython >= 3.13 sets ``sys.platform == "ios"``.
    * An app-bundled interpreter lives under the iOS app sandbox, whose
      paths start with ``/var/mobile/`` or ``/private/var/mobile/``.
    * ``platform.machine()`` reports ``iPad...`` / ``iPhone...`` on some
      builds.
    """
    if sys.platform == "ios":
        return True
    machine = platform.machine()
    if machine.startswith(("iPad", "iPhone")):
        return True
    for path in (sys.prefix, sys.executable or ""):
        if "/var/mobile/" in path or "/Juno" in path:
            return True
    return False


def _looks_like_ipad_safari() -> bool:
    """Detect modern iPadOS Safari in Pyodide.

    Since iPadOS 13, Safari on iPad requests desktop sites by default,
    sending a macOS User-Agent ('Macintosh; Intel Mac OS X ...').
    The standard detection is navigator.maxTouchPoints > 1 on Mac platforms.
    """
    if _detect_host() != "pyodide" or importlib.util.find_spec("js") is None:
        return False
    try:  # pragma: no cover - only reachable under Pyodide
        import js  # type: ignore[import-not-found]

        nav = getattr(js, "navigator", None)
        if nav is None:
            return False
        max_touch = int(getattr(nav, "maxTouchPoints", 0))
        platform_str = str(getattr(nav, "platform", ""))
        ua = str(getattr(nav, "userAgent", ""))
        return max_touch > 1 and ("MacIntel" in platform_str or "Macintosh" in ua)
    except Exception:
        return False


def _detect_device(host: str) -> str:
    if _looks_like_ios():
        return "tablet"
    if host == "pyodide":
        if _looks_like_ipad_safari():
            return "tablet"
        # A browser kernel on an unknown screen. Check the user agent if
        # the JS bridge is there — that is the only way to tell an iPad
        # Safari kernel from a desktop one.
        ua = _user_agent()
        if ua and any(tag in ua for tag in ("iPad", "iPhone", "Android")):
            return "tablet"
        return "browser"
    if host == "cpython":
        return "workstation"
    return "unknown"


def _user_agent() -> str | None:
    """The browser UA string, or None outside a browser."""
    if _detect_host() != "pyodide" or importlib.util.find_spec("js") is None:
        return None
    try:  # pragma: no cover - only reachable under Pyodide
        import js  # type: ignore[import-not-found]

        return str(js.navigator.userAgent)
    except Exception:
        return None


def _detect_sockets(host: str) -> bool:
    # Pyodide ships a `socket` module that raises on connect, so presence
    # of the module proves nothing; the host is the reliable signal.
    return host != "pyodide"


def _detect_js_fetch(host: str) -> bool:
    """True if the JS bridge is genuinely reachable.

    Gated on the host: an unrelated ``js`` namespace package in
    site-packages (several PyPI distributions ship one) makes a bare
    ``find_spec("js")`` a false positive on a workstation.
    """
    if host != "pyodide":
        return False
    return importlib.util.find_spec("js") is not None


def _detect_parquet() -> bool:
    return any(
        importlib.util.find_spec(name) is not None
        for name in ("pyarrow", "fastparquet")
    )


def _detect_threads() -> bool:
    if importlib.util.find_spec("_thread") is None:
        return False
    try:
        import threading

        # Pyodide without the pthread build raises on thread creation.
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
        return True
    except Exception:
        return False


def _detect_writable_fs() -> bool:
    try:
        with tempfile.NamedTemporaryFile(prefix="puremacro-probe-") as fh:
            fh.write(b"1")
        return True
    except Exception:
        return False


def _detect_backends() -> tuple[str, ...]:
    try:
        from puremacro._backend import available_backends

        return tuple(available_backends())
    except Exception:  # pragma: no cover - _backend is always importable
        return ("numpy",)


def _detect_memory_mb() -> int | None:
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
    if not isinstance(page, int) or not isinstance(pages, int):
        return None
    if page <= 0 or pages <= 0:
        return None
    return int(page * pages / (1024 * 1024))


def _coerce(field: str, raw: str):
    """Parse an override string into the field's type."""
    if field in ("sockets", "parquet"):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    value = raw.strip().lower()
    if field == "host" and value not in HOSTS:
        raise ValueError(
            f"PUREMACRO_HOST={raw!r} is not one of {HOSTS}"
        )
    if field == "device" and value not in DEVICES:
        raise ValueError(
            f"PUREMACRO_DEVICE={raw!r} is not one of {DEVICES}"
        )
    return value


def _detect() -> Capabilities:
    host = _detect_host()
    # Heterogeneous by construction: str, bool, tuple and int fields all
    # land here before being handed to the frozen dataclass.
    caps: dict[str, Any] = {
        "host": host,
        "device": _detect_device(host),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "sockets": _detect_sockets(host),
        "js_fetch": _detect_js_fetch(host),
        "parquet": _detect_parquet(),
        "threads": _detect_threads(),
        "writable_fs": _detect_writable_fs(),
        "backends": _detect_backends(),
        "cpu_count": os.cpu_count() or 1,
        "memory_mb": _detect_memory_mb(),
    }
    overridden = []
    for env_name, field in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        caps[field] = _coerce(field, raw)
        overridden.append(field)
        # An overridden host implies a re-derived device unless the
        # device was itself pinned.
        if field == "host" and "PUREMACRO_DEVICE" not in os.environ:
            caps["device"] = _detect_device(str(caps["host"]))
    caps["overridden"] = tuple(sorted(overridden))
    return Capabilities(**caps)


_CACHE: Capabilities | None = None


def capabilities() -> Capabilities:
    """Return the (cached) capability snapshot for this process."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _detect()
    return _CACHE


def refresh() -> Capabilities:
    """Re-run detection, discarding the cache.

    Call this after changing one of the ``PUREMACRO_*`` environment
    variables inside a live session.
    """
    global _CACHE
    _CACHE = None
    return capabilities()


def is_pyodide() -> bool:
    """True when running under a Pyodide/WASM kernel."""
    return capabilities().host == "pyodide"


def is_tablet() -> bool:
    """True when the device class is a tablet (iPad, Juno, phone)."""
    return capabilities().device == "tablet"


def report() -> str:
    """A human-readable capability summary.

    >>> print(puremacro.runtime.report())  # doctest: +SKIP
    puremacro runtime
      host       : cpython 3.12.4 (arm64)
      device     : workstation
      network    : sockets
      parquet    : available
      ...
    """
    c = capabilities()
    if c.sockets:
        network = "sockets"
    elif c.js_fetch:
        network = "js-fetch (call runtime.enable_browser_network())"
    else:
        network = "unavailable"
    mem = f"{c.memory_mb} MB" if c.memory_mb is not None else "unknown"
    lines = [
        "puremacro runtime",
        f"  host       : {c.host} {c.python} ({c.machine})",
        f"  device     : {c.device}",
        f"  network    : {network}",
        f"  parquet    : {'available' if c.parquet else 'unavailable -> use puremacro.runtime.store / pocket'}",
        f"  threads    : {'yes' if c.threads else 'no'} ({c.cpu_count} cpu, {mem})",
        f"  writable fs: {'yes' if c.writable_fs else 'no'}",
        f"  backends   : {', '.join(c.backends)}",
    ]
    if c.overridden:
        lines.append(f"  overridden : {', '.join(c.overridden)}")
    return "\n".join(lines)
