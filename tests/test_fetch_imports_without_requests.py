"""Importing puremacro must not require a network stack.

`tests/test_pyodide_compat.py` deliberately permits ``import requests`` at module
scope in ``fetch/*`` — ARCHITECTURE.md and the README both document that as
intended — so its sweeps never noticed that ``puremacro.build_panel`` could not
be imported at all without it. That is the property users actually feel on a
tablet build, and it is what these pin.

Run in a subprocess with an import blocker installed, because ``requests`` is
installed in the dev environment and may already be in ``sys.modules``.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# NB: `find_module`/`load_module` were REMOVED in Python 3.12. A blocker written
# against that protocol is silently inert and every test using it passes without
# testing anything, which is worse than no test at all.
_BLOCKER = textwrap.dedent("""
    import sys
    from importlib.abc import MetaPathFinder

    class _Block(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "requests" or name.startswith("requests."):
                raise ImportError("requests is unavailable in this build")
            return None

    for _m in [m for m in sys.modules if m == "requests" or m.startswith("requests.")]:
        del sys.modules[_m]
    sys.meta_path.insert(0, _Block())
""")


@pytest.mark.mechanism_control
def test_the_blocker_actually_blocks():
    """Guards every other test in this file. If the import hook silently stops
    working — as it did when find_module was removed in 3.12 — the rest of these
    would pass while testing nothing."""
    p = _run("import requests")
    assert p.returncode != 0, "the blocker is inert; every test below is vacuous"
    assert "unavailable in this build" in p.stderr

_ENTRY_POINTS = [
    "puremacro",
    "puremacro.fetch",
    "puremacro.build_panel",
    "puremacro.build_subnational_panel",
    "puremacro.capital",
]


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", _BLOCKER + body],
                          capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("module", _ENTRY_POINTS)
def test_entry_point_imports_without_requests(module):
    p = _run(f"import {module}")
    assert p.returncode == 0, f"{module} needs requests to import:\n{p.stderr[-2000:]}"


def test_every_fetch_module_imports_without_requests():
    """Not just the package: each submodule, since build_panel reaches several
    of them directly and a single module-scope import breaks the whole chain."""
    p = _run(textwrap.dedent("""
        import pkgutil, puremacro.fetch as F
        bad = []
        for m in pkgutil.iter_modules(F.__path__):
            try:
                __import__(f"puremacro.fetch.{m.name}")
            except ImportError as e:
                if "unavailable in this build" in str(e):
                    bad.append(m.name)
            except Exception:
                pass          # an unrelated failure is not this test's business
        print(",".join(sorted(bad)))
    """))
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip() == "", f"still need requests at import: {p.stdout.strip()}"


def test_a_live_fetch_says_what_is_missing_rather_than_failing_obscurely():
    """Degrading at import is only half of it: the call has to explain itself."""
    p = _run(textwrap.dedent("""
        from puremacro.fetch._http import cached_get
        try:
            cached_get("https://example.invalid/x", refresh=True)
        except RuntimeError as e:
            assert "requests" in str(e), str(e)
            print("explained")
    """))
    assert "explained" in p.stdout, p.stderr[-2000:]


def test_the_transport_shim_can_still_rebind_requests():
    """puremacro.runtime.transport swaps this name for a browser-fetch shim and
    restores it. Guarding the import must leave the name bound for that."""
    p = _run(textwrap.dedent("""
        from puremacro.fetch import _http
        assert hasattr(_http, "requests"), "the name must stay bound for transport"
        sentinel = object()
        original = _http.requests
        _http.requests = sentinel
        assert _http.requests is sentinel
        _http.requests = original
        print("rebindable")
    """))
    assert "rebindable" in p.stdout, p.stderr[-2000:]
