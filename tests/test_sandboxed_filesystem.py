"""puremacro must import and fetch on a sandboxed filesystem (iOS / Juno).

`pathlib` only swallows ENOENT / ENOTDIR / EBADF / ELOOP, so on iOS a stat of
anything outside the app sandbox raises ``PermissionError`` instead of
answering False. Every filesystem probe puremacro runs at import time, and the
HTTP cache it writes, has to survive that.
"""
from __future__ import annotations

import errno
import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture
def deny_all_stats(monkeypatch):
    """Every os.stat raises EPERM, as it does outside an iOS sandbox."""
    def _boom(path, *a, **kw):
        raise PermissionError(errno.EPERM, "Operation not permitted", str(path))

    monkeypatch.setattr(os, "stat", _boom)
    return _boom


def test_pathlib_still_does_not_swallow_eperm(deny_all_stats):
    """The premise of this module. If CPython ever starts ignoring EPERM the
    guards become redundant — but until then they are load-bearing."""
    with pytest.raises(PermissionError):
        Path("/nowhere/at/all").is_dir()


def test_x13_dir_resolution_survives_a_denied_filesystem(deny_all_stats):
    """_resolve_x13_dir runs at import of puremacro.sa.x13."""
    from puremacro.sa import x13

    assert x13._resolve_x13_dir() is None          # "no binary", not a crash


def test_sa_imports_under_a_denied_filesystem(monkeypatch):
    """Re-import the module with every stat denied, as an iPad would."""
    def _boom(path, *a, **kw):
        raise PermissionError(errno.EPERM, "Operation not permitted", str(path))

    import puremacro.sa.x13 as x13

    monkeypatch.setattr(os, "stat", _boom)
    reloaded = importlib.reload(x13)
    assert reloaded.x13_available() is False


def test_cached_get_falls_back_to_an_uncached_fetch(monkeypatch, tmp_path):
    """An unwritable (or unreadable) cache directory must not fail the fetch."""
    from puremacro.fetch import _http

    monkeypatch.setattr(_http, "CACHE_ROOT", tmp_path / "cache")

    def _denied(self, *a, **kw):
        raise PermissionError(errno.EPERM, "Operation not permitted", str(self))

    monkeypatch.setattr(Path, "exists", _denied)
    monkeypatch.setattr(Path, "mkdir", _denied)
    monkeypatch.setattr(Path, "write_bytes", _denied)

    class _Resp:
        content = b"payload"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(_http.requests, "get", lambda *a, **kw: _Resp())
    assert _http.cached_get("https://example.invalid/x.csv") == b"payload"


def test_manifest_helpers_never_raise(monkeypatch, tmp_path):
    from puremacro.fetch import _http

    monkeypatch.setattr(_http, "CACHE_ROOT", tmp_path / "cache")

    def _denied(self, *a, **kw):
        raise PermissionError(errno.EPERM, "Operation not permitted", str(self))

    monkeypatch.setattr(Path, "exists", _denied)
    monkeypatch.setattr(Path, "mkdir", _denied)
    assert _http._load_manifest() == {}
    _http._write_manifest({"a": 1})               # must not raise


# --------------------------------------------------------------------------
# No HTTP stack at all
#
# A Juno / Pyodide build may ship without `requests`. Every fetcher in this
# package documents an empty frame — never an exception — as its failure mode,
# because callers routinely have a frozen snapshot to fall back to. An
# ImportError escaping a fetch takes those callers down for no reason.
# --------------------------------------------------------------------------

@pytest.fixture
def no_requests(monkeypatch):
    """Make `import requests` fail, as on a tablet build without the scraper stack."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *a, **kw):
        if name.split(".")[0] in {"requests", "urllib3"}:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    # `_http` caches the module object at import time; drop it so the guarded
    # re-import inside get_sdmx_csv is the one that runs.
    for mod in ("requests", "puremacro.fetch._http"):
        monkeypatch.delitem(__import__("sys").modules, mod, raising=False)


def test_sdmx_fetch_returns_empty_without_requests(no_requests):
    from puremacro.fetch._oecd_sdmx import get_sdmx_csv

    got = get_sdmx_csv("OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,", "Q...........", "2020")
    assert got.empty


def test_qna_panel_returns_empty_without_requests(no_requests):
    """The iPad path: the fetch fails, the notebook falls back to its snapshot."""
    from puremacro.fetch import qna_meta, qna_panel

    panel = qna_panel(["USA", "ESP"], start="2020", output=True, income=True)
    assert panel.empty
    assert qna_meta(panel).empty          # metadata helper survives it too


def test_xrate_fetch_returns_empty_without_requests(no_requests):
    from puremacro.fetch import fetch_xrate_monthly

    assert fetch_xrate_monthly(["MEX"], start_period="2020").empty


def test_qna_countries_falls_back_to_the_frozen_list_without_requests(no_requests):
    from puremacro.fetch import QNA_AGGREGATES, qna_countries

    codes = qna_countries()
    assert len(codes) >= 45 and "USA" in codes
    assert not (set(codes) & QNA_AGGREGATES)


@pytest.mark.mechanism_control
def test_the_no_requests_fixture_actually_blocks(no_requests):
    """Positive control for the `no_requests` fixture.

    It patches ``builtins.__import__``, which is a mechanism: if the predicate
    stopped matching, every test using this fixture would exercise the ordinary
    requests-present path while claiming to prove the opposite.
    """
    with pytest.raises(ImportError):
        import requests            # noqa: F401
