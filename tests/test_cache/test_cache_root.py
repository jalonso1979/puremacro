from __future__ import annotations

import os
from pathlib import Path

from puremacro.cache import _default_root


def test_default_root_with_env_var(monkeypatch):
    test_dir = "/tmp/my_custom_puremacro_cache"
    monkeypatch.setenv("PUREMACRO_CACHE_DIR", test_dir)
    assert _default_root() == Path(test_dir)


def test_default_root_without_env_var(monkeypatch):
    monkeypatch.delenv("PUREMACRO_CACHE_DIR", raising=False)
    expected = Path.home() / ".cache" / "puremacro"
    assert _default_root() == expected
