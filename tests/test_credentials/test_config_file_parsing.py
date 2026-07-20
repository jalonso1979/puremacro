"""F2.0 — TOML config-file resolution."""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest


def test_default_config_path_uses_env_override(monkeypatch, tmp_path):
    from puremacro.credentials import default_config_path

    custom = tmp_path / "custom-creds.toml"
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(custom))
    assert default_config_path() == custom


def test_default_config_path_uses_xdg(monkeypatch, tmp_path):
    from puremacro.credentials import default_config_path

    monkeypatch.delenv("PUREMACRO_CREDENTIALS_FILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_config_path() == tmp_path / "puremacro" / "credentials.toml"


def test_default_config_path_fallback_to_home(monkeypatch, tmp_path):
    from puremacro.credentials import default_config_path

    monkeypatch.delenv("PUREMACRO_CREDENTIALS_FILE", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert default_config_path() == tmp_path / ".puremacro" / "credentials.toml"


def test_config_lookup_finds_key(monkeypatch, tmp_path):
    from puremacro.credentials import get

    cfg = tmp_path / "credentials.toml"
    cfg.write_text('[fred]\napi_key = "from-toml"\n')
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    # Wipe env so config-file path is the only source.
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    # Reset module-level cache by reimporting.
    import importlib, puremacro.credentials as C
    importlib.reload(C)
    assert C.get("fred") == "from-toml"


def test_missing_config_file_returns_silently(monkeypatch, tmp_path):
    from puremacro.credentials import get

    cfg = tmp_path / "nonexistent.toml"
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    import importlib, puremacro.credentials as C
    importlib.reload(C)
    # No warning, no error, just None.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert C.get("fred") is None
    assert not caught, f"unexpected warning(s): {[str(w.message) for w in caught]}"


def test_malformed_toml_warns_and_falls_through(monkeypatch, tmp_path):
    from puremacro.credentials import get

    cfg = tmp_path / "bad.toml"
    cfg.write_text("[fred\napi_key = no quotes here\n")
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    import importlib, puremacro.credentials as C
    importlib.reload(C)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert C.get("fred") is None
    user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warns) == 1, f"expected one UserWarning, got {caught}"
    assert str(cfg) in str(user_warns[0].message)
