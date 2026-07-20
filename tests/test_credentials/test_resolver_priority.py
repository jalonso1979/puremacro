"""F2.0 — Resolver priority: explicit > primary env > secondary env > config > None."""
from __future__ import annotations

import importlib

import pytest


def _reset_credentials_module():
    import puremacro.credentials as C
    C._CONFIG_CACHE = None
    importlib.reload(C)
    return C


def test_explicit_kwarg_wins_over_env(monkeypatch):
    C = _reset_credentials_module()
    monkeypatch.setenv("FRED_API_KEY", "from-env")
    assert C.get("fred", explicit="from-caller") == "from-caller"


def test_primary_env_var_wins_over_secondary(monkeypatch):
    C = _reset_credentials_module()
    monkeypatch.setenv("FRED_API_KEY", "primary")
    monkeypatch.setenv("PUREMACRO_FRED_API_KEY", "secondary")
    assert C.get("fred") == "primary"


def test_secondary_env_var_used_when_primary_missing(monkeypatch):
    C = _reset_credentials_module()
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("PUREMACRO_FRED_API_KEY", "secondary")
    assert C.get("fred") == "secondary"


def test_config_file_used_when_env_missing(monkeypatch, tmp_path):
    cfg = tmp_path / "credentials.toml"
    cfg.write_text('[fred]\napi_key = "from-config"\n')
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    assert C.get("fred") == "from-config"


def test_env_wins_over_config_file(monkeypatch, tmp_path):
    cfg = tmp_path / "credentials.toml"
    cfg.write_text('[fred]\napi_key = "from-config"\n')
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.setenv("FRED_API_KEY", "from-env")
    C = _reset_credentials_module()
    assert C.get("fred") == "from-env"


def test_none_when_nothing_set(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "missing.toml"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    assert C.get("fred") is None


def test_unknown_service_raises_keyerror():
    from puremacro.credentials import get
    with pytest.raises(KeyError, match="bogus_service"):
        get("bogus_service")
