"""F2.0 — MissingCredentialError message includes all four resolver tiers."""
from __future__ import annotations

import importlib

import pytest


def _reset_credentials_module():
    import puremacro.credentials as C
    C._CONFIG_CACHE = None
    importlib.reload(C)
    return C


def test_message_names_service_description(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    with pytest.raises(C.MissingCredentialError) as exc_info:
        C.require("fred")
    msg = str(exc_info.value)
    assert "FRED + ALFRED real-time macro data" in msg


def test_message_lists_every_env_var_checked(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    with pytest.raises(C.MissingCredentialError) as exc_info:
        C.require("fred")
    msg = str(exc_info.value)
    assert "FRED_API_KEY" in msg
    assert "PUREMACRO_FRED_API_KEY" in msg


def test_message_names_config_path_and_not_found(monkeypatch, tmp_path):
    cfg = tmp_path / "absent.toml"
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    with pytest.raises(C.MissingCredentialError) as exc_info:
        C.require("fred")
    msg = str(exc_info.value)
    assert str(cfg) in msg
    assert "not found" in msg


def test_message_includes_signup_url(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    with pytest.raises(C.MissingCredentialError) as exc_info:
        C.require("fred")
    assert "https://fred.stlouisfed.org/docs/api/api_key.html" in str(exc_info.value)


def test_require_returns_key_when_present(monkeypatch):
    import puremacro.credentials as C
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    assert C.require("fred") == "abc123"
