"""F2.0 — status() shape, columns, and no-key-leak guarantee."""
from __future__ import annotations

import importlib
import re

import pandas as pd
import pytest


def _reset_credentials_module():
    import puremacro.credentials as C
    C._CONFIG_CACHE = None
    importlib.reload(C)
    return C


def test_status_returns_dataframe_with_expected_columns(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    for v in ("FRED_API_KEY", "PUREMACRO_FRED_API_KEY", "BEA_API_KEY",
              "PUREMACRO_BEA_API_KEY", "ANTHROPIC_API_KEY",
              "PUREMACRO_ANTHROPIC_API_KEY", "OPENAI_API_KEY",
              "PUREMACRO_OPENAI_API_KEY", "CENSUS_API_KEY",
              "PUREMACRO_CENSUS_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    C = _reset_credentials_module()
    df = C.status()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"service", "configured", "source",
                                "description", "signup_url"}
    assert set(df["service"]) == set(C.SERVICES.keys())
    assert (df["configured"] == False).all()
    assert (df["source"] == "missing").all()


def test_status_marks_env_source(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    C = _reset_credentials_module()
    df = C.status().set_index("service")
    assert df.loc["fred", "configured"] == True
    assert df.loc["fred", "source"] == "env:FRED_API_KEY"


def test_status_marks_config_file_source(monkeypatch, tmp_path):
    cfg = tmp_path / "credentials.toml"
    cfg.write_text('[fred]\napi_key = "from-config"\n')
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    C = _reset_credentials_module()
    df = C.status().set_index("service")
    assert df.loc["fred", "configured"] == True
    assert df.loc["fred", "source"] == "config_file"


def test_status_never_includes_key_values(monkeypatch):
    import puremacro.credentials as C
    # Set obviously-sensitive values and confirm none reach the DataFrame.
    monkeypatch.setenv("FRED_API_KEY", "SECRET-FRED-VALUE-9999")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SHOULD-NOT-LEAK-XXXX")
    df = C.status()
    flat = df.to_csv(index=False)
    assert "SECRET-FRED-VALUE-9999" not in flat
    assert "sk-ant-SHOULD-NOT-LEAK-XXXX" not in flat
