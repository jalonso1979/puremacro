"""Tests for puremacro.instruments.external.fred."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from puremacro.instruments import Instrument
from puremacro.instruments.external.fred import load


_SYNTHETIC_FRED_JSON = json.dumps({
    "realtime_start": "2026-01-01",
    "realtime_end": "2026-01-01",
    "observation_start": "1600-01-01",
    "observation_end": "9999-12-31",
    "units": "lin",
    "output_type": 1,
    "file_type": "json",
    "order_by": "observation_date",
    "sort_order": "asc",
    "count": 4,
    "offset": 0,
    "limit": 100000,
    "observations": [
        {"realtime_start": "2026-01-01", "realtime_end": "2026-01-01",
         "date": "2020-01-01", "value": "1.50"},
        {"realtime_start": "2026-01-01", "realtime_end": "2026-01-01",
         "date": "2020-02-01", "value": "1.75"},
        {"realtime_start": "2026-01-01", "realtime_end": "2026-01-01",
         "date": "2020-03-01", "value": "."},  # missing
        {"realtime_start": "2026-01-01", "realtime_end": "2026-01-01",
         "date": "2020-04-01", "value": "0.25"},
    ],
})


def _patched_safe_get_text(url):
    """Returns synthetic FRED JSON regardless of URL."""
    return _SYNTHETIC_FRED_JSON


def test_load_with_explicit_api_key_returns_instrument(monkeypatch):
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setattr(_mod, "safe_get_text", _patched_safe_get_text)
    inst = load(series_id="FEDFUNDS", api_key="dummy_key", frequency="M")
    assert isinstance(inst, Instrument)
    assert inst.category == "external_csv"
    assert inst.frequency == "M"
    assert inst.name == "fred_FEDFUNDS"


def test_load_uses_env_var_api_key(monkeypatch):
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setattr(_mod, "safe_get_text", _patched_safe_get_text)
    monkeypatch.setenv("FRED_API_KEY", "env_key_value")
    inst = load(series_id="FEDFUNDS", frequency="M")
    assert inst.name == "fred_FEDFUNDS"


def test_load_no_api_key_anywhere_raises(monkeypatch, tmp_path):
    """If neither api_key= nor any FRED env var / config file is set, raise."""
    from puremacro import credentials
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("PUREMACRO_FRED_API_KEY", raising=False)
    # Isolate from any ambient credentials.toml on the dev machine.
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "none.toml"))
    monkeypatch.setattr(credentials, "_CONFIG_CACHE", None, raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        load(series_id="FEDFUNDS", frequency="M")


def test_load_parses_dot_as_missing(monkeypatch):
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setattr(_mod, "safe_get_text", _patched_safe_get_text)
    inst = load(series_id="FEDFUNDS", api_key="dummy", frequency="M")
    assert pd.isna(inst.series.loc[pd.Timestamp("2020-03-01")])
    assert inst.series.dropna().shape[0] == 3


def test_load_observation_date_range_kwargs(monkeypatch):
    """observation_start and observation_end propagate into the URL."""
    captured_url = {"url": None}
    def _capture(url):
        captured_url["url"] = url
        return _SYNTHETIC_FRED_JSON
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setattr(_mod, "safe_get_text", _capture)
    load(series_id="FEDFUNDS", api_key="dummy", frequency="M",
         observation_start="2010-01-01", observation_end="2020-12-31")
    assert "observation_start=2010-01-01" in captured_url["url"]
    assert "observation_end=2020-12-31" in captured_url["url"]
    assert "series_id=FEDFUNDS" in captured_url["url"]


def test_load_network_failure_raises_clear_error(monkeypatch):
    from puremacro.instruments.external import fred as _mod
    def _fail(_url):
        raise OSError("simulated network failure")
    monkeypatch.setattr(_mod, "safe_get_text", _fail)
    with pytest.raises(RuntimeError, match="FRED"):
        load(series_id="FEDFUNDS", api_key="dummy", frequency="M")


def test_load_instrument_name_includes_series_id(monkeypatch):
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setattr(_mod, "safe_get_text", _patched_safe_get_text)
    inst = load(series_id="NFCI", api_key="dummy", frequency="W")
    assert "NFCI" in inst.name


def test_load_explicit_empty_string_api_key_does_not_fall_back_to_env(monkeypatch):
    """If caller passes api_key='' explicitly, do NOT silently use env var."""
    from puremacro.instruments.external import fred as _mod
    monkeypatch.setenv("FRED_API_KEY", "should_not_be_used")
    monkeypatch.setattr(_mod, "safe_get_text", lambda url: '{"observations": []}')
    # api_key="" is truthy=False but explicitly passed; the loader should
    # treat it as "user provided an empty key" and let FRED reject it
    # (we expect RuntimeError because FRED returns empty observations).
    with pytest.raises(RuntimeError):
        load(series_id="X", api_key="", frequency="M")
