"""Comprehensive tests for Latin America Real-Time Connectors & Regional Ecosystem.

Covers:
- Credentials resolution and masking (banxico, inegi, bcch).
- SQLite realtime_vintages table bootstrap, writing, and querying.
- Realistic offline payload parsing for Banxico, INEGI, BCB, and BCCh.
- Catalog canonical variable and series specification resolution.
- Schema drift canary validation, error reporting, and warnings.
- Offline fetch simulation with SQLite cache and fallback.
- Portable .pmz cartridge packing and loading via puremacro.pocket.
"""
from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from puremacro import credentials, pocket
from puremacro._cache_db import (
    bootstrap_schema,
    query_realtime_vintages,
    record_connector_event,
    store_realtime_vintages,
)
from puremacro.fetch.realtime import (
    SchemaCanary,
    SchemaDriftError,
    SchemaDriftWarning,
    VintagePanel,
    available_providers,
    load_realtime_cartridge,
    pack_realtime_cartridge,
    providers_for,
    resolve_series,
    validate_payload,
)
from puremacro.fetch.realtime.banxico import (
    BANXICO_SERIES_URL,
    fetch_banxico_panel,
    fetch_banxico_vintages,
    parse_banxico_json,
)
from puremacro.fetch.realtime.bcb import (
    BCB_SGS_URL,
    fetch_bcb_panel,
    fetch_bcb_vintages,
    parse_bcb_json,
)
from puremacro.fetch.realtime.bcch import (
    BCCH_SIETE_URL,
    fetch_bcch_panel,
    fetch_bcch_vintages,
    parse_bcch_json,
)
from puremacro.fetch.realtime.catalog import (
    BANXICO_SERIES,
    BCB_SERIES,
    BCCH_SERIES,
    INEGI_SERIES,
    canonical_variable,
    resolve_spec,
)
from puremacro.fetch.realtime.inegi import (
    INEGI_SERIES_URL,
    fetch_inegi_panel,
    fetch_inegi_vintages,
    parse_inegi_json,
)

# ---------------------------------------------------------------------------
# Offline Realistic Fixtures
# ---------------------------------------------------------------------------

BANXICO_FIXTURE_JSON = json.dumps({
    "bmx": {
        "series": [
            {
                "idSerie": "SF61745",
                "titulo": "Tasa de interés interbancaria a 1 día (tasa objetivo)",
                "datos": [
                    {"fecha": "15/01/2026", "dato": "10.75"},
                    {"fecha": "15/02/2026", "dato": "10.50"},
                    {"fecha": "15/03/2026", "dato": "10.25"},
                    {"fecha": "15/04/2026", "dato": "N/E"},
                ],
            }
        ]
    }
}).encode("utf-8")

INEGI_GDP_FIXTURE_JSON = json.dumps({
    "Header": {
        "Name": "PIB Trimestral",
        "Email": "contacto@inegi.org.mx",
    },
    "Series": [
        {
            "INDICADOR": "735848",
            "FREQ": "Trimestral",
            "TOPIC": "1",
            "UNIT": "Millones de pesos",
            "OBSERVATIONS": [
                {"TIME_PERIOD": "2025/01", "OBS_VALUE": "24950123.0"},
                {"TIME_PERIOD": "2025/02", "OBS_VALUE": "25080456.0"},
                {"TIME_PERIOD": "2025/03", "OBS_VALUE": "25195789.0"},
                {"TIME_PERIOD": "2025/04", "OBS_VALUE": "25310012.0"},
            ],
        }
    ],
}).encode("utf-8")

INEGI_MONTHLY_FIXTURE_JSON = json.dumps({
    "Header": {
        "Name": "INPC General",
        "Email": "contacto@inegi.org.mx",
    },
    "Series": [
        {
            "INDICADOR": "628197",
            "FREQ": "Mensual",
            "TOPIC": "1",
            "UNIT": "Índice base 2da quincena julio 2018=100",
            "OBSERVATIONS": [
                {"TIME_PERIOD": "2026/01", "OBS_VALUE": "135.2"},
                {"TIME_PERIOD": "2026/02", "OBS_VALUE": "135.8"},
            ],
        }
    ],
}).encode("utf-8")

BCB_FIXTURE_JSON = json.dumps([
    {"data": "01/01/2026", "valor": "12.25"},
    {"data": "01/02/2026", "valor": "12.00"},
    {"data": "01/03/2026", "valor": "11.75"},
]).encode("utf-8")

BCCH_FIXTURE_JSON = json.dumps({
    "Codigo": 0,
    "Descripcion": "Exito",
    "Series": {
        "descripEsp": "Tasa de Política Monetaria",
        "seriesId": "F022.TPM.TPO.D001.NO.Z.D",
        "obs": [
            {"indexDateString": "01-01-2026", "value": "5.50", "statusCode": "OK"},
            {"indexDateString": "01-02-2026", "value": "5.25", "statusCode": "OK"},
            {"indexDateString": "01-03-2026", "value": "5.00", "statusCode": "OK"},
        ],
    },
}).encode("utf-8")


# ---------------------------------------------------------------------------
# 1. Credentials Integration Tests
# ---------------------------------------------------------------------------

def test_credentials_latam_registered():
    for name in ("banxico", "inegi", "bcch"):
        assert name in credentials.SERVICES
        spec = credentials.SERVICES[name]
        assert len(spec.env_vars) > 0
        assert spec.signup_url.startswith("https://")
        assert len(spec.description) > 0


def test_credentials_resolution(monkeypatch):
    monkeypatch.setenv("BANXICO_API_KEY", "test_banxico_key_123")
    monkeypatch.setenv("INEGI_API_KEY", "test_inegi_key_456")
    monkeypatch.setenv("BCCH_API_USER", "user@test.cl")

    assert credentials.get("banxico") == "test_banxico_key_123"
    assert credentials.get_credential("banxico") == "test_banxico_key_123"
    assert credentials.get("inegi") == "test_inegi_key_456"
    assert credentials.get_credential("bcch") == "user@test.cl"


def test_credentials_masking_in_status(monkeypatch):
    monkeypatch.setenv("BANXICO_API_KEY", "super_secret_token")
    df = credentials.status()
    assert isinstance(df, pd.DataFrame)
    row = df[df["service"] == "banxico"].iloc[0]
    assert bool(row["configured"]) is True
    assert row["source"] == "env:BANXICO_API_KEY"
    # Ensure raw secret is never in the status table
    assert "super_secret_token" not in str(df.to_dict())


def test_credentials_require_raises_actionable(monkeypatch):
    monkeypatch.delenv("BANXICO_API_KEY", raising=False)
    monkeypatch.delenv("BMX_TOKEN", raising=False)
    monkeypatch.delenv("PUREMACRO_BANXICO_API_KEY", raising=False)

    with pytest.raises(credentials.MissingCredentialError) as excinfo:
        credentials.require("banxico")
    msg = str(excinfo.value)
    assert "Banco de México SIE API" in msg
    assert "BANXICO_API_KEY" in msg
    assert "signup" in msg.lower() or "token_req" in msg.lower()


# ---------------------------------------------------------------------------
# 2. SQLite Real-Time Vintages Table Tests
# ---------------------------------------------------------------------------

def test_sqlite_realtime_vintages_bootstrap():
    conn = sqlite3.connect(":memory:")
    bootstrap_schema(conn)

    cur = conn.cursor()
    cur.execute("PRAGMA table_info(realtime_vintages)")
    cols = {row[1] for row in cur.fetchall()}
    assert cols == {"provider", "country", "series_id", "observation_date", "vintage_date", "value"}


def test_sqlite_store_and_query_vintages():
    conn = sqlite3.connect(":memory:")
    bootstrap_schema(conn)

    df = pd.DataFrame({
        "provider": ["banxico", "banxico", "bcb"],
        "country": ["MEX", "MEX", "BRA"],
        "series_id": ["SF61745", "SF61745", "432"],
        "date": ["2026-01-01", "2026-02-01", "2026-01-01"],
        "vintage": ["2026-03-01", "2026-03-01", "2026-03-01"],
        "value": [10.5, 10.25, 12.0],
    })

    n = store_realtime_vintages(df, conn=conn)
    assert n == 3

    # Query banxico
    res = query_realtime_vintages("banxico", "MEX", "SF61745", conn=conn)
    assert len(res) == 2
    assert list(res["value"]) == [10.5, 10.25]
    assert set(res["provider"]) == {"banxico"}

    # Query with vintage filter
    res_as_of = query_realtime_vintages("banxico", "MEX", "SF61745", vintage_date="2026-02-28", conn=conn)
    assert len(res_as_of) == 0

    # Query BCB
    res_bcb = query_realtime_vintages("bcb", "BRA", "432", conn=conn)
    assert len(res_bcb) == 1
    assert res_bcb["value"].iloc[0] == 12.0


def test_sqlite_record_connector_event():
    conn = sqlite3.connect(":memory:")
    bootstrap_schema(conn)

    record_connector_event("banxico", "success", "none", conn=conn)
    record_connector_event("inegi", "schema_drift", "sqlite_cache", conn=conn)

    cur = conn.cursor()
    cur.execute("SELECT source, outcome, fallback_used FROM connector_events ORDER BY ts")
    rows = cur.fetchall()
    assert len(rows) == 2
    assert rows[0] == ("banxico", "success", "none")
    assert rows[1] == ("inegi", "schema_drift", "sqlite_cache")


# ---------------------------------------------------------------------------
# 3. Provider Payload Parsing Tests
# ---------------------------------------------------------------------------

def test_parse_banxico_json():
    df = parse_banxico_json(BANXICO_FIXTURE_JSON, series_id="SF61745", vintage_date="2026-04-15")
    assert list(df.columns) == ["date", "vintage", "value"]
    # 3 valid records (the N/E one is dropped)
    assert len(df) == 3
    assert list(df["value"]) == [10.75, 10.50, 10.25]
    assert set(df["vintage"]) == {pd.Timestamp("2026-04-15")}
    assert set(df["date"]) == {
        pd.Timestamp("2026-01-15"),
        pd.Timestamp("2026-02-15"),
        pd.Timestamp("2026-03-15"),
    }


def test_parse_banxico_empty_and_corrupt():
    assert parse_banxico_json(b"").empty
    assert parse_banxico_json("{}").empty


def test_parse_inegi_quarterly_gdp():
    df = parse_inegi_json(INEGI_GDP_FIXTURE_JSON, series_id="735848", vintage_date="2026-05-01")
    assert len(df) == 4
    assert list(df["value"]) == [24950123.0, 25080456.0, 25195789.0, 25310012.0]
    dates = sorted(df["date"])
    assert dates == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-04-01"),
        pd.Timestamp("2025-07-01"),
        pd.Timestamp("2025-10-01"),
    ]
    assert (dates[1] - dates[0]).days == 90


def test_parse_inegi_monthly():
    df = parse_inegi_json(INEGI_MONTHLY_FIXTURE_JSON, series_id="628197", vintage_date="2026-03-01")
    assert len(df) == 2
    assert list(df["value"]) == [135.2, 135.8]
    dates = sorted(df["date"])
    assert dates == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]


def test_parse_bcb_sgs():
    df = parse_bcb_json(BCB_FIXTURE_JSON, series_id="432", vintage_date="2026-04-01")
    assert len(df) == 3
    assert list(df["value"]) == [12.25, 12.00, 11.75]
    dates = sorted(df["date"])
    assert dates == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-02-01"),
        pd.Timestamp("2026-03-01"),
    ]


def test_parse_bcch_siete():
    df = parse_bcch_json(BCCH_FIXTURE_JSON, series_id="F022.TPM.TPO.D001.NO.Z.D", vintage_date="2026-04-01")
    assert len(df) == 3
    assert list(df["value"]) == [5.50, 5.25, 5.00]
    dates = sorted(df["date"])
    assert dates == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-02-01"),
        pd.Timestamp("2026-03-01"),
    ]


# ---------------------------------------------------------------------------
# 4. Catalog Resolution Tests
# ---------------------------------------------------------------------------

def test_catalog_latam_canonical_variables():
    assert canonical_variable("policy_rate") == "policy_rate"
    assert canonical_variable("tpm") == "policy_rate"
    assert canonical_variable("selic") == "policy_rate"
    assert canonical_variable("tasa_objetivo") == "policy_rate"


def test_catalog_providers_registered():
    all_provs = available_providers()
    for prov in ("banxico", "inegi", "bcb", "bcch"):
        assert prov in all_provs


def test_catalog_series_mappings():
    # Banxico
    assert "banxico" in providers_for("MEX", "policy_rate")
    assert resolve_series("banxico", "MEX", "policy_rate") == "SF61745"
    assert resolve_spec("banxico", "MEX", "policy_rate").units == "rate"

    # INEGI
    assert "inegi" in providers_for("MEX", "gdp_real")
    assert resolve_series("inegi", "MEX", "gdp_real") == "735848"
    assert resolve_spec("inegi", "MEX", "gdp_real").units == "level"

    # BCB
    assert "bcb" in providers_for("BRA", "policy_rate")
    assert resolve_series("bcb", "BRA", "policy_rate") == "432"
    assert resolve_series("bcb", "BRA", "gdp_real") == "4380"

    # BCCh
    assert "bcch" in providers_for("CHL", "policy_rate")
    assert resolve_series("bcch", "CHL", "policy_rate") == "F022.TPM.TPO.D001.NO.Z.D"
    assert resolve_series("bcch", "CHL", "gdp_real") == "F032.PIB.VOL.Z.Z.18.Z.Z.0.Q"


# ---------------------------------------------------------------------------
# 5. Schema Drift Canary Tests
# ---------------------------------------------------------------------------

def test_schema_canary_valid_payloads():
    ok_banxico, _ = SchemaCanary.validate_banxico(BANXICO_FIXTURE_JSON)
    assert ok_banxico is True

    ok_inegi, _ = SchemaCanary.validate_inegi(INEGI_GDP_FIXTURE_JSON)
    assert ok_inegi is True

    ok_bcb, _ = SchemaCanary.validate_bcb(BCB_FIXTURE_JSON)
    assert ok_bcb is True

    ok_bcch, _ = SchemaCanary.validate_bcch(BCCH_FIXTURE_JSON)
    assert ok_bcch is True


def test_schema_canary_detects_banxico_mutations():
    # Missing 'bmx' envelope
    mutated = json.dumps({"data": [{"fecha": "15/01/2026", "dato": "10.0"}]})
    ok, reason = SchemaCanary.validate_banxico(mutated)
    assert ok is False
    assert "missing 'bmx'" in reason

    # Renamed 'dato' to 'value'
    mutated2 = json.dumps({"bmx": {"series": [{"datos": [{"fecha": "15/01/2026", "value": "10.0"}]}]}})
    ok2, reason2 = SchemaCanary.validate_banxico(mutated2)
    assert ok2 is False
    assert "missing 'fecha' or 'dato'" in reason2

    # Verify raise on drift
    with pytest.raises(SchemaDriftError):
        validate_payload("banxico", mutated, raise_on_drift=True)


def test_schema_canary_detects_inegi_mutations():
    # Missing 'Series'
    mutated = json.dumps({"Indicators": [{"OBSERVATIONS": []}]})
    ok, reason = SchemaCanary.validate_inegi(mutated)
    assert ok is False
    assert "Series" in reason


def test_schema_canary_detects_bcb_mutations():
    # Dict root instead of list
    mutated = json.dumps({"data": "01/01/2026", "valor": "12.0"})
    ok, reason = SchemaCanary.validate_bcb(mutated)
    assert ok is False
    assert "list root" in reason

    # Invalid date format
    mutated2 = json.dumps([{"data": "2026-01-01", "valor": "12.0"}])
    ok2, reason2 = SchemaCanary.validate_bcb(mutated2)
    assert ok2 is False
    assert "DD/MM/YYYY" in reason2


def test_schema_canary_detects_bcch_mutations():
    # Error code in response
    mutated = json.dumps({"Codigo": 99, "Descripcion": "Servicio no disponible"})
    ok, reason = SchemaCanary.validate_bcch(mutated)
    assert ok is False
    assert "error code 99" in reason


def test_schema_canary_emits_warning_and_logs_event(tmp_path, monkeypatch):
    db_path = tmp_path / "test_canary.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))

    bad_payload = {"unknown": True}
    with pytest.warns(SchemaDriftWarning, match="Schema drift detected"):
        ok, reason = validate_payload("banxico", bad_payload, raise_on_drift=False)
    assert ok is False


# ---------------------------------------------------------------------------
# 6. Connector Fetch & Offline Fallback Tests
# ---------------------------------------------------------------------------

class MockHTTPResponse:
    def __init__(self, data: bytes):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_banxico_fetch_with_mock_and_cache(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
    monkeypatch.setenv("BANXICO_API_KEY", "mock_key")

    # Mock urllib.request.urlopen to return BANXICO_FIXTURE_JSON
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BANXICO_FIXTURE_JSON)):
        df = fetch_banxico_vintages("SF61745", vintage_date="2026-04-01")
        assert len(df) == 3
        assert list(df["value"]) == [10.75, 10.50, 10.25]

    # Verify rows stored in SQLite
    cached = query_realtime_vintages("banxico", "MEX", "SF61745")
    assert len(cached) == 3

    # Test network failure triggers cache fallback with a warning
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("No connection")):
        with pytest.warns(UserWarning, match="falling back to cached vintages"):
            fallback_df = fetch_banxico_vintages("SF61745")
            assert len(fallback_df) == 3
            assert list(fallback_df["value"]) == [10.75, 10.50, 10.25]


def test_inegi_fetch_with_mock_and_cache(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
    monkeypatch.setenv("INEGI_API_KEY", "mock_key")

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(INEGI_GDP_FIXTURE_JSON)):
        df = fetch_inegi_vintages("735848", vintage_date="2026-05-01")
        assert len(df) == 4

    # Network failure fallback
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Timeout")):
        with pytest.warns(UserWarning, match="falling back to cached vintages"):
            fallback_df = fetch_inegi_vintages("735848")
            assert len(fallback_df) == 4


def test_bcb_fetch_with_mock_and_cache(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BCB_FIXTURE_JSON)):
        df = fetch_bcb_vintages("432", vintage_date="2026-04-01")
        assert len(df) == 3

    # Network failure fallback
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network down")):
        with pytest.warns(UserWarning, match="falling back to cached vintages"):
            fallback_df = fetch_bcb_vintages("432")
            assert len(fallback_df) == 3


def test_bcch_fetch_with_mock_and_cache(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
    monkeypatch.setenv("BCCH_API_USER", "mock_user@bcch.cl")
    monkeypatch.setenv("BCCH_API_PASS", "mock_pass")

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BCCH_FIXTURE_JSON)):
        df = fetch_bcch_vintages("F022.TPM.TPO.D001.NO.Z.D", vintage_date="2026-04-01")
        assert len(df) == 3

    # Network failure fallback
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network down")):
        with pytest.warns(UserWarning, match="falling back to cached vintages"):
            fallback_df = fetch_bcch_vintages("F022.TPM.TPO.D001.NO.Z.D")
            assert len(fallback_df) == 3


# ---------------------------------------------------------------------------
# 7. Panel Fetch Entry Points Tests
# ---------------------------------------------------------------------------

def test_fetch_latam_panels(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
    monkeypatch.setenv("BANXICO_API_KEY", "mock_key")
    monkeypatch.setenv("INEGI_API_KEY", "mock_key")
    monkeypatch.setenv("BCCH_API_USER", "mock_user")
    monkeypatch.setenv("BCCH_API_PASS", "mock_pass")

    # Banxico panel
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BANXICO_FIXTURE_JSON)):
        p_banxico = fetch_banxico_panel(["MEX"], ["policy_rate"])
        assert isinstance(p_banxico, VintagePanel)
        assert p_banxico.countries == ["MEX"]
        assert "policy_rate" in p_banxico.variables
        assert len(p_banxico) == 3

    # INEGI panel
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(INEGI_GDP_FIXTURE_JSON)):
        p_inegi = fetch_inegi_panel(["MEX"], ["gdp_real"])
        assert isinstance(p_inegi, VintagePanel)
        assert p_inegi.countries == ["MEX"]
        assert "gdp_real" in p_inegi.variables
        assert len(p_inegi) == 4

    # BCB panel
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BCB_FIXTURE_JSON)):
        p_bcb = fetch_bcb_panel(["BRA"], ["policy_rate"])
        assert isinstance(p_bcb, VintagePanel)
        assert p_bcb.countries == ["BRA"]
        assert "policy_rate" in p_bcb.variables
        assert len(p_bcb) == 3

    # BCCh panel
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BCCH_FIXTURE_JSON)):
        p_bcch = fetch_bcch_panel(["CHL"], ["policy_rate"])
        assert isinstance(p_bcch, VintagePanel)
        assert p_bcch.countries == ["CHL"]
        assert "policy_rate" in p_bcch.variables
        assert len(p_bcch) == 3


# ---------------------------------------------------------------------------
# 8. Portable Cartridge (.pmz) Integration Tests
# ---------------------------------------------------------------------------

def test_pack_and_load_latam_cartridge(tmp_path):
    rows = [
        {"country": "MEX", "variable": "policy_rate", "date": "2026-01-01", "vintage": "2026-04-01", "value": 10.75, "provider": "banxico", "series_id": "SF61745", "units": "rate"},
        {"country": "BRA", "variable": "policy_rate", "date": "2026-01-01", "vintage": "2026-04-01", "value": 12.25, "provider": "bcb", "series_id": "432", "units": "rate"},
        {"country": "CHL", "variable": "policy_rate", "date": "2026-01-01", "vintage": "2026-04-01", "value": 5.50, "provider": "bcch", "series_id": "F022.TPM.TPO.D001.NO.Z.D", "units": "rate"},
    ]
    df = pd.DataFrame(rows)
    panel = VintagePanel(df)

    pmz_file = tmp_path / "latam_test.pmz"
    pack_realtime_cartridge(panel, pmz_file, source="LatAm Central Banks", vintage="2026-04-01", notes="Test cartridge")
    assert pmz_file.exists()

    loaded_panel = load_realtime_cartridge(pmz_file, verify=True)
    assert isinstance(loaded_panel, VintagePanel)
    assert loaded_panel.countries == ["BRA", "CHL", "MEX"]
    assert loaded_panel.variables == ["policy_rate"]
    assert len(loaded_panel) == 3

    # Validate values match exactly
    mex_val = loaded_panel.df[loaded_panel.df["country"] == "MEX"]["value"].iloc[0]
    bra_val = loaded_panel.df[loaded_panel.df["country"] == "BRA"]["value"].iloc[0]
    chl_val = loaded_panel.df[loaded_panel.df["country"] == "CHL"]["value"].iloc[0]
    assert mex_val == 10.75
    assert bra_val == 12.25
    assert chl_val == 5.50
