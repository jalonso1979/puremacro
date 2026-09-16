"""Adversarial Stress Tests for Milestone 4: Schema Drift, Telemetry & Canaries.

Authored by orch20_challenger_m4_1 (Empirical Challenger).

Stress-tests:
1. Adversarial Schema Drift Injection:
   - Key mutations across Banxico, INEGI, BCB, and BCCh.
   - Unexpected date formats (MM/DD/YYYY vs DD/MM/YYYY vs ISO vs invalid tokens).
   - Missing headers & HTTP error bodies formatted as 200 OK JSON.
   - Empty observation lists across all 4 central bank providers.
2. Telemetry and Warning Event Emission:
   - Verify SchemaDriftWarning and SchemaDriftError behavior.
   - Verify connector_events logging into persistent SQLite table.
3. End-to-End Graceful Fallback:
   - Verify fallback to SQLite cached vintages upon schema drift without crash.
   - Verify fallback to portable .pmz cartridges without crash.
4. Cartridge Integrity & Tampering:
   - Corrupt archives (truncated bytes, invalid zip structure).
   - Missing manifest.json.
   - Checksum mismatches on tampered payloads.
   - Unsupported format tags or versions.
"""
from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from puremacro import credentials, pocket
from puremacro._cache_db import (
    bootstrap_schema,
    close_conn,
    get_conn,
    query_realtime_vintages,
    record_connector_event,
    store_realtime_vintages,
)
from puremacro.fetch.realtime import (
    SchemaCanary,
    SchemaDriftError,
    SchemaDriftWarning,
    VintagePanel,
    load_realtime_cartridge,
    pack_realtime_cartridge,
    validate_payload,
)
from puremacro.fetch.realtime.banxico import (
    fetch_banxico_panel,
    fetch_banxico_vintages,
    parse_banxico_json,
)
from puremacro.fetch.realtime.bcb import (
    fetch_bcb_panel,
    fetch_bcb_vintages,
    parse_bcb_json,
)
from puremacro.fetch.realtime.bcch import (
    fetch_bcch_panel,
    fetch_bcch_vintages,
    parse_bcch_json,
)
from puremacro.fetch.realtime.inegi import (
    fetch_inegi_panel,
    fetch_inegi_vintages,
    parse_inegi_json,
)
from puremacro.pocket import CartridgeError


class MockHTTPResponse:
    """Mock urllib response object."""
    def __init__(self, data: bytes, code: int = 200):
        self.data = data
        self.code = code

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# ===========================================================================
# 1. Adversarial Key Mutations Across 4 Providers
# ===========================================================================

class TestAdversarialKeyMutations:
    """Adversarial stress-testing of key mutations across all 4 central banks."""

    def test_banxico_key_mutations(self):
        """Banxico: Renamed keys ('fecha'->'date', 'dato'->'val', 'datos'->'obs', 'bmx'->'data')."""
        # fecha -> date
        payload_date = {"bmx": {"series": [{"idSerie": "SF61745", "datos": [{"date": "15/01/2026", "dato": "10.75"}]}]}}
        ok, reason = SchemaCanary.validate_banxico(payload_date)
        assert not ok, "Canary must reject renamed 'fecha' -> 'date'"
        assert "missing 'fecha' or 'dato'" in reason
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(payload_date)

        # dato -> val
        payload_val = {"bmx": {"series": [{"idSerie": "SF61745", "datos": [{"fecha": "15/01/2026", "val": "10.75"}]}]}}
        ok, reason = SchemaCanary.validate_banxico(payload_val)
        assert not ok, "Canary must reject renamed 'dato' -> 'val'"
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(payload_val)

        # datos -> observations
        payload_datos = {"bmx": {"series": [{"idSerie": "SF61745", "observations": [{"fecha": "15/01/2026", "dato": "10.75"}]}]}}
        ok, reason = SchemaCanary.validate_banxico(payload_datos)
        assert not ok, "Canary must reject renamed 'datos' -> 'observations'"
        assert "missing 'datos'" in reason
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(payload_datos)

        # bmx -> data
        payload_bmx = {"data": {"series": [{"idSerie": "SF61745", "datos": [{"fecha": "15/01/2026", "dato": "10.75"}]}]}}
        ok, reason = SchemaCanary.validate_banxico(payload_bmx)
        assert not ok, "Canary must reject renamed 'bmx' -> 'data'"
        assert "missing 'bmx'" in reason
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(payload_bmx)

        # series -> series_list
        payload_series = {"bmx": {"series_list": [{"idSerie": "SF61745", "datos": [{"fecha": "15/01/2026", "dato": "10.75"}]}]}}
        ok, reason = SchemaCanary.validate_banxico(payload_series)
        assert not ok, "Canary must reject renamed 'series' -> 'series_list'"
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(payload_series)

    def test_inegi_key_mutations(self):
        """INEGI: Renamed keys ('OBS_VALUE'->'value', 'TIME_PERIOD'->'date', 'OBSERVATIONS'->'datos')."""
        # OBS_VALUE -> value
        payload_val = {"Series": [{"INDICADOR": "735848", "OBSERVATIONS": [{"TIME_PERIOD": "2025/01", "value": "24950123"}]}]}
        ok, reason = SchemaCanary.validate_inegi(payload_val)
        assert not ok, "Canary must reject renamed 'OBS_VALUE' -> 'value'"
        assert "missing 'TIME_PERIOD' or 'OBS_VALUE'" in reason
        with pytest.raises(SchemaDriftError):
            parse_inegi_json(payload_val)

        # TIME_PERIOD -> date
        payload_tp = {"Series": [{"INDICADOR": "735848", "OBSERVATIONS": [{"date": "2025/01", "OBS_VALUE": "24950123"}]}]}
        ok, reason = SchemaCanary.validate_inegi(payload_tp)
        assert not ok, "Canary must reject renamed 'TIME_PERIOD' -> 'date'"
        with pytest.raises(SchemaDriftError):
            parse_inegi_json(payload_tp)

        # OBSERVATIONS -> observations (lowercase) or datos
        payload_obs = {"Series": [{"INDICADOR": "735848", "datos": [{"TIME_PERIOD": "2025/01", "OBS_VALUE": "24950123"}]}]}
        ok, reason = SchemaCanary.validate_inegi(payload_obs)
        assert not ok, "Canary must reject renamed 'OBSERVATIONS' -> 'datos'"
        assert "missing or invalid 'OBSERVATIONS'" in reason
        with pytest.raises(SchemaDriftError):
            parse_inegi_json(payload_obs)

        # Series -> Indicators
        payload_series = {"Indicators": [{"INDICADOR": "735848", "OBSERVATIONS": [{"TIME_PERIOD": "2025/01", "OBS_VALUE": "24950123"}]}]}
        ok, reason = SchemaCanary.validate_inegi(payload_series)
        assert not ok, "Canary must reject renamed 'Series' -> 'Indicators'"
        with pytest.raises(SchemaDriftError):
            parse_inegi_json(payload_series)

    def test_bcb_key_mutations(self):
        """BCB: Renamed keys ('data'->'date'/'fecha', 'valor'->'val'/'value')."""
        # data -> date
        payload_data = [{"date": "01/01/2026", "valor": "12.25"}]
        ok, reason = SchemaCanary.validate_bcb(payload_data)
        assert not ok, "Canary must reject renamed 'data' -> 'date'"
        assert "missing 'data' or 'valor'" in reason
        with pytest.raises(SchemaDriftError):
            parse_bcb_json(payload_data)

        # valor -> val
        payload_val = [{"data": "01/01/2026", "val": "12.25"}]
        ok, reason = SchemaCanary.validate_bcb(payload_val)
        assert not ok, "Canary must reject renamed 'valor' -> 'val'"
        with pytest.raises(SchemaDriftError):
            parse_bcb_json(payload_val)

        # Root dictionary instead of list (bytes payload from network)
        payload_dict_bytes = json.dumps({"data": [{"data": "01/01/2026", "valor": "12.25"}]}).encode("utf-8")
        ok, reason = SchemaCanary.validate_bcb(json.loads(payload_dict_bytes))
        assert not ok, "Canary must reject dict root"
        assert "expected list root" in reason
        with pytest.raises(SchemaDriftError):
            parse_bcb_json(payload_dict_bytes)

        # In-memory dict root directly passed
        with pytest.raises(SchemaDriftError):
            parse_bcb_json({"error": "Resource not found"})

    def test_bcch_key_mutations(self):
        """BCCh: Renamed keys ('indexDateString'->'date', 'value'->'val', 'obs'->'records')."""
        # indexDateString -> date
        payload_date = {"Codigo": 0, "Series": {"obs": [{"date": "01-01-2026", "value": "5.50"}]}}
        ok, reason = SchemaCanary.validate_bcch(payload_date)
        assert not ok, "Canary must reject renamed 'indexDateString' -> 'date'"
        assert "missing 'indexDateString' or 'value'" in reason
        with pytest.raises(SchemaDriftError):
            parse_bcch_json(payload_date)

        # value -> val
        payload_val = {"Codigo": 0, "Series": {"obs": [{"indexDateString": "01-01-2026", "val": "5.50"}]}}
        ok, reason = SchemaCanary.validate_bcch(payload_val)
        assert not ok, "Canary must reject renamed 'value' -> 'val'"
        with pytest.raises(SchemaDriftError):
            parse_bcch_json(payload_val)

        # obs -> records
        payload_obs = {"Codigo": 0, "Series": {"records": [{"indexDateString": "01-01-2026", "value": "5.50"}]}}
        ok, reason = SchemaCanary.validate_bcch(payload_obs)
        assert not ok, "Canary must reject missing 'obs' array"
        assert ("missing 'Series' or 'obs'" in reason) or ("'obs' is not a list" in reason)
        with pytest.raises(SchemaDriftError):
            parse_bcch_json(payload_obs)


# ===========================================================================
# 2. Unexpected Date Formats Across 4 Providers
# ===========================================================================

class TestAdversarialDateFormats:
    """Stress tests for date format drift."""

    def test_banxico_date_drift_iso_and_dots(self):
        """Banxico: ISO strings, dotted dates, or slashes with year-first must be caught."""
        # ISO timestamp
        payload_iso = {"bmx": {"series": [{"idSerie": "SF61745", "datos": [{"fecha": "2026-01-15T00:00:00Z", "dato": "10.75"}]}]}}
        ok, reason = SchemaCanary.validate_banxico(payload_iso)
        assert not ok, "Canary must detect ISO timestamp drift"
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(payload_iso)

        # Dotted format (15.01.2026)
        payload_dot = {"bmx": {"series": [{"idSerie": "SF61745", "datos": [{"fecha": "15.01.2026", "dato": "10.75"}]}]}}
        ok, reason = SchemaCanary.validate_banxico(payload_dot)
        assert not ok, "Canary must detect dotted date drift"
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(payload_dot)

        # Year-first slash (2026/01/15)
        payload_slash = {"bmx": {"series": [{"idSerie": "SF61745", "datos": [{"fecha": "2026/01/15", "dato": "10.75"}]}]}}
        ok, reason = SchemaCanary.validate_banxico(payload_slash)
        assert not ok, "Canary must detect YYYY/MM/DD slash drift"
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(payload_slash)

    def test_banxico_date_drift_mm_dd_yyyy_vulnerability(self):
        """Verify MM/DD/YYYY format in Banxico (e.g. 01/15/2026) is caught by SchemaCanary."""
        payload_mm_dd = {"bmx": {"series": [{"idSerie": "SF61745", "datos": [{"fecha": "01/15/2026", "dato": "10.75"}]}]}}
        ok, reason = SchemaCanary.validate_banxico(payload_mm_dd)
        assert not ok, "Canary must reject invalid DD/MM/YYYY date"
        assert "invalid DD/MM/YYYY" in reason or "does not match" in reason
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(payload_mm_dd)

    def test_bcb_date_drift(self):
        """BCB: ISO strings or hyphens must be caught by Canary."""
        # ISO string 2026-01-15
        payload_iso = [{"data": "2026-01-15", "valor": "12.25"}]
        ok, reason = SchemaCanary.validate_bcb(payload_iso)
        assert not ok, "Canary must reject ISO date format in BCB"
        assert "DD/MM/YYYY" in reason
        with pytest.raises(SchemaDriftError):
            parse_bcb_json(payload_iso)

        # Hyphenated 01-01-2026
        payload_hyphen = [{"data": "01-01-2026", "valor": "12.25"}]
        ok, reason = SchemaCanary.validate_bcb(payload_hyphen)
        assert not ok, "Canary must reject hyphenated dates in BCB"

    def test_inegi_date_drift_unparseable(self):
        """INEGI: Unparseable TIME_PERIOD strings (e.g. CORRUPTED_DATE) must be caught by Canary."""
        payload = {"Series": [{"INDICADOR": "735848", "OBSERVATIONS": [{"TIME_PERIOD": "CORRUPTED_DATE", "OBS_VALUE": "25000000"}]}]}
        ok, reason = SchemaCanary.validate_inegi(payload)
        assert not ok, "Canary must reject unparseable TIME_PERIOD"
        assert "TIME_PERIOD" in reason
        with pytest.raises(SchemaDriftError):
            parse_inegi_json(payload)

    def test_bcch_date_drift_unparseable(self):
        """BCCh: Unparseable indexDateString strings must be caught by Canary."""
        payload = {"Codigo": 0, "Series": {"obs": [{"indexDateString": "CORRUPTED_DATE", "value": "5.50"}]}}
        ok, reason = SchemaCanary.validate_bcch(payload)
        assert not ok, "Canary must reject unparseable indexDateString"
        assert "indexDateString" in reason
        with pytest.raises(SchemaDriftError):
            parse_bcch_json(payload)


# ===========================================================================
# 3. Missing Headers & HTTP Error Bodies Formatted as 200 OK
# ===========================================================================

class TestAdversarialHTTPResponses:
    """Stress tests for HTTP error bodies formatted as 200 OK JSON."""

    def test_banxico_http_200_error_payload(self):
        """Banxico: 200 OK returning error body."""
        # Top-level error object
        err_payload1 = {"error": "401 Unauthorized", "message": "Invalid token provided"}
        ok, reason = SchemaCanary.validate_banxico(err_payload1)
        assert not ok
        assert "missing 'bmx'" in reason
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(err_payload1)

        # BMX with error
        err_payload2 = {"bmx": {"error": "Token expired"}}
        ok, reason = SchemaCanary.validate_banxico(err_payload2)
        assert not ok
        assert "missing or empty 'bmx.series'" in reason

    def test_inegi_http_200_error_payload(self):
        """INEGI: 200 OK returning error body."""
        err_payload = {"Error": "El token proporcionado no es válido", "Codigo": 401}
        ok, reason = SchemaCanary.validate_inegi(err_payload)
        assert not ok
        assert "missing or empty 'Series'" in reason
        with pytest.raises(SchemaDriftError):
            parse_inegi_json(err_payload)

    def test_bcb_http_200_error_payload(self):
        """BCB: 200 OK returning error body as dict."""
        err_payload1_bytes = json.dumps({"error": "Resource not found", "status": 404}).encode("utf-8")
        ok, reason = SchemaCanary.validate_bcb(json.loads(err_payload1_bytes))
        assert not ok
        assert "expected list root" in reason
        with pytest.raises(SchemaDriftError):
            parse_bcb_json(err_payload1_bytes)

        # List containing error object instead of observations
        err_payload2 = [{"error": "Internal database error"}]
        ok, reason = SchemaCanary.validate_bcb(err_payload2)
        assert not ok
        assert "missing 'data' or 'valor'" in reason
        with pytest.raises(SchemaDriftError):
            parse_bcb_json(err_payload2)

    def test_bcch_http_200_error_payload(self):
        """BCCh: 200 OK returning non-zero Codigo."""
        err_payload = {"Codigo": 99, "Descripcion": "Servicio de consultas temporalmente suspendido"}
        ok, reason = SchemaCanary.validate_bcch(err_payload)
        assert not ok
        assert "error code 99" in reason
        with pytest.raises(SchemaDriftError):
            parse_bcch_json(err_payload)


# ===========================================================================
# 4. Empty Observation Lists
# ===========================================================================

class TestAdversarialEmptyObservations:
    """Stress tests for empty observation lists."""

    def test_banxico_empty_observations(self):
        """Banxico: empty datos array."""
        payload = {"bmx": {"series": [{"idSerie": "SF61745", "datos": []}]}}
        ok, reason = SchemaCanary.validate_banxico(payload)
        df = parse_banxico_json(payload)
        assert df.empty

    def test_inegi_empty_observations(self):
        """INEGI: empty OBSERVATIONS array."""
        payload = {"Series": [{"INDICADOR": "735848", "OBSERVATIONS": []}]}
        ok, reason = SchemaCanary.validate_inegi(payload)
        df = parse_inegi_json(payload)
        assert df.empty

    def test_bcb_empty_observations(self):
        """BCB: empty list []."""
        payload = []
        ok, reason = SchemaCanary.validate_bcb(payload)
        df = parse_bcb_json(payload)
        assert df.empty

    def test_bcch_empty_observations(self):
        """BCCh: empty obs array."""
        payload = {"Codigo": 0, "Series": {"obs": []}}
        ok, reason = SchemaCanary.validate_bcch(payload)
        df = parse_bcch_json(payload)
        assert df.empty


# ===========================================================================
# 5. Schema Canary Telemetry & Event Logging
# ===========================================================================

class TestSchemaCanaryTelemetryAndEvents:
    """Verify warning emissions and SQLite connector_events logging."""

    def test_canary_emits_warning_and_logs_event(self, tmp_path, monkeypatch):
        """validate_payload with raise_on_drift=False emits SchemaDriftWarning and logs event."""
        db_path = tmp_path / "telemetry_test.db"
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
        close_conn()
        conn = get_conn(db_path)

        bad_payload = {"invalid_envelope": True}
        with pytest.warns(SchemaDriftWarning, match="Schema drift detected for banxico"):
            ok, reason = validate_payload("banxico", bad_payload, raise_on_drift=False)
        assert ok is False
        assert len(reason) > 0

        # Verify event was recorded in SQLite connector_events
        cur = conn.execute("SELECT source, outcome, fallback_used FROM connector_events")
        rows = cur.fetchall()
        assert len(rows) >= 1
        assert any(r[0] == "banxico" and r[1] == "schema_drift" for r in rows)
        close_conn()

    def test_canary_raise_on_drift(self):
        """validate_payload with raise_on_drift=True raises SchemaDriftError."""
        bad_payload = {"corrupted": True}
        with pytest.raises(SchemaDriftError) as exc_info:
            validate_payload("bcb", bad_payload, raise_on_drift=True)
        assert "Schema drift detected for bcb" in str(exc_info.value)


# ===========================================================================
# 6. End-to-End Graceful Fallback to SQLite Cache & .pmz Cartridges
# ===========================================================================

class TestEndToEndCacheAndCartridgeFallback:
    """Verify connectors fall back to SQLite cache or .pmz cartridges upon schema drift."""

    def test_banxico_fallback_to_sqlite_cache(self, tmp_path, monkeypatch):
        """Mutated Banxico schema must cleanly fall back to cached SQLite vintages."""
        db_path = tmp_path / "fallback_test.db"
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
        monkeypatch.setenv("BANXICO_API_KEY", "test_key")
        close_conn()
        conn = get_conn(db_path)

        # Seed cache
        seed_df = pd.DataFrame([
            {"provider": "banxico", "country": "MEX", "series_id": "SF61745", "date": "2026-01-01", "vintage": "2026-02-01", "value": 10.75},
            {"provider": "banxico", "country": "MEX", "series_id": "SF61745", "date": "2026-02-01", "vintage": "2026-02-01", "value": 10.50},
        ])
        store_realtime_vintages(seed_df, conn=conn)

        # Mutated payload (renamed key: 'dato' -> 'value')
        mutated_payload = json.dumps({
            "bmx": {
                "series": [
                    {
                        "idSerie": "SF61745",
                        "datos": [{"fecha": "01/03/2026", "value": "10.25"}]
                    }
                ]
            }
        }).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mutated_payload)):
            with pytest.warns(UserWarning, match="falling back to cached vintages"):
                df = fetch_banxico_vintages("SF61745")
                assert len(df) == 2
                assert list(df["value"]) == [10.75, 10.50]

        # Verify connector_events logged fallback
        cur = conn.execute("SELECT outcome, fallback_used FROM connector_events WHERE source='banxico'")
        events = cur.fetchall()
        assert any(e == ("fallback", "sqlite_cache") for e in events)
        close_conn()

    def test_inegi_fallback_to_sqlite_cache(self, tmp_path, monkeypatch):
        """Mutated INEGI schema must cleanly fall back to cached SQLite vintages."""
        db_path = tmp_path / "fallback_test.db"
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
        monkeypatch.setenv("INEGI_API_KEY", "test_key")
        close_conn()
        conn = get_conn(db_path)

        seed_df = pd.DataFrame([
            {"provider": "inegi", "country": "MEX", "series_id": "735848", "date": "2025-01-01", "vintage": "2025-06-01", "value": 24950123.0},
        ])
        store_realtime_vintages(seed_df, conn=conn)

        # Mutated payload (missing 'Series' envelope)
        mutated_payload = json.dumps({"Indicadores": [{"INDICADOR": "735848"}]}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mutated_payload)):
            with pytest.warns(UserWarning, match="falling back to cached vintages"):
                df = fetch_inegi_vintages("735848")
                assert len(df) == 1
                assert df.iloc[0]["value"] == 24950123.0
        close_conn()

    def test_bcb_fallback_to_sqlite_cache(self, tmp_path, monkeypatch):
        """Mutated BCB schema must cleanly fall back to cached SQLite vintages."""
        db_path = tmp_path / "fallback_test.db"
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
        close_conn()
        conn = get_conn(db_path)

        seed_df = pd.DataFrame([
            {"provider": "bcb", "country": "BRA", "series_id": "432", "date": "2026-01-01", "vintage": "2026-02-01", "value": 12.25},
        ])
        store_realtime_vintages(seed_df, conn=conn)

        # Mutated payload (dict instead of list)
        mutated_payload = json.dumps({"data": "01/02/2026", "valor": "12.00"}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mutated_payload)):
            with pytest.warns(UserWarning, match="falling back to cached vintages"):
                df = fetch_bcb_vintages("432")
                assert len(df) == 1
                assert df.iloc[0]["value"] == 12.25
        close_conn()

    def test_bcch_fallback_to_sqlite_cache(self, tmp_path, monkeypatch):
        """Mutated BCCh schema must cleanly fall back to cached SQLite vintages."""
        db_path = tmp_path / "fallback_test.db"
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
        monkeypatch.setenv("BCCH_API_USER", "user@bcch.cl")
        monkeypatch.setenv("BCCH_API_PASS", "secret")
        close_conn()
        conn = get_conn(db_path)

        seed_df = pd.DataFrame([
            {"provider": "bcch", "country": "CHL", "series_id": "F022", "date": "2026-01-01", "vintage": "2026-02-01", "value": 5.50},
        ])
        store_realtime_vintages(seed_df, conn=conn)

        # Mutated payload (API error code 99)
        mutated_payload = json.dumps({"Codigo": 99, "Descripcion": "Servicio saturado"}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=MockHTTPResponse(mutated_payload)):
            with pytest.warns(UserWarning, match="falling back to cached vintages"):
                df = fetch_bcch_vintages("F022")
                assert len(df) == 1
                assert df.iloc[0]["value"] == 5.50
        close_conn()

    def test_cartridge_as_offline_fallback(self, tmp_path):
        """Verify .pmz cartridge serves as reliable fallback when network and cache are unavailable."""
        df = pd.DataFrame([
            {"provider": "banxico", "country": "MEX", "series_id": "SF61745", "date": "2026-01-01", "vintage": "2026-02-01", "value": 10.75, "units": "rate", "variable": "policy_rate"},
            {"provider": "bcb", "country": "BRA", "series_id": "432", "date": "2026-01-01", "vintage": "2026-02-01", "value": 12.25, "units": "rate", "variable": "policy_rate"},
        ])
        panel = VintagePanel(df)

        cartridge_file = tmp_path / "offline_latam.pmz"
        pack_realtime_cartridge(panel, cartridge_file, source="LatAm Regional Backup")
        assert cartridge_file.exists()

        # Simulate offline load
        loaded_panel = load_realtime_cartridge(cartridge_file, verify=True)
        assert isinstance(loaded_panel, VintagePanel)
        assert len(loaded_panel) == 2
        assert loaded_panel.countries == ["BRA", "MEX"]
        mex_val = loaded_panel.df[loaded_panel.df["country"] == "MEX"]["value"].iloc[0]
        bra_val = loaded_panel.df[loaded_panel.df["country"] == "BRA"]["value"].iloc[0]
        assert mex_val == 10.75
        assert bra_val == 12.25


# ===========================================================================
# 7. Cartridge Integrity & Tampering Stress Tests
# ===========================================================================

class TestCartridgeIntegrityStress:
    """Stress tests for .pmz cartridge packing and loading under corruption and tampering."""

    def test_corrupt_archive_raises_cartridge_error(self, tmp_path):
        """Corrupt / non-zip file must raise CartridgeError."""
        corrupt_file = tmp_path / "corrupt.pmz"
        corrupt_file.write_bytes(b"NOT_A_VALID_ZIP_HEADER_OR_FILE")

        with pytest.raises(CartridgeError) as exc_info:
            load_realtime_cartridge(corrupt_file)
        assert "not a cartridge" in str(exc_info.value)

    def test_missing_manifest_raises_cartridge_error(self, tmp_path):
        """Zip archive lacking manifest.json must raise CartridgeError."""
        no_manifest = tmp_path / "no_manifest.pmz"
        with zipfile.ZipFile(no_manifest, "w") as zf:
            zf.writestr("frames/data.npz", b"some dummy data")

        with pytest.raises(CartridgeError) as exc_info:
            load_realtime_cartridge(no_manifest)
        assert "no manifest.json" in str(exc_info.value)

    def test_wrong_format_tag_raises_cartridge_error(self, tmp_path):
        """Manifest with invalid format tag must raise CartridgeError."""
        bad_tag_file = tmp_path / "bad_tag.pmz"
        manifest = {
            "format": "unauthorized-format",
            "version": 1,
            "provenance": {},
            "frames": []
        }
        with zipfile.ZipFile(bad_tag_file, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))

        with pytest.raises(CartridgeError) as exc_info:
            load_realtime_cartridge(bad_tag_file)
        assert "wrong format tag" in str(exc_info.value)

    def test_unsupported_version_raises_cartridge_error(self, tmp_path):
        """Manifest with future/unsupported version must raise CartridgeError."""
        future_version_file = tmp_path / "future.pmz"
        manifest = {
            "format": "puremacro-cartridge",
            "version": 999,
            "provenance": {},
            "frames": []
        }
        with zipfile.ZipFile(future_version_file, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))

        with pytest.raises(CartridgeError) as exc_info:
            load_realtime_cartridge(future_version_file)
        assert "format version 999 is not readable" in str(exc_info.value)

    def test_checksum_mismatch_raises_cartridge_error(self, tmp_path):
        """Tampered payload frame must fail SHA-256 verification and raise CartridgeError."""
        df = pd.DataFrame([
            {"provider": "bcch", "country": "CHL", "series_id": "F022", "date": "2026-01-01", "vintage": "2026-02-01", "value": 5.50}
        ])
        valid_cart = tmp_path / "valid.pmz"
        pack_realtime_cartridge(df, valid_cart)

        # Read valid zip and tamper with the data frame payload
        tampered_cart = tmp_path / "tampered.pmz"
        with zipfile.ZipFile(valid_cart, "r") as zf_in, zipfile.ZipFile(tampered_cart, "w") as zf_out:
            for item in zf_in.infolist():
                data = zf_in.read(item.filename)
                if item.filename.endswith(".npz"):
                    # Modify byte content to cause SHA-256 mismatch
                    data = data + b"_adversarial_mutation_"
                zf_out.writestr(item, data)

        with pytest.raises(CartridgeError) as exc_info:
            load_realtime_cartridge(tampered_cart, verify=True)
        assert "corrupt cartridge" in str(exc_info.value)
        assert "checksum does not match" in str(exc_info.value)


# ===========================================================================
# 8. Empirical Defect Demonstrations: Sampling Gap & Silent Drops
# ===========================================================================

class TestCanarySamplingAndSilentDropDefects:
    """Empirical reproduction of Canary blind spots and silent record dropping."""

    def test_canary_sampling_window_misses_late_mutations(self):
        """Verify SchemaCanary samples head/tail or full array, catching late mutations."""
        records = []
        for i in range(10):
            records.append({"fecha": f"{i+1:02d}/01/2026", "dato": f"10.{i:02d}"})
        for i in range(10, 50):
            # Mutate key: 'dato' -> 'val'
            records.append({"fecha": f"{i%28+1:02d}/02/2026", "val": f"11.{i:02d}"})

        payload = {"bmx": {"series": [{"idSerie": "SF61745", "datos": records}]}}
        ok, reason = SchemaCanary.validate_banxico(payload)
        assert not ok, "Canary must catch mutated key occurring after index 9"
        assert "missing 'fecha' or 'dato'" in reason
        with pytest.raises(SchemaDriftError):
            parse_banxico_json(payload)

    def test_empty_observation_list_bypasses_cache_fallback(self, tmp_path, monkeypatch):
        """Regression: when an upstream API returns an empty observation list
        (e.g. datos: []), the canary passes and parse_banxico_json returns an
        empty DataFrame; fetch_banxico_vintages must then fall back to the
        cached snapshots with a warning rather than return the empty frame.
        """
        db_path = tmp_path / "empty_obs_test.db"
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
        monkeypatch.setenv("BANXICO_API_KEY", "test_key")
        close_conn()
        conn = get_conn(db_path)

        # Seed cache with historical data
        seed_df = pd.DataFrame([
            {"provider": "banxico", "country": "MEX", "series_id": "SF61745", "date": "2026-01-01", "vintage": "2026-02-01", "value": 10.75},
        ])
        store_realtime_vintages(seed_df, conn=conn)

        # API returns 200 OK with empty datos list
        empty_datos_payload = json.dumps({
            "bmx": {
                "series": [
                    {"idSerie": "SF61745", "datos": []}
                ]
            }
        }).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=MockHTTPResponse(empty_datos_payload)):
            with pytest.warns(UserWarning, match="falling back to cached vintages"):
                result_df = fetch_banxico_vintages("SF61745")
                assert len(result_df) == 1, "Must fall back to 1-row cache"
                assert result_df.iloc[0]["value"] == 10.75

        close_conn()

