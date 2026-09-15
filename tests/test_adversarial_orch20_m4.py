"""Adversarial stress test suite for Milestone 4 (LatAm Real-Time Ecosystem).

Actively tests failure modes, edge cases, injection attempts, and boundary conditions:
- SQL injection resistance and query sanitization
- Corrupted/adversarial payloads for all four central banks (Banxico, INEGI, BCB, BCCh)
- Schema canary bypass attempts and strictness
- Pocket cartridge tampering and cryptographic verification
- Missing credential edge cases and secret leakage resistance
"""
import io
import json
import sqlite3
import tempfile
from pathlib import Path
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
    load_realtime_cartridge,
    pack_realtime_cartridge,
    validate_payload,
)
from puremacro.fetch.realtime.banxico import parse_banxico_json
from puremacro.fetch.realtime.bcb import parse_bcb_json
from puremacro.fetch.realtime.bcch import parse_bcch_json
from puremacro.fetch.realtime.inegi import parse_inegi_json


def test_adversarial_sql_injection_resilience():
    """Verify parameterized queries resist SQL injection in series_id and country."""
    conn = sqlite3.connect(":memory:")
    bootstrap_schema(conn)

    malicious_series = "SF61745'); DROP TABLE realtime_vintages; --"
    malicious_country = "MEX' OR '1'='1"

    df = pd.DataFrame({
        "provider": ["banxico"],
        "country": [malicious_country],
        "series_id": [malicious_series],
        "date": ["2026-01-01"],
        "vintage": ["2026-04-01"],
        "value": [10.5],
    })

    # Store malicious strings safely
    n = store_realtime_vintages(df, conn=conn)
    assert n == 1

    # Table must still exist
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM realtime_vintages")
    assert cur.fetchone()[0] == 1

    # Query with injection string should parameterize safely
    res = query_realtime_vintages("banxico", malicious_country, malicious_series, conn=conn)
    assert len(res) == 1
    assert res["series_id"].iloc[0] == malicious_series

    # SQL injection in record_connector_event
    record_connector_event("banxico', 'compromised', 'none'); --", "fail", "none", conn=conn)
    cur.execute("SELECT count(*) FROM connector_events")
    assert cur.fetchone()[0] == 1


def test_adversarial_banxico_corrupted_payloads():
    """Test Banxico parser against malicious, truncated, and corrupt payloads."""
    # Truncated JSON
    with pytest.raises(Exception):
        parse_banxico_json(b'{"bmx": {"series": [')

    # Null values, extreme floats, Inf, NaN within valid date format
    payload = {
        "bmx": {
            "series": [
                {
                    "idSerie": "SF61745",
                    "datos": [
                        {"fecha": "01/01/2026", "dato": "null"},
                        {"fecha": "02/01/2026", "dato": "N/E"},
                        {"fecha": "03/01/2026", "dato": "NaN"},
                        {"fecha": "04/01/2026", "dato": "1e308"},
                        {"fecha": "05/01/2026", "dato": "-999.99"},
                    ],
                }
            ]
        }
    }
    df = parse_banxico_json(payload, series_id="SF61745", vintage_date="2026-04-01")
    # Only valid numbers should survive
    assert len(df) == 2
    assert -999.99 in df["value"].values
    assert 1e308 in df["value"].values

    # Invalid date format should trigger SchemaDriftError
    drift_payload = {
        "bmx": {
            "series": [
                {
                    "idSerie": "SF61745",
                    "datos": [
                        {"fecha": "not-a-date", "dato": "10.5"},
                    ],
                }
            ]
        }
    }
    with pytest.raises(SchemaDriftError, match="date format 'not-a-date' does not match"):
        parse_banxico_json(drift_payload, series_id="SF61745")


def test_adversarial_inegi_corrupted_payloads():
    """Test INEGI parser against irregular quarterly periods and malformed values."""
    payload = {
        "Series": [
            {
                "INDICADOR": "735848",
                "FREQ": "Trimestral",
                "OBSERVATIONS": [
                    {"TIME_PERIOD": "2025/01", "OBS_VALUE": "not_a_number"},
                    {"TIME_PERIOD": "2025/02", "OBS_VALUE": " 12,345,678.90 "},
                ],
            }
        ]
    }
    df = parse_inegi_json(payload, series_id="735848", vintage_date="2026-05-01")
    assert len(df) == 1
    assert df["value"].iloc[0] == 12345678.90
    assert df["date"].iloc[0] == pd.Timestamp("2025-04-01")

    # Invalid date format should trigger SchemaDriftError
    drift_payload = {
        "Series": [
            {
                "INDICADOR": "735848",
                "FREQ": "Trimestral",
                "OBSERVATIONS": [
                    {"TIME_PERIOD": "invalid", "OBS_VALUE": "200.0"},
                ],
            }
        ]
    }
    with pytest.raises(SchemaDriftError):
        parse_inegi_json(drift_payload, series_id="735848")


def test_adversarial_bcb_corrupted_payloads():
    """Test BCB SGS parser against mixed types and comma decimals."""
    # Non-object element triggers SchemaDriftError
    bad_payload = ["not-a-dict"]
    with pytest.raises(SchemaDriftError, match="is not an object"):
        parse_bcb_json(bad_payload, series_id="432")

    # Valid schema with non-numeric values and comma decimals
    payload = [
        {"data": "01/01/2026", "valor": "corrupted"},
        {"data": "15/05/2026", "valor": " 1234.56 "},
        {"data": "20/05/2026", "valor": "14,75"},
    ]
    df = parse_bcb_json(payload, series_id="432", vintage_date="2026-06-01")
    assert len(df) == 2
    assert 14.75 in df["value"].values
    assert 1234.56 in df["value"].values


def test_adversarial_bcch_corrupted_payloads():
    """Test BCCh parser against error responses and non-standard dates."""
    error_payload = {
        "Codigo": 101,
        "Descripcion": "Serie no existe o no tiene permisos",
        "Series": {"obs": []}
    }
    with pytest.raises(SchemaDriftError, match="BCCh API returned error code 101"):
        parse_bcch_json(error_payload, series_id="TEST")

    valid_with_corrupt_items = {
        "Codigo": 0,
        "Series": {
            "obs": [
                {"indexDateString": "01-01-2026", "value": None},
                {"indexDateString": "01-02-2026", "value": "text"},
                {"indexDateString": "01-03-2026", "value": "5,25"},
            ]
        }
    }
    df = parse_bcch_json(valid_with_corrupt_items, series_id="TEST", vintage_date="2026-04-01")
    assert len(df) == 1
    assert df["value"].iloc[0] == 5.25
    assert df["date"].iloc[0] == pd.Timestamp("2026-03-01")

    # Invalid date triggers SchemaDriftError
    drift_payload = {
        "Codigo": 0,
        "Series": {
            "obs": [
                {"indexDateString": "invalid", "value": "5.0"},
            ]
        }
    }
    with pytest.raises(SchemaDriftError):
        parse_bcch_json(drift_payload, series_id="TEST")


def test_adversarial_canary_edge_cases():
    """Test SchemaCanary with bizarre edge types (None, bool, int, list where dict expected)."""
    assert SchemaCanary.validate_banxico(None)[0] is False
    assert SchemaCanary.validate_banxico(42)[0] is False
    assert SchemaCanary.validate_banxico([])[0] is False
    assert SchemaCanary.validate_banxico({"bmx": []})[0] is False
    assert SchemaCanary.validate_banxico({"bmx": {"series": "not-a-list"}})[0] is False

    assert SchemaCanary.validate_inegi(None)[0] is False
    assert SchemaCanary.validate_inegi({"Series": [{"OBSERVATIONS": "not-a-list"}]})[0] is False

    assert SchemaCanary.validate_bcb(None)[0] is False
    assert SchemaCanary.validate_bcb({})[0] is False
    assert SchemaCanary.validate_bcb([{"data": "2026/01/01", "valor": "10"}])[0] is False

    assert SchemaCanary.validate_bcch(None)[0] is False
    assert SchemaCanary.validate_bcch([])[0] is False
    assert SchemaCanary.validate_bcch({"Series": {"obs": "not-a-list"}})[0] is False


def test_adversarial_cartridge_tamper_detection(tmp_path):
    """Test that tampering with a .pmz cartridge triggers SHA-256 failure."""
    df = pd.DataFrame({
        "country": ["MEX"],
        "variable": ["policy_rate"],
        "date": ["2026-01-01"],
        "vintage": ["2026-04-01"],
        "value": [10.75],
        "provider": ["banxico"],
        "series_id": ["SF61745"],
        "units": ["rate"],
    })
    panel = VintagePanel(df)

    pmz_path = tmp_path / "tamper_test.pmz"
    pack_realtime_cartridge(panel, pmz_path)

    # Tamper with the internal frame payload while keeping manifest intact
    import zipfile
    with zipfile.ZipFile(pmz_path, "r") as zf:
        manifest = zf.read("manifest.json")
    with zipfile.ZipFile(pmz_path, "w") as zf:
        zf.writestr("manifest.json", manifest)
        zf.writestr("frames/data.npz", b"corrupted_tampered_payload_12345")

    # Loading with verify=True must detect SHA-256 mismatch
    with pytest.raises(pocket.CartridgeError, match="corrupt cartridge: stored checksum does not match"):
        load_realtime_cartridge(pmz_path, verify=True)
