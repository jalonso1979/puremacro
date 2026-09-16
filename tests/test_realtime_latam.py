"""Comprehensive tests for Latin America Real-Time Connectors & Regional Ecosystem.

Covers:
- Credentials resolution and masking (banxico, inegi, bcch), including the
  two-part BCCh user + password credential.
- SQLite realtime_vintages table bootstrap, writing, and querying.
- Offline payload parsing for Banxico, INEGI, BCB, and BCCh against the
  providers' documented payload shapes.
- Request URL grammar for every connector (no network: urlopen is patched).
- Catalog canonical variable, frequency and series specification resolution.
- Schema drift canary validation, policies (raise / warn / ignore),
  error reporting, warnings and telemetry.
- Snapshot history: every local fetch is one vintage, and all of them are
  reachable through fetch_*_vintages, fetch_*_panel and vintage_panel.
- Offline fetch simulation with SQLite cache and fallback, use_cache=False.
- Portable .pmz cartridge packing and loading via puremacro.pocket.

No test here touches the network: every fetch patches ``urllib.request.urlopen``.
"""
from __future__ import annotations

import io
import json
import sqlite3
import urllib.error
import urllib.parse
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from puremacro import _cache_db, credentials, pocket
from puremacro._cache_db import (
    bootstrap_schema,
    query_realtime_vintages,
    record_connector_event,
    store_realtime_vintages,
)
from puremacro.fetch.realtime import (
    SUPPORTED_FREQUENCIES,
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
    vintage_panel,
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
    BCCH_SIETE_ENDPOINT,
    BCCH_SIETE_URL,
    fetch_bcch_panel,
    fetch_bcch_vintages,
    parse_bcch_json,
)
from puremacro.fetch.realtime.canary import DRIFT_POLICIES
from puremacro.fetch.realtime.catalog import (
    BANXICO_SERIES,
    BCB_SERIES,
    BCCH_SERIES,
    INEGI_SERIES,
    UNITS_TRANSFORM,
    VERIFY_ONLINE,
    canonical_variable,
    resolve_spec,
    vintage_catalog,
)
from puremacro.fetch.realtime.inegi import (
    INEGI_SERIES_URL,
    fetch_inegi_panel,
    fetch_inegi_vintages,
    parse_inegi_json,
)

# ---------------------------------------------------------------------------
# Offline fixtures in the providers' documented payload shapes
# ---------------------------------------------------------------------------

# Banxico SIE: {"bmx": {"series": [{"idSerie", "titulo", "datos": [{"fecha":
# "dd/mm/yyyy", "dato": "1,234.56"}]}]}}; "N/E" marks a missing observation.
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

# INEGI Indicadores (BIE): {"Series": [{"INDICADOR", "FREQ", "OBSERVATIONS":
# [{"TIME_PERIOD": "YYYY/NN", "OBS_VALUE": "..."}]}]}. FREQ is a numeric
# catalogue code on the live service; the parser must not depend on it —
# eight quarters across two years is what makes the pattern quarterly.
INEGI_GDP_FIXTURE_JSON = json.dumps({
    "Header": {"Name": "PIB Trimestral", "Email": "contacto@inegi.org.mx"},
    "Series": [
        {
            "INDICADOR": "735848",
            "FREQ": "5",
            "TOPIC": "1",
            "UNIT": "Millones de pesos a precios de 2018",
            "OBSERVATIONS": [
                {"TIME_PERIOD": "2024/01", "OBS_VALUE": "24500100.0"},
                {"TIME_PERIOD": "2024/02", "OBS_VALUE": "24610200.0"},
                {"TIME_PERIOD": "2024/03", "OBS_VALUE": "24720300.0"},
                {"TIME_PERIOD": "2024/04", "OBS_VALUE": "24830400.0"},
                {"TIME_PERIOD": "2025/01", "OBS_VALUE": "24950123.0"},
                {"TIME_PERIOD": "2025/02", "OBS_VALUE": "25080456.0"},
                {"TIME_PERIOD": "2025/03", "OBS_VALUE": "25195789.0"},
                {"TIME_PERIOD": "2025/04", "OBS_VALUE": "25310012.0"},
            ],
        }
    ],
}).encode("utf-8")

INEGI_QUARTER_STARTS = [
    pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01"),
    pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-01"),
    pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01"),
    pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-01"),
]

INEGI_MONTHLY_FIXTURE_JSON = json.dumps({
    "Header": {"Name": "INPC General", "Email": "contacto@inegi.org.mx"},
    "Series": [
        {
            "INDICADOR": "628197",
            "FREQ": "6",
            "TOPIC": "1",
            "UNIT": "Índice base 2da quincena julio 2018=100",
            "OBSERVATIONS": [
                {"TIME_PERIOD": "2026/01", "OBS_VALUE": "135.2"},
                {"TIME_PERIOD": "2026/02", "OBS_VALUE": "135.8"},
            ],
        }
    ],
}).encode("utf-8")

# BCB SGS (formato=json): a list of {"data": "dd/mm/yyyy", "valor": "1.23"}.
BCB_FIXTURE_JSON = json.dumps([
    {"data": "01/01/2026", "valor": "12.25"},
    {"data": "01/02/2026", "valor": "12.00"},
    {"data": "01/03/2026", "valor": "11.75"},
]).encode("utf-8")

# BCCh SIETE GetSeries: {"Codigo": 0, "Descripcion": "", "Series":
# {"descripEsp", "seriesId", "Obs": [{"indexDateString": "dd-MM-yyyy",
# "value": "123.4", "statusCode": "OK"}]}, "SeriesInfos": []}; value "NaN"
# where the observation is missing. Capital O in Obs.
BCCH_FIXTURE_JSON = json.dumps({
    "Codigo": 0,
    "Descripcion": "",
    "Series": {
        "descripEsp": "Tasa de Política Monetaria",
        "seriesId": "F022.TPM.TPO.D001.NO.Z.D",
        "Obs": [
            {"indexDateString": "01-01-2026", "value": "5.50", "statusCode": "OK"},
            {"indexDateString": "01-02-2026", "value": "5.25", "statusCode": "OK"},
            {"indexDateString": "01-03-2026", "value": "5.00", "statusCode": "OK"},
            {"indexDateString": "01-04-2026", "value": "NaN", "statusCode": "ND"},
        ],
    },
    "SeriesInfos": [],
}).encode("utf-8")


class MockHTTPResponse:
    def __init__(self, data: bytes):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    """A private cache DB for the test; the singleton is reset around it."""
    db_path = tmp_path / "cache.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
    monkeypatch.delenv("PUREMACRO_NARRATIVE_TELEMETRY", raising=False)
    _cache_db.close_conn()
    yield db_path
    _cache_db.close_conn()


@pytest.fixture
def no_credentials(tmp_path, monkeypatch):
    """No token or user/password anywhere: env cleared, TOML absent."""
    for var in ("BANXICO_API_KEY", "BMX_TOKEN", "PUREMACRO_BANXICO_API_KEY",
                "INEGI_API_KEY", "PUREMACRO_INEGI_API_KEY",
                "BCCH_API_USER", "PUREMACRO_BCCH_API_USER",
                "BCCH_API_PASS", "PUREMACRO_BCCH_API_PASS",
                "BCCH_API_KEY", "PUREMACRO_BCCH_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
    monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)


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


def test_credentials_resolution(monkeypatch, no_credentials):
    monkeypatch.setenv("BANXICO_API_KEY", "test_banxico_key_123")
    monkeypatch.setenv("INEGI_API_KEY", "test_inegi_key_456")
    monkeypatch.setenv("BCCH_API_USER", "user@test.cl")
    monkeypatch.setenv("BCCH_API_PASS", "pw")

    assert credentials.get("banxico") == "test_banxico_key_123"
    assert credentials.get_credential("banxico") == "test_banxico_key_123"
    assert credentials.get("inegi") == "test_inegi_key_456"
    assert credentials.get_credential("bcch") == "user@test.cl"
    assert credentials.get_password("bcch") == "pw"
    assert credentials.require_credential("bcch") == "user@test.cl"


def test_credentials_masking_in_status(monkeypatch):
    monkeypatch.setenv("BANXICO_API_KEY", "super_secret_token")
    df = credentials.status()
    assert isinstance(df, pd.DataFrame)
    row = df[df["service"] == "banxico"].iloc[0]
    assert bool(row["configured"]) is True
    assert row["source"] == "env:BANXICO_API_KEY"
    # Ensure raw secret is never in the status table
    assert "super_secret_token" not in str(df.to_dict())


def test_credentials_require_raises_actionable(monkeypatch, no_credentials):
    with pytest.raises(credentials.MissingCredentialError) as excinfo:
        credentials.require("banxico")
    msg = str(excinfo.value)
    assert "Banco de México SIE API" in msg
    assert "BANXICO_API_KEY" in msg
    assert "signup" in msg.lower() or "token_req" in msg.lower()


def test_bcch_password_alone_is_not_a_credential(monkeypatch, no_credentials):
    """Regression: BCCH_API_PASS used to be in the generic env-var list, so a
    password alone was returned as the credential and sent as the user."""
    monkeypatch.setenv("BCCH_API_PASS", "my_secret_pass")
    assert credentials.get("bcch") is None
    assert credentials.get_password("bcch") == "my_secret_pass"
    assert "BCCH_API_PASS" not in credentials.SERVICES["bcch"].env_vars
    row = credentials.status().set_index("service").loc["bcch"]
    assert bool(row["configured"]) is False
    assert "my_secret_pass" not in credentials.status().to_string()
    with pytest.raises(credentials.MissingCredentialError) as excinfo:
        credentials.require("bcch")
    msg = str(excinfo.value)
    assert "user and a password" in msg
    assert "BCCH_API_USER" in msg and "BCCH_API_PASS" in msg


def test_bcch_user_without_password_is_incomplete(monkeypatch, no_credentials):
    monkeypatch.setenv("BCCH_API_USER", "user@test.cl")
    assert credentials.get("bcch") == "user@test.cl"
    assert credentials.get_password("bcch") is None
    row = credentials.status().set_index("service").loc["bcch"]
    assert bool(row["configured"]) is False
    assert "no password" in row["source"]
    with pytest.raises(credentials.MissingCredentialError, match="password"):
        credentials.require("bcch")


def test_bcch_two_part_credential_from_toml(tmp_path, monkeypatch, no_credentials):
    """The TOML file carries both halves as [bcch].user / [bcch].password."""
    cfg = tmp_path / "credentials.toml"
    cfg.write_text('[bcch]\nuser = "cfg@x.cl"\npassword = "cfg_pw"\n', encoding="utf-8")
    monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg))
    monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)
    assert credentials.get("bcch") == "cfg@x.cl"
    assert credentials.get_password("bcch") == "cfg_pw"
    assert credentials.require("bcch") == "cfg@x.cl"
    row = credentials.status().set_index("service").loc["bcch"]
    assert bool(row["configured"]) is True
    assert row["source"] == "config_file"
    # [bcch].api_key is still read as the user, for configs written for 3.4.0-dev.
    cfg.write_text('[bcch]\napi_key = "legacy@x.cl"\npassword = "cfg_pw"\n', encoding="utf-8")
    monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)
    assert credentials.get("bcch") == "legacy@x.cl"


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


def test_record_connector_event_never_raises(fresh_cache, monkeypatch):
    """Telemetry never breaks a fetch: a broken DB warns and returns."""
    closed = sqlite3.connect(":memory:")
    closed.close()
    with pytest.warns(UserWarning, match="record_connector_event failed"):
        record_connector_event("bcb", "success", "none", conn=closed)
    # Kill-switch: no row is written, no error either.
    monkeypatch.setenv("PUREMACRO_NARRATIVE_TELEMETRY", "0")
    conn = _cache_db.get_conn()
    before = conn.execute("SELECT count(*) FROM connector_events").fetchone()[0]
    record_connector_event("bcb", "success", "none")
    after = conn.execute("SELECT count(*) FROM connector_events").fetchone()[0]
    assert before == after


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


def test_parse_banxico_thousands_separator():
    payload = {"bmx": {"series": [{"idSerie": "SP1", "titulo": "INPC", "datos": [
        {"fecha": "01/01/2026", "dato": "1,234.56"},
        {"fecha": "01/02/2026", "dato": "N/E"},
    ]}]}}
    df = parse_banxico_json(payload, series_id="SP1", vintage_date="2026-03-01")
    assert list(df["value"]) == [1234.56]


def test_parse_banxico_empty_and_corrupt():
    assert parse_banxico_json(b"").empty
    assert parse_banxico_json("{}").empty


def test_parse_inegi_quarterly_gdp():
    df = parse_inegi_json(INEGI_GDP_FIXTURE_JSON, series_id="735848", vintage_date="2026-05-01")
    assert len(df) == 8
    assert list(df["value"])[-4:] == [24950123.0, 25080456.0, 25195789.0, 25310012.0]
    assert sorted(df["date"]) == INEGI_QUARTER_STARTS


def test_parse_inegi_monthly():
    df = parse_inegi_json(INEGI_MONTHLY_FIXTURE_JSON, series_id="628197", vintage_date="2026-03-01")
    assert len(df) == 2
    assert list(df["value"]) == [135.2, 135.8]
    dates = sorted(df["date"])
    assert dates == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]


def test_parse_inegi_quarters_are_inferred_not_hardcoded():
    """Regression: the quarter mapping used to fire only for series_id
    '735848'; any other quarterly indicator was dated Jan..Apr."""
    payload = json.loads(INEGI_GDP_FIXTURE_JSON)
    payload["Series"][0]["INDICADOR"] = "493911"
    for sid in ("493911", ""):
        df = parse_inegi_json(payload, series_id=sid, vintage_date="2026-05-01")
        assert sorted(df["date"]) == INEGI_QUARTER_STARTS, sid


def test_parse_inegi_structure_beats_frequency_hints():
    """Twelve periods in a year cannot be quarters whatever FREQ or the
    catalogue says; a short series falls back to the hints."""
    monthly = {"Series": [{"INDICADOR": "x", "FREQ": "Trimestral", "OBSERVATIONS": [
        {"TIME_PERIOD": f"2024/{m:02d}", "OBS_VALUE": str(m)} for m in range(1, 13)
    ]}]}
    df = parse_inegi_json(monthly, freq="Q", vintage_date="2026-01-01")
    assert df["date"].dt.month.tolist() == list(range(1, 13))

    short_obs = [{"TIME_PERIOD": "2025/01", "OBS_VALUE": "1"},
                 {"TIME_PERIOD": "2025/02", "OBS_VALUE": "2"}]
    q_from_catalog = parse_inegi_json(
        {"Series": [{"INDICADOR": "x", "OBSERVATIONS": short_obs}]}, freq="Q",
        vintage_date="2026-01-01")
    assert sorted(q_from_catalog["date"]) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01")]
    q_from_field = parse_inegi_json(
        {"Series": [{"INDICADOR": "x", "FREQ": "Trimestral", "OBSERVATIONS": short_obs}]},
        vintage_date="2026-01-01")
    assert sorted(q_from_field["date"]) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01")]
    m_default = parse_inegi_json(
        {"Series": [{"INDICADOR": "x", "OBSERVATIONS": short_obs}]}, vintage_date="2026-01-01")
    assert sorted(m_default["date"]) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")]


def test_parse_bcb_scalar_body_is_schema_drift():
    """A JSON scalar where the list should be is drift, not a TypeError
    from ``len()`` (regression guard for the shared load_json path)."""
    for body in (b"5", b"true"):
        with pytest.raises(SchemaDriftError, match="expected list root"):
            parse_bcb_json(body)
    with pytest.raises(SchemaDriftError, match="Schema drift detected for bcb"):
        parse_bcb_json(b'"texto"')
    assert parse_bcb_json(b"[]").empty
    assert parse_bcb_json(b"null").empty


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
    """The documented SIETE shape: Series.Obs (capital O), statusCode, NaN."""
    df = parse_bcch_json(BCCH_FIXTURE_JSON, series_id="F022.TPM.TPO.D001.NO.Z.D", vintage_date="2026-04-01")
    assert len(df) == 3                      # the NaN observation is dropped
    assert list(df["value"]) == [5.50, 5.25, 5.00]
    dates = sorted(df["date"])
    assert dates == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-02-01"),
        pd.Timestamp("2026-03-01"),
    ]
    ok, reason = SchemaCanary.validate_bcch(BCCH_FIXTURE_JSON)
    assert ok is True, reason


def test_parse_bcch_lowercase_obs_still_accepted():
    """Older fixtures spell the key 'obs'; both spellings must parse."""
    payload = {"Codigo": 0, "Descripcion": "", "Series": {
        "obs": [{"indexDateString": "01-01-2026", "value": "5.5", "statusCode": "OK"}]}}
    assert SchemaCanary.validate_bcch(payload) == (True, "")
    assert list(parse_bcch_json(payload, vintage_date="2026-02-01")["value"]) == [5.5]
    root = {"Codigo": 0, "obs": [{"indexDateString": "01-01-2026", "value": "5.5"}]}
    assert list(parse_bcch_json(root, vintage_date="2026-02-01")["value"]) == [5.5]


def test_parsers_count_malformed_rows():
    """Rows that cannot be read are dropped with a warning that counts them;
    documented missing markers are not counted."""
    with pytest.warns(UserWarning, match="skipped 1 of 3"):
        df = parse_bcb_json([
            {"data": "01/01/2026", "valor": "garbage"},
            {"data": "02/01/2026", "valor": ""},
            {"data": "03/01/2026", "valor": "1.5"},
        ], vintage_date="2026-02-01")
    assert list(df["value"]) == [1.5]
    with pytest.warns(UserWarning, match="parse_bcch_json skipped 1 of 3"):
        parse_bcch_json({"Codigo": 0, "Series": {"Obs": [
            {"indexDateString": "01-01-2026", "value": "text"},
            {"indexDateString": "01-02-2026", "value": "NaN"},
            {"indexDateString": "01-03-2026", "value": "5.0"},
        ]}}, vintage_date="2026-02-01")


# ---------------------------------------------------------------------------
# 4. Request URL grammar (documented endpoints, no network)
# ---------------------------------------------------------------------------

def test_inegi_url_matches_documented_constructor():
    """INDICATOR / id / es / geo / recientes=false / BIE / 2.0 / token ?type=json.

    'false' matters: 'true' returns only the latest datum, which would make
    every snapshot a single observation and the cache useless."""
    url = INEGI_SERIES_URL.format(series_id="735848", token="TOKEN")
    path, _, query = url.partition("?")
    segments = path.split("/jsonxml/", 1)[1].split("/")
    assert segments == ["INDICATOR", "735848", "es", "00", "false", "BIE", "2.0", "TOKEN"]
    assert query == "type=json"
    assert url.startswith("https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/")


def test_banxico_url_and_token_header(fresh_cache, monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["token"] = req.get_header("Bmx-token")
        return MockHTTPResponse(BANXICO_FIXTURE_JSON)

    monkeypatch.setenv("BANXICO_API_KEY", "mock_key")
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        fetch_banxico_vintages("SF61745", vintage_date="2026-04-01")
    assert seen["url"] == BANXICO_SERIES_URL.format(series_id="SF61745")
    assert seen["url"].endswith("/series/SF61745/datos")
    assert seen["token"] == "mock_key"


def test_bcb_url_grammar():
    assert BCB_SGS_URL.format(series_id="432") == (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados?formato=json")


def test_bcch_url_matches_documented_endpoint(fresh_cache, monkeypatch):
    """SieteRestWS.ashx with user / pass / function=GetSeries / timeseries /
    firstdate / lastdate, every value URL-encoded."""
    assert BCCH_SIETE_ENDPOINT.endswith("/SieteRestWS/SieteRestWS.ashx")
    assert BCCH_SIETE_URL.startswith(BCCH_SIETE_ENDPOINT + "?")
    template_params = dict(p.split("=", 1) for p in BCCH_SIETE_URL.split("?", 1)[1].split("&"))
    assert set(template_params) == {"user", "pass", "function", "timeseries", "firstdate", "lastdate"}
    assert template_params["function"] == "GetSeries"

    monkeypatch.setenv("BCCH_API_USER", "analyst+x@bank.cl")
    monkeypatch.setenv("BCCH_API_PASS", "p&ss#w%rd")
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return MockHTTPResponse(BCCH_FIXTURE_JSON)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        fetch_bcch_vintages("F022.TPM.TPO.D001.NO.Z.D", vintage_date="2026-04-01",
                            firstdate="2026-01-01", lastdate="2026-03-31")
    base, _, query = seen["url"].partition("?")
    assert base == BCCH_SIETE_ENDPOINT
    params = dict(urllib.parse.parse_qsl(query, keep_blank_values=True))
    assert params == {
        "user": "analyst+x@bank.cl", "pass": "p&ss#w%rd", "function": "GetSeries",
        "timeseries": "F022.TPM.TPO.D001.NO.Z.D",
        "firstdate": "2026-01-01", "lastdate": "2026-03-31",
    }
    assert "p&ss#w%rd" not in query          # encoded, never raw
    assert "password=" not in query          # the parameter is `pass`


def test_bcch_fetch_redacts_credentials_from_errors_and_warnings(fresh_cache, monkeypatch):
    """The password travels in the query string; it must not travel further."""
    monkeypatch.setenv("BCCH_API_USER", "analyst@bank.cl")
    monkeypatch.setenv("BCCH_API_PASS", "SENSITIVE_CHILE_PASSWORD_123")

    def unauthorized(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

    with patch("urllib.request.urlopen", side_effect=unauthorized):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            fetch_bcch_vintages("F022.TPM.TPO.D001.NO.Z.D")
    err = excinfo.value
    assert err.code == 401
    for text in (getattr(err, "url", ""), getattr(err, "filename", ""), str(err)):
        assert "SENSITIVE_CHILE_PASSWORD_123" not in (text or "")
        assert "analyst@bank.cl" not in (text or "")
    assert "user=***" in err.url and "pass=***" in err.url

    # With a cached snapshot the failure becomes a warning; scrubbed too.
    store_realtime_vintages(pd.DataFrame([{
        "provider": "bcch", "country": "CHL", "series_id": "F022.TPM.TPO.D001.NO.Z.D",
        "date": "2026-01-01", "vintage": "2026-02-01", "value": 5.0}]))

    def down(req, timeout=None):
        raise urllib.error.URLError("unreachable " + req.full_url)

    with patch("urllib.request.urlopen", side_effect=down):
        with pytest.warns(UserWarning, match="falling back to cached vintages") as rec:
            df = fetch_bcch_vintages("F022.TPM.TPO.D001.NO.Z.D")
    assert len(df) == 1
    assert all("SENSITIVE_CHILE_PASSWORD_123" not in str(w.message) for w in rec)


# ---------------------------------------------------------------------------
# 5. Catalog Resolution Tests
# ---------------------------------------------------------------------------

def test_catalog_latam_canonical_variables():
    assert canonical_variable("policy_rate") == "policy_rate"
    assert canonical_variable("tpm") == "policy_rate"
    assert canonical_variable("selic") == "policy_rate"
    assert canonical_variable("tasa_objetivo") == "policy_rate"
    # Monthly activity indices are not industrial production.
    for alias in ("activity", "igae", "imacec", "ibc_br", "economic_activity"):
        assert canonical_variable(alias) == "activity"


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

    # BCB: 4380 is monthly nominal GDP in BRL millions, not real GDP;
    # 22099 is the quarterly seasonally adjusted chained-volume index.
    assert "bcb" in providers_for("BRA", "policy_rate")
    assert resolve_series("bcb", "BRA", "policy_rate") == "432"
    assert resolve_series("bcb", "BRA", "gdp_real") == "22099"
    assert resolve_spec("bcb", "BRA", "gdp_real").units == "index"
    # 433 is the IPCA monthly % change, so it must not be log-differenced.
    assert resolve_spec("bcb", "BRA", "cpi").units == "growth_mom"
    assert resolve_spec("bcb", "BRA", "cpi").default_transform() == "level"
    assert UNITS_TRANSFORM["growth_mom"] == "level"

    # BCCh: quarterly codes end in .T, never .Q; IPC is chapter F074.
    assert "bcch" in providers_for("CHL", "policy_rate")
    assert resolve_series("bcch", "CHL", "policy_rate") == "F022.TPM.TPO.D001.NO.Z.D"
    gdp_code = resolve_series("bcch", "CHL", "gdp_real")
    assert gdp_code.endswith(".T") and not gdp_code.endswith(".Q")
    assert resolve_series("bcch", "CHL", "cpi").startswith("F074.IPC.")

    # Activity indices live under 'activity', and 'ip' is no longer claimed.
    for prov, country in (("banxico", "MEX"), ("inegi", "MEX"), ("bcb", "BRA"), ("bcch", "CHL")):
        assert resolve_series(prov, country, "activity") is not None
        assert resolve_series(prov, country, "ip") is None


def test_catalog_latam_frequencies_declared():
    """Policy rates are daily, price and activity indices monthly, GDP
    quarterly — declared so vintage_panel cannot stamp them 'Q'."""
    expected = {"policy_rate": "D", "cpi": "M", "activity": "M", "gdp_real": "Q"}
    for table in (BANXICO_SERIES, INEGI_SERIES, BCB_SERIES, BCCH_SERIES):
        for variable, spec in table.items():
            assert spec.freq == expected[variable], (spec.series_id, variable)
    cat = vintage_catalog()
    assert "freq" in cat.columns
    assert set(cat.loc[cat["provider"].isin(["banxico", "inegi", "bcb", "bcch"]), "freq"]) <= {"Q", "M", "D"}
    assert (cat.loc[cat["provider"] == "oecd_stes", "freq"] == "").all()


def test_catalog_unverified_ids_are_marked():
    """Identifiers that could not be checked against the live service say
    so in their note, so the live audit knows where to look."""
    cat = vintage_catalog()
    latam = cat[cat["provider"].isin(["banxico", "inegi", "bcb", "bcch"])]
    flagged = latam[latam["note"].str.contains(VERIFY_ONLINE)]
    assert set(zip(flagged["provider"], flagged["variable"])) >= {
        ("bcch", "gdp_real"), ("bcch", "cpi"), ("inegi", "gdp_real"),
    }
    # The ones the connectors were written against are not flagged.
    unflagged = latam[~latam["note"].str.contains(VERIFY_ONLINE)]
    assert {("bcb", "gdp_real"), ("bcb", "cpi"), ("bcb", "policy_rate"),
            ("bcch", "policy_rate"), ("banxico", "policy_rate")} <= set(
        zip(unflagged["provider"], unflagged["variable"]))


# ---------------------------------------------------------------------------
# 6. Schema Drift Canary Tests
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
    # Dict root instead of list: the upstream text is surfaced
    mutated = json.dumps({"erro": "consulta limitada a 10 anos", "valor": "12.0"})
    ok, reason = SchemaCanary.validate_bcb(mutated)
    assert ok is False
    assert "list root" in reason
    assert "consulta limitada a 10 anos" in reason

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
    # Neither spelling of the observation list
    ok2, reason2 = SchemaCanary.validate_bcch({"Codigo": 0, "Series": {"records": []}})
    assert ok2 is False
    assert "Series.Obs" in reason2


def test_schema_canary_emits_warning_and_logs_event(fresh_cache):
    bad_payload = {"unknown": True}
    with pytest.warns(SchemaDriftWarning, match="Schema drift detected"):
        ok, reason = validate_payload("banxico", bad_payload, raise_on_drift=False)
    assert ok is False
    rows = _cache_db.get_conn().execute(
        "SELECT source, outcome, fallback_used FROM connector_events").fetchall()
    assert ("banxico", "schema_drift", "warning_emitted") in rows


def test_drift_policy_warn_lets_parser_fallbacks_work(fresh_cache):
    """Regression: the canary used to raise before the parser's own ISO-date
    fallback could run, and the fetch never recorded a schema_drift event."""
    iso = [{"data": "2026-01-01", "valor": "1.0"}]
    assert DRIFT_POLICIES == ("raise", "warn", "ignore")
    with pytest.raises(SchemaDriftError, match="Schema drift detected for bcb"):
        parse_bcb_json(iso)
    with pytest.warns(SchemaDriftWarning, match="does not match DD/MM/YYYY"):
        df = parse_bcb_json(iso, vintage_date="2026-02-01", on_drift="warn")
    assert list(df["date"]) == [pd.Timestamp("2026-01-01")]
    df2 = parse_bcb_json(iso, vintage_date="2026-02-01", on_drift="ignore")
    assert list(df2["value"]) == [1.0]
    rows = _cache_db.get_conn().execute(
        "SELECT outcome, fallback_used FROM connector_events WHERE source='bcb'").fetchall()
    assert ("schema_drift", "none") in rows
    assert ("schema_drift", "warning_emitted") in rows
    assert ("schema_drift", "ignored") in rows
    with pytest.raises(ValueError, match="on_drift"):
        parse_bcb_json(iso, on_drift="explode")


def test_invalid_on_drift_is_rejected_before_any_drift_occurs():
    """A mistyped policy must fail on the first call, valid payload or
    not, rather than surface months later inside a fallback path."""
    valid = [{"data": "01/01/2026", "valor": "1.0"}]
    with pytest.raises(ValueError, match="on_drift='explode'"):
        parse_bcb_json(valid, on_drift="explode")
    with pytest.raises(ValueError, match="on_drift"):
        SchemaCanary.check("bcb", valid, on_drift="explode")
    with pytest.raises(ValueError, match="on_drift"):
        parse_banxico_json(BANXICO_FIXTURE_JSON, on_drift="")
    # The three policies, in any case, are accepted on a valid payload.
    for policy in DRIFT_POLICIES + ("RAISE", "Warn"):
        assert len(parse_bcb_json(valid, vintage_date="2026-02-01", on_drift=policy)) == 1


def test_fetch_on_drift_policies(fresh_cache):
    """Through the fetch path: 'raise' falls back to the cache with a
    SchemaDriftWarning; 'warn' parses the live payload."""
    store_realtime_vintages(pd.DataFrame([{
        "provider": "bcb", "country": "BRA", "series_id": "432",
        "date": "2026-01-01", "vintage": "2026-02-01", "value": 12.25}]))
    iso_body = json.dumps([{"data": "2026-03-01", "valor": "11.5"}]).encode("utf-8")
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(iso_body)):
        with pytest.warns(SchemaDriftWarning, match="falling back to cached vintages"):
            df = fetch_bcb_vintages("432")
        assert list(df["value"]) == [12.25]
        with pytest.warns(SchemaDriftWarning, match="Schema drift detected for bcb"):
            df_live = fetch_bcb_vintages("432", vintage_date="2026-03-15", on_drift="warn")
    # Both snapshots are now part of the history.
    assert sorted(df_live["vintage"].unique()) == [pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-15")]
    assert 11.5 in df_live["value"].values


# ---------------------------------------------------------------------------
# 7. Connector Fetch, Snapshot History & Offline Fallback Tests
# ---------------------------------------------------------------------------

def test_banxico_fetch_with_mock_and_cache(fresh_cache, monkeypatch):
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


def test_inegi_fetch_with_mock_and_cache(fresh_cache, monkeypatch):
    monkeypatch.setenv("INEGI_API_KEY", "mock_key")

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(INEGI_GDP_FIXTURE_JSON)):
        df = fetch_inegi_vintages("735848", vintage_date="2026-05-01")
        assert len(df) == 8

    # Network failure fallback
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Timeout")):
        with pytest.warns(UserWarning, match="falling back to cached vintages"):
            fallback_df = fetch_inegi_vintages("735848")
            assert len(fallback_df) == 8


def test_bcb_fetch_with_mock_and_cache(fresh_cache):
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BCB_FIXTURE_JSON)):
        df = fetch_bcb_vintages("432", vintage_date="2026-04-01")
        assert len(df) == 3

    # Network failure fallback
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network down")):
        with pytest.warns(UserWarning, match="falling back to cached vintages"):
            fallback_df = fetch_bcb_vintages("432")
            assert len(fallback_df) == 3


def test_bcch_fetch_with_mock_and_cache(fresh_cache, monkeypatch):
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


def test_snapshot_history_is_returned_as_vintages(fresh_cache):
    """Regression: connectors stored every snapshot but returned only
    today's, so revision tracking was unreachable through the public API."""
    first = json.dumps([{"data": "01/01/2026", "valor": "12.25"},
                        {"data": "01/02/2026", "valor": "12.00"}]).encode("utf-8")
    second = json.dumps([{"data": "01/01/2026", "valor": "12.30"},   # revised
                         {"data": "01/02/2026", "valor": "12.00"},
                         {"data": "01/03/2026", "valor": "11.75"}]).encode("utf-8")
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(first)):
        df1 = fetch_bcb_vintages("432", vintage_date="2026-02-15")
    assert sorted(df1["vintage"].unique()) == [pd.Timestamp("2026-02-15")]
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(second)):
        df2 = fetch_bcb_vintages("432", vintage_date="2026-03-15")
        only_today = fetch_bcb_vintages("432", vintage_date="2026-03-15", history=False)
    assert sorted(df2["vintage"].unique()) == [pd.Timestamp("2026-02-15"), pd.Timestamp("2026-03-15")]
    assert len(df2) == 5
    assert sorted(only_today["vintage"].unique()) == [pd.Timestamp("2026-03-15")]
    assert len(only_today) == 3

    # The panel entry points see the same history. They stamp today's
    # snapshot themselves, which adds a third vintage on top of the two
    # dated ones.
    today = pd.Timestamp.now(tz=None).normalize()
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(second)):
        panel = fetch_bcb_panel(["BRA"], ["policy_rate"])
        wide = vintage_panel(["BRA"], series="policy_rate", providers="bcb", freq="D")
    for p in (panel, wide):
        assert p.coverage()["n_vintages"].tolist() == [3]
        assert sorted(p.df["vintage"].unique()) == [
            pd.Timestamp("2026-02-15"), pd.Timestamp("2026-03-15"), today]
    assert wide.metadata["freq"] == "D"
    assert wide.metadata["failed"] == {}
    long = wide.long("BRA", "policy_rate")
    jan = long[long["date"] == pd.Timestamp("2026-01-01")].sort_values("vintage")
    assert jan["value"].tolist() == [12.25, 12.30, 12.30]      # first estimate, then revised
    assert wide.as_of("2026-02-28").loc[("BRA", pd.Timestamp("2026-01-01")), "policy_rate"] == 12.25
    assert wide.as_of("2026-03-31").loc[("BRA", pd.Timestamp("2026-01-01")), "policy_rate"] == 12.30


def test_use_cache_false_never_touches_the_cache(fresh_cache, monkeypatch, no_credentials):
    """Regression: use_cache=False still served cached rows on failure."""
    store_realtime_vintages(pd.DataFrame([{
        "provider": "bcb", "country": "BRA", "series_id": "432",
        "date": "2026-01-01", "vintage": "2026-02-01", "value": 12.25}]))
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(urllib.error.URLError):
            fetch_bcb_vintages("432", use_cache=False)
    # A fresh fetch with use_cache=False is returned as parsed and not stored.
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BCB_FIXTURE_JSON)):
        df = fetch_bcb_vintages("432", vintage_date="2026-04-01", use_cache=False)
    assert sorted(df["vintage"].unique()) == [pd.Timestamp("2026-04-01")]
    assert len(query_realtime_vintages("bcb", "BRA", "432")) == 1
    # Missing credentials with use_cache=False do not fall back to the cache either.
    store_realtime_vintages(pd.DataFrame([{
        "provider": "banxico", "country": "MEX", "series_id": "SF61745",
        "date": "2026-01-01", "vintage": "2026-02-01", "value": 10.0}]))
    assert len(fetch_banxico_vintages("SF61745")) == 1
    with pytest.raises(credentials.MissingCredentialError):
        fetch_banxico_vintages("SF61745", use_cache=False)


def test_telemetry_failure_does_not_break_fetch(fresh_cache, monkeypatch):
    """Regression: a raising record_connector_event turned a good fetch
    into a fallback or an exception."""
    def broken(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(_cache_db, "record_connector_event", broken)
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BCB_FIXTURE_JSON)):
        with pytest.warns(UserWarning, match="could not record connector telemetry"):
            df = fetch_bcb_vintages("432", vintage_date="2026-04-01")
    assert len(df) == 3
    assert len(query_realtime_vintages("bcb", "BRA", "432")) == 3


def test_fetch_records_success_and_fallback_events(fresh_cache):
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BCB_FIXTURE_JSON)):
        fetch_bcb_vintages("432", vintage_date="2026-04-01")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.warns(UserWarning, match="falling back"):
            fetch_bcb_vintages("432")
    rows = _cache_db.get_conn().execute(
        "SELECT outcome, fallback_used FROM connector_events WHERE source='bcb'").fetchall()
    assert ("success", "none") in rows
    assert ("fallback", "sqlite_cache") in rows


# ---------------------------------------------------------------------------
# 8. Panel Fetch Entry Points & vintage_panel frequency handling
# ---------------------------------------------------------------------------

def test_fetch_latam_panels(fresh_cache, monkeypatch):
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
        assert len(p_inegi) == 8
        assert sorted(p_inegi.df["date"]) == INEGI_QUARTER_STARTS

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


def test_vintage_panel_serves_undeclared_archives_at_q_only(monkeypatch):
    """An entry that predates ``SeriesSpec.freq`` (every quarterly vintage
    archive, and any bare-string ``catalog=`` override) declares nothing.
    It is quarterly: asking for it at ``freq="M"`` or ``"D"`` must be
    refused with a reason, never served stamped monthly or daily."""
    from puremacro.fetch.realtime import _base as rt_base
    from puremacro.fetch.realtime import catalog as rt_catalog

    seen = []

    def fake_fetch(countries, variables, **kwargs):
        seen.append((list(countries), list(variables)))
        rows = [
            {"country": c, "variable": v, "date": d, "vintage": vin, "value": 1.0,
             "provider": "fakeq", "series_id": "X", "units": "level"}
            for c in countries for v in variables
            for vin in ("2020-04-01", "2020-07-01")
            for d in ("2019-10-01", "2020-01-01")
        ]
        return VintagePanel(pd.DataFrame(rows), metadata={"provider": "fakeq", "failed": {}})

    monkeypatch.setitem(rt_base._PROVIDER_REGISTRY, "fakeq", fake_fetch)
    monkeypatch.setitem(rt_base._PROVIDER_COVERAGE, "fakeq", frozenset({"DEU"}))
    monkeypatch.setitem(rt_catalog._CATALOGS, "fakeq", {"DEU": {"gdp_real": "X"}})
    assert resolve_spec("fakeq", "DEU", "gdp_real").freq == ""

    quarterly = vintage_panel(["DEU"], providers="fakeq")
    assert len(quarterly) == 4 and quarterly.metadata["freq"] == "Q"
    for freq in ("M", "D"):
        with pytest.warns(UserWarning, match="returned nothing for 1 of 1"):
            other = vintage_panel(["DEU"], providers="fakeq", freq=freq)
        assert other.is_empty(), freq
        assert other.metadata["freq"] == freq
        reason = other.metadata["failed"]["DEU:gdp_real:fakeq"]
        assert "declares no freq" in reason and f"requested freq={freq!r}" in reason
    assert seen == [(["DEU"], ["gdp_real"])]      # the M / D requests never fetched

    # A hand-written override with a declared frequency is honoured.
    override = {"fakeq": {"DEU": {"gdp_real": rt_catalog.SeriesSpec("X", freq="M")}}}
    monthly = vintage_panel(["DEU"], providers="fakeq", catalog=override, freq="M")
    assert len(monthly) == 4 and monthly.metadata["failed"] == {}


def test_vintage_panel_refuses_to_stamp_daily_series_quarterly(fresh_cache):
    """Regression: vintage_panel stamped freq='Q' on the daily Selic and
    refused freq='D'."""
    assert SUPPORTED_FREQUENCIES == {"Q", "M", "D"}
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(BCB_FIXTURE_JSON)):
        with pytest.warns(UserWarning, match="returned nothing for 1 of 1"):
            quarterly = vintage_panel(["BRA"], series="policy_rate", providers="bcb")
        daily = vintage_panel(["BRA"], series="policy_rate", providers="bcb", freq="D")
    assert quarterly.is_empty()
    assert quarterly.metadata["missing"] == ["BRA"]
    reason = quarterly.metadata["failed"]["BRA:policy_rate:bcb"]
    assert "432" in reason and "daily" in reason and "requested freq='Q'" in reason
    assert len(daily) == 3
    assert daily.metadata["freq"] == "D"
    assert daily.metadata["failed"] == {}
    with pytest.raises(ValueError, match="freq='A' is not supported"):
        vintage_panel(["BRA"], series="policy_rate", providers="bcb", freq="A")


def test_vintage_panel_monthly_series(fresh_cache, monkeypatch):
    monkeypatch.setenv("INEGI_API_KEY", "mock_key")
    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(INEGI_MONTHLY_FIXTURE_JSON)):
        monthly = vintage_panel(["MEX"], series="cpi", providers="inegi", freq="M")
    assert len(monthly) == 2
    assert monthly.metadata["freq"] == "M"
    assert monthly.units_for("MEX", "cpi") == "index"


# ---------------------------------------------------------------------------
# 9. Portable Cartridge (.pmz) Integration Tests
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


def test_cartridge_keeps_panel_metadata_and_honours_verify(tmp_path):
    """Regression: pack dropped VintagePanel.metadata and load verified even
    with verify=False. The container is a zip of manifest.json + npz read
    with allow_pickle=False — no pickle anywhere."""
    df = pd.DataFrame([{
        "country": "BRA", "variable": "policy_rate", "date": "2026-01-01",
        "vintage": "2026-04-01", "value": 12.25, "provider": "bcb",
        "series_id": "432", "units": "rate"}])
    panel = VintagePanel(df, metadata={
        "provider_used": {"BRA": "bcb"}, "failed": {}, "missing": [],
        "requested_countries": ["BRA", "CHL"], "freq": "D",
        "unadjusted_dropped": pd.DataFrame({"median_range": [1.5]}, index=["SWE"]),
    })
    path = tmp_path / "meta.pmz"
    pack_realtime_cartridge(panel, path, source="test")

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert all(n == "manifest.json" or n.endswith(".npz") for n in names)
        for name in names:
            if name.endswith(".npz"):
                with np.load(io.BytesIO(zf.read(name)), allow_pickle=False):
                    pass
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    assert "pickle" not in json.dumps(manifest).lower()

    loaded = load_realtime_cartridge(path, verify=True)
    assert loaded.metadata["provider_used"] == {"BRA": "bcb"}
    assert loaded.metadata["failed"] == {}
    assert loaded.metadata["missing"] == []
    assert loaded.metadata["requested_countries"] == ["BRA", "CHL"]
    assert loaded.metadata["freq"] == "D"
    assert "SWE" in loaded.metadata["unadjusted_dropped"]       # str-encoded, not lost
    assert loaded.metadata["provenance_source"] == "test"
    assert len(loaded) == 1

    # Tamper with the data frame: verify=True rejects, verify=False loads.
    tampered = tmp_path / "tampered.pmz"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tampered, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "frames/data.npz":
                data = data + b"_mutation_"
            zout.writestr(item, data)
    with pytest.raises(pocket.CartridgeError, match="checksum does not match"):
        load_realtime_cartridge(tampered, verify=True)
    unverified = load_realtime_cartridge(tampered, verify=False)
    assert len(unverified) == 1
    assert unverified.metadata["freq"] == "D"
