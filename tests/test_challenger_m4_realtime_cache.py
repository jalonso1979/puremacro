"""Adversarial Stress Test Suite for Milestone 4: Regional Data Ecosystem & Latin America Connectors.

Authored by orch20_challenger_m4_2 (Empirical Challenger).
Stress-tests:
1. Credentials resolution, precedence order, secret masking, and MissingCredentialError semantics.
2. SQLite realtime_vintages table schema verification, duplicate insertions, batch replacement, and concurrency.
3. Multi-threaded cache writing and concurrent read/write stress.
4. Readonly database and locked database edge cases.
5. Portable .pmz cartridge packing, loading, round-trip fidelity, empty frames, and corruption detection.
6. Pyodide purity verification across all new modules.
"""
from __future__ import annotations

import concurrent.futures
import io
import json
import os
import sqlite3
import tempfile
import time
import urllib
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from puremacro.credentials import (
    SERVICES,
    MissingCredentialError,
    get_credential,
    require_credential,
    status,
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
from puremacro.fetch.realtime.inegi import (
    INEGI_SERIES_URL,
    fetch_inegi_panel,
    fetch_inegi_vintages,
    parse_inegi_json,
)
from puremacro.pocket import CartridgeError


# ===========================================================================
# 1. Credentials Resolution, Precedence, and Error Semantics
# ===========================================================================

class TestCredentialsSecurityAndPrecedence:
    """Adversarial challenge of API key resolution and error ergonomics."""

    def test_precedence_explicit_over_env_and_config(self, tmp_path, monkeypatch):
        """Explicit argument must override environment variables and config file."""
        cfg_file = tmp_path / "credentials.toml"
        cfg_file.write_text('[banxico]\napi_key = "CFG_KEY"\n', encoding="utf-8")
        monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(cfg_file))
        monkeypatch.setenv("BANXICO_API_KEY", "ENV_KEY")
        # Clear config cache
        monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)

        res = credentials.get("banxico", explicit="EXP_KEY")
        assert res == "EXP_KEY"

    def test_precedence_env_vars_order(self, monkeypatch):
        """First listed env_var in spec must take precedence over subsequent ones."""
        # For banxico: ("BANXICO_API_KEY", "BMX_TOKEN", "PUREMACRO_BANXICO_API_KEY")
        monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)
        monkeypatch.setenv("BMX_TOKEN", "SECOND_TOKEN")
        monkeypatch.setenv("PUREMACRO_BANXICO_API_KEY", "THIRD_TOKEN")
        assert credentials.get("banxico") == "SECOND_TOKEN"

        monkeypatch.setenv("BANXICO_API_KEY", "FIRST_TOKEN")
        assert credentials.get("banxico") == "FIRST_TOKEN"

    def test_precedence_empty_env_var_falls_through(self, monkeypatch):
        """Empty environment variable must fall through to next env var."""
        monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)
        monkeypatch.setenv("BANXICO_API_KEY", "")
        monkeypatch.setenv("BMX_TOKEN", "FALLTHROUGH_TOKEN")
        assert credentials.get("banxico") == "FALLTHROUGH_TOKEN"

    def test_missing_credential_raises_actionable_error(self, monkeypatch, tmp_path):
        """require() on missing credential must raise MissingCredentialError with actionable details."""
        empty_cfg = tmp_path / "empty.toml"
        empty_cfg.write_text("", encoding="utf-8")
        monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(empty_cfg))
        monkeypatch.delenv("BANXICO_API_KEY", raising=False)
        monkeypatch.delenv("BMX_TOKEN", raising=False)
        monkeypatch.delenv("PUREMACRO_BANXICO_API_KEY", raising=False)
        monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)

        with pytest.raises(MissingCredentialError) as exc_info:
            credentials.require("banxico")

        err_msg = str(exc_info.value)
        assert "BANXICO_API_KEY" in err_msg
        assert "BMX_TOKEN" in err_msg
        assert "https://www.banxico.org.mx" in err_msg

    def test_credentials_status_never_leaks_secrets(self, monkeypatch, tmp_path):
        """status() table must never reveal actual secret keys."""
        secret = "SUPER_SECRET_TOKEN_XYZ_12345"
        monkeypatch.setenv("INEGI_API_KEY", secret)
        monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)

        st = credentials.status()
        assert isinstance(st, pd.DataFrame)
        assert set(st.columns) == {"service", "configured", "source", "description", "signup_url"}
        
        # Check that secret does not appear anywhere in dataframe values
        for col in st.columns:
            for val in st[col].astype(str):
                assert secret not in val, f"Secret leaked in column {col}: {val}"

        inegi_row = st[st["service"] == "inegi"].iloc[0]
        assert inegi_row["configured"] is True or inegi_row["configured"] == True
        assert inegi_row["source"] == "env:INEGI_API_KEY"

    def test_bcch_password_alone_is_not_a_credential(self, monkeypatch, tmp_path):
        """Regression for the BCCh password-only anomaly.

        BCCH_API_PASS used to sit in credentials.SERVICES['bcch'].env_vars, so
        with only the password set credentials.get('bcch') returned it and the
        connector sent it as the *user*. The two halves are now resolved
        separately: get() is the user, get_password() the password, and
        require() insists on both.
        """
        monkeypatch.setenv("PUREMACRO_CREDENTIALS_FILE", str(tmp_path / "absent.toml"))
        for var in ("BCCH_API_USER", "PUREMACRO_BCCH_API_USER",
                    "BCCH_API_KEY", "PUREMACRO_BCCH_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("BCCH_API_PASS", "my_secret_pass")
        monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)

        assert "BCCH_API_PASS" not in SERVICES["bcch"].env_vars
        assert credentials.get("bcch") is None
        assert credentials.get_password("bcch") == "my_secret_pass"
        st = status().set_index("service")
        assert bool(st.loc["bcch", "configured"]) is False
        assert "my_secret_pass" not in st.to_string()
        with pytest.raises(MissingCredentialError) as exc_info:
            require_credential("bcch")
        assert "BCCH_API_USER" in str(exc_info.value)
        assert "BCCH_API_PASS" in str(exc_info.value)

    def test_inegi_missing_credential_in_fetch(self, monkeypatch, tmp_path):
        """fetch_inegi_vintages without token or cache must raise MissingCredentialError."""
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
        close_conn()
        monkeypatch.delenv("INEGI_API_KEY", raising=False)
        monkeypatch.delenv("PUREMACRO_INEGI_API_KEY", raising=False)
        monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)

        with pytest.raises(MissingCredentialError):
            fetch_inegi_vintages("735848", token=None, use_cache=True)
        close_conn()



# ===========================================================================
# 2. SQLite realtime_vintages Table & Concurrency Stress
# ===========================================================================

class TestSQLiteCacheStress:
    """Stress tests for SQLite cache concurrency, schema, and duplicate insertions."""

    def test_schema_columns_audit(self, tmp_path):
        """Audit actual SQLite table schema against claimed schema.
        
        Note: The worker handoff claimed realtime_vintages has columns:
        (id, provider, country, series_id, canonical_variable, observation_date, vintage_date, value, unit, frequency, source_metadata, created_at)
        Let's verify the actual columns created.
        """
        db_path = tmp_path / "test_schema.db"
        close_conn()
        conn = get_conn(db_path)
        cur = conn.execute("PRAGMA table_info(realtime_vintages)")
        columns = {row[1]: row[2] for row in cur.fetchall()}
        close_conn()

        expected_actual = {
            "provider": "TEXT",
            "country": "TEXT",
            "series_id": "TEXT",
            "observation_date": "TEXT",
            "vintage_date": "TEXT",
            "value": "REAL",
        }
        assert set(columns.keys()) == set(expected_actual.keys())

    def test_duplicate_primary_key_insert_or_replace(self, tmp_path):
        """Duplicate rows on (provider, country, series_id, observation_date, vintage_date)
        must replace the existing value and not duplicate or raise."""
        db_path = tmp_path / "test_dup.db"
        close_conn()
        conn = get_conn(db_path)

        df1 = pd.DataFrame([
            {
                "provider": "banxico",
                "country": "MEX",
                "series_id": "SF61745",
                "date": "2026-01-15",
                "vintage": "2026-02-01",
                "value": 10.75,
            }
        ])
        count1 = store_realtime_vintages(df1, conn=conn)
        assert count1 == 1

        # Query back
        res1 = query_realtime_vintages("banxico", "MEX", "SF61745", conn=conn)
        assert len(res1) == 1
        assert res1.iloc[0]["value"] == 10.75

        # Re-insert with updated value (11.25)
        df2 = pd.DataFrame([
            {
                "provider": "banxico",
                "country": "MEX",
                "series_id": "SF61745",
                "date": "2026-01-15",
                "vintage": "2026-02-01",
                "value": 11.25,
            }
        ])
        count2 = store_realtime_vintages(df2, conn=conn)
        assert count2 == 1

        # Must still be exactly 1 row, with value 11.25
        res2 = query_realtime_vintages("banxico", "MEX", "SF61745", conn=conn)
        assert len(res2) == 1
        assert res2.iloc[0]["value"] == 11.25
        close_conn()

    def test_intra_batch_duplicates(self, tmp_path):
        """A batch containing internal duplicate primary keys must resolve to the last row."""
        db_path = tmp_path / "test_batch_dup.db"
        close_conn()
        conn = get_conn(db_path)

        df = pd.DataFrame([
            {"provider": "bcb", "country": "BRA", "series_id": "432", "date": "2026-01-01", "vintage": "2026-01-02", "value": 13.75},
            {"provider": "bcb", "country": "BRA", "series_id": "432", "date": "2026-01-01", "vintage": "2026-01-02", "value": 14.00},
            {"provider": "bcb", "country": "BRA", "series_id": "432", "date": "2026-01-01", "vintage": "2026-01-02", "value": 14.25},
        ])
        stored = store_realtime_vintages(df, conn=conn)
        assert stored == 3

        res = query_realtime_vintages("bcb", "BRA", "432", conn=conn)
        assert len(res) == 1
        assert res.iloc[0]["value"] == 14.25
        close_conn()

    def test_missing_and_invalid_values_handling(self, tmp_path):
        """Test NaN, None, and unparseable string values."""
        db_path = tmp_path / "test_nan.db"
        close_conn()
        conn = get_conn(db_path)

        df = pd.DataFrame([
            {"provider": "inegi", "country": "MEX", "series_id": "gdp", "date": "2026-01-01", "vintage": "2026-02-01", "value": np.nan},
            {"provider": "inegi", "country": "MEX", "series_id": "gdp", "date": "2026-04-01", "vintage": "2026-05-01", "value": None},
            {"provider": "inegi", "country": "MEX", "series_id": "gdp", "date": "2026-07-01", "vintage": "2026-08-01", "value": "unparseable"},
            {"provider": "inegi", "country": "MEX", "series_id": "gdp", "date": "2026-10-01", "vintage": "2026-11-01", "value": "105.4"},
        ])
        store_realtime_vintages(df, conn=conn)
        res = query_realtime_vintages("inegi", "MEX", "gdp", conn=conn)
        
        # 'unparseable' was skipped, valid and NaNs stored
        assert len(res) == 3
        # First two values should be NaN
        assert pd.isna(res.iloc[0]["value"])
        assert pd.isna(res.iloc[1]["value"])
        assert res.iloc[2]["value"] == 105.4
        close_conn()

    def test_query_vintage_date_filtering(self, tmp_path):
        """query_realtime_vintages with vintage_date cutoff must return only vintages <= cutoff."""
        db_path = tmp_path / "test_vin_filter.db"
        close_conn()
        conn = get_conn(db_path)

        df = pd.DataFrame([
            {"provider": "bcch", "country": "CHL", "series_id": "tpm", "date": "2026-01-01", "vintage": "2026-01-15", "value": 5.0},
            {"provider": "bcch", "country": "CHL", "series_id": "tpm", "date": "2026-01-01", "vintage": "2026-02-15", "value": 5.25},
            {"provider": "bcch", "country": "CHL", "series_id": "tpm", "date": "2026-01-01", "vintage": "2026-03-15", "value": 5.50},
        ])
        store_realtime_vintages(df, conn=conn)

        # Cutoff at 2026-02-01: should only return first vintage
        res1 = query_realtime_vintages("bcch", "CHL", "tpm", vintage_date="2026-02-01", conn=conn)
        assert len(res1) == 1
        assert res1.iloc[0]["value"] == 5.0

        # Cutoff at 2026-02-20: should return first two
        res2 = query_realtime_vintages("bcch", "CHL", "tpm", vintage_date="2026-02-20", conn=conn)
        assert len(res2) == 2

        # No cutoff: returns all three
        res3 = query_realtime_vintages("bcch", "CHL", "tpm", conn=conn)
        assert len(res3) == 3
        close_conn()

    def test_concurrent_multithreaded_writes_and_reads(self, tmp_path):
        """Stress-test concurrent thread writes and reads against the SQLite database."""
        db_path = tmp_path / "test_concurrent.db"
        close_conn()
        # Initialize DB in WAL mode
        init_conn = get_conn(db_path)
        close_conn()

        num_threads = 8
        rows_per_thread = 50
        dates = [f"2026-{(i//28)+1:02d}-{(i%28)+1:02d}" for i in range(rows_per_thread)]

        def worker_write(thread_id: int):
            conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
            records = []
            for i in range(rows_per_thread):
                records.append({
                    "provider": "banxico",
                    "country": "MEX",
                    "series_id": f"series_{thread_id}",
                    "date": dates[i],
                    "vintage": f"2026-03-{thread_id + 1:02d}",
                    "value": float(thread_id * 100 + i),
                })
            df = pd.DataFrame(records)
            store_realtime_vintages(df, conn=conn)
            conn.close()
            return thread_id

        def worker_read(thread_id: int):
            conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
            df = query_realtime_vintages("banxico", "MEX", f"series_{thread_id}", conn=conn)
            conn.close()
            return len(df)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            write_futures = [executor.submit(worker_write, t) for t in range(num_threads)]
            for f in concurrent.futures.as_completed(write_futures):
                assert f.result() is not None

        # Verify all rows written
        verify_conn = sqlite3.connect(db_path, timeout=30.0)
        cur = verify_conn.execute("SELECT count(*) FROM realtime_vintages")
        total_rows = cur.fetchone()[0]
        verify_conn.close()
        assert total_rows == num_threads * rows_per_thread

        # Concurrently read while writing new data
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            read_futures = [executor.submit(worker_read, t) for t in range(num_threads)]
            for f in concurrent.futures.as_completed(read_futures):
                assert f.result() == rows_per_thread

    def test_database_locked_handling(self, tmp_path):
        """Test behavior when an exclusive transaction lock is held by another connection."""
        db_path = tmp_path / "test_locked.db"
        close_conn()
        conn1 = get_conn(db_path)

        # Open connection 2 and begin an exclusive transaction
        conn2 = sqlite3.connect(db_path, timeout=0.1)
        conn2.execute("BEGIN EXCLUSIVE")

        # Now attempting to write via conn1 with small timeout should raise OperationalError
        conn1_short = sqlite3.connect(db_path, timeout=0.1)
        df = pd.DataFrame([{
            "provider": "bcb",
            "country": "BRA",
            "series_id": "test",
            "date": "2026-01-01",
            "vintage": "2026-01-01",
            "value": 1.0,
        }])
        with pytest.raises(sqlite3.OperationalError):
            store_realtime_vintages(df, conn=conn1_short)

        conn2.rollback()
        conn2.close()
        conn1_short.close()
        close_conn()


# ===========================================================================
# 3. Pocket .pmz Cartridge Serialization & Integrity Stress
# ===========================================================================

class TestPocketCartridgeStress:
    """Stress tests for .pmz cartridge packing, loading, and integrity validation."""

    def test_round_trip_panel(self, tmp_path):
        """Verify round-trip fidelity between VintagePanel and .pmz cartridge."""
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        vintages = pd.date_range("2021-01-01", periods=12, freq="MS")
        df = pd.DataFrame({
            "country": ["MEX"] * 12,
            "variable": ["policy_rate"] * 12,
            "provider": ["banxico"] * 12,
            "series_id": ["SF61745"] * 12,
            "date": dates,
            "vintage": vintages,
            "value": np.linspace(5.0, 11.25, 12),
            "units": ["percent"] * 12,
        })
        panel = VintagePanel(df=df, metadata={"key": "test_panel"})

        cart_path = tmp_path / "test_mex.pmz"
        pack_realtime_cartridge(
            panel,
            cart_path,
            source="Banxico SIE API",
            notes="Test cartridge for Banxico policy rate",
        )
        assert cart_path.exists()
        assert cart_path.stat().st_size > 0

        # Load back
        loaded = load_realtime_cartridge(cart_path, verify=True)
        assert isinstance(loaded, VintagePanel)
        assert len(loaded.df) == 12
        assert loaded.metadata["provenance_source"] == "Banxico SIE API"
        assert loaded.metadata["provenance_notes"] == "Test cartridge for Banxico policy rate"
        assert np.allclose(loaded.df["value"].values, df["value"].values)

    def test_round_trip_with_nans(self, tmp_path):
        """Cartridge must preserve NaN values and datetimes accurately."""
        df = pd.DataFrame({
            "country": ["BRA", "BRA"],
            "variable": ["gdp", "gdp"],
            "provider": ["bcb", "bcb"],
            "series_id": ["4380", "4380"],
            "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")],
            "vintage": [pd.Timestamp("2026-05-01"), pd.Timestamp("2026-08-01")],
            "value": [np.nan, 2.5],
            "units": ["index", "index"],
        })
        cart_path = tmp_path / "test_nan.pmz"
        pack_realtime_cartridge(df, cart_path)

        # Raw pocket frame preserves NaN values without data loss
        raw_cart = pocket.load(cart_path)
        raw_df = raw_cart.frame()
        assert len(raw_df) == 2
        assert pd.isna(raw_df.iloc[0]["value"])
        assert raw_df.iloc[1]["value"] == 2.5

        # VintagePanel drops missing observations by design per revision semantics
        loaded = load_realtime_cartridge(cart_path, verify=True)
        assert len(loaded.df) == 1
        assert loaded.df.iloc[0]["value"] == 2.5


    def test_corrupted_cartridge_verification_failure(self, tmp_path):
        """A tampered or corrupt .pmz cartridge must fail SHA-256 verification."""
        import zipfile
        df = pd.DataFrame({
            "country": ["CHL"],
            "variable": ["cpi"],
            "provider": ["bcch"],
            "series_id": ["F073"],
            "date": [pd.Timestamp("2026-01-01")],
            "vintage": [pd.Timestamp("2026-02-01")],
            "value": [3.2],
            "units": ["percent"],
        })
        cart_path = tmp_path / "valid.pmz"
        pack_realtime_cartridge(df, cart_path)

        # Tamper with the zip archive (corrupt internal payload)
        raw_bytes = bytearray(cart_path.read_bytes())
        # Flip some bytes in the middle
        mid = len(raw_bytes) // 2
        raw_bytes[mid:mid+10] = b"CORRUPTED!"
        corrupt_path = tmp_path / "corrupt.pmz"
        corrupt_path.write_bytes(raw_bytes)

        # Loading corrupt cartridge with verify=True must raise
        with pytest.raises(Exception):
            load_realtime_cartridge(corrupt_path, verify=True)

    def test_empty_panel_cartridge(self, tmp_path):
        """Packing and loading an empty DataFrame cartridge must not fail."""
        empty_df = pd.DataFrame(columns=["date", "vintage", "value", "country", "variable", "provider", "series_id"])
        cart_path = tmp_path / "empty.pmz"
        pack_realtime_cartridge(empty_df, cart_path)
        assert cart_path.exists()

        loaded = load_realtime_cartridge(cart_path, verify=True)
        assert len(loaded.df) == 0


# ===========================================================================
# 4. Pyodide Purity & Import Inspection
# ===========================================================================

class TestPyodidePurityAndZeroForbiddenImports:
    """Ensure zero unauthorized runtime dependencies in new modules."""

    FORBIDDEN = {
        "requests",
        "urllib3",
        "httpx",
        "aiohttp",
        "statsmodels",
        "linearmodels",
        "arch",
        "numba",
        "torch",
        "jax",
        "tensorflow",
        "polars",
        "pyarrow",
        "duckdb",
        "pydantic",
    }

    def test_new_modules_import_ast_sweep(self):
        """Inspect AST imports of all Milestone 4 modules."""
        import ast

        repo_root = Path(__file__).resolve().parent.parent
        target_files = [
            repo_root / "puremacro" / "credentials.py",
            repo_root / "puremacro" / "_cache_db.py",
            repo_root / "puremacro" / "fetch" / "realtime" / "cartridge.py",
            repo_root / "puremacro" / "fetch" / "realtime" / "canary.py",
            repo_root / "puremacro" / "fetch" / "realtime" / "banxico.py",
            repo_root / "puremacro" / "fetch" / "realtime" / "inegi.py",
            repo_root / "puremacro" / "fetch" / "realtime" / "bcb.py",
            repo_root / "puremacro" / "fetch" / "realtime" / "bcch.py",
        ]

        violations = []
        for file_path in target_files:
            assert file_path.exists(), f"File {file_path} missing!"
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in self.FORBIDDEN:
                            violations.append((file_path.name, top, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if top in self.FORBIDDEN:
                            violations.append((file_path.name, top, node.lineno))

        assert not violations, f"Forbidden runtime imports found: {violations}"


# ===========================================================================
# 5. Empirical Bug Demonstrations & Architectural Findings
# ===========================================================================

class TestEmpiricalDefectsAndDiscrepancies:
    """Explicit empirical tests reproducing identified defects, edge cases, and discrepancies."""

    def test_store_realtime_vintages_nat_date_crash(self, tmp_path):
        """Verify store_realtime_vintages gracefully skips rows with NaT/None or invalid dates."""
        db_path = tmp_path / "test_nat_crash.db"
        close_conn()
        conn = get_conn(db_path)

        df_mixed = pd.DataFrame([
            {
                "provider": "banxico",
                "country": "MEX",
                "series_id": "SF61745",
                "date": pd.NaT,
                "vintage": "2026-01-01",
                "value": 10.0,
            },
            {
                "provider": "banxico",
                "country": "MEX",
                "series_id": "SF61745",
                "date": "not-a-valid-date",
                "vintage": "2026-01-01",
                "value": 10.0,
            },
            {
                "provider": "banxico",
                "country": "MEX",
                "series_id": "SF61745",
                "date": "2026-01-01",
                "vintage": "2026-01-01",
                "value": 10.75,
            },
        ])
        stored = store_realtime_vintages(df_mixed, conn=conn)
        assert stored == 1
        res = query_realtime_vintages("banxico", "MEX", "SF61745", conn=conn)
        assert len(res) == 1
        assert res.iloc[0]["value"] == 10.75
        close_conn()

    def test_readonly_database_can_still_be_queried(self, tmp_path):
        """Regression: get_conn() used to fail on a read-only cache file.

        It ran PRAGMA journal_mode=WAL and the schema DDL even when the caller
        only wanted to read, so a cache on a read-only volume was unusable.
        The file is now opened read-only: reads work, writes still raise.
        """
        db_path = tmp_path / "readonly_cache.db"
        close_conn()
        conn = get_conn(db_path)
        store_realtime_vintages(pd.DataFrame([{
            "provider": "banxico",
            "country": "MEX",
            "series_id": "SF61745",
            "date": "2026-01-01",
            "vintage": "2026-01-01",
            "value": 10.0,
        }]), conn=conn)
        close_conn()

        # Set read-only file permissions
        os.chmod(db_path, 0o444)
        if os.access(db_path, os.W_OK):
            os.chmod(db_path, 0o666)
            pytest.skip("file permissions do not restrict this account (root?)")
        try:
            conn_ro = get_conn(db_path)
            res = query_realtime_vintages("banxico", "MEX", "SF61745", conn=conn_ro)
            assert len(res) == 1
            assert res.iloc[0]["value"] == 10.0
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                store_realtime_vintages(pd.DataFrame([{
                    "provider": "banxico", "country": "MEX", "series_id": "SF61745",
                    "date": "2026-02-01", "vintage": "2026-02-01", "value": 11.0,
                }]), conn=conn_ro)
            # Telemetry degrades to a warning rather than an exception.
            with pytest.warns(UserWarning, match="record_connector_event failed"):
                record_connector_event("banxico", "success", "none", conn=conn_ro)
        finally:
            close_conn()
            os.chmod(db_path, 0o666)

    def test_catalog_claimed_policy_rate_aliases_missing(self):
        """Verify all policy_rate aliases (including central_bank_rate, target_rate, overnight_rate) resolve."""
        from puremacro.fetch.realtime.catalog import canonical_variable

        aliases = [
            "central_bank_rate", "target_rate", "overnight_rate",
            "policy_rate", "tpm", "selic", "tasa_objetivo", "interest_rate", "mp_rate",
        ]
        for alias in aliases:
            assert canonical_variable(alias) == "policy_rate"

    def test_bcch_fetch_scrubs_password_from_exception_url(self, tmp_path, monkeypatch):
        """Regression: the BCCh password travels in the query string (the API
        requires it) and urllib copies the URL onto HTTPError/URLError. The
        connector now redacts it from the exception it re-raises and from the
        warning it emits when it falls back to the cache."""
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
        close_conn()
        monkeypatch.setenv("BCCH_API_USER", "analyst@bank.cl")
        secret_pass = "SENSITIVE_CHILE_PASSWORD_123"
        monkeypatch.setenv("BCCH_API_PASS", secret_pass)
        monkeypatch.setattr(credentials, "_CONFIG_CACHE", None)
        assert "{password}" in BCCH_SIETE_URL and "pass=" in BCCH_SIETE_URL

        seen = {}

        def unauthorized(req, timeout=None):
            seen["url"] = req.full_url
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)

        try:
            with patch("urllib.request.urlopen", side_effect=unauthorized):
                with pytest.raises(urllib.error.HTTPError) as exc_info:
                    fetch_bcch_vintages("F032", use_cache=False)
            assert secret_pass in seen["url"]                 # sent to the API...
            err = exc_info.value
            assert secret_pass not in err.url                  # ...but not re-raised
            assert secret_pass not in (err.filename or "")
            assert "pass=***" in err.url
        finally:
            close_conn()

