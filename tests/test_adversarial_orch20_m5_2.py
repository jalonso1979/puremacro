"""Adversarial stress test suite for Milestone 5 deliverables (Challenger 2).

Systematically tests:
1. SQLite Cache Concurrency, Transactional Isolation & SQL Injection Resilience
2. Portable .pmz Pocket Cartridge Tampering, Corrupted Archives & Boundary Conditions
3. LatAm Real-Time Connectors & Schema Drift Canary Attack Vectors
4. Hardware Acceleration Backend Fallback Consistency & Tensor Contractions
5. Parallel Bootstrap WASM Single-Threaded Fallbacks & Defensive Error Handling
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import io
import json
import os
import sqlite3
import tempfile
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from puremacro import credentials, pocket
from puremacro._backend import (
    backend_available,
    get_array_namespace,
    to_numpy,
)
from puremacro._cache_db import (
    bootstrap_schema,
    close_conn,
    default_db_path,
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
    fetch_banxico_vintages,
    parse_banxico_json,
)
from puremacro.fetch.realtime.bcb import (
    fetch_bcb_vintages,
    parse_bcb_json,
)
from puremacro.fetch.realtime.bcch import (
    fetch_bcch_vintages,
    parse_bcch_json,
)
from puremacro.fetch.realtime.inegi import (
    fetch_inegi_vintages,
    parse_inegi_json,
)
from puremacro.inference.block_bootstrap import block_bootstrap
from puremacro.inference.lp_block_bootstrap import cum_irf_block_bootstrap
from puremacro.inference.wild_bootstrap import wild_bootstrap, wild_bootstrap_var
from puremacro.pocket import CartridgeError
from puremacro.runtime import capabilities, refresh
from puremacro.spatial.allen_arkolakis import AllenArkolakisModel
from puremacro.trade.calibration import calibrate_trade_model
from puremacro.trade.solver import solve_trade_equilibrium
from puremacro.var.bootstrap import bootstrap_bands


# ===========================================================================
# 1. SQLite Cache Concurrency, Transactions & SQL Injection
# ===========================================================================

class TestSQLiteCacheAdversarial:
    """Stress testing SQLite cache for concurrency, isolation, and injection."""

    def test_high_concurrency_multithreaded_writes(self, tmp_path):
        """Simulate 16 threads writing 100 rows each to SQLite simultaneously."""
        db_file = tmp_path / "concurrent_stress.db"
        conn_init = sqlite3.connect(db_file, isolation_level=None)
        conn_init.execute("PRAGMA journal_mode=WAL")
        bootstrap_schema(conn_init)
        conn_init.close()

        num_threads = 16
        rows_per_thread = 100

        def worker_write(thread_id: int):
            conn = sqlite3.connect(db_file, timeout=60.0, isolation_level=None)
            dates = pd.date_range("2020-01-01", periods=rows_per_thread, freq="MS")
            df = pd.DataFrame({
                "provider": [f"prov_{thread_id % 4}"] * rows_per_thread,
                "country": ["MEX"] * rows_per_thread,
                "series_id": [f"SERIES_{thread_id}"] * rows_per_thread,
                "date": dates,
                "vintage": ["2026-09-01"] * rows_per_thread,
                "value": np.arange(rows_per_thread, dtype=float) + thread_id,
            })
            n = store_realtime_vintages(df, conn=conn)
            conn.close()
            return n

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker_write, t) for t in range(num_threads)]
            results = [f.result() for f in futures]

        assert sum(results) == num_threads * rows_per_thread

        conn_check = sqlite3.connect(db_file)
        cur = conn_check.cursor()
        cur.execute("SELECT count(*) FROM realtime_vintages")
        total_count = cur.fetchone()[0]
        assert total_count == num_threads * rows_per_thread
        conn_check.close()

    def test_sql_injection_in_all_dimensions(self, tmp_path):
        """Verify comprehensive SQL injection resistance across all query fields."""
        conn = sqlite3.connect(":memory:")
        bootstrap_schema(conn)

        injections = [
            "'; DROP TABLE realtime_vintages; --",
            "' UNION SELECT '1', '2', '3', '4', '5', '6' --",
            "\" OR \"\"=\"",
            "MEX'; UPDATE realtime_vintages SET value=999999 WHERE '1'='1",
        ]

        for i, inj in enumerate(injections):
            df = pd.DataFrame({
                "provider": [f"prov_{i}{inj}"],
                "country": [f"C_{i}{inj}"],
                "series_id": [f"S_{i}{inj}"],
                "date": ["2026-01-01"],
                "vintage": ["2026-06-01"],
                "value": [float(i + 1)],
            })
            store_realtime_vintages(df, conn=conn)

        # Table must remain healthy
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM realtime_vintages")
        assert cur.fetchone()[0] == len(injections)

        # Parameterized query with injection payload in provider, country, series_id
        res = query_realtime_vintages(
            f"prov_0{injections[0]}",
            f"C_0{injections[0]}",
            f"S_0{injections[0]}",
            vintage_date="2026-12-31",
            conn=conn,
        )
        assert len(res) == 1
        assert res["value"].iloc[0] == 1.0

        # Attempting unparseable date injection in vintage_date raises DateParseError
        with pytest.raises(Exception):
            query_realtime_vintages(
                f"prov_0{injections[0]}",
                f"C_0{injections[0]}",
                f"S_0{injections[0]}",
                vintage_date="2026-12-31' OR '1'='1",
                conn=conn,
            )

    def test_extreme_and_pathological_values_in_cache(self, tmp_path):
        """Verify handling of extreme floats, NaNs, NaTs, infs, and empty records."""
        conn = sqlite3.connect(":memory:")
        bootstrap_schema(conn)

        df = pd.DataFrame({
            "provider": ["banxico", "banxico", "banxico", "banxico", "banxico"],
            "country": ["MEX", "MEX", "MEX", "MEX", "MEX"],
            "series_id": ["TEST_EXTREME", "TEST_EXTREME", "TEST_EXTREME", "TEST_EXTREME", "TEST_EXTREME"],
            "date": ["2026-01-01", "2026-02-01", "2026-03-01", "not-a-date", "2026-05-01"],
            "vintage": ["2026-09-01", "2026-09-01", "not-a-vintage", "2026-09-01", "2026-09-01"],
            "value": [1e308, -1e308, 42.0, 10.0, np.nan],
        })

        n = store_realtime_vintages(df, conn=conn)
        assert n == 3  # row 0 (1e308), row 1 (-1e308), row 4 (np.nan -> None)

        res = query_realtime_vintages("banxico", "MEX", "TEST_EXTREME", conn=conn)
        assert len(res) == 3
        assert 1e308 in res["value"].values
        assert -1e308 in res["value"].values
        assert res["value"].isna().sum() == 1


# ===========================================================================
# 2. Portable .pmz Cartridge Tampering & Roundtrip Integrity
# ===========================================================================

class TestCartridgeAdversarial:
    """Stress testing pocket cartridge packaging, metadata, and tampering."""

    def test_corrupted_zip_payload(self, tmp_path):
        """Cartridge with truncated or random binary payload must fail verification."""
        bad_pmz = tmp_path / "bad.pmz"
        bad_pmz.write_bytes(b"PK\x03\x04not_a_valid_zip_archive_data_payload_12345")
        with pytest.raises((CartridgeError, zipfile.BadZipFile)):
            load_realtime_cartridge(bad_pmz, verify=True)

    def test_tampered_manifest_version_or_format(self, tmp_path):
        """Cartridge with altered manifest format tag must raise CartridgeError."""
        df = pd.DataFrame({
            "country": ["BRA"],
            "variable": ["policy_rate"],
            "date": ["2026-01-01"],
            "vintage": ["2026-06-01"],
            "value": [10.5],
            "provider": ["bcb"],
            "series_id": ["432"],
            "units": ["rate"],
        })
        panel = VintagePanel(df)
        pmz_path = tmp_path / "valid.pmz"
        pack_realtime_cartridge(panel, pmz_path)

        with zipfile.ZipFile(pmz_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            other_files = {name: zf.read(name) for name in zf.namelist() if name != "manifest.json"}

        manifest["format"] = "tampered_format_v99"
        tampered_pmz = tmp_path / "tampered.pmz"
        with zipfile.ZipFile(tampered_pmz, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            for name, content in other_files.items():
                zf.writestr(name, content)

        with pytest.raises(CartridgeError, match="not a puremacro cartridge.*wrong format tag"):
            load_realtime_cartridge(tampered_pmz, verify=True)

    def test_empty_panel_roundtrip(self, tmp_path):
        """Empty VintagePanel must pack and load without crashing."""
        empty_df = pd.DataFrame(columns=[
            "date", "vintage", "value", "provider", "country", "series_id", "variable", "units"
        ])
        panel = VintagePanel(empty_df)
        pmz_path = tmp_path / "empty.pmz"
        pack_realtime_cartridge(panel, pmz_path, notes="empty panel test")

        loaded = load_realtime_cartridge(pmz_path, verify=True)
        assert loaded.df.empty
        assert loaded.metadata.get("provenance_notes") == "empty panel test"


# ===========================================================================
# 3. LatAm Connectors & Schema Drift Canaries
# ===========================================================================

class TestLatAmCanaryAdversarial:
    """Stress testing SchemaCanary against hostile and mutated inputs."""

    @pytest.fixture(autouse=True)
    def _isolated_cache_db(self, tmp_path, monkeypatch):
        """SchemaCanary.check records drift events through record_connector_event, which
        opens the default cache DB; point it at tmp_path so no test writes to the
        developer's ~/.cache/puremacro/cache.db."""
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path / "canary_cache.db"))
        close_conn()
        yield
        close_conn()

    def test_canary_hostile_html_response(self):
        """Canary must reject HTML error pages (e.g. Cloudflare or gateway 502)."""
        html_payload = """
        <!DOCTYPE html>
        <html>
        <head><title>502 Bad Gateway</title></head>
        <body><center><h1>502 Bad Gateway</h1></center><hr><center>cloudflare</center></body>
        </html>
        """
        for prov in ("banxico", "inegi", "bcb", "bcch"):
            ok, reason = validate_payload(prov, html_payload)
            assert ok is False
            assert "invalid JSON" in reason

    def test_canary_banxico_mutated_dates_deep_in_payload(self):
        """Verify canary detection of malformed date format in Banxico."""
        payload = {
            "bmx": {
                "series": [
                    {
                        "idSerie": "SF61745",
                        "datos": [
                            {"fecha": "2026/01/01", "dato": "10.5"},
                        ],
                    }
                ]
            }
        }
        ok, reason = SchemaCanary.validate_banxico(payload)
        assert ok is False
        assert "date format '2026/01/01' does not match DD/MM/YYYY" in reason

    def test_canary_inegi_unsupported_period_format(self):
        """INEGI with invalid period format string must fail validation."""
        payload = {
            "Series": [
                {
                    "INDICADOR": "735848",
                    "OBSERVATIONS": [
                        {"TIME_PERIOD": "2026-Q5-invalid", "OBS_VALUE": "100.0"},
                    ],
                }
            ]
        }
        ok, reason = SchemaCanary.validate_inegi(payload)
        assert ok is False
        assert "invalid TIME_PERIOD" in reason

    def test_canary_bcch_error_code_rejection(self):
        """BCCh returning non-zero error code must be identified immediately."""
        payload = {
            "Codigo": 401,
            "Descripcion": "No autorizado para consultar la serie",
            "Series": {"obs": []},
        }
        ok, reason = SchemaCanary.validate_bcch(payload)
        assert ok is False
        assert "error code 401" in reason

    def test_end_to_end_network_failure_falls_back_to_sqlite(self, tmp_path, monkeypatch):
        """When network raises URLError, fetch_banxico_vintages must transparently use SQLite cache."""
        db_path = tmp_path / "fallback_test.db"
        monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_path))
        close_conn()

        conn = get_conn()
        seed_df = pd.DataFrame({
            "provider": ["banxico"],
            "country": ["MEX"],
            "series_id": ["SF61745"],
            "date": ["2026-01-01"],
            "vintage": ["2026-05-01"],
            "value": [11.25],
        })
        store_realtime_vintages(seed_df, conn=conn)

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network unreachable")):
            with pytest.warns(UserWarning, match="falling back to cached vintages"):
                df = fetch_banxico_vintages("SF61745", token="DUMMY_TOKEN", use_cache=True)

        assert len(df) == 1
        assert df["value"].iloc[0] == 11.25
        close_conn()


# ===========================================================================
# 4. Hardware Backend & Array Namespace Fallbacks
# ===========================================================================

class TestBackendAccelerationAdversarial:
    """Stress testing hardware backend abstraction and array namespaces."""

    @pytest.fixture
    def canonical_trade_calib(self):
        nc, ns, nfd = 2, 2, 3
        data = np.zeros((7, 10), dtype=float)
        data[:4, :4] = np.array([
            [20.0, 15.0, 5.0, 2.0],
            [10.0, 25.0, 2.0, 8.0],
            [5.0, 5.0, 12.0, 18.0],
            [10.0, 10.0, 18.0, 22.0],
        ])
        inter_col_sums = data[:4, :4].sum(axis=0)
        y = np.array([100.0, 150.0, 120.0, 180.0])
        va = y - inter_col_sums
        taxes = 0.05 * y
        va_fac = va - taxes
        labor = (2.0 / 3.0) * va_fac
        capital = (1.0 / 3.0) * va_fac

        data[4, :4] = taxes
        data[5, :4] = labor
        data[6, :4] = capital

        inter_row_sums = data[:4, :4].sum(axis=1)
        fd_row_sums = y - inter_row_sums

        for i in range(4):
            tot_fd = fd_row_sums[i]
            if i < 2:
                data[i, 4] = tot_fd * 0.50
                data[i, 5] = tot_fd * 0.25
                data[i, 6] = tot_fd * 0.05
                data[i, 7] = tot_fd * 0.10
                data[i, 8] = tot_fd * 0.08
                data[i, 9] = tot_fd * 0.02
            else:
                data[i, 4] = tot_fd * 0.10
                data[i, 5] = tot_fd * 0.08
                data[i, 6] = tot_fd * 0.02
                data[i, 7] = tot_fd * 0.50
                data[i, 8] = tot_fd * 0.25
                data[i, 9] = tot_fd * 0.05

        fd_col_sums = data[:4, 4:].sum(axis=0)
        data[4, 4:] = 0.02 * fd_col_sums
        return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)

    def test_invalid_backend_name_handling(self):
        """Unsupported backend names must raise actionable ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            backend_available("opencl")

        with pytest.raises(ValueError, match="Unknown backend"):
            get_array_namespace("cuda_unsupported")

        with pytest.raises(ValueError, match="numba backend uses compiled kernels"):
            get_array_namespace("numba")

    def test_to_numpy_conversion_variety(self):
        """to_numpy must convert scalars, numpy arrays, lists, and tuples reliably."""
        assert to_numpy(5) == 5
        assert to_numpy(3.14) == 3.14
        arr = np.array([1, 2, 3])
        assert to_numpy(arr) is arr
        converted_list = to_numpy([10, 20])
        assert isinstance(converted_list, np.ndarray)
        np.testing.assert_array_equal(converted_list, [10, 20])

    def test_trade_solver_fallback_warning_and_numerical_equivalence(self, canonical_trade_calib):
        """solve_trade_equilibrium with unavailable backend must warn and match numpy exactly."""
        res_numpy = solve_trade_equilibrium(canonical_trade_calib, method="condensed", backend="numpy")
        assert res_numpy.converged is True

        with pytest.warns(RuntimeWarning, match="falling back to 'numpy'"):
            res_cupy = solve_trade_equilibrium(canonical_trade_calib, method="condensed", backend="cupy")

        assert res_cupy.converged is True
        np.testing.assert_allclose(res_numpy.w_sol, res_cupy.w_sol, atol=1e-12)
        np.testing.assert_allclose(res_numpy.r_sol, res_cupy.r_sol, atol=1e-12)
        np.testing.assert_allclose(res_numpy.p_sol, res_cupy.p_sol, atol=1e-12)

    def test_allen_arkolakis_fallback_warning_and_numerical_equivalence(self):
        """AllenArkolakisModel with unavailable backend must warn and match numpy."""
        coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        model = AllenArkolakisModel.from_coordinates(coords)

        res_numpy = model.solve_equilibrium(backend="numpy")
        assert res_numpy.converged is True

        with pytest.warns(RuntimeWarning, match="falling back to 'numpy'"):
            res_cupy = model.solve_equilibrium(backend="cupy")

        assert res_cupy.converged is True
        np.testing.assert_allclose(res_numpy.wages, res_cupy.wages, atol=1e-12)
        np.testing.assert_allclose(res_numpy.population, res_cupy.population, atol=1e-12)


# ===========================================================================
# 5. Parallel Bootstrap WASM Single-Threaded Fallbacks
# ===========================================================================

class TestBootstrapWasmFallbackAdversarial:
    """Stress testing bootstrap routines under WASM/single-threaded conditions."""

    @pytest.fixture(autouse=True)
    def restore_runtime(self):
        yield
        os.environ.pop("PUREMACRO_THREADS", None)
        refresh()

    def test_wild_bootstrap_under_wasm_and_thread_exception(self):
        """wild_bootstrap must fall back to serial on ThreadPoolExecutor failure."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(size=40)

        def refit_fn(e):
            return np.array([np.mean(e), np.std(e)])

        res_ref = wild_bootstrap(residuals, refit_fn, n_boot=20, rng=np.random.default_rng(10), n_jobs=1)

        # 1. Under PUREMACRO_THREADS=0
        os.environ["PUREMACRO_THREADS"] = "0"
        refresh()
        assert capabilities().threads is False
        res_wasm = wild_bootstrap(residuals, refit_fn, n_boot=20, rng=np.random.default_rng(10), n_jobs=4)
        np.testing.assert_allclose(res_ref, res_wasm, rtol=1e-12)

        # 2. When ThreadPoolExecutor raises PermissionError (WASM web worker sandbox)
        os.environ.pop("PUREMACRO_THREADS", None)
        refresh()
        with patch("concurrent.futures.ThreadPoolExecutor", side_effect=PermissionError("WASM thread creation blocked")):
            res_caught = wild_bootstrap(residuals, refit_fn, n_boot=20, rng=np.random.default_rng(10), n_jobs=4)
            np.testing.assert_allclose(res_ref, res_caught, rtol=1e-12)

    def test_block_bootstrap_under_thread_exception(self):
        """block_bootstrap must defensively catch thread pool initialization failures."""
        rng = np.random.default_rng(123)
        series = rng.normal(size=50)

        def refit_fn(b):
            return np.array([np.mean(b)])

        res_ref = block_bootstrap(series, refit_fn=refit_fn, B=15, rng=np.random.default_rng(7), n_jobs=1)

        with patch("concurrent.futures.ThreadPoolExecutor", side_effect=RuntimeError("thread pool fail")):
            res_caught = block_bootstrap(series, refit_fn=refit_fn, B=15, rng=np.random.default_rng(7), n_jobs=4)
            np.testing.assert_allclose(res_ref, res_caught, rtol=1e-12)

    def test_bootstrap_bands_var_under_wasm(self):
        """bootstrap_bands in puremacro.var must run serially when threads are disabled."""
        rng = np.random.default_rng(999)
        Y = rng.normal(size=(30, 2))

        def id_fn(A_list, Sigma, **kwargs):
            return np.linalg.cholesky(Sigma)

        os.environ["PUREMACRO_THREADS"] = "0"
        refresh()
        res = bootstrap_bands(
            Y, p=1, identify_fn=id_fn, horizon=3, n_boot=10, rng=np.random.default_rng(5), n_jobs=4
        )
        assert res["point"].shape == (4, 2, 2)
        assert res["lower"].shape == (4, 2, 2)
        assert res["upper"].shape == (4, 2, 2)
