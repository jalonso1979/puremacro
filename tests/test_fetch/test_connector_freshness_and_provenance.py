"""Permanent regression suite for bounded connector provenance, freshness, and panel contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
import urllib.parse
import warnings

import pandas as pd
import pytest

from puremacro import _cache_db, build_climate_panel, build_financial_panel
from puremacro.fetch import freshness as fr, registry
from puremacro.fetch.realtime.bcb import BCB_SGS_URL, fetch_bcb_vintages
from puremacro.fetch.realtime.canary import SchemaDriftError, SchemaDriftWarning
from puremacro.narrative.sources._telemetry import connector_health


class _MockHTTPResponse:
    def __init__(self, body: bytes | str):
        self._body = body.encode("utf-8") if isinstance(body, str) else body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_supported_cohort_and_panel_builders_resolve_in_registry() -> None:
    """Every cohort connector and cross-country panel builder is registered and loadable."""
    panel_ids = set(registry.sources("panel")["id"])
    expected_panels = {
        "wdi.panel",
        "oecd.ana_panel",
        "oecd.stes_panel",
        "eurostat.na_panel",
        "imf.nea_panel",
        "bis.panel",
        "rates.panel",
        "ilostat.panel",
        "pwt.panel",
        "emissions.panel",
        "oecd.qna_panel",
    }
    assert expected_panels.issubset(panel_ids)
    for sid in expected_panels:
        assert callable(registry.load(sid))

    for spec in fr.SUPPORTED_CONNECTOR_COHORT:
        assert callable(registry.load(spec.source_id))


def test_recorded_mode_verifies_identities_and_revisions_offline() -> None:
    """Recorded mode runs 100% offline, checks identities, and detects multi-vintage revisions."""
    report = fr.connector_freshness_report(mode="recorded")
    assert report.mode == "recorded"
    assert report.requests_used == 0
    assert report.has_outages is False
    assert len(report.records) == len(fr.SUPPORTED_CONNECTOR_COHORT)

    by_id = {r.source_id: r for r in report.records}
    for sid, rec in by_id.items():
        assert rec.identity_ok is True, f"{sid}: {rec.identity_detail}"
        assert rec.n_obs > 0
        assert rec.obs_start is not None and rec.obs_end is not None

    assert by_id["fred.alfred"].status == "revised"
    assert by_id["fred.alfred"].n_vintages == 2
    assert by_id["fred.alfred"].revision_detected is True
    assert by_id["vintage.oecd_stes"].status == "revised"
    assert by_id["bls.laus_state"].status == "fresh"
    assert by_id["oecd.qna_panel"].status == "fresh"
    assert by_id["worldbank.energy_transition"].status == "fresh"

    md_en = report.to_markdown(lang="en")
    md_es = report.to_markdown(lang="es")
    assert "Bounded Connector Provenance & Freshness Report" in md_en
    assert "Informe acotado de procedencia y frescura de conectores" in md_es


def test_domain_identity_guards_reject_corrupted_units_and_fixtures() -> None:
    """Domain identity checks catch LAUS accounting errors, QNA deflator drift, and fixture leaks."""
    # 1. LAUS LF != E + U
    bad_laus = pd.DataFrame([
        {"state_fips": "06", "date": "2026-08-01", "lf": 10000.0, "emp": 9000.0, "une": 500.0, "urate": 5.0}
    ])
    ok, msg = fr.check_laus_identity(bad_laus)
    assert ok is False and "LF = E + U violated" in msg

    # 2. QNA deflator inconsistency: gdp_real != gdp / gdp_defl * 100
    bad_qna = pd.DataFrame([
        {"code": "DEU", "date": "2026-01-01", "gdp": 1000.0, "gdp_defl": 100.0, "gdp_real": 850.0}
    ])
    ok_q, msg_q = fr.check_qna_identity(bad_qna)
    assert ok_q is False and "violated" in msg_q

    # 3. Energy share > 100% and fixture provenance
    bad_share = pd.DataFrame([
        {"code": "USA", "date": "2025-01-01", "variable": "elec_gen_renewable_pct_a", "value": 125.0, "source": "ember"}
    ])
    ok_e1, msg_e1 = fr.check_energy_identity(bad_share)
    assert ok_e1 is False and "outside [0, 100]%" in msg_e1

    bad_fixture = pd.DataFrame([
        {"code": "USA", "date": "2025-01-01", "variable": "elec_gen_renewable_pct_a", "value": 25.0, "source": "tests/data/ember_fixture.csv"}
    ])
    ok_e2, msg_e2 = fr.check_energy_identity(bad_fixture)
    assert ok_e2 is False and "prohibited fixture" in msg_e2


def test_all_nine_statuses_across_cache_and_live_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise all 9 statuses in FRESHNESS_STATUSES across cache and rate-budgeted live modes."""
    for env_var in ("BANXICO_API_KEY", "BMX_TOKEN", "PUREMACRO_BANXICO_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    # Ensure ~/.puremacro/credentials.toml does not supply a banxico key
    monkeypatch.setattr("puremacro.fetch.freshness.get_credential", lambda svc: None)

    missing_db = tmp_path / "nonexistent.db"
    rep_empty_cache = fr.connector_freshness_report(
        mode="cache",
        sources=["fred.csv", "vintage.banxico"],
        db_path=missing_db,
        as_of="2026-11-01",
    )
    by_id0 = {r.source_id: r.status for r in rep_empty_cache.records}
    assert by_id0["fred.csv"] == "not_cached"
    assert by_id0["vintage.banxico"] == "credential_missing"

    # Populate a dedicated SQLite cache DB to exercise fresh, stale, revised, empty, degraded_fallback, schema_drift, failed
    db_file = tmp_path / "populated_cache.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_file))
    _cache_db.close_conn()
    try:
        conn = _cache_db.get_conn()
        now_ts = int(time.time())
        # fred.csv -> fresh
        conn.execute(
            "INSERT INTO http_cache (key, url, fetched_at, content_type, body) VALUES (?, ?, ?, ?, ?)",
            ("k1", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPC1", now_ts, "text/csv", b"DATE,GDPC1\n2026-07-01,23950.0\n"),
        )
        # wdi.panel -> stale (obs_end 2018-01-01 vs as_of 2026-11-01 > 550 days)
        wdi_payload = json.dumps([{"page": 1}, [{"countryiso3code": "USA", "date": "2018", "value": 20000.0}]]).encode("utf-8")
        conn.execute(
            "INSERT INTO http_cache (key, url, fetched_at, content_type, body) VALUES (?, ?, ?, ?, ?)",
            ("k2", "https://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.KD?format=json", now_ts, "application/json", wdi_payload),
        )
        # fred.alfred -> revised (two vintages with different values for 2026-07-01)
        conn.executemany(
            "INSERT INTO alfred_vintages (series_id, observation_date, vintage_date, value) VALUES (?, ?, ?, ?)",
            [
                ("GDPC1", "2026-07-01", "2026-08-28", 23910.0),
                ("GDPC1", "2026-07-01", "2026-09-26", 23955.1),
            ],
        )
        # oecd.qna_panel -> empty (header only, 0 observations)
        conn.execute(
            "INSERT INTO http_cache (key, url, fetched_at, content_type, body) VALUES (?, ?, ?, ?, ?)",
            ("k3", "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR,/Q", now_ts, "text/csv", b"TIME_PERIOD,OBS_VALUE\n"),
        )
        # vintage.bcb -> degraded_fallback (cached snapshot + fallback telemetry event)
        conn.execute(
            "INSERT INTO realtime_vintages (provider, country, series_id, observation_date, vintage_date, value) VALUES (?, ?, ?, ?, ?, ?)",
            ("bcb", "BRA", "432", "2026-10-25", "2026-10-26", 11.0),
        )
        conn.execute(
            "INSERT INTO connector_events (ts, source, outcome, fallback_used) VALUES (?, ?, ?, ?)",
            (now_ts, "bcb", "fallback", "sqlite_cache"),
        )
        # worldbank.energy_transition -> schema_drift (share > 100% triggers domain identity drift)
        bad_ember = (
            b"Area,ISO 3 code,Year,Area type,Electricity source,Generation (TWh),Share of generation (%)\n"
            b"United States,USA,2026,Country or economy,Renewables,1000.0,140.0\n"
        )
        conn.execute(
            "INSERT INTO http_cache (key, url, fetched_at, content_type, body) VALUES (?, ?, ?, ?, ?)",
            ("k4", "https://files.ember-energy.org/public-downloads/generation/outputs/release_generation_yearly_lower.csv", now_ts, "text/csv", bad_ember),
        )
        # vintage.oecd_stes -> failed (telemetry recorded timeout with no cached rows)
        conn.execute(
            "INSERT INTO connector_events (ts, source, outcome, fallback_used) VALUES (?, ?, ?, ?)",
            (now_ts, "oecd_stes", "timeout", "none"),
        )
        conn.commit()

        rep_cache = fr.connector_freshness_report(mode="cache", as_of="2026-11-01")
        statuses = {r.source_id: r.status for r in rep_cache.records}
        assert statuses["fred.csv"] == "fresh"
        assert statuses["wdi.panel"] == "stale"
        assert statuses["fred.alfred"] == "revised"
        assert statuses["oecd.qna_panel"] == "empty"
        assert statuses["vintage.bcb"] == "degraded_fallback"
        assert statuses["worldbank.energy_transition"] == "schema_drift"
        assert statuses["vintage.oecd_stes"] == "failed"
        assert statuses["bls.laus_state"] == "not_cached"
        assert statuses["vintage.banxico"] == "credential_missing"
        assert set(statuses.values()) == set(fr.FRESHNESS_STATUSES)
    finally:
        _cache_db.close_conn()

    # Verify rate-budgeted mode="live" enforces max_requests and attributes warnings/errors
    def _ok_probe() -> pd.DataFrame:
        return pd.DataFrame([{"date": "2026-10-01", "value": 24000.0}])

    def _fb_probe() -> pd.DataFrame:
        warnings.warn("BCB live down; falling back to cached vintages", UserWarning)
        return pd.DataFrame([{"date": "2026-10-20", "vintage": "2026-10-20", "value": 10.75}])

    def _drift_probe() -> pd.DataFrame:
        raise SchemaDriftError("bcb", "valor key renamed")

    rep_live = fr.connector_freshness_report(
        mode="live",
        sources=["vintage.banxico", "fred.csv", "vintage.bcb", "wdi.panel", "oecd.qna_panel"],
        as_of="2026-11-01",
        rate_budget=fr.RateBudget(max_requests=3, min_interval_seconds=0.0, timeout_seconds=5.0),
        probe_overrides={
            "fred.csv": _ok_probe,
            "vintage.bcb": _fb_probe,
            "wdi.panel": _drift_probe,
            "oecd.qna_panel": _ok_probe,
        },
    )
    assert rep_live.requests_used == 3
    live_map = {r.source_id: r for r in rep_live.records}
    # vintage.banxico skipped before burning a request because credential is missing
    assert live_map["vintage.banxico"].status == "credential_missing"
    assert live_map["fred.csv"].status == "fresh"
    assert live_map["vintage.bcb"].status == "degraded_fallback"
    assert live_map["wdi.panel"].status == "schema_drift"
    # 4th live probe blocked by max_requests=3
    assert live_map["oecd.qna_panel"].status == "not_cached"
    assert "rate_budget_exhausted" in live_map["oecd.qna_panel"].note


def test_freshness_cli_exit_codes_and_portable_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI writes portable hashed bundle and exits 0 by default, 1 on --fail-on-outage."""
    out_ok = tmp_path / "bundle_ok"
    rc_ok = fr.main(["--mode", "recorded", "--lang", "es", "--output", str(out_ok), "--fail-on-outage"])
    assert rc_ok == 0
    _ = capsys.readouterr()

    manifest_text = (out_ok / "manifest.json").read_text(encoding="utf-8")
    report_text = (out_ok / "freshness_report.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest_text
    assert str(tmp_path) not in report_text

    manifest = json.loads(manifest_text)
    for fname, sha in manifest["files"].items():
        assert hashlib.sha256((out_ok / fname).read_bytes()).hexdigest() == sha

    # Populate a DB with an outage and test default exit 0 vs --fail-on-outage exit 1
    bad_db = tmp_path / "outage.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(bad_db))
    _cache_db.close_conn()
    try:
        conn = _cache_db.get_conn()
        conn.execute(
            "INSERT INTO connector_events (ts, source, outcome, fallback_used) VALUES (?, ?, ?, ?)",
            (int(time.time()), "oecd_stes", "timeout", "none"),
        )
        conn.commit()
    finally:
        _cache_db.close_conn()

    assert fr.main(["--mode", "cache", "--db-path", str(bad_db), "--sources", "vintage.oecd_stes"]) == 0
    assert (
        fr.main([
            "--mode", "cache",
            "--db-path", str(bad_db),
            "--sources", "vintage.oecd_stes",
            "--fail-on-outage",
        ])
        == 1
    )
    _ = capsys.readouterr()


def test_connector_health_does_not_count_live_snapshots_as_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Snapshot live successes ('success', 'none') have n_fallback == 0 while true fallbacks count."""
    db_file = tmp_path / "telemetry_fix.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_file))
    _cache_db.close_conn()
    try:
        conn = _cache_db.get_conn()
        now = int(time.time())
        conn.executemany(
            "INSERT INTO connector_events (ts, source, outcome, fallback_used) VALUES (?, ?, ?, ?)",
            [
                (now, "bcb", "success", "none"),
                (now, "bcb", "success", "none"),
                (now, "banxico", "success", "none"),
                (now, "banxico", "fallback", "sqlite_cache"),
                (now, "us_cbo", "success", "live"),
                (now, "us_cbo", "parser_schema_mismatch", "none"),
            ],
        )
        conn.commit()
        health = connector_health().set_index("source")
        assert int(health.loc["bcb", "n_total"]) == 2
        assert int(health.loc["bcb", "n_success"]) == 2
        assert int(health.loc["bcb", "n_fallback"]) == 0
        assert float(health.loc["bcb", "fallback_rate"]) == 0.0

        assert int(health.loc["banxico", "n_fallback"]) == 1
        assert float(health.loc["banxico", "fallback_rate"]) == pytest.approx(0.5)

        assert int(health.loc["us_cbo", "n_fallback"]) == 1
        assert float(health.loc["us_cbo", "fallback_rate"]) == pytest.approx(0.5)
    finally:
        _cache_db.close_conn()


def test_climate_and_financial_panels_honor_kwargs_and_surface_source_status() -> None:
    """Panel builders accept documented kwargs, warn on upstream exceptions, and attach source_status."""
    with patch("puremacro.fetch.emissions.fetch_emissions_panel", side_effect=RuntimeError("WDI down")), patch(
        "puremacro.fetch.energy_transition.fetch_energy_transition", side_effect=RuntimeError("Ember down")
    ):
        with pytest.warns(RuntimeWarning, match="upstream 'emissions' fetch failed"):
            clim = build_climate_panel(
                codes=["USA"],
                start_year=2015,
                include_emissions=True,
                include_energy=True,
                include_gdp=False,
            )
    assert clim.empty
    c_status = clim.attrs["source_status"]
    assert c_status["emissions"]["status"] == "failed"
    assert "WDI down" in c_status["emissions"]["error"]
    assert c_status["energy_transition"]["status"] == "failed"
    assert c_status["macro_output"]["status"] == "skipped"

    # Genuine empty country return emits NO warning and marks status='empty'
    empty_em = pd.DataFrame(columns=["code", "date", "variable", "value", "sa_source", "source"])
    with patch("puremacro.fetch.emissions.fetch_emissions_panel", return_value=empty_em), patch(
        "puremacro.fetch.energy_transition.fetch_energy_transition", return_value=empty_em
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clim_zz = build_climate_panel(codes=["ZZZ"], include_gdp=False)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert clim_zz.attrs["source_status"]["emissions"]["status"] == "empty"

    # Financial panel honors include_credit=False and warns on upstream failure
    with patch("puremacro.fetch.financial.fetch_sovereign_yields", side_effect=RuntimeError("FRED 503")), patch(
        "puremacro.fetch.financial.fetch_policy_rates", return_value=empty_em
    ):
        with pytest.warns(RuntimeWarning, match="upstream 'sovereign_yields' fetch failed"):
            fin = build_financial_panel(
                codes=["DEU"],
                start_date="2015-01-01",
                include_credit=False,
                include_commodities=False,
                include_conditions=False,
            )
    assert fin.empty
    f_status = fin.attrs["source_status"]
    assert f_status["sovereign_yields"]["status"] == "failed"
    assert f_status["policy_rates"]["status"] == "empty"
    assert f_status["bis_macroprudential"]["status"] == "skipped"
    assert f_status["commodities"]["status"] == "skipped"


def test_bcb_daily_series_uses_bounded_ten_year_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Daily BCB series ('432') automatically supplies <=10-year dataInicial/dataFinal bounds."""
    db_file = tmp_path / "bcb_win.db"
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(db_file))
    _cache_db.close_conn()
    seen_urls: list[str] = []

    def fake_urlopen(req: object, timeout: float | None = None) -> _MockHTTPResponse:
        url = getattr(req, "full_url", str(req))
        seen_urls.append(url)
        return _MockHTTPResponse('[{"data": "15/03/2026", "valor": "10.75"}]')

    try:
        assert BCB_SGS_URL.format(series_id="432") == (
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados?formato=json"
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            df_daily = fetch_bcb_vintages("432", vintage_date="2026-04-01", use_cache=False)
            df_monthly = fetch_bcb_vintages("433", vintage_date="2026-04-01", use_cache=False)
            df_custom = fetch_bcb_vintages(
                "433",
                vintage_date="2026-04-01",
                start_date="2020-01-01",
                end_date="2026-03-31",
                use_cache=False,
            )

        assert len(df_daily) == 1 and len(df_monthly) == 1 and len(df_custom) == 1
        # Daily '432' got automatic 10-year window ending 01/04/2026
        q_daily = dict(urllib.parse.parse_qsl(seen_urls[0].partition("?")[2]))
        assert q_daily["formato"] == "json"
        assert q_daily["dataFinal"] == "01/04/2026"
        assert q_daily["dataInicial"] == "02/04/2016"

        # Monthly '433' without explicit bounds stays bare
        assert seen_urls[1] == BCB_SGS_URL.format(series_id="433")

        # Explicit start_date / end_date formatted as dd/mm/yyyy
        q_custom = dict(urllib.parse.parse_qsl(seen_urls[2].partition("?")[2]))
        assert q_custom["dataInicial"] == "01/01/2020"
        assert q_custom["dataFinal"] == "31/03/2026"
    finally:
        _cache_db.close_conn()
