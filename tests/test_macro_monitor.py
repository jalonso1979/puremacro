"""Tests for puremacro.macro_monitor (Track 4A)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro.macro_monitor import (
    MacroMonitorConfig,
    MacroMonitorReport,
    run_macro_monitor,
)


def _make_clean_latam_panel(
    *,
    n_periods: int = 32,
    countries: tuple[str, ...] = ("BRA", "CHL", "COL", "MEX"),
    spike_last: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    dates = pd.date_range("2018-01-01", periods=n_periods, freq="QS-JAN")
    rows: list[dict[str, object]] = []
    for c_idx, code in enumerate(countries):
        base_g = 0.6 + 0.1 * c_idx
        rate = 6.5
        infl = 1.0
        for t_idx, dt in enumerate(dates):
            rate = 6.5 + 0.35 * (rate - 6.5) + float(rng.normal(0.0, 0.35))
            infl = 1.0 + 0.25 * (infl - 1.0) + float(rng.normal(0.0, 0.25))
            gdp_g = base_g - 0.18 * (rate - 6.5) + 0.10 * (infl - 1.0) + float(rng.normal(0.0, 0.30))
            if spike_last and code == "MEX" and t_idx == n_periods - 1:
                gdp_g = 9.5
            rows.append(
                {
                    "code": code,
                    "date": dt.strftime("%Y-%m-%d"),
                    "gdp_growth": float(gdp_g),
                    "policy_rate": float(rate),
                    "inflation": float(infl),
                }
            )
    return pd.DataFrame(rows)


def _assert_no_host_paths(text: str, forbidden_roots: tuple[str, ...] = ()) -> None:
    for root in forbidden_roots:
        if root and len(root) > 3:
            assert root not in text, f"Host path leaked into serialized artifact: {root!r}"
    for prefix in ("/Users/", "/home/", "/private/var/", "/var/folders/", "/tmp/"):
        assert prefix not in text, f"Host filesystem prefix {prefix!r} leaked into output."


def test_recorded_end_to_end_offline() -> None:
    """Default recorded mode runs offline with finite country indicators, preflight, and LP grid."""
    report = run_macro_monitor(mode="recorded")

    assert isinstance(report, MacroMonitorReport)
    assert report.mode == "recorded"
    assert report.passed is True
    assert report.overall_status in {"OK", "WARN"}
    assert report.has_outages is False
    assert report.connectors_total == 9
    assert report.connectors_ok == 9
    assert report.outage_connectors == ()
    assert report.stale_connectors == ()
    assert len(report.input_sha256) == 64
    assert report.input_source == "recorded:qna40_panel"
    assert report.n_countries == 6
    assert set(report.countries) == {"DEU", "ESP", "ITA", "JPN", "MEX", "USA"}
    assert report.sample_start < report.sample_end
    assert report.n_lp_specs_ok == 4
    assert report.n_lp_specs_failed == 0

    fresh_df = report.freshness_df
    assert len(fresh_df) == 9
    assert {"source_id", "status", "provider", "registry_kind"}.issubset(fresh_df.columns)

    reg_df = report.registry_df
    assert not reg_df.empty
    assert "oecd.qna_panel" in set(reg_df["id"])

    ind_df = report.latest_indicators_df
    assert not ind_df.empty
    assert set(ind_df["country"]) == {"DEU", "ESP", "ITA", "JPN", "MEX", "USA"}
    assert np.all(np.isfinite(ind_df["latest_value"]))

    nc_df = report.nowcast_df
    assert len(nc_df) == 6
    assert np.all(np.isfinite(nc_df["nowcast_next"]))

    lp_metrics = report.lp_metrics_df
    assert len(lp_metrics) == 4
    assert not report.lp_table_df.empty


def test_custom_latam_csv_and_dataframe_with_anomaly_flag(tmp_path: Path) -> None:
    """Custom CSV and DataFrame panels run cleanly, strip directory paths, and flag z-score spikes."""
    df = _make_clean_latam_panel(n_periods=32, spike_last=True)
    csv_path = tmp_path / "secret_host_dir" / "latam_custom_panel.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    rep_csv = run_macro_monitor(
        csv_path,
        mode="recorded",
        sources=("vintage.banxico", "vintage.bcb", "oecd.qna_panel"),
        y="gdp_growth",
        x="policy_rate",
        controls=("inflation",),
        lang="en",
    )
    assert rep_csv.passed is True
    assert rep_csv.overall_status in {"OK", "WARN"}
    assert rep_csv.input_source == "file:latam_custom_panel.csv"
    assert rep_csv.input_sha256 == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert rep_csv.n_countries == 4
    assert rep_csv.n_lp_specs_ok == 4

    ind_df = rep_csv.latest_indicators_df
    mex_gdp = ind_df[(ind_df["country"] == "MEX") & (ind_df["variable"] == "gdp_growth")].iloc[0]
    assert mex_gdp["flag"] == "Z_ALERT_HIGH"
    assert float(mex_gdp["z_score"]) >= 2.0

    # Ensure zero directory path leakage when ingested from a host file path.
    _assert_no_host_paths(rep_csv.to_json(), forbidden_roots=(str(tmp_path), str(csv_path)))
    _assert_no_host_paths(rep_csv.to_markdown(), forbidden_roots=(str(tmp_path), str(csv_path)))
    _assert_no_host_paths(rep_csv.to_html(), forbidden_roots=(str(tmp_path), str(csv_path)))

    # Equivalent DataFrame path produces matching indicators.
    rep_df = run_macro_monitor(
        df,
        mode="recorded",
        sources=("vintage.banxico", "vintage.bcb", "oecd.qna_panel"),
        y="gdp_growth",
        x="policy_rate",
        controls=("inflation",),
    )
    assert rep_df.input_source == "dataframe"
    assert len(rep_df.latest_indicators_df) == len(rep_csv.latest_indicators_df)


def test_fail_closed_unsorted_and_corrupted_dates() -> None:
    """Unsorted and corrupted dates must explicitly fail rather than silently sorting/passing."""
    base_df = _make_clean_latam_panel(n_periods=24)

    # 1) Unsorted dates inside MEX.
    unsorted_df = base_df.copy()
    mex_idx = unsorted_df.index[unsorted_df["code"] == "MEX"].tolist()
    unsorted_df.loc[[mex_idx[1], mex_idx[5]]] = unsorted_df.loc[[mex_idx[5], mex_idx[1]]].to_numpy()
    rep_unsorted = run_macro_monitor(
        unsorted_df,
        mode="recorded",
        sources=("vintage.banxico",),
        y="gdp_growth",
        x="policy_rate",
    )
    assert rep_unsorted.overall_status == "FAILED"
    assert rep_unsorted.passed is False
    assert rep_unsorted.preflight_status == "fail"
    assert any("unsorted_dates" in err for err in rep_unsorted.errors)
    assert "errors=" in rep_unsorted.headline()

    # 2) Corrupted unparseable date string.
    bad_date_df = base_df.copy()
    bad_date_df.loc[bad_date_df.index[3], "date"] = "NOT-A-VALID-DATE"
    rep_corrupt = run_macro_monitor(
        bad_date_df,
        mode="recorded",
        sources=("vintage.banxico",),
        y="gdp_growth",
        x="policy_rate",
    )
    assert rep_corrupt.overall_status == "FAILED"
    assert rep_corrupt.passed is False
    assert rep_corrupt.preflight_status == "fail"
    assert any("corrupted_dates" in err for err in rep_corrupt.errors)


def test_fail_closed_missing_required_columns_and_bad_paths(tmp_path: Path) -> None:
    """Missing required columns and empty panels fail closed; missing files raise FileNotFoundError."""
    df = _make_clean_latam_panel(n_periods=16)

    rep_missing_y = run_macro_monitor(
        df.drop(columns=["gdp_growth"]),
        mode="recorded",
        sources=("vintage.banxico",),
        y="gdp_growth",
        x="policy_rate",
    )
    assert rep_missing_y.overall_status == "FAILED"
    assert rep_missing_y.passed is False
    assert any("missing_required_column" in err for err in rep_missing_y.errors)

    rep_no_time = run_macro_monitor(
        df.drop(columns=["date"]),
        mode="recorded",
        sources=("vintage.banxico",),
    )
    assert rep_no_time.overall_status == "FAILED"
    assert rep_no_time.passed is False
    assert any("missing_required_column" in err for err in rep_no_time.errors)

    rep_empty = run_macro_monitor(
        df.iloc[0:0],
        mode="recorded",
        sources=("vintage.banxico",),
    )
    assert rep_empty.overall_status == "FAILED"
    assert rep_empty.passed is False
    assert any("empty" in err for err in rep_empty.errors)

    with pytest.raises(FileNotFoundError):
        run_macro_monitor(tmp_path / "does_not_exist.csv", mode="recorded")


def test_fail_closed_live_outages_stale_and_empty_cache(tmp_path: Path) -> None:
    """Live connector failures, empty returns, stale data, and uncached DB surface explicit failures."""
    clean_df = _make_clean_latam_panel(n_periods=24)

    def _boom() -> pd.DataFrame:
        raise RuntimeError(f"blocked upstream endpoint near {tmp_path}/secret.json")

    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=["date", "value"])

    def _stale() -> pd.DataFrame:
        return pd.DataFrame({"date": ["2010-01-01", "2010-04-01"], "value": [10.0, 10.5]})

    def _fallback() -> pd.DataFrame:
        warnings.warn("upstream endpoint blocked; falling back to cached vintage", UserWarning, stacklevel=2)
        return pd.DataFrame({"date": ["2026-09-01", "2026-10-01"], "value": [10.5, 10.75]})

    # Live failed + empty probes -> explicit FAILED status + outages visible in headline & zero host path leakage.
    rep_outage = run_macro_monitor(
        clean_df,
        mode="live",
        sources=("vintage.banxico", "vintage.bcb"),
        as_of="2026-10-15",
        y="gdp_growth",
        x="policy_rate",
        probe_overrides={"vintage.banxico": _boom, "vintage.bcb": _empty},
    )
    assert rep_outage.has_outages is True
    assert rep_outage.passed is False
    assert rep_outage.overall_status == "FAILED"
    assert set(rep_outage.outage_connectors) == {"vintage.banxico", "vintage.bcb"}
    assert "outages=" in rep_outage.headline()
    assert any("status=failed" in w for w in rep_outage.warnings)
    assert any("status=empty" in w for w in rep_outage.warnings)
    _assert_no_host_paths(rep_outage.to_json(), forbidden_roots=(str(tmp_path),))

    # Stale probe -> explicit DEGRADED status & stale connector headline/warnings.
    rep_stale = run_macro_monitor(
        clean_df,
        mode="live",
        sources=("vintage.banxico",),
        as_of="2026-10-15",
        y="gdp_growth",
        x="policy_rate",
        probe_overrides={"vintage.banxico": _stale},
    )
    assert rep_stale.passed is False
    assert rep_stale.overall_status == "DEGRADED"
    assert rep_stale.stale_connectors == ("vintage.banxico",)
    assert "stale=vintage.banxico" in rep_stale.headline()
    assert any("status=stale" in w for w in rep_stale.warnings)

    # Fallback probe -> explicit degraded_fallback outage status & DEGRADED overall status.
    rep_fb = run_macro_monitor(
        clean_df,
        mode="live",
        sources=("vintage.banxico",),
        as_of="2026-10-15",
        y="gdp_growth",
        x="policy_rate",
        probe_overrides={"vintage.banxico": _fallback},
    )
    assert rep_fb.has_outages is True
    assert rep_fb.passed is False
    assert rep_fb.overall_status == "DEGRADED"
    assert rep_fb.outage_connectors == ("vintage.banxico",)
    assert "outages=vintage.banxico" in rep_fb.headline()
    assert any("degraded_fallback" in w for w in rep_fb.warnings)

    # Empty cache SQLite database -> connectors_ok == 0 -> DEGRADED, not clean OK.
    empty_db = tmp_path / "empty_vintage_cache.sqlite"
    rep_cache = run_macro_monitor(
        clean_df,
        mode="cache",
        sources=("vintage.banxico", "vintage.bcb"),
        db_path=empty_db,
        y="gdp_growth",
        x="policy_rate",
    )
    assert rep_cache.connectors_ok == 0
    assert rep_cache.passed is False
    assert rep_cache.overall_status == "DEGRADED"
    assert any("not_cached" in w for w in rep_cache.warnings)
    _assert_no_host_paths(rep_cache.to_json(), forbidden_roots=(str(tmp_path), str(empty_db)))


def test_bilingual_bundle_export_sha256_and_zero_leakage(tmp_path: Path) -> None:
    """write_bundle writes valid EN/ES UTF-8 artifacts, verifies SHA-256 manifest, and leaks zero paths."""
    df = _make_clean_latam_panel(n_periods=28)
    csv_in = tmp_path / "panel_in.csv"
    df.to_csv(csv_in, index=False, encoding="utf-8")

    rep = run_macro_monitor(
        csv_in,
        mode="recorded",
        sources=("vintage.banxico", "vintage.bcb", "oecd.qna_panel"),
        y="gdp_growth",
        x="policy_rate",
        controls=("inflation",),
        lang="en",
    )

    for lang_code, expected_marker in (
        ("en", "Automated LatAm & Macro Freshness Monitor"),
        ("es", "Monitor Automatizado de Actualidad Macro y LatAm"),
    ):
        bundle_dir = tmp_path / f"bundle_{lang_code}"
        written = rep.write_bundle(bundle_dir, lang=lang_code)
        expected_files = {
            "monitor.md",
            "monitor.html",
            "monitor.json",
            "freshness.csv",
            "latest_indicators.csv",
            "manifest.json",
        }
        assert set(written) == expected_files
        for fname, path_obj in written.items():
            assert path_obj.is_file(), f"Missing bundle file {fname}"

        manifest = json.loads(written["manifest.json"].read_text(encoding="utf-8"))
        assert manifest["lang"] == lang_code
        assert set(manifest["files"]) == expected_files - {"manifest.json"}
        for rel_name, meta in manifest["files"].items():
            raw = (bundle_dir / rel_name).read_bytes()
            assert meta["bytes"] == len(raw)
            assert meta["sha256"] == hashlib.sha256(raw).hexdigest()

        md_text = written["monitor.md"].read_text(encoding="utf-8")
        html_text = written["monitor.html"].read_text(encoding="utf-8")
        json_text = written["monitor.json"].read_text(encoding="utf-8")
        fresh_csv = written["freshness.csv"].read_text(encoding="utf-8")
        ind_csv = written["latest_indicators.csv"].read_text(encoding="utf-8")
        manifest_text = written["manifest.json"].read_text(encoding="utf-8")

        assert expected_marker in md_text
        assert html.escape(expected_marker) in html_text
        assert f'<html lang="{lang_code}">' in html_text
        assert '<meta charset="utf-8">' in html_text

        # Strict RFC 8259 JSON check (disallowing NaN/Infinity).
        parsed_json = json.loads(json_text)
        assert parsed_json["schema"] == "puremacro.macro_monitor.report.v1"
        assert len(parsed_json["indicators"]) == 12
        json.dumps(parsed_json, allow_nan=False)

        assert len(pd.read_csv(written["freshness.csv"])) == 3
        assert len(pd.read_csv(written["latest_indicators.csv"])) == 12

        for payload_txt in (md_text, html_text, json_text, fresh_csv, ind_csv, manifest_text):
            _assert_no_host_paths(
                payload_txt,
                forbidden_roots=(str(tmp_path), str(bundle_dir), str(csv_in)),
            )


def test_short_sample_lp_instability_surface_warning() -> None:
    """Short panel series surfaces short_sample / instability warnings rather than crashing."""
    short_df = _make_clean_latam_panel(n_periods=8, countries=("MEX", "BRA"))
    rep = run_macro_monitor(
        short_df,
        mode="recorded",
        sources=("vintage.banxico",),
        y="gdp_growth",
        x="policy_rate",
        horizons=(0, 1, 2, 3, 4, 5),
        lp_lags=(2, 3),
        short_sample_min_obs=24,
    )
    assert any("short_sample" in w or "lp_" in w for w in rep.warnings)
    assert json.loads(rep.to_json())["overall_status"] == rep.overall_status


def test_immutability_and_config_validation() -> None:
    """Configs/reports are frozen and returned DataFrames/dicts are isolated deep copies."""
    cfg = MacroMonitorConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.mode = "live"  # type: ignore[misc]

    with pytest.raises(ValueError):
        MacroMonitorConfig(mode="invalid_mode")
    with pytest.raises(ValueError):
        MacroMonitorConfig(lang="fr")
    with pytest.raises(ValueError):
        MacroMonitorConfig(ci=1.2)

    df = _make_clean_latam_panel(n_periods=24)
    rep = run_macro_monitor(
        df,
        mode="recorded",
        sources=("vintage.banxico",),
        y="gdp_growth",
        x="policy_rate",
    )
    with pytest.raises(FrozenInstanceError):
        rep.overall_status = "MUTATED"  # type: ignore[misc]

    # Mutating returned tables or dicts must not affect subsequent reads from rep.
    fresh_copy = rep.freshness_df
    fresh_copy.loc[0, "status"] = "MUTATED"
    assert rep.freshness_df.loc[0, "status"] != "MUTATED"

    ind_copy = rep.to_frame("latest_indicators")
    ind_copy.loc[0, "flag"] = "MUTATED"
    assert rep.to_frame("latest_indicators").loc[0, "flag"] != "MUTATED"

    sc = rep.status_counts
    sc["fake"] = 999
    assert "fake" not in rep.status_counts


def test_cli_recorded_bundle_windows_utf8_and_failures(tmp_path: Path) -> None:
    """Subprocess CLI exits 0 on recorded run under cp1252 stdio and exits 1 on outages/bad CSV."""
    out_dir = tmp_path / "cli_out"
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"
    env["PYTHONUTF8"] = "0"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "puremacro.macro_monitor",
            "--mode",
            "recorded",
            "--lang",
            "es",
            "--output-dir",
            str(out_dir),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "monitor.html").is_file()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["lang"] == "es"
    _assert_no_host_paths(proc.stdout, forbidden_roots=(str(tmp_path),))

    # --fail-on-outage exits non-zero when cache DB is empty.
    empty_db = tmp_path / "cli_empty.sqlite"
    fail_outage = subprocess.run(
        [
            sys.executable,
            "-m",
            "puremacro.macro_monitor",
            "--mode",
            "cache",
            "--db-path",
            str(empty_db),
            "--sources",
            "vintage.banxico",
            "--fail-on-outage",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert fail_outage.returncode == 1

    # Corrupted date CSV input causes CLI to fail closed with exit code 1.
    bad_csv = tmp_path / "bad_dates.csv"
    bad_df = _make_clean_latam_panel(n_periods=16)
    bad_df.loc[2, "date"] = "CORRUPTED"
    bad_df.to_csv(bad_csv, index=False, encoding="utf-8")
    fail_csv = subprocess.run(
        [
            sys.executable,
            "-m",
            "puremacro.macro_monitor",
            "--mode",
            "recorded",
            "--input",
            str(bad_csv),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert fail_csv.returncode == 1


def test_pyodide_safe_top_level_imports_subprocess() -> None:
    """Importing puremacro.macro_monitor must never pull network or native-heavy modules."""
    code = (
        "import sys, puremacro.macro_monitor as mm; "
        "blocked = ['requests', 'statsmodels', 'linearmodels', 'arch', 'sqlite3', 'ssl', 'scipy', 'matplotlib']; "
        "found = [m for m in blocked if m in sys.modules]; "
        "assert not found, f'Disallowed eager imports: {found}'; "
        "assert hasattr(mm, 'run_macro_monitor') and hasattr(mm, 'MacroMonitorConfig')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
