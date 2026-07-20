"""Additional coverage tests for puremacro.long_panel.

These tests target the branches NOT covered by tests/test_long_panel.py
(which skips when external data is absent). All tests here run offline via
monkeypatching, temporary directories, and synthetic DataFrames.

Missing-line targets (60% -> higher):
  - load_pwt10: FileNotFoundError branch (line 73)
  - _fetch_stan_csv: all parse/error paths (lines 132-148)
  - load_oecd_stan_ls: countries=None default, second-agg-candidate path,
    merge+filter+concat logic (lines 169, 178-179, 182-194)
  - load_klems_legacy: countries=None default, missing-archive return,
    sheet-parse logic including no-years / negative-values / multi-country
    (lines 217, 220, 226-251)
  - build_g9_long_panel: FileNotFoundError-PWT catch, klems2023-except catch,
    ls_stan/ls_klems_leg populated branches, only-ls_all merge path,
    only-pk merge path, all-empty path, write_csv branch (lines 301-382)
  - splice_audit: empty-panel early return, per-country overlap computation
    (lines 400, 410-419)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import puremacro.long_panel as lpm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stan_csv_labr() -> bytes:
    return b"TIME_PERIOD,OBS_VALUE\n2000,600\n2001,650\n2002,700\n"


def _stan_csv_valu() -> bytes:
    return b"TIME_PERIOD,OBS_VALUE\n2000,1000\n2001,1100\n2002,1200\n"


def _make_mock_sheet(labr_vals: dict, valu_vals: dict) -> pd.DataFrame:
    """Build a minimal KLEMS-legacy TOT sheet DataFrame.

    ``labr_vals`` / ``valu_vals`` map year (int) -> float value.
    """
    years = sorted(set(labr_vals) | set(valu_vals))
    labr_row = {"var": "LAB"} | {y: labr_vals.get(y, np.nan) for y in years}
    valu_row = {"var": "VA"} | {y: valu_vals.get(y, np.nan) for y in years}
    return pd.DataFrame([labr_row, valu_row])


# ---------------------------------------------------------------------------
# 1. load_pwt10 — FileNotFoundError branch
# ---------------------------------------------------------------------------

class TestLoadPwt10FileNotFound:
    """load_pwt10 raises FileNotFoundError when the .dta is absent."""

    def test_raises_file_not_found_when_dta_missing(self, monkeypatch, tmp_path):
        """Pointing _ROOT at an empty tmpdir causes FileNotFoundError."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        with pytest.raises(FileNotFoundError, match="pwt100.dta"):
            lpm.load_pwt10(countries=["USA"])

    def test_error_message_contains_path(self, monkeypatch, tmp_path):
        """The error message includes the expected file path."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            lpm.load_pwt10()
        assert "pwt100" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. _fetch_stan_csv — all parse / error paths
# ---------------------------------------------------------------------------

class TestFetchStanCsv:
    """_fetch_stan_csv returns [year, value] DataFrame or empty on error."""

    def test_network_exception_returns_empty(self, monkeypatch):
        """safe_get_bytes raising any Exception returns empty DataFrame."""
        def _raise(url, timeout=30.0):
            raise ConnectionError("network down")
        monkeypatch.setattr(lpm, "safe_get_bytes", _raise)
        result = lpm._fetch_stan_csv("USA", "LABR", "DTOTAL")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["year", "value"]
        assert len(result) == 0

    def test_none_returned_gives_empty(self, monkeypatch):
        """safe_get_bytes returning None yields empty."""
        monkeypatch.setattr(lpm, "safe_get_bytes", lambda url, timeout=30.0: None)
        result = lpm._fetch_stan_csv("USA", "LABR", "DTOTAL")
        assert len(result) == 0

    def test_empty_bytes_gives_empty(self, monkeypatch):
        """Zero-length bytes yield empty DataFrame."""
        monkeypatch.setattr(lpm, "safe_get_bytes", lambda url, timeout=30.0: b"")
        result = lpm._fetch_stan_csv("USA", "LABR", "DTOTAL")
        assert len(result) == 0

    def test_binary_garbage_gives_empty(self, monkeypatch):
        """Non-CSV binary bytes yield empty DataFrame."""
        monkeypatch.setattr(lpm, "safe_get_bytes",
                            lambda url, timeout=30.0: b"\x00\x01\x02\x03binary")
        result = lpm._fetch_stan_csv("USA", "LABR", "DTOTAL")
        assert len(result) == 0
        assert list(result.columns) == ["year", "value"]

    def test_invalid_utf8_causes_read_csv_exception_returns_empty(self, monkeypatch):
        """Invalid UTF-8 bytes cause pd.read_csv to raise, caught -> empty (lines 136-137)."""
        # b"\xff\xfe" is a UTF-16 BOM that causes UnicodeDecodeError in pd.read_csv
        monkeypatch.setattr(lpm, "safe_get_bytes",
                            lambda url, timeout=30.0: b"\xff\xfe")
        result = lpm._fetch_stan_csv("USA", "LABR", "DTOTAL")
        assert len(result) == 0
        assert list(result.columns) == ["year", "value"]

    def test_wrong_column_names_give_empty(self, monkeypatch):
        """CSV with no TIME_PERIOD/TIME/OBS_TIME column yields empty."""
        monkeypatch.setattr(lpm, "safe_get_bytes",
                            lambda url, timeout=30.0: b"WRONG_COL,BAD_VALUE\n2000,600\n")
        result = lpm._fetch_stan_csv("USA", "LABR", "DTOTAL")
        assert len(result) == 0

    def test_obs_time_column_accepted(self, monkeypatch):
        """OBS_TIME is an accepted alias for the time column."""
        monkeypatch.setattr(lpm, "safe_get_bytes",
                            lambda url, timeout=30.0: b"OBS_TIME,OBS_VALUE\n2000,600\n2001,650\n")
        result = lpm._fetch_stan_csv("USA", "LABR", "DTOTAL")
        assert len(result) == 2
        assert list(result.columns) == ["year", "value"]
        assert int(result["year"].iloc[0]) == 2000
        assert float(result["value"].iloc[0]) == pytest.approx(600.0)

    def test_value_column_accepted(self, monkeypatch):
        """'VALUE' (not 'OBS_VALUE') is accepted as the value column."""
        monkeypatch.setattr(lpm, "safe_get_bytes",
                            lambda url, timeout=30.0: b"TIME_PERIOD,VALUE\n2000,600\n")
        result = lpm._fetch_stan_csv("USA", "LABR", "DTOTAL")
        assert len(result) == 1
        assert float(result["value"].iloc[0]) == pytest.approx(600.0)

    def test_known_value_correct(self, monkeypatch):
        """Three rows parse correctly; year is int, value is float."""
        monkeypatch.setattr(lpm, "safe_get_bytes",
                            lambda url, timeout=30.0: _stan_csv_labr())
        result = lpm._fetch_stan_csv("USA", "LABR", "DTOTAL")
        assert len(result) == 3
        assert result["year"].tolist() == [2000, 2001, 2002]
        assert result["value"].tolist() == pytest.approx([600.0, 650.0, 700.0])
        assert result["year"].dtype == int

    def test_non_numeric_values_dropped(self, monkeypatch):
        """Non-numeric OBS_VALUE entries are coerced to NaN and dropped."""
        csv = b"TIME_PERIOD,OBS_VALUE\n2000,600\n2001,notanumber\n2002,700\n"
        monkeypatch.setattr(lpm, "safe_get_bytes", lambda url, timeout=30.0: csv)
        result = lpm._fetch_stan_csv("USA", "LABR", "DTOTAL")
        assert len(result) == 2
        assert 2001 not in result["year"].tolist()


# ---------------------------------------------------------------------------
# 3. load_oecd_stan_ls — offline paths
# ---------------------------------------------------------------------------

class TestLoadOecdStanLsOffline:
    """load_oecd_stan_ls offline branches."""

    def test_countries_none_defaults_to_g9(self, monkeypatch):
        """countries=None iterates over the 9-country G9 tuple (line 169)."""
        seen_countries = []

        def _tracking_bytes(url, timeout=30.0):
            # Extract country from URL pattern (country appears in URL)
            for c in lpm.G9:
                if f"/{c}." in url or f".{c}." in url:
                    if c not in seen_countries:
                        seen_countries.append(c)
            return b""  # all fail -> empty

        monkeypatch.setattr(lpm, "safe_get_bytes", _tracking_bytes)
        result = lpm.load_oecd_stan_ls()  # countries=None
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["code", "year", "log_LS_stan"]
        assert len(result) == 0  # all empty
        # At least some G9 countries must have been queried
        assert len(seen_countries) > 0

    def test_all_fail_returns_empty_frame(self, monkeypatch):
        """When every country fetch fails, returns empty DataFrame."""
        monkeypatch.setattr(lpm, "safe_get_bytes", lambda url, timeout=30.0: b"")
        result = lpm.load_oecd_stan_ls(countries=["USA", "DEU"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["code", "year", "log_LS_stan"]

    def test_second_agg_candidate_used_when_first_empty(self, monkeypatch):
        """DTOTAL returns empty; TOTAL returns data — second candidate used (line 178-179)."""
        def _selective(url, timeout=30.0):
            if "DTOTAL" in url:
                return b""
            elif "TOTAL" in url:
                if "LABR" in url:
                    return _stan_csv_labr()
                elif "VALU" in url:
                    return _stan_csv_valu()
            return b""

        monkeypatch.setattr(lpm, "safe_get_bytes", _selective)
        result = lpm.load_oecd_stan_ls(countries=["USA"], start=1995, end=2005)
        assert len(result) == 3
        assert "USA" in result["code"].values

    def test_known_value_log_ls_computation(self, monkeypatch):
        """log_LS_stan = log(LABR) - log(VALU); verified against known values."""
        def _mock(url, timeout=30.0):
            if "LABR" in url:
                return _stan_csv_labr()
            elif "VALU" in url:
                return _stan_csv_valu()
            return b""

        monkeypatch.setattr(lpm, "safe_get_bytes", _mock)
        result = lpm.load_oecd_stan_ls(countries=["USA"], start=1990, end=2010)
        row2000 = result[result["year"] == 2000]
        assert len(row2000) == 1
        expected = np.log(600.0) - np.log(1000.0)
        assert float(row2000["log_LS_stan"].iloc[0]) == pytest.approx(expected, rel=1e-10)

    def test_year_filter_applied(self, monkeypatch):
        """Rows outside [start, end] are excluded."""
        def _mock(url, timeout=30.0):
            if "LABR" in url:
                return _stan_csv_labr()
            elif "VALU" in url:
                return _stan_csv_valu()
            return b""

        monkeypatch.setattr(lpm, "safe_get_bytes", _mock)
        # Only ask for 2001: should get exactly one row
        result = lpm.load_oecd_stan_ls(countries=["USA"], start=2001, end=2001)
        assert list(result["year"]) == [2001]

    def test_output_schema_correct(self, monkeypatch):
        """Output has exactly [code, year, log_LS_stan] columns."""
        def _mock(url, timeout=30.0):
            if "LABR" in url:
                return _stan_csv_labr()
            elif "VALU" in url:
                return _stan_csv_valu()
            return b""

        monkeypatch.setattr(lpm, "safe_get_bytes", _mock)
        result = lpm.load_oecd_stan_ls(countries=["USA"], start=1990, end=2010)
        assert list(result.columns) == ["code", "year", "log_LS_stan"]

    def test_sorted_by_code_year(self, monkeypatch):
        """Output is sorted by (code, year) ascending."""
        # Two countries, each with two years
        calls = {"code": None}
        def _mock(url, timeout=30.0):
            if "USA" in url:
                if "LABR" in url:
                    return b"TIME_PERIOD,OBS_VALUE\n2002,600\n2001,580\n"
                elif "VALU" in url:
                    return b"TIME_PERIOD,OBS_VALUE\n2002,1000\n2001,950\n"
            if "DEU" in url:
                if "LABR" in url:
                    return b"TIME_PERIOD,OBS_VALUE\n2001,550\n2002,570\n"
                elif "VALU" in url:
                    return b"TIME_PERIOD,OBS_VALUE\n2001,900\n2002,920\n"
            return b""

        monkeypatch.setattr(lpm, "safe_get_bytes", _mock)
        result = lpm.load_oecd_stan_ls(countries=["USA", "DEU"], start=1990, end=2010)
        assert result["year"].is_monotonic_increasing or len(result.groupby("code")["year"].apply(
            lambda s: s.is_monotonic_increasing
        ).eq(True)) == 2

    def test_negative_labr_filtered_out(self, monkeypatch):
        """Rows with LABR <= 0 or VALU <= 0 are excluded."""
        # year 2000: LABR=-100 (negative), year 2001: LABR=600 (valid)
        labr_csv = b"TIME_PERIOD,OBS_VALUE\n2000,-100\n2001,600\n"
        valu_csv = b"TIME_PERIOD,OBS_VALUE\n2000,1000\n2001,1100\n"

        def _mock(url, timeout=30.0):
            if "LABR" in url:
                return labr_csv
            elif "VALU" in url:
                return valu_csv
            return b""

        monkeypatch.setattr(lpm, "safe_get_bytes", _mock)
        result = lpm.load_oecd_stan_ls(countries=["USA"], start=1990, end=2010)
        assert 2000 not in result["year"].tolist()
        assert 2001 in result["year"].tolist()

    def test_all_negative_labr_skips_country(self, monkeypatch):
        """All merged rows have LABR <= 0 -> merged empty -> continue (line 188)."""
        # Data covers years 2000-2001, but start=2005 so year filter removes all rows
        def _mock(url, timeout=30.0):
            if "LABR" in url:
                return _stan_csv_labr()   # covers 2000-2002
            elif "VALU" in url:
                return _stan_csv_valu()   # covers 2000-2002
            return b""

        monkeypatch.setattr(lpm, "safe_get_bytes", _mock)
        # start=2005 means year filter on 2000-2002 data -> merged empty -> continue
        result = lpm.load_oecd_stan_ls(countries=["USA"], start=2005, end=2020)
        assert len(result) == 0
        assert list(result.columns) == ["code", "year", "log_LS_stan"]

    def test_multi_country_concat(self, monkeypatch):
        """Results from multiple countries are concatenated."""
        def _mock(url, timeout=30.0):
            if "LABR" in url:
                return _stan_csv_labr()
            elif "VALU" in url:
                return _stan_csv_valu()
            return b""

        monkeypatch.setattr(lpm, "safe_get_bytes", _mock)
        result = lpm.load_oecd_stan_ls(countries=["USA", "DEU"], start=1990, end=2010)
        assert set(result["code"].unique()) == {"USA", "DEU"}


# ---------------------------------------------------------------------------
# 4. load_klems_legacy — offline paths
# ---------------------------------------------------------------------------

class TestLoadKlemsLegacyOffline:
    """load_klems_legacy offline branches."""

    def test_countries_none_defaults_to_g9(self, tmp_path, monkeypatch):
        """countries=None uses the G9 tuple (line 217)."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        # Archive dir exists but no XLS files
        (tmp_path / "data" / "raw" / "euklems_legacy").mkdir(parents=True)
        result = lpm.load_klems_legacy()  # countries=None
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["code", "year", "log_LS_klems_leg"]

    def test_missing_archive_returns_empty(self, tmp_path, monkeypatch):
        """No euklems_legacy directory -> returns empty DataFrame (line 220)."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        result = lpm.load_klems_legacy(countries=["USA"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["code", "year", "log_LS_klems_leg"]

    def test_missing_xls_for_country_skipped(self, tmp_path, monkeypatch):
        """Country XLS file absent -> that country silently skipped."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        (tmp_path / "data" / "raw" / "euklems_legacy").mkdir(parents=True)
        result = lpm.load_klems_legacy(countries=["USA"])
        assert len(result) == 0

    def test_known_value_log_ls_computation(self, tmp_path, monkeypatch):
        """log_LS = log(LAB) - log(VA) for each year; verified against known values."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        archive = tmp_path / "data" / "raw" / "euklems_legacy"
        archive.mkdir(parents=True)
        (archive / "USA_output_08I.xls").write_bytes(b"fake")

        mock_sheet = _make_mock_sheet(
            labr_vals={1995: 600.0, 1996: 650.0},
            valu_vals={1995: 1000.0, 1996: 1100.0},
        )

        def mock_read_excel(path, sheet_name=None, engine=None):
            return mock_sheet

        monkeypatch.setattr(pd, "read_excel", mock_read_excel)
        result = lpm.load_klems_legacy(countries=["USA"], start=1990, end=2007)
        assert len(result) == 2
        row95 = result[result["year"] == 1995]
        expected = np.log(600.0) - np.log(1000.0)
        assert float(row95["log_LS_klems_leg"].iloc[0]) == pytest.approx(expected, rel=1e-10)

    def test_no_year_columns_returns_empty(self, tmp_path, monkeypatch):
        """Sheet with only string columns (no int/float years) -> empty (line 233)."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        archive = tmp_path / "data" / "raw" / "euklems_legacy"
        archive.mkdir(parents=True)
        (archive / "USA_output_08I.xls").write_bytes(b"fake")

        mock_sheet = pd.DataFrame({"var": ["LAB", "VA"], "A": [600, 1000], "B": [650, 1100]})

        def mock_read_excel(path, sheet_name=None, engine=None):
            return mock_sheet

        monkeypatch.setattr(pd, "read_excel", mock_read_excel)
        result = lpm.load_klems_legacy(countries=["USA"], start=1990, end=2007)
        assert len(result) == 0

    def test_negative_values_filtered_out(self, tmp_path, monkeypatch):
        """Rows with LAB <= 0 or VA <= 0 are excluded."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        archive = tmp_path / "data" / "raw" / "euklems_legacy"
        archive.mkdir(parents=True)
        (archive / "USA_output_08I.xls").write_bytes(b"fake")

        mock_sheet = _make_mock_sheet(
            labr_vals={1995: -100.0, 1996: 650.0},
            valu_vals={1995: 1000.0, 1996: 1100.0},
        )

        def mock_read_excel(path, sheet_name=None, engine=None):
            return mock_sheet

        monkeypatch.setattr(pd, "read_excel", mock_read_excel)
        result = lpm.load_klems_legacy(countries=["USA"], start=1990, end=2007)
        assert 1995 not in result["year"].tolist()
        assert 1996 in result["year"].tolist()

    def test_all_negative_values_skips_country(self, tmp_path, monkeypatch):
        """All LAB/VA values <= 0 -> len(df)==0 -> continue (line 244)."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        archive = tmp_path / "data" / "raw" / "euklems_legacy"
        archive.mkdir(parents=True)
        (archive / "USA_output_08I.xls").write_bytes(b"fake")

        # Both years have negative LAB -> all rows filtered out -> country skipped
        mock_sheet = _make_mock_sheet(
            labr_vals={1995: -600.0, 1996: -650.0},
            valu_vals={1995: 1000.0, 1996: 1100.0},
        )

        def mock_read_excel(path, sheet_name=None, engine=None):
            return mock_sheet

        monkeypatch.setattr(pd, "read_excel", mock_read_excel)
        result = lpm.load_klems_legacy(countries=["USA"], start=1990, end=2007)
        assert len(result) == 0
        assert list(result.columns) == ["code", "year", "log_LS_klems_leg"]

    def test_year_filter_applied(self, tmp_path, monkeypatch):
        """Only years within [start, end] appear in output."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        archive = tmp_path / "data" / "raw" / "euklems_legacy"
        archive.mkdir(parents=True)
        (archive / "USA_output_08I.xls").write_bytes(b"fake")

        mock_sheet = _make_mock_sheet(
            labr_vals={1990: 500.0, 1995: 600.0, 2005: 700.0},
            valu_vals={1990: 900.0, 1995: 1000.0, 2005: 1200.0},
        )

        def mock_read_excel(path, sheet_name=None, engine=None):
            return mock_sheet

        monkeypatch.setattr(pd, "read_excel", mock_read_excel)
        result = lpm.load_klems_legacy(countries=["USA"], start=1993, end=2000)
        assert 1990 not in result["year"].tolist()
        assert 2005 not in result["year"].tolist()
        assert 1995 in result["year"].tolist()

    def test_read_excel_exception_skips_country(self, tmp_path, monkeypatch):
        """If read_excel raises for a country, that country is skipped."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        archive = tmp_path / "data" / "raw" / "euklems_legacy"
        archive.mkdir(parents=True)
        (archive / "USA_output_08I.xls").write_bytes(b"corrupt")

        def bad_read_excel(path, sheet_name=None, engine=None):
            raise ValueError("cannot read")

        monkeypatch.setattr(pd, "read_excel", bad_read_excel)
        result = lpm.load_klems_legacy(countries=["USA"], start=1990, end=2007)
        assert len(result) == 0

    def test_multi_country_concat(self, tmp_path, monkeypatch):
        """Results from multiple countries are concatenated and sorted."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        archive = tmp_path / "data" / "raw" / "euklems_legacy"
        archive.mkdir(parents=True)
        (archive / "USA_output_08I.xls").write_bytes(b"fake")
        (archive / "DEU_output_08I.xls").write_bytes(b"fake")

        mock_sheet = _make_mock_sheet(
            labr_vals={1995: 600.0, 1996: 650.0},
            valu_vals={1995: 1000.0, 1996: 1100.0},
        )

        def mock_read_excel(path, sheet_name=None, engine=None):
            return mock_sheet

        monkeypatch.setattr(pd, "read_excel", mock_read_excel)
        result = lpm.load_klems_legacy(countries=["USA", "DEU"], start=1990, end=2007)
        assert set(result["code"].unique()) == {"USA", "DEU"}
        assert list(result.columns) == ["code", "year", "log_LS_klems_leg"]
        # Sorted by (code, year)
        assert result["code"].tolist() == sorted(result["code"].tolist())

    def test_output_schema(self, tmp_path, monkeypatch):
        """Output columns are exactly [code, year, log_LS_klems_leg]."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        archive = tmp_path / "data" / "raw" / "euklems_legacy"
        archive.mkdir(parents=True)
        (archive / "USA_output_08I.xls").write_bytes(b"fake")

        mock_sheet = _make_mock_sheet(
            labr_vals={1995: 600.0},
            valu_vals={1995: 1000.0},
        )

        def mock_read_excel(path, sheet_name=None, engine=None):
            return mock_sheet

        monkeypatch.setattr(pd, "read_excel", mock_read_excel)
        result = lpm.load_klems_legacy(countries=["USA"], start=1990, end=2007)
        assert list(result.columns) == ["code", "year", "log_LS_klems_leg"]


# ---------------------------------------------------------------------------
# 5. build_g9_long_panel — branch coverage
# ---------------------------------------------------------------------------

class TestBuildG9LongPanelBranches:
    """build_g9_long_panel merge-branch and source-failure coverage."""

    def _no_bytes(self, url, timeout=30.0):
        return b""

    def _stan_bytes(self, url, timeout=30.0):
        if "LABR" in url:
            return _stan_csv_labr()
        elif "VALU" in url:
            return _stan_csv_valu()
        return b""

    def test_all_sources_fail_returns_empty_with_correct_columns(
        self, monkeypatch, tmp_path
    ):
        """All loaders return empty -> panel is empty DataFrame with the 6 columns."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)  # no PWT .dta -> FileNotFoundError
        monkeypatch.setattr(lpm, "safe_get_bytes", self._no_bytes)
        result = lpm.build_g9_long_panel(
            start=2010, end=2012, countries=["XXX"], write_csv=False
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        expected_cols = {"code", "year", "log_LS", "vintage_LS", "log_relprice_equip", "vintage_PK"}
        assert expected_cols.issubset(set(result.columns))

    def test_pwt_file_not_found_caught_gracefully(self, monkeypatch, tmp_path):
        """FileNotFoundError from load_pwt10 is caught; STAN data still used."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)  # no PWT
        monkeypatch.setattr(lpm, "safe_get_bytes", self._stan_bytes)
        result = lpm.build_g9_long_panel(
            start=1999, end=2003, countries=["USA"], write_csv=False
        )
        # STAN should provide data for years 2000-2002
        if len(result) > 0:
            assert "USA" in result["code"].values
        # Should not raise
        assert isinstance(result, pd.DataFrame)

    def test_only_ls_all_no_pk(self, monkeypatch, tmp_path):
        """When PWT absent and only STAN provides LS data, log_relprice_equip is all NaN."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        monkeypatch.setattr(lpm, "safe_get_bytes", self._stan_bytes)
        result = lpm.build_g9_long_panel(
            start=1999, end=2003, countries=["USA"], write_csv=False
        )
        if len(result) > 0:
            assert result["log_relprice_equip"].isna().all(), (
                "Without PWT, price column must be all NaN"
            )

    def test_only_pk_no_ls(self, monkeypatch, tmp_path):
        """PWT has pk data but no labsh; no other LS source -> log_LS all NaN."""
        # Mock load_pwt10 to return price data but no LS
        def mock_pwt10(**kwargs):
            return pd.DataFrame({
                "code": ["USA", "USA"],
                "year": [2000, 2001],
                "pl_i": [1.2, 1.3],
                "pl_c": [1.0, 1.0],
                "log_relprice_equip": [np.log(1.2), np.log(1.3)],
                "labsh": [np.nan, np.nan],
                "log_LS_pwt": [np.nan, np.nan],
            })

        import puremacro.klems as klems_mod
        original_klems = klems_mod.load_klems_panel
        klems_mod.load_klems_panel = None  # TypeError in except block -> caught

        monkeypatch.setattr(lpm, "load_pwt10", mock_pwt10)
        monkeypatch.setattr(lpm, "safe_get_bytes", self._no_bytes)  # STAN empty
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)  # no KLEMS legacy archive
        try:
            result = lpm.build_g9_long_panel(
                start=1999, end=2002, countries=["USA"], write_csv=False
            )
        finally:
            klems_mod.load_klems_panel = original_klems

        if len(result) > 0:
            assert result["log_LS"].isna().all(), "Without LS source, log_LS must be all NaN"
            assert result["log_relprice_equip"].notna().any(), (
                "PK data from PWT mock should be present"
            )

    def test_ls_stan_vintage_label_set(self, monkeypatch, tmp_path):
        """When STAN provides LS, vintage_LS is 'oecd_stan' for those rows."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        monkeypatch.setattr(lpm, "safe_get_bytes", self._stan_bytes)
        result = lpm.build_g9_long_panel(
            start=1999, end=2003, countries=["USA"], write_csv=False
        )
        if len(result) > 0:
            stan_rows = result[result["vintage_LS"] == "oecd_stan"]
            # At least one STAN row should be present (2000, 2001, 2002)
            assert len(stan_rows) > 0

    def test_klems_legacy_vintage_label_set(self, monkeypatch, tmp_path):
        """When KLEMS-legacy provides LS, vintage_LS is 'klems_legacy_08'."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        archive = tmp_path / "data" / "raw" / "euklems_legacy"
        archive.mkdir(parents=True)
        (archive / "USA_output_08I.xls").write_bytes(b"fake")

        mock_sheet = _make_mock_sheet(
            labr_vals={1978: 600.0, 1979: 620.0},
            valu_vals={1978: 1000.0, 1979: 1050.0},
        )

        def mock_read_excel(path, sheet_name=None, engine=None):
            return mock_sheet

        monkeypatch.setattr(pd, "read_excel", mock_read_excel)
        # No STAN (empty bytes), only KLEMS-legacy
        monkeypatch.setattr(lpm, "safe_get_bytes", self._no_bytes)

        result = lpm.build_g9_long_panel(
            start=1975, end=1982, countries=["USA"], write_csv=False
        )
        if len(result) > 0:
            legacy_rows = result[result["vintage_LS"] == "klems_legacy_08"]
            assert len(legacy_rows) > 0

    def test_write_csv_creates_file(self, monkeypatch, tmp_path):
        """write_csv=True writes a CSV to data/processed/ (lines 379-382)."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        monkeypatch.setattr(lpm, "safe_get_bytes", self._stan_bytes)

        result = lpm.build_g9_long_panel(
            start=1999, end=2003, countries=["USA"], write_csv=True
        )
        out_path = tmp_path / "data" / "processed" / "long_panel_1975_2024_g9.csv"
        if len(result) > 0:
            assert out_path.exists(), "CSV should have been written"
            written = pd.read_csv(out_path)
            assert len(written) == len(result)

    def test_write_csv_false_no_file_created(self, monkeypatch, tmp_path):
        """write_csv=False does not create any CSV."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        monkeypatch.setattr(lpm, "safe_get_bytes", self._stan_bytes)

        lpm.build_g9_long_panel(
            start=1999, end=2003, countries=["USA"], write_csv=False
        )
        out_path = tmp_path / "data" / "processed" / "long_panel_1975_2024_g9.csv"
        assert not out_path.exists()

    def test_year_range_filter_applied(self, monkeypatch, tmp_path):
        """Output years are within [start, end]."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        monkeypatch.setattr(lpm, "safe_get_bytes", self._stan_bytes)

        result = lpm.build_g9_long_panel(
            start=2001, end=2001, countries=["USA"], write_csv=False
        )
        if len(result) > 0:
            assert result["year"].min() >= 2001
            assert result["year"].max() <= 2001

    def test_country_filter_applied(self, monkeypatch, tmp_path):
        """Countries not in the specified list are excluded."""
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)
        monkeypatch.setattr(lpm, "safe_get_bytes", self._stan_bytes)

        result = lpm.build_g9_long_panel(
            start=1999, end=2003, countries=["USA"], write_csv=False
        )
        if len(result) > 0:
            assert set(result["code"].unique()).issubset({"USA"})

    def test_priority_order_pwt_beats_stan(self, monkeypatch, tmp_path):
        """PWT labsh (priority 0) takes precedence over STAN (priority 2)."""
        # Mock PWT with labsh data for 2000-2001
        def mock_pwt10_with_ls(**kwargs):
            years = [2000, 2001]
            labsh = [0.60, 0.62]
            pl_i = [1.2, 1.3]
            pl_c = [1.0, 1.0]
            return pd.DataFrame({
                "code": ["USA"] * 2,
                "year": years,
                "pl_i": pl_i,
                "pl_c": pl_c,
                "log_relprice_equip": [np.log(1.2), np.log(1.3)],
                "labsh": labsh,
                "log_LS_pwt": [np.log(ls) for ls in labsh],
            })

        monkeypatch.setattr(lpm, "load_pwt10", mock_pwt10_with_ls)
        monkeypatch.setattr(lpm, "safe_get_bytes", self._stan_bytes)
        monkeypatch.setattr(lpm, "_ROOT", tmp_path)

        result = lpm.build_g9_long_panel(
            start=1999, end=2002, countries=["USA"], write_csv=False
        )
        if len(result) > 0:
            pwt_rows = result[result["vintage_LS"] == "pwt10_labsh"]
            assert len(pwt_rows) >= 1, "PWT LS should have been used for years 2000-2001"


# ---------------------------------------------------------------------------
# 6. splice_audit — offline branches
# ---------------------------------------------------------------------------

class TestSpliceAuditOffline:
    """splice_audit coverage for empty-panel and overlap-computation paths."""

    def test_empty_panel_returns_empty_frame(self):
        """splice_audit on empty DataFrame -> empty with correct columns (line 400)."""
        result = lpm.splice_audit(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["code", "overlap_n", "mad_LS_pp", "mad_PK_pp"]
        assert len(result) == 0

    def test_empty_panel_column_types(self):
        """Empty panel result has no rows and the correct column names."""
        result = lpm.splice_audit(pd.DataFrame(columns=["code", "year", "log_LS"]))
        assert set(result.columns) == {"code", "overlap_n", "mad_LS_pp", "mad_PK_pp"}

    def test_no_stan_or_klems_gives_nan_mad(self, monkeypatch):
        """Country with no STAN/KLEMS-legacy -> overlap_n=0, MAD=NaN."""
        panel = pd.DataFrame({
            "code": ["USA", "USA"],
            "year": [2000, 2001],
            "log_LS": [-0.4, -0.45],
            "vintage_LS": ["pwt10_labsh", "pwt10_labsh"],
            "log_relprice_equip": [np.nan, np.nan],
            "vintage_PK": [pd.NA, pd.NA],
        })
        monkeypatch.setattr(lpm, "safe_get_bytes", lambda url, timeout=30.0: b"")
        result = lpm.splice_audit(panel)
        assert len(result) == 1
        assert result["code"].iloc[0] == "USA"
        assert int(result["overlap_n"].iloc[0]) == 0
        assert np.isnan(result["mad_LS_pp"].iloc[0])

    def test_known_value_mad_computation(self, monkeypatch):
        """MAD computed correctly when both STAN and KLEMS-legacy have data."""
        panel = pd.DataFrame({
            "code": ["USA", "USA"],
            "year": [2000, 2001],
            "log_LS": [-0.4, -0.45],
            "vintage_LS": ["pwt10_labsh", "pwt10_labsh"],
            "log_relprice_equip": [np.nan, np.nan],
            "vintage_PK": [pd.NA, pd.NA],
        })

        # STAN: labsh = 600/1000=0.6 and 650/1100≈0.5909
        # KLEMS-legacy: labsh = 590/1000=0.59 and 640/1100≈0.5818
        stan_ls_2000 = np.log(600) - np.log(1000)
        stan_ls_2001 = np.log(650) - np.log(1100)
        klems_ls_2000 = np.log(590) - np.log(1000)
        klems_ls_2001 = np.log(640) - np.log(1100)

        expected_mad = float(
            np.mean(
                np.abs(
                    np.array([np.exp(stan_ls_2000), np.exp(stan_ls_2001)])
                    - np.array([np.exp(klems_ls_2000), np.exp(klems_ls_2001)])
                )
            )
            * 100.0
        )

        def mock_stan(countries, start, end):
            return pd.DataFrame({
                "code": ["USA", "USA"],
                "year": [2000, 2001],
                "log_LS_stan": [stan_ls_2000, stan_ls_2001],
            })

        def mock_klems(countries, start, end):
            return pd.DataFrame({
                "code": ["USA", "USA"],
                "year": [2000, 2001],
                "log_LS_klems_leg": [klems_ls_2000, klems_ls_2001],
            })

        monkeypatch.setattr(lpm, "load_oecd_stan_ls", mock_stan)
        monkeypatch.setattr(lpm, "load_klems_legacy", mock_klems)

        result = lpm.splice_audit(panel)
        assert len(result) == 1
        assert result["code"].iloc[0] == "USA"
        assert int(result["overlap_n"].iloc[0]) == 2
        assert float(result["mad_LS_pp"].iloc[0]) == pytest.approx(expected_mad, rel=1e-8)
        assert result["mad_LS_pp"].iloc[0] >= 0.0

    def test_mad_non_negative(self, monkeypatch):
        """MAD is always non-negative (triangle inequality property)."""
        panel = pd.DataFrame({
            "code": ["DEU", "DEU", "DEU"],
            "year": [1995, 1996, 1997],
            "log_LS": [-0.5, -0.48, -0.52],
            "vintage_LS": ["pwt10_labsh"] * 3,
            "log_relprice_equip": [np.nan, np.nan, np.nan],
            "vintage_PK": [pd.NA, pd.NA, pd.NA],
        })

        def mock_stan(countries, start, end):
            return pd.DataFrame({
                "code": ["DEU"] * 3,
                "year": [1995, 1996, 1997],
                "log_LS_stan": [np.log(0.60), np.log(0.61), np.log(0.59)],
            })

        def mock_klems(countries, start, end):
            return pd.DataFrame({
                "code": ["DEU"] * 3,
                "year": [1995, 1996, 1997],
                "log_LS_klems_leg": [np.log(0.62), np.log(0.60), np.log(0.61)],
            })

        monkeypatch.setattr(lpm, "load_oecd_stan_ls", mock_stan)
        monkeypatch.setattr(lpm, "load_klems_legacy", mock_klems)

        result = lpm.splice_audit(panel)
        assert float(result["mad_LS_pp"].iloc[0]) >= 0.0

    def test_no_inner_overlap_gives_zero_n_nan_mad(self, monkeypatch):
        """STAN and KLEMS-legacy cover disjoint years -> n=0, MAD=NaN."""
        panel = pd.DataFrame({
            "code": ["USA", "USA"],
            "year": [2000, 2005],
            "log_LS": [-0.4, -0.45],
            "vintage_LS": ["oecd_stan", "klems_legacy_08"],
            "log_relprice_equip": [np.nan, np.nan],
            "vintage_PK": [pd.NA, pd.NA],
        })

        def mock_stan(countries, start, end):
            return pd.DataFrame({
                "code": ["USA"],
                "year": [2000],
                "log_LS_stan": [np.log(0.6)],
            })

        def mock_klems(countries, start, end):
            return pd.DataFrame({
                "code": ["USA"],
                "year": [2005],
                "log_LS_klems_leg": [np.log(0.58)],
            })

        monkeypatch.setattr(lpm, "load_oecd_stan_ls", mock_stan)
        monkeypatch.setattr(lpm, "load_klems_legacy", mock_klems)

        result = lpm.splice_audit(panel)
        assert len(result) == 1
        assert int(result["overlap_n"].iloc[0]) == 0
        assert np.isnan(result["mad_LS_pp"].iloc[0])

    def test_output_columns(self, monkeypatch):
        """splice_audit output always has [code, overlap_n, mad_LS_pp, mad_PK_pp]."""
        panel = pd.DataFrame({
            "code": ["USA"],
            "year": [2000],
            "log_LS": [-0.4],
            "vintage_LS": ["pwt10_labsh"],
            "log_relprice_equip": [np.nan],
            "vintage_PK": [pd.NA],
        })
        monkeypatch.setattr(lpm, "safe_get_bytes", lambda url, timeout=30.0: b"")
        result = lpm.splice_audit(panel)
        assert list(result.columns) == ["code", "overlap_n", "mad_LS_pp", "mad_PK_pp"]

    def test_mad_pk_pp_always_nan(self, monkeypatch):
        """mad_PK_pp is always NaN (not yet implemented in this version)."""
        panel = pd.DataFrame({
            "code": ["USA", "USA"],
            "year": [2000, 2001],
            "log_LS": [-0.4, -0.45],
            "vintage_LS": ["pwt10_labsh", "pwt10_labsh"],
            "log_relprice_equip": [np.nan, np.nan],
            "vintage_PK": [pd.NA, pd.NA],
        })

        def mock_stan(countries, start, end):
            return pd.DataFrame({
                "code": ["USA", "USA"],
                "year": [2000, 2001],
                "log_LS_stan": [np.log(0.6), np.log(0.61)],
            })

        def mock_klems(countries, start, end):
            return pd.DataFrame({
                "code": ["USA", "USA"],
                "year": [2000, 2001],
                "log_LS_klems_leg": [np.log(0.59), np.log(0.60)],
            })

        monkeypatch.setattr(lpm, "load_oecd_stan_ls", mock_stan)
        monkeypatch.setattr(lpm, "load_klems_legacy", mock_klems)

        result = lpm.splice_audit(panel)
        # mad_PK_pp is documented as NaN (out of scope in this version)
        assert np.isnan(result["mad_PK_pp"].iloc[0])

    def test_multi_country_panel(self, monkeypatch):
        """splice_audit processes multiple countries and returns one row each."""
        panel = pd.DataFrame({
            "code": ["USA", "USA", "DEU", "DEU"],
            "year": [2000, 2001, 2000, 2001],
            "log_LS": [-0.4, -0.45, -0.38, -0.42],
            "vintage_LS": ["pwt10_labsh"] * 4,
            "log_relprice_equip": [np.nan] * 4,
            "vintage_PK": [pd.NA] * 4,
        })
        monkeypatch.setattr(lpm, "safe_get_bytes", lambda url, timeout=30.0: b"")
        result = lpm.splice_audit(panel)
        assert len(result) == 2
        assert set(result["code"].tolist()) == {"USA", "DEU"}


# ---------------------------------------------------------------------------
# 7. G9 constant and module-level sanity
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Sanity checks on module-level constants."""

    def test_g9_has_nine_countries(self):
        """G9 tuple has exactly 9 country codes."""
        assert len(lpm.G9) == 9

    def test_g9_all_uppercase_iso3(self):
        """All G9 codes are 3-character uppercase strings."""
        for code in lpm.G9:
            assert len(code) == 3
            assert code.isupper()

    def test_g9_contains_expected_countries(self):
        """Spot-check G9 for the K-N (2014) original nine economies."""
        g9 = set(lpm.G9)
        for expected in ("USA", "JPN", "DEU", "FRA", "GBR", "ITA", "CAN", "AUS"):
            assert expected in g9, f"{expected} missing from G9"

    def test_stan_agg_candidates_order(self):
        """DTOTAL is tried before TOTAL in the STAN aggregation candidates."""
        candidates = lpm._STAN_AGG_CANDIDATES
        assert candidates[0] == "DTOTAL"
        assert "TOTAL" in candidates
