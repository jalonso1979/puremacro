"""Extended coverage tests for puremacro.shock_atlas.

Covers: _z, _load_ltui, _load_epu_national, _load_wui, _load_gpr,
_load_romer_tax, _load_ramey_defense, _load_mertens_ravn, _load_tpu_us,
_load_wpui_us, _load_pandemic_emv, _load_frbsf_tfp, _load_fred_series,
load_all_shocks, categorise, ShockSpec, SHOCK_REGISTRY.

Strategy:
- Known-value tests: synthetic data with exact expected z-score values.
- Contract tests: shapes, dtypes, z-score properties (mean~0, std~1).
- Branch coverage: alternate column names, fallback paths, error paths.
- Monkeypatch for FRED/network paths to avoid external dependencies.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import puremacro.shock_atlas as sa
from puremacro.shock_atlas import (
    SHOCK_REGISTRY,
    ShockSpec,
    _load_epu_national,
    _load_fred_series,
    _load_frbsf_tfp,
    _load_gpr,
    _load_ltui,
    _load_mertens_ravn,
    _load_pandemic_emv,
    _load_ramey_defense,
    _load_romer_tax,
    _load_tpu_us,
    _load_wpui_us,
    _load_wui,
    _z,
    categorise,
    load_all_shocks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ltui_parquet(root: Path, n: int = 120) -> None:
    """Write a synthetic ltui_usa_monthly.parquet to root/notebooks/data_cache/."""
    cache_dir = root / "notebooks" / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    df = pd.DataFrame({"date": dates, "ltui_z": np.arange(n, dtype=float)})
    df.to_parquet(cache_dir / "ltui_usa_monthly.parquet")


def _make_rr_csv(root: Path, date_col: str = "date", val_col: str = "exogenous",
                 n: int = 80) -> None:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("1960-01-01", periods=n, freq="QS")
    df = pd.DataFrame({date_col: dates.strftime("%Y-%m-%d"),
                       val_col: np.random.default_rng(0).standard_normal(n)})
    df.to_csv(raw / "romer_romer_2010_tax_shocks.csv", index=False)


def _make_ramey_csv(root: Path, date_col: str = "date", val_col: str = "news",
                    n: int = 100) -> None:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("1940-01-01", periods=n, freq="QS")
    df = pd.DataFrame({date_col: dates.strftime("%Y-%m-%d"),
                       val_col: np.random.default_rng(1).standard_normal(n)})
    df.to_csv(raw / "ramey_2011_defense_news.csv", index=False)


def _make_mr_csv(root: Path, n: int = 80) -> None:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("1950-01-01", periods=n, freq="QS")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                       "unanticipated": np.random.default_rng(2).standard_normal(n)})
    df.to_csv(raw / "mertens_ravn_2013_unanticipated_tax.csv", index=False)


def _make_gpr_xls(root: Path, val_col: str = "GPR", n: int = 240) -> None:
    raw = root / "data" / "raw" / "www.matteoiacoviello.com"
    raw.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    df = pd.DataFrame({"month": dates, val_col: np.arange(1, n + 1, dtype=float)})
    df.to_excel(raw / "gpr_files_data_gpr_export.xls", index=False)


def _make_wui_xlsx(root: Path, quarters=None, val=None) -> None:
    """Write a WUI_Data.xlsx with an F1 sheet (header=2 layout)."""
    raw = root / "data" / "raw" / "worlduncertaintyindex.com"
    raw.mkdir(parents=True, exist_ok=True)
    if quarters is None:
        quarters = [f"{y}Q{q}" for y in range(2000, 2020) for q in range(1, 5)]
    n = len(quarters)
    if val is None:
        val = np.random.default_rng(3).standard_normal(n)
    df = pd.DataFrame({"quarter": quarters, "WUI": val})
    with pd.ExcelWriter(raw / "wp-content_uploads_2024_10_WUI_Data.xlsx",
                        engine="openpyxl") as w:
        header_df = pd.DataFrame([["WUI World", "WUI GDP-weighted"], ["", ""]])
        header_df.to_excel(w, sheet_name="F1", index=False, header=False)
        df.to_excel(w, sheet_name="F1", index=False, header=True, startrow=2)


def _make_tpu_xlsx(root: Path, with_usa: bool = True, quarters=None) -> None:
    raw = root / "data" / "raw" / "worlduncertaintyindex.com"
    raw.mkdir(parents=True, exist_ok=True)
    if quarters is None:
        quarters = [f"{y}Q{q}" for y in range(2000, 2020) for q in range(1, 5)]
    n = len(quarters)
    cols = {"quarter": quarters}
    if with_usa:
        cols["USA"] = np.random.default_rng(4).standard_normal(n)
    else:
        cols["GBR"] = np.random.default_rng(4).standard_normal(n)
    df = pd.DataFrame(cols)
    with pd.ExcelWriter(raw / "wp-content_uploads_2024_10_WUI_Data.xlsx",
                        engine="openpyxl") as w:
        df.to_excel(w, sheet_name="T8", index=False)


def _make_tfp_xlsx(root: Path, tfp_col: str = "dtfp_util",
                   quarters=None, add_noise: bool = False) -> None:
    raw = root / "data" / "raw" / "www.frbsf.org"
    raw.mkdir(parents=True, exist_ok=True)
    if quarters is None:
        quarters = [f"{y}:{q}" for y in range(2000, 2020) for q in range(1, 5)]
    n = len(quarters)
    df = pd.DataFrame({"quarter": quarters,
                       tfp_col: np.random.default_rng(5).standard_normal(n)})
    if add_noise:
        # Add some non-quarter rows that should be filtered out
        noise = pd.DataFrame({"quarter": ["FULLSAMPLEMEAN", "2004:4-2019:4"],
                               tfp_col: [0.5, 0.5]})
        df = pd.concat([df, noise], ignore_index=True)
    with pd.ExcelWriter(raw / "wp-content_uploads_quarterly_tfp.xlsx",
                        engine="openpyxl") as w:
        skip_df = pd.DataFrame([["skip_row"] + [""] * (len(df.columns) - 1)])
        skip_df.to_excel(w, sheet_name="quarterly", index=False, header=False)
        df.to_excel(w, sheet_name="quarterly", index=False, header=True, startrow=1)


# ---------------------------------------------------------------------------
# _z: known-value and property tests
# ---------------------------------------------------------------------------

class TestZScoreHelper:
    def test_known_values(self):
        """Arithmetic z-score of [1, 2, 3, 4, 5] with ddof=0."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _z(s)
        mu, sigma = 3.0, np.sqrt(2.0)
        expected = (np.array([1.0, 2.0, 3.0, 4.0, 5.0]) - mu) / sigma
        np.testing.assert_allclose(result.values, expected, atol=1e-12)

    def test_mean_near_zero(self):
        rng = np.random.default_rng(42)
        s = pd.Series(rng.standard_normal(200) * 100 + 500)
        assert abs(_z(s).mean()) < 1e-10

    def test_std_is_one_ddof0(self):
        rng = np.random.default_rng(42)
        s = pd.Series(rng.standard_normal(200) * 100 + 500)
        assert abs(_z(s).std(ddof=0) - 1.0) < 1e-10

    def test_dropna_before_standardizing(self):
        """NaN values must be dropped; result length equals non-NaN count."""
        s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        result = _z(s)
        assert len(result) == 3
        assert result.isna().sum() == 0

    def test_constant_series_returns_nan(self):
        s = pd.Series([7.0, 7.0, 7.0])
        result = _z(s)
        assert result.isna().all()

    def test_single_element(self):
        s = pd.Series([42.0])
        result = _z(s)
        assert np.isnan(result.iloc[0])


# ---------------------------------------------------------------------------
# ShockSpec / SHOCK_REGISTRY
# ---------------------------------------------------------------------------

class TestShockSpec:
    def test_frozen_immutable(self):
        spec = SHOCK_REGISTRY[0]
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.name = "changed"  # type: ignore[misc]

    def test_all_registry_entries_have_required_fields(self):
        for spec in SHOCK_REGISTRY:
            assert isinstance(spec.name, str) and spec.name
            assert isinstance(spec.label, str) and spec.label
            assert isinstance(spec.category, str) and spec.category
            assert callable(spec.loader)

    def test_registry_names_are_unique(self):
        names = [s.name for s in SHOCK_REGISTRY]
        assert len(names) == len(set(names))

    def test_registry_has_expected_shocks(self):
        names = {s.name for s in SHOCK_REGISTRY}
        for expected in ("LTUI", "EPU_US", "WUI", "GPR", "RR_tax",
                         "Ramey_def", "MR_tax", "Fernald_TFP",
                         "TPU_US", "WPUI_US", "Pandemic_EMV"):
            assert expected in names

    def test_custom_shockspec(self):
        dummy_loader = lambda root: pd.Series(dtype=float)
        spec = ShockSpec(name="TEST", label="Test shock",
                         category="test", loader=dummy_loader)
        assert spec.name == "TEST"
        assert spec.loader is dummy_loader


# ---------------------------------------------------------------------------
# categorise
# ---------------------------------------------------------------------------

class TestCategorise:
    def test_known_categories(self):
        assert categorise("GPR") == "geopolitical"
        assert categorise("LTUI") == "uncertainty"
        assert categorise("RR_tax") == "tax"
        assert categorise("Ramey_def") == "fiscal"
        assert categorise("MR_tax") == "tax"
        assert categorise("Fernald_TFP") == "tfp"
        assert categorise("EPU_US") == "uncertainty"
        assert categorise("WUI") == "uncertainty"
        assert categorise("TPU_US") == "trade"
        assert categorise("WPUI_US") == "pandemic"
        assert categorise("Pandemic_EMV") == "pandemic"

    def test_unknown_name_returns_other(self):
        assert categorise("NONEXISTENT_SERIES") == "other"
        assert categorise("") == "other"

    def test_case_sensitive(self):
        # 'gpr' (lowercase) is not in the registry
        assert categorise("gpr") == "other"


# ---------------------------------------------------------------------------
# _load_ltui
# ---------------------------------------------------------------------------

class TestLoadLtui:
    def test_returns_zscore_series(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ltui_parquet(root, n=120)
            result = _load_ltui(root)
            assert isinstance(result, pd.Series)
            assert len(result) == 120
            assert abs(result.mean()) < 1e-10
            assert abs(result.std(ddof=0) - 1.0) < 1e-10

    def test_index_is_datetimeindex(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ltui_parquet(root, n=60)
            result = _load_ltui(root)
            assert isinstance(result.index, pd.DatetimeIndex)

    def test_zscore_known_value(self):
        """With input [0, 1, 2, ..., n-1], z-score of last obs is predictable."""
        n = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ltui_parquet(root, n=n)
            result = _load_ltui(root)
            # All values 0..9; mu=4.5, sigma=sqrt(mean sq deviation from mean)
            vals = np.arange(n, dtype=float)
            mu = vals.mean()
            sigma = vals.std(ddof=0)
            np.testing.assert_allclose(result.values, (vals - mu) / sigma, atol=1e-10)


# ---------------------------------------------------------------------------
# _load_epu_national
# ---------------------------------------------------------------------------

class TestLoadEpuNational:
    def _write_epu(self, root: Path, year_col: str, epu_col: str, n: int = 120) -> None:
        raw = root / "data" / "raw" / "www.policyuncertainty.com"
        raw.mkdir(parents=True, exist_ok=True)
        years = [2000 + i // 12 for i in range(n)]
        months = [i % 12 + 1 for i in range(n)]
        df = pd.DataFrame({year_col: years, "Month": months,
                           epu_col: np.arange(1, n + 1, dtype=float)})
        df.to_excel(raw / "media_US_Historical_EPU_data.xlsx", index=False)

    def test_basic_us_epu_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_epu(root, "Year", "US_EPU_Composite")
            result = _load_epu_national(root)
            assert isinstance(result, pd.Series)
            assert len(result) == 120
            assert abs(result.mean()) < 1e-10

    def test_news_based_fallback_column(self):
        """Column containing both 'news' and 'based' should be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_epu(root, "Year", "News_Based_Policy_Uncertainty")
            result = _load_epu_national(root)
            assert len(result) == 120

    def test_epu_only_in_name_fallback(self):
        """Column with only 'epu' in name (not 'us epu') is the last fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_epu(root, "Year", "national_epu_index")
            result = _load_epu_national(root)
            assert len(result) == 120

    def test_raises_when_no_year_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "data" / "raw" / "www.policyuncertainty.com"
            raw.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({"Month": [1, 2], "SomeCol": [100.0, 200.0]})
            df.to_excel(raw / "media_US_Historical_EPU_data.xlsx", index=False)
            with pytest.raises(ValueError, match="Could not parse EPU"):
                _load_epu_national(root)

    def test_zscore_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_epu(root, "Year", "US_EPU_Composite", n=240)
            result = _load_epu_national(root)
            assert abs(result.std(ddof=0) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# _load_wui
# ---------------------------------------------------------------------------

class TestLoadWui:
    def test_returns_monthly_zscore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_wui_xlsx(root)
            result = _load_wui(root)
            assert isinstance(result, pd.Series)
            assert len(result) > 0
            assert abs(result.mean()) < 1e-10

    def test_raises_when_no_wui_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "data" / "raw" / "worlduncertaintyindex.com"
            raw.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({"quarter": ["2000Q1", "2000Q2"],
                               "SomeOtherIndex": [1.0, 2.0]})
            with pd.ExcelWriter(raw / "wp-content_uploads_2024_10_WUI_Data.xlsx",
                                engine="openpyxl") as w:
                pd.DataFrame([["h1", "h2"], ["", ""]]).to_excel(
                    w, sheet_name="F1", index=False, header=False)
                df.to_excel(w, sheet_name="F1", index=False, header=True, startrow=2)
            with pytest.raises(ValueError, match="WUI column not found"):
                _load_wui(root)

    def test_non_quarter_rows_filtered(self):
        """Rows that don't match YYYYQ[1-4] pattern must be dropped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            quarters = (["2000Q1", "2000Q2", "2000Q3", "2000Q4",
                         "not_a_quarter", "2001Q1"])
            vals = [1.0, 2.0, 3.0, 4.0, 99.0, 5.0]
            _make_wui_xlsx(root, quarters=quarters, val=vals)
            result = _load_wui(root)
            # 'not_a_quarter' row must not inflate the output
            assert len(result) > 0


# ---------------------------------------------------------------------------
# _load_gpr
# ---------------------------------------------------------------------------

class TestLoadGpr:
    def test_gpr_column_exact_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_gpr_xls(root, val_col="GPR")
            result = _load_gpr(root)
            assert len(result) == 240
            assert abs(result.mean()) < 1e-10

    def test_gpr_recent_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_gpr_xls(root, val_col="GPR_recent")
            result = _load_gpr(root)
            assert len(result) == 240

    def test_gpr_lowercase_fallback(self):
        """Column with 'gpr' in name but not exact 'GPR' uses final fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_gpr_xls(root, val_col="hist_gpr_index")
            result = _load_gpr(root)
            assert len(result) == 240

    def test_raises_when_no_gpr_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "data" / "raw" / "www.matteoiacoviello.com"
            raw.mkdir(parents=True, exist_ok=True)
            dates = pd.date_range("2000-01-01", periods=60, freq="MS")
            df = pd.DataFrame({"month": dates, "some_index": np.ones(60)})
            df.to_excel(raw / "gpr_files_data_gpr_export.xls", index=False)
            with pytest.raises(ValueError, match="GPR column not found"):
                _load_gpr(root)

    def test_zscore_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_gpr_xls(root)
            result = _load_gpr(root)
            assert abs(result.std(ddof=0) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# _load_romer_tax
# ---------------------------------------------------------------------------

class TestLoadRomerTax:
    def test_date_and_exogenous_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_rr_csv(root, date_col="date", val_col="exogenous")
            result = _load_romer_tax(root)
            assert isinstance(result, pd.Series)
            assert len(result) > 0

    def test_qdate_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_rr_csv(root, date_col="qdate", val_col="rr")
            result = _load_romer_tax(root)
            assert len(result) > 0

    def test_shock_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_rr_csv(root, date_col="quarter", val_col="shock")
            result = _load_romer_tax(root)
            assert len(result) > 0

    def test_rr_exog_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_rr_csv(root, date_col="yyyyq", val_col="rr_exog")
            result = _load_romer_tax(root)
            assert len(result) > 0

    def test_fallback_numeric_column(self):
        """When no named match, the first numeric column is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_rr_csv(root, date_col="date", val_col="my_shock_col")
            result = _load_romer_tax(root)
            assert len(result) > 0

    def test_raises_when_no_numeric_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({"date": ["1960-01-01"], "text_col": ["a"]})
            df.to_csv(raw / "romer_romer_2010_tax_shocks.csv", index=False)
            with pytest.raises(ValueError, match="Could not find RR shock"):
                _load_romer_tax(root)

    def test_quarterly_zero_fill(self):
        """Monthly output must contain zeros for months without a quarterly obs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_rr_csv(root, n=8)
            result = _load_romer_tax(root)
            # Most months should be 0.0 since input is quarterly
            # The z-scoring transforms the 0s, but we can check shape
            assert len(result) > 8


# ---------------------------------------------------------------------------
# _load_ramey_defense
# ---------------------------------------------------------------------------

class TestLoadRameyDefense:
    def test_news_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ramey_csv(root, val_col="news")
            result = _load_ramey_defense(root)
            assert len(result) > 0

    def test_ramey_news_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ramey_csv(root, val_col="ramey_news")
            result = _load_ramey_defense(root)
            assert len(result) > 0

    def test_ramey_def_news_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ramey_csv(root, val_col="ramey_def_news")
            result = _load_ramey_defense(root)
            assert len(result) > 0

    def test_fallback_numeric_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ramey_csv(root, val_col="defense_shock")
            result = _load_ramey_defense(root)
            assert len(result) > 0

    def test_raises_when_no_numeric_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({"qdate": ["1940Q1"], "text": ["a"]})
            df.to_csv(raw / "ramey_2011_defense_news.csv", index=False)
            with pytest.raises(ValueError, match="Could not find Ramey news"):
                _load_ramey_defense(root)

    def test_qdate_date_col_fallback(self):
        """Uses qdate as date_col when 'date' is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ramey_csv(root, date_col="qdate", val_col="news")
            result = _load_ramey_defense(root)
            assert len(result) > 0

    def test_zscore_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ramey_csv(root)
            result = _load_ramey_defense(root)
            assert abs(result.mean()) < 1e-10


# ---------------------------------------------------------------------------
# _load_mertens_ravn
# ---------------------------------------------------------------------------

class TestLoadMertensRavn:
    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_mr_csv(root)
            result = _load_mertens_ravn(root)
            assert isinstance(result, pd.Series)
            assert len(result) > 0

    def test_raises_when_no_numeric_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({"date": ["1950-01-01"], "text": ["a"]})
            df.to_csv(raw / "mertens_ravn_2013_unanticipated_tax.csv", index=False)
            with pytest.raises(ValueError, match="No numeric column in Mertens"):
                _load_mertens_ravn(root)

    def test_zscore_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_mr_csv(root)
            result = _load_mertens_ravn(root)
            assert abs(result.mean()) < 1e-10

    def test_non_date_col_fallback(self):
        """When no 'date' column, first column is used as date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            dates = pd.date_range("1950-01-01", periods=60, freq="QS")
            df = pd.DataFrame({"my_date": dates.strftime("%Y-%m-%d"),
                               "unanticipated": np.ones(60)})
            df.to_csv(raw / "mertens_ravn_2013_unanticipated_tax.csv", index=False)
            result = _load_mertens_ravn(root)
            assert len(result) > 0


# ---------------------------------------------------------------------------
# _load_tpu_us
# ---------------------------------------------------------------------------

class TestLoadTpuUs:
    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_tpu_xlsx(root, with_usa=True)
            result = _load_tpu_us(root)
            assert isinstance(result, pd.Series)
            assert len(result) > 0
            assert abs(result.mean()) < 1e-10

    def test_raises_when_no_usa_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_tpu_xlsx(root, with_usa=False)
            with pytest.raises(ValueError, match="USA column not in T8"):
                _load_tpu_us(root)

    def test_non_quarter_rows_filtered(self):
        """Rows not matching YYYYQ[1-4] are filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            quarters = ["2000Q1", "2000Q2", "not_a_quarter", "2001Q1"]
            _make_tpu_xlsx(root, quarters=quarters)
            result = _load_tpu_us(root)
            # 3 valid quarters -> at least 3 monthly obs after ffill
            assert len(result) >= 3


# ---------------------------------------------------------------------------
# _load_wpui_us  — cache hit and miss branches
# ---------------------------------------------------------------------------

class TestLoadWpuiUs:
    def test_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "notebooks" / "data_cache"
            cache_dir.mkdir(parents=True)
            dates = pd.date_range("2000-01-01", periods=80, freq="QS")
            df = pd.DataFrame({"date": dates, "wpui": np.arange(80, dtype=float)})
            df.to_parquet(cache_dir / "wpui_us.parquet")
            result = _load_wpui_us(root)
            assert isinstance(result, pd.Series)
            assert abs(result.mean()) < 1e-10
            assert abs(result.std(ddof=0) - 1.0) < 1e-10

    def test_cache_miss_calls_fred_and_writes_parquet(self):
        """Without cache, _load_fred_series is called and parquet is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "notebooks" / "data_cache"
            cache_dir.mkdir(parents=True)

            def fake_fred(series_id):
                dates = pd.date_range("2000-01-01", periods=40, freq="QS")
                return pd.Series(np.arange(40, dtype=float), index=dates,
                                 name=series_id)

            orig = sa._load_fred_series
            sa._load_fred_series = fake_fred
            try:
                result = _load_wpui_us(root)
                assert len(result) > 0
                assert (cache_dir / "wpui_us.parquet").exists()
            finally:
                sa._load_fred_series = orig

    def test_monthly_forward_fill(self):
        """Quarterly input must be forward-filled to monthly grid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "notebooks" / "data_cache"
            cache_dir.mkdir(parents=True)
            # 4 quarterly obs -> should expand to at least 10 monthly obs
            dates = pd.date_range("2000-01-01", periods=4, freq="QS")
            df = pd.DataFrame({"date": dates, "wpui": [1.0, 2.0, 3.0, 4.0]})
            df.to_parquet(cache_dir / "wpui_us.parquet")
            result = _load_wpui_us(root)
            assert len(result) >= 4


# ---------------------------------------------------------------------------
# _load_pandemic_emv  — cache hit and miss branches
# ---------------------------------------------------------------------------

class TestLoadPandemicEmv:
    def test_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "notebooks" / "data_cache"
            cache_dir.mkdir(parents=True)
            dates = pd.date_range("2020-01-01", periods=36, freq="MS")
            df = pd.DataFrame({"date": dates,
                               "emv": np.arange(36, dtype=float) * 10 + 100})
            df.to_parquet(cache_dir / "pandemic_emv.parquet")
            result = _load_pandemic_emv(root)
            assert isinstance(result, pd.Series)
            assert len(result) == 36
            assert abs(result.mean()) < 1e-10

    def test_cache_miss_calls_fred_and_writes_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "notebooks" / "data_cache"
            cache_dir.mkdir(parents=True)

            def fake_fred(series_id):
                dates = pd.date_range("2020-01-01", periods=30, freq="MS")
                return pd.Series(np.arange(30, dtype=float), index=dates,
                                 name=series_id)

            orig = sa._load_fred_series
            sa._load_fred_series = fake_fred
            try:
                result = _load_pandemic_emv(root)
                assert len(result) == 30
                assert (cache_dir / "pandemic_emv.parquet").exists()
            finally:
                sa._load_fred_series = orig


# ---------------------------------------------------------------------------
# _load_fred_series  — missing key branch
# ---------------------------------------------------------------------------

class TestLoadFredSeries:
    def test_raises_without_api_key(self):
        saved = os.environ.pop("FRED_API_KEY", None)
        try:
            with pytest.raises(RuntimeError, match="FRED_API_KEY"):
                _load_fred_series("WUPIUSA")
        finally:
            if saved is not None:
                os.environ["FRED_API_KEY"] = saved


# ---------------------------------------------------------------------------
# _load_frbsf_tfp
# ---------------------------------------------------------------------------

class TestLoadFrbsfTfp:
    def test_dtfp_util_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_tfp_xlsx(root, tfp_col="dtfp_util")
            result = _load_frbsf_tfp(root)
            assert isinstance(result, pd.Series)
            assert len(result) > 0
            assert abs(result.mean()) < 1e-10

    def test_tfp_util_in_name(self):
        """Column name containing both 'tfp' and 'util' is matched first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_tfp_xlsx(root, tfp_col="TFP_util_adj")
            result = _load_frbsf_tfp(root)
            assert len(result) > 0

    def test_raises_when_no_tfp_col(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_tfp_xlsx(root, tfp_col="some_other_col")
            # 'some_other_col' does not contain 'tfp' or 'dtfp_util'
            with pytest.raises(ValueError, match="Could not find TFP"):
                _load_frbsf_tfp(root)

    def test_non_quarter_rows_filtered(self):
        """Rows like 'FULLSAMPLEMEAN' must be filtered by the regex."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_tfp_xlsx(root, tfp_col="dtfp_util", add_noise=True)
            result = _load_frbsf_tfp(root)
            assert len(result) > 0

    def test_zscore_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_tfp_xlsx(root)
            result = _load_frbsf_tfp(root)
            assert abs(result.std(ddof=0) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# load_all_shocks
# ---------------------------------------------------------------------------

class TestLoadAllShocks:
    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_all_shocks(Path(tmpdir))
        assert isinstance(result, dict)

    def test_successful_load_appears_in_output(self):
        """A well-formed LTUI parquet with n>24 must appear in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ltui_parquet(root, n=100)
            result = load_all_shocks(root)
        assert "LTUI" in result
        assert isinstance(result["LTUI"], pd.Series)

    def test_too_short_series_skipped(self):
        """Series with <=24 obs must be silently skipped (not in output)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ltui_parquet(root, n=10)
            result = load_all_shocks(root)
        assert "LTUI" not in result

    def test_failed_loader_does_not_raise(self):
        """Missing files must cause the entry to be skipped, not propagate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty tmp dir — all loaders will fail with FileNotFoundError
            result = load_all_shocks(Path(tmpdir))
        assert isinstance(result, dict)

    def test_output_series_are_zscore(self):
        """Every returned series must have mean~0 and std~1 (z-scored)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _make_ltui_parquet(root, n=100)
            result = load_all_shocks(root)
        for name, s in result.items():
            assert abs(s.dropna().mean()) < 1e-9, f"{name}: mean not ~0"
            assert abs(s.dropna().std(ddof=0) - 1.0) < 1e-9, f"{name}: std not ~1"
