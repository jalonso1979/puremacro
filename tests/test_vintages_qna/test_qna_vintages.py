"""Test suite for puremacro.fetch.qna_vintages and QNAVintagePanel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.fetch.qna_vintages import (
    COUNTRY_METADATA,
    VARIABLE_MAP,
    QNAVintagePanel,
    _resolve_series_id,
    fetch_qna_vintages,
    get_qna_vintage_catalog,
)
from puremacro.vintages import AlfredVintageStore


def test_catalog_coverage():
    """Verify catalog has 45+ countries and all standard QNA variables."""
    cat = get_qna_vintage_catalog()
    assert isinstance(cat, pd.DataFrame)
    assert len(cat) > 300
    expected_cols = {"country", "iso2", "country_name", "region", "variable", "series_id", "description"}
    assert expected_cols.issubset(set(cat.columns))

    unique_countries = cat["country"].unique()
    assert len(unique_countries) >= 45
    assert "USA" in unique_countries
    assert "DEU" in unique_countries
    assert "GBR" in unique_countries
    assert "JPN" in unique_countries
    assert "FRA" in unique_countries
    assert "MEX" in unique_countries
    assert "BRA" in unique_countries

    unique_vars = cat["variable"].unique()
    assert "gdp_real" in unique_vars
    assert "con_real" in unique_vars
    assert "gfcf_real" in unique_vars
    assert "govcon_real" in unique_vars
    assert "exports_real" in unique_vars
    assert "imports_real" in unique_vars
    assert "deflator" in unique_vars


def test_resolve_series_id():
    """Verify series ID resolution for direct US vs OECD template codes."""
    assert _resolve_series_id("USA", "gdp_real") == "GDPC1"
    assert _resolve_series_id("USA", "con_real") == "PCECC96"
    assert _resolve_series_id("USA", "log_gdp_real") == "GDPC1"

    assert _resolve_series_id("DEU", "gdp_real") == "NAEXKP01DEQ652S"
    assert _resolve_series_id("GBR", "gdp_real") == "NAEXKP01GBQ652S"
    assert _resolve_series_id("JPN", "con_real") == "NAEXKP02JPQ652S"
    assert _resolve_series_id("MEX", "gfcf_real") == "NAEXKP04MXQ652S"

    with pytest.raises(ValueError, match="Unknown country code"):
        _resolve_series_id("XYZ123", "gdp_real")

    with pytest.raises(ValueError, match="Unknown QNA variable"):
        _resolve_series_id("USA", "invalid_var_name")


def test_qna_vintage_panel_analytics():
    """Test QNAVintagePanel methods: as_of, revision_matrix, first/latest release, stats."""
    # Synthetic panel data
    rows = []
    dates = pd.date_range("2020-01-01", periods=4, freq="QS")
    vintages = pd.date_range("2020-04-30", periods=5, freq="QE")

    # Build synthetic GDP revisions for USA and DEU
    for d_idx, d in enumerate(dates):
        for v_idx, v in enumerate(vintages):
            if v > d:
                # Value starts with small noise and settles
                base_us = 100.0 + d_idx * 2.0
                base_de = 80.0 + d_idx * 1.5
                val_us = base_us + (5 - v_idx) * 0.1
                val_de = base_de + (5 - v_idx) * 0.05
                rows.append({"country": "USA", "variable": "gdp_real", "date": d, "vintage": v, "value": val_us})
                rows.append({"country": "USA", "variable": "con_real", "date": d, "vintage": v, "value": val_us * 0.7})
                rows.append({"country": "DEU", "variable": "gdp_real", "date": d, "vintage": v, "value": val_de})

    df = pd.DataFrame(rows)
    panel = QNAVintagePanel(df=df)

    assert sorted(panel.countries) == ["DEU", "USA"]
    assert sorted(panel.variables) == ["con_real", "gdp_real"]
    assert len(panel.vintages) == 5

    # 1. Test as_of
    snap = panel.as_of("2020-08-01")
    assert isinstance(snap, pd.DataFrame)
    assert not snap.empty
    assert "gdp_real" in snap.columns

    # 2. Test revision_matrix
    rev_mat = panel.revision_matrix("USA", "gdp_real")
    assert isinstance(rev_mat, pd.DataFrame)
    assert rev_mat.shape[0] == 4
    assert rev_mat.columns.name == "vintage"

    # 3. Test first_release & latest_release
    first_rel = panel.first_release("USA", "gdp_real")
    latest_rel = panel.latest_release("USA", "gdp_real")
    assert isinstance(first_rel, pd.Series)
    assert isinstance(latest_rel, pd.Series)
    assert len(first_rel) == 4
    assert len(latest_rel) == 4
    assert (latest_rel <= first_rel).all()  # because (5 - v_idx) decreases

    # 4. Test revision_stats & Mankiw-Shapiro test
    stats = panel.revision_stats("USA", "gdp_real")
    assert "mean_revision" in stats
    assert "mankiw_shapiro_beta" in stats
    assert stats["n_obs"] == 4
    assert stats["hypothesis"] in {"news", "noise", "mixed"}

    # 5. Test to_panel_q exporter
    panel_q = panel.to_panel_q(log_transform=True)
    assert "log_gdp_real" in panel_q["variable"].values
    assert set(panel_q.columns) == {"code", "date", "variable", "value", "sa_source", "source"}

    # 6. Test filter
    filtered = panel.filter(countries=["USA"], variables=["gdp_real"])
    assert filtered.countries == ["USA"]
    assert filtered.variables == ["gdp_real"]


def test_fetch_qna_vintages_with_mock(monkeypatch, tmp_path):
    """Test fetch_qna_vintages with mocked live API and store caching."""
    db_file = tmp_path / "test_cache.db"
    store = AlfredVintageStore(db_path=db_file)

    def _mock_safe_urlopen(url, timeout=60.0):
        # Return CSV bytes formatted like ALFRED graph CSV
        csv_text = (
            "observation_date,GDPC1_2022-04-28,GDPC1_2022-05-26,GDPC1_2022-06-29\n"
            "2021-10-01,19806.2,19806.2,19806.2\n"
            "2022-01-01,19727.9,19735.4,19735.4\n"
        )
        return csv_text.encode("utf-8")

    monkeypatch.setattr("puremacro.fetch.qna_vintages._safe_urlopen", _mock_safe_urlopen)

    panel = fetch_qna_vintages(
        countries=["USA"],
        variables=["gdp_real"],
        store=store,
        refresh=True,
    )

    assert isinstance(panel, QNAVintagePanel)
    assert panel.countries == ["USA"]
    assert panel.variables == ["gdp_real"]
    assert len(panel.vintages) == 3
    assert len(panel.df) == 6

    # Verify rows were cached in store
    assert store.has_series("GDPC1")
    cached = store.get("GDPC1")
    assert len(cached) == 6

    # Second call without refresh should read directly from store
    panel_cached = fetch_qna_vintages(
        countries=["USA"],
        variables=["gdp_real"],
        store=store,
        refresh=False,
    )
    assert len(panel_cached.df) == 6
