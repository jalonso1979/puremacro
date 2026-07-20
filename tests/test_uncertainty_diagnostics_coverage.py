"""Coverage tests for puremacro.uncertainty.diagnostics.

Tests exercise: _extract_tier, _extract_proxies, build_diagnostics.

Strategy:
- Known-value tests: source strings with known tier/proxy tags.
- Property tests: output columns, types, n_obs counts, first/last_obs dates,
  bounds (p-values in [0,1] when unit-root tests are available),
  ar2_sigma non-negative (variance >= 0 => sd >= 0).
- Edge cases: bad freq raises ValueError, too-few observations skip country,
  missing innovation variable produces NaN ar2_sigma,
  source strings with various prefixes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.uncertainty.diagnostics import (
    _extract_proxies,
    _extract_tier,
    build_diagnostics,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic panels
# ---------------------------------------------------------------------------

def _make_pc_panel(
    codes: list[str],
    n: int = 30,
    freq_suffix: str = "q",
    source_template: str = "PC1(wui_m+gpr_m+epu_m; tier=T1; rescaled)",
    seed: int = 0,
) -> pd.DataFrame:
    """Build a long-form panel with unc_pc_{freq} rows per country."""
    rng = np.random.default_rng(seed)
    if freq_suffix == "q":
        dates = pd.date_range("2000-01-01", periods=n, freq="QS")
    else:
        dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    rows = []
    for code in codes:
        vals = rng.standard_normal(n) * 20 + 100
        for d, v in zip(dates, vals):
            rows.append(
                {
                    "code": code,
                    "date": d,
                    "variable": f"unc_pc_{freq_suffix}",
                    "value": float(v),
                    "sa_source": "derived",
                    "source": source_template,
                }
            )
    return pd.DataFrame(rows)


def _make_innov_panel(
    codes: list[str],
    n: int = 28,
    freq_suffix: str = "q",
    seed: int = 1,
) -> pd.DataFrame:
    """Build a long-form panel with unc_innov_{freq} rows per country."""
    rng = np.random.default_rng(seed)
    if freq_suffix == "q":
        dates = pd.date_range("2000-07-01", periods=n, freq="QS")
    else:
        dates = pd.date_range("2000-03-01", periods=n, freq="MS")
    rows = []
    for code in codes:
        vals = rng.standard_normal(n)
        for d, v in zip(dates, vals):
            rows.append(
                {
                    "code": code,
                    "date": d,
                    "variable": f"unc_innov_{freq_suffix}",
                    "value": float(v),
                    "sa_source": "derived",
                    "source": f"AR(2)-resid; tier=T1",
                }
            )
    return pd.DataFrame(rows)


def _make_full_panel(
    codes: list[str],
    n_pc: int = 30,
    n_innov: int = 28,
    freq_suffix: str = "q",
    source_template: str = "PC1(wui_m+gpr_m+epu_m; tier=T1; rescaled)",
) -> pd.DataFrame:
    pc = _make_pc_panel(codes, n=n_pc, freq_suffix=freq_suffix, source_template=source_template)
    innov = _make_innov_panel(codes, n=n_innov, freq_suffix=freq_suffix)
    return pd.concat([pc, innov], ignore_index=True)


# ===========================================================================
# _extract_tier
# ===========================================================================


def test_extract_tier_t1():
    src = "PC1(wui_m+gpr_m; tier=T1; rescaled mean=100.0, sd=20.0)"
    assert _extract_tier(src) == "T1"


def test_extract_tier_t2():
    src = "PC1(gpr_m+epu_m; tier=T2; rescaled mean=100.0, sd=20.0)"
    assert _extract_tier(src) == "T2"


def test_extract_tier_t3():
    src = "single_proxy(gpr_m, gpr; tier=T3; rescaled mean=100.0, sd=20.0)"
    assert _extract_tier(src) == "T3"


def test_extract_tier_no_tag_returns_question_mark():
    assert _extract_tier("some random source string") == "?"


def test_extract_tier_t1_takes_priority_over_no_other_tier():
    """First matching tag wins; T1 tag present returns T1."""
    src = "PC1(a+b; tier=T1; extra info)"
    assert _extract_tier(src) == "T1"


# ===========================================================================
# _extract_proxies
# ===========================================================================


def test_extract_proxies_pc1_format():
    src = "PC1(wui_m+gpr_m+epu_m; tier=T1; rescaled)"
    assert _extract_proxies(src) == "wui_m+gpr_m+epu_m"


def test_extract_proxies_single_proxy_format():
    src = "single_proxy(gpr_m, gpr; tier=T3; rescaled)"
    assert _extract_proxies(src) == "gpr_m"


def test_extract_proxies_resampled_from_m_prefix():
    """Tolerates 'resampled_from_M:' prefix before PC1(...)."""
    src = "resampled_from_M:PC1(wui_m+gpr_m; tier=T1; rescaled)"
    assert _extract_proxies(src) == "wui_m+gpr_m"


def test_extract_proxies_unknown_format_returns_question_mark():
    assert _extract_proxies("something_else(foo; bar)") == "?"


def test_extract_proxies_resampled_single_proxy():
    """resampled_from_M: prefix with single_proxy(...)."""
    src = "resampled_from_M:single_proxy(gpr_m, gpr; tier=T3)"
    assert _extract_proxies(src) == "gpr_m"


def test_extract_proxies_pc1_two_proxies():
    src = "PC1(gpr_m+epu_m; tier=T2; rescaled mean=100.0)"
    assert _extract_proxies(src) == "gpr_m+epu_m"


# ===========================================================================
# build_diagnostics — bad freq raises ValueError
# ===========================================================================


def test_build_diagnostics_bad_freq_raises():
    panel = pd.DataFrame(
        {
            "code": ["USA"],
            "date": [pd.Timestamp("2000-01-01")],
            "variable": ["unc_pc_q"],
            "value": [100.0],
            "source": ["PC1(gpr_m; tier=T1)"],
        }
    )
    with pytest.raises(ValueError):
        build_diagnostics(panel, freq="x")


def test_build_diagnostics_bad_freq_y_raises():
    panel = pd.DataFrame()
    with pytest.raises(ValueError):
        build_diagnostics(panel, freq="y")


# ===========================================================================
# build_diagnostics — output schema (column names, dtypes, shape)
# ===========================================================================


def test_build_diagnostics_output_columns_quarterly():
    panel = _make_full_panel(["USA", "DEU"], freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    expected_cols = {
        "code", "freq", "tier", "n_proxies", "proxies", "n_obs",
        "adf_stat", "adf_p", "kpss_stat", "kpss_p", "ar2_sigma",
        "first_obs", "last_obs",
    }
    assert expected_cols.issubset(set(diag.columns)), (
        f"Missing columns: {expected_cols - set(diag.columns)}"
    )


def test_build_diagnostics_output_columns_monthly():
    panel = _make_full_panel(["USA"], freq_suffix="m")
    diag = build_diagnostics(panel, freq="m")
    expected_cols = {
        "code", "freq", "tier", "n_proxies", "proxies", "n_obs",
        "adf_stat", "adf_p", "kpss_stat", "kpss_p", "ar2_sigma",
        "first_obs", "last_obs",
    }
    assert expected_cols.issubset(set(diag.columns))


def test_build_diagnostics_one_row_per_country():
    panel = _make_full_panel(["USA", "DEU", "GBR"], freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    assert set(diag["code"].unique()) == {"USA", "DEU", "GBR"}
    assert len(diag) == 3


def test_build_diagnostics_freq_column_correct():
    panel = _make_full_panel(["USA"], freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    assert (diag["freq"] == "q").all()


def test_build_diagnostics_freq_m_column_correct():
    panel = _make_full_panel(["USA"], freq_suffix="m")
    diag = build_diagnostics(panel, freq="m")
    assert (diag["freq"] == "m").all()


# ===========================================================================
# build_diagnostics — n_obs
# ===========================================================================


def test_build_diagnostics_n_obs_matches_input():
    """n_obs must equal the number of non-NaN pc values per country."""
    n = 40
    panel = _make_full_panel(["USA"], n_pc=n, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    assert int(diag.loc[diag["code"] == "USA", "n_obs"].iloc[0]) == n


def test_build_diagnostics_n_obs_is_integer():
    panel = _make_full_panel(["USA"], n_pc=30, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    val = diag["n_obs"].iloc[0]
    assert isinstance(val, (int, np.integer))


# ===========================================================================
# build_diagnostics — tier and proxies extraction
# ===========================================================================


def test_build_diagnostics_tier_t1():
    panel = _make_full_panel(
        ["USA"],
        source_template="PC1(wui_m+gpr_m+epu_m; tier=T1; rescaled)",
    )
    diag = build_diagnostics(panel, freq="q")
    assert diag["tier"].iloc[0] == "T1"


def test_build_diagnostics_tier_t2():
    panel = _make_full_panel(
        ["USA"],
        source_template="PC1(gpr_m+epu_m; tier=T2; rescaled)",
    )
    diag = build_diagnostics(panel, freq="q")
    assert diag["tier"].iloc[0] == "T2"


def test_build_diagnostics_tier_t3():
    panel = _make_full_panel(
        ["USA"],
        source_template="single_proxy(gpr_m, gpr; tier=T3; rescaled)",
    )
    diag = build_diagnostics(panel, freq="q")
    assert diag["tier"].iloc[0] == "T3"


def test_build_diagnostics_n_proxies_three():
    panel = _make_full_panel(
        ["USA"],
        source_template="PC1(wui_m+gpr_m+epu_m; tier=T1; rescaled)",
    )
    diag = build_diagnostics(panel, freq="q")
    assert diag["n_proxies"].iloc[0] == 3


def test_build_diagnostics_n_proxies_two():
    panel = _make_full_panel(
        ["USA"],
        source_template="PC1(gpr_m+epu_m; tier=T2; rescaled)",
    )
    diag = build_diagnostics(panel, freq="q")
    assert diag["n_proxies"].iloc[0] == 2


def test_build_diagnostics_n_proxies_unknown_source_is_zero():
    panel = _make_full_panel(
        ["USA"],
        source_template="unknown_format(stuff)",
    )
    diag = build_diagnostics(panel, freq="q")
    assert diag["n_proxies"].iloc[0] == 0


def test_build_diagnostics_proxies_string_correct():
    panel = _make_full_panel(
        ["USA"],
        source_template="PC1(wui_m+gpr_m+epu_m; tier=T1; rescaled)",
    )
    diag = build_diagnostics(panel, freq="q")
    assert diag["proxies"].iloc[0] == "wui_m+gpr_m+epu_m"


# ===========================================================================
# build_diagnostics — first_obs / last_obs
# ===========================================================================


def test_build_diagnostics_first_last_obs_iso_format():
    """first_obs and last_obs must be ISO-format date strings (YYYY-MM-DD)."""
    panel = _make_full_panel(["USA"], n_pc=30, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    row = diag[diag["code"] == "USA"].iloc[0]
    # Parse to ensure they are valid dates
    pd.Timestamp(row["first_obs"])
    pd.Timestamp(row["last_obs"])
    # first_obs must be before last_obs
    assert row["first_obs"] <= row["last_obs"]


def test_build_diagnostics_first_obs_matches_earliest_date():
    """first_obs must match the earliest non-NaN date in the pc series."""
    n = 20
    dates = pd.date_range("2005-01-01", periods=n, freq="QS")
    rows = []
    for d, v in zip(dates, np.random.default_rng(5).standard_normal(n)):
        rows.append(
            {
                "code": "USA",
                "date": d,
                "variable": "unc_pc_q",
                "value": float(v),
                "sa_source": "derived",
                "source": "PC1(gpr_m; tier=T1; rescaled)",
            }
        )
    panel = pd.DataFrame(rows)
    diag = build_diagnostics(panel, freq="q")
    expected_first = dates[0].date().isoformat()
    expected_last = dates[-1].date().isoformat()
    assert diag["first_obs"].iloc[0] == expected_first
    assert diag["last_obs"].iloc[0] == expected_last


# ===========================================================================
# build_diagnostics — ar2_sigma
# ===========================================================================


def test_build_diagnostics_ar2_sigma_nonneg():
    """ar2_sigma is a standard deviation, must be >= 0."""
    panel = _make_full_panel(["USA", "DEU"], freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    for _, row in diag.iterrows():
        if not np.isnan(row["ar2_sigma"]):
            assert row["ar2_sigma"] >= 0.0


def test_build_diagnostics_ar2_sigma_nan_when_no_innov():
    """When there are no unc_innov rows for a country, ar2_sigma must be NaN."""
    # Build only pc rows, no innov rows
    panel = _make_pc_panel(["USA"], n=30, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    assert np.isnan(diag["ar2_sigma"].iloc[0])


def test_build_diagnostics_ar2_sigma_near_one_for_standardized_innov():
    """If the innovation series has mean~0 and sd~1 (std population), ar2_sigma ~ 1."""
    rng = np.random.default_rng(42)
    n_innov = 50
    dates = pd.date_range("2000-01-01", periods=n_innov, freq="QS")
    innov_vals = rng.standard_normal(n_innov)  # sd (pop) ~ 1
    # Manually standardize to exact population sd=1
    innov_vals = (innov_vals - innov_vals.mean()) / innov_vals.std(ddof=0)

    innov_rows = [
        {
            "code": "USA",
            "date": d,
            "variable": "unc_innov_q",
            "value": float(v),
            "sa_source": "derived",
            "source": "AR(2)-resid; tier=T1",
        }
        for d, v in zip(dates, innov_vals)
    ]
    # Add enough pc rows for the country not to be skipped
    pc_rows = [
        {
            "code": "USA",
            "date": d,
            "variable": "unc_pc_q",
            "value": float(v) * 20 + 100,
            "sa_source": "derived",
            "source": "PC1(wui_m+gpr_m; tier=T1; rescaled)",
        }
        for d, v in zip(dates, innov_vals)
    ]
    panel = pd.DataFrame(innov_rows + pc_rows)
    diag = build_diagnostics(panel, freq="q")
    sigma = diag["ar2_sigma"].iloc[0]
    # ar2_sigma is std(ddof=0) of innov values = 1.0 exactly
    assert abs(sigma - 1.0) < 1e-10


# ===========================================================================
# build_diagnostics — ADF / KPSS p-values in [0,1]
# ===========================================================================


def test_build_diagnostics_adf_p_in_bounds_or_nan():
    """ADF p-value must be in [0, 1] when not NaN."""
    panel = _make_full_panel(["USA", "DEU"], n_pc=60, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    for val in diag["adf_p"]:
        if not np.isnan(val):
            assert 0.0 <= val <= 1.0, f"adf_p out of bounds: {val}"


def test_build_diagnostics_kpss_p_in_bounds_or_nan():
    """KPSS p-value must be in [0, 1] when not NaN."""
    panel = _make_full_panel(["USA", "DEU"], n_pc=60, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    for val in diag["kpss_p"]:
        if not np.isnan(val):
            assert 0.0 <= val <= 1.0, f"kpss_p out of bounds: {val}"


def test_build_diagnostics_adf_stat_is_finite_or_nan():
    panel = _make_full_panel(["USA"], n_pc=60, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    adf_stat = diag["adf_stat"].iloc[0]
    assert np.isnan(adf_stat) or np.isfinite(adf_stat)


def test_build_diagnostics_kpss_stat_nonneg_or_nan():
    """KPSS statistic is always non-negative when available."""
    panel = _make_full_panel(["USA"], n_pc=60, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    kpss_stat = diag["kpss_stat"].iloc[0]
    if not np.isnan(kpss_stat):
        assert kpss_stat >= 0.0


# ===========================================================================
# build_diagnostics — too few observations (< 10) skips country
# ===========================================================================


def test_build_diagnostics_skips_country_with_fewer_than_10_obs():
    """Countries with fewer than 10 pc observations must be skipped."""
    rng = np.random.default_rng(0)
    n = 9  # < 10
    dates = pd.date_range("2000-01-01", periods=n, freq="QS")
    rows = [
        {
            "code": "USA",
            "date": d,
            "variable": "unc_pc_q",
            "value": float(v),
            "sa_source": "derived",
            "source": "PC1(gpr_m; tier=T1; rescaled)",
        }
        for d, v in zip(dates, rng.standard_normal(n))
    ]
    panel = pd.DataFrame(rows)
    diag = build_diagnostics(panel, freq="q")
    assert len(diag) == 0


def test_build_diagnostics_exactly_10_obs_included():
    """Exactly 10 observations is the threshold — country should appear."""
    rng = np.random.default_rng(0)
    n = 10
    dates = pd.date_range("2000-01-01", periods=n, freq="QS")
    rows = [
        {
            "code": "USA",
            "date": d,
            "variable": "unc_pc_q",
            "value": float(v),
            "sa_source": "derived",
            "source": "PC1(gpr_m; tier=T1; rescaled)",
        }
        for d, v in zip(dates, rng.standard_normal(n))
    ]
    panel = pd.DataFrame(rows)
    diag = build_diagnostics(panel, freq="q")
    assert len(diag) == 1
    assert diag["code"].iloc[0] == "USA"


# ===========================================================================
# build_diagnostics — empty panel
# ===========================================================================


def test_build_diagnostics_empty_panel_returns_empty_df():
    """Empty panel should return an empty DataFrame (no rows)."""
    panel = pd.DataFrame(
        columns=["code", "date", "variable", "value", "sa_source", "source"]
    )
    diag = build_diagnostics(panel, freq="q")
    assert isinstance(diag, pd.DataFrame)
    assert len(diag) == 0


def test_build_diagnostics_panel_without_pc_var_returns_empty():
    """Panel with no unc_pc_q rows should return empty DataFrame."""
    panel = pd.DataFrame(
        {
            "code": ["USA"] * 20,
            "date": pd.date_range("2000-01-01", periods=20, freq="QS"),
            "variable": ["unc_innov_q"] * 20,
            "value": np.ones(20),
            "sa_source": ["derived"] * 20,
            "source": ["AR(2)-resid; tier=T1"] * 20,
        }
    )
    diag = build_diagnostics(panel, freq="q")
    assert len(diag) == 0


# ===========================================================================
# build_diagnostics — multiple countries, country isolation
# ===========================================================================


def test_build_diagnostics_codes_match_input_countries():
    codes = ["USA", "DEU", "GBR", "FRA"]
    panel = _make_full_panel(codes, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    assert set(diag["code"].unique()) == set(codes)


def test_build_diagnostics_each_country_independent_n_obs():
    """Different n_obs per country are preserved correctly."""
    rng = np.random.default_rng(7)
    rows = []
    for code, n in [("USA", 20), ("DEU", 40)]:
        dates = pd.date_range("2000-01-01", periods=n, freq="QS")
        for d, v in zip(dates, rng.standard_normal(n)):
            rows.append(
                {
                    "code": code,
                    "date": d,
                    "variable": "unc_pc_q",
                    "value": float(v),
                    "sa_source": "derived",
                    "source": "PC1(gpr_m+epu_m; tier=T2; rescaled)",
                }
            )
    panel = pd.DataFrame(rows)
    diag = build_diagnostics(panel, freq="q")
    usa_obs = int(diag[diag["code"] == "USA"]["n_obs"].iloc[0])
    deu_obs = int(diag[diag["code"] == "DEU"]["n_obs"].iloc[0])
    assert usa_obs == 20
    assert deu_obs == 40


# ===========================================================================
# build_diagnostics — source with resampled_from_M prefix
# ===========================================================================


def test_build_diagnostics_resampled_source_extracts_proxies():
    """Source strings starting with 'resampled_from_M:' are handled correctly."""
    panel = _make_full_panel(
        ["USA"],
        source_template="resampled_from_M:PC1(wui_m+gpr_m+epu_m; tier=T1; rescaled)",
    )
    diag = build_diagnostics(panel, freq="q")
    assert diag["proxies"].iloc[0] == "wui_m+gpr_m+epu_m"
    assert diag["tier"].iloc[0] == "T1"
    assert diag["n_proxies"].iloc[0] == 3


# ===========================================================================
# build_diagnostics — monthly freq
# ===========================================================================


def test_build_diagnostics_monthly_freq_works():
    panel = _make_full_panel(["USA", "DEU"], freq_suffix="m")
    diag = build_diagnostics(panel, freq="m")
    assert len(diag) == 2
    assert (diag["freq"] == "m").all()


def test_build_diagnostics_monthly_n_obs_correct():
    n = 36
    panel = _make_full_panel(["USA"], n_pc=n, freq_suffix="m")
    diag = build_diagnostics(panel, freq="m")
    assert int(diag["n_obs"].iloc[0]) == n


# ===========================================================================
# build_diagnostics — exception paths in ADF / KPSS
# ===========================================================================


def test_build_diagnostics_adf_exception_produces_nan(monkeypatch):
    """When adf_test raises, the adf_stat and adf_p must be NaN."""
    import puremacro.uncertainty.diagnostics as _diag

    def _bad_adf(*args, **kwargs):
        raise RuntimeError("simulated ADF failure")

    monkeypatch.setattr(_diag, "adf_test", _bad_adf)
    # Keep kpss_test working normally so we only exercise the ADF except branch
    panel = _make_full_panel(["USA"], n_pc=30, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    assert np.isnan(diag["adf_stat"].iloc[0])
    assert np.isnan(diag["adf_p"].iloc[0])
    # kpss should still be populated (not NaN) since it didn't fail
    assert not np.isnan(diag["kpss_stat"].iloc[0])


def test_build_diagnostics_kpss_exception_produces_nan(monkeypatch):
    """When kpss_test raises, the kpss_stat and kpss_p must be NaN."""
    import puremacro.uncertainty.diagnostics as _diag

    def _bad_kpss(*args, **kwargs):
        raise RuntimeError("simulated KPSS failure")

    monkeypatch.setattr(_diag, "kpss_test", _bad_kpss)
    panel = _make_full_panel(["USA"], n_pc=30, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    assert np.isnan(diag["kpss_stat"].iloc[0])
    assert np.isnan(diag["kpss_p"].iloc[0])
    # adf should still be populated
    assert not np.isnan(diag["adf_stat"].iloc[0])


def test_build_diagnostics_no_unit_root_tests_produces_all_nan(monkeypatch):
    """When both adf_test and kpss_test are None, all stat/p cells are NaN."""
    import puremacro.uncertainty.diagnostics as _diag

    monkeypatch.setattr(_diag, "adf_test", None)
    monkeypatch.setattr(_diag, "kpss_test", None)
    panel = _make_full_panel(["USA"], n_pc=30, freq_suffix="q")
    diag = build_diagnostics(panel, freq="q")
    assert np.isnan(diag["adf_stat"].iloc[0])
    assert np.isnan(diag["adf_p"].iloc[0])
    assert np.isnan(diag["kpss_stat"].iloc[0])
    assert np.isnan(diag["kpss_p"].iloc[0])
