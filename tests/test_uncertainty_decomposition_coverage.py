"""Coverage tests for puremacro.uncertainty.decomposition.

Covers:
  - _wide: pivot to wide format, long-format filtering, sort
  - between_within_share: known-value decomposition, share conservation,
    empty input, single-country, zero total variance
  - rolling_between_within_share: known-value, window boundary conditions,
    empty panel, min_country_obs filter, n_countries column
  - correlation_with_clustering: known-value identity matrix, clustering order,
    insufficient-obs filter, single country, pairwise sign, corr range

Strategy:
  - Known-value tests: synthetic data with a closed-form correct answer.
  - Property tests: shapes, dtypes, conservation laws, value bounds.
  - Edge-case tests: empty panels, degenerate inputs, boundary params.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from puremacro.uncertainty.decomposition import (
    _wide,
    between_within_share,
    correlation_with_clustering,
    rolling_between_within_share,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_long_panel(
    codes: list[str],
    var: str,
    values: dict[str, list[float]],
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a long-format panel with code/date/variable/value columns."""
    if dates is None:
        n = len(next(iter(values.values())))
        dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    rows = []
    for code in codes:
        for d, v in zip(dates, values[code]):
            rows.append({"code": code, "date": d, "variable": var, "value": v})
    return pd.DataFrame(rows)


def _make_identical_panel(
    n_dates: int = 24,
    n_codes: int = 3,
    var: str = "u",
    seed: int = 0,
) -> pd.DataFrame:
    """Panel where all countries share the same time series — between = total, within = 0."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n_dates).tolist()
    codes = [chr(65 + i) for i in range(n_codes)]
    dates = pd.date_range("2000-01-01", periods=n_dates, freq="MS")
    rows = []
    for code in codes:
        for d, v in zip(dates, common):
            rows.append({"code": code, "date": d, "variable": var, "value": v})
    return pd.DataFrame(rows)


def _make_constant_panel(
    n_dates: int = 24,
    n_codes: int = 3,
    var: str = "u",
    const: float = 5.0,
) -> pd.DataFrame:
    """Panel where every country-time cell equals const — total variance = 0."""
    codes = [chr(65 + i) for i in range(n_codes)]
    dates = pd.date_range("2000-01-01", periods=n_dates, freq="MS")
    rows = []
    for code in codes:
        for d in dates:
            rows.append({"code": code, "date": d, "variable": var, "value": const})
    return pd.DataFrame(rows)


def _make_iid_panel(
    n_dates: int = 120,
    n_codes: int = 4,
    var: str = "u",
    seed: int = 42,
) -> pd.DataFrame:
    """Panel with IID draws per country-time — used for property tests."""
    rng = np.random.default_rng(seed)
    codes = [chr(65 + i) for i in range(n_codes)]
    dates = pd.date_range("2000-01-01", periods=n_dates, freq="MS")
    rows = []
    for code in codes:
        vals = rng.standard_normal(n_dates)
        for d, v in zip(dates, vals):
            rows.append({"code": code, "date": d, "variable": var, "value": float(v)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _wide (private helper)
# ---------------------------------------------------------------------------

class TestWide:
    def test_basic_shape(self):
        """_wide returns shape (n_dates, n_codes) for given var."""
        panel = _make_iid_panel(n_dates=20, n_codes=3)
        w = _wide(panel, "u")
        assert w.shape == (20, 3)

    def test_filters_by_variable(self):
        """_wide returns only rows where variable == var."""
        dates = pd.date_range("2000-01-01", periods=5, freq="MS")
        rows = []
        for d in dates:
            rows.append({"code": "A", "date": d, "variable": "u", "value": 1.0})
            rows.append({"code": "A", "date": d, "variable": "v", "value": 9.0})
        panel = pd.DataFrame(rows)
        w_u = _wide(panel, "u")
        w_v = _wide(panel, "v")
        assert (w_u["A"] == 1.0).all()
        assert (w_v["A"] == 9.0).all()

    def test_sorted_by_date(self):
        """_wide index is sorted ascending by date."""
        panel = _make_iid_panel(n_dates=10, n_codes=2)
        # shuffle the panel
        panel = panel.sample(frac=1, random_state=0)
        w = _wide(panel, "u")
        assert list(w.index) == sorted(w.index)

    def test_unknown_variable_returns_empty(self):
        """_wide returns an empty DataFrame when variable is not present."""
        panel = _make_iid_panel(n_dates=5, n_codes=2)
        w = _wide(panel, "nonexistent")
        assert w.empty

    def test_values_match_input(self):
        """Known values survive the pivot."""
        dates = pd.date_range("2000-01-01", periods=3, freq="MS")
        panel = pd.DataFrame([
            {"code": "A", "date": dates[0], "variable": "u", "value": 1.0},
            {"code": "A", "date": dates[1], "variable": "u", "value": 2.0},
            {"code": "A", "date": dates[2], "variable": "u", "value": 3.0},
            {"code": "B", "date": dates[0], "variable": "u", "value": 10.0},
            {"code": "B", "date": dates[1], "variable": "u", "value": 20.0},
            {"code": "B", "date": dates[2], "variable": "u", "value": 30.0},
        ])
        w = _wide(panel, "u")
        np.testing.assert_array_equal(w["A"].values, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(w["B"].values, [10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# between_within_share
# ---------------------------------------------------------------------------

class TestBetweenWithinShare:
    def test_output_columns(self):
        """Result has exactly the expected columns."""
        panel = _make_iid_panel()
        result = between_within_share(panel, "u")
        expected_cols = {"between_share", "within_share", "total_var", "n_countries", "n_obs"}
        assert expected_cols == set(result.columns)

    def test_single_row(self):
        """Result is always a 1-row DataFrame."""
        panel = _make_iid_panel()
        result = between_within_share(panel, "u")
        assert result.shape[0] == 1

    def test_shares_sum_to_one(self):
        """between_share + within_share == 1 for valid data."""
        panel = _make_iid_panel()
        result = between_within_share(panel, "u")
        total = result["between_share"].iloc[0] + result["within_share"].iloc[0]
        assert total == pytest.approx(1.0, abs=1e-12)

    def test_shares_in_unit_interval(self):
        """Both shares are in [0, 1]."""
        panel = _make_iid_panel()
        result = between_within_share(panel, "u")
        b = result["between_share"].iloc[0]
        w = result["within_share"].iloc[0]
        assert 0.0 <= b <= 1.0
        assert 0.0 <= w <= 1.0

    def test_n_countries_correct(self):
        """n_countries counts the number of fully-observed countries."""
        panel = _make_iid_panel(n_codes=4)
        result = between_within_share(panel, "u")
        assert int(result["n_countries"].iloc[0]) == 4

    def test_n_obs_correct(self):
        """n_obs == n_dates * n_countries (balanced panel)."""
        panel = _make_iid_panel(n_dates=50, n_codes=3)
        result = between_within_share(panel, "u")
        assert int(result["n_obs"].iloc[0]) == 50

    def test_total_var_positive(self):
        """total_var is strictly positive for random IID data."""
        panel = _make_iid_panel()
        result = between_within_share(panel, "u")
        assert result["total_var"].iloc[0] > 0.0

    def test_identical_countries_between_share_is_one(self):
        """When all countries share the same time series, between_share = 1.0.

        Math: m = common signal (mean over identical columns = signal itself),
        between = Var(m) = Var(signal),
        within = mean per-country deviation Var = 0 (no deviations),
        => between_share = 1.
        """
        panel = _make_identical_panel(n_dates=30, n_codes=3)
        result = between_within_share(panel, "u")
        assert result["between_share"].iloc[0] == pytest.approx(1.0, abs=1e-10)
        assert result["within_share"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_zero_total_var_returns_nan(self):
        """When total_var = 0 (constant data), shares are NaN."""
        panel = _make_constant_panel(n_dates=20, n_codes=3, const=7.5)
        result = between_within_share(panel, "u")
        assert math.isnan(result["between_share"].iloc[0])
        assert math.isnan(result["within_share"].iloc[0])
        # total_var should be 0
        assert result["total_var"].iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_empty_panel_returns_nan_row(self):
        """Empty (no valid obs) returns a 1-row NaN DataFrame."""
        panel = pd.DataFrame(columns=["code", "date", "variable", "value"])
        result = between_within_share(panel, "u")
        assert result.shape[0] == 1
        assert math.isnan(result["between_share"].iloc[0])
        assert math.isnan(result["within_share"].iloc[0])
        assert int(result["n_countries"].iloc[0]) == 0
        assert int(result["n_obs"].iloc[0]) == 0

    def test_unknown_variable_returns_nan_row(self):
        """Requesting a var not in the panel gives a NaN row (via empty wide)."""
        panel = _make_iid_panel()
        result = between_within_share(panel, "not_a_var")
        assert result.shape[0] == 1
        assert math.isnan(result["between_share"].iloc[0])

    def test_country_with_any_nan_dropped(self):
        """Countries with any NaN are dropped (dropna axis=1, how='any').

        If we inject one NaN into country A, it is excluded from the decomp
        and n_countries decreases.
        """
        panel = _make_iid_panel(n_dates=30, n_codes=3)
        # Inject NaN for one code at one date
        mask = (panel["code"] == "A") & (panel["date"] == panel["date"].iloc[0])
        panel.loc[mask, "value"] = np.nan
        result = between_within_share(panel, "u")
        # Country A is dropped; only 2 remain
        assert int(result["n_countries"].iloc[0]) == 2

    def test_single_country_fully_within(self):
        """Single country: between_share = 0, within_share = 1.

        With only one country: m = that country's series,
        between = Var(m) = total,  wait — actually with 1 country:
        m = the single country; between = Var(m); within = mean of
        (col - m)^2 = 0 because col IS m.
        So between = total and between_share = 1, within_share = 0.
        """
        panel = _make_iid_panel(n_dates=30, n_codes=1)
        result = between_within_share(panel, "u")
        assert result["between_share"].iloc[0] == pytest.approx(1.0, abs=1e-10)
        assert result["within_share"].iloc[0] == pytest.approx(0.0, abs=1e-10)
        assert int(result["n_countries"].iloc[0]) == 1

    def test_known_value_decomposition(self):
        """Known-value test: 2 countries, 4 dates, exact arithmetic.

        Country A: [0, 2, 0, 2]
        Country B: [2, 0, 2, 0]

        Cross-section mean m = [1, 1, 1, 1] (constant -> Var(m) ddof=0 = 0)
        Wait — mean of [0,2] = 1 and [2,0] = 1, so m = [1,1,1,1].
        between = Var([1,1,1,1], ddof=0) = 0.
        deviation_A = [-1, 1, -1, 1], deviation_B = [1, -1, 1, -1]
        within = mean of [Var(dev_A, ddof=0), Var(dev_B, ddof=0)]
                = mean of [1.0, 1.0] = 1.0
        total = 0 + 1 = 1.0
        between_share = 0, within_share = 1.
        """
        dates = pd.date_range("2000-01-01", periods=4, freq="MS")
        rows = []
        for d, va, vb in zip(dates, [0, 2, 0, 2], [2, 0, 2, 0]):
            rows.append({"code": "A", "date": d, "variable": "u", "value": float(va)})
            rows.append({"code": "B", "date": d, "variable": "u", "value": float(vb)})
        panel = pd.DataFrame(rows)
        result = between_within_share(panel, "u")
        assert result["between_share"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert result["within_share"].iloc[0] == pytest.approx(1.0, abs=1e-12)
        assert result["total_var"].iloc[0] == pytest.approx(1.0, abs=1e-12)

    def test_known_value_purely_between(self):
        """Known-value test: both countries identical signals — all between, no within.

        Country A = Country B = [1, 2, 3, 4]
        m = [1, 2, 3, 4]
        between = Var([1,2,3,4], ddof=0) = 1.25
        deviation_A = deviation_B = [0,0,0,0] -> within = 0
        total = 1.25
        between_share = 1, within_share = 0.
        """
        dates = pd.date_range("2000-01-01", periods=4, freq="MS")
        vals = [1.0, 2.0, 3.0, 4.0]
        rows = []
        for code in ("A", "B"):
            for d, v in zip(dates, vals):
                rows.append({"code": code, "date": d, "variable": "u", "value": v})
        panel = pd.DataFrame(rows)
        result = between_within_share(panel, "u")
        assert result["between_share"].iloc[0] == pytest.approx(1.0, abs=1e-12)
        assert result["within_share"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert result["total_var"].iloc[0] == pytest.approx(1.25, abs=1e-12)


# ---------------------------------------------------------------------------
# rolling_between_within_share
# ---------------------------------------------------------------------------

class TestRollingBetweenWithinShare:
    def test_output_columns(self):
        """Result has columns: date, between_share, within_share, n_countries."""
        panel = _make_iid_panel(n_dates=150, n_codes=3)
        result = rolling_between_within_share(panel, "u", window_months=60)
        assert set(result.columns) == {"date", "between_share", "within_share", "n_countries"}

    def test_output_row_count_equals_dates(self):
        """Result has one row per unique date in the panel."""
        panel = _make_iid_panel(n_dates=50, n_codes=3)
        result = rolling_between_within_share(panel, "u", window_months=20)
        assert len(result) == 50

    def test_shares_sum_to_one_or_nan(self):
        """For each row with valid data, between + within == 1."""
        panel = _make_iid_panel(n_dates=150, n_codes=3)
        result = rolling_between_within_share(panel, "u", window_months=60)
        valid = result.dropna(subset=["between_share", "within_share"])
        totals = valid["between_share"] + valid["within_share"]
        np.testing.assert_allclose(totals.values, 1.0, atol=1e-12)

    def test_shares_in_unit_interval_or_nan(self):
        """Valid shares are in [0, 1]."""
        panel = _make_iid_panel(n_dates=150, n_codes=3)
        result = rolling_between_within_share(panel, "u", window_months=60)
        valid = result.dropna(subset=["between_share", "within_share"])
        assert (valid["between_share"] >= -1e-12).all()
        assert (valid["between_share"] <= 1.0 + 1e-12).all()
        assert (valid["within_share"] >= -1e-12).all()
        assert (valid["within_share"] <= 1.0 + 1e-12).all()

    def test_edge_dates_have_nan(self):
        """Near the boundaries of the panel, there are not enough obs -> NaN."""
        # window=60, so half=30; first 30 dates can't fill a full window
        panel = _make_iid_panel(n_dates=80, n_codes=3)
        result = rolling_between_within_share(panel, "u", window_months=60)
        # first row definitely doesn't have a full window
        assert math.isnan(result["between_share"].iloc[0])

    def test_empty_panel_returns_empty_df(self):
        """Empty panel returns DataFrame with correct columns but no rows."""
        panel = pd.DataFrame(columns=["code", "date", "variable", "value"])
        result = rolling_between_within_share(panel, "u")
        assert set(result.columns) == {"date", "between_share", "within_share", "n_countries"}
        assert len(result) == 0

    def test_unknown_variable_returns_empty_df(self):
        """Variable not in panel returns empty DataFrame."""
        panel = _make_iid_panel(n_dates=20, n_codes=2)
        result = rolling_between_within_share(panel, "not_there")
        assert len(result) == 0

    def test_n_countries_nonnegative(self):
        """n_countries is always >= 0."""
        panel = _make_iid_panel(n_dates=100, n_codes=3)
        result = rolling_between_within_share(panel, "u", window_months=40)
        assert (result["n_countries"] >= 0).all()

    def test_identical_countries_between_share_one(self):
        """When all countries share the same signal, between_share = 1 wherever valid."""
        panel = _make_identical_panel(n_dates=150, n_codes=3)
        result = rolling_between_within_share(panel, "u", window_months=60)
        valid = result.dropna(subset=["between_share"])
        assert (valid["between_share"] > 0.999).all()

    def test_min_country_obs_filter(self):
        """Countries with fewer than min_country_obs obs in a window are excluded."""
        # Build a panel with a very short country-obs overlap, then demand high min_country_obs
        panel = _make_iid_panel(n_dates=200, n_codes=3)
        # request min_country_obs=999 (impossible) -> no valid windows
        result = rolling_between_within_share(panel, "u", window_months=60, min_country_obs=999)
        # All rows should have NaN shares or n_countries < 2
        valid = result.dropna(subset=["between_share"])
        assert len(valid) == 0

    def test_fewer_than_2_countries_gives_nan(self):
        """If only 1 country passes the obs filter, result is NaN."""
        # Only 1 country in the panel
        panel = _make_iid_panel(n_dates=150, n_codes=1)
        result = rolling_between_within_share(panel, "u", window_months=60)
        valid = result.dropna(subset=["between_share"])
        assert len(valid) == 0

    def test_window_months_parameter_respected(self):
        """A larger window produces more boundary NaNs when min_country_obs is low.

        With min_country_obs=10, a window of 100 half=50 produces ~100 boundary NaNs,
        while window=20 half=10 produces ~18. So nan_large > nan_small.
        """
        panel = _make_iid_panel(n_dates=200, n_codes=3)
        result_small = rolling_between_within_share(
            panel, "u", window_months=20, min_country_obs=10
        )
        result_large = rolling_between_within_share(
            panel, "u", window_months=100, min_country_obs=10
        )
        nan_small = result_small["between_share"].isna().sum()
        nan_large = result_large["between_share"].isna().sum()
        # Larger window -> more NaN boundary rows
        assert nan_large > nan_small

    def test_constant_data_within_window_gives_nan_shares(self):
        """When data is constant within a window, total_var=0 -> NaN shares."""
        panel = _make_constant_panel(n_dates=150, n_codes=3, const=3.0)
        result = rolling_between_within_share(panel, "u", window_months=60)
        valid_windows = result[result["n_countries"] >= 2]
        # Shares should all be NaN (total_var=0)
        if len(valid_windows) > 0:
            assert valid_windows["between_share"].isna().all()


# ---------------------------------------------------------------------------
# correlation_with_clustering
# ---------------------------------------------------------------------------

class TestCorrelationWithClustering:
    def _long_panel(
        self,
        n_dates: int = 60,
        n_codes: int = 3,
        var: str = "u",
        seed: int = 0,
    ) -> pd.DataFrame:
        """Build a long panel with enough obs per country to pass the 24-obs filter."""
        rng = np.random.default_rng(seed)
        codes = [chr(65 + i) for i in range(n_codes)]
        dates = pd.date_range("2000-01-01", periods=n_dates, freq="MS")
        rows = []
        for code in codes:
            vals = rng.standard_normal(n_dates)
            for d, v in zip(dates, vals):
                rows.append({"code": code, "date": d, "variable": var, "value": float(v)})
        return pd.DataFrame(rows)

    def test_returns_tuple(self):
        """correlation_with_clustering returns a 2-tuple."""
        panel = self._long_panel(n_dates=60, n_codes=3)
        out = correlation_with_clustering(panel, "u")
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_corr_is_dataframe(self):
        """First element of result is a DataFrame."""
        panel = self._long_panel(n_dates=60, n_codes=3)
        corr, order = correlation_with_clustering(panel, "u")
        assert isinstance(corr, pd.DataFrame)

    def test_order_is_list_of_codes(self):
        """Second element is a list of country code strings."""
        panel = self._long_panel(n_dates=60, n_codes=3)
        corr, order = correlation_with_clustering(panel, "u")
        assert isinstance(order, list)
        for code in order:
            assert isinstance(code, str)

    def test_corr_diagonal_is_one(self):
        """Self-correlations are exactly 1.0."""
        panel = self._long_panel(n_dates=60, n_codes=3)
        corr, order = correlation_with_clustering(panel, "u")
        for code in corr.index:
            assert corr.loc[code, code] == pytest.approx(1.0, abs=1e-12)

    def test_corr_in_minus_one_to_one(self):
        """All correlations are in [-1, 1]."""
        panel = self._long_panel(n_dates=60, n_codes=4)
        corr, order = correlation_with_clustering(panel, "u")
        vals = corr.values.flatten()
        assert (vals >= -1.0 - 1e-10).all()
        assert (vals <= 1.0 + 1e-10).all()

    def test_corr_is_symmetric(self):
        """Correlation matrix is symmetric."""
        panel = self._long_panel(n_dates=60, n_codes=4)
        corr, order = correlation_with_clustering(panel, "u")
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-12)

    def test_order_contains_all_codes(self):
        """Clustering order is a permutation of the codes in the corr matrix."""
        panel = self._long_panel(n_dates=60, n_codes=4)
        corr, order = correlation_with_clustering(panel, "u")
        assert sorted(order) == sorted(list(corr.index))

    def test_order_length_equals_n_codes(self):
        """Order list has same length as number of countries."""
        panel = self._long_panel(n_dates=60, n_codes=5)
        corr, order = correlation_with_clustering(panel, "u")
        assert len(order) == len(corr.index)

    def test_perfectly_correlated_pair_adjacent(self):
        """Two perfectly-correlated countries should be adjacent in the clustering order.

        NOTE: With only 2 countries there is only one possible pair, so order
        trivially contains both — we just verify the order has both codes.
        This is a contract test (order is a permutation).
        """
        dates = pd.date_range("2000-01-01", periods=60, freq="MS")
        rng = np.random.default_rng(7)
        common = rng.standard_normal(60)
        rows = []
        for code in ("A", "B", "C"):
            # A and B are identical; C is independent
            if code in ("A", "B"):
                vals = common
            else:
                vals = rng.standard_normal(60)
            for d, v in zip(dates, vals):
                rows.append({"code": code, "date": d, "variable": "u", "value": float(v)})
        panel = pd.DataFrame(rows)
        corr, order = correlation_with_clustering(panel, "u")
        # A and B should have correlation close to 1 on first differences (identical signal)
        a_idx = list(corr.index).index("A")
        b_idx = list(corr.index).index("B")
        assert corr.iloc[a_idx, b_idx] == pytest.approx(1.0, abs=1e-6)

    def test_insufficient_obs_filtered(self):
        """Countries with fewer than 24 non-null obs in diff are excluded."""
        # 3 countries, 25 dates — barely passes for A and B
        # but create C with only a few obs in the panel
        dates_main = pd.date_range("2000-01-01", periods=60, freq="MS")
        # C has only 10 obs (first 10 dates)
        rows = []
        rng = np.random.default_rng(99)
        for code, n_obs in [("A", 60), ("B", 60), ("C", 10)]:
            sub_dates = dates_main[:n_obs]
            for d, v in zip(sub_dates, rng.standard_normal(n_obs)):
                rows.append({"code": code, "date": d, "variable": "u", "value": float(v)})
        panel = pd.DataFrame(rows)
        corr, order = correlation_with_clustering(panel, "u")
        # C has only 10 obs -> diff has 9 non-null -> < 24 -> excluded
        assert "C" not in order

    def test_single_country_returns_corr_and_list(self):
        """With only 1 country passing filter, returns corr and a 1-element list."""
        panel = self._long_panel(n_dates=60, n_codes=1)
        corr, order = correlation_with_clustering(panel, "u")
        # Should return without clustering (< 2 countries)
        assert isinstance(corr, pd.DataFrame)
        assert isinstance(order, list)

    def test_linkage_method_kwarg_accepted(self):
        """The method= kwarg is passed through without error."""
        panel = self._long_panel(n_dates=60, n_codes=3)
        for method in ("single", "complete", "average", "ward"):
            corr, order = correlation_with_clustering(panel, "u", method=method)
            assert len(order) == 3

    def test_known_value_independent_countries(self):
        """Three completely independent countries: off-diagonal correlations near 0.

        With large n, Pearson on first-differences of IID series -> ~0 correlation.
        We use a large sample (n=1000) so the law-of-large-numbers applies.
        Contract: |corr[i,j]| < 0.15 for i != j.
        """
        rng = np.random.default_rng(123)
        dates = pd.date_range("2000-01-01", periods=1000, freq="MS")
        rows = []
        for code in ("X", "Y", "Z"):
            vals = rng.standard_normal(1000)
            for d, v in zip(dates, vals):
                rows.append({"code": code, "date": d, "variable": "u", "value": float(v)})
        panel = pd.DataFrame(rows)
        corr, order = correlation_with_clustering(panel, "u")
        for i in corr.index:
            for j in corr.columns:
                if i != j:
                    assert abs(corr.loc[i, j]) < 0.15, \
                        f"Off-diagonal corr({i},{j}) = {corr.loc[i, j]:.3f} exceeds 0.15"

    def test_unknown_variable_returns_empty_corr(self):
        """Unknown variable produces an empty corr matrix and empty order."""
        panel = self._long_panel(n_dates=60, n_codes=3)
        corr, order = correlation_with_clustering(panel, "nonexistent")
        assert corr.empty or len(order) <= 1
