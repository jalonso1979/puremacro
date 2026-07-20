"""Coverage tests for puremacro.labor_flows.

Test strategy
-------------
- known_value: synthetic inputs with analytically derivable correct answers
  (row-sums, chained-product arithmetic, balanced-flow accounting identity).
- property: shapes, dtypes, column names, row-stochasticity, non-negativity,
  stochastic-matrix property of quarterly chains.
- edge cases + error paths: missing columns raises ValueError, incomplete
  quarter yields empty quarterly DataFrame, percent vs. decimal urate
  conversion, 2-state vs. 3-state decomposition branches.
- contract: output types, index alignment, IRF output columns/dtypes.

No network I/O: all inputs are synthetic NumPy/Pandas data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.labor_flows import (
    CPS_FLOW_SERIES,
    CPS_STOCK_SERIES,
    TransitionPanel,
    _quarterly_chain,
    supply_demand_decomposition,
    transition_shock_response,
    transitions_from_cps_flows,
    transitions_from_shimer,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_monthly_dates(n: int = 24, start: str = "2010-01-01") -> pd.DatetimeIndex:
    idx = pd.date_range(start, periods=n, freq="MS")
    return idx


def _constant_urate(u_decimal: float = 0.06, n: int = 24) -> pd.Series:
    """Constant unemployment rate series."""
    idx = _make_monthly_dates(n)
    s = pd.Series([u_decimal] * n, index=idx)
    s.index.freq = "MS"
    return s


def _varying_urate(n: int = 48, seed: int = 42) -> pd.Series:
    """Smooth random unemployment rate in [0.03, 0.15]."""
    rng = np.random.default_rng(seed)
    idx = _make_monthly_dates(n)
    u = 0.07 + 0.02 * rng.standard_normal(n)
    s = pd.Series(u, index=idx).clip(0.03, 0.15)
    s.index.freq = "MS"
    return s


def _make_cps_data(
    n: int = 24,
    E: float = 100.0,
    U: float = 10.0,
    N: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Balanced CPS flows + stocks with exact known proportions."""
    idx = _make_monthly_dates(n)
    stocks = pd.DataFrame({"E": [E] * n, "U": [U] * n, "N": [N] * n}, index=idx)
    # flows: rows sum exactly to origin stock
    # E-row: EE=90, EU=5, EN=5   (sum=100=E)
    # U-row: UE=5,  UU=4, UN=1   (sum=10=U)
    # N-row: NE=5,  NU=2, NN=43  (sum=50=N)
    flows = pd.DataFrame(
        {
            "EE": [90.0] * n,
            "EU": [5.0] * n,
            "EN": [5.0] * n,
            "UE": [5.0] * n,
            "UU": [4.0] * n,
            "UN": [1.0] * n,
            "NE": [5.0] * n,
            "NU": [2.0] * n,
            "NN": [43.0] * n,
        },
        index=idx,
    )
    return flows, stocks


# ===========================================================================
# 1. TransitionPanel.matrix_at
# ===========================================================================

class TestTransitionPanelMatrixAt:
    """Tests for the matrix_at() method of TransitionPanel."""

    def _get_panel(self) -> TransitionPanel:
        return transitions_from_shimer(_constant_urate(0.06, 12))

    def test_matrix_at_returns_ndarray(self):
        """matrix_at returns a numpy ndarray."""
        panel = self._get_panel()
        date = panel.monthly.index[0]
        result = panel.matrix_at(date)
        assert isinstance(result, np.ndarray)

    def test_matrix_at_shape_3x3(self):
        """matrix_at returns a (3, 3) matrix."""
        panel = self._get_panel()
        date = panel.monthly.index[0]
        result = panel.matrix_at(date)
        assert result.shape == (3, 3)

    def test_matrix_at_row_sums_to_one(self):
        """Each row of the returned matrix sums to 1 (stochastic matrix)."""
        panel = self._get_panel()
        for date in panel.monthly.index[:5]:
            mat = panel.matrix_at(date)
            np.testing.assert_allclose(mat.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-12)

    def test_matrix_at_values_match_row(self):
        """matrix_at returns values that exactly match the corresponding DataFrame row."""
        panel = self._get_panel()
        date = panel.monthly.index[2]
        mat = panel.matrix_at(date)
        row = panel.monthly.loc[date]
        expected = np.array(
            [
                [row["p_EE"], row["p_EU"], row["p_EN"]],
                [row["p_UE"], row["p_UU"], row["p_UN"]],
                [row["p_NE"], row["p_NU"], row["p_NN"]],
            ]
        )
        np.testing.assert_array_equal(mat, expected)

    def test_matrix_at_non_negative(self):
        """All entries are non-negative (probabilities)."""
        panel = self._get_panel()
        for date in panel.monthly.index:
            mat = panel.matrix_at(date)
            assert np.all(mat >= 0.0)

    def test_states_property(self):
        """states property returns the ('E', 'U', 'N') tuple."""
        panel = self._get_panel()
        assert panel.states == ("E", "U", "N")


# ===========================================================================
# 2. transitions_from_cps_flows
# ===========================================================================

class TestTransitionsFromCpsFlows:
    """Tests for the transitions_from_cps_flows function."""

    def test_returns_transition_panel(self):
        """Return type is TransitionPanel."""
        flows, stocks = _make_cps_data()
        result = transitions_from_cps_flows(flows, stocks)
        assert isinstance(result, TransitionPanel)

    def test_monthly_columns(self):
        """Monthly DataFrame has the 9 expected p_XY columns."""
        flows, stocks = _make_cps_data()
        panel = transitions_from_cps_flows(flows, stocks)
        expected_cols = {
            "p_EE", "p_EU", "p_EN",
            "p_UE", "p_UU", "p_UN",
            "p_NE", "p_NU", "p_NN",
        }
        assert expected_cols.issubset(set(panel.monthly.columns))

    def test_monthly_row_sums_to_one_after_normalization(self):
        """After row-normalization, each state row sums to 1."""
        # Deliberately unbalanced flows to exercise normalization
        n = 12
        idx = _make_monthly_dates(n)
        stocks = pd.DataFrame({"E": [100.0] * n, "U": [10.0] * n, "N": [50.0] * n}, index=idx)
        flows = pd.DataFrame(
            {
                "EE": [91.0] * n,  # off by +1
                "EU": [5.0] * n,
                "EN": [5.0] * n,
                "UE": [4.0] * n,
                "UU": [4.0] * n,
                "UN": [1.0] * n,
                "NE": [5.0] * n,
                "NU": [2.0] * n,
                "NN": [43.0] * n,
            },
            index=idx,
        )
        panel = transitions_from_cps_flows(flows, stocks)
        for origin in ["E", "U", "N"]:
            cols = [f"p_{origin}{d}" for d in ["E", "U", "N"]]
            row_sums = panel.monthly[cols].sum(axis=1)
            np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-12,
                                       err_msg=f"Row {origin} sums not 1")

    def test_probabilities_in_unit_interval(self):
        """All transition probabilities are in [0, 1]."""
        flows, stocks = _make_cps_data()
        panel = transitions_from_cps_flows(flows, stocks)
        vals = panel.monthly.values
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0 + 1e-12)

    def test_known_value_transition_probabilities(self):
        """With balanced flows (E=100, EU=5 out of 100), p_EU = 5/100 = 0.05."""
        flows, stocks = _make_cps_data(n=12)
        panel = transitions_from_cps_flows(flows, stocks)
        # p_EU = (EU_flow / E_lag) / row_sum; since flows sum exactly to
        # origin stock the row_sum should be 1 (no rounding), so p_EU = 5/100
        np.testing.assert_allclose(
            panel.monthly["p_EU"].values, 5.0 / 100.0, atol=1e-10
        )

    def test_raises_on_missing_flow_columns(self):
        """ValueError if flow DataFrame is missing required columns."""
        idx = _make_monthly_dates(6)
        flows_bad = pd.DataFrame({"EE": [90.0] * 6}, index=idx)
        stocks = pd.DataFrame({"E": [100.0] * 6, "U": [10.0] * 6, "N": [50.0] * 6}, index=idx)
        with pytest.raises(ValueError, match="Missing columns"):
            transitions_from_cps_flows(flows_bad, stocks)

    def test_raises_on_missing_stock_columns(self):
        """ValueError if stocks DataFrame is missing required columns."""
        flows, _ = _make_cps_data(n=6)
        stocks_bad = pd.DataFrame({"E": [100.0] * 6}, index=_make_monthly_dates(6))
        with pytest.raises(ValueError, match="Missing columns"):
            transitions_from_cps_flows(flows, stocks_bad)

    def test_quarterly_is_dataframe(self):
        """Quarterly attribute is a DataFrame."""
        flows, stocks = _make_cps_data(n=12)
        panel = transitions_from_cps_flows(flows, stocks)
        assert isinstance(panel.quarterly, pd.DataFrame)

    def test_quarterly_row_sums_to_one(self):
        """Quarterly chained matrix rows also sum to 1."""
        flows, stocks = _make_cps_data(n=12)
        panel = transitions_from_cps_flows(flows, stocks)
        if not panel.quarterly.empty:
            for origin in ["E", "U", "N"]:
                cols = [f"p_{origin}{d}" for d in ["E", "U", "N"]]
                row_sums = panel.quarterly[cols].sum(axis=1)
                np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_stocks_attached_to_panel(self):
        """TransitionPanel.stocks is the stocks DataFrame passed in."""
        flows, stocks = _make_cps_data(n=12)
        panel = transitions_from_cps_flows(flows, stocks)
        pd.testing.assert_frame_equal(panel.stocks, stocks)

    def test_monthly_has_fewer_rows_due_to_lag(self):
        """Monthly DataFrame drops first row (NaN from lag) so len < input length."""
        flows, stocks = _make_cps_data(n=12)
        panel = transitions_from_cps_flows(flows, stocks)
        # First row is NaN because stock is lagged; dropna() removes it.
        assert len(panel.monthly) < len(flows)


# ===========================================================================
# 3. transitions_from_shimer
# ===========================================================================

class TestTransitionsFromShimer:
    """Tests for the transitions_from_shimer function."""

    def test_returns_transition_panel(self):
        """Return type is TransitionPanel."""
        result = transitions_from_shimer(_constant_urate())
        assert isinstance(result, TransitionPanel)

    def test_monthly_columns_present(self):
        """Monthly DataFrame has all 9 p_XY columns."""
        panel = transitions_from_shimer(_constant_urate())
        expected_cols = {
            "p_EE", "p_EU", "p_EN",
            "p_UE", "p_UU", "p_UN",
            "p_NE", "p_NU", "p_NN",
        }
        assert expected_cols.issubset(set(panel.monthly.columns))

    def test_row_sums_to_one(self):
        """Each row of the monthly panel sums to 1."""
        panel = transitions_from_shimer(_varying_urate())
        for origin in ["E", "U", "N"]:
            cols = [f"p_{origin}{d}" for d in ["E", "U", "N"]]
            row_sums = panel.monthly[cols].sum(axis=1)
            np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-12)

    def test_probabilities_in_unit_interval(self):
        """All transition probabilities are in [0, 1]."""
        panel = transitions_from_shimer(_varying_urate())
        vals = panel.monthly.values
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0 + 1e-12)

    def test_percent_urate_same_as_decimal(self):
        """Percent-scale urate (e.g. 6.0) gives the same result as decimal (0.06)."""
        u_dec = _constant_urate(0.07, 24)
        u_pct = u_dec * 100.0  # convert to percent

        panel_dec = transitions_from_shimer(u_dec)
        panel_pct = transitions_from_shimer(u_pct)

        # Both should have the same shape and approximately same values
        assert panel_dec.monthly.shape == panel_pct.monthly.shape
        np.testing.assert_allclose(
            panel_dec.monthly.values,
            panel_pct.monthly.values,
            atol=1e-10,
        )

    def test_n_row_is_identity(self):
        """Shimer 2-state: N-row is always (0, 0, 1) — N stays in N."""
        panel = transitions_from_shimer(_varying_urate())
        np.testing.assert_allclose(panel.monthly["p_NE"].values, 0.0, atol=1e-12)
        np.testing.assert_allclose(panel.monthly["p_NU"].values, 0.0, atol=1e-12)
        np.testing.assert_allclose(panel.monthly["p_NN"].values, 1.0, atol=1e-12)

    def test_en_and_un_are_zero(self):
        """Shimer 2-state: E and U never transition to N."""
        panel = transitions_from_shimer(_varying_urate())
        np.testing.assert_allclose(panel.monthly["p_EN"].values, 0.0, atol=1e-12)
        np.testing.assert_allclose(panel.monthly["p_UN"].values, 0.0, atol=1e-12)

    def test_ee_plus_eu_equals_one(self):
        """In 2-state Shimer, p_EE + p_EU = 1 (E stays employed or separates)."""
        panel = transitions_from_shimer(_varying_urate())
        row_sums = panel.monthly["p_EE"] + panel.monthly["p_EU"]
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-12)

    def test_ue_plus_uu_equals_one(self):
        """In 2-state Shimer, p_UE + p_UU = 1 (U finds a job or stays)."""
        panel = transitions_from_shimer(_varying_urate())
        row_sums = panel.monthly["p_UE"] + panel.monthly["p_UU"]
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-12)

    def test_quarterly_produced(self):
        """Quarterly attribute is non-empty when enough months exist."""
        panel = transitions_from_shimer(_varying_urate(n=24))
        assert not panel.quarterly.empty

    def test_quarterly_row_sums_to_one(self):
        """Quarterly rows also sum to 1 (product of stochastic matrices)."""
        panel = transitions_from_shimer(_varying_urate(n=24))
        for origin in ["E", "U", "N"]:
            cols = [f"p_{origin}{d}" for d in ["E", "U", "N"]]
            row_sums = panel.quarterly[cols].sum(axis=1)
            np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)

    def test_stocks_e_plus_u_equals_one_approx(self):
        """Shimer stocks: E = 1-u and U = u so E + U ≈ 1."""
        u = _constant_urate(0.08, 24)
        panel = transitions_from_shimer(u)
        eu_sum = panel.stocks["E"] + panel.stocks["U"]
        # Allow for minor differences from the dropna alignment
        np.testing.assert_allclose(eu_sum.values, 1.0, atol=1e-12)

    def test_stocks_n_is_nan(self):
        """Shimer 2-state: N stock is NaN (no N data)."""
        panel = transitions_from_shimer(_constant_urate())
        assert panel.stocks["N"].isna().all()

    def test_monthly_index_aligned_with_input(self):
        """Monthly index is a subset of the input urate index."""
        u = _varying_urate(n=36)
        panel = transitions_from_shimer(u)
        # Monthly index must be contained in the urate index
        assert set(panel.monthly.index).issubset(set(u.index))

    def test_dropna_removes_last_row(self):
        """Shimer uses shift(-1), so last row has NaN and is dropped."""
        u = _constant_urate(0.06, 10)
        panel = transitions_from_shimer(u)
        # Dropna from shift(-1) removes the last obs
        assert len(panel.monthly) == len(u) - 1


# ===========================================================================
# 4. _quarterly_chain (internal)
# ===========================================================================

class TestQuarterlyChain:
    """Tests for the internal _quarterly_chain aggregation helper."""

    def _identity_monthly(self, n: int) -> pd.DataFrame:
        """Monthly DataFrame where every matrix is the identity."""
        idx = _make_monthly_dates(n)
        data = {
            "p_EE": 1.0, "p_EU": 0.0, "p_EN": 0.0,
            "p_UE": 0.0, "p_UU": 1.0, "p_UN": 0.0,
            "p_NE": 0.0, "p_NU": 0.0, "p_NN": 1.0,
        }
        return pd.DataFrame({k: [v] * n for k, v in data.items()}, index=idx)

    def test_returns_dataframe(self):
        """_quarterly_chain returns a DataFrame."""
        p = self._identity_monthly(6)
        result = _quarterly_chain(p)
        assert isinstance(result, pd.DataFrame)

    def test_empty_on_incomplete_quarter(self):
        """Fewer than 3 months in any quarter → empty DataFrame."""
        p = self._identity_monthly(2)  # only 2 months: Jan + Feb (no complete quarter)
        result = _quarterly_chain(p)
        assert result.empty

    def test_one_complete_quarter(self):
        """Exactly 3 months → one quarterly row."""
        p = self._identity_monthly(3)
        result = _quarterly_chain(p)
        assert len(result) == 1

    def test_identity_chain_is_identity(self):
        """Product of identity matrices is the identity matrix."""
        p = self._identity_monthly(6)
        result = _quarterly_chain(p)
        # Both Q1 and Q2 quarterly matrices should be identity
        for _, row in result.iterrows():
            np.testing.assert_allclose(row["p_EE"], 1.0, atol=1e-12)
            np.testing.assert_allclose(row["p_EU"], 0.0, atol=1e-12)
            np.testing.assert_allclose(row["p_EU"], 0.0, atol=1e-12)
            np.testing.assert_allclose(row["p_UU"], 1.0, atol=1e-12)
            np.testing.assert_allclose(row["p_NN"], 1.0, atol=1e-12)

    def test_chain_known_value(self):
        """Chained product matches manual matrix multiplication."""
        u = pd.Series(
            [0.05, 0.06, 0.07, 0.06, 0.05, 0.06],
            index=_make_monthly_dates(6),
        )
        u.index.freq = "MS"
        panel = transitions_from_shimer(u)
        # Manual chain: first 3 months
        m1 = panel.matrix_at(panel.monthly.index[0])
        m2 = panel.matrix_at(panel.monthly.index[1])
        m3 = panel.matrix_at(panel.monthly.index[2])
        manual_EE = (m1 @ m2 @ m3)[0, 0]
        q_idx = panel.quarterly.index[0]
        np.testing.assert_allclose(
            panel.quarterly.loc[q_idx, "p_EE"], manual_EE, atol=1e-14
        )

    def test_quarterly_index_is_quarter_end(self):
        """Quarterly index uses quarter-end dates."""
        p = self._identity_monthly(6)
        result = _quarterly_chain(p)
        # For Jan–Mar 2010 the quarter-end is 2010-03-31
        qe = pd.Timestamp("2010-03-31")
        assert qe in result.index

    def test_quarterly_row_sums_to_one(self):
        """Quarterly transition rows sum to 1."""
        flows, stocks = _make_cps_data(n=12)
        panel = transitions_from_cps_flows(flows, stocks)
        if not panel.quarterly.empty:
            for origin in ["E", "U", "N"]:
                cols = [f"p_{origin}{d}" for d in ["E", "U", "N"]]
                row_sums = panel.quarterly[cols].sum(axis=1)
                np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-10)


# ===========================================================================
# 5. supply_demand_decomposition
# ===========================================================================

class TestSupplyDemandDecomposition:
    """Tests for supply_demand_decomposition."""

    def test_returns_dataframe(self):
        """Return type is DataFrame."""
        panel = transitions_from_shimer(_varying_urate())
        result = supply_demand_decomposition(panel)
        assert isinstance(result, pd.DataFrame)

    def test_2state_columns(self):
        """2-state Shimer: output has all 5 columns including supply columns."""
        panel = transitions_from_shimer(_varying_urate())
        result = supply_demand_decomposition(panel)
        expected_cols = {"demand_in", "demand_out", "supply_in", "supply_out", "net_E_change"}
        assert expected_cols.issubset(set(result.columns))

    def test_2state_supply_columns_are_nan(self):
        """2-state path: supply_in and supply_out are all NaN (no N stock)."""
        panel = transitions_from_shimer(_varying_urate())
        result = supply_demand_decomposition(panel)
        assert result["supply_in"].isna().all()
        assert result["supply_out"].isna().all()

    def test_2state_demand_in_not_all_nan(self):
        """2-state path: demand_in (hires from U) is not entirely NaN."""
        panel = transitions_from_shimer(_varying_urate(n=48))
        result = supply_demand_decomposition(panel)
        assert not result["demand_in"].isna().all()

    def test_3state_columns(self):
        """3-state (CPS-flow) path: all 5 columns present."""
        flows, stocks = _make_cps_data(n=12)
        panel = transitions_from_cps_flows(flows, stocks)
        result = supply_demand_decomposition(panel)
        expected_cols = {"demand_in", "demand_out", "supply_in", "supply_out", "net_E_change"}
        assert expected_cols.issubset(set(result.columns))

    def test_3state_supply_not_nan(self):
        """3-state path: supply_in and supply_out have non-NaN values."""
        flows, stocks = _make_cps_data(n=12)
        panel = transitions_from_cps_flows(flows, stocks)
        result = supply_demand_decomposition(panel)
        assert not result["supply_in"].isna().all()
        assert not result["supply_out"].isna().all()

    def test_3state_known_value_balanced_flows(self):
        """With balanced flows (each flow = exact share of origin stock),
        net_E_change = demand_in + supply_in - demand_out - supply_out.

        Manually: demand_in = p_UE * U_lag = 0.5 * 10 = 5
                  demand_out = p_EU * E_lag = 0.05 * 100 = 5
                  supply_in = p_NE * N_lag = 0.1 * 50 = 5
                  supply_out = p_EN * E_lag = 0.05 * 100 = 5
                  net_E_change = 5+5-5-5 = 0
        """
        flows, stocks = _make_cps_data(n=12, E=100.0, U=10.0, N=50.0)
        panel = transitions_from_cps_flows(flows, stocks)
        result = supply_demand_decomposition(panel)
        non_nan = result["net_E_change"].dropna()
        np.testing.assert_allclose(non_nan.values, 0.0, atol=1e-10)

    def test_3state_known_demand_in(self):
        """demand_in = p_UE * U_lag should be 5.0 for the balanced fixture."""
        flows, stocks = _make_cps_data(n=12, E=100.0, U=10.0, N=50.0)
        panel = transitions_from_cps_flows(flows, stocks)
        result = supply_demand_decomposition(panel)
        non_nan = result["demand_in"].dropna()
        # p_UE = 5/10 = 0.5; U_lag = 10 → demand_in = 5.0
        np.testing.assert_allclose(non_nan.values, 5.0, atol=1e-10)

    def test_3state_known_supply_in(self):
        """supply_in = p_NE * N_lag should be 5.0 for the balanced fixture."""
        flows, stocks = _make_cps_data(n=12, E=100.0, U=10.0, N=50.0)
        panel = transitions_from_cps_flows(flows, stocks)
        result = supply_demand_decomposition(panel)
        non_nan = result["supply_in"].dropna()
        # p_NE = 5/50 = 0.1; N_lag = 50 → supply_in = 5.0
        np.testing.assert_allclose(non_nan.values, 5.0, atol=1e-10)

    def test_net_e_change_additive_identity(self):
        """net_E_change == demand_in + supply_in - demand_out - supply_out."""
        flows, stocks = _make_cps_data(n=12)
        panel = transitions_from_cps_flows(flows, stocks)
        result = supply_demand_decomposition(panel).dropna()
        manual_net = (
            result["demand_in"] + result["supply_in"]
            - result["demand_out"] - result["supply_out"]
        )
        pd.testing.assert_series_equal(result["net_E_change"], manual_net, check_names=False)


# ===========================================================================
# 6. transition_shock_response
# ===========================================================================

class TestTransitionShockResponse:
    """Tests for the LP-style IRF helper transition_shock_response."""

    def _get_panel_and_shock(self, n: int = 60, seed: int = 0) -> tuple:
        """Helper: shimer panel + aligned shock series."""
        u = _varying_urate(n=n, seed=seed)
        panel = transitions_from_shimer(u)
        rng = np.random.default_rng(seed)
        shock = pd.Series(rng.standard_normal(n), index=u.index, name="shock")
        shock.index.freq = "MS"
        return panel, shock

    def test_returns_dataframe(self):
        """Return type is DataFrame."""
        panel, shock = self._get_panel_and_shock()
        result = transition_shock_response(panel, shock, horizons=[0], transitions=["p_UE"])
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self):
        """Output has columns: transition, h, beta, se, t, n."""
        panel, shock = self._get_panel_and_shock()
        result = transition_shock_response(panel, shock, horizons=[0], transitions=["p_UE"])
        assert set(result.columns) == {"transition", "h", "beta", "se", "t", "n"}

    def test_one_row_per_transition_per_horizon(self):
        """One row per (transition, horizon) combination."""
        panel, shock = self._get_panel_and_shock()
        horizons = [0, 1, 2, 3]
        transitions = ["p_UE", "p_EU"]
        result = transition_shock_response(panel, shock, horizons=horizons,
                                          transitions=transitions)
        assert len(result) == len(horizons) * len(transitions)

    def test_transition_column_values(self):
        """transition column contains only the requested transition names."""
        panel, shock = self._get_panel_and_shock()
        transitions = ["p_UE", "p_EU"]
        result = transition_shock_response(panel, shock, horizons=[0, 1],
                                          transitions=transitions)
        assert set(result["transition"].unique()) == set(transitions)

    def test_h_column_values(self):
        """h column contains all requested horizon values."""
        panel, shock = self._get_panel_and_shock()
        horizons = list(range(5))
        result = transition_shock_response(panel, shock, horizons=horizons,
                                          transitions=["p_UE"])
        assert set(result["h"].unique()) == set(horizons)

    def test_beta_se_are_float(self):
        """beta and se are float-typed."""
        panel, shock = self._get_panel_and_shock()
        result = transition_shock_response(panel, shock, horizons=[0], transitions=["p_UE"])
        assert result["beta"].dtype == float
        assert result["se"].dtype == float

    def test_n_column_positive(self):
        """n column (number of observations) is positive for real transitions."""
        panel, shock = self._get_panel_and_shock()
        result = transition_shock_response(panel, shock, horizons=[0], transitions=["p_UE"])
        assert (result["n"] > 0).all()

    def test_se_non_negative(self):
        """Standard errors are non-negative."""
        panel, shock = self._get_panel_and_shock()
        result = transition_shock_response(panel, shock, horizons=[0, 1, 2],
                                          transitions=["p_UE", "p_EU"])
        assert (result["se"] >= 0.0).all()

    def test_missing_transition_skipped(self):
        """Transition not in the panel produces an empty DataFrame (skipped silently)."""
        panel, shock = self._get_panel_and_shock()
        result = transition_shock_response(panel, shock, horizons=[0],
                                          transitions=["p_NONEXISTENT"])
        assert result.empty

    def test_n_decreases_at_longer_horizons(self):
        """At longer horizons, observations are lost (shift fills NaN), so n decreases."""
        panel, shock = self._get_panel_and_shock(n=80)
        result = transition_shock_response(panel, shock, horizons=[0, 5, 10],
                                          transitions=["p_UE"])
        n_by_h = result.set_index("h")["n"]
        # n at h=10 must be strictly less than n at h=0
        assert n_by_h.loc[10] < n_by_h.loc[0]

    def test_multiple_transitions_returned(self):
        """Multiple transitions produce rows for each."""
        panel, shock = self._get_panel_and_shock()
        transitions = ["p_UE", "p_EU", "p_EN", "p_NE", "p_NU", "p_UN"]
        result = transition_shock_response(panel, shock, horizons=[0], transitions=transitions)
        assert set(result["transition"].unique()) == set(transitions)

    def test_ols_exception_path_yields_nan_row(self, monkeypatch):
        """If OLS.fit raises, the except branch appends a NaN row with n=0."""
        import statsmodels.api as sm

        panel, shock = self._get_panel_and_shock()

        def bad_fit(self, **kwargs):
            raise RuntimeError("forced OLS failure")

        monkeypatch.setattr(sm.OLS, "fit", bad_fit)
        result = transition_shock_response(panel, shock, horizons=[0], transitions=["p_UE"])
        assert len(result) == 1
        assert np.isnan(result["beta"].iloc[0])
        assert np.isnan(result["se"].iloc[0])
        assert result["n"].iloc[0] == 0

    @pytest.mark.slow
    def test_quarterly_shock_uses_quarterly_panel(self):
        """A quarterly-frequency shock leads to using panel.quarterly, not monthly."""
        u = _varying_urate(n=60, seed=5)
        panel = transitions_from_shimer(u)

        # Build a quarterly shock aligned to the quarterly panel index
        q_idx = panel.quarterly.index
        rng = np.random.default_rng(5)
        shock_q = pd.Series(rng.standard_normal(len(q_idx)), index=q_idx, name="shock")
        # quarterly index has QE-DEC inferred_freq → will use panel.quarterly
        assert shock_q.index.inferred_freq not in ("M", "MS", "ME")

        result = transition_shock_response(panel, shock_q, horizons=[0, 1],
                                          transitions=["p_UE"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ===========================================================================
# 7. Module-level constants / public API
# ===========================================================================

class TestModuleConstants:
    """Tests for the module-level constant dictionaries."""

    def test_cps_flow_series_has_8_keys(self):
        """CPS_FLOW_SERIES has exactly 8 origin-destination pairs."""
        assert len(CPS_FLOW_SERIES) == 8

    def test_cps_flow_keys_are_two_letter_uppercase(self):
        """All CPS_FLOW_SERIES keys are two-letter uppercase strings."""
        for key in CPS_FLOW_SERIES:
            assert len(key) == 2
            assert key == key.upper()

    def test_cps_stock_series_has_3_keys(self):
        """CPS_STOCK_SERIES has exactly 3 state keys: E, U, N."""
        assert set(CPS_STOCK_SERIES.keys()) == {"E", "U", "N"}

    def test_cps_flow_values_are_strings(self):
        """All CPS_FLOW_SERIES values are FRED ID strings."""
        for v in CPS_FLOW_SERIES.values():
            assert isinstance(v, str)
            assert len(v) > 0

    def test_nn_not_in_cps_flow_series(self):
        """NN (N-to-N) is NOT in CPS_FLOW_SERIES (computed as residual)."""
        assert "NN" not in CPS_FLOW_SERIES
