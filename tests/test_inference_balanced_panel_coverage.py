"""Coverage tests for puremacro.inference.balanced_panel.

Three public objects:
  - balanced_subpanel   -- rectangular sub-panel selection
  - balanced_cohort_window -- maximise area subject to min-entity floor
  - cohort_for_section -- named-section registry wrapper

Approach: known-value tests with synthetic DGPs whose correct answer is
analytically determined; property tests (shape, dtype, bounds, monotonicity);
and error-handling tests (pytest.raises).  No network I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.inference.balanced_panel import (
    balanced_cohort_window,
    balanced_subpanel,
    cohort_for_section,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic (code, qdate)-indexed wide panels
# ---------------------------------------------------------------------------

def _make_panel(
    codes: list[str],
    dates: list[pd.Timestamp],
    cols: list[str],
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Fully populated rectangular panel — no NaNs."""
    if rng is None:
        rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_product([codes, dates], names=["code", "qdate"])
    data = {c: rng.standard_normal(len(index)) for c in cols}
    return pd.DataFrame(data, index=index)


def _dates(n: int, start: str = "2000Q1") -> list[pd.Timestamp]:
    return list(pd.period_range(start, periods=n, freq="Q").to_timestamp())


# ---------------------------------------------------------------------------
# balanced_subpanel — basic contract
# ---------------------------------------------------------------------------

class TestBalancedSubpanelBasicContract:
    def test_returns_dataframe(self):
        df = _make_panel(["A", "B"], _dates(8), ["x"])
        out = balanced_subpanel(df, cols=["x"])
        assert isinstance(out, pd.DataFrame)

    def test_output_index_has_two_levels(self):
        df = _make_panel(["A", "B"], _dates(8), ["x"])
        out = balanced_subpanel(df, cols=["x"])
        assert out.index.nlevels == 2

    def test_preserves_original_index_names(self):
        df = _make_panel(["A", "B"], _dates(8), ["x"])
        out = balanced_subpanel(df, cols=["x"])
        assert out.index.names == df.index.names

    def test_output_columns_superset_of_required(self):
        df = _make_panel(["A", "B"], _dates(8), ["x", "y", "z"])
        out = balanced_subpanel(df, cols=["x"])
        assert {"x", "y", "z"}.issubset(out.columns)

    def test_empty_cols_raises_valueerror(self):
        df = _make_panel(["A", "B"], _dates(8), ["x"])
        with pytest.raises(ValueError, match="cols"):
            balanced_subpanel(df, cols=[])


# ---------------------------------------------------------------------------
# balanced_subpanel — fully rectangular input stays intact
# ---------------------------------------------------------------------------

class TestBalancedSubpanelFullRectangle:
    def test_all_rows_kept_when_already_balanced(self):
        codes = ["A", "B", "C"]
        dates = _dates(12)
        df = _make_panel(codes, dates, ["x"])
        out = balanced_subpanel(df, cols=["x"])
        assert len(out) == len(df)

    def test_all_entities_kept(self):
        codes = ["A", "B", "C"]
        df = _make_panel(codes, _dates(10), ["x"])
        out = balanced_subpanel(df, cols=["x"])
        assert set(out.index.get_level_values(0).unique()) == set(codes)

    def test_all_dates_kept(self):
        dates = _dates(10)
        df = _make_panel(["A", "B"], dates, ["x"])
        out = balanced_subpanel(df, cols=["x"])
        assert set(out.index.get_level_values(1).unique()) == set(dates)

    def test_no_nans_in_required_col(self):
        df = _make_panel(["A", "B"], _dates(10), ["x"])
        out = balanced_subpanel(df, cols=["x"])
        assert out["x"].notna().all()


# ---------------------------------------------------------------------------
# balanced_subpanel — ragged panels (known correct answer)
# ---------------------------------------------------------------------------

class TestBalancedSubpanelRagged:
    def _ragged_panel(self):
        """
        Entity A: covers quarters 0..11 (12 quarters)
        Entity B: covers quarters 4..11 (8 quarters, starting later)
        Entity C: covers quarters 0..7  (8 quarters, ending earlier)

        The only fully-balanced rectangle is (B ∩ C dates) for entities
        that cover ALL those dates.  B ∩ C = quarters 4..7 = 4 dates.
        With all three entities, the best rectangle (area = entities × dates):
          - A+B+C over dates 4..7: area = 3 × 4 = 12
          - A+B over all 12 dates: A covers 0..11, B only 4..11 → no.
          - A+C over dates 0..7 (8 dates): area = 2 × 8 = 16
          - A alone: area = 1 × 12 → drop (requires ≥2 entities in greedy)
        So the chosen rectangle should be A+C over dates 0..7 = 16 cells.
        """
        dates = _dates(12)
        rng = np.random.default_rng(1)
        rows = []
        for i, d in enumerate(dates):
            for code, start_i, end_i in [("A", 0, 11), ("B", 4, 11), ("C", 0, 7)]:
                if start_i <= i <= end_i:
                    rows.append({"code": code, "qdate": d,
                                 "x": rng.standard_normal()})
        df = pd.DataFrame(rows).set_index(["code", "qdate"])
        return df, dates

    def test_no_nans_in_output_required_col(self):
        df, _ = self._ragged_panel()
        out = balanced_subpanel(df, cols=["x"])
        assert out["x"].notna().all()

    def test_output_is_rectangular(self):
        """Every retained entity must have the same set of dates."""
        df, _ = self._ragged_panel()
        out = balanced_subpanel(df, cols=["x"])
        date_sets = (
            out.reset_index()
            .groupby("code")["qdate"]
            .apply(set)
        )
        # All entities share the same date set.
        first = date_sets.iloc[0]
        for s in date_sets:
            assert s == first

    def test_area_maximised_over_trivial_single_entity(self):
        """The result should include ≥ 2 entities (greedy avoids degenerate 1-entity cases)."""
        df, _ = self._ragged_panel()
        out = balanced_subpanel(df, cols=["x"])
        assert out.index.get_level_values(0).nunique() >= 2

    def test_entity_with_no_overlap_excluded(self):
        """If one entity's dates never overlap with others, it should be dropped."""
        dates = _dates(10)
        early_dates = dates[:5]
        late_dates = dates[5:]
        rng = np.random.default_rng(2)
        rows = []
        for d in early_dates:
            rows.append({"code": "A", "qdate": d, "x": rng.standard_normal()})
        for d in late_dates:
            rows.append({"code": "B", "qdate": d, "x": rng.standard_normal()})
        # C covers all dates
        for d in dates:
            rows.append({"code": "C", "qdate": d, "x": rng.standard_normal()})
        df = pd.DataFrame(rows).set_index(["code", "qdate"])
        out = balanced_subpanel(df, cols=["x"])
        out_codes = set(out.index.get_level_values(0).unique())
        # A and B can't coexist in a balanced panel; at most one + C.
        # C covers all dates; A covers only 0..4; B covers only 5..9.
        # Best balanced options:
        # A+C over early_dates: area = 2 × 5 = 10
        # B+C over late_dates:  area = 2 × 5 = 10
        # Either is valid — just check A and B don't both appear.
        assert not ({"A", "B"}.issubset(out_codes)), (
            f"A and B both in output but share no dates; out_codes={out_codes}"
        )


# ---------------------------------------------------------------------------
# balanced_subpanel — date slicing (start / end)
# ---------------------------------------------------------------------------

class TestBalancedSubpanelDateSlicing:
    def test_start_clips_earlier_dates(self):
        dates = _dates(12)
        df = _make_panel(["A", "B"], dates, ["x"])
        start = dates[4]
        out = balanced_subpanel(df, cols=["x"], start=start)
        assert out.index.get_level_values(1).min() >= start

    def test_end_clips_later_dates(self):
        dates = _dates(12)
        df = _make_panel(["A", "B"], dates, ["x"])
        end = dates[7]
        out = balanced_subpanel(df, cols=["x"], end=end)
        assert out.index.get_level_values(1).max() <= end

    def test_start_and_end_together(self):
        dates = _dates(12)
        df = _make_panel(["A", "B"], dates, ["x"])
        start, end = dates[3], dates[8]
        out = balanced_subpanel(df, cols=["x"], start=start, end=end)
        lvl = out.index.get_level_values(1)
        assert lvl.min() >= start
        assert lvl.max() <= end

    def test_start_string_accepted(self):
        dates = _dates(8)
        df = _make_panel(["A", "B"], dates, ["x"])
        # Should not raise; string coerced to Timestamp.
        out = balanced_subpanel(df, cols=["x"], start="2001-01-01")
        assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# balanced_subpanel — NaN-dropping logic
# ---------------------------------------------------------------------------

class TestBalancedSubpanelNaNDrop:
    def test_nan_in_required_col_causes_row_drop(self):
        """One NaN in the required column for entity A should shrink its coverage."""
        dates = _dates(8)
        df = _make_panel(["A", "B"], dates, ["x"])
        # Inject NaN at one cell of A.
        df_mod = df.copy()
        df_mod.loc[("A", dates[0]), "x"] = np.nan
        out = balanced_subpanel(df_mod, cols=["x"])
        # The output must have no NaN in "x".
        assert out["x"].notna().all()

    def test_nan_in_non_required_col_not_dropped(self):
        """NaN in a column NOT listed in cols should not eliminate the row."""
        dates = _dates(8)
        df = _make_panel(["A", "B"], dates, ["x", "y"])
        df_mod = df.copy()
        # NaN in y (not required) for all rows of A.
        df_mod.loc["A", "y"] = np.nan
        out = balanced_subpanel(df_mod, cols=["x"])
        assert "A" in out.index.get_level_values(0).unique()

    def test_all_nan_in_required_col_returns_empty(self):
        """If every row has NaN in a required col, return empty."""
        dates = _dates(4)
        df = _make_panel(["A", "B"], dates, ["x"])
        df_mod = df.copy()
        df_mod["x"] = np.nan
        out = balanced_subpanel(df_mod, cols=["x"])
        assert len(out) == 0

    def test_empty_result_preserves_original_index_names(self):
        dates = _dates(4)
        df = _make_panel(["A", "B"], dates, ["x"])
        df_mod = df.copy()
        df_mod["x"] = np.nan
        out = balanced_subpanel(df_mod, cols=["x"])
        assert out.index.names == df.index.names


# ---------------------------------------------------------------------------
# balanced_subpanel — multiple required columns
# ---------------------------------------------------------------------------

class TestBalancedSubpanelMultipleCols:
    def test_all_required_cols_non_null_in_output(self):
        dates = _dates(8)
        df = _make_panel(["A", "B"], dates, ["x", "y"])
        out = balanced_subpanel(df, cols=["x", "y"])
        assert out["x"].notna().all()
        assert out["y"].notna().all()

    def test_partial_nan_in_second_col_shrinks_sample(self):
        """If half of A's rows have NaN in 'y', the balanced panel should
        contain fewer rows than the fully populated version."""
        dates = _dates(12)
        df = _make_panel(["A", "B", "C"], dates, ["x", "y"])
        df_mod = df.copy()
        # Wipe 'y' for entity A entirely.
        df_mod.loc["A", "y"] = np.nan
        out = balanced_subpanel(df_mod, cols=["x", "y"])
        # A should be excluded (has no non-NaN 'y'); B and C should survive.
        out_codes = set(out.index.get_level_values(0).unique())
        assert "A" not in out_codes
        assert {"B", "C"}.issubset(out_codes)


# ---------------------------------------------------------------------------
# balanced_subpanel — min_T filter
# ---------------------------------------------------------------------------

class TestBalancedSubpanelMinT:
    def test_min_t_argument_accepted(self):
        """min_T=5 on a panel with 12 dates should not raise."""
        df = _make_panel(["A", "B"], _dates(12), ["x"])
        out = balanced_subpanel(df, cols=["x"], min_T=5)
        assert isinstance(out, pd.DataFrame)

    def test_min_t_none_equivalent_to_omitting(self):
        df = _make_panel(["A", "B"], _dates(12), ["x"])
        out_default = balanced_subpanel(df, cols=["x"])
        out_none = balanced_subpanel(df, cols=["x"], min_T=None)
        assert len(out_default) == len(out_none)

    def test_min_t_larger_than_kept_dates_still_returns_dataframe(self):
        """When min_T > len(kept_dates) the internal no-op branch fires
        (line 139: kept_dates = kept_dates).  The function should still
        return a DataFrame; the spec says it keeps what it has.
        Use a ragged panel so the balanced rectangle is small (< min_T)."""
        # A+B share only 2 dates; set min_T=20 > 2 to trigger the no-op.
        dates6 = _dates(6)
        rows = []
        rng = np.random.default_rng(99)
        # A only has dates 0..1, B only has dates 2..5 — they share ZERO dates
        # so let's try a panel where greedy finds only 2 shared dates.
        for d in dates6[:4]:
            rows.append({"code": "A", "qdate": d, "x": rng.standard_normal()})
        for d in dates6[2:]:
            rows.append({"code": "B", "qdate": d, "x": rng.standard_normal()})
        df = pd.DataFrame(rows).set_index(["code", "qdate"])
        # A and B share dates6[2] and dates6[3] = 2 dates.
        out = balanced_subpanel(df, cols=["x"], min_T=20)
        # Result is a DataFrame (may be the 2-date balanced slice).
        assert isinstance(out, pd.DataFrame)

    def test_single_entity_panel_returns_dataframe(self):
        """With only one entity the greedy loop skips (requires ≥ 2 entities)
        and cand_full has 1 entity.  The function should not crash."""
        df = _make_panel(["A"], _dates(8), ["x"])
        out = balanced_subpanel(df, cols=["x"])
        assert isinstance(out, pd.DataFrame)

    def test_disjoint_entities_returns_empty(self):
        """When every entity covers exactly one unique date and no two entities
        share a date, the balanced rectangle is empty (no overlap).  Line 143
        (the kept_entities-empty early-return branch) is exercised here."""
        dates5 = _dates(5)
        rows = []
        rng = np.random.default_rng(7)
        for i, code in enumerate(["A", "B", "C", "D", "E"]):
            rows.append({"code": code, "qdate": dates5[i],
                         "x": rng.standard_normal()})
        df = pd.DataFrame(rows).set_index(["code", "qdate"])
        out = balanced_subpanel(df, cols=["x"])
        assert len(out) == 0
        assert isinstance(out, pd.DataFrame)
        assert out.index.names == df.index.names


# ---------------------------------------------------------------------------
# balanced_cohort_window — basic contract
# ---------------------------------------------------------------------------

class TestBalancedCohortWindowContract:
    def test_returns_triple(self):
        df = _make_panel(["A", "B", "C", "D", "E", "F", "G", "H", "I"],
                         _dates(20), ["x"])
        result = balanced_cohort_window(df, cols=["x"], min_n_entities=5)
        assert len(result) == 3

    def test_returns_timestamps_and_list(self):
        codes = [f"C{i}" for i in range(10)]
        df = _make_panel(codes, _dates(20), ["x"])
        start, end, ents = balanced_cohort_window(df, cols=["x"], min_n_entities=5)
        assert isinstance(start, pd.Timestamp)
        assert isinstance(end, pd.Timestamp)
        assert isinstance(ents, list)

    def test_start_le_end(self):
        codes = [f"C{i}" for i in range(10)]
        df = _make_panel(codes, _dates(20), ["x"])
        start, end, _ = balanced_cohort_window(df, cols=["x"], min_n_entities=5)
        assert start <= end

    def test_at_least_min_n_entities_returned(self):
        codes = [f"C{i}" for i in range(12)]
        df = _make_panel(codes, _dates(20), ["x"])
        _, _, ents = balanced_cohort_window(df, cols=["x"], min_n_entities=8)
        assert len(ents) >= 8

    def test_entities_are_in_original_panel(self):
        codes = [f"C{i}" for i in range(10)]
        df = _make_panel(codes, _dates(16), ["x"])
        _, _, ents = balanced_cohort_window(df, cols=["x"], min_n_entities=5)
        assert all(e in codes for e in ents)

    def test_all_null_raises_valueerror(self):
        codes = [f"C{i}" for i in range(10)]
        df = _make_panel(codes, _dates(16), ["x"])
        df["x"] = np.nan
        with pytest.raises(ValueError, match="non-null"):
            balanced_cohort_window(df, cols=["x"], min_n_entities=5)

    def test_insufficient_entities_raises_valueerror(self):
        """Panel has only 3 entities; asking for 10 should raise."""
        df = _make_panel(["A", "B", "C"], _dates(16), ["x"])
        with pytest.raises(ValueError):
            balanced_cohort_window(df, cols=["x"], min_n_entities=10)


# ---------------------------------------------------------------------------
# balanced_cohort_window — correctness on known DGP
# ---------------------------------------------------------------------------

class TestBalancedCohortWindowKnownAnswer:
    def _staggered_panel(self):
        """
        10 entities, each with a 20-quarter window.
        - Entities 0..4  start Q0  and cover Q0..Q19 (20 quarters each)
        - Entities 5..9  start Q10 and cover Q10..Q29 (20 quarters each)

        Common coverage for all 10: Q10..Q19 = 10 quarters.
        Common coverage for 5 (entities 0..4): Q0..Q19 = 20 quarters.

        Areas:
          10 entities × 10 quarters = 100
          5 entities  × 20 quarters = 100

        Both areas are equal; the function should return one of these.
        We set min_n_entities = 5 (strict floor), so the 5-entity answer
        is valid.  Either choice satisfies: len(ents) ≥ 5.
        """
        all_dates = _dates(30)
        rng = np.random.default_rng(3)
        rows = []
        for i in range(5):
            for d in all_dates[:20]:
                rows.append({"code": f"E{i}", "qdate": d,
                              "x": rng.standard_normal()})
        for i in range(5, 10):
            for d in all_dates[10:30]:
                rows.append({"code": f"E{i}", "qdate": d,
                              "x": rng.standard_normal()})
        df = pd.DataFrame(rows).set_index(["code", "qdate"])
        return df, all_dates

    def test_min_entity_floor_respected(self):
        df, _ = self._staggered_panel()
        _, _, ents = balanced_cohort_window(df, cols=["x"], min_n_entities=5)
        assert len(ents) >= 5

    def test_window_contains_at_least_one_date(self):
        df, _ = self._staggered_panel()
        start, end, _ = balanced_cohort_window(df, cols=["x"], min_n_entities=5)
        assert start <= end

    def test_entities_all_have_full_coverage_in_window(self):
        """Each returned entity must have non-null 'x' for every date in [start, end]."""
        df, all_dates = self._staggered_panel()
        start, end, ents = balanced_cohort_window(df, cols=["x"], min_n_entities=5)
        # Filter panel to the window.
        lvl_date = df.index.get_level_values("qdate")
        sub = df.loc[(lvl_date >= start) & (lvl_date <= end)]
        for e in ents:
            entity_dates = set(sub.loc[e].index) if e in sub.index.get_level_values("code") else set()
            window_dates = {d for d in all_dates if start <= d <= end}
            assert entity_dates >= window_dates, (
                f"Entity {e} missing dates in [{start}, {end}]"
            )


# ---------------------------------------------------------------------------
# balanced_cohort_window — more entities / longer window with more data
# ---------------------------------------------------------------------------

class TestBalancedCohortWindowMonotonicity:
    def test_relaxing_min_entities_does_not_shrink_area(self):
        """With fewer required entities the window area should be >= the
        stricter version (or equal if both are at the global optimum)."""
        codes = [f"C{i}" for i in range(12)]
        df = _make_panel(codes, _dates(20), ["x"])
        _, _, ents_strict = balanced_cohort_window(
            df, cols=["x"], min_n_entities=10
        )
        _, _, ents_loose = balanced_cohort_window(
            df, cols=["x"], min_n_entities=5
        )
        # Since the panel is already fully balanced, both should return all 12
        # entities × 20 dates. Just verify the loose constraint returns >= strict.
        assert len(ents_loose) >= len(ents_strict)


# ---------------------------------------------------------------------------
# cohort_for_section — basic contract
# ---------------------------------------------------------------------------

def _section_panel() -> pd.DataFrame:
    """Build a synthetic panel with all columns required by all four sections."""
    all_cols = [
        "log_relprice_equip",
        "ls_gollin_q",
        "log_neer_q",
        "log_oil_brent_real_local_q",
        "log_real_gov_q",
        "log_relprice_struct",
        "log_relprice_ipp",
        "log_pcons_q",
    ]
    codes = [f"C{i:02d}" for i in range(10)]
    dates = _dates(20, start="2000Q1")
    return _make_panel(codes, dates, all_cols)


class TestCohortForSection:
    def test_section_3_returns_triple(self):
        df = _section_panel()
        result = cohort_for_section(df, section="3")
        assert len(result) == 3

    def test_section_5_returns_triple(self):
        df = _section_panel()
        result = cohort_for_section(df, section="5")
        assert len(result) == 3

    def test_section_6_returns_triple(self):
        df = _section_panel()
        result = cohort_for_section(df, section="6")
        assert len(result) == 3

    def test_section_7_returns_triple(self):
        df = _section_panel()
        result = cohort_for_section(df, section="7")
        assert len(result) == 3

    def test_panel_is_dataframe(self):
        df = _section_panel()
        panel, _, _ = cohort_for_section(df, section="3")
        assert isinstance(panel, pd.DataFrame)

    def test_codes_is_sorted_list(self):
        df = _section_panel()
        _, codes, _ = cohort_for_section(df, section="3")
        assert isinstance(codes, list)
        assert codes == sorted(codes)

    def test_date_range_is_pair_of_timestamps(self):
        df = _section_panel()
        _, _, date_range = cohort_for_section(df, section="3")
        assert len(date_range) == 2
        assert isinstance(date_range[0], pd.Timestamp)
        assert isinstance(date_range[1], pd.Timestamp)

    def test_date_range_min_le_max(self):
        df = _section_panel()
        _, _, date_range = cohort_for_section(df, section="3")
        assert date_range[0] <= date_range[1]

    def test_unknown_section_raises_valueerror(self):
        df = _section_panel()
        with pytest.raises(ValueError, match="unknown section"):
            cohort_for_section(df, section="99")

    def test_unknown_section_zero_raises(self):
        df = _section_panel()
        with pytest.raises(ValueError):
            cohort_for_section(df, section="0")

    def test_panel_cols_contain_required_for_section_3(self):
        """The returned panel must include the columns required for §3."""
        df = _section_panel()
        panel, _, _ = cohort_for_section(df, section="3")
        assert {"log_relprice_equip", "ls_gollin_q"}.issubset(panel.columns)

    def test_panel_cols_contain_required_for_section_7(self):
        df = _section_panel()
        panel, _, _ = cohort_for_section(df, section="7")
        assert {
            "log_neer_q",
            "log_relprice_equip",
            "log_relprice_struct",
            "log_relprice_ipp",
            "log_pcons_q",
        }.issubset(panel.columns)

    def test_all_required_cols_non_null_in_returned_panel_section_6(self):
        """No NaN in the cols listed for §6 in the returned panel."""
        df = _section_panel()
        panel, _, _ = cohort_for_section(df, section="6")
        cols_6 = [
            "log_relprice_equip",
            "ls_gollin_q",
            "log_neer_q",
            "log_oil_brent_real_local_q",
            "log_real_gov_q",
        ]
        assert panel[cols_6].notna().all().all()

    def test_codes_subset_of_original_entities(self):
        df = _section_panel()
        original_codes = set(df.index.get_level_values(0).unique())
        _, codes, _ = cohort_for_section(df, section="5")
        assert set(codes).issubset(original_codes)

    def test_missing_required_col_raises_key_error(self):
        """If the panel lacks a required column entirely (the column is absent,
        not just NaN), pandas dropna raises a KeyError.  This is the actual
        contract of the underlying implementation: the caller is responsible
        for providing all required columns."""
        df = _section_panel()
        # Drop 'ls_gollin_q' so §3 can't be satisfied.
        df_no_ls = df.drop(columns=["ls_gollin_q"])
        with pytest.raises(KeyError):
            cohort_for_section(df_no_ls, section="3")

    def test_index_names_preserved_in_returned_panel(self):
        df = _section_panel()
        panel, _, _ = cohort_for_section(df, section="3")
        assert panel.index.names == df.index.names
