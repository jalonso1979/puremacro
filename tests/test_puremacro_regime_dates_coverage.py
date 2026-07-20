"""Coverage tests for puremacro.regime_dates.

This module (puremacro/regime_dates.py) was at 0% coverage before these tests.
It defines three regime-date schemes (REGIMES_GLOBAL, REGIMES_US,
REGIMES_US_POLITICAL), the ``Regime`` frozen dataclass, and the public
``assign_regime`` dispatcher.

Test strategy
-------------
- known_value: boundary dates for every scheme transition → verify exact
  inclusive/exclusive label assignment.
- property: output dtype/type, correct label sets per scope, contiguous
  non-overlapping regimes, REGIMES alias identity.
- Regime dataclass: instantiation, frozen enforcement, field access.
- assign_regime: all three scopes, default scope, OTHER for out-of-range dates,
  input coercion (list of strings, DatetimeIndex, Series).
- Error handling: unknown scope raises ValueError with informative message.
- No network I/O or external binaries: module is pure Pandas/dataclasses.
"""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.regime_dates import (
    REGIMES,
    REGIMES_GLOBAL,
    REGIMES_US,
    REGIMES_US_POLITICAL,
    Regime,
    _assign,
    assign_regime,
)


# ===========================================================================
# 1. Regime dataclass
# ===========================================================================

class TestRegimeDataclass:
    """Tests for the Regime frozen dataclass."""

    def test_instantiation_stores_fields(self):
        """Regime stores name, start, end correctly."""
        r = Regime(
            name="TestRegime",
            start=pd.Timestamp("2000-01-01"),
            end=pd.Timestamp("2000-12-31"),
        )
        assert r.name == "TestRegime"
        assert r.start == pd.Timestamp("2000-01-01")
        assert r.end == pd.Timestamp("2000-12-31")

    def test_frozen_name_raises(self):
        """Assigning to name raises FrozenInstanceError."""
        r = Regime("R", pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31"))
        with pytest.raises(Exception):  # FrozenInstanceError (dataclasses)
            r.name = "other"

    def test_frozen_start_raises(self):
        """Assigning to start raises FrozenInstanceError."""
        r = Regime("R", pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31"))
        with pytest.raises(Exception):
            r.start = pd.Timestamp("1990-01-01")

    def test_frozen_end_raises(self):
        """Assigning to end raises FrozenInstanceError."""
        r = Regime("R", pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31"))
        with pytest.raises(Exception):
            r.end = pd.Timestamp("2099-12-31")

    def test_repr_contains_name(self):
        """repr(Regime) includes the name field."""
        r = Regime("GR", pd.Timestamp("2007-01-01"), pd.Timestamp("2009-12-31"))
        assert "GR" in repr(r)

    def test_equality_by_value(self):
        """Two Regime objects with same fields compare equal (dataclass __eq__)."""
        r1 = Regime("A", pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31"))
        r2 = Regime("A", pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31"))
        assert r1 == r2

    def test_inequality_different_name(self):
        """Different name → not equal."""
        r1 = Regime("A", pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31"))
        r2 = Regime("B", pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31"))
        assert r1 != r2


# ===========================================================================
# 2. REGIMES_GLOBAL constants
# ===========================================================================

class TestRegimesGlobal:
    """Structural/property tests for the REGIMES_GLOBAL tuple."""

    def test_has_six_regimes(self):
        """REGIMES_GLOBAL has exactly 6 regime bins."""
        assert len(REGIMES_GLOBAL) == 6

    def test_names_are_correct(self):
        """All six regime names match the documented scheme."""
        names = [r.name for r in REGIMES_GLOBAL]
        assert names == ["PreGR", "GR", "EuroDebt", "Inter", "COVID", "PostCOVID"]

    def test_starts_are_timestamps(self):
        """Every start is a pd.Timestamp."""
        for r in REGIMES_GLOBAL:
            assert isinstance(r.start, pd.Timestamp)

    def test_ends_are_timestamps(self):
        """Every end is a pd.Timestamp."""
        for r in REGIMES_GLOBAL:
            assert isinstance(r.end, pd.Timestamp)

    def test_start_before_end(self):
        """start < end for every regime."""
        for r in REGIMES_GLOBAL:
            assert r.start < r.end, f"{r.name}: start not before end"

    def test_contiguous_no_gaps(self):
        """Consecutive regimes are adjacent (gap == 1 day)."""
        for i in range(len(REGIMES_GLOBAL) - 1):
            gap = (REGIMES_GLOBAL[i + 1].start - REGIMES_GLOBAL[i].end).days
            assert gap == 1, (
                f"Gap between {REGIMES_GLOBAL[i].name} and "
                f"{REGIMES_GLOBAL[i + 1].name} is {gap} days, expected 1"
            )

    def test_no_overlapping_intervals(self):
        """No two regimes overlap (second starts strictly after first ends)."""
        for i in range(len(REGIMES_GLOBAL) - 1):
            assert REGIMES_GLOBAL[i + 1].start > REGIMES_GLOBAL[i].end

    def test_first_regime_starts_1995(self):
        """First regime (PreGR) starts 1995-01-01."""
        assert REGIMES_GLOBAL[0].start == pd.Timestamp("1995-01-01")

    def test_last_regime_open_ended(self):
        """Last regime (PostCOVID) end is far-future sentinel."""
        assert REGIMES_GLOBAL[-1].end.year >= 2099


# ===========================================================================
# 3. REGIMES_US constants
# ===========================================================================

class TestRegimesUS:
    """Structural/property tests for the REGIMES_US tuple."""

    def test_has_five_regimes(self):
        """REGIMES_US has exactly 5 regime bins."""
        assert len(REGIMES_US) == 5

    def test_names_are_correct(self):
        """All five regime names match the documented US scheme."""
        names = [r.name for r in REGIMES_US]
        assert names == ["PreGR", "GR", "PreCOVID", "COVID", "PostCOVID"]

    def test_no_euro_debt_regime(self):
        """EuroDebt regime is absent (folded into PreCOVID for US-only analysis)."""
        names = [r.name for r in REGIMES_US]
        assert "EuroDebt" not in names

    def test_contiguous_no_gaps(self):
        """Consecutive regimes are adjacent (gap == 1 day)."""
        for i in range(len(REGIMES_US) - 1):
            gap = (REGIMES_US[i + 1].start - REGIMES_US[i].end).days
            assert gap == 1

    def test_pregr_matches_global(self):
        """US PreGR has same start/end as global PreGR."""
        global_pregr = REGIMES_GLOBAL[0]
        us_pregr = REGIMES_US[0]
        assert us_pregr.start == global_pregr.start
        assert us_pregr.end == global_pregr.end

    def test_gr_matches_global(self):
        """US GR has same start/end as global GR."""
        global_gr = REGIMES_GLOBAL[1]
        us_gr = REGIMES_US[1]
        assert us_gr.start == global_gr.start
        assert us_gr.end == global_gr.end


# ===========================================================================
# 4. REGIMES_US_POLITICAL constants
# ===========================================================================

class TestRegimesUSPolitical:
    """Structural/property tests for the REGIMES_US_POLITICAL tuple."""

    def test_has_five_regimes(self):
        """REGIMES_US_POLITICAL has exactly 5 presidency bins."""
        assert len(REGIMES_US_POLITICAL) == 5

    def test_names_are_correct(self):
        """All five presidency names match the documented scheme."""
        names = [r.name for r in REGIMES_US_POLITICAL]
        assert names == ["Bush43", "Obama", "Trump1", "Biden", "Trump2"]

    def test_contiguous_no_gaps(self):
        """Consecutive regimes are adjacent (gap == 1 day)."""
        for i in range(len(REGIMES_US_POLITICAL) - 1):
            gap = (REGIMES_US_POLITICAL[i + 1].start - REGIMES_US_POLITICAL[i].end).days
            assert gap == 1

    def test_bush43_starts_2001(self):
        """Bush43 starts 2001-01-01."""
        assert REGIMES_US_POLITICAL[0].start == pd.Timestamp("2001-01-01")

    def test_trump2_open_ended(self):
        """Trump2 end is far-future sentinel."""
        assert REGIMES_US_POLITICAL[-1].end.year >= 2099

    def test_no_overlap(self):
        """No two political regimes overlap."""
        for i in range(len(REGIMES_US_POLITICAL) - 1):
            assert REGIMES_US_POLITICAL[i + 1].start > REGIMES_US_POLITICAL[i].end


# ===========================================================================
# 5. REGIMES alias
# ===========================================================================

class TestRegimesAlias:
    """Tests for the backwards-compatible REGIMES alias."""

    def test_regimes_is_regimes_global(self):
        """REGIMES is the same object as REGIMES_GLOBAL (not a copy)."""
        assert REGIMES is REGIMES_GLOBAL

    def test_regimes_length(self):
        """REGIMES (alias) also has 6 entries."""
        assert len(REGIMES) == 6


# ===========================================================================
# 6. _assign (private helper)
# ===========================================================================

class TestAssignPrivate:
    """Tests for the private _assign helper."""

    def test_empty_series_returns_empty(self):
        """Empty input produces empty output Series."""
        result = _assign(pd.Series([], dtype="datetime64[ns]"), REGIMES_GLOBAL)
        assert len(result) == 0
        assert isinstance(result, pd.Series)

    def test_datetimeindex_input_accepted(self):
        """DatetimeIndex input is coerced correctly."""
        idx = pd.DatetimeIndex(["2008-01-01", "2015-01-01"])
        result = _assign(idx, REGIMES_GLOBAL)
        assert list(result) == ["GR", "Inter"]

    def test_series_input_accepted(self):
        """pd.Series of timestamps is accepted."""
        s = pd.Series(pd.to_datetime(["2008-01-01", "2015-01-01"]))
        result = _assign(s, REGIMES_GLOBAL)
        assert list(result) == ["GR", "Inter"]

    def test_output_index_reset(self):
        """Output index is RangeIndex (reset_index(drop=True) applied)."""
        s = pd.Series(pd.to_datetime(["2008-01-01"]), index=[99])
        result = _assign(s, REGIMES_GLOBAL)
        assert result.index[0] == 0

    def test_out_of_range_gets_other(self):
        """Dates before first regime start get 'OTHER'."""
        s = pd.Series(pd.to_datetime(["1990-01-01"]))
        result = _assign(s, REGIMES_GLOBAL)
        assert result.iloc[0] == "OTHER"

    def test_dtype_is_object(self):
        """Output dtype is 'object' (string labels)."""
        s = pd.Series(pd.to_datetime(["2008-01-01"]))
        result = _assign(s, REGIMES_GLOBAL)
        assert result.dtype == object


# ===========================================================================
# 7. assign_regime — global scope (default)
# ===========================================================================

class TestAssignRegimeGlobal:
    """Known-value boundary tests for assign_regime with scope='global'."""

    def _dates(self, *date_strings):
        return pd.DatetimeIndex(list(date_strings))

    def test_default_scope_is_global(self):
        """Calling assign_regime without scope= uses 'global' scheme."""
        dates = self._dates("2008-01-01")
        result_default = assign_regime(dates)
        result_explicit = assign_regime(dates, scope="global")
        assert list(result_default) == list(result_explicit)

    def test_before_pregr_is_other(self):
        """Date before 1995-01-01 → 'OTHER'."""
        result = assign_regime(self._dates("1994-12-31"), "global")
        assert result.iloc[0] == "OTHER"

    def test_pregr_first_day(self):
        """1995-01-01 → 'PreGR' (inclusive start)."""
        result = assign_regime(self._dates("1995-01-01"), "global")
        assert result.iloc[0] == "PreGR"

    def test_pregr_last_day(self):
        """2006-12-31 → 'PreGR' (inclusive end)."""
        result = assign_regime(self._dates("2006-12-31"), "global")
        assert result.iloc[0] == "PreGR"

    def test_gr_first_day(self):
        """2007-01-01 → 'GR' (inclusive start)."""
        result = assign_regime(self._dates("2007-01-01"), "global")
        assert result.iloc[0] == "GR"

    def test_gr_last_day(self):
        """2009-12-31 → 'GR' (inclusive end)."""
        result = assign_regime(self._dates("2009-12-31"), "global")
        assert result.iloc[0] == "GR"

    def test_eurodebt_first_day(self):
        """2010-01-01 → 'EuroDebt'."""
        result = assign_regime(self._dates("2010-01-01"), "global")
        assert result.iloc[0] == "EuroDebt"

    def test_eurodebt_last_day(self):
        """2013-12-31 → 'EuroDebt'."""
        result = assign_regime(self._dates("2013-12-31"), "global")
        assert result.iloc[0] == "EuroDebt"

    def test_inter_first_day(self):
        """2014-01-01 → 'Inter'."""
        result = assign_regime(self._dates("2014-01-01"), "global")
        assert result.iloc[0] == "Inter"

    def test_inter_last_day(self):
        """2019-12-31 → 'Inter'."""
        result = assign_regime(self._dates("2019-12-31"), "global")
        assert result.iloc[0] == "Inter"

    def test_covid_first_day(self):
        """2020-01-01 → 'COVID'."""
        result = assign_regime(self._dates("2020-01-01"), "global")
        assert result.iloc[0] == "COVID"

    def test_covid_last_day(self):
        """2021-12-31 → 'COVID'."""
        result = assign_regime(self._dates("2021-12-31"), "global")
        assert result.iloc[0] == "COVID"

    def test_postcovid_first_day(self):
        """2022-01-01 → 'PostCOVID'."""
        result = assign_regime(self._dates("2022-01-01"), "global")
        assert result.iloc[0] == "PostCOVID"

    def test_postcovid_far_future(self):
        """2040-01-01 → 'PostCOVID' (open-ended bin)."""
        result = assign_regime(self._dates("2040-01-01"), "global")
        assert result.iloc[0] == "PostCOVID"

    def test_output_is_series(self):
        """assign_regime returns a pd.Series."""
        result = assign_regime(self._dates("2008-01-01"), "global")
        assert isinstance(result, pd.Series)

    def test_output_dtype_is_object(self):
        """Output Series has object dtype (string labels)."""
        result = assign_regime(self._dates("2008-01-01"), "global")
        assert result.dtype == object

    def test_all_global_labels_attainable(self):
        """A broad date range produces all 7 unique labels (6 regimes + OTHER)."""
        dates = pd.date_range("1990-01-01", "2030-01-01", freq="QS")
        result = assign_regime(dates, "global")
        unique = set(result)
        assert unique == {"OTHER", "PreGR", "GR", "EuroDebt", "Inter", "COVID", "PostCOVID"}

    def test_length_matches_input(self):
        """Output length equals input length."""
        dates = pd.date_range("2000-01-01", periods=20, freq="QS")
        result = assign_regime(dates, "global")
        assert len(result) == len(dates)

    def test_single_date_series_input(self):
        """pd.Series of Timestamps works as input."""
        s = pd.Series(pd.to_datetime(["2008-06-01"]))
        result = assign_regime(s, "global")
        assert result.iloc[0] == "GR"

    def test_string_list_input_coerced(self):
        """Python list of date strings is accepted and converted."""
        result = assign_regime(["2008-01-01", "2015-01-01"], "global")
        assert list(result) == ["GR", "Inter"]


# ===========================================================================
# 8. assign_regime — US scope
# ===========================================================================

class TestAssignRegimeUS:
    """Known-value boundary tests for assign_regime with scope='us'."""

    def _dates(self, *date_strings):
        return pd.DatetimeIndex(list(date_strings))

    def test_before_pregr_is_other(self):
        """Date before 1995-01-01 → 'OTHER'."""
        result = assign_regime(self._dates("1994-12-31"), "us")
        assert result.iloc[0] == "OTHER"

    def test_pregr_first_day(self):
        """1995-01-01 → 'PreGR'."""
        result = assign_regime(self._dates("1995-01-01"), "us")
        assert result.iloc[0] == "PreGR"

    def test_pregr_last_day(self):
        """2006-12-31 → 'PreGR'."""
        result = assign_regime(self._dates("2006-12-31"), "us")
        assert result.iloc[0] == "PreGR"

    def test_gr_first_day(self):
        """2007-01-01 → 'GR'."""
        result = assign_regime(self._dates("2007-01-01"), "us")
        assert result.iloc[0] == "GR"

    def test_gr_last_day(self):
        """2009-12-31 → 'GR'."""
        result = assign_regime(self._dates("2009-12-31"), "us")
        assert result.iloc[0] == "GR"

    def test_precovid_first_day(self):
        """2010-01-01 → 'PreCOVID' (EuroDebt era folded into PreCOVID for US)."""
        result = assign_regime(self._dates("2010-01-01"), "us")
        assert result.iloc[0] == "PreCOVID"

    def test_precovid_last_day(self):
        """2019-12-31 → 'PreCOVID'."""
        result = assign_regime(self._dates("2019-12-31"), "us")
        assert result.iloc[0] == "PreCOVID"

    def test_no_eurodebt_label(self):
        """EuroDebt dates (2010-2013) get 'PreCOVID' in US scheme, not 'EuroDebt'."""
        result = assign_regime(self._dates("2012-06-01"), "us")
        assert result.iloc[0] == "PreCOVID"
        assert result.iloc[0] != "EuroDebt"

    def test_covid_first_day(self):
        """2020-01-01 → 'COVID'."""
        result = assign_regime(self._dates("2020-01-01"), "us")
        assert result.iloc[0] == "COVID"

    def test_covid_last_day(self):
        """2021-12-31 → 'COVID'."""
        result = assign_regime(self._dates("2021-12-31"), "us")
        assert result.iloc[0] == "COVID"

    def test_postcovid_first_day(self):
        """2022-01-01 → 'PostCOVID'."""
        result = assign_regime(self._dates("2022-01-01"), "us")
        assert result.iloc[0] == "PostCOVID"

    def test_all_us_labels_attainable(self):
        """A broad date range produces all 6 unique labels (5 regimes + OTHER)."""
        dates = pd.date_range("1990-01-01", "2030-01-01", freq="QS")
        result = assign_regime(dates, "us")
        unique = set(result)
        assert unique == {"OTHER", "PreGR", "GR", "PreCOVID", "COVID", "PostCOVID"}


# ===========================================================================
# 9. assign_regime — US political scope
# ===========================================================================

class TestAssignRegimeUSPolitical:
    """Known-value boundary tests for assign_regime with scope='us_political'."""

    def _dates(self, *date_strings):
        return pd.DatetimeIndex(list(date_strings))

    def test_before_bush43_is_other(self):
        """Date before 2001-01-01 → 'OTHER'."""
        result = assign_regime(self._dates("2000-12-31"), "us_political")
        assert result.iloc[0] == "OTHER"

    def test_bush43_first_day(self):
        """2001-01-01 → 'Bush43'."""
        result = assign_regime(self._dates("2001-01-01"), "us_political")
        assert result.iloc[0] == "Bush43"

    def test_bush43_last_day(self):
        """2008-12-31 → 'Bush43'."""
        result = assign_regime(self._dates("2008-12-31"), "us_political")
        assert result.iloc[0] == "Bush43"

    def test_obama_first_day(self):
        """2009-01-01 → 'Obama'."""
        result = assign_regime(self._dates("2009-01-01"), "us_political")
        assert result.iloc[0] == "Obama"

    def test_obama_last_day(self):
        """2016-12-31 → 'Obama'."""
        result = assign_regime(self._dates("2016-12-31"), "us_political")
        assert result.iloc[0] == "Obama"

    def test_trump1_first_day(self):
        """2017-01-01 → 'Trump1'."""
        result = assign_regime(self._dates("2017-01-01"), "us_political")
        assert result.iloc[0] == "Trump1"

    def test_trump1_last_day(self):
        """2020-12-31 → 'Trump1'."""
        result = assign_regime(self._dates("2020-12-31"), "us_political")
        assert result.iloc[0] == "Trump1"

    def test_biden_first_day(self):
        """2021-01-01 → 'Biden'."""
        result = assign_regime(self._dates("2021-01-01"), "us_political")
        assert result.iloc[0] == "Biden"

    def test_biden_last_day(self):
        """2024-12-31 → 'Biden'."""
        result = assign_regime(self._dates("2024-12-31"), "us_political")
        assert result.iloc[0] == "Biden"

    def test_trump2_first_day(self):
        """2025-01-01 → 'Trump2'."""
        result = assign_regime(self._dates("2025-01-01"), "us_political")
        assert result.iloc[0] == "Trump2"

    def test_trump2_far_future(self):
        """2040-01-01 → 'Trump2' (open-ended)."""
        result = assign_regime(self._dates("2040-01-01"), "us_political")
        assert result.iloc[0] == "Trump2"

    def test_all_political_labels_attainable(self):
        """A broad date range produces all 6 labels (5 presidencies + OTHER)."""
        dates = pd.date_range("1995-01-01", "2030-01-01", freq="QS")
        result = assign_regime(dates, "us_political")
        unique = set(result)
        assert unique == {"OTHER", "Bush43", "Obama", "Trump1", "Biden", "Trump2"}


# ===========================================================================
# 10. assign_regime — error handling
# ===========================================================================

class TestAssignRegimeErrors:
    """Error-handling tests for assign_regime."""

    def test_unknown_scope_raises_value_error(self):
        """Unrecognised scope raises ValueError."""
        dates = pd.DatetimeIndex(["2008-01-01"])
        with pytest.raises(ValueError):
            assign_regime(dates, scope="invalid")

    def test_error_message_mentions_scope(self):
        """Error message contains the bad scope name."""
        dates = pd.DatetimeIndex(["2008-01-01"])
        with pytest.raises(ValueError, match="invalid"):
            assign_regime(dates, scope="invalid")

    def test_error_message_lists_valid_scopes(self):
        """Error message lists 'global', 'us', 'us_political' as valid options."""
        dates = pd.DatetimeIndex(["2008-01-01"])
        with pytest.raises(ValueError, match="global"):
            assign_regime(dates, scope="wrong")

    def test_case_sensitive_scope(self):
        """Scope is case-sensitive: 'Global' (capital G) raises ValueError."""
        dates = pd.DatetimeIndex(["2008-01-01"])
        with pytest.raises(ValueError):
            assign_regime(dates, scope="Global")

    def test_empty_string_scope_raises(self):
        """Empty string scope raises ValueError."""
        dates = pd.DatetimeIndex(["2008-01-01"])
        with pytest.raises(ValueError):
            assign_regime(dates, scope="")


# ===========================================================================
# 11. Cross-scope consistency checks
# ===========================================================================

class TestCrossScopeConsistency:
    """Consistency checks across scopes for dates in shared regimes."""

    def _dates(self, *date_strings):
        return pd.DatetimeIndex(list(date_strings))

    def test_pregr_same_in_global_and_us(self):
        """PreGR dates get 'PreGR' in both global and US schemes."""
        dates = self._dates("2003-01-01", "2005-06-01", "2006-12-31")
        global_labels = assign_regime(dates, "global")
        us_labels = assign_regime(dates, "us")
        for g, u in zip(global_labels, us_labels):
            assert g == "PreGR"
            assert u == "PreGR"

    def test_gr_same_in_global_and_us(self):
        """GR dates get 'GR' in both global and US schemes."""
        dates = self._dates("2007-01-01", "2008-06-01", "2009-12-31")
        global_labels = assign_regime(dates, "global")
        us_labels = assign_regime(dates, "us")
        for g, u in zip(global_labels, us_labels):
            assert g == "GR"
            assert u == "GR"

    def test_covid_same_in_global_and_us(self):
        """COVID dates get 'COVID' in both global and US schemes."""
        dates = self._dates("2020-01-01", "2021-06-01", "2021-12-31")
        global_labels = assign_regime(dates, "global")
        us_labels = assign_regime(dates, "us")
        for g, u in zip(global_labels, us_labels):
            assert g == "COVID"
            assert u == "COVID"

    def test_postcovid_same_in_global_and_us(self):
        """PostCOVID dates get 'PostCOVID' in both global and US schemes."""
        dates = self._dates("2022-01-01", "2023-06-01", "2030-01-01")
        global_labels = assign_regime(dates, "global")
        us_labels = assign_regime(dates, "us")
        for g, u in zip(global_labels, us_labels):
            assert g == "PostCOVID"
            assert u == "PostCOVID"

    def test_inter_and_eurodebt_differ_from_us(self):
        """Dates 2010-2019 that are EuroDebt/Inter in global are PreCOVID in US."""
        dates = self._dates("2012-01-01", "2016-01-01")
        global_labels = assign_regime(dates, "global")
        us_labels = assign_regime(dates, "us")
        # global has EuroDebt/Inter, US has PreCOVID
        assert set(global_labels).issubset({"EuroDebt", "Inter"})
        assert all(l == "PreCOVID" for l in us_labels)

    def test_output_length_consistent_all_scopes(self):
        """All three scopes produce output of same length for same input."""
        dates = pd.date_range("1995-01-01", "2030-01-01", freq="QS")
        for scope in ("global", "us", "us_political"):
            result = assign_regime(dates, scope)
            assert len(result) == len(dates), f"Length mismatch for scope={scope}"
