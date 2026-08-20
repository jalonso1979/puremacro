"""District-level BBUI: Fed district crosswalk + tidy district output.

Offline throughout: Beige Book HTTP is replayed from
``tests/fixtures/beige_book_mini`` (two releases x three districts +
national summary, ``modern_2023*_mini*.html``) through the shared
``mock_http`` conftest fixture that patches the connector's
``safe_get_text`` / ``safe_get_bytes``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from puremacro.narrative.indices._fed_districts import (
    DISTRICT_NAMES,
    DISTRICT_NUMBER,
    FED_DISTRICTS,
    SPLIT_STATES,
    STATE_TO_DISTRICT,
    STATE_TO_DISTRICTS_FULL,
    TERRITORY_TO_DISTRICT,
    district_crosswalk,
    district_states,
    state_district,
)
from puremacro.narrative.sources.beige_book import DISTRICTS as BB_DISTRICTS


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "beige_book_mini"

_BASE = "https://www.federalreserve.gov/monetarypolicy/beigebook"
_MINI_RELEASES = ("202303", "202310")
_MINI_SLUGS = ("boston", "dallas", "san-francisco")
# Modern release months inside the test window that have no fixture —
# registered as empty pages so the connector skips them cleanly (the
# connector probes every month because release months drift).
_EMPTY_MONTHS = ("202304", "202305", "202306", "202307", "202308",
                 "202309")


# ---------------------------------------------------------------------------
# Crosswalk invariants
# ---------------------------------------------------------------------------

class TestFedDistrictCrosswalk:
    def test_twelve_districts_numbered_1_to_12(self):
        assert len(FED_DISTRICTS) == 12
        assert [d.number for d in FED_DISTRICTS] == list(range(1, 13))
        assert len(DISTRICT_NAMES) == 12
        assert DISTRICT_NUMBER["Boston"] == 1
        assert DISTRICT_NUMBER["San Francisco"] == 12

    def test_names_match_beige_book_connector(self):
        # The crosswalk must join directly onto Beige Book records.
        assert set(DISTRICT_NAMES) == set(BB_DISTRICTS)

    def test_all_50_states_plus_dc_mapped(self):
        assert len(STATE_TO_DISTRICT) == 51
        assert "DC" in STATE_TO_DISTRICT
        assert all(len(s) == 2 and s.isupper() for s in STATE_TO_DISTRICT)
        assert set(STATE_TO_DISTRICT.values()) <= set(DISTRICT_NAMES)

    def test_every_district_has_at_least_one_primary_state(self):
        assert set(STATE_TO_DISTRICT.values()) == set(DISTRICT_NAMES)

    def test_full_mapping_consistent_with_primary(self):
        assert set(STATE_TO_DISTRICTS_FULL) == set(STATE_TO_DISTRICT)
        for state, districts in STATE_TO_DISTRICTS_FULL.items():
            assert STATE_TO_DISTRICT[state] in districts, state
            assert districts <= set(DISTRICT_NAMES), state

    def test_split_states(self):
        # 14 split states under the current official map.
        assert SPLIT_STATES == frozenset({
            "CT", "NJ", "PA", "KY", "WV", "LA", "MS", "TN",
            "IL", "IN", "MI", "WI", "MO", "NM",
        })
        for state in SPLIT_STATES:
            assert len(STATE_TO_DISTRICTS_FULL[state]) == 2, state
        for state in set(STATE_TO_DISTRICT) - SPLIT_STATES:
            assert len(STATE_TO_DISTRICTS_FULL[state]) == 1, state

    def test_missouri_convention(self):
        # The famous split: both the 8th and 10th District banks sit in MO.
        assert STATE_TO_DISTRICTS_FULL["MO"] == frozenset(
            {"St. Louis", "Kansas City"})
        assert STATE_TO_DISTRICT["MO"] == "St. Louis"

    def test_district_states_partition(self):
        seen: list[str] = []
        for name in DISTRICT_NAMES:
            seen.extend(district_states(name))
        assert len(seen) == 51
        assert len(set(seen)) == 51  # primary assignment partitions states

    def test_district_states_include_partial(self):
        with_partial = district_states("Kansas City", include_partial=True)
        assert "MO" in with_partial
        assert "MO" not in district_states("Kansas City")

    def test_state_district_lookup(self):
        assert state_district("TX") == "Dallas"
        assert state_district("mo") == "St. Louis"
        assert state_district("MO", all_districts=True) == frozenset(
            {"St. Louis", "Kansas City"})
        assert state_district("PR") == "New York"

    def test_unknown_inputs_raise(self):
        with pytest.raises(KeyError, match="unknown Federal Reserve district"):
            district_states("Springfield")
        with pytest.raises(KeyError, match="unknown state"):
            state_district("XX")

    def test_territories(self):
        assert set(TERRITORY_TO_DISTRICT) == {"PR", "VI", "GU", "AS", "MP"}
        assert set(TERRITORY_TO_DISTRICT) & set(STATE_TO_DISTRICT) == set()

    def test_crosswalk_dataframe(self):
        xw = district_crosswalk()
        assert list(xw.columns) == ["state", "district",
                                    "district_number", "primary"]
        # 51 primary rows + one extra row per split state.
        assert int(xw["primary"].sum()) == 51
        assert len(xw) == 51 + len(SPLIT_STATES)
        # Row set consistent with the many-to-many mapping.
        for state, grp in xw.groupby("state"):
            assert set(grp["district"]) == set(STATE_TO_DISTRICTS_FULL[state])
            prim = grp.loc[grp["primary"], "district"]
            assert list(prim) == [STATE_TO_DISTRICT[state]]


# ---------------------------------------------------------------------------
# Offline Beige Book records via the shared mock_http fixture
# ---------------------------------------------------------------------------

def _register_mini_fixtures(mock_http) -> None:
    text: dict[str, str] = {}
    for yyyymm in _MINI_RELEASES:
        text[f"{_BASE}{yyyymm}.htm"] = (
            FIXTURE_DIR / f"modern_{yyyymm}_mini.html").read_text(encoding="utf-8")
        text[f"{_BASE}{yyyymm}-summary.htm"] = (
            FIXTURE_DIR / f"modern_{yyyymm}_mini_summary.html").read_text(encoding="utf-8")
        for slug in _MINI_SLUGS:
            text[f"{_BASE}{yyyymm}-{slug}.htm"] = (
                FIXTURE_DIR / f"modern_{yyyymm}_mini_{slug}.html").read_text(encoding="utf-8")
    for yyyymm in _EMPTY_MONTHS:
        text[f"{_BASE}{yyyymm}.htm"] = ""
    mock_http(text=text)


@pytest.fixture
def bb_records(mock_http):
    from puremacro.narrative.sources import iter_beige_book
    _register_mini_fixtures(mock_http)
    return list(iter_beige_book(since="2023-03-01", until="2023-10-31"))


class TestDistrictParsingOffline:
    def test_expected_districts_parsed(self, bb_records):
        districts = {r[1] for r in bb_records}
        assert districts == {"National", "Boston", "Dallas", "San Francisco"}

    def test_two_releases_parsed(self, bb_records):
        import datetime as dt
        dates = {r[0] for r in bb_records}
        assert dates == {dt.date(2023, 3, 1), dt.date(2023, 10, 1)}

    def test_records_are_6_tuples_with_canonical_sections(self, bb_records):
        from puremacro.narrative.sources.beige_book import CANONICAL_SECTIONS
        assert len(bb_records) >= 24  # 2 releases x 4 units x 3 sections
        for r in bb_records:
            assert len(r) == 6
            assert r[2] in CANONICAL_SECTIONS
            assert r[3].strip()

    def test_each_district_has_three_sections_per_release(self, bb_records):
        df = pd.DataFrame(bb_records, columns=[
            "date", "district", "section", "text", "source_url", "metadata"])
        counts = df.groupby(["date", "district"])["section"].nunique()
        assert (counts == 3).all(), counts.to_dict()


@pytest.fixture
def bb_inline_records(mock_http):
    """Records from the ~2017-2023 single-page inline layout fixture."""
    from puremacro.narrative.sources import iter_beige_book
    text = {
        f"{_BASE}202206.htm":
            (FIXTURE_DIR / "modern_202206_mini_inline.html").read_text(encoding="utf-8"),
        f"{_BASE}202206-summary.htm":
            (FIXTURE_DIR / "modern_202206_mini_inline_summary.html")
            .read_text(encoding="utf-8"),
    }
    mock_http(text=text)
    return list(iter_beige_book(since="2022-06-01", until="2022-06-30"))


class TestInlineSinglePageLayout:
    """The ~2017-2023 layout: districts inline on the index page as
    h4 bank headers + <p><strong>Section</strong> body</p>; only the
    National Summary has a sub-page (same p>strong style)."""

    def test_districts_parsed_from_index_page(self, bb_inline_records):
        districts = {r[1] for r in bb_inline_records}
        assert districts == {"National", "Boston", "Dallas", "San Francisco"}

    def test_sections_canonicalised(self, bb_inline_records):
        df = pd.DataFrame(bb_inline_records, columns=[
            "date", "district", "section", "text", "source_url", "metadata"])
        boston = df[df["district"] == "Boston"]
        assert set(boston["section"]) == {"overall", "labor", "prices"}
        # Section header must not leak into the body text.
        overall = boston[boston["section"] == "overall"]["text"].iloc[0]
        assert not overall.startswith("Summary of Economic Activity")

    def test_summary_page_p_strong_sections(self, bb_inline_records):
        national = [r for r in bb_inline_records if r[1] == "National"]
        assert {r[2] for r in national} == {"overall", "labor", "prices"}
        # Highlights blurbs must not be swept into National records.
        assert all("Short district blurb" not in r[3] for r in national)

    def test_boilerplate_paragraph_dropped(self, bb_inline_records):
        assert all("For more information" not in r[3]
                   for r in bb_inline_records)

    def test_per_district_granularity_bundles(self, mock_http):
        from puremacro.narrative.sources import iter_beige_book
        text = {
            f"{_BASE}202206.htm":
                (FIXTURE_DIR / "modern_202206_mini_inline.html").read_text(encoding="utf-8"),
            f"{_BASE}202206-summary.htm":
                (FIXTURE_DIR / "modern_202206_mini_inline_summary.html")
                .read_text(encoding="utf-8"),
        }
        mock_http(text=text)
        records = list(iter_beige_book(
            since="2022-06-01", until="2022-06-30",
            granularity="per_district"))
        bundles = [r for r in records if r[2] == "all"]
        assert {r[1] for r in bundles} == {"Boston", "Dallas",
                                           "San Francisco"}

    def test_bbui_tidy_works_on_inline_records(self, bb_inline_records):
        from puremacro.narrative.indices import bbui
        tidy = bbui(bb_inline_records, level="district", output="tidy",
                    normalize="raw")
        covered = tidy[tidy["n_docs"] > 0]
        assert set(covered["district"]) == {"National", "Boston", "Dallas",
                                            "San Francisco"}
        assert (covered["n_docs"] == 3).all()


class TestBBUIDistrictTidy:
    def test_tidy_shape_and_columns(self, bb_records):
        from puremacro.narrative.indices import bbui
        tidy = bbui(bb_records, level="district", output="tidy",
                    normalize="raw")
        assert list(tidy.columns) == ["date", "district", "value",
                                      "n_docs", "n_sections"]
        assert set(tidy["district"]) == {"National", "Boston", "Dallas",
                                         "San Francisco"}
        # 2023Q1..2023Q4 reindexed range -> 4 quarters x 4 units.
        assert len(tidy) == 16
        assert tidy["n_docs"].dtype.kind == "i"

    def test_tidy_coverage_counts(self, bb_records):
        from puremacro.narrative.indices import bbui
        tidy = bbui(bb_records, level="district", output="tidy",
                    normalize="raw")
        covered = tidy[tidy["n_docs"] > 0]
        # Each unit has exactly 2 covered quarters (Q1 and Q4 2023) with
        # 3 section-documents each.
        assert (covered.groupby("district").size() == 2).all()
        assert (covered["n_docs"] == 3).all()
        assert (covered["n_sections"] == 3).all()
        assert covered["value"].notna().all()
        gaps = tidy[tidy["n_docs"] == 0]
        assert gaps["value"].isna().all()
        assert (gaps["n_sections"] == 0).all()

    def test_tidy_matches_wide_values(self, bb_records):
        from puremacro.narrative.indices import bbui
        tidy = bbui(bb_records, level="district", output="tidy",
                    normalize="raw")
        wide = bbui(bb_records, level="district", normalize="raw")
        for _, row in tidy[tidy["n_docs"] > 0].iterrows():
            assert row["value"] == pytest.approx(
                wide.loc[row["date"], row["district"]])

    def test_banking_stress_release_scores_higher_in_dallas(self, bb_records):
        # March 2023 fixture text (banking stress, layoffs, uncertainty)
        # must out-score the calm October 2023 text.
        from puremacro.narrative.indices import bbui
        tidy = bbui(bb_records, level="district", output="tidy",
                    normalize="raw")
        dallas = tidy[(tidy["district"] == "Dallas")
                      & (tidy["n_docs"] > 0)].set_index("date")["value"]
        assert dallas.loc["2023-01-01"] > dallas.loc["2023-10-01"]

    def test_wide_output_unchanged(self, bb_records):
        from puremacro.narrative.indices import bbui
        wide = bbui(bb_records, level="district", normalize="raw")
        assert isinstance(wide, pd.DataFrame)
        assert set(wide.columns) == {"National", "Boston", "Dallas",
                                     "San Francisco"}

    def test_tidy_with_national_level_raises(self, bb_records):
        from puremacro.narrative.indices import bbui
        with pytest.raises(ValueError, match="only meaningful"):
            bbui(bb_records, level="national", output="tidy")

    def test_invalid_output_raises(self, bb_records):
        from puremacro.narrative.indices import bbui
        with pytest.raises(ValueError, match="output must be"):
            bbui(bb_records, level="district", output="long")

    def test_national_level_still_returns_riskindex(self, bb_records):
        from puremacro.narrative.indices import bbui
        from puremacro.narrative.types import RiskIndex
        ri = bbui(bb_records, level="national", normalize="raw")
        assert isinstance(ri, RiskIndex)
        assert len(ri.series) >= 2
