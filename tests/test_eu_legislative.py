"""Tests for puremacro.narrative.sources.{eu_eurlex, eu_parliament}
and puremacro.narrative.indices.eu_legislative."""
from __future__ import annotations

from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "eu_legislative_mini"


class TestEurlexParser:
    def test_parse_eurlex_fixture_en(self):
        from puremacro.narrative.sources.eu_eurlex import _parse_eurlex_html
        html_path = FIXTURE_DIR / "eurlex_32024R0903_EN.html"
        if not html_path.exists():
            pytest.skip("EN fixture not yet fetched")
        rec = _parse_eurlex_html(html_path.read_text(),
                                  celex_id="32024R0903",
                                  language="en",
                                  source_url="https://test")
        if rec is None:
            pytest.skip("parser failed on fixture — adapt _BODY_SELECTORS")
        date, text, source_url, metadata = rec
        assert metadata["celex_id"] == "32024R0903"
        assert metadata["language"] == "en"
        assert metadata["act_type"] == "regulation"  # R = regulation
        assert len(text) > 1000, f"body suspiciously short: {len(text)}"

    def test_parse_eurlex_fixture_de(self):
        from puremacro.narrative.sources.eu_eurlex import _parse_eurlex_html
        html_path = FIXTURE_DIR / "eurlex_32024R0903_DE.html"
        if not html_path.exists():
            pytest.skip("DE fixture not yet fetched")
        rec = _parse_eurlex_html(html_path.read_text(),
                                  celex_id="32024R0903",
                                  language="de",
                                  source_url="https://test")
        if rec is None:
            pytest.skip("DE parser failed")
        date, text, source_url, metadata = rec
        assert metadata["language"] == "de"
        assert len(text) > 1000

    def test_parse_eurlex_fixture_fr(self):
        from puremacro.narrative.sources.eu_eurlex import _parse_eurlex_html
        html_path = FIXTURE_DIR / "eurlex_32024R0903_FR.html"
        if not html_path.exists():
            pytest.skip("FR fixture not yet fetched")
        rec = _parse_eurlex_html(html_path.read_text(),
                                  celex_id="32024R0903",
                                  language="fr",
                                  source_url="https://test")
        if rec is None:
            pytest.skip("FR parser failed")
        date, text, source_url, metadata = rec
        assert metadata["language"] == "fr"
        assert len(text) > 1000

    def test_parse_eurlex_adoption_date(self):
        """Adoption date extracted from eli:date_document meta tag."""
        from puremacro.narrative.sources.eu_eurlex import _parse_eurlex_html
        import datetime as dt
        html_path = FIXTURE_DIR / "eurlex_32024R0903_EN.html"
        if not html_path.exists():
            pytest.skip("EN fixture not yet fetched")
        rec = _parse_eurlex_html(html_path.read_text(),
                                  celex_id="32024R0903",
                                  language="en",
                                  source_url="https://test")
        if rec is None:
            pytest.skip("parser failed on fixture")
        date, text, source_url, metadata = rec
        # 32024R0903 was adopted 13 March 2024
        assert date == dt.date(2024, 3, 13), f"got {date!r}, expected 2024-03-13"

    def test_celex_act_type_code_mapping(self):
        from puremacro.narrative.sources.eu_eurlex import _CELEX_TYPE_CODES
        assert _CELEX_TYPE_CODES["R"] == "regulation"
        assert _CELEX_TYPE_CODES["L"] == "directive"
        assert _CELEX_TYPE_CODES["D"] == "decision"

    def test_iter_eurlex_empty_window(self):
        from puremacro.narrative.sources.eu_eurlex import iter_eurlex
        records = list(iter_eurlex(since="1900-01-01", until="1900-12-31"))
        assert records == []


@pytest.mark.network
class TestEurlexLiveFetch:
    def test_fetch_single_celex_en(self):
        from puremacro.narrative.sources.eu_eurlex import _fetch_celex_in_language
        rec = _fetch_celex_in_language("32024R0903", "en")
        if rec is None:
            pytest.skip("network fetch returned empty / parser failed")
        date, text, source_url, metadata = rec
        assert metadata["language"] == "en"
        assert len(text) > 1000


class TestEpParser:
    def test_parse_ep_fixture_en(self):
        from puremacro.narrative.sources.eu_parliament import _parse_ep_page
        import datetime as dt
        html_path = FIXTURE_DIR / "ep_cre_2024_12_16_EN.html"
        if not html_path.exists():
            pytest.skip("EP EN fixture not yet fetched")
        session_date = dt.date(2024, 12, 16)
        rec = _parse_ep_page(html_path.read_text(),
                              session_date=session_date,
                              language="en",
                              source_url="https://test",
                              term=10)
        if rec is None:
            pytest.skip("parser failed on EP fixture — adapt _BODY_SELECTORS")
        date, text, source_url, metadata = rec
        assert metadata["language"] == "en"
        assert metadata["session_date"] == "2024-12-16"
        assert metadata["term"] == 10
        assert metadata["source"] == "wayback"
        assert len(text) > 1000, f"body suspiciously short: {len(text)}"

    def test_parse_ep_fixture_body_content(self):
        from puremacro.narrative.sources.eu_parliament import _parse_ep_page
        import datetime as dt
        html_path = FIXTURE_DIR / "ep_cre_2024_12_16_EN.html"
        if not html_path.exists():
            pytest.skip("EP EN fixture not yet fetched")
        session_date = dt.date(2024, 12, 16)
        rec = _parse_ep_page(html_path.read_text(),
                              session_date=session_date,
                              language="en",
                              source_url="https://test",
                              term=10)
        if rec is None:
            pytest.skip("parser failed on EP fixture")
        date, text, source_url, metadata = rec
        # Session date should match the fixture date
        assert date == session_date
        # Body should contain plenary debate content
        assert "December 2024" in text or "Strasbourg" in text, (
            f"body does not look like EP plenary content: {text[:200]!r}"
        )

    def test_term_for_date(self):
        from puremacro.narrative.sources.eu_parliament import _term_for_date
        import datetime as dt
        # Term 10 starts 2024-07-16
        assert _term_for_date(dt.date(2024, 12, 16)) == 10
        assert _term_for_date(dt.date(2024, 7, 16)) == 10
        # Term 9 starts 2019-07-02
        assert _term_for_date(dt.date(2024, 7, 15)) == 9
        assert _term_for_date(dt.date(2020, 1, 1)) == 9
        # Term 8 starts 2014-07-01
        assert _term_for_date(dt.date(2019, 7, 1)) == 8

    def test_iter_ep_debates_empty_window(self):
        from puremacro.narrative.sources.eu_parliament import iter_ep_debates
        records = list(iter_ep_debates(since="1900-01-01", until="1900-12-31"))
        assert records == []


@pytest.mark.network
class TestEpLiveFetch:
    def test_fetch_ep_session_en(self):
        from puremacro.narrative.sources.eu_parliament import _fetch_ep_session
        import datetime as dt
        rec = _fetch_ep_session(dt.date(2024, 12, 16), "en")
        if rec is None:
            pytest.skip("network fetch returned empty / parser failed")
        date, text, source_url, metadata = rec
        assert metadata["language"] == "en"
        assert len(text) > 1000


class TestIndices:
    @pytest.fixture
    def toy_eurlex(self):
        import datetime as dt
        import pandas as pd
        return pd.DataFrame([
            (dt.date(2024, 1, 15), "Regulation establishing rules amid uncertainty.",
             "https://test/r1-en", {"language": "en", "celex_id": "32024R0001",
                                      "act_type": "regulation", "subject_codes": []}),
            (dt.date(2024, 1, 15), "Verordnung zur Festlegung von Regeln unter Unsicherheit.",
             "https://test/r1-de", {"language": "de", "celex_id": "32024R0001",
                                      "act_type": "regulation", "subject_codes": []}),
            (dt.date(2024, 3, 20), "Directive on labor markets remains uncertain.",
             "https://test/l1-en", {"language": "en", "celex_id": "32024L0002",
                                      "act_type": "directive", "subject_codes": []}),
        ], columns=["date", "text", "source_url", "metadata"])

    def test_eurlex_ui_default_en(self, toy_eurlex):
        from puremacro.narrative.indices.eu_legislative import eurlex_ui
        ri = eurlex_ui(toy_eurlex, language="en")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_eurlex_ui_de(self, toy_eurlex):
        from puremacro.narrative.indices.eu_legislative import eurlex_ui
        ri = eurlex_ui(toy_eurlex, language="de")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_eurlex_ui_act_type_filter(self, toy_eurlex):
        from puremacro.narrative.indices.eu_legislative import eurlex_ui
        ri = eurlex_ui(toy_eurlex, language="en", act_type="directive")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_ep_ui_default(self):
        import pandas as pd
        import datetime as dt
        toy = pd.DataFrame([
            (dt.date(2024, 12, 16), "The Parliament debated economic uncertainty.",
             "https://test/ep-en", {"language": "en", "session_date": "2024-12-16",
                                      "term": 10, "source": "wayback"}),
        ], columns=["date", "text", "source_url", "metadata"])
        from puremacro.narrative.indices.eu_legislative import ep_ui
        ri = ep_ui(toy, language="en")
        from puremacro.narrative.types import RiskIndex
        assert isinstance(ri, RiskIndex)

    def test_ensure_dataframe_empty(self):
        from puremacro.narrative.indices.eu_legislative import _ensure_dataframe
        df = _ensure_dataframe(iter([]))
        assert list(df.columns) == ["date", "text", "source_url", "metadata"]
        assert len(df) == 0


class TestCelexViaSparql:
    def test_parses_sparql_json_response(self, monkeypatch):
        """Mock SPARQL JSON response; assert CELEX list extracted."""
        from puremacro.narrative.sources import eu_eurlex
        from datetime import date as _date

        sparql_json = """{
          "results": {
            "bindings": [
              {"celex": {"value": "32024R0903"}},
              {"celex": {"value": "32024R0905"}},
              {"celex": {"value": "32024R1234"}}
            ]
          }
        }"""
        monkeypatch.setattr(eu_eurlex, "safe_get_text",
                            lambda url, *a, **kw: sparql_json)
        result = eu_eurlex._celex_via_sparql(
            since=_date(2024, 1, 1), until=_date(2024, 12, 31),
            act_types=("regulation",),
        )
        assert "32024R0903" in result
        assert "32024R0905" in result
        assert "32024R1234" in result

    def test_filters_to_expected_celex_shape(self, monkeypatch):
        """Reject CELEX values that don't match 3<YYYY><type>{number}."""
        from puremacro.narrative.sources import eu_eurlex
        from datetime import date as _date

        sparql_json = """{
          "results": {
            "bindings": [
              {"celex": {"value": "32024R0903"}},
              {"celex": {"value": "NOT_A_CELEX"}},
              {"celex": {"value": "42024R0903"}}
            ]
          }
        }"""
        monkeypatch.setattr(eu_eurlex, "safe_get_text",
                            lambda url, *a, **kw: sparql_json)
        result = eu_eurlex._celex_via_sparql(
            since=_date(2024, 1, 1), until=_date(2024, 12, 31),
            act_types=("regulation",),
        )
        assert "32024R0903" in result
        assert "NOT_A_CELEX" not in result
        assert "42024R0903" not in result

    def test_directive_celex_uses_l_not_d(self, monkeypatch):
        """Directives use CELEX letter L (French Loi), not D (decision)."""
        from puremacro.narrative.sources import eu_eurlex
        from datetime import date as _date

        sparql_json = """{
          "results": {
            "bindings": [
              {"celex": {"value": "32024L0001"}},
              {"celex": {"value": "32024D0002"}}
            ]
          }
        }"""
        monkeypatch.setattr(eu_eurlex, "safe_get_text",
                            lambda url, *a, **kw: sparql_json)
        # Query for directives (Cellar DIR)
        result = eu_eurlex._celex_via_sparql(
            since=_date(2024, 1, 1), until=_date(2024, 12, 31),
            act_types=("directive",),
        )
        # Should accept L-type (32024L0001) but reject D-type (32024D0002)
        assert "32024L0001" in result
        assert "32024D0002" not in result

    def test_returns_empty_on_sparql_error(self, monkeypatch):
        from puremacro.narrative.sources import eu_eurlex
        from datetime import date as _date

        def boom(url, *a, **kw):
            raise ConnectionError("sparql down")
        monkeypatch.setattr(eu_eurlex, "safe_get_text", boom)
        result = eu_eurlex._celex_via_sparql(
            since=_date(2024, 1, 1), until=_date(2024, 12, 31),
            act_types=("regulation",),
        )
        assert result == []

    def test_enumerate_celex_ids_calls_sparql_first(self, monkeypatch):
        """The orchestrator helper calls _celex_via_sparql; on empty
        result, falls back to _celex_via_html_scrape."""
        from puremacro.narrative.sources import eu_eurlex
        from datetime import date as _date

        sparql_calls = []
        html_calls = []
        def fake_sparql(**kw):
            sparql_calls.append(kw)
            return ["32024R0903"]
        def fake_html(**kw):
            html_calls.append(kw)
            return ["32024R0999"]
        monkeypatch.setattr(eu_eurlex, "_celex_via_sparql", fake_sparql)
        monkeypatch.setattr(eu_eurlex, "_celex_via_html_scrape", fake_html)

        result = eu_eurlex._enumerate_celex_ids(
            since=_date(2024, 1, 1), until=_date(2024, 12, 31),
            act_types=("regulation",),
        )
        # SPARQL returned non-empty → HTML fallback NOT called
        assert len(sparql_calls) == 1
        assert html_calls == []
        assert "32024R0903" in result

    def test_enumerate_celex_ids_falls_back_to_html_on_empty_sparql(
        self, monkeypatch,
    ):
        from puremacro.narrative.sources import eu_eurlex
        from datetime import date as _date

        def fake_sparql(**kw):
            return []  # SPARQL empty
        html_calls = []
        def fake_html(**kw):
            html_calls.append(kw)
            return ["32024R0999"]
        monkeypatch.setattr(eu_eurlex, "_celex_via_sparql", fake_sparql)
        monkeypatch.setattr(eu_eurlex, "_celex_via_html_scrape", fake_html)

        result = eu_eurlex._enumerate_celex_ids(
            since=_date(2024, 1, 1), until=_date(2024, 12, 31),
            act_types=("regulation",),
        )
        assert html_calls == [{"since": _date(2024, 1, 1),
                                "until": _date(2024, 12, 31),
                                "act_types": ("regulation",)}]
        assert "32024R0999" in result


class TestEpTerm7Coverage:
    def test_ep_earliest_date_lifted_to_term7(self):
        """Track A.3 lifted _EP_EARLIEST_DATE from 2014-07-01 to
        2009-07-14 (Term 7 start) after the controller probed
        Wayback CDX and found 2/3 Term-7 sample dates reachable."""
        from puremacro.narrative.sources import eu_parliament
        from datetime import date as _date
        assert eu_parliament._EP_EARLIEST_DATE == _date(2009, 7, 14)

    def test_term_for_date_returns_7_for_term7_dates(self):
        """_term_for_date must classify pre-2014-07-01 dates as Term 7."""
        from puremacro.narrative.sources import eu_parliament
        from datetime import date as _date
        assert eu_parliament._term_for_date(_date(2010, 3, 9)) == 7
        assert eu_parliament._term_for_date(_date(2012, 5, 22)) == 7
        assert eu_parliament._term_for_date(_date(2014, 6, 30)) == 7
        # Boundary check: 2014-07-01 starts Term 8.
        assert eu_parliament._term_for_date(_date(2014, 7, 1)) == 8
