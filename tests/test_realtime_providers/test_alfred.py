"""ALFRED vintage accessor: pure parser + the cached/stored accessor.

The HTTP boundary is patched (``safe_get_bytes_cached`` / ``safe_get_bytes``
inside the ``alfred`` module), never ``alfred_vintages`` itself — patching
the function under test would make the assertions statements about the
patch. See CONTRIBUTING.md, "Do not patch the thing you are asserting
about".

Both retrieval routes are covered, and each test pins which one runs by
patching ``_resolve_key``: otherwise the branch taken would depend on
whether the machine running the suite happens to have a FRED key.
"""
from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

from puremacro.fetch.realtime import alfred as A
from puremacro.fetch.realtime import parse_alfredgraph_csv


TWO_VINTAGE_CSV = (
    b"observation_date,CLVMNACSCAB1GQDE_20160223,CLVMNACSCAB1GQDE_20160524\n"
    b"1991-01-01,500.0,502.0\n"
    b"1991-04-01,505.0,507.0\n"
    b"1991-07-01,,509.0\n"
)


# ---------------------------------------------------------------------------
# parse_alfredgraph_csv — pure, no network
# ---------------------------------------------------------------------------
def test_parses_compact_vintage_suffix():
    out = parse_alfredgraph_csv(TWO_VINTAGE_CSV)
    assert list(out.columns) == ["date", "vintage", "value"]
    assert set(out["vintage"].unique()) == {
        pd.Timestamp("2016-02-23"), pd.Timestamp("2016-05-24")}
    row = out[(out["date"] == pd.Timestamp("1991-01-01"))
              & (out["vintage"] == pd.Timestamp("2016-05-24"))]
    assert row["value"].iloc[0] == 502.0


def test_parses_hyphenated_vintage_suffix():
    csv = (b"observation_date,GDPC1_2013-04-26,GDPC1_2013-05-30\n"
           b"2012-10-01,100.0,101.0\n")
    out = parse_alfredgraph_csv(csv)
    assert set(out["vintage"].unique()) == {
        pd.Timestamp("2013-04-26"), pd.Timestamp("2013-05-30")}


def test_missing_cells_are_dropped_not_zero_filled():
    out = parse_alfredgraph_csv(TWO_VINTAGE_CSV)
    q3 = out[out["date"] == pd.Timestamp("1991-07-01")]
    # 1991Q3 was absent from the first vintage: one row, not two, and no 0.0.
    assert len(q3) == 1
    assert q3["vintage"].iloc[0] == pd.Timestamp("2016-05-24")


def test_dots_parse_as_missing():
    csv = (b"observation_date,GDPC1_20130426\n"
           b"2012-10-01,.\n2013-01-01,105.0\n")
    out = parse_alfredgraph_csv(csv)
    assert len(out) == 1
    assert out["value"].iloc[0] == 105.0


def test_plain_fred_csv_yields_no_vintages():
    """A non-archival CSV has no vintage suffix; claiming one would be a lie."""
    out = parse_alfredgraph_csv(b"observation_date,GDPC1\n2012-10-01,100.0\n")
    assert out.empty
    assert list(out.columns) == ["date", "vintage", "value"]


def test_columns_without_a_date_suffix_are_skipped():
    csv = (b"observation_date,GDPC1_notadate,GDPC1_20130426\n"
           b"2012-10-01,1.0,100.0\n")
    out = parse_alfredgraph_csv(csv)
    assert len(out) == 1
    assert out["vintage"].iloc[0] == pd.Timestamp("2013-04-26")


@pytest.mark.parametrize("raw", [b"", b"   ", b"\n"])
def test_empty_payloads_return_empty_frame(raw):
    out = parse_alfredgraph_csv(raw)
    assert out.empty
    assert list(out.columns) == ["date", "vintage", "value"]


def test_accepts_str_as_well_as_bytes():
    out = parse_alfredgraph_csv(TWO_VINTAGE_CSV.decode())
    assert len(out) == 5


def test_parses_api_observations_payload():
    payload = {"observations": [
        {"realtime_start": "2016-04-04", "realtime_end": "2016-08-11",
         "date": "1991-01-01",
         "CLVMNACSCAB1GQDE_20160404": "512528.9",
         "CLVMNACSCAB1GQDE_20160812": "512600.0"},
        {"date": "1991-04-01",
         "CLVMNACSCAB1GQDE_20160404": ".",
         "CLVMNACSCAB1GQDE_20160812": "515263.5"},
    ]}
    out = A.parse_alfred_api_observations(payload)
    assert list(out.columns) == ["date", "vintage", "value"]
    # The "." is missing, not zero, so 1991Q2 has one edition not two.
    assert len(out) == 3
    assert out["value"].max() == 515263.5
    # realtime_start/realtime_end must not be mistaken for vintage columns.
    assert out["vintage"].nunique() == 2


def test_api_parser_accepts_raw_json_text():
    out = A.parse_alfred_api_observations(
        '{"observations":[{"date":"2020-01-01","X_20200401":"1.5"}]}')
    assert len(out) == 1
    assert out["vintage"].iloc[0] == pd.Timestamp("2020-04-01")


def test_api_parser_on_empty_payload():
    assert A.parse_alfred_api_observations({"observations": []}).empty
    assert A.parse_alfred_api_observations({}).empty


def test_vintage_dates_html_parser():
    html = ('<select name="form[selected_vintage_dates][]">'
            '<option value="2016-04-04">2016-04-04</option>'
            '<option value="2016-08-12">2016-08-12</option>'
            '<option value="2016-04-04">dup</option></select>')
    out = A.parse_alfred_vintage_dates_html(html)
    assert out == [pd.Timestamp("2016-04-04"), pd.Timestamp("2016-08-12")]
    assert A.parse_alfred_vintage_dates_html("") == []


# ---------------------------------------------------------------------------
# alfred_vintages — both routes, each pinned explicitly
#
# The FRED key is resolved from the developer's own credentials, so a
# test that did not pin the route would exercise a different branch on
# a machine with a key than on one without. Every test below forces
# `_resolve_key`.
# ---------------------------------------------------------------------------
API_OBSERVATIONS = json.dumps({"observations": [
    {"date": "1991-01-01", "X_20160223": "500.0", "X_20160524": "502.0"},
    {"date": "1991-04-01", "X_20160223": "505.0", "X_20160524": "507.0"},
    {"date": "1991-07-01", "X_20160223": ".", "X_20160524": "509.0"},
]}).encode()

VINTAGE_DATES_JSON = json.dumps(
    {"vintage_dates": ["2016-02-23", "2016-05-24"]}).encode()

DOWNLOAD_HTML = (
    '<select name="form[selected_vintage_dates][]">'
    '<option value="2016-02-23">a</option>'
    '<option value="2016-05-24">b</option></select>'
).encode()

GRAPH_CSV = {
    "2016-02-23": (b"observation_date,X_20160223\n"
                   b"1991-01-01,500.0\n1991-04-01,505.0\n"),
    "2016-05-24": (b"observation_date,X_20160524\n"
                   b"1991-01-01,502.0\n1991-04-01,507.0\n1991-07-01,509.0\n"),
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    import puremacro._cache_db as M
    M.close_conn()
    from puremacro.vintages import AlfredVintageStore
    s = AlfredVintageStore()
    yield s
    M.close_conn()


@pytest.fixture
def router(monkeypatch):
    """Serve every ALFRED endpoint from memory; record the URLs asked for."""
    calls = []

    def _fake(url, timeout=None, **kw):
        calls.append(url)
        if "series/vintagedates" in url:
            return VINTAGE_DATES_JSON
        if "series/observations" in url:
            return API_OBSERVATIONS
        if "downloaddata" in url:
            return DOWNLOAD_HTML
        if "alfredgraph.csv" in url:
            if "vintage_date=" in url:
                return GRAPH_CSV[url.split("vintage_date=")[-1][:10]]
            return GRAPH_CSV["2016-05-24"]
        raise LookupError(f"router: unexpected URL {url}")

    monkeypatch.setattr(A, "safe_get_bytes_cached", _fake)
    monkeypatch.setattr(A, "safe_get_bytes", _fake)
    return calls


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setattr(A, "_resolve_key", lambda k: k or "TESTKEY")


@pytest.fixture
def without_key(monkeypatch):
    monkeypatch.setattr(A, "_resolve_key", lambda k: k)


def test_api_route_returns_the_whole_triangle_in_one_request(
        router, store, with_key):
    out = A.alfred_vintages("X", store=store)
    assert list(out.columns) == ["date", "vintage", "value"]
    assert out["vintage"].nunique() == 2
    assert len(out) == 5
    obs_calls = [u for u in router if "series/observations" in u]
    assert len(obs_calls) == 1, "the API route must not paginate per vintage"
    assert "output_type=2" in obs_calls[0]


def test_keyless_route_fetches_one_csv_per_edition(router, store, without_key):
    out = A.alfred_vintages("X", store=store)
    assert out["vintage"].nunique() == 2
    assert len(out) == 5
    graph_calls = [u for u in router if "alfredgraph.csv" in u]
    assert len(graph_calls) == 2
    assert all("vintage_date=" in u for u in graph_calls), (
        "a bare graph CSV request returns only the current edition")


def test_both_routes_agree(router, store, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "_resolve_key", lambda k: "K")
    via_api = A.alfred_vintages("X", store=None, use_cache=False)
    monkeypatch.setattr(A, "_resolve_key", lambda k: None)
    via_csv = A.alfred_vintages("X", store=None, use_cache=False)
    pd.testing.assert_frame_equal(via_api, via_csv, check_dtype=False)


def test_keyless_route_warns_when_it_would_issue_many_requests(
        router, store, without_key, monkeypatch):
    monkeypatch.setattr(A, "KEYLESS_REQUEST_WARN_THRESHOLD", 1)
    with pytest.warns(UserWarning, match="one request per edition"):
        A.alfred_vintages("X", store=store)


def test_no_warning_below_the_threshold(router, store, without_key):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        A.alfred_vintages("X", store=store)


def test_wide_output_is_the_revision_triangle(router, store, with_key):
    wide = A.alfred_vintages("X", output="wide", store=store)
    assert wide.shape == (3, 2)
    assert list(wide.columns) == [pd.Timestamp("2016-02-23"),
                                  pd.Timestamp("2016-05-24")]
    assert wide.columns.name == "vintage"
    assert pd.isna(wide.loc[pd.Timestamp("1991-07-01"),
                            pd.Timestamp("2016-02-23")])


def test_vintages_first_and_latest_select_editions(router, store, with_key):
    first = A.alfred_vintages("X", vintages="first", store=store)
    assert first[first["date"] == pd.Timestamp("1991-01-01")]["value"].iloc[0] == 500.0
    latest = A.alfred_vintages("X", vintages="latest", store=store)
    assert latest[latest["date"] == pd.Timestamp("1991-01-01")]["value"].iloc[0] == 502.0
    assert len(first) == len(latest) == 3


def test_explicit_vintage_list_filters(router, store, with_key):
    out = A.alfred_vintages("X", vintages=["2016-05-24"], store=store)
    assert set(out["vintage"].unique()) == {pd.Timestamp("2016-05-24")}


def test_start_and_end_vintage_bounds(router, store, with_key):
    out = A.alfred_vintages("X", start_vintage="2016-03-01", store=store)
    assert out["vintage"].min() == pd.Timestamp("2016-05-24")
    out2 = A.alfred_vintages("X", end_vintage="2016-03-01", store=store)
    assert out2["vintage"].max() == pd.Timestamp("2016-02-23")


def test_bad_output_argument_raises(router, store, with_key):
    with pytest.raises(ValueError, match="output must be"):
        A.alfred_vintages("X", output="triangle", store=store)


def test_bad_vintages_argument_raises(router, store, with_key):
    with pytest.raises(ValueError, match="vintages must be"):
        A.alfred_vintages("X", vintages="most_recent", store=store)


def test_second_call_is_served_from_the_store(router, store, with_key):
    A.alfred_vintages("X", store=store)
    n_after_first = len(router)
    assert store.has_series("X")
    A.alfred_vintages("X", store=store)
    assert len(router) == n_after_first, "second call should not hit the network"


def test_refresh_forces_a_refetch(router, store, with_key):
    A.alfred_vintages("X", store=store)
    n = len(router)
    A.alfred_vintages("X", store=store, refresh=True)
    assert len(router) > n


def test_vintage_dates_helper_uses_api_when_keyed(router, with_key):
    out = A.alfred_vintage_dates("X")
    assert out == [pd.Timestamp("2016-02-23"), pd.Timestamp("2016-05-24")]
    assert any("series/vintagedates" in u for u in router)


def test_vintage_dates_helper_scrapes_when_keyless(router, without_key):
    out = A.alfred_vintage_dates("X")
    assert out == [pd.Timestamp("2016-02-23"), pd.Timestamp("2016-05-24")]
    assert any("downloaddata" in u for u in router)
    assert not any("series/vintagedates" in u for u in router)


def test_a_bounded_fetch_does_not_poison_the_store(router, store, with_key,
                                                   monkeypatch):
    """A vintage-bounded fetch must not be cached as the whole archive.

    Both routes push the bounds down to the server, and the store is
    keyed by series_id alone — so persisting a bounded result would make
    every later unbounded call read the fragment back as complete. The
    router below honours realtime_start the way the real API does, which
    is what makes this test able to fail.
    """
    import urllib.parse

    full = {"1991-01-01": {"X_20160223": "500.0", "X_20160524": "502.0"},
            "1991-04-01": {"X_20160223": "505.0", "X_20160524": "507.0"}}

    def _server_side_filtering(url, timeout=None, **kw):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        start = q.get("realtime_start", ["1776-07-04"])[0]
        obs = []
        for date, cols in full.items():
            row = {"date": date}
            for col, val in cols.items():
                token = col.split("_")[-1]
                vintage = f"{token[:4]}-{token[4:6]}-{token[6:]}"
                if vintage >= start:
                    row[col] = val
            obs.append(row)
        return json.dumps({"observations": obs}).encode()

    monkeypatch.setattr(A, "safe_get_bytes_cached", _server_side_filtering)

    bounded = A.alfred_vintages("X", store=store, start_vintage="2016-03-01")
    assert bounded["vintage"].nunique() == 1

    complete = A.alfred_vintages("X", store=store)
    assert complete["vintage"].nunique() == 2, (
        "the bounded fetch was persisted and read back as the full archive"
    )
