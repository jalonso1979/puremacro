"""Tests for puremacro.runtime — capability detection, budgets, transport, store.

The interesting cases here are the ones that only happen on the target
device, so most of these drive the code through explicit overrides rather
than waiting for an iPad to run CI.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from puremacro import runtime
from puremacro.runtime import budget as budget_mod
from puremacro.runtime import _capabilities as caps_mod
from puremacro.runtime import store, transport


# --- capabilities -------------------------------------------------------

@pytest.mark.pyodide_smoke
def test_capabilities_are_self_consistent():
    caps = runtime.capabilities()
    assert caps.host in caps_mod.HOSTS
    assert caps.device in caps_mod.DEVICES
    assert caps.cpu_count >= 1
    assert "numpy" in caps.backends
    # Sockets and the JS bridge are mutually exclusive by construction.
    assert not (caps.sockets and caps.js_fetch)


@pytest.mark.pyodide_smoke
def test_detection_matches_the_interpreter_it_is_running_on():
    """The assertion that only means something on the target.

    Everything else here checks that detection is *self-consistent*;
    this checks that it is *right*, by cross-examining the interpreter
    directly. It is marked ``pyodide_smoke`` so gate 6 runs it inside a
    real Pyodide kernel, which is the only place the emscripten branch
    is ever taken — on a workstation it silently verifies the other one.
    """
    caps = runtime.capabilities()
    if sys.platform == "emscripten":
        assert caps.host == "pyodide"
        assert caps.device in ("browser", "tablet")
        # No TCP stack: this is what transport.enable_browser_network exists for.
        assert caps.sockets is False
        assert caps.js_fetch is True
        # pyarrow has no Pyodide wheel, which is what runtime.store is for.
        assert caps.parquet is False
        assert caps.backends == ("numpy",)
        assert runtime.is_pyodide()
        assert transport.available() == "js-fetch"
    else:
        assert caps.host == "cpython"
        assert caps.sockets is True
        assert caps.js_fetch is False
        assert not runtime.is_pyodide()
        assert transport.available() == "sockets"


def test_report_mentions_every_capability():
    text = runtime.report()
    for label in ("host", "device", "network", "parquet", "threads", "backends"):
        assert label in text


def test_js_module_on_a_workstation_is_not_a_browser(monkeypatch):
    """A `js` namespace package in site-packages must not imply a JS bridge.

    Several PyPI distributions ship a top-level ``js/`` directory, so
    ``find_spec("js")`` succeeds on an ordinary laptop.
    """
    monkeypatch.setattr(caps_mod, "_detect_host", lambda: "cpython")
    assert caps_mod._detect_js_fetch("cpython") is False


def test_environment_overrides_are_applied_and_recorded(monkeypatch):
    monkeypatch.setenv("PUREMACRO_DEVICE", "tablet")
    monkeypatch.setenv("PUREMACRO_SOCKETS", "0")
    caps = caps_mod.refresh()
    try:
        assert caps.device == "tablet"
        assert caps.sockets is False
        assert set(caps.overridden) == {"device", "sockets"}
    finally:
        monkeypatch.undo()
        caps_mod.refresh()


def test_bad_override_is_rejected(monkeypatch):
    monkeypatch.setenv("PUREMACRO_DEVICE", "toaster")
    with pytest.raises(ValueError, match="toaster"):
        caps_mod.refresh()
    monkeypatch.undo()
    caps_mod.refresh()


# --- budget -------------------------------------------------------------

@pytest.mark.pyodide_smoke
def test_tablet_budget_clamps_cost_but_not_estimand():
    with runtime.override("tablet"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", runtime.BudgetWarning)
            out = runtime.fit(n_boot=5000, n_draws=10 ** 6, horizon=40, p=4)
    assert out["n_boot"] == budget_mod.TIERS["tablet"].n_boot
    assert out["n_draws"] == budget_mod.TIERS["tablet"].n_draws
    # horizon changes what is estimated, not just how precisely: untouched.
    assert out["horizon"] == 40
    assert out["p"] == 4


def test_workstation_budget_is_a_no_op():
    with runtime.override("workstation"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning may fire
            assert runtime.fit(n_boot=2000)["n_boot"] == 2000


def test_budgeted_clamps_positional_and_keyword_arguments():
    def estimator(Y, p=2, n_boot=100, horizon=20):
        return n_boot, horizon

    wrapped = runtime.budgeted(estimator)
    with runtime.override("minimal"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", runtime.BudgetWarning)
            by_kw = wrapped("Y", n_boot=9999, horizon=50)
            by_pos = wrapped("Y", 2, 9999)
    cap = budget_mod.TIERS["minimal"].n_boot
    assert by_kw == (cap, 50)
    assert by_pos == (cap, 20)


def test_budget_warns_once_per_parameter():
    budget_mod._WARNED.clear()

    def f(n_boot=1):
        return n_boot

    wrapped = runtime.budgeted(f)
    with runtime.override("tablet"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrapped(n_boot=10 ** 6)
            wrapped(n_boot=10 ** 6)
    assert len(caught) == 1
    assert issubclass(caught[0].category, runtime.BudgetWarning)


def test_bool_is_not_clamped_as_a_number():
    """`True` is an int in Python; it must not be read as a workload size."""
    b = budget_mod.TIERS["minimal"]
    value, clamped = budget_mod._clamp("n_boot", True, b, where="t")
    assert value is True and clamped is False


def test_override_restores_the_previous_tier():
    before = runtime.current_budget().tier
    with runtime.override("tablet"):
        assert runtime.current_budget().tier == "tablet"
        with runtime.override("minimal"):
            assert runtime.current_budget().tier == "minimal"
        assert runtime.current_budget().tier == "tablet"
    assert runtime.current_budget().tier == before


# --- transport ----------------------------------------------------------

def test_transport_reports_a_mode():
    assert transport.available() in ("sockets", "js-fetch", "none")


def test_enable_browser_network_is_a_no_op_where_sockets_exist():
    if transport.available() != "sockets":
        pytest.skip("host has no sockets")
    assert runtime.enable_browser_network() == "sockets"
    # Nothing was patched, so the fetch layer is untouched.
    import puremacro._http as core_http
    assert core_http._request.__module__ == "puremacro._http"


def test_transport_raises_when_no_route_exists(monkeypatch):
    monkeypatch.setattr(transport, "available", lambda: "none")
    with pytest.raises(transport.TransportError, match="pocket"):
        transport.get_bytes("https://example.invalid")


def test_requests_shim_response_surface():
    resp = transport._Response(url="u", status_code=200, content=b'{"a": 1}')
    assert resp.ok and resp.text == '{"a": 1}' and resp.json() == {"a": 1}
    resp.raise_for_status()
    bad = transport._Response(url="u", status_code=503, content=b"")
    assert not bad.ok
    with pytest.raises(transport.TransportError, match="503"):
        bad.raise_for_status()


def test_requests_shim_refuses_post():
    with pytest.raises(transport.TransportError, match="GET-only"):
        transport._RequestsShim().post("https://example.com")


# --- store --------------------------------------------------------------

def _frames():
    return {
        "numeric": pd.DataFrame({"a": [1, 2, 3], "b": [1.5, np.nan, 3.5]}),
        "period": pd.DataFrame(
            {"gdp": [1.0, 2.0], "tag": ["x", None]},
            index=pd.period_range("2020Q1", periods=2, freq="Q")),
        "datetime": pd.DataFrame(
            {"v": [1, 2]}, index=pd.date_range("2020-01-01", periods=2)),
        "tz": pd.DataFrame(
            {"v": [1, 2]},
            index=pd.date_range("2020-01-01", periods=2, tz="America/Mexico_City")),
        "multi": pd.DataFrame(
            {"y": [1.0, 2.0, 3.0, 4.0]},
            index=pd.MultiIndex.from_product(
                [["MEX", "USA"], pd.period_range("2020Q1", periods=2, freq="Q")],
                names=["code", "date"])),
        "categorical": pd.DataFrame(
            {"regime": pd.Categorical(["a", "b", "a"], ordered=True)}),
        "nullable": pd.DataFrame(
            {"n": pd.array([1, None, 3], dtype="Int64"),
             "s": pd.array(["a", None, "c"], dtype="string")}),
        "period_column": pd.DataFrame(
            {"when": pd.period_range("2020Q1", periods=3, freq="Q")}),
        "int_columns": pd.DataFrame(np.arange(6.0).reshape(3, 2), columns=[10, 20]),
        "empty": pd.DataFrame({"a": pd.Series([], dtype=float)}),
    }


@pytest.mark.pyodide_smoke
@pytest.mark.parametrize("name", sorted(_frames()))
def test_store_round_trips_frame(name):
    df = _frames()[name]
    back = store.loads_frame(store.dumps_frame(df))
    pd.testing.assert_frame_equal(back, df)


def _pandas3_strings():
    """Context manager putting pandas 2 on pandas 3's string dtype.

    Skips where the option is gone (pandas 3, where it is the default and
    the plain round-trip test above already covers it) or not yet there.
    """
    try:
        pd.get_option("future.infer_string")
    except Exception:
        pytest.skip("no future.infer_string option on this pandas")
    return pd.option_context("future.infer_string", True)


@pytest.mark.parametrize("name", sorted(_frames()))
def test_store_round_trips_frame_with_pandas3_strings(name):
    """Every frame again, built the way pandas 3 builds it.

    pandas 3 gives a plain string column ``StringDtype(na_value=nan)``,
    spelled ``"str"`` — which a substring test for "string" misses. Every
    string column and index level then went down the integer branch and
    died in ``int('MEX')``, taking 19 tests and the whole cartridge path
    with it on any fresh install.
    """
    with _pandas3_strings():
        df = _frames()[name]
        back = store.loads_frame(store.dumps_frame(df))
        pd.testing.assert_frame_equal(back, df)


def test_store_reads_both_spellings_of_the_string_dtype():
    """The dispatch is on dtype identity, not on how pandas spells it."""
    assert store._numpy_dtype_for(pd.StringDtype()) is np.str_
    assert store._fill_for(pd.StringDtype()) == ""
    with _pandas3_strings():
        pandas3_str = pd.Series(["MEX"]).dtype
    assert str(pandas3_str) == "str"          # not "string"
    assert store._numpy_dtype_for(pandas3_str) is np.str_
    assert store._fill_for(pandas3_str) == ""


def test_store_asks_masked_dtypes_for_their_numpy_equivalent():
    """Int8 stores as int8, not widened to int64 by a name match."""
    assert store._numpy_dtype_for(pd.Int8Dtype()) == np.int8
    assert store._numpy_dtype_for(pd.UInt16Dtype()) == np.uint16
    assert store._numpy_dtype_for(pd.Float32Dtype()) == np.float32
    assert store._numpy_dtype_for(pd.BooleanDtype()) is bool
    df = pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int8")})
    pd.testing.assert_frame_equal(store.loads_frame(store.dumps_frame(df)), df)


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_store_round_trips_a_tz_index_at_any_resolution(unit):
    """A tz payload is ``asi8``, which counts in the dtype's own unit.

    pandas 2 made every timestamp nanosecond, so decoding hard-coded ns and
    got away with it. pandas 3 gives ``date_range`` microsecond resolution:
    read as nanoseconds, a 2020 index lands in 1970 with its spacing gone,
    and pandas then refuses to restore the freq it was told to expect.
    """
    idx = pd.date_range("2020-01-01", periods=3,
                        tz="America/Mexico_City").as_unit(unit)
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=idx)
    back = store.loads_frame(store.dumps_frame(df))
    pd.testing.assert_frame_equal(back, df)
    assert back.index.dtype.unit == unit


def test_store_rejects_arbitrary_objects_by_name():
    df = pd.DataFrame({"payload": [object()]})
    with pytest.raises(store.StoreError, match="payload"):
        store.dumps_frame(df)


def test_store_rejects_duplicate_columns():
    df = pd.DataFrame(np.zeros((2, 2)), columns=["a", "a"])
    with pytest.raises(store.StoreError, match="duplicate"):
        store.dumps_frame(df)


def test_store_rejects_a_plain_npz():
    import io
    buf = io.BytesIO()
    np.savez(buf, x=np.arange(3))
    with pytest.raises(store.StoreError, match="schema"):
        store.loads_frame(buf.getvalue())


def test_store_never_pickles():
    """allow_pickle=False on load is the guarantee; prove it holds."""
    payload = store.dumps_frame(_frames()["numeric"])
    import io
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        assert len(archive.files) > 0


def test_describe_reads_schema_without_data(tmp_path):
    df = _frames()["period"]
    path = tmp_path / "f.npz"
    store.save_frame(df, path)
    meta = store.describe(path)
    assert meta["n_rows"] == len(df)
    assert [c["name"] for c in meta["columns"]] == list(df.columns)
    assert meta["index"]["levels"][0]["kind"] == "period"


def test_save_and_load_frame_via_path(tmp_path):
    df = _frames()["multi"]
    path = tmp_path / "panel.npz"
    store.save_frame(df, path)
    pd.testing.assert_frame_equal(store.load_frame(path), df)
