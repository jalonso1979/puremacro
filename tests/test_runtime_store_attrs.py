"""Panel provenance must survive the same portable route as its observations."""
import datetime as dt
import io
import json
from pathlib import Path
import struct
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from puremacro import pocket
from puremacro.runtime import store


def _panel():
    index = pd.MultiIndex.from_product(
        [["MEX", "USA", "DEU"], pd.date_range("2020-01-01", periods=2, freq="YS")],
        names=["code", "date"],
    )
    panel = pd.DataFrame({"gdp": np.arange(6.0)}, index=index)
    panel.attrs = {
        "source": "WDI + OECD", "fetched_at": "2026-10-10T00:00:00Z",
        "meta": ({"unit": "millions LCU", "first": pd.Timestamp("2020-01-01"),
                  "last": pd.Timestamp("2021-01-01"), "n": np.int64(6)},),
        "by_series": {("MEX", "gdp"): {"scale": np.float32(1e-6)}},
        "requests": [{"key": "A.MEX", "status": "HTTP 429"}],
        "missing": ({"code": "MEX", "variable": "hours", "reason": "timeout"},),
        "complete": False,
        "seams": [{"source": "archive", "accepted": False, "reason": "no overlap"}],
    }
    return panel


@pytest.mark.pyodide_smoke
@pytest.mark.parametrize("route", ["npz", "pmz", "npz_file", "pmz_file"])
def test_panel_provenance_survives_portable_roundtrip(route, tmp_path):
    panel = _panel()
    if route == "npz":
        loaded = store.loads_frame(store.dumps_frame(panel))
    elif route == "pmz":
        cart = pocket.loads(pocket.packs(panel))
        assert cart.verify()
        loaded = cart.frame()
    elif route == "npz_file":
        path = tmp_path / "panel.npz"
        store.save_frame(panel, path, compress=False)
        loaded = store.load_frame(path)
    else:
        path = tmp_path / "panel.pmz"
        pocket.pack(panel, path)
        cart = pocket.load(path)
        assert cart.verify()
        loaded = cart.frame()
    pd.testing.assert_frame_equal(loaded, panel)
    assert loaded.attrs == panel.attrs
    assert isinstance(loaded.attrs["meta"], tuple)
    assert type(loaded.attrs["by_series"][("MEX", "gdp")]["scale"]) is np.float32


def test_cartridge_verification_catches_metadata_mutation():
    cart = pocket.loads(pocket.packs(_panel()))
    cart.frame().attrs["complete"] = True
    with pytest.raises(pocket.CartridgeError, match="changed since"):
        cart.verify()


def test_original_v1_cartridge_loads_and_verifies():
    # Written by the unmodified v1 implementation, not a v2 writer in disguise.
    path = Path(__file__).parent / "fixtures" / "runtime_store_v1.pmz"
    cart = pocket.load(path)
    assert cart.provenance.source == "legacy storage regression"
    assert cart.frame().attrs == {}
    assert cart.verify()
    with zipfile.ZipFile(path) as archive:
        frame = store.loads_frame(archive.read("frames/data.npz"))
    pd.testing.assert_frame_equal(frame, cart.frame())
    cart.frame().iloc[0, 0] = -1
    with pytest.raises(pocket.CartridgeError, match="changed since"):
        cart.verify()


def test_original_v1_cartridge_can_be_repacked_with_new_metadata():
    cart = pocket.load(Path(__file__).parent / "fixtures" / "runtime_store_v1.pmz")
    cart.frame().attrs["source"] = "documented after the original export"
    # Version 1 had no attrs in its digest. Repacking upgrades the frame and
    # includes the new metadata in subsequent verification.
    assert cart.verify()
    updated = pocket.loads(pocket.packs(cart.frame()))
    assert updated.frame().attrs == cart.frame().attrs
    assert updated.verify()
    updated.frame().attrs["source"] = "changed"
    with pytest.raises(pocket.CartridgeError, match="changed since"):
        updated.verify()


@pytest.mark.parametrize("value", [
    None, "", True, 12, -0.0, float("inf"), float("-inf"), float("nan"),
    np.int16(4), np.uint64(2**63 + 1), np.float32(1.25), np.complex128(1 + 2j),
    np.bool_(False), np.str_(""), np.str_("méxico"), np.bytes_(b""),
    np.str_("a\0"), np.str_("\0"), np.bytes_(b"a\0"), np.bytes_(b"\0"),
    np.datetime64("2020-01-01", "D"), np.datetime64("NaT", "ns"),
    np.timedelta64(10, "h"), pd.NA, pd.NaT,
    pd.Timestamp("2026-01-01").as_unit("s"),
    pd.Timestamp("2026-01-01 01:02:03.000000004", tz="America/Mexico_City"),
    pd.Period("2020Q1"), pd.Timedelta(1, unit="ns"),
    dt.date(2020, 1, 1), dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc),
    dt.datetime(2020, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1),
    dt.timedelta(days=-1, microseconds=7),
    {("MEX", "gdp"): [1, None, ("LCU", 6)]},
    {"numpy": ["dict", "timestamp"]},  # User keys never act as type tags.
])
def test_metadata_scalar_and_container_types_roundtrip(value):
    frame = pd.DataFrame({"x": [1.0]})
    frame.attrs["value"] = value
    cart = pocket.loads(pocket.packs(frame))
    back = cart.frame().attrs["value"]
    assert type(back) is type(value)
    if isinstance(value, np.generic):
        assert back.dtype == value.dtype
        assert back.tobytes() == value.tobytes()
    elif value is pd.NA or value is pd.NaT:
        assert back is value
    elif type(value) is float and np.isnan(value):
        assert np.isnan(back)
    else:
        assert back == value
    assert cart.verify()  # Re-encoding is byte-stable, including date precision.


@pytest.mark.parametrize("month, day, hour, fold", [(1, 1, 12, 0), (7, 1, 12, 0), (11, 1, 1, 1)])
def test_dateutil_named_timestamp_retains_zone_offset_and_nanoseconds(month, day, hour, fold):
    from dateutil.tz import tzfile

    # A small TZif v1 fixture avoids depending on dateutil's optional bundled
    # database or a Unix-only /usr/share/zoneinfo path. It has New York's 2020
    # spring/fall transitions, including the repeated hour exercised below.
    raw = (b"TZif\0" + b"\0" * 15 + struct.pack(">6i", 0, 0, 0, 2, 2, 8)
           + struct.pack(">2i", 1583650800, 1604210400) + b"\1\0"
           + struct.pack(">iBBiBB", -18000, 0, 0, -14400, 1, 4) + b"EST\0EDT\0")
    zone = tzfile(io.BytesIO(raw), filename="/fixture/zoneinfo/America/New_York")
    stamp = pd.Timestamp(dt.datetime(2020, month, day, hour, 30, tzinfo=zone, fold=fold))
    stamp += pd.Timedelta(4, unit="ns")
    frame = pd.DataFrame({"x": [1.]})
    frame.attrs["timestamp"] = stamp
    cart = pocket.loads(pocket.packs(frame))
    restored = cart.frame().attrs["timestamp"]
    assert restored == stamp and restored.nanosecond == 4
    assert restored.utcoffset() == stamp.utcoffset() and restored.tzname() == stamp.tzname()
    assert restored.fold == stamp.fold
    assert (getattr(restored.tz, "key", None) or getattr(restored.tz, "zone", None)) == "America/New_York"
    assert cart.verify()


@pytest.mark.parametrize("name", ["CUSTOM", "America/New_York"])
@pytest.mark.parametrize("offset", [dt.timedelta(hours=2), dt.timedelta(hours=-3, minutes=-30)])
def test_custom_named_fixed_timestamp_preserves_name_without_becoming_a_named_zone(name, offset):
    stamp = pd.Timestamp("2020-07-01 12:34:56.123456789", tz=dt.timezone(offset, name))
    frame = pd.DataFrame({"x": [1.]})
    frame.attrs["timestamp"] = stamp
    cart = pocket.loads(pocket.packs(frame))
    restored = cart.frame().attrs["timestamp"]
    assert restored == stamp and restored.nanosecond == stamp.nanosecond
    assert isinstance(restored.tz, dt.timezone)
    assert restored.utcoffset() == offset and restored.tzname() == name
    assert cart.verify()


@pytest.mark.parametrize("zone, wire_zone", [(None, None), (dt.timezone.utc, "UTC"),
    (dt.timezone(dt.timedelta(hours=2)), "UTC+02:00"), (ZoneInfo("America/New_York"), "America/New_York")])
def test_existing_v2_timestamp_timezone_strings_remain_byte_stable(zone, wire_zone):
    frame = pd.DataFrame({"x": [1.]})
    stamp = pd.Timestamp("2020-01-01", tz=zone)
    frame.attrs["timestamp"] = stamp
    payload = store.dumps_frame(frame)
    schema = store.describe(io.BytesIO(payload))
    # This is the original draft-v2 tag. Its bytes and subsequent cartridge
    # digests should not change for zones that already round-tripped correctly.
    assert schema["attrs"] == ["dict", [["timestamp", ["timestamp", [
        ["numpy", [stamp.asm8.dtype.str, stamp.asm8.tobytes().hex()]], wire_zone,
    ]]]]]
    assert store.dumps_frame(store.loads_frame(payload)) == payload


def test_unrecognized_timestamp_timezone_is_rejected_before_file_is_written(tmp_path):
    class UnrecognizedZone(dt.tzinfo):
        def utcoffset(self, value):
            return dt.timedelta(hours=2)

        def dst(self, value):
            return dt.timedelta(0)

        def tzname(self, value):
            return "not a reconstructible zone"

    frame = pd.DataFrame({"x": [1.]})
    frame.attrs["timestamp"] = pd.Timestamp("2020-01-01", tz=UnrecognizedZone())
    path = tmp_path / "unsupported.npz"
    with pytest.raises(store.StoreError, match=r"attrs\['timestamp'\]: unsupported timestamp timezone"):
        store.save_frame(frame, path)
    assert not path.exists()


def test_nested_metadata_is_independent_after_loading():
    original = _panel()
    shared = [1, 2]
    original.attrs["shared"] = [shared, shared]  # Shared is allowed; cyclic is not.
    back = store.loads_frame(store.dumps_frame(original))
    back.attrs["requests"][0]["status"] = "ok"
    assert original.attrs["requests"][0]["status"] == "HTTP 429"
    assert back.attrs["shared"] == [[1, 2], [1, 2]]


@pytest.mark.parametrize("value", [object(), np.arange(3), {1, 2}, pd.DataFrame({"x": [1]})])
def test_unsupported_metadata_names_the_path(value):
    frame = pd.DataFrame({"x": [1.0]})
    frame.attrs["meta"] = {"unsupported": value}
    with pytest.raises(store.StoreError, match=r"attrs\['meta'\]\['unsupported'\].*unsupported"):
        store.dumps_frame(frame)


@pytest.mark.parametrize("kind", ["dict", "list"])
def test_cyclic_metadata_is_refused(kind):
    frame = pd.DataFrame({"x": [1.0]})
    value = {} if kind == "dict" else []
    if kind == "dict":
        value["cycle"] = value
    else:
        value.append(value)
    frame.attrs["meta"] = value
    with pytest.raises(store.StoreError, match="attrs.*cyclic metadata"):
        store.dumps_frame(frame)


def test_excessively_nested_metadata_is_refused_on_write_and_read():
    frame = pd.DataFrame({"x": [1.0]})
    nested = "leaf"
    encoded = "leaf"
    for _ in range(102):
        nested = [nested]
        encoded = ["list", [encoded]]
    frame.attrs["nested"] = nested
    with pytest.raises(store.StoreError, match="nesting exceeds"):
        store.dumps_frame(frame)
    payload = _replace_metadata(
        store.dumps_frame(_panel()),
        lambda meta: meta.update(attrs=["dict", [["nested", encoded]]]),
    )
    with pytest.raises(store.StoreError, match="nesting exceeds"):
        store.loads_frame(payload)


def _replace_metadata(payload, mutate):
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    key = "__puremacro_schema__"
    meta = json.loads(bytes(arrays[key]))
    mutate(meta)
    arrays[key] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    return out.getvalue()


@pytest.mark.parametrize("attrs", [
    ["unknown", "anything"], ["dict", [[[], 1]]],
    ["numpy", ["object", "0000000000000000"]],
    ["numpy", ["float64", "00"]],
    ["dict", [["source", "one"], ["source", "two"]]],
    ["dict", ["ab"]], ["dict", {}], ["list", ""], ["tuple", {}],
    [[], "anything"], ["numpy", ["int64", "0000000000000000", "ignored"]],
    ["numpy", ["object"]], ["timestamp", ["2020-01-01", None]],
    ["timedelta", 1], ["datetime", []],
    ["timestamp", [["numpy", ["<M8[s]", "0000000000000000"]], ["fixed_offset", True, "CUSTOM"]]],
    ["timestamp", [["numpy", ["<M8[s]", "0000000000000000"]], ["fixed_offset", 86400000000, "CUSTOM"]]],
    ["timestamp", [["numpy", ["<M8[s]", "0000000000000000"]], ["fixed_offset", 0, {}]]],
])
def test_malformed_or_unsafe_metadata_is_refused(attrs):
    payload = _replace_metadata(
        store.dumps_frame(_panel()),
        lambda meta: meta.update(attrs=["dict", [["value", attrs]]]),
    )
    with pytest.raises(store.StoreError, match="attrs"):
        store.loads_frame(payload)


def test_frame_metadata_must_be_a_mapping():
    payload = _replace_metadata(
        store.dumps_frame(_panel()), lambda meta: meta.update(attrs=["list", [1, 2]]),
    )
    with pytest.raises(store.StoreError, match="must be a dict"):
        store.loads_frame(payload)


def test_future_schema_version_and_missing_v2_metadata_are_refused():
    payload = store.dumps_frame(_panel())
    future = _replace_metadata(payload, lambda meta: meta.update(version=store.SCHEMA_VERSION + 1))
    with pytest.raises(store.StoreError, match="schema version"):
        store.loads_frame(future)
    incomplete = _replace_metadata(payload, lambda meta: meta.pop("attrs"))
    with pytest.raises(store.StoreError, match="lacks metadata"):
        store.loads_frame(incomplete)


def test_metadata_archive_needs_no_pickle():
    with np.load(io.BytesIO(store.dumps_frame(_panel())), allow_pickle=False) as archive:
        for key in archive.files:
            assert not archive[key].dtype.hasobject
