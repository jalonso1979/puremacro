"""Tests for puremacro.pocket — offline data cartridges."""
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from puremacro import pocket
from puremacro.runtime import store


@pytest.fixture
def panel():
    idx = pd.MultiIndex.from_product(
        [["MEX", "USA"], pd.period_range("2020Q1", periods=4, freq="Q")],
        names=["code", "date"],
    )
    return pd.DataFrame(
        {"gdp": np.arange(8.0), "infl": np.arange(8.0) / 2}, index=idx,
    )


@pytest.fixture
def meta():
    return pd.DataFrame({"code": ["MEX", "USA"], "currency": ["MXN", "USD"]})


@pytest.mark.pyodide_smoke
def test_round_trip_single_frame(tmp_path, panel):
    path = pocket.pack(panel, tmp_path / "one.pmz", source="OECD QNA",
                       vintage="2026-08-19", notes="demo")
    cart = pocket.load(path)
    pd.testing.assert_frame_equal(cart.frame(), panel)
    assert cart.provenance.source == "OECD QNA"
    assert cart.provenance.vintage == "2026-08-19"
    assert cart.provenance.notes == "demo"
    assert cart.provenance.created.endswith("Z")


def test_round_trip_multiple_frames(tmp_path, panel, meta):
    path = pocket.pack({"panel": panel, "meta": meta}, tmp_path / "two.pmz")
    cart = pocket.load(path)
    assert len(cart) == 2 and "panel" in cart
    pd.testing.assert_frame_equal(cart["panel"], panel)
    pd.testing.assert_frame_equal(cart["meta"], meta)
    assert cart.verify() is True


def test_frame_without_a_name_needs_one_when_ambiguous(tmp_path, panel, meta):
    path = pocket.pack({"panel": panel, "meta": meta}, tmp_path / "two.pmz")
    cart = pocket.load(path)
    with pytest.raises(pocket.CartridgeError, match="name the one you want"):
        cart.frame()


def test_unknown_frame_name_lists_what_is_there(tmp_path, panel):
    cart = pocket.load(pocket.pack(panel, tmp_path / "one.pmz"))
    with pytest.raises(KeyError, match="data"):
        cart["nope"]


def test_manifest_records_shape_and_index(tmp_path, panel):
    path = pocket.pack(panel, tmp_path / "one.pmz")
    manifest = pocket.inspect_cartridge(path)
    assert manifest["format"] == "puremacro-cartridge"
    (record,) = manifest["frames"]
    assert record["n_rows"] == 8 and record["n_cols"] == 2
    assert record["columns"] == ["gdp", "infl"]
    assert "MultiIndex" in record["index"] and "PeriodIndex[Q-DEC]" in record["index"]


def test_corrupt_payload_is_detected(tmp_path, panel):
    """Flip bytes inside the stored frame, leaving the zip structure valid."""
    path = pocket.pack(panel, tmp_path / "one.pmz")
    with zipfile.ZipFile(path) as zf:
        manifest = zf.read("manifest.json")
        payload = bytearray(zf.read("frames/data.npz"))
    payload[-20:] = bytes(20)  # inside the deflated block, not the header
    rewritten = tmp_path / "corrupt.pmz"
    with zipfile.ZipFile(rewritten, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json", manifest)
        zf.writestr("frames/data.npz", bytes(payload))
    with pytest.raises(pocket.CartridgeError, match="checksum"):
        pocket.load(rewritten)


def test_verify_catches_a_frame_mutated_after_load(tmp_path, panel):
    cart = pocket.load(pocket.pack(panel, tmp_path / "one.pmz"))
    cart.frames["data"].iloc[0, 0] = 999.0
    with pytest.raises(pocket.CartridgeError, match="changed since"):
        cart.verify()


def test_not_a_cartridge(tmp_path):
    junk = tmp_path / "junk.pmz"
    junk.write_bytes(b"not a zip at all")
    with pytest.raises(pocket.CartridgeError, match="not a cartridge"):
        pocket.load(junk)


def test_zip_without_manifest_is_rejected(tmp_path):
    path = tmp_path / "bare.pmz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(pocket.CartridgeError, match="manifest"):
        pocket.load(path)


def test_future_format_version_is_refused(tmp_path, panel):
    path = pocket.pack(panel, tmp_path / "one.pmz")
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        payload = zf.read("frames/data.npz")
    manifest["version"] = pocket.FORMAT_VERSION + 1
    newer = tmp_path / "newer.pmz"
    with zipfile.ZipFile(newer, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("frames/data.npz", payload)
    with pytest.raises(pocket.CartridgeError, match="not readable"):
        pocket.load(newer)


@pytest.mark.pyodide_smoke
def test_base64_transport_round_trip(tmp_path, panel):
    path = pocket.pack(panel, tmp_path / "one.pmz")
    text = pocket.to_base64(path)
    assert max(len(line) for line in text.splitlines()) <= 76
    cart = pocket.from_base64(text)
    pd.testing.assert_frame_equal(cart.frame(), panel)
    # ... and via a file, the way it arrives on the tablet.
    written = pocket.from_base64(text, tmp_path / "pasted.pmz")
    pd.testing.assert_frame_equal(pocket.load(written).frame(), panel)


def test_base64_survives_reflowed_whitespace(tmp_path, panel):
    text = pocket.to_base64(pocket.packs(panel))
    mangled = "  " + text.replace("\n", " \n\t") + "\n\n"
    pd.testing.assert_frame_equal(pocket.from_base64(mangled).frame(), panel)


def test_mangled_base64_says_so():
    with pytest.raises(pocket.CartridgeError, match="not valid base64"):
        pocket.from_base64("this is not base64 !!!")


def test_snapshot_records_the_reproducing_call(tmp_path):
    def build_panel(codes, start, freq="Q"):
        return pd.DataFrame({"c": [1.0]}, index=[start])

    path = pocket.snapshot(build_panel, ["USA", "MEX"], 1990, freq="Q",
                           path=tmp_path / "snap.pmz", source="test")
    cart = pocket.load(path)
    assert cart.provenance.call == "build_panel(['USA', 'MEX'], 1990, freq='Q')"
    assert cart.provenance.source == "test"


def test_snapshot_without_a_path_returns_bytes():
    payload = pocket.snapshot(lambda: pd.DataFrame({"a": [1.0]}))
    assert isinstance(payload, bytes)
    assert pocket.loads(payload).frame().shape == (1, 1)


def test_pack_rejects_non_frames():
    with pytest.raises(pocket.CartridgeError, match="not a DataFrame"):
        pocket.packs({"a": [1, 2, 3]})
    with pytest.raises(pocket.CartridgeError, match="DataFrame or a"):
        pocket.packs(np.arange(3))
    with pytest.raises(pocket.CartridgeError, match="empty"):
        pocket.packs({})


def test_summary_mentions_provenance_and_frames(tmp_path, panel, meta):
    path = pocket.pack({"panel": panel, "meta": meta}, tmp_path / "two.pmz",
                       source="OECD", vintage="2026-08-19")
    text = pocket.load(path).summary()
    assert "OECD" in text and "2026-08-19" in text
    assert "[panel]" in text and "[meta]" in text


def test_cartridge_reads_with_numpy_and_stdlib_only(tmp_path, panel):
    """A cartridge must be openable without puremacro's own loader."""
    path = pocket.pack(panel, tmp_path / "one.pmz")
    with zipfile.ZipFile(path) as zf:
        assert json.loads(zf.read("manifest.json"))["version"] == pocket.FORMAT_VERSION
        raw = zf.read("frames/data.npz")
    pd.testing.assert_frame_equal(store.loads_frame(raw), panel)
