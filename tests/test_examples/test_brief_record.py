import hashlib, json, os, sys
import puremacro.examples._brief_record as br


def test_environment_keys():
    env = br.environment()
    assert set(env) == {"python", "platform", "numpy", "scipy", "pandas", "matplotlib",
                        "puremacro", "load_average", "peak_rss_gb"}


def test_load_average_none_without_getloadavg(monkeypatch):
    monkeypatch.delattr(os, "getloadavg", raising=False)
    assert br.environment()["load_average"] is None


def test_peak_rss_none_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert br.environment()["peak_rss_gb"] is None


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "a.bin"; p.write_bytes(b"puremacro")
    assert br.sha256(p) == hashlib.sha256(b"puremacro").hexdigest()


def test_write_json_atomic_replaces_and_leaves_nothing_else(tmp_path):
    p = tmp_path / "r.json"
    br.write_json_atomic(p, {"a": 1}); br.write_json_atomic(p, {"año": "holgura"})
    assert json.loads(p.read_text(encoding="utf-8")) == {"año": "holgura"}
    assert "año" in p.read_text(encoding="utf-8") and [q.name for q in tmp_path.iterdir()] == ["r.json"]
