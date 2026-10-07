from pathlib import Path
import pytest
from puremacro.cache import disk_cache_path

def test_disk_cache_path_defaults(monkeypatch):
    # Ensure environment variable is not set to avoid interference from other tests
    monkeypatch.delenv("PUREMACRO_CACHE_DIR", raising=False)

    path = disk_cache_path("mykey")
    expected_base = Path.home() / ".cache" / "puremacro"
    assert path == expected_base / "default" / "mykey.parquet"

def test_disk_cache_path_custom_namespace_and_suffix(monkeypatch):
    monkeypatch.delenv("PUREMACRO_CACHE_DIR", raising=False)

    path = disk_cache_path("mykey", namespace="custom", suffix=".json")
    expected_base = Path.home() / ".cache" / "puremacro"
    assert path == expected_base / "custom" / "mykey.json"

def test_disk_cache_path_sanitization(monkeypatch):
    monkeypatch.delenv("PUREMACRO_CACHE_DIR", raising=False)

    # Alphanumeric, '-', '_', '.' are preserved. Everything else becomes '_'.
    key = "a/b:c d?e@f#g$h%i^j&k*l(m)n+o-p_q.r"
    # replacements:
    # / -> _
    # : -> _
    # space -> _
    # ? -> _
    # @ -> _
    # # -> _
    # $ -> _
    # % -> _
    # ^ -> _
    # & -> _
    # * -> _
    # ( -> _
    # ) -> _
    # + -> _
    path = disk_cache_path(key)
    expected_filename = "a_b_c_d_e_f_g_h_i_j_k_l_m_n_o-p_q.r.parquet"
    assert path.name == expected_filename

def test_disk_cache_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PUREMACRO_CACHE_DIR", str(tmp_path / "my_custom_cache"))

    path = disk_cache_path("key")
    assert path == tmp_path / "my_custom_cache" / "default" / "key.parquet"
