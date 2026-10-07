"""X-13 results are memoised on disk, keyed by the input series."""
import numpy as np
import pandas as pd

from puremacro.sa import x13


def _series(n=48, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    v = 100 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 4) + rng.normal(0, 0.3, n)
    return v, pd.date_range("2000-01-01", periods=n, freq="QS")


def test_second_adjustment_of_the_same_series_reads_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PUREMACRO_HTTP_NO_CACHE", raising=False)
    v, d = _series()
    calls = []
    real = x13._x13_one_uncached

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(x13, "_x13_one_uncached", counting)
    tag1, tag2 = [], []
    a = x13._x13_one(v, d, 4, engine=tag1)
    b = x13._x13_one(v, d, 4, engine=tag2)
    assert len(calls) == 1
    np.testing.assert_array_equal(a, b)
    assert tag1 == tag2 and tag1[0] in ("binary", "native", "stl")


def test_a_changed_value_misses_and_no_cache_bypasses(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREMACRO_HTTP_CACHE_DIR", str(tmp_path))
    v, d = _series()
    w = v.copy(); w[10] += 1.0
    assert x13._memo_key(v, d, 4) != x13._memo_key(w, d, 4)
    monkeypatch.setenv("PUREMACRO_HTTP_NO_CACHE", "1")
    x13._x13_one(v, d, 4)
    assert x13._memo_get(x13._memo_key(v, d, 4)) is None
