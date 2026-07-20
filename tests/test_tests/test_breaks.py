import numpy as np

from puremacro.tests.breaks import bai_perron


def test_bai_perron_finds_known_breaks():
    """Series with two known breaks at t=80 and t=160 → algorithm should
    find them within ±10."""
    rng = np.random.default_rng(11)
    T = 240
    y = np.zeros(T)
    # Three regimes with different means
    y[:80] = rng.standard_normal(80)
    y[80:160] = 3.0 + rng.standard_normal(80)
    y[160:] = -2.0 + rng.standard_normal(80)
    out = bai_perron(y, max_breaks=3, trim=0.10)
    # Best-by-SSR partition for m=2 should hit our breaks
    breaks_m2 = out["breaks"][2]
    assert len(breaks_m2) == 2
    assert any(abs(b - 80) < 10 for b in breaks_m2)
    assert any(abs(b - 160) < 10 for b in breaks_m2)


def test_bai_perron_returns_documented_keys():
    rng = np.random.default_rng(13)
    y = rng.standard_normal(150)
    out = bai_perron(y, max_breaks=2, trim=0.15)
    for key in ("breaks", "ssrs", "selected_m", "max_breaks"):
        assert key in out
    # `breaks` is a dict mapping m -> list of break dates
    assert 0 in out["breaks"]
    assert 1 in out["breaks"]
    assert 2 in out["breaks"]
    assert out["breaks"][0] == []  # no breaks → empty list
