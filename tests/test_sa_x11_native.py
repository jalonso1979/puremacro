"""Native X-11/ARIMA engine vs frozen X-13ARIMA-SEATS goldens.

The goldens under ``tests/fixtures/sa_goldens/`` were produced by the
REAL X-13ARIMA-SEATS binary (v1.1.57) with the spec pinned to the
native v1 configuration (log/none + airline + no outlier + 60-period
extension + x11 seasonalma=msr); see ``tools/gen_validation_goldens_sa.py``.
CI never runs the binary — only these frozen CSVs.

Tolerances are MEASURED (2026-07-21), then frozen with headroom — they
are a record of achieved fidelity, not a promise. Two documented
divergence sources: (i) the binary does not use backcasts inside the
X-11 filters (its B2 starts period/2 observations in) and applies
asymmetric filters at the series START, while the native engine uses a
symmetric-filter-over-backcasts design — hence the looser boundary
tolerance; (ii) extreme-value replacement details differ at flagged
outlier months on wild small-count series — hence interior MAX
tolerances of several percent for tiny counties while interior MEDIANS
sit under ~1.2% everywhere and under 0.1% for smooth macro series.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.sa._airline import double_difference, extend, fit_airline
from puremacro.sa.x11 import (
    deseasonalize_x11,
    henderson_weights,
    x11_arima,
)

GOLD = Path(__file__).parent / "fixtures" / "sa_goldens"

#: (interior max, interior median, boundary max) — RE-measured
#: 2026-07-22 after the B17/B20 weight cascade landed (batch 8; interior
#: maxima roughly halved and medians ~2x tighter vs the v1 condensed
#: chain), frozen with ~1.4x headroom. Interior = 5 years from each end.
TOL = {
    "allegheny_pa_u": (0.055, 0.006, 0.045),
    "fayette_ky_u": (0.090, 0.007, 0.070),
    "marquette_mi_u": (0.080, 0.012, 0.075),
    "caddo_la_u": (0.050, 0.009, 0.065),
    "keweenaw_mi_u": (0.060, 0.008, 0.170),
    "hidalgo_nm_u": (0.095, 0.012, 0.045),
    "indpro_nsa": (0.010, 0.0014, 0.008),
    "allegheny_pa_u_q": (0.032, 0.0035, 0.020),
    "keweenaw_mi_u_q": (0.045, 0.010, 0.030),
}
_FAST = ["indpro_nsa", "keweenaw_mi_u", "allegheny_pa_u_q"]


def test_henderson_weights_match_published_13_term():
    w = henderson_weights(13)
    ref = np.array([-0.019, -0.028, 0.0, 0.066, 0.147, 0.214, 0.240])
    assert np.allclose(w[:7], ref, atol=6e-4)   # table rounds to 3 dp
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, w[::-1])
    for n in (5, 7, 9, 23):
        assert np.isclose(henderson_weights(n).sum(), 1.0)


def test_airline_mle_recovers_parameters():
    rng = np.random.default_rng(1)
    s, T, th, Th = 12, 480, 0.6, 0.5
    eps = rng.normal(0, 1.0, T + 200)
    w = eps.copy()
    w[1:] += -th * eps[:-1]
    w[s:] += -Th * eps[:-s]
    w[s + 1:] += th * Th * eps[:-s - 1]
    w = w[100:100 + T]
    y = np.zeros(T + s + 1) + 100.0
    for t in range(s + 1, T + s + 1):
        y[t] = w[t - s - 1] + y[t - 1] + y[t - s] - y[t - s - 1]
    fit = fit_airline(y[s + 1:], s)
    assert fit.converged
    assert abs(fit.theta - th) < 0.12
    assert abs(fit.Theta - Th) < 0.12
    assert abs(fit.sigma2 - 1.0) < 0.25
    ext = extend(y[s + 1:], fit, 60, 60)
    assert len(ext) == T + 120 and np.isfinite(ext).all()
    assert len(double_difference(y, s)) == len(y) - s - 1


def _check_golden(name: str) -> None:
    meta = json.loads((GOLD / "meta.json").read_text())["series"][name]
    df = pd.read_csv(GOLD / f"{name}.csv")
    y = df["y"].to_numpy()
    p = meta["period"]
    res = x11_arima(y, period=p,
                    mode="mult" if meta["transform"] == "log" else "add")
    tol_max, tol_med, tol_bound = TOL[name]
    b = 5 * p
    for tab, nat in (("d11", res.sa), ("d10", res.seasonal)):
        rel = np.abs(nat / df[tab].to_numpy() - 1.0)
        interior = rel[b: len(y) - b]
        assert interior.max() < tol_max, (name, tab, interior.max())
        assert np.median(interior) < tol_med, (name, tab,
                                               np.median(interior))
        assert max(rel[:b].max(), rel[-b:].max()) < tol_bound, (
            name, tab, max(rel[:b].max(), rel[-b:].max()))
    assert res.seasonal_filter in ("3x3", "3x5", "3x9")
    assert np.isfinite(res.sa).all() and len(res.sa) == len(y)


@pytest.mark.parametrize("name", _FAST)
def test_x11_matches_binary_goldens_fast(name):
    _check_golden(name)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(set(TOL) - set(_FAST)))
def test_x11_matches_binary_goldens_full(name):
    _check_golden(name)


def test_x11_removes_seasonality_multiplicative_toy():
    rng = np.random.default_rng(0)
    t = np.arange(420)
    y = np.exp(4 + 0.002 * t + 0.08 * np.sin(2 * np.pi * (t % 12) / 12)
               + rng.normal(0, 0.01, 420))
    res = x11_arima(y, period=12)          # mode='auto' -> mult
    assert res.mode == "mult"

    def amp(x):
        return np.abs(np.array([x[(t % 12) == m].mean()
                                for m in range(12)]) - x.mean()).max()

    # The calendar-month-mean metric retains trend-curvature effects:
    # the REAL binary's D11 on this exact series scores ~0.15*amp(y)
    # too (native matches it to 0.6%), so 0.2 is the honest bound.
    assert amp(res.sa) < 0.2 * amp(y)
    assert np.allclose(res.sa * res.seasonal, y, rtol=1e-10)


def test_deseasonalize_x11_group_contract():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2000-01-01", periods=160, freq="QS")
    t = np.arange(160)
    rows = []
    for unit in ("A", "B"):
        y = 50 + 0.1 * t + 5 * np.sin(2 * np.pi * (t % 4) / 4) \
            + rng.normal(0, 0.5, 160)
        rows.append(pd.DataFrame({"unit": unit, "date": dates, "v": y}))
    short = pd.DataFrame({"unit": "C", "date": dates[:10],
                          "v": np.ones(10)})
    df = pd.concat(rows + [short], ignore_index=True)
    sa = deseasonalize_x11(df, "v", by="unit", date_col="date", freq="Q")
    assert sa.index.equals(df.index)
    assert sa[df["unit"] == "C"].isna().all()      # too short -> NaN
    assert sa[df["unit"] == "A"].notna().all()


def test_x11_arima_input_validation():
    y = np.ones(200)
    with pytest.raises(ValueError, match="period"):
        x11_arima(y, period=7)
    with pytest.raises(ValueError, match="gap-free"):
        x11_arima(np.array([1.0, np.nan] * 100), period=4)
    with pytest.raises(ValueError, match="positive"):
        x11_arima(np.concatenate([[-1.0], np.ones(199)]), period=4,
                  mode="mult")


@pytest.mark.parametrize("name,bound_tol", [
    ("keweenaw_mi_u", 0.35), ("allegheny_pa_u", 0.055),
    ("indpro_nsa", 0.008),
])
def test_x11_vs_binary_default_left_end(name, bound_tol):
    """Against the binary's DEFAULT left-end mode (maxback=0 goldens,
    sa_goldens_lb/). Batch-7 measurements used this set to prove the
    residual noisy-series gaps were extreme-replacement (B17/B20)
    differences, not end-filter differences; batch 8 then implemented
    the cascade, which tightened agreement with the PINNED maxback=60
    spec everywhere (see TOL) — at the price of drifting from this
    no-backcast mode at the wild Keweenaw left boundary (9.6% -> 24.6%
    measured; our pinned spec generates backcasts, the default does
    not). Clean-series numbers improved against BOTH modes."""
    gd = Path(__file__).parent / "fixtures" / "sa_goldens_lb"
    meta = json.loads((gd / "meta.json").read_text())["series"][name]
    df = pd.read_csv(gd / f"{name}.csv")
    y = df["y"].to_numpy()
    p = meta["period"]
    res = x11_arima(y, period=p, mode="mult")
    rel = np.abs(res.sa / df["d11"].to_numpy() - 1.0)
    b = 5 * p
    assert max(rel[:b].max(), rel[-b:].max()) < bound_tol, (
        name, max(rel[:b].max(), rel[-b:].max()))


@pytest.mark.pyodide_smoke
def test_x11_runs_pure_python_no_binary():
    """The native engine needs no external binary — the property that
    makes it the browser default. A short synthetic multiplicative
    series must adjust and strip its seasonal."""
    import numpy as _np
    t = _np.arange(180)
    y = _np.exp(3.0 + 0.001 * t
                + 0.06 * _np.sin(2 * _np.pi * (t % 12) / 12))
    res = x11_arima(y, period=12, mode="mult")

    def _amp(x):
        return _np.abs(_np.array([x[(t % 12) == m].mean()
                                  for m in range(12)]) - x.mean()).max()

    assert res.mode == "mult"
    assert _amp(res.sa) < 0.25 * _amp(y)
    assert _np.isfinite(res.sa).all()


def test_deseasonalize_x13_falls_back_to_native_then_stl(monkeypatch):
    """The x13 group adjuster prefers binary -> native X-11 -> STL. With
    the binary forced absent, a well-behaved series must route to the
    NATIVE engine (not STL); a too-short one falls through to STL."""
    import puremacro.sa.x13 as x13
    from puremacro.sa.x13 import deseasonalize_x13
    monkeypatch.setattr(x13, "_X13_DIR", None)  # simulate no binary

    rng = np.random.default_rng(0)
    dates = pd.date_range("1990-01-01", periods=180, freq="MS")
    t = np.arange(180)
    y = np.exp(4 + 0.002 * t + 0.08 * np.sin(2 * np.pi * (t % 12) / 12)
               + rng.normal(0, 0.01, 180))
    good = pd.DataFrame({"u": "A", "date": dates, "v": y})
    eng: dict = {}
    sa = deseasonalize_x13(good, "v", by="u", date_col="date", freq="M",
                           engines=eng)
    assert eng["A"] == "native"
    assert sa.notna().all()

    short = pd.DataFrame({"u": "B", "date": dates[:30], "v": y[:30]})
    eng2: dict = {}
    deseasonalize_x13(short, "v", by="u", date_col="date", freq="M",
                      min_obs=24, engines=eng2)
    assert eng2["B"] == "stl"
