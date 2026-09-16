"""Unit tests for Montiel Olea & Pflueger (2013) weak IV in LP-IV and LA-LP.

Verifies:
1. ``mop_critical_values`` reproduces the published MOP (2013) one-instrument
   critical values through the non-central chi-square closed form, honours
   ``alpha`` and ``tau``, rejects invalid inputs and agrees with the 23.1
   cutoff documented in ``puremacro.inference.weak_iv``.
2. ``compute_mop_effective_f`` equals the squared HAC t-stat for kz=1, equals
   ``puremacro.inference.weak_iv.olea_pflueger_f`` for kz=2 at zero lags,
   matches a hand-computed HAC F_eff, and is invariant to rescaling an
   instrument.
3. Multi-instrument Anderson-Rubin sets are exact: the endpoints solve
   AR(beta) = crit, agree with a brute-force fine-grid inversion, are
   equivariant under y -> c*y, do not depend on the 2SLS seed of the search,
   and reduce to the single-instrument closed form when kz=1.
4. Weak-instrument designs give small F_eff and explicitly unbounded AR sets.
5. ``la_lp`` / ``la_lp_iv`` (White-robust) share the same critical values and
   inversion, and ``la_lp_iv`` is re-exported from ``puremacro.lp``.
6. v3.3.0-style ``lp_iv`` / ``la_lp`` calls still work unchanged.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, ncx2

import puremacro.lp
from puremacro.inference.weak_iv import olea_pflueger_f
from puremacro.lp.iv import (
    _compute_anderson_rubin_ci,
    _compute_anderson_rubin_multi,
    compute_mop_effective_f,
    lp_iv,
    mop_critical_values,
)
from puremacro.lp.la_lp import la_lp, la_lp_iv

# ``puremacro.lp.la_lp`` the attribute is the re-exported *function*; the
# modules are needed to spy on the AR inversion.
_IV_MOD = importlib.import_module("puremacro.lp.iv")
_LA_MOD = importlib.import_module("puremacro.lp.la_lp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strong_kz2(scale: float = 1.0) -> pd.DataFrame:
    """Strong two-instrument design (true slope -0.6 at h=1)."""
    rng = np.random.default_rng(123)
    T = 300
    z1 = rng.standard_normal(T)
    z2 = rng.standard_normal(T)
    x = 0.7 * z1 + 0.6 * z2 + 0.5 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] - 0.6 * x[t - 1] + rng.standard_normal()
    return pd.DataFrame({"x": x, "y": y * scale, "z1": z1, "z2": z2})


def _weak_kz2(scale: float = 1.0) -> pd.DataFrame:
    """Two instruments with near-zero relevance (true slope -0.4 at h=1)."""
    rng = np.random.default_rng(303)
    T = 200
    z1 = rng.standard_normal(T)
    z2 = rng.standard_normal(T)
    x = 0.05 * z1 + 0.05 * z2 + rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] - 0.4 * x[t - 1] + rng.standard_normal()
    return pd.DataFrame({"x": x, "y": y * scale, "z1": z1, "z2": z2})


def _ar_stat_brute(y_target, x, W_ctl, Z_mat, lags, b0):
    """Per-point regression AR statistic, independent of the package's blockwise form."""
    W_inv = np.linalg.pinv(W_ctl.T @ W_ctl)
    yt = y_target - W_ctl @ (W_inv @ (W_ctl.T @ y_target))
    xt = x - W_ctl @ (W_inv @ (W_ctl.T @ x))
    Zt = Z_mat - W_ctl @ (W_inv @ (W_ctl.T @ Z_mat))
    G = np.linalg.pinv(Zt.T @ Zt)
    w = yt - b0 * xt
    delta = G @ (Zt.T @ w)
    e = w - Zt @ delta
    S = Zt * e[:, None]
    Om = S.T @ S
    for ell in range(1, lags + 1):
        wt = 1.0 - ell / (lags + 1.0)
        g = S[ell:].T @ S[:-ell]
        Om = Om + wt * (g + g.T)
    V = G @ Om @ G
    return float(delta @ np.linalg.solve(V, delta))


def _spy_multi(monkeypatch, module):
    """Capture the arguments the estimator hands to the multi-instrument AR inversion."""
    cap: dict = {}
    orig = _IV_MOD._compute_anderson_rubin_multi

    def spy(y_target, x, W_ctl, Z_mat, lags, alpha, beta_hat=0.0, se_hat=1.0):
        cap.update(y_target=y_target, x=x, W_ctl=W_ctl, Z_mat=Z_mat, lags=lags,
                   alpha=alpha, beta_hat=beta_hat, se_hat=se_hat)
        return orig(y_target, x, W_ctl, Z_mat, lags, alpha, beta_hat, se_hat)

    monkeypatch.setattr(module, "_compute_anderson_rubin_multi", spy)
    return cap


def _brute_stat_from_capture(cap, lags=None):
    lags = cap["lags"] if lags is None else lags
    return lambda b0: _ar_stat_brute(cap["y_target"], cap["x"], cap["W_ctl"], cap["Z_mat"], lags, b0)


# ---------------------------------------------------------------------------
# 1. Critical values
# ---------------------------------------------------------------------------

def test_mop_critical_values_match_published_mop_2013():
    """k_z=1 values are the published MOP (2013) TSLS critical values at the 5% level."""
    cv5, cv10, cv20, cv30 = mop_critical_values(1, tau=(0.05, 0.10, 0.20, 0.30))
    assert cv5 == pytest.approx(37.418, abs=5e-4)
    assert cv10 == pytest.approx(23.109, abs=5e-4)
    assert cv20 == pytest.approx(15.062, abs=5e-4)
    assert cv30 == pytest.approx(12.04, abs=1e-2)

    # Default keeps the (cv_10, cv_20) return shape.
    cv_10, cv_20 = mop_critical_values(1)
    assert cv_10 == pytest.approx(23.109, abs=5e-4)
    assert cv_20 == pytest.approx(15.062, abs=5e-4)
    # The same cutoff puremacro.inference.weak_iv.olea_pflueger_f documents.
    assert round(cv_10, 1) == 23.1

    # A scalar tau is accepted and returns a 1-tuple.
    assert mop_critical_values(1, tau=0.10) == (cv_10,)


def test_mop_critical_values_honour_alpha_and_k_z():
    """The closed form ncx2.ppf(1 - alpha, k, k / tau) / k is used for every k and alpha."""
    for k in (1, 2, 3, 5, 8):
        for alpha in (0.05, 0.10, 0.01):
            cv_10, cv_20 = mop_critical_values(k, alpha)
            assert cv_10 == pytest.approx(ncx2.ppf(1 - alpha, k, k / 0.10) / k, rel=1e-12)
            assert cv_20 == pytest.approx(ncx2.ppf(1 - alpha, k, k / 0.20) / k, rel=1e-12)
            assert cv_10 > cv_20
    # alpha is not dead: a looser level gives a smaller critical value.
    assert mop_critical_values(1, 0.10)[0] < mop_critical_values(1, 0.05)[0]
    # Decreasing in the number of instruments.
    cvs = [mop_critical_values(k)[0] for k in range(1, 11)]
    assert all(a > b for a, b in zip(cvs[:-1], cvs[1:]))


@pytest.mark.parametrize(
    "args, kwargs",
    [((0,), {}), ((-1,), {}), ((1, 0.0), {}), ((1, 1.0), {}),
     ((1,), {"tau": 0.0}), ((1,), {"tau": (0.1, 1.5)}), ((1,), {"tau": ()})],
)
def test_mop_critical_values_reject_invalid_inputs(args, kwargs):
    with pytest.raises(ValueError):
        mop_critical_values(*args, **kwargs)


# ---------------------------------------------------------------------------
# 2. Effective F
# ---------------------------------------------------------------------------

def test_lp_iv_mop_f_single_instrument():
    """mop_f equals first_stage_f (squared HAC t-stat) for a single instrument."""
    rng = np.random.default_rng(101)
    T = 250
    z = rng.standard_normal(T)
    x = 0.8 * z + 0.4 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] - 0.5 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z": z})
    out = lp_iv(df, y="y", x="x", z="z", horizons=[0, 1, 2], n_lags=1)

    assert "mop_f" in out.columns
    assert "mop_cv_10" in out.columns
    assert "mop_cv_20" in out.columns

    cv_10, cv_20 = mop_critical_values(1)
    for h in [0, 1, 2]:
        row = out.loc[out["h"] == h]
        fs_f = row["first_stage_f"].iloc[0]
        mop_f = row["mop_f"].iloc[0]
        # For kz=1, MOP effective F is mathematically identical to squared HAC t-stat
        np.testing.assert_allclose(mop_f, fs_f, rtol=1e-6)
        assert row["mop_cv_10"].iloc[0] == cv_10
        assert row["mop_cv_20"].iloc[0] == cv_20
        assert row["mop_cv_10"].iloc[0] == pytest.approx(23.109, abs=5e-4)


def test_compute_mop_effective_f_kz2_matches_olea_pflueger_f_and_hand_computation():
    """kz=2: equals inference.weak_iv.olea_pflueger_f at lags=0, a hand-computed HAC F_eff at lags=2,
    and is invariant to rescaling one instrument (the trace form, not tr(A) tr(B))."""
    rng = np.random.default_rng(7)
    T = 400
    W = np.column_stack([np.ones(T), rng.standard_normal(T)])
    Z = rng.standard_normal((T, 2))
    x = Z @ np.array([0.3, 0.2]) + W @ np.array([1.0, 0.5]) + rng.standard_normal(T)

    W_inv = np.linalg.pinv(W.T @ W)
    x_t = x - W @ (W_inv @ (W.T @ x))
    Z_t = Z - W @ (W_inv @ (W.T @ Z))

    # (a) lags=0 is exactly the pre-existing White-robust olea_pflueger_f on partialled data.
    np.testing.assert_allclose(compute_mop_effective_f(x, Z, W, lags=0), olea_pflueger_f(x_t, Z_t), rtol=1e-10)

    # (b) lags=2: hand-computed pi' Z'Z pi / tr((Z'Z)^-1 Omega_NW).
    lags = 2
    ZtZ = Z_t.T @ Z_t
    pi = np.linalg.solve(ZtZ, Z_t.T @ x_t)
    v = x_t - Z_t @ pi
    S = Z_t * v[:, None]
    Om = S.T @ S
    for ell in range(1, lags + 1):
        g = S[ell:].T @ S[:-ell]
        Om = Om + (1.0 - ell / (lags + 1.0)) * (g + g.T)
    f_hand = float(pi @ ZtZ @ pi) / float(np.trace(np.linalg.solve(ZtZ, Om)))
    np.testing.assert_allclose(compute_mop_effective_f(x, Z, W, lags=lags), f_hand, rtol=1e-10)

    # (c) invariance to rescaling an instrument.
    Z_scaled = Z * np.array([1.0, 1000.0])
    np.testing.assert_allclose(
        compute_mop_effective_f(x, Z_scaled, W, lags=lags), compute_mop_effective_f(x, Z, W, lags=lags), rtol=1e-8
    )


# ---------------------------------------------------------------------------
# 3. Multi-instrument Anderson-Rubin sets
# ---------------------------------------------------------------------------

def test_lp_iv_multiple_instruments(monkeypatch):
    """kz=2: MOP F, the reported critical value, and an AR set that is exact rather than a grid read-off."""
    cap = _spy_multi(monkeypatch, _IV_MOD)
    out = lp_iv(_strong_kz2(), y="y", x="x", z=["z1", "z2"], horizons=[0, 1, 2], n_lags=1, anderson_rubin=True)

    assert "mop_f" in out.columns
    assert {"ar_lo", "ar_hi", "ar_set_type"} <= set(out.columns)

    row = out.loc[out["h"] == 1].iloc[0]
    # Strong instruments: F_eff far above the 10%-bias critical value, which is the kz=2 closed form.
    assert row["mop_f"] > 100.0
    assert row["mop_cv_10"] == mop_critical_values(2)[0]
    assert row["mop_f"] > row["mop_cv_10"]
    # Bounded set containing the true slope.
    assert row["ar_set_type"] == "bounded"
    lo, hi = row["ar_lo"], row["ar_hi"]
    assert lo < -0.6 < hi
    assert hi - lo > 0.0

    # Endpoints solve AR(beta) = chi2_{2, 0.90} exactly (the row captured last is h=2, so rerun h=1).
    lp_iv(_strong_kz2(), y="y", x="x", z=["z1", "z2"], horizons=[1], n_lags=1, anderson_rubin=True)
    stat = _brute_stat_from_capture(cap)
    crit = chi2.ppf(1 - cap["alpha"], df=2)
    assert stat(lo) == pytest.approx(crit, rel=1e-8)
    assert stat(hi) == pytest.approx(crit, rel=1e-8)
    assert stat(0.5 * (lo + hi)) < crit
    assert stat(lo - 1e-3) > crit
    assert stat(hi + 1e-3) > crit

    # Brute-force fine-grid inversion agrees to within one grid step.
    grid = np.linspace(row["beta"] - 4 * row["se"], row["beta"] + 4 * row["se"], 8001)
    step = grid[1] - grid[0]
    accepted = np.array([stat(b) <= crit for b in grid])
    assert abs(grid[accepted].min() - lo) <= step
    assert abs(grid[accepted].max() - hi) <= step
    # And the grid step is much finer than the old fixed 0.05 resolution.
    assert step < 1e-3


def test_lp_iv_multi_ar_set_is_scale_equivariant():
    """y -> c*y scales the AR endpoints by c exactly and leaves F_eff and the set type unchanged."""
    base = lp_iv(_strong_kz2(1.0), y="y", x="x", z=["z1", "z2"], horizons=[1], n_lags=1, anderson_rubin=True).iloc[0]
    assert base["ar_set_type"] == "bounded"
    for c in (0.01, 0.1, 100.0):
        row = lp_iv(_strong_kz2(c), y="y", x="x", z=["z1", "z2"], horizons=[1], n_lags=1, anderson_rubin=True).iloc[0]
        assert row["ar_set_type"] == "bounded"
        np.testing.assert_allclose(row["ar_lo"], c * base["ar_lo"], rtol=1e-8)
        np.testing.assert_allclose(row["ar_hi"], c * base["ar_hi"], rtol=1e-8)
        assert row["ar_hi"] > row["ar_lo"]
        np.testing.assert_allclose(row["mop_f"], base["mop_f"], rtol=1e-10)


def test_lp_iv_multi_ar_set_independent_of_search_seed(monkeypatch):
    """The 2SLS (beta_hat, se_hat) only seed the search: far-off or non-finite seeds give the same set."""
    cap = _spy_multi(monkeypatch, _IV_MOD)
    out = lp_iv(_weak_kz2(), y="y", x="x", z=["z1", "z2"], horizons=[1], n_lags=1, weak_iv_robust=True).iloc[0]
    args = (cap["y_target"], cap["x"], cap["W_ctl"], cap["Z_mat"], cap["lags"], cap["alpha"])
    for beta_hat, se_hat in ((-12.0, 0.1), (-30.0, 0.1), (10.0, 0.1), (0.0, np.nan), (np.nan, np.nan)):
        lo, hi, kind = _compute_anderson_rubin_multi(*args, beta_hat=beta_hat, se_hat=se_hat)
        assert kind == out["ar_set_type"]
        np.testing.assert_allclose(lo, out["ar_lo"], rtol=1e-8)
        np.testing.assert_allclose(hi, out["ar_hi"], rtol=1e-8)


def test_multi_inversion_reduces_to_single_instrument_closed_form():
    """With one instrument the root-finding inversion reproduces the exact quadratic closed form."""
    designs = []
    # bounded
    rng = np.random.default_rng(101)
    T = 250
    z = rng.standard_normal(T)
    x = 0.8 * z + 0.4 * rng.standard_normal(T)
    designs.append((z, x, rng, T))
    # unbounded rays (seed 3) and all_real (seed 0), as in the stress suite
    for seed, pi in ((3, 0.05), (0, 0.0001)):
        rng = np.random.default_rng(seed)
        T = 150
        z = rng.standard_normal(T)
        x = pi * z + rng.standard_normal(T)
        designs.append((z, x, rng, T))

    kinds = []
    for z, x, rng, T in designs:
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = 0.4 * y[t - 1] + 1.0 * x[t - 1] + rng.standard_normal()
        dy = y[2:] - y[:-2]           # y_{t+1} - y_{t-1}
        x_t, z_t = x[1:-1], z[1:-1]
        x_l, y_l = x[:-2], y[:-2]
        n = len(dy)
        W_mat = np.column_stack([np.ones(n), z_t, x_l, y_l])
        W_ctl = np.column_stack([np.ones(n), x_l, y_l])
        lo1, hi1, kind1 = _compute_anderson_rubin_ci(dy, x_t, W_mat, lags=2, alpha=0.10)
        lo2, hi2, kind2 = _compute_anderson_rubin_multi(dy, x_t, W_ctl, z_t[:, None], lags=2, alpha=0.10)
        assert kind1 == kind2
        if kind1 == "all_real":
            assert np.isneginf(lo2) and np.isposinf(hi2)
        else:
            np.testing.assert_allclose(lo2, lo1, rtol=1e-8)
            np.testing.assert_allclose(hi2, hi1, rtol=1e-8)
        kinds.append(kind1)
    assert kinds == ["bounded", "unbounded_rays", "all_real"]


# ---------------------------------------------------------------------------
# 4. Weak instruments
# ---------------------------------------------------------------------------

def test_lp_iv_weak_instruments_multiple(monkeypatch):
    """Weak kz=2: small F_eff and an explicitly unbounded AR set whose ray endpoints solve AR = crit."""
    cap = _spy_multi(monkeypatch, _IV_MOD)
    out = lp_iv(_weak_kz2(), y="y", x="x", z=["z1", "z2"], horizons=[1], n_lags=1, weak_iv_robust=True)
    row = out.iloc[0]

    # Effective F below both bias thresholds.
    assert row["mop_f"] < row["mop_cv_20"] < row["mop_cv_10"]

    # The set is (-inf, ar_hi] U [ar_lo, inf): the accepted tails, a rejected gap in between.
    assert row["ar_set_type"] == "unbounded_rays"
    lo, hi = row["ar_lo"], row["ar_hi"]
    assert lo > hi
    stat = _brute_stat_from_capture(cap)
    crit = chi2.ppf(1 - cap["alpha"], df=2)
    assert stat(lo) == pytest.approx(crit, rel=1e-8)
    assert stat(hi) == pytest.approx(crit, rel=1e-8)
    assert stat(0.5 * (lo + hi)) > crit
    assert stat(lo + row["se"]) < crit
    assert stat(hi - row["se"]) < crit
    assert stat(lo + 100.0) < crit and stat(hi - 100.0) < crit
    # Independent fine-grid reference for this design: (-inf, -4.89] U [-1.06, inf).
    assert lo == pytest.approx(-1.06, abs=0.02)
    assert hi == pytest.approx(-4.89, abs=0.02)

    # Rescaling the outcome by 0.01 (where the old fixed grid collapsed) keeps the classification
    # and scales the endpoints.
    small = lp_iv(_weak_kz2(0.01), y="y", x="x", z=["z1", "z2"], horizons=[1], n_lags=1, weak_iv_robust=True).iloc[0]
    assert small["ar_set_type"] == "unbounded_rays"
    np.testing.assert_allclose(small["ar_lo"], 0.01 * lo, rtol=1e-8)
    np.testing.assert_allclose(small["ar_hi"], 0.01 * hi, rtol=1e-8)


# ---------------------------------------------------------------------------
# 5. Lag-augmented LP-IV
# ---------------------------------------------------------------------------

def test_la_lp_iv_single_instrument():
    """Lag-augmented LP-IV computes White-robust MOP F and AR sets."""
    rng = np.random.default_rng(404)
    T = 300
    z = rng.standard_normal(T)
    x = 0.9 * z + 0.4 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 1.2 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z": z})

    out = la_lp_iv(df, y="y", x="x", z="z", horizons=[0, 1, 2], n_lags=2, extra_lags=2, anderson_rubin=True)

    assert out.method == "la_lp_iv"
    assert "mop_f" in out.columns
    assert "mop_cv_10" in out.columns
    assert {"ar_lo", "ar_hi", "ar_set_type"} <= set(out.columns)

    # Same critical values as lp_iv (one implementation).
    assert out["mop_cv_10"].iloc[0] == mop_critical_values(1)[0]
    assert out["mop_cv_20"].iloc[0] == mop_critical_values(1)[1]
    # Strong instrument: MOP F is large
    assert out.loc[out["h"] == 1, "mop_f"].iloc[0] > out.loc[out["h"] == 1, "mop_cv_10"].iloc[0]
    # Bounded White AR interval contains true effect 1.2
    assert out.loc[out["h"] == 1, "ar_set_type"].iloc[0] == "bounded"
    lo = out.loc[out["h"] == 1, "ar_lo"].iloc[0]
    hi = out.loc[out["h"] == 1, "ar_hi"].iloc[0]
    assert lo < 1.2 < hi


def test_la_lp_iv_multiple_instruments(monkeypatch):
    """la_lp with kz=2: White-robust MOP F and an exact White AR set (zero-lag inversion)."""
    cap = _spy_multi(monkeypatch, _LA_MOD)
    rng = np.random.default_rng(505)
    T = 350
    z1 = rng.standard_normal(T)
    z2 = rng.standard_normal(T)
    x = 0.8 * z1 + 0.7 * z2 + 0.4 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.4 * y[t - 1] + 0.8 * x[t - 1] + rng.standard_normal()

    df = pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})

    out = la_lp(df, y="y", x="x", z=["z1", "z2"], horizons=[1], n_lags=2, extra_lags=2, weak_iv_robust=True)
    row = out.iloc[0]

    assert out.method == "la_lp_iv"
    assert row["mop_f"] > row["mop_cv_10"]
    assert row["mop_cv_10"] == mop_critical_values(2)[0]
    assert row["ar_set_type"] == "bounded"
    assert row["ar_lo"] < 0.8 < row["ar_hi"]

    # The White (lags=0) statistic equals the critical value at both endpoints.
    assert cap["lags"] == 0
    stat = _brute_stat_from_capture(cap, lags=0)
    crit = chi2.ppf(1 - cap["alpha"], df=2)
    assert stat(row["ar_lo"]) == pytest.approx(crit, rel=1e-8)
    assert stat(row["ar_hi"]) == pytest.approx(crit, rel=1e-8)
    assert stat(0.5 * (row["ar_lo"] + row["ar_hi"])) < crit

    # Scale equivariance through the public API.
    df_small = df.assign(y=df["y"] * 0.01)
    small = la_lp(df_small, y="y", x="x", z=["z1", "z2"], horizons=[1], n_lags=2, extra_lags=2, weak_iv_robust=True).iloc[0]
    assert small["ar_set_type"] == "bounded"
    np.testing.assert_allclose(small["ar_lo"], 0.01 * row["ar_lo"], rtol=1e-8)
    np.testing.assert_allclose(small["ar_hi"], 0.01 * row["ar_hi"], rtol=1e-8)


def test_la_lp_iv_reexported_from_puremacro_lp():
    """la_lp_iv is public at package level like every other lp_*_iv estimator."""
    from puremacro.lp import la_lp_iv as pkg_la_lp_iv

    assert pkg_la_lp_iv is _LA_MOD.la_lp_iv
    assert "la_lp_iv" in puremacro.lp.__all__
    assert "la_lp" in puremacro.lp.__all__


# ---------------------------------------------------------------------------
# 6. Backward compatibility with v3.3.0 call patterns
# ---------------------------------------------------------------------------

def test_v330_style_calls_still_work():
    rng = np.random.default_rng(13)
    T = 200
    z = rng.standard_normal(T)
    c = rng.standard_normal(T)
    x = 0.7 * z + 0.3 * rng.standard_normal(T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] - 0.4 * x[t - 1] + 0.2 * c[t] + rng.standard_normal()
    df = pd.DataFrame({"x": x, "y": y, "z": z, "c": c})

    # lp_iv: v3.3.0 positional signature (df, y, x, z, horizons, n_lags, controls, alpha).
    pos = lp_iv(df, "y", "x", "z", [0, 1, 2], 2, None, 0.10)
    assert {"h", "beta", "se", "t", "lo", "hi", "first_stage_f"} <= set(pos.columns)
    assert list(pos["h"]) == [0, 1, 2]
    kw = lp_iv(df, y="y", x="x", z="z", horizons=[0, 1, 2], n_lags=2, controls=["c"], alpha=0.05,
               anderson_rubin=True, lags=None, horizon=None, ci=None)
    assert {"ar_lo", "ar_hi", "ar_set_type"} <= set(kw.columns)
    assert kw.method == "LP-IV" and kw.ci_level == pytest.approx(0.95)
    # A one-element instrument list is the single-instrument path.
    one = lp_iv(df, y="y", x="x", z=["z"], horizons=[0, 1, 2], n_lags=2, anderson_rubin=True)
    both = lp_iv(df, y="y", x="x", z="z", horizons=[0, 1, 2], n_lags=2, anderson_rubin=True)
    for col in ("beta", "se", "first_stage_f", "mop_f", "ar_lo", "ar_hi"):
        np.testing.assert_allclose(one[col].values.astype(float), both[col].values.astype(float), rtol=1e-12)

    # la_lp: v3.3.0 positional signature (df, y, x, horizons, n_lags, extra_lags, controls, alpha),
    # no IV columns, method "la_lp".
    la = la_lp(df, "y", "x", [0, 1, 2], 2, 3, None, 0.10)
    assert list(la.columns) == ["h", "beta", "se", "lo", "hi", "p_aug"]
    assert la.method == "la_lp"
    assert (la["p_aug"] == 5).all()
    la_kw = la_lp(df, y="y", x="x", horizon=2, lags=1, ci=0.90, controls=["c"])
    assert la_kw.method == "la_lp" and "mop_f" not in la_kw.columns
