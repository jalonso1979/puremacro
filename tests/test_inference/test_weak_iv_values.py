"""Value tests for the weak-instrument diagnostics.

These functions had exactly one test between them before this file
(`test_kleibergen_paap_collinear_instruments_diagnostic`, which asserts that a
rank-deficient design *raises*) and none that checked a returned number. Both
defects below are things only a number check can see: the statistic was
internally consistent, monotone in instrument strength, and positive.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from puremacro.inference.weak_iv import anderson_rubin_band, kleibergen_paap_f


def _robust_first_stage_f(X, Z):
    """HC0-robust Wald F for H0: Pi = 0 in X = Z Pi + v, with one endogenous X.

    At k = 1 the Kleibergen-Paap rk Wald F reduces to exactly this, which is
    what makes it a usable known truth rather than a second opinion.
    """
    n, l = Z.shape
    x = np.asarray(X).ravel()
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    pi = ZtZ_inv @ (Z.T @ x)
    v = x - Z @ pi
    meat = (Z * v[:, None]).T @ (Z * v[:, None])
    V = ZtZ_inv @ meat @ ZtZ_inv
    return float(pi @ np.linalg.inv(V) @ pi) / l


@pytest.mark.parametrize("n", [200, 400, 800])
def test_kleibergen_paap_f_matches_the_robust_first_stage_f(n):
    """The rk Wald F must equal the robust first-stage F when k = 1.

    It used to be exactly n**2 too large: the sandwich divided `meat` by n and
    then divided the whole sandwich by n again, while the bread inverted the
    RAW cross-product Z'Z and so already carried both factors. Measured
    against this reference the shipped value was 235,470 vs 5.89 at n = 200
    and 8,755,485 vs 13.68 at n = 800 — ratios of exactly 40,000, 160,000 and
    640,000.

    The direction is what made it dangerous. Stock-Yogo thresholds sit near
    10, so a six-figure F reads as "instruments overwhelmingly strong" on
    every dataset, and this diagnostic could never fire on the weak
    instruments it exists to detect.
    """
    rng = np.random.default_rng(n)
    l = 3
    Z = rng.standard_normal((n, l))
    v = rng.standard_normal(n) * (0.5 + np.abs(Z[:, 0]))  # heteroskedastic
    X = (Z @ np.array([0.3, -0.2, 0.15]) + v)[:, None]
    y = X[:, 0] + rng.standard_normal(n)

    got = kleibergen_paap_f(y, X, Z)
    expected = _robust_first_stage_f(X, Z)
    assert got == pytest.approx(expected, rel=1e-8), (
        f"rk Wald F {got:.4f} != robust first-stage F {expected:.4f} "
        f"(ratio {got / expected:.1f}, n^2 = {n ** 2})"
    )


def test_kleibergen_paap_f_is_in_a_usable_range_for_weak_instruments():
    """A weak design must produce a small F, not a huge one.

    The scaling bug made every F astronomically large, so no design could ever
    be diagnosed as weak. This pins the property the function exists for.
    """
    rng = np.random.default_rng(3)
    n, l = 500, 2
    Z = rng.standard_normal((n, l))
    X = (Z @ np.array([0.01, 0.01]) + rng.standard_normal(n))[:, None]  # near-zero first stage
    y = X[:, 0] + rng.standard_normal(n)
    f = kleibergen_paap_f(y, X, Z)
    assert f < 10.0, f"a near-zero first stage returned F = {f:.2f}"


def test_anderson_rubin_band_covers_at_its_nominal_rate_for_df_above_one():
    """`f_stat_fn` returns an F, and it is df*F — not F — that is chi2(df).

    The cutoff `chi2.ppf(ci, df)` was therefore df times too large, so the
    band was too wide by that factor for any over-identified design. Measured
    coverage of the nominal 90% band at n = 300 was 100.0% at df = 2 and
    100.0% at df = 4. df = 1 is unaffected, which is the default and the only
    case anything exercised.
    """
    def ar_factory(y, X, Z):
        n, l = Z.shape
        Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)

        def f(b):
            e = y - X * b
            num = (e @ Pz @ e) / l
            den = (e @ e - e @ Pz @ e) / (n - l)
            return float(num / den)
        return f

    grid = np.linspace(-2.0, 4.0, 241)
    for l in (2, 4):
        rng = np.random.default_rng(7)
        covered = 0
        reps = 150
        for _ in range(reps):
            n = 300
            Z = rng.standard_normal((n, l))
            v = rng.standard_normal(n)
            u = 0.6 * v + rng.standard_normal(n)
            X = Z @ np.full(l, 0.5) + v
            y = 1.0 * X + u
            lo, hi = anderson_rubin_band(grid, ar_factory(y, X, Z), ci=0.90, df=l)
            covered += (lo <= 1.0 <= hi)
        rate = covered / reps
        assert rate < 0.99, (
            f"df={l}: nominal 90% AR band covered {rate:.1%} — a band that "
            f"never misses is not a 90% band"
        )
        assert rate > 0.80, f"df={l}: coverage {rate:.1%} is far below nominal"


def test_anderson_rubin_cutoff_is_the_chi2_quantile_divided_by_df():
    """Probe the cutoff directly by choosing f_stat_fn = |beta|.

    Then a beta is accepted exactly when |beta| <= crit, so the band endpoints
    read the cutoff straight off the grid. Written this way because the
    obvious version of this test -- asserting the formula against itself --
    cannot fail.
    """
    grid = np.linspace(0.0, 10.0, 100001)   # spacing 1e-4
    for df in (1, 2, 4, 8):
        lo, hi = anderson_rubin_band(grid, lambda b: abs(b), ci=0.90, df=df)
        expected = chi2.ppf(0.90, df) / df
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert hi == pytest.approx(expected, abs=2e-4), (
            f"df={df}: acceptance region ends at {hi:.5f}, implying a cutoff "
            f"of {hi:.5f}; expected chi2.ppf(0.9, {df})/{df} = {expected:.5f} "
            f"(the unscaled quantile is {chi2.ppf(0.90, df):.5f})"
        )
