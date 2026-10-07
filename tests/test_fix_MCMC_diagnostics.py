"""Calibration and parity tests for the MCMC diagnostics fix (key MCMC).

Defect (reviews/2026-09-30-notebook-review, key MCMC):

* ``geweke_z`` estimated the spectral density at zero with a Bartlett
  window of ``int(4 (m/100)**(2/9))`` lags (5-7 lags for 500-2500 draws),
  so on stationary but persistent chains it rejected far above the nominal
  5% (33% at phi = 0.9, 76% at phi = 0.99 for n = 5000).
* ``effective_sample_size`` paired autocorrelations from lag 1, had no
  monotone step and was capped at ``n`` (an antithetic AR(1) with
  phi = -0.5 always returned ``n`` instead of ``3n``).
* ``gelman_rubin`` had no split option.

Repair round 1 (adversarial review of the first fix):

* ``effective_sample_size(x, max_lag=2)`` returned the cap ``n log10 n``
  for any chain; ``max_lag < 3`` now has explicit, documented formulas.
* ``geweke_z(np.full(n, 0.3))`` returned an arbitrary finite value
  because the constant-series test compared rounding noise with itself;
  constant chains now give ``nan`` whatever the constant, and ``n < 4``
  gives ``nan`` instead of using overlapping segments.

Oracles used here:

* Closed forms for a stationary AR(1) with coefficient phi:
  long-run variance ``S(0) = (1 + phi) / (1 - phi)`` (unit marginal
  variance) and ``ESS = n (1 - phi) / (1 + phi)``; the Geweke z-score has
  size 5% at |z| > 1.96.
* R 4.6.0 with coda 0.19-4.1: ``spectrum0.ar`` on fixed series (golden
  values computed in the fix session, 2026-09-30, by
  ``Rscript r_geweke.R``; the series are regenerated below from the same
  numpy seeds).
* arviz 0.21.0 ``arviz.stats.diagnostics._ess``, ``_rhat_identity`` and
  ``_rhat_split`` on fixed series (golden values computed in the same
  session).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from puremacro.mcmc import (
    _spectrum0_ar,
    effective_sample_size,
    gelman_rubin,
    geweke_z,
    trace_summary,
)


def _ar1(seed: int, phi: float, n: int) -> np.ndarray:
    """Stationary AR(1), unit marginal variance (same recipe as the goldens)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) * np.sqrt(1.0 - phi ** 2)
    e[0] = rng.standard_normal()
    return lfilter([1.0], [1.0, -phi], e)


def _ar1_panel(rng: np.random.Generator, phi: float, n: int, reps: int) -> np.ndarray:
    e = rng.standard_normal((reps, n)) * np.sqrt(1.0 - phi ** 2)
    e[:, 0] = rng.standard_normal(reps)
    return lfilter([1.0], [1.0, -phi], e, axis=1)


# ---------------------------------------------------------------------------
# Parity with R coda::spectrum0.ar
# ---------------------------------------------------------------------------

# seed, phi, n -> (S_A(0), order_A, S_B(0), order_B, z) on segments
# x[1:nA], x[(n-nB+1):n] with nA = floor(0.1 n), nB = floor(0.5 n).
CODA_GOLDEN = {
    (11, 0.5, 5000): (2.44633123790029, 1, 3.38029015501362, 1, 0.565374429656862),
    (12, 0.9, 5000): (7.89539354002655, 2, 21.7564021464311, 1, 0.295823389392506),
    (13, 0.95, 5000): (19.9611678701911, 1, 40.3013435331866, 1, -0.879190529979667),
    (14, 0.99, 5000): (275.952043892534, 1, 175.826039633422, 1, -0.33510672494159),
    (15, 0.0, 1000): (1.04692258763239, 0, 0.871189981552079, 0, 0.996470026464902),
    (16, 0.9, 400): (2.01950842497744, 1, 30.7549922623453, 1, 1.28646502462446),
}


@pytest.mark.parametrize("case", sorted(CODA_GOLDEN))
def test_spectrum0_ar_and_geweke_match_coda(case):
    seed, phi, n = case
    s_a, o_a, s_b, o_b, z = CODA_GOLDEN[case]
    x = _ar1(seed, phi, n)
    n_a, n_b = int(0.1 * n), int(0.5 * n)
    out_a = _spectrum0_ar(x[:n_a])
    out_b = _spectrum0_ar(x[-n_b:])
    assert out_a["order"] == o_a and out_b["order"] == o_b
    assert out_a["spec"] == pytest.approx(s_a, rel=1e-9)
    assert out_b["spec"] == pytest.approx(s_b, rel=1e-9)
    assert geweke_z(x) == pytest.approx(z, rel=1e-9)


# Near-unit-root series, where 1 - sum(a) is small and the relative error
# against R is largest.  Goldens from R 4.6.0 / coda 0.19-4.1 spectrum0.ar
# (repair round, 2026-09-30).
CODA_NEAR_UNIT_ROOT = {
    "random_walk_504": (lambda: np.cumsum(np.random.default_rng(504).standard_normal(2500)),
                        214350.22875153643, 1),
    "ar1_0.999_512": (lambda: _ar1(512, 0.999, 2500), 934.21356849752851, 1),
}


@pytest.mark.parametrize("name", sorted(CODA_NEAR_UNIT_ROOT))
def test_spectrum0_ar_matches_coda_near_unit_root(name):
    make, spec, order = CODA_NEAR_UNIT_ROOT[name]
    out = _spectrum0_ar(make())
    assert out["order"] == order
    assert out["spec"] == pytest.approx(spec, rel=1e-10)


def test_spectrum0_ar_long_run_variance_of_ar1():
    """S(0) of a long AR(1) is close to the closed form (1+phi)/(1-phi)."""
    for seed, phi in [(41, 0.5), (42, 0.9), (43, -0.5)]:
        x = _ar1(seed, phi, 200_000)
        s0 = _spectrum0_ar(x)["spec"]
        assert s0 == pytest.approx((1 + phi) / (1 - phi), rel=0.05)


def test_spectrum0_ar_degenerate_series():
    assert _spectrum0_ar(np.full(50, 3.0))["spec"] == 0.0
    assert _spectrum0_ar(np.arange(50.0))["spec"] == 0.0   # exact trend, as coda
    small = 1e-12 * _ar1(44, 0.5, 400)                      # tiny units are not "constant"
    big = _ar1(44, 0.5, 400)
    assert _spectrum0_ar(small)["spec"] == pytest.approx(
        1e-24 * _spectrum0_ar(big)["spec"], rel=1e-8
    )
    assert np.isnan(_spectrum0_ar(np.array([1.0, np.nan, 2.0]))["spec"])
    # Constants whose computed mean is not exact (x - mean is ~1e-17, not 0)
    for c in (0.3, 0.1, 1.0 / 3.0, 7e5 / 3.0, 1e-9 / 3.0, 0.0):
        assert _spectrum0_ar(np.full(5000, c))["spec"] == 0.0


# ---------------------------------------------------------------------------
# Geweke size on persistent stationary chains (Monte Carlo, closed-form null)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phi,n,reps,seed,upper",
    [(0.5, 5000, 1000, 101, 0.08), (0.9, 5000, 1000, 102, 0.10),
     (0.95, 5000, 1000, 103, 0.12), (0.99, 50_000, 400, 104, 0.12)],
)
def test_geweke_size_close_to_nominal(phi, n, reps, seed, upper):
    """Stationary AR(1): P(|z| > 1.96) must be near 0.05.

    Pre-fix rejection rates (seeded Monte Carlo, 1000 reps): 0.075 (0.5),
    0.330 (0.9), 0.485 (0.95), 0.648 (0.99, n = 50,000).  Post-fix, from
    4000 reps (2000 at phi = 0.99): 0.057, 0.072, 0.088 and 0.074 -- the
    finite-sample size of coda's geweke.diag, whose Yule-Walker fit is
    biased towards zero on a 500-draw first segment of a persistent chain.
    """
    rng = np.random.default_rng(seed)
    X = _ar1_panel(rng, phi, n, reps)
    z = np.array([geweke_z(x) for x in X])
    reject = float(np.mean(np.abs(z) > 1.96))
    assert 0.02 <= reject <= upper, f"phi={phi}: rejection rate {reject:.3f}"
    assert 0.85 <= float(np.std(z)) <= 1.25


def test_geweke_legacy_bartlett_still_available_and_oversized():
    """spectrum='bartlett' reproduces the pre-fix estimator (and its defect)."""
    rng = np.random.default_rng(105)
    X = _ar1_panel(rng, 0.9, 5000, 300)
    z_old = np.array([geweke_z(x, spectrum="bartlett") for x in X])
    z_new = np.array([geweke_z(x) for x in X])
    assert np.mean(np.abs(z_old) > 1.96) > 0.2
    assert np.mean(np.abs(z_new) > 1.96) < 0.1


def test_geweke_detects_an_unconverged_start():
    """First 10% of a persistent chain shifted by 2 marginal s.d.: flagged.

    Closed form: z ~= 2 / sqrt(19/500 + 19/2500) = 9.4 for phi = 0.9.
    """
    x = _ar1(106, 0.9, 5000)
    x[:500] += 2.0
    assert geweke_z(x) > 5.0


def test_geweke_edge_cases():
    assert np.isnan(geweke_z(np.ones(100)))
    step = np.zeros(1000)
    step[500:] = 10.0
    assert geweke_z(step) == -np.inf
    with pytest.raises(ValueError, match="overlap"):
        geweke_z(np.arange(100.0), first=0.6, last=0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        geweke_z(np.arange(100.0), first=0.0)
    with pytest.raises(ValueError, match="spectrum"):
        geweke_z(np.arange(100.0), spectrum="parzen")
    x = _ar1(107, 0.5, 300)
    x[5] = np.nan
    assert np.isnan(geweke_z(x))
    assert np.isfinite(geweke_z(_ar1(108, 0.5, 20)))   # 2-draw first segment


def test_geweke_constant_chain_is_nan_whatever_the_constant():
    """Round 1 left an arbitrary finite z for most constants (0.3 -> -0.71).

    coda gives NaN for np.full(5000, 0.3) and np.full(5000, 1/3) (R 4.6.0,
    coda 0.19-4.1, geweke.diag).
    """
    for c in (0.3, 0.1, 1.0 / 3.0, 2.5, 7e5 / 3.0, 1e-9 / 3.0, 0.0):
        for n in (1000, 5000):
            assert np.isnan(geweke_z(np.full(n, c))), (c, n)
    rng = np.random.default_rng(9)
    assert all(np.isnan(geweke_z(np.full(5000, v))) for v in rng.uniform(0.01, 5, 200))
    # Two different constants are still +-inf; a stuck first segment
    # contributes no variance (S_A = 0 exactly, as in coda).
    two = np.full(5000, 0.3)
    two[2500:] = 0.7
    assert geweke_z(two) == -np.inf
    x = _ar1(109, 0.5, 5000)
    x[:500] = 0.3
    s_b = _spectrum0_ar(x[-2500:])["spec"]
    assert _spectrum0_ar(x[:500])["spec"] == 0.0
    assert geweke_z(x) == pytest.approx((0.3 - x[-2500:].mean()) / np.sqrt(s_b / 2500))


def test_geweke_short_chains():
    """n < 4: no two disjoint 2-draw segments -> nan; n = 4, 5: both
    segments have 2 draws (exactly linear, S = 0) -> +-inf; n >= 6 finite."""
    for n in range(0, 4):
        assert np.isnan(geweke_z(np.random.default_rng(n).standard_normal(n)))
    for n in (4, 5):
        assert np.isinf(geweke_z(np.random.default_rng(n).standard_normal(n)))
    for n in range(6, 40):
        assert np.isfinite(geweke_z(np.random.default_rng(n).standard_normal(n)))


# ---------------------------------------------------------------------------
# Effective sample size: closed form n(1-phi)/(1+phi) and arviz parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phi,tol", [(-0.5, 0.05), (0.0, 0.05), (0.5, 0.05), (0.9, 0.06), (0.95, 0.08)]
)
def test_ess_matches_ar1_closed_form_on_average(phi, tol):
    n, reps = 5000, 200
    rng = np.random.default_rng(int(1000 * (phi + 1)))
    X = _ar1_panel(rng, phi, n, reps)
    ess = np.array([effective_sample_size(x) for x in X])
    true = n * (1 - phi) / (1 + phi)
    assert float(np.mean(ess)) / true == pytest.approx(1.0, abs=tol)


def test_ess_exceeds_n_for_antithetic_chain():
    """Pre-fix: ESS was capped at n (returned exactly n for phi = -0.5)."""
    x = _ar1(201, -0.5, 4000)
    ess = effective_sample_size(x)
    assert ess > 2.0 * 4000
    assert ess <= 4000 * np.log10(4000)


# seed, phi, n -> arviz 0.21.0 _ess(x) for a single chain.  Seeds 21, 22
# and 23 exercise the monotone step (their initial positive sequence is not
# monotone).
ARVIZ_ESS_GOLDEN = {
    (21, 0.9, 2000): 59.733192192587765,
    (22, -0.5, 1000): 2665.1349035892213,
    (23, 0.99, 5000): 28.595143770154923,
    (24, 0.0, 777): 788.1900930979303,
    (25, 0.95, 3000): 90.67252699296772,
}


@pytest.mark.parametrize("case", sorted(ARVIZ_ESS_GOLDEN))
def test_ess_matches_arviz(case):
    seed, phi, n = case
    assert effective_sample_size(_ar1(seed, phi, n)) == pytest.approx(
        ARVIZ_ESS_GOLDEN[case], rel=1e-10
    )


def test_ess_edge_cases():
    assert effective_sample_size(np.full(100, 2.5)) == 100.0
    assert effective_sample_size(np.array([1.0, 2.0, 3.0])) == 3.0
    assert np.isnan(effective_sample_size(np.array([1.0, np.inf, 0.0, 2.0, 1.0])))
    x = _ar1(202, 0.5, 1000)
    assert effective_sample_size(x, max_lag=1) == pytest.approx(
        1000 / (1 + 2 * (np.corrcoef(x[1:], x[:-1])[0, 1])), rel=0.02
    )
    assert 0 < effective_sample_size(x, max_lag=10) < 1000
    # scale invariance (no absolute floors)
    assert effective_sample_size(1e-9 * x) == pytest.approx(
        effective_sample_size(x), rel=1e-9
    )


def test_ess_small_max_lag_never_returns_the_cap():
    """Round 1: max_lag=2 skipped the pair loop and returned n log10 n
    (6602 for n = 2000) for any chain -- a false pass on a phi = 0.9 chain
    whose true ESS is 105."""
    n = 2000
    x = _ar1(21, 0.9, n)
    true = n * 0.1 / 1.9
    cap = n * np.log10(n)
    ess = {L: effective_sample_size(x, max_lag=L) for L in (0, 1, 2, 3, 5, 10, None)}
    assert ess[0] == n                                 # no correction
    assert effective_sample_size(x, max_lag=-3) == n
    rho = _autocov_rho(x)
    assert ess[1] == pytest.approx(n / (1.0 + 2.0 * rho[1]), rel=1e-12)
    assert ess[2] == pytest.approx(n / (1.0 + 2.0 * rho[1] + max(rho[2], 0.0)), rel=1e-12)
    assert ess[2] == pytest.approx(ess[3], rel=1e-12)  # same sum, lags 0..2
    for L in (1, 2, 3, 5, 10):
        assert ess[L] < n < cap
    # Longer windows see more positive autocorrelation.
    assert ess[0] >= ess[1] >= ess[2] >= ess[5] >= ess[10] >= ess[None]
    assert ess[None] < 2 * true
    # iid: every max_lag gives about n, never the cap.
    z = np.random.default_rng(5).standard_normal(n)
    for L in (0, 1, 2, 3, 5):
        assert abs(effective_sample_size(z, max_lag=L) / n - 1.0) < 0.15


def _autocov_rho(x: np.ndarray) -> np.ndarray:
    """rho_t = gamma_t / gamma_0 - 1/(n-1), gamma with divisor n (arviz)."""
    n = len(x)
    xc = x - x.mean()
    g = np.array([xc[: n - t] @ xc[t:] / n for t in range(3)])
    return g / g[0] - 1.0 / (n - 1)


# ---------------------------------------------------------------------------
# Split R-hat
# ---------------------------------------------------------------------------

def _rhat_chains(drift: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 1001)
    base = np.vstack([_ar1(31 + m, 0.5, 1001) for m in range(4)])
    return base + drift * t


@pytest.mark.parametrize(
    "drift,classic,split",
    [(0.0, 1.00071388940022, 1.0040617521365562),
     (1.5, 1.000563353965639, 1.052910297708167),
     (3.0, 1.0002359793240387, 1.204171441560839)],
)
def test_rhat_classic_and_split_match_arviz(drift, classic, split):
    chains = _rhat_chains(drift)
    assert gelman_rubin(chains)["R_hat"] == pytest.approx(classic, rel=1e-12)
    assert gelman_rubin(chains, split=False)["R_hat"] == pytest.approx(classic, rel=1e-12)
    assert gelman_rubin(chains, split=True)["R_hat"] == pytest.approx(split, rel=1e-12)


def test_split_rhat_sees_common_drift_classic_does_not():
    chains = _rhat_chains(3.0)
    assert gelman_rubin(chains)["R_hat"] < 1.01
    assert gelman_rubin(chains, split=True)["R_hat"] > 1.1


def test_split_rhat_single_chain_and_errors():
    x = _ar1(301, 0.5, 2000)[None, :]
    assert gelman_rubin(x, split=True)["R_hat"] == pytest.approx(1.0, abs=0.02)
    with pytest.raises(ValueError, match="at least 4"):
        gelman_rubin(np.ones((2, 3)), split=True)


def test_trace_summary_reports_split_rhat():
    chains = _rhat_chains(3.0)
    out = trace_summary(chains)
    assert out["R_hat"] == pytest.approx(gelman_rubin(chains)["R_hat"])
    assert out["R_hat_split"] == pytest.approx(
        gelman_rubin(chains, split=True)["R_hat"]
    )
    single = trace_summary(_ar1(302, 0.5, 400))
    assert "R_hat" not in single and "R_hat_split" in single
