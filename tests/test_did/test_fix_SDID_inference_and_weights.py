"""Regression tests for the SDID inference and weight-solver defect (key SDID).

Pre-fix behaviour of ``puremacro.did.synthetic_did`` (4.3.0), each pinned
by a test below:

* The only interval was a *donor-only* bootstrap that held the treated
  units fixed. With one treated unit a nominal 90 % interval covered the
  truth 50-67 % of the time. Arkhangelsky et al. (2021, arXiv 1812.09970v4)
  resample every unit (Algorithm 2), call the bootstrap and the jackknife
  "not well-defined" for N_tr = 1 (p. 30, Table 4 note) and give the placebo
  estimator (Algorithm 4) for that case. The CI is Gaussian, eq. (5.1).
* ``_solve_simplex_quadratic`` returned the uniform starting weights,
  silently, whenever SLSQP reported non-success, even when its iterate was
  already optimal. SLSQP's absolute ``ftol`` made that depend on the units
  of ``y``: multiplying the notebook-29 panel by 100 turned SDID into plain
  DiD (tau/c = -3.917 instead of -4.094), and on the paper's own California
  Prop 99 data the estimate was the DiD number, -27.35, instead of SDID -15.6.
* The noise level sigma-hat was period-demeaned, not the paper's eq. (2.2)
  (demeaned by the overall mean of the first differences, as in the synthdid
  R package), which moved the Prop 99 estimate to -15.19.
"""
from __future__ import annotations

import sys
import urllib.request
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from puremacro.did import sdid_multi_cohort, synthetic_did

# The package re-exports the function under the module's name, so reach
# the module object through sys.modules.
import puremacro.did.synthetic_did  # noqa: F401  (registers the module)

SDID_MOD = sys.modules["puremacro.did.synthetic_did"]


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _nb29_panel(scale: float = 1.0, seed: int = 2021) -> pd.DataFrame:
    """The DGP of notebooks/29_synthetic_did.py: 15 donors, one treated unit,
    T = 20, treatment at t = 12, true effect -4.5, noise sd 0.4."""
    rng = np.random.default_rng(seed)
    n_units, T, g = 16, 20, 12
    trend = np.linspace(10, 25, T)
    macro = np.sin(np.linspace(0, 3 * np.pi, T)) * 3.0
    fe = rng.uniform(5, 20, size=n_units)
    load = rng.uniform(0.5, 2.0, size=n_units)
    load[0] = 1.4
    rows = []
    for i in range(n_units):
        for t in range(T):
            y = fe[i] + trend[t] + load[i] * macro[t] + rng.normal(0, 0.4)
            if i == 0 and t >= g:
                y += -4.5
            rows.append({"unit": f"unit_{i:02d}", "time": t, "y": y * scale,
                         "treat_time": float(g) if i == 0 else np.nan})
    return pd.DataFrame(rows)


def _twfe_panel(seed: int, *, n_co: int = 30, n_tr: int = 1, T: int = 20,
                T0: int = 15, tau: float = 0.0) -> pd.DataFrame:
    """Two-way fixed effects plus iid N(0, 1) noise, common adoption at T0."""
    rng = np.random.default_rng(seed)
    n = n_co + n_tr
    a = rng.normal(size=n)
    b = rng.normal(size=T)
    y = a[:, None] + b[None, :] + rng.normal(size=(n, T))
    y[n_co:, T0:] += tau
    rows = [{"unit": i, "time": t, "y": y[i, t],
             "treat_time": float(T0) if i >= n_co else np.nan}
            for i in range(n) for t in range(T)]
    return pd.DataFrame(rows)


def _docs_panel() -> pd.DataFrame:
    """The docs/did.md SDID example: 8 reform states of 40, adoption at q12."""
    rng = np.random.default_rng(1)
    rows = []
    for s in range(40):
        reform = s < 8
        a = rng.normal(scale=2.0)
        for q in range(24):
            eff = 0.8 if (reform and q >= 12) else 0.0
            rows.append({"unit": f"S{s:02d}", "time": q,
                         "treat_time": 12.0 if reform else np.nan,
                         "y": a + 0.05 * q + eff + rng.normal(scale=0.3)})
    return pd.DataFrame(rows)


def _no_warnings(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Weights: scale equivariance and no silent uniform fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [1e-3, 10.0, 100.0, 1e4, 1e6])
def test_tau_is_scale_equivariant(c):
    """Old code: tau/c = -4.0801 at c = 10 (omega uniform) and -3.9170 at
    c >= 100 (omega and lambda uniform, i.e. plain DiD); -4.0940 at c = 1."""
    base = _no_warnings(synthetic_did, _nb29_panel(), n_boot=0)
    res = _no_warnings(synthetic_did, _nb29_panel(scale=c), n_boot=0)
    assert abs(res.tau / c - base.tau) < 1e-6 * abs(base.tau), (c, res.tau / c, base.tau)
    np.testing.assert_allclose(res.omega.values, base.omega.values, atol=1e-6)
    np.testing.assert_allclose(res.lambda_w.values, base.lambda_w.values, atol=1e-6)
    assert not np.allclose(res.omega.values, 1.0 / len(res.omega))


def test_scale_equivariance_of_placebo_se():
    """The placebo draws reuse the full-sample regularisation, so se(c y) = c se(y)."""
    base = _no_warnings(synthetic_did, _nb29_panel(), n_boot=30, seed=3)
    res = _no_warnings(synthetic_did, _nb29_panel(scale=1e3), n_boot=30, seed=3)
    np.testing.assert_allclose(res.se / 1e3, base.se, rtol=1e-6)
    np.testing.assert_allclose(res.lo / 1e3, base.lo, rtol=1e-6)


def _omega_problem(scale: float = 1.0):
    df = _nb29_panel(scale=scale)
    Y = df.pivot(index="unit", columns="time", values="y").to_numpy()
    Yd_pre, Yd_post, ytr = Y[1:, :12], Y[1:, 12:], Y[0, :12]
    zo, _ = SDID_MOD._sdid_zeta(Yd_pre, n_treated=1, T_post=Yd_post.shape[1])
    return Yd_pre.T, ytr, zo ** 2 * 12


def test_non_success_at_an_optimal_iterate_keeps_the_iterate(monkeypatch):
    """Old code: ``if not res.success: return w0`` threw away an optimal
    iterate (objective 353.8) and returned uniform weights (845.3)."""
    A, b, ridge = _omega_problem()
    w_ok = SDID_MOD._solve_simplex_quadratic(A, b, ridge)
    real_minimize = SDID_MOD.minimize

    def flaky(*args, **kwargs):
        res = real_minimize(*args, **kwargs)
        res.success = False
        res.status = 8
        res.message = "Positive directional derivative for linesearch"
        return res

    monkeypatch.setattr(SDID_MOD, "minimize", flaky)
    w = _no_warnings(SDID_MOD._solve_simplex_quadratic, A, b, ridge)
    np.testing.assert_allclose(w, w_ok, atol=1e-8)
    assert not np.allclose(w, 1.0 / len(w))


def test_failed_slsqp_is_polished_to_the_optimum(monkeypatch):
    """SLSQP failing at the starting point: the projected-gradient polish
    reaches the optimum instead of returning the start."""
    A, b, ridge = _omega_problem()
    w_ok = SDID_MOD._solve_simplex_quadratic(A, b, ridge)
    real_minimize = SDID_MOD.minimize

    def broken(fun, x0, *args, **kwargs):
        res = real_minimize(fun, x0, *args, **kwargs)
        res.x = np.asarray(x0, dtype=float).copy()
        res.success = False
        res.status = 4
        res.message = "Inequality constraints incompatible"
        return res

    monkeypatch.setattr(SDID_MOD, "minimize", broken)
    w = _no_warnings(SDID_MOD._solve_simplex_quadratic, A, b, ridge)
    np.testing.assert_allclose(w, w_ok, atol=1e-4)


def test_unconverged_solver_warns_instead_of_returning_uniform(monkeypatch):
    """When neither SLSQP nor the polish reaches the optimum the solver warns
    and returns its best feasible point, never the uniform start silently."""
    A, b, ridge = _omega_problem()
    real_minimize = SDID_MOD.minimize

    def broken(fun, x0, *args, **kwargs):
        res = real_minimize(fun, x0, *args, **kwargs)
        res.x = np.asarray(x0, dtype=float).copy()
        res.success = False
        res.status = 4
        res.message = "Inequality constraints incompatible"
        return res

    monkeypatch.setattr(SDID_MOD, "minimize", broken)
    monkeypatch.setattr(SDID_MOD, "_refine_on_support", lambda *a, **k: None)
    monkeypatch.setattr(SDID_MOD, "_POLISH_MAX_ITER", 3)
    with pytest.warns(UserWarning, match="did not reach the optimum"):
        w = SDID_MOD._solve_simplex_quadratic(A, b, ridge, what="unit weights")
    assert np.isclose(w.sum(), 1.0) and (w >= 0).all()
    assert not np.allclose(w, 1.0 / len(w))


# ---------------------------------------------------------------------------
# Noise level: the paper's eq. (2.2)
# ---------------------------------------------------------------------------


def test_noise_level_is_the_papers_eq_2_2():
    """Eq. (2.2): the first differences Delta_it of the control pre-period
    outcomes are demeaned by their *overall* mean Delta-bar. The divisor is
    n - 1 (synthdid's ``sd()``, which produced Table 1) where the paper
    writes n = N_co (T_pre - 1). 4.3.0 demeaned period by period instead."""
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(7, 9)) + np.linspace(0, 5, 9) ** 2
    d = np.diff(Y, axis=1)
    expected = np.sqrt(((d - d.mean()) ** 2).sum() / (d.size - 1))
    assert np.isclose(SDID_MOD._noise_level(Y), expected, rtol=1e-12)
    period = np.std(d - d.mean(axis=0), ddof=1)
    assert not np.isclose(SDID_MOD._noise_level(Y), period)


def test_noise_level_override_and_invariance_to_linear_time_trends():
    df = _nb29_panel()
    base = synthetic_did(df, n_boot=0)
    # A common linear trend shifts every first difference by the same constant,
    # so eq. (2.2) and tau are unchanged.
    lin = synthetic_did(df.assign(y=df.y + 0.7 * df.time), n_boot=0)
    assert abs(lin.tau - base.tau) < 1e-7
    # With sigma-hat supplied, any common time shift leaves tau unchanged.
    sig = SDID_MOD._noise_level(
        df.pivot(index="unit", columns="time", values="y").to_numpy()[1:, :12])
    a = synthetic_did(df, n_boot=0, noise_level=sig)
    b = synthetic_did(df.assign(y=df.y + 2.0 * (df.time % 3)), n_boot=0, noise_level=sig)
    assert abs(a.tau - b.tau) < 1e-7
    assert abs(a.tau - base.tau) < 1e-12
    with pytest.raises(ValueError, match="noise_level"):
        synthetic_did(df, n_boot=0, noise_level=-1.0)


# ---------------------------------------------------------------------------
# Inference: placebo (Alg. 4), bootstrap (Alg. 2), jackknife (Alg. 3)
# ---------------------------------------------------------------------------


def test_default_with_one_treated_unit_is_the_placebo_estimator():
    """Old code: donor bootstrap, se 0.2528 and a percentile CI
    [-4.6432, -3.8887] on the notebook-29 panel (n_boot=100, seed=42); the
    Monte Carlo sd of tau-hat on that DGP is about 0.24 while the donor
    bootstrap averaged 0.13."""
    df = _nb29_panel()
    res = _no_warnings(synthetic_did, df, n_boot=100, seed=42)
    plc = _no_warnings(synthetic_did, df, n_boot=100, seed=42, se_method="placebo")
    assert res.se == plc.se and res.lo == plc.lo and res.hi == plc.hi
    # Gaussian interval, eq. (5.1)
    z = norm.ppf(0.95)
    np.testing.assert_allclose([res.lo, res.hi],
                               [res.tau - z * res.se, res.tau + z * res.se], rtol=1e-12)
    assert getattr(res, "se_method", "placebo") == "placebo"


def test_placebo_se_matches_the_exhaustive_placebo_distribution():
    """With one treated unit each placebo draw makes one control the treated
    one, so as B grows the placebo se converges to the population sd of the
    N_co leave-one-as-treated estimates computed at the full-sample sigma-hat."""
    df = _nb29_panel()
    Y = df.pivot(index="unit", columns="time", values="y")
    sig = SDID_MOD._noise_level(Y.to_numpy()[1:, :12])
    controls = df[df.treat_time.isna()]
    taus = []
    for u in controls.unit.unique():
        d = controls.assign(treat_time=np.where(controls.unit == u, 12.0, np.nan))
        taus.append(synthetic_did(d, n_boot=0, noise_level=sig).tau)
    sd_exhaustive = float(np.std(taus))
    res = synthetic_did(df, n_boot=400, seed=0)
    assert abs(res.se / sd_exhaustive - 1.0) < 0.10, (res.se, sd_exhaustive)


def test_bootstrap_with_one_treated_unit_warns():
    with pytest.warns(UserWarning, match="one treated unit"):
        res = synthetic_did(_nb29_panel(), n_boot=20, seed=0, se_method="bootstrap")
    assert np.isfinite(res.se)


def test_auto_uses_placebo_for_two_treated_units_and_bootstrap_from_three():
    """Monte Carlo (iid two-way FE, 30 controls, nominal 90 %, 200 draws):
    Algorithm 2 covered 0.790 with two treated units, 0.885 with three;
    the placebo covered 0.885 with two."""
    two = _twfe_panel(7, n_tr=2)
    auto = _no_warnings(synthetic_did, two, n_boot=20, seed=1)
    plc = _no_warnings(synthetic_did, two, n_boot=20, seed=1, se_method="placebo")
    assert auto.se == plc.se
    with pytest.warns(UserWarning, match="unreliable with only 2 treated units"):
        synthetic_did(two, n_boot=10, seed=1, se_method="bootstrap")
    three = _twfe_panel(7, n_tr=3)
    auto3 = _no_warnings(synthetic_did, three, n_boot=20, seed=1)
    boot3 = _no_warnings(synthetic_did, three, n_boot=20, seed=1, se_method="bootstrap")
    assert auto3.se == boot3.se


def test_bootstrap_resamples_treated_units_too():
    """Old code, docs example (8 treated of 40): donor bootstrap se 0.0190.
    The treated units' own noise alone gives sd(tau-hat) of about
    0.3 * sqrt(2 / (8 * 12)) = 0.043; Algorithm 2 captures it."""
    res = _no_warnings(synthetic_did, _docs_panel(), n_boot=200, seed=0)
    assert getattr(res, "se_method", "bootstrap") == "bootstrap"
    assert 0.035 < res.se < 0.08, res.se


def test_jackknife_matches_algorithm_3_and_is_undefined_for_one_treated():
    df = _docs_panel()
    res = _no_warnings(synthetic_did, df, se_method="jackknife")
    Y = df.pivot(index="unit", columns="time", values="y")
    tr = df.loc[df.treat_time.notna(), "unit"].unique()
    co = [u for u in Y.index if u not in set(tr)]
    pre = [t for t in Y.columns if t < 12]
    post = [t for t in Y.columns if t >= 12]
    om = res.omega.reindex(co).to_numpy()
    lam = res.lambda_w.reindex(pre).to_numpy()

    def tau_of(cos, trs, w):
        yc_pre = Y.loc[cos, pre].to_numpy(); yc_post = Y.loc[cos, post].to_numpy()
        yt_pre = Y.loc[trs, pre].to_numpy().mean(0); yt_post = Y.loc[trs, post].to_numpy().mean(0)
        w = w / w.sum()
        return (yt_post.mean() - w @ yc_post.mean(1)) - lam @ (yt_pre - w @ yc_pre)

    loo = [tau_of([c for c in co if c != u], list(tr), om[np.array(co) != u]) for u in co]
    loo += [tau_of(co, [t for t in tr if t != u], om) for u in tr]
    n = len(loo)
    se = np.sqrt((n - 1) / n * np.sum((np.array(loo) - res.tau) ** 2))
    np.testing.assert_allclose(res.se, se, rtol=1e-10)
    with pytest.warns(UserWarning, match="jackknife"):
        one = synthetic_did(_nb29_panel(), se_method="jackknife")
    assert np.isnan(one.se) and np.isnan(one.lo) and np.isnan(one.hi)


def test_inference_argument_validation():
    df = _nb29_panel()
    with pytest.raises(ValueError, match="se_method"):
        synthetic_did(df, n_boot=5, se_method="donor")
    # placebo needs more controls than treated units
    small = _twfe_panel(0, n_co=3, n_tr=3, T=8, T0=5)
    with pytest.raises(ValueError, match="more control units than treated"):
        synthetic_did(small, n_boot=5, se_method="placebo")
    r0 = synthetic_did(df, n_boot=0)
    assert np.isnan(r0.se) and np.isnan(r0.lo) and np.isnan(r0.hi)


# ---------------------------------------------------------------------------
# Multi-cohort wrapper
# ---------------------------------------------------------------------------


def test_sdid_multi_cohort_is_scale_equivariant():
    """sdid_multi_cohort calls synthetic_did per cohort, so it inherited the
    silent uniform fallback: at y * 1e4 its ATT was the DiD number."""
    rng = np.random.default_rng(5)
    n, T = 24, 16
    cohort = np.array([6] * 4 + [10] * 4 + [0] * 16)
    a = rng.normal(size=n)
    load = rng.uniform(0.5, 2.0, size=n)
    f = np.sin(np.linspace(0, 3 * np.pi, T)) * 3
    rows = []
    for i in range(n):
        for t in range(T):
            d = int(cohort[i] > 0 and t >= cohort[i])
            rows.append((a[i] + load[i] * f[t] + 1.5 * d + rng.normal(scale=0.3), d, i, t))
    y, d, p, t = map(np.asarray, zip(*rows))
    base = _no_warnings(sdid_multi_cohort, y, d, p, t, n_boot=0, seed=0)
    big = _no_warnings(sdid_multi_cohort, y * 1e4, d, p, t, n_boot=0, seed=0)
    np.testing.assert_allclose(big.cohort_atts / 1e4, base.cohort_atts, rtol=1e-6)
    np.testing.assert_allclose(big.att / 1e4, base.att, rtol=1e-6)


def test_sdid_multi_cohort_warns_with_one_treated_unit():
    df = _nb29_panel()
    D = ((df.treat_time.notna()) & (df.time >= df.treat_time)).astype(int).to_numpy()
    with pytest.warns(UserWarning, match="only 1 treated unit"):
        sdid_multi_cohort(df.y.to_numpy(), D, df.unit.to_numpy(), df.time.to_numpy(),
                          n_boot=5, seed=0)


# ---------------------------------------------------------------------------
# California Prop 99 (Arkhangelsky et al. 2021, Table 1, p. 8)
# ---------------------------------------------------------------------------

_PROP99_URL = ("https://raw.githubusercontent.com/synth-inference/synthdid/"
               "master/data/california_prop99.csv")


@pytest.fixture(scope="module")
def prop99():
    try:
        with urllib.request.urlopen(_PROP99_URL, timeout=30) as fh:
            raw = fh.read().decode()
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"cannot download the synthdid Prop 99 panel: {exc}")
    from io import StringIO
    d = pd.read_csv(StringIO(raw), sep=";")
    d["treat_time"] = np.where(d.State == "California", 1989.0, np.nan)
    return d


@pytest.mark.network
def test_prop99_matches_table_1(prop99):
    """Table 1: SDID -15.6 (placebo s.e. 8.4), SC -19.6, DID -27.3.
    Old code: -27.349 (the DID number, both weight vectors uniform)."""
    d = prop99
    res = _no_warnings(synthetic_did, d, unit="State", time="Year",
                       outcome="PacksPerCapita", treat_time="treat_time",
                       n_boot=200, seed=0)
    assert abs(res.tau - (-15.6)) < 0.05, res.tau
    lam = res.lambda_w[res.lambda_w > 1e-6]
    assert set(lam.index.astype(int)) == {1986, 1987, 1988}
    # The paper's 8.4 is one draw of B placebo replications; with B = 200 the
    # se varies by about 0.64 around its B -> infinity limit, 9.37.
    assert 7.0 < res.se < 11.5, res.se
    # SC and DID from the same weight solver: SC drops the intercept and the
    # time weights and uses zeta = 1e-6 sigma-hat (paper p. 7, footnote 5).
    # The exact SC optimum is -19.514 (objective 52.130, duality gap 1e-10,
    # confirmed by 4e5 accelerated projected-gradient steps); synthdid's
    # Frank-Wolfe stops earlier (objective 52.661) at -19.620, the paper's
    # -19.6. SDID is not affected: exact -15.605 vs Frank-Wolfe -15.604.
    Y = d.pivot(index="State", columns="Year", values="PacksPerCapita")
    co = [s for s in Y.index if s != "California"]
    pre = [t for t in Y.columns if t < 1989]
    post = [t for t in Y.columns if t >= 1989]
    Yc_pre = Y.loc[co, pre].to_numpy()
    sig = SDID_MOD._noise_level(Yc_pre)
    w_sc = SDID_MOD._solve_simplex_quadratic(
        Yc_pre.T, Y.loc["California", pre].to_numpy(),
        ridge=(1e-6 * sig) ** 2 * len(pre), intercept=False)
    tau_sc = Y.loc["California", post].mean() - w_sc @ Y.loc[co, post].to_numpy().mean(1)
    assert abs(tau_sc - (-19.514)) < 0.005, tau_sc
    assert abs(tau_sc - (-19.6)) < 0.1, tau_sc
    tau_did = ((Y.loc["California", post].mean() - Y.loc["California", pre].mean())
               - (Y.loc[co, post].to_numpy().mean() - Y.loc[co, pre].to_numpy().mean()))
    assert abs(tau_did - (-27.3)) < 0.05, tau_did


@pytest.mark.network
def test_prop99_is_scale_equivariant(prop99):
    d = prop99.assign(PacksPerCapita=prop99.PacksPerCapita * 1000.0)
    res = _no_warnings(synthetic_did, d, unit="State", time="Year",
                       outcome="PacksPerCapita", treat_time="treat_time", n_boot=0)
    assert abs(res.tau / 1000.0 - (-15.6)) < 0.05


# ---------------------------------------------------------------------------
# Coverage (slow tier)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_placebo_interval_covers_near_nominal_with_one_treated_unit():
    """Old code: the donor-bootstrap 90 % interval covered 0.50-0.67.
    Arkhangelsky et al. report 0.97 placebo coverage at nominal 0.95 for
    N_tr = 1 (Table 4); here nominal 0.90 on an iid TWFE design."""
    hits = []
    for r in range(200):
        df = _twfe_panel(1000 + r, n_co=30, n_tr=1, T=20, T0=15, tau=0.0)
        res = synthetic_did(df, n_boot=50, seed=r)
        hits.append(res.lo <= 0.0 <= res.hi)
    cov = float(np.mean(hits))
    assert 0.83 <= cov <= 0.97, cov
