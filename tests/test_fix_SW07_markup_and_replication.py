"""Regression tests for the SW07 fix (review 2026-09-30, defect key SW07).

1. The price and wage markup disturbances of the native Smets-Wouters (2007)
   model are ARMA(1,1) with a *contemporaneous* innovation,

       eps^p_t = rho_p eps^p_{t-1} + eta^p_t - mu_p eta^p_{t-1}
       eps^w_t = rho_w eps^w_{t-1} + eta^w_t - mu_w eta^w_{t-1}

   (Smets & Wouters 2007, ECB WP 722, printed pp. 14-15, below eqs. (10) and
   (13); ``sw07_pfeifer.mod`` lines 307-311). Before the fix the MA terms were
   dropped (``cmap``/``cmaw`` never entered) and the innovation reached
   ``spinf``/``sw`` one quarter late.
2. The bundled hours observable is 100 x log of per-capita hours, as in the
   SW07 data appendix (WP 722 printed p. 47).
3. The SW07 replication cases do not present puremacro-generated numbers as
   published, compare the optimised mode with the Table 1a/1b *Mode* column,
   and load their fixture from package data (so they run from a wheel).
"""
from __future__ import annotations

import importlib.resources
import re
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.smets_wouters import (
    SHOCK_NAMES,
    SW07_POSTERIOR_MODE,
    _shock_irf,
    solve_sw07,
)

ROOT = Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "puremacro" / "dsge" / "_references" / "sw07_pfeifer.mod"

# SW07 Table 1b posterior modes of the markup processes (ECB WP 722, PDF p.36).
MA_ON = {"crhopinf": 0.90, "cmap": 0.74, "crhow": 0.97, "cmaw": 0.88}


def _arma11_weights(rho: float, mu: float, n: int) -> np.ndarray:
    """MA(inf) weights of x_t = rho x_{t-1} + e_t - mu e_{t-1}."""
    psi = np.empty(n)
    psi[0] = 1.0
    for j in range(1, n):
        psi[j] = rho ** (j - 1) * (rho - mu)
    return psi


def _mod_with(overrides: dict[str, float]):
    """Load the reference .mod FRESH with the parameter lines rewritten.

    A fresh load (not ``load_mod(params=...)``) evaluates the .mod's
    parameter-only model-local ``#`` variables at the requested values.
    """
    from puremacro.dsge import load_mod

    text = MOD_PATH.read_text(encoding="utf-8")
    for k, v in overrides.items():
        text, n = re.subn(rf"^(\s*{k}\s*=\s*)[^;]*;", rf"\g<1>{v!r};", text,
                          count=1, flags=re.M)
        assert n == 1, k
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_mod(text, order=1)


# ---------------------------------------------------------------- 1. model --

def test_cmap_cmaw_enter_the_solution():
    base = dict(crhopinf=0.9, crhow=0.97)
    a = solve_sw07({**base, "cmap": 0.74, "cmaw": 0.88})
    b = solve_sw07({**base, "cmap": 0.0, "cmaw": 0.0})
    assert np.abs(a.G - b.G).max() > 0.1


@pytest.mark.parametrize("shock,state,rho,mu", [
    ("epinf", "spinf", "crhopinf", "cmap"),
    ("ew", "sw", "crhow", "cmaw"),
])
@pytest.mark.parametrize("params", [{}, MA_ON], ids=["mod_calibration", "table1b_mode"])
def test_markup_irf_is_contemporaneous_arma11(shock, state, rho, mu, params):
    p = {**SW07_POSTERIOR_MODE, **params}
    irf = _shock_irf(shock, range(12), 1.0, params or None)
    expected = _arma11_weights(p[rho], p[mu], 12)
    np.testing.assert_allclose(irf[state].to_numpy(), expected, atol=1e-12)


@pytest.mark.parametrize("params", [{}, MA_ON], ids=["mod_calibration", "table1b_mode"])
def test_native_irfs_match_reference_mod(params):
    """All 7 shocks x 9 variables x 20 quarters coincide with the .mod engine."""
    m = _mod_with(params)
    worst = 0.0
    for shk in SHOCK_NAMES:
        ref = m.irf(shk, horizon=20, size=1.0)
        nat = _shock_irf(shk, range(20), 1.0, params or None).set_index("h")
        for v in ("y", "c", "inve", "lab", "pinf", "w", "r", "spinf", "sw"):
            d = np.abs(ref[v].to_numpy()[:20] - nat[v].to_numpy()[:20]).max()
            worst = max(worst, float(d))
    assert worst < 1e-9, worst


def test_cmap_moves_the_likelihood():
    from puremacro.dsge.estimate import _make_neg_log_posterior
    from puremacro.dsge.sw07_estimate import _FIXED_PARAMS, _load_bundled_data
    from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
    from puremacro.dsge.sw07_priors import PRIORS, param_names

    names = param_names()
    y = _load_bundled_data()[list(OBSERVED_VARS)].to_numpy()
    nlp = _make_neg_log_posterior(y, make_state_space, PRIORS, names, _FIXED_PARAMS)
    # An in-support parameter vector: SW07's published modes (Tables 1a/1b).
    theta = {**_TABLE1_MODE, **_TABLE1B_SHOCK_SD_MODE, "constelab": 0.0}
    vec = np.array([theta[n] for n in names], dtype=float)
    for par in ("cmap", "cmaw"):
        i = names.index(par)
        lo, hi = vec.copy(), vec.copy()
        lo[i], hi[i] = 0.2, 0.8
        f_lo, f_hi = nlp(lo), nlp(hi)
        assert np.isfinite(f_lo) and np.isfinite(f_hi), par
        assert abs(f_lo - f_hi) > 1.0, (par, f_lo, f_hi)


# ----------------------------------------------------------------- 2. data --

def _bundled():
    from puremacro.dsge.sw07_estimate import _load_bundled_data
    return _load_bundled_data()


def test_hours_are_100_times_log_per_capita():
    df = _bundled()
    # 100 x log units: the std of per-capita hours over 1966-2004 is a few
    # percent, the same order as the other percent-unit observables. Before
    # the fix it was 0.046 (plain log).
    sd = float(df["log_hours"].std())
    assert 1.0 < sd < 10.0, sd
    header = (importlib.resources.files("puremacro.dsge") / "_sw07_data.csv").read_text(
        encoding="utf-8").splitlines()
    comments = "\n".join(line for line in header if line.startswith("#"))
    # SW07 data appendix: NFB average hours index x civilian employment 16+,
    # divided by civilian population 16+.
    for series in ("PRS85006023", "CE16OV", "CNP16OV"):
        assert series in comments, series
    assert "100" in comments and "log" in comments


def test_builder_applies_the_sw07_definitions():
    """tools/build_sw07_data.transform on synthetic levels: hours are
    100*log(avg hours index * employment / population), demeaned; C and I
    are nominal series deflated by the GDP deflator, per capita, 100*dlog."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_build_sw07_data", ROOT / "tools" / "build_sw07_data.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)

    q = pd.period_range("1965Q4", "2004Q4", freq="Q").to_timestamp()
    m = pd.period_range("1965-10", "2004-12", freq="M").to_timestamp()
    rng = np.random.default_rng(0)
    nq, nm = len(q), len(m)

    def walk(n, lvl, sd):
        return lvl * np.exp(np.cumsum(rng.normal(0.004, sd, n)))

    raw = {
        "gdp": pd.Series(walk(nq, 3000, .01), q), "cons_nom": pd.Series(walk(nq, 400, .01), q),
        "inv_nom": pd.Series(walk(nq, 100, .03), q), "deflator": pd.Series(walk(nq, 20, .005), q),
        "hours_avg": pd.Series(100 + rng.normal(0, 1, nq), q),
        "emp": pd.Series(walk(nm, 70000, .002), m), "wage_nom": pd.Series(walk(nq, 5, .01), q),
        "ffr": pd.Series(5 + rng.normal(0, 1, nm), m), "pop": pd.Series(walk(nm, 130000, .001), m),
    }
    out = B.transform(raw)
    assert len(out) == 156 and str(out.index[0]) == "1966Q1"

    def qavg(s):
        return s.groupby(s.index.to_period("Q")).mean()

    pop, emp = qavg(raw["pop"]), qavg(raw["emp"])
    hours = 100 * np.log(qavg(raw["hours_avg"]) * emp / pop)
    hours = hours.loc["1966Q1":]
    np.testing.assert_allclose(out["log_hours"].to_numpy(), (hours - hours.mean()).to_numpy())
    inv = qavg(raw["inv_nom"]) / qavg(raw["deflator"]) / pop
    np.testing.assert_allclose(out["inv_growth"].to_numpy(),
                               (100 * np.log(inv / inv.shift(1))).loc["1966Q1":].to_numpy())
    np.testing.assert_allclose(out["ffr"].to_numpy(), (qavg(raw["ffr"]) / 4).loc["1966Q1":].to_numpy())


def test_sw07_dataset_sample():
    df = _bundled()
    assert str(df.index.min().date()) == "1966-01-01"
    assert str(df.index.max().date()) == "2004-10-01"
    assert len(df) == 156
    assert df.notna().all().all()


# ---------------------------------------------------------- 3. replication --

# SW07 posterior MODE column, read in ECB WP 722: Table 1a (PDF p.35) and
# Table 1b (PDF p.36). The Mean column differs for csadjcost (5.74),
# csigma (1.38), csigl (1.83), cprobp (0.66), cfc (1.60), ...
_TABLE1_MODE = {
    "csadjcost": 5.48, "csigma": 1.39, "chabb": 0.71, "cprobw": 0.73,
    "csigl": 1.92, "cprobp": 0.65, "cindw": 0.59, "cindp": 0.22,
    "czcap": 0.54, "cfc": 1.61, "crpi": 2.03, "crr": 0.81, "cry": 0.08,
    "crdy": 0.22, "constepinf": 0.81, "constebeta": 0.16, "ctrend": 0.43,
    "calfa": 0.19,
    "crhoa": 0.95, "crhob": 0.18, "crhog": 0.97, "crhoqs": 0.71,
    "crhoms": 0.12, "crhopinf": 0.90, "crhow": 0.97, "cmap": 0.74,
    "cmaw": 0.88, "cgy": 0.52,
}


# Table 1b (PDF p.36) Mode column for the shock standard deviations.
_TABLE1B_SHOCK_SD_MODE = {"ea": 0.45, "eb": 0.24, "eg": 0.52, "eqs": 0.45,
                          "em": 0.24, "epinf": 0.14, "ew": 0.24}


def _cases():
    from puremacro.replication import cases_dsge_estimation as C
    return {c.id: c for c in C.CASES}


def test_mode_case_targets_the_table1_mode_column():
    case = _cases()["dsge_estimation.sw07_structural_parameters_mode"]
    assert case.target, "empty target"
    for k, v in case.target.items():
        assert k in _TABLE1_MODE, k
        assert v == _TABLE1_MODE[k], (k, v, _TABLE1_MODE[k])
    assert "Mode" in case.citation and "Table 1" in case.citation


def test_regression_values_are_not_presented_as_published():
    cases = _cases()
    for cid in ("dsge_estimation.sw07_log_posterior_at_mode",
                "dsge_estimation.sw07_laplace_marginal_data_density",
                "dsge_estimation.sw07_harmonic_mean_mdd_consistency"):
        c = cases[cid]
        text = (c.citation + " " + c.notes + " " + c.title).lower()
        assert "regression" in text and "not published" in text, cid
        for bad in (-1673.72, -1686.09, -2524.36):
            assert bad not in [float(t) for t in c.target.values()], cid


def test_fixture_ships_as_package_data():
    from puremacro.replication import cases_dsge_estimation as C

    res = importlib.resources.files("puremacro.replication.data") / C._FIXTURE_NAME
    assert res.is_file()
    # No path outside the installed package.
    src = Path(C.__file__).read_text(encoding="utf-8")
    assert "parents[2]" not in src and '"tests"' not in src
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    globs = pyproject["tool"]["setuptools"]["package-data"]["puremacro.replication.data"]
    assert any(g.endswith(".npz") for g in globs), globs


def test_fixture_mode_is_a_stationary_point():
    """The stored mode is an optimiser's mode on the bundled data (not the
    best of a short MCMC chain): the log-posterior does not rise along any
    coordinate by more than a small tolerance."""
    from puremacro.dsge.estimate import _make_neg_log_posterior
    from puremacro.dsge.sw07_estimate import _FIXED_PARAMS, _load_bundled_data
    from puremacro.dsge.sw07_observation import OBSERVED_VARS, make_state_space
    from puremacro.dsge.sw07_priors import PRIORS, param_names
    from puremacro.replication import cases_dsge_estimation as C

    fix = C._load_fixture()
    names = param_names()
    assert tuple(str(n) for n in fix["param_names"]) == names
    mode = np.asarray(fix["mode_values"], dtype=float)
    y = _load_bundled_data()[list(OBSERVED_VARS)].to_numpy()
    nlp = _make_neg_log_posterior(y, make_state_space, PRIORS, names, _FIXED_PARAMS)
    f0 = nlp(mode)
    np.testing.assert_allclose(-f0, float(fix["log_post_mode"]), atol=1e-6)
    lb = np.array([PRIORS[n]["lb"] for n in names])
    ub = np.array([PRIORS[n]["ub"] for n in names])
    for i in range(len(names)):
        h = 1e-3 * max(abs(mode[i]), 0.1)
        for s in (-1.0, 1.0):
            v = mode.copy()
            v[i] = np.clip(v[i] + s * h, lb[i], ub[i])
            assert nlp(v) >= f0 - 1e-3, (names[i], s, f0 - nlp(v))


def test_dsge_estimation_cases_pass():
    from puremacro.replication import run_all

    results = {r.id: r for r in run_all(family="dsge_estimation")}
    # 4.4.0: mode, Laplace, harmonic mean, structural parameters; 4.6.0 adds the
    # Dynare 8 check of the log posterior at the authors' mode.
    assert len(results) == 5
    for cid, r in results.items():
        assert r.passed, (cid, r.margin, r.error, r.metrics)
