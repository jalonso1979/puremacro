"""FU-MISC: follow-ups to defects found while integrating the 2026-09-30 fixes.

1. The LP HAC truncation lag h+1 is no longer credited to Plagborg-Moller and
   Wolf (2021), which makes no HAC recommendation; it is stated as a
   convention (teaching.lp_sm, validation case lp.jorda_hac_se_vs_statsmodels).
2. tools/build_sw07_data.py writes the SW07 fixture once, as package data; the
   orphaned tests/fixtures copy is gone.
3. The SW07 harmonic-mean replication case pins the estimator's
   non-convergence explicitly and no longer hides its warning.
4. synthetic_did(n_boot=0) records no variance estimator.
5. proxy_svar: shock_target_idx selects the residual of the first-stage F only.
6. The legacy (non-Hicksian) Nash tariff search records
   metadata["boundary_bands_overlap"] like the Hicksian searches.
7. docs/vfi_continuous_transition.md (+es) describe the multiplicative
   discount-factor shock and the permanent-rate-wedge limitation.
9. SpatialLPResult._metadata lists each attribute once.
(8, girf with a 1-D Y, is tested in tests/test_var_regime_girf_coverage.py.)
"""
from __future__ import annotations

import importlib.util
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. HAC bandwidth attribution
# ---------------------------------------------------------------------------

def test_hac_bandwidth_is_not_credited_to_pmw():
    from puremacro.teaching import lp_sm
    from puremacro.validation.cases_lp import CASES

    case = {c.id: c for c in CASES}["lp.jorda_hac_se_vs_statsmodels"]
    assert "Plagborg" not in case.citation
    assert "convention" in case.citation
    assert "NUMERIC" not in case.notes          # the case runs at Tol.TIGHT
    doc = inspect.getdoc(lp_sm.lp_ols_hac)
    assert "convention" in doc
    assert "(Plagborg-Møller & Wolf 2021)" not in doc


# ---------------------------------------------------------------------------
# 2. One SW07 fixture
# ---------------------------------------------------------------------------

def _build_sw07_module():
    spec = importlib.util.spec_from_file_location(
        "_fu_misc_build_sw07_data", ROOT / "tools" / "build_sw07_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sw07_fixture_is_written_once_as_package_data():
    mod = _build_sw07_module()
    assert mod.DEFAULT_FIXTURES == (
        ROOT / "puremacro" / "replication" / "data" / mod.FIXTURE_NAME,)
    assert not (ROOT / "tests" / "fixtures" / mod.FIXTURE_NAME).exists()
    assert (ROOT / "puremacro" / "replication" / "data" / mod.FIXTURE_NAME).is_file()


# ---------------------------------------------------------------------------
# 3. Harmonic-mean case: non-convergence stated, warning not hidden
# ---------------------------------------------------------------------------

def test_harmonic_mean_case_pins_nonconvergence_and_warns():
    from puremacro.dsge import marginal
    from puremacro.replication import cases_dsge_estimation as C
    from puremacro.replication._model import run

    case = {c.id: c for c in C.CASES}["dsge_estimation.sw07_harmonic_mean_mdd_consistency"]
    assert case.target["converged"] == 0.0
    assert case.target["truncation_spread"] > marginal._SPREAD_TOL
    text = (case.title + " " + case.notes).lower()
    assert "not converged" in text

    with pytest.warns(UserWarning, match="has not converged"):
        metrics = C._eval_sw07_harmonic_mean_mdd()
    assert metrics["converged"] == 0.0
    assert metrics["truncation_spread"] > marginal._SPREAD_TOL

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = run(case)
    assert result.passed, (result.margin, result.error, result.metrics)
    assert result.metrics["converged"] == {"got": 0.0, "target": 0.0}

    src = inspect.getsource(C._eval_sw07_harmonic_mean_mdd)
    assert "simplefilter" not in src and "catch_warnings" not in src


# ---------------------------------------------------------------------------
# 4. synthetic_did(n_boot=0)
# ---------------------------------------------------------------------------

def _sdid_panel(n_tr: int = 1, n_co: int = 12, T: int = 10, T0: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = n_co + n_tr
    y = rng.normal(size=n)[:, None] + rng.normal(size=T)[None, :] + 0.3 * rng.normal(size=(n, T))
    y[n_co:, T0:] += 1.0
    return pd.DataFrame([{"unit": i, "time": t, "y": y[i, t],
                          "treat_time": float(T0) if i >= n_co else np.nan}
                         for i in range(n) for t in range(T)])


@pytest.mark.parametrize("se_method", ["auto", "placebo", "bootstrap", "jackknife"])
def test_sdid_without_replications_records_no_se_method(se_method):
    from puremacro.did import synthetic_did

    df = _sdid_panel(n_tr=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # no bootstrap caveat without a bootstrap
        res = synthetic_did(df, n_boot=0, se_method=se_method)
    assert res.se_method is None
    assert res.n_reps == 0
    assert np.isnan(res.se) and np.isnan(res.lo) and np.isnan(res.hi)
    text = res.summary()
    assert "se method" not in text and "bootstrap" not in text
    assert "se_method" not in res.to_frame().columns
    assert np.isfinite(res.tau)


def test_sdid_with_replications_still_records_its_method():
    from puremacro.did import synthetic_did

    res = synthetic_did(_sdid_panel(n_tr=1), n_boot=20, seed=1)
    assert res.se_method == "placebo" and res.n_reps == 20
    assert "placebo (Algorithm 4), 20 replications" in res.summary()
    with pytest.warns(UserWarning, match="not well-defined with one treated unit"):
        synthetic_did(_sdid_panel(n_tr=1), n_boot=5, se_method="bootstrap")


# ---------------------------------------------------------------------------
# 5. proxy_svar: shock_target_idx only selects the first-stage residual
# ---------------------------------------------------------------------------

def test_proxy_first_stage_f_uses_the_target_residual():
    from puremacro.inference.weak_iv import olea_pflueger_f
    from puremacro.var.estimate import estimate_var
    from puremacro.var.identify.proxy import proxy_svar

    rng = np.random.default_rng(7)
    T, n = 300, 3
    eps = rng.standard_normal((T, n))
    z = eps[:, 1] + 0.5 * rng.standard_normal(T)       # proxy moves variable 1
    B = np.array([[1.0, 0.3, 0.0], [0.2, 1.0, 0.1], [0.0, 0.4, 1.0]])
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = 0.5 * Y[t - 1] + B @ eps[t]
    r0 = proxy_svar(Y, z, p=1, horizon=4, n_boot=10, shock_target_idx=0)
    r1 = proxy_svar(Y, z, p=1, horizon=4, n_boot=10, shock_target_idx=1)
    np.testing.assert_allclose(r0.B, r1.B, rtol=0, atol=1e-12)
    _, _, _, resid, _ = estimate_var(Y, 1)
    zz = z[-resid.shape[0]:].reshape(-1, 1)
    assert r1.first_stage_F == pytest.approx(olea_pflueger_f(resid[:, 1], zz), rel=1e-12)
    assert r0.first_stage_F == pytest.approx(olea_pflueger_f(resid[:, 0], zz), rel=1e-12)
    assert r1.first_stage_F > r0.first_stage_F
    doc = inspect.getdoc(proxy_svar)
    assert "resid[:, shock_target_idx]" in doc


# ---------------------------------------------------------------------------
# 6. Legacy Nash search records boundary_bands_overlap
# ---------------------------------------------------------------------------

def _teaching_calibration():
    from puremacro.trade import calibrate_trade_model

    flows = np.array([[10., 12., 24., 8., 6., 4., 20., 16.],
                      [8., 14., 4.5, 15., 10.5, 35., 18., 15.]])
    taxes = np.array([4., 6., 1.2, 1., .8, 1.8, 2., 1.2])
    value_added = flows.sum(1) - flows[:, :2].sum(0) - taxes[:2]
    data = np.vstack([flows, taxes, np.r_[2 * value_added / 3, np.zeros(6)],
                      np.r_[value_added / 3, np.zeros(6)]])
    return calibrate_trade_model(data, ns=1, nc=2, nfd=3, country_codes=["A", "B"])


@pytest.mark.parametrize("tariff_max, overlap", [(0.2, False), (5e-5, True)])
def test_legacy_nash_records_boundary_bands_overlap(tariff_max, overlap):
    from puremacro.trade import solve_multilateral_nash_tariffs

    nash = solve_multilateral_nash_tariffs(
        _teaching_calibration(), player_countries=("A", "B"), tariff_max=tariff_max,
        best_response_grid_size=3, max_iter=1, relaxation=1.0, tol=1e-4)
    assert nash.welfare_metric != "hicksian_ev"
    assert nash.metadata["boundary_bands_overlap"] is overlap
    assert set(nash.metadata["best_response_boundaries"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# 7. Continuous-transition docs match the code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["docs/vfi_continuous_transition.md",
                                  "docs/es/vfi_continuous_transition.md"])
def test_transition_docs_state_beta_rule_and_wedge_limitation(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    assert r"\beta_t = \beta\,(1 + s_t)" in text
    assert r"\beta_{\text{ss}} + \Delta \beta" not in text   # the old additive form
    assert "### 3.1" in text and "r_T = r^* = 0.0394" in text
    assert "TypeError" in text and 'metadata["params"]' in text
    for kw in ("egm_tol", "egm_max_iter", "xtol", "tol_ge", "max_evals",
               "dist_options", "r_min", "r_max"):
        assert f"`{kw}`" in text, kw


def test_transition_beta_shock_is_multiplicative_in_code():
    """The rule the docs describe, read off the code path that builds beta_t."""
    from puremacro.vfi import continuous_transition as ct

    src = inspect.getsource(ct.solve_continuous_transition)
    assert "beta_path[:len(s_arr)] = beta * (1.0 + s_arr)" in src
    assert "if np.max(s_arr) > 0.5:" in src


# ---------------------------------------------------------------------------
# 9. SpatialLPResult._metadata
# ---------------------------------------------------------------------------

def test_spatial_lp_metadata_has_no_duplicates():
    from puremacro.lp._results import LPResult
    from puremacro.spatial.lp import SpatialLPResult

    meta = SpatialLPResult._metadata
    assert len(meta) == len(set(meta))
    assert "n_lags" in meta
    assert meta[:len(LPResult._metadata)] == LPResult._metadata
    for name in ("spillover", "cumulative", "total_scale", "spec", "notes"):
        assert name in meta
