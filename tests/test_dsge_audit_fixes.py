"""Regression tests for the 4.3.0 audit findings of the DSGE cluster.

Findings covered (see the 2026-09-22 audit, dimension ``dsge-vfi-4.3.0``):

1. ``verify_dynare_parity`` at order 2 called ``theoretical_moments(ar=...)``
   on a ``PrunedDSGESolution`` whose keyword is ``lags``; every order-2
   comparison with supplied ``oo_.mean/var/autocorr`` ended UNAVAILABLE.
   A ``LinearModel`` passed with ``order=2`` compared first-order moments,
   so the mean failed by the order-2 risk correction (2.22e-2 on rbc).
2. Order-1 parity never compared ``M_.Sigma_e`` because ``LinearModel``
   keeps its covariance in ``_shock_cov``.
3. ``oo_.dr`` present but empty (Dynare's initial ``oo_.dr = []``) raised
   ``AttributeError`` instead of returning UNAVAILABLE.
4. ``solve_ms_dsge`` gated convergence on an absolute residual, so a model
   whose structural matrices are written in large units (scale 1e7 and up)
   was reported unsolved with all-NaN outputs although the fixed point was
   exact to roundoff. The gate is now entrywise (componentwise backward
   error): a first fix compared max|F| with rtol times the largest term of
   the whole system and certified O(1) equations at residuals set by a
   large-unit equation (T wrong by up to 1.9e-2); the row-scaled, column-scaled
   and user-``tol`` tests below pin the entrywise behaviour.
5. ``CollocationProblem`` evaluated the auxiliary value coefficients under
   CRRA(``gamma``) while the Euler residual read only ``sigma``: passing
   ``{'gamma': 3}`` solved a log-utility policy and valued it under CRRA(3).
   The Euler residual now honours ``gamma``, so IFT gradients with respect
   to ``gamma`` are the true CRRA sensitivity (4.3.0 returned a zero column).
6. ``docs/vfi_analytic_gradients.md`` promised Aiyagari adjoint-distribution
   sensitivities that raise ``NotImplementedError``.

The live Dynare 7.0 fixtures under ``tests/fixtures/dynare_live`` carry real
decision rules but were exported with ``nomoments``: the moment rows of the
reference below are therefore puremacro's own order-2 moments (a
self-consistency reference), while the decision-rule rows are compared with
Dynare's numbers. A perturbed copy of the reference must fail.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from puremacro.dsge.dynare import load_mod
from puremacro.dsge.dynare_results import load_dynare_dr
from puremacro.dsge.markov_switching import solve_ms_dsge
from puremacro.dsge.parity import run_parity_suite, verify_dynare_parity
from puremacro.vfi.collocation import (
    CollocationProblem,
    _crra_curvature,
    _evaluate_euler_residual,
)

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "tests" / "fixtures" / "dynare_live"
RBC_MOD = LIVE / "rbc.mod"


# ---------------------------------------------------------------------------
# Dynare parity: order-2 moments, Sigma_e, empty oo_.dr
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rbc_reference():
    """Dynare 7.0 decision rules of rbc.mod plus puremacro's own order-2 moments."""
    d = np.load(LIVE / "rbc_order2.npz", allow_pickle=True)
    vn = [str(v).strip() for v in d["variable_names"]]
    sn = [str(v).strip() for v in d["state_names"]]
    en = [str(v).strip() for v in d["shock_names"]]
    dr = {
        "ghx": d["ghx"], "ghu": d["ghu"], "ys": d["ys"],
        "state_var": np.array([vn.index(s) + 1 for s in sn]),
        "ghxx": d["ghxx"], "ghxu": d["ghxu"], "ghuu": d["ghuu"], "ghs2": d["ghs2"],
    }
    pruned = load_mod(RBC_MOD, order=2)
    th = pruned.theoretical_moments()
    moments = {
        "mean": th.moments.loc[vn, "Mean"].to_numpy(),
        "var": th.covariance.loc[vn, vn].to_numpy(),
        "autocorr": th.autocorr.loc[vn].to_numpy(),
    }
    M_ = {"endo_names": np.array(vn), "exo_names": np.array(en), "Sigma_e": d["shock_cov"]}
    return {"dr": dr, "moments": moments, "M_": M_, "pruned": pruned, "vn": vn}


def _oo(ref, *, with_moments=True, order=2, mean_shift=0.0, sigma_scale=1.0):
    dr = dict(ref["dr"])
    if order == 1:
        dr = {k: dr[k] for k in ("ghx", "ghu", "ys", "state_var")}
    oo = {"dr": dr}
    if with_moments:
        oo.update({
            "mean": ref["moments"]["mean"] + mean_shift,
            "var": ref["moments"]["var"],
            "autocorr": ref["moments"]["autocorr"],
        })
    M_ = dict(ref["M_"], Sigma_e=sigma_scale * ref["M_"]["Sigma_e"])
    return {"oo_": oo, "M_": M_}


@pytest.mark.parametrize("kind", ["pruned_object", "mod_path", "linear_model"])
def test_order2_supplied_moments_are_compared_and_pass(rbc_reference, kind):
    """Order-2 references carrying mean/var/autocorr are COMPARED (4.3.0: UNAVAILABLE)."""
    model = {
        "pruned_object": rbc_reference["pruned"],
        "mod_path": str(RBC_MOD),
        "linear_model": load_mod(RBC_MOD, order=1),
    }[kind]
    res = verify_dynare_parity(model, _oo(rbc_reference), order=2)
    assert res.details["moments_status"] == "COMPARED"
    assert res.details["status"] == "PASS"
    assert res.passed and res.score == 100.0
    rows = res.moments_diff.set_index("moment")
    assert list(rows.index) == ["mean", "var", "autocorr"]
    assert (rows["status"] == "PASS").all()
    # The moments come from the object solved at order 2: with a LinearModel
    # input the mean row used to fail by the risk correction (2.22e-2 on rbc).
    assert rows.loc["mean", "deviation"] <= 1e-12
    assert rows.loc["var", "deviation"] <= 1e-12
    assert rows.loc["autocorr", "deviation"] <= 1e-12
    assert np.isfinite(res.max_dev_moments)
    # Real Dynare 7 decision rules still agree to roundoff
    assert res.max_dev_dr <= 1e-12
    assert res.details["covariance_status"] == "PASS"


def _dynare_order2_moments(dr, sigma_e, lags):
    """Order-2 ``oo_.mean``/``oo_.var``/``oo_.autocorr`` from a Dynare decision rule.

    Independent of puremacro's moment code: the pruned second-order mean
    ``E[y] = ys + ghx m_s + 0.5 (ghxx vec(Sx) + ghuu vec(Se) + ghs2)`` with
    ``m_s = (I - hx)^{-1} 0.5 (hxx vec(Sx) + huu vec(Se) + hs2)`` and the
    first-order (co)variances and autocorrelations, which is what Dynare's
    ``stoch_simul(order=2)`` reports. ``Sx`` solves ``Sx = hx Sx hx' + hu Se hu'``.
    """
    import scipy.linalg

    V, X, U = list(dr.variable_names), list(dr.state_variables), list(dr.shock_names)
    gx = dr.ghx.loc[V, X].to_numpy(float)
    gu = dr.ghu.loc[V, U].to_numpy(float)
    xi = [V.index(s) for s in X]
    hx, hu = gx[xi], gu[xi]
    sx = scipy.linalg.solve_discrete_lyapunov(hx, hu @ sigma_e @ hu.T)
    gxx = dr.ghxx.loc[V, [f"{a}_{b}" for a in X for b in X]].to_numpy(float)
    guu = dr.ghuu.loc[V, [f"{a}_{b}" for a in U for b in U]].to_numpy(float)
    correction = 0.5 * (gxx @ sx.reshape(-1) + guu @ sigma_e.reshape(-1) + dr.ghs2.loc[V].to_numpy(float))
    m_s = np.linalg.solve(np.eye(len(X)) - hx, correction[xi])
    mean = dr.ys.loc[V].to_numpy(float) + gx @ m_s + correction
    var = gx @ sx @ gx.T + gu @ sigma_e @ gu.T
    autocorr = np.empty((len(V), lags))
    hk = np.eye(len(X))
    for k in range(1, lags + 1):
        cov_k = gx @ hk @ hx @ sx @ gx.T + gx @ hk @ hu @ sigma_e @ gu.T
        hk = hk @ hx
        autocorr[:, k - 1] = np.diag(cov_k) / np.diag(var)
    return V, mean, var, autocorr


def test_order2_moments_match_formulas_applied_to_dynare_decision_rules(rbc_reference):
    """Reference moments built from Dynare 7's own rbc tensors, not from puremacro.

    The live fixtures were exported with ``nomoments``; the stoch_simul
    order-2 moment formulas applied to Dynare's ghx/ghu/ghxx/ghuu/ghs2 and
    M_.Sigma_e give an independent reference. Measured agreement:
    mean 3.6e-15, var 7.0e-14, autocorr 1.8e-14.
    """
    dyn = load_dynare_dr(_oo(rbc_reference, with_moments=False), order=2)
    en = list(rbc_reference["M_"]["exo_names"])
    perm_u = [en.index(u) for u in dyn.shock_names]
    sigma_e = np.asarray(rbc_reference["M_"]["Sigma_e"], dtype=float)[np.ix_(perm_u, perm_u)]
    V, mean, var, autocorr = _dynare_order2_moments(dyn, sigma_e, lags=5)
    p = [V.index(v) for v in rbc_reference["vn"]]  # oo_ rows follow M_.endo_names
    oo = _oo(rbc_reference, with_moments=False)
    oo["oo_"].update({"mean": mean[p], "var": var[np.ix_(p, p)], "autocorr": autocorr[p]})
    for model in (rbc_reference["pruned"], str(RBC_MOD), load_mod(RBC_MOD, order=1)):
        res = verify_dynare_parity(model, oo, order=2)
        assert res.details["moments_status"] == "COMPARED"
        assert res.passed and res.details["status"] == "PASS"
        rows = res.moments_diff.set_index("moment")
        assert list(rows.index) == ["mean", "var", "autocorr"]
        assert rows["deviation"].max() <= 1e-12
    # the order-2 mean really differs from the deterministic steady state
    assert np.max(np.abs(mean - dyn.ys.loc[V].to_numpy(float))) > 1e-3


def test_resolve_shock_covariance_uses_declared_covariances(rbc_reference):
    """Only a covariance the object declares (or carries, for pruned solutions) is compared."""
    import dataclasses

    from puremacro.dsge.parity import _resolve_shock_covariance

    sigma_e = np.asarray(rbc_reference["M_"]["Sigma_e"], dtype=float)
    pruned = rbc_reference["pruned"]
    np.testing.assert_array_equal(_resolve_shock_covariance(pruned), pruned.shock_cov)
    m1 = load_mod(RBC_MOD, order=1)
    np.testing.assert_allclose(_resolve_shock_covariance(m1), sigma_e, atol=0, rtol=0)
    # A LinearModel without a shocks block declares nothing: its moments assume
    # the identity, but that convention is not compared with M_.Sigma_e.
    assert _resolve_shock_covariance(dataclasses.replace(m1, _shock_cov=None)) is None
    # A pruned solution always carries shock_cov (identity filled in at construction).
    undeclared = dataclasses.replace(pruned, shock_cov=None)
    np.testing.assert_array_equal(_resolve_shock_covariance(undeclared), np.eye(pruned.n_shocks))
    assert _resolve_shock_covariance(load_dynare_dr(_oo(rbc_reference, order=1), order=1)) is None


def test_order1_undeclared_covariance_linear_model_is_not_a_false_fail(rbc_reference):
    """Order-1 decision rules do not depend on Sigma_e: an undeclared covariance is UNAVAILABLE.

    The first fix compared the implicit identity with rbc's Sigma_e = 1e-4 and
    turned a passing decision-rule parity (DR deviation 2.7e-15) into FAIL, score 92.3.
    """
    import dataclasses

    m_undeclared = dataclasses.replace(load_mod(RBC_MOD, order=1), _shock_cov=None)
    for scale in (1.0, 100.0):
        res = verify_dynare_parity(m_undeclared, _oo(rbc_reference, with_moments=False, order=1, sigma_scale=scale),
                                   order=1)
        assert res.details["covariance_status"] == "UNAVAILABLE"
        assert res.passed and res.score == 100.0
        assert res.max_dev_dr <= 1e-12


def _live_reference(stem):
    """Dynare 7.0 order-2 decision rules and M_ of a live fixture, as a loadmat-style dict."""
    d = np.load(LIVE / f"{stem}_order2.npz", allow_pickle=True)
    vn = [str(v).strip() for v in d["variable_names"]]
    sn = [str(v).strip() for v in d["state_names"]]
    en = [str(v).strip() for v in d["shock_names"]]
    dr = {"ghx": d["ghx"], "ghu": d["ghu"], "ys": d["ys"], "state_var": np.array([vn.index(x) + 1 for x in sn]),
          "ghxx": d["ghxx"], "ghxu": d["ghxu"], "ghuu": d["ghuu"], "ghs2": d["ghs2"]}
    return {"oo_": {"dr": dr}, "M_": {"endo_names": np.array(vn), "exo_names": np.array(en),
                                       "Sigma_e": d["shock_cov"]}}, vn


@pytest.mark.parametrize("stem", ["correlated_cubic", "nonlinear_multishock"])
def test_order2_zero_variance_autocorr_rows_are_matched_not_unavailable(stem):
    """A zero-variance variable has NaN autocorrelations on both sides: matched, listed, not UNAVAILABLE.

    The first fix left the whole moment block UNAVAILABLE on these fixtures
    although mean/var matched to 0.0 (the variable ``y`` has variance 0).
    """
    oo, vn = _live_reference(stem)
    lm = load_mod(LIVE / f"{stem}.mod", order=1)
    th = lm.solve(order=2).theoretical_moments()
    autocorr = th.autocorr.loc[vn].to_numpy()
    assert (~np.isfinite(autocorr)).any()  # the premise: undefined rows exist
    oo["oo_"].update({"mean": th.moments.loc[vn, "Mean"].to_numpy(), "var": th.covariance.loc[vn, vn].to_numpy(),
                      "autocorr": autocorr})
    res = verify_dynare_parity(lm, oo, order=2)
    assert res.details["moments_status"] == "COMPARED"
    assert res.passed and res.details["status"] == "PASS"
    assert res.details["moments_excluded"] == {"autocorr": ["y"]}
    rows = res.moments_diff.set_index("moment")
    assert list(rows.index) == ["mean", "var", "autocorr"] and (rows["status"] == "PASS").all()
    # a real mismatch in a defined row still fails
    bad = dict(oo["oo_"], autocorr=autocorr.copy())
    finite_row = int(np.flatnonzero(np.isfinite(autocorr).all(axis=1))[0])
    bad["autocorr"][finite_row, 0] += 1e-3
    res_bad = verify_dynare_parity(lm, {"oo_": bad, "M_": oo["M_"]}, order=2)
    assert res_bad.details["moments_status"] == "COMPARED" and not res_bad.passed
    assert res_bad.moments_diff.set_index("moment").loc["autocorr", "status"] == "FAIL"
    # NaN on one side only is still UNAVAILABLE
    one_sided = dict(oo["oo_"], autocorr=np.nan_to_num(autocorr, nan=0.0))
    res_one = verify_dynare_parity(lm, {"oo_": one_sided, "M_": oo["M_"]}, order=2)
    assert res_one.details["moments_status"] == "UNAVAILABLE" and not res_one.passed


def test_parity_notes_state_the_pruning_convention():
    """Order-2 second moments follow stoch_simul(order=2) without ``pruning``; the docstring says so."""
    doc = " ".join(verify_dynare_parity.__doc__.split())
    assert "*without* the ``pruning`` option" in doc
    assert "disp_th_moments_pruned_state_space" in doc


def test_record_array_oo_names_the_loadmat_option(tmp_path, rbc_reference):
    """scipy.io.loadmat's default struct_as_record=True gives a record array: say how to load it."""
    path = tmp_path / "rbc_results.mat"
    oo = _oo(rbc_reference, with_moments=False, order=1)
    scipy.io.savemat(str(path), {"oo_": oo["oo_"], "M_": {"Sigma_e": oo["M_"]["Sigma_e"]}})
    raw = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=True)
    res = verify_dynare_parity(load_mod(RBC_MOD, order=1), raw, order=1)
    assert res.details["status"] == "UNAVAILABLE" and not res.passed
    assert "struct_as_record=False" in res.details["error"]
    assert "Missing decision-rule fields" not in res.details["error"]


def test_order2_supplied_moments_fail_when_perturbed(rbc_reference):
    """A wrong reference mean is a COMPARED FAIL, not UNAVAILABLE and not a pass."""
    res = verify_dynare_parity(rbc_reference["pruned"], _oo(rbc_reference, mean_shift=1e-3), order=2)
    assert res.details["moments_status"] == "COMPARED"
    assert res.details["status"] == "FAIL"
    assert not res.passed
    rows = res.moments_diff.set_index("moment")
    assert rows.loc["mean", "status"] == "FAIL"
    assert rows.loc["mean", "deviation"] == pytest.approx(1e-3, rel=1e-6)
    assert rows.loc["var", "status"] == "PASS"
    assert rows.loc["autocorr", "status"] == "PASS"


def test_order2_moments_lag_count_follows_reference(rbc_reference):
    """The number of autocorrelation lags compared follows the supplied array."""
    oo = _oo(rbc_reference)
    oo["oo_"]["autocorr"] = oo["oo_"]["autocorr"][:, :2]
    res = verify_dynare_parity(rbc_reference["pruned"], oo, order=2)
    assert res.details["moments_status"] == "COMPARED"
    assert res.passed


def test_order1_parity_compares_sigma_e(rbc_reference):
    """M_.Sigma_e is compared with the covariance a LinearModel declares (4.3.0: never)."""
    m1 = load_mod(RBC_MOD, order=1)
    res = verify_dynare_parity(m1, _oo(rbc_reference, with_moments=False, order=1), order=1)
    assert res.details["covariance_status"] == "PASS"
    assert res.passed and res.score == 100.0

    wrong = verify_dynare_parity(m1, _oo(rbc_reference, with_moments=False, order=1, sigma_scale=100.0), order=1)
    assert wrong.details["covariance_status"] == "FAIL"
    assert not wrong.passed
    assert wrong.score < 100.0
    # The decision rules are Sigma_e-independent and still agree.
    assert wrong.max_dev_dr <= 1e-12


def test_covariance_unavailable_for_decision_rule_only_object(rbc_reference):
    """A DynareDR carries no covariance: honest UNAVAILABLE, not counted as a check."""
    oo = _oo(rbc_reference, with_moments=False, order=1)
    dr_only = load_dynare_dr(oo, order=1)
    res = verify_dynare_parity(dr_only, oo, order=1)
    assert res.details["covariance_status"] == "UNAVAILABLE"
    assert res.passed and res.score == 100.0


@pytest.mark.parametrize("bad_dr", [None, np.array([]), [], "not a struct", 0.0])
def test_empty_dr_is_unavailable_not_an_exception(bad_dr):
    """Dynare initialises oo_.dr = []; a run that never reached stoch_simul is UNAVAILABLE."""
    m1 = load_mod(RBC_MOD, order=1)
    res = verify_dynare_parity(m1, {"oo_": {"dr": bad_dr}}, order=1)
    assert res.details["status"] == "UNAVAILABLE"
    assert not res.passed and res.score == 0.0
    assert "oo_.dr" in res.details["error"] or "Missing decision-rule fields" in res.details["error"]


@pytest.mark.parametrize("bad_oo", [None, np.array([]), [], 1.0])
def test_empty_oo_is_unavailable_not_an_exception(bad_oo):
    m1 = load_mod(RBC_MOD, order=1)
    res = verify_dynare_parity(m1, {"oo_": bad_oo}, order=1)
    assert res.details["status"] == "UNAVAILABLE"
    assert not res.passed


def test_empty_dr_from_scipy_roundtrip(tmp_path):
    """The exact object scipy.io.loadmat produces for a Dynare results file with oo_.dr = []."""
    path = tmp_path / "never_simulated_results.mat"
    scipy.io.savemat(str(path), {"oo_": {"dr": np.zeros((0, 0)), "steady_state": np.array([1.0])}})
    raw = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    res = verify_dynare_parity(load_mod(RBC_MOD, order=1), raw, order=1)
    assert res.details["status"] == "UNAVAILABLE"
    assert not res.passed


def test_run_parity_suite_reports_empty_dr_as_unavailable():
    """The batch runner labels the empty reference UNAVAILABLE rather than an opaque FAIL."""
    suite = run_parity_suite(LIVE, dynare_results={"rbc": {"oo_": {"dr": np.array([])}}},
                             pattern="rbc.mod", order=1)
    assert suite.results[0].status == "UNAVAILABLE"
    assert not suite.passed


# ---------------------------------------------------------------------------
# Markov-switching DSGE: scale-aware convergence gate
# ---------------------------------------------------------------------------

def _reviewer_ms_model():
    n = 3
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    A = [np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.0, 0.0, 0.3]]),
         np.array([[0.6, 0.0, 0.1], [0.1, 0.3, 0.0], [0.0, 0.1, 0.2]])]
    B = [-np.eye(n), -np.eye(n)]
    C = [np.diag([0.5, 0.4, 0.3]), np.diag([0.3, 0.5, 0.2])]
    D = [np.eye(n), 0.5 * np.eye(n)]
    return A, B, C, D, P


def _scaled(mats, scale):
    return [scale * m for m in mats]


@pytest.mark.parametrize("method", ["newton", "functional_iteration"])
@pytest.mark.parametrize("scale", [1e7, 1e9])
def test_ms_dsge_convergence_gate_is_scale_aware(method, scale):
    """Rescaling every structural matrix leaves T unchanged and must not null the outputs.

    4.3.0 reported converged=False with all-NaN R, moments and stability at
    scale 1e7 (residual 9.3e-10 > tol=1e-10, pure cancellation noise).
    """
    A, B, C, D, P = _reviewer_ms_model()
    ref = solve_ms_dsge(A, B, C, D, P, method=method)
    # O(1) model: every residual entry meets tol, so the 4.3.0 absolute gate alone
    # certifies it and no relative slack is needed.
    assert ref.converged and ref.diff <= 1e-10
    assert ref.relative_residual == 0.0
    res = solve_ms_dsge(_scaled(A, scale), _scaled(B, scale), _scaled(C, scale), _scaled(D, scale), P, method=method)
    assert res.converged
    assert 0.0 < res.relative_residual <= MS_RTOL  # entries above tol are rounding noise of their own terms
    assert res.diff > 1e-10  # the absolute residual really is above the absolute tolerance
    for i in range(2):
        np.testing.assert_allclose(res.T[i], ref.T[i], atol=1e-8, rtol=0)
        np.testing.assert_allclose(res.R[i], ref.R[i], atol=1e-8, rtol=0)
    assert res.mean_square_stable
    assert "relative=" in res.summary()
    assert res.spectral_radius_mss == pytest.approx(ref.spectral_radius_mss, abs=1e-8)
    np.testing.assert_allclose(res.ergodic_mean.to_numpy(), ref.ergodic_mean.to_numpy(), atol=1e-8, rtol=0)
    # ergodic covariances are O(1..10) and amplify the O(1e-11) functional-iteration
    # stopping error of the unscaled reference: compare relatively.
    np.testing.assert_allclose(res.ergodic_cov.to_numpy(), ref.ergodic_cov.to_numpy(), atol=1e-10, rtol=1e-7)


MS_RTOL = 64.0 * np.finfo(float).eps


def _block_ms_model():
    """Variable 0 obeys a purely backward-looking equation (row 0 of A is zero);
    variables 1-2 form a slowly contracting forward-looking block (review probe)."""
    n = 3
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    A = [np.array([[0.0, 0.0, 0.0], [0.0, 0.9, 0.05], [0.0, 0.05, 0.85]]),
         np.array([[0.0, 0.0, 0.0], [0.0, 0.85, 0.0], [0.0, 0.1, 0.9]])]
    B = [-np.eye(n), -np.eye(n)]
    B[0][1, 1], B[0][2, 2], B[1][1, 1], B[1][2, 2] = -1.9, -1.85, -1.9, -1.95
    C = [np.array([[0.5, 0, 0], [0.2, 0.95, 0.0], [0.0, 0.0, 0.9]]),
         np.array([[0.5, 0, 0], [0.1, 0.9, 0.0], [0.0, 0.05, 0.95]])]
    D = [np.eye(n), 0.5 * np.eye(n)]
    return A, B, C, D, P


def _fast_slow_ms_model():
    """Variable 0 is a fast block of its own, variables 1-2 converge slowly (review probe)."""
    n = 3
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    A = [np.array([[0.1, 0.0, 0.0], [0.0, 0.45, 0.1], [0.0, 0.05, 0.45]]),
         np.array([[0.1, 0.0, 0.0], [0.0, 0.40, 0.1], [0.0, 0.10, 0.45]])]
    B = [-np.eye(n), -np.eye(n)]
    C = [np.diag([0.2, 0.45, 0.45]), np.diag([0.2, 0.5, 0.4])]
    D = [np.eye(n), 0.5 * np.eye(n)]
    return A, B, C, D, P


def _row_scaled(mats, row, factor):
    S = np.eye(mats[0].shape[0])
    S[row, row] = factor
    return [S @ m for m in mats]


@pytest.mark.parametrize("method", ["newton", "functional_iteration"])
@pytest.mark.parametrize("big", [1e6, 1e9])
@pytest.mark.parametrize("model", [_block_ms_model, _fast_slow_ms_model, _reviewer_ms_model])
def test_ms_dsge_large_unit_equation_does_not_loosen_the_others(model, big, method):
    """One equation in large units (row scaling leaves T unchanged) must not relax the O(1) rows.

    The first fix compared max|F| with rtol * max(terms) over the whole system:
    on the block model with row 0 scaled by 1e9/1e10 it returned converged=True
    with T wrong by 1.1e-3 / 1.9e-2 (4.3.0: 0.0). The entrywise gate accepts an
    entry above tol only at the rounding floor of its own terms.
    """
    A, B, C, D, P = model()
    ref = solve_ms_dsge(A, B, C, D, P, method="newton", tol=1e-14, rtol=0.0)
    assert ref.diff <= 1e-13
    tol = 1e-10
    As, Bs, Cs, Ds = (_row_scaled(M, 0, big) for M in (A, B, C, D))
    res = solve_ms_dsge(As, Bs, Cs, Ds, P, method=method, tol=tol)
    assert res.converged
    assert max(float(np.max(np.abs(res.T[i] - ref.T[i]))) for i in range(2)) <= 1e-9
    for i in range(2):
        F = (As[i] @ sum(P[i, k] * res.T[k] for k in range(2)) + Bs[i]) @ res.T[i] + Cs[i]
        assert float(np.max(np.abs(F[1:]))) <= tol  # the O(1) equations still meet tol
    assert res.relative_residual <= MS_RTOL
    assert not np.isnan(res.ergodic_mean.to_numpy()).any()


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")  # units spanning 1e14 make J badly scaled
@pytest.mark.parametrize("method", ["newton", "functional_iteration"])
@pytest.mark.parametrize("units", [(1e6, 1.0, 1.0), (1.0, 1e-6, 1e8)])
def test_ms_dsge_gate_is_invariant_to_variable_units(units, method):
    """Variables in mixed units (y = D x, column scaling) solve to the same x-unit rule.

    4.3.0 failed functional iteration on units (1, 1e-6, 1e8) (converged=False after 1000 iterations).
    """
    A, B, C, D, P = _reviewer_ms_model()
    ref = solve_ms_dsge(A, B, C, D, P, method="newton", tol=1e-14, rtol=0.0)
    Dm = np.diag(units)
    Dinv = np.diag(1.0 / np.asarray(units))
    res = solve_ms_dsge([a @ Dm for a in A], [b @ Dm for b in B], [c @ Dm for c in C], D, P, method=method)
    assert res.converged
    for i in range(2):  # T_y = D^{-1} T_x D
        np.testing.assert_allclose(Dm @ res.T[i] @ Dinv, ref.T[i], atol=1e-9, rtol=0)


def _nk_regime_switching_model():
    """The notebook-46 New Keynesian model (determinate and passive policy regimes), O(1) coefficients."""
    beta, sigma, kappa, rho_i, phi_x = 0.99, 1.0, 0.10, 0.80, 0.10
    P = np.array([[0.90, 0.10], [0.20, 0.80]])
    A, B, C, D = [], [], [], []
    for phi_pi in (1.80, 0.80):
        A.append(np.array([[1.0, 1.0 / sigma, 0.0], [0.0, beta, 0.0], [0.0, 0.0, 0.0]]))
        B.append(np.array([[-1.0, 0.0, -1.0 / sigma], [kappa, -1.0, 0.0],
                           [(1 - rho_i) * phi_x, (1 - rho_i) * phi_pi, -1.0]]))
        C.append(np.array([[0.0, 0, 0], [0, 0, 0], [0, 0, rho_i]]))
        D.append(np.eye(3))
    return A, B, C, D, P


@pytest.mark.parametrize("method", ["newton", "functional_iteration"])
@pytest.mark.parametrize("tol", [1e-10, 1e-12, 1e-13])
def test_ms_dsge_user_tol_is_honoured_on_order_one_models(tol, method):
    """A user-tightened tol is not loosened by the relative branch on an O(1) model.

    The first fix stopped functional iteration on this model at diff 4.2e-12 with tol=1e-12.
    """
    A, B, C, D, P = _nk_regime_switching_model()
    res = solve_ms_dsge(A, B, C, D, P, method=method, tol=tol)
    absolute = solve_ms_dsge(A, B, C, D, P, method=method, tol=tol, rtol=0.0)
    assert res.converged and res.diff <= tol
    assert absolute.converged and absolute.diff <= tol
    assert res.relative_residual == 0.0
    # Same code path, but threaded BLAS is not bit-deterministic: at tol=1e-13 noise in
    # the 1e-21 entries can cost one extra iteration (seen on macOS CI).
    assert abs(res.iterations - absolute.iterations) <= 1
    for i in range(2):
        np.testing.assert_allclose(res.T[i], absolute.T[i], rtol=0, atol=1e-10)


@pytest.mark.parametrize("method", ["newton", "functional_iteration"])
@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0, 1e3])
def test_ms_dsge_order_one_scales_are_bit_identical_to_the_absolute_gate(scale, method):
    """While every term magnitude stays below tol / rtol (about 7e3) the default gate is the 4.3.0 gate."""
    A, B, C, D, P = _reviewer_ms_model()
    args = (_scaled(A, scale), _scaled(B, scale), _scaled(C, scale), _scaled(D, scale), P)
    res = solve_ms_dsge(*args, method=method)
    absolute = solve_ms_dsge(*args, method=method, rtol=0.0)
    assert res.converged and absolute.converged
    assert res.iterations == absolute.iterations and res.diff == absolute.diff
    for i in range(2):
        np.testing.assert_array_equal(res.T[i], absolute.T[i])


def test_ms_dsge_newton_reaches_the_rounding_floor_before_the_stall_exit():
    """Newton's 1e-12 step-stall exit fired at relative residual 1.5e-14 (just above rtol) on this
    10-variable model at scale 1e9; up to two more such steps now reach 2.4e-16 and certify it."""
    rng = np.random.default_rng(3)
    n = 10
    A = [0.3 * rng.standard_normal((n, n)) / np.sqrt(n) for _ in range(2)]
    B = [-np.eye(n) + 0.1 * rng.standard_normal((n, n)) / np.sqrt(n) for _ in range(2)]
    C = [0.4 * rng.standard_normal((n, n)) / np.sqrt(n) for _ in range(2)]
    D = [np.eye(n), np.eye(n)]
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    ref = solve_ms_dsge(A, B, C, D, P, method="newton")
    assert ref.converged
    res = solve_ms_dsge(_scaled(A, 1e9), _scaled(B, 1e9), _scaled(C, 1e9), _scaled(D, 1e9), P, method="newton")
    assert res.converged and res.relative_residual <= MS_RTOL
    for i in range(2):
        np.testing.assert_allclose(res.T[i], ref.T[i], atol=1e-10, rtol=0)


def test_ms_dsge_rtol_zero_restores_absolute_gate():
    A, B, C, D, P = _reviewer_ms_model()
    res = solve_ms_dsge(_scaled(A, 1e7), _scaled(B, 1e7), _scaled(C, 1e7), _scaled(D, 1e7), P, rtol=0.0)
    assert not res.converged
    assert np.isnan(res.ergodic_mean.to_numpy()).all()
    assert not res.mean_square_stable
    with pytest.raises(ValueError, match="rtol"):
        solve_ms_dsge(A, B, C, D, P, rtol=-1.0)


def test_ms_dsge_non_solution_still_fails_with_relative_gate():
    """t^2 - 3t + 2.5 = 0 has no real root: the relative gate must not certify the stall."""
    res = solve_ms_dsge(A=[np.array([[1.0]])], B=[np.array([[-3.0]])], C=[np.array([[2.5]])],
                        D=[np.array([[1.0]])], transition_matrix=np.ones((1, 1)), max_iter=10)
    assert not res.converged and not res.mean_square_stable
    assert np.isnan(res.spectral_radius_mss)
    assert res.relative_residual > 1e-3
    t = res.T[0][0, 0]
    assert abs(res.diff - abs(t * t - 3 * t + 2.5)) < 1e-10


# ---------------------------------------------------------------------------
# Collocation: one CRRA curvature for policy and value
# ---------------------------------------------------------------------------

def test_crra_curvature_resolution():
    assert _crra_curvature({}) == 1.0
    assert _crra_curvature(None) == 1.0
    assert _crra_curvature({"gamma": 3.0}) == 3.0
    assert _crra_curvature({"sigma": 2.0}) == 2.0
    assert _crra_curvature({"sigma": 2.0, "gamma": 3.0}) == 2.0  # sigma wins
    with pytest.raises(TypeError, match="scalar"):
        _crra_curvature({"sigma": np.array([0.01, 0.02])})


def test_collocation_custom_return_fn_params_are_not_read_as_curvature():
    """A user return_fn may carry a non-scalar 'sigma' (e.g. shock stds): the built-in
    curvature is not resolved on that path (the first fix raised TypeError here)."""
    def ret(kp, k, sigma=None, **_):
        return np.log(np.maximum(k ** 0.36 - kp, 1e-12))

    def solve(params):
        return CollocationProblem(domain=(0.05, 0.4), orders=5, beta=0.96, method="bellman", return_fn=ret,
                                  params=params, options={"max_iter": 30}).solve()

    vector = solve({"sigma": np.array([0.01, 0.02])})
    scalar = solve({"sigma": 5.0})
    np.testing.assert_array_equal(vector.coefficients, scalar.coefficients)


def test_ift_gradient_wrt_gamma_is_the_crra_sensitivity():
    """IFT differentiates the Euler residual: with the curvature spelled 'gamma' the column is now
    the true CRRA sensitivity (4.3.0: exactly 0), equal to the 'sigma' column and to an FD re-solve."""
    from dataclasses import replace

    from puremacro.vfi.analytic_gradients import compute_ift_gradients

    alpha, beta = 0.36, 0.96
    k = (alpha * beta) ** (1.0 / (1.0 - alpha))

    def problem(key, value):
        return CollocationProblem(domain=(0.5 * k, 1.5 * k), orders=7, beta=beta,
                                  params={"alpha": alpha, "delta": 1.0, key: value}, options={"tol": 1e-12})

    prob_g, prob_s = problem("gamma", 1.0), problem("sigma", 1.0)
    grad_g = compute_ift_gradients(prob_g.solve(), prob_g, params=["alpha", "gamma"]).grad_coefficients
    grad_s = compute_ift_gradients(prob_s.solve(), prob_s, params=["alpha", "sigma"]).grad_coefficients
    np.testing.assert_allclose(grad_g, grad_s, atol=1e-10, rtol=0)
    assert np.max(np.abs(grad_g[:, 1])) > 1e-2  # measured 1.65e-2
    h = 1e-4
    c_plus = replace(prob_g, params={"alpha": alpha, "delta": 1.0, "gamma": 1.0 + h}).solve().coefficients
    c_minus = replace(prob_g, params={"alpha": alpha, "delta": 1.0, "gamma": 1.0 - h}).solve().coefficients
    fd = (np.asarray(c_plus) - np.asarray(c_minus)) / (2 * h)
    assert np.max(np.abs(grad_g[:, 1] - fd)) <= 1e-6 * np.max(np.abs(fd))  # measured 2.5e-9


def _growth_problem(params, method="euler", orders=8):
    return CollocationProblem(beta=0.96, params={"alpha": 0.36, "delta": 0.1, **params},
                              domain=[(1.0, 6.0)], orders=[orders], method=method)


def test_collocation_gamma_alias_matches_sigma():
    """{'gamma': 3} must solve the CRRA(3) problem, not a log policy valued under CRRA(3)."""
    sol_gamma = _growth_problem({"gamma": 3.0}).solve()
    sol_sigma = _growth_problem({"sigma": 3.0}).solve()
    sol_log = _growth_problem({"sigma": 1.0}).solve()
    assert sol_gamma.converged and sol_sigma.converged and sol_log.converged
    np.testing.assert_allclose(sol_gamma.metadata["policy_coefficients"],
                               sol_sigma.metadata["policy_coefficients"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(sol_gamma.metadata["value_coefficients"],
                               sol_sigma.metadata["value_coefficients"], atol=1e-8, rtol=0)
    # and it is a different policy from the log one (4.3.0 returned the log policy)
    assert np.max(np.abs(sol_gamma.metadata["policy_coefficients"]
                         - sol_log.metadata["policy_coefficients"])) > 1e-2
    # The Euler residual evaluated under CRRA(3) is small at the returned policy
    prob3 = _growth_problem({"sigma": 3.0})
    basis = sol_gamma.basis
    nodes = basis.nodes()
    resid = _evaluate_euler_residual(prob3, basis, sol_gamma.metadata["policy_coefficients"], nodes, "numpy")
    assert np.max(np.abs(resid)) <= 1e-8
    # sigma wins when both are supplied
    sol_both = _growth_problem({"sigma": 3.0, "gamma": 1.0}).solve()
    np.testing.assert_allclose(sol_both.metadata["policy_coefficients"],
                               sol_sigma.metadata["policy_coefficients"], atol=1e-10, rtol=0)


@pytest.mark.parametrize("params", [{"sigma": 1.0}, {"gamma": 1.0}, {}])
def test_collocation_log_growth_closed_form_under_every_spelling(params):
    """Deterministic log-growth fixture (delta = 1): policy alpha*beta*k^alpha, value log(c)/(1-beta)."""
    alpha, beta = 0.36, 0.96
    k = (alpha * beta) ** (1.0 / (1.0 - alpha))
    sol = CollocationProblem(domain=(0.5 * k, 1.5 * k), orders=13, beta=beta,
                             params={"alpha": alpha, "delta": 1.0, **params}).solve()
    assert sol.converged
    grid = np.linspace(0.6 * k, 1.4 * k, 21)
    pol = np.array([float(sol.policy(x)) for x in grid])
    np.testing.assert_allclose(pol, alpha * beta * grid ** alpha, atol=1e-6, rtol=0)
    assert abs(sol.value(k) - np.log(k ** alpha - k) / (1.0 - beta)) < 1e-5


@pytest.mark.parametrize("params", [{"sigma": 2.0}, {"gamma": 3.0}])
def test_collocation_value_satisfies_bellman_identity_under_crra(params):
    """V(k) = u(c) + beta V(k') at the nodes with u the same CRRA utility the policy solves."""
    alpha, beta, delta = 0.36, 0.96, 0.1
    sigma = _crra_curvature(params)
    sol = _growth_problem(params, orders=10).solve()
    assert sol.converged
    nodes = sol.metadata["nodes"][:, 0]
    kp = np.array([float(sol.policy(x)) for x in nodes])
    c = nodes ** alpha + (1.0 - delta) * nodes - kp
    u = np.log(c) if sigma == 1.0 else (c ** (1.0 - sigma) - 1.0) / (1.0 - sigma)
    lhs = np.array([float(sol.value(x)) for x in nodes])
    rhs = u + beta * np.array([float(sol.value(x)) for x in kp])
    np.testing.assert_allclose(lhs, rhs, atol=1e-8, rtol=0)
    # A log-utility valuation of the same policy is not a fixed point of this identity
    u_log = np.log(c)
    assert np.max(np.abs(lhs - (u_log + beta * np.array([float(sol.value(x)) for x in kp])))) > 1e-2


def test_collocation_bellman_path_uses_the_same_curvature():
    """The Bellman method solves the CRRA(sigma) problem the Euler method solves."""
    alpha, beta, delta, sigma = 0.36, 0.96, 1.0, 2.0
    k = (alpha * beta) ** (1.0 / (1.0 - alpha))
    common = dict(domain=(0.5 * k, 1.5 * k), orders=10, beta=beta,
                  params={"alpha": alpha, "delta": delta, "sigma": sigma})
    euler = CollocationProblem(method="euler", options={"tol": 1e-10}, **common).solve()
    bellman = CollocationProblem(method="bellman", options={"tol": 1e-9, "max_iter": 800}, **common).solve()
    assert euler.converged and bellman.converged
    grid = np.linspace(0.6 * k, 1.4 * k, 25)
    pol_e = np.array([float(euler.policy(x)) for x in grid])
    pol_b = np.array([float(bellman.policy(x)) for x in grid])
    pol_log = alpha * beta * grid ** alpha  # the sigma = 1 policy the Bellman path used to return
    # Measured: |Bellman - Euler| = 1.1e-6, |Bellman - log policy| = 9.6e-3 (4.3.0: 0).
    assert np.max(np.abs(pol_e - pol_b)) < BELLMAN_EULER_TOL
    assert np.max(np.abs(pol_b - pol_log)) > 10 * BELLMAN_EULER_TOL


BELLMAN_EULER_TOL = 1e-4


# ---------------------------------------------------------------------------
# Documentation scope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["docs/vfi_analytic_gradients.md", "docs/es/vfi_analytic_gradients.md"])
def test_analytic_gradients_docs_state_current_scope(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    assert "NotImplementedError" in text
    solution_line = next(line for line in text.splitlines() if line.startswith("- `solution`:"))
    assert "NotImplementedError" in solution_line
    aggregates_line = next(line for line in text.splitlines() if line.startswith("- `grad_aggregates`:"))
    assert '"mu": None' in aggregates_line
    assert "70x" not in text
    # J_c and J_theta are central differences, also for a user residual_fn:
    # the page must not call the IFT path step-size free or chatter free.
    assert "step_c" in text and "`h`" in text
    for overclaim in ("step-size free", "Zero chatter", "libre de tamaño de paso", "Cero ruido",
                      "residual_fn` can be supplied instead", "residual_fn` analítica en su lugar",
                      "Machine precision", "Precisión de máquina", "Exact Analytic", "analíticos exactos",
                      "analítico exacto", "linear solve is exact", "lineal del TFI es exacta"):
        assert overclaim not in text, overclaim
