"""Tests for var.identify result dataclasses."""
import dataclasses

import numpy as np
import pytest

from puremacro.var.identify._results import (
    MagMavSVARResult,
    ProxySVARResult,
    CholeskySVARResult,
    BQSVARResult,
    SignRestrictionResult,
    GKRobustBandsResult,
    NonGaussianSVARResult,
    SignZeroResult,
)


def test_proxy_svar_result_is_frozen():
    res = ProxySVARResult(
        irf_point=np.zeros((5, 3, 3)),
        irf_lower=np.zeros((5, 3, 3)),
        irf_upper=np.zeros((5, 3, 3)),
        B=np.eye(3),
        first_stage_F=25.0,
        n_boot=500,
        ci=0.9,
    )
    with pytest.raises(Exception):  # FrozenInstanceError or similar
        res.first_stage_F = 99.0


def test_proxy_svar_result_summary():
    res = ProxySVARResult(
        irf_point=np.zeros((5, 3, 3)),
        irf_lower=np.zeros((5, 3, 3)),
        irf_upper=np.zeros((5, 3, 3)),
        B=np.eye(3),
        first_stage_F=25.0,
        n_boot=500,
        ci=0.9,
    )
    s = res.summary()
    assert "ProxySVAR" in s
    assert "25.0" in s or "25.00" in s


def test_proxy_svar_result_picklable():
    import pickle
    res = ProxySVARResult(
        irf_point=np.zeros((3, 2, 2)),
        irf_lower=np.zeros((3, 2, 2)),
        irf_upper=np.zeros((3, 2, 2)),
        B=np.eye(2),
        first_stage_F=12.0,
        n_boot=100,
        ci=0.9,
    )
    blob = pickle.dumps(res)
    res2 = pickle.loads(blob)
    np.testing.assert_array_equal(res2.B, res.B)
    assert res2.first_stage_F == 12.0


# ----------------------------------------------------------------------
# CholeskySVARResult, BQSVARResult, SignRestrictionResult: end-to-end calls
# ----------------------------------------------------------------------


def _toy_var2_array():
    rng = np.random.default_rng(7)
    T, n = 200, 2
    A = np.array([[0.5, 0.1], [0.0, 0.6]])
    Sigma = np.array([[1.0, 0.2], [0.2, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_normal(n)
    return Y


def test_cholesky_svar_returns_dataclass_and_is_frozen():
    from puremacro.var.identify.cholesky import cholesky_svar

    Y = _toy_var2_array()
    res = cholesky_svar(Y, p=1, horizon=4, n_boot=20, ci=0.9, seed=0)
    assert isinstance(res, CholeskySVARResult)
    assert res.irf_point.shape == (5, 2, 2)
    assert res.irf_lower.shape == (5, 2, 2)
    assert res.irf_upper.shape == (5, 2, 2)
    assert res.n_boot == 20
    assert res.ci == 0.9
    with pytest.raises(Exception):
        res.n_boot = 99  # frozen


def test_bq_svar_returns_dataclass_and_is_frozen():
    from puremacro.var.identify.bq import bq_svar

    Y = _toy_var2_array()
    res = bq_svar(Y, p=1, horizon=4, n_boot=20, ci=0.9, seed=0)
    assert isinstance(res, BQSVARResult)
    assert res.irf_point.shape == (5, 2, 2)
    with pytest.raises(Exception):
        res.n_boot = 99


def test_sign_restriction_svar_returns_dataclass_and_is_frozen():
    from puremacro.var.identify.sign import sign_restriction_svar

    Y = _toy_var2_array()
    restrictions = {0: [+1, +1]}
    res = sign_restriction_svar(Y, p=1, horizon=4,
                                  restrictions=restrictions,
                                  n_draws=200, ci=0.9, seed=0)
    assert isinstance(res, SignRestrictionResult)
    assert res.n_accepted > 0
    assert res.irf_median.shape == (5, 2, 2)
    with pytest.raises(Exception):
        res.n_accepted = 99


def test_gk_robust_bands_returns_dataclass_and_is_frozen():
    from puremacro.var.identify.sign_robust import gk_robust_bands

    Y = _toy_var2_array()
    restrictions = {0: [+1, +1]}
    res = gk_robust_bands(Y, p=1, horizon=4,
                            restrictions=restrictions, shock_idx=0,
                            n_var_draws=8, n_q_per_draw=80, ci=0.9, seed=0)
    assert isinstance(res, GKRobustBandsResult)
    assert res.irf_lower.shape == (5, 2)
    assert res.irf_upper.shape == (5, 2)
    assert res.n_accepted_per_draw.shape == (8,)
    with pytest.raises(Exception):
        res.n_accepted_per_draw = np.zeros(8)


def test_non_gaussian_svar_returns_dataclass_and_is_frozen():
    from puremacro.var.identify.non_gaussian import non_gaussian_svar

    rng = np.random.default_rng(0)
    A = np.array([[0.5, 0.1], [0.05, 0.4]])
    B0 = np.array([[1.0, 0.5], [0.3, 1.0]])
    shocks = rng.laplace(scale=0.7, size=(400, 2))
    Y = np.zeros((400, 2))
    for t in range(1, 400):
        Y[t] = A @ Y[t - 1] + B0 @ shocks[t]
    res = non_gaussian_svar(Y, p=1, horizon=4, seed=0)
    assert isinstance(res, NonGaussianSVARResult)
    assert res.B0.shape == (2, 2)
    assert res.kurtosis.shape == (2,)
    assert res.irf.shape == (5, 2, 2)
    with pytest.raises(Exception):
        res.B0 = np.zeros((2, 2))


def test_sign_zero_returns_dataclass_with_success_flag():
    from puremacro.var.identify.sign_zero import sign_zero
    from puremacro.var import fit_var

    Y = _toy_var2_array()
    res = fit_var(Y, p=1)
    A_list, _, Sigma, _, _ = res
    out = sign_zero(A_list, Sigma,
                    zero_constraints=[(0, 1)],
                    sign_constraints={(1, 0): +1},
                    n_draws=100, rng=np.random.default_rng(0))
    assert isinstance(out, SignZeroResult)
    assert isinstance(out.success, bool)
    if out.success:
        assert out.B0 is not None and out.Q is not None
    with pytest.raises(Exception):
        out.success = False  # frozen


def test_sign_zero_failure_returns_none_fields():
    """When no admissible draw is found, success=False and B0/Q are None
    (replacing the prior `return None` behaviour)."""
    from puremacro.var.identify.sign_zero import sign_zero
    from puremacro.var import fit_var

    Y = _toy_var2_array()
    A_list, _, Sigma, _, _ = fit_var(Y, p=1)
    # Impossible: zero-out the entire column 0 of B0
    out = sign_zero(A_list, Sigma,
                    zero_constraints=[(0, 0), (1, 0)],
                    sign_constraints={(0, 0): +1},
                    n_draws=20, rng=np.random.default_rng(0))
    assert isinstance(out, SignZeroResult)
    if not out.success:
        assert out.B0 is None
        assert out.Q is None


# ----------------------------------------------------------------------
# HeteroResult must be frozen
# ----------------------------------------------------------------------


def test_hetero_result_is_frozen():
    from puremacro.var.identify.hetero import HeteroResult

    res = HeteroResult(
        B=np.eye(2),
        variance_ratios=np.array([1.0, 1.5]),
        irfs=np.zeros((3, 2, 2)),
        fevd=np.zeros((3, 2, 2)),
        lower=None,
        upper=None,
        point=np.zeros((3, 2, 2)),
    )
    with pytest.raises(Exception):
        res.B = np.zeros((2, 2))


# ----------------------------------------------------------------------
# Result-class summary() smoke tests — verify they produce non-empty strings
# ----------------------------------------------------------------------


def test_cholesky_svar_summary_smoke():
    res = CholeskySVARResult(
        irf_point=np.zeros((5, 2, 2)),
        irf_lower=np.zeros((5, 2, 2)),
        irf_upper=np.zeros((5, 2, 2)),
        n_boot=100,
        n_fail=3,
        ci=0.9,
    )
    s = res.summary()
    assert "Cholesky" in s


def test_non_gaussian_svar_summary_smoke():
    res = NonGaussianSVARResult(
        B0=np.eye(2),
        Q=np.eye(2),
        kurtosis=np.array([1.5, 0.5]),
        irf=np.zeros((5, 2, 2)),
        ordering_by_kurt=np.array([0, 1]),
    )
    s = res.summary()
    assert "Non-Gaussian" in s


def test_sign_zero_summary_distinguishes_success_and_failure():
    s_ok = SignZeroResult(
        success=True, B0=np.eye(2), Q=np.eye(2), n_draws_used=42,
    ).summary()
    s_bad = SignZeroResult(
        success=False, B0=None, Q=None, n_draws_used=1000,
    ).summary()
    assert "SUCCESS" in s_ok
    assert "FAILED" in s_bad


def test_gk_robust_bands_from_gibbs_returns_dataclass():
    """gk_robust_bands_from_gibbs shares GKRobustBandsResult with the
    bootstrap-fed sibling — verify the contract."""
    from puremacro.var.identify.sign_robust import gk_robust_bands_from_gibbs

    rng = np.random.default_rng(0)
    D, p, n = 6, 1, 2
    A_draws = np.zeros((D, p, n, n))
    for d in range(D):
        A_draws[d, 0] = np.array([[0.5, 0.1], [0.0, 0.6]]) + 0.01 * rng.standard_normal((n, n))
    Sigma_draws = np.tile(np.eye(n)[None], (D, 1, 1)) + 0.05 * rng.standard_normal((D, n, n))
    # Symmetrise
    for d in range(D):
        Sigma_draws[d] = 0.5 * (Sigma_draws[d] + Sigma_draws[d].T) + 0.5 * np.eye(n)
    restrictions = {0: [+1, +1]}
    res = gk_robust_bands_from_gibbs(
        A_draws, Sigma_draws,
        horizon=4, restrictions=restrictions,
        shock_idx=0, n_q_per_draw=80, ci=0.9, seed=0,
    )
    assert isinstance(res, GKRobustBandsResult)
    with pytest.raises(Exception):
        res.irf_lower = np.zeros_like(res.irf_lower)


def test_magmav_svar_result_dataclass_is_frozen_and_has_expected_fields():
    res = MagMavSVARResult(
        irf_point=np.zeros((4, 2, 2)),
        irf_lower=np.zeros((4, 2, 2)),
        irf_upper=np.zeros((4, 2, 2)),
        B=np.eye(2),
        variance_change_dates=(50, 120),
        k_breaks=2,
        n_boot=200,
        ci=0.9,
        eu=(1, 1),
        n_fail=0,
    )
    assert res.irf_point.shape == (4, 2, 2)
    assert res.k_breaks == 2
    assert res.eu == (1, 1)
    assert dataclasses.is_dataclass(res)
    # frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.k_breaks = 3
    # summary() smoke — success branch
    s_ok = res.summary()
    assert isinstance(s_ok, str)
    assert "OK" in s_ok
    # summary() smoke — failure branch
    res_fail = MagMavSVARResult(
        irf_point=np.zeros((4, 2, 2)),
        irf_lower=np.zeros((4, 2, 2)),
        irf_upper=np.zeros((4, 2, 2)),
        B=np.eye(2),
        variance_change_dates=(),
        k_breaks=0,
        n_boot=200,
        ci=0.9,
        eu=(0, 0),
        n_fail=5,
    )
    s_fail = res_fail.summary()
    assert isinstance(s_fail, str)
    assert "FAIL" in s_fail


def test_non_gaussian_svar_result_has_optional_diagnostic_fields():
    from puremacro.var.identify._results import NonGaussianSVARResult

    # Construct without the new fields — they default to None.
    res = NonGaussianSVARResult(
        B0=np.eye(2),
        Q=np.eye(2),
        kurtosis=np.zeros(2),
        irf=np.zeros((3, 2, 2)),
        ordering_by_kurt=np.arange(2),
    )
    assert res.lr_test is None
    assert res.consistency_check is None

    # Construct with the fields populated.
    res2 = NonGaussianSVARResult(
        B0=np.eye(2),
        Q=np.eye(2),
        kurtosis=np.zeros(2),
        irf=np.zeros((3, 2, 2)),
        ordering_by_kurt=np.arange(2),
        lr_test={"stat": 1.2, "df": 1, "p_value": 0.27},
        consistency_check={"max_abs_diff": 1e-12, "rms_diff": 1e-13, "passed": True},
    )
    assert res2.lr_test["df"] == 1
    assert res2.consistency_check["passed"] is True


def test_svar_results_plot_method():
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    res = CholeskySVARResult(
        irf_point=np.ones((6, 2, 2)) * 0.5,
        irf_lower=np.zeros((6, 2, 2)),
        irf_upper=np.ones((6, 2, 2)),
        n_boot=100,
        n_fail=0,
        ci=0.9,
    )
    fig = res.plot(target_idx=0, shock_idx=0)
    assert isinstance(fig, Figure)
    plt.close(fig)

    res_proxy = ProxySVARResult(
        irf_point=np.ones((6, 2, 2)) * 0.3,
        irf_lower=np.zeros((6, 2, 2)),
        irf_upper=np.ones((6, 2, 2)),
        B=np.eye(2),
        first_stage_F=35.0,
        n_boot=100,
        ci=0.9,
    )
    fig2 = res_proxy.plot(target_idx=1, shock_idx=0)
    assert isinstance(fig2, Figure)
    plt.close(fig2)

    from puremacro.var.identify.hetero import HeteroResult
    res_hetero = HeteroResult(
        B=np.eye(2),
        variance_ratios=np.ones(2),
        irfs=np.ones((6, 2, 2)) * 0.4,
        fevd=np.ones((6, 2, 2)),
        lower=np.zeros((6, 2, 2)),
        upper=np.ones((6, 2, 2)),
        point=np.ones((6, 2, 2)) * 0.4,
    )
    fig3 = res_hetero.plot(target_idx=0, shock_idx=1)
    assert isinstance(fig3, Figure)
    plt.close(fig3)
