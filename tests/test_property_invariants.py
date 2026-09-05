"""Statistical invariant and property-based tests for econometric estimators.

Verifies mathematical conservation laws, asymptotic decay, Loewner ordering,
and algebraic identities across randomly generated DGPs.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.var.estimate import estimate_var, companion
from puremacro.var.irf import irf, fevd, historical_decomp
from puremacro.var.identify.cholesky import cholesky_factor
from puremacro.state_space import StateSpaceModel, kalman_filter, kalman_smoother


@pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("p", [1, 2])
def test_var_companion_stability_implies_irf_decay(seed: int, n: int, p: int):
    """Property: If companion spectral radius < 1, IRF must converge to zero as h -> inf."""
    rng = np.random.default_rng(seed)
    max_eig = 2.0
    A_list = []
    while max_eig >= 0.85:
        A_list = [rng.standard_normal((n, n)) * 0.2 / p for _ in range(p)]
        Cc = companion(A_list)
        max_eig = np.abs(np.linalg.eigvals(Cc)).max()

    B0 = np.eye(n)
    horizon = 80
    irfs = irf(A_list, B0, horizon=horizon)  # shape (H+1, n, n)

    # Initial impact must be B0
    assert np.allclose(irfs[0], B0, atol=1e-12)

    # Decay: magnitude at horizon 80 must be negligible (< 1e-3) and strictly smaller than peak
    tail_norm = np.linalg.norm(irfs[-1])
    peak_norm = max(np.linalg.norm(irfs[h]) for h in range(min(5, horizon)))
    assert tail_norm < 1e-3
    assert tail_norm < peak_norm


@pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_cholesky_covariance_factorization_exact_identity(seed: int, n: int):
    """Property: B0 = cholesky_factor(Sigma) satisfies B0 @ B0.T == Sigma to machine precision."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T + np.eye(n) * 0.1

    B0 = cholesky_factor(Sigma)
    # Must be lower triangular
    assert np.allclose(np.triu(B0, k=1), 0.0)
    # Diagonal must be positive
    assert (np.diag(B0) > 0.0).all()
    # Factorization identity
    reconstructed = B0 @ B0.T
    diff = np.max(np.abs(reconstructed - Sigma))
    assert diff < 1e-13, f"Cholesky identity violated: max diff {diff}"


@pytest.mark.parametrize("seed", [71, 72, 73, 74])
@pytest.mark.parametrize("n", [2, 3])
def test_fevd_row_sums_to_unity_at_all_horizons(seed: int, n: int):
    """Property: FEVD shares across all orthogonal shocks must sum to 1.0 for each variable."""
    rng = np.random.default_rng(seed)
    p = 2
    max_eig = 2.0
    while max_eig >= 0.85:
        A_list = [rng.standard_normal((n, n)) * 0.15 for _ in range(p)]
        Cc = companion(A_list)
        max_eig = np.abs(np.linalg.eigvals(Cc)).max()

    A = rng.standard_normal((n, n))
    Sigma = A @ A.T + np.eye(n) * 0.1
    B0 = np.linalg.cholesky(Sigma)

    horizon = 20
    f = fevd(A_list, B0, horizon=horizon)  # shape (H+1, n, n)

    # Across all horizons h and all variables i, sum over shocks j must equal 1.0
    row_sums = f.sum(axis=2)  # shape (H+1, n)
    assert np.allclose(row_sums, 1.0, atol=1e-10)

    # Every element must be a valid probability share in [0, 1]
    assert (f >= -1e-12).all()
    assert (f <= 1.0 + 1e-12).all()


@pytest.mark.parametrize("seed", [81, 82, 83])
def test_kalman_filter_and_smoother_uncertainty_reduction(seed: int):
    """Property: In Loewner order:
    1. Filtered variance P_{t|t} <= Predicted variance P_{t|t-1}
    2. Smoothed variance P_{t|T} <= Filtered variance P_{t|t}
    """
    rng = np.random.default_rng(seed)
    T_obs = 60
    m = 2  # state dim
    n = 2  # obs dim

    Tm = np.array([[0.8, 0.1], [0.0, 0.7]])
    Z = np.array([[1.0, 0.5], [0.2, 1.0]])
    Q = np.eye(m) * 0.2
    H = np.eye(n) * 0.5

    model = StateSpaceModel(T=Tm, Z=Z, Q=Q, H=H)

    alpha = np.zeros((T_obs, m))
    y = np.zeros((T_obs, n))
    curr_a = rng.standard_normal(m)
    for t in range(T_obs):
        curr_a = Tm @ curr_a + rng.multivariate_normal(np.zeros(m), Q)
        alpha[t] = curr_a
        y[t] = Z @ curr_a + rng.multivariate_normal(np.zeros(n), H)

    res = kalman_smoother(y, model)
    P_pred = res["P_pred"]  # (T_obs+1, m, m)
    P_filt = res["P_filt"]  # (T_obs, m, m)
    P_sm = res["P_smooth"]  # (T_obs, m, m)

    for t in range(10, T_obs - 5):
        # 1. Prediction minus Filter is PSD: P_{t|t-1} - P_{t|t} >= 0
        diff_pred_filt = P_pred[t] - P_filt[t]
        eigs1 = np.linalg.eigvalsh(diff_pred_filt)
        assert (eigs1 >= -1e-10).all(), f"P_pred - P_filt not PSD at t={t}: min eig {eigs1.min()}"

        # 2. Filter minus Smoother is PSD: P_{t|t} - P_{t|T} >= 0
        diff_filt_sm = P_filt[t] - P_sm[t]
        eigs2 = np.linalg.eigvalsh(diff_filt_sm)
        assert (eigs2 >= -1e-10).all(), f"P_filt - P_smooth not PSD at t={t}: min eig {eigs2.min()}"


@pytest.mark.parametrize("seed", [91, 92, 93])
def test_historical_decomposition_reconstruction_property(seed: int):
    """Property: Sum of shock contributions + deterministic in historical decomposition equals observed data."""
    rng = np.random.default_rng(seed)
    T, n, p = 150, 2, 1
    A = np.array([[0.5, 0.1], [0.0, 0.4]])
    Sigma = np.array([[1.0, 0.2], [0.2, 0.8]])
    B0 = np.linalg.cholesky(Sigma)

    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + B0 @ rng.standard_normal(n)

    fit = estimate_var(Y, p=p)
    out = historical_decomp(fit.A_list, B0, fit.resid, init_y=Y[:p], intercept=fit.c)
    Y_eff = Y[p: p + out["shocks"].shape[0]]
    reconstructed = out["shocks"].sum(axis=2) + out["deterministic"]
    assert np.allclose(reconstructed, Y_eff, atol=1e-7)


@pytest.mark.parametrize("seed", [601, 602, 603])
def test_unit_root_adf_distinguishes_stationary_from_integrated(seed: int):
    """Property: Stationary process yields significantly more negative ADF stat than unit root."""
    from puremacro.tests.unit_root import adf_test

    rng = np.random.default_rng(seed)
    T = 250

    # 1. Stationary AR(1) with rho = 0.4
    y_stat = np.zeros(T)
    for t in range(1, T):
        y_stat[t] = 0.4 * y_stat[t - 1] + rng.standard_normal()
    res_stat = adf_test(y_stat)

    # 2. Random walk with rho = 1.0
    y_rw = np.cumsum(rng.standard_normal(T))
    res_rw = adf_test(y_rw)

    # Stationary series must have more negative t-statistic and smaller p-value
    assert res_stat["stat"] < res_rw["stat"]
    assert res_stat["stat"] < -2.8  # rejects at 5% for constant model
    assert res_stat["p_value"] < 0.05
    assert res_rw["p_value"] > 0.05


@pytest.mark.parametrize("rho", [0.2, 0.5, 0.7])
@pytest.mark.parametrize("a", [0.3, 0.5])
def test_dsge_klein_analytic_and_bk_invariant(rho: float, a: float):
    """Property: Klein QZ solution on hand-solvable forward model matches exact closed form:
    x_t = a E_t[x_{t+1}] + b z_t,  z_{t+1} = rho z_t + eps_t  =>  x_t = (b / (1 - a*rho)) z_t.
    """
    from puremacro.dsge.klein import klein_solve

    b = 0.8
    # Klein system: A E_t[z_{t+1}] = B z_t + C eps_t
    A = np.array([[1.0, 0.0], [0.0, a]])
    B = np.array([[rho, 0.0], [-b, 1.0]])
    C = np.array([[1.0], [0.0]])

    sol = klein_solve(A, B, 1, C)

    expected_F = b / (1.0 - a * rho)
    assert sol.eu == (1, 1), "Blanchard-Kahn condition must hold for stable forward model"
    assert np.isclose(sol.F[0, 0], expected_F, atol=1e-11)
    assert np.isclose(sol.G[0, 0], rho, atol=1e-11)
