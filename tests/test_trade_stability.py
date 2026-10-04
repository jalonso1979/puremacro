"""Reduced local-stability diagnostic: closed-form oracle, Schur/re-solve agreement and closure invariants.

Tolerances: closed-form reduced Jacobian atol 1e-8 (observed 2.3e-10 with the
default second-order step 1e-6 and 1.8e-13 with fd_order=4, step=1e-3);
Schur complement versus an independent fsolve re-solve of the fast block on
the two-country fixture 1e-8, and the built-in chord re-solve check 1e-8
(observed 1.1e-9 to 2.8e-9); homogeneity and value-weighted Walras identity
1e-8 (observed <= 2e-9, also at off-equilibrium points of the fast-cleared
manifold); reference invariance of the spectrum 1e-8 (observed 1.8e-10 under
world factor income units); adjustment speeds: eig(R) equals eig(diag(d) A)
without its zero to 1e-8 (observed <= 2.2e-10); mode loadings are right
eigenvectors modulo the constant vector to 1e-8 (observed 5e-16); unit
invariance under flows x1e6 1e-8; legacy versus consistent agreement 1e-8
(observed 1.3e-10).
"""
from dataclasses import replace
import time
import warnings

import numpy as np
import pytest
from scipy.optimize import fsolve

from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium, solve_policy_equilibrium
from puremacro.trade._accounting import evaluate
from puremacro.trade.stability import QUALIFICATION, StabilityError, StabilityResult, reduced_stability

TOL = 1e-8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def toy_table(balance):
    """Two countries, one sector, no intermediates, no taxes: a closed-form reduced Jacobian."""
    ns, nc, nfd = 1, 2, 2
    n = ns*nc
    data = np.zeros((n+3, n+nfd*nc))
    y0 = np.array([100., 150.])
    alpha = np.array([.3, .4])
    data[3, :2] = (1-alpha)*y0
    data[4, :2] = alpha*y0
    # Rows sum to y0; with balance=True destination spending is (110, 140), so B = (-10, +10).
    F = np.array([[60., 40.], [50., 100.]]) if balance else np.array([[60., 40.], [40., 110.]])
    shares = np.array([.7, .3])
    for c in range(nc):
        data[:2, n+c*nfd:n+(c+1)*nfd] = F[:, c][:, None]*shares[None, :]
    return data, y0, alpha, F


def closed_form_reduced_jacobian(y0, alpha, F, B0, units, ref):
    """d e / d log(r_0, r_1, w_0, w_1) at the benchmark for the toy, rows (K_0, K_1, L_0, L_1).

    p_k = (r_k/alpha_k)^alpha_k (w_k/(1-alpha_k))^(1-alpha_k)/beta_k, spending
    S_k = w_k L_k + r_k K_k - B_k, Leontief baskets Q_k = sum_i afd_ik p_i,
    y_i = sum_k afd_ik S_k/Q_k, e_K,k = alpha_k p_k y_k/(r_k K_k) - 1,
    e_L,k = (1-alpha_k) p_k y_k/(w_k L_k) - 1, with B_k re-denominated per ``units``.
    """
    nc = 2
    K, L = alpha*y0, (1-alpha)*y0
    afd = F/F.sum(0)[None, :]
    omega = float(L.sum()+K.sum())
    dB = np.zeros((nc, 2*nc))
    if units == "world_factor_income":
        dB[:, :nc] = B0[:, None]*K[None, :]/omega
        dB[:, nc:] = B0[:, None]*L[None, :]/omega
    elif units == "reference_factor_price":
        dB[:, ref] = B0
    A = np.zeros((2*nc, 2*nc))
    S = y0-B0
    for xi in range(2*nc):
        m, is_r = xi % nc, xi < nc
        dlogp = np.zeros(nc)
        dlogp[m] = alpha[m] if is_r else 1-alpha[m]
        dS = np.zeros(nc)
        dS[m] = K[m] if is_r else L[m]
        dS = dS-dB[:, xi]
        dQ = afd[m, :]*dlogp[m]
        dy = afd@(dS-S*dQ)
        dv = dlogp*y0+dy
        for k in range(nc):
            A[k, xi] = alpha[k]/K[k]*dv[k]-(1. if (is_r and m == k) else 0.)
            A[nc+k, xi] = (1-alpha[k])/L[k]*dv[k]-(1. if ((not is_r) and m == k) else 0.)
    return A


def synthetic_2c_2s(labour_share=(.6, .75, .55, .8), tax_rate=.05, fd_tax=.02):
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns*nc+3, ns*nc+nfd*nc))
    data[:4, :4] = np.array([[10., 15., 5., 5.], [15., 20., 10., 10.], [5., 5., 12., 18.], [10., 10., 18., 22.]])
    y = np.array([100., 150., 120., 180.])
    va = y-data[:4, :4].sum(0)
    taxes = tax_rate*y
    vaf = va-taxes
    ls = np.asarray(labour_share, float)
    data[4, :4] = taxes
    data[5, :4] = ls*vaf
    data[6, :4] = (1-ls)*vaf
    fd = y-data[:4, :4].sum(1)
    for i in range(4):
        sh = [.5, .25, .05, .1, .08, .02] if i < 2 else [.1, .08, .02, .5, .25, .05]
        data[i, 4:] = fd[i]*np.array(sh)
    data[4, 4:] = fd_tax*data[:4, 4:].sum(0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, country_codes=["AAA", "BBB"], validate=True)


def synthetic_3c_3s(seed=42):
    nc, ns, nfd = 3, 3, 3
    rng = np.random.default_rng(seed)
    M = ns*nc
    data = np.zeros((M+3, M+nfd*nc))
    Z = rng.uniform(1., 10., (M, M))
    for c in range(nc):
        Z[c*ns:(c+1)*ns, c*ns:(c+1)*ns] *= 3.
    data[:M, :M] = Z
    col, row = Z.sum(0), Z.sum(1)
    y = np.maximum(col, row)*2.+rng.uniform(20., 60., M)
    taxes = .05*y
    vaf = y-col-taxes
    share = rng.uniform(.55, .8, M)
    data[M, :M] = taxes
    data[M+1, :M] = share*vaf
    data[M+2, :M] = (1-share)*vaf
    fdt = y-row
    for i in range(M):
        oc = i//ns
        sh = rng.uniform(.5, 1.5, nfd*nc)
        sh[oc*nfd:(oc+1)*nfd] *= 4.
        sh /= sh.sum()
        data[i, M:] = fdt[i]*sh
    data[M, M:] = .02*data[:M, M:].sum(0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


def tariff_schedules(calib, rate, importer=0):
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    n = nc*ns
    tau = np.ones((n, ns, nc))
    tau[:, :, importer] = 1+rate
    tau[importer*ns:(importer+1)*ns, :, importer] = 1.
    tfd = np.ones((n, nfd, nc))
    tfd[:, :, importer] = 1+rate
    tfd[importer*ns:(importer+1)*ns, :, importer] = 1.
    return tau, tfd


def solve_consistent(calib, **kwargs):
    options = dict(accounting="consistent", tol=1e-11, max_iter=100)
    options.update(kwargs)
    return solve_trade_equilibrium(calib, **options)


def fast_cleared_map(calib, base):
    """Independent fast-cleared map of a consistent result: fsolve on (goods, prices, budgets) at given (log r, log w).

    Foreign balances are held in world-factor-income units of the base state, as
    in the module's default closure. Returns ``state(z)`` (the full unknown vector
    on the fast-cleared manifold) and ``excess(z)`` (excess-demand ratios ordered
    capital then labour).
    """
    nc, ns = calib.nc, calib.ns
    n = nc*ns
    meta = base.metadata
    ta, tf = meta["intermediate_tariff_multipliers"], meta["final_tariff_multipliers"]
    x = np.asarray(base.x_sol, float).copy()
    i_r, i_w = np.arange(2*n, 2*n+nc), np.arange(2*n+nc, 2*n+2*nc)
    i_T, i_XN = np.arange(2*n+2*nc, 2*n+3*nc), np.arange(2*n+3*nc, 2*n+4*nc-1)
    slow, fast = np.r_[i_r, i_w], np.r_[np.arange(0, 2*n), i_T]
    fast_rows = np.r_[np.arange(0, 2*n), np.arange(2*n+3*nc, 2*n+4*nc)]
    factor_rows = np.arange(2*n, 2*n+2*nc)
    L, K = calib.l_endow.ravel(), calib.k_endow.ravel()
    omega0 = np.exp(x[i_w])@L+np.exp(x[i_r])@K

    def with_balances(xx):
        xx = xx.copy()
        xx[i_XN] = x[i_XN]*(np.exp(xx[i_w])@L+np.exp(xx[i_r])@K)/omega0
        return xx

    def physical(xx):
        return evaluate(with_balances(xx), calib, ta, tf, None, None)["physical_residuals"]

    def state(z):
        xx = x.copy()
        xx[slow] = z

        def fast_residual(u):
            xx[fast] = u
            return physical(xx)[fast_rows]
        u, info, ier, _ = fsolve(fast_residual, x[fast], full_output=True, xtol=1e-13)
        assert ier == 1 and np.max(np.abs(info["fvec"])) < 1e-9
        xx[fast] = u
        return with_balances(xx)

    def excess(z):
        e = -physical(state(z))[factor_rows]/np.r_[L, K]
        return np.r_[e[nc:], e[:nc]]
    return state, excess


@pytest.fixture(scope="module")
def two_country():
    calib = synthetic_2c_2s()
    base = solve_consistent(calib)
    return calib, base


@pytest.fixture(scope="module")
def three_country():
    calib = synthetic_3c_3s()
    return calib, solve_consistent(calib)


# ---------------------------------------------------------------------------
# Closed-form oracle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("balance", [False, True])
@pytest.mark.parametrize("units", ["world_factor_income", "reference_factor_price", "numeraire"])
@pytest.mark.parametrize("reference,factor", [(0, "w"), (1, "r")])
def test_closed_form_reduced_jacobian(balance, units, reference, factor):
    data, y0, alpha, F = toy_table(balance)
    calib = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"])
    result = solve_consistent(calib, tol=1e-12)
    ref = reference if factor == "r" else 2+reference
    expected = closed_form_reduced_jacobian(y0, alpha, F, calib.invforT.ravel(), units, ref)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # numeraire units are not homogeneous by design
        st = reduced_stability(calib, result, balance_units=units, reference=reference, reference_factor=factor)
        fourth = reduced_stability(calib, result, balance_units=units, reference=reference, reference_factor=factor,
                                   step=1e-3, fd_order=4, dense=False)
    np.testing.assert_allclose(st.reduced_jacobian, expected, rtol=0, atol=TOL)
    np.testing.assert_allclose(fourth.reduced_jacobian, expected, rtol=0, atol=1e-11)
    assert st.dense_check_error is not None and st.dense_check_error <= TOL
    assert st.classification == "unstable" and st.rank == 3
    assert st.reference == ("r[B]" if factor == "r" else "w[A]")
    if units != "numeraire" or not balance:
        assert st.homogeneity_error <= TOL and st.metadata["homogeneous"]
    else:
        assert st.homogeneity_error > 1e-3 and not st.metadata["homogeneous"]
    assert st.walras_error <= TOL
    assert st.metadata["world_identity_error"] <= 1e-12
    # The toy's unstable root is 2/9 when balances vanish (price-inelastic Leontief demand).
    if not balance:
        assert st.maximum_real_eigenvalue == pytest.approx(2/9, abs=1e-8)


def test_closed_form_spectrum_is_reference_invariant_only_with_world_units():
    data, y0, alpha, F = toy_table(True)
    calib = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"])
    result = solve_consistent(calib, tol=1e-12)
    world = [np.sort_complex(reduced_stability(calib, result, reference=c, reference_factor=f, dense=False).eigenvalues)
             for c, f in ((0, "w"), (1, "w"), (0, "r"), (1, "r"))]
    for spectrum in world[1:]:
        np.testing.assert_allclose(spectrum, world[0], rtol=0, atol=TOL)
    by_reference = [reduced_stability(calib, result, balance_units="reference_factor_price", reference=c,
                                      reference_factor=f, dense=False).maximum_real_eigenvalue
                    for c, f in ((0, "w"), (1, "r"))]
    assert abs(by_reference[0]-by_reference[1]) > 1e-3   # documented: depends on the reference when B != 0


# ---------------------------------------------------------------------------
# Schur complement, homogeneity, Walras, references, degeneracy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["two_country", "three_country"])
def test_schur_complement_matches_independent_re_solve_and_identities(name, request):
    calib, base = request.getfixturevalue(name)
    nc = calib.nc
    st = reduced_stability(calib, base)
    assert isinstance(st, StabilityResult)
    assert st.dense_check_error <= TOL
    assert st.homogeneity_error <= TOL and st.walras_error <= TOL
    assert st.metadata["world_identity_error"] <= 1e-12
    assert st.reduced_jacobian.shape == (2*nc, 2*nc) and st.relative_matrix.shape == (2*nc-1, 2*nc-1)
    assert st.rank == 2*nc-1
    assert st.equilibrium_residual <= 1e-11
    np.testing.assert_allclose(st.metadata["excess_demand_at_state"], 0., atol=1e-11)
    # Value-weighted left null vector explicitly.
    weights = np.r_[base.r_sol.ravel()*calib.k_endow.ravel(), base.w_sol.ravel()*calib.l_endow.ravel()]
    assert np.max(np.abs(weights@st.reduced_jacobian))/np.max(np.abs(weights)) <= TOL
    np.testing.assert_allclose(st.reduced_jacobian@np.ones(2*nc), 0., atol=TOL)
    # Both synthetic fixtures classify unstable under this partition (documented, closure dependent).
    assert st.classification == "unstable"
    assert 0.2 < st.maximum_real_eigenvalue < 0.3
    assert st.eigenvalues[0].real == st.maximum_real_eigenvalue
    assert np.all(np.diff(st.eigenvalues.real) <= 1e-12)
    assert st.qualification == QUALIFICATION and "uniqueness" in st.qualification


def test_reference_invariance_and_speeds(three_country):
    calib, base = three_country
    spectra = [np.sort_complex(reduced_stability(calib, base, reference=c, reference_factor=f, dense=False).eigenvalues)
               for c, f in ((0, "w"), (2, "w"), (0, "r"), (2, "r"))]
    for spectrum in spectra[1:]:
        np.testing.assert_allclose(spectrum, spectra[0], rtol=0, atol=TOL)
    # Absolute spectrum = relative spectrum plus the homogeneity zero.
    absolute = np.sort_complex(reduced_stability(calib, base, dense=False).metadata["absolute_eigenvalues"])
    np.testing.assert_allclose(np.sort_complex(np.r_[spectra[0], 0.]), absolute, rtol=0, atol=1e-7)
    rng = np.random.default_rng(1)
    A = np.asarray(reduced_stability(calib, base, dense=False).reduced_jacobian)
    for _ in range(8):
        speeds = np.exp(rng.uniform(np.log(.25), np.log(4.), 2*calib.nc))
        st = reduced_stability(calib, base, speeds=speeds, dense=False)
        assert st.classification == "unstable"
        np.testing.assert_array_equal(st.metadata["speeds"], speeds)
        # Speeds rescale the law only: the stored reduced Jacobian is unscaled.
        np.testing.assert_allclose(st.reduced_jacobian, A, rtol=0, atol=TOL)
        # Oracle: the law is diag(d) A (rows scaled), so eig(R) = eig(diag(d) A) without its homogeneity zero.
        law = speeds[:, None]*A
        expected = np.linalg.eigvals(law)
        expected = np.delete(expected, np.argmin(np.abs(expected)))
        np.testing.assert_allclose(np.sort_complex(st.eigenvalues), np.sort_complex(expected), rtol=0, atol=TOL)
        ref, keep = st.metadata["reference_index"], st.metadata["keep_indices"]
        np.testing.assert_allclose(st.relative_matrix, law[np.ix_(keep, keep)]-law[ref, keep][None, :],
                                   rtol=0, atol=1e-14)


@pytest.mark.parametrize("speeds_seed", [None, 5])
def test_mode_loadings_are_right_eigenvectors_of_the_law(three_country, speeds_seed):
    """Lifted loadings u (zero at the reference) satisfy law u = lambda u + c 1 for the relative law."""
    calib, base = three_country
    speeds = None if speeds_seed is None else np.exp(np.random.default_rng(speeds_seed).uniform(-1., 1., 2*calib.nc))
    st = reduced_stability(calib, base, speeds=speeds, reference=1, reference_factor="r", dense=False)
    law = np.asarray(st.reduced_jacobian) if speeds is None else speeds[:, None]*st.reduced_jacobian
    for mode in range(len(st.eigenvalues)):
        loadings = st.mode_loadings(mode)
        assert loadings.loc[st.reference, "real"] == 0. and loadings.loc[st.reference, "imag"] == 0.
        u = loadings["real"].to_numpy()+1j*loadings["imag"].to_numpy()
        assert np.linalg.norm(u) == pytest.approx(1., abs=1e-12)
        resid = law@u-st.eigenvalues[mode]*u
        assert np.max(np.abs(resid-resid.mean())) < TOL


@pytest.mark.parametrize("name", ["two_country", "three_country"])
def test_naive_current_account_partition_is_degenerate(name, request):
    calib, base = request.getfixturevalue(name)
    st = reduced_stability(calib, base, current_account="fast", dense=False)
    assert st.classification == "degenerate"
    assert st.rank == calib.nc
    assert np.sum(np.abs(st.eigenvalues.real) < 1e-7) == calib.nc-1
    assert "current accounts" in st.closure and "fast unknowns" in st.closure
    assert st.metadata["balance_units"] is None


def test_naive_partition_and_default_path_in_realistic_units(three_country):
    """Flows x1e6: XN steps scale with income, so the naive partition stays degenerate and A is unit invariant."""
    calib, base = three_country
    scaled = calibrate_trade_model(np.asarray(calib.data_calibra)*1e6, ns=3, nc=3, nfd=3)
    result = solve_consistent(scaled, tol=1e-4)
    naive = reduced_stability(scaled, result, current_account="fast")
    assert naive.classification == "degenerate" and naive.rank == scaled.nc
    assert naive.metadata["singular_values"][scaled.nc] < 1e-7 < naive.metadata["singular_values"][scaled.nc-1]
    default = reduced_stability(scaled, result)
    np.testing.assert_allclose(default.reduced_jacobian, reduced_stability(calib, base).reduced_jacobian,
                               rtol=0, atol=TOL)


def test_rank_tolerance_is_separate_from_the_sign_band(two_country):
    calib, base = two_country
    wide_band = reduced_stability(calib, base, tol=.2, dense=False)       # max Re +0.245, smallest singular value 0.19
    assert wide_band.classification == "unstable" and wide_band.rank == 3
    assert wide_band.metadata["tol"] == .2 and wide_band.metadata["rank_tol"] == 1e-7
    coarse_rank = reduced_stability(calib, base, tol=.2, rank_tol=.2, dense=False)
    assert coarse_rank.classification == "degenerate" and coarse_rank.rank == 2
    assert reduced_stability(calib, base, tol=.3, dense=False).classification == "nonhyperbolic"


def test_eigenvalue_continuity_across_small_tariff_change(two_country):
    calib, base = two_country
    st0 = reduced_stability(calib, base, dense=False)
    previous = 0.
    for rate in (1e-3, 1e-2):
        tau, tfd = tariff_schedules(calib, rate)
        cf = solve_consistent(calib, tau=tau, tau_fd=tfd)
        st = reduced_stability(calib, cf, dense=False)
        gap = np.max(np.abs(np.sort_complex(st.eigenvalues)-np.sort_complex(st0.eigenvalues)))
        assert gap < 50*rate and gap > previous
        previous = gap
        assert st.classification == st0.classification


def test_euler_tatonnement_sign_on_nonlinear_fast_cleared_map(two_country):
    """Independent fsolve re-solve of the fast block; norms decrease along the stable mode and grow along the unstable one."""
    calib, base = two_country
    nc, n = calib.nc, calib.nc*calib.ns
    _, excess = fast_cleared_map(calib, base)
    st = reduced_stability(calib, base, dense=False)
    A = st.reduced_jacobian
    values, vectors = np.linalg.eig(A)
    order = np.argsort(values.real)
    z0 = np.asarray(base.x_sol, float)[2*n:2*n+2*nc].copy()
    # Independent central-difference Jacobian of the re-solved map.
    direct = np.empty_like(A)
    for j in range(2*nc):
        zp, zm = z0.copy(), z0.copy()
        zp[j] += 1e-5
        zm[j] -= 1e-5
        direct[:, j] = (excess(zp)-excess(zm))/2e-5
    np.testing.assert_allclose(direct, A, rtol=0, atol=TOL)
    for k, expect_growth in ((order[0], False), (order[-1], True)):
        v = vectors[:, k].real
        v -= v.mean()
        v /= np.linalg.norm(v)
        z = z0+1e-3*v
        norms = []
        for _ in range(5):
            e = excess(z)
            norms.append(np.linalg.norm(e-e.mean()))
            z = z+.5*e
        diffs = np.diff(norms)
        assert np.all(diffs > 0) if expect_growth else np.all(diffs < 0), (values[k], norms)


def test_identities_hold_off_equilibrium_on_the_fast_cleared_manifold(two_country):
    """At a point where goods, prices and budgets clear but factor markets do not, A 1 = 0 and W A + W e = 0."""
    calib, base = two_country
    nc, n = calib.nc, calib.nc*calib.ns
    state, _ = fast_cleared_map(calib, base)
    z = np.asarray(base.x_sol, float)[2*n:2*n+2*nc]+np.array([.08, -.05, .03, -.1])
    x = state(z)
    st = reduced_stability(calib, replace(base, x_sol=x), equilibrium_tol=1e3)
    e = st.metadata["excess_demand_at_state"]
    assert np.max(np.abs(e)) > .02                               # genuinely off equilibrium
    weights = np.r_[np.exp(z[:nc])*calib.k_endow.ravel(), np.exp(z[nc:])*calib.l_endow.ravel()]
    assert abs(weights@e)/np.max(weights) < 1e-12                # W.e = 0 on the manifold
    assert np.max(np.abs(weights@st.reduced_jacobian)) > 1e-3*np.max(weights)   # W A alone is not zero here
    assert st.walras_error <= TOL and st.homogeneity_error <= TOL
    assert st.dense_check_error <= TOL


def test_intermediate_substitution_stabilises_both_fixtures():
    """Documented observation: max Re eig falls monotonically in sigma and turns negative at sigma = 1."""
    for calib in (synthetic_2c_2s(), synthetic_3c_3s()):
        maxima = []
        for sigma in (0., .5, 1., 2.):
            st = reduced_stability(calib, solve_consistent(calib, sigma=sigma), dense=(sigma == .5))
            assert st.metadata["sigma"] == sigma
            assert st.homogeneity_error <= TOL and st.walras_error <= TOL
            maxima.append(st.maximum_real_eigenvalue)
        assert np.all(np.diff(maxima) < 0)
        assert maxima[0] > 0 and maxima[2] < 0 and maxima[3] < 0


# ---------------------------------------------------------------------------
# Legacy accounting
# ---------------------------------------------------------------------------
def test_legacy_and_consistent_agree_when_accountings_coincide():
    calib = synthetic_2c_2s(tax_rate=0., fd_tax=0.)
    assert np.max(np.abs(calib.tax)) == 0 and np.max(np.abs(calib.tax_fd)) == 0
    consistent = reduced_stability(calib, solve_consistent(calib))
    legacy_result = solve_trade_equilibrium(calib, accounting="legacy", tol=1e-11, max_iter=100,
                                            replicate_matlab_precedence=False)
    assert "intermediate_tariff_multipliers" not in legacy_result.metadata
    legacy = reduced_stability(calib, legacy_result)
    np.testing.assert_allclose(legacy.reduced_jacobian, consistent.reduced_jacobian, rtol=0, atol=TOL)
    np.testing.assert_allclose(np.sort_complex(legacy.eigenvalues), np.sort_complex(consistent.eigenvalues), rtol=0, atol=TOL)
    assert legacy.classification == consistent.classification == "unstable"
    assert legacy.homogeneity_error <= TOL and legacy.walras_error <= TOL
    assert not legacy.metadata["walras_identity_exact"] and consistent.metadata["walras_identity_exact"]
    # MATLAB operator precedence is not a degree-one homogeneous cost function: the maps differ and a warning is issued.
    matlab = solve_trade_equilibrium(calib, accounting="legacy", tol=1e-11, max_iter=100, replicate_matlab_precedence=True)
    with pytest.warns(RuntimeWarning, match="not homogeneous"):
        matlab_st = reduced_stability(calib, matlab, dense=False)
    assert not matlab_st.metadata["homogeneous"]
    assert np.max(np.abs(matlab_st.reduced_jacobian-consistent.reduced_jacobian)) > 1e-2


def test_legacy_result_needs_its_tariffs_and_reports_non_homogeneity():
    calib = synthetic_2c_2s()
    tau, tfd = tariff_schedules(calib, .25)
    legacy = solve_trade_equilibrium(calib, tau=tau, tau_fd=tfd, accounting="legacy", tariff_revenue_mode="schedule",
                                     sigma=.5, tol=1e-11, max_iter=100)
    with pytest.raises(StabilityError, match="violates"):
        reduced_stability(calib, legacy, dense=False)
    with pytest.warns(RuntimeWarning, match="not homogeneous"):
        st = reduced_stability(calib, legacy, tau=tau, tau_fd=tfd)
    assert st.dense_check_error <= TOL          # the Schur complement is still exact for the map that was evaluated
    assert st.metadata["accounting"] == "legacy" and st.metadata["tariff_revenue_mode"] == "schedule"
    assert st.homogeneity_error > 1e-2 and not st.metadata["homogeneous"]
    assert not st.metadata["homogeneity_exact"] and not st.metadata["walras_identity_exact"]
    assert len(st.metadata["absolute_eigenvalues"]) == 2*calib.nc
    # The caveat travels with the verdict, not only with the warning.
    assert "not homogeneous of degree zero" in st.closure and "not homogeneous of degree zero" in st.summary()


# ---------------------------------------------------------------------------
# Bundled and reference fixtures
# ---------------------------------------------------------------------------
def test_oecd_three_region_fixture_classification_depends_on_accounting():
    from puremacro.trade.data import package_mrio_to_calibration_result
    from puremacro.trade._oecd_icio import condense_final_demand
    from tools.reference_validation.validate_oecd import load_fixture
    calib = package_mrio_to_calibration_result(condense_final_demand(load_fixture()))
    base = solve_trade_equilibrium(calib, accounting="consistent", tol=1e-5)
    st = reduced_stability(calib, base)
    assert st.factor_labels == ("r[USA]", "r[CHN]", "r[REST]", "w[USA]", "w[CHN]", "w[REST]")
    assert st.reference == "w[USA]"
    assert st.classification == "unstable" and st.dense_check_error <= TOL
    assert st.homogeneity_error <= TOL and st.walras_error <= TOL
    assert st.fast_block_condition < 1e3         # equilibrated: raw cond(G) mixes log unknowns with million-USD levels
    # Million-USD units: the naive partition is still detected as degenerate (XN steps scale with income).
    naive = reduced_stability(calib, base, current_account="fast")
    assert naive.classification == "degenerate" and naive.rank == calib.nc
    # Intermediate substitution: unstable under fixed coefficients, stable once sigma reaches one
    # (observed max Re +0.1445, +0.0496, -0.0216 at sigma 0, 0.5, 1).
    maxima = [st.maximum_real_eigenvalue]
    for sigma, expected in ((.5, "unstable"), (1., "stable")):
        st_sigma = reduced_stability(calib, solve_trade_equilibrium(calib, accounting="consistent", tol=1e-5,
                                                                    sigma=sigma, max_iter=200))
        assert st_sigma.classification == expected and st_sigma.metadata["sigma"] == sigma
        maxima.append(st_sigma.maximum_real_eigenvalue)
    assert np.all(np.diff(maxima) < 0) and maxima[-1] < -.01
    rates = np.array([.1, 0., 0.])
    counter = solve_trade_equilibrium(calib, tau=rates, tau_fd=rates, accounting="consistent", tol=1e-4, max_iter=100)
    st_cf = reduced_stability(calib, counter)
    assert st_cf.classification == "unstable"
    assert abs(st_cf.maximum_real_eigenvalue-st.maximum_real_eigenvalue) < .01
    legacy = solve_trade_equilibrium(calib, tau=rates, tau_fd=rates, accounting="legacy",
                                     tariff_revenue_mode="schedule", tol=1e-4, max_iter=100)
    with pytest.warns(RuntimeWarning, match="not homogeneous"):
        st_legacy = reduced_stability(calib, legacy, tau=rates, tau_fd=rates)
    # Same table, different accounting, different verdict: the classification is closure dependent.
    assert st_legacy.classification == "stable" and st_legacy.homogeneity_error > .1


def test_hicksian_two_country_ces_model_is_stable_and_reports_labels():
    from tools.reference_validation.validate_trade_welfare import consumption_benchmark
    data, tau, fd = consumption_benchmark()
    calib = calibrate_trade_model(data, ns=1, nc=2, nfd=3, country_codes=["A", "B"])
    base = solve_policy_equilibrium(calib, sigma=2., tol=1e-9)
    st = reduced_stability(calib, base)
    assert st.classification == "stable" and st.maximum_real_eigenvalue < -.3
    policy = solve_policy_equilibrium(calib, tau, fd, sigma=2., tol=1e-9)
    st_policy = reduced_stability(calib, policy, reference="B", reference_factor="r")
    assert st_policy.classification == "stable" and st_policy.reference == "r[B]"
    loadings = st_policy.mode_loadings(0)
    assert list(loadings.index) == ["r[A]", "r[B]", "w[A]", "w[B]"]
    assert loadings.loc["r[B]", "real"] == 0.


@pytest.mark.slow
def test_bundled_77x11_legacy_run():
    from puremacro.trade.data import load_icio_data
    calib = calibrate_trade_model(load_icio_data(source="legacy"), ns=11, nc=77, nfd=3)
    result = solve_trade_equilibrium(calib)
    assert result.converged
    tic = time.perf_counter()
    with pytest.warns(RuntimeWarning, match="not homogeneous"):
        st = reduced_stability(calib, result, dense=False)
    elapsed = time.perf_counter()-tic
    assert elapsed < 600
    assert st.classification in ("stable", "unstable", "nonhyperbolic", "degenerate")
    assert st.rank == 2*calib.nc-1 and st.relative_matrix.shape == (153, 153)
    assert np.isfinite(st.eigenvalues).all()
    assert st.metadata["jacobian_evaluations"] == 2*(2*847+3*77)+1
    # The bundled legacy accounting is not degree-one homogeneous; the record documents it.
    assert not st.metadata["homogeneous"] and st.homogeneity_error > 1e-2
    assert st.fast_block_condition > 1.


# ---------------------------------------------------------------------------
# Contracts and reporting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs,error", [
    (dict(reference="ZZZ"), ValueError), (dict(reference=1.5), TypeError), (dict(reference=7), ValueError),
    (dict(adjustment="absolute"), ValueError), (dict(balance_units="gold"), ValueError),
    (dict(current_account="ignored"), ValueError), (dict(reference_factor="p"), ValueError),
    (dict(speeds=np.ones(3)), ValueError), (dict(speeds=-np.ones(4)), ValueError), (dict(fd_order=3), ValueError),
    (dict(step=0.), ValueError), (dict(tol=np.nan), ValueError), (dict(equilibrium_tol=-1.), ValueError),
    (dict(rank_tol=0.), ValueError), (dict(step=.5), ValueError), (dict(dense_step=.1), ValueError),
    (dict(tau=np.array([.1, 0.])), ValueError), (dict(dense_tol=1e-12), StabilityError),
    # Roundoff-dominated step: exact homogeneity fails under the consistent accounting, an error not a caveat.
    (dict(step=1e-12, dense=False), StabilityError)])
def test_invalid_inputs_raise(two_country, kwargs, error):
    calib, base = two_country
    with pytest.raises(error):
        reduced_stability(calib, base, **kwargs)


def test_state_and_closure_contracts(two_country):
    calib, base = two_country
    with pytest.raises(ValueError, match="converged"):
        reduced_stability(calib, replace(base, converged=False))
    with pytest.raises(StabilityError, match="violates"):
        reduced_stability(calib, replace(base, x_sol=base.x_sol+1e-3))
    with pytest.raises(NotImplementedError, match="lump-sum"):
        reduced_stability(calib, replace(base, metadata={**base.metadata, "fiscal_closure": "capital_tax"}))
    with pytest.raises(NotImplementedError, match="capacity"):
        reduced_stability(calib, replace(base, metadata={**base.metadata, "capacity_margins": {"MANU": .1}}))
    with pytest.raises(ValueError, match="recorded tariff"):
        meta = {k: v for k, v in base.metadata.items() if k != "final_tariff_multipliers"}
        reduced_stability(calib, replace(base, metadata=meta))
    with pytest.raises(TypeError):
        reduced_stability(base, base)
    with pytest.raises(TypeError):
        reduced_stability(calib, calib)
    # Homogeneity is exact under the consistent accounting in world units: its failure is an error, not a warning.
    with pytest.raises(StabilityError, match="homogeneity"):
        reduced_stability(calib, base, step=1e-12, dense=False)
    # Explicit schedules equal to the recorded ones are accepted.
    st = reduced_stability(calib, base, tau=base.metadata["intermediate_tariff_multipliers"],
                           tau_fd=base.metadata["final_tariff_multipliers"], dense=False)
    assert st.classification == "unstable"


def test_result_reporting_quintet_and_immutability(two_country):
    calib, base = two_country
    st = reduced_stability(calib, base, dense=False)
    frame = st.to_dataframe()
    assert list(frame.columns) == ["real", "imag", "modulus"] and len(frame) == 3
    assert frame.index.name == "mode"
    for text in (st.to_markdown(), st.to_latex(), st.to_typst()):
        assert "0.2453" in text
    summary = st.summary()
    assert "unstable" in summary and "not a uniqueness result" in summary
    assert "world factor income" in st.closure and "certificate only" in st.closure
    assert "not homogeneous" not in st.closure
    assert st.metadata["homogeneity_exact"] and st.metadata["walras_identity_exact"]
    assert "w[AAA]" in st.adjustment and "e_i - e_ref" in st.adjustment
    assert st.metadata["experimental"] is True
    with pytest.raises(AttributeError):
        st.classification = "stable"
    for array in (st.reduced_jacobian, st.relative_matrix, st.eigenvalues):
        assert not array.flags.writeable
    for key in ("keep_indices", "eigenvectors", "singular_values", "excess_demand_at_state", "absolute_eigenvalues"):
        assert not st.metadata[key].flags.writeable, key
    with pytest.raises(ValueError):
        st.mode_loadings(5)
    loadings = st.mode_loadings(0)
    assert loadings.loc["w[AAA]", "real"] == 0. and np.isclose(np.linalg.norm(loadings["real"]+1j*loadings["imag"]), 1.)
