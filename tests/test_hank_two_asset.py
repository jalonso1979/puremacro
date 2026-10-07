"""Two-asset HANK sequence-space solver: correctness tests.

What is checked (each property failed on the pre-4.3.1 solver):

1. The household block iterates EGM on (V_a, V_b) to a real tolerance: the
   returned policies are a fixed point of one more backward step, and
   ``converged`` is False (with a warning) when the iteration limit is hit.
2. The default calibration has beta (1 + r_a) < 1 and an interior illiquid
   distribution (no mass at a_max); the old calibration triggers a warning.
3. No wealth leaks at the grid edges: the aggregate budget holds exactly in
   the steady state and date by date along every Jacobian column.
4. Every Fake-News Jacobian column (all inputs, several dates, all outputs)
   equals the brute-force perturbation of the non-linear household block
   (direct method); the Jacobians carry anticipation effects and the ex-ante
   r_b dating (one-period shift).
5. The present-value household budget closes for each column: PV(dC + dCHI) =
   PV(income) + (r_a - r_b) PV(dA_{t-1}) with a vanishing terminal term.
6. The income closure z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1} is paid on the
   households' own lagged holdings: the closed Jacobians equal the brute-force
   response of the non-linear closed household block and satisfy the aggregate
   budget J_Q + (I - L) J_W = I (Y) / 0 (rates), so in GE total wealth A + B
   does not move (Walras's law; a fixed-stock closure has an explosive root).
7. GE: after a rate cut, consumption and output rise on impact at horizons
   16, 20, 30, 40, 60, the IRFs decay (also at T = 300, where an explosive
   wealth root would show), whole paths do not depend on the horizon, and the
   household block fed the GE prices reproduces the GE consumption path.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from puremacro.models.hank_sequence_space import (
    _TA_FD_STEP,
    _TA_OUTPUTS,
    _transaction_cost,
    _lottery_2d,
    _build_transition_matrix_2d,
    _stationary_distribution_2d,
    _solve_two_asset_household_block,
    _two_asset_backward,
    _two_asset_closed_transition,
    _two_asset_closure_matrices,
    _two_asset_floored_states,
    _two_asset_ge_paths,
    _two_asset_ge_solve,
    _two_asset_household_jacobians,
    _two_asset_jacobians,
    _two_asset_transition,
    solve_two_asset_hank_sequence_space,
    TwoAssetSequenceSpaceHANKResult,
)
from puremacro.dsge.hank import (
    HANKModel,
    HANKResult,
    load_hank_mod,
    solve_hank_bridge,
)

MOD = "puremacro/dsge/_references/hank_two_asset.mod"


@pytest.fixture(scope="module")
def hh():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the default calibration must not warn
        return _solve_two_asset_household_block()


@pytest.fixture(scope="module")
def raw_jac(hh):
    return _two_asset_household_jacobians(hh, 24)


def _direct_column(hh, x, s, T, eps):
    """Brute-force column: central difference of the non-linear household block."""
    base = {"r_b": hh.r_b_ss, "r_a": hh.r_a_ss, "z": hh.w_ss}
    up = {k: np.full(T, v) for k, v in base.items()}
    dn = {k: np.full(T, v) for k, v in base.items()}
    up[x][s] += eps
    dn[x][s] -= eps
    a_up = _two_asset_transition(hh, up["r_b"], up["r_a"], up["z"])
    a_dn = _two_asset_transition(hh, dn["r_b"], dn["r_a"], dn["z"])
    return {o: (a_up[o] - a_dn[o]) / (2.0 * eps) for o in _TA_OUTPUTS}


# ---------------------------------------------------------------------------
# 1. Primitives
# ---------------------------------------------------------------------------

def test_stationary_distribution_normalization(hh):
    """Lambda is column-stochastic, D* solves Lambda D = D and sums to one."""
    assert np.allclose(hh.Lambda.sum(axis=0), 1.0, atol=1e-12)
    D = hh.D_ss.ravel()
    assert np.max(np.abs(hh.Lambda @ D - D)) < 1e-12
    assert np.sum(hh.D_ss) == pytest.approx(1.0, abs=1e-12)
    assert np.all(hh.D_ss >= 0.0)
    assert np.sum(hh.marginal_distribution_a) == pytest.approx(1.0, abs=1e-12)
    assert np.sum(hh.marginal_distribution_b) == pytest.approx(1.0, abs=1e-12)
    assert np.sum(hh.joint_distribution) == pytest.approx(1.0, abs=1e-12)
    # the standalone helper reproduces it
    D2 = _stationary_distribution_2d(_build_transition_matrix_2d(hh.a_ss, hh.b_ss, hh.a_grid, hh.b_grid, hh.pi_s))
    assert np.allclose(D2, D, atol=1e-12)


def test_lottery_is_mean_preserving_inside_the_grid():
    a_grid = np.linspace(0.0, 10.0, 11) ** 1.5
    b_grid = np.linspace(0.0, 5.0, 9)
    rng = np.random.default_rng(0)
    a_dest = rng.uniform(0.0, a_grid[-1], 200)
    b_dest = rng.uniform(0.0, b_grid[-1], 200)
    ial, iah, wal, wah, ibl, ibh, wbl, wbh = _lottery_2d(a_dest, b_dest, a_grid, b_grid)
    assert np.allclose(wal * a_grid[ial] + wah * a_grid[iah], a_dest, atol=1e-12)
    assert np.allclose(wbl * b_grid[ibl] + wbh * b_grid[ibh], b_dest, atol=1e-12)


def test_transaction_cost_properties():
    """chi(0, a) = 0, chi > 0 otherwise, symmetric and increasing in |d|; finite at a = 0 with a_bar."""
    a = 5.0
    assert _transaction_cost(0.0, a, chi_0=0.25, chi_1=1.0) == 0.0
    assert _transaction_cost(1.0, a, chi_0=0.25, chi_1=1.0) > 0.0
    assert _transaction_cost(-1.0, a, chi_0=0.25, chi_1=1.0) > 0.0
    assert _transaction_cost(2.0, a, chi_0=0.25, chi_1=1.0) == pytest.approx(_transaction_cost(-2.0, a, chi_0=0.25, chi_1=1.0))
    costs = [_transaction_cost(d, a, chi_0=0.25, chi_1=1.0) for d in (0.5, 1.0, 2.0, 3.0)]
    assert np.all(np.diff(costs) > 0.0)
    assert _transaction_cost(1.5, a, chi_0=0.3, chi_1=1.5) > 0.0
    # quadratic case with the shift: 0.5 chi_0 d^2 / (a + a_bar)
    assert _transaction_cost(0.4, 0.0, chi_0=1.0, chi_1=1.0, a_bar=0.25) == pytest.approx(0.5 * 0.16 / 0.25)
    # general chi_1 form chi_0/(1+chi_1) |d|^(1+chi_1) / (a + a_bar)^chi_1
    assert _transaction_cost(0.7, 3.0, chi_0=0.3, chi_1=1.5, a_bar=0.25) == pytest.approx(
        0.3 / 2.5 * 0.7 ** 2.5 / 3.25 ** 1.5)


def test_policy_function_properties(hh):
    assert np.all(np.diff(hh.c_ss, axis=1) > 0.0)  # consumption increasing in liquid wealth
    assert np.all(hh.c_ss > 0.0)
    assert np.all(hh.b_ss >= hh.b_grid[0]) and np.all(hh.b_ss <= hh.b_grid[-1])
    assert np.all(hh.a_ss >= 0.0) and np.all(hh.a_ss <= hh.a_grid[-1])
    assert np.mean(hh.c_ss[:, :, 1]) > np.mean(hh.c_ss[:, :, 0])
    mpc = hh.mpc()
    assert np.all(mpc > 0.0) and np.all(mpc < 1.0 + 1e-9)
    assert 0.05 < float(np.sum(hh.D_ss * mpc)) < 0.6


# ---------------------------------------------------------------------------
# 2. Convergence and calibration
# ---------------------------------------------------------------------------

def test_household_block_is_a_fixed_point(hh):
    """The returned policies reproduce themselves under one more backward step."""
    assert hh.converged is True
    assert hh.residual < hh.tol
    assert hh.iterations < 10000
    Va, Vb, c, ap, bp, d, chi = _two_asset_backward(hh, hh.Va_ss, hh.Vb_ss)
    assert np.max(np.abs(c - hh.c_ss)) < 1e-9
    assert np.max(np.abs(ap - hh.a_ss)) < 1e-9
    assert np.max(np.abs(bp - hh.b_ss)) < 1e-9
    assert np.max(np.abs(Vb - hh.Vb_ss) / hh.Vb_ss) < 1e-9


def test_non_convergence_is_reported():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hh5 = _solve_two_asset_household_block(max_iter=5)
    messages = [str(w.message) for w in rec if issubclass(w.category, RuntimeWarning)]
    assert any("did not converge" in m for m in messages)
    assert hh5.converged is False
    assert hh5.residual > hh5.tol


def test_default_calibration_has_interior_illiquid_distribution(hh):
    assert hh.beta * (1.0 + hh.r_a_ss) < 1.0
    assert hh.mass_at_a_max < 1e-10
    assert hh.mass_at_b_max < 1e-10
    Da = hh.marginal_distribution_a
    assert int(np.sum(Da > 0.01)) >= 8  # spread over the grid, not piled on one node
    assert 0.0 < hh.A_ss < 0.5 * hh.a_grid[-1]
    assert hh.B_ss > 0.0
    assert 0.05 < hh.htm_share < 0.95
    assert 0.0 < hh.wealthy_htm_share < hh.htm_share


def test_old_calibration_warns_about_mass_at_the_grid_top():
    """beta (1 + r_a) = 0.985 * 1.03 > 1 piles illiquid wealth at a_max: the solver must say so."""
    with pytest.warns(RuntimeWarning, match="top illiquid grid point"):
        old = _solve_two_asset_household_block(
            beta=0.985, r_b_ss=0.01, r_a_ss=0.03, chi_0=0.25, a_max=30.0, b_max=15.0)
    assert old.mass_at_a_max > 0.5


def test_grid_top_warning_reports_small_masses_and_only_the_offending_grid():
    """A mass of 1e-5 must not print as '0.00%', and only the grid that exceeds the threshold is named."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hh10 = _solve_two_asset_household_block(chi_0=10.0)
    msgs = [str(w.message) for w in rec if "grid point" in str(w.message)]
    assert len(msgs) == 1
    assert 1e-8 < hh10.mass_at_b_max < 1e-4 and hh10.mass_at_a_max <= 1e-8
    assert f"{hh10.mass_at_b_max:.3e}" in msgs[0]
    assert "top liquid grid point" in msgs[0] and "illiquid" not in msgs[0]
    assert "0.00%" not in msgs[0]


def test_unsupported_curvature_warns_and_stops_early():
    """chi_1 < 1 is outside the supported range: warn up front, detect the cycle, report converged=False."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        hh_bad = _solve_two_asset_household_block(chi_1=0.5, n_a=15, n_b=15)
    messages = [str(w.message) for w in rec if issubclass(w.category, RuntimeWarning)]
    assert any("outside the supported range" in m for m in messages)
    assert any("stalled" in m for m in messages)
    assert hh_bad.converged is False
    assert hh_bad.iterations <= 2000  # stopped at the stall check instead of running to max_iter=10000


def test_consumption_floor_is_detected():
    """The floor c >= _TA_C_FLOOR breaks the budget constraint; it must be counted and reported, not hidden."""
    hh0 = _solve_two_asset_household_block(n_a=15, n_b=15)
    args = (hh0.a_grid, hh0.b_grid, hh0.s_grid, hh0.r_b_ss, hh0.r_a_ss, hh0.w_ss)
    assert _two_asset_floored_states(*args, hh0.a_ss, hh0.b_ss, hh0.chi_ss) == 0
    greedy = hh0.b_ss.copy()
    greedy[0, 0, 0] = hh0.b_grid[-1]  # a liquid choice the household cannot afford
    assert _two_asset_floored_states(*args, hh0.a_ss, greedy, hh0.chi_ss) == 1
    # a shock that makes cash-on-hand negative somewhere must warn in the transition
    T = 4
    z = np.full(T, hh0.w_ss)
    z[1] = -50.0
    with pytest.warns(RuntimeWarning, match="consumption floor binds"):
        _two_asset_transition(hh0, np.full(T, hh0.r_b_ss), np.full(T, hh0.r_a_ss), z)


def test_no_wealth_leak_in_steady_state(hh):
    """C + CHI = w N + r_b B + r_a A holds exactly: the lottery never clips a policy."""
    assert abs(hh.budget_residual) < 1e-12
    assert np.sum(hh.D_ss * hh.a_ss) == pytest.approx(hh.A_ss, abs=1e-12)
    assert np.sum(hh.D_ss * hh.b_ss) == pytest.approx(hh.B_ss, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Fake-News Jacobians
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x", ["r_b", "r_a", "z"])
@pytest.mark.parametrize("s", [0, 4, 12])
def test_fake_news_matches_direct_method(hh, raw_jac, x, s):
    """Every output column equals the brute-force response of the non-linear household block.

    The direct method (central difference of the whole non-linear transition)
    agrees with the Fake-News column both at the Fake-News step and at a 10x
    smaller step, so the Jacobian is the local derivative and not a step artefact.
    """
    T = raw_jac[("C", x)].shape[0]
    h = _TA_FD_STEP[x]
    same = _direct_column(hh, x, s, T, h)
    smaller = _direct_column(hh, x, s, T, h / 10.0)
    for o in _TA_OUTPUTS:
        fn = raw_jac[(o, x)][:, s]
        scale = np.max(np.abs(same[o]))
        assert np.max(np.abs(fn - same[o])) <= 1e-6 * scale, (o, x, s)
        assert np.max(np.abs(fn - smaller[o])) <= 1e-5 * scale, (o, x, s)
    if s > 0:
        # anticipation: a shock known in advance moves consumption before it happens
        assert np.max(np.abs(same["C"][:s])) > 1e-3 * np.max(np.abs(same["C"]))


def test_jacobians_have_anticipation_and_ex_ante_dating(hh):
    T = 20
    J = _two_asset_jacobians(hh, T)
    raw = _two_asset_household_jacobians(hh, T + 1)
    for key in ("J_C_rb", "J_C_ra", "J_C_Y", "J_D_rb", "J_D_ra", "J_D_Y"):
        assert J[key].shape == (T, T)
        assert np.all(np.isfinite(J[key]))
        assert np.max(np.abs(np.triu(J[key], 1))) > 1e-3 * np.max(np.abs(J[key])), key
    # an expected rise in the ex-ante liquid rate lowers consumption today and before it
    assert np.all(J["J_C_rb"][0, :6] < 0.0)
    assert np.all(J["J_C_Y"][0, :6] > 0.0)
    # the realised date-0 return raises cash-on-hand; the ex-ante rate r_0 is realised at t = 1
    assert raw[("C", "r_b")][0, 0] > 0.0
    # closed Jacobians = raw Jacobians composed with the income closure, then shifted one date
    G = _two_asset_closure_matrices(hh, raw)
    shifted = (raw[("C", "r_b")] + raw[("C", "z")] @ G["r_b"])[:T, 1:T + 1]
    assert np.allclose(J["J_C_rb"], shifted, rtol=1e-12, atol=1e-14)
    shifted_a = (raw[("C", "r_a")] + raw[("C", "z")] @ G["r_a"])[:T, 1:T + 1]
    assert np.allclose(J["J_C_ra"], shifted_a, rtol=1e-12, atol=1e-14)
    assert np.allclose(J["J_C_Y"], (raw[("C", "z")] @ G["Y"])[:T, :T], rtol=1e-12, atol=1e-14)
    # date-0 income depends only on date-0 output: lagged holdings and realised returns are predetermined
    assert J["J_z_Y"][0, 0] == pytest.approx(1.0 / hh.N_ss, rel=1e-12)
    assert np.all(J["J_z_Y"][0, 1:] == 0.0) and np.all(J["J_z_rb"][0] == 0.0)


def test_closure_is_paid_on_lagged_household_holdings(hh, raw_jac):
    """N dz_t = dY_t - B dr^b_t - A dr^a_t - r_b dB_{t-1} - r_a dA_{t-1}, with dA, dB the households' own responses."""
    T = raw_jac[("C", "z")].shape[0]
    G = _two_asset_closure_matrices(hh, raw_jac)
    rng = np.random.default_rng(1)
    dY, drb, dra = rng.normal(size=(3, T)) * np.array([1e-3, 1e-4, 1e-4])[:, None]
    dz = G["Y"] @ dY + G["r_b"] @ drb + G["r_a"] @ dra
    resp = {o: raw_jac[(o, "z")] @ dz + raw_jac[(o, "r_b")] @ drb + raw_jac[(o, "r_a")] @ dra for o in ("A", "B")}
    lag = lambda x: np.concatenate([[0.0], x[:-1]])  # noqa: E731
    rhs = (dY - hh.B_ss * drb - hh.A_ss * dra
           - hh.r_b_ss * lag(resp["B"]) - hh.r_a_ss * lag(resp["A"]))
    assert np.max(np.abs(hh.N_ss * dz - rhs)) < 1e-14


@pytest.fixture(scope="module")
def closed_jac(hh):
    return _two_asset_jacobians(hh, 24)


@pytest.mark.parametrize("x", ["Y", "rb", "ra"])
@pytest.mark.parametrize("s", [0, 5, 15])
def test_closed_jacobians_match_direct_method(hh, closed_jac, x, s):
    """Each closed column equals the brute-force response of the non-linear closed household block.

    The direct method solves the non-linear fixed point z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1}
    along the perturbed path; the ex-ante rate r_s is perturbed as the realised return at s + 1.
    """
    T = closed_jac["J_C_Y"].shape[0]
    Tp = T + 1
    eps = {"Y": 1e-4, "rb": 1e-5, "ra": 1e-5}[x]
    base = {"Y": np.full(Tp, hh.Y_ss), "rb": np.full(Tp, hh.r_b_ss), "ra": np.full(Tp, hh.r_a_ss)}
    up = {k: v.copy() for k, v in base.items()}
    dn = {k: v.copy() for k, v in base.items()}
    date = s if x == "Y" else s + 1
    up[x][date] += eps
    dn[x][date] -= eps
    a_up = _two_asset_closed_transition(hh, up["Y"], up["rb"], up["ra"])
    a_dn = _two_asset_closed_transition(hh, dn["Y"], dn["rb"], dn["ra"])
    for o in _TA_OUTPUTS + ("z",):
        direct = ((a_up[o] - a_dn[o]) / (2.0 * eps))[:T]
        fn = closed_jac[f"J_{o}_{x}"][:, s]
        assert np.max(np.abs(fn - direct)) <= 1e-6 * np.max(np.abs(direct)), (o, x, s)


def test_closed_jacobians_satisfy_the_aggregate_budget(closed_jac):
    """Household income equals Y: J_Q + (I - L) J_W = I for Y and 0 for the rates (Q = C + CHI, W = A + B)."""
    T = closed_jac["J_C_Y"].shape[0]
    diff = np.eye(T) - np.eye(T, k=-1)
    for x, target in (("Y", np.eye(T)), ("rb", np.zeros((T, T))), ("ra", np.zeros((T, T)))):
        Q = closed_jac[f"J_C_{x}"] + closed_jac[f"J_CHI_{x}"]
        W = closed_jac[f"J_A_{x}"] + closed_jac[f"J_B_{x}"]
        assert np.max(np.abs(Q + diff @ W - target)) < 1e-9, x
    # not a trivial 0 = 0: the rates do move spending and the portfolio
    assert np.max(np.abs(closed_jac["J_A_rb"])) > 1e-2
    assert np.max(np.abs(closed_jac["J_C_rb"])) > 1e-2


@pytest.mark.parametrize("x", ["r_b", "r_a", "z"])
def test_budget_identity_holds_date_by_date_for_every_column(hh, raw_jac, x):
    """dC + dCHI + dA + dB - (1+r_a) dA_{-1} - (1+r_b) dB_{-1} = income shock, for every (t, s)."""
    T = raw_jac[("C", x)].shape[0]
    income = {"r_b": hh.B_ss, "r_a": hh.A_ss, "z": hh.N_ss}[x]
    JA, JB = raw_jac[("A", x)], raw_jac[("B", x)]
    lag = lambda M: np.vstack([np.zeros((1, T)), M[:-1]])  # noqa: E731
    resid = (raw_jac[("C", x)] + raw_jac[("CHI", x)] + JA + JB
             - (1.0 + hh.r_a_ss) * lag(JA) - (1.0 + hh.r_b_ss) * lag(JB) - income * np.eye(T))
    assert np.max(np.abs(resid)) < 1e-9


def test_present_value_budget_closes(hh):
    """PV(dC + dCHI) = PV(income) + (r_a - r_b) PV(dA_{t-1}); the terminal wealth term vanishes."""
    T = 300
    raw = _two_asset_household_jacobians(hh, T, inputs=("z", "r_b"), outputs=("C", "CHI", "A", "B"))
    q = (1.0 + hh.r_b_ss) ** (-np.arange(T))
    for x, income in (("z", hh.N_ss), ("r_b", hh.B_ss)):
        for s in (0, 10):
            col = {o: raw[(o, x)][:, s] for o in ("C", "CHI", "A", "B")}
            pv_c = q @ (col["C"] + col["CHI"])
            lag_a = np.concatenate([[0.0], col["A"][:-1]])
            premium = (hh.r_a_ss - hh.r_b_ss) * (q @ lag_a)
            terminal = q[-1] * (col["A"][-1] + col["B"][-1])
            assert abs(terminal) < 1e-3 * income
            # exact up to the (vanishing) terminal term and rounding over the T-period sums;
            # 1e-9 * income was missed by 8% on Windows CI (6.6e-10), so allow 1e-8
            assert abs(pv_c - (income * q[s] + premium) + terminal) < 1e-8 * income, (x, s)
            assert abs(pv_c - (income * q[s] + premium)) < 1e-3 * income, (x, s)


# ---------------------------------------------------------------------------
# 4. General equilibrium
# ---------------------------------------------------------------------------

HORIZONS = (16, 20, 30, 40, 60, 300)


def _shock(T):
    return -0.0025 * 0.5 ** np.arange(T)


@pytest.fixture(scope="module")
def ge_by_horizon(hh):
    """GE result and full GE paths (incl. A, B, z) for each horizon, from one Jacobian computation each."""
    out = {}
    for T in HORIZONS:
        paths = _two_asset_ge_paths(hh, T, 1.5, 0.1, _shock(T), 0.5)
        res = _two_asset_ge_solve(hh, T, 1.5, 0.1, _shock(T), 0.5)
        out[T] = (res, paths)
    return out


@pytest.mark.parametrize("T", [16, 20, 30, 40, 60, 300])
def test_rate_cut_raises_consumption_at_every_horizon(ge_by_horizon, T):
    res, paths = ge_by_horizon[T]
    assert res.converged is True
    dC, dY = res.irf_consumption, res.irf_output
    assert np.array_equal(dC, paths["C"]) and np.array_equal(dY, paths["Y"])
    assert dC[0] > 0.0 and dY[0] > 0.0
    assert res.irf_rate_b[0] < 0.0 and res.irf_rate_a[0] < 0.0
    assert res.irf_inflation[0] > 0.0
    # decay: the response dies out instead of blowing up at the truncation horizon
    assert np.all(np.abs(dC[8:]) < 0.1 * dC[0])
    assert abs(dC[-1]) < 0.01 * dC[0]


@pytest.mark.parametrize("T", [16, 40, 300])
def test_ge_total_wealth_does_not_move(ge_by_horizon, T):
    """Walras's law: household income = Y and Y = C + CHI imply d(A + B) = 0 at every date.

    With returns paid on fixed steady-state stocks instead, d(A + B) grows at rate r_b
    (3e-3 at t = 299 for this shock, 75% of the impact portfolio shift).
    """
    res, p = ge_by_horizon[T]
    assert np.max(np.abs(p["Y"] - p["C"] - p["CHI"])) < 1e-15  # goods market
    assert abs(p["A"][0]) > 1e-3  # the cut does shift the portfolio towards the illiquid asset
    assert np.max(np.abs(p["A"] + p["B"])) < 1e-8 * abs(p["A"][0])


def test_long_horizon_irfs_decay(ge_by_horizon):
    """At the SSJ horizon T = 300 the tail is numerically zero (no explosive wealth root)."""
    res, p = ge_by_horizon[300]
    dC = res.irf_consumption
    assert abs(dC[-1]) < 1e-7 * dC[0]
    assert np.max(np.abs(dC[100:])) < 1e-4 * dC[0]
    for k in ("A", "B", "z", "D"):
        assert np.max(np.abs(p[k][150:])) < 1e-4 * np.max(np.abs(p[k])), k


def test_ge_response_does_not_depend_on_the_horizon(ge_by_horizon):
    """Whole paths at T = 16..60 agree with the T = 300 solution (not only the first periods)."""
    _, ref = ge_by_horizon[300]
    for T in (16, 20, 30, 40, 60):
        _, p = ge_by_horizon[T]
        for name in ("C", "Y", "r_b", "pi", "A", "D"):
            peak = np.max(np.abs(ref[name]))
            assert np.max(np.abs(p[name] - ref[name][:T])) < 5e-3 * peak, (T, name)


def test_household_block_reproduces_ge_consumption(hh, ge_by_horizon):
    """Feed the GE prices to the non-linear closed household block: it does what the GE solve says.

    Non-financial income is re-solved non-linearly from z_t N = Y_t - r^b_t B_{t-1} - r^a_t A_{t-1}
    on the simulated holdings; consumption, spending and total wealth match the linear GE paths.
    """
    res, p = ge_by_horizon[40]
    scale = 0.1  # small shock so that second-order terms are negligible
    dY, drb, dra = scale * res.irf_output, scale * res.irf_rate_b, scale * res.irf_rate_a
    rb = hh.r_b_ss + np.concatenate([[0.0], drb[:-1]])  # ex-ante rate at t-1 is realised at t
    ra = hh.r_a_ss + np.concatenate([[0.0], dra[:-1]])
    agg = _two_asset_closed_transition(hh, hh.Y_ss + dY, rb, ra)
    dC_direct = (agg["C"] - hh.C_ss) / scale
    dQ_direct = (agg["C"] + agg["CHI"] - hh.Y_ss) / scale
    dW_direct = (agg["A"] + agg["B"] - hh.A_ss - hh.B_ss) / scale
    dz_direct = (agg["z"] - hh.w_ss) / scale
    peak = np.max(np.abs(res.irf_consumption))
    assert np.max(np.abs(dC_direct - res.irf_consumption)) < 5e-3 * peak
    assert np.max(np.abs(dQ_direct - res.irf_output)) < 5e-3 * peak  # goods market clears
    assert np.max(np.abs(dz_direct - p["z"])) < 5e-3 * np.max(np.abs(p["z"]))
    assert np.max(np.abs(dW_direct)) < 5e-3 * abs(p["A"][0])  # total wealth stays put (second order)


# ---------------------------------------------------------------------------
# 5. .mod bridge and standalone API
# ---------------------------------------------------------------------------

def test_two_asset_mod_file_parsing():
    model = load_hank_mod(MOD)
    assert model.is_two_asset is True
    assert model.hetagent_config["model"] == "hank_two_asset"
    assert int(float(model.hetagent_config["assets"])) == 2
    assert model.hetagent_config["liquid_asset"] == "b"
    assert model.hetagent_config["illiquid_asset"] == "a"
    assert model.beta * (1.0 + model.r_a_ss) < 1.0
    assert len(model.asset_grid) == 25
    assert model.liquid_asset_grid is not None and len(model.liquid_asset_grid) == 25
    assert model.joint_distribution is not None and model.joint_distribution.shape == (25, 25)
    assert np.sum(model.joint_distribution) == pytest.approx(1.0, abs=1e-12)
    assert model.steady_state["mass_at_a_max"] < 1e-10
    # genuine MPCs (conditional on illiquid wealth), not a placeholder ramp
    assert model.mpc_distribution is not None and model.mpc_distribution.shape == (25,)
    assert np.all((model.mpc_distribution > 0.0) & (model.mpc_distribution < 1.0))


@pytest.mark.parametrize("horizon", [16, 20, 60])
def test_solve_hank_bridge_two_asset_end_to_end(horizon):
    res = solve_hank_bridge(MOD, shock="eps_m", magnitude=-0.0025, horizon=horizon)
    assert isinstance(res, HANKResult)
    assert res.converged is True
    assert res.horizon == horizon
    assert res.liquid_asset_grid is not None
    assert res.joint_distribution is not None
    assert res.marginal_distribution_b is not None
    for col in ("Y", "C", "CHI", "D", "r_b", "r_a", "pi", "i"):
        assert col in res.transition_paths.columns
    p = res.transition_paths
    # monetary easing: the real rate falls, output, consumption and inflation rise
    assert p["r_b"].iloc[0] < 0.0
    assert p["Y"].iloc[0] > 0.0
    assert p["C"].iloc[0] > 0.0
    assert p["pi"].iloc[0] > 0.0
    assert abs(p["C"].iloc[-1]) < 0.05 * p["C"].iloc[0]
    assert np.allclose(p["Y"], p["C"] + p["CHI"])

    df_sum = res.summary()
    assert isinstance(df_sum, pd.DataFrame)
    assert "impact_response" in df_sum.columns
    assert "| Y" in res.to_markdown() or "| variable" in res.to_markdown()
    assert r"\begin{tabular}" in res.to_latex()
    assert "#table(" in res.to_typst()
    fig_tr, _ = res.plot_transition()
    assert fig_tr is not None
    fig_dist, _ = res.plot_distribution()
    assert fig_dist is not None


def test_bridge_matches_standalone_solver():
    res_b = solve_hank_bridge(MOD, shock="eps_m", magnitude=-0.0025, horizon=30)
    res_s = solve_two_asset_hank_sequence_space(T=30)
    assert np.allclose(res_b.transition_paths["C"].to_numpy(), res_s.irf_consumption, rtol=1e-10, atol=1e-14)


def test_solve_two_asset_hank_standalone():
    res = solve_two_asset_hank_sequence_space(T=20, n_a=15, n_b=15)
    assert isinstance(res, TwoAssetSequenceSpaceHANKResult)
    assert res.horizon == 20
    assert res.converged is True
    assert res.irf_consumption[0] > 0.0
    assert len(res.irf_output) == 20 and len(res.irf_deposit) == 20
    assert res.jacobian_c_rb.shape == (20, 20) and res.jacobian_d_rb.shape == (20, 20)
    assert res.steady_state["hh_residual"] < 1e-10
    assert abs(res.steady_state["budget_residual"]) < 1e-12
    assert "Two-Asset Sequence-Space HANK" in res.summary()
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame) and len(df) == 20
    fig = res.plot()
    assert fig is not None
