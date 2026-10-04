"""Regression tests for the 2026-09 audit of the trade solver/accounting stack.

Each test pins one confirmed finding (cluster SOLVER): scaled finite-difference
steps for value-unit unknowns, the consistent-mode stopping rule, the
``ViabilityResult`` contract, tariff conventions of the Hawkins-Simon check,
the ``eps`` shift, the policy pre-screen, the NaN guard of damped Newton, the
scalar Keller convention, the foreign-saving unit closure, welfare state
compatibility and the runnable documentation examples.
"""
from __future__ import annotations

import contextlib
import io
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import puremacro.trade.policy_solver as policy_mod
import puremacro.trade.solver as solver_mod
from puremacro.trade import (
    PolicyEquilibriumError,
    calibrate_trade_model,
    compute_hicksian_welfare,
    solve_policy_equilibrium,
    solve_trade_equilibrium,
)
from puremacro.trade._accounting import evaluate
from puremacro.trade._oecd_icio import condense_final_demand
from puremacro.trade.data import generate_synthetic_mrio, package_mrio_to_calibration_result
from puremacro.trade.equilibrium import compute_equilibrium_residuals
from puremacro.trade.solver import (
    ViabilityResult,
    _consistent_fd_scale,
    _damped_newton_solve,
    _fd_jacobian_dense,
    _fd_steps,
    _resolve_tariffs,
    build_initial_guess,
    check_hawkins_simon_viability,
    solve_keller_pac,
)
from tools.reference_validation.validate_oecd import load_fixture
from tools.reference_validation.validate_trade_accounting import benchmark_table
from tools.reference_validation.validate_trade_welfare import consumption_benchmark

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _scaled(raw, s, unit="USD"):
    tfd = raw.taxes_less_subsidies_fd
    return replace(raw, intermediate_matrix=raw.Z * s, final_demand_matrix=raw.F * s,
                   value_added=raw.VA * s, taxes_less_subsidies=raw.TLS * s,
                   gross_output=raw.gross_output * s,
                   taxes_less_subsidies_fd=None if tfd is None else np.asarray(tfd) * s, unit=unit)


@pytest.fixture(scope="module")
def oecd_calib():
    """Frozen OECD 2019 3-region x 3-sector fixture in million USD."""
    return package_mrio_to_calibration_result(condense_final_demand(load_fixture()))


@pytest.fixture(scope="module")
def oecd_usd_calib():
    """The same fixture in USD (x1e6). The calibration's absolute 1e-3 budget
    check rejects USD magnitudes, so it is built with ``validate=False``."""
    return package_mrio_to_calibration_result(condense_final_demand(_scaled(load_fixture(), 1e6)),
                                              validate=False)


@pytest.fixture(scope="module")
def calib_2c2s():
    """Balanced 2-country x 2-sector table with three final uses (reviewer fixture)."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc))
    data[:4, :4] = np.array([[20., 10., 5., 2.], [8., 25., 2., 4.], [5., 5., 12., 18.], [10., 10., 18., 22.]])
    y = np.array([100., 150., 120., 180.])
    va = y - data[:4, :4].sum(0)
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2 / 3) * va_fac
    data[6, :4] = (1 / 3) * va_fac
    fd_row = y - data[:4, :4].sum(1)
    for i in range(4):
        t = fd_row[i]
        data[i, 4:10] = ([t * .5, t * .25, t * .05, t * .1, t * .08, t * .02] if i < 2
                         else [t * .1, t * .08, t * .02, t * .5, t * .25, t * .05])
    data[4, 4:] = 0.02 * data[:4, 4:].sum(0)
    return calibrate_trade_model(data, ns=ns, nc=nc, nfd=nfd, validate=True)


@pytest.fixture(scope="module")
def bench():
    """Two-country consumption-welfare reference table (tools/reference_validation)."""
    data, tau, fd = consumption_benchmark()
    return calibrate_trade_model(data, ns=1, nc=2, nfd=3, country_codes=["A", "B"]), tau, fd


def _usa_rates(calib, rate=.1):
    t = np.zeros(calib.nc)
    t[calib.country_codes.index("USA")] = rate
    return t


def _phys(result):
    return float(np.max(np.abs(result.metadata["physical_residuals"])))


# ---------------------------------------------------------------------------
# 1. Scaled finite-difference steps (consistent mode)
# ---------------------------------------------------------------------------
def test_fd_steps_follow_the_natural_scale_of_each_unknown(oecd_usd_calib):
    calib = oecd_usd_calib
    x = build_initial_guess(calib)
    scale = _consistent_fd_scale(calib, len(x))
    M, nc = calib.ns * calib.nc, calib.nc
    money = float(np.median(np.abs(calib.l_endow.ravel() + calib.k_endow.ravel())))
    np.testing.assert_array_equal(scale[:2 * M + 2 * nc], 1.0)
    np.testing.assert_array_equal(scale[2 * M + 2 * nc:], money)
    h = _fd_steps(x, 1e-5, scale)
    np.testing.assert_allclose(h, 1e-5 * np.maximum(np.abs(x), scale), rtol=1e-10)
    # Exactly representable increments; an absolute 1e-5 step on T ~ 1e12 is lost.
    np.testing.assert_array_equal((x + h) - x, h)
    t_idx = 2 * M + 2 * nc
    assert (x[t_idx] + 1e-5) - x[t_idx] == 0.0


def test_transfer_and_balance_columns_match_central_differences_at_usd_scale(oecd_usd_calib):
    """Finding: fixed absolute FD step on value-unit unknowns (consistent mode)."""
    calib = oecd_usd_calib
    f = lambda xx: compute_equilibrium_residuals(xx, calib, accounting="consistent")  # noqa: E731
    x = build_initial_guess(calib)
    f0 = f(x)
    M, nc = calib.ns * calib.nc, calib.nc
    scaled = _fd_jacobian_dense(f, x, f0, 1e-5, _consistent_fd_scale(calib, len(x)))
    absolute = _fd_jacobian_dense(f, x, f0, 1e-5, None)
    for j in range(2 * M + 2 * nc, len(x)):
        step = 1e-3 * abs(x[j])
        xp, xm = x.copy(), x.copy()
        xp[j] += step
        xm[j] -= step
        ref = (f(xp) - f(xm)) / (xp[j] - xm[j])  # residual is affine in T and XN
        scale = float(np.max(np.abs(ref)))
        assert np.max(np.abs(scaled[:, j] - ref)) <= 1e-6 * scale
        # The historical absolute step is below ulp(x_j): the column is garbage.
        assert np.max(np.abs(absolute[:, j] - ref)) > 1e-2 * scale


def test_usd_scale_oecd_counterfactual_converges_only_with_scaled_steps(oecd_usd_calib, monkeypatch):
    calib = oecd_usd_calib
    t = _usa_rates(calib)
    tol = 10.0  # ten dollars, the 1e-5 M-USD tolerance of validate_oecd in USD
    base = solve_trade_equilibrium(calib, accounting="consistent", tol=tol)
    fixed = solve_trade_equilibrium(calib, tau=t, tau_fd=t, accounting="consistent", tol=tol,
                                    x0=base.x_sol, max_iter=30)
    assert fixed.converged, (fixed.max_residual, fixed.metadata["account_residuals"])
    assert fixed.max_residual <= tol and _phys(fixed) <= tol
    assert fixed.metadata["fd_step"] == "scaled"
    # Pre-fix behaviour: absolute steps (the scale helper disabled).
    monkeypatch.setattr(solver_mod, "_consistent_fd_scale", lambda calib, n: None)
    stock = solve_trade_equilibrium(calib, tau=t, tau_fd=t, accounting="consistent", tol=tol,
                                    x0=base.x_sol, max_iter=30)
    assert not stock.converged
    # Same equilibrium in M USD (the unit-invariant check of validate_oecd).
    musd = package_mrio_to_calibration_result(condense_final_demand(load_fixture()))
    ref = solve_trade_equilibrium(musd, tau=t, tau_fd=t, accounting="consistent", tol=1e-6)
    np.testing.assert_allclose(fixed.p_sol, ref.p_sol, rtol=1e-8)
    np.testing.assert_allclose(fixed.y_sol * 1e-6, ref.y_sol, rtol=1e-8)


def test_seeded_20x11_nonnegative_table_in_usd_converges():
    raw = generate_synthetic_mrio("oecd", custom_c=20, custom_s=11, seed=11)
    raw = replace(raw, taxes_less_subsidies_fd=np.zeros(raw.C * raw.K_F))
    assert np.min(raw.Z) >= 0 and np.min(raw.F) >= 0
    calib = package_mrio_to_calibration_result(condense_final_demand(_scaled(raw, 1e6)), validate=False)
    tol = 10.0
    base = solve_trade_equilibrium(calib, accounting="consistent", tol=tol)
    assert base.converged
    t = np.zeros(calib.nc)
    t[0] = .1
    cf = solve_trade_equilibrium(calib, tau=t, tau_fd=t, accounting="consistent", tol=tol,
                                 x0=base.x_sol, max_iter=12)
    assert cf.converged, cf.metadata["solver_residual_history"]
    assert cf.iterations <= 8 and _phys(cf) <= tol


def _legacy_reference_newton(f, x_init, tol, max_iter, eps_fd):
    """Verbatim 4.3.0 _damped_newton_solve (pre-audit) for the bit-identity check."""
    x = np.asarray(x_init, dtype=float).copy()
    n = len(x)
    f_val = f(x)
    max_res = float(np.max(np.abs(f_val)))
    diff = float(np.sum(np.abs(f_val)))
    if max_res <= tol or diff <= tol:
        return x, True, 0, max_res, diff, f_val
    for it in range(max_iter):
        J = np.empty((n, n), dtype=float)
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += eps_fd
            J[:, j] = (f(x_pert) - f_val) / eps_fd
        try:
            delta = np.linalg.solve(J, -f_val)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, -f_val, rcond=1e-10)[0]
        xscal = 0.5 if it < 3 else 1.0
        alpha_step = xscal
        norm_0 = max_res
        x_trial = x + alpha_step * delta
        f_trial = f(x_trial)
        for _ in range(10):
            res_trial_max = float(np.max(np.abs(f_trial)))
            res_trial_diff = float(np.sum(np.abs(f_trial)))
            if res_trial_max <= tol or res_trial_diff <= tol or res_trial_max < norm_0:
                break
            alpha_step *= 0.5
            x_trial = x + alpha_step * delta
            f_trial = f(x_trial)
        x = x_trial
        f_val = f_trial
        max_res = float(np.max(np.abs(f_val)))
        diff = float(np.sum(np.abs(f_val)))
        if max_res <= tol or diff <= tol:
            return x, True, it + 1, max_res, diff, f_val
    return x, False, max_iter, max_res, diff, f_val


def test_legacy_newton_numerics_are_bit_identical(calib_2c2s):
    ta, tf, tv, tfv = _resolve_tariffs(calib_2c2s, tau=.25, tau_fd=.25)
    f = lambda xx: compute_equilibrium_residuals(xx, calib_2c2s, tau=ta, tau_fd=tf, tauf=tv, tauf_fd=tfv)  # noqa: E731
    x0 = build_initial_guess(calib_2c2s)
    new = _damped_newton_solve(f, x0, tol=1e-12, max_iter=40, eps_fd=1e-2)
    old = _legacy_reference_newton(f, x0, tol=1e-12, max_iter=40, eps_fd=1e-2)
    np.testing.assert_array_equal(new[0], old[0])
    assert new[1:5] == old[1:5]
    result = solve_trade_equilibrium(calib_2c2s, tau=ta, tau_fd=tf, tauf=tv, tauf_fd=tfv, tol=1e-12, max_iter=40)
    np.testing.assert_array_equal(result.x_sol, old[0])


# ---------------------------------------------------------------------------
# 2. Consistent stopping rule = postprocessing acceptance audit
# ---------------------------------------------------------------------------
def _assert_flag_matches_audit(result, tol):
    audit = bool(_phys(result) <= tol and result.max_residual <= tol
                 and result.metadata["demand_feasible"])
    assert result.converged == audit
    assert result.converged, (tol, result.max_residual, _phys(result), result.metadata.get("solver_termination"))


@pytest.mark.parametrize("tol", [1e-4, 1e-5, 1e-6, 1e-7])
def test_oecd_usa_tariff_converges_at_every_attainable_tolerance(oecd_calib, tol):
    """Finding: Newton stopped at 8.3e-6 (tol 1e-5) and postprocessing rejected 1.03e-5."""
    t = _usa_rates(oecd_calib)
    result = solve_trade_equilibrium(oecd_calib, tau=t, tau_fd=t, accounting="consistent", tol=tol)
    _assert_flag_matches_audit(result, tol)
    assert result.metadata["solver_termination"] == "tolerance"


@pytest.mark.parametrize("method", ["newton", "condensed", "sparse_lu", "broyden", "krylov"])
def test_2c2s_tol_1e8_converges_for_every_fd_solver(calib_2c2s, method):
    """Finding: at tol=1e-8 the solver stopped at 3.2e-9 with the dropped goods row at 1.005e-8."""
    result = solve_trade_equilibrium(calib_2c2s, tau=.1, tau_fd=.1, accounting="consistent",
                                     method=method, tol=1e-8, max_iter=100)
    _assert_flag_matches_audit(result, 1e-8)


@pytest.mark.parametrize("tol", [1e-6, 1e-8, 1e-10])
def test_two_country_benchmark_flag_matches_audit(tol):
    data, tau, fd = benchmark_table()
    calib = calibrate_trade_model(data, ns=1, nc=2, nfd=2, country_codes=["A", "B"], sector_codes=["GOOD"])
    for method in ("newton", "keller_pac"):
        result = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent",
                                         method=method, tol=tol, max_iter=100, max_steps=40)
        _assert_flag_matches_audit(result, tol)


@pytest.mark.parametrize("fixture, rate, method, tol", [
    ("bench", .05, "sparse_lu", 1e-7),
    ("bench", .2, "sparse_lu", 1e-8),
    ("2c2s", .1, "krylov", 1e-9),
    ("2c2s", .2, "krylov", 1e-8),
    ("bench", .1, "broyden", 1e-5),
    ("bench", .3, "broyden", 1e-7),
    ("bench", .5, "keller_pac", 3e-5),
])
def test_knife_edge_cases_pin_the_audit_stopping_rule(fixture, rate, method, tol, calib_2c2s, bench):
    """Cases in which the solved residuals meet ``tol`` before the audited physical
    equations do. Before the fix (and with ``_consistent_acceptor`` disabled) each
    returned ``converged=False`` at a state with ``max_residual <= tol``: e.g.
    bench/.05/sparse_lu/1e-7 stopped at 5.2e-8 with the physical residual at 1.2e-7,
    and bench/.5/keller_pac/3e-5 at 1.75e-5 with 4.3e-5 (the terminal polish stopped
    at ``res <= tol``)."""
    calib = calib_2c2s if fixture == "2c2s" else bench[0]
    result = solve_trade_equilibrium(calib, tau=rate, tau_fd=rate, accounting="consistent",
                                     method=method, tol=tol, max_iter=100, max_steps=40)
    _assert_flag_matches_audit(result, tol)


def test_policy_solver_accepts_newton_at_the_validation_tolerance(oecd_calib):
    t = _usa_rates(oecd_calib)
    ta, tf, _, _ = _resolve_tariffs(oecd_calib, t, t)
    result = solve_policy_equilibrium(oecd_calib, ta, tf, tol=1e-5)
    assert result.metadata["policy_solver_method"] == "newton"
    assert result.metadata["policy_solver_fallback_used"] is False


# ---------------------------------------------------------------------------
# 3. ViabilityResult is a plain named tuple
# ---------------------------------------------------------------------------
def _three(rho, lo, hi, ok):
    return rho, lo, hi


def test_viability_result_iterates_as_four_regardless_of_context(calib_2c2s):
    r = ViabilityResult(0.5, 0.4, 0.6, True)
    assert ViabilityResult.__iter__ is tuple.__iter__
    assert ViabilityResult._fields == ("rho", "cw_lower", "cw_upper", "is_viable")
    listed = list(r)
    a, b, c = 1, 2, 3  # a 3-unpack right after list(r) used to truncate it
    assert len(listed) == 4 and (a, b, c) == (1, 2, 3)
    arr = np.asarray(r)
    x1, x2, x3 = _three(*r)  # star-args followed by a 3-unpack used to pass 3 args
    assert arr.shape == (4,) and (x1, x2, x3) == (0.5, 0.4, 0.6)
    assert tuple(r) == (0.5, 0.4, 0.6, True)
    rho, lo, hi, ok = r
    assert (rho, lo, hi, ok) == (r.rho, r.cw_lower, r.cw_upper, r.is_viable)
    real = check_hawkins_simon_viability(calib_2c2s, tau=.2)
    assert isinstance(real, ViabilityResult) and len(list(real)) == 4
    assert np.asarray(real).shape == (4,)
    assert "ViabilityResult" in solver_mod.__all__


# ---------------------------------------------------------------------------
# 4. One tariff convention for the check and the solver
# ---------------------------------------------------------------------------
def _true_rho(calib, tau_a):
    M = calib.ns * calib.nc
    a2 = calib.a.reshape((M, M), order="F")
    t = np.clip(np.asarray(calib.tax, dtype=float).flatten(order="F"), -.9, .999)
    B = a2 * tau_a.reshape((M, M), order="F") / (1 - t)[None, :]
    return float(np.max(np.abs(np.linalg.eigvals(B))))


def test_array_schedules_are_multipliers_as_in_resolve_tariffs(calib_2c2s):
    c = calib_2c2s
    ns, nc, M = c.ns, c.nc, c.ns * c.nc
    three = np.ones((M, ns, nc)) * 1.10
    r3 = check_hawkins_simon_viability(c, tau=three)
    r2 = check_hawkins_simon_viability(c, tau=three.reshape((M, M), order="F"))
    r4 = check_hawkins_simon_viability(c, tau=np.ones((ns, nc, ns, nc)) * 1.10)
    solver_view = _resolve_tariffs(c, three, None)[0]
    assert solver_view.max() == pytest.approx(1.10)
    expected = _true_rho(c, solver_view)
    for r in (r2, r3, r4):
        assert r.cw_lower - 1e-10 <= expected <= r.cw_upper + 1e-10
        assert r.rho == pytest.approx(expected, rel=1e-8)
    # Scalar and (nc,) inputs keep their RATE semantics, identical to the solver.
    rs = check_hawkins_simon_viability(c, tau=.1)
    rv = check_hawkins_simon_viability(c, tau=np.full(nc, .1))
    rho_rate = _true_rho(c, _resolve_tariffs(c, .1, None)[0])
    assert rs.rho == pytest.approx(rho_rate, rel=1e-8) and rv.rho == pytest.approx(rho_rate, rel=1e-8)
    # A zero-diagonal rate matrix is read as multipliers, exactly as the solver reads it.
    rate2 = np.zeros((M, M))
    rate2[:ns, ns:] = rate2[ns:, :ns] = .5
    r_rate = check_hawkins_simon_viability(c, tau=rate2)
    assert r_rate.rho == pytest.approx(_true_rho(c, _resolve_tariffs(c, rate2, None)[0]), rel=1e-8, abs=1e-12)
    with pytest.raises(ValueError, match="not a supported schedule"):
        check_hawkins_simon_viability(c, tau=np.ones((3, 5)))


def test_list_and_zero_dim_inputs_follow_the_array_conventions(calib_2c2s):
    """4.3.0 read any non-ndarray (a list) as the baseline and a 0-d array as a
    whole-matrix multiplier; both now follow ``_resolve_tariffs``."""
    c = calib_2c2s
    base = check_hawkins_simon_viability(c)
    as_list = check_hawkins_simon_viability(c, tau=[.5, .5])
    as_array = check_hawkins_simon_viability(c, tau=np.array([.5, .5]))
    assert as_list.rho == as_array.rho and as_list.rho > base.rho + 1e-3
    assert check_hawkins_simon_viability(c, tau=np.array(.1)).rho == check_hawkins_simon_viability(c, tau=.1).rho
    with pytest.raises(ValueError, match="not a supported schedule"):
        check_hawkins_simon_viability(c, tau=[[.1]])


# ---------------------------------------------------------------------------
# 5. eps is the power-iteration shift
# ---------------------------------------------------------------------------
def test_eps_is_forwarded_as_the_shift(calib_2c2s, monkeypatch):
    import puremacro.trade.regularize as reg

    import inspect

    seen = []
    original = reg.compute_spectral_radius
    upstream_shift = "shift" in inspect.signature(original).parameters

    def recording(B, max_iter=300, tol=1e-12, *, shift=1e-3):
        seen.append(shift)
        extra = {"shift": shift} if upstream_shift else {}
        return original(B, max_iter=max_iter, tol=tol, **extra)

    monkeypatch.setattr(reg, "compute_spectral_radius", recording)
    check_hawkins_simon_viability(calib_2c2s, tau=.3, eps=0.02)
    assert seen and all(s == 0.02 for s in seen)
    with pytest.raises(ValueError, match="eps"):
        check_hawkins_simon_viability(calib_2c2s, tau=.3, eps=0.0)
    doc = check_hawkins_simon_viability.__doc__ + solver_mod.__doc__
    assert "< 0.15" not in doc and "in < 0.15s" not in doc


def test_local_shift_fallback_gives_valid_bounds(calib_2c2s, monkeypatch):
    """Without a ``shift`` keyword upstream, a local shifted iteration is used."""
    import puremacro.trade.regularize as reg

    original = reg.compute_spectral_radius

    def no_shift(B, max_iter=300, tol=1e-12):
        return original(B, max_iter=max_iter, tol=tol)

    monkeypatch.setattr(reg, "compute_spectral_radius", no_shift)
    r = check_hawkins_simon_viability(calib_2c2s, tau=.3)
    expected = _true_rho(calib_2c2s, _resolve_tariffs(calib_2c2s, .3, None)[0])
    assert r.cw_lower - 1e-10 <= expected <= r.cw_upper + 1e-10
    assert r.cw_upper - r.cw_lower <= 1e-8


def test_bundled_77x11_threshold_is_certified_on_both_sides():
    from puremacro.trade import load_icio_data
    from puremacro.trade.solver import _hawkins_simon_certificate

    calib = calibrate_trade_model(load_icio_data(source="legacy"))
    assert _hawkins_simon_certificate(calib, 3.6)[3] == "viable"
    assert _hawkins_simon_certificate(calib, 3.8)[3] == "violated"
    with pytest.raises(ValueError, match="violates"):
        check_hawkins_simon_viability(calib, tau=10.0)


# ---------------------------------------------------------------------------
# 6. Recovery chain: Hawkins-Simon pre-screen before Newton and hybrid
# ---------------------------------------------------------------------------
def test_certified_violation_raises_before_any_solver(bench, monkeypatch):
    calib, _, _ = bench
    ta, tf, _, _ = _resolve_tariffs(calib, tau=9.0, tau_fd=9.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("no solver may run for a certified Hawkins-Simon violation")

    monkeypatch.setattr(policy_mod, "solve_trade_equilibrium", forbidden)
    with pytest.raises(PolicyEquilibriumError, match="Hawkins-Simon") as info:
        solve_policy_equilibrium(calib, ta, tf)
    assert info.value.attempts == []
    assert info.value.viability["status"] == "violated"
    assert info.value.viability["cw_lower"] >= 1 - 1e-6


def test_unresolved_or_viable_certificate_proceeds_and_is_recorded(bench, monkeypatch):
    calib, tau, fd = bench
    ok = solve_policy_equilibrium(calib, tau, fd, tol=1e-10)
    assert ok.metadata["policy_viability"]["status"] == "viable"
    ces = solve_policy_equilibrium(calib, tau, fd, tol=1e-10, sigma=.5)
    assert ces.metadata["policy_viability"]["status"] == "not_applicable"
    monkeypatch.setattr(policy_mod, "_hawkins_simon_certificate",
                        lambda calib, ta, **kw: (0.999, 0.99, 1.01, "unresolved"))
    unresolved = solve_policy_equilibrium(calib, tau, fd, tol=1e-10)
    assert unresolved.converged
    assert unresolved.metadata["policy_viability"]["status"] == "unresolved"


def _rate_at_spectral_radius(calib, target):
    """Uniform import-tariff rate at which rho(B_tau) (exact 1 - t) equals ``target``."""
    def rho(rate):
        ta = _resolve_tariffs(calib, rate, None)[0]
        B = solver_mod._viability_cost_matrix(calib, ta, exact_tax=True)
        return float(np.max(np.abs(np.linalg.eigvals(B))))
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rho(mid) < target else (lo, mid)
    return lo


def _count_attempts(monkeypatch):
    calls = []

    def failing(*args, **kwargs):
        calls.append(kwargs.get("method"))
        raise ValueError("stub solver")

    monkeypatch.setattr(policy_mod, "solve_trade_equilibrium", failing)
    return calls


def test_near_critical_schedule_is_not_reported_as_nonexistent(bench, monkeypatch):
    """rho in [1 - 1e-6, 1): a positive price system exists; the chain must run.
    Before the fix the screen called it 'violated' and raised 'no positive price
    system exists' with no attempt (bench at rate 8.4648: bounds 0.9999995)."""
    calib = bench[0]
    rate = _rate_at_spectral_radius(calib, 1 - 5e-7)
    ta, tf, _, _ = _resolve_tariffs(calib, rate, None)
    calls = _count_attempts(monkeypatch)
    with pytest.raises(PolicyEquilibriumError, match="All policy equilibrium attempts failed") as info:
        solve_policy_equilibrium(calib, ta, tf)
    assert calls == ["newton", "hybr", "keller_pac"] and len(info.value.attempts) == 3
    v = info.value.viability
    assert v["status"] == "near_critical"
    assert 1 - 1e-6 <= v["cw_lower"] <= v["cw_upper"] < 1.0


def test_screen_uses_the_exact_production_tax_wedge(calib_2c2s, monkeypatch):
    """Production subsidies above 90% (t < -0.9) are valid in consistent accounting.
    The 4.3.0 clip of t to [-0.9, 0.999] overstated B_tau (rho 1.123 instead of
    0.853 at rate 25) and the screen raised before any solver."""
    calib = replace(calib_2c2s, tax=np.full_like(calib_2c2s.tax, -1.5))
    ta, tf, _, _ = _resolve_tariffs(calib, 25.0, None)
    exact = float(np.max(np.abs(np.linalg.eigvals(solver_mod._viability_cost_matrix(calib, ta, exact_tax=True)))))
    clipped = float(np.max(np.abs(np.linalg.eigvals(solver_mod._viability_cost_matrix(calib, ta)))))
    assert exact == pytest.approx(0.8533662979916653, rel=1e-9) and clipped > 1.1
    screen = policy_mod._viability_screen(calib, ta, 0.0)
    assert screen["status"] == "viable" and screen["cw_lower"] - 1e-12 <= exact <= screen["cw_upper"] + 1e-12
    calls = _count_attempts(monkeypatch)
    with pytest.raises(PolicyEquilibriumError, match="All policy equilibrium attempts failed"):
        solve_policy_equilibrium(calib, ta, tf)
    assert calls == ["newton", "hybr", "keller_pac"]


# ---------------------------------------------------------------------------
# 7. Non-finite Jacobians end Newton without a numpy traceback
# ---------------------------------------------------------------------------
def _nan_above_one(x):
    out = np.array([x[0] - 2.0, 0.0 * x[1]])
    if x[0] > 1.0:
        out[1] = np.nan
    return out


def _sqrt_domain(x):
    return np.array([np.sqrt(1.0 - x[0]) - 2.0, x[1] ** 2 + 1.0])


@pytest.mark.parametrize("f, x0", [(_nan_above_one, [1.0, 0.0]), (_sqrt_domain, [0.999, 1.0])])
def test_nan_jacobian_returns_non_converged_at_a_finite_iterate(f, x0):
    info = {}
    with np.errstate(all="ignore"):
        x, conv, iters, max_res, diff, f_val = _damped_newton_solve(
            f, np.asarray(x0, dtype=float), tol=1e-10, max_iter=20, eps_fd=1e-2, info=info)
    assert conv is False
    assert np.isfinite(x).all() and np.isfinite(max_res)
    assert info["termination"] in ("non_finite_jacobian", "non_finite_residual", "linear_solve_failed")
    assert info["residual_history"] and np.isfinite(info["residual_history"]).all()


def test_cold_start_prohibitive_consistent_newton_does_not_raise(bench):
    calib, _, _ = bench
    ta, tf, _, _ = _resolve_tariffs(calib, tau=5.0, tau_fd=5.0)
    with np.errstate(all="ignore"):
        result = solve_trade_equilibrium(calib, tau=ta, tau_fd=tf, method="newton",
                                         accounting="consistent", tol=1e-8, max_iter=200)
    assert result.converged is False
    assert np.isfinite(result.x_sol).all()
    assert result.metadata["solver_termination"] in (
        "max_iter", "non_finite_jacobian", "non_finite_residual", "linear_solve_failed")


@pytest.mark.filterwarnings("ignore::scipy.sparse.linalg.MatrixRankWarning")
def test_non_finite_iterate_is_recorded_as_such(calib_2c2s):
    """Sparse LU can exhaust max_iter at a non-finite state (rate 9, legacy)."""
    with np.errstate(all="ignore"):
        result = solve_trade_equilibrium(calib_2c2s, tau=9.0, tau_fd=9.0, method="sparse_lu",
                                         tol=1e-8, max_iter=50)
    assert result.converged is False
    assert not np.isfinite(result.x_sol).all()
    assert result.metadata["solver_termination"] == "non_finite_iterate"


# ---------------------------------------------------------------------------
# 8. Scalar Keller target: explicit final-demand convention
# ---------------------------------------------------------------------------
def test_scalar_keller_target_convention_is_explicit(calib_2c2s):
    c = calib_2c2s
    both = solve_keller_pac(c, tau_target=.3, tol=1e-10)
    newton_both = solve_trade_equilibrium(c, tau=.3, tau_fd=.3, tol=1e-10)
    newton_inter = solve_trade_equilibrium(c, tau=.3, tol=1e-10)
    assert both.converged and newton_both.converged and newton_inter.converged
    assert np.max(np.abs(both.x_sol - newton_both.x_sol)) < 1e-8
    assert "final-demand" in both.metadata["scalar_tariff_convention"]
    inter = solve_keller_pac(c, tau_target=.3, tol=1e-10, scalar_applies_to_final_demand=False)
    assert inter.converged
    assert np.max(np.abs(inter.x_sol - newton_inter.x_sol)) < 1e-8
    assert "intermediate imports only" in inter.metadata["scalar_tariff_convention"]
    # The dispatch path passes explicit arrays and mirrors solve_trade_equilibrium.
    dispatched = solve_trade_equilibrium(c, tau=.3, method="keller_pac", tol=1e-10)
    assert np.max(np.abs(dispatched.x_sol - newton_inter.x_sol)) < 1e-8
    assert dispatched.metadata["scalar_tariff_convention"].startswith("not applicable")


@pytest.mark.parametrize("target", [
    0.3, np.float64(0.3), np.float32(0.3), np.array(0.3), 1, np.int64(1), True, np.bool_(True), np.array(1),
], ids=lambda v: f"{type(v).__module__}.{type(v).__name__}({float(v):g})")
def test_every_zero_dim_keller_target_uses_the_scalar_convention(calib_2c2s, target):
    """One convention for every 0-d real input. 4.3.0 applied the scalar
    convention to Python int/float/bool and np.float64 only; np.float32, np.int64,
    np.bool_ and 0-d arrays tariffed intermediates only (np.int64(1) on this
    fixture: x_sol[:4] = [0.0965, -0.0050, 0.1579, 0.1302] instead of
    [0.2008, 0.0809, 0.3446, 0.3215])."""
    c = calib_2c2s
    v = float(target)
    both = solve_trade_equilibrium(c, tau=v, tau_fd=v, tol=1e-10)
    result = solve_keller_pac(c, tau_target=target, tol=1e-10)
    assert result.converged and both.converged
    assert np.max(np.abs(result.x_sol - both.x_sol)) < 1e-8
    assert "intermediate and final-demand" in result.metadata["scalar_tariff_convention"]
    inter = solve_keller_pac(c, tau_target=target, tol=1e-10, scalar_applies_to_final_demand=False)
    assert "intermediate imports only" in inter.metadata["scalar_tariff_convention"]
    assert np.max(np.abs(inter.x_sol - solve_trade_equilibrium(c, tau=v, tol=1e-10).x_sol)) < 1e-8


# ---------------------------------------------------------------------------
# 9. Foreign-saving units
# ---------------------------------------------------------------------------
def _permute_table(d, ns, nc, nfd, perm):
    n = ns * nc
    rows = np.r_[np.concatenate([np.arange(p * ns, (p + 1) * ns) for p in perm]), n, n + 1, n + 2]
    cols = np.r_[np.concatenate([np.arange(p * ns, (p + 1) * ns) for p in perm]),
                 n + np.concatenate([np.arange(p * nfd, (p + 1) * nfd) for p in perm])]
    return d[np.ix_(rows, cols)]


def _two_orders(units):
    data, tau, fd = consumption_benchmark()
    out = []
    for perm, codes in (([0, 1], ["A", "B"]), ([1, 0], ["B", "A"])):
        c = calibrate_trade_model(_permute_table(data, 1, 2, 3, perm), ns=1, nc=2, nfd=3, country_codes=codes)
        t, f = tau[perm][:, :, perm], fd[perm][:, :, perm]
        kw = {} if units is None else {"foreign_saving_units": units}
        b = solve_trade_equilibrium(c, accounting="consistent", tol=1e-10, **kw)
        x = solve_trade_equilibrium(c, tau=t, tau_fd=f, accounting="consistent", tol=1e-10, **kw)
        out.append({code: compute_hicksian_welfare(c, x, base_result=b, target_country=code,
                                                   consumption_categories=(0, 2)).ev_pct_consumption
                    for code in "AB"})
    return out


def test_world_income_foreign_saving_is_invariant_to_country_order():
    first, second = _two_orders("world_income")
    for code in "AB":
        assert abs(first[code] - second[code]) <= 1e-9
    # The default numeraire closure is order-dependent (documented).
    n1, n2 = _two_orders(None)
    assert abs(n1["A"] - n2["A"]) > 1.0


def test_numeraire_default_reproduces_the_4_3_0_closure(bench, calib_2c2s):
    calib, tau, fd = bench
    rng = np.random.default_rng(3)
    x = build_initial_guess(calib) + rng.normal(scale=1e-3, size=len(build_initial_guess(calib)))
    default = evaluate(x, calib, tau, fd, None, None)
    explicit = evaluate(x, calib, tau, fd, None, None, foreign_saving_units="numeraire")
    for key, value in default.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, explicit[key])
    nc = calib.nc
    balances = default["residuals"][-2 * nc + 1:-nc]
    np.testing.assert_array_equal(balances, x[len(x) - (nc - 1):] - np.asarray(calib.invforT).ravel()[:nc - 1])
    # Welfare numbers recorded with the pre-audit 4.3.0 code (2026-09-23).
    n1, n2 = _two_orders(None)
    assert n1["A"] == pytest.approx(-27.554287549035585, rel=1e-9)
    assert n2["A"] == pytest.approx(-45.99164907019217, rel=1e-9)
    assert n1["B"] == pytest.approx(25.711306631120145, rel=1e-9)
    with pytest.raises(ValueError, match="consistent"):
        solve_trade_equilibrium(calib_2c2s, foreign_saving_units="world_income")
    with pytest.raises(ValueError, match="foreign_saving_units"):
        solve_trade_equilibrium(calib, accounting="consistent", foreign_saving_units="usd")


def test_world_income_oecd_three_region_permutation(oecd_calib):
    raw = condense_final_demand(load_fixture())
    S, K = raw.S, raw.K_F
    evs = []
    for perm in ([0, 1, 2], [2, 1, 0]):
        node = np.concatenate([np.arange(p * S, (p + 1) * S) for p in perm])
        fdc = np.concatenate([np.arange(p * K, (p + 1) * K) for p in perm])
        VA = raw.VA[:, node] if np.ndim(raw.VA) == 2 else raw.VA[node]
        permuted = replace(raw, countries=[raw.countries[p] for p in perm],
                           intermediate_matrix=raw.Z[np.ix_(node, node)],
                           final_demand_matrix=raw.F[np.ix_(node, fdc)], value_added=VA,
                           taxes_less_subsidies=raw.TLS[node], gross_output=raw.gross_output[node],
                           taxes_less_subsidies_fd=np.asarray(raw.taxes_less_subsidies_fd)[fdc])
        c = package_mrio_to_calibration_result(permuted)
        t = _usa_rates(c)
        b = solve_trade_equilibrium(c, accounting="consistent", tol=1e-6, foreign_saving_units="world_income")
        x = solve_trade_equilibrium(c, tau=t, tau_fd=t, accounting="consistent", tol=1e-6,
                                    foreign_saving_units="world_income")
        evs.append({code: compute_hicksian_welfare(c, x, base_result=b, target_country=code).ev_pct_consumption
                    for code in ("USA", "CHN", "REST")})
    for code in ("USA", "CHN", "REST"):
        assert abs(evs[0][code] - evs[1][code]) <= 1e-9


# ---------------------------------------------------------------------------
# 10. Welfare requires states of one model
# ---------------------------------------------------------------------------
def test_welfare_rejects_states_with_different_sigma_or_closure(bench):
    calib, tau, fd = bench
    half_tau = 1 + (tau - 1) / 2
    half_fd = 1 + (fd - 1) / 2
    base0 = solve_trade_equilibrium(calib, tau=half_tau, tau_fd=half_fd, accounting="consistent", tol=1e-10)
    cf2 = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent", tol=1e-10, sigma=2.)
    with pytest.raises(ValueError, match="different sigma"):
        compute_hicksian_welfare(calib, cf2, base_result=base0, target_country="A")
    cf_world = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent", tol=1e-10,
                                       foreign_saving_units="world_income")
    with pytest.raises(ValueError, match="different foreign_saving_units"):
        compute_hicksian_welfare(calib, cf_world, base_result=base0, target_country="A")
    same = solve_trade_equilibrium(calib, tau=tau, tau_fd=fd, accounting="consistent", tol=1e-10)
    result = compute_hicksian_welfare(calib, same, base_result=base0, target_country="A")
    assert result.metadata["sigma"] == 0. and result.metadata["foreign_saving_units"] == "numeraire"


# ---------------------------------------------------------------------------
# 11. Documentation examples run verbatim
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("page", ["docs/trade_accounting.md", "docs/trade_welfare.md"])
def test_documentation_example_runs_verbatim(page):
    text = (REPO / page).read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert blocks, page
    namespace: dict = {"__name__": "__doc_example__"}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(blocks[0], page, "exec"), namespace)  # noqa: S102 - trusted repository docs
    printed = buffer.getvalue()
    if page.endswith("trade_accounting.md"):
        assert namespace["result"].converged
        assert "physical_equations" in printed
    else:
        assert "Hicksian consumption welfare [A]" in printed
        assert namespace["welfare"].decomposition_residual == pytest.approx(0.0, abs=1e-9)
