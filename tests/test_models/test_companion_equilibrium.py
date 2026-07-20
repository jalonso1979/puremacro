"""Steady-state equilibrium of the nested het-firm DMP (Phase 2a)."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.models.nested_dmp.params import NestedDMPParameters
from puremacro.models.nested_dmp import equilibrium as S
from puremacro.models.nested_dmp.kernels import expected_discount


def test_expected_discount_kernel_matches_closed_form():
    # phi here are DECIMAL rates (the kernel does no unit conversion).
    pi, sigma, r_star, phi_D, phi_H = 0.5, 1.0, 0.01, -0.032, 0.015
    got = expected_discount(pi, sigma, r_star=r_star, phi_D=phi_D, phi_H=phi_H)
    want = pi / (1 + r_star + phi_D * sigma) + (1 - pi) / (1 + r_star + phi_H * sigma)
    assert got == pytest.approx(want)



def test_effective_discount_sign_flip_in_belief():
    # At sigma>0, a dovish belief raises beta_eff (dove cuts -> high beta),
    # a hawkish belief lowers it (hawk hikes -> low beta).
    p = NestedDMPParameters()
    base = 1.0 / (1.0 + p.r_star)
    assert S.effective_discount(p, sigma=1.0, pi=0.95) > base
    assert S.effective_discount(p, sigma=1.0, pi=0.02) < base




def test_nash_wage_formula_and_monotonicity():
    p = NestedDMPParameters(alpha=0.5, b=0.4, c_convex=0.20)
    y = np.array([0.8, 1.0, 1.3])
    theta = 1.5
    w = S.nash_wage(p, y, theta)
    want = p.alpha * y + (1 - p.alpha) * p.b + p.alpha * p.c_convex * theta
    np.testing.assert_allclose(w, want)
    # Increasing in productivity and in tightness.
    assert np.all(np.diff(w) > 0)
    assert np.all(S.nash_wage(p, y, 3.0) > w)


def test_match_value_monotone_with_thresholds():
    p = NestedDMPParameters()
    J, grid, x_star, xe_star = S.match_value(p, sigma=0.0, pi=p.prior_pi0, theta=1.0)
    assert J.shape == grid.shape
    # Value rises with productivity; thresholds ordered (entry hurdle >= sep).
    assert np.all(np.diff(np.asarray(J)) >= -1e-9)
    assert np.isfinite(x_star) and np.isfinite(xe_star)
    assert xe_star >= x_star


def test_match_value_dovish_belief_lowers_separation_threshold():
    # At sigma>0, a dovish belief raises beta_eff -> matches are worth more ->
    # the firm keeps lower-productivity matches -> x_star falls.
    p = NestedDMPParameters()
    _, _, xs_dove, _ = S.match_value(p, sigma=1.0, pi=0.95, theta=1.0)
    _, _, xs_hawk, _ = S.match_value(p, sigma=1.0, pi=0.05, theta=1.0)
    assert xs_dove < xs_hawk


def test_match_value_guards_non_contraction():
    # Force beta_eff*(1-s_bar) >= 1 by a dovish belief at high sigma combined
    # with a near-zero separation rate: the VFI would not contract.
    # phi_D=-0.1 passes the pole guard (worst_r=-0.19>-1 at sigma=2) but at
    # sigma=3, pi=0.999 gives beta_eff≈1.4, so disc>=1 triggers the guard.
    p = NestedDMPParameters(s_bar=0.001, phi_D=-0.1, phi_H=0.5)
    with pytest.raises(ValueError, match="contract"):
        S.match_value(p, sigma=3.0, pi=0.999, theta=1.0)


def _pi_star(p: NestedDMPParameters) -> float:
    return p.phi_H / (p.phi_H - p.phi_D)


def test_free_entry_residual_is_zero_at_solution():
    p = NestedDMPParameters()
    theta, J, grid, x_star, xe_star = S.free_entry_theta(p, sigma=0.0, pi=p.prior_pi0)
    _, _, erg = S.sigma_scaled_process(p, sigma=0.0)
    beta_eff = S.effective_discount(p, sigma=0.0, pi=p.prior_pi0)
    ev = float(np.sum(erg * np.maximum(np.asarray(J) - p.f_fixed, 0.0)))
    q = p.mu * theta ** (-p.alpha)
    assert (p.c_convex - beta_eff * q * ev) == pytest.approx(0.0, abs=1e-6)


def test_free_entry_theta_sign_flip_in_belief():
    # THE headline structural result: at sigma>0, dovish belief -> high beta_eff
    # -> high J -> more vacancy posting -> higher theta; hawkish -> lower theta.
    # pi* is interior for the default phi's, so 0.95 and 0.05 straddle it.
    p = NestedDMPParameters()
    assert 0.0 < _pi_star(p) < 1.0
    theta_dove, *_ = S.free_entry_theta(p, sigma=1.0, pi=0.95)
    theta_hawk, *_ = S.free_entry_theta(p, sigma=1.0, pi=0.05)
    assert theta_dove > theta_hawk


def test_free_entry_theta_belief_irrelevant_at_zero_sigma():
    # At sigma=0 both Fed branches give the same rate, so theta is pi-invariant.
    p = NestedDMPParameters()
    theta_a, *_ = S.free_entry_theta(p, sigma=0.0, pi=0.9)
    theta_b, *_ = S.free_entry_theta(p, sigma=0.0, pi=0.1)
    assert theta_a == pytest.approx(theta_b, rel=1e-6)


def test_stationary_distribution_accounting_and_flow_balance():
    p = NestedDMPParameters()
    theta, J, grid, x_star, xe_star = S.free_entry_theta(p, sigma=0.0, pi=p.prior_pi0)
    _, P, _ = S.sigma_scaled_process(p, sigma=0.0)
    phi, u, n, s_eff = S.stationary_distribution(
        p, sigma=0.0, pi=p.prior_pi0, theta=theta, J=J, grid=grid, P=P,
        x_star=x_star, xe_star=xe_star,
    )
    # Labor-force accounting: employment + unemployment = 1.
    assert (n + u) == pytest.approx(1.0, abs=1e-8)
    assert 0.0 < u < 1.0
    # Effective separation exceeds the exogenous floor (endogenous adds to it).
    assert s_eff >= p.s_bar - 1e-12
    # Steady-state flow balance: hires f(theta)*u == separations s_eff*n.
    f_theta = p.mu * theta ** (1.0 - p.alpha)
    assert (f_theta * u) == pytest.approx(s_eff * n, rel=1e-6)


def test_stationary_unemployment_higher_under_hawkish_uncertainty():
    # At sigma>0, hawkish belief contracts theta -> fewer hires -> higher u.
    p = NestedDMPParameters()

    def solve_u(pi):
        theta, J, grid, x_star, xe_star = S.free_entry_theta(p, sigma=1.0, pi=pi)
        _, P, _ = S.sigma_scaled_process(p, sigma=1.0)
        _, u, _, _ = S.stationary_distribution(
            p, sigma=1.0, pi=pi, theta=theta, J=J, grid=grid, P=P,
            x_star=x_star, xe_star=xe_star,
        )
        return u

    assert solve_u(0.05) > solve_u(0.95)


def test_solve_steady_state_end_to_end_defaults():
    p = NestedDMPParameters()
    eq = S.solve_steady_state(p, sigma=0.0, pi=p.prior_pi0)
    assert isinstance(eq, S.SteadyState)
    assert 0.0 < eq.u < 1.0
    assert eq.theta > 0.0
    assert eq.v == pytest.approx(eq.theta * eq.u)
    assert (eq.n + eq.u) == pytest.approx(1.0, abs=1e-8)
    assert np.isfinite(np.asarray(eq.J)).all()
    assert eq.beta_bar == pytest.approx(1.0 / (1.0 + p.r_star))  # sigma=0


def test_solve_steady_state_aggregate_sign_flip():
    # Aggregate-level sign-flip: at sigma>0, dovish belief -> tighter market
    # (higher theta) and lower unemployment than hawkish.
    p = NestedDMPParameters()
    dove = S.solve_steady_state(p, sigma=1.0, pi=0.95)
    hawk = S.solve_steady_state(p, sigma=1.0, pi=0.05)
    assert dove.theta > hawk.theta
    assert dove.u < hawk.u


def test_solve_steady_state_is_frozen():
    import dataclasses
    eq = S.solve_steady_state(NestedDMPParameters(), sigma=0.0, pi=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        eq.theta = 2.0  # type: ignore[misc]


def test_delta_shortens_effective_horizon():
    # A firm-exit hazard delta>0 lowers the effective continuation discount
    # beta_bar*(1-delta)*(1-s_bar), so match values fall and the separation
    # threshold rises (fewer matches worth keeping). delta=0 is the baseline.
    p0 = NestedDMPParameters(delta=0.0)
    pd = NestedDMPParameters(delta=0.10)
    J0, _, xs0, _ = S.match_value(p0, sigma=0.0, pi=0.5, theta=1.0)
    Jd, _, xsd, _ = S.match_value(pd, sigma=0.0, pi=0.5, theta=1.0)
    assert np.all(np.asarray(Jd) <= np.asarray(J0) + 1e-12)
    assert xsd >= xs0


def test_steady_state_reports_aggregates():
    eq = S.solve_steady_state(NestedDMPParameters(), sigma=0.0)
    # Flow rates and output present and sensible.
    assert eq.urate == pytest.approx(eq.u)
    assert eq.jf_rate == pytest.approx(
        NestedDMPParameters().mu * eq.theta ** (1.0 - NestedDMPParameters().alpha)
    )
    assert eq.jd_rate == pytest.approx(eq.s_eff)
    assert eq.output > 0.0
    assert eq.converged is True
    assert abs(eq.free_entry_residual) < 1e-6


def test_comparative_statics_reproduces_sign_flip():
    p = NestedDMPParameters()
    pi_grid = np.array([0.05, 0.319, 0.95])
    sigma_grid = np.array([0.0, 1.0])
    cs = S.comparative_statics(p, pi_grid=pi_grid, sigma_grid=sigma_grid)
    assert cs.theta.shape == (3, 2)        # (pi, sigma)
    # At sigma=0 theta is pi-invariant (belief dormant).
    assert cs.theta[0, 0] == pytest.approx(cs.theta[2, 0], rel=1e-6)
    # At sigma=1 the sign-flip: dovish (pi=0.95) tighter than hawkish (pi=0.05).
    assert cs.theta[2, 1] > cs.theta[0, 1]
    # pi* recorded and interior.
    assert 0.0 < cs.pi_star < 1.0


def test_worker_values_solve_linear_system():
    p = NestedDMPParameters()
    theta, w_bar, s_eff = 1.5, 0.8, 0.05
    W_U, W_E = S.worker_values(p, theta=theta, w_bar=w_bar, s_eff=s_eff)
    f_theta = p.mu * theta ** (1.0 - p.alpha)
    b = p.beta
    # The two Bellman identities must hold at the returned (W_U, W_E).
    assert W_E == pytest.approx(w_bar + b * ((1 - s_eff) * W_E + s_eff * W_U), rel=1e-9)
    assert W_U == pytest.approx(p.b + b * (f_theta * W_E + (1 - f_theta) * W_U), rel=1e-9)
    # Employment is worth more than search; both positive for a viable market.
    assert W_E > W_U > 0.0


def test_participation_rate_threshold_and_toggle():
    p_off = NestedDMPParameters(h_max=0.0)
    p_on = NestedDMPParameters(h_max=0.5)
    W_U = 12.0
    # Off: everyone participates regardless of W_U.
    assert S.participation_rate(p_off, W_U) == 1.0
    # On: lfpr = min(h*/h_max, 1), h* = (1-beta)*W_U.
    h_star = (1.0 - p_on.beta) * W_U
    assert S.participation_rate(p_on, W_U) == pytest.approx(min(h_star / p_on.h_max, 1.0))
    # Higher search value -> (weakly) higher participation.
    assert S.participation_rate(p_on, 20.0) >= S.participation_rate(p_on, 5.0)
    # Clamped to [0,1].
    assert 0.0 <= S.participation_rate(p_on, 1e-6) <= 1.0


def test_solve_steady_state_participation_fields():
    # Off (default): lfpr=1, N=0, and U/E equal the urate/employment-rate levels.
    eq0 = S.solve_steady_state(NestedDMPParameters(), sigma=0.0)
    assert eq0.lfpr == 1.0 and eq0.N == pytest.approx(0.0)
    assert eq0.W_U > 0.0 and eq0.W_E > eq0.W_U
    # On: lfpr<1, N>0, and accounting holds E + U + N == 1.
    # h_max=2.0 exceeds h_star=(1-beta)*W_U~1.31, so G(h*)<1 and lfpr is interior.
    eq1 = S.solve_steady_state(NestedDMPParameters(h_max=2.0), sigma=0.0)
    assert 0.0 < eq1.lfpr < 1.0
    assert eq1.N == pytest.approx(1.0 - eq1.lfpr)
    assert (eq1.E + eq1.U + eq1.N) == pytest.approx(1.0, abs=1e-8)
    # The unemployment RATE is unchanged by participation (LF-independent).
    assert eq1.urate == pytest.approx(eq0.urate, rel=1e-9)


def test_participation_responds_to_uncertainty_via_tightness():
    # With the margin on, a dovish belief at sigma>0 raises theta -> higher
    # job-finding -> higher search value -> higher LFPR than a hawkish belief.
    # (Indirect channel: uncertainty moves participation through tightness.)
    # h_max=3.0 keeps LFPR interior: at sigma=1 W_U≈244 so h*≈2.44, requiring
    # h_max>2.44; h_max=2.0 clamps both to 1.0 at the actual (high-theta) W_U.
    p = NestedDMPParameters(h_max=3.0)
    dove = S.solve_steady_state(p, sigma=1.0, pi=0.95)
    hawk = S.solve_steady_state(p, sigma=1.0, pi=0.05)
    assert dove.theta > hawk.theta          # the firm-side sign-flip still holds
    assert 0.0 < dove.lfpr < 1.0            # interior participation
    assert dove.lfpr >= hawk.lfpr           # participation co-moves with tightness
    assert dove.W_U > hawk.W_U              # via the search value
