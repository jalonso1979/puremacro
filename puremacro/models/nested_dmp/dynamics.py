"""Perfect-foresight aggregate dynamics + state-dependent IRFs of the nested DMP.

The free-entry equilibrium has no cross-sectional-distribution feedback into
prices, so the aggregate state is the 2-D (sigma, pi) and the response to an
LTUI shock is an exact perfect-foresight transition (NOT a Krusell-Smith global
solve). The het-firm analogue of dmp_regime_dependent.dmp_irf: endogenous
Bayesian belief dynamics + a backward (theta, J) sweep + a forward (u, phi)
simulation on a FIXED grid, returning log-deviation IRFs under the dove/hawk
(loose/tight) scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from puremacro.models.nested_dmp.kernels import (
    belief_update,
    rouwenhorst,
    tauchen_transition,
)
from puremacro.models.nested_dmp.params import NestedDMPParameters
from puremacro.models.nested_dmp.equilibrium import (
    effective_discount,
    match_value,
    nash_wage,
    participation_rate,
    worker_values,
)


@dataclass(frozen=True)
class IRFResult:
    """State-dependent model IRFs (log-deviations from the sigma=0 steady state)."""
    fed_type: str
    horizons: np.ndarray
    log_urate: np.ndarray
    log_v: np.ndarray
    log_theta: np.ndarray
    log_lfpr: np.ndarray


def belief_path(params: NestedDMPParameters, *, sigma_path, fed_type: str,
                prior: float | None = None):
    """Recursive Bayesian belief pi_t along a shock path, given the Fed type.

    Realised rate r_t = r* + (phi_tau - phi_sigma)*sigma_t for the true type tau
    (dove: phi_D<0, hawk: phi_H>0); agents update pi_t = belief_update(pi_{t-1},
    r_t, sigma_t, ...) from the starting belief ``prior`` (defaults to
    params.prior_pi0). At sigma_t=0 both type-rates equal r* so the update is
    uninformative and pi stays put.

    ``prior`` is the persistent-regime starting belief: the state-dependent IRF
    conditions on a loose regime (high prior, e.g. 0.9) or a tight regime (low
    prior, e.g. 0.1), so the dove and hawk paths straddle the sign-flip threshold
    pi* = phi_H/(phi_H - phi_D). With a common 0.5 prior the weak rate signals
    barely move pi (both stay above pi*), which would suppress the flip.
    """
    if fed_type not in ("dove", "hawk"):
        raise ValueError(f"fed_type must be 'dove' or 'hawk'; got {fed_type!r}")
    if prior is None:
        prior = params.prior_pi0
    sigma_path = np.asarray(sigma_path, dtype=float)
    phi_tau = params.phi_D if fed_type == "dove" else params.phi_H
    pi = np.empty_like(sigma_path)
    # pi[0] records the prior before the first signal; pi[t] (t>=1) is the
    # posterior after observing sigma_path[t-1].  This preserves the initial
    # condition pi_0 = prior as the t=0 entry of the returned path.
    pi[0] = prior
    for t in range(len(sigma_path)):
        sig = sigma_path[t]
        r_obs = params.r_star + (phi_tau - params.phi_sigma) * sig
        prior = float(belief_update(
            prior, r_obs=r_obs, sigma=sig, r_star=params.r_star,
            phi_D=params.phi_D, phi_H=params.phi_H, signal_sd=params.signal_sd,
        ))
        # Store the post-signal belief one step ahead; the last period wraps
        # around but we never read pi[T] beyond the path length, so write into
        # a temporary slot and overwrite only if within bounds.
        if t + 1 < len(sigma_path):
            pi[t + 1] = prior
    return pi


def _fixed_grid(params: NestedDMPParameters, *, sigma_ref: float):
    """Fixed productivity grid (+ ergodic) wide enough for the peak shock.

    Built once by Rouwenhorst at the reference innovation sd
    sigma_x*(1 + sigma_x_loading*sigma_ref); loading=0 (default) leaves the
    grid at the baseline sigma_x width regardless of sigma_ref, so dynamics run
    on a fixed grid and sigma reaches the model only through beliefs.
    Transitions for each period's sigma_t are formed on THIS grid by Tauchen.
    """
    grid, P_ref = rouwenhorst(params.n_x, params.rho_x, params.sigma_x * (1.0 + params.sigma_x_loading * sigma_ref))
    erg = np.full(params.n_x, 1.0 / params.n_x)
    for _ in range(100_000):
        nxt = erg @ P_ref
        if np.max(np.abs(nxt - erg)) < 1e-14:
            erg = nxt
            break
        erg = nxt
    return grid, erg / erg.sum()


def calibrate_mu(params: NestedDMPParameters, *, sigma_ref: float = 1.0) -> float:
    """Matching efficiency mu such that theta_ss = 1 on the fixed grid.

    With theta_ss=1 the job-finding rate f(theta_ss)=mu*1=mu is < 1 at standard
    costs, so the forward unemployment law of motion is a contraction (stable
    transition dynamics + a valid matching probability). The default mu leaves
    theta_ss huge (f>1: invalid and the forward LoM diverges), so normalize
    before simulating IRFs. The realistic point calibration is a Phase-3 concern;
    this just gives the stable anchor the dynamics need (mirrors how
    calibrate_dmp_signal_extraction normalizes mu to theta_ss=1).
    """
    grid, erg = _fixed_grid(params, sigma_ref=sigma_ref)
    P0 = _P_sigma(params, grid, 0.0)
    J, _, _, _ = match_value(params, 0.0, params.prior_pi0, 1.0, grid=grid, P=P0)
    beta_bar = effective_discount(params, 0.0, params.prior_pi0)
    ev = float(np.sum(erg * np.maximum(np.asarray(J) - params.f_fixed, 0.0)))
    # Free entry at theta=1: c = beta_bar*q(1)*ev = beta_bar*mu*ev  =>  mu = c/(beta_bar*ev).
    return params.c_convex / (beta_bar * ev)


def _P_sigma(params: NestedDMPParameters, grid, sigma: float):
    """Tauchen transition on the fixed grid for uncertainty sigma.

    Loading=0 (default) passes sigma_x unchanged -- the grid-to-grid transition
    reflects only the baseline volatility, and sigma reaches the model through
    the belief channel beta_bar alone.
    """
    return tauchen_transition(grid, params.rho_x, params.sigma_x * (1.0 + params.sigma_x_loading * sigma))


def _bellman_step(params, *, flow, P, beta_bar, J_next):
    """One backward Bellman step: J = max(flow + beta_bar*(1-delta)*(1-s)*P@J_next, 0)."""
    disc = beta_bar * (1.0 - params.delta) * (1.0 - params.s_bar)
    return np.maximum(flow + disc * (P @ J_next), 0.0)


def _theta_given(params, *, sigma, pi, grid, erg, J_next, theta_lo=1e-4, theta_hi=1e3,
                 wage=None):
    """Free-entry theta for one period given next-period value J_next.

    Root-finds theta s.t. c = beta_bar*q(theta)*E_erg[max(J(theta)-f_fixed,0)],
    where J(theta) is one Bellman step from J_next under (sigma, pi).
    When ``wage`` is provided (a length-n vector), it is used as the period wage
    directly (the rigid-wage path from the outer fixed point); otherwise the
    flexible Nash wage is computed internally. Returns (theta, J).
    """
    beta_bar = effective_discount(params, sigma, pi)
    P = _P_sigma(params, grid, sigma)
    y = np.exp(grid)

    def J_of(theta):
        # Use the predetermined wage when given; fall back to Nash (flexible).
        w = wage if wage is not None else nash_wage(params, y, theta)
        flow = y - w
        return _bellman_step(params, flow=flow, P=P, beta_bar=beta_bar, J_next=J_next)

    def resid(theta):
        J = J_of(theta)
        ev = float(np.sum(erg * np.maximum(J - params.f_fixed, 0.0)))
        q = params.mu * theta ** (-params.alpha)
        return params.c_convex - beta_bar * q * ev

    try:
        theta = float(brentq(resid, theta_lo, theta_hi, xtol=1e-8))
    except ValueError:
        # No interior root: the (rigid) wage compressed the surplus so the entry
        # option value can't equal the vacancy cost at any theta. Degrade to the
        # boundary -- theta_lo if entry is never profitable (residual>0), else theta_hi.
        theta = theta_lo if resid(theta_lo) > 0.0 else theta_hi
    return theta, J_of(theta)


def _grid_steady_state(params, *, grid, erg, sigma, pi):
    """Steady state ON the fixed grid: the J/theta fixed point (J_next == J)."""
    J = np.zeros(params.n_x)
    theta = 1.0
    for _ in range(10_000):
        theta_new, J_new = _theta_given(params, sigma=sigma, pi=pi, grid=grid, erg=erg, J_next=J)
        if abs(theta_new - theta) < 1e-10 and np.max(np.abs(J_new - J)) < 1e-10:
            theta, J = theta_new, J_new
            break
        theta, J = theta_new, J_new
    return {"theta": theta, "J": J}


def _theta_sweep(params, *, sigma_path, pi_path, grid, erg, ss):
    """Backward sweep: theta_t, J_t for t=0..T given the (sigma,pi) paths.

    Boundary J_T = ss['J'] (fixed-grid steady state). For t=T-1..0, theta_t/J_t
    solve free entry with J_{t+1} given. theta/J are distribution-free, so this
    sweep needs only (sigma_t, pi_t).

    When psi_down > 0 the flexible sweep is used as an initialiser and then a
    damped {theta-path, w-path} fixed point is iterated: wages are predetermined
    (the het-firm analogue of the Hall damped-path iteration), rising freely to
    the Nash wage but falling sluggishly toward it at rate (1-psi_down).
    """
    T = len(sigma_path) - 1
    n = params.n_x

    # ------------------------------------------------------------------
    # Flexible backward sweep (always computed; serves as initial guess
    # for the rigid iteration and as the exact answer when psi_down == 0).
    # ------------------------------------------------------------------
    J_path = np.zeros((T + 1, n))
    theta_path = np.zeros(T + 1)
    J_path[T] = ss["J"]
    theta_path[T] = ss["theta"]
    for t in range(T - 1, -1, -1):
        theta_path[t], J_path[t] = _theta_given(
            params, sigma=sigma_path[t], pi=pi_path[t], grid=grid, erg=erg, J_next=J_path[t + 1]
        )

    if params.psi_down <= 0.0:
        # psi_down=0 is the exact flexible path — return directly, no iteration.
        return theta_path, J_path

    # ------------------------------------------------------------------
    # Damped {theta-path, w-path} fixed point for psi_down > 0.
    # w_ss is the pre-shock steady-state wage — the floor the lagged wage
    # anchors to at t=0 (wages cannot open the contract below this level).
    # NOTE: psi_down>0 is an appendix/robustness lever, NOT the baseline. As the
    # dove expansion fades the sticky wage stays high, transiently depressing J;
    # because separation is a HARD x-threshold this can dump a dense mass of
    # firms across it in one step (a spurious one-period unemployment spike).
    # The headline calibration (default_baseline) keeps psi_down=0.
    # ------------------------------------------------------------------
    y = np.exp(grid)
    w_ss = nash_wage(params, y, ss["theta"])      # scalar per grid point
    psi = params.psi_down

    for _iter in range(50):
        # (a) Forward wage path from the current theta-path (asymmetric Hall rule).
        # Wages rise freely to the Nash wage but fall sluggishly.
        w_path = np.empty((T + 1, n))
        w_lag = w_ss.copy()
        for t in range(T + 1):
            w_nash = nash_wage(params, y, theta_path[t])
            # Wages rise to Nash freely; fall only partially toward Nash when below.
            w_path[t] = np.maximum(w_nash, psi * w_lag + (1.0 - psi) * w_nash)
            w_lag = w_path[t]

        # (b) Backward sweep with the FIXED wage path from (a).
        J_new = np.zeros((T + 1, n))
        theta_new = np.zeros(T + 1)
        J_new[T] = ss["J"]
        theta_new[T] = ss["theta"]
        for t in range(T - 1, -1, -1):
            theta_new[t], J_new[t] = _theta_given(
                params, sigma=sigma_path[t], pi=pi_path[t], grid=grid, erg=erg,
                J_next=J_new[t + 1], wage=w_path[t],
            )

        # (c) Damped update; stop when theta-path has converged.
        delta_max = float(np.max(np.abs(theta_new - theta_path)))
        theta_path = 0.5 * theta_path + 0.5 * theta_new
        J_path = J_new  # use the latest J corresponding to the accepted wage path
        if delta_max < 1e-8:
            break

    return theta_path, J_path


def _x_threshold(J, grid, level: float) -> float:
    pos = np.nonzero(np.asarray(J) > level)[0]
    return float(grid[pos[0]]) if pos.size else float("inf")


def _phi_step(params, *, phi, theta, J_next, sigma, grid, erg):
    """One forward step of the employment measure (intensive, LF=1)."""
    P = _P_sigma(params, grid, sigma)
    x_sep = _x_threshold(J_next, grid, 0.0)
    x_post = _x_threshold(J_next, grid, params.f_fixed)
    survive = (grid >= x_sep).astype(float)
    T = (1.0 - params.s_bar) * P * survive[None, :]
    e = erg * (grid >= x_post).astype(float)
    e = e / e.sum() if e.sum() > 0 else e
    u = float(1.0 - phi.sum())
    f_theta = params.mu * theta ** (1.0 - params.alpha)
    return phi @ T + (f_theta * u) * e


def _stationary_phi(params, *, theta, J, sigma, grid, erg):
    """Fixed-grid stationary employment measure (the linear-solve, LF=1)."""
    P = _P_sigma(params, grid, sigma)
    x_sep = _x_threshold(J, grid, 0.0)
    x_post = _x_threshold(J, grid, params.f_fixed)
    survive = (grid >= x_sep).astype(float)
    Tm = (1.0 - params.s_bar) * P * survive[None, :]
    e = erg * (grid >= x_post).astype(float); e = e / e.sum()
    f_theta = params.mu * theta ** (1.0 - params.alpha)
    n = params.n_x
    A = np.eye(n) - Tm + f_theta * np.outer(np.ones(n), e)
    return np.linalg.solve(A.T, f_theta * e)


def _forward_sim(params, *, theta_path, J_path, sigma_path, pi_path, grid, erg, ss):
    """Forward intensive sim (LF=1) of (urate, phi, lfpr): PREDETERMINED u.

    Unemployment is a predetermined state. phi_0 is the pre-shock (sigma=0)
    steady-state measure (``ss``); phi_{t+1} = _phi_step(phi_t, theta_t,
    J_{t+1}, sigma_t), so u_t = 1 - sum(phi_t) starts at u_ss and responds
    gradually (the hump). Requires f(theta)<1 (use calibrate_mu) or the
    employment LoM is not a contraction and the forward sim diverges. lfpr_t
    from the worker search value (a labor-force-independent byproduct).
    Returns dict of length-(T+1) paths: urate, theta(copy), v, lfpr.
    """
    T = len(sigma_path) - 1
    y = np.exp(grid)
    # Predetermined initial distribution = the pre-shock sigma=0 steady state.
    phi = _stationary_phi(params, theta=ss["theta"], J=ss["J"], sigma=0.0, grid=grid, erg=erg)
    urate = np.empty(T + 1); lfpr = np.empty(T + 1); v = np.empty(T + 1)
    for t in range(T + 1):
        u_t = float(1.0 - phi.sum())
        urate[t] = u_t
        v[t] = theta_path[t] * u_t
        x_sep_t = _x_threshold(J_path[t], grid, 0.0)
        cont = (1.0 - params.s_bar) * (_P_sigma(params, grid, sigma_path[t]) *
                                       (grid >= x_sep_t)[None, :]).sum(axis=1)
        n_t = float(phi.sum())
        s_eff_t = float(np.sum(phi * (1.0 - cont)) / n_t) if n_t > 0 else float("nan")
        w_bar_t = float(np.sum(phi * nash_wage(params, y, theta_path[t])) / n_t) if n_t > 0 else 0.0
        W_U, _ = worker_values(params, theta=theta_path[t], w_bar=w_bar_t, s_eff=s_eff_t)
        lfpr[t] = participation_rate(params, W_U)
        if t < T:
            phi = _phi_step(params, phi=phi, theta=theta_path[t], J_next=J_path[t + 1],
                            sigma=sigma_path[t], grid=grid, erg=erg)
    return {"urate": urate, "theta": np.asarray(theta_path), "v": v, "lfpr": lfpr}


def simulate_irf(params, *, sigma0: float, fed_type: str, horizon: int = 8,
                 rho_sigma: float | None = None, T_simul: int = 60,
                 pi_loose: float = 0.9, pi_hawk: float = 0.1) -> IRFResult:
    """Perfect-foresight IRF to a unit-sigma0 LTUI shock under the Fed regime.

    sigma_t = sigma0*rho_sigma^t (0 at T_simul). The IRF conditions on a
    PERSISTENT regime: dove/loose (true type dove, starting belief pi_loose) or
    hawk/tight (true type hawk, starting belief pi_hawk); these straddle
    pi*=phi_H/(phi_H-phi_D) so the dove path expands and the hawk path contracts.
    theta/J by backward sweep; u/phi/lfpr by the predetermined-u forward sim.
    IRFs are log-deviations from the sigma=0 steady state (a flat base sim, since
    with slow job-finding u may not fully return by T_simul); the horizon-h value
    is model period t=h+1 (the +1 alignment -- urate is predetermined so its t=0
    response is mechanically 0). Requires f(theta_ss)<1 (use calibrate_mu); raises
    otherwise (the forward LoM is unstable when f>=1).
    """
    if rho_sigma is None:
        rho_sigma = params.rho_sigma
    if T_simul <= horizon + 1:
        raise ValueError(f"T_simul (={T_simul}) must exceed horizon+1 (={horizon+1}).")
    sigma_path = np.zeros(T_simul + 1)
    sigma_path[:T_simul] = sigma0 * (rho_sigma ** np.arange(T_simul))
    regime_prior = pi_loose if fed_type == "dove" else pi_hawk
    pi_path = belief_path(params, sigma_path=sigma_path, fed_type=fed_type, prior=regime_prior)

    grid, erg = _fixed_grid(params, sigma_ref=max(sigma0, 1e-6))
    ss = _grid_steady_state(params, grid=grid, erg=erg, sigma=0.0, pi=params.prior_pi0)

    # Guard: f(theta_ss)>=1 means the forward unemployment LoM is not a contraction.
    f_ss = params.mu * ss["theta"] ** (1.0 - params.alpha)
    if f_ss >= 1.0:
        raise ValueError(
            f"f(theta_ss)={f_ss:.3f} >= 1: the forward unemployment law of motion "
            f"is unstable. Normalize mu via calibrate_mu for stable IRF dynamics."
        )
    theta_path, J_path = _theta_sweep(params, sigma_path=sigma_path, pi_path=pi_path,
                                      grid=grid, erg=erg, ss=ss)
    paths = _forward_sim(params, theta_path=theta_path, J_path=J_path,
                         sigma_path=sigma_path, pi_path=pi_path, grid=grid, erg=erg, ss=ss)

    # Base = sigma=0 steady state as a FLAT forward sim (robust to slow u-return).
    n_t1 = T_simul + 1
    base = _forward_sim(
        params, theta_path=np.full(n_t1, ss["theta"]),
        J_path=np.tile(np.asarray(ss["J"]), (n_t1, 1)),
        sigma_path=np.zeros(n_t1), pi_path=np.full(n_t1, params.prior_pi0),
        grid=grid, erg=erg, ss=ss,
    )
    base_u, base_lf = float(base["urate"][0]), float(base["lfpr"][0])
    base_th = float(ss["theta"])
    base_v = base_th * base_u

    sl = slice(1, horizon + 2)  # +1 alignment: model t=h+1 <-> empirical horizon h

    def logdev(x, base_val):
        return np.log(np.maximum(np.asarray(x)[sl], 1e-12)) - np.log(max(base_val, 1e-12))

    return IRFResult(
        fed_type=fed_type, horizons=np.arange(horizon + 1),
        log_urate=logdev(paths["urate"], base_u),
        log_v=logdev(paths["v"], base_v),
        log_theta=logdev(paths["theta"], base_th),
        log_lfpr=logdev(paths["lfpr"], base_lf),
    )


__all__ = [
    "IRFResult",
    "simulate_irf",
    "belief_path",
    "calibrate_mu",
    "_fixed_grid",
    "_P_sigma",
    "_bellman_step",
    "_theta_given",
    "_grid_steady_state",
    "_theta_sweep",
    "_x_threshold",
    "_phi_step",
    "_stationary_phi",
    "_forward_sim",
]
