"""Steady-state equilibrium of the nested heterogeneous-firm DMP (free-entry closure).

Solves the model at a FIXED aggregate state (sigma, pi) -- the inner block the
Krusell-Smith aggregate solve calls repeatedly. Assembles the Phase-1 kernels
(rouwenhorst, bellman_iterate, expected_discount) into: match value J(x) by VFI
(separated firms exit at value 0; optional firm-exit hazard delta), endogenous
separation threshold x_sep, free-entry tightness theta (unbounded), the
stationary employment measure, and reporting aggregates. The expected-discount
belief channel beta_bar(pi, sigma) generates the regime sign-flip at the
threshold pi* = phi_H/(phi_H - phi_D).

Locked modeling decisions: sigma-scaled (mean-preserving-spread) productivity,
flexible Nash wage, non-convex hiring as an entry hurdle (f_fixed), 2-state
{E,U} core (participation N deferred).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from puremacro.models.nested_dmp.kernels import (
    bellman_iterate,
    expected_discount,
    rouwenhorst,
)
from puremacro.models.nested_dmp.params import NestedDMPParameters


@dataclass(frozen=True)
class ComparativeStatics:
    """Sign-flip surface: per-(pi,sigma) steady states + extracted arrays."""
    pi_grid: np.ndarray
    sigma_grid: np.ndarray
    pi_star: float
    theta: np.ndarray     # (n_pi, n_sigma)
    urate: np.ndarray     # (n_pi, n_sigma)
    x_sep: np.ndarray     # (n_pi, n_sigma)
    states: list          # list of lists of SteadyState


@dataclass(frozen=True)
class SteadyState:
    """Solved steady-state equilibrium at a fixed aggregate state (free-entry closure)."""
    sigma: float
    pi: float
    theta: float
    beta_bar: float
    x_sep: float          # endogenous separation threshold
    x_post: float         # posting/entry threshold (firm posts iff J>f_fixed)
    u: float              # unemployment LEVEL = lfpr * urate (= urate when h_max=0)
    n: float              # employment LEVEL = lfpr * (1-urate) (= 1-urate when h_max=0)
    v: float
    s_eff: float
    urate: float          # unemployment RATE in the labor force (LF-independent)
    jf_rate: float        # job-finding rate f(theta)=mu*theta^(1-alpha)
    jd_rate: float        # job-destruction rate (= s_eff)
    output: float         # sum_x phi(x)*exp(x)
    converged: bool
    free_entry_residual: float
    lfpr: float           # labor-force participation rate (1.0 when h_max=0)
    N: float              # non-participants = 1 - lfpr
    W_U: float            # worker search value
    W_E: float            # worker employment value
    grid: np.ndarray
    J: np.ndarray
    phi: np.ndarray

    @property
    def E(self) -> float:
        """Employment level (alias of n)."""
        return self.n

    @property
    def U(self) -> float:
        """Unemployment level (alias of u)."""
        return self.u


def effective_discount(params: NestedDMPParameters, sigma: float, pi: float) -> float:
    """E[beta | pi, sigma]; phi are already decimal on the parameter object."""
    return float(
        expected_discount(
            pi, sigma, r_star=params.r_star, phi_D=params.phi_D, phi_H=params.phi_H,
            phi_sigma=params.phi_sigma,
        )
    )


def sigma_scaled_process(params: NestedDMPParameters, sigma: float):
    """(grid, P, ergodic) for match productivity x under uncertainty sigma.

    The MPS selection channel is gated by sigma_x_loading (default 0 = OFF):
    sd_eff = sigma_x*(1 + sigma_x_loading*sigma). At loading=0 sigma does not
    widen the idiosyncratic process -- sigma operates only through beliefs
    (beta_bar), restoring the theory paper's belief-only sign-flip. Set
    sigma_x_loading=1 to recover the always-on Bloom (2018) MPS.
    Requires sigma > -1 so sd_eff > 0 (binding only when loading > 0).
    """
    if sigma <= -1.0:
        raise ValueError(f"sigma must be > -1 so sd_eff>0; got {sigma}")
    sd_eff = params.sigma_x * (1.0 + params.sigma_x_loading * sigma)
    grid, P = rouwenhorst(params.n_x, params.rho_x, sd_eff)
    erg = np.full(params.n_x, 1.0 / params.n_x)
    for _ in range(100_000):
        nxt = erg @ P
        if np.max(np.abs(nxt - erg)) < 1e-14:
            erg = nxt
            break
        erg = nxt
    erg = erg / erg.sum()
    return grid, P, erg


def nash_wage(params: NestedDMPParameters, y_grid, theta: float):
    """Flexible Nash wage under Hosios (worker bargaining share = alpha):

        w(x) = alpha*y(x) + (1-alpha)*b + alpha*c*theta,

    with b the flow value of unemployment and c the vacancy cost (c_convex).
    The asymmetric downward rigidity (psi_down) is dynamic and is deferred to
    Phase 2b; the stationary block is flexible-wage.
    """
    a = params.alpha
    return a * y_grid + (1.0 - a) * params.b + a * params.c_convex * theta


def worker_values(params: NestedDMPParameters, *, theta: float, w_bar: float, s_eff: float):
    """Scalar worker search/employment values (W_U, W_E) at the household discount.

    Workers discount at params.beta (< 1), NOT the firm belief factor beta_bar,
    so these are finite for all (pi, sigma); uncertainty reaches participation
    only through equilibrium theta. Solves the 2x2 linear Bellman system

        W_E = w_bar + beta*[(1-s_eff)*W_E + s_eff*W_U]
        W_U = b     + beta*[f(theta)*W_E + (1-f(theta))*W_U],   f(theta)=mu*theta^(1-alpha).
    """
    beta = params.beta
    f_theta = params.mu * theta ** (1.0 - params.alpha)
    A = np.array([
        [1.0 - beta * (1.0 - s_eff), -beta * s_eff],
        [-beta * f_theta,            1.0 - beta * (1.0 - f_theta)],
    ])
    rhs = np.array([w_bar, params.b])
    W_E, W_U = np.linalg.solve(A, rhs)
    return float(W_U), float(W_E)


def participation_rate(params: NestedDMPParameters, W_U: float) -> float:
    """Labor-force participation rate from the search value W_U.

    Worker participates iff home value h <= h* = (1-beta)*W_U, with h~U[0,h_max];
    so LFPR = G(h*) = min(h*/h_max, 1), clamped to [0,1]. h_max=0 turns the margin
    off (everyone participates -> the 2-state {E,U} core).
    """
    if params.h_max <= 0.0:
        return 1.0
    h_star = (1.0 - params.beta) * W_U
    return float(min(max(h_star / params.h_max, 0.0), 1.0))


def match_value(
    params: NestedDMPParameters,
    sigma: float,
    pi: float,
    theta: float,
    *,
    grid=None,
    P=None,
    xp=np,
    tol: float = 1e-10,
    max_iter: int = 10_000,
):
    """Firm match value J(x) at a candidate tightness theta.

        J(x) = max( y(x) - w(x;theta) + beta_bar*(1-delta)*(1-s_bar)*E[J(x')|x], 0 ),

    solved by VFI (kernels.bellman_iterate); the outer max is the endogenous
    separation option (outside value 0). y(x)=exp(x); delta is the firm-exit
    hazard. Returns (J, grid, x_star, xe_star): x_star = lowest x with J>0
    (separation threshold); xe_star = lowest x with J>f_fixed (entry threshold --
    firms pay the fixed start-up cost only above it). The heavy VFI runs on
    ``xp`` (numpy oracle or mlx); thresholds are read off the numpy view.

    Raises if beta_bar*(1-delta)*(1-s_bar) >= 1 (dovish belief x high sigma can
    push beta_bar above 1, and the VFI would not contract).
    """
    if grid is None or P is None:
        grid, P, _ = sigma_scaled_process(params, sigma)
    beta_bar = effective_discount(params, sigma, pi)
    disc = beta_bar * (1.0 - params.delta) * (1.0 - params.s_bar)
    # Guard: dovish belief x high sigma can push beta_bar above 1; the exit
    # hazard delta and separation s_bar must bring the effective discount below
    # 1 or the VFI is not a contraction.
    if disc >= 1.0:
        raise ValueError(
            f"beta_bar*(1-delta)*(1-s_bar)={disc:.4f} >= 1; VFI will not contract. "
            f"Raise delta or check (pi={pi}, sigma={sigma})."
        )
    y_grid = np.exp(grid)
    w_grid = nash_wage(params, y_grid, theta)
    flow_np = y_grid - w_grid
    if xp is np:
        J = bellman_iterate(flow_np, P, disc, tol=tol, max_iter=max_iter)
    else:
        J = bellman_iterate(
            xp.array(flow_np), xp.array(P), disc, tol=tol, max_iter=max_iter, xp=xp
        )
    Jnp = np.asarray(J)
    pos = np.nonzero(Jnp > 0.0)[0]
    x_star = float(grid[pos[0]]) if pos.size else float("inf")
    ent = np.nonzero(Jnp > params.f_fixed)[0]
    xe_star = float(grid[ent[0]]) if ent.size else float("inf")
    return J, grid, x_star, xe_star


def free_entry_theta(
    params: NestedDMPParameters,
    sigma: float,
    pi: float,
    *,
    theta_lo: float = 1e-4,
    theta_hi: float = 1e3,
    xtol: float = 1e-8,
):
    """Solve free entry for market tightness theta* (scipy brentq).

    Free entry with a non-convex start-up cost f_fixed:

        c_convex = beta_bar * q(theta) * E_{x_e ~ G_e}[ max(J(x_e;theta) - f_fixed, 0) ],

    q(theta)=mu*theta^{-alpha}, G_e the ergodic distribution of the x-process.
    The residual c - RHS is strictly increasing in theta (q falls, and the Nash
    wage rises so J falls), so the root is unique. Returns
    (theta_star, J, grid, x_star, xe_star) evaluated at the solution.
    """
    grid, P, erg = sigma_scaled_process(params, sigma)
    beta_bar = effective_discount(params, sigma, pi)

    def residual(theta: float) -> float:
        J, _, _, _ = match_value(params, sigma, pi, theta, grid=grid, P=P)
        ev = float(np.sum(erg * np.maximum(np.asarray(J) - params.f_fixed, 0.0)))
        q = params.mu * theta ** (-params.alpha)
        return params.c_convex - beta_bar * q * ev

    theta_star = float(brentq(residual, theta_lo, theta_hi, xtol=xtol))
    J, grid, x_star, xe_star = match_value(params, sigma, pi, theta_star, grid=grid, P=P)
    return theta_star, J, grid, x_star, xe_star


def stationary_distribution(
    params: NestedDMPParameters,
    sigma: float,
    pi: float,
    theta: float,
    J,
    grid,
    P,
    x_star: float,
    xe_star: float,
):
    """Stationary employment measure phi(x) + aggregates at a solved theta.

    Continuation operator T[x,x'] = (1-s_bar)*P[x,x']*1[x'>=x_star]; new hires
    enter at e (ergodic distribution restricted to startable states x>=xe_star,
    renormalized). Labor force = 1, so u = 1 - sum(phi); hires per period
    H = f(theta)*u with f(theta)=mu*theta^{1-alpha}.

    Fixed-point equation phi = phi@T + f_theta*(1-phi.sum())*e rearranges to
    the linear system phi*(I - T + f_theta*outer(1,e)) = f_theta*e, solved
    directly via numpy.linalg.solve (the naive iteration diverges when f_theta>1,
    which is common at empirically reasonable tightness values). Returns
    (phi, u, n, s_eff) with s_eff the period outflow divided by employment.
    """
    survive = (grid >= x_star).astype(float)          # 1[x' >= x_star]
    T = (1.0 - params.s_bar) * P * survive[None, :]    # T[x, x']
    _, _, erg = sigma_scaled_process(params, sigma)
    e = erg * (grid >= xe_star).astype(float)
    if e.sum() <= 0.0:
        raise ValueError("no startable entry states (J<=f_fixed everywhere)")
    e = e / e.sum()
    f_theta = params.mu * theta ** (1.0 - params.alpha)

    # Solve the row-vector system phi @ A = b (A = I - T + f_theta*outer(ones,e),
    # b = f_theta*e) as A.T @ phi.T = b.T, i.e. np.linalg.solve(A.T, b).
    n_x = len(grid)
    ones = np.ones(n_x)
    A = np.eye(n_x) - T + f_theta * np.outer(ones, e)
    b = f_theta * e
    phi = np.linalg.solve(A.T, b)
    u = float(1.0 - phi.sum())
    n = float(phi.sum())
    cont_prob = (1.0 - params.s_bar) * (P * survive[None, :]).sum(axis=1)
    outflow = float(np.sum(phi * (1.0 - cont_prob)))
    s_eff = outflow / n if n > 0 else float("nan")
    return phi, u, n, s_eff


def solve_steady_state(
    params: NestedDMPParameters, *, sigma: float = 0.0, pi: float | None = None,
) -> SteadyState:
    """Deterministic steady-state equilibrium at a fixed aggregate state (sigma, pi).

    pi=None uses params.prior_pi0 (the no-belief-info anchor). At sigma=0 the
    belief channel is dormant (beta_bar=1/(1+r*) for every pi), so pi only bites
    at sigma>0; the sign-flip is a sigma>0 comparative static (see comparative_statics).
    """
    if pi is None:
        pi = params.prior_pi0
    theta, J, grid, x_star, xe_star = free_entry_theta(params, sigma, pi)
    _, P, _ = sigma_scaled_process(params, sigma)
    phi, u, n, s_eff = stationary_distribution(
        params, sigma, pi, theta, J, grid, P, x_star, xe_star
    )
    grid_arr = np.asarray(grid)
    jf_rate = params.mu * theta ** (1.0 - params.alpha)
    output = float(np.sum(phi * np.exp(grid_arr)))
    beta_bar = effective_discount(params, sigma, pi)
    # Recompute the free-entry residual at the solved theta to confirm
    # it is ~0; reusing the ergodic draw avoids a redundant process call.
    erg = sigma_scaled_process(params, sigma)[2]
    ev = float(np.sum(erg * np.maximum(np.asarray(J) - params.f_fixed, 0.0)))
    q = params.mu * theta ** (-params.alpha)
    fe_resid = params.c_convex - beta_bar * q * ev
    # Worker side + participation (a separable layer; LF-independent under free
    # entry, so no extra fixed point -- compute W_U then scale levels by lfpr).
    w_grid = nash_wage(params, np.exp(grid_arr), theta)
    w_bar = float(np.sum(phi * w_grid) / n) if n > 0 else 0.0
    W_U, W_E = worker_values(params, theta=theta, w_bar=w_bar, s_eff=s_eff)
    lfpr = participation_rate(params, W_U)
    u_rate = u  # u from the firm solve is the unemployment RATE (labor force=1 there)
    U_lvl = lfpr * u_rate
    E_lvl = lfpr * n
    N_lvl = 1.0 - lfpr
    return SteadyState(
        sigma=sigma, pi=pi, theta=theta, beta_bar=beta_bar,
        x_sep=x_star, x_post=xe_star, u=U_lvl, n=E_lvl, v=theta * U_lvl, s_eff=s_eff,
        urate=u_rate, jf_rate=jf_rate, jd_rate=s_eff, output=output,
        converged=True, free_entry_residual=fe_resid,
        lfpr=lfpr, N=N_lvl, W_U=W_U, W_E=W_E,
        grid=grid_arr, J=np.asarray(J), phi=phi,
    )


def comparative_statics(
    params: NestedDMPParameters, *, pi_grid, sigma_grid
) -> ComparativeStatics:
    """Re-solve the steady state across a (pi, sigma) grid for the sign-flip plots.

    pi* = phi_H/(phi_H - phi_D) is the belief threshold where d beta_bar/d sigma
    changes sign; at sigma>0 the equilibrium theta rises with pi (dove -> expansion)
    above pi* and falls below it.
    """
    pi_grid = np.asarray(pi_grid, dtype=float)
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    n_pi, n_sig = pi_grid.size, sigma_grid.size
    theta = np.empty((n_pi, n_sig))
    urate = np.empty((n_pi, n_sig))
    x_sep = np.empty((n_pi, n_sig))
    states: list = []
    for i, pi in enumerate(pi_grid):
        row = []
        for j, sig in enumerate(sigma_grid):
            eq = solve_steady_state(params, sigma=float(sig), pi=float(pi))
            theta[i, j] = eq.theta
            urate[i, j] = eq.urate
            x_sep[i, j] = eq.x_sep
            row.append(eq)
        states.append(row)
    # pi* is the threshold where d beta_bar/d pi changes sign under sigma>0.
    pi_star = params.phi_H / (params.phi_H - params.phi_D)
    return ComparativeStatics(
        pi_grid=pi_grid, sigma_grid=sigma_grid, pi_star=pi_star,
        theta=theta, urate=urate, x_sep=x_sep, states=states,
    )


__all__ = [
    "effective_discount",
    "sigma_scaled_process",
    "nash_wage",
    "worker_values",
    "participation_rate",
    "match_value",
    "free_entry_theta",
    "stationary_distribution",
    "SteadyState",
    "solve_steady_state",
    "ComparativeStatics",
    "comparative_statics",
]
