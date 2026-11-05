"""Sovereign default with fixed maturity parameters and default taste shocks.

Implements:
1. Arellano (2008) short-term debt sovereign default model (delta = 1.0, kappa = 0.0).
2. A long-term maturing coupon-debt extension inspired by Chatterjee & Eyigungor
   (2012) / Hatchondo & Martinez (2009), with smoothed quantal debt choices.
3. Extreme-value (Gumbel) taste shocks sigma_eps > 0 for smooth logistic default
   probabilities and differentiable log-sum-exp value functions.
4. Bond-price schedule derivatives dq(b', y)/db' from a monotone PCHIP
   interpolant (exact node tangents, vectorized).
5. Optional monotonicity enforcement and unadjusted simulated spread diagnostics.

The long-term repayment value averages objectives under smoothed choice
weights without an entropy bonus. It is not a deterministic optimal policy or
the inclusive value of a random-utility debt-choice model. See result metadata
for its choice probabilities and deterministic-optimality gap.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

import numpy as np
from scipy.stats import norm

from puremacro.sovereign._monotone import exhaustive_maximize, maximize, min_candidate_consumption
from puremacro.sovereign._results import SovereignDefaultResult

__all__ = [
    "solve_sovereign_default",
]


def _discretize_endowment(
    n_y: int,
    rho_y: float,
    sigma_y: float,
    m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize log AR(1) endowment process via Tauchen (1986).

    Normalizes endowment so that unconditional mean E[y] = 1.0.
    """
    if n_y < 2:
        raise ValueError(f"n_y must be >= 2; got {n_y}")
    if not (0.0 <= abs(rho_y) < 1.0):
        raise ValueError(f"|rho_y| must be < 1.0; got {rho_y}")
    if sigma_y <= 0.0:
        raise ValueError(f"sigma_y must be > 0; got {sigma_y}")

    sigma_z = sigma_y / np.sqrt(1.0 - rho_y ** 2)
    z_max = m * sigma_z
    log_y_grid = np.linspace(-z_max, z_max, n_y)
    step = log_y_grid[1] - log_y_grid[0]
    p_trans = np.zeros((n_y, n_y), dtype=np.float64)

    for i in range(n_y):
        mu_i = rho_y * log_y_grid[i]
        p_trans[i, 0] = norm.cdf((log_y_grid[0] - mu_i + step / 2.0) / sigma_y)
        p_trans[i, n_y - 1] = norm.sf((log_y_grid[n_y - 1] - mu_i - step / 2.0) / sigma_y)
        for j in range(1, n_y - 1):
            hi = norm.cdf((log_y_grid[j] - mu_i + step / 2.0) / sigma_y)
            lo = norm.cdf((log_y_grid[j] - mu_i - step / 2.0) / sigma_y)
            p_trans[i, j] = hi - lo

    # Stationary distribution
    eigvals, eigvecs = np.linalg.eig(p_trans.T)
    stat_idx = np.argmin(np.abs(eigvals - 1.0))
    stat = np.real(eigvecs[:, stat_idx])
    stat = stat / np.sum(stat)
    stat = np.maximum(0.0, stat)
    stat /= np.sum(stat)

    y_raw = np.exp(log_y_grid)
    mean_y = float(np.dot(stat, y_raw))
    y_grid = y_raw / mean_y

    return y_grid, p_trans, stat


def _compute_default_penalty(
    y_grid: np.ndarray,
    penalty_type: str = "chatterjee_eyigungor",
    d0: float = -0.188,
    d1: float = 0.245,
    mean_y: float | None = None,
) -> np.ndarray:
    """Compute endowment under default h(y) <= y."""
    if penalty_type == "chatterjee_eyigungor":
        loss = np.maximum(0.0, d0 * y_grid + d1 * (y_grid ** 2))
        y_def = y_grid - loss
    elif penalty_type == "arellano":
        # Arellano (2008) asymmetric default penalty: min(y, gamma_y * mean(y))
        y_def = np.minimum(y_grid, 0.969 * (float(np.mean(y_grid)) if mean_y is None else mean_y))
    else:
        raise ValueError(f"Unknown penalty_type {penalty_type!r}")

    # Ensure strictly positive default consumption
    return np.maximum(1e-4, y_def)


def _compute_dq_db_dualarray(q: np.ndarray, b_grid: np.ndarray) -> np.ndarray:
    """Evaluate the derivative dq(b', y)/db' of a monotone PCHIP interpolant.

    Uses Fritsch-Carlson harmonic-mean tangents, so derivatives satisfy
    dq/db <= 0. At each grid node the cubic Hermite derivative equals the
    node's tangent, which is computed for all nodes at once.
    """
    if q.shape[0] < 2:
        return np.zeros_like(q)
    delta = np.diff(q, axis=0) / np.diff(b_grid)[:, None]
    d = np.zeros_like(q)
    left, right = delta[:-1], delta[1:]
    same_sign = left * right > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = 2.0 / (1.0 / left + 1.0 / right)
    d[1:-1] = np.where(same_sign, harmonic, 0.0)
    d[0] = np.where(delta[0] <= 0.0, delta[0], 0.0)
    d[-1] = np.where(delta[-1] <= 0.0, delta[-1], 0.0)
    return np.minimum(d, 0.0)


def _crra_utility(gamma: float) -> Callable[[np.ndarray], np.ndarray]:
    """CRRA utility with a linear barrier for consumption at or below 1e-6."""
    def u_func(c_arr: np.ndarray) -> np.ndarray:
        c_safe = np.maximum(c_arr, 1e-6)
        if abs(gamma - 1.0) < 1e-4:
            val = np.log(c_safe)
        else:
            val = (c_safe ** (1.0 - gamma) - 1.0) / (1.0 - gamma)
        return np.where(c_arr <= 1e-6, -1e12 + 1e6 * c_arr, val)

    return u_func


def _make_debt_grid(b_min: float, b_max: float, n_b: int) -> tuple[np.ndarray, int]:
    """Validated debt grid with an exact zero node; returns ``(b_grid, zero_idx)``."""
    if n_b < 2 or b_min >= b_max or not b_min <= 0.0 <= b_max:
        raise ValueError("The debt grid must be increasing, have at least two points and include zero")
    if b_min < 0.0 < b_max and n_b < 3:
        raise ValueError("n_b must be >= 3 to represent zero debt and both saving/borrowing bounds")
    b_grid = np.linspace(b_min, b_max, n_b)
    if b_min < 0.0 < b_max:
        # Reentry must be the same state in Bellman iteration and simulation.
        # Preserve endpoints and shape instead of snapping to a nonzero debt.
        b_grid[1 + int(np.argmin(np.abs(b_grid[1:-1])))] = 0.0
    zero_idx = int(np.argmin(np.abs(b_grid)))
    return b_grid, zero_idx


def _validate_debt_grid(debt_grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Copy a supplied grid without moving its nodes or rounding its zero."""
    if np.iscomplexobj(debt_grid):
        raise ValueError("debt_grid must contain real coordinates")
    grid = np.array(debt_grid, dtype=float, copy=True)
    if (grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid))
            or not np.all(np.diff(grid) > 0.0)):
        raise ValueError("debt_grid must be a finite, strictly increasing vector with at least two nodes")
    zero = np.flatnonzero(grid == 0.0)
    if zero.size != 1:
        raise ValueError("debt_grid must include an exact zero-debt node")
    return grid, int(zero[0])


class _CoreSolution(NamedTuple):
    """Converged objects of the Bellman and bond-price fixed point."""

    v_rep: np.ndarray
    v_def: np.ndarray
    v_exp: np.ndarray
    q: np.ndarray
    p_def: np.ndarray
    m_opt: np.ndarray
    residual: float
    converged: bool
    iterations: int
    certification_gap: float | None
    exhaustive_residual: float | None
    q_rf: float


def _solve_bellman(
    *,
    b_grid: np.ndarray,
    p_trans: np.ndarray,
    resources: np.ndarray,
    resources_default: np.ndarray,
    choice_utility: Callable[[np.ndarray], np.ndarray],
    flow_repay: np.ndarray | None,
    flow_default: np.ndarray | None,
    beta: float,
    r_star: float,
    delta: float,
    kappa: float,
    sigma_eps: float,
    theta: float,
    tol: float,
    max_iter: int,
    damping: float,
    recovery_rate: float,
    howard_steps: int,
    enforce_monotonicity: bool,
    policy_search: str,
) -> _CoreSolution:
    """Solve the sovereign Bellman equations and the bond-price schedule.

    The government chooses debt to maximise ``choice_utility(c) + beta * EV``
    with ``c = resources - service * b + q * (b' - retention * b)``; the
    income-only flows ``flow_repay`` / ``flow_default`` (``(n_y,)`` or
    ``None``) are added to the repayment and default values but do not depend
    on the choice. ``None`` flows leave every floating-point operation as in
    the original short-term/long-term solver.
    """
    n_b = b_grid.shape[0]
    n_y = resources.shape[0]
    monotone = policy_search == "monotone"
    u_func = choice_utility

    def with_flow(x: np.ndarray) -> np.ndarray:
        # Exact no-op when there is no flow term.
        return x if flow_repay is None else x + flow_repay[None, :]

    u_def = u_func(resources_default)
    if flow_default is not None:
        u_def = u_def + flow_default
    v_def = u_def / (1.0 - beta)

    # Risk-free bond price bound: q_rf = (delta + (1-delta)*kappa) / (r_star + delta)
    q_rf = (delta + (1.0 - delta) * kappa) / (r_star + delta)
    q = np.full((n_b, n_y), q_rf, dtype=np.float64)

    # Initial repayment value
    c_init = np.maximum(1e-4, resources[None, :] - (delta + (1.0 - delta) * kappa) * b_grid[:, None])
    v_rep = with_flow(u_func(c_init)) / (1.0 - beta)

    # Grid broadcast tensors for vectorized policy search
    # b_cur: (n_b, 1, 1), b_next: (1, n_b, 1), y_cur: (1, 1, n_y)
    b_cur = b_grid[:, None, None]
    b_next = b_grid[None, :, None]
    y_cur = resources[None, None, :]

    converged = False
    residual = 1.0
    iteration = 0

    m_opt = np.zeros((n_b, n_y), dtype=int)
    v_rep_new = np.zeros_like(v_rep)
    v_def_new = np.zeros_like(v_def)
    p_def = np.zeros((n_b, n_y), dtype=np.float64)
    v_exp = np.zeros((n_b, n_y), dtype=np.float64)

    service_coeff = delta + (1.0 - delta) * kappa
    retention_coeff = 1.0 - delta

    # Vectorized policy search with smooth choice temperature for long-term debt
    tau_b = min(0.005, max(1e-4, sigma_eps / 2.0))

    for it in range(max_iter):
        iteration = it + 1

        # sigma_eps=0 recovers the original deterministic default decision.
        if sigma_eps == 0.0:
            p_def = (v_def[None, :] > v_rep).astype(float)
            v_exp = np.maximum(v_rep, v_def[None, :])
        else:
            diff = np.clip((v_rep - v_def[None, :]) / sigma_eps, -60.0, 60.0)
            p_def = 1.0 / (1.0 + np.exp(diff))
            v_exp = sigma_eps * np.logaddexp(v_rep / sigma_eps, v_def[None, :] / sigma_eps)

        # Expected continuation value EV = V @ P_trans.T of shape (n_b, n_y)
        ev = v_exp @ p_trans.T

        if monotone:
            if min_candidate_consumption(b_grid, resources, q) <= 1e-6:
                raise ValueError('Candidate consumption reaches the 1e-6 utility barrier, where '
                                 'utility is not concave; use policy_search="dense"')
            v_rep_new, m_opt = maximize(b_grid, resources, q, ev, beta, u_func)
            v_rep_new = with_flow(v_rep_new)
            c_opt = (resources[None, :] - 1.0 * b_grid[:, None]) + np.take_along_axis(q, m_opt, axis=0) * b_grid[m_opt]
        else:
            # Full tensor consumption c: shape (n_b, n_b, n_y)
            # c(i, m, j) = y_j - service * b_i + q(m, j) * [b_m - retention * b_i]
            q_next = q[None, :, :]
            c_tensor = y_cur - service_coeff * b_cur + q_next * (b_next - retention_coeff * b_cur)

            # Bellman objective
            w_tensor = u_func(c_tensor) + beta * ev[None, :, :]
            m_opt = np.argmax(w_tensor, axis=1)

        if delta < 0.999:
            # Long-term debt: smooth choice eliminates grid chattering and secures residual < 1e-7
            w_max = np.max(w_tensor, axis=1, keepdims=True)
            exp_w = np.exp(np.clip((w_tensor - w_max) / tau_b, -50.0, 50.0))
            prob_b = exp_w / np.sum(exp_w, axis=1, keepdims=True)
            v_rep_new = with_flow(np.sum(prob_b * w_tensor, axis=1))
            q_tomorrow = np.sum(prob_b * q[None, :, :], axis=1)
        else:
            # Short-term debt (Arellano 2008): delta = 1.0, retention = 0.0
            if not monotone:
                v_rep_new = with_flow(np.take_along_axis(w_tensor, m_opt[:, None, :], axis=1).squeeze(axis=1))
                c_opt = np.take_along_axis(c_tensor, m_opt[:, None, :], axis=1).squeeze(axis=1)
            # Howard policy iteration steps for fast convergence
            for _ in range(howard_steps):
                if sigma_eps == 0.0:
                    v_exp_h = np.maximum(v_rep_new, v_def[None, :])
                else:
                    v_exp_h = sigma_eps * np.logaddexp(v_rep_new / sigma_eps, v_def[None, :] / sigma_eps)
                ev_h = v_exp_h @ p_trans.T
                ev_opt = np.take_along_axis(ev_h, m_opt, axis=0)
                v_rep_new = with_flow(u_func(c_opt) + beta * ev_opt)
            q_tomorrow = np.take_along_axis(q, m_opt, axis=0)

        # Update default value
        # Re-entry wipes out debt. The first grid point need not be zero
        # when the government can save (negative debt).
        v_zero = np.array([np.interp(0.0, b_grid, v_exp[:, j]) for j in range(n_y)])
        ev0 = v_zero @ p_trans.T
        v_def_new = u_def + beta * (theta * ev0 + (1.0 - theta) * (v_def @ p_trans.T))

        # Update bond price schedule
        payoff_tomorrow = (1.0 - p_def) * (service_coeff + retention_coeff * q_tomorrow) + p_def * recovery_rate
        q_target = (payoff_tomorrow @ p_trans.T) / (1.0 + r_star)
        q_target = np.clip(q_target, 0.0, q_rf)

        if enforce_monotonicity:
            q_target = np.minimum.accumulate(q_target, axis=0)

        err_v = max(float(np.max(np.abs(v_rep_new - v_rep))), float(np.max(np.abs(v_def_new - v_def))))
        err_q = float(np.max(np.abs(q_target - q)))
        residual = max(err_v, err_q)

        v_rep = v_rep_new
        v_def = v_def_new
        q = (1.0 - damping) * q + damping * q_target

        if residual < tol:
            converged = True
            break

    # Report default decisions and total value at the returned value functions.
    if sigma_eps == 0.0:
        p_def = (v_def[None, :] > v_rep).astype(float)
        v_exp = np.maximum(v_rep, v_def[None, :])
    else:
        p_def = 1.0 / (1.0 + np.exp(np.clip((v_rep-v_def[None, :])/sigma_eps, -60.0, 60.0)))
        v_exp = sigma_eps * np.logaddexp(v_rep/sigma_eps, v_def[None, :]/sigma_eps)

    certification_gap = exhaustive_residual = None
    if monotone:
        # Certify the accelerator against every candidate at the returned prices.
        ev_final = v_exp @ p_trans.T
        accelerated, _ = maximize(b_grid, resources, q, ev_final, beta, u_func)
        exhaustive, _ = exhaustive_maximize(b_grid, resources, q, ev_final, beta, u_func)
        certification_gap = float(np.max(np.abs(accelerated - exhaustive)))
        if certification_gap > 1e-11:
            raise RuntimeError(f"Monotone policy search failed exhaustive certification: gap {certification_gap:.3e}")
        exhaustive_residual = float(np.max(np.abs(with_flow(exhaustive) - v_rep)))

    return _CoreSolution(
        v_rep=v_rep, v_def=v_def, v_exp=v_exp, q=q, p_def=p_def, m_opt=m_opt,
        residual=residual, converged=converged, iterations=iteration,
        certification_gap=certification_gap, exhaustive_residual=exhaustive_residual, q_rf=q_rf,
    )


def _spread_schedule(q: np.ndarray, delta: float, kappa: float, r_star: float) -> np.ndarray:
    """Annualized spread ``4 * max(0, service / q - delta - r_star)`` per (b', y)."""
    service_coeff = delta + (1.0 - delta) * kappa
    # Sovereign spread over risk-free rate
    # Yield R(b', y) = service_coeff / q - delta
    # Quarterly spread = max(0, R - r_star), annualized = 4 * spread
    q_safe = np.maximum(1e-8, q)
    yield_quarterly = service_coeff / q_safe - delta
    return 4.0 * np.maximum(0.0, yield_quarterly - r_star)


def _ergodic_spread_statistics(
    p_def: np.ndarray,
    m_opt: np.ndarray,
    spread: np.ndarray,
    p_trans: np.ndarray,
    y_grid: np.ndarray,
    theta: float,
    zero_idx: int,
    debt_probabilities: np.ndarray | None = None,
) -> tuple[float, float, str, int]:
    """Mean spread, spread-income correlation, its status and the observation count.

    One 1,500-period path with seed 42 from zero debt and the middle income
    node; reentry happens before the current period's decision.
    """
    n_y = y_grid.shape[0]
    sim_rng = np.random.default_rng(42)
    cur_y = n_y // 2
    cur_b = zero_idx
    in_def = False
    sim_spreads: list[float] = []
    sim_ys: list[float] = []
    for _ in range(1500):
        cur_y = int(sim_rng.choice(n_y, p=p_trans[cur_y]))
        y_val = float(y_grid[cur_y])
        if in_def:
            if sim_rng.random() < theta:
                in_def = False
                cur_b = zero_idx
        # A successful reentry draw gives access now, rather than consuming
        # another excluded period: V_D = u_D + beta E[theta V(0) + (1-theta)V_D].
        if not in_def:
            p_d = float(p_def[cur_b, cur_y])
            if sim_rng.random() < p_d:
                in_def = True
            else:
                b_next_idx = (int(m_opt[cur_b, cur_y]) if debt_probabilities is None else
                              int(sim_rng.choice(len(m_opt), p=debt_probabilities[cur_b, :, cur_y])))
                s_val = float(spread[b_next_idx, cur_y])
                sim_spreads.append(s_val)
                sim_ys.append(y_val)
                cur_b = b_next_idx

    spread_mean = float(np.mean(sim_spreads)) if sim_spreads else float("nan")
    if len(sim_spreads) > 50 and float(np.std(sim_spreads)) > 1e-7 and float(np.std(sim_ys)) > 1e-7:
        spread_y_corr = float(np.corrcoef(sim_spreads, sim_ys)[0, 1])
        corr_status = "simulated"
    else:
        spread_y_corr = float("nan")
        corr_status = "undefined_zero_variation" if len(sim_spreads) > 50 else "insufficient_observations"
    return spread_mean, spread_y_corr, corr_status, len(sim_spreads)


def solve_sovereign_default(
    beta: float = 0.953,
    gamma: float = 2.0,
    r_star: float = 0.01,
    delta: float = 1.0,
    kappa: float = 0.0,
    sigma_eps: float = 0.01,
    *,
    rho_y: float = 0.945,
    sigma_y: float = 0.025,
    theta: float = 0.282,
    d0: float = -0.188,
    d1: float = 0.245,
    n_b: int = 40,
    b_max: float = 0.40,
    b_min: float = 0.0,
    n_y: int = 15,
    m_tauchen: float = 3.0,
    tol: float = 1e-7,
    max_iter: int = 2000,
    damping: float = 0.40,
    recovery_rate: float = 0.0,
    howard_steps: int = 15,
    enforce_monotonicity: bool = True,
    y_penalty_type: str = "chatterjee_eyigungor",
    policy_search: str = "dense",
    debt_grid: np.ndarray | None = None,
) -> SovereignDefaultResult:
    """Solve sovereign default with short- or long-term debt and taste shocks.

    Supports short-term debt (delta = 1.0, kappa = 0.0, Arellano 2008) and
    long-term maturing coupon bonds (delta in (0, 1], kappa >= 0, Chatterjee &
    Eyigungor 2012).

    Parameters
    ----------
    beta : float, default 0.953
        Subjective quarterly discount factor.
    gamma : float, default 2.0
        CRRA relative risk aversion parameter u(c) = (c^(1-gamma) - 1)/(1-gamma).
    r_star : float, default 0.01
        Quarterly world risk-free interest rate.
    delta : float, default 1.0
        Decay/maturing fraction of debt per period (average maturity = 1/delta).
    kappa : float, default 0.0
        Coupon payment rate on debt.
    sigma_eps : float, default 0.01
        Extreme-value taste shock dispersion; zero gives deterministic default choice.
    rho_y : float, default 0.945
        Autoregressive persistence of log endowment.
    sigma_y : float, default 0.025
        Innovation standard deviation of endowment shock.
    theta : float, default 0.282
        Probability in [0, 1] of regaining market access in each period after
        default. Reentry occurs before that period's default/repayment choice,
        with debt reset to zero, as in the default Bellman equation.
    d0, d1 : float
        Parameters for Chatterjee-Eyigungor quadratic default output penalty.
    n_b : int, default 40
        Number of grid points for sovereign debt. If saving and borrowing are
        both allowed, at least three nodes are required; the nearest interior
        node is set to zero while preserving both bounds and the node count.
    b_max : float, default 0.40
        Upper bound on sovereign debt grid.
    b_min : float, default 0.0
        Lower bound on sovereign debt grid.
    n_y : int, default 15
        Number of grid points for endowment process.
    m_tauchen : float, default 3.0
        Number of standard deviations for Tauchen discretization.
    tol : float, default 1e-7
        Convergence tolerance on Bellman fixed-point residual.
    max_iter : int, default 2000
        Maximum number of value and price iterations.
    damping : float, default 0.40
        Damping factor for bond price schedule updates.
    recovery_rate : float, default 0.0
        Recovery fraction on defaulted bonds.
    howard_steps : int, default 15
        Number of Howard policy improvement steps per outer iteration.
    enforce_monotonicity : bool, default True
        If True, enforce dq/db' <= 0 at each iteration.
    y_penalty_type : str, default "chatterjee_eyigungor"
        Default output cost function: "chatterjee_eyigungor" or "arellano".
    policy_search : {"dense", "monotone"}, default "dense"
        Debt-choice search. "dense" evaluates every candidate at once and
        needs arrays of n_b * n_b * n_y entries. "monotone" (short-term debt
        only, delta = 1 and kappa = 0) exploits concave utility to search
        in O(n_b log n_b) per income state (gamma >= 0). It returns the same
        values as "dense" and the same policies except at exact value ties
        between choices with different bond proceeds. After convergence a
        blocked exhaustive search at the returned prices and values
        certifies it: a repayment-value gap above 1e-11 raises RuntimeError.
        It raises ValueError if any candidate consumption reaches the 1e-6
        utility barrier, where utility is not concave; use "dense" there.
    debt_grid : ndarray, optional
        Explicit debt coordinates, overriding ``n_b``, ``b_min`` and ``b_max``.
        Must be a finite, strictly increasing vector with at least two nodes
        and an exact zero. The solver copies it and preserves every node.
        Allows nonuniform refinement without changing the debt support.
        Price tangents on this grid use nonuniform PCHIP weights, clipped at
        zero as in the default tangent convention.

    Returns
    -------
    SovereignDefaultResult
        Solved value functions, policy function, bond price schedule,
        sovereign spreads, and PCHIP derivatives dq/db'.
    """
    if delta <= 0.0 or delta > 1.0:
        raise ValueError(f"delta must be in (0, 1]; got {delta}")
    if kappa < 0.0:
        raise ValueError(f"kappa must be >= 0; got {kappa}")
    if sigma_eps < 0.0 or not np.isfinite(sigma_eps):
        raise ValueError(f"sigma_eps must be finite and >= 0; got {sigma_eps}")
    if beta <= 0.0 or beta >= 1.0:
        raise ValueError(f"beta must be in (0, 1); got {beta}")
    if r_star <= 0.0:
        raise ValueError(f"r_star must be > 0; got {r_star}")
    if not np.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta must be finite and in [0, 1]; got {theta}")
    if policy_search not in ("dense", "monotone"):
        raise ValueError(f'policy_search must be "dense" or "monotone"; got {policy_search!r}')
    monotone = policy_search == "monotone"
    if monotone and (delta != 1.0 or kappa != 0.0):
        raise ValueError('policy_search="monotone" supports short-term debt only (delta=1, kappa=0)')
    if monotone and gamma < 0.0:
        raise ValueError(f'policy_search="monotone" requires concave utility, gamma >= 0; got {gamma}')

    supplied_grid = None if debt_grid is None else _validate_debt_grid(debt_grid)
    y_grid, p_trans, stat_dist = _discretize_endowment(n_y, rho_y, sigma_y, m=m_tauchen)
    y_def = _compute_default_penalty(y_grid, penalty_type=y_penalty_type, d0=d0, d1=d1,
                                     mean_y=float(stat_dist @ y_grid))
    b_grid, zero_idx = (_make_debt_grid(b_min, b_max, n_b)
                        if supplied_grid is None else supplied_grid)

    core = _solve_bellman(
        b_grid=b_grid, p_trans=p_trans, resources=y_grid, resources_default=y_def,
        choice_utility=_crra_utility(gamma), flow_repay=None, flow_default=None,
        beta=beta, r_star=r_star, delta=delta, kappa=kappa, sigma_eps=sigma_eps, theta=theta,
        tol=tol, max_iter=max_iter, damping=damping, recovery_rate=recovery_rate,
        howard_steps=howard_steps, enforce_monotonicity=enforce_monotonicity,
        policy_search=policy_search,
    )
    v_rep, v_def, v_exp, q, p_def, m_opt = core.v_rep, core.v_def, core.v_exp, core.q, core.p_def, core.m_opt
    residual, converged, iteration = core.residual, core.converged, core.iterations

    # Final policy b'(b, y)
    policy_b = b_grid[m_opt]

    # Exact PCHIP node derivatives dq/db'(b', y)
    if supplied_grid is None:
        dq_db = _compute_dq_db_dualarray(q, b_grid)
    else:
        from scipy.interpolate import PchipInterpolator

        dq_db = np.minimum(PchipInterpolator(b_grid, q, axis=0).derivative()(b_grid), 0.0)

    spread_annualized = _spread_schedule(q, delta, kappa, r_star)

    debt_probabilities = None
    deterministic_gap = None
    if delta < .999:
        # Pricing integrates over this smoothed policy. Simulating only its
        # modal debt choice describes a different transition law.
        service = delta + (1 - delta) * kappa
        consumption = (y_grid[None, None, :] - service * b_grid[:, None, None]
                       + q[None, :, :] * (b_grid[None, :, None] - (1 - delta) * b_grid[:, None, None]))
        objective = _crra_utility(gamma)(consumption) + beta * (v_exp @ p_trans.T)[None, :, :]
        best = np.max(objective, axis=1, keepdims=True)
        temperature = min(.005, max(.0001, sigma_eps / 2))
        weights = np.exp(np.clip((objective - best) / temperature, -50., 50.))
        debt_probabilities = weights / weights.sum(axis=1, keepdims=True)
        deterministic_gap = float(np.max(np.abs(best[:, 0, :] - v_rep)))

    # Ergodic simulation to compute spread_mean and countercyclical correlation
    spread_mean, spread_y_corr, corr_status, spread_observations = _ergodic_spread_statistics(
        p_def, m_opt, spread_annualized, p_trans, y_grid, theta, zero_idx, debt_probabilities)

    metadata: dict[str, Any] = {
        "theta": theta,
        "gamma": gamma,
        "beta": beta,
        "r_star": r_star,
        "rho_y": rho_y,
        "sigma_y": sigma_y,
        "y_penalty_type": y_penalty_type,
        "policy_search": policy_search,
        "monotone_certification_gap": core.certification_gap,
        "exhaustive_bellman_residual": core.exhaustive_residual,
        "d0": d0,
        "d1": d1,
        "m_tauchen": m_tauchen,
        "recovery_rate": recovery_rate,
        "damping": damping,
        "howard_steps": howard_steps,
        "enforce_monotonicity": enforce_monotonicity,
        "tol": tol,
        "max_iter": max_iter,
        "y_default_vec": y_def,
        "stationary_dist": stat_dist,
        "q_rf": core.q_rf,
        "spread_observations": spread_observations,
        "spread_y_corr_status": corr_status,
        "simulation_reentry_timing": "before_current_decision",
        "simulation_reset_debt": float(b_grid[zero_idx]),
        "simulation_periods": 1500,
        "simulation_seed": 42,
        "simulation_burn_in": 0,
        "spread_sample": "repayment_periods",
        "spread_annualization": "4 * max(0, quarterly_yield - r_star)",
        "income_normalization": "stationary_mean_one",
    }
    if supplied_grid is not None:
        metadata["debt_grid_supplied"] = True
    if debt_probabilities is not None:
        metadata.update(debt_choice_probabilities=debt_probabilities,
                        debt_choice_temperature=temperature,
                        debt_choice_rule='smoothed_quantal_policy',
                        reported_policy_b='modal_debt_choice',
                        deterministic_optimality_gap=deterministic_gap)

    return SovereignDefaultResult(
        b_grid=b_grid,
        y_grid=y_grid,
        v_r=v_rep,
        v_d=v_def,
        v_star=v_exp,
        q_bond=q,
        default_prob=p_def,
        policy_b_prime=policy_b,
        spread=spread_annualized,
        spread_mean=spread_mean,
        spread_y_corr=spread_y_corr,
        bellman_residual=residual,
        converged=converged,
        v_repay=v_rep,
        v_default=v_def,
        v_expected=v_exp,
        policy_b=policy_b,
        bond_price=q,
        dq_db=dq_db,
        p_trans=p_trans,
        maturity_param_delta=delta,
        coupon_rate_kappa=kappa,
        taste_shock_scale=sigma_eps,
        residual=residual,
        iterations=iteration,
        metadata=metadata,
    )
