"""Sovereign default with a government-spending block.

The government taxes a fixed share ``tau`` of the endowment, chooses next
period's short-term debt and spends the rest:

    g = tau*y - b + q(b', y) b'.

Private consumption ``c = (1 - tau) y`` is outside its control, so the debt
choice is the short-term Arellano (2008) problem with resources ``tau*y`` and
utility ``psi * v(g)``; ``u((1 - tau) y)`` is an income-only flow added to the
repayment and default values. The same Bellman core as
:func:`puremacro.sovereign.solve_sovereign_default` solves it, including the
certified monotone policy search.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np

from puremacro.sovereign._results import FiscalSovereignResult
from puremacro.sovereign.rules import FiscalRule, _compile_rule, _infeasible_message
from puremacro.sovereign.arellano import (
    _compute_default_penalty,
    _compute_dq_db_dualarray,
    _crra_utility,
    _discretize_endowment,
    _ergodic_spread_statistics,
    _make_debt_grid,
    _solve_bellman,
    _spread_schedule,
)

__all__ = ["solve_fiscal_sovereign_default"]


def _finite(x: float) -> bool:
    return bool(np.isfinite(x))


def solve_fiscal_sovereign_default(
    *,
    tau: float,
    psi: float,
    gamma_c: float = 2.0,
    gamma_g: float = 2.0,
    beta: float = 0.95282,
    r_star: float = 0.017,
    theta: float = 0.282,
    rho_y: float = 0.945,
    sigma_y: float = 0.025,
    y_penalty_type: str = "arellano",
    b_min: float = -0.03,
    b_max: float = 0.066,
    n_b: int = 401,
    n_y: int = 21,
    sigma_eps: float = 0.0,
    policy_search: str = "monotone",
    tol: float = 1e-10,
    max_iter: int = 10000,
    private_consumption: bool = True,
    d0: float = -0.188,
    d1: float = 0.245,
    m_tauchen: float = 3.0,
    damping: float = 0.40,
    recovery_rate: float = 0.0,
    howard_steps: int = 15,
    enforce_monotonicity: bool = False,
    rule: FiscalRule | None = None,
) -> FiscalSovereignResult:
    r"""Solve a short-term sovereign default model with government spending in utility.

    Model (quarterly; mean endowment 1)::

        V_R(b, y) = max_{b'} u((1 - tau) y) + psi v(g) + beta E V(b', y'),
                    g = tau y - b + q(b', y) b'
        V_D(y)    = u((1 - tau) y_def) + psi v(tau y_def)
                    + beta E[theta V(0, y') + (1 - theta) V_D(y')]
        V(b, y)   = max(V_R, V_D)  (log-sum-exp with taste shocks sigma_eps > 0)
        q(b', y)  = E[1 - default(b', y') | y] / (1 + r_star)

    ``u`` and ``v`` are CRRA with curvatures ``gamma_c`` and ``gamma_g``;
    ``y_def`` is the Arellano (2008) or Chatterjee-Eyigungor output under
    default. Primary balance ``pb = tau y - g = b - q b'``; debt-to-output
    ``b'/y``.

    ``tau`` and ``psi`` have no defaults. ``tau=0.2, psi=1`` with the
    remaining defaults is an *illustrative* calibration (public Arellano
    parameters, debt grid scaled by 0.2); it is not fitted to any data.
    Debt and spending are in units of mean output per quarter: an annual
    debt-to-GDP ratio is a quarter of these numbers.

    Parameters
    ----------
    tau : float
        Tax share of the endowment, in (0, 1]. ``tau=1`` needs
        ``private_consumption=False`` (private consumption would be zero).
    psi : float
        Weight on public-spending utility, finite and > 0. Spending at or
        below 1e-6 meets the utility barrier ``psi * (-1e12 + 1e6 g)``, so
        in dense mode the barrier's strength scales with ``psi``. Checked
        for psi from 1e-18 to 1 (dense, ``n_b=61, n_y=7, b_max=0.5``): no
        repayment state chose such spending. Below ``psi = 1e-12`` the
        barrier is worth less than one util and is not covered by tests.
        The monotone search raises ValueError whenever a candidate reaches
        the barrier, whatever ``psi``.
    gamma_c, gamma_g : float, default 2.0
        CRRA curvatures of private and public consumption (>= 0).
    beta, r_star, theta, rho_y, sigma_y, y_penalty_type, d0, d1, m_tauchen
        As in :func:`puremacro.sovereign.solve_sovereign_default`; defaults
        are the public Arellano (2008) calibration.
    b_min, b_max : float
        Debt grid bounds (must include zero); defaults are the public grid
        scaled by 0.2.
    n_b, n_y : int
        Debt and income grid sizes.
    sigma_eps : float, default 0.0
        Extreme-value taste-shock scale for the default choice.
    policy_search : {"monotone", "dense", "exhaustive"}, default "monotone"
        "monotone" searches the debt choice in O(n_b log n_b) per income
        state using the concavity of ``psi v``; it returns the same values as
        "dense" and the same policies except at exact value ties between
        choices with different proceeds. After convergence a blocked
        exhaustive search at the returned prices certifies it: a gap above
        1e-11 raises RuntimeError. It raises ValueError if any candidate
        spending reaches the 1e-6 utility barrier; use "dense" there.
        "exhaustive" runs the blocked exhaustive search every iteration
        (O(n_b^2 n_y) work, O(128 n_b) memory); a deficit limit needs it
        or "dense". Debt ceilings, escape clauses and penalties work with
        all three.
    tol, max_iter, damping, recovery_rate, howard_steps, enforce_monotonicity
        Fixed-point controls, as in ``solve_sovereign_default``.
    private_consumption : bool, default True
        Include the flow ``u((1 - tau) y)``. It does not affect the debt
        choice, only the values (and so the default decision).
    rule : FiscalRule, optional
        A fiscal rule known to lenders (it changes the price schedule).
        Thresholds are scaled by the stationary mean of ``y`` (one). States
        where a hard rule leaves no candidate raise ValueError before
        solving, or default for sure with ``infeasible="default"``.

    Returns
    -------
    FiscalSovereignResult
        The sovereign result plus ``policy_index``, ``g_policy``,
        ``g_default``, ``primary_balance``, ``debt_to_gdp`` and private
        consumption in repayment and default.
    """
    if not _finite(tau) or not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1]; got {tau}")
    if tau == 1.0 and private_consumption:
        raise ValueError("tau=1 leaves zero private consumption; set private_consumption=False")
    if not _finite(psi) or psi <= 0.0:
        raise ValueError(f"psi must be finite and > 0; got {psi}")
    for name, value in (("gamma_c", gamma_c), ("gamma_g", gamma_g)):
        if not _finite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0; got {value}")
    if sigma_eps < 0.0 or not np.isfinite(sigma_eps):
        raise ValueError(f"sigma_eps must be finite and >= 0; got {sigma_eps}")
    if beta <= 0.0 or beta >= 1.0:
        raise ValueError(f"beta must be in (0, 1); got {beta}")
    if r_star <= 0.0:
        raise ValueError(f"r_star must be > 0; got {r_star}")
    if not np.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError(f"theta must be finite and in [0, 1]; got {theta}")
    if policy_search not in ("dense", "monotone", "exhaustive"):
        raise ValueError(f'policy_search must be "dense", "monotone" or "exhaustive"; got {policy_search!r}')
    if rule is not None and not isinstance(rule, FiscalRule):
        raise TypeError(f"rule must be a FiscalRule or None; got {type(rule).__name__}")
    if rule is not None and rule.deficit_limit is not None and policy_search == "monotone":
        raise ValueError('A deficit limit is not monotone-compatible; use policy_search="exhaustive" '
                         '(or "dense" on small grids)')

    y_grid, p_trans, stat_dist = _discretize_endowment(n_y, rho_y, sigma_y, m=m_tauchen)
    y_def = _compute_default_penalty(y_grid, penalty_type=y_penalty_type, d0=d0, d1=d1,
                                     mean_y=float(stat_dist @ y_grid))
    b_grid, zero_idx = _make_debt_grid(b_min, b_max, n_b)
    compiled = None
    if rule is not None:
        compiled = _compile_rule(rule, b_grid, y_grid, float(stat_dist @ y_grid))
        if compiled.infeasible.any() and rule.infeasible == "raise":
            raise ValueError(_infeasible_message(compiled, b_grid, y_grid))

    spending_utility = _crra_utility(gamma_g)

    def choice_utility(g: np.ndarray) -> np.ndarray:
        # psi * v(g); 1.0 * x is exact, so tau = psi = 1 reproduces the plain solver.
        return psi * spending_utility(g)

    resources = tau * y_grid
    resources_default = tau * y_def
    flow_repay = flow_default = None
    if private_consumption:
        private_utility: Callable[[np.ndarray], np.ndarray] = _crra_utility(gamma_c)
        flow_repay = private_utility((1.0 - tau) * y_grid)
        flow_default = private_utility((1.0 - tau) * y_def)

    core = _solve_bellman(
        b_grid=b_grid, p_trans=p_trans, resources=resources, resources_default=resources_default,
        choice_utility=choice_utility, flow_repay=flow_repay, flow_default=flow_default,
        beta=beta, r_star=r_star, delta=1.0, kappa=0.0, sigma_eps=sigma_eps, theta=theta,
        tol=tol, max_iter=max_iter, damping=damping, recovery_rate=recovery_rate,
        howard_steps=howard_steps, enforce_monotonicity=enforce_monotonicity,
        policy_search=policy_search, rule=compiled,
    )
    q, m_opt = core.q, core.m_opt
    policy_b = b_grid[m_opt]
    dq_db = _compute_dq_db_dualarray(q, b_grid)
    spread = _spread_schedule(q, 1.0, 0.0, r_star)
    spread_mean, spread_y_corr, corr_status, spread_observations = _ergodic_spread_statistics(
        core.p_def, m_opt, spread, p_trans, y_grid, theta, zero_idx)

    q_chosen = np.take_along_axis(q, m_opt, axis=0)
    g_policy = tau * y_grid[None, :] - b_grid[:, None] + q_chosen * policy_b
    primary_balance = b_grid[:, None] - q_chosen * policy_b

    metadata: dict[str, Any] = {
        "theta": theta,
        "gamma_c": gamma_c,
        "gamma_g": gamma_g,
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
        "tau": tau,
        "psi": psi,
        "private_consumption": private_consumption,
        "resources": "tau * y",
        "rule": None if rule is None else dataclasses.asdict(rule),
        "rule_infeasible_states": 0 if compiled is None else int(compiled.infeasible.sum()),
        "zero_index": zero_idx,
    }

    return FiscalSovereignResult(
        b_grid=b_grid,
        y_grid=y_grid,
        v_r=core.v_rep,
        v_d=core.v_def,
        v_star=core.v_exp,
        q_bond=q,
        default_prob=core.p_def,
        policy_b_prime=policy_b,
        spread=spread,
        spread_mean=spread_mean,
        spread_y_corr=spread_y_corr,
        bellman_residual=core.residual,
        converged=core.converged,
        v_repay=core.v_rep,
        v_default=core.v_def,
        v_expected=core.v_exp,
        policy_b=policy_b,
        bond_price=q,
        dq_db=dq_db,
        p_trans=p_trans,
        maturity_param_delta=1.0,
        coupon_rate_kappa=0.0,
        taste_shock_scale=sigma_eps,
        residual=core.residual,
        iterations=core.iterations,
        metadata=metadata,
        tau=tau,
        psi=psi,
        gamma_c=gamma_c,
        gamma_g=gamma_g,
        private_consumption=private_consumption,
        rule=rule,
        policy_index=m_opt,
        g_policy=g_policy,
        g_default=tau * y_def,
        primary_balance=primary_balance,
        debt_to_gdp=policy_b / y_grid[None, :],
        private_consumption_repay=(1.0 - tau) * y_grid,
        private_consumption_default=(1.0 - tau) * y_def,
    )
