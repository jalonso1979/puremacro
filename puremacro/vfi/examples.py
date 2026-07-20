"""Worked examples that showcase the puremacro.vfi pipeline end to end.

These double as documentation and regression tests for the full
solve -> stationary distribution -> aggregates -> general-equilibrium stack.
"""
from __future__ import annotations

import numpy as np

from puremacro.vfi.aggregate import lorenz_and_gini
from puremacro.vfi.discretize import markov_stationary, tauchen
from puremacro.vfi.distribution import joint_transition_matrix
from puremacro.vfi.firm_dynamics import firm_stationary_distribution, free_entry_price
from puremacro.vfi.equilibrium import stationary_equilibrium
from puremacro.vfi.finite_horizon import (
    FiniteHorizonProblem,
    age_profile,
    cross_section,
    life_cycle_distribution,
)
from puremacro.vfi.problem import VFIProblem


def aiyagari_steady_state(*, alpha: float = 0.36, delta: float = 0.08,
                          beta: float = 0.96, gamma: float = 1.0,
                          rho: float = 0.9, sigma: float = 0.2,
                          n_z: int = 5, n_a: int = 150, a_max: float = 80.0,
                          r_bracket=None, backend: str = "numpy",
                          solve_options=None):
    """Solve the canonical Aiyagari (1994) stationary general equilibrium.

    Households face idiosyncratic AR(1) labor-productivity risk (Tauchen) and a
    no-borrowing constraint (a >= 0); the firm is Cobb-Douglas with depreciation
    ``delta``. ``gamma`` is the CRRA coefficient (gamma=1 -> log utility). Solves
    for the interest rate that clears the capital market and returns a dict:

      'equilibrium' : EquilibriumResult
      'r', 'w'      : equilibrium interest rate and wage
      'K', 'L', 'Y' : aggregate capital, effective labor, output
      'wealth_gini' : Gini of the cross-sectional wealth (asset) distribution

    Note: realistic wealth dispersion needs enough income states -- the default
    n_z=5 gives a stable wealth Gini ~0.6 (insensitive to n_a), whereas very few
    states (e.g. n_z=3) collapse the asset distribution toward a single node
    (Gini ~ 0). Use n_z >= 5 for a meaningful inequality statistic.
    """
    z_grid, P = tauchen(n=n_z, rho=rho, sigma=sigma)
    pi_z = markov_stationary(P)
    L = float(pi_z @ np.exp(z_grid))                       # aggregate effective labor
    a_grid = np.linspace(1e-4, a_max, n_a)
    solve_options = solve_options or dict(tol=1e-9, n_howard=40)

    def util(c, xp):
        cc = xp.maximum(c, 1e-12)
        if gamma == 1.0:
            return xp.log(cc)
        return (cc ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

    def KL_of_r(r):
        return (alpha / (r + delta)) ** (1.0 / (1.0 - alpha))

    def wage_of_r(r):
        return (1.0 - alpha) * KL_of_r(r) ** alpha

    def build_problem(r):
        w = wage_of_r(r)

        def rf(ap, a, z, r_, w_, xp=np):
            c = w_ * xp.exp(z) + (1.0 + r_) * a - ap
            return xp.where(c > 0.0, util(c, xp), -np.inf)

        return VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                          beta=beta, params={"r": r, "w": w}, options=solve_options)

    def market_residual(r, sol, mu, prob):
        K_supply = float(np.sum(mu * a_grid[:, None]))
        K_demand = L * KL_of_r(r)
        return K_supply - K_demand

    if r_bracket is None:
        r_bracket = (0.005, 1.0 / beta - 1.0 - 0.002)
    eq = stationary_equilibrium(build_problem, market_residual, r_bracket,
                                backend=backend)

    K = float(np.sum(eq.distribution * a_grid[:, None]))
    Y = K ** alpha * L ** (1.0 - alpha)
    wealth = np.broadcast_to(a_grid[:, None], eq.distribution.shape)
    _, _, wealth_gini = lorenz_and_gini(eq.distribution, wealth)
    return {
        "equilibrium": eq,
        "r": eq.price,
        "w": wage_of_r(eq.price),
        "K": K,
        "L": L,
        "Y": Y,
        "wealth_gini": wealth_gini,
    }


def huggett_steady_state(*, beta: float = 0.96, gamma: float = 1.5,
                         rho: float = 0.9, sigma: float = 0.2, n_z: int = 7,
                         n_a: int = 200, a_min: float = -2.0, a_max: float = 24.0,
                         r_bracket=None, backend: str = "numpy",
                         solve_options=None):
    """Solve the Huggett (1993) stationary equilibrium.

    A pure-exchange economy: a continuum of agents face idiosyncratic AR(1)
    endowment risk (Tauchen) and trade a single risk-free bond in ZERO net
    supply, borrowing down to ``a_min`` (< 0). The equilibrium interest rate
    clears the bond market -- aggregate net asset holdings equal 0. ``gamma`` is
    the CRRA coefficient. Returns a dict:

      'equilibrium'    : EquilibriumResult at the clearing rate
      'r'              : equilibrium interest rate (< 1/beta - 1)
      'mean_assets'    : aggregate net assets (~ 0 at the clearing rate)
      'frac_borrowing' : population share with a < 0

    Net assets can be negative, so (unlike Aiyagari) this reports the borrowing
    share rather than a wealth Gini. ``a_min`` must be loose enough that c > 0 is
    feasible at the borrowing limit for the lowest endowment.
    """
    z_grid, P = tauchen(n=n_z, rho=rho, sigma=sigma)
    a_grid = np.linspace(a_min, a_max, n_a)
    solve_options = solve_options or dict(tol=1e-9, n_howard=40)

    def util(c, xp):
        cc = xp.maximum(c, 1e-12)
        if gamma == 1.0:
            return xp.log(cc)
        return (cc ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

    def build_problem(r):
        def rf(ap, a, z, r_, xp=np):
            c = (1.0 + r_) * a + xp.exp(z) - ap
            return xp.where(c > 0.0, util(c, xp), -np.inf)

        return VFIProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                          beta=beta, params={"r": r}, options=solve_options)

    def market_residual(r, sol, mu, prob):
        return float(np.sum(mu * a_grid[:, None]))        # aggregate net assets

    if r_bracket is None:
        # wide search range: net asset demand is increasing in r with a unique
        # clearing rate; under strong precautionary motives (high risk/risk
        # aversion) that rate can be deeply negative, so bracket well below 0.
        r_bracket = (-0.5, 1.0 / beta - 1.0 - 0.002)
    eq = stationary_equilibrium(build_problem, market_residual, r_bracket,
                                backend=backend)

    mean_assets = float(np.sum(eq.distribution * a_grid[:, None]))
    frac_borrowing = float(eq.distribution[a_grid < 0.0].sum())
    return {
        "equilibrium": eq,
        "r": eq.price,
        "mean_assets": mean_assets,
        "frac_borrowing": frac_borrowing,
    }


def life_cycle_profile(*, beta: float = 0.96, gamma: float = 2.0, r: float = 0.04,
                       rho: float = 0.95, sigma: float = 0.15, horizon: int = 40,
                       n_z: int = 5, n_a: int = 120, a_max: float = 40.0,
                       backend: str = "numpy"):
    """Solve a finite-horizon (life-cycle) consumption-saving problem.

    Agents live ``horizon`` ages, earn a hump-shaped deterministic income
    ``kappa[age]`` times idiosyncratic AR(1) risk ``exp(z)``, save in a single
    asset at a fixed rate ``r`` (partial equilibrium), cannot borrow (a >= 0), and
    leave no bequest (so they run assets down toward the final age). Showcases the
    finite-horizon stack (`FiniteHorizonProblem` -> `life_cycle_distribution` ->
    `age_profile` / `cross_section`). Returns a dict:

      'solution'           : FiniteHorizonSolution (V/policy are age-indexed)
      'life_cycle_dist'    : (horizon, n_a, n_z) cohort distribution by age
      'assets_by_age'      : (horizon,) mean assets held at each age
      'consumption_by_age' : (horizon,) mean consumption at each age
      'cross_section'      : (n_a, n_z) population distribution (uniform cohorts)
    """
    z_grid, P = tauchen(n=n_z, rho=rho, sigma=sigma)
    a_grid = np.linspace(0.0, a_max, n_a)
    ages = np.arange(horizon)
    kappa = np.exp(0.1 * ages - 0.0025 * ages ** 2)        # hump-shaped earnings

    def util(c, xp):
        cc = xp.maximum(c, 1e-12)
        if gamma == 1.0:
            return xp.log(cc)
        return (cc ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

    def rf(ap, a, z, age, r_, xp=np):
        c = (1.0 + r_) * a + kappa[age] * xp.exp(z) - ap
        return xp.where(c > 0.0, util(c, xp), -np.inf)

    sol = FiniteHorizonProblem(a_grid=a_grid, z_grid=z_grid, P_z=P, return_fn=rf,
                               beta=beta, horizon=horizon,
                               params={"r": r}).solve(backend)
    dist = life_cycle_distribution(sol, P)
    assets_by_age = age_profile(dist, a_grid)

    aprime_val = a_grid[sol.policy_aprime]                  # (J, n_a, n_z) realized a'
    ez = np.exp(z_grid)[None, :]
    cons_by_age = np.empty(horizon)
    for j in range(horizon):
        c_j = (1.0 + r) * a_grid[:, None] + kappa[j] * ez - aprime_val[j]
        cons_by_age[j] = float(np.sum(dist[j] * c_j))

    return {
        "solution": sol,
        "life_cycle_dist": dist,
        "assets_by_age": assets_by_age,
        "consumption_by_age": cons_by_age,
        "cross_section": cross_section(dist),
    }


def two_asset_profile(*, beta: float = 0.96, gamma: float = 2.0,
                      r_liquid: float = 0.01, r_illiquid: float = 0.05,
                      adjust_cost: float = 0.05, rho: float = 0.9,
                      sigma: float = 0.25, n_z: int = 5, n_m: int = 20,
                      n_k: int = 20, m_max: float = 15.0, k_max: float = 15.0,
                      backend: str = "numpy"):
    """Solve a two-asset (liquid m + illiquid k) savings problem.

    Liquid is freely adjustable and earns ``r_liquid``; illiquid earns the higher
    ``r_illiquid`` but rebalancing it costs ``adjust_cost`` per unit moved
    (|k'-k|), so households hold liquid as a buffer against income risk and
    illiquid for its return. CRRA, AR(1) income, no borrowing. Showcases the
    multiple-endogenous-state API: ``a_grid`` is a LIST of grids (flat product),
    and aggregates use the separate-positional component convention. Returns a
    dict:

      'solution'       : VFISolution (policy over the flat (m,k) product)
      'distribution'   : (n_m*n_k, n_z) joint stationary measure
      'mean_liquid'    : mean liquid holdings E[m]
      'mean_illiquid'  : mean illiquid holdings E[k]
      'illiquid_share' : E[k] / E[m + k]
      'wealth_gini'    : Gini of total wealth (m + k)
    """
    z_grid, P = tauchen(n=n_z, rho=rho, sigma=sigma)
    m_grid = np.linspace(0.0, m_max, n_m)
    k_grid = np.linspace(0.0, k_max, n_k)

    def util(c, xp):
        cc = xp.maximum(c, 1e-12)
        if gamma == 1.0:
            return xp.log(cc)
        return (cc ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

    def rf(m_p, k_p, m, k, z, xp=np):
        adj = adjust_cost * xp.abs(k_p - k)
        c = (1.0 + r_liquid) * m + (1.0 + r_illiquid) * k + xp.exp(z) - m_p - k_p - adj
        return xp.where(c > 0.0, util(c, xp), -np.inf)

    prob = VFIProblem(a_grid=[m_grid, k_grid], z_grid=z_grid, P_z=P, return_fn=rf,  # type: ignore[arg-type]  # VFIProblem.a_grid accepts list at runtime but is annotated np.ndarray
                      beta=beta, options=dict(tol=1e-9, n_howard=30))
    sol = prob.solve(backend)
    mu = prob.stationary_distribution(sol)

    def liquid(m_p, k_p, m, k, z, *params, xp=np):
        return m + 0.0 * k

    def illiquid(m_p, k_p, m, k, z, *params, xp=np):
        return k + 0.0 * m

    mean_m = prob.aggregate(sol, mu, liquid)
    mean_k = prob.aggregate(sol, mu, illiquid)
    total = mean_m + mean_k
    illiquid_share = (mean_k / total) if total > 0.0 else 0.0
    mm, kk = np.meshgrid(m_grid, k_grid, indexing="ij")
    wealth = (mm + kk).reshape(-1)[:, None] + np.zeros((1, n_z))
    _, _, wealth_gini = lorenz_and_gini(mu, wealth)
    return {
        "solution": sol,
        "distribution": mu,
        "mean_liquid": mean_m,
        "mean_illiquid": mean_k,
        "illiquid_share": illiquid_share,
        "wealth_gini": wealth_gini,
    }


def neoclassical_growth(*, beta: float = 0.984, gamma: float = 2.0,
                        alpha: float = 0.35, delta: float = 0.01, rho: float = 0.95,
                        sigma: float = 0.005, n_z: int = 4, n_k: int = 400,
                        k_lo_mult: float = 0.7, k_hi_mult: float = 1.4,
                        backend: str = "numpy"):
    """Stochastic neoclassical growth model -- a port of VFIToolkit's
    StochasticNeoClassicalGrowthModel (Aldrich, Fernandez-Villaverde, Gallant &
    Rubio-Ramirez 2011 calibration; Diaz-Gimenez 2001).

    A representative agent chooses next capital ``k'`` under AR(1) TFP ``z``:
    ``c = exp(z) * k^alpha - (k' - (1-delta) k)``, CRRA utility (``gamma=1`` is
    log). Returns a dict:

      'K_ss'          : analytical deterministic steady-state capital
                        ``(alpha*beta/(1-beta(1-delta)))^(1/(1-alpha))``
      'solution'      : VFISolution (V, policy over k)
      'distribution'  : ergodic measure over (k, z)
      'mean_capital'  : E[k] (~ K_ss under the small TFP shocks)
      'k_grid'        : the capital grid

    The capital grid spans ``[k_lo_mult, k_hi_mult] * K_ss``.
    """
    z_grid, P = tauchen(n=n_z, rho=rho, sigma=sigma)
    K_ss = (alpha * beta / (1.0 - beta * (1.0 - delta))) ** (1.0 / (1.0 - alpha))
    k_grid = np.linspace(k_lo_mult * K_ss, k_hi_mult * K_ss, n_k)

    def util(c, xp):
        cc = xp.maximum(c, 1e-12)
        if gamma == 1.0:
            return xp.log(cc)
        return (cc ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

    def rf(kp, k, z, xp=np):
        c = xp.exp(z) * k ** alpha - (kp - (1.0 - delta) * k)
        return xp.where(c > 0.0, util(c, xp), -np.inf)

    prob = VFIProblem(a_grid=k_grid, z_grid=z_grid, P_z=P, return_fn=rf, beta=beta,
                      options=dict(tol=1e-10, n_howard=40))
    sol = prob.solve(backend)
    # near-deterministic capital dynamics (tiny TFP shocks) make the Tan two-step
    # iteration oscillate, so take the ergodic measure as the dominant left
    # eigenvector of the explicit joint transition operator (robust to periodicity).
    G = joint_transition_matrix(sol.policy_aprime, P)
    mu = markov_stationary(G).reshape(len(k_grid), len(z_grid))
    mean_k = float(np.sum(mu * k_grid[:, None]))
    return {
        "K_ss": K_ss,
        "solution": sol,
        "distribution": mu,
        "mean_capital": mean_k,
        "k_grid": k_grid,
    }


def hopenhayn_equilibrium(*, beta: float = 0.8, alpha: float = 2.0 / 3.0,
                          cf: float = 20.0, ce: float = 40.0, rho: float = 0.9,
                          sigma: float = 0.2, log_zbar: float = 1.4, n_z: int = 101,
                          price_bracket=(0.5, 20.0)):
    """Hopenhayn (1992) stationary competitive equilibrium with firm entry/exit.

    A port of VFIToolkit's Hopenhayn1992 example (numerical example from Chris
    Edmond's notes). Firms have AR(1) log-productivity (mean ``log_zbar``); each
    produces with DRS technology choosing labor ``n`` (profit
    ``pi* = max_n p*s*n^alpha - n - cf``, closed form), then decides whether to
    exit. The output price clears free entry (``E_nu[V] = ce``, entrant draw
    ``nu`` = the productivity stationary distribution). Returns a dict:

      'equilibrium'                 : FirmEntryExitEquilibrium
      'price'                       : equilibrium output price
      'distribution'                : stationary incumbent measure over s
      'exit_rate'                   : mass-weighted exit rate of incumbents
      'mean_incumbent_productivity' : E_g[s]
      'mean_entrant_productivity'   : E_nu[s]
    """
    z_grid, P = tauchen(n=n_z, rho=rho, sigma=sigma)
    s = np.exp(z_grid + log_zbar)                       # productivity (mean log = log_zbar)
    nu = markov_stationary(P)                           # entrant productivity distribution

    def profit_at(p):
        n_star = (alpha * p * s) ** (1.0 / (1.0 - alpha))
        return p * s * n_star ** alpha - n_star - cf

    eq = free_entry_price(profit_at, nu, ce, price_bracket, P_z=P, beta=beta)
    g = firm_stationary_distribution(P, eq.survive, nu)
    exit_rate = float(g @ (~eq.survive).astype(float))
    return {
        "equilibrium": eq,
        "price": eq.price,
        "distribution": g,
        "exit_rate": exit_rate,
        "mean_incumbent_productivity": float(g @ s),
        "mean_entrant_productivity": float(nu @ s),
    }


__all__ = ["aiyagari_steady_state", "huggett_steady_state", "life_cycle_profile",
           "two_asset_profile", "neoclassical_growth", "hopenhayn_equilibrium"]
