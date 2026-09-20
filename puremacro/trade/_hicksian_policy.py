"""Fixed-baseline Hicksian tariff searches; legacy policy objectives stay separate."""
from __future__ import annotations

import hashlib
from numbers import Integral
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ._results import OptimalTariffResult, NashTariffResult, WelfarePayoffMatrixResult
from .policy_solver import solve_policy_equilibrium, PolicyEquilibriumError
from .welfare import compute_hicksian_welfare, _checked_state


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _count(value, name, minimum):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


class _Context:
    def __init__(self, calib, x0, policy_mode, *, consumption_categories=(0,), base_equilibrium=None,
                 sigma=0., ge_tol=1e-8, ge_method="auto", ge_max_iter=100, ge_max_steps=100,
                 accounting="consistent"):
        from .optimal_tariffs import resolve_country_indices
        if accounting != "consistent":
            raise ValueError('metric="hicksian_ev" requires accounting="consistent"')
        if policy_mode not in ("universal", "final_only", "intermediate_only"):
            raise ValueError("policy_mode must be universal, final_only or intermediate_only")
        _positive(ge_tol, "ge_tol")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        if ge_method not in ("auto", "newton", "hybr", "keller_pac"):
            raise ValueError("ge_method must be auto, newton, hybr or keller_pac")
        _count(ge_max_iter, "ge_max_iter", 1)
        _count(ge_max_steps, "ge_max_steps", 1)
        self.calib, self.policy_mode = calib, policy_mode
        self.categories = tuple(consumption_categories)
        self.options = dict(sigma=sigma, tol=ge_tol, method=ge_method,
                            max_iter=ge_max_iter, max_steps=ge_max_steps)
        self.runs, self.cache = [], {}
        self.base = base_equilibrium
        if self.base is None:
            self.base = self._solve(None, None, x0, {})
        # Fixed reference must be the same zero-tariff economy for every player
        # and deviation; changing the reference would change reported payoffs.
        for key in ("intermediate_tariff_multipliers", "final_tariff_multipliers"):
            schedule = self.base.metadata.get(key)
            if schedule is None or not np.all(np.asarray(schedule) == 1.):
                raise ValueError("Policy comparison baseline must be a consistent zero-tariff equilibrium")
        if self.base.metadata.get("sigma") != sigma:
            raise ValueError("Policy baseline must use the requested sigma")
        blocks, _ = _checked_state(self.base, calib, 1e-8)
        residual = np.r_[blocks["residuals"], blocks["physical_residuals"]]
        if not np.isfinite(residual).all() or np.max(np.abs(residual)) > ge_tol:
            raise ValueError("Policy baseline audit exceeds the requested ge_tol")
        self.expenses = np.array([compute_hicksian_welfare(
            calib, self.base, base_result=self.base, target_country=i,
            consumption_categories=self.categories).consumption_base for i in range(calib.nc)])
        self.warm = self.base.x_sol.copy()
        self.reference_id = hashlib.sha256(
            np.asarray(self.base.Pfd_final).tobytes()+np.asarray(self.base.c_fd).tobytes()
            +repr(self.categories).encode()).hexdigest()
        self.resolve = resolve_country_indices

    def indices(self, player):
        indices = self.resolve(self.calib, player)
        if not indices or any(i < 0 or i >= self.calib.nc for i in indices):
            raise ValueError(f"Player {player!r} has no valid countries in this calibration")
        return indices

    def players(self, players):
        from .optimal_tariffs import get_country_code
        normalized, used = [], set()
        for player in players:
            code = get_country_code(self.calib, player)
            indices = self.indices(code)
            if used.intersection(indices):
                raise ValueError("Strategic players must be distinct and non-overlapping")
            used.update(indices); normalized.append(code)
        if not normalized:
            raise ValueError("Supply at least one strategic player")
        return tuple(normalized)

    def _solve(self, ta, tf, seed, profile):
        try:
            eq = solve_policy_equilibrium(self.calib, ta, tf, x0=seed, **self.options)
        except PolicyEquilibriumError as exc:
            self.runs.append({"profile": dict(profile), "attempts": exc.attempts, "accepted": False})
            exc.profile = dict(profile)
            raise
        self.runs.append({"profile": dict(profile), "attempts": eq.metadata["policy_solver_attempts"], "accepted": True})
        return eq

    def evaluate(self, profile, target_countries=None):
        from .optimal_tariffs import build_strategic_tariffs
        for player, rate in profile.items():
            self.indices(player)
            if not np.isfinite(rate) or rate < 0:
                raise ValueError("Policy tariff rates must be finite and nonnegative")
        targets = None
        if target_countries is not None:
            if not self.calib.country_codes:
                raise ValueError("Named target countries require calibration country codes")
            ids = set(i for p in target_countries for i in self.indices(p))
            if not ids:
                raise ValueError("Target countries cannot be empty")
            targets = tuple(self.calib.country_codes[i] for i in sorted(ids))
        key = (tuple(sorted((p, float(v)) for p, v in profile.items())), targets)
        if key not in self.cache:
            ta, tf = build_strategic_tariffs(self.calib, profile, policy_mode=self.policy_mode,
                                            target_countries=targets)
            eq = self.base if np.all(ta == 1.) and np.all(tf == 1.) else self._solve(ta, tf, self.warm, profile)
            # Failures do not enter this cache or update the warm start.
            ev = np.array([compute_hicksian_welfare(self.calib, eq, base_result=self.base,
                target_country=i, consumption_categories=self.categories).ev for i in range(self.calib.nc)])
            if not np.isfinite(ev).all():
                raise ValueError("Nonfinite Hicksian policy payoff")
            self.warm = eq.x_sol.copy()
            self.cache[key] = eq, ev, ta, tf
        return self.cache[key]

    def denominator(self, player):
        return float(self.expenses[self.indices(player)].sum())

    def terms_of_trade(self, eq, player):
        members = self.indices(player)
        # No external-trade price index has been derived for multi-country
        # blocs. Reporting the first member's index as a bloc total is misleading.
        return float(eq.terms_of_trade[members[0]]) if len(members) == 1 else float("nan")

    def payoff(self, profile, player, targets=None):
        return float(self.evaluate(profile, targets)[1][self.indices(player)].sum())

    def metadata(self):
        return {"accounting": "consistent", "welfare_metric": "hicksian_ev",
                "consumption_categories": self.categories, "comparison_baseline": "fixed zero-tariff equilibrium",
                "comparison_baseline_id": self.reference_id, "baseline_equilibrium": self.base,
                "baseline_consumption_by_country": self.expenses.copy(),
                "percent_denominator": "baseline selected consumption expenditure",
                "regret_denominator": "baseline selected consumption expenditure (fraction, not percent)",
                "terms_of_trade_scope": "individual country; unavailable for multi-country blocs",
                "sigma": self.options["sigma"], "ge_tol": self.options["tol"],
                "policy_mode": self.policy_mode, "solver_runs": list(self.runs),
                "unit": self.calib.metadata.get("unit", "calibration value units")}


def _search(objective, maximum, size, tol, current=0., refine=True):
    """Cover the interval and refine each sampled local maximum, including ends."""
    grid = np.unique(np.r_[np.linspace(0., maximum, size), current])
    values = np.array([objective(float(rate)) for rate in grid])
    if not np.isfinite(values).all():
        raise RuntimeError("Deviation coverage is incomplete; no best response is available")
    candidates = list(zip(values, grid))
    if refine and np.ptp(values) != 0:
        for j in range(len(grid)):
            if (j == 0 or values[j] >= values[j-1]) and (j == len(grid)-1 or values[j] >= values[j+1]):
                lo, hi = grid[max(0, j-1)], grid[min(len(grid)-1, j+1)]
                result = minimize_scalar(lambda rate: -objective(float(rate)), bounds=(lo, hi),
                                         method="bounded", options={"xatol": tol, "maxiter": 100})
                if not result.success or not np.isfinite(result.fun):
                    raise RuntimeError("Best-response refinement failed; optimization is unresolved")
                candidates.append((-float(result.fun), float(result.x)))
    # Retain the incumbent if payoffs tie, so indifference is not a tariff gap.
    value, rate = max(candidates, key=lambda pair: (pair[0], -abs(pair[1]-current)))
    return float(rate), float(value), grid, values


def _optimal(ctx, player, targets, maximum, size, tol, method):
    _positive(maximum, "tariff_max"); _positive(tol, "tol"); _count(size, "num_grid", 2)
    if method not in ("grid", "bounded"):
        raise ValueError("method must be grid or bounded")
    player = ctx.players([player])[0]
    rate, value, grid, curve = _search(lambda t: ctx.payoff({player: t}, player, targets),
                                      maximum, size, tol, refine=method == "bounded")
    eq = ctx.evaluate({player: rate}, targets)[0]
    return OptimalTariffResult(country_code=player, optimal_tariff_rate=rate,
        welfare_gain_pct=100*value/ctx.denominator(player), baseline_welfare=0., optimal_welfare=value,
        welfare_metric="hicksian_ev", terms_of_trade_initial=ctx.terms_of_trade(ctx.base, player),
        terms_of_trade_optimal=ctx.terms_of_trade(eq, player), tariff_grid=grid, welfare_curve=curve,
        equilibrium=eq, target_countries=None if targets is None else tuple(targets),
        metadata={**ctx.metadata(), "num_grid": size, "tariff_max": maximum, "tol": tol,
                  "method": method, "boundary": "lower" if rate <= tol else "upper" if rate >= maximum-tol else "interior",
                  "search_scope": "full-interval grid and all sampled local-peak brackets; no global proof"})


def unilateral(calib, country_idx, target_countries, tariff_max, num_grid, tol, x0, method, policy_mode, **kwargs):
    ctx = _Context(calib, x0, policy_mode, **kwargs)
    return _optimal(ctx, country_idx, target_countries, tariff_max, num_grid, tol, method)


def nash(calib, player_countries, method, relaxation, tol, max_iter, tariff_max, x0,
         policy_mode, initial_tariffs, *, best_response_grid_size=9, regret_tol=None, **kwargs):
    _positive(tol, "tol"); _positive(tariff_max, "tariff_max")
    _count(max_iter, "max_iter", 1); _count(best_response_grid_size, "best_response_grid_size", 3)
    if method != "best_response":
        raise NotImplementedError("Hicksian tariff games currently support best_response only")
    if not np.isfinite(relaxation) or not 0 < relaxation <= 1:
        raise ValueError("Require 0 < relaxation <= 1")
    regret_tol = tol if regret_tol is None else regret_tol
    if not np.isfinite(regret_tol) or regret_tol < 0:
        raise ValueError("regret_tol must be finite and nonnegative")
    start = time.perf_counter()
    ctx = _Context(calib, x0, policy_mode, **kwargs)
    players = ctx.players(player_countries)
    supplied = {} if initial_tariffs is None else dict(initial_tariffs)
    if set(supplied)-set(players):
        raise ValueError("Initial tariffs contain unknown strategic players")
    profile = {p: float(supplied.get(p, 0.)) for p in players}
    if any(not np.isfinite(v) or not 0 <= v <= tariff_max for v in profile.values()):
        raise ValueError("Initial tariffs must lie within [0, tariff_max]")

    def response(policy, player):
        return _search(lambda rate: ctx.payoff({**policy, player: rate}, player),
                       tariff_max, best_response_grid_size, max(tol*.1, 1e-10), policy[player])[:2]

    def verify(policy):
        eq, all_ev, ta, tf = ctx.evaluate(policy)
        payoffs = {p: float(all_ev[ctx.indices(p)].sum()) for p in players}
        replies, regrets = {}, {}
        for p in players:
            rate, value = response(policy, p)
            replies[p], regrets[p] = rate, max(0., value-payoffs[p])
        gap = max(abs(replies[p]-policy[p]) for p in players)
        regret = max(regrets[p]/ctx.denominator(p) for p in players)
        return eq, all_ev, ta, tf, payoffs, replies, regrets, gap, regret

    history, verified = [], None
    for iteration in range(max_iter):
        previous = profile.copy()
        update_gap = 0.
        for p in players:
            rate, _ = response(profile, p)
            update_gap = max(update_gap, abs(rate-profile[p]))
            profile[p] = (1-relaxation)*profile[p]+relaxation*rate
        history.append({"iteration": iteration+1, "tariffs": profile.copy(),
                        "undamped_update_gap": update_gap,
                        "step": max(abs(profile[p]-previous[p]) for p in players)})
        if update_gap <= tol:
            verified = verify(profile)
            if verified[-2] <= tol and verified[-1] <= regret_tol:
                break
            verified = None
    if verified is None:
        verified = verify(profile)
    eq, ev, ta, tf, payoffs, replies, regrets, gap, regret = verified
    tot_changes = {}
    for p in players:
        tot_changes[p] = 100*(ctx.terms_of_trade(eq, p)/ctx.terms_of_trade(ctx.base, p)-1)
    return NashTariffResult(strategic_players=players, nash_tariffs=profile,
        welfare_changes_pct={p: 100*payoffs[p]/ctx.denominator(p) for p in players},
        terms_of_trade_changes_pct=tot_changes,
        world_welfare_change_pct=float(100*ev.sum()/ctx.expenses.sum()),
        outer_iterations=len(history), converged=bool(gap <= tol and regret <= regret_tol),
        outer_error=gap, equilibrium=eq, tau_nash=ta, tau_fd_nash=tf, policy_mode=policy_mode,
        welfare_metric="hicksian_ev", method=method, player_welfares=payoffs,
        baseline_welfares={p: 0. for p in players}, iteration_history=history,
        metadata={**ctx.metadata(), "player_regrets": regrets, "max_regret": max(regrets.values()),
            "relative_max_regret": regret, "best_responses": replies,
            "best_response_boundaries": {p: "lower" if replies[p] <= tol else "upper" if replies[p] >= tariff_max-tol else "interior" for p in players},
            "best_response_grid_size": best_response_grid_size, "regret_tol": regret_tol,
            "tol": tol, "tariff_max": tariff_max, "relaxation": relaxation, "max_iter": max_iter,
            "world_welfare_scope": "all calibrated countries, including nonplayers",
            "world_ev": float(ev.sum()), "inner_solver_failures": [],
            "verification": "final simultaneous profile, full-interval grid and local refinement; no global proof",
            "duration_seconds": time.perf_counter()-start})


def payoff_matrix(calib, player_a, player_b, optimal_a, optimal_b, nash_a, nash_b, x0,
                  policy_mode, *, tariff_max=.5, num_grid=25, tol=1e-4, **kwargs):
    ctx = _Context(calib, x0, policy_mode, **kwargs)
    players = ctx.players([player_a, player_b])
    rates, sources = [], []
    for p, opt, supplied_nash in zip(players, (optimal_a, optimal_b), (nash_a, nash_b)):
        if opt is not None and supplied_nash is not None and opt != supplied_nash:
            raise ValueError("A normal-form action must use one tariff across cells; conflicting optimal/nash rates")
        rate = opt if opt is not None else supplied_nash
        if rate is None:
            rate = _optimal(ctx, p, None, tariff_max, num_grid, tol, "bounded").optimal_tariff_rate
            sources.append("unilateral optimum against zero foreign tariffs")
        else:
            sources.append("supplied fixed action; not verified as a continuous Nash rate")
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("Action tariffs must be finite and nonnegative")
        rates.append(float(rate))
    matrix = np.empty((2, 2, 2)); scenarios = {}; profiles = {}
    for i in range(2):
        for j in range(2):
            profile = {players[0]: rates[0]*i, players[1]: rates[1]*j}
            eq, ev, _, _ = ctx.evaluate(profile)
            key = ("D" if i else "C")+("D" if j else "C")
            scenarios[key], profiles[key] = eq, profile
            for k, p in enumerate(players):
                matrix[i, j, k] = 100*ev[ctx.indices(p)].sum()/ctx.denominator(p)
    labels = [[f"{p}: 0%", f"{p}: {100*rate:g}%"] for p, rate in zip(players, rates)]
    summary = pd.DataFrame([[f"({matrix[i,j,0]:+.4f}%, {matrix[i,j,1]:+.4f}%)"
                             for j in range(2)] for i in range(2)], index=labels[0], columns=labels[1])
    return WelfarePayoffMatrixResult(players=players, strategies=("Cooperate (0%)", "Fixed tariff action"),
        payoff_matrix=matrix, scenarios=scenarios, summary_df=summary, welfare_metric="hicksian_ev",
        metadata={**ctx.metadata(), "action_tariffs": dict(zip(players, rates)),
                  "action_sources": dict(zip(players, sources)), "cell_profiles": profiles,
                  "continuous_nash_verified": False})
