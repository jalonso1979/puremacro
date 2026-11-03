"""Finite aggregate-risk extension of the published one-asset HANK equations.

At a node, households know that node and its conditional successor law. EGM
uses the probability-weighted successor marginal value *before* choosing
saving and labor. Distributions then propagate conditional on each history.
The nominal policy rate is chosen at the parent and realized real returns
depend on child inflation. Idiosyncratic and aggregate shocks are independent.

The stochastic firm-valuation closure is explicitly M(child)=1/(1+r_child),
equivalent to risk-neutral valuation in nominal units. The Phillips equation
uses conditional expectations under that kernel. It reduces to the existing
published deterministic equations on a chain; the stochastic extension itself
is not a published-model replication or a stationary aggregate-risk solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import root

from .one_asset_hank import OneAssetHANKSteadyState, _backward, _forward

__all__ = ["HANKScenarioTreeResult", "monetary_scenario_tree", "solve_hank_scenario_tree"]


@dataclass(frozen=True)
class HANKScenarioTreeResult:
    nodes: pd.DataFrame
    paths: pd.DataFrame
    policy_a: np.ndarray
    policy_c: np.ndarray
    policy_n: np.ndarray
    distributions: np.ndarray
    steady_state: OneAssetHANKSteadyState
    converged: bool
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_frame(self) -> pd.DataFrame:
        return self.nodes.join(self.paths)

    def expected_paths(self) -> pd.DataFrame:
        """Unconditional expectations at each date, preserving node probabilities."""
        weighted = self.paths.mul(self.nodes["probability"], axis=0)
        return weighted.groupby(self.nodes["period"], sort=True).sum()


def monetary_scenario_tree(*, horizon: int, innovations=(-.0025, .0025),
                          probabilities=(.5, .5), branch_date: int = 1,
                          persistence: float = .61) -> pd.DataFrame:
    """One unresolved monetary innovation followed by deterministic decay.

    All branches are announced at date zero; the innovation is learned at
    ``branch_date``. Before that date the intercept shock is zero. A single
    branch may start at date zero and reproduces a perfect-foresight chain.
    Each row records a parent, conditional probability and monetary shock.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 2:
        raise ValueError("horizon must be an integer >= 2")
    if isinstance(branch_date, bool) or not isinstance(branch_date, (int, np.integer)) or not 0 <= branch_date < horizon:
        raise ValueError("branch_date must be an integer inside the horizon")
    if np.iscomplexobj(innovations) or np.iscomplexobj(probabilities):
        raise ValueError("innovations and probabilities must be real")
    eps, prob = np.asarray(innovations, dtype=float), np.asarray(probabilities, dtype=float)
    if eps.ndim != 1 or not eps.size or prob.shape != eps.shape or not np.isfinite(eps).all():
        raise ValueError("innovations and probabilities require matching nonempty vectors")
    if not np.isfinite(prob).all() or np.any(prob <= 0) or not np.isclose(prob.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("probabilities must be positive and sum to one")
    if len(eps) > 1 and branch_date == 0:
        raise ValueError("an unresolved innovation requires a shared root before branch_date")
    if not np.isfinite(persistence) or abs(persistence) >= 1:
        raise ValueError("persistence must be finite with absolute value below one")
    records = []
    for t in range(branch_date):
        records.append((f"common:{t}", None if t == 0 else f"common:{t-1}", 1., 0.))
    for t in range(branch_date, horizon):
        for j in range(len(eps)):
            parent = (None if t == 0 else f"common:{t-1}") if t == branch_date else f"branch{j}:{t-1}"
            records.append((f"branch{j}:{t}", parent, float(prob[j]) if t == branch_date else 1.,
                            float(eps[j] * persistence**(t-branch_date))))
    return pd.DataFrame(records, columns=["node", "parent", "conditional_probability", "monetary_shock"]).set_index("node")


def _tree(tree):
    required = ["parent", "conditional_probability", "monetary_shock"]
    if not isinstance(tree, pd.DataFrame) or not 2 <= len(tree) <= 256 or any(k not in tree for k in required):
        raise ValueError("tree requires 2..256 nodes with parent, conditional_probability and monetary_shock")
    if not tree.index.is_unique or any(not isinstance(k, str) or not k.strip() for k in tree.index):
        raise ValueError("node labels must be unique nonempty strings")
    nodes = tree[required].copy(deep=True)
    if np.iscomplexobj(nodes.conditional_probability) or np.iscomplexobj(nodes.monetary_shock):
        raise ValueError("tree probabilities and shocks must be real")
    conditional = nodes.conditional_probability.to_numpy(dtype=float)
    shocks = nodes.monetary_shock.to_numpy(dtype=float)
    if not np.isfinite(shocks).all() or not np.isfinite(conditional).all() or np.any(conditional <= 0) or np.any(conditional > 1):
        raise ValueError("tree requires finite shocks and probabilities in (0,1]")
    labels = dict(zip(nodes.index, range(len(nodes))))
    parents, depth = np.full(len(nodes), -1, dtype=int), np.zeros(len(nodes), dtype=int)
    probability = np.ones(len(nodes))
    children: list[list[int]] = [[] for _ in nodes.index]
    for j, parent in enumerate(nodes.parent):
        if j == 0:
            if not pd.isna(parent) or conditional[j] != 1:
                raise ValueError("first node must be the unique root with probability one")
            continue
        if parent not in labels or labels[parent] >= j:
            raise ValueError("each nonroot parent must occur earlier; recombining trees are unsupported")
        i = labels[parent]
        parents[j], depth[j] = i, depth[i] + 1
        children[i].append(j)
        probability[j] = probability[i] * conditional[j]
    for ch in children:
        if ch and not np.isclose(conditional[ch].sum(), 1., rtol=0, atol=1e-12):
            raise ValueError("conditional child probabilities must sum to one at every parent")
    leaf_depths = {int(depth[i]) for i, ch in enumerate(children) if not ch}
    if len(leaf_depths) != 1:
        raise ValueError("all terminal nodes must share the same horizon")
    nodes["period"], nodes["probability"] = depth, probability
    return nodes, parents, children, conditional, shocks


def solve_hank_scenario_tree(
    steady_state: OneAssetHANKSteadyState,
    tree: pd.DataFrame,
    *,
    tolerance: float = 1e-8,
    max_evaluations: int | None = None,
) -> HANKScenarioTreeResult:
    """Solve inflation, wages and output at every history of a finite tree.

    Asset, labor and conditional expected Phillips equations are solved
    jointly. Government debt is fixed; taxes service its realized return.
    Households have endogenous labor and saving, using the existing validated
    household equations. Each history retains its own distribution. At leaves,
    future marginal values and aggregates equal the supplied steady state.

    ``converged`` checks market residuals, budgets, conditional mass and
    policy-grid exits, not discretization accuracy or stationary stochastic
    welfare. Euler residuals and upper-grid mass are additional diagnostics. The declared firm
    valuation kernel is a modeling assumption, not inferred from household
    marginal utility. A Powell hybrid root solve updates the Jacobian between
    full finite-difference sweeps. ``max_evaluations`` includes Jacobian probes;
    ``None`` uses SciPy's dimension-dependent budget. Node limits keep dense
    numerical Jacobians bounded.
    """
    if not isinstance(steady_state, OneAssetHANKSteadyState) or not steady_state.converged:
        raise ValueError("a converged OneAssetHANKSteadyState is required")
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
        raise ValueError("tolerance must lie in (0,1e-6]")
    if max_evaluations is not None and (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < 1):
        raise ValueError("max_evaluations must be a positive integer")
    nodes, parents, children, probabilities, shock = _tree(tree)
    ss = steady_state
    p, a, e, transition = ss.calibration, ss.asset_grid, ss.e_grid, ss.transition
    N = len(nodes)
    calls = 0

    def evaluate(x, retain=False):
        nonlocal calls
        calls += 1
        logits = x.reshape(3, N)
        pi, wage, output = np.expm1(logits[0]), p["w"] * np.exp(logits[1]), np.exp(logits[2])
        nominal = p["r"] + shock + p["phi"] * pi
        lag_nominal = np.r_[p["r"], nominal[parents[1:]]]
        real = (1 + lag_nominal) / (1 + pi) - 1
        if not np.isfinite(np.concatenate([pi, wage, output, nominal, real])).all() or np.any(nominal <= -1):
            raise ValueError("scenario tree left the positive-price/gross-return domain")
        adjustment = p["mu"] / (p["mu"]-1) / (2*p["kappa"]) * np.log1p(pi)**2 * output
        dividends = output - wage * output - adjustment
        taxes = p["B"] * real
        nkpc = p["kappa"] * (wage-p["w"]) - np.log1p(pi)
        for i,ch in enumerate(children):
            if ch:
                nkpc[i] += float(probabilities[ch] @ (output[ch]/output[i]*np.log1p(pi[ch])/(1+real[ch])))
        marginal, saving, consumption, hours = [np.empty((N,)+ss.policy_a.shape) for _ in range(4)]
        for i in range(N-1,-1,-1):
            ch = children[i]
            expected = (np.einsum("i,ijk->jk", probabilities[ch], marginal[ch]) if ch else ss.marginal_value)
            marginal[i], saving[i], consumption[i], hours[i] = _backward(
                expected, a, e, transition, p, real[i], wage[i], dividends[i]-taxes[i])
        distributions = np.empty_like(saving)
        distributions[0] = ss.distribution
        assets, cc, labor = np.empty(N), np.empty(N), np.empty(N)
        mass_error, min_mass, top_mass, budget_error, exit_mass = 0., 0., 0., 0., 0.
        for i,ch in enumerate(children):
            d = distributions[i]
            assets[i] = np.sum(d*saving[i])
            cc[i] = np.sum(d*consumption[i])
            labor[i] = np.sum(d*hours[i]*e)
            post = _forward(d,saving[i],a,transition)
            for j in ch:
                distributions[j] = post
            for mass in (d,post):
                mass_error = max(mass_error,abs(float(mass.sum())-1))
                min_mass = min(min_mass,float(mass.min()))
                top_mass = max(top_mass,float(mass[-1].sum()))
            exit_mass = max(exit_mass,float(np.sum(d[(saving[i] < a[0]-1e-12) | (saving[i] > a[-1])])) )
            budget = consumption[i]+saving[i]-(1+real[i])*a[:,None]-wage[i]*e*hours[i]-(dividends[i]-taxes[i])*e
            budget_error = max(budget_error,float(np.max(np.abs(budget))))
        residual = np.r_[nkpc,assets-p["B"],labor-output]
        if not retain:
            return residual
        euler_max, euler_weighted = 0., 0.
        for i,ch in enumerate(children):
            expected = np.einsum("i,ijk->jk",probabilities[ch],marginal[ch]) if ch else ss.marginal_value
            rhs_grid = p["beta"]*(expected@transition.T)
            rhs = np.empty_like(rhs_grid)
            for z in range(len(e)):
                rhs[:,z] = np.interp(saving[i,:,z],a,rhs_grid[:,z])
            lhs = consumption[i]**(-1/p["eis"])
            violation = np.where(saving[i] <= a[0]+1e-10,np.maximum(rhs-lhs,0),np.abs(lhs-rhs))/lhs
            euler_max = max(euler_max,float(np.max(violation)))
            euler_weighted = max(euler_weighted,float(np.sum(distributions[i]*violation)))
        paths = pd.DataFrame({"pi":pi,"w":wage-p["w"],"Y":output-1,"C":cc-ss.aggregates["C"],
            "A":assets-ss.aggregates["A"],"NE":labor-ss.aggregates["NE"],"r":real-p["r"],
            "Div":dividends-ss.aggregates["Div"],"Tax":taxes-ss.aggregates["Tax"],
            "goods_mkt":output-cc-adjustment,"asset_mkt":assets-p["B"],"labor_mkt":labor-output,
            "nkpc_res":nkpc},index=nodes.index.copy())
        diagnostics = {"mass_error":mass_error,"minimum_mass":min_mass,"upper_grid_mass":top_mass,
            "household_budget_error":budget_error,"policy_grid_exit_mass":exit_mass,
            "euler_kkt_max_relative":euler_max,"euler_kkt_max_node_mean":euler_weighted,
            "euler_scope":"linear interpolation at continuous policies; grid exits reported separately",
            "minimum_consumption":float(consumption.min()),"minimum_saving":float(saving.min())}
        return paths,saving,consumption,hours,distributions,diagnostics

    fit = root(evaluate,np.zeros(3*N),method="hybr",
               options={"xtol":1e-10,"maxfev":0 if max_evaluations is None else int(max_evaluations)})
    paths,saving,consumption,hours,distributions,diagnostics = evaluate(fit.x,retain=True)
    residual = float(np.max(np.abs(paths[["asset_mkt","labor_mkt","goods_mkt","nkpc_res"]].to_numpy())))
    converged = bool(fit.success and residual<=tolerance and diagnostics["mass_error"]<1e-10
        and diagnostics["minimum_mass"]>=-1e-12 and diagnostics["household_budget_error"]<1e-9
        and diagnostics["policy_grid_exit_mass"]<1e-10)
    diagnostics.update(market_residual_max_abs=residual,optimizer_success=bool(fit.success),
        optimizer_message=str(fit.message),optimizer_method="Powell hybrid",
        evaluations=calls,tolerance=tolerance)
    return HANKScenarioTreeResult(nodes,paths,saving,consumption,hours,distributions,ss,converged,diagnostics,{
        "scope":"finite-tree aggregate-risk HANK under stated household, firm-valuation and terminal closures",
        "firm_valuation_kernel":"M_child=1/(1+r_child); risk-neutral nominal valuation",
        "terminal_condition":"steady household marginal value and zero terminal inflation; no solved stochastic tail",
        "information":"households condition on the current history and average successor marginal values before choosing",
        "income_transition":"idiosyncratic chain independent of aggregate branch",
        "fiscal_rule":"constant real government debt; taxes equal realized net interest times debt",
        "validation_scope":"deterministic reduction and node accounting; stochastic grid/horizon accuracy must be assessed separately",
        "household_distribution_solved":True,"general_stationary_aggregate_risk_solver":False})
