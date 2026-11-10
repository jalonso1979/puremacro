"""Finite-horizon debt/default and public adaptation with exact grid transitions.

Disaster risk is exogenous; installed adaptation reduces damage severity.
Investment protects from the following date. Foreign grants and state-dependent
one-period debt forgiveness are explicit contracts, with no domestic banking.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from puremacro._numeric_contracts import freeze as _freeze_metadata

from puremacro.reports import _df_to_latex, _df_to_markdown, _df_to_typst
from puremacro._numeric_contracts import real_scalar as _number, readonly_array as _owned, real_vector as _vector

__all__ = ["SovereignAdaptationParameters", "SovereignAdaptationResult",
           "AdaptationDistribution", "solve_sovereign_adaptation",
           "audit_sovereign_adaptation", "audit_adaptation_distribution"]


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return int(value)


@dataclass(frozen=True)
class SovereignAdaptationParameters:
    """One-good units. Relief forgives payment; grants reimburse investment.

    Exclusion is permanent until the finite terminal date. The exclusion loss
    applies to output after disaster damage, including the default-entry date.
    """
    beta: float = .96
    gross_rate: float = 1.02
    depreciation: float = .15
    effectiveness: float = 6.
    default_loss: float = .15
    grant_rate: float = 0.
    disaster_relief: float = 0.

    def __post_init__(self) -> None:
        for name in vars(self):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if (not 0 < self.beta <= 1 or self.gross_rate <= 0 or not 0 <= self.depreciation <= 1
                or self.effectiveness < 0 or not 0 <= self.default_loss < 1
                or not 0 <= self.grant_rate <= 1 or not 0 <= self.disaster_relief <= 1):
            raise ValueError("invalid sovereign-adaptation parameters")


def _inputs(income: Any, damage: Any, disaster: Any, transition: Any, debt_grid: Any,
            adaptation_grid: Any, horizon: int, decision_tolerance: float) -> tuple[np.ndarray, ...]:
    y, theta, b, a = (_vector(x, name) for x, name in zip(
        (income, damage, debt_grid, adaptation_grid), ("income", "damage", "debt_grid", "adaptation_grid")))
    h, raw = np.asarray(disaster), np.asarray(transition)
    if (theta.shape != y.shape or h.shape != y.shape or h.dtype.kind != "b"
            or raw.dtype.kind not in "iuf" or raw.shape != (len(y), len(y))):
        raise ValueError("incompatible joint-state inputs or nonboolean disaster indicators")
    P = np.array(raw, dtype=float, copy=True)
    if (np.any(y <= 0) or np.any(theta < 0) or np.any(theta >= 1) or np.any(theta[~h] != 0)
            or not np.isfinite(P).all() or np.any(P < 0) or np.any(np.abs(P.sum(axis=1)-1) > 1e-12)
            or b[0] != 0 or a[0] != 0 or np.any(np.diff(b) <= 0) or np.any(np.diff(a) <= 0)):
        raise ValueError("invalid income, damage, probabilities or grids (grids must start at exact zero)")
    _integer(horizon, "horizon", 1, 30)
    if not 0 <= decision_tolerance <= 1e-8:
        raise ValueError("decision_tolerance must be in [0,1e-8]")
    if horizon*len(b)*len(a)*len(y) > 500_000 or horizon*len(b)**2*len(a)**2*len(y) > 250_000_000:
        raise ValueError("grid budget exceeded: 500000 state cells or 250000000 state/action evaluations")
    P /= P.sum(axis=1, keepdims=True)
    return y, theta, h.copy(), P, b, a


def _utility(consumption: np.ndarray) -> np.ndarray:
    value = np.full(consumption.shape, -np.inf)
    feasible = consumption > 0
    value[feasible] = np.log(consumption[feasible])
    return value


class _Report:
    def to_frame(self) -> pd.DataFrame:
        raise NotImplementedError

    def to_markdown(self) -> str:
        return _df_to_markdown(self.to_frame())

    def to_latex(self) -> str:
        return _df_to_latex(self.to_frame())

    def to_typst(self) -> str:
        return _df_to_typst(self.to_frame())


@dataclass(frozen=True)
class SovereignAdaptationResult(_Report):
    parameters: SovereignAdaptationParameters
    income: np.ndarray
    damage: np.ndarray
    disaster: np.ndarray
    transition: np.ndarray
    debt_grid: np.ndarray
    adaptation_grid: np.ndarray
    horizon: int
    decision_tolerance: float
    access_value: np.ndarray
    repayment_value: np.ndarray
    excluded_value: np.ndarray
    default: np.ndarray
    repayment_debt_policy: np.ndarray
    repayment_adaptation_policy: np.ndarray
    excluded_adaptation_policy: np.ndarray
    prices: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("income", "damage", "disaster", "transition", "debt_grid", "adaptation_grid", "access_value",
                     "repayment_value", "excluded_value", "default", "repayment_debt_policy",
                     "repayment_adaptation_policy", "excluded_adaptation_policy", "prices"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, "diagnostics", _freeze_metadata(self.diagnostics))

    @property
    def fingerprint(self) -> str:
        data = {name: getattr(self, name).tolist() for name in
                ("income", "damage", "disaster", "transition", "debt_grid", "adaptation_grid")}
        data.update(parameters=asdict(self.parameters), horizon=self.horizon, decision_tolerance=self.decision_tolerance)
        return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def to_frame(self, date: int = 0, *, excluded: bool = False) -> pd.DataFrame:
        """Realized optimal accounts for every state at one date.

        Terminal/excluded price entries are zero placeholders with
        issuance_allowed=False. Undefined forced-repayment utilities are missing.
        """
        t = _integer(date, "date", 0, self.horizon-1)
        if not isinstance(excluded, (bool, np.bool_)):
            raise ValueError("excluded must be boolean")
        p = self.parameters
        bi, ai, z = (coordinates.ravel() for coordinates in np.meshgrid(
            [0] if excluded else np.arange(len(self.debt_grid)), np.arange(len(self.adaptation_grid)),
            np.arange(len(self.income)), indexing="ij"))
        b, stock = self.debt_grid[bi], self.adaptation_grid[ai]
        new_default = np.zeros(len(bi), dtype=bool) if excluded else self.default[t,bi,ai,z]
        loss_state = np.ones(len(bi), dtype=bool) if excluded else new_default
        next_bi = np.zeros(len(bi), dtype=int) if excluded else np.where(new_default, 0, self.repayment_debt_policy[t,bi,ai,z])
        next_ai = self.excluded_adaptation_policy[t,ai,z] if excluded else np.where(
            new_default, self.excluded_adaptation_policy[t,ai,z], self.repayment_adaptation_policy[t,bi,ai,z])
        terminal = t == self.horizon-1
        next_stock = (1-p.depreciation)*stock if terminal else self.adaptation_grid[next_ai]
        investment = np.maximum(next_stock-(1-p.depreciation)*stock, 0.)
        scrapping = np.maximum((1-p.depreciation)*stock-next_stock, 0.)
        new_debt = np.zeros(len(bi)) if terminal else self.debt_grid[next_bi]
        issuance = (~loss_state) & (not terminal)
        prices = np.zeros(len(bi)) if terminal else np.where(issuance, self.prices[t,next_bi,next_ai,z], 0.)
        damage = self.income[z]*self.damage[z]*np.exp(-p.effectiveness*stock)
        net_output = self.income[z]-damage
        exclusion_loss = p.default_loss*net_output*loss_state
        relief = p.disaster_relief*self.disaster[z]*b
        writeoff = (b-relief)*new_default
        service = b-relief-writeoff
        grant = p.grant_rate*investment
        proceeds = prices*new_debt
        tax = service+investment-grant-proceeds
        consumption = net_output-exclusion_loss-tax
        repay_value = np.full(len(bi), np.nan) if excluded else self.repayment_value[t,bi,ai,z].copy()
        feasible = np.isfinite(repay_value)
        repay_value[~feasible] = np.nan
        value = self.excluded_value[t,ai,z] if excluded else self.access_value[t,bi,ai,z]
        return pd.DataFrame(dict(date=t, debt_index=bi, adaptation_index=ai, state=z, income=self.income[z],
            damage_fraction=self.damage[z], debt=b, stock=stock, disaster_damage=damage,
            exclusion_loss=exclusion_loss, new_default=new_default, in_exclusion=loss_state,
            contract_relief=relief, default_writeoff=writeoff, debt_service=service,
            new_debt=new_debt, next_stock=next_stock, investment=investment, scrapping=scrapping,
            grant=grant, bond_price=prices, borrowing_proceeds=proceeds, net_tax=tax, consumption=consumption,
            log_consumption=np.log(consumption), issuance_allowed=issuance, value=value,
            repayment_feasible=feasible, repayment_value=repay_value,
            debt_upper_boundary=issuance & (len(self.debt_grid)>1) & (next_bi==len(self.debt_grid)-1),
            stock_upper_boundary=(not terminal) & (len(self.adaptation_grid)>1) & (next_ai==len(self.adaptation_grid)-1)))

    def pricing_frame(self, date: int = 0) -> pd.DataFrame:
        """Quotes for next-period face/stock choices; unavailable at terminal date."""
        t = _integer(date, "pricing date", 0, self.horizon-2)
        bi, ai, z = (coordinates.ravel() for coordinates in np.meshgrid(np.arange(len(self.debt_grid)),
            np.arange(len(self.adaptation_grid)), np.arange(len(self.income)), indexing="ij"))
        probability = self.default[t+1].astype(float)@self.transition.T
        benchmark = self.transition@(1-self.parameters.disaster_relief*self.disaster)/self.parameters.gross_rate
        return pd.DataFrame(dict(debt_index=bi, adaptation_index=ai, state=z, next_debt=self.debt_grid[bi],
            next_stock=self.adaptation_grid[ai], price=self.prices[t].ravel(),
            no_default_price=benchmark[z], next_default_probability=probability.ravel()))

    def distribution(self, initial_probabilities: Any, *, debt_index: int = 0,
                     adaptation_index: int = 0, market_access: bool = True) -> AdaptationDistribution:
        """Propagate exact finite-state probabilities, with permanent exclusion."""
        weights = _initial(self, initial_probabilities, debt_index, adaptation_index, market_access)
        market = np.zeros_like(self.access_value)
        excluded = np.zeros_like(self.excluded_value)
        if market_access: market[0,debt_index,adaptation_index] = weights
        else: excluded[0,adaptation_index] = weights
        for t in range(self.horizon-1):
            for bi, ai, z in np.argwhere(market[t] > 0):
                flow = market[t,bi,ai,z]*self.transition[z]
                if self.default[t,bi,ai,z]:
                    excluded[t+1,self.excluded_adaptation_policy[t,ai,z]] += flow
                else:
                    market[t+1,self.repayment_debt_policy[t,bi,ai,z],self.repayment_adaptation_policy[t,bi,ai,z]] += flow
            for ai, z in np.argwhere(excluded[t] > 0):
                excluded[t+1,self.excluded_adaptation_policy[t,ai,z]] += excluded[t,ai,z]*self.transition[z]
        result = AdaptationDistribution(self, weights, debt_index, adaptation_index, market_access, market, excluded, {})
        audit = audit_adaptation_distribution(result)
        if not audit["accepted"]:
            raise ValueError(f"adaptation distribution audit failed: {audit}")
        return AdaptationDistribution(self, weights, debt_index, adaptation_index, market_access, market, excluded, audit)


def _initial(result: SovereignAdaptationResult, probabilities: Any, debt_index: int,
             adaptation_index: int, market_access: bool) -> np.ndarray:
    _integer(debt_index, "debt_index", 0, len(result.debt_grid)-1)
    _integer(adaptation_index, "adaptation_index", 0, len(result.adaptation_grid)-1)
    if not isinstance(market_access, (bool, np.bool_)) or (not market_access and debt_index != 0):
        raise ValueError("excluded initial states must have zero debt; market_access must be boolean")
    weights = _vector(probabilities, "initial probabilities")
    if weights.shape != result.income.shape or np.any(weights < 0) or abs(float(weights.sum())-1) > 1e-12:
        raise ValueError("initial probabilities must be nonnegative and sum to one")
    return weights/weights.sum()


@dataclass(frozen=True)
class AdaptationDistribution(_Report):
    result: SovereignAdaptationResult
    initial_probabilities: np.ndarray
    initial_debt_index: int
    initial_adaptation_index: int
    initial_access: bool
    access_mass: np.ndarray
    excluded_mass: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("initial_probabilities", "access_mass", "excluded_mass"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, "diagnostics", _freeze_metadata(self.diagnostics))

    def to_frame(self) -> pd.DataFrame:
        columns = ("income", "stock", "debt", "disaster_damage", "exclusion_loss", "investment", "scrapping",
                   "grant", "debt_service", "contract_relief", "default_writeoff", "new_debt", "next_stock",
                   "borrowing_proceeds", "consumption", "log_consumption", "new_default", "in_exclusion",
                   "debt_upper_boundary", "stock_upper_boundary")
        rows = []
        for t in range(self.result.horizon):
            accounts = np.vstack([self.result.to_frame(t)[list(columns)].to_numpy(dtype=float),
                                  self.result.to_frame(t, excluded=True)[list(columns)].to_numpy(dtype=float)])
            weights = np.r_[self.access_mass[t].ravel(), self.excluded_mass[t].ravel()]
            rows.append(weights@accounts)
        return pd.DataFrame(rows, columns=columns, index=pd.RangeIndex(self.result.horizon, name="date")).rename(
            columns=dict(log_consumption="expected_log_consumption", new_default="new_default_probability",
                in_exclusion="excluded_probability", debt_upper_boundary="debt_upper_boundary_probability",
                stock_upper_boundary="stock_upper_boundary_probability"))

    @property
    def value(self) -> float:
        return float(self.result.parameters.beta**np.arange(self.result.horizon)@self.to_frame().expected_log_consumption)

    def consumption_equivalent(self, reference: AdaptationDistribution) -> float:
        """Uniform scaling of reference consumption; excludes donor welfare."""
        if (not isinstance(reference, AdaptationDistribution) or self.result.horizon != reference.result.horizon
                or self.result.parameters.beta != reference.result.parameters.beta):
            raise ValueError("consumption equivalents require identical beta and horizon")
        normalization = float(np.sum(self.result.parameters.beta**np.arange(self.result.horizon)))
        return float(np.expm1((self.value-reference.value)/normalization))


def solve_sovereign_adaptation(*, income: Any, damage: Any, disaster: Any, transition: Any,
        debt_grid: Any, adaptation_grid: Any, horizon: int,
        parameters: SovereignAdaptationParameters | None = None,
        decision_tolerance: float = 1e-12) -> SovereignAdaptationResult:
    """Backward induction with exhaustive discrete choices and lender pricing.

    Current adaptation protects before depreciation; new investment protects
    next period. Stock disposal is free without salvage. Default implies
    permanent exclusion through T-1. At T-1, borrowing/investment and terminal
    continuation value are zero. There is no infinite-horizon claim.
    """
    p = SovereignAdaptationParameters() if parameters is None else parameters
    if not isinstance(p, SovereignAdaptationParameters):
        raise ValueError("parameters must be SovereignAdaptationParameters")
    tol = _number(decision_tolerance, "decision_tolerance")
    y, theta, h, P, b, a = _inputs(income, damage, disaster, transition, debt_grid, adaptation_grid, horizon, tol)
    shape = (horizon, len(b), len(a), len(y))
    V, W, Q = np.zeros(shape), np.full(shape, -np.inf), np.zeros(shape)
    E = np.zeros((horizon,len(a),len(y)))
    default = np.zeros(shape, dtype=bool)
    bp, ap = np.full(shape, -1, dtype=int), np.full(shape, -1, dtype=int)
    ep = np.full(E.shape, -1, dtype=int)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            output = y[None,:]*(1-theta[None,:]*np.exp(-p.effectiveness*a[:,None]))
            investment = np.maximum(a[None,:]-(1-p.depreciation)*a[:,None], 0.)
            domestic_cost = (1-p.grant_rate)*investment
            multiplier = 1-p.disaster_relief*h
            for t in range(horizon-1,-1,-1):
                if t == horizon-1:
                    E[t] = np.log((1-p.default_loss)*output)
                    W[t] = _utility(output[None,:,:]-b[:,None,None]*multiplier)
                    bp[t] = np.where(np.isfinite(W[t]), 0, -1)
                else:
                    next_excluded, next_access = E[t+1]@P.T, V[t+1]@P.T
                    Q[t] = ((~default[t+1])*multiplier)@P.T/p.gross_rate
                    for ai in range(len(a)):
                        for z in range(len(y)):
                            objectives = _utility((1-p.default_loss)*output[ai,z]-domestic_cost[ai])+p.beta*next_excluded[:,z]
                            ep[t,ai,z] = int(np.argmax(objectives))
                            E[t,ai,z] = objectives[ep[t,ai,z]]
                            c = (output[ai,z]-multiplier[z]*b[:,None,None]+Q[t,:,:,z][None,:,:]*b[None,:,None]
                                 -domestic_cost[ai][None,None,:])
                            objective = _utility(c)+p.beta*next_access[:,:,z][None,:,:]
                            flat = objective.reshape(len(b), -1)
                            choices = np.argmax(flat, axis=1)
                            W[t,:,ai,z] = flat[np.arange(len(b)), choices]
                            feasible = np.isfinite(W[t,:,ai,z])
                            bp[t,:,ai,z] = np.where(feasible, choices//len(a), -1)
                            ap[t,:,ai,z] = np.where(feasible, choices%len(a), -1)
                default[t] = E[t][None,:,:] > W[t]+tol
                V[t] = np.where(default[t], E[t][None,:,:], W[t])
    except FloatingPointError as exc:
        raise ValueError("nonfinite adaptation recursion; no clipped outcome returned") from exc
    result = SovereignAdaptationResult(p,y,theta,h,P,b,a,horizon,tol,V,W,E,default,bp,ap,ep,Q,{})
    audit = audit_sovereign_adaptation(result)
    if not audit["accepted"]:
        raise ValueError(f"sovereign adaptation audit failed: {audit}")
    diagnostics = dict(audit, scope="finite horizon and finite debt/stock grids; exogenous hazard; free disposal",
                       terminal="no borrowing/investment; zero continuation and salvage value",
                       state_cells=int(V.size), state_action_bound=int(horizon*len(b)**2*len(a)**2*len(y)))
    return SovereignAdaptationResult(p,y,theta,h,P,b,a,horizon,tol,V,W,E,default,bp,ap,ep,Q,diagnostics)


def audit_sovereign_adaptation(result: SovereignAdaptationResult, *, tolerance: float = 1e-9) -> dict[str,Any]:
    """Recompute full Bellman optima, pricing, policy feasibility and accounts."""
    from puremacro.sovereign._adaptation_audit import audit_result
    tol = _number(tolerance, "tolerance")
    if not isinstance(result,SovereignAdaptationResult) or not 0 < tol <= 1e-6:
        raise ValueError("require SovereignAdaptationResult and tolerance in (0,1e-6]")
    return audit_result(result,tol)


def audit_adaptation_distribution(distribution: AdaptationDistribution, *, tolerance: float = 1e-9) -> dict[str,Any]:
    """Check forward mass flows, default timing and Bellman/expected-log welfare."""
    from puremacro.sovereign._adaptation_audit import audit_distribution
    tol = _number(tolerance, "tolerance")
    if not isinstance(distribution,AdaptationDistribution) or not 0 < tol <= 1e-6:
        raise ValueError("require AdaptationDistribution and tolerance in (0,1e-6]")
    return audit_distribution(distribution,tol)
