"""Full Bellman and flow checks independent of the backward solver's kernel."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from puremacro.sovereign.adaptation import SovereignAdaptationResult, AdaptationDistribution


class _Checks:
    def __init__(self) -> None:
        self.errors: dict[str,float] = {}

    def add(self, name: str, actual: Any, reference: Any) -> None:
        a, b = np.asarray(actual), np.asarray(reference)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {name}")
        paired_infeasible = np.isneginf(a) & np.isneginf(b)
        good = np.isfinite(a) & np.isfinite(b)
        if not np.all(paired_infeasible | good):
            raise ValueError(f"nonfinite mismatch: {name}")
        error = float(np.max(np.abs(a[good]-b[good])/(1+np.abs(b[good])), initial=0.))
        self.errors[name] = max(self.errors.get(name,0.),error)

    def finish(self, tolerance: float, **extra: Any) -> dict[str,Any]:
        maximum = max(self.errors.values(),default=0.)
        return dict(accepted=maximum <= tolerance,max_error=maximum,checks=self.errors,tolerance=tolerance,**extra)


def _log(c: np.ndarray) -> np.ndarray:
    answer = np.full(c.shape,-np.inf)
    valid = c > 0
    answer[valid] = np.log(c[valid])
    return answer


def audit_result(r: SovereignAdaptationResult, tolerance: float) -> dict[str,Any]:
    from puremacro.sovereign.adaptation import SovereignAdaptationParameters, _inputs
    checks = _Checks()
    regret = 0.
    try:
        p = SovereignAdaptationParameters(**asdict(r.parameters))
        y, theta, h, P, b, a = _inputs(r.income,r.damage,r.disaster,r.transition,r.debt_grid,
                                      r.adaptation_grid,r.horizon,r.decision_tolerance)
        T, nb, na, nz = r.horizon,len(b),len(a),len(y)
        shape = (T,nb,na,nz)
        for name in ("access_value","repayment_value","default","repayment_debt_policy",
                     "repayment_adaptation_policy","prices"):
            if getattr(r,name).shape != shape: raise ValueError(f"invalid shape: {name}")
        for name in ("excluded_value","excluded_adaptation_policy"):
            if getattr(r,name).shape != (T,na,nz): raise ValueError(f"invalid shape: {name}")
        if r.default.dtype.kind != "b": raise ValueError("default decisions must be boolean")
        for name in ("repayment_debt_policy","repayment_adaptation_policy","excluded_adaptation_policy"):
            if getattr(r,name).dtype.kind not in "iu": raise ValueError("policy indices must be integers")
        for name in ("access_value","excluded_value","prices"):
            if not np.isfinite(getattr(r,name)).all(): raise ValueError(f"nonfinite {name}")
        if np.isnan(r.repayment_value).any() or np.isposinf(r.repayment_value).any():
            raise ValueError("invalid repayment values")
        checks.add("transition_normalization",r.transition,P)
        payoff = 1-p.disaster_relief*h
        for t in range(T-1,-1,-1):
            if t == T-1:
                checks.add("terminal_price_placeholders",r.prices[t],np.zeros((nb,na,nz)))
                checks.add("terminal_stock_policy",r.repayment_adaptation_policy[t],np.full((nb,na,nz),-1))
                checks.add("terminal_excluded_policy",r.excluded_adaptation_policy[t],np.full((na,nz),-1))
            else:
                for z in range(nz):
                    expected_price = np.sum(P[z][None,None,:]*payoff[None,None,:]*(1-r.default[t+1]),axis=2)/p.gross_rate
                    checks.add("lender_price",r.prices[t,:,:,z],expected_price)
            for ai, stock in enumerate(a):
                for z in range(nz):
                    damage = float(y[z]*theta[z]*np.exp(-p.effectiveness*stock))
                    net = float(y[z]-damage)
                    if t == T-1:
                        excluded_best = float(np.log((1-p.default_loss)*net))
                        rep_candidates = _log(net-payoff[z]*b)
                        rep_best = rep_candidates
                        selected_rep = rep_candidates
                        checks.add("terminal_debt_policy",r.repayment_debt_policy[t,:,ai,z],
                                   np.where(np.isfinite(rep_best),0,-1))
                    else:
                        investment = np.array([max(float(target-(1-p.depreciation)*stock),0.) for target in a])
                        transfers = p.grant_rate*investment
                        excluded_c = (1-p.default_loss)*net-investment+transfers
                        excluded_objective = _log(excluded_c)+p.beta*np.sum(r.excluded_value[t+1]*P[z],axis=1)
                        excluded_best = float(np.max(excluded_objective))
                        ei = int(r.excluded_adaptation_policy[t,ai,z])
                        if not 0 <= ei < na: raise ValueError("invalid exclusion stock policy")
                        checks.add("exclusion_action_regret",max(0.,excluded_best-excluded_objective[ei]),0.)
                        checks.add("exclusion_selected_value",r.excluded_value[t,ai,z],excluded_objective[ei])
                        # Construct net taxes first, separately from solver's cash-on-hand kernel.
                        tax = (payoff[z]*b[:,None,None]+investment[None,None,:]-transfers[None,None,:]
                               -r.prices[t,:,:,z][None,:,:]*b[None,:,None])
                        continuation = np.sum(r.access_value[t+1]*P[z],axis=2)
                        objective = _log(net-tax)+p.beta*continuation[None,:,:]
                        rep_best = np.max(objective,axis=(1,2))
                        selected_rep = np.full(nb,-np.inf)
                        for bi in range(nb):
                            bp, ap = int(r.repayment_debt_policy[t,bi,ai,z]), int(r.repayment_adaptation_policy[t,bi,ai,z])
                            if np.isfinite(rep_best[bi]):
                                if not (0 <= bp < nb and 0 <= ap < na): raise ValueError("invalid repayment policy")
                                selected_rep[bi] = objective[bi,bp,ap]
                            elif bp != -1 or ap != -1:
                                raise ValueError("infeasible repayment must have missing policies")
                        valid = np.isfinite(rep_best)
                        checks.add("repayment_action_regret",rep_best[valid]-selected_rep[valid],np.zeros(np.sum(valid)))
                    checks.add("excluded_bellman",r.excluded_value[t,ai,z],excluded_best)
                    checks.add("repayment_bellman",r.repayment_value[t,:,ai,z],rep_best)
                    checks.add("repayment_selected_value",r.repayment_value[t,:,ai,z],selected_rep)
                    decisions = excluded_best > rep_best+r.decision_tolerance
                    checks.add("default_decision",r.default[t,:,ai,z].astype(int),decisions.astype(int))
                    selected = np.where(decisions,excluded_best,rep_best)
                    checks.add("access_bellman",r.access_value[t,:,ai,z],selected)
                    regret = max(regret,float(np.max(np.maximum(rep_best,excluded_best)-selected)))
            for excluded in (False,True):
                f = r.to_frame(t,excluded=excluded)
                checks.add("stock_account",f.next_stock.to_numpy(),
                           ((1-p.depreciation)*f.stock+f.investment-f.scrapping).to_numpy())
                checks.add("grant_account",f.grant.to_numpy(),(p.grant_rate*f.investment).to_numpy())
                checks.add("debt_face_account",f.debt.to_numpy(),(f.debt_service+f.contract_relief+f.default_writeoff).to_numpy())
                checks.add("government_household_budget",(f.consumption+f.net_tax).to_numpy(),
                           (f.income-f.disaster_damage-f.exclusion_loss).to_numpy())
                checks.add("external_resource_account",(f.consumption+f.investment+f.debt_service).to_numpy(),
                           (f.income-f.disaster_damage-f.exclusion_loss+f.grant+f.borrowing_proceeds).to_numpy())
                if not (f.consumption > 0).all(): raise ValueError("infeasible realized consumption")
                future = np.zeros(len(f))
                if t < T-1:
                    bn,an = np.searchsorted(b,f.new_debt),np.searchsorted(a,f.next_stock)
                    zz = f.state.to_numpy(dtype=int)
                    checks.add("realized_debt_grid",f.new_debt.to_numpy(),b[bn])
                    checks.add("realized_stock_grid",f.next_stock.to_numpy(),a[an])
                    access_continuation = np.sum(r.access_value[t+1][bn,an]*P[zz],axis=1)
                    exclusion_continuation = np.sum(r.excluded_value[t+1][an]*P[zz],axis=1)
                    future = np.where(f.in_exclusion,exclusion_continuation,access_continuation)
                checks.add("realized_policy_value",f.value.to_numpy(),np.log(f.consumption.to_numpy())+p.beta*future)
        checks.add("default_regret",max(0.,regret-r.decision_tolerance),0.)
    except (ValueError,TypeError,IndexError,OverflowError,ZeroDivisionError,AttributeError,FloatingPointError) as exc:
        return dict(accepted=False,max_error=None,reason=str(exc))
    return checks.finish(tolerance,maximum_default_regret=regret,states_checked=int(np.prod(shape)))


def audit_distribution(d: AdaptationDistribution, tolerance: float) -> dict[str,Any]:
    from puremacro.sovereign.adaptation import _initial
    checks = _Checks()
    try:
        r = d.result
        weights = _initial(r,d.initial_probabilities,d.initial_debt_index,d.initial_adaptation_index,d.initial_access)
        if d.access_mass.shape != r.access_value.shape or d.excluded_mass.shape != r.excluded_value.shape:
            raise ValueError("invalid mass shapes")
        if (not np.isfinite(d.access_mass).all() or not np.isfinite(d.excluded_mass).all()
                or np.any(d.access_mass < 0) or np.any(d.excluded_mass < 0)):
            raise ValueError("nonfinite or negative probability mass")
        initial_market,initial_excluded = np.zeros_like(d.access_mass[0]),np.zeros_like(d.excluded_mass[0])
        if d.initial_access: initial_market[d.initial_debt_index,d.initial_adaptation_index] = weights
        else: initial_excluded[d.initial_adaptation_index] = weights
        checks.add("initial_access_mass",d.access_mass[0],initial_market)
        checks.add("initial_excluded_mass",d.excluded_mass[0],initial_excluded)
        frame = d.to_frame()
        for t in range(r.horizon):
            checks.add("total_mass",np.sum(d.access_mass[t])+np.sum(d.excluded_mass[t]),1.)
            new_defaults = float(np.sum(d.access_mass[t]*r.default[t]))
            checks.add("new_default_probability",frame.new_default_probability.iloc[t],new_defaults)
            checks.add("excluded_probability",frame.excluded_probability.iloc[t],np.sum(d.excluded_mass[t])+new_defaults)
            if t == r.horizon-1: continue
            market,excluded = np.zeros_like(d.access_mass[0]),np.zeros_like(d.excluded_mass[0])
            for znext in range(len(r.income)):
                for bi,ai,z in np.argwhere(d.access_mass[t] > 0):
                    probability = d.access_mass[t,bi,ai,z]*r.transition[z,znext]
                    if r.default[t,bi,ai,z]:
                        excluded[r.excluded_adaptation_policy[t,ai,z],znext] += probability
                    else:
                        market[r.repayment_debt_policy[t,bi,ai,z],r.repayment_adaptation_policy[t,bi,ai,z],znext] += probability
                for ai,z in np.argwhere(d.excluded_mass[t] > 0):
                    excluded[r.excluded_adaptation_policy[t,ai,z],znext] += d.excluded_mass[t,ai,z]*r.transition[z,znext]
            checks.add("access_transition",d.access_mass[t+1],market)
            checks.add("exclusion_transition",d.excluded_mass[t+1],excluded)
        expected_value = (r.access_value[0,d.initial_debt_index,d.initial_adaptation_index] if d.initial_access
                          else r.excluded_value[0,d.initial_adaptation_index])@weights
        checks.add("bellman_expected_log_value",d.value,expected_value)
    except (ValueError,TypeError,IndexError,OverflowError,ZeroDivisionError,AttributeError) as exc:
        return dict(accepted=False,max_error=None,reason=str(exc))
    return checks.finish(tolerance,dates_checked=r.horizon)
