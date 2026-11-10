"""Experimental second-order local policies for small, full-grid one-asset HANK.

State curvature and anticipated-variance corrections solve separate equations.
Household constraints and grid interpolation require local smoothness checks.
No welfare, global accuracy, reduced-basis or two-asset claim is made here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from puremacro._numeric_contracts import freeze, readonly_array, real_scalar, real_vector
from ._hank_one_asset_recursive import _OneAssetKernel
from ._hank_second_order import OneAssetEquations, solve_curvature
from .hank_recursive import HANKStochasticResult
from .hank_reduction import ReducedStochasticHANKPolicy

__all__=['HANKSecondOrderAudit','SecondOrderHANKPolicy','solve_hank_second_order']


def _positive(value, name):
    answer=real_scalar(value,name)
    if answer<=0:
        raise ValueError(f'{name} must be positive')
    return answer


def _integer(value,name,minimum=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _quadrature(order):
    order=_integer(order,'quadrature_order',2)
    if order>12:
        raise ValueError('quadrature_order must not exceed 12')
    draws,weights=np.polynomial.hermite.hermgauss(order)
    return np.sqrt(2)*draws,weights/np.sqrt(np.pi)


@dataclass(frozen=True)
class HANKSecondOrderAudit:
    """Pointwise finite-grid equation and feasibility audit, without clipping."""
    accepted: bool
    residual: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self,'residual',readonly_array(self.residual,dtype=float))
        object.__setattr__(self,'diagnostics',freeze(self.diagnostics))


@dataclass(frozen=True)
class SecondOrderHANKPolicy:
    """State and variance tensors around the deterministic household equilibrium.

    ``feedback_curvature`` indexes full marginal coefficients then aggregate
    log controls (inflation, wage, output). Every tensor uses the one-half
    Taylor convention. ``transition_variance`` and ``feedback_variance`` are
    derivatives with respect to unit innovation variance, not already scaled
    by ``shock_std**2``. The parent first-order policy is privately copied.
    """
    transition_curvature: np.ndarray
    feedback_curvature: np.ndarray
    transition_variance: np.ndarray
    feedback_variance: np.ndarray
    diagnostics: Mapping[str, Any]
    _linear: ReducedStochasticHANKPolicy = field(repr=False)

    def __post_init__(self) -> None:
        parent=deepcopy(self._linear)
        ns=len(parent.transition);nu=len(parent.reduced_feedback)
        shapes=dict(transition_curvature=(ns,ns,ns),feedback_curvature=(nu,ns,ns),
                    transition_variance=(ns,),feedback_variance=(nu,))
        for name,shape in shapes.items():
            raw=np.asarray(getattr(self,name))
            if raw.dtype.kind not in 'iuf' or raw.shape!=shape or not np.isfinite(raw).all():
                raise ValueError(f'invalid {name}')
            object.__setattr__(self,name,readonly_array(raw,dtype=float))
        object.__setattr__(self,'diagnostics',freeze(self.diagnostics))
        object.__setattr__(self,'_linear',parent)

    @property
    def shock_std(self) -> float:
        return self._linear.shock_std

    def state(self, distribution: np.ndarray | None = None, *, lag_nominal: float = 0.,
              monetary_shock: float = 0.) -> np.ndarray:
        """Encode a feasible distribution and the two observed aggregate states."""
        return self._linear.state(distribution,lag_nominal=real_scalar(lag_nominal,'lag_nominal'),
                                  monetary_shock=real_scalar(monetary_shock,'monetary_shock'))

    def _state(self,value):
        x=real_vector(value,'state')
        if x.shape!=(len(self._linear.transition),):
            raise ValueError('state has the wrong dimension')
        return x

    def _sigma(self,value):
        sigma=self.shock_std if value is None else real_scalar(value,'shock_std')
        if sigma<0:
            raise ValueError('shock_std must be nonnegative')
        if sigma>np.sqrt(np.finfo(float).max):
            raise ValueError('shock_std variance must be finite')
        return sigma

    def _controls_pair(self,x1,x2,sigma):
        with np.errstate(over='ignore',invalid='ignore'):
            result=self._linear.reduced_feedback@(x1+x2)+.5*np.einsum(
                'aij,i,j->a',self.feedback_curvature,x1,x1)+.5*self.feedback_variance*sigma**2
        if not np.isfinite(result).all():
            raise ValueError('nonfinite second-order controls')
        return result

    def _next_pair(self,x1,x2,sigma):
        H=self._linear.transition
        with np.errstate(over='ignore',invalid='ignore'):
            first=H@x1
            second=H@x2+.5*np.einsum('aij,i,j->a',self.transition_curvature,x1,x1)+.5*self.transition_variance*sigma**2
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError('nonfinite second-order transition')
        return first,second

    def controls(self, state: np.ndarray, *, shock_std: float | None = None) -> np.ndarray:
        """Evaluate full marginal coefficients and aggregate log controls.

        Polynomial evaluation is not an equation/feasibility certificate; use
        ``audit`` for the specified state, variance and quadrature order.
        """
        x=self._state(state)
        return self._controls_pair(x,np.zeros_like(x),self._sigma(shock_std))

    def next_state(self, state: np.ndarray, *, shock_std: float | None = None) -> np.ndarray:
        """Conditional next state, before the next zero-mean innovation."""
        x=self._state(state)
        a,b=self._next_pair(x,np.zeros_like(x),self._sigma(shock_std))
        return a+b

    def _audit_pair(self,x1,x2,sigma,tolerance,order):
        p=self._linear;equations=OneAssetEquations(p)
        draws,weights=_quadrature(order)
        try:
            u=self._controls_pair(x1,x2,sigma)
            xp1,xp2=self._next_pair(x1,x2,sigma)
            future1=xp1+sigma*draws[:,None]*equations.eta
            future=future1+xp2
            controls=np.array([self._controls_pair(v,xp2,sigma) for v in future1])
            residual,checks=equations.evaluate(x1+x2,u,xp1+xp2,future,controls,weights,inspect=True)
            accepted=(max(checks['equation_error'],checks['budget_error'],checks['goods_market_error'],
                          checks['distribution_error'])<=tolerance and checks['mass_error']<1e-10
                      and checks['minimum_mass']>=-1e-12 and checks['minimum_consumption']>0)
            accepted=accepted and checks['grid_exit_mass']<1e-10
            checks.update(scope='local finite-grid equations and feasibility',quadrature_order=order,
                shock_std=sigma,tolerance=tolerance,euler_kkt_is_grid_diagnostic=True,
                reason='accepted' if accepted else 'equations_or_feasibility_failed')
            return HANKSecondOrderAudit(bool(accepted),residual,checks)
        except (ValueError,FloatingPointError,OverflowError) as exc:
            return HANKSecondOrderAudit(False,np.full(equations.dim,np.inf),dict(
                reason='outside_local_domain',detail=str(exc),equation_error=float('inf'),
                quadrature_order=order,shock_std=sigma,tolerance=tolerance))

    def audit(self, state: np.ndarray, *, shock_std: float | None = None,
              tolerance: float = 1e-5, quadrature_order: int = 3) -> HANKSecondOrderAudit:
        """Audit nonlinear discrete equations and quadrature branches at a point.

        Household Euler/KKT errors are reported separately as discretization
        diagnostics. Acceptance does not certify continuous-grid convergence.
        """
        x=self._state(state);sigma=self._sigma(shock_std)
        tolerance=_positive(tolerance,'tolerance');_quadrature(quadrature_order)
        return self._audit_pair(x,np.zeros_like(x),sigma,tolerance,quadrature_order)

    def mean_decomposition(self, *, shock_std: float | None = None) -> dict[str, np.ndarray]:
        """Mean of the pruned approximation: anticipated risk versus dispersion.

        These are coordinate/log means. Nonlinear level means require their
        own Taylor transformation. No household welfare measure is returned.
        """
        from scipy.linalg import solve_discrete_lyapunov
        p=self._linear;H=p.transition;G=p.reduced_feedback;sigma=self._sigma(shock_std)
        Q=np.zeros_like(H);Q[p.shock_index,p.shock_index]=sigma**2
        covariance=solve_discrete_lyapunov(H,Q)
        risk=np.linalg.solve(np.eye(len(H))-H,.5*self.transition_variance*sigma**2)
        dispersion=np.linalg.solve(np.eye(len(H))-H,.5*np.einsum('aij,ij->a',self.transition_curvature,covariance))
        control_risk=G@risk+.5*self.feedback_variance*sigma**2
        control_dispersion=G@dispersion+.5*np.einsum('aij,ij->a',self.feedback_curvature,covariance)
        return dict(state_covariance=covariance,state_risk=risk,state_dispersion=dispersion,
            state_mean=risk+dispersion,control_risk=control_risk,control_dispersion=control_dispersion,
            control_mean=control_risk+control_dispersion)

    def simulate(self, innovations: np.ndarray, *, initial_state: np.ndarray | None = None,
                 shock_std: float | None = None, tolerance: float = 1e-5,
                 quadrature_order: int = 3) -> HANKStochasticResult:
        """Pruned second-order recursion with a nonlinear audit at every date.

        The supplied innovations are realized level innovations; ``shock_std``
        describes anticipated risk. Quadratic terms use first-order states
        only, preventing unintended fourth-order feedback.
        """
        eps=real_vector(innovations,'innovations');p=self._linear
        sigma=self._sigma(shock_std);tolerance=_positive(tolerance,'tolerance');_quadrature(quadrature_order)
        x1=np.zeros(len(p.transition)) if initial_state is None else self._state(initial_state)
        x2=np.zeros_like(x1);rows=[];states=[];distributions=[];reports=[]
        for innovation in eps:
            x1[p.shock_index]+=innovation
            u=self._controls_pair(x1,x2,sigma)
            reports.append(self._audit_pair(x1,x2,sigma,tolerance,quadrature_order))
            rows.append(u[-3:]);states.append((x1+x2).copy())
            distributions.append(p.basis.decode((x1+x2)[:p.basis.rank],check_positive=False))
            x1,x2=self._next_pair(x1,x2,sigma)
        data=np.asarray(distributions)
        diagnostics=dict(scope='pruned second-order full-grid one-asset policy',approximation_order=2,
            pruned=True,shock_std=sigma,quadrature_order=quadrature_order,
            maximum_equation_error=max(r.diagnostics['equation_error'] for r in reports),
            minimum_reconstructed_mass=float(data.min()),mass_error=float(np.max(abs(data.sum(axis=(1,2))-1))),
            failed_dates=[i for i,r in enumerate(reports) if not r.accepted],
            aggregate_units='log deviations',continuous_grid_certified=False,welfare_supported=False)
        return HANKStochasticResult(pd.DataFrame(rows,columns=p._kernel.aggregate_names),np.asarray(states),
                                    data,diagnostics,all(r.accepted for r in reports))


def solve_hank_second_order(linear_policy: ReducedStochasticHANKPolicy, *,
        difference_step: float = 1e-3, curvature_tolerance: float = 1e-3,
        max_states: int = 48, max_workspace_bytes: int = 256*1024**2) -> SecondOrderHANKPolicy:
    """Solve checked curvature for a small one-asset policy retaining full bases.

    Centered derivatives at h and h/2 must agree before coefficient solution.
    The tolerance controls normalized derivative agreement, not global policy
    accuracy. Nonlinear point/path audits and amplitude checks remain needed.
    Resource limits bound the state count and a conservative working-array
    estimate, not total process memory. No derivative regularization is used.
    """
    step=_positive(difference_step,'difference_step')
    agreement=_positive(curvature_tolerance,'curvature_tolerance')
    max_states=_integer(max_states,'max_states');budget=_integer(max_workspace_bytes,'max_workspace_bytes')
    p=linear_policy
    if not isinstance(p,ReducedStochasticHANKPolicy) or type(p._kernel) is not _OneAssetKernel or not p.converged:
        raise ValueError('a solved one-asset ReducedStochasticHANKPolicy is required')
    if p.basis.rank!=p._kernel.d.size-p._kernel.constraints.shape[1] or p.marginal_basis.rank!=len(p._kernel.m):
        raise ValueError('second order requires full distribution and marginal bases')
    ns=len(p.transition);dim=len(p._F0)
    estimated=8*(24*dim*ns**2+128*dim**2)
    if ns>max_states or estimated>budget:
        raise ValueError('second-order dimensions exceed max_states or max_workspace_bytes')
    equations=OneAssetEquations(p)
    scales=1/np.maximum(1,np.max(abs(p.reduced_feedback)/(1+np.r_[p._kernel.m,np.ones(3)])[:,None],axis=0))
    C0,S0,D0=equations.derivatives(step,scales)
    C1,S1,D1=equations.derivatives(step/2,scales)
    normalization=scales[None,:,None]*scales[None,None,:]/equations.row_scale[:,None,None]
    curvature_error=float(np.max(abs((C1-C0)*normalization))/(1+np.max(abs(C1*normalization))))
    variance_error=float(np.max(abs(S1-S0)/equations.row_scale)/(1+np.max(abs(S1)/equations.row_scale)))
    first_error=float(np.max(abs((4*D1-D0)/3)/equations.row_scale[:,None]))
    if not np.isfinite([curvature_error,variance_error,first_error]).all() or max(curvature_error,variance_error)>agreement or first_error>1e-6:
        raise ValueError(f'local derivative refinement failed: state={curvature_error:.3g}, variance={variance_error:.3g}, first={first_error:.3g}')
    # Richardson removes the leading centered-difference error after agreement.
    C=(4*C1-C0)/3;Cs=(4*S1-S0)/3
    Hxx,Gxx,Hss,Gss,diagnostics=solve_curvature(p._F0,p._F1,p.transition,p.reduced_feedback,C,Cs,
                                             shock_index=p.shock_index)
    diagnostics.update(approximation_order=2,model='one_asset',full_bases=True,difference_step=step,
        curvature_refinement=curvature_error,variance_refinement=variance_error,first_derivative_error=first_error,
        curvature_tolerance=agreement,state_dimension=ns,estimated_workspace_bytes=estimated,
        scope='local finite-grid second-order policy',global_solution=False,welfare_supported=False,
        derivative_method='centered differences with step refinement and Richardson extrapolation')
    return SecondOrderHANKPolicy(Hxx,Gxx,Hss,Gss,diagnostics,p)
