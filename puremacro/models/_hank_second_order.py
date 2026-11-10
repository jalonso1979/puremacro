"""Second-order coefficient equations for a small state/control system."""
from __future__ import annotations

import numpy as np

from puremacro.dsge._sylvester import solve_generalized_sylvester_schur


class OneAssetEquations:
    """Full-grid equations, retaining expectation timing of the household step."""

    def __init__(self, policy):
        self.policy=policy
        self.kernel=policy._kernel
        self.k=policy.basis.rank
        self.ns=len(policy.transition)
        self.nr=policy.marginal_basis.rank
        self.dim=self.ns+self.nr+3
        self.eta=np.eye(self.ns)[policy.shock_index]
        self.row_scale=np.r_[np.ones(self.ns),1+self.kernel.m,np.ones(3)]

    def evaluate(self,x,u,xp,future_states,future_controls,weights,*,inspect=False):
        p,K,k,nr=self.policy,self.kernel,self.k,self.nr
        d=p.basis.decode(x[:k],check_positive=False)
        v=K.prices(u[-3:],*x[k:])
        future_m=K.m+future_controls[:,:nr]@p.marginal_basis.vectors.T
        current_m=K.m+p.marginal_basis.vectors@u[:nr]
        expected=weights@future_m
        if min(float(np.min(future_m)),float(np.min(current_m)),float(np.min(expected)))<=0:
            raise ValueError('second-order household marginal values left the positive domain')
        h=K.backward(expected,v)
        dp=K.forward(d,h)
        continuation=0.
        for state,control,weight in zip(future_states,future_controls,weights):
            vf=K.prices(control[-3:],*state[k:])
            continuation+=weight*vf['Y']/v['Y']*control[-3]/(1+vf['r'])
        market,accounts=K.markets(d,h,v,continuation)
        residual=np.r_[xp[:k]-p.basis.project(dp-K.d),xp[k]-(v['nominal']-K.r),
            xp[k+1]-p.rho*x[k+1],u[:nr]-p.marginal_basis.project(h['m']-K.m),market]
        if not np.isfinite(residual).all():
            raise ValueError('nonfinite second-order equilibrium residual')
        if not inspect:
            return residual
        reconstructed=p.basis.decode(xp[:k],check_positive=False)
        checks=K.household_checks(d,h,v,expected)
        checks.update(minimum_mass=float(min(d.min(),dp.min(),reconstructed.min())),
            mass_error=float(max(abs(d.sum()-1),abs(dp.sum()-1),abs(reconstructed.sum()-1))),
            distribution_error=float(np.max(abs(dp-reconstructed))),
            minimum_marginal=float(min(future_m.min(),current_m.min(),expected.min())),
            goods_market_error=float(abs(accounts['goods_mkt'])),
            equation_error=float(np.max(abs(residual)/self.row_scale)))
        return residual,checks

    def first_order_residual(self,x,sigma=0.):
        p=self.policy
        xp=p.transition@x
        future=np.array([xp-sigma*self.eta,xp+sigma*self.eta])
        return self.evaluate(x,p.reduced_feedback@x,xp,future,future@p.reduced_feedback.T,np.array([.5,.5]))

    def derivatives(self,step,scales):
        zero=np.zeros(self.ns)
        base=self.first_order_residual(zero)
        C=np.zeros((self.dim,self.ns,self.ns))
        odd=np.empty((self.dim,self.ns))
        for i in range(self.ns):
            xi=np.eye(self.ns)[i]*step*scales[i]
            plus,minus=self.first_order_residual(xi),self.first_order_residual(-xi)
            odd[:,i]=(plus-minus)/(2*step)
            C[:,i,i]=(plus+minus-2*base)/(step*scales[i])**2
            for j in range(i):
                xj=np.eye(self.ns)[j]*step*scales[j]
                value=(self.first_order_residual(xi+xj)-self.first_order_residual(xi-xj)
                       -self.first_order_residual(-xi+xj)+self.first_order_residual(-xi-xj))
                C[:,i,j]=C[:,j,i]=value/(4*step**2*scales[i]*scales[j])
        Cs=2*(self.first_order_residual(zero,step)-base)/step**2
        return C,Cs,odd


def solve_curvature(F0, F1, H, G, curvature, variance_curvature, *, shock_index):
    """Solve state and unit-innovation-variance coefficients without a ridge.

    The residual convention is F0 [x,y] + F1 E[x_next,y_next]. Curvature is
    evaluated along the first-order policy, with shape (equations, states,
    states). The variance forcing holds future policies linear; future Gxx
    contributes separately. Coefficients multiply one half of xx or sigma^2.
    """
    values=[]
    for value in (F0,F1,H,G,curvature,variance_curvature):
        raw=np.asarray(value)
        if raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
            raise ValueError('coefficient data must be finite and real')
        values.append(np.asarray(raw,dtype=float))
    F0,F1,H,G,C,Cs=values
    if H.ndim!=2 or H.shape[0]!=H.shape[1] or not len(H):
        raise ValueError('invalid transition shape')
    ns=len(H)
    if G.ndim!=2 or G.shape[1]!=ns or not len(G):
        raise ValueError('invalid feedback shape')
    dim=ns+len(G)
    if F0.shape!=(dim,dim) or F1.shape!=(dim,dim) or C.shape!=(dim,ns,ns) or Cs.shape!=(dim,):
        raise ValueError('invalid coefficient shape')
    if isinstance(shock_index,(bool,np.bool_)) or not isinstance(shock_index,(int,np.integer)) or not 0<=shock_index<ns:
        raise ValueError('shock_index must name a state coordinate')
    if not np.allclose(C,C.swapaxes(1,2),rtol=1e-10,atol=1e-12):
        raise ValueError('state curvature must be symmetric')
    stack=np.vstack([np.eye(ns),G])
    first_error=np.max(abs(F0@stack+F1@stack@H))
    if first_error>1e-8 or np.max(abs(np.linalg.eigvals(H)))>=1-1e-8:
        raise ValueError('a stable consistent first-order policy is required')
    A=np.column_stack([F1@stack,F0[:,ns:]])
    B=np.column_stack([np.zeros((dim,ns)),F1[:,ns:]])
    variance_condition=float(np.linalg.cond(A+B))
    if not np.isfinite(variance_condition) or variance_condition>1e12:
        raise ValueError('variance coefficient equation is singular or ill-conditioned')
    try:
        raw=solve_generalized_sylvester_schur(A,B,H,C.reshape(dim,-1))
        X=np.asarray(raw).reshape(dim,ns,ns)
        X=(X+X.swapaxes(1,2))/2
        forcing=Cs+F1[:,ns:]@X[ns:,shock_index,shock_index]
        Xs=np.linalg.solve(A+B,-forcing)
    except np.linalg.LinAlgError as exc:
        raise ValueError('second-order coefficient equations could not be solved') from exc
    future=np.einsum('aij,ik,jl->akl',X,H,H)
    error=np.einsum('ab,bij->aij',A,X)+np.einsum('ab,bij->aij',B,future)+C
    state_error=float(np.max(abs(error))/(1+np.max(abs(C))))
    variance_error=float(np.max(abs((A+B)@Xs+forcing))/(1+np.max(abs(forcing))))
    if (not np.isfinite(X).all() or not np.isfinite(Xs).all()
            or not np.isfinite([state_error,variance_error]).all()
            or max(state_error,variance_error)>1e-8):
        raise ValueError('second-order coefficient equation residual exceeds tolerance')
    return X[:ns],X[ns:],Xs[:ns],Xs[ns:],dict(
        state_equation_error=state_error,variance_equation_error=variance_error,
        variance_condition=variance_condition,first_order_equation_error=float(first_error))
