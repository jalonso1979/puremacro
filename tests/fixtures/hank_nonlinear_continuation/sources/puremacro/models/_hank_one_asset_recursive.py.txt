"""One-period equations and derivatives for the recursive HANK solver."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .one_asset_hank import OneAssetHANKSteadyState, _backward, _forward
from ._hank_tangent import _household_tape, _household_jvp, _forward_tape, _forward_jvp


class _OneAssetKernel:
    aggregate_names = ("pi", "w", "Y")
    model = "one_asset"

    def __init__(self, ss):
        if not isinstance(ss, OneAssetHANKSteadyState) or not ss.converged:
            raise ValueError("a converged OneAssetHANKSteadyState is required")
        self.ss = deepcopy(ss)
        self.ss._jacobians = {}
        self.d = self.ss.distribution
        self.m = self.ss.marginal_value.ravel()
        self.p = self.ss.calibration
        self.shape = self.d.shape
        self.w = self.p["w"]
        self.Y = 1.
        self.r = self.p["r"]
        self.phi = self.p["phi"]
        a, e = self.ss.asset_grid, self.ss.e_grid
        self.constraints = np.column_stack([
            *[np.broadcast_to(np.eye(len(e))[j], self.shape).ravel() for j in range(len(e))],
            np.broadcast_to(a[:, None], self.shape).ravel()])
        self.hh_tape = _household_tape(self.ss.marginal_value, a, e, self.ss.transition,
            self.p, self.r, self.w, self.ss.aggregates["Div"] - self.ss.aggregates["Tax"])
        self.forward_tape = _forward_tape(self.ss.policy_a, a)

    def prices(self, u, lag_nominal, shock, distribution=None, lag_q=None):
        p = self.p
        pi, w, Y = np.expm1(u[0]), self.w * np.exp(u[1]), np.exp(u[2])
        nominal = self.r + shock + self.phi * pi
        r = (1 + self.r + lag_nominal) / (1 + pi) - 1
        adj = p["mu"] / (p["mu"] - 1) / (2 * p["kappa"]) * u[0]**2 * Y
        div, tax = (1 - w) * Y - adj, p["B"] * r
        if min(1 + nominal, 1 + r, w, Y) <= 0 or not np.isfinite([pi, w, Y, nominal, r]).all():
            raise ValueError("recursive HANK left the positive price/return domain")
        return dict(pi=pi, w=w, Y=Y, nominal=nominal, r=r, adj=adj, div=div, tax=tax,
                    transfer=div-tax, q=1.)

    def backward(self, expected, prices):
        s = self.ss
        m, a, c, n = _backward(expected.reshape(self.shape), s.asset_grid, s.e_grid,
            s.transition, self.p, prices["r"], prices["w"], prices["transfer"])
        return dict(m=m.ravel(), a=a, c=c, n=n)

    def forward(self, d, h):
        return _forward(d, h["a"], self.ss.asset_grid, self.ss.transition)

    def fixed_forward(self, delta):
        return _forward(delta.reshape(self.shape), self.ss.policy_a,
                        self.ss.asset_grid, self.ss.transition).ravel()

    def _tangent(self, dm, dr, dw, dt):
        s = self.ss
        return _household_jvp(self.hh_tape, dm.reshape(self.shape), s.asset_grid, s.e_grid,
            s.transition, self.p, self.r, self.w, s.policy_c, s.policy_n, dr, dw, dt)

    def seed_directions(self):
        s = self.ss
        seeds = []
        for v in np.eye(3):
            _, da, _, _ = self._tangent(np.zeros_like(self.m), *v)
            seeds.append(_forward_jvp(self.d, np.zeros_like(self.d), da,
                                     self.forward_tape, s.transition).ravel())
        return np.column_stack(seeds)

    def moment_directions(self):
        s = self.ss
        return np.column_stack([s.policy_a.ravel(), (s.policy_n*s.e_grid).ravel(), s.policy_c.ravel()])

    def linear_residual(self, cur, nxt, basis, rho):
        B, p, s = basis.vectors, self.p, self.ss
        k, nm = basis.rank, len(self.m)
        ns = k + 2
        dd = (B @ cur[:k]).reshape(self.shape)
        ilag, shock = cur[k:ns]
        pi, w, Y = cur[-3:]
        dr, dw = ilag - (1 + self.r) * pi, self.w * w
        transfer = (1 - self.w) * Y - dw - p["B"] * dr
        dm, da, _, dn = self._tangent(nxt[ns:ns+nm], dr, dw, transfer)
        dist = _forward_jvp(self.d, dd, da, self.forward_tape, s.transition)
        return np.r_[nxt[:k] - B.T @ dist.ravel(), nxt[k] - shock - self.phi*pi,
            nxt[k+1] - rho*shock, cur[ns:ns+nm] - dm.ravel(),
            p["kappa"]*dw - pi + nxt[-3]/(1+self.r),
            np.sum(self.d*da + dd*s.policy_a),
            np.sum((self.d*dn + dd*s.policy_n)*s.e_grid) - Y]

    def markets(self, d, h, prices, continuation):
        p, s, v = self.p, self.ss, prices
        A, C, L = np.sum(d*h["a"]), np.sum(d*h["c"]), np.sum(d*h["n"]*s.e_grid)
        nkpc = p["kappa"]*(v["w"]-self.w) - np.log1p(v["pi"]) + continuation
        return np.array([nkpc, A-p["B"], L-v["Y"]]), dict(
            pi=v["pi"], w=v["w"]-self.w, Y=v["Y"]-1, C=C-s.aggregates["C"],
            A=A-s.aggregates["A"], r=v["r"]-self.r, Div=v["div"]-s.aggregates["Div"],
            Tax=v["tax"]-s.aggregates["Tax"], nkpc_res=nkpc, asset_mkt=A-p["B"],
            labor_mkt=L-v["Y"], goods_mkt=v["Y"]-C-v["adj"])

    def household_checks(self, d, h, v, expected):
        s, p = self.ss, self.p
        a, e = s.asset_grid, s.e_grid
        budget = h["c"] + h["a"] - (1+v["r"])*a[:, None] - v["w"]*e*h["n"] - v["transfer"]*e
        rhs_grid = p["beta"] * (expected.reshape(self.shape) @ s.transition.T)
        rhs = np.column_stack([np.interp(h["a"][:, j], a, rhs_grid[:, j]) for j in range(len(e))])
        lhs = h["c"]**(-1/p["eis"])
        error = np.where(h["a"] <= a[0]+1e-10, np.maximum(rhs-lhs, 0), abs(lhs-rhs))/lhs
        return dict(budget_error=float(np.max(abs(budget))), euler_kkt_max=float(error.max()),
            euler_kkt_mean=float(np.sum(d*error)), minimum_consumption=float(h["c"].min()),
            grid_exit_mass=float(np.sum(d[(h["a"] < a[0]-1e-12) | (h["a"] > a[-1])])),
            upper_grid_mass=float(d[-1].sum()))
