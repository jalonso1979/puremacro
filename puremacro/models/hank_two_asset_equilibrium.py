"""Two-asset HANK equilibrium with nominal bonds and an illiquid real consol.

The consol has fixed unit supply and a constant real coupon. Its price clears
the illiquid asset market; capital gains affect household returns, while taxes
finance coupons and liquid interest. Common hours are set by a flexible union.
This aggregate closure is distinct from a model with productive capital.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .hank_sequence_space import (_solve_two_asset_household_block, _two_asset_step,
    _two_asset_forward, _two_asset_expectation, _two_asset_cost, _lottery_2d)

__all__ = ["TwoAssetHANKSteadyState", "calibrate_two_asset_equilibrium"]


@dataclass
class TwoAssetHANKSteadyState:
    """Solved household distribution and a fully accounted aggregate steady state.

    Bond supply and consol unit supply are the household demands at the supplied
    steady returns. Labor disutility is calibrated to common hours equal to
    output. These are calibration targets, not externally fixed stock targets.
    """
    calibration: dict[str, float]
    distribution: np.ndarray
    aggregates: dict[str, float]
    diagnostics: dict[str, Any]
    converged: bool
    _household: Any = field(repr=False)


def calibrate_two_asset_equilibrium(*, n_a: int = 15, n_b: int = 15, a_max: float = 80., b_max: float = 10.,
                                    beta: float = .98, gamma: float = 1., r_b: float = .005, r_a: float = .0125,
                                    income: float = 1., chi_0: float = 1., chi_1: float = 1., a_bar: float = .25,
                                    mu: float = 1.2, kappa: float = .1, phi: float = 1.5, frisch: float = .5,
                                    household_tolerance: float = 1e-10) -> TwoAssetHANKSteadyState:
    """Calibrate a liquid-bond/illiquid-consol economy from the two-asset EGM.

    Idiosyncratic productivity uses the existing two-state household process.
    Portfolio adjustment costs consume goods. Nonconvergence, consumption
    floors, or an inconsistent aggregate budget raise instead of supplying a
    false equilibrium. Upper-grid mass and Euler errors require refinement.
    """
    from .hank_recursive import _integer
    _integer(n_a, "n_a", 3)
    _integer(n_b, "n_b", 3)
    values = [a_max, b_max, beta, gamma, r_b, r_a, income, chi_0, chi_1, a_bar,
              mu, kappa, phi, frisch, household_tolerance]
    if not np.isfinite(values).all() or not (0 < beta < 1 and mu > 1 and phi > 1
            and kappa > 0 and frisch > 0 and r_a > r_b > -1 and r_a > 0
            and chi_1 >= 1 and household_tolerance > 0):
        raise ValueError("invalid two-asset equilibrium parameters")
    hh = _solve_two_asset_household_block(n_a=n_a, n_b=n_b, a_max=a_max, b_max=b_max,
        beta=beta, gamma=gamma, r_b_ss=r_b, r_a_ss=r_a, w_ss=income,
        chi_0=chi_0, chi_1=chi_1, a_bar=a_bar, tol=household_tolerance)
    if not hh.converged or abs(hh.budget_residual) > 1e-8 or min(hh.A_ss, hh.B_ss) <= 0:
        raise ValueError("two-asset stationary household or aggregate budget did not converge with positive asset supplies")
    if abs(hh.N_ss-1) > 1e-10:
        raise ValueError("two-asset productivity must have stationary mean one")
    Y, w = hh.Y_ss, 1/mu
    uc_e = float(np.sum(hh.D_ss*hh.s_grid*hh.c_ss**(-gamma)))
    p = dict(beta=beta, gamma=gamma, r=r_b, r_a=r_a, coupon=r_a, B=hh.B_ss,
             consol_units=hh.A_ss, q=1., Y=Y, w=w, mu=mu, kappa=kappa, phi=phi,
             frisch=frisch, vphi=w*uc_e/Y**(1/frisch), income=income,
             chi_0=chi_0, chi_1=chi_1, a_bar=a_bar)
    tax, div = r_b*hh.B_ss+r_a*hh.A_ss, (1-w)*Y
    diagnostics = dict(household_policy_error=hh.residual,
        distribution_error=float(np.max(abs(hh.Lambda@hh.D_ss.ravel()-hh.D_ss.ravel()))),
        budget_error=abs(hh.budget_residual), upper_illiquid_mass=hh.mass_at_a_max,
        upper_liquid_mass=hh.mass_at_b_max, income_process="fixed two-state chain",
        scope="liquid nominal government bonds and fixed-supply illiquid real consols")
    return TwoAssetHANKSteadyState(p, hh.D_ss.copy(),
        dict(Y=Y, C=hh.C_ss, CHI=hh.CHI_ss, A=hh.A_ss, B=hh.B_ss, Tax=tax, Div=div,
             w=w, q=1., N=Y), diagnostics, True, hh)


class _TwoAssetKernel:
    aggregate_names = ("pi", "w", "Y", "q")
    model = "two_asset_consol"

    def __init__(self, ss):
        if not isinstance(ss, TwoAssetHANKSteadyState) or not ss.converged:
            raise ValueError("a converged one- or two-asset HANK steady state is required")
        self.ss = deepcopy(ss)
        self.hh = self.ss._household
        self.d, self.p = self.ss.distribution, self.ss.calibration
        self.shape = self.d.shape
        self.m = np.r_[self.hh.Va_ss.ravel(), self.hh.Vb_ss.ravel()]
        self.Y, self.w, self.r, self.phi = [self.p[k] for k in ("Y", "w", "r", "phi")]
        h = self.hh
        self.constraints = np.column_stack([
            *[np.broadcast_to(np.eye(len(h.s_grid))[j], self.shape).ravel() for j in range(len(h.s_grid))],
            np.broadcast_to(h.b_grid[None, :, None], self.shape).ravel()])
        self._derivatives = None
        self.fd_step = 1e-5

    def prices(self, u, lag_nominal, shock, distribution=None, lag_q=None):
        p = self.p
        pi, w, Y, q = np.expm1(u[0]), self.w*np.exp(u[1]), self.Y*np.exp(u[2]), np.exp(u[3])
        nominal = self.r+shock+self.phi*pi
        rb = (1+self.r+lag_nominal)/(1+pi)-1
        if lag_q is None:
            d = self.d if distribution is None else distribution
            lag_q = np.sum(d*self.hh.a_grid[:, None, None])/p["consol_units"]
        ra = (q+p["coupon"])/lag_q-1
        adj = p["mu"]/(p["mu"]-1)/(2*p["kappa"])*u[0]**2*Y
        tax = rb*p["B"]+p["coupon"]*p["consol_units"]
        income = Y-adj-tax
        if not np.isfinite([pi, w, Y, q, rb, ra, income]).all() or min(1+nominal, 1+rb, 1+ra, w, Y, q, lag_q, income) <= 0:
            raise ValueError("two-asset equilibrium left the positive price/income domain")
        return dict(pi=pi, w=w, Y=Y, q=q, nominal=nominal, r=rb, r_a=ra, adj=adj,
                    tax=tax, div=(1-w)*Y-adj, income=income)

    def backward(self, expected, prices):
        h, nm = self.hh, self.d.size
        result = _two_asset_step(expected[:nm].reshape(self.shape), expected[nm:].reshape(self.shape),
            h.a_grid, h.b_grid, h.s_grid, h.pi_s, h.beta, h.gamma, prices["r"], prices["r_a"],
            prices["income"], h.chi_0, h.chi_1, h.a_bar)
        Va, Vb, c, a, b, deposit, cost = result
        return dict(m=np.r_[Va.ravel(), Vb.ravel()], c=c, a=a, b=b, deposit=deposit, chi=cost)

    def forward(self, d, h):
        hh = self.hh
        return _two_asset_forward(d, h["a"], h["b"], hh.a_grid, hh.b_grid, hh.pi_s)

    def fixed_forward(self, delta):
        return self.hh.Lambda@delta

    def _pack(self, h):
        return np.r_[h["m"], h["c"].ravel(), h["a"].ravel(), h["b"].ravel(), h["chi"].ravel()]

    def _response(self, dm, dp, epsilon):
        v = dict(r=self.r, r_a=self.p["r_a"], income=self.p["income"])
        out = []
        for sign in (1, -1):
            prices = {key: v[key]+sign*epsilon*value for key, value in zip(v, dp)}
            out.append(self._pack(self.backward(self.m+sign*epsilon*dm, prices)))
        return (out[0]-out[1])/(2*epsilon)

    def _forward_dot(self, dd, da, db):
        h = self.hh
        al, ah, wal, wah, bl, bh, wbl, wbh = _lottery_2d(h.a_ss, h.b_ss, h.a_grid, h.b_grid)
        ad = np.divide(da, h.a_grid[ah]-h.a_grid[al], out=np.zeros_like(da), where=ah!=al)
        bd = np.divide(db, h.b_grid[bh]-h.b_grid[bl], out=np.zeros_like(db), where=bh!=bl)
        na, nb, ne = self.shape
        out = np.zeros_like(self.d)
        for ia, wa, dwa in ((al, wal, -ad), (ah, wah, ad)):
            for ib, wb, dwb in ((bl, wbl, -bd), (bh, wbh, bd)):
                mass = dd*wa*wb+self.d*(dwa*wb+wa*dwb)
                for e in range(ne):
                    out[:, :, e] += np.bincount((ia[:, :, e]*nb+ib[:, :, e]).ravel(),
                        weights=mass[:, :, e].ravel(), minlength=na*nb).reshape(na, nb)
        return out@h.pi_s

    def seed_directions(self):
        n = self.d.size
        seeds = []
        for dp in np.eye(3):
            out = self._response(np.zeros_like(self.m), dp, self.fd_step)
            seeds.append(self._forward_dot(np.zeros_like(self.d), out[3*n:4*n].reshape(self.shape),
                                          out[4*n:5*n].reshape(self.shape)).ravel())
        return np.column_stack(seeds)

    def moment_directions(self):
        h = self.hh
        return np.column_stack([h.a_ss.ravel(), h.b_ss.ravel(), h.c_ss.ravel(), h.chi_ss.ravel(),
            (h.s_grid*h.c_ss**(-h.gamma)).ravel(),
            np.broadcast_to(h.a_grid[:, None, None], self.shape).ravel()])

    def _tangent(self, dm, dp):
        if self._derivatives is None:
            nm = len(self.m)
            eye = np.eye(nm+3)
            self._derivatives = np.column_stack([self._response(v[:nm], v[nm:],
                self.fd_step*max(1., abs(self.m[j])) if j < nm else self.fd_step)
                for j, v in enumerate(eye)])
        return self._derivatives@np.r_[dm, dp]

    def linear_residual(self, cur, nxt, basis, rho):
        k, n, p, hh = basis.rank, self.d.size, self.p, self.hh
        ns = k+2
        dd = (basis.vectors@cur[:k]).reshape(self.shape)
        ilag, shock = cur[k:ns]
        pi, w, Y, q = cur[-4:]
        drb = ilag-(1+self.r)*pi
        lagq = np.sum(dd*hh.a_grid[:, None, None])/p["consol_units"]
        dra = q-(1+p["r_a"])*lagq
        dz = self.Y*Y-p["B"]*drb
        response = self._tangent(nxt[ns:ns+2*n], np.array([drb, dra, dz]))
        dm = response[:2*n]
        dc, da, db, dchi = [v.reshape(self.shape) for v in np.split(response[2*n:], 4)]
        dist = self._forward_dot(dd, da, db)
        uc = hh.c_ss**(-hh.gamma)
        duce = np.sum(hh.s_grid*(dd*uc-self.d*hh.gamma*uc/hh.c_ss*dc))
        labor = self.w*(w*np.sum(self.d*hh.s_grid*uc)+duce)-p["vphi"]*self.Y**(1/p["frisch"])*Y/p["frisch"]
        return np.r_[nxt[:k]-basis.vectors.T@dist.ravel(), nxt[k]-shock-self.phi*pi,
            nxt[k+1]-rho*shock, cur[ns:ns+2*n]-dm,
            p["kappa"]*self.w*w-pi+nxt[-4]/(1+self.r),
            np.sum(self.d*db+dd*hh.b_ss),
            labor, np.sum(self.d*da+dd*hh.a_ss)-p["consol_units"]*q]

    def markets(self, d, h, v, continuation):
        p, hh = self.p, self.hh
        A, B, C, chi = [float(np.sum(d*h[key])) for key in ("a", "b", "c", "chi")]
        labor = v["w"]*np.sum(d*hh.s_grid*h["c"]**(-hh.gamma))-p["vphi"]*v["Y"]**(1/p["frisch"])
        nkpc = p["kappa"]*(v["w"]-self.w)-np.log1p(v["pi"])+continuation
        return np.array([nkpc, B-p["B"], labor, A-v["q"]*p["consol_units"]]), dict(
            pi=v["pi"], w=v["w"]-self.w, Y=v["Y"]-self.Y, q=v["q"]-1,
            C=C-self.ss.aggregates["C"], A=A-p["consol_units"], B=B-p["B"],
            r=v["r"]-self.r, r_a=v["r_a"]-p["r_a"], CHI=chi-self.ss.aggregates["CHI"],
            Tax=v["tax"]-self.ss.aggregates["Tax"], Div=v["div"]-self.ss.aggregates["Div"],
            nkpc_res=nkpc, liquid_mkt=B-p["B"], illiquid_mkt=A-v["q"]*p["consol_units"],
            labor_mkt=labor, goods_mkt=v["Y"]-C-chi-v["adj"])

    def household_checks(self, d, h, v, expected):
        hh, n = self.hh, self.d.size
        a, b = hh.a_grid[:, None, None], hh.b_grid[None, :, None]
        budget = h["c"]+h["a"]+h["b"]+h["chi"]-(1+v["r"])*b-(1+v["r_a"])*a-v["income"]*hh.s_grid
        uc = h["c"]**(-hh.gamma)
        cd = _two_asset_cost(h["a"], a, v["r_a"], hh.chi_0, hh.chi_1, hh.a_bar)[1]
        errors = []
        for marginal, lhs, policy, grid in ((expected[:n], uc*(1+cd), h["a"], hh.a_grid),
                                            (expected[n:], uc, h["b"], hh.b_grid)):
            rhs = hh.beta*_two_asset_expectation(marginal.reshape(self.shape), h["a"], h["b"],
                                                 hh.a_grid, hh.b_grid, hh.pi_s)
            gap = np.where(policy <= grid[0]+1e-10, np.maximum(rhs-lhs, 0),
                           np.where(policy >= grid[-1]-1e-10, np.maximum(lhs-rhs, 0), abs(lhs-rhs)))
            errors.append(gap/uc)
        error = np.maximum(*errors)
        return dict(budget_error=float(abs(budget).max()), euler_kkt_max=float(error.max()),
            euler_kkt_mean=float(np.sum(d*error)), minimum_consumption=float(h["c"].min()),
            grid_exit_mass=float(np.sum(d[(h["a"]<hh.a_grid[0]) | (h["a"]>hh.a_grid[-1]) |
                                         (h["b"]<hh.b_grid[0]) | (h["b"]>hh.b_grid[-1])])),
            upper_grid_mass=float(max(d[-1].sum(), d[:, -1].sum())))
