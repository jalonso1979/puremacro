"""Two-asset HANK with productive capital and installation costs.

Households own nominal government bonds and installed capital directly. Firms'
operating profits are rebated with efficiency-weighted income. This is a
separate economy from both the fixed-supply consol model and the ABRS (2021)
equity/intermediary economy; its production and investment blocks match the
ABRS equations, including the installation rent in capital returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from scipy.optimize import brentq

from .hank_sequence_space import _solve_two_asset_household_block
from .hank_two_asset_equilibrium import TwoAssetHANKSteadyState, _TwoAssetKernel

__all__ = ["CapitalHANKSteadyState", "calibrate_capital_hank", "capital_production_block"]


@dataclass
class CapitalHANKSteadyState(TwoAssetHANKSteadyState):
    """Stationary direct-capital economy with fixed physical parameters.

    The discount factor clears the capital market at the specified steady
    returns. Liquid debt supply equals stationary household demand; TFP and
    labor disutility normalize steady hours to one. No production parameter is
    recalibrated when the household grid changes.
    """


def capital_production_block(*, Y: float, w: float, K: float, q: float,
                             lag_q: float, Z: float, alpha: float = .33,
                             delta: float = .02, epsI: float = 4., pi: float = 0.,
                             mu: float = 1.015, kappa: float = .1) -> dict[str, float]:
    """Evaluate production, investment, and the realized return in levels.

    ``K`` is inherited physical capital, ``lag_q`` its actual previous purchase
    price, and ``q`` the current price of new installed capital. ``I`` excludes
    installation costs; ``investment_goods`` includes them. Negative physical
    investment is allowed, as in the reference's reversible technology, but
    next-period capital and gross capital returns must remain positive.

    With psi=1/(delta*epsI), Phi=psi/2*(I/K-delta)**2*K and
    q=1+psi*(I/K-delta), capital's payoff includes -Phi_K. Equivalently it
    includes the installation rent (q-1)*I-Phi; omitting that term violates the
    capital valuation equation away from the stationary allocation.
    """
    values = [Y, w, K, q, lag_q, Z, alpha, delta, epsI, pi, mu, kappa]
    if (not np.isfinite(values).all() or min(Y, w, K, q, lag_q, Z, epsI, kappa) <= 0
            or not (0 < alpha < 1 and 0 < delta < 1 and mu > 1 and pi > -1)):
        raise ValueError("invalid capital production parameters or positive-price domain")
    N = (Y / (Z * K**alpha))**(1 / (1-alpha))
    mc = w*N / ((1-alpha)*Y)
    rental = alpha*mc*Y/K
    psi = 1 / (delta*epsI)
    I = K*(delta+(q-1)/psi)
    Knext = (1-delta)*K+I
    Phi = .5*psi*(I/K-delta)**2*K
    installation_rent = (q-1)*I-Phi
    payoff = rental+(1-delta)*q+installation_rent/K
    ra = payoff/lag_q-1
    price_cost = mu/(mu-1)/(2*kappa)*np.log1p(pi)**2*Y
    div = Y-w*N-rental*K-price_cost
    if not np.isfinite([N, mc, rental, I, Knext, Phi, ra, div]).all() or min(Knext, 1+ra) <= 0:
        raise ValueError("capital production left the positive-capital/return domain")
    return dict(Y=float(Y), w=float(w), K=float(K), q=float(q), lag_q=float(lag_q),
                N=float(N), mc=float(mc), rental=float(rental), I=float(I),
                Knext=float(Knext), capital_adjustment=float(Phi),
                investment_goods=float(I+Phi), installation_rent=float(installation_rent),
                r_a=float(ra), adj=float(price_cost), div=float(div))


def calibrate_capital_hank(*, n_a: int = 15, n_b: int = 15, a_max: float = 200.,
                           b_max: float = 30., gamma: float = 2., r_b: float = .0075,
                           r_a: float = .0125, income: float = 1., chi_0: float = 1.,
                           chi_1: float = 1., a_bar: float = .25, alpha: float = .33,
                           delta: float = .02, epsI: float = 4., mu: float = 1.015,
                           government_share: float = .2, kappa: float = .1,
                           phi: float = 1.5, frisch: float = 1.,
                           beta_bracket: tuple[float, float] = (.96, .985),
                           household_tolerance: float = 1e-10,
                           calibration_tolerance: float = 1e-10) -> CapitalHANKSteadyState:
    """Solve a stationary two-asset productive-capital equilibrium.

    Beta solves ``(r_a+delta)*A = alpha*Y/mu`` with q=1 and
    ``Y=(C+CHI+delta*A)/(1-government_share)``. This retains fixed alpha,
    depreciation, markups, and steady returns across grid refinements. Debt
    supply is calibrated from B; government purchases are subsequently fixed
    in levels, not a feedback rule proportional to fluctuating output.

    Alpha=.33, delta=.02, epsI=4, mu=1.015, inverse IES=2 and Frisch=1
    follow rounded ABRS reference parameters. The two-state income process,
    portfolio cost denominator, direct capital ownership, flexible wages, and
    fiscal incidence differ. This function does not replicate their model.
    """
    from .hank_recursive import _integer
    _integer(n_a, "n_a", 3)
    _integer(n_b, "n_b", 3)
    if len(beta_bracket) != 2:
        raise ValueError("beta_bracket must contain two ordered discount factors")
    values = [a_max, b_max, gamma, r_b, r_a, income, chi_0, chi_1, a_bar, alpha,
              delta, epsI, mu, government_share, kappa, phi, frisch,
              household_tolerance, calibration_tolerance, *beta_bracket]
    if (not np.isfinite(values).all()
            or min(a_max, b_max, gamma, income, chi_0, a_bar, epsI, kappa, frisch,
                   household_tolerance, calibration_tolerance) <= 0
            or not (0 < alpha < 1 and 0 < delta < 1 and mu > 1 and phi > 1
                    and r_a > r_b > -1 and r_a > 0 and chi_1 >= 1
                    and 0 <= government_share < 1
                    and 0 < beta_bracket[0] < beta_bracket[1] < 1/(1+r_a))):
        raise ValueError("invalid productive-capital equilibrium parameters")
    if mu*(1-government_share) <= alpha:
        raise ValueError("government share and capital share preclude positive-income equilibrium")

    cache: dict[float, Any] = {}
    history: list[dict[str, Any]] = []

    def evaluate(beta: float) -> float:
        if beta not in cache:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                hh = _solve_two_asset_household_block(n_a=n_a, n_b=n_b,
                    a_max=a_max, b_max=b_max, beta=beta, gamma=gamma, r_b_ss=r_b,
                    r_a_ss=r_a, w_ss=income, chi_0=chi_0, chi_1=chi_1,
                    a_bar=a_bar, tol=household_tolerance)
            history.append(dict(beta=float(beta), converged=bool(hh.converged),
                household_error=float(hh.residual), A=float(hh.A_ss), B=float(hh.B_ss),
                warnings=[str(item.message) for item in caught]))
            if (not hh.converged or abs(hh.budget_residual) > 1e-8
                    or min(hh.A_ss, hh.B_ss) <= 0):
                warning_text = "; ".join(str(item.message) for item in caught)
                raise ValueError("capital calibration household solve failed at "
                    f"beta={beta:.12g}; policy error={hh.residual:.3e}; "
                    f"household warnings: {warning_text or 'none'}")
            cache[beta] = hh
            # Each block owns a full stationary transition matrix. Retain only
            # the two most recent solves, while preserving every trial's scalar
            # diagnostics and warnings in history.
            if len(cache) > 2:
                del cache[next(iter(cache))]
        hh = cache[beta]
        Y = (hh.Y_ss+delta*hh.A_ss)/(1-government_share)
        return float((r_a+delta)*hh.A_ss-alpha*Y/mu)

    lower, upper = (evaluate(float(beta)) for beta in beta_bracket)
    if lower*upper > 0:
        raise ValueError("beta_bracket does not bracket capital-market clearing; "
                         f"residuals are {lower:.6g} and {upper:.6g}")
    beta = float(brentq(evaluate, *beta_bracket, xtol=calibration_tolerance, rtol=1e-14))
    error = evaluate(beta)
    hh = cache[beta]
    if abs(hh.N_ss-1) > 1e-10:
        raise ValueError("capital model requires stationary productivity mean one")
    K = float(hh.A_ss)
    Y = float((hh.Y_ss+delta*K)/(1-government_share))
    G, N = government_share*Y, 1.
    Z, w = Y/K**alpha, (1-alpha)*Y/mu
    block = capital_production_block(Y=Y, w=w, K=K, q=1., lag_q=1., Z=Z,
        alpha=alpha, delta=delta, epsI=epsI, mu=mu, kappa=kappa)
    uce = float(np.sum(hh.D_ss*hh.s_grid*hh.c_ss**(-gamma)))
    tax = r_b*hh.B_ss+G
    p = dict(beta=beta, gamma=gamma, r=r_b, r_a=r_a, B=float(hh.B_ss), K=K,
             q=1., Y=Y, N=N, Z=Z, w=w, alpha=alpha, delta=delta, epsI=epsI,
             psi=1/(delta*epsI), G=G, government_share=government_share,
             mu=mu, kappa=kappa, phi=phi, frisch=frisch, vphi=w*uce,
             income=income, chi_0=chi_0, chi_1=chi_1, a_bar=a_bar)
    goods = Y-hh.C_ss-hh.CHI_ss-delta*K-G
    implied_income = w*N+block["div"]-tax
    diagnostics = dict(capital_calibration_error=float(abs(error)),
        household_policy_error=float(hh.residual), budget_error=float(abs(hh.budget_residual)),
        distribution_error=float(np.max(abs(hh.Lambda@hh.D_ss.ravel()-hh.D_ss.ravel()))),
        goods_market_error=float(abs(goods)), income_error=float(abs(implied_income-income)),
        return_error=float(abs(block["r_a"]-r_a)),
        upper_illiquid_mass=float(hh.mass_at_a_max), upper_liquid_mass=float(hh.mass_at_b_max),
        calibration_history=history, calibration_unknown="beta",
        fixed_physical_parameters=("alpha", "delta", "epsI", "mu"),
        income_process="fixed two-state chain", reference_scope="production and installation block",
        scope="direct productive capital and nominal government bonds; flexible common hours")
    if max(abs(goods), abs(implied_income-income), abs(block["r_a"]-r_a)) > 1e-7:
        raise ValueError("productive-capital stationary accounts did not converge")
    aggregates = dict(Y=Y, C=float(hh.C_ss), CHI=float(hh.CHI_ss), A=K,
        B=float(hh.B_ss), K=K, Knext=K, I=delta*K, capital_adjustment=0.,
        investment_goods=delta*K, installation_rent=0., G=G, Tax=float(tax),
        Div=block["div"], w=w, q=1., N=N, mc=1/mu, rental=r_a+delta)
    return CapitalHANKSteadyState(p, hh.D_ss.copy(), aggregates, diagnostics, True, hh)


class _CapitalKernel(_TwoAssetKernel):
    """Aggregate-capital extension of the existing household kernel protocol."""

    model = "two_asset_productive_capital"
    aggregate_state_names = ("lag_nominal", "monetary_shock", "log_capital")

    def __init__(self, ss: CapitalHANKSteadyState):
        if not isinstance(ss, CapitalHANKSteadyState):
            raise ValueError("a converged CapitalHANKSteadyState is required")
        super().__init__(ss)
        self.fd_step = 1e-6

    def prices_from_state(self, u, aggregate_state, distribution=None, *, lag_q=None):
        p = self.p
        state = np.asarray(aggregate_state)
        if state.shape != (3,) or np.shape(u) != (4,) or not np.isfinite(state).all():
            raise ValueError("capital prices require three aggregate states and four controls")
        pi, w, Y, q = np.expm1(u[0]), self.w*np.exp(u[1]), self.Y*np.exp(u[2]), np.exp(u[3])
        K = p["K"]*np.exp(state[2])
        if lag_q is None:
            d = self.d if distribution is None else distribution
            lag_q = float(np.sum(d*self.hh.a_grid[:, None, None])/K)
        v = capital_production_block(Y=Y, w=w, K=K, q=q, lag_q=lag_q,
            Z=p["Z"], alpha=p["alpha"], delta=p["delta"], epsI=p["epsI"],
            pi=pi, mu=p["mu"], kappa=p["kappa"])
        nominal = self.r+state[1]+self.phi*pi
        rb = (1+self.r+state[0])/(1+pi)-1
        tax = rb*p["B"]+p["G"]
        income = w*v["N"]+v["div"]-tax
        if not np.isfinite([nominal, rb, income]).all() or min(1+nominal, 1+rb, income) <= 0:
            raise ValueError("capital equilibrium left the positive income/return domain")
        v.update(pi=float(pi), nominal=float(nominal), r=float(rb), tax=float(tax),
                 income=float(income), G=p["G"])
        return v

    def prices(self, u, lag_nominal, shock, distribution=None, lag_q=None):
        raise ValueError("productive capital requires prices_from_state with inherited log_capital")

    def next_aggregate_state(self, prices, aggregate_state, rho):
        return np.array([prices["nominal"]-self.r, rho*aggregate_state[1],
                         np.log(prices["Knext"]/self.p["K"])])

    def _tangent(self, dm, dp):
        # Directional household steps only: never construct the full-grid
        # marginal-by-marginal derivative cache of the legacy kernel.
        scale = max(float(np.max(abs(dm)/(1+abs(self.m)))), float(np.max(abs(dp))))
        if scale == 0:
            return np.zeros(6*self.d.size)
        return self._response(dm/scale, dp/scale, self.fd_step)*scale

    def linear_residual(self, cur, nxt, basis, rho):
        k, n, p, hh = basis.rank, self.d.size, self.p, self.hh
        ns = k+3
        dd = (basis.vectors@cur[:k]).reshape(self.shape)
        ilag, shock, capital = cur[k:ns]
        pi, wage, output, q = cur[-4:]
        hours = (output-p["alpha"]*capital)/(1-p["alpha"])
        marginal_cost = wage+hours-output
        drb = ilag-(1+self.r)*pi
        lagq = np.sum(dd*hh.a_grid[:, None, None])/p["K"]-capital
        rental = (p["r_a"]+p["delta"])*(wage+hours-capital)
        # Installation rent contributes delta*dq: together with remaining
        # capital, the derivative of the ex-dividend payoff is exactly dq.
        dra = rental+q-(1+p["r_a"])*lagq
        dz = self.Y*output-(p["r_a"]+p["delta"])*p["K"]*(wage+hours)-p["B"]*drb
        response = self._tangent(nxt[ns:ns+2*n], np.array([drb, dra, dz]))
        dm = response[:2*n]
        dc, da, db, _ = [v.reshape(self.shape) for v in np.split(response[2*n:], 4)]
        dist = self._forward_dot(dd, da, db)
        uc = hh.c_ss**(-hh.gamma)
        duce = np.sum(hh.s_grid*(dd*uc-self.d*hh.gamma*uc/hh.c_ss*dc))
        labor = self.w*(wage*np.sum(self.d*hh.s_grid*uc)+duce)-p["vphi"]*hours/p["frisch"]
        capital_next = capital+q/p["psi"]
        return np.r_[nxt[:k]-basis.vectors.T@dist.ravel(),
            nxt[k]-shock-self.phi*pi, nxt[k+1]-rho*shock, nxt[k+2]-capital_next,
            cur[ns:ns+2*n]-dm,
            p["kappa"]/p["mu"]*marginal_cost-pi+nxt[-4]/(1+self.r),
            np.sum(self.d*db+dd*hh.b_ss), labor,
            np.sum(self.d*da+dd*hh.a_ss)-p["K"]*(q+capital_next)]

    def markets(self, d, h, v, continuation):
        p, hh = self.p, self.hh
        A, B, C, chi = [float(np.sum(d*h[key])) for key in ("a", "b", "c", "chi")]
        A_in = float(np.sum(d*hh.a_grid[:, None, None]))
        B_in = float(np.sum(d*hh.b_grid[None, :, None]))
        labor = v["w"]*np.sum(d*hh.s_grid*h["c"]**(-hh.gamma))-p["vphi"]*v["N"]**(1/p["frisch"])
        nkpc = p["kappa"]*(v["mc"]-1/p["mu"])-np.log1p(v["pi"])+continuation
        liquid, illiquid = B-p["B"], A-v["q"]*v["Knext"]
        goods = v["Y"]-C-chi-v["investment_goods"]-v["adj"]-p["G"]
        inherited = A_in-v["lag_q"]*v["K"]
        budget_implied_goods = illiquid+liquid-(1+v["r"])*(B_in-p["B"])-(1+v["r_a"])*inherited
        report = dict(pi=v["pi"], r=v["r"]-self.r, r_a=v["r_a"]-p["r_a"],
            nkpc_res=float(nkpc), liquid_mkt=liquid, illiquid_mkt=illiquid,
            labor_mkt=float(labor), goods_mkt=goods,
            fiscal_res=v["tax"]-v["r"]*p["B"]-p["G"],
            capital_law_res=v["Knext"]-(1-p["delta"])*v["K"]-v["I"],
            investment_foc_res=v["q"]-1-p["psi"]*(v["I"]/v["K"]-p["delta"]),
            return_res=(1+v["r_a"])*v["lag_q"]-v["rental"]
                +v["investment_goods"]/v["K"]-v["q"]*v["Knext"]/v["K"],
            inherited_capital_mkt=inherited, inherited_liquid_mkt=B_in-p["B"],
            resource_accounting_res=goods-budget_implied_goods)
        levels = dict(C=C, A=A, B=B, CHI=chi, Tax=v["tax"], Div=v["div"])
        levels.update({name: v[name] for name in ("w", "Y", "q", "K", "Knext", "N", "I",
            "mc", "rental", "capital_adjustment", "investment_goods", "installation_rent", "G")})
        report.update({name: value-self.ss.aggregates[name] for name, value in levels.items()})
        return np.array([nkpc, liquid, labor, illiquid]), report
