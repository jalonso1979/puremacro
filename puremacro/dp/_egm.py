"""Compile a dp model to the endogenous grid method (``puremacro.vfi.solve_egm``).

EGM applies when the model is the one-asset income-fluctuation problem in
disguise. ``EGMPlan`` checks that from the specification, symbolically where it
can and numerically on the grid where it must:

* one state ``a`` (infinite or finite horizon), optionally one discrete choice;
* the reward depends on the choices only through one local, consumption
  ``c``, and otherwise only on parameters and the discrete choice, so
  ``u(c, d)`` is well defined;
* the budget is ``c + a(+1) = R*a + y(z)``: ``c + a(+1)`` does not depend on
  ``a(+1)`` and is linear in ``a`` with one slope ``R`` across shocks (in a
  life cycle, ``R`` and ``y`` may vary with age);
* the declared constraints carve out exactly ``c > 0`` (the borrowing limit is
  the bottom of the grid, as in VFI).

Marginal utility is the symbolic derivative of the reward in ``c``. Its inverse
is closed form for ``crra(c, gamma)`` and ``log(c)`` and found by bisection
otherwise, so any increasing, strictly concave felicity works.

Life cycles are solved by backward induction with one EGM step per age: the
Euler equation u'(c_j) = beta s_j R_{j+1} E u'(c_{j+1}) with survival ``s_j``.
With a ``terminal`` value its marginal value is taken by finite differences on
the grid; without one, the last age consumes everything above the borrowing limit.

With a discrete choice each option's
conditional value is not concave, so the Euler inversion is followed by an upper
envelope (Iskhakov, Jorgensen, Rust and Schjerning 2017): every pair of adjacent
endogenous points is a candidate segment, the corner choices a(+1) at the ends of
the grid are candidates where their Kuhn-Tucker condition holds, and each target point takes the best candidate,
valued as u(c) plus the interpolated continuation at the implied a(+1).
Infinite horizons with a discrete choice iterate that step to a fixed point.
With EV1 taste shocks of scale sigma (``Model.taste_shocks``) the value is the
log-sum over options, choices are logit, the Euler right side averages marginal
utility over options, and the distribution splits mass by choice probability.
With them, ``c``/``aprime`` on the result are probability-weighted means and
model expressions are averaged over options the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve

from puremacro.dp._expr import ModelSpecError, substitute, symbols
from puremacro.dp._felicity import Felicity
from puremacro.dsge._ast import Param
from puremacro.vfi.distribution import lottery_distribution, lottery_push
from puremacro.vfi.egm import EGMSolution, egm_step, solve_egm
from puremacro.vfi.equilibrium import EquilibriumResult


@dataclass
class EGMHouseholdSolution:
    """Raw EGM result as seen by ``DPSolution``: continuous policies plus the
    value of following them (policy evaluation on the lottery transition)."""
    V: np.ndarray            # (n_a, n_z)
    c: np.ndarray            # (n_a, n_z) consumption
    aprime: np.ndarray       # (n_a, n_z) next assets (values, not indices)
    n_iter: int
    sup_norm: float
    R: float                 # gross return the budget implies
    income: np.ndarray       # y(z), (n_z,); (n_d, n_z) with a discrete choice
    egm: EGMSolution | None = field(repr=False)
    policy_d: np.ndarray | None = None     # (n_a, n_z) discrete-choice indices (most likely option)
    choice_prob: np.ndarray | None = None  # (n_d, n_a, n_z) logit probabilities (taste shocks only)
    c_by_choice: np.ndarray | None = None
    aprime_by_choice: np.ndarray | None = None


@dataclass
class EGMLifeCycleSolution:
    """Age-indexed EGM result: arrays are (horizon, n_a, n_z)."""
    V: np.ndarray
    c: np.ndarray
    aprime: np.ndarray       # next assets (values, not indices)
    horizon: int
    R: np.ndarray            # (horizon,) gross return by age
    income: np.ndarray       # (horizon, n_z) y_j(z); (horizon, n_d, n_z) with a discrete choice
    policy_d: np.ndarray | None = None     # (horizon, n_a, n_z) discrete-choice indices (most likely option)
    choice_prob: np.ndarray | None = None  # (horizon, n_d, n_a, n_z) logit probabilities (taste shocks only)
    c_by_choice: np.ndarray | None = None
    aprime_by_choice: np.ndarray | None = None


class EGMPlan:
    """Validated EGM compilation of a ``_Compiled`` model."""

    def __init__(self, comp):
        m = comp.model
        if len(comp.state_names) != 1:
            raise ModelSpecError(f"method='egm' needs exactly one state; got {comp.state_names}")
        self.comp = comp
        self.dname, self.dvals = m._discrete if m._discrete is not None else (None, np.zeros(1))
        self.a = comp.state_grids[0]
        self.n_z = comp.P.shape[0]
        self.cname = self._consumption_local()
        inl: dict = {}
        for name, node in comp.locals:
            if name != self.cname:
                inl[name] = substitute(node, inl)
        self.felicity = Felicity(substitute(comp.reward, inl), self.cname, self.dname, comp.param_names,
                                 "method='egm'")
        self._c_fn = comp._household_function("_dp_c", Param(self.cname))

    # ------------------------------------------------------------ structure
    def _consumption_local(self) -> str:
        comp = self.comp
        endo = {(n, lead) for n in comp.state_names for lead in (0, 1)}
        direct = {n for n, lead in symbols(comp.reward) if lead == 0} & set(comp.local_names)
        if {s for s in symbols(comp.reward) if s in endo}:
            raise ModelSpecError("method='egm' needs consumption as a local, e.g. "
                                 "local('c = w*exp(z) + (1 + r)*a - a(+1)') and reward('crra(c, gamma)')")
        cands = [n for n in direct if comp._local_deps[n] & endo]
        if len(cands) != 1:
            raise ModelSpecError("method='egm' needs the reward to depend on the choices through exactly one "
                                 f"local (consumption); found {sorted(cands) or 'none'}")
        return cands[0]

    def _u_funcs(self, params: dict, c_max: float, d: float = 0.0):
        return self.felicity.funcs(params, c_max, d)

    def _zs(self, ndim: int):
        comp = self.comp
        if not comp.z_grids:
            return [np.zeros((1,) * ndim)]
        return [zc.reshape((1,) * (ndim - 1) + (self.n_z,))
                for zc in np.meshgrid(*comp.z_grids, indexing="ij")]

    def _call(self, fn, nxt, cur, ndim: int, params: dict, age, d=None):
        args = ([self.dvals[0] if d is None else d] if self.dname else [])
        args += [nxt, cur, *self._zs(ndim)] + ([int(age)] if self.comp.finite else [])
        return fn(*args, *params.values(), xp=np)

    @staticmethod
    def at_age(params: dict, age: int) -> dict:
        return {k: (v[age] if np.ndim(v) else v) for k, v in params.items()}

    def budgets(self, params: dict, age=None) -> tuple[float, np.ndarray]:
        """(R, Y) with Y[d] the income of each discrete option; R may not depend on the option."""
        out = [self.budget(params, age, d) for d in self.dvals]
        R = out[0][0]
        if any(abs(r - R) > 1e-10 * (1.0 + abs(R)) for r, _ in out):
            raise ModelSpecError("method='egm' needs the return on assets R not to depend on the discrete choice")
        return R, np.stack([y for _, y in out])

    def budget(self, params: dict, age=None, d=None) -> tuple[float, np.ndarray]:
        """Return (R, y) with c + a(+1) = R*a + y(z), or raise if the budget is not of that form."""
        a, n_a = self.a, self.a.size
        cur, nxt = a.reshape(n_a, 1, 1), a.reshape(1, n_a, 1)
        c3 = np.broadcast_to(self._call(self._c_fn, nxt, cur, 3, params, age, d), (n_a, n_a, self.n_z))
        coh = c3 + nxt
        scale = 1.0 + np.nanmax(np.abs(coh))
        if not np.all(np.isfinite(coh)) or np.max(np.abs(coh - coh[:, :1, :])) > 1e-9 * scale:
            raise ModelSpecError(f"method='egm' needs {self.cname} + a(+1) not to depend on a(+1) "
                                 "(budget c + a(+1) = R*a + y(z))")
        coh2 = coh[:, 0, :]
        slopes = np.diff(coh2, axis=0) / np.diff(a)[:, None]
        R = float(np.mean(slopes))
        if np.max(np.abs(slopes - R)) > 1e-8 * (1.0 + abs(R)):
            raise ModelSpecError(f"method='egm' needs {self.cname} + a(+1) linear in a with one slope "
                                 "for every shock (budget c + a(+1) = R*a + y(z))")
        if R <= 0:
            raise ModelSpecError(f"method='egm' needs a positive gross return R; got {R:.6g}")
        feasible = np.isfinite(np.broadcast_to(
            self._call(self.comp.household, nxt, cur, 3, params, age, d), c3.shape))
        if not np.array_equal(feasible, c3 > 0):
            raise ModelSpecError(f"method='egm' supports exactly the constraint {self.cname} > 0 (plus the "
                                 "grid's borrowing limit); the declared constraints or reward domain differ")
        return R, coh2[0] - R * a[0]

    # ------------------------------------------------------------ solving
    def solve_household(self, params: dict, *, tol: float, max_iter: int, c0=None) -> EGMHouseholdSolution:
        if self.dname:
            return self._solve_discrete_infinite(params, tol=tol, max_iter=max_iter)
        R, y = self.budget(params)
        _, u_prime, u_prime_inv = self._u_funcs(params, float(np.max(R * self.a[-1] + y) - self.a[0]))
        egm = solve_egm(self.a, np.arange(self.n_z), y, self.comp.P, beta=self.comp.beta, r=R - 1.0,
                        u_prime=u_prime, u_prime_inv=u_prime_inv, c0=c0, tol=tol, max_iter=max_iter)
        V = self._policy_value(egm.aprime, params)
        return EGMHouseholdSolution(V=V, c=egm.c, aprime=egm.aprime, n_iter=egm.n_iter,
                                    sup_norm=egm.sup_norm, R=R, income=np.asarray(y), egm=egm)

    def _policy_value(self, aprime: np.ndarray, params: dict) -> np.ndarray:
        """V solving V = u + beta * E V(a', z') with a' split linearly between grid nodes."""
        a, n_a, n_z = self.a, self.a.size, self.n_z
        u = np.broadcast_to(self.comp.household(aprime, a[:, None], *self._zs(2), *params.values(), xp=np),
                            (n_a, n_z))
        j = np.clip(np.searchsorted(a, aprime) - 1, 0, n_a - 2)
        w = np.clip((a[j + 1] - aprime) / (a[j + 1] - a[j]), 0.0, 1.0)
        rows, cols, vals = [], [], []
        src = np.arange(n_a * n_z).reshape(n_a, n_z)
        for zp in range(n_z):
            pz = self.comp.P[:, zp][None, :]
            rows += [src.ravel(), src.ravel()]
            cols += [(j * n_z + zp).ravel(), ((j + 1) * n_z + zp).ravel()]
            vals += [(w * pz).ravel(), ((1.0 - w) * pz).ravel()]
        T = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n_a * n_z, n_a * n_z))
        A = sp.identity(n_a * n_z, format="csr") - self.comp.beta * T
        return np.asarray(spsolve(A.tocsc(), np.ascontiguousarray(u).ravel())).reshape(n_a, n_z)

    def distribution(self, sol) -> np.ndarray:
        if isinstance(sol, EGMLifeCycleSolution):
            return self._cohort_distribution(sol)
        if sol.choice_prob is None:
            return lottery_distribution(sol.aprime, self.a, self.comp.P)
        mu = np.full(sol.c.shape, 1.0 / sol.c.size)
        for _ in range(100_000):
            new = self._push(mu, sol)
            if np.max(np.abs(new - mu)) < 1e-12:
                return new / new.sum()
            mu = new
        raise RuntimeError("stationary distribution with taste shocks did not converge")

    def _terminal(self):
        m, a = self.comp.model, self.a
        if m._terminal is None:
            return None, None
        terminal = np.asarray(m._terminal, dtype=float)
        if terminal.shape != (a.size, self.n_z):
            raise ModelSpecError(f"terminal value must have shape ({a.size}, {self.n_z}); got {terminal.shape}")
        dV = np.gradient(terminal, a, axis=0)
        if not np.all(dV > 0):
            raise ModelSpecError("method='egm' needs a terminal value strictly increasing in assets")
        return terminal, dV

    def _survival(self):
        m = self.comp.model
        J = int(m._horizon or 0)
        surv = np.ones(J) if m._survival is None else np.asarray(m._survival, dtype=float)
        if surv.shape != (J,) or np.any(surv <= 0) or np.any(surv > 1):
            raise ModelSpecError(f"survival must be a length-{J} array in (0, 1]")
        return surv

    def solve_life_cycle(self, params: dict) -> EGMLifeCycleSolution:
        if self.dname:
            return self._solve_discrete_life_cycle(params)
        m, comp = self.comp.model, self.comp
        a, n_a, n_z, P = self.a, self.a.size, self.n_z, comp.P
        J = int(m._horizon or 0)
        surv = self._survival()
        Rs, ys, ups, invs = np.empty(J), np.empty((J, n_z)), [], []
        for j in range(J):
            Rs[j], ys[j] = self.budget(params, age=j)
            _, up, inv = self._u_funcs(self.at_age(params, j), float(np.max(Rs[j] * a[-1] + ys[j]) - a[0]))
            ups.append(up)
            invs.append(inv)
        terminal, dV = self._terminal()
        c = np.empty((J, n_a, n_z))
        for j in range(J - 1, -1, -1):
            coh = Rs[j] * a[:, None] + ys[j][None, :]
            if j == J - 1 and terminal is None:
                c[j] = coh - a[0]
            else:
                emv = dV if j == J - 1 else Rs[j + 1] * ups[j + 1](c[j + 1])
                c[j] = egm_step(comp.beta * surv[j] * (emv @ P.T), a, ys[j], Rs[j], invs[j])
        coh = Rs[:, None, None] * a[None, :, None] + ys[:, None, :]
        aprime = np.clip(coh - c, a[0], a[-1])
        c = coh - aprime
        V = np.empty((J, n_a, n_z))
        V_next = np.zeros((n_a, n_z)) if terminal is None else terminal
        for j in range(J - 1, -1, -1):
            u = np.broadcast_to(self._call(comp.household, aprime[j], a[:, None], 2, params, j), (n_a, n_z))
            EV = V_next @ P.T
            cont = np.stack([np.interp(aprime[j][:, z], a, EV[:, z]) for z in range(n_z)], axis=1)
            V[j] = u + comp.beta * surv[j] * cont
            V_next = V[j]
        return EGMLifeCycleSolution(V=V, c=c, aprime=aprime, horizon=J, R=Rs, income=ys)

    # ------------------------------------------------------------ discrete choice
    def _dc_funcs(self, params: dict, R: float, Y: np.ndarray):
        c_max = float(np.max(R * self.a[-1] + Y) - self.a[0])
        return [self._u_funcs(params, c_max, float(d)) for d in self.dvals]

    def _dc_step(self, funcs, R, Y, W, rhs):
        """One DC-EGM step. ``W[a', z]`` is the discounted expected continuation and
        ``rhs[a', z]`` the discounted expected marginal value (None: last age, no
        continuation). Returns consumption and value by option, (n_d, n_a, n_z)."""
        a, n_z = self.a, self.n_z
        n_d = len(funcs)
        c_all = np.empty((n_d, a.size, n_z))
        v_all = np.empty((n_d, a.size, n_z))
        for k, (u, _, inv) in enumerate(funcs):
            coh = R * a[:, None] + Y[k][None, :]
            if rhs is None:
                c_all[k] = coh - a[0]
                with np.errstate(all="ignore"):
                    v_all[k] = np.where(c_all[k] > 0, u(c_all[k]), -np.inf)
                continue
            with np.errstate(all="ignore"):
                c_e = inv(rhs)
            m_e = c_e + a[:, None]
            for z in range(n_z):
                c_all[k, :, z], v_all[k, :, z] = _upper_envelope(coh[:, z], m_e[:, z], c_e[:, z], a, W[:, z], u)
        return c_all, v_all

    @staticmethod
    def _combine(v_all, sigma: float):
        """Value and choice probabilities: the max and its one-hot choice without
        taste shocks, the log-sum and logit probabilities with EV1 scale ``sigma``."""
        d_idx = np.argmax(v_all, axis=0)
        vmax = np.take_along_axis(v_all, d_idx[None], axis=0)[0]
        if sigma == 0.0:
            probs = (np.arange(v_all.shape[0])[:, None, None] == d_idx[None]).astype(float)
            return vmax, probs, d_idx
        with np.errstate(all="ignore"):
            e = np.exp((v_all - vmax[None]) / sigma)
        tot = e.sum(axis=0)
        return vmax + sigma * np.log(tot), e / tot[None], d_idx

    @staticmethod
    def _marginal(funcs, c_all, probs):
        mu = np.zeros(c_all.shape[1:])
        for k, (_, up, _) in enumerate(funcs):
            mu += np.where(probs[k] > 0, probs[k] * up(np.maximum(c_all[k], 1e-300)), 0.0)
        return mu

    def _by_choice(self, R, Y, c_all):
        """Next assets by option (clipped to the grid) and the consumption the clip implies."""
        coh = R * self.a[:, None] + Y[:, None, :]
        ap = np.clip(coh - c_all, self.a[0], self.a[-1])
        return coh - ap, ap

    def _taste_scale(self, params: dict) -> float:
        t = self.comp.model._taste
        if t is None:
            return 0.0
        val = float(params[f"p_{t}"]) if isinstance(t, str) else float(t)
        if not np.isfinite(val) or val < 0:
            raise ModelSpecError(f"taste-shock scale must be >= 0; got {val}")
        return val

    def _solve_discrete_life_cycle(self, params: dict) -> EGMLifeCycleSolution:
        comp, P = self.comp, self.comp.P
        n_a, n_z, n_d = self.a.size, self.n_z, self.dvals.size
        J = int(comp.model._horizon or 0)
        surv = self._survival()
        terminal, dV = self._terminal()
        sigma = self._taste_scale(params)
        Rs, Ys = np.empty(J), np.empty((J, n_d, n_z))
        funcs = []
        for j in range(J):
            Rs[j], Ys[j] = self.budgets(params, age=j)
            funcs.append(self._dc_funcs(self.at_age(params, j), Rs[j], Ys[j]))
        c_all = np.empty((J, n_d, n_a, n_z))
        ap_all = np.empty((J, n_d, n_a, n_z))
        probs = np.empty((J, n_d, n_a, n_z))
        V = np.empty((J, n_a, n_z))
        d_idx = np.empty((J, n_a, n_z), dtype=np.int64)
        for j in range(J - 1, -1, -1):
            b = comp.beta * surv[j]
            if j == J - 1 and terminal is None:
                W = rhs = None
            elif j == J - 1:
                W, rhs = b * (terminal @ P.T), b * (dV @ P.T)
            else:
                W = b * (V[j + 1] @ P.T)
                rhs = b * ((Rs[j + 1] * self._marginal(funcs[j + 1], c_all[j + 1], probs[j + 1])) @ P.T)
            cj, vj = self._dc_step(funcs[j], Rs[j], Ys[j], W, rhs)
            V[j], probs[j], d_idx[j] = self._combine(vj, sigma)
            c_all[j], ap_all[j] = self._by_choice(Rs[j], Ys[j], cj)
        return self._discrete_result(EGMLifeCycleSolution, sigma, c_all, ap_all, probs, d_idx,
                                     V=V, horizon=J, R=Rs, income=Ys)

    def _solve_discrete_infinite(self, params: dict, *, tol: float, max_iter: int) -> EGMHouseholdSolution:
        P, b = self.comp.P, self.comp.beta
        R, Y = self.budgets(params)
        sigma = self._taste_scale(params)
        funcs = self._dc_funcs(params, R, Y)
        c_all, v_all = self._dc_step(funcs, R, Y, None, None)
        V, probs, d_idx = self._combine(v_all, sigma)
        sup = np.inf
        for it in range(1, max_iter + 1):
            W = b * (V @ P.T)
            rhs = b * ((R * self._marginal(funcs, c_all, probs)) @ P.T)
            c_new, v_all = self._dc_step(funcs, R, Y, W, rhs)
            V_new, probs, d_idx = self._combine(v_all, sigma)
            with np.errstate(invalid="ignore"):
                dc = np.where(probs > 0, np.abs(c_new - c_all), 0.0)
            sup = float(max(np.max(dc), np.max(np.abs(V_new - V))))
            c_all, V = c_new, V_new
            if sup < tol:
                break
        else:
            raise RuntimeError(f"DC-EGM did not converge in {max_iter} iterations (sup-norm {sup:.3e} > tol {tol:.1e})")
        c_all, ap_all = self._by_choice(R, Y, c_all)
        return self._discrete_result(EGMHouseholdSolution, sigma, c_all, ap_all, probs, d_idx,
                                     V=V, n_iter=it, sup_norm=sup, R=R, income=Y, egm=None)

    @staticmethod
    def _discrete_result(cls, sigma, c_all, ap_all, probs, d_idx, **kw):
        ax = c_all.ndim - 3                      # option axis: 0, or 1 after the age axis
        if sigma == 0.0:
            c = np.take_along_axis(c_all, np.expand_dims(d_idx, ax), axis=ax).squeeze(ax)
            ap = np.take_along_axis(ap_all, np.expand_dims(d_idx, ax), axis=ax).squeeze(ax)
            return cls(c=c, aprime=ap, policy_d=d_idx, **kw)
        return cls(c=(probs * c_all).sum(axis=ax), aprime=(probs * ap_all).sum(axis=ax), policy_d=d_idx,
                   choice_prob=probs, c_by_choice=c_all, aprime_by_choice=ap_all, **kw)

    def _push(self, mu, sol, age=None):
        a, P = self.a, self.comp.P
        if sol.choice_prob is None:
            return lottery_push(mu, sol.aprime if age is None else sol.aprime[age], a, P)
        probs = sol.choice_prob if age is None else sol.choice_prob[age]
        aps = sol.aprime_by_choice if age is None else sol.aprime_by_choice[age]
        return sum(lottery_push(mu * probs[k], aps[k], a, P) for k in range(probs.shape[0]))

    def _cohort_distribution(self, sol: EGMLifeCycleSolution) -> np.ndarray:
        from puremacro.vfi.finite_horizon import _z_stationary

        n_a, n_z, P = self.a.size, self.n_z, self.comp.P
        nb = self.comp.model._newborns
        if nb is None:
            mu0 = np.zeros((n_a, n_z))
            mu0[0] = _z_stationary(P)
        else:
            mu0 = np.asarray(nb, dtype=float)
            if mu0.shape != (n_a, n_z):
                raise ModelSpecError(f"newborns must have shape ({n_a}, {n_z}); got {mu0.shape}")
            mu0 = mu0 / mu0.sum()
        dist = np.empty((sol.horizon, n_a, n_z))
        dist[0] = mu0
        for j in range(sol.horizon - 1):
            dist[j + 1] = self._push(dist[j], sol, j)
        return dist

    def equilibrium(self, *, tol, max_iter, xtol, max_evals):
        comp = self.comp
        (pname, (lo, hi)), = comp.model._prices.items()
        state = {"n": 0, "c": None}

        def resid(price):
            state["n"] += 1
            params = comp.param_values({pname: price})
            sol = self.solve_household(params, tol=tol, max_iter=max_iter, c0=state["c"])
            state["c"] = sol.c
            mu = self.distribution(sol)
            aggs = comp.aggregate_values(sol, mu, params)
            return float(comp._clear_fn(*aggs.values(), *params.values(), xp=np))

        p_star = brentq(resid, lo, hi, xtol=xtol, maxiter=max_evals)
        params = comp.param_values({pname: p_star})
        sol = self.solve_household(params, tol=tol, max_iter=max_iter, c0=state["c"])
        mu = self.distribution(sol)
        aggs = comp.aggregate_values(sol, mu, params)
        res = float(comp._clear_fn(*aggs.values(), *params.values(), xp=np))
        eq = EquilibriumResult(price=p_star, residual=res, solution=sol, distribution=mu,
                               problem=None, n_evals=state["n"])
        return eq, params, aggs


def _upper_envelope(m_x, m_e, c_e, a, W, u):
    """Best of the corner choices and the interpolated Euler segments at cash on hand ``m_x``.

    ``m_e``/``c_e`` are the endogenous cash on hand and consumption for next
    assets ``a`` (not necessarily monotone in ``a``), ``W`` the discounted
    expected continuation on ``a``. Each segment [m_e[i], m_e[i+1]] proposes
    consumption by linear interpolation; next assets then lie on [a[i], a[i+1]]
    at the same weight, so the candidate value is u(c) + the interpolated W.
    Saving the grid minimum is a candidate where it satisfies the Kuhn-Tucker
    condition u'(c) >= beta E V'(a[0]), i.e. ``m_x <= m_e[0]``, and saving the
    maximum where ``m_x >= m_e[-1]``. Returns (c, v).
    """
    with np.errstate(all="ignore"):
        cands_c = [m_x - a[0], m_x - a[-1]]
        cands_v = [np.where(m_x <= m_e[0], u(m_x - a[0]) + W[0], -np.inf),
                   np.where(m_x >= m_e[-1], u(m_x - a[-1]) + W[-1], -np.inf)]
        m0, m1 = m_e[:-1, None], m_e[1:, None]
        dm = m1 - m0
        t = (m_x[None, :] - m0) / np.where(dm == 0, np.nan, dm)
        inside = (t >= 0.0) & (t <= 1.0)
        cs = c_e[:-1, None] + t * (c_e[1:, None] - c_e[:-1, None])
        ws = W[:-1, None] + t * (W[1:, None] - W[:-1, None])
        vs = np.where(inside & (cs > 0), u(cs) + ws, -np.inf)
        C = np.vstack([np.vstack(cands_c), cs])
        Vv = np.vstack([np.vstack(cands_v), vs])
        Vv = np.where(np.isfinite(Vv) & (C > 0), Vv, -np.inf)
    k = np.argmax(Vv, axis=0)
    cols = np.arange(m_x.size)
    return C[k, cols], Vv[k, cols]
