"""Compile a dp model to the endogenous grid method (``puremacro.vfi.solve_egm``).

EGM applies when the model is the one-asset income-fluctuation problem in
disguise. ``EGMPlan`` checks that from the specification, symbolically where it
can and numerically on the grid where it must:

* infinite horizon, one state ``a``, no discrete choice;
* the reward depends on the choices only through one local, consumption
  ``c``, and otherwise only on parameters, so ``u(c)`` is well defined;
* the budget is ``c + a(+1) = R*a + y(z)``: ``c + a(+1)`` does not depend on
  ``a(+1)`` and is linear in ``a`` with one slope ``R`` across shocks;
* the declared constraints carve out exactly ``c > 0`` (the borrowing limit is
  the bottom of the grid, as in VFI).

Marginal utility is the symbolic derivative of the reward in ``c``. Its inverse
is closed form for ``crra(c, gamma)`` and ``log(c)`` and found by bisection
otherwise, so any increasing, strictly concave felicity works.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve

from puremacro.dp._expr import ModelSpecError, compile_function, diff, emit, substitute, symbols
from puremacro.dsge._ast import Call, Param
from puremacro.vfi.distribution import lottery_distribution
from puremacro.vfi.egm import EGMSolution, solve_egm
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
    income: np.ndarray       # y(z), (n_z,)
    egm: EGMSolution = field(repr=False)
    policy_d: None = None


class EGMPlan:
    """Validated EGM compilation of a ``_Compiled`` model."""

    def __init__(self, comp):
        m = comp.model
        if comp.finite:
            raise ModelSpecError("method='egm' needs an infinite horizon; life-cycle EGM is not built yet")
        if len(comp.state_names) != 1:
            raise ModelSpecError(f"method='egm' needs exactly one state; got {comp.state_names}")
        if m._discrete is not None:
            raise ModelSpecError("method='egm' does not take a discrete choice (DC-EGM is not built yet)")
        self.comp = comp
        self.a = comp.state_grids[0]
        self.n_z = comp.P.shape[0]
        self.cname = self._consumption_local()
        allowed = {(p, 0) for p in comp.param_names}

        inl: dict = {}
        for name, node in comp.locals:
            if name != self.cname:
                inl[name] = substitute(node, inl)
        u_node = substitute(comp.reward, inl)
        bad = symbols(u_node) - allowed - {(self.cname, 0)}
        if bad:
            raise ModelSpecError(
                f"method='egm' needs a reward that depends on choices only through {self.cname!r} "
                f"and otherwise on parameters; it also uses {sorted(n for n, _ in bad)}")

        def resolve(name, lead):
            return "c" if name == self.cname else f"p_{name}"

        pargs = [f"p_{p}" for p in comp.param_names]
        self._u_prime = compile_function("_dp_u_prime", ["c"] + pargs, [],
                                         emit(diff(u_node, self.cname), resolve))
        self._inv_kind, self._inv_fn = "numeric", None
        if isinstance(u_node, Call) and u_node.args and u_node.args[0] == Param(self.cname):
            fn = u_node.func.lower()
            if fn == "crra":
                self._inv_kind = "crra"
                self._inv_fn = compile_function("_dp_gamma", pargs, [], emit(u_node.args[1], resolve))
            elif fn in ("log", "ln"):
                self._inv_kind = "log"
        self._c_fn = comp._household_function("_dp_c", Param(self.cname))

    # ------------------------------------------------------------ structure
    def _consumption_local(self) -> str:
        comp = self.comp
        endo = comp._endogenous_syms()
        direct = {n for n, lead in symbols(comp.reward) if lead == 0} & set(comp.local_names)
        if {s for s in symbols(comp.reward) if s in endo}:
            raise ModelSpecError("method='egm' needs consumption as a local, e.g. "
                                 "local('c = w*exp(z) + (1 + r)*a - a(+1)') and reward('crra(c, gamma)')")
        cands = [n for n in direct if comp._local_deps[n] & endo]
        if len(cands) != 1:
            raise ModelSpecError("method='egm' needs the reward to depend on the choices through exactly one "
                                 f"local (consumption); found {sorted(cands) or 'none'}")
        return cands[0]

    def _u_funcs(self, params: dict, c_max: float):
        pv = list(params.values())

        def u_prime(c):
            with np.errstate(all="ignore"):
                return np.broadcast_to(self._u_prime(c, *pv, xp=np), np.shape(c))

        if self._inv_kind == "crra":
            g = float(self._inv_fn(*pv, xp=np))

            def u_prime_inv(x):
                return x ** (-1.0 / g)
        elif self._inv_kind == "log":
            def u_prime_inv(x):
                return 1.0 / x
        else:
            def u_prime_inv(x):
                lo = np.full(np.shape(x), np.log(1e-12))
                hi = np.full(np.shape(x), np.log(1e12))
                for _ in range(64):
                    mid = 0.5 * (lo + hi)
                    up = u_prime(np.exp(mid)) > x       # still above target: c too small
                    lo = np.where(up, mid, lo)
                    hi = np.where(up, hi, mid)
                return np.exp(0.5 * (lo + hi))

        cs = np.geomspace(1e-3 * min(1.0, c_max), c_max, 81)
        mu = u_prime(cs)
        if not (np.all(np.isfinite(mu)) and np.all(mu > 0) and np.all(np.diff(mu) < 0)):
            raise ModelSpecError("method='egm' needs a reward that is increasing and strictly concave in "
                                 f"{self.cname!r} (marginal utility positive and decreasing)")
        return u_prime, u_prime_inv

    def _zs(self, ndim: int):
        comp = self.comp
        if not comp.z_grids:
            return [np.zeros((1,) * ndim)]
        return [zc.reshape((1,) * (ndim - 1) + (self.n_z,))
                for zc in np.meshgrid(*comp.z_grids, indexing="ij")]

    def budget(self, params: dict) -> tuple[float, np.ndarray]:
        """Return (R, y) with c + a(+1) = R*a + y(z), or raise if the budget is not of that form."""
        a, n_a = self.a, self.a.size
        cur, nxt = a.reshape(n_a, 1, 1), a.reshape(1, n_a, 1)
        c3 = np.broadcast_to(self._c_fn(nxt, cur, *self._zs(3), *params.values(), xp=np),
                             (n_a, n_a, self.n_z))
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
            self.comp.household(nxt, cur, *self._zs(3), *params.values(), xp=np), c3.shape))
        if not np.array_equal(feasible, c3 > 0):
            raise ModelSpecError(f"method='egm' supports exactly the constraint {self.cname} > 0 (plus the "
                                 "grid's borrowing limit); the declared constraints or reward domain differ")
        return R, coh2[0] - R * a[0]

    # ------------------------------------------------------------ solving
    def solve_household(self, params: dict, *, tol: float, max_iter: int, c0=None) -> EGMHouseholdSolution:
        R, y = self.budget(params)
        u_prime, u_prime_inv = self._u_funcs(params, float(np.max(R * self.a[-1] + y) - self.a[0]))
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

    def distribution(self, sol: EGMHouseholdSolution) -> np.ndarray:
        return lottery_distribution(sol.aprime, self.a, self.comp.P)

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
