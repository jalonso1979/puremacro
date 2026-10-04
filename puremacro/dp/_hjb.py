"""Compile a continuous-time dp model to the HJB upwind scheme (``vfi.solve_hjb_achdou``).

A continuous-time model declares a control and the drift of its one state
instead of a next-period value::

    m = dp.Model("huggett, continuous time")
    m.parameters(rho=0.05, gamma=2.0)
    m.exogenous("z", dp.Jump([0.1, 0.2], [[-1.2, 1.2], [1.2, -1.2]]))
    m.state("a", np.linspace(-0.15, 5.0, 500))
    m.control("c")
    m.drift("a", "r*a + z - c")
    m.reward("crra(c, gamma)")
    m.discount_rate("rho")

The compiler checks that the drift is linear in the control with slope -1
(so the first-order condition is u'(c) = V_a), takes the rest of the drift as
income, derives u' and its inverse from the reward, and passes them to the
solver. The lower end of the state grid is a state constraint, as in Achdou,
Han, Lasry, Lions and Moll (2022). The distribution is the KFE stationary
density times the grid's quadrature weights, i.e. the mass at each node.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from puremacro.dp._expr import ModelSpecError, compile_function, diff, emit, parse, substitute, symbols
from puremacro.dp._felicity import Felicity
from puremacro.dp.processes import Jump
from puremacro.vfi.equilibrium import EquilibriumResult
from puremacro.vfi.hjb_achdou import _quadrature_weights, solve_hjb_achdou


class HJBPlan:
    """Validated HJB compilation of a continuous-time ``Model``; quacks like ``_Compiled``
    for ``DPSolution`` (names, grids, ``P`` is the generator)."""

    finite = False

    def __init__(self, m):
        self.model = m
        if m._horizon is not None:
            raise ModelSpecError("continuous-time models have an infinite horizon; remove horizon()")
        if m._discrete is not None or m._taste is not None:
            raise ModelSpecError("method='hjb' does not take a discrete choice")
        if len(m._states) != 1 or set(m._drift) != {m._states[0][0]}:
            raise ModelSpecError("method='hjb' needs exactly one state with a drift(); got states "
                                 f"{[s for s, _ in m._states]} and drifts {sorted(m._drift)}")
        if len(m._controls) != 1:
            raise ModelSpecError(f"method='hjb' needs exactly one control; got {m._controls}")
        if m._reward is None:
            raise ModelSpecError("declare the payoff with reward()")
        if m._constraints:
            raise ModelSpecError("method='hjb' takes no subject_to(): the state constraint is the bottom of "
                                 "the grid and consumption is positive by the first-order condition")
        for name, proc in m._shocks:
            if not isinstance(proc, Jump):
                raise ModelSpecError(f"shock {name!r}: continuous-time models take dp.Jump processes")
        rate = m._rate
        if rate not in m._params or np.ndim(m._params[rate]):
            raise ModelSpecError(f"discount rate {rate!r} must be a scalar parameter")
        if len(m._prices) > 1:
            raise ModelSpecError("one equilibrium price is supported (scalar brentq)")
        if m._prices and m._clear is None:
            raise ModelSpecError("prices were declared but no clear() condition")
        if m._clear is not None and not m._prices:
            raise ModelSpecError("clear() needs a price declared with prices()")

        self.state_names = [m._states[0][0]]
        self.state_grids = [m._states[0][1]]
        self.a = self.state_grids[0]
        self.shock_names = [s for s, _ in m._shocks]
        self.cname = m._controls[0]
        self.param_names = list(m._params) + list(m._prices)
        names = self.state_names + self.shock_names + [self.cname]
        if len(set(names)) != len(names):
            raise ModelSpecError(f"state, shock and control names must be distinct; got {names}")

        # exogenous chain: product of independent jumps, first declared slowest
        if m._shocks:
            chains = [p.generator() for _, p in m._shocks]
            self.z_grids = [g for g, _ in chains]
            A = chains[0][1]
            for _, Ak in chains[1:]:
                A = np.kron(A, np.eye(Ak.shape[0])) + np.kron(np.eye(A.shape[0]), Ak)
            self.P = A
            self._zs = [zc.ravel() for zc in np.meshgrid(*self.z_grids, indexing="ij")]
        else:
            self.z_grids, self.P, self._zs = [], np.zeros((1, 1)), []
        self.n_z = self.P.shape[0]

        timed = self.state_names + self.shock_names
        inl: dict = {}
        for name, expr in m._locals:
            inl[name] = substitute(parse(expr, timed), inl)

        def node(text):
            n = substitute(parse(text, timed), inl)
            for name, lead in n.variables():
                if lead != 0:
                    raise ModelSpecError(f"continuous-time models have no timing: {name}({lead:+d}) in {text!r}")
            return n

        self._drift_node = node(m._drift[self.state_names[0]])
        reward = node(m._reward)
        self.felicity = Felicity(reward, self.cname, None, self.param_names, "method='hjb'")
        self._drift = self._fn("_dp_drift", self._drift_node)
        self._drift_c = self._fn("_dp_drift_c", diff(self._drift_node, self.cname))
        self._agg = {k: self._fn(f"_dp_agg_{k}", node(e)) for k, e in m._aggregates.items()}
        self._clear = None
        if m._clear is not None:
            cl = substitute(parse(m._clear, []), inl)
            ok = set(m._aggregates) | set(self.param_names)
            bad = {n for n, _ in symbols(cl)} - ok
            if bad:
                raise ModelSpecError(f"clear: unknown or state-dependent names {sorted(bad)}")

            def res(name, lead):
                return f"g_{name}" if name in m._aggregates else f"p_{name}"

            self._clear = compile_function("_dp_clear", [f"g_{k}" for k in m._aggregates]
                                           + [f"p_{p}" for p in self.param_names], [], emit(cl, res))

    # ------------------------------------------------------------------ codegen
    def _fn(self, fname, node):
        states, shocks = set(self.state_names), set(self.shock_names)
        params = set(self.param_names)

        def resolve(name, lead):
            if name == self.cname:
                return "c"
            if name in states:
                return f"s_{name}"
            if name in shocks:
                return f"x_{name}"
            if name in params:
                return f"p_{name}"
            raise ModelSpecError(f"{fname.replace('_dp_', '')}: unknown name {name!r}")

        args = ["c", f"s_{self.state_names[0]}"] + ([f"x_{s}" for s in self.shock_names] or ["_x_none"])
        return compile_function(fname, args + [f"p_{p}" for p in self.param_names], [], emit(node, resolve))

    def _call(self, fn, c, params: dict):
        zs = [z[None, :] for z in self._zs] or [np.zeros((1, 1))]
        out = fn(c, self.a[:, None], *zs, *params.values(), xp=np)
        return np.broadcast_to(np.asarray(out, dtype=float), (self.a.size, self.n_z))

    def param_values(self, prices: dict | None = None) -> dict:
        vals = {f"p_{k}": v for k, v in self.model._params.items()}
        for k in self.model._prices:
            vals[f"p_{k}"] = np.nan if prices is None or k not in prices else float(prices[k])
        return vals

    # ------------------------------------------------------------------ solving
    def income(self, params: dict) -> np.ndarray:
        """The drift at zero consumption, after checking the drift is ``income - c``."""
        for c in (0.5, 1.0, 2.0):
            slope = self._call(self._drift_c, np.full((1, 1), c), params)
            if not np.allclose(slope, -1.0, rtol=0, atol=1e-12):
                raise ModelSpecError(f"method='hjb' needs a drift linear in {self.cname!r} with slope -1, "
                                     f"e.g. 'r*a + w*z - {self.cname}'")
        s0 = np.array(self._call(self._drift, np.zeros((1, 1)), params))
        if not np.all(np.isfinite(s0)):
            raise ModelSpecError("the drift at zero consumption is not finite on the grid")
        return s0

    def solve_household(self, params: dict, *, tol, max_iter, kfe: bool):
        s0 = self.income(params)
        u, up, inv = self.felicity.funcs(params, float(max(np.max(s0), 1.0)))
        rho = float(params[f"p_{self.model._rate}"])
        return solve_hjb_achdou(rho_val=rho, a_grid=self.a, e_grid=np.arange(self.n_z, dtype=float),
                                A_z=self.P, income=s0, utility=u, u_prime=up, u_prime_inv=inv,
                                tol=tol, max_iter=max_iter, compute_kfe=kfe)

    def mass(self, raw) -> np.ndarray:
        return raw.g_dist * _quadrature_weights(self.a)[:, None]

    def policy_values(self, expr: str, raw, params: dict) -> np.ndarray:
        timed = self.state_names + self.shock_names
        return np.array(self._call(self._fn("_dp_policy", parse(expr, timed)), raw.c_policy, params))

    def aggregate_values(self, raw, mass, params: dict) -> dict:
        return {k: float(np.sum(mass * self._call(fn, raw.c_policy, params))) for k, fn in self._agg.items()}

    def solve(self, *, tol, max_iter, distribution, xtol, max_evals):
        from puremacro.dp._results import DPSolution

        m = self.model
        if not m._prices:
            params = self.param_values()
            raw = self.solve_household(params, tol=tol, max_iter=max_iter, kfe=distribution)
            mass = self.mass(raw) if distribution else None
            aggs = self.aggregate_values(raw, mass, params) if mass is not None else {}
            return DPSolution(self, raw, params=params, distribution=mass, aggregates=aggs, method="hjb")
        (pname, (lo, hi)), = m._prices.items()
        count = {"n": 0}

        def resid(price):
            count["n"] += 1
            params = self.param_values({pname: price})
            raw = self.solve_household(params, tol=tol, max_iter=max_iter, kfe=True)
            aggs = self.aggregate_values(raw, self.mass(raw), params)
            return float(self._clear(*aggs.values(), *params.values(), xp=np))

        p_star = brentq(resid, lo, hi, xtol=xtol, maxiter=max_evals)
        params = self.param_values({pname: p_star})
        raw = self.solve_household(params, tol=tol, max_iter=max_iter, kfe=True)
        mass = self.mass(raw)
        aggs = self.aggregate_values(raw, mass, params)
        res = float(self._clear(*aggs.values(), *params.values(), xp=np))
        eq = EquilibriumResult(price=p_star, residual=res, solution=raw, distribution=mass,
                               problem=None, n_evals=count["n"])
        return DPSolution(self, raw, params=params, distribution=mass, aggregates=aggs,
                          prices={pname: p_star}, equilibrium=eq, method="hjb")
