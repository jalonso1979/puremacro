"""Declarative front end for dynamic programs: ``puremacro.dp.Model``.

A model is declared with symbolic equations (strings in Dynare expression
syntax) and compiled to the existing ``puremacro.vfi`` solvers, so the same
specification drives the infinite-horizon VFI engine (``VFIProblem``), the
life-cycle engine (``FiniteHorizonProblem``) and the stationary-equilibrium
loop (``stationary_equilibrium``) without the user writing return-function
closures, grid broadcasting or positional parameter lists.

Example (Aiyagari 1994)::

    from puremacro import dp

    m = dp.Model("aiyagari")
    m.parameters(beta=0.96, gamma=1.0, alpha=0.36, delta=0.08)
    m.prices(r=(0.005, 0.0396))
    m.exogenous("z", dp.AR1(rho=0.9, sigma=0.2, n=5))
    m.state("a", np.linspace(1e-4, 80.0, 150))
    m.local("w = (1 - alpha)*(alpha/(r + delta))^(alpha/(1 - alpha))")
    m.local("c = w*exp(z) + (1 + r)*a - a(+1)")
    m.reward("crra(c, gamma)")
    m.subject_to("c > 0")
    m.aggregate(K="a", L="exp(z)")
    m.clear("K - L*(alpha/(r + delta))^(1/(1 - alpha))")
    sol = m.solve(tol=1e-9, howard=40)

Timing: ``a`` is the current value of a state and ``a(+1)`` its next-period
value, which the discrete solvers choose on the state's own grid. Identifiers
that are neither states, shocks nor the discrete choice resolve, in order, to
locals, parameters, prices, aggregates (only in ``clear``) and ``age`` (only in
finite-horizon models).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from puremacro.dp._expr import (
    ModelSpecError,
    compile_function,
    emit,
    is_relational,
    parse,
    split_assignment,
    symbols,
)
from puremacro.dp._results import DPSolution
from puremacro.vfi.discretize import markov_stationary
from puremacro.vfi.equilibrium import stationary_equilibrium
from puremacro.vfi.finite_horizon import FiniteHorizonProblem, life_cycle_distribution
from puremacro.vfi.problem import VFIProblem

_METHODS = ("vfi", "egm")


class Model:
    """A dynamic program declared symbolically and solved by compilation.

    Every declaration method returns the model, so calls chain. Nothing is
    solved or validated until :meth:`solve` (or :meth:`check`).
    """

    def __init__(self, name: str = "dp model"):
        self.name = str(name)
        self._params: dict[str, Any] = {}
        self._prices: dict[str, tuple[float, float]] = {}
        self._shocks: list[tuple[str, Any]] = []
        self._states: list[tuple[str, np.ndarray]] = []
        self._chosen: list[str] | None = None
        self._discrete: tuple[str, np.ndarray] | None = None
        self._locals: list[tuple[str, Any]] = []
        self._reward: str | None = None
        self._constraints: list = []
        self._beta = "beta"
        self._horizon: int | None = None
        self._survival = None
        self._terminal = None
        self._newborns = None
        self._aggregates: dict[str, Any] = {}
        self._clear: str | None = None
        self._taste: float | str | None = None

    # ------------------------------------------------------------------ declarations
    def parameters(self, **values) -> Model:
        """Calibrated parameters. A length-T array is an age-varying parameter
        (finite horizon only) and means its value at the current age."""
        for k, v in values.items():
            arr = np.asarray(v, dtype=float)
            self._params[k] = float(arr) if arr.ndim == 0 else arr
        return self

    def prices(self, **brackets) -> Model:
        """Equilibrium prices with their root-finding bracket, e.g. ``r=(0.0, 0.04)``.
        Phase 1 supports one price (scalar ``stationary_equilibrium``)."""
        for k, br in brackets.items():
            lo, hi = (float(x) for x in br)
            if not lo < hi:
                raise ModelSpecError(f"price {k!r}: bracket must satisfy lo < hi; got {br}")
            self._prices[k] = (lo, hi)
        return self

    def exogenous(self, name: str, process) -> Model:
        """An exogenous Markov shock (``dp.AR1`` or ``dp.Markov``). Several shocks
        are independent and enter as their product chain, first-declared slowest."""
        if not hasattr(process, "discretize"):
            raise ModelSpecError(f"shock {name!r}: process must be dp.AR1 or dp.Markov")
        self._shocks.append((str(name), process))
        return self

    def state(self, name: str, grid) -> Model:
        """An endogenous state and its grid; its next value ``name(+1)`` is chosen on this grid."""
        g = np.asarray(grid, dtype=float).ravel()
        if g.size < 2 or not np.all(np.isfinite(g)) or np.any(np.diff(g) <= 0):
            raise ModelSpecError(f"state {name!r}: grid must be finite, strictly increasing, >= 2 points")
        self._states.append((str(name), g))
        return self

    def choose(self, *targets: str) -> Model:
        """Declare which next-period states are choices, e.g. ``choose("a(+1)")``.
        Optional: by default every state's next value is chosen."""
        names = []
        for t in targets:
            node = parse(t, [s for s, _ in self._states])
            syms = symbols(node)
            if len(syms) != 1 or next(iter(syms))[1] != 1:
                raise ModelSpecError(f"choose() takes next-period states like 'a(+1)'; got {t!r}")
            names.append(next(iter(syms))[0])
        self._chosen = (self._chosen or []) + names
        return self

    def discrete(self, name: str, values) -> Model:
        """A contemporaneous discrete choice over ``values`` (e.g. labour supply)."""
        if self._discrete is not None:
            raise ModelSpecError("only one discrete choice is supported")
        v = np.asarray(values, dtype=float).ravel()
        if v.size < 1:
            raise ModelSpecError(f"discrete choice {name!r} needs at least one value")
        self._discrete = (str(name), v)
        return self

    def local(self, *definitions: str) -> Model:
        """Named subexpressions, ``"c = w*exp(z) + (1+r)*a - a(+1)"``, usable in any later equation."""
        for d in definitions:
            name, expr = split_assignment(d)
            self._locals.append((name, expr))
        return self

    def reward(self, expr: str) -> Model:
        """Per-period payoff (felicity). ``crra(c, gamma)`` is built in."""
        self._reward = str(expr)
        return self

    def subject_to(self, *constraints: str) -> Model:
        """Feasibility conditions such as ``"c > 0"``; infeasible choices get payoff -inf."""
        self._constraints.extend(str(c) for c in constraints)
        return self

    def taste_shocks(self, scale) -> Model:
        """Type-I extreme value shocks on the discrete choice with scale ``scale``
        (a number or a parameter name): values become log-sums and choices logit.
        Solved by ``method="egm"`` only."""
        if not isinstance(scale, str):
            scale = float(scale)
            if not np.isfinite(scale) or scale < 0:
                raise ModelSpecError(f"taste-shock scale must be >= 0; got {scale}")
        self._taste = scale
        return self

    def discount(self, beta: str = "beta") -> Model:
        """Name of the discount-factor parameter (default ``beta``)."""
        self._beta = str(beta)
        return self

    def horizon(self, T: int, *, survival=None, terminal=None, newborns=None) -> Model:
        """Finite horizon of ``T`` ages (life cycle). ``survival`` is a length-T
        array of conditional survival probabilities, ``terminal`` the
        (n_states, n_shocks) continuation value after the last age, ``newborns``
        the age-0 distribution (default: all mass at the lowest grid point, shocks
        at their stationary distribution)."""
        if int(T) < 1:
            raise ModelSpecError(f"horizon must be >= 1; got {T}")
        self._horizon = int(T)
        self._survival = survival
        self._terminal = terminal
        self._newborns = newborns
        return self

    def aggregate(self, **expressions: str) -> Model:
        """Aggregates as integrals over the stationary distribution, e.g. ``K="a"``.
        An expression in shocks and parameters only is integrated against the
        shocks' stationary distribution."""
        for k, e in expressions.items():
            self._aggregates[k] = str(e)
        return self

    def clear(self, expr: str) -> Model:
        """Market-clearing residual (zero in equilibrium) in aggregates, prices and parameters."""
        self._clear = str(expr)
        return self

    # ------------------------------------------------------------------ compilation
    def check(self) -> _Compiled:
        """Validate the specification and compile it; raises ModelSpecError with the reason."""
        return _Compiled(self)

    def solve(self, method: str = "vfi", *, backend: str = "numpy", tol: float = 1e-8,
              howard: int = 20, max_iter: int = 10_000, distribution: bool = True,
              xtol: float = 1e-6, max_evals: int = 100) -> DPSolution:
        """Compile and solve.

        ``method="vfi"``: discrete VFI with ``howard`` policy-improvement steps
        for infinite horizons, backward induction for a finite horizon.
        ``method="egm"``: the endogenous grid method (continuous policies, no
        grid search) for one-asset models whose budget is
        ``c + a(+1) = R*a + y(z)``; the compiler checks that and raises
        ModelSpecError otherwise. ``tol`` is the EGM consumption tolerance and
        ``howard``/``backend`` do not apply. With prices and a ``clear``
        condition the household solve is wrapped in a brentq root-find (``xtol``).
        """
        if method not in _METHODS:
            raise ModelSpecError(f"method must be one of {_METHODS}; got {method!r}")
        return self.check().solve(method=method, backend=backend, tol=tol, howard=howard, max_iter=max_iter,
                                  distribution=distribution, xtol=xtol, max_evals=max_evals)

    def __repr__(self) -> str:
        return (f"Model({self.name!r}, states={[s for s, _ in self._states]}, "
                f"shocks={[s for s, _ in self._shocks]}, prices={list(self._prices)})")


class _Compiled:
    """A validated model with generated household, aggregate and clearing functions."""

    model: Model
    state_names: list[str]
    state_grids: list[np.ndarray]
    shock_names: list[str]
    finite: bool
    z_grids: list[np.ndarray]
    P: np.ndarray
    z_arg: Any
    beta: float
    param_names: list[str]
    locals: list[tuple[str, Any]]
    local_names: list[str]
    reward: Any
    constraints: list[Any]
    aggregates: dict[str, Any]
    household: Any
    _local_deps: dict[str, set]
    _agg_fns: dict[str, Any]
    _agg_exogenous: dict[str, bool]
    _clear_fn: Any

    def __init__(self, m: Model):
        self.model = m
        if not m._states:
            raise ModelSpecError("declare at least one state with state()")
        if m._reward is None:
            raise ModelSpecError("declare the payoff with reward()")
        self.state_names = [s for s, _ in m._states]
        self.state_grids = [g for _, g in m._states]
        if m._chosen is not None and sorted(m._chosen) != sorted(self.state_names):
            raise ModelSpecError(
                "the discrete VFI engine chooses every state's next value; "
                f"choose() named {m._chosen}, states are {self.state_names}"
            )
        self.shock_names = [s for s, _ in m._shocks]
        self.finite = m._horizon is not None
        names = self.state_names + self.shock_names + ([m._discrete[0]] if m._discrete else [])
        if len(set(names)) != len(names):
            raise ModelSpecError(f"state, shock and choice names must be distinct; got {names}")
        reserved = set(names) | {"age"}
        for k in list(m._params) + list(m._prices) + [n for n, _ in m._locals] + list(m._aggregates):
            if k in reserved:
                raise ModelSpecError(f"name {k!r} is already a state, shock, choice or 'age'")
        clash = set(m._params) & set(m._prices)
        if clash:
            raise ModelSpecError(f"{sorted(clash)} declared as both parameter and price")

        # exogenous chain
        if m._shocks:
            chains = [p.discretize() for _, p in m._shocks]
            self.z_grids = [np.asarray(g, dtype=float) for g, _ in chains]
            P = np.asarray(chains[0][1], dtype=float)
            for _, Pk in chains[1:]:
                P = np.kron(P, np.asarray(Pk, dtype=float))
            self.P = P
        else:
            self.z_grids = []
            self.P = np.ones((1, 1))
        if self.finite and len(self.z_grids) > 1:
            raise ModelSpecError("FiniteHorizonProblem takes one exogenous shock; combine them with dp.Markov")
        self.z_arg = (self.z_grids[0] if len(self.z_grids) == 1
                      else list(self.z_grids) if self.z_grids else np.zeros(1))

        # beta
        if m._beta not in m._params or np.ndim(m._params[m._beta]) != 0:
            raise ModelSpecError(f"discount factor {m._beta!r} must be a scalar parameter")
        self.beta = float(m._params[m._beta])

        # parameters passed to the household function: params then prices
        for k, v in m._params.items():
            if np.ndim(v) and not self.finite:
                raise ModelSpecError(f"parameter {k!r} is an array; age-varying parameters need horizon()")
            if np.ndim(v) and np.shape(v) != (m._horizon,):
                raise ModelSpecError(f"age-varying parameter {k!r} must have length {m._horizon}")
        self.param_names = list(m._params) + list(m._prices)
        if len(m._prices) > 1:
            raise ModelSpecError("phase 1 supports one equilibrium price (scalar brentq)")
        if m._prices and m._clear is None:
            raise ModelSpecError("prices were declared but no clear() condition")
        if m._clear is not None and not m._prices:
            raise ModelSpecError("clear() needs a price declared with prices()")
        if m._taste is not None:
            if m._discrete is None:
                raise ModelSpecError("taste_shocks() needs a discrete choice")
            if isinstance(m._taste, str) and (m._taste not in m._params or np.ndim(m._params[m._taste])):
                raise ModelSpecError(f"taste-shock scale {m._taste!r} must be a scalar parameter")

        # parse
        timed = self.state_names + self.shock_names + ([m._discrete[0]] if m._discrete else [])
        self.locals = [(n, parse(e, timed)) for n, e in m._locals]
        self.local_names = [n for n, _ in self.locals]
        if len(set(self.local_names)) != len(self.local_names):
            raise ModelSpecError(f"local names must be unique; got {self.local_names}")
        self.reward = parse(m._reward, timed)
        if is_relational(self.reward):
            raise ModelSpecError("reward() must be an expression, not a condition")
        self.constraints = [parse(c, timed) for c in m._constraints]
        for c, txt in zip(self.constraints, m._constraints):
            if not is_relational(c):
                raise ModelSpecError(f"constraint {txt!r} must be a comparison such as 'c > 0'")
        self.aggregates = {k: parse(e, timed) for k, e in m._aggregates.items()}

        self._local_deps = self._resolve_local_deps()
        self.household = self._household_function("_dp_reward", self.reward, mask=True)
        self._agg_fns = {k: self._household_function(f"_dp_agg_{k}", n) for k, n in self.aggregates.items()}
        self._agg_exogenous = {k: not (self._deps(n) & self._endogenous_syms()) for k, n in self.aggregates.items()}
        self._clear_fn = self._clear_function() if m._clear is not None else None

    # -------------------------------------------------------------- symbol handling
    def _endogenous_syms(self) -> set:
        out = {(s, 0) for s in self.state_names} | {(s, 1) for s in self.state_names}
        if self.model._discrete:
            out.add((self.model._discrete[0], 0))
        return out

    def _resolve_local_deps(self) -> dict:
        deps: dict[str, set] = {}
        for i, (name, node) in enumerate(self.locals):
            earlier = set(self.local_names[:i])
            acc = set()
            for sym in symbols(node):
                if sym[0] in earlier and sym[1] == 0:
                    acc |= deps[sym[0]]
                else:
                    acc.add(sym)
            deps[name] = acc
        return deps

    def _deps(self, node) -> set:
        acc = set()
        for sym in symbols(node):
            if sym[0] in self._local_deps and sym[1] == 0:
                acc |= self._local_deps[sym[0]]
            else:
                acc.add(sym)
        return acc

    def _resolver(self, where: str, *, allow_aggregates: bool = False):
        m = self.model
        states, shocks = set(self.state_names), set(self.shock_names)
        dname = m._discrete[0] if m._discrete else None
        locals_ = set(self.local_names)
        params = set(self.param_names)

        def resolve(name: str, lead: int) -> str:
            if name in states:
                if lead in (0, 1):
                    return f"{'s' if lead == 0 else 'n'}_{name}"
                raise ModelSpecError(f"{where}: {name}({lead:+d}) — only {name} and {name}(+1) are allowed")
            if lead != 0:
                raise ModelSpecError(f"{where}: {name}({lead:+d}) — only states take a timing index")
            if name in shocks:
                return f"x_{name}"
            if name == dname:
                return f"d_{name}"
            if name in locals_:
                return f"l_{name}"
            if name in params:
                return f"p_{name}"
            if allow_aggregates and name in self.aggregates:
                return f"g_{name}"
            if name == "age" and self.finite:
                return "age"
            raise ModelSpecError(f"{where}: unknown name {name!r} (not a state, shock, choice, local, parameter or price)")

        return resolve

    def _household_args(self) -> list[str]:
        m = self.model
        args = [f"d_{m._discrete[0]}"] if m._discrete else []
        args += [f"n_{s}" for s in self.state_names] + [f"s_{s}" for s in self.state_names]
        args += [f"x_{s}" for s in self.shock_names] if self.shock_names else ["_x_none"]
        if self.finite:
            args.append("age")
        args += [f"p_{p}" for p in self.param_names]
        return args

    def _household_function(self, fname: str, node, *, mask: bool = False):
        body = []
        for k, v in self.model._params.items():
            if np.ndim(v):
                body.append(f"p_{k} = p_{k}[age]")
        for name, lnode in self.locals:
            body.append(f"l_{name} = {emit(lnode, self._resolver(f'local {name!r}'))}")
        expr = emit(node, self._resolver(fname.replace('_dp_', '')))
        if not mask:
            return compile_function(fname, self._household_args(), body, expr)
        body.append(f"_R = {expr}")
        ok = ["(_R == _R)"] + [f"({emit(c, self._resolver('constraint'))})" for c in self.constraints]
        body.append(f"_ok = {' & '.join(ok)}")
        return compile_function(fname, self._household_args(), body, "xp.where(_ok, _R, -_inf)")

    def _clear_function(self):
        scalar_locals = [
            (n, node) for n, node in self.locals
            if not (self._local_deps[n] & (self._endogenous_syms() | {(s, 0) for s in self.shock_names}))
        ]
        args = [f"g_{k}" for k in self.aggregates] + [f"p_{p}" for p in self.param_names]
        resolve = self._resolver("clear", allow_aggregates=True)
        scalar_names = {n for n, _ in scalar_locals}

        def resolve_clear(name, lead):
            if name in self.local_names and name not in scalar_names:
                raise ModelSpecError(f"clear: local {name!r} depends on states or shocks; use an aggregate")
            return resolve(name, lead)

        body = [f"l_{n} = {emit(node, resolve_clear)}" for n, node in scalar_locals]
        node = parse(self.model._clear, [])
        return compile_function("_dp_clear", args, body, emit(node, resolve_clear))

    # ------------------------------------------------------------------ parameters
    def param_values(self, prices: dict | None = None) -> dict:
        vals = {f"p_{k}": v for k, v in self.model._params.items()}
        for k in self.model._prices:
            if prices is None or k not in prices:
                vals[f"p_{k}"] = np.nan
            else:
                vals[f"p_{k}"] = float(prices[k])
        return vals

    @property
    def a_arg(self):
        return self.state_grids[0] if len(self.state_grids) == 1 else list(self.state_grids)

    @property
    def d_grid(self):
        return None if self.model._discrete is None else self.model._discrete[1]

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, fn, sol, params: dict, age: int | None = None) -> np.ndarray:
        """Evaluate a household-signature function at the solved policy, shape (n_a, n_z)."""
        pold = None
        if sol.policy_d is not None:
            pold = sol.policy_d if age is None else sol.policy_d[age]
        shape = tuple(g.size for g in self.state_grids)
        n_a = int(np.prod(shape))
        cur = [c.reshape(n_a, 1) for c in np.meshgrid(*self.state_grids, indexing="ij")]
        if hasattr(sol, "aprime"):          # EGM: continuous next-state values
            pol = np.asarray(sol.aprime if age is None else sol.aprime[age])
            nxt = [pol]
        else:
            pol = sol.policy_aprime if age is None else sol.policy_aprime[age]
            nxt_idx = np.unravel_index(np.asarray(pol), shape)
            nxt = [g[i] for g, i in zip(self.state_grids, nxt_idx)]
        zs: list[np.ndarray] = [np.zeros((1, 1))]
        if self.z_grids:
            n_z = int(np.prod([g.size for g in self.z_grids]))
            zs = [zc.reshape(1, n_z) for zc in np.meshgrid(*self.z_grids, indexing="ij")]

        def call(dv, nxt_):
            args: list[Any] = [dv] if self.model._discrete else []
            args += nxt_ + cur + zs
            if self.finite:
                args.append(int(age or 0))
            args += list(params.values())
            return np.broadcast_to(np.asarray(fn(*args, xp=np), dtype=float), pol.shape)

        probs = getattr(sol, "choice_prob", None)
        if probs is None:
            return call(self.model._discrete[1][pold] if self.model._discrete else None, nxt).copy()
        # taste shocks: average over options with their choice probabilities
        probs = probs if age is None else probs[age]
        aps = sol.aprime_by_choice if age is None else sol.aprime_by_choice[age]
        out = np.zeros(pol.shape)
        for k, dv in enumerate(self.model._discrete[1]):
            with np.errstate(invalid="ignore"):
                out += np.where(probs[k] > 0, probs[k] * call(dv, [aps[k]]), 0.0)
        return out

    def aggregate_values(self, sol, mu, params: dict) -> dict:
        out = {}
        for k, fn in self._agg_fns.items():
            if self._agg_exogenous[k]:
                pi = markov_stationary(self.P)
                v = self.evaluate(fn, sol, params)[0]
                out[k] = float(pi @ v)
            else:
                out[k] = float(np.sum(mu * self.evaluate(fn, sol, params)))
        return out

    # ------------------------------------------------------------------ solving
    def _vfi_problem(self, params: dict, options: dict) -> VFIProblem:
        return VFIProblem(a_grid=self.a_arg, z_grid=self.z_arg, P_z=self.P,
                          return_fn=self.household, beta=self.beta, params=params,
                          d_grid=self.d_grid, options=options)

    def solve(self, *, method="vfi", backend, tol, howard, max_iter, distribution, xtol,
              max_evals) -> DPSolution:
        m = self.model
        if method == "egm":
            return self._solve_egm(tol, max_iter, distribution, xtol, max_evals)
        if m._taste is not None:
            raise ModelSpecError("taste shocks are solved by method='egm' only")
        options = dict(tol=tol, n_howard=int(howard), howard=int(howard) > 0, max_iter=max_iter)
        if self.finite:
            if m._prices:
                raise ModelSpecError("equilibrium with a finite horizon (OLG) is not in phase 1")
            return self._solve_finite(backend)
        if m._prices:
            return self._solve_equilibrium(backend, options, xtol, max_evals)
        params = self.param_values()
        prob = self._vfi_problem(params, options)
        sol = prob.solve(backend)
        mu = prob.stationary_distribution(sol) if distribution else None
        aggs = self.aggregate_values(sol, mu, params) if (mu is not None and self.aggregates) else {}
        return DPSolution(self, sol, params=params, distribution=mu, aggregates=aggs)

    def _solve_egm(self, tol, max_iter, distribution, xtol, max_evals) -> DPSolution:
        from puremacro.dp._egm import EGMPlan

        plan = EGMPlan(self)
        if self.finite:
            if self.model._prices:
                raise ModelSpecError("equilibrium with a finite horizon (OLG) is not supported yet")
            params = self.param_values()
            sol = plan.solve_life_cycle(params)
            return DPSolution(self, sol, params=params, distribution=plan.distribution(sol), method="egm")
        if self.model._prices:
            eq, params, aggs = plan.equilibrium(tol=tol, max_iter=max_iter, xtol=xtol, max_evals=max_evals)
            (pname, _), = self.model._prices.items()
            return DPSolution(self, eq.solution, params=params, distribution=eq.distribution,
                              aggregates=aggs, prices={pname: eq.price}, equilibrium=eq, method="egm")
        params = self.param_values()
        sol = plan.solve_household(params, tol=tol, max_iter=max_iter)
        mu = plan.distribution(sol) if distribution else None
        aggs = self.aggregate_values(sol, mu, params) if (mu is not None and self.aggregates) else {}
        return DPSolution(self, sol, params=params, distribution=mu, aggregates=aggs, method="egm")

    def _solve_equilibrium(self, backend, options, xtol, max_evals) -> DPSolution:
        (pname, (lo, hi)), = self.model._prices.items()

        def build_problem(price):
            return self._vfi_problem(self.param_values({pname: price}), options)

        def market_residual(price, sol, mu, prob):
            params = self.param_values({pname: price})
            aggs = self.aggregate_values(sol, mu, params)
            return float(self._clear_fn(*aggs.values(), *params.values(), xp=np))

        eq = stationary_equilibrium(build_problem, market_residual, (lo, hi), backend=backend,
                                    xtol=xtol, max_evals=max_evals)
        params = self.param_values({pname: eq.price})
        aggs = self.aggregate_values(eq.solution, eq.distribution, params)
        return DPSolution(self, eq.solution, params=params, distribution=eq.distribution,
                          aggregates=aggs, prices={pname: eq.price}, equilibrium=eq)

    def _solve_finite(self, backend) -> DPSolution:
        m = self.model
        params = self.param_values()
        prob = FiniteHorizonProblem(a_grid=self.a_arg, z_grid=self.z_arg, P_z=self.P,
                                    return_fn=self.household, beta=self.beta,
                                    horizon=int(m._horizon or 0), params=params, d_grid=self.d_grid,
                                    terminal_value=m._terminal, survival=m._survival)
        sol = prob.solve(backend)
        dist = life_cycle_distribution(sol, self.P, m._newborns)
        return DPSolution(self, sol, params=params, distribution=dist)


__all__ = ["Model"]
