"""Felicity u(c[, d]) compiled from a symbolic reward, with u' and its inverse.

Shared by the EGM and HJB compilers: both invert marginal utility. The
derivative is symbolic (``_expr.diff``); the inverse is closed form for
``crra(c, gamma)`` and ``log(c)`` (also inside an additively separable reward such
as ``crra(c, gamma) - chi*h``) and found by bisection on log c otherwise.
"""
from __future__ import annotations

import numpy as np

from puremacro.dp._expr import ModelSpecError, compile_function, diff, emit, symbols
from puremacro.dsge._ast import BinOp, Call, Param


class Felicity:
    """Compiled ``u(c, d, *params)`` for a reward ``u_node`` in consumption ``cname``.

    ``dname`` is the discrete choice the reward may also depend on (or None);
    every other symbol must be a parameter in ``param_names``. ``label`` names the
    solver in error messages.
    """

    def __init__(self, u_node, cname: str, dname: str | None, param_names, label: str):
        self.cname, self.dname, self.label = cname, dname, label
        allowed = {(p, 0) for p in param_names} | ({(dname, 0)} if dname else set())
        bad = symbols(u_node) - allowed - {(cname, 0)}
        if bad:
            raise ModelSpecError(
                f"{label} needs a reward that depends on choices only through {cname!r} "
                f"and otherwise on parameters; it also uses {sorted(n for n, _ in bad)}")

        def resolve(name, lead):
            return "c" if name == cname else "dv" if name == dname else f"p_{name}"

        args = ["c", "dv"] + [f"p_{p}" for p in param_names]
        self._u = compile_function("_dp_u", args, [], emit(u_node, resolve))
        self._u_prime = compile_function("_dp_u_prime", args, [], emit(diff(u_node, cname), resolve))
        self._inv_kind, self._inv_fn = "numeric", None
        core = self._c_part(u_node)
        if isinstance(core, Call) and core.args and core.args[0] == Param(cname):
            fn = core.func.lower()
            if fn == "crra" and not ({(cname, 0), (dname, 0)} & symbols(core.args[1])):
                self._inv_kind = "crra"
                self._inv_fn = compile_function("_dp_gamma", args, [], emit(core.args[1], resolve))
            elif fn in ("log", "ln"):
                self._inv_kind = "log"

    def _c_part(self, node):
        """The term of an additively separable reward that holds consumption, if any."""
        def has(n):
            return (self.cname, 0) in symbols(n)

        if isinstance(node, BinOp) and node.op in ("+", "-"):
            if not has(node.right):
                return self._c_part(node.left)
            if node.op == "+" and not has(node.left):
                return self._c_part(node.right)
        return node

    def funcs(self, params: dict, c_max: float, d: float = 0.0):
        """(u, u', u'^-1) at scalar parameter values and discrete choice ``d``.

        Raises ModelSpecError unless u' is positive and decreasing on
        [1e-3 * min(1, c_max), c_max].
        """
        pv = list(params.values())

        def u(c):
            with np.errstate(all="ignore"):
                return np.broadcast_to(self._u(c, d, *pv, xp=np), np.shape(c))

        def u_prime(c):
            with np.errstate(all="ignore"):
                return np.broadcast_to(self._u_prime(c, d, *pv, xp=np), np.shape(c))

        if self._inv_kind == "crra":
            g = float(self._inv_fn(0.0, d, *pv, xp=np))

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
            raise ModelSpecError(f"{self.label} needs a reward that is increasing and strictly concave in "
                                 f"{self.cname!r} (marginal utility positive and decreasing)")
        return u, u_prime, u_prime_inv
