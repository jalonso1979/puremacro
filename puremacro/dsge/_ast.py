"""Expression DAG Abstract Syntax Tree (AST) for puremacro DSGE models.

Provides immutable, hashable, frozen dataclass nodes representing mathematical
expressions in dynamic macroeconomic models. Supports symbolic differentiation,
algebraic simplification, lead/lag shifting, numerical evaluation, and code
generation under the zero-dependency Pyodide four-package contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Set, Tuple


def _to_node(x: Node | float | int) -> Node:
    """Convert a numeric scalar to a Const node if not already a Node."""
    if isinstance(x, Node):
        return x
    if isinstance(x, (int, float)):
        return Const(x)
    raise TypeError(f"Cannot convert object of type {type(x)} to AST Node: {x!r}")


class Node:
    """Abstract base class for Expression DAG nodes.

    All subclasses are immutable, hashable, frozen dataclasses with slots=True,
    enabling efficient structural equality, hashing, and Common Subexpression
    Elimination (CSE).
    """

    __slots__ = ()

    def variables(self) -> set[tuple[str, int]]:
        """Return all (name, lead) dynamic variable coordinates in this expression."""
        raise NotImplementedError

    def parameters(self) -> set[str]:
        """Return all parameter names referenced in this expression."""
        raise NotImplementedError

    def simplify(self) -> Node:
        """Apply constant folding and canonical algebraic simplification rules."""
        return self

    def diff(self, var_name: str, var_lead: int) -> Node:
        """Symbolic differentiation with respect to dynamic coordinate (var_name, var_lead)."""
        raise NotImplementedError

    def diff_param(self, param_name: str) -> Node:
        """Symbolic differentiation with respect to structural parameter param_name."""
        raise NotImplementedError

    def shift(self, offset: int) -> Node:
        """Shift time lead of all dynamic variables by offset."""
        raise NotImplementedError

    def eval(
        self,
        var_values: Mapping[tuple[str, int] | str, float],
        param_values: Mapping[str, float],
    ) -> float:
        """Numerically evaluate node given variable and parameter values."""
        raise NotImplementedError

    def to_python(
        self,
        lead_ns: str = "lead",
        curr_ns: str = "curr",
        lag_ns: str = "lag",
        shock_ns: str = "shocks",
        param_ns: str = "params",
        shock_names: set[str] | Sequence[str] | None = None,
    ) -> str:
        """Generate executable Python expression string."""
        raise NotImplementedError

    def to_latex(self, symbol_map: Mapping[str, str] | None = None) -> str:
        """Generate publication-grade LaTeX representation."""
        raise NotImplementedError

    # Python operator overloading for ease of AST construction
    def __add__(self, other: Node | float | int) -> Node:
        return BinOp("+", self, _to_node(other))

    def __radd__(self, other: Node | float | int) -> Node:
        return BinOp("+", _to_node(other), self)

    def __sub__(self, other: Node | float | int) -> Node:
        return BinOp("-", self, _to_node(other))

    def __rsub__(self, other: Node | float | int) -> Node:
        return BinOp("-", _to_node(other), self)

    def __mul__(self, other: Node | float | int) -> Node:
        return BinOp("*", self, _to_node(other))

    def __rmul__(self, other: Node | float | int) -> Node:
        return BinOp("*", _to_node(other), self)

    def __truediv__(self, other: Node | float | int) -> Node:
        return BinOp("/", self, _to_node(other))

    def __rtruediv__(self, other: Node | float | int) -> Node:
        return BinOp("/", _to_node(other), self)

    def __pow__(self, other: Node | float | int) -> Node:
        return BinOp("^", self, _to_node(other))

    def __rpow__(self, other: Node | float | int) -> Node:
        return BinOp("^", _to_node(other), self)

    def __neg__(self) -> Node:
        return UnaryOp("-", self)

    def __pos__(self) -> Node:
        return UnaryOp("+", self)


@dataclass(frozen=True, slots=True)
class Const(Node):
    """A numerical constant (integer or floating point)."""

    value: float | int

    def variables(self) -> set[tuple[str, int]]:
        return set()

    def parameters(self) -> set[str]:
        return set()

    def simplify(self) -> Node:
        return self

    def diff(self, var_name: str, var_lead: int) -> Node:
        return Const(0)

    def diff_param(self, param_name: str) -> Node:
        return Const(0)

    def shift(self, offset: int) -> Node:
        return self

    def eval(
        self,
        var_values: Mapping[tuple[str, int] | str, float],
        param_values: Mapping[str, float],
    ) -> float:
        return float(self.value)

    def to_python(
        self,
        lead_ns: str = "lead",
        curr_ns: str = "curr",
        lag_ns: str = "lag",
        shock_ns: str = "shocks",
        param_ns: str = "params",
        shock_names: set[str] | Sequence[str] | None = None,
    ) -> str:
        return str(self.value)

    def to_latex(self, symbol_map: Mapping[str, str] | None = None) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class Param(Node):
    """A model parameter identified by name."""

    name: str

    def variables(self) -> set[tuple[str, int]]:
        return set()

    def parameters(self) -> set[str]:
        return {self.name}

    def simplify(self) -> Node:
        return self

    def diff(self, var_name: str, var_lead: int) -> Node:
        return Const(0)

    def diff_param(self, param_name: str) -> Node:
        if self.name == param_name:
            return Const(1)
        return Const(0)

    def shift(self, offset: int) -> Node:
        return self

    def eval(
        self,
        var_values: Mapping[tuple[str, int] | str, float],
        param_values: Mapping[str, float],
    ) -> float:
        if self.name in param_values:
            return float(param_values[self.name])
        raise KeyError(f"Parameter '{self.name}' not found in param_values")

    def to_python(
        self,
        lead_ns: str = "lead",
        curr_ns: str = "curr",
        lag_ns: str = "lag",
        shock_ns: str = "shocks",
        param_ns: str = "params",
        shock_names: set[str] | Sequence[str] | None = None,
    ) -> str:
        return f"{param_ns}.{self.name}" if param_ns else self.name

    def to_latex(self, symbol_map: Mapping[str, str] | None = None) -> str:
        if symbol_map and self.name in symbol_map:
            return symbol_map[self.name]
        return self.name


@dataclass(frozen=True, slots=True)
class Var(Node):
    """A dynamic endogenous or exogenous model variable coordinate (name, lead)."""

    name: str
    lead: int = 0  # -1 = lag (t-1), 0 = current (t), +1 = lead (t+1)

    def variables(self) -> set[tuple[str, int]]:
        return {(self.name, self.lead)}

    def parameters(self) -> set[str]:
        return set()

    def simplify(self) -> Node:
        return self

    def diff(self, var_name: str, var_lead: int) -> Node:
        if self.name == var_name and self.lead == var_lead:
            return Const(1)
        return Const(0)

    def diff_param(self, param_name: str) -> Node:
        return Const(0)

    def shift(self, offset: int) -> Node:
        return Var(self.name, self.lead + offset)

    def eval(
        self,
        var_values: Mapping[tuple[str, int] | str, float],
        param_values: Mapping[str, float],
    ) -> float:
        coord = (self.name, self.lead)
        if coord in var_values:
            return float(var_values[coord])
        if self.lead == 0 and self.name in var_values:
            return float(var_values[self.name])
        if self.name in var_values and isinstance(var_values[self.name], (int, float)):
            return float(var_values[self.name])
        raise KeyError(
            f"Variable coordinate ({self.name!r}, {self.lead}) not found in var_values"
        )

    def to_python(
        self,
        lead_ns: str = "lead",
        curr_ns: str = "curr",
        lag_ns: str = "lag",
        shock_ns: str = "shocks",
        param_ns: str = "params",
        shock_names: set[str] | Sequence[str] | None = None,
    ) -> str:
        if shock_names and self.name in shock_names:
            return f"{shock_ns}.{self.name}" if shock_ns else self.name
        if self.lead == 1:
            ns = lead_ns
        elif self.lead > 1:
            ns = f"{lead_ns}_{self.lead}"
        elif self.lead == -1:
            ns = lag_ns
        elif self.lead < -1:
            ns = f"{lag_ns}_{abs(self.lead)}"
        else:
            ns = curr_ns
        return f"{ns}.{self.name}" if ns else self.name

    def to_latex(self, symbol_map: Mapping[str, str] | None = None) -> str:
        base = symbol_map.get(self.name, self.name) if symbol_map else self.name
        if self.lead == 0:
            return f"{base}_{{t}}"
        elif self.lead == 1:
            return f"{base}_{{t+1}}"
        elif self.lead > 1:
            return f"{base}_{{t+{self.lead}}}"
        elif self.lead == -1:
            return f"{base}_{{t-1}}"
        else:
            return f"{base}_{{t{self.lead}}}"


@dataclass(frozen=True, slots=True)
class UnaryOp(Node):
    """A unary operator (+, -) applied to an expression."""

    op: str  # '+', '-'
    expr: Node

    def variables(self) -> set[tuple[str, int]]:
        return self.expr.variables()

    def parameters(self) -> set[str]:
        return self.expr.parameters()

    def simplify(self) -> Node:
        s_expr = self.expr.simplify()
        if self.op == "+":
            return s_expr
        if self.op == "-":
            # -(-x) -> x
            if isinstance(s_expr, UnaryOp) and s_expr.op == "-":
                return s_expr.expr
            # -Const(c) -> Const(-c)
            if isinstance(s_expr, Const):
                return Const(-s_expr.value)
            return UnaryOp("-", s_expr)
        return UnaryOp(self.op, s_expr)

    def diff(self, var_name: str, var_lead: int) -> Node:
        d = self.expr.diff(var_name, var_lead)
        if self.op == "+":
            return d.simplify()
        if self.op == "-":
            return UnaryOp("-", d).simplify()
        return UnaryOp(self.op, d).simplify()

    def diff_param(self, param_name: str) -> Node:
        d = self.expr.diff_param(param_name)
        if self.op == "+":
            return d.simplify()
        if self.op == "-":
            return UnaryOp("-", d).simplify()
        return UnaryOp(self.op, d).simplify()

    def shift(self, offset: int) -> Node:
        return UnaryOp(self.op, self.expr.shift(offset))

    def eval(
        self,
        var_values: Mapping[tuple[str, int] | str, float],
        param_values: Mapping[str, float],
    ) -> float:
        val = self.expr.eval(var_values, param_values)
        if self.op == "+":
            return +val
        if self.op == "-":
            return -val
        raise ValueError(f"Unknown unary operator: {self.op}")

    def to_python(
        self,
        lead_ns: str = "lead",
        curr_ns: str = "curr",
        lag_ns: str = "lag",
        shock_ns: str = "shocks",
        param_ns: str = "params",
        shock_names: set[str] | Sequence[str] | None = None,
    ) -> str:
        s = self.expr.to_python(
            lead_ns, curr_ns, lag_ns, shock_ns, param_ns, shock_names
        )
        return f"({self.op}{s})"

    def to_latex(self, symbol_map: Mapping[str, str] | None = None) -> str:
        s = self.expr.to_latex(symbol_map)
        return f"{self.op}{s}"


@dataclass(frozen=True, slots=True)
class BinOp(Node):
    """A binary operator (+, -, *, /, ^, ==, !=, <, <=, >, >=) between two expressions."""

    op: str
    left: Node
    right: Node

    def variables(self) -> set[tuple[str, int]]:
        return self.left.variables() | self.right.variables()

    def parameters(self) -> set[str]:
        return self.left.parameters() | self.right.parameters()

    def simplify(self) -> Node:
        l = self.left.simplify()
        r = self.right.simplify()

        # Constant folding
        if isinstance(l, Const) and isinstance(r, Const):
            lv, rv = l.value, r.value
            if self.op == "+":
                return Const(lv + rv)
            if self.op == "-":
                return Const(lv - rv)
            if self.op == "*":
                return Const(lv * rv)
            if self.op == "/":
                if rv != 0:
                    if isinstance(lv, int) and isinstance(rv, int) and lv % rv == 0:
                        return Const(lv // rv)
                    return Const(lv / rv)
            if self.op == "^":
                try:
                    res = lv**rv
                    if isinstance(res, (int, float)) and not (
                        isinstance(res, float) and math.isnan(res)
                    ):
                        return Const(res)
                except (ValueError, OverflowError, ZeroDivisionError):
                    pass

        # Addition: 0 + x -> x, x + 0 -> x
        if self.op == "+":
            if isinstance(l, Const) and l.value == 0:
                return r
            if isinstance(r, Const) and r.value == 0:
                return l
            # x + (-y) -> x - y
            if isinstance(r, UnaryOp) and r.op == "-":
                return BinOp("-", l, r.expr).simplify()
            # (-x) + y -> y - x
            if isinstance(l, UnaryOp) and l.op == "-":
                return BinOp("-", r, l.expr).simplify()

        # Subtraction: x - 0 -> x, 0 - x -> -x, x - x -> 0
        if self.op == "-":
            if isinstance(r, Const) and r.value == 0:
                return l
            if isinstance(l, Const) and l.value == 0:
                return UnaryOp("-", r).simplify()
            if l == r:
                return Const(0)
            # x - (-y) -> x + y
            if isinstance(r, UnaryOp) and r.op == "-":
                return BinOp("+", l, r.expr).simplify()

        # Multiplication: 0 * x -> 0, x * 0 -> 0, 1 * x -> x, x * 1 -> x, -1 * x -> -x
        if self.op == "*":
            if (isinstance(l, Const) and l.value == 0) or (
                isinstance(r, Const) and r.value == 0
            ):
                return Const(0)
            if isinstance(l, Const) and l.value == 1:
                return r
            if isinstance(r, Const) and r.value == 1:
                return l
            if isinstance(l, Const) and l.value == -1:
                return UnaryOp("-", r).simplify()
            if isinstance(r, Const) and r.value == -1:
                return UnaryOp("-", l).simplify()

        # Division: 0 / x -> 0, x / 1 -> x, x / x -> 1
        if self.op == "/":
            if isinstance(l, Const) and l.value == 0:
                return Const(0)
            if isinstance(r, Const) and r.value == 1:
                return l
            if l == r:
                return Const(1)
            # x / (x ^ 2) -> 1 / x
            if isinstance(r, BinOp) and r.op == "^" and l == r.left:
                exp_minus_1 = BinOp("-", r.right, Const(1)).simplify()
                if isinstance(exp_minus_1, Const) and exp_minus_1.value == 1:
                    return BinOp("/", Const(1), l)
                return BinOp("/", Const(1), BinOp("^", l, exp_minus_1)).simplify()
            # (x ^ n) / x -> x ^ (n - 1)
            if isinstance(l, BinOp) and l.op == "^" and l.left == r:
                exp_minus_1 = BinOp("-", l.right, Const(1)).simplify()
                return BinOp("^", r, exp_minus_1).simplify()

        # Exponentiation: x ^ 0 -> 1, x ^ 1 -> x, 1 ^ x -> 1, 0 ^ x -> 0 (for x > 0)
        if self.op == "^":
            if isinstance(r, Const) and r.value == 0:
                return Const(1)
            if isinstance(r, Const) and r.value == 1:
                return l
            if isinstance(l, Const) and l.value == 1:
                return Const(1)
            if isinstance(l, Const) and l.value == 0:
                return Const(0)

        return BinOp(self.op, l, r)

    def _diff_with_derivatives(self, dl: Node, dr: Node) -> Node:
        if self.op == "+":
            return BinOp("+", dl, dr).simplify()
        if self.op == "-":
            return BinOp("-", dl, dr).simplify()
        if self.op == "*":
            # Product rule: dl * r + l * dr
            term1 = BinOp("*", dl, self.right)
            term2 = BinOp("*", self.left, dr)
            return BinOp("+", term1, term2).simplify()
        if self.op == "/":
            # Quotient rule: if denominator does not vary, dl / r
            dl_is_zero = isinstance(dl, Const) and dl.value == 0
            dr_is_zero = isinstance(dr, Const) and dr.value == 0
            if dl_is_zero and dr_is_zero:
                return Const(0)
            if dr_is_zero:
                return BinOp("/", dl, self.right).simplify()
            if dl_is_zero:
                num = UnaryOp("-", BinOp("*", self.left, dr))
                den = BinOp("^", self.right, Const(2))
                return BinOp("/", num, den).simplify()
            num = BinOp("-", BinOp("*", dl, self.right), BinOp("*", self.left, dr))
            den = BinOp("^", self.right, Const(2))
            return BinOp("/", num, den).simplify()
        if self.op == "^":
            # Power rule: u ^ v
            dr_is_zero = isinstance(dr, Const) and dr.value == 0
            dl_is_zero = isinstance(dl, Const) and dl.value == 0

            if dr_is_zero and dl_is_zero:
                return Const(0)
            if dr_is_zero:
                # v * u^(v - 1) * dl
                exp_minus_1 = BinOp("-", self.right, Const(1)).simplify()
                u_pow = BinOp("^", self.left, exp_minus_1)
                term = BinOp("*", self.right, u_pow)
                return BinOp("*", term, dl).simplify()
            if dl_is_zero:
                # u^v * ln(u) * dr
                term = BinOp(
                    "*", BinOp("^", self.left, self.right), Call("log", (self.left,))
                )
                return BinOp("*", term, dr).simplify()

            # General: u^v * (dr * ln(u) + v * dl / u)
            log_u = Call("log", (self.left,))
            t1 = BinOp("*", dr, log_u)
            t2 = BinOp("/", BinOp("*", self.right, dl), self.left)
            inner = BinOp("+", t1, t2)
            return BinOp("*", BinOp("^", self.left, self.right), inner).simplify()

        # Comparisons and equality relations have zero derivative
        return Const(0)

    def diff(self, var_name: str, var_lead: int) -> Node:
        dl = self.left.diff(var_name, var_lead)
        dr = self.right.diff(var_name, var_lead)
        return self._diff_with_derivatives(dl, dr)

    def diff_param(self, param_name: str) -> Node:
        dl = self.left.diff_param(param_name)
        dr = self.right.diff_param(param_name)
        return self._diff_with_derivatives(dl, dr)

    def shift(self, offset: int) -> Node:
        return BinOp(self.op, self.left.shift(offset), self.right.shift(offset))

    def eval(
        self,
        var_values: Mapping[tuple[str, int] | str, float],
        param_values: Mapping[str, float],
    ) -> float:
        lv = self.left.eval(var_values, param_values)
        rv = self.right.eval(var_values, param_values)

        if self.op == "+":
            return lv + rv
        if self.op == "-":
            return lv - rv
        if self.op == "*":
            return lv * rv
        if self.op == "/":
            return lv / rv
        if self.op == "^":
            return lv**rv
        if self.op == "==":
            return 1.0 if lv == rv else 0.0
        if self.op == "!=":
            return 1.0 if lv != rv else 0.0
        if self.op == "<":
            return 1.0 if lv < rv else 0.0
        if self.op == "<=":
            return 1.0 if lv <= rv else 0.0
        if self.op == ">":
            return 1.0 if lv > rv else 0.0
        if self.op == ">=":
            return 1.0 if lv >= rv else 0.0
        raise ValueError(f"Unknown binary operator: {self.op}")

    def to_python(
        self,
        lead_ns: str = "lead",
        curr_ns: str = "curr",
        lag_ns: str = "lag",
        shock_ns: str = "shocks",
        param_ns: str = "params",
        shock_names: set[str] | Sequence[str] | None = None,
    ) -> str:
        l_str = self.left.to_python(
            lead_ns, curr_ns, lag_ns, shock_ns, param_ns, shock_names
        )
        r_str = self.right.to_python(
            lead_ns, curr_ns, lag_ns, shock_ns, param_ns, shock_names
        )
        py_op = "**" if self.op == "^" else self.op
        return f"({l_str} {py_op} {r_str})"

    def to_latex(self, symbol_map: Mapping[str, str] | None = None) -> str:
        l_str = self.left.to_latex(symbol_map)
        r_str = self.right.to_latex(symbol_map)
        if self.op == "+":
            return f"{l_str} + {r_str}"
        if self.op == "-":
            return f"{l_str} - {r_str}"
        if self.op == "*":
            return f"{l_str} \\cdot {r_str}"
        if self.op == "/":
            return f"\\frac{{{l_str}}}{{{r_str}}}"
        if self.op == "^":
            return f"{{{l_str}}}^{{{r_str}}}"
        if self.op == "==":
            return f"{l_str} = {r_str}"
        return f"{l_str} {self.op} {r_str}"


@dataclass(frozen=True, slots=True)
class Call(Node):
    """A function call with arguments (exp, log, sin, normcdf, etc.)."""

    func: str
    args: tuple[Node, ...]

    def variables(self) -> set[tuple[str, int]]:
        res: set[tuple[str, int]] = set()
        for a in self.args:
            res |= a.variables()
        return res

    def parameters(self) -> set[str]:
        res: set[str] = set()
        for a in self.args:
            res |= a.parameters()
        return res

    def simplify(self) -> Node:
        s_args = tuple(a.simplify() for a in self.args)

        # Constant folding for unary math functions if argument is Const
        if len(s_args) == 1 and isinstance(s_args[0], Const):
            v = s_args[0].value
            fn = self.func.lower()
            try:
                if fn == "exp":
                    return Const(math.exp(v))
                if fn in ("log", "ln"):
                    if v > 0:
                        return Const(math.log(v))
                if fn == "log10":
                    if v > 0:
                        return Const(math.log10(v))
                if fn == "sqrt":
                    if v >= 0:
                        return Const(math.sqrt(v))
                if fn == "sin":
                    return Const(math.sin(v))
                if fn == "cos":
                    return Const(math.cos(v))
                if fn == "tan":
                    return Const(math.tan(v))
                if fn == "abs":
                    return Const(abs(v))
                if fn == "sign":
                    return Const(1.0 if v > 0 else (-1.0 if v < 0 else 0.0))
                if fn == "erf":
                    return Const(math.erf(v))
                if fn == "erfc":
                    return Const(math.erfc(v))
                if fn == "normpdf":
                    return Const(math.exp(-0.5 * v * v) / math.sqrt(2.0 * math.pi))
                if fn == "normcdf":
                    return Const(0.5 * (1.0 + math.erf(v / math.sqrt(2.0))))
                if fn == "steady_state":
                    return Const(v)
            except (ValueError, OverflowError):
                pass

        return Call(self.func, s_args)

    def _diff_fn(self, du: Node) -> Node:
        fn = self.func.lower()
        if isinstance(du, Const) and du.value == 0:
            return Const(0)

        u = self.args[0]
        if fn == "exp":
            return BinOp("*", Call("exp", (u,)), du).simplify()
        if fn in ("log", "ln"):
            return BinOp("/", du, u).simplify()
        if fn == "log10":
            den = BinOp("*", u, Const(math.log(10.0)))
            return BinOp("/", du, den).simplify()
        if fn == "sqrt":
            den = BinOp("*", Const(2), Call("sqrt", (u,)))
            return BinOp("/", du, den).simplify()
        if fn == "cbrt":
            den = BinOp("*", Const(3), BinOp("^", u, Const(2.0 / 3.0)))
            return BinOp("/", du, den).simplify()
        if fn == "sin":
            return BinOp("*", Call("cos", (u,)), du).simplify()
        if fn == "cos":
            return BinOp("*", UnaryOp("-", Call("sin", (u,))), du).simplify()
        if fn == "tan":
            sec2 = BinOp("+", Const(1), BinOp("^", Call("tan", (u,)), Const(2)))
            return BinOp("*", sec2, du).simplify()
        if fn == "asin":
            den = Call("sqrt", (BinOp("-", Const(1), BinOp("^", u, Const(2))),))
            return BinOp("/", du, den).simplify()
        if fn == "acos":
            den = Call("sqrt", (BinOp("-", Const(1), BinOp("^", u, Const(2))),))
            return BinOp("/", UnaryOp("-", du), den).simplify()
        if fn == "atan":
            den = BinOp("+", Const(1), BinOp("^", u, Const(2)))
            return BinOp("/", du, den).simplify()
        if fn == "sinh":
            return BinOp("*", Call("cosh", (u,)), du).simplify()
        if fn == "cosh":
            return BinOp("*", Call("sinh", (u,)), du).simplify()
        if fn == "tanh":
            sech2 = BinOp("-", Const(1), BinOp("^", Call("tanh", (u,)), Const(2)))
            return BinOp("*", sech2, du).simplify()
        if fn == "normpdf":
            # d/du normpdf(u) = -u * normpdf(u) * du
            deriv = BinOp("*", UnaryOp("-", u), Call("normpdf", (u,)))
            return BinOp("*", deriv, du).simplify()
        if fn == "normcdf":
            # d/du normcdf(u) = normpdf(u) * du
            return BinOp("*", Call("normpdf", (u,)), du).simplify()
        if fn == "erf":
            # 2 / sqrt(pi) * exp(-u^2) * du
            pre = Const(2.0 / math.sqrt(math.pi))
            core = Call("exp", (UnaryOp("-", BinOp("^", u, Const(2))),))
            return BinOp("*", BinOp("*", pre, core), du).simplify()
        if fn == "erfc":
            pre = Const(-2.0 / math.sqrt(math.pi))
            core = Call("exp", (UnaryOp("-", BinOp("^", u, Const(2))),))
            return BinOp("*", BinOp("*", pre, core), du).simplify()
        if fn == "abs":
            return BinOp("*", Call("sign", (u,)), du).simplify()
        if fn == "sign":
            return Const(0)

        raise NotImplementedError(
            f"Symbolic derivative not implemented for function: {self.func!r}"
        )

    def diff(self, var_name: str, var_lead: int) -> Node:
        fn = self.func.lower()

        # Steady state operator has zero dynamic derivative by definition
        if fn == "steady_state":
            return Const(0)

        # Expectation operator: conditional expectation at current info set
        if fn == "expectation":
            if len(self.args) == 2:
                return self.args[1].diff(var_name, var_lead)
            if len(self.args) == 1:
                return self.args[0].diff(var_name, var_lead)

        if len(self.args) == 1:
            du = self.args[0].diff(var_name, var_lead)
            return self._diff_fn(du)

        raise NotImplementedError(
            f"Symbolic derivative not implemented for function: {self.func!r}"
        )

    def diff_param(self, param_name: str) -> Node:
        fn = self.func.lower()

        # Steady state operator: static derivative inside steady_state(expr)
        if fn == "steady_state":
            return self.args[0].diff_param(param_name)

        # Expectation operator
        if fn == "expectation":
            if len(self.args) == 2:
                return self.args[1].diff_param(param_name)
            if len(self.args) == 1:
                return self.args[0].diff_param(param_name)

        if len(self.args) == 1:
            du = self.args[0].diff_param(param_name)
            return self._diff_fn(du)

        raise NotImplementedError(
            f"Symbolic derivative not implemented for function: {self.func!r}"
        )

    def shift(self, offset: int) -> Node:
        return Call(self.func, tuple(a.shift(offset) for a in self.args))

    def eval(
        self,
        var_values: Mapping[tuple[str, int] | str, float],
        param_values: Mapping[str, float],
    ) -> float:
        fn = self.func.lower()
        if fn == "steady_state":
            # For STEADY_STATE(expr), evaluate with coordinates mapped to lead=0 (steady-state values)
            ss_var_values = {}
            for k, v in var_values.items():
                if isinstance(k, tuple):
                    ss_var_values[(k[0], 0)] = v
                    ss_var_values[k[0]] = v
                else:
                    ss_var_values[k] = v
                    ss_var_values[(k, 0)] = v
            return self.args[0].eval(ss_var_values, param_values)

        arg_vals = [a.eval(var_values, param_values) for a in self.args]

        if fn == "exp":
            return math.exp(arg_vals[0])
        if fn in ("log", "ln"):
            return math.log(arg_vals[0])
        if fn == "log10":
            return math.log10(arg_vals[0])
        if fn == "sqrt":
            return math.sqrt(arg_vals[0])
        if fn == "cbrt":
            return math.pow(arg_vals[0], 1.0 / 3.0)
        if fn == "sin":
            return math.sin(arg_vals[0])
        if fn == "cos":
            return math.cos(arg_vals[0])
        if fn == "tan":
            return math.tan(arg_vals[0])
        if fn == "asin":
            return math.asin(arg_vals[0])
        if fn == "acos":
            return math.acos(arg_vals[0])
        if fn == "atan":
            return math.atan(arg_vals[0])
        if fn == "sinh":
            return math.sinh(arg_vals[0])
        if fn == "cosh":
            return math.cosh(arg_vals[0])
        if fn == "tanh":
            return math.tanh(arg_vals[0])
        if fn == "erf":
            return math.erf(arg_vals[0])
        if fn == "erfc":
            return math.erfc(arg_vals[0])
        if fn == "abs":
            return abs(arg_vals[0])
        if fn == "sign":
            x = arg_vals[0]
            return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        if fn == "normpdf":
            if len(arg_vals) == 1:
                x = arg_vals[0]
                return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
            x, mu, sig = arg_vals
            return math.exp(-0.5 * ((x - mu) / sig) ** 2) / (
                sig * math.sqrt(2.0 * math.pi)
            )
        if fn == "normcdf":
            if len(arg_vals) == 1:
                x = arg_vals[0]
                return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            x, mu, sig = arg_vals
            return 0.5 * (1.0 + math.erf((x - mu) / (sig * math.sqrt(2.0))))
        if fn == "max":
            return max(arg_vals)
        if fn == "min":
            return min(arg_vals)
        if fn == "expectation":
            return arg_vals[-1]

        raise ValueError(f"Unknown function in eval: {self.func!r}")

    def to_python(
        self,
        lead_ns: str = "lead",
        curr_ns: str = "curr",
        lag_ns: str = "lag",
        shock_ns: str = "shocks",
        param_ns: str = "params",
        shock_names: set[str] | Sequence[str] | None = None,
    ) -> str:
        fn = self.func.lower()
        args_py = [
            a.to_python(lead_ns, curr_ns, lag_ns, shock_ns, param_ns, shock_names)
            for a in self.args
        ]
        joined = ", ".join(args_py)

        if fn in (
            "exp",
            "log",
            "sqrt",
            "sin",
            "cos",
            "tan",
            "sinh",
            "cosh",
            "tanh",
            "abs",
            "sign",
        ):
            return f"np.{fn}({joined})"
        if fn == "ln":
            return f"np.log({joined})"
        if fn == "max":
            return f"np.maximum({joined})"
        if fn == "min":
            return f"np.minimum({joined})"
        if fn == "normcdf":
            # Pure numpy implementation of standard normal CDF via erf
            return f"(0.5 * (1.0 + scipy.special.erf(({args_py[0]}) / 1.4142135623730951)))"
        if fn == "normpdf":
            return f"(np.exp(-0.5 * ({args_py[0]})**2) / 2.5066282746310002)"
        if fn == "erf":
            return f"scipy.special.erf({joined})"
        if fn == "erfc":
            return f"scipy.special.erfc({joined})"
        if fn == "steady_state":
            return args_py[0]

        return f"{self.func}({joined})"

    def to_latex(self, symbol_map: Mapping[str, str] | None = None) -> str:
        fn = self.func.lower()
        args_ltx = [a.to_latex(symbol_map) for a in self.args]
        if fn == "exp":
            return f"\\exp\\left({args_ltx[0]}\\right)"
        if fn in ("log", "ln"):
            return f"\\ln\\left({args_ltx[0]}\\right)"
        if fn == "sqrt":
            return f"\\sqrt{{{args_ltx[0]}}}"
        if fn == "sin":
            return f"\\sin\\left({args_ltx[0]}\\right)"
        if fn == "cos":
            return f"\\cos\\left({args_ltx[0]}\\right)"
        if fn == "normcdf":
            return f"\\Phi\\left({args_ltx[0]}\\right)"
        if fn == "normpdf":
            return f"\\phi\\left({args_ltx[0]}\\right)"
        if fn == "abs":
            return f"\\left|{args_ltx[0]}\\right|"
        return f"\\text{{{self.func}}}\\left({', '.join(args_ltx)}\\right)"
