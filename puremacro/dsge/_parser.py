"""Hand-written pure-Python lexer and recursive-descent parser for Dynare .mod files.

Replaces legacy regular expression substitutions with an exact Expression DAG
parser supporting:
- Variable and parameter declarations (var, varexo, parameters, predetermined_variables, varobs)
- Extended math functions (exp, log, sqrt, sin, cos, tan, normcdf, normpdf, erf, abs, sign)
- Lead/lag indexing (x(+1), x(-1), x(0), x(+k), x(-k), x(k))
- Model-local '#' variables with topological sort and cycle detection
- Multi-period lead/lag companion form auxiliary variable expansion (|lead| >= 2)
- Equation tags ([name='...'])
- Shocks blocks and parameter assignments
- Zero substring collisions and zero external dependencies
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Set, Tuple

import numpy as np

from puremacro.dsge._ast import BinOp, Call, Const, Node, Param, UnaryOp, Var, _py_attr
from puremacro.dsge.build import ModelError
from puremacro.dsge._estimated_params import (
    parse_estimated_params,
    parse_estimated_params_bounds,
    parse_estimated_params_init,
)

# Optional import of macro preprocessor (Worker M1)
preprocess_macro: Any
try:
    from puremacro.dsge._macro import preprocess_macro  # type: ignore
except ImportError:
    preprocess_macro = None


class DynareParseError(Exception):
    """Raised when a syntax or grammatical error occurs while parsing a Dynare model."""

    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.message = message
        self.line = line
        self.col = col
        super().__init__(f"Line {line}, Col {col}: {message}" if line > 0 else message)


class DynareFeatureError(NotImplementedError):
    """A .mod construct puremacro recognises but does not implement yet."""


class _ShockValuesParseError(DynareParseError, ValueError):
    """Located dated-value syntax error that cannot enter the legacy fallback."""


@dataclass(frozen=True, slots=True)
class Token:
    """A lexical token."""

    type: str
    value: Any
    line: int
    col: int


@dataclass(frozen=True, slots=True)
class _LocalRef(Node):
    """Internal placeholder for references to model-local '#' variables before inlining."""

    name: str
    lead: int = 0

    def variables(self) -> set[tuple[str, int]]:
        return set()

    def parameters(self) -> set[str]:
        return set()

    def simplify(self) -> Node:
        return self

    def diff(self, var_name: str, var_lead: int) -> Node:
        raise RuntimeError(
            f"Cannot differentiate unresolved local variable reference '{self.name}'"
        )

    def shift(self, offset: int) -> Node:
        return _LocalRef(self.name, self.lead + offset)

    def eval(self, var_values: Mapping, param_values: Mapping) -> float:
        raise RuntimeError(
            f"Cannot evaluate unresolved local variable reference '{self.name}'"
        )

    def to_python(self, *args: Any, **kwargs: Any) -> str:
        return self.name

    def to_latex(self, *args: Any, **kwargs: Any) -> str:
        return self.name


MATH_FUNCTIONS = {
    "exp",
    "log",
    "ln",
    "log10",
    "sqrt",
    "cbrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "erf",
    "erfc",
    "gamma",
    "lgamma",
    "normpdf",
    "normcdf",
    "abs",
    "sign",
    "max",
    "min",
    "steady_state",
    "expectation",
    "diff",
}

KEYWORDS = {
    "var",
    "varexo",
    "parameters",
    "predetermined_variables",
    "varobs",
    "model",
    "initval",
    "steady",
    "steady_state_model",
    "shocks",
    "estimated_params",
    "estimated_params_init",
    "estimated_params_bounds",
    "stoch_simul",
    "end",
    "linear",
    "stderr",
    "corr",
    "histval",
    "endval",
    "varexo_det",
    "trend_var",
    "log_trend_var",
    "deflator",
    "log_deflator",
    "shock_groups",
    "hetagent_block",
    "periods",
    "values",
    "markov_switching",
    "planner_objective",
    "ramsey_policy",
    "ramsey_model",
    "optim_weights",
    "osr_params",
    "osr_params_bounds",
    "osr",
    "calib_smoother",
    "shock_decomposition",
    "perfect_foresight_setup",
    "perfect_foresight_solver",
    "simul",
}



class Tokenizer:
    """Pure-Python character-by-character tokenizer for Dynare .mod source text."""

    def __init__(self, text: str, *, strict: bool = False):
        self.strict = strict
        self.comments: list[dict[str, Any]] = []
        self.text = text
        self.n = len(text)
        self.pos = 0
        self.line = 1
        self.col = 1

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < self.n:
            return self.text[idx]
        return ""

    def _advance(self) -> str:
        if self.pos >= self.n:
            return ""
        ch = self.text[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while self.pos < self.n:
            ch = self._peek()

            # Whitespace
            if ch.isspace():
                self._advance()
                continue

            # Line comments: // or %
            if (ch == "/" and self._peek(1) == "/") or ch == "%":
                comment_start = (self.line, self.col)
                while self.pos < self.n and self._peek() != "\n":
                    self._advance()
                self.comments.append(dict(name="comment", category="comment", line=comment_start[0],
                    col=comment_start[1], end_line=self.line, end_col=self.col))
                continue

            # Block comments: /* ... */
            if ch == "/" and self._peek(1) == "*":
                comment_start = (self.line, self.col)
                self._advance()  # /
                self._advance()  # *
                while self.pos < self.n and not (
                    self._peek() == "*" and self._peek(1) == "/"
                ):
                    self._advance()
                if self.pos < self.n:
                    self._advance()  # *
                    self._advance()  # /
                elif self.strict:
                    raise DynareParseError("Unclosed block comment", *comment_start)
                self.comments.append(dict(name="comment", category="comment", line=comment_start[0],
                    col=comment_start[1], end_line=self.line, end_col=self.col))
                continue

            start_line = self.line
            start_col = self.col

            # TeX label: ${...}$ or $...$
            if ch == "$":
                self._advance()  # $
                is_braced = self._peek() == "{"
                if is_braced:
                    self._advance()  # {
                buf = []
                while self.pos < self.n:
                    curr = self._peek()
                    if is_braced and curr == "}" and self._peek(1) == "$":
                        self._advance()  # }
                        self._advance()  # $
                        break
                    elif not is_braced and curr == "$":
                        self._advance()  # $
                        break
                    buf.append(self._advance())
                tokens.append(Token("TEX", "".join(buf), start_line, start_col))
                continue

            # String literal: "..." or '...'
            if ch in ('"', "'"):
                quote_char = self._advance()
                closed = False
                buf = []
                while self.pos < self.n:
                    c = self._advance()
                    if c == quote_char:
                        closed = True
                        break
                    if c == "\\" and self.pos < self.n:
                        buf.append(self._advance())
                    else:
                        buf.append(c)
                if self.strict and not closed:
                    raise DynareParseError("Unclosed quoted string", start_line, start_col)
                tokens.append(Token("STRING", "".join(buf), start_line, start_col))
                continue

            # Numbers: integers, floats, scientific notation, leading dot (.025)
            if ch.isdigit() or (ch == "." and self._peek(1).isdigit()):
                buf = []
                has_dot = False
                has_exp = False
                if ch == ".":
                    has_dot = True
                    buf.append(self._advance())
                while self.pos < self.n:
                    c = self._peek()
                    if c.isdigit():
                        buf.append(self._advance())
                    elif c == "." and not has_dot and not has_exp:
                        has_dot = True
                        buf.append(self._advance())
                    elif c in ("e", "E") and not has_exp:
                        has_exp = True
                        buf.append(self._advance())
                        if self._peek() in ("+", "-"):
                            buf.append(self._advance())
                    else:
                        break
                num_str = "".join(buf)
                if has_dot or has_exp:
                    val = float(num_str)
                else:
                    val = int(num_str)
                tokens.append(Token("NUMBER", val, start_line, start_col))
                continue

            # Identifiers and keywords: [A-Za-z_][A-Za-z0-9_]*
            if ch.isalpha() or ch == "_":
                buf = []
                while self.pos < self.n and (
                    self._peek().isalnum() or self._peek() == "_"
                ):
                    buf.append(self._advance())
                ident = "".join(buf)
                ident_lower = ident.lower()
                if ident_lower in KEYWORDS:
                    tokens.append(Token("KEYWORD", ident_lower, start_line, start_col))
                else:
                    tokens.append(Token("IDENT", ident, start_line, start_col))
                continue

            # Multi-character comparison operators
            if ch == "=" and self._peek(1) == "=":
                self._advance()
                self._advance()
                tokens.append(Token("EQUAL", "==", start_line, start_col))
                continue
            if ch == "!" and self._peek(1) == "=":
                self._advance()
                self._advance()
                tokens.append(Token("NOT_EQUAL", "!=", start_line, start_col))
                continue
            if ch == "<" and self._peek(1) == "=":
                self._advance()
                self._advance()
                tokens.append(Token("LESS_EQUAL", "<=", start_line, start_col))
                continue
            if ch == ">" and self._peek(1) == "=":
                self._advance()
                self._advance()
                tokens.append(Token("GREATER_EQUAL", ">=", start_line, start_col))
                continue

            # Single-character tokens
            ch = self._advance()
            if ch == "+":
                tokens.append(Token("PLUS", "+", start_line, start_col))
            elif ch == "-":
                tokens.append(Token("MINUS", "-", start_line, start_col))
            elif ch == "*":
                tokens.append(Token("STAR", "*", start_line, start_col))
            elif ch == "/":
                tokens.append(Token("SLASH", "/", start_line, start_col))
            elif ch == "^":
                tokens.append(Token("CARET", "^", start_line, start_col))
            elif ch == "=":
                tokens.append(Token("ASSIGN", "=", start_line, start_col))
            elif ch == "<":
                tokens.append(Token("LESS", "<", start_line, start_col))
            elif ch == ">":
                tokens.append(Token("GREATER", ">", start_line, start_col))
            elif ch == "(":
                tokens.append(Token("LPAREN", "(", start_line, start_col))
            elif ch == ")":
                tokens.append(Token("RPAREN", ")", start_line, start_col))
            elif ch == "[":
                tokens.append(Token("LBRACKET", "[", start_line, start_col))
            elif ch == "]":
                tokens.append(Token("RBRACKET", "]", start_line, start_col))
            elif ch == "{":
                tokens.append(Token("LBRACE", "{", start_line, start_col))
            elif ch == "}":
                tokens.append(Token("RBRACE", "}", start_line, start_col))
            elif ch == ";":
                tokens.append(Token("SEMI", ";", start_line, start_col))
            elif ch == ",":
                tokens.append(Token("COMMA", ",", start_line, start_col))
            elif ch == ":":
                tokens.append(Token("COLON", ":", start_line, start_col))
            elif ch == ".":
                tokens.append(Token("DOT", ".", start_line, start_col))
            elif ch == "#":
                tokens.append(Token("HASH", "#", start_line, start_col))
            elif ch == "@":
                tokens.append(Token("AT", "@", start_line, start_col))
            elif ch == "~":
                tokens.append(Token("TILDE", "~", start_line, start_col))
            elif ch == "!":
                tokens.append(Token("EXCL", "!", start_line, start_col))
            elif ch == "&":
                tokens.append(Token("AMP", "&", start_line, start_col))
            elif ch == "|":
                tokens.append(Token("BAR", "|", start_line, start_col))
            elif ch == "\\":
                tokens.append(Token("BACKSLASH", "\\", start_line, start_col))
            else:
                raise DynareParseError(
                    f"Unexpected character {ch!r}", start_line, start_col
                )

        tokens.append(Token("EOF", None, self.line, self.col))
        return tokens


def _substitute_node(node: Node, subst_fn: Callable[[Node], Node | None]) -> Node:
    """Recursively replace nodes in an AST using subst_fn."""
    custom = subst_fn(node)
    if custom is not None:
        return custom

    if isinstance(node, (Const, Param, Var)):
        return node
    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, _substitute_node(node.expr, subst_fn))
    if isinstance(node, BinOp):
        return BinOp(
            node.op,
            _substitute_node(node.left, subst_fn),
            _substitute_node(node.right, subst_fn),
        )
    if isinstance(node, Call):
        return Call(node.func, tuple(_substitute_node(a, subst_fn) for a in node.args))
    if isinstance(node, _LocalRef):
        return node
    return node


def _substitute_vars(node: Node, replacements: dict[tuple[str, int], Node]) -> Node:
    """Replace Var(name, lead) occurrences matching replacements mapping."""

    def repl(n: Node) -> Node | None:
        if isinstance(n, Var) and (n.name, n.lead) in replacements:
            return replacements[(n.name, n.lead)]
        return None

    return _substitute_node(node, repl)


def _substitute_locals(node: Node, resolved_locals: dict[str, Node]) -> Node:
    """Replace _LocalRef occurrences with inlined resolved local AST nodes shifted appropriately."""

    def repl(n: Node) -> Node | None:
        if isinstance(n, _LocalRef) and n.name in resolved_locals:
            target = resolved_locals[n.name]
            if n.lead != 0:
                target = target.shift(n.lead)
            # Recursively inline in target in case target had further refs
            return _substitute_locals(target, resolved_locals)
        return None

    return _substitute_node(node, repl)


def _find_local_deps(node: Node, local_names: set[str]) -> set[str]:
    """Find all names in local_names referenced by node."""
    deps: set[str] = set()

    def visit(n: Node) -> None:
        if isinstance(n, _LocalRef) and n.name in local_names:
            deps.add(n.name)
        elif isinstance(n, (Param, Var)) and n.name in local_names:
            deps.add(n.name)
        elif isinstance(n, UnaryOp):
            visit(n.expr)
        elif isinstance(n, BinOp):
            visit(n.left)
            visit(n.right)
        elif isinstance(n, Call):
            for a in n.args:
                visit(a)

    visit(node)
    return deps


def _eval_steady_state_defs(
    defs: Mapping[str, Node],
    param_values: Mapping[str, float],
    variables: Sequence[str],
    auxiliary: Mapping[str, Mapping[str, Any]],
    local_variables: Mapping[str, Node] | None = None,
) -> dict[str, float]:
    """Evaluate a ``steady_state_model`` block in order at ``param_values``.

    Assignments run top to bottom, so helper quantities defined early
    (``rk = 1/beta - 1 + delta;``) are visible to later lines, exactly as in
    Dynare's generated ``steadystate`` function. A variable the block never
    assigns gets ``0.0``; a lead/lag auxiliary variable inherits the value of
    the variable it shifts. An assignment that cannot be evaluated raises
    ``ValueError`` naming it.
    """
    ss_scope: dict[tuple[str, int] | str, float] = {}
    ss_scope.update(param_values)
    if local_variables:
        for loc_name, loc_node in local_variables.items():
            if not loc_node.variables():
                try:
                    p_scope = {k: v for k, v in ss_scope.items() if isinstance(k, str)}
                    ss_scope[loc_name] = float(loc_node.eval(ss_scope, p_scope))
                except Exception:
                    pass
    for s_name, s_expr in defs.items():
        try:
            p_scope = {**param_values, **{k: v for k, v in ss_scope.items() if isinstance(k, str)}}
            val = s_expr.eval(ss_scope, p_scope)
            ss_scope[s_name] = float(val)
        except Exception as exc:
            raise ValueError(
                f"could not evaluate '{s_name}' in the steady_state_model block: {exc}"
            ) from exc
    steady_state = {v: ss_scope.get(v, 0.0) for v in variables}
    for aux, info in auxiliary.items():
        if info.get("kind") in ("exo_lag", "exo_lead"):
            steady_state[aux] = 0.0
        else:
            steady_state[aux] = steady_state.get(info["root"], 0.0)
    return steady_state


def _eval_initval_defs(
    defs: Mapping[str, Node],
    param_values: Mapping[str, float],
    variables: Sequence[str],
    auxiliary: Mapping[str, Mapping[str, Any]],
    local_variables: Mapping[str, Node] | None = None,
) -> dict[str, float]:
    """Evaluate an ``initval`` block at ``param_values`` into a solver guess.

    Lines that cannot be evaluated are skipped (``initval`` values are only a
    starting point); auxiliary lead/lag variables inherit their root's guess,
    defaulting to ``1.0`` (or ``0.0`` for exogenous shock auxiliaries).
    """
    init_scope: dict[tuple[str, int] | str, float] = {}
    init_scope.update(param_values)
    if local_variables:
        for loc_name, loc_node in local_variables.items():
            if not loc_node.variables():
                try:
                    p_scope = {k: v for k, v in init_scope.items() if isinstance(k, str)}
                    init_scope[loc_name] = float(loc_node.eval(init_scope, p_scope))
                except Exception:
                    pass
    for i_name, i_expr in defs.items():
        try:
            p_scope = {**param_values, **{k: v for k, v in init_scope.items() if isinstance(k, str)}}
            val = i_expr.eval(init_scope, p_scope)
            init_scope[i_name] = float(val)
        except Exception:
            pass
    guess = {v: init_scope[v] for v in variables if v in init_scope}
    for aux, info in auxiliary.items():
        if info.get("kind") in ("exo_lag", "exo_lead"):
            guess[aux] = 0.0
        else:
            guess[aux] = guess.get(info["root"], 1.0)
    return guess



_COMPOSITE = (UnaryOp, BinOp, Call)


def _render_equations_hoisted(
    equations: Sequence[Node], shock_names: set[str]
) -> tuple[list[str], list[str]]:
    """Python source for residual expressions with parameter work hoisted.

    Returns ``(prologue, expressions)``. The prologue binds every parameter
    the equations use to a local (``_p0 = params.csigma``) and every
    parameter-only composite sub-expression that is either maximal (its
    parent involves a variable) or repeated to a temporary (``_c3 = ...``),
    in increasing size so each temporary only uses earlier ones. The
    expressions then refer to those names. The arithmetic is exactly the
    one the plain ``Node.to_python`` rendering performs, only not repeated.
    """
    counts: dict[Node, int] = {}
    maximal: dict[Node, None] = {}
    params_used: dict[str, None] = {}

    def visit(node: Node) -> bool:
        """Record parameter-only composites; return True if ``node`` has no variable."""
        if isinstance(node, Param):
            params_used.setdefault(node.name, None)
            return True
        if isinstance(node, Const):
            return True
        if isinstance(node, UnaryOp):
            kids: tuple[Node, ...] = (node.expr,)
        elif isinstance(node, BinOp):
            kids = (node.left, node.right)
        elif isinstance(node, Call):
            kids = tuple(node.args)
        else:
            return False
        flags = [visit(k) for k in kids]
        if all(flags):
            counts[node] = counts.get(node, 0) + 1
            return True
        for k, f in zip(kids, flags):
            if f and isinstance(k, _COMPOSITE):
                maximal.setdefault(k, None)
        return False

    for eq in equations:
        if visit(eq) and isinstance(eq, _COMPOSITE):
            maximal.setdefault(eq, None)

    hoisted = dict(maximal)
    for node, c in counts.items():
        if c >= 2:
            hoisted.setdefault(node, None)

    def size(n: Node) -> int:
        if isinstance(n, UnaryOp):
            return 1 + size(n.expr)
        if isinstance(n, BinOp):
            return 1 + size(n.left) + size(n.right)
        if isinstance(n, Call):
            return 1 + sum(size(a) for a in n.args)
        return 1

    pnames = {p: f"_p{i}" for i, p in enumerate(params_used)}
    names: dict[Node, str] = {}
    ns = ("lead", "curr", "lag", "shocks", "params")

    def emit_ss(node: Node, defining: bool = False) -> str:
        if not defining and isinstance(node, _COMPOSITE):
            t = names.get(node)
            if t is not None:
                return t
        if isinstance(node, Param):
            return pnames[node.name]
        if isinstance(node, Const):
            return str(node.value)
        if isinstance(node, Var):
            if shock_names and node.name in shock_names:
                raise ModelError(f"STEADY_STATE() cannot contain exogenous variable {node.name!r}")
            return _py_attr("steady_state", node.name)
        if isinstance(node, UnaryOp):
            return f"({node.op}{emit_ss(node.expr)})"
        if isinstance(node, BinOp):
            py_op = "**" if node.op == "^" else node.op
            return f"({emit_ss(node.left)} {py_op} {emit_ss(node.right)})"
        if isinstance(node, Call):
            fn = node.func.lower()
            if fn == "steady_state":
                return emit_ss(node.args[0])
            return node.python_call([emit_ss(a) for a in node.args])
        return node.to_python(
            "steady_state",
            "steady_state",
            "steady_state",
            "shocks",
            "params",
            shock_names=shock_names,
            steady_state_ns="steady_state",
        )

    def emit(node: Node, defining: bool = False) -> str:
        if not defining and isinstance(node, _COMPOSITE):
            t = names.get(node)
            if t is not None:
                return t
        if isinstance(node, Param):
            return pnames[node.name]
        if isinstance(node, UnaryOp):
            return f"({node.op}{emit(node.expr)})"
        if isinstance(node, BinOp):
            py_op = "**" if node.op == "^" else node.op
            return f"({emit(node.left)} {py_op} {emit(node.right)})"
        if isinstance(node, Call):
            if node.func.lower() == "steady_state":
                return emit_ss(node.args[0])
            return node.python_call([emit(a) for a in node.args])
        return node.to_python(*ns, shock_names=shock_names)

    prologue = [f"{t} = {_py_attr('params', p)}" for p, t in pnames.items()]
    order = sorted(hoisted, key=size)
    for i, node in enumerate(order):
        t = f"_c{i}"
        prologue.append(f"{t} = {emit(node, defining=True)}")
        names[node] = t
    return prologue, [emit(eq) for eq in equations]


@dataclass
class ParsedModelDAG:
    """Structured representation of a parsed Dynare DSGE model as an Expression DAG.

    Model-local ``#`` variables are **symbolic**: ``local_variables`` maps each
    name to its fully inlined expression, and ``equations`` contain that
    expression (with ``Param`` nodes) wherever the local is used, shifted in
    time when the local appears with a lead or lag. This is Dynare's
    semantics (Reference Manual, section 4.5 "Model declaration": "every time
    this variable appears in the model, Dynare will substitute it by the
    expression assigned to the variable"), so a local such as
    ``#cbetabar = cbeta*cgamma^(-csigma);`` follows ``csigma`` whenever the
    compiled equations or derivatives are evaluated at new parameter values
    (estimation, ``load_mod(params=)``, ``osr``, widgets, gradients). Use
    :meth:`evaluate_locals` for their numerical values at a parameter vector.

    ``steady_state`` and ``guess`` hold the ``steady_state_model`` and
    ``initval`` blocks evaluated at the file's calibration
    (``parameter_values``); :meth:`evaluate_steady_state` and
    :meth:`evaluate_initval` re-evaluate them at other parameter values.
    """

    variables: list[str]
    shocks: list[str]
    parameters: list[str]
    parameter_values: dict[str, float]
    equations: list[Node]
    equation_tags: dict[int, dict[str, str]] = field(default_factory=dict)
    is_linear: bool = False
    steady_state_model: dict[str, Node] | None = None
    initval: dict[str, Node] | None = None
    shocks_config: dict[str, Any] = field(default_factory=dict)
    predetermined_variables: list[str] | None = None
    varobs: list[str] | None = None
    options: dict[str, Any] = field(default_factory=dict)
    estimated_params: Any | None = None
    estimated_params_init: Any | None = None
    estimated_params_bounds: Any | None = None
    local_variables: dict[str, Node] = field(default_factory=dict)
    auxiliary_variables: dict[str, dict[str, Any]] = field(default_factory=dict)
    tex_labels: dict[str, str] = field(default_factory=dict)
    shock_cov: np.ndarray | None = None
    steady_state: dict[str, float] | None = None
    guess: dict[str, float] | None = None
    det_shocks: list[str] = field(default_factory=list)
    histval: dict[str, Node] | None = None
    endval: dict[str, Node] | None = None
    histval_values: dict[str, float] | None = None
    endval_values: dict[str, float] | None = None
    trend_vars: list[str] = field(default_factory=list)
    deflators: dict[str, str] = field(default_factory=dict)
    log_trend_vars: list[str] = field(default_factory=list)
    log_deflators: dict[str, str] = field(default_factory=dict)
    shock_groups: dict[str, list[str]] = field(default_factory=dict)
    named_shock_groups: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    hetagent_block: dict[str, Any] | None = None
    markov_switching_config: dict[str, Any] | None = None
    planner_objective: Node | None = None
    ramsey_policy_options: dict[str, Any] = field(default_factory=dict)
    optim_weights: dict[Any, Node] = field(default_factory=dict)
    osr_params: list[str] = field(default_factory=list)
    osr_params_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    osr_options: dict[str, Any] = field(default_factory=dict)
    calib_smoother_options: dict[str, Any] = field(default_factory=dict)
    shock_decomposition_options: dict[str, Any] = field(default_factory=dict)
    perfect_foresight_options: dict[str, Any] = field(default_factory=dict)
    commands: list[dict[str, Any]] = field(default_factory=list)
    source_statements: list[dict[str, Any]] = field(default_factory=list)
    param_assignments: dict[str, Node] = field(default_factory=dict)

    @property
    def varexo_det(self) -> list[str]:
        return self.det_shocks

    @varexo_det.setter
    def varexo_det(self, val: list[str]) -> None:
        self.det_shocks = val

    def _merged_params(self, params: Mapping[str, float] | None) -> dict[str, float]:
        merged = dict(self.parameter_values)
        if params:
            merged.update({str(k): float(v) for k, v in dict(params).items()})
        return merged

    def evaluate_locals(
        self, params: Mapping[str, float] | None = None
    ) -> dict[str, float]:
        """Numerical values of the parameter-only model-local variables.

        Parameters
        ----------
        params : Mapping[str, float], optional
            Parameter values overriding the file's calibration
            (``parameter_values``); omitted names keep their calibrated value.

        Returns
        -------
        dict[str, float]
            ``{local_name: value}`` for every ``#`` local whose inlined
            expression involves no endogenous or exogenous variable and can be
            evaluated. Locals over variables have no single value and are left
            out.
        """
        merged = self._merged_params(params)
        out: dict[str, float] = {}
        for name, node in self.local_variables.items():
            if node.variables():
                continue
            try:
                out[name] = float(node.eval({}, merged))
            except Exception:
                continue
        return out

    def evaluate_steady_state(
        self, params: Mapping[str, float] | None = None
    ) -> dict[str, float] | None:
        """Re-evaluate the ``steady_state_model`` block at new parameter values.

        Dynare compiles ``steady_state_model`` into a function of the
        parameters and calls it at every parameter vector it solves at, so a
        steady state that depends on parameters (SW07's
        ``robs = (cpie/(cbeta*cgamma^(-csigma)) - 1)*100``) must be recomputed
        whenever they change. ``self.steady_state`` is this method at the
        file's calibration.

        Parameters
        ----------
        params : Mapping[str, float], optional
            Parameter values overriding ``parameter_values``. Parameter
            assignments at the top of the file are **not** re-run (as in
            Dynare, where they are executed once when the file is read).

        Returns
        -------
        dict[str, float] or None
            Steady-state level of every endogenous variable (auxiliary
            lead/lag variables included), or ``None`` when the file has no
            ``steady_state_model`` block.

        Raises
        ------
        ValueError
            An assignment in the block cannot be evaluated at ``params``.
        """
        if not self.steady_state_model:
            return None
        return _eval_steady_state_defs(
            self.steady_state_model,
            self._merged_params(params),
            self.variables,
            self.auxiliary_variables,
        )

    def _steady_state_model_fn(
        self, params_dict: Mapping[str, float] | None = None
    ) -> np.ndarray:
        """Compiled steady-state evaluator mapping ``params_dict -> y_ss`` array."""
        ss = self.evaluate_steady_state(params_dict)
        if ss is None:
            raise ValueError("Model does not define a steady_state_model block.")
        return np.array([float(ss.get(v, 0.0)) for v in self.variables], dtype=float)

    def evaluate_optim_weights(
        self, params: Mapping[str, float] | None = None
    ) -> dict[Any, float]:
        """Evaluate ``optim_weights`` block expressions at ``params``."""
        merged = self._merged_params(params)
        out: dict[Any, float] = {}
        for k, node in self.optim_weights.items():
            out[k] = float(node.eval({}, merged))
        return out

    def evaluate_initval(
        self, params: Mapping[str, float] | None = None
    ) -> dict[str, float] | None:
        """Re-evaluate the ``initval`` block (the steady-state solver's guess).

        Same conventions as :meth:`evaluate_steady_state`, except that lines
        that cannot be evaluated are skipped. Returns ``None`` when the file
        has no ``initval`` block.
        """
        if not self.initval:
            return None
        return _eval_initval_defs(
            self.initval,
            self._merged_params(params),
            self.variables,
            self.auxiliary_variables,
        )

    def compile_equations(self) -> Callable:
        """Compile AST equations into a Python callable compatible with legacy LinearModel.

        The callable is ``eqs(lead, curr, lag, shocks, params, steady_state=None)`` and reads
        every parameter, including those inside model-local variables, from
        ``params`` on each call. Names that are Python keywords (a parameter
        called ``lambda``) are read with ``getattr(params, 'lambda')``.
        Dynamic callers supply a fixed ``steady_state`` namespace for
        ``STEADY_STATE()``; static callers omit it to use ``curr``.

        Each parameter is read once per call, and every parameter-only
        sub-expression (typically an inlined ``#`` local such as SW07's
        ``cbetabar*cgamma``) is computed once per call before the residuals,
        so keeping locals symbolic does not multiply their cost by the
        number of places they are used.
        """
        prologue, py_exprs = _render_equations_hoisted(
            self.equations, set(self.shocks)
        )
        code_body = (
            "def _generated_equations(lead, curr, lag, shocks, params, steady_state=None):\n"
            "    if steady_state is None:\n"
            "        steady_state = curr\n"
        )
        for line in prologue:
            code_body += f"    {line}\n"
        code_body += "    return [\n"
        for pe in py_exprs:
            code_body += f"        {pe},\n"
        code_body += "    ]\n"

        scope: dict[str, Any] = {"np": np, "math": math}
        try:
            import scipy.special  # type: ignore

            scope["scipy"] = scipy
        except ImportError:
            pass

        exec(code_body, scope)
        return scope["_generated_equations"]

    def to_dynare_dict(self) -> dict[str, Any]:
        """Convert container to the dictionary format expected by load_mod and build_dynare."""
        return {
            "variables": list(self.variables),
            "shocks": list(self.shocks),
            "det_shocks": list(self.det_shocks),
            "varexo_det": list(self.det_shocks),
            "params": dict(self.parameter_values),
            "predetermined_variables": (
                list(self.predetermined_variables)
                if self.predetermined_variables
                else None
            ),
            "guess": dict(self.guess) if self.guess else None,
            "steady_state": dict(self.steady_state) if self.steady_state else None,
            "steady_state_model_fn": (
                self._steady_state_model_fn if self.steady_state_model else None
            ),
            "histval": dict(self.histval_values) if self.histval_values else None,
            "endval": dict(self.endval_values) if self.endval_values else None,
            "equations": self.compile_equations(),
            "shock_cov": self.shock_cov,
            "options": dict(self.options),
            "varobs": list(self.varobs) if self.varobs else None,
            "estimated_params": self.estimated_params,
            "estimated_params_init": self.estimated_params_init,
            "estimated_params_bounds": self.estimated_params_bounds,
            "shock_groups": dict(self.shock_groups),
            "named_shock_groups": dict(self.named_shock_groups),
            "shocks_config": dict(self.shocks_config),
            "anticipated_shocks": (
                dict(self.shocks_config["anticipated"])
                if self.shocks_config.get("anticipated")
                else None
            ),
            "planner_objective": self.planner_objective,
            "ramsey_policy_options": dict(self.ramsey_policy_options),
            "optim_weights": (
                self.evaluate_optim_weights() if self.optim_weights else {}
            ),
            "osr_params": list(self.osr_params),
            "osr_params_bounds": dict(self.osr_params_bounds),
            "osr_options": dict(self.osr_options),
            "calib_smoother_options": dict(self.calib_smoother_options),
            "shock_decomposition_options": dict(self.shock_decomposition_options),
            "perfect_foresight_options": dict(self.perfect_foresight_options),
            "commands": list(self.commands),
        }


class Parser:
    """Recursive-descent parser building expression DAGs from token streams."""

    def __init__(self, tokens: list[Token], raw_text: str = "", *, collect_only: bool = False):
        self.tokens = tokens
        self.pos = 0
        self.raw_text = raw_text
        self.collect_only = collect_only
        self.source_statements: list[dict[str, Any]] = []
        self._skipped = False
        self._pending_command: tuple[dict[str, Any], list[str]] | None = None

        # Symbol tables
        self.variables: list[str] = []
        self.shocks: list[str] = []
        self.det_shocks: list[str] = []
        self.parameters: list[str] = []
        self.predetermined_variables: list[str] = []
        self.varobs: list[str] = []
        self.param_assignments: dict[str, Node] = {}
        self.param_values: dict[str, float] = {}
        self._seen_numerical_command: bool = False
        self._indexed_zero_targets: set[str] = set()

        # Model contents
        self.equations: list[Node] = []
        self.equation_tags: dict[int, dict[str, str]] = {}
        self.local_var_defs: dict[str, Node] = {}
        self.local_var_names: set[str] = set()
        self._in_model_block: bool = False
        self.tex_labels: dict[str, str] = {}
        self.is_linear: bool = False

        # Blocks
        self.initval_defs: dict[str, Node] = {}
        self.histval_defs: dict[str, Node] = {}
        self.endval_defs: dict[str, Node] = {}
        self.steady_state_defs: dict[str, Node] = {}
        self.shocks_config: dict[str, Any] = {
            "variances": {},
            "stderrs": {},
            "covariances": {},
            "correlations": {},
            "anticipated": {},
        }
        self.has_shocks_block: bool = False
        self.options: dict[str, Any] = {}
        self.trend_vars: list[str] = []
        self.log_trend_vars: list[str] = []
        self.deflators: dict[str, str] = {}
        self.log_deflators: dict[str, str] = {}
        self.shock_groups: dict[str, list[str]] = {}
        self.named_shock_groups: dict[str, dict[str, list[str]]] = {}
        self.hetagent_config: dict[str, Any] | None = None
        self.markov_switching_config: dict[str, Any] | None = None

        # Native Command Blocks
        self.planner_objective: Node | None = None
        self.ramsey_policy_options: dict[str, Any] = {}
        self.optim_weights: dict[Any, Node] = {}
        self.osr_params: list[str] = []
        self.osr_params_bounds: dict[str, tuple[Node, Node]] = {}
        self.osr_options: dict[str, Any] = {}
        self.calib_smoother_options: dict[str, Any] = {}
        self.shock_decomposition_options: dict[str, Any] = {}
        self.perfect_foresight_options: dict[str, Any] = {}
        self.commands: list[dict[str, Any]] = []

    def _curr(self) -> Token:
        return self.tokens[self.pos]

    def _peek_token(self, offset: int = 1) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def _advance(self) -> Token:
        t = self.tokens[self.pos]
        if t.type != "EOF":
            self.pos += 1
        return t

    def _match(self, token_type: str, value: Any = None) -> bool:
        t = self._curr()
        if t.type == token_type:
            if value is None or t.value == value:
                self._advance()
                return True
        return False

    def _expect(self, token_type: str, value: Any = None) -> Token:
        t = self._curr()
        if t.type != token_type or (value is not None and t.value != value):
            expected = f"{token_type}({value})" if value is not None else token_type
            got = f"{t.type}({t.value})"
            raise DynareParseError(f"Expected {expected}, got {got}", t.line, t.col)
        return self._advance()

    # -------------------------------------------------------------------------
    # Expression Parsing
    # -------------------------------------------------------------------------

    def parse_expression(self) -> Node:
        return self._parse_relational()

    def _parse_relational(self) -> Node:
        left = self._parse_additive()
        while self._curr().type in (
            "EQUAL",
            "NOT_EQUAL",
            "LESS",
            "LESS_EQUAL",
            "GREATER",
            "GREATER_EQUAL",
        ):
            op_tok = self._advance()
            right = self._parse_additive()
            left = BinOp(op_tok.value, left, right)
        return left

    def _parse_additive(self) -> Node:
        left = self._parse_multiplicative()
        while self._curr().type in ("PLUS", "MINUS"):
            op_tok = self._advance()
            right = self._parse_multiplicative()
            left = BinOp(op_tok.value, left, right)
        return left

    def _parse_multiplicative(self) -> Node:
        left = self._parse_unary()
        while self._curr().type in ("STAR", "SLASH"):
            op_tok = self._advance()
            right = self._parse_unary()
            left = BinOp(op_tok.value, left, right)
        return left

    def _parse_unary(self) -> Node:
        if self._curr().type == "PLUS":
            self._advance()
            return UnaryOp("+", self._parse_unary())
        if self._curr().type == "MINUS":
            self._advance()
            return UnaryOp("-", self._parse_unary())
        return self._parse_power()

    def _parse_power(self) -> Node:
        left = self._parse_primary()
        if self._curr().type == "CARET":
            self._advance()
            # Right-associative exponentiation: a ^ b ^ c == a ^ (b ^ c)
            right = self._parse_unary()
            return BinOp("^", left, right)
        return left

    def _parse_primary(self) -> Node:
        t = self._curr()

        # Constant numbers
        if t.type == "NUMBER":
            self._advance()
            return Const(t.value)

        # Parenthesised expression
        if t.type == "LPAREN":
            self._advance()
            expr = self.parse_expression()
            self._expect("RPAREN")
            return expr

        # Identifiers, keywords, operators
        if t.type in ("IDENT", "KEYWORD"):
            ident = str(t.value)
            ident_lower = ident.lower()

            # Constants nan, inf
            if ident_lower == "nan":
                self._advance()
                return Const(float("nan"))
            if ident_lower == "inf":
                self._advance()
                return Const(float("inf"))

            # Operators STEADY_STATE(expr)
            if ident_lower == "steady_state":
                self._advance()
                self._expect("LPAREN")
                expr = self.parse_expression()
                self._expect("RPAREN")
                forbidden = {name for name, _ in expr.variables()} & set(self.shocks + self.det_shocks)
                if forbidden:
                    raise ModelError(
                        "STEADY_STATE() cannot contain exogenous variables: "
                        + ", ".join(sorted(forbidden))
                    )
                return Call("STEADY_STATE", (expr,))

            # Operator diff(expr) -> expr - expr(-1)
            if ident_lower == "diff":
                self._advance()
                self._expect("LPAREN")
                expr = self.parse_expression()
                self._expect("RPAREN")
                return BinOp("-", expr, expr.shift(-1))

            # Operator EXPECTATION(k)(expr)
            if ident_lower == "expectation":
                self._advance()
                self._expect("LPAREN")
                lead_k = 0
                if self._curr().type == "NUMBER":
                    lead_k = int(self._advance().value)
                elif (
                    self._curr().type in ("PLUS", "MINUS")
                    and self._peek_token(1).type == "NUMBER"
                ):
                    sign = -1 if self._advance().type == "MINUS" else 1
                    lead_k = sign * int(self._advance().value)
                self._expect("RPAREN")
                self._expect("LPAREN")
                expr = self.parse_expression()
                self._expect("RPAREN")
                return Call("EXPECTATION", (Const(lead_k), expr))

            # Identifier followed by parenthesis
            if self._peek_token(1).type == "LPAREN":
                # Check whether this is lead/lag coordinate indexing: IDENT ( [+-]? INT )
                # It is a lead/lag index if:
                # 1. ident is NOT a standard math function (or is in variables/shocks/locals)
                # 2. tokens inside parens match [+-]? INT )
                is_local = self._in_model_block and (ident in self.local_var_names)
                is_math_fn = (
                    ident_lower in MATH_FUNCTIONS
                    and ident not in self.variables
                    and not is_local
                )

                can_be_lead_lag = not is_math_fn
                if can_be_lead_lag:
                    tok2 = self._peek_token(2)
                    tok3 = self._peek_token(3)
                    tok4 = self._peek_token(4)
                    # Pattern: ( INT )
                    if tok2.type == "NUMBER" and tok3.type == "RPAREN":
                        self._advance()  # ident
                        self._advance()  # (
                        lead_val = int(self._advance().value)
                        self._advance()  # )
                        if is_local:
                            return _LocalRef(ident, lead_val)
                        return Var(ident, lead_val)
                    # Pattern: ( +/- INT )
                    elif (
                        tok2.type in ("PLUS", "MINUS")
                        and tok3.type == "NUMBER"
                        and tok4.type == "RPAREN"
                    ):
                        self._advance()  # ident
                        self._advance()  # (
                        sign = -1 if self._advance().type == "MINUS" else 1
                        lead_val = sign * int(self._advance().value)
                        self._advance()  # )
                        if is_local:
                            return _LocalRef(ident, lead_val)
                        return Var(ident, lead_val)

                # Otherwise, it's a function call!
                self._advance()  # ident
                self._advance()  # (
                args = []
                if self._curr().type != "RPAREN":
                    args.append(self.parse_expression())
                    while self._match("COMMA"):
                        args.append(self.parse_expression())
                self._expect("RPAREN")
                return Call(ident, tuple(args))

            # Identifier without parenthesis
            self._advance()
            if self._in_model_block and ident in self.local_var_names:
                return _LocalRef(ident, 0)
            if ident in self.variables or ident in self.shocks:
                return Var(ident, 0)
            if ident in self.parameters:
                return Param(ident)
            # Default to Param for undefined symbols in equations
            return Param(ident)

        raise DynareParseError(
            f"Unexpected token {t.type}({t.value!r}) in expression", t.line, t.col
        )

    # -------------------------------------------------------------------------
    # Block and Declaration Parsing
    # -------------------------------------------------------------------------

    def _parse_id_list(self) -> list[str]:
        """Parse identifier list separated by spaces or commas until semicolon."""
        ids: list[str] = []
        seen: set[str] = set()
        while self._curr().type != "SEMI" and self._curr().type != "EOF":
            # Skip optional comma
            self._match("COMMA")
            if self._curr().type in ("IDENT", "KEYWORD"):
                tok = self._curr()
                name = str(self._advance().value)
                if name in seen:
                    raise DynareParseError(
                        f"Symbol '{name}' declared twice", tok.line, tok.col
                    )
                seen.add(name)
                ids.append(name)
                # Capture optional TeX label
                if self._curr().type == "TEX":
                    tex_tok = self._advance()
                    tex_str = str(tex_tok.value).strip("$").strip()
                    self.tex_labels[name] = tex_str
                # Skip optional options in parentheses e.g. (long_name='...')
                if self._curr().type == "LPAREN":
                    self._advance()
                    while self._curr().type != "RPAREN" and self._curr().type != "EOF":
                        self._advance()
                    self._match("RPAREN")
            else:
                break
        self._expect("SEMI")
        return ids

    def _parse_equation_tags(self) -> dict[str, str]:
        """Parse equation tag [name='...', ...] immediately before an equation."""
        tags: dict[str, str] = {}
        if not self._match("LBRACKET"):
            return tags
        while self._curr().type != "RBRACKET" and self._curr().type != "EOF":
            self._match("COMMA")
            if self._curr().type in ("IDENT", "KEYWORD", "STRING"):
                key = str(self._advance().value)
                val = key
                if self._match("ASSIGN"):
                    if self._curr().type in ("STRING", "IDENT", "KEYWORD", "NUMBER"):
                        val = str(self._advance().value)
                tags[key] = val
            else:
                self._advance()
        self._expect("RBRACKET")
        return tags

    def _parse_model_block(self) -> None:
        """Parse model; or model(linear); block."""
        self._expect("KEYWORD", "model")
        if self._match("LPAREN"):
            if (
                self._curr().type in ("KEYWORD", "IDENT")
                and self._curr().value == "linear"
            ):
                self._advance()
                self.is_linear = True
            self._expect("RPAREN")
        self._expect("SEMI")

        self._in_model_block = True
        try:
            # First pass: collect all '#' local variable names so parser knows them as _LocalRef
            lookahead_pos = self.pos
            while lookahead_pos < len(self.tokens):
                tok = self.tokens[lookahead_pos]
                if tok.type == "KEYWORD" and tok.value == "end":
                    break
                if tok.type == "HASH":
                    next_tok = self.tokens[lookahead_pos + 1]
                    if next_tok.type in ("IDENT", "KEYWORD"):
                        loc_name = str(next_tok.value)
                        if loc_name in self.variables:
                            raise ModelError(
                                f"Model-local variable '{loc_name}' collides with declared endogenous variable"
                            )
                        self.local_var_names.add(loc_name)
                lookahead_pos += 1

            # Second pass: parse equations and '#' definitions
            saw_end = False
            while self._curr().type != "EOF":
                if self._curr().type == "KEYWORD" and self._curr().value == "end":
                    self._advance()
                    self._expect("SEMI")
                    saw_end = True
                    break

                statement_start = self.pos
                # Parse optional tags [name='...']
                tags = self._parse_equation_tags()

                # Check for model-local '#' variable definition
                if self._match("HASH"):
                    var_tok = self._expect("IDENT")
                    name = str(var_tok.value)
                    if name in self.variables:
                        raise ModelError(
                            f"Model-local variable '{name}' collides with declared endogenous variable"
                        )
                    self._expect("ASSIGN")
                    expr = self.parse_expression()
                    self._expect("SEMI")
                    self.local_var_defs[name] = expr
                    self._record_statement(statement_start, "model-local")
                    continue

                # Regular model equation: lhs = rhs; or expr;
                lhs = self.parse_expression()
                eq_node: Node
                if self._match("ASSIGN"):
                    rhs = self.parse_expression()
                    eq_node = BinOp("-", lhs, rhs)
                else:
                    eq_node = lhs
                self._expect("SEMI")

                eq_idx = len(self.equations)
                if tags:
                    self.equation_tags[eq_idx] = tags
                self.equations.append(eq_node)
                self._record_statement(statement_start, "equation")

            if not saw_end:
                raise DynareParseError(
                    "Unclosed model block: expected 'end;'",
                    self._curr().line,
                    self._curr().col,
                )
        finally:
            self._in_model_block = False

    def _parse_assignment_block(self, block_name: str) -> dict[str, Node]:
        """Parse assignment block (initval, histval, endval, steady_state_model) storing AST expressions."""
        self._expect("KEYWORD", block_name)
        self._expect("SEMI")
        defs: dict[str, Node] = {}
        self._indexed_zero_targets = set()
        while self._curr().type != "EOF":
            if self._curr().type == "KEYWORD" and self._curr().value == "end":
                self._advance()
                self._expect("SEMI")
                break
            if self._curr().type in ("IDENT", "KEYWORD"):
                name = str(self._advance().value)
                idx_val: int | None = None
                if self._match("LPAREN"):
                    # Handle possible lead/lag index like y(0), y(-1), y(-2), y(+1)
                    sign = 1
                    if self._curr().type in ("NUMBER", "INT", "MINUS", "PLUS"):
                        if self._curr().type in ("MINUS", "PLUS"):
                            if self._advance().type == "MINUS":
                                sign = -1
                        if self._curr().type in ("NUMBER", "INT"):
                            raw_idx = self._advance().value
                            if isinstance(raw_idx, (int, float)) and int(raw_idx) == raw_idx:
                                idx_val = sign * int(raw_idx)
                    self._expect("RPAREN")

                self._expect("ASSIGN")
                expr = self.parse_expression()
                self._expect("SEMI")
                target_key = name
                if idx_val is not None:
                    if block_name == "histval":
                        if name in self.shocks:
                            if idx_val <= 0:
                                target_key = f"AUX_EXO_LAG_{name}_{abs(idx_val) + 1}"
                        elif idx_val == 0:
                            target_key = name
                            self._indexed_zero_targets.add(name)
                        elif idx_val < 0:
                            target_key = f"AUX_LAG_{name}_{abs(idx_val)}"
                    elif block_name == "endval":
                        if name in self.shocks:
                            if idx_val > 0:
                                target_key = f"AUX_EXO_LEAD_{name}_{idx_val}"
                        elif idx_val == 0:
                            target_key = name
                            self._indexed_zero_targets.add(name)
                        elif idx_val > 0:
                            target_key = f"AUX_LEAD_{name}_{idx_val}"
                defs[target_key] = expr
            else:
                if self.collect_only:
                    raise DynareParseError("Unhandled token in recognized block", self._curr().line, self._curr().col)
                self._advance()
        return defs

    def _parse_period_list(self) -> list[int]:
        """Parse periods range or list (e.g. 1:4, 1:2:8, 1, 2, 3, [1:4])."""
        periods: list[int] = []
        has_bracket = bool(self._match("LBRACKET") or self._match("LPAREN"))
        while self._curr().type not in ("SEMI", "RBRACKET", "RPAREN", "EOF"):
            if self._curr().type == "NUMBER":
                raw1 = self._advance().value
                if raw1 != int(raw1):
                    raise ValueError("Shock periods must be positive integers")
                val1 = int(raw1)
                if self._match("COLON"):
                    raw2 = self._expect("NUMBER").value
                    if raw2 != int(raw2):
                        raise ValueError("Shock periods must be positive integers")
                    val2 = int(raw2)
                    if self._match("COLON"):
                        raw3 = self._expect("NUMBER").value
                        if raw3 != int(raw3):
                            raise ValueError("Shock periods must be positive integers")
                        val3 = int(raw3)
                        period_range = range(val1, val3 + 1, val2)
                        if self.collect_only and len(period_range) > 100_000:
                            raise DynareParseError("Inspection period range resource limit exceeded", self._curr().line, self._curr().col)
                        periods.extend(period_range)
                    else:
                        period_range = range(val1, val2 + 1)
                        if self.collect_only and len(period_range) > 100_000:
                            raise DynareParseError("Inspection period range resource limit exceeded", self._curr().line, self._curr().col)
                        periods.extend(period_range)
                else:
                    periods.append(val1)
            elif self._match("COMMA"):
                continue
            else:
                break
        if has_bracket:
            if not (self._match("RBRACKET") or self._match("RPAREN")):
                raise DynareParseError(
                    "Expected closing bracket/parenthesis for periods",
                    self._curr().line,
                    self._curr().col,
                )
        self._expect("SEMI")
        return periods

    def _parse_values_list(self) -> list[float]:
        """Parse dated values with optional commas and parenthesized expressions.

        Outside parentheses, a plus/minus starts the next signed value rather
        than extending the preceding expression. Bare parameters and products
        remain accepted for compatibility; addition/subtraction belongs inside
        parentheses, as in Dynare's deterministic-shock grammar.
        """
        try:
            return self._parse_values_list_impl()
        except DynareParseError as exc:
            # parse_mod may fall back on general grammar errors, but its legacy
            # parser cannot preserve dated shocks. Keep syntax failures in this
            # numerical declaration fatal, with their location for inspection.
            raise _ShockValuesParseError(
                f"Invalid shock values: {exc.message}", exc.line, exc.col,
            ) from exc

    def _parse_values_list_impl(self) -> list[float]:
        values: list[float] = []
        # Full-file parsing reaches dated shocks before its calibration
        # post-pass. Resolve the preceding declarations here; the ordered
        # executor instead seeds param_values with its calibration snapshot.
        scope = dict(self.param_values)
        if not self.collect_only:
            for _ in range(len(self.param_assignments) + 1):
                previous_size = len(scope)
                for name, assignment in self.param_assignments.items():
                    if name not in scope:
                        try:
                            scope[name] = float(assignment.eval({}, scope))
                        except (KeyError, ValueError, ArithmeticError):
                            pass
                if len(scope) == previous_size:
                    break
        has_bracket = self._match("LBRACKET")
        while self._curr().type not in ("SEMI", "RBRACKET", "EOF"):
            # Parsing a full additive expression here turns ``.02 -.01 -.01``
            # into one zero, which the caller then broadcasts over all dates.
            # Parenthesized expressions are still fully parsed by _parse_primary.
            expr = self._parse_multiplicative()
            if self.collect_only:
                values.append(0.0)  # shape-only placeholder, never returned as calibration
                self._match("COMMA")
                continue
            try:
                val = float(expr.eval({}, scope))
            except (KeyError, ValueError, ArithmeticError) as exc:
                raise ValueError("Shock values must be evaluable from the preceding calibration") from exc
            if not math.isfinite(val):
                raise ValueError("Shock values must be finite")
            values.append(val)
            self._match("COMMA")
        if has_bracket:
            self._expect("RBRACKET")
        self._expect("SEMI")
        return values

    def _record_anticipated_shock(self, name: str, periods: list[int], values: list[float]) -> None:
        """Merge dated declarations; a later value replaces only its own date."""
        if not periods or any(period < 1 for period in periods) or len(periods) != len(values):
            # ValueError is load-bearing: parse_mod must not send a malformed
            # numerical declaration through its permissive legacy fallback.
            raise ValueError("Shock periods must be positive and match the number of values")
        anticipated = self.shocks_config.setdefault("anticipated", {})
        previous = anticipated.get(name, {})
        dates = dict(zip(previous.get("periods", []), previous.get("values", [])))
        dates.update(zip(periods, values))
        ordered = sorted(dates)
        anticipated[name] = {"periods": ordered, "values": [dates[date] for date in ordered]}
        self.options["anticipated_shocks"] = anticipated

    def _parse_shocks_block(self) -> None:
        """Parse shocks; block."""
        self._expect("KEYWORD", "shocks")
        if self._match("LPAREN"):
            if self._curr().type in ("IDENT", "KEYWORD"):
                option = str(self._advance().value)
                if option == "overwrite":
                    self.shocks_config = {"variances": {}, "stderrs": {}, "covariances": {},
                                          "correlations": {}, "anticipated": {}}
                    self.options["anticipated_shocks"] = self.shocks_config["anticipated"]
                self.shocks_config["type"] = option
            self._expect("RPAREN")
        self._expect("SEMI")
        self.has_shocks_block = True

        last_var: str | None = None

        while self._curr().type != "EOF":
            if self._curr().type == "KEYWORD" and self._curr().value == "end":
                self._advance()
                self._expect("SEMI")
                break

            if self._match("KEYWORD", "var"):
                shock1 = str(self._expect("IDENT").value)
                last_var = shock1
                if self._curr().value == "periods":
                    self._advance()
                    periods = self._parse_period_list()
                    if self._curr().value != "values":
                        raise DynareParseError(
                            f"Expected 'values' after 'periods' in shocks block for shock '{shock1}'",
                            self._curr().line,
                            self._curr().col,
                        )
                    self._advance()
                    values = self._parse_values_list()
                    if len(values) == 1 and len(periods) > 1:
                        values = [values[0]] * len(periods)
                    self._record_anticipated_shock(shock1, periods, values)
                elif self._match("COMMA"):
                    if (
                        self._curr().type in ("KEYWORD", "IDENT")
                        and self._curr().value == "stderr"
                    ):
                        self._advance()
                        expr = self.parse_expression()
                        self._expect("SEMI")
                        self.shocks_config["stderrs"][shock1] = expr
                    else:
                        shock2 = str(self._expect("IDENT").value)
                        self._expect("ASSIGN")
                        expr = self.parse_expression()
                        self._expect("SEMI")
                        self.shocks_config["covariances"][(shock1, shock2)] = expr
                elif self._match("SEMI"):
                    if self._curr().value == "periods":
                        self._advance()
                        periods = self._parse_period_list()
                        if self._curr().value != "values":
                            raise DynareParseError(
                                f"Expected 'values' after 'periods' in shocks block for shock '{shock1}'",
                                self._curr().line,
                                self._curr().col,
                            )
                        self._advance()
                        values = self._parse_values_list()
                        if len(values) == 1 and len(periods) > 1:
                            values = [values[0]] * len(periods)
                        self._record_anticipated_shock(shock1, periods, values)
                    elif (
                        self._curr().type in ("KEYWORD", "IDENT")
                        and self._curr().value == "stderr"
                    ):
                        self._advance()
                        expr = self.parse_expression()
                        self._expect("SEMI")
                        self.shocks_config["stderrs"][shock1] = expr
                    elif (
                        self._curr().type in ("KEYWORD", "IDENT")
                        and self._curr().value == "variance"
                    ):
                        self._advance()
                        expr = self.parse_expression()
                        self._expect("SEMI")
                        self.shocks_config["variances"][shock1] = expr
                elif self._match("ASSIGN"):
                    expr = self.parse_expression()
                    self._expect("SEMI")
                    self.shocks_config["variances"][shock1] = expr
            elif self._curr().value == "periods" and last_var is not None:
                self._advance()
                periods = self._parse_period_list()
                if self._curr().value != "values":
                    raise DynareParseError(
                        f"Expected 'values' after 'periods' in shocks block for shock '{last_var}'",
                        self._curr().line,
                        self._curr().col,
                    )
                self._advance()
                values = self._parse_values_list()
                if len(values) == 1 and len(periods) > 1:
                    values = [values[0]] * len(periods)
                self._record_anticipated_shock(last_var, periods, values)
            elif self._match("KEYWORD", "corr"):
                shock1 = str(self._expect("IDENT").value)
                self._expect("COMMA")
                shock2 = str(self._expect("IDENT").value)
                self._expect("ASSIGN")
                expr = self.parse_expression()
                self._expect("SEMI")
                self.shocks_config["correlations"][(shock1, shock2)] = expr
            elif self._match("SEMI"):
                continue
            else:
                if self.collect_only:
                    raise DynareParseError("Unhandled token in recognized block", self._curr().line, self._curr().col)
                self._advance()

    def _parse_command_options_and_vars(
        self, cmd_name: str
    ) -> tuple[dict[str, Any], list[str]]:
        """Parse ``cmd_name(opt1=val1, opt2=(a, b), flag) var1 var2;``."""
        self._expect("KEYWORD", cmd_name)
        opts: dict[str, Any] = {}
        if self._match("LPAREN"):
            while self._curr().type != "RPAREN" and self._curr().type != "EOF":
                if self._curr().type in ("IDENT", "KEYWORD"):
                    opt_name = str(self._advance().value)
                    if self.collect_only and opt_name in opts:
                        raise DynareParseError(f"Repeated option {opt_name!r}", self._curr().line, self._curr().col)
                    if self._match("ASSIGN"):
                        if self.collect_only and self._curr().type not in {"LPAREN", "LBRACKET", "MINUS", "NUMBER", "IDENT", "KEYWORD", "STRING"}:
                            raise DynareParseError(f"Missing value for option {opt_name!r}", self._curr().line, self._curr().col)
                        if self._match("LPAREN") or self._match("LBRACKET"):
                            items: list[Any] = []
                            while self._curr().type not in ("RPAREN", "RBRACKET", "EOF"):
                                if self._curr().type == "COMMA":
                                    self._advance()
                                    continue
                                if self._curr().type == "MINUS" and self._peek_token(1).type == "NUMBER":
                                    self._advance()
                                    items.append(-self._advance().value)
                                elif self._curr().type == "NUMBER":
                                    items.append(self._advance().value)
                                elif self._curr().type in ("IDENT", "KEYWORD", "STRING"):
                                    items.append(str(self._advance().value).strip("'\""))
                                else:
                                    self._advance()
                            if not (self._match("RPAREN") or self._match("RBRACKET")):
                                pass
                            opts[opt_name] = items
                        elif self._curr().type == "MINUS" and self._peek_token(1).type == "NUMBER":
                            self._advance()
                            opts[opt_name] = -self._advance().value
                        elif self._curr().type == "NUMBER":
                            opts[opt_name] = self._advance().value
                        elif self._curr().type in ("IDENT", "KEYWORD", "STRING"):
                            val_tok = self._advance()
                            # Could also be an expression like beta or 0.99
                            opts[opt_name] = val_tok.value
                    else:
                        opts[opt_name] = True
                else:
                    if self.collect_only:
                        raise DynareParseError("Malformed command option", self._curr().line, self._curr().col)
                    self._advance()
                self._match("COMMA")
            self._expect("RPAREN")

        var_list: list[str] = []
        while self._curr().type != "SEMI" and self._curr().type != "EOF":
            if self._curr().type in ("IDENT", "KEYWORD"):
                var_list.append(str(self._advance().value))
            else:
                if self.collect_only:
                    raise DynareParseError("Unhandled token in recognized block", self._curr().line, self._curr().col)
                self._advance()
        self._expect("SEMI")
        self._pending_command = (dict(opts), list(var_list))
        return opts, var_list

    def _parse_stoch_simul(self) -> None:
        """Parse stoch_simul options."""
        opts, var_list = self._parse_command_options_and_vars("stoch_simul")
        self.options.update(opts)
        self.commands.append(
            {"command": "stoch_simul", "options": dict(opts), "variables": var_list}
        )

    def _parse_planner_objective(self) -> None:
        """Parse ``planner_objective expr;`` or ``planner_objective(expr);``."""
        self._expect("KEYWORD", "planner_objective")
        expr = self.parse_expression()
        self._expect("SEMI")
        self.planner_objective = expr
        self.options["planner_objective"] = expr

    def _parse_ramsey_policy(self, kw: str = "ramsey_policy") -> None:
        """Parse ``ramsey_policy(...) var_list;`` or ``ramsey_model(...);``."""
        opts, var_list = self._parse_command_options_and_vars(kw)
        if var_list and "var_list" not in opts:
            opts["var_list"] = var_list
        self.ramsey_policy_options.update(opts)
        self.options["ramsey_policy"] = dict(self.ramsey_policy_options)
        if kw == "ramsey_policy":
            self.commands.append(
                {"command": "ramsey_policy", "options": dict(self.ramsey_policy_options), "variables": var_list}
            )
        else:
            self.commands.append(
                {"command": "ramsey_model", "options": dict(self.ramsey_policy_options), "variables": var_list}
            )

    def _parse_optim_weights_block(self) -> None:
        """Parse ``optim_weights; var expr; var1, var2 expr; end;``."""
        self._expect("KEYWORD", "optim_weights")
        self._expect("SEMI")
        while self._curr().type != "EOF":
            if self._curr().type == "KEYWORD" and self._curr().value == "end":
                self._advance()
                self._expect("SEMI")
                break
            if self._curr().type == "SEMI":
                self._advance()
                continue
            if self._curr().type in ("IDENT", "KEYWORD"):
                v1 = str(self._advance().value)
                if self._match("COMMA"):
                    v2 = str(self._expect("IDENT").value)
                    self._match("ASSIGN")
                    expr = self.parse_expression()
                    self._expect("SEMI")
                    self.optim_weights[(v1, v2)] = expr
                else:
                    self._match("ASSIGN")
                    expr = self.parse_expression()
                    self._expect("SEMI")
                    self.optim_weights[v1] = expr
            else:
                self._advance()

    def _parse_osr_params_bounds_block(self) -> None:
        """Parse ``osr_params_bounds; param, lb, ub; end;``."""
        self._expect("KEYWORD", "osr_params_bounds")
        self._expect("SEMI")
        while self._curr().type != "EOF":
            if self._curr().type == "KEYWORD" and self._curr().value == "end":
                self._advance()
                self._expect("SEMI")
                break
            if self._curr().type == "SEMI":
                self._advance()
                continue
            if self._curr().type in ("IDENT", "KEYWORD"):
                pname = str(self._advance().value)
                self._expect("COMMA")
                lb_expr = self.parse_expression()
                self._expect("COMMA")
                ub_expr = self.parse_expression()
                self._expect("SEMI")
                self.osr_params_bounds[pname] = (lb_expr, ub_expr)
            else:
                self._advance()

    def _parse_osr_command(self) -> None:
        """Parse ``osr(...) var_list;``."""
        opts, var_list = self._parse_command_options_and_vars("osr")
        if var_list:
            opts["var_list"] = var_list
        self.osr_options.update(opts)
        self.options["osr"] = dict(self.osr_options)
        self.commands.append(
            {"command": "osr", "options": dict(self.osr_options), "variables": var_list}
        )

    def _parse_calib_smoother_command(self) -> None:
        """Parse ``calib_smoother(...) var_list;``."""
        opts, var_list = self._parse_command_options_and_vars("calib_smoother")
        if var_list:
            opts["var_list"] = var_list
        self.calib_smoother_options.update(opts)
        self.options["calib_smoother"] = dict(self.calib_smoother_options)
        self.commands.append(
            {"command": "calib_smoother", "options": dict(self.calib_smoother_options), "variables": var_list}
        )

    def _parse_shock_decomposition_command(self) -> None:
        """Parse ``shock_decomposition(...) var_list;``."""
        opts, var_list = self._parse_command_options_and_vars("shock_decomposition")
        if var_list:
            opts["var_list"] = var_list
        self.shock_decomposition_options.update(opts)
        self.options["shock_decomposition"] = dict(self.shock_decomposition_options)
        self.commands.append(
            {"command": "shock_decomposition", "options": dict(self.shock_decomposition_options), "variables": var_list}
        )

    def _parse_perfect_foresight_command(self, kw: str) -> None:
        """Parse ``perfect_foresight_setup(...);``, ``perfect_foresight_solver(...);``, or ``simul(...);``."""
        opts, var_list = self._parse_command_options_and_vars(kw)
        self.perfect_foresight_options.update(opts)
        self.options["perfect_foresight"] = dict(self.perfect_foresight_options)
        if kw in ("perfect_foresight_solver", "simul"):
            self.commands.append(
                {"command": "perfect_foresight_solver", "options": dict(self.perfect_foresight_options), "variables": var_list}
            )
        else:
            self.commands.append(
                {"command": kw, "options": dict(self.perfect_foresight_options), "variables": var_list}
            )

    def _parse_shock_groups_block(self) -> None:
        """Parse Dynare shock_groups; or shock_groups(name = ...); block."""
        self._expect("KEYWORD", "shock_groups")
        group_set_name = "default"
        if self._match("LPAREN"):
            while self._curr().type != "RPAREN" and self._curr().type != "EOF":
                if (
                    self._curr().type in ("IDENT", "KEYWORD")
                    and str(self._curr().value).lower() in ("name", "group_name")
                ):
                    self._advance()
                    self._expect("ASSIGN")
                    if self._curr().type in ("IDENT", "KEYWORD", "STRING"):
                        group_set_name = str(self._advance().value).strip("'\"")
                else:
                    self._advance()
                self._match("COMMA")
            self._expect("RPAREN")
        self._expect("SEMI")

        target_dict: dict[str, list[str]] = {}
        while self._curr().type != "EOF":
            if self._curr().type == "KEYWORD" and self._curr().value == "end":
                self._advance()
                self._expect("SEMI")
                break
            if self._curr().type == "SEMI":
                self._advance()
                continue
            if self._curr().type in ("IDENT", "KEYWORD", "STRING"):
                grp_tok = self._advance()
                grp_name = str(grp_tok.value).strip("'\"")
                self._expect("ASSIGN")
                shocks_list: list[str] = []
                while self._curr().type not in ("SEMI", "EOF"):
                    if self._curr().type in ("IDENT", "KEYWORD", "STRING"):
                        s_name = str(self._advance().value).strip("'\"")
                        shocks_list.append(s_name)
                    elif self._curr().type == "COMMA":
                        self._advance()
                    else:
                        self._advance()
                self._match("SEMI")
                target_dict[grp_name] = shocks_list
            else:
                self._advance()

        self.named_shock_groups[group_set_name] = target_dict
        if group_set_name == "default" or not self.shock_groups:
            self.shock_groups.update(target_dict)

    def _parse_hetagent_block(self) -> None:
        """Parse hetagent_block; ... end; block."""
        self._expect("KEYWORD", "hetagent_block")
        self._expect("SEMI")
        config: dict[str, Any] = {}
        while self._curr().type != "EOF":
            if self._curr().type == "KEYWORD" and self._curr().value == "end":
                self._advance()
                self._expect("SEMI")
                break
            if self._curr().type == "SEMI":
                self._advance()
                continue
            if self._curr().type in ("IDENT", "KEYWORD"):
                key_tok = self._advance()
                key = str(key_tok.value)
                self._expect("ASSIGN")
                val_list = []
                while self._curr().type not in ("SEMI", "EOF"):
                    if self._curr().type == "COMMA":
                        self._advance()
                        continue
                    val = self._advance().value
                    val_list.append(val)
                self._match("SEMI")
                if len(val_list) == 1:
                    v = val_list[0]
                    v_str = str(v).strip("'\"")
                    v_conv: float | str
                    try:
                        v_conv = float(v_str)
                    except ValueError:
                        v_conv = v_str
                    config[key] = v_conv
                else:
                    config[key] = [str(x).strip("'\"") for x in val_list]
            else:
                self._advance()
        self.hetagent_config = config

    def _parse_ms_val(self) -> Any:
        """Parse value, list, or matrix for markov_switching block."""
        tok = self._curr()
        if tok.type == "LBRACKET":
            self._advance()
            rows: list[Any] = []
            curr_row: list[Any] = []
            while self._curr().type not in ("RBRACKET", "EOF"):
                ctok = self._curr()
                if ctok.type == "COMMA":
                    self._advance()
                    continue
                elif ctok.type == "SEMI":
                    self._advance()
                    if curr_row:
                        rows.append(curr_row)
                        curr_row = []
                    continue
                elif ctok.type == "LBRACKET":
                    sub = self._parse_ms_val()
                    rows.append(sub)
                elif ctok.type == "MINUS":
                    self._advance()
                    num_tok = self._advance()
                    curr_row.append(-num_tok.value)
                elif ctok.type == "NUMBER":
                    curr_row.append(self._advance().value)
                elif ctok.type in ("IDENT", "STRING"):
                    curr_row.append(str(self._advance().value))
                else:
                    self._advance()
            self._match("RBRACKET")
            if curr_row:
                rows.append(curr_row)

            if len(rows) > 1 and all(isinstance(r, (list, tuple, np.ndarray)) for r in rows):
                try:
                    return np.asarray(rows, dtype=float)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError, KeyError, LookupError, AssertionError, TypeError, RuntimeError, OSError, ConnectionError, Exception):
                    return rows
            elif len(rows) == 1 and isinstance(rows[0], list):
                return rows[0]
            else:
                return rows
        elif tok.type == "MINUS":
            self._advance()
            num_tok = self._advance()
            return -num_tok.value
        elif tok.type == "NUMBER":
            return self._advance().value
        elif tok.type in ("IDENT", "STRING"):
            return str(self._advance().value)
        else:
            return str(self._advance().value)

    def _parse_markov_switching_block(self) -> None:
        """Parse markov_switching; ... end; block."""
        self._expect("KEYWORD", "markov_switching")
        self._expect("SEMI")
        config: dict[str, Any] = {
            "num_regimes": None,
            "transition_matrix": None,
            "parameters": {},
            "regimes": {},
        }
        current_regime: int | str | None = None

        while self._curr().type != "EOF":
            if self._curr().type == "KEYWORD" and self._curr().value == "end":
                self._advance()
                self._expect("SEMI")
                break
            if self._curr().type == "SEMI":
                self._advance()
                continue

            tok = self._curr()
            tok_val_lower = str(tok.value).lower()
            if tok.type in ("IDENT", "KEYWORD") and tok_val_lower == "regime":
                self._advance()
                reg_tok = self._curr()
                if reg_tok.type == "NUMBER":
                    current_regime = int(reg_tok.value)
                    self._advance()
                else:
                    current_regime = str(reg_tok.value)
                    self._advance()
                self._match("SEMI")
                if current_regime not in config["regimes"]:
                    config["regimes"][current_regime] = {}
                continue

            if tok.type in ("IDENT", "KEYWORD") and tok_val_lower in ("param", "parameters"):
                self._advance()
                tok = self._curr()

            if tok.type in ("IDENT", "KEYWORD"):
                key = str(self._advance().value)
                if self._curr().type == "ASSIGN":
                    self._advance()
                    val = self._parse_ms_val()
                    self._match("SEMI")
                    if current_regime is not None:
                        config["regimes"][current_regime][key] = val
                    else:
                        key_lower = key.lower()
                        if key_lower in ("num_regimes", "regimes", "n_regimes"):
                            config["num_regimes"] = int(val)
                        elif key_lower in ("p", "transition_matrix", "trans_mat"):
                            config["transition_matrix"] = np.asarray(val, dtype=float)
                        else:
                            config["parameters"][key] = val
                            config[key] = val
                else:
                    self._match("SEMI")
            else:
                self._advance()

        if config["num_regimes"] is None:
            if config["transition_matrix"] is not None:
                config["num_regimes"] = int(config["transition_matrix"].shape[0])
            elif config["regimes"]:
                config["num_regimes"] = len(config["regimes"])
            elif config["parameters"]:
                first_val = next(iter(config["parameters"].values()))
                if isinstance(first_val, (list, tuple, np.ndarray)):
                    config["num_regimes"] = len(first_val)
        self.markov_switching_config = config

    def _skip_unrecognised_statement(self) -> None:
        """Safely skip MATLAB scripting or unhandled commands outside model blocks."""
        self._skipped = True
        t = self._curr()
        # Handle MATLAB loops/conditionals like 'while ... end' or 'if ... end'
        if t.type in ("KEYWORD", "IDENT") and t.value in ("while", "if", "for"):
            self._advance()
            depth = 1
            while depth > 0 and self._curr().type != "EOF":
                cur = self._curr()
                if cur.type in ("KEYWORD", "IDENT"):
                    if cur.value in ("while", "if", "for"):
                        depth += 1
                    elif cur.value == "end":
                        depth -= 1
                self._advance()
            self._match("SEMI")
            return

        # Skip until next semicolon
        while self._curr().type != "SEMI" and self._curr().type != "EOF":
            self._advance()
        self._match("SEMI")

    # -------------------------------------------------------------------------
    # Main Parsing Entry Point
    # -------------------------------------------------------------------------

    def _record_statement(self, start: int, category: str, **metadata: Any) -> None:
        first, last = self.tokens[start], self.tokens[max(start, self.pos - 1)]
        self.source_statements.append(dict(
            name=str(first.value), category=category,
            line=first.line, col=first.col, end_line=last.line,
            end_col=last.col + (1 if last.type == "SEMI" else len(str(last.value))),
            **metadata,
        ))

    def parse(self) -> ParsedModelDAG:
        while self._curr().type != "EOF":
            start = self.pos
            self._skipped = False
            self._pending_command = None
            stmt_metadata: dict[str, Any] = {}
            before_commands = len(self.commands)
            t = self._curr()

            if t.type == "KEYWORD":
                if t.value == "var":
                    self._advance()
                    deflator_name = None
                    is_log_deflator = False
                    if self._curr().type == "LPAREN":
                        self._advance()
                        while self._curr().type != "RPAREN" and self._curr().type != "EOF":
                            cur = self._curr()
                            if cur.type in ("KEYWORD", "IDENT") and cur.value in ("deflator", "log_deflator"):
                                is_log_deflator = (cur.value == "log_deflator")
                                self._advance()
                                if self._curr().type == "ASSIGN":
                                    self._advance()
                                    if self._curr().type in ("IDENT", "KEYWORD"):
                                        deflator_name = str(self._advance().value)
                            else:
                                self._advance()
                        self._match("RPAREN")
                    new_vars = self._parse_id_list()
                    for v in new_vars:
                        if v in self.shocks or v in self.parameters:
                            raise DynareParseError(
                                f"Symbol '{v}' declared twice", t.line, t.col
                            )
                        if v in self.variables:
                            if deflator_name:
                                if is_log_deflator:
                                    self.log_deflators[v] = deflator_name
                                else:
                                    self.deflators[v] = deflator_name
                            continue
                        self.variables.append(v)
                        if deflator_name:
                            if is_log_deflator:
                                self.log_deflators[v] = deflator_name
                            else:
                                self.deflators[v] = deflator_name
                elif t.value in ("trend_var", "log_trend_var"):
                    is_log = (t.value == "log_trend_var")
                    self._advance()
                    growth_param = None
                    if self._curr().type == "LPAREN":
                        self._advance()
                        if self._curr().type in ("IDENT", "KEYWORD"):
                            growth_param = str(self._advance().value)
                        self._match("RPAREN")
                    new_trends = self._parse_id_list()
                    if is_log:
                        self.log_trend_vars.extend(new_trends)
                    else:
                        self.trend_vars.extend(new_trends)
                    if growth_param:
                        for tr in new_trends:
                            self.deflators[tr] = growth_param
                elif t.value == "varexo":
                    self._advance()
                    new_shocks = self._parse_id_list()
                    for s in new_shocks:
                        if s in self.variables or s in self.shocks or s in self.parameters:
                            raise DynareParseError(
                                f"Symbol '{s}' declared twice", t.line, t.col
                            )
                    self.shocks.extend(new_shocks)
                elif t.value == "parameters":
                    self._advance()
                    new_params = self._parse_id_list()
                    for p in new_params:
                        if p in self.variables or p in self.shocks or p in self.parameters:
                            raise DynareParseError(
                                f"Symbol '{p}' declared twice", t.line, t.col
                            )
                    self.parameters.extend(new_params)
                elif t.value == "predetermined_variables":
                    self._advance()
                    self.predetermined_variables.extend(self._parse_id_list())
                elif t.value == "varexo_det":
                    self._advance()
                    self.det_shocks.extend(self._parse_id_list())
                elif t.value == "varobs":
                    self._advance()
                    self.varobs.extend(self._parse_id_list())
                elif t.value == "model":
                    self._parse_model_block()
                elif t.value == "initval":
                    parsed_init = self._parse_assignment_block("initval")
                    stmt_metadata = {"defs": dict(parsed_init)}
                    if not self._seen_numerical_command or not self.initval_defs:
                        self.initval_defs = parsed_init
                elif t.value == "histval":
                    parsed_hist = self._parse_assignment_block("histval")
                    stmt_metadata = {"defs": dict(parsed_hist)}
                    if not self._seen_numerical_command or not self.histval_defs:
                        self.histval_defs = parsed_hist
                        self._histval_indexed_zero = set(self._indexed_zero_targets)
                    self.commands.append({"command": "histval"})
                elif t.value == "endval":
                    parsed_end = self._parse_assignment_block("endval")
                    stmt_metadata = {"defs": dict(parsed_end)}
                    if not self._seen_numerical_command or not self.endval_defs:
                        self.endval_defs = parsed_end
                    self.commands.append({"command": "endval"})
                elif t.value == "steady_state_model":
                    self.steady_state_defs = self._parse_assignment_block(
                        "steady_state_model"
                    )
                    stmt_metadata = {"defs": dict(self.steady_state_defs)}
                elif t.value == "steady":
                    opts, var_list = self._parse_command_options_and_vars("steady")
                    self.commands.append({"command": "steady", "options": opts, "variables": var_list})
                    self._seen_numerical_command = True
                elif t.value == "shocks":
                    self._parse_shocks_block()
                elif t.value == "stoch_simul":
                    self._parse_stoch_simul()
                    self._seen_numerical_command = True
                elif t.value == "planner_objective":
                    self._parse_planner_objective()
                elif t.value in ("ramsey_policy", "ramsey_model"):
                    self._parse_ramsey_policy(str(t.value))
                    self._seen_numerical_command = True
                elif t.value == "optim_weights":
                    self._parse_optim_weights_block()
                elif t.value == "osr_params":
                    self._advance()
                    self.osr_params.extend(self._parse_id_list())
                elif t.value == "osr_params_bounds":
                    self._parse_osr_params_bounds_block()
                elif t.value == "osr":
                    self._parse_osr_command()
                    self._seen_numerical_command = True
                elif t.value == "calib_smoother":
                    self._parse_calib_smoother_command()
                    self._seen_numerical_command = True
                elif t.value == "shock_decomposition":
                    self._parse_shock_decomposition_command()
                    self._seen_numerical_command = True
                elif t.value in (
                    "perfect_foresight_setup",
                    "perfect_foresight_solver",
                    "simul",
                ):
                    self._parse_perfect_foresight_command(str(t.value))
                    self._seen_numerical_command = True
                elif t.value == "shock_groups":
                    self._parse_shock_groups_block()
                elif t.value == "hetagent_block":
                    self._parse_hetagent_block()
                elif t.value == "markov_switching":
                    self._parse_markov_switching_block()
                elif t.value in (
                    "estimated_params",
                    "estimated_params_init",
                    "estimated_params_bounds",
                ):
                    # Skip tokens of this block; we delegate specialized parsing to _estimated_params
                    self._advance()
                    self._expect("SEMI")
                    while self._curr().type != "EOF":
                        if (
                            self._curr().type == "KEYWORD"
                            and self._curr().value == "end"
                        ):
                            self._advance()
                            self._expect("SEMI")
                            break
                        self._advance()
                else:
                    self._skip_unrecognised_statement()

            elif t.type == "IDENT":
                # Check for parameter assignment outside blocks: IDENT = expr;
                if self._peek_token(1).type == "ASSIGN":
                    pname = str(self._advance().value)
                    self._expect("ASSIGN")
                    expr = self.parse_expression()
                    self._expect("SEMI")
                    stmt_metadata = {"expr": expr}
                    if not self._seen_numerical_command or pname not in self.param_assignments:
                        self.param_assignments[pname] = expr
                else:
                    self._skip_unrecognised_statement()
            elif t.type == "SEMI":
                self._advance()
            else:
                self._skip_unrecognised_statement()

            category = "declaration"
            if self._skipped:
                category = "external/script statement"
            elif t.type == "IDENT":
                category = "calibration assignment"
            elif len(self.commands) > before_commands or self._pending_command is not None:
                category = "command"
            elif t.type == "SEMI":
                category = "empty statement"
            metadata: dict[str, Any] = dict(stmt_metadata)
            if self._pending_command is not None:
                explicit_options, requested_variables = self._pending_command
                metadata.update({"command": str(t.value), "options": explicit_options,
                                 "variables": requested_variables})
            elif len(self.commands) > before_commands:
                metadata.update({"command": self.commands[-1].get("command"),
                                 "options": dict(self.commands[-1].get("options", {})),
                                 "variables": list(self.commands[-1].get("variables", []))})
            self._record_statement(start, category, **metadata)

        if self.collect_only:
            # Syntax collection only: never evaluate calibration, steady state,
            # shock expressions, local definitions, or compile derivatives.
            return ParsedModelDAG(
                variables=self.variables, shocks=self.shocks,
                parameters=self.parameters, parameter_values={}, equations=self.equations,
                is_linear=self.is_linear, varobs=self.varobs, det_shocks=self.det_shocks,
                commands=self.commands, source_statements=self.source_statements,
                hetagent_block=self.hetagent_config, local_variables=self.local_var_defs,
                equation_tags=self.equation_tags,
            )

        # ---------------------------------------------------------------------
        # Post-Processing: Parameter Evaluation
        # ---------------------------------------------------------------------
        # Evaluate parameters sequentially, with multi-pass resolution for dependencies
        eval_scope: dict[str, float] = {}
        for _ in range(len(self.param_assignments) + 1):
            progress = False
            for pname, pexpr in list(self.param_assignments.items()):
                if pname not in eval_scope:
                    try:
                        val = pexpr.eval({}, eval_scope)
                        eval_scope[pname] = float(val)
                        progress = True
                    except (ValueError, ArithmeticError, np.linalg.LinAlgError, KeyError, LookupError, AssertionError, TypeError, RuntimeError, OSError, ConnectionError, Exception):
                        pass
            if not progress:
                break
        self.param_values = eval_scope

        # Resolve parameter references inside command options (e.g. planner_discount=beta)
        for opt_dict in (
            self.ramsey_policy_options,
            self.osr_options,
            self.calib_smoother_options,
            self.shock_decomposition_options,
            self.perfect_foresight_options,
        ):
            for k_opt, v_opt in list(opt_dict.items()):
                if isinstance(v_opt, str) and v_opt in self.param_values:
                    opt_dict[k_opt] = float(self.param_values[v_opt])

        # Evaluate osr_params_bounds expressions
        evaluated_osr_bounds: dict[str, tuple[float, float]] = {}
        for pname, (lb_node, ub_node) in self.osr_params_bounds.items():
            try:
                lb_val = float(lb_node.eval({}, self.param_values))
                ub_val = float(ub_node.eval({}, self.param_values))
                evaluated_osr_bounds[pname] = (lb_val, ub_val)
            except Exception:
                pass

        # ---------------------------------------------------------------------
        # Post-Processing: Model-Local '#' Variables Inlining & Cycle Detection
        # ---------------------------------------------------------------------
        resolved_locals: dict[str, Node] = {}
        if self.local_var_defs:
            local_names = set(self.local_var_defs.keys())

            # Build dependency graph between local variables
            deps: dict[str, set[str]] = {
                k: _find_local_deps(v, local_names)
                for k, v in self.local_var_defs.items()
            }

            # Detect circular dependencies using Tarjan / DFS
            visited: dict[str, int] = {}  # 0 = visiting, 1 = visited
            order: list[str] = []
            cycle: list[str] = []

            def dfs(u: str, path: list[str]) -> bool:
                visited[u] = 0
                path.append(u)
                for v in sorted(deps.get(u, set())):
                    if v not in local_names:
                        continue
                    if visited.get(v) == 0:
                        cycle_start = path.index(v)
                        cycle.extend(path[cycle_start:] + [v])
                        return True
                    if visited.get(v) is None:
                        if dfs(v, path):
                            return True
                path.pop()
                visited[u] = 1
                order.append(u)
                return False

            for loc_k in sorted(local_names):
                if loc_k not in visited:
                    if dfs(loc_k, []):
                        raise ModelError(
                            f"circular dependency in model-local variables: {' -> '.join(cycle)}"
                        )

            # Inline dependencies into other local variables in topological order.
            #
            # A local is a *symbolic* shorthand (Dynare Reference Manual, 4.5:
            # "every time this variable appears in the model, Dynare will
            # substitute it by the expression assigned to the variable"). It
            # must never be folded to a number at the parse-time calibration,
            # even when it involves parameters only: the compiled equations and
            # their derivatives are re-evaluated at new parameter values by
            # estimation, load_mod(params=), osr, widgets and gradients, and a
            # folded local would keep its load-time value there (SW07's
            # constebeta entered the model only through locals, so its
            # likelihood was flat). ``simplify`` still folds the purely numeric
            # sub-expressions; Param nodes stay, and CSE in compile_derivatives
            # evaluates each repeated parameter expression once per call.
            for u in order:
                inlined_ast = _substitute_locals(
                    self.local_var_defs[u], resolved_locals
                ).simplify()
                resolved_locals[u] = inlined_ast

            # Inline resolved locals into all model equations
            inlined_equations = []
            for eq in self.equations:
                inlined_eq = _substitute_locals(eq, resolved_locals).simplify()
                inlined_equations.append(inlined_eq)
            self.equations = inlined_equations

            if self.planner_objective is not None:
                self.planner_objective = _substitute_locals(
                    self.planner_objective, resolved_locals
                ).simplify()

        # Check after local expansion as well: a model-local shorthand may
        # hide an exogenous variable inside STEADY_STATE. Native Dynare rejects
        # this construct; silently using zero changes the declared model.
        def validate_steady_state(node: Node) -> None:
            if isinstance(node, Call):
                if node.func.lower() == "steady_state":
                    forbidden = {name for name, _ in node.variables()} & set(self.shocks + self.det_shocks)
                    if forbidden:
                        raise ModelError(
                            "STEADY_STATE() cannot contain exogenous variables: "
                            + ", ".join(sorted(forbidden))
                        )
                for arg in node.args:
                    validate_steady_state(arg)
            elif isinstance(node, BinOp):
                validate_steady_state(node.left)
                validate_steady_state(node.right)
            elif isinstance(node, UnaryOp):
                validate_steady_state(node.expr)

        for equation in self.equations:
            validate_steady_state(equation)

        # ---------------------------------------------------------------------
        # Post-Processing: Multi-Period Lead/Lag & Dated Exogenous Shock Expansion
        # ---------------------------------------------------------------------
        lag_leads_needed: set[tuple[str, int]] = set()
        exo_lag_leads_needed: set[tuple[str, int]] = set()
        for eq in self.equations:
            for v_name, v_lead in eq.variables():
                if abs(v_lead) >= 2 and v_name in self.variables:
                    lag_leads_needed.add((v_name, v_lead))
                elif v_lead != 0 and v_name in self.shocks:
                    exo_lag_leads_needed.add((v_name, v_lead))

        aux_vars: list[str] = []
        aux_eqs: list[Node] = []
        replacements: dict[tuple[str, int], Node] = {}
        auxiliary: dict[str, dict[str, Any]] = {}

        for v, offset in sorted(lag_leads_needed, key=lambda x: (x[0], abs(x[1]))):
            if offset <= -2:
                k = abs(offset)
                for step in range(1, k):
                    aux_name = f"AUX_LAG_{v}_{step}"
                    if aux_name not in aux_vars:
                        aux_vars.append(aux_name)
                        auxiliary[aux_name] = {"root": v, "kind": "lag", "step": step}
                        prev_node = (
                            Var(v, -1)
                            if step == 1
                            else Var(f"AUX_LAG_{v}_{step-1}", -1)
                        )
                        aux_eq = BinOp("-", Var(aux_name, 0), prev_node)
                        aux_eqs.append(aux_eq)
                replacements[(v, offset)] = Var(f"AUX_LAG_{v}_{k-1}", -1)
            elif offset >= 2:
                k = offset
                for step in range(1, k):
                    aux_name = f"AUX_LEAD_{v}_{step}"
                    if aux_name not in aux_vars:
                        aux_vars.append(aux_name)
                        auxiliary[aux_name] = {"root": v, "kind": "lead", "step": step}
                        prev_node = (
                            Var(v, 1) if step == 1 else Var(f"AUX_LEAD_{v}_{step-1}", 1)
                        )
                        aux_eq = BinOp("-", Var(aux_name, 0), prev_node)
                        aux_eqs.append(aux_eq)
                replacements[(v, offset)] = Var(f"AUX_LEAD_{v}_{k-1}", 1)

        # Dated exogenous shocks e(-k) and e(+k) (including through '#' model-local variables)
        for s_name, offset in sorted(exo_lag_leads_needed, key=lambda x: (x[0], abs(x[1]))):
            if offset <= -1:
                k = abs(offset)
                for step in range(1, k):
                    aux_name = f"AUX_EXO_LAG_{s_name}_{step}"
                    if aux_name not in aux_vars:
                        aux_vars.append(aux_name)
                        auxiliary[aux_name] = {
                            "root": s_name,
                            "kind": "exo_lag",
                            "step": step,
                        }
                        prev_node = (
                            Var(s_name, 0)
                            if step == 1
                            else Var(f"AUX_EXO_LAG_{s_name}_{step-1}", -1)
                        )
                        aux_eq = BinOp("-", Var(aux_name, 0), prev_node)
                        aux_eqs.append(aux_eq)
                # Final auxiliary variable AUX_EXO_LAG_{s}_{k} only needs to exist if k >= 1
                # Wait: for e(-1), AUX_EXO_LAG_{s}_1(0) = s(0), and e(-1) -> AUX_EXO_LAG_{s}_1(-1)!
                # For e(-k), we need steps 1..k, where step 1 = s(0), step j = step_{j-1}(-1),
                # and e(-k) -> AUX_EXO_LAG_{s}_{k}(-1) (Wait: if step 1 = s(0) at t, then step 1(-1) = s(-1);
                # step 2(0) = step 1(-1) = s(-1), so step 2(-1) = s(-2);
                # step k(0) = step_{k-1}(-1) = s(-(k-1)), so step k(-1) = s(-k)!)
                aux_k = f"AUX_EXO_LAG_{s_name}_{k}"
                if aux_k not in aux_vars:
                    aux_vars.append(aux_k)
                    auxiliary[aux_k] = {
                        "root": s_name,
                        "kind": "exo_lag",
                        "step": k,
                    }
                    prev_node = (
                        Var(s_name, 0)
                        if k == 1
                        else Var(f"AUX_EXO_LAG_{s_name}_{k-1}", -1)
                    )
                    aux_eq = BinOp("-", Var(aux_k, 0), prev_node)
                    aux_eqs.append(aux_eq)
                replacements[(s_name, offset)] = Var(aux_k, -1)
            elif offset >= 1:
                k = offset
                for step in range(1, k + 1):
                    aux_name = f"AUX_EXO_LEAD_{s_name}_{step}"
                    if aux_name not in aux_vars:
                        aux_vars.append(aux_name)
                        auxiliary[aux_name] = {
                            "root": s_name,
                            "kind": "exo_lead",
                            "step": step,
                        }
                        prev_node = (
                            Var(s_name, 0)
                            if step == 1
                            else Var(f"AUX_EXO_LEAD_{s_name}_{step-1}", 1)
                        )
                        aux_eq = BinOp("-", Var(aux_name, 0), prev_node)
                        aux_eqs.append(aux_eq)
                replacements[(s_name, offset)] = Var(
                    f"AUX_EXO_LEAD_{s_name}_{k}", 1
                )

        if replacements:
            new_eqs = [_substitute_vars(eq, replacements) for eq in self.equations]
            self.equations = new_eqs + aux_eqs
            self.variables = list(self.variables) + aux_vars

        # ---------------------------------------------------------------------
        # Post-Processing: Steady State & Initval Evaluation
        # ---------------------------------------------------------------------
        # Both blocks are functions of the parameters; the same helpers back
        # ParsedModelDAG.evaluate_steady_state / evaluate_initval, which
        # re-run them at other parameter values.
        steady_state: dict[str, float] | None = None
        if self.steady_state_defs:
            steady_state = _eval_steady_state_defs(
                self.steady_state_defs, self.param_values, self.variables, auxiliary
            )

        guess: dict[str, float] | None = None
        if self.initval_defs:
            guess = _eval_initval_defs(
                self.initval_defs, self.param_values, self.variables, auxiliary
            )

        histval_values: dict[str, float] | None = None
        if self.histval_defs:
            h_scope: dict[tuple[str, int] | str, float] = {}
            h_scope.update(self.param_values)
            for h_name, h_expr in self.histval_defs.items():
                try:
                    p_scope = {**self.param_values, **{k: v for k, v in h_scope.items() if isinstance(k, str)}}
                    val = float(h_expr.eval(h_scope, p_scope))
                    h_scope[h_name] = val
                    h_scope[(h_name, 0)] = val
                    if h_name in auxiliary:
                        info = auxiliary[h_name]
                        if info.get("kind") == "lag":
                            h_scope[(info["root"], -int(info["step"]))] = val
                        elif info.get("kind") == "exo_lag":
                            h_scope[(info["root"], -(int(info["step"]) - 1))] = val
                except (ValueError, ArithmeticError, np.linalg.LinAlgError, KeyError, LookupError, AssertionError, TypeError, RuntimeError, OSError, ConnectionError, Exception):
                    pass
            histval_values = {
                v: h_scope[v]
                for v in self.histval_defs
                if v in h_scope and isinstance(v, str)
            }
            for h_name in list(histval_values):
                if h_name.startswith("AUX_LAG_") and h_name not in auxiliary:
                    m_lag = re.fullmatch(r"AUX_LAG_(.+)_1", h_name)
                    if m_lag and m_lag.group(1) in self.variables and m_lag.group(1) not in histval_values:
                        histval_values[m_lag.group(1)] = histval_values.pop(h_name)
            indexed_zero: set[str] = getattr(self, "_histval_indexed_zero", set())
            for aux_name, info in auxiliary.items():
                if info.get("kind") == "lag" and aux_name not in histval_values:
                    root = info["root"]
                    if root in histval_values and root not in indexed_zero:
                        histval_values[aux_name] = histval_values[root]

        endval_values: dict[str, float] | None = None
        if self.endval_defs:
            e_scope: dict[tuple[str, int] | str, float] = {}
            e_scope.update(self.param_values)
            for e_name, e_expr in self.endval_defs.items():
                try:
                    p_scope = {**self.param_values, **{k: v for k, v in e_scope.items() if isinstance(k, str)}}
                    val = float(e_expr.eval(e_scope, p_scope))
                    e_scope[e_name] = val
                    e_scope[(e_name, 0)] = val
                    if e_name in auxiliary:
                        info = auxiliary[e_name]
                        if info.get("kind") == "lead":
                            e_scope[(info["root"], int(info["step"]))] = val
                except (ValueError, ArithmeticError, np.linalg.LinAlgError, KeyError, LookupError, AssertionError, TypeError, RuntimeError, OSError, ConnectionError, Exception):
                    pass
            endval_values = {
                v: e_scope[v]
                for v in self.endval_defs
                if v in e_scope and isinstance(v, str)
            }
            for aux_name, info in auxiliary.items():
                if info.get("kind") in ("lag", "lead") and aux_name not in endval_values:
                    root = info["root"]
                    if root in endval_values:
                        endval_values[aux_name] = endval_values[root]

        # ---------------------------------------------------------------------
        # Post-Processing: Shock Covariance Matrix
        # ---------------------------------------------------------------------
        shock_cov: np.ndarray | None = None
        if self.has_shocks_block:
            n_e = len(self.shocks)
            shock_cov = np.zeros((n_e, n_e))
            variance_declared: set[str] = set()
            for s_name, s_expr in self.shocks_config["stderrs"].items():
                if s_name in self.shocks:
                    idx = self.shocks.index(s_name)
                    val = s_expr.eval({}, self.param_values)
                    shock_cov[idx, idx] = float(val) ** 2
                    variance_declared.add(s_name)
            for s_name, s_expr in self.shocks_config["variances"].items():
                if s_name in self.shocks:
                    idx = self.shocks.index(s_name)
                    val = s_expr.eval({}, self.param_values)
                    shock_cov[idx, idx] = float(val)
                    variance_declared.add(s_name)
            for (s1, s2), c_expr in self.shocks_config["covariances"].items():
                if s1 in self.shocks and s2 in self.shocks:
                    i1, i2 = self.shocks.index(s1), self.shocks.index(s2)
                    val = float(c_expr.eval({}, self.param_values))
                    shock_cov[i1, i2] = val
                    shock_cov[i2, i1] = val
            for (s1, s2), r_expr in self.shocks_config["correlations"].items():
                if s1 in self.shocks and s2 in self.shocks:
                    i1, i2 = self.shocks.index(s1), self.shocks.index(s2)
                    rho = float(r_expr.eval({}, self.param_values))
                    cov = rho * math.sqrt(shock_cov[i1, i1] * shock_cov[i2, i2])
                    shock_cov[i1, i2] = cov
                    shock_cov[i2, i1] = cov
            undeclared = [sh for sh in self.shocks if sh not in variance_declared]
            if undeclared:
                warnings.warn(
                    f"parse_mod: the shocks; block declares no variance for "
                    f"{undeclared} — following Dynare (M_.Sigma_e starts at zero), "
                    "their innovation variance is 0, so they are inert in the "
                    "moments, the IRFs and the second-order risk correction ghs2. "
                    "Add 'var <name>; stderr <value>;' to the shocks; block if that "
                    "is not what you meant.",
                    UserWarning,
                    stacklevel=2,
                )
        elif self.shocks:
            shock_cov = np.eye(len(self.shocks))

        # ---------------------------------------------------------------------
        # Post-Processing: Estimated Parameters
        # ---------------------------------------------------------------------
        estimated_params = None
        estimated_params_init = None
        estimated_params_bounds = None
        if self.raw_text:
            if "estimated_params" in self.raw_text:
                try:
                    estimated_params = parse_estimated_params(
                        self.raw_text,
                        shocks=self.shocks,
                        varobs=self.varobs,
                        params=sorted(self.parameters),
                        variables=self.variables,
                    )
                except (ValueError, ArithmeticError, np.linalg.LinAlgError, KeyError, LookupError, AssertionError, TypeError, RuntimeError, OSError, ConnectionError, Exception):
                    pass
            if "estimated_params_init" in self.raw_text:
                try:
                    estimated_params_init = parse_estimated_params_init(
                        self.raw_text, shocks=self.shocks, varobs=self.varobs
                    )
                except (ValueError, ArithmeticError, np.linalg.LinAlgError, KeyError, LookupError, AssertionError, TypeError, RuntimeError, OSError, ConnectionError, Exception):
                    pass
            if "estimated_params_bounds" in self.raw_text:
                try:
                    estimated_params_bounds = parse_estimated_params_bounds(
                        self.raw_text, shocks=self.shocks, varobs=self.varobs
                    )
                except (ValueError, ArithmeticError, np.linalg.LinAlgError, KeyError, LookupError, AssertionError, TypeError, RuntimeError, OSError, ConnectionError, Exception):
                    pass

        is_linear = self.is_linear
        if not is_linear:
            try:
                from puremacro.dsge._utils import detect_linear_model
                is_linear = detect_linear_model(self)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError, KeyError, LookupError, AssertionError, TypeError, RuntimeError, OSError, ConnectionError, Exception):
                pass

        return ParsedModelDAG(
            variables=self.variables,
            shocks=self.shocks,
            parameters=self.parameters,
            parameter_values=self.param_values,
            equations=self.equations,
            equation_tags=self.equation_tags,
            is_linear=is_linear,
            steady_state_model=(
                self.steady_state_defs if self.steady_state_defs else None
            ),
            initval=self.initval_defs if self.initval_defs else None,
            shocks_config=self.shocks_config,
            predetermined_variables=(
                self.predetermined_variables if self.predetermined_variables else None
            ),
            varobs=self.varobs if self.varobs else None,
            options=self.options,
            estimated_params=estimated_params,
            estimated_params_init=estimated_params_init,
            estimated_params_bounds=estimated_params_bounds,
            local_variables=resolved_locals,
            auxiliary_variables=auxiliary,
            tex_labels=self.tex_labels,
            shock_cov=shock_cov,
            steady_state=steady_state,
            guess=guess,
            det_shocks=self.det_shocks,
            histval=self.histval_defs if self.histval_defs else None,
            endval=self.endval_defs if self.endval_defs else None,
            histval_values=histval_values,
            endval_values=endval_values,
            trend_vars=self.trend_vars,
            deflators=self.deflators,
            log_trend_vars=self.log_trend_vars,
            log_deflators=self.log_deflators,
            shock_groups=self.shock_groups,
            named_shock_groups=self.named_shock_groups,
            hetagent_block=self.hetagent_config,
            markov_switching_config=self.markov_switching_config,
            planner_objective=self.planner_objective,
            ramsey_policy_options=self.ramsey_policy_options,
            optim_weights=self.optim_weights,
            osr_params=self.osr_params,
            osr_params_bounds=evaluated_osr_bounds,
            osr_options=self.osr_options,
            calib_smoother_options=self.calib_smoother_options,
            shock_decomposition_options=self.shock_decomposition_options,
            perfect_foresight_options=self.perfect_foresight_options,
            commands=self.commands,
            source_statements=self.source_statements,
            param_assignments=dict(self.param_assignments),
        )


def parse_mod_to_dag(text: str, base_dir: Path | str | None = None) -> ParsedModelDAG:
    """Parse Dynare .mod text into a ParsedModelDAG expression graph.

    Parameters
    ----------
    text : str
        Source text of a Dynare .mod file.
    base_dir : Path or str, optional
        Base directory for macro processor file inclusion resolution.

    Returns
    -------
    ParsedModelDAG
        Structured model container with AST equations, variables, parameters,
        shocks, and diagnostics.
    """
    # If text uses macro directives (@# or @{) and preprocess_macro is available, run pre-pass
    if "@#" in text or "@{" in text:
        if preprocess_macro is not None:
            text = preprocess_macro(text, base_dir=base_dir)
        else:
            found = re.search(r"^[^\S\n]*@#\s*\w+", text, re.M) or re.search(
                r"@\{", text
            )
            if found:
                raise DynareFeatureError(
                    f"this .mod file uses the Dynare macro processor ({found.group(0).strip()!r}); "
                    "preprocessor module not yet loaded."
                )

    tokenizer = Tokenizer(text)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens, raw_text=text)
    return parser.parse()
