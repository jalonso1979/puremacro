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

from puremacro.dsge._ast import BinOp, Call, Const, Node, Param, UnaryOp, Var
from puremacro.dsge.build import ModelError
from puremacro.dsge._estimated_params import (
    parse_estimated_params,
    parse_estimated_params_bounds,
    parse_estimated_params_init,
)

# Optional import of macro preprocessor (Worker M1)
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
}


class Tokenizer:
    """Pure-Python character-by-character tokenizer for Dynare .mod source text."""

    def __init__(self, text: str):
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
                while self.pos < self.n and self._peek() != "\n":
                    self._advance()
                continue

            # Block comments: /* ... */
            if ch == "/" and self._peek(1) == "*":
                self._advance()  # /
                self._advance()  # *
                while self.pos < self.n and not (
                    self._peek() == "*" and self._peek(1) == "/"
                ):
                    self._advance()
                if self.pos < self.n:
                    self._advance()  # *
                    self._advance()  # /
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
                buf = []
                while self.pos < self.n:
                    c = self._advance()
                    if c == quote_char:
                        break
                    if c == "\\" and self.pos < self.n:
                        buf.append(self._advance())
                    else:
                        buf.append(c)
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


@dataclass
class ParsedModelDAG:
    """Structured representation of a parsed Dynare DSGE model as an Expression DAG."""

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

    @property
    def varexo_det(self) -> list[str]:
        return self.det_shocks

    @varexo_det.setter
    def varexo_det(self, val: list[str]) -> None:
        self.det_shocks = val

    def compile_equations(self) -> Callable:
        """Compile AST equations into a Python callable compatible with legacy LinearModel."""
        py_exprs = [
            eq.to_python(
                lead_ns="lead",
                curr_ns="curr",
                lag_ns="lag",
                shock_ns="shocks",
                param_ns="params",
                shock_names=set(self.shocks),
            )
            for eq in self.equations
        ]
        code_body = (
            "def _generated_equations(lead, curr, lag, shocks, params):\n    return [\n"
        )
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
        }


class Parser:
    """Recursive-descent parser building expression DAGs from token streams."""

    def __init__(self, tokens: list[Token], raw_text: str = ""):
        self.tokens = tokens
        self.pos = 0
        self.raw_text = raw_text

        # Symbol tables
        self.variables: list[str] = []
        self.shocks: list[str] = []
        self.det_shocks: list[str] = []
        self.parameters: list[str] = []
        self.predetermined_variables: list[str] = []
        self.varobs: list[str] = []
        self.param_assignments: dict[str, Node] = {}
        self.param_values: dict[str, float] = {}

        # Model contents
        self.equations: list[Node] = []
        self.equation_tags: dict[int, dict[str, str]] = {}
        self.local_var_defs: dict[str, Node] = {}
        self.local_var_names: set[str] = set()
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
        }
        self.has_shocks_block: bool = False
        self.options: dict[str, Any] = {}
        self.trend_vars: list[str] = []
        self.log_trend_vars: list[str] = []
        self.deflators: dict[str, str] = {}
        self.log_deflators: dict[str, str] = {}
        self.shock_groups: dict[str, list[str]] = {}
        self.named_shock_groups: dict[str, dict[str, list[str]]] = {}

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
                is_math_fn = (
                    ident_lower in MATH_FUNCTIONS
                    and ident not in self.variables
                    and ident not in self.local_var_names
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
                        if ident in self.local_var_names:
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
                        if ident in self.local_var_names:
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
            if ident in self.local_var_names:
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
                continue

            # Regular model equation: lhs = rhs; or expr;
            lhs = self.parse_expression()
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

        if not saw_end:
            raise DynareParseError(
                "Unclosed model block: expected 'end;'",
                self._curr().line,
                self._curr().col,
            )

    def _parse_assignment_block(self, block_name: str) -> dict[str, Node]:
        """Parse assignment block (initval, histval, endval, steady_state_model) storing AST expressions."""
        self._expect("KEYWORD", block_name)
        self._expect("SEMI")
        defs: dict[str, Node] = {}
        while self._curr().type != "EOF":
            if self._curr().type == "KEYWORD" and self._curr().value == "end":
                self._advance()
                self._expect("SEMI")
                break
            if self._curr().type in ("IDENT", "KEYWORD"):
                name = str(self._advance().value)
                if self._match("LPAREN"):
                    # Handle possible lead/lag index like y(0) or y(-1)
                    if self._curr().type in ("NUMBER", "INT", "MINUS", "PLUS"):
                        if self._curr().type in ("MINUS", "PLUS"):
                            self._advance()
                        if self._curr().type in ("NUMBER", "INT"):
                            self._advance()
                    self._expect("RPAREN")

                self._expect("ASSIGN")
                expr = self.parse_expression()
                self._expect("SEMI")
                defs[name] = expr
            else:
                self._advance()
        return defs

    def _parse_shocks_block(self) -> None:
        """Parse shocks; block."""
        self._expect("KEYWORD", "shocks")
        if self._match("LPAREN"):
            if self._curr().type in ("IDENT", "KEYWORD"):
                self.shocks_config["type"] = str(self._advance().value)
            self._expect("RPAREN")
        self._expect("SEMI")
        self.has_shocks_block = True

        while self._curr().type != "EOF":
            if self._curr().type == "KEYWORD" and self._curr().value == "end":
                self._advance()
                self._expect("SEMI")
                break

            if self._match("KEYWORD", "var"):
                shock1 = str(self._expect("IDENT").value)
                if self._match("COMMA"):
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
                    if (
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
                self._advance()

    def _parse_stoch_simul(self) -> None:
        """Parse stoch_simul options."""
        self._expect("KEYWORD", "stoch_simul")
        if self._match("LPAREN"):
            while self._curr().type != "RPAREN" and self._curr().type != "EOF":
                if self._curr().type in ("IDENT", "KEYWORD"):
                    opt_name = str(self._advance().value)
                    if self._match("ASSIGN"):
                        if self._curr().type == "NUMBER":
                            self.options[opt_name] = self._advance().value
                        elif self._curr().type in ("IDENT", "KEYWORD", "STRING"):
                            self.options[opt_name] = self._advance().value
                    else:
                        self.options[opt_name] = True
                self._match("COMMA")
            self._expect("RPAREN")
        # Consume any trailing variable list until ';'
        while self._curr().type != "SEMI" and self._curr().type != "EOF":
            self._advance()
        self._expect("SEMI")

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

    def _skip_unrecognised_statement(self) -> None:
        """Safely skip MATLAB scripting or unhandled commands outside model blocks."""
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

    def parse(self) -> ParsedModelDAG:
        while self._curr().type != "EOF":
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
                    self.initval_defs = self._parse_assignment_block("initval")
                elif t.value == "histval":
                    self.histval_defs = self._parse_assignment_block("histval")
                elif t.value == "endval":
                    self.endval_defs = self._parse_assignment_block("endval")
                elif t.value == "steady_state_model":
                    self.steady_state_defs = self._parse_assignment_block(
                        "steady_state_model"
                    )
                elif t.value == "shocks":
                    self._parse_shocks_block()
                elif t.value == "stoch_simul":
                    self._parse_stoch_simul()
                elif t.value == "shock_groups":
                    self._parse_shock_groups_block()
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
                    self.param_assignments[pname] = expr
                else:
                    self._skip_unrecognised_statement()
            elif t.type == "SEMI":
                self._advance()
            else:
                self._skip_unrecognised_statement()

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
                    except Exception:
                        pass
            if not progress:
                break
        self.param_values = eval_scope

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

            for k in sorted(local_names):
                if k not in visited:
                    if dfs(k, []):
                        raise ModelError(
                            f"circular dependency in model-local variables: {' -> '.join(cycle)}"
                        )

            # Inline dependencies into other local variables in topological order
            for u in order:
                inlined_ast = _substitute_locals(
                    self.local_var_defs[u], resolved_locals
                )
                # If this local variable depends only on parameters and constants, evaluate or simplify
                if not inlined_ast.variables():
                    try:
                        c_val = inlined_ast.eval({}, self.param_values)
                        inlined_ast = Const(c_val)
                    except Exception:
                        inlined_ast = inlined_ast.simplify()
                else:
                    inlined_ast = inlined_ast.simplify()
                resolved_locals[u] = inlined_ast

            # Inline resolved locals into all model equations
            inlined_equations = []
            for eq in self.equations:
                inlined_eq = _substitute_locals(eq, resolved_locals).simplify()
                inlined_equations.append(inlined_eq)
            self.equations = inlined_equations

        # ---------------------------------------------------------------------
        # Post-Processing: Multi-Period Lead/Lag Expansion (|lead| >= 2)
        # ---------------------------------------------------------------------
        lag_leads_needed: set[tuple[str, int]] = set()
        for eq in self.equations:
            for v_name, v_lead in eq.variables():
                if abs(v_lead) >= 2 and v_name in self.variables:
                    lag_leads_needed.add((v_name, v_lead))

        aux_vars: list[str] = []
        aux_eqs: list[Node] = []
        replacements: dict[tuple[str, int], Node] = {}

        for v, offset in sorted(lag_leads_needed, key=lambda x: (x[0], abs(x[1]))):
            if offset <= -2:
                k = abs(offset)
                for step in range(1, k):
                    aux_name = f"AUX_LAG_{v}_{step}"
                    if aux_name not in aux_vars:
                        aux_vars.append(aux_name)
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
                        prev_node = (
                            Var(v, 1) if step == 1 else Var(f"AUX_LEAD_{v}_{step-1}", 1)
                        )
                        aux_eq = BinOp("-", Var(aux_name, 0), prev_node)
                        aux_eqs.append(aux_eq)
                replacements[(v, offset)] = Var(f"AUX_LEAD_{v}_{k-1}", 1)

        if replacements:
            new_eqs = [_substitute_vars(eq, replacements) for eq in self.equations]
            self.equations = new_eqs + aux_eqs
            self.variables = list(self.variables) + aux_vars

        # ---------------------------------------------------------------------
        # Post-Processing: Steady State & Initval Evaluation
        # ---------------------------------------------------------------------
        steady_state: dict[str, float] | None = None
        if self.steady_state_defs:
            ss_scope: dict[str, float] = dict(self.param_values)
            for s_name, s_expr in self.steady_state_defs.items():
                try:
                    val = s_expr.eval(ss_scope, {**self.param_values, **ss_scope})
                    ss_scope[s_name] = float(val)
                except Exception as exc:
                    raise ValueError(
                        f"could not evaluate '{s_name}' in the steady_state_model block: {exc}"
                    ) from exc
            steady_state = {v: ss_scope.get(v, 0.0) for v in self.variables}
            for aux in aux_vars:
                parts = aux.split("_")
                root_var = parts[2] if len(parts) >= 3 else aux
                steady_state[aux] = steady_state.get(root_var, 0.0)

        guess: dict[str, float] | None = None
        if self.initval_defs:
            init_scope: dict[str, float] = dict(self.param_values)
            for i_name, i_expr in self.initval_defs.items():
                try:
                    val = i_expr.eval(init_scope, {**self.param_values, **init_scope})
                    init_scope[i_name] = float(val)
                except Exception:
                    pass
            guess = {v: init_scope[v] for v in self.variables if v in init_scope}
            for aux in aux_vars:
                parts = aux.split("_")
                root_var = parts[2] if len(parts) >= 3 else aux
                guess[aux] = guess.get(root_var, 1.0)

        histval_values: dict[str, float] | None = None
        if self.histval_defs:
            h_scope = dict(self.param_values)
            for h_name, h_expr in self.histval_defs.items():
                try:
                    val = h_expr.eval(h_scope, {**self.param_values, **h_scope})
                    h_scope[h_name] = float(val)
                except Exception:
                    pass
            histval_values = {
                v: h_scope[v]
                for v in (self.variables + self.shocks + self.det_shocks + self.parameters)
                if v in h_scope
            }

        endval_values: dict[str, float] | None = None
        if self.endval_defs:
            e_scope = dict(self.param_values)
            for e_name, e_expr in self.endval_defs.items():
                try:
                    val = e_expr.eval(e_scope, {**self.param_values, **e_scope})
                    e_scope[e_name] = float(val)
                except Exception:
                    pass
            endval_values = {
                v: e_scope[v]
                for v in (self.variables + self.shocks + self.det_shocks + self.parameters)
                if v in e_scope
            }

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
                except Exception:
                    pass
            if "estimated_params_init" in self.raw_text:
                try:
                    estimated_params_init = parse_estimated_params_init(
                        self.raw_text, shocks=self.shocks, varobs=self.varobs
                    )
                except Exception:
                    pass
            if "estimated_params_bounds" in self.raw_text:
                try:
                    estimated_params_bounds = parse_estimated_params_bounds(
                        self.raw_text, shocks=self.shocks, varobs=self.varobs
                    )
                except Exception:
                    pass

        is_linear = self.is_linear
        if not is_linear:
            try:
                from puremacro.dsge._utils import detect_linear_model
                is_linear = detect_linear_model(self)
            except Exception:
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
