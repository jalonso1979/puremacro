"""Dynare macro preprocessor for puremacro.

Implements Dynare's macro preprocessor (@# directives and @{...} interpolation)
as a separate pre-pass prior to model parsing.

Features:
- @#define IDENT (= EXPR)? and macro functions @#define F(arg1, ...) = EXPR
- @#undef IDENT
- @#for IDENT in EXPR (when COND)? ... @#endfor (over ranges, arrays, and tuples)
- Destructuring @#for (v1, v2) in EXPR ... @#endfor
- @#if EXPR ... (@#elseif EXPR ...)* (@#else ...)? @#endif
- @#ifdef IDENT ... @#endif and @#ifndef IDENT ... @#endif
- @#include STRING_OR_EXPR and @#includepath STRING_OR_EXPR
- Circular @#include detection tracking canonical include stack
- Inline interpolation @{EXPR}
- Array comprehensions [expr for i in arr when cond]
- Rich built-in functions: math, collections, type checks, and defined()
- Pure Python standard library only (zero external runtime dependencies).
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import re
from typing import Any, Callable, Sequence


class DynareMacroError(Exception):
    """Exception raised for errors during Dynare macro preprocessing."""

    def __init__(
        self,
        message: str,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        self.message = message
        self.line = line
        self.column = column
        loc_parts = []
        if line is not None:
            loc_parts.append(f"line {line}")
        if column is not None:
            loc_parts.append(f"column {column}")
        loc_str = f" (at {', '.join(loc_parts)})" if loc_parts else ""
        super().__init__(f"{message}{loc_str}")

    @property
    def col(self) -> int | None:
        """Alias for column."""
        return self.column


class Scope:
    """Lexical scope frame supporting nested parent lookups and variable restoration."""

    def __init__(self, parent: Scope | None = None) -> None:
        self.parent = parent
        self.vars: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        if name in self.vars:
            return self.vars[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise KeyError(name)

    def is_defined(self, name: str) -> bool:
        if name in self.vars:
            return True
        if self.parent is not None:
            return self.parent.is_defined(name)
        return False

    def set(self, name: str, value: Any) -> None:
        """Assign to an existing variable in the nearest enclosing scope, or locally."""
        curr: Scope | None = self
        while curr is not None:
            if name in curr.vars:
                curr.vars[name] = value
                return
            curr = curr.parent
        self.vars[name] = value

    def set_local(self, name: str, value: Any) -> None:
        """Explicitly set variable in the local frame (used for loop iteration variables)."""
        self.vars[name] = value

    def delete(self, name: str) -> None:
        """Delete variable from the nearest enclosing scope where it is defined."""
        curr: Scope | None = self
        while curr is not None:
            if name in curr.vars:
                del curr.vars[name]
                return
            curr = curr.parent
        raise KeyError(name)

    def all_vars(self) -> dict[str, Any]:
        """Return a merged dictionary of all visible variables in scope."""
        res = self.parent.all_vars() if self.parent is not None else {}
        res.update(self.vars)
        return res


def _cbrt(x: float | int) -> float:
    if hasattr(math, "cbrt"):
        return math.cbrt(x)
    return x ** (1.0 / 3.0) if x >= 0 else -((-x) ** (1.0 / 3.0))


BUILTIN_FUNCS: dict[str, Callable[..., Any]] = {
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "cbrt": _cbrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "trunc": math.trunc,
    "abs": abs,
    "sign": lambda x: 1 if x > 0 else (-1 if x < 0 else 0),
    "mod": lambda x, y: x % y,
    "erf": math.erf,
    "erfc": math.erfc,
    "gamma": math.gamma,
    "lgamma": math.lgamma,
    "normpdf": lambda x: (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x),
    "normcdf": lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))),
    "max": lambda *args: max(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else max(args),
    "min": lambda *args: min(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else min(args),
    "length": len,
    "empty": lambda x: len(x) == 0,
    "sum": sum,
    "isboolean": lambda x: isinstance(x, bool),
    "isreal": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
    "isstring": lambda x: isinstance(x, str),
    "istuple": lambda x: isinstance(x, tuple),
    "isarray": lambda x: isinstance(x, list),
}

BUILTIN_CONSTS: dict[str, Any] = {
    "true": True,
    "false": False,
    "True": True,
    "False": False,
    "nan": float("nan"),
    "inf": float("inf"),
    "pi": math.pi,
}


# ---------------------------------------------------------------------------
# Macro Expression AST Nodes
# ---------------------------------------------------------------------------

class ExprNode:
    """Base class for macro expression AST nodes."""

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        raise NotImplementedError


class LiteralNode(ExprNode):

    def __init__(self, value: Any) -> None:
        self.value = value

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        return self.value


class IdentifierNode(ExprNode):

    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        if self.name in BUILTIN_CONSTS:
            return BUILTIN_CONSTS[self.name]
        if scope.is_defined(self.name):
            return scope.get(self.name)
        if self.name in BUILTIN_FUNCS:
            return BUILTIN_FUNCS[self.name]
        raise DynareMacroError(f"Undefined macro variable '{self.name}'", line, col)


class UnaryOpNode(ExprNode):

    def __init__(self, op: str, expr: ExprNode) -> None:
        self.op = op
        self.expr = expr

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        val = self.expr.eval(scope, line, col)
        if self.op == "+":
            return +val
        if self.op == "-":
            return -val
        if self.op in ("!", "not"):
            return not bool(val)
        if self.op in ("(bool)", "bool"):
            return bool(val)
        if self.op in ("(real)", "real"):
            return float(val)
        if self.op in ("(integer)", "(int)", "int", "integer"):
            return int(val)
        if self.op in ("(string)", "string"):
            return str(val)
        raise DynareMacroError(f"Unknown unary operator '{self.op}'", line, col)


class BinaryOpNode(ExprNode):

    def __init__(self, op: str, left: ExprNode, right: ExprNode) -> None:
        self.op = op
        self.left = left
        self.right = right

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        # Short-circuit logical operators
        if self.op in ("&&", "and"):
            left_val = self.left.eval(scope, line, col)
            if not bool(left_val):
                return False
            return bool(self.right.eval(scope, line, col))
        if self.op in ("||", "or"):
            left_val = self.left.eval(scope, line, col)
            if bool(left_val):
                return True
            return bool(self.right.eval(scope, line, col))

        l_val = self.left.eval(scope, line, col)
        r_val = self.right.eval(scope, line, col)

        if self.op == "+":
            if isinstance(l_val, list) and isinstance(r_val, list):
                return l_val + r_val
            if isinstance(l_val, str) or isinstance(r_val, str):
                return str(l_val) + str(r_val)
            return l_val + r_val

        if self.op == "-":
            if isinstance(l_val, list) and isinstance(r_val, list):
                r_set = set(r_val)
                return [x for x in l_val if x not in r_set]
            return l_val - r_val

        if self.op == "*":
            # Cartesian product if both are lists
            if isinstance(l_val, list) and isinstance(r_val, list):
                res = []
                for x in l_val:
                    x_tup = x if isinstance(x, tuple) else (x,)
                    for y in r_val:
                        y_tup = y if isinstance(y, tuple) else (y,)
                        res.append(x_tup + y_tup)
                return res
            return l_val * r_val

        if self.op == "/":
            if r_val == 0:
                raise DynareMacroError("Division by zero", line, col)
            div_res = l_val / r_val
            if isinstance(div_res, float) and div_res.is_integer():
                return int(div_res)
            return div_res

        if self.op == "%":
            if r_val == 0:
                raise DynareMacroError("Modulo by zero", line, col)
            return l_val % r_val

        if self.op == "^":
            # Cartesian power if list
            if isinstance(l_val, list) and isinstance(r_val, int):
                if r_val < 0:
                    raise DynareMacroError(
                        "Negative power not supported for array Cartesian product", line, col
                    )
                if r_val == 0:
                    return [()]
                if r_val == 1:
                    return list(l_val)
                return list(itertools.product(l_val, repeat=r_val))
            return l_val ** r_val

        if self.op == "==":
            return l_val == r_val
        if self.op in ("!=", "<>"):
            return l_val != r_val
        if self.op == "<":
            return l_val < r_val
        if self.op == "<=":
            return l_val <= r_val
        if self.op == ">":
            return l_val > r_val
        if self.op == ">=":
            return l_val >= r_val
        if self.op == "in":
            return l_val in r_val
        if self.op == "&":
            if isinstance(l_val, list) and isinstance(r_val, list):
                r_set = set(r_val)
                return [x for x in l_val if x in r_set]
            return l_val & r_val
        if self.op == "|":
            if isinstance(l_val, list) and isinstance(r_val, list):
                res = list(l_val)
                l_set = set(l_val)
                for x in r_val:
                    if x not in l_set:
                        res.append(x)
                        l_set.add(x)
                return res
            return l_val | r_val

        raise DynareMacroError(f"Unknown binary operator '{self.op}'", line, col)


class RangeNode(ExprNode):

    def __init__(self, start: ExprNode, end: ExprNode, step: ExprNode | None = None) -> None:
        self.start = start
        self.end = end
        self.step = step

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        s = self.start.eval(scope, line, col)
        e = self.end.eval(scope, line, col)
        st = self.step.eval(scope, line, col) if self.step is not None else 1
        if not isinstance(s, (int, float)) or not isinstance(e, (int, float)) or not isinstance(st, (int, float)):
            raise DynareMacroError("Range bounds and step must be numeric", line, col)
        s, e, st = int(s), int(e), int(st)
        if st == 0:
            raise DynareMacroError("Range step cannot be 0", line, col)
        res = []
        if st > 0:
            curr = s
            while curr <= e:
                res.append(curr)
                curr += st
        else:
            curr = s
            while curr >= e:
                res.append(curr)
                curr += st
        return res


class IndexNode(ExprNode):

    def __init__(self, target: ExprNode, index: ExprNode) -> None:
        self.target = target
        self.index = index

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        tgt = self.target.eval(scope, line, col)
        idx = self.index.eval(scope, line, col)
        if not isinstance(tgt, (list, tuple, str)):
            raise DynareMacroError(
                f"Cannot index object of type '{type(tgt).__name__}'", line, col
            )
        # 1-based indexing
        if isinstance(idx, (list, tuple)):
            res = []
            for i in idx:
                if not isinstance(i, int):
                    raise DynareMacroError(
                        f"Array index must be integer, got '{type(i).__name__}'", line, col
                    )
                if i < 1 or i > len(tgt):
                    raise DynareMacroError(
                        f"Index {i} out of bounds for sequence of length {len(tgt)}", line, col
                    )
                res.append(tgt[i - 1])
            if isinstance(tgt, str):
                return "".join(res)
            return res
        if not isinstance(idx, int):
            raise DynareMacroError(
                f"Index must be integer, got '{type(idx).__name__}'", line, col
            )
        if idx < 1 or idx > len(tgt):
            raise DynareMacroError(
                f"Index {idx} out of bounds for sequence of length {len(tgt)}", line, col
            )
        return tgt[idx - 1]


class CallNode(ExprNode):

    def __init__(self, func_expr: ExprNode, args: list[ExprNode]) -> None:
        self.func_expr = func_expr
        self.args = args

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        # Special case: defined() can take a bare identifier without raising undefined error
        if isinstance(self.func_expr, IdentifierNode) and self.func_expr.name == "defined":
            if len(self.args) != 1:
                raise DynareMacroError(
                    f"defined() takes exactly 1 argument, got {len(self.args)}", line, col
                )
            arg0 = self.args[0]
            if isinstance(arg0, IdentifierNode):
                return scope.is_defined(arg0.name)
            arg_val = arg0.eval(scope, line, col)
            if isinstance(arg_val, str):
                return scope.is_defined(arg_val)
            return False

        fn = self.func_expr.eval(scope, line, col)
        evaluated_args = [a.eval(scope, line, col) for a in self.args]

        if isinstance(fn, MacroFunction):
            return fn.call(evaluated_args, line, col)
        if callable(fn):
            try:
                return fn(*evaluated_args)
            except Exception as e:
                raise DynareMacroError(f"Error evaluating function: {e}", line, col) from e
        raise DynareMacroError(f"'{type(fn).__name__}' object is not callable", line, col)


class MacroFunction:

    def __init__(
        self,
        name: str,
        params: list[str],
        body_expr: ExprNode,
        closure_scope: Scope,
    ) -> None:
        self.name = name
        self.params = params
        self.body_expr = body_expr
        self.closure_scope = closure_scope

    def call(self, args: list[Any], line: int = 1, col: int = 1) -> Any:
        if len(args) != len(self.params):
            raise DynareMacroError(
                f"Macro function '{self.name}' expects {len(self.params)} arguments, got {len(args)}",
                line,
                col,
            )
        call_scope = Scope(parent=self.closure_scope)
        for param_name, arg_val in zip(self.params, args):
            call_scope.set_local(param_name, arg_val)
        return self.body_expr.eval(call_scope, line, col)


class ArrayLitNode(ExprNode):

    def __init__(self, elements: list[ExprNode]) -> None:
        self.elements = elements

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        return [el.eval(scope, line, col) for el in self.elements]


class TupleLitNode(ExprNode):

    def __init__(self, elements: list[ExprNode]) -> None:
        self.elements = elements

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        return tuple(el.eval(scope, line, col) for el in self.elements)


class ComprehensionNode(ExprNode):

    def __init__(
        self,
        expr: ExprNode,
        targets: list[str],
        iter_expr: ExprNode,
        cond_expr: ExprNode | None = None,
    ) -> None:
        self.expr = expr
        self.targets = targets
        self.iter_expr = iter_expr
        self.cond_expr = cond_expr

    def eval(self, scope: Scope, line: int = 1, col: int = 1) -> Any:
        iterable = self.iter_expr.eval(scope, line, col)
        if not isinstance(iterable, (list, tuple)):
            raise DynareMacroError(
                f"Comprehension iteration over non-sequence of type '{type(iterable).__name__}'",
                line,
                col,
            )
        res = []
        comp_scope = Scope(parent=scope)
        for item in iterable:
            _bind_targets(comp_scope, self.targets, item, line, col)
            if self.cond_expr is not None:
                cond_val = self.cond_expr.eval(comp_scope, line, col)
                if not bool(cond_val):
                    continue
            res.append(self.expr.eval(comp_scope, line, col))
        return res


def _bind_targets(
    scope: Scope, targets: list[str], item: Any, line: int = 1, col: int = 1
) -> None:
    if len(targets) == 1:
        scope.set_local(targets[0], item)
    else:
        if not isinstance(item, (list, tuple)):
            raise DynareMacroError(
                f"Cannot unpack non-sequence '{type(item).__name__}' into {len(targets)} variables",
                line,
                col,
            )
        if len(item) != len(targets):
            raise DynareMacroError(
                f"Cannot unpack {len(item)} elements into {len(targets)} variables",
                line,
                col,
            )
        for t_name, val in zip(targets, item):
            scope.set_local(t_name, val)


# ---------------------------------------------------------------------------
# Macro Expression Tokenizer & Parser
# ---------------------------------------------------------------------------

class Token:

    def __init__(self, type_: str, value: Any, line: int = 1, col: int = 1) -> None:
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r}, {self.line}, {self.col})"


_TOKEN_RE = re.compile(
    r"""
    (?P<WHITESPACE>\s+)
    |(?P<FLOAT>(?:[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?|\.[0-9]+(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+))
    |(?P<INT>[0-9]+)
    |(?P<STRING>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
    |(?P<TYPECAST>\((?:bool|real|int|integer|string)\))
    |(?P<OP_2>==|!=|<>|<=|>=|&&|\|\|)
    |(?P<OP_1>[+\-*/%^<>=!&|:.])
    |(?P<PUNCT>[(),\[\]])
    |(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
""",
    re.VERBOSE,
)


def _unescape_string(s: str) -> str:
    content = s[1:-1]
    res = []
    i = 0
    n = len(content)
    while i < n:
        if content[i] == "\\" and i + 1 < n:
            c = content[i + 1]
            if c == "n":
                res.append("\n")
            elif c == "t":
                res.append("\t")
            elif c == "r":
                res.append("\r")
            elif c == "\\":
                res.append("\\")
            elif c == '"':
                res.append('"')
            elif c == "'":
                res.append("'")
            else:
                res.append(c)
            i += 2
        else:
            res.append(content[i])
            i += 1
    return "".join(res)


def tokenize_macro_expr(expr_str: str, line_offset: int = 1, col_offset: int = 1) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    length = len(expr_str)
    while pos < length:
        match = _TOKEN_RE.match(expr_str, pos)
        if not match:
            ch = expr_str[pos]
            raise DynareMacroError(
                f"Unexpected character in macro expression: '{ch}'",
                line_offset,
                col_offset + pos,
            )
        pos = match.end()
        kind = match.lastgroup
        text = match.group()

        if kind == "WHITESPACE":
            continue
        if kind == "INT":
            tokens.append(Token("NUMBER", int(text), line_offset, col_offset + match.start()))
        elif kind == "FLOAT":
            tokens.append(Token("NUMBER", float(text), line_offset, col_offset + match.start()))
        elif kind == "STRING":
            tokens.append(Token("STRING", _unescape_string(text), line_offset, col_offset + match.start()))
        elif kind == "TYPECAST":
            tokens.append(Token("TYPECAST", text, line_offset, col_offset + match.start()))
        elif kind in ("OP_2", "OP_1"):
            tokens.append(Token("OP", text, line_offset, col_offset + match.start()))
        elif kind == "PUNCT":
            tokens.append(Token("PUNCT", text, line_offset, col_offset + match.start()))
        elif kind == "IDENT":
            tokens.append(Token("IDENT", text, line_offset, col_offset + match.start()))

    tokens.append(Token("EOF", "", line_offset, col_offset + pos))
    return tokens


class MacroExprParser:

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != "EOF":
            self.pos += 1
        return tok

    def match(self, type_: str, value: Any = None) -> bool:
        tok = self.peek()
        if tok.type == type_ and (value is None or tok.value == value):
            self.advance()
            return True
        return False

    def expect(self, type_: str, value: Any = None) -> Token:
        tok = self.peek()
        if tok.type != type_ or (value is not None and tok.value != value):
            expected = f"{type_} '{value}'" if value is not None else type_
            raise DynareMacroError(
                f"Expected {expected}, got {tok.type} '{tok.value}'", tok.line, tok.col
            )
        return self.advance()

    def parse(self) -> ExprNode:
        expr = self.parse_or()
        if self.peek().type != "EOF":
            tok = self.peek()
            raise DynareMacroError(
                f"Unexpected trailing tokens in macro expression: '{tok.value}'", tok.line, tok.col
            )
        return expr

    def parse_or(self) -> ExprNode:
        left = self.parse_and()
        while self.peek().type in ("OP", "IDENT") and self.peek().value in ("||", "or"):
            op_tok = self.advance()
            right = self.parse_and()
            left = BinaryOpNode(op_tok.value, left, right)
        return left

    def parse_and(self) -> ExprNode:
        left = self.parse_bitwise_or()
        while self.peek().type in ("OP", "IDENT") and self.peek().value in ("&&", "and"):
            op_tok = self.advance()
            right = self.parse_bitwise_or()
            left = BinaryOpNode(op_tok.value, left, right)
        return left

    def parse_bitwise_or(self) -> ExprNode:
        left = self.parse_bitwise_and()
        while self.peek().type == "OP" and self.peek().value == "|":
            op_tok = self.advance()
            right = self.parse_bitwise_and()
            left = BinaryOpNode(op_tok.value, left, right)
        return left

    def parse_bitwise_and(self) -> ExprNode:
        left = self.parse_equality()
        while self.peek().type == "OP" and self.peek().value == "&":
            op_tok = self.advance()
            right = self.parse_equality()
            left = BinaryOpNode(op_tok.value, left, right)
        return left

    def parse_equality(self) -> ExprNode:
        left = self.parse_relational()
        while self.peek().type == "OP" and self.peek().value in ("==", "!=", "<>"):
            op_tok = self.advance()
            right = self.parse_relational()
            left = BinaryOpNode(op_tok.value, left, right)
        return left

    def parse_relational(self) -> ExprNode:
        left = self.parse_in()
        while self.peek().type == "OP" and self.peek().value in ("<", "<=", ">", ">="):
            op_tok = self.advance()
            right = self.parse_in()
            left = BinaryOpNode(op_tok.value, left, right)
        return left

    def parse_in(self) -> ExprNode:
        left = self.parse_range()
        while self.peek().type == "IDENT" and self.peek().value == "in":
            op_tok = self.advance()
            right = self.parse_range()
            left = BinaryOpNode(op_tok.value, left, right)
        return left

    def parse_range(self) -> ExprNode:
        left = self.parse_additive()
        if self.peek().type == "OP" and self.peek().value == ":":
            self.advance()  # consume first ':'
            second = self.parse_additive()
            if self.peek().type == "OP" and self.peek().value == ":":
                self.advance()  # consume second ':'
                third = self.parse_additive()
                # start : step : end
                return RangeNode(start=left, step=second, end=third)
            # start : end
            return RangeNode(start=left, end=second, step=None)
        return left

    def parse_additive(self) -> ExprNode:
        left = self.parse_multiplicative()
        while self.peek().type == "OP" and self.peek().value in ("+", "-"):
            op_tok = self.advance()
            right = self.parse_multiplicative()
            left = BinaryOpNode(op_tok.value, left, right)
        return left

    def parse_multiplicative(self) -> ExprNode:
        left = self.parse_unary()
        while self.peek().type == "OP" and self.peek().value in ("*", "/", "%"):
            op_tok = self.advance()
            right = self.parse_unary()
            left = BinaryOpNode(op_tok.value, left, right)
        return left

    def parse_unary(self) -> ExprNode:
        tok = self.peek()
        if tok.type == "OP" and tok.value in ("+", "-", "!"):
            self.advance()
            operand = self.parse_unary()
            return UnaryOpNode(tok.value, operand)
        if tok.type == "IDENT" and tok.value == "not":
            self.advance()
            operand = self.parse_unary()
            return UnaryOpNode(tok.value, operand)
        if tok.type == "TYPECAST":
            self.advance()
            operand = self.parse_unary()
            return UnaryOpNode(tok.value, operand)
        return self.parse_power()

    def parse_power(self) -> ExprNode:
        left = self.parse_postfix()
        if self.peek().type == "OP" and self.peek().value == "^":
            op_tok = self.advance()
            right = self.parse_unary()  # right-associative
            return BinaryOpNode(op_tok.value, left, right)
        return left

    def parse_postfix(self) -> ExprNode:
        node = self.parse_primary()
        while True:
            tok = self.peek()
            if tok.type == "PUNCT" and tok.value == "[":
                self.advance()
                index_expr = self.parse_or()
                self.expect("PUNCT", "]")
                node = IndexNode(node, index_expr)
            elif tok.type == "PUNCT" and tok.value == "(":
                self.advance()
                args: list[ExprNode] = []
                if not (self.peek().type == "PUNCT" and self.peek().value == ")"):
                    args.append(self.parse_or())
                    while self.match("PUNCT", ","):
                        args.append(self.parse_or())
                self.expect("PUNCT", ")")
                node = CallNode(node, args)
            else:
                break
        return node

    def parse_primary(self) -> ExprNode:
        tok = self.peek()

        if tok.type == "NUMBER":
            self.advance()
            return LiteralNode(tok.value)

        if tok.type == "STRING":
            self.advance()
            return LiteralNode(tok.value)

        if tok.type == "IDENT":
            self.advance()
            if tok.value in ("true", "True"):
                return LiteralNode(True)
            if tok.value in ("false", "False"):
                return LiteralNode(False)
            return IdentifierNode(tok.value)

        if tok.type == "PUNCT" and tok.value == "[":
            self.advance()  # consume '['
            if self.match("PUNCT", "]"):
                return ArrayLitNode([])

            first_expr = self.parse_or()
            # Check for array comprehension: [expr for i in arr (when cond)?]
            if self.peek().type == "IDENT" and self.peek().value == "for":
                self.advance()  # consume 'for'
                targets = []
                if self.match("PUNCT", "("):
                    t_tok = self.expect("IDENT")
                    targets.append(t_tok.value)
                    while self.match("PUNCT", ","):
                        targets.append(self.expect("IDENT").value)
                    self.expect("PUNCT", ")")
                else:
                    t_tok = self.expect("IDENT")
                    targets.append(t_tok.value)
                    while self.match("PUNCT", ","):
                        targets.append(self.expect("IDENT").value)

                for_tok = self.expect("IDENT", "in")
                iter_expr = self.parse_or()
                cond_expr = None
                if self.peek().type == "IDENT" and self.peek().value in ("when", "if"):
                    self.advance()
                    cond_expr = self.parse_or()
                self.expect("PUNCT", "]")
                return ComprehensionNode(first_expr, targets, iter_expr, cond_expr)

            # Regular array literal
            elements = [first_expr]
            while self.match("PUNCT", ","):
                if self.peek().type == "PUNCT" and self.peek().value == "]":
                    break  # trailing comma
                elements.append(self.parse_or())
            self.expect("PUNCT", "]")
            return ArrayLitNode(elements)

        if tok.type == "PUNCT" and tok.value == "(":
            self.advance()  # consume '('
            if self.match("PUNCT", ")"):
                return TupleLitNode([])

            expr1 = self.parse_or()
            if self.match("PUNCT", ","):
                elements = [expr1]
                if not (self.peek().type == "PUNCT" and self.peek().value == ")"):
                    elements.append(self.parse_or())
                    while self.match("PUNCT", ","):
                        if self.peek().type == "PUNCT" and self.peek().value == ")":
                            break
                        elements.append(self.parse_or())
                self.expect("PUNCT", ")")
                return TupleLitNode(elements)

            self.expect("PUNCT", ")")
            return expr1

        raise DynareMacroError(
            f"Unexpected token in primary expression: {tok.type} '{tok.value}'", tok.line, tok.col
        )


def parse_macro_expr(expr_str: str, line_offset: int = 1, col_offset: int = 1) -> ExprNode:
    tokens = tokenize_macro_expr(expr_str, line_offset, col_offset)
    parser = MacroExprParser(tokens)
    return parser.parse()


# ---------------------------------------------------------------------------
# Comment Stripping & Line Classification
# ---------------------------------------------------------------------------

def _is_percent_comment(rest: str) -> bool:
    """Return True if text after % is a comment, False if part of an expression (modulo)."""
    rest_clean = rest.strip()
    if not rest_clean:
        return True
    # If rest contains operators or closing brackets, it is part of an expression
    if any(op in rest_clean for op in ("==", "!=", "<=", ">=", "<", ">", "+", "-", "*", "/", "]", ")", ";")):
        return False
    words = rest_clean.split()
    if len(words) >= 2 and all(w.isalpha() for w in words[:2]):
        return True
    if rest_clean.isalpha():
        return True
    return False


def _strip_trailing_comment(line: str) -> str:
    """Strip trailing // or % comments from a directive line, respecting quotes."""
    i = 0
    n = len(line)
    in_quote: str | None = None
    while i < n:
        c = line[i]
        if in_quote:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == in_quote:
                in_quote = None
        else:
            if c in ('"', "'"):
                in_quote = c
            elif c == "/" and i + 1 < n and line[i + 1] == "/":
                return line[:i].rstrip()
            elif c == "%":
                if _is_percent_comment(line[i + 1:]):
                    return line[:i].rstrip()
        i += 1
    return line.rstrip()


def _classify_line(line: str, in_block_comment: bool) -> tuple[bool, str, bool]:
    """Determine if a line is a macro directive.

    Returns (is_directive, clean_text, next_in_block_comment).
    """
    i = 0
    n = len(line)

    if in_block_comment:
        end_idx = line.find("*/")
        if end_idx == -1:
            return False, line, True
        in_block_comment = False
        i = end_idx + 2

    # Skip leading whitespace
    while i < n and line[i] in (" ", "\t"):
        i += 1

    if i >= n:
        return False, line, in_block_comment

    # Check for block comment open
    if line[i : i + 2] == "/*":
        close_idx = line.find("*/", i + 2)
        if close_idx == -1:
            return False, line, True
        # Closed on same line, recursive check on remaining
        return _classify_line(line[close_idx + 2 :], False)

    # Check for line comment
    if line[i] == "%" or line[i : i + 2] == "//":
        return False, line, in_block_comment

    # Check for directive prefix @#
    if line[i : i + 2] == "@#":
        cleaned = _strip_trailing_comment(line[i:])
        return True, cleaned, in_block_comment

    return False, line, in_block_comment


# ---------------------------------------------------------------------------
# Preprocessor Statements & AST
# ---------------------------------------------------------------------------

class Stmt:
    """Base class for preprocessor AST statements."""

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        raise NotImplementedError


class TextStmt(Stmt):

    def __init__(self, lines: list[tuple[int, str]]) -> None:
        self.lines = lines

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        for line_num, text in self.lines:
            output.append(_interpolate_text(text, scope, line_num))


class DefineStmt(Stmt):

    def __init__(
        self,
        line: int,
        col: int,
        name: str,
        params: list[str] | None,
        expr: ExprNode,
        is_func: bool,
    ) -> None:
        self.line = line
        self.col = col
        self.name = name
        self.params = params
        self.expr = expr
        self.is_func = is_func

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        if self.is_func:
            assert self.params is not None
            fn = MacroFunction(self.name, self.params, self.expr, scope)
            scope.set(self.name, fn)
        else:
            val = self.expr.eval(scope, self.line, self.col)
            scope.set(self.name, val)


class UndefStmt(Stmt):

    def __init__(self, line: int, col: int, name: str) -> None:
        self.line = line
        self.col = col
        self.name = name

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        try:
            scope.delete(self.name)
        except KeyError:
            pass


class ForStmt(Stmt):

    def __init__(
        self,
        line: int,
        col: int,
        targets: list[str],
        iter_expr: ExprNode,
        cond_expr: ExprNode | None,
        body: list[Stmt],
    ) -> None:
        self.line = line
        self.col = col
        self.targets = targets
        self.iter_expr = iter_expr
        self.cond_expr = cond_expr
        self.body = body

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        iterable = self.iter_expr.eval(scope, self.line, self.col)
        if not isinstance(iterable, (list, tuple)):
            raise DynareMacroError(
                f"@#for iteration over non-sequence of type '{type(iterable).__name__}'",
                self.line,
                self.col,
            )

        # Lexical scoping frame with restoration on exit
        loop_scope = Scope(parent=scope)
        for item in iterable:
            _bind_targets(loop_scope, self.targets, item, self.line, self.col)
            if self.cond_expr is not None:
                cond_val = self.cond_expr.eval(loop_scope, self.line, self.col)
                if not bool(cond_val):
                    continue
            for stmt in self.body:
                stmt.execute(loop_scope, base_dir, include_paths, include_stack, output)


class IfStmt(Stmt):

    def __init__(
        self,
        line: int,
        col: int,
        branches: list[tuple[ExprNode, list[Stmt]]],
        else_branch: list[Stmt] | None,
    ) -> None:
        self.line = line
        self.col = col
        self.branches = branches
        self.else_branch = else_branch

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        for cond_expr, branch_stmts in self.branches:
            cond_val = cond_expr.eval(scope, self.line, self.col)
            if bool(cond_val):
                for stmt in branch_stmts:
                    stmt.execute(scope, base_dir, include_paths, include_stack, output)
                return
        if self.else_branch is not None:
            for stmt in self.else_branch:
                stmt.execute(scope, base_dir, include_paths, include_stack, output)


class IncludeStmt(Stmt):

    def __init__(self, line: int, col: int, expr: ExprNode) -> None:
        self.line = line
        self.col = col
        self.expr = expr

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        path_val = self.expr.eval(scope, self.line, self.col)
        if not isinstance(path_val, str):
            raise DynareMacroError(
                f"@#include expression must evaluate to string, got '{type(path_val).__name__}'",
                self.line,
                self.col,
            )

        resolved_path = _resolve_include_file(
            path_val, base_dir, include_paths, self.line, self.col
        )

        if resolved_path in include_stack:
            cycle = " -> ".join(str(p) for p in include_stack + [resolved_path])
            raise DynareMacroError(
                f"circular @#include detected: {cycle}", self.line, self.col
            )

        sub_text = resolved_path.read_text(encoding="utf-8")
        sub_preprocessed = _preprocess_macro_internal(
            text=sub_text,
            base_dir=resolved_path.parent,
            include_paths=include_paths,
            include_stack=include_stack + [resolved_path],
            scope=scope,
        )
        output.append(sub_preprocessed)


class IncludePathStmt(Stmt):

    def __init__(self, line: int, col: int, expr: ExprNode) -> None:
        self.line = line
        self.col = col
        self.expr = expr

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        path_val = self.expr.eval(scope, self.line, self.col)
        if not isinstance(path_val, str):
            raise DynareMacroError(
                f"@#includepath expression must evaluate to string, got '{type(path_val).__name__}'",
                self.line,
                self.col,
            )
        p = Path(path_val)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        else:
            p = p.resolve()
        if p not in include_paths:
            include_paths.append(p)


class EchoStmt(Stmt):

    def __init__(self, line: int, col: int, expr: ExprNode) -> None:
        self.line = line
        self.col = col
        self.expr = expr

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        val = self.expr.eval(scope, self.line, self.col)
        # Diagnostic print
        print(f"[Dynare Macro Echo] {val}")


class ErrorStmt(Stmt):

    def __init__(self, line: int, col: int, expr: ExprNode) -> None:
        self.line = line
        self.col = col
        self.expr = expr

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        val = self.expr.eval(scope, self.line, self.col)
        raise DynareMacroError(str(val), self.line, self.col)


class EchoMacroVarsStmt(Stmt):

    def __init__(self, line: int, col: int) -> None:
        self.line = line
        self.col = col

    def execute(
        self,
        scope: Scope,
        base_dir: Path,
        include_paths: list[Path],
        include_stack: list[Path],
        output: list[str],
    ) -> None:
        vars_dict = scope.all_vars()
        print(f"[Dynare Macro Vars] {vars_dict}")


def _resolve_include_file(
    path_str: str,
    base_dir: Path,
    include_paths: list[Path],
    line: int,
    col: int,
) -> Path:
    p = Path(path_str)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(base_dir / p)
        candidates.append(Path.cwd() / p)
        for inc in include_paths:
            candidates.append(inc / p)

    for c in candidates:
        if c.is_file():
            return c.resolve()

    searched_str = ", ".join(str(c) for c in candidates)
    raise DynareMacroError(
        f"Include file '{path_str}' not found (searched: {searched_str})", line, col
    )


# ---------------------------------------------------------------------------
# Expression Interpolation @{...}
# ---------------------------------------------------------------------------

def _format_interpolated(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(val)
    if isinstance(val, list):
        return "[" + ", ".join(_format_repr(x) for x in val) + "]"
    if isinstance(val, tuple):
        if len(val) == 1:
            return f"({_format_repr(val[0])},)"
        return "(" + ", ".join(_format_repr(x) for x in val) + ")"
    return str(val)


def _format_repr(val: Any) -> str:
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def _find_matching_brace(text: str, start: int) -> int:
    """Find the matching closing brace } for @{ starting at index start."""
    i = start + 2
    n = len(text)
    brace_depth = 0
    bracket_depth = 0
    paren_depth = 0
    in_quote: str | None = None

    while i < n:
        c = text[i]
        if in_quote:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == in_quote:
                in_quote = None
        else:
            if c in ('"', "'"):
                in_quote = c
            elif c == "{":
                brace_depth += 1
            elif c == "}":
                if brace_depth == 0 and bracket_depth == 0 and paren_depth == 0:
                    return i
                brace_depth -= 1
            elif c == "[":
                bracket_depth += 1
            elif c == "]":
                bracket_depth -= 1
            elif c == "(":
                paren_depth += 1
            elif c == ")":
                paren_depth -= 1
        i += 1
    return -1


def _interpolate_text(text: str, scope: Scope, line_num: int) -> str:
    if "@{" not in text:
        return text

    res: list[str] = []
    pos = 0
    n = len(text)

    while pos < n:
        idx = text.find("@{", pos)
        if idx == -1:
            res.append(text[pos:])
            break

        res.append(text[pos:idx])
        close_idx = _find_matching_brace(text, idx)
        if close_idx == -1:
            raise DynareMacroError(
                "Unclosed macro interpolation '@{'", line_num, idx + 1
            )

        expr_str = text[idx + 2 : close_idx].strip()
        if not expr_str:
            raise DynareMacroError(
                "Empty macro interpolation expression '@{}'", line_num, idx + 1
            )

        expr_ast = parse_macro_expr(expr_str, line_num, idx + 3)
        val = expr_ast.eval(scope, line_num, idx + 3)
        res.append(_format_interpolated(val))
        pos = close_idx + 1

    return "".join(res)


# ---------------------------------------------------------------------------
# Source File Tokenizer (Line Continuation & Directive Slicing)
# ---------------------------------------------------------------------------

class RawItem:
    """Classified line or directive item."""

    def __init__(
        self,
        is_directive: bool,
        line_num: int,
        col_num: int,
        name: str | None,
        content: str,
    ) -> None:
        self.is_directive = is_directive
        self.line_num = line_num
        self.col_num = col_num
        self.name = name  # e.g. 'define', 'for', 'endfor', 'if', etc.
        self.content = content


def _split_into_items(text: str) -> list[RawItem]:
    """Preprocess lines handling continuation (\\) and block comments."""
    raw_lines = text.splitlines(keepends=True)
    items: list[RawItem] = []
    in_block_comment = False
    i = 0
    total = len(raw_lines)

    while i < total:
        orig_line_num = i + 1
        line = raw_lines[i]

        is_dir, clean_text, next_in_block = _classify_line(line, in_block_comment)
        in_block_comment = next_in_block

        if not is_dir:
            items.append(RawItem(False, orig_line_num, 1, None, line))
            i += 1
            continue

        # Handle line continuation with \\
        combined_dir = clean_text
        while combined_dir.endswith("\\"):
            combined_dir = combined_dir[:-1].rstrip()
            i += 1
            if i < total:
                next_line = raw_lines[i]
                next_clean = _strip_trailing_comment(next_line.strip())
                combined_dir = combined_dir + " " + next_clean
            else:
                break

        # Extract directive name
        m_dir = re.match(r"^@#\s*([A-Za-z_][A-Za-z0-9_]*)(\s+.*|\(.*)?$", combined_dir, re.DOTALL)
        if not m_dir:
            raise DynareMacroError(
                f"Malformed macro directive: '{combined_dir}'", orig_line_num, 1
            )
        dir_name = m_dir.group(1).lower()
        items.append(RawItem(True, orig_line_num, 1, dir_name, combined_dir))
        i += 1

    return items


# ---------------------------------------------------------------------------
# Preprocessor Directive Parser (AST Builder)
# ---------------------------------------------------------------------------

class DirectiveParser:

    def __init__(self, items: list[RawItem]) -> None:
        self.items = items
        self.pos = 0

    def peek(self) -> RawItem | None:
        if self.pos < len(self.items):
            return self.items[self.pos]
        return None

    def advance(self) -> RawItem:
        item = self.items[self.pos]
        self.pos += 1
        return item

    def parse_all(self) -> list[Stmt]:
        stmts, stop_dir = self.parse_block(stop_dirs=set())
        if stop_dir is not None:
            raise DynareMacroError(
                f"Unexpected directive '@#{stop_dir.name}' outside block",
                stop_dir.line_num,
                stop_dir.col_num,
            )
        return stmts

    def parse_block(self, stop_dirs: set[str]) -> tuple[list[Stmt], RawItem | None]:
        stmts: list[Stmt] = []
        text_accumulator: list[tuple[int, str]] = []

        def flush_text():
            if text_accumulator:
                stmts.append(TextStmt(list(text_accumulator)))
                text_accumulator.clear()

        while self.pos < len(self.items):
            item = self.peek()
            assert item is not None

            if not item.is_directive:
                self.advance()
                text_accumulator.append((item.line_num, item.content))
                continue

            # Directive
            if item.name in stop_dirs:
                flush_text()
                return stmts, self.advance()

            flush_text()
            item = self.advance()

            if item.name == "define":
                stmts.append(self._parse_define(item))
            elif item.name == "undef":
                stmts.append(self._parse_undef(item))
            elif item.name == "for":
                stmts.append(self._parse_for(item))
            elif item.name in ("if", "ifdef", "ifndef"):
                stmts.append(self._parse_if(item))
            elif item.name == "include":
                stmts.append(self._parse_include(item))
            elif item.name == "includepath":
                stmts.append(self._parse_includepath(item))
            elif item.name == "echo":
                stmts.append(self._parse_echo(item))
            elif item.name == "error":
                stmts.append(self._parse_error(item))
            elif item.name == "echomacrovars":
                stmts.append(EchoMacroVarsStmt(item.line_num, item.col_num))
            elif item.name in ("endfor", "endif", "else", "elseif", "elif"):
                raise DynareMacroError(
                    f"Unexpected directive '@#{item.name}'", item.line_num, item.col_num
                )
            else:
                raise DynareMacroError(
                    f"Unknown macro directive '@#{item.name}'", item.line_num, item.col_num
                )

        flush_text()
        return stmts, None

    def _parse_define(self, item: RawItem) -> DefineStmt:
        # Check function: @#define IDENT(args) = EXPR
        m_func = re.match(
            r"^@#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*=(.*)$",
            item.content,
            re.DOTALL,
        )
        if m_func:
            fn_name = m_func.group(1)
            args_str = m_func.group(2).strip()
            params = [a.strip() for a in args_str.split(",") if a.strip()] if args_str else []
            for p in params:
                if not p.isidentifier():
                    raise DynareMacroError(
                        f"Invalid parameter name '{p}' in macro function definition",
                        item.line_num,
                        item.col_num,
                    )
            expr_str = m_func.group(3).strip()
            body_expr = parse_macro_expr(expr_str, item.line_num, item.col_num)
            return DefineStmt(item.line_num, item.col_num, fn_name, params, body_expr, is_func=True)

        # Variable: @#define IDENT (= EXPR)?
        m_var = re.match(
            r"^@#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*=(.*))?$",
            item.content,
            re.DOTALL,
        )
        if m_var:
            var_name = m_var.group(1)
            expr_part = m_var.group(2)
            if expr_part is not None:
                val_expr = parse_macro_expr(expr_part.strip(), item.line_num, item.col_num)
            else:
                val_expr = LiteralNode(1)
            return DefineStmt(item.line_num, item.col_num, var_name, None, val_expr, is_func=False)

        raise DynareMacroError(
            f"Malformed @#define directive: '{item.content}'", item.line_num, item.col_num
        )

    def _parse_undef(self, item: RawItem) -> UndefStmt:
        m = re.match(r"^@#\s*undef\s+([A-Za-z_][A-Za-z0-9_]*)$", item.content)
        if m:
            return UndefStmt(item.line_num, item.col_num, m.group(1))
        raise DynareMacroError(
            f"Malformed @#undef directive: '{item.content}'", item.line_num, item.col_num
        )

    def _parse_for(self, item: RawItem) -> ForStmt:
        # Bracket-aware parsing: track [], (), {}, quotes to avoid collision with inner 'when'
        # in inline array comprehensions: [expr for var in arr when cond]
        m_start = re.match(r"^@#\s*for\s+", item.content, re.IGNORECASE)
        if not m_start:
            raise DynareMacroError(
                f"Malformed @#for directive: '{item.content}'", item.line_num, item.col_num
            )
        rest = item.content[m_start.end():]

        # Scan for 'in' at bracket depth 0 outside string literals
        n = len(rest)
        i = 0
        bracket_stack: list[str] = []
        quote_char: str | None = None
        in_pos = -1

        while i < n:
            ch = rest[i]
            if quote_char is not None:
                if ch == "\\":
                    i += 2
                    continue
                elif ch == quote_char:
                    quote_char = None
                i += 1
                continue

            if ch in ('"', "'"):
                quote_char = ch
                i += 1
                continue

            if ch in ('[', '(', '{'):
                bracket_stack.append(ch)
                i += 1
                continue
            elif ch in (']', ')', '}'):
                if bracket_stack:
                    bracket_stack.pop()
                i += 1
                continue

            if len(bracket_stack) == 0:
                if rest[i:i + 2].lower() == "in":
                    prev_ok = (i == 0) or (not (rest[i - 1].isalnum() or rest[i - 1] == "_"))
                    next_ok = (i + 2 == n) or (not (rest[i + 2].isalnum() or rest[i + 2] == "_"))
                    if prev_ok and next_ok:
                        in_pos = i
                        break
            i += 1

        if in_pos == -1:
            raise DynareMacroError(
                f"Malformed @#for directive: '{item.content}'", item.line_num, item.col_num
            )

        target_str = rest[:in_pos].strip()
        after_in = rest[in_pos + 2:].strip()

        # Scan after_in for 'when' at bracket depth 0 outside string literals
        n_after = len(after_in)
        j = 0
        bracket_stack.clear()
        quote_char = None
        when_pos = -1

        while j < n_after:
            ch = after_in[j]
            if quote_char is not None:
                if ch == "\\":
                    j += 2
                    continue
                elif ch == quote_char:
                    quote_char = None
                j += 1
                continue

            if ch in ('"', "'"):
                quote_char = ch
                j += 1
                continue

            if ch in ('[', '(', '{'):
                bracket_stack.append(ch)
                j += 1
                continue
            elif ch in (']', ')', '}'):
                if bracket_stack:
                    bracket_stack.pop()
                j += 1
                continue

            if len(bracket_stack) == 0:
                if after_in[j:j + 4].lower() == "when":
                    prev_ok = (j == 0) or (not (after_in[j - 1].isalnum() or after_in[j - 1] == "_"))
                    next_ok = (j + 4 == n_after) or (not (after_in[j + 4].isalnum() or after_in[j + 4] == "_"))
                    if prev_ok and next_ok:
                        when_pos = j
                        break
            j += 1

        if when_pos != -1:
            iter_str = after_in[:when_pos].strip()
            cond_str = after_in[when_pos + 4:].strip()
            if not cond_str:
                raise DynareMacroError(
                    f"Malformed @#for directive: missing condition after 'when'",
                    item.line_num,
                    item.col_num,
                )
        else:
            iter_str = after_in
            cond_str = None

        if not target_str or not iter_str:
            raise DynareMacroError(
                f"Malformed @#for directive: '{item.content}'", item.line_num, item.col_num
            )

        if target_str.startswith("(") and target_str.endswith(")"):
            target_str = target_str[1:-1].strip()
        targets = [t.strip() for t in target_str.split(",") if t.strip()]
        for t in targets:
            if not t.isidentifier():
                raise DynareMacroError(
                    f"Invalid loop target identifier '{t}' in @#for", item.line_num, item.col_num
                )

        iter_expr = parse_macro_expr(iter_str, item.line_num, item.col_num)
        cond_expr = (
            parse_macro_expr(cond_str, item.line_num, item.col_num)
            if cond_str
            else None
        )

        body, stop_dir = self.parse_block(stop_dirs={"endfor"})
        if stop_dir is None or stop_dir.name != "endfor":
            raise DynareMacroError(
                f"Unclosed @#for block started at line {item.line_num}",
                item.line_num,
                item.col_num,
            )

        return ForStmt(item.line_num, item.col_num, targets, iter_expr, cond_expr, body)

    def _parse_if(self, item: RawItem) -> IfStmt:
        branches: list[tuple[ExprNode, list[Stmt]]] = []
        else_branch: list[Stmt] | None = None

        # Build initial condition
        if item.name == "if":
            cond_str = re.sub(r"^@#\s*if\s+", "", item.content).strip()
            curr_cond = parse_macro_expr(cond_str, item.line_num, item.col_num)
        elif item.name == "ifdef":
            ident = re.sub(r"^@#\s*ifdef\s+", "", item.content).strip()
            curr_cond = CallNode(IdentifierNode("defined"), [IdentifierNode(ident)])
        elif item.name == "ifndef":
            ident = re.sub(r"^@#\s*ifndef\s+", "", item.content).strip()
            curr_cond = UnaryOpNode(
                "!", CallNode(IdentifierNode("defined"), [IdentifierNode(ident)])
            )
        else:
            raise DynareMacroError(f"Unexpected if directive: {item.name}", item.line_num, item.col_num)

        while True:
            branch_body, stop_dir = self.parse_block(
                stop_dirs={"elseif", "elif", "else", "endif"}
            )
            branches.append((curr_cond, branch_body))

            if stop_dir is None:
                raise DynareMacroError(
                    f"Unclosed @#if block started at line {item.line_num}",
                    item.line_num,
                    item.col_num,
                )

            if stop_dir.name in ("elseif", "elif"):
                cond_str = re.sub(r"^@#\s*(?:elseif|elif)\s+", "", stop_dir.content).strip()
                curr_cond = parse_macro_expr(cond_str, stop_dir.line_num, stop_dir.col_num)
            elif stop_dir.name == "else":
                else_body, end_dir = self.parse_block(stop_dirs={"endif"})
                if end_dir is None or end_dir.name != "endif":
                    raise DynareMacroError(
                        f"Unclosed @#if block started at line {item.line_num}",
                        item.line_num,
                        item.col_num,
                    )
                else_branch = else_body
                break
            elif stop_dir.name == "endif":
                break

        return IfStmt(item.line_num, item.col_num, branches, else_branch)

    def _parse_include(self, item: RawItem) -> IncludeStmt:
        expr_str = re.sub(r"^@#\s*include\s+", "", item.content).strip()
        expr = parse_macro_expr(expr_str, item.line_num, item.col_num)
        return IncludeStmt(item.line_num, item.col_num, expr)

    def _parse_includepath(self, item: RawItem) -> IncludePathStmt:
        expr_str = re.sub(r"^@#\s*includepath\s+", "", item.content).strip()
        expr = parse_macro_expr(expr_str, item.line_num, item.col_num)
        return IncludePathStmt(item.line_num, item.col_num, expr)

    def _parse_echo(self, item: RawItem) -> EchoStmt:
        expr_str = re.sub(r"^@#\s*echo\s+", "", item.content).strip()
        expr = parse_macro_expr(expr_str, item.line_num, item.col_num)
        return EchoStmt(item.line_num, item.col_num, expr)

    def _parse_error(self, item: RawItem) -> ErrorStmt:
        expr_str = re.sub(r"^@#\s*error\s+", "", item.content).strip()
        expr = parse_macro_expr(expr_str, item.line_num, item.col_num)
        return ErrorStmt(item.line_num, item.col_num, expr)


# ---------------------------------------------------------------------------
# Internal Preprocess Engine
# ---------------------------------------------------------------------------

def _preprocess_macro_internal(
    text: str,
    base_dir: Path,
    include_paths: list[Path],
    include_stack: list[Path],
    scope: Scope,
) -> str:
    raw_items = _split_into_items(text)
    parser = DirectiveParser(raw_items)
    ast = parser.parse_all()

    output_lines: list[str] = []
    for stmt in ast:
        stmt.execute(scope, base_dir, include_paths, include_stack, output_lines)

    return "".join(output_lines)


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def preprocess_macro(
    text: str,
    base_dir: Path | str | None = None,
    include_paths: Sequence[Path | str] | None = None,
) -> str:
    """Preprocess Dynare macro directives and expression interpolations in `.mod` text.

    Parameters
    ----------
    text : str
        The raw Dynare `.mod` text containing macro directives.
    base_dir : Path | str | None, optional
        The directory against which relative @#include paths are resolved.
        Defaults to Path.cwd() if not provided.
    include_paths : Sequence[Path | str] | None, optional
        Additional search directories for @#include resolution.

    Returns
    -------
    str
        The preprocessed `.mod` text ready for model parsing.

    Raises
    ------
    DynareMacroError
        On syntax errors, undefined macro variables, unclosed blocks, or circular includes.
    """
    b_dir = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    inc_paths = [Path(p).resolve() for p in include_paths] if include_paths is not None else []

    root_scope = Scope()
    return _preprocess_macro_internal(
        text=text,
        base_dir=b_dir,
        include_paths=inc_paths,
        include_stack=[],
        scope=root_scope,
    )


__all__ = [
    "DynareMacroError",
    "Scope",
    "preprocess_macro",
]
