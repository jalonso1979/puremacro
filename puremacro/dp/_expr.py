"""Expression layer of puremacro.dp: parse model equations, emit xp-generic code.

Equations are written as strings in Dynare's expression syntax (``^`` for
powers, ``x(+1)`` for next-period values, ``exp``/``log``/``sqrt``/``abs``/
``max``/``min``) and parsed by the same lexer and recursive-descent parser as
``.mod`` files (``puremacro.dsge._parser``). The result is a ``dsge._ast`` tree,
so the model stays symbolic: it can be inspected, differentiated and rendered.

Code generation is done here rather than through ``Node.to_python`` because the
solvers need broadcastable array code on any backend namespace (``xp``), with
each symbol mapped to a positional argument the compiler controls.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping

import numpy as np

from puremacro.dsge._ast import BinOp, Call, Const, Node, Param, UnaryOp, Var
from puremacro.dsge._parser import DynareParseError, Parser, Tokenizer

_FUNCS = {
    "exp": "xp.exp",
    "log": "xp.log",
    "ln": "xp.log",
    "sqrt": "xp.sqrt",
    "abs": "xp.abs",
    "max": "xp.maximum",
    "min": "xp.minimum",
}
_BINOPS = {"+": "+", "-": "-", "*": "*", "/": "/", "^": "**",
           "<": "<", "<=": "<=", ">": ">", ">=": ">=", "==": "==", "=": "==", "!=": "!="}
_RELATIONAL = {"<", "<=", ">", ">=", "==", "=", "!="}
_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)(.*)$", re.DOTALL)


class ModelSpecError(ValueError):
    """A dp model is incomplete, inconsistent, or uses an unsupported construct."""


def crra(c, gamma, xp=np):
    """CRRA felicity ``(c^(1-gamma) - 1) / (1 - gamma)``, equal to ``log(c)`` at gamma = 1.

    The ``-1`` makes log utility the gamma -> 1 limit; it shifts the value
    function by a constant and leaves policies unchanged.
    """
    if float(gamma) == 1.0:
        return xp.log(c)
    return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)


def parse(text: str, variables=()) -> Node:
    """Parse one expression into a ``dsge._ast`` node.

    ``variables`` are the names that may carry a timing index (states, shocks,
    choices); every other identifier parses as a ``Param`` and is resolved by the
    compiler (parameter, price, local, aggregate or ``age``).
    """
    src = str(text).replace("**", "^")
    try:
        tokens = Tokenizer(src).tokenize()
        parser = Parser(tokens, src)
        parser.variables = list(variables)
        node = parser.parse_expression()
        if parser._curr().type != "EOF":
            tok = parser._curr()
            raise DynareParseError(f"unexpected {tok.value!r} after the expression", tok.line, tok.col)
    except DynareParseError as exc:
        raise ModelSpecError(f"cannot parse {text!r}: {exc}") from None
    return node


def split_assignment(text: str) -> tuple[str, str]:
    """Split ``"name = expr"`` into ``(name, expr)``; raise if it is not one."""
    m = _ASSIGN.match(str(text))
    if not m:
        raise ModelSpecError(f"expected 'name = expression', got {text!r}")
    return m.group(1), m.group(2).strip()


def is_relational(node: Node) -> bool:
    return isinstance(node, BinOp) and node.op in _RELATIONAL


def symbols(node: Node) -> set[tuple[str, int]]:
    """All (name, lead) symbols in ``node``; parameters appear with lead 0."""
    return set(node.variables()) | {(p, 0) for p in node.parameters()}


def emit(node: Node, resolve: Callable[[str, int], str]) -> str:
    """Python source for ``node``; ``resolve(name, lead)`` gives each symbol's identifier."""
    if isinstance(node, Const):
        return repr(float(node.value))
    if isinstance(node, Param):
        return resolve(node.name, 0)
    if isinstance(node, Var):
        return resolve(node.name, node.lead)
    if isinstance(node, UnaryOp):
        return f"({node.op}{emit(node.expr, resolve)})"
    if isinstance(node, BinOp):
        if node.op not in _BINOPS:
            raise ModelSpecError(f"unsupported operator {node.op!r}")
        return f"({emit(node.left, resolve)} {_BINOPS[node.op]} {emit(node.right, resolve)})"
    if isinstance(node, Call):
        fn = node.func.lower()
        args = [emit(a, resolve) for a in node.args]
        if fn == "crra":
            if len(args) != 2:
                raise ModelSpecError("crra(c, gamma) takes two arguments")
            return f"_crra({args[0]}, {args[1]}, xp)"
        if fn not in _FUNCS:
            raise ModelSpecError(
                f"function {node.func!r} is not supported in dp models; "
                f"use one of {sorted(_FUNCS) + ['crra']}"
            )
        if fn in ("max", "min"):
            if len(args) != 2:
                raise ModelSpecError(f"{fn}() takes two arguments")
        elif len(args) != 1:
            raise ModelSpecError(f"{fn}() takes one argument")
        return f"{_FUNCS[fn]}({', '.join(args)})"
    raise ModelSpecError(f"unsupported expression node {type(node).__name__}")


def compile_function(name: str, args: list[str], body: list[str], ret: str,
                     namespace: Mapping | None = None) -> Callable:
    """Build ``def name(*args, xp=np): body; return ret`` with numpy warnings silenced.

    Infeasible grid points routinely hit ``log`` of a negative number before the
    feasibility mask removes them, so floating-point warnings are suppressed
    inside the generated function only.
    """
    lines = [f"def {name}({', '.join(args + ['xp=np'])}):",
             "    with np.errstate(all='ignore'):"]
    lines += [f"        {b}" for b in body]
    lines.append(f"        return {ret}")
    src = "\n".join(lines)
    glb = {"np": np, "_crra": crra, "_inf": np.inf}
    if namespace:
        glb.update(namespace)
    loc: dict = {}
    exec(compile(src, f"<puremacro.dp:{name}>", "exec"), glb, loc)
    fn = loc[name]
    fn.__source__ = src
    return fn

