#!/usr/bin/env python3
"""Find tests that cannot fail, by breaking the code and seeing what stays green.

    python tools/mutation_check.py puremacro/capital.py
    python tools/mutation_check.py puremacro/fetch/oecd_qna_panel.py \
        --tests tests/test_oecd_qna_panel.py --max-mutants 60

A *survivor* is a mutation that the tests did not notice. That is not always a
missing test — some mutations are semantically inert — but every survivor is a
question worth answering, and in practice the interesting ones are guards: a
filter, a `dropna`, a mask that can be deleted with the suite still green
because no fixture ever produces the row it removes.

This is the third of the three layers described in CONTRIBUTING's "Making sure a
test can fail", and it is the only one that finds that case. The other two —
positive controls for a test's own mechanism, and coverage attribution — live in
``tests/test_test_quality.py`` and run on every push. This does not: mutation
cost is (mutants x suite time), so it is a tool you point at a module, not a
gate. Run it when you add a module, or when a test file looks suspiciously green.

What it deliberately does not do: mutate everything. The operators below are
aimed at the mistakes that actually got through here — deleted guards, flipped
comparisons, neutralised constants — rather than at exhaustive coverage of the
grammar. A survivor list nobody reads is worth nothing.
"""
from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

#: Calls that mark a statement as a guard: it removes or reshapes rows, so
#: deleting it is exactly the "the fixture never produced that row" mutation.
_GUARD_CALLS = ("isin", "dropna", "notna", "notnull", "clip", "drop_duplicates",
                "fillna", "astype", "round")

_COMPARE_FLIP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}


@dataclass
class Mutant:
    """One applied change: what, where, and how to describe it in a report."""
    kind: str
    lineno: int
    description: str
    source: str


class _Mutator(ast.NodeTransformer):
    """Applies exactly the ``target``-th mutation found, and no other."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.seen = 0
        self.applied: Mutant | None = None
        self._lines: list[str] = []

    def _take(self, kind: str, node: ast.AST, description: str):
        """Return True if this site is the one to mutate."""
        if self.seen != self.target:
            self.seen += 1
            return False
        self.seen += 1
        src = self._lines[node.lineno - 1].strip() if node.lineno - 1 < len(self._lines) else ""
        self.applied = Mutant(kind, node.lineno, description, src)
        return True

    # -- guard deletion ---------------------------------------------------
    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        calls = {n.func.attr for n in ast.walk(node.value)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        is_guard = bool(calls & set(_GUARD_CALLS)) or any(
            isinstance(n, (ast.Compare, ast.BoolOp)) for n in ast.walk(node.value))
        if is_guard and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            # only safe to drop a self-assignment (`df = df[...]`); dropping a
            # fresh binding would raise NameError and kill every mutant trivially
            if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node.value)):
                if self._take("guard-deleted", node, f"dropped the guard on `{name}`"):
                    return ast.Pass()
        return node

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if (len(node.body) == 1
                and isinstance(node.body[0], (ast.Continue, ast.Return, ast.Raise))
                and not node.orelse):
            what = type(node.body[0]).__name__.lower()
            if self._take("guard-deleted", node, f"removed an early-{what} guard"):
                return ast.Pass()
        return node

    # -- comparison flips -------------------------------------------------
    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if len(node.ops) == 1 and type(node.ops[0]) in _COMPARE_FLIP:
            old = type(node.ops[0])
            new = _COMPARE_FLIP[old]
            if self._take("compare-flipped", node,
                          f"{old.__name__} -> {new.__name__}"):
                node.ops = [new()]
        return node

    # -- constants --------------------------------------------------------
    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool):
            if self._take("bool-flipped", node, f"{node.value} -> {not node.value}"):
                return ast.Constant(value=not node.value)
        elif isinstance(node.value, (int, float)) and node.value not in (0, 1):
            if self._take("constant-neutralised", node, f"{node.value!r} -> 1"):
                return ast.Constant(value=1)
        return node


def _count_sites(tree: ast.AST, lines: list[str]) -> int:
    m = _Mutator(target=-1)
    m._lines = lines
    m.visit(ast.parse(ast.unparse(tree)))
    return m.seen


def _apply(source: str, index: int) -> tuple[str, Mutant] | None:
    tree = ast.parse(source)
    m = _Mutator(target=index)
    m._lines = source.splitlines()
    mutated = m.visit(tree)
    if m.applied is None:
        return None
    ast.fix_missing_locations(mutated)
    return ast.unparse(mutated), m.applied


def _default_tests(module: Path) -> list[str]:
    """`puremacro/fetch/oecd_qna_panel.py` -> `tests/test_oecd_qna_panel.py`."""
    guess = _ROOT / "tests" / f"test_{module.stem}.py"
    return [str(guess.relative_to(_ROOT))] if guess.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("module", help="path to the module to mutate")
    ap.add_argument("--tests", nargs="*", default=None,
                    help="test files to run (default: tests/test_<module>.py)")
    ap.add_argument("--max-mutants", type=int, default=0,
                    help="stop after this many (0 = all). Reported if it truncates.")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    module = (_ROOT / args.module).resolve()
    if not module.exists():
        print(f"no such module: {args.module}")
        return 2
    tests = args.tests if args.tests is not None else _default_tests(module)
    if not tests:
        print(f"no test file found for {args.module}; pass --tests explicitly")
        return 2

    source = module.read_text()
    total = _count_sites(ast.parse(source), source.splitlines())
    rel = module.relative_to(_ROOT)
    print(f"mutation_check — {rel}")
    print(f"  tests: {' '.join(tests)}")
    print(f"  mutation sites: {total}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(_ROOT, work, symlinks=True, ignore=shutil.ignore_patterns(
            ".git", ".mypy_cache", ".pytest_cache", "__pycache__", "dist",
            "notebooks", "data", "playground", "paper"))
        (work / "data").symlink_to(_ROOT / "data", target_is_directory=True)
        target = work / rel

        def run() -> bool:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", *tests, "-q", "-x",
                 "-p", "no:randomly"],
                cwd=work, capture_output=True, text=True, timeout=args.timeout)
            return p.returncode == 0

        target.write_text(source)
        if not run():
            print("  BASELINE FAILS — fix the tests first; survivors are meaningless")
            return 2
        print("  baseline: green\n")

        limit = args.max_mutants or total
        survivors: list[Mutant] = []
        killed = inert = 0
        for i in range(min(total, limit)):
            applied = _apply(source, i)
            if applied is None:
                continue
            mutated_src, mutant = applied
            try:
                compile(mutated_src, str(target), "exec")
            except SyntaxError:
                inert += 1
                continue
            target.write_text(mutated_src)
            if run():
                survivors.append(mutant)
                print(f"  SURVIVED  {rel}:{mutant.lineno}  {mutant.kind}: "
                      f"{mutant.description}\n            {mutant.source[:96]}")
            else:
                killed += 1
        target.write_text(source)

    print(f"\n  {killed} killed, {len(survivors)} survived"
          + (f", {inert} unparseable" if inert else ""))
    if limit < total:
        print(f"  NOTE: stopped at {limit} of {total} sites (--max-mutants); "
              "the rest were not tried")
    if survivors:
        print("\n  Each survivor is a change the tests did not notice. Some are "
              "semantically inert;\n  the ones worth acting on are guards a "
              "fixture never exercises.")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
