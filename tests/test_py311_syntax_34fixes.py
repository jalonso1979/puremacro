"""Regression gate: every shipped module must parse on the oldest supported Python (3.11).

puremacro 3.3.0 shipped ``puremacro/trade/plot.py`` with a backslash inside an
f-string replacement field, which PEP 701 allows only from Python 3.12. On 3.11
that was a SyntaxError that made ``puremacro.trade``, ``puremacro.spatial`` and
``puremacro.did`` unimportable, and nothing in the release gate caught it because
the heavier CI steps run on 3.12 only.

``ast.parse(feature_version=(3, 11))`` does *not* reject PEP 701 f-strings, so this
test does two things:

1. compiles every ``puremacro/**/*.py`` with the running interpreter (a real
   check when the suite runs on 3.11);
2. on 3.12+, walks the tokenizer output and flags the PEP 701-only constructs
   (nested same-quote strings, backslashes, comments or newlines inside a
   replacement field), which is how the defect is detected on newer interpreters.
"""
from __future__ import annotations

import ast
import io
from pathlib import Path
import sys
import tokenize

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "puremacro"
SOURCES = sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _pep701_issues(source: str) -> list[tuple[int, str]]:
    """Return (line, reason) for every PEP 701-only f-string construct in ``source``."""
    issues: list[tuple[int, str]] = []
    if not hasattr(tokenize, "FSTRING_START"):  # Python < 3.12: compile() is the real check
        return issues
    toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    stack: list[tuple[str, int]] = []
    for i, tok in enumerate(toks):
        if tok.type == tokenize.FSTRING_START:
            quote = tok.string.lstrip("rRfFbBuUtT")
            stack.append((quote, i))
        elif tok.type == tokenize.FSTRING_END and stack:
            quote, start = stack.pop()
            triple = len(quote) == 3
            for inner in toks[start + 1 : i]:
                if inner.type == tokenize.FSTRING_MIDDLE:
                    continue
                if inner.type == tokenize.STRING:
                    if "\\" in inner.string:
                        issues.append((inner.start[0], "backslash inside f-string replacement field"))
                    if not triple and quote[0] in inner.string:
                        issues.append((inner.start[0], "same-quote string nested inside f-string replacement field"))
                    if triple and quote in inner.string:
                        issues.append((inner.start[0], "triple quote nested inside f-string replacement field"))
                elif inner.type == tokenize.COMMENT:
                    issues.append((inner.start[0], "comment inside f-string replacement field"))
                elif not triple and inner.type in (tokenize.NL, tokenize.NEWLINE):
                    issues.append((inner.start[0], "newline inside single-quoted f-string replacement field"))
    return issues


def test_package_has_sources():
    assert len(SOURCES) > 100, "package sources not found; PACKAGE_ROOT is wrong"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PACKAGE_ROOT)))
def test_module_parses_on_python_311(path: Path):
    source = path.read_text(encoding="utf-8")
    # 1. syntax accepted by the running interpreter and by the 3.11 grammar
    compile(source, str(path), "exec")
    ast.parse(source, filename=str(path), feature_version=(3, 11))
    # 2. PEP 701 constructs the 3.11 tokenizer rejects (detectable only on 3.12+)
    issues = _pep701_issues(source)
    assert not issues, (
        f"{path.relative_to(PACKAGE_ROOT)} uses Python 3.12-only f-string syntax "
        f"(pyproject declares requires-python >= 3.11): {issues}"
    )


def test_tokenizer_scan_detects_the_original_defect():
    """The scanner must flag the exact construct that broke 3.3.0 on Python 3.11."""
    if sys.version_info < (3, 12):
        pytest.skip("tokenizer-based scan only runs on Python 3.12+")
    bad = "label = f\"Terms ({scenario.replace('_', r'\\_')})\"\n"
    assert any("backslash" in reason for _, reason in _pep701_issues(bad))
    good = "label = scenario.replace('_', r'\\_')\ntitle = f\"Terms ({label})\"\n"
    assert _pep701_issues(good) == []
