"""tools/release_check.py — single pre-tag gate for puremacro releases.

Run me before `git tag X.Y.Z`. Up to seven gates (all run; no fail-fast):

  Gate 1 (test baseline)    — pytest failures and setup errors must equal
                              tests/known_failures.json
  Gate 2 (Pyodide contract)  — tests/test_pyodide_compat.py must be green (static)
  Gate 3 (public API snap)   — re-generated snapshot must match the fixture
  Gate 4 (version sync)      — pyproject / __init__ / CHANGELOG / CITATION.cff /
                              playground wheel pin agree
  Gate 5 (examples gallery)  — opt-in via --examples; reads docs/examples_gallery.json
  Gate 6 (pyodide smoke)     — opt-in via --pyodide; boots Pyodide, runs marked tests
  Gate 7 (min-Python syntax) — every .py parses on the requires-python floor
                              (default-on, seconds); catches what only the
                              oldest CI leg would otherwise reject at collection

Exit 0 iff every gate run passed.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_pyproject_version(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise ValueError(f"version not found in {path}")
    return m.group(1)


def read_init_version(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise ValueError(f"__version__ not found in {path}")
    return m.group(1)


def read_changelog_version(path: Path) -> str:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+(\d+\.\d+\.\d+)", line)
        if m:
            return m.group(1)
    raise ValueError(f"no '## X.Y.Z' heading in {path}")


def read_citation_version(path: Path) -> str:
    """`version:` from CITATION.cff.

    Read as plain text rather than via a YAML parser: pyyaml is not a
    dependency of this repo, and the field is a single flat key.
    """
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r'^version:\s*"?([0-9][^"\s]*)"?\s*$', text, re.MULTILINE)
    if not m:
        raise ValueError(f"version not found in {path}")
    return m.group(1)


def read_playground_pin_version(path: Path) -> str:
    """Version of the wheel pinned in playground/jupyter_lite_config.json.

    ``build_playground.sh`` rewrites the pin from the wheel it builds, so the
    deployed site never lags — but the tracked file did (it read 1.9.0 while the
    package shipped 3.3.0). Reading it here makes it a fifth version-bearing
    file for Gate 4, as RELEASING.md asks for any new place the version lives.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    urls = data.get("PipliteAddon", {}).get("piplite_urls", [])
    for url in urls:
        m = re.search(r"puremacro-(\d+\.\d+\.\d+[0-9A-Za-z.]*)-", str(url))
        if m:
            return m.group(1)
    raise ValueError(f"no puremacro wheel pin in {path}")


def read_requires_python_min(path: Path) -> tuple[int, int]:
    """The ``(major, minor)`` floor from ``requires-python = ">=X.Y"`` in pyproject."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r'^requires-python\s*=\s*"\s*>=\s*(\d+)\.(\d+)', text, re.MULTILINE)
    if not m:
        raise ValueError(f"requires-python floor not found in {path}")
    return int(m.group(1)), int(m.group(2))


def load_whitelist(path: Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {e["nodeid"] for e in data.get("entries", [])}


def compare_failures(failing: set[str], whitelist: set[str]) -> dict:
    new = failing - whitelist
    recovered = whitelist - failing
    passed = len(new) == 0
    lines = []
    if passed and not recovered:
        lines.append(f"  Gate 1 (test baseline): PASS — {len(failing)} known red, no new")
    elif passed and recovered:
        lines.append(
            f"  Gate 1 (test baseline): PASS with warning — "
            f"{len(recovered)} previously-red now green; shrink whitelist."
        )
        for nid in sorted(recovered):
            lines.append(f"    recovered: {nid}")
    else:
        lines.append(f"  Gate 1 (test baseline): FAIL — {len(new)} new failure(s)")
        for nid in sorted(new):
            lines.append(f"    NEW: {nid}")
        if recovered:
            lines.append(f"    plus {len(recovered)} previously-red now green:")
            for nid in sorted(recovered):
                lines.append(f"    recovered: {nid}")
    return {
        "name": "test_baseline",
        "passed": passed,
        "report": "\n".join(lines),
        "new": new,
        "recovered": recovered,
    }


#: Wall-clock budget for Gate 1's full-suite run.  The suite takes ~47 min
#: locally and ~50 min on CI, so the previous hard-coded 1800s could never
#: finish and reported a green suite as a gate failure.
PYTEST_BASELINE_TIMEOUT_S = float(os.environ.get("PUREMACRO_BASELINE_TIMEOUT_S", 5400))


def run_pytest_collect_failures(repo_root: Path) -> set[str]:
    """Run the full suite (minus network) and return the red-nodeid set.

    Both ``FAILED`` and ``ERROR`` summary lines count: a fixture that raises
    turns every test using it into a setup ERROR, which pytest reports on its
    own line and which used to be invisible to this gate (sixteen of them hid
    behind a missing data file while CI was red).

    Raises:
        RuntimeError: if pytest itself errors out (collection error, internal error,
            usage error) rather than running tests normally.
        subprocess.TimeoutExpired: if the run exceeds ``PYTEST_BASELINE_TIMEOUT_S``.

    The budget is wall-clock, not CPU: the suite passed 13,900 tests in about
    47 minutes on an unloaded 12-core laptop and 50 minutes on CI, so the old
    1800s ceiling made this gate impossible to pass and reported a green suite
    as ``FAIL``. Override with ``PUREMACRO_BASELINE_TIMEOUT_S`` on a slower box.
    """
    cmd = [
        sys.executable, "-m", "pytest",
        "puremacro/tests/", "tests/",
        "-m", "not network and not slow",
        "--tb=no", "-q",
        "--no-header",
    ]
    proc = subprocess.run(
        cmd, cwd=repo_root, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=PYTEST_BASELINE_TIMEOUT_S,
    )
    for line in reversed(proc.stdout.splitlines()):
        if re.search(r"\b\d+ (passed|failed|errors?|skipped|deselected)\b", line):
            print(f"    pytest: {line}", flush=True)
            break
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"pytest exited with code {proc.returncode} "
            f"(expected 0 or 1). Last stderr lines:\n"
            + "\n".join(proc.stderr.splitlines()[-20:])
        )
    failures = set()
    for line in proc.stdout.splitlines():
        for prefix in ("FAILED ", "ERROR "):
            if line.startswith(prefix):
                nid = line[len(prefix):].split(" ", 1)[0]
                failures.add(nid)
    return failures


def gate_pyodide(repo_root: Path) -> dict:
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_pyodide_compat.py",
        "--tb=short", "-q",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": "pyodide",
            "passed": False,
            "report": "  Gate 2 (Pyodide contract): FAIL — pytest exceeded 300s timeout",
        }
    passed = proc.returncode == 0
    head = "  Gate 2 (Pyodide contract): " + ("PASS" if passed else "FAIL")
    tail = "" if passed else "\n" + "\n".join(
        f"    {line}" for line in proc.stdout.splitlines()[-20:]
    )
    return {"name": "pyodide", "passed": passed, "report": head + tail}


def gate_test_baseline(repo_root: Path) -> dict:
    whitelist = load_whitelist(repo_root / "tests" / "known_failures.json")
    try:
        failing = run_pytest_collect_failures(repo_root)
    except subprocess.TimeoutExpired:
        return {
            "name": "test_baseline",
            "passed": False,
            "report": (
                f"  Gate 1 (test baseline): FAIL — pytest exceeded "
                f"{PYTEST_BASELINE_TIMEOUT_S:.0f}s timeout (raise "
                f"PUREMACRO_BASELINE_TIMEOUT_S if this box is slower)"
            ),
            "new": set(),
            "recovered": set(),
        }
    except RuntimeError as e:
        return {
            "name": "test_baseline",
            "passed": False,
            "report": f"  Gate 1 (test baseline): FAIL — pytest could not run\n    {e}",
            "new": set(),
            "recovered": set(),
        }
    return compare_failures(failing, whitelist)


def compare_snapshot(fresh: dict, fixture_path: Path) -> dict:
    on_disk = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    if fresh == on_disk:
        return {
            "name": "public_api_snapshot",
            "passed": True,
            "report": "  Gate 3 (public API snapshot): PASS",
        }
    lines = ["  Gate 3 (public API snapshot): FAIL — diff:"]

    # Diff the "all" map (module → symbol list).
    fresh_modules = set(fresh.get("all", {}).keys())
    disk_modules = set(on_disk.get("all", {}).keys())
    for mod in sorted(fresh_modules - disk_modules):
        lines.append(f"    + module {mod}")
    for mod in sorted(disk_modules - fresh_modules):
        lines.append(f"    - module {mod}")
    for mod in sorted(fresh_modules & disk_modules):
        added = set(fresh["all"][mod]) - set(on_disk["all"][mod])
        removed = set(on_disk["all"][mod]) - set(fresh["all"][mod])
        for sym in sorted(added):
            lines.append(f"    + {mod}.{sym}")
        for sym in sorted(removed):
            lines.append(f"    - {mod}.{sym}")

    # Diff the "result_classes" map (class qualname → field list).
    fresh_classes = set(fresh.get("result_classes", {}).keys())
    disk_classes = set(on_disk.get("result_classes", {}).keys())
    for cls in sorted(fresh_classes - disk_classes):
        lines.append(f"    + class {cls}")
    for cls in sorted(disk_classes - fresh_classes):
        lines.append(f"    - class {cls}")
    for cls in sorted(fresh_classes & disk_classes):
        added = set(fresh["result_classes"][cls]) - set(on_disk["result_classes"][cls])
        removed = set(on_disk["result_classes"][cls]) - set(fresh["result_classes"][cls])
        for field in sorted(added):
            lines.append(f"    + {cls}.{field}")
        for field in sorted(removed):
            lines.append(f"    - {cls}.{field}")

    return {
        "name": "public_api_snapshot",
        "passed": False,
        "report": "\n".join(lines),
    }


def gate_snapshot(repo_root: Path) -> dict:
    # Import the helper from tests/ — Task 7 Step 1 promoted it to a public name.
    sys.path.insert(0, str(repo_root / "tests"))
    try:
        from test_public_api import collect_current_api  # type: ignore
    finally:
        sys.path.pop(0)
    fresh = collect_current_api()
    return compare_snapshot(fresh, repo_root / "tests" / "fixtures" / "public_api_snapshot.json")


def gate_examples_gallery(json_path: Path, *, examples_source_dir: Path) -> dict:
    """Gate 5 — examples gallery health.

    Reads docs/examples_gallery.json. Fails on any FAIL entry. Warns
    (does not fail) if the JSON's generated_at is older than the newest
    *.py file under examples_source_dir.
    """
    name = "examples_gallery"
    if not Path(json_path).exists():
        return {
            "name": name,
            "passed": False,
            "report": (
                f"  Gate 5 (examples gallery): FAIL — examples gallery not rendered\n"
                f"    expected at {json_path}\n"
                f"    run: python tools/render_examples_gallery.py"
            ),
        }
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {
            "name": name,
            "passed": False,
            "report": f"  Gate 5 (examples gallery): FAIL — JSON malformed: {e}",
        }
    examples = data.get("examples", {})
    counts = {"PASS": 0, "SKIP": 0, "FAIL": 0}
    fails = []
    for nm, e in examples.items():
        s = e.get("status", "FAIL")
        counts[s] = counts.get(s, 0) + 1
        if s == "FAIL":
            fails.append((nm, e.get("reason", "(no reason)")))

    # Stale-JSON warning (not a failure).
    warn_lines = []
    if examples_source_dir.exists():
        from datetime import datetime, timezone
        try:
            gen = datetime.strptime(data["generated_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            newest_src = max(
                (p.stat().st_mtime for p in examples_source_dir.glob("*.py")),
                default=0.0,
            )
            if newest_src > gen.timestamp():
                warn_lines.append(
                    "    stale: an example source file is newer than the gallery JSON; "
                    "consider re-rendering"
                )
        except (KeyError, ValueError):
            pass

    if fails:
        lines = [f"  Gate 5 (examples gallery): FAIL — {len(fails)} example(s) failed"]
        for nm, reason in sorted(fails):
            lines.append(f"    FAIL {nm} — {reason}")
        lines.extend(warn_lines)
        return {"name": name, "passed": False, "report": "\n".join(lines)}

    head = f"  Gate 5 (examples gallery): PASS — {counts['PASS']} PASS, {counts['SKIP']} SKIP, 0 FAIL"
    if warn_lines:
        return {"name": name, "passed": True, "report": "\n".join([head] + warn_lines)}
    return {"name": name, "passed": True, "report": head}


def gate_pyodide_smoke(repo_root: Path) -> dict:
    """Gate 6 — real Pyodide smoke.

    Delegates to tools/pyodide_smoke.py::run, which builds the wheel +
    invokes the Node runner. Slow gate (~60-180s typical; can be ~6s
    on modern hardware); opt-in via --pyodide.
    """
    sys.path.insert(0, str(repo_root / "tools"))
    try:
        import pyodide_smoke
    finally:
        sys.path.pop(0)
    result = pyodide_smoke.run(repo_root)
    return {"name": "pyodide_smoke", **result}


def gate_version_sync(
    *,
    pyproject_version: str,
    init_version: str,
    changelog_version: str,
    citation_version: str,
    playground_pin_version: str | None = None,
) -> dict:
    # CITATION.cff is here because it silently drifted: it still read 1.3.0
    # while the package shipped 1.3.1, and nothing in the release path noticed.
    # The playground wheel pin joined for the same reason (1.9.0 at 3.3.0); it
    # is optional so callers without a playground checkout can skip it.
    versions = {
        "pyproject.toml": pyproject_version,
        "puremacro/__init__.py": init_version,
        "CHANGELOG.md": changelog_version,
        "CITATION.cff": citation_version,
    }
    if playground_pin_version is not None:
        versions["playground/jupyter_lite_config.json"] = playground_pin_version
    passed = len(set(versions.values())) == 1
    if passed:
        report = f"  Gate 4 (version sync): PASS — all read {pyproject_version}"
    else:
        lines = ["  Gate 4 (version sync): FAIL"]
        for name, ver in versions.items():
            lines.append(f"    {name:<30} {ver}")
        report = "\n".join(lines)
    return {"name": "version_sync", "passed": passed, "report": report}


# ---------------------------------------------------------------------------
# Gate 7 — minimum-supported-Python syntax
# ---------------------------------------------------------------------------

#: Directories whose .py files must parse on the requires-python floor: the
#: shipped package, the release tooling and the suite every CI leg collects.
#: notebooks/ is deliberately absent: jupytext percent sources may carry bare
#: IPython magics (`%matplotlib inline`), which no Python version parses.
SYNTAX_ROOTS: tuple[str, ...] = ("puremacro", "tools", "tests")
_SYNTAX_SKIP_DIRS = frozenset(
    {"__pycache__", "node_modules", ".git", "dist", "build", ".ipynb_checkpoints"}
)
_STRING_PREFIX_CHARS = "rRbBuUfFtT"


def _quote_of(token_string: str) -> str:
    """Opening quote character of a STRING / FSTRING_START token."""
    return token_string.lstrip(_STRING_PREFIX_CHARS)[:1]


def _is_triple_quoted(token_string: str) -> bool:
    body = token_string.lstrip(_STRING_PREFIX_CHARS)
    return body[:3] in ('"""', "'''")


def scan_fstrings_for_pep701(src: str, filename: str = "<src>") -> list[str]:
    """Return the f-string constructs in ``src`` that only Python >= 3.12 accepts.

    ``ast.parse(..., feature_version=(3, 11))`` on a 3.12+ interpreter still
    accepts PEP 701 f-strings (quote reuse, backslashes and comments inside the
    expression part, newlines inside single-quoted f-strings), so a gate that
    only calls ``ast.parse`` passes code the 3.11 CI legs reject at collection.
    This walks the 3.12+ tokenizer output instead, where every f-string is an
    explicit FSTRING_START .. FSTRING_END span, and applies the pre-3.12 rules,
    each checked against CPython 3.11.15 (tests/test_release_check_34fixes.py):

    * inside a replacement field, no string literal (or nested f-string) may
      open with the quote character of any enclosing single-quoted f-string;
    * inside a replacement field, no string token may contain a backslash
      (a backslash in the *format spec* is fine, as it was before 3.12);
    * inside a replacement field, no comment;
    * a single-quoted f-string may not span lines, even inside a field.

    Empty on interpreters older than 3.12: their tokenizer has no FSTRING_START
    token, and their own parser already rejects these forms.
    """
    if sys.version_info < (3, 12):
        return []
    fstring_start = getattr(tokenize, "FSTRING_START", None)
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    fstring_end = getattr(tokenize, "FSTRING_END", None)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError) as e:
        return [f"{filename}: cannot tokenize: {e}"]

    problems: list[str] = []
    # One entry per open f-string: its quote, whether it is triple-quoted, and
    # the brace depth inside it (> 0 means "inside a replacement field").
    stack: list[dict] = []

    def flag(tok: tokenize.TokenInfo, msg: str) -> None:
        problems.append(f"{filename}:{tok.start[0]}: {msg} (PEP 701; Python >= 3.12 only)")

    for tok in tokens:
        in_field = bool(stack) and stack[-1]["depth"] > 0
        if tok.type == fstring_start:
            quote = _quote_of(tok.string)
            if in_field and any(not e["triple"] and e["quote"] == quote for e in stack):
                flag(tok, "nested f-string reuses the enclosing f-string's quote")
            stack.append({"quote": quote, "triple": _is_triple_quoted(tok.string), "depth": 0})
            continue
        if not stack:
            continue
        if tok.type == fstring_end:
            stack.pop()
            continue
        if tok.type == fstring_middle:
            # Literal text or format spec of the innermost f-string: fine for
            # that f-string, but not when the whole nested f-string sits in an
            # outer f-string's expression part (or spans an outer single-quoted
            # f-string's line).
            if "\\" in tok.string and any(e["depth"] > 0 for e in stack[:-1]):
                flag(tok, "backslash inside an f-string expression part")
            if "\n" in tok.string and any(not e["triple"] for e in stack[:-1]):
                flag(tok, "newline inside a single-quoted f-string expression")
            continue
        if tok.type == tokenize.OP and tok.string == "{":
            stack[-1]["depth"] += 1
            continue
        if tok.type == tokenize.OP and tok.string == "}":
            stack[-1]["depth"] = max(0, stack[-1]["depth"] - 1)
            continue
        if not in_field:
            continue
        if tok.type == tokenize.STRING:
            quote = _quote_of(tok.string)
            if any(not e["triple"] and e["quote"] == quote for e in stack):
                flag(tok, "string literal inside an f-string expression reuses the f-string's quote")
            if "\\" in tok.string:
                flag(tok, "backslash inside an f-string expression part")
        elif tok.type == tokenize.COMMENT:
            flag(tok, "comment inside an f-string expression part")
        elif tok.type in (tokenize.NL, tokenize.NEWLINE):
            if any(not e["triple"] for e in stack):
                flag(tok, "newline inside a single-quoted f-string expression")
    return problems


def _may_have_pep701_fstrings(src: str, tree: ast.AST) -> bool:
    """Cheap pre-filter for the tokenizer scan: does any f-string *look* suspicious?

    Tokenizing every file costs seconds across the tree; walking the AST that
    ``check_min_python_syntax`` already built costs milliseconds. Every PEP 701
    form puts one of four things between an f-string's outer quotes — its own
    quote character, a backslash, a ``#`` or a newline — so a file whose
    f-string source segments contain none of them cannot need the scan. The
    check is a superset (triple-quoted and implicitly concatenated f-strings
    trigger it too); the tokenizer remains the judge.
    """
    lines = [ln.encode("utf-8", errors="surrogatepass") for ln in src.splitlines(keepends=True)]
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr) or node.end_lineno is None:
            continue
        first, last = node.lineno - 1, node.end_lineno - 1
        if first >= len(lines) or last >= len(lines):
            return True
        if first == last:
            seg = lines[first][node.col_offset:node.end_col_offset]
        else:
            seg = (
                lines[first][node.col_offset:]
                + b"".join(lines[first + 1:last])
                + lines[last][:node.end_col_offset]
            )
        body = seg.decode("utf-8", errors="replace").lstrip(_STRING_PREFIX_CHARS)
        quote, inner = body[:1], body[1:-1]
        if "\\" in inner or "#" in inner or "\n" in inner or quote in inner:
            return True
    return False


def check_min_python_syntax(
    path: Path, min_version: tuple[int, int], *, display: str | None = None
) -> list[str]:
    """Everything that stops ``path`` from parsing on Python ``min_version``.

    ``ast.parse(feature_version=...)`` rejects grammar the floor lacks (PEP 695
    ``type`` statements and generics, for instance); the PEP 701 scan covers the
    f-string forms it lets through. Each problem is ``display:line: message``.
    """
    label = display or str(path)
    data = Path(path).read_bytes()
    problems: list[str] = []
    tree: ast.AST | None = None
    try:
        tree = ast.parse(data, filename=label, feature_version=min_version)
    except SyntaxError as e:
        problems.append(f"{label}:{e.lineno}: {e.msg}")
    except ValueError as e:  # e.g. a null byte in the source
        problems.append(f"{label}: {e}")
    if min_version < (3, 12) and sys.version_info >= (3, 12):
        try:
            src = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            src = data.decode("utf-8", errors="replace")
        if tree is None or _may_have_pep701_fstrings(src, tree):
            problems.extend(scan_fstrings_for_pep701(src, label))
    return problems


def iter_syntax_files(repo_root: Path, roots: tuple[str, ...] = SYNTAX_ROOTS):
    """Yield every .py under ``roots`` that Gate 7 must parse, in sorted order."""
    repo_root = Path(repo_root)
    for root in roots:
        base = repo_root / root
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(repo_root)
            if _SYNTAX_SKIP_DIRS.intersection(rel.parts[:-1]):
                continue
            yield p


# Parsing ~17 MB of source takes ~10 s in one process, so the file list is
# fanned out across a few subprocesses. Paths travel on stdin (argv would
# overflow on Windows); results come back one problem per stdout line, then a
# CHECKED sentinel that tells a crash from a clean run.
_CHECKED_SENTINEL = "CHECKED"

# Runs under THIS interpreter: imports the gate by path (no pickling, works
# however release_check was loaded) and calls check_min_python_syntax.
_WORKER_SCRIPT = (
    "import importlib.util, sys\n"
    "spec = importlib.util.spec_from_file_location('_release_check_worker', sys.argv[1])\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "floor = tuple(int(x) for x in sys.argv[2].split('.'))\n"
    "for rel in sys.stdin.read().splitlines():\n"
    "    if rel:\n"
    "        for problem in mod.check_min_python_syntax(rel, floor, display=rel):\n"
    "            print(problem)\n"
    "print('CHECKED')\n"
)

# Runs under a real interpreter of the floor version when one is on PATH; the
# script is plain 3.x so that interpreter can execute it.
_COMPILE_UNDER_SCRIPT = (
    "import sys\n"
    "for line in sys.stdin.read().splitlines():\n"
    "    if not line:\n"
    "        continue\n"
    "    try:\n"
    "        with open(line, 'rb') as fh:\n"
    "            compile(fh.read(), line, 'exec')\n"
    "    except SyntaxError as e:\n"
    "        print(line + ':' + str(e.lineno) + ': ' + str(e.msg))\n"
    "print('CHECKED')\n"
)


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _fan_out(
    cmd: list[str], rels: list[str], *, cwd: Path, workers: int, timeout: float = 300.0
) -> list[str] | None:
    """Run ``cmd`` in ``workers`` subprocesses, each fed a slice of ``rels`` on stdin.

    Returns the concatenated problem lines, or ``None`` if any worker could not
    be started, timed out, crashed, or did not print the sentinel — the caller
    then falls back to doing the work itself.
    """
    slices = [rels[i::workers] for i in range(max(1, workers))]
    procs: list[subprocess.Popen] = []
    try:
        for sl in slices:
            if not sl:
                continue
            proc = subprocess.Popen(
                cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            )
            procs.append(proc)
            assert proc.stdin is not None
            proc.stdin.write("\n".join(sl) + "\n")
            proc.stdin.close()
            # communicate() otherwise tries to flush this already-closed pipe.
            proc.stdin = None
        problems: list[str] = []
        for proc in procs:
            out, _err = proc.communicate(timeout=timeout)
            lines = out.splitlines()
            if proc.returncode != 0 or _CHECKED_SENTINEL not in lines:
                return None
            problems.extend(ln for ln in lines if ln and ln != _CHECKED_SENTINEL)
        return problems
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()


def _check_files(
    repo_root: Path, files: list[Path], min_version: tuple[int, int], workers: int
) -> list[str]:
    """``check_min_python_syntax`` over ``files``, in parallel when it pays."""
    rels = [p.relative_to(repo_root).as_posix() for p in files]
    if workers > 1 and len(rels) >= 64:
        tag = f"{min_version[0]}.{min_version[1]}"
        cmd = [sys.executable, "-c", _WORKER_SCRIPT, str(Path(__file__).resolve()), tag]
        result = _fan_out(cmd, rels, cwd=repo_root, workers=workers)
        if result is not None:
            return result
    problems: list[str] = []
    for p, rel in zip(files, rels):
        problems.extend(check_min_python_syntax(p, min_version, display=rel))
    return problems


def _compile_under(
    exe: str, repo_root: Path, files: list[Path], workers: int = 1
) -> tuple[list[str], str]:
    """Compile ``files`` under ``exe``; returns (problems, note for the report)."""
    rels = [p.relative_to(repo_root).as_posix() for p in files]
    result = _fan_out([exe, "-c", _COMPILE_UNDER_SCRIPT], rels, cwd=repo_root, workers=workers)
    if result is None:
        return [], f"{exe} could not run the compile pass; in-process checks only"
    return result, f"compile() under {exe}"


def gate_min_python_syntax(
    repo_root: Path,
    *,
    roots: tuple[str, ...] = SYNTAX_ROOTS,
    min_version: tuple[int, int] | None = None,
    workers: int | None = None,
) -> dict:
    """Gate 7 — every .py under ``roots`` parses on the requires-python floor.

    Gates 1-6 run under whatever interpreter invokes the script, so a
    SyntaxError that only the floor version raises passed every one of them
    while the oldest CI leg died at collection. Default-on; a few seconds.
    """
    name = "min_python_syntax"
    repo_root = Path(repo_root)
    if min_version is None:
        min_version = read_requires_python_min(repo_root / "pyproject.toml")
    if workers is None:
        workers = _default_workers()
    tag = f"{min_version[0]}.{min_version[1]}"
    files = list(iter_syntax_files(repo_root, roots))
    problems = _check_files(repo_root, files, min_version, workers)
    methods = [f"ast.parse(feature_version={tag})"]
    if min_version < (3, 12):
        methods.append("PEP 701 f-string scan" if sys.version_info >= (3, 12) else "native parser")
    exe = None if sys.version_info[:2] == min_version else shutil.which(f"python{tag}")
    if exe:
        extra, note = _compile_under(exe, repo_root, files, workers=workers)
        problems.extend(extra)
        methods.append(note)
    problems = sorted(set(problems))
    head = f"  Gate 7 (min-Python {tag} syntax): "
    if problems:
        lines = [head + f"FAIL — {len(problems)} problem(s) in files that must parse on Python {tag}"]
        lines.extend(f"    {p}" for p in problems[:25])
        if len(problems) > 25:
            lines.append(f"    ... {len(problems) - 25} more")
        return {"name": name, "passed": False, "report": "\n".join(lines)}
    return {
        "name": name,
        "passed": True,
        "report": head + f"PASS — {len(files)} files parse ({'; '.join(methods)})",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="release_check",
        description="Release-gate for puremacro — runs 5 default gates + up to 2 opt-in pre-tag.",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip Gate 1 (test baseline). Useful for fast iteration on the other gates.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print which gates would run and exit 0, without executing any.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Also run Gate 5 (examples gallery health). Reads docs/examples_gallery.json.",
    )
    parser.add_argument(
        "--pyodide",
        action="store_true",
        help="Also run Gate 6 (real Pyodide smoke). Builds the wheel + boots "
             "Pyodide via node tools/pyodide/runner.js. Slow (~60-180s); "
             "requires node + one-time `npm install` in tools/pyodide/.",
    )
    args = parser.parse_args(argv)

    print("release_check — puremacro pre-tag gate")
    print(f"  repo root: {REPO_ROOT}")
    if args.report_only:
        print("  (report-only — no gates executed)")
        return 0

    gates = []
    if not args.no_tests:
        g1 = gate_test_baseline(REPO_ROOT)
        gates.append(g1)
    g2 = gate_pyodide(REPO_ROOT)
    gates.append(g2)
    g3 = gate_snapshot(REPO_ROOT)
    gates.append(g3)
    playground_cfg = REPO_ROOT / "playground" / "jupyter_lite_config.json"
    g4 = gate_version_sync(
        pyproject_version=read_pyproject_version(REPO_ROOT / "pyproject.toml"),
        init_version=read_init_version(REPO_ROOT / "puremacro" / "__init__.py"),
        changelog_version=read_changelog_version(REPO_ROOT / "CHANGELOG.md"),
        citation_version=read_citation_version(REPO_ROOT / "CITATION.cff"),
        playground_pin_version=(
            read_playground_pin_version(playground_cfg) if playground_cfg.exists() else None
        ),
    )
    gates.append(g4)
    g7 = gate_min_python_syntax(REPO_ROOT)
    gates.append(g7)
    if args.examples:
        g5 = gate_examples_gallery(
            REPO_ROOT / "docs" / "examples_gallery.json",
            examples_source_dir=REPO_ROOT / "puremacro" / "examples",
        )
        gates.append(g5)
    if args.pyodide:
        g6 = gate_pyodide_smoke(REPO_ROOT)
        gates.append(g6)

    for g in gates:
        print(g["report"])

    passed_count = sum(1 for g in gates if g["passed"])
    total = len(gates)
    failed = [g["name"] for g in gates if not g["passed"]]
    if not failed:
        print(f"\nall {total} gates PASS")
        return 0
    print(f"\n{passed_count}/{total} gates passed; FAIL: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
