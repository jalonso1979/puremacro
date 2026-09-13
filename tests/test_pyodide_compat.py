"""Pyodide-compatibility regression test.

`puremacro`'s numerical core is "pure numpy + scipy + pandas +
matplotlib", which is what keeps it importable under Pyodide (iPad /
juno.sh). That promise is **load-bearing** — if a contributor adds a
top-level ``import statsmodels`` (or ``linearmodels`` / ``arch``) the
package silently stops working on the intended deployment target.

Note the distinction the last tests in this file make explicit: the
*import* contract (four packages) is narrower than the set of *declared*
runtime dependencies (five — ``requests`` is also required to install, and
is documented in ``ARCHITECTURE.md``). All five ship with Pyodide, which the
JupyterLite playground's install depends on.

This test imports every shippable submodule and asserts none of the
dev-only optional dependencies leaked into ``sys.modules``. The
allow-list intentionally **excludes**:

* ``puremacro.examples.*`` — research scripts, not part of the wheel.
* ``puremacro.teaching`` — MATLAB-parity teaching prototypes; these
  intentionally use statsmodels / linearmodels / arch to compare
  puremacro's pure-numpy estimators against canonical packages. Not
  part of the Pyodide promise.
* ``puremacro.narrative.sources`` — HTTP/scraping side-channel,
  documented as out-of-scope for Pyodide in ``ARCHITECTURE.md``.
* ``puremacro.narrative.scoring.llm`` — LLM SDK side-channel, same.
* ``puremacro.tests`` — break tests / unit-root tests, distinct from
  the project ``tests/`` directory.
"""
from __future__ import annotations

import importlib
import json
import pkgutil
import re
import subprocess
import sys

import pytest

import puremacro


# Deps that no browser-clean (non-skip-listed) shippable module may import: the
# dev-only parity stack (statsmodels/linearmodels/arch) and the scraper/PDF stack
# (bs4/pdfplumber/pypdf, used only by the skip-listed narrative.sources
# connectors). `requests`/`numba` are deliberately NOT here — fetch/* and the
# numba VFI kernels import them at module level by design; the narrative-path
# requests-cleanliness is covered by tests/test_pyodide/test_narrative_importable.
_FORBIDDEN = ("statsmodels", "linearmodels", "arch", "bs4", "pdfplumber", "pypdf")
# The `io` extra's file-format engines. Not forbidden (pandas itself imports
# pyarrow at import time wherever it is installed, so the in-process sweep could
# not tell a puremacro import from a pandas one), but optional: every shippable
# module must import without them, so the subprocess sweep blocks them too.
_IO_ENGINES = ("pyarrow", "openpyxl")
_ABSENT_IN_BROWSER = _FORBIDDEN + _IO_ENGINES
_SKIP_PREFIXES = (
    "puremacro.examples",
    "puremacro.teaching",
    "puremacro.narrative.sources",
    "puremacro.narrative.scoring.llm",
    "puremacro.tests.",  # in-package "tests/" submodule (breaks, unit root)
    # numba backend kernels import numba at module level by design; the numpy
    # oracle modules are the Pyodide path (see vfi/_backend selection).
    "puremacro.vfi.kernels_numba",
    "puremacro.models.nested_dmp.kernels_numba",
)


def _shippable_modules() -> list[str]:
    out = []
    for info in pkgutil.walk_packages(puremacro.__path__, prefix="puremacro."):
        name = info.name
        if any(name == p or name.startswith(p + ".") or name.startswith(p)
               for p in _SKIP_PREFIXES):
            continue
        out.append(name)
    return out


@pytest.fixture
def pristine_sys_modules():
    """Snapshot ``sys.modules`` so any leak introduced by the import
    sweep is attributable to puremacro and not to the test harness."""
    before = set(sys.modules)
    yield before
    # Teardown: the in-process sweep imports same-named submodules (e.g.
    # ``puremacro.inference.newey_west``), which rebinds the package attribute
    # from the re-exported *function* to the *submodule* and would otherwise
    # leak into later tests (``from puremacro.inference import newey_west`` then
    # yields the module — see tests/test_inference). Reload the affected package
    # so its ``__init__`` re-exports are restored.
    mod = sys.modules.get("puremacro.inference")
    if mod is not None:
        try:
            importlib.reload(mod)
        except Exception:
            pass


def test_no_forbidden_runtime_imports_after_full_sweep(pristine_sys_modules):
    """Walk the full puremacro tree and confirm no forbidden module
    snuck into sys.modules after every shippable submodule is imported.

    Dev-only deps (statsmodels / linearmodels / arch) live behind
    parity tests under ``tests/`` only; they must never ride along with
    a vanilla ``import puremacro.*`` chain.
    """
    # Pre-condition sanity check: the parent test process may already
    # have imported a forbidden module (from a parity test earlier in
    # the same session). That's OK — we measure deltas.
    forbidden_before = {
        m for m in sys.modules
        if any(m == f or m.startswith(f + ".") for f in _FORBIDDEN)
    }

    imported = []
    failures = []
    for name in _shippable_modules():
        try:
            importlib.import_module(name)
            imported.append(name)
        except Exception as exc:  # pragma: no cover — surface for debugging
            failures.append((name, type(exc).__name__, str(exc)[:200]))

    assert not failures, (
        "Some shippable submodules failed to import:\n  "
        + "\n  ".join(f"{n}: {t}: {msg}" for n, t, msg in failures)
    )

    forbidden_after = {
        m for m in sys.modules
        if any(m == f or m.startswith(f + ".") for f in _FORBIDDEN)
    }
    leaked = sorted(forbidden_after - forbidden_before)
    assert not leaked, (
        f"Forbidden runtime dependency leaked through plain imports of "
        f"shippable puremacro modules: {leaked}. Move the import to a "
        f"lazy-loaded backend (see narrative/scoring/llm.py for the "
        f"pattern) or guard it with a try/except + _HAS_* flag."
    )

    assert len(imported) > 50, (
        f"Sanity check failed: only imported {len(imported)} modules; "
        "the walk likely missed something."
    )


# Subprocess body for the "deps absent" sweep: block the forbidden deps via a
# meta_path finder (mirroring Pyodide, where they are simply not installed), then
# import every module in the handed-in list and report any that fail.
_ABSENT_SWEEP = """
import sys, json, importlib
_BLOCKED = set(json.loads(sys.argv[1]))
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BLOCKED:
            raise ModuleNotFoundError("absent (simulated Pyodide): " + name)
        return None
sys.meta_path.insert(0, _Blocker())
fails = []
for m in json.loads(sys.argv[2]):
    try:
        importlib.import_module(m)
    except Exception as e:
        fails.append([m, type(e).__name__ + ": " + str(e)[:160]])
print(json.dumps(fails))
"""


def test_shippable_modules_import_with_forbidden_deps_absent():
    """The strong Pyodide guarantee: every shippable (non-skip-listed) module
    imports even when the forbidden deps are ABSENT — exactly the browser, where
    statsmodels/linearmodels/arch/bs4/pdfplumber/pypdf are simply not installed —
    and when the optional `io` engines (pyarrow/openpyxl) are absent too.

    Complements the in-process membership sweep above, which only proves the deps
    don't enter ``sys.modules`` on a host where they ARE installed; here we prove
    the modules import with them genuinely gone (catching a broken lazy guard).
    The module list is computed on the host (where the scraper deps are present so
    the package walk succeeds) and handed to a fresh subprocess that blocks them.
    """
    mods = _shippable_modules()
    r = subprocess.run(
        [sys.executable, "-c", _ABSENT_SWEEP, json.dumps(list(_ABSENT_IN_BROWSER)),
         json.dumps(mods)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert r.returncode == 0, f"sweep subprocess crashed:\n{r.stderr}"
    fails = json.loads(r.stdout.strip().splitlines()[-1])
    assert not fails, (
        "Shippable modules that fail to import when the forbidden deps are "
        "absent (move the import to a lazy/function-level guard, or add the "
        "module to _SKIP_PREFIXES if it is a documented out-of-Pyodide "
        "side-channel):\n  " + "\n  ".join(f"{m}: {e}" for m, e in fails)
    )


# The four packages a shippable estimator module may import at top level. This
# is the *import* contract the two sweeps above enforce; it is deliberately
# narrower than the set of declared runtime dependencies (see below).
_PYODIDE_IMPORT_CORE = frozenset({"numpy", "scipy", "pandas", "matplotlib"})


def _dep_name(spec: str) -> str:
    """'numpy   >= 1.26' / 'numpy>=1.26' -> 'numpy'."""
    return re.split(r"[<>=!~;\[\s]", spec.strip(), maxsplit=1)[0].strip()


def _documented_runtime_deps() -> list[str]:
    """Parse the fenced dependency block under ARCHITECTURE.md's
    '### Allowed runtime dependencies' heading."""
    from pathlib import Path

    arch = Path(__file__).resolve().parent.parent / "ARCHITECTURE.md"
    text = arch.read_text(encoding="utf-8")
    m = re.search(
        r"^### Allowed runtime dependencies\s*$(.*?)^```\s*$(.*?)^```\s*$",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m, (
        "ARCHITECTURE.md no longer has a fenced code block under "
        "'### Allowed runtime dependencies'. That block is the single source of "
        "truth this test compares pyproject.toml against — restore it (or update "
        "this parser deliberately), do not delete the contract."
    )
    return [_dep_name(ln) for ln in m.group(2).splitlines() if ln.strip()]


def test_pyproject_runtime_deps_match_documentation():
    """`pyproject.toml [project.dependencies]` must match, exactly, the list
    documented in ARCHITECTURE.md -> 'Allowed runtime dependencies'.

    The package advertises a runtime contract; the contract is only worth
    something if the metadata users actually install agrees with the prose they
    actually read. Rather than hard-coding the expected set here (which silently
    goes stale the moment the docs change), we *parse* the documented block, so
    widening the contract requires editing the documentation — the point of the
    guard.

    Two separate invariants:

    1. The declared set equals the documented set (no silent drift either way).
    2. The Pyodide import core (numpy/scipy/pandas/matplotlib) is still declared.
       Extra declared deps beyond it are allowed if documented — `requests`
       (imported at module level by ``fetch/*`` by design) is the one case. What
       it may NOT do is enter ``sys.modules`` through an estimator import chain;
       that is what the two sweeps above enforce, and it is the invariant that
       actually protects the browser target. Whether a declared dep can be
       *installed* in the browser is the next test's job.
    """
    import tomllib
    from pathlib import Path

    here = Path(__file__).resolve()
    pyproject = here.parent.parent / "pyproject.toml"
    with pyproject.open("rb") as fh:
        cfg = tomllib.load(fh)
    declared = {_dep_name(d) for d in cfg["project"]["dependencies"]}
    documented = set(_documented_runtime_deps())

    assert declared == documented, (
        f"Runtime dependency set drifted from the documented contract.\n"
        f"  pyproject.toml declares : {sorted(declared)}\n"
        f"  ARCHITECTURE.md documents: {sorted(documented)}\n"
        f"  only in pyproject       : {sorted(declared - documented)}\n"
        f"  only in ARCHITECTURE.md : {sorted(documented - declared)}\n"
        "Update ARCHITECTURE.md ('Allowed runtime dependencies'), README.md "
        "('Pyodide compatibility') and pyproject.toml in the SAME commit."
    )

    missing_core = _PYODIDE_IMPORT_CORE - declared
    assert not missing_core, (
        f"The Pyodide import core is no longer fully declared: {sorted(missing_core)} "
        "missing from [project.dependencies]. These four are not optional."
    )


# Packages in the Pyodide distribution itself (pyodide-lock.json) that puremacro
# may declare as base dependencies. The JupyterLite playground installs with
# `%pip install puremacro` and PyPI fallback disabled
# (playground/content_static/jupyter-lite.json), so every base dependency has to
# resolve from the distribution: one that Pyodide lacks makes the first cell of
# every notebook fail, as openpyxl and pyarrow did through 3.2.1. All five have
# been in the distribution since before 0.28; add a package here only after
# checking the pyodide-lock.json of the kernel pinned in
# .github/workflows/pages.yml -- and remember older kernels (juno.sh) too.
_SHIPS_WITH_PYODIDE = frozenset({"numpy", "scipy", "pandas", "matplotlib", "requests"})


def test_runtime_deps_ship_with_pyodide():
    """Every `[project.dependencies]` entry must be installable in the browser
    without PyPI, or the playground's `%pip install puremacro` cannot resolve."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as fh:
        declared = {_dep_name(d) for d in tomllib.load(fh)["project"]["dependencies"]}
    not_in_pyodide = sorted(declared - _SHIPS_WITH_PYODIDE)
    assert not not_in_pyodide, (
        f"Base dependencies that the Pyodide distribution does not ship: {not_in_pyodide}. "
        "The playground installs with PyPI fallback disabled, so these would break "
        "`%pip install puremacro` in every notebook. Put them in an extra (see `io` in "
        "pyproject.toml) and make the code that needs them fail loudly without them."
    )


@pytest.mark.mechanism_control
def test_the_absent_dep_blocker_actually_blocks():
    """Positive control for `_ABSENT_SWEEP`.

    The sweep proves every shippable module imports with the forbidden deps
    absent — but only if they really are absent. An inert blocker turns the
    strongest Pyodide guarantee in this file into a test that imports modules on
    a host where the deps are installed and calls that a pass. It is currently
    live; this keeps it that way.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _ABSENT_SWEEP,
         json.dumps(list(_ABSENT_IN_BROWSER)), json.dumps(list(_ABSENT_IN_BROWSER))],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    blocked = {name for name, _ in json.loads(proc.stdout)}
    assert blocked == set(_ABSENT_IN_BROWSER), (
        "the meta_path blocker did not stop every forbidden dep — the absent-deps "
        f"sweep is not testing what it claims. Blocked: {sorted(blocked)}")
