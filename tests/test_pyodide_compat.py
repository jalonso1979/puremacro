"""Pyodide-compatibility regression test.

`puremacro`'s pitch is "pure-numpy + scipy + pandas + matplotlib, runs
on the iPad via Pyodide / juno.sh." That promise is **load-bearing** —
if a contributor adds a top-level ``import statsmodels`` (or
``linearmodels`` / ``arch``) the package silently stops working on the
intended deployment target.

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
    statsmodels/linearmodels/arch/bs4/pdfplumber/pypdf are simply not installed.

    Complements the in-process membership sweep above, which only proves the deps
    don't enter ``sys.modules`` on a host where they ARE installed; here we prove
    the modules import with them genuinely gone (catching a broken lazy guard).
    The module list is computed on the host (where the scraper deps are present so
    the package walk succeeds) and handed to a fresh subprocess that blocks them.
    """
    mods = _shippable_modules()
    r = subprocess.run(
        [sys.executable, "-c", _ABSENT_SWEEP, json.dumps(list(_FORBIDDEN)),
         json.dumps(mods)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"sweep subprocess crashed:\n{r.stderr}"
    fails = json.loads(r.stdout.strip().splitlines()[-1])
    assert not fails, (
        "Shippable modules that fail to import when the forbidden deps are "
        "absent (move the import to a lazy/function-level guard, or add the "
        "module to _SKIP_PREFIXES if it is a documented out-of-Pyodide "
        "side-channel):\n  " + "\n  ".join(f"{m}: {e}" for m, e in fails)
    )


def test_pyproject_runtime_deps_match_documentation():
    """The ARCHITECTURE.md Pyodide section names exactly four runtime
    deps: numpy, scipy, pandas, matplotlib. The pyproject must agree."""
    import tomllib
    from pathlib import Path

    here = Path(__file__).resolve()
    pyproject = here.parent.parent / "pyproject.toml"
    with pyproject.open("rb") as fh:
        cfg = tomllib.load(fh)
    deps = cfg["project"]["dependencies"]
    names = {d.split(">=")[0].split("==")[0].split("<")[0].strip() for d in deps}
    assert names == {"numpy", "scipy", "pandas", "matplotlib"}, (
        f"Runtime dependency set drifted from documented Pyodide "
        f"contract. Got {names}; expected {{numpy, scipy, pandas, "
        f"matplotlib}}. Update ARCHITECTURE.md *and* pyproject.toml "
        "together if this is intentional."
    )
