"""Regression tests for the 2026-09-22 documentation and test-hygiene audit (cluster DOCS).

Covered findings (see the audit's CONFIRMED_FINDINGS for the evidence):

- ``puremacro.lp.garch_utils``: the FutureWarning named 4.0.0 as its removal release after
  4.0.0 had shipped keeping the shim (the same drift was fixed once before at "2.0.0").
  The warning must name a major release that is still in the future.
- README documentation lists omitted the 4.3.0 trade guides and the validation status page,
  described the notebook catalogue as "00-58" although it runs to 65, and the curated
  showcase bullets stopped at notebook 58.
- The four 4.3.0 pages had no Spanish mirrors and no language switchers, so the bilingual
  parity test passed vacuously. The mirrors must keep their code blocks identical to the
  English pages (the examples are runnable contracts, not prose). The eight pages added
  after 4.3.0 (``NEW_MODULE_DOCS``) are held to the same contract.
- ``test_target_15_png_opacity_and_contrast`` hard-coded 36 embedded figures; the count is
  now derived from a per-notebook table that must stay in step with the target list.

Pure file checks plus one cheap import of the shim; no numerical work. Other test
modules are read as source (``ast``), never imported.
"""
from __future__ import annotations

import ast
import importlib
import re
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

NEW_TRADE_DOCS = [
    "trade_accounting.md",
    "trade_welfare.md",
    "trade_policy.md",
    "STRUCTURAL_VALIDATION_STATUS.md",
]
# Pages added after 4.3.0 (MRIO engines and stacked Newton-Krylov): same bilingual contract.
NEW_MODULE_DOCS = [
    "trade_mrio.md",
    "trade_condensed.md",
    "trade_ces_newton.md",
    "trade_household.md",
    "trade_continuation.md",
    "trade_stability.md",
    "trade_dynamic.md",
    "dsge_stacked_newton.md",
]
BILINGUAL_PAGES = NEW_TRADE_DOCS + NEW_MODULE_DOCS

_FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_RELEASE_IN_PARENS = re.compile(r"\((\d+)\.(\d+)\.(\d+)\)")


def _module_assignment(path: Path, name: str) -> ast.expr:
    """The value expression of the module-level assignment ``name = ...`` in ``path``.

    Other test modules are read as source, never imported: ``tests/`` is not a package, so
    ``from tests import ...`` fails under CI's console-script ``pytest`` (no repo root on
    ``sys.path``) and resolves to ``puremacro/tests`` once a collected module has put
    ``puremacro/`` on ``sys.path``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.value  # type: ignore[return-value]
    raise AssertionError(f"{path.relative_to(ROOT)} no longer assigns {name} at module level")


def _module_constant(path: Path, name: str):
    """Literal value of the module-level constant ``name`` in ``path`` (see ``_module_assignment``)."""
    return ast.literal_eval(_module_assignment(path, name))


def _version_tuple(text: str) -> tuple[int, ...]:
    """Leading numeric release components of a version string ('4.3.0', '4.4.0rc1' -> (4, 4, 0))."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", text)
    assert m, f"unparseable version {text!r}"
    return tuple(int(x) for x in m.groups())


# ---------------------------------------------------------------------------
# garch_utils shim: the named removal release must still be in the future
# ---------------------------------------------------------------------------

def test_garch_utils_deprecation_names_a_future_major_release():
    import puremacro

    sys.modules.pop("puremacro.lp.garch_utils", None)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        importlib.import_module("puremacro.lp.garch_utils")
    messages = [
        str(w.message)
        for w in record
        if issubclass(w.category, FutureWarning) and "puremacro.lp.garch_utils" in str(w.message)
    ]
    assert messages, "importing puremacro.lp.garch_utils must emit its FutureWarning"
    m = _RELEASE_IN_PARENS.search(messages[0])
    assert m, f"the warning must name the removal release in parentheses: {messages[0]!r}"
    target = tuple(int(x) for x in m.groups())
    current = _version_tuple(puremacro.__version__)
    assert target > current, (
        f"garch_utils FutureWarning names removal in {target} but puremacro is already "
        f"{current}; retarget the shim (this drifted at 2.0.0 and again at 4.0.0)"
    )
    assert target[1:] == (0, 0), f"removal must be scheduled for a major release, got {target}"


def test_sigma_numpy_deprecation_names_a_future_major_release():
    """The sibling shim ``puremacro.sigma.sigma_numpy`` drifted the same way (it named 4.0.0)."""
    import puremacro
    from puremacro.sigma.sigma_numpy import SigmaObject

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        SigmaObject(sigma=[0.1, 0.2], R=[[1.0, 0.0], [0.0, 1.0]], labels=["A", "B"])
    messages = [str(w.message) for w in record
                if issubclass(w.category, FutureWarning) and "SigmaObject is deprecated" in str(w.message)]
    assert messages, "SigmaObject must emit its FutureWarning"
    m = _RELEASE_IN_PARENS.search(messages[0])
    assert m, f"the warning must name the removal release in parentheses: {messages[0]!r}"
    target = tuple(int(x) for x in m.groups())
    assert target > _version_tuple(puremacro.__version__) and target[1:] == (0, 0), target


# ---------------------------------------------------------------------------
# README documentation lists
# ---------------------------------------------------------------------------

def _highest_showcase_notebook() -> int:
    numbers = [
        int(p.name[:2])
        for p in (ROOT / "notebooks").glob("[0-9][0-9]_*.py")
        if not p.stem.endswith("_es")
    ]
    assert numbers, "no showcase notebook sources found"
    return max(numbers)


@pytest.mark.parametrize(
    "readme,docs_prefix",
    [("README.md", "docs/"), ("README.es.md", "docs/es/")],
)
def test_readme_documentation_list_names_the_4_3_0_trade_docs(readme: str, docs_prefix: str):
    txt = (ROOT / readme).read_text(encoding="utf-8")
    missing = [name for name in BILINGUAL_PAGES if f"`{docs_prefix}{name}`" not in txt]
    assert not missing, f"{readme} documentation list omits {missing}"


@pytest.mark.parametrize("readme", ["README.md", "README.es.md"])
def test_readme_notebook_range_matches_the_catalogue(readme: str):
    txt = (ROOT / readme).read_text(encoding="utf-8")
    highest = _highest_showcase_notebook()
    stale = re.findall(r"00[–-](\d\d)\)", txt)
    assert stale, f"{readme} no longer states the notebook range '(notebooks 00–NN)'"
    assert all(int(n) == highest for n in stale), (
        f"{readme} states notebook range(s) {sorted(set(stale))} but the catalogue runs to {highest:02d}"
    )


@pytest.mark.parametrize("readme", ["README.md", "README.es.md"])
def test_readme_showcase_list_reaches_the_newest_notebooks(readme: str):
    """The curated showcase bullets stopped at (`58`) while notebooks 59-65 shipped."""
    txt = (ROOT / readme).read_text(encoding="utf-8")
    highest = _highest_showcase_notebook()
    missing = [n for n in range(59, highest + 1) if f"(`{n:02d}`)" not in txt]
    assert not missing, f"{readme} showcase list omits notebooks {missing}"


# ---------------------------------------------------------------------------
# Spanish mirrors of the 4.3.0 pages: switchers both ways, identical code blocks,
# same section and table structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", BILINGUAL_PAGES)
def test_spanish_mirror_is_structurally_faithful(name: str):
    en_path = ROOT / "docs" / name
    es_path = ROOT / "docs" / "es" / name
    assert es_path.is_file(), f"docs/es/{name} is missing"
    en = en_path.read_text(encoding="utf-8")
    es = es_path.read_text(encoding="utf-8")

    assert en.startswith(f"> 🇬🇧 English · 🇪🇸 [Español](es/{name})\n"), f"docs/{name} lacks the switcher"
    assert es.startswith(f"> 🇬🇧 [English](../{name}) · 🇪🇸 Español\n"), f"docs/es/{name} lacks the switcher"

    en_blocks = _FENCED_BLOCK.findall(en)
    es_blocks = _FENCED_BLOCK.findall(es)
    assert en_blocks == es_blocks, (
        f"code blocks in docs/es/{name} differ from docs/{name}; the examples are contracts, "
        "resync the Spanish mirror verbatim"
    )

    en_headings = [ln for ln in en.splitlines() if ln.startswith("## ")]
    es_headings = [ln for ln in es.splitlines() if ln.startswith("## ")]
    assert len(en_headings) == len(es_headings), (
        f"docs/es/{name} has {len(es_headings)} sections, docs/{name} has {len(en_headings)}"
    )

    en_rows = [ln for ln in en.splitlines() if ln.startswith("| ")]
    es_rows = [ln for ln in es.splitlines() if ln.startswith("| ")]
    assert len(en_rows) == len(es_rows), (
        f"docs/es/{name} has {len(es_rows)} table rows, docs/{name} has {len(en_rows)}"
    )

    # Cross-links between these pages must stay inside docs/es/ (relative names resolve
    # to the Spanish twins), never point the Spanish reader back to the English page.
    for other in BILINGUAL_PAGES:
        assert f"](../{other})" not in es.split("\n", 1)[1], (
            f"docs/es/{name} links to the English {other}; link the Spanish twin instead"
        )


def test_new_trade_docs_are_in_the_bilingual_allowlist():
    user_docs = _module_constant(ROOT / "tests" / "test_bilingual_docs.py", "_USER_DOCS")
    for name in BILINGUAL_PAGES:
        assert name in user_docs, f"{name} missing from tests/test_bilingual_docs.py::_USER_DOCS"


# ---------------------------------------------------------------------------
# Figure-count table of the m4 challenger test
# ---------------------------------------------------------------------------

def test_m4_figure_table_matches_target_list():
    m4 = ROOT / "tests" / "test_notebooks" / "test_m4_challenger_schema_and_outputs.py"
    table = _module_constant(m4, "EXPECTED_FIGURES_PER_TARGET")
    targets = _module_constant(m4, "TARGET_15_NOTEBOOKS")

    assert set(table) == set(targets)
    # The total is derived from the table, never hard-coded again.
    total = _module_assignment(m4, "EXPECTED_TOTAL_FIGURES")
    assert ast.unparse(total) == "sum(EXPECTED_FIGURES_PER_TARGET.values())", ast.unparse(total)
    assert table["34_penalized_macro_forecasting.ipynb"] == 3
    assert sum(table.values()) == 37
