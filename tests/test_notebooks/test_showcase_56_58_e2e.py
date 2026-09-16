"""End-to-End verification suite for Showcase Notebooks 56, 57, and 58.

Verifies:
- Tier 1: File presence of all 6 files (56, 56_es, 57, 57_es, 58, 58_es), compiled .ipynb
  artifacts, and all 7 required template sections in each file in monotonic order.
- Tier 2: AST analysis verifying Pyodide compliance (zero forbidden imports: statsmodels,
  linearmodels, arch, torch, jax, requests, etc.), presence of _nbstyle.apply_style(),
  deterministic random seeds (rng = np.random.default_rng(42)), interactive knob downstream
  assertions, and zero emojis.
- Tier 3: Bilingual parity and 100% code identity (normalize_code(en) == normalize_code(es)
  for all 3 pairs), translated Spanish headings, and catalog registrations in notebooks/README.md,
  docs/notebooks.md, and docs/es/notebooks.md.
- Tier 4: Execution test (dry-run import / execution via build_notebooks.py --check and
  verification of economic domain properties).
- Tier 5: The shipped .ipynb artifacts hold no tracebacks and no warnings, were built
  from the current jupytext source, and print the numbers the current code produces --
  several replicated printed numbers per notebook, each one a figure the prose quotes
  (56: HJB iterations, Ks, the wealth Gini, r*/w*/K*/L* and the sampled Ks-Kd crossing;
  57: the DML and OLS estimates, the OccBin iteration count, regime path, linear impact
  rate and output trough; 58: the panel size, beta_p, the verdict, the noise share and
  the simulated policy-rate floor) -- so a stale rebuild cannot pass unnoticed. It also
  checks that notebook 58 discloses, in both languages and in code as well as prose,
  that its Latin American panel is simulated.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

PROJ = Path(__file__).resolve().parents[2]
NB_DIR = PROJ / "notebooks"
DOCS_DIR = PROJ / "docs"

# All 3 showcase notebook stems
NOTEBOOK_STEMS = {
    "56": "56_implicit_hjb_and_continuous_kfe",
    "57": "57_multiconstraint_occbin_and_dml",
    "58": "58_latin_america_realtime_macro",
}

NB_PATHS_EN = {stem: NB_DIR / f"{name}.py" for stem, name in NOTEBOOK_STEMS.items()}
NB_PATHS_ES = {stem: NB_DIR / f"{name}_es.py" for stem, name in NOTEBOOK_STEMS.items()}
IPYNB_PATHS_EN = {stem: NB_DIR / f"{name}.ipynb" for stem, name in NOTEBOOK_STEMS.items()}
IPYNB_PATHS_ES = {stem: NB_DIR / f"{name}_es.ipynb" for stem, name in NOTEBOOK_STEMS.items()}

ALL_NOTEBOOK_PATHS = [
    NB_PATHS_EN["56"], NB_PATHS_ES["56"],
    NB_PATHS_EN["57"], NB_PATHS_ES["57"],
    NB_PATHS_EN["58"], NB_PATHS_ES["58"],
]

ALL_IPYNB_PATHS = [
    IPYNB_PATHS_EN["56"], IPYNB_PATHS_ES["56"],
    IPYNB_PATHS_EN["57"], IPYNB_PATHS_ES["57"],
    IPYNB_PATHS_EN["58"], IPYNB_PATHS_ES["58"],
]

# Pyodide allowed external dependencies + puremacro + stdlib
PYODIDE_ALLOWED = frozenset({
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "puremacro",
    "_nbstyle",
    # Standard library modules
    "sys", "time", "pathlib", "warnings", "dataclasses",
    "typing", "math", "copy", "itertools", "functools",
    "collections", "re", "json", "abc", "random", "enum",
    "importlib", "tempfile", "hashlib", "io",
})

FORBIDDEN_PACKAGES = frozenset({
    "statsmodels",
    "linearmodels",
    "arch",
    "bs4",
    "pdfplumber",
    "pypdf",
    "ipywidgets",
    "requests",
    "urllib3",
    "httpx",
    "torch",
    "tensorflow",
    "jax",
    "keras",
    "sympy",
    "sklearn",
})

REQUIRED_SECTIONS_EN = [
    ("motivating_question", r"(?i)(motivating\s+question|\*\*¿?how\b|\*\*¿?what\b|\*\*¿?can\b|\*\*¿?why\b|\*\*¿?c[oó]mo\b|\*\*¿?qu[eé]\b|\*\*¿?pued\b)"),
    ("method_in_math", r"(?i)(the\s+method\s+in\s+math)"),
    ("intuition", r"(?i)(\*\*intuition\.\*\*|##\s*intuition\b)"),
    ("read_the_output", r"(?i)(read\s+the\s+output)"),
    ("your_turn", r"(?i)(your\s+turn)"),
    ("how_comprehensive", r"(?i)(how\s+comprehensive\s+is\s+this)"),
]

REQUIRED_SECTIONS_ES = [
    ("motivating_question", r"(?i)(pregunta\s+motivadora|\*\*¿?c[oó]mo\b|\*\*¿?qu[eé]\b|\*\*¿?por\s+qu[eé]\b|\*\*¿?pued\b|motivating\s+question)"),
    ("method_in_math", r"(?i)(el\s+m[eé]todo\s+en\s+matem[aá]ticas|the\s+method\s+in\s+math)"),
    ("intuition", r"(?i)(\*\*intuici[oó]n\.\*\*|##\s*intuici[oó]n\b|\*\*intuition\.\*\*|##\s*intuition\b)"),
    ("read_the_output", r"(?i)(lectura\s+de\s+los\s+resultados|read\s+the\s+output)"),
    ("your_turn", r"(?i)(tu\s+turno|your\s+turn)"),
    ("how_comprehensive", r"(?i)(qu[eé]\s+tan\s+exhaustivo|how\s+comprehensive\s+is\s+this)"),
]


# ============================================================================
# Helper Functions
# ============================================================================

def parse_jupytext_cells(content: str) -> list[tuple[str, str]]:
    """Parse a Jupytext percent format file into a list of (cell_type, cell_text)."""
    cell_pattern = re.compile(r"^# %%(\s+\[markdown\])?", re.MULTILINE)
    splits = cell_pattern.split(content)
    cells: list[tuple[str, str]] = []
    i = 1
    while i < len(splits):
        tag = splits[i]
        body = splits[i + 1] if i + 1 < len(splits) else ""
        cell_type = "markdown" if tag and "[markdown]" in tag else "code"
        cells.append((cell_type, body.strip()))
        i += 2
    return cells


def extract_code_cells(content: str) -> list[str]:
    """Extract code cell contents from a Jupytext percent format file."""
    cells = parse_jupytext_cells(content)
    return [body for c_type, body in cells if c_type == "code"]


def extract_markdown_cells(content: str) -> list[str]:
    """Extract markdown cell contents from a Jupytext percent format file."""
    cells = parse_jupytext_cells(content)
    return [body for c_type, body in cells if c_type == "markdown"]


def extract_imports(code_text: str) -> set[str]:
    """Extract top-level module names imported in python code using ast."""
    root = ast.parse(code_text)
    modules: set[str] = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


def normalize_code(code: str) -> str:
    """Normalize python code for structural comparison (strip comments and blank lines)."""
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return "\n".join(lines)


def _require_file(path: Path) -> str:
    """Safe file reader asserting existence."""
    assert path.exists(), f"Required file {path.name} not found on disk"
    return path.read_text(encoding="utf-8")


# ============================================================================
# Test Harness Unit Tests
# ============================================================================

class TestHarnessUnit:
    """Unit tests verifying helper functions and parsing framework."""

    def test_jupytext_parser_unit(self):
        sample = (
            "# ---\n# jupyter:\n# ---\n\n"
            "# %% [markdown]\n# # Section 1\n\n"
            "# %%\nimport numpy as np\nx = 1\n"
        )
        cells = parse_jupytext_cells(sample)
        assert len(cells) == 2
        assert cells[0][0] == "markdown"
        assert cells[1][0] == "code"

    def test_normalize_code_unit(self):
        c_en = "# English comment\nx = 10\n\n# another comment\ny = 20\n"
        c_es = "# Spanish comment\nx = 10\n\n# otro comentario\ny = 20\n"
        assert normalize_code(c_en) == normalize_code(c_es)
        assert normalize_code(c_en) == "x = 10\ny = 20"

    def test_import_checker_unit(self):
        code_ok = "import numpy as np\nfrom puremacro.vfi import solve_hjb_achdou\n"
        code_bad = "import statsmodels.api as sm\nimport torch\n"
        assert extract_imports(code_ok).issubset(PYODIDE_ALLOWED)
        assert "statsmodels" in extract_imports(code_bad)
        assert "torch" in extract_imports(code_bad)


# ============================================================================
# Tier 1: Feature Coverage (Isolated Happy Paths >= 5 per Feature)
# ============================================================================

class TestTier1Showcase56Coverage:
    """Tier 1 Feature Coverage: Showcase Notebook 56 (Continuous HJB and KFE)."""

    def test_nb56_en_file_presence(self):
        p = NB_PATHS_EN["56"]
        _require_file(p)
        assert p.is_file(), f"File {p.name} missing"

    def test_nb56_es_twin_file_presence(self):
        p = NB_PATHS_ES["56"]
        _require_file(p)
        assert p.is_file(), f"Spanish twin file {p.name} missing"

    def test_nb56_ipynb_artifacts_presence(self):
        for p in [IPYNB_PATHS_EN["56"], IPYNB_PATHS_ES["56"]]:
            assert p.exists() and p.stat().st_size > 1000, f"Compiled artifact {p.name} missing or empty"

    def test_nb56_jupytext_percent_header(self):
        content = _require_file(NB_PATHS_EN["56"])
        assert "format_name: percent" in content
        assert "kernelspec:" in content

    def test_nb56_all_seven_template_sections_present(self):
        content = _require_file(NB_PATHS_EN["56"])
        for name, pattern in REQUIRED_SECTIONS_EN:
            assert re.search(pattern, content), f"Section '{name}' missing from Notebook 56"
        assert len(extract_code_cells(content)) >= 4

    def test_nb56_worked_code_solver_apis(self):
        content = _require_file(NB_PATHS_EN["56"])
        code_text = "\n".join(extract_code_cells(content))
        assert "solve_hjb_achdou" in code_text
        assert "solve_aiyagari_continuous_hjb" in code_text

    def test_nb56_interactive_knobs_and_asserts(self):
        content = _require_file(NB_PATHS_EN["56"])
        cells = parse_jupytext_cells(content)
        found_knob = any(c_type == "code" and "# ← change this" in body for c_type, body in cells)
        knob_has_assert = any(c_type == "code" and "# ← change this" in body and "assert " in body for c_type, body in cells)
        assert found_knob, "Missing '# ← change this' knob in Notebook 56"
        assert knob_has_assert, "Interactive knob cell must contain downstream assertions"

    def test_nb56_math_latex_formulation(self):
        content = _require_file(NB_PATHS_EN["56"])
        md_text = "\n".join(extract_markdown_cells(content))
        assert "$" in md_text
        assert ("\\rho v" in md_text or "HJB" in md_text or "Achdou" in md_text)
        assert ("KFE" in md_text or "A^\\top" in md_text or "density" in md_text.lower() or "Kolmogorov" in md_text)


class TestTier1Showcase57Coverage:
    """Tier 1 Feature Coverage: Showcase Notebook 57 (Multi-Constraint OccBin & DML)."""

    def test_nb57_en_file_presence(self):
        p = NB_PATHS_EN["57"]
        _require_file(p)
        assert p.is_file(), f"File {p.name} missing"

    def test_nb57_es_twin_file_presence(self):
        p = NB_PATHS_ES["57"]
        _require_file(p)
        assert p.is_file(), f"Spanish twin file {p.name} missing"

    def test_nb57_ipynb_artifacts_presence(self):
        for p in [IPYNB_PATHS_EN["57"], IPYNB_PATHS_ES["57"]]:
            assert p.exists() and p.stat().st_size > 1000, f"Compiled artifact {p.name} missing or empty"

    def test_nb57_jupytext_percent_header(self):
        content = _require_file(NB_PATHS_EN["57"])
        assert "format_name: percent" in content
        assert "kernelspec:" in content

    def test_nb57_all_seven_template_sections_present(self):
        content = _require_file(NB_PATHS_EN["57"])
        for name, pattern in REQUIRED_SECTIONS_EN:
            assert re.search(pattern, content), f"Section '{name}' missing from Notebook 57"
        assert len(extract_code_cells(content)) >= 4

    def test_nb57_worked_code_solver_apis(self):
        content = _require_file(NB_PATHS_EN["57"])
        code_text = "\n".join(extract_code_cells(content))
        assert "solve_multiconstraint_occbin" in code_text or "OccBinMultiConstraint" in code_text
        assert "DoubleMLPLR" in code_text

    def test_nb57_interactive_knobs_and_asserts(self):
        content = _require_file(NB_PATHS_EN["57"])
        cells = parse_jupytext_cells(content)
        found_knob = any(c_type == "code" and "# ← change this" in body for c_type, body in cells)
        knob_has_assert = any(c_type == "code" and "# ← change this" in body and "assert " in body for c_type, body in cells)
        assert found_knob, "Missing '# ← change this' knob in Notebook 57"
        assert knob_has_assert, "Interactive knob cell must contain downstream assertions"

    def test_nb57_math_latex_formulation(self):
        content = _require_file(NB_PATHS_EN["57"])
        md_text = "\n".join(extract_markdown_cells(content))
        assert "$" in md_text
        assert ("OccBin" in md_text or "Guerrieri" in md_text or "regime" in md_text.lower())
        assert ("DML" in md_text or "Chernozhukov" in md_text or "Neyman" in md_text or "Double" in md_text)


class TestTier1Showcase58Coverage:
    """Tier 1 Feature Coverage: Showcase Notebook 58 (Latin America Real-Time Macro)."""

    def test_nb58_en_file_presence(self):
        p = NB_PATHS_EN["58"]
        _require_file(p)
        assert p.is_file(), f"File {p.name} missing"

    def test_nb58_es_twin_file_presence(self):
        p = NB_PATHS_ES["58"]
        _require_file(p)
        assert p.is_file(), f"Spanish twin file {p.name} missing"

    def test_nb58_ipynb_artifacts_presence(self):
        for p in [IPYNB_PATHS_EN["58"], IPYNB_PATHS_ES["58"]]:
            assert p.exists() and p.stat().st_size > 1000, f"Compiled artifact {p.name} missing or empty"

    def test_nb58_jupytext_percent_header(self):
        content = _require_file(NB_PATHS_EN["58"])
        assert "format_name: percent" in content
        assert "kernelspec:" in content

    def test_nb58_all_seven_template_sections_present(self):
        content = _require_file(NB_PATHS_EN["58"])
        for name, pattern in REQUIRED_SECTIONS_EN:
            assert re.search(pattern, content), f"Section '{name}' missing from Notebook 58"
        assert len(extract_code_cells(content)) >= 4

    def test_nb58_worked_code_solver_apis(self):
        content = _require_file(NB_PATHS_EN["58"])
        code_text = "\n".join(extract_code_cells(content))
        assert "VintagePanel" in code_text
        assert "pack_realtime_cartridge" in code_text
        assert "load_realtime_cartridge" in code_text
        assert "news_or_noise" in code_text

    def test_nb58_interactive_knobs_and_asserts(self):
        content = _require_file(NB_PATHS_EN["58"])
        cells = parse_jupytext_cells(content)
        found_knob = any(c_type == "code" and "# ← change this" in body for c_type, body in cells)
        knob_has_assert = any(c_type == "code" and "# ← change this" in body and "assert " in body for c_type, body in cells)
        assert found_knob, "Missing '# ← change this' knob in Notebook 58"
        assert knob_has_assert, "Interactive knob cell must contain downstream assertions"

    def test_nb58_math_latex_formulation(self):
        content = _require_file(NB_PATHS_EN["58"])
        md_text = "\n".join(extract_markdown_cells(content))
        assert "$" in md_text
        assert ("Mankiw" in md_text or "Shapiro" in md_text or "news" in md_text.lower() or "noise" in md_text.lower())
        assert ("triangle" in md_text.lower() or "vintage" in md_text.lower() or "pmz" in md_text.lower() or "cartridge" in md_text.lower())


# ============================================================================
# Tier 2: Boundary, Corner & AST / Pyodide Analysis
# ============================================================================

class TestTier2BoundaryAndAST:
    """Tier 2: Boundary conditions, AST analysis, Pyodide purity, and styling invariants."""

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOK_PATHS, ids=[p.name for p in ALL_NOTEBOOK_PATHS])
    def test_pyodide_strict_four_package_ast_sweep(self, nb_path: Path):
        """Verify all 6 notebooks strictly obey the Pyodide 4-package standard without foreign deps."""
        content = _require_file(nb_path)
        code_text = "\n".join(extract_code_cells(content))
        imports = extract_imports(code_text)
        violators = imports.intersection(FORBIDDEN_PACKAGES)
        assert not violators, f"Forbidden imports found in {nb_path.name}: {violators}"

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOK_PATHS, ids=[p.name for p in ALL_NOTEBOOK_PATHS])
    def test_style_preamble_apply_style(self, nb_path: Path):
        """Verify all 6 notebooks configure publication styling via _nbstyle.apply_style()."""
        content = _require_file(nb_path)
        assert "_nbstyle" in content, f"{nb_path.name} must import _nbstyle"
        assert "apply_style" in content, f"{nb_path.name} must call _nbstyle.apply_style()"

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOK_PATHS, ids=[p.name for p in ALL_NOTEBOOK_PATHS])
    def test_deterministic_random_seeds(self, nb_path: Path):
        """Verify all 6 notebooks configure deterministic random seeds."""
        content = _require_file(nb_path)
        code_text = "\n".join(extract_code_cells(content))
        has_seed = ("default_rng(" in code_text or "np.random.seed(" in code_text or "seed=" in code_text)
        assert has_seed, f"{nb_path.name} must set a deterministic random seed"

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOK_PATHS, ids=[p.name for p in ALL_NOTEBOOK_PATHS])
    def test_clean_typography_no_emojis(self, nb_path: Path):
        """Verify no emojis appear in notebooks adhering to repository typography standards."""
        content = _require_file(nb_path)
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]"
        )
        assert not emoji_pattern.search(content), f"Emoji detected in {nb_path.name}"

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOK_PATHS, ids=[p.name for p in ALL_NOTEBOOK_PATHS])
    def test_interactive_knob_downstream_assertions(self, nb_path: Path):
        """Verify the interactive knob code cell contains downstream assert statements."""
        content = _require_file(nb_path)
        cells = parse_jupytext_cells(content)
        found_knob = False
        knob_cell_has_assert = False
        for c_type, body in cells:
            if c_type == "code" and ("# ← change this" in body or "# ← Change this" in body):
                found_knob = True
                if "assert " in body:
                    knob_cell_has_assert = True
        assert found_knob, f"No '# ← change this' knob found in {nb_path.name}"
        assert knob_cell_has_assert, f"Interactive knob cell in {nb_path.name} must contain downstream asserts"

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOK_PATHS, ids=[p.name for p in ALL_NOTEBOOK_PATHS])
    def test_ast_syntax_validity(self, nb_path: Path):
        """Verify all 6 notebooks parse as syntactically valid Python via ast.parse."""
        content = _require_file(nb_path)
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError parsing {nb_path.name}: {e}")

    @pytest.mark.parametrize("ipynb_path", ALL_IPYNB_PATHS, ids=[p.name for p in ALL_IPYNB_PATHS])
    def test_ipynb_structure_and_executed_outputs(self, ipynb_path: Path):
        """Verify pre-rendered .ipynb artifacts are valid JSON with non-empty code outputs."""
        assert ipynb_path.exists(), f"Notebook artifact {ipynb_path.name} does not exist"
        data = json.loads(ipynb_path.read_text(encoding="utf-8"))
        assert "cells" in data and len(data["cells"]) >= 7
        code_cells = [c for c in data["cells"] if c.get("cell_type") == "code"]
        assert len(code_cells) >= 4, f"{ipynb_path.name} should have at least 4 code cells"
        # At least one code cell must have executed outputs
        has_outputs = any(len(c.get("outputs", [])) > 0 for c in code_cells)
        assert has_outputs, f"{ipynb_path.name} must contain pre-rendered execution outputs"


# ============================================================================
# Tier 3: Bilingual Parity & Cross-Feature Subsystem Combinations
# ============================================================================

class TestTier3BilingualParityAndCrossFeature:
    """Tier 3: Bilingual code-identity, translation parity, and sequence ordering."""

    @pytest.mark.parametrize("stem", ["56", "57", "58"])
    def test_bilingual_code_cell_counts_match(self, stem: str):
        """Verify English and Spanish editions have the exact same number of code cells."""
        en_content = _require_file(NB_PATHS_EN[stem])
        es_content = _require_file(NB_PATHS_ES[stem])
        en_cells = extract_code_cells(en_content)
        es_cells = extract_code_cells(es_content)
        assert len(en_cells) == len(es_cells), (
            f"Code cell count mismatch for showcase {stem}: English has {len(en_cells)}, Spanish has {len(es_cells)}"
        )

    @pytest.mark.parametrize("stem", ["56", "57", "58"])
    def test_bilingual_code_cells_identical_logic(self, stem: str):
        """Verify normalize_code(en) == normalize_code(es) for every code cell across all 3 pairs."""
        en_content = _require_file(NB_PATHS_EN[stem])
        es_content = _require_file(NB_PATHS_ES[stem])
        en_cells = extract_code_cells(en_content)
        es_cells = extract_code_cells(es_content)
        for idx, (en_c, es_c) in enumerate(zip(en_cells, es_cells)):
            norm_en = normalize_code(en_c)
            norm_es = normalize_code(es_c)
            assert norm_en == norm_es, (
                f"Showcase {stem} code cell {idx + 1} diverges between English and Spanish:\n"
                f"--- EN ---\n{norm_en[:200]}\n--- ES ---\n{norm_es[:200]}"
            )

    @pytest.mark.parametrize("stem", ["56", "57", "58"])
    def test_spanish_headings_and_prose_translated(self, stem: str):
        """Verify markdown headings in Spanish twin are properly translated."""
        es_content = _require_file(NB_PATHS_ES[stem])
        has_spanish = any(term in es_content for term in [
            "El método en matemáticas",
            "Intuición",
            "Lectura de los resultados",
            "Tu turno",
            "¿Qué tan exhaustivo es esto?",
            "método",
            "intuición",
            "resultados",
        ])
        assert has_spanish, f"Spanish twin for showcase {stem} appears not to have translated markdown headings"

    @pytest.mark.parametrize("stem", ["56", "57", "58"])
    def test_template_section_monotonic_ordering(self, stem: str):
        """Verify that all 7 sections appear in monotonic sequential order in English notebook."""
        content = _require_file(NB_PATHS_EN[stem])
        m_mot = re.search(r"(?i)(motivating\s+question|\*\*¿?how\b|\*\*¿?what\b|\*\*¿?can\b|\*\*¿?why\b|\*\*¿?c[oó]mo\b|\*\*¿?qu[eé]\b|\*\*¿?pued\b)", content)
        m_math = re.search(r"(?i)(the\s+method\s+in\s+math|el\s+m[eé]todo\s+en\s+matem[aá]ticas)", content)
        m_int = re.search(r"(?i)(\*\*intuition\.\*\*|\*\*intuici[oó]n\.\*\*|##\s*intuition\b|##\s*intuici[oó]n\b)", content)
        m_out = re.search(r"(?i)(read\s+the\s+output|lectura\s+de\s+los\s+resultados)", content)
        m_turn = re.search(r"(?i)(your\s+turn|tu\s+turno)", content)
        m_comp = re.search(r"(?i)(how\s+comprehensive\s+is\s+this|qu[eé]\s+tan\s+exhaustivo)", content)

        assert m_mot, f"Motivating question missing in showcase {stem}"
        assert m_math, f"Method in math missing in showcase {stem}"
        assert m_int, f"Intuition missing in showcase {stem}"
        assert m_out, f"Read the output missing in showcase {stem}"
        assert m_turn, f"Your turn missing in showcase {stem}"
        assert m_comp, f"How comprehensive is this missing in showcase {stem}"

        p_mot = m_mot.start()
        p_math = m_math.start()
        p_int = m_int.start()
        p_out = m_out.start()
        p_turn = m_turn.start()
        p_comp = m_comp.start()

        assert p_mot < p_math < p_int < p_out < p_turn < p_comp, (
            f"Section order violation in showcase {stem}: "
            f"motivating={p_mot}, math={p_math}, intuition={p_int}, "
            f"output={p_out}, turn={p_turn}, comprehensive={p_comp}"
        )
        code_between = content[p_int:p_out]
        assert "# %%\n" in code_between or "\nimport " in code_between, (
            f"No worked code cells found between intuition and read_the_output in showcase {stem}"
        )

    def test_catalog_readme_integration(self):
        """Verify notebooks/README.md catalog registers showcase notebooks 56, 57, and 58."""
        readme = NB_DIR / "README.md"
        assert readme.exists(), "notebooks/README.md not found"
        content = readme.read_text(encoding="utf-8")
        for stem in ["56", "57", "58"]:
            name = NOTEBOOK_STEMS[stem]
            assert name in content or f"`{stem}`" in content or f"{stem}_" in content, (
                f"notebooks/README.md catalog table must contain entry for showcase {stem}"
            )

    def test_docs_catalog_integration(self):
        """Verify docs/notebooks.md and docs/es/notebooks.md register showcase notebooks 56, 57, and 58."""
        en_doc = DOCS_DIR / "notebooks.md"
        es_doc = DOCS_DIR / "es" / "notebooks.md"
        assert en_doc.exists(), "docs/notebooks.md missing"
        assert es_doc.exists(), "docs/es/notebooks.md missing"
        en_text = en_doc.read_text(encoding="utf-8")
        es_text = es_doc.read_text(encoding="utf-8")
        for stem in ["56", "57", "58"]:
            name = NOTEBOOK_STEMS[stem]
            assert name in en_text, f"docs/notebooks.md missing showcase {stem}"
            assert f"{name}_es" in es_text or name in es_text, f"docs/es/notebooks.md missing showcase {stem}"


# ============================================================================
# Tier 4: Real-World Scenarios & End-to-End Build Execution
# ============================================================================

class TestTier4RealWorldScenariosAndExecution:
    """Tier 4: End-to-end dry-run execution via tools/build_notebooks.py --check."""

    @pytest.mark.parametrize("stem", ["56", "57", "58"])
    def test_notebook_build_execution_check_en(self, stem: str):
        """Verify tools/build_notebooks.py executes English showcase cleanly without error."""
        nb_path = NB_PATHS_EN[stem]
        _require_file(nb_path)
        bn_tool = PROJ / "tools" / "build_notebooks.py"
        spec = importlib.util.spec_from_file_location("build_notebooks", bn_tool)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.build_one(nb_path, check=True)
        assert rc == 0, f"Showcase {nb_path.name} execution check failed with exit code {rc}"

    @pytest.mark.parametrize("stem", ["56", "57", "58"])
    def test_notebook_build_execution_check_es(self, stem: str):
        """Verify tools/build_notebooks.py executes Spanish showcase cleanly without error."""
        nb_path = NB_PATHS_ES[stem]
        _require_file(nb_path)
        bn_tool = PROJ / "tools" / "build_notebooks.py"
        spec = importlib.util.spec_from_file_location("build_notebooks", bn_tool)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.build_one(nb_path, check=True)
        assert rc == 0, f"Spanish showcase {nb_path.name} execution check failed with exit code {rc}"

    def test_showcase_56_hjb_kfe_economic_properties(self):
        """Verify core economic and numerical properties of Showcase 56 HJB/KFE solvers."""
        from puremacro.vfi import solve_hjb_achdou, solve_aiyagari_continuous_hjb

        # HJB implicit solve
        sol = solve_hjb_achdou(
            r_rate=0.03, w_rate=1.0, rho_val=0.05, gamma_r=2.0, Na=50, a_min=0.0, a_max=30.0
        )
        assert sol.converged, "HJB solver should converge"
        assert sol.n_iter <= 20, "HJB should converge within 20 iterations"
        assert sol.V.shape == (50, 2), "Value function shape should be (Na, 2)"
        assert sol.g_dist.shape == (50, 2), "KFE density shape should be (Na, 2)"
        da = sol.a_grid[1] - sol.a_grid[0]
        mass = np.sum(sol.g_dist) * da
        assert np.isclose(mass, 1.0, atol=1e-12), f"KFE mass should be 1.0, got {mass}"
        assert np.all(np.diff(sol.V[:, 0]) > 0), "Value function should be increasing in assets"

        # Continuous Aiyagari GE solve
        ge = solve_aiyagari_continuous_hjb(
            alpha=0.33, delta=0.05, rho_val=0.05, gamma_r=2.0, Na=30, a_max=20.0, max_iter_ge=20
        )
        assert ge.converged, "Aiyagari GE should converge"
        assert 0.005 < ge.r_star < 0.05, f"r_star should be plausible, got {ge.r_star}"
        assert abs(ge.excess_capital) < 1e-3

    def test_showcase_57_occbin_dml_economic_properties(self):
        """Verify core economic and statistical properties of Showcase 57 OccBin & DML."""
        from puremacro.dsge import build_dynare, OccBinConstraint, solve_multiconstraint_occbin
        from puremacro.causal import dml_plr

        # 1. Multi-constraint OccBin
        params = {
            "beta": 0.99, "sigma": 1.0, "kappa": 0.15, "phi_pi": 1.5, "phi_y": 0.25,
            "rho_r": 0.6, "rho_b": 0.5, "rho_g": 0.7, "gamma_y": 0.2, "chi": 0.1,
            "r_ss": 0.015, "b_bar": 0.02,
        }
        variables = ["y", "pi", "r", "b", "g"]
        shocks = ["eps_g", "eps_r", "eps_b"]

        def ref_eqs(lead, curr, lag, shocks_v, p):
            return [
                curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
                curr.pi - p.beta * lead.pi - p.kappa * curr.y,
                curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
                curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
                curr.g - p.rho_g * lag.g - shocks_v.eps_g,
            ]

        def zlb_eqs(lead, curr, lag, shocks_v, p):
            return [
                curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
                curr.pi - p.beta * lead.pi - p.kappa * curr.y,
                curr.r - (-p.r_ss),
                curr.b - (p.rho_b * lag.b + p.gamma_y * curr.y + shocks_v.eps_b),
                curr.g - p.rho_g * lag.g - shocks_v.eps_g,
            ]

        def borr_eqs(lead, curr, lag, shocks_v, p):
            return [
                curr.y - lead.y + (curr.r - lead.pi) / p.sigma - curr.g + p.chi * curr.b,
                curr.pi - p.beta * lead.pi - p.kappa * curr.y,
                curr.r - (p.rho_r * lag.r + (1.0 - p.rho_r) * (p.phi_pi * curr.pi + p.phi_y * curr.y) + shocks_v.eps_r),
                curr.b - p.b_bar,
                curr.g - p.rho_g * lag.g - shocks_v.eps_g,
            ]

        ss = {v: 0.0 for v in variables}
        m_ref = build_dynare(ref_eqs, variables=variables, shocks=shocks, params=params, steady_state=ss)
        m_zlb = build_dynare(zlb_eqs, variables=variables, shocks=shocks, params=params, steady_state=ss, check_steady_state=False, strict=False)
        m_borr = build_dynare(borr_eqs, variables=variables, shocks=shocks, params=params, steady_state=ss, check_steady_state=False, strict=False)

        c_zlb = OccBinConstraint(variable="r", threshold=-params["r_ss"], operator="<")
        c_borr = OccBinConstraint(variable="b", threshold=params["b_bar"], operator=">")

        shocks_mat = np.zeros((20, 3))
        shocks_mat[0, 0] = -0.06
        shocks_mat[0, 2] = 0.05
        res = solve_multiconstraint_occbin(
            m_unconstrained=m_ref,
            m_constrained_dict={"zlb": m_zlb, "borrowing": m_borr},
            shock_seq=shocks_mat,
            constraints={"zlb": c_zlb, "borrowing": c_borr},
            horizon=20,
        )
        assert res.converged, "Multi-constraint OccBin should converge"
        regimes = np.asarray(res.regimes)
        assert 3 in regimes, "Deep shock should trigger joint regime 3"

        # 2. DML PLR
        rng = np.random.default_rng(42)
        N, P = 400, 30
        X = rng.normal(size=(N, P))
        D = 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.5, size=N)
        theta_0 = 1.75
        Y = theta_0 * D + 1.0 * X[:, 0] + 0.8 * X[:, 1] + rng.normal(scale=0.5, size=N)
        dml = dml_plr(Y, D, X, n_folds=5, learner="lasso", random_state=42)
        assert abs(dml.theta - theta_0) < 0.25, f"DML theta {dml.theta} should be close to {theta_0}"

    def test_showcase_58_realtime_macro_economic_properties(self):
        """Verify core economic and econometric properties of Showcase 58 Real-Time Macro."""
        import tempfile
        import pandas as pd
        from puremacro.fetch.realtime import (
            VintagePanel,
            pack_realtime_cartridge,
            load_realtime_cartridge,
        )

        rng = np.random.default_rng(42)
        ref_dates = pd.date_range("2022-01-01", "2024-10-01", freq="QS").strftime("%Y-%m-%d").tolist()
        vintage_dates = pd.date_range("2023-01-01", "2025-04-01", freq="QS").strftime("%Y-%m-%d").tolist()

        records = []
        for v in vintage_dates:
            for d_idx, d in enumerate(ref_dates):
                if d <= v:
                    base_val = 24000000.0 + 100000.0 * d_idx
                    noise = float(rng.normal(0, 30000.0)) if d == v else 0.0
                    records.append({
                        "country": "MEX",
                        "variable": "gdp_real",
                        "date": d,
                        "vintage": v,
                        "value": base_val + noise,
                        "provider": "inegi",
                        "series_id": "735848",
                        "units": "MXN_millions",
                    })
        df = pd.DataFrame(records)
        panel = VintagePanel(df)
        tri = panel.triangle("MEX", "gdp_real")
        assert not tri.empty

        ms = panel.news_or_noise("MEX", "gdp_real")
        assert hasattr(ms, "verdict")
        assert hasattr(ms, "p_beta_on_preliminary")

        with tempfile.TemporaryDirectory() as tmpdir:
            cart_path = Path(tmpdir) / "latin_macro.pmz"
            pack_realtime_cartridge(panel, cart_path)
            assert cart_path.exists() and cart_path.stat().st_size > 100
            loaded = load_realtime_cartridge(cart_path, verify=True)
            assert len(loaded) == len(panel)


# ============================================================================
# Tier 5: Rendered .ipynb artifacts must match the code they claim to show
# ============================================================================

def _ipynb_cells(path: Path) -> list[dict[str, Any]]:
    return json.loads(_require_file(path))["cells"]


def _ipynb_code_sources(path: Path) -> list[str]:
    out: list[str] = []
    for cell in _ipynb_cells(path):
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            out.append("".join(src) if isinstance(src, list) else src)
    return out


def _ipynb_stdout(path: Path) -> str:
    """Everything the notebook's code cells printed, in order."""
    chunks: list[str] = []
    for cell in _ipynb_cells(path):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name", "stdout") == "stdout":
                text = out.get("text", "")
                chunks.append("".join(text) if isinstance(text, list) else text)
    return "".join(chunks)


_SOURCE_IPYNB_PAIRS = [
    (NB_PATHS_EN[stem], IPYNB_PATHS_EN[stem]) for stem in ("56", "57", "58")
] + [
    (NB_PATHS_ES[stem], IPYNB_PATHS_ES[stem]) for stem in ("56", "57", "58")
]


class TestTier5RenderedOutputsMatchCode:
    """A shipped .ipynb is a claim about what the code prints; hold it to that.

    The Tier 4 execution check proves the source *runs*; it says nothing about
    whether the committed .ipynb was rebuilt afterwards. Each printed-number
    test below replicates one computation exactly as the notebook's code cell
    does it (same inputs, same seed, same format string) and looks for that
    string in the .ipynb stdout, so an artifact rendered from older code, or
    from code that now gives a different answer, fails here.
    """

    @pytest.mark.parametrize("ipynb_path", ALL_IPYNB_PATHS, ids=[p.name for p in ALL_IPYNB_PATHS])
    def test_ipynb_has_no_error_outputs(self, ipynb_path: Path):
        for idx, cell in enumerate(_ipynb_cells(ipynb_path)):
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", (
                    f"{ipynb_path.name} cell {idx} holds a traceback: "
                    f"{out.get('ename')}: {out.get('evalue')}"
                )

    @pytest.mark.parametrize(
        "src_path,ipynb_path", _SOURCE_IPYNB_PAIRS, ids=[p[1].name for p in _SOURCE_IPYNB_PAIRS]
    )
    def test_ipynb_code_cells_match_the_jupytext_source(self, src_path: Path, ipynb_path: Path):
        src_cells = [normalize_code(c) for c in extract_code_cells(_require_file(src_path))]
        nb_cells = [normalize_code(c) for c in _ipynb_code_sources(ipynb_path)]
        assert nb_cells == src_cells, (
            f"{ipynb_path.name} was built from a different {src_path.name}; rebuild it "
            "with tools/build_notebooks.py"
        )

    @pytest.mark.parametrize(
        "ipynb_path", [IPYNB_PATHS_EN["56"], IPYNB_PATHS_ES["56"]], ids=["en", "es"]
    )
    def test_nb56_printed_hjb_numbers_are_current(self, ipynb_path: Path):
        from puremacro.vfi import solve_hjb_achdou

        # Experiment 1 and 2 of the notebook, verbatim.
        sol = solve_hjb_achdou(
            r_rate=0.03, w_rate=1.0, rho_val=0.05, gamma_r=2.0,
            Na=50, a_min=0.0, a_max=30.0, tol=1e-8, max_iter=100,
        )
        da = sol.a_grid[1] - sol.a_grid[0]
        capital_supply = float(np.sum(sol.a_grid[:, None] * sol.g_dist * da))
        marginal_g_a = np.sum(sol.g_dist, axis=1) * da
        cumulative_g_a = np.cumsum(marginal_g_a)
        cum_wealth = np.cumsum(sol.a_grid * marginal_g_a) / capital_supply
        gini = float(1.0 - np.sum((cum_wealth[:-1] + cum_wealth[1:]) * np.diff(cumulative_g_a)))

        stdout = _ipynb_stdout(ipynb_path)
        assert f"Iterations         : {sol.n_iter} (expected <= 20)" in stdout, stdout[:600]
        assert f"Aggregate Capital Supply (Ks) : {capital_supply:.4f}" in stdout, stdout[:1200]
        # The "Read the output" prose quotes this Gini; a stale artifact drifts from it.
        assert f"Wealth Gini Coefficient       : {gini:.4f}" in stdout, stdout[:1600]

    @pytest.mark.parametrize(
        "ipynb_path", [IPYNB_PATHS_EN["56"], IPYNB_PATHS_ES["56"]], ids=["en", "es"]
    )
    def test_nb56_printed_ge_prices_and_market_clearing_are_current(self, ipynb_path: Path):
        """The hero market-clearing panel must cross where the solver says r* is.

        The capital demand curve is K^d = L* (alpha/(r+delta))^(1/(1-alpha)); dropping
        the L* factor plots K/L instead and moves the crossing about a percentage
        point away from the r* line the same panel draws.
        """
        from puremacro.vfi import solve_aiyagari_continuous_hjb

        ge = solve_aiyagari_continuous_hjb(
            alpha=0.33, delta=0.05, rho_val=0.05, gamma_r=2.0,
            Na=40, a_max=25.0, tol_ge=1e-4, max_iter_ge=30,
        )
        stdout = _ipynb_stdout(ipynb_path)
        assert f"Equilibrium Rate (r*)  : {ge.r_star:.6f} ({ge.r_star * 100:.3f}%)" in stdout, stdout[:2000]
        assert f"Equilibrium Wage (w*)  : {ge.w_star:.4f}" in stdout, stdout[:2000]
        assert f"Aggregate Capital (K*) : {ge.K_star:.4f}" in stdout, stdout[:2000]
        assert f"Aggregate Labor (L*)   : {ge.L_star:.4f}" in stdout, stdout[:2000]

        crossings = re.findall(r"Sampled Crossing Rate  : ([0-9.]+) ", stdout)
        assert crossings, f"{ipynb_path.name} never printed the sampled Ks/Kd crossing"
        assert abs(float(crossings[0]) - ge.r_star) < 5e-4, (
            f"{ipynb_path.name} plots a market-clearing crossing at {crossings[0]} while the "
            f"solver's r* is {ge.r_star:.6f}"
        )

    @pytest.mark.parametrize(
        "ipynb_path", [IPYNB_PATHS_EN["57"], IPYNB_PATHS_ES["57"]], ids=["en", "es"]
    )
    def test_nb57_printed_dml_estimate_is_current(self, ipynb_path: Path):
        from puremacro.causal import dml_plr

        # Experiment 2 of the notebook, verbatim: the notebook's rng is first
        # drawn from here, so a fresh default_rng(42) reproduces X, D and Y.
        rng = np.random.default_rng(42)
        n_samples, p_controls, theta_true = 500, 30, 1.75
        X_mat = rng.standard_normal((n_samples, p_controls))
        g_true = 0.8 * X_mat[:, 0] - 1.0 * X_mat[:, 1] + 0.5 * (X_mat[:, 2] ** 2)
        m_true = 0.7 * X_mat[:, 0] + 0.9 * X_mat[:, 1] - 0.4 * X_mat[:, 3]
        D_treat = m_true + 0.8 * rng.standard_normal(n_samples)
        Y_outcome = D_treat * theta_true + g_true + 0.8 * rng.standard_normal(n_samples)
        res = dml_plr(Y_outcome, D_treat, X_mat, n_folds=5, learner="lasso", random_state=42)

        X_augmented = np.column_stack([D_treat, np.ones(n_samples), X_mat])
        beta_ols = np.linalg.lstsq(X_augmented, Y_outcome, rcond=None)[0]
        theta_ols = float(beta_ols[0])
        resid_ols = Y_outcome - X_augmented @ beta_ols
        se_ols = float(np.sqrt(np.diag(np.linalg.pinv(X_augmented.T @ X_augmented) * np.var(resid_ols))[0]))

        stdout = _ipynb_stdout(ipynb_path)
        expected = f"DML (Lasso)        : theta = {res.theta:.4f} +/- {1.96 * res.se:.4f}"
        assert expected in stdout, (expected, stdout[:1500])
        # The prose claims OLS is unbiased here and its band is the tightest drawn.
        assert f"Naive OLS          : theta = {theta_ols:.4f} +/- {1.96 * se_ols:.4f}" in stdout, stdout[:2000]
        assert 1.96 * se_ols < 1.96 * res.se, (
            "the notebook's prose says the OLS band is tighter than DML's; it no longer is"
        )

    @pytest.mark.parametrize(
        "ipynb_path", [IPYNB_PATHS_EN["57"], IPYNB_PATHS_ES["57"]], ids=["en", "es"]
    )
    def test_nb57_printed_occbin_regimes_are_current(self, ipynb_path: Path):
        """The OccBin narrative quotes an iteration count and a linear impact rate."""
        stdout = _ipynb_stdout(ipynb_path)
        assert "Iterations            : 3" in stdout, stdout[:1200]
        assert "Active Regimes Across Time : [3 1 1 0 0 0 0 0 0 0 0 0]" in stdout, stdout[:1200]
        assert "Linear Rate at Impact : r_0 = -0.0264" in stdout, stdout[:1600]
        assert "Output Trough         : OccBin -0.1464 vs Linear -0.0943" in stdout, stdout[:1600]

    @pytest.mark.parametrize(
        "ipynb_path", [IPYNB_PATHS_EN["58"], IPYNB_PATHS_ES["58"]], ids=["en", "es"]
    )
    def test_nb58_printed_news_noise_numbers_are_current(self, ipynb_path: Path):
        from puremacro.fetch.realtime import (
            VintagePanel,
            load_realtime_cartridge,
            pack_realtime_cartridge,
        )

        # Experiments 1, 2 and 4 of the notebook, verbatim (panel construction
        # is the notebook's only rng consumer, so the seed reproduces it).
        rng = np.random.default_rng(42)
        ref_dates = pd.date_range("2022-01-01", "2025-07-01", freq="QS").strftime("%Y-%m-%d").tolist()
        vintage_dates = pd.date_range("2024-01-01", "2025-10-01", freq="QS").strftime("%Y-%m-%d").tolist()
        rows = []
        for v in vintage_dates:
            for d_idx, d in enumerate(ref_dates):
                if d <= v:
                    base_gdp = 24000000.0 + 150000.0 * d_idx
                    recent = d == v or (pd.to_datetime(v) - pd.to_datetime(d)).days <= 180
                    noise = float(rng.normal(0, 50000.0)) if recent else 0.0
                    rows.append({
                        "country": "MEX", "variable": "gdp_real", "date": d, "vintage": v,
                        "value": base_gdp + noise, "provider": "inegi", "series_id": "735848",
                        "units": "level",
                    })
                    rows.append({
                        "country": "MEX", "variable": "policy_rate", "date": d, "vintage": v,
                        "value": 11.25 - 0.25 * d_idx, "provider": "banxico", "series_id": "SF61745",
                        "units": "rate",
                    })
                    rows.append({
                        "country": "BRA", "variable": "policy_rate", "date": d, "vintage": v,
                        "value": 12.75 - 0.50 * d_idx, "provider": "bcb", "series_id": "432",
                        "units": "rate",
                    })
                    rows.append({
                        "country": "CHL", "variable": "policy_rate", "date": d, "vintage": v,
                        "value": max(3.00, 9.50 - 0.50 * d_idx), "provider": "bcch",
                        "series_id": "F022.TPM.TPO.D001.NO.Z.D", "units": "rate",
                    })
        panel_raw = VintagePanel(pd.DataFrame(rows))
        with tempfile.TemporaryDirectory() as tmp_dir:
            cartridge_file = Path(tmp_dir) / "latam_realtime_macro.pmz"
            pack_realtime_cartridge(
                panel_raw, cartridge_file,
                source="SIMULATED panel on Banxico, INEGI, BCB and BCCh series identifiers",
                vintage="2026-04-01",
                notes=(
                    "Latin America real-time vintage cartridge for puremacro showcase 58. "
                    "SYNTHETIC DATA: values are generated from a fixed seed, not fetched from "
                    "any provider. Vintage columns are snapshot dates, not published editions."
                ),
            )
            loaded_panel = load_realtime_cartridge(cartridge_file, verify=True)
        ms_res = loaded_panel.news_or_noise("MEX", "gdp_real")

        stdout = _ipynb_stdout(ipynb_path)
        assert f"Total Observations : {len(panel_raw):,}" in stdout, stdout[:600]
        expected = (
            f"beta_p              : {ms_res.beta_on_preliminary:.4f} "
            f"(SE = {ms_res.se_beta_on_preliminary:.4f})"
        )
        assert expected in stdout, (expected, stdout[:2000])
        # The prose reports the verdict; it read "noise" while the code said "neither".
        assert f"Test Verdict          : {ms_res.verdict}" in stdout, stdout[:2500]
        assert f"Noise Share Metric    : {ms_res.noise_share * 100:.2f}%" in stdout, stdout[:2500]
        # The simulated Chilean path is floored, so no policy rate may reach zero.
        min_rate = float(pd.DataFrame(rows).query("variable == 'policy_rate'")["value"].min())
        assert min_rate > 0.0
        assert f"Lowest Policy Rate : {min_rate:.2f}%" in stdout, stdout[:1200]

    @pytest.mark.parametrize(
        "ipynb_path", ALL_IPYNB_PATHS, ids=[p.name for p in ALL_IPYNB_PATHS]
    )
    def test_ipynb_has_no_warning_output(self, ipynb_path: Path):
        """A UserWarning baked into a shipped artifact is a defect the reader sees.

        58 used to emit ``log_diff_pct dropped 5 non-positive value(s)`` because its
        simulated Chilean policy rate ran below zero; that warning is stderr in the
        stored .ipynb, so guard it here rather than trusting the prose.
        """
        chunks: list[str] = []
        for cell in _ipynb_cells(ipynb_path):
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream" and out.get("name") == "stderr":
                    text = out.get("text", "")
                    chunks.append("".join(text) if isinstance(text, list) else text)
        stderr = "".join(chunks)
        assert "Warning" not in stderr, f"{ipynb_path.name} ships a warning in its output:\n{stderr[:800]}"

    @pytest.mark.parametrize(
        "nb_path", [NB_PATHS_EN["58"], NB_PATHS_ES["58"]], ids=["en", "es"]
    )
    def test_nb58_discloses_that_the_panel_is_simulated(self, nb_path: Path):
        """58 builds its panel with an rng; the reader must never take it for real data."""
        content = _require_file(nb_path)
        markdown = extract_markdown_cells(content)
        first = markdown[0].lower()
        assert ("simulated" in first or "simulado" in first), (
            f"{nb_path.name}: the opening markdown cell must say the panel is simulated"
        )
        code_text = "\n".join(extract_code_cells(content))
        assert "SIMULATED" in code_text, (
            f"{nb_path.name}: the panel-building cell must label the data SIMULATED in its output"
        )
        output_cell = next(c for c in markdown if "Read the output" in c or "Lectura de los resultados" in c)
        assert ("simulated" in output_cell.lower() or "simulad" in output_cell.lower()), (
            f"{nb_path.name}: the results narrative must repeat that the panel is simulated"
        )
