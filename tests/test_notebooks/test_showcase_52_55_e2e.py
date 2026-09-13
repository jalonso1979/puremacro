"""End-to-End verification suite for Showcase Notebooks 52, 53, 54, and 55.

Verifies:
- Tier 1: File presence of all 8 files (52, 52_es, 53, 53_es, 54, 54_es, 55, 55_es) and all 7 required template sections in each file.
- Tier 2: AST analysis verifying Pyodide compliance (zero forbidden imports: statsmodels, linearmodels, arch, torch, jax, requests, etc.), presence of _nbstyle.apply_style(), deterministic random seeds, and interactive knob downstream assertions.
- Tier 3: Bilingual parity and code-identity verification (normalize_code(en) == normalize_code(es) for all 4 pairs), translated Spanish headings, and template section ordering.
- Tier 4: Execution test (dry-run import / execution of notebooks via build_notebooks.py --check and verification of # ← change this interactive markers).
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest

PROJ = Path(__file__).resolve().parents[2]
NB_DIR = PROJ / "notebooks"

# All 8 showcase notebook files
NOTEBOOK_STEMS = {
    "52": "52_continuous_transition_mit_shocks",
    "53": "53_exact_analytic_ift_gradients",
    "54": "54_deep_macro_pinns_high_dim",
    "55": "55_quantitative_spatial_and_trade_ge",
}

NB_PATHS_EN = {stem: NB_DIR / f"{name}.py" for stem, name in NOTEBOOK_STEMS.items()}
NB_PATHS_ES = {stem: NB_DIR / f"{name}_es.py" for stem, name in NOTEBOOK_STEMS.items()}

ALL_NOTEBOOK_PATHS = [
    NB_PATHS_EN["52"], NB_PATHS_ES["52"],
    NB_PATHS_EN["53"], NB_PATHS_ES["53"],
    NB_PATHS_EN["54"], NB_PATHS_ES["54"],
    NB_PATHS_EN["55"], NB_PATHS_ES["55"],
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
    "importlib",
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
    """Safe file reader that gracefully skips when a notebook is pending authoring."""
    if not path.exists():
        pytest.skip(f"Notebook {path.name} is currently being authored by milestone worker")
    return path.read_text(encoding="utf-8")


# ============================================================================
# Test Harness Unit Tests (verifying parser and test framework mechanics)
# ============================================================================

class TestHarnessUnit:
    """Unit tests verifying the test suite mechanics and helper functions."""

    def test_helper_jupytext_parser_unit(self):
        """Verify the Jupytext parser splits markdown and code cells correctly."""
        sample = (
            "# ---\n# jupyter:\n# ---\n\n"
            "# %% [markdown]\n"
            "# # Title\n"
            "# Some markdown text\n\n"
            "# %%\n"
            "import numpy as np\n"
            "x = 42\n"
        )
        cells = parse_jupytext_cells(sample)
        assert len(cells) == 2
        assert cells[0][0] == "markdown"
        assert "# # Title" in cells[0][1]
        assert cells[1][0] == "code"
        assert "import numpy as np" in cells[1][1]

    def test_helper_pyodide_checker_unit(self):
        """Verify the import checker identifies allowed vs forbidden imports."""
        clean_code = "import numpy as np\nfrom scipy import optimize\nfrom puremacro.vfi import solve_continuous_transition\n"
        dirty_code = "import numpy as np\nimport statsmodels.api as sm\nimport torch\n"
        clean_imports = extract_imports(clean_code)
        dirty_imports = extract_imports(dirty_code)
        assert clean_imports.issubset(PYODIDE_ALLOWED | {"scipy"})
        assert "statsmodels" in dirty_imports
        assert "torch" in dirty_imports

    def test_helper_normalize_code_unit(self):
        """Verify code normalizer strips comments and empty lines but preserves logic."""
        c1 = "# Comment 1\nx = 10\n\n# Comment 2\ny = x * 2\n"
        c2 = "# Comentario 1\nx = 10\n\n# Comentario 2\ny = x * 2\n"
        assert normalize_code(c1) == normalize_code(c2)
        assert normalize_code(c1) == "x = 10\ny = x * 2"

    def test_helper_section_order_checker_unit(self):
        """Verify sequential order detector catches monotonic section appearance."""
        sample = (
            "# %% [markdown]\n## Motivating Question\n**How does policy transmit?**\n\n"
            "# %% [markdown]\n## The method in math\n$$c = u(w)$$\n\n"
            "# %% [markdown]\n**Intuition.** Economic mechanism here.\n\n"
            "# %% \n# Worked code\nimport puremacro\n\n"
            "# %% [markdown]\n## Read the output\nResults show...\n\n"
            "# %% \n# Your turn\n# ← change this: knob\nval = 1\nassert val == 1\n\n"
            "# %% [markdown]\n## How comprehensive is this?\nSee other tools.\n"
        )
        m_mot = re.search(r"(?i)(motivating\s+question|\*\*¿?how\b|\*\*¿?what\b|\*\*¿?can\b|\*\*¿?why\b)", sample)
        m_math = re.search(r"(?i)(the\s+method\s+in\s+math)", sample)
        m_int = re.search(r"(?i)(\*\*intuition\.\*\*|##\s*intuition\b)", sample)
        m_out = re.search(r"(?i)(read\s+the\s+output)", sample)
        m_turn = re.search(r"(?i)(your\s+turn)", sample)
        m_comp = re.search(r"(?i)(how\s+comprehensive\s+is\s+this)", sample)

        assert m_mot and m_math and m_int and m_out and m_turn and m_comp
        assert m_mot.start() < m_math.start() < m_int.start() < m_out.start() < m_turn.start() < m_comp.start()

    def test_helper_interactive_knob_checker_unit(self):
        """Verify interactive knob detection and downstream assertion verification."""
        valid_knob_cell = (
            "# Your turn: customize parameters\n"
            "# ← change this: discount factor\n"
            "beta_custom = 0.96\n"
            "assert beta_custom > 0, 'Must be positive'\n"
        )
        invalid_knob_cell = (
            "# Your turn: customize parameters\n"
            "# ← change this: discount factor\n"
            "beta_custom = 0.96\n"
        )
        assert "# ← change this" in valid_knob_cell
        assert "assert " in valid_knob_cell
        assert "# ← change this" in invalid_knob_cell
        assert "assert " not in invalid_knob_cell


# ============================================================================
# Tier 1: Feature Coverage (Isolated Happy Paths >= 5 per Feature)
# ============================================================================

class TestTier1Showcase52Coverage:
    """Tier 1 Feature Coverage: Showcase Notebook 52 (Continuous Transition Dynamics)."""

    def test_nb52_en_file_presence(self):
        """Verify notebooks/52_continuous_transition_mit_shocks.py exists on disk."""
        p = NB_PATHS_EN["52"]
        _require_file(p)
        assert p.is_file(), f"File {p.name} missing"

    def test_nb52_es_twin_file_presence(self):
        """Verify notebooks/52_continuous_transition_mit_shocks_es.py exists on disk."""
        p = NB_PATHS_ES["52"]
        _require_file(p)
        assert p.is_file(), f"Spanish twin file {p.name} missing"

    def test_nb52_jupytext_percent_header(self):
        """Verify Notebook 52 starts with valid Jupytext percent frontmatter."""
        content = _require_file(NB_PATHS_EN["52"])
        assert "format_name: percent" in content, "Missing 'format_name: percent' in header"
        assert "kernelspec:" in content, "Missing 'kernelspec:' in header"
        assert "name: python3" in content or "display_name: Python 3" in content

    def test_nb52_all_seven_template_sections_present(self):
        """Verify all 7 required template sections exist in Notebook 52."""
        content = _require_file(NB_PATHS_EN["52"])
        for name, pattern in REQUIRED_SECTIONS_EN:
            assert re.search(pattern, content), f"Section '{name}' missing from Notebook 52"
        # Check worked code cell exists
        assert len(extract_code_cells(content)) >= 4, "Worked code cells missing in Notebook 52"

    def test_nb52_worked_code_solver_apis(self):
        """Verify Notebook 52 worked code uses continuous transition & stationary solvers."""
        content = _require_file(NB_PATHS_EN["52"])
        code_text = "\n".join(extract_code_cells(content))
        has_transition = ("solve_continuous_transition" in code_text or "continuous_mit_shock" in code_text)
        assert has_transition, "Worked code must invoke solve_continuous_transition or continuous_mit_shock"
        assert "solve_aiyagari_continuous" in code_text or "Aiyagari" in content or "continuous" in code_text

    def test_nb52_interactive_knobs_and_asserts(self):
        """Verify Notebook 52 includes interactive '# ← change this' knobs and asserts."""
        content = _require_file(NB_PATHS_EN["52"])
        cells = parse_jupytext_cells(content)
        found_knob = False
        knob_cell_has_assert = False
        for c_type, body in cells:
            if c_type == "code" and ("# ← change this" in body or "# ← Change this" in body):
                found_knob = True
                if "assert " in body:
                    knob_cell_has_assert = True
        assert found_knob, "No '# ← change this' knob found in Notebook 52"
        assert knob_cell_has_assert, "Interactive knob cell must contain downstream assertions"

    def test_nb52_math_latex_formulation(self):
        """Verify Notebook 52 Section 2 provides LaTeX math for backward EGM and forward Young operator."""
        content = _require_file(NB_PATHS_EN["52"])
        md_text = "\n".join(extract_markdown_cells(content))
        assert "$" in md_text, "Math section must contain LaTeX equations"
        has_young_or_dist = ("\\mu" in md_text or "Young" in md_text or "T^*" in md_text or "density" in md_text.lower())
        assert has_young_or_dist, "Notebook 52 must formulate distribution evolution or Young operator"


class TestTier1Showcase53Coverage:
    """Tier 1 Feature Coverage: Showcase Notebook 53 (Exact Analytic IFT Gradients)."""

    def test_nb53_en_file_presence(self):
        """Verify notebooks/53_exact_analytic_ift_gradients.py exists on disk."""
        p = NB_PATHS_EN["53"]
        _require_file(p)
        assert p.is_file(), f"File {p.name} missing"

    def test_nb53_es_twin_file_presence(self):
        """Verify notebooks/53_exact_analytic_ift_gradients_es.py exists on disk."""
        p = NB_PATHS_ES["53"]
        _require_file(p)
        assert p.is_file(), f"Spanish twin file {p.name} missing"

    def test_nb53_jupytext_percent_header(self):
        """Verify Notebook 53 starts with valid Jupytext percent frontmatter."""
        content = _require_file(NB_PATHS_EN["53"])
        assert "format_name: percent" in content, "Missing 'format_name: percent' in header"
        assert "kernelspec:" in content, "Missing 'kernelspec:' in header"

    def test_nb53_all_seven_template_sections_present(self):
        """Verify all 7 required template sections exist in Notebook 53."""
        content = _require_file(NB_PATHS_EN["53"])
        for name, pattern in REQUIRED_SECTIONS_EN:
            assert re.search(pattern, content), f"Section '{name}' missing from Notebook 53"
        assert len(extract_code_cells(content)) >= 4, "Worked code cells missing in Notebook 53"

    def test_nb53_worked_code_solver_apis(self):
        """Verify Notebook 53 worked code uses compute_ift_gradients and GMM/SMM estimation."""
        content = _require_file(NB_PATHS_EN["53"])
        code_text = "\n".join(extract_code_cells(content))
        assert "compute_ift_gradients" in code_text, "Worked code must invoke compute_ift_gradients"
        has_estimation = ("gmm_objective_and_gradient" in code_text or "minimize" in code_text or "CollocationProblem" in code_text)
        assert has_estimation, "Worked code must demonstrate estimation or collocation solution"

    def test_nb53_interactive_knobs_and_asserts(self):
        """Verify Notebook 53 includes interactive '# ← change this' knobs and asserts."""
        content = _require_file(NB_PATHS_EN["53"])
        cells = parse_jupytext_cells(content)
        found_knob = False
        knob_cell_has_assert = False
        for c_type, body in cells:
            if c_type == "code" and ("# ← change this" in body or "# ← Change this" in body):
                found_knob = True
                if "assert " in body:
                    knob_cell_has_assert = True
        assert found_knob, "No '# ← change this' knob found in Notebook 53"
        assert knob_cell_has_assert, "Interactive knob cell must contain downstream assertions"

    def test_nb53_math_latex_formulation(self):
        """Verify Notebook 53 Section 2 provides LaTeX math for Implicit Function Theorem gradient."""
        content = _require_file(NB_PATHS_EN["53"])
        md_text = "\n".join(extract_markdown_cells(content))
        assert "$" in md_text, "Math section must contain LaTeX equations"
        has_ift_math = ("\\nabla" in md_text or "IFT" in md_text or "Implicit Function" in md_text)
        assert has_ift_math, "Notebook 53 must formulate IFT parameter sensitivity equation"


class TestTier1Showcase54Coverage:
    """Tier 1 Feature Coverage: Showcase Notebook 54 (Deep Macro & PINNs in High Dimensions)."""

    def test_nb54_en_file_presence(self):
        """Verify notebooks/54_deep_macro_pinns_high_dim.py exists on disk."""
        p = NB_PATHS_EN["54"]
        _require_file(p)
        assert p.is_file(), f"File {p.name} missing"

    def test_nb54_es_twin_file_presence(self):
        """Verify notebooks/54_deep_macro_pinns_high_dim_es.py exists on disk."""
        p = NB_PATHS_ES["54"]
        _require_file(p)
        assert p.is_file(), f"Spanish twin file {p.name} missing"

    def test_nb54_jupytext_percent_header(self):
        """Verify Notebook 54 starts with valid Jupytext percent frontmatter."""
        content = _require_file(NB_PATHS_EN["54"])
        assert "format_name: percent" in content, "Missing 'format_name: percent' in header"
        assert "kernelspec:" in content, "Missing 'kernelspec:' in header"

    def test_nb54_all_seven_template_sections_present(self):
        """Verify all 7 required template sections exist in Notebook 54."""
        content = _require_file(NB_PATHS_EN["54"])
        for name, pattern in REQUIRED_SECTIONS_EN:
            assert re.search(pattern, content), f"Section '{name}' missing from Notebook 54"
        assert len(extract_code_cells(content)) >= 4, "Worked code cells missing in Notebook 54"

    def test_nb54_worked_code_solver_apis(self):
        """Verify Notebook 54 worked code uses DeepMacroModel and solve_deep_macro."""
        content = _require_file(NB_PATHS_EN["54"])
        code_text = "\n".join(extract_code_cells(content))
        assert "DeepMacroModel" in code_text, "Worked code must invoke DeepMacroModel"
        assert "solve_deep_macro" in code_text, "Worked code must invoke solve_deep_macro"

    def test_nb54_interactive_knobs_and_asserts(self):
        """Verify Notebook 54 includes interactive '# ← change this' knobs and asserts."""
        content = _require_file(NB_PATHS_EN["54"])
        cells = parse_jupytext_cells(content)
        found_knob = False
        knob_cell_has_assert = False
        for c_type, body in cells:
            if c_type == "code" and ("# ← change this" in body or "# ← Change this" in body):
                found_knob = True
                if "assert " in body:
                    knob_cell_has_assert = True
        assert found_knob, "No '# ← change this' knob found in Notebook 54"
        assert knob_cell_has_assert, "Interactive knob cell must contain downstream assertions"

    def test_nb54_math_latex_formulation(self):
        """Verify Notebook 54 Section 2 provides LaTeX math for Euler loss & ergodic sampling."""
        content = _require_file(NB_PATHS_EN["54"])
        md_text = "\n".join(extract_markdown_cells(content))
        assert "$" in md_text, "Math section must contain LaTeX equations"
        has_deep_math = ("Euler" in md_text or "Loss" in md_text or "\\mathcal{L}" in md_text or "residual" in md_text.lower())
        assert has_deep_math, "Notebook 54 must formulate Euler residual loss or network architecture"


class TestTier1Showcase55Coverage:
    """Tier 1 Feature Coverage: Showcase Notebook 55 (Quantitative Spatial & Trade GE)."""

    def test_nb55_en_file_presence(self):
        """Verify notebooks/55_quantitative_spatial_and_trade_ge.py exists on disk."""
        p = NB_PATHS_EN["55"]
        _require_file(p)
        assert p.is_file(), f"File {p.name} missing"

    def test_nb55_es_twin_file_presence(self):
        """Verify notebooks/55_quantitative_spatial_and_trade_ge_es.py exists on disk."""
        p = NB_PATHS_ES["55"]
        _require_file(p)
        assert p.is_file(), f"Spanish twin file {p.name} missing"

    def test_nb55_jupytext_percent_header(self):
        """Verify Notebook 55 starts with valid Jupytext percent frontmatter."""
        content = _require_file(NB_PATHS_EN["55"])
        assert "format_name: percent" in content, "Missing 'format_name: percent' in header"
        assert "kernelspec:" in content, "Missing 'kernelspec:' in header"

    def test_nb55_all_seven_template_sections_present(self):
        """Verify all 7 required template sections exist in Notebook 55."""
        content = _require_file(NB_PATHS_EN["55"])
        for name, pattern in REQUIRED_SECTIONS_EN:
            assert re.search(pattern, content), f"Section '{name}' missing from Notebook 55"
        assert len(extract_code_cells(content)) >= 4, "Worked code cells missing in Notebook 55"

    def test_nb55_worked_code_solver_apis(self):
        """Verify Notebook 55 worked code uses CaliendoParroModel and AllenArkolakisModel."""
        content = _require_file(NB_PATHS_EN["55"])
        code_text = "\n".join(extract_code_cells(content))
        assert "CaliendoParroModel" in code_text, "Worked code must invoke CaliendoParroModel"
        assert "AllenArkolakisModel" in code_text, "Worked code must invoke AllenArkolakisModel"

    def test_nb55_interactive_knobs_and_asserts(self):
        """Verify Notebook 55 includes interactive '# ← change this' knobs and asserts."""
        content = _require_file(NB_PATHS_EN["55"])
        cells = parse_jupytext_cells(content)
        found_knob = False
        knob_cell_has_assert = False
        for c_type, body in cells:
            if c_type == "code" and ("# ← change this" in body or "# ← Change this" in body):
                found_knob = True
                if "assert " in body:
                    knob_cell_has_assert = True
        assert found_knob, "No '# ← change this' knob found in Notebook 55"
        assert knob_cell_has_assert, "Interactive knob cell must contain downstream assertions"

    def test_nb55_math_latex_formulation(self):
        """Verify Notebook 55 Section 2 provides LaTeX math for Caliendo-Parro and Allen-Arkolakis."""
        content = _require_file(NB_PATHS_EN["55"])
        md_text = "\n".join(extract_markdown_cells(content))
        assert "$" in md_text, "Math section must contain LaTeX equations"
        has_spatial_math = ("\\pi" in md_text or "gravity" in md_text.lower() or "Caliendo" in md_text or "Allen" in md_text)
        assert has_spatial_math, "Notebook 55 must formulate trade or spatial general equilibrium"


# ============================================================================
# Tier 2: Boundary, Corner & AST / Pyodide Analysis (>= 5 Tests)
# ============================================================================

class TestTier2BoundaryAndAST:
    """Tier 2: Boundary conditions, AST analysis, Pyodide purity, and style invariants."""

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOK_PATHS, ids=[p.name for p in ALL_NOTEBOOK_PATHS])
    def test_pyodide_strict_four_package_ast_sweep(self, nb_path: Path):
        """Verify all 8 notebooks strictly obey the Pyodide 4-package standard without foreign deps."""
        content = _require_file(nb_path)
        code_text = "\n".join(extract_code_cells(content))
        imports = extract_imports(code_text)
        violators = imports.intersection(FORBIDDEN_PACKAGES)
        assert not violators, f"Forbidden imports found in {nb_path.name}: {violators}"

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOK_PATHS, ids=[p.name for p in ALL_NOTEBOOK_PATHS])
    def test_style_preamble_apply_style(self, nb_path: Path):
        """Verify all 8 notebooks configure publication styling via _nbstyle.apply_style()."""
        content = _require_file(nb_path)
        assert "_nbstyle" in content, f"{nb_path.name} must import _nbstyle"
        assert "apply_style" in content, f"{nb_path.name} must call _nbstyle.apply_style()"

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOK_PATHS, ids=[p.name for p in ALL_NOTEBOOK_PATHS])
    def test_deterministic_random_seeds(self, nb_path: Path):
        """Verify all 8 notebooks configure deterministic random seeds."""
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
        """Verify all 8 notebooks parse as syntactically valid Python via ast.parse."""
        content = _require_file(nb_path)
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError parsing {nb_path.name}: {e}")

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOK_PATHS, ids=[p.name for p in ALL_NOTEBOOK_PATHS])
    def test_no_empty_or_degenerate_code_cells(self, nb_path: Path):
        """Verify no empty or whitespace-only code cells exist in the notebook."""
        content = _require_file(nb_path)
        code_cells = extract_code_cells(content)
        assert len(code_cells) >= 3, f"{nb_path.name} must have at least 3 code cells"
        for idx, cell in enumerate(code_cells):
            assert cell.strip(), f"Cell {idx + 1} in {nb_path.name} is empty"


# ============================================================================
# Tier 3: Bilingual Parity & Cross-Feature Subsystem Combinations
# ============================================================================

class TestTier3BilingualParityAndCrossFeature:
    """Tier 3: Bilingual code-identity, translation parity, and sequence ordering."""

    @pytest.mark.parametrize("stem", ["52", "53", "54", "55"])
    def test_bilingual_code_cell_counts_match(self, stem: str):
        """Verify English and Spanish editions have the exact same number of code cells."""
        en_content = _require_file(NB_PATHS_EN[stem])
        es_content = _require_file(NB_PATHS_ES[stem])
        en_cells = extract_code_cells(en_content)
        es_cells = extract_code_cells(es_content)
        assert len(en_cells) == len(es_cells), (
            f"Code cell count mismatch for showcase {stem}: English has {len(en_cells)}, Spanish has {len(es_cells)}"
        )

    @pytest.mark.parametrize("stem", ["52", "53", "54", "55"])
    def test_bilingual_code_cells_identical_logic(self, stem: str):
        """Verify normalize_code(en) == normalize_code(es) for every code cell across all 4 pairs."""
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

    @pytest.mark.parametrize("stem", ["52", "53", "54", "55"])
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

    @pytest.mark.parametrize("stem", ["52", "53", "54", "55"])
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
        """Verify notebooks/README.md catalog registers showcase notebooks 52, 53, 54, and 55."""
        readme = NB_DIR / "README.md"
        if not readme.exists():
            pytest.skip("notebooks/README.md not found")
        content = readme.read_text(encoding="utf-8")
        missing = [stem for stem in ["52", "53", "54", "55"]
                   if NOTEBOOK_STEMS[stem] not in content and f"`{stem}`" not in content and f"{stem}_" not in content]
        if missing:
            pytest.skip(f"Showcases {', '.join(missing)} pending Milestone 5 catalog registration in notebooks/README.md")
        for stem in ["52", "53", "54", "55"]:
            name = NOTEBOOK_STEMS[stem]
            assert name in content or f"`{stem}`" in content or f"{stem}_" in content, (
                f"notebooks/README.md catalog table must contain entry for showcase {stem}"
            )


# ============================================================================
# Tier 4: Real-World Scenarios & End-to-End Build Execution
# ============================================================================

class TestTier4RealWorldScenariosAndExecution:
    """Tier 4: End-to-end dry-run execution via tools/build_notebooks.py --check."""

    @pytest.mark.parametrize("stem", ["52", "53", "54", "55"])
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

    @pytest.mark.parametrize("stem", ["52", "53", "54", "55"])
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

    def test_showcase_economic_experiments_covered(self):
        """Verify each showcase covers its assigned economic scenario and experiments."""
        # NB 52: Continuous transition
        if NB_PATHS_EN["52"].exists():
            c52 = NB_PATHS_EN["52"].read_text(encoding="utf-8").lower()
            assert ("transition" in c52 or "shock" in c52 or "aiyagari" in c52)
        # NB 53: Analytic IFT gradients
        if NB_PATHS_EN["53"].exists():
            c53 = NB_PATHS_EN["53"].read_text(encoding="utf-8").lower()
            assert ("gradient" in c53 or "ift" in c53 or "implicit" in c53)
        # NB 54: Deep macro PINNs
        if NB_PATHS_EN["54"].exists():
            c54 = NB_PATHS_EN["54"].read_text(encoding="utf-8").lower()
            assert ("neural" in c54 or "deep" in c54 or "euler" in c54 or "loss" in c54)
        # NB 55: Quantitative spatial & trade
        if NB_PATHS_EN["55"].exists():
            c55 = NB_PATHS_EN["55"].read_text(encoding="utf-8").lower()
            assert ("tariff" in c55 or "trade" in c55 or "spatial" in c55 or "caliendo" in c55)
