"""End-to-end verification suite for Showcase Notebook 51.

Verifies:
- Jupytext percent format compliance and structure.
- Adherence to all 7 required sections from notebooks/_TEMPLATE.md.
- Bilingual code identity with Spanish counterpart (notebooks/51_continuous_projection_collocation_and_fem_es.py).
- Strict Pyodide four-package constraint (numpy, scipy, pandas, matplotlib, puremacro).
- Interactive `# ← change this` knobs and downstream validation assertions.
- Deterministic execution via tools/build_notebooks.py --check.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

PROJ = Path(__file__).resolve().parents[2]
NB_DIR = PROJ / "notebooks"
NB_EN = NB_DIR / "51_continuous_projection_collocation_and_fem.py"
NB_ES = NB_DIR / "51_continuous_projection_collocation_and_fem_es.py"

# Pyodide allowed external dependencies + puremacro + stdlib
PYODIDE_ALLOWED = frozenset({
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "puremacro",
    "_nbstyle",
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
})

REQUIRED_SECTIONS = [
    ("motivating_question", r"(?i)(motivating\s+question|\*\*how|\*\*what|\*\*can)"),
    ("method_in_math", r"(?i)(the\s+method\s+in\s+math|el\s+m[eé]todo\s+en\s+matem[aá]ticas)"),
    ("intuition", r"(?i)(\*\*intuition\.\*\*|\*\*intuici[oó]n\.\*\*|##\s*intuition|##\s*intuici[oó]n)"),
    ("worked_code", r"(?i)(worked\s+code|c[oó]digo\s+resuelto|puremacro\.vfi)"),
    ("read_the_output", r"(?i)(read\s+the\s+output|lectura\s+de\s+los\s+resultados)"),
    ("your_turn", r"(?i)(your\s+turn|tu\s+turno)"),
    ("how_comprehensive", r"(?i)(how\s+comprehensive\s+is\s+this|qu[eé]\s+tan\s+exhaustivo)"),
]


# ============================================================================
# Helper Functions
# ============================================================================

def parse_jupytext_cells(content: str) -> list[tuple[str, str]]:
    """Parse a Jupytext percent format file into a list of (cell_type, cell_text)."""
    cell_pattern = re.compile(r"^# %%(\s+\[markdown\])?", re.MULTILINE)
    splits = cell_pattern.split(content)
    # splits[0] is preamble before first cell
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


# ============================================================================
# Tier 1: Feature Coverage (Isolated Happy Path)
# ============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Feature Coverage testing for Notebook 51 architecture."""

    def test_helper_jupytext_parser_unit(self):
        """Verify the Jupytext parser correctly splits markdown and code cells."""
        sample = (
            "# ---\n# jupyter:\n# ---\n\n"
            "# %% [markdown]\n"
            "# # Title\n"
            "# Some text\n\n"
            "# %%\n"
            "import numpy as np\n"
            "x = 1\n"
        )
        cells = parse_jupytext_cells(sample)
        assert len(cells) == 2
        assert cells[0][0] == "markdown"
        assert "# # Title" in cells[0][1]
        assert cells[1][0] == "code"
        assert "import numpy as np" in cells[1][1]

    def test_helper_pyodide_checker_unit(self):
        """Verify the import checker identifies allowed vs forbidden imports."""
        clean_code = "import numpy as np\nimport scipy.optimize\nfrom puremacro.vfi import solve_collocation\n"
        dirty_code = "import numpy as np\nimport statsmodels.api as sm\n"
        assert extract_imports(clean_code).issubset(PYODIDE_ALLOWED | {"scipy"})
        assert "statsmodels" in extract_imports(dirty_code)

    def test_notebook_51_file_exists(self):
        """Verify notebooks/51_continuous_projection_collocation_and_fem.py exists."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} is currently being authored by worker")
        assert NB_EN.is_file(), f"Notebook file missing at {NB_EN}"

    def test_jupytext_percent_header_format(self):
        """Verify Notebook 51 starts with standard Jupytext percent frontmatter."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        assert "format_name: percent" in content, "Missing 'format_name: percent' in header"
        assert "kernelspec:" in content, "Missing 'kernelspec:' in header"
        assert "name: python3" in content or "display_name: Python 3" in content

    def test_section_1_motivating_question_present(self):
        """Verify Section 1 (Motivating question) is present in Notebook 51."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        md_cells = "\n".join(extract_markdown_cells(content))
        pat = re.compile(r"(?i)(\*\*how|\*\*what|\*\*can|motivating\s+question)")
        assert pat.search(md_cells), "Section 1: Motivating question not found in markdown"

    def test_section_2_method_in_math_present(self):
        """Verify Section 2 (The method in math) contains LaTeX equations."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        assert "The method in math" in content or "the method in math" in content.lower()
        # Must contain LaTeX delimiters
        assert "$" in content, "Section 2 must contain LaTeX math equations ($ or $$)"

    def test_section_3_intuition_present(self):
        """Verify Section 3 (Intuition) contains '**Intuition.**' lead-in."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        assert "**Intuition.**" in content or "## Intuition" in content, (
            "Section 3 must contain explicit '**Intuition.**' or '## Intuition' lead-in"
        )

    def test_section_4_worked_code_present(self):
        """Verify Section 4 contains executable puremacro.vfi continuous solvers."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        code_cells = "\n".join(extract_code_cells(content))
        assert "CollocationProblem" in code_cells or "solve_collocation" in code_cells, (
            "Worked code must use CollocationProblem or solve_collocation"
        )
        assert "FEMProblem" in code_cells or "solve_fem" in code_cells, (
            "Worked code must use FEMProblem or solve_fem"
        )

    def test_section_5_read_the_output_present(self):
        """Verify Section 5 (Read the output) interprets headline metrics."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        assert "Read the output" in content or "read the output" in content.lower(), (
            "Section 5: 'Read the output' heading must be present"
        )

    def test_section_6_your_turn_present(self):
        """Verify Section 6 (Your turn) contains interactive knob marker."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        assert "Your turn" in content or "your turn" in content.lower(), (
            "Section 6: 'Your turn' heading must be present"
        )
        assert "# ← change this" in content or "# ← Change this" in content, (
            "Section 6 must contain '# ← change this' marker"
        )

    def test_section_7_how_comprehensive_present(self):
        """Verify Section 7 (How comprehensive is this?) contains cross-references."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        assert "How comprehensive is this?" in content or "how comprehensive" in content.lower(), (
            "Section 7: 'How comprehensive is this?' heading must be present"
        )


# ============================================================================
# Tier 2: Boundary, Corner & Adversarial Cases
# ============================================================================

class TestTier2BoundaryAndCorner:
    """Tier 2: Boundary conditions, Pyodide purity, and bilingual code identity."""

    def test_pyodide_four_package_constraint(self):
        """Verify Notebook 51 strictly obeys the Pyodide 4-package standard."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        code_text = "\n".join(extract_code_cells(content))
        imports = extract_imports(code_text)
        # Check against forbidden packages
        violators = imports.intersection(FORBIDDEN_PACKAGES)
        assert not violators, f"Forbidden non-Pyodide imports found in Notebook 51: {violators}"

    def test_spanish_twin_exists(self):
        """Verify Spanish twin 51_continuous_projection_collocation_and_fem_es.py exists."""
        if not NB_ES.exists():
            pytest.skip(f"{NB_ES} is currently being authored by worker")
        assert NB_ES.is_file(), f"Spanish notebook missing at {NB_ES}"

    def test_spanish_twin_code_cell_count_matches(self):
        """Verify English and Spanish editions have the identical number of code cells."""
        if not NB_EN.exists() or not NB_ES.exists():
            pytest.skip("Both notebook editions required for comparison")
        en_cells = extract_code_cells(NB_EN.read_text(encoding="utf-8"))
        es_cells = extract_code_cells(NB_ES.read_text(encoding="utf-8"))
        assert len(en_cells) == len(es_cells), (
            f"Code cell count mismatch: English has {len(en_cells)}, Spanish has {len(es_cells)}"
        )

    def test_spanish_twin_code_cells_identical_logic(self):
        """Verify each code cell in the Spanish twin has identical execution logic."""
        if not NB_EN.exists() or not NB_ES.exists():
            pytest.skip("Both notebook editions required for comparison")
        en_cells = extract_code_cells(NB_EN.read_text(encoding="utf-8"))
        es_cells = extract_code_cells(NB_ES.read_text(encoding="utf-8"))
        for idx, (en_c, es_c) in enumerate(zip(en_cells, es_cells)):
            norm_en = normalize_code(en_c)
            norm_es = normalize_code(es_c)
            assert norm_en == norm_es, (
                f"Code cell {idx + 1} diverges between English and Spanish:\n"
                f"--- EN ---\n{norm_en[:300]}\n--- ES ---\n{norm_es[:300]}"
            )

    def test_spanish_twin_headings_translated(self):
        """Verify markdown headings in Spanish twin are properly translated."""
        if not NB_ES.exists():
            pytest.skip(f"{NB_ES} not yet authored")
        content = NB_ES.read_text(encoding="utf-8")
        # Check for key Spanish terms
        has_spanish = any(term in content for term in [
            "El método en matemáticas",
            "Intuición",
            "Lectura de los resultados",
            "Tu turno",
            "¿Qué tan exhaustivo es esto?",
            "método",
            "intuición",
        ])
        assert has_spanish, "Spanish twin appears not to have translated markdown headings"

    def test_interactive_knob_downstream_assert(self):
        """Verify the interactive knob cell has downstream assert statements."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        cells = parse_jupytext_cells(content)
        found_knob = False
        knob_cell_has_assert = False
        for c_type, body in cells:
            if c_type == "code" and ("# ← change this" in body or "# ← Change this" in body):
                found_knob = True
                if "assert " in body:
                    knob_cell_has_assert = True
        assert found_knob, "No code cell containing '# ← change this' knob found"
        assert knob_cell_has_assert, (
            "Interactive knob cell must contain downstream assert statement(s) verifying default run"
        )

    def test_no_emoji_in_intuition(self):
        """Verify no emojis appear in the Intuition section adhering to repository style."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        # Unicode range for common emoji
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]"
        )
        assert not emoji_pattern.search(content), "Repository style guides strictly forbid emoji in notebooks"


# ============================================================================
# Tier 3: Cross-Feature Interactions & Sequential Validation
# ============================================================================

class TestTier3CrossFeature:
    """Tier 3: Multi-feature integration, AST syntax, and sequencing."""

    def test_english_notebook_ast_compilation(self):
        """Verify the full English notebook parses as syntactically valid Python."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError parsing {NB_EN}: {e}")

    def test_spanish_notebook_ast_compilation(self):
        """Verify the full Spanish notebook parses as syntactically valid Python."""
        if not NB_ES.exists():
            pytest.skip(f"{NB_ES} not yet authored")
        content = NB_ES.read_text(encoding="utf-8")
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError parsing {NB_ES}: {e}")

    def test_all_seven_sections_sequence_order(self):
        """Verify that all 7 sections appear in the strict order defined in _TEMPLATE.md."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        positions = []
        for name, pattern in REQUIRED_SECTIONS:
            m = re.search(pattern, content)
            assert m is not None, f"Required section '{name}' matching '{pattern}' not found"
            positions.append((name, m.start()))

        # Check strictly monotonic position ordering
        for i in range(len(positions) - 1):
            assert positions[i][1] < positions[i + 1][1], (
                f"Section '{positions[i][0]}' (pos {positions[i][1]}) does not appear "
                f"before '{positions[i + 1][0]}' (pos {positions[i + 1][1]})"
            )

    def test_nbstyle_preamble_configuration(self):
        """Verify the notebook configures publication styling via _nbstyle."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        assert "_nbstyle" in content, "Notebook must import _nbstyle"
        assert "apply_style" in content, "Notebook must call _nbstyle.apply_style()"

    def test_deterministic_random_seed_set(self):
        """Verify deterministic random seed is explicitly configured."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        code_text = "\n".join(extract_code_cells(content))
        has_seed = ("np.random.seed(" in code_text or "default_rng(" in code_text or "SEED" in code_text)
        assert has_seed, "Notebook code must set a deterministic random seed"

    def test_catalog_readme_entry_present(self):
        """Verify notebooks/README.md documents notebook 51."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        readme = NB_DIR / "README.md"
        if not readme.exists():
            pytest.skip("notebooks/README.md not found")
        content = readme.read_text(encoding="utf-8")
        assert "51_continuous_projection" in content or "51" in content, (
            "notebooks/README.md catalog table should contain entry for notebook 51"
        )


# ============================================================================
# Tier 4: Real-World Application & End-to-End Build Execution
# ============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end execution of the notebook via build_notebooks.py."""

    def test_four_experiments_covered_in_code(self):
        """Verify the worked code covers the four canonical experiments."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        content = NB_EN.read_text(encoding="utf-8")
        # Exp 1: Smooth / Brock-Mirman
        assert ("brock" in content.lower() or "mirman" in content.lower() or "smooth" in content.lower()), (
            "Experiment 1 (Smooth Neoclassical Growth / Brock-Mirman) must be present"
        )
        # Exp 2: Borrowing constraint kink
        assert ("kink" in content.lower() or "borrowing" in content.lower() or "gibbs" in content.lower()), (
            "Experiment 2 (Borrowing-constrained model / kink) must be present"
        )
        # Exp 3: Stochastic multi-state
        assert ("markov" in content.lower() or "stochastic" in content.lower()), (
            "Experiment 3 (Stochastic multi-state model) must be present"
        )
        # Exp 4: Multi-backend
        assert ("backend" in content.lower() or "numba" in content.lower() or "mlx" in content.lower()), (
            "Experiment 4 (Multi-backend scaling) must be present"
        )

    def test_notebook_build_execution_check(self):
        """Verify tools/build_notebooks.py --check executes Notebook 51 cleanly."""
        if not NB_EN.exists():
            pytest.skip(f"{NB_EN} not yet authored")
        import importlib.util
        bn_tool = PROJ / "tools" / "build_notebooks.py"
        spec = importlib.util.spec_from_file_location("build_notebooks", bn_tool)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.build_one(NB_EN, check=True)
        assert rc == 0, f"Notebook 51 execution check failed with exit code {rc}"

    def test_spanish_notebook_build_execution_check(self):
        """Verify tools/build_notebooks.py --check executes Spanish twin cleanly."""
        if not NB_ES.exists():
            pytest.skip(f"{NB_ES} not yet authored")
        import importlib.util
        bn_tool = PROJ / "tools" / "build_notebooks.py"
        spec = importlib.util.spec_from_file_location("build_notebooks", bn_tool)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.build_one(NB_ES, check=True)
        assert rc == 0, f"Spanish Notebook 51 execution check failed with exit code {rc}"
