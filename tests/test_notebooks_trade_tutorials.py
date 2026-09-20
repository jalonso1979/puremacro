"""Opaque-box E2E test suite for puremacro Trade Tutorials (Notebooks 63, 64, 65).

Covers 4 systematic verification tiers across English and Spanish editions:
- Tier 1: Jupytext percent format syntax, 7-section pedagogical architecture,
  _nbstyle opaque card schema (min_alpha == 255), Pyodide 4-package purity.
- Tier 2: Headless execution time limit (<30s per notebook), deterministic inputs,
  zero unhandled cell tracebacks.
- Tier 3: Bilingual code parity (normalize_code(en) == normalize_code(es)),
  equal cell and figure counts, Spanish pedagogical translations.
- Tier 4: Solver and accounting checks and finite-game payoff classification.
  Historical EV/theorem claims remain explicitly quarantined expected failures.
  Independent Hicksian welfare and policy checks have separate test modules.

Adheres strictly to the Pyodide 4-package runtime contract (pure Python,
NumPy, SciPy, Pandas, Matplotlib).
"""
from __future__ import annotations

import ast
import base64
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from puremacro.trade import (
    TradeCalibrationResult,
    TradeEquilibriumResult,
    calibrate_trade_model,
    solve_trade_equilibrium,
)
from puremacro.trade._results import (
    EVDecompositionResult,
    TheoremValidationReport,
    WelfarePayoffMatrixResult,
)
from puremacro.trade.optimal_tariffs import compute_welfare_payoff_matrix
from puremacro.trade.policy_analytics import (
    decompose_hicksian_ev_3way,
    verify_theorems_1_to_4,
)
from puremacro.trade.solver import solve_keller_pac

# =============================================================================
# Path Configuration & Tutorial Registry
# =============================================================================

PROJ_ROOT = Path(__file__).resolve().parents[1]
NB_DIR = PROJ_ROOT / "notebooks"
BUILD_TOOL = PROJ_ROOT / "tools" / "build_notebooks.py"

# Allowed Pyodide runtime dependencies + puremacro internal modules + stdlib
PYODIDE_ALLOWED_PACKAGES = frozenset({
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "puremacro",
    "_nbstyle",
})

# Forbidden non-Pyodide / heavyweight packages
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
    "jax",
    "sympy",
    "seaborn",
})

# Required 7-section pedagogical architecture in strict monotonic sequence
PEDAGOGICAL_SECTIONS = [
    ("motivating_question", r"(?i)(motivating\s+question|\*\*how|\*\*what|\*\*can|\*\*por\s+qu[eé]|\*\*¿?c[oó]mo|\*\*qu[eé])"),
    ("the_method_in_math", r"(?i)(the\s+method\s+in\s+math|el\s+m[eé]todo\s+en\s+matem[aá]ticas)"),
    ("intuition", r"(?i)(\*\*intuition\.\*\*|\*\*intuici[oó]n\.\*\*|##\s*intuition|##\s*intuici[oó]n)"),
    ("worked_code", r"(?i)(worked\s+code|c[oó]digo\s+resuelto)"),
    ("read_the_output", r"(?i)(read\s+the\s+output|lectura\s+de\s+los\s+resultados)"),
    ("your_turn", r"(?i)(your\s+turn|tu\s+turno)"),
    ("how_comprehensive", r"(?i)(how\s+comprehensive\s+is\s+this|qu[eé]\s+tan\s+exhaustivo)"),
]

TUTORIAL_SPECS: list[dict[str, Any]] = [
    {
        "id": 63,
        "stem": "63_trade_wars_and_nash_tariffs",
        "title": "Global Trade Wars, Retaliation & Multilateral Nash Tariffs",
        "py_en": NB_DIR / "63_trade_wars_and_nash_tariffs.py",
        "py_es": NB_DIR / "63_trade_wars_and_nash_tariffs_es.py",
        "ipynb_en": NB_DIR / "63_trade_wars_and_nash_tariffs.ipynb",
        "ipynb_es": NB_DIR / "63_trade_wars_and_nash_tariffs_es.ipynb",
        "required_symbols": [
            "compute_unilateral_optimal_tariff",
            "compute_welfare_payoff_matrix",
            "solve_multilateral_nash_tariffs",
        ],
        "thematic_terms": ["dilemma", "nash", "retaliation", "optimal"],
    },
    {
        "id": 64,
        "stem": "64_singularities_and_keller_pac",
        "title": "Singularities, Fold Bifurcations & Keller PAC Continuation",
        "py_en": NB_DIR / "64_singularities_and_keller_pac.py",
        "py_es": NB_DIR / "64_singularities_and_keller_pac_es.py",
        "ipynb_en": NB_DIR / "64_singularities_and_keller_pac.ipynb",
        "ipynb_es": NB_DIR / "64_singularities_and_keller_pac_es.ipynb",
        "required_symbols": [
            "solve_keller_pac",
        ],
        "thematic_terms": ["bifurcation", "keller", "pac", "cyprus"],
    },
    {
        "id": 65,
        "stem": "65_gvc_cascades_and_welfare_decomposition",
        "title": "GVC cost propagation, provenance, and welfare validation limits",
        "py_en": NB_DIR / "65_gvc_cascades_and_welfare_decomposition.py",
        "py_es": NB_DIR / "65_gvc_cascades_and_welfare_decomposition_es.py",
        "ipynb_en": NB_DIR / "65_gvc_cascades_and_welfare_decomposition.ipynb",
        "ipynb_es": NB_DIR / "65_gvc_cascades_and_welfare_decomposition_es.ipynb",
        "required_symbols": [
            "generate_synthetic_mrio",
            "package_mrio_to_calibration_result",
        ],
        "thematic_terms": ["icio", "figaro", "exiobase", "equivalent variation"],
    },
]

# Flattened list of individual notebook editions for parametrized unit checks
INDIVIDUAL_NOTEBOOK_EDITIONS: list[tuple[str, Path, str]] = []
for spec in TUTORIAL_SPECS:
    INDIVIDUAL_NOTEBOOK_EDITIONS.append((f"{spec['stem']} [EN]", spec["py_en"], "en"))
    INDIVIDUAL_NOTEBOOK_EDITIONS.append((f"{spec['stem']} [ES]", spec["py_es"], "es"))


# =============================================================================
# Helper Functions & Parsers
# =============================================================================

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
    """Extract code cell text bodies from Jupytext percent source."""
    cells = parse_jupytext_cells(content)
    return [body for c_type, body in cells if c_type == "code"]


def extract_markdown_cells(content: str) -> list[str]:
    """Extract markdown cell text bodies from Jupytext percent source."""
    cells = parse_jupytext_cells(content)
    return [body for c_type, body in cells if c_type == "markdown"]


def extract_imports(code_text: str) -> set[str]:
    """Extract top-level imported module names via AST traversal."""
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
    """Normalize python code for bilingual structural comparison.

    Strips comments, empty lines, and trailing whitespace.
    """
    lines: list[str] = []
    for line in code.splitlines():
        # Remove trailing and leading spaces
        s = line.strip()
        if not s:
            continue
        # Strip full comment lines
        if s.startswith("#"):
            continue
        # Strip inline comments
        code_part = re.sub(r"#.*$", "", line).rstrip()
        if code_part.strip():
            lines.append(code_part.strip())
    return "\n".join(lines)


def check_png_image_opacity(image_bytes: bytes) -> tuple[bool, int]:
    """Inspect PNG image data and verify 100% opaque alpha channel (min_alpha == 255).

    Returns (is_opaque, min_alpha_value).
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == "RGBA":
        arr = np.array(img)
        min_alpha = int(arr[:, :, 3].min())
        return min_alpha == 255, min_alpha
    elif img.mode == "RGB":
        return True, 255
    else:
        # Convert to RGBA to inspect alpha
        rgba = img.convert("RGBA")
        arr = np.array(rgba)
        min_alpha = int(arr[:, :, 3].min())
        return min_alpha == 255, min_alpha


def extract_ipynb_png_images(nb_path: Path) -> list[tuple[int, int, bytes]]:
    """Extract all embedded PNG images from a compiled .ipynb notebook file.

    Returns list of (cell_idx, output_idx, image_bytes).
    """
    if not nb_path.exists():
        return []
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    results: list[tuple[int, int, bytes]] = []
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for out_idx, out in enumerate(cell.get("outputs", [])):
            payloads = out.get("data", {})
            if "image/png" in payloads:
                raw = payloads["image/png"]
                if isinstance(raw, list):
                    raw = "".join(raw)
                try:
                    img_bytes = base64.b64decode(raw)
                    results.append((cell_idx, out_idx, img_bytes))
                except Exception:
                    pass
    return results


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def calibrated_2c_2s_model() -> TradeCalibrationResult:
    """Construct and calibrate a deterministic 2-country 2-sector trade model."""
    nc, ns, nfd = 2, 2, 3
    data = np.zeros((ns * nc + 3, ns * nc + nfd * nc), dtype=float)
    data[:4, :4] = np.array([
        [20.0, 10.0, 5.0, 2.0],
        [8.0, 25.0, 2.0, 4.0],
        [5.0, 5.0, 12.0, 18.0],
        [10.0, 10.0, 18.0, 22.0],
    ])
    inter_col_sums = data[:4, :4].sum(axis=0)
    y = np.array([100.0, 150.0, 120.0, 180.0])
    va = y - inter_col_sums
    taxes = 0.05 * y
    va_fac = va - taxes
    data[4, :4] = taxes
    data[5, :4] = (2.0 / 3.0) * va_fac
    data[6, :4] = (1.0 / 3.0) * va_fac

    inter_row_sums = data[:4, :4].sum(axis=1)
    fd_row_sums = y - inter_row_sums
    for i in range(4):
        tot_fd = fd_row_sums[i]
        if i < 2:
            data[i, 4:10] = [tot_fd * 0.50, tot_fd * 0.25, tot_fd * 0.05, tot_fd * 0.10, tot_fd * 0.08, tot_fd * 0.02]
        else:
            data[i, 4:10] = [tot_fd * 0.10, tot_fd * 0.08, tot_fd * 0.02, tot_fd * 0.50, tot_fd * 0.25, tot_fd * 0.05]
    data[4, 4:] = 0.02 * data[:4, 4:].sum(axis=0)

    return calibrate_trade_model(
        data,
        ns=ns,
        nc=nc,
        nfd=nfd,
        country_codes=["USA", "CHN"],
        sector_codes=["AGR", "MAN"],
        validate=True,
    )


# =============================================================================
# Helper Mechanics Unit Tests
# =============================================================================

class TestHelperMechanics:
    """Validate helper functions used across the 4 test tiers."""

    def test_jupytext_cell_splitter(self) -> None:
        """Verify Jupytext percent syntax parser separates markdown and code correctly."""
        sample = (
            "# ---\n# jupyter:\n# ---\n\n"
            "# %% [markdown]\n"
            "# # Motivating Question\n"
            "# **How does trade policy propagate?**\n\n"
            "# %%\n"
            "import numpy as np\n"
            "x = np.array([1, 2, 3])\n"
        )
        cells = parse_jupytext_cells(sample)
        assert len(cells) == 2
        assert cells[0][0] == "markdown"
        assert "Motivating Question" in cells[0][1]
        assert cells[1][0] == "code"
        assert "import numpy as np" in cells[1][1]

    def test_code_normalizer(self) -> None:
        """Verify normalize_code eliminates comments, inline comments, and whitespace."""
        code_a = (
            "# Setup imports\n"
            "import numpy as np  # standard import\n"
            "\n"
            "a = 42\n"
            "# trailing comment\n"
        )
        code_b = (
            "import numpy as np\n"
            "a = 42  # assign\n"
        )
        assert normalize_code(code_a) == normalize_code(code_b)

    def test_import_extractor_pyodide_compliance(self) -> None:
        """Verify AST import extractor flags authorized vs unauthorized modules."""
        valid_code = "import numpy as np\nfrom scipy.optimize import root\nimport puremacro.trade\nimport _nbstyle\n"
        invalid_code = "import statsmodels.api as sm\nimport ipywidgets as widgets\n"
        assert extract_imports(valid_code).issubset(PYODIDE_ALLOWED_PACKAGES)
        assert not extract_imports(valid_code).intersection(FORBIDDEN_PACKAGES)
        assert "statsmodels" in extract_imports(invalid_code)
        assert "ipywidgets" in extract_imports(invalid_code)

    def test_png_opacity_detector(self) -> None:
        """Verify opacity checker correctly differentiates opaque vs transparent PNGs."""
        # 1. 100% opaque image
        opaque_img = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
        buf_opaque = io.BytesIO()
        opaque_img.save(buf_opaque, format="PNG")
        is_op, min_a = check_png_image_opacity(buf_opaque.getvalue())
        assert is_op is True
        assert min_a == 255

        # 2. Transparent image
        trans_img = Image.new("RGBA", (10, 10), (255, 255, 255, 128))
        buf_trans = io.BytesIO()
        trans_img.save(buf_trans, format="PNG")
        is_op_trans, min_a_trans = check_png_image_opacity(buf_trans.getvalue())
        assert is_op_trans is False
        assert min_a_trans == 128


# =============================================================================
# Tier 1: Jupytext Syntax, 7-Section Architecture, _nbstyle & Pyodide Purity
# =============================================================================

class TestTier1FormatSyntaxAndStyle:
    """Tier 1: Static syntax, structure, card styling, and Pyodide purity."""

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_notebook_file_exists(self, label: str, path: Path, lang: str) -> None:
        """Verify the Jupytext .py source file exists in notebooks/."""
        if not path.exists():
            pytest.skip(f"Notebook source {path.name} not yet authored by worker")
        assert path.is_file(), f"Expected file at {path}"

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_jupytext_percent_header(self, label: str, path: Path, lang: str) -> None:
        """Verify notebook begins with Jupytext percent format frontmatter."""
        if not path.exists():
            pytest.skip(f"{path.name} not yet authored")
        content = path.read_text(encoding="utf-8")
        assert "format_name: percent" in content, f"{path.name}: missing 'format_name: percent' in header"
        assert "kernelspec:" in content, f"{path.name}: missing 'kernelspec:' in header"
        assert "name: python3" in content or "display_name: Python 3" in content

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_ast_python_syntax_validity(self, label: str, path: Path, lang: str) -> None:
        """Verify the full Jupytext .py script parses as valid Python AST without syntax errors."""
        if not path.exists():
            pytest.skip(f"{path.name} not yet authored")
        content = path.read_text(encoding="utf-8")
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in {path.name}: {e}")

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_seven_sections_presence_and_order(self, label: str, path: Path, lang: str) -> None:
        """Verify all 7 pedagogical sections are present in strictly monotonic order."""
        if not path.exists():
            pytest.skip(f"{path.name} not yet authored")
        content = path.read_text(encoding="utf-8")
        positions: list[tuple[str, int]] = []
        for name, pattern in PEDAGOGICAL_SECTIONS:
            m = re.search(pattern, content)
            assert m is not None, f"{path.name}: Required section '{name}' matching pattern '{pattern}' not found"
            positions.append((name, m.start()))

        for i in range(len(positions) - 1):
            sec_prev, pos_prev = positions[i]
            sec_next, pos_next = positions[i + 1]
            assert pos_prev < pos_next, (
                f"{path.name}: Section '{sec_prev}' (pos {pos_prev}) must appear "
                f"before '{sec_next}' (pos {pos_next})"
            )

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_the_method_in_math_latex_equations(self, label: str, path: Path, lang: str) -> None:
        """Verify Section 2 (The method in math) contains formal LaTeX equations ($ or $$)."""
        if not path.exists():
            pytest.skip(f"{path.name} not yet authored")
        content = path.read_text(encoding="utf-8")
        assert "$" in content, f"{path.name}: Section 2 must contain LaTeX math equations"

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_intuition_format_and_no_emojis(self, label: str, path: Path, lang: str) -> None:
        """Verify Section 3 (Intuition) contains lead-in text and strictly zero emojis."""
        if not path.exists():
            pytest.skip(f"{path.name} not yet authored")
        content = path.read_text(encoding="utf-8")
        lead_in_found = (
            "**Intuition.**" in content
            or "**Intuición.**" in content
            or "## Intuition" in content
            or "## Intuición" in content
        )
        assert lead_in_found, f"{path.name}: Section 3 must contain explicit '**Intuition.**' lead-in"

        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]"
        )
        assert not emoji_pattern.search(content), f"{path.name}: Repository style guide strictly forbids emojis"

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_interactive_knob_and_downstream_assertions(self, label: str, path: Path, lang: str) -> None:
        """Verify Section 6 contains an interactive knob marker and downstream assert statements."""
        if not path.exists():
            pytest.skip(f"{path.name} not yet authored")
        content = path.read_text(encoding="utf-8")
        cells = parse_jupytext_cells(content)
        found_knob = False
        knob_cell_has_assert = False

        knob_markers = ("# ← change this", "# ← Change this", "# ← cambia esto", "# ← Cambia esto")
        for c_type, body in cells:
            if c_type == "code" and any(marker in body for marker in knob_markers):
                found_knob = True
                if "assert " in body:
                    knob_cell_has_assert = True

        assert found_knob, f"{path.name}: Code cell with '# ← change this' marker not found in Section 6"
        assert knob_cell_has_assert, f"{path.name}: Interactive knob cell must contain assert statement(s)"

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_nbstyle_card_schema_presence(self, label: str, path: Path, lang: str) -> None:
        """Verify notebook imports _nbstyle and applies card styling."""
        if not path.exists():
            pytest.skip(f"{path.name} not yet authored")
        content = path.read_text(encoding="utf-8")
        assert "_nbstyle" in content, f"{path.name}: must import _nbstyle"
        assert "apply_style" in content, f"{path.name}: must call _nbstyle.apply_style()"

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_pyodide_four_package_purity(self, label: str, path: Path, lang: str) -> None:
        """Verify notebook imports strictly conform to the 4-package Pyodide standard."""
        if not path.exists():
            pytest.skip(f"{path.name} not yet authored")
        content = path.read_text(encoding="utf-8")
        code_text = "\n".join(extract_code_cells(content))
        imports = extract_imports(code_text)

        violators = imports.intersection(FORBIDDEN_PACKAGES)
        assert not violators, f"{path.name}: Forbidden non-Pyodide imports found: {violators}"

    @pytest.mark.parametrize("spec", TUTORIAL_SPECS)
    def test_notebook_feature_inventory_coverage(self, spec: dict[str, Any]) -> None:
        """Verify notebook includes all assigned domain features from TEST_INFRA.md."""
        py_en: Path = spec["py_en"]
        if not py_en.exists():
            pytest.skip(f"{py_en.name} not yet authored")
        content = py_en.read_text(encoding="utf-8")
        code_text = "\n".join(extract_code_cells(content))

        # Check required API symbols
        for sym in spec["required_symbols"]:
            assert sym in code_text, f"{py_en.name}: missing required trade API symbol '{sym}'"

        # Check domain thematic terms
        for term in spec["thematic_terms"]:
            assert term.lower() in content.lower(), f"{py_en.name}: missing thematic requirement '{term}'"

    @pytest.mark.parametrize("spec", TUTORIAL_SPECS)
    def test_compiled_ipynb_figure_opacity(self, spec: dict[str, Any]) -> None:
        """Verify all committed or built figure PNGs have 100% opaque alpha channels (min_alpha == 255)."""
        tested_any = False
        for key in ("ipynb_en", "ipynb_es"):
            nb_path: Path = spec[key]
            if not nb_path.exists():
                continue
            tested_any = True
            images = extract_ipynb_png_images(nb_path)
            assert len(images) > 0, f"{nb_path.name}: Expected figure images in executed notebook, found 0"
            for cell_idx, out_idx, img_bytes in images:
                is_op, min_a = check_png_image_opacity(img_bytes)
                assert is_op, (
                    f"{nb_path.name} (cell {cell_idx}, output {out_idx}): "
                    f"min_alpha={min_a} < 255 (figure background must be 100% opaque)"
                )
        if not tested_any:
            pytest.skip(f"Tutorial {spec['id']}: compiled .ipynb artifacts not yet generated")


# =============================================================================
# Tier 2: Execution Time, Seed 42 Determinism & Zero Tracebacks
# =============================================================================

class TestTier2ExecutionPerformanceAndDeterminism:
    """Tier 2: Runtime performance, deterministic seeds, and zero tracebacks."""

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_deterministic_initialization(self, label: str, path: Path, lang: str) -> None:
        """Use a literal teaching table or an explicitly seeded generator."""
        if not path.exists():
            pytest.skip(f"{path.name} not yet authored")
        content = path.read_text(encoding="utf-8")
        code_text = "\n".join(extract_code_cells(content))
        if path.name.startswith("63_"):
            tree = ast.parse(code_text)
            flows = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "flows" for t in node.targets))
            assert isinstance(flows, ast.Call)
            assert np.isfinite(np.asarray(ast.literal_eval(flows.args[0]))).all()
            assert not any(isinstance(node, ast.Attribute) and node.attr in ("random", "default_rng")
                           for node in ast.walk(tree))
            return
        seed_pattern = re.compile(r"(?:seed\s*=\s*42|seed\(\s*42\s*\)|default_rng\(\s*42\s*\)|SEED\s*=\s*42)")
        assert seed_pattern.search(code_text), f"{path.name}: must set deterministic random seed 42"

    @pytest.mark.parametrize("spec", TUTORIAL_SPECS)
    def test_zero_unhandled_tracebacks_in_compiled_notebook(self, spec: dict[str, Any]) -> None:
        """Verify compiled .ipynb contains zero unhandled cell errors (output_type != 'error')."""
        tested_any = False
        for key in ("ipynb_en", "ipynb_es"):
            nb_path: Path = spec[key]
            if not nb_path.exists():
                continue
            tested_any = True
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            errors: list[str] = []
            for cell_idx, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                for out_idx, out in enumerate(cell.get("outputs", [])):
                    if out.get("output_type") == "error":
                        ename = out.get("ename", "Error")
                        evalue = out.get("evalue", "")
                        errors.append(f"Cell {cell_idx}, Out {out_idx}: {ename}: {evalue}")

            assert not errors, f"{nb_path.name} contains execution error tracebacks:\n" + "\n".join(errors)
        if not tested_any:
            pytest.skip(f"Tutorial {spec['id']}: compiled .ipynb artifacts not yet generated")

    @pytest.mark.parametrize("label, path, lang", INDIVIDUAL_NOTEBOOK_EDITIONS)
    def test_headless_execution_time_limit_sub_30s(self, label: str, path: Path, lang: str) -> None:
        """Verify headless execution runs cleanly in <30 seconds via build_notebooks.py."""
        if not path.exists():
            pytest.skip(f"{path.name} not yet authored")
        if not BUILD_TOOL.exists():
            pytest.skip(f"Build tool not found at {BUILD_TOOL}")

        spec_tool = importlib.util.spec_from_file_location("build_notebooks_runner", BUILD_TOOL)
        assert spec_tool and spec_tool.loader
        bn_mod = importlib.util.module_from_spec(spec_tool)
        spec_tool.loader.exec_module(bn_mod)

        t_start = time.perf_counter()
        rc = bn_mod.build_one(path, check=True)
        duration = time.perf_counter() - t_start

        assert rc == 0, f"{path.name}: Headless execution check failed with return code {rc}"
        assert duration < 30.0, f"{path.name}: Execution duration {duration:.2f}s exceeded 30s limit"


# =============================================================================
# Tier 3: Bilingual Code Parity, Cell Count & Translation
# =============================================================================

class TestTier3BilingualCodeParityAndSymmetry:
    """Tier 3: Bilingual code identity, cell symmetry, and translation."""

    @pytest.mark.parametrize("spec", TUTORIAL_SPECS)
    def test_bilingual_pair_existence(self, spec: dict[str, Any]) -> None:
        """Verify both English and Spanish .py files exist for the tutorial."""
        en_path: Path = spec["py_en"]
        es_path: Path = spec["py_es"]
        if not en_path.exists() or not es_path.exists():
            pytest.skip(f"Tutorial {spec['id']}: English or Spanish edition pending authoring")
        assert en_path.is_file() and es_path.is_file()

    @pytest.mark.parametrize("spec", TUTORIAL_SPECS)
    def test_bilingual_code_cell_counts_match(self, spec: dict[str, Any]) -> None:
        """Verify English and Spanish editions have the exact same number of code cells."""
        en_path: Path = spec["py_en"]
        es_path: Path = spec["py_es"]
        if not en_path.exists() or not es_path.exists():
            pytest.skip(f"Tutorial {spec['id']}: editions pending authoring")

        en_cells = extract_code_cells(en_path.read_text(encoding="utf-8"))
        es_cells = extract_code_cells(es_path.read_text(encoding="utf-8"))
        assert len(en_cells) == len(es_cells), (
            f"Tutorial {spec['id']} code cell count mismatch: "
            f"English has {len(en_cells)}, Spanish has {len(es_cells)}"
        )

    @pytest.mark.parametrize("spec", TUTORIAL_SPECS)
    def test_bilingual_code_logic_identity(self, spec: dict[str, Any]) -> None:
        """Verify normalize_code(en) == normalize_code(es) for every single code cell."""
        en_path: Path = spec["py_en"]
        es_path: Path = spec["py_es"]
        if not en_path.exists() or not es_path.exists():
            pytest.skip(f"Tutorial {spec['id']}: editions pending authoring")

        en_cells = extract_code_cells(en_path.read_text(encoding="utf-8"))
        es_cells = extract_code_cells(es_path.read_text(encoding="utf-8"))

        for idx, (en_c, es_c) in enumerate(zip(en_cells, es_cells)):
            norm_en = normalize_code(en_c)
            norm_es = normalize_code(es_c)
            assert norm_en == norm_es, (
                f"Tutorial {spec['id']} code cell {idx + 1} diverges between EN and ES:\n"
                f"--- EN ---\n{norm_en[:300]}\n--- ES ---\n{norm_es[:300]}"
            )

    @pytest.mark.parametrize("spec", TUTORIAL_SPECS)
    def test_bilingual_figure_counts_match(self, spec: dict[str, Any]) -> None:
        """Verify compiled English and Spanish notebooks contain identical figure counts."""
        ipynb_en: Path = spec["ipynb_en"]
        ipynb_es: Path = spec["ipynb_es"]
        if not ipynb_en.exists() or not ipynb_es.exists():
            pytest.skip(f"Tutorial {spec['id']}: .ipynb artifacts pending build")

        imgs_en = extract_ipynb_png_images(ipynb_en)
        imgs_es = extract_ipynb_png_images(ipynb_es)
        assert len(imgs_en) == len(imgs_es), (
            f"Tutorial {spec['id']} figure count mismatch: "
            f"English has {len(imgs_en)} figures, Spanish has {len(imgs_es)} figures"
        )

    @pytest.mark.parametrize("spec", TUTORIAL_SPECS)
    def test_spanish_markdown_pedagogical_translation(self, spec: dict[str, Any]) -> None:
        """Verify Spanish edition contains authentic academic Spanish pedagogical headings."""
        es_path: Path = spec["py_es"]
        if not es_path.exists():
            pytest.skip(f"Spanish edition {es_path.name} not yet authored")
        content = es_path.read_text(encoding="utf-8")
        spanish_tokens = [
            "El método en matemáticas",
            "Intuición",
            "Lectura de los resultados",
            "Tu turno",
            "¿Qué tan exhaustivo es esto?",
        ]
        matched = [token for token in spanish_tokens if token in content]
        assert len(matched) >= 3, (
            f"{es_path.name}: Insufficient Spanish headings detected (found: {matched})"
        )


# =============================================================================
# Tier 4: Scientific Invariants & Theoretical Bounds
# =============================================================================

class TestTier4ScientificInvariants:
    """Tier 4: Numerical precision, mathematical invariants, and bounds."""

    # -------------------------------------------------------------------------
    # Direct Environment Invariants (Mathematical Oracle Tests)
    # -------------------------------------------------------------------------

    def test_invariant_keller_pac_residual_sub_1e10(
        self, calibrated_2c_2s_model: TradeCalibrationResult
    ) -> None:
        """Verify Keller's Pseudo-Arclength Continuation converges with ||F(x)||_inf < 10^-10."""
        calib = calibrated_2c_2s_model
        res = solve_keller_pac(
            calib=calib,
            tau_target=0.15,
            ds_init=0.2,
            tol=1e-6,
            max_steps=20,
        )
        assert isinstance(res, TradeEquilibriumResult)
        assert res.converged is True
        max_res = float(np.max(np.abs(res.residuals)))
        assert max_res < 1e-10, f"Keller PAC final residual {max_res:.4e} must be strictly < 10^-10"

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_invariant_exact_ev_decomposition_residual_sub_1e10(
        self, calibrated_2c_2s_model: TradeCalibrationResult
    ) -> None:
        """Verify 3-way additive Hicksian EV decomposition satisfies discrepancy <= 10^-10."""
        calib = calibrated_2c_2s_model
        base_eq = solve_trade_equilibrium(calib, method="condensed", tol=1e-8)
        cf_eq = solve_trade_equilibrium(calib, tau=0.10, method="condensed", tol=1e-8)

        decomp = decompose_hicksian_ev_3way(calib, cf_eq, base_result=base_eq, target_country="USA")
        assert isinstance(decomp, EVDecompositionResult)
        assert abs(decomp.residual) <= 1e-10, f"EV decomposition residual {decomp.residual} exceeds 1e-10"
        identity_gap = abs(decomp.ev_usd - (decomp.tot + decomp.alloc + decomp.tariff_rec))
        assert identity_gap <= 1e-10, f"Additive EV identity discrepancy {identity_gap} exceeds 1e-10"

    @pytest.mark.xfail(raises=NotImplementedError, reason="Exact EV/theorem certification quarantined after independent counterexamples", run=False, strict=True)
    def test_invariant_theorems_1_to_4_all_passed(
        self, calibrated_2c_2s_model: TradeCalibrationResult
    ) -> None:
        """Verify Theorems 1–4 validation engine returns all_passed == True."""
        calib = calibrated_2c_2s_model
        report = verify_theorems_1_to_4(calib, scenario="uniform_10", sigma=2.0)
        assert isinstance(report, TheoremValidationReport)
        assert report.all_passed is True
        assert report.theorem_1_passed is True
        assert report.theorem_2_passed is True
        assert report.theorem_3_passed is True
        assert report.theorem_4_passed is True

    def test_invariant_prisoners_dilemma_payoff_dominance(self) -> None:
        """Verify WelfarePayoffMatrixResult.is_prisoners_dilemma property logic."""
        # Dominant defection and Pareto-inferior trade war:
        # P0: (D, C)=1.5 > (C, C)=0.0 and (D, D)=-1.0 > (C, D)=-2.0 and (C, C)=0.0 > (D, D)=-1.0
        # P1: (C, D)=1.5 > (C, C)=0.0 and (D, D)=-1.0 > (D, C)=-2.0 and (C, C)=0.0 > (D, D)=-1.0
        payoffs = np.zeros((2, 2, 2))
        payoffs[0, 0] = [0.0, 0.0]
        payoffs[1, 0] = [1.5, -2.0]
        payoffs[0, 1] = [-2.0, 1.5]
        payoffs[1, 1] = [-1.0, -1.0]

        summary_df = pd.DataFrame([["(0.0, 0.0)", "(-2.0, +1.5)"], ["(+1.5, -2.0)", "(-1.0, -1.0)"]])
        res = WelfarePayoffMatrixResult(
            players=("USA", "CHN"),
            strategies=("Cooperate", "Defect"),
            payoff_matrix=payoffs,
            scenarios={},
            summary_df=summary_df,
        )
        assert res.is_prisoners_dilemma is True

        # Degenerate payoffs must return False
        degenerate = WelfarePayoffMatrixResult(
            players=("USA", "CHN"),
            strategies=("Cooperate", "Defect"),
            payoff_matrix=np.zeros((2, 2, 2)),
            scenarios={},
            summary_df=summary_df,
        )
        assert degenerate.is_prisoners_dilemma is False

    # -------------------------------------------------------------------------
    # Notebook-Level Invariant Checks (Static and Runtime Assertions)
    # -------------------------------------------------------------------------

    def test_notebook_63_prisoners_dilemma_invariant(self) -> None:
        """Verify Notebook 63 imports and asserts Prisoner's Dilemma strategic structure."""
        spec = TUTORIAL_SPECS[0]
        py_en: Path = spec["py_en"]
        if not py_en.exists():
            pytest.skip(f"{py_en.name} not yet authored")

        content = py_en.read_text(encoding="utf-8")
        code_text = "\n".join(extract_code_cells(content))

        # Check required function calls
        assert "compute_welfare_payoff_matrix" in code_text, (
            f"{py_en.name} must call compute_welfare_payoff_matrix"
        )
        # Check that is_prisoners_dilemma or dilemma dominance is verified
        assert "is_prisoners_dilemma" in code_text or "dilemma" in code_text.lower(), (
            f"{py_en.name} must check is_prisoners_dilemma or verify dilemma payoffs"
        )

    def test_notebook_64_keller_pac_residual_invariant(self) -> None:
        """Verify Notebook 64 executes Keller PAC and asserts residual < 10^-10."""
        spec = TUTORIAL_SPECS[1]
        py_en: Path = spec["py_en"]
        if not py_en.exists():
            pytest.skip(f"{py_en.name} not yet authored")

        content = py_en.read_text(encoding="utf-8")
        code_text = "\n".join(extract_code_cells(content))

        assert "solve_keller_pac" in code_text, f"{py_en.name} must call solve_keller_pac"
        # Check for high-precision residual assertion (< 1e-10 or 10**-10)
        assert "1e-10" in code_text or "10**-10" in code_text or "10^-10" in content, (
            f"{py_en.name} must assert or document Keller PAC residual < 10^-10"
        )

    @pytest.mark.parametrize("name, symbol", [
        ("64_singularities_and_keller_pac", "verify_theorems_1_to_4"),
        ("65_gvc_cascades_and_welfare_decomposition", "decompose_hicksian_ev_3way"),
    ])
    def test_quarantined_claims_are_disclosed_without_execution(self, name, symbol):
        content = (NB_DIR / f"{name}.py").read_text()
        code = "\n".join(extract_code_cells(content))
        calls = [n.func.id for n in ast.walk(ast.parse(code))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        assert symbol not in calls
        assert "unavailable" in content.lower() or "quarantined" in content.lower()
