"""End-to-End Test Suite for Documentation Elevation across puremacro Notebooks.

Authoritative E2E verification suite validating the transformation of notebook markdown
documentation into publication-grade scientific treatises, strictly derived from
ORIGINAL_REQUEST.md and notebooks/_TEMPLATE.md.

Tiers Covered:
- Tier 1: Feature Coverage (motivating economic question, displayed LaTeX $$...$$,
  parameter definition tables, formal **Intuition.** blocks, seminal literature citations,
  **Read the output.** narrative interpretations, Your turn exercises with `# ← change this`).
- Tier 2: Boundary & Corner Cases (empty/degenerate markdown cells, LaTeX delimiter
  balance $/ $$, escaped currency \\$, strict emoji ban, canvas opacity min_alpha == 255,
  valid nbformat v4 schema).
- Tier 3: Cross-Feature Interactions (exact bilingual code logic identity
  normalize_code(en) == normalize_code(es), section sequence monotonicity, formula count parity).
- Tier 4: Real-World Acceptance Scenarios (target showcase cohort compliance, course suite
  pedagogical tag compliance, zero runtime tracebacks, valid execution counts > 0).
"""
from __future__ import annotations

import ast
import base64
import io
import json
import re
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pytest
from PIL import Image

# =============================================================================
# Path Configuration & Registry
# =============================================================================

PROJ_ROOT = Path(__file__).resolve().parents[2]
NB_DIR = PROJ_ROOT / "notebooks"
COURSE_DIR = NB_DIR / "course"
CURSO_DIR = PROJ_ROOT / "curso" / "notebooks"

# Conforming showcase exemplars adhering to the 7-section publication standard
SHOWCASE_CONFORMING_EXEMPLARS = [
    "00_whats_new_in_puremacro_3_0",
    "01_wealth_inequality",
    "02_aggregate_shocks",
    "03_life_cycle_and_demographics",
    "04_firm_dynamics",
    "05_portfolios_and_preferences",
    "06_svar_identification",
    "07_local_projections",
    "08_garch_volatility",
    "09_growth_at_risk",
    "10_staggered_did",
    "11_narrative_uncertainty",
    "12_validation_gallery",
    "13_build_your_own_index",
    "14_tax_multiplier_three_ways",
    "15_lp_did",
    "16_regime_girf",
    "17_identification_spec_curve",
    "18_beveridge_curve",
    "51_continuous_projection_collocation_and_fem",
    "52_continuous_transition_mit_shocks",
    "53_exact_analytic_ift_gradients",
    "54_deep_macro_pinns_high_dim",
    "55_quantitative_spatial_and_trade_ge",
    "56_implicit_hjb_and_continuous_kfe",
    "57_multiconstraint_occbin_and_dml",
    "58_latin_america_realtime_macro",
]

# Exemplars featuring seminal literature citations
SHOWCASE_CITED_EXEMPLARS = [
    "01_wealth_inequality",
    "02_aggregate_shocks",
    "03_life_cycle_and_demographics",
    "04_firm_dynamics",
    "05_portfolios_and_preferences",
    "51_continuous_projection_collocation_and_fem",
    "52_continuous_transition_mit_shocks",
    "55_quantitative_spatial_and_trade_ge",
    "58_latin_america_realtime_macro",
]

# Target 15 showcase cohort from Milestone 4 verification gate
TARGET_15_NOTEBOOKS = [
    "00_whats_new_in_puremacro_2_0.ipynb",
    "00_whats_new_in_puremacro_2_0_es.ipynb",
    "00_whats_new_in_puremacro_3_0.ipynb",
    "00_whats_new_in_puremacro_3_0_es.ipynb",
    "31_sequence_space_hank.ipynb",
    "32_climate_macro_dice.ipynb",
    "33_gdp_nowcasting_news.ipynb",
    "34_penalized_macro_forecasting.ipynb",
    "36_climate_sovereign_debt_risk_es.ipynb",
    "37_central_bank_narrative_sentiment.ipynb",
    "37_central_bank_narrative_sentiment_es.ipynb",
    "43_dsge_nuts_and_analytic_gradients.ipynb",
    "43_dsge_nuts_and_analytic_gradients_es.ipynb",
    "62_flexible_trade_cge.ipynb",
    "62_flexible_trade_cge_es.ipynb",
]

# Elevated parameter table cohort (progressively populated as M1–M4 authoring proceeds)
ELEVATED_PARAMETER_TABLE_COHORT: list[str] = [
    "00_whats_new_in_puremacro_3_0",
    "01_wealth_inequality",
    "02_aggregate_shocks",
    "03_life_cycle_and_demographics",
    "04_firm_dynamics",
    "05_portfolios_and_preferences",
    "06_svar_identification",
    "07_local_projections",
    "08_garch_volatility",
    "09_growth_at_risk",
    "10_staggered_did",
    "11_narrative_uncertainty",
    "12_validation_gallery",
    "13_build_your_own_index",
    "14_tax_multiplier_three_ways",
    "15_lp_did",
    "16_regime_girf",
    "17_identification_spec_curve",
    "18_beveridge_curve",
]

# Bilingual exemplar pairs
BILINGUAL_EXEMPLAR_PAIRS = [
    ("51_continuous_projection_collocation_and_fem.py", "51_continuous_projection_collocation_and_fem_es.py"),
    ("52_continuous_transition_mit_shocks.py", "52_continuous_transition_mit_shocks_es.py"),
    ("53_exact_analytic_ift_gradients.py", "53_exact_analytic_ift_gradients_es.py"),
    ("54_deep_macro_pinns_high_dim.py", "54_deep_macro_pinns_high_dim_es.py"),
    ("55_quantitative_spatial_and_trade_ge.py", "55_quantitative_spatial_and_trade_ge_es.py"),
    ("56_implicit_hjb_and_continuous_kfe.py", "56_implicit_hjb_and_continuous_kfe_es.py"),
    ("57_multiconstraint_occbin_and_dml.py", "57_multiconstraint_occbin_and_dml_es.py"),
    ("58_latin_america_realtime_macro.py", "58_latin_america_realtime_macro_es.py"),
]

# Sequence of 6 required content sections across showcase scripts
REQUIRED_SECTIONS = [
    ("motivating_question", r"(?i)(motivating\s+question|\*\*¿?(?:how|what|can|why|where|c[oó]mo|qu[eé]|por\s+qu[eé]|pued\w*|(?:de\s+)?d[oó]nde)\b|^#\s+[^\n]+(?:\?|¿))"),
    ("the_method_in_math",  r"(?i)(the\s+method\s+in\s+math|the\s+model\s+in|el\s+m[eé]todo\s+en\s+matem[aá]ticas|el\s+modelo\s+en)"),
    ("intuition",           r"(?i)(\*\*intuition\.\*\*|\*\*intuici[oó]n\.\*\*|##\s*intuition\b|##\s*intuici[oó]n\b)"),
    ("read_the_output",     r"(?i)(\*\*read\s+the\s+output\.\*\*|\*\*lectura\s+de\s+los\s+resultados\.\*\*|##\s*read\s+the\s+output\b|##\s*lectura\s+de\s+los\s+resultados\b)"),
    ("your_turn",           r"(?i)(your\s+turn|tu\s+turno)"),
    ("how_comprehensive",   r"(?i)(how\s+comprehensive\s+is\s+this|qu[eé]\s+tan\s+exhaustivo)"),
]

# Course companion required markers
COURSE_REQUIRED_SECTIONS = {
    "objetivos": ("objetivo",),
    "ejercicios": ("ejercicio", "pregunta"),
    "exploración con IA": ("explora con ia", "tutor", "inteligencia artificial"),
}

# =============================================================================
# Helper Detection & Normalization Mechanics
# =============================================================================

def parse_jupytext_cells(content: str) -> list[tuple[str, str]]:
    """Parse Jupytext percent source into list of (cell_type, cell_text)."""
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
    """Extract code cell bodies from Jupytext percent source."""
    cells = parse_jupytext_cells(content)
    return [body for c_type, body in cells if c_type == "code"]


def extract_markdown_cells(content: str) -> list[str]:
    """Extract markdown cell bodies from Jupytext percent source."""
    cells = parse_jupytext_cells(content)
    return [body for c_type, body in cells if c_type == "markdown"]


def normalize_code(code: str) -> str:
    """Normalize python code for bilingual structural comparison (strips comments and whitespace)."""
    lines: list[str] = []
    for line in code.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        code_part = re.sub(r"#.*$", "", line).rstrip()
        if code_part.strip():
            lines.append(code_part.strip())
    return "\n".join(lines)


def detect_parameter_table_or_defs(markdown_text: str) -> tuple[bool, str]:
    """Detect whether markdown contains a parameter definition table or unit itemization."""
    table_pattern = re.compile(
        r"(?i)\|\s*(parameter|parámetro|symbol|símbolo|variable)\s*\|"
        r".*\|\s*(unit|unidades?|meaning|descrip|valor|value)",
        re.MULTILINE,
    )
    if table_pattern.search(markdown_text):
        return True, "markdown_table"

    if re.search(r"\\begin\{tabular\}", markdown_text) and re.search(r"(?i)(parameter|parámetro|units?|unidades?)", markdown_text):
        return True, "latex_tabular"

    item_pattern = re.compile(
        r"(?m)^[\s*-]+\$?[\\a-zA-Z_0-9]+\$?[\s:]+.*\[.*(unit|unid|dimensionless|adimensional|quarter|year|annual|mensual|anual).*\]",
        re.IGNORECASE,
    )
    if item_pattern.search(markdown_text):
        return True, "itemized_units"

    return False, "missing"


def check_latex_delimiter_balance(text: str) -> tuple[bool, str]:
    """Verify parity of $$ and unescaped $ delimiters, correctly respecting escaped \\$."""
    dd_count = text.count("$$")
    if dd_count % 2 != 0:
        return False, f"Unbalanced $$ delimiters (count={dd_count})"

    no_dd = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    single_dollars = re.findall(r"(?<!\\)\$", no_dd)
    if len(single_dollars) % 2 != 0:
        return False, f"Unbalanced unescaped $ delimiters (count={len(single_dollars)})"

    return True, "balanced"


def extract_latex_formulas(text: str) -> tuple[list[str], list[str]]:
    """Extract display formulas ($$...$$) and inline formulas ($...$)."""
    display_formulas = re.findall(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)
    clean_text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    inline_formulas = re.findall(r"(?<!\\)\$(.*?)(?<!\\)\$", clean_text, flags=re.DOTALL)
    return display_formulas, inline_formulas


def detect_literature_citations(text: str) -> list[str]:
    """Extract academic citations matching Author (YYYY), Author et al. (YYYY), Author & Author (YYYY), or (Author, YYYY)."""
    citation_pattern = re.compile(
        r"(?:\b[A-Z][a-zA-Z\u00C0-\u017F\-]+(?:\s+et\s+al\.|\s+(?:and|y|&)\s+[A-Z][a-zA-Z\u00C0-\u017F\-]+)?\s*\((?:19|20)\d{2}\))|"
        r"(?:\([A-Z][a-zA-Z\u00C0-\u017F\-]+(?:\s+et\s+al\.|\s+(?:and|y|&)\s+[A-Z][a-zA-Z\u00C0-\u017F\-]+)?,\s*(?:19|20)\d{2}\))"
    )
    return citation_pattern.findall(text)


def detect_emojis(text: str) -> list[str]:
    """Detect Unicode emojis strictly prohibited in academic treatises."""
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F"  # Emoticons
        r"\U0001F300-\U0001F5FF"  # Misc Symbols & Pictographs
        r"\U0001F680-\U0001F6FF"  # Transport & Map
        r"\U0001F700-\U0001F77F"  # Alchemical
        r"\U0001F900-\U0001F9FF"  # Supplemental Symbols
        r"\U0001FA70-\U0001FAFF]"  # Extended-A
    )
    return emoji_pattern.findall(text)


def check_png_image_opacity(image_bytes: bytes) -> tuple[bool, int]:
    """Verify PNG image has 100% opaque alpha channel (min_alpha == 255)."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode == "RGBA":
        arr = np.array(img)
        min_alpha = int(arr[:, :, 3].min())
        return min_alpha == 255, min_alpha
    elif img.mode == "RGB":
        return True, 255
    else:
        rgba = img.convert("RGBA")
        arr = np.array(rgba)
        min_alpha = int(arr[:, :, 3].min())
        return min_alpha == 255, min_alpha


def extract_ipynb_png_images(nb_path: Path) -> list[tuple[int, int, bytes]]:
    """Extract embedded PNG images from a compiled .ipynb file as (cell_idx, out_idx, bytes)."""
    if not nb_path.exists():
        return []
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    results = []
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


def get_all_repository_notebooks() -> list[Path]:
    """Discover all committed .ipynb notebook files across primary repository suites."""
    showcase = [
        p for p in NB_DIR.glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ]
    course_sub = [
        p for p in COURSE_DIR.glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ] if COURSE_DIR.exists() else []
    curso = [
        p for p in CURSO_DIR.glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ] if CURSO_DIR.exists() else []
    return sorted(showcase + course_sub + curso)


# =============================================================================
# Tier 1: Feature Coverage (Isolated Happy Path)
# =============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Feature Coverage for documentation elevation (F1–F8)."""

    def test_helper_jupytext_cell_splitter(self) -> None:
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

    def test_f1_motivating_question_detector_mechanics(self) -> None:
        """Verify motivating question regex identifies valid research questions."""
        pat = re.compile(REQUIRED_SECTIONS[0][1])
        valid_en = "**How do continuous household wealth distributions transition non-linearly?**"
        valid_es = "**¿Cómo transicionan de forma no lineal las distribuciones de riqueza?**"
        valid_title = "# Where does wealth inequality come from?\n"
        invalid = "This notebook runs some macro code."
        assert pat.search(valid_en) is not None
        assert pat.search(valid_es) is not None
        assert pat.search(valid_title) is not None
        assert pat.search(invalid) is None

    def test_f2_latex_math_detector_mechanics(self) -> None:
        """Verify formal display LaTeX ($$...$$) detection logic."""
        text_with_disp = "The Euler equation is:\n$$ u'(c_t) = \\beta (1+r_{t+1}) u'(c_{t+1}) $$\n"
        text_without_disp = "The Euler equation is u'(c) = beta*(1+r)*u'(c)."
        disp, inline = extract_latex_formulas(text_with_disp)
        assert len(disp) == 1
        assert "u'(c_t)" in disp[0]
        disp_empty, _ = extract_latex_formulas(text_without_disp)
        assert len(disp_empty) == 0

    def test_f3_parameter_table_detector_mechanics(self) -> None:
        """Verify detection of parameter tables with units in markdown."""
        valid_md_table = (
            "| Parameter | Description | Value | Units |\n"
            "|---|---|---|---|\n"
            "| $\\beta$ | Discount factor | 0.96 | Dimensionless |\n"
            "| $\\alpha$ | Capital share | 0.36 | Dimensionless |\n"
        )
        valid_es_table = (
            "| Parámetro | Símbolo | Descripción | Unidades |\n"
            "|---|---|---|---|\n"
            "| $\\beta$ | $\\beta$ | Factor de descuento | Adimensional |\n"
        )
        invalid_prose = "We set beta = 0.96 and alpha = 0.36."
        is_table, kind = detect_parameter_table_or_defs(valid_md_table)
        assert is_table is True
        assert kind == "markdown_table"
        is_table_es, _ = detect_parameter_table_or_defs(valid_es_table)
        assert is_table_es is True
        is_table_neg, _ = detect_parameter_table_or_defs(invalid_prose)
        assert is_table_neg is False

    def test_f4_intuition_block_detector_mechanics(self) -> None:
        """Verify formal Intuition block detector recognizes standard lead-ins."""
        pat = re.compile(REQUIRED_SECTIONS[2][1])
        valid_en = "**Intuition.** Precautionary savings bid down interest rates below complete markets."
        valid_es = "**Intuición.** El ahorro precautorio presiona la tasa de interés hacia abajo."
        invalid = "Here is an explanation of the model."
        assert pat.search(valid_en) is not None
        assert pat.search(valid_es) is not None
        assert pat.search(invalid) is None

    def test_f5_literature_citation_detector_mechanics(self) -> None:
        """Verify academic literature citation extractor on canonical formats."""
        sample = (
            "Following Aiyagari (1994) and Huggett (1993), we discretize income risk. "
            "Recent advances by Auclert et al. (2021) and Galí (1999) provide sequence space tools."
        )
        citations = detect_literature_citations(sample)
        assert len(citations) >= 4
        assert any("Aiyagari" in c for c in citations)
        assert any("Auclert" in c for c in citations)
        assert any("Galí" in c for c in citations)

    def test_f6_read_the_output_detector_mechanics(self) -> None:
        """Verify Read the output narrative block detector."""
        pat = re.compile(REQUIRED_SECTIONS[3][1])
        valid_en = "**Read the output.** The precautionary wedge pushes equilibrium return to 3.8%."
        valid_es = "**Lectura de los resultados.** La cuña precautoria empuja el rendimiento al 3.8%."
        invalid = "Output: Gini = 0.6."
        assert pat.search(valid_en) is not None
        assert pat.search(valid_es) is not None
        assert pat.search(invalid) is None

    def test_f7_interactive_knob_and_assert_detector_mechanics(self) -> None:
        """Verify interactive knob (# ← change this) and downstream assert detector."""
        valid_code = (
            "# ← change this discount factor\n"
            "beta_val = 0.95\n"
            "res = solve(beta_val)\n"
            "assert res > 0.0\n"
        )
        invalid_knob_no_assert = (
            "# ← change this discount factor\n"
            "beta_val = 0.95\n"
            "res = solve(beta_val)\n"
        )
        has_knob = "# ← change this" in valid_code
        has_assert = "assert " in valid_code
        assert has_knob and has_assert
        assert "# ← change this" in invalid_knob_no_assert and "assert " not in invalid_knob_no_assert

    def test_f8_machinery_cross_reference_detector_mechanics(self) -> None:
        """Verify ecosystem cross-reference detector."""
        pat = re.compile(REQUIRED_SECTIONS[5][1])
        valid_en = "**How comprehensive is this?** puremacro.vfi powers Krusell-Smith and OLG models."
        valid_es = "**¿Qué tan exhaustivo es esto?** puremacro.vfi alimenta los modelos Krusell-Smith."
        invalid = "puremacro is good."
        assert pat.search(valid_en) is not None
        assert pat.search(valid_es) is not None
        assert pat.search(invalid) is None

    @pytest.mark.parametrize("stem", SHOWCASE_CONFORMING_EXEMPLARS)
    def test_f1_motivating_question_showcase_exemplars(self, stem: str) -> None:
        """Verify motivating economic question presence across conforming showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        pat = re.compile(REQUIRED_SECTIONS[0][1])
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            assert pat.search(content) is not None, f"{p.name}: Motivating question missing"

    @pytest.mark.parametrize("stem", SHOWCASE_CONFORMING_EXEMPLARS)
    def test_f2_latex_math_showcase_exemplars(self, stem: str) -> None:
        """Verify displayed LaTeX ($$...$$) mathematical derivations in showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            disp, _ = extract_latex_formulas(content)
            assert len(disp) >= 1, f"{p.name}: Must contain at least one displayed LaTeX equation ($$...$$)"

    @pytest.mark.parametrize("stem", SHOWCASE_CONFORMING_EXEMPLARS)
    def test_f4_intuition_showcase_exemplars(self, stem: str) -> None:
        """Verify formal **Intuition.** blocks in showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        pat = re.compile(REQUIRED_SECTIONS[2][1])
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            assert pat.search(content) is not None, f"{p.name}: Formal Intuition block missing"

    @pytest.mark.parametrize("stem", SHOWCASE_CITED_EXEMPLARS)
    def test_f5_literature_citations_showcase_exemplars(self, stem: str) -> None:
        """Verify seminal academic literature citations in cited showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            citations = detect_literature_citations(content)
            assert len(citations) >= 1, f"{p.name}: Must contain at least one formal academic literature citation"

    @pytest.mark.parametrize("stem", SHOWCASE_CONFORMING_EXEMPLARS)
    def test_f6_read_the_output_showcase_exemplars(self, stem: str) -> None:
        """Verify **Read the output.** narrative interpretations in showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        pat = re.compile(REQUIRED_SECTIONS[3][1])
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            assert pat.search(content) is not None, f"{p.name}: 'Read the output' interpretation missing"

    @pytest.mark.parametrize("stem", SHOWCASE_CONFORMING_EXEMPLARS)
    def test_f7_interactive_knob_and_assert_showcase_exemplars(self, stem: str) -> None:
        """Verify interactive knob (# ← change this) and downstream assert in showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        knob_markers = ("# ← change this", "# ← Change this", "# ← cambia esto", "# ← Cambia esto")
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            cells = parse_jupytext_cells(content)
            found_knob = False
            knob_has_assert = False
            for c_type, body in cells:
                if c_type == "code" and any(k in body for k in knob_markers):
                    found_knob = True
                    if "assert " in body:
                        knob_has_assert = True
            assert found_knob, f"{p.name}: Interactive knob marker '# ← change this' missing"
            assert knob_has_assert, f"{p.name}: Interactive knob cell missing downstream assertion"

    @pytest.mark.parametrize("stem", SHOWCASE_CONFORMING_EXEMPLARS)
    def test_f8_machinery_cross_references_showcase_exemplars(self, stem: str) -> None:
        """Verify **How comprehensive is this?** cross-references in showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        pat = re.compile(REQUIRED_SECTIONS[5][1])
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            assert pat.search(content) is not None, f"{p.name}: Ecosystem cross-reference section missing"

    def test_f3_parameter_tables_in_elevated_cohort(self) -> None:
        """Verify designated elevated notebooks contain parameter definition tables with units."""
        if not ELEVATED_PARAMETER_TABLE_COHORT:
            pytest.skip("ELEVATED_PARAMETER_TABLE_COHORT empty (awaiting Milestone 1 elevation rollout)")
        for stem in ELEVATED_PARAMETER_TABLE_COHORT:
            p_en = NB_DIR / f"{stem}.py"
            p_es = NB_DIR / f"{stem}_es.py"
            for p in (p_en, p_es):
                assert p.exists(), f"Elevated notebook {p.name} does not exist"
                content = p.read_text(encoding="utf-8")
                is_table, kind = detect_parameter_table_or_defs(content)
                assert is_table, f"{p.name}: Missing parameter definition table with economic units"


# =============================================================================
# Tier 2: Boundary, Extreme & Adversarial Cases
# =============================================================================

class TestTier2BoundaryAndCorner:
    """Tier 2: Boundary conditions, syntax integrity, delimiter balance, and image opacity."""

    def test_boundary_empty_or_degenerate_markdown_cells_detector(self) -> None:
        """Verify detector catches empty or whitespace-only markdown cells."""
        sample_with_empty = (
            "# %% [markdown]\n"
            "# # Real Title\n\n"
            "# %% [markdown]\n"
            "#    \n"
            "# %% [markdown]\n"
            "# Substantial content paragraph.\n"
        )
        cells = parse_jupytext_cells(sample_with_empty)
        empty_indices = [
            idx for idx, (c_type, body) in enumerate(cells)
            if c_type == "markdown" and not "\n".join(
                re.sub(r"^#\s?", "", l).strip() for l in body.splitlines() if l.strip()
            )
        ]
        assert len(empty_indices) == 1
        assert empty_indices[0] == 1

    @pytest.mark.parametrize("stem", SHOWCASE_CONFORMING_EXEMPLARS)
    def test_boundary_no_empty_markdown_cells_in_exemplars(self, stem: str) -> None:
        """Verify zero empty or degenerate markdown cells across showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            cells = parse_jupytext_cells(content)
            for idx, (c_type, body) in enumerate(cells):
                if c_type == "markdown":
                    clean_lines = [
                        re.sub(r"^#\s?", "", l).strip()
                        for l in body.splitlines()
                        if l.strip() and not l.strip() == "#"
                    ]
                    assert len(clean_lines) > 0, f"{p.name}: Markdown cell #{idx} is empty or whitespace-only"

    def test_boundary_latex_delimiter_balance_detector(self) -> None:
        """Verify LaTeX delimiter balancer flags unmatched $$ and unescaped $."""
        valid_disp = "$$ x = y $$ and $a = b$"
        invalid_disp = "$$ x = y and $a = b$"
        invalid_inline = "$$ x = y $$ and $a = b and c = d"
        valid_escaped = "The price is \\$100 and revenue is \\$500."
        invalid_unescaped = "The price is $100 and revenue is $500."

        is_bal1, _ = check_latex_delimiter_balance(valid_disp)
        is_bal2, msg2 = check_latex_delimiter_balance(invalid_disp)
        is_bal3, msg3 = check_latex_delimiter_balance(invalid_inline)
        is_bal4, _ = check_latex_delimiter_balance(valid_escaped)

        assert is_bal1 is True
        assert is_bal2 is False and "$$" in msg2
        assert is_bal3 is False and "$" in msg3
        assert is_bal4 is True

    @pytest.mark.parametrize("stem", SHOWCASE_CONFORMING_EXEMPLARS)
    def test_boundary_latex_delimiter_balance_showcase_exemplars(self, stem: str) -> None:
        """Verify 100% LaTeX delimiter balance across conforming showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            is_bal, err = check_latex_delimiter_balance(content)
            assert is_bal, f"{p.name}: LaTeX balance violation: {err}"

    def test_boundary_escaped_currency_syntax(self) -> None:
        """Verify that currency dollar amounts must be escaped (\\$) to avoid corrupting math parsing."""
        raw_currency_sample = "Total funding was $500 million for the project."
        escaped_currency_sample = "Total funding was \\$500 million for the project."
        math_sample = "The price level satisfies $p = 500$ in equilibrium."

        is_bal_raw, err_raw = check_latex_delimiter_balance(raw_currency_sample)
        is_bal_esc, err_esc = check_latex_delimiter_balance(escaped_currency_sample)
        is_bal_math, err_math = check_latex_delimiter_balance(math_sample)

        assert is_bal_raw is False, "Raw unescaped currency dollar should trigger parity imbalance"
        assert "$" in err_raw
        assert is_bal_esc is True, f"Escaped currency dollar should be balanced: {err_esc}"
        assert is_bal_math is True, f"Valid inline math should be balanced: {err_math}"

    def test_boundary_strict_emoji_ban_detector(self) -> None:
        """Verify detector catches emojis prohibited by repository style guides."""
        sample_clean = "This model evaluates general equilibrium dynamics."
        sample_emoji = "This model evaluates general equilibrium dynamics! 🚀🔥💡"
        assert len(detect_emojis(sample_clean)) == 0
        assert len(detect_emojis(sample_emoji)) == 3

    @pytest.mark.parametrize("stem", SHOWCASE_CONFORMING_EXEMPLARS)
    def test_boundary_strict_emoji_ban_showcase_exemplars(self, stem: str) -> None:
        """Verify zero emojis in markdown prose across conforming showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            emojis = detect_emojis(content)
            assert len(emojis) == 0, f"{p.name}: Prohibited emojis found: {emojis}"

    def test_boundary_canvas_opacity_detector_unit(self) -> None:
        """Verify canvas opacity detector identifies transparent vs opaque PNG buffers."""
        # 1. Fully opaque RGBA image
        opaque_img = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
        buf_opaque = io.BytesIO()
        opaque_img.save(buf_opaque, format="PNG")
        is_op, min_a = check_png_image_opacity(buf_opaque.getvalue())
        assert is_op is True and min_a == 255

        # 2. Transparent RGBA image
        trans_img = Image.new("RGBA", (10, 10), (255, 255, 255, 200))
        buf_trans = io.BytesIO()
        trans_img.save(buf_trans, format="PNG")
        is_op_trans, min_a_trans = check_png_image_opacity(buf_trans.getvalue())
        assert is_op_trans is False and min_a_trans == 200

    @pytest.mark.parametrize("rel_name", TARGET_15_NOTEBOOKS)
    def test_boundary_canvas_opacity_target_cohort(self, rel_name: str) -> None:
        """Verify 100% canvas opacity (min_alpha == 255) on all PNGs in target cohort."""
        nb_path = NB_DIR / rel_name
        if not nb_path.exists():
            pytest.skip(f"{rel_name} not found in notebooks/")
        images = extract_ipynb_png_images(nb_path)
        for cell_idx, out_idx, raw_bytes in images:
            is_op, min_a = check_png_image_opacity(raw_bytes)
            assert is_op, (
                f"{rel_name}: cell #{cell_idx} output #{out_idx} has transparent canvas (min_alpha={min_a} < 255)"
            )

    @pytest.mark.parametrize("rel_name", TARGET_15_NOTEBOOKS)
    def test_boundary_nbformat_v4_schema_validation(self, rel_name: str) -> None:
        """Verify official nbformat v4 JSON schema validation on target cohort."""
        nb_path = NB_DIR / rel_name
        if not nb_path.exists():
            pytest.skip(f"{rel_name} not found in notebooks/")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)


# =============================================================================
# Tier 3: Cross-Feature Interactions & Parity
# =============================================================================

class TestTier3CrossFeature:
    """Tier 3: Bilingual code logic identity, monotonic sequencing, and formula parity."""

    def test_interaction_code_normalizer_unit(self) -> None:
        """Verify normalize_code eliminates comments and blank lines while preserving semantics."""
        code_en = (
            "# Set random seed\n"
            "rng = np.random.default_rng(123)  # generator\n"
            "\n"
            "a_star = 4.2\n"
        )
        code_es = (
            "# Fijar semilla aleatoria\n"
            "rng = np.random.default_rng(123)\n"
            "a_star = 4.2  # valor estrella\n"
        )
        assert normalize_code(code_en) == normalize_code(code_es)

    @pytest.mark.parametrize("en_file, es_file", BILINGUAL_EXEMPLAR_PAIRS)
    def test_interaction_bilingual_code_logic_identity(self, en_file: str, es_file: str) -> None:
        """Verify exact bilingual execution logic identity: normalize_code(en) == normalize_code(es)."""
        p_en = NB_DIR / en_file
        p_es = NB_DIR / es_file
        if not p_en.exists() or not p_es.exists():
            pytest.skip(f"Pair {en_file} / {es_file} not found")

        en_cells = extract_code_cells(p_en.read_text(encoding="utf-8"))
        es_cells = extract_code_cells(p_es.read_text(encoding="utf-8"))
        assert len(en_cells) == len(es_cells), (
            f"{en_file} ({len(en_cells)} code cells) vs {es_file} ({len(es_cells)} code cells) count mismatch"
        )

        for idx, (c_en, c_es) in enumerate(zip(en_cells, es_cells)):
            norm_en = normalize_code(c_en)
            norm_es = normalize_code(c_es)
            assert norm_en == norm_es, (
                f"Code logic diverges in pair {en_file} / {es_file} at cell #{idx + 1}:\n"
                f"--- EN ---\n{norm_en[:300]}\n--- ES ---\n{norm_es[:300]}"
            )

    @pytest.mark.parametrize("stem", SHOWCASE_CONFORMING_EXEMPLARS)
    def test_interaction_section_sequence_monotonicity(self, stem: str) -> None:
        """Verify strictly monotonic ordering of all 6 template sections in showcase exemplars."""
        p_en = NB_DIR / f"{stem}.py"
        p_es = NB_DIR / f"{stem}_es.py"
        for p in (p_en, p_es):
            if not p.exists():
                pytest.skip(f"{p.name} not found")
            content = p.read_text(encoding="utf-8")
            positions: list[tuple[str, int]] = []
            for name, pat in REQUIRED_SECTIONS:
                m = re.search(pat, content)
                assert m is not None, f"{p.name}: Section '{name}' missing"
                positions.append((name, m.start()))

            for i in range(len(positions) - 1):
                sec_prev, pos_prev = positions[i]
                sec_next, pos_next = positions[i + 1]
                assert pos_prev < pos_next, (
                    f"{p.name}: Section '{sec_prev}' (pos {pos_prev}) must precede '{sec_next}' (pos {pos_next})"
                )

    @pytest.mark.parametrize("en_file, es_file", BILINGUAL_EXEMPLAR_PAIRS)
    def test_interaction_bilingual_formula_count_parity(self, en_file: str, es_file: str) -> None:
        """Verify Spanish editions maintain mathematical formula parity against English twins."""
        p_en = NB_DIR / en_file
        p_es = NB_DIR / es_file
        if not p_en.exists() or not p_es.exists():
            pytest.skip(f"Pair {en_file} / {es_file} not found")

        en_disp, en_inl = extract_latex_formulas(p_en.read_text(encoding="utf-8"))
        es_disp, es_inl = extract_latex_formulas(p_es.read_text(encoding="utf-8"))

        assert len(es_disp) >= len(en_disp), (
            f"Display formula parity mismatch in {en_file} vs {es_file}: {len(en_disp)} EN vs {len(es_disp)} ES"
        )
        assert len(es_inl) >= int(0.80 * len(en_inl)), (
            f"Inline formula parity mismatch in {en_file} vs {es_file}: {len(en_inl)} EN vs {len(es_inl)} ES"
        )


# =============================================================================
# Tier 4: Real-World Acceptance Scenarios
# =============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Acceptance scenarios, repository-wide tracebacks, and execution counts."""

    def test_acceptance_target_15_showcase_cohort(self) -> None:
        """Verify all 15 target showcase notebooks exist and have substantial size (> 10,000 bytes)."""
        for rel_name in TARGET_15_NOTEBOOKS:
            p = NB_DIR / rel_name
            assert p.exists(), f"Target notebook {rel_name} missing from notebooks/"
            sz = p.stat().st_size
            assert sz > 10_000, f"Target notebook {rel_name} unexpectedly small ({sz} bytes)"

    def test_acceptance_course_companion_pedagogical_tags(self) -> None:
        """Verify required educational markers in the course companion suite."""
        lessons = [p for p in COURSE_DIR.glob("[0-9]*_es.py") if p.name != "00_syllabus_es.py"]
        assert len(lessons) >= 10, "Too few course lessons discovered"
        for p in lessons:
            text = p.read_text(encoding="utf-8").lower()
            for section, needles in COURSE_REQUIRED_SECTIONS.items():
                assert any(n in text for n in needles), f"{p.name}: Missing required section '{section}'"
            assert "slide_type" in text, f"{p.name}: Missing slide metadata tags"

    def test_acceptance_repository_wide_zero_error_tracebacks(self) -> None:
        """Traverse all notebooks repository-wide and assert zero runtime exception tracebacks."""
        all_nbs = get_all_repository_notebooks()
        assert len(all_nbs) >= 210, f"Discovered only {len(all_nbs)} notebooks (expected >= 210)"

        tracebacks: list[str] = []
        for p in all_nbs:
            with open(p, "r", encoding="utf-8") as f:
                try:
                    nb = json.load(f)
                except Exception as e:
                    tracebacks.append(f"{p.name}: JSON parse failure: {e}")
                    continue
            for cell_idx, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "error":
                        tracebacks.append(
                            f"{p.relative_to(PROJ_ROOT)} cell #{cell_idx}: "
                            f"{out.get('ename')}: {out.get('evalue')}"
                        )
        assert not tracebacks, f"Found {len(tracebacks)} cell error tracebacks:\n" + "\n".join(tracebacks[:10])

    @pytest.mark.parametrize("rel_name", TARGET_15_NOTEBOOKS)
    def test_acceptance_target_15_valid_execution_counts(self, rel_name: str) -> None:
        """Verify 100% of executable code cells in target notebooks have integer execution_count > 0."""
        nb_path = NB_DIR / rel_name
        if not nb_path.exists():
            pytest.skip(f"{rel_name} not found in notebooks/")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        assert len(code_cells) > 0, f"{rel_name} has no code cells"

        unexecuted: list[int] = []
        for idx, cell in enumerate(code_cells):
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            has_code = any(l.strip() and not l.strip().startswith("#") for l in src.splitlines())
            if not has_code:
                continue
            ec = cell.get("execution_count")
            if ec is None or not isinstance(ec, int) or ec <= 0:
                unexecuted.append(idx)

        assert not unexecuted, f"{rel_name}: Unexecuted code cells at indices {unexecuted}"

    def test_acceptance_build_node_validation_clean(self) -> None:
        """Verify validate_notebook_node logic from build_notebooks.py runs clean on target cohort."""
        import importlib.util
        bn_tool = PROJ_ROOT / "tools" / "build_notebooks.py"
        assert bn_tool.exists(), "tools/build_notebooks.py missing"
        spec = importlib.util.spec_from_file_location("build_notebooks", bn_tool)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        validate_notebook_node = mod.validate_notebook_node

        for rel_name in TARGET_15_NOTEBOOKS:
            nb_path = NB_DIR / rel_name
            if not nb_path.exists():
                pytest.skip(f"{rel_name} not found in notebooks/")
            errs = validate_notebook_node(nb_path)
            assert not errs, f"{rel_name} failed node validation: {errs}"
