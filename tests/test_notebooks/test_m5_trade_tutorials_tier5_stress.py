"""Empirical Challenger Test Suite: Tier 5 Stress Testing for Trade Tutorials (63, 64, 65).

Author: teamwork_preview_challenger_m5_2
Mission: Adversarially challenge and empirically verify Milestone 5 trade tutorials across 4 strict gates:
1. Headless Execution Timing: Run all 6 notebooks headlessly; assert every notebook finishes in < 15 seconds.
2. Pyodide Import Purity: AST check across all 6 notebooks for zero unauthorized packages (no torch, jax, statsmodels, linearmodels, requests, socket).
3. Image Opacity Forensics: Inspect all generated figure PNGs in the 6 .ipynb files; assert 100% opaque alpha channel (min_alpha == 255) and OKLab contrast ratio >= 7:1.
4. Code Parity Invariant: Assert normalize_code(en) == normalize_code(es) across 100% of code cells.
"""
from __future__ import annotations

import ast
import base64
import importlib.util
import io
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from PIL import Image
import pytest

PROJ_ROOT = Path(__file__).resolve().parents[2]
NB_DIR = PROJ_ROOT / "notebooks"
BUILD_TOOL = PROJ_ROOT / "tools" / "build_notebooks.py"

TARGET_NOTEBOOKS = [
    {
        "id": 63,
        "stem": "63_trade_wars_and_nash_tariffs",
        "py_en": NB_DIR / "63_trade_wars_and_nash_tariffs.py",
        "py_es": NB_DIR / "63_trade_wars_and_nash_tariffs_es.py",
        "ipynb_en": NB_DIR / "63_trade_wars_and_nash_tariffs.ipynb",
        "ipynb_es": NB_DIR / "63_trade_wars_and_nash_tariffs_es.ipynb",
    },
    {
        "id": 64,
        "stem": "64_singularities_and_keller_pac",
        "py_en": NB_DIR / "64_singularities_and_keller_pac.py",
        "py_es": NB_DIR / "64_singularities_and_keller_pac_es.py",
        "ipynb_en": NB_DIR / "64_singularities_and_keller_pac.ipynb",
        "ipynb_es": NB_DIR / "64_singularities_and_keller_pac_es.ipynb",
    },
    {
        "id": 65,
        "stem": "65_gvc_cascades_and_welfare_decomposition",
        "py_en": NB_DIR / "65_gvc_cascades_and_welfare_decomposition.py",
        "py_es": NB_DIR / "65_gvc_cascades_and_welfare_decomposition_es.py",
        "ipynb_en": NB_DIR / "65_gvc_cascades_and_welfare_decomposition.ipynb",
        "ipynb_es": NB_DIR / "65_gvc_cascades_and_welfare_decomposition_es.ipynb",
    },
]

ALL_SIX_PY_FILES = [
    NB_DIR / "63_trade_wars_and_nash_tariffs.py",
    NB_DIR / "63_trade_wars_and_nash_tariffs_es.py",
    NB_DIR / "64_singularities_and_keller_pac.py",
    NB_DIR / "64_singularities_and_keller_pac_es.py",
    NB_DIR / "65_gvc_cascades_and_welfare_decomposition.py",
    NB_DIR / "65_gvc_cascades_and_welfare_decomposition_es.py",
]

ALL_SIX_IPYNB_FILES = [
    NB_DIR / "63_trade_wars_and_nash_tariffs.ipynb",
    NB_DIR / "63_trade_wars_and_nash_tariffs_es.ipynb",
    NB_DIR / "64_singularities_and_keller_pac.ipynb",
    NB_DIR / "64_singularities_and_keller_pac_es.ipynb",
    NB_DIR / "65_gvc_cascades_and_welfare_decomposition.ipynb",
    NB_DIR / "65_gvc_cascades_and_welfare_decomposition_es.ipynb",
]

UNAUTHORIZED_PACKAGES = frozenset({
    "torch",
    "jax",
    "statsmodels",
    "linearmodels",
    "requests",
    "socket",
    "urllib3",
    "httpx",
    "aiohttp",
})

ALLOWED_PYODIDE_PACKAGES = frozenset({
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "puremacro",
    "_nbstyle",
    "time",
    "io",
    "base64",
    "json",
    "re",
    "math",
    "sys",
    "os",
    "pathlib",
    "typing",
    "dataclasses",
    "warnings",
})


def parse_jupytext_cells(content: str) -> list[tuple[str, str]]:
    """Split Jupytext percent source into (cell_type, cell_text) pairs."""
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
    """Return all code cell texts from Jupytext percent source."""
    return [body for c_type, body in parse_jupytext_cells(content) if c_type == "code"]


def normalize_code(code: str) -> str:
    """Normalize python code for bilingual structural comparison.

    Removes comments, inline comments, docstrings, and empty lines.
    """
    lines: list[str] = []
    for line in code.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Strip inline comments
        code_part = re.sub(r"#.*$", "", line).rstrip()
        if code_part.strip():
            lines.append(code_part.strip())
    return "\n".join(lines)


def extract_imports_from_ast(code_text: str) -> set[str]:
    """Traverse AST and return all top-level imported module names."""
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


def extract_ipynb_pngs(nb_path: Path) -> list[tuple[int, int, bytes]]:
    """Extract embedded PNG byte payloads from .ipynb file."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    images: list[tuple[int, int, bytes]] = []
    for c_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for o_idx, out in enumerate(cell.get("outputs", [])):
            data = out.get("data", {})
            if "image/png" in data:
                raw = data["image/png"]
                if isinstance(raw, list):
                    raw = "".join(raw)
                images.append((c_idx, o_idx, base64.b64decode(raw)))
    return images


def compute_oklab_and_luminance_contrast(rgb: np.ndarray) -> tuple[float, float, float]:
    """Compute OKLab lightness metrics and WCAG relative luminance contrast ratio.

    Returns (contrast_ratio, min_oklab_L, max_oklab_L).
    """
    rgb_norm = rgb.astype(float) / 255.0
    # sRGB to linear RGB
    linear = np.where(
        rgb_norm <= 0.04045,
        rgb_norm / 12.92,
        ((rgb_norm + 0.055) / 1.055) ** 2.4,
    )
    # Relative luminance Y
    lum = 0.2126 * linear[:, :, 0] + 0.7152 * linear[:, :, 1] + 0.0722 * linear[:, :, 2]
    min_lum = float(np.min(lum))
    max_lum = float(np.max(lum))
    cr = (max_lum + 0.05) / (min_lum + 0.05)

    # OKLab lightness L = (Y_linear)^(1/3) for neutrals/grays
    # or general OKLab formula:
    # LMS from linear sRGB
    l = 0.4122214708 * linear[:, :, 0] + 0.5363325363 * linear[:, :, 1] + 0.0514459929 * linear[:, :, 2]
    m = 0.2119034982 * linear[:, :, 0] + 0.6806995451 * linear[:, :, 1] + 0.1073969566 * linear[:, :, 2]
    s = 0.0883024619 * linear[:, :, 0] + 0.2817188376 * linear[:, :, 1] + 0.6299787005 * linear[:, :, 2]
    l_ = np.cbrt(np.maximum(l, 0.0))
    m_ = np.cbrt(np.maximum(m, 0.0))
    s_ = np.cbrt(np.maximum(s, 0.0))
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    min_L = float(np.min(L))
    max_L = float(np.max(L))

    return cr, min_L, max_L


# =============================================================================
# Gate 1: Headless Execution Timing (< 15 seconds)
# =============================================================================

class TestGate1HeadlessExecutionTiming:
    """Stress test execution latency headlessly for all 6 notebooks (< 15s)."""

    @pytest.mark.parametrize("py_path", ALL_SIX_PY_FILES, ids=lambda p: p.name)
    def test_headless_execution_time_under_15s(self, py_path: Path) -> None:
        """Assert notebook executes headlessly from top to bottom in < 15.0s with rc == 0."""
        assert py_path.exists(), f"Source file {py_path.name} does not exist"
        assert BUILD_TOOL.exists(), f"Build tool {BUILD_TOOL} does not exist"

        spec = importlib.util.spec_from_file_location("build_notebooks_runner", BUILD_TOOL)
        assert spec and spec.loader
        bn_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bn_mod)

        t0 = time.perf_counter()
        rc = bn_mod.build_one(py_path, check=True)
        elapsed = time.perf_counter() - t0

        assert rc == 0, f"{py_path.name}: Headless execution check failed (rc={rc})"
        assert elapsed < 15.0, (
            f"{py_path.name}: Execution duration {elapsed:.2f}s exceeded strict 15.0s threshold!"
        )


# =============================================================================
# Gate 2: Pyodide Import Purity (Zero Unauthorized Packages)
# =============================================================================

class TestGate2PyodideImportPurity:
    """AST check across all 6 notebooks for zero unauthorized packages."""

    @pytest.mark.parametrize("py_path", ALL_SIX_PY_FILES, ids=lambda p: p.name)
    def test_zero_unauthorized_imports(self, py_path: Path) -> None:
        """Assert zero unauthorized packages (torch, jax, statsmodels, linearmodels, requests, socket)."""
        content = py_path.read_text(encoding="utf-8")
        code_text = "\n".join(extract_code_cells(content))
        imports = extract_imports_from_ast(code_text)

        violations = imports.intersection(UNAUTHORIZED_PACKAGES)
        assert not violations, (
            f"{py_path.name}: Found unauthorized package imports: {violations}"
        )

    @pytest.mark.parametrize("py_path", ALL_SIX_PY_FILES, ids=lambda p: p.name)
    def test_strict_pyodide_four_package_compliance(self, py_path: Path) -> None:
        """Assert all imports strictly adhere to the Pyodide standard library & approved packages."""
        content = py_path.read_text(encoding="utf-8")
        code_text = "\n".join(extract_code_cells(content))
        imports = extract_imports_from_ast(code_text)

        unrecognized = imports - ALLOWED_PYODIDE_PACKAGES
        assert not unrecognized, (
            f"{py_path.name}: Found imports outside Pyodide allowed standard: {unrecognized}"
        )


# =============================================================================
# Gate 3: Image Opacity Forensics (Alpha == 255 & OKLab Contrast >= 7:1)
# =============================================================================

class TestGate3ImageOpacityAndContrastForensics:
    """Inspect all generated figure PNGs in the 6 .ipynb files."""

    @pytest.mark.parametrize("ipynb_path", ALL_SIX_IPYNB_FILES, ids=lambda p: p.name)
    def test_image_alpha_is_100_percent_opaque(self, ipynb_path: Path) -> None:
        """Assert 100% opaque alpha channel (min_alpha == 255) across all embedded PNGs."""
        assert ipynb_path.exists(), f"Notebook {ipynb_path.name} does not exist"
        images = extract_ipynb_pngs(ipynb_path)
        assert len(images) > 0, f"{ipynb_path.name} contains zero generated figures!"

        transparency_failures = []
        for c_idx, o_idx, raw in images:
            im = Image.open(io.BytesIO(raw))
            assert im.mode == "RGBA", f"{ipynb_path.name} cell {c_idx}: unexpected mode {im.mode}"
            arr = np.array(im)
            alpha = arr[:, :, 3]
            min_alpha = int(alpha.min())
            if min_alpha < 255:
                num_transparent = int(np.sum(alpha < 255))
                transparency_failures.append(
                    f"cell {c_idx} out {o_idx}: min_alpha={min_alpha} ({num_transparent} pixels)"
                )

        assert not transparency_failures, (
            f"{ipynb_path.name}: Embedded PNGs have transparent pixels:\n"
            + "\n".join(transparency_failures)
        )

    @pytest.mark.parametrize("ipynb_path", ALL_SIX_IPYNB_FILES, ids=lambda p: p.name)
    def test_image_oklab_contrast_ratio_ge_7(self, ipynb_path: Path) -> None:
        """Assert OKLab / relative luminance contrast ratio >= 7:1 (WCAG AAA threshold)."""
        assert ipynb_path.exists(), f"Notebook {ipynb_path.name} does not exist"
        images = extract_ipynb_pngs(ipynb_path)
        assert len(images) > 0, f"{ipynb_path.name} contains zero generated figures!"

        contrast_failures = []
        for c_idx, o_idx, raw in images:
            im = Image.open(io.BytesIO(raw))
            arr = np.array(im)
            rgb = arr[:, :, :3]
            cr, min_L, max_L = compute_oklab_and_luminance_contrast(rgb)
            if cr < 7.0:
                contrast_failures.append(
                    f"cell {c_idx} out {o_idx}: contrast ratio {cr:.2f}:1 < 7:1 (OKLab L: [{min_L:.2f}, {max_L:.2f}])"
                )

        assert not contrast_failures, (
            f"{ipynb_path.name}: Embedded PNGs failed contrast ratio >= 7:1:\n"
            + "\n".join(contrast_failures)
        )


# =============================================================================
# Gate 4: Code Parity Invariant (normalize_code(en) == normalize_code(es))
# =============================================================================

class TestGate4CodeParityInvariant:
    """Assert normalize_code(en) == normalize_code(es) across 100% of code cells."""

    @pytest.mark.parametrize("pair", TARGET_NOTEBOOKS, ids=lambda p: p["stem"])
    def test_code_cell_parity_across_pairs(self, pair: dict[str, Any]) -> None:
        """Verify identical code logic between English and Spanish editions."""
        py_en: Path = pair["py_en"]
        py_es: Path = pair["py_es"]
        assert py_en.exists(), f"Missing {py_en.name}"
        assert py_es.exists(), f"Missing {py_es.name}"

        en_cells = extract_code_cells(py_en.read_text(encoding="utf-8"))
        es_cells = extract_code_cells(py_es.read_text(encoding="utf-8"))

        assert len(en_cells) == len(es_cells), (
            f"{pair['stem']}: Mismatch in code cell counts! EN has {len(en_cells)}, ES has {len(es_cells)}"
        )

        divergences = []
        for idx, (en_code, es_code) in enumerate(zip(en_cells, es_cells)):
            norm_en = normalize_code(en_code)
            norm_es = normalize_code(es_code)
            if norm_en != norm_es:
                divergences.append(
                    f"Cell {idx + 1}:\n-- EN --\n{norm_en}\n-- ES --\n{norm_es}\n"
                )

        assert not divergences, (
            f"{pair['stem']}: Found {len(divergences)} code cell discrepancies:\n"
            + "\n".join(divergences)
        )
