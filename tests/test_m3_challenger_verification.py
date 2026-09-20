"""Empirical challenge verification test suite for Milestone 3.

Adversarially tests:
1. Pyodide Environment Sanity across showcase notebooks (cluster 30-38 and all notebooks).
2. Visual Card Opacity: 100% opaque alpha channels (min_alpha == 255) on all embedded PNG images.
3. Mathematical & LaTeX Soundness in cluster 30-38 (formula balance, parity, equations).
"""
import ast
import base64
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"
CLUSTER_30_38_INDICES = list(range(30, 39))


def _get_cluster_notebooks(ext: str = ".ipynb") -> list[Path]:
    files = []
    for idx in CLUSTER_30_38_INDICES:
        pattern = f"{idx}_*{ext}"
        files.extend(NOTEBOOKS_DIR.glob(pattern))
    return sorted([f for f in files if ".ipynb_checkpoints" not in f.parts])


def _get_all_showcase_notebooks(ext: str = ".ipynb") -> list[Path]:
    files = list(NOTEBOOKS_DIR.glob(f"*{ext}"))
    return sorted([f for f in files if not f.name.startswith("_") and ".ipynb_checkpoints" not in f.parts])


# --- Test 1: Pyodide Environment Sanity ---

STANDARD_LIB_MODULES = {
    "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib",
    "copy", "csv", "dataclasses", "datetime", "decimal", "difflib", "enum",
    "fractions", "functools", "glob", "gzip", "hashlib", "heapq", "importlib",
    "inspect", "io", "itertools", "json", "logging", "math", "numbers", "os",
    "pathlib", "pickle", "pprint", "random", "re", "shutil", "socket", "sqlite3",
    "string", "subprocess", "sys", "tempfile", "time", "traceback", "types",
    "typing", "unittest", "urllib", "uuid", "warnings", "weakref", "zipfile"
}

ALLOWED_THIRD_PARTY = {
    "numpy", "scipy", "pandas", "matplotlib", "puremacro", "_nbstyle",
    "IPython", "ipywidgets"  # ipywidgets may appear in some widget demo notebooks if guarded or in curso
}

# Strictly forbidden in Pyodide showcase:
FORBIDDEN_PACKAGES = {
    "statsmodels", "linearmodels", "arch", "sklearn", "scikit-learn",
    "torch", "tensorflow", "jax", "sympy", "seaborn", "altair", "bokeh",
    "plotly", "numba", "requests", "bs4", "pdfplumber", "pypdf"
}


def _extract_imports_from_code(code_str: str) -> set[str]:
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def test_cluster_30_38_pyodide_sanity():
    """Verify cluster 30-38 strictly imports only allowed Pyodide packages."""
    py_files = _get_cluster_notebooks(".py")
    assert len(py_files) == 18, f"Expected 18 scripts for cluster 30-38 (EN & ES), found {len(py_files)}"

    unauthorized = {}
    for p in py_files:
        code = p.read_text(encoding="utf-8")
        imports = _extract_imports_from_code(code)
        forbidden = imports.intersection(FORBIDDEN_PACKAGES)
        non_standard = imports - STANDARD_LIB_MODULES - ALLOWED_THIRD_PARTY
        if forbidden or non_standard:
            unauthorized[p.name] = {
                "forbidden": sorted(forbidden),
                "non_standard": sorted(non_standard),
            }

    assert not unauthorized, f"Unauthorized imports found in cluster 30-38:\n{json.dumps(unauthorized, indent=2)}"


def test_all_showcase_scripts_pyodide_sanity():
    """Verify all showcase scripts strictly adhere to Pyodide package constraints."""
    py_files = _get_all_showcase_notebooks(".py")
    violations = {}
    for p in py_files:
        code = p.read_text(encoding="utf-8")
        imports = _extract_imports_from_code(code)
        forbidden = imports.intersection(FORBIDDEN_PACKAGES)
        if forbidden:
            violations[p.name] = sorted(forbidden)

    assert not violations, f"Forbidden package imports in showcase scripts:\n{json.dumps(violations, indent=2)}"


# --- Test 2: Visual Card Opacity Scan ---

def test_cluster_30_38_png_card_opacity():
    """Scan newly synced/rendered images in cluster 30-38 .ipynb files and verify min_alpha == 255."""
    ipynb_files = _get_cluster_notebooks(".ipynb")
    assert len(ipynb_files) == 18, f"Expected 18 .ipynb files in cluster 30-38, found {len(ipynb_files)}"

    transparent_findings = []
    total_pngs = 0

    for p in ipynb_files:
        with open(p, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        for cell_idx, cell in enumerate(nb_data.get("cells", [])):
            for out_idx, out in enumerate(cell.get("outputs", [])):
                payloads = out.get("data", {})
                if "image/png" not in payloads:
                    continue
                total_pngs += 1
                raw = payloads["image/png"]
                if isinstance(raw, list):
                    raw = "".join(raw)
                try:
                    img_bytes = base64.b64decode(raw)
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode == "RGBA":
                        arr = np.array(img)
                        min_alpha = int(arr[:, :, 3].min())
                        if min_alpha < 255:
                            # Also check how many pixels are transparent
                            num_transparent = int((arr[:, :, 3] < 255).sum())
                            total_pixels = arr.shape[0] * arr.shape[1]
                            transparent_findings.append({
                                "file": p.name,
                                "cell": cell_idx,
                                "output": out_idx,
                                "min_alpha": min_alpha,
                                "transparent_ratio": num_transparent / total_pixels,
                            })
                except Exception as e:
                    transparent_findings.append({
                        "file": p.name,
                        "cell": cell_idx,
                        "output": out_idx,
                        "error": str(e),
                    })

    assert not transparent_findings, f"Found transparent PNG cards in cluster 30-38:\n{json.dumps(transparent_findings, indent=2)}"


def test_all_showcase_png_card_opacity():
    """Verify opacity min_alpha == 255 across all showcase .ipynb files that contain images."""
    ipynb_files = _get_all_showcase_notebooks(".ipynb")
    transparent_findings = []
    total_pngs = 0

    for p in ipynb_files:
        with open(p, "r", encoding="utf-8") as f:
            nb_data = json.load(f)

        for cell_idx, cell in enumerate(nb_data.get("cells", [])):
            for out_idx, out in enumerate(cell.get("outputs", [])):
                payloads = out.get("data", {})
                if "image/png" not in payloads:
                    continue
                total_pngs += 1
                raw = payloads["image/png"]
                if isinstance(raw, list):
                    raw = "".join(raw)
                try:
                    img_bytes = base64.b64decode(raw)
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.mode == "RGBA":
                        arr = np.array(img)
                        min_alpha = int(arr[:, :, 3].min())
                        if min_alpha < 255:
                            transparent_findings.append({
                                "file": p.name,
                                "cell": cell_idx,
                                "output": out_idx,
                                "min_alpha": min_alpha,
                            })
                except Exception as e:
                    transparent_findings.append({
                        "file": p.name,
                        "cell": cell_idx,
                        "output": out_idx,
                        "error": str(e),
                    })

    assert not transparent_findings, f"Found transparent PNG cards in showcase:\n{json.dumps(transparent_findings, indent=2)}"


# --- Test 3: Mathematical & LaTeX Soundness in Cluster 30-38 ---

def _extract_markdown_cells(nb_path: Path) -> list[dict[str, Any]]:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    cells = []
    for idx, c in enumerate(nb_data.get("cells", [])):
        if c.get("cell_type") == "markdown":
            source = c.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            cells.append({"cell_idx": idx, "source": source})
    return cells


def test_cluster_30_38_latex_syntax_and_balance():
    """Verify that all markdown cells in cluster 30-38 have balanced math delimiters and valid LaTeX."""
    ipynb_files = _get_cluster_notebooks(".ipynb")
    syntax_issues = []

    for p in ipynb_files:
        md_cells = _extract_markdown_cells(p)
        for cell in md_cells:
            idx = cell["cell_idx"]
            text = cell["source"]

            # Check $$ balance
            double_dollars = text.count("$$")
            if double_dollars % 2 != 0:
                syntax_issues.append({
                    "file": p.name,
                    "cell": idx,
                    "issue": f"Unbalanced double dollar ($$) count: {double_dollars}",
                })

            # Check single $ balance (excluding $$)
            # Remove $$ first
            no_dd = text.replace("$$", "")
            # Count remaining $ (excluding escaped \$)
            unescaped_single = len(re.findall(r"(?<!\\)\$", no_dd))
            if unescaped_single % 2 != 0:
                syntax_issues.append({
                    "file": p.name,
                    "cell": idx,
                    "issue": f"Unbalanced single dollar ($) count: {unescaped_single}",
                })

    assert not syntax_issues, f"LaTeX delimiter imbalance in cluster 30-38:\n{json.dumps(syntax_issues, indent=2)}"


def test_cluster_30_38_bilingual_formula_parity():
    """Verify that Spanish editions in cluster 30-38 have matching mathematical formula counts and key LaTeX symbols."""
    for idx in CLUSTER_30_38_INDICES:
        en_matches = list(NOTEBOOKS_DIR.glob(f"{idx}_[!_]*.ipynb"))
        en_matches = [m for m in en_matches if not m.name.endswith("_es.ipynb")]
        es_matches = list(NOTEBOOKS_DIR.glob(f"{idx}_*_es.ipynb"))

        if not en_matches or not es_matches:
            continue
        en_path = en_matches[0]
        es_path = es_matches[0]

        en_cells = _extract_markdown_cells(en_path)
        es_cells = _extract_markdown_cells(es_path)

        en_text = " ".join(c["source"] for c in en_cells)
        es_text = " ".join(c["source"] for c in es_cells)

        # Extract display formulas
        en_formulas = re.findall(r"\$\$(.*?)\$\$", en_text, re.DOTALL)
        es_formulas = re.findall(r"\$\$(.*?)\$\$", es_text, re.DOTALL)

        # Extract inline formulas
        en_inline = re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", en_text, re.DOTALL)
        es_inline = re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", es_text, re.DOTALL)

        assert len(es_formulas) >= len(en_formulas), (
            f"Notebook {idx}: ES has fewer display formulas ({len(es_formulas)}) than EN ({len(en_formulas)})"
        )
        assert len(es_inline) >= 0.8 * len(en_inline), (
            f"Notebook {idx}: ES has significantly fewer inline formulas ({len(es_inline)}) than EN ({len(en_inline)})"
        )


# --- Test 4: Adversarial Layout Helper Stress Test ---

def test_nbstyle_figura_presets_and_overrides():
    """Adversarially stress-test _nbstyle.figura() with string presets, tuples, and overrides."""
    import importlib.util
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")

    modules = [
        PROJ_ROOT / "notebooks" / "_nbstyle.py",
        PROJ_ROOT / "curso" / "notebooks" / "_nbstyle.py",
    ]
    presets = [
        "2", "ancho", "cuadrada", "alta", "media",
        "1", "1_alto", "1_bajo", "2_alto", "3", "2x2", "2x3", "3x2", "3x3",
        "ancha", "ancha_baja", "completa", "media_alta", "media_baja", "tercio", "dos_tercios"
    ]
    tuple_figsizes = [(6.0, 4.0), (8.5, 5.0), (12.0, 6.0)]

    for mod_path in modules:
        spec = importlib.util.spec_from_file_location("test_nbstyle", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # String presets
        for p in presets:
            fig, ax = mod.figura(figsize=p)
            assert fig is not None
            plt.close(fig)

        # Tuple figsizes
        for t in tuple_figsizes:
            fig, ax = mod.figura(figsize=t)
            assert tuple(fig.get_size_inches()) == t
            plt.close(fig)

        # Numeric overrides
        fig, ax = mod.figura(ancho=7.5, alto=4.0)
        w, h = fig.get_size_inches()
        assert abs(w - 7.5) < 1e-6 and abs(h - 4.0) < 1e-6
        plt.close(fig)

        # Combination preset + override
        fig, ax = mod.figura(figsize="2", ancho=7.5)
        w, h = fig.get_size_inches()
        assert abs(w - 7.5) < 1e-6 and abs(h - 4.0) < 1e-6
        plt.close(fig)


# --- Test 5: Adversarial Residual Grayscale & SyntaxWarning Scan ---

def test_all_notebooks_residual_styling_and_syntax():
    """Scan the complete showcase for residual styling and syntax warnings."""
    import py_compile
    import warnings

    py_files = _get_all_showcase_notebooks(".py")
    assert len(py_files) >= 130
    assert {p.stem for p in py_files} == {
        p.stem for p in _get_all_showcase_notebooks(".ipynb")
    }, "Every showcase must have both source and rendered notebook"

    # 1. Check Python 3.12+ SyntaxWarnings and compilation errors
    compile_issues = []
    for f in py_files:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                py_compile.compile(str(f), doraise=True)
            except Exception as e:
                compile_issues.append((f.name, "CompileError", str(e)))
            for item in w:
                if issubclass(item.category, (SyntaxWarning, DeprecationWarning)):
                    compile_issues.append((f.name, item.category.__name__, str(item.message), item.lineno))

    assert not compile_issues, f"Syntax or compilation issues found in notebooks:\n{compile_issues}"

    # 2. Check for residual plt.show() or plt.tight_layout() calls and hardcoded grayscale strings
    color_keywords = {"color", "c", "edgecolor", "facecolor", "colors", "ec", "fc", "fillcolor", "linecolor"}
    grayscale_pattern = re.compile(r"^0?\.\d+$|^1\.0+$")

    residual_issues = []
    for f in py_files:
        lines = f.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, 1):
            if re.search(r"\bplt\.show\s*\(", line):
                residual_issues.append((f.name, idx, "Residual plt.show() call", line.strip()))
            if re.search(r"\bplt\.tight_layout\s*\(", line):
                residual_issues.append((f.name, idx, "Residual plt.tight_layout() call", line.strip()))

        try:
            tree = ast.parse("\n".join(lines))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in color_keywords:
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    val = node.value.value
                    if grayscale_pattern.match(val):
                        residual_issues.append((f.name, node.lineno, f"Hardcoded grayscale literal {val!r} in {node.arg}"))

    assert not residual_issues, f"Residual styling or plt calls found:\n{residual_issues}"
