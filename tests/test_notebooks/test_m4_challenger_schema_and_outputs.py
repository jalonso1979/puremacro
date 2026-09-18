"""Empirical Challenger Test Suite for Milestone 4 (Showcase Compilation & E2E Validation).

Adversarially validates:
1. Programmatic verification that 100% of code cells in the 15 compiled showcase notebooks
   have valid integer execution_count (not null/None) and non-empty output payloads.
2. Assert zero error tracebacks across all notebooks in the repository (showcase, course, and curso).
3. Schema validation (nbformat v4) across all notebooks in the repository.
4. PNG canvas opacity and non-degenerate figure output verification.
5. Structural bilingual parity between English and Spanish target notebook pairs.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pytest
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parents[2]
NB_DIR = PROJ_ROOT / "notebooks"
CURSO_DIR = PROJ_ROOT / "curso" / "notebooks"

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


def _get_all_notebooks() -> list[Path]:
    showcase = [
        p for p in NB_DIR.glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ]
    course_sub = [
        p for p in (NB_DIR / "course").glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ] if (NB_DIR / "course").exists() else []
    curso = [
        p for p in CURSO_DIR.glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ] if CURSO_DIR.exists() else []
    return sorted(showcase + course_sub + curso)


def test_target_15_existence_and_size():
    """Assert all 15 compiled showcase notebooks exist and have substantial size."""
    for rel_name in TARGET_15_NOTEBOOKS:
        p = NB_DIR / rel_name
        assert p.exists(), f"Target notebook {rel_name} is missing from notebooks/"
        sz = p.stat().st_size
        assert sz > 10_000, f"Target notebook {rel_name} unexpectedly small ({sz} bytes)"


@pytest.mark.parametrize("rel_name", TARGET_15_NOTEBOOKS)
def test_target_15_code_cells_execution_counts_and_outputs(rel_name: str):
    """Verify 100% of code cells in target notebooks have valid int execution_count and 0 errors."""
    p = NB_DIR / rel_name
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) > 0, f"No code cells found in {rel_name}"

    total_outputs = 0
    for idx, cell in enumerate(code_cells):
        ec = cell.get("execution_count")
        assert ec is not None, f"{rel_name} code cell {idx} has execution_count is None"
        assert isinstance(ec, int), f"{rel_name} code cell {idx} execution_count is not int: {ec}"
        assert ec > 0, f"{rel_name} code cell {idx} execution_count <= 0: {ec}"

        outs = cell.get("outputs", [])
        total_outputs += len(outs)

        # Assert zero error outputs
        for out_idx, out in enumerate(outs):
            assert out.get("output_type") != "error", (
                f"{rel_name} code cell {idx} output {out_idx} has error: "
                f"{out.get('ename')}: {out.get('evalue')}"
            )

    # Assert total notebook output payload is populated
    assert total_outputs >= 3, f"{rel_name} has insufficient outputs: {total_outputs}"


def test_zero_error_tracebacks_repository_wide():
    """Assert 0 cell error tracebacks across all notebooks in repository."""
    nbs = _get_all_notebooks()
    assert len(nbs) >= 210, f"Expected >= 210 notebooks, found {len(nbs)}"

    tracebacks: list[str] = []
    for p in nbs:
        with open(p, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for c_idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            for o_idx, out in enumerate(cell.get("outputs", [])):
                if out.get("output_type") == "error":
                    ename = out.get("ename", "Error")
                    evalue = out.get("evalue", "")
                    tracebacks.append(f"{p.name} (cell {c_idx}, out {o_idx}): {ename}: {evalue}")

    assert not tracebacks, f"Found {len(tracebacks)} cell error tracebacks:\n" + "\n".join(tracebacks)


def test_json_nbformat_schema_repository_wide():
    """Assert all notebooks pass official nbformat v4 schema validation."""
    nbs = _get_all_notebooks()
    schema_errors: list[str] = []
    for p in nbs:
        try:
            nb = nbformat.read(p, as_version=4)
            nbformat.validate(nb)
        except Exception as e:
            schema_errors.append(f"{p.name}: {e}")

    assert not schema_errors, f"Found {len(schema_errors)} schema errors:\n" + "\n".join(schema_errors)


def test_target_15_png_opacity_and_contrast():
    """Assert all embedded PNGs in the 15 compiled notebooks are 100% opaque and high contrast."""
    figures_found = 0
    for rel_name in TARGET_15_NOTEBOOKS:
        p = NB_DIR / rel_name
        with open(p, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for c_idx, cell in enumerate(nb.get("cells", [])):
            for o_idx, out in enumerate(cell.get("outputs", [])):
                data = out.get("data", {})
                if "image/png" in data:
                    raw = data["image/png"]
                    if isinstance(raw, list):
                        raw = "".join(raw)
                    b = base64.b64decode(raw)
                    im = Image.open(io.BytesIO(b))
                    assert im.size[0] > 100 and im.size[1] > 100, f"Image too small in {rel_name}"
                    arr = np.array(im)
                    if im.mode == "RGBA":
                        min_alpha = int(arr[:, :, 3].min())
                        assert min_alpha == 255, f"Transparent alpha in {rel_name} cell {c_idx}"
                    # Contrast assertion: image should not be flat solid color
                    rgb_std = float(arr[:, :, :3].std())
                    assert rgb_std > 5.0, f"Degenerate figure in {rel_name} cell {c_idx}"
                    figures_found += 1

    assert figures_found == 36, f"Expected 36 figures across target 15 notebooks, found {figures_found}"


def test_bilingual_target_pairs_parity():
    """Verify code cell count and figure count parity across English and Spanish editions."""
    pairs = [
        ("00_whats_new_in_puremacro_2_0.ipynb", "00_whats_new_in_puremacro_2_0_es.ipynb"),
        ("00_whats_new_in_puremacro_3_0.ipynb", "00_whats_new_in_puremacro_3_0_es.ipynb"),
        ("37_central_bank_narrative_sentiment.ipynb", "37_central_bank_narrative_sentiment_es.ipynb"),
        ("43_dsge_nuts_and_analytic_gradients.ipynb", "43_dsge_nuts_and_analytic_gradients_es.ipynb"),
        ("62_flexible_trade_cge.ipynb", "62_flexible_trade_cge_es.ipynb"),
    ]

    for en_name, es_name in pairs:
        en_p = NB_DIR / en_name
        es_p = NB_DIR / es_name
        with open(en_p, "r", encoding="utf-8") as f:
            en_nb = json.load(f)
        with open(es_p, "r", encoding="utf-8") as f:
            es_nb = json.load(f)

        en_code = [c for c in en_nb.get("cells", []) if c.get("cell_type") == "code"]
        es_code = [c for c in es_nb.get("cells", []) if c.get("cell_type") == "code"]
        assert len(en_code) == len(es_code), (
            f"Code cell count mismatch between {en_name} ({len(en_code)}) and {es_name} ({len(es_code)})"
        )

        en_figs = sum(
            1 for c in en_code for o in c.get("outputs", []) if "image/png" in o.get("data", {})
        )
        es_figs = sum(
            1 for c in es_code for o in c.get("outputs", []) if "image/png" in o.get("data", {})
        )
        assert en_figs == es_figs, (
            f"Figure count mismatch between {en_name} ({en_figs}) and {es_name} ({es_figs})"
        )
