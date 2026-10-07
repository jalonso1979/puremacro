"""Empirical Challenger Test Suite for Milestone 1 (Showcase Cohort A: 00 through 18).

Adversarially validates:
1. 100% of all 40 committed showcase notebook artifacts in Cohort A (00-18, both EN and ES)
   exist, parse as valid JSON, and pass strict nbformat v4 schema validation.
2. 100% of code cells have valid integer execution_count (not null/None) with execution_count > 0,
   forming a contiguous strictly increasing sequence [1, 2, ..., N].
3. Assert zero cell error tracebacks and zero unhandled stream exception tracebacks.
4. Total outputs >= 3 in every notebook.
5. All embedded PNG figures have 100% opaque alpha channels (min_alpha == 255) and high contrast (rgb_std > 5.0).
6. Exact bilingual parity: len(en_code_cells) == len(es_code_cells) and en_figures == es_figures across all 20 pairs.
"""
from __future__ import annotations

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

PROJ_ROOT = Path(__file__).resolve().parents[2]
NB_DIR = PROJ_ROOT / "notebooks"

COHORT_A_STEMS = [
    "00_whats_new_in_puremacro_2_0",
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

COHORT_A_PAIRS = [(f"{stem}.ipynb", f"{stem}_es.ipynb") for stem in COHORT_A_STEMS]
COHORT_A_ALL_NOTEBOOKS = [nb for pair in COHORT_A_PAIRS for nb in pair]


def test_cohort_a_all_40_notebooks_exist_and_size():
    """Assert all 40 Cohort A showcase notebooks exist and have substantial file size (> 10KB)."""
    assert len(COHORT_A_ALL_NOTEBOOKS) == 40
    for rel_name in COHORT_A_ALL_NOTEBOOKS:
        p = NB_DIR / rel_name
        assert p.exists(), f"Cohort A notebook {rel_name} is missing from notebooks/"
        sz = p.stat().st_size
        assert sz > 10_000, f"Cohort A notebook {rel_name} unexpectedly small ({sz} bytes)"


@pytest.mark.parametrize("rel_name", COHORT_A_ALL_NOTEBOOKS)
def test_cohort_a_json_and_nbformat_v4_schema(rel_name: str):
    """Verify strict nbformat v4 schema compliance for each notebook."""
    p = NB_DIR / rel_name
    raw_text = p.read_text(encoding="utf-8")

    # JSON validity
    data = json.loads(raw_text)
    assert isinstance(data, dict), f"{rel_name} root is not a dict"
    assert data.get("nbformat", 0) >= 4, f"{rel_name} nbformat < 4"
    assert "cells" in data and isinstance(data["cells"], list), f"{rel_name} missing 'cells' list"

    # Official nbformat validator
    nb_node = nbformat.reads(raw_text, as_version=4)
    nbformat.validate(nb_node)


@pytest.mark.parametrize("rel_name", COHORT_A_ALL_NOTEBOOKS)
def test_cohort_a_code_cells_execution_counts_and_contiguity(rel_name: str):
    """Verify 100% of code cells have valid int execution_count > 0 in strictly contiguous sequence."""
    p = NB_DIR / rel_name
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    assert len(code_cells) > 0, f"No code cells in {rel_name}"

    exec_counts: list[int] = []
    for idx, cell in enumerate(code_cells):
        ec = cell.get("execution_count")
        assert ec is not None, f"{rel_name} code cell {idx} has execution_count None"
        assert isinstance(ec, int), f"{rel_name} code cell {idx} execution_count not int: {ec}"
        assert ec > 0, f"{rel_name} code cell {idx} execution_count <= 0: {ec}"
        exec_counts.append(ec)

    # Assert contiguous sequence starting at 1
    expected_sequence = list(range(1, len(code_cells) + 1))
    assert exec_counts == expected_sequence, (
        f"{rel_name} non-contiguous execution counts: {exec_counts} != {expected_sequence}"
    )


@pytest.mark.parametrize("rel_name", COHORT_A_ALL_NOTEBOOKS)
def test_cohort_a_zero_error_tracebacks_and_outputs(rel_name: str):
    """Verify zero cell error tracebacks and non-empty output arrays."""
    p = NB_DIR / rel_name
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    total_outputs = 0

    for idx, cell in enumerate(code_cells):
        outs = cell.get("outputs", [])
        assert isinstance(outs, list), f"{rel_name} code cell {idx} outputs is not a list"
        total_outputs += len(outs)

        for out_idx, out in enumerate(outs):
            assert out.get("output_type") != "error", (
                f"{rel_name} cell {idx} out {out_idx} has error: "
                f"{out.get('ename')}: {out.get('evalue')}"
            )
            if out.get("output_type") == "stream":
                text = out.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                assert "Traceback (most recent call last)" not in text, (
                    f"{rel_name} cell {idx} out {out_idx} stream contains Traceback"
                )

    assert total_outputs >= 3, f"{rel_name} has insufficient outputs: {total_outputs} < 3"


@pytest.mark.parametrize("rel_name", COHORT_A_ALL_NOTEBOOKS)
def test_cohort_a_png_opacity_and_contrast(rel_name: str):
    """Verify all embedded PNG figures have 100% opaque alpha (min_alpha == 255) and high contrast."""
    p = NB_DIR / rel_name
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for out_idx, out in enumerate(cell.get("outputs", [])):
            data = out.get("data", {})
            if "image/png" in data:
                raw = data["image/png"]
                if isinstance(raw, list):
                    raw = "".join(raw)
                b = base64.b64decode(raw)
                assert b.startswith(b"\x89PNG\r\n\x1a\n"), f"{rel_name} cell {idx} invalid PNG magic"
                im = Image.open(io.BytesIO(b))
                assert im.size[0] > 100 and im.size[1] > 100, f"{rel_name} cell {idx} image too small: {im.size}"

                arr = np.array(im)
                if im.mode == "RGBA":
                    min_alpha = int(arr[:, :, 3].min())
                    assert min_alpha == 255, (
                        f"{rel_name} cell {idx} out {out_idx} transparent canvas (min_alpha={min_alpha} < 255)"
                    )

                rgb_std = float(arr[:, :, :3].std())
                assert rgb_std > 5.0, f"{rel_name} cell {idx} degenerate figure (rgb_std={rgb_std:.2f} <= 5.0)"


@pytest.mark.parametrize("en_name, es_name", COHORT_A_PAIRS)
def test_cohort_a_bilingual_pair_cell_and_figure_parity(en_name: str, es_name: str):
    """Verify exact code cell count and figure count parity across English and Spanish editions."""
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
