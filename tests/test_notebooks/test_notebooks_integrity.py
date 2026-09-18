"""Automated static integrity and output verification for puremacro notebooks.

Verifies:
1. Valid JSON nbformat v4 structure across all committed notebooks.
2. Complete absence of cell error tracebacks (output_type != 'error').
3. Strict 100% opacity on all committed figure PNG alpha channels (min_alpha == 255).
4. _nbstyle.apply_style() presence, dual-theme support, and token availability
   in both notebooks/_nbstyle.py and curso/notebooks/_nbstyle.py.
5. Multi-suite discovery and validation tooling.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parents[2]
SHOWCASE_NB_DIR = PROJ_ROOT / "notebooks"
CURSO_NB_DIR = PROJ_ROOT / "curso" / "notebooks"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_all_committed_notebooks() -> list[Path]:
    showcase_nbs = [
        p for p in SHOWCASE_NB_DIR.glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ]
    curso_nbs = [
        p for p in CURSO_NB_DIR.glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ] if CURSO_NB_DIR.exists() else []
    return sorted(showcase_nbs + curso_nbs)


def test_notebook_json_structure():
    """Verify that all committed notebooks are valid Jupyter nbformat v4 JSON."""
    nbs = _get_all_committed_notebooks()
    assert len(nbs) >= 190, f"Expected >= 190 notebooks across suites, found {len(nbs)}"

    malformed: list[str] = []
    for path in nbs:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                malformed.append(f"{path.name}: root is not a dict")
                continue
            if data.get("nbformat", 0) < 4:
                malformed.append(f"{path.name}: nbformat {data.get('nbformat')} < 4")
            if "cells" not in data or not isinstance(data["cells"], list):
                malformed.append(f"{path.name}: missing or invalid 'cells' list")
        except Exception as e:
            malformed.append(f"{path.name}: JSON parse error: {e}")

    assert not malformed, f"Found malformed notebook JSON files:\n" + "\n".join(malformed)


def test_notebooks_no_cell_error_tracebacks():
    """Verify that no committed notebook contains an unhandled cell error traceback."""
    nbs = _get_all_committed_notebooks()
    errors_found: list[str] = []

    for path in nbs:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cell_idx, cell in enumerate(data.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            for out_idx, out in enumerate(cell.get("outputs", [])):
                if out.get("output_type") == "error":
                    ename = out.get("ename", "Error")
                    evalue = out.get("evalue", "")
                    errors_found.append(
                        f"{path.name} (cell {cell_idx}, output {out_idx}): {ename}: {evalue}"
                    )

    assert not errors_found, (
        f"Found cell execution error tracebacks in committed notebooks:\n"
        + "\n".join(errors_found)
    )


def test_notebooks_png_image_opacity():
    """Verify that all embedded PNG images have 100% opaque alpha channels (min_alpha == 255).

    Figures must never render transparent backgrounds that wash out in dark/light mode.
    """
    nbs = _get_all_committed_notebooks()
    transparent_images: list[str] = []
    total_images = 0

    for path in nbs:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cell_idx, cell in enumerate(data.get("cells", [])):
            for out_idx, out in enumerate(cell.get("outputs", [])):
                payloads = out.get("data", {})
                if "image/png" not in payloads:
                    continue
                total_images += 1
                raw = payloads["image/png"]
                if isinstance(raw, list):
                    raw = "".join(raw)
                try:
                    img = Image.open(io.BytesIO(base64.b64decode(raw)))
                    if img.mode == "RGBA":
                        arr = np.array(img)
                        min_alpha = int(arr[:, :, 3].min())
                        if min_alpha < 255:
                            transparent_images.append(
                                f"{path.name} (cell {cell_idx}, output {out_idx}): "
                                f"min_alpha={min_alpha} < 255"
                            )
                except Exception as ex:
                    transparent_images.append(
                        f"{path.name} (cell {cell_idx}, output {out_idx}): decode error: {ex}"
                    )

    assert total_images > 400, f"Expected >400 committed PNG images, found {total_images}"
    assert not transparent_images, (
        f"Found {len(transparent_images)} transparent PNG images:\n"
        + "\n".join(transparent_images)
    )


@pytest.mark.parametrize("mod_path", [
    SHOWCASE_NB_DIR / "_nbstyle.py",
    CURSO_NB_DIR / "_nbstyle.py",
])
def test_nbstyle_presence_and_token_availability(mod_path: Path):
    """Verify that _nbstyle provides the canonical card styling tokens and functions."""
    assert mod_path.exists(), f"Missing style module: {mod_path}"
    nbstyle = _load_module(mod_path, f"_test_{mod_path.parent.name}_nbstyle")

    # Core function presence
    for fn in ("apply_style", "figura", "etiquetar", "tono", "gris", "palette", "styles"):
        assert hasattr(nbstyle, fn), f"{mod_path.name} missing function: {fn}"
        assert callable(getattr(nbstyle, fn)), f"{mod_path.name} {fn} is not callable"

    # Semantic card tokens
    for token in ("FONDO", "TINTA", "TEXTO", "NOTA", "SPINE", "REJILLA", "S1", "S2", "S3", "S4", "S5", "S6"):
        assert hasattr(nbstyle, token), f"{mod_path.name} missing token: {token}"

    # Verify series configurations
    for i in range(1, 7):
        s = getattr(nbstyle, f"S{i}")
        assert isinstance(s, dict), f"S{i} must be a dict"
        assert "color" in s and "linestyle" in s and "linewidth" in s

    # Palette and styles scaling
    assert len(nbstyle.palette(3)) == 3
    assert len(nbstyle.palette(20)) == 20
    assert len(nbstyle.styles(2)) == 2
    assert len(nbstyle.styles(9)) == 9

    # Dual-theme card switching
    assert "grafito" in nbstyle.TEMAS and "papel" in nbstyle.TEMAS
    nbstyle.apply_style(theme="grafito")
    assert nbstyle.FONDO.upper() == "#161616"
    assert nbstyle.TINTA.upper() == "#EDEDED"

    nbstyle.apply_style(theme="papel")
    assert nbstyle.FONDO.upper() == "#FFFFFF"
    assert nbstyle.TINTA.upper() == "#141414"

    # Reset to default
    nbstyle.apply_style(theme="grafito")


def test_build_tooling_multi_suite_discovery():
    """Verify that tools/build_notebooks.py discovers showcase, curso, and all suites."""
    bn = _load_module(PROJ_ROOT / "tools" / "build_notebooks.py", "build_notebooks_tool")

    showcase_srcs = bn.discover_sources("showcase")
    assert all(p.suffix == ".py" for p in showcase_srcs)
    assert any("62_flexible_trade_cge.py" in p.name for p in showcase_srcs)
    assert any("local_llm_uncertainty.py" in p.name for p in showcase_srcs)

    curso_srcs = bn.discover_sources("curso")
    assert all(p.suffix == ".ipynb" for p in curso_srcs)
    assert len(curso_srcs) == 62

    all_srcs = bn.discover_sources("all")
    assert len(all_srcs) == len(showcase_srcs) + len(curso_srcs)
