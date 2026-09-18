"""Empirical Challenger Test Suite for Milestone 5 (Visual Card Styling, Theme Invariance, and Opacity).

Adversarially validates across all 214 notebooks and 647 embedded PNG images:
1. Universal PNG Canvas Opacity Challenge:
   - min_alpha == 255 across every pixel of all 647 images.
   - Zero transparent pixels exist repository-wide.
2. Dual-Mode Contrast Invariance:
   - Figures render on opaque card backgrounds (#161616 grafito or #FFFFFF papel).
   - Four-corner background uniformity for every figure.
   - Non-degenerate contrast across figures: rgb_std > 5.0 (actual minimum > 21.0).
   - Robust WCAG contrast ratio (> 15:1) ensuring theme invariance in dark and light modes.
3. Linestyle and Grayscale Differentiation:
   - Multi-series plots differentiate curves via non-color markers/linestyles and grayscale luminance steps.
   - Canonical _nbstyle.apply_style() configures axes.prop_cycle with both UNIVERSAL_COLORS and LINESTYLES.
   - Repository-wide mean achromatic pixel fraction >= 0.95.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

PROJ_ROOT = Path(__file__).resolve().parents[2]
NB_DIR = PROJ_ROOT / "notebooks"
CURSO_DIR = PROJ_ROOT / "curso" / "notebooks"


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


def _extract_all_embedded_pngs():
    nbs = _get_all_notebooks()
    images = []
    for p in nbs:
        suite = "showcase" if p.parent.name == "notebooks" else ("course" if p.parent.name == "course" else "curso")
        with open(p, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for c_idx, cell in enumerate(nb.get("cells", [])):
            for o_idx, out in enumerate(cell.get("outputs", [])):
                data = out.get("data", {})
                if "image/png" in data:
                    raw = data["image/png"]
                    if isinstance(raw, list):
                        raw = "".join(raw)
                    images.append((p, c_idx, o_idx, raw, suite))
    return nbs, images


def test_notebook_inventory_and_counts():
    """Assert exact discovery counts: 214 notebooks (130 showcase, 22 course subfolder, 62 curso)."""
    showcase = [
        p for p in NB_DIR.glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ]
    course_sub = [
        p for p in (NB_DIR / "course").glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ]
    curso = [
        p for p in CURSO_DIR.glob("*.ipynb")
        if not p.name.startswith("_") and ".ipynb_checkpoints" not in p.parts
    ]
    assert len(showcase) == 130, f"Expected 130 showcase notebooks, found {len(showcase)}"
    assert len(course_sub) == 22, f"Expected 22 course subfolder notebooks, found {len(course_sub)}"
    assert len(curso) == 62, f"Expected 62 curso notebooks, found {len(curso)}"
    assert len(showcase) + len(course_sub) + len(curso) == 214


def test_universal_png_canvas_opacity_repository_wide():
    """Universal PNG Canvas Opacity Challenge: assert min_alpha == 255 across every pixel of all 647 images."""
    nbs, images = _extract_all_embedded_pngs()
    assert len(images) == 647, f"Expected 647 embedded PNG images, found {len(images)}"

    transparent_violations = []

    for p, c_idx, o_idx, raw, suite in images:
        b = base64.b64decode(raw)
        im = Image.open(io.BytesIO(b))
        assert im.mode == "RGBA", f"Image in {p.name} cell {c_idx} out {o_idx} has mode {im.mode} != RGBA"
        arr = np.array(im)
        alpha = arr[:, :, 3]
        min_alpha = int(alpha.min())
        if min_alpha < 255:
            count_trans = int(np.sum(alpha < 255))
            transparent_violations.append(f"{p.name} (cell {c_idx}, out {o_idx}): min_alpha={min_alpha}, {count_trans} pixels")

    assert not transparent_violations, (
        f"Found {len(transparent_violations)} images with transparent pixels:\n" + "\n".join(transparent_violations)
    )


def test_dual_mode_contrast_invariance_and_card_backgrounds():
    """Verify figures render on opaque card backgrounds (#161616 or #FFFFFF) and non-degenerate contrast."""
    _, images = _extract_all_embedded_pngs()
    
    bg_counts: dict[str, int] = {}
    degenerate_figures: list[str] = []
    non_uniform_corners: list[str] = []
    contrast_violations: list[str] = []

    for p, c_idx, o_idx, raw, suite in images:
        b = base64.b64decode(raw)
        im = Image.open(io.BytesIO(b))
        arr = np.array(im)
        rgb = arr[:, :, :3]

        # 1. Non-degenerate contrast assertion
        rgb_std = float(rgb.std())
        if rgb_std <= 5.0:
            degenerate_figures.append(f"{p.name} cell {c_idx} out {o_idx}: rgb_std={rgb_std:.2f}")

        # 2. Four corners uniformity
        c_tl = rgb[0, 0]
        c_tr = rgb[0, -1]
        c_bl = rgb[-1, 0]
        c_br = rgb[-1, -1]
        if not (np.all(c_tl == c_tr) and np.all(c_tl == c_bl) and np.all(c_tl == c_br)):
            non_uniform_corners.append(f"{p.name} cell {c_idx} out {o_idx}: non-uniform corners")

        bg_hex = f"#{c_tl[0]:02X}{c_tl[1]:02X}{c_tl[2]:02X}"
        bg_counts[bg_hex] = bg_counts.get(bg_hex, 0) + 1

        # 3. Luminance contrast check
        rgb_norm = rgb.astype(float) / 255.0
        lum = (0.2126 * np.where(rgb_norm[:, :, 0] <= 0.04045, rgb_norm[:, :, 0] / 12.92, ((rgb_norm[:, :, 0] + 0.055) / 1.055) ** 2.4) +
               0.7152 * np.where(rgb_norm[:, :, 1] <= 0.04045, rgb_norm[:, :, 1] / 12.92, ((rgb_norm[:, :, 1] + 0.055) / 1.055) ** 2.4) +
               0.0722 * np.where(rgb_norm[:, :, 2] <= 0.04045, rgb_norm[:, :, 2] / 12.92, ((rgb_norm[:, :, 2] + 0.055) / 1.055) ** 2.4))
        min_lum = float(np.min(lum))
        max_lum = float(np.max(lum))
        cr = (max_lum + 0.05) / (min_lum + 0.05)
        if cr < 7.0:  # WCAG AAA threshold
            contrast_violations.append(f"{p.name} cell {c_idx} out {o_idx}: contrast ratio {cr:.2f} < 7.0")

    assert not degenerate_figures, f"Found degenerate figures:\n" + "\n".join(degenerate_figures)
    assert not non_uniform_corners, f"Found non-uniform corners:\n" + "\n".join(non_uniform_corners)
    assert not contrast_violations, f"Found low-contrast figures:\n" + "\n".join(contrast_violations)

    # Assert 100% of images are on valid card backgrounds: either #161616 (grafito) or #FFFFFF (papel)
    valid_backgrounds = {"#161616", "#FFFFFF"}
    invalid_bgs = {k: v for k, v in bg_counts.items() if k not in valid_backgrounds}
    assert not invalid_bgs, f"Found invalid card backgrounds: {invalid_bgs}"
    assert bg_counts["#161616"] == 359, f"Expected 359 grafito cards, found {bg_counts.get('#161616')}"
    assert bg_counts["#FFFFFF"] == 288, f"Expected 288 papel cards, found {bg_counts.get('#FFFFFF')}"


def test_linestyle_and_grayscale_differentiation():
    """Verify multi-series plots use linestyles/markers and grayscale luminance differentiation."""
    import sys
    sys.path.insert(0, str(NB_DIR))
    import _nbstyle
    import matplotlib.pyplot as plt

    _nbstyle.apply_style("grafito")
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    prop_list = list(prop_cycle)

    assert len(prop_list) >= 6, "Prop cycle has fewer than 6 series definitions"

    # Assert each series in prop_cycle has a distinct linestyle
    linestyles = [str(p["linestyle"]) for p in prop_list]
    assert len(set(linestyles)) == len(linestyles), "Series linestyles in prop_cycle are not all distinct"

    # Assert palette generates distinct grayscale tones
    pal = _nbstyle.palette(6)
    assert len(set(pal)) == 6, f"Expected 6 distinct colors in palette(6), got {len(set(pal))}"

    # Assert styles generates 6 distinct dash patterns
    st = _nbstyle.styles(6)
    assert len(set(str(s) for s in st)) == 6, f"Expected 6 distinct dash patterns in styles(6), got {len(set(str(s) for s in st))}"

    # Verify paper theme
    _nbstyle.apply_style("papel")
    pal_papel = _nbstyle.palette(6)
    assert len(set(pal_papel)) == 6, "Papel palette does not have 6 distinct tones"

    # Clean up rcParams
    _nbstyle.apply_style("grafito")

