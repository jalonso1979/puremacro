"""Verification suite for PureMacro v6.0.0 documentation completeness and bilingual parity."""
from __future__ import annotations

import ast
from datetime import date
from pathlib import Path
import re
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"

V6_GUIDE_PAIRS = [
    ("autodiff_and_protocol.md", "es/autodiff_and_protocol.md"),
    ("smc_inference.md", "es/smc_inference.md"),
    ("hank_ssj_and_aggregate_risk.md", "es/hank_ssj_and_aggregate_risk.md"),
    ("sovereign_default.md", "es/sovereign_default.md"),
    ("distributional_lp.md", "es/distributional_lp.md"),
    ("network_spatial_climate_iam.md", "es/network_spatial_climate_iam.md"),
]
ALL_V6_GUIDES = [rel for pair in V6_GUIDE_PAIRS for rel in pair]


def test_v6_guide_pairs_exist():
    for en_rel, es_rel in V6_GUIDE_PAIRS:
        assert (DOCS_DIR / en_rel).is_file(), f"Missing English guide docs/{en_rel}"
        assert (DOCS_DIR / es_rel).is_file(), f"Missing Spanish guide docs/{es_rel}"


def test_v6_guides_wired_in_mkdocs_nav():
    mkdocs_txt = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "\nnav:" in mkdocs_txt
    nav_part = mkdocs_txt.split("\nnav:", 1)[1]

    # Split into English and Spanish nav sections
    assert "English Documentation:" in nav_part
    assert "Documentación en Español:" in nav_part
    en_nav = nav_part.split("English Documentation:", 1)[1].split("Documentación en Español:", 1)[0]
    es_nav = nav_part.split("Documentación en Español:", 1)[1]

    for en_rel, es_rel in V6_GUIDE_PAIRS:
        assert en_rel in en_nav, f"docs/{en_rel} not listed under English Documentation in mkdocs.yml"
        assert es_rel in es_nav, f"docs/{es_rel} not listed under Documentación en Español in mkdocs.yml"


@pytest.mark.parametrize("rel_path", ALL_V6_GUIDES + ["index.md", "es/index.md"])
def test_v6_guides_contain_mermaid_diagrams(rel_path):
    path = DOCS_DIR / rel_path
    text = path.read_text(encoding="utf-8")
    mermaid_blocks = re.findall(r"```mermaid\s*\n(.*?)```", text, flags=re.DOTALL)
    assert len(mermaid_blocks) >= 1, f"docs/{rel_path} must contain at least 1 Mermaid diagram"
    first = mermaid_blocks[0].strip()
    valid_starters = ("graph", "flowchart", "sequenceDiagram", "classDiagram", "stateDiagram")
    assert any(first.startswith(s) for s in valid_starters), f"docs/{rel_path} has invalid mermaid syntax"


@pytest.mark.parametrize("rel_path", ALL_V6_GUIDES)
def test_v6_guides_contain_parameter_reference_tables(rel_path):
    path = DOCS_DIR / rel_path
    text = path.read_text(encoding="utf-8")
    # Match markdown table header followed by separator line
    table_pattern = re.compile(r"\|[^\n]+\|\n\|\s*(?::?---+:?\s*\|)+\n((?:\|[^\n]+\|\n?)+)")
    matches = table_pattern.findall(text)
    assert len(matches) >= 1, f"docs/{rel_path} must contain at least 1 parameter reference table"
    # Ensure at least 3 rows in the table
    rows = [r for r in matches[0].strip().splitlines() if r.strip()]
    assert len(rows) >= 3, f"docs/{rel_path} parameter table has fewer than 3 rows"


@pytest.mark.parametrize("rel_path", ALL_V6_GUIDES)
def test_v6_guides_contain_executable_python_blocks(rel_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    path = DOCS_DIR / rel_path
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) >= 1, f"docs/{rel_path} must contain at least 1 python code block"

    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(path),
    }

    for idx, block in enumerate(blocks):
        try:
            ast.parse(block)
        except SyntaxError as e:
            pytest.fail(f"docs/{rel_path} block {idx} has SyntaxError: {e}")
        try:
            exec(block, namespace)
        except Exception as e:
            pytest.fail(f"docs/{rel_path} block {idx} failed execution: {e}")
        finally:
            plt.close("all")


@pytest.mark.parametrize("rel_path", ALL_V6_GUIDES)
def test_v6_guides_have_language_switchers(rel_path):
    path = DOCS_DIR / rel_path
    header = path.read_text(encoding="utf-8")[:600]
    if rel_path.startswith("es/"):
        assert "English" in header, f"docs/{rel_path} missing English language switcher in header"
    else:
        assert "Español" in header, f"docs/{rel_path} missing Spanish language switcher in header"


def test_v6_frontdoors_and_catalogs_synchronized():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    for rel_path in ("README.md", "README.es.md", "docs/index.md", "docs/es/index.md"):
        txt = (ROOT / rel_path).read_text(encoding="utf-8")
        notices = [line for line in txt.splitlines() if line.startswith((
            "**Version guide:", "**Guía de versiones:",
            "**Status checked ", "**Estado comprobado ",
        ))]
        assert len(notices) == 1, f"{rel_path} must have one explicit version notice"
        notice = notices[0]
        assert f"**{version}**" in notice, f"{rel_path} does not match the source version"
        if notice.startswith(("**Version guide:", "**Guía de versiones:")):
            # Source documentation may precede PyPI publication. Do not require
            # an unavailable version link or manufacture a publication date.
            assert "[PyPI](https://pypi.org/project/puremacro/)" in notice
        else:
            checked = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", notice)
            assert checked is not None, f"{rel_path} needs a publication-check date"
            date.fromisoformat(checked.group(1))
            assert f"[PyPI](https://pypi.org/project/puremacro/{version}/)" in notice

    for idx_file in ("index.md", "es/index.md"):
        txt = (DOCS_DIR / idx_file).read_text(encoding="utf-8")
        assert "6.0.0" in txt, f"docs/{idx_file} does not mention v6.0.0"
        assert "https://pypi.org/project/puremacro/6.0.0/" not in txt

    for nb_cat in ("notebooks.md", "es/notebooks.md"):
        txt = (DOCS_DIR / nb_cat).read_text(encoding="utf-8")
        for num in ("73", "74", "75"):
            assert num in txt, f"docs/{nb_cat} missing catalog entry for notebook {num}"

    for readme in ("README.md", "README.es.md"):
        txt = (ROOT / readme).read_text(encoding="utf-8")
        assert "6.0.0" in txt, f"{readme} does not mention v6.0.0"
        for num in ("73", "74", "75"):
            assert num in txt, f"{readme} missing mention of notebook {num}"
