"""Regression tests verifying 6.5.0 status and validation meta-docs stay in sync."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

from puremacro.validation.runner import _discover_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
ES_DOCS_DIR = DOCS_DIR / "es"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_roadmap_docs_sync_to_6_5_0() -> None:
    en = _read(DOCS_DIR / "1.0_path.md")
    es = _read(ES_DOCS_DIR / "1.0_path.md")

    for text in (en, es):
        assert "6.5.0" in text
        assert "2026-11-14" in text
        assert "2026-11-03" not in text
        assert "6.1.0 is the" not in text
        assert "6.1.0 es" not in text
        for marker in (
            "EquilibriumSystem",
            "linearize_equilibrium",
            "continue_equilibrium",
            "fit_simulated_moments",
            "identification_robust_inference.md",
            "solve_hank_second_order",
            "puremacro.models.hank_fiscal",
            "solve_hank_fiscal",
            "compare_fiscal_closures",
            "first_brief_fiscal.md",
            "first_brief_tariffs.md",
            "connector_freshness_report",
            "CONNECTOR_HEALTH.md",
            "run_empirical_structural_workbench",
            "workbench_and_scorecards.md",
            "puremacro.mcp_server",
            "mcp_server.md",
            "puremacro.lp.cce",
            'estimator="mean_group"',
            "cce_mg_panel_lp",
            "run_scenario_batch",
            "puremacro.sovereign.smm",
            "RELEASING.md",
        ):
            assert marker in text

    for heading in ("## Development priorities", "## Compatibility policy", "## Release evidence"):
        assert heading in en
    for heading in (
        "## Prioridades de desarrollo",
        "## Política de compatibilidad",
        "## Evidencia para publicar",
    ):
        assert heading in es
    assert "experimental" in en.lower()
    assert "experimental" in es.lower()


def test_structural_validation_status_examples_gallery_count() -> None:
    gallery = json.loads((DOCS_DIR / "examples_gallery.json").read_text(encoding="utf-8"))
    examples = gallery["examples"]
    total = len(examples)
    entries = examples.values() if isinstance(examples, dict) else examples
    counts = Counter(entry["status"] for entry in entries)
    assert total > 0

    en = _read(DOCS_DIR / "STRUCTURAL_VALIDATION_STATUS.md")
    es = _read(ES_DOCS_DIR / "STRUCTURAL_VALIDATION_STATUS.md")

    assert re.search(rf"examples_gallery\.json.*?{total}\s+recorded example checks", en, re.S)
    assert f"{counts['PASS']} passing" in en
    assert f"{counts['SKIP']} skipped" in en
    assert f"{counts['FAIL']} documented failure" in en
    assert "romer_romer_2010_replication" in en

    assert re.search(rf"examples_gallery\.json.*?{total}\s+comprobaciones", es, re.S)
    assert f"{counts['PASS']} superadas" in es
    assert f"{counts['SKIP']} omitidas" in es
    assert f"{counts['FAIL']} fallo documentado" in es
    assert "romer_romer_2010_replication" in es


def test_structural_validation_status_validation_gallery_count() -> None:
    cases = _discover_cases()
    total = len(cases)
    subsystems = len({case.subsystem for case in cases})
    by_mech = Counter(
        case.mechanism.value if hasattr(case.mechanism, "value") else str(case.mechanism)
        for case in cases
    )

    en = _read(DOCS_DIR / "STRUCTURAL_VALIDATION_STATUS.md")
    es = _read(ES_DOCS_DIR / "STRUCTURAL_VALIDATION_STATUS.md")

    assert "113-case" not in en
    assert "62 internal" not in en
    assert "13 package" not in en
    assert f"{total}-case validation gallery ({subsystems} subsystems" in en
    assert f"{by_mech['internal']} internal" in en
    assert f"{by_mech['analytical']} analytical" in en
    assert f"{by_mech['package']} package" in en

    assert "113 casos" not in es
    assert "13 de paquetes" not in es
    assert not re.search(r"\b62\s+casos internos\b", es)
    assert f"galería de validación de {total} casos ({subsystems} subsistemas" in es
    assert re.search(rf"\b{by_mech['internal']}\s+casos internos\b", es)
    assert f"{by_mech['analytical']} analíticos" in es
    assert f"{by_mech['package']} de paquetes" in es


def test_structural_validation_status_new_6_5_0_rows() -> None:
    en = _read(DOCS_DIR / "STRUCTURAL_VALIDATION_STATUS.md")
    es = _read(ES_DOCS_DIR / "STRUCTURAL_VALIDATION_STATUS.md")

    required_tokens = (
        "6.5.0",
        "puremacro.models.hank_fiscal",
        "solve_hank_fiscal",
        "compare_fiscal_closures",
        "balanced_budget",
        "debt_rule",
        "boundary_asset_truncation",
        "puremacro.workbench",
        "run_empirical_structural_workbench",
        "EmpiricalStructuralDossier",
        "VALID_OVERALL_STATUSES",
        "determine_overall_status",
        "lp_moment_targets",
        "puremacro.trade.scenarios",
        "CompositeTariffScenario",
        "run_scenario_batch",
        "summarize_scenario_batch",
        "welfare_status",
        "puremacro.sovereign.smm",
        "_stationary_access_exclusion_dist",
        "boundary_choice_share_pct",
        "puremacro.lp.cce",
        "cce_mg_panel_lp",
        'estimator="mean_group"',
        "pesaran_mg",
        "excluded_entities_by_horizon",
        "_MIN_MG_ENTITIES",
        "allow_index_gaps",
    )

    for text in (en, es):
        for token in required_tokens:
            assert token in text


def test_bilingual_switchers_preserved() -> None:
    assert "es/1.0_path.md" in _read(DOCS_DIR / "1.0_path.md")[:600]
    assert "../1.0_path.md" in _read(ES_DOCS_DIR / "1.0_path.md")[:600]
    assert "es/STRUCTURAL_VALIDATION_STATUS.md" in _read(
        DOCS_DIR / "STRUCTURAL_VALIDATION_STATUS.md"
    )[:600]
    assert "../STRUCTURAL_VALIDATION_STATUS.md" in _read(
        ES_DOCS_DIR / "STRUCTURAL_VALIDATION_STATUS.md"
    )[:600]
