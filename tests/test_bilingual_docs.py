"""Bilingual parity: every English user-facing doc / showcase notebook has a
Spanish counterpart, and the English entry points carry the language switcher.

Pure file-existence + a header check — fast, no imports of the package, no network.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # package root (README.md, docs/, notebooks/)

# User-facing docs that get a docs/es/<same-name> Spanish version (exact case).
_USER_DOCS = [
    "index.md",
    "quickstart.md",
    "dsge_build.md",
    "models.md",
    "reporting.md",
    "var.md",
    "lp.md",
    "did.md",
    "nowcast.md",
    "climate.md",
    "forecast.md",
    "tablet.md",
    "benchmarks.md",
    "national_accounts.md",
    "real_time_data.md",
    "long_panel.md",
    "CREDENTIALS.md",
    "CACHE_DB.md",
    "CONNECTOR_HEALTH.md",
    "SIGNAL_CONTRACT.md",
    "examples_gallery.md",
    "1.0_path.md",
    "lexicon_review.md",
    "VALIDATION.md",
    "ADVISORY.md",
    "narrative_sign_svar.md",
    "honest_did.md",
    "smooth_lp.md",
    "hank_nonlinear.md",
    "gertler_karadi.md",
    "bvar_sv.md",
]


def test_readme_es_exists():
    assert (ROOT / "README.es.md").is_file(), "README.es.md missing"


def test_user_docs_have_spanish():
    missing = [d for d in _USER_DOCS if not (ROOT / "docs" / "es" / d).is_file()]
    assert not missing, f"missing docs/es/ counterparts: {missing}"


def test_user_docs_have_language_switchers():
    for doc in _USER_DOCS:
        en_file = ROOT / "docs" / doc
        if en_file.is_file():
            en_txt = en_file.read_text(encoding="utf-8")
            assert "Español" in en_txt[:600], f"Language switcher missing in docs/{doc}"

        es_file = ROOT / "docs" / "es" / doc
        if es_file.is_file():
            es_txt = es_file.read_text(encoding="utf-8")
            assert "English" in es_txt[:600], f"Language switcher missing in docs/es/{doc}"


def test_notebooks_have_spanish_sibling():
    nb = ROOT / "notebooks"
    english = [
        p for p in list(nb.glob("*.py")) + list((nb / "course").glob("*.py"))
        if not p.name.startswith("_") and not p.stem.endswith("_es")
    ]
    assert english, "no English notebook sources found"
    missing = [p.name for p in english if not (p.parent / f"{p.stem}_es.py").is_file()]
    assert not missing, f"notebooks lacking an _es sibling: {missing}"


def test_language_switcher_in_english_readme():
    txt = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Español" in txt[:600], "language switcher missing from the top of README.md"


def test_language_switcher_in_spanish_readme():
    txt = (ROOT / "README.es.md").read_text(encoding="utf-8")
    assert "English" in txt[:600], "language switcher missing from the top of README.es.md"

