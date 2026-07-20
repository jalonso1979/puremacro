"""Bilingual parity: every English user-facing doc / showcase notebook has a
Spanish counterpart, and the English entry points carry the language switcher.

Pure file-existence + a header check — fast, no imports of the package, no network.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # package root (README.md, docs/, notebooks/)

# User-facing docs that get a docs/es/<same-name> Spanish version (exact case).
_USER_DOCS = [
    "CREDENTIALS.md",
    "CACHE_DB.md",
    "CONNECTOR_HEALTH.md",
    "SIGNAL_CONTRACT.md",
    "examples_gallery.md",
    "1.0_path.md",
    "lexicon_review.md",
    "VALIDATION.md",
]


def test_readme_es_exists():
    assert (ROOT / "README.es.md").is_file(), "README.es.md missing"


def test_user_docs_have_spanish():
    missing = [d for d in _USER_DOCS if not (ROOT / "docs" / "es" / d).is_file()]
    assert not missing, f"missing docs/es/ counterparts: {missing}"


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
