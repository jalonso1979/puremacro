"""Guard the JOSS paper: structure, JOSS word-count range, citations resolve,
and the library-vs-users separation (no regime-uncertainty/equipment papers).
"""
import re
from pathlib import Path

import yaml

PAPER = Path(__file__).resolve().parents[1] / "paper" / "paper.md"
BIB = Path(__file__).resolve().parents[1] / "paper" / "paper.bib"


def _frontmatter_and_body():
    text = PAPER.read_text(encoding="utf-8")
    assert text.startswith("---"), "paper.md must start with YAML frontmatter"
    _, fm, body = text.split("---", 2)
    return yaml.safe_load(fm), body


def test_frontmatter_valid():
    fm, _ = _frontmatter_and_body()
    assert fm["title"] and fm["bibliography"] == "paper.bib"
    authors = fm["authors"]
    assert authors and any(a.get("corresponding") for a in authors)
    assert fm["affiliations"]


# JOSS asks for 750-1750 words (joss.readthedocs.io/en/latest/paper.html,
# checked 2026-08-20). This bound was 250-1000, which was the older guidance;
# re-check it if the paper is ever rejected on length.
JOSS_MIN_WORDS, JOSS_MAX_WORDS = 750, 1750


def test_word_count_in_joss_range():
    _, body = _frontmatter_and_body()
    body_wo_refs = re.split(r"#\s*References", body)[0]
    body_wo_fig = re.sub(r"!\[.*?\]\(.*?\)\{.*?\}", "", body_wo_refs, flags=re.S)
    # HTML comments are notes to the author, not prose JOSS will typeset.
    body_wo_notes = re.sub(r"<!--.*?-->", "", body_wo_fig, flags=re.S)
    words = len(re.findall(r"[A-Za-z0-9'-]+", body_wo_notes))
    assert JOSS_MIN_WORDS <= words <= JOSS_MAX_WORDS, (
        f"JOSS body word count {words} outside "
        f"{JOSS_MIN_WORDS}-{JOSS_MAX_WORDS}")


def test_citations_resolve():
    _, body = _frontmatter_and_body()
    cited = set(re.findall(r"@([A-Za-z0-9_]+)", body))
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
    assert cited, "paper cites no references"
    assert cited <= bibkeys, f"cited keys missing from bib: {cited - bibkeys}"


def test_library_users_separation():
    # The regime-uncertainty / equipment papers are downstream USERS, not part of
    # the library paper (spec §2/§8). The body must not reference them.
    _, body = _frontmatter_and_body()
    low = body.lower()
    for forbidden in ("regime uncertainty", "regime_uncertainty", "equipment paper"):
        assert forbidden not in low, f"paper references a downstream user paper: {forbidden!r}"
