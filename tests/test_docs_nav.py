"""The mkdocs nav must point at files that exist.

`mkdocs build --strict` catches this, but mkdocs is not a dev dependency and
the docs site was built nowhere at all until CI grew a step for it — which is
how the nav came to carry five entries pointing at pages that had never been
written. This test needs nothing but the standard library, so it runs in the
default offline suite and fails within a second of the drift appearing.

Parsing mkdocs.yml with `yaml.safe_load` is not an option: the file uses
`!!python/name:` tags for the superfences custom fence, which SafeLoader
refuses. The nav is a flat list of `Label: path.md` lines, so it is read with a
regex rather than by pulling in an unsafe loader to read a config file.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MKDOCS = _ROOT / "mkdocs.yml"
_DOCS = _ROOT / "docs"

_NAV_ENTRY = re.compile(r":\s*([A-Za-z0-9_./-]+\.md)\s*$", re.M)


def _nav_targets() -> list[str]:
    text = _MKDOCS.read_text(encoding="utf-8")
    assert "\nnav:" in text, "mkdocs.yml has no nav block"
    return _NAV_ENTRY.findall(text.split("\nnav:", 1)[1])


def test_the_nav_block_is_actually_being_read():
    """Positive control: if the regex stops matching, everything below passes
    vacuously — which is the failure mode this file exists to prevent."""
    targets = _nav_targets()
    assert len(targets) >= 10, targets
    assert "index.md" in targets


@pytest.mark.parametrize("target", _nav_targets())
def test_every_nav_entry_resolves_to_a_file(target):
    path = _DOCS / target
    assert path.is_file(), (
        f"mkdocs.yml nav points at docs/{target}, which does not exist. "
        f"`mkdocs build --strict` fails on this, and the rendered site shows a "
        f"section that 404s. Write the page or drop the nav entry."
    )


def test_no_nav_entry_is_listed_twice():
    targets = _nav_targets()
    dupes = {t for t in targets if targets.count(t) > 1}
    assert not dupes, f"duplicated nav entries: {sorted(dupes)}"
