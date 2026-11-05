"""Regenerate the fiscal brief's frozen record, its bilingual pages and their figures.

Run from the repository root::

    python -m tools.reference_validation.refresh_fiscal_brief --output /tmp/fiscal-brief

The brief runs once with Ramey and Zubairy's World War II exclusion (``WWII``), and
its outputs are written to ``OUTPUT/en`` and, in Spanish, to ``OUTPUT/es``. Then:

- ``headline.json`` is copied to ``tests/fixtures/first_brief_fiscal/``;
- the four marker blocks of ``docs/first_brief_fiscal.md`` and
  ``docs/es/first_brief_fiscal.md`` are rewritten from the record;
- the two ``multipliers.png`` are copied to ``docs/assets/first_brief_fiscal/``.

A missing, duplicate or reversed marker in either page raises ``ValueError``
before anything in the repository is written.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from puremacro.examples.first_brief_fiscal import (
    WWII,
    page_block,
    run_brief,
    summary_text,
    write_outputs,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "first_brief_fiscal"
ASSETS = ROOT / "docs" / "assets" / "first_brief_fiscal"
PAGES = {"en": ROOT / "docs" / "first_brief_fiscal.md",
         "es": ROOT / "docs" / "es" / "first_brief_fiscal.md"}
BLOCKS = ("headline", "headline_reading", "experiment", "experiment_reading")


def fill_blocks(text: str, record: dict[str, Any], lang: str, page: str = "page") -> str:
    """``text`` with each marker block replaced by ``page_block(record, block, lang)``.

    A block is ``<!-- fiscal-brief:{block} -->``, the generated text on the lines
    between, and ``<!-- /fiscal-brief:{block} -->``. Raises ``ValueError`` naming
    ``page`` when a block's markers are missing, duplicated or reversed.
    """
    for block in BLOCKS:
        start, end = f"<!-- fiscal-brief:{block} -->", f"<!-- /fiscal-brief:{block} -->"
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"missing or duplicate {block} block in {page}")
        head, rest = text.split(start, 1)
        if end not in rest:
            raise ValueError(f"the {block} block in {page} ends before it starts")
        tail = rest.split(end, 1)[1]
        text = f"{head}{start}\n{page_block(record, block, lang)}\n{end}{tail}"
    return text


def refresh(output: Path) -> None:
    """Run the brief into ``output``, then refresh the fixture, both pages and both figures."""
    output = Path(output)
    study = run_brief(output / "en", exclude=WWII)
    write_outputs(study["record"], study["paths"], output / "es", lang="es")
    record = study["record"]
    pages = {lang: fill_blocks(path.read_text(encoding="utf-8"), record, lang,
                               str(path.relative_to(ROOT)))
             for lang, path in PAGES.items()}           # every page checked before any write

    FIXTURE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "en" / "headline.json", FIXTURE / "headline.json")
    ASSETS.mkdir(parents=True, exist_ok=True)
    for lang, path in PAGES.items():
        path.write_text(pages[lang], encoding="utf-8")
        shutil.copyfile(output / lang / "multipliers.png", ASSETS / f"multipliers_{lang}.png")
    written = [FIXTURE / "headline.json", *PAGES.values(),
               *(ASSETS / f"multipliers_{lang}.png" for lang in PAGES)]
    print(summary_text(record, "en"))
    print("Refreshed: " + ", ".join(str(path.relative_to(ROOT)) for path in written))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.reference_validation.refresh_fiscal_brief",
        description=("Regenerate the fiscal brief's frozen record, the four blocks of its EN "
                     "and ES pages and its two figures."))
    parser.add_argument("--output", required=True, type=Path,
                        help="directory for the brief's EN and ES outputs")
    refresh(parser.parse_args(argv).output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
