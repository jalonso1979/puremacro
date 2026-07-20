"""Render reveal.js decks from the course lesson notebooks (single source).

Run from the package dir:  python tools/build_course_slides.py
Uses the ``slideshow`` cell metadata already in the notebooks; figures come from
the executed cells. Output: notebooks/course/slides/<stem>.slides.html

    python tools/build_course_slides.py                       # all course .ipynb
    python tools/build_course_slides.py 01_business_cycle_facts  # one (by stem)
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert import SlidesExporter

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "course"
OUT = NB_DIR / "slides"


def main(argv=None) -> int:
    OUT.mkdir(exist_ok=True)
    stems = argv or [p.stem for p in sorted(NB_DIR.glob("[0-9]*.ipynb"))]
    exporter = SlidesExporter()
    exporter.reveal_scroll = True
    for stem in stems:
        src = NB_DIR / f"{stem}.ipynb"
        if not src.exists():
            print(f"skip {stem} (no .ipynb)")
            continue
        nb = nbformat.read(src, as_version=4)
        body, _ = exporter.from_notebook_node(nb)
        dest = OUT / f"{stem}.slides.html"
        dest.write_text(body, encoding="utf-8")
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
