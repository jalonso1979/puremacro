"""Guard the course companion: EN/ES parity, required lesson sections, syllabus."""
from pathlib import Path

COURSE = Path(__file__).resolve().parents[1] / "notebooks" / "course"


def _lessons():
    return [
        p
        for p in COURSE.glob("[0-9]*.py")
        if not p.stem.endswith("_es") and p.stem != "00_syllabus"
    ]


def test_every_course_notebook_has_es_sibling():
    for p in COURSE.glob("[0-9]*.py"):
        if p.stem.endswith("_es"):
            continue
        assert (COURSE / f"{p.stem}_es.py").is_file(), f"missing ES sibling for {p.name}"


def test_lessons_have_required_sections():
    for p in _lessons():
        text = p.read_text(encoding="utf-8").lower()
        for needle in ("objective", "exercise", "ai"):
            assert needle in text, f"{p.name} missing a '{needle}' section"


def test_lessons_are_slide_tagged():
    for p in _lessons():
        assert "slide_type" in p.read_text(encoding="utf-8"), f"{p.name} has no slide tags"


def test_syllabus_lists_module_one():
    syl = (COURSE / "00_syllabus.py").read_text(encoding="utf-8").lower()
    assert "business" in syl and "module" in syl
