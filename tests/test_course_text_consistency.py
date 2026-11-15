"""The course and playground text names only labs, notebooks and recipes that exist.

``curso/notebooks/T00_syllabus{,_es}.ipynb`` tabulate the labs of every module
and keep the rendered table as a stored output; ``playground/00_start_here.py``
names notebooks by file stem; the optional ``puremacro.text`` and
``puremacro.dp`` sections of the course print a Colab install recipe. No test
read any of them, so the syllabus listed a lab that never existed (T04_B) and
left out five that do, the start page named two companion lessons without the
``_es`` suffix they carry and called every notebook "fully synthetic", and four
sections told Colab users to install the repository head without the commit pin
the course README declares reproducible, framed as a release "not yet
published" after it had shipped. Every check here is static: the notebooks are
parsed, never executed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CURSO = ROOT / "curso" / "notebooks"
SHOWCASE = ROOT / "notebooks"
LESSONS = ROOT / "notebooks" / "course"
START_HERE = ROOT / "playground" / "00_start_here.py"
SYLLABI = ("T00_syllabus.ipynb", "T00_syllabus_es.ipynb")

#: A lab id as the syllabus writes it: ``T04_A`` in "T04_A, T04_C" and in
#: "T01_A_data_panels", but not the syllabus itself (``T00_syllabus``).
_LAB_ID = re.compile(r"\bT(\d\d)_([A-Z])(?![A-Za-z0-9])")
#: A range row, "T05_A to T05_G" (EN) or "T05_A a T05_G" (ES).
_LAB_RANGE = re.compile(r"\bT(\d\d)_([A-Z]) (?:to|a) T\1_([A-Z])(?![A-Za-z0-9])")
#: A notebook the start page names by stem, such as ``06_svar_identification_es``.
_NB_STEM = re.compile(r"`(\d\d[a-z]?_[a-z0-9_]+)`")
#: A repository install hint; a 40-hex commit pin may follow the URL.
_GIT_INSTALL = re.compile(
    r"git\+https://github\.com/jalonso1979/puremacro(?:\.git)?(@[0-9a-f]{40})?(?![0-9a-f])")
_COMMIT = re.compile(r"\b[0-9a-f]{40}\b")
#: Cells the stale-wording scan applies to: those that talk about installing or
#: importing the two modules, so data-vintage prose elsewhere is never flagged.
_INSTALL_CONTEXT = re.compile(r"puremacro\.(?:text|dp)|pip install|git\+https")
#: Wording that froze while puremacro.text and puremacro.dp were development-only.
_STALE = re.compile(
    r"not yet published|not yet released|not released yet|a[úu]n no publicado|"
    r"will come with 4\.9|llegar[áa] con la 4\.9|not on PyPI yet|"
    r"todav[íi]a no (?:est[áa] )?en PyPI|a[úu]n no est[áa] en PyPI|"
    r"not yet in the published (?:version|release)|"
    r"(?:a[úu]n|todav[íi]a) no est[áa] en la versi[óo]n publicada|"
    r"does not carry it yet|a[úu]n no lo trae")


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def _rendered(cell: dict) -> str:
    """Every stored output of a code cell as text: streams and MIME bundles."""
    parts: list[str] = []
    for out in cell.get("outputs", []):
        if "text" in out:
            parts.append("".join(out["text"]))
        for value in out.get("data", {}).values():
            parts.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(parts)


def _ids(text: str) -> set[str]:
    return {f"T{m.group(1)}_{m.group(2)}" for m in _LAB_ID.finditer(text)}


def _ids_with_ranges(text: str) -> set[str]:
    found = _ids(text)
    for m in _LAB_RANGE.finditer(text):
        module, lo, hi = m.group(1), m.group(2), m.group(3)
        found |= {f"T{module}_{chr(c)}" for c in range(ord(lo), ord(hi) + 1)}
    return found


def _shipped_lab_ids() -> set[str]:
    """``T04_A`` for every ``curso/notebooks/T04_A_*.ipynb`` (EN and ES share the id)."""
    ids = set()
    for path in CURSO.glob("T*.ipynb"):
        m = re.match(r"T\d\d_[A-Z](?=_)", path.name)
        if m:
            ids.add(m.group(0))
    assert len(ids) >= 30, sorted(ids)          # positive control: the folder was read
    return ids


def _table_cell(nb: dict) -> dict:
    cells = [c for c in nb["cells"]
             if c["cell_type"] == "code" and "modules = pd.DataFrame(" in _text(c)]
    assert len(cells) == 1, "the syllabus holds exactly one module table"
    return cells[0]


@pytest.mark.parametrize("name", SYLLABI)
def test_syllabus_lab_ids_name_existing_notebooks(name):
    """Every lab id the syllabus writes, in prose or in the table, has a notebook."""
    nb = _notebook(CURSO / name)
    named: set[str] = set()
    for cell in nb["cells"]:
        named |= _ids(_text(cell))
    assert len(named) >= 15, sorted(named)      # positive control: the ids were read
    missing = sorted(named - _shipped_lab_ids())
    assert not missing, f"{name} names labs with no notebook in curso/notebooks: {missing}"


@pytest.mark.parametrize("name", SYLLABI)
def test_syllabus_table_covers_every_lesson(name):
    """Every lab that ships sits in some module row (ranges count)."""
    table = _table_cell(_notebook(CURSO / name))
    listed = _ids_with_ranges(_text(table))
    unlisted = sorted(_shipped_lab_ids() - listed)
    assert not unlisted, f"{name}: labs that ship but are in no module row: {unlisted}"


@pytest.mark.parametrize("name", SYLLABI)
def test_syllabus_stored_output_matches_source(name):
    """The rendered table a reader sees on GitHub says what the source says."""
    table = _table_cell(_notebook(CURSO / name))
    rendered = _rendered(table)
    assert "T01_A_data_panels" in rendered, f"{name}: the table output is not stored"
    assert _ids(rendered) == _ids(_text(table)), (
        f"{name}: the stored table output differs from the source; edit them together")
    stale = sorted(_ids(rendered) - _shipped_lab_ids())
    assert not stale, f"{name}: the stored table renders labs that do not exist: {stale}"


def test_start_here_names_shipped_notebooks():
    """Every notebook the browser start page names is one the build copies."""
    text = START_HERE.read_text(encoding="utf-8")
    stems = sorted(set(_NB_STEM.findall(text)))
    assert len(stems) >= 20, stems              # positive control: the page was read
    problems = []
    for stem in stems:
        if (LESSONS / f"{stem}.ipynb").exists():
            continue                            # companion lessons are Spanish-only
        if (START_HERE.parent / f"{stem}.py").exists():
            continue                            # jupytext-backed playground demos built by build_playground.sh
        if not (SHOWCASE / f"{stem}.ipynb").exists():
            problems.append(f"{stem}: neither notebooks/{stem}.ipynb, "
                            f"notebooks/course/{stem}.ipynb, nor playground/{stem}.py exists")
        elif not stem.endswith("_es") and not (SHOWCASE / f"{stem}_es.ipynb").exists():
            problems.append(f"{stem}: the page promises an _es twin and there is none")
    assert not problems, "\n".join(problems)
    assert "fully synthetic" not in text, (
        "several playground notebooks read frozen empirical data files; "
        "the start page may say offline, not fully synthetic")


def test_course_install_hints_use_released_edition():
    """The course requires released DP/text and never installs an obsolete source tree."""
    readme = (CURSO / "README.md").read_text(encoding="utf-8")
    assert "puremacro==5.5.0" in readme
    assert "course-2026A-5.5.0" in readme
    checked = 0
    for path in sorted(CURSO.glob("*.ipynb")):
        checked += 1
        for i, cell in enumerate(_notebook(path)["cells"]):
            text = _text(cell)
            assert not _GIT_INSTALL.search(text), f"{path.name} cell {i}: source reinstall"
            if _INSTALL_CONTEXT.search(text):
                assert not _STALE.search(text + "\n" + _rendered(cell)), (
                    f"{path.name} cell {i}: obsolete installation guidance in source or stored output")
    assert checked == 72
