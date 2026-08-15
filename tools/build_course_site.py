"""Build the *course website* for the puremacro complementary course.

Single source of truth: the Spanish lesson notebooks written in jupytext
percent format at ``notebooks/course/*_es.py``. For each discovered lesson this
tool

  1. converts + executes it to ``notebooks/course/<stem>.ipynb`` with jupytext
     (kernel cwd pinned to ``notebooks/`` -- same recipe as
     ``tools/build_notebooks.py``), then
  2. renders a **self-contained** reveal.js deck ``<stem>.html`` with
     nbconvert's ``SlidesExporter`` (same recipe as
     ``tools/build_course_slides.py``), inlining the reveal.js / jQuery CSS+JS
     so the page needs no CDN at view time (see ``inline_reveal_assets``).

It then

  (b) zips the **runnable** material into ``materiales_cuadernos.zip`` (one
      ``.ipynb`` per lesson + ``notebooks/course/data/`` + the ``_nbstyle.py`` /
      ``_tutor.py`` helpers), mirroring the ``notebooks/`` tree so the relative
      data paths inside the lessons keep resolving on the student's machine,
  (c) writes an ``index.html`` course-calendar landing page linking every
      lesson (its footer points at the ZIP above, i.e. at something the student
      actually receives),
  (d) packages a Canvas **Common Cartridge** ``curso_mav.imscc`` (IMS CC 1.3:
      a valid ``imsmanifest.xml`` + the lesson HTML as web content + the
      ``Slides*_preview.pdf`` transparencies + the materials ZIP as file
      resources), and
  (e) mirrors the lesson pages + index into a ``wordpress/`` folder ready to
      upload.

All outputs land under ``<slides-root>/curso_web/``. The command is idempotent:
re-running overwrites its own artifacts and rebuilds a notebook only when its
``.py`` source is newer (unless ``--force``).

    python tools/build_course_site.py                 # build the whole site
    python tools/build_course_site.py 00_syllabus_es  # one lesson (by stem)
    python tools/build_course_site.py --list          # list discovered lessons
    python tools/build_course_site.py --help
    # force the course folder (decks may sit directly in it, no slides/ needed):
    python tools/build_course_site.py --slides-root ~/Documents/TEACHING/MAV

Requires the ``notebooks`` extra:  pip install -e ".[notebooks]"

HARD RULES honoured by this file: it only *adds* a new tool. It never touches
``puremacro/fetch/``, ``pyproject.toml`` or the sibling build scripts, and it
does not bump the package version.
"""
from __future__ import annotations

import argparse
import html as _html
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
PROJ_ROOT = Path(__file__).resolve().parent.parent          # .../MAV/uncertainty_examples/puremacro
NB_DIR = PROJ_ROOT / "notebooks"
COURSE_DIR = NB_DIR / "course"
KERNEL_NAME = "python3"


#: Local working copy of the course when the repo is checked out *outside* MAV
#: (today: ``~/Documents/RESEARCH/puremacro`` vs ``~/Documents/TEACHING/MAV``),
#: so no ancestor of the repo can lead to the decks. Tried before the Drive copy.
LOCAL_MAV = Path.home() / "Documents" / "TEACHING" / "MAV"

#: Google Drive mirror (a *snapshot*, usually behind the local working copy).
#: Last resort before the historical guess.
DRIVE_MAV = (Path.home() / "Library/CloudStorage"
             / "GoogleDrive-jorge.alonsoortiz@gmail.com/My Drive/MAV/slides")


def _has_decks(p: Path) -> bool:
    """True when *p* holds the Beamer decks themselves — sources
    ``Slides*mav.tex`` or the ``Slides*_preview.pdf`` transparencies.

    Identifying a candidate by the decks (and not by, say, the presence of a
    ``curso_web/``) keeps a stray folder left by an earlier mis-rooted run from
    winning the resolution.
    """
    return p.is_dir() and (
        any(p.glob("Slides*mav.tex")) or any(p.glob("Slides*_preview.pdf"))
    )


def _find_slides_root(proj: Path) -> Path:
    """Locate the folder holding the Beamer decks.

    Two layouts exist in the wild and BOTH must work:

      * *nested*: the decks under ``<MAV>/slides/`` (the historical layout, and
        the one the Google Drive mirror still uses);
      * *flat*: the decks sitting **directly** in ``<MAV>/`` with no ``slides/``
        subfolder — this is the current ``~/Documents/TEACHING/MAV``.

    The repo used to live *inside* MAV (``MAV/uncertainty_examples/puremacro``);
    it is now checked out at ``~/Documents/RESEARCH/puremacro``, so no ancestor
    of the repo leads to the course at all. Looking only for ``<ancestor>/slides``
    (the pre-2026-08 behaviour) therefore fell through to the Drive snapshot and
    silently rebuilt the site *there* instead of in the live ``MAV/curso_web``.

    Resolution order:
      1. ``<ancestor>/slides`` holding the decks (nested layout);
      2. an ancestor holding the decks directly (flat layout, repo inside MAV);
      3. the local working copy ``~/Documents/TEACHING/MAV`` (flat layout);
      4. the Google Drive mirror;
      5. the historical guess.

    Pass ``--slides-root`` to force it (and ``--out`` to override only where the
    site is written).
    """
    for anc in proj.parents:
        cand = anc / "slides"
        if _has_decks(cand):
            return cand
        if _has_decks(anc):
            return anc
    if _has_decks(LOCAL_MAV):
        return LOCAL_MAV
    if DRIVE_MAV.is_dir():
        return DRIVE_MAV
    return proj.parents[1] / "slides"


def _rel(p: Path) -> str:
    """``p`` shown relative to ``SLIDES_ROOT`` for logging, or absolute when it
    lives elsewhere (``--out`` may point anywhere) — never raises."""
    try:
        return str(p.relative_to(SLIDES_ROOT))
    except ValueError:
        return str(p)


SLIDES_ROOT = _find_slides_root(PROJ_ROOT)                 # .../MAV or .../MAV/slides
#: ``.../MAV``. In the *flat* layout SLIDES_ROOT already **is** MAV, so only the
#: nested layout takes the parent.
MAV_ROOT = SLIDES_ROOT.parent if SLIDES_ROOT.name == "slides" else SLIDES_ROOT
DEFAULT_OUT = SLIDES_ROOT / "curso_web"                    # all deliverables live here
DEFAULT_CACHE = Path.home() / ".cache" / "mav_course_site"  # fetched reveal/jquery assets

# reveal.js / jQuery assets nbconvert's SlidesExporter references from CDNs.
# We fetch them once (cached) and inline them so each deck is self-contained.
CDN_ASSETS = {
    "reveal_css": "https://unpkg.com/reveal.js@4.0.2/dist/reveal.css",
    "theme_css": "https://unpkg.com/reveal.js@4.0.2/dist/theme/simple.css",
    "reveal_js": "https://unpkg.com/reveal.js@4.0.2/dist/reveal.js",
    "notes_js": "https://unpkg.com/reveal.js@4.0.2/plugin/notes/notes.js",
    "jquery_js": "https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js",
    "require_js": "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js",
}

#: The eight Beamer decks of the course, in **week order** (this is the order
#: the index and the cartridge present them in). Keyed by the PDF stem so the
#: transparencies discovered on disk can be given a human title.
#: Source of truth for the calendar: ``notebooks/course/00_syllabus_es.py``.
DECKS: tuple[dict[str, str], ...] = (
    {"stem": "Slides01mav", "semanas": "1–2",
     "tema": "Medición del ciclo: contabilidad nacional, filtros y hechos estilizados"},
    {"stem": "Slides02mav", "semanas": "3–4",
     "tema": "El modelo neoclásico de crecimiento"},
    {"stem": "Slides03mav", "semanas": "5",
     "tema": "Riesgo e incertidumbre"},
    {"stem": "Slides04mav", "semanas": "7–8",
     "tema": "Ciclos económicos reales y contabilidad del ciclo"},
    {"stem": "Slides05mav", "semanas": "9–10",
     "tema": "Identificación empírica: SVAR, evidencia narrativa y proyecciones locales"},
    {"stem": "Slides06mav", "semanas": "11",
     "tema": "Mercados incompletos, heterogeneidad y HANK"},
    {"stem": "Slides07mav", "semanas": "13–14",
     "tema": "Mecanismos: trabajo, capital y economía pequeña y abierta"},
    {"stem": "Slides08mav", "semanas": "15–16",
     "tema": "Mercados no competitivos: precios rígidos y fricciones laborales"},
)
DECK_BY_STEM = {d["stem"]: d for d in DECKS}
DECK_ORDER = {d["stem"]: i for i, d in enumerate(DECKS)}

#: Exam / assessment milestones shown in the calendar (weeks with no deck).
HITOS: tuple[tuple[str, str], ...] = (
    ("6", "Primer examen parcial — 23 de septiembre"),
    ("12", "Segundo examen parcial — 4 de noviembre"),
    ("diciembre", "Exámenes finales: teórico y computacional"),
)

#: Week each weekly lesson accompanies, in calendar order. Drives both the
#: "Semanas" column of the index and the order lessons appear in the index and
#: in the cartridge manifest (plain stem sort put S11 after S13/S16).
LESSON_WEEKS: tuple[tuple[str, str], ...] = (
    ("00_syllabus_es", "Programa"),
    ("01_business_cycle_facts_es", "1–2"),
    ("01b_kaldor_capital_es", "1–2"),
    ("02_neoclasico_scb_es", "3–4"),
    ("03_riesgo_incertidumbre_es", "5"),
    ("04_rbc_momentos_es", "7–8"),
    ("04b_contabilidad_ciclo_es", "8"),
    ("05_svar_identificacion_es", "9–10"),
    ("06_lp_narrativa_es", "9–10"),
    ("09_a5_panel_narrativa_es", "9–10"),
    ("20_heterogeneidad_hank_es", "11"),
    ("07_competitivos_laboral_es", "13–14"),
    ("10_b1_capital_ajuste_es", "13–14"),
    ("10b_economia_abierta_es", "14"),
    ("10c_rigideces_nominales_es", "15"),
    ("08_desempleo_flujos_ia_es", "15–16"),
    ("11_b2_mp_hosios_es", "15–16"),
)
WEEK_BY_LESSON = dict(LESSON_WEEKS)
LESSON_ORDER = {stem: i for i, (stem, _) in enumerate(LESSON_WEEKS)}

#: Name of the ZIP with the *runnable* material (notebooks + data) that ships
#: next to the site and inside the cartridge. The index footer and the cartridge
#: both point at this exact file, so the student's instructions match what the
#: student actually receives.
MATERIALS_ZIP_NAME = "materiales_cuadernos.zip"

COURSE_TITLE = "Macroeconomía Avanzada — Curso complementario puremacro"
COURSE_SUBTITLE = (
    "Lecciones en español construidas sobre puremacro (Python puro, instalación "
    "local, $0). Cada lección se lee como diapositivas y se ejecuta como cuaderno "
    f"en tu Jupyter local: descarga «{MATERIALS_ZIP_NAME}» y corre "
    "«pip install puremacro»."
)


# --------------------------------------------------------------------------- #
# Discovery / metadata
# --------------------------------------------------------------------------- #
def discover_lessons(names: list[str] | None = None) -> list[Path]:
    """Sorted Spanish lesson sources ``course/*_es.py`` (excluding ``_*.py``).

    ``names`` (stems or filenames) filters the set; empty/None means all.
    """
    srcs = sorted(
        p for p in COURSE_DIR.glob("*_es.py") if not p.name.startswith("_")
    )
    if names:
        wanted = set(names)
        srcs = [s for s in srcs if s.stem in wanted or s.name in wanted]
    return srcs


_H1_RE = re.compile(r"^#\s+#\s+(.+?)\s*$", re.MULTILINE)


def lesson_title(src_py: Path) -> str:
    """First markdown H1 (``# # Title``) of a percent-format lesson, or a
    humanised stem as fallback."""
    m = _H1_RE.search(src_py.read_text(encoding="utf-8"))
    if m:
        return m.group(1).strip()
    return src_py.stem.replace("_", " ")


#: Numeric ``NN_`` prefixes reserved for the advanced *elective* modules that
#: sit apart from the weekly lesson track (they are stand-alone decks, not tied
#: to a course week). Kept as a small explicit set so the split is obvious and
#: easy to extend. Note ``20_heterogeneidad_hank_es`` is **not** elective: it is
#: the lesson for week 11 (``Slides06mav``); it keeps its ``20_`` stem only so
#: its published URL does not change. Membership in ``LESSON_WEEKS`` wins.
ELECTIVE_NUMS = frozenset(range(21, 26))  # stems 21..25


def is_elective(stem: str) -> bool:
    """True for the advanced elective modules (``21``–``25``), which the index
    and cartridge group separately from the weekly lessons."""
    if stem in WEEK_BY_LESSON:
        return False
    num = stem[:2]
    return num.isdigit() and int(num) in ELECTIVE_NUMS


def module_label(stem: str) -> str:
    """Calendar label: the week(s) the lesson accompanies when it is part of the
    weekly track, ``Electivo`` for the stand-alone advanced modules."""
    week = WEEK_BY_LESSON.get(stem)
    if week:
        return week if week == "Programa" else f"Semana {week}"
    if is_elective(stem):
        return "Electivo"
    return "—"


def lesson_sort_key(stem: str) -> tuple[int, str]:
    """Calendar order for the weekly track, then alphabetical for the rest."""
    return (LESSON_ORDER.get(stem, 10_000), stem)


# --------------------------------------------------------------------------- #
# Step 1 -- execute lesson to .ipynb (imitates tools/build_notebooks.py)
# --------------------------------------------------------------------------- #
def ensure_kernel() -> None:
    """Register a ``python3`` kernelspec pointing at the current interpreter, so
    ``jupytext --execute`` runs the notebooks with the Python that has puremacro
    installed. Idempotent."""
    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--user", "--name", KERNEL_NAME],
        capture_output=True,
    )


def build_ipynb(src: Path, *, force: bool = False, execute: bool = True) -> Path:
    """Convert+execute ``course/<stem>.py`` -> ``course/<stem>.ipynb``.

    Skips the (slow) execution when the ``.ipynb`` already exists and is newer
    than its ``.py`` source, unless ``force``. ``--run-path NB_DIR`` pins the
    kernel cwd to ``notebooks/`` so ``import _nbstyle`` / ``from _tutor import
    tutor`` resolve regardless of where the output is written.
    """
    out = src.with_suffix(".ipynb")
    if not execute and out.exists():
        return out
    if (
        not force
        and out.exists()
        and out.stat().st_mtime >= src.stat().st_mtime
    ):
        print(f"  · up-to-date  {out.relative_to(PROJ_ROOT)}")
        return out
    rel = str(src.relative_to(NB_DIR))  # e.g. course/00_syllabus_es.py
    cmd = [
        sys.executable, "-m", "jupytext",
        "--to", "ipynb", "--execute", "--set-kernel", KERNEL_NAME,
        "--run-path", str(NB_DIR), rel,
    ]
    print("  › " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=NB_DIR).returncode
    if rc != 0:
        raise RuntimeError(f"jupytext failed for {src.name} (rc={rc})")
    return out


# --------------------------------------------------------------------------- #
# Step 2 -- render self-contained reveal.js deck (imitates build_course_slides)
# --------------------------------------------------------------------------- #
def _fetch_asset(name: str, url: str, cache_dir: Path) -> str | None:
    """Return the text of a CDN asset, caching it under ``cache_dir``. Returns
    ``None`` (and warns) if it is neither cached nor fetchable -- callers then
    fall back to leaving the CDN reference in place."""
    ext = ".css" if url.endswith(".css") else ".js"
    cached = cache_dir / f"{name}{ext}"
    if cached.exists() and cached.stat().st_size > 0:
        return cached.read_text(encoding="utf-8", errors="replace")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=45).read()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        return data.decode("utf-8", errors="replace")
    except Exception as exc:  # offline / blocked -> graceful fallback
        print(f"  ! could not fetch {url} ({type(exc).__name__}); leaving CDN ref")
        return None


def inline_reveal_assets(body: str, cache_dir: Path) -> tuple[str, bool]:
    """Inline reveal.js / jQuery CSS+JS into an nbconvert slides document.

    nbconvert loads reveal from a CDN via require.js. We (a) inline the CSS, (b)
    inline jQuery and drop require.js so reveal/notes fall back to browser
    globals, (c) inject reveal.js + notes.js as plain scripts, and (d) rewrite
    the ``require([...], fn)`` bootstrap into a direct call. MathJax stays on its
    CDN loader (bundling MathJax v2 into one file is impractical); math degrades
    to readable source when offline. Returns ``(html, fully_inlined)``.
    """
    assets = {k: _fetch_asset(k, u, cache_dir) for k, u in CDN_ASSETS.items()}
    if any(v is None for v in assets.values()):
        return body, False  # keep CDN refs; still works online

    a = CDN_ASSETS
    # (a) CSS: <link href="reveal.css" .../> and themed <link id="theme" .../>
    body = body.replace(
        f'<link href="{a["reveal_css"]}" rel="stylesheet"/>',
        f"<style>\n{assets['reveal_css']}\n</style>",
    )
    theme_css = re.sub(r"@import url\([^)]*\);?", "", assets["theme_css"])  # drop web-font @import
    body = body.replace(
        f'<link href="{a["theme_css"]}" id="theme" rel="stylesheet"/>',
        f'<style id="theme">\n{theme_css}\n</style>',
    )
    # (b) jQuery inline (global $, since require.js is removed next)
    body = body.replace(
        f'<script src="{a["jquery_js"]}"></script>',
        f"<script>{assets['jquery_js']}</script>",
    )
    body = body.replace(f'<script src="{a["require_js"]}"></script>', "")
    # (c)+(d) rewrite the require() bootstrap into a direct call, injecting
    # reveal.js + notes.js as plain scripts just before it.
    inject = (
        f"<script>{assets['reveal_js']}</script>\n"
        f"<script>{assets['notes_js']}</script>\n"
        "<script>\n(function(Reveal, RevealNotes){"
    )
    body, n = re.subn(
        r"<script>\s*require\(\s*\{.*?\}\s*,\s*\[.*?\]\s*,\s*"
        r"function\(Reveal,\s*RevealNotes\)\{",
        lambda _m: inject,  # function repl: insert JS verbatim (no backslash escapes)
        body,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        return body, False
    # close the IIFE: the trailing `\n);\n</script>` becomes an invocation.
    tail = "\n);\n</script>"
    idx = body.rfind(tail)
    if idx == -1:
        return body, False
    body = (
        body[:idx]
        + "\n)(window.Reveal, window.RevealNotes);\n</script>"
        + body[idx + len(tail):]
    )
    # harden the MathJax rerender hook so navigation never throws when offline.
    body = body.replace(
        "if(MathJax.Hub.getAllJax(",
        "if(window.MathJax && MathJax.Hub.getAllJax(",
    )
    return body, True


def render_deck(stem: str, title: str, cache_dir: Path, *, inline: bool) -> str:
    """Render ``course/<stem>.ipynb`` to a reveal.js HTML string (with the lesson
    title in ``<title>`` and, when possible, all reveal assets inlined)."""
    import nbformat
    from nbconvert import SlidesExporter

    src = COURSE_DIR / f"{stem}.ipynb"
    nb = nbformat.read(src, as_version=4)
    exporter = SlidesExporter()
    exporter.reveal_scroll = True
    body, _ = exporter.from_notebook_node(nb)
    body = body.replace(
        "<title>Notebook slides</title>",
        f"<title>{_html.escape(title)}</title>",
        1,
    )
    if inline:
        body, ok = inline_reveal_assets(body, cache_dir)
        print(f"  · assets {'inlined (self-contained)' if ok else 'left on CDN'}")
    return body


# --------------------------------------------------------------------------- #
# Step (b) -- runnable materials ZIP (lesson notebooks + data + helpers)
# --------------------------------------------------------------------------- #
#: Shipped inside the ZIP. Deliberately says nothing about schedule, Canvas or
#: grading — those live in the syllabus, not here.
MATERIALS_README = """Cuadernos ejecutables del curso complementario puremacro
==========================================================

Contenido
---------
  notebooks/_nbstyle.py            estilo común de las figuras
  notebooks/course/*.ipynb         una lección por cuaderno
  notebooks/course/_tutor.py       ayudante de estudio (funciona sin conexión)
  notebooks/course/data/           los datos que leen las lecciones

Cómo ejecutarlo
---------------
  1. Descomprime este archivo donde quieras.
  2. Instala el paquete:      pip install puremacro
     (y, para abrir cuadernos: pip install jupyterlab)
  3. Arranca Jupyter DESDE la carpeta `notebooks/`:
         cd notebooks
         jupyter lab
     y abre `course/<lección>.ipynb`.

Importante: no muevas los cuadernos fuera de `notebooks/course/`. Las lecciones
localizan los datos por ruta relativa (`notebooks/course/data/`), de modo que
corren sin conexión mientras se conserve esta estructura de carpetas.
"""


def build_materials_zip(out_dir: Path, lessons: list[dict]) -> Path | None:
    """Zip the material the pages tell the student to *run*, and return its path.

    Before this existed, the site and the cartridge shipped only HTML, while the
    index footer told the student to execute ``notebooks/course/<lección>.ipynb``
    and said the data lived in ``notebooks/course/data/`` — paths that existed
    only on the instructor's machine. This packages exactly those paths.

    The archive mirrors the ``notebooks/`` tree (``notebooks/_nbstyle.py``,
    ``notebooks/course/<stem>.ipynb``, ``notebooks/course/_tutor.py``,
    ``notebooks/course/data/**``) because each lesson resolves its helpers and
    its data *relative to* that layout (``_nb = cwd if (cwd/"_nbstyle.py") else
    cwd.parent``; ``DATA = _nb/"course"/"data"``). Flattening it would break both.

    Returns ``None`` when there is nothing to ship (e.g. ``--no-exec`` on a tree
    with no built ``.ipynb``).
    """
    entries: list[tuple[Path, str]] = []
    missing: list[str] = []
    for L in lessons:
        nb = COURSE_DIR / f"{L['stem']}.ipynb"
        if nb.exists():
            entries.append((nb, f"notebooks/course/{nb.name}"))
        else:
            missing.append(nb.name)
    for helper in (NB_DIR / "_nbstyle.py", COURSE_DIR / "_nbstyle.py",
                   NB_DIR / "_tutor.py", COURSE_DIR / "_tutor.py"):
        if helper.exists():
            entries.append((helper, f"notebooks/{helper.relative_to(NB_DIR).as_posix()}"))
    data_dir = COURSE_DIR / "data"
    n_data = 0
    if data_dir.is_dir():
        for f in sorted(data_dir.rglob("*")):
            if not f.is_file() or f.name.startswith(".") or "__pycache__" in f.parts:
                continue
            entries.append((f, f"notebooks/course/data/{f.relative_to(data_dir).as_posix()}"))
            n_data += 1
    if not entries:
        print("  ! no notebooks/data to package; skipping the materials ZIP")
        return None
    if missing:
        print(f"  ! missing .ipynb (not packaged): {', '.join(missing)}")
    dest = out_dir / MATERIALS_ZIP_NAME
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("LEEME.txt", MATERIALS_README)
        for src, arc in entries:
            z.write(src, arc)
    print(f"→ materiales {_rel(dest)}  "
          f"({len(entries) - n_data} cuaderno(s)/ayudante(s) + {n_data} archivo(s) de datos)")
    return dest


# --------------------------------------------------------------------------- #
# Step (c) -- course-calendar index
# --------------------------------------------------------------------------- #
def _index_table(lessons: list[dict]) -> str:
    """One ``.table-wrap`` table for a group of lessons (rows only differ by
    content; the electives and the weekly lessons share the same markup)."""
    rows = []
    for L in lessons:
        rows.append(
            "<tr>"
            f'<td class="num">{_html.escape(L["num"])}</td>'
            f'<td class="mod">{_html.escape(L["module"])}</td>'
            f'<td class="topic">{_html.escape(L["title"])}</td>'
            f'<td class="link"><a href="./{_html.escape(L["stem"])}.html">Abrir presentación →</a></td>'
            "</tr>"
        )
    body_rows = "\n".join(rows)
    return f"""<div class="table-wrap">
        <table>
          <thead>
            <tr><th>#</th><th>Semanas</th><th>Tema</th><th>Lección</th></tr>
          </thead>
          <tbody>
{body_rows}
          </tbody>
        </table>
      </div>"""


def _calendar_table() -> str:
    """The 16-week calendar: the eight Beamer decks plus the exam milestones,
    in week order. Static content — it mirrors ``00_syllabus_es.py``."""
    hitos = dict(HITOS)
    rows: list[tuple[str, str, str]] = []
    for d in DECKS:
        first = d["semanas"].split("–")[0]
        # exam weeks slot in ahead of the block that follows them
        for wk in list(hitos):
            if wk.isdigit() and int(wk) < int(first):
                rows.append((wk, hitos.pop(wk), "—"))
        rows.append((d["semanas"], d["tema"], d["stem"] + "_preview.pdf"))
    for wk, txt in hitos.items():
        rows.append((wk, txt, "—"))

    body = "\n".join(
        "<tr>"
        f'<td class="num">{_html.escape(w)}</td>'
        f'<td class="topic">{_html.escape(t)}</td>'
        f'<td class="mod">{_html.escape(f)}</td>'
        "</tr>"
        for w, t, f in rows
    )
    return f"""<div class="table-wrap">
        <table>
          <thead>
            <tr><th>Semana</th><th>Bloque</th><th>Transparencias</th></tr>
          </thead>
          <tbody>
{body}
          </tbody>
        </table>
      </div>"""


def render_index(lessons: list[dict], materials: str | None = None) -> str:
    """Self-contained, theme-aware ``index.html`` with the course calendar.

    Weekly lessons and the advanced **elective** modules (stems 21–25; ``20_``
    is the week-11 HANK lesson, see ``ELECTIVE_NUMS``) are shown
    as two clearly separated sections. Each lesson links to a sibling
    ``<stem>.html`` so the same file works both at ``curso_web/`` and in the
    ``wordpress/`` mirror.

    ``materials`` is the file name of the runnable-materials ZIP shipped next to
    this page (see :func:`build_materials_zip`); when given, the footer links to
    it. When it is ``None`` the footer says the notebooks are not part of this
    download instead of pointing at instructor-only paths.
    """
    weekly = [L for L in lessons if not L.get("elective")]
    electives = [L for L in lessons if L.get("elective")]

    sections = [
        '<section>',
        '<h2 class="section-title">Calendario: 16 semanas, ocho mazos</h2>',
        '<p class="section-note">El curso avanza por bloques de una o dos semanas; '
        'cada bloque tiene un mazo de transparencias (los PDF viajan dentro del '
        'paquete de Canvas, en el módulo «Transparencias (PDF)»). Hay '
        '<strong>dos exámenes parciales</strong> —semana 6, 23 de septiembre, y semana 12, '
        '4 de noviembre— y <strong>dos exámenes finales</strong> en diciembre: uno teórico y uno '
        'computacional.</p>',
        _calendar_table(),
        '</section>',
        '<section>',
        '<h2 class="section-title">Lecciones semanales</h2>',
        '<p class="section-note">Los cuadernos que acompañan a cada mazo, en orden '
        'de calendario. No sustituyen a las transparencias: las complementan con '
        'código ejecutable.</p>',
        _index_table(weekly if weekly else lessons),
        '</section>',
    ]
    if electives:
        sections += [
            '<section class="electives">',
            '<h2 class="section-title">Módulos electivos (avanzados)</h2>',
            '<p class="section-note">Módulos independientes de profundización '
            '(IA y KORV, riesgo de orden 3, Growth-at-Risk, mercado laboral de '
            '4 estados, cómputo de valor iterado). No forman parte del calendario semanal: '
            'se pueden cursar en cualquier orden como complemento. Los electivos 23 y 24 '
            'profundizan temas que sí forman parte del temario examinable (A3 y B2); '
            'el resto es material adicional.</p>',
            _index_table(electives),
            '</section>',
        ]
    main_sections = "\n      ".join(sections)

    built = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if materials:
        m = _html.escape(materials)
        howto = (
            '<p><strong>Cómo usar una lección.</strong> Ábrela como presentación '
            '(reveal.js) desde el enlace de la tabla, o ejecútala como cuaderno: '
            f'descarga <a href="./{m}"><code>{m}</code></a> —viaja junto a esta '
            'página y dentro del paquete de Canvas—, descomprímelo y abre '
            '<code>notebooks/course/&lt;lección&gt;.ipynb</code> arrancando Jupyter '
            'desde la carpeta <code>notebooks/</code>. El ZIP incluye los datos '
            'que leen las lecciones (<code>notebooks/course/data/</code>), así que '
            'corren sin conexión; sólo hace falta <code>pip install puremacro</code>.</p>'
        )
    else:
        howto = (
            '<p><strong>Cómo usar una lección.</strong> Ábrela como presentación '
            '(reveal.js) desde el enlace de la tabla. Esta descarga contiene sólo '
            'las presentaciones: los cuadernos ejecutables y sus datos se '
            'distribuyen aparte.</p>'
        )
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_html.escape(COURSE_TITLE)}</title>
<style>
  :root {{ color-scheme: light dark; --bg:#ffffff; --fg:#1a1a1a; --muted:#5a5f66;
           --line:#e4e7eb; --accent:#1e88e5; --card:#f7f8fa; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#14171a; --fg:#e7eaee; --muted:#9aa1a9; --line:#2a2f35;
             --accent:#5aa8f0; --card:#1b1f24; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    line-height:1.5; }}
  .wrap {{ max-width:920px; margin:0 auto; padding:2.6rem 1.25rem 4rem; }}
  header p.kicker {{ text-transform:uppercase; letter-spacing:.08em; font-size:.72rem;
    color:var(--accent); font-weight:700; margin:0 0 .4rem; }}
  h1 {{ font-size:1.7rem; line-height:1.2; margin:0 0 .6rem; }}
  p.sub {{ color:var(--muted); max-width:60ch; margin:0 0 2rem; }}
  section {{ margin:0 0 2.2rem; }}
  section.electives {{ margin-top:2.6rem; padding-top:1.6rem; border-top:2px solid var(--line); }}
  h2.section-title {{ font-size:1.12rem; margin:0 0 .3rem; }}
  section.electives h2.section-title::before {{ content:"★ "; color:var(--accent); }}
  p.section-note {{ color:var(--muted); font-size:.9rem; max-width:64ch; margin:0 0 1rem; }}
  .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px; }}
  table {{ border-collapse:collapse; width:100%; min-width:560px; }}
  th, td {{ text-align:left; padding:.72rem .9rem; border-bottom:1px solid var(--line);
    vertical-align:top; }}
  thead th {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.05em;
    color:var(--muted); background:var(--card); }}
  tbody tr:last-child td {{ border-bottom:none; }}
  td.num {{ font-variant-numeric:tabular-nums; color:var(--muted); width:3.2rem; }}
  td.mod {{ white-space:nowrap; font-weight:600; }}
  td.topic {{ min-width:16rem; }}
  td.link a {{ color:var(--accent); text-decoration:none; font-weight:600; white-space:nowrap; }}
  td.link a:hover {{ text-decoration:underline; }}
  footer {{ margin-top:2rem; color:var(--muted); font-size:.86rem; }}
  code {{ background:var(--card); padding:.1rem .35rem; border-radius:5px; font-size:.85em; }}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <p class="kicker">Macroeconomía Avanzada</p>
      <h1>{_html.escape(COURSE_TITLE)}</h1>
      <p class="sub">{_html.escape(COURSE_SUBTITLE)}</p>
    </header>
    <main>
      {main_sections}
      <footer>
        {howto}
        <p>Sitio generado el {built} · {len(lessons)} lección(es).</p>
      </footer>
    </main>
  </div>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Step (d) -- Common Cartridge (IMS CC 1.3)
# --------------------------------------------------------------------------- #
CC_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
LOM_RES_NS = "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource"
LOM_MAN_NS = "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
CC_SCHEMA_LOCATION = (
    f"{CC_NS} http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscp_v1p2_v1p0.xsd "
    f"{LOM_RES_NS} http://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lomresource_v1p0.xsd "
    f"{LOM_MAN_NS} http://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lommanifest_v1p0.xsd"
)


def _q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def _res_id(stem: str) -> str:
    """XML-ID-safe resource identifier (stems start with a digit)."""
    return "res_" + re.sub(r"[^A-Za-z0-9_]", "_", stem)


def build_manifest_xml(lessons: list[dict], pdfs: list[Path],
                       materials: Path | None = None) -> bytes:
    """Build a valid IMS Common Cartridge 1.3 ``imsmanifest.xml``.

    ``index.html`` + each lesson deck become ``webcontent`` resources under
    ``web_resources/``; the transparency PDFs become file resources under
    ``web_resources/pdf/``; the runnable-materials ZIP (when present) becomes a
    ``webcontent`` resource at ``web_resources/<zip>`` — a *sibling* of
    ``index.html``, never in a subfolder, so the footer's relative ``./<zip>``
    link keeps resolving (see :func:`build_imscc`). The single rooted
    organization groups them into a "Lecciones semanales" module, an optional
    "Módulos electivos (avanzados)" module, a "Transparencias (PDF)" module and
    a "Cuadernos ejecutables y datos (ZIP)" entry right after the landing page.
    """
    ET.register_namespace("", CC_NS)
    ET.register_namespace("lomimscc", LOM_MAN_NS)
    ET.register_namespace("lom", LOM_RES_NS)
    ET.register_namespace("xsi", XSI_NS)

    manifest = ET.Element(_q(CC_NS, "manifest"), {
        "identifier": "curso_mav_cartridge",
        _q(XSI_NS, "schemaLocation"): CC_SCHEMA_LOCATION,
    })

    # ---- metadata ----
    meta = ET.SubElement(manifest, _q(CC_NS, "metadata"))
    ET.SubElement(meta, _q(CC_NS, "schema")).text = "IMS Common Cartridge"
    ET.SubElement(meta, _q(CC_NS, "schemaversion")).text = "1.3.0"
    lom = ET.SubElement(meta, _q(LOM_MAN_NS, "lom"))
    general = ET.SubElement(lom, _q(LOM_MAN_NS, "general"))
    title_el = ET.SubElement(general, _q(LOM_MAN_NS, "title"))
    ET.SubElement(title_el, _q(LOM_MAN_NS, "string"), {"language": "es-MX"}).text = COURSE_TITLE
    desc_el = ET.SubElement(general, _q(LOM_MAN_NS, "description"))
    ET.SubElement(desc_el, _q(LOM_MAN_NS, "string"), {"language": "es-MX"}).text = COURSE_SUBTITLE

    # ---- organizations ----
    orgs = ET.SubElement(manifest, _q(CC_NS, "organizations"))
    org = ET.SubElement(orgs, _q(CC_NS, "organization"),
                        {"identifier": "org_1", "structure": "rooted-hierarchy"})
    root = ET.SubElement(org, _q(CC_NS, "item"), {"identifier": "root"})

    # index item
    idx_item = ET.SubElement(root, _q(CC_NS, "item"),
                             {"identifier": "item_index", "identifierref": "res_index"})
    ET.SubElement(idx_item, _q(CC_NS, "title")).text = "Inicio y calendario"

    # runnable materials (notebooks + data), right after the landing page so the
    # student meets it before the lessons that ask them to run it.
    if materials is not None:
        mat_item = ET.SubElement(root, _q(CC_NS, "item"),
                                 {"identifier": "item_materiales",
                                  "identifierref": "res_materiales"})
        ET.SubElement(mat_item, _q(CC_NS, "title")).text = (
            "Cuadernos ejecutables y datos (ZIP)")

    # lessons module (weekly track)
    weekly = [L for L in lessons if not L.get("elective")]
    electives = [L for L in lessons if L.get("elective")]
    lec_mod = ET.SubElement(root, _q(CC_NS, "item"), {"identifier": "mod_lecciones"})
    ET.SubElement(lec_mod, _q(CC_NS, "title")).text = "Lecciones semanales"
    for L in weekly:
        rid = _res_id(L["stem"])
        it = ET.SubElement(lec_mod, _q(CC_NS, "item"),
                           {"identifier": f"item_{rid}", "identifierref": rid})
        ET.SubElement(it, _q(CC_NS, "title")).text = L["title"]

    # elective modules (advanced, stand-alone) -- only if any are present
    if electives:
        ele_mod = ET.SubElement(root, _q(CC_NS, "item"), {"identifier": "mod_electivos"})
        ET.SubElement(ele_mod, _q(CC_NS, "title")).text = "Módulos electivos (avanzados)"
        for L in electives:
            rid = _res_id(L["stem"])
            it = ET.SubElement(ele_mod, _q(CC_NS, "item"),
                               {"identifier": f"item_{rid}", "identifierref": rid})
            ET.SubElement(it, _q(CC_NS, "title")).text = L["title"]

    # PDF module (only if there are transparencies)
    if pdfs:
        pdf_mod = ET.SubElement(root, _q(CC_NS, "item"), {"identifier": "mod_pdf"})
        ET.SubElement(pdf_mod, _q(CC_NS, "title")).text = "Transparencias (PDF)"
        for p in pdfs:
            rid = _res_id("pdf_" + p.stem)
            it = ET.SubElement(pdf_mod, _q(CC_NS, "item"),
                               {"identifier": f"item_{rid}", "identifierref": rid})
            ET.SubElement(it, _q(CC_NS, "title")).text = deck_title(p)

    # ---- resources ----
    resources = ET.SubElement(manifest, _q(CC_NS, "resources"))

    def _add_resource(rid: str, href: str) -> None:
        res = ET.SubElement(resources, _q(CC_NS, "resource"),
                            {"identifier": rid, "type": "webcontent", "href": href})
        ET.SubElement(res, _q(CC_NS, "file"), {"href": href})

    _add_resource("res_index", "web_resources/index.html")
    if materials is not None:
        # sibling of index.html on purpose: the footer's ``./<zip>`` link then
        # resolves identically in curso_web/, in wordpress/ and inside Canvas.
        _add_resource("res_materiales", f"web_resources/{materials.name}")
    for L in lessons:
        _add_resource(_res_id(L["stem"]), f"web_resources/{L['stem']}.html")
    for p in pdfs:
        _add_resource(_res_id("pdf_" + p.stem), f"web_resources/pdf/{p.name}")

    ET.indent(manifest, space="  ")
    xml = ET.tostring(manifest, encoding="utf-8", xml_declaration=True)
    return xml


def build_imscc(out_dir: Path, lessons: list[dict], deck_html: dict[str, str],
                index_html: str, pdfs: list[Path],
                materials: Path | None = None) -> Path:
    """Zip the Common Cartridge ``curso_mav.imscc`` (manifest + web content).

    ``materials`` (the runnable-materials ZIP) travels as a sibling of
    ``index.html`` so the footer's relative link works once Canvas unpacks it.
    """
    manifest = build_manifest_xml(lessons, pdfs, materials)
    dest = out_dir / "curso_mav.imscc"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("imsmanifest.xml", manifest)
        z.writestr("web_resources/index.html", index_html)
        if materials is not None:
            # already deflated; stored to avoid re-compressing a ZIP.
            z.write(materials, f"web_resources/{materials.name}",
                    compress_type=zipfile.ZIP_STORED)
        for L in lessons:
            z.writestr(f"web_resources/{L['stem']}.html", deck_html[L["stem"]])
        for p in pdfs:
            z.write(p, f"web_resources/pdf/{p.name}")
    return dest


# --------------------------------------------------------------------------- #
# Step (e) -- WordPress mirror
# --------------------------------------------------------------------------- #
def mirror_wordpress(out_dir: Path, lessons: list[dict], deck_html: dict[str, str],
                     index_html: str, materials: Path | None = None) -> Path:
    """Copy the index + one HTML per lesson into ``curso_web/wordpress/`` (flat,
    same relative links), ready to upload.

    The materials ZIP is copied too: the index footer links to it relatively, so
    the mirror would otherwise publish a dead link.
    """
    wp = out_dir / "wordpress"
    wp.mkdir(parents=True, exist_ok=True)
    (wp / "index.html").write_text(index_html, encoding="utf-8")
    for L in lessons:
        (wp / f"{L['stem']}.html").write_text(deck_html[L["stem"]], encoding="utf-8")
    if materials is not None and materials.exists():
        shutil.copy2(materials, wp / materials.name)
    return wp


def deck_title(pdf: Path) -> str:
    """Human title for a transparency PDF, e.g.
    ``Mazo 3 — Riesgo e incertidumbre (semana 5)``. Falls back to the stem."""
    stem = pdf.stem.replace("_preview", "")
    d = DECK_BY_STEM.get(stem)
    if not d:
        return pdf.stem
    n = DECK_ORDER[stem] + 1
    semanas = "semana" if "–" not in d["semanas"] else "semanas"
    return f"Mazo {n} — {d['tema']} ({semanas} {d['semanas']})"


def discover_pdfs() -> list[Path]:
    """Transparency PDFs ``slides/[Ss]lides*_preview.pdf`` (if any), in week
    order (``Slides01mav`` … ``Slides08mav``); unknown ones go last, by name.

    Case-insensitive on the leading S so that a lowercase deck filename (the
    historical ``slidesA4mav_preview.pdf``) would be included too.

    ⚠️ SEGURIDAD — NO conviertas esto en ``rglob("**/*_preview.pdf")`` ni
    aflojes el prefijo ``Slides``. Este paquete se entrega a los ALUMNOS, y en
    el árbol de ``slides/`` conviven materiales que jamás deben salir: los
    exámenes (``PrimerExamenParcialMAV_2026.pdf``, ``SegundoExamen…``,
    ``ExamenFinalMAV_2026.pdf``, ``ExamenComputacionalMAV_2026.pdf``), el
    ``solucionario_profesor_master_2026.pdf`` y —esto es lo que un rglob
    filtraría hacia dentro— ``Tareas/solucionTarea*_MAV_2026_preview.pdf``.
    Hoy quedan fuera por DOS razones a la vez: el glob no es recursivo y exige
    el prefijo ``Slides``. Si alguna vez hace falta más flexibilidad, usa una
    lista blanca explícita a partir de ``DECKS``.

    ⚠️ Desde el arreglo del *layout plano*, ``SLIDES_ROOT`` puede ser la propia
    carpeta ``MAV/`` (donde viven los exámenes y el solucionario) y no un
    subdirectorio ``slides/``. Las dos protecciones de arriba siguen bastando
    —los exámenes no empiezan por ``Slides``—, pero cualquier relajación del
    glob se volvería inmediatamente una fuga.
    """
    seen = {p.name: p for p in SLIDES_ROOT.glob("Slides*_preview.pdf")}
    seen.update({p.name: p for p in SLIDES_ROOT.glob("slides*_preview.pdf")})
    return sorted(
        seen.values(),
        key=lambda p: (DECK_ORDER.get(p.stem.replace("_preview", ""), 10_000), p.name),
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    # ``--slides-root`` rebinds these; declared up front because the argparse
    # help strings below already read SLIDES_ROOT.
    global SLIDES_ROOT, MAV_ROOT
    ap = argparse.ArgumentParser(
        description="Build the puremacro course website (decks + index + Canvas cartridge).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("names", nargs="*",
                    help="lesson stems/filenames to build (default: all *_es.py)")
    ap.add_argument("--list", action="store_true",
                    help="list discovered lessons and exit")
    ap.add_argument("--slides-root", type=Path, default=None,
                    help="folder holding the Beamer decks (Slides*mav.tex / "
                         "Slides*_preview.pdf); it may be <MAV> itself (flat "
                         "layout) or <MAV>/slides. Overrides autodetection and "
                         f"moves --out with it (autodetected: {SLIDES_ROOT})")
    ap.add_argument("--out", type=Path, default=None,
                    help="output root (default: <slides-root>/curso_web, today "
                         f"{DEFAULT_OUT})")
    ap.add_argument("--assets-cache", type=Path, default=DEFAULT_CACHE,
                    help=f"cache dir for fetched reveal/jquery assets (default: {DEFAULT_CACHE})")
    ap.add_argument("--force", action="store_true",
                    help="re-execute notebooks even if the .ipynb is up to date")
    ap.add_argument("--no-exec", action="store_true",
                    help="reuse existing .ipynb; never run jupytext (fails if missing)")
    ap.add_argument("--no-inline-assets", action="store_true",
                    help="leave reveal.js/jQuery on their CDNs instead of inlining")
    ap.add_argument("--no-imscc", action="store_true",
                    help="skip building the Common Cartridge .imscc")
    ap.add_argument("--no-materials", action="store_true",
                    help="skip the runnable-materials ZIP (notebooks + data); the "
                         "index footer then stops promising it")
    ns = ap.parse_args(argv)

    # --slides-root wins over autodetection; ``SLIDES_ROOT`` is module-level
    # because ``_rel`` and ``discover_pdfs`` read it.
    if ns.slides_root is not None:
        SLIDES_ROOT = ns.slides_root.expanduser().resolve()
        MAV_ROOT = SLIDES_ROOT.parent if SLIDES_ROOT.name == "slides" else SLIDES_ROOT
        if not _has_decks(SLIDES_ROOT):
            print(f"  ! {SLIDES_ROOT} has no Slides*mav.tex / Slides*_preview.pdf; "
                  f"the cartridge will carry zero transparencies", file=sys.stderr)

    lessons_src = discover_lessons(ns.names or None)
    if not lessons_src:
        print("no lessons match", file=sys.stderr)
        return 2
    if ns.list:
        for s in lessons_src:
            print(f"{s.stem:32} {lesson_title(s)}")
        return 0

    out_dir: Path = ns.out if ns.out is not None else SLIDES_ROOT / "curso_web"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"slides root: {SLIDES_ROOT}\noutput root: {out_dir}")
    if not ns.no_exec:
        ensure_kernel()

    lessons: list[dict] = []
    deck_html: dict[str, str] = {}
    for src in lessons_src:
        stem = src.stem
        title = lesson_title(src)
        print(f"\n== {stem} — {title}")
        build_ipynb(src, force=ns.force, execute=not ns.no_exec)
        html_doc = render_deck(stem, title, ns.assets_cache,
                               inline=not ns.no_inline_assets)
        deck_path = out_dir / f"{stem}.html"
        deck_path.write_text(html_doc, encoding="utf-8")
        print(f"  → {_rel(deck_path)}")
        deck_html[stem] = html_doc
        lessons.append({
            "stem": stem,
            "title": title,
            "num": stem[:2] if stem[:2].isdigit() else "",
            "module": module_label(stem),
            "elective": is_elective(stem),
        })

    # calendar order (index + cartridge share it): weekly track by week, then
    # the electives alphabetically.
    lessons.sort(key=lambda L: lesson_sort_key(L["stem"]))

    # (b) runnable materials (notebooks + data) -- built BEFORE the index so the
    # footer can link to a file that really exists.
    print()
    materials = None if ns.no_materials else build_materials_zip(out_dir, lessons)

    # (c) index
    index_html = render_index(lessons, materials.name if materials else None)
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")
    print(f"→ index    {_rel(out_dir / 'index.html')}")

    # (d) Common Cartridge
    if not ns.no_imscc:
        pdfs = discover_pdfs()
        imscc = build_imscc(out_dir, lessons, deck_html, index_html, pdfs, materials)
        print(f"→ cartridge {_rel(imscc)}  "
              f"({len(lessons)} decks + {len(pdfs)} PDF"
              f"{' + cuadernos' if materials else ''})")

    # (e) WordPress mirror
    wp = mirror_wordpress(out_dir, lessons, deck_html, index_html, materials)
    print(f"→ wordpress {_rel(wp)}/  ({len(lessons)} pages + index)")

    print(f"\nDone. Outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
