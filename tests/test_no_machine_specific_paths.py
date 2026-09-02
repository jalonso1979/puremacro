"""No shipped module may hardcode a path that exists on exactly one machine.

Five modules did — `fetch/wui.py`, `fetch/wui_extras.py`, `fetch/oecd_qna_local.py`
and both `fetch/cdc_births_*.py` — each naming an absolute path into one
person's Google Drive, complete with their email address, which therefore
shipped inside the wheel. The damage was not the disclosure but the silence:
every one of those paths is a *fallback*, so on any other machine the fallback
could not fire, and `build_panel.build_all` catches each producer's failure and
turns it into a line on stdout. The user got a well-formed panel missing a
proxy and no error.

The fix in every case is an environment variable with a documented default
(`PUREMACRO_MAV_ROOT`, matching what `puremacro/examples/*.py` already used).
This test is the guard that keeps it that way.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PKG = _ROOT / "puremacro"

#: Literal shapes that can only ever be right on one machine.
_FORBIDDEN = {
    "unix home directory": re.compile(r"[\"']/(?:Users|home)/[A-Za-z0-9._-]+/"),
    "windows user directory": re.compile(r"[A-Za-z]:\\\\Users\\\\"),
    "cloud account in a path": re.compile(r"GoogleDrive-[^\"'\s/]+|OneDrive-[^\"'\s/]+"),
    "bare email address": re.compile(r"[\"'][^\"'\s]*@(?:gmail|hotmail|outlook|yahoo)\.[a-z]+"),
}

_SOURCES = sorted(p for p in _PKG.rglob("*.py") if "__pycache__" not in p.parts)


def test_the_scan_has_something_to_scan():
    """Positive control: an empty file list would pass everything below."""
    assert len(_SOURCES) > 100, len(_SOURCES)


@pytest.mark.parametrize("label, pattern", sorted(_FORBIDDEN.items()))
def test_no_shipped_module_hardcodes_a_machine_specific_path(label, pattern):
    hits = []
    for path in _SOURCES:
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue          # prose about the old path is allowed
            if pattern.search(line):
                hits.append(f"{path.relative_to(_ROOT)}:{lineno}: {line.strip()}")
    assert not hits, (
        f"{label} hardcoded in shipped code:\n  " + "\n  ".join(hits) +
        "\n\nUse an environment variable with a documented default instead "
        "(PUREMACRO_MAV_ROOT is the established one), and make the "
        "missing-file error name the path and the variable."
    )


def test_the_patterns_would_actually_fire():
    """Each pattern must match the literal it was written for."""
    samples = {
        "unix home directory": '_P = Path("/Users/someone/Documents/x")',
        "windows user directory": r'_P = Path("C:\\Users\\someone\\x")',
        "cloud account in a path": '"Library/CloudStorage/GoogleDrive-a.b@gmail.com/My Drive"',
        "bare email address": 'AUTHOR = "a.b@gmail.com"',
    }
    for label, pattern in _FORBIDDEN.items():
        assert pattern.search(samples[label]), label
