"""Split documents into paragraphs or sentences.

The right unit of analysis is often not the document: FOMC minutes by
paragraph, Beige Book reports by district section, IMF Article IV reports by
paragraph. These splitters are deliberately simple and deterministic; for
linguistically careful sentence boundaries use spaCy and pass its splitter as
a callable wherever ``by=`` is accepted.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import pandas as pd

_PARA_RE = re.compile(r"\n[ \t\r\f\v]*\n")
# Sentence end: . ! ? (optionally followed by closing quotes/brackets), then
# whitespace, then something that can start a sentence.
_SENT_RE = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[\"'“‘(\[¿¡]?[A-ZÀ-ÖØ-Þ0-9])")
# Tokens that end in a period without ending a sentence.
_ABBREV = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "sra", "st", "vs", "etc", "e.g", "i.e",
    "u.s", "u.k", "u.n", "e.u", "no", "nos", "art", "fig", "approx", "inc",
    "ltd", "co", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "gov", "rep", "sen", "op", "cit", "p", "pp",
})


def split_paragraphs(text: str, *, min_chars: int = 1) -> list[str]:
    """Split on blank lines; inner line breaks are joined with a space."""
    paras = (" ".join(p.split()) for p in _PARA_RE.split(text or ""))
    return [p for p in paras if len(p) >= min_chars]


def split_sentences(text: str, *, min_chars: int = 1) -> list[str]:
    """Rule-based sentence splitter that respects common abbreviations and decimals."""
    text = " ".join((text or "").split())
    if not text:
        return []
    pieces = _SENT_RE.split(text)
    out: list[str] = []
    for piece in pieces:
        if out and _ends_with_abbreviation(out[-1]):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return [s.strip() for s in out if len(s.strip()) >= min_chars]


def _ends_with_abbreviation(sentence: str) -> bool:
    last = sentence.rstrip("\"'”’)]").split(" ")[-1]
    if not last.endswith("."):
        return False
    word = last[:-1].lower().lstrip("(\"'“")
    # Single capital initials ("J. Powell") also do not end sentences.
    return word in _ABBREV or (len(word) == 1 and word.isalpha())


Splitter = Callable[[str], Sequence[str]]


def get_splitter(by: str | Splitter) -> Splitter:
    if callable(by):
        return by
    if by == "paragraph":
        return split_paragraphs
    if by == "sentence":
        return split_sentences
    raise ValueError(f"by must be 'paragraph', 'sentence' or a callable, got {by!r}")


def segment(
    texts: Sequence[str],
    by: str | Splitter = "paragraph",
    *,
    ids: Sequence[object] | None = None,
    min_chars: int = 1,
) -> pd.DataFrame:
    """Long frame of segments: ``parent_id``, ``position`` (0-based), ``text``.

    ``ids`` defaults to the row position of each text.
    """
    split = get_splitter(by)
    if ids is None:
        ids = range(len(texts))
    rows = []
    for pid, text in zip(ids, texts):
        pos = 0
        for seg in split(text):
            seg = seg.strip()
            if len(seg) < min_chars:
                continue
            rows.append((pid, pos, seg))
            pos += 1
    return pd.DataFrame(rows, columns=["parent_id", "position", "text"])


__all__ = ["split_paragraphs", "split_sentences", "segment"]
