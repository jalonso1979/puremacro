"""Multi-speaker central bank press conference and testimony parser.

Parses unstructured transcripts (FOMC press conferences, ECB press briefings,
Congressional testimonies, Banxico Q&A sessions) into structured dialogue turns:
- Identifies speaker roles (Chair/Governor vs. Press/Lawmakers vs. Moderator).
- Separates Opening Remarks from the Q&A segment.
- Enables measuring tone, sentiment, and forward guidance asymmetry between
  policymaker responses and journalist questioning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator, Sequence


@dataclass(frozen=True)
class DialogueTurn:
    """A single spoken turn in a press conference or hearing.

    Attributes
    ----------
    speaker : str
        Name of the speaker (e.g. 'CHAIR POWELL', 'STEVE LIESMAN', 'PRESIDENT LAGARDE').
    role : str
        Categorized role: 'chair', 'press', 'board', 'lawmaker', or 'unknown'.
    segment : str
        'statement' (prepared opening remarks) or 'qa' (interactive questioning).
    text : str
        Spoken text for this turn.
    turn_index : int
        Sequential turn counter (0-indexed).
    """
    speaker: str
    role: str
    segment: str
    text: str
    turn_index: int


@dataclass(frozen=True)
class TranscriptDocument:
    """Structured press conference or testimony transcript.

    Attributes
    ----------
    title : str
        Document title.
    date : str
        Announcement date (YYYY-MM-DD).
    institution : str
        'FED', 'ECB', 'BANXICO', 'BOE', etc.
    turns : tuple of DialogueTurn
        All parsed speaker turns.
    """
    title: str
    date: str
    institution: str
    turns: tuple[DialogueTurn, ...]

    def opening_statement(self) -> str:
        """Return combined text of the prepared opening remarks."""
        return " ".join(t.text for t in self.turns if t.segment == "statement")

    def chair_qa_text(self) -> str:
        """Return combined text of the Chair's Q&A answers."""
        return " ".join(t.text for t in self.turns if t.segment == "qa" and t.role == "chair")

    def press_questions_text(self) -> str:
        """Return combined text of all journalist questions."""
        return " ".join(t.text for t in self.turns if t.segment == "qa" and t.role == "press")


_SPEAKER_PATTERN = re.compile(
    r"^(CHAIR(?:MAN)?\s+[A-Z]+|PRESIDENT\s+[A-Z]+|GOVERNOR\s+[A-Z]+|"
    r"MR\.\s+[A-Z]+|MS\.\s+[A-Z]+|[A-Z\s]{3,30}):\s*(.*)",
    re.MULTILINE,
)

_CHAIR_KEYWORDS = {
    "CHAIR", "CHAIRMAN", "POWELL", "BERNANKE", "YELLEN", "VOLCKER", "GREENSPAN",
    "LAGARDE", "DRAGHI", "TRICHET", "RODRIGUEZ", "DIAZ DE LEON", "CARSTENS", "BAILEY", "CARNEY",
}


def parse_transcript(
    raw_text: str,
    *,
    title: str = "",
    date: str = "",
    institution: str = "FED",
) -> TranscriptDocument:
    """Parse raw press conference or testimony transcript into structured turns.

    Parameters
    ----------
    raw_text : str
        Full transcript text.
    title : str, default ""
        Document title.
    date : str, default ""
        Date of the event.
    institution : str, default 'FED'
        Central bank or governing body identifier.

    Returns
    -------
    TranscriptDocument
    """
    lines = raw_text.splitlines()
    turns: list[DialogueTurn] = []
    
    current_speaker: str | None = None
    current_role = "chair"
    current_segment = "statement"
    current_text_chunks: list[str] = []
    turn_idx = 0

    def _flush():
        nonlocal turn_idx
        if current_speaker is None:
            current_text_chunks.clear()
            return
        text = " ".join(current_text_chunks).strip()
        if text:
            turns.append(
                DialogueTurn(
                    speaker=current_speaker,
                    role=current_role,
                    segment=current_segment,
                    text=text,
                    turn_index=turn_idx,
                )
            )
            turn_idx += 1
        current_text_chunks.clear()

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check for Q&A section delimiter
        if re.search(r"QUESTION\s+AND\s+ANSWER\s+PERIOD|Q&A\s+SESSION|PREGUNTAS\s+Y\s+RESPUESTAS", line_clean, re.I):
            _flush()
            current_segment = "qa"
            continue

        # Check for speaker label header
        m = _SPEAKER_PATTERN.match(line_clean)
        if m:
            _flush()
            speaker_candidate = m.group(1).strip().upper()
            body_text = m.group(2).strip()

            # Classify speaker role
            words = set(re.findall(r"\b[A-Z]+\b", speaker_candidate))
            if words & _CHAIR_KEYWORDS:
                current_speaker = speaker_candidate
                current_role = "chair"
            elif any(w in ("QUESTION", "REPORTER", "PRESS", "MEDIA") for w in words):
                current_speaker = speaker_candidate
                current_role = "press"
                current_segment = "qa"
            else:
                current_speaker = speaker_candidate
                current_role = "press" if current_segment == "qa" else "unknown"

            if body_text:
                current_text_chunks.append(body_text)
        else:
            if current_speaker is not None:
                current_text_chunks.append(line_clean)

    _flush()
    return TranscriptDocument(
        title=title,
        date=date,
        institution=institution,
        turns=tuple(turns),
    )


__all__ = [
    "DialogueTurn",
    "TranscriptDocument",
    "parse_transcript",
]
