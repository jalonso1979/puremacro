"""Tests for multi-speaker transcript parser."""
from __future__ import annotations

import pytest

from puremacro.narrative.sources.transcripts import (
    DialogueTurn,
    TranscriptDocument,
    parse_transcript,
)


def test_parse_transcript_speakers_and_segments():
    raw_sample = """
    FEDERAL RESERVE PRESS CONFERENCE
    WASHINGTON, D.C.
    MARCH 20, 2024
    
    CHAIR POWELL: Good afternoon. My colleagues and I remain squarely focused on our dual mandate to promote maximum employment and stable prices. Today, the FOMC decided to leave our policy interest rate unchanged.
    
    QUESTION AND ANSWER PERIOD
    
    STEVE LIESMAN: Steve Liesman, CNBC. Mr. Chairman, given the recent uptick in inflation data, are you considering delaying interest rate cuts?
    
    CHAIR POWELL: Thanks Steve. As I noted, we believe our policy rate is likely at its peak for this tightening cycle. We will carefully assess incoming data before making any adjustments.
    
    JEANNA SMIALEK: Jeanna Smialek, New York Times. What conditions in the labor market would prompt you to act sooner?
    
    CHAIR POWELL: An unexpected weakening in the labor market would certainly warrant a policy response.
    """
    doc = parse_transcript(raw_sample, title="FOMC March 2024", date="2024-03-20", institution="FED")
    
    assert doc.title == "FOMC March 2024"
    assert doc.institution == "FED"
    assert len(doc.turns) >= 4
    
    # Check opening statement
    statement = doc.opening_statement()
    assert "dual mandate" in statement
    assert "leave our policy interest rate unchanged" in statement
    
    # Check Q&A separation
    chair_qa = doc.chair_qa_text()
    assert "likely at its peak" in chair_qa
    assert "unexpected weakening in the labor market" in chair_qa
    
    press_qa = doc.press_questions_text()
    assert "Steve Liesman" in press_qa or "uptick in inflation" in press_qa
    assert "Jeanna Smialek" in press_qa or "labor market" in press_qa
