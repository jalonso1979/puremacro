"""Keyword + regex-based event scoring (deterministic baseline).

Each text gets a single ``NarrativeEvent`` (or zero, if no policy
keywords are matched). The heuristics here are intentionally simple
— they're a fast, free, audit-friendly baseline against which LLM
scoring is compared, not a state-of-the-art classifier.
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from ..types import NarrativeEvent


# Default fiscal lexicon: maps lowercase token regex → (target, sign,
# subtarget). The first match wins. Order is significant.
DEFAULT_FISCAL_LEXICON = [
    # Investment-style language.
    (r"\binfrastructure\b", ("investment", +1, "infra")),
    (r"\bbroadband\b",       ("investment", +1, "infra")),
    (r"\bhighway\b",         ("investment", +1, "infra")),
    (r"\bdefense procurement\b", ("investment", +1, "defense")),
    (r"\bweapons program\b", ("investment", +1, "defense")),
    (r"\bresearch and development\b", ("investment", +1, "rd")),
    (r"\bcapital expenditure\b", ("investment", +1, "general")),
    (r"\bpublic investment\b", ("investment", +1, "general")),
    (r"\bcapital budget\b", ("investment", +1, "general")),
    # Consumption-style language.
    (r"\bgovernment purchases\b", ("consumption", +1, "general")),
    (r"\bgovernment wages\b", ("consumption", +1, "wages")),
    (r"\bfederal salaries\b", ("consumption", +1, "wages")),
    (r"\btransfer payments\b", ("consumption", +1, "transfers")),
    (r"\bunemployment benefits\b", ("consumption", +1, "transfers")),
    (r"\bsocial security expansion\b", ("consumption", +1, "transfers")),
    # "Both" / ambiguous.
    (r"\bstimulus package\b", ("both", +1, "stimulus")),
    (r"\brecovery act\b",      ("both", +1, "stimulus")),
    (r"\bbudget increase\b",   ("both", +1, "general")),
    # Contractionary signals.
    (r"\bspending cut\b",       ("both", -1, "general")),
    (r"\baustere?ity\b",        ("both", -1, "general")),
    (r"\bsequester\b",          ("both", -1, "general")),
    (r"\bbudget freeze\b",      ("both", -1, "general")),
    (r"\bdefense cut\b",        ("investment", -1, "defense")),
]


# Default monetary lexicon (English). First match wins; hawkish vs dovish
# encoded by sign.
DEFAULT_MONETARY_LEXICON = [
    # Hawkish (+1) — tightening / hike / QT
    (r"\brais(?:e|ed|es|ing)\s+(?:the\s+)?(?:federal funds rate|policy rate|interest rate|bank rate|target range)\b",
                                                  ("policy_rate", +1, "rate_up")),
    (r"\brate\s+(?:hike|increase)\b",             ("policy_rate", +1, "rate_up")),
    (r"\btighten(?:ed|s|ing)?\s+monetary\s+policy\b",
                                                  ("policy_rate", +1, "tighten")),
    (r"\bhik(?:e|ed|es|ing)\s+(?:rates|the policy rate)\b",
                                                  ("policy_rate", +1, "rate_up")),
    (r"\bquantitative tightening\b",              ("asset_purchase", +1, "qt")),
    (r"\bbalance[- ]sheet runoff\b",              ("asset_purchase", +1, "qt")),
    (r"\bwithdraw(?:s|ing|n)?\s+accommodation\b", ("forward_guidance", +1, "hawkish")),
    # Dovish (-1) — cut / ease / QE
    (r"\brate cut\b",                             ("policy_rate", -1, "rate_down")),
    (r"\bcut(?:s|ting)?\s+(?:the\s+)?(?:federal funds rate|policy rate|interest rate|bank rate|target range)\b",
                                                  ("policy_rate", -1, "rate_down")),
    (r"\blower(?:ed|s|ing)?\s+(?:the\s+)?(?:federal funds rate|policy rate|interest rate|bank rate|target range)\b",
                                                  ("policy_rate", -1, "rate_down")),
    (r"\beas(?:e|ed|es|ing)\s+monetary\s+policy\b",
                                                  ("policy_rate", -1, "ease")),
    (r"\bquantitative easing\b",                  ("asset_purchase", -1, "qe")),
    (r"\basset[- ]purchase (?:expansion|programme|program)\b",
                                                  ("asset_purchase", -1, "qe")),
    (r"\bforward guidance.*lower for longer\b",   ("forward_guidance", -1, "dovish")),
]


_LEXICON_BY_KIND = {
    "fiscal":   DEFAULT_FISCAL_LEXICON,
    "monetary": DEFAULT_MONETARY_LEXICON,
}


def regex_billions(text: str) -> float:
    """Extract the largest dollar magnitude in billions.

    Recognises:
        '$5.2 billion', '5,200 million', '$3.4B', '7.1 trillion',
        '500 million dollars', 'USD 1.2 billion'.
    Returns 0.0 if nothing is found.
    """
    text = text.replace(",", "")
    patterns = [
        (r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*trillion\b", 1000.0),
        (r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*billion\b", 1.0),
        (r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*B\b", 1.0),
        (r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*million\b", 1.0e-3),
        (r"\$?\s*([0-9]+(?:\.[0-9]+)?)\s*M\b", 1.0e-3),
    ]
    best = 0.0
    for rx, scale in patterns:
        for m in re.finditer(rx, text, flags=re.IGNORECASE):
            try:
                v = float(m.group(1)) * scale
                if v > best:
                    best = v
            except ValueError:
                continue
    return best


def regex_basis_points(text: str) -> float:
    """Extract basis-point magnitude from text. Returns 0.0 if absent."""
    text_lower = text.lower()
    patterns = [
        (r"\b([0-9]+(?:\.[0-9]+)?)\s*basis points?\b", 1.0),
        (r"\b([0-9]+(?:\.[0-9]+)?)\s*bps?\b", 1.0),
        (r"\bby ([0-9]+(?:\.[0-9]+)?)\s*percentage points?\b", 100.0),
    ]
    best = 0.0
    for rx, mult in patterns:
        for m in re.finditer(rx, text_lower):
            try:
                v = float(m.group(1)) * mult
                if v > best:
                    best = v
            except ValueError:
                continue
    return best


_MAGNITUDE_EXTRACTOR_BY_KIND = {
    "fiscal":   regex_billions,
    "monetary": regex_basis_points,
}

_MAGNITUDE_UNIT_BY_KIND = {
    "fiscal":   "USD_bn",
    "monetary": "bps",
}


def _classify(text: str, lexicon=None) -> tuple[str, int, str | None] | None:
    """Return (target, sign, subtarget) by first-match against lexicon."""
    lex = lexicon or DEFAULT_FISCAL_LEXICON
    for rx, (target, sign, subtarget) in lex:
        if re.search(rx, text, flags=re.IGNORECASE):
            return target, sign, subtarget
    return None


def score_keyword(
    text_iter: Iterable[tuple],
    *,
    country: str = "USA",
    kind: str = "fiscal",
    lexicon=None,
    magnitude_extractor=None,
    confidence: float = 0.5,
    language: str = "en",
) -> list[NarrativeEvent]:
    """Score each text in ``text_iter`` to at most one event.

    Parameters
    ----------
    text_iter : iterable of ``(date, text, source_url)`` 3-tuples or
        ``(date, text, source_url, metadata)`` 4-tuples (metadata ignored).
    country : ISO3 code applied to every event.
    kind : ``"fiscal"`` (default) or ``"monetary"``. Selects the default
        lexicon, magnitude extractor, and magnitude unit. Other kinds
        require explicit ``lexicon=`` and ``magnitude_extractor=``.
    lexicon : list of ``(regex, (target, sign, subtarget))`` tuples.
        If ``None``, picked by ``kind``.
    magnitude_extractor : callable text → float in the unit for ``kind``.
        If ``None``, picked by ``kind``.
    confidence : flat confidence assigned to keyword-scored events.
    language : ISO-639-1 stamped on each event.

    Returns
    -------
    list[NarrativeEvent]. Texts with no lexicon match contribute zero
    events.
    """
    if kind not in _LEXICON_BY_KIND and lexicon is None:
        raise ValueError(
            f"kind {kind!r}: no built-in lexicon. Pass lexicon= explicitly, "
            f"or use kind in {sorted(_LEXICON_BY_KIND)}."
        )
    if lexicon is None:
        lexicon = _LEXICON_BY_KIND[kind]
    if magnitude_extractor is None:
        magnitude_extractor = _MAGNITUDE_EXTRACTOR_BY_KIND.get(kind, regex_billions)
    magnitude_unit = _MAGNITUDE_UNIT_BY_KIND.get(kind, "USD_bn")

    out: list[NarrativeEvent] = []
    for record in text_iter:
        if len(record) == 4:
            date, text, source_url, _meta = record
        else:
            date, text, source_url = record
        cls = _classify(text, lexicon=lexicon)
        if cls is None:
            continue
        target, sign, subtarget = cls
        magnitude = magnitude_extractor(text)
        if magnitude == 0.0:
            local_conf = 0.25
        else:
            local_conf = confidence
        out.append(NarrativeEvent(
            date=pd.Timestamp(date),
            country=country,
            magnitude=magnitude,
            magnitude_unit=magnitude_unit,
            target=target,
            subtarget=subtarget,
            sign=sign,
            confidence=local_conf,
            source_text=text[:500],
            source_url=str(source_url),
            scoring_method="keyword",
            kind=kind,
            language=language,
        ))
    return out


__all__ = ["DEFAULT_FISCAL_LEXICON", "DEFAULT_MONETARY_LEXICON",
           "regex_billions", "regex_basis_points", "score_keyword"]
