"""Structured Policy Action Classifier for narrative text.

Transforms raw narrative documents into typed NarrativeEvent objects with verified
policy kinds, targets, signs, magnitude extractions, and implementation profiles.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .quality.schedule_estimator import estimate_implementation_profile
from .scoring.multilingual import infer_policy_stance, score_multilingual
from .types import (
    VALID_KINDS,
    VALID_TARGETS_BY_KIND,
    NarrativeEvent,
)


# Regular expressions for magnitude extraction
_PCT_GDP_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent|por\s*ciento|prozent|pour\s*cent)\s*(?:of\s*GDP|del\s*PIB|des\s*BIP|du\s*PIB)",
    flags=re.IGNORECASE,
)
_BILLION_RX = re.compile(
    r"[$€£¥]?\s*(\d+(?:\.\d+)?)\s*(?:billion|bn|mil\s*millones|milliarden|milliards)",
    flags=re.IGNORECASE,
)
_BPS_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:basis\s*points|bps|puntos\s*base|basispunkte|points\s*de\s*base)",
    flags=re.IGNORECASE,
)


class PolicyActionClassifier:
    """Classifies narrative documents into structured, typed NarrativeEvents."""

    def __init__(
        self,
        *,
        default_confidence_floor: float = 0.3,
        estimate_schedules: bool = True,
    ):
        self.default_confidence_floor = default_confidence_floor
        self.estimate_schedules = estimate_schedules

    def _extract_magnitude(self, text: str) -> tuple[float, str]:
        """Extract magnitude and unit from text."""
        m_pct = _PCT_GDP_RX.search(text)
        if m_pct:
            return float(m_pct.group(1)), "pct_gdp"

        m_bn = _BILLION_RX.search(text)
        if m_bn:
            return float(m_bn.group(1)), "USD_bn"

        m_bps = _BPS_RX.search(text)
        if m_bps:
            return float(m_bps.group(1)), "bps"

        return 0.0, "z"

    def _classify_target(self, kind: str, text: str) -> tuple[str, str | None]:
        """Infer target and subtarget based on text keywords."""
        txt_low = text.lower()
        if kind == "fiscal":
            if any(k in txt_low for k in ["infrastructure", "infraestructura", "infrastruktur", "public investment", "inversión pública"]):
                return "investment", "infra"
            if any(k in txt_low for k in ["defense", "defensa", "military", "rüstungs", "procurement"]):
                return "investment", "defense"
            if any(k in txt_low for k in ["transfer", "transferencia", "social aid", "subsidy", "subsidio", "zuschuss", "aide"]):
                return "consumption", "transfers"
            if any(k in txt_low for k in ["tax cut", "tax hike", "tax reform", "reforma tributaria", "steuersenkung", "impôt"]):
                return "consumption", "tax"
            return "both", "general"

        if kind == "monetary":
            if any(k in txt_low for k in ["rate", "tasa", "leitzins", "taux", "selic"]):
                return "policy_rate", "rate_decision"
            if any(k in txt_low for k in ["asset purchase", "quantitative easing", "anleihekauf", "qe"]):
                return "asset_purchase", "qe"
            if any(k in txt_low for k in ["forward guidance", "orientación"]):
                return "forward_guidance", "guidance"
            return "policy_rate", "general"

        if kind == "macropru":
            if any(k in txt_low for k in ["buffer", "colchón", "puffer"]):
                return "capital_buffer", "ccyb"
            if any(k in txt_low for k in ["ltv", "dsti", "loan-to-value"]):
                return "ltv_dsti", "mortgage"
            return "capital_buffer", "general"

        if kind == "fx":
            if any(k in txt_low for k in ["peg", "paridad", "banda"]):
                return "peg_change", "peg"
            return "intervention", "spot_fx"

        if kind == "structural":
            if any(k in txt_low for k in ["labor", "laboral", "arbeitsmarkt", "emploi"]):
                return "labor", "reform"
            return "product_market", "reform"

        return "both", None

    def classify_document(self, doc: Any) -> NarrativeEvent | None:
        """Classify a single narrative document into a NarrativeEvent.

        Accepts a NarrativeDocument, dict, or tuple.
        """
        # Unpack fields
        if hasattr(doc, "text") and hasattr(doc, "date"):
            date = doc.date
            text = doc.text
            url = doc.url
            country = doc.country
            lang = getattr(doc, "language", "en")
            source = getattr(doc, "source", "narrative_harvester")
            policy_domain = getattr(doc, "policy_domain", "fiscal")
            raw_mag = getattr(doc, "magnitude", None)
            meta = getattr(doc, "metadata", {}) or {}
        elif isinstance(doc, dict):
            date = doc.get("date")
            text = doc.get("text", "")
            url = doc.get("url", "")
            country = doc.get("country", "USA")
            lang = doc.get("language", "en")
            source = doc.get("source", "narrative_harvester")
            policy_domain = doc.get("policy_domain", "fiscal")
            raw_mag = doc.get("magnitude")
            meta = doc.get("metadata", {})
        elif isinstance(doc, tuple):
            if len(doc) == 4:
                date, text, url, meta = doc
                raw_mag = None
            elif len(doc) == 5:
                date, text, url, meta, raw_mag = doc
            else:
                return None
            country = meta.get("country", "USA")
            lang = meta.get("language", "en")
            source = meta.get("source", "narrative_connector")
            policy_domain = meta.get("policy_domain", "fiscal")
        else:
            return None

        if not text:
            return None

        # Resolve kind
        kind = policy_domain if policy_domain in VALID_KINDS else "fiscal"

        # Resolve target & subtarget
        target, subtarget = self._classify_target(kind, text)

        # Infer sign and confidence
        sign, conf, details = infer_policy_stance(text, language=lang, domain=kind)
        if conf < self.default_confidence_floor and sign == 0:
            return None

        # Resolve magnitude
        if raw_mag is not None and float(raw_mag) > 0:
            magnitude = float(raw_mag)
            unit = "pct_gdp"
        else:
            magnitude, unit = self._extract_magnitude(text)

        # Estimate implementation profile
        if self.estimate_schedules:
            profile = estimate_implementation_profile(
                announcement_date=date,
                kind=kind,
                target=target,
                subtarget=subtarget,
            )
        else:
            profile = []

        event = NarrativeEvent(
            date=pd.Timestamp(date),
            country=country.lower(),
            magnitude=magnitude,
            magnitude_unit=unit,
            target=target,
            subtarget=subtarget,
            sign=sign,
            confidence=conf,
            source_text=text[:300].strip(),
            source_url=url or "",
            scoring_method="keyword",
            metadata={
                **meta,
                "source": source,
                "language": lang,
                "classification_details": details,
            },
            implementation_profile=profile,
            kind=kind,
            language=lang,
        )
        return event

    def classify_corpus(self, corpus: Iterable[Any]) -> list[NarrativeEvent]:
        """Classify an entire sequence/corpus of documents into NarrativeEvents."""
        events: list[NarrativeEvent] = []
        for doc in corpus:
            ev = self.classify_document(doc)
            if ev is not None:
                events.append(ev)
        return events


__all__ = ["PolicyActionClassifier"]
