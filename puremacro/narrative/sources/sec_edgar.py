"""SEC EDGAR 10-K / 10-Q Corporate Macro Risk Extractor.

Extracts macroeconomic risk narratives from SEC corporate filings (Item 7: MD&A
and Item 1A: Risk Factors) across corporate sectors:
- Supply chain disruptions & shipping logistics.
- Tariffs, trade sanctions, and export controls.
- Interest rate exposure & debt refinancing pressures.
- Labor shortages, wage escalation, and collective bargaining.
- Inflation pass-through and input commodity prices.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

# Macro risk category patterns in corporate disclosures
CORPORATE_MACRO_PATTERNS = {
    "supply_chain": [
        r"supply\s+chain\s+(?:disruption|bottleneck|constraint|delay)",
        r"raw\s+material\s+(?:shortage|price\s+increase|inflation)",
        r"freight\s+costs?|shipping\s+rates?|container\s+shortages?",
    ],
    "tariffs_trade": [
        r"tariffs?|customs\s+duties|trade\s+(?:barriers?|disputes?|sanctions?)",
        r"export\s+controls?|geopolitical\s+tensions?",
    ],
    "interest_rates_credit": [
        r"higher\s+interest\s+rates?|rising\s+interest\s+rates?",
        r"borrowing\s+costs?|refinancing\s+risk|debt\s+service",
        r"credit\s+facility|liquidity\s+pressures?",
    ],
    "labor_costs": [
        r"wage\s+(?:inflation|pressures?|increases?)",
        r"labor\s+(?:shortage|turnover|retention|constraints?)",
        r"increased\s+compensation\s+costs?",
    ],
    "inflation_pricing": [
        r"inflationary\s+pressures?|cost\s+inflation",
        r"price\s+elasticity|ability\s+to\s+pass\s+through\s+costs",
        r"consumer\s+spending\s+(?:slowdown|reduction|weakness)",
    ],
}


@dataclass(frozen=True)
class CorporateMacroRiskMatch:
    """A matched macroeconomic risk passage in corporate disclosures."""
    category: str
    pattern_matched: str
    excerpt: str
    sentence_index: int


def extract_corporate_macro_risks(
    filing_text: str,
    *,
    categories: Sequence[str] | None = None,
) -> list[CorporateMacroRiskMatch]:
    """Scan corporate filing text (MD&A or Risk Factors) for macroeconomic risk narratives.

    Parameters
    ----------
    filing_text : str
        Raw filing text or extracted MD&A section.
    categories : sequence of str, optional
        Subset of risk categories to scan. Defaults to all in `CORPORATE_MACRO_PATTERNS`.

    Returns
    -------
    list of CorporateMacroRiskMatch
    """
    if categories is None:
        target_categories = list(CORPORATE_MACRO_PATTERNS.keys())
    else:
        target_categories = list(categories)

    sentences = [s.strip() for s in re.split(r"[.!?]\s+", filing_text) if len(s.strip()) > 20]
    matches: list[CorporateMacroRiskMatch] = []

    for s_idx, sentence in enumerate(sentences):
        for cat in target_categories:
            patterns = CORPORATE_MACRO_PATTERNS.get(cat, [])
            for pat in patterns:
                if re.search(pat, sentence, re.IGNORECASE):
                    matches.append(
                        CorporateMacroRiskMatch(
                            category=cat,
                            pattern_matched=pat,
                            excerpt=sentence,
                            sentence_index=s_idx,
                        )
                    )
                    break  # One match per category per sentence is enough

    return matches


__all__ = [
    "CORPORATE_MACRO_PATTERNS",
    "CorporateMacroRiskMatch",
    "extract_corporate_macro_risks",
]
