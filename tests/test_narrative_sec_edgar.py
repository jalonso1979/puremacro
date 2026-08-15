"""Tests for SEC EDGAR corporate macro risk extractor."""
from __future__ import annotations

import pytest

from puremacro.narrative.sources.sec_edgar import (
    CORPORATE_MACRO_PATTERNS,
    CorporateMacroRiskMatch,
    extract_corporate_macro_risks,
)


def test_extract_corporate_macro_risks():
    sample_mda = """
    Item 7. Management's Discussion and Analysis of Financial Condition.
    Our operating margins experienced pressure due to raw material price increases and freight costs across international routes.
    Furthermore, rising interest rates increased our borrowing costs under the revolving credit facility.
    The company faced acute wage pressures and labor shortages in key manufacturing facilities, raising compensation expenses.
    New trade barriers and tariffs enacted in overseas markets could adversely affect future export volumes.
    """
    matches = extract_corporate_macro_risks(sample_mda)
    assert len(matches) >= 3
    
    categories = [m.category for m in matches]
    assert "supply_chain" in categories
    assert "interest_rates_credit" in categories
    assert "labor_costs" in categories
    assert "tariffs_trade" in categories
    
    for m in matches:
        assert isinstance(m, CorporateMacroRiskMatch)
        assert len(m.excerpt) > 20
