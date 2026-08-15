"""Tests for narrative burst anomaly detection."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.anomalies import detect_narrative_bursts


def test_detect_narrative_bursts():
    # Build 14 months of synthetic texts where "tariffs" suddenly spikes in the 14th month (2025-02-01)
    dates = pd.date_range("2024-01-01", periods=14, freq="MS")
    dated_texts = []
    
    # Months 1 to 13: standard economic discussion without "tariffs"
    for d in dates[:-1]:
        dated_texts.append((d, "Economic activity is expanding with steady job growth and stable inflation."))
        dated_texts.append((d, "Monetary policy remains focused on achieving price stability and employment."))
        
    # Month 14: sudden surge in "tariffs"
    target_d = dates[-1]
    for _ in range(5):
        dated_texts.append((target_d, "Trade tensions erupted with major new tariffs announced on imports."))
        dated_texts.append((target_d, "The new tariffs policy increased tariff rates and trade dispute uncertainty."))
        
    bursts = detect_narrative_bursts(
        dated_texts,
        target_date=target_d,
        window_periods=12,
        freq="MS",
        min_count=2,
    )
    
    assert len(bursts) > 0
    burst_terms = [b.term for b in bursts]
    assert "tariffs" in burst_terms or "trade" in burst_terms
    
    top_burst = bursts[0]
    assert top_burst.z_score > 1.5
    assert top_burst.burst_magnitude > 1.2
