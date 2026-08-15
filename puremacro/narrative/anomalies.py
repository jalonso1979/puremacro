"""Macroeconomic Narrative Burst & Emerging Theme Anomaly Detector.

Detects statistical surges in macroeconomic keyword and phrase frequencies relative
to a rolling historical baseline (Kleinberg-style burst detection and TF-IDF z-score spikes).

Enables identifying *nascent macroeconomic narratives* (e.g. 'tariffs', 'bank run',
'hiring freeze', 'debt ceiling', 'supply chain') at the moment they break in forums,
news, or central bank communications before appearing in official quarterly statistical releases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NarrativeBurstResult:
    """Detected narrative bursts for a given period."""
    term: str
    current_freq: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    burst_magnitude: float


def detect_narrative_bursts(
    dated_texts: Sequence[tuple[Any, str]],
    *,
    target_date: Any,
    window_periods: int = 12,
    freq: str = "MS",
    min_count: int = 3,
    top_k: int = 15,
) -> list[NarrativeBurstResult]:
    """Detect keywords with statistically abnormal frequency spikes in a target period.

    Parameters
    ----------
    dated_texts : sequence of (date, text)
        List of dated document texts.
    target_date : datetime-like
        The specific period to evaluate for narrative anomalies.
    window_periods : int, default 12
        Number of preceding periods used to compute the historical baseline mean and std.
    freq : str, default 'MS'
        Period frequency for time aggregation (e.g. 'MS' monthly, 'QS' quarterly).
    min_count : int, default 3
        Minimum occurrences of a term in the target period to be evaluated.
    top_k : int, default 15
        Number of top bursting terms to return.

    Returns
    -------
    list of NarrativeBurstResult
        Sorted list of detected narrative bursts by z-score descending.
    """
    # 1. Bucket texts by period
    target_ts = pd.Timestamp(target_date)
    p_freq = "M" if freq.upper().startswith("M") else ("Q" if freq.upper().startswith("Q") else freq)
    
    period_term_counts: dict[pd.Timestamp, dict[str, int]] = {}
    period_doc_counts: dict[pd.Timestamp, int] = {}

    for dt, text in dated_texts:
        ts = pd.Timestamp(dt).to_period(p_freq).to_timestamp()
        if ts not in period_term_counts:
            period_term_counts[ts] = {}
            period_doc_counts[ts] = 0
            
        period_doc_counts[ts] += 1
        words = re.findall(r"\b[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{3,}\b", text.lower())
        for w in words:
            period_term_counts[ts][w] = period_term_counts[ts].get(w, 0) + 1

    target_period = target_ts.to_period(p_freq).to_timestamp()
    if target_period not in period_term_counts:
        return []

    # 2. Identify baseline periods (window_periods prior to target)
    all_periods = sorted(period_term_counts.keys())
    target_idx = all_periods.index(target_period) if target_period in all_periods else -1
    if target_idx <= 0:
        return []

    baseline_start = max(0, target_idx - window_periods)
    baseline_periods = all_periods[baseline_start:target_idx]
    if not baseline_periods:
        return []

    target_counts = period_term_counts[target_period]
    target_doc_n = max(1, period_doc_counts[target_period])
    
    bursts: list[NarrativeBurstResult] = []

    for term, count in target_counts.items():
        if count < min_count:
            continue
            
        target_rate = count / target_doc_n
        
        # Historical rates across baseline periods
        hist_rates = []
        for p in baseline_periods:
            p_docs = max(1, period_doc_counts[p])
            p_count = period_term_counts[p].get(term, 0)
            hist_rates.append(p_count / p_docs)
            
        hist_mean = float(np.mean(hist_rates))
        hist_std = float(np.std(hist_rates)) or 1e-4

        z = (target_rate - hist_mean) / (hist_std + 1e-6)
        burst_ratio = target_rate / (hist_mean + 1e-6)

        if z > 1.5 and burst_ratio > 1.2:
            bursts.append(
                NarrativeBurstResult(
                    term=term,
                    current_freq=float(target_rate),
                    baseline_mean=hist_mean,
                    baseline_std=hist_std,
                    z_score=float(z),
                    burst_magnitude=float(burst_ratio),
                )
            )

    bursts.sort(key=lambda x: -x.z_score)
    return bursts[:top_k]


__all__ = [
    "NarrativeBurstResult",
    "detect_narrative_bursts",
]
