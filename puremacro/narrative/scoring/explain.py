"""Shapley keyword and sentence attribution for narrative macro indices.

Explains the drivers of peaks, troughs, and shifts in text-derived uncertainty,
sentiment, or policy stance indices using leave-one-out and Shapley-style
marginal contributions.
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np
import pandas as pd


def shapley_keyword_attribution(
    text: str,
    scoring_fn: Callable[[str], float],
    top_k: int = 10,
) -> pd.DataFrame:
    """Compute marginal contribution (leave-one-out Shapley attribution) of words to a text score.

    Parameters
    ----------
    text : str
        The raw document text.
    scoring_fn : callable
        Function taking a string and returning a float score.
    top_k : int, default 10
        Number of top positive and negative contributing words to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['word', 'count', 'marginal_contribution', 'direction'].
    """
    base_score = scoring_fn(text)
    words = re.findall(r"\b[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]{3,}\b", text)
    unique_words = sorted(set(w.lower() for w in words))
    
    word_counts: dict[str, int] = {}
    for w in words:
        wl = w.lower()
        word_counts[wl] = word_counts.get(wl, 0) + 1

    records = []
    for w in unique_words:
        # Pattern to replace word with empty string
        pattern = re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)
        ablated_text = pattern.sub("", text)
        ablated_score = scoring_fn(ablated_text)
        # Marginal contribution is: score with word - score without word
        delta = base_score - ablated_score
        
        if abs(delta) > 1e-7:
            direction = "positive" if delta > 0 else "negative"
            records.append({
                "word": w,
                "count": word_counts[w],
                "marginal_contribution": float(delta),
                "direction": direction,
            })

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["word", "count", "marginal_contribution", "direction"])
        
    # Sort by absolute marginal contribution descending
    df["abs_contrib"] = df["marginal_contribution"].abs()
    df = df.sort_values(by="abs_contrib", ascending=False).drop(columns=["abs_contrib"]).head(top_k)
    return df.reset_index(drop=True)


def explain_sentence_contributions(
    text: str,
    scoring_fn: Callable[[str], float],
    top_k: int = 5,
) -> pd.DataFrame:
    """Compute the top sentences driving a document's narrative score.

    Parameters
    ----------
    text : str
        Full document text.
    scoring_fn : callable
        Function mapping string to float score.
    top_k : int, default 5
        Number of sentences to return.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['sentence', 'marginal_contribution', 'standalone_score'].
    """
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 15]
    if not sentences:
        return pd.DataFrame(columns=["sentence", "marginal_contribution", "standalone_score"])

    base_score = scoring_fn(text)
    records = []
    for s in sentences:
        ablated = text.replace(s, "")
        delta = base_score - scoring_fn(ablated)
        standalone = scoring_fn(s)
        records.append({
            "sentence": s,
            "marginal_contribution": float(delta),
            "standalone_score": float(standalone),
        })

    df = pd.DataFrame(records)
    df["abs_contrib"] = df["marginal_contribution"].abs()
    df = df.sort_values(by="abs_contrib", ascending=False).drop(columns=["abs_contrib"]).head(top_k)
    return df.reset_index(drop=True)


__all__ = [
    "shapley_keyword_attribution",
    "explain_sentence_contributions",
]
