"""Multinomial-logit paragraph-level scoring kernel (Cluster B, item B1).

A Picault-Renault-style MNL classifies each text window (sentence or
paragraph) into one of K topical or stance categories. Each term has
a per-category coefficient; the per-paragraph score for category k is

    P(k | window) = softmax_k( bias_k + Σ_t β_{t,k} · count_t(window) )

Aggregated per document as the *mean* of paragraph-level
P(category=target | paragraph), giving a continuous score in [0, 1].

Compared with the lexicon co-occurrence kernels this captures:
  - Term polarity: "displacement" pulls toward downside; "augmentation"
    pulls toward upside — even if both appear in the labor-uncertainty
    sentence.
  - Term weighting: a high-coefficient term contributes more than a
    low-coefficient one.

Limitations:
  - **Requires pre-trained weights.** Picault-Renault 2017 published
    coefficients are not bundled with puremacro (license unclear).
  - Bag-of-words representation: ignores word order.

Pyodide-clean (pure numpy + frozenset lexicon scan).
"""
from __future__ import annotations

import math
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from ._kernels import _split_paragraphs, _split_sentences, count_keywords


# weights schema:
#   {"terms": [t1, t2, ...],
#    "categories": [c1, c2, ...],
#    "beta": np.ndarray of shape (T, K),
#    "bias": np.ndarray of shape (K,)}
#
# A standalone JSON serialiser is below. Weights can also be passed as
# a dict-of-dicts mapping ``term -> category -> coefficient``; the
# canonicaliser converts that to the array form.

def canonicalize_weights(weights: dict) -> dict:
    """Normalise a weights dict to the array-keyed schema used by the kernel.

    Accepts either:
        (a) {"terms":..., "categories":..., "beta": [[...]], "bias": [...]}
        (b) {"beta": {term: {category: coef, ...}, ...},
             "bias": {category: coef, ...}}
    """
    if "terms" in weights:
        terms = list(weights["terms"])
        cats = list(weights["categories"])
        beta = np.asarray(weights["beta"], dtype=float)
        bias = np.asarray(weights["bias"], dtype=float)
        if beta.shape != (len(terms), len(cats)):
            raise ValueError(
                f"beta shape {beta.shape} ≠ (n_terms={len(terms)}, "
                f"n_categories={len(cats)})"
            )
        if bias.shape != (len(cats),):
            raise ValueError(
                f"bias shape {bias.shape} ≠ (n_categories={len(cats)},)"
            )
        return {"terms": terms, "categories": cats, "beta": beta, "bias": bias}

    if "beta" in weights and isinstance(weights["beta"], dict):
        beta_dict = weights["beta"]
        bias_dict = weights.get("bias", {})
        cats = sorted({
            c for term_row in beta_dict.values() for c in term_row.keys()
        } | set(bias_dict.keys()))
        terms = sorted(beta_dict.keys())
        beta = np.zeros((len(terms), len(cats)), dtype=float)
        for i, t in enumerate(terms):
            for j, c in enumerate(cats):
                beta[i, j] = float(beta_dict[t].get(c, 0.0))
        bias = np.array(
            [float(bias_dict.get(c, 0.0)) for c in cats], dtype=float,
        )
        return {"terms": terms, "categories": cats, "beta": beta, "bias": bias}

    raise ValueError(
        "weights must contain either ('terms', 'categories', 'beta', 'bias') "
        "or a dict-of-dicts 'beta'"
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax along the last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _count_vector(
    text: str,
    terms: Sequence[str],
    language: str,
    negation: bool = False,
) -> np.ndarray:
    """Return a length-T count vector of term frequencies in ``text``."""
    out = np.zeros(len(terms), dtype=float)
    if not text:
        return out
    for i, t in enumerate(terms):
        out[i] = float(count_keywords(text, frozenset({t}), language=language,
                                       negation=negation))
    return out


def _score_window(
    window: str,
    weights: dict,
    *,
    target_idx: int,
    language: str,
    negation: bool,
) -> float:
    """P(category=target | window) via stable softmax over linear logits."""
    counts = _count_vector(window, weights["terms"], language=language,
                           negation=negation)
    logits = weights["bias"] + counts @ weights["beta"]
    return float(_softmax(logits)[target_idx])


def mnl_kernel(
    records: Iterable[tuple],
    *,
    weights: dict,
    category: str,
    window: str = "paragraph",
    language: str = "en",
    negation: bool = False,
) -> Iterator[tuple]:
    """Yield ``(date, mean_P_category)`` per document.

    For each document, splits into windows (sentence or paragraph),
    computes per-window softmax probability of ``category``, returns
    the mean.

    Parameters
    ----------
    weights : dict
        Schema described at top of module. Pass the dict-of-dicts form
        for convenience; will be canonicalised internally.
    category : str
        The category label (must appear in ``weights["categories"]``)
        whose probability we report.
    window : "paragraph" | "sentence"
    language : ISO-639-1 (selects ``count_keywords`` behaviour).
    negation : if True, suppress hits within 4 tokens after a negation
        marker (see ``_kernels.count_keywords``).
    """
    if window not in ("paragraph", "sentence"):
        raise ValueError(f"window must be 'sentence' or 'paragraph'; got {window!r}")
    canon = canonicalize_weights(weights)
    if category not in canon["categories"]:
        raise ValueError(
            f"category {category!r} not in {canon['categories']}"
        )
    target_idx = canon["categories"].index(category)

    splitter = _split_paragraphs if window == "paragraph" else _split_sentences

    for record in records:
        if len(record) == 4:
            date, text, _, meta = record
            lang = (meta or {}).get("language", language)
        else:
            date, text, _ = record
            lang = language
        windows = splitter(text or "", lang)
        if not windows:
            yield (pd.Timestamp(date), 0.0)
            continue
        scores = [
            _score_window(w, canon, target_idx=target_idx,
                          language=lang, negation=negation)
            for w in windows
        ]
        yield (pd.Timestamp(date), float(np.mean(scores)))


__all__ = ["mnl_kernel", "canonicalize_weights"]
