"""Validation cases for the ``narrative`` subsystem.

Scope: the deterministic, offline text-index core — ``count_keywords``
(plus its negation-aware variant), the Baker-Bloom-Davis co-occurrence
EPU builder, and the Ahir-Bloom-Furceri length-normalised WUI builder.
Every reference here is ANALYTICAL (a hand-computed value on a tiny
crafted corpus) or INTERNAL (a structural identity: monotonicity in
keyword frequency, and the mean/std fixed points of the z-score and
BBD-100 normalisations). No PACKAGE/PUBLISHED references → no frozen
goldens and no drift-guard: the LLM and live-fetch paths are out of
scope (see ``skipped`` in the gallery report).

Pure-deps discipline: the puremacro estimators (which pull in pandas)
are imported INSIDE the compute callables, so importing this module
stays Pyodide-light. The demo corpora are defined by the seeded helper
``_narrative_demo_corpus`` below — local to this file, not the shared
``_fixtures.py``.
"""
from __future__ import annotations

import numpy as np

from ._model import Mechanism, Tol, ValidationCase


# --------------------------------------------------------------------------- #
# Seeded demo corpora. These are *crafted* (not random) so every reference is
# a value computed by hand from the text — the seed only documents intent and
# guarantees byte-identical inputs across calls / processes.
# --------------------------------------------------------------------------- #
_DEMO_SEED = 20260531


def _narrative_demo_corpus() -> dict:
    """Deterministic, hand-countable text fixtures shared by the cases.

    Returns a dict of corpora keyed by case family. Each corpus is a list
    of ``(date_iso, text, source_url, metadata)`` 4-tuples — the record
    shape consumed by the indices layer — except ``count_*`` which expose
    bare strings for the per-document ``count_keywords`` kernel.
    """
    rng = np.random.default_rng(_DEMO_SEED)  # documents the fixture as seeded
    _ = rng  # (corpora below are crafted, not sampled — kept for provenance)

    # -- count_keywords: a single crafted paragraph with KNOWN hit counts. --
    # "uncertainty" appears 4x (case-insensitive, whole-word); "economic"
    # appears 2x; "uncertain" appears 0x (no bare whole-word occurrence).
    count_text = (
        "Economic uncertainty about economic policy persists. "
        "Uncertainty, uncertainty, UNCERTAINTY!"
    )

    # -- negation: two sentences, each with exactly one "uncertainty". The
    # first is preceded by the marker "no" within the 4-token window and is
    # suppressed when negation=True; the second is clean. Counts: plain=2,
    # negation-aware=1.
    neg_text = (
        "There is no uncertainty in the baseline outlook. "
        "Markets nonetheless priced fresh uncertainty into long bonds."
    )

    # -- EPU co-occurrence (Economy / Policy / Uncertainty), raw mean over a
    # single quarter (2020Q1) of three documents. Doc1 hits all three groups
    # (-> 1.0), Doc2 misses policy+uncertainty (-> 0.0), Doc3 hits all three
    # (-> 1.0). Quarterly mean = (1+0+1)/3 = 2/3.
    epu_docs: list[tuple[str, str, str, dict]] = [
        ("2020-01-15", "The economy faces serious policy uncertainty ahead.", "doc://1", {}),
        ("2020-02-15", "The economy is fundamentally strong this year.", "doc://2", {}),
        ("2020-03-15", "Policy debates, the economy, and uncertainty dominate.", "doc://3", {}),
    ]

    # -- WUI length-normalisation: one document, "uncertainty" twice out of 7
    # whitespace tokens -> (2 / 7) * 1000 hits-per-1000-words.
    wui_one_doc: list[tuple[str, str, str, dict]] = [
        ("2021-01-10", "uncertainty uncertainty about the economy here now", "doc://w", {}),
    ]

    # -- WUI normalisation fixed points + monotonicity: six quarters with a
    # controlled, varying number of "uncertainty" hits per single document so
    # the cross-quarter series has positive dispersion.
    wui_norm_specs = [
        ("2020-01-10", 1), ("2020-04-10", 3), ("2020-07-10", 0),
        ("2020-10-10", 5), ("2021-01-10", 2), ("2021-04-10", 4),
    ]
    wui_norm_docs: list[tuple[str, str, str, dict]] = [
        (d, " ".join(["uncertainty"] * n + ["fixed", "filler", "tokens"]), "doc://n", {})
        for d, n in wui_norm_specs
    ]

    # -- Monotonicity: corpus B has >= the per-quarter hit count of corpus A
    # in every quarter, with identical lengths-modulo-the-extra-keywords, so
    # the raw length-normalised index must be weakly larger everywhere.
    mono_docs_a: list[tuple[str, str, str, dict]] = [
        ("2020-01-10", "uncertainty filler tokens here", "doc://a", {}),
        ("2020-04-10", "uncertainty uncertainty filler tokens here", "doc://a", {}),
    ]
    mono_docs_b: list[tuple[str, str, str, dict]] = [
        ("2020-01-10", "uncertainty uncertainty filler tokens here", "doc://b", {}),
        ("2020-04-10", "uncertainty uncertainty uncertainty uncertainty filler tokens here", "doc://b", {}),
    ]

    return {
        "count_text": count_text,
        "neg_text": neg_text,
        "epu_docs": epu_docs,
        "wui_one_doc": wui_one_doc,
        "wui_norm_docs": wui_norm_docs,
        "mono_docs_a": mono_docs_a,
        "mono_docs_b": mono_docs_b,
    }


# --------------------------------------------------------------------------- #
# Compute callables (puremacro imports kept INSIDE each function).
# --------------------------------------------------------------------------- #
def _count_keywords_known() -> dict:
    from puremacro.narrative.indices._kernels import count_keywords

    text = _narrative_demo_corpus()["count_text"]
    return {
        "uncertainty": float(count_keywords(text, frozenset({"uncertainty"}))),
        "economic": float(count_keywords(text, frozenset({"economic"}))),
        "uncertain_wholeword": float(count_keywords(text, frozenset({"uncertain"}))),
        "both_terms": float(count_keywords(text, frozenset({"economic", "uncertainty"}))),
    }


def _count_keywords_negation() -> dict:
    from puremacro.narrative.indices._kernels import count_keywords

    text = _narrative_demo_corpus()["neg_text"]
    terms = frozenset({"uncertainty"})
    return {
        "plain": float(count_keywords(text, terms, negation=False)),
        "negation_aware": float(count_keywords(text, terms, negation=True)),
    }


def _epu_cooccurrence_raw_mean() -> dict:
    from puremacro.narrative.indices.epu import epu

    docs = _narrative_demo_corpus()["epu_docs"]
    lexicon = {
        "economy": frozenset({"economy"}),
        "policy": frozenset({"policy"}),
        "uncertainty": frozenset({"uncertainty"}),
    }
    ri = epu(docs, country="USA", lexicon=lexicon, normalize="raw", agg="mean")
    s = ri.series.dropna()
    return {"q1_value": float(s.iloc[0]), "n_quarters": float(s.shape[0])}


def _wui_length_normalized_value() -> dict:
    from puremacro.narrative.indices.wui import wui

    docs = _narrative_demo_corpus()["wui_one_doc"]
    ri = wui(docs, country="USA", lexicon=frozenset({"uncertainty"}), normalize="raw")
    s = ri.series.dropna()
    return {"hits_per_1000_words": float(s.iloc[0])}


def _wui_zscore_fixed_point() -> dict:
    from puremacro.narrative.indices.wui import wui

    docs = _narrative_demo_corpus()["wui_norm_docs"]
    ri = wui(docs, country="USA", lexicon=frozenset({"uncertainty"}), normalize="zscore")
    s = ri.series.dropna().to_numpy(dtype=float)
    return {"mean": float(s.mean()), "std_pop": float(s.std(ddof=0))}


def _wui_bbd100_fixed_point() -> dict:
    from puremacro.narrative.indices.wui import wui

    docs = _narrative_demo_corpus()["wui_norm_docs"]
    ri = wui(docs, country="USA", lexicon=frozenset({"uncertainty"}), normalize="bbd_100")
    s = ri.series.dropna().to_numpy(dtype=float)
    return {"mean": float(s.mean()), "std_pop": float(s.std(ddof=0))}


def _wui_monotone_in_frequency() -> dict:
    from puremacro.narrative.indices.wui import wui

    d = _narrative_demo_corpus()
    terms = frozenset({"uncertainty"})
    a = wui(d["mono_docs_a"], country="USA", lexicon=terms, normalize="raw").series.dropna().to_numpy(float)
    b = wui(d["mono_docs_b"], country="USA", lexicon=terms, normalize="raw").series.dropna().to_numpy(float)
    # QUALITATIVE lower-bound check: b - a >= 0 elementwise.
    return {"b_minus_a": b - a}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="narrative.count_keywords_known_counts",
        subsystem="narrative",
        title="count_keywords returns exact whole-word, case-insensitive hit counts",
        title_es="count_keywords devuelve recuentos exactos de coincidencias por palabra completa e insensibles a mayúsculas",
        mechanism=Mechanism.ANALYTICAL,
        compute=_count_keywords_known,
        reference=lambda: {
            "uncertainty": 4.0,        # 4 whole-word, case-insensitive matches
            "economic": 2.0,           # "Economic"/"economic"
            "uncertain_wholeword": 0.0,  # no bare "uncertain" token (word boundary)
            "both_terms": 6.0,         # 4 + 2 summed across the term set
        },
        tol=Tol.EXACT,
        citation="Whole-word, case-insensitive keyword count (Baker-Bloom-Davis 2016, QJE 131(4), text-matching step).",
        notes="Hand-counted from the crafted paragraph in _narrative_demo_corpus().",
    ),
    ValidationCase(
        id="narrative.count_keywords_negation",
        subsystem="narrative",
        title="Negation-aware count_keywords suppresses hits within the negation window",
        title_es="count_keywords con negación suprime coincidencias dentro de la ventana de negación",
        mechanism=Mechanism.ANALYTICAL,
        compute=_count_keywords_negation,
        reference=lambda: {"plain": 2.0, "negation_aware": 1.0},
        tol=Tol.EXACT,
        citation="Negation handling in lexicon sentiment scoring (Loughran-McDonald 2011, J. Finance 66(1), §I.B).",
        notes="First 'uncertainty' is preceded by 'no' within the 4-token window (suppressed); the second is clean.",
    ),
    ValidationCase(
        id="narrative.epu_cooccurrence_raw_mean",
        subsystem="narrative",
        title="EPU co-occurrence index equals the hand-computed quarterly mean indicator",
        title_es="El índice EPU por coocurrencia iguala la media trimestral del indicador calculada a mano",
        mechanism=Mechanism.ANALYTICAL,
        compute=_epu_cooccurrence_raw_mean,
        # 3 docs in 2020Q1: indicators [1, 0, 1] -> mean 2/3.
        reference=lambda: {"q1_value": 2.0 / 3.0, "n_quarters": 1.0},
        tol=Tol.TIGHT,
        citation="Three-group (Economy/Policy/Uncertainty) co-occurrence count (Baker, Bloom & Davis 2016, QJE 131(4):1593-1636).",
        notes="Raw normalisation; quarterly aggregator = mean of per-document {0,1} co-occurrence indicators.",
    ),
    ValidationCase(
        id="narrative.wui_length_normalized_value",
        subsystem="narrative",
        title="WUI score equals (hits / total words) × 1000 on a one-document quarter",
        title_es="La puntuación WUI iguala (coincidencias / palabras totales) × 1000 en un trimestre de un documento",
        mechanism=Mechanism.ANALYTICAL,
        compute=_wui_length_normalized_value,
        # 2 "uncertainty" hits in a 7-token document -> 2/7 * 1000.
        reference=lambda: {"hits_per_1000_words": 2.0 / 7.0 * 1000.0},
        tol=Tol.TIGHT,
        citation="Frequency-of-'uncertainty' per 1000 words (Ahir, Bloom & Furceri 2022, NBER WP 29763, World Uncertainty Index).",
        notes="Raw normalisation; single document so the quarterly mean equals the per-doc length-normalised score.",
    ),
    ValidationCase(
        id="narrative.wui_zscore_fixed_point",
        subsystem="narrative",
        title="z-score normalisation yields full-sample mean 0 and population std 1",
        title_es="La normalización z produce media muestral 0 y desviación típica poblacional 1",
        mechanism=Mechanism.INTERNAL,
        compute=_wui_zscore_fixed_point,
        reference=lambda: {"mean": 0.0, "std_pop": 1.0},
        tol=Tol.NUMERIC,
        citation="Standardisation identity: (x-μ)/σ has mean 0 and population variance 1 (Lütkepohl 2005, App. A).",
        notes="Full-series base period; std uses ddof=0 to match normalize_series.",
    ),
    ValidationCase(
        id="narrative.wui_bbd100_fixed_point",
        subsystem="narrative",
        title="BBD-100 normalisation yields full-sample mean 100 and population std 50",
        title_es="La normalización BBD-100 produce media muestral 100 y desviación típica poblacional 50",
        mechanism=Mechanism.INTERNAL,
        compute=_wui_bbd100_fixed_point,
        reference=lambda: {"mean": 100.0, "std_pop": 50.0},
        tol=Tol.NUMERIC,
        citation="BBD 100/50 rescaling 100 + 50·z (Baker, Bloom & Davis 2016, QJE 131(4), normalisation convention).",
        notes="Affine map of the z-score: mean 100, population std 50 by construction.",
    ),
    ValidationCase(
        id="narrative.wui_monotone_in_frequency",
        subsystem="narrative",
        title="Length-normalised index is monotone non-decreasing in keyword frequency",
        title_es="El índice normalizado por longitud es monótono no decreciente en la frecuencia de palabras clave",
        mechanism=Mechanism.INTERNAL,
        compute=_wui_monotone_in_frequency,
        # Corpus B dominates A in per-quarter hit count -> b - a >= 0 everywhere.
        reference=lambda: {"b_minus_a": np.zeros(2)},
        tol=Tol.QUALITATIVE,
        citation="Monotonicity of a frequency-count index in keyword incidence (Ahir, Bloom & Furceri 2022, NBER WP 29763).",
        notes="QUALITATIVE lower bound: each element of (index_B - index_A) must be >= 0.",
    ),
]
