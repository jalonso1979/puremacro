"""Slice 1 — `with_quality=` rollout across canonical narrative indices."""
from __future__ import annotations

import importlib
import inspect

import pandas as pd
import pytest


# 4-tuple records: (date, text, source_url, metadata).
_RECS = [
    (pd.Timestamp("2020-01-15"), "policy uncertainty about economic outlook", "u", {}),
    (pd.Timestamp("2020-02-20"), "fiscal policy and tax reform uncertain", "u", {}),
    (pd.Timestamp("2020-05-10"), "monetary policy stable, economy improving", "u", {}),
]


def test_index_to_quarterly_with_quality_true_attaches_sparsity_report():
    from puremacro.narrative.aggregate import index_to_quarterly
    from puremacro.narrative.types import SignalQualityReport

    def _kernel(records):
        # Trivial kernel: 1.0 per doc.
        return [(r[0], 1.0) for r in records]

    ri = index_to_quarterly(
        _RECS, kernel=_kernel,
        country="USA", language="en",
        name="t", method="keyword_count", corpus="x",
        normalization="zscore",
        with_quality=True,
    )
    assert ri.quality is not None
    assert isinstance(ri.quality, SignalQualityReport)
    assert int(ri.quality.n_docs_per_period.loc[pd.Period("2020Q1", "Q")]) == 2
    assert int(ri.quality.n_docs_per_period.loc[pd.Period("2020Q2", "Q")]) == 1


def test_index_to_quarterly_with_quality_false_keeps_quality_none():
    from puremacro.narrative.aggregate import index_to_quarterly

    def _kernel(records):
        return [(r[0], 1.0) for r in records]

    ri = index_to_quarterly(
        _RECS, kernel=_kernel,
        country="USA", language="en",
        name="t", method="keyword_count", corpus="x",
        normalization="zscore",
        # with_quality defaults to False
    )
    assert ri.quality is None


# 14 canonical text-based index functions whose `with_quality=` plumbing
# is verified directly. The wrapper indices (bbui, bluesky_ui, erpui,
# sotuui, cboui, eurlex_ui, ep_ui) are tested in Task 6 alongside the
# coverage assertion.
_DIRECT_INDICES = [
    "puremacro.narrative.indices.epu:epu",
    "puremacro.narrative.indices.mpu:mpu",
    "puremacro.narrative.indices.gpr:gpr",
    "puremacro.narrative.indices.tone:tone",
    "puremacro.narrative.indices.wui:wui",
    "puremacro.narrative.indices.lui:lui",
    "puremacro.narrative.indices.ltui:ltui",
    "puremacro.narrative.indices.ltui:ltui_up",
    "puremacro.narrative.indices.ltui:ltui_down",
    "puremacro.narrative.indices.lwui:lwui",
    "puremacro.narrative.indices.lwui:lwui_wage",
]


@pytest.mark.parametrize("dotted", _DIRECT_INDICES)
def test_direct_index_accepts_with_quality_kwarg(dotted):
    mod_path, fn_name = dotted.split(":")
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name)
    sig = inspect.signature(fn)
    assert "with_quality" in sig.parameters, (
        f"{dotted}: missing `with_quality=` kwarg (Slice 1 contract)"
    )
    param = sig.parameters["with_quality"]
    assert param.default is False, (
        f"{dotted}: with_quality default must be False; got {param.default!r}"
    )


_WRAPPER_INDICES = [
    "puremacro.narrative.indices.beige_book:bbui",
    "puremacro.narrative.indices.bluesky:bluesky_ui",
    "puremacro.narrative.indices.us_executive:erpui",
    "puremacro.narrative.indices.us_executive:sotuui",
    "puremacro.narrative.indices.us_executive:cboui",
    "puremacro.narrative.indices.eu_legislative:eurlex_ui",
    "puremacro.narrative.indices.eu_legislative:ep_ui",
]


@pytest.mark.parametrize("dotted", _WRAPPER_INDICES)
def test_wrapper_index_accepts_with_quality_kwarg(dotted):
    mod_path, fn_name = dotted.split(":")
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, fn_name)
    sig = inspect.signature(fn)
    assert "with_quality" in sig.parameters, (
        f"{dotted}: missing `with_quality=` kwarg (Slice 1 contract)"
    )
    assert sig.parameters["with_quality"].default is False


def test_every_public_index_in__all__has_with_quality():
    """Cross-check: every name in `narrative.indices.__all__` that is a
    single-index constructor (i.e. takes records and returns a RiskIndex
    or DataFrame) accepts `with_quality=`. The excluded names are the
    kernel exports and `consensus_disagreement` — they're tracked here
    explicitly so adding a new index without `with_quality=` is caught."""
    import puremacro.narrative.indices as I

    EXCLUDED = {
        # Cross-source derived (no records argument).
        "consensus_disagreement", "CROSS_SOURCE_GROUPS",
        # Kernel exports (not indices).
        "embedding_similarity_kernel", "build_seed_prototype",
        "make_sentence_transformer_embedder",
        "mnl_kernel", "canonicalize_weights",
        "llm_prob_kernel", "LLMProvider", "MockProvider", "AnthropicProvider",
        # Local-LLM providers/factories (not indices).
        "LocalProvider", "OllamaProvider", "get_default_provider",
        # Lexicons constant (not a callable).
        "LEXICONS",
        # Fed-district crosswalk metadata + lookup helpers (not indices;
        # they support bbui(level='district')).
        "FedDistrict", "FED_DISTRICTS", "DISTRICT_NAMES", "DISTRICT_NUMBER",
        "SPLIT_STATES", "STATE_TO_DISTRICT", "STATE_TO_DISTRICTS_FULL",
        "TERRITORY_TO_DISTRICT", "district_crosswalk", "district_states",
        "state_district",
    }
    public_index_names = [n for n in I.__all__ if n not in EXCLUDED]
    missing = []
    for name in public_index_names:
        fn = getattr(I, name)
        if "with_quality" not in inspect.signature(fn).parameters:
            missing.append(name)
    assert not missing, (
        "Slice-1 contract violation: every canonical index in "
        "`narrative.indices.__all__` must accept `with_quality=False`. "
        f"Missing: {missing}"
    )


def test_lui_end_to_end_attaches_quality():
    """One end-to-end check: lui with `with_quality=True` produces a
    RiskIndex whose .quality.n_docs_per_period reflects the input."""
    from puremacro.narrative.indices import lui

    ri = lui(
        _RECS,
        country="USA",
        language="en",
        with_quality=True,
    )
    assert ri.quality is not None
    # The 3 fixture records span 2020Q1 (2) and 2020Q2 (1).
    assert int(ri.quality.n_docs_per_period.loc[pd.Period("2020Q1", "Q")]) == 2
    assert int(ri.quality.n_docs_per_period.loc[pd.Period("2020Q2", "Q")]) == 1


def test_bbui_district_with_quality_warns_and_does_not_attach():
    """bbui's district branch returns a DataFrame and cannot carry per-period
    quality. with_quality=True must emit a warning rather than silently no-op."""
    from puremacro.narrative.indices import bbui
    import warnings

    # Minimal 6-tuple records that bbui's _ensure_dataframe accepts.
    # bbui records are (date, district, section, text, source_url, metadata).
    recs = [
        (pd.Timestamp("2020-01-15"), "Boston", "overall", "policy uncertainty", "u", {}),
        (pd.Timestamp("2020-02-10"), "New York", "overall", "uncertain outlook", "u", {}),
        (pd.Timestamp("2020-05-10"), "Boston", "overall", "monetary policy", "u", {}),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = bbui(recs, level="district", with_quality=True)
        # district branch returns a DataFrame, not a RiskIndex.
        assert isinstance(result, pd.DataFrame)
        # Exactly one UserWarning emitted.
        ucw = [w for w in caught if issubclass(w.category, UserWarning)
               and "bbui(level='district')" in str(w.message)]
        assert len(ucw) == 1, f"expected one bbui warning, got {[str(w.message) for w in caught]}"
