"""Tests for the narrative quality pipeline components."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative.types import NarrativeEvent


def _make_event(**overrides) -> NarrativeEvent:
    base = dict(
        date=pd.Timestamp("2010-10-01"),   # UK budget month
        country="GBR",
        magnitude=1.0,
        magnitude_unit="pct_gdp",
        target="both",
        subtarget="expenditure",
        sign=-1,
        confidence=0.9,
        source_text="...",
        source_url="https://example.org",
        scoring_method="manual",
    )
    base.update(overrides)
    return NarrativeEvent(**base)


# ---------------------------------------------------------------------------
# SurpriseFilter
# ---------------------------------------------------------------------------


def test_surprise_filter_decays_budget_calendar_events():
    from puremacro.narrative.quality.surprise_filter import SurpriseFilter
    event = _make_event(
        date=pd.Timestamp("2010-10-15"),   # GBR budget month = 10
        subtarget="expenditure",
    )
    sf = SurpriseFilter(decay_factor=0.25)
    result = sf.apply([event])
    assert len(result) == 1
    assert result[0].confidence == pytest.approx(0.9 * 0.25)
    assert result[0].metadata.get("anticipation_flag") is True


def test_surprise_filter_spares_unanticipated_flag():
    from puremacro.narrative.quality.surprise_filter import SurpriseFilter
    event = _make_event(
        date=pd.Timestamp("2010-10-15"),
        subtarget="expenditure",
        metadata={"unanticipated": True},
    )
    sf = SurpriseFilter(decay_factor=0.25)
    result = sf.apply([event])
    assert result[0].confidence == pytest.approx(0.9)  # unchanged


def test_surprise_filter_decays_multi_quarter_profile_events():
    from puremacro.narrative.quality.surprise_filter import SurpriseFilter
    event = _make_event(
        date=pd.Timestamp("2011-04-01"),
        country="USA",
        implementation_profile=[
            (pd.Timestamp("2011-07-01"), 0.5),
            (pd.Timestamp("2011-10-01"), 0.5),
        ],
    )
    sf = SurpriseFilter(decay_factor=0.25)
    result = sf.apply([event])
    assert result[0].confidence == pytest.approx(0.9 * 0.25)
    assert result[0].metadata.get("anticipation_flag") is True


def test_surprise_filter_drops_below_confidence_floor():
    from puremacro.narrative.quality.surprise_filter import SurpriseFilter
    event = _make_event(
        date=pd.Timestamp("2010-10-15"),
        confidence=0.08,          # will decay to 0.02 < floor 0.1
        subtarget="expenditure",
    )
    sf = SurpriseFilter(decay_factor=0.25, confidence_floor=0.1)
    result = sf.apply([event])
    assert result == []


def test_surprise_filter_non_budget_event_unchanged():
    from puremacro.narrative.quality.surprise_filter import SurpriseFilter
    event = _make_event(
        date=pd.Timestamp("2010-03-01"),  # March != GBR budget month (Oct)
        subtarget="expenditure",
    )
    sf = SurpriseFilter(decay_factor=0.25)
    result = sf.apply([event])
    assert result[0].confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# MagnitudeHarmonizer
# ---------------------------------------------------------------------------


def test_harmonizer_pct_gdp_event_deflated():
    from puremacro.narrative.quality.magnitude_harmonizer import MagnitudeHarmonizer
    # Provide a tiny GDP deflator table: country, date, deflator (2015=1).
    gdp_deflator = pd.DataFrame({
        "country": ["GBR", "GBR"],
        "date":    [pd.Timestamp("2000-01-01"), pd.Timestamp("2015-01-01")],
        "deflator": [0.80, 1.00],
    }).set_index(["country", "date"])
    event = _make_event(
        date=pd.Timestamp("2000-01-01"),
        magnitude=2.0,
        magnitude_unit="pct_gdp",
    )
    mh = MagnitudeHarmonizer(gdp_deflator=gdp_deflator)
    result = mh.apply([event])
    assert len(result) == 1
    # real_2015 = nominal_pct_gdp / deflator_2000 = 2.0 / 0.80 = 2.5
    assert abs(result[0].magnitude - 2.5) < 1e-9
    assert result[0].magnitude_unit == "pct_gdp"


def test_harmonizer_stub_event_gets_fixed_magnitude():
    from puremacro.narrative.quality.magnitude_harmonizer import MagnitudeHarmonizer
    event = _make_event(magnitude=0.0, magnitude_unit="pct_gdp")
    mh = MagnitudeHarmonizer()
    result = mh.apply([event])
    assert result[0].magnitude == 0.05
    assert result[0].magnitude_unit == "stub_pct_gdp"


def test_harmonizer_z_score_left_unchanged_with_warning():
    from puremacro.narrative.quality.magnitude_harmonizer import MagnitudeHarmonizer
    import warnings
    event = _make_event(magnitude=1.5, magnitude_unit="z")
    mh = MagnitudeHarmonizer()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = mh.apply([event])
    assert result[0].magnitude == 1.5
    assert result[0].magnitude_unit == "z"
    assert any("z-score" in str(x.message) for x in w)


def test_harmonizer_nearest_quarter_fallback():
    from puremacro.narrative.quality.magnitude_harmonizer import MagnitudeHarmonizer
    # Deflator has 2000-01-01 but event is at 2000-04-01 — should use nearest
    gdp_deflator = pd.DataFrame({
        "country": ["GBR"],
        "date":    [pd.Timestamp("2000-01-01")],
        "deflator": [0.80],
    }).set_index(["country", "date"])
    event = _make_event(
        date=pd.Timestamp("2000-04-01"),
        magnitude=2.0,
        magnitude_unit="pct_gdp",
    )
    mh = MagnitudeHarmonizer(gdp_deflator=gdp_deflator)
    result = mh.apply([event])
    # 2000-04-01 not exact, falls back to nearest = 2000-01-01, deflator=0.80
    assert abs(result[0].magnitude - 2.5) < 1e-9


def test_harmonizer_missing_country_passes_through():
    from puremacro.narrative.quality.magnitude_harmonizer import MagnitudeHarmonizer
    # Deflator has GBR but event is DEU — should return unchanged
    gdp_deflator = pd.DataFrame({
        "country": ["GBR"],
        "date":    [pd.Timestamp("2000-01-01")],
        "deflator": [0.80],
    }).set_index(["country", "date"])
    event = _make_event(
        date=pd.Timestamp("2000-01-01"),
        country="DEU",
        magnitude=1.5,
        magnitude_unit="pct_gdp",
    )
    mh = MagnitudeHarmonizer(gdp_deflator=gdp_deflator)
    result = mh.apply([event])
    assert abs(result[0].magnitude - 1.5) < 1e-9


# ---------------------------------------------------------------------------
# TargetAuditor
# ---------------------------------------------------------------------------


class _MockBackend:
    """Fake LLM backend that classifies by keyword in source_text."""
    def classify_target(self, source_text: str):
        text = source_text.lower()
        if "infrastructure" in text or "capital" in text:
            return {"target": "investment", "confidence": 0.95, "reasoning": "capex keyword"}
        if "wages" in text or "transfers" in text:
            return {"target": "consumption", "confidence": 0.90, "reasoning": "wages keyword"}
        return {"target": "both", "confidence": 0.50, "reasoning": "ambiguous"}


def test_target_auditor_reclassifies_investment_event():
    from puremacro.narrative.quality.target_auditor import TargetAuditor
    event = _make_event(
        target="both",
        source_text="Investment in road infrastructure and capital works.",
    )
    auditor = TargetAuditor(backend=_MockBackend(), confidence_threshold=0.7)
    result, audit_df = auditor.apply([event])
    assert result[0].target == "investment"
    assert len(audit_df) == 1
    assert audit_df.iloc[0]["new_target"] == "investment"


def test_target_auditor_reclassifies_consumption_event():
    from puremacro.narrative.quality.target_auditor import TargetAuditor
    event = _make_event(
        target="both",
        source_text="Increase in public sector wages and transfers to households.",
    )
    auditor = TargetAuditor(backend=_MockBackend(), confidence_threshold=0.7)
    result, audit_df = auditor.apply([event])
    assert result[0].target == "consumption"


def test_target_auditor_leaves_ambiguous_as_both():
    from puremacro.narrative.quality.target_auditor import TargetAuditor
    event = _make_event(target="both", source_text="Fiscal adjustment measures.")
    auditor = TargetAuditor(backend=_MockBackend(), confidence_threshold=0.7)
    result, audit_df = auditor.apply([event])
    assert result[0].target == "both"
    # Ambiguous events should not appear in the reclassification audit.
    assert len(audit_df) == 0


def test_target_auditor_skips_already_classified():
    from puremacro.narrative.quality.target_auditor import TargetAuditor
    event = _make_event(target="investment", source_text="Capital expenditure.")
    auditor = TargetAuditor(backend=_MockBackend(), confidence_threshold=0.7)
    result, audit_df = auditor.apply([event])
    assert result[0].target == "investment"
    assert len(audit_df) == 0


def test_target_auditor_no_backend_returns_unchanged():
    from puremacro.narrative.quality.target_auditor import TargetAuditor
    event = _make_event(target="both", source_text="Some text.")
    auditor = TargetAuditor(backend=None)
    result, audit_df = auditor.apply([event])
    assert result[0].target == "both"
    assert audit_df.empty


def test_target_auditor_no_backend_audit_df_has_columns():
    from puremacro.narrative.quality.target_auditor import TargetAuditor
    event = _make_event(target="both", source_text="Some text.")
    auditor = TargetAuditor(backend=None)
    _, audit_df = auditor.apply([event])
    assert list(audit_df.columns) == [
        "country", "date", "old_target", "new_target", "llm_confidence", "reasoning"
    ]


def test_target_auditor_backend_exception_passes_through_with_warning():
    from puremacro.narrative.quality.target_auditor import TargetAuditor
    import warnings

    class _BrokenBackend:
        def classify_target(self, source_text: str):
            raise RuntimeError("API quota exceeded")

    event = _make_event(target="both", source_text="Some text.")
    auditor = TargetAuditor(backend=_BrokenBackend())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result, audit_df = auditor.apply([event])
    assert result[0].target == "both"
    assert len(audit_df) == 0
    assert any("TargetAuditor backend failed" in str(x.message) for x in w)


def test_target_auditor_reclassified_event_has_audit_metadata():
    from puremacro.narrative.quality.target_auditor import TargetAuditor
    event = _make_event(
        target="both",
        source_text="Investment in road infrastructure and capital works.",
    )
    auditor = TargetAuditor(backend=_MockBackend(), confidence_threshold=0.7)
    result, _ = auditor.apply([event])
    assert result[0].metadata.get("target_auditor_confidence") == pytest.approx(0.95)
    assert result[0].metadata.get("target_auditor_reasoning") == "capex keyword"


# ---------------------------------------------------------------------------
# Pipeline + HomogeneousFiscalPanel integration
# ---------------------------------------------------------------------------


def _make_panel_with_both_events():
    from puremacro.narrative import HomogeneousFiscalPanel, NarrativeEvent
    events = [
        NarrativeEvent(
            date=pd.Timestamp("2010-10-01"), country="GBR", magnitude=1.0,
            magnitude_unit="pct_gdp", target="both", subtarget="expenditure",
            sign=-1, confidence=0.9, source_text="Capital works and wages cut.",
            source_url="https://example.org", scoring_method="manual",
        ),
        NarrativeEvent(
            date=pd.Timestamp("2010-03-01"), country="USA", magnitude=2.0,
            magnitude_unit="pct_gdp", target="investment", subtarget="defense",
            sign=+1, confidence=0.8, source_text="Defense capital buildup.",
            source_url="https://example.org", scoring_method="manual",
        ),
    ]
    return HomogeneousFiscalPanel(events=events, countries=["GBR", "USA"])


def test_apply_quality_pipeline_all_off_is_noop():
    panel = _make_panel_with_both_events()
    from puremacro.narrative import HomogeneousFiscalPanel
    result = panel.apply_quality_pipeline(
        surprise_filter=False,
        harmonize_magnitudes=False,
        audit_targets=False,
        confidence_floor=0.0,
    )
    assert len(result.events) == len(panel.events)
    for orig, new in zip(panel.events, result.events):
        assert orig.magnitude == new.magnitude
        assert orig.target == new.target


def test_apply_quality_pipeline_returns_new_panel():
    panel = _make_panel_with_both_events()
    result = panel.apply_quality_pipeline(
        surprise_filter=True,
        harmonize_magnitudes=False,
        audit_targets=False,
    )
    assert result is not panel


def test_apply_quality_pipeline_metadata_records_counts():
    panel = _make_panel_with_both_events()
    result = panel.apply_quality_pipeline(
        surprise_filter=True,
        harmonize_magnitudes=False,
        audit_targets=False,
    )
    meta = result.metadata["quality_pipeline"]
    assert "n_raw" in meta
    assert "n_after_surprise_filter" in meta
    assert meta["n_raw"] == 2


def test_apply_quality_pipeline_confidence_floor_drops_low_confidence_events():
    panel = _make_panel_with_both_events()
    # Both events have confidence 0.9 and 0.8 — a floor of 0.85 should drop the USA event.
    result = panel.apply_quality_pipeline(
        surprise_filter=False,
        harmonize_magnitudes=False,
        audit_targets=False,
        confidence_floor=0.85,
    )
    assert result.metadata["quality_pipeline"]["n_dropped_low_confidence"] == 1
    assert len(result.events) == 1
    assert result.events[0].country == "GBR"


def test_apply_quality_pipeline_does_not_corrupt_preexisting_audit():
    """Quality pipeline must not concat audits with incompatible schemas."""
    import pandas as pd
    from puremacro.narrative import HomogeneousFiscalPanel
    # Panel with a dedup-style audit (different columns from target-auditor).
    dedup_audit = pd.DataFrame({
        "rep_country": ["GBR"],
        "rep_date":    [pd.Timestamp("2010-10-01")],
        "rep_method":  ["exact"],
    })
    panel = HomogeneousFiscalPanel(
        events=_make_panel_with_both_events().events,
        countries=["GBR", "USA"],
        audit=dedup_audit,
    )
    result = panel.apply_quality_pipeline(
        surprise_filter=False, harmonize_magnitudes=False, audit_targets=False,
        confidence_floor=0.0,
    )
    # With no target reclassifications, the old audit should be preserved or None.
    # It must NOT have been corrupted by concat with an incompatible empty DataFrame.
    if result.audit is not None:
        assert set(result.audit.columns) == {"rep_country", "rep_date", "rep_method"}
