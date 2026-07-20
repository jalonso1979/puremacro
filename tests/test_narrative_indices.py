"""Tests for puremacro.narrative.indices subpackage (Slice 2)."""
from __future__ import annotations


def test_indices_subpackage_imports_cleanly():
    """The bare subpackage skeleton must import without errors."""
    from puremacro.narrative.indices import _kernels, _lexicons
    assert hasattr(_kernels, "__all__")
    assert hasattr(_lexicons, "__all__")


# ---------------------------------------------------------------------------
# Lexicon structural tests
# ---------------------------------------------------------------------------
def test_lexicons_top_level_keys():
    from puremacro.narrative.indices._lexicons import LEXICONS
    assert set(LEXICONS) == {"epu", "mpu", "gpr", "tone", "wui", "lui", "ltui", "ltui_up", "ltui_down", "lwui", "lwui_wage"}


def test_epu_lexicon_has_three_groups_in_english():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["epu"]["en"]
    assert set(en) == {"economy", "policy", "uncertainty"}
    assert {"economic"} <= en["economy"]
    assert {"policy"} <= en["policy"]
    assert {"uncertain", "uncertainty"} <= en["uncertainty"]


def test_mpu_lexicon_english_has_monetary_terms():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["mpu"]["en"]
    assert "monetary" in en
    assert "policy" in en
    assert "uncertain" in en or "uncertainty" in en


def test_gpr_lexicon_english_has_geopolitical_terms():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["gpr"]["en"]
    assert "war" in en
    assert "terror" in en or "terrorism" in en
    assert "geopolitical" in en


def test_tone_lexicon_english_has_hawkish_dovish_groups():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["tone"]["en"]
    assert set(en) == {"hawkish", "dovish"}
    assert {"hawkish", "tighten", "tightening"} <= en["hawkish"]
    assert {"dovish", "ease", "easing"} <= en["dovish"]


def test_wui_lexicon_english_has_uncertainty_terms():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["wui"]["en"]
    assert "uncertainty" in en
    assert "uncertain" in en


def test_lui_lexicon_english_has_labor_terms():
    from puremacro.narrative.indices._lexicons import LEXICONS
    en = LEXICONS["lui"]["en"]
    phrases = en["phrases"]
    assert "layoff" in phrases or "layoffs" in phrases
    assert "hiring freeze" in phrases or "hiring-freeze" in phrases
    assert "wage compression" in phrases or "wage-compression" in phrases
    assert "labor shortage" in phrases or "labor-shortage" in phrases
    assert "unemployment" in phrases


# ---------------------------------------------------------------------------
# Kernel tests
# ---------------------------------------------------------------------------
import pandas as pd
import pytest


def _doc(date, text):
    """Synthetic 4-tuple SourceRecord."""
    return (pd.Timestamp(date), text, "https://test/" + str(date), {"language": "en"})


def test_count_keywords_basic():
    from puremacro.narrative.indices._kernels import count_keywords
    text = "Economic policy uncertainty rose this quarter."
    terms = frozenset({"uncertain", "uncertainty"})
    n = count_keywords(text, terms, language="en")
    # "uncertainty" matches once (single-token "uncertainty"); "uncertain"
    # is a prefix that should NOT separately match a single-token regex.
    assert n == 1


def test_count_keywords_multi_term_phrase():
    from puremacro.narrative.indices._kernels import count_keywords
    text = "The federal reserve raised the policy rate."
    terms = frozenset({"federal reserve", "policy rate"})
    n = count_keywords(text, terms, language="en")
    # Both two-word phrases match once each.
    assert n == 2


def test_count_keywords_substring_match_for_non_latin():
    """Japanese / Chinese tokenization is hard; substring match instead."""
    from puremacro.narrative.indices._kernels import count_keywords
    text = "経済政策不確実性"  # economy + policy + uncertainty (concatenated)
    terms = frozenset({"不確実性"})
    n = count_keywords(text, terms, language="ja")
    assert n == 1


def test_keyword_count_kernel_emits_per_doc_score():
    from puremacro.narrative.indices._kernels import keyword_count_kernel
    records = [
        _doc("2020-01-15", "uncertain uncertainty uncertainty rose"),
        _doc("2020-02-15", "no hits here"),
    ]
    terms = frozenset({"uncertain", "uncertainty"})
    out = list(keyword_count_kernel(records, terms=terms, language="en"))
    assert out[0][1] == 3   # uncertain + 2x uncertainty
    assert out[1][1] == 0


def test_cooccurrence_kernel_all_groups_present():
    from puremacro.narrative.indices._kernels import cooccurrence_kernel
    records = [
        _doc("2020-01-15", "economic policy uncertainty rose this quarter"),
    ]
    groups = [
        frozenset({"economic"}),
        frozenset({"policy"}),
        frozenset({"uncertainty", "uncertain"}),
    ]
    out = list(cooccurrence_kernel(records, term_groups=groups, language="en"))
    assert out[0][1] == 1.0


def test_cooccurrence_kernel_one_group_missing():
    from puremacro.narrative.indices._kernels import cooccurrence_kernel
    records = [
        _doc("2020-01-15", "economic policy went well"),  # no uncertainty term
    ]
    groups = [
        frozenset({"economic"}),
        frozenset({"policy"}),
        frozenset({"uncertainty", "uncertain"}),
    ]
    out = list(cooccurrence_kernel(records, term_groups=groups, language="en"))
    assert out[0][1] == 0.0


def test_tone_kernel_net_value():
    from puremacro.narrative.indices._kernels import tone_kernel
    records = [
        _doc("2022-03-15", "raised hike tightening hawkish ease"),  # 4 hawk, 1 dove
    ]
    out = list(tone_kernel(
        records,
        hawkish_terms=frozenset({"raised", "hike", "tightening", "hawkish"}),
        dovish_terms=frozenset({"ease", "easing", "cut"}),
        language="en",
    ))
    # Net: 4 hawk - 1 dove = +3, normalized by 5 token hits = 0.6
    assert out[0][1] == pytest.approx(0.6)


def test_tone_kernel_no_hits_returns_zero():
    from puremacro.narrative.indices._kernels import tone_kernel
    records = [_doc("2020-01-15", "no relevant words here")]
    out = list(tone_kernel(
        records,
        hawkish_terms=frozenset({"hike"}),
        dovish_terms=frozenset({"cut"}),
        language="en",
    ))
    assert out[0][1] == 0.0


# ---------------------------------------------------------------------------
# normalize_series tests
# ---------------------------------------------------------------------------
def test_normalize_raw_passes_through():
    from puremacro.narrative.indices._kernels import normalize_series
    s = pd.Series([100.0, 110.0, 95.0],
                  index=pd.date_range("2020-01-01", periods=3, freq="QS"))
    out = normalize_series(s, "raw")
    assert (out == s).all()


def test_normalize_zscore_has_zero_mean_unit_std():
    from puremacro.narrative.indices._kernels import normalize_series
    s = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0],
                  index=pd.date_range("2020-01-01", periods=5, freq="QS"))
    out = normalize_series(s, "zscore")
    assert out.mean() == pytest.approx(0.0, abs=1e-10)
    assert out.std(ddof=0) == pytest.approx(1.0, abs=1e-10)


def test_normalize_bbd_100_has_mean_100_std_50():
    from puremacro.narrative.indices._kernels import normalize_series
    s = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0],
                  index=pd.date_range("2020-01-01", periods=5, freq="QS"))
    out = normalize_series(s, "bbd_100")
    assert out.mean() == pytest.approx(100.0)
    assert out.std(ddof=0) == pytest.approx(50.0)


def test_normalize_bbd_100_with_base_period():
    """BBD's published series uses 1985-2009 as the base; normalization
    should target the BASE PERIOD's mean/std, not the full series."""
    from puremacro.narrative.indices._kernels import normalize_series
    idx = pd.date_range("2020-01-01", periods=8, freq="QS")
    # Base period 2020Q1-2020Q4 (rows 0..3) has mean 100 std ~3.535.
    # Post-base values are 200 — should become much higher than 100.
    s = pd.Series([95.0, 100.0, 105.0, 100.0, 200.0, 200.0, 200.0, 200.0],
                  index=idx)
    base = ("2020-01-01", "2020-12-31")
    out = normalize_series(s, "bbd_100", base_period=base)
    base_mask = (out.index >= "2020-01-01") & (out.index <= "2020-12-31")
    assert out[base_mask].mean() == pytest.approx(100.0, rel=0.1)
    # Post-base values should be far above 100.
    assert (out[~base_mask] > 500).all()


def test_normalize_invalid_kind_raises():
    from puremacro.narrative.indices._kernels import normalize_series
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="normalization"):
        normalize_series(s, "not_a_norm")


# ---------------------------------------------------------------------------
# epu()
# ---------------------------------------------------------------------------
def test_epu_returns_riskindex_with_correct_metadata():
    from puremacro.narrative.indices import epu
    records = [
        _doc("2020-01-15", "economic policy uncertainty rose"),
        _doc("2020-02-15", "economic policy uncertainty rose again"),
        _doc("2020-04-15", "no relevant content here"),
    ]
    ri = epu(records, country="USA", language="en", normalize="raw")
    assert ri.country == "USA"
    assert ri.method == "keyword_count"
    assert ri.normalization == "raw"
    # Q1: 2 hits, Q2: 0 — under "mean" agg the values are 1.0 and 0.0 (raw).
    assert ri.series.iloc[0] == pytest.approx(1.0)
    assert ri.series.iloc[1] == pytest.approx(0.0)


def test_epu_uses_default_english_lexicon_when_language_en():
    from puremacro.narrative.indices import epu
    records = [_doc("2020-01-15", "economic policy uncertainty rose"),
               _doc("2020-02-15", "economic policy uncertainty rose")]
    ri = epu(records, country="USA", language="en", normalize="raw")
    assert ri.series.iloc[0] == pytest.approx(1.0)


def test_epu_zero_when_one_group_missing():
    from puremacro.narrative.indices import epu
    records = [_doc("2020-01-15", "economic policy went well"),
               _doc("2020-02-15", "economic activity was strong")]
    ri = epu(records, country="USA", language="en", normalize="raw")
    assert ri.series.iloc[0] == pytest.approx(0.0)


def test_epu_bbd_100_normalization_applied():
    from puremacro.narrative.indices import epu
    records = []
    high_text = "economic policy uncertainty rose"
    low_text = "no hits"
    for q, text in enumerate([high_text, low_text] * 4):
        d = pd.Timestamp(f"2020-{q + 1:02d}-15")
        records.append(_doc(str(d.date()), text))
    ri = epu(records, country="USA", language="en", normalize="bbd_100")
    s = ri.series.dropna()
    assert s.mean() == pytest.approx(100.0)
    assert s.std(ddof=0) == pytest.approx(50.0)


def test_epu_custom_lexicon_overrides_default():
    from puremacro.narrative.indices import epu
    custom = {
        "economy":     frozenset({"widget"}),
        "policy":      frozenset({"sprocket"}),
        "uncertainty": frozenset({"flummox"}),
    }
    records = [_doc("2020-01-15", "the widget sprocket flummox happened")]
    ri = epu(records, country="USA", language="en",
             lexicon=custom, normalize="raw")
    assert ri.series.iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mpu()
# ---------------------------------------------------------------------------
def test_mpu_counts_monetary_uncertainty_terms():
    from puremacro.narrative.indices import mpu
    records = [
        _doc("2020-01-15",
             "monetary policy uncertainty around the federal reserve increased"),
        _doc("2020-02-15", "no relevant words"),
    ]
    ri = mpu(records, country="USA", language="en", normalize="raw")
    assert ri.series.iloc[0] > 0
    assert ri.country == "USA"


def test_mpu_zscore_normalization():
    from puremacro.narrative.indices import mpu
    records = []
    high = "monetary policy uncertainty federal reserve interest rate"
    low = "weather"
    for q in range(8):
        d = pd.Timestamp("2020-01-01") + pd.DateOffset(months=q)
        records.append(_doc(str(d.date()), high if q % 2 == 0 else low))
    ri = mpu(records, country="USA", language="en", normalize="zscore")
    assert ri.series.dropna().mean() == pytest.approx(0.0, abs=1e-9)


def test_mpu_metadata_records_index_label():
    from puremacro.narrative.indices import mpu
    records = [_doc("2020-01-15", "monetary policy")]
    ri = mpu(records, country="USA", language="en", normalize="raw")
    assert ri.metadata.get("index") == "mpu"


# ---------------------------------------------------------------------------
# gpr()
# ---------------------------------------------------------------------------
def test_gpr_counts_geopolitical_terms():
    from puremacro.narrative.indices import gpr
    records = [
        _doc("2020-01-15", "war terrorism geopolitical sanctions invasion"),
        _doc("2020-02-15", "ordinary peaceful day"),
    ]
    ri = gpr(records, country="USA", language="en", normalize="raw")
    assert ri.series.iloc[0] > 0
    assert ri.country == "USA"


def test_gpr_metadata_records_index_label():
    from puremacro.narrative.indices import gpr
    records = [_doc("2020-01-15", "war broke out")]
    ri = gpr(records, country="USA", language="en", normalize="raw")
    assert ri.metadata.get("index") == "gpr"


# ---------------------------------------------------------------------------
# tone()
# ---------------------------------------------------------------------------
def test_tone_apel_blix_grimaldi_hawkish_corpus_positive():
    from puremacro.narrative.indices import tone
    records = [
        _doc("2020-01-15", "raised hike tightening hawkish"),
        _doc("2020-02-15", "raise hawkish withdraw"),
    ]
    ri = tone(records, country="USA", language="en",
              method="apel_blix_grimaldi", normalize="raw")
    # All hawkish hits, no dovish — net = +1 per doc, mean +1
    assert ri.series.iloc[0] == pytest.approx(1.0)


def test_tone_apel_blix_grimaldi_dovish_corpus_negative():
    from puremacro.narrative.indices import tone
    records = [_doc("2020-01-15", "cut ease dovish accommodative")]
    ri = tone(records, country="USA", language="en",
              method="apel_blix_grimaldi", normalize="raw")
    assert ri.series.iloc[0] == pytest.approx(-1.0)


def test_tone_neutral_empty_text_yields_zero():
    from puremacro.narrative.indices import tone
    records = [_doc("2020-01-15", "weather report")]
    ri = tone(records, country="USA", language="en",
              method="apel_blix_grimaldi", normalize="raw")
    assert ri.series.iloc[0] == pytest.approx(0.0)


def test_tone_method_picault_renault_falls_back_to_count_for_now():
    """Picault-Renault uses paragraph-level multinomial classification.
    For Slice 2 we ship a count-based approximation; the call must not
    raise and the metadata records the method requested."""
    from puremacro.narrative.indices import tone
    records = [_doc("2020-01-15", "raised hike")]
    ri = tone(records, country="USA", language="en",
              method="picault_renault", normalize="raw")
    assert ri.metadata["method_requested"] == "picault_renault"


def test_tone_unknown_method_raises():
    from puremacro.narrative.indices import tone
    records = [_doc("2020-01-15", "x")]
    with pytest.raises(ValueError, match="method"):
        tone(records, country="USA", language="en", method="not_a_method")


# ---------------------------------------------------------------------------
# wui()
# ---------------------------------------------------------------------------
def test_wui_counts_uncertainty_terms_only():
    from puremacro.narrative.indices import wui
    records = [
        _doc("2020-01-15", "uncertain uncertainty unpredictable ambiguity"),
        _doc("2020-02-15", "the weather was nice"),
    ]
    ri = wui(records, country="MEX", language="en", normalize="raw")
    assert ri.series.iloc[0] > 0
    assert ri.country == "MEX"


def test_wui_metadata_records_index_label():
    from puremacro.narrative.indices import wui
    records = [_doc("2020-01-15", "uncertainty")]
    ri = wui(records, country="MEX", language="en", normalize="raw")
    assert ri.metadata.get("index") == "wui"


# ---------------------------------------------------------------------------
# lui()
# ---------------------------------------------------------------------------
def test_lui_counts_labor_uncertainty_terms():
    from puremacro.narrative.indices import lui
    records = [
        _doc("2020-01-15",
             "layoffs hiring freeze wage compression labor shortage rising unemployment"),
        _doc("2020-02-15", "ordinary day"),
    ]
    ri = lui(records, country="USA", language="en", normalize="raw")
    assert ri.series.iloc[0] > 0
    assert ri.country == "USA"


def test_lui_metadata_records_index_label():
    from puremacro.narrative.indices import lui
    records = [_doc("2020-01-15", "layoff")]
    ri = lui(records, country="USA", language="en", normalize="raw")
    assert ri.metadata.get("index") == "lui"


def test_lui_distinguishes_high_low_periods():
    """A clearly labor-stressed corpus should rank above a quiet one."""
    from puremacro.narrative.indices import lui
    records = []
    high_text = "layoffs hiring freeze rising unemployment wage compression"
    low_text = "ordinary"
    for q in range(8):
        d = pd.Timestamp("2020-01-01") + pd.DateOffset(months=q * 3)
        records.append(_doc(str(d.date()), high_text if q < 4 else low_text))
    ri = lui(records, country="USA", language="en", normalize="zscore")
    s = ri.series.dropna()
    # First half (high) z-scores should average above the second half (low).
    assert s.iloc[:4].mean() > s.iloc[4:].mean()


# ---------------------------------------------------------------------------
# Multilingual lexicons
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lang", ["es", "pt", "de", "fr", "it", "ja", "zh"])
def test_epu_lexicon_present_for_all_supported_languages(lang):
    from puremacro.narrative.indices._lexicons import LEXICONS
    assert lang in LEXICONS["epu"], f"EPU missing language {lang}"
    groups = LEXICONS["epu"][lang]
    assert set(groups) == {"economy", "policy", "uncertainty"}, (
        f"EPU/{lang} groups: {sorted(groups)}"
    )
    assert all(len(groups[g]) >= 1 for g in groups), (
        f"EPU/{lang} has empty group: {groups}"
    )


@pytest.mark.parametrize("lang", ["es", "pt", "de", "fr", "it", "ja", "zh"])
def test_other_indices_have_each_supported_language(lang):
    from puremacro.narrative.indices._lexicons import LEXICONS
    for index in ("mpu", "gpr", "wui", "lui"):
        assert lang in LEXICONS[index], (
            f"{index} missing language {lang}"
        )
        if index == "lui":
            # Slice 6a: LUI is a dict of three frozensets per language.
            sub = LEXICONS["lui"][lang]
            assert "labor_domain" in sub and len(sub["labor_domain"]) >= 1
            assert "uncertainty_tone" in sub and len(sub["uncertainty_tone"]) >= 1
            assert "phrases" in sub and len(sub["phrases"]) >= 1
        else:
            terms = LEXICONS[index][lang]
            assert len(terms) >= 1, f"{index}/{lang} is empty"


def test_lui_english_has_substantive_coverage():
    """LUI/EN phrases lexicon expanded in Slice 5: ≥ 100 terms across 6 conceptual groups."""
    from puremacro.narrative.indices._lexicons import LEXICONS
    lui_en_phrases = LEXICONS["lui"]["en"]["phrases"]
    assert len(lui_en_phrases) >= 100, f"LUI/EN has only {len(lui_en_phrases)} terms"
    # Spot-check coverage of each conceptual group.
    assert any("layoff" in t for t in lui_en_phrases)
    assert any("hiring freeze" in t for t in lui_en_phrases)
    assert any("wage" in t for t in lui_en_phrases)
    assert any("shortage" in t for t in lui_en_phrases)
    assert any("participation" in t for t in lui_en_phrases)
    assert any("unemployment" in t for t in lui_en_phrases)


@pytest.mark.parametrize("lang", ["es", "pt", "de", "fr", "it", "ja", "zh"])
def test_tone_lexicon_present_for_all_supported_languages(lang):
    """Tone lexicon ships for all 8 languages in Slice 3 (was Latin-only in Slice 2)."""
    from puremacro.narrative.indices._lexicons import LEXICONS
    assert lang in LEXICONS["tone"]
    groups = LEXICONS["tone"][lang]
    assert {"hawkish", "dovish"} <= set(groups)
    assert len(groups["hawkish"]) >= 1
    assert len(groups["dovish"]) >= 1


def test_epu_works_with_spanish_corpus():
    """End-to-end: the spanish lexicon should produce non-trivial scores
    on a recognisable Spanish EPU phrase."""
    from puremacro.narrative.indices import epu
    records = [
        _doc("2020-01-15",
             "incertidumbre sobre la política económica del banco central"),
        _doc("2020-02-15", "tiempo agradable hoy"),
    ]
    ri = epu(records, country="MEX", language="es", normalize="raw")
    assert ri.series.iloc[0] > 0, (
        f"Spanish EPU should detect 'incertidumbre/política/económica'; got {ri.series.tolist()}"
    )


def test_lui_works_with_spanish_corpus():
    from puremacro.narrative.indices import lui
    records = [
        _doc("2020-01-15", "despidos congelación de contrataciones desempleo"),
    ]
    ri = lui(records, country="MEX", language="es", normalize="raw")
    assert ri.series.iloc[0] > 0


@pytest.mark.parametrize("lang,min_count", [
    ("es", 100),
    ("pt", 100),
    ("de", 100),
    ("fr", 100),
    ("it", 100),
])
def test_lui_latin_lexicon_substantive_coverage(lang, min_count):
    from puremacro.narrative.indices._lexicons import LEXICONS
    terms = LEXICONS["lui"][lang]["phrases"]
    assert len(terms) >= min_count, (
        f"LUI/{lang} has only {len(terms)} terms; expected ≥ {min_count}"
    )


@pytest.mark.parametrize("lang,min_count", [
    ("ja", 60),
    ("zh", 60),
])
def test_lui_cjk_lexicon_substantive_coverage(lang, min_count):
    """JA/ZH labor lexicons are denser per concept than Latin scripts;
    target ≥ 60 terms per language."""
    from puremacro.narrative.indices._lexicons import LEXICONS
    terms = LEXICONS["lui"][lang]["phrases"]
    assert len(terms) >= min_count, (
        f"LUI/{lang} has only {len(terms)} terms; expected ≥ {min_count}"
    )


@pytest.mark.parametrize("lang,min_count", [
    ("en", 35),
    ("es", 30), ("pt", 30), ("de", 30), ("fr", 30), ("it", 30),
    ("ja", 25), ("zh", 25),
])
def test_labor_domain_lexicon_substantive_coverage(lang, min_count):
    """Slice 6a labor-domain lexicons must be broad enough for sentence
    co-occurrence to fire. Per-language thresholds reflect concept density
    in each script."""
    from puremacro.narrative.indices._lexicons import (
        _LABOR_DOMAIN_EN, _LABOR_DOMAIN_ES, _LABOR_DOMAIN_PT,
        _LABOR_DOMAIN_DE, _LABOR_DOMAIN_FR, _LABOR_DOMAIN_IT,
        _LABOR_DOMAIN_JA, _LABOR_DOMAIN_ZH,
    )
    name_to_lex = {
        "en": _LABOR_DOMAIN_EN, "es": _LABOR_DOMAIN_ES,
        "pt": _LABOR_DOMAIN_PT, "de": _LABOR_DOMAIN_DE,
        "fr": _LABOR_DOMAIN_FR, "it": _LABOR_DOMAIN_IT,
        "ja": _LABOR_DOMAIN_JA, "zh": _LABOR_DOMAIN_ZH,
    }
    terms = name_to_lex[lang]
    assert len(terms) >= min_count, (
        f"_LABOR_DOMAIN_{lang.upper()} has {len(terms)} terms; "
        f"need ≥ {min_count}"
    )


@pytest.mark.parametrize("lang,min_count", [
    ("en", 35),  # EN frozenset has ~50 - duplicates
    ("es", 25), ("pt", 25), ("de", 25), ("fr", 25), ("it", 25),
    ("ja", 15), ("zh", 15),
])
def test_uncertainty_tone_lexicon_substantive_coverage(lang, min_count):
    from puremacro.narrative.indices._lexicons import (
        _UNCERTAINTY_TONE_EN, _UNCERTAINTY_TONE_ES, _UNCERTAINTY_TONE_PT,
        _UNCERTAINTY_TONE_DE, _UNCERTAINTY_TONE_FR, _UNCERTAINTY_TONE_IT,
        _UNCERTAINTY_TONE_JA, _UNCERTAINTY_TONE_ZH,
    )
    name_to_lex = {
        "en": _UNCERTAINTY_TONE_EN, "es": _UNCERTAINTY_TONE_ES,
        "pt": _UNCERTAINTY_TONE_PT, "de": _UNCERTAINTY_TONE_DE,
        "fr": _UNCERTAINTY_TONE_FR, "it": _UNCERTAINTY_TONE_IT,
        "ja": _UNCERTAINTY_TONE_JA, "zh": _UNCERTAINTY_TONE_ZH,
    }
    terms = name_to_lex[lang]
    assert len(terms) >= min_count, (
        f"_UNCERTAINTY_TONE_{lang.upper()} has {len(terms)} terms; "
        f"need ≥ {min_count}"
    )


def test_wui_is_length_normalized():
    """Slice 6a: WUI now returns hits per 1000 words, not raw counts."""
    import pandas as pd
    from puremacro.narrative.indices import wui
    text_short = "uncertainty " * 5 + "the " * 95     # 100 words, 5 hits
    text_long = "uncertainty " * 5 + "the " * 195    # 200 words, 5 hits
    records_s = [(pd.Timestamp("2024-01-01"), text_short, "u", {})]
    records_l = [(pd.Timestamp("2024-04-01"), text_long, "u", {})]
    s_short = wui(records_s, country="USA", language="en", normalize="raw")
    s_long = wui(records_l, country="USA", language="en", normalize="raw")
    val_short = float(s_short.series.dropna().iloc[0])
    val_long = float(s_long.series.dropna().iloc[0])
    # Length-norm: text_short has higher density → higher score.
    assert val_short > val_long
    # Specifically: 50 vs 25 (per 1000 words).
    assert abs(val_short - 50.0) < 5.0
    assert abs(val_long - 25.0) < 5.0
