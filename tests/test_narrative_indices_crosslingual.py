"""Slice 3: cross-lingual validation smokes.

Network-marked tests that build the same EPU-style index from the ECB's
English vs Spanish press feeds and check the resulting quarterly series
correlate ρ ≥ 0.7 on the overlapping window.

Skip when feeds are empty (per the project network-tests-skip-on-empty
convention).
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.mark.network
def test_ecb_epu_en_vs_es_correlation():
    """ECB press feed in en vs es should yield highly correlated EPU
    quarterly series. Skip on empty feeds."""
    from puremacro.narrative.indices import epu
    from puremacro.narrative.sources import iter_ecb_decision

    en_records = list(iter_ecb_decision(language="en"))
    es_records = list(iter_ecb_decision(language="es"))

    if not en_records or not es_records:
        pytest.skip("ECB feed returned empty for one of the languages.")

    ri_en = epu(en_records, country="EA20", language="en", normalize="raw")
    ri_es = epu(es_records, country="EA20", language="es", normalize="raw")

    s_en = ri_en.series.dropna()
    s_es = ri_es.series.dropna()
    common = s_en.index.intersection(s_es.index)
    if len(common) < 4:
        pytest.skip(f"Insufficient overlap: only {len(common)} common quarters.")

    rho = float(s_en.loc[common].corr(s_es.loc[common]))
    if pd.isna(rho):
        pytest.skip("Correlation is NaN (constant series); cross-lingual signal too weak.")

    assert rho >= 0.7, (
        f"EN-vs-ES EPU on ECB press should correlate ρ ≥ 0.7; got {rho:.3f} "
        f"on {len(common)} common quarters."
    )


@pytest.mark.network
def test_ecb_lui_en_vs_es_correlation():
    """LUI lexicon coverage parity check across en / es ECB press."""
    from puremacro.narrative.indices import lui
    from puremacro.narrative.sources import iter_ecb_decision

    en_records = list(iter_ecb_decision(language="en"))
    es_records = list(iter_ecb_decision(language="es"))

    if not en_records or not es_records:
        pytest.skip("ECB feed returned empty for one of the languages.")

    ri_en = lui(en_records, country="EA20", language="en", normalize="raw")
    ri_es = lui(es_records, country="EA20", language="es", normalize="raw")

    s_en = ri_en.series.dropna()
    s_es = ri_es.series.dropna()
    common = s_en.index.intersection(s_es.index)
    if len(common) < 4:
        pytest.skip(f"Insufficient overlap: only {len(common)} common quarters.")

    # Looser threshold for LUI: labor language is rarer in CB text than
    # uncertainty language, so cross-lingual signal is weaker.
    rho = float(s_en.loc[common].corr(s_es.loc[common]))
    if pd.isna(rho):
        pytest.skip("Correlation is NaN.")
    assert rho >= 0.4, (
        f"EN-vs-ES LUI on ECB press should correlate ρ ≥ 0.4; got {rho:.3f}."
    )
