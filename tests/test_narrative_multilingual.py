"""Test suite for multilingual scoring, schedule estimation, and policy action classification."""
from __future__ import annotations

import pandas as pd
import pytest

from puremacro.narrative import (
    MULTILINGUAL_LEXICONS,
    MultilingualScoreResult,
    NarrativeDocument,
    NarrativeEvent,
    PolicyActionClassifier,
    estimate_implementation_profile,
    infer_policy_stance,
    score_multilingual,
)


def test_multilingual_scoring_across_languages():
    """Verify macro lexicon scoring across EN, ES, PT, DE, FR, IT, JA, ZH."""
    test_cases = [
        ("en", "The government announced a major fiscal stimulus and infrastructure spending package.", 1),
        ("en", "The central bank decided to hike rates and implement monetary tightening to cool inflation.", -1),
        ("es", "El ministro anunció un estímulo fiscal y aumento de gasto en inversión pública.", 1),
        ("es", "El banco central decidió un alza de tasa de interés para combatir la inflación.", -1),
        ("pt", "O governo aprovou um estímulo fiscal com aumento de gastos e investimento em infraestrutura.", 1),
        ("pt", "O Banco Central do Brasil determinou a alta da selic e aperto monetário.", -1),
        ("de", "Die Bundesregierung beschloss ein neues Konjunkturpaket und Steuersenkung für Unternehmen.", 1),
        ("de", "Zur Haushaltskonsolidierung sind Ausgabenkürzung und Sparprogramm erforderlich.", -1),
        ("fr", "Le gouvernement lance un vaste plan de relance et hausse des dépenses d'investissement public.", 1),
        ("fr", "La Banque centrale poursuit la hausse des taux et le resserrement monétaire.", -1),
        ("it", "Il governo ha varato un piano di stimolo e taglio delle tasse sugli investimenti.", 1),
        ("it", "Misure di consolidamento fiscale e tagli alla spesa pubblica.", -1),
        ("ja", "政府は大規模な経済対策と財政出動、公共投資の拡大を決定した。", 1),
        ("ja", "日本銀行は金融引き締めと利上げを実施した。", -1),
        ("zh", "实施积极财政政策，加力提效，支持重大基础设施投资和专项债发行。", 1),
        ("zh", "央行决定加息并收紧流动性以防范金融风险。", -1),
    ]

    for lang, text, expected_sign in test_cases:
        res = score_multilingual(text, language=lang)
        assert isinstance(res, MultilingualScoreResult)
        assert res.language == lang
        assert res.token_count > 0

        sign, conf, details = infer_policy_stance(text, language=lang)
        assert sign == expected_sign, f"Failed for lang={lang}: got {sign}, expected {expected_sign}"
        assert 0.0 <= conf <= 1.0


def test_schedule_estimation():
    """Verify realization schedule estimation generates valid probability weights summing to 1.0."""
    # When announcement date is quarter-start
    dt_q = "2023-01-01"
    prof_infra = estimate_implementation_profile(dt_q, kind="fiscal", target="investment", subtarget="infra")
    assert len(prof_infra) == 5
    weights_infra = [w for _, w in prof_infra]
    assert abs(sum(weights_infra) - 1.0) < 1e-6
    assert prof_infra[0][0] == pd.Timestamp("2023-01-01")  # 2023Q1
    assert prof_infra[4][0] == pd.Timestamp("2024-01-01")  # 2024Q1

    # When announcement date is mid-quarter (e.g. March 15), Q0 lands on announcement date
    dt_mid = "2023-03-15"
    prof_mid = estimate_implementation_profile(dt_mid, kind="fiscal", target="investment", subtarget="infra")
    assert prof_mid[0][0] == pd.Timestamp("2023-03-15")

    # 2. Transfer payments (frontloaded)
    prof_transfers = estimate_implementation_profile(dt_q, kind="fiscal", target="consumption", subtarget="transfers")
    assert len(prof_transfers) == 2
    assert abs(sum(w for _, w in prof_transfers) - 1.0) < 1e-6
    assert prof_transfers[0][1] > prof_transfers[1][1]  # frontloaded

    # 3. Monetary rate change (immediate realization)
    prof_monetary = estimate_implementation_profile(dt_q, kind="monetary", target="policy_rate")
    assert len(prof_monetary) == 1
    assert prof_monetary[0][1] == 1.0


def test_policy_action_classifier():
    """Verify PolicyActionClassifier transforms text and documents into NarrativeEvents."""
    classifier = PolicyActionClassifier()

    doc = NarrativeDocument(
        doc_id="test_01",
        source="fed_decision",
        country="USA",
        date=pd.Timestamp("2020-03-15"),
        url="https://federalreserve.gov/test",
        title="FOMC Statement",
        text="The Federal Reserve announced a cut interest rate and emergency asset purchase program to support the economy.",
        language="en",
        policy_domain="monetary",
        doc_type="decision",
    )

    ev = classifier.classify_document(doc)
    assert ev is not None
    assert isinstance(ev, NarrativeEvent)
    assert ev.country == "usa"
    assert ev.kind == "monetary"
    assert ev.target in ("policy_rate", "asset_purchase")
    assert ev.sign == 1  # expansionary / rate cut
    assert ev.confidence > 0.4
    assert len(ev.implementation_profile) == 1  # monetary takes effect immediately

    # Fiscal infrastructure document in Spanish with magnitude
    doc_es = NarrativeDocument(
        doc_id="test_02",
        source="banxico",
        country="MEX",
        date=pd.Timestamp("2021-06-01"),
        url="https://shcp.gob.mx/test",
        title="Paquete Fiscal",
        text="La Secretaría de Hacienda presentó un plan de estímulo fiscal con inversión pública e infraestructura por un 2.5% del PIB.",
        language="es",
        policy_domain="fiscal",
        doc_type="report",
    )

    ev_es = classifier.classify_document(doc_es)
    assert ev_es is not None
    assert ev_es.country == "mex"
    assert ev_es.kind == "fiscal"
    assert ev_es.target == "investment"
    assert ev_es.sign == 1
    assert ev_es.magnitude == 2.5
    assert ev_es.magnitude_unit == "pct_gdp"
    assert len(ev_es.implementation_profile) == 5  # multi-quarter infrastructure
