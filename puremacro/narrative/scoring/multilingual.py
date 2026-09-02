"""Multilingual macroeconomic policy lexicons and stance scoring.

Supports English, Spanish, Portuguese, German, French, Italian, Japanese, and Chinese
for classifying central bank communications, fiscal announcements, and macro uncertainty.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Multilingual Macro Lexicons
# ─────────────────────────────────────────────────────────────────────────────

MULTILINGUAL_LEXICONS: dict[str, dict[str, list[str]]] = {
    # ── English ──
    "en": {
        "fiscal_expansion": [
            "fiscal stimulus", "tax cut", "tax relief", "infrastructure spending",
            "infrastructure investment", "public investment", "spending increase",
            "spending package", "investment package", "stimulus package",
            "supplementary budget", "cash transfers", "subsidies", "transfer payments",
            "fiscal expansion", "boost spending", "emergency spending", "capital expenditure",
        ],
        "fiscal_contraction": [
            "fiscal austerity", "fiscal consolidation", "tax hike", "tax increase",
            "spending cuts", "expenditure reduction", "budget cut", "deficit reduction",
            "austerity measures", "spending ceiling", "revenue enhancement", "spending reduction",
        ],
        "monetary_easing": [
            "cut interest rate", "rate cut", "lower policy rate", "quantitative easing",
            "asset purchase", "monetary stimulus", "accommodative stance", "dovish",
            "lower reserve requirement", "liquidity injection", "rate reduction",
        ],
        "monetary_tightening": [
            "raise interest rate", "rate hike", "increase policy rate", "quantitative tightening",
            "monetary tightening", "restrictive stance", "hawkish", "tapering",
            "hike rates", "cool inflation", "tighter financial conditions",
        ],
        "uncertainty": [
            "uncertainty", "uncertain", "unclear", "volatility", "risk", "instability",
            "unpredictable", "turbulence", "vulnerability", "downside risk",
        ],
    },

    # ── Spanish ──
    "es": {
        "fiscal_expansion": [
            "estímulo fiscal", "rebaja tributaria", "reducción de impuestos", "gasto en infraestructura",
            "inversión pública", "aumento de gasto", "presupuesto complementario",
            "transferencias directas", "subsidios", "expansión fiscal", "ayudas sociales",
        ],
        "fiscal_contraction": [
            "austeridad fiscal", "consolidación fiscal", "alza de impuestos", "reforma tributaria",
            "recorte de gasto", "reducción del déficit", "ajuste presupuestario", "regla fiscal",
            "contención del gasto",
        ],
        "monetary_easing": [
            "recorte de tasa", "baja de tasa", "reducción de tasa", "flexibilización cuantitativa",
            "estímulo monetario", "postura acomodaticia", "inyección de liquidez", "tasa de interés a la baja",
        ],
        "monetary_tightening": [
            "alza de tasa", "aumento de tasa", "subida de tasa", "endurecimiento monetario",
            "postura restrictiva", "política monetaria restrictiva", "retirar estímulo",
        ],
        "uncertainty": [
            "incertidumbre", "incierto", "volatilidad", "riesgo", "inestabilidad",
            "turbulencia", "vulnerabilidad", "riesgos a la baja",
        ],
    },

    # ── Portuguese (Brazil / BCB) ──
    "pt": {
        "fiscal_expansion": [
            "estímulo fiscal", "corte de impostos", "isenção fiscal", "investimento em infraestrutura",
            "investimento público", "aumento de gastos", "crédito extraordinário",
            "transferência de renda", "subsídios", "expansão fiscal", "auxílio emergencial",
        ],
        "fiscal_contraction": [
            "austeridade fiscal", "consolidação fiscal", "aumento de impostos", "reforma tributária",
            "corte de gastos", "redução do déficit", "ajuste fiscal", "teto de gastos",
            "contingenciamento", "meta fiscal",
        ],
        "monetary_easing": [
            "corte da taxa selic", "redução de juros", "afrouxamento monetário", "estímulo monetário",
            "postura acomodatícia", "injeção de liquidez", "redução da taxa básica",
        ],
        "monetary_tightening": [
            "alta da selic", "aumento de juros", "elevação da taxa básica", "aperto monetário",
            "postura restritiva", "combate à inflação", "política monetária contracionista",
        ],
        "uncertainty": [
            "incerteza", "incerto", "volatilidade", "risco", "instabilidade",
            "turbulência", "vulnerabilidade", "risco fiscal",
        ],
    },

    # ── German (Germany / BMF / Bundesbank) ──
    "de": {
        "fiscal_expansion": [
            "konjunkturpaket", "steuersenkung", "steuerentlastung", "infrastrukturinvestitionen",
            "öffentliche investitionen", "ausgabenerhöhung", "sondervermögen",
            "zuschüsse", "subventionen", "fiskalische expansion",
        ],
        "fiscal_contraction": [
            "haushaltskonsolidierung", "sparprogramm", "steuererhöhung", "schuldenbremse",
            "ausgabenkürzung", "defizitabbau", "haushaltsdisziplin", "kürzungsmaßnahmen",
        ],
        "monetary_easing": [
            "zinssenkung", "leitzinssenkung", "quantitative lockerung", "wertpapierkaufprogramm",
            "monetäre lockerung", "expansive geldpolitik", "liquiditätsbereitstellung",
        ],
        "monetary_tightening": [
            "zinserhöhung", "leitzinserhöhung", "geldpolitische straffung", "restriktive geldpolitik",
            "inflationsbekämpfung", "bilanzverkürzung",
        ],
        "uncertainty": [
            "unsicherheit", "ungewissheit", "volatilität", "risiko", "instabilität",
            "abwärtsrisiko", "schwankung",
        ],
    },

    # ── French (France / Trésor / BdF) ──
    "fr": {
        "fiscal_expansion": [
            "plan de relance", "baisse d'impôts", "réduction fiscale", "dépenses d'infrastructure",
            "investissement public", "hausse des dépenses", "crédits supplémentaires",
            "aides d'urgence", "subventions", "relance budgétaire",
        ],
        "fiscal_contraction": [
            "consolidation budgétaire", "rigueur budgétaire", "hausse d'impôts", "augmentation fiscale",
            "baisse des dépenses", "réduction du déficit", "ajustement budgétaire", "règle budgétaire",
        ],
        "monetary_easing": [
            "baisse des taux", "assouplissement quantitatif", "politique monétaire accommodante",
            "baisse du taux directeur", "injection de liquidités", "soutien monétaire",
        ],
        "monetary_tightening": [
            "hausse des taux", "relèvement des taux", "resserrement monétaire",
            "politique monétaire restrictive", "lutte contre l'inflation",
        ],
        "uncertainty": [
            "incertitude", "incertain", "volatilité", "risque", "instabilité",
            "turbulences", "risques baissiers",
        ],
    },

    # ── Italian (Italy / MEF / BdI) ──
    "it": {
        "fiscal_expansion": [
            "piano di stimolo", "taglio delle tasse", "sgravi fiscali", "investimenti in infrastrutture",
            "investimento pubblico", "aumento della spesa", "scostamento di bilancio",
            "sussidi", "espansione fiscale", "aiuti economici",
        ],
        "fiscal_contraction": [
            "consolidamento fiscale", "austerità", "aumento delle tasse", "riforma fiscale",
            "tagli alla spesa", "riduzione del disavanzo", "manovra correttiva", "patto di stabilità",
        ],
        "monetary_easing": [
            "taglio dei tassi", "allentamento quantitativo", "politica monetaria espansiva",
            "riduzione del tasso", "immissione di liquidità",
        ],
        "monetary_tightening": [
            "rialzo dei tassi", "aumento dei tassi", "stretta monetaria", "politica monetaria restrittiva",
            "contrasto all'inflazione",
        ],
        "uncertainty": [
            "incertezza", "incerto", "volatilità", "rischio", "instabilità", "turbolenza",
        ],
    },

    # ── Japanese (BoJ / MoF) ──
    "ja": {
        "fiscal_expansion": [
            "財政出動", "減税", "インフラ投資", "公共投資", "補正予算",
            "給付金", "補助金", "経済対策", "財政刺激",
        ],
        "fiscal_contraction": [
            "財政健全化", "増税", "歳出削減", "財政規律", "赤字削減", "消費税率引き上げ",
        ],
        "monetary_easing": [
            "利下げ", "金融緩和", "マイナス金利", "イールドカーブ・コントロール",
            "資産買い入れ", "流動性供給", "緩和姿勢",
        ],
        "monetary_tightening": [
            "利上げ", "金融引き締め", "金融政策の正常化", "テーパリング", "タカ派",
        ],
        "uncertainty": [
            "不確実性", "不透明感", "リスク", "変動性", "下振れリスク",
        ],
    },

    # ── Chinese (PBoC / MoF) ──
    "zh": {
        "fiscal_expansion": [
            "积极财政政策", "减税降费", "专项债", "基础设施投资", "公共投资",
            "财政补贴", "超长期特别国债", "扩大财政支出",
        ],
        "fiscal_contraction": [
            "财政紧缩", "加税", "压缩一般性支出", "防范化解地方债务风险", "严控预算",
        ],
        "monetary_easing": [
            "降息", "降准", "宽松货币政策", "流动性充裕", "再贷款", "引导利率下行",
        ],
        "monetary_tightening": [
            "加息", "收紧流动性", "紧缩货币政策", "提高准备金率", "防范金融风险",
        ],
        "uncertainty": [
            "不确定性", "风险", "波动性", "不稳定", "下行压力",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Multilingual Scoring Functions
# ─────────────────────────────────────────────────────────────────────────────

def _build_regex(terms: list[str]) -> re.Pattern:
    escaped = [re.escape(t) for t in terms]
    # For Latin scripts, match word boundaries; for CJK characters, simple match
    pat = "|".join(escaped)
    return re.compile(pat, flags=re.IGNORECASE)


@dataclass
class MultilingualScoreResult:
    language: str
    token_count: int
    net_sentiment: float
    expansion_intensity: float
    contraction_intensity: float
    uncertainty_intensity: float
    matched_expansion_terms: list[str] = field(default_factory=list)
    matched_contraction_terms: list[str] = field(default_factory=list)
    matched_uncertainty_terms: list[str] = field(default_factory=list)


def score_multilingual(
    text: str,
    language: str = "en",
    *,
    domain: str = "all",
) -> MultilingualScoreResult:
    """Score text sentiment and policy stance using language-specific macro lexicons.

    Parameters
    ----------
    text : string text of the document or excerpt.
    language : ISO-639-1 code (``"en"``, ``"es"``, ``"pt"``, ``"de"``, ``"fr"``, ``"it"``, ``"ja"``, ``"zh"``).
        Falls back to English if language is not registered.
    domain : ``"fiscal"``, ``"monetary"``, or ``"all"``.

    Returns
    -------
    MultilingualScoreResult
        Net stance polarity, intensities per 1000 tokens, and matched keyword terms.
    """
    if not text:
        return MultilingualScoreResult(
            language=language, token_count=0, net_sentiment=0.0,
            expansion_intensity=0.0, contraction_intensity=0.0, uncertainty_intensity=0.0
        )

    lang = language.lower() if language.lower() in MULTILINGUAL_LEXICONS else "en"
    lex = MULTILINGUAL_LEXICONS[lang]

    tokens = text.split()
    n_tokens = max(1, len(tokens))

    exp_terms: list[str] = []
    con_terms: list[str] = []

    if domain in ("fiscal", "all"):
        exp_terms.extend(lex.get("fiscal_expansion", []))
        con_terms.extend(lex.get("fiscal_contraction", []))
    if domain in ("monetary", "all"):
        exp_terms.extend(lex.get("monetary_easing", []))
        con_terms.extend(lex.get("monetary_tightening", []))

    unc_terms = lex.get("uncertainty", [])

    # One lowercase copy of the document, not one per lexicon term. With ~62
    # terms across the three lists that was 62 full copies of the text per
    # call, which dominates the scoring loop on a long transcript.
    text_lower = text.lower()

    matched_exp = [t for t in exp_terms if t.lower() in text_lower]
    matched_con = [t for t in con_terms if t.lower() in text_lower]
    matched_unc = [t for t in unc_terms if t.lower() in text_lower]

    n_exp = len(matched_exp)
    n_con = len(matched_con)
    n_unc = len(matched_unc)

    total_stance = n_exp + n_con
    if total_stance > 0:
        net = (n_exp - n_con) / total_stance
    else:
        net = 0.0

    return MultilingualScoreResult(
        language=lang,
        token_count=n_tokens,
        net_sentiment=float(net),
        expansion_intensity=float(n_exp * 1000.0 / n_tokens),
        contraction_intensity=float(n_con * 1000.0 / n_tokens),
        uncertainty_intensity=float(n_unc * 1000.0 / n_tokens),
        matched_expansion_terms=matched_exp,
        matched_contraction_terms=matched_con,
        matched_uncertainty_terms=matched_unc,
    )


def infer_policy_stance(
    text: str,
    language: str = "en",
    *,
    domain: str = "all",
) -> tuple[int, float, dict[str, Any]]:
    """Infer discrete policy sign (+1, -1, 0) and confidence score for a document.

    Returns
    -------
    tuple of (sign: int, confidence: float in [0, 1], details: dict)
    """
    res = score_multilingual(text, language=language, domain=domain)
    n_exp = len(res.matched_expansion_terms)
    n_con = len(res.matched_contraction_terms)

    if n_exp > n_con:
        sign = 1
        confidence = min(0.95, 0.5 + (n_exp - n_con) * 0.15)
    elif n_con > n_exp:
        sign = -1
        confidence = min(0.95, 0.5 + (n_con - n_exp) * 0.15)
    else:
        sign = 0
        confidence = 0.0

    details = {
        "language": res.language,
        "net_sentiment": res.net_sentiment,
        "matched_expansion": res.matched_expansion_terms,
        "matched_contraction": res.matched_contraction_terms,
        "uncertainty_intensity": res.uncertainty_intensity,
    }
    return sign, float(confidence), details


__all__ = [
    "MULTILINGUAL_LEXICONS",
    "MultilingualScoreResult",
    "score_multilingual",
    "infer_policy_stance",
]
