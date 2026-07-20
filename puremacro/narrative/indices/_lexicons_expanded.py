"""LLM-suggested + author-reviewed lexicon expansions (Sprint alpha.2).

Each ``_EXPANDED`` variant is a superset of the corresponding base
lexicon in :mod:`._lexicons`, capped at ``+20%`` terms. Used exclusively
by :class:`puremacro.narrative.validation.NarrativeStabilityReport` to
measure the sensitivity of headline correlations to lexicon-curation
choices.

Reviewer: Jorge Alonso Ortiz, 2026-05-12.
Review log: ``puremacro/docs/lexicon_review.md``.

The expansions are driven by a small per-family English seed list and
translated where applicable into each of the 8 target languages. They
are *not* meant to be a production-grade lexicon; their sole purpose is
to perturb the headline indices and measure correlation stability.
"""
from __future__ import annotations

# Import every base lexicon explicitly so static analysis can find the
# dependencies. We also import LEXICONS to mirror its construction.
from ._lexicons import (
    LEXICONS,
    _LABOR_DOMAIN_EN, _LABOR_DOMAIN_ES, _LABOR_DOMAIN_PT, _LABOR_DOMAIN_DE,
    _LABOR_DOMAIN_FR, _LABOR_DOMAIN_IT, _LABOR_DOMAIN_JA, _LABOR_DOMAIN_ZH,
    _UNCERTAINTY_TONE_EN, _UNCERTAINTY_TONE_ES, _UNCERTAINTY_TONE_PT,
    _UNCERTAINTY_TONE_DE, _UNCERTAINTY_TONE_FR, _UNCERTAINTY_TONE_IT,
    _UNCERTAINTY_TONE_JA, _UNCERTAINTY_TONE_ZH,
    _TECH_DOMAIN_EN, _TECH_DOMAIN_ES, _TECH_DOMAIN_PT, _TECH_DOMAIN_DE,
    _TECH_DOMAIN_FR, _TECH_DOMAIN_IT, _TECH_DOMAIN_JA, _TECH_DOMAIN_ZH,
    _WAR_DOMAIN_EN, _WAR_DOMAIN_ES, _WAR_DOMAIN_PT, _WAR_DOMAIN_DE,
    _WAR_DOMAIN_FR, _WAR_DOMAIN_IT, _WAR_DOMAIN_JA, _WAR_DOMAIN_ZH,
    _TECH_LABOR_UPSIDE_EN, _TECH_LABOR_UPSIDE_ES, _TECH_LABOR_UPSIDE_PT,
    _TECH_LABOR_UPSIDE_DE, _TECH_LABOR_UPSIDE_FR, _TECH_LABOR_UPSIDE_IT,
    _TECH_LABOR_UPSIDE_JA, _TECH_LABOR_UPSIDE_ZH,
    _TECH_LABOR_DOWNSIDE_EN, _TECH_LABOR_DOWNSIDE_ES, _TECH_LABOR_DOWNSIDE_PT,
    _TECH_LABOR_DOWNSIDE_DE, _TECH_LABOR_DOWNSIDE_FR, _TECH_LABOR_DOWNSIDE_IT,
    _TECH_LABOR_DOWNSIDE_JA, _TECH_LABOR_DOWNSIDE_ZH,
    _LUI_PHRASES_EN, _LUI_PHRASES_ES, _LUI_PHRASES_PT, _LUI_PHRASES_DE,
    _LUI_PHRASES_FR, _LUI_PHRASES_IT, _LUI_PHRASES_JA, _LUI_PHRASES_ZH,
)


# ---------------------------------------------------------------------------
# _LABOR_DOMAIN_*_EXPANDED — adds workforce / staffing / payrolls / labour
# supply synonyms. Cap is floor(0.20 * |base|).
# ---------------------------------------------------------------------------

_LABOR_DOMAIN_EN_EXPANDED = _LABOR_DOMAIN_EN | frozenset({
    "staffing", "headcount", "labor pool", "labour pool",
    "hiring trend", "employment trend", "labor market conditions",
    "talent pool", "manpower", "workforce participation",
})

_LABOR_DOMAIN_ES_EXPANDED = _LABOR_DOMAIN_ES | frozenset({
    "plantilla", "personal", "ocupados",
    "tendencia de empleo", "condiciones del mercado laboral",
    "mano de obra", "reserva laboral",
})

_LABOR_DOMAIN_PT_EXPANDED = _LABOR_DOMAIN_PT | frozenset({
    "quadro de pessoal", "pessoal", "ocupados",
    "tendência do emprego", "mão de obra",
    "condições do mercado de trabalho",
})

_LABOR_DOMAIN_DE_EXPANDED = _LABOR_DOMAIN_DE | frozenset({
    "personalbestand", "personal", "belegschaft",
    "beschäftigungstrend", "arbeitskräfte",
    "arbeitsmarktbedingungen",
})

_LABOR_DOMAIN_FR_EXPANDED = _LABOR_DOMAIN_FR | frozenset({
    "effectifs", "personnel", "actifs",
    "tendance de l'emploi", "conditions du marché du travail",
    "vivier de talents", "ressources humaines",
})

_LABOR_DOMAIN_IT_EXPANDED = _LABOR_DOMAIN_IT | frozenset({
    "organico", "personale", "addetti",
    "tendenza occupazionale", "manodopera",
    "condizioni del mercato del lavoro",
})

_LABOR_DOMAIN_JA_EXPANDED = _LABOR_DOMAIN_JA | frozenset({
    "人員", "従業員数", "雇用情勢",
    "労働力人口", "労働市場環境",
    "人材プール",
})

_LABOR_DOMAIN_ZH_EXPANDED = _LABOR_DOMAIN_ZH | frozenset({
    "人员", "员工人数", "就业形势",
    "劳动力人口", "劳动力市场状况",
    "人才储备",
})


# ---------------------------------------------------------------------------
# _UNCERTAINTY_TONE_*_EXPANDED — adds subdued/muted/cloudy/volatile/fragile
# synonyms. Cap is floor(0.20 * |base|).
# ---------------------------------------------------------------------------

_UNCERTAINTY_TONE_EN_EXPANDED = _UNCERTAINTY_TONE_EN | frozenset({
    "muted", "cloudy", "wary", "guarded",
    "hesitant", "anxious", "tepid",
    "lackluster", "sluggish",
})

_UNCERTAINTY_TONE_ES_EXPANDED = _UNCERTAINTY_TONE_ES | frozenset({
    "apagado", "nublado", "cauteloso",
    "vacilante", "ansioso", "tibio",
})

_UNCERTAINTY_TONE_PT_EXPANDED = _UNCERTAINTY_TONE_PT | frozenset({
    "abafado", "nublado", "cauteloso",
    "hesitante", "ansioso", "morno",
})

_UNCERTAINTY_TONE_DE_EXPANDED = _UNCERTAINTY_TONE_DE | frozenset({
    "verhalten", "trüb", "vorsichtig",
    "zögerlich", "besorgt", "lustlos",
})

_UNCERTAINTY_TONE_FR_EXPANDED = _UNCERTAINTY_TONE_FR | frozenset({
    "atone", "nuageux", "prudent",
    "hésitant", "inquiet",
})

_UNCERTAINTY_TONE_IT_EXPANDED = _UNCERTAINTY_TONE_IT | frozenset({
    "sommesso", "nuvoloso", "cauto",
    "esitante", "ansioso", "tiepido",
})

_UNCERTAINTY_TONE_JA_EXPANDED = _UNCERTAINTY_TONE_JA | frozenset({
    "慎重", "曇り", "不透明",
    "ためらい",
})

_UNCERTAINTY_TONE_ZH_EXPANDED = _UNCERTAINTY_TONE_ZH | frozenset({
    "谨慎", "阴云", "不明朗",
    "犹豫",
})


# ---------------------------------------------------------------------------
# _TECH_DOMAIN_*_EXPANDED — adds machine learning / generative AI / LLM /
# digitisation / automation-wave synonyms. Cap floor(0.20 * |base|).
# ---------------------------------------------------------------------------

_TECH_DOMAIN_EN_EXPANDED = _TECH_DOMAIN_EN | frozenset({
    "large language model", "large language models", "llm", "llms",
    "foundation model", "foundation models",
    "generative model", "generative models",
    "ai assistant", "ai assistants",
    "ai-driven", "ai driven",
    "digitisation", "digitalisation",
    "automation wave", "automation push", "tech-enabled",
})

_TECH_DOMAIN_ES_EXPANDED = _TECH_DOMAIN_ES | frozenset({
    "modelo de lenguaje", "modelos de lenguaje",
    "modelo generativo", "asistente de ia",
    "digitalización masiva", "ola de automatización",
    "impulso digital",
})

_TECH_DOMAIN_PT_EXPANDED = _TECH_DOMAIN_PT | frozenset({
    "modelo de linguagem", "modelos de linguagem",
    "modelo generativo", "assistente de ia",
    "digitalização acelerada", "onda de automação",
    "impulso digital",
})

_TECH_DOMAIN_DE_EXPANDED = _TECH_DOMAIN_DE | frozenset({
    "sprachmodell", "großes sprachmodell",
    "generatives modell", "ki-assistent",
    "digitalisierungsschub", "automatisierungswelle",
    "ki-getrieben",
})

_TECH_DOMAIN_FR_EXPANDED = _TECH_DOMAIN_FR | frozenset({
    "modèle de langage", "grand modèle de langage",
    "modèle génératif", "assistant ia",
    "numérisation accélérée", "vague d'automatisation",
    "poussée numérique",
})

_TECH_DOMAIN_IT_EXPANDED = _TECH_DOMAIN_IT | frozenset({
    "modello linguistico", "grande modello linguistico",
    "modello generativo", "assistente ia",
    "digitalizzazione accelerata", "ondata di automazione",
    "spinta digitale",
})

_TECH_DOMAIN_JA_EXPANDED = _TECH_DOMAIN_JA | frozenset({
    "大規模言語モデル", "言語モデル",
    "生成モデル", "AIアシスタント",
    "デジタル化の加速", "自動化の波",
    "AI駆動",
})

_TECH_DOMAIN_ZH_EXPANDED = _TECH_DOMAIN_ZH | frozenset({
    "大语言模型", "语言模型",
    "生成式模型", "AI助手",
    "数字化加速", "自动化浪潮",
    "AI驱动",
})


# ---------------------------------------------------------------------------
# _TECH_LABOR_UPSIDE_*_EXPANDED — adds augmentation / skill upgrade /
# productivity boost / new occupations / complementarity synonyms.
# ---------------------------------------------------------------------------

_TECH_LABOR_UPSIDE_EN_EXPANDED = _TECH_LABOR_UPSIDE_EN | frozenset({
    "skill upgrade", "skill upgrading", "reskilling boost",
    "productivity boost", "human-ai complementarity",
    "new occupations",
})

_TECH_LABOR_UPSIDE_ES_EXPANDED = _TECH_LABOR_UPSIDE_ES | frozenset({
    "mejora de habilidades", "impulso de productividad",
    "nuevas ocupaciones",
})

_TECH_LABOR_UPSIDE_PT_EXPANDED = _TECH_LABOR_UPSIDE_PT | frozenset({
    "atualização de habilidades", "impulso de produtividade",
    "novas ocupações",
})

_TECH_LABOR_UPSIDE_DE_EXPANDED = _TECH_LABOR_UPSIDE_DE | frozenset({
    "qualifikationsausbau", "produktivitätsschub",
    "neue berufe",
})

_TECH_LABOR_UPSIDE_FR_EXPANDED = _TECH_LABOR_UPSIDE_FR | frozenset({
    "montée en compétences", "gains de productivité",
    "nouveaux métiers",
})

_TECH_LABOR_UPSIDE_IT_EXPANDED = _TECH_LABOR_UPSIDE_IT | frozenset({
    "miglioramento delle competenze", "spinta alla produttività",
    "nuove professioni",
})

_TECH_LABOR_UPSIDE_JA_EXPANDED = _TECH_LABOR_UPSIDE_JA | frozenset({
    "スキル向上", "生産性向上",
})

_TECH_LABOR_UPSIDE_ZH_EXPANDED = _TECH_LABOR_UPSIDE_ZH | frozenset({
    "技能提升", "生产率提升",
})


# ---------------------------------------------------------------------------
# _TECH_LABOR_DOWNSIDE_*_EXPANDED — adds displacement / replacement /
# obsolete / labour-saving / redundant synonyms.
# ---------------------------------------------------------------------------

_TECH_LABOR_DOWNSIDE_EN_EXPANDED = _TECH_LABOR_DOWNSIDE_EN | frozenset({
    "labour-saving", "labor-saving",
    "obsolete skill", "obsolete skills",
    "redundant role", "redundant roles",
    "ai replacement",
})

_TECH_LABOR_DOWNSIDE_ES_EXPANDED = _TECH_LABOR_DOWNSIDE_ES | frozenset({
    "ahorro de mano de obra", "habilidades obsoletas",
    "puestos redundantes", "reemplazo por ia",
})

_TECH_LABOR_DOWNSIDE_PT_EXPANDED = _TECH_LABOR_DOWNSIDE_PT | frozenset({
    "economia de mão de obra", "habilidades obsoletas",
    "cargos redundantes", "substituição por ia",
})

_TECH_LABOR_DOWNSIDE_DE_EXPANDED = _TECH_LABOR_DOWNSIDE_DE | frozenset({
    "arbeitssparend", "veraltete fähigkeiten",
    "überflüssige stellen", "ki-ersatz",
})

_TECH_LABOR_DOWNSIDE_FR_EXPANDED = _TECH_LABOR_DOWNSIDE_FR | frozenset({
    "économie de main-d'œuvre", "compétences obsolètes",
    "postes redondants", "remplacement par l'ia",
})

_TECH_LABOR_DOWNSIDE_IT_EXPANDED = _TECH_LABOR_DOWNSIDE_IT | frozenset({
    "risparmio di manodopera", "competenze obsolete",
    "ruoli ridondanti", "sostituzione tramite ia",
})

_TECH_LABOR_DOWNSIDE_JA_EXPANDED = _TECH_LABOR_DOWNSIDE_JA | frozenset({
    "省力化", "陳腐化したスキル",
    "AI代替",
})

_TECH_LABOR_DOWNSIDE_ZH_EXPANDED = _TECH_LABOR_DOWNSIDE_ZH | frozenset({
    "节省劳动", "技能过时",
    "AI替代", "岗位冗余",
})


# ---------------------------------------------------------------------------
# _WAR_DOMAIN_*_EXPANDED — adds conflict / hostilities / mobilisation /
# refugee / sanctions synonyms.
# ---------------------------------------------------------------------------

_WAR_DOMAIN_EN_EXPANDED = _WAR_DOMAIN_EN | frozenset({
    "hostilities", "armed conflict",
    "refugee flows", "refugee crisis",
    "war economy", "wartime economy",
    "sanctions regime", "trade sanctions",
    "geopolitical flashpoint", "military mobilisation",
})

_WAR_DOMAIN_ES_EXPANDED = _WAR_DOMAIN_ES | frozenset({
    "hostilidades", "conflicto armado",
    "flujos de refugiados", "crisis de refugiados",
    "economía de guerra", "régimen de sanciones",
    "movilización militar", "punto crítico geopolítico",
})

_WAR_DOMAIN_PT_EXPANDED = _WAR_DOMAIN_PT | frozenset({
    "hostilidades", "conflito armado",
    "fluxos de refugiados", "crise de refugiados",
    "economia de guerra", "regime de sanções",
    "mobilização militar", "ponto crítico geopolítico",
})

_WAR_DOMAIN_DE_EXPANDED = _WAR_DOMAIN_DE | frozenset({
    "feindseligkeiten", "bewaffneter konflikt",
    "flüchtlingsströme", "flüchtlingskrise",
    "kriegswirtschaft", "sanktionsregime",
    "militärische mobilmachung", "geopolitischer brennpunkt",
})

_WAR_DOMAIN_FR_EXPANDED = _WAR_DOMAIN_FR | frozenset({
    "hostilités", "conflit armé",
    "flux de réfugiés", "crise des réfugiés",
    "économie de guerre", "régime de sanctions",
    "mobilisation militaire", "point chaud géopolitique",
})

_WAR_DOMAIN_IT_EXPANDED = _WAR_DOMAIN_IT | frozenset({
    "ostilità", "conflitto armato",
    "flussi di rifugiati", "crisi dei rifugiati",
    "economia di guerra", "regime sanzionatorio",
    "mobilitazione militare", "focolaio geopolitico",
})

_WAR_DOMAIN_JA_EXPANDED = _WAR_DOMAIN_JA | frozenset({
    "敵対行為", "武力衝突",
    "難民の流入", "難民危機",
    "戦時経済", "制裁措置",
    "軍事動員",
})

_WAR_DOMAIN_ZH_EXPANDED = _WAR_DOMAIN_ZH | frozenset({
    "敌对行动", "武装冲突",
    "难民潮", "难民危机",
    "战时经济", "制裁机制",
    "军事动员",
})


# ---------------------------------------------------------------------------
# Phrase lexicons (LUI) — passed through unchanged. The phrase axis is
# multi-word and not a target of the alpha.2 perturbation; we expose it
# verbatim so that LEXICONS_EXPANDED["lui"][lang] still carries the
# "phrases" component identical to LEXICONS["lui"][lang]["phrases"].
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Build LEXICONS_EXPANDED with the same shape as LEXICONS for the 5
# in-scope indices: lui, ltui, ltui_up, ltui_down, lwui. Each cell maps
# language -> {component_name: frozenset}.
# ---------------------------------------------------------------------------

LEXICONS_EXPANDED: dict[str, dict[str, dict[str, frozenset[str]]]] = {
    "lui": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN_EXPANDED,
            "phrases": _LUI_PHRASES_EN,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES_EXPANDED,
            "phrases": _LUI_PHRASES_ES,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT_EXPANDED,
            "phrases": _LUI_PHRASES_PT,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE_EXPANDED,
            "phrases": _LUI_PHRASES_DE,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR_EXPANDED,
            "phrases": _LUI_PHRASES_FR,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT_EXPANDED,
            "phrases": _LUI_PHRASES_IT,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA_EXPANDED,
            "phrases": _LUI_PHRASES_JA,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH_EXPANDED,
            "phrases": _LUI_PHRASES_ZH,
        },
    },
    "ltui": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN_EXPANDED,
            "tech_domain": _TECH_DOMAIN_EN_EXPANDED,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES_EXPANDED,
            "tech_domain": _TECH_DOMAIN_ES_EXPANDED,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT_EXPANDED,
            "tech_domain": _TECH_DOMAIN_PT_EXPANDED,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE_EXPANDED,
            "tech_domain": _TECH_DOMAIN_DE_EXPANDED,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR_EXPANDED,
            "tech_domain": _TECH_DOMAIN_FR_EXPANDED,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT_EXPANDED,
            "tech_domain": _TECH_DOMAIN_IT_EXPANDED,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA_EXPANDED,
            "tech_domain": _TECH_DOMAIN_JA_EXPANDED,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH_EXPANDED,
            "tech_domain": _TECH_DOMAIN_ZH_EXPANDED,
        },
    },
    "ltui_up": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN_EXPANDED,
            "tech_domain": _TECH_DOMAIN_EN_EXPANDED,
            "polarity": _TECH_LABOR_UPSIDE_EN_EXPANDED,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES_EXPANDED,
            "tech_domain": _TECH_DOMAIN_ES_EXPANDED,
            "polarity": _TECH_LABOR_UPSIDE_ES_EXPANDED,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT_EXPANDED,
            "tech_domain": _TECH_DOMAIN_PT_EXPANDED,
            "polarity": _TECH_LABOR_UPSIDE_PT_EXPANDED,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE_EXPANDED,
            "tech_domain": _TECH_DOMAIN_DE_EXPANDED,
            "polarity": _TECH_LABOR_UPSIDE_DE_EXPANDED,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR_EXPANDED,
            "tech_domain": _TECH_DOMAIN_FR_EXPANDED,
            "polarity": _TECH_LABOR_UPSIDE_FR_EXPANDED,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT_EXPANDED,
            "tech_domain": _TECH_DOMAIN_IT_EXPANDED,
            "polarity": _TECH_LABOR_UPSIDE_IT_EXPANDED,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA_EXPANDED,
            "tech_domain": _TECH_DOMAIN_JA_EXPANDED,
            "polarity": _TECH_LABOR_UPSIDE_JA_EXPANDED,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH_EXPANDED,
            "tech_domain": _TECH_DOMAIN_ZH_EXPANDED,
            "polarity": _TECH_LABOR_UPSIDE_ZH_EXPANDED,
        },
    },
    "ltui_down": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN_EXPANDED,
            "tech_domain": _TECH_DOMAIN_EN_EXPANDED,
            "polarity": _TECH_LABOR_DOWNSIDE_EN_EXPANDED,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES_EXPANDED,
            "tech_domain": _TECH_DOMAIN_ES_EXPANDED,
            "polarity": _TECH_LABOR_DOWNSIDE_ES_EXPANDED,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT_EXPANDED,
            "tech_domain": _TECH_DOMAIN_PT_EXPANDED,
            "polarity": _TECH_LABOR_DOWNSIDE_PT_EXPANDED,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE_EXPANDED,
            "tech_domain": _TECH_DOMAIN_DE_EXPANDED,
            "polarity": _TECH_LABOR_DOWNSIDE_DE_EXPANDED,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR_EXPANDED,
            "tech_domain": _TECH_DOMAIN_FR_EXPANDED,
            "polarity": _TECH_LABOR_DOWNSIDE_FR_EXPANDED,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT_EXPANDED,
            "tech_domain": _TECH_DOMAIN_IT_EXPANDED,
            "polarity": _TECH_LABOR_DOWNSIDE_IT_EXPANDED,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA_EXPANDED,
            "tech_domain": _TECH_DOMAIN_JA_EXPANDED,
            "polarity": _TECH_LABOR_DOWNSIDE_JA_EXPANDED,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH_EXPANDED,
            "tech_domain": _TECH_DOMAIN_ZH_EXPANDED,
            "polarity": _TECH_LABOR_DOWNSIDE_ZH_EXPANDED,
        },
    },
    "lwui": {
        "en": {
            "labor_domain": _LABOR_DOMAIN_EN_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_EN_EXPANDED,
            "war_domain": _WAR_DOMAIN_EN_EXPANDED,
        },
        "es": {
            "labor_domain": _LABOR_DOMAIN_ES_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_ES_EXPANDED,
            "war_domain": _WAR_DOMAIN_ES_EXPANDED,
        },
        "pt": {
            "labor_domain": _LABOR_DOMAIN_PT_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_PT_EXPANDED,
            "war_domain": _WAR_DOMAIN_PT_EXPANDED,
        },
        "de": {
            "labor_domain": _LABOR_DOMAIN_DE_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_DE_EXPANDED,
            "war_domain": _WAR_DOMAIN_DE_EXPANDED,
        },
        "fr": {
            "labor_domain": _LABOR_DOMAIN_FR_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_FR_EXPANDED,
            "war_domain": _WAR_DOMAIN_FR_EXPANDED,
        },
        "it": {
            "labor_domain": _LABOR_DOMAIN_IT_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_IT_EXPANDED,
            "war_domain": _WAR_DOMAIN_IT_EXPANDED,
        },
        "ja": {
            "labor_domain": _LABOR_DOMAIN_JA_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_JA_EXPANDED,
            "war_domain": _WAR_DOMAIN_JA_EXPANDED,
        },
        "zh": {
            "labor_domain": _LABOR_DOMAIN_ZH_EXPANDED,
            "uncertainty_tone": _UNCERTAINTY_TONE_ZH_EXPANDED,
            "war_domain": _WAR_DOMAIN_ZH_EXPANDED,
        },
    },
}


__all__ = ["LEXICONS_EXPANDED"]
